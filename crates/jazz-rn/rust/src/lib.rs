// jazz-rn (Rust) — UniFFI surface for React Native.
//
// Note: This crate intentionally uses UniFFI proc-macros (no UDL). The RN bindings
// generator runs UniFFI in "library mode", reading this crate's metadata.
uniffi::setup_scaffolding!();

use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::pin::Pin;
#[cfg(test)]
use std::sync::atomic::AtomicU64;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::task::{Context, Poll, Waker};
use std::time::Duration;
#[cfg(test)]
use std::time::Instant;

use futures::future::FutureExt;
use serde::Deserialize;

use jazz_tools::binding_support::{
    default_read_durability_options as default_binding_read_durability_options,
    parse_batch_id_input, parse_batch_mode_input, parse_durability_tier as parse_binding_tier,
    parse_external_object_id, parse_query_input, parse_read_durability_options,
    parse_session_input, parse_write_context_input, serialize_mutation_error_event,
};
use jazz_tools::object::ObjectId;
use jazz_tools::query_manager::query::Query;
use jazz_tools::query_manager::session::{Session, WriteContext};
use jazz_tools::query_manager::types::{Schema, SchemaHash, Value};
use jazz_tools::runtime_core::{
    MutationErrorCallback as CoreMutationErrorCallback, ReadDurabilityOptions, RuntimeCore,
    Scheduler, SubscriptionDelta, SubscriptionHandle,
};
use jazz_tools::schema_manager::{rehydrate_schema_manager_from_catalogue, AppId, SchemaManager};
use jazz_tools::storage::{SqliteStorage, Storage};
use jazz_tools::sync_manager::{DurabilityTier, QueryPropagation, SyncManager};

mod engine_log;

// ============================================================================
// Errors
// ============================================================================

#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum JazzRnError {
    #[error("invalid json: {message}")]
    InvalidJson { message: String },

    #[error("invalid uuid: {message}")]
    InvalidUuid { message: String },

    #[error("invalid persistence tier: {message}")]
    InvalidTier { message: String },

    #[error("schema error: {message}")]
    Schema { message: String },

    #[error("runtime error: {message}")]
    Runtime { message: String },

    #[error("internal error: {message}")]
    Internal { message: String },
}

fn json_err(e: serde_json::Error) -> JazzRnError {
    JazzRnError::InvalidJson {
        message: e.to_string(),
    }
}

fn runtime_err<E: std::fmt::Display>(e: E) -> JazzRnError {
    JazzRnError::Runtime {
        message: e.to_string(),
    }
}

fn panic_payload_to_string(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<String>() {
        return message.clone();
    }
    if let Some(message) = payload.downcast_ref::<&str>() {
        return (*message).to_string();
    }
    "non-string panic payload".to_string()
}

fn panic_to_jazz_error(
    context: &'static str,
    payload: Box<dyn std::any::Any + Send>,
) -> JazzRnError {
    let panic_message = panic_payload_to_string(payload);
    let backtrace = std::backtrace::Backtrace::force_capture();
    JazzRnError::Internal {
        message: format!("panic in {context}: {panic_message}\n{backtrace}"),
    }
}

fn with_panic_boundary<T, F>(context: &'static str, f: F) -> Result<T, JazzRnError>
where
    F: FnOnce() -> Result<T, JazzRnError>,
{
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
        .unwrap_or_else(|payload| Err(panic_to_jazz_error(context, payload)))
}

async fn with_async_panic_boundary<T, F, Fut>(context: &'static str, f: F) -> Result<T, JazzRnError>
where
    F: FnOnce() -> Fut,
    Fut: Future<Output = Result<T, JazzRnError>>,
{
    std::panic::AssertUnwindSafe(f())
        .catch_unwind()
        .await
        .unwrap_or_else(|payload| Err(panic_to_jazz_error(context, payload)))
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", content = "value")]
enum FfiJsonValue {
    Integer(i32),
    BigInt(i64),
    Double(f64),
    Boolean(bool),
    Text(String),
    Timestamp(u64),
    Uuid(ObjectId),
    Bytea(String),
    /// Index into the sidecar `blobs` argument of the `*_with_blobs` methods. Exists so
    /// multi-megabyte payloads cross the FFI as raw bytes instead of hex-in-JSON — the
    /// hex round-trip measured ~0.5 s of JS-thread time per MiB on device.
    BlobRef(usize),
    Array(Vec<FfiJsonValue>),
    Row(FfiJsonRow),
    Null,
}

#[derive(Debug, Clone, Deserialize)]
struct FfiJsonRow {
    #[serde(default)]
    id: Option<ObjectId>,
    values: Vec<FfiJsonValue>,
}

fn ffi_json_err(message: impl Into<String>) -> JazzRnError {
    JazzRnError::InvalidJson {
        message: message.into(),
    }
}

fn decode_ffi_json_value(value: FfiJsonValue, blobs: &[Vec<u8>]) -> Result<Value, JazzRnError> {
    match value {
        FfiJsonValue::Integer(value) => Ok(Value::Integer(value)),
        FfiJsonValue::BigInt(value) => Ok(Value::BigInt(value)),
        FfiJsonValue::Double(value) => Ok(Value::Double(value)),
        FfiJsonValue::Boolean(value) => Ok(Value::Boolean(value)),
        FfiJsonValue::Text(value) => Ok(Value::Text(value)),
        FfiJsonValue::Timestamp(value) => Ok(Value::Timestamp(value)),
        FfiJsonValue::Uuid(value) => Ok(Value::Uuid(value)),
        FfiJsonValue::Bytea(value) => hex::decode(value)
            .map(Value::Bytea)
            .map_err(|error| ffi_json_err(format!("invalid Bytea hex payload: {error}"))),
        // Clone, not take: the originals stay untouched in `blobs` so the return path can
        // recognize them by content and echo a BlobRef instead of the bytes.
        FfiJsonValue::BlobRef(index) => {
            blobs.get(index).cloned().map(Value::Bytea).ok_or_else(|| {
                ffi_json_err(format!(
                    "BlobRef {index} out of range: {} blob(s) provided",
                    blobs.len()
                ))
            })
        }
        FfiJsonValue::Array(values) => values
            .into_iter()
            .map(|value| decode_ffi_json_value(value, blobs))
            .collect::<Result<Vec<_>, _>>()
            .map(Value::Array),
        FfiJsonValue::Row(row) => row
            .values
            .into_iter()
            .map(|value| decode_ffi_json_value(value, blobs))
            .collect::<Result<Vec<_>, _>>()
            .map(|values| Value::Row { id: row.id, values }),
        FfiJsonValue::Null => Ok(Value::Null),
    }
}

fn decode_ffi_json_record_with_blobs(
    values_json: &str,
    blobs: &[Vec<u8>],
) -> Result<HashMap<String, Value>, JazzRnError> {
    let values: HashMap<String, FfiJsonValue> =
        serde_json::from_str(values_json).map_err(json_err)?;
    values
        .into_iter()
        .map(|(key, value)| decode_ffi_json_value(value, blobs).map(|value| (key, value)))
        .collect()
}

fn decode_ffi_json_record(values_json: &str) -> Result<HashMap<String, Value>, JazzRnError> {
    decode_ffi_json_record_with_blobs(values_json, &[])
}

/// Serialize returned row values for a `*_with_blobs` call. Bytea payloads that are
/// byte-identical to an input blob come back as `BlobRef` so the megabyte the caller just
/// handed over is never re-encoded and re-parsed; any other Bytea (possible on restore,
/// which can resurrect columns absent from the input) is hex, which the adapter's decoder
/// understands. Non-Bytea values serialize exactly as the legacy methods do, via
/// `Value`'s own human-readable serde.
fn encode_return_value_with_blob_refs(value: &Value, blobs: &[Vec<u8>]) -> serde_json::Value {
    match value {
        Value::Bytea(bytes) => {
            let matched = blobs
                .iter()
                .position(|blob| blob.len() == bytes.len() && blob == bytes);
            match matched {
                Some(index) => serde_json::json!({ "type": "BlobRef", "value": index }),
                None => serde_json::json!({ "type": "Bytea", "value": hex::encode(bytes) }),
            }
        }
        Value::Array(values) => serde_json::json!({
            "type": "Array",
            "value": values
                .iter()
                .map(|value| encode_return_value_with_blob_refs(value, blobs))
                .collect::<Vec<_>>(),
        }),
        Value::Row { id, values } => {
            let mut row = serde_json::Map::new();
            if let Some(id) = id {
                row.insert("id".into(), serde_json::json!(id));
            }
            row.insert(
                "values".into(),
                serde_json::Value::Array(
                    values
                        .iter()
                        .map(|value| encode_return_value_with_blob_refs(value, blobs))
                        .collect(),
                ),
            );
            serde_json::json!({ "type": "Row", "value": row })
        }
        other => {
            serde_json::to_value(other).expect("scalar Value serialization to JSON cannot fail")
        }
    }
}

fn encode_return_values_with_blob_refs(
    values: &[Value],
    blobs: &[Vec<u8>],
) -> Vec<serde_json::Value> {
    values
        .iter()
        .map(|value| encode_return_value_with_blob_refs(value, blobs))
        .collect()
}

fn convert_insert_values(values_json: &str) -> Result<HashMap<String, Value>, JazzRnError> {
    decode_ffi_json_record(values_json)
}

fn convert_updates(values_json: &str) -> Result<Vec<(String, Value)>, JazzRnError> {
    let partial = decode_ffi_json_record(values_json)?;
    Ok(partial.into_iter().collect())
}

fn parse_query(query_json: &str) -> Result<Query, JazzRnError> {
    parse_query_input(query_json).map_err(|message| JazzRnError::InvalidJson { message })
}

fn parse_session(session_json: Option<String>) -> Result<Option<Session>, JazzRnError> {
    parse_session_input(session_json.as_deref())
        .map_err(|message| JazzRnError::InvalidJson { message })
}

fn parse_write_context(
    write_context_json: Option<String>,
) -> Result<Option<WriteContext>, JazzRnError> {
    parse_write_context_input(write_context_json.as_deref())
        .map_err(|message| JazzRnError::InvalidJson { message })
}

fn parse_tier(tier: &str) -> Result<DurabilityTier, JazzRnError> {
    parse_binding_tier(tier).map_err(|message| JazzRnError::InvalidTier { message })
}

fn default_read_durability_options(tier: Option<DurabilityTier>) -> ReadDurabilityOptions {
    default_binding_read_durability_options(tier)
}

fn parse_subscription_inputs(
    query_json: &str,
    session_json: Option<String>,
    tier: Option<String>,
) -> Result<(Query, Option<Session>, ReadDurabilityOptions), JazzRnError> {
    let query = parse_query(query_json)?;
    let session = parse_session(session_json)?;
    let tier = tier.as_deref().map(parse_tier).transpose()?;
    Ok((query, session, default_read_durability_options(tier)))
}

/// Encode a value for delivery to JS, moving every blob into `sidecar` and
/// leaving a `BlobRef` in its place.
///
/// The read-direction twin of [`encode_return_value_with_blob_refs`]. That one
/// answers a `*_with_blobs` call, so it can only reference blobs the caller
/// already holds and falls back to hex for anything else. Deliveries have no
/// such caller, so they collect their own sidecar as they go and never encode a
/// payload as text at all.
fn encode_value_into_sidecar(value: &Value, sidecar: &mut Vec<Vec<u8>>) -> serde_json::Value {
    match value {
        Value::Bytea(bytes) => {
            let index = sidecar.len();
            sidecar.push(bytes.clone());
            serde_json::json!({ "type": "BlobRef", "value": index })
        }
        Value::Array(values) => serde_json::json!({
            "type": "Array",
            "value": values
                .iter()
                .map(|value| encode_value_into_sidecar(value, sidecar))
                .collect::<Vec<_>>(),
        }),
        Value::Row { id, values } => {
            let mut row = serde_json::Map::new();
            if let Some(id) = id {
                row.insert("id".into(), serde_json::json!(id));
            }
            row.insert(
                "values".into(),
                serde_json::Value::Array(
                    values
                        .iter()
                        .map(|value| encode_value_into_sidecar(value, sidecar))
                        .collect(),
                ),
            );
            serde_json::json!({ "type": "Row", "value": row })
        }
        other => {
            serde_json::to_value(other).expect("scalar Value serialization to JSON cannot fail")
        }
    }
}

/// Re-encode a delta so its blobs travel beside the JSON rather than inside it.
///
/// Mirrors `subscription_delta_to_json` key for key — same `kind` numbering,
/// `updated` still carrying an optional row — and differs only in what a `Bytea`
/// becomes. A megabyte inlined as an array of Numbers is ~3.7 MB of text that
/// the receiver parses and then walks byte by byte; as a reference it is about
/// thirty characters.
fn subscription_delta_with_blob_sidecar(
    delta: &SubscriptionDelta,
) -> (serde_json::Value, Vec<Vec<u8>>) {
    let mut sidecar: Vec<Vec<u8>> = Vec::new();
    let descriptor = &delta.descriptor;
    let row_to_json = |row: &jazz_tools::query_manager::types::Row, sidecar: &mut Vec<Vec<u8>>| {
        let values = jazz_tools::row_format::decode_row(descriptor, &row.data).unwrap_or_default();
        serde_json::json!({
            "id": row.id.uuid().to_string(),
            "values": values
                .iter()
                .map(|value| encode_value_into_sidecar(value, sidecar))
                .collect::<Vec<_>>(),
        })
    };

    let mut changes: Vec<serde_json::Value> = Vec::new();
    for change in &delta.ordered_delta.removed {
        changes.push(serde_json::json!({
            "kind": 1,
            "id": change.id.uuid().to_string(),
            "index": change.index,
        }));
    }
    for change in &delta.ordered_delta.updated {
        let row = change
            .row
            .as_ref()
            .map(|row| row_to_json(row, &mut sidecar));
        changes.push(serde_json::json!({
            "kind": 2,
            "id": change.id.uuid().to_string(),
            "index": change.new_index,
            "row": row,
        }));
    }
    for change in &delta.ordered_delta.added {
        let row = row_to_json(&change.row, &mut sidecar);
        changes.push(serde_json::json!({
            "kind": 0,
            "id": change.id.uuid().to_string(),
            "index": change.index,
            "row": row,
        }));
    }

    (serde_json::Value::Array(changes), sidecar)
}

// ============================================================================
// What the engine tells JS
// ============================================================================
//
// The engine's scheduled tick runs on a thread of the binding's own, not on the JS
// thread: a tick that stores what a server delivered is hundreds of milliseconds of
// SQLite, and on the JS thread that is hundreds of milliseconds without a frame.
//
// A callback into JS from another thread is a blocking hop: the caller waits until
// the JS thread has run it. Made while holding the core lock, with the JS thread
// inside any call that wants that lock, it never returns. So nothing calls JS while
// the core lock is held, on any thread: what a tick has to tell JS is recorded, and
// told on the JS thread once the lock is released.
//
// Two things a tick does under the lock do reach the glue all the same: it wakes the
// future it resolves, and it drops the callback of a subscription it removes. Both
// are safe only because of how the glue makes them — the continuation of a future
// and the release of a callback are posted to the JS thread and not waited for
// (`invokeNonBlocking` in the generated `jazz_rn.cpp`; a callback's methods and its
// clone are the blocking ones). A glue that waited for either would hang here.

/// One thing the engine has to tell JS.
enum JsEvent {
    Delta {
        handle: u64,
        callback: Arc<dyn SubscriptionCallback>,
        json: String,
        blobs: Vec<Vec<u8>>,
    },
    AuthFailure {
        callback: Arc<dyn AuthFailureCallback>,
        reason: String,
    },
}

#[derive(Default)]
struct OutboxState {
    /// Recorded under the core lock by whoever holds it now.
    staging: Vec<JsEvent>,
    /// What whole holds of the core lock recorded, in the order they held it.
    ready: VecDeque<JsEvent>,
    /// Somebody holds the core lock.
    section_open: bool,
    /// Holds of the core lock that have ended.
    sections_closed: u64,
    /// Polls waiting for a hold of the core lock to end.
    section_wakers: Vec<Waker>,
    /// The JS thread is inside [`JsOutbox::drain`].
    draining: bool,
    closed: bool,
}

/// What the engine has recorded for JS and not told it yet.
///
/// JS is told in the order the engine recorded, and what one hold of the core lock
/// recorded it is told together: nothing of a tick is visible to a drain until the
/// tick has released the lock.
#[derive(Default)]
struct JsOutbox {
    /// Never held across a call into JS or while taking the core lock.
    state: Mutex<OutboxState>,
}

impl JsOutbox {
    fn state(&self) -> std::sync::MutexGuard<'_, OutboxState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Under the core lock only.
    fn record(&self, event: JsEvent) {
        let mut state = self.state();
        if !state.closed {
            state.staging.push(event);
        }
    }

    fn open_section(&self) {
        self.state().section_open = true;
    }

    /// The holder of the core lock is about to release it. What it recorded becomes
    /// deliverable, or is given up if it is unwinding: the core it describes is then
    /// poisoned. Returns whether anything became deliverable, who waits for this, and
    /// what was given up — to be woken and dropped once the core lock is released.
    fn close_section(&self, keep: bool) -> (bool, Vec<Waker>, Vec<JsEvent>) {
        let mut state = self.state();
        let mut staged = std::mem::take(&mut state.staging);
        let recorded = keep && !staged.is_empty();
        if keep {
            state.ready.extend(staged.drain(..));
        }
        state.section_open = false;
        state.sections_closed += 1;
        (recorded, std::mem::take(&mut state.section_wakers), staged)
    }

    /// The hold of the core lock whose end makes deliverable everything recorded up
    /// to now.
    fn covering_section(&self) -> u64 {
        let state = self.state();
        state.sections_closed + u64::from(state.section_open)
    }

    /// Whether `section` has ended. If not, `waker` is woken when one does.
    fn section_closed(&self, section: u64, waker: &Waker) -> bool {
        let mut state = self.state();
        if state.sections_closed >= section || state.closed {
            return true;
        }
        state.section_wakers.push(waker.clone());
        false
    }

    fn has_ready(&self) -> bool {
        !self.state().ready.is_empty()
    }

    #[cfg(test)]
    fn has_staged(&self) -> bool {
        !self.state().staging.is_empty()
    }

    /// Nothing more is to be told about `handle`: it was unsubscribed.
    ///
    /// Called after the hold of the core lock that unsubscribed it has ended. Every
    /// event for the handle was recorded under an earlier hold, or under that one, so
    /// all of them are deliverable by now: there is nothing to look for among what
    /// the current holder is recording.
    fn forget(&self, handle: u64) {
        let forgotten: VecDeque<JsEvent> = {
            let mut state = self.state();
            let (forgotten, kept) = std::mem::take(&mut state.ready).into_iter().partition(
                |event| matches!(event, JsEvent::Delta { handle: of, .. } if *of == handle),
            );
            state.ready = kept;
            forgotten
        };
        // Released outside the outbox: dropping a callback tells JS to free it.
        drop(forgotten);
    }

    /// Tells JS what is deliverable, in order. On the JS thread only.
    ///
    /// Not re-entrant: a handler that calls back into the runtime does not get the
    /// next event inside itself. The drain it was called from tells it next.
    fn drain(&self) {
        {
            let mut state = self.state();
            if state.draining || state.closed {
                return;
            }
            state.draining = true;
        }
        loop {
            let next = {
                let mut state = self.state();
                match state.ready.pop_front() {
                    Some(event) => event,
                    None => {
                        // In the same hold as the empty pop: whoever records next
                        // finds nobody draining and asks for a drain.
                        state.draining = false;
                        return;
                    }
                }
            };
            // The call re-enters JS; a panic (a JS exception) must not skip the rest.
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match next {
                JsEvent::Delta {
                    callback,
                    json,
                    blobs,
                    ..
                } => callback.on_update(json, blobs),
                JsEvent::AuthFailure { callback, reason } => callback.on_failure(reason),
            }));
        }
    }

    /// The runtime is closed: nothing recorded is told, nothing more is recorded.
    fn close(&self) {
        let (undelivered, waiting) = {
            let mut state = self.state();
            state.closed = true;
            let mut undelivered: Vec<JsEvent> = std::mem::take(&mut state.staging);
            undelivered.extend(std::mem::take(&mut state.ready));
            (undelivered, std::mem::take(&mut state.section_wakers))
        };
        drop(undelivered);
        for waker in waiting {
            waker.wake();
        }
    }
}

fn make_subscription_callback(
    handle: u64,
    callback: Box<dyn SubscriptionCallback>,
    outbox: Arc<JsOutbox>,
) -> impl Fn(SubscriptionDelta) + Send + 'static {
    let callback: Arc<dyn SubscriptionCallback> = Arc::from(callback);
    move |delta: SubscriptionDelta| {
        let (payload, blobs) = subscription_delta_with_blob_sidecar(&delta);
        if let Ok(json) = serde_json::to_string(&payload) {
            outbox.record(JsEvent::Delta {
                handle,
                callback: Arc::clone(&callback),
                json,
                blobs,
            });
        }
    }
}

/// A future of the runtime's, resolved only once JS has been told everything the
/// engine recorded before the value existed.
///
/// The value is produced under the core lock — a tick sends it — and the waker fires
/// there and then, while what the same tick recorded for JS is still waiting for the
/// lock to be released. Polled at that moment the future would resolve ahead of the
/// deltas that came with it. So a value seen while somebody holds the core lock is
/// held back until that hold ends.
struct ToldFirst<F: Future> {
    inner: Pin<Box<F>>,
    outbox: Arc<JsOutbox>,
    /// Whether a poll that finds nothing tells JS what is deliverable all the same:
    /// for a call that ticks as it starts, as the call's own deltas.
    tells_while_pending: bool,
    /// The value, and the hold of the core lock that has to end before it is given.
    settled: Option<(F::Output, u64)>,
}

// `inner` is pinned on the heap and `settled` is never pinned.
impl<F: Future> Unpin for ToldFirst<F> {}

impl<F: Future> ToldFirst<F> {
    fn new(outbox: Arc<JsOutbox>, tells_while_pending: bool, inner: F) -> Self {
        Self {
            inner: Box::pin(inner),
            outbox,
            tells_while_pending,
            settled: None,
        }
    }
}

impl<F: Future> Future for ToldFirst<F> {
    type Output = F::Output;

    /// On the JS thread: uniffi polls a future on the thread that asks for it.
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = &mut *self;
        if this.settled.is_none() {
            match this.inner.as_mut().poll(cx) {
                Poll::Pending => {
                    if this.tells_while_pending {
                        this.outbox.drain();
                    }
                    return Poll::Pending;
                }
                Poll::Ready(value) => {
                    this.settled = Some((value, this.outbox.covering_section()));
                }
            }
        }
        let section = this.settled.as_ref().map_or(0, |(_, section)| *section);
        if !this.outbox.section_closed(section, cx.waker()) {
            // What the call's own tick recorded is deliverable already, and must not
            // wait for somebody else's hold of the lock to end — or for ever, if
            // this future is dropped before it is polled again.
            if this.tells_while_pending {
                this.outbox.drain();
            }
            return Poll::Pending;
        }
        this.outbox.drain();
        match this.settled.take() {
            Some((value, _)) => Poll::Ready(value),
            None => Poll::Pending,
        }
    }
}

// ============================================================================
// Callbacks (JS-implemented) for scheduling + sync output
// ============================================================================

#[uniffi::export(callback_interface)]
pub trait BatchedTickCallback: Send + Sync {
    /// Called by Rust when it wants JS to call `runtime.batched_tick()`.
    fn request_batched_tick(&self);
}

#[uniffi::export(callback_interface)]
pub trait SubscriptionCallback: Send + Sync {
    /// Called when a subscription produces an update.
    ///
    /// `delta_json` refers to entries of `blobs` via
    /// `{"type":"BlobRef","value":<idx>}`, the same convention the `*_with_blobs`
    /// write methods use in the other direction. Bytes never enter the JSON: a
    /// megabyte inlined as an array of Numbers is ~3.7 MB of text to serialize,
    /// hand across, parse, and then walk byte by byte on the JS side.
    fn on_update(&self, delta_json: String, blobs: Vec<Vec<u8>>);
}

#[uniffi::export(callback_interface)]
pub trait AuthFailureCallback: Send + Sync {
    /// Invoked when the Rust transport receives an auth rejection from the server.
    /// `reason` is a human-readable string (e.g. "Unauthorized").
    fn on_failure(&self, reason: String);
}

#[uniffi::export(callback_interface)]
pub trait MutationErrorCallback: Send + Sync {
    /// Invoked when a rejected local mutation was not handled by wait_for_batch.
    fn on_error(&self, event_json: String);
}

// ============================================================================
// RnScheduler
// ============================================================================

#[derive(Clone, Copy)]
enum SchedulerJob {
    /// Run the engine's batched tick, and ask JS to take what it recorded.
    Tick,
    /// Ask JS to take what was recorded by a call that will not deliver it itself.
    Notify,
    DeliverMutationErrors,
}

type SharedTickCallback = Arc<Mutex<Option<Arc<dyn BatchedTickCallback>>>>;

/// Whether the worker has hold of the engine: for a tick, or to take from it what it
/// has to report to JS.
///
/// Whoever lets go of a runtime waits here for the worker to give the engine back.
/// The engine then goes on the thread that let go of the runtime, and never on the
/// worker, in the middle of whatever that thread is tearing down beside it.
#[derive(Default)]
struct EngineGate {
    held: Mutex<bool>,
    given_back: std::sync::Condvar,
}

/// The worker's hold of the engine, from before it reaches for it until it has let go
/// of it.
struct WorkerHold(Arc<EngineGate>);

impl Drop for WorkerHold {
    fn drop(&mut self) {
        *self
            .0
            .held
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = false;
        self.0.given_back.notify_all();
    }
}

/// The engine as the worker has it: reached under a hold, and gone before the hold is.
/// The borrow checker holds both for a value made by `RnScheduler::engine`. Nothing but
/// this module's own care keeps code in it from reaching the engine another way, or
/// from taking a second hold while it has one: the gate knows of one hold, and is free
/// when any is given up.
struct HeldEngine<'hold> {
    core: Arc<SharedCore>,
    _hold: std::marker::PhantomData<&'hold WorkerHold>,
}

impl Drop for HeldEngine<'_> {
    /// Does nothing. It is here so that letting go of the engine counts as a use of
    /// the hold: without it the hold could be dropped first and this compile.
    fn drop(&mut self) {}
}

impl std::ops::Deref for HeldEngine<'_> {
    type Target = SharedCore;

    fn deref(&self) -> &SharedCore {
        &self.core
    }
}

type SharedCoreRef = Arc<Mutex<Option<Weak<SharedCore>>>>;

#[derive(Clone, Default)]
struct RnScheduler {
    scheduled: Arc<AtomicBool>,
    mutation_error_delivery_scheduled: Arc<AtomicBool>,
    core_ref: SharedCoreRef,
    callback: SharedTickCallback,
    outbox: Arc<JsOutbox>,
    // The worker thread is detached: it exits once the sender is dropped and
    // its queue drains. It must never be joined — see `shutdown` below.
    worker: Arc<Mutex<Option<std::sync::mpsc::Sender<SchedulerJob>>>>,
    shutdown: Arc<AtomicBool>,
    engine_gate: Arc<EngineGate>,
    #[cfg(test)]
    probe: Arc<TickProbe>,
}

/// What a test may see of, and hold up in, the scheduler's worker.
#[cfg(test)]
#[derive(Default)]
struct TickProbe {
    /// While set, a tick on the worker that recorded something for JS waits, after the
    /// engine's tick and before it releases the core lock: what it recorded is not
    /// deliverable yet, and whatever it resolved has been resolved.
    held_before_release: AtomicBool,
    /// The worker is waiting at `held_before_release`.
    holding: AtomicBool,
    /// The worker is about to ask for the core lock, or waiting for it.
    waiting_for_core: AtomicBool,
    /// Engine ticks the worker has begun, the lock taken and the scheduler found open.
    engine_ticks: AtomicU64,
    /// Tick jobs the worker has finished with, lock released, whatever they came to.
    ticks: AtomicU64,
    /// The next tick on the worker that recorded something for JS panics before it
    /// releases the core lock.
    panic_before_release: AtomicBool,
    /// The worker's thread has ended.
    worker_exited: AtomicBool,
    /// The worker has taken a job off its queue and is not done with it.
    working: AtomicBool,
    /// Threads asking for the core lock that do not have it yet.
    at_the_lock: AtomicU64,
    /// How long a tick on the worker holds the core lock for, at least.
    tick_takes_millis: AtomicU64,
    /// The thread the engine was dropped on.
    engine_dropped_on: Mutex<Option<std::thread::ThreadId>>,
    /// While set, the worker waits where it takes what it has to report of refused
    /// writes: the engine reached, and not yet given back.
    held_at_report: AtomicBool,
    /// The worker is waiting at `held_at_report`.
    holding_at_report: AtomicBool,
}

impl RnScheduler {
    fn set_core_ref(&self, core_ref: Weak<SharedCore>) {
        if let Ok(mut slot) = self.core_ref.lock() {
            *slot = Some(core_ref);
        }
    }

    fn set_callback(&self, cb: Option<Box<dyn BatchedTickCallback>>) {
        if let Ok(mut slot) = self.callback.lock() {
            *slot = cb.map(Arc::from);
        }
    }

    fn send_job(&self, job: SchedulerJob) {
        if self.shutdown.load(Ordering::SeqCst) {
            self.clear_job_scheduled(job);
            return;
        }

        let mut slot = match self.worker.lock() {
            Ok(slot) => slot,
            Err(_) => {
                self.clear_job_scheduled(job);
                return;
            }
        };

        if self.shutdown.load(Ordering::SeqCst) {
            self.clear_job_scheduled(job);
            return;
        }

        if slot.is_none() {
            *slot = self.spawn_worker();
        }

        let sent = slot.as_ref().is_some_and(|sender| sender.send(job).is_ok());

        if !sent {
            // Spawn failed or the worker died; drop the job and reset both
            // the slot and the debounce flag so the next schedule retries.
            self.clear_job_scheduled(job);
            *slot = None;
        }
    }

    fn clear_job_scheduled(&self, job: SchedulerJob) {
        match job {
            SchedulerJob::Tick => self.scheduled.store(false, Ordering::SeqCst),
            SchedulerJob::Notify => {}
            SchedulerJob::DeliverMutationErrors => self
                .mutation_error_delivery_scheduled
                .store(false, Ordering::SeqCst),
        }
    }

    fn spawn_worker(&self) -> Option<std::sync::mpsc::Sender<SchedulerJob>> {
        let (sender, receiver) = std::sync::mpsc::channel::<SchedulerJob>();
        // The worker's own handle on the scheduler must not hold the queue's sender:
        // the worker ends when the last sender is dropped, and would keep itself —
        // and the callbacks it holds — alive for ever.
        let scheduler = Self {
            worker: Arc::default(),
            ..self.clone()
        };
        let spawned = std::thread::Builder::new()
            .name("jazz-rn-scheduler".into())
            .spawn(move || {
                while let Ok(job) = receiver.recv() {
                    #[cfg(test)]
                    scheduler.probe.working.store(true, Ordering::SeqCst);
                    // Coalesces a burst of schedule calls into one job. It is also
                    // what lets a call from JS in between two ticks: the core lock is
                    // not fair, and a worker that took it again at once would keep
                    // JS out for as long as there is work.
                    std::thread::sleep(Duration::from_millis(1));
                    match job {
                        SchedulerJob::Tick => {
                            scheduler.scheduled.store(false, Ordering::SeqCst);
                            scheduler.tick();
                            scheduler.ask_js_to_drain();
                        }
                        SchedulerJob::Notify => scheduler.ask_js_to_drain(),
                        SchedulerJob::DeliverMutationErrors => {
                            scheduler
                                .mutation_error_delivery_scheduled
                                .store(false, Ordering::SeqCst);
                            scheduler.deliver_mutation_errors();
                        }
                    }
                    #[cfg(test)]
                    scheduler.probe.working.store(false, Ordering::SeqCst);
                }
                #[cfg(test)]
                scheduler.probe.worker_exited.store(true, Ordering::SeqCst);
            });

        match spawned {
            Ok(_) => Some(sender),
            Err(error) => {
                // Do not panic: this runs while `send_job` holds the worker
                // mutex, and poisoning it would silently disable the
                // scheduler for good.
                eprintln!("jazz-rn: failed to spawn scheduler thread: {error}");
                None
            }
        }
    }

    /// The engine's batched tick, on the worker. What it has to tell JS it records;
    /// nothing here calls JS.
    fn tick(&self) {
        let Some(hold) = self.take_hold() else {
            return;
        };
        let Some(core) = self.engine(&hold) else {
            return;
        };
        let ticked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            #[cfg(test)]
            self.probe.waiting_for_core.store(true, Ordering::SeqCst);
            let locked = core.lock_delivering();
            #[cfg(test)]
            self.probe.waiting_for_core.store(false, Ordering::SeqCst);
            let Ok(mut core) = locked else {
                return;
            };
            // `close()` shuts the scheduler down before it takes the lock: a tick
            // that waited for the lock behind it finds the store closed.
            if self.shutdown.load(Ordering::SeqCst) {
                return;
            }
            #[cfg(test)]
            self.probe.engine_ticks.fetch_add(1, Ordering::SeqCst);
            core.batched_tick();
            // The engine has logged it and retries; there is nobody to return it to.
            let _ = core.take_storage_flush_error();
            #[cfg(test)]
            std::thread::sleep(Duration::from_millis(
                self.probe.tick_takes_millis.load(Ordering::SeqCst),
            ));
            #[cfg(test)]
            if self.outbox.has_staged() {
                while self.probe.held_before_release.load(Ordering::SeqCst) {
                    self.probe.holding.store(true, Ordering::SeqCst);
                    std::thread::sleep(Duration::from_millis(1));
                }
                self.probe.holding.store(false, Ordering::SeqCst);
                if self
                    .probe
                    .panic_before_release
                    .swap(false, Ordering::SeqCst)
                {
                    panic!("the tick panicked");
                }
            }
        }));
        if ticked.is_err() {
            // The guard released the lock as it unwound, poisoned: every later call
            // reports it, as it would have with the tick on the caller's thread.
            eprintln!("jazz-rn: the engine's tick panicked");
        }
        #[cfg(test)]
        self.probe.ticks.fetch_add(1, Ordering::SeqCst);
    }

    /// The worker is about to take hold of the engine. `None` once the scheduler is
    /// shut down: looked at under the gate, so that whoever shut it down and then waits
    /// at the gate is not passed by a job that had not started.
    fn take_hold(&self) -> Option<WorkerHold> {
        let mut held = self
            .engine_gate
            .held
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if self.shutdown.load(Ordering::SeqCst) {
            return None;
        }
        *held = true;
        Some(WorkerHold(Arc::clone(&self.engine_gate)))
    }

    /// The engine, for the worker. Asked for with the hold it is reached under, and
    /// borrowed from it: the worker cannot have the engine before the hold or keep it
    /// after.
    fn engine<'hold>(&self, _hold: &'hold WorkerHold) -> Option<HeldEngine<'hold>> {
        let core = self
            .core_ref
            .lock()
            .ok()
            .and_then(|slot| slot.clone())
            .and_then(|core| core.upgrade())?;
        Some(HeldEngine {
            core,
            _hold: std::marker::PhantomData,
        })
    }

    /// Returns once the worker does not hold the engine. On a scheduler that is shut
    /// down it does not take it afterwards. The wait is not one for JS: the calls the
    /// worker makes that wait for the JS thread, it makes only after it has given the
    /// engine back. What goes to JS while it holds the engine — the wake of a future a
    /// tick resolves, or of one whose poll waited for a hold of the core lock to end;
    /// the release of a callback that is dropped — the generated glue posts to the JS
    /// thread and does not wait for (read in the released package's
    /// `cpp/generated/jazz_rn.cpp`; that file is generated at build time and is not in
    /// this repository).
    fn wait_for_the_worker(&self) {
        let mut held = self
            .engine_gate
            .held
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while *held {
            held = self
                .engine_gate
                .given_back
                .wait(held)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    /// Asks JS to call `batched_tick()` if there is something for it to take. The call
    /// blocks until the JS thread has run it, so it is made with no lock held.
    fn ask_js_to_drain(&self) {
        if self.outbox.has_ready() {
            Self::request_batched_tick(&self.callback);
        }
    }

    fn request_batched_tick(callback: &SharedTickCallback) {
        // Clone the callback out before invoking it: the call blocks until
        // the JS thread services it, and `set_callback(None)` (run by
        // `close()` on that same JS thread) must never wait on it.
        let callback = callback.lock().ok().and_then(|slot| slot.clone());
        if let Some(cb) = callback {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                cb.request_batched_tick();
            }));
        }
    }

    /// Reports to JS the writes the server refused. What there is to report is taken
    /// from the engine first and the engine given back: each report waits for the JS
    /// thread, and a worker that held the engine meanwhile would be the one left with
    /// it when JS lets go of the runtime.
    fn deliver_mutation_errors(&self) {
        let delivery = {
            let Some(hold) = self.take_hold() else {
                return;
            };
            let Some(core) = self.engine(&hold) else {
                return;
            };
            #[cfg(test)]
            {
                while self.probe.held_at_report.load(Ordering::SeqCst) {
                    self.probe.holding_at_report.store(true, Ordering::SeqCst);
                    std::thread::sleep(Duration::from_millis(1));
                }
                self.probe.holding_at_report.store(false, Ordering::SeqCst);
            }
            let delivery = match core.lock() {
                Ok(mut core) => core.pending_mutation_error_delivery(),
                Err(error) => {
                    eprintln!("jazz-rn: deliver pending mutation errors: {error:?}");
                    None
                }
            };
            delivery
        };
        let Some((callback, events)) = delivery else {
            return;
        };
        for event in events {
            // The callback re-enters JS; a panic (e.g. a JS exception) must not
            // kill the scheduler worker or skip the remaining events.
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| callback(&event)));
        }
    }

    fn shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
        self.scheduled.store(false, Ordering::SeqCst);
        self.mutation_error_delivery_scheduled
            .store(false, Ordering::SeqCst);

        // Dropping the sender lets the worker drain its queue and exit on its own. A
        // job still in the queue takes no hold of the engine (`take_hold`). A tick or
        // a notify job still asks JS to come for what is deliverable, if there is any
        // and the callback is still set. Never join it: JS
        // callbacks run via a blocking hop to the JS thread, and `close()`
        // runs on that same thread — joining here can deadlock the app.
        if let Ok(mut slot) = self.worker.lock() {
            *slot = None;
        }
    }
}

impl Scheduler for RnScheduler {
    fn schedule_batched_tick(&self) {
        // Debounce: only one pending tick request at a time.
        if self.scheduled.swap(true, Ordering::SeqCst) {
            return;
        }

        // The tick runs on the scheduler's worker, never where it was asked for:
        // the asker is the transport's thread, or a call that holds the core lock.
        // `NapiScheduler::schedule_batched_tick` in `jazz-napi` does the same.
        self.send_job(SchedulerJob::Tick);
    }

    fn schedule_mutation_error_delivery(&self) {
        if self
            .mutation_error_delivery_scheduled
            .swap(true, Ordering::SeqCst)
        {
            return;
        }

        self.send_job(SchedulerJob::DeliverMutationErrors);
    }
}

// ============================================================================
// The core, and the one way to it
// ============================================================================

type RnCoreType = RuntimeCore<SqliteStorage, RnScheduler>;

mod shared_core;
use shared_core::SharedCore;
#[cfg(test)]
use shared_core::CORE_LOCKS_HELD;

// ============================================================================
// RnRuntime
// ============================================================================

#[derive(uniffi::Object)]
pub struct RnRuntime {
    core: Arc<SharedCore>,
    outbox: Arc<JsOutbox>,
    /// The core holds a clone. This one is reached without the core lock: by `close()`,
    /// which must stop the worker before it waits for it.
    scheduler: RnScheduler,
}

impl Drop for RnRuntime {
    /// The last reference to the runtime is gone, closed or not. The worker takes no
    /// hold of the engine from here on, and the hold it has is waited for: the engine
    /// goes here, on the thread that released the runtime, whichever that is — the one
    /// that destroys the JS object, ends the last call or frees the last pending
    /// future. Not on the worker, at the end of a tick, beside the teardown of the JS
    /// that runtime was told. A call to JS the worker has set out on, or a refusal it
    /// has taken from the engine and has yet to report, is not waited for and is
    /// still made; it holds nothing of the engine.
    ///
    /// Two limits. This runs only once nothing holds the runtime, and an `async` call
    /// holds it from the moment it is made until JS frees its future: a JS world that
    /// goes away with a `query` or a `wait_for_batch` unanswered (a reload of the
    /// bundle) never frees it. That runtime — worker, transport, store connection —
    /// then lives as long as the process, and every such reload leaves one more. Not
    /// observed, read from the code: its connection stays up, and its worker goes on
    /// ticking on what the connection brings for as long as the call it makes after a
    /// tick, asking JS to come for what was recorded, returns. That call goes through
    /// a callback handle of the world that is gone: it either never returns — the
    /// worker stops there — or lands in the world that owns the callbacks now. Nobody
    /// comes for what such a runtime records for JS, and it is kept. Telling a runtime
    /// that its world is gone takes a call JS makes when a new world starts; there is
    /// none yet. And a runtime that was not closed is not flushed here: what it wrote
    /// since its last flush goes with the connection, as it did before the worker
    /// existed.
    fn drop(&mut self) {
        self.scheduler.set_callback(None);
        self.scheduler.shutdown();
        self.outbox.close();
        // A test that fails with the worker held up must end, not wait here for it.
        #[cfg(test)]
        if std::thread::panicking() {
            let probe = &self.scheduler.probe;
            probe.held_before_release.store(false, Ordering::SeqCst);
            probe.held_at_report.store(false, Ordering::SeqCst);
        }
        self.scheduler.wait_for_the_worker();
    }
}

impl RnRuntime {
    /// A call from JS that ticks: when it returns, JS has been told what its tick
    /// recorded, after whatever earlier ticks recorded.
    fn js_call<T>(
        &self,
        context: &'static str,
        f: impl FnOnce() -> Result<T, JazzRnError>,
    ) -> Result<T, JazzRnError> {
        let result = with_panic_boundary(context, f);
        self.outbox.drain();
        result
    }
}

#[uniffi::export]
impl RnRuntime {
    /// `release_declared_indexes` releases the store from the app's declared indexes,
    /// for a rollback to an engine that does not maintain them (see
    /// `QueryManager::release_declared_indexes`).
    #[uniffi::constructor(default(release_declared_indexes = None))]
    pub fn new(
        schema_json: String,
        app_id: String,
        jazz_env: String,
        user_branch: String,
        tier: Option<String>,
        data_path: Option<String>,
        release_declared_indexes: Option<bool>,
    ) -> Result<Arc<Self>, JazzRnError> {
        with_panic_boundary("new", || {
            // Put the engine-log subscriber in place with everything filtered
            // out, so `setEngineLogLevel` has a filter to reload later.
            engine_log::install();

            let schema: Schema = serde_json::from_str(&schema_json).map_err(json_err)?;

            let persistence_tier = tier.as_deref().map(parse_tier).transpose()?;

            let mut sync_manager = SyncManager::new();
            if let Some(t) = persistence_tier {
                sync_manager = sync_manager.with_durability_tier(t);
            }

            let app_id_obj =
                AppId::from_string(&app_id).unwrap_or_else(|_| AppId::from_name(&app_id));
            let mut schema_manager =
                SchemaManager::new(sync_manager, schema, app_id_obj, &jazz_env, &user_branch)
                    .map_err(|e| JazzRnError::Schema {
                        message: format!("{:?}", e),
                    })?;

            let resolved_data_path = data_path.unwrap_or_else(|| {
                let sanitized_app_id: String = app_id
                    .chars()
                    .map(|c| {
                        if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                            c
                        } else {
                            '_'
                        }
                    })
                    .collect();
                let mut default_path = std::env::temp_dir();
                default_path.push(format!("{sanitized_app_id}.sqlite"));
                default_path.to_string_lossy().into_owned()
            });
            let storage = SqliteStorage::open_read_through(&resolved_data_path).map_err(|e| {
                JazzRnError::Runtime {
                    message: format!(
                        "Failed to open SQLite storage at '{}': {:?}",
                        resolved_data_path, e
                    ),
                }
            })?;

            // Load previously-persisted schema history, permissions bundle, and lens
            // catalogue entries from storage into the in-memory schema manager so
            // offline cold-starts can decode and serve locally stored rows.
            if let Err(error) =
                rehydrate_schema_manager_from_catalogue(&mut schema_manager, &storage, app_id_obj)
            {
                eprintln!(
                    "jazz-rn: failed to rehydrate schema manager from catalogue storage for app {app_id_obj}: {error}"
                );
            }

            if release_declared_indexes == Some(true) {
                schema_manager
                    .query_manager_mut()
                    .release_declared_indexes();
            }

            let scheduler = RnScheduler::default();
            let outbox = Arc::clone(&scheduler.outbox);

            let mut core = RuntimeCore::new(schema_manager, storage, scheduler.clone());
            core.persist_schema();
            let core = Arc::new(SharedCore::new(
                core,
                Arc::clone(&outbox),
                scheduler.clone(),
            ));
            scheduler.set_core_ref(Arc::downgrade(&core));

            Ok(Arc::new(Self {
                core,
                outbox,
                scheduler,
            }))
        })
    }

    /// Register a JS callback that schedules `batched_tick()` calls.
    pub fn on_batched_tick_needed(
        &self,
        callback: Option<Box<dyn BatchedTickCallback>>,
    ) -> Result<(), JazzRnError> {
        with_panic_boundary("on_batched_tick_needed", || {
            // Not through the core: `close()` takes the callback away first, and
            // must not wait for a tick to do it.
            let installed = callback.is_some();
            self.scheduler.set_callback(callback);
            if installed {
                // What was recorded while nobody could be asked.
                self.scheduler.send_job(SchedulerJob::Notify);
            }
            Ok(())
        })
    }

    /// Run a batched tick. JS should call this when asked via `on_batched_tick_needed`.
    pub fn batched_tick(&self) -> Result<(), JazzRnError> {
        // The doc comment above is part of the exported interface: the checksum the
        // generated bindings verify as they load covers it, so it stays as it shipped.
        //
        // What it does now is tell JS what the engine's ticks have recorded for it.
        // The tick itself has already run, on the scheduler's worker: this takes no
        // lock and waits for nothing.
        self.js_call("batched_tick", || Ok(()))
    }

    // =========================================================================
    // CRUD
    // =========================================================================

    pub fn insert(
        &self,
        table: String,
        values_json: String,
        write_context_json: Option<String>,
        object_id: Option<String>,
    ) -> Result<String, JazzRnError> {
        self.js_call("insert", || {
            let named_values = convert_insert_values(&values_json)?;
            let write_context = parse_write_context(write_context_json)?;
            let object_id = parse_external_object_id(object_id.as_deref())
                .map_err(|message| JazzRnError::InvalidUuid { message })?;
            let mut core = self.core.lock_delivering()?;
            let ((id, row_values), batch_id) = core
                .insert_with_id(&table, named_values, object_id, write_context.as_ref())
                .map_err(runtime_err)?;
            serde_json::to_string(&serde_json::json!({
                "id": id.uuid().to_string(),
                "values": row_values,
                "batchId": batch_id.to_string(),
            }))
            .map_err(|e| JazzRnError::Internal {
                message: format!("insert serialization failed: {e}"),
            })
        })
    }

    /// `insert` with Bytea payloads passed as raw bytes instead of hex-in-JSON.
    ///
    /// `values_json` refers to entries of `blobs` via `{"type":"BlobRef","value":<idx>}`.
    /// The returned row encodes any Bytea that is byte-identical to an input blob as the
    /// same `BlobRef`, so a megabyte chunk is neither hex-encoded on the way in nor
    /// serialized back on the way out. See `FfiJsonValue::BlobRef`.
    pub fn insert_with_blobs(
        &self,
        table: String,
        values_json: String,
        blobs: Vec<Vec<u8>>,
        write_context_json: Option<String>,
        object_id: Option<String>,
    ) -> Result<String, JazzRnError> {
        self.js_call("insert_with_blobs", || {
            let named_values = decode_ffi_json_record_with_blobs(&values_json, &blobs)?;
            let write_context = parse_write_context(write_context_json)?;
            let object_id = parse_external_object_id(object_id.as_deref())
                .map_err(|message| JazzRnError::InvalidUuid { message })?;
            let mut core = self.core.lock_delivering()?;
            let ((id, row_values), batch_id) = core
                .insert_with_id(&table, named_values, object_id, write_context.as_ref())
                .map_err(runtime_err)?;
            serde_json::to_string(&serde_json::json!({
                "id": id.uuid().to_string(),
                "values": encode_return_values_with_blob_refs(&row_values, &blobs),
                "batchId": batch_id.to_string(),
            }))
            .map_err(|e| JazzRnError::Internal {
                message: format!("insert_with_blobs serialization failed: {e}"),
            })
        })
    }

    /// `restore` with Bytea payloads passed as raw bytes. See [`Self::insert_with_blobs`].
    pub fn restore_with_blobs(
        &self,
        table: String,
        object_id: String,
        values_json: String,
        blobs: Vec<Vec<u8>>,
        write_context_json: Option<String>,
    ) -> Result<String, JazzRnError> {
        self.js_call("restore_with_blobs", || {
            let uuid = uuid::Uuid::parse_str(&object_id).map_err(|e| JazzRnError::InvalidUuid {
                message: e.to_string(),
            })?;
            let oid = ObjectId::from_uuid(uuid);
            let named_values = decode_ffi_json_record_with_blobs(&values_json, &blobs)?;
            let write_context = parse_write_context(write_context_json)?;
            let mut core = self.core.lock_delivering()?;
            let ((id, row_values), batch_id) = core
                .restore(&table, oid, named_values, write_context.as_ref())
                .map_err(runtime_err)?;
            serde_json::to_string(&serde_json::json!({
                "id": id.uuid().to_string(),
                "values": encode_return_values_with_blob_refs(&row_values, &blobs),
                "batchId": batch_id.to_string(),
            }))
            .map_err(|e| JazzRnError::Internal {
                message: format!("restore_with_blobs serialization failed: {e}"),
            })
        })
    }

    /// `update` with Bytea payloads passed as raw bytes. See [`Self::insert_with_blobs`].
    pub fn update_with_blobs(
        &self,
        object_id: String,
        values_json: String,
        blobs: Vec<Vec<u8>>,
        write_context_json: Option<String>,
    ) -> Result<String, JazzRnError> {
        self.js_call("update_with_blobs", || {
            let uuid = uuid::Uuid::parse_str(&object_id).map_err(|e| JazzRnError::InvalidUuid {
                message: e.to_string(),
            })?;
            let oid = ObjectId::from_uuid(uuid);
            let updates: Vec<(String, Value)> =
                decode_ffi_json_record_with_blobs(&values_json, &blobs)?
                    .into_iter()
                    .collect();
            let write_context = parse_write_context(write_context_json)?;
            let mut core = self.core.lock_delivering()?;
            let batch_id = core
                .update(oid, updates, write_context.as_ref())
                .map_err(runtime_err)?;
            serde_json::to_string(&serde_json::json!({
                "batchId": batch_id.to_string(),
            }))
            .map_err(|e| JazzRnError::Internal {
                message: format!("update_with_blobs serialization failed: {e}"),
            })
        })
    }

    /// `upsert` with Bytea payloads passed as raw bytes. See [`Self::insert_with_blobs`].
    pub fn upsert_with_blobs(
        &self,
        table: String,
        object_id: String,
        values_json: String,
        blobs: Vec<Vec<u8>>,
        write_context_json: Option<String>,
    ) -> Result<String, JazzRnError> {
        self.js_call("upsert_with_blobs", || {
            let uuid = uuid::Uuid::parse_str(&object_id).map_err(|e| JazzRnError::InvalidUuid {
                message: e.to_string(),
            })?;
            let oid = ObjectId::from_uuid(uuid);
            let named_values = decode_ffi_json_record_with_blobs(&values_json, &blobs)?;
            let write_context = parse_write_context(write_context_json)?;
            let mut core = self.core.lock_delivering()?;
            let batch_id = core
                .upsert(&table, oid, named_values, write_context.as_ref())
                .map_err(runtime_err)?;
            serde_json::to_string(&serde_json::json!({
                "batchId": batch_id.to_string(),
            }))
            .map_err(|e| JazzRnError::Internal {
                message: format!("upsert_with_blobs serialization failed: {e}"),
            })
        })
    }

    pub fn restore(
        &self,
        table: String,
        object_id: String,
        values_json: String,
        write_context_json: Option<String>,
    ) -> Result<String, JazzRnError> {
        self.js_call("restore", || {
            let uuid = uuid::Uuid::parse_str(&object_id).map_err(|e| JazzRnError::InvalidUuid {
                message: e.to_string(),
            })?;
            let oid = ObjectId::from_uuid(uuid);
            let named_values = convert_insert_values(&values_json)?;
            let write_context = parse_write_context(write_context_json)?;
            let mut core = self.core.lock_delivering()?;
            let ((id, row_values), batch_id) = core
                .restore(&table, oid, named_values, write_context.as_ref())
                .map_err(runtime_err)?;
            serde_json::to_string(&serde_json::json!({
                "id": id.uuid().to_string(),
                "values": row_values,
                "batchId": batch_id.to_string(),
            }))
            .map_err(|e| JazzRnError::Internal {
                message: format!("restore serialization failed: {e}"),
            })
        })
    }

    pub fn update(
        &self,
        object_id: String,
        values_json: String,
        write_context_json: Option<String>,
    ) -> Result<String, JazzRnError> {
        self.js_call("update", || {
            let uuid = uuid::Uuid::parse_str(&object_id).map_err(|e| JazzRnError::InvalidUuid {
                message: e.to_string(),
            })?;
            let oid = ObjectId::from_uuid(uuid);
            let updates = convert_updates(&values_json)?;
            let write_context = parse_write_context(write_context_json)?;
            let mut core = self.core.lock_delivering()?;
            let batch_id = core
                .update(oid, updates, write_context.as_ref())
                .map_err(runtime_err)?;
            serde_json::to_string(&serde_json::json!({
                "batchId": batch_id.to_string(),
            }))
            .map_err(|e| JazzRnError::Internal {
                message: format!("update serialization failed: {e}"),
            })
        })
    }

    pub fn upsert(
        &self,
        table: String,
        object_id: String,
        values_json: String,
        write_context_json: Option<String>,
    ) -> Result<String, JazzRnError> {
        self.js_call("upsert", || {
            let uuid = uuid::Uuid::parse_str(&object_id).map_err(|e| JazzRnError::InvalidUuid {
                message: e.to_string(),
            })?;
            let oid = ObjectId::from_uuid(uuid);
            let named_values = convert_insert_values(&values_json)?;
            let write_context = parse_write_context(write_context_json)?;
            let mut core = self.core.lock_delivering()?;
            let batch_id = core
                .upsert(&table, oid, named_values, write_context.as_ref())
                .map_err(runtime_err)?;
            serde_json::to_string(&serde_json::json!({
                "batchId": batch_id.to_string(),
            }))
            .map_err(|e| JazzRnError::Internal {
                message: format!("upsert serialization failed: {e}"),
            })
        })
    }

    pub fn begin_batch(&self, batch_mode: String) -> Result<String, JazzRnError> {
        with_panic_boundary("begin_batch", || {
            let batch_mode = parse_batch_mode_input(&batch_mode)
                .map_err(|message| JazzRnError::InvalidJson { message })?;
            let mut core = self.core.lock()?;
            Ok(core.begin_batch(batch_mode).to_string())
        })
    }

    pub fn rollback_batch(&self, batch_id: String) -> Result<bool, JazzRnError> {
        self.js_call("rollback_batch", || {
            let batch_id = parse_batch_id_input(&batch_id)
                .map_err(|message| JazzRnError::InvalidUuid { message })?;
            let mut core = self.core.lock_delivering()?;
            core.rollback_batch(batch_id).map_err(runtime_err)
        })
    }

    #[uniffi::method(name = "delete")]
    pub fn delete_row(
        &self,
        object_id: String,
        write_context_json: Option<String>,
    ) -> Result<String, JazzRnError> {
        self.js_call("delete", || {
            let uuid = uuid::Uuid::parse_str(&object_id).map_err(|e| JazzRnError::InvalidUuid {
                message: e.to_string(),
            })?;
            let oid = ObjectId::from_uuid(uuid);
            let write_context = parse_write_context(write_context_json)?;
            let mut core = self.core.lock_delivering()?;
            let batch_id = core
                .delete(oid, write_context.as_ref())
                .map_err(runtime_err)?;
            serde_json::to_string(&serde_json::json!({
                "batchId": batch_id.to_string(),
            }))
            .map_err(|e| JazzRnError::Internal {
                message: format!("delete serialization failed: {e}"),
            })
        })
    }

    /// Wait for a local batch to settle at the requested durability tier.
    pub async fn wait_for_batch(&self, batch_id: String, tier: String) -> Result<(), JazzRnError> {
        // Registering the wait does not tick: nothing is told until it resolves.
        let waited = with_async_panic_boundary("wait_for_batch", || async move {
            let batch_id = parse_batch_id_input(&batch_id)
                .map_err(|message| JazzRnError::InvalidUuid { message })?;
            let tier = parse_tier(&tier)?;
            let receiver = {
                let mut core = self.core.lock()?;
                core.wait_for_batch(batch_id, tier).map_err(runtime_err)?
            };

            match receiver.await {
                Ok(Ok(())) => Ok(()),
                Ok(Err(rejection)) => Err(JazzRnError::Runtime {
                    message: format!(
                        "Persisted batch {} was rejected ({}): {}",
                        rejection.batch_id, rejection.code, rejection.reason
                    ),
                }),
                Err(_) => Err(JazzRnError::Runtime {
                    message: "Wait for batch cancelled".into(),
                }),
            }
        });
        ToldFirst::new(Arc::clone(&self.outbox), false, waited).await
    }

    // =========================================================================
    // Queries
    // =========================================================================

    /// One-shot query returning a JSON string:
    /// `[{ "id": "<uuid>", "values": [ {type, value}, ... ] }, ...]`.
    ///
    /// `async` so the JS thread is not blocked while the query future is
    /// waiting on a later `batched_tick` to settle (which is itself driven
    /// from JS via the `on_batched_tick_needed` callback). A synchronous
    /// `block_on` here can deadlock for any query that needs more than the
    /// inline `immediate_tick` to resolve.
    pub async fn query(
        &self,
        query_json: String,
        session_json: Option<String>,
        tier: Option<String>,
        options_json: Option<String>,
    ) -> Result<String, JazzRnError> {
        // The first poll ticks, and tells JS what that tick recorded as a write does.
        let queried = with_async_panic_boundary("query", || async move {
            let query = parse_query(&query_json)?;
            let session = parse_session(session_json)?;
            let (durability, propagation, transaction_batch_id) =
                parse_read_durability_options(tier.as_deref(), options_json.as_deref())
                    .map_err(|message| JazzRnError::InvalidJson { message })?;

            let fut = {
                let mut core = self.core.lock_delivering()?;
                core.query_with_local_batch(
                    query,
                    session,
                    durability,
                    propagation,
                    transaction_batch_id,
                )
                .map_err(runtime_err)?
            };
            let results = fut.await.map_err(runtime_err)?;

            let rows_json: Vec<serde_json::Value> = results
                .into_iter()
                .map(|(id, values)| {
                    serde_json::json!({
                        "id": id.uuid().to_string(),
                        "values": values,
                    })
                })
                .collect();

            serde_json::to_string(&rows_json).map_err(json_err)
        });
        ToldFirst::new(Arc::clone(&self.outbox), true, queried).await
    }

    // =========================================================================
    // Subscriptions
    // =========================================================================

    pub fn unsubscribe(&self, handle: u64) -> Result<(), JazzRnError> {
        with_panic_boundary("unsubscribe", || {
            // After this no tick records anything for the handle: recording happens
            // under the same lock.
            let unsubscribed = self
                .core
                .lock()
                .map(|mut core| core.unsubscribe(SubscriptionHandle(handle)));
            // And what was recorded before it is not told, whatever the lock came to:
            // the app has let go of the handle even if a tick that panicked left the
            // core poisoned. Nothing is told inside this call at all: none was before
            // the tick left the JS thread.
            self.outbox.forget(handle);
            unsubscribed
        })
    }

    /// Phase 1 of 2-phase subscribe: allocate a handle and store query params.
    pub fn create_subscription(
        &self,
        query_json: String,
        session_json: Option<String>,
        tier: Option<String>,
    ) -> Result<u64, JazzRnError> {
        with_panic_boundary("create_subscription", || {
            let (query, session, durability) =
                parse_subscription_inputs(&query_json, session_json, tier)?;

            let mut core = self.core.lock()?;

            let handle =
                core.create_subscription(query, session, durability, QueryPropagation::Full);

            Ok(handle.0)
        })
    }

    /// Phase 2 of 2-phase subscribe: compile, register, sync, attach callback, tick.
    pub fn execute_subscription(
        &self,
        handle: u64,
        callback: Box<dyn SubscriptionCallback>,
    ) -> Result<(), JazzRnError> {
        self.js_call("execute_subscription", || {
            let mut core = self.core.lock_delivering()?;
            let callback = make_subscription_callback(handle, callback, Arc::clone(&self.outbox));

            core.execute_subscription(SubscriptionHandle(handle), callback)
                .map_err(runtime_err)?;

            Ok(())
        })
    }

    // =========================================================================
    // Schema/state access
    // =========================================================================

    pub fn get_schema_hash(&self) -> Result<String, JazzRnError> {
        with_panic_boundary("get_schema_hash", || {
            let core = self.core.lock()?;
            let schema = core.current_schema();
            Ok(SchemaHash::compute(schema).to_string())
        })
    }

    pub fn on_mutation_error(
        &self,
        callback: Box<dyn MutationErrorCallback>,
    ) -> Result<(), JazzRnError> {
        with_panic_boundary("on_mutation_error", || {
            let callback: CoreMutationErrorCallback = Arc::new(move |event| {
                let Ok(event_json) = serde_json::to_string(&serialize_mutation_error_event(event))
                else {
                    return;
                };
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    callback.on_error(event_json);
                }));
            });
            self.core
                .lock()?
                .set_mutation_error_callback(Some(callback));
            Ok(())
        })
    }

    pub fn commit_batch(&self, batch_id: String) -> Result<(), JazzRnError> {
        self.js_call("commit_batch", || {
            let batch_id = parse_batch_id_input(&batch_id)
                .map_err(|message| JazzRnError::InvalidUuid { message })?;
            let mut core = self.core.lock_delivering()?;
            core.commit_batch(batch_id).map_err(runtime_err)
        })
    }

    /// Flush and close the underlying storage, releasing filesystem locks.
    pub fn close(&self) -> Result<(), JazzRnError> {
        let closed = with_panic_boundary("close", || {
            // Before the lock is asked for: a tick that has not started must not
            // start, and one that waits for the lock behind this call finds the
            // scheduler shut down when it gets it.
            self.scheduler.set_callback(None);
            self.scheduler.shutdown();
            // Waits for a tick in flight. It cannot wait for ever: no tick waits for
            // JS while it holds the lock.
            let mut core = self.core.lock()?;
            core.set_mutation_error_callback(None);
            let flush_result = core.flush_storage();
            let flush_wal_result = core.flush_wal();
            let close_result = core.storage().close();

            flush_result.map_err(runtime_err)?;
            flush_wal_result.map_err(runtime_err)?;
            close_result.map_err(runtime_err)
        });
        // Whatever the close came to, a closed runtime tells JS nothing more.
        self.outbox.close();
        closed
    }

    /// Connect to a Jazz server over WebSocket.
    ///
    /// Parses `auth_json` into `AuthConfig`, wires a `TransportManager` into
    /// `RuntimeCore`, and spawns the manager loop on a dedicated Tokio thread.
    pub fn connect(&self, url: String, auth_json: String) -> Result<(), JazzRnError> {
        with_panic_boundary("connect", || {
            let auth: jazz_tools::transport_manager::AuthConfig =
                serde_json::from_str(&auth_json).map_err(json_err)?;
            let tick = RnTickNotifier {
                scheduler: self.scheduler.clone(),
            };
            let manager = {
                let mut core = self.core.lock()?;
                jazz_tools::runtime_core::install_transport::<
                    _,
                    _,
                    jazz_tools::ws_stream::NativeWsStream,
                    _,
                >(&mut core, url, auth, tick)
            };
            std::thread::spawn(move || {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("tokio rt");
                rt.block_on(manager.run());
            });
            Ok(())
        })
    }

    /// Disconnect from the Jazz server and drop the transport handle.
    pub fn disconnect(&self) {
        if let Ok(mut core) = self.core.lock() {
            let server_id = if let Some(handle) = core.transport() {
                handle.disconnect();
                Some(handle.server_id)
            } else {
                None
            };
            if let Some(server_id) = server_id {
                core.remove_server(server_id);
            }
            core.clear_transport();
        }
    }

    /// Push updated auth credentials into the live transport.
    pub fn update_auth(&self, auth_json: String) -> Result<(), JazzRnError> {
        with_panic_boundary("update_auth", || {
            let auth: jazz_tools::transport_manager::AuthConfig =
                serde_json::from_str(&auth_json).map_err(json_err)?;
            if let Ok(core) = self.core.lock() {
                if let Some(handle) = core.transport() {
                    handle.update_auth(auth);
                }
            }
            Ok(())
        })
    }

    /// Register a callback that fires when the transport receives an auth
    /// rejection from the server during the WS handshake.
    pub fn on_auth_failure(
        &self,
        callback: Box<dyn AuthFailureCallback>,
    ) -> Result<(), JazzRnError> {
        with_panic_boundary("on_auth_failure", || {
            // Reported from inside a tick, under the core lock: recorded there, told
            // on the JS thread like a delta.
            let callback: Arc<dyn AuthFailureCallback> = Arc::from(callback);
            let outbox = Arc::clone(&self.outbox);
            self.core.lock()?.set_auth_failure_callback(move |reason| {
                outbox.record(JsEvent::AuthFailure {
                    callback: Arc::clone(&callback),
                    reason,
                });
            });
            Ok(())
        })
    }
}

// ============================================================================
// RnTickNotifier
// ============================================================================

/// `TickNotifier` implementation for the React Native (UniFFI) runtime.
///
/// Holds a clone of `RnScheduler` and calls `schedule_batched_tick()` whenever
/// the transport layer needs to wake up `batched_tick`.
struct RnTickNotifier {
    scheduler: RnScheduler,
}

impl jazz_tools::transport_manager::TickNotifier for RnTickNotifier {
    fn notify(&self) {
        self.scheduler.schedule_batched_tick();
    }
}

#[cfg(test)]
mod delta_blob_size_tests {
    use super::*;

    /// A megabyte of blob, measured on each of the three routes it can take out
    /// of the bridge.
    ///
    /// Size rather than duration on purpose: the JSON text length is
    /// deterministic, so this cannot flake, and it is a direct proxy for the work
    /// on both sides — every character is one the Rust side writes and the JS
    /// side parses, and for the array form `toByteArray` then walks the result
    /// once per byte.
    fn json_len(value: &serde_json::Value) -> usize {
        serde_json::to_string(value).expect("json").len()
    }

    #[test]
    fn a_blob_must_not_be_inlined_into_the_delta_text() {
        // Byte values spread over the whole range, not a fill: a run of single-digit
        // bytes renders as two characters each and would flatter the number. Real
        // media averages closer to four.
        let payload: Vec<u8> = (0..1024 * 1024).map(|index| (index % 256) as u8).collect();
        let blob = Value::Bytea(payload.clone());

        // What a subscription delta does today: `Value`'s own human-readable
        // serde, which renders `Vec<u8>` as an array of Numbers.
        let as_numbers = json_len(&serde_json::to_value(&blob).expect("serde"));

        // What the write direction falls back to when it has no matching input
        // blob: hex. Half the characters, still linear in the payload.
        let as_hex = json_len(&encode_return_value_with_blob_refs(&blob, &[]));

        // What a delivery does now: the bytes travel beside the JSON, not inside
        // it, so the text is a fixed handful of characters whatever the blob weighs.
        let mut collected: Vec<Vec<u8>> = Vec::new();
        let as_reference = json_len(&encode_value_into_sidecar(&blob, &mut collected));
        assert_eq!(
            collected,
            vec![payload],
            "the bytes must reach the sidecar intact"
        );

        assert!(
            as_reference < 64,
            "a referenced blob should leave ~30 characters of JSON, got {as_reference}",
        );
        // The two inline forms are what a delivery must never fall back to. Kept
        // measured rather than deleted so the gap stays visible: 3.7 MB of text for
        // a megabyte of payload, or 2 MB as hex.
        assert!(
            as_numbers > 3_000_000,
            "expected the inline form to be huge, got {as_numbers}"
        );
        assert!(
            as_hex > 2_000_000,
            "expected hex to be linear too, got {as_hex}"
        );
    }
}

#[cfg(test)]
mod blob_codec_tests {
    use super::*;

    fn record(json: &str, blobs: &[Vec<u8>]) -> HashMap<String, Value> {
        decode_ffi_json_record_with_blobs(json, blobs).expect("record should decode")
    }

    #[test]
    fn blob_ref_decodes_to_the_referenced_bytes() {
        let blobs = vec![vec![1u8, 2, 3], vec![0xff; 4]];
        let values = record(
            r#"{"a":{"type":"BlobRef","value":0},"b":{"type":"BlobRef","value":1}}"#,
            &blobs,
        );
        assert_eq!(values["a"], Value::Bytea(vec![1, 2, 3]));
        assert_eq!(values["b"], Value::Bytea(vec![0xff; 4]));
    }

    #[test]
    fn blob_ref_decodes_inside_nested_arrays() {
        let blobs = vec![vec![7u8, 8]];
        let values = record(
            r#"{"a":{"type":"Array","value":[{"type":"BlobRef","value":0}]}}"#,
            &blobs,
        );
        assert_eq!(values["a"], Value::Array(vec![Value::Bytea(vec![7, 8])]));
    }

    #[test]
    fn blob_ref_out_of_range_is_an_error() {
        let result = decode_ffi_json_record_with_blobs(
            r#"{"a":{"type":"BlobRef","value":1}}"#,
            &[vec![1u8]],
        );
        assert!(matches!(result, Err(JazzRnError::InvalidJson { .. })));
    }

    #[test]
    fn hex_bytea_still_decodes_alongside_blob_refs() {
        let values = record(r#"{"a":{"type":"Bytea","value":"0aff"}}"#, &[]);
        assert_eq!(values["a"], Value::Bytea(vec![0x0a, 0xff]));
    }

    #[test]
    fn legacy_record_decoder_rejects_blob_refs() {
        let result = decode_ffi_json_record(r#"{"a":{"type":"BlobRef","value":0}}"#);
        assert!(matches!(result, Err(JazzRnError::InvalidJson { .. })));
    }

    #[test]
    fn return_encoding_swaps_matching_bytes_for_blob_refs() {
        let blobs = vec![vec![9u8; 16]];
        let encoded = encode_return_value_with_blob_refs(&Value::Bytea(vec![9u8; 16]), &blobs);
        assert_eq!(encoded, serde_json::json!({"type": "BlobRef", "value": 0}));
    }

    #[test]
    fn return_encoding_falls_back_to_hex_for_unknown_bytes() {
        let blobs = vec![vec![9u8; 16]];
        let encoded = encode_return_value_with_blob_refs(&Value::Bytea(vec![1u8, 2]), &blobs);
        assert_eq!(
            encoded,
            serde_json::json!({"type": "Bytea", "value": "0102"})
        );
    }

    #[test]
    fn return_encoding_matches_legacy_serde_for_non_bytea_values() {
        // The adapter parses both legacy and blob returns with the same decoder, so every
        // non-Bytea value must keep the exact legacy wire shape.
        let samples = vec![
            Value::Integer(41),
            Value::Text("hello".into()),
            Value::Timestamp(1_700_000_000_000),
            Value::Boolean(true),
            Value::Null,
            Value::Array(vec![Value::Integer(1), Value::Text("x".into())]),
            Value::Row {
                id: None,
                values: vec![Value::Integer(5)],
            },
        ];
        for value in samples {
            let legacy = serde_json::to_value(&value).expect("legacy serialization");
            let with_blobs = encode_return_value_with_blob_refs(&value, &[]);
            assert_eq!(with_blobs, legacy, "shape diverged for {value:?}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::thread::ThreadId;

    struct BlockingTickCallback {
        invocations: AtomicUsize,
        entered_tx: mpsc::Sender<(usize, ThreadId)>,
        unblock_first_rx: Mutex<mpsc::Receiver<()>>,
    }

    impl BatchedTickCallback for BlockingTickCallback {
        fn request_batched_tick(&self) {
            let invocation = self.invocations.fetch_add(1, Ordering::SeqCst) + 1;
            self.entered_tx
                .send((invocation, std::thread::current().id()))
                .expect("test should receive tick callback");

            if invocation == 1 {
                self.unblock_first_rx
                    .lock()
                    .expect("test unblock receiver should not be poisoned")
                    .recv_timeout(Duration::from_secs(1))
                    .expect("test should unblock first tick callback");
            }
        }
    }

    struct NoAuthFailures;

    impl AuthFailureCallback for NoAuthFailures {
        fn on_failure(&self, _reason: String) {}
    }

    /// A scheduler with something recorded for JS that nobody takes: every tick job
    /// asks JS to come for it.
    fn scheduler_with_something_for_js() -> RnScheduler {
        let scheduler = RnScheduler::default();
        scheduler
            .outbox
            .state()
            .ready
            .push_back(JsEvent::AuthFailure {
                callback: Arc::new(NoAuthFailures),
                reason: "left for JS".into(),
            });
        scheduler
    }

    #[test]
    fn scheduler_coalesces_bursts_on_one_worker_for_follow_up_ticks() {
        // This is intentionally an internal scheduler test: constant worker
        // count and callback serialization are not observable through the
        // public RN runtime API.
        let scheduler = scheduler_with_something_for_js();
        let (entered_tx, entered_rx) = mpsc::channel();
        let (unblock_first_tx, unblock_first_rx) = mpsc::channel();

        scheduler.set_callback(Some(Box::new(BlockingTickCallback {
            invocations: AtomicUsize::new(0),
            entered_tx,
            unblock_first_rx: Mutex::new(unblock_first_rx),
        })));

        scheduler.schedule_batched_tick();
        let (first_invocation, first_thread_id) = entered_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("first tick callback should run");
        assert_eq!(first_invocation, 1);

        for _ in 0..5 {
            scheduler.schedule_batched_tick();
        }
        assert!(
            entered_rx.recv_timeout(Duration::from_millis(100)).is_err(),
            "follow-up tick callback ran concurrently with the blocked callback"
        );

        unblock_first_tx
            .send(())
            .expect("first tick callback should still be waiting");
        let (second_invocation, second_thread_id) = entered_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("follow-up tick callback should run after the first returns");
        assert_eq!(second_invocation, 2);
        assert_eq!(
            second_thread_id, first_thread_id,
            "follow-up tick ran on a freshly spawned scheduler thread"
        );
        assert!(
            entered_rx.recv_timeout(Duration::from_millis(100)).is_err(),
            "burst schedules produced more than one follow-up tick"
        );
    }

    struct NotifyTickCallback {
        entered_tx: mpsc::Sender<()>,
    }

    impl BatchedTickCallback for NotifyTickCallback {
        fn request_batched_tick(&self) {
            let _ = self.entered_tx.send(());
        }
    }

    #[test]
    fn shutdown_scheduler_drops_new_ticks_and_clears_the_debounce_flag() {
        // Internal scheduler test: post-shutdown scheduling behavior is not
        // observable through the public RN runtime API.
        let scheduler = scheduler_with_something_for_js();
        let (entered_tx, entered_rx) = mpsc::channel();
        scheduler.set_callback(Some(Box::new(NotifyTickCallback { entered_tx })));

        scheduler.shutdown();
        scheduler.schedule_batched_tick();

        assert!(
            entered_rx.recv_timeout(Duration::from_millis(100)).is_err(),
            "tick callback ran after shutdown"
        );
        assert!(
            !scheduler.scheduled.load(Ordering::SeqCst),
            "shutdown left the tick debounce flag wedged"
        );
    }

    /// Whoever shuts the scheduler down and then waits for the worker must not be
    /// passed by a job that was queued and had not started: it would take hold of the
    /// engine after the wait, and be the one left holding it.
    #[test]
    fn a_job_that_starts_after_shutdown_takes_no_hold_of_the_engine() {
        let scheduler = RnScheduler::default();
        let hold = scheduler.take_hold();
        assert!(
            hold.is_some(),
            "the worker could not take hold on a live scheduler"
        );
        drop(hold);
        scheduler.shutdown();
        assert!(
            scheduler.take_hold().is_none(),
            "the worker took hold of the engine on a scheduler that was shut down"
        );
        // And nobody is left waiting for it.
        scheduler.wait_for_the_worker();
    }
}

// ============================================================================
// Module-level utilities
// ============================================================================

/// Set the level of the engine's native `tracing` logs, effective immediately.
///
/// Engine logs are off until this is called and go to Apple unified logging
/// (subsystem `io.linsa.jazz`, one category per tracing target) — never through
/// the RN bridge, because these are hot-path lines. They are therefore invisible
/// in the Metro/Expo terminal; read them with, on the simulator,
/// `xcrun simctl spawn booted log stream --predicate 'subsystem ==
/// "io.linsa.jazz"' --style compact` (add `--level debug` for tracing
/// debug/trace lines), and on a device `xcrun devicectl device console` or
/// Console.app filtered on the subsystem.
///
/// `spec` is a `tracing` `EnvFilter` directive string, so it accepts both a bare
/// level — `"off"`, `"error"`, `"warn"`, `"info"`, `"debug"`, `"trace"` — and
/// per-target filtering, e.g. `"off,jazz::settle_cost=info"` to raise the settle
/// cost counters alone. An empty string means `"off"`.
#[uniffi::export]
pub fn set_engine_log_level(spec: String) -> Result<(), JazzRnError> {
    with_panic_boundary("set_engine_log_level", || {
        engine_log::set_level(&spec).map_err(|message| JazzRnError::Internal { message })
    })
}

/// Mint a local-first JWT from a base64url-encoded 32-byte seed.
///
/// Returns a signed JWT that can be used as a bearer token for local-first auth.
/// `audience` should be the app ID (UUID) or a human-readable app name.
/// `ttl_seconds` controls token lifetime (e.g. 3600 for one hour).
#[uniffi::export]
pub fn mint_local_first_token(
    seed_b64: String,
    audience: String,
    ttl_seconds: i64,
) -> Result<String, JazzRnError> {
    mint_token(
        seed_b64,
        audience,
        ttl_seconds,
        jazz_tools::identity::LOCAL_FIRST_ISSUER,
    )
}

/// Mint an anonymous JWT from a base64url-encoded 32-byte seed.
///
/// Returns a signed JWT that can be used as a bearer token for anonymous auth.
/// `audience` should be the app ID (UUID) or a human-readable app name.
/// `ttl_seconds` controls token lifetime (e.g. 3600 for one hour).
#[uniffi::export]
pub fn mint_anonymous_token(
    seed_b64: String,
    audience: String,
    ttl_seconds: i64,
) -> Result<String, JazzRnError> {
    mint_token(
        seed_b64,
        audience,
        ttl_seconds,
        jazz_tools::identity::ANONYMOUS_ISSUER,
    )
}

fn mint_token(
    seed_b64: String,
    audience: String,
    ttl_seconds: i64,
    issuer: &'static str,
) -> Result<String, JazzRnError> {
    use base64::Engine;
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(&seed_b64)
        .map_err(|e| JazzRnError::Internal {
            message: format!("invalid base64 seed: {e}"),
        })?;
    let seed: [u8; 32] = bytes.try_into().map_err(|_| JazzRnError::Internal {
        message: "seed must be exactly 32 bytes".to_string(),
    })?;
    jazz_tools::identity::mint_jazz_self_signed_token(&seed, issuer, &audience, ttl_seconds as u64)
        .map_err(|e| JazzRnError::Internal { message: e })
}

#[cfg(test)]
#[path = "tick_thread_tests.rs"]
mod tick_thread_tests;

/// The generated bindings (`src/generated`, `cpp/`) are checked in and shipped as they
/// are. As they load they compare a checksum of every exported function, method and
/// callback with the one this library reports, and refuse to load on a mismatch: the app
/// then starts to a black screen. A checksum covers the item's signature and its doc
/// comment, so rewording a doc comment on an exported item is an interface change.
#[cfg(test)]
mod binding_checksum_tests {
    macro_rules! exported {
        ($($name:ident,)*) => {
            extern "C" {
                $(fn $name() -> u16;)*
            }
            fn reported() -> Vec<(&'static str, u16)> {
                // SAFETY: each is a function of this crate, generated by uniffi, that
                // takes nothing and returns a constant.
                vec![$((stringify!($name), unsafe { $name() }),)*]
            }
        };
    }

    exported! {
        uniffi_jazz_rn_checksum_constructor_rnruntime_new,
        uniffi_jazz_rn_checksum_func_mint_anonymous_token,
        uniffi_jazz_rn_checksum_func_mint_local_first_token,
        uniffi_jazz_rn_checksum_func_set_engine_log_level,
        uniffi_jazz_rn_checksum_method_authfailurecallback_on_failure,
        uniffi_jazz_rn_checksum_method_batchedtickcallback_request_batched_tick,
        uniffi_jazz_rn_checksum_method_mutationerrorcallback_on_error,
        uniffi_jazz_rn_checksum_method_rnruntime_batched_tick,
        uniffi_jazz_rn_checksum_method_rnruntime_begin_batch,
        uniffi_jazz_rn_checksum_method_rnruntime_close,
        uniffi_jazz_rn_checksum_method_rnruntime_commit_batch,
        uniffi_jazz_rn_checksum_method_rnruntime_connect,
        uniffi_jazz_rn_checksum_method_rnruntime_create_subscription,
        uniffi_jazz_rn_checksum_method_rnruntime_delete,
        uniffi_jazz_rn_checksum_method_rnruntime_disconnect,
        uniffi_jazz_rn_checksum_method_rnruntime_execute_subscription,
        uniffi_jazz_rn_checksum_method_rnruntime_get_schema_hash,
        uniffi_jazz_rn_checksum_method_rnruntime_insert,
        uniffi_jazz_rn_checksum_method_rnruntime_insert_with_blobs,
        uniffi_jazz_rn_checksum_method_rnruntime_on_auth_failure,
        uniffi_jazz_rn_checksum_method_rnruntime_on_batched_tick_needed,
        uniffi_jazz_rn_checksum_method_rnruntime_on_mutation_error,
        uniffi_jazz_rn_checksum_method_rnruntime_query,
        uniffi_jazz_rn_checksum_method_rnruntime_restore,
        uniffi_jazz_rn_checksum_method_rnruntime_restore_with_blobs,
        uniffi_jazz_rn_checksum_method_rnruntime_rollback_batch,
        uniffi_jazz_rn_checksum_method_rnruntime_unsubscribe,
        uniffi_jazz_rn_checksum_method_rnruntime_update,
        uniffi_jazz_rn_checksum_method_rnruntime_update_auth,
        uniffi_jazz_rn_checksum_method_rnruntime_update_with_blobs,
        uniffi_jazz_rn_checksum_method_rnruntime_upsert,
        uniffi_jazz_rn_checksum_method_rnruntime_upsert_with_blobs,
        uniffi_jazz_rn_checksum_method_rnruntime_wait_for_batch,
        uniffi_jazz_rn_checksum_method_subscriptioncallback_on_update,
    }

    #[test]
    fn the_library_reports_the_checksums_the_checked_in_bindings_expect() {
        let bindings = include_str!("../../src/generated/jazz_rn.ts");
        let mut expected = Vec::new();
        let mut rest = bindings;
        while let Some(at) = rest.find("ubrn_uniffi_jazz_rn_checksum_") {
            rest = &rest[at + "ubrn_".len()..];
            let name = &rest[..rest.find('(').unwrap()];
            let Some(after) = rest[name.len()..].strip_prefix("() !==") else {
                continue;
            };
            let value: String = after
                .trim_start()
                .chars()
                .take_while(char::is_ascii_digit)
                .collect();
            expected.push((name, value.parse::<u16>().unwrap()));
        }
        assert!(expected.len() > 20, "the bindings were not read");

        let reported = reported();
        for (name, value) in &expected {
            let found = reported
                .iter()
                .find(|(of, _)| of == name)
                .unwrap_or_else(|| {
                    panic!("{name} is checked by the bindings and not listed in this test")
                });
            assert_eq!(
                found.1, *value,
                "{name}: the bindings would refuse to load this library"
            );
        }
        assert_eq!(reported.len(), expected.len());
    }
}
