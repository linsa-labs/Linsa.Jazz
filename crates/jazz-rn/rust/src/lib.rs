// jazz-rn (Rust) — UniFFI surface for React Native.
//
// Note: This crate intentionally uses UniFFI proc-macros (no UDL). The RN bindings
// generator runs UniFFI in "library mode", reading this crate's metadata.
uniffi::setup_scaffolding!();

use std::collections::HashMap;
use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;

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

fn make_subscription_callback(
    callback: Box<dyn SubscriptionCallback>,
) -> impl Fn(SubscriptionDelta) + Send + 'static {
    move |delta: SubscriptionDelta| {
        let (payload, blobs) = subscription_delta_with_blob_sidecar(&delta);
        if let Ok(json) = serde_json::to_string(&payload) {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                callback.on_update(json, blobs);
            }));
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
    Tick,
    DeliverMutationErrors,
}

type SharedTickCallback = Arc<Mutex<Option<Arc<dyn BatchedTickCallback>>>>;
type SharedCoreRef = Arc<Mutex<Option<Weak<Mutex<RnCoreType>>>>>;

#[derive(Clone, Default)]
struct RnScheduler {
    scheduled: Arc<AtomicBool>,
    mutation_error_delivery_scheduled: Arc<AtomicBool>,
    core_ref: SharedCoreRef,
    callback: SharedTickCallback,
    // The worker thread is detached: it exits once the sender is dropped and
    // its queue drains. It must never be joined — see `shutdown` below.
    worker: Arc<Mutex<Option<std::sync::mpsc::Sender<SchedulerJob>>>>,
    shutdown: Arc<AtomicBool>,
}

impl RnScheduler {
    fn set_core_ref(&self, core_ref: Weak<Mutex<RnCoreType>>) {
        if let Ok(mut slot) = self.core_ref.lock() {
            *slot = Some(core_ref);
        }
    }

    fn set_callback(&self, cb: Option<Box<dyn BatchedTickCallback>>) {
        if let Ok(mut slot) = self.callback.lock() {
            *slot = cb.map(Arc::from);
        }
    }

    fn clear_scheduled(&self) {
        self.scheduled.store(false, Ordering::SeqCst);
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
            *slot = Self::spawn_worker(
                Arc::clone(&self.scheduled),
                Arc::clone(&self.mutation_error_delivery_scheduled),
                Arc::clone(&self.core_ref),
                Arc::clone(&self.callback),
            );
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
            SchedulerJob::DeliverMutationErrors => self
                .mutation_error_delivery_scheduled
                .store(false, Ordering::SeqCst),
        }
    }

    fn spawn_worker(
        scheduled: Arc<AtomicBool>,
        mutation_error_delivery_scheduled: Arc<AtomicBool>,
        core_ref: SharedCoreRef,
        callback: SharedTickCallback,
    ) -> Option<std::sync::mpsc::Sender<SchedulerJob>> {
        let (sender, receiver) = std::sync::mpsc::channel::<SchedulerJob>();
        let spawned = std::thread::Builder::new()
            .name("jazz-rn-scheduler".into())
            .spawn(move || {
                while let Ok(job) = receiver.recv() {
                    std::thread::sleep(Duration::from_millis(1));
                    match job {
                        SchedulerJob::Tick => {
                            scheduled.store(false, Ordering::SeqCst);
                            Self::request_batched_tick(&callback);
                        }
                        SchedulerJob::DeliverMutationErrors => {
                            mutation_error_delivery_scheduled.store(false, Ordering::SeqCst);
                            Self::deliver_mutation_errors(&core_ref);
                        }
                    }
                }
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

    fn deliver_mutation_errors(core_ref: &SharedCoreRef) {
        let core_ref = core_ref.lock().ok().and_then(|slot| slot.clone());

        if let Some(core) = core_ref.and_then(|core_ref| core_ref.upgrade()) {
            if let Err(error) = deliver_pending_mutation_errors(&core) {
                eprintln!("jazz-rn: deliver pending mutation errors: {error:?}");
            }
        }
    }

    fn shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
        self.scheduled.store(false, Ordering::SeqCst);
        self.mutation_error_delivery_scheduled
            .store(false, Ordering::SeqCst);

        // Dropping the sender lets the worker drain its queue (no-ops once
        // the callbacks are cleared) and exit on its own. Never join it: JS
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

        // Defer firing the JS callback through the scheduler worker so we do
        // not synchronously re-enter `RnRuntime::batched_tick` from inside
        // `core.batched_tick()`. Without this delay,
        // `cb.request_batched_tick()` enqueues a JS microtask that runs
        // another `batched_tick` immediately, hot-looping the JS thread and
        // starving `setInterval`/render. The 1ms sleep also coalesces bursts
        // of schedule calls within a tick into a single follow-up callback.
        // This mirrors `schedule_mutation_error_delivery` below and
        // `NapiScheduler::schedule_batched_tick` in `jazz-napi`.
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
// RnRuntime
// ============================================================================

type RnCoreType = RuntimeCore<SqliteStorage, RnScheduler>;

fn deliver_pending_mutation_errors(core: &Arc<Mutex<RnCoreType>>) -> Result<(), JazzRnError> {
    let delivery = {
        let mut core = core.lock().map_err(|_| JazzRnError::Internal {
            message: "lock poisoned".into(),
        })?;
        core.pending_mutation_error_delivery()
    };

    let Some((callback, events)) = delivery else {
        return Ok(());
    };

    for event in events {
        // The callback re-enters JS; a panic (e.g. a JS exception) must not
        // kill the scheduler worker or skip the remaining events.
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| callback(&event)));
    }

    Ok(())
}

#[derive(uniffi::Object)]
pub struct RnRuntime {
    core: Arc<Mutex<RnCoreType>>,
}

#[uniffi::export]
impl RnRuntime {
    #[uniffi::constructor]
    pub fn new(
        schema_json: String,
        app_id: String,
        jazz_env: String,
        user_branch: String,
        tier: Option<String>,
        data_path: Option<String>,
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
            let storage =
                SqliteStorage::open(&resolved_data_path).map_err(|e| JazzRnError::Runtime {
                    message: format!(
                        "Failed to open SQLite storage at '{}': {:?}",
                        resolved_data_path, e
                    ),
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

            let scheduler = RnScheduler::default();

            let mut core = RuntimeCore::new(schema_manager, storage, scheduler);
            core.persist_schema();
            let core = Arc::new(Mutex::new(core));
            {
                let core_guard = core.lock().map_err(|_| JazzRnError::Internal {
                    message: "lock poisoned".into(),
                })?;
                core_guard.scheduler().set_core_ref(Arc::downgrade(&core));
            }

            Ok(Arc::new(Self { core }))
        })
    }

    /// Register a JS callback that schedules `batched_tick()` calls.
    pub fn on_batched_tick_needed(
        &self,
        callback: Option<Box<dyn BatchedTickCallback>>,
    ) -> Result<(), JazzRnError> {
        with_panic_boundary("on_batched_tick_needed", || {
            let mut core = self.core.lock().map_err(|_| JazzRnError::Internal {
                message: "lock poisoned".into(),
            })?;
            core.scheduler_mut().set_callback(callback);
            Ok(())
        })
    }

    /// Run a batched tick. JS should call this when asked via `on_batched_tick_needed`.
    pub fn batched_tick(&self) -> Result<(), JazzRnError> {
        with_panic_boundary("batched_tick", || {
            {
                let mut core = self.core.lock().map_err(|_| JazzRnError::Internal {
                    message: "lock poisoned".into(),
                })?;
                core.scheduler_mut().clear_scheduled();
                core.batched_tick();
                if let Some(error) = core.take_storage_flush_error() {
                    return Err(runtime_err(format!(
                        "storage flush or read-pass commit failed: {error}"
                    )));
                }
                // v18 item 4 (diff r20 SF6): the carrier reports ONCE. A store that lost
                // writes is dead until it is reopened, so every later tick must say so too
                // — the rule `TokioRuntime::flush` follows. Without this the RN host, the
                // mandate's primary target, gets one error and then `Ok(())` forever on a
                // store that persists nothing.
                //
                // The host must REPORT AND KEEP TICKING (diff r21 SF2): `clear_scheduled()`
                // above runs before the tick, so a host that stops calling `batched_tick` on
                // this error leaves `RnScheduler.scheduled` set forever and every later
                // `schedule_batched_tick()` is deduped away — sync included. The error is a
                // durability report, not a stop signal.
                if let Some(error) = core.lost_writes() {
                    return Err(runtime_err(format!(
                        "storage flush or read-pass commit failed: {error}"
                    )));
                }
            }
            Ok(())
        })
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
        with_panic_boundary("insert", || {
            let named_values = convert_insert_values(&values_json)?;
            let write_context = parse_write_context(write_context_json)?;
            let object_id = parse_external_object_id(object_id.as_deref())
                .map_err(|message| JazzRnError::InvalidUuid { message })?;
            let mut core = self.core.lock().map_err(|_| JazzRnError::Internal {
                message: "lock poisoned".into(),
            })?;
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
        with_panic_boundary("insert_with_blobs", || {
            let named_values = decode_ffi_json_record_with_blobs(&values_json, &blobs)?;
            let write_context = parse_write_context(write_context_json)?;
            let object_id = parse_external_object_id(object_id.as_deref())
                .map_err(|message| JazzRnError::InvalidUuid { message })?;
            let mut core = self.core.lock().map_err(|_| JazzRnError::Internal {
                message: "lock poisoned".into(),
            })?;
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
        with_panic_boundary("restore_with_blobs", || {
            let uuid = uuid::Uuid::parse_str(&object_id).map_err(|e| JazzRnError::InvalidUuid {
                message: e.to_string(),
            })?;
            let oid = ObjectId::from_uuid(uuid);
            let named_values = decode_ffi_json_record_with_blobs(&values_json, &blobs)?;
            let write_context = parse_write_context(write_context_json)?;
            let mut core = self.core.lock().map_err(|_| JazzRnError::Internal {
                message: "lock poisoned".into(),
            })?;
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
        with_panic_boundary("update_with_blobs", || {
            let uuid = uuid::Uuid::parse_str(&object_id).map_err(|e| JazzRnError::InvalidUuid {
                message: e.to_string(),
            })?;
            let oid = ObjectId::from_uuid(uuid);
            let updates: Vec<(String, Value)> =
                decode_ffi_json_record_with_blobs(&values_json, &blobs)?
                    .into_iter()
                    .collect();
            let write_context = parse_write_context(write_context_json)?;
            let mut core = self.core.lock().map_err(|_| JazzRnError::Internal {
                message: "lock poisoned".into(),
            })?;
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
        with_panic_boundary("upsert_with_blobs", || {
            let uuid = uuid::Uuid::parse_str(&object_id).map_err(|e| JazzRnError::InvalidUuid {
                message: e.to_string(),
            })?;
            let oid = ObjectId::from_uuid(uuid);
            let named_values = decode_ffi_json_record_with_blobs(&values_json, &blobs)?;
            let write_context = parse_write_context(write_context_json)?;
            let mut core = self.core.lock().map_err(|_| JazzRnError::Internal {
                message: "lock poisoned".into(),
            })?;
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
        with_panic_boundary("restore", || {
            let uuid = uuid::Uuid::parse_str(&object_id).map_err(|e| JazzRnError::InvalidUuid {
                message: e.to_string(),
            })?;
            let oid = ObjectId::from_uuid(uuid);
            let named_values = convert_insert_values(&values_json)?;
            let write_context = parse_write_context(write_context_json)?;
            let mut core = self.core.lock().map_err(|_| JazzRnError::Internal {
                message: "lock poisoned".into(),
            })?;
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
        with_panic_boundary("update", || {
            let uuid = uuid::Uuid::parse_str(&object_id).map_err(|e| JazzRnError::InvalidUuid {
                message: e.to_string(),
            })?;
            let oid = ObjectId::from_uuid(uuid);
            let updates = convert_updates(&values_json)?;
            let write_context = parse_write_context(write_context_json)?;
            let mut core = self.core.lock().map_err(|_| JazzRnError::Internal {
                message: "lock poisoned".into(),
            })?;
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
        with_panic_boundary("upsert", || {
            let uuid = uuid::Uuid::parse_str(&object_id).map_err(|e| JazzRnError::InvalidUuid {
                message: e.to_string(),
            })?;
            let oid = ObjectId::from_uuid(uuid);
            let named_values = convert_insert_values(&values_json)?;
            let write_context = parse_write_context(write_context_json)?;
            let mut core = self.core.lock().map_err(|_| JazzRnError::Internal {
                message: "lock poisoned".into(),
            })?;
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
            let mut core = self.core.lock().map_err(|_| JazzRnError::Internal {
                message: "lock poisoned".into(),
            })?;
            Ok(core.begin_batch(batch_mode).to_string())
        })
    }

    pub fn rollback_batch(&self, batch_id: String) -> Result<bool, JazzRnError> {
        with_panic_boundary("rollback_batch", || {
            let batch_id = parse_batch_id_input(&batch_id)
                .map_err(|message| JazzRnError::InvalidUuid { message })?;
            let mut core = self.core.lock().map_err(|_| JazzRnError::Internal {
                message: "lock poisoned".into(),
            })?;
            core.rollback_batch(batch_id).map_err(runtime_err)
        })
    }

    #[uniffi::method(name = "delete")]
    pub fn delete_row(
        &self,
        object_id: String,
        write_context_json: Option<String>,
    ) -> Result<String, JazzRnError> {
        with_panic_boundary("delete", || {
            let uuid = uuid::Uuid::parse_str(&object_id).map_err(|e| JazzRnError::InvalidUuid {
                message: e.to_string(),
            })?;
            let oid = ObjectId::from_uuid(uuid);
            let write_context = parse_write_context(write_context_json)?;
            let mut core = self.core.lock().map_err(|_| JazzRnError::Internal {
                message: "lock poisoned".into(),
            })?;
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
        with_async_panic_boundary("wait_for_batch", || async move {
            let batch_id = parse_batch_id_input(&batch_id)
                .map_err(|message| JazzRnError::InvalidUuid { message })?;
            let tier = parse_tier(&tier)?;
            let receiver = {
                let mut core = self.core.lock().map_err(|_| JazzRnError::Internal {
                    message: "lock poisoned".into(),
                })?;
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
        })
        .await
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
        with_async_panic_boundary("query", || async move {
            let query = parse_query(&query_json)?;
            let session = parse_session(session_json)?;
            let (durability, propagation, transaction_batch_id, _timeout_ms) =
                parse_read_durability_options(tier.as_deref(), options_json.as_deref())
                    .map_err(|message| JazzRnError::InvalidJson { message })?;

            let fut = {
                let mut core = self.core.lock().map_err(|_| JazzRnError::Internal {
                    message: "lock poisoned".into(),
                })?;
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
        })
        .await
    }

    // =========================================================================
    // Subscriptions
    // =========================================================================

    pub fn unsubscribe(&self, handle: u64) -> Result<(), JazzRnError> {
        with_panic_boundary("unsubscribe", || {
            let mut core = self.core.lock().map_err(|_| JazzRnError::Internal {
                message: "lock poisoned".into(),
            })?;
            core.unsubscribe(SubscriptionHandle(handle));
            Ok(())
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

            let mut core = self.core.lock().map_err(|_| JazzRnError::Internal {
                message: "lock poisoned".into(),
            })?;

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
        with_panic_boundary("execute_subscription", || {
            let mut core = self.core.lock().map_err(|_| JazzRnError::Internal {
                message: "lock poisoned".into(),
            })?;
            let callback = make_subscription_callback(callback);

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
            let core = self.core.lock().map_err(|_| JazzRnError::Internal {
                message: "lock poisoned".into(),
            })?;
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
                .lock()
                .map_err(|_| JazzRnError::Internal {
                    message: "lock poisoned".into(),
                })?
                .set_mutation_error_callback(Some(callback));
            Ok(())
        })
    }

    pub fn commit_batch(&self, batch_id: String) -> Result<(), JazzRnError> {
        with_panic_boundary("commit_batch", || {
            let batch_id = parse_batch_id_input(&batch_id)
                .map_err(|message| JazzRnError::InvalidUuid { message })?;
            let mut core = self.core.lock().map_err(|_| JazzRnError::Internal {
                message: "lock poisoned".into(),
            })?;
            core.commit_batch(batch_id).map_err(runtime_err)
        })
    }

    /// Flush and close the underlying storage, releasing filesystem locks.
    pub fn close(&self) -> Result<(), JazzRnError> {
        with_panic_boundary("close", || {
            let mut core = self.core.lock().map_err(|_| JazzRnError::Internal {
                message: "lock poisoned".into(),
            })?;
            core.scheduler_mut().set_callback(None);
            core.set_mutation_error_callback(None);
            core.scheduler().shutdown();
            let flush_result = core.flush_storage();
            let flush_wal_result = core.flush_wal();
            let close_result = core.storage().close();

            flush_result.map_err(runtime_err)?;
            flush_wal_result.map_err(runtime_err)?;
            close_result.map_err(runtime_err)
        })
    }

    /// Connect to a Jazz server over WebSocket.
    ///
    /// Parses `auth_json` into `AuthConfig`, wires a `TransportManager` into
    /// `RuntimeCore`, and spawns the manager loop on a dedicated Tokio thread.
    pub fn connect(&self, url: String, auth_json: String) -> Result<(), JazzRnError> {
        with_panic_boundary("connect", || {
            let auth: jazz_tools::transport_manager::AuthConfig =
                serde_json::from_str(&auth_json).map_err(json_err)?;
            let scheduler = self
                .core
                .lock()
                .map_err(|_| JazzRnError::Internal {
                    message: "lock poisoned".into(),
                })?
                .scheduler()
                .clone();
            let tick = RnTickNotifier { scheduler };
            let manager = {
                let mut core = self.core.lock().map_err(|_| JazzRnError::Internal {
                    message: "lock poisoned".into(),
                })?;
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
            let mut core = self.core.lock().map_err(|_| JazzRnError::Internal {
                message: "lock poisoned".into(),
            })?;
            core.set_auth_failure_callback(move |reason| {
                callback.on_failure(reason);
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

    #[test]
    fn scheduler_coalesces_bursts_on_one_worker_for_follow_up_ticks() {
        // This is intentionally an internal scheduler test: constant worker
        // count and callback serialization are not observable through the
        // public RN runtime API.
        let scheduler = RnScheduler::default();
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
        let scheduler = RnScheduler::default();
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
