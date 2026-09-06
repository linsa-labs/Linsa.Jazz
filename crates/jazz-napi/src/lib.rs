//! jazz-napi — Native Node.js bindings for Jazz.
//!
//! Provides `NapiRuntime` wrapping `RuntimeCore<SqliteStorage>` via napi-rs.
//! Exposed as the `jazz-napi` npm package for server-side TypeScript apps.
//!
//! # Architecture
//!
//! - `SqliteStorage` provides persistent on-disk storage
//! - `NapiScheduler` implements `Scheduler` using `ThreadsafeFunction` to schedule
//!   `batched_tick()` on the Node.js event loop (debounced)
//! - `NapiRuntime` wraps `Arc<Mutex<RuntimeCore<...>>>`
//! - Server sync uses the Rust-owned WebSocket transport via `connect()`
//!
//! # Allocator
//!
//! This crate uses `mimalloc-safe` (napi-rs–maintained mimalloc fork) as Rust's
//! `#[global_allocator]`. It does NOT override the host process's `malloc`/`free` —
//! Node.js / V8 keep their own allocator. The two coexist safely as long as
//! memory crosses the FFI boundary **by copy**, which is what napi-rs does today
//! for Vec/String/Buffer returns.
//!
//! Footgun: never `Vec::leak` / `Box::into_raw` an allocation across FFI and let
//! the host call `free()` on it — that mixes allocators and corrupts the heap.
//! If a future zero-copy shim is added, hand the host a Rust-defined finalizer
//! callback that frees through mimalloc instead.

#[global_allocator]
static GLOBAL: mimalloc_safe::MiMalloc = mimalloc_safe::MiMalloc;

use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi_derive::napi;
use serde::Deserialize;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::Duration;

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use jazz_tools::binding_support::{
    parse_batch_id_input, parse_batch_mode_input, parse_durability_tier as parse_binding_tier,
    parse_external_object_id, parse_query_input, parse_read_durability_options,
    parse_runtime_schema_input, parse_session_input, parse_write_context_input,
    serialize_mutation_error_event,
};
use jazz_tools::identity;
use jazz_tools::middleware::AuthConfig;
use jazz_tools::object::ObjectId;
use jazz_tools::query_manager::query::Query;
use jazz_tools::query_manager::session::{Session, WriteContext};
use jazz_tools::query_manager::types::{Schema, SchemaHash, Value};
use jazz_tools::runtime_core::{
    MutationErrorCallback, ReadDurabilityOptions, RuntimeCore, Scheduler, SubscriptionDelta,
    SubscriptionHandle,
};
use jazz_tools::schema_manager::{AppId, SchemaManager, rehydrate_schema_manager_from_catalogue};
use jazz_tools::server::{
    JazzServer as CoreJazzServer, ServerBuilder, ServerDataDir, StorageBackend,
    TestJwtIssuer as JazzTestJwtIssuer, TestJwtOptions,
};
use jazz_tools::storage::{MemoryStorage, SqliteStorage, Storage};
use jazz_tools::sync_manager::QueryPropagation;
use jazz_tools::sync_manager::{DurabilityTier, SyncManager};

fn convert_updates(values: HashMap<String, Value>) -> Vec<(String, Value)> {
    values.into_iter().collect()
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", content = "value")]
enum FfiValue {
    Integer(i32),
    BigInt(i64),
    Double(f64),
    Boolean(bool),
    Text(String),
    Timestamp(u64),
    Uuid(ObjectId),
    Bytea(#[serde(with = "serde_bytes")] Vec<u8>),
    Array(Vec<FfiValue>),
    Row(FfiRow),
    Null,
}

#[derive(Debug, Clone, Deserialize)]
struct FfiRow {
    #[serde(default)]
    id: Option<ObjectId>,
    values: Vec<FfiValue>,
}

impl From<FfiValue> for Value {
    fn from(value: FfiValue) -> Self {
        match value {
            FfiValue::Integer(value) => Value::Integer(value),
            FfiValue::BigInt(value) => Value::BigInt(value),
            FfiValue::Double(value) => Value::Double(value),
            FfiValue::Boolean(value) => Value::Boolean(value),
            FfiValue::Text(value) => Value::Text(value),
            FfiValue::Timestamp(value) => Value::Timestamp(value),
            FfiValue::Uuid(value) => Value::Uuid(value),
            FfiValue::Bytea(value) => Value::Bytea(value),
            FfiValue::Array(values) => Value::Array(values.into_iter().map(Value::from).collect()),
            FfiValue::Row(row) => Value::Row {
                id: row.id,
                values: row.values.into_iter().map(Value::from).collect(),
            },
            FfiValue::Null => Value::Null,
        }
    }
}

pub struct FfiRecordArg(HashMap<String, Value>);

impl TypeName for FfiRecordArg {
    fn type_name() -> &'static str {
        "Record<string, unknown>"
    }

    fn value_type() -> ValueType {
        ValueType::Object
    }
}

impl FromNapiValue for FfiRecordArg {
    unsafe fn from_napi_value(
        env: napi::sys::napi_env,
        napi_val: napi::sys::napi_value,
    ) -> Result<Self> {
        let env = Env::from_raw(env);
        let unknown = unsafe { Unknown::from_napi_value(env.raw(), napi_val)? };
        let values = env
            .from_js_value::<HashMap<String, FfiValue>, _>(unknown)
            .map_err(|error| napi::Error::from_reason(format!("Invalid values: {}", error)))?;
        Ok(Self(
            values
                .into_iter()
                .map(|(key, value)| (key, Value::from(value)))
                .collect(),
        ))
    }
}

/// Hand any convertible value to JS as an opaque `Unknown`, so the branches of
/// [`value_to_js`] can return one type without each knowing the next.
fn into_unknown<'env, T: ToNapiValue>(env: &'env Env, value: T) -> napi::Result<Unknown<'env>> {
    unsafe {
        let raw = ToNapiValue::to_napi_value(env.raw(), value)?;
        Unknown::from_napi_value(env.raw(), raw)
    }
}

/// A row value on its way to JS, with blobs handed over as bytes.
///
/// The obvious route — `serde_json::json!({"values": values})` — costs one napi
/// call per BYTE of a `Bytea` column. `ValueHuman::Bytea` is a plain `Vec<u8>`,
/// so serde renders a megabyte as an array of a million Numbers and napi then
/// sets each one into a JS array individually; the JS side walks it back per
/// byte. Measured on this binding with no server and no network in the loop:
/// ~10 MiB/s, with 97% of the wall time spent as user CPU on the JS thread.
///
/// Only `Bytea` is special-cased here. Everything else still goes through
/// `serde_json`, so the `{type, value}` shape the TS layer parses stays defined
/// in exactly one place (`ValueHuman`) rather than being restated — the two
/// composite variants below reproduce their wrapper only, and recurse for what
/// is inside.
fn value_to_js<'env>(env: &'env Env, value: &Value) -> napi::Result<Unknown<'env>> {
    match value {
        Value::Bytea(bytes) => {
            let mut tagged = Object::new(env)?;
            tagged.set("type", "Bytea")?;
            // Zero-copy hand-off: napi-rs attaches a finalizer that frees the Vec
            // back through Rust's allocator, which is what this crate's header
            // asks for, and falls back to a copy on runtimes that refuse
            // external buffers.
            //
            // The `clone` is a known cost, deliberately left: it is one memcpy,
            // invisible against the 3 GiB/s this path now runs at, but it does
            // mean a blob exists twice for the length of the conversion — the
            // source stays alive in the result set while the copy is pinned by
            // the Buffer. That doubles PEAK memory in proportion to the size of
            // one query's results. It does not show up in this app, which reads
            // media in windows of four ~1 MiB parts; it would on a query that
            // returns a large result set. Removing it means a consuming
            // `value_into_js(env, value: Value)` so the Vec can be moved. Note
            // there is no honest test for it — the win is peak RSS, which flakes
            // as an assertion, so this is a comment rather than a gate.
            tagged.set("value", BufferSlice::from_data(env, bytes.clone())?)?;
            into_unknown(env, tagged)
        }
        Value::Array(items) => {
            let mut tagged = Object::new(env)?;
            tagged.set("type", "Array")?;
            tagged.set("value", values_to_js(env, items)?)?;
            into_unknown(env, tagged)
        }
        Value::Row { id, values } => {
            let mut inner = Object::new(env)?;
            // `RowHuman` skips a missing id rather than sending null, so match it.
            if let Some(id) = id {
                inner.set("id", id.uuid().to_string())?;
            }
            inner.set("values", values_to_js(env, values)?)?;
            let mut tagged = Object::new(env)?;
            tagged.set("type", "Row")?;
            tagged.set("value", inner)?;
            into_unknown(env, tagged)
        }
        scalar => {
            let json = serde_json::to_value(scalar).map_err(|error| {
                napi::Error::from_reason(format!("Failed to encode value: {error}"))
            })?;
            into_unknown(env, json)
        }
    }
}

fn values_to_js<'env>(env: &'env Env, values: &[Value]) -> napi::Result<Vec<Unknown<'env>>> {
    values.iter().map(|value| value_to_js(env, value)).collect()
}

/// One written row on its way back to JS.
///
/// `insert`/`restore` echo the row they just stored, blob included. The bytes
/// were supplied by JS microseconds earlier, so the echo looks redundant — but
/// `restore` can resurrect columns the caller never sent, so it carries real
/// information and is not simply removable. What it should not do is cost one
/// napi call per byte, which the `serde_json` route did.
///
/// (jazz-rn answers with a `BlobRef` instead. That is not a better design so
/// much as a necessary one there: its bridge ships a JSON *string*, so it has
/// no way to hand over bytes at all.)
pub struct WrittenRow {
    id: ObjectId,
    values: Vec<Value>,
    batch_id: String,
}

impl ToNapiValue for WrittenRow {
    unsafe fn to_napi_value(
        raw_env: napi::sys::napi_env,
        written: Self,
    ) -> napi::Result<napi::sys::napi_value> {
        let env = Env::from_raw(raw_env);
        let mut row = Object::new(&env)?;
        row.set("id", written.id.uuid().to_string())?;
        row.set("values", values_to_js(&env, &written.values)?)?;
        row.set("batchId", written.batch_id)?;
        unsafe { ToNapiValue::to_napi_value(raw_env, row) }
    }
}

/// Query results, converted on the JS thread by [`value_to_js`].
pub struct QueryRows(Vec<(ObjectId, Vec<Value>)>);

impl ToNapiValue for QueryRows {
    unsafe fn to_napi_value(
        raw_env: napi::sys::napi_env,
        rows: Self,
    ) -> napi::Result<napi::sys::napi_value> {
        let env = Env::from_raw(raw_env);
        let mut js_rows = Vec::with_capacity(rows.0.len());
        for (id, values) in rows.0.iter() {
            let mut row = Object::new(&env)?;
            row.set("id", id.uuid().to_string())?;
            row.set("values", values_to_js(&env, values)?)?;
            js_rows.push(into_unknown(&env, row)?);
        }
        unsafe { ToNapiValue::to_napi_value(raw_env, js_rows) }
    }
}

fn parse_node_durability_tiers(tier: Option<&str>) -> napi::Result<Vec<DurabilityTier>> {
    let Some(raw) = tier else {
        return Ok(Vec::new());
    };
    Ok(vec![parse_tier(raw)?])
}

fn parse_node_durability_tier(tier: Option<String>) -> napi::Result<Vec<DurabilityTier>> {
    parse_node_durability_tiers(tier.as_deref())
}

fn open_sqlite_storage(data_path: &str) -> napi::Result<SqliteStorage> {
    SqliteStorage::open(data_path)
        .map_err(|e| napi::Error::from_reason(format!("Failed to open storage: {:?}", e)))
}

// ============================================================================
fn parse_tier(tier: &str) -> napi::Result<DurabilityTier> {
    parse_binding_tier(tier).map_err(napi::Error::from_reason)
}

fn parse_query(json: &str) -> napi::Result<Query> {
    parse_query_input(json).map_err(napi::Error::from_reason)
}

fn parse_session_json(session_json: Option<String>) -> napi::Result<Option<Session>> {
    parse_session_input(session_json.as_deref())
        .map_err(|err| napi::Error::from_reason(format!("Invalid session JSON: {}", err)))
}

fn parse_write_context_json(
    write_context_json: Option<String>,
) -> napi::Result<Option<WriteContext>> {
    parse_write_context_input(write_context_json.as_deref())
        .map_err(|err| napi::Error::from_reason(format!("Invalid write context JSON: {}", err)))
}

fn parse_subscription_inputs(
    query_json: &str,
    session_json: Option<String>,
    tier: Option<String>,
    options_json: Option<String>,
) -> napi::Result<(
    Query,
    Option<Session>,
    ReadDurabilityOptions,
    QueryPropagation,
)> {
    let query = parse_query(query_json)?;
    let session = parse_session_json(session_json)?;
    let (durability, propagation, _transaction_batch_id, _timeout_ms) =
        parse_read_durability_options(tier.as_deref(), options_json.as_deref())
            .map_err(napi::Error::from_reason)?;
    Ok((query, session, durability, propagation))
}

/// One subscription delta on its way to JS.
///
/// Mirrors `subscription_delta_to_json` (`jazz_tools::query_manager::bindings`)
/// exactly — same `kind` numbering, same keys, `updated` carrying an optional
/// row — but decodes row values through [`value_to_js`] so a blob crosses as
/// bytes. This is the path the app actually reads media on: it subscribes to a
/// window of `file_parts` rather than querying, so fixing `query` alone left
/// every download on the per-byte route.
///
/// NOTE this fixes node only. `jazz-rn` has the same defect and cannot use this
/// fix: its bridge is uniffi with a string callback, so its delta goes out as
/// `serde_json::to_string` (`crates/jazz-rn/rust/src/lib.rs`) with the blob as
/// an array of numbers inside the JSON text, and the JS side then walks it byte
/// by byte. A phone is slow at downloads for that reason and stays slow after
/// this commit. Fixing it needs either base64/hex in the JSON or a
/// bytes-capable uniffi type — a different change, not a port of this one.
pub struct SubscriptionDeltaJs(SubscriptionDelta);

impl ToNapiValue for SubscriptionDeltaJs {
    unsafe fn to_napi_value(
        raw_env: napi::sys::napi_env,
        delta: Self,
    ) -> napi::Result<napi::sys::napi_value> {
        let env = Env::from_raw(raw_env);
        let delta = delta.0;
        let descriptor = &delta.descriptor;
        let row_to_js = |env: &Env, row: &jazz_tools::query_manager::types::Row| {
            let values =
                jazz_tools::row_format::decode_row(descriptor, &row.data).unwrap_or_default();
            let mut js_row = Object::new(env)?;
            js_row.set("id", row.id.uuid().to_string())?;
            js_row.set("values", values_to_js(env, &values)?)?;
            napi::Result::Ok(js_row)
        };

        let mut changes: Vec<Unknown> = Vec::new();
        for change in &delta.ordered_delta.removed {
            let mut js = Object::new(&env)?;
            js.set("kind", 1u32)?;
            js.set("id", change.id.uuid().to_string())?;
            js.set("index", change.index as u32)?;
            changes.push(into_unknown(&env, js)?);
        }
        for change in &delta.ordered_delta.updated {
            let mut js = Object::new(&env)?;
            js.set("kind", 2u32)?;
            js.set("id", change.id.uuid().to_string())?;
            js.set("index", change.new_index as u32)?;
            match change.row.as_ref() {
                Some(row) => js.set("row", row_to_js(&env, row)?)?,
                // `serde_json::json!` emitted an explicit null here, so keep it:
                // the TS side distinguishes "no row carried" from "key absent".
                None => js.set("row", Null)?,
            }
            changes.push(into_unknown(&env, js)?);
        }
        for change in &delta.ordered_delta.added {
            let mut js = Object::new(&env)?;
            js.set("kind", 0u32)?;
            js.set("id", change.id.uuid().to_string())?;
            js.set("index", change.index as u32)?;
            js.set("row", row_to_js(&env, &change.row)?)?;
            changes.push(into_unknown(&env, js)?);
        }

        unsafe { ToNapiValue::to_napi_value(raw_env, changes) }
    }
}

fn make_subscription_callback(
    tsfn: ThreadsafeFunction<SubscriptionDeltaJs>,
) -> impl Fn(SubscriptionDelta) + Send + 'static {
    move |delta: SubscriptionDelta| {
        tsfn.call(
            Ok(SubscriptionDeltaJs(delta)),
            ThreadsafeFunctionCallMode::NonBlocking,
        );
    }
}

// ============================================================================
// NapiScheduler
// ============================================================================

type NapiCoreType = RuntimeCore<Box<dyn Storage + Send>, NapiScheduler>;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct JazzServerStartOptions {
    app_id: String,
    port: Option<u16>,
    data_dir: Option<String>,
    in_memory: Option<bool>,
    jwks_url: Option<String>,
    backend_secret: String,
    admin_secret: String,
    upstream_url: Option<String>,
    allow_local_first_auth: Option<bool>,
    telemetry_collector_url: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TestJwtForUserOptions {
    expires_in_seconds: Option<u64>,
    issuer: Option<String>,
}

fn parse_jazz_server_start_options(options: JsonValue) -> napi::Result<JazzServerStartOptions> {
    serde_json::from_value(options)
        .map_err(|error| napi::Error::from_reason(format!("Invalid JazzServer options: {error}")))
}

static JAZZ_SERVER_OTEL_PROVIDER: OnceLock<opentelemetry_sdk::trace::SdkTracerProvider> =
    OnceLock::new();
static JAZZ_SERVER_TELEMETRY_INIT: OnceLock<()> = OnceLock::new();

fn init_jazz_server_telemetry(collector_url: Option<&str>) {
    let Some(collector_url) = collector_url else {
        return;
    };

    JAZZ_SERVER_TELEMETRY_INIT.get_or_init(|| {
        use tracing_subscriber::layer::SubscriberExt as _;

        let endpoint = jazz_tools::otel::normalize_otlp_traces_endpoint(collector_url);
        let provider =
            jazz_tools::otel::init_tracer_provider_with_endpoint("jazz-server", Some(&endpoint));
        let otel_layer = jazz_tools::otel::layer(&provider);
        let filter = tracing_subscriber::EnvFilter::from_default_env()
            .add_directive("jazz_tools=trace".parse().expect("valid tracing directive"))
            .add_directive("tower_http=debug".parse().expect("valid tracing directive"));

        if tracing::subscriber::set_global_default(
            tracing_subscriber::registry().with(filter).with(otel_layer),
        )
        .is_ok()
        {
            let _ = JAZZ_SERVER_OTEL_PROVIDER.set(provider);
        }
    });
}

/// Scheduler that schedules `batched_tick()` after a short background delay.
/// Debounced: only one tick is pending at a time.
type MutationErrorTsfn = ThreadsafeFunction<serde_json::Value>;

pub struct NapiScheduler {
    scheduled: Arc<AtomicBool>,
    mutation_error_delivery_scheduled: Arc<AtomicBool>,
    core_ref: Weak<Mutex<NapiCoreType>>,
}

impl NapiScheduler {
    fn new() -> Self {
        Self {
            scheduled: Arc::new(AtomicBool::new(false)),
            mutation_error_delivery_scheduled: Arc::new(AtomicBool::new(false)),
            core_ref: Weak::new(),
        }
    }

    fn set_core_ref(&mut self, core_ref: Weak<Mutex<NapiCoreType>>) {
        self.core_ref = core_ref;
    }
}

impl Scheduler for NapiScheduler {
    fn schedule_batched_tick(&self) {
        if !self.scheduled.swap(true, Ordering::SeqCst) {
            let scheduled = Arc::clone(&self.scheduled);
            let core_ref = self.core_ref.clone();
            std::thread::spawn(move || {
                // Give bursts of inbound websocket frames a chance to coalesce
                // before the runtime drains the queue.
                std::thread::sleep(Duration::from_millis(1));
                scheduled.store(false, Ordering::SeqCst);
                if let Some(core_arc) = core_ref.upgrade()
                    && let Ok(mut core) = core_arc.lock()
                {
                    core.batched_tick();
                }
            });
        }
    }

    fn schedule_mutation_error_delivery(&self) {
        if self
            .mutation_error_delivery_scheduled
            .swap(true, Ordering::SeqCst)
        {
            return;
        }

        let scheduled = Arc::clone(&self.mutation_error_delivery_scheduled);
        let core_ref = self.core_ref.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(1));
            scheduled.store(false, Ordering::SeqCst);
            if let Some(core_arc) = core_ref.upgrade() {
                deliver_pending_mutation_errors(&core_arc);
            }
        });
    }
}

fn deliver_pending_mutation_errors(core_arc: &Arc<Mutex<NapiCoreType>>) {
    let delivery = {
        let Ok(mut core) = core_arc.lock() else {
            return;
        };
        core.pending_mutation_error_delivery()
    };

    let Some((callback, events)) = delivery else {
        return;
    };

    for event in events {
        callback(&event);
    }
}

/// The directives the client engine should log under, or `None` for "stay silent".
///
/// Split out from the install below because the install is a process-global, once-only side
/// effect and this is the whole of the policy: opt in on a non-empty value, and treat an empty
/// or whitespace-only one as unset. Compose files set variables to `""` all the time, and a
/// client engine that started logging because of that would be a surprise nobody asked for.
fn napi_logging_directives(raw: Option<&str>) -> Option<&str> {
    let directives = raw?.trim();
    (!directives.is_empty()).then_some(directives)
}

/// Whether the client engine's logging has been installed. Process-global by nature.
static NAPI_RUNTIME_LOG_INIT: OnceLock<bool> = OnceLock::new();

/// v18 measurement instrument: give the CLIENT engine a log.
///
/// `init_jazz_server_telemetry` above is for an embedded `JazzServer` and returns immediately
/// unless a collector URL is passed; prod passes none. So `NapiRuntime` — the engine the
/// rpc-server does all of its reads and writes through — has never emitted a line: no
/// `settle_cost`, no span, not even a thread name, since its tick runs on an anonymous
/// `std::thread`. `archive/prodrepro/REPRO.md` records what that cost: the prod bottleneck had
/// to be localised to this engine from the OUTSIDE, by elimination, because there was nothing
/// to read from the inside, and the v18 counters (`autocommit_reads`,
/// `locator_ladder_recoveries`, `checkpoints`) are unreadable here for the same reason.
///
/// Deliberately narrow, and the variable is `JAZZ_NAPI_LOG` rather than `RUST_LOG` **because
/// diff r28 measured that `RUST_LOG` is not opt-in here**: `Linsa.Server/docker/compose.prod.yml`
/// sets `RUST_LOG: "${RPC_SERVER_RUST_LOG:-info}"` on the rpc-server — the very process that
/// embeds this engine — so keying on it would have turned on a ~30-field line per settle pass, in
/// production, on the day this tag landed, written synchronously to stderr from inside the pass
/// while it holds the whole-`RuntimeCore` mutex. A dedicated variable is opt-in in fact and not
/// only in the comment.
///
/// ANSI is off unconditionally. This is a machine-read log — `archive/prodrepro/after-report.py`
/// scrapes `field=value` — and the escape codes land BETWEEN the name and the `=`, so a coloured
/// line reports every counter as absent. Relying on `NO_COLOR` would not have helped: it is set
/// on jazz-sync in all three compose files and on none of the rpc-server blocks.
///
/// `set_global_default` fails harmlessly when `init_jazz_server_telemetry` already claimed the
/// slot, so the two never fight over it; the returned bool says which happened, and a failed
/// install warns rather than disappearing.
fn init_napi_runtime_logging() -> bool {
    *NAPI_RUNTIME_LOG_INIT.get_or_init(|| {
        use tracing_subscriber::layer::SubscriberExt as _;

        let raw = std::env::var("JAZZ_NAPI_LOG").ok();
        let Some(directives) = napi_logging_directives(raw.as_deref()) else {
            return false;
        };
        let installed = tracing::subscriber::set_global_default(
            tracing_subscriber::registry()
                .with(tracing_subscriber::EnvFilter::new(directives))
                .with(
                    tracing_subscriber::fmt::layer()
                        .with_writer(std::io::stderr)
                        .with_ansi(false),
                ),
        )
        .is_ok();
        if !installed {
            // Someone already owns the global subscriber — today that can only be
            // `init_jazz_server_telemetry`. Say so, because the alternative is an operator who
            // set `JAZZ_NAPI_LOG` and sees nothing, with no way to tell that from a build
            // without this at all.
            eprintln!(
                "jazz: JAZZ_NAPI_LOG is set but a tracing subscriber was already installed; \
                 the client engine will not log"
            );
        }
        installed
    })
}

fn build_napi_runtime(
    schema_json: String,
    app_id: String,
    jazz_env: String,
    user_branch: String,
    storage: Box<dyn Storage + Send>,
    tier: Option<String>,
) -> napi::Result<NapiRuntime> {
    // Before anything else, so that a failure while building the runtime is itself visible.
    init_napi_runtime_logging();

    // Parse schema
    let runtime_schema = parse_runtime_schema_input(&schema_json)
        .map_err(|e| napi::Error::from_reason(format!("Invalid schema JSON: {}", e)))?;
    let schema = runtime_schema.schema;
    let declared_schema = schema.clone();

    // Parse optional tier
    let node_tiers = parse_node_durability_tier(tier)?;

    // Create sync manager
    let mut sync_manager = SyncManager::new();
    if !node_tiers.is_empty() {
        sync_manager = sync_manager.with_durability_tiers(node_tiers);
    }

    // One binding, used by both the manager and the rehydrate below.
    // `rehydrate_schema_manager_from_catalogue` keeps only entries whose AppId
    // metadata equals this value, and a mismatch is silent — it returns Ok
    // having matched nothing. The derivation is deterministic, so a second call
    // would agree; binding it once is what keeps that true if either side's
    // fallback ever changes.
    let app_id_obj = AppId::from_string(&app_id).unwrap_or_else(|_| AppId::from_name(&app_id));

    // Create schema manager
    let mut schema_manager = SchemaManager::new_with_policy_mode(
        sync_manager,
        schema,
        app_id_obj,
        &jazz_env,
        &user_branch,
        if runtime_schema.loaded_policy_bundle {
            jazz_tools::query_manager::types::RowPolicyMode::Enforcing
        } else {
            jazz_tools::query_manager::types::RowPolicyMode::PermissiveLocal
        },
    )
    .map_err(|e| napi::Error::from_reason(format!("Failed to create SchemaManager: {:?}", e)))?;

    // Load the schema generations, permissions bundle and lenses this store has
    // already persisted, BEFORE the runtime is built over it.
    //
    // The branch universe every read is scoped to comes from the manager's
    // `live_schemas` (`SchemaContext::all_branch_names`), never from the store,
    // and `RuntimeCore::new` does not read the catalogue for anyone — closing
    // that gap is each binding's own job (jazz-rn/rust/src/lib.rs:734,
    // jazz-wasm/src/runtime.rs:1364, jazz-tools/src/client.rs:127 and :150,
    // jazz-tools/src/server/builder.rs:236). This binding used to skip it, so a
    // node runtime came up knowing only the schema it was handed: every row
    // written under an earlier generation was indexed, persisted, and
    // unreadable at every durability tier. MEASURED on the rpc-server
    // (2026-08-16, store spanning the 2026-08-07 migration): users 2 of 22,
    // chat_members 4 of 11, chats 2 of 4 — and every chat-membership check 403'd.
    //
    // Nor did sync heal it: a peer re-sending the catalogue sends bytes this
    // store already holds, so only the first boot after a migration learned the
    // generation and every restart after that did not.
    //
    // Warn-don't-fail, as jazz-rn does: a store whose catalogue cannot be
    // scanned is still worth serving for the current generation. The silent
    // half of this failure — a scan that succeeds and matches nothing — is
    // caught after construction by `unknown_store_schema_generations`.
    if let Err(error) =
        rehydrate_schema_manager_from_catalogue(&mut schema_manager, &*storage, app_id_obj)
    {
        tracing::error!(
            app_id = %app_id_obj,
            %error,
            "failed to rehydrate the schema manager from catalogue storage; rows written \
             under earlier schema generations will be unreadable"
        );
    }

    // Create components
    let scheduler = NapiScheduler::new();

    // Create RuntimeCore and wrap
    let core = RuntimeCore::new(schema_manager, storage, scheduler);
    let core_arc = Arc::new(Mutex::new(core));

    // Set up the scheduler's weak reference and persist schema catalogue state.
    {
        let core_weak = Arc::downgrade(&core_arc);
        let mut core_guard = core_arc
            .lock()
            .map_err(|_| napi::Error::from_reason("lock"))?;
        core_guard.scheduler_mut().set_core_ref(core_weak);

        // Persist schema to catalogue for server sync
        core_guard.persist_schema();
    }

    Ok(NapiRuntime {
        core: core_arc,
        declared_schema,
    })
}

// ============================================================================
// NapiRuntime
// ============================================================================

#[napi]
pub struct NapiRuntime {
    core: Arc<Mutex<NapiCoreType>>,
    declared_schema: Schema,
}

#[napi]
impl NapiRuntime {
    /// Create a new NapiRuntime with SQLite-backed persistent storage.
    #[napi(constructor)]
    pub fn new(
        schema_json: String,
        app_id: String,
        jazz_env: String,
        user_branch: String,
        data_path: String,
        tier: Option<String>,
    ) -> napi::Result<Self> {
        let storage = open_sqlite_storage(&data_path)?;

        build_napi_runtime(
            schema_json,
            app_id,
            jazz_env,
            user_branch,
            Box::new(storage),
            tier,
        )
    }

    /// Create a new NapiRuntime with in-memory storage (no local persistence).
    #[napi(js_name = "inMemory")]
    pub fn in_memory(
        schema_json: String,
        app_id: String,
        jazz_env: String,
        user_branch: String,
        tier: Option<String>,
    ) -> napi::Result<Self> {
        build_napi_runtime(
            schema_json,
            app_id,
            jazz_env,
            user_branch,
            Box::new(MemoryStorage::new()),
            tier,
        )
    }

    // =========================================================================
    // CRUD Operations
    // =========================================================================

    #[napi(ts_return_type = "{ id: string; values: any[]; batchId: string }")]
    pub fn insert(
        &self,
        table: String,
        #[napi(ts_arg_type = "Record<string, unknown>")] values: FfiRecordArg,
        write_context_json: Option<String>,
        object_id: Option<String>,
    ) -> napi::Result<WrittenRow> {
        let write_context = parse_write_context_json(write_context_json)?;
        let object_id =
            parse_external_object_id(object_id.as_deref()).map_err(napi::Error::from_reason)?;
        let mut core = self
            .core
            .lock()
            .map_err(|_| napi::Error::from_reason("lock"))?;
        let ((object_id, row_values), batch_id) = core
            .insert_with_id(&table, values.0, object_id, write_context.as_ref())
            .map_err(|e| napi::Error::from_reason(format!("Insert failed: {:?}", e)))?;

        Ok(WrittenRow {
            id: object_id,
            values: row_values,
            batch_id: batch_id.to_string(),
        })
    }

    #[napi]
    pub fn update(
        &self,
        object_id: String,
        #[napi(ts_arg_type = "any")] values: FfiRecordArg,
        write_context_json: Option<String>,
    ) -> napi::Result<serde_json::Value> {
        let uuid = uuid::Uuid::parse_str(&object_id)
            .map_err(|e| napi::Error::from_reason(format!("Invalid ObjectId: {}", e)))?;
        let oid = ObjectId::from_uuid(uuid);
        let write_context = parse_write_context_json(write_context_json)?;

        let updates = convert_updates(values.0);

        let mut core = self
            .core
            .lock()
            .map_err(|_| napi::Error::from_reason("lock"))?;
        let batch_id = core
            .update(oid, updates, write_context.as_ref())
            .map_err(|e| napi::Error::from_reason(format!("Update failed: {:?}", e)))?;

        Ok(serde_json::json!({
            "batchId": batch_id.to_string(),
        }))
    }

    #[napi]
    pub fn upsert(
        &self,
        table: String,
        object_id: String,
        #[napi(ts_arg_type = "Record<string, unknown>")] values: FfiRecordArg,
        write_context_json: Option<String>,
    ) -> napi::Result<serde_json::Value> {
        let uuid = uuid::Uuid::parse_str(&object_id)
            .map_err(|e| napi::Error::from_reason(format!("Invalid ObjectId: {}", e)))?;
        let oid = ObjectId::from_uuid(uuid);
        let write_context = parse_write_context_json(write_context_json)?;

        let mut core = self
            .core
            .lock()
            .map_err(|_| napi::Error::from_reason("lock"))?;
        let batch_id = core
            .upsert(&table, oid, values.0, write_context.as_ref())
            .map_err(|e| napi::Error::from_reason(format!("Upsert failed: {:?}", e)))?;

        Ok(serde_json::json!({
            "batchId": batch_id.to_string(),
        }))
    }

    #[napi(js_name = "delete")]
    pub fn delete_row(
        &self,
        object_id: String,
        write_context_json: Option<String>,
    ) -> napi::Result<serde_json::Value> {
        let uuid = uuid::Uuid::parse_str(&object_id)
            .map_err(|e| napi::Error::from_reason(format!("Invalid ObjectId: {}", e)))?;
        let oid = ObjectId::from_uuid(uuid);
        let write_context = parse_write_context_json(write_context_json)?;

        let mut core = self
            .core
            .lock()
            .map_err(|_| napi::Error::from_reason("lock"))?;
        let batch_id = core
            .delete(oid, write_context.as_ref())
            .map_err(|e| napi::Error::from_reason(format!("Delete failed: {:?}", e)))?;

        Ok(serde_json::json!({
            "batchId": batch_id.to_string(),
        }))
    }

    #[napi(ts_return_type = "{ id: string; values: any[]; batchId: string }")]
    pub fn restore(
        &self,
        table: String,
        object_id: String,
        #[napi(ts_arg_type = "Record<string, unknown>")] values: FfiRecordArg,
        write_context_json: Option<String>,
    ) -> napi::Result<WrittenRow> {
        let uuid = uuid::Uuid::parse_str(&object_id)
            .map_err(|e| napi::Error::from_reason(format!("Invalid ObjectId: {}", e)))?;
        let oid = ObjectId::from_uuid(uuid);
        let write_context = parse_write_context_json(write_context_json)?;

        let mut core = self
            .core
            .lock()
            .map_err(|_| napi::Error::from_reason("lock"))?;
        let ((object_id, row_values), batch_id) = core
            .restore(&table, oid, values.0, write_context.as_ref())
            .map_err(|e| napi::Error::from_reason(format!("Restore failed: {:?}", e)))?;

        Ok(WrittenRow {
            id: object_id,
            values: row_values,
            batch_id: batch_id.to_string(),
        })
    }

    #[napi(
        js_name = "onMutationError",
        ts_args_type = "callback: (event: any) => void"
    )]
    pub fn on_mutation_error(&self, callback: MutationErrorTsfn) -> napi::Result<()> {
        let callback: MutationErrorCallback = Arc::new(move |event| {
            callback.call(
                Ok(serialize_mutation_error_event(event)),
                ThreadsafeFunctionCallMode::NonBlocking,
            );
        });
        self.core
            .lock()
            .map_err(|_| napi::Error::from_reason("lock"))?
            .set_mutation_error_callback(Some(callback));
        Ok(())
    }

    #[napi(js_name = "rollbackBatch")]
    pub fn rollback_batch(&self, batch_id: String) -> napi::Result<bool> {
        let batch_id = parse_batch_id_input(&batch_id).map_err(napi::Error::from_reason)?;
        let mut core = self
            .core
            .lock()
            .map_err(|_| napi::Error::from_reason("lock"))?;
        core.rollback_batch(batch_id)
            .map_err(|e| napi::Error::from_reason(format!("Rollback batch failed: {e}")))
    }

    #[napi(js_name = "beginBatch")]
    pub fn begin_batch(&self, batch_mode: String) -> napi::Result<String> {
        let batch_mode = parse_batch_mode_input(&batch_mode).map_err(napi::Error::from_reason)?;
        let mut core = self
            .core
            .lock()
            .map_err(|_| napi::Error::from_reason("lock"))?;
        Ok(core.begin_batch(batch_mode).to_string())
    }

    #[napi(js_name = "commitBatch")]
    pub fn commit_batch(&self, batch_id: String) -> napi::Result<()> {
        let batch_id = parse_batch_id_input(&batch_id).map_err(napi::Error::from_reason)?;
        let mut core = self
            .core
            .lock()
            .map_err(|_| napi::Error::from_reason("lock"))?;
        core.commit_batch(batch_id)
            .map_err(|e| napi::Error::from_reason(format!("Commit batch failed: {e}")))
    }

    #[napi(js_name = "waitForBatch", ts_return_type = "Promise<void>")]
    pub async fn wait_for_batch(&self, batch_id: String, tier: String) -> napi::Result<()> {
        let batch_id = parse_batch_id_input(&batch_id).map_err(napi::Error::from_reason)?;
        let tier = parse_binding_tier(&tier).map_err(napi::Error::from_reason)?;
        let receiver = {
            let mut core = self
                .core
                .lock()
                .map_err(|_| napi::Error::from_reason("lock"))?;
            core.wait_for_batch(batch_id, tier)
                .map_err(|e| napi::Error::from_reason(format!("Wait for batch failed: {e}")))?
        };

        match receiver.await {
            Ok(Ok(())) => Ok(()),
            Ok(Err(rejection)) => Err(napi::Error::from_reason(format!(
                "Persisted batch {} was rejected ({}): {}",
                rejection.batch_id, rejection.code, rejection.reason
            ))),
            Err(_) => Err(napi::Error::from_reason("Wait for batch cancelled")),
        }
    }

    // =========================================================================
    // Queries
    // =========================================================================

    #[napi(ts_return_type = "Promise<any>")]
    pub async fn query(
        &self,
        query_json: String,
        session_json: Option<String>,
        tier: Option<String>,
        options_json: Option<String>,
    ) -> napi::Result<QueryRows> {
        let query = parse_query(&query_json)?;
        let session = parse_session_json(session_json)?;

        let (durability, propagation, transaction_batch_id, timeout_ms) =
            parse_read_durability_options(tier.as_deref(), options_json.as_deref())
                .map_err(napi::Error::from_reason)?;

        let (handle, future) = {
            let mut core = self
                .core
                .lock()
                .map_err(|_| napi::Error::from_reason("lock"))?;
            core.query_with_local_batch_tracked(
                query,
                session,
                durability,
                propagation,
                transaction_batch_id,
            )
            .map_err(|e| napi::Error::from_reason(format!("Query setup failed: {e}")))?
        };

        let result = match timeout_ms {
            Some(ms) => {
                match tokio::time::timeout(std::time::Duration::from_millis(ms), future).await {
                    Ok(result) => result,
                    Err(_elapsed) => {
                        // The caller has given up: release the engine's subscription now
                        // so the abandoned read stops costing settle passes here and at
                        // the server. `false` means it settled in the meantime — the
                        // tick already cleaned it up.
                        let cancelled = self
                            .core
                            .lock()
                            .map_err(|_| napi::Error::from_reason("lock"))?
                            .cancel_one_shot_query(handle);
                        return Err(napi::Error::from_reason(format!(
                            "Query timed out after {ms}ms and was cancelled (released={cancelled})"
                        )));
                    }
                }
            }
            None => future.await,
        };

        let rows =
            result.map_err(|e| napi::Error::from_reason(format!("Query failed: {:?}", e)))?;

        Ok(QueryRows(rows))
    }

    // =========================================================================
    // Subscriptions
    // =========================================================================

    #[napi]
    pub fn unsubscribe(&self, handle: f64) -> napi::Result<()> {
        let mut core = self
            .core
            .lock()
            .map_err(|_| napi::Error::from_reason("lock"))?;
        core.unsubscribe(SubscriptionHandle(handle as u64));
        Ok(())
    }

    /// Phase 1 of 2-phase subscribe: allocate a handle and store query params.
    #[napi(js_name = "createSubscription")]
    pub fn create_subscription(
        &self,
        query_json: String,
        session_json: Option<String>,
        tier: Option<String>,
        options_json: Option<String>,
    ) -> napi::Result<f64> {
        let (query, session, durability, propagation) =
            parse_subscription_inputs(&query_json, session_json, tier, options_json)?;

        let mut core = self
            .core
            .lock()
            .map_err(|_| napi::Error::from_reason("lock"))?;
        let handle = core.create_subscription(query, session, durability, propagation);

        Ok(handle.0 as f64)
    }

    /// Phase 2 of 2-phase subscribe: compile, register, sync, attach callback, tick.
    #[napi(js_name = "executeSubscription")]
    pub fn execute_subscription(
        &self,
        handle: f64,
        #[napi(ts_arg_type = "(...args: any[]) => any")] on_update: ThreadsafeFunction<
            SubscriptionDeltaJs,
        >,
    ) -> napi::Result<()> {
        let sub_handle = SubscriptionHandle(handle as u64);
        let callback = make_subscription_callback(on_update);

        let mut core = self
            .core
            .lock()
            .map_err(|_| napi::Error::from_reason("lock"))?;
        core.execute_subscription(sub_handle, callback)
            .map_err(|e| {
                napi::Error::from_reason(format!("Execute subscription failed: {:?}", e))
            })?;

        Ok(())
    }

    // =========================================================================
    // Schema Access
    // =========================================================================

    #[napi(js_name = "getSchema", ts_return_type = "any")]
    pub fn get_schema(&self) -> napi::Result<serde_json::Value> {
        serde_json::to_value(&self.declared_schema)
            .map_err(|e| napi::Error::from_reason(format!("Schema serialization failed: {}", e)))
    }

    #[napi(js_name = "getSchemaHash")]
    pub fn get_schema_hash(&self) -> napi::Result<String> {
        let core = self
            .core
            .lock()
            .map_err(|_| napi::Error::from_reason("lock"))?;
        let schema = core.current_schema();
        Ok(SchemaHash::compute(schema).to_string())
    }

    #[napi]
    pub fn flush(&self) -> napi::Result<()> {
        let core = self
            .core
            .lock()
            .map_err(|_| napi::Error::from_reason("lock"))?;
        core.storage()
            .flush()
            .map_err(|e| napi::Error::from_reason(format!("Failed to flush storage: {:?}", e)))?;
        Ok(())
    }

    /// Flush and close the underlying storage, releasing filesystem locks.
    #[napi]
    pub fn close(&self) -> napi::Result<()> {
        let core = self
            .core
            .lock()
            .map_err(|_| napi::Error::from_reason("lock"))?;
        let flush_result = core.storage().flush();
        let flush_wal_result = core.storage().flush_wal();
        let close_result = core.storage().close();

        flush_result
            .map_err(|e| napi::Error::from_reason(format!("Failed to flush storage: {:?}", e)))?;
        flush_wal_result.map_err(|e| {
            napi::Error::from_reason(format!("Failed to flush storage WAL: {:?}", e))
        })?;
        close_result
            .map_err(|e| napi::Error::from_reason(format!("Failed to close storage: {:?}", e)))
    }

    /// Connect to a Jazz server over WebSocket.
    ///
    /// Parses `auth_json` into `AuthConfig`, wires a `TransportManager` into
    /// `RuntimeCore` via `install_transport` (which seeds the catalogue state
    /// hash on the handle), and spawns the manager loop as a Tokio task.
    #[napi]
    pub fn connect(&self, url: String, auth_json: String) -> napi::Result<()> {
        let auth: jazz_tools::transport_manager::AuthConfig = serde_json::from_str(&auth_json)
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        let tick = NapiTickNotifier {
            core: Arc::clone(&self.core),
        };
        let manager = {
            let mut core = self
                .core
                .lock()
                .map_err(|_| napi::Error::from_reason("lock"))?;
            jazz_tools::runtime_core::install_transport::<
                _,
                _,
                jazz_tools::ws_stream::NativeWsStream,
                _,
            >(&mut core, url, auth, tick)
        };
        // Spawn the TransportManager loop. If we're inside an active Tokio
        // runtime (typical: Node.js with napi-rs bootstrapping one), use it.
        // Otherwise (e.g. Next.js SSG build workers that load the addon
        // without a runtime) fall back to a dedicated runtime on a background
        // thread so `tokio::spawn` never panics.
        match tokio::runtime::Handle::try_current() {
            Ok(rt_handle) => {
                rt_handle.spawn(manager.run());
            }
            Err(_) => {
                std::thread::spawn(move || {
                    let rt = match tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                    {
                        Ok(rt) => rt,
                        Err(e) => {
                            eprintln!("jazz-napi: failed to build fallback tokio runtime: {e}");
                            return;
                        }
                    };
                    rt.block_on(manager.run());
                });
            }
        }
        Ok(())
    }

    /// Disconnect from the Jazz server and drop the transport handle.
    #[napi]
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
    #[napi]
    pub fn update_auth(&self, auth_json: String) -> napi::Result<()> {
        let auth: jazz_tools::transport_manager::AuthConfig = serde_json::from_str(&auth_json)
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        if let Ok(core) = self.core.lock()
            && let Some(handle) = core.transport()
        {
            handle.update_auth(auth);
        }
        Ok(())
    }

    /// Register a JS callback that fires when the Rust transport receives an
    /// auth rejection from the server during the WS handshake.
    ///
    /// The callback receives a single string argument: the rejection reason.
    #[napi(ts_args_type = "callback: (reason: string) => void")]
    pub fn on_auth_failure(
        &self,
        // CalleeHandled=false: JS callback receives (reason) not (error, reason).
        callback: ThreadsafeFunction<String, (), String, napi::Status, false, false, 0>,
    ) -> napi::Result<()> {
        let mut core = self
            .core
            .lock()
            .map_err(|_| napi::Error::from_reason("lock"))?;
        core.set_auth_failure_callback(move |reason| {
            callback.call(reason, ThreadsafeFunctionCallMode::NonBlocking);
        });
        Ok(())
    }
}

// ============================================================================
// NapiTickNotifier
// ============================================================================

/// `TickNotifier` implementation for the NAPI (Node.js) runtime.
///
/// Holds a weak-upgradeable reference to `RuntimeCore` and schedules a
/// `batched_tick` on the Node.js event loop whenever the transport layer
/// needs to wake up.
struct NapiTickNotifier {
    core: Arc<Mutex<NapiCoreType>>,
}

impl jazz_tools::transport_manager::TickNotifier for NapiTickNotifier {
    fn notify(&self) {
        if let Ok(core) = self.core.lock() {
            core.scheduler().schedule_batched_tick();
        }
    }
}

// ============================================================================
// TestJwtIssuer
// ============================================================================

#[napi]
pub struct TestJwtIssuer {
    inner: Mutex<Option<JazzTestJwtIssuer>>,
    jwks_url: String,
}

#[napi]
impl TestJwtIssuer {
    #[napi(factory, ts_return_type = "Promise<TestJwtIssuer>")]
    pub async fn start() -> napi::Result<Self> {
        let issuer = JazzTestJwtIssuer::start().await;
        let jwks_url = issuer.endpoint();
        Ok(Self {
            inner: Mutex::new(Some(issuer)),
            jwks_url,
        })
    }

    #[napi(getter, js_name = "jwksUrl")]
    pub fn jwks_url(&self) -> String {
        self.jwks_url.clone()
    }

    #[napi(js_name = "jwtForUser")]
    pub fn jwt_for_user(
        &self,
        user_id: String,
        #[napi(ts_arg_type = "Record<string, unknown> | undefined")] claims: Option<JsonValue>,
        #[napi(ts_arg_type = "{ expiresInSeconds?: number; issuer?: string } | undefined")]
        options: Option<JsonValue>,
    ) -> napi::Result<String> {
        let claims = claims.unwrap_or_else(|| serde_json::json!({ "role": "user" }));
        let options = match options {
            None | Some(JsonValue::Null) => TestJwtForUserOptions::default(),
            Some(value) => {
                serde_json::from_value::<TestJwtForUserOptions>(value).map_err(|error| {
                    napi::Error::from_reason(format!("Invalid JWT options: {error}"))
                })?
            }
        };
        let expires_in_seconds = options.expires_in_seconds.unwrap_or(3600);

        Ok(JazzTestJwtIssuer::jwt_for_user_with_options(
            &user_id,
            claims,
            TestJwtOptions {
                expires_in: Duration::from_secs(expires_in_seconds),
                issuer: options.issuer,
            },
        ))
    }

    #[napi]
    pub async fn stop(&self) -> napi::Result<()> {
        self.inner
            .lock()
            .map_err(|_| napi::Error::from_reason("lock"))?
            .take();
        Ok(())
    }
}

// ============================================================================
// JazzServer
// ============================================================================

#[napi]
pub struct JazzServer {
    inner: Mutex<Option<CoreJazzServer>>,
}

#[napi]
impl JazzServer {
    #[napi(factory, ts_return_type = "Promise<JazzServer>")]
    pub async fn start(
        #[napi(
            ts_arg_type = "{ appId: string; backendSecret: string; adminSecret: string; port?: number; dataDir?: string; inMemory?: boolean; jwksUrl?: string; allowLocalFirstAuth?: boolean; upstreamUrl?: string; telemetryCollectorUrl?: string }"
        )]
        options: JsonValue,
    ) -> napi::Result<Self> {
        let opts = parse_jazz_server_start_options(options)?;
        init_jazz_server_telemetry(opts.telemetry_collector_url.as_deref());

        let app_id =
            AppId::from_string(&opts.app_id).unwrap_or_else(|_| AppId::from_name(&opts.app_id));

        let auth_config = AuthConfig {
            jwks_url: opts.jwks_url,
            allow_local_first_auth: opts.allow_local_first_auth.unwrap_or(true),
            backend_secret: Some(opts.backend_secret.clone()),
            admin_secret: Some(opts.admin_secret.clone()),
            ..Default::default()
        };

        let in_memory = opts.in_memory.unwrap_or(false);
        let data_dir = if in_memory {
            String::new()
        } else {
            opts.data_dir.unwrap_or_else(|| "./data".to_string())
        };

        let mut server_builder = ServerBuilder::new(app_id).with_auth_config(auth_config);
        if let Some(upstream_url) = opts.upstream_url.clone() {
            server_builder = server_builder.with_upstream_url(upstream_url);
        }

        if in_memory {
            server_builder = server_builder.with_storage(StorageBackend::InMemory);
        } else {
            server_builder = server_builder.with_storage(StorageBackend::Sqlite {
                path: data_dir.clone().into(),
            });
        }

        let built = server_builder
            .build()
            .await
            .map_err(napi::Error::from_reason)?;

        let data_dir_path = std::path::PathBuf::from(&data_dir);

        let server = CoreJazzServer::from_built(
            built,
            opts.port,
            app_id,
            ServerDataDir::from_path(data_dir_path),
            opts.admin_secret.clone(),
            opts.backend_secret.clone(),
        )
        .await;

        Ok(Self {
            inner: Mutex::new(Some(server)),
        })
    }

    #[napi(getter, js_name = "appId")]
    pub fn app_id(&self) -> napi::Result<String> {
        self.with_server(|server| server.app_id().to_string())
    }

    #[napi(getter)]
    pub fn url(&self) -> napi::Result<String> {
        self.with_server(|server| server.base_url())
    }

    #[napi(getter)]
    pub fn port(&self) -> napi::Result<u16> {
        self.with_server(|server| server.port())
    }

    #[napi(getter, js_name = "dataDir")]
    pub fn data_dir(&self) -> napi::Result<String> {
        self.with_server(|server| server.data_dir().to_string_lossy().into_owned())
    }

    #[napi(getter, js_name = "backendSecret")]
    pub fn backend_secret(&self) -> napi::Result<String> {
        self.with_server(|server| server.backend_secret().to_string())
    }

    #[napi(getter, js_name = "adminSecret")]
    pub fn admin_secret(&self) -> napi::Result<String> {
        self.with_server(|server| server.admin_secret().to_string())
    }

    #[napi]
    pub async fn stop(&self) -> napi::Result<()> {
        let server = self
            .inner
            .lock()
            .map_err(|_| napi::Error::from_reason("lock"))?
            .take();

        if let Some(server) = server {
            server.shutdown().await;
        }

        Ok(())
    }

    fn with_server<T>(&self, f: impl FnOnce(&CoreJazzServer) -> T) -> napi::Result<T> {
        let server = self
            .inner
            .lock()
            .map_err(|_| napi::Error::from_reason("lock"))?;
        let server = server
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason("JazzServer has been stopped"))?;
        Ok(f(server))
    }
}

// ============================================================================
// Module-level utility functions
// ============================================================================

// ============================================================================
// Identity crypto utilities
// ============================================================================

fn decode_seed_napi(seed_b64: &str) -> napi::Result<[u8; 32]> {
    let bytes = URL_SAFE_NO_PAD
        .decode(seed_b64)
        .map_err(|e| napi::Error::from_reason(format!("seed base64 decode error: {e}")))?;
    bytes
        .try_into()
        .map_err(|_| napi::Error::from_reason("seed must be exactly 32 bytes"))
}

#[napi(js_name = "mintLocalFirstToken")]
pub fn mint_local_first_token(
    seed_b64: String,
    audience: String,
    ttl_seconds: u32,
) -> napi::Result<String> {
    let seed = decode_seed_napi(&seed_b64)?;
    identity::mint_jazz_self_signed_token(
        &seed,
        identity::LOCAL_FIRST_ISSUER,
        &audience,
        ttl_seconds as u64,
    )
    .map_err(napi::Error::from_reason)
}

#[napi(object)]
pub struct VerifyTokenResult {
    pub ok: bool,
    pub id: String,
    pub error: Option<String>,
}

#[napi(js_name = "verifyLocalFirstIdentityProof")]
pub fn verify_local_first_identity_proof_napi(
    token: Option<String>,
    expected_audience: String,
) -> VerifyTokenResult {
    let token = match token {
        Some(t) if !t.is_empty() => t,
        _ => {
            return VerifyTokenResult {
                ok: false,
                id: String::new(),
                error: Some("proofToken is required".to_string()),
            };
        }
    };
    match identity::verify_jazz_self_signed_proof(&token, &expected_audience) {
        Ok(verified) => VerifyTokenResult {
            ok: true,
            id: verified.user_id,
            error: None,
        },
        Err(e) => VerifyTokenResult {
            ok: false,
            id: String::new(),
            error: Some(e),
        },
    }
}

#[cfg(test)]
mod tests {
    /// Gate G5d: the deadline the caller passes as `timeout_ms` rejects the read AND releases
    /// the engine's subscription, through the real binding path (`NapiRuntime::query` on a
    /// tokio runtime with the time driver), because that glue has no other test and the
    /// rpc-server's facade depends on its error text.
    #[test]
    fn a_query_deadline_rejects_the_read_and_releases_its_subscription() {
        use jazz_tools::query_manager::types::{ColumnType, SchemaBuilder, TableSchema};
        use jazz_tools::sync_manager::ServerId;

        let schema = SchemaBuilder::new()
            .table(TableSchema::builder("docs").column("body", ColumnType::Text))
            .build();
        let dir = std::env::temp_dir().join(format!(
            "jazz-napi-query-deadline-gate-{}-{:?}",
            std::process::id(),
            std::time::SystemTime::now()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let runtime = super::NapiRuntime::new(
            serde_json::to_string(&schema).expect("serialize schema"),
            "query-deadline-gate".to_string(),
            "dev".to_string(),
            "main".to_string(),
            dir.join("runtime.sqlite").to_string_lossy().into_owned(),
            None,
        )
        .expect("runtime should construct");
        // An upstream that never answers: the tiered read below waits on its settle.
        let baseline = {
            let mut core = runtime.core.lock().expect("lock");
            core.schema_manager_mut()
                .query_manager_mut()
                .sync_manager_mut()
                .add_pending_server(ServerId::new());
            core.schema_manager().query_manager().subscription_count()
        };

        let tokio_runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("tokio runtime");
        let query_json =
            serde_json::to_string(&jazz_tools::query_manager::query::Query::new("docs"))
                .expect("serialize query");
        let started = std::time::Instant::now();
        let outcome = tokio_runtime.block_on(runtime.query(
            query_json,
            None,
            Some("edge".to_string()),
            Some(r#"{"timeout_ms":50}"#.to_string()),
        ));

        let err = outcome
            .err()
            .expect("the read must be rejected by its deadline");
        let regex = regex_lite_matches(&err.reason);
        assert!(
            regex,
            "the rejection must carry the timeout contract the facade matches on, got {:?}",
            err.reason
        );
        assert!(
            started.elapsed() < std::time::Duration::from_secs(5),
            "the deadline, not some other limit, must have ended the read"
        );
        let core = runtime.core.lock().expect("lock");
        assert_eq!(core.pending_one_shot_query_count(), 0);
        assert_eq!(
            core.schema_manager().query_manager().subscription_count(),
            baseline,
            "the abandoned read must not keep its subscription"
        );
        drop(core);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The facade's contract (`/timed out|cancelled/i`), spelled out without a regex crate.
    fn regex_lite_matches(reason: &str) -> bool {
        let lower = reason.to_ascii_lowercase();
        lower.contains("timed out") || lower.contains("cancelled")
    }
    use jazz_tools::query_manager::types::{
        ColumnType, ComposedBranchName, Schema, SchemaBuilder, TableName, TableSchema, Value,
    };

    use super::NapiRuntime;

    /// THE GATE for this binding.
    ///
    /// A runtime this binding constructs over a store whose catalogue already
    /// records an older schema generation must be able to read that generation.
    ///
    /// It could not. `build_napi_runtime` went from `SchemaManager::new*`
    /// straight into `RuntimeCore::new`, and the core reads the catalogue for
    /// nobody — every other binding does it itself before construction
    /// (jazz-rn/rust/src/lib.rs:734, jazz-wasm/src/runtime.rs:1364,
    /// jazz-tools/src/client.rs:127 and :150, server/builder.rs:236). So a node
    /// runtime's branch universe held only the schema it was handed
    /// (`SchemaContext::all_branch_names`), and every row written before the
    /// last migration was unreadable at every durability tier.
    ///
    /// MEASURED on the rpc-server's own replica (2026-08-16), a store spanning
    /// the 2026-08-07 migration: this construction served users 2 of 22,
    /// chat_members 4 of 11, chats 2 of 4, messages 17 of 416 — and every
    /// chat-membership check answered 403. Sync could not heal it either: the
    /// catalogue a peer re-sends is byte-identical to what the store already
    /// holds, so only the FIRST boot after a migration learned the generation.
    ///
    /// The fixture is the migration itself: generation A boots and persists its
    /// catalogue entry, then the same store comes up under generation B.
    #[test]
    fn a_runtime_reads_the_schema_generations_its_own_store_records() {
        let docs = || {
            TableSchema::builder("docs")
                .column("owner", ColumnType::Text)
                .column("body", ColumnType::Text)
        };
        let generation_a = SchemaBuilder::new().table(docs()).build();
        // An added table moves the schema hash, and therefore the composed
        // branch name — the shape of the production migration.
        let generation_b = SchemaBuilder::new()
            .table(docs())
            .table(TableSchema::builder("tags").column("label", ColumnType::Text))
            .build();

        let dir = std::env::temp_dir().join(format!(
            "jazz-napi-cold-boot-gate-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir).expect("scratch dir");
        let path = dir.join("runtime.sqlite");
        let path_arg = path.to_string_lossy().into_owned();
        let app_id = "cold-boot-generation-gate".to_string();

        let branch_of = |schema: &Schema| {
            ComposedBranchName::from_schema("dev", schema, "main")
                .to_branch_name()
                .to_string()
        };

        {
            let runtime = NapiRuntime::new(
                serde_json::to_string(&generation_a).expect("serialize generation A"),
                app_id.clone(),
                "dev".to_string(),
                "main".to_string(),
                path_arg.clone(),
                None,
            )
            .expect("generation A runtime should construct");
            // A row under generation A, so the store actually holds a visible
            // raw-table family for it. Without one, `visible_schema_generations`
            // has nothing to enumerate and the `unknown` assertion below would
            // pass no matter what.
            {
                let mut core = runtime.core.lock().expect("lock");
                core.insert_with_id(
                    "docs",
                    std::collections::HashMap::from([
                        (
                            "owner".to_string(),
                            jazz_tools::query_manager::types::Value::Text("alice".to_string()),
                        ),
                        (
                            "body".to_string(),
                            jazz_tools::query_manager::types::Value::Text("draft".to_string()),
                        ),
                    ]),
                    None,
                    None,
                )
                .expect("the row inserts under generation A");
            }
            runtime.flush().expect("generation A should reach the disk");
        }

        let runtime = NapiRuntime::new(
            serde_json::to_string(&generation_b).expect("serialize generation B"),
            app_id,
            "dev".to_string(),
            "main".to_string(),
            path_arg,
            None,
        )
        .expect("generation B runtime should construct");

        let (universe, unknown) = {
            let core = runtime.core.lock().expect("lock");
            (
                core.schema_manager().query_manager().all_query_branches(),
                core.unknown_store_schema_generations().to_vec(),
            )
        };

        let _ = std::fs::remove_dir_all(&dir);

        assert!(
            universe.contains(&branch_of(&generation_a)),
            "a runtime this binding constructs must include generation A's branch {} in its \
             query universe — its own catalogue records that generation. Universe was {universe:?}. \
             Every read is scoped to this list, so a missing branch makes the rows under it \
             unreadable at every durability tier.",
            branch_of(&generation_a)
        );
        assert!(
            universe.contains(&branch_of(&generation_b)),
            "the current generation's branch must still be there; universe was {universe:?}"
        );
        assert!(
            unknown.is_empty(),
            "and the runtime must not be reporting generations it cannot enumerate: {unknown:?}"
        );
    }

    #[test]
    fn schema_json_roundtrip_preserves_enum_fk_and_defaults() {
        let schema = SchemaBuilder::new()
            .table(TableSchema::builder("files").column("name", ColumnType::Text))
            .table(
                TableSchema::builder("todos")
                    .column_with_default("done", ColumnType::Boolean, Value::Boolean(false))
                    .column(
                        "status",
                        ColumnType::Enum {
                            variants: vec!["done".to_string(), "todo".to_string()],
                        },
                    )
                    .fk_column("image", "files"),
            )
            .build();

        let encoded = serde_json::to_string(&schema).expect("serialize schema");
        let decoded: Schema = serde_json::from_str(&encoded).expect("deserialize schema");

        let status = decoded
            .get(&TableName::new("todos"))
            .unwrap()
            .columns
            .column("status")
            .unwrap();
        assert_eq!(
            status.column_type,
            ColumnType::Enum {
                variants: vec!["done".to_string(), "todo".to_string()]
            }
        );

        let image = decoded
            .get(&TableName::new("todos"))
            .unwrap()
            .columns
            .column("image")
            .unwrap();
        assert_eq!(image.references, Some(TableName::new("files")));

        let done = decoded
            .get(&TableName::new("todos"))
            .unwrap()
            .columns
            .column("done")
            .unwrap();
        assert_eq!(done.default, Some(Value::Boolean(false)));
    }
}

#[cfg(test)]
mod napi_logging_gates {
    use super::napi_logging_directives;

    /// Internal on purpose: `napi_logging_directives` is a private policy helper with no public
    /// surface, and the thing it guards — a process-global `set_global_default` — cannot be
    /// asserted from an integration test without a second subscriber install racing it. What is
    /// testable is the decision, so that is what is pinned here.
    ///
    /// G-LOG. The client engine logs only on a non-empty `JAZZ_NAPI_LOG`. Two things are pinned
    /// here and both were nearly wrong:
    ///
    /// * the empty-string case — compose and `.env` files set variables to `""` routinely, and
    ///   an engine that started writing every trace event to stderr because of one would be a
    ///   regression in a shipping build, not a measurement;
    /// * the VARIABLE. The first version keyed on `RUST_LOG`, and `compose.prod.yml` sets
    ///   `RUST_LOG` on the rpc-server, the process that embeds this engine — so "opt-in" would
    ///   have meant "on in production immediately" (diff r28). The gate below names the variable
    ///   so that changing it back is a visible edit.
    #[test]
    fn the_client_engine_logs_only_when_its_own_variable_says_something() {
        assert_eq!(
            napi_logging_directives(None),
            None,
            "no JAZZ_NAPI_LOG at all is today's behaviour and must stay silent"
        );
        assert_eq!(
            napi_logging_directives(Some("")),
            None,
            "an empty JAZZ_NAPI_LOG is how a compose file spells `unset`, not `log everything`"
        );
        assert_eq!(
            napi_logging_directives(Some("   ")),
            None,
            "whitespace is the same thing with a typo in it"
        );
        assert_eq!(
            napi_logging_directives(Some(" jazz_tools=info ")),
            Some("jazz_tools=info"),
            "a real directive is passed through trimmed, so `EnvFilter` sees what was meant"
        );
    }
}
