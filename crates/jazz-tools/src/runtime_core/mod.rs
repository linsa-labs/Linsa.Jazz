//! RuntimeCore - Unified synchronous runtime logic for both native and WASM.
//!
//! This module provides the shared core logic that both jazz-tokio
//! and jazz-wasm wrap. RuntimeCore is generic over `Storage` and `Scheduler`
//! which provide platform-specific behavior.
//!
//! ## Design
//!
//! - `immediate_tick()` - processes managers synchronously, schedules batched_tick if needed
//! - `batched_tick()` - sends sync messages, applies parked responses/messages, calls immediate_tick
//! - Queries return `QueryFuture` for cross-platform awaiting
//! - Sync messages are "parked" and processed in batched_tick
//!
//! ## Usage
//!
//! ```ignore
//! let runtime = RuntimeCore::new(schema_manager, storage, scheduler);
//! runtime.insert(
//!     "users",
//!     std::collections::HashMap::from([
//!         ("id".to_string(), id),
//!         ("name".to_string(), name),
//!     ]),
//! )?;
//! runtime.immediate_tick();
//! let future = runtime.query(query);
//! let results = future.await?;
//! ```

use std::any::Any;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use futures::channel::oneshot;
use tracing::{debug, debug_span, info, trace, trace_span};

use crate::batch_fate::BatchMode;
use crate::object::{BranchName, ObjectId};
use crate::query_manager::QuerySubscriptionId;
use crate::query_manager::manager::{QueryError, QueryUpdate};
use crate::query_manager::query::Query;
use crate::query_manager::session::{Session, WriteContext};
use crate::query_manager::types::{
    OrderedRowDelta, Schema, SchemaHash, TableName, TablePolicies, Value,
};
use crate::row_format::decode_row;
use crate::row_histories::BatchId;
use crate::schema_manager::{Lens, SchemaManager};
use crate::storage::{PassOutcome, Storage, StorageError};
use crate::sync_manager::{ClientId, DurabilityTier, InboxEntry, OutboxEntry, ServerId};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MutationErrorEvent {
    pub code: String,
    pub reason: String,
    pub batch: crate::batch_fate::LocalBatchRecord,
}

#[cfg(target_arch = "wasm32")]
pub type MutationErrorCallback = std::rc::Rc<dyn Fn(&MutationErrorEvent) + 'static>;

#[cfg(not(target_arch = "wasm32"))]
pub type MutationErrorCallback =
    std::sync::Arc<dyn Fn(&MutationErrorEvent) + Send + Sync + 'static>;

#[cfg(target_arch = "wasm32")]
pub type RejectedBatchAcknowledgedCallback = std::rc::Rc<dyn Fn(BatchId) + 'static>;

#[cfg(not(target_arch = "wasm32"))]
pub type RejectedBatchAcknowledgedCallback =
    std::sync::Arc<dyn Fn(BatchId) + Send + Sync + 'static>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct QueryLocalOverlay {
    pub(crate) batch_id: BatchId,
    pub(crate) branch_name: BranchName,
    pub(crate) row_ids: Vec<ObjectId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RuntimeBatchContext {
    pub(crate) batch_mode: BatchMode,
    pub(crate) batch_id: BatchId,
    pub(crate) target_branch_name: BranchName,
}

// ============================================================================
// Scheduler and SyncSender traits
// ============================================================================

/// Schedules batched ticks on the platform's event loop.
///
/// No `Send` bound — WASM types (`Rc`, `Function`) are `!Send`.
/// Tokio enforces `Send` at the point of use (`Arc<Mutex<...>>`).
pub trait Scheduler {
    /// Must NOT re-enter the core synchronously (v18 item 6, diff r4 S2). The mandate holds
    /// at all thirteen call sites; the sharpest is the re-arm one (`ticks.rs`), where the
    /// caller holds the core's lock and has already released the settle clock, so a
    /// synchronous `batched_tick` would begin a fresh clock inside the same hold, defer
    /// again, re-arm again — unbounded recursion (diff r5). Every implementation defers to
    /// the platform's event loop (a worker, a spawned thread, a channel, `setTimeout`).
    fn schedule_batched_tick(&self);

    fn schedule_mutation_error_delivery(&self) {}
}

/// Sends sync messages to the network.
///
/// No `Send` bound — WASM types are `!Send`. Send is enforced
/// by the concrete wrapping type where needed.
pub trait SyncSender {
    fn send_sync_message(&self, message: OutboxEntry);
    fn as_any(&self) -> &dyn Any;
}

// ============================================================================
// Test helpers
// ============================================================================

/// No-op scheduler for tests — tests call tick explicitly.
pub struct NoopScheduler;

impl Scheduler for NoopScheduler {
    fn schedule_batched_tick(&self) {}
}

/// Collects sync messages for test inspection.
pub struct VecSyncSender {
    messages: std::sync::Mutex<Vec<OutboxEntry>>,
}

impl Default for VecSyncSender {
    fn default() -> Self {
        Self {
            messages: std::sync::Mutex::new(Vec::new()),
        }
    }
}

impl VecSyncSender {
    pub fn new() -> Self {
        Self::default()
    }

    /// Take all collected messages.
    pub fn take(&self) -> Vec<OutboxEntry> {
        std::mem::take(&mut self.messages.lock().unwrap())
    }
}

impl SyncSender for VecSyncSender {
    fn send_sync_message(&self, message: OutboxEntry) {
        self.messages.lock().unwrap().push(message);
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Handle to a subscription managed by RuntimeCore.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SubscriptionHandle(pub u64);

// Re-export QueryHandle from query_manager for convenience
pub use crate::query_manager::manager::QueryHandle as QMQueryHandle;
pub use subscriptions::ReadDurabilityOptions;

/// Errors from runtime operations.
#[derive(Debug, Clone)]
pub enum RuntimeError {
    QueryError(String),
    WriteError(String),
    NotFound,
    /// A one-shot query was cancelled by its caller's deadline before it settled.
    QueryCancelled,
    AnonymousWriteDenied {
        table: crate::query_manager::types::TableName,
        operation: crate::query_manager::policy::Operation,
    },
}

impl std::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RuntimeError::QueryError(s) => write!(f, "Query error: {}", s),
            RuntimeError::WriteError(s) => write!(f, "Write error: {}", s),
            RuntimeError::NotFound => write!(f, "Not found"),
            RuntimeError::QueryCancelled => write!(f, "Query cancelled by its deadline"),
            RuntimeError::AnonymousWriteDenied { table, operation } => {
                write!(
                    f,
                    "anonymous session cannot {} on table {}",
                    operation, table
                )
            }
        }
    }
}

impl std::error::Error for RuntimeError {}

impl From<QueryError> for RuntimeError {
    fn from(e: QueryError) -> Self {
        match e {
            QueryError::AnonymousWriteDenied { table, operation } => {
                RuntimeError::AnonymousWriteDenied { table, operation }
            }
            other => RuntimeError::QueryError(other.to_string()),
        }
    }
}

/// Convert a `QueryError` from a write path, preserving the
/// `AnonymousWriteDenied` variant and mapping anything else to `WriteError`.
pub(crate) fn write_error_from_query(e: QueryError) -> RuntimeError {
    match e {
        QueryError::AnonymousWriteDenied { table, operation } => {
            RuntimeError::AnonymousWriteDenied { table, operation }
        }
        other => RuntimeError::WriteError(other.to_string()),
    }
}

/// Type alias for query results.
pub type QueryResult = Result<Vec<(ObjectId, Vec<Value>)>, RuntimeError>;
/// Type alias for inserted row payloads.
pub type InsertedRow = (ObjectId, Vec<Value>);
/// Type alias for plain insert results that carry the inserted row plus its logical batch id.
pub type DirectInsertResult = (InsertedRow, BatchId);

/// Structured rejection returned by persisted writes when their batch is rejected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistedWriteRejection {
    pub batch_id: BatchId,
    pub code: String,
    pub reason: String,
}

/// Terminal outcome for a persisted write wait.
pub type PersistedWriteAck = std::result::Result<(), PersistedWriteRejection>;

/// Future that resolves to query results.
///
/// Cross-platform future implementation using `futures::channel::oneshot`.
/// Works with both tokio and wasm_bindgen_futures executors.
pub struct QueryFuture {
    receiver: oneshot::Receiver<QueryResult>,
}

impl QueryFuture {
    /// Create a new QueryFuture from a oneshot receiver.
    pub fn new(receiver: oneshot::Receiver<QueryResult>) -> Self {
        Self { receiver }
    }
}

impl Future for QueryFuture {
    type Output = QueryResult;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        Pin::new(&mut self.receiver)
            .poll(cx)
            .map(|r| r.unwrap_or_else(|_| Err(RuntimeError::QueryError("Query cancelled".into()))))
    }
}

/// Sender for fulfilling a QueryFuture.
pub type QuerySender = oneshot::Sender<QueryResult>;

/// Result of an immediate_tick cycle.
#[derive(Debug, Default)]
pub struct TickOutput {
    /// Subscription updates for this tick.
    pub subscription_updates: Vec<QueryUpdate>,
}

/// Delta for a subscription callback.
#[derive(Debug, Clone)]
pub struct SubscriptionDelta {
    /// The subscription handle.
    pub handle: SubscriptionHandle,
    /// The row changes with position-annotated ordering.
    pub ordered_delta: OrderedRowDelta,
    /// Output descriptor for decoding the binary row data.
    /// Use with `decode_row(&descriptor, &row.data)` to get `Vec<Value>`.
    pub descriptor: crate::query_manager::types::RowDescriptor,
}

/// Callback type for subscriptions.
///
/// On native platforms, callbacks must be `Send` for thread safety.
/// On WASM (single-threaded), `Send` is not required.
#[cfg(target_arch = "wasm32")]
pub type SubscriptionCallback = Box<dyn Fn(SubscriptionDelta) + 'static>;

#[cfg(not(target_arch = "wasm32"))]
pub type SubscriptionCallback = Box<dyn Fn(SubscriptionDelta) + Send + 'static>;

/// State for a subscription.
struct SubscriptionState {
    /// QueryManager's internal subscription ID.
    query_sub_id: QuerySubscriptionId,
    /// Callback invoked on updates.
    callback: SubscriptionCallback,
}

/// Pending one-shot query waiting for first subscription callback.
struct PendingOneShotQuery {
    subscription_id: QuerySubscriptionId,
    sender: Option<QuerySender>,
}

/// Unified runtime core for both native and WASM platforms.
///
/// Generic over `Storage` for data persistence and `Scheduler` for tick scheduling.
/// All business logic is synchronous.
pub struct RuntimeCore<S: Storage, Sch: Scheduler> {
    schema_manager: SchemaManager,
    pub(crate) storage: S,
    scheduler: Sch,
    /// True when storage was mutated since the last WAL flush barrier.
    storage_write_pending_flush: bool,
    /// v18 item 6 (tests only): re-arms scheduled for deferred settle work by this core —
    /// one per lock hold that left the flag set (design v8, gates G6-7(e)/(f)/(g)).
    #[cfg(any(test, feature = "test"))]
    settle_rearms: u64,
    /// True after scheduling one retry for the current failed WAL flush barrier.
    storage_flush_retry_scheduled: bool,
    /// Last storage flush error recorded by a durability barrier.
    storage_flush_error: Option<StorageError>,
    /// v18 item 4 (flag 2): the first `LostWrites` was logged at `error!`; later reports of
    /// the same loss are `trace!`. Never cleared.
    lost_writes_logged: bool,
    /// v18 item 4 (flag 3): the durability barrier reported a `LostWrites`. Set in one place
    /// (`flush_wal_barrier`'s `Err` arm), never cleared; gates only the STORAGE arm of the
    /// immediate tick's tail test and the barrier retry, so a node whose store lost a
    /// transaction does not schedule itself forever for a barrier that can never succeed.
    lost_writes_barrier_reported: bool,
    /// v18 item 4: the first `LostWrites` the store reported, kept apart from the carrier
    /// (`storage_flush_error`, which hosts take) for `TokioRuntime::flush`.
    lost_writes: Option<StorageError>,
    /// Schema generations this store holds visible rows under that the schema
    /// manager did not know at construction — see
    /// [`RuntimeCore::unknown_store_schema_generations`].
    unknown_store_schema_generations: Vec<SchemaHash>,
    /// Transport handle for WebSocket sync.
    pub(crate) transport: Option<crate::transport_manager::TransportHandle>,
    /// True when an inbound catalogue sync changed local catalogue state and
    /// the transport handshake hash must be refreshed after the tick applies it.
    transport_catalogue_state_hash_dirty: bool,
    /// Fallback outbox sender used when no `TransportHandle` is set (e.g. on
    /// the server side, where the runtime fans out via `ConnectionEventHub`
    /// instead of a WebSocket connection).
    ///
    /// On wasm32 the bound is `dyn SyncSender` because WASM is single-threaded
    /// and the JS/web-sys types it holds (`JsValue`, `Function`, `Rc`) are
    /// `!Send`. On other targets the multi-threaded Tokio backend requires
    /// the sender to be `Send`.
    #[cfg(target_arch = "wasm32")]
    pub(crate) sync_sender: Option<Box<dyn SyncSender>>,
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) sync_sender: Option<Box<dyn SyncSender + Send>>,
    /// When true, server-bound outbox entries are retained while no transport
    /// or fallback sync sender is installed. Browser broker tab runtimes enable
    /// this so writes made during leader election/reconnect remain awaitable.
    buffer_outbox_without_sync_sender: bool,

    /// Parked sync messages (from network).
    parked_sync_messages: Vec<InboxEntry>,
    /// Sequenced server messages buffered for in-order application.
    parked_sync_messages_by_server_seq: HashMap<ServerId, BTreeMap<u64, InboxEntry>>,
    /// Next expected per-server stream sequence.
    next_expected_server_seq: HashMap<ServerId, u64>,
    /// Highest per-server stream sequence already applied to the inbox.
    last_applied_server_seq: HashMap<ServerId, u64>,
    /// Subscription tracking with callbacks.
    subscriptions: HashMap<SubscriptionHandle, SubscriptionState>,
    /// Reverse map for routing updates.
    subscription_reverse: HashMap<QuerySubscriptionId, SubscriptionHandle>,
    next_subscription_handle: u64,
    /// Created-but-not-yet-executed subscriptions (2-phase subscribe).
    pending_subscriptions: HashMap<SubscriptionHandle, subscriptions::PendingSubscription>,

    /// Pending one-shot queries (query() calls waiting for first callback).
    pending_one_shot_queries: HashMap<SubscriptionHandle, PendingOneShotQuery>,

    /// Per-batch durability bookkeeping: ack watchers + rejection set.
    pub(crate) durability: DurabilityTracker,

    mutation_error_callback: Option<MutationErrorCallback>,
    rejected_batch_acknowledged_callback: Option<RejectedBatchAcknowledgedCallback>,

    acknowledged_rejected_batches: HashSet<BatchId>,

    /// Recently mutated local batch records. Large direct batches append one
    /// member per write, so reloading the record from storage on every insert
    /// means repeatedly decoding the full accumulated member array.
    /// Note: local_batch_record_cache is not authoritative after a batch has been sealed.
    /// commit_batch writes the record to storage and also leaves a copy in the cache;
    /// later incoming batch fate is applied to storage in apply_received_batch_fate,
    /// but the cached copy is not updated.
    local_batch_record_cache: HashMap<BatchId, crate::batch_fate::LocalBatchRecord>,

    /// Active in-memory context for JS batch/transaction handles. Handles do
    /// not survive process restarts, so this intentionally stays runtime-local.
    batch_contexts: HashMap<BatchId, RuntimeBatchContext>,

    /// Batches whose full-store fallback scan already answered "no rows".
    ///
    /// The fallback in `local_batch_rows` costs a scan of every table's
    /// history (seconds on a real store, holding the core lock the whole
    /// time). Its result is deterministic while no new rows for that batch
    /// land, so a repeat question — reconnect reconciliation, a re-received
    /// rejected fate, a seal retry — must answer from here instead of
    /// scanning again. Production incident 2026-08-09: one diverged client's
    /// retries drove one such scan every ~3.5s for 38 minutes, pinning a core.
    /// Entries are invalidated when a row batch with that id is parked or
    /// tracked locally. In-memory only: after a restart the first question
    /// pays one scan and re-learns.
    known_empty_batch_scans: std::cell::RefCell<HashSet<BatchId>>,

    /// Full-store batch scans this runtime has paid. The process-wide
    /// `LOCAL_BATCH_FULL_SCANS` counts every node in the process, which makes
    /// it useless both for a per-node metric and for a test asserting its own
    /// runtime's cost while the suite runs in parallel.
    local_batch_full_scans: std::cell::Cell<u64>,

    /// Label for tracing (e.g. "local", "edge", "client").
    tier_label: &'static str,

    /// Whether direct local writes should synthesize a local durable fate when sealed.
    ///
    /// Browser main-thread runtimes are optimistic peers in front of the durable
    /// worker runtime, so they must wait for worker-originated `BatchFate`
    /// instead of self-confirming.
    synthesize_direct_write_fate: bool,

    /// Optional sync-message tracer used by tests to record outgoing/incoming
    /// payloads under a human-readable participant name. `None` in production.
    pub(crate) sync_tracer: Option<(crate::sync_tracer::SyncTracer, String)>,

    /// Called when the transport rejects auth during the WS handshake.
    /// The String argument is a human-readable reason (e.g. "Unauthorized").
    pub(crate) auth_failure_callback: Option<Box<dyn Fn(String) + Send + 'static>>,
}

fn recover_pending_mutation_error_events<S: Storage>(
    storage: &S,
    acknowledged_rejected_batches: &HashSet<BatchId>,
) -> BTreeMap<BatchId, MutationErrorEvent> {
    let mut events = BTreeMap::new();

    for mut record in storage.scan_local_batch_records().unwrap_or_default() {
        let fate = record.latest_fate.clone().or_else(|| {
            storage
                .load_authoritative_batch_fate(record.batch_id)
                .ok()
                .flatten()
        });
        let Some(fate) = fate else {
            continue;
        };
        let crate::batch_fate::BatchFate::Rejected {
            batch_id,
            code,
            reason,
        } = &fate
        else {
            continue;
        };
        if acknowledged_rejected_batches.contains(batch_id) {
            continue;
        }
        record.latest_fate = Some(fate.clone());
        events.insert(
            *batch_id,
            MutationErrorEvent {
                code: code.clone(),
                reason: reason.clone(),
                batch: record,
            },
        );
    }

    events
}

/// Schema generations the store holds visible rows under that this schema
/// manager cannot enumerate — see
/// [`RuntimeCore::unknown_store_schema_generations`].
///
/// Reading the catalogue before construction is a per-binding obligation
/// (`rehydrate_schema_manager_from_catalogue`), honoured by jazz-rn, jazz-wasm,
/// the tokio client and the server builder, and — until this was written — not
/// by jazz-napi. Nothing failed when it was skipped: the runtime came up, every
/// read succeeded, and it simply answered "no such row" for everything written
/// before the last migration. This is what makes the next omission audible.
fn detect_unknown_store_schema_generations<S: Storage>(
    storage: &S,
    schema_manager: &SchemaManager,
) -> Vec<SchemaHash> {
    let stored = match crate::storage::visible_schema_generations(storage) {
        Ok(stored) => stored,
        Err(error) => {
            tracing::warn!(
                %error,
                "could not enumerate the store's schema generations; skipping the \
                 branch-universe coverage check"
            );
            return Vec::new();
        }
    };

    // A server-mode manager has no current schema at all: its context is
    // `SchemaContext::empty()` (`SchemaManager::new_server`), and
    // `process_catalogue_schema` files learned generations only in
    // `known_schemas` while `is_initialized` is false. Its universe is not
    // derived from `live_schemas` the way a client's is, so comparing against
    // `live_schemas` would report EVERY generation as missing — an ERROR on
    // every sync-server boot, accusing the one binding that does call the
    // rehydrate. There is nothing to diagnose here.
    if !schema_manager.has_current_schema() {
        return Vec::new();
    }

    let context = schema_manager.context();
    let unknown: Vec<SchemaHash> = stored
        .into_iter()
        // Against `live_schemas`, NOT `is_schema_known`: the branch universe is
        // current + live (`all_branch_names`), and a generation parked in
        // `pending_schemas` is also in `known_schemas` — treating "known" as
        // covered would make the unactivatable case below unreachable while its
        // rows stay just as unreadable.
        .filter(|schema_hash| {
            *schema_hash != context.current_hash && !context.live_schemas.contains_key(schema_hash)
        })
        .collect();

    // Four different faults land here and their remedies have nothing in
    // common, so name the one that actually applies rather than blaming the
    // binding for all of them.
    let mut unread = Vec::new();
    let mut unactivatable = Vec::new();
    let mut schemaless = Vec::new();
    let mut foreign_app = Vec::new();
    for schema_hash in &unknown {
        let rendered = schema_hash.to_string();
        let stored_entry = storage
            .load_catalogue_entry(schema_hash.to_object_id())
            .ok()
            .flatten();
        let entry_app_id = stored_entry.as_ref().and_then(|entry| {
            entry
                .metadata
                .get(crate::metadata::MetadataKey::AppId.as_str())
                .cloned()
        });
        if entry_app_id
            .as_deref()
            .is_some_and(|app_id| app_id != schema_manager.app_id().uuid().to_string())
        {
            // The entry is on disk under a DIFFERENT app id, so the rehydrate
            // filtered it out (`entry_matches_app`) and returned Ok having
            // matched nothing. The binding is fine; the app id it was handed is
            // not the one the store was written under.
            foreign_app.push(rendered);
        } else if context.pending_schemas.contains_key(schema_hash) {
            // The catalogue was read and holds this generation, but nothing
            // activates it: no non-draft lens path, and not identity-compatible
            // with the current schema (`SchemaContext::try_activate_pending`).
            unactivatable.push(rendered);
        } else if stored_entry.is_some() {
            // The entry is right there on disk, under this app id, and the
            // manager knows nothing about it — the catalogue was not read, or a
            // per-entry decode failed inside the rehydrate (which warns and
            // continues). This is the jazz-napi shape that blinded the
            // rpc-server.
            unread.push(rendered);
        } else {
            // Rows for a generation whose schema never arrived: inbound batches
            // are applied to the branch named on the wire, so a peer can create
            // a visible family here for a schema this store has no entry for.
            schemaless.push(rendered);
        }
    }

    if !foreign_app.is_empty() {
        tracing::error!(
            generations = ?foreign_app,
            app_id = %schema_manager.app_id(),
            "this store holds visible rows under schema generations whose catalogue entries \
             carry a DIFFERENT app id, so the catalogue read matched nothing and returned \
             successfully. Check the app id this runtime was constructed with against the one \
             the store was written under — not the binding's rehydrate call."
        );
    }
    if !unread.is_empty() {
        tracing::error!(
            generations = ?unread,
            current_generation = %context.current_hash,
            "this store's own catalogue records schema generations this runtime did not \
             load; every row under them is unreadable at every durability tier. Either the \
             binding skipped rehydrate_schema_manager_from_catalogue before constructing the \
             runtime, or the rehydrate failed to decode these entries (it warns and continues)."
        );
    }
    if !unactivatable.is_empty() {
        tracing::warn!(
            generations = ?unactivatable,
            current_generation = %context.current_hash,
            "this store holds visible rows under schema generations that were read from the \
             catalogue but cannot be activated — no lens path to the current schema and not \
             identity-compatible with it. Their rows stay unreadable until a lens is \
             published; this clears itself if one arrives over sync."
        );
    }
    if !schemaless.is_empty() {
        tracing::warn!(
            generations = ?schemaless,
            current_generation = %context.current_hash,
            "this store holds visible rows under schema generations it has no catalogue entry \
             for at all — rows arrived over sync ahead of their schema. Their rows stay \
             unreadable until the catalogue entry arrives."
        );
    }

    unknown
}

fn should_recover_pending_mutation_error_events(schema_manager: &SchemaManager) -> bool {
    schema_manager
        .query_manager()
        .sync_manager()
        .max_local_durability_tier()
        .is_none_or(|tier| tier < DurabilityTier::EdgeServer)
}

impl<S: Storage, Sch: Scheduler> RuntimeCore<S, Sch> {
    /// Create a new RuntimeCore.
    pub fn new(mut schema_manager: SchemaManager, mut storage: S, scheduler: Sch) -> Self {
        let _ = schema_manager.ensure_current_schema_persisted(&mut storage);
        // Heal defect 27's damage before anything reads: a store that was
        // written by an engine older than this one can hold a `(row, branch)`
        // with a visible head in two schema-generation families, and every read
        // would serve the stale one forever — no write ever revisits the row,
        // and a restart does not clear it. On a store with one generation per
        // table this is one header prefix scan and nothing else.
        match crate::storage::repair_all_split_visible_row_families(&mut storage) {
            Ok(report) if report.is_noop() => {}
            Ok(report) => tracing::warn!(
                split_rows = report.split_rows,
                dropped_heads = report.dropped_heads,
                rebuilt_rows = report.rebuilt_rows,
                unresolved_rows = report.unresolved_rows,
                repaired = ?report.repaired,
                unresolved = ?report.unresolved,
                failed_tables = ?report.failed_tables,
                "startup repaired visible heads split across schema generations"
            ),
            Err(error) => tracing::error!(
                %error,
                "startup sweep for schema-generation-split visible heads failed; the store \
                 may still serve a stale head for a split row"
            ),
        }
        let unknown_store_schema_generations =
            detect_unknown_store_schema_generations(&storage, &schema_manager);
        let acknowledged_rejected_batches: HashSet<BatchId> = storage
            .scan_acknowledged_rejected_batch_fates()
            .map(|batch_ids| batch_ids.into_iter().collect())
            .unwrap_or_default();
        let pending_mutation_error_events =
            if should_recover_pending_mutation_error_events(&schema_manager) {
                recover_pending_mutation_error_events(&storage, &acknowledged_rejected_batches)
            } else {
                BTreeMap::new()
            };

        Self {
            schema_manager,
            storage,
            scheduler,
            storage_write_pending_flush: false,
            #[cfg(any(test, feature = "test"))]
            settle_rearms: 0,
            storage_flush_retry_scheduled: false,
            storage_flush_error: None,
            lost_writes_logged: false,
            lost_writes_barrier_reported: false,
            lost_writes: None,
            unknown_store_schema_generations,
            transport: None,
            transport_catalogue_state_hash_dirty: false,
            sync_sender: None,
            buffer_outbox_without_sync_sender: true,
            parked_sync_messages: Vec::new(),
            parked_sync_messages_by_server_seq: HashMap::new(),
            next_expected_server_seq: HashMap::new(),
            last_applied_server_seq: HashMap::new(),
            subscriptions: HashMap::new(),
            subscription_reverse: HashMap::new(),
            next_subscription_handle: 0,
            pending_subscriptions: HashMap::new(),
            pending_one_shot_queries: HashMap::new(),
            durability: DurabilityTracker::with_initial_mutation_error_events(
                pending_mutation_error_events,
            ),
            mutation_error_callback: None,
            rejected_batch_acknowledged_callback: None,
            acknowledged_rejected_batches,
            local_batch_record_cache: HashMap::new(),
            known_empty_batch_scans: std::cell::RefCell::new(HashSet::new()),
            local_batch_full_scans: std::cell::Cell::new(0),
            batch_contexts: HashMap::new(),
            tier_label: "unknown",
            synthesize_direct_write_fate: true,
            sync_tracer: None,
            auth_failure_callback: None,
        }
    }

    /// Set the tier label used in tracing spans.
    pub fn set_tier_label(&mut self, label: &'static str) {
        self.tier_label = label;
    }

    /// Mark this runtime as an optimistic client whose direct writes are not
    /// durable until a tiered peer returns a `BatchFate`.
    pub fn set_non_durable_client_runtime(&mut self) {
        self.synthesize_direct_write_fate = false;
    }

    /// Register a callback that fires when the transport receives an auth failure
    /// from the server during the WS handshake.  The callback receives a
    /// human-readable reason string (e.g. "Unauthorized").
    pub fn set_auth_failure_callback(&mut self, cb: impl Fn(String) + Send + 'static) {
        self.auth_failure_callback = Some(Box::new(cb));
    }

    /// Attach a sync-message tracer. All outbox entries this runtime sends
    /// and all inbox entries it receives will be recorded under `name`.
    pub fn set_sync_tracer(&mut self, tracer: crate::sync_tracer::SyncTracer, name: String) {
        self.sync_tracer = Some((tracer, name));
    }

    pub fn set_mutation_error_callback(&mut self, callback: Option<MutationErrorCallback>) {
        self.mutation_error_callback = callback;
        self.schedule_mutation_error_delivery_if_needed();
    }

    pub fn set_rejected_batch_acknowledged_callback(
        &mut self,
        callback: Option<RejectedBatchAcknowledgedCallback>,
    ) {
        self.rejected_batch_acknowledged_callback = callback;
    }

    pub fn pending_mutation_error_delivery(
        &mut self,
    ) -> Option<(MutationErrorCallback, Vec<MutationErrorEvent>)> {
        let callback = self.mutation_error_callback.clone()?;
        let events = self.durability.drain_mutation_error_events();
        if events.is_empty() {
            return None;
        }
        for event in &events {
            if let Err(error) = self.acknowledge_handled_rejected_batch(event.batch.batch_id) {
                tracing::warn!(
                    batch_id = ?event.batch.batch_id,
                    %error,
                    "acknowledge delivered mutation error"
                );
            }
        }
        Some((callback, events))
    }

    pub fn acknowledge_handled_rejected_batch(
        &mut self,
        batch_id: BatchId,
    ) -> Result<bool, RuntimeError> {
        let acknowledged = self.acknowledge_rejected_batch(batch_id)?;
        if let Some(callback) = &self.rejected_batch_acknowledged_callback {
            callback(batch_id);
        }
        Ok(acknowledged)
    }

    fn queue_mutation_error_event(&mut self, event: MutationErrorEvent) {
        self.durability.queue_mutation_error_event(event);
        self.schedule_mutation_error_delivery_if_needed();
    }

    fn schedule_mutation_error_delivery_if_needed(&self) {
        if self.mutation_error_callback.is_some()
            && self.durability.has_pending_mutation_error_events()
        {
            self.scheduler.schedule_mutation_error_delivery();
        }
    }

    /// Get mutable reference to the Storage.
    pub fn storage_mut(&mut self) -> &mut S {
        &mut self.storage
    }

    /// Get reference to the Storage.
    pub fn storage(&self) -> &S {
        &self.storage
    }

    /// Schema generations this store held visible rows under that the schema
    /// manager did not know **at construction time**.
    ///
    /// A BOOT SNAPSHOT, not a live query: it is computed once in
    /// [`RuntimeCore::new`] and never revised, so a generation that activates
    /// later — a lens arriving over sync — stays listed here. Read it as "what
    /// this runtime came up unable to enumerate", which is the question worth
    /// asking, because that is when a binding's omission is diagnosable.
    ///
    /// Non-empty means every row written under those generations was unreadable
    /// at startup — the branch universe comes from `live_schemas`
    /// (`SchemaContext::all_branch_names`, schema_manager/context.rs:212), and a
    /// durability tier cannot rescue a row outside it (the scope filter only
    /// ever REMOVES tuples, `QueryManager::filter_synced_query_scope_tuples`).
    ///
    /// The four causes are distinguished in the logs
    /// (`detect_unknown_store_schema_generations`), not here — this list does
    /// not say WHY, only that coverage was incomplete.
    pub fn unknown_store_schema_generations(&self) -> &[SchemaHash] {
        &self.unknown_store_schema_generations
    }

    /// Flush the storage to persistent medium.
    pub fn flush_storage(&mut self) -> Result<(), StorageError> {
        match self.storage.flush() {
            Ok(()) => {
                self.clear_storage_write_pending_flush();
                Ok(())
            }
            Err(error) => {
                self.record_storage_flush_error(error.clone());
                Err(error)
            }
        }
    }

    /// Flush only the WAL buffer (not the full snapshot).
    pub fn flush_wal(&mut self) -> Result<(), StorageError> {
        self.flush_wal_barrier()
    }

    pub(crate) fn mark_storage_write_pending_flush(&mut self) {
        self.storage_write_pending_flush = true;
        self.storage_flush_retry_scheduled = false;
    }

    /// v18 item 6 (tests only): settle re-arms scheduled by this core.
    #[cfg(any(test, feature = "test"))]
    pub fn settle_rearms_for_test(&self) -> u64 {
        self.settle_rearms
    }
    pub(crate) fn has_storage_write_pending_flush(&self) -> bool {
        self.storage_write_pending_flush
    }

    pub(crate) fn has_storage_flush_error(&self) -> bool {
        self.storage_flush_error.is_some()
    }

    pub(crate) fn has_storage_flush_retry_scheduled(&self) -> bool {
        self.storage_flush_retry_scheduled
    }

    pub(crate) fn clear_storage_write_pending_flush(&mut self) {
        self.storage_write_pending_flush = false;
        self.storage_flush_retry_scheduled = false;
        self.storage_flush_error = None;
    }

    pub(crate) fn should_schedule_storage_flush_retry(&mut self) -> bool {
        if self.storage_flush_retry_scheduled {
            return false;
        }
        self.storage_flush_retry_scheduled = true;
        true
    }

    /// v18 item 4 (design v14, the carrier rule): a `LostWrites` is logged once at `error!`
    /// and snapshotted; later reports are `trace!` and leave the carrier AS IT IS — the first
    /// report is the one a host takes, and a host that already took it must not be handed a
    /// second copy of the same loss every tick. Every other error overwrites the carrier as
    /// before.
    pub(crate) fn record_storage_flush_error(&mut self, error: StorageError) {
        if let StorageError::LostWrites { detail } = &error {
            if self.lost_writes_logged {
                // `trace`, not `debug` (diff r21 SF4): both pass boundaries of every tick
                // reach here on a dead store. The first report was `error!`; this arm exists
                // to show the repeat when someone is watching, not to fill the log.
                tracing::trace!(detail, "storage lost writes (already reported)");
                return;
            }
            self.lost_writes_logged = true;
            tracing::error!(
                detail,
                "storage lost writes: the transaction was ended behind the store's back"
            );
            self.lost_writes = Some(error.clone());
        }
        self.storage_flush_error = Some(error);
    }

    pub fn take_storage_flush_error(&mut self) -> Option<StorageError> {
        self.storage_flush_error.take()
    }

    /// v18 item 4: the first `LostWrites` this core's store reported, if any. Unlike the
    /// carrier it is never taken: a store that lost writes never flushes again until it is
    /// reopened, and every later `flush()` must say so.
    pub fn lost_writes(&self) -> Option<&StorageError> {
        self.lost_writes.as_ref()
    }
    #[cfg(any(test, feature = "test"))]
    pub fn lost_writes_barrier_reported_for_test(&self) -> bool {
        self.lost_writes_barrier_reported
    }
    #[cfg(any(test, feature = "test"))]
    pub fn lost_writes_for_test(&self) -> Option<&StorageError> {
        self.lost_writes.as_ref()
    }

    /// v18 item 4 (C1): a settle pass begins — the store opens its pass transaction (SQLite)
    /// or does nothing. A failed begin is recorded and the pass runs autocommit; the store
    /// counted it.
    pub(crate) fn begin_read_pass_in_tick(&mut self) {
        if let Err(error) = self.storage.begin_read_pass() {
            self.record_storage_flush_error(error);
        }
    }

    /// v18 item 4 (design v14): end the pass at this site. `Ok(wrote)` hands a dirty
    /// transaction to the barrier; `Err` records the error (`LostWrites` logs once, flag 2)
    /// and marks the barrier pending so the barrier runs and reports; flag 3 is set by the
    /// barrier alone, on `LostWrites` alone.
    pub(crate) fn end_read_pass_in_tick(&mut self) {
        match self.storage.end_read_pass() {
            Ok(PassOutcome { wrote }) => {
                if wrote {
                    self.mark_storage_write_pending_flush();
                }
            }
            Err(error) => {
                self.record_storage_flush_error(error);
                self.mark_storage_write_pending_flush();
            }
        }
    }

    pub(crate) fn flush_wal_barrier(&mut self) -> Result<(), StorageError> {
        match self.storage.flush_wal() {
            Ok(()) => {
                self.clear_storage_write_pending_flush();
                Ok(())
            }
            Err(error) => {
                self.record_storage_flush_error(error.clone());
                if matches!(error, StorageError::LostWrites { .. }) {
                    self.lost_writes_barrier_reported = true;
                }
                Err(error)
            }
        }
    }

    /// Consume RuntimeCore and return the Storage.
    /// Used for cold-start testing to transfer driver state.
    pub fn into_storage(self) -> S {
        self.storage
    }

    /// Get reference to the Scheduler.
    pub fn scheduler(&self) -> &Sch {
        &self.scheduler
    }

    /// Get mutable reference to the Scheduler.
    pub fn scheduler_mut(&mut self) -> &mut Sch {
        &mut self.scheduler
    }

    /// Persist the current schema to the catalogue for server sync.
    pub fn persist_schema(&mut self) -> ObjectId {
        let id = self.schema_manager.persist_schema(&mut self.storage);
        self.mark_storage_write_pending_flush();
        self.refresh_transport_catalogue_state_hash();
        info!(object_id = %id, "persisted schema to catalogue");
        id
    }

    /// Publish any known schema object to the catalogue and in-memory schema manager.
    pub fn publish_schema(&mut self, schema: Schema) -> ObjectId {
        let schema_hash = crate::query_manager::types::SchemaHash::compute(&schema);

        if self.schema_manager.get_known_schema(&schema_hash).is_none() {
            self.schema_manager.add_known_schema(schema.clone());
        }

        let id = self
            .schema_manager
            .persist_schema_object(&mut self.storage, &schema);
        self.mark_storage_write_pending_flush();
        self.refresh_transport_catalogue_state_hash();
        self.immediate_tick();
        id
    }

    pub fn publish_permissions_bundle(
        &mut self,
        schema_hash: SchemaHash,
        permissions: HashMap<TableName, TablePolicies>,
        expected_parent_bundle_object_id: Option<ObjectId>,
    ) -> Result<Option<ObjectId>, crate::schema_manager::SchemaError> {
        let id = self.schema_manager.publish_permissions_bundle(
            &mut self.storage,
            schema_hash,
            permissions,
            expected_parent_bundle_object_id,
        )?;
        if id.is_some() {
            self.mark_storage_write_pending_flush();
            self.refresh_transport_catalogue_state_hash();
        }
        self.immediate_tick();
        Ok(id)
    }

    /// Publish a reviewed lens edge to the active schema manager and catalogue.
    pub fn publish_lens(&mut self, lens: &Lens) -> Result<ObjectId, RuntimeError> {
        let id = self
            .schema_manager
            .publish_lens(&mut self.storage, lens)
            .map_err(|error| RuntimeError::WriteError(error.to_string()))?;
        self.mark_storage_write_pending_flush();
        self.refresh_transport_catalogue_state_hash();
        self.immediate_tick();
        Ok(id)
    }
    // =========================================================================
    // Schema/State Access
    // =========================================================================

    /// Get the current schema.
    pub fn current_schema(&self) -> &Schema {
        self.schema_manager.current_schema()
    }

    /// Get mutable access to the underlying SchemaManager.
    pub fn schema_manager_mut(&mut self) -> &mut SchemaManager {
        &mut self.schema_manager
    }

    /// Add a historical live schema and persist both schema and lens catalogue objects.
    pub fn add_live_schema_and_persist_catalogue(
        &mut self,
        schema: Schema,
    ) -> Result<(), crate::schema_manager::context::SchemaError> {
        let lens = self.schema_manager.add_live_schema(schema.clone())?.clone();
        self.schema_manager
            .persist_schema_object(&mut self.storage, &schema);
        self.schema_manager.persist_lens(&mut self.storage, &lens);
        self.refresh_transport_catalogue_state_hash();
        Ok(())
    }

    /// Number of one-shot queries still waiting for their first settled snapshot.
    pub fn pending_one_shot_query_count(&self) -> usize {
        self.pending_one_shot_queries.len()
    }

    /// Get access to the underlying SchemaManager.
    pub fn schema_manager(&self) -> &SchemaManager {
        &self.schema_manager
    }
}

/// Create a `TransportManager`, seed it with the current catalogue state hash,
/// install its handle on the given core, and return the manager for the caller
/// to spawn on an appropriate executor.
///
/// Centralises the boilerplate that would otherwise be duplicated in every
/// binding (Tokio, NAPI, RN, WASM).
#[cfg(feature = "transport")]
pub fn install_transport<S, Sch, W, T>(
    core: &mut RuntimeCore<S, Sch>,
    url: String,
    auth: crate::transport_manager::AuthConfig,
    tick: T,
) -> crate::transport_manager::TransportManager<W, T>
where
    S: crate::storage::Storage,
    Sch: Scheduler,
    W: crate::transport_manager::StreamAdapter + 'static,
    T: crate::transport_manager::TickNotifier + 'static,
{
    install_transport_with_retry_config(
        core,
        url,
        auth,
        tick,
        crate::transport_manager::TransportRetryConfig::default(),
    )
}

#[cfg(feature = "transport")]
pub fn install_transport_with_retry_config<S, Sch, W, T>(
    core: &mut RuntimeCore<S, Sch>,
    url: String,
    auth: crate::transport_manager::AuthConfig,
    tick: T,
    retry_config: crate::transport_manager::TransportRetryConfig,
) -> crate::transport_manager::TransportManager<W, T>
where
    S: crate::storage::Storage,
    Sch: Scheduler,
    W: crate::transport_manager::StreamAdapter + 'static,
    T: crate::transport_manager::TickNotifier + 'static,
{
    debug_assert!(
        core.transport().is_none(),
        "install_transport called while a transport is already installed; call clear_transport / disconnect first"
    );
    // Resolve the wire ClientId from the runtime's storage so every binding
    // (client, RN, NAPI) presents a stable identity across process restarts.
    // The server keys its per-client delivery frontier by this id; a fresh id
    // per launch makes every relaunch replay the full visible dataset.
    let client_id = Some(stable_wire_client_id(core.storage_mut()));
    let (handle, manager) = crate::transport_manager::create_with_retry_config::<W, T>(
        url,
        auth,
        tick,
        retry_config,
        client_id,
    );
    handle.set_catalogue_state_hash(Some(core.schema_manager().catalogue_state_hash()));
    handle.set_declared_schema_hash(
        core.schema_manager()
            .has_current_schema()
            .then(|| core.schema_manager().current_hash().to_string()),
    );
    core.schema_manager
        .query_manager_mut()
        .sync_manager_mut()
        .add_pending_server(handle.server_id);
    core.set_transport(handle);
    manager
}

/// Load the store's stable wire ClientId, minting and persisting one on
/// first use.
///
/// The id lives INSIDE the store (raw KV table `__jazz_meta`) on purpose:
/// wiping the local store must also rotate the identity, otherwise the
/// server's per-client delivery frontier would skip batches the client no
/// longer has. Backends without raw-table support fall back to a fresh id
/// per process — the pre-existing behavior.
#[cfg(feature = "transport")]
const WIRE_CLIENT_ID_TABLE: &str = "__jazz_meta";
#[cfg(feature = "transport")]
const WIRE_CLIENT_ID_KEY: &str = "wire_client_id";

/// Pin the store's wire ClientId, replacing any existing one.
///
/// Callers that already own an identity (an explicit `AppContext::client_id`,
/// a restored backup) use this before connecting.
#[cfg(feature = "transport")]
pub fn seed_wire_client_id<S: crate::storage::Storage + ?Sized>(
    storage: &mut S,
    client_id: crate::sync_manager::ClientId,
) {
    if let Err(error) = storage.raw_table_put(
        WIRE_CLIENT_ID_TABLE,
        WIRE_CLIENT_ID_KEY,
        client_id.to_string().as_bytes(),
    ) {
        tracing::debug!(?error, "could not seed wire client id");
    }
}

#[cfg(feature = "transport")]
fn stable_wire_client_id<S: crate::storage::Storage>(
    storage: &mut S,
) -> crate::sync_manager::ClientId {
    use crate::sync_manager::ClientId;

    const META_TABLE: &str = WIRE_CLIENT_ID_TABLE;
    const META_KEY: &str = WIRE_CLIENT_ID_KEY;

    if let Ok(Some(bytes)) = storage.raw_table_get(META_TABLE, META_KEY)
        && let Ok(text) = std::str::from_utf8(&bytes)
        && let Some(id) = ClientId::parse(text.trim())
    {
        return id;
    }

    let id = ClientId::new();
    if let Err(error) = storage.raw_table_put(META_TABLE, META_KEY, id.to_string().as_bytes()) {
        tracing::debug!(
            ?error,
            "could not persist wire client id; falling back to per-process identity"
        );
    }
    id
}

impl<S: Storage, Sch: Scheduler> RuntimeCore<S, Sch> {
    /// Attach a transport handle. Replaces any existing transport.
    pub fn set_transport(&mut self, handle: crate::transport_manager::TransportHandle) {
        self.transport = Some(handle);
    }

    pub(crate) fn mark_transport_catalogue_state_hash_dirty(&mut self) {
        self.transport_catalogue_state_hash_dirty = true;
    }

    fn refresh_transport_catalogue_state_hash(&mut self) {
        if let Some(handle) = self.transport.as_ref() {
            handle.set_catalogue_state_hash(Some(self.schema_manager.catalogue_state_hash()));
        }
        self.transport_catalogue_state_hash_dirty = false;
    }

    /// Detach the transport handle and remove its server from sync state.
    pub fn clear_transport(&mut self) {
        if let Some(h) = self.transport.take() {
            self.remove_server(h.server_id);
        }
    }

    /// Returns a reference to the active transport handle, if any.
    pub fn transport(&self) -> Option<&crate::transport_manager::TransportHandle> {
        self.transport.as_ref()
    }
}

impl<S: Storage, Sch: Scheduler> RuntimeCore<S, Sch> {
    /// Install a fallback sync sender used when no `TransportHandle` is set.
    /// On the server side, this is the bridge from the runtime's outbox into
    /// the per-connection `ConnectionEventHub` channels.
    #[cfg(target_arch = "wasm32")]
    pub fn set_sync_sender(&mut self, sender: Box<dyn SyncSender>) {
        self.sync_sender = Some(sender);
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn set_sync_sender(&mut self, sender: Box<dyn SyncSender + Send>) {
        self.sync_sender = Some(sender);
    }

    /// Remove the fallback sync sender without draining or acknowledging the
    /// current outbox. Subsequent ticks retain server-bound messages until a
    /// transport or replacement sender is installed.
    pub fn clear_sync_sender(&mut self) {
        self.sync_sender = None;
    }

    pub fn set_buffer_outbox_without_sync_sender(&mut self, enabled: bool) {
        self.buffer_outbox_without_sync_sender = enabled;
    }

    #[cfg(test)]
    pub fn sync_sender(&self) -> &VecSyncSender {
        self.sync_sender
            .as_ref()
            .expect("test runtime must install a VecSyncSender")
            .as_any()
            .downcast_ref::<VecSyncSender>()
            .expect("test runtime sync sender must be VecSyncSender")
    }
}

mod durability;
mod subscriptions;
mod sync;
mod ticks;
mod writes;

pub use ticks::LOCAL_BATCH_FULL_SCANS;

use durability::DurabilityTracker;

#[cfg(test)]
mod tests;
