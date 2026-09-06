use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::batch_fate::BatchFate;
use crate::catalogue::CatalogueEntry;
use crate::metadata::{MetadataKey, ObjectType};
use crate::object::{BranchName, ObjectId};
use crate::row_histories::{BatchId, QueryRowBatch, RowState, RowVisibilityChange, StoredRowBatch};
use crate::schema_manager::{
    LensTransformer, SchemaContext, encoding::encode_schema, resolve_current_table_name,
    translate_table_name_to_schema,
};
use crate::storage::{RowLocator, Storage, StorageError};
use crate::sync_manager::{
    ClientId, DurabilityTier, PendingPermissionCheck, PendingUpdateId, QueryId, QueryPropagation,
    RowBatchKey, SchemaWarning, SyncManager,
};

use super::encoding::decode_row;
use super::graph::{QueryCompileError, QueryGraph};
use super::graph_nodes::output::QuerySubscriptionId;
use super::policy::{Operation, PolicyExpr};
use super::policy_graph::PolicyGraph;
use super::query::Query;
use super::session::Session;
use super::settlement_eval_cache::SettlementEvalCache;
use super::types::{
    ColumnName, ComposedBranchName, LoadedRow, OrderedAdded, OrderedRowDelta, Row, RowDelta,
    RowDescriptor, RowPolicyMode, Schema, SchemaHash, TableName, TablePolicies, Tuple, Value,
    build_ordered_delta_with_post_ids,
};

/// Error types for QueryManager operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryError {
    TableNotFound(TableName),
    ColumnCountMismatch {
        expected: usize,
        actual: usize,
    },
    EncodingError(String),
    ObjectNotFound(ObjectId),
    QueryCompilationError(String),
    IndexValueTooLarge {
        table: TableName,
        column: String,
        branch: String,
        key_bytes: usize,
        max_key_bytes: usize,
    },
    IndexError(String),
    /// Cannot restore or truncate a row that is not soft-deleted.
    RowNotDeleted(ObjectId),
    /// Cannot write to an already-deleted row.
    RowAlreadyDeleted(ObjectId),
    /// Cannot operate on a hard-deleted row (it no longer exists).
    RowHardDeleted(ObjectId),
    /// Policy denied the operation.
    PolicyDenied {
        table: TableName,
        operation: Operation,
    },
    /// Write denied because the session is anonymous.
    /// Short-circuited before policy evaluation; surfaces as ANONYMOUS_WRITE_DENIED on the wire.
    AnonymousWriteDenied {
        table: TableName,
        operation: Operation,
    },
    /// Unknown schema hash - client should sync schema first.
    UnknownSchema(SchemaHash),
}

impl std::fmt::Display for QueryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QueryError::TableNotFound(t) => write!(f, "table not found: {}", t),
            QueryError::ColumnCountMismatch { expected, actual } => {
                write!(
                    f,
                    "column count mismatch: expected {expected}, got {actual}"
                )
            }
            QueryError::EncodingError(msg) => write!(f, "encoding error: {msg}"),
            QueryError::ObjectNotFound(id) => write!(f, "object not found: {:?}", id),
            QueryError::QueryCompilationError(msg) => write!(f, "query compilation error: {msg}"),
            QueryError::IndexValueTooLarge {
                table,
                column,
                branch,
                key_bytes,
                max_key_bytes,
            } => write!(
                f,
                "indexed value too large for {table}.{column} on branch {branch}: index key would be {key_bytes} bytes (max {max_key_bytes})"
            ),
            QueryError::IndexError(msg) => write!(f, "index error: {msg}"),
            QueryError::RowNotDeleted(id) => write!(f, "row not deleted: {id}"),
            QueryError::RowAlreadyDeleted(id) => write!(f, "row already deleted: {id}"),
            QueryError::RowHardDeleted(id) => write!(f, "row hard deleted: {:?}", id),
            QueryError::PolicyDenied { table, operation } => {
                write!(f, "policy denied {} on table {}", operation, table)
            }
            QueryError::AnonymousWriteDenied { table, operation } => {
                write!(
                    f,
                    "anonymous session cannot {} on table {}",
                    operation, table
                )
            }
            QueryError::UnknownSchema(hash) => {
                write!(
                    f,
                    "unknown schema: {} - client should sync schema first",
                    hash.short()
                )
            }
        }
    }
}

impl std::error::Error for QueryError {}

/// Handle to a pending query.
///
/// Used to correlate query results with the original request.
/// Wrappers (jazz-runtime, jazz-wasm) use this to fulfill
/// platform-specific futures/promises.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QueryHandle(pub u64);

/// Whose local writes a subscription may be exempted from server-scope filtering for.
///
/// The two sources are ADDITIVE, not alternatives, which is why this is a struct and not
/// an enum of cases. Getting that wrong in either direction is a real defect:
///
/// | overlay   | `LocalUpdates` | authority                |
/// | --------- | -------------- | ------------------------ |
/// | non-empty | `Immediate`    | overlay ∪ process-wide   |
/// | non-empty | `Deferred`     | overlay only             |
/// | empty     | `Immediate`    | process-wide             |
/// | empty     | `Deferred`     | none                     |
///
/// Row one is the subtle one and it is the common case, since `Immediate` is the default
/// for a transaction-scoped read. Such a read downgrades its durability tier for EVERY row
/// it loads, not only staged ones (`lacks_authoritative_remote_scope` tests `LocalUpdates`
/// and says nothing about the overlay), so it materialises the node's ordinary unsettled
/// writes too — and dropping those would blind it to writes it made outside the
/// transaction, which is exactly what `Immediate` asks not to happen.
///
/// Row two is the other direction: a `Deferred` read that staged rows asked for ITS batch,
/// and handing it the process-wide maps would exempt it for every unsettled write in the
/// process, most of them nothing to do with it.
struct LocalWriteAuthority<'a> {
    /// Rows this read staged itself, if any.
    overlay: Option<&'a HashMap<ObjectId, RowBatchKey>>,
    /// Whether this node's own unsettled writes count for this reader.
    process_wide: bool,
}

impl LocalWriteAuthority<'_> {
    fn grants_nothing(&self) -> bool {
        self.overlay.is_none() && !self.process_wide
    }
}

/// What the three maps tracking a local write say about one row. See
/// `QueryManager::retire_local_row_tracking`. Exists for the gates that assert the family
/// retires together, so it is test-only by construction.
#[cfg(test)]
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct LocalRowTracking {
    pub(crate) pending_batch: bool,
    pub(crate) scope_exempt: bool,
    pub(crate) awaiting_scope: bool,
}

#[cfg(test)]
impl LocalRowTracking {
    /// True when no map still holds the row.
    pub(crate) fn is_retired(&self) -> bool {
        !self.pending_batch && !self.scope_exempt && !self.awaiting_scope
    }
}

/// A local write that is durable at the settlement target and is waiting for the scope
/// snapshot that accounts for it. See `QueryManager::confirmed_local_rows_awaiting_scope`.
#[derive(Debug, Clone)]
pub(super) struct ParkedExemption {
    pub(super) branch: BranchName,
    pub(super) parked_in_pass: u64,
}

/// Result of an insert, including durability metadata and row values.
///
/// Poll via `is_complete()` to check if the row is persisted.
/// Poll via `is_indexed()` to check if the row is indexed.
#[derive(Debug, Clone)]
pub struct InsertResult {
    /// The row's ObjectId.
    pub row_id: ObjectId,
    /// Logical batch identity for the written row member.
    pub batch_id: BatchId,
    /// Inserted row values in table column order.
    pub row_values: Vec<Value>,
}

/// Handle for tracking delete completion.
#[derive(Debug, Clone)]
pub struct DeleteHandle {
    /// The row's ObjectId.
    pub row_id: ObjectId,
    /// Logical batch identity for the tombstone row member.
    pub batch_id: BatchId,
}

impl InsertResult {
    /// Check if the row data is durable (persisted to storage).
    ///
    /// Must call `QueryManager::process()` between checks to drive storage operations.
    pub fn is_complete(&self, qm: &QueryManager, storage: &dyn Storage) -> bool {
        qm.is_version_stored(storage, self.row_id, &self.batch_id)
    }

    /// Check if the row is indexed (appears in the _id index).
    ///
    /// After insert + process(), the row should be indexed.
    pub fn is_indexed(&self, qm: &QueryManager, storage: &dyn Storage, table: &str) -> bool {
        qm.row_is_indexed(storage, table, self.row_id)
    }
}

/// Query subscription info.
#[derive(Debug)]
pub(crate) struct QuerySubscription {
    /// Original query for recompilation when schemas change.
    pub(crate) query: Query,
    /// Compiled query graph.
    pub(crate) graph: QueryGraph,
    /// Branches to read from (updated on recompile).
    pub(crate) branches: Vec<String>,
    /// Session for policy filtering (if any).
    pub(crate) session: Option<Session>,
    /// Flag indicating this subscription needs recompilation due to schema change.
    pub(crate) needs_recompile: bool,
    /// Flag indicating this subscription has settled at least once.
    /// Used to ensure one-shot queries receive an initial callback (even if empty).
    pub(crate) settled_once: bool,
    /// True when visibility can change without graph dirtiness, e.g. initial
    /// frontier completion or a remote query-scope snapshot.
    pub(crate) needs_visibility_recompute: bool,
    /// Required durability tier before non-local delivery (None = immediate).
    pub(crate) durability_tier: Option<DurabilityTier>,
    /// How local writes behave while waiting for durability.
    pub(crate) local_updates: LocalUpdates,
    /// True when this subscription observed a local write since last delivery.
    pub(crate) has_pending_local_updates: bool,
    /// Row ids that should use the local current version as an overlay while
    /// waiting for a stricter settled tier.
    pub(crate) pending_local_row_ids: HashSet<ObjectId>,
    /// Optional one-shot overlay keyed by row id for a specific local batch.
    /// When present, reads must not fall back to unrelated pending local rows.
    pub(crate) local_overlay_rows: HashMap<ObjectId, RowBatchKey>,
    /// Highest durability tier at which the initial upstream query frontier has settled.
    pub(crate) query_frontier_settled_tier: Option<DurabilityTier>,
    /// Current ordered IDs for ordered delta construction.
    pub(crate) current_ordered_ids: Vec<ObjectId>,
    /// Last visible rows delivered to the subscriber when explicit auth filtering is active.
    pub(crate) current_visible_rows: HashMap<ObjectId, Row>,
    /// Extra tables whose rows must be available locally to evaluate this
    /// subscription's bundled policy context.
    pub(crate) policy_context_tables: Vec<String>,
    /// Whether this subscription uses post-settle auth filtering instead of graph policies.
    pub(crate) uses_explicit_authorization_filtering: bool,
    /// Whether visible rows must stay aligned to the latest upstream query scope.
    pub(crate) sync_backed: bool,
    /// Whether this subscription should be forwarded to upstream servers.
    pub(crate) propagation: QueryPropagation,
    /// Schema mismatch warnings already emitted for the latest settled state.
    pub(crate) reported_schema_warnings: HashSet<SchemaWarningKey>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LocalUpdates {
    #[default]
    Immediate,
    Deferred,
}

#[derive(Debug, Clone, Copy)]
enum SubscriptionRowMark {
    Updated,
    Deleted,
    UpdatedAndDeleted,
}

#[derive(Debug, Clone)]
struct SubscriptionVisibilityEffect {
    tables: Vec<String>,
    row_id: ObjectId,
    local_dirty: bool,
    row_mark: SubscriptionRowMark,
    local_row_overlay: bool,
}

#[derive(Debug, Default)]
struct BatchedSubscriptionVisibilityEffects {
    remote_dirty_tables: HashSet<String>,
    local_dirty_tables: HashSet<String>,
    remote_updated: HashMap<String, ahash::AHashSet<ObjectId>>,
    local_updated: HashMap<String, ahash::AHashSet<ObjectId>>,
    remote_deleted: HashMap<String, ahash::AHashSet<ObjectId>>,
    local_deleted: HashMap<String, ahash::AHashSet<ObjectId>>,
}

impl BatchedSubscriptionVisibilityEffects {
    fn push(&mut self, effect: SubscriptionVisibilityEffect) {
        let SubscriptionVisibilityEffect {
            tables,
            row_id,
            local_dirty,
            row_mark,
            local_row_overlay,
        } = effect;

        for table in tables {
            if local_dirty {
                self.local_dirty_tables.insert(table.clone());
            } else {
                self.remote_dirty_tables.insert(table.clone());
            }

            match (row_mark, local_row_overlay) {
                (SubscriptionRowMark::Updated, true) => {
                    self.local_updated.entry(table).or_default().insert(row_id);
                }
                (SubscriptionRowMark::Updated, false) => {
                    self.remote_updated.entry(table).or_default().insert(row_id);
                }
                (SubscriptionRowMark::Deleted, true) => {
                    self.local_deleted.entry(table).or_default().insert(row_id);
                }
                (SubscriptionRowMark::Deleted, false) => {
                    self.remote_deleted.entry(table).or_default().insert(row_id);
                }
                (SubscriptionRowMark::UpdatedAndDeleted, true) => {
                    self.local_updated
                        .entry(table.clone())
                        .or_default()
                        .insert(row_id);
                    self.local_deleted.entry(table).or_default().insert(row_id);
                }
                (SubscriptionRowMark::UpdatedAndDeleted, false) => {
                    self.remote_updated
                        .entry(table.clone())
                        .or_default()
                        .insert(row_id);
                    self.remote_deleted.entry(table).or_default().insert(row_id);
                }
            }
        }
    }

    fn is_empty(&self) -> bool {
        self.remote_dirty_tables.is_empty()
            && self.local_dirty_tables.is_empty()
            && self.remote_updated.is_empty()
            && self.local_updated.is_empty()
            && self.remote_deleted.is_empty()
            && self.local_deleted.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ServerSubscriptionTelemetryGroup {
    #[serde(rename = "groupKey")]
    pub group_key: String,
    pub count: usize,
    pub table: String,
    pub query: String,
    pub branches: Vec<String>,
    pub propagation: QueryPropagation,
}

/// Update for a query subscription.
#[derive(Debug, Clone)]
pub struct QueryUpdate {
    pub subscription_id: QuerySubscriptionId,
    pub delta: RowDelta,
    pub ordered_delta: OrderedRowDelta,
    /// Output descriptor for decoding the binary row data.
    /// This matches the query's output schema (handles JOINs, projections, etc).
    pub descriptor: RowDescriptor,
}

/// Terminal failure for a local query subscription.
#[derive(Debug, Clone)]
pub struct QuerySubscriptionFailure {
    pub subscription_id: QuerySubscriptionId,
    pub code: String,
    pub reason: String,
}

/// State for an active policy check (graphs and associated data).
#[derive(Debug)]
pub(super) struct PolicyCheckState {
    /// Policy graphs that need to settle.
    pub(super) graphs: Vec<PolicyGraph>,
    /// Table name for error messages.
    pub(super) table: TableName,
    /// Branch the write is being evaluated on.
    pub(super) branch: BranchName,
    /// The original pending permission check.
    pub(super) pending_check: PendingPermissionCheck,
}

#[derive(Debug)]
pub(super) struct WriteTableCacheEntry {
    pub(super) descriptor: Arc<RowDescriptor>,
    pub(super) indexed_columns: Option<Arc<Vec<ColumnName>>>,
    pub(super) row_layout: Arc<crate::row_format::CompiledRowLayout>,
    pub(super) row_locator: RowLocator,
    pub(super) insert_policy: Option<Arc<PolicyExpr>>,
    pub(super) update_using_policy: Option<Arc<PolicyExpr>>,
    pub(super) update_check_policy: Option<Arc<PolicyExpr>>,
    pub(super) delete_using_policy: Option<Arc<PolicyExpr>>,
    pub(super) select_policy: Option<Arc<PolicyExpr>>,
}

/// Server-side query subscription state.
///
/// When a client sends a QuerySubscription, the server builds a QueryGraph
/// and tracks contributing ObjectIds. This struct holds that state.
#[derive(Debug)]
pub(super) struct ServerQuerySubscription {
    /// The original query.
    pub(super) query: Query,
    /// Compiled QueryGraph (with client's session for policy filtering).
    pub(super) graph: QueryGraph,
    /// Subscription-specific schema context derived from the downstream client schema.
    pub(super) schema_context: SchemaContext,
    /// Client's session for permission evaluation.
    pub(super) session: Option<Session>,
    /// Resolved branches (from query.branches or schema context at creation time).
    pub(super) branches: Vec<String>,
    /// Extra tables whose rows must be synced so downstream clients can
    /// reproduce bundled policy context locally.
    pub(super) policy_context_tables: Vec<String>,
    /// Durability tier requested by the downstream subscriber. `None` means the
    /// subscriber did not request frontier gating.
    pub(super) required_tier: Option<DurabilityTier>,
    /// Lower-tier settlements are useful as an initial remote scope snapshot,
    /// but repeated below-required settlements should not churn downstream
    /// subscribers that are still waiting for their requested tier.
    pub(super) sent_below_required_settled: bool,
    /// Highest query settlement tier that has actually been emitted downstream.
    ///
    /// Dirty graph passes can leave a server subscription's scope unchanged. In
    /// that case a repeated QuerySettled at the same tier carries no new
    /// information, but large scopes are expensive for clients to decode and
    /// apply.
    pub(super) last_emitted_settled_tier: Option<DurabilityTier>,
    /// Last computed scope (for detecting changes).
    pub(super) last_scope: HashSet<(ObjectId, BranchName)>,
    /// Flag indicating this subscription needs recompilation due to schema change.
    pub(super) needs_recompile: bool,
    /// Flag indicating this server subscription has settled at least once.
    /// Used to emit QuerySettled to the client on first settlement.
    pub(super) settled_once: bool,
    /// Whether this subscription should be propagated to upstream servers.
    pub(super) propagation: QueryPropagation,
    /// Schema mismatch warnings already emitted for the latest settled state.
    pub(super) reported_schema_warnings: HashSet<SchemaWarningKey>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct SchemaWarningKey {
    pub(crate) table_name: String,
    pub(crate) from_hash: SchemaHash,
    pub(crate) to_hash: SchemaHash,
}

impl SchemaWarningKey {
    fn from_warning(warning: &SchemaWarning) -> Self {
        Self {
            table_name: warning.table_name.clone(),
            from_hash: warning.from_hash,
            to_hash: warning.to_hash,
        }
    }
}

#[derive(Debug, Default)]
pub(super) struct SchemaWarningAccumulator {
    counts: HashMap<SchemaWarningKey, usize>,
}

impl SchemaWarningAccumulator {
    pub(super) fn record(&mut self, table_name: &str, from_hash: SchemaHash, to_hash: SchemaHash) {
        let key = SchemaWarningKey {
            table_name: table_name.to_string(),
            from_hash,
            to_hash,
        };
        *self.counts.entry(key).or_default() += 1;
    }

    pub(super) fn warnings_for_query(&self, query_id: QueryId) -> Vec<SchemaWarning> {
        let mut warnings: Vec<SchemaWarning> = self
            .counts
            .iter()
            .map(|(key, row_count)| SchemaWarning {
                query_id,
                table_name: key.table_name.clone(),
                row_count: *row_count,
                from_hash: key.from_hash,
                to_hash: key.to_hash,
            })
            .collect();
        warnings.sort_by(|a, b| {
            a.table_name
                .cmp(&b.table_name)
                .then_with(|| a.from_hash.to_string().cmp(&b.from_hash.to_string()))
                .then_with(|| a.to_hash.to_string().cmp(&b.to_hash.to_string()))
        });
        warnings
    }
}

/// A catalogue object update received via sync.
///
/// Used to pass schema/lens updates from QueryManager to SchemaManager.
#[derive(Debug, Clone)]
pub struct CatalogueUpdate {
    /// The object ID of the catalogue object.
    pub object_id: ObjectId,
    /// Metadata from the object (includes type, app_id, etc.).
    pub metadata: HashMap<String, String>,
    /// Content from the latest commit.
    pub content: Vec<u8>,
}

/// v18 item 6: the key of a settle unit that can stall (registrations never enter the set).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum UnitKey {
    Server(ClientId, QueryId),
    Local(QuerySubscriptionId),
}

/// v18 item 6: a slot of the fair rotation — one per downstream client, plus the local
/// pseudo-client, ordered by client id with `Local` last.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(super) enum RotationSlot {
    Client(ClientId),
    Local,
}

/// v18 item 6: one deadline per lock hold (design v3 § B2). The first unit of the tick
/// always runs; every further unit runs only while the tick is under its budget.
#[derive(Debug)]
pub(super) struct SettleClock {
    started: web_time::Instant,
    budget: Option<std::time::Duration>,
    first_unit_ran: bool,
}

impl SettleClock {
    pub(super) fn new(budget_micros: Option<u64>) -> Self {
        Self {
            started: web_time::Instant::now(),
            budget: budget_micros.map(std::time::Duration::from_micros),
            first_unit_ran: false,
        }
    }

    /// Whether the next charged unit may run. `None` budget: always.
    pub(super) fn may_run_unit(&self) -> bool {
        match self.budget {
            None => true,
            Some(budget) => !self.first_unit_ran || self.started.elapsed() < budget,
        }
    }

    /// A charged unit ran.
    pub(super) fn note_unit_ran(&mut self) {
        self.first_unit_ran = true;
    }

    pub(super) fn is_bounded(&self) -> bool {
        self.budget.is_some()
    }
}

/// Manages reactive SQL queries over storage-backed relational state.
///
/// No global Setup/Ready state machine: indices and rows are loaded lazily from
/// storage. Operations work immediately; queries return empty/Pending results
/// until their required data is available.
pub struct QueryManager {
    pub(super) sync_manager: SyncManager,
    pub(super) schema: Arc<Schema>,
    pub(super) row_policy_mode: RowPolicyMode,
    pub(super) authorization_schema: Option<Arc<Schema>>,
    pub(super) authorization_schema_required: bool,
    pub(super) authorization_context_cache: HashMap<(String, String), Arc<SchemaContext>>,
    /// Cross-tick row-authorization verdicts. See `authz_cache`.
    pub(super) authz_verdicts: super::authz_cache::AuthzVerdictCache,
    /// Bumped on every authorization-schema assignment; part of the verdict-cache
    /// marker so republished permissions invalidate cached verdicts even when the
    /// data-schema hash is unchanged.
    pub(super) authz_schema_generation: u64,

    /// Pending catalogue updates (schemas/lenses received via sync).
    /// SchemaManager should call take_pending_catalogue_updates() to process these.
    pub(super) pending_catalogue_updates: Vec<CatalogueUpdate>,

    /// Active query subscriptions (local)
    pub(super) subscriptions: HashMap<QuerySubscriptionId, QuerySubscription>,
    pub(super) next_subscription_id: u64,

    /// Pending query updates
    pub(super) update_outbox: Vec<QueryUpdate>,

    /// Terminal local subscription failures.
    pub(super) failed_subscriptions: Vec<QuerySubscriptionFailure>,

    /// Active policy checks being evaluated.
    pub(super) active_policy_checks: HashMap<PendingUpdateId, PolicyCheckState>,

    /// Server-side query subscriptions from downstream clients.
    /// Key is (client_id, query_id) to allow multiple queries per client.
    pub(super) server_subscriptions: HashMap<(ClientId, QueryId), ServerQuerySubscription>,
    /// Schema context for multi-schema queries.
    /// Starts empty; initialized via set_current_schema().
    /// Enables lens transforms for rows from old schema branches.
    pub(super) schema_context: SchemaContext,

    /// Maps branch name to schema hash (derived from schema_context).
    /// Used to determine which schema a branch uses.
    pub(super) branch_schema_map: HashMap<String, SchemaHash>,

    /// Buffered row visibility changes for unknown schema branches.
    /// These are retried when new schemas activate via try_activate_pending().
    pub(super) pending_row_visibility_changes: Vec<RowVisibilityChange>,

    /// Latest locally-authored row batch entry per row id.
    ///
    /// Used to let `local_updates = Immediate` queries fall back to the current
    /// local row batch entry when the requested remote durability tier has not been
    /// reached yet.
    pub(super) pending_local_row_batches: HashMap<ObjectId, RowBatchKey>,
    /// Rows this node authored that the REMOTE SCOPE is not yet known to cover.
    ///
    /// Split out of `pending_local_row_batches`, which serves three unrelated roles: the
    /// IndexScan source overlay and the row loader's durability downgrade both ask "does
    /// the durable store lack my write?" and correctly retire on durability, while the
    /// scope filter asks "does the server not know about my write?" and must retire on
    /// evidence from the scope channel. One map, one clear, three retirements was the
    /// design error: it left the scope exemption with no expiry at all for a node whose
    /// reads are one-shot, because the clear only ever fired on an inbound update for the
    /// row and the server never sends a row back to its author.
    ///
    /// Branch-aware, unlike the map it came from: the filter used to compare ids only, so
    /// a row pending on one branch exempted the same id in a query over another.
    pub(super) scope_exempt_local_rows: HashMap<ObjectId, BranchName>,
    /// Rows whose write has reached the settlement target but for which no fresh scope
    /// snapshot has arrived yet. Still exempt — dropping them here would make the row
    /// vanish from a reader holding a scope older than the write, which is worse than the
    /// staleness being fixed.
    ///
    /// They retire when a `QuerySettled` at or above the settlement target is APPLIED in a
    /// later pass than the one that parked them, which is why the pass number is stored
    /// alongside the branch. Two things this is deliberately not:
    ///
    /// * not `remote_query_scope_dirty` — `SyncManager::remove_server` raises that too, and
    ///   losing a server is the opposite of learning a fresh answer; retiring on it would
    ///   blind a node to its own writes exactly when it can least afford it.
    /// * not any settle at any tier — a `Local`-tier snapshot for an unrelated query says
    ///   nothing about a read answered at `GlobalServer`.
    ///
    /// The later-pass requirement is what the two-phase release is for: a settle applied in
    /// the same pass that parked the row would collapse it to a single phase and let the
    /// row blink out of a reader still holding the older scope.
    pub(super) confirmed_local_rows_awaiting_scope: HashMap<ObjectId, ParkedExemption>,

    /// Counts settle passes, so a parked exemption can tell "a scope snapshot arrived
    /// after I was parked" from "one arrived in the same pass".
    pub(super) settle_pass: u64,
    /// v18 item 6: the settle budget per pass in microseconds (`None` = unbounded). A pass
    /// always runs its first unit of work; every further unit runs only while the pass is
    /// under the budget. Set from `JAZZ_SETTLE_BUDGET_MS` by the server binary, or by tests.
    pub(super) settle_budget_micros: Option<u64>,
    /// v18 item 6: the deadline of the tick this pass runs in. Established by the outermost
    /// tick (`begin_tick`) and inherited by every `process` inside it; a standalone `process`
    /// (the gates) begins its own. `None` budget → the clock never expires.
    /// Nobody re-arms for a standalone `process` (a `QueryManager` has no scheduler);
    /// every production `SchemaManager::process` site sits inside `immediate_tick_inner`,
    /// so the clock a standalone pass begins is a gate's, never a host's (diff r2 S8).
    pub(super) tick_clock: Option<SettleClock>,
    /// v18 item 6: set by a pass that deferred a NON-stalled unit for budget (or tripped the
    /// outbox limiter); peeked by `immediate_tick` to schedule a batched tick, taken by
    /// `batched_tick`'s continuation. Recomputed by every pass.
    pub(super) settle_work_remains: bool,
    /// v18 item 6: units that ran without progress since the last un-stalling event for
    /// their key (design v4 § B1, v5 § B1). Ordered last in the pool; never re-arm a tick.
    pub(super) stalled: HashSet<UnitKey>,
    /// v18 item 6: the rotation cursor — the slot served last; the next pass starts after it.
    pub(super) rotation_cursor: Option<RotationSlot>,
    /// v18 item 6: whether the last pass saw a live pending server; the true→false flip
    /// un-stalls every local key (server arrival or `PENDING_SERVER_TIMEOUT` expiry).
    pub(super) had_live_pending_servers: bool,
    /// v18 item 6 (diff r1 S3): the last server — connected or pending — going away frees
    /// every local unit waiting on the frontier; polled once per pass like the flag above.
    pub(super) had_servers: bool,
    /// v18 item 6 (tests only): passes run by this manager — the continuation must not spend
    /// one under a spent clock (design v7).
    #[cfg(any(test, feature = "test"))]
    pub(super) passes: u64,
    /// v18 item 6 (tests only): server settles past the clean short-circuit, per manager —
    /// the global `SUBSCRIPTIONS_SETTLED` cross-talks between parallel tests.
    #[cfg(any(test, feature = "test"))]
    pub(super) server_settles: u64,
    /// v18 item 6 (tests only): charged units the pool dispatched, per manager.
    #[cfg(any(test, feature = "test"))]
    pub(super) pool_units_run: u64,

    /// Per query, the pass in which it last received a settle FROM A SERVER at or above
    /// its own required tier. This is the authority a parked exemption is measured
    /// against — see `record_authoritative_snapshot` for why it cannot be one global
    /// number.
    pub(super) authoritative_snapshot_pass: HashMap<QueryId, u64>,

    /// Visible rows observed through normal row visibility processing, keyed by
    /// batch. Batch fate processing uses this to mark affected query rows
    /// without rescanning and decoding every subscribed visible region.
    pub(super) visible_rows_by_batch: HashMap<BatchId, HashSet<(String, ObjectId)>>,

    /// Authoritative batch fates loaded while settling subscriptions.
    ///
    /// A single replay can ask whether the same batch is transactional and
    /// complete for every subscribed query. Keep that storage fact at manager
    /// scope instead of reloading it once per subscription emission.
    pub(super) authoritative_batch_fate_cache: HashMap<BatchId, Option<BatchFate>>,

    /// Canonical shared allocations for loaded row content, keyed by
    /// (row, batch). Persistent backends deserialize a private copy per
    /// load; without this cache every subscription retains its own copy
    /// of identical bytes. Interior mutability because row loaders are
    /// shared-borrow closures built during settle.
    pub(super) row_bytes_dedup: std::cell::RefCell<super::row_bytes_dedup::RowBytesDedup>,

    /// Currently queued SyncManager batch fates whose query effects have
    /// already been applied by this manager.
    ///
    /// RuntimeCore owns draining the pending fate queue so write waiters and
    /// persisted fate state still see the same events. QueryManager may process
    /// multiple times before that drain happens, so it must only apply the
    /// subscription dirtiness effects for the newly appended suffix.
    pub(super) applied_pending_batch_fates: Vec<BatchFate>,

    /// Known schemas (for server-mode operation).
    /// Synced from SchemaManager's known_schemas to enable lazy branch activation.
    /// When a row arrives with unknown branch, we parse the branch name to extract
    /// the short hash, then look up the full schema in this map.
    pub(super) known_schemas: Arc<HashMap<SchemaHash, Schema>>,

    /// Schema hashes that still need catalogue persistence for the current
    /// storage namespace.
    pub(super) pending_catalogue_schema_hashes: HashSet<SchemaHash>,

    /// Storage namespaces where all live schemas have already been upserted
    /// into the catalogue for this manager.
    pub(super) catalogued_storage_namespaces: HashSet<usize>,

    /// Application id for catalogue schema persistence, when available.
    pub(super) catalogue_app_id: Option<String>,

    /// Per-schema, per-table write metadata cached to avoid cloning policy
    /// trees and descriptors on every hot write.
    pub(super) write_table_cache: HashMap<(SchemaHash, TableName), Arc<WriteTableCacheEntry>>,
}

impl QueryManager {
    fn mark_schema_catalogue_dirty(&mut self, schema_hash: SchemaHash) {
        self.pending_catalogue_schema_hashes.insert(schema_hash);
        self.catalogued_storage_namespaces.clear();
    }

    fn mark_all_live_schemas_catalogue_dirty(&mut self) {
        for schema_hash in self.schema_context.all_live_hashes() {
            self.pending_catalogue_schema_hashes.insert(schema_hash);
        }
        self.catalogued_storage_namespaces.clear();
    }

    pub(super) fn finalize_schema_warnings(
        reported: &mut HashSet<SchemaWarningKey>,
        warnings: Vec<SchemaWarning>,
    ) -> Vec<SchemaWarning> {
        let current_keys: HashSet<SchemaWarningKey> = warnings
            .iter()
            .map(SchemaWarningKey::from_warning)
            .collect();
        let new_warnings = warnings
            .into_iter()
            .filter(|warning| !reported.contains(&SchemaWarningKey::from_warning(warning)))
            .collect();
        *reported = current_keys;
        new_warnings
    }

    pub fn server_subscription_telemetry(&self) -> Vec<ServerSubscriptionTelemetryGroup> {
        let mut groups: HashMap<String, ServerSubscriptionTelemetryGroup> = HashMap::new();

        for subscription in self.server_subscriptions.values() {
            let query = serde_json::to_string(&subscription.query)
                .unwrap_or_else(|_| "{\"error\":\"query serialization failed\"}".to_string());
            let propagation = propagation_label(subscription.propagation);
            let group_key = subscription_group_key(&query, &subscription.branches, propagation);

            groups
                .entry(group_key.clone())
                .and_modify(|group| group.count += 1)
                .or_insert_with(|| ServerSubscriptionTelemetryGroup {
                    group_key,
                    count: 1,
                    table: subscription.query.table.as_str().to_string(),
                    query,
                    branches: subscription.branches.clone(),
                    propagation: subscription.propagation,
                });
        }

        groups.into_values().collect()
    }

    /// v18 item 6 (tests only): the stalled set's size.
    #[cfg(any(test, feature = "test"))]
    pub fn stalled_units_for_test(&self) -> usize {
        self.stalled.len()
    }

    /// v18 item 6 (tests only): the differential's quiescence predicate — a pass with no
    /// pending registration and no non-stalled unit has nothing left to do.
    #[cfg(any(test, feature = "test"))]
    pub fn has_live_units_for_test(&self) -> bool {
        self.sync_manager.has_pending_query_subscriptions()
            || self.server_subscriptions.iter().any(|(key, sub)| {
                Self::server_subscription_is_unit(sub)
                    && !self.stalled.contains(&UnitKey::Server(key.0, key.1))
            })
            || self.subscriptions.iter().any(|(id, sub)| {
                Self::local_subscription_is_unit(sub)
                    && !self.stalled.contains(&UnitKey::Local(*id))
                    // A local subscription whose ONLY reason is `has_pending_local_updates`
                    // is sticky (v5 § B3: a local write that produced no visible delta leaves
                    // the flag set for the life of the subscription): under `None` it is
                    // re-settled to an empty delta every pass, under a budget it runs once
                    // and stalls. Not progress-bearing either way — excluded here so a
                    // quiescence loop can end under `None` too (differential, diff r1 S2).
                    && !Self::local_unit_is_sticky_only(sub)
            })
    }

    /// v18 item 6 (tests only): passes run by this manager.
    #[cfg(any(test, feature = "test"))]
    pub fn passes_for_test(&self) -> u64 {
        self.passes
    }

    /// See `has_live_units_for_test`: a unit by the pending-local-updates flag alone.
    #[cfg(any(test, feature = "test"))]
    fn local_unit_is_sticky_only(subscription: &QuerySubscription) -> bool {
        subscription.has_pending_local_updates
            && subscription.settled_once
            && !subscription.needs_recompile
            && !subscription.needs_visibility_recompute
            && !subscription.graph.has_dirty_nodes()
    }

    /// v18 item 6 (tests only): server settles past the clean short-circuit on this manager.
    #[cfg(any(test, feature = "test"))]
    pub fn server_settles_for_test(&self) -> u64 {
        self.server_settles
    }

    /// v18 item 6 (tests only): charged units the pool dispatched on this manager.
    #[cfg(any(test, feature = "test"))]
    pub fn pool_units_run_for_test(&self) -> u64 {
        self.pool_units_run
    }

    /// v18 item 6 (tests only): whether an authorization schema is set (G6-7(d)'s fixture
    /// asserts it is NOT after `require_authorization_schema`).
    #[cfg(any(test, feature = "test"))]
    pub fn has_authorization_schema_for_test(&self) -> bool {
        self.authorization_schema.is_some()
    }

    /// v18 item 6: `JAZZ_SETTLE_BUDGET_MS` → the budget in microseconds. Unset, empty, `0`
    /// or unparsable → `None` (unbounded, today's behaviour); otherwise `Some(ms · 1000)`,
    /// saturating: a huge value becomes `Duration::from_micros(u64::MAX)` in
    /// `SettleClock::new` (≈ 5.8 × 10⁵ years — unbounded, reached from the other end). It
    /// must stay a saturation, never a wrap (a tiny budget by accident) or a clamp.
    pub fn settle_budget_micros_from_env(raw: Option<&str>) -> Option<u64> {
        let text = raw.map(str::trim).filter(|text| !text.is_empty())?;
        match text.parse::<u64>() {
            Ok(0) => None,
            Ok(millis) => Some(millis.saturating_mul(1000)),
            Err(_) => {
                tracing::warn!(
                    value = text,
                    "JAZZ_SETTLE_BUDGET_MS is not a number; unbounded"
                );
                None
            }
        }
    }

    /// v18 item 6: establish the tick's settle clock if none is running. Returns `true`
    /// when this call created it (the caller then ends it); nested ticks inherit.
    pub fn begin_tick(&mut self) -> bool {
        if self.tick_clock.is_some() {
            return false;
        }
        self.tick_clock = Some(SettleClock::new(self.settle_budget_micros));
        true
    }

    /// v18 item 6: end the clock begun by `begin_tick` (only the creator calls this).
    pub fn end_tick(&mut self) {
        self.tick_clock = None;
    }

    /// v18 item 6: peek — did the last pass leave non-stalled work behind for budget?
    pub fn settle_work_remains(&self) -> bool {
        self.settle_work_remains
    }

    /// Whether the current tick's clock would let another charged unit run: `true` with no
    /// clock or no budget, `false` once the budget is spent (the first unit ran and the
    /// deadline passed). The continuation asks before spending a pass on nothing.
    pub fn settle_clock_has_room(&self) -> bool {
        self.tick_clock
            .as_ref()
            .is_none_or(SettleClock::may_run_unit)
    }

    /// v18 item 6: take the flag (the batched tick's continuation).
    pub fn take_settle_work_remains(&mut self) -> bool {
        std::mem::take(&mut self.settle_work_remains)
    }

    /// v18 item 6: bound the work a settle pass does under the engine lock. `Some(0)` runs
    /// exactly one unit per pass, which is what the gates use to observe the fair order.
    pub fn set_settle_budget_micros(&mut self, micros: Option<u64>) {
        self.settle_budget_micros = micros;
    }

    /// Create a new QueryManager with empty schema context.
    ///
    /// Call `set_current_schema()` to initialize the current schema before queries.
    /// Use `add_live_schema()` and `register_lens()` to add additional schemas.
    ///
    /// Row-level security is evaluated via `process()` which handles pending
    /// permission checks from SyncManager.
    pub fn new(sync_manager: SyncManager) -> Self {
        Self {
            sync_manager,
            schema: Arc::new(Schema::new()),
            row_policy_mode: RowPolicyMode::PermissiveLocal,
            authorization_schema: None,
            authorization_schema_required: false,
            authorization_context_cache: HashMap::new(),
            authz_verdicts: super::authz_cache::AuthzVerdictCache::default(),
            authz_schema_generation: 0,
            pending_catalogue_updates: Vec::new(),
            subscriptions: HashMap::new(),
            next_subscription_id: 0,
            update_outbox: Vec::new(),
            failed_subscriptions: Vec::new(),
            active_policy_checks: HashMap::new(),
            server_subscriptions: HashMap::new(),
            schema_context: SchemaContext::empty(),
            branch_schema_map: HashMap::new(),
            pending_row_visibility_changes: Vec::new(),
            pending_local_row_batches: HashMap::new(),
            scope_exempt_local_rows: HashMap::new(),
            confirmed_local_rows_awaiting_scope: HashMap::new(),
            settle_pass: 0,
            settle_budget_micros: None,
            tick_clock: None,
            settle_work_remains: false,
            stalled: HashSet::new(),
            rotation_cursor: None,
            had_live_pending_servers: false,
            had_servers: false,
            #[cfg(any(test, feature = "test"))]
            passes: 0,
            #[cfg(any(test, feature = "test"))]
            server_settles: 0,
            #[cfg(any(test, feature = "test"))]
            pool_units_run: 0,
            authoritative_snapshot_pass: HashMap::new(),
            visible_rows_by_batch: HashMap::new(),
            authoritative_batch_fate_cache: HashMap::new(),
            row_bytes_dedup: Default::default(),
            applied_pending_batch_fates: Vec::new(),
            known_schemas: Arc::new(HashMap::new()),
            pending_catalogue_schema_hashes: HashSet::new(),
            catalogued_storage_namespaces: HashSet::new(),
            catalogue_app_id: None,
            write_table_cache: HashMap::new(),
        }
    }

    pub fn set_catalogue_app_id(&mut self, app_id: impl Into<String>) {
        self.catalogue_app_id = Some(app_id.into());
        self.catalogued_storage_namespaces.clear();
    }

    /// Set the current schema (the one this client writes to).
    ///
    /// Must be called before queries. Can only be called once.
    /// Creates indices for the current schema's branch.
    pub fn set_current_schema(&mut self, schema: Schema, env: &str, user_branch: &str) {
        let row_policy_mode = if Self::schema_has_any_explicit_policies(&schema) {
            RowPolicyMode::Enforcing
        } else {
            RowPolicyMode::PermissiveLocal
        };
        self.set_current_schema_with_policy_mode(schema, env, user_branch, row_policy_mode);
    }

    pub fn set_current_schema_with_policy_mode(
        &mut self,
        schema: Schema,
        env: &str,
        user_branch: &str,
        row_policy_mode: RowPolicyMode,
    ) {
        self.schema_context
            .set_current(schema.clone(), env, user_branch);
        self.schema = Arc::new(schema.clone());
        self.row_policy_mode = row_policy_mode;
        self.authorization_schema = if matches!(row_policy_mode, RowPolicyMode::Enforcing) {
            Some(Arc::new(schema.clone()))
        } else {
            None
        };
        self.authz_schema_generation += 1;
        self.authorization_context_cache.clear();
        self.authorization_schema_required = false;
        self.write_table_cache.clear();

        // Update branch -> schema hash map
        let branch = self.schema_context.branch_name();
        self.branch_schema_map.insert(
            branch.as_str().to_string(),
            self.schema_context.current_hash,
        );
        self.pending_catalogue_schema_hashes.clear();
        self.mark_schema_catalogue_dirty(self.schema_context.current_hash);
        // v18 item 6: a schema change with a policy mode un-stalls everything (r4 #5).
        self.stalled.clear();
    }

    /// How many row-authorization verdicts were served from the cross-tick cache.
    /// Exposed for tests and diagnostics.
    pub fn authz_cache_hit_count(&self) -> u64 {
        self.authz_verdicts.hit_count()
    }

    /// How many times the authorization schema has been replaced. Each bump
    /// clears the authorization cache and marks EVERY subscription for
    /// recompilation, so this counter is the cost signal for a redundant
    /// permissions-head apply. Exposed for tests and diagnostics.
    pub fn authz_schema_generation(&self) -> u64 {
        self.authz_schema_generation
    }

    pub fn set_authorization_schema(&mut self, schema: Schema) {
        self.authorization_schema = Some(Arc::new(schema));
        self.authz_schema_generation += 1;
        self.authorization_context_cache.clear();
        self.row_policy_mode = RowPolicyMode::Enforcing;
        self.authorization_schema_required = true;
        // v18 item 6: the authorization schema arriving is the un-stalling event for class
        // (i) — and the recompile marking above IS the hook: the recompile step of the next
        // pass un-stalls every key it recompiles (falsified: an explicit clear here changed
        // no gate, the differential's authorization rounds converge through the recompile).
        self.mark_subscriptions_for_recompile();
    }

    pub fn require_authorization_schema(&mut self) {
        self.row_policy_mode = RowPolicyMode::Enforcing;
        self.authorization_schema_required = true;
        self.authorization_context_cache.clear();
    }

    /// Add a live schema (one we can read from but don't write to).
    ///
    /// Creates indices for the schema's branch.
    /// Marks subscriptions for recompilation to include the new branch.
    pub fn add_live_schema(&mut self, schema: Schema) {
        let hash = SchemaHash::compute(&schema);

        // Skip if already live or is current
        if self.schema_context.is_live(&hash) {
            return;
        }

        // Build branch name for this schema
        let branch = ComposedBranchName::new(
            &self.schema_context.env,
            hash,
            &self.schema_context.user_branch,
        )
        .to_branch_name();

        // Add to live_schemas (without lens - caller should register lens separately)
        self.schema_context
            .live_schemas
            .insert(hash, schema.clone());

        // Update branch -> schema hash map
        self.branch_schema_map
            .insert(branch.as_str().to_string(), hash);
        self.mark_schema_catalogue_dirty(hash);

        // A cached authorization context predating this activation stays
        // blind to the new generation and denies its rows (defect 23).
        self.authorization_context_cache.clear();

        // Mark subscriptions for recompile to pick up new branch
        self.mark_subscriptions_for_recompile();
    }

    /// Register a lens between two schemas.
    ///
    /// Also attempts to activate any pending schemas that may now be reachable.
    pub fn register_lens(&mut self, lens: super::super::schema_manager::lens::Lens) {
        self.schema_context.register_lens(lens);
        self.authorization_context_cache.clear();

        // Try to activate pending schemas
        let activated = self.schema_context.try_activate_pending();
        if !activated.is_empty() {
            // New schemas activated - register branches and mark for recompile
            for hash in activated {
                if let Some(_schema) = self.schema_context.live_schemas.get(&hash).cloned() {
                    let branch = ComposedBranchName::new(
                        &self.schema_context.env,
                        hash,
                        &self.schema_context.user_branch,
                    )
                    .to_branch_name();

                    self.branch_schema_map
                        .insert(branch.as_str().to_string(), hash);
                    self.mark_schema_catalogue_dirty(hash);
                }
            }
            self.mark_subscriptions_for_recompile();
        }
    }

    pub(super) fn compile_graph(
        query: &Query,
        schema: &Schema,
        session: Option<Session>,
        schema_context: &SchemaContext,
        row_policy_mode: RowPolicyMode,
    ) -> Result<QueryGraph, QueryCompileError> {
        QueryGraph::try_compile_with_schema_context(
            query,
            schema,
            session,
            schema_context,
            row_policy_mode,
        )
    }

    pub(super) fn local_subscription_uses_explicit_authorization(
        &self,
        session: Option<&Session>,
    ) -> bool {
        session.is_some()
            && self
                .authorization_schema
                .as_ref()
                .map(|auth_schema| auth_schema.as_ref() != self.schema.as_ref())
                .unwrap_or(false)
    }

    pub(super) fn local_subscription_compile_schema(&self, session: Option<&Session>) -> Schema {
        if self.local_subscription_uses_explicit_authorization(session) {
            self.schema
                .iter()
                .map(|(table_name, table_schema)| {
                    let mut structural = table_schema.clone();
                    structural.policies = TablePolicies::default();
                    (*table_name, structural)
                })
                .collect()
        } else {
            self.schema.as_ref().clone()
        }
    }

    pub(crate) fn schema_has_any_explicit_policies(schema: &Schema) -> bool {
        schema
            .values()
            .any(|table_schema| table_schema.policies.has_any_explicit_policy())
    }

    /// Mark all subscriptions for recompilation.
    ///
    /// Called when live schemas change to ensure subscriptions pick up new branches.
    fn mark_subscriptions_for_recompile(&mut self) {
        for sub in self.subscriptions.values_mut() {
            sub.needs_recompile = true;
        }
        for sub in self.server_subscriptions.values_mut() {
            sub.needs_recompile = true;
        }
    }

    pub(super) fn has_stale_subscriptions(&self) -> bool {
        self.subscriptions.values().any(|sub| sub.needs_recompile)
            || self
                .server_subscriptions
                .values()
                .any(|sub| sub.needs_recompile)
    }

    pub(crate) fn ensure_known_schemas_catalogued<H: Storage>(
        &mut self,
        storage: &mut H,
    ) -> Result<(), StorageError> {
        if !self.schema_context.is_initialized() {
            return Ok(());
        }

        let storage_namespace = storage.storage_cache_namespace();
        if !self
            .catalogued_storage_namespaces
            .contains(&storage_namespace)
        {
            self.mark_all_live_schemas_catalogue_dirty();
        }
        if self.pending_catalogue_schema_hashes.is_empty() {
            return Ok(());
        }

        let mut pending_hashes = self
            .pending_catalogue_schema_hashes
            .iter()
            .copied()
            .collect::<Vec<_>>();
        pending_hashes.sort_by_key(|schema_hash| schema_hash.to_string());

        for schema_hash in pending_hashes {
            let Some(schema) = self.schema_context.get_schema(&schema_hash) else {
                self.pending_catalogue_schema_hashes.remove(&schema_hash);
                continue;
            };
            let object_id = schema_hash.to_object_id();
            let mut metadata = storage
                .load_catalogue_entry(object_id)?
                .map(|entry| entry.metadata)
                .unwrap_or_default();
            metadata.insert(
                MetadataKey::Type.to_string(),
                ObjectType::CatalogueSchema.to_string(),
            );
            metadata.insert(MetadataKey::SchemaHash.to_string(), schema_hash.to_string());
            if let Some(app_id) = &self.catalogue_app_id {
                metadata.insert(MetadataKey::AppId.to_string(), app_id.clone());
            }
            storage.upsert_catalogue_entry(&CatalogueEntry {
                object_id,
                metadata,
                content: encode_schema(schema),
            })?;
            self.pending_catalogue_schema_hashes.remove(&schema_hash);
        }

        self.catalogued_storage_namespaces.insert(storage_namespace);
        Ok(())
    }

    /// Recompile subscriptions that are marked as stale.
    ///
    /// Called during process() to rebuild QueryGraphs when schemas change.
    fn recompile_stale_subscriptions(&mut self) {
        if !self.has_stale_subscriptions() {
            return;
        }

        let mut failed_local: Vec<(QuerySubscriptionId, String)> = Vec::new();
        let current_schema = self.schema.clone();
        let current_schema_context = self.schema_context.clone();
        let authorization_schema = self.authorization_schema.clone();
        // v18 item 6: a recompile marks its key un-stalled (design v5 § B1). Under `None`
        // the set is always empty (nothing outside the pool inserts) — skip the walks (r1 S7).
        if !self.stalled.is_empty() {
            for (sub_id, sub) in &self.subscriptions {
                if sub.needs_recompile {
                    self.stalled.remove(&UnitKey::Local(*sub_id));
                }
            }
            for (key, sub) in &self.server_subscriptions {
                if sub.needs_recompile {
                    self.stalled.remove(&UnitKey::Server(key.0, key.1));
                }
            }
        }

        // Recompile local subscriptions
        for (sub_id, sub) in &mut self.subscriptions {
            if sub.needs_recompile {
                // Resolve next branches from current schema context.
                let next_branches: Vec<String> = current_schema_context
                    .all_branch_names()
                    .into_iter()
                    .map(|b| b.as_str().to_string())
                    .collect();
                let uses_explicit_authorization_filtering = sub.session.is_some()
                    && authorization_schema
                        .as_ref()
                        .map(|auth_schema| auth_schema.as_ref() != current_schema.as_ref())
                        .unwrap_or(false);
                let compile_schema = if uses_explicit_authorization_filtering {
                    current_schema
                        .iter()
                        .map(|(table_name, table_schema)| {
                            let mut structural = table_schema.clone();
                            structural.policies = TablePolicies::default();
                            (*table_name, structural)
                        })
                        .collect()
                } else {
                    current_schema.as_ref().clone()
                };

                // Recompile the graph
                let compile_row_policy_mode = if uses_explicit_authorization_filtering {
                    RowPolicyMode::PermissiveLocal
                } else {
                    self.row_policy_mode
                };
                match Self::compile_graph(
                    &sub.query,
                    &compile_schema,
                    sub.session.clone(),
                    &current_schema_context,
                    compile_row_policy_mode,
                ) {
                    Ok(new_graph) => {
                        let policy_context_tables =
                            Self::policy_context_tables_for_graph(&new_graph);
                        sub.graph = new_graph;
                        sub.branches = next_branches;
                        sub.policy_context_tables = policy_context_tables;
                        sub.uses_explicit_authorization_filtering =
                            uses_explicit_authorization_filtering;
                        sub.needs_recompile = false;
                    }
                    Err(err) => {
                        let reason = err.to_string();
                        tracing::error!(
                            sub_id = sub_id.0,
                            table = %sub.graph.table,
                            error = %reason,
                            "subscription stale recompile failed; dropping subscription"
                        );
                        failed_local.push((*sub_id, reason));
                    }
                }
            }
        }

        for (sub_id, reason) in failed_local {
            self.stalled.remove(&UnitKey::Local(sub_id));
            let propagation = self
                .subscriptions
                .remove(&sub_id)
                .map(|sub| sub.propagation)
                .unwrap_or(QueryPropagation::Full);
            self.failed_subscriptions.push(QuerySubscriptionFailure {
                subscription_id: sub_id,
                code: "query_recompile_failed".to_string(),
                reason: reason.clone(),
            });
            if propagation == QueryPropagation::Full {
                // Keep upstream state in sync for subscriptions created via subscribe_with_sync.
                self.sync_manager
                    .send_query_unsubscription_to_servers(QueryId(sub_id.0));
            }
        }

        let mut failed_server: Vec<(ClientId, QueryId, String, String, QueryPropagation)> =
            Vec::new();

        // Recompile server-side subscriptions
        let stale_server_subscription_keys: Vec<_> = self
            .server_subscriptions
            .iter()
            .filter_map(|(key, sub)| sub.needs_recompile.then_some(*key))
            .collect();

        for (client_id, query_id) in stale_server_subscription_keys {
            let Some((query, session, propagation)) = self
                .server_subscriptions
                .get(&(client_id, query_id))
                .map(|sub| (sub.query.clone(), sub.session.clone(), sub.propagation))
            else {
                continue;
            };

            let Some((schema_for_compile, subscription_context)) =
                self.build_server_subscription_context(&query)
            else {
                let reason = "schema context unavailable for query recompile".to_string();
                tracing::error!(
                    %client_id,
                    query_id = query_id.0,
                    error = %reason,
                    "server subscription stale recompile failed; dropping subscription"
                );
                failed_server.push((
                    client_id,
                    query_id,
                    "query_recompile_failed".to_string(),
                    reason,
                    propagation,
                ));
                continue;
            };

            let query_for_compile = Self::query_for_server_compile(&query, &subscription_context);
            let compile_schema: Schema = schema_for_compile
                .iter()
                .map(|(table_name, table_schema)| {
                    let mut structural = table_schema.clone();
                    structural.policies = TablePolicies::default();
                    (*table_name, structural)
                })
                .collect();

            // Recompile the graph
            match Self::compile_graph(
                &query_for_compile,
                &compile_schema,
                session,
                &subscription_context,
                RowPolicyMode::PermissiveLocal,
            ) {
                Ok(new_graph) => {
                    let branches = Self::resolved_server_query_branches(
                        &query_for_compile,
                        &subscription_context,
                    );
                    if let Some(sub) = self.server_subscriptions.get_mut(&(client_id, query_id)) {
                        sub.schema_context = subscription_context;
                        sub.branches = branches;
                        sub.graph = new_graph;
                        sub.needs_recompile = false;
                    }
                }
                Err(err) => {
                    let reason = err.to_string();
                    tracing::error!(
                        %client_id,
                        query_id = query_id.0,
                        error = %reason,
                        "server subscription stale recompile failed; dropping subscription"
                    );
                    failed_server.push((
                        client_id,
                        query_id,
                        "query_recompile_failed".to_string(),
                        reason,
                        propagation,
                    ));
                }
            }
        }

        for (client_id, query_id, code, reason, propagation) in failed_server {
            self.stalled.remove(&UnitKey::Server(client_id, query_id));
            if self
                .server_subscriptions
                .remove(&(client_id, query_id))
                .is_some()
            {
                tracing::info!(
                    client_id = %client_id,
                    query_id = query_id.0,
                    total = self.server_subscriptions.len(),
                    "server subscription removed"
                );
            }
            self.sync_manager
                .drop_client_query_subscription(client_id, query_id);
            self.sync_manager.forget_client_query(client_id, query_id);
            if propagation == QueryPropagation::Full {
                self.sync_manager
                    .send_query_unsubscription_to_servers(query_id);
            }
            self.sync_manager.emit_query_subscription_rejected(
                client_id,
                query_id,
                code,
                format!(
                    "query recompilation failed for query_id {}: {}",
                    query_id.0, reason
                ),
            );
        }
    }

    /// Get the schema context.
    pub fn schema_context(&self) -> &SchemaContext {
        &self.schema_context
    }

    fn process_pending_query_rejections(&mut self) {
        for rejection in self.sync_manager.take_pending_query_rejections() {
            let sub_id = QuerySubscriptionId(rejection.query_id.0);
            if !self.subscriptions.contains_key(&sub_id) {
                tracing::warn!(
                    sub_id = sub_id.0,
                    code = %rejection.code,
                    error = %rejection.reason,
                    "received rejection for unknown local subscription"
                );
                continue;
            }

            self.unsubscribe_with_sync(sub_id);
            self.failed_subscriptions.push(QuerySubscriptionFailure {
                subscription_id: sub_id,
                code: rejection.code,
                reason: rejection.reason,
            });
        }
    }

    /// Get the current branch name for writes.
    ///
    /// Returns the branch for the current schema, or "main" if context isn't initialized.
    pub(super) fn current_branch(&self) -> String {
        if self.schema_context.is_initialized() {
            self.schema_context.branch_name().as_str().to_string()
        } else {
            "main".to_string()
        }
    }

    /// Get all branches to query for a table (current + live schemas).
    pub fn all_query_branches(&self) -> Vec<String> {
        self.schema_context
            .all_branch_names()
            .into_iter()
            .map(|b| b.as_str().to_string())
            .collect()
    }

    /// Get the underlying SyncManager.
    pub fn sync_manager(&self) -> &SyncManager {
        &self.sync_manager
    }

    /// Get mutable reference to the underlying SyncManager.
    pub fn sync_manager_mut(&mut self) -> &mut SyncManager {
        &mut self.sync_manager
    }

    /// Record that this query has an answer it can trust, and let its parked exemptions go.
    ///
    /// Phase two of the two-phase release, and it is deliberately PER QUERY. Two earlier
    /// shapes of this were global and both went permanently inert, for different reasons:
    ///
    /// * against `SyncManager::settlement_target()` — that answers "how durable must MY
    ///   WRITE be", is `GlobalServer` for any node with a server attached whatever that
    ///   server can attest, and a settle carries the EMITTING server's tier. Behind an edge
    ///   the comparison never holds and nothing ever retires.
    /// * against the maximum tier across all readers — one `GlobalServer` subscription
    ///   anywhere in the process then pins every `EdgeServer` reader into permanent
    ///   exemption, although an `EdgeServer` snapshot is fully authoritative for that
    ///   reader. Same inertness, triggered by subscription mix instead of topology.
    ///
    /// The quantity is not global at all. An exemption is consumed by ONE subscription, in
    /// `filter_synced_query_scope_tuples`, against `remote_query_scope_at_least(query_id,
    /// subscription.durability_tier)` — so authority is recorded here per query, and the
    /// filter compares it to the pass a row was parked in. No bar to pick, no cross-query
    /// premature loss, and nothing to fan out.
    fn record_authoritative_snapshot(&mut self, query_id: QueryId, tier: DurabilityTier) {
        let sub_id = QuerySubscriptionId(query_id.0);
        let Some(subscription) = self.subscriptions.get(&sub_id) else {
            return;
        };
        // A reader with no tier is filtered against the untiered scope union, for which any
        // settle is authoritative.
        if subscription
            .durability_tier
            .is_some_and(|required| tier < required)
        {
            return;
        }
        let previous = self
            .authoritative_snapshot_pass
            .insert(query_id, self.settle_pass);
        // The filter's answer for this subscription's parked rows just changed. Nothing
        // else will notice — the scope itself may be identical to the last one — so say so
        // here rather than leaving the row to disappear at whatever unrelated event dirties
        // this subscription next.
        if previous != Some(self.settle_pass)
            && !self.confirmed_local_rows_awaiting_scope.is_empty()
            && let Some(subscription) = self.subscriptions.get_mut(&sub_id)
        {
            subscription.needs_visibility_recompute = true;
        }
    }

    /// Drop parked exemptions that every live scope-filtered reader has already passed.
    ///
    /// Pure bookkeeping: the filter decides per subscription, so an entry left here changes
    /// no answer. It exists so the map does not accumulate one entry per write for the
    /// process lifetime — the shape of leak this whole family was split to end. A
    /// subscription that has no authoritative snapshot yet holds everything back, because
    /// its exemptions are still live.
    fn evict_exemptions_every_reader_has_passed(&mut self) {
        if self.confirmed_local_rows_awaiting_scope.is_empty() {
            return;
        }
        // No scope-filtered reader is NOT the same proposition as "an answer accounting for
        // the write has arrived", and only the second may retire an exemption. Falling out
        // of the `u64::MAX` sentinel below would wipe the whole map on the first, which is
        // reachable between one-shot reads and inverts the defect: the author would be
        // unable to read a write it had already waited on at global tier, and a uniqueness
        // check would report free what is taken.
        let mut has_scope_filtered_reader = false;
        let mut passed_by_every_reader = u64::MAX;
        for (sub_id, subscription) in &self.subscriptions {
            if !subscription.sync_backed {
                continue;
            }
            has_scope_filtered_reader = true;
            let pass = self
                .authoritative_snapshot_pass
                .get(&QueryId(sub_id.0))
                .copied()
                .unwrap_or(0);
            passed_by_every_reader = passed_by_every_reader.min(pass);
        }
        if !has_scope_filtered_reader || passed_by_every_reader == 0 {
            return;
        }
        self.confirmed_local_rows_awaiting_scope
            .retain(|_, parked| parked.parked_in_pass >= passed_by_every_reader);
    }

    /// `from_server` distinguishes a settle a SERVER sent us from one a downstream CLIENT
    /// relayed (`sync_manager::inbox` pushes both into the same queue, and the client
    /// branch copies the tier straight off the peer's payload with no validation). Only
    /// the former is evidence about how authoritative anyone's answer is; letting a peer's
    /// number retire our exemptions would hand a downstream client control over what this
    /// node may read of its own writes.
    pub(crate) fn apply_query_settled(
        &mut self,
        query_id: QueryId,
        tier: DurabilityTier,
        from_server: bool,
    ) {
        if from_server {
            self.record_authoritative_snapshot(query_id, tier);
        }
        let sub_id = QuerySubscriptionId(query_id.0);
        // v18 item 6: a settle from above un-stalls the frontier-waiting local unit.
        self.stalled.remove(&UnitKey::Local(sub_id));
        if let Some(sub) = self.subscriptions.get_mut(&sub_id) {
            let was_unsatisfied = !Self::subscription_query_frontier_satisfied(sub);
            let before = sub.query_frontier_settled_tier;
            let required_tier = sub.durability_tier;
            sub.query_frontier_settled_tier = Some(
                sub.query_frontier_settled_tier
                    .map_or(tier, |current| current.max(tier)),
            );
            let now_satisfied = Self::subscription_query_frontier_satisfied(sub);
            tracing::trace!(
                query_id = query_id.0,
                tier = ?tier,
                required_tier = ?required_tier,
                before = ?before,
                after = ?sub.query_frontier_settled_tier,
                was_unsatisfied,
                now_satisfied,
                "jazz trace query settled applied"
            );
            if was_unsatisfied && now_satisfied {
                sub.needs_visibility_recompute = true;
            }
        }
    }

    pub(crate) fn subscription_query_frontier_satisfied(sub: &QuerySubscription) -> bool {
        match sub.durability_tier {
            Some(required_tier) => sub
                .query_frontier_settled_tier
                .is_some_and(|settled_tier| settled_tier >= required_tier),
            None => true,
        }
    }

    pub(crate) fn mark_subscriptions_visibility_recompute_for_batch(&mut self, batch_id: BatchId) {
        for subscription in self.subscriptions.values_mut() {
            if subscription
                .graph
                .current_output_tuples_ref()
                .iter()
                .any(|tuple| tuple.batch_provenance().contains(&batch_id))
            {
                subscription.needs_visibility_recompute = true;
            }
        }
    }

    fn apply_pending_batch_fate_effects<H: Storage>(&mut self, storage: &H) {
        let pending_batch_fates = self.sync_manager.pending_batch_fates();
        let already_applied_count =
            if pending_batch_fates.starts_with(&self.applied_pending_batch_fates) {
                self.applied_pending_batch_fates.len()
            } else {
                // RuntimeCore may have drained the queue and SyncManager may have
                // appended new fates before QueryManager gets another process pass.
                // In that case, the current queue is a new sequence, not a suffix
                // of the old one.
                0
            };

        let batch_fates = pending_batch_fates[already_applied_count..].to_vec();
        self.applied_pending_batch_fates = pending_batch_fates.to_vec();
        let fate_count = batch_fates.len();
        if fate_count == 0 {
            return;
        }

        let max_confirmed_tier = batch_fates
            .iter()
            .filter_map(BatchFate::confirmed_tier)
            .max();
        if let Some(confirmed_tier) = max_confirmed_tier {
            self.mark_subscriptions_visibility_recompute_for_tier(confirmed_tier);
        }

        // Phase one of releasing the scope exemption: a write that has reached the
        // settlement target no longer needs to be shielded from the server's answer on its
        // own merits. It is only PARKED here, not dropped — see the drain in `process`,
        // which retires it once a fresh scope snapshot has actually arrived.
        let settlement_target = self.sync_manager.settlement_target();
        for fate in &batch_fates {
            if fate
                .confirmed_tier()
                .is_none_or(|tier| tier < settlement_target)
            {
                continue;
            }
            let batch_id = fate.batch_id();
            let confirmed_rows: Vec<ObjectId> = self
                .scope_exempt_local_rows
                .keys()
                .copied()
                .filter(|object_id| {
                    self.pending_local_row_batches
                        .get(object_id)
                        .is_some_and(|key| key.batch_id == batch_id)
                })
                .collect();
            for object_id in confirmed_rows {
                if let Some(branch) = self.scope_exempt_local_rows.remove(&object_id) {
                    self.confirmed_local_rows_awaiting_scope.insert(
                        object_id,
                        ParkedExemption {
                            branch,
                            parked_in_pass: self.settle_pass,
                        },
                    );
                }
            }
        }

        let mut batch_ids = batch_fates
            .iter()
            .map(BatchFate::batch_id)
            .collect::<Vec<_>>();
        batch_ids.sort();
        batch_ids.dedup();

        let unique_batch_count = batch_ids.len();
        let mut marked_row_count = 0usize;
        for batch_id in batch_ids {
            self.authoritative_batch_fate_cache.insert(
                batch_id,
                storage
                    .load_authoritative_batch_fate(batch_id)
                    .ok()
                    .flatten(),
            );
            self.mark_subscriptions_visibility_recompute_for_batch(batch_id);
            let mut rows = self
                .visible_rows_by_batch
                .get(&batch_id)
                .cloned()
                .unwrap_or_default();
            if let Ok(Some(record)) = storage.load_local_batch_record(batch_id) {
                for member in record.members {
                    rows.insert((member.table_name, member.object_id));
                }
            }
            marked_row_count += rows.len();
            for (table_name, object_id) in rows {
                self.mark_local_row_updated_in_subscriptions(table_name.as_str(), object_id);
            }
        }
        tracing::trace!(
            fate_count,
            unique_batch_count,
            marked_row_count,
            max_confirmed_tier = ?max_confirmed_tier,
            "jazz trace batch fate effects applied"
        );
    }

    pub(crate) fn mark_subscriptions_visibility_recompute_for_tier(
        &mut self,
        confirmed_tier: DurabilityTier,
    ) {
        for subscription in self.subscriptions.values_mut() {
            if subscription
                .durability_tier
                .is_some_and(|required_tier| confirmed_tier >= required_tier)
            {
                subscription.needs_visibility_recompute = true;
            }
        }
    }

    /// Remove a client and all its server-side state (subscriptions, in-flight policy checks).
    ///
    /// Returns `false` if the client has unprocessed inbox entries.
    /// The caller should retry later.
    pub fn remove_client(&mut self, client_id: ClientId) -> bool {
        if !self.sync_manager.remove_client(client_id) {
            return false;
        }
        self.server_subscriptions
            .retain(|&(cid, _), _| cid != client_id);
        self.active_policy_checks
            .retain(|_, state| state.pending_check.client_id != client_id);
        // v18 item 6: purge the client's stalled keys with its subscriptions.
        self.stalled
            .retain(|key| !matches!(key, UnitKey::Server(owner, _) if *owner == client_id));
        true
    }

    /// Get the schema.
    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// Get subscription results as decoded rows with ObjectIds (for testing).
    /// Process pending changes and settle all subscription graphs.
    ///
    /// This method drives async progress:
    /// - Processes SyncManager inbox (receives client writes)
    /// - Evaluates pending permission checks
    /// - Settles policy graphs and finalizes completed checks
    /// - Processes object updates from SyncManager
    /// - Flushes pending index updates when indices become ready
    /// - Marks subscriptions with pending IDs dirty when rows become available
    /// - Settles all subscription graphs (row data loaded on-demand from storage)
    pub fn process<H: Storage>(&mut self, storage: &mut H) {
        let _span = tracing::trace_span!("QueryManager::process").entered();
        // Settle-cost accounting: this call IS the settle pass. The guard
        // snapshots the counters here and emits one line on drop if the pass
        // ran longer than `JAZZ_SETTLE_LOG_MS`.
        let _settle_cost = super::settle_cost::SettlePass::begin();
        self.settle_pass = self.settle_pass.wrapping_add(1);
        // v18 item 6: a standalone `process` (no tick around it) begins its own clock. The
        // clock is released even if the pass panics (diff r2 S7): a clock left set would
        // make every later `begin_tick` look nested, and no unit would run under a budget
        // again on a host that survives a caught panic.
        // Only the owner catches (diff r4 S1): in production every `process` runs inside
        // `immediate_tick_inner`, so this is the always-nested path and the landing pad
        // would be pure cost on the very pass this item budgets; a nested pass has no clock
        // to release and its panic reaches the owner's catch above it.
        let owns_clock = self.begin_tick();
        let outcome = if owns_clock {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.process_pass(storage)))
        } else {
            self.process_pass(storage);
            Ok(())
        };
        if owns_clock {
            self.end_tick();
        }
        if let Err(payload) = outcome {
            std::panic::resume_unwind(payload);
        }
    }

    /// The body of one settle pass; `process` wraps it with the accounting guard and the
    /// clock ownership.
    fn process_pass<H: Storage>(&mut self, storage: &mut H) {
        #[cfg(any(test, feature = "test"))]
        {
            self.passes += 1;
        }
        // v18 item 6: the un-stalling events polled once per pass (design v5 § B1; r4 #3, #4):
        // a client's role actually changed → its server units; the last live pending server
        // arrived or timed out → every local unit. The flag is recomputed by every pass.
        for client_id in self.sync_manager.take_role_changes() {
            self.stalled
                .retain(|key| !matches!(key, UnitKey::Server(owner, _) if *owner == client_id));
        }
        let live_pending_servers = self.sync_manager.has_live_pending_servers();
        if self.had_live_pending_servers && !live_pending_servers {
            self.stalled.retain(|key| !matches!(key, UnitKey::Local(_)));
        }
        self.had_live_pending_servers = live_pending_servers;
        let servers = self.sync_manager.has_servers_or_pending_servers();
        if self.had_servers && !servers {
            self.stalled.retain(|key| !matches!(key, UnitKey::Local(_)));
        }
        self.had_servers = servers;
        self.settle_work_remains = false;
        let bounded = self
            .tick_clock
            .as_ref()
            .is_some_and(SettleClock::is_bounded);

        if let Err(error) = self.ensure_known_schemas_catalogued(storage) {
            tracing::warn!(%error, "failed to persist known schemas to catalogue storage");
        }

        // 1. Process SyncManager inbox (receives client writes)
        self.sync_manager.process_inbox(storage);
        let remote_scope_dirty_query_ids = self.sync_manager.take_remote_query_scope_dirty();
        for query_id in remote_scope_dirty_query_ids {
            if let Some(sub) = self.subscriptions.get_mut(&QuerySubscriptionId(query_id.0)) {
                sub.needs_visibility_recompute = true;
                // New work for the unit (r1 S3): a stalled key must be able to run it.
                self.stalled
                    .remove(&UnitKey::Local(QuerySubscriptionId(query_id.0)));
            }
        }
        self.pending_catalogue_updates.extend(
            self.sync_manager
                .take_pending_catalogue_updates()
                .into_iter()
                .map(|entry| CatalogueUpdate {
                    object_id: entry.object_id,
                    metadata: entry.metadata,
                    content: entry.content,
                }),
        );

        // 2. Process row visibility changes from SyncManager FIRST so indices are current
        // before subscriptions are processed.
        let mut row_visibility_changes = std::mem::take(&mut self.pending_row_visibility_changes);
        row_visibility_changes.extend(self.sync_manager.take_pending_row_visibility_changes());
        if !row_visibility_changes.is_empty() {
            tracing::debug!(
                count = row_visibility_changes.len(),
                "processing row visibility changes"
            );
        }
        self.handle_row_updates_batched(storage, row_visibility_changes);

        // 3. Process pending query unsubscriptions from downstream clients
        // before new subscriptions from the same tick. One-shot query helpers
        // often unsubscribe and immediately resubscribe; draining removals
        // first prevents stale per-client scope from suppressing the replay
        // that the new subscription depends on.
        self.process_pending_query_unsubscriptions();

        // 3b. Process pending query subscriptions from downstream clients
        // (after indices are updated, so initial settle finds existing data).
        // v18 item 6: under a bounded budget the registrations join the one unit pool
        // dispatched at step 8 (design v4 SF1); under `None` this pass is today's.
        if !bounded {
            self.process_pending_query_subscriptions(storage);
        }

        // 4. Pick up new permission check intents from SyncManager
        self.pick_up_pending_permission_checks(storage);

        // 4b. Settle policy graphs and finalize completed checks
        self.settle_policy_checks(storage);

        let post_permission_row_visibility_changes =
            self.sync_manager.take_pending_row_visibility_changes();
        if !post_permission_row_visibility_changes.is_empty() {
            tracing::debug!(
                count = post_permission_row_visibility_changes.len(),
                "processing row visibility changes from accepted permission checks"
            );
        }
        self.handle_row_updates_batched(storage, post_permission_row_visibility_changes);
        self.apply_pending_batch_fate_effects(storage);

        // 4c. Apply QuerySettled messages that do not depend on any earlier
        // sequenced sync updates. Watermarked settlements stay queued for
        // RuntimeCore, which tracks per-server stream progress.
        let pending_query_settled = self.sync_manager.take_pending_query_settled();
        if !pending_query_settled.is_empty() {
            let mut blocked = Vec::new();
            for pending_settled in pending_query_settled {
                if pending_settled.through_seq == 0 {
                    if let Some(server_id) = pending_settled.server_id {
                        self.sync_manager.relay_query_settled_to_origins(
                            server_id,
                            pending_settled.query_id,
                            pending_settled.tier,
                        );
                    }
                    self.apply_query_settled(
                        pending_settled.query_id,
                        pending_settled.tier,
                        pending_settled.server_id.is_some(),
                    );
                } else {
                    blocked.push(pending_settled);
                }
            }
            if !blocked.is_empty() {
                self.sync_manager.requeue_pending_query_settled(blocked);
            }
        }

        self.evict_exemptions_every_reader_has_passed();
        self.process_pending_query_rejections();

        // 5. Index storage is handled by Storage via batched_tick() - not here.
        // Tests/benchmarks that don't need real storage use NullStorage.

        // 6. Recompile any subscriptions marked as stale due to schema changes
        self.recompile_stale_subscriptions();

        // 7. Settle all subscriptions - row_loader reads from subscription's branches
        // Extract references to avoid borrowing self in the closure
        // v18 item 6: under a bounded budget the local units are part of the pool (step 8).
        if bounded {
            self.dispatch_unit_pool(storage);
            return;
        }
        let dirty_count = self
            .subscriptions
            .values()
            .filter(|s| s.graph.has_dirty_nodes())
            .count();
        if dirty_count > 0 {
            tracing::debug!(
                dirty_count,
                total = self.subscriptions.len(),
                "settling subscriptions"
            );
        }
        let storage_ref: &dyn Storage = storage;
        let subscription_ids: Vec<_> = self.subscriptions.keys().copied().collect();

        for sub_id in subscription_ids {
            let should_process_subscription = self
                .subscriptions
                .get(&sub_id)
                .is_some_and(Self::local_subscription_is_unit);
            if !should_process_subscription {
                continue;
            }
            self.settle_one_local_subscription(storage_ref, sub_id);
        }

        // Note: With sync storage, object loading is immediate. No need to request
        // async loads - objects are available when we query for them.

        // 8. Settle server-side subscriptions and update scopes
        self.settle_server_subscriptions(storage_ref);
    }

    /// Whether a local subscription is a settle unit (design v4 § B1): the loop predicate.
    pub(super) fn local_subscription_is_unit(subscription: &QuerySubscription) -> bool {
        subscription.needs_recompile
            || !subscription.settled_once
            || subscription.needs_visibility_recompute
            || subscription.has_pending_local_updates
            || subscription.graph.has_dirty_nodes()
    }

    /// One (L) unit. Extracted from the local settle loop unchanged (v18 item 6).
    pub(super) fn settle_one_local_subscription(
        &mut self,
        storage_ref: &dyn Storage,
        sub_id: QuerySubscriptionId,
    ) {
        let Some(mut subscription) = self.subscriptions.remove(&sub_id) else {
            return;
        };

        let _sub_span = tracing::trace_span!("settle_subscription", sub_id = sub_id.0, table = %subscription.graph.table).entered();
        let branches = subscription.branches.clone();
        let table = subscription.graph.table.as_str().to_string();
        let mut schema_warnings = SchemaWarningAccumulator::default();
        let include_deleted = subscription.query.include_deleted;
        let local_durability_satisfies_subscription = subscription
            .durability_tier
            .is_some_and(|tier| self.sync_manager.has_local_durability_at_least(tier));
        let remote_scope_satisfies_subscription = subscription.durability_tier.is_none_or(|tier| {
            self.sync_manager
                .has_remote_query_scope_snapshot_at_least(QueryId(sub_id.0), tier)
        });

        // Settle-cost accounting: this subscription is past every
        // short-circuit above, so it is about to do real settle work — the
        // clock read sits next to a graph settle, never next to a skip.
        let settle_started = web_time::Instant::now();
        let delta = {
            let schema_context = &self.schema_context;
            let branch_schema_map = &self.branch_schema_map;
            let row_bytes_dedup = &self.row_bytes_dedup;
            let row_loader = |id: ObjectId, table_hint: Option<TableName>| -> Option<LoadedRow> {
                let lacks_authoritative_remote_scope = subscription.sync_backed
                    && subscription.local_updates == LocalUpdates::Immediate
                    && !remote_scope_satisfies_subscription;
                let durability_tier = if lacks_authoritative_remote_scope
                    || (subscription.local_updates == LocalUpdates::Immediate
                        && subscription.pending_local_row_ids.contains(&id))
                {
                    None
                } else {
                    subscription.durability_tier
                };
                let local_pending_version = if !subscription.local_overlay_rows.is_empty() {
                    subscription.local_overlay_rows.get(&id).copied()
                } else {
                    (subscription.local_updates == LocalUpdates::Immediate)
                        .then(|| self.pending_local_row_batches.get(&id).copied())
                        .flatten()
                };
                Self::load_visible_row_for_query(
                    storage_ref,
                    id,
                    table_hint.as_ref().map(TableName::as_str),
                    &branches,
                    durability_tier,
                    local_pending_version,
                    !subscription.local_overlay_rows.is_empty(),
                    !subscription.local_overlay_rows.is_empty()
                        || (subscription.sync_backed
                            && subscription.durability_tier.is_some()
                            && subscription.local_updates == LocalUpdates::Immediate),
                    include_deleted,
                    schema_context,
                    branch_schema_map,
                    &table,
                    sub_id,
                    &mut schema_warnings,
                    row_bytes_dedup,
                )
            };

            // This map is QueryManager-global and table-blind: every entry in it is
            // walked, and its full row bytes read, by every full IndexScanNode rescan
            // in every qualifying subscription. Its length is therefore a multiplier
            // on a settle's storage reads, and it is the number that says whether a
            // settle reading hundreds of megabytes does so because the overlay never
            // drained.
            crate::query_manager::settle_cost::set_gauge(
                &crate::query_manager::settle_cost::PENDING_LOCAL_ROW_BATCHES,
                self.pending_local_row_batches.len() as u64,
            );
            let source_overlay_rows = if !subscription.local_overlay_rows.is_empty() {
                Some(&subscription.local_overlay_rows)
            } else if subscription.local_updates == LocalUpdates::Immediate
                && subscription.sync_backed
                && subscription.durability_tier.is_some()
                && !self.pending_local_row_batches.is_empty()
            {
                Some(&self.pending_local_row_batches)
            } else {
                None
            };
            subscription.graph.settle_with_source_overlay(
                storage_ref,
                source_overlay_rows,
                row_loader,
            )
        };
        super::settle_cost::bump(&super::settle_cost::SUBSCRIPTIONS_SETTLED);
        super::settle_cost::add(
            &super::settle_cost::ROWS_EMITTED,
            (delta.added.len() + delta.removed.len() + delta.updated.len()) as u64,
        );
        // Local subscriptions have no downstream client; the query id alone
        // identifies them.
        super::settle_cost::note_subscription_settle(None, sub_id.0, settle_started.elapsed());
        subscription.needs_visibility_recompute = false;
        let new_schema_warnings = Self::finalize_schema_warnings(
            &mut subscription.reported_schema_warnings,
            schema_warnings.warnings_for_query(QueryId(sub_id.0)),
        );
        for warning in &new_schema_warnings {
            crate::sync_manager::log_schema_warning(warning, None, Some(sub_id.0));
        }
        if !delta.added.is_empty() || !delta.removed.is_empty() {
            tracing::debug!(
                sub_id = sub_id.0,
                added = delta.added.len(),
                removed = delta.removed.len(),
                "settle delta"
            );
        }

        if !subscription.settled_once
            && !Self::subscription_query_frontier_satisfied(&subscription)
            && self.sync_manager.has_servers_or_pending_servers()
        {
            // Graph state updated by settle(), but don't deliver until the
            // initial upstream frontier has been replayed — or until every
            // still-pending server has exceeded PENDING_SERVER_TIMEOUT,
            // which means nothing upstream is going to replay.
            tracing::trace!(
                sub_id = sub_id.0,
                table = %table,
                required_tier = ?subscription.durability_tier,
                settled_tier = ?subscription.query_frontier_settled_tier,
                dirty = subscription.graph.has_dirty_nodes(),
                needs_recompile = subscription.needs_recompile,
                needs_visibility_recompute = subscription.needs_visibility_recompute,
                pending_local_updates = subscription.has_pending_local_updates,
                has_servers_or_pending_servers = self.sync_manager.has_servers_or_pending_servers(),
                "jazz trace subscription waiting for initial frontier"
            );
            self.subscriptions.insert(sub_id, subscription);
            return;
        }

        let mut visible_tuples = if subscription.uses_explicit_authorization_filtering {
            let auth_schema_context = self.schema_context.clone();
            let auth_branch_schema_map = self.branch_schema_map.clone();
            let mut settlement_eval_cache = SettlementEvalCache::default();
            Cow::Owned(self.authorized_tuples_from_graph_with_cache(
                storage_ref,
                &mut settlement_eval_cache,
                &subscription.graph,
                &auth_schema_context,
                &auth_branch_schema_map,
                subscription.session.as_ref(),
            ))
        } else {
            Cow::Borrowed(subscription.graph.current_output_tuples_ref())
        };

        if subscription.sync_backed
            && Self::subscription_query_frontier_satisfied(&subscription)
            && (!local_durability_satisfies_subscription || remote_scope_satisfies_subscription)
            && self
                .sync_manager
                .has_remote_query_scope_snapshot(QueryId(sub_id.0))
            && (subscription.propagation == QueryPropagation::Full
                || !self.sync_manager.has_durability_identity())
        {
            visible_tuples = Cow::Owned(
                self.filter_synced_query_scope_tuples(
                    QueryId(sub_id.0),
                    subscription.durability_tier,
                    LocalWriteAuthority {
                        overlay: (!subscription.local_overlay_rows.is_empty())
                            .then_some(&subscription.local_overlay_rows),
                        process_wide: subscription.local_updates == LocalUpdates::Immediate,
                    },
                    visible_tuples.into_owned(),
                ),
            );
        }

        visible_tuples =
            self.filter_transaction_visible_tuples(storage_ref, QueryId(sub_id.0), visible_tuples);

        if !subscription.settled_once {
            let visible_rows = Self::rows_from_tuples(&subscription.graph, visible_tuples.as_ref());
            let row_count = visible_rows.len();
            let ordered_ids_after: Vec<_> = visible_rows.iter().map(|row| row.id).collect();
            let ordered_delta = OrderedRowDelta {
                added: visible_rows
                    .iter()
                    .cloned()
                    .enumerate()
                    .map(|(index, row)| OrderedAdded {
                        id: row.id,
                        index,
                        row,
                    })
                    .collect(),
                removed: Vec::new(),
                updated: Vec::new(),
                pending: false,
            };
            let visible_rows_by_id: HashMap<_, _> = visible_rows
                .iter()
                .cloned()
                .map(|row| (row.id, row))
                .collect();
            let visible_delta = RowDelta {
                added: visible_rows,
                removed: Vec::new(),
                moved: Vec::new(),
                updated: Vec::new(),
            };
            tracing::trace!(
                sub_id = sub_id.0,
                table = %table,
                rows = row_count,
                added = visible_delta.added.len(),
                settled_tier = ?subscription.query_frontier_settled_tier,
                required_tier = ?subscription.durability_tier,
                "jazz trace subscription first delivery"
            );
            subscription.settled_once = true;
            subscription.current_ordered_ids = ordered_ids_after;
            subscription.current_visible_rows = visible_rows_by_id;
            self.update_outbox.push(QueryUpdate {
                subscription_id: sub_id,
                delta: visible_delta,
                ordered_delta,
                descriptor: subscription.graph.combined_descriptor.clone(),
            });
            subscription.has_pending_local_updates = false;
            subscription
                .pending_local_row_ids
                .retain(|id| self.pending_local_row_batches.contains_key(id));
        } else {
            let visible_rows = Self::rows_from_tuples(&subscription.graph, visible_tuples.as_ref());
            let visible_rows_by_id: HashMap<_, _> = visible_rows
                .iter()
                .cloned()
                .map(|row| (row.id, row))
                .collect();
            let visible_delta = Self::row_delta_from_rows(
                &subscription.current_visible_rows,
                &subscription.current_ordered_ids,
                &visible_rows,
            );
            if visible_delta.is_empty() {
                self.subscriptions.insert(sub_id, subscription);
                return;
            }
            let ordered_ids_after: Vec<ObjectId> = visible_rows.iter().map(|row| row.id).collect();
            let ordered = build_ordered_delta_with_post_ids(
                &subscription.current_ordered_ids,
                &ordered_ids_after,
                &visible_delta,
                false,
            );
            subscription.current_ordered_ids = ordered.ordered_ids_after;
            subscription.current_visible_rows = visible_rows_by_id;
            tracing::debug!(
                sub_id = sub_id.0,
                added = visible_delta.added.len(),
                removed = visible_delta.removed.len(),
                updated = visible_delta.updated.len(),
                "incremental delivery"
            );
            self.update_outbox.push(QueryUpdate {
                subscription_id: sub_id,
                delta: visible_delta,
                ordered_delta: ordered.delta,
                descriptor: subscription.graph.combined_descriptor.clone(),
            });
            subscription.has_pending_local_updates = false;
            subscription
                .pending_local_row_ids
                .retain(|id| self.pending_local_row_batches.contains_key(id));
        }

        self.subscriptions.insert(sub_id, subscription);
    }

    pub(super) fn handle_row_update_with_origin(
        &mut self,
        storage: &mut dyn Storage,
        update: RowVisibilityChange,
        local_update: bool,
        apply_index_mutations: bool,
    ) {
        if let Some(effect) = self.prepare_row_update_with_origin(
            storage,
            update,
            local_update,
            apply_index_mutations,
        ) {
            self.apply_subscription_visibility_effect(effect);
        }
    }

    fn prepare_row_update_with_origin(
        &mut self,
        storage: &mut dyn Storage,
        update: RowVisibilityChange,
        local_update: bool,
        apply_index_mutations: bool,
    ) -> Option<SubscriptionVisibilityEffect> {
        let original_table = update.row_locator.table.to_string();
        let branch = update.row.branch.as_str();
        let origin_schema_hash = update.row_locator.origin_schema_hash;

        let schema_hash = match self.branch_schema_map.get(branch) {
            Some(&hash) => hash,
            None => {
                let branch_name = BranchName::new(branch);
                if let Some(composed) = ComposedBranchName::parse(&branch_name) {
                    if let Some(full_hash) = self.find_schema_by_short_hash(&composed.schema_hash) {
                        self.branch_schema_map.insert(branch.to_string(), full_hash);
                        full_hash
                    } else {
                        tracing::error!(
                            object_id = %update.object_id,
                            branch = %branch,
                            schema_hash = %composed.schema_hash.short(),
                            local_update,
                            "buffering row update for unknown schema hash; schema not yet known"
                        );
                        self.pending_row_visibility_changes.push(update);
                        return None;
                    }
                } else {
                    tracing::error!(
                        object_id = %update.object_id,
                        branch = %branch,
                        local_update,
                        "buffering row update for unknown branch; cannot parse schema hash"
                    );
                    self.pending_row_visibility_changes.push(update);
                    return None;
                }
            }
        };

        let logical_table = resolve_current_table_name(
            &self.schema_context,
            &original_table,
            origin_schema_hash.as_ref(),
        )
        .unwrap_or_else(|| original_table.to_string());
        let branch_table = if schema_hash == self.schema_context.current_hash {
            logical_table.clone()
        } else {
            translate_table_name_to_schema(&self.schema_context, &logical_table, &schema_hash)
                .unwrap_or_else(|| original_table.to_string())
        };
        let subscription_tables =
            self.subscription_table_aliases(&logical_table, &original_table, &branch_table);
        let table_name = TableName::new(&branch_table);

        let table_schema = if schema_hash == self.schema_context.current_hash {
            match self.schema.get(&table_name) {
                Some(schema) => schema.clone(),
                None => return None,
            }
        } else if let Some(schema) = self.schema_context.get_schema(&schema_hash) {
            match schema.get(&table_name) {
                Some(table_schema) => table_schema.clone(),
                None => return None,
            }
        } else if let Some(schema) = self.known_schemas.get(&schema_hash) {
            match schema.get(&table_name) {
                Some(table_schema) => table_schema.clone(),
                None => return None,
            }
        } else {
            tracing::error!(
                object_id = %update.object_id,
                branch = %branch,
                schema_hash = %schema_hash.short(),
                "buffering row update because schema for branch is not available yet"
            );
            self.pending_row_visibility_changes.push(update);
            return None;
        };

        let descriptor = table_schema.columns.clone();
        let old_row = update.previous_row.as_ref();
        let current_batch_id = update.row.batch_id;
        let current_row_key = RowBatchKey::from_row(&update.row);

        if let Some(previous_row) = old_row
            && previous_row.state.is_visible()
            && let Some(rows) = self.visible_rows_by_batch.get_mut(&previous_row.batch_id)
        {
            rows.remove(&(logical_table.clone(), update.object_id));
            if rows.is_empty() {
                self.visible_rows_by_batch.remove(&previous_row.batch_id);
            }
        }
        if update.row.state.is_visible() {
            let rows = self
                .visible_rows_by_batch
                .entry(current_batch_id)
                .or_default();
            rows.insert((logical_table.clone(), update.object_id));
        }

        if local_update {
            self.pending_local_row_batches
                .insert(update.object_id, current_row_key);
            self.scope_exempt_local_rows.insert(
                update.object_id,
                BranchName::new(update.row.branch.as_str()),
            );
            self.confirmed_local_rows_awaiting_scope
                .remove(&update.object_id);
        } else if let Some(pending_row_key) = self
            .pending_local_row_batches
            .get(&update.object_id)
            .copied()
            && (pending_row_key.branch_name.as_str() == update.row.branch.as_str())
            && (pending_row_key.batch_id != current_batch_id
                || update.row.confirmed_tier == Some(DurabilityTier::GlobalServer))
        {
            self.retire_local_row_tracking(update.object_id);
        }

        if self.visible_row_is_hard_deleted(storage, update.object_id, &update.row.branch)
            && !update.row.is_hard_deleted()
        {
            return None;
        }

        if update.row.is_hard_deleted() {
            if apply_index_mutations {
                let old_data = old_row.map(|row| row.data.as_ref());
                let _ = Self::update_indices_for_hard_delete_on_branch(
                    storage,
                    &branch_table,
                    branch,
                    update.object_id,
                    old_data,
                    &descriptor,
                    table_schema.indexed_columns.as_deref(),
                );
            }
            return Some(SubscriptionVisibilityEffect {
                tables: subscription_tables,
                row_id: update.object_id,
                local_dirty: local_update,
                row_mark: SubscriptionRowMark::Deleted,
                local_row_overlay: local_update,
            });
        }

        if update.row.is_soft_deleted() {
            if apply_index_mutations {
                if let Some(old_row) = old_row {
                    let _ = Self::update_indices_for_soft_delete_on_branch(
                        storage,
                        &branch_table,
                        branch,
                        update.object_id,
                        &old_row.data,
                        &descriptor,
                        table_schema.indexed_columns.as_deref(),
                    );
                } else {
                    let _ = storage.index_remove(
                        &branch_table,
                        "_id",
                        branch,
                        &Value::Uuid(update.object_id),
                        update.object_id,
                    );
                    if let Err(error) = storage.index_insert(
                        &branch_table,
                        "_id_deleted",
                        branch,
                        &Value::Uuid(update.object_id),
                        update.object_id,
                    ) {
                        tracing::error!(
                            table = branch_table,
                            branch,
                            object_id = %update.object_id,
                            %error,
                            "failed to insert synced _id_deleted index entry"
                        );
                    }
                }
            }
            return Some(SubscriptionVisibilityEffect {
                tables: subscription_tables,
                row_id: update.object_id,
                local_dirty: local_update,
                row_mark: SubscriptionRowMark::Deleted,
                local_row_overlay: false,
            });
        }

        let was_soft_deleted = old_row.is_some_and(StoredRowBatch::is_soft_deleted);
        let new_data = &update.row.data;

        if was_soft_deleted {
            if apply_index_mutations
                && let Err(error) = Self::update_indices_for_restore_on_branch(
                    storage,
                    &branch_table,
                    branch,
                    update.object_id,
                    new_data,
                    &descriptor,
                    table_schema.indexed_columns.as_deref(),
                )
            {
                tracing::error!(
                    table = branch_table,
                    branch,
                    object_id = %update.object_id,
                    %error,
                    "failed to update indices for synced restore"
                );
            }
            return Some(SubscriptionVisibilityEffect {
                tables: subscription_tables,
                row_id: update.object_id,
                local_dirty: local_update,
                row_mark: SubscriptionRowMark::Updated,
                local_row_overlay: local_update,
            });
        }

        if old_row.is_none() {
            if apply_index_mutations
                && let Err(error) = Self::update_indices_for_insert_on_branch(
                    storage,
                    &branch_table,
                    branch,
                    update.object_id,
                    new_data,
                    &descriptor,
                    table_schema.indexed_columns.as_deref(),
                )
            {
                tracing::error!(
                    table = branch_table,
                    branch,
                    object_id = %update.object_id,
                    index_column = error.column.as_str(),
                    error = %error.source,
                    "failed to update indices for synced insert"
                );
            }
        } else if let Some(old_row) = old_row
            && apply_index_mutations
            && let Err(error) = Self::update_indices_for_update_on_branch(
                storage,
                super::indices::BranchIndexTarget {
                    table: &branch_table,
                    branch,
                    descriptor: &descriptor,
                    indexed_columns: table_schema.indexed_columns.as_deref(),
                },
                update.object_id,
                &old_row.data,
                new_data,
            )
        {
            tracing::error!(
                table = branch_table,
                branch,
                object_id = %update.object_id,
                %error,
                "failed to update indices for synced update"
            );
        }

        if local_update {
            Some(SubscriptionVisibilityEffect {
                tables: subscription_tables,
                row_id: update.object_id,
                local_dirty: true,
                row_mark: SubscriptionRowMark::Updated,
                local_row_overlay: true,
            })
        } else {
            let self_referential_table_update = old_row.is_some()
                && table_schema.columns.columns.iter().any(|column| {
                    column.references.as_ref().is_some_and(|referenced| {
                        referenced.as_str() == logical_table.as_str()
                            || referenced.as_str() == original_table.as_str()
                            || referenced.as_str() == branch_table.as_str()
                    })
                });
            if self_referential_table_update
                || Self::select_policy_columns_changed(
                    table_schema.policies.select_policy(),
                    &table_name,
                    &descriptor,
                    old_row.map(|row| row.data.as_ref()),
                    new_data,
                )
            {
                return Some(SubscriptionVisibilityEffect {
                    tables: subscription_tables,
                    row_id: update.object_id,
                    local_dirty: false,
                    row_mark: SubscriptionRowMark::UpdatedAndDeleted,
                    local_row_overlay: false,
                });
            }
            Some(SubscriptionVisibilityEffect {
                tables: subscription_tables,
                row_id: update.object_id,
                local_dirty: false,
                row_mark: SubscriptionRowMark::Updated,
                local_row_overlay: false,
            })
        }
    }

    fn subscription_table_aliases(
        &self,
        logical_table: &str,
        original_table: &str,
        branch_table: &str,
    ) -> Vec<String> {
        let mut aliases = HashSet::from([
            logical_table.to_string(),
            original_table.to_string(),
            branch_table.to_string(),
        ]);

        for schema_hash in self.schema_context.all_live_hashes() {
            if let Some(table) =
                translate_table_name_to_schema(&self.schema_context, logical_table, &schema_hash)
            {
                aliases.insert(table);
            }
        }

        let mut aliases = aliases.into_iter().collect::<Vec<_>>();
        aliases.sort();
        aliases
    }

    fn select_policy_columns_changed(
        policy: Option<&PolicyExpr>,
        table_name: &TableName,
        descriptor: &RowDescriptor,
        old_data: Option<&[u8]>,
        new_data: &[u8],
    ) -> bool {
        let Some(policy) = policy else {
            return false;
        };
        let Some(old_data) = old_data else {
            return !matches!(policy, PolicyExpr::True);
        };
        let columns = Self::policy_local_columns(policy);
        let Ok(old_values) = decode_row(descriptor, old_data) else {
            return true;
        };
        let Ok(new_values) = decode_row(descriptor, new_data) else {
            return true;
        };
        if descriptor
            .columns
            .iter()
            .enumerate()
            .any(|(index, column)| {
                column
                    .references
                    .as_ref()
                    .is_some_and(|referenced| referenced == table_name)
                    && old_values.get(index) != new_values.get(index)
            })
        {
            return true;
        }
        if columns.is_empty() {
            return false;
        }

        columns.into_iter().any(|column| {
            descriptor
                .column_index(&column)
                .is_some_and(|index| old_values.get(index) != new_values.get(index))
        })
    }

    fn policy_local_columns(policy: &PolicyExpr) -> HashSet<String> {
        let mut columns = HashSet::new();
        Self::collect_policy_local_columns(policy, &mut columns);
        columns
    }

    fn collect_policy_local_columns(policy: &PolicyExpr, columns: &mut HashSet<String>) {
        match policy {
            PolicyExpr::Cmp { column, .. }
            | PolicyExpr::IsNull { column }
            | PolicyExpr::IsNotNull { column }
            | PolicyExpr::Contains { column, .. }
            | PolicyExpr::In { column, .. }
            | PolicyExpr::InList { column, .. }
            | PolicyExpr::Inherits {
                via_column: column, ..
            } => {
                columns.insert(column.clone());
            }
            PolicyExpr::And(exprs) | PolicyExpr::Or(exprs) => {
                for expr in exprs {
                    Self::collect_policy_local_columns(expr, columns);
                }
            }
            PolicyExpr::Not(expr)
            | PolicyExpr::Exists {
                condition: expr, ..
            } => {
                Self::collect_policy_local_columns(expr, columns);
            }
            PolicyExpr::SessionCmp { .. }
            | PolicyExpr::SessionIsNull { .. }
            | PolicyExpr::SessionIsNotNull { .. }
            | PolicyExpr::SessionContains { .. }
            | PolicyExpr::SessionInList { .. }
            | PolicyExpr::ExistsRel { .. }
            | PolicyExpr::InheritsReferencing { .. }
            | PolicyExpr::True
            | PolicyExpr::False => {}
        }
    }

    pub(crate) fn handle_row_update(
        &mut self,
        storage: &mut dyn Storage,
        update: RowVisibilityChange,
    ) {
        self.handle_row_update_with_origin(storage, update, false, true);
    }

    fn handle_row_updates_batched(
        &mut self,
        storage: &mut dyn Storage,
        updates: Vec<RowVisibilityChange>,
    ) {
        if updates.is_empty() {
            return;
        }

        let mut effects = BatchedSubscriptionVisibilityEffects::default();
        for update in updates {
            if let Some(effect) = self.prepare_row_update_with_origin(storage, update, false, true)
            {
                effects.push(effect);
            }
        }
        self.apply_batched_subscription_visibility_effects(effects);
    }

    fn apply_subscription_visibility_effect(&mut self, effect: SubscriptionVisibilityEffect) {
        let mut effects = BatchedSubscriptionVisibilityEffects::default();
        effects.push(effect);
        self.apply_batched_subscription_visibility_effects(effects);
    }

    fn apply_batched_subscription_visibility_effects(
        &mut self,
        effects: BatchedSubscriptionVisibilityEffects,
    ) {
        if effects.is_empty() {
            return;
        }

        // Authorization verdicts must drop before any settle can read them: for the
        // changed rows themselves, and for every row whose table's policy reads one of
        // the changed tables.
        self.authz_verdicts.invalidate(
            effects
                .remote_dirty_tables
                .iter()
                .chain(effects.local_dirty_tables.iter())
                .map(String::as_str),
            effects
                .remote_updated
                .values()
                .chain(effects.local_updated.values())
                .chain(effects.remote_deleted.values())
                .chain(effects.local_deleted.values())
                .flat_map(|ids| ids.iter().copied()),
        );

        // Every effect that marks a table dirty also records its row id in one of the
        // updated/deleted maps (see `BatchedSubscriptionVisibilityEffects::push`), so
        // dirty tables can be delivered to the scans row-precisely — the next settle
        // re-evaluates only the changed rows instead of rescanning the whole index.
        // Should the row coverage be missing anyway, fall back to the table-level full
        // rescan rather than risk a stale scan.
        fn changed_rows_for(
            table: &str,
            updated: &HashMap<String, ahash::AHashSet<ObjectId>>,
            deleted: &HashMap<String, ahash::AHashSet<ObjectId>>,
        ) -> ahash::AHashSet<ObjectId> {
            let mut rows = updated.get(table).cloned().unwrap_or_default();
            if let Some(ids) = deleted.get(table) {
                rows.extend(ids.iter().copied());
            }
            rows
        }

        for table in &effects.remote_dirty_tables {
            let rows = changed_rows_for(table, &effects.remote_updated, &effects.remote_deleted);
            if rows.is_empty() {
                self.mark_subscriptions_dirty(table);
            } else {
                self.mark_subscriptions_rows_changed(table, &rows, false);
            }
        }
        for table in &effects.local_dirty_tables {
            let rows = changed_rows_for(table, &effects.local_updated, &effects.local_deleted);
            if rows.is_empty() {
                self.mark_subscriptions_dirty_local(table);
            } else {
                self.mark_subscriptions_rows_changed(table, &rows, true);
            }
        }

        for (table, ids) in &effects.remote_updated {
            self.mark_rows_updated_in_subscriptions(table, ids, false);
        }
        for (table, ids) in &effects.local_updated {
            self.mark_rows_updated_in_subscriptions(table, ids, true);
        }
        for (table, ids) in &effects.remote_deleted {
            self.mark_rows_deleted_in_subscriptions(table, ids, false);
        }
        for (table, ids) in &effects.local_deleted {
            self.mark_rows_deleted_in_subscriptions(table, ids, true);
        }
    }

    /// Mark subscriptions dirty for a table based on update origin.
    fn mark_subscriptions_dirty_with_origin(&mut self, table: &str, local_update: bool) {
        // Mark local subscriptions dirty
        for (sub_id, subscription) in self.subscriptions.iter_mut() {
            if Self::subscription_involves_table(&subscription.graph, table) {
                subscription.graph.mark_dirty_for_table(table);
                if local_update {
                    subscription.has_pending_local_updates = true;
                    // v18 item 6: a local write is an un-stalling event for its unit (r4 #6).
                    self.stalled.remove(&UnitKey::Local(*sub_id));
                }
            }
        }

        // Mark server subscriptions dirty (for downstream clients)
        for server_sub in self.server_subscriptions.values_mut() {
            if Self::subscription_involves_table(&server_sub.graph, table) {
                server_sub.graph.mark_dirty_for_table(table);
            }
        }
    }

    /// Row-precise sibling of [`Self::mark_subscriptions_dirty_with_origin`].
    fn mark_subscriptions_rows_changed(
        &mut self,
        table: &str,
        rows: &ahash::AHashSet<ObjectId>,
        local_update: bool,
    ) {
        for (sub_id, subscription) in self.subscriptions.iter_mut() {
            if Self::subscription_involves_table(&subscription.graph, table) {
                subscription.graph.mark_rows_changed_for_table(table, rows);
                if local_update {
                    subscription.has_pending_local_updates = true;
                    self.stalled.remove(&UnitKey::Local(*sub_id));
                }
            }
        }

        for server_sub in self.server_subscriptions.values_mut() {
            if Self::subscription_involves_table(&server_sub.graph, table) {
                server_sub.graph.mark_rows_changed_for_table(table, rows);
            }
        }
    }

    /// Mark subscriptions dirty from external updates (default behavior).
    ///
    /// Checks all tables involved in the subscription (including joined tables).
    /// Also marks server-side subscriptions for downstream clients.
    pub(super) fn mark_subscriptions_dirty(&mut self, table: &str) {
        self.mark_subscriptions_dirty_with_origin(table, false);
    }

    /// Mark subscriptions dirty from local writes.
    pub(super) fn mark_subscriptions_dirty_local(&mut self, table: &str) {
        self.mark_subscriptions_dirty_with_origin(table, true);
    }

    fn mark_rows_updated_in_subscriptions(
        &mut self,
        table: &str,
        ids: &ahash::AHashSet<ObjectId>,
        local_overlay: bool,
    ) {
        for subscription in self.subscriptions.values_mut() {
            if Self::subscription_involves_table(&subscription.graph, table) {
                subscription.graph.mark_rows_updated(table, ids);
                if local_overlay {
                    subscription
                        .pending_local_row_ids
                        .extend(ids.iter().copied());
                }
            }
        }
        for server_sub in self.server_subscriptions.values_mut() {
            if Self::subscription_involves_table(&server_sub.graph, table) {
                server_sub.graph.mark_rows_updated(table, ids);
            }
        }
    }

    pub(crate) fn mark_local_row_updated_in_subscriptions(&mut self, table: &str, id: ObjectId) {
        let ids = ahash::AHashSet::from_iter([id]);
        for subscription in self.subscriptions.values_mut() {
            if Self::subscription_involves_table(&subscription.graph, table) {
                subscription.graph.mark_rows_updated(table, &ids);
                subscription.pending_local_row_ids.insert(id);
            }
        }
        for server_sub in self.server_subscriptions.values_mut() {
            if Self::subscription_involves_table(&server_sub.graph, table) {
                server_sub.graph.mark_rows_updated(table, &ids);
            }
        }
    }

    fn mark_rows_deleted_in_subscriptions(
        &mut self,
        table: &str,
        ids: &ahash::AHashSet<ObjectId>,
        local_overlay: bool,
    ) {
        for subscription in self.subscriptions.values_mut() {
            if Self::subscription_involves_table(&subscription.graph, table) {
                subscription.graph.mark_rows_deleted(table, ids);
                if local_overlay {
                    subscription
                        .pending_local_row_ids
                        .extend(ids.iter().copied());
                }
            }
        }
        for server_sub in self.server_subscriptions.values_mut() {
            if Self::subscription_involves_table(&server_sub.graph, table) {
                server_sub.graph.mark_rows_deleted(table, ids);
            }
        }
    }

    pub(super) fn mark_local_row_deleted_in_subscriptions(&mut self, table: &str, id: ObjectId) {
        let ids = ahash::AHashSet::from_iter([id]);
        for subscription in self.subscriptions.values_mut() {
            if Self::subscription_involves_table(&subscription.graph, table) {
                subscription.graph.mark_rows_deleted(table, &ids);
                subscription.pending_local_row_ids.insert(id);
            }
        }
        for server_sub in self.server_subscriptions.values_mut() {
            if Self::subscription_involves_table(&server_sub.graph, table) {
                server_sub.graph.mark_rows_deleted(table, &ids);
            }
        }
    }

    /// What the three local-write maps currently say about one row. Exists so a gate can
    /// assert the family retires together — the whole point of `retire_local_row_tracking`.
    #[cfg(test)]
    pub(crate) fn local_row_tracking(&self, id: ObjectId) -> LocalRowTracking {
        LocalRowTracking {
            pending_batch: self.pending_local_row_batches.contains_key(&id),
            scope_exempt: self.scope_exempt_local_rows.contains_key(&id),
            awaiting_scope: self.confirmed_local_rows_awaiting_scope.contains_key(&id),
        }
    }

    /// Drop every trace of a local write from the three maps that track one.
    ///
    /// They are separate because they answer separate questions (source overlay,
    /// durability downgrade, scope exemption) and retire on different terms — but a row
    /// that is going away entirely must leave all of them together. Splitting the scope
    /// exemption out of `pending_local_row_batches` and forgetting one removal site is
    /// exactly how a stale exemption outlives the write it stood for, which is the defect
    /// this family was split to fix. Route every removal through here.
    pub(super) fn retire_local_row_tracking(&mut self, id: ObjectId) {
        self.pending_local_row_batches.remove(&id);
        self.scope_exempt_local_rows.remove(&id);
        self.confirmed_local_rows_awaiting_scope.remove(&id);
    }

    pub(crate) fn clear_local_pending_row_overlay(&mut self, table: &str, id: ObjectId) {
        self.retire_local_row_tracking(id);
        self.mark_subscriptions_dirty_local(table);
        self.mark_local_row_updated_in_subscriptions(table, id);
    }

    fn load_row_locator(storage: &dyn Storage, row_id: ObjectId) -> Option<RowLocator> {
        storage.load_row_locator(row_id).ok().flatten()
    }

    pub(super) fn load_best_visible_row_batch(
        &self,
        storage: &dyn Storage,
        row_id: ObjectId,
        branches: &[String],
        durability_tier: Option<DurabilityTier>,
        schema_context: &SchemaContext,
        branch_schema_map: &HashMap<String, SchemaHash>,
    ) -> Option<(String, QueryRowBatch)> {
        Self::load_best_visible_row_batch_from_storage(
            storage,
            row_id,
            branches,
            durability_tier,
            schema_context,
            branch_schema_map,
        )
    }

    pub(super) fn load_best_visible_row_batch_from_storage(
        storage: &dyn Storage,
        row_id: ObjectId,
        branches: &[String],
        durability_tier: Option<DurabilityTier>,
        schema_context: &SchemaContext,
        branch_schema_map: &HashMap<String, SchemaHash>,
    ) -> Option<(String, QueryRowBatch)> {
        let locator = Self::load_row_locator(storage, row_id)?;
        Self::load_best_visible_row_batch_from_storage_with_locator(
            storage,
            row_id,
            &locator,
            branches,
            durability_tier,
            schema_context,
            branch_schema_map,
        )
    }

    fn branch_schema_hash_for_visible_load(
        branch: &str,
        schema_context: &SchemaContext,
        branch_schema_map: &HashMap<String, SchemaHash>,
    ) -> Option<SchemaHash> {
        branch_schema_map
            .get(branch)
            .copied()
            .or_else(|| {
                (branch == schema_context.branch_name().as_str())
                    .then_some(schema_context.current_hash)
            })
            .or_else(|| {
                ComposedBranchName::parse(&BranchName::new(branch)).and_then(|composed| {
                    if composed.schema_hash.short() == schema_context.current_hash.short() {
                        Some(schema_context.current_hash)
                    } else {
                        schema_context
                            .live_schemas
                            .keys()
                            .copied()
                            .find(|hash| hash.short() == composed.schema_hash.short())
                    }
                })
            })
    }

    fn load_visible_query_row_from_candidate_tables(
        storage: &dyn Storage,
        primary_table: &str,
        fallback_table: Option<&str>,
        branch: &str,
        row_id: ObjectId,
        durability_tier: Option<DurabilityTier>,
    ) -> Option<QueryRowBatch> {
        let load = |table: &str| match durability_tier {
            Some(required_tier) => {
                storage.load_visible_query_row_for_tier(table, branch, row_id, required_tier)
            }
            None => storage.load_visible_query_row(table, branch, row_id),
        };

        load(primary_table).ok().flatten().or_else(|| {
            fallback_table
                .filter(|fallback| *fallback != primary_table)
                .and_then(|fallback| load(fallback).ok().flatten())
        })
    }

    fn load_local_pending_query_row_from_candidate_tables(
        storage: &dyn Storage,
        primary_table: &str,
        fallback_table: Option<&str>,
        row_batch_key: RowBatchKey,
    ) -> Option<QueryRowBatch> {
        let load = |table: &str| {
            storage.load_history_query_row_batch(
                table,
                row_batch_key.branch_name.as_str(),
                row_batch_key.row_id,
                row_batch_key.batch_id,
            )
        };

        load(primary_table).ok().flatten().or_else(|| {
            fallback_table
                .filter(|fallback| *fallback != primary_table)
                .and_then(|fallback| load(fallback).ok().flatten())
        })
    }

    fn load_local_pending_query_row_with_hint_or_locator(
        storage: &dyn Storage,
        row_batch_key: RowBatchKey,
        table_hint: Option<&str>,
        schema_context: &SchemaContext,
    ) -> Option<(String, QueryRowBatch)> {
        if let Some(hint) = table_hint
            && let Some(row) = Self::load_local_pending_query_row_from_candidate_tables(
                storage,
                hint,
                None,
                row_batch_key,
            )
        {
            return Some((hint.to_string(), row));
        }

        let locator = Self::load_row_locator(storage, row_batch_key.row_id)?;
        let original_table = locator.table.as_str();
        let current_table = locator
            .origin_schema_hash
            .filter(|hash| *hash != schema_context.current_hash)
            .and_then(|origin_schema_hash| {
                resolve_current_table_name(
                    schema_context,
                    original_table,
                    Some(&origin_schema_hash),
                )
            })
            .filter(|translated| translated != original_table);
        let current_table_name = current_table.as_deref().unwrap_or(original_table);
        let row = Self::load_local_pending_query_row_from_candidate_tables(
            storage,
            current_table_name,
            Some(original_table),
            row_batch_key,
        )?;
        Some((current_table_name.to_string(), row))
    }

    fn load_best_visible_row_batch_from_storage_with_table_hint(
        storage: &dyn Storage,
        row_id: ObjectId,
        table_hint: &str,
        branches: &[String],
        durability_tier: Option<DurabilityTier>,
        schema_context: &SchemaContext,
        branch_schema_map: &HashMap<String, SchemaHash>,
    ) -> Option<(String, QueryRowBatch)> {
        let mut best: Option<(BatchId, QueryRowBatch)> = None;

        for branch in branches {
            let branch_schema_hash = Self::branch_schema_hash_for_visible_load(
                branch,
                schema_context,
                branch_schema_map,
            );
            let translated_table = branch_schema_hash.and_then(|hash| {
                (hash != schema_context.current_hash)
                    .then(|| translate_table_name_to_schema(schema_context, table_hint, &hash))
                    .flatten()
            });
            let primary_table = translated_table.as_deref().unwrap_or(table_hint);
            let loaded_row = Self::load_visible_query_row_from_candidate_tables(
                storage,
                primary_table,
                Some(table_hint),
                branch,
                row_id,
                durability_tier,
            );
            let Some(row) = loaded_row else {
                continue;
            };

            if !row.state.is_visible() {
                continue;
            }

            let batch_id = row.batch_id;
            // A tombstone outranks a live head, whatever the clocks say.
            //
            // Within one generation the comparison below is ordered by causality: a delete
            // is a child of the row's chain on that branch. Across a generation crossing it
            // is not — the delete is authored as a parentless root on the writer's own
            // generation, so nothing links it to the head it retires, and `updated_at` — a
            // per-node wall clock that any caller may set outright — becomes the only
            // arbiter. One second of skew between the node that wrote and the node that
            // deleted then resurrects the row permanently.
            //
            // Deletion is monotone, so a rule is available that needs no clock: a live head
            // is never evidence against a tombstone. `repair_split_visible_row_families`
            // already refuses this comparison for the same reason, in its own words —
            // "comparing the heads is exactly the reasoning that made the stale one look
            // defensible." This is the live read path finally agreeing with it.
            let row_retires = row.delete_kind.is_some();
            let best_retires = best
                .as_ref()
                .is_some_and(|(_, best_row)| best_row.delete_kind.is_some());
            match &best {
                None => best = Some((batch_id, row)),
                Some(_) if best_retires && !row_retires => {}
                Some(_) if row_retires && !best_retires => best = Some((batch_id, row)),
                Some((best_batch_id, best_row))
                    if (row.updated_at, batch_id) > (best_row.updated_at, *best_batch_id) =>
                {
                    best = Some((batch_id, row));
                }
                _ => {}
            }
        }

        best.map(|(_, row)| (table_hint.to_string(), row))
    }

    pub(super) fn load_best_visible_row_batch_with_hint_or_locator(
        storage: &dyn Storage,
        row_id: ObjectId,
        table_hint: Option<&str>,
        branches: &[String],
        durability_tier: Option<DurabilityTier>,
        schema_context: &SchemaContext,
        branch_schema_map: &HashMap<String, SchemaHash>,
    ) -> Option<(String, QueryRowBatch)> {
        table_hint
            .and_then(|hint| {
                Self::load_best_visible_row_batch_from_storage_with_table_hint(
                    storage,
                    row_id,
                    hint,
                    branches,
                    durability_tier,
                    schema_context,
                    branch_schema_map,
                )
            })
            .or_else(|| {
                Self::load_best_visible_row_batch_from_storage(
                    storage,
                    row_id,
                    branches,
                    durability_tier,
                    schema_context,
                    branch_schema_map,
                )
            })
    }

    fn load_best_visible_row_batch_from_storage_with_locator(
        storage: &dyn Storage,
        row_id: ObjectId,
        locator: &RowLocator,
        branches: &[String],
        durability_tier: Option<DurabilityTier>,
        schema_context: &SchemaContext,
        branch_schema_map: &HashMap<String, SchemaHash>,
    ) -> Option<(String, QueryRowBatch)> {
        let original_table = locator.table.as_str();
        let current_table = locator
            .origin_schema_hash
            .filter(|hash| *hash != schema_context.current_hash)
            .and_then(|origin_schema_hash| {
                resolve_current_table_name(
                    schema_context,
                    original_table,
                    Some(&origin_schema_hash),
                )
            })
            .filter(|translated| translated != original_table);
        let current_table_name = current_table.as_deref().unwrap_or(original_table);

        let mut best: Option<(BatchId, QueryRowBatch)> = None;

        for branch in branches {
            let branch_schema_hash = Self::branch_schema_hash_for_visible_load(
                branch,
                schema_context,
                branch_schema_map,
            );
            let translated_table = match branch_schema_hash {
                Some(hash) if hash == schema_context.current_hash => None,
                Some(hash) => {
                    translate_table_name_to_schema(schema_context, current_table_name, &hash)
                }
                None => None,
            };
            let primary_table = match branch_schema_hash {
                Some(hash) if hash == schema_context.current_hash => current_table_name,
                Some(_) => translated_table.as_deref().unwrap_or(original_table),
                None => original_table,
            };
            let loaded_row = Self::load_visible_query_row_from_candidate_tables(
                storage,
                primary_table,
                Some(original_table),
                branch,
                row_id,
                durability_tier,
            );
            let Some(row) = loaded_row else {
                continue;
            };

            if !row.state.is_visible() {
                continue;
            }

            let batch_id = row.batch_id;
            // A tombstone outranks a live head, whatever the clocks say.
            //
            // Within one generation the comparison below is ordered by causality: a delete
            // is a child of the row's chain on that branch. Across a generation crossing it
            // is not — the delete is authored as a parentless root on the writer's own
            // generation, so nothing links it to the head it retires, and `updated_at` — a
            // per-node wall clock that any caller may set outright — becomes the only
            // arbiter. One second of skew between the node that wrote and the node that
            // deleted then resurrects the row permanently.
            //
            // Deletion is monotone, so a rule is available that needs no clock: a live head
            // is never evidence against a tombstone. `repair_split_visible_row_families`
            // already refuses this comparison for the same reason, in its own words —
            // "comparing the heads is exactly the reasoning that made the stale one look
            // defensible." This is the live read path finally agreeing with it.
            let row_retires = row.delete_kind.is_some();
            let best_retires = best
                .as_ref()
                .is_some_and(|(_, best_row)| best_row.delete_kind.is_some());
            match &best {
                None => best = Some((batch_id, row)),
                Some(_) if best_retires && !row_retires => {}
                Some(_) if row_retires && !best_retires => best = Some((batch_id, row)),
                Some((best_batch_id, best_row))
                    if (row.updated_at, batch_id) > (best_row.updated_at, *best_batch_id) =>
                {
                    best = Some((batch_id, row));
                }
                _ => {}
            }
        }

        best.map(|(_, row)| (current_table_name.to_string(), row))
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn load_visible_row_for_query(
        storage: &dyn Storage,
        row_id: ObjectId,
        table_hint: Option<&str>,
        branches: &[String],
        durability_tier: Option<DurabilityTier>,
        local_pending_version: Option<RowBatchKey>,
        prefer_local_overlay: bool,
        allow_staged_overlay: bool,
        include_deleted: bool,
        schema_context: &SchemaContext,
        branch_schema_map: &HashMap<String, SchemaHash>,
        table_for_warnings: &str,
        sub_id: QuerySubscriptionId,
        schema_warnings: &mut SchemaWarningAccumulator,
        row_bytes_dedup: &std::cell::RefCell<super::row_bytes_dedup::RowBytesDedup>,
    ) -> Option<LoadedRow> {
        // Settle-cost accounting: the query row loader is the dominant storage
        // read driver of a settle.
        super::settle_cost::bump(&super::settle_cost::ROW_LOADS);
        let exact_pending_visible_row = || {
            let pending_version = local_pending_version?;
            let resolved = Self::load_best_visible_row_batch_with_hint_or_locator(
                storage,
                row_id,
                table_hint,
                branches,
                None,
                schema_context,
                branch_schema_map,
            )?;
            let (_, row) = &resolved;
            (row.batch_id == pending_version.batch_id
                && row.branch.as_str() == pending_version.branch_name.as_str())
            .then_some(resolved)
        };
        let pending_staged_row = || {
            let pending_version = local_pending_version?;
            let resolved = Self::load_local_pending_query_row_with_hint_or_locator(
                storage,
                pending_version,
                table_hint,
                schema_context,
            )?;
            let (_, row) = &resolved;
            (row.batch_id == pending_version.batch_id
                && row.branch.as_str() == pending_version.branch_name.as_str()
                && matches!(row.state, RowState::StagingPending))
            .then_some(resolved)
        };
        let best_visible_row = || {
            Self::load_best_visible_row_batch_with_hint_or_locator(
                storage,
                row_id,
                table_hint,
                branches,
                durability_tier,
                schema_context,
                branch_schema_map,
            )
        };
        let resolved = if prefer_local_overlay {
            exact_pending_visible_row()
                .or_else(pending_staged_row)
                .or_else(best_visible_row)
        } else if allow_staged_overlay {
            best_visible_row()
                .or_else(exact_pending_visible_row)
                .or_else(pending_staged_row)
        } else {
            best_visible_row().or_else(exact_pending_visible_row)
        }?;
        let (table, row) = resolved;

        if row.is_hard_deleted() {
            return None;
        }

        if row.is_soft_deleted() && !include_deleted {
            return None;
        }

        let batch_id = row.batch_id;
        let row_provenance = row.row_provenance();
        let source_branch = row.branch.as_str();

        if let Some(&source_hash) = branch_schema_map.get(source_branch)
            && source_hash != schema_context.current_hash
        {
            let transformer = LensTransformer::new(schema_context, &table);
            match transformer.transform(&row.data, batch_id, source_hash) {
                Ok(result) => {
                    return Some(LoadedRow::new(
                        result.data,
                        row_provenance,
                        [(row_id, BranchName::new(source_branch))]
                            .into_iter()
                            .collect(),
                        result.batch_id,
                    ));
                }
                Err(err) => {
                    schema_warnings.record(
                        table_for_warnings,
                        source_hash,
                        schema_context.current_hash,
                    );
                    tracing::debug!(
                        sub_id = sub_id.0,
                        row_id = %row_id,
                        table = %table,
                        source_branch = source_branch,
                        source_schema = %source_hash.short(),
                        target_schema = %schema_context.current_hash.short(),
                        error = %err,
                        "lens transform failed; row will be counted in aggregated schema warning"
                    );
                    return None;
                }
            }
        }

        // Canonicalize the freshly loaded bytes so every subscription over
        // this (row, batch) shares one allocation. The lens path above is
        // deliberately NOT deduplicated: its transformed bytes share the
        // source batch id and would collide with the raw content.
        let data = row_bytes_dedup
            .borrow_mut()
            .dedup(row_id, row.batch_id, row.data);
        Some(LoadedRow::new(
            data,
            row_provenance,
            [(row_id, BranchName::new(source_branch))]
                .into_iter()
                .collect(),
            row.batch_id,
        ))
    }

    /// Check if a subscription involves a given table (base table, joined table, or array subquery inner table).
    pub(super) fn subscription_involves_table(
        graph: &super::graph::QueryGraph,
        table: &str,
    ) -> bool {
        graph.involves_table(table)
    }

    pub(super) fn row_delta_from_rows(
        previous_rows: &HashMap<ObjectId, Row>,
        previous_order: &[ObjectId],
        next_rows: &[Row],
    ) -> RowDelta {
        let next_rows_by_id: HashMap<_, _> =
            next_rows.iter().cloned().map(|row| (row.id, row)).collect();
        let previous_indices: HashMap<_, _> = previous_order
            .iter()
            .enumerate()
            .map(|(index, id)| (*id, index))
            .collect();
        let next_indices: HashMap<_, _> = next_rows
            .iter()
            .enumerate()
            .map(|(index, row)| (row.id, index))
            .collect();

        let added = next_rows
            .iter()
            .filter(|row| !previous_rows.contains_key(&row.id))
            .cloned()
            .collect();
        let removed = previous_order
            .iter()
            .filter_map(|id| previous_rows.get(id))
            .filter(|row| !next_rows_by_id.contains_key(&row.id))
            .cloned()
            .collect();
        let updated = next_rows
            .iter()
            .filter_map(|row| {
                previous_rows.get(&row.id).and_then(|previous| {
                    (previous.data != row.data || previous.batch_id != row.batch_id)
                        .then(|| (previous.clone(), row.clone()))
                })
            })
            .collect();
        let moved = next_rows
            .iter()
            .filter(|row| {
                previous_rows.contains_key(&row.id)
                    && previous_rows
                        .get(&row.id)
                        .map(|previous| {
                            previous.data == row.data && previous.batch_id == row.batch_id
                        })
                        .unwrap_or(false)
                    && previous_indices.get(&row.id) != next_indices.get(&row.id)
            })
            .map(|row| row.id)
            .collect();

        RowDelta {
            added,
            removed,
            moved,
            updated,
        }
    }

    pub(super) fn rows_from_tuples(graph: &QueryGraph, tuples: &[Tuple]) -> Vec<Row> {
        tuples
            .iter()
            .filter_map(|tuple| {
                if tuple.len() == 1 {
                    tuple.to_single_row()
                } else {
                    tuple
                        .flatten_with_descriptors(
                            &graph.table_descriptors,
                            &graph.combined_descriptor,
                        )
                        .and_then(|flattened| flattened.to_single_row())
                }
            })
            .collect()
    }

    /// Keep a tuple when the remote scope covers it, or when it carries a local write the
    /// server is not yet known to have accounted for.
    ///
    /// The exemption is deliberately narrow on three axes the previous version ignored:
    ///
    /// * **branch** — matched on `(id, branch)` from the tuple's provenance, not on the id
    ///   alone. A row pending on one branch used to exempt the same id everywhere.
    /// * **the reader's choice** — carried by `LocalWriteAuthority`, which names WHOSE
    ///   local writes this reader may be exempted for, not merely whether it may be.
    ///   `Deferred` with no overlay opted out and gets nothing; `Immediate` gets the
    ///   process-wide maps. A transaction-scoped read is the third case and the reason
    ///   this is not a bool: it carries its staged rows in `local_overlay_rows`, and both
    ///   consumers of that map — the row loader and the source overlay — read it before
    ///   they look at `LocalUpdates` at all, so those rows are materialised whatever the
    ///   flag says and dropping them here would load them only to throw them away. But the
    ///   process-wide maps hold every unsettled write of every table, and that read asked
    ///   for ONE batch: it is exempted for its own staged rows and nobody else's.
    /// * **durability** — a write that has reached the settlement target and been followed
    ///   by a scope refresh is no longer exempt, because from that point the server's
    ///   answer accounts for it. Without this bound the exemption never expired on a node
    ///   whose reads are one-shot, and every row it had ever written stayed frozen in its
    ///   own reads for the process lifetime.
    fn filter_synced_query_scope_tuples(
        &self,
        query_id: QueryId,
        durability_tier: Option<DurabilityTier>,
        local_writes: LocalWriteAuthority<'_>,
        tuples: Vec<Tuple>,
    ) -> Vec<Tuple> {
        let authoritative_snapshot_pass = self.authoritative_snapshot_pass.get(&query_id).copied();
        let remote_scope = match durability_tier {
            Some(tier) => self
                .sync_manager
                .remote_query_scope_at_least(query_id, tier),
            None => self.sync_manager.remote_query_scope(query_id),
        };
        tuples
            .into_iter()
            .filter(|tuple| {
                let in_remote_scope = tuple
                    .provenance()
                    .iter()
                    .any(|scoped_object| remote_scope.contains(scoped_object));
                if in_remote_scope {
                    return true;
                }
                if local_writes.grants_nothing() {
                    return false;
                }
                if let Some(overlay) = local_writes.overlay
                    && tuple.provenance().iter().any(|(id, branch)| {
                        overlay
                            .get(id)
                            .is_some_and(|staged| staged.branch_name == *branch)
                    })
                {
                    return true;
                }
                if !local_writes.process_wide {
                    return false;
                }
                // The per-subscription `pending_local_row_ids` is deliberately NOT consulted
                // here. It serves two other roles (the IndexScan overlay and the row-loader
                // durability downgrade) and is retired on different terms, so as a scope
                // exemption it outlived the write it stood for. The two maps below are the
                // single authority for this role:
                //
                //   `scope_exempt_local_rows`            — written but not yet durable at the
                //                                          settlement target; unbounded on
                //                                          purpose, nothing upstream can know
                //                                          about it yet. A row staged in an
                //                                          open transaction stays here until
                //                                          the batch commits and confirms.
                //   `confirmed_local_rows_awaiting_scope` — durable, but no scope refresh has
                //                                          landed since. Held one more round
                //                                          so the exemption does not drop
                //                                          before the server's answer can
                //                                          account for the write.
                tuple.provenance().iter().any(|(id, branch)| {
                    // Chained, not `or_else`: the two maps are disjoint by construction
                    // (every insert into one removes from the other, and phase one moves an
                    // entry across), but a short-circuit would make a violation of that
                    // invariant silently drop the tuple instead of failing loudly.
                    debug_assert!(
                        !(self.scope_exempt_local_rows.contains_key(id)
                            && self.confirmed_local_rows_awaiting_scope.contains_key(id)),
                        "a local row must be exempt-pending or awaiting-scope, never both"
                    );
                    if self
                        .scope_exempt_local_rows
                        .get(id)
                        .is_some_and(|pending_branch| pending_branch == branch)
                    {
                        // Written, not yet durable at the settlement target. Unbounded on
                        // purpose: nothing upstream can know about it.
                        return true;
                    }
                    self.confirmed_local_rows_awaiting_scope
                        .get(id)
                        .is_some_and(|parked| {
                            parked.branch == *branch
                                // Durable, and still exempt for THIS reader until an answer
                                // this reader can trust arrives after the write did.
                                && authoritative_snapshot_pass
                                    .is_none_or(|pass| pass <= parked.parked_in_pass)
                        })
                })
            })
            .collect()
    }

    fn scope_from_tuples(tuples: &[Tuple]) -> HashSet<(ObjectId, BranchName)> {
        tuples
            .iter()
            .flat_map(|tuple| tuple.provenance().iter().copied())
            .collect()
    }

    fn authoritative_batch_fate_cached(
        &mut self,
        storage: &dyn Storage,
        batch_id: BatchId,
    ) -> Option<BatchFate> {
        if let Some(settlement) = self.authoritative_batch_fate_cache.get(&batch_id) {
            return settlement.clone();
        }

        let settlement = match storage.load_authoritative_batch_fate(batch_id) {
            Ok(settlement) => settlement,
            Err(error) => {
                tracing::warn!(?batch_id, %error, "failed to load authoritative batch settlement");
                None
            }
        };
        self.authoritative_batch_fate_cache
            .insert(batch_id, settlement.clone());
        settlement
    }

    fn transactional_batch_complete_for_query_scope(
        &mut self,
        storage: &dyn Storage,
        batch_id: BatchId,
        local_scope: &HashSet<(ObjectId, BranchName)>,
        query_scope: &HashSet<(ObjectId, BranchName)>,
    ) -> bool {
        let settlement = self.authoritative_batch_fate_cached(storage, batch_id);

        !matches!(settlement, Some(BatchFate::AcceptedTransaction { .. }))
            || query_scope.is_subset(local_scope)
    }

    fn filter_transaction_visible_tuples<'a>(
        &mut self,
        storage: &dyn Storage,
        query_id: QueryId,
        tuples: Cow<'a, [Tuple]>,
    ) -> Cow<'a, [Tuple]> {
        if tuples.is_empty() {
            return tuples;
        }

        let local_scope = Self::scope_from_tuples(tuples.as_ref());
        let mut query_scope = local_scope.clone();
        query_scope.extend(self.sync_manager.remote_query_scope(query_id));

        let mut first_hidden = None;

        for (index, tuple) in tuples.as_ref().iter().enumerate() {
            let is_visible = tuple.batch_provenance().iter().copied().all(|batch_id| {
                self.transactional_batch_complete_for_query_scope(
                    storage,
                    batch_id,
                    &local_scope,
                    &query_scope,
                )
            });
            if !is_visible {
                first_hidden = Some(index);
                break;
            }
        }

        let Some(first_hidden) = first_hidden else {
            return tuples;
        };

        let mut filtered = Vec::with_capacity(tuples.len().saturating_sub(1));
        filtered.extend_from_slice(&tuples.as_ref()[..first_hidden]);
        for tuple in &tuples.as_ref()[first_hidden + 1..] {
            if tuple.batch_provenance().iter().copied().all(|batch_id| {
                self.transactional_batch_complete_for_query_scope(
                    storage,
                    batch_id,
                    &local_scope,
                    &query_scope,
                )
            }) {
                filtered.push(tuple.clone());
            }
        }
        Cow::Owned(filtered)
    }
    // ========================================================================
    // No-op storage driver (for tests)
    // ========================================================================

    // ========================================================================
    // Memory profiling
    // ========================================================================

    /// Calculate memory usage breakdown for profiling.
    ///
    /// Returns a tuple: (indices, subscriptions, policy_checks, total)
    /// Note: indices are managed by Storage, so index memory is reported as 0.
    pub fn memory_size(&self) -> (usize, usize, usize, usize) {
        let indices = 0usize; // Indices managed by Storage

        // Subscriptions (QueryGraph can be large)
        let mut subscriptions = 0usize;
        for (id, sub) in &self.subscriptions {
            subscriptions += std::mem::size_of_val(id);
            subscriptions += std::mem::size_of::<QuerySubscription>();
            subscriptions += sub.graph.estimate_memory_size();
            subscriptions += 48; // HashMap entry overhead
        }
        subscriptions += self.update_outbox.len() * 256; // QueryUpdate overhead

        // Active policy checks
        let mut policy_checks = 0usize;
        for state in self.active_policy_checks.values() {
            policy_checks += 48; // HashMap entry
            policy_checks += state.graphs.len() * 1024; // Rough estimate per PolicyGraph
            policy_checks += state.table.0.len();
            policy_checks += state.branch.as_str().len();
        }

        let total = indices + subscriptions + policy_checks;
        (indices, subscriptions, policy_checks, total)
    }
}

fn propagation_label(propagation: QueryPropagation) -> &'static str {
    match propagation {
        QueryPropagation::Full => "full",
        QueryPropagation::LocalOnly => "local-only",
    }
}

fn subscription_group_key(query: &str, branches: &[String], propagation: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(query.as_bytes());
    hasher.update([0]);
    hasher.update(propagation.as_bytes());
    hasher.update([0]);
    for branch in branches {
        hasher.update(branch.as_bytes());
        hasher.update([0]);
    }
    hex::encode(hasher.finalize())
}
