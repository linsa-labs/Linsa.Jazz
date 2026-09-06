use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Duration;

use web_time::Instant;

use crate::batch_fate::{BatchFate, SealedBatchSubmission};
use crate::catalogue::CatalogueEntry;
use crate::object::{BranchName, ObjectId};
use crate::query_manager::query::Query;
use crate::query_manager::session::Session;
use crate::query_manager::types::SchemaHash;
use crate::row_histories::{BatchId, RowVisibilityChange};
use crate::storage::{PreparedRowTableContext, Storage};

// Module declarations
pub mod clock;
pub mod forwarding;
pub mod inbox;
pub mod permissions;
pub mod sync_logic;
pub mod sync_tracer;
pub mod types;

use clock::MonotonicClock;

#[cfg(test)]
mod tests;

// Re-export all public types
pub mod admission;
pub mod wire_depth;
pub use admission::{Principal, SubscriptionCaps};
pub use types::*;

/// How long an installed transport may sit in `pending_servers` before callers
/// treat it as offline.
pub const PENDING_SERVER_TIMEOUT: Duration = Duration::from_secs(2);

#[derive(Clone)]
pub(crate) struct OutgoingQuerySubscription {
    pub(crate) query_id: QueryId,
    pub(crate) query: Query,
    pub(crate) session: Option<Session>,
    pub(crate) required_tier: Option<DurabilityTier>,
    pub(crate) propagation: QueryPropagation,
    pub(crate) policy_context_tables: Vec<String>,
}

// ============================================================================
// SyncManager
// ============================================================================

/// How many times a row is offered to a peer that never confirms it before the sender
/// gives up on hearing back.
///
/// Some rows can never be applied by a given peer — a rejected fate, a decode failure, a
/// bug on its side — and nothing will ever confirm them. Without a bound the peer stays
/// marked as owed rows for as long as it is connected, and every subscription registration
/// re-derives its scope and re-sends: a livelock that is worse than the loss it replaced,
/// because it never ends. On giving up the claim is recorded, which is exactly the old
/// behaviour for that one row, and a warning names it.
pub const MAX_REDELIVERY_ATTEMPTS: u32 = 5;

/// How long a row must have been outstanding, in microseconds, before the attempt cap may
/// give up on it. Confirmation is a full round trip — deliver, apply, tick, WAL barrier,
/// ack — while re-offers arrive in registration bursts; without this floor a startup storm
/// burns the cap in milliseconds and converts eventual delivery into permanent loss.
pub const REDELIVERY_GIVE_UP_AFTER_MICROS: u64 = 60_000_000;

/// How often, how many times, and for how long this authority tells a client that a batch
/// is `Missing`.
///
/// `Missing` is not advice, it is an instruction: on the peer it drives
/// `retransmit_local_batch_to_servers`, and `force_row_batch_to_servers` clears the dedup
/// bookkeeping so nothing suppresses the resend. Every answer therefore costs the peer a
/// full retransmission of the batch, and each retransmitted row asks the question again —
/// the replay short-circuit queues a fate request for its batch, so rows alone regenerate
/// it. When the batch cannot be completed, the answer buys the work that produces the next
/// question and the cycle's only limit is how fast this side answers: one pinned core,
/// sync dead behind it (production 2026-08-10).
///
/// The bound is on the answer rather than on any single reason a batch fails to complete,
/// because the reasons are many — a member this authority never indexed, a branch
/// mismatch, drift in a payload or its authorship — and the amplification is common to all
/// of them.
///
/// Three limits, because they bound different things. The interval bounds the RATE, and is
/// what a burst runs into first: without it a peer retrying in a tight loop collects
/// thousands of retransmission instructions inside any grace we could choose. The cap and
/// the grace are the give-up policy, carried over unchanged from `MAX_REDELIVERY_ATTEMPTS`
/// / `REDELIVERY_GIVE_UP_AFTER_MICROS` for the reasons stated there: the cap alone would
/// be spent by a reconnect storm in milliseconds, and the grace keeps a merely slow peer
/// working.
/// All three are wall-clock microseconds, compared against `MonotonicClock::reserve_timestamp`
/// — the same units and the same source as the redelivery bound they are drawn from.
pub const MISSING_ANSWER_MIN_INTERVAL_MICROS: u64 = 5_000_000;
pub const MAX_MISSING_ANSWERS: u32 = MAX_REDELIVERY_ATTEMPTS;
pub const MISSING_ANSWER_GIVE_UP_AFTER_MICROS: u64 = REDELIVERY_GIVE_UP_AFTER_MICROS;

/// How many distinct batches are tracked per client, and how many a single request may ask
/// about.
///
/// The budget above needs one entry per batch to count against, so the tracking is itself
/// something a peer can grow, and the first answer for a batch is deliberately free — a
/// genuinely interrupted upload must be repaired without waiting out an interval. Those
/// two together mean the tracking cap is what stands between a peer and an endless supply
/// of free answers: it can have this many, and then only as fast as it can drive batches
/// to silence.
///
/// So room is made only by evicting a batch this authority has already given up on, and a
/// `BatchFateNeeded` naming more ids than can be tracked is answered up to the cap and no
/// further. A peer that cycles fresh ids to dodge the budget hits both.
pub(super) const MAX_TRACKED_MISSING_ANSWERS: usize = 1024;

/// How many recoverably-failed row batches are kept parked per `(row, branch)`.
///
/// A parked batch is one whose apply failed for a reason a later arrival can cure — a
/// transient storage error, or a parent that has not landed yet. The queue must be capped
/// because its depth is peer-driven: a wedged row keeps receiving one new child per
/// heartbeat for as long as the wedge lasts (548 batches over one incident afternoon,
/// production 2026-08-15). Past the cap the OLDEST parked batch is dropped with a warning —
/// dropped from the park only, not lost: every parked batch is still held by its sender,
/// and the ancestor-request path re-obtains it when the healed prefix reaches its gap.
pub(super) const MAX_PARKED_ROW_BATCHES_PER_ROW: usize = 32;

/// How many distinct `(row, branch)` keys may hold parked batches at once.
///
/// Bounds the second axis a peer controls: fresh row ids. Past the cap the least-recently
/// parked row's whole queue is dropped with a warning, on the same reasoning as the per-row
/// cap — the senders still hold everything dropped here.
pub(super) const MAX_PARKED_ROWS: usize = 256;

/// What this authority has already told one client about the batches it cannot complete.
///
/// The tracked ids are kept in creation order so making room is O(1): the cap is reached
/// exactly when a peer is producing ids faster than they settle, which is when a scan over
/// the tracked set would be a cost the peer sets the size of.
#[derive(Debug, Clone, Default)]
pub(super) struct ClientMissingAnswers {
    pub(super) budgets: HashMap<BatchId, MissingAnswerBudget>,
    pub(super) order: VecDeque<BatchId>,
}

/// What this authority has already told one client about one batch it cannot complete.
///
/// Deliberately holds no copy of what the peer declared. An earlier draft remembered the
/// declaration so a *different* one could re-arm the budget, which reads as fairness and
/// is in fact the hole: a peer alternating two declarations, or perturbing one member per
/// round, resets the budget every round and the bound never engages. A declaration that
/// can be matched never reaches here at all, so re-arming on a changed one buys nothing
/// except that hole.
/// How often a row that keeps failing to apply may repeat its WARN.
///
/// The first failure per `(row, branch)` is always loud — that is the line that names the
/// problem. After it, repeats carry no new information and are counted instead of logged,
/// with one summary allowed per interval. Production 2026-08-18: one wedged row emitted
/// 130,634 identical WARN lines in three hours, peaking at 49,606 a minute, and not one of
/// them said "this row is stuck". The volume was its own outage surface, and the incident
/// was found from a CPU graph rather than from the log that was describing it all along.
pub(super) const UNAPPLIABLE_ROW_WARN_INTERVAL_MICROS: u64 = 30_000_000;

/// What one `(row, branch)` has cost while it could not be applied.
#[derive(Clone)]
pub(super) struct UnappliableRowNotice {
    pub(super) attempts: u64,
    pub(super) first_failed_at: u64,
    pub(super) last_warned_at: u64,
    /// Which failure the last spoken line described. A row that changes failure mode is
    /// loud again immediately, because the interval would otherwise hide the change and
    /// production runs at `info` — the DEBUG repeat line does not exist there. Silence for
    /// half a minute is how a storage failure would arrive behind a `ParentNotFound` that
    /// preceded it on the same row, which is the shape of the 2026-08-15 ENOSPC incident.
    pub(super) last_error: &'static str,
}

#[derive(Debug, Clone)]
pub(super) struct MissingAnswerBudget {
    pub(super) answers: u32,
    pub(super) first_answered_at: u64,
    pub(super) last_answered_at: u64,
    /// Set when the budget ran out, so the warning names it once instead of per attempt.
    pub(super) silenced: bool,
}

/// Manages synchronization state atop storage-backed row and catalogue state.
///
/// Coordinates:
/// - Upstream servers (trusted, receive all our objects)
/// - Downstream clients (untrusted, receive query-filtered subsets)
#[derive(Clone)]
pub struct SyncManager {
    pub(super) clock: MonotonicClock,
    pub(super) catalogue_entries: HashMap<ObjectId, CatalogueEntry>,
    pub(super) allow_unprivileged_schema_catalogue_writes: bool,

    pub(super) servers: HashMap<ServerId, ServerState>,
    pub(super) pending_servers: HashMap<ServerId, Instant>,
    /// v18 item 6: clients whose role actually changed since the query manager last drained
    /// this list (an un-stalling event for their server subscriptions).
    pub(super) role_changed: Vec<ClientId>,
    pub(super) pending_server_query_subscriptions: HashSet<(ServerId, QueryId)>,
    pub(super) clients: HashMap<ClientId, ClientState>,
    /// Admission control for downstream registrations (item 2 / defect #31).
    pub(super) admission: admission::Admission,
    pub(super) inbox: Vec<InboxEntry>,
    pub(super) outbox: Vec<OutboxEntry>,

    /// Rows queued to a client and not yet confirmed by it, per client per row.
    ///
    /// The claim used to be written at enqueue. A payload is dropped silently when the
    /// client has no registered stream or its channel is dead, and nothing rolled the claim
    /// back — so the row was never offered again and the receiver lost it for as long as
    /// its server-side state lived. Entries wait here until the RECEIVER says it applied
    /// the row: no sender-side signal is trustworthy, because a socket that is dying still
    /// accepts writes for as long as TCP takes to notice.
    pub(super) pending_client_deliveries:
        HashMap<ClientId, HashMap<(ObjectId, BranchName), PendingDelivery>>,

    /// What this authority has already said about batches it cannot complete, per client,
    /// so an answer that cannot help stops repeating.
    ///
    /// Connection-scoped on purpose: dropped with the client, which re-arms the answer on
    /// reconnect. Silence defers a batch, it never abandons one — the submission stays put
    /// and no fate is invented, because on the peer a `Rejected` destroys the row and the
    /// graft tool is offline-only.
    pub(super) missing_answers: HashMap<ClientId, ClientMissingAnswers>,
    /// Rows that could not be applied, and what they have cost since. Keeps the repeat
    /// failures out of the log while making the row itself nameable — see
    /// `UNAPPLIABLE_ROW_WARN_INTERVAL_MICROS`.
    pub(super) unappliable_rows: HashMap<(ObjectId, BranchName), UnappliableRowNotice>,

    /// Row batches whose apply failed recoverably, kept until a later arrival on the same
    /// row makes them applicable.
    ///
    /// Two failure shapes land here, and both used to be terminal (the batch was dropped
    /// with one warn line, while the sender's dedup bookkeeping recorded it as delivered):
    /// a transient storage error on the commit, and `ParentNotFound` — which the first
    /// shape then manufactures for every subsequent batch of the row, forever (production
    /// 2026-08-15: one ENOSPC commit, 548 cascading `ParentNotFound` failures over two
    /// wedged rows, presence and profile edits dead for every device). Parked batches are
    /// retried whenever a batch for their row applies; a missing ancestor is requested
    /// from the sending client via the existing `BatchFate::Missing` retransmission
    /// instruction, budget-gated by `may_tell_client_a_batch_is_missing`.
    ///
    /// In-memory on purpose: everything parked is still held by its sender, so a restart
    /// loses nothing that the ancestor-request path cannot re-obtain.
    pub(super) parked_row_batches: HashMap<(ObjectId, BranchName), VecDeque<inbox::ParkedRowBatch>>,
    /// Insertion order of the keys in `parked_row_batches`, for least-recently-parked
    /// eviction at `MAX_PARKED_ROWS`. Kept exactly in sync with the map's key set.
    pub(super) parked_rows_order: VecDeque<(ObjectId, BranchName)>,

    /// Rows this node has applied and not yet reported upstream.
    ///
    /// Drained after the write barrier, never before: a confirmation that leaves ahead of
    /// the fsync lets the sender clear its claim for a row this node would lose on a crash
    /// — the original defect, reproduced from the other side.
    pub(super) applied_rows_to_confirm: Vec<ConfirmedRow>,
    /// Whether the upstream server understands confirmations, from its handshake response.
    pub(super) upstream_supports_delivery_acks: bool,
    /// Pending permission checks awaiting policy evaluation.
    pub(super) pending_permission_checks: Vec<PendingPermissionCheck>,
    /// Pending query subscriptions awaiting QueryGraph building by QueryManager.
    pub(super) pending_query_subscriptions: Vec<PendingQuerySubscription>,
    /// Pending query unsubscriptions awaiting cleanup by QueryManager.
    pub(super) pending_query_unsubscriptions: Vec<PendingQueryUnsubscription>,
    /// Row visibility changes applied through row-history sync.
    pub(super) pending_row_visibility_changes: Vec<RowVisibilityChange>,
    /// Catalogue/system entry updates awaiting SchemaManager processing.
    pub(super) pending_catalogue_updates: Vec<CatalogueEntry>,
    /// Digest of every catalogue entry this node has already handed to the
    /// schema layer, keyed by object id.
    ///
    /// Deliberately NOT the same question as "does storage hold these bytes"
    /// (`catalogue_entries` plus the storage fallback, both consulted by
    /// `persist_catalogue_entry`). Storage is written before a restart; the
    /// in-memory schema layer is rebuilt after it. Conflating the two is what
    /// made a generation already on disk unlearnable by a running process — the
    /// entry came back over the wire byte-identical, "storage unchanged" was the
    /// answer, and `pending_catalogue_updates` never saw it.
    ///
    /// It is also NOT seeded by the connect-replay paths
    /// (`queue_catalogue_sync_to_client_from_storage`,
    /// `queue_catalogue_sync_to_server_from_storage`), which populate
    /// `catalogue_entries` wholesale from storage; seeding it there would
    /// restore exactly the bug.
    ///
    /// Bounded by the number of catalogue objects — schemas, permissions
    /// bundles, lenses — not by traffic.
    pub(super) handed_to_schema_layer: HashMap<ObjectId, [u8; 32]>,

    pub(super) next_pending_id: u64,

    /// This node's durability identities (empty = don't emit durability notifications).
    pub(super) my_tiers: HashSet<DurabilityTier>,
    /// Tracks which clients are interested in row batch-member state updates.
    pub(super) row_batch_interest: HashMap<RowBatchKey, HashSet<ClientId>>,
    /// Tracks clients that explicitly requested the current or next known fate
    /// for a batch whose row member state may not be present on this peer.
    pub(super) batch_fate_interest: HashMap<BatchId, HashSet<ClientId>>,

    /// Tracks which clients originated each query (for relaying QuerySettled).
    pub(super) query_origin: HashMap<QueryId, HashSet<ClientId>>,
    /// Latest remote scope snapshots keyed by upstream server and query id.
    pub(super) remote_query_scopes: HashMap<(ServerId, QueryId), HashSet<(ObjectId, BranchName)>>,
    /// Durability tier associated with each latest remote scope snapshot.
    pub(super) remote_query_scope_tiers: HashMap<(ServerId, QueryId), DurabilityTier>,
    /// Query ids whose remote scope changed since the last QueryManager process.
    pub(super) remote_query_scope_dirty: HashSet<QueryId>,
    /// Pending QuerySettled notifications for QueryManager to process.
    pub(super) pending_query_settled: Vec<PendingQuerySettled>,
    /// Pending query rejections waiting for QueryManager to fail local subscriptions.
    pub(super) pending_query_rejections: Vec<PendingQueryRejection>,
    /// Pending replayable batch fates for RuntimeCore to process.
    pub(super) pending_batch_fates: Vec<BatchFate>,

    /// Batch fates to send to clients after a full inbox batch has been processed.
    pub(super) pending_client_batch_fates: HashMap<ClientId, HashSet<BatchId>>,
    /// Per-sync-manager replay cache for table/schema row write context.
    ///
    /// Incoming sync rows usually carry table + origin schema metadata. Rows in
    /// a large replay share this context, so cache it above the per-row
    /// visibility work instead of re-resolving descriptors and raw table IDs
    /// from storage for every row.
    pub(super) replay_table_contexts:
        HashMap<(String, SchemaHash), std::sync::Arc<PreparedRowTableContext>>,
}

impl std::fmt::Debug for SyncManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SyncManager")
            .field("clock", &self.clock)
            .field("catalogue_entries", &self.catalogue_entries)
            .field(
                "allow_unprivileged_schema_catalogue_writes",
                &self.allow_unprivileged_schema_catalogue_writes,
            )
            .field("servers", &self.servers)
            .field("pending_servers", &self.pending_servers)
            .field("clients", &self.clients)
            .field("inbox", &self.inbox)
            .field("outbox", &self.outbox)
            .field("pending_permission_checks", &self.pending_permission_checks)
            .field(
                "pending_query_subscriptions",
                &self.pending_query_subscriptions,
            )
            .field(
                "pending_query_unsubscriptions",
                &self.pending_query_unsubscriptions,
            )
            .field(
                "pending_row_visibility_changes",
                &self.pending_row_visibility_changes,
            )
            .field("pending_catalogue_updates", &self.pending_catalogue_updates)
            .field("next_pending_id", &self.next_pending_id)
            .field("my_tiers", &self.my_tiers)
            .field("row_batch_interest", &self.row_batch_interest)
            .field("batch_fate_interest", &self.batch_fate_interest)
            .field("query_origin", &self.query_origin)
            .field("remote_query_scopes", &self.remote_query_scopes)
            .field("remote_query_scope_tiers", &self.remote_query_scope_tiers)
            .field("pending_query_settled", &self.pending_query_settled)
            .field("pending_query_rejections", &self.pending_query_rejections)
            .field("pending_batch_fates", &self.pending_batch_fates)
            .field(
                "pending_client_batch_fates",
                &self.pending_client_batch_fates,
            )
            .finish()
    }
}

impl Default for SyncManager {
    fn default() -> Self {
        Self::new()
    }
}

fn short_hash(hash: &impl ToString) -> String {
    hash.to_string().chars().take(12).collect()
}

pub(crate) fn log_schema_warning(
    warning: &SchemaWarning,
    origin: Option<&str>,
    subscription_id: Option<u64>,
) {
    tracing::warn!(
        origin = origin,
        sub_id = subscription_id,
        query_id = warning.query_id.0,
        table = warning.table_name,
        row_count = warning.row_count,
        from_hash = %warning.from_hash,
        to_hash = %warning.to_hash,
        "Detected {} rows of {} with differing schema versions. To ensure data visibility and forward/backward compatibility, run `npx jazz-tools@alpha schema export --schema-hash {}`. Then generate a migration with `npx jazz-tools@alpha migrations create --fromHash {} --toHash <targetHash>`.",
        warning.row_count,
        warning.table_name,
        short_hash(&warning.from_hash),
        short_hash(&warning.from_hash),
    );
}

pub(crate) fn log_connection_schema_diagnostics(
    diagnostics: &ConnectionSchemaDiagnostics,
    origin: Option<&str>,
) {
    let client_hash = short_hash(&diagnostics.client_schema_hash);

    if let Some(permissions_hash) = diagnostics.disconnected_permissions_schema_hash {
        let permissions_hash = short_hash(&permissions_hash);
        tracing::error!(
            origin = origin,
            client_schema_hash = %client_hash,
            permissions_schema_hash = %permissions_hash,
            "Your declared schema {} is disconnected from the schema used to enforce permissions: {}. Reads and writes may fail until you add a migration. To recover, run `npx jazz-tools@alpha migrations create --fromHash {} --toHash {}`.",
            client_hash,
            permissions_hash,
            permissions_hash,
            client_hash,
        );
    }

    if !diagnostics.unreachable_schema_hashes.is_empty() {
        let unreachable_hashes: Vec<String> = diagnostics
            .unreachable_schema_hashes
            .iter()
            .map(short_hash)
            .collect();
        tracing::trace!(
            origin = origin,
            client_schema_hash = %client_hash,
            unreachable_schema_hashes = ?unreachable_hashes,
            "Server knows schema branches that are unreachable from your declared schema {}: {}. Some data may be missing from reads until you add migrations. To recover, run `npx jazz-tools@alpha migrations create --fromHash <unreachableHash> --toHash {}` for each listed schema.",
            client_hash,
            unreachable_hashes.join(", "),
            client_hash,
        );
    }
}

impl SyncManager {
    pub fn new() -> Self {
        Self {
            clock: MonotonicClock::new(),
            catalogue_entries: HashMap::new(),
            allow_unprivileged_schema_catalogue_writes: false,
            servers: HashMap::new(),
            pending_servers: HashMap::new(),
            role_changed: Vec::new(),
            pending_server_query_subscriptions: HashSet::new(),
            // No cap bites on a node that is not a server; the server builder installs the
            // production caps (`with_subscription_caps`).
            clients: HashMap::new(),
            admission: admission::Admission::new(admission::SubscriptionCaps::unlimited()),
            inbox: Vec::new(),
            outbox: Vec::new(),
            pending_client_deliveries: HashMap::new(),
            missing_answers: HashMap::new(),
            unappliable_rows: HashMap::new(),
            parked_row_batches: HashMap::new(),
            parked_rows_order: VecDeque::new(),
            applied_rows_to_confirm: Vec::new(),
            upstream_supports_delivery_acks: false,
            pending_permission_checks: Vec::new(),
            pending_query_subscriptions: Vec::new(),
            pending_query_unsubscriptions: Vec::new(),
            pending_row_visibility_changes: Vec::new(),
            pending_catalogue_updates: Vec::new(),
            handed_to_schema_layer: HashMap::new(),
            next_pending_id: 0,
            my_tiers: HashSet::new(),
            row_batch_interest: HashMap::new(),
            batch_fate_interest: HashMap::new(),
            query_origin: HashMap::new(),
            remote_query_scopes: HashMap::new(),
            remote_query_scope_tiers: HashMap::new(),
            remote_query_scope_dirty: HashSet::new(),
            pending_query_settled: Vec::new(),
            pending_query_rejections: Vec::new(),
            pending_batch_fates: Vec::new(),
            pending_client_batch_fates: HashMap::new(),
            replay_table_contexts: HashMap::new(),
        }
    }

    pub fn reserve_timestamp(&mut self) -> u64 {
        self.clock.reserve_timestamp()
    }

    /// Add a durability identity for this node (enables durability notifications).
    pub fn with_durability_tier(mut self, tier: DurabilityTier) -> Self {
        self.my_tiers.insert(tier);
        self
    }

    /// Caps applied to registrations from downstream clients.
    pub fn with_subscription_caps(mut self, caps: SubscriptionCaps) -> Self {
        self.admission.set_caps(caps);
        self
    }

    pub fn subscription_caps(&self) -> &SubscriptionCaps {
        self.admission.caps()
    }

    /// Standing registrations charged to a principal (exact; see `admission`).
    pub fn admitted_subscription_count(&self, principal: &Principal) -> usize {
        self.admission.admitted_count(principal)
    }

    /// Standing registrations on this node, every principal together.
    pub fn total_admitted_subscriptions(&self) -> usize {
        self.admission.total_admitted()
    }

    /// A registration this node admitted has ended without an unsubscription from the client
    /// (compile failure, recompile failure): release its admission and its settled-relay
    /// origin, so a refused-or-failed query leaves nothing behind.
    pub fn forget_client_query(&mut self, client_id: ClientId, query_id: QueryId) {
        self.admission.release(client_id, query_id);
        if let Some(clients) = self.query_origin.get_mut(&query_id) {
            clients.remove(&client_id);
            if clients.is_empty() {
                self.query_origin.remove(&query_id);
            }
        }
    }

    /// Allow authenticated user clients to publish structural schema catalogue
    /// objects directly. Intended for development servers only.
    pub fn with_unprivileged_schema_catalogue_writes(mut self) -> Self {
        self.allow_unprivileged_schema_catalogue_writes = true;
        self
    }

    /// Add multiple durability identities for this node.
    pub fn with_durability_tiers<I>(mut self, tiers: I) -> Self
    where
        I: IntoIterator<Item = DurabilityTier>,
    {
        self.my_tiers.extend(tiers);
        self
    }

    /// True when this runtime instance represents a durability tier identity
    /// (worker/edge/global) rather than a top-level client.
    pub fn has_durability_identity(&self) -> bool {
        !self.my_tiers.is_empty()
    }

    /// True when this node can satisfy acknowledgements for the requested tier
    /// using one of its local durability identities.
    pub fn has_local_durability_at_least(&self, requested_tier: DurabilityTier) -> bool {
        self.my_tiers
            .iter()
            .any(|local_tier| *local_tier >= requested_tier)
    }

    /// Return this node's local durability identities.
    pub fn local_durability_tiers(&self) -> HashSet<DurabilityTier> {
        self.my_tiers.clone()
    }

    /// True when this runtime currently has at least one upstream server.
    pub fn has_connected_servers(&self) -> bool {
        !self.servers.is_empty()
    }

    /// Return the strongest durability tier this node can attest to locally.
    pub fn max_local_durability_tier(&self) -> Option<DurabilityTier> {
        self.my_tiers.iter().copied().max()
    }

    /// The durability tier at which this runtime considers a batch settled:
    /// batches confirmed at or above this tier need no further reconciliation,
    /// replay, or retained bookkeeping on this node.
    ///
    /// With an upstream server registered or pending the target is
    /// `GlobalServer` (local fates are provisional until the upstream
    /// confirms). Without one, this node's own strongest tier is terminal:
    /// there is nobody else to wait for.
    pub fn settlement_target(&self) -> DurabilityTier {
        if self.has_servers_or_pending_servers() {
            DurabilityTier::GlobalServer
        } else {
            self.max_local_durability_tier()
                .unwrap_or(DurabilityTier::Local)
        }
    }

    /// True when `fate` is terminal at `target`: rejected outright, or
    /// confirmed at/above it. `Missing` is neither — it pends retransmission,
    /// not retirement.
    pub fn fate_settled_at(fate: &BatchFate, target: DurabilityTier) -> bool {
        matches!(fate, BatchFate::Rejected { .. })
            || fate.confirmed_tier().is_some_and(|tier| tier >= target)
    }

    /// True when a batch with `fate` still pends settlement reconciliation at
    /// `target`. `Missing` is excluded: it never retires bookkeeping, but it
    /// is owned by the retransmission paths: the live fate handler resends
    /// immediately, and the pending-set derivation special-cases stored
    /// `Missing` fates for submissions it still retains.
    pub fn fate_needs_settlement_at(fate: Option<&BatchFate>, target: DurabilityTier) -> bool {
        match fate {
            None => true,
            Some(BatchFate::Missing { .. }) => false,
            Some(fate) => !Self::fate_settled_at(fate, target),
        }
    }

    /// [`Self::fate_settled_at`] against this node's settlement target.
    pub fn batch_fate_is_settled(&self, fate: &BatchFate) -> bool {
        Self::fate_settled_at(fate, self.settlement_target())
    }

    /// [`Self::fate_needs_settlement_at`] against this node's settlement
    /// target.
    pub fn batch_needs_settlement(&self, fate: Option<&BatchFate>) -> bool {
        Self::fate_needs_settlement_at(fate, self.settlement_target())
    }

    /// Approximate heap-backed memory owned by sync state, grouped for benches.
    ///
    /// Returns `(catalogue, connections, subscriptions, queues, total)`.
    pub fn memory_size(&self) -> (usize, usize, usize, usize, usize) {
        let mut catalogue = 0usize;
        for (object_id, entry) in &self.catalogue_entries {
            catalogue += std::mem::size_of_val(object_id);
            catalogue += std::mem::size_of_val(entry);
            catalogue += 48;
        }

        let mut connections = 0usize;
        for (server_id, state) in &self.servers {
            connections += std::mem::size_of_val(server_id);
            connections += std::mem::size_of_val(state);
            connections += 48;
            connections += state.sent_metadata.len() * std::mem::size_of::<ObjectId>();
            for ((object_id, branch_name), batch_ids) in &state.sent_batch_ids {
                connections += std::mem::size_of_val(object_id);
                connections += std::mem::size_of_val(branch_name);
                connections += batch_ids.len() * std::mem::size_of::<BatchId>();
                connections += 48;
            }
        }
        for (client_id, state) in &self.clients {
            connections += std::mem::size_of_val(client_id);
            connections += std::mem::size_of_val(state);
            connections += 48;
            connections += state.sent_metadata.len() * std::mem::size_of::<ObjectId>();
            if let Some(session) = &state.session {
                connections += session.user_id.len();
            }
            for ((object_id, branch_name), batch_ids) in &state.sent_batch_ids {
                connections += std::mem::size_of_val(object_id);
                connections += std::mem::size_of_val(branch_name);
                connections += batch_ids.len() * std::mem::size_of::<BatchId>();
                connections += 48;
            }
        }
        connections += self.my_tiers.len() * std::mem::size_of::<DurabilityTier>();

        let mut subscriptions = 0usize;
        for state in self.clients.values() {
            for (query_id, scope) in &state.queries {
                subscriptions += std::mem::size_of_val(query_id);
                subscriptions += std::mem::size_of_val(scope);
                subscriptions += scope.scope.len() * std::mem::size_of::<(ObjectId, BranchName)>();
                if let Some(session) = &scope.session {
                    subscriptions += session.user_id.len();
                }
                subscriptions += 48;
            }
        }
        for (row_batch_key, clients) in &self.row_batch_interest {
            subscriptions += std::mem::size_of_val(row_batch_key);
            subscriptions += clients.len() * std::mem::size_of::<ClientId>();
            subscriptions += 48;
        }
        for (batch_id, clients) in &self.batch_fate_interest {
            subscriptions += std::mem::size_of_val(batch_id);
            subscriptions += clients.len() * std::mem::size_of::<ClientId>();
            subscriptions += 48;
        }
        for (query_id, clients) in &self.query_origin {
            subscriptions += std::mem::size_of_val(query_id);
            subscriptions += clients.len() * std::mem::size_of::<ClientId>();
            subscriptions += 48;
        }
        for (key, scope) in &self.remote_query_scopes {
            subscriptions += std::mem::size_of_val(key);
            subscriptions += scope.len() * std::mem::size_of::<(ObjectId, BranchName)>();
            subscriptions += 48;
        }

        let queues = self.inbox.len() * std::mem::size_of::<InboxEntry>()
            + self.outbox.len() * std::mem::size_of::<OutboxEntry>()
            + self.pending_permission_checks.len() * std::mem::size_of::<PendingPermissionCheck>()
            + self.pending_query_subscriptions.len()
                * std::mem::size_of::<PendingQuerySubscription>()
            + self.pending_query_unsubscriptions.len()
                * std::mem::size_of::<PendingQueryUnsubscription>()
            + self.pending_row_visibility_changes.len()
                * std::mem::size_of::<RowVisibilityChange>()
            + self.pending_catalogue_updates.len() * std::mem::size_of::<CatalogueEntry>()
            + self.pending_query_settled.len() * std::mem::size_of::<PendingQuerySettled>()
            + self.pending_batch_fates.len() * std::mem::size_of::<BatchFate>()
            + self
                .pending_client_batch_fates
                .values()
                .map(|batch_ids| {
                    std::mem::size_of::<ClientId>()
                        + batch_ids.len() * std::mem::size_of::<BatchId>()
                })
                .sum::<usize>();

        let total = catalogue + connections + subscriptions + queues;
        (catalogue, connections, subscriptions, queues, total)
    }

    // ========================================================================
    // Connection Management
    // ========================================================================

    /// Add a server connection using storage-backed current-state replay.
    pub fn add_server_with_storage<H: Storage>(
        &mut self,
        server_id: ServerId,
        skip_catalogue_sync: bool,
        storage: &H,
    ) {
        self.pending_servers.remove(&server_id);
        self.servers.insert(server_id, ServerState::default());
        // TODO: this proved to be too resource intensive, replace with a more robust
        // full-storage reconciliation strategy
        // self.queue_full_sync_to_server_from_storage(server_id, storage);
        if !skip_catalogue_sync {
            self.queue_catalogue_sync_to_server_from_storage(server_id, storage);
        }
    }

    pub fn add_pending_server(&mut self, server_id: ServerId) {
        if self.servers.contains_key(&server_id) {
            return;
        }
        self.pending_servers.insert(server_id, Instant::now());
    }

    /// A connection attempt failed. The transport keeps its outbox across attempts, so every
    /// registration pushed while the upstream was pending is still buffered there and will be
    /// delivered on the next connection. The `pending_server_query_subscriptions` markers say
    /// exactly that — "the transport holds this registration" — and must survive here; only
    /// `remove_server`, which tears the transport down, may clear them. Clearing them on a
    /// failed attempt made a later cancellation skip the unsubscription for that server, and
    /// the buffered registration then landed with nothing to withdraw it.
    pub fn remove_pending_server(&mut self, server_id: ServerId) {
        self.pending_servers.remove(&server_id);
    }

    pub fn server_ids(&self) -> impl Iterator<Item = ServerId> + '_ {
        self.servers.keys().copied()
    }

    pub fn has_servers_or_pending_servers(&self) -> bool {
        if !self.servers.is_empty() {
            return true;
        }
        if self.pending_servers.is_empty() {
            return false;
        }
        let now = Instant::now();
        self.pending_servers
            .values()
            .any(|since| now.duration_since(*since) < PENDING_SERVER_TIMEOUT)
    }

    pub fn request_batch_fates_from_server(
        &mut self,
        server_id: ServerId,
        mut batch_ids: Vec<crate::row_histories::BatchId>,
    ) {
        if batch_ids.is_empty() {
            return;
        }
        batch_ids.sort();
        batch_ids.dedup();

        self.outbox.push(OutboxEntry {
            destination: Destination::Server(server_id),
            payload: SyncPayload::BatchFateNeeded { batch_ids },
        });
    }

    pub fn seal_batch_to_servers(&mut self, submission: SealedBatchSubmission) {
        for server_id in self.outbound_server_ids() {
            self.outbox.push(OutboxEntry {
                destination: Destination::Server(server_id),
                payload: SyncPayload::SealBatch {
                    submission: submission.clone(),
                },
            });
        }
    }

    fn outbound_server_ids(&self) -> Vec<ServerId> {
        let now = Instant::now();
        let mut server_ids: Vec<_> = self.servers.keys().copied().collect();
        server_ids.extend(
            self.pending_servers
                .iter()
                .filter(|(_, since)| now.duration_since(**since) < PENDING_SERVER_TIMEOUT)
                .map(|(server_id, _)| *server_id),
        );
        server_ids
    }

    /// Remove a server connection.
    pub fn remove_server(&mut self, server_id: ServerId) {
        self.servers.remove(&server_id);
        self.pending_servers.remove(&server_id);
        self.pending_server_query_subscriptions
            .retain(|(id, _)| *id != server_id);
        let mut removed_query_ids = HashSet::new();
        self.remote_query_scopes
            .retain(|(remote_server_id, query_id), _| {
                let keep = *remote_server_id != server_id;
                if !keep {
                    removed_query_ids.insert(*query_id);
                }
                keep
            });
        self.remote_query_scope_tiers
            .retain(|(remote_server_id, _), _| *remote_server_id != server_id);
        self.remote_query_scope_dirty.extend(removed_query_ids);
    }

    /// Add a client connection without automatically replaying catalogue state.
    pub fn add_client(&mut self, client_id: ClientId) {
        self.clients.insert(client_id, ClientState::default());
        // A fresh connection is new information about what the peer can send, so whatever
        // this authority stopped answering for the previous one gets another chance.
        self.missing_answers.remove(&client_id);
    }

    /// Add a client connection using storage-backed catalogue replay.
    pub fn add_client_with_storage<H: Storage>(&mut self, storage: &H, client_id: ClientId) {
        self.add_client(client_id);
        self.queue_catalogue_sync_to_client_from_storage(client_id, storage);
    }

    /// Replay catalogue entries to a client when its digest is missing or stale.
    ///
    /// Returns true when a replay was queued.
    pub fn queue_catalogue_sync_to_client_if_hash_mismatch<H: Storage>(
        &mut self,
        storage: &H,
        client_id: ClientId,
        remote_catalogue_state_hash: Option<&str>,
        local_catalogue_state_hash: &str,
    ) -> bool {
        if remote_catalogue_state_hash == Some(local_catalogue_state_hash) {
            return false;
        }

        self.queue_catalogue_sync_to_client_from_storage(client_id, storage);
        true
    }

    /// Remove a client connection and all associated state.
    ///
    /// Returns `false` if the client has unprocessed inbox entries — the
    /// caller should retry later to avoid dropping data that hasn't been
    /// persisted to storage yet.
    pub fn remove_client(&mut self, client_id: ClientId) -> bool {
        let has_inbox = self
            .inbox
            .iter()
            .any(|e| e.source == Source::Client(client_id));

        if has_inbox {
            tracing::warn!(
                %client_id,
                "skipping reap: client has unprocessed inbox entries"
            );
            return false;
        }

        // A reaped client's unconfirmed rows go with it: its state is rebuilt from
        // scratch on the next connection.
        self.pending_client_deliveries.remove(&client_id);
        self.missing_answers.remove(&client_id);
        self.admission.release_client(client_id);
        self.clients.remove(&client_id);
        // Clean up interest map
        self.row_batch_interest.retain(|_, clients| {
            clients.remove(&client_id);
            !clients.is_empty()
        });
        self.batch_fate_interest.retain(|_, clients| {
            clients.remove(&client_id);
            !clients.is_empty()
        });
        // Clean up query origin map
        self.query_origin.retain(|_, clients| {
            clients.remove(&client_id);
            !clients.is_empty()
        });
        // Clean up pending queues
        self.pending_permission_checks
            .retain(|c| c.client_id != client_id);
        self.pending_query_subscriptions
            .retain(|s| s.client_id != client_id);
        self.pending_query_unsubscriptions
            .retain(|u| u.client_id != client_id);
        // Drop queued outbox messages for this client
        self.outbox
            .retain(|e| e.destination != Destination::Client(client_id));
        true
    }

    /// Get server state.
    pub fn get_server(&self, server_id: ServerId) -> Option<&ServerState> {
        self.servers.get(&server_id)
    }

    pub fn has_servers(&self) -> bool {
        !self.servers.is_empty()
    }

    /// Get client state.
    pub fn get_client(&self, client_id: ClientId) -> Option<&ClientState> {
        self.clients.get(&client_id)
    }

    /// Set the session for a client.
    pub fn set_client_session(&mut self, client_id: ClientId, session: Session) {
        if let Some(client) = self.clients.get_mut(&client_id) {
            client.session = Some(session);
        }
    }

    /// Note that a client is on a fresh connection.
    ///
    /// A reconnect does not always mint a new client, and the session is not a reliable
    /// marker either — the same user reconnecting presents the same session value.
    /// `ensure_client_with_session` updates the client in place, and the server pulls a
    /// reconnecting client back out of the disconnect candidates rather than reaping it,
    /// so nothing else here observes the new socket.
    ///
    /// A fresh connection is new information about what the peer can send, so it re-arms
    /// the answers this authority stopped giving. That is what keeps a bounded answer a
    /// deferral rather than an abandonment.
    pub fn note_client_connected(&mut self, client_id: ClientId) {
        self.missing_answers.remove(&client_id);
    }

    /// Set the role for a client.
    pub fn set_client_role(&mut self, client_id: ClientId, role: ClientRole) {
        if let Some(client) = self.clients.get_mut(&client_id) {
            if client.role != role {
                self.role_changed.push(client_id);
            }
            client.role = role;
        }
    }

    /// v18 item 6: drain the clients whose role changed since the last drain.
    pub fn take_role_changes(&mut self) -> Vec<ClientId> {
        std::mem::take(&mut self.role_changed)
    }

    /// v18 item 6: whether an installed transport still sits in `pending_servers` within
    /// `PENDING_SERVER_TIMEOUT` — the same test `has_servers_or_pending_servers` applies to
    /// the pending half. Polled once per pass; its true→false flip un-stalls local units.
    pub fn has_live_pending_servers(&self) -> bool {
        let now = Instant::now();
        self.pending_servers
            .values()
            .any(|since| now.duration_since(*since) < PENDING_SERVER_TIMEOUT)
    }

    // ========================================================================
    // Outbox / Inbox
    // ========================================================================

    /// Take all outbox entries, clearing the outbox.
    /// Apply the delivery claims the receiver has confirmed.
    ///
    /// Only clears when the confirmation names the batch still owed: a late confirmation
    /// for a superseded batch leaves the newer one outstanding.
    /// Note that this node applied a row, so it can be reported upstream.
    pub fn note_applied_row(&mut self, row_id: ObjectId, branch: &str, batch_id: BatchId) {
        if !self.upstream_supports_delivery_acks {
            return;
        }
        self.applied_rows_to_confirm.push(ConfirmedRow {
            row_id,
            branch: branch.to_string(),
            batch_id,
        });
    }

    /// Record whether the upstream server understands confirmations.
    pub fn set_upstream_supports_delivery_acks(&mut self, supported: bool) {
        self.upstream_supports_delivery_acks = supported;
    }

    /// Queue one confirmation for everything applied since the last drain.
    ///
    /// Batched deliberately: one message per tick regardless of how many rows landed, and
    /// nothing at all on a tick that applied nothing — so an idle client stays silent.
    pub fn queue_applied_row_confirmations(&mut self, server_id: ServerId) {
        if self.applied_rows_to_confirm.is_empty() {
            return;
        }
        let rows = std::mem::take(&mut self.applied_rows_to_confirm);
        tracing::debug!(
            target: "jazz::delivery",
            %server_id,
            rows = rows.len(),
            "confirming applied rows upstream"
        );
        self.outbox.push(OutboxEntry {
            destination: Destination::Server(server_id),
            payload: SyncPayload::DeliveryConfirmed { rows },
        });
    }

    /// Record whether this client confirms the rows it applies. Set from the handshake.
    pub fn set_client_acks_deliveries(&mut self, client_id: ClientId, acks: bool) {
        if let Some(client) = self.clients.get_mut(&client_id) {
            client.acks_deliveries = acks;
        }
    }

    pub fn confirm_client_deliveries(
        &mut self,
        confirmed: &[(ClientId, ObjectId, BranchName, BatchId)],
    ) {
        for (client_id, row_id, branch_name, batch_id) in confirmed.iter().cloned() {
            let Some(owed) = self.pending_client_deliveries.get_mut(&client_id) else {
                continue;
            };
            let key = (row_id, branch_name);
            let Some(pending) = owed.get(&key).filter(|p| p.batch_id == batch_id).cloned() else {
                continue;
            };
            owed.remove(&key);
            if owed.is_empty() {
                self.pending_client_deliveries.remove(&client_id);
            }
            let Some(client) = self.clients.get_mut(&client_id) else {
                continue;
            };
            tracing::debug!(
                target: "jazz::delivery",
                %client_id, object_id = %row_id, ?batch_id,
                "receiver confirmed the row"
            );
            if pending.include_metadata {
                client.sent_metadata.insert(row_id);
            }
            client
                .sent_batch_ids
                .entry((row_id, pending.branch_name))
                .or_default()
                .record_delivery(batch_id, &pending.parent_ids);
        }
    }

    /// Whether this client is owed rows it never confirmed.
    pub fn client_has_undelivered_payloads(&self, client_id: ClientId) -> bool {
        self.pending_client_deliveries
            .get(&client_id)
            .is_some_and(|owed| owed.values().any(|pending| !pending.demoted))
    }

    /// The rows this client is owed, as scope entries.
    ///
    /// A re-offer is built from this rather than from the whole scope: the trigger is
    /// per-client, so without narrowing one unconfirmed row would retransmit an entire
    /// query on every resubscribe — and a phone resubscribes whenever it comes back to the
    /// foreground.
    fn undelivered_scope_entries(&self, client_id: ClientId) -> HashSet<(ObjectId, BranchName)> {
        self.pending_client_deliveries
            .get(&client_id)
            .map(|owed| owed.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// Publish how much is owed right now. Called once per tick; the map is empty on a
    /// healthy server, so this is two length checks.
    pub fn publish_undelivered_gauges(&self) {
        crate::query_manager::settle_cost::set_gauge(
            &crate::query_manager::settle_cost::UNDELIVERED_PAYLOADS,
            self.pending_client_deliveries
                .values()
                .map(|owed| owed.len() as u64)
                .sum(),
        );
        crate::query_manager::settle_cost::set_gauge(
            &crate::query_manager::settle_cost::UNDELIVERED_CLIENTS,
            self.pending_client_deliveries.len() as u64,
        );
    }

    #[cfg(any(test, feature = "test"))]
    pub fn outbox_len_for_test(&self) -> usize {
        self.outbox.len()
    }

    pub fn take_outbox(&mut self) -> Vec<OutboxEntry> {
        std::mem::take(&mut self.outbox)
    }

    /// Restore previously dequeued outbox entries ahead of any newly queued ones.
    pub(crate) fn prepend_outbox(&mut self, mut entries: Vec<OutboxEntry>) {
        if entries.is_empty() {
            return;
        }
        entries.append(&mut self.outbox);
        self.outbox = entries;
    }

    /// Get a reference to the outbox (for checking if empty).
    pub fn outbox(&self) -> &[OutboxEntry] {
        &self.outbox
    }

    /// Push an entry to the inbox for processing.
    pub fn push_inbox(&mut self, entry: InboxEntry) {
        self.inbox.push(entry);
    }

    /// Process all inbox entries.
    pub fn process_inbox<H: Storage>(&mut self, storage: &mut H) {
        let entries = std::mem::take(&mut self.inbox);
        for entry in entries {
            self.process_inbox_entry(storage, entry);
        }
        let pending_client_batch_fates = std::mem::take(&mut self.pending_client_batch_fates);
        for (client_id, batch_ids) in pending_client_batch_fates {
            self.respond_to_batch_fate_request(
                storage,
                Destination::Client(client_id),
                batch_ids.into_iter().collect(),
            );
        }
    }

    // ========================================================================
    // Pending Query Subscriptions
    // ========================================================================

    /// Take pending query subscriptions for QueryManager to process.
    ///
    /// QueryManager will build QueryGraphs for these and call back with computed scopes.
    pub fn take_pending_query_subscriptions(&mut self) -> Vec<PendingQuerySubscription> {
        std::mem::take(&mut self.pending_query_subscriptions)
    }

    pub fn has_pending_query_subscriptions(&self) -> bool {
        !self.pending_query_subscriptions.is_empty()
    }

    /// Re-queue pending query subscriptions that couldn't be processed yet.
    ///
    /// Called by QueryManager when schema isn't available for some subscriptions.
    pub fn requeue_pending_query_subscriptions(&mut self, subs: Vec<PendingQuerySubscription>) {
        self.pending_query_subscriptions.extend(subs);
    }

    /// Take pending query unsubscriptions for QueryManager to process.
    ///
    /// QueryManager will remove server-side QueryGraphs and forward upstream.
    pub fn take_pending_query_unsubscriptions(&mut self) -> Vec<PendingQueryUnsubscription> {
        std::mem::take(&mut self.pending_query_unsubscriptions)
    }

    /// Storage-backed version of `set_client_query_scope` that can replay row
    /// objects directly from storage-backed visible rows.
    pub fn set_client_query_scope_with_storage<H: Storage + ?Sized>(
        &mut self,
        storage: &H,
        client_id: ClientId,
        query_id: QueryId,
        scope: HashSet<(ObjectId, BranchName)>,
        session: Option<Session>,
    ) {
        let Some(client) = self.clients.get_mut(&client_id) else {
            return;
        };

        let old_query_scope = client
            .queries
            .get(&query_id)
            .map(|query| query.scope.clone())
            .unwrap_or_default();
        let old_scope: HashSet<(ObjectId, BranchName)> = client
            .queries
            .values()
            .flat_map(|q| q.scope.iter().cloned())
            .collect();

        client.queries.insert(
            query_id,
            QueryScope {
                scope: scope.clone(),
                session,
            },
        );

        let new_scope: HashSet<(ObjectId, BranchName)> = client
            .queries
            .values()
            .flat_map(|q| q.scope.iter().cloned())
            .collect();

        let no_longer_visible: HashSet<(ObjectId, BranchName)> =
            old_scope.difference(&new_scope).cloned().collect();
        // Plus whatever this peer never confirmed that is still in scope. For a returning
        // peer the difference is empty — its scope did not change, it was away — so the
        // owed rows are the whole of the re-offer. Re-offering a row the peer already holds
        // is harmless: applying an identical batch is an idempotent no-op on the receiver.
        let newly_visible_for_query: Vec<(ObjectId, BranchName)> = {
            let mut entries: HashSet<(ObjectId, BranchName)> =
                scope.difference(&old_query_scope).cloned().collect();
            let owed = self.undelivered_scope_entries(client_id);
            let re_offered = owed.intersection(&scope).count();
            if re_offered > 0 {
                tracing::info!(
                    target: "jazz::conn",
                    %client_id,
                    query_id = query_id.0,
                    rows = re_offered,
                    scope = scope.len(),
                    "re-offering rows the peer never confirmed"
                );
            }
            entries.extend(owed.intersection(&scope).cloned());
            entries.into_iter().collect()
        };

        self.prune_client_scope_tracking(client_id, &no_longer_visible);

        let mut newly_visible_batch_ids = HashSet::new();
        for (object_id, branch_name) in newly_visible_for_query {
            if let Some(batch_id) = self.queue_initial_row_to_client_with_storage(
                storage,
                client_id,
                object_id,
                branch_name,
                true,
            ) {
                newly_visible_batch_ids.insert(batch_id);
            }
        }

        // Initial query scope delivery can include many rows from the same
        // sealed batch. Queue rows first so client interest is complete, then
        // send one fate per batch instead of one growing fate per row.
        for batch_id in newly_visible_batch_ids {
            if let Some(fate) = self.load_batch_fate_by_batch_id_from_storage(storage, batch_id) {
                self.queue_batch_fate_to_client(client_id, fate);
            }
        }
    }

    /// Drop a client's query subscription state.
    ///
    /// Removes per-query scope and origin tracking.
    pub fn drop_client_query_subscription(&mut self, client_id: ClientId, query_id: QueryId) {
        if let Some(client) = self.clients.get_mut(&client_id) {
            let old_scope: HashSet<(ObjectId, BranchName)> = client
                .queries
                .values()
                .flat_map(|q| q.scope.iter().cloned())
                .collect();
            client.queries.remove(&query_id);
            let new_scope: HashSet<(ObjectId, BranchName)> = client
                .queries
                .values()
                .flat_map(|q| q.scope.iter().cloned())
                .collect();
            let no_longer_visible: HashSet<(ObjectId, BranchName)> =
                old_scope.difference(&new_scope).cloned().collect();
            self.prune_client_scope_tracking(client_id, &no_longer_visible);
        }

        if let Some(clients) = self.query_origin.get_mut(&query_id) {
            clients.remove(&client_id);
            if clients.is_empty() {
                self.query_origin.remove(&query_id);
            }
        }
    }

    fn prune_client_scope_tracking(
        &mut self,
        client_id: ClientId,
        removed_scope: &HashSet<(ObjectId, BranchName)>,
    ) {
        if removed_scope.is_empty() {
            return;
        }

        let mut removed_row_batches = Vec::new();
        let Some(client) = self.clients.get_mut(&client_id) else {
            return;
        };

        // The sent set is the delivered frontier (fix D1), so this cleanup
        // only reaches interest entries for frontier ids. Entries for pruned
        // ancestor ids outlive a scope drop and are reclaimed on client
        // removal (`remove_client`), same as before D1 for clients that never
        // shrink their scope.
        for &(object_id, branch_name) in removed_scope {
            if let Some(batch_ids) = client.sent_batch_ids.remove(&(object_id, branch_name)) {
                removed_row_batches.extend(
                    batch_ids
                        .into_iter()
                        .map(|batch_id| RowBatchKey::new(object_id, branch_name, batch_id)),
                );
            }
        }

        for key in removed_row_batches {
            if let Some(clients) = self.row_batch_interest.get_mut(&key) {
                clients.remove(&client_id);
                if clients.is_empty() {
                    self.row_batch_interest.remove(&key);
                }
            }
        }
    }

    /// Send a QuerySubscription to all connected servers.
    ///
    /// Called by QueryManager when a client creates a subscription that should
    /// be forwarded upstream for server-side evaluation.
    pub fn send_query_subscription_to_servers(
        &mut self,
        query_id: QueryId,
        query: Query,
        session: Option<Session>,
        required_tier: Option<DurabilityTier>,
        propagation: QueryPropagation,
        policy_context_tables: Vec<String>,
    ) {
        let server_ids = self.outbound_server_ids();
        tracing::trace!(
            server_count = server_ids.len(),
            query_id = query_id.0,
            table = %query.table,
            ?propagation,
            "jazz trace send query subscription to servers"
        );
        for server_id in server_ids {
            self.send_query_subscription_to_server(
                server_id,
                OutgoingQuerySubscription {
                    query_id,
                    query: query.clone(),
                    session: session.clone(),
                    required_tier,
                    propagation,
                    policy_context_tables: policy_context_tables.clone(),
                },
            );
        }
    }

    /// Send a QuerySubscription to one specific server.
    ///
    /// Used when replaying existing subscriptions after a late server connect.
    pub(crate) fn send_query_subscription_to_server(
        &mut self,
        server_id: ServerId,
        subscription: OutgoingQuerySubscription,
    ) {
        let OutgoingQuerySubscription {
            query_id,
            query,
            session,
            required_tier,
            propagation,
            policy_context_tables,
        } = subscription;
        let is_connected = self.servers.contains_key(&server_id);
        let is_pending = self.pending_servers.contains_key(&server_id);
        if !is_connected && !is_pending {
            return;
        }

        tracing::trace!(
            %server_id,
            query_id = query_id.0,
            table = %query.table,
            ?propagation,
            "jazz trace sending query subscription upstream"
        );
        self.outbox.push(OutboxEntry {
            destination: Destination::Server(server_id),
            payload: SyncPayload::QuerySubscription {
                query_id,
                query: Box::new(query),
                session,
                required_tier,
                propagation,
                policy_context_tables,
            },
        });

        if !is_connected && is_pending {
            self.pending_server_query_subscriptions
                .insert((server_id, query_id));
        }
    }

    pub(crate) fn consume_pending_query_subscription_marker(
        &mut self,
        server_id: ServerId,
        query_id: QueryId,
    ) -> bool {
        self.pending_server_query_subscriptions
            .remove(&(server_id, query_id))
    }

    /// Send a QueryUnsubscription to all connected servers.
    ///
    /// Called by QueryManager when a client unsubscribes from a synced query.
    pub fn send_query_unsubscription_to_servers(&mut self, query_id: QueryId) {
        // Every server a registration for this query could have reached — not only the ones
        // `outbound_server_ids` would still write to. A registration pushed while the
        // upstream was pending sits in the transport's outbox and is delivered on the next
        // connection however long that takes, so an unsubscription filtered by the pending
        // age would leave that registration alive at the server with nothing local left to
        // withdraw it (v18 item 1, diff review). An unsubscription for a query the server
        // never registered is a no-op there.
        let mut server_ids: HashSet<ServerId> = self.servers.keys().copied().collect();
        server_ids.extend(self.pending_servers.keys().copied());
        server_ids.extend(
            self.pending_server_query_subscriptions
                .iter()
                .filter(|(_, pending_query_id)| *pending_query_id == query_id)
                .map(|(server_id, _)| *server_id),
        );
        for server_id in server_ids {
            self.outbox.push(OutboxEntry {
                destination: Destination::Server(server_id),
                payload: SyncPayload::QueryUnsubscription { query_id },
            });
        } // The marker only spares a replay for a subscription that still exists.
        self.pending_server_query_subscriptions
            .retain(|(_, pending_query_id)| *pending_query_id != query_id);
    }

    /// Test hook: make a pending upstream look older than it is, so the
    /// `PENDING_SERVER_TIMEOUT` branches can be exercised without sleeping.
    #[cfg(any(test, feature = "test"))]
    pub fn age_pending_server_for_test(&mut self, server_id: ServerId, by: Duration) {
        if let Some(since) = self.pending_servers.get_mut(&server_id)
            && let Some(older) = since.checked_sub(by)
        {
            *since = older;
        }
    }

    #[cfg(any(test, feature = "test"))]
    pub fn pending_server_query_subscription_count_for_test(&self) -> usize {
        self.pending_server_query_subscriptions.len()
    }

    /// Take pending QuerySettled notifications for QueryManager to process.
    pub fn take_pending_query_settled(&mut self) -> Vec<PendingQuerySettled> {
        std::mem::take(&mut self.pending_query_settled)
    }

    /// Re-queue QuerySettled notifications that are still blocked on stream sequencing.
    pub fn requeue_pending_query_settled(&mut self, pending: Vec<PendingQuerySettled>) {
        self.pending_query_settled.extend(pending);
    }

    /// Take pending query rejections for QueryManager to process.
    pub fn take_pending_query_rejections(&mut self) -> Vec<PendingQueryRejection> {
        std::mem::take(&mut self.pending_query_rejections)
    }

    /// Return the union of latest upstream scope snapshots for this query.
    pub fn remote_query_scope(&self, query_id: QueryId) -> HashSet<(ObjectId, BranchName)> {
        self.remote_query_scopes
            .iter()
            .filter(|((_, remote_query_id), _)| *remote_query_id == query_id)
            .flat_map(|(_, scope)| scope.iter().copied())
            .collect()
    }

    /// Return the union of latest upstream scope snapshots at or above the
    /// requested tier for this query.
    pub fn remote_query_scope_at_least(
        &self,
        query_id: QueryId,
        requested_tier: DurabilityTier,
    ) -> HashSet<(ObjectId, BranchName)> {
        self.remote_query_scopes
            .iter()
            .filter(|((server_id, remote_query_id), _)| {
                *remote_query_id == query_id
                    && self
                        .remote_query_scope_tiers
                        .get(&(*server_id, *remote_query_id))
                        .is_some_and(|tier| *tier >= requested_tier)
            })
            .flat_map(|(_, scope)| scope.iter().copied())
            .collect()
    }

    /// Whether we have received at least one upstream scope snapshot for this query.
    pub fn has_remote_query_scope_snapshot(&self, query_id: QueryId) -> bool {
        self.remote_query_scopes
            .keys()
            .any(|(_, remote_query_id)| *remote_query_id == query_id)
    }

    /// Whether we have received at least one upstream scope snapshot for this
    /// query at a tier that can authoritatively satisfy the requested tier.
    pub fn has_remote_query_scope_snapshot_at_least(
        &self,
        query_id: QueryId,
        requested_tier: DurabilityTier,
    ) -> bool {
        self.remote_query_scope_tiers
            .iter()
            .any(|((_, remote_query_id), tier)| {
                *remote_query_id == query_id && *tier >= requested_tier
            })
    }

    /// Take query ids whose upstream scope changed since the last process pass.
    pub fn take_remote_query_scope_dirty(&mut self) -> HashSet<QueryId> {
        std::mem::take(&mut self.remote_query_scope_dirty)
    }

    /// Take pending replayable batch fates for RuntimeCore to process.
    pub fn take_pending_batch_fates(&mut self) -> Vec<BatchFate> {
        std::mem::take(&mut self.pending_batch_fates)
    }

    pub fn pending_batch_fates(&self) -> &[BatchFate] {
        &self.pending_batch_fates
    }

    pub fn push_pending_batch_fate(&mut self, fate: BatchFate) {
        self.pending_batch_fates.push(fate);
    }

    /// Take pending row visibility changes for QueryManager to materialize
    /// into indices and subscriptions.
    pub fn take_pending_row_visibility_changes(&mut self) -> Vec<RowVisibilityChange> {
        std::mem::take(&mut self.pending_row_visibility_changes)
    }

    /// Take pending catalogue/system entry updates for QueryManager/SchemaManager.
    pub fn take_pending_catalogue_updates(&mut self) -> Vec<CatalogueEntry> {
        std::mem::take(&mut self.pending_catalogue_updates)
    }

    /// Requeue row visibility changes that could not be processed yet,
    /// typically because the corresponding schema has not been activated yet.
    pub fn requeue_pending_row_visibility_changes(&mut self, updates: Vec<RowVisibilityChange>) {
        self.pending_row_visibility_changes.extend(updates);
    }

    /// Emit a QuerySettled notification to a client.
    ///
    /// Called by QueryManager when a server subscription settles for the first time.
    pub fn emit_query_settled(
        &mut self,
        client_id: ClientId,
        query_id: QueryId,
        tier: DurabilityTier,
        scope: &HashSet<(ObjectId, BranchName)>,
    ) {
        tracing::trace!(
            %client_id,
            query_id = query_id.0,
            ?tier,
            scope_len = scope.len(),
            "jazz trace emitting query settled to client"
        );
        self.outbox.push(OutboxEntry {
            destination: Destination::Client(client_id),
            payload: SyncPayload::QuerySettled {
                query_id,
                tier,
                scope: sorted_query_scope_snapshot(scope),
                through_seq: 0,
            },
        });
    }

    pub(crate) fn relay_query_settled_to_origins(
        &mut self,
        server_id: ServerId,
        query_id: QueryId,
        tier: DurabilityTier,
    ) {
        let Some(scope) = self
            .remote_query_scopes
            .get(&(server_id, query_id))
            .cloned()
        else {
            return;
        };
        let Some(clients) = self.query_origin.get(&query_id).cloned() else {
            return;
        };

        for client_id in clients {
            self.emit_query_settled(client_id, query_id, tier, &scope);
        }
    }

    /// Emit a schema warning to a client.
    pub fn emit_schema_warning(&mut self, client_id: ClientId, warning: SchemaWarning) {
        self.outbox.push(OutboxEntry {
            destination: Destination::Client(client_id),
            payload: SyncPayload::SchemaWarning(warning),
        });
    }

    /// Emit a query subscription rejection error to a client.
    pub fn emit_query_subscription_rejected(
        &mut self,
        client_id: ClientId,
        query_id: QueryId,
        code: impl Into<String>,
        reason: impl Into<String>,
    ) {
        self.outbox.push(OutboxEntry {
            destination: Destination::Client(client_id),
            payload: SyncPayload::Error(SyncError::QuerySubscriptionRejected {
                query_id,
                code: code.into(),
                reason: reason.into(),
            }),
        });
    }
}

fn sorted_query_scope_snapshot(
    scope: &HashSet<(ObjectId, BranchName)>,
) -> Vec<(ObjectId, BranchName)> {
    let mut entries: Vec<_> = scope.iter().copied().collect();
    entries.sort_by(
        |(left_object_id, left_branch), (right_object_id, right_branch)| {
            left_object_id
                .cmp(right_object_id)
                .then_with(|| left_branch.as_str().cmp(right_branch.as_str()))
        },
    );
    entries
}
