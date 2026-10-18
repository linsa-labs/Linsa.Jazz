use std::collections::{HashMap, HashSet};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::batch_fate::{BatchFate, SealedBatchSubmission};
use crate::catalogue::CatalogueEntry;
use crate::object::{BranchName, ObjectId};
use crate::query_manager::policy::Operation;
use crate::query_manager::query::Query;
use crate::query_manager::session::Session;
use crate::query_manager::types::SchemaHash;
use crate::row_histories::{BatchId, StoredRowBatch};

/// Error returned when a policy denies an operation.
#[derive(Debug, Clone)]
pub struct PolicyError {
    pub message: String,
}

// ============================================================================
// ID Types
// ============================================================================

/// Persistence tier — declaration order defines Ord (Local < EdgeServer < GlobalServer).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub enum DurabilityTier {
    Local,
    EdgeServer,
    GlobalServer,
}

/// Unique identifier for a server connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ServerId(pub Uuid);

impl ServerId {
    pub fn new() -> Self {
        Self(Uuid::now_v7())
    }
}

impl Default for ServerId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for ServerId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Unique identifier for a client connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ClientId(pub Uuid);

impl ClientId {
    pub fn new() -> Self {
        Self(Uuid::now_v7())
    }

    /// Parse from UUID string.
    pub fn parse(s: &str) -> Option<Self> {
        Uuid::parse_str(s).ok().map(ClientId)
    }
}

impl Default for ClientId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for ClientId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Unique identifier for a query subscription.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct QueryId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum QueryPropagation {
    #[default]
    #[serde(rename = "full")]
    Full,
    #[serde(rename = "local-only")]
    LocalOnly,
}

/// Unique identifier for a pending permission check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PendingUpdateId(pub u64);

/// Stable identity for one concrete row batch entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RowBatchKey {
    pub row_id: ObjectId,
    pub branch_name: BranchName,
    pub batch_id: BatchId,
}

/// The clients interested in each row batch, held by batch: a batch's fate goes to every
/// client interested in one of its rows, and is asked for once per batch a scope delivers —
/// walking every row batch any client was sent, per batch, cost the server 8 % of a search's
/// tick.
#[derive(Debug, Default, Clone)]
pub(crate) struct RowBatchInterest {
    by_batch: HashMap<BatchId, BatchRows>,
}

/// The rows of one batch some client is interested in. Almost every batch has one row, held
/// inline so that it costs no table of its own; a batch whose rows go down to one is held
/// inline again.
#[derive(Debug, Clone)]
enum BatchRows {
    One((ObjectId, BranchName), HashSet<ClientId>),
    Many(HashMap<(ObjectId, BranchName), HashSet<ClientId>>),
}

impl BatchRows {
    fn iter(&self) -> impl Iterator<Item = (&(ObjectId, BranchName), &HashSet<ClientId>)> {
        let (one, many) = match self {
            BatchRows::One(row, clients) => (Some((row, clients)), None),
            BatchRows::Many(rows) => (None, Some(rows.iter())),
        };
        one.into_iter().chain(many.into_iter().flatten())
    }

    /// A batch of more than one row whose rows went down to one is held inline again.
    fn settle(&mut self) {
        if let BatchRows::Many(rows) = self
            && rows.len() == 1
            && let Some((row, clients)) = std::mem::take(rows).into_iter().next()
        {
            *self = BatchRows::One(row, clients);
        }
    }

    /// Takes `client_id` out of every row; whether any row is left.
    fn remove_client(&mut self, client_id: ClientId) -> bool {
        let held = match self {
            BatchRows::One(_, clients) => {
                clients.remove(&client_id);
                !clients.is_empty()
            }
            BatchRows::Many(rows) => {
                rows.retain(|_, clients| {
                    clients.remove(&client_id);
                    !clients.is_empty()
                });
                !rows.is_empty()
            }
        };
        self.settle();
        held
    }
}

impl RowBatchInterest {
    pub(crate) fn insert(&mut self, key: RowBatchKey, client_id: ClientId) {
        let row = (key.row_id, key.branch_name);
        let Some(rows) = self.by_batch.get_mut(&key.batch_id) else {
            self.by_batch.insert(
                key.batch_id,
                BatchRows::One(row, HashSet::from([client_id])),
            );
            return;
        };
        match rows {
            BatchRows::One(held, clients) if *held == row => {
                clients.insert(client_id);
            }
            BatchRows::One(held, clients) => {
                let mut many = HashMap::with_capacity(2);
                many.insert(*held, std::mem::take(clients));
                many.insert(row, HashSet::from([client_id]));
                *rows = BatchRows::Many(many);
            }
            BatchRows::Many(many) => {
                many.entry(row).or_default().insert(client_id);
            }
        }
    }

    pub(crate) fn remove(&mut self, key: &RowBatchKey, client_id: ClientId) {
        let row = (key.row_id, key.branch_name);
        let Some(rows) = self.by_batch.get_mut(&key.batch_id) else {
            return;
        };
        let held = match rows {
            BatchRows::One(held, clients) => {
                if *held != row {
                    return;
                }
                clients.remove(&client_id);
                !clients.is_empty()
            }
            BatchRows::Many(many) => {
                let Some(clients) = many.get_mut(&row) else {
                    return;
                };
                clients.remove(&client_id);
                if clients.is_empty() {
                    many.remove(&row);
                }
                !many.is_empty()
            }
        };
        if held {
            rows.settle();
        } else {
            self.by_batch.remove(&key.batch_id);
        }
    }

    pub(crate) fn remove_client(&mut self, client_id: ClientId) {
        self.by_batch
            .retain(|_, rows| rows.remove_client(client_id));
    }

    /// Every row batch some client is interested in, with its clients.
    #[cfg(test)]
    pub(crate) fn iter(&self) -> impl Iterator<Item = (RowBatchKey, &HashSet<ClientId>)> {
        self.by_batch.iter().flat_map(|(batch_id, rows)| {
            rows.iter().map(|((row_id, branch_name), clients)| {
                (RowBatchKey::new(*row_id, *branch_name, *batch_id), clients)
            })
        })
    }

    /// How many batches some client is interested in a row of.
    #[cfg(test)]
    pub(crate) fn batch_count(&self) -> usize {
        self.by_batch.len()
    }

    /// Whether the rows of `batch_id` are held inline: one row, no table.
    #[cfg(test)]
    pub(crate) fn holds_inline(&self, batch_id: BatchId) -> bool {
        matches!(self.by_batch.get(&batch_id), Some(BatchRows::One(..)))
    }

    /// Bytes held, by the estimate `SyncManager::memory_size` uses: a batch's entry, which
    /// holds a batch of one row inline, each row of a batch of more than one, and the clients
    /// of each row. A batch of one row and one client counts as the flat map of row batches
    /// it replaced counted one entry.
    pub(crate) fn memory_size(&self) -> usize {
        let clients = |clients: &HashSet<ClientId>| clients.len() * std::mem::size_of::<ClientId>();
        self.by_batch
            .values()
            .map(|rows| {
                let held = match rows {
                    BatchRows::One(_, held) => clients(held),
                    BatchRows::Many(many) => many
                        .values()
                        .map(|held| {
                            std::mem::size_of::<(ObjectId, BranchName)>()
                                + std::mem::size_of::<HashSet<ClientId>>()
                                + clients(held)
                        })
                        .sum(),
                };
                std::mem::size_of::<BatchId>() + std::mem::size_of::<BatchRows>() + held
            })
            .sum()
    }

    /// The row batches of `batch_id` some client is interested in.
    pub(crate) fn keys_of_batch(
        &self,
        batch_id: BatchId,
    ) -> impl Iterator<Item = RowBatchKey> + '_ {
        self.by_batch
            .get(&batch_id)
            .into_iter()
            .flat_map(move |rows| {
                rows.iter().map(move |((row_id, branch_name), _)| {
                    RowBatchKey::new(*row_id, *branch_name, batch_id)
                })
            })
    }

    /// The clients interested in a row of `batch_id`: a client interested in several of its
    /// rows comes once for each.
    pub(crate) fn clients_of_batch(
        &self,
        batch_id: BatchId,
    ) -> impl Iterator<Item = ClientId> + '_ {
        self.by_batch
            .get(&batch_id)
            .into_iter()
            .flat_map(|rows| rows.iter())
            .flat_map(|(_, clients)| clients.iter().copied())
    }

    /// Whether `client_id` is interested in a row of `batch_id`.
    pub(crate) fn batch_has_client(&self, batch_id: BatchId, client_id: ClientId) -> bool {
        self.by_batch
            .get(&batch_id)
            .is_some_and(|rows| rows.iter().any(|(_, clients)| clients.contains(&client_id)))
    }
}

impl RowBatchKey {
    pub fn new(row_id: ObjectId, branch_name: BranchName, batch_id: BatchId) -> Self {
        Self {
            row_id,
            branch_name,
            batch_id,
        }
    }

    pub fn from_row(row: &StoredRowBatch) -> Self {
        Self::new(row.row_id, BranchName::new(&row.branch), row.batch_id)
    }
}

/// Deferred query settlement waiting for stream sequencing prerequisites.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PendingQuerySettled {
    pub server_id: Option<ServerId>,
    pub query_id: QueryId,
    pub tier: DurabilityTier,
    pub through_seq: u64,
}

/// Deferred query rejection waiting for QueryManager to drop local state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingQueryRejection {
    pub query_id: QueryId,
    pub code: String,
    pub reason: String,
}

// ============================================================================
// Client Roles
// ============================================================================

/// Role-based access control for client connections.
///
/// Determines how incoming writes from a client are routed:
/// - `User`: Requires session, ReBAC for rows, rejected for catalogue unless
///   development-only schema auto-push is enabled
/// - `Backend`: Trusted backend data access (rows only, no catalogue writes)
/// - `Admin`: Full access (catalogue + data, no ReBAC)
/// - `Peer`: Trusted relay (server-to-server), bypasses all auth
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ClientRole {
    #[default]
    User,
    Backend,
    Admin,
    Peer,
}

// ============================================================================
// Connection State
// ============================================================================

/// The set of batch ids already sent to a peer for one `(row, branch)`.
///
/// A newtype around the underlying set purely so its `Clone` can be observed.
/// An earlier regression cloned the whole set on every queued batch just to
/// test membership, making each forward O(n) in the history length. Membership
/// is now checked by borrow; the custom `Clone` is instrumented under
/// `cfg(test)` so a guard test can assert the forwarding hot path never clones
/// the set again.
///
/// Since fix D1 the set is a *delivered-frontier cursor*, not the full
/// delivered history: [`Self::record_delivery`] prunes ids dominated by each
/// newly delivered batch, so for serial histories the set stays O(1) instead
/// of growing with every batch ever sent. See `record_delivery` for the
/// invariant.
#[derive(Debug, Default)]
pub struct SentBatchIds(HashSet<BatchId>);

impl SentBatchIds {
    /// Record `batch_id` as delivered to this peer and prune the ids this
    /// delivery dominates: the batch's direct parents.
    ///
    /// This is the frontier-cursor pruning rule (fix D1). A parent id may be
    /// dropped because the just-delivered batch proves the peer's delivered
    /// set covers it: any future ancestor walk from a newer batch reaches
    /// `batch_id` before (or instead of) the parent, and a dedup miss on a
    /// pruned id merely re-sends a batch the peer already applied — an
    /// idempotent no-op on the receiver (`apply_row_batch` early-returns on an
    /// identical stored batch).
    ///
    /// Invariant (hard): pruning must only ever *under*-claim. Ids are only
    /// removed, never invented, and only ids listed as parents of a batch
    /// being recorded as delivered are removed — provably ancestors of a
    /// delivered batch. Over-claim (skipping a batch the receiver actually
    /// lacks) would make the receiver drop rows on `ParentNotFound` with no
    /// repair protocol, so no recency/LRU eviction is allowed here in any
    /// form.
    ///
    /// Termination property preserved: for a serial history the direct parent
    /// of the next write is exactly the last recorded batch, which this rule
    /// never removes (only *its* parents), so the ancestor DFS in
    /// `queue_row_to_server_with_missing_parents` still terminates on its
    /// first membership probe.
    pub fn record_delivery(&mut self, batch_id: BatchId, parents: &[BatchId]) {
        self.0.insert(batch_id);
        for parent in parents {
            // A self-parent would be malformed input; never let it evict the
            // id we just recorded.
            if *parent != batch_id {
                self.0.remove(parent);
            }
        }
    }
}

impl Clone for SentBatchIds {
    fn clone(&self) -> Self {
        #[cfg(test)]
        sent_batch_clone_probe::record();
        Self(self.0.clone())
    }
}

impl std::ops::Deref for SentBatchIds {
    type Target = HashSet<BatchId>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for SentBatchIds {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl IntoIterator for SentBatchIds {
    type Item = BatchId;
    type IntoIter = std::collections::hash_set::IntoIter<BatchId>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<const N: usize> From<[BatchId; N]> for SentBatchIds {
    fn from(batch_ids: [BatchId; N]) -> Self {
        Self(HashSet::from(batch_ids))
    }
}

/// Test-only probe counting clones of [`SentBatchIds`] on the current thread, so
/// a guard test can assert the forwarding hot path checks membership by borrow
/// rather than by cloning the whole set.
#[cfg(test)]
pub(crate) mod sent_batch_clone_probe {
    use std::cell::Cell;

    thread_local! {
        static CLONES: Cell<usize> = Cell::new(0);
    }

    pub(crate) fn reset() {
        CLONES.with(|clones| clones.set(0));
    }

    pub(crate) fn record() {
        CLONES.with(|clones| clones.set(clones.get() + 1));
    }

    pub(crate) fn count() -> usize {
        CLONES.with(Cell::get)
    }
}

/// Tracking state for a connected server.
#[derive(Debug, Clone, Default)]
pub struct ServerState {
    /// What we've pushed to this server for row-history sync:
    /// (row object, branch) -> set of known batch ids.
    pub sent_batch_ids: HashMap<(ObjectId, BranchName), SentBatchIds>,
    /// Row IDs for which we've sent metadata.
    pub sent_metadata: HashSet<ObjectId>,
    /// Whether this client confirms the rows it applies.
    ///
    /// Negotiated in the handshake, so it is known before the client subscribes. A client
    /// that does not confirm keeps the old behaviour — the claim is recorded when the row is
    /// queued — because otherwise nothing would ever clear its outstanding rows and every
    /// subscription would re-offer them.
    pub acks_deliveries: bool,
}

/// A query's scope and session for policy filtering.
#[derive(Debug, Clone, Default)]
pub struct QueryScope {
    /// The scope of objects/branches this query covers.
    pub scope: HashSet<(ObjectId, BranchName)>,
    /// The session to use for policy filtering (captured at registration time).
    pub session: Option<Session>,
}

/// Tracking state for a connected client.
#[derive(Debug, Clone, Default)]
pub struct ClientState {
    /// Client's role for access control.
    pub role: ClientRole,
    /// Client's session for policy evaluation.
    pub session: Option<Session>,
    /// Active queries from this client.
    pub queries: HashMap<QueryId, QueryScope>,
    /// What we've sent to this client for row-history sync:
    /// (row object, branch) -> set of known batch ids.
    pub sent_batch_ids: HashMap<(ObjectId, BranchName), SentBatchIds>,
    /// Row IDs for which we've sent metadata.
    pub sent_metadata: HashSet<ObjectId>,
    /// Batches of rows that entered this client's scope with a settled fate, each with the
    /// highest settlement target that fate meets (a rejection meets every target). The scope
    /// prune reads this instead of loading the fate again when the rows leave, and takes the
    /// entry out. Any later fate sent to the client for the same batch takes the entry out too:
    /// the stored fate has moved since, and the prune reads it again.
    ///
    /// A cache, never the only record: a missing entry costs one fate load. So an entry the
    /// prune could no longer reach goes as soon as it is unreachable: a row queued to the client
    /// takes out the batches it supersedes, the same ones `SentBatchIds::record_delivery` drops
    /// from the delivered frontier, and the batch of an unconfirmed offer it replaces (on the
    /// durable stores, which load a row without its parents, that is the exit). Past
    /// [`MAX_SETTLED_FATES_SENT`] the whole map is dropped.
    ///
    /// [`MAX_SETTLED_FATES_SENT`]: super::MAX_SETTLED_FATES_SENT
    pub settled_fates_sent: HashMap<BatchId, DurabilityTier>,
    /// Whether this client confirms the rows it applies.
    ///
    /// Negotiated in the handshake, so it is known before the client subscribes. A client
    /// that does not confirm keeps the old behaviour — the claim is recorded when the row is
    /// queued — because otherwise nothing would ever clear its outstanding rows and every
    /// subscription registration would re-offer them.
    pub acks_deliveries: bool,
}

impl ClientState {
    /// Create a new ClientState with an optional session.
    pub fn with_session(session: Option<Session>) -> Self {
        Self {
            session,
            ..Default::default()
        }
    }

    /// Check if an object/branch is in any of this client's query scopes.
    pub fn is_in_scope(&self, object_id: ObjectId, branch_name: &BranchName) -> bool {
        self.queries
            .values()
            .any(|q| q.scope.contains(&(object_id, *branch_name)))
    }
}

// ============================================================================
// Errors
// ============================================================================

/// Strongly typed errors for sync operations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SyncError {
    /// Operation denied due to insufficient permission.
    PermissionDenied {
        object_id: ObjectId,
        branch_name: BranchName,
        code: String,
        reason: String,
    },
    /// Client must have a session to write.
    SessionRequired {
        object_id: ObjectId,
        branch_name: BranchName,
    },
    /// This client role cannot write catalogue objects.
    CatalogueWriteDenied {
        object_id: ObjectId,
        branch_name: BranchName,
    },
    /// Query subscription was rejected (e.g. query compilation failed).
    QuerySubscriptionRejected {
        query_id: QueryId,
        code: String,
        reason: String,
    },
}

// ============================================================================
// Message Protocol
// ============================================================================

/// Row metadata sent once per destination.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RowMetadata {
    pub id: ObjectId,
    pub metadata: HashMap<String, String>,
}

/// Payload for sync messages between peers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SyncPayload {
    /// Semantic update for one catalogue/system entry.
    CatalogueEntryUpdated { entry: CatalogueEntry },

    /// Upstream replication of a newly created or newly learned row batch entry.
    RowBatchCreated {
        metadata: Option<RowMetadata>,
        row: StoredRowBatch,
    },

    /// Downstream delivery of a row batch entry that is needed for a subscriber's scope.
    RowBatchNeeded {
        metadata: Option<RowMetadata>,
        row: StoredRowBatch,
    },

    /// Replayable fate for one logical batch.
    BatchFate { fate: BatchFate },

    /// Request current replayable fate for specific batch ids.
    BatchFateNeeded { batch_ids: Vec<BatchId> },

    /// Explicitly seal a transactional batch so the authority can validate it.
    SealBatch { submission: SealedBatchSubmission },

    /// Subscribe to a query (client to server).
    /// Server will build QueryGraph and send matching objects.
    QuerySubscription {
        query_id: QueryId,
        query: Box<Query>,
        #[serde(with = "query_subscription_session_serde")]
        session: Option<Session>,
        #[serde(default)]
        required_tier: Option<DurabilityTier>,
        #[serde(default)]
        propagation: QueryPropagation,
        #[serde(default)]
        policy_context_tables: Vec<String>,
    },

    /// Unsubscribe from a query (client to server).
    QueryUnsubscription { query_id: QueryId },

    /// Query frontier settlement notification with the authoritative query scope
    /// for the settled server result.
    ///
    /// This means the upstream server has reached a complete first frontier for the
    /// subscription. Per-batch durability and visibility are replayed via `BatchFate`.
    QuerySettled {
        query_id: QueryId,
        tier: DurabilityTier,
        scope: Vec<(ObjectId, BranchName)>,
        /// Highest stream sequence known to be emitted before this notification.
        through_seq: u64,
    },

    /// Warning that rows exist on an older schema branch but are currently unreachable.
    SchemaWarning(SchemaWarning),

    /// Connection-time schema diagnostics for observability.
    ConnectionSchemaDiagnostics(ConnectionSchemaDiagnostics),

    /// Error response.
    Error(SyncError),

    /// Rows the receiver has applied, so the sender may finally record them as delivered.
    ///
    /// Appended LAST on purpose: postcard encodes a variant by index, so inserting anywhere
    /// else would renumber every payload on the wire. Only sent by a client whose handshake
    /// said it acks and whose server said it understands acks.
    DeliveryConfirmed { rows: Vec<ConfirmedRow> },
}

/// Warning emitted when a query encounters rows that cannot be transformed into the
/// subscriber's target schema because no reviewed migration path exists yet.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SchemaWarning {
    pub query_id: QueryId,
    pub table_name: String,
    pub row_count: usize,
    pub from_hash: SchemaHash,
    pub to_hash: SchemaHash,
}

/// Warning sent to the client when its schema is either disconnected from the permissions schema
/// or not connected to other schemas known to the server.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConnectionSchemaDiagnostics {
    pub client_schema_hash: SchemaHash,
    pub disconnected_permissions_schema_hash: Option<SchemaHash>,
    pub unreachable_schema_hashes: Vec<SchemaHash>,
}

impl ConnectionSchemaDiagnostics {
    pub fn has_issues(&self) -> bool {
        self.disconnected_permissions_schema_hash.is_some()
            || !self.unreachable_schema_hashes.is_empty()
    }
}

/// Sessions contain claims as a JSON object.
/// postcard does not support the dynamic deserialization style it expects (deserialize_any)
/// so we need a custom serializer/deserializer to serialize/deserialize the claims as a string.
mod query_subscription_session_serde {
    use crate::query_manager::session::{AuthMode, Session};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    struct SessionWire {
        user_id: String,
        claims_json: String,
        auth_mode: AuthMode,
    }

    pub fn serialize<S>(value: &Option<Session>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if serializer.is_human_readable() {
            return value.serialize(serializer);
        }

        let wire: Option<SessionWire> = value
            .as_ref()
            .map(|session| {
                let claims_json =
                    serde_json::to_string(&session.claims).map_err(serde::ser::Error::custom)?;
                Ok(SessionWire {
                    user_id: session.user_id.clone(),
                    claims_json,
                    auth_mode: session.auth_mode,
                })
            })
            .transpose()?;

        wire.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Session>, D::Error>
    where
        D: Deserializer<'de>,
    {
        if deserializer.is_human_readable() {
            return Option::<Session>::deserialize(deserializer);
        }

        let wire = Option::<SessionWire>::deserialize(deserializer)?;
        wire.map(|session_wire| {
            let claims = serde_json::from_str(&session_wire.claims_json)
                .map_err(serde::de::Error::custom)?;
            Ok(Session {
                user_id: session_wire.user_id,
                claims,
                auth_mode: session_wire.auth_mode,
            })
        })
        .transpose()
    }
}

impl SyncPayload {
    pub fn object_id(&self) -> Option<ObjectId> {
        match self {
            SyncPayload::CatalogueEntryUpdated { entry } => Some(entry.object_id),
            SyncPayload::RowBatchCreated { row, .. } | SyncPayload::RowBatchNeeded { row, .. } => {
                Some(row.row_id)
            }
            SyncPayload::BatchFate { .. } => None,
            SyncPayload::BatchFateNeeded { .. } => None,
            SyncPayload::SealBatch { submission } => {
                submission.members.first().map(|member| member.object_id)
            }
            SyncPayload::QuerySettled { scope, .. } => {
                scope.first().map(|(object_id, _)| *object_id)
            }
            _ => None,
        }
    }

    pub fn branch_name(&self) -> Option<BranchName> {
        match self {
            SyncPayload::CatalogueEntryUpdated { .. } => None,
            SyncPayload::RowBatchCreated { row, .. } | SyncPayload::RowBatchNeeded { row, .. } => {
                Some(BranchName::new(&row.branch))
            }
            SyncPayload::BatchFate { .. } => None,
            SyncPayload::BatchFateNeeded { .. } => None,
            SyncPayload::SealBatch { .. } => None,
            SyncPayload::QuerySettled { scope, .. } => {
                scope.first().map(|(_, branch_name)| *branch_name)
            }
            _ => None,
        }
    }

    /// True when handling this payload may mutate local storage.
    pub fn writes_storage(&self) -> bool {
        matches!(
            self,
            SyncPayload::CatalogueEntryUpdated { .. }
                | SyncPayload::RowBatchCreated { .. }
                | SyncPayload::RowBatchNeeded { .. }
                | SyncPayload::BatchFate { .. }
                | SyncPayload::SealBatch { .. }
        )
        // `DeliveryConfirmed` deliberately absent: it mutates only in-memory delivery
        // bookkeeping, and marking it as a storage write would schedule a WAL flush for
        // every ack tick.
    }

    /// Encode this payload using postcard.
    pub fn to_bytes(&self) -> Result<Vec<u8>, postcard::Error> {
        postcard::to_allocvec(self)
    }

    /// Decode a payload from postcard bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, postcard::Error> {
        postcard::from_bytes(bytes)
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Check if this payload carries a catalogue object (schema or lens).
    pub fn is_catalogue(&self) -> bool {
        match self {
            SyncPayload::CatalogueEntryUpdated { entry } => entry.is_catalogue(),
            SyncPayload::RowBatchCreated { metadata, .. }
            | SyncPayload::RowBatchNeeded { metadata, .. } => metadata
                .as_ref()
                .and_then(|metadata| {
                    metadata
                        .metadata
                        .get(crate::metadata::MetadataKey::Type.as_str())
                })
                .is_some_and(|kind| crate::metadata::ObjectType::is_catalogue_type_str(kind)),
            _ => false,
        }
    }

    /// Check if this payload carries a structural schema catalogue object.
    pub fn is_structural_schema_catalogue(&self) -> bool {
        matches!(self, SyncPayload::CatalogueEntryUpdated { entry } if entry.is_structural_schema_catalogue())
    }

    /// Get the variant name for debugging.
    pub fn variant_name(&self) -> &'static str {
        match self {
            SyncPayload::CatalogueEntryUpdated { .. } => "CatalogueEntryUpdated",
            SyncPayload::RowBatchCreated { .. } => "RowBatchCreated",
            SyncPayload::RowBatchNeeded { .. } => "RowBatchNeeded",
            SyncPayload::BatchFate { .. } => "BatchFate",
            SyncPayload::BatchFateNeeded { .. } => "BatchFateNeeded",
            SyncPayload::SealBatch { .. } => "SealBatch",
            SyncPayload::QuerySubscription { .. } => "QuerySubscription",
            SyncPayload::QueryUnsubscription { .. } => "QueryUnsubscription",
            SyncPayload::QuerySettled { .. } => "QuerySettled",
            SyncPayload::DeliveryConfirmed { .. } => "DeliveryConfirmed",
            SyncPayload::SchemaWarning(_) => "SchemaWarning",
            SyncPayload::ConnectionSchemaDiagnostics(_) => "ConnectionSchemaDiagnostics",
            SyncPayload::Error(_) => "Error",
        }
    }
}

/// Either end of a peer relationship. `Source` and `Destination` are mirror
/// images, and both expose the same peer identity fields for telemetry.
trait PeerEnd {
    fn descriptor(&self) -> (&'static str, Uuid);
}

/// Destination for an outbox entry.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Destination {
    Server(ServerId),
    Client(ClientId),
}

impl PeerEnd for Destination {
    fn descriptor(&self) -> (&'static str, Uuid) {
        match self {
            Destination::Server(id) => ("server", id.0),
            Destination::Client(id) => ("client", id.0),
        }
    }
}

impl Destination {
    pub fn peer_kind(&self) -> &'static str {
        PeerEnd::descriptor(self).0
    }

    pub fn peer_uuid(&self) -> Uuid {
        PeerEnd::descriptor(self).1
    }
}

/// Source of an inbox entry.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Source {
    Server(ServerId),
    Client(ClientId),
}

impl PeerEnd for Source {
    fn descriptor(&self) -> (&'static str, Uuid) {
        match self {
            Source::Server(id) => ("server", id.0),
            Source::Client(id) => ("client", id.0),
        }
    }
}

impl Source {
    pub fn peer_kind(&self) -> &'static str {
        PeerEnd::descriptor(self).0
    }

    pub fn peer_uuid(&self) -> Uuid {
        PeerEnd::descriptor(self).1
    }
}

/// What a queued row still owes its client's bookkeeping until the receiver confirms it.
///
/// Held aside rather than applied at enqueue: the payload can be dropped after queueing,
/// and a claim recorded for a payload nobody received is permanent — it short-circuits
/// every later attempt to offer the row.
#[derive(Debug, Clone)]
pub struct PendingDelivery {
    /// How many times this row has been offered without a confirmation coming back.
    pub attempts: u32,
    /// When the first of those offers was made, in the sync clock's microseconds.
    ///
    /// Attempts alone cannot justify giving up: an app registers its subscriptions in
    /// waves at startup, and every wave re-offers before the previous offer's
    /// confirmation could possibly have completed its round trip. Measured in the field:
    /// six waves inside one round trip burned a count-only cap and permanently dropped a
    /// message. Giving up requires attempts over the cap AND enough elapsed time for many
    /// round trips.
    pub first_offered_at: u64,
    /// Past the attempt cap and the grace, the row stops FORCING scope re-derivations —
    /// but stays owed. Giving it up entirely would record a delivery that never happened:
    /// the exact lie this bookkeeping exists to remove, acceptable for a heartbeat that
    /// the next beat supersedes, and unacceptable for a message nothing will ever rewrite.
    /// A demoted row still rides along whenever a re-derivation happens for other reasons,
    /// a late confirmation still clears it, and a newer batch resets it.
    pub demoted: bool,
    /// The batch this entry is about. Kept in the value, not the key: a later batch for the
    /// same row REPLACES this entry, because a re-offer can only ever ship the row's
    /// current state. Keying by batch would strand every superseded entry — nothing would
    /// send that batch again, so nothing could confirm it, and the peer would count as owed
    /// rows forever.
    pub batch_id: BatchId,
    pub branch_name: BranchName,
    /// The row's metadata, kept so a re-offer is identical to the original attempt rather
    /// than a reconstruction.
    pub metadata: HashMap<String, String>,
    /// Parents as they were BEFORE scope stripping — `scope_delivery_row` clears them on
    /// the delivered copy, but the frontier cursor prunes by them.
    pub parent_ids: Vec<BatchId>,
    pub include_metadata: bool,
}

/// One row the receiver has applied.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfirmedRow {
    pub row_id: ObjectId,
    pub branch: String,
    pub batch_id: BatchId,
}

/// Outgoing message to be sent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutboxEntry {
    pub destination: Destination,
    pub payload: SyncPayload,
}

/// Incoming message to be processed.
#[derive(Debug, Clone)]
pub struct InboxEntry {
    pub source: Source,
    pub payload: SyncPayload,
}

/// A pending query subscription that needs QueryGraph building.
#[derive(Debug, Clone)]
pub struct PendingQuerySubscription {
    pub client_id: ClientId,
    pub query_id: QueryId,
    pub query: Query,
    pub session: Option<Session>,
    pub required_tier: Option<DurabilityTier>,
    pub propagation: QueryPropagation,
    pub policy_context_tables: Vec<String>,
    /// The client's connection this was sent on (`SyncManager::client_connection`).
    pub connection: u64,
}

/// A pending query unsubscription that needs cleanup.
#[derive(Debug, Clone)]
pub struct PendingQueryUnsubscription {
    pub client_id: ClientId,
    pub query_id: QueryId,
    /// The withdrawal cancelled this client's subscription while it was still queued.
    pub cancelled_queued_subscription: bool,
}

/// A write from a User client awaiting permission check (policy evaluation).
///
/// Row-level policy evaluation which may require async graph settling.
#[derive(Debug, Clone)]
pub struct PendingPermissionCheck {
    pub id: PendingUpdateId,
    pub client_id: ClientId,
    pub payload: SyncPayload,
    pub session: Session,
    /// When schema resolution started deferring this check.
    pub schema_wait_started_at: Option<Instant>,
    /// Object metadata for policy evaluation.
    pub metadata: HashMap<String, String>,
    /// Old content for UPDATE/DELETE (None for INSERT).
    pub old_content: Option<Vec<u8>>,
    /// The schema the old content was AUTHORED under, from the row's locator.
    ///
    /// Bytes without their shape are the trap this codebase keeps paying for:
    /// every consumer has to re-derive the shape, and the only key at hand is
    /// the incoming write's branch — which names the WRITER's schema, not the
    /// row's. A row authored before a schema deployment then gets decoded with
    /// the wrong descriptor and its policy comparisons read garbage
    /// ("Update denied by USING policy … cannot see old row" for a legitimate
    /// owner). Carrying the hash with the bytes fixes the class.
    pub old_content_schema_hash: Option<crate::query_manager::types::branch::SchemaHash>,
    /// New content for INSERT/UPDATE (None for DELETE).
    pub new_content: Option<Vec<u8>>,
    /// Inferred operation type.
    pub operation: Operation,
    /// Where the old row comes from; see [`OldRowSource`].
    pub old_row: OldRowSource,
}

/// Where a pending check's old row comes from.
///
/// The inbox reads the old row from the incoming batch's own branch: its parents, or the
/// visible version they descend from. A write that reaches no visible ancestor there, on a
/// row with history on ANOTHER branch, is the first write a client makes after a schema
/// deployment — a copy-on-write onto the new generation's branch, with no parents. Its
/// old row is the row readers serve across the generations' branches: one head per
/// branch, each in its own generation's shape, the reader's winner among them. Only the
/// query layer knows which branches readers see and how each generation's bytes are
/// shaped, so the inbox defers it.
///
/// The inbox used to merge every branch's tips as one frontier, decoded with ONE
/// descriptor. Tips of an older shape failed to decode, the error read as "no old row",
/// and the update was judged an insert: local stack 2026-10-18, every presence beat on
/// `users` refused with `Insert denied by policy` for a week after a deployment.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum OldRowSource {
    /// `old_content` is the inbox's answer, read from the incoming batch's branch.
    #[default]
    Inline,
    /// No visible ancestor on the incoming branch, and history on another branch: the
    /// query layer resolves the old row at check time, through the reader.
    AcrossBranches,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query_manager::session::AuthMode;

    /// Random inserts and removals of row batch interest, and clients going away, against
    /// a plain map: what is held by batch is what walking the map answers. Windows that mostly
    /// add alternate with windows that mostly take away, so batches empty and fill again all
    /// through a seed.
    #[test]
    fn row_batch_interest_by_batch_matches_a_walk_of_every_row_batch() {
        struct Rng(u64);
        impl Rng {
            fn below(&mut self, bound: u64) -> u64 {
                self.0 ^= self.0 << 13;
                self.0 ^= self.0 >> 7;
                self.0 ^= self.0 << 17;
                self.0 % bound
            }
        }
        let by_row = |key: &RowBatchKey| {
            (
                *key.row_id.uuid().as_bytes(),
                key.branch_name.as_str().to_string(),
            )
        };
        let mut emptied_by_a_removal = 0;
        let mut emptied_by_a_client_leaving = 0;
        for seed in 1..=8u64 {
            let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
            let clients: Vec<ClientId> = (0..4).map(|_| ClientId::new()).collect();
            let batches: Vec<BatchId> = (0..6).map(|_| BatchId::new()).collect();
            let rows: Vec<ObjectId> = (0..8).map(|_| ObjectId::new()).collect();
            let branches = [BranchName::new("main"), BranchName::new("draft")];
            let mut interest = RowBatchInterest::default();
            let mut model: HashMap<RowBatchKey, HashSet<ClientId>> = HashMap::new();
            for step in 0..3000 {
                let key = RowBatchKey::new(
                    rows[rng.below(8) as usize],
                    branches[rng.below(2) as usize],
                    batches[rng.below(6) as usize],
                );
                let client = clients[rng.below(4) as usize];
                let adding = (step / 250) % 2 == 0;
                let inserts_below = if adding { 5 } else { 1 };
                let held_before: HashSet<BatchId> = model.keys().map(|key| key.batch_id).collect();
                let mut client_left = false;
                match rng.below(10) {
                    roll if roll < inserts_below => {
                        interest.insert(key, client);
                        model.entry(key).or_default().insert(client);
                    }
                    roll if roll < 8 => {
                        // Taking away, a removal hits a held pair, picked in a fixed order.
                        let (key, client) = if adding || model.is_empty() {
                            (key, client)
                        } else {
                            let mut held: Vec<RowBatchKey> = model.keys().copied().collect();
                            held.sort_by_key(|key| (by_row(key), key.batch_id.0));
                            let key = held[rng.below(held.len() as u64) as usize];
                            let holders: Vec<ClientId> = clients
                                .iter()
                                .copied()
                                .filter(|client| model[&key].contains(client))
                                .collect();
                            (key, holders[rng.below(holders.len() as u64) as usize])
                        };
                        interest.remove(&key, client);
                        if let Some(held) = model.get_mut(&key) {
                            held.remove(&client);
                            if held.is_empty() {
                                model.remove(&key);
                            }
                        }
                    }
                    8 => {
                        if rng.below(4) == 0 {
                            client_left = true;
                            interest.remove_client(client);
                            model.retain(|_, held| {
                                held.remove(&client);
                                !held.is_empty()
                            });
                        }
                    }
                    _ => {}
                }
                let walked: HashMap<RowBatchKey, HashSet<ClientId>> = interest
                    .iter()
                    .map(|(key, held)| (key, held.clone()))
                    .collect();
                assert_eq!(walked, model, "seed {seed}, step {step}: the row batches");
                let held: HashSet<BatchId> = interest.by_batch.keys().copied().collect();
                let want: HashSet<BatchId> = model.keys().map(|key| key.batch_id).collect();
                assert_eq!(
                    held, want,
                    "seed {seed}, step {step}: a batch is held exactly while a row of it is"
                );
                assert_eq!(interest.batch_count(), want.len());
                for rows in interest.by_batch.values() {
                    assert!(
                        matches!(rows, BatchRows::One(_, clients) if !clients.is_empty())
                            || matches!(rows, BatchRows::Many(many)
                                if many.len() > 1 && many.values().all(|clients| !clients.is_empty())),
                        "seed {seed}, step {step}: a batch of one row is held inline, and nothing held is empty"
                    );
                }
                let emptied = held_before.difference(&want).count();
                if client_left {
                    emptied_by_a_client_leaving += emptied;
                } else {
                    emptied_by_a_removal += emptied;
                }
                for batch_id in &batches {
                    let mut keys: Vec<RowBatchKey> = interest.keys_of_batch(*batch_id).collect();
                    let mut want: Vec<RowBatchKey> = model
                        .keys()
                        .filter(|key| key.batch_id == *batch_id)
                        .copied()
                        .collect();
                    keys.sort_by_key(by_row);
                    want.sort_by_key(by_row);
                    assert_eq!(keys, want, "seed {seed}, step {step}: the rows of a batch");
                    let got: HashSet<ClientId> = interest.clients_of_batch(*batch_id).collect();
                    let want: HashSet<ClientId> = model
                        .iter()
                        .filter(|(key, _)| key.batch_id == *batch_id)
                        .flat_map(|(_, held)| held.iter().copied())
                        .collect();
                    assert_eq!(
                        got, want,
                        "seed {seed}, step {step}: the clients of a batch"
                    );
                    for client in &clients {
                        assert_eq!(
                            interest.batch_has_client(*batch_id, *client),
                            want.contains(client),
                            "seed {seed}, step {step}: whether a client holds a row of a batch"
                        );
                    }
                }
            }
        }
        assert!(
            emptied_by_a_removal > 100 && emptied_by_a_client_leaving > 5,
            "batches must empty all through the seeds, by a removal ({emptied_by_a_removal}) \
             and by a client leaving ({emptied_by_a_client_leaving}), else this gates neither"
        );
    }

    /// A batch of one row is counted as its entry and its clients — what the flat map it
    /// replaced counted for one row batch — and a second row brings the rows' own entries.
    #[test]
    fn a_batch_of_one_row_is_counted_as_one_entry() {
        let mut interest = RowBatchInterest::default();
        let batch_id = BatchId::new();
        let client_id = ClientId::new();
        interest.insert(
            RowBatchKey::new(ObjectId::new(), BranchName::new("main"), batch_id),
            client_id,
        );
        let entry = std::mem::size_of::<BatchId>() + std::mem::size_of::<BatchRows>();
        let client = std::mem::size_of::<ClientId>();
        assert_eq!(interest.memory_size(), entry + client);
        assert_eq!(
            interest.memory_size(),
            std::mem::size_of::<RowBatchKey>() + client + std::mem::size_of::<HashSet<ClientId>>(),
            "the flat map's count of one row batch"
        );
        interest.insert(
            RowBatchKey::new(ObjectId::new(), BranchName::new("main"), batch_id),
            client_id,
        );
        let row = std::mem::size_of::<(ObjectId, BranchName)>()
            + std::mem::size_of::<HashSet<ClientId>>();
        assert_eq!(interest.memory_size(), entry + 2 * (row + client));
    }

    #[test]
    fn destination_exposes_peer_identity_for_telemetry() {
        let server_id = ServerId::new();
        let client_id = ClientId::new();

        let server = Destination::Server(server_id);
        let client = Destination::Client(client_id);

        assert_eq!(server.peer_kind(), "server");
        assert_eq!(server.peer_uuid(), server_id.0);
        assert_eq!(client.peer_kind(), "client");
        assert_eq!(client.peer_uuid(), client_id.0);
    }

    #[test]
    fn source_exposes_peer_identity_for_telemetry() {
        let server_id = ServerId::new();
        let client_id = ClientId::new();

        let server = Source::Server(server_id);
        let client = Source::Client(client_id);

        assert_eq!(server.peer_kind(), "server");
        assert_eq!(server.peer_uuid(), server_id.0);
        assert_eq!(client.peer_kind(), "client");
        assert_eq!(client.peer_uuid(), client_id.0);
    }

    #[test]
    fn query_subscription_postcard_roundtrip_preserves_session_auth_mode() {
        let payload = SyncPayload::QuerySubscription {
            query_id: QueryId(7),
            query: Box::new(Query::new("todos")),
            session: Some(Session::new("alice").with_auth_mode(AuthMode::LocalFirst)),
            required_tier: None,
            propagation: QueryPropagation::Full,
            policy_context_tables: Vec::new(),
        };

        let bytes = payload.to_bytes().expect("encode payload");
        let decoded = SyncPayload::from_bytes(&bytes).expect("decode payload");

        match decoded {
            SyncPayload::QuerySubscription {
                session: Some(session),
                ..
            } => assert_eq!(session.auth_mode, AuthMode::LocalFirst),
            other => panic!("expected QuerySubscription with session, got {other:?}"),
        }
    }
}
