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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
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

    /// Decode a payload from postcard bytes. Nesting is bounded at
    /// `wire_depth::WIRE_MAX_NESTING`: the wire is untrusted, the recursive types are
    /// derived deserializers, and an overflowed worker stack aborts the process.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, postcard::Error> {
        super::wire_depth::from_postcard_bounded(bytes, super::wire_depth::WIRE_MAX_NESTING)
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
}

/// A pending query unsubscription that needs cleanup.
#[derive(Debug, Clone)]
pub struct PendingQueryUnsubscription {
    pub client_id: ClientId,
    pub query_id: QueryId,
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query_manager::session::AuthMode;

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
