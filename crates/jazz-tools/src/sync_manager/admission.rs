//! Admission control for client subscriptions (linsa-v18, item 2 / defect #31).
//!
//! Every registration a client sends is checked here, in the inbox, BEFORE it is queued for
//! the query manager: a refused subscription costs the server one predicate over the payload
//! and no compile, no settle, no state. The caps are structural (shape of the query) and
//! per-principal (how many standing registrations, how many registrations per window).
//!
//! Counting is exact and owned here: `admitted` is inserted at admission and removed at every
//! point a server subscription ends (unsubscription, compile failure, recompile failure,
//! client removal). `ClientState.queries` is NOT a mirror of live subscriptions — it is
//! written after the first settle and cleared only on the recompile-failure path — so it is
//! not used.

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use web_time::Instant;

use super::types::{ClientId, ClientRole, QueryId};
use super::wire_depth::WIRE_MAX_NESTING;
use crate::query_manager::query::{ArraySubquerySpec, Query};
use crate::query_manager::relation_ir::{
    PredicateExpr, RELATION_GATHER_MAX_DEPTH_HARD_CAP, RelExpr,
};
use crate::query_manager::session::Session;

/// The caps, read once by the server binary from the environment (`JAZZ_MAX_*`), or set
/// through the builders by tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubscriptionCaps {
    /// Nesting levels of `ArraySubquerySpec` below the root (a query without includes is 0).
    pub max_include_depth: usize,
    /// Total `ArraySubquerySpec` count across the tree.
    pub max_include_nodes: usize,
    /// Largest explicit `limit`: every `Limit` node of the relation IR (the root limit lives
    /// only there on the wire) and every nested spec's `limit`.
    pub max_query_limit: usize,
    /// Nodes in the relation IR the planner will consume.
    pub max_ir_nodes: usize,
    /// `Join` nodes in the IR plus `joins` on every nested spec.
    pub max_ir_joins: usize,
    /// Inputs of the widest `Union` node.
    pub max_union_arity: usize,
    /// `max_depth` of the deepest `Gather` node (the engine's hard cap is 64 and nothing
    /// applied it to wire-supplied IR before this).
    pub max_gather_depth: usize,
    /// `branches` named by the query (each one is probed per row by the loader).
    pub max_branches: usize,
    /// Predicate terms the planner evaluates per row: comparison leaves of every `Filter`
    /// (each value of an `In` counts), plus every include spec's `filters`, `order_by` and
    /// `select_columns` entries, which are replayed into a fresh builder per subgraph
    /// instantiation. Without it a three-node IR can carry a million-term `Or`.
    pub max_predicate_leaves: usize,
    /// Encoded size of the registration (query, tier, propagation, policy context tables
    /// and the session's claims) the identity hash is allowed to consume. Nothing else
    /// bounds bytes: a 60 MiB text literal in one comparison is one predicate leaf and three
    /// IR nodes, so it passes every structural cap, would be re-serialized under the engine
    /// lock on every registration and compared against every row at settle.
    pub max_query_bytes: usize,
    /// Standing registrations per user (all of that user's client ids together).
    pub max_subscriptions_per_user: usize,
    /// Standing registrations per backend client id.
    pub max_subscriptions_per_backend: usize,
    /// Standing registrations on this node, every principal together: the ceiling identity
    /// minting (self-signed local-first sessions) cannot escape.
    pub max_total_subscriptions: usize,
    /// NEW registrations per user per `registration_window` (a replay of an id this node
    /// already holds is free), refused ones included.
    pub max_registrations_per_user_per_window: u32,
    pub registration_window: Duration,
}

impl Default for SubscriptionCaps {
    fn default() -> Self {
        Self {
            max_include_depth: 6,
            max_include_nodes: 32,
            max_query_limit: 1_000,
            max_ir_nodes: 128,
            max_ir_joins: 8,
            max_union_arity: 16,
            max_gather_depth: RELATION_GATHER_MAX_DEPTH_HARD_CAP,
            max_branches: 2,
            max_predicate_leaves: 256,
            max_query_bytes: 1 << 20,
            max_subscriptions_per_user: 256,
            max_subscriptions_per_backend: 4_096,
            max_total_subscriptions: 10_000,
            max_registrations_per_user_per_window: 120,
            registration_window: Duration::from_secs(10),
        }
    }
}

impl SubscriptionCaps {
    /// Defaults overridden by `JAZZ_MAX_INCLUDE_DEPTH`, `JAZZ_MAX_INCLUDE_NODES`,
    /// `JAZZ_MAX_QUERY_LIMIT`, `JAZZ_MAX_IR_NODES`, `JAZZ_MAX_IR_JOINS`,
    /// `JAZZ_MAX_UNION_ARITY`, `JAZZ_MAX_GATHER_DEPTH`, `JAZZ_MAX_QUERY_BRANCHES`,
    /// `JAZZ_MAX_PREDICATE_LEAVES`, `JAZZ_MAX_QUERY_BYTES`, `JAZZ_MAX_SUBSCRIPTIONS_PER_USER`,
    /// `JAZZ_MAX_SUBSCRIPTIONS_PER_BACKEND`, `JAZZ_MAX_TOTAL_SUBSCRIPTIONS`,
    /// `JAZZ_MAX_REGISTRATIONS_PER_USER_PER_10S`. An unparsable value and a `0` both keep
    /// the default, with a warning. The nesting bound (`wire_depth::WIRE_MAX_NESTING`) is
    /// not a cap: it is a constant.
    pub fn from_env() -> Self {
        /// An unparsable value keeps the default; so does `0`, which would refuse every
        /// registration (a cap of zero is a misconfiguration, not a policy). Both warn.
        fn read<T: std::str::FromStr + PartialEq + Default + Copy>(name: &str, default: T) -> T {
            let Ok(raw) = std::env::var(name) else {
                return default;
            };
            match raw.trim().parse::<T>() {
                Ok(value) if value == T::default() => {
                    tracing::warn!(env = name, "cap of 0 ignored; keeping the default");
                    default
                }
                Ok(value) => value,
                Err(_) => {
                    tracing::warn!(
                        env = name,
                        value = %raw,
                        "unparsable cap ignored; keeping the default"
                    );
                    default
                }
            }
        }
        let defaults = Self::default();
        Self {
            max_include_depth: read("JAZZ_MAX_INCLUDE_DEPTH", defaults.max_include_depth),
            max_include_nodes: read("JAZZ_MAX_INCLUDE_NODES", defaults.max_include_nodes),
            max_query_limit: read("JAZZ_MAX_QUERY_LIMIT", defaults.max_query_limit),
            max_ir_nodes: read("JAZZ_MAX_IR_NODES", defaults.max_ir_nodes),
            max_ir_joins: read("JAZZ_MAX_IR_JOINS", defaults.max_ir_joins),
            max_union_arity: read("JAZZ_MAX_UNION_ARITY", defaults.max_union_arity),
            max_gather_depth: read("JAZZ_MAX_GATHER_DEPTH", defaults.max_gather_depth)
                .min(RELATION_GATHER_MAX_DEPTH_HARD_CAP),
            max_branches: read("JAZZ_MAX_QUERY_BRANCHES", defaults.max_branches),
            max_predicate_leaves: read("JAZZ_MAX_PREDICATE_LEAVES", defaults.max_predicate_leaves),
            max_query_bytes: read("JAZZ_MAX_QUERY_BYTES", defaults.max_query_bytes),
            max_subscriptions_per_user: read(
                "JAZZ_MAX_SUBSCRIPTIONS_PER_USER",
                defaults.max_subscriptions_per_user,
            ),
            max_subscriptions_per_backend: read(
                "JAZZ_MAX_SUBSCRIPTIONS_PER_BACKEND",
                defaults.max_subscriptions_per_backend,
            ),
            max_total_subscriptions: read(
                "JAZZ_MAX_TOTAL_SUBSCRIPTIONS",
                defaults.max_total_subscriptions,
            ),
            max_registrations_per_user_per_window: read(
                "JAZZ_MAX_REGISTRATIONS_PER_USER_PER_10S",
                defaults.max_registrations_per_user_per_window,
            ),
            registration_window: defaults.registration_window,
        }
    }

    /// No cap bites. For nodes that are not servers, and for tests of other things.
    pub fn unlimited() -> Self {
        Self {
            max_include_depth: usize::MAX,
            max_include_nodes: usize::MAX,
            max_query_limit: usize::MAX,
            max_ir_nodes: usize::MAX,
            max_ir_joins: usize::MAX,
            max_union_arity: usize::MAX,
            max_gather_depth: usize::MAX,
            max_branches: usize::MAX,
            max_predicate_leaves: usize::MAX,
            max_query_bytes: usize::MAX,
            max_subscriptions_per_user: usize::MAX,
            max_subscriptions_per_backend: usize::MAX,
            max_total_subscriptions: usize::MAX,
            max_registrations_per_user_per_window: u32::MAX,
            registration_window: Duration::from_secs(10),
        }
    }
}

/// Who a registration is charged to. `client_id` is client-supplied in the handshake and
/// never bound to the session, so a user's caps are keyed by the server-established user
/// id and shared by every client id that user mints.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Principal {
    User(String),
    Backend(ClientId),
    /// A user-role client without any session: caps per client id, nothing better exists.
    Anonymous(ClientId),
}

impl Principal {
    pub fn for_client(role: ClientRole, client_id: ClientId, session: Option<&Session>) -> Self {
        match role {
            ClientRole::User => match session {
                Some(session) => Principal::User(session.user_id.clone()),
                None => Principal::Anonymous(client_id),
            },
            ClientRole::Backend | ClientRole::Admin | ClientRole::Peer => {
                Principal::Backend(client_id)
            }
        }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Principal::User(_) => "user",
            Principal::Backend(_) => "backend",
            Principal::Anonymous(_) => "anonymous",
        }
    }
}

/// Why a registration was refused. `Display` is the wire `reason`; it starts with the
/// rejection code because the one-shot failure path on the client formats only the reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    /// Nesting past `wire_depth::WIRE_MAX_NESTING` in the relation IR, the predicate tree or
    /// the include tree. The decode already refuses such a frame; this is the walk's own
    /// guard for values that did not come through it.
    Nesting {
        value: usize,
        cap: usize,
    },
    IncludeDepth {
        value: usize,
        cap: usize,
    },
    IncludeNodes {
        value: usize,
        cap: usize,
    },
    QueryLimit {
        value: usize,
        cap: usize,
    },
    IrNodes {
        value: usize,
        cap: usize,
    },
    IrJoins {
        value: usize,
        cap: usize,
    },
    UnionArity {
        value: usize,
        cap: usize,
    },
    GatherDepth {
        value: usize,
        cap: usize,
    },
    Branches {
        value: usize,
        cap: usize,
    },
    PredicateLeaves {
        value: usize,
        cap: usize,
    },
    /// The encoded registration is larger than `max_query_bytes`; the size is not reported
    /// because the hash stops at the cap instead of measuring the frame.
    QueryBytes {
        cap: usize,
    },
    Subscriptions {
        value: usize,
        cap: usize,
        principal: &'static str,
    },
    TotalSubscriptions {
        value: usize,
        cap: usize,
    },
    RegistrationRate {
        value: u32,
        cap: u32,
        window: Duration,
    },
}

/// The rejection code every refusal carries.
pub const SUBSCRIPTION_OVER_CAP: &str = "subscription_over_cap";

impl Refusal {
    pub fn cap_name(&self) -> &'static str {
        match self {
            Refusal::Nesting { .. } => "nesting",
            Refusal::IncludeDepth { .. } => "include_depth",
            Refusal::IncludeNodes { .. } => "include_nodes",
            Refusal::QueryLimit { .. } => "query_limit",
            Refusal::IrNodes { .. } => "ir_nodes",
            Refusal::IrJoins { .. } => "ir_joins",
            Refusal::UnionArity { .. } => "union_arity",
            Refusal::GatherDepth { .. } => "gather_depth",
            Refusal::Branches { .. } => "branches",
            Refusal::PredicateLeaves { .. } => "predicate_leaves",
            Refusal::QueryBytes { .. } => "query_bytes",
            Refusal::Subscriptions { .. } => "subscriptions",
            Refusal::TotalSubscriptions { .. } => "total_subscriptions",
            Refusal::RegistrationRate { .. } => "registration_rate",
        }
    }
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{SUBSCRIPTION_OVER_CAP}: ")?;
        match self {
            Refusal::Nesting { value, cap } => write!(f, "nesting {value} > {cap}"),
            Refusal::IncludeDepth { value, cap } => write!(f, "include depth {value} > {cap}"),
            Refusal::IncludeNodes { value, cap } => write!(f, "include nodes {value} > {cap}"),
            Refusal::QueryLimit { value, cap } => write!(f, "limit {value} > {cap}"),
            Refusal::IrNodes { value, cap } => write!(f, "relation nodes {value} > {cap}"),
            Refusal::IrJoins { value, cap } => write!(f, "joins {value} > {cap}"),
            Refusal::UnionArity { value, cap } => write!(f, "union arity {value} > {cap}"),
            Refusal::GatherDepth { value, cap } => write!(f, "gather depth {value} > {cap}"),
            Refusal::Branches { value, cap } => write!(f, "branches {value} > {cap}"),
            Refusal::PredicateLeaves { value, cap } => {
                write!(f, "predicate leaves {value} > {cap}")
            }
            Refusal::QueryBytes { cap } => write!(f, "encoded registration > {cap} bytes"),
            Refusal::TotalSubscriptions { value, cap } => {
                write!(f, "total standing subscriptions {value} > {cap}")
            }
            Refusal::Subscriptions {
                value,
                cap,
                principal,
            } => write!(
                f,
                "standing subscriptions for this {principal} {value} > {cap}"
            ),
            Refusal::RegistrationRate { value, cap, window } => {
                write!(f, "registrations {value} > {cap} in {}s", window.as_secs())
            }
        }
    }
}

/// Shape of a query as the planner will consume it: the relation IR as shipped (on the wire
/// the root filters, order, offset and limit live only there) plus the include tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct QueryShape {
    /// Deepest nesting level the walk reached (IR, predicate and include trees together).
    pub nesting: usize,
    pub include_depth: usize,
    pub include_nodes: usize,
    pub max_limit: usize,
    pub ir_nodes: usize,
    pub ir_joins: usize,
    pub union_arity: usize,
    pub gather_depth: usize,
    pub branches: usize,
    pub predicate_leaves: usize,
}

/// The walks stop past the wire bound: past it the shape is refused whatever else it holds,
/// and a walk that recursed on regardless would be the stack overflow the bound exists for.
fn past_the_bound(depth: usize, shape: &mut QueryShape) -> bool {
    shape.nesting = shape.nesting.max(depth);
    depth > WIRE_MAX_NESTING
}

fn walk_predicate(predicate: &PredicateExpr, shape: &mut QueryShape, depth: usize) {
    if past_the_bound(depth, shape) {
        return;
    }
    match predicate {
        PredicateExpr::Cmp { .. }
        | PredicateExpr::Contains { .. }
        | PredicateExpr::IsNull { .. }
        | PredicateExpr::IsNotNull { .. } => shape.predicate_leaves += 1,
        PredicateExpr::In { values, .. } => shape.predicate_leaves += values.len().max(1),
        PredicateExpr::And(items) | PredicateExpr::Or(items) => {
            for item in items {
                walk_predicate(item, shape, depth + 1);
            }
        }
        PredicateExpr::Not(inner) => walk_predicate(inner, shape, depth + 1),
        PredicateExpr::True | PredicateExpr::False => {}
    }
}

fn walk_ir(expr: &RelExpr, shape: &mut QueryShape, depth: usize) {
    if past_the_bound(depth, shape) {
        return;
    }
    shape.ir_nodes += 1;
    match expr {
        RelExpr::TableScan { .. } => {}
        RelExpr::Filter { input, predicate } => {
            walk_predicate(predicate, shape, depth + 1);
            walk_ir(input, shape, depth + 1);
        }
        RelExpr::Project { input, .. }
        | RelExpr::Distinct { input, .. }
        | RelExpr::OrderBy { input, .. }
        | RelExpr::Offset { input, .. } => walk_ir(input, shape, depth + 1),
        RelExpr::Limit { input, limit } => {
            shape.max_limit = shape.max_limit.max(*limit);
            walk_ir(input, shape, depth + 1);
        }
        RelExpr::Union { inputs } => {
            shape.union_arity = shape.union_arity.max(inputs.len());
            for input in inputs {
                walk_ir(input, shape, depth + 1);
            }
        }
        RelExpr::Join { left, right, .. } => {
            shape.ir_joins += 1;
            walk_ir(left, shape, depth + 1);
            walk_ir(right, shape, depth + 1);
        }
        RelExpr::Gather {
            seed,
            step,
            max_depth,
            ..
        } => {
            shape.gather_depth = shape.gather_depth.max(*max_depth);
            walk_ir(seed, shape, depth + 1);
            walk_ir(step, shape, depth + 1);
        }
    }
}

fn walk_specs(specs: &[ArraySubquerySpec], level: usize, shape: &mut QueryShape) {
    if past_the_bound(level, shape) {
        return;
    }
    for spec in specs {
        shape.include_nodes += 1;
        shape.include_depth = shape.include_depth.max(level);
        shape.ir_joins += spec.joins.len();
        shape.predicate_leaves += spec.filters.len()
            + spec.order_by.len()
            + spec
                .select_columns
                .as_ref()
                .map_or(0, |columns| columns.len());
        if let Some(limit) = spec.limit {
            shape.max_limit = shape.max_limit.max(limit);
        }
        walk_specs(&spec.nested_arrays, level + 1, shape);
    }
}

pub fn query_shape(query: &Query) -> QueryShape {
    let mut shape = QueryShape {
        max_limit: query.limit.unwrap_or(0),
        branches: query.branches.len(),
        ..QueryShape::default()
    };
    walk_ir(&query.relation_ir, &mut shape, 1);
    walk_specs(&query.array_subqueries, 1, &mut shape);
    shape
}

/// Pure predicate over the query's shape.
pub fn check_shape(caps: &SubscriptionCaps, query: &Query) -> Result<QueryShape, Refusal> {
    let shape = query_shape(query);
    let over = |value: usize, cap: usize| value > cap;
    if over(shape.nesting, WIRE_MAX_NESTING) {
        return Err(Refusal::Nesting {
            value: shape.nesting,
            cap: WIRE_MAX_NESTING,
        });
    }
    if over(shape.include_depth, caps.max_include_depth) {
        return Err(Refusal::IncludeDepth {
            value: shape.include_depth,
            cap: caps.max_include_depth,
        });
    }
    if over(shape.include_nodes, caps.max_include_nodes) {
        return Err(Refusal::IncludeNodes {
            value: shape.include_nodes,
            cap: caps.max_include_nodes,
        });
    }
    if over(shape.max_limit, caps.max_query_limit) {
        return Err(Refusal::QueryLimit {
            value: shape.max_limit,
            cap: caps.max_query_limit,
        });
    }
    if over(shape.ir_nodes, caps.max_ir_nodes) {
        return Err(Refusal::IrNodes {
            value: shape.ir_nodes,
            cap: caps.max_ir_nodes,
        });
    }
    if over(shape.ir_joins, caps.max_ir_joins) {
        return Err(Refusal::IrJoins {
            value: shape.ir_joins,
            cap: caps.max_ir_joins,
        });
    }
    if over(shape.union_arity, caps.max_union_arity) {
        return Err(Refusal::UnionArity {
            value: shape.union_arity,
            cap: caps.max_union_arity,
        });
    }
    if over(shape.gather_depth, caps.max_gather_depth) {
        return Err(Refusal::GatherDepth {
            value: shape.gather_depth,
            cap: caps.max_gather_depth,
        });
    }
    if over(shape.branches, caps.max_branches) {
        return Err(Refusal::Branches {
            value: shape.branches,
            cap: caps.max_branches,
        });
    }
    if over(shape.predicate_leaves, caps.max_predicate_leaves) {
        return Err(Refusal::PredicateLeaves {
            value: shape.predicate_leaves,
            cap: caps.max_predicate_leaves,
        });
    }
    Ok(shape)
}

/// Per-principal state. Lives on the `SyncManager` of a node that serves clients.
#[derive(Debug, Clone)]
pub struct Admission {
    caps: SubscriptionCaps,
    admitted: HashMap<Principal, HashSet<(ClientId, QueryId)>>,
    /// Principal of every admitted (client, query) and the fingerprint of the query it holds,
    /// so release needs no session lookup and a replay can be told from a changed query.
    owner: HashMap<(ClientId, QueryId), (Principal, u64)>,
    registrations: HashMap<Principal, (Instant, u32)>,
    /// Expired windows are swept at most once per window, so a flood of minted identities
    /// costs one sweep per window, not one per identity.
    last_prune: Option<Instant>,
}

impl Default for Admission {
    fn default() -> Self {
        Self::new(SubscriptionCaps::default())
    }
}

/// One registration as the server compares it before taking the equivalent fast path
/// (`server_queries.rs`, `existing_subscription_state`: query, session, required tier,
/// propagation, policy context tables). The admission's replay exemption is keyed on the
/// hash of the same five, so it can never be wider than the server's: whatever makes the
/// server re-derive makes the admission charge.
#[derive(Clone, Copy)]
pub struct Registration<'a> {
    pub query: &'a Query,
    /// The EFFECTIVE session — the client session merged with the payload's claims — which
    /// is what the server stores and compares; a payload-only claim survives that merge.
    pub session: Option<&'a Session>,
    pub required_tier: Option<super::DurabilityTier>,
    pub propagation: super::QueryPropagation,
    pub policy_context_tables: &'a [String],
}

impl Registration<'_> {
    /// The hash of the wire form of everything the server compares, or `QueryBytes` once
    /// `max_bytes` of it have been fed to the hasher. The bytes are never materialized, so
    /// the cost of a refused registration is bounded by the cap, not by the frame.
    /// One value the server compares and the encoding drops: `Value::Row { id }` is
    /// compared by `PartialEq` but the binary form carries the row's fields alone. No client
    /// frame can carry it — the only frame decode is postcard through the wire bound — so a
    /// JSON transport, if one is ever added, has to revisit this.
    fn identity(&self, max_bytes: usize) -> Result<u64, Refusal> {
        let over = || Refusal::QueryBytes { cap: max_bytes };
        let mut hasher = HashingFlavor::new(max_bytes);
        hasher = postcard::serialize_with_flavor(self.query, hasher).map_err(|_| over())?;
        hasher =
            postcard::serialize_with_flavor(&self.required_tier, hasher).map_err(|_| over())?;
        hasher = postcard::serialize_with_flavor(&self.propagation, hasher).map_err(|_| over())?;
        hasher = postcard::serialize_with_flavor(&self.policy_context_tables, hasher)
            .map_err(|_| over())?;
        if let Some(session) = self.session {
            hasher =
                postcard::serialize_with_flavor(&(&session.user_id, session.auth_mode), hasher)
                    .map_err(|_| over())?;
            // Claims are JSON on the wire and `serde_json::Value` in the session; postcard
            // has no encoding for a JSON object, JSON itself does. `Value::Object` is a
            // `BTreeMap` in every build of this crate (`preserve_order` is enabled nowhere),
            // so the writer emits keys sorted and two claim sets hash alike exactly when the
            // server's `Session ==` calls them equal — no narrower, no wider.
            serde_json::to_writer(&mut hasher, &session.claims).map_err(|_| over())?;
        }
        Ok(hasher.finish())
    }
}

/// A postcard flavor that hashes what it is fed instead of storing it, and refuses past a
/// byte budget BEFORE copying: a text or bytes literal reaches `try_extend` as one slice and
/// is refused at once; a byte array or a value list arrives one element at a time, so the
/// work is bounded by the cap, never by the frame.
struct HashingFlavor {
    hasher: std::collections::hash_map::DefaultHasher,
    written: usize,
    budget: usize,
}

impl HashingFlavor {
    fn new(budget: usize) -> Self {
        Self {
            hasher: std::collections::hash_map::DefaultHasher::new(),
            written: 0,
            budget,
        }
    }

    fn feed(&mut self, data: &[u8]) -> bool {
        if self.written.saturating_add(data.len()) > self.budget {
            return false;
        }
        self.written += data.len();
        std::hash::Hasher::write(&mut self.hasher, data);
        true
    }

    fn finish(&self) -> u64 {
        std::hash::Hasher::finish(&self.hasher)
    }
}

impl postcard::ser_flavors::Flavor for HashingFlavor {
    type Output = Self;

    fn try_push(&mut self, data: u8) -> postcard::Result<()> {
        self.try_extend(&[data])
    }

    fn try_extend(&mut self, data: &[u8]) -> postcard::Result<()> {
        if self.feed(data) {
            Ok(())
        } else {
            Err(postcard::Error::SerializeBufferFull)
        }
    }

    fn finalize(self) -> postcard::Result<Self> {
        Ok(self)
    }
}

impl std::io::Write for HashingFlavor {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if self.feed(buf) {
            Ok(buf.len())
        } else {
            Err(std::io::Error::other("registration over the byte budget"))
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Admission {
    pub fn new(caps: SubscriptionCaps) -> Self {
        Self {
            caps,
            admitted: HashMap::new(),
            owner: HashMap::new(),
            registrations: HashMap::new(),
            last_prune: None,
        }
    }

    pub fn caps(&self) -> &SubscriptionCaps {
        &self.caps
    }

    pub fn set_caps(&mut self, caps: SubscriptionCaps) {
        self.caps = caps;
    }

    /// Standing registrations charged to `principal`.
    pub fn admitted_count(&self, principal: &Principal) -> usize {
        self.admitted.get(principal).map_or(0, HashSet::len)
    }

    /// Whether this node already holds an admitted registration for `(client, query)`.
    pub fn holds(&self, client_id: ClientId, query_id: QueryId) -> bool {
        self.owner.contains_key(&(client_id, query_id))
    }

    /// Standing registrations on this node, every principal together.
    pub fn total_admitted(&self) -> usize {
        self.owner.len()
    }

    /// Decide on one registration. On `Ok` the registration is recorded as admitted; on
    /// `Err` nothing is recorded except the attempt itself against the rate window.
    pub fn admit(
        &mut self,
        principal: Principal,
        client_id: ClientId,
        query_id: QueryId,
        registration: Registration<'_>,
        now: Instant,
    ) -> Result<QueryShape, Refusal> {
        let key = (client_id, query_id);
        let held_by_same = self
            .owner
            .get(&key)
            .is_some_and(|(owner, _)| *owner == principal);

        // The structural walk first: it stops at the first cap it finds over, so it is the
        // cheapest thing that can be said about the frame, and nothing below touches the
        // bytes before it passed. An over-cap shape cannot be a replay of a held
        // registration (that one passed the same caps), so it is charged as a new attempt.
        let shape = match check_shape(&self.caps, registration.query) {
            Ok(shape) => shape,
            Err(refusal) => {
                self.charge(&principal, now);
                return Err(refusal);
            }
        };
        // Then the identity, serialized once and bounded: the hash stops at
        // `max_query_bytes` without materializing anything, so a 64 MiB frame costs the
        // engine lock nothing beyond the walk.
        let identity = match registration.identity(self.caps.max_query_bytes) {
            Ok(identity) => identity,
            Err(refusal) => {
                self.charge(&principal, now);
                return Err(refusal);
            }
        };
        let already = self
            .owner
            .get(&key)
            .is_some_and(|(owner, held)| *owner == principal && *held == identity);

        // Every NEW attempt counts, refused ones included: a client that is refused a hundred
        // times a second is exactly the client the window exists for. A replay of an id this
        // node already holds is free — every successful handshake replays every standing
        // subscription, and a link that flaps would otherwise lose them all. A replay is the
        // SAME registration, in every field the server compares before it takes the
        // equivalent path (`Registration`): a held id re-registered with any of them changed
        // is a fresh derivation at the server (compile + first settle) and is charged like
        // one, or one id flipped between two filters, two claim sets or two tiers
        // re-derives at line speed for free.
        if !already
            && let Some(count) = self.charge(&principal, now)
            && count > self.caps.max_registrations_per_user_per_window
        {
            return Err(Refusal::RegistrationRate {
                value: count,
                cap: self.caps.max_registrations_per_user_per_window,
                window: self.caps.registration_window,
            });
        }

        if !already {
            // A key that moves to a new principal (session change on the same client id)
            // does not grow the total; charge it as a move, not as a new registration.
            let moving = usize::from(self.owner.contains_key(&key));
            // The ceiling bounds what free identities can fill (a self-signed keypair costs
            // nothing, so per-user caps alone are evadable). The backend is the trusted path
            // with its own cap: it must keep working when the ceiling is full, otherwise
            // filling the ceiling is a way to take the product down.
            let total = self.owner.len() - moving;
            if !matches!(principal, Principal::Backend(_))
                && total >= self.caps.max_total_subscriptions
            {
                return Err(Refusal::TotalSubscriptions {
                    value: total + 1,
                    cap: self.caps.max_total_subscriptions,
                });
            }
            let cap = match principal {
                Principal::User(_) | Principal::Anonymous(_) => {
                    self.caps.max_subscriptions_per_user
                }
                Principal::Backend(_) => self.caps.max_subscriptions_per_backend,
            };
            // A changed registration on an id this principal already holds replaces that
            // registration; it is not a second standing subscription.
            let standing = self.admitted_count(&principal) - usize::from(held_by_same);
            if standing >= cap {
                return Err(Refusal::Subscriptions {
                    value: standing + 1,
                    cap,
                    principal: principal.kind(),
                });
            }
            // A re-registration that changed principal (session change on the same client
            // id) moves the charge.
            self.release(client_id, query_id);
            self.admitted
                .entry(principal.clone())
                .or_default()
                .insert(key);
            self.owner.insert(key, (principal, identity));
        }
        Ok(shape)
    }

    /// Charge one attempt to `principal`'s window. The count after charging for a user;
    /// `None` for the principals the window does not apply to (the backend is trusted, an
    /// anonymous keypair is bounded by its standing count instead).
    fn charge(&mut self, principal: &Principal, now: Instant) -> Option<u32> {
        if !matches!(principal, Principal::User(_)) {
            return None;
        }
        let window = self.caps.registration_window;
        let due = self
            .last_prune
            .is_none_or(|last| now.duration_since(last) >= window);
        if self.registrations.len() > 256 && due {
            self.registrations
                .retain(|_, (start, _)| now.duration_since(*start) < window);
            self.last_prune = Some(now);
        }
        let entry = self
            .registrations
            .entry(principal.clone())
            .or_insert((now, 0));
        if now.duration_since(entry.0) >= window {
            *entry = (now, 0);
        }
        entry.1 = entry.1.saturating_add(1);
        Some(entry.1)
    }

    /// The registration ended (unsubscribed, failed to compile, dropped on recompile).
    pub fn release(&mut self, client_id: ClientId, query_id: QueryId) {
        let key = (client_id, query_id);
        if let Some((principal, _)) = self.owner.remove(&key)
            && let Some(set) = self.admitted.get_mut(&principal)
        {
            set.remove(&key);
            if set.is_empty() {
                self.admitted.remove(&principal);
            }
        }
    }

    /// The client is gone: every registration it held ends.
    pub fn release_client(&mut self, client_id: ClientId) {
        let keys: Vec<(ClientId, QueryId)> = self
            .owner
            .keys()
            .filter(|(id, _)| *id == client_id)
            .copied()
            .collect();
        for (client_id, query_id) in keys {
            self.release(client_id, query_id);
        }
    }
}
