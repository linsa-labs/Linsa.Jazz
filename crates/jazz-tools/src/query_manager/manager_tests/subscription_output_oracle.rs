//! Randomized differential oracle for subscription OUTPUT streams (v13-1,
//! include-plan-sharing design §6.1).
//!
//! Two independent `QueryManager` universes replay the SAME seeded mutation
//! stream (inserts / updates / soft deletes on parent + child + nested-child
//! tables, `is_deleted` filter flips, correlate-value moves) while serving
//! include-bearing subscriptions: a 2-deep nested include
//! (`parents -> children -> grandchildren`), `order_by` + `limit` on the
//! outer query, and session-scoped SELECT policies ON (schema policies force
//! `RowPolicyMode::Enforcing`). After EVERY mutation batch the full
//! observable output — the initial snapshot and every subsequent delta batch
//! — must be identical between the two engines, and must reconstruct both the
//! engine's own materialized state and an independent generator-side model.
//!
//! Each side runs a declared [`EnginePath`] — `Legacy`
//! (`JAZZ_PRECISE_DIRTY=0`, the pre-F2 mark-all path) or `PreciseDirty` (the
//! default) — forced per engine call through `force_precise_dirty`, the same
//! plug shape as `force_history_fastpath` in
//! `storage/conformance_differential.rs`. The default oracle runs the FULL
//! op matrix precise-vs-precise (the model layer carries the proof); the
//! legacy-vs-precise dual-run is confined to [`OpProfile::LegacyParity`],
//! the subset on which the legacy path is actually correct.
//!
//! What is checked per batch, per subscription:
//! - **A/B equality**: every `QueryUpdate` (RowDelta rows byte-for-byte,
//!   ordered delta indices, descriptor) matches between the engines;
//! - **stream completeness**: applying the emitted deltas to a client-side
//!   mirror reproduces the engine's internal `current_ordered_ids` /
//!   `current_visible_rows` exactly — a missed delta (F2's under-marking
//!   class) desynchronizes the mirror and fails here;
//! - **pre-image fidelity**: `removed`/`updated` rows must carry exactly the
//!   bytes the client currently holds;
//! - **model equality**: the decoded (parent, children, grandchildren) id
//!   tree in mirror order equals the generator-side reference model —
//!   including per-session policy visibility;
//! - **initial-snapshot completeness / timeliness** (v13-0 census finding:
//!   7/16 subs served empty results for minutes on cold open): a cold-open
//!   subscription must emit its snapshot within ONE `process()` call, the
//!   snapshot must be non-empty whenever seeded data matches, and it must
//!   equal the model immediately;
//! - **bounded convergence**: after a mutation batch the engines are pumped
//!   in lockstep (every pass compared A-vs-B) and must go quiescent within
//!   `SETTLE_PASS_BOUND` passes — include arrays legitimately settle one
//!   pass behind their inner-table writes, but unbounded re-settling is the
//!   spin failure mode.
//!
//! Delta-batch normalization (deliberate, minimal — exactly two legitimate
//! nondeterminisms are pinned, nothing else is normalized):
//! 1. the interleaving of updates across DIFFERENT subscriptions inside one
//!    `process()` outbox (HashMap iteration order in the settle loop) —
//!    updates are grouped per subscription before comparison;
//! 2. row batch-id VALUES: every direct write mints its version identity
//!    from `Uuid::now_v7()` at write time (`BatchId::new`), so two engines
//!    replaying identical writes hold different 16-byte version tags.
//!    Cross-engine row comparison is therefore modulo the batch-id value —
//!    but its CHANGE structure still must match exactly: which rows appear
//!    in `updated`, in which batches, is driven by batch-id transitions and
//!    is compared verbatim. (The `WriteContext.batch_id` override cannot pin
//!    the value: it means "staged transactional write" and flips rows to
//!    `StagingPending`.)
//! Everything else — batch boundaries, intra-batch vector order
//! (deterministic: deltas are diffed against ordered previous state), row
//! bytes, provenance (timestamps and authors are fixed via `WriteContext`
//! overrides), descriptors — is compared exactly. Both the outer query and
//! every include carry an `order_by` on a unique column, so engine output
//! order is fully specified; harness rows never tie.
//!
//! Generator limitations (conscious, documented):
//! - No hard deletes and no restores: those write paths mint provenance
//!   timestamps from the wall clock (no `WriteContext` override), which would
//!   make the two universes byte-diverge for reasons that are not bugs.
//!   §6.2 fixtures cover them as pinned scenarios instead.
//! - Single branch, no sync tiers: this oracle targets the local
//!   subscription settle path; cross-tier delivery is exercised by the e2e
//!   suites.
//!
//! FINDING (v13-1, caught by this oracle's model layer on its first run;
//! FIXED by F2 in v13-2 — the fix points are noted inline): content-only
//! updates to an include's INNER rows never reached subscription outputs —
//! the include array stayed permanently stale, on the local-write path AND
//! the remote sync-inbox path, through any number of settle passes. Root
//! cause was exactly the scope cut §9/F2 names: row-precise changed ids were
//! discarded at `mark_table_dependents_dirty` (`graph/execute.rs` — no ids
//! parameter), so a reused subquery instance re-ran its correlate scan
//! (membership refreshed) but its materialize state never re-loaded content
//! for ids it already held; `process_with_context` likewise reused the old
//! array whenever the correlation value was unchanged. The manifestations,
//! and where each is fixed:
//! - a child edit was invisible in every parent's include until the NEXT
//!   membership change of that include — fixed by forwarding
//!   `mark_rows_updated` into the include's subgraph instances
//!   (`ArraySubqueryNode::forward_rows_updated`);
//! - an `is_deleted` FILTER FLIP is a content update, so rows flipped in the
//!   model but not in the engine — same fix (the instance's FilterNode sees
//!   the reloaded content and retracts);
//! - NESTED-include membership was stale too: a grandchild insert/move under
//!   an unchanged child set never surfaced, because the nested subquery's
//!   instance caches inside a reused outer instance never re-ran their scans
//!   — fixed by `note_inner_rows_changed` recursing through the instance
//!   graphs' own dependent routing;
//! - mixing an OUTER-row update with an inner membership change in ONE
//!   settle corrupted the output tuple set (double-booked ordered index plus
//!   a stale `updated` pre-image): tuple identity is ID-based, so the two
//!   chained update pairs the old settle emitted for one row id
//!   double-booked downstream sort state — fixed by processing the outer
//!   delta FIRST and evaluating the fresh array inside the update when the
//!   instance carries pending inner dirt, so each settle emits exactly one
//!   coalesced pair per row (`graph/execute.rs` ArraySubquery arm +
//!   `process_with_context`'s freshness check).
//! The LEGACY path (`JAZZ_PRECISE_DIRTY=0`) retains the first three
//! manifestations by design; [`OpProfile::LegacyParity`] excludes exactly
//! those op classes so the kill-switch dual-run stays meaningful.

use std::collections::HashMap;

use uuid::Uuid;

use super::*;
use crate::query_manager::graph_nodes::output::QuerySubscriptionId;
use crate::query_manager::manager::QueryUpdate;
use crate::query_manager::session::WriteContext;
use crate::query_manager::types::{OrderedRowDelta, Row};

const PARENT_TABLE: &str = "parents";
const CHILD_TABLE: &str = "children";
const GRANDCHILD_TABLE: &str = "grandchildren";

const OWNERS: [&str; 2] = ["alice", "bob"];
const AUTHORS: [&str; 2] = ["writer-a", "writer-b"];

const PARENT_LIMIT: usize = 5;
const MUTATION_BATCHES: usize = 48;
const MAX_OPS_PER_BATCH: usize = 3;

const MAX_PARENTS: usize = 14;
const MAX_CHILDREN: usize = 36;
const MAX_GRANDCHILDREN: usize = 28;

/// A cold-open subscription must serve its first (complete) snapshot within
/// this many `process()` calls. The v13-0 census run broke exactly this:
/// subscriptions sat on empty results for minutes.
const SETTLE_PROCESS_BOUND: usize = 1;

/// After a mutation batch, both engines must emit all resulting deltas and
/// go quiescent within this many settle passes. One `process()` can leave an
/// include array a pass behind its inner-table write (the inner delta
/// re-dirties the outer graph), so convergence is multi-pass but must stay
/// tightly bounded — unbounded passes are the settle-spin failure mode.
const SETTLE_PASS_BOUND: usize = 4;

const SEEDS: [u64; 6] = [
    0x5AB5_C21B_0000_0001,
    0x5AB5_C21B_0000_0002,
    0x5AB5_C21B_0000_0003,
    0x5AB5_C21B_0000_0004,
    0x0D1F_F5AB_0000_0005,
    0x0D1F_F5AB_0000_0006,
];

// ============================================================================
// Engine-path plug point for the F2 kill-switch dual-runs.
// ============================================================================

/// Which evaluation path a differential side runs.
///
/// Each engine call (write or settle) is wrapped in a scoped
/// `force_precise_dirty` guard for the engine's path — the same plug shape
/// as `force_history_fastpath`, but held per call instead of per engine
/// lifetime so a legacy engine and a precise engine can interleave in one
/// dual-run without deadlocking on the override mutex.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EnginePath {
    /// The pre-F2 mark-all path (`JAZZ_PRECISE_DIRTY=0`). Keeps the known
    /// include-staleness bugs; byte-correct only for [`OpProfile::LegacyParity`].
    Legacy,
    /// Row-precise include dirtiness with correlation routing OFF
    /// (`JAZZ_INCLUDE_ROUTING=0`): every buffered mark reaches every cached
    /// instance, which is the v13-3 behaviour v14 L1 replaces. Correct but
    /// O(instances) — and therefore the right reference for proving that
    /// routing changes cost and nothing else.
    PreciseUnrouted,
    /// The default path with empty bindings OFF (`JAZZ_INCLUDE_EMPTY_PROBE=0`):
    /// every include binding gets a compiled instance, as before v24. The
    /// reference for proving that answering an empty binding from the index
    /// changes cost and nothing else.
    PreciseNoEmptyProbe,
    /// Row-precise include dirtiness with correlation routing on (the default
    /// runtime path since v14).
    PreciseDirty,
}

/// The kill switches, held for one engine call. Acquired in a fixed order
/// (precise, then routing, then empty bindings) so two engines on different
/// paths cannot deadlock against each other inside one dual-run.
struct EngagedPath {
    _precise: crate::query_manager::precise_dirty::PreciseDirtyMode,
    _routing: crate::query_manager::graph_nodes::include_routing::IncludeRoutingMode,
    _empty_probe: crate::query_manager::graph_nodes::include_routing::IncludeEmptyProbeMode,
    /// Every binding the default path answers from the index is also compiled and
    /// settled, and must yield nothing — on every seed, storage and op profile here.
    _empty_binding_parity: crate::query_manager::graph_nodes::include_routing::EmptyBindingParity,
}

impl EnginePath {
    fn engage(self) -> EngagedPath {
        EngagedPath {
            _precise: crate::query_manager::precise_dirty::force_precise_dirty(!matches!(
                self,
                EnginePath::Legacy
            )),
            _routing: crate::query_manager::graph_nodes::include_routing::force_include_routing(
                matches!(
                    self,
                    EnginePath::PreciseDirty | EnginePath::PreciseNoEmptyProbe
                ),
            ),
            _empty_probe:
                crate::query_manager::graph_nodes::include_routing::force_include_empty_probe(
                    !matches!(self, EnginePath::PreciseNoEmptyProbe),
                ),
            _empty_binding_parity:
                crate::query_manager::graph_nodes::include_routing::check_empty_bindings_against_instances(),
        }
    }
}

/// Which mutation classes the generator draws from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OpProfile {
    /// Everything except the op classes whose subscription-output effect the
    /// LEGACY path swallows by design (inner-row content updates: child
    /// title edits and `is_deleted` filter flips; nested-include membership:
    /// grandchild inserts/moves). On this profile the legacy and precise
    /// paths must produce byte-identical output streams — the kill-switch
    /// regression guard.
    LegacyParity,
    /// The full §6.2 op matrix including filter flips and grandchild
    /// inserts/moves — the default oracle profile since F2 (v13-2) threads
    /// row-precise changed ids into include instances (nested ones included).
    FullIncludingFilterFlips,
}

// ============================================================================
// Deterministic randomness (same shape as storage/conformance_differential).
// ============================================================================

/// xorshift64* — deterministic, dependency-free.
struct Prng(u64);

impl Prng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }

    fn chance(&mut self, numerator: usize, denominator: usize) -> bool {
        self.below(denominator) < numerator
    }
}

/// Deterministic id mint: both engines see identical row and batch ids.
struct IdMint {
    seed: u64,
    counter: u64,
}

impl IdMint {
    fn next_uuid(&mut self) -> Uuid {
        self.counter += 1;
        Uuid::from_u128((u128::from(self.seed) << 64) | u128::from(self.counter))
    }

    fn next_object_id(&mut self) -> ObjectId {
        ObjectId::from_uuid(self.next_uuid())
    }
}

// ============================================================================
// Schema and subscriptions.
// ============================================================================

fn owner_scoped_policies() -> TablePolicies {
    TablePolicies::new()
        .with_select(PolicyExpr::eq_session("owner_id", vec!["user_id".into()]))
        .with_insert(PolicyExpr::True)
        .with_update(None, PolicyExpr::True)
        .with_delete(PolicyExpr::True)
}

fn oracle_schema() -> Schema {
    let mut schema = Schema::new();
    schema.insert(
        TableName::new(PARENT_TABLE),
        TableSchema::with_policies(
            RowDescriptor::new(vec![
                ColumnDescriptor::new("title", ColumnType::Text),
                ColumnDescriptor::new("ord", ColumnType::Integer),
                ColumnDescriptor::new("owner_id", ColumnType::Text),
            ]),
            owner_scoped_policies(),
        ),
    );
    schema.insert(
        TableName::new(CHILD_TABLE),
        TableSchema::with_policies(
            RowDescriptor::new(vec![
                ColumnDescriptor::new("title", ColumnType::Text),
                ColumnDescriptor::new("ord", ColumnType::Integer),
                ColumnDescriptor::new("owner_id", ColumnType::Text),
                ColumnDescriptor::new("is_deleted", ColumnType::Boolean),
                ColumnDescriptor::new("parent_id", ColumnType::Uuid),
            ]),
            owner_scoped_policies(),
        ),
    );
    schema.insert(
        TableName::new(GRANDCHILD_TABLE),
        TableSchema::with_policies(
            RowDescriptor::new(vec![
                ColumnDescriptor::new("title", ColumnType::Text),
                ColumnDescriptor::new("ord", ColumnType::Integer),
                ColumnDescriptor::new("owner_id", ColumnType::Text),
                ColumnDescriptor::new("child_id", ColumnType::Uuid),
            ]),
            owner_scoped_policies(),
        ),
    );
    schema
}

/// Column index of the `children` include array in the combined parent
/// descriptor (base columns first, include columns appended).
const PARENT_INCLUDE_COLUMN: usize = 3;
/// Column index of the `grandchildren` include array inside a child row.
const CHILD_INCLUDE_COLUMN: usize = 5;

#[derive(Clone, Copy, Debug)]
struct SubSpec {
    user: &'static str,
    descending: bool,
    limit: Option<usize>,
}

fn sub_specs() -> Vec<SubSpec> {
    vec![
        // The census shape: newest-first window with the nested include.
        SubSpec {
            user: "alice",
            descending: true,
            limit: Some(PARENT_LIMIT),
        },
        // Same include SHAPE, different session — the F1 plan-key leak class.
        SubSpec {
            user: "bob",
            descending: true,
            limit: Some(PARENT_LIMIT),
        },
        // Unwindowed ascending variant.
        SubSpec {
            user: "alice",
            descending: false,
            limit: None,
        },
    ]
}

// ============================================================================
// Generator-side reference model.
// ============================================================================

#[derive(Clone, Debug)]
struct ParentModel {
    owner: &'static str,
    ord: i32,
    title: String,
    deleted: bool,
}

#[derive(Clone, Debug)]
struct ChildModel {
    owner: &'static str,
    ord: i32,
    title: String,
    flag_deleted: bool,
    parent: ObjectId,
    deleted: bool,
}

#[derive(Clone, Debug)]
struct GrandchildModel {
    owner: &'static str,
    ord: i32,
    title: String,
    child: ObjectId,
    deleted: bool,
}

#[derive(Default)]
struct Model {
    parents: Vec<(ObjectId, ParentModel)>,
    children: Vec<(ObjectId, ChildModel)>,
    grandchildren: Vec<(ObjectId, GrandchildModel)>,
}

/// (parent, [(child, [grandchild])]) in exact expected output order.
type ExpectedTree = Vec<(ObjectId, Vec<(ObjectId, Vec<ObjectId>)>)>;

impl Model {
    fn live_parents(&self) -> Vec<usize> {
        (0..self.parents.len())
            .filter(|index| !self.parents[*index].1.deleted)
            .collect()
    }

    fn live_children(&self) -> Vec<usize> {
        (0..self.children.len())
            .filter(|index| !self.children[*index].1.deleted)
            .collect()
    }

    fn live_grandchildren(&self) -> Vec<usize> {
        (0..self.grandchildren.len())
            .filter(|index| !self.grandchildren[*index].1.deleted)
            .collect()
    }

    fn expected_tree(&self, spec: &SubSpec) -> ExpectedTree {
        let mut parents: Vec<(ObjectId, &ParentModel)> = self
            .parents
            .iter()
            .filter(|(_, parent)| !parent.deleted && parent.owner == spec.user)
            .map(|(id, parent)| (*id, parent))
            .collect();
        parents.sort_by_key(|(_, parent)| parent.ord);
        if spec.descending {
            parents.reverse();
        }
        if let Some(limit) = spec.limit {
            parents.truncate(limit);
        }

        parents
            .into_iter()
            .map(|(parent_id, _)| {
                let mut children: Vec<(ObjectId, &ChildModel)> = self
                    .children
                    .iter()
                    .filter(|(_, child)| {
                        !child.deleted
                            && !child.flag_deleted
                            && child.parent == parent_id
                            && child.owner == spec.user
                    })
                    .map(|(id, child)| (*id, child))
                    .collect();
                children.sort_by_key(|(_, child)| child.ord);
                let children = children
                    .into_iter()
                    .map(|(child_id, _)| {
                        let mut grandchildren: Vec<(ObjectId, &GrandchildModel)> = self
                            .grandchildren
                            .iter()
                            .filter(|(_, grandchild)| {
                                !grandchild.deleted
                                    && grandchild.child == child_id
                                    && grandchild.owner == spec.user
                            })
                            .map(|(id, grandchild)| (*id, grandchild))
                            .collect();
                        grandchildren.sort_by_key(|(_, grandchild)| grandchild.ord);
                        (
                            child_id,
                            grandchildren.into_iter().map(|(id, _)| id).collect(),
                        )
                    })
                    .collect();
                (parent_id, children)
            })
            .collect()
    }
}

fn parent_values(parent: &ParentModel) -> Vec<Value> {
    vec![
        Value::Text(parent.title.clone()),
        Value::Integer(parent.ord),
        Value::Text(parent.owner.to_string()),
    ]
}

fn child_values(child: &ChildModel) -> Vec<Value> {
    vec![
        Value::Text(child.title.clone()),
        Value::Integer(child.ord),
        Value::Text(child.owner.to_string()),
        Value::Boolean(child.flag_deleted),
        Value::Uuid(child.parent),
    ]
}

fn grandchild_values(grandchild: &GrandchildModel) -> Vec<Value> {
    vec![
        Value::Text(grandchild.title.clone()),
        Value::Integer(grandchild.ord),
        Value::Text(grandchild.owner.to_string()),
        Value::Uuid(grandchild.child),
    ]
}

// ============================================================================
// Mutation stream.
// ============================================================================

/// One deterministic write, fully stamped: both engines replay it with
/// identical row id, provenance author, and timestamp (`WriteContext`
/// overrides), so their storage states match modulo engine-minted batch ids
/// (see the module docs on normalization).
struct StampedOp {
    kind: StampedOpKind,
    label: &'static str,
    ts: u64,
    author: &'static str,
}

enum StampedOpKind {
    Insert {
        table: &'static str,
        id: ObjectId,
        values: Vec<Value>,
    },
    Update {
        id: ObjectId,
        values: Vec<Value>,
    },
    SoftDelete {
        id: ObjectId,
    },
}

struct OpGenerator {
    prng: Prng,
    mint: IdMint,
    next_ts: u64,
    next_ord: i32,
    op_serial: u64,
    profile: OpProfile,
}

impl OpGenerator {
    fn new(seed: u64, profile: OpProfile) -> Self {
        Self {
            prng: Prng::new(seed),
            mint: IdMint { seed, counter: 0 },
            next_ts: 1_000_000,
            next_ord: 100,
            op_serial: 0,
            profile,
        }
    }

    fn stamp(&mut self, kind: StampedOpKind, label: &'static str) -> StampedOp {
        self.next_ts += 10;
        self.op_serial += 1;
        StampedOp {
            kind,
            label,
            ts: self.next_ts,
            author: AUTHORS[(self.op_serial % 2) as usize],
        }
    }

    fn next_ord(&mut self) -> i32 {
        self.next_ord += 1;
        self.next_ord
    }

    fn owner(&mut self) -> &'static str {
        OWNERS[self.prng.below(OWNERS.len())]
    }

    fn insert_parent(&mut self, model: &mut Model, owner: &'static str) -> StampedOp {
        let id = self.mint.next_object_id();
        let parent = ParentModel {
            owner,
            ord: self.next_ord(),
            title: format!("p{}", self.op_serial),
            deleted: false,
        };
        let values = parent_values(&parent);
        model.parents.push((id, parent));
        self.stamp(
            StampedOpKind::Insert {
                table: PARENT_TABLE,
                id,
                values,
            },
            "insert-parent",
        )
    }

    fn insert_child(&mut self, model: &mut Model, forced_parent: Option<ObjectId>) -> StampedOp {
        let id = self.mint.next_object_id();
        let parent = forced_parent.unwrap_or_else(|| {
            // 1-in-6 dangling correlate value: a UUID no parent row carries.
            if model.parents.is_empty() || self.prng.chance(1, 6) {
                self.mint.next_object_id()
            } else {
                model.parents[self.prng.below(model.parents.len())].0
            }
        });
        let owner = if self.prng.chance(4, 5) {
            model
                .parents
                .iter()
                .find(|(parent_id, _)| *parent_id == parent)
                .map(|(_, parent)| parent.owner)
                .unwrap_or_else(|| OWNERS[self.prng.below(OWNERS.len())])
        } else {
            // Cross-owner child: parent visible, child policy-hidden.
            self.owner()
        };
        let child = ChildModel {
            owner,
            ord: self.next_ord(),
            title: format!("c{}", self.op_serial),
            flag_deleted: false,
            parent,
            deleted: false,
        };
        let values = child_values(&child);
        model.children.push((id, child));
        self.stamp(
            StampedOpKind::Insert {
                table: CHILD_TABLE,
                id,
                values,
            },
            "insert-child",
        )
    }

    fn insert_grandchild(
        &mut self,
        model: &mut Model,
        forced_child: Option<ObjectId>,
    ) -> StampedOp {
        let id = self.mint.next_object_id();
        let child = forced_child.unwrap_or_else(|| {
            if model.children.is_empty() || self.prng.chance(1, 6) {
                self.mint.next_object_id()
            } else {
                model.children[self.prng.below(model.children.len())].0
            }
        });
        let owner = if self.prng.chance(4, 5) {
            model
                .children
                .iter()
                .find(|(child_id, _)| *child_id == child)
                .map(|(_, child)| child.owner)
                .unwrap_or_else(|| OWNERS[self.prng.below(OWNERS.len())])
        } else {
            self.owner()
        };
        let grandchild = GrandchildModel {
            owner,
            ord: self.next_ord(),
            title: format!("g{}", self.op_serial),
            child,
            deleted: false,
        };
        let values = grandchild_values(&grandchild);
        model.grandchildren.push((id, grandchild));
        self.stamp(
            StampedOpKind::Insert {
                table: GRANDCHILD_TABLE,
                id,
                values,
            },
            "insert-grandchild",
        )
    }

    /// One mutation batch (1..=`MAX_OPS_PER_BATCH` ops), freely mixing outer
    /// and inner op classes.
    ///
    /// (Historical note: mixing the classes in one settle used to corrupt
    /// the engine's output tuple set — FINDING manifestation 3 — so the
    /// pre-F2 green profile had to keep batches class-homogeneous. Fixed by
    /// the outer-first + per-row-coalesced settle in
    /// `graph/execute.rs` / `graph_nodes/array_subquery.rs`; both profiles
    /// now draw from the mixed stream.)
    fn batch_ops(&mut self, model: &mut Model) -> Vec<StampedOp> {
        let count = 1 + self.prng.below(MAX_OPS_PER_BATCH);
        (0..count).map(|_| self.mutation(model)).collect()
    }

    /// One random mutation from the profile's matrix; falls back to inserts
    /// when a target pool is empty or capped, so an op is always produced.
    fn mutation(&mut self, model: &mut Model) -> StampedOp {
        let roll = self.prng.below(100);
        let live_parents = model.live_parents();
        let live_children = model.live_children();
        let live_grandchildren = model.live_grandchildren();
        match roll {
            // insert parent
            0..8 => {
                if model.parents.len() < MAX_PARENTS {
                    let owner = self.owner();
                    self.insert_parent(model, owner)
                } else {
                    self.move_parent_ord(model, &live_parents)
                }
            }
            // insert child
            8..20 => {
                if model.children.len() < MAX_CHILDREN {
                    self.insert_child(model, None)
                } else {
                    self.move_child_parent(model, &live_children)
                }
            }
            // insert grandchild (FullIncludingFilterFlips only: nested-include
            // membership changes under an unchanged child set are swallowed by
            // the LEGACY path — the nested subquery's instance caches inside a
            // reused outer instance never re-run their scans)
            20..28 => match self.profile {
                OpProfile::FullIncludingFilterFlips
                    if model.grandchildren.len() < MAX_GRANDCHILDREN =>
                {
                    self.insert_grandchild(model, None)
                }
                OpProfile::FullIncludingFilterFlips => {
                    self.move_grandchild(model, &live_grandchildren)
                }
                OpProfile::LegacyParity => self.insert_child(model, None),
            },
            // parent content update
            28..36 => self.update_parent_title(model, &live_parents),
            // parent order move (window churn under order_by + limit)
            36..48 => self.move_parent_ord(model, &live_parents),
            // child content update (Full only: the LEGACY path never re-serves
            // include-inner content — FINDING manifestation 1)
            48..54 => match self.profile {
                OpProfile::FullIncludingFilterFlips => {
                    self.update_child_title(model, &live_children)
                }
                OpProfile::LegacyParity => self.move_child_parent(model, &live_children),
            },
            // include-filter flip (Full only: the flip is an inner content
            // update, swallowed by the LEGACY path — FINDING manifestation 1)
            54..66 => match self.profile {
                OpProfile::FullIncludingFilterFlips => self.flip_child_flag(model, &live_children),
                OpProfile::LegacyParity => self.move_child_parent(model, &live_children),
            },
            // correlate-value move: child re-homed to another parent
            66..78 => self.move_child_parent(model, &live_children),
            // correlate-value move on the nested include (Full only — see the
            // grandchild-insert arm above)
            78..85 => match self.profile {
                OpProfile::FullIncludingFilterFlips => {
                    self.move_grandchild(model, &live_grandchildren)
                }
                OpProfile::LegacyParity => self.move_child_parent(model, &live_children),
            },
            // soft deletes
            85..91 => self.soft_delete_parent(model, &live_parents),
            _ => self.soft_delete_child(model, &live_children),
        }
    }

    fn update_parent_title(&mut self, model: &mut Model, live: &[usize]) -> StampedOp {
        if live.is_empty() {
            let owner = self.owner();
            return self.insert_parent(model, owner);
        }
        let index = live[self.prng.below(live.len())];
        let (id, parent) = &mut model.parents[index];
        parent.title = format!("p{}", self.op_serial + 1);
        let (id, values) = (*id, parent_values(parent));
        self.stamp(StampedOpKind::Update { id, values }, "update-parent-title")
    }

    fn move_parent_ord(&mut self, model: &mut Model, live: &[usize]) -> StampedOp {
        if live.is_empty() {
            let owner = self.owner();
            return self.insert_parent(model, owner);
        }
        let index = live[self.prng.below(live.len())];
        let ord = self.next_ord();
        let (id, parent) = &mut model.parents[index];
        parent.ord = ord;
        let (id, values) = (*id, parent_values(parent));
        self.stamp(StampedOpKind::Update { id, values }, "move-parent-ord")
    }

    fn update_child_title(&mut self, model: &mut Model, live: &[usize]) -> StampedOp {
        if live.is_empty() {
            return self.insert_child(model, None);
        }
        let index = live[self.prng.below(live.len())];
        let (id, child) = &mut model.children[index];
        child.title = format!("c{}", self.op_serial + 1);
        let (id, values) = (*id, child_values(child));
        self.stamp(StampedOpKind::Update { id, values }, "update-child-title")
    }

    fn flip_child_flag(&mut self, model: &mut Model, live: &[usize]) -> StampedOp {
        if live.is_empty() {
            return self.insert_child(model, None);
        }
        let index = live[self.prng.below(live.len())];
        let (id, child) = &mut model.children[index];
        child.flag_deleted = !child.flag_deleted;
        let (id, values) = (*id, child_values(child));
        self.stamp(
            StampedOpKind::Update { id, values },
            "flip-child-is-deleted",
        )
    }

    fn move_child_parent(&mut self, model: &mut Model, live: &[usize]) -> StampedOp {
        if live.is_empty() || model.parents.is_empty() {
            return self.insert_child(model, None);
        }
        let index = live[self.prng.below(live.len())];
        let parent = if self.prng.chance(1, 8) {
            self.mint.next_object_id()
        } else {
            model.parents[self.prng.below(model.parents.len())].0
        };
        let (id, child) = &mut model.children[index];
        child.parent = parent;
        let (id, values) = (*id, child_values(child));
        self.stamp(StampedOpKind::Update { id, values }, "move-child-parent")
    }

    fn move_grandchild(&mut self, model: &mut Model, live: &[usize]) -> StampedOp {
        if live.is_empty() || model.children.is_empty() {
            return self.insert_grandchild(model, None);
        }
        let index = live[self.prng.below(live.len())];
        let child = if self.prng.chance(1, 8) {
            self.mint.next_object_id()
        } else {
            model.children[self.prng.below(model.children.len())].0
        };
        let (id, grandchild) = &mut model.grandchildren[index];
        grandchild.child = child;
        let (id, values) = (*id, grandchild_values(grandchild));
        self.stamp(
            StampedOpKind::Update { id, values },
            "move-grandchild-child",
        )
    }

    /// Soft-deleted rows stay deleted (no restore op — see module docs), so
    /// keep a floor of live parents/children for the other ops to target.
    fn soft_delete_parent(&mut self, model: &mut Model, live: &[usize]) -> StampedOp {
        if live.len() <= 3 {
            let owner = self.owner();
            return self.insert_parent(model, owner);
        }
        let index = live[self.prng.below(live.len())];
        let (id, parent) = &mut model.parents[index];
        parent.deleted = true;
        let id = *id;
        self.stamp(StampedOpKind::SoftDelete { id }, "soft-delete-parent")
    }

    fn soft_delete_child(&mut self, model: &mut Model, live: &[usize]) -> StampedOp {
        if live.len() <= 4 {
            return self.insert_child(model, None);
        }
        let index = live[self.prng.below(live.len())];
        let (id, child) = &mut model.children[index];
        child.deleted = true;
        let id = *id;
        self.stamp(StampedOpKind::SoftDelete { id }, "soft-delete-child")
    }

    /// Deterministic cold-open dataset: every subscription's snapshot is
    /// non-empty (both owners get parents with visible children and
    /// grandchildren), plus a few randomized extras.
    fn seed_ops(&mut self, model: &mut Model) -> Vec<StampedOp> {
        let mut ops = Vec::new();
        for round in 0..4 {
            let owner = OWNERS[round % 2];
            ops.push(self.insert_parent(model, owner));
            let parent_id = model.parents.last().expect("parent just inserted").0;
            for _ in 0..2 {
                ops.push(self.insert_child(model, Some(parent_id)));
                let child_id = model.children.last().expect("child just inserted").0;
                ops.push(self.insert_grandchild(model, Some(child_id)));
            }
        }
        for _ in 0..6 {
            let op = self.mutation(model);
            ops.push(op);
        }
        ops
    }
}

// ============================================================================
// Engine wrapper + client-side mirror.
// ============================================================================

#[derive(Default)]
struct Mirror {
    descriptor: Option<RowDescriptor>,
    ordered: Vec<ObjectId>,
    rows: HashMap<ObjectId, Row>,
}

impl Mirror {
    /// Apply one emitted update exactly as a client would, with pre-image
    /// checks: `removed`/`updated` must carry the bytes the client holds.
    fn apply_update(&mut self, context: &str, update: &QueryUpdate) {
        match &self.descriptor {
            None => self.descriptor = Some(update.descriptor.clone()),
            Some(descriptor) => assert!(
                *descriptor == update.descriptor,
                "{context}: subscription-oracle divergence: descriptor changed mid-stream"
            ),
        }

        for row in &update.delta.removed {
            let held = self.rows.remove(&row.id).unwrap_or_else(|| {
                panic!(
                    "{context}: subscription-oracle divergence: removed row {} was never delivered",
                    row.id
                )
            });
            assert!(
                held == *row,
                "{context}: subscription-oracle divergence: removed row {} pre-image differs from \
                 client-held state\nheld: {held:?}\nsent: {row:?}",
                row.id
            );
        }
        for (old, new) in &update.delta.updated {
            let held = self.rows.get(&old.id).unwrap_or_else(|| {
                panic!(
                    "{context}: subscription-oracle divergence: updated row {} was never delivered",
                    old.id
                )
            });
            assert!(
                *held == *old,
                "{context}: subscription-oracle divergence: updated row {} pre-image differs from \
                 client-held state\nheld: {held:?}\nsent old: {old:?}",
                old.id
            );
            self.rows.insert(new.id, new.clone());
        }
        for row in &update.delta.added {
            let previous = self.rows.insert(row.id, row.clone());
            assert!(
                previous.is_none(),
                "{context}: subscription-oracle divergence: added row {} was already held",
                row.id
            );
        }

        self.ordered = apply_ordered_delta(context, &self.ordered, &update.ordered_delta);
        assert_eq!(
            self.ordered.len(),
            self.rows.len(),
            "{context}: subscription-oracle divergence: ordered ids and row map disagree after \
             applying the delta"
        );
        for id in &self.ordered {
            assert!(
                self.rows.contains_key(id),
                "{context}: subscription-oracle divergence: ordered id {id} has no row"
            );
        }
    }

    /// Decode the mirrored rows into the (parent, children, grandchildren)
    /// id tree, in mirror order.
    fn tree(&self, context: &str) -> ExpectedTree {
        if self.ordered.is_empty() {
            return Vec::new();
        }
        let descriptor = self
            .descriptor
            .as_ref()
            .unwrap_or_else(|| panic!("{context}: mirror holds rows but saw no descriptor"));
        self.ordered
            .iter()
            .map(|parent_id| {
                let row = &self.rows[parent_id];
                let values = decode_row(descriptor, &row.data)
                    .unwrap_or_else(|err| panic!("{context}: decode parent row failed: {err}"));
                let children = include_array(context, &values, PARENT_INCLUDE_COLUMN)
                    .iter()
                    .map(|child| {
                        let (child_id, child_values) = as_included_row(context, child);
                        let grandchildren =
                            include_array(context, child_values, CHILD_INCLUDE_COLUMN)
                                .iter()
                                .map(|grandchild| as_included_row(context, grandchild).0)
                                .collect();
                        (child_id, grandchildren)
                    })
                    .collect();
                (*parent_id, children)
            })
            .collect()
    }
}

fn include_array<'a>(context: &str, values: &'a [Value], column: usize) -> &'a [Value] {
    match values.get(column) {
        Some(Value::Array(items)) => items,
        other => panic!("{context}: expected include array at column {column}, got {other:?}"),
    }
}

fn as_included_row<'a>(context: &str, value: &'a Value) -> (ObjectId, &'a Vec<Value>) {
    match value {
        Value::Row {
            id: Some(id),
            values,
        } => (*id, values),
        other => panic!("{context}: expected included row with id, got {other:?}"),
    }
}

/// Reconstruct the post order from the pre order and an ordered delta, with
/// index-correctness checks. Survivor rows keep their relative order; every
/// added/updated entry names its exact post index.
fn apply_ordered_delta(context: &str, pre: &[ObjectId], delta: &OrderedRowDelta) -> Vec<ObjectId> {
    use std::collections::HashSet;

    let removed: HashSet<ObjectId> = delta.removed.iter().map(|entry| entry.id).collect();
    for entry in &delta.removed {
        assert_eq!(
            pre.get(entry.index),
            Some(&entry.id),
            "{context}: subscription-oracle divergence: removed index {} does not address row {}",
            entry.index,
            entry.id
        );
    }
    let repositioned: HashSet<ObjectId> = delta.updated.iter().map(|entry| entry.id).collect();
    for entry in &delta.updated {
        assert_eq!(
            pre.get(entry.old_index),
            Some(&entry.id),
            "{context}: subscription-oracle divergence: updated old index {} does not address \
             row {}",
            entry.old_index,
            entry.id
        );
    }

    let survivors: Vec<ObjectId> = pre
        .iter()
        .filter(|id| !removed.contains(id) && !repositioned.contains(id))
        .copied()
        .collect();
    let total = survivors.len() + delta.updated.len() + delta.added.len();
    let mut slots: Vec<Option<ObjectId>> = vec![None; total];
    for entry in &delta.added {
        assert!(
            entry.index < total && slots[entry.index].is_none(),
            "{context}: subscription-oracle divergence: added index {} is out of range or double-\
             booked (total {total})",
            entry.index
        );
        slots[entry.index] = Some(entry.id);
    }
    for entry in &delta.updated {
        assert!(
            entry.new_index < total && slots[entry.new_index].is_none(),
            "{context}: subscription-oracle divergence: updated new index {} is out of range or \
             double-booked (total {total})",
            entry.new_index
        );
        slots[entry.new_index] = Some(entry.id);
    }
    let mut survivors = survivors.into_iter();
    let post: Vec<ObjectId> = slots
        .into_iter()
        .map(|slot| {
            slot.unwrap_or_else(|| {
                survivors.next().unwrap_or_else(|| {
                    panic!(
                        "{context}: subscription-oracle divergence: ordered delta leaves \
                         unfillable holes"
                    )
                })
            })
        })
        .collect();
    assert!(
        survivors.next().is_none(),
        "{context}: subscription-oracle divergence: ordered delta drops surviving rows"
    );
    post
}

struct Engine<H: Storage> {
    qm: QueryManager,
    storage: H,
    branch: String,
    write_schema: Schema,
    subs: Vec<QuerySubscriptionId>,
    mirrors: Vec<Mirror>,
    name: &'static str,
    path: EnginePath,
    settle_passes: usize,
}

impl<H: Storage> Engine<H> {
    fn new(
        name: &'static str,
        path: EnginePath,
        schema: &Schema,
        make_storage: &dyn Fn(&Schema) -> H,
    ) -> Self {
        let mut qm = QueryManager::new(SyncManager::new());
        qm.set_current_schema(schema.clone(), "dev", "main");
        let storage = make_storage(&qm.schema_context().current_schema);
        let branch = get_branch(&qm);
        let write_schema = (*qm.schema).clone();
        Self {
            qm,
            storage,
            branch,
            write_schema,
            subs: Vec::new(),
            mirrors: Vec::new(),
            name,
            path,
            settle_passes: 0,
        }
    }

    fn subscribe_all(&mut self, context: &str) {
        let _path = self.path.engage();
        for spec in sub_specs() {
            let builder = self.qm.query(PARENT_TABLE);
            let builder = if spec.descending {
                builder.order_by_desc("ord")
            } else {
                builder.order_by("ord")
            };
            let builder = match spec.limit {
                Some(limit) => builder.limit(limit),
                None => builder,
            };
            let query = builder
                .with_array("children", |sub| {
                    sub.from(CHILD_TABLE)
                        .correlate("parent_id", "parents.id")
                        .filter_eq("is_deleted", Value::Boolean(false))
                        .order_by("ord")
                        .with_array("grandchildren", |sub2| {
                            sub2.from(GRANDCHILD_TABLE)
                                .correlate("child_id", "children.id")
                                .order_by("ord")
                        })
                })
                .build();
            let sub_id = self
                .qm
                .subscribe_with_session(query, Some(PolicySession::new(spec.user)), None)
                .unwrap_or_else(|err| {
                    panic!("{context}: engine {}: subscribe failed: {err:?}", self.name)
                });
            self.subs.push(sub_id);
            self.mirrors.push(Mirror::default());
        }
    }

    fn apply(&mut self, context: &str, op: &StampedOp) {
        let _path = self.path.engage();
        let write_context = WriteContext {
            session: None,
            attribution: Some(op.author.to_string()),
            updated_at: Some(op.ts),
            batch_mode: None,
            // NEVER set batch_id here: it flips the write to StagingPending
            // (open transaction batch), not a direct visible write.
            batch_id: None,
            target_branch_name: None,
        };
        match &op.kind {
            StampedOpKind::Insert { table, id, values } => {
                self.qm
                    .insert_on_branch_with_schema_and_write_context_and_id(
                        &mut self.storage,
                        table,
                        &self.branch.clone(),
                        values,
                        Some(*id),
                        &self.write_schema.clone(),
                        Some(&write_context),
                        true,
                    )
                    .unwrap_or_else(|err| {
                        panic!(
                            "{context}: engine {}: [{}] insert into {table} failed: {err:?}",
                            self.name, op.label
                        )
                    });
            }
            StampedOpKind::Update { id, values } => {
                self.qm
                    .update_with_write_context(&mut self.storage, *id, values, Some(&write_context))
                    .unwrap_or_else(|err| {
                        panic!(
                            "{context}: engine {}: [{}] update of {id} failed: {err:?}",
                            self.name, op.label
                        )
                    });
            }
            StampedOpKind::SoftDelete { id } => {
                self.qm
                    .delete_with_write_context(&mut self.storage, *id, Some(&write_context))
                    .unwrap_or_else(|err| {
                        panic!(
                            "{context}: engine {}: [{}] soft delete of {id} failed: {err:?}",
                            self.name, op.label
                        )
                    });
            }
        }
    }

    /// One settle pass; returns this pass's updates grouped per subscription.
    /// A single `process()` emits at most one update per subscription — the
    /// grouping is the ONLY normalization applied (see module docs).
    fn process_and_take(&mut self, context: &str) -> HashMap<u64, QueryUpdate> {
        let _path = self.path.engage();
        // Settle the way a runtime tick does: inside a read scope. Every other pass
        // flushes first, so on SQLite the settle reads in the scope's own transaction
        // one time and in the write transaction its writes left open the next — the
        // two places the read memo answers from.
        self.settle_passes += 1;
        if self.settle_passes.is_multiple_of(2) {
            self.storage.flush().expect("the oracle storage flushes");
        }
        self.storage.begin_read_scope();
        self.qm.process(&mut self.storage);
        self.storage.end_read_scope();
        let mut grouped: HashMap<u64, QueryUpdate> = HashMap::new();
        for update in self.qm.take_updates() {
            let key = update.subscription_id.0;
            assert!(
                !grouped.contains_key(&key),
                "{context}: engine {}: subscription {key} emitted more than one update in a \
                 single process() — the harness's one-batch-per-pass grouping no longer holds",
                self.name
            );
            grouped.insert(key, update);
        }
        grouped
    }

    fn assert_mirror_matches_engine(&self, context: &str, sub_index: usize) {
        let sub_id = self.subs[sub_index];
        let subscription = self
            .qm
            .subscriptions
            .get(&sub_id)
            .unwrap_or_else(|| panic!("{context}: engine {}: subscription lost", self.name));
        let mirror = &self.mirrors[sub_index];
        assert_eq!(
            mirror.ordered, subscription.current_ordered_ids,
            "{context}: engine {}: the emitted delta stream does not reconstruct the engine's \
             ordered result — a delta batch was missed or wrong",
            self.name
        );
        assert!(
            mirror.rows == subscription.current_visible_rows,
            "{context}: engine {}: the emitted delta stream does not reconstruct the engine's \
             visible rows — a delta batch was missed or wrong",
            self.name
        );
    }
}

// ============================================================================
// Fault injection (mutation-validation of the oracle itself).
// ============================================================================

/// Deliberate breakage for the `#[should_panic]` validation tests: each fault
/// simulates a bug class the oracle exists to catch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fault {
    None,
    /// Engine B silently misses one write — the data-divergence class.
    SkipOpOnB {
        op_index: usize,
    },
    /// Engine B drops one emitted delta batch — the missed-delta
    /// (F2 under-marking) class.
    DropFirstMutationUpdateOnB,
}

// ============================================================================
// The oracle run.
// ============================================================================

/// Cross-engine row identity: everything except the engine-minted batch-id
/// value (see the module docs on normalization).
type RowFingerprint = (
    ObjectId,
    crate::query_manager::types::RowBytes,
    RowProvenance,
);

fn row_fingerprint(row: &Row) -> RowFingerprint {
    (row.id, row.data.clone(), row.provenance.clone())
}

fn rows_fingerprint(rows: &[Row]) -> Vec<RowFingerprint> {
    rows.iter().map(row_fingerprint).collect()
}

fn assert_updates_equal(context: &str, a: &QueryUpdate, b: &QueryUpdate) {
    assert!(
        a.descriptor == b.descriptor,
        "{context}: subscription-oracle divergence: descriptors differ"
    );
    let updated_a: Vec<(RowFingerprint, RowFingerprint)> = a
        .delta
        .updated
        .iter()
        .map(|(old, new)| (row_fingerprint(old), row_fingerprint(new)))
        .collect();
    let updated_b: Vec<(RowFingerprint, RowFingerprint)> = b
        .delta
        .updated
        .iter()
        .map(|(old, new)| (row_fingerprint(old), row_fingerprint(new)))
        .collect();
    assert!(
        rows_fingerprint(&a.delta.added) == rows_fingerprint(&b.delta.added)
            && rows_fingerprint(&a.delta.removed) == rows_fingerprint(&b.delta.removed)
            && a.delta.moved == b.delta.moved
            && updated_a == updated_b,
        "{context}: subscription-oracle divergence: row deltas differ\nA: {:?}\nB: {:?}",
        a.delta,
        b.delta
    );
    let ordered_a = ordered_delta_fingerprint(&a.ordered_delta);
    let ordered_b = ordered_delta_fingerprint(&b.ordered_delta);
    assert!(
        ordered_a == ordered_b,
        "{context}: subscription-oracle divergence: ordered deltas differ\nA: {ordered_a:?}\nB: {ordered_b:?}"
    );
}

type OrderedDeltaFingerprint = (
    Vec<(ObjectId, usize, RowFingerprint)>,
    Vec<(ObjectId, usize)>,
    Vec<(ObjectId, usize, usize, Option<RowFingerprint>)>,
    bool,
);

fn ordered_delta_fingerprint(delta: &OrderedRowDelta) -> OrderedDeltaFingerprint {
    (
        delta
            .added
            .iter()
            .map(|entry| (entry.id, entry.index, row_fingerprint(&entry.row)))
            .collect(),
        delta
            .removed
            .iter()
            .map(|entry| (entry.id, entry.index))
            .collect(),
        delta
            .updated
            .iter()
            .map(|entry| {
                (
                    entry.id,
                    entry.old_index,
                    entry.new_index,
                    entry.row.as_ref().map(row_fingerprint),
                )
            })
            .collect(),
        delta.pending,
    )
}

/// The v13-0 finding, as a dedicated assertion: a cold-open snapshot may not
/// be empty when seeded data matches the subscription.
fn assert_initial_snapshot_completeness(
    context: &str,
    expected: &ExpectedTree,
    update: &QueryUpdate,
) {
    if !expected.is_empty() {
        assert!(
            !update.delta.added.is_empty(),
            "{context}: initial snapshot is empty but seeded data matches — a cold-open \
             subscription must serve matching rows in its first emit (v13-0 census finding: \
             7/16 subs served empty results for minutes)"
        );
    }
}

fn run_oracle<H: Storage>(
    seed: u64,
    make_storage: &dyn Fn(&Schema) -> H,
    fault: Fault,
    profile: OpProfile,
    (path_a, path_b): (EnginePath, EnginePath),
) {
    let schema = oracle_schema();
    let specs = sub_specs();
    let mut engine_a = Engine::new("A", path_a, &schema, make_storage);
    let mut engine_b = Engine::new("B", path_b, &schema, make_storage);
    let mut model = Model::default();
    let mut generator = OpGenerator::new(seed, profile);
    let mut op_index = 0usize;
    let mut drop_pending = fault == Fault::DropFirstMutationUpdateOnB;

    let apply_everywhere = |context: &str,
                            engine_a: &mut Engine<H>,
                            engine_b: &mut Engine<H>,
                            op: &StampedOp,
                            op_index: &mut usize| {
        engine_a.apply(context, op);
        if fault
            != (Fault::SkipOpOnB {
                op_index: *op_index,
            })
        {
            engine_b.apply(context, op);
        }
        *op_index += 1;
    };

    // ---- Seed phase (cold-open dataset exists BEFORE any subscription) ----
    let context = format!("seed {seed:#x} seed-phase");
    for op in generator.seed_ops(&mut model) {
        apply_everywhere(&context, &mut engine_a, &mut engine_b, &op, &mut op_index);
    }
    let leftover_a = engine_a.process_and_take(&context);
    let leftover_b = engine_b.process_and_take(&context);
    assert!(
        leftover_a.is_empty() && leftover_b.is_empty(),
        "{context}: updates emitted before any subscription existed"
    );

    // ---- Cold open: subscribe, then at most SETTLE_PROCESS_BOUND passes ----
    let context = format!("seed {seed:#x} cold-open");
    engine_a.subscribe_all(&context);
    engine_b.subscribe_all(&context);
    let mut updates_a = HashMap::new();
    let mut updates_b = HashMap::new();
    for _pass in 0..SETTLE_PROCESS_BOUND {
        updates_a.extend(engine_a.process_and_take(&context));
        updates_b.extend(engine_b.process_and_take(&context));
        if updates_a.len() == specs.len() && updates_b.len() == specs.len() {
            break;
        }
    }
    for (sub_index, spec) in specs.iter().enumerate() {
        let sub_context = format!("{context} sub#{sub_index} ({spec:?})");
        let key = engine_a.subs[sub_index].0;
        let update_a = updates_a.get(&key).unwrap_or_else(|| {
            panic!(
                "{sub_context}: no initial snapshot within {SETTLE_PROCESS_BOUND} process() \
                 pass(es) — settle-to-first-emit bound violated (v13-0 census finding)"
            )
        });
        let update_b = updates_b.get(&key).unwrap_or_else(|| {
            panic!(
                "{sub_context}: subscription-oracle divergence: engine B produced no initial \
                 snapshot while engine A did"
            )
        });
        assert_updates_equal(&sub_context, update_a, update_b);
        let expected = model.expected_tree(spec);
        assert_initial_snapshot_completeness(&sub_context, &expected, update_a);
        engine_a.mirrors[sub_index].apply_update(&sub_context, update_a);
        engine_b.mirrors[sub_index].apply_update(&sub_context, update_b);
        engine_a.assert_mirror_matches_engine(&sub_context, sub_index);
        engine_b.assert_mirror_matches_engine(&sub_context, sub_index);
        assert_eq!(
            engine_a.mirrors[sub_index].tree(&sub_context),
            expected,
            "{sub_context}: subscription-oracle model mismatch: initial snapshot does not match \
             the reference model"
        );
    }

    // ---- Mutation phase ----
    for batch in 0..MUTATION_BATCHES {
        let context = format!("seed {seed:#x} batch #{batch}");
        for op in generator.batch_ops(&mut model) {
            let op_context = format!("{context} [{}]", op.label);
            apply_everywhere(
                &op_context,
                &mut engine_a,
                &mut engine_b,
                &op,
                &mut op_index,
            );
        }

        // Settle passes: one `process()` can leave include arrays a pass
        // behind their inner-table writes (the inner delta re-dirties the
        // outer graph), so pump BOTH engines in lockstep until quiescent —
        // comparing every pass — and bound the pass count. The model check
        // then runs against the quiesced state.
        let mut passes = 0usize;
        loop {
            let pass_context = format!("{context} settle-pass #{passes}");
            let updates_a = engine_a.process_and_take(&pass_context);
            let mut updates_b = engine_b.process_and_take(&pass_context);
            if drop_pending && !updates_b.is_empty() {
                let dropped = *updates_b.keys().min().expect("non-empty update map");
                updates_b.remove(&dropped);
                drop_pending = false;
            }

            let mut keys_a: Vec<u64> = updates_a.keys().copied().collect();
            let mut keys_b: Vec<u64> = updates_b.keys().copied().collect();
            keys_a.sort_unstable();
            keys_b.sort_unstable();
            assert_eq!(
                keys_a, keys_b,
                "{pass_context}: subscription-oracle divergence: the sets of subscriptions \
                 emitting updates differ between the engines"
            );

            for key in keys_a.iter().copied() {
                let sub_index = engine_a
                    .subs
                    .iter()
                    .position(|sub_id| sub_id.0 == key)
                    .expect("update for an unknown subscription");
                let sub_context = format!("{pass_context} sub#{sub_index}");
                assert_updates_equal(&sub_context, &updates_a[&key], &updates_b[&key]);
                engine_a.mirrors[sub_index].apply_update(&sub_context, &updates_a[&key]);
                engine_b.mirrors[sub_index].apply_update(&sub_context, &updates_b[&key]);
            }

            if keys_a.is_empty() {
                break;
            }
            passes += 1;
            assert!(
                passes <= SETTLE_PASS_BOUND,
                "{context}: engines did not quiesce within {SETTLE_PASS_BOUND} settle passes \
                 after one mutation batch"
            );
        }

        // Model + engine-state checks run for EVERY subscription, updated or
        // not: a change the engine failed to emit (under-marking) leaves the
        // mirror stale and fails the model comparison right here.
        for (sub_index, spec) in specs.iter().enumerate() {
            let sub_context = format!("{context} sub#{sub_index} ({spec:?})");
            engine_a.assert_mirror_matches_engine(&sub_context, sub_index);
            engine_b.assert_mirror_matches_engine(&sub_context, sub_index);
            assert_eq!(
                engine_a.mirrors[sub_index].tree(&sub_context),
                model.expected_tree(spec),
                "{sub_context}: subscription-oracle model mismatch after mutation batch"
            );
        }
    }
}

fn memory_storage_factory(schema: &Schema) -> MemoryStorage {
    seeded_memory_storage(schema)
}

// ============================================================================
// Tests.
// ============================================================================

/// The kill-switch regression guard: on the ops whose effect the legacy
/// path handles correctly, `JAZZ_PRECISE_DIRTY=0` and the precise default
/// must produce byte-identical output streams. (The full matrix cannot run
/// legacy-vs-precise: the legacy side swallows inner content updates and
/// nested membership changes by design — that gap is exactly what F2 fixed,
/// and it is pinned by `include_array_reflects_inner_row_content_update`.)
#[test]
fn subscription_output_differential_legacy_vs_precise_on_legacy_parity_ops() {
    for seed in SEEDS {
        run_oracle(
            seed,
            &memory_storage_factory,
            Fault::None,
            OpProfile::LegacyParity,
            (EnginePath::Legacy, EnginePath::PreciseDirty),
        );
    }
}

#[cfg(feature = "rocksdb")]
#[test]
fn subscription_output_differential_random_ops_rocksdb() {
    use crate::storage::RocksDBStorage;
    use crate::test_support::persist_test_schema;

    let factory = |schema: &Schema| {
        let dir = tempfile::TempDir::new().expect("tempdir for rocksdb oracle");
        let path = dir.path().join("subscription-oracle.rocksdb");
        let mut storage =
            RocksDBStorage::open(&path, 8 * 1024 * 1024).expect("open rocksdb oracle storage");
        // Keep the directory alive for the whole process; the OS reclaims it.
        std::mem::forget(dir);
        persist_test_schema(&mut storage, schema);
        storage
    };
    // Two seeds: rocksdb runs the identical logic through the persistent
    // backend; the memory run carries the seed breadth.
    for seed in &SEEDS[..2] {
        run_oracle(
            *seed,
            &factory,
            Fault::None,
            OpProfile::FullIncludingFilterFlips,
            (EnginePath::PreciseDirty, EnginePath::PreciseDirty),
        );
    }
}

/// The FULL §6.2 op matrix, filter flips included — the default oracle
/// profile since F2 (v13-2). Both engines run the precise path; the model
/// layer (per-batch id-tree comparison against the generator-side reference)
/// is what proves every mutation class reaches the output stream.
#[test]
fn subscription_output_differential_full_ops_including_filter_flips() {
    for seed in SEEDS {
        run_oracle(
            seed,
            &memory_storage_factory,
            Fault::None,
            OpProfile::FullIncludingFilterFlips,
            (EnginePath::PreciseDirty, EnginePath::PreciseDirty),
        );
    }
}

/// The v14 L1 guard: correlation routing must change COST, not output.
///
/// One engine broadcasts every buffered mark to every cached instance (the
/// v13-3 path, `JAZZ_INCLUDE_ROUTING=0`), the other routes each mark to the
/// instances its correlate resolves to. On the FULL op matrix — nested
/// grandchild inserts and moves, `is_deleted` filter flips, correlate-value
/// moves, an ordered+limited outer window, per-session SELECT policies — the
/// two output streams must be byte-identical.
///
/// This is the only thing standing between the lever and a silent staleness
/// regression: under-routing produces no error and no log, just a subscriber
/// that never learns a row changed. The broadcast side cannot under-route by
/// construction, so any divergence here is a routing hole. It found one during
/// development — a re-parented child left in the OLD instance's incremental
/// scan baseline, because the reverse index was built from output arrays
/// instead of scan membership.
#[test]
fn subscription_output_differential_routed_vs_unrouted_on_full_ops() {
    for seed in SEEDS {
        run_oracle(
            seed,
            &memory_storage_factory,
            Fault::None,
            OpProfile::FullIncludingFilterFlips,
            (EnginePath::PreciseUnrouted, EnginePath::PreciseDirty),
        );
    }
}

/// The v24 guard: answering an empty include binding from the index must change
/// COST, not output.
///
/// One engine compiles an instance for every binding (the v23 path,
/// `JAZZ_INCLUDE_EMPTY_PROBE=0`), the other holds an `EmptyBinding` wherever the
/// correlation index proves the instance would scan nothing. On the FULL op
/// matrix the two output streams must be byte-identical, and the model layer
/// checks both against the reference tree.
///
/// What this guards is a binding that stays "empty" after a row arrived for it:
/// like under-routing it raises no error and no log, only an include array that
/// never fills. The instance side cannot make that mistake by construction.
///
/// The run also has to be about empty bindings at all: an include the probe
/// judges ineligible would make both sides run instances and pass vacuously.
#[test]
fn subscription_output_differential_empty_bindings_vs_instances_on_full_ops() {
    use crate::query_manager::graph_nodes::include_routing::empty_bindings_bound_on_this_thread;

    let bound_before = empty_bindings_bound_on_this_thread();
    for seed in SEEDS {
        run_oracle(
            seed,
            &memory_storage_factory,
            Fault::None,
            OpProfile::FullIncludingFilterFlips,
            (EnginePath::PreciseNoEmptyProbe, EnginePath::PreciseDirty),
        );
    }
    let bound = empty_bindings_bound_on_this_thread() - bound_before;
    assert!(
        bound > 100,
        "the oracle bound only {bound} empty bindings: its includes are not what the probe answers"
    );
}

/// The same guard on the backend the application runs: the probe reads the
/// correlation index through `Storage::index_lookup`, and SQLite answers it
/// from its own key scans and read memo rather than from `MemoryStorage`'s maps.
#[test]
fn subscription_output_differential_empty_bindings_vs_instances_on_sqlite() {
    use crate::query_manager::graph_nodes::include_routing::empty_bindings_bound_on_this_thread;
    use crate::storage::SqliteStorage;
    use crate::test_support::persist_test_schema;

    let factory = |schema: &Schema| {
        let mut storage = SqliteStorage::open(":memory:").expect("in-memory sqlite storage");
        persist_test_schema(&mut storage, schema);
        storage
    };
    let bound_before = empty_bindings_bound_on_this_thread();
    for seed in &SEEDS[..3] {
        run_oracle(
            *seed,
            &factory,
            Fault::None,
            OpProfile::FullIncludingFilterFlips,
            (EnginePath::PreciseNoEmptyProbe, EnginePath::PreciseDirty),
        );
    }
    // Both engines are the same engine wherever the probe does not fire — a probe that
    // SQLite never answers would pass this differential and prove nothing.
    let bound = empty_bindings_bound_on_this_thread() - bound_before;
    assert!(
        bound > 50,
        "the oracle bound only {bound} empty bindings on sqlite: the probe is not answering there"
    );
}

/// Minimal pin of the fixed FINDING (module docs): a child row's `is_deleted`
/// flip must retract it from a `filter_eq("is_deleted", false)` include.
/// Before F2 the include array was byte-stale forever on inner content
/// updates (also via the remote sync-inbox path); the precise-dirty path
/// forwards content marks into the include's subgraph instances.
#[test]
fn include_array_reflects_inner_row_content_update() {
    let _precise = crate::query_manager::precise_dirty::force_precise_dirty(true);
    let sync_manager = SyncManager::new();
    let mut schema = Schema::new();
    schema.insert(
        TableName::new("parents"),
        RowDescriptor::new(vec![ColumnDescriptor::new("ord", ColumnType::Integer)]).into(),
    );
    schema.insert(
        TableName::new("children"),
        RowDescriptor::new(vec![
            ColumnDescriptor::new("ord", ColumnType::Integer),
            ColumnDescriptor::new("is_deleted", ColumnType::Boolean),
            ColumnDescriptor::new("parent_id", ColumnType::Uuid),
        ])
        .into(),
    );
    let (mut qm, mut storage) = create_query_manager(sync_manager, schema);
    let parent = qm
        .insert(&mut storage, "parents", &[Value::Integer(1)])
        .unwrap();
    let child = qm
        .insert(
            &mut storage,
            "children",
            &[
                Value::Integer(2),
                Value::Boolean(false),
                Value::Uuid(parent.row_id),
            ],
        )
        .unwrap();
    let query = qm
        .query("parents")
        .with_array("children", |sub| {
            sub.from("children")
                .correlate("parent_id", "parents.id")
                .filter_eq("is_deleted", Value::Boolean(false))
                .order_by("ord")
        })
        .build();
    let sub_id = qm.subscribe(query).unwrap();
    qm.process(&mut storage);
    let initial = qm.take_updates();
    let initial = initial
        .iter()
        .find(|update| update.subscription_id == sub_id)
        .expect("initial snapshot");
    let values = decode_row(&initial.descriptor, &initial.delta.added[0].data).unwrap();
    assert_eq!(
        values[1].as_array().map(<[Value]>::len),
        Some(1),
        "child starts inside the include"
    );

    // Flip the include's filter column on the child.
    qm.update(
        &mut storage,
        child.row_id,
        &[
            Value::Integer(2),
            Value::Boolean(true),
            Value::Uuid(parent.row_id),
        ],
    )
    .unwrap();
    for _ in 0..SETTLE_PASS_BOUND {
        qm.process(&mut storage);
    }
    qm.take_updates();

    let results = qm.get_subscription_results(sub_id);
    let include = results[0].1[1]
        .as_array()
        .expect("include column stays an array");
    assert!(
        include.is_empty(),
        "a child whose is_deleted flipped to true must leave the \
         filter_eq(is_deleted, false) include; the engine still serves {include:?}"
    );
}

/// Mutation-validation: an engine that silently misses a write must be
/// caught. Op #0 is the first seeded parent, so engine B's cold-open
/// snapshot diverges immediately.
#[test]
#[should_panic(expected = "subscription-oracle divergence")]
fn oracle_catches_an_engine_that_missed_a_write() {
    run_oracle(
        SEEDS[0],
        &memory_storage_factory,
        Fault::SkipOpOnB { op_index: 0 },
        OpProfile::FullIncludingFilterFlips,
        (EnginePath::PreciseDirty, EnginePath::PreciseDirty),
    );
}

/// Mutation-validation: a dropped delta batch (the F2 under-marking symptom)
/// must be caught by the stream comparison.
#[test]
#[should_panic(expected = "subscription-oracle divergence")]
fn oracle_catches_a_dropped_delta_batch() {
    run_oracle(
        SEEDS[0],
        &memory_storage_factory,
        Fault::DropFirstMutationUpdateOnB,
        OpProfile::FullIncludingFilterFlips,
        (EnginePath::PreciseDirty, EnginePath::PreciseDirty),
    );
}

/// Mutation-validation: the empty-cold-open detector itself (v13-0 finding).
#[test]
#[should_panic(expected = "initial snapshot is empty but seeded data matches")]
fn oracle_catches_an_empty_initial_snapshot() {
    use crate::query_manager::types::RowDelta;

    let expected: ExpectedTree = vec![(ObjectId::new(), Vec::new())];
    let empty_update = QueryUpdate {
        subscription_id: QuerySubscriptionId(0),
        delta: RowDelta::new(),
        ordered_delta: OrderedRowDelta::default(),
        descriptor: RowDescriptor::new(vec![]),
    };
    assert_initial_snapshot_completeness("validation", &expected, &empty_update);
}
