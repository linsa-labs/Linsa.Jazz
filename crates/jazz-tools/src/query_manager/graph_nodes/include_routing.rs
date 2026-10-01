//! Correlation routing for include (array-subquery) dirt — v14 lever L1.
//!
//! ## Why
//!
//! v13-3 moved inner-table marks off the write path into
//! [`PendingInnerDirt`], a per-node buffer applied to a cached subgraph
//! instance the next time that instance is evaluated. The buffer carried ONE
//! node-global generation, so a single mark made EVERY cached instance look
//! stale and `reevaluate_all` visited all of them. Measured on the live
//! server: one presence heartbeat (a write to one row of `users`) cost 23 818
//! instance evaluations to deliver 0–7 rows, ~0.5 CPU-seconds every 11 s, and
//! the same recomputation on the phone blocks the app's own reads.
//!
//! Routing replaces the generation: each buffered row is resolved ONCE per
//! settle to the cache entries that can hold it, and only those instances are
//! marked. An instance that cannot hold the row is never touched, so it stays
//! clean and `reevaluate_all` skips it.
//!
//! ## How a row is resolved
//!
//! Two indices, both maintained by [`DirtRouting`]:
//!
//! * `by_correlation` — the instance bindings. Every cache entry is bound to
//!   one correlation value (the outer row's correlate column, or its object id
//!   for `Correlate::Id`), and the include's inner query is exactly
//!   `inner_column = <that value>`. So an inner row whose `inner_column` reads
//!   `V` can only ever be served by the instances bound to `V`. This is what
//!   routes INSERTS and the NEW home of a re-correlated row.
//! * `held_by` — the reverse index `row id → cache entries`, rebuilt from what
//!   an instance's subtree SCANS every time it is evaluated: its own index
//!   scans plus, recursively, the rows its nested includes track. Scan
//!   membership rather than array membership, because a row an instance scans
//!   but filters, windows or policy-hides is still a row whose change that
//!   instance must re-check. This is what routes UPDATES, DELETES, and the OLD
//!   home of a re-correlated row.
//!
//! A mark is delivered to the union of the two: a row that moved from parent A
//! to parent B is retracted from A (held) and added to B (correlation), in the
//! same settle.
//!
//! ## When routing is not worth it
//!
//! Resolving a row costs ONE row load per node per changed id, and buys the
//! instances it lets the settle skip. A node holding one or two instances has
//! nothing worth skipping, so `MIN_ROUTABLE_INSTANCES` sends it down the
//! broadcast path instead — measured, not assumed: the production-schema
//! profile averages ~1.3 instances per include node, and routing every node
//! there cost more allocation per write than it saved.
//!
//! Nested tables take the same path with a different key extraction: a
//! grandchild's own correlate column names its PARENT row, and the parent is a
//! row some instance holds — so `held_by[parent id]` is the answer. That only
//! works when the nested include correlates on the parent's identity
//! (`correlate("child_id", "children.id")`, the shape every production include
//! uses) and when the nested include cannot itself remove a parent from its
//! containing array (`ArraySubqueryRequirement::Optional`). Anything else is
//! [`Route::Broadcast`] — correct, just not flat, and named rather than
//! silently wrong.
//!
//! ## Why under-routing is the only dangerous direction
//!
//! Every mark is a "re-check this" instruction, so delivering one to an
//! instance that does not need it costs work and nothing else. NOT delivering
//! one to an instance that does need it is silent staleness: no error, no log.
//! Every case this module cannot resolve therefore broadcasts, and the
//! [`routing_enabled`] kill switch turns the whole resolution into a broadcast
//! so `subscription_output_oracle` can prove routed and unrouted output
//! byte-identical on the full op matrix.

use std::hash::{Hash, Hasher};

use ahash::{AHashMap, AHashSet};
use smallvec::SmallVec;

use crate::object::ObjectId;
use crate::query_manager::query::{ArraySubqueryRequirement, ArraySubquerySpec};
use crate::query_manager::types::{RowDescriptor, Schema, TableName, Value};

/// Identifies one cached subgraph instance: the outer row plus, for UUID[]
/// forward includes, the index of the correlate array element it serves.
pub(super) type CacheKey = (ObjectId, usize);

/// Cache entries a mark resolves to. One is the overwhelmingly common case
/// (one parent holds a given child); two covers a re-correlation moving a row
/// between instances without spilling.
pub(super) type CacheKeys = SmallVec<[CacheKey; 2]>;

/// Kill switch for correlation routing (v14 L1).
///
/// Default ON. `JAZZ_INCLUDE_ROUTING=0` (or `false`) resolves every mark to
/// [`Route::Broadcast`], which reproduces the v13-3 behaviour exactly — every
/// cached instance receives the whole payload and is re-evaluated. This is a
/// separate switch from `JAZZ_PRECISE_DIRTY`: that one reverts to the LEGACY
/// coarse path (which has known staleness bugs), this one keeps precise
/// dirtiness and only removes the routing decision, which is what makes a
/// routed-vs-unrouted differential run meaningful.
pub fn routing_enabled() -> bool {
    #[cfg(any(test, feature = "test"))]
    if let Some(forced) = test_override::forced() {
        return forced;
    }

    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("JAZZ_INCLUDE_ROUTING").as_deref(),
            Ok("0") | Ok("false")
        )
    })
}

#[cfg(any(test, feature = "test"))]
mod test_override {
    use std::sync::atomic::{AtomicU8, Ordering};
    use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};

    const UNSET: u8 = 0;
    const FORCED_ON: u8 = 1;
    const FORCED_OFF: u8 = 2;

    static STATE: AtomicU8 = AtomicU8::new(UNSET);

    fn lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    /// Holds include routing in a forced mode for the guard's lifetime. Same
    /// plug shape as `precise_dirty::force_precise_dirty`.
    pub struct IncludeRoutingMode {
        _serialised: MutexGuard<'static, ()>,
    }

    impl Drop for IncludeRoutingMode {
        fn drop(&mut self) {
            STATE.store(UNSET, Ordering::SeqCst);
        }
    }

    /// Force correlation routing on or off for the returned guard's lifetime.
    pub fn force_include_routing(enabled: bool) -> IncludeRoutingMode {
        let guard = lock().lock().unwrap_or_else(PoisonError::into_inner);
        STATE.store(
            if enabled { FORCED_ON } else { FORCED_OFF },
            Ordering::SeqCst,
        );
        IncludeRoutingMode { _serialised: guard }
    }

    pub(super) fn forced() -> Option<bool> {
        match STATE.load(Ordering::SeqCst) {
            FORCED_ON => Some(true),
            FORCED_OFF => Some(false),
            _ => None,
        }
    }
}

#[cfg(any(test, feature = "test"))]
pub use test_override::{IncludeRoutingMode, force_include_routing};

// ============================================================================
// Correlation keys
// ============================================================================

/// Hashable wrapper over a correlation [`Value`].
///
/// `Value` is `Eq` but not `Hash` — its `PartialEq` compares `f64` by bit
/// pattern on purpose, and deriving `Hash` on the public type would commit
/// every other consumer to that choice. The wrapper hashes the same way it
/// compares (discriminant + bits), which is all a hash map needs: equal values
/// hash equally, and a collision is resolved by `Eq`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct CorrKey(pub Value);

impl Hash for CorrKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        fn hash_value<H: Hasher>(value: &Value, state: &mut H) {
            std::mem::discriminant(value).hash(state);
            match value {
                Value::Integer(v) => v.hash(state),
                Value::BigInt(v) => v.hash(state),
                Value::Double(v) => v.to_bits().hash(state),
                Value::Boolean(v) => v.hash(state),
                Value::Text(v) => v.hash(state),
                Value::Timestamp(v) => v.hash(state),
                Value::Uuid(v) => v.hash(state),
                Value::BatchId(v) => v.hash(state),
                Value::Bytea(v) => v.hash(state),
                Value::Array(values) => {
                    values.len().hash(state);
                    for value in values {
                        hash_value(value, state);
                    }
                }
                Value::Row { id, values } => {
                    id.hash(state);
                    values.len().hash(state);
                    for value in values {
                        hash_value(value, state);
                    }
                }
                Value::Null => {}
            }
        }
        hash_value(&self.0, state);
    }
}

// ============================================================================
// Link resolution
// ============================================================================

/// Where a correlate value lives on a row of one inner table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CorrelateSource {
    /// A real column of the table's current descriptor.
    Column(usize),
    /// The row's own object id — `filter_eq("id"/"_id", v)` is a row-id
    /// condition (`query::is_row_id_condition_column`), not a column read.
    RowId,
}

/// How a changed row of one inner table resolves to cache entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TableLink {
    /// The include's own inner table. The row's correlate value IS an
    /// instance binding, so it resolves through `by_correlation`.
    Direct(CorrelateSource),
    /// A nested include's table whose correlate names the PARENT row's
    /// identity. The value read there is a row id held by some instance, so it
    /// resolves through `held_by`.
    NestedByParentId(CorrelateSource),
    /// Resolvable only by broadcast (see the module docs for the two shapes
    /// that land here).
    Unroutable,
}

fn correlate_source(descriptor: &RowDescriptor, column: &str) -> Option<CorrelateSource> {
    // An actual column of that name wins: a table may genuinely declare `id`.
    if let Some(index) = descriptor.column_index(column) {
        return Some(CorrelateSource::Column(index));
    }
    if crate::query_manager::query::is_row_id_condition_column(column) {
        return Some(CorrelateSource::RowId);
    }
    // `alias.column` reaches the filter builder qualified; the column itself
    // lives on the base table, exactly as the outer side resolves it in
    // `compile_array_subquery`.
    let unqualified = column.split('.').next_back()?;
    if unqualified == column {
        return None;
    }
    if let Some(index) = descriptor.column_index(unqualified) {
        return Some(CorrelateSource::Column(index));
    }
    crate::query_manager::query::is_row_id_condition_column(unqualified)
        .then_some(CorrelateSource::RowId)
}

// ============================================================================
// The routing state
// ============================================================================

/// Where a mark must go.
pub(super) enum Route {
    /// Exactly these cache entries. May be empty — that is the whole point:
    /// a row no live instance can hold costs nothing.
    Instances(CacheKeys),
    /// Every cached instance. Correct but O(instances); reached only from the
    /// shapes named in the module docs and from the kill switch.
    Broadcast,
}

/// Both routing indices plus the static link table, for one
/// `ArraySubqueryNode`.
#[derive(Debug)]
pub(super) struct DirtRouting {
    /// Rows an instance's evaluated array contains, nested rows included.
    held_by: AHashMap<ObjectId, CacheKeys>,
    /// Instances bound to a correlation value.
    by_correlation: AHashMap<CorrKey, CacheKeys>,
    /// Per inner table registered against this node, how to resolve a row of
    /// it. A table absent from this map is not one this node reads, and the
    /// caller (`QueryGraph::mark_*`) never forwards those here.
    links: AHashMap<TableName, TableLink>,
    /// False disables routing for this node entirely (see [`Self::new`]).
    routable: bool,
    /// The include's own inner table: the only table whose rows can turn an
    /// empty binding into a non-empty one (see [`EmptyBinding`]).
    inner_table: TableName,
}

impl DirtRouting {
    /// Build the static half from the include's own shape.
    ///
    /// `offset` is the inner query's OFFSET. Include subqueries never carry
    /// one (`ArraySubquerySpec` has no offset field), and the soundness of
    /// routing a vanished row to only the instances that held it depends on
    /// that: with an offset, deleting a row BEFORE an instance's window shifts
    /// the window even though the instance never held the row. The guard makes
    /// the dependency enforce itself instead of living in a comment.
    pub(super) fn new(
        schema: &Schema,
        inner_table: TableName,
        inner_column: &str,
        offset: usize,
        nested: &[ArraySubquerySpec],
    ) -> Self {
        let mut links = AHashMap::new();
        let direct = schema
            .get(&inner_table)
            .and_then(|table| correlate_source(&table.columns, inner_column))
            .map_or(TableLink::Unroutable, TableLink::Direct);
        links.insert(inner_table, direct);

        // Nested includes are registered against THIS node (see the
        // `nested_stack` walk in `graph/compile.rs`), so their tables arrive
        // here and must resolve to a parent row rather than to a binding.
        let mut stack: Vec<&ArraySubquerySpec> = nested.iter().collect();
        while let Some(spec) = stack.pop() {
            stack.extend(spec.nested_arrays.iter());
            let link = nested_link(schema, spec);
            match links.entry(spec.table) {
                std::collections::hash_map::Entry::Vacant(slot) => {
                    slot.insert(link);
                }
                std::collections::hash_map::Entry::Occupied(mut slot) => {
                    // The same table reached through two different nested
                    // shapes: keep it only if both agree, otherwise a single
                    // extraction rule cannot serve both.
                    if *slot.get() != link {
                        slot.insert(TableLink::Unroutable);
                    }
                }
            }
        }

        Self {
            held_by: AHashMap::new(),
            by_correlation: AHashMap::new(),
            links,
            routable: offset == 0,
            inner_table,
        }
    }

    /// Whether a mark for `table` can make an empty binding non-empty.
    ///
    /// Only a row of the include's own inner table can: an empty binding has no
    /// inner rows, so nothing nested under them exists to change. A table this
    /// node has no link for is answered "yes" — an unrecognised name must cost
    /// a re-probe, never a missed row.
    pub(super) fn can_fill_empty_binding(&self, table: &TableName) -> bool {
        *table == self.inner_table || !self.links.contains_key(table)
    }

    /// Which cache entries a mark for `(table, id)` must reach.
    ///
    /// `correlate` is the value read off the changed row at the position
    /// [`Self::correlate_position`] asked for, or `None` when the row could not
    /// be loaded. A row the subscription's loader cannot see cannot be
    /// materialized by any instance during this settle either, so the only
    /// instances that can be affected are the ones already holding it.
    pub(super) fn route(
        &self,
        table: &TableName,
        id: ObjectId,
        correlate: Option<&Value>,
    ) -> Route {
        if !self.routable || !routing_enabled() {
            return Route::Broadcast;
        }
        let mut keys: CacheKeys = self
            .held_by
            .get(&id)
            .map(|held| held.as_slice().into())
            .unwrap_or_default();

        let Some(link) = self.links.get(table) else {
            // Not a table this node reads. Nothing can hold its rows.
            return Route::Instances(keys);
        };
        let mut add = |extra: Option<&CacheKeys>| {
            for key in extra.into_iter().flatten() {
                if !keys.contains(key) {
                    keys.push(*key);
                }
            }
        };
        match (link, correlate) {
            (TableLink::Unroutable, _) => return Route::Broadcast,
            // The row is gone from this subscription's view: it can only leave
            // instances, never enter one.
            (_, None) => {}
            // A UUID[] foreign key: the row names several bindings at once, and
            // each element is a binding's correlation value in its own right.
            // Looked up whole, the array matches no binding and the row reaches
            // nobody.
            (TableLink::Direct(_), Some(Value::Array(elements))) => {
                for element in elements {
                    add(self.by_correlation.get(&CorrKey(element.clone())));
                }
            }
            (TableLink::Direct(_), Some(value)) => {
                add(self.by_correlation.get(&CorrKey(value.clone())));
            }
            (TableLink::NestedByParentId(_), Some(Value::Uuid(parent))) => {
                add(self.held_by.get(parent));
            }
            // A nested correlate that is not a row id cannot name a parent row.
            (TableLink::NestedByParentId(_), Some(_)) => return Route::Broadcast,
        }
        Route::Instances(keys)
    }

    /// Whether a scan of `table` by `column` is a scan by this include's
    /// correlation: the same table, and the same place on its rows that
    /// routing reads the correlate from. A planner free to scan by another
    /// column — the correlation column carries no index, a filter's does —
    /// produces a scan that says nothing about which bindings are empty.
    pub(super) fn scans_the_correlate(
        &self,
        schema: &Schema,
        table: &TableName,
        column: &str,
    ) -> bool {
        let Some(TableLink::Direct(source)) = self.links.get(&self.inner_table) else {
            return false;
        };
        *table == self.inner_table
            && schema
                .get(table)
                .and_then(|table_schema| correlate_source(&table_schema.columns, column))
                == Some(*source)
    }

    /// Where the correlate value of a row of `table` sits, or `None` when this
    /// node routes rows of that table by broadcast (so the caller must not pay
    /// for a row load at all).
    pub(super) fn correlate_position(&self, table: &TableName) -> Option<CorrelateColumn> {
        if !self.routable || !routing_enabled() {
            return None;
        }
        match self.links.get(table)? {
            TableLink::Unroutable => None,
            TableLink::Direct(source) | TableLink::NestedByParentId(source) => {
                Some(CorrelateColumn(*source))
            }
        }
    }

    /// Register a cache entry's correlation binding.
    pub(super) fn bind(&mut self, key: CacheKey, correlation_value: &Value) {
        let slot = self
            .by_correlation
            .entry(CorrKey(correlation_value.clone()))
            .or_default();
        if !slot.contains(&key) {
            slot.push(key);
        }
    }

    /// Drop a cache entry from both indices.
    pub(super) fn unbind(&mut self, key: CacheKey, correlation_value: &Value, held: &[ObjectId]) {
        remove_key(
            &mut self.by_correlation,
            &CorrKey(correlation_value.clone()),
            key,
        );
        for id in held {
            remove_key(&mut self.held_by, id, key);
        }
    }

    /// Replace the rows a cache entry holds, keeping `held_by` exact.
    ///
    /// Both slices are sorted and deduplicated, so the diff is a pair of
    /// binary-searched passes: a re-evaluation that changes nothing touches no
    /// map slot at all, which is what keeps a settled include's index free of
    /// churn.
    pub(super) fn set_held(&mut self, key: CacheKey, previous: &[ObjectId], current: &[ObjectId]) {
        debug_assert!(current.windows(2).all(|pair| pair[0] < pair[1]));
        for id in previous {
            if current.binary_search(id).is_err() {
                remove_key(&mut self.held_by, id, key);
            }
        }
        for id in current {
            if previous.binary_search(id).is_ok() {
                continue;
            }
            let slot = self.held_by.entry(*id).or_default();
            if !slot.contains(&key) {
                slot.push(key);
            }
        }
    }

    /// Every row this node's instances track, for an ENCLOSING include's index.
    pub(super) fn extend_tracked_row_ids(&self, out: &mut Vec<ObjectId>) {
        out.extend(self.held_by.keys().copied());
    }

    /// Rows tracked in the reverse index. Exposed for the size gate: the index
    /// must stay Σ(rows the instances scan), the same order as the data they
    /// already hold.
    #[cfg(any(test, feature = "test"))]
    pub(super) fn tracked_rows(&self) -> usize {
        self.held_by.len()
    }
}

/// Where to read a changed row's correlate value.
#[derive(Debug, Clone, Copy)]
pub(super) struct CorrelateColumn(CorrelateSource);

impl CorrelateColumn {
    /// Read the correlate value out of a decoded row.
    pub(super) fn read(&self, id: ObjectId, values: &[Value]) -> Option<Value> {
        match self.0 {
            CorrelateSource::RowId => Some(Value::Uuid(id)),
            CorrelateSource::Column(index) => values.get(index).cloned(),
        }
    }

    /// True when the value needs no row load at all.
    pub(super) fn is_row_id(&self) -> bool {
        matches!(self.0, CorrelateSource::RowId)
    }
}

fn nested_link(schema: &Schema, spec: &ArraySubquerySpec) -> TableLink {
    // A nested include that can drop its PARENT from the containing array
    // (`AtLeastOne` / `MatchCorrelationCardinality`) makes a parent that is
    // absent today re-appear when a child arrives — and an absent parent is
    // exactly what `held_by` cannot name.
    if spec.requirement != ArraySubqueryRequirement::Optional {
        return TableLink::Unroutable;
    }
    // Only a parent-identity correlate lands in `held_by`; a value correlate
    // would need a per-column index over held rows, which no production
    // include shape asks for.
    let outer = spec.outer_column.split('.').next_back().unwrap_or_default();
    if !crate::query_manager::query::is_row_id_condition_column(outer) {
        return TableLink::Unroutable;
    }
    schema
        .get(&spec.table)
        .and_then(|table| correlate_source(&table.columns, &spec.inner_column))
        .map_or(TableLink::Unroutable, TableLink::NestedByParentId)
}

fn remove_key<K: Hash + Eq>(index: &mut AHashMap<K, CacheKeys>, slot: &K, key: CacheKey) {
    let Some(keys) = index.get_mut(slot) else {
        return;
    };
    keys.retain(|held| *held != key);
    if keys.is_empty() {
        index.remove(slot);
    }
}

// ============================================================================
// The write-time buffer
// ============================================================================

/// Inner-table dirt recorded at WRITE time and routed to the instances that
/// can hold it at SETTLE time (v13-3 buffered it, v14 routes it).
///
/// WHY the buffer: v13-2 threaded every mark into every cached instance
/// eagerly, inside `mark_table_dependents_dirty` and the `forward_rows_*`
/// siblings. That made one write cost O(live outer rows) on the write path —
/// for every table the subscription touches, because the content and removal
/// channels were table-blind. Buffering keeps the write path O(1) in the
/// instance count (`include_inner_write_marking_is_flat_in_cached_instance_count`
/// pins that axis); routing at settle time then keeps the settle O(changed
/// rows) instead of O(instances), which is what the write path's sibling gate
/// pins.
///
/// The buffer is a SNAPSHOT, not a log. It is drained whole on the first
/// settle that reaches the node, so nothing carries across ticks.
#[derive(Debug, Default)]
pub(super) struct PendingInnerDirt {
    /// Row-precise membership marks, per inner table.
    pub(super) rows: AHashMap<TableName, AHashSet<ObjectId>>,
    /// Inner tables marked with no row information: a full rescan on apply.
    /// Dominates `rows` for the same table, exactly as `IndexScanNode`'s
    /// `needs_full` dominates its pending set. Carries no row id, so it cannot
    /// be routed — it broadcasts.
    pub(super) full_tables: AHashSet<TableName>,
    /// Content re-load marks, per inner table.
    pub(super) updated: AHashMap<TableName, AHashSet<ObjectId>>,
    /// Removal marks, per inner table.
    pub(super) deleted: AHashMap<TableName, AHashSet<ObjectId>>,
}

/// Which table-keyed buffer a row mark belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Channel {
    /// Membership: `QueryGraph::mark_rows_changed_for_table`.
    Rows,
    /// Content re-load: `QueryGraph::mark_rows_updated`.
    Updated,
    /// Removal delta: `QueryGraph::mark_rows_deleted`.
    Deleted,
}

impl PendingInnerDirt {
    pub(super) fn is_empty(&self) -> bool {
        self.rows.is_empty()
            && self.full_tables.is_empty()
            && self.updated.is_empty()
            && self.deleted.is_empty()
    }

    /// Ids currently buffered, across every channel and inner table.
    pub(super) fn buffered_ids(&self) -> usize {
        fn total(per_table: &AHashMap<TableName, AHashSet<ObjectId>>) -> usize {
            per_table.values().map(|ids| ids.len()).sum()
        }
        total(&self.rows) + total(&self.updated) + total(&self.deleted)
    }

    pub(super) fn channel_mut(
        &mut self,
        channel: Channel,
    ) -> &mut AHashMap<TableName, AHashSet<ObjectId>> {
        match channel {
            Channel::Rows => &mut self.rows,
            Channel::Updated => &mut self.updated,
            Channel::Deleted => &mut self.deleted,
        }
    }

    pub(super) fn insert(&mut self, channel: Channel, table: TableName, id: ObjectId) {
        self.channel_mut(channel)
            .entry(table)
            .or_default()
            .insert(id);
    }

    /// Every row mark, in a stable channel order.
    pub(super) fn marks(&self) -> impl Iterator<Item = (Channel, TableName, ObjectId)> + '_ {
        let channel = |channel: Channel, per_table: &'_ AHashMap<TableName, AHashSet<ObjectId>>| {
            per_table
                .iter()
                .flat_map(move |(table, ids)| ids.iter().map(move |id| (channel, *table, *id)))
                .collect::<Vec<_>>()
        };
        channel(Channel::Rows, &self.rows)
            .into_iter()
            .chain(channel(Channel::Updated, &self.updated))
            .chain(channel(Channel::Deleted, &self.deleted))
    }

    /// Thread the whole payload into one instance's graph, in the order the
    /// eager path delivered it: table-level marks (which force a full rescan)
    /// before row-precise ones, membership before content before removals —
    /// the order `apply_batched_subscription_visibility_effects` marks in.
    pub(super) fn apply_to(&self, graph: &mut crate::query_manager::graph::QueryGraph) {
        for table in &self.full_tables {
            graph.mark_dirty_for_table(table.as_str());
        }
        for (table, ids) in &self.rows {
            graph.mark_rows_changed_for_table(table.as_str(), ids);
        }
        for (table, ids) in &self.updated {
            graph.mark_rows_updated(table.as_str(), ids);
        }
        for (table, ids) in &self.deleted {
            graph.mark_rows_deleted(table.as_str(), ids);
        }
    }

    pub(super) fn clear(&mut self) {
        self.rows.clear();
        self.full_tables.clear();
        self.updated.clear();
        self.deleted.clear();
    }

    /// Every table the payload carries a mark for, on any channel.
    pub(super) fn tables(&self) -> impl Iterator<Item = &TableName> + '_ {
        self.full_tables
            .iter()
            .chain(self.rows.keys())
            .chain(self.updated.keys())
            .chain(self.deleted.keys())
    }
}

// ============================================================================
// Empty bindings
// ============================================================================

/// How an include answers "this parent has no rows" without an instance.
///
/// WHY: an instance is a compiled query graph, and an include compiles one per
/// outer row. Most of them hold nothing — a message has no attachments, no
/// calls, no task refs — yet each paid a plan compile, a settle, a re-settle on
/// every broadcast mark, and a graph drop. On a 50-row page of the app's thread
/// query that was ~340 of ~375 instances.
///
/// A plain include instance reads exactly `inner_column = <binding>` from the
/// column's index on each branch, and filters from there. If those lookups
/// return nothing, no later node can produce a row, so the answer is the empty
/// array and no graph is needed. The lookups are taken from a compiled
/// instance rather than re-derived, so they are the reads the instance itself
/// would make. Each has to be a scan BY the correlation
/// ([`DirtRouting::scans_the_correlate`]): a planner that scans by another
/// column, or a branch whose schema names the correlation column differently,
/// leaves the include ineligible.
#[derive(Debug, Default)]
pub(super) enum EmptyProbe {
    /// No instance compiled yet: the inner plan's reads are not known.
    #[default]
    Unknown,
    /// The inner plan reads something other than plain `inner_column = value`
    /// index lookups (a join, a window, soft-deleted rows, a second condition
    /// on the correlation column). Every binding gets an instance.
    Ineligible,
    /// The lookups one instance makes: `(table, index column, branch)`.
    Lookups(Vec<(TableName, crate::query_manager::types::ColumnName, String)>),
}

/// A cache slot proven empty by [`EmptyProbe`] instead of holding an instance.
///
/// It keeps the instance's place in `DirtRouting::by_correlation`, so a row
/// written with this correlation value routes here exactly as it would to an
/// empty instance, and `dirty` is that instance's "has dirty nodes": the slot
/// is probed again at the next evaluation.
#[derive(Debug)]
pub(super) struct EmptyBinding {
    pub(super) correlation_value: Value,
    pub(super) dirty: bool,
}

/// Kill switch for empty bindings.
///
/// Default ON. `JAZZ_INCLUDE_EMPTY_PROBE=0` (or `false`) compiles an instance
/// for every binding, as before — the mode the output oracle compares against.
pub fn empty_probe_enabled() -> bool {
    #[cfg(any(test, feature = "test"))]
    if let Some(forced) = empty_probe_override::forced() {
        return forced;
    }

    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("JAZZ_INCLUDE_EMPTY_PROBE").as_deref(),
            Ok("0") | Ok("false")
        )
    })
}

#[cfg(any(test, feature = "test"))]
mod empty_probe_override {
    use std::sync::atomic::{AtomicU8, Ordering};
    use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};

    const UNSET: u8 = 0;
    const FORCED_ON: u8 = 1;
    const FORCED_OFF: u8 = 2;

    static STATE: AtomicU8 = AtomicU8::new(UNSET);

    fn lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    /// Holds empty bindings in a forced mode for the guard's lifetime.
    pub struct IncludeEmptyProbeMode {
        _serialised: MutexGuard<'static, ()>,
    }

    impl Drop for IncludeEmptyProbeMode {
        fn drop(&mut self) {
            STATE.store(UNSET, Ordering::SeqCst);
        }
    }

    /// Force empty bindings on or off for the returned guard's lifetime.
    pub fn force_include_empty_probe(enabled: bool) -> IncludeEmptyProbeMode {
        let guard = lock().lock().unwrap_or_else(PoisonError::into_inner);
        STATE.store(
            if enabled { FORCED_ON } else { FORCED_OFF },
            Ordering::SeqCst,
        );
        IncludeEmptyProbeMode { _serialised: guard }
    }

    pub(super) fn forced() -> Option<bool> {
        match STATE.load(Ordering::SeqCst) {
            FORCED_ON => Some(true),
            FORCED_OFF => Some(false),
            _ => None,
        }
    }
}

#[cfg(any(test, feature = "test"))]
pub use empty_probe_override::{IncludeEmptyProbeMode, force_include_empty_probe};

/// Whether every empty binding is checked against the instance it stands in for.
///
/// "The correlation index holds nothing for this value, so the instance would yield
/// nothing" rests on what an instance reads: plain index lookups by the correlation,
/// with nothing laid over them. Nothing in the types says so, and a later change that
/// gives instances another source would break includes without a failing test — an
/// array that stays empty raises no error. With the check on, each binding answered
/// from the index is also compiled and settled, and must come back empty.
///
/// Off by default, in every build: it compiles the very instance the probe exists to
/// spare, so every count the probe moves reads as if the probe were off. Tests that
/// compare outputs turn it on for their thread
/// ([`check_empty_bindings_against_instances`]); `JAZZ_INCLUDE_EMPTY_PROBE_PARITY=1`
/// turns it on for a whole debug run.
#[cfg(any(test, feature = "test", debug_assertions))]
pub(super) fn empty_probe_parity_enabled() -> bool {
    #[cfg(any(test, feature = "test"))]
    if empty_probe_parity::on_this_thread() {
        return true;
    }

    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("JAZZ_INCLUDE_EMPTY_PROBE_PARITY").as_deref(),
            Ok("1") | Ok("true")
        )
    })
}

#[cfg(any(test, feature = "test"))]
mod empty_probe_parity {
    use std::cell::Cell;

    thread_local! {
        static DEPTH: Cell<u32> = const { Cell::new(0) };
    }

    /// Keeps the empty-binding check on for this thread until dropped. Not `Send`: the
    /// depth it holds is this thread's.
    pub struct EmptyBindingParity(std::marker::PhantomData<*const ()>);

    impl Drop for EmptyBindingParity {
        fn drop(&mut self) {
            DEPTH.with(|depth| depth.set(depth.get() - 1));
        }
    }

    /// Check every empty binding made on this thread against a compiled instance, for
    /// the returned guard's lifetime.
    pub fn check_empty_bindings_against_instances() -> EmptyBindingParity {
        DEPTH.with(|depth| depth.set(depth.get() + 1));
        EmptyBindingParity(std::marker::PhantomData)
    }

    pub(super) fn on_this_thread() -> bool {
        DEPTH.with(|depth| depth.get() > 0)
    }
}

#[cfg(any(test, feature = "test"))]
pub use empty_probe_parity::{EmptyBindingParity, check_empty_bindings_against_instances};

#[cfg(test)]
thread_local! {
    static EMPTY_BINDINGS_BOUND: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Empty bindings made on this thread so far. A gate reads the difference to
/// prove a run exercised them; unlike the `settle_cost` counter it is not
/// blended with whatever other tests are running in the process.
#[cfg(test)]
pub(crate) fn empty_bindings_bound_on_this_thread() -> u64 {
    EMPTY_BINDINGS_BOUND.with(std::cell::Cell::get)
}

#[cfg(test)]
pub(super) fn note_empty_binding_bound() {
    EMPTY_BINDINGS_BOUND.with(|bound| bound.set(bound.get() + 1));
}

/// The whole per-node include-dirt state, boxed as one unit by
/// `ArraySubqueryNode`.
///
/// Boxed because `ArraySubqueryNode` is the largest `GraphNode` variant and
/// `GraphNode` is sized by its fattest member, so every node slot of every
/// compiled graph pays for anything added here (v14 §2.2). One pointer, one
/// allocation per compiled include node.
#[derive(Debug)]
pub(super) struct IncludeDirt {
    pub(super) pending: PendingInnerDirt,
    pub(super) routing: DirtRouting,
    /// Recycled buffer for rebuilding one instance's tracked-row set.
    ///
    /// The rebuild is O(rows the instance scans) and runs on every instance
    /// evaluation, so a fresh `Vec` per evaluation would put the reverse
    /// index's maintenance cost straight into the allocator — which is exactly
    /// the "the index must not become the next O(N)" trap. The retired set is
    /// rotated back in here, so after warm-up the rebuild allocates nothing.
    pub(super) scratch: Vec<ObjectId>,
    /// How this include proves a binding empty (see [`EmptyProbe`]).
    pub(super) probe: EmptyProbe,
    /// Cache slots proven empty. A key is here or in the node's
    /// `subgraph_cache`, never both.
    pub(super) empty: AHashMap<CacheKey, EmptyBinding>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query_manager::types::{ColumnDescriptor, ColumnType, RowDescriptor};

    fn schema() -> Schema {
        let mut schema = Schema::new();
        schema.insert(
            TableName::new("children"),
            RowDescriptor::new(vec![
                ColumnDescriptor::new("title", ColumnType::Text),
                ColumnDescriptor::new("parent_id", ColumnType::Uuid),
            ])
            .into(),
        );
        schema.insert(
            TableName::new("grandchildren"),
            RowDescriptor::new(vec![
                ColumnDescriptor::new("title", ColumnType::Text),
                ColumnDescriptor::new("child_id", ColumnType::Uuid),
            ])
            .into(),
        );
        schema
    }

    fn nested_grandchildren() -> ArraySubquerySpec {
        let mut spec = ArraySubquerySpec::new("grandchildren", TableName::new("grandchildren"));
        spec.inner_column = "child_id".to_string();
        spec.outer_column = "children.id".to_string();
        spec
    }

    fn routing() -> DirtRouting {
        DirtRouting::new(
            &schema(),
            TableName::new("children"),
            "parent_id",
            0,
            &[nested_grandchildren()],
        )
    }

    #[test]
    fn a_row_no_instance_can_hold_routes_to_nothing() {
        let routing = routing();
        let stranger = ObjectId::new();
        let route = routing.route(
            &TableName::new("children"),
            stranger,
            Some(&Value::Uuid(ObjectId::new())),
        );
        assert!(matches!(route, Route::Instances(keys) if keys.is_empty()));
    }

    #[test]
    fn an_inserted_row_routes_to_the_instances_bound_to_its_correlate() {
        let mut routing = routing();
        let parent = ObjectId::new();
        let other_parent = ObjectId::new();
        routing.bind((parent, 0), &Value::Uuid(parent));
        routing.bind((other_parent, 0), &Value::Uuid(other_parent));

        let route = routing.route(
            &TableName::new("children"),
            ObjectId::new(),
            Some(&Value::Uuid(parent)),
        );
        let Route::Instances(keys) = route else {
            panic!("an inserted child with a known correlate must not broadcast");
        };
        assert_eq!(keys.as_slice(), &[(parent, 0)]);
    }

    #[test]
    fn a_recorrelated_row_reaches_both_its_old_and_its_new_home() {
        let mut routing = routing();
        let old_parent = ObjectId::new();
        let new_parent = ObjectId::new();
        let child = ObjectId::new();
        routing.bind((old_parent, 0), &Value::Uuid(old_parent));
        routing.bind((new_parent, 0), &Value::Uuid(new_parent));
        routing.set_held((old_parent, 0), &[], &[child]);

        let Route::Instances(keys) = routing.route(
            &TableName::new("children"),
            child,
            Some(&Value::Uuid(new_parent)),
        ) else {
            panic!("a re-correlated child must not broadcast");
        };
        assert!(keys.contains(&(old_parent, 0)), "retraction target missing");
        assert!(keys.contains(&(new_parent, 0)), "insertion target missing");
    }

    #[test]
    fn a_vanished_row_reaches_only_the_instances_that_held_it() {
        let mut routing = routing();
        let parent = ObjectId::new();
        let child = ObjectId::new();
        routing.bind((parent, 0), &Value::Uuid(parent));
        routing.set_held((parent, 0), &[], &[child]);

        let Route::Instances(keys) = routing.route(&TableName::new("children"), child, None) else {
            panic!("a deleted child must not broadcast");
        };
        assert_eq!(keys.as_slice(), &[(parent, 0)]);
    }

    #[test]
    fn a_grandchild_routes_through_the_instance_holding_its_parent() {
        let mut routing = routing();
        let parent = ObjectId::new();
        let child = ObjectId::new();
        routing.bind((parent, 0), &Value::Uuid(parent));
        routing.set_held((parent, 0), &[], &[child]);

        let Route::Instances(keys) = routing.route(
            &TableName::new("grandchildren"),
            ObjectId::new(),
            Some(&Value::Uuid(child)),
        ) else {
            panic!("a grandchild with a held parent must not broadcast");
        };
        assert_eq!(keys.as_slice(), &[(parent, 0)]);
    }

    #[test]
    fn a_nested_include_that_can_drop_its_parent_is_unroutable() {
        let mut nested = nested_grandchildren();
        nested.requirement = ArraySubqueryRequirement::AtLeastOne;
        let routing = DirtRouting::new(
            &schema(),
            TableName::new("children"),
            "parent_id",
            0,
            &[nested],
        );
        assert!(matches!(
            routing.route(
                &TableName::new("grandchildren"),
                ObjectId::new(),
                Some(&Value::Uuid(ObjectId::new()))
            ),
            Route::Broadcast
        ));
    }

    #[test]
    fn an_offset_bearing_subgraph_never_routes() {
        let routing = DirtRouting::new(
            &schema(),
            TableName::new("children"),
            "parent_id",
            3,
            &[nested_grandchildren()],
        );
        assert!(matches!(
            routing.route(&TableName::new("children"), ObjectId::new(), None),
            Route::Broadcast
        ));
    }

    #[test]
    fn dropping_an_instance_clears_both_indices() {
        let mut routing = routing();
        let parent = ObjectId::new();
        let child = ObjectId::new();
        routing.bind((parent, 0), &Value::Uuid(parent));
        routing.set_held((parent, 0), &[], &[child]);
        routing.unbind((parent, 0), &Value::Uuid(parent), &[child]);

        assert_eq!(routing.tracked_rows(), 0);
        assert!(matches!(
            routing.route(&TableName::new("children"), child, Some(&Value::Uuid(parent))),
            Route::Instances(keys) if keys.is_empty()
        ));
    }

    #[test]
    fn the_reverse_index_holds_one_entry_per_tracked_row() {
        let mut routing = routing();
        let parent = ObjectId::new();
        let mut rows = vec![ObjectId::new(), ObjectId::new(), ObjectId::new()];
        rows.sort_unstable();
        routing.bind((parent, 0), &Value::Uuid(parent));
        routing.set_held((parent, 0), &[], &rows);
        assert_eq!(routing.tracked_rows(), rows.len());

        // A re-evaluation that drops one row must shrink the index, not leak.
        routing.set_held((parent, 0), &rows, &rows[..2]);
        assert_eq!(routing.tracked_rows(), 2);
    }
}
