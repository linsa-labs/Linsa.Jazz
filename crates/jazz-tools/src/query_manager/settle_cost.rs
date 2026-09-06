//! Per-settle-pass cost accounting for the query settle path.
//!
//! # Why this exists
//!
//! On the live server, one chat message send costs a ~1 s burst of one core.
//! CPU-triggered `eu-stack` sampling during real sends caught the chain
//! `batched_tick -> QueryManager::process -> settle_server_subscriptions ->
//! settle_with_context -> ArraySubqueryNode::reevaluate_all ->
//! evaluate_subgraph_for_single -> SubgraphTemplate::instantiate ->
//! try_compile_with_schema_context_shared`, but only ~10 samples were caught,
//! so the RELATIVE split of that second is unproven. An isolated backend-role
//! repro cost ~70 ms/message; the gap to 1 s was attributed to policies plus
//! subscription fan-out and never measured.
//!
//! This module makes the split readable straight out of `docker logs`: every
//! settle pass that exceeds a threshold emits ONE `tracing::info!` line with
//! the exact counts of the work it did.
//!
//! # What a "settle pass" is
//!
//! One [`QueryManager::process`](crate::query_manager::manager::QueryManager::process)
//! call — the tick body that `batched_tick` drives. It covers both subscription
//! settle loops (local subscriptions and `settle_server_subscriptions`) plus
//! the write application that feeds them, which is exactly the unit the
//! operator sees as the CPU burst.
//!
//! # Cost model
//!
//! Counters are process-global relaxed atomics, always on and exact (never
//! sampled). A pass is measured as a snapshot difference: [`SettlePass::begin`]
//! records the counters and the clock, the drop computes the delta and only
//! then decides whether to log. Below the threshold a pass costs two
//! `Instant::now()` calls, ~11 relaxed loads at each end, and one relaxed
//! `fetch_add` per counted event. No allocation, no formatting, no per-row
//! logging.
//!
//! Global (rather than per-`QueryManager`) counters are correct because a
//! settle pass is single-threaded by construction: `process` takes `&mut self`
//! and, in the server runtime, the whole `RuntimeCore` is behind one mutex, so
//! ticks of one node never overlap. Several nodes ticking in ONE process (test
//! fleets) would blend their counts into whichever pass is open; tests that
//! assert on the numbers must serialise, exactly as the allocator-based gates
//! in `tests/include_instance_flatness.rs` already do.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use web_time::Instant;

use crate::sync_manager::ClientId;

/// Settle passes above this wall duration emit a cost line. Overridable with
/// `JAZZ_SETTLE_LOG_MS`; `0` logs every pass.
const DEFAULT_THRESHOLD_MS: u64 = 5;

/// Subscription graphs settled — one per subscription that actually did settle
/// work in the pass, at both the local (`QueryManager::process`) and the server
/// (`settle_server_subscriptions`) sites. Subscriptions short-circuited as
/// clean are NOT counted: they did no work.
pub static SUBSCRIPTIONS_SETTLED: AtomicU64 = AtomicU64::new(0);

/// Dirty graph nodes evaluated by `QueryGraph::settle_with_context`, summed
/// over every graph settled in the pass — including the per-instance subgraphs
/// of array subqueries, which settle through the same function.
pub static GRAPH_NODES_EVALUATED: AtomicU64 = AtomicU64::new(0);

/// Cached array-subquery instances (re-)evaluated —
/// `ArraySubqueryNode::evaluate_subgraph_for_single` calls. This is the (a) of
/// the split: work proportional to the number of live include instances rather
/// than to the size of the change.
pub static SUBQUERY_INSTANCE_EVALS: AtomicU64 = AtomicU64::new(0);

/// Array-subquery instantiations — `SubgraphTemplate::instantiate` calls, i.e.
/// instance evaluations that could NOT reuse a cached subgraph and had to
/// compile a fresh one. This is the (b) of the split.
pub static SUBQUERY_INSTANTIATIONS: AtomicU64 = AtomicU64::new(0);

/// Query plans compiled —
/// `QueryGraph::compile_execution_plan_with_schema_context_shared` calls and, since
/// v18 item 7, `compile_plan_with_branch_map` calls (the template's bound-plan path
/// bumps it there). A superset of [`SUBQUERY_INSTANTIATIONS`]: every instantiation
/// compiles one plan, and nested includes inside it compile more.
pub static PLAN_COMPILES: AtomicU64 = AtomicU64::new(0);

/// v18 item 7: include shapes lowered to an execution plan — every `lower_query`
/// call under `SubgraphTemplate` (`lower_shape` for the cached shape, the uncached
/// per-instance lowering, and the `RecursiveRelationNode` path), attempts included.
/// One per template after item 7 (one per instance before it); a template whose shape
/// cannot be cached (`uncacheable`) counts its failed attempt plus one per instance —
/// N + 1 for N instances. [`PLAN_COMPILES`] stays per instance and is the control.
pub static SHAPE_LOWERINGS: AtomicU64 = AtomicU64::new(0);

/// v18 item 6: settle units a pass left behind because the tick's budget was spent (a
/// registration, a dirty server subscription or a dirty local subscription). Under an
/// unbounded budget this never moves.
pub static SETTLE_UNITS_DEFERRED: AtomicU64 = AtomicU64::new(0);

/// v18 item 6: batched ticks re-armed by the settle flag — a pass that deferred
/// NON-stalled work under a budget, or an outbox-limiter trip that left registrations
/// behind under any budget. One per lock hold whose re-arm call RETURNED (diff r5).
pub static SETTLE_TICKS_REARMED: AtomicU64 = AtomicU64::new(0);

/// Per-row policy evaluations — `PolicyEvaluator::evaluate_row_access` calls,
/// counting recursive descent through referencing/inherited policies, since
/// that descent is the cost. This is the (c) of the split.
pub static POLICY_ROW_EVALS: AtomicU64 = AtomicU64::new(0);

/// Per-row select-policy checks run to authorize a session-scoped
/// subscription's sync scope —
/// `QueryManager::provenance_row_matches_current_select_policy` calls,
/// including the ones served from the cross-tick verdict cache.
pub static SCOPE_AUTHZ_CHECKS: AtomicU64 = AtomicU64::new(0);

/// The subset of [`SCOPE_AUTHZ_CHECKS`] that missed the verdict cache and
/// actually evaluated a policy against storage. The ratio of the two is what
/// says whether authorization is being paid per tick or amortised.
pub static SCOPE_AUTHZ_EVALS: AtomicU64 = AtomicU64::new(0);

/// Rows loaded through the query row loader
/// (`QueryManager::load_visible_row_for_query`) — the dominant storage read
/// driver of a settle. Part of the (d) of the split.
pub static ROW_LOADS: AtomicU64 = AtomicU64::new(0);

/// Index reads issued by `IndexScanNode`: one per full index scan and one per
/// incremental point membership probe. The rest of the (d) of the split.
pub static INDEX_READS: AtomicU64 = AtomicU64::new(0);

/// Rows a settled subscription emitted downstream — added + removed + updated
/// over the `RowDelta` of every subscription settled in the pass. The
/// denominator for everything above: the useful output the pass produced.
pub static ROWS_EMITTED: AtomicU64 = AtomicU64::new(0);

/// Cached subgraph instances currently held by every `ArraySubqueryNode` in the
/// process — a GAUGE, not a counter: it goes down as well as up, and a settle
/// pass reports its VALUE, never a delta (see [`SettleCounts::since`]).
///
/// It exists because [`SUBQUERY_INSTANCE_EVALS`] alone cannot say whether a
/// pass's evaluations came from many instances visited once or few instances
/// visited many times, and the two imply completely different fixes (routing
/// versus plan sharing). Divided into RSS it is also the only way to turn
/// "the server holds 1.6 GB" into a per-instance number without assuming that
/// evaluations and instances are the same population.
pub static LIVE_SUBQUERY_INSTANCES: AtomicU64 = AtomicU64::new(0);

/// Live `ArraySubqueryNode`s — the include nodes those instances are spread
/// over, one per include in each compiled (sub)graph. Also a gauge.
///
/// `LIVE_SUBQUERY_INSTANCES / LIVE_SUBQUERY_NODES` is the mean fan-out of an
/// include, which is what says whether the instance population comes from a
/// few wide includes or from many nested ones.
pub static LIVE_SUBQUERY_NODES: AtomicU64 = AtomicU64::new(0);

/// History scans issued — one per `scan_history_row_batches` or
/// `scan_history_region` call, whatever the pass was nominally doing.
///
/// The counters above are all bounded by the SIZE OF THE CHANGE: rows loaded,
/// index probes, instances re-evaluated. None of them can express work
/// proportional to what a row has ACCUMULATED, and a history scan decodes every
/// revision a row ever had. On a device store that difference is four orders of
/// magnitude — a fresh row scans in microseconds, a row with 9,885 heartbeat
/// revisions takes 45 ms, and a row holding a megabyte blob takes 11 ms for its
/// single entry. A settle pass reporting `row_loads=7` and a wall time of 900 ms
/// is unexplainable until these three are on the line.
pub static HISTORY_SCANS: AtomicU64 = AtomicU64::new(0);

/// Rows found only by walking every visible raw table because their locator named a
/// family that does not hold them.
///
/// The ladder is a correctness net for a real condition — a row delivered before the
/// catalogue knew its origin schema is placed in the CURRENT family while its locator keeps
/// the server-stamped origin hash — and it returns `needs_exact_locator` so the store can
/// correct itself. Nothing on the READ path acts on that, so a split row pays the walk on
/// every read, forever: production 2026-08-18 measured 1,200 recoveries in five minutes
/// against a single `users` row on an otherwise idle server, while settle passes accounted
/// for a fifth of the process's CPU.
///
/// A number rather than a log line, because "did the store heal" is a question about a
/// count converging to zero, and 1,200 identical INFO lines answer it only to whoever
/// thinks to count them.
pub static LOCATOR_LADDER_RECOVERIES: AtomicU64 = AtomicU64::new(0);

/// v18 item 5: the history twin of [`LOCATOR_LADDER_RECOVERIES`] — the history ladder's last
/// arm (`load_history_row_batch_row_bytes_with_storage`). Kept apart because the history walk
/// has no D2 hook: nothing heals it, and a count that never converges is the signal.
pub static HISTORY_LOCATOR_LADDER_RECOVERIES: AtomicU64 = AtomicU64::new(0);

/// v18 item 4: read statements that ran OUTSIDE a pass transaction on a store with
/// statement-level transaction cost (SQLite). With C1 a settle pass's reads run inside one
/// deferred transaction; this counts the ones that did not — direct `&self` reads, or a pass
/// whose `BEGIN` failed.
pub static AUTOCOMMIT_READS: AtomicU64 = AtomicU64::new(0);

/// v18 item 4: pass `BEGIN`s that failed (that pass ran autocommit; see [`AUTOCOMMIT_READS`]).
pub static READ_PASS_BEGIN_FAILURES: AtomicU64 = AtomicU64::new(0);

/// v18 item 8: explicit WAL checkpoints that DRAINED the log. Before item 8 the engine ran one
/// on every durability barrier, so a settle-heavy node paid a checkpoint per pass; this is how
/// the stand reads whether that stopped, rather than inferring it from latency.
///
/// It lives here, and not on `SqliteStorage`, because the shipped storage is a
/// `Box<dyn Storage>`: an inherent accessor on the concrete type is unreachable from the
/// runtime no matter how public it is (diff r25 B4).
pub static CHECKPOINTS: AtomicU64 = AtomicU64::new(0);

/// v18 item 8: checkpoints that ran and moved nothing because a reader still pinned the WAL.
///
/// SQLite reports this as SUCCESS — for a PASSIVE checkpoint it deliberately resets `SQLITE_BUSY`
/// to OK rather than "report a checkpoint failure just because there are active readers"
/// (`sqlite3.c:67453-67457`). So the only honest discriminator is the frame columns, and without
/// this counter every blocked attempt would be indistinguishable from a drained one.
pub static CHECKPOINTS_BLOCKED: AtomicU64 = AtomicU64::new(0);

/// v18 item 8: explicit checkpoints that returned an error. Never a barrier failure — the
/// commit already made the writes durable — but on a full or failing disk the checkpoint is
/// what fails FIRST, while commits keep appending to a WAL nothing is draining. This is the
/// number that says so.
pub static CHECKPOINT_FAILURES: AtomicU64 = AtomicU64::new(0);

/// The three checkpoint counters read together, for the one caller that must read them OUTSIDE
/// a settle pass.
///
/// The barrier that runs the checkpoint (`runtime_core/ticks.rs`, `flush_wal_barrier`) executes
/// AFTER `QueryManager::process` has closed its `SettlePass`, so the `checkpoints=` field on a
/// settle line is structurally always zero — measured on the stand, 175 passes, 0 every time,
/// while the policy underneath was working. A counter that cannot be non-zero where it is
/// printed is not an instrument, so the barrier reports its own.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CheckpointCounts {
    pub checkpoints: u64,
    pub blocked: u64,
    pub failures: u64,
}

impl CheckpointCounts {
    pub fn read() -> Self {
        Self {
            checkpoints: CHECKPOINTS.load(Ordering::Relaxed),
            blocked: CHECKPOINTS_BLOCKED.load(Ordering::Relaxed),
            failures: CHECKPOINT_FAILURES.load(Ordering::Relaxed),
        }
    }

    /// What happened between `base` and `self`. Saturating for the same reason the settle line
    /// is: these are process-global counters and another runtime in the same process may have
    /// advanced them, which must read as zero here rather than as an enormous number.
    pub fn since(self, base: Self) -> Self {
        Self {
            checkpoints: self.checkpoints.saturating_sub(base.checkpoints),
            blocked: self.blocked.saturating_sub(base.blocked),
            failures: self.failures.saturating_sub(base.failures),
        }
    }

    pub fn is_empty(self) -> bool {
        self == Self::default()
    }
}

/// History entries decoded — counted at `decode_history_row_bytes_in_table`,
/// the choke point every scan path funnels through. Divided by
/// [`HISTORY_SCANS`] it gives the mean depth the pass paid for.
pub static HISTORY_ENTRIES: AtomicU64 = AtomicU64::new(0);

/// Bytes of row payload decoded out of history. Separated from the entry count
/// because the two failure modes are different: many small revisions (heartbeat
/// depth) versus few enormous ones (blob rows), and they need different fixes.
pub static HISTORY_BYTES: AtomicU64 = AtomicU64::new(0);

/// Wall time spent inside storage READS and WRITES, and the bytes written.
///
/// Every other counter here answers "how much work", none answer "where did the
/// wall time go". A settle reporting 459 ms with `history_scans=0` says only
/// that one suspect is innocent; splitting its wall time into storage versus
/// everything else is the first fork that actually narrows anything, and it is
/// the fork that separates "the store got big" from "the engine recomputes too
/// much", which imply different fixes.
pub static STORAGE_READ_MICROS: AtomicU64 = AtomicU64::new(0);
/// Storage read CALLS. Without it, read time cannot be divided into "each read
/// is slow" and "there are far too many reads", and those have opposite fixes.
pub static STORAGE_READ_OPS: AtomicU64 = AtomicU64::new(0);
/// Bytes returned by storage reads. The history counters above cover only the
/// history table; a blob upload lands megabyte rows in the VISIBLE table, and
/// without this axis a settle that spends 37 of its 41 ms inside reads cannot
/// say whether it read thirty small rows or one enormous one.
pub static STORAGE_READ_BYTES: AtomicU64 = AtomicU64::new(0);
pub static STORAGE_WRITE_MICROS: AtomicU64 = AtomicU64::new(0);
pub static STORAGE_WRITE_BYTES: AtomicU64 = AtomicU64::new(0);

/// Run `f`, adding its wall time to `counter`.
#[inline]
pub(crate) fn timed<T>(counter: &AtomicU64, f: impl FnOnce() -> T) -> T {
    let started = Instant::now();
    let out = f();
    add(counter, micros_of(started.elapsed()));
    out
}

/// Entries in `QueryManager::pending_local_row_batches` at the moment a settle
/// took the overlay path — a GAUGE, reported as a value, never as a delta.
///
/// The map is global across tables and drains only when a non-local update for
/// the same object arrives confirmed at the global tier, so an upload whose rows
/// never get that echo leaves it pinned at the part count for the life of the
/// process. That is what makes a restart "fix" the freeze while the data on disk
/// is untouched.
pub static PENDING_LOCAL_ROW_BATCHES: AtomicU64 = AtomicU64::new(0);

/// Payloads queued to clients that never reached a connection, and how many distinct
/// clients are owed them. Gauges.
///
/// This is the early warning for silent delivery loss: a healthy server sits at zero, and
/// anything that stays non-zero is a peer that is missing rows right now. It is the number
/// that would have shown the 3-second-blip defect without a user noticing missing messages
/// first.
pub static UNDELIVERED_PAYLOADS: AtomicU64 = AtomicU64::new(0);
pub static UNDELIVERED_CLIENTS: AtomicU64 = AtomicU64::new(0);

/// Publish a gauge reading.
#[inline]
pub(crate) fn set_gauge(gauge: &AtomicU64, value: u64) {
    gauge.store(value, Ordering::Relaxed);
}

/// Count one event.
#[inline]
pub(crate) fn bump(counter: &AtomicU64) {
    counter.fetch_add(1, Ordering::Relaxed);
}

/// Give back one gauge membership counted with [`bump`].
#[inline]
pub(crate) fn unbump(gauge: &AtomicU64) {
    gauge.fetch_sub(1, Ordering::Relaxed);
}

/// RAII membership token for a gauge: one relaxed increment on construction,
/// one relaxed decrement on drop, nothing else.
///
/// Cached subgraph instances leave through four different paths — the LRU
/// eviction, the per-outer-row `retain`, replacement by a re-instantiation, and
/// plain drop of the map when the node or the whole graph goes away. A
/// hand-written decrement at each site would miss the last one, and a gauge
/// that only leaks upward is worse than no gauge, so membership is tied to the
/// entry's lifetime instead of to the call sites.
#[derive(Debug)]
pub(crate) struct LiveGauge(&'static AtomicU64);

impl LiveGauge {
    /// Join `gauge` until the returned token is dropped.
    #[inline]
    pub(crate) fn enter(gauge: &'static AtomicU64) -> Self {
        bump(gauge);
        Self(gauge)
    }
}

impl Drop for LiveGauge {
    #[inline]
    fn drop(&mut self) {
        unbump(self.0);
    }
}

/// Count `amount` events at once, for sites that already know the batch size.
#[inline]
pub(crate) fn add(counter: &AtomicU64, amount: u64) {
    if amount != 0 {
        counter.fetch_add(amount, Ordering::Relaxed);
    }
}

/// Identity of the most expensive single subscription settle in the open pass.
/// Reset by [`SettlePass::begin`]; only written when a subscription beats the
/// current maximum, so the common case is one relaxed load.
static HOT_MICROS: AtomicU64 = AtomicU64::new(0);
static HOT_CLIENT_HIGH: AtomicU64 = AtomicU64::new(0);
static HOT_CLIENT_LOW: AtomicU64 = AtomicU64::new(0);
static HOT_QUERY: AtomicU64 = AtomicU64::new(0);

/// Record how long one subscription's settle took, keeping the pass's maximum.
///
/// Called only from sites that have already decided to do real settle work, so
/// the two `Instant::now()` calls it costs sit next to a graph settle, never
/// next to a short-circuit.
pub fn note_subscription_settle(client_id: Option<ClientId>, query_id: u64, elapsed: Duration) {
    let micros = micros_of(elapsed);
    if micros <= HOT_MICROS.load(Ordering::Relaxed) {
        return;
    }
    let (high, low) = match client_id {
        Some(ClientId(uuid)) => {
            let bits = uuid.as_u128();
            ((bits >> 64) as u64, bits as u64)
        }
        None => (0, 0),
    };
    HOT_MICROS.store(micros, Ordering::Relaxed);
    HOT_CLIENT_HIGH.store(high, Ordering::Relaxed);
    HOT_CLIENT_LOW.store(low, Ordering::Relaxed);
    HOT_QUERY.store(query_id, Ordering::Relaxed);
}

fn micros_of(elapsed: Duration) -> u64 {
    u64::try_from(elapsed.as_micros()).unwrap_or(u64::MAX)
}

/// One reading of every settle counter.
///
/// Exposed so tests can assert the accounting itself: take a snapshot, run a
/// scenario of known shape, take another, and compare the difference against
/// what the scenario implies.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SettleCounts {
    pub subscriptions: u64,
    pub graph_nodes: u64,
    pub instance_evals: u64,
    pub subquery_instantiations: u64,
    pub plan_compiles: u64,
    pub shape_lowerings: u64,
    pub units_deferred: u64,
    pub ticks_rearmed: u64,
    pub policy_row_evals: u64,
    pub scope_authz_checks: u64,
    pub scope_authz_evals: u64,
    pub row_loads: u64,
    pub index_reads: u64,
    pub rows_emitted: u64,
    pub locator_ladder_recoveries: u64,
    pub history_locator_ladder_recoveries: u64,
    pub autocommit_reads: u64,
    pub read_pass_begin_failures: u64,
    pub checkpoints: u64,
    pub checkpoints_blocked: u64,
    pub checkpoint_failures: u64,
    pub history_scans: u64,
    pub history_entries: u64,
    pub history_bytes: u64,
    pub storage_read_micros: u64,
    pub storage_read_ops: u64,
    pub storage_read_bytes: u64,
    pub storage_write_micros: u64,
    pub storage_write_bytes: u64,
    pub pending_local_row_batches: u64,
    pub undelivered_payloads: u64,
    pub undelivered_clients: u64,
    /// Gauge, not a counter: live cached subgraph instances at the moment of
    /// the reading.
    pub live_instances: u64,
    /// Gauge, not a counter: live `ArraySubqueryNode`s at the moment of the
    /// reading.
    pub live_instance_nodes: u64,
}

impl SettleCounts {
    /// Read every counter. Counters are independent atomics, so a snapshot
    /// taken while a pass is running on another thread is not a consistent
    /// cut — see the module note on single-threaded passes.
    pub fn snapshot() -> Self {
        Self {
            subscriptions: SUBSCRIPTIONS_SETTLED.load(Ordering::Relaxed),
            graph_nodes: GRAPH_NODES_EVALUATED.load(Ordering::Relaxed),
            instance_evals: SUBQUERY_INSTANCE_EVALS.load(Ordering::Relaxed),
            subquery_instantiations: SUBQUERY_INSTANTIATIONS.load(Ordering::Relaxed),
            plan_compiles: PLAN_COMPILES.load(Ordering::Relaxed),
            shape_lowerings: SHAPE_LOWERINGS.load(Ordering::Relaxed),
            units_deferred: SETTLE_UNITS_DEFERRED.load(Ordering::Relaxed),
            ticks_rearmed: SETTLE_TICKS_REARMED.load(Ordering::Relaxed),
            policy_row_evals: POLICY_ROW_EVALS.load(Ordering::Relaxed),
            scope_authz_checks: SCOPE_AUTHZ_CHECKS.load(Ordering::Relaxed),
            scope_authz_evals: SCOPE_AUTHZ_EVALS.load(Ordering::Relaxed),
            row_loads: ROW_LOADS.load(Ordering::Relaxed),
            index_reads: INDEX_READS.load(Ordering::Relaxed),
            rows_emitted: ROWS_EMITTED.load(Ordering::Relaxed),
            locator_ladder_recoveries: LOCATOR_LADDER_RECOVERIES.load(Ordering::Relaxed),
            history_locator_ladder_recoveries: HISTORY_LOCATOR_LADDER_RECOVERIES
                .load(Ordering::Relaxed),
            autocommit_reads: AUTOCOMMIT_READS.load(Ordering::Relaxed),
            read_pass_begin_failures: READ_PASS_BEGIN_FAILURES.load(Ordering::Relaxed),
            checkpoints: CHECKPOINTS.load(Ordering::Relaxed),
            checkpoints_blocked: CHECKPOINTS_BLOCKED.load(Ordering::Relaxed),
            checkpoint_failures: CHECKPOINT_FAILURES.load(Ordering::Relaxed),
            history_scans: HISTORY_SCANS.load(Ordering::Relaxed),
            history_entries: HISTORY_ENTRIES.load(Ordering::Relaxed),
            history_bytes: HISTORY_BYTES.load(Ordering::Relaxed),
            storage_read_micros: STORAGE_READ_MICROS.load(Ordering::Relaxed),
            storage_read_ops: STORAGE_READ_OPS.load(Ordering::Relaxed),
            storage_read_bytes: STORAGE_READ_BYTES.load(Ordering::Relaxed),
            storage_write_micros: STORAGE_WRITE_MICROS.load(Ordering::Relaxed),
            storage_write_bytes: STORAGE_WRITE_BYTES.load(Ordering::Relaxed),
            pending_local_row_batches: PENDING_LOCAL_ROW_BATCHES.load(Ordering::Relaxed),
            undelivered_payloads: UNDELIVERED_PAYLOADS.load(Ordering::Relaxed),
            undelivered_clients: UNDELIVERED_CLIENTS.load(Ordering::Relaxed),
            live_instances: LIVE_SUBQUERY_INSTANCES.load(Ordering::Relaxed),
            live_instance_nodes: LIVE_SUBQUERY_NODES.load(Ordering::Relaxed),
        }
    }

    /// Work done since `base`, field by field — except for the gauges, which
    /// carry the LATER reading through unchanged. Subtracting them would report
    /// "instances created minus destroyed during the pass", which is a number
    /// nobody wants; what the pass needs to publish is how many instances
    /// existed while it ran.
    pub fn since(self, base: Self) -> Self {
        Self {
            subscriptions: self.subscriptions.saturating_sub(base.subscriptions),
            graph_nodes: self.graph_nodes.saturating_sub(base.graph_nodes),
            instance_evals: self.instance_evals.saturating_sub(base.instance_evals),
            subquery_instantiations: self
                .subquery_instantiations
                .saturating_sub(base.subquery_instantiations),
            plan_compiles: self.plan_compiles.saturating_sub(base.plan_compiles),
            shape_lowerings: self.shape_lowerings.saturating_sub(base.shape_lowerings),
            units_deferred: self.units_deferred.saturating_sub(base.units_deferred),
            ticks_rearmed: self.ticks_rearmed.saturating_sub(base.ticks_rearmed),
            policy_row_evals: self.policy_row_evals.saturating_sub(base.policy_row_evals),
            scope_authz_checks: self
                .scope_authz_checks
                .saturating_sub(base.scope_authz_checks),
            scope_authz_evals: self
                .scope_authz_evals
                .saturating_sub(base.scope_authz_evals),
            row_loads: self.row_loads.saturating_sub(base.row_loads),
            index_reads: self.index_reads.saturating_sub(base.index_reads),
            rows_emitted: self.rows_emitted.saturating_sub(base.rows_emitted),
            locator_ladder_recoveries: self
                .locator_ladder_recoveries
                .saturating_sub(base.locator_ladder_recoveries),
            history_locator_ladder_recoveries: self
                .history_locator_ladder_recoveries
                .saturating_sub(base.history_locator_ladder_recoveries),
            autocommit_reads: self.autocommit_reads.saturating_sub(base.autocommit_reads),
            read_pass_begin_failures: self
                .read_pass_begin_failures
                .saturating_sub(base.read_pass_begin_failures),
            checkpoints: self.checkpoints.saturating_sub(base.checkpoints),
            checkpoints_blocked: self
                .checkpoints_blocked
                .saturating_sub(base.checkpoints_blocked),
            checkpoint_failures: self
                .checkpoint_failures
                .saturating_sub(base.checkpoint_failures),
            history_scans: self.history_scans.saturating_sub(base.history_scans),
            history_entries: self.history_entries.saturating_sub(base.history_entries),
            history_bytes: self.history_bytes.saturating_sub(base.history_bytes),
            storage_read_micros: self
                .storage_read_micros
                .saturating_sub(base.storage_read_micros),
            storage_read_ops: self.storage_read_ops.saturating_sub(base.storage_read_ops),
            storage_read_bytes: self
                .storage_read_bytes
                .saturating_sub(base.storage_read_bytes),
            storage_write_micros: self
                .storage_write_micros
                .saturating_sub(base.storage_write_micros),
            storage_write_bytes: self
                .storage_write_bytes
                .saturating_sub(base.storage_write_bytes),
            pending_local_row_batches: self.pending_local_row_batches,
            undelivered_payloads: self.undelivered_payloads,
            undelivered_clients: self.undelivered_clients,
            live_instances: self.live_instances,
            live_instance_nodes: self.live_instance_nodes,
        }
    }
}

/// Wall-duration threshold above which a pass logs its cost record.
fn threshold_micros() -> u64 {
    #[cfg(any(test, feature = "test"))]
    if let Some(forced) = test_override::forced_ms() {
        return forced.saturating_mul(1_000);
    }

    static THRESHOLD: OnceLock<u64> = OnceLock::new();
    *THRESHOLD.get_or_init(|| {
        std::env::var("JAZZ_SETTLE_LOG_MS")
            .ok()
            .and_then(|raw| raw.trim().parse::<u64>().ok())
            .unwrap_or(DEFAULT_THRESHOLD_MS)
            .saturating_mul(1_000)
    })
}

/// Scoped accumulator for one settle pass. Emits at most one line on drop.
pub struct SettlePass {
    started: Instant,
    base: SettleCounts,
}

impl SettlePass {
    /// Open a pass: reset the hot-subscription slot and snapshot the counters.
    pub fn begin() -> Self {
        HOT_MICROS.store(0, Ordering::Relaxed);
        Self {
            started: Instant::now(),
            base: SettleCounts::snapshot(),
        }
    }

    /// Work recorded so far in this pass.
    pub fn cost(&self) -> SettleCounts {
        SettleCounts::snapshot().since(self.base)
    }
}

impl Drop for SettlePass {
    fn drop(&mut self) {
        let micros = micros_of(self.started.elapsed());
        if micros < threshold_micros() {
            return;
        }

        let cost = self.cost();
        let hot_micros = HOT_MICROS.load(Ordering::Relaxed);
        let hot_client = u128::from(HOT_CLIENT_HIGH.load(Ordering::Relaxed)) << 64
            | u128::from(HOT_CLIENT_LOW.load(Ordering::Relaxed));
        let hot_client = if hot_client == 0 {
            "local".to_string()
        } else {
            uuid::Uuid::from_u128(hot_client).to_string()
        };

        tracing::info!(
            target: "jazz::settle_cost",
            micros,
            subscriptions = cost.subscriptions,
            graph_nodes = cost.graph_nodes,
            instance_evals = cost.instance_evals,
            subquery_instantiations = cost.subquery_instantiations,
            plan_compiles = cost.plan_compiles,
            shape_lowerings = cost.shape_lowerings,
            units_deferred = cost.units_deferred,
            ticks_rearmed = cost.ticks_rearmed,
            policy_row_evals = cost.policy_row_evals,
            scope_authz_checks = cost.scope_authz_checks,
            scope_authz_evals = cost.scope_authz_evals,
            row_loads = cost.row_loads,
            index_reads = cost.index_reads,
            rows_emitted = cost.rows_emitted,
            locator_ladder_recoveries = cost.locator_ladder_recoveries,
            history_locator_ladder_recoveries = cost.history_locator_ladder_recoveries,
            autocommit_reads = cost.autocommit_reads,
            read_pass_begin_failures = cost.read_pass_begin_failures,
            checkpoints = cost.checkpoints,
            checkpoints_blocked = cost.checkpoints_blocked,
            checkpoint_failures = cost.checkpoint_failures,
            history_scans = cost.history_scans,
            history_entries = cost.history_entries,
            history_bytes = cost.history_bytes,
            storage_read_micros = cost.storage_read_micros,
            storage_read_ops = cost.storage_read_ops,
            storage_read_bytes = cost.storage_read_bytes,
            storage_write_micros = cost.storage_write_micros,
            storage_write_bytes = cost.storage_write_bytes,
            pending_local_row_batches = cost.pending_local_row_batches,
            undelivered_payloads = cost.undelivered_payloads,
            undelivered_clients = cost.undelivered_clients,
            live_instances = cost.live_instances,
            live_instance_nodes = cost.live_instance_nodes,
            hot_micros,
            hot_client,
            hot_query = HOT_QUERY.load(Ordering::Relaxed),
            "jazz settle pass cost"
        );
    }
}

#[cfg(any(test, feature = "test"))]
mod test_override {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};

    const UNSET: u64 = u64::MAX;

    static FORCED_MS: AtomicU64 = AtomicU64::new(UNSET);

    fn lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    /// Holds the settle-cost log threshold for the guard's lifetime.
    ///
    /// The embedded mutex guard serialises every test that forces a threshold,
    /// the same plug shape as `precise_dirty::force_precise_dirty`, so
    /// force-users cannot observe each other's override and no test has to
    /// mutate the process environment.
    pub struct SettleLogThreshold {
        _serialised: MutexGuard<'static, ()>,
    }

    impl Drop for SettleLogThreshold {
        fn drop(&mut self) {
            FORCED_MS.store(UNSET, Ordering::SeqCst);
        }
    }

    /// Force the settle-cost log threshold to `milliseconds` (0 logs every
    /// pass) for the returned guard's lifetime.
    pub fn force_settle_log_ms(milliseconds: u64) -> SettleLogThreshold {
        let guard = lock().lock().unwrap_or_else(PoisonError::into_inner);
        FORCED_MS.store(milliseconds, Ordering::SeqCst);
        SettleLogThreshold { _serialised: guard }
    }

    pub(super) fn forced_ms() -> Option<u64> {
        match FORCED_MS.load(Ordering::SeqCst) {
            UNSET => None,
            milliseconds => Some(milliseconds),
        }
    }
}

#[cfg(any(test, feature = "test"))]
pub use test_override::{SettleLogThreshold, force_settle_log_ms};
