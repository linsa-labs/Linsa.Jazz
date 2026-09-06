//! v18 item 4 (C): a settle pass over `SqliteStorage` runs inside ONE read transaction.
//!
//! `SqliteStorage` opens a transaction lazily on the first WRITE of a tick and commits it
//! from the durability barrier. A pass that only reads — every `immediate_tick` under a
//! one-shot `query()`, which is the backend read path — therefore runs each `SELECT` as its
//! own autocommit statement: SQLite begins and ends a read transaction per statement
//! (wal-index header read, shared-memory lock, page validation). Measured on the rpc-server
//! engine (REPRO.md, run n2): row loads are 89 % of that tick, `sqlite3_step` 56–61 % of it,
//! `__fcntl` 10 % of the leaves.
//!
//! Internal on purpose: the observable is whether a statement ran in autocommit mode, which
//! only the connection knows (`SqliteStorage::autocommit_read_statements_for_test`,
//! per instance so a parallel test cannot pollute the delta). `SqliteStorage` only:
//! `MemoryStorage` has no transactions and `RocksDbStorage` no statement-level cost.

use super::support::{docs_v2, read_on, split_store};
use super::*;
use crate::query_manager::manager::LocalUpdates;
use crate::storage::SqliteStorage;
use crate::sync_manager::QueryPropagation;

/// A core over an already-open store with a scheduler whose calls are counted. The gates
/// below need both halves: the store must carry the split fixture's state (so it is moved,
/// not reopened), and the scheduler must be observable (so a pass that schedules its own
/// barrier can be told from one that relies on an external tick). `split_store` hands back a
/// `NoopScheduler` core, so the rebuild is over `core.into_storage()` — design v14 § G-C11.
fn counting_core_over(
    storage: SqliteStorage,
    app_name: &str,
    scheduler: CountingScheduler,
) -> RuntimeCore<SqliteStorage, CountingScheduler> {
    let app_id = AppId::from_name(app_name);
    let mut schema_manager =
        SchemaManager::new(SyncManager::new(), docs_v2(), app_id, "dev", "main").unwrap();
    crate::schema_manager::rehydrate_schema_manager_from_catalogue(
        &mut schema_manager,
        &storage,
        app_id,
    )
    .expect("rehydrate from the persisted catalogue");
    let mut core = new_test_core(schema_manager, storage, scheduler);
    core.immediate_tick();
    core
}

/// Run `batched_tick` while the core keeps asking for another one, at most `cap` times.
/// Returns the number of ticks run. The cap is the gate's own liveness bound: a core that
/// re-arms forever fails the assertion after it, not by hanging the suite.
fn drain_scheduled_ticks(
    core: &mut RuntimeCore<SqliteStorage, CountingScheduler>,
    scheduler: &CountingScheduler,
    cap: usize,
) -> usize {
    let mut ticks = 0;
    let mut seen = scheduler.schedule_count();
    while ticks < cap {
        core.batched_tick();
        ticks += 1;
        let now = scheduler.schedule_count();
        if now == seen {
            break;
        }
        seen = now;
    }
    ticks
}

fn docs_schema() -> Schema {
    SchemaBuilder::new()
        .table(
            TableSchema::builder("docs")
                .column("owner", ColumnType::Text)
                .column("body", ColumnType::Text),
        )
        .build()
}

fn core_over(path: &std::path::Path) -> RuntimeCore<SqliteStorage, NoopScheduler> {
    let storage = SqliteStorage::open(path).expect("sqlite store opens");
    let app_id = AppId::from_name("read-pass");
    let schema_manager =
        SchemaManager::new(SyncManager::new(), docs_schema(), app_id, "dev", "main").unwrap();
    let mut core = new_test_core(schema_manager, storage, NoopScheduler);
    core.immediate_tick();
    core
}

fn seed_docs(core: &mut RuntimeCore<SqliteStorage, NoopScheduler>, rows: usize) {
    let alice = WriteContext::from_session(Session::new("alice"));
    for index in 0..rows {
        // The settle receiver is dropped deliberately: seeding only needs the row applied,
        // and the three ticks below are what actually settle it. Bound rather than left to
        // `unused_must_use` (diff r23 SF8) so the next fixture copied from this one does not
        // inherit a warning as if it were the house style.
        let (_seeded, _settled) = insert_and_wait_for_batch(
            core,
            "docs",
            HashMap::from([
                ("owner".to_string(), Value::Text("alice".to_string())),
                ("body".to_string(), Value::Text(format!("doc-{index}"))),
            ]),
            Some(&alice),
            DurabilityTier::Local,
        )
        .expect("the seed row inserts");
    }
    // The barrier commits the seeding transaction; the tree is clean afterwards.
    core.batched_tick();
    core.immediate_tick();
    core.batched_tick();
}

/// One local one-shot read of every doc: the pass is `immediate_tick` under `query()`.
fn read_docs(core: &mut RuntimeCore<SqliteStorage, NoopScheduler>) {
    let (_handle, _future) = core
        .query_with_local_batch_tracked(
            Query::new("docs"),
            None,
            ReadDurabilityOptions {
                tier: None,
                local_updates: LocalUpdates::Immediate,
            },
            QueryPropagation::LocalOnly,
            None,
        )
        .expect("query setup");
}

/// G-C1. Red on the base tree: every read of the pass is its own autocommit statement.
#[test]
fn a_read_only_pass_runs_inside_one_transaction() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let mut core = core_over(&dir.path().join("pass.sqlite"));
    seed_docs(&mut core, 200);
    assert!(
        core.storage().is_autocommit_for_test(),
        "fixture: no transaction may be open before the read-only pass"
    );

    let reads_before = core.storage().read_statements_for_test();
    let autocommit_before = core.storage().autocommit_read_statements_for_test();
    read_docs(&mut core);
    let reads = core.storage().read_statements_for_test() - reads_before;
    let autocommit = core.storage().autocommit_read_statements_for_test() - autocommit_before;

    assert!(
        reads >= 200,
        "fixture: the pass must read the seeded rows through storage (read {reads} statements)"
    );
    assert_eq!(
        autocommit, 0,
        "a settle pass must run its {reads} reads inside one transaction; {autocommit} of \
         them ran as their own autocommit statement, each paying SQLite's per-statement \
         transaction start and end"
    );
}

/// G-C3. A read transaction left open pins the WAL: PASSIVE checkpoints can never complete
/// past the reader and the WAL grows without bound. Green on the base tree (nothing opens a
/// transaction for reads); red when the end of the pass is disarmed.
#[test]
fn a_read_only_pass_leaves_no_transaction_open() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let mut core = core_over(&dir.path().join("pass.sqlite"));
    seed_docs(&mut core, 20);
    read_docs(&mut core);
    assert!(
        core.storage().is_autocommit_for_test(),
        "a read-only pass must close its transaction when it ends; a reader left open pins \
         the WAL for every checkpoint that follows"
    );
    assert!(
        !core.has_storage_write_pending_flush(),
        "a pass that wrote nothing must not be flagged for the durability barrier"
    );
}

/// G-C2. The durability barrier owns the write transaction: a pass that wrote must leave it
/// open for the barrier, and the barrier must commit it. Guards C1 against closing a dirty
/// transaction early or leaving it open past the barrier.
#[test]
fn a_pass_that_writes_leaves_its_transaction_for_the_durability_barrier() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let path = dir.path().join("pass.sqlite");
    let mut core = core_over(&path);
    let alice = WriteContext::from_session(Session::new("alice"));
    let ((row_id, _), _) = insert_and_wait_for_batch(
        &mut core,
        "docs",
        HashMap::from([
            ("owner".to_string(), Value::Text("alice".to_string())),
            ("body".to_string(), Value::Text("durable".to_string())),
        ]),
        Some(&alice),
        DurabilityTier::Local,
    )
    .expect("the row inserts");
    core.immediate_tick();
    assert!(
        !core.storage().is_autocommit_for_test(),
        "a pass that wrote must leave its transaction open for the durability barrier"
    );
    assert!(
        core.has_storage_write_pending_flush(),
        "a pass that wrote must be flagged for the durability barrier"
    );
    core.batched_tick();
    assert!(
        core.storage().is_autocommit_for_test(),
        "the durability barrier must commit the pass's transaction"
    );
    let branch = crate::storage::sole_branch_name(core.storage())
        .expect("branch registry readable")
        .expect("the seeded row registered a branch");
    drop(core);
    let reopened = SqliteStorage::open(&path).expect("reopen");
    assert!(
        reopened
            .load_visible_region_row("docs", branch.as_str(), row_id)
            .expect("visible row readable")
            .is_some(),
        "the committed row must be readable through a fresh connection"
    );
}

/// G-C4. The C×D crossing: a read-only pass whose ladder recovers a locator has WRITTEN, so
/// it must leave its transaction for the durability barrier AND ask for that barrier itself.
/// Before this item the recovery did not exist; a pass that wrote nothing scheduled nothing,
/// and a locator persisted by a read would have sat in an uncommitted transaction until some
/// unrelated write happened to arrive.
///
/// Internal on purpose: the observables are the store's transaction state and the scheduler's
/// call count, neither of which any public API reports.
#[test]
fn a_locator_recovered_during_a_read_only_pass_reaches_the_barrier() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let path = dir.path().join("c4.sqlite");
    let (core, row_id, branch_a) = split_store(&path, "locator-barrier");
    let storage = core.into_storage();
    let scheduler = CountingScheduler::default();
    let mut core = counting_core_over(storage, "locator-barrier", scheduler.clone());
    let recoveries_before = core.storage().visible_ladder_recoveries_for_test();
    let scheduled_before = scheduler.schedule_count();
    read_on(&mut core, &branch_a);
    assert!(
        core.storage().visible_ladder_recoveries_for_test() > recoveries_before,
        "fixture precondition: the read must walk the ladder, else the split was not reproduced"
    );
    assert!(
        core.has_storage_write_pending_flush(),
        "a pass whose ladder persisted a locator has written: it must be flagged for the \
         durability barrier"
    );
    assert!(
        scheduler.schedule_count() > scheduled_before,
        "the recovering pass must ask for its own barrier; nothing else is going to"
    );
    core.batched_tick();
    assert!(
        core.storage().is_autocommit_for_test(),
        "the barrier must commit the transaction the recovering pass left open"
    );
    drop(core);
    let reopened = SqliteStorage::open(&path).expect("reopen");
    assert!(
        reopened
            .load_visible_row_table_locator(branch_a.as_str(), row_id)
            .expect("locator readable")
            .is_some(),
        "the locator the pass recovered must survive the process that recovered it"
    );
}

/// G-C4'. The same crossing, driven by nobody: with no external ticks beyond the ones the
/// core asks for, a recovering pass reaches quiescence — no pending flush, no open
/// transaction. This is the liveness half of G-C4, which only shows that ONE barrier was
/// requested.
///
/// Internal on purpose: quiescence is a statement about the core's own bookkeeping and the
/// store's connection, both private.
#[test]
fn a_recovering_pass_reaches_quiescence_without_external_ticks() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let path = dir.path().join("c4prime.sqlite");
    let (core, _row_id, branch_a) = split_store(&path, "locator-quiescence");
    let storage = core.into_storage();
    let scheduler = CountingScheduler::default();
    let mut core = counting_core_over(storage, "locator-quiescence", scheduler.clone());
    read_on(&mut core, &branch_a);
    let ticks = drain_scheduled_ticks(&mut core, &scheduler, 8);
    assert!(
        ticks < 8,
        "the core must stop asking for batched ticks; it asked for at least {ticks}"
    );
    assert!(
        !core.has_storage_write_pending_flush(),
        "quiescence means the barrier ran: nothing may still be pending"
    );
    assert!(
        core.storage().is_autocommit_for_test(),
        "quiescence means no transaction is left open"
    );
}

/// G-C5. `impl<T: Storage + ?Sized> Storage for Box<T>` forwards four methods this item adds
/// (`begin_read_pass`, `end_read_pass`, `record_visible_row_table_locator_recovery`,
/// `note_visible_locator_recovery`). A dropped forward is silent: the pass degrades to
/// autocommit reads, or the ladder walks for the life of the row. This gate is the pin for
/// all four.
///
/// `Box<SqliteStorage>`, not `Box<dyn Storage>`: the assertions read the store's own
/// `_for_test` accessors, which a trait object hides (design v23 § retraction of v16's "over
/// `Box<dyn Storage + Send>`"). The blanket impl under test is the same one either way.
///
/// Internal on purpose: every observable here — autocommit statements, pass depth, the
/// per-store ladder count — belongs to the store, and the point of the gate is that they are
/// reached THROUGH the Box.
#[test]
fn boxing_the_storage_keeps_the_pass_and_the_locator_forwards() {
    // Falsification measured (chain rows E8–E11): clauses 1 and 2 gate the Box's OWN forwards
    // of `begin_read_pass`/`end_read_pass` — drop either and this test reds. Clauses 3 and 4
    // do NOT gate the Box's forwards of the two recovery methods, and no test can: every
    // concrete store overrides `load_visible_region_row_bytes`, so the hook always runs with
    // the concrete store and the Box is peeled before it (see `storage_trait.rs`'s note on
    // those two forwards). What clauses 3 and 4 gate is the D2 hook end-to-end with the core
    // held over a boxed store, which is what E10/E11 disarm.
    let dir = tempfile::TempDir::new().expect("temp dir");
    let path = dir.path().join("c5.sqlite");
    let (core, _row_id, branch_a) = split_store(&path, "boxed-pass");
    let storage = Box::new(core.into_storage());
    let app_id = AppId::from_name("boxed-pass");
    let mut schema_manager =
        SchemaManager::new(SyncManager::new(), docs_v2(), app_id, "dev", "main").unwrap();
    crate::schema_manager::rehydrate_schema_manager_from_catalogue(
        &mut schema_manager,
        storage.as_ref(),
        app_id,
    )
    .expect("rehydrate from the persisted catalogue");
    let mut core = new_test_core(schema_manager, storage, NoopScheduler);
    core.immediate_tick();
    core.batched_tick();
    let autocommit_before = core.storage().autocommit_read_statements_for_test();
    let recoveries_before = core.storage().visible_ladder_recoveries_for_test();
    read_on(&mut core, &branch_a);
    let autocommit = core.storage().autocommit_read_statements_for_test() - autocommit_before;
    assert_eq!(
        autocommit, 0,
        "the pass must reach the boxed store: {autocommit} reads ran in autocommit, which is \
         what a dropped `begin_read_pass` forward looks like"
    );
    assert_eq!(
        core.storage().pass_depth_for_test(),
        0,
        "every begin must be matched by an end through the Box"
    );
    assert_eq!(
        core.storage().visible_ladder_recoveries_for_test() - recoveries_before,
        1,
        "the ladder must walk exactly once with the core over a boxed store"
    );
    core.batched_tick();
    assert!(
        core.storage().is_autocommit_for_test(),
        "a pass that recovered a locator through the Box must leave a committable transaction"
    );
    let recoveries_after_first = core.storage().visible_ladder_recoveries_for_test();
    read_on(&mut core, &branch_a);
    assert_eq!(
        core.storage().visible_ladder_recoveries_for_test(),
        recoveries_after_first,
        "the second read must answer from the persisted pointer: a D2 hook that does not \
         persist walks the families again"
    );
}

/// G-C11. A loss discovered BY A READ PASS reaches the barrier. The ladder's own persist is
/// the write that arms the store's rollback hook, so the loss is discovered at the pass end —
/// a site that, before this item, had no way to report anything: a read pass could not fail.
///
/// The tick that discovers it must schedule exactly one barrier (and no more: an unsplit
/// latch would re-arm on every tick forever), the barrier must report the loss once, and a
/// later tick with nothing outbound must schedule nothing.
///
/// Internal on purpose: the hook, the flags and the scheduler count are all internal, and the
/// mechanism cannot be reached from any public API — SQLite ends the transaction behind the
/// store's back only under NOMEM/IOERR/INTERRUPT/FULL.
#[test]
fn a_loss_discovered_by_a_read_pass_reaches_the_barrier() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let path = dir.path().join("c11.sqlite");
    let (core, _row_id, branch_a) = split_store(&path, "loss-in-pass");
    let storage = core.into_storage();
    let scheduler = CountingScheduler::default();
    let mut core = counting_core_over(storage, "loss-in-pass", scheduler.clone());
    core.storage().roll_back_after_next_write_for_test();
    let scheduled_before = scheduler.schedule_count();
    read_on(&mut core, &branch_a);
    let scheduled_by_the_pass = scheduler.schedule_count() - scheduled_before;
    assert!(
        !core.schema_manager.query_manager().settle_work_remains(),
        "the confounder is pinned: no settle work may remain, or the schedule below could be \
         item 6's re-arm rather than the storage error's"
    );
    assert_eq!(
        scheduled_by_the_pass, 1,
        "the tick that discovers the loss must ask for exactly one barrier"
    );
    assert!(
        core.has_storage_flush_error(),
        "the pass end must record the loss for the barrier to report"
    );
    assert_eq!(
        core.storage().pass_depth_for_test(),
        0,
        "the pass must balance its depth even on the loss path: the boundary's bookkeeping \
         runs before the reconcile that reports (design v15 row 4, diff r20 B1)"
    );
    // diff r22 B3: `recovery_persist_failures` and `read_pass_begin_failures` are bumped by
    // the fix and were read by nothing — instrumentation with no gate is a number nobody can
    // trust. The design named `== 2` for this gate; MEASURED it is 1, and 1 is right: the
    // store is armed to roll back after ONE write, and the D2 hook's locator put is that
    // write. The begin count pins the confounder — the pass began cleanly, so the `Err` the
    // barrier reports came from the persist, not from a transaction that never opened.
    assert_eq!(
        core.storage().recovery_persist_failures_for_test(),
        1,
        "the rolled-back write must be counted as exactly one recovery-persist failure"
    );
    assert_eq!(
        core.storage().read_pass_begin_failures_for_test(),
        0,
        "the pass itself must have begun cleanly, or this gate is measuring the wrong failure"
    );
    core.batched_tick();
    assert!(
        matches!(
            core.take_storage_flush_error(),
            Some(StorageError::LostWrites { .. })
        ),
        "the barrier must hand the host a LostWrites, not a generic failure"
    );
    assert!(
        core.lost_writes_barrier_reported_for_test(),
        "the barrier must latch that it reported the loss"
    );
    assert_eq!(
        core.storage().pass_depth_for_test(),
        0,
        "the barrier's own tick must leave the depth balanced too"
    );
    let before_tail = scheduler.schedule_count();
    core.immediate_tick();
    assert_eq!(
        core.storage().pass_depth_for_test(),
        0,
        "and the tail tick, on a store that reports Err from every boundary"
    );
    assert_eq!(
        scheduler.schedule_count(),
        before_tail,
        "with the loss latched and nothing outbound, a later tick must schedule nothing: a \
         store that cannot persist must not spin the host"
    );
}

/// G-C12. A store that has reported its loss must stop arming the durability barrier — and
/// must NOT stop sync. `ticks.rs`'s disjunct is
/// `has_outbound() || (storage_write_pending_flush && !lost_writes_barrier_reported)`, and both
/// halves matter: drop the guard and a dead store re-arms the barrier on every tick forever
/// (the spin item 4 exists to remove); widen the guard to the whole disjunct and a dead store
/// stops sending, which loses the outbound messages the user can still recover by other means.
///
/// Clause 1 — outbound present, flag 3 set: the tick schedules.
/// Clause 2 — outbound drained, flag 3 set, barrier still pending: the tick does NOT schedule.
///
/// Clause 2 is r10 Blocking 2's direct falsification: it is red under the ungated v11 disjunct.
///
/// Internal on purpose: `lost_writes_barrier_reported_for_test` is the core's own latch,
/// `take_outbox` is the sync manager's private queue, and the observable is a count of
/// scheduler calls. None of the three is reachable through a client API — a client sees a
/// storage error, not the decision the tick made about re-arming.
#[test]
fn a_latched_loss_does_not_stop_outbound_sync() {
    let scheduler = CountingScheduler::default();
    let app_id = AppId::from_name("latched-loss-outbound");
    let schema_manager =
        SchemaManager::new(SyncManager::new(), test_schema(), app_id, "dev", "main").unwrap();
    let storage = MemoryStorage::new().with_flush_wal_error(StorageError::LostWrites {
        detail: "the write transaction was ended behind the store's back".to_string(),
    });
    let mut core = new_test_core(schema_manager, storage, scheduler.clone());
    // A write arms the barrier; the tick runs it, it fails with the loss, and flag 3 latches.
    core.insert("users", user_insert_values(ObjectId::new(), "Alice"), None)
        .unwrap();
    core.batched_tick();
    assert!(
        core.lost_writes_barrier_reported_for_test(),
        "fixture: the barrier must have run and reported the loss before either clause"
    );
    assert!(
        core.has_storage_write_pending_flush(),
        "fixture: the barrier stays pending on a dead store — that is what makes clause 2 a \
         real question rather than a vacuous one"
    );
    // Clause 1: with something to send, the tick must still schedule. A local insert alone
    // leaves the outbox EMPTY (that is what
    // `rc_local_write_without_outbox_still_schedules_batched_tick_for_flush` is named for), so
    // the outbound traffic has to come from a real destination: `add_server` queues a full sync.
    core.add_server(ServerId::new());
    assert!(
        core.has_outbound(),
        "fixture: registering a server must queue a full sync — without outbound traffic \
         clause 1 cannot distinguish the two arms of the disjunct"
    );
    let before = scheduler.schedule_count();
    core.immediate_tick();
    assert!(
        scheduler.schedule_count() > before,
        "a latched loss must not stop sync: the store is dead, the outbound messages are not"
    );
    // Clause 2: with the outbox drained, the pending barrier alone must not re-arm.
    core.schema_manager_mut()
        .query_manager_mut()
        .sync_manager_mut()
        .take_outbox();
    assert!(
        !core.has_outbound(),
        "fixture: the outbox must be empty for clause 2 to test the storage arm alone"
    );
    let before = scheduler.schedule_count();
    core.immediate_tick();
    assert_eq!(
        scheduler.schedule_count(),
        before,
        "a barrier that can never succeed must not be re-armed: the retry costs a tick under \
         the core lock and buys nothing until the store is reopened"
    );
}

/// G-C6. The parked-message exit ends the pass and defers the barrier to the NEXT tick.
///
/// `batched_tick` has two exits. The one every other gate here takes falls through to the
/// durability barrier. The other — taken when the tick drained a parked message and query
/// subscriptions are still pending — ends the read pass and returns before the barrier: it is
/// the `if drained_any && has_pending_query_subscriptions()` early return in
/// `batched_tick_inner`. (Cited structurally on purpose: diff r23 SF4 caught these as line
/// numbers that had rotted by ten lines and pointed a reader into the OTHER branch — the one
/// this gate exists to distinguish itself from.) That asymmetry is deliberate (more inbound work
/// is queued; committing now would pay a WAL checkpoint per parked message) and it is the
/// one place where "the pass ended" and "the barrier ran" come apart. If `end_read_pass_in_tick`
/// were ever dropped from this arm, the pass would leak depth and every later boundary in the
/// process would be a no-op — silently, because a leaked depth reads exactly like a pass that
/// is still legitimately open.
///
/// Internal on purpose: the claim is about which of `batched_tick`'s two exits ran and what
/// each left behind — pass depth, autocommit, the pending-flush flag, the scheduler count.
/// None of that is visible from a query result; a public-API gate could only observe that the
/// data eventually arrives, which is true down both exits. `mark_storage_write_pending_flush`
/// stands in for a storage write: it is the crate's own setter and sets exactly the state a
/// write leaves, without needing a server-mode schema this fixture cannot have (the
/// subscription is deferred precisely BECAUSE the schema is empty).
///
/// The deferred-subscription fixture is `batched_tick_parked_drain.rs`'s
/// `server_with_deferred_subscription` re-expressed over `SqliteStorage`: that one is typed
/// to `MemoryStorage`, which has no transactions and so cannot answer this gate's question.
/// It is copied rather than made generic so the existing tests keep running on the store they
/// were written against.
#[test]
fn the_parked_message_exit_ends_the_pass_and_defers_the_barrier() {
    use crate::query_manager::query::Query;
    use crate::sync_manager::QueryId;
    let dir = tempfile::tempdir().expect("tempdir");
    let storage = SqliteStorage::open(&dir.path().join("parked.sqlite")).expect("sqlite opens");
    let scheduler = CountingScheduler::default();
    let sync_manager = SyncManager::new().with_durability_tier(DurabilityTier::Local);
    let manager = SchemaManager::new_server(sync_manager, AppId::from_name("gc6-parked"), "dev");
    let mut core = new_test_core(manager, storage, scheduler.clone());
    let client_id = ClientId::new();
    core.add_client(client_id, None);
    core.push_sync_inbox(InboxEntry {
        source: Source::Client(client_id),
        payload: SyncPayload::QuerySubscription {
            query_id: QueryId(1),
            query: Box::new(Query {
                branches: vec!["main".to_string()],
                ..Query::new("users")
            }),
            session: None,
            required_tier: None,
            propagation: QueryPropagation::Full,
            policy_context_tables: Vec::new(),
        },
    });
    core.immediate_tick();
    assert!(
        core.schema_manager()
            .query_manager()
            .sync_manager()
            .has_pending_query_subscriptions(),
        "fixture precondition: the subscription must be deferred, or the tick below takes the \
         ordinary exit and this gate measures nothing"
    );
    // Something to drain, and a write for the barrier to owe.
    core.park_sync_message(InboxEntry {
        source: Source::Server(ServerId::new()),
        payload: SyncPayload::BatchFateNeeded {
            batch_ids: Vec::new(),
        },
    });
    core.mark_storage_write_pending_flush();
    // Fixture precondition, both sides of the window (diff r24 B3). `rearms == 3` is a pin on a
    // SUM, and two of its three contributors are nested `immediate_tick` tails that belong to
    // items 4/5. There are two more re-arm sites in `batched_tick` — the 3b continuation and
    // `rearm_for_deferred_settle_work` — and both are gated on `settle_work_remains()`, which is
    // item 6's surface. r24 measured that forcing it true takes this gate to `left: 5`, with a
    // message accusing a control that is not broken; the cheapest reading of that red would be
    // "the pin is stale, delete it". So the gate says out loud which world it is counting in: a
    // future item-6 change fails HERE, on a fixture line, instead of there, on the subject.
    assert!(
        !core.schema_manager.query_manager().settle_work_remains(),
        "fixture precondition: this gate counts re-arms in a tick with NO deferred settle work. \
         Two further re-arm sites are gated on `settle_work_remains()`, and with it true the \
         count below is 5, not 3"
    );
    let baseline = scheduler.schedule_count();
    core.batched_tick();
    assert_eq!(
        core.parked_sync_messages.len(),
        0,
        "fixture precondition: the tick must have drained the parked message, or `drained_any` \
         is false and the early return was never entered"
    );
    assert!(
        core.has_storage_write_pending_flush(),
        "the parked-message exit must NOT run the barrier: it returns above the flush, because more inbound work is queued and a commit here would pay a WAL \
         checkpoint per parked message"
    );
    // diff r23 B1: this was `schedule_count() > baseline`, and that assertion was VACUOUS —
    // measured, by deleting the exit's own `schedule_batched_tick()` and watching the gate stay
    // green. Two other re-arms land in this window before the exit is reached, so `>` was
    // already satisfied and could not tell three from two. The delta can.
    //
    // The decomposition is MEASURED, not reasoned: each of the three sites was disarmed in turn
    // and the count dropped to 2 in every case, so all three are real and nothing else
    // contributes.
    //
    //   1. `batched_tick_inner`'s deferred-subscription nested `immediate_tick`, whose tail
    //      schedules because the fixture set `storage_write_pending_flush` above the baseline;
    //   2. the nested `immediate_tick` `handle_sync_messages` runs for the message it applied,
    //      for the same reason;
    //   3. the parked exit's own — and only this one is the gate's subject.
    //
    // Pinning the sum couples the gate to 1 and 2, which is the price of measuring 3 at all:
    // there is no second baseline to take inside a single tick.
    //
    // Two further sites exist and are NOT in this count: `batched_tick_inner`'s 3b continuation
    // and `rearm_for_deferred_settle_work`. Both are gated on `settle_work_remains()`, which the
    // fixture preconditions above and below pin false — measured by r24, which forced it true
    // and took this gate to `left: 5`. The chain row that deletes the
    // exit's re-arm (E22) is what keeps the pin honest — without it this is again an assertion
    // nobody has falsified.
    //
    // The design predicted 2 here. It was wrong, the way it was wrong about `== 2` for
    // `recovery_persist_failures` last round; the number below is the one the run produced.
    //
    // This is the clause the whole deferral rests on. The barrier is skipped here only because
    // a tick is guaranteed to come back and pay it; if that guarantee is not gated, the
    // asymmetry above is an unbacked claim rather than a design.
    assert!(
        !core.schema_manager.query_manager().settle_work_remains(),
        "fixture precondition, after the tick: no deferred settle work was created during it \
         either, so the two `settle_work_remains()`-gated re-arm sites did not contribute to \
         the count below"
    );
    let rearms = scheduler.schedule_count() - baseline;
    assert_eq!(
        rearms, 3,
        "the exit must re-arm the scheduler ITSELF: it is returning with work still owed, and \
         nothing else will come back for it. Two of the three re-arms are nested \
         `immediate_tick`s (the deferred subscription's and `handle_sync_messages`'); the third \
         is the exit's own, and only the third is this gate's subject"
    );
    assert_eq!(
        core.storage().pass_depth_for_test(),
        0,
        "the exit must END the read pass before returning: a leaked depth is invisible — every \
         later boundary in the process becomes a no-op and reads exactly like a pass that is \
         still legitimately open"
    );
    assert!(
        core.storage().is_autocommit_for_test(),
        "ending the pass must have closed the transaction, not merely decremented the depth"
    );
    // Nothing left to drain -> `drained_any` is false -> the ordinary exit -> the barrier.
    core.batched_tick();
    assert!(
        !core.has_storage_write_pending_flush(),
        "the barrier the parked-message exit deferred must run on the next tick: deferring it \
         is only sound because the re-arm above guarantees that tick happens"
    );
}

/// G-C16 (diff r28, corrected by r30). IGNORED: an open defect, not a passing gate.
///
/// **The defect is real.** `begin_read_pass_in_tick` is the first statement of
/// `immediate_tick_inner` (`ticks.rs:706`) and `end_read_pass_in_tick` is near its end
/// (`:917`); a panic between them unwinds past the end, the `catch_unwind` at `:687` returns
/// control with `pass_depth` still at 1, and `SqliteStorage` only opens the pass transaction on
/// the 0→1 edge (`sqlite.rs:175`). Measured here: left 1, right 0.
///
/// **Two things I claimed about it were wrong, and both are corrected rather than deleted.**
///
/// 1. *"A host that survives a caught panic sees item 4 disabled for the life of the process."*
///    No shipping host survives one. Every in-tree host holds the core behind a `Mutex` and none
///    of them calls `into_inner()`: `runtime_tokio.rs:166-173` logs "core mutex poisoned; the
///    tick thread exits and the runtime is unusable" and returns, and `jazz-rn` (`:773-779`)
///    turns it into `lock poisoned`. A panic through a tick ends the runtime; nothing runs a
///    later pass to observe the leak. The settle clock's comment at `ticks.rs:680` asserts the
///    opposite ("a host that survives a caught panic — RN's panic boundary") and
///    `sqlite.rs:737` asserts this; the tree contradicts itself and `sqlite.rs` is the one the
///    sources support.
/// 2. *The injection is the production shape.* It is not. RN wraps the JS callback in its OWN
///    `catch_unwind` before the delta reaches it (`jazz-rn/rust/src/lib.rs:399-410`), so a
///    host-callback panic never unwinds into the tick at all. The gate below reaches the defect
///    by a path RN closes.
///
/// **And the fix is bigger than the depth.** The same unwind skips the barrier-scheduling tail
/// (`ticks.rs:919-927`) and the re-arm (`:694`, gated on `outcome.is_ok()`), so resetting the
/// depth alone would leave a DIRTY transaction open with nothing scheduled to commit it — the
/// WAL write lock held against every other connection. Two leaks on the same path outrank it:
/// the outbox (`ticks.rs:1135-1195` — `take_outbox` drains and `prepend_outbox` only runs at
/// the end, so a panic in the send loop destroys every remaining message and nothing retries)
/// and the deferred settle work (`take_settle_work_remains` at `:1027` with the re-arm vetoed
/// at `:953`). The design r30 recommends is to count opens in `RuntimeCore` and replay
/// `end_read_pass_in_tick()` that many times in the landing pad, then schedule the barrier — no
/// trait method, no second disposal contract, and `debug_assert!(pass_depth > 0)` stays alive.
///
/// It is ignored rather than deleted or fixed in a hurry because it is a correct measurement of
/// a real invariant violation that no shipping build can currently observe, and because a
/// half-fix here would cure the depth and leave the open dirty transaction — the exact shape of
/// defect 27, whose fix cured the local write path and left the inbound one.
///
/// The second assertion this gate used to carry — "a later pass runs its reads in one
/// transaction again" — was NOT load-bearing and has been removed rather than left to look like
/// evidence. `note_read` counts autocommit from `conn.is_autocommit()` (`sqlite.rs:422-431`),
/// not from `pass_depth`; after the panic the transaction is still open and no `batched_tick`
/// commits it, so the recovery reads run inside it and `autocommit == 0` holds WITH the defect
/// present. It never executed — the depth assertion above it is the one that reds.
///
/// Internal on purpose: `pass_depth` is engine bookkeeping. From outside, a store with item 4
/// disabled and one with it working return identical rows.
#[ignore = "open defect: a caught panic leaks pass_depth AND the dirty transaction and the \
            barrier schedule with it; unobservable in every shipping host (poisoned core \
            mutex), so it does not block linsa-v18. Design in diff r30; see the doc comment."]
#[test]
fn a_caught_panic_does_not_disable_the_pass_transaction_for_the_rest_of_the_process() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    let dir = tempfile::TempDir::new().expect("temp dir");
    let mut core = core_over(&dir.path().join("pass.sqlite"));
    seed_docs(&mut core, 200);

    let armed = Arc::new(AtomicBool::new(false));
    let trigger = Arc::clone(&armed);
    let _handle = core
        .subscribe(
            QueryBuilder::new("docs").build(),
            move |_delta| {
                if trigger.swap(false, Ordering::SeqCst) {
                    panic!("host callback panicked inside the settle pass");
                }
            },
            Some(Session::new("alice")),
        )
        .expect("the subscription registers");
    core.immediate_tick();

    // The write is what drives the delivery, and `insert` ticks internally — so the write is
    // what the landing pad has to wrap. Arming first and catching only a later `immediate_tick`
    // catches nothing: the delta is already delivered by then and the callback never runs.
    armed.store(true, Ordering::SeqCst);
    let alice = WriteContext::from_session(Session::new("alice"));
    let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        core.insert(
            "docs",
            HashMap::from([
                ("owner".to_string(), Value::Text("alice".to_string())),
                ("body".to_string(), Value::Text("panics".to_string())),
            ]),
            Some(&alice),
        )
        .expect("the row inserts");
    }))
    .is_err();
    assert!(
        unwound,
        "fixture: the armed callback must actually panic, or this gate proves nothing"
    );
    assert!(
        !armed.load(Ordering::SeqCst),
        "fixture: the panic must have come from the callback, not from somewhere else"
    );

    assert_eq!(
        core.storage().pass_depth_for_test(),
        0,
        "a caught panic unwound past `end_read_pass_in_tick`, so the store still believes a \
         settle pass is open. Nothing ever brings this back down: every later pass sees a \
         depth of 1, takes the nested branch, and never opens a transaction again"
    );

    // A second assertion used to stand here and it is gone on purpose — see the note in the
    // doc comment. It read "a later pass runs its reads in one transaction again", which sounds
    // like the behavioural half of this gate and is not: autocommit is counted from
    // `conn.is_autocommit()`, the panicked tick's transaction is still open, and nothing here
    // commits it, so it would read zero with the defect fully present.
}
