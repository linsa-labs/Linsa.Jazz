//! v18 item 6, the tick half of the settle budget: the clock is owned by the outermost tick,
//! a pass that defers NON-stalled work for budget re-arms exactly the batched ticks needed to
//! finish it, and stalled units — class (i): a server subscription whose scope is `None`
//! while the authorization schema is required and missing — never re-arm a tick.
//!
//! Internal on purpose: the observables are the scheduler's re-arm count (`CountingScheduler`,
//! pre-existing in `batched_tick_parked_drain`), the query manager's stalled set and its
//! `settle_work_remains` peek — engine state no client API exposes. `test_schema` is the
//! module's `RowDescriptor` helper, pre-existing. Budget `Some(0)`: one unit per tick.

use super::*;

use crate::query_manager::query::Query;
use crate::sync_manager::{Destination, QueryId, QueryPropagation};

type BudgetCore = RuntimeCore<MemoryStorage, CountingScheduler>;

fn server(scheduler: CountingScheduler, name: &str) -> BudgetCore {
    let schema_manager = SchemaManager::new(
        SyncManager::new(),
        test_schema(),
        AppId::from_name(name),
        "dev",
        "main",
    )
    .unwrap();
    new_test_core(schema_manager, MemoryStorage::new(), scheduler)
}

fn register(core: &mut BudgetCore, client_id: ClientId, query_id: u64) {
    core.push_sync_inbox(InboxEntry {
        source: Source::Client(client_id),
        payload: SyncPayload::QuerySubscription {
            query_id: QueryId(query_id),
            query: Box::new(Query::new("users")),
            session: None,
            required_tier: None,
            propagation: QueryPropagation::Full,
            policy_context_tables: Vec::new(),
        },
    });
}

fn budget(core: &mut BudgetCore, micros: Option<u64>) {
    core.schema_manager_mut()
        .query_manager_mut()
        .set_settle_budget_micros(micros);
}

fn pending(core: &BudgetCore) -> bool {
    core.schema_manager()
        .query_manager()
        .sync_manager()
        .has_pending_query_subscriptions()
}

fn settled_markers(core: &BudgetCore) -> usize {
    core.sync_sender()
        .take()
        .iter()
        .filter(|entry| matches!(entry.payload, SyncPayload::QuerySettled { .. }))
        .count()
}

/// Run batched ticks while the previous one re-armed the scheduler; the number run.
fn drive_until_quiet(core: &mut BudgetCore, scheduler: &CountingScheduler, cap: usize) -> usize {
    let mut seen = scheduler.schedule_count();
    let mut ticks = 0;
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

/// G6-7(a). Three registrations under a zero budget settle within a bounded number of
/// ticks with no external input: the first immediate tick serves one and re-arms, each
/// batched tick's pass serves the next and its continuation flushes and re-arms. Red
/// without the continuation: tick 2's pre-loop guard fails on the unflushed outbox and
/// the rest strands.
#[test]
fn deferred_registrations_are_finished_by_the_continuation_ticks() {
    let scheduler = CountingScheduler::default();
    let mut core = server(scheduler.clone(), "budget-ticks-to-quiescence");
    let client_id = ClientId::new();
    core.add_client(client_id, None);
    core.immediate_tick();
    core.batched_tick();
    let _ = core.sync_sender().take();

    budget(&mut core, Some(0));
    for query_id in 1..=3 {
        register(&mut core, client_id, query_id);
    }
    core.immediate_tick();
    let mut markers = settled_markers(&core);
    let ticks = drive_until_quiet(&mut core, &scheduler, 20);
    markers += settled_markers(&core);

    assert!(!pending(&core), "every registration was served");
    assert_eq!(markers, 3, "every registration settled exactly once");
    assert!(
        ticks <= 3,
        "one unit per tick: three registrations need at most three batched ticks after the \
         immediate one, took {ticks}"
    );
}

/// G6-7(b). An idle core under a zero budget re-arms nothing: the flag is set only by a
/// deferred non-stalled unit, never by pending or dirty state as such.
#[test]
fn an_idle_core_under_a_budget_re_arms_nothing() {
    let scheduler = CountingScheduler::default();
    let mut core = server(scheduler.clone(), "budget-idle");
    let client_id = ClientId::new();
    core.add_client(client_id, None);
    register(&mut core, client_id, 1);
    core.immediate_tick();
    drive_until_quiet(&mut core, &scheduler, 20);
    let _ = core.sync_sender().take();
    budget(&mut core, Some(0));
    let baseline = scheduler.schedule_count();
    for _ in 0..10 {
        core.batched_tick();
    }
    assert_eq!(
        scheduler.schedule_count() - baseline,
        0,
        "ten batched ticks on an idle core must not re-arm the scheduler"
    );
    assert!(!core.schema_manager().query_manager().settle_work_remains());
}

/// Fixture for G6-7(d)/(e): a server that REQUIRES an authorization schema and has none, two
/// non-bypass clients, the `add_client` catalogue replay drained so that the scheduler count
/// afterwards is continuation ticks only (class-(i) settles emit nothing).
fn auth_required_server(
    scheduler: CountingScheduler,
    name: &str,
) -> (BudgetCore, ClientId, ClientId) {
    let mut core = server(scheduler, name);
    core.schema_manager_mut()
        .query_manager_mut()
        .require_authorization_schema();
    assert!(
        !core
            .schema_manager()
            .query_manager()
            .has_authorization_schema_for_test(),
        "fixture: the authorization schema is required and missing"
    );
    let a = ClientId::new();
    let b = ClientId::new();
    core.add_client(a, None);
    core.add_client(b, None);
    core.immediate_tick();
    core.batched_tick();
    let _ = core.sync_sender().take();
    (core, a, b)
}

/// G6-7(d). Two class-(i) registrations under a zero budget: each is a unit that makes no
/// progress (scope `None`), so after the two registrations ran the flag-driven ticks stop —
/// at most two re-arms, then zero over ten more batched ticks — and both keys are stalled.
/// Red under a predicate that re-arms on any deferred unit: the two settles chase each other
/// every tick.
#[test]
fn stalled_units_do_not_re_arm_the_scheduler() {
    let scheduler = CountingScheduler::default();
    let (mut core, a, b) = auth_required_server(scheduler.clone(), "budget-stalled");
    budget(&mut core, Some(0));
    let baseline = scheduler.schedule_count();
    register(&mut core, a, 1);
    register(&mut core, b, 1);
    core.immediate_tick();
    let ticks = drive_until_quiet(&mut core, &scheduler, 20);
    let re_arms = scheduler.schedule_count() - baseline;
    assert!(
        re_arms <= 2,
        "flag-driven ticks are bounded by the registrations (two), got {re_arms} over {ticks} ticks"
    );
    assert!(!pending(&core), "both registrations were served");
    assert_eq!(
        core.schema_manager()
            .query_manager()
            .stalled_units_for_test(),
        2,
        "both subscriptions settled to no scope and are stalled"
    );
    let quiet = scheduler.schedule_count();
    for _ in 0..10 {
        core.batched_tick();
    }
    assert_eq!(
        scheduler.schedule_count() - quiet,
        0,
        "stalled units never re-arm a tick"
    );
    assert!(!core.schema_manager().query_manager().settle_work_remains());
}

/// G6-7(e). A write before every batched tick keeps two server subscriptions dirty (a
/// local insert: the same dirty mark a peer's frame leaves once applied). The insert's own
/// immediate tick (a fresh clock) serves the first unit; under a zero budget the batched
/// tick's pass then settles exactly ONE — the other — and its continuation serves nothing
/// in that lock hold (the clock is spent) and re-arms one batched tick. Every write is seen
/// by both subscriptions (10/10 markers over ten writes, the batched tick's marker flushed
/// one tick late); every tick ends quiescent: at most one re-arm per tick and never a
/// standing flag. Red under a flag that re-arms on any dirty unit, or under a continuation
/// that spends a nested pass under a spent clock (a second settle in the batched tick).
/// Rotation itself is G6-3's: with one unit left after the write's tick this fixture
/// cannot see it (diff r1 S2).
#[test]
fn a_write_per_tick_is_settled_within_the_tick_without_a_standing_re_arm() {
    let scheduler = CountingScheduler::default();
    let mut core = server(scheduler.clone(), "budget-write-per-tick");
    let a = ClientId::new();
    let b = ClientId::new();
    core.add_client(a, None);
    core.add_client(b, None);
    register(&mut core, a, 1);
    register(&mut core, b, 1);
    core.immediate_tick();
    drive_until_quiet(&mut core, &scheduler, 20);
    let _ = core.sync_sender().take();
    assert!(!pending(&core), "fixture: both subscriptions registered");
    // The passes one immediate tick runs (`immediate_tick_inner` processes twice). A batched
    // tick with deferred work runs exactly one immediate tick — after its send step — and
    // nothing nested: the continuation re-arms instead of spending a nested tick under a
    // spent clock (design v7). Under v6's shape the nested tick doubled this.
    let before = core.schema_manager().query_manager().passes_for_test();
    core.immediate_tick();
    let tick_passes = core.schema_manager().query_manager().passes_for_test() - before;
    assert!(tick_passes >= 1, "fixture: an immediate tick runs a pass");
    let _ = core.sync_sender().take();
    budget(&mut core, Some(0));
    let mut settles_total = 0;
    let mut markers_for_a = 0;
    let mut markers_for_b = 0;
    for n in 0..10 {
        core.insert(
            "users",
            user_insert_values(ObjectId::new(), &format!("row-{n}")),
            None,
        )
        .expect("a local write");
        let settles = core
            .schema_manager()
            .query_manager()
            .server_settles_for_test();
        let passes = core.schema_manager().query_manager().passes_for_test();
        let re_arms = scheduler.schedule_count();
        let settle_rearms = core.settle_rearms_for_test();
        core.batched_tick();
        let settled = core
            .schema_manager()
            .query_manager()
            .server_settles_for_test()
            - settles;
        assert_eq!(
            core.schema_manager().query_manager().passes_for_test() - passes,
            tick_passes,
            "tick {n}: one immediate tick in the batched tick and nothing nested — the \
             continuation must re-arm, not spend a nested tick under a spent clock"
        );
        settles_total += settled;
        assert_eq!(
            settled, 1,
            "tick {n}: the first unit of the pass and nothing more under a spent clock, got {settled}"
        );
        assert!(
            scheduler.schedule_count() - re_arms <= 1,
            "tick {n}: at most one re-arm per tick"
        );
        assert_eq!(
            core.settle_rearms_for_test(),
            settle_rearms,
            "tick {n}: nothing deferred, so no settle re-arm (the re-arm above is the outbound one)"
        );
        assert!(
            !core.schema_manager().query_manager().settle_work_remains(),
            "tick {n}: the tick must end quiescent"
        );
        for entry in core.sync_sender().take() {
            if let (Destination::Client(client), SyncPayload::QuerySettled { .. }) =
                (&entry.destination, &entry.payload)
            {
                if *client == a {
                    markers_for_a += 1;
                } else if *client == b {
                    markers_for_b += 1;
                }
            }
        }
    }
    // The batched tick's own settle is flushed by the NEXT tick's send step: one more tick
    // and take to see the last marker.
    core.batched_tick();
    for entry in core.sync_sender().take() {
        if let (Destination::Client(client), SyncPayload::QuerySettled { .. }) =
            (&entry.destination, &entry.payload)
        {
            if *client == a {
                markers_for_a += 1;
            } else if *client == b {
                markers_for_b += 1;
            }
        }
    }
    assert_eq!(
        settles_total, 10,
        "one settle per batched tick over ten ticks"
    );
    assert_eq!(
        (markers_for_a, markers_for_b),
        (10, 10),
        "both subscriptions settled on every write: one in the write's own immediate tick \
         (a fresh clock), the other in the batched tick"
    );
}

/// G6-7(f) (falsification F6-10, F6-11). A batched tick whose hold still leaves deferred
/// work reaches the continuation (3b). Three subscriptions, budget zero: the write's own tick
/// settles one; the batched tick begins a fresh clock and runs no pass of its own here (no
/// transport, nothing parked), so 3b finds room, takes the flag and runs ONE nested immediate
/// tick, which settles one unit and defers the last; the owner then re-arms exactly once for
/// it. Pinned: settles +1; passes == one immediate tick's (exactly one nested tick, nothing
/// doubled); `settle_rearms` +1 (F6-10: no owner re-arm → 0; F6-11: the nested wrapper
/// re-arms too → 2); the flag still set for the owner's re-arm; quiescence once the re-armed
/// tick runs. G6-7(e) has two units and never reaches the continuation — which is why the
/// continuation disarms stayed green on it. The spent-clock case is G6-7(g).
#[test]
fn a_batched_tick_with_work_left_re_arms_instead_of_spending_a_nested_tick() {
    let scheduler = CountingScheduler::default();
    let mut core = server(scheduler.clone(), "budget-continuation-shape");
    let clients = [ClientId::new(), ClientId::new(), ClientId::new()];
    for client in clients {
        core.add_client(client, None);
        register(&mut core, client, 1);
    }
    core.immediate_tick();
    drive_until_quiet(&mut core, &scheduler, 20);
    let _ = core.sync_sender().take();
    assert!(
        !pending(&core),
        "fixture: all three subscriptions registered"
    );
    let before = core.schema_manager().query_manager().passes_for_test();
    core.immediate_tick();
    let tick_passes = core.schema_manager().query_manager().passes_for_test() - before;
    assert!(tick_passes >= 1, "fixture: an immediate tick runs a pass");
    let _ = core.sync_sender().take();
    budget(&mut core, Some(0));
    core.insert("users", user_insert_values(ObjectId::new(), "row"), None)
        .expect("a local write");
    // The write's own immediate tick (a fresh clock) settled one unit and deferred two.
    let settles = core
        .schema_manager()
        .query_manager()
        .server_settles_for_test();
    let passes = core.schema_manager().query_manager().passes_for_test();
    let re_arms = core.settle_rearms_for_test();
    core.batched_tick();
    assert_eq!(
        core.schema_manager()
            .query_manager()
            .server_settles_for_test()
            - settles,
        1,
        "3b's one nested tick settles one unit and defers the last"
    );
    assert_eq!(
        core.schema_manager().query_manager().passes_for_test() - passes,
        tick_passes,
        "exactly one nested tick's passes: the batched tick runs no pass of its own here \
         and nothing is doubled"
    );
    assert_eq!(
        core.settle_rearms_for_test() - re_arms,
        1,
        "the owner re-arms exactly once for the unit the continuation left (the nested \
         tick's outbound re-arm is a different mechanism and not counted here)"
    );
    assert!(
        core.schema_manager().query_manager().settle_work_remains(),
        "the flag stays set for the batched tick this hold re-armed (flag iff re-arm)"
    );
    drive_until_quiet(&mut core, &scheduler, 20);
    assert_eq!(
        core.schema_manager()
            .query_manager()
            .server_settles_for_test()
            - settles,
        2,
        "the re-armed batched tick settles the last unit"
    );
    assert!(
        !core.schema_manager().query_manager().settle_work_remains(),
        "nothing deferred, nothing re-armed: the flag is clear"
    );
}

/// G6-7(g) (falsification F6-8, the spent-clock shape). A batched tick whose clock is
/// already spent when it reaches the continuation — a parked sync message applied in
/// `handle_sync_messages` ran a nested tick that served one unit and deferred the last —
/// does NOT spend a second nested tick there: under a spent clock it would refuse every
/// unit. The continuation leaves the flag, the owner re-arms once, the next batched tick
/// (a fresh clock) finishes. Red under an unconditional nested tick: the batched tick's pass
/// count exceeds one immediate tick's.
#[test]
fn a_batched_tick_with_a_spent_clock_leaves_the_continuation_to_the_next_tick() {
    let scheduler = CountingScheduler::default();
    let mut core = server(scheduler.clone(), "budget-spent-clock-continuation");
    let clients = [ClientId::new(), ClientId::new(), ClientId::new()];
    for client in clients {
        core.add_client(client, None);
        register(&mut core, client, 1);
    }
    core.immediate_tick();
    drive_until_quiet(&mut core, &scheduler, 20);
    let _ = core.sync_sender().take();
    assert!(
        !pending(&core),
        "fixture: all three subscriptions registered"
    );
    let before = core.schema_manager().query_manager().passes_for_test();
    core.immediate_tick();
    let tick_passes = core.schema_manager().query_manager().passes_for_test() - before;
    assert!(tick_passes >= 1, "fixture: an immediate tick runs a pass");
    let _ = core.sync_sender().take();
    budget(&mut core, Some(0));
    core.insert("users", user_insert_values(ObjectId::new(), "row"), None)
        .expect("a local write");
    // A parked message the batched tick applies before its continuation: the apply runs a
    // nested immediate tick under the batched tick's clock, which spends it.
    core.park_sync_message(InboxEntry {
        source: Source::Server(ServerId::new()),
        payload: SyncPayload::BatchFateNeeded {
            batch_ids: Vec::new(),
        },
    });
    let settles = core
        .schema_manager()
        .query_manager()
        .server_settles_for_test();
    let passes = core.schema_manager().query_manager().passes_for_test();
    let re_arms = core.settle_rearms_for_test();
    core.batched_tick();
    assert_eq!(
        core.schema_manager()
            .query_manager()
            .server_settles_for_test()
            - settles,
        1,
        "the applied message's tick settles one unit and spends the clock; the continuation \
         serves nothing more"
    );
    assert_eq!(
        core.schema_manager().query_manager().passes_for_test() - passes,
        tick_passes,
        "one tick's passes: the continuation must not run a nested tick under a spent clock"
    );
    assert_eq!(
        core.settle_rearms_for_test() - re_arms,
        1,
        "the owner re-arms exactly once for the unit the continuation left"
    );
    assert!(
        core.schema_manager().query_manager().settle_work_remains(),
        "the flag stays set for the batched tick this hold re-armed"
    );
    drive_until_quiet(&mut core, &scheduler, 20);
    assert_eq!(
        core.schema_manager()
            .query_manager()
            .server_settles_for_test()
            - settles,
        2,
        "the next batched tick, on a fresh clock, settles the last unit"
    );
}
