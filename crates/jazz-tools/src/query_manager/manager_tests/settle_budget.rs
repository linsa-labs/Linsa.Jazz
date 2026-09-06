//! v18 item 6: a settle pass with a budget, and fairness between clients.
//!
//! A pass (`QueryManager::process`) takes EVERY pending registration and EVERY dirty
//! subscription to completion, in arrival order for registrations and in `HashMap` order
//! for the rest, with no elapsed check — so one client that queues eight expensive
//! registrations holds the engine lock for all eight, and a quiet client's single cheap
//! registration waits behind them (prod 2026-09-04: 1.2 s passes, every backend read at
//! the 5 s facade deadline).
//!
//! What exists today is `MAX_INITIAL_QUERY_REPLAY_OUTBOX_PER_PASS` (server_queries.rs): after
//! a first settle, if the outbox already holds 32 entries the remaining registrations wait
//! for the next pass, in arrival order. It bounds the outbox, not the work: a registration
//! that stays under thirty-two entries is not counted at all (G6-6), a dirty settle is not
//! covered (G6-2), and arrival order is not fairness (G6-1, G6-3).
//!
//! Internal on purpose: the observable is WHICH units a pass settled and in what order,
//! which no client API exposes; the fixture is a server query manager with clients attached
//! directly and registrations pushed as inbox frames (the path a WebSocket frame takes).
//! The gates run with a budget of zero microseconds, under which a pass settles exactly one
//! unit — its first — so the fair order shows one client per pass and the assertions are
//! structural, not timed. Wall-time behaviour is measured on the stand, not here.

use super::*;
use crate::query_manager::graph_nodes::output::QuerySubscriptionId;
use crate::sync_manager::{ClientId, Destination, DurabilityTier, QueryId, ServerId};
use rand::{Rng, SeedableRng};

const USERS: i32 = 40;
const POSTS_PER_USER: i32 = 2;

/// A server query manager over `users`/`posts`, seeded so that an include of `posts`
/// instantiates one subgraph per user — the shape of a chat's `messages` subscription with
/// attachments, and the expensive first settle of prod.
fn seeded_server() -> (QueryManager, MemoryStorage) {
    let (mut qm, mut storage) = create_query_manager(SyncManager::new(), users_posts_schema());
    for user in 1..=USERS {
        qm.insert(
            &mut storage,
            "users",
            &[Value::Integer(user), Value::Text(format!("user-{user}"))],
        )
        .unwrap();
        for post in 0..POSTS_PER_USER {
            qm.insert(
                &mut storage,
                "posts",
                &[
                    Value::Integer(user * 100 + post),
                    Value::Text(format!("post-{user}-{post}")),
                    Value::Integer(user),
                ],
            )
            .unwrap();
        }
    }
    qm.process(&mut storage);
    let _ = qm.sync_manager_mut().take_outbox();
    (qm, storage)
}

fn expensive(qm: &QueryManager) -> crate::query_manager::query::Query {
    qm.query("users")
        .with_array("posts", |sub| {
            sub.from("posts").correlate("author_id", "users.id")
        })
        .build()
}

fn cheap(qm: &QueryManager) -> crate::query_manager::query::Query {
    qm.query("users").limit(1).build()
}

fn client(qm: &mut QueryManager, storage: &MemoryStorage) -> ClientId {
    let id = ClientId::new();
    connect_client(qm, storage, id);
    id
}

/// One pass; the `QuerySettled` markers it emitted, in emission order.
fn pass(qm: &mut QueryManager, storage: &mut MemoryStorage) -> Vec<(ClientId, QueryId)> {
    qm.process(storage);
    let markers: Vec<(ClientId, QueryId)> = qm
        .sync_manager_mut()
        .take_outbox()
        .into_iter()
        .filter_map(|entry| match (&entry.destination, &entry.payload) {
            (Destination::Client(client_id), SyncPayload::QuerySettled { query_id, .. }) => {
                Some((*client_id, *query_id))
            }
            _ => None,
        })
        .collect();
    // G6-4, checked on every pass of every gate: a marker is emitted only for a subscription
    // whose graph has nothing dirty left — a deferred unit must never look settled.
    for (client_id, query_id) in &markers {
        let sub = qm
            .server_subscriptions
            .get(&(*client_id, *query_id))
            .expect("a settled marker names a live server subscription");
        assert!(
            !sub.graph.has_dirty_nodes(),
            "a QuerySettled marker was emitted for {client_id}/{} with a dirty graph",
            query_id.0
        );
    }
    markers
}

fn passes_until(
    qm: &mut QueryManager,
    storage: &mut MemoryStorage,
    limit: usize,
    mut done: impl FnMut(&[Vec<(ClientId, QueryId)>]) -> bool,
) -> Vec<Vec<(ClientId, QueryId)>> {
    let mut history = Vec::new();
    for _ in 0..limit {
        history.push(pass(qm, storage));
        if done(&history) {
            break;
        }
    }
    history
}

fn pass_of(
    history: &[Vec<(ClientId, QueryId)>],
    client_id: ClientId,
    query_id: QueryId,
) -> Option<usize> {
    history
        .iter()
        .position(|markers| markers.contains(&(client_id, query_id)))
        .map(|index| index + 1)
}

/// G6-1. A quiet client's registration is answered within a bounded number of passes, and
/// no pass settles more than its budget allows (one unit at a zero budget). Red today at
/// the per-pass bound: the first pass settles all nine.
#[test]
fn a_quiet_client_is_answered_within_two_passes_behind_a_loud_one() {
    // Every gate in this file writes and then observes what settled, so every one of them
    // DEPENDS on precise dirtiness — and this guard is how a test says so. Without it, a
    // parallel force-user in the same binary (the subscription-output differential engages
    // `force_precise_dirty(false)` for its Legacy leg) can hold the process-global switch OFF
    // across this gate's write and have it back ON by the settle. The write then marks the
    // graph node dirty without buffering any instance dirt, the settle consults the instance
    // dirt and finds none, and twenty subscriptions go clean having emitted nothing:
    // `left: 0`, `settled per pass: [0; 25]`. Measured, not reasoned — forcing that exact tear
    // by hand reproduces the intermittent failure byte for byte.
    //
    // The guard also serialises against those force-users, since it holds their mutex.
    let _precise = crate::query_manager::precise_dirty::force_precise_dirty(true);
    let (mut qm, mut storage) = seeded_server();
    qm.set_settle_budget_micros(Some(0));
    let loud = client(&mut qm, &storage);
    let quiet = client(&mut qm, &storage);
    for query_id in 1..=8 {
        let query = expensive(&qm);
        push_query_subscription(&mut qm, loud, query_id, query);
    }
    let query = cheap(&qm);
    push_query_subscription(&mut qm, quiet, 1, query);

    let history = passes_until(&mut qm, &mut storage, 12, |history| {
        pass_of(history, quiet, QueryId(1)).is_some()
    });
    for (index, markers) in history.iter().enumerate() {
        assert!(
            markers.len() <= 1,
            "pass {} settled {} units under a zero budget; a pass must settle its first unit \
             and then stop — this is the lock hold every other client waits behind",
            index + 1,
            markers.len()
        );
    }
    assert!(
        pass_of(&history, quiet, QueryId(1)).is_some_and(|pass| pass <= 2),
        "the quiet client's one cheap registration must be served within two passes, not \
         behind all eight of the loud client's (served in pass {:?})",
        pass_of(&history, quiet, QueryId(1))
    );
}

/// Confirm the rows a pass offered, as the receiver would, so the server's delivery
/// bookkeeping advances and a later registration over the same rows adds nothing to the
/// outbox (the peer-already-has-it case of a reconnect or a re-registration).
fn confirm_delivered(qm: &mut QueryManager, entries: &[crate::sync_manager::OutboxEntry]) {
    let confirmed: Vec<_> = entries
        .iter()
        .filter_map(|entry| match (&entry.destination, &entry.payload) {
            (Destination::Client(client_id), SyncPayload::RowBatchNeeded { row, .. }) => Some((
                *client_id,
                row.row_id,
                crate::object::BranchName::new(row.branch.as_str()),
                row.batch_id,
            )),
            _ => None,
        })
        .collect();
    qm.sync_manager_mut().confirm_client_deliveries(&confirmed);
}

/// One pass whose outbox is drained AND confirmed; the markers it emitted.
fn confirmed_pass(qm: &mut QueryManager, storage: &mut MemoryStorage) -> Vec<(ClientId, QueryId)> {
    qm.process(storage);
    let outbox = qm.sync_manager_mut().take_outbox();
    confirm_delivered(qm, &outbox);
    outbox
        .iter()
        .filter_map(|entry| match (&entry.destination, &entry.payload) {
            (Destination::Client(client_id), SyncPayload::QuerySettled { query_id, .. }) => {
                Some((*client_id, *query_id))
            }
            _ => None,
        })
        .collect()
}

/// G6-2. A dirty settle is bounded too: twenty settled subscriptions made dirty by one
/// write settle one per pass under a zero budget, every one of them exactly once, within
/// twenty-one passes. Red today: the write's pass settles all twenty (the outbox limiter
/// covers first settles only).
#[test]
fn twenty_dirty_subscriptions_settle_one_per_pass_and_all_of_them_settle() {
    // Precise dirtiness pinned for the same reason as the first gate in this file: a
    // parallel force-user can otherwise tear this gate's write from its settle.
    let _precise = crate::query_manager::precise_dirty::force_precise_dirty(true);
    let (mut qm, mut storage) = seeded_server();
    let clients: Vec<ClientId> = (0..20).map(|_| client(&mut qm, &storage)).collect();
    for client_id in &clients {
        let query = expensive(&qm);
        push_query_subscription(&mut qm, *client_id, 1, query);
    }
    // Registration under no budget: let every first settle land and be confirmed.
    let mut settled = 0;
    for _ in 0..40 {
        settled += confirmed_pass(&mut qm, &mut storage).len();
        if settled >= 20 {
            break;
        }
    }
    assert_eq!(
        settled, 20,
        "fixture: every registration settled before the write"
    );
    qm.set_settle_budget_micros(Some(0));
    // One post for user 1 dirties the `posts` subgraph of every subscription.
    qm.insert(
        &mut storage,
        "posts",
        &[
            Value::Integer(9_999),
            Value::Text("late".into()),
            Value::Integer(1),
        ],
    )
    .unwrap();
    let history = passes_until(&mut qm, &mut storage, 25, |history| {
        history.iter().map(Vec::len).sum::<usize>() >= 20
    });
    for (index, markers) in history.iter().enumerate() {
        assert!(
            markers.len() <= 1,
            "pass {} settled {} dirty subscriptions under a zero budget",
            index + 1,
            markers.len()
        );
    }
    let all: Vec<(ClientId, QueryId)> = history.iter().flatten().copied().collect();
    // The per-pass shape is in the message on purpose. This gate has failed intermittently in
    // full-suite runs with `left: 0`, and `{all:?}` on an empty vector says nothing at all about
    // whether 25 passes ran and settled nothing, or fewer passes ran — which are different
    // defects. Diff r24 B5.
    let shape: Vec<usize> = history.iter().map(Vec::len).collect();
    // Units, not just markers. `left: 0` on the markers alone has two readings that this gate
    // could not tell apart, and the wrong one cost two oracle rounds: (a) no unit ran, which
    // would be a liveness bug in the budget path, or (b) every unit ran and none of them
    // changed a scope, so nothing was emitted — dirt lost upstream of the settle. It was
    // always (b). Asserting units first makes the next failure name its own reading.
    assert!(
        qm.pool_units_run_for_test() >= 20,
        "the pool must have RUN at least one unit per dirty subscription — {} units over {} \
         passes. If this is the assertion that fails, the defect is in the budget path; if it \
         passes and the marker count below fails, the units ran and their dirt was lost before \
         the settle",
        qm.pool_units_run_for_test(),
        history.len()
    );
    assert_eq!(
        all.len(),
        20,
        "every dirty subscription settles, none twice: {all:?} (passes run: {}, settled per \
         pass: {shape:?})",
        history.len()
    );
    for client_id in &clients {
        assert_eq!(
            all.iter().filter(|(id, _)| id == client_id).count(),
            1,
            "client {client_id} must be settled exactly once"
        );
    }
    assert!(
        history.len() <= 21,
        "twenty dirty settles must complete within twenty-one passes, took {}",
        history.len()
    );
}

/// A registration under the outbox limiter's threshold: ten users and their posts are thirty
/// outbox entries, two short of the limiter, and still ten subgraph instantiations each.
fn small_expensive(qm: &QueryManager) -> crate::query_manager::query::Query {
    qm.query("users")
        .limit(10)
        .with_array("posts", |sub| {
            sub.from("posts").correlate("author_id", "users.id")
        })
        .build()
}

/// G6-6. The outbox limiter counts entries, not work: eight registrations that each stay
/// under its threshold all settle in the pass they arrive in, and the quiet client's cheap
/// registration behind them waits for all eight. Under a zero budget: one unit per pass and
/// the quiet client is served within two. Red today at the per-pass bound.
#[test]
fn registrations_under_the_outbox_limiter_are_still_one_unit_each() {
    // Precise dirtiness pinned for the same reason as the first gate in this file: a
    // parallel force-user can otherwise tear this gate's write from its settle.
    let _precise = crate::query_manager::precise_dirty::force_precise_dirty(true);
    let (mut qm, mut storage) = seeded_server();
    qm.set_settle_budget_micros(Some(0));
    let loud = client(&mut qm, &storage);
    let quiet = client(&mut qm, &storage);
    for query_id in 1..=8 {
        let query = small_expensive(&qm);
        push_query_subscription(&mut qm, loud, query_id, query);
    }
    let query = cheap(&qm);
    push_query_subscription(&mut qm, quiet, 1, query);
    let history = passes_until(&mut qm, &mut storage, 12, |history| {
        pass_of(history, quiet, QueryId(1)).is_some()
    });
    for (index, markers) in history.iter().enumerate() {
        assert!(
            markers.len() <= 1,
            "pass {} settled {} units under a zero budget — each stays under the outbox \
             limiter, which counts entries and not work",
            index + 1,
            markers.len()
        );
    }
    assert!(
        pass_of(&history, quiet, QueryId(1)).is_some_and(|pass| pass <= 2),
        "the quiet client must be served within two passes (served in pass {:?})",
        pass_of(&history, quiet, QueryId(1))
    );
}

/// G6-3. Rotation: two loud clients and one quiet, in either registration order, the quiet
/// one is served within three passes (one round of the round-robin), and while two clients
/// both have pending units no client is served twice in a row. Red today (one pass settles
/// everything, in arrival order).
#[test]
fn clients_are_served_round_robin_whichever_order_they_registered_in() {
    // Precise dirtiness pinned for the same reason as the first gate in this file: a
    // parallel force-user can otherwise tear this gate's write from its settle.
    let _precise = crate::query_manager::precise_dirty::force_precise_dirty(true);
    for quiet_first in [true, false] {
        let (mut qm, mut storage) = seeded_server();
        qm.set_settle_budget_micros(Some(0));
        let loud_a = client(&mut qm, &storage);
        let loud_b = client(&mut qm, &storage);
        let quiet = client(&mut qm, &storage);
        let register_quiet = |qm: &mut QueryManager| {
            let query = cheap(qm);
            push_query_subscription(qm, quiet, 1, query);
        };
        if quiet_first {
            register_quiet(&mut qm);
        }
        // a, a, a then b, b, b: arrival order does not alternate by itself, so the
        // alternation below can only come from the rotation.
        for loud in [loud_a, loud_b] {
            for query_id in 1..=3 {
                let query = expensive(&qm);
                push_query_subscription(&mut qm, loud, query_id, query);
            }
        }
        if !quiet_first {
            register_quiet(&mut qm);
        }
        let history = passes_until(&mut qm, &mut storage, 10, |history| {
            history.iter().map(Vec::len).sum::<usize>() >= 7
        });
        for (index, markers) in history.iter().enumerate() {
            assert!(
                markers.len() <= 1,
                "quiet_first={quiet_first}: pass {} settled {} units under a zero budget",
                index + 1,
                markers.len()
            );
        }
        assert!(
            pass_of(&history, quiet, QueryId(1)).is_some_and(|pass| pass <= 3),
            "quiet_first={quiet_first}: the quiet client must be served within one round \
             (three passes), was served in pass {:?}",
            pass_of(&history, quiet, QueryId(1))
        );
        // While both loud clients still have pending units, they alternate.
        let served: Vec<ClientId> = history
            .iter()
            .filter_map(|markers| markers.first().map(|(client_id, _)| *client_id))
            .filter(|client_id| *client_id != quiet)
            .collect();
        for window in served.windows(2).take(4) {
            assert_ne!(
                window[0], window[1],
                "quiet_first={quiet_first}: a loud client was served twice in a row while the \
                 other loud client still had pending units: {served:?}"
            );
        }
    }
}

/// G6-5. Under continuous load — the loud client registers one more expensive query before
/// every pass — the quiet client registered once at the start is still served within two
/// passes. Red today at the per-pass bound.
#[test]
fn a_quiet_client_is_not_starved_by_a_client_that_keeps_registering() {
    // Precise dirtiness pinned for the same reason as the first gate in this file: a
    // parallel force-user can otherwise tear this gate's write from its settle.
    let _precise = crate::query_manager::precise_dirty::force_precise_dirty(true);
    let (mut qm, mut storage) = seeded_server();
    qm.set_settle_budget_micros(Some(0));
    let loud = client(&mut qm, &storage);
    let quiet = client(&mut qm, &storage);
    let query = expensive(&qm);
    push_query_subscription(&mut qm, loud, 1, query);
    let query = cheap(&qm);
    push_query_subscription(&mut qm, quiet, 1, query);
    let mut history = Vec::new();
    for round in 2..=10 {
        history.push(pass(&mut qm, &mut storage));
        let query = expensive(&qm);
        push_query_subscription(&mut qm, loud, round, query);
        if pass_of(&history, quiet, QueryId(1)).is_some() {
            break;
        }
    }
    for (index, markers) in history.iter().enumerate() {
        assert!(
            markers.len() <= 1,
            "pass {} settled {} units under a zero budget",
            index + 1,
            markers.len()
        );
    }
    assert!(
        pass_of(&history, quiet, QueryId(1)).is_some_and(|pass| pass <= 2),
        "the quiet client must be served within two passes under continuous load from the \
         loud one (served in pass {:?})",
        pass_of(&history, quiet, QueryId(1))
    );
}

/// G6-5′. Registrations do not starve dirty settles: the loud client registers one more
/// expensive query before every pass, a write dirties the quiet client's settled
/// subscription, and under a zero budget the quiet client's marker lands within two passes.
/// Red today at the per-pass bound (and, once units are rationed naively, at the marker).
#[test]
fn a_dirty_subscription_is_not_starved_by_a_client_that_keeps_registering() {
    // Precise dirtiness pinned for the same reason as the first gate in this file: a
    // parallel force-user can otherwise tear this gate's write from its settle.
    let _precise = crate::query_manager::precise_dirty::force_precise_dirty(true);
    let (mut qm, mut storage) = seeded_server();
    let loud = client(&mut qm, &storage);
    let quiet = client(&mut qm, &storage);
    let query = expensive(&qm);
    push_query_subscription(&mut qm, quiet, 1, query);
    let first = confirmed_pass(&mut qm, &mut storage);
    assert_eq!(
        first,
        vec![(quiet, QueryId(1))],
        "fixture: the quiet subscription settled"
    );
    qm.set_settle_budget_micros(Some(0));
    qm.insert(
        &mut storage,
        "posts",
        &[
            Value::Integer(9_998),
            Value::Text("late".into()),
            Value::Integer(2),
        ],
    )
    .unwrap();
    let mut history = Vec::new();
    for round in 1..=10 {
        let query = expensive(&qm);
        push_query_subscription(&mut qm, loud, round, query);
        history.push(pass(&mut qm, &mut storage));
        if pass_of(&history, quiet, QueryId(1)).is_some() {
            break;
        }
    }
    for (index, markers) in history.iter().enumerate() {
        assert!(
            markers.len() <= 1,
            "pass {} settled {} units under a zero budget",
            index + 1,
            markers.len()
        );
    }
    assert!(
        pass_of(&history, quiet, QueryId(1)).is_some_and(|pass| pass <= 2),
        "the quiet client's dirty subscription must settle within two passes while the loud \
         client keeps registering (settled in pass {:?})",
        pass_of(&history, quiet, QueryId(1))
    );
}

/// G6-9 (falsify-only). The pool is dispatched at step 8's position, after the recompile at
/// step 6: a subscription recompiled by a schema change settles in the pass that recompiled
/// it. Red with the pool at 3b (the recompile would dirty a graph the pool already passed).
#[test]
fn a_recompiled_subscription_settles_in_the_pass_that_recompiled_it() {
    // Precise dirtiness pinned for the same reason as the first gate in this file: a
    // parallel force-user can otherwise tear this gate's write from its settle.
    let _precise = crate::query_manager::precise_dirty::force_precise_dirty(true);
    let (mut qm, mut storage) = seeded_server();
    let quiet = client(&mut qm, &storage);
    let query = cheap(&qm);
    push_query_subscription(&mut qm, quiet, 1, query);
    let first = confirmed_pass(&mut qm, &mut storage);
    assert_eq!(
        first,
        vec![(quiet, QueryId(1))],
        "fixture: the subscription settled"
    );
    qm.set_authorization_schema(users_posts_schema());
    qm.set_settle_budget_micros(Some(0));
    qm.process(&mut storage);
    let sub = qm
        .server_subscriptions
        .get(&(quiet, QueryId(1)))
        .expect("the subscription is live");
    assert!(
        !sub.graph.has_dirty_nodes(),
        "the recompiled subscription must settle in the pass that recompiled it"
    );
}

/// The state the differential compares: per live server subscription its scope, whether it
/// settled once and the tier it last emitted; per client the rows it was offered over the
/// whole replay. Strings, so `HashSet` order and foreign types cannot leak into the
/// comparison.
#[derive(Debug, PartialEq)]
struct ReplayState {
    subscriptions: std::collections::BTreeMap<String, (Vec<String>, bool, String)>,
    offered: std::collections::BTreeMap<String, std::collections::BTreeSet<String>>,
    passes_to_quiescence: usize,
}

#[derive(Debug, Clone)]
enum ReplayOp {
    Register {
        client: usize,
        query: u64,
        expensive: bool,
    },
    /// A LOCAL subscription on the server (diff r1 S2): `Unit::Local`, the L progress
    /// predicate and the Local rotation slot are inside the oracle through it.
    Subscribe {
        expensive: bool,
    },
    Insert {
        user: i32,
    },
    Pass,
    /// The server starts requiring an authorization schema it does not have: every
    /// subscription settles to no scope (class (i)) until `SetAuth`.
    RequireAuth,
    SetAuth,
}

/// Nested rows in a local result carry their `ObjectId`s, minted per replay: blank them so
/// the two replays' results compare by content.
fn without_object_ids(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find("ObjectId(") {
        let after = at + "ObjectId(".len();
        out.push_str(&rest[..after]);
        out.push('_');
        let close = rest[after..]
            .find(')')
            .map(|i| after + i)
            .unwrap_or(rest.len());
        rest = &rest[close..];
    }
    out.push_str(rest);
    out
}

/// A row's identity across replays: row ids are minted per replay, the table and the encoded
/// values are the same.
fn logical_row(storage: &MemoryStorage, row_id: ObjectId, branch: &str) -> String {
    let locator = storage
        .load_row_locator(row_id)
        .unwrap()
        .expect("a row in a scope has a locator");
    let batch = load_visible_row(storage, row_id, branch);
    format!("{}:{:02x?}", locator.table.as_str(), batch.data)
}

/// Replay `ops` on a fresh seeded server over `clients`, under `budget`; then drive the
/// manager to quiescence and read the state.
fn replay(clients: &[ClientId], ops: &[ReplayOp], budget: Option<u64>) -> ReplayState {
    let (mut qm, mut storage) = seeded_server();
    for client_id in clients {
        connect_client(&mut qm, &storage, *client_id);
    }
    qm.set_settle_budget_micros(budget);
    let mut offered: std::collections::BTreeMap<String, std::collections::BTreeSet<String>> =
        std::collections::BTreeMap::new();
    let trace = std::env::var("JAZZ_SETTLE_TRACE").is_ok();
    let mut pass_no = 0usize;
    let mut one_pass = |qm: &mut QueryManager, storage: &mut MemoryStorage| {
        qm.process(storage);
        let outbox = qm.sync_manager_mut().take_outbox();
        confirm_delivered(qm, &outbox);
        pass_no += 1;
        if trace {
            let mut kinds: std::collections::BTreeMap<String, usize> = Default::default();
            for entry in &outbox {
                let kind = format!("{:?}", entry.destination)
                    + " "
                    + format!("{:?}", entry.payload)
                        .split(' ')
                        .next()
                        .unwrap_or("?");
                *kinds.entry(kind).or_default() += 1;
            }
            eprintln!("[trace budget={budget:?}] pass {pass_no}: outbox {kinds:?}");
        }
        for entry in &outbox {
            if let (Destination::Client(client_id), SyncPayload::RowBatchNeeded { row, .. }) =
                (&entry.destination, &entry.payload)
            {
                offered
                    .entry(format!("{client_id:?}"))
                    .or_default()
                    .insert(logical_row(storage, row.row_id, row.branch.as_str()));
            }
        }
    };
    for op in ops {
        if trace {
            eprintln!("[trace budget={budget:?}] op {op:?}");
        }
        match op {
            ReplayOp::Register {
                client,
                query,
                expensive: true,
            } => {
                let q = expensive(&qm);
                push_query_subscription(&mut qm, clients[*client], *query, q);
            }
            ReplayOp::Register {
                client,
                query,
                expensive: false,
            } => {
                let q = cheap(&qm);
                push_query_subscription(&mut qm, clients[*client], *query, q);
            }
            ReplayOp::Insert { user } => {
                qm.insert(
                    &mut storage,
                    "users",
                    &[Value::Integer(*user), Value::Text(format!("user-{user}"))],
                )
                .unwrap();
            }
            ReplayOp::Subscribe {
                expensive: is_expensive,
            } => {
                let q = if *is_expensive {
                    expensive(&qm)
                } else {
                    cheap(&qm)
                };
                qm.subscribe(q)
                    .expect("a local subscription over the seeded schema");
            }
            ReplayOp::Pass => one_pass(&mut qm, &mut storage),
            ReplayOp::RequireAuth => qm.require_authorization_schema(),
            ReplayOp::SetAuth => qm.set_authorization_schema(users_posts_schema()),
        }
    }
    // Quiescence. Unbounded: three full passes finish everything the sequence left (a pass
    // takes every registration and every dirty unit; the outbox limiter can hold a
    // registration over once). Bounded: until no live unit remains — a bounded pass runs at
    // least one unit whenever one is live, so the tail is bounded by the units it ran.
    // The sequence may end with frames still in the inbox; a pass moves them to the
    // pending registrations (a live unit) — the quiescence check sees the inbox only
    // through that first pass.
    one_pass(&mut qm, &mut storage);
    let units_before = qm.pool_units_run_for_test();
    let mut passes_to_quiescence = 1;
    while qm.has_live_units_for_test() {
        one_pass(&mut qm, &mut storage);
        passes_to_quiescence += 1;
        if budget.is_some() {
            let units = qm.pool_units_run_for_test() - units_before;
            assert!(
                passes_to_quiescence <= (units as usize).max(1) + 2,
                "budget {budget:?}: {passes_to_quiescence} passes ran only {units} pool units \
                 — a pass that runs no unit while units are live is a hang"
            );
        }
        assert!(
            passes_to_quiescence < 1_000,
            "budget {budget:?}: no quiescence"
        );
    }
    let mut subscriptions: std::collections::BTreeMap<String, (Vec<String>, bool, String)> = qm
        .server_subscriptions
        .iter()
        .map(|((client_id, query_id), sub)| {
            let mut scope: Vec<String> = sub
                .last_scope
                .iter()
                .map(|(row_id, branch)| logical_row(&storage, *row_id, branch.as_str()))
                .collect();
            scope.sort();
            (
                format!("{client_id:?}/{}", query_id.0),
                (
                    scope,
                    sub.settled_once,
                    format!("{:?}", sub.last_emitted_settled_tier),
                ),
            )
        })
        .collect();
    // Local subscriptions (diff r1 S2): the rows they would deliver, sorted, and whether they
    // settled; ids are minted per replay in op order, so the keys line up across replays.
    let local_ids: Vec<QuerySubscriptionId> = qm.subscriptions.keys().cloned().collect();
    for sub_id in local_ids {
        let mut rows: Vec<String> = qm
            .get_subscription_results(sub_id)
            .into_iter()
            .map(|(_, values)| without_object_ids(&format!("{values:?}")))
            .collect();
        rows.sort();
        let settled_once = qm.subscriptions[&sub_id].settled_once;
        subscriptions.insert(
            format!("local/{}", sub_id.0),
            (rows, settled_once, "local".to_string()),
        );
    }
    ReplayState {
        subscriptions,
        offered,
        passes_to_quiescence,
    }
}

/// Differential (methodology step 9). The same random sequence of registrations, writes and
/// passes, over the same clients, replayed on an unbounded manager (`None`) and on one under
/// a zero budget (one unit per pass); both driven to quiescence afterwards. The observable
/// state must agree: the set of live server subscriptions and, per subscription, its scope,
/// `settled_once` and the tier it last emitted; per client, the set of rows it was offered.
/// Excluded on purpose: `SUBSCRIPTIONS_SETTLED` and marker counts — the bounded manager
/// splits and coalesces settles differently by construction. `JAZZ_SETTLE_SEED` replays.
#[test]
fn a_bounded_manager_reaches_the_same_state_as_the_unbounded_one() {
    // Precise dirtiness pinned for the same reason as the first gate in this file: a
    // parallel force-user can otherwise tear this gate's write from its settle.
    let _precise = crate::query_manager::precise_dirty::force_precise_dirty(true);
    let seed = std::env::var("JAZZ_SETTLE_SEED")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or_else(rand::random::<u64>);
    eprintln!("settle differential seed = {seed} (JAZZ_SETTLE_SEED to replay)");
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut next_user = 1_000;
    let mut next_query = 0;
    for round in 0..12 {
        // Every third round re-registers keys (query ids drawn from three); the others use
        // a fresh id per registration.
        let re_registers = round % 3 == 2;
        let clients: Vec<ClientId> = (0..3).map(|_| ClientId::new()).collect();
        let mut ops: Vec<ReplayOp> = (0..rng.gen_range(6..=24))
            .map(|_| match rng.gen_range(0..10) {
                0..=3 => ReplayOp::Register {
                    client: rng.gen_range(0..3),
                    query: if re_registers {
                        rng.gen_range(1..=3)
                    } else {
                        next_query += 1;
                        next_query
                    },
                    expensive: rng.gen_bool(0.5),
                },
                4..=6 => {
                    next_user += 1;
                    ReplayOp::Insert { user: next_user }
                }
                7 => ReplayOp::Subscribe {
                    expensive: rng.gen_bool(0.5),
                },
                _ => ReplayOp::Pass,
            })
            .collect();
        // Every other round has an auth-required phase: from a random point every settle is
        // class (i) (stalled under the budget, a unit every pass without it) until the
        // schema arrives — appended last so both managers can go quiet.
        if round % 2 == 1 {
            let at = rng.gen_range(0..ops.len());
            ops.insert(at, ReplayOp::RequireAuth);
            let set_at = rng.gen_range(at + 1..=ops.len());
            ops.insert(set_at, ReplayOp::SetAuth);
        }
        let unbounded = replay(&clients, &ops, None);
        let bounded = replay(&clients, &ops, Some(0));
        // diff r27 §5a: `Some(0)` lets exactly one unit run per pass, so the round-robin past
        // the first slot, the live-before-stalled ordering and `limiter_tripped` are dead in
        // every gate in this file AND in this differential. Until this arm existed, the shape
        // production would actually run — a budget in the tens of milliseconds, many units per
        // pass — had no test in the crate at all. 50 ms is the shipping candidate; at test
        // speeds it rarely trips, which is the point: it exercises the pool's MULTI-unit path,
        // which `Some(0)` never reaches.
        let production = replay(&clients, &ops, Some(50_000));
        for (name, other) in [("bounded", &bounded), ("production", &production)] {
            if unbounded.subscriptions == other.subscriptions {
                continue;
            }
            let mut report = String::new();
            for (key, state) in &unbounded.subscriptions {
                match other.subscriptions.get(key) {
                    None => report.push_str(&format!("\n{key}: only unbounded ({state:?})")),
                    Some(o) if o != state => report.push_str(&format!(
                        "\n{key}: unbounded {state:?}\n{key}: {name}   {o:?}"
                    )),
                    Some(_) => {}
                }
            }
            for key in other.subscriptions.keys() {
                if !unbounded.subscriptions.contains_key(key) {
                    report.push_str(&format!("\n{key}: only {name}"));
                }
            }
            panic!(
                "seed {seed} round {round}: subscription state differs under {name}{report}\nops = {ops:?}"
            );
        }
        // Offers are order-dependent by construction: a registration the budget defers sees
        // a later world (a write between the frame and its first settle lands in the first
        // scope instead of a delta), and a key re-registered before its settle is served
        // rows under `None` that the budget never offers. The invariant that holds
        // regardless: every row of a final scope was offered to its client in BOTH replays
        // (nothing lost). `JAZZ_SETTLE_TRACE=1` prints every op and every pass's outbox.
        for (name, state) in [
            ("unbounded", &unbounded),
            ("bounded", &bounded),
            ("production", &production),
        ] {
            for (key, (scope, _, _)) in &state.subscriptions {
                if key.starts_with("local/") {
                    continue;
                    // local subscriptions deliver, they are not offered rows
                }
                let client = key.split('/').next().unwrap_or_default();
                let offered = state.offered.get(client).cloned().unwrap_or_default();
                let missing: Vec<_> = scope.iter().filter(|row| !offered.contains(*row)).collect();
                assert!(
                    missing.is_empty(),
                    "seed {seed} round {round} ({name}): {key} has rows in its final scope \
                     that were never offered: {missing:?}\nops = {ops:?}"
                );
            }
        }
        // Rounds with neither confounder (no re-registration, no authorization flip) have
        // no order-dependence left: the offered sets must be EQUAL (diff r1 S1) — an
        // over-offer by the bounded manager is a security-relevant difference.
        if !re_registers && round % 2 == 0 {
            assert_eq!(
                unbounded.offered, bounded.offered,
                "seed {seed} round {round}: the bounded manager offered a different row set\nops = {ops:?}"
            );
            assert_eq!(
                unbounded.offered, production.offered,
                "seed {seed} round {round}: the manager at a PRODUCTION-shaped budget offered a \
                 different row set\nops = {ops:?}"
            );
        }
        // No offer-subset check in the other rounds: with inserts only, scopes grow monotonically, so a row the
        // budget offered is in a final scope and coverage above already proves the unbounded
        // manager offered it too; under a re-registration or an authorization flip the rows
        // offered before the replacement/flip depend on which registration each manager
        // reached first (FIFO at step 3b, round-robin in the pool) — order, not correctness.
        let _ = (
            unbounded.passes_to_quiescence,
            bounded.passes_to_quiescence,
            production.passes_to_quiescence,
        );
    }
}

/// G6-10. Stalled units are a global last tier: a live registration is never deferred
/// behind a stalled unit that comes first in the rotation. Fixture: the server requires an
/// authorization schema it does not have, so every settle is class (i) and every
/// subscription is stalled from birth; A has the lower client id (first in rotation after
/// B's slot); the cursor sits at B; B registers a new query. Under a zero budget the pass
/// must serve B's registration (the pool's first unit), not A's stalled settle. Red with
/// the tier removed (rotation order puts A's stalled unit first).
#[test]
fn a_live_registration_is_not_deferred_behind_a_stalled_unit() {
    // Precise dirtiness pinned for the same reason as the first gate in this file: a
    // parallel force-user can otherwise tear this gate's write from its settle.
    let _precise = crate::query_manager::precise_dirty::force_precise_dirty(true);
    let (mut qm, mut storage) = seeded_server();
    qm.require_authorization_schema();
    let mut ids = [ClientId::new(), ClientId::new()];
    ids.sort();
    let [a, b] = ids;
    connect_client(&mut qm, &storage, a);
    connect_client(&mut qm, &storage, b);
    qm.set_settle_budget_micros(Some(0));
    let query = cheap(&qm);
    push_query_subscription(&mut qm, a, 1, query);
    let query = cheap(&qm);
    push_query_subscription(&mut qm, b, 1, query);
    let _ = confirmed_pass(&mut qm, &mut storage);
    // R_a
    let _ = confirmed_pass(&mut qm, &mut storage);
    // R_b: the cursor sits at B
    assert!(
        !qm.sync_manager().has_pending_query_subscriptions(),
        "fixture: both registrations ran"
    );
    assert_eq!(
        qm.stalled_units_for_test(),
        2,
        "fixture: both follow-up settles are class (i), stalled from birth"
    );
    let query = cheap(&qm);
    push_query_subscription(&mut qm, b, 2, query);
    let _ = confirmed_pass(&mut qm, &mut storage);
    assert!(
        qm.server_subscriptions.contains_key(&(b, QueryId(2))),
        "B's registration must be the pass's first unit, ahead of A's stalled settle"
    );
    assert!(
        !qm.sync_manager().has_pending_query_subscriptions(),
        "no registration is left behind a stalled unit"
    );
}

/// G6-11 (diff r1 S3). A local subscription that waits on the frontier behind a CONNECTED
/// upstream server runs once under the budget, makes no progress and stalls; the last server
/// going away is an un-stall event: the unit is served as a live unit within two passes,
/// ahead of a loud client that registers before every pass. Three confounders are held off
/// on purpose, each learned from a green falsification run (F6-9): the server is connected,
/// never pending (a pending server's departure flips `has_live_pending_servers`, G6-9's
/// hook); the scope-dirty un-stall is drained (`remove_server` marks the server's scopes
/// dirty); and the loud client supplies contention (a stalled unit is not excluded, only
/// last — alone it is served from the stalled tier anyway). Red under a manager that does
/// not poll `has_servers_or_pending_servers`: the key stays stalled behind the loud client's
/// registrations and the subscription does not settle inside the bound.
#[test]
fn the_last_server_leaving_un_stalls_the_local_units() {
    // Precise dirtiness pinned for the same reason as the first gate in this file: a
    // parallel force-user can otherwise tear this gate's write from its settle.
    let _precise = crate::query_manager::precise_dirty::force_precise_dirty(true);
    let (mut qm, mut storage) = seeded_server();
    let server_id = ServerId::new();
    qm.sync_manager_mut()
        .add_server_with_storage(server_id, true, &storage);
    assert!(
        !qm.sync_manager().has_live_pending_servers(),
        "fixture: the server is connected, so the pending-flip hook cannot fire"
    );
    let loud = client(&mut qm, &storage);
    // A required tier: the frontier is unsatisfied until the upstream server replays it, so
    // the wait behind the connected server is real (a tier-less subscription never waits).
    let sub_id = qm
        .subscribe_with_session(cheap(&qm), None, Some(DurabilityTier::GlobalServer))
        .expect("a local subscription over the seeded schema");
    qm.set_settle_budget_micros(Some(0));
    qm.process(&mut storage);
    assert!(
        !qm.subscriptions[&sub_id].settled_once,
        "fixture: the frontier wait behind the connected server keeps the subscription unsettled"
    );
    assert_eq!(
        qm.stalled_units_for_test(),
        1,
        "fixture: the unit made no progress and stalled"
    );
    qm.sync_manager_mut().remove_server(server_id);
    let _ = qm.sync_manager_mut().take_remote_query_scope_dirty();
    let mut settled_in = None;
    for round in 1..=6 {
        let query = expensive(&qm);
        push_query_subscription(&mut qm, loud, round, query);
        qm.process(&mut storage);
        if qm.subscriptions[&sub_id].settled_once {
            settled_in = Some(round);
            break;
        }
    }
    assert!(
        settled_in.is_some_and(|pass| pass <= 2),
        "the last server leaving un-stalls the local unit: served as a live unit within two \
         passes despite the loud client's registrations (settled in pass {settled_in:?})"
    );
    assert_eq!(
        qm.stalled_units_for_test(),
        0,
        "nothing is left stalled once the wait is gone"
    );
}

/// G6-12 (falsification F6-7b). A server subscription stalled from birth — class (i): the
/// authorization schema is required and missing — is un-stalled by the recompile that
/// follows `set_authorization_schema`, and settles within two passes even though a loud
/// client registers a fresh expensive query before every pass. Red under a recompile that
/// leaves the server key stalled: the unit stays in the stalled tier behind the loud
/// client's live registrations and never settles inside the bound.
#[test]
fn a_stalled_server_subscription_is_un_stalled_by_its_recompile() {
    // Precise dirtiness pinned for the same reason as the first gate in this file: a
    // parallel force-user can otherwise tear this gate's write from its settle.
    let _precise = crate::query_manager::precise_dirty::force_precise_dirty(true);
    let (mut qm, mut storage) = seeded_server();
    qm.require_authorization_schema();
    let quiet = client(&mut qm, &storage);
    let loud = client(&mut qm, &storage);
    qm.set_settle_budget_micros(Some(0));
    let query = cheap(&qm);
    push_query_subscription(&mut qm, quiet, 1, query);
    let _ = confirmed_pass(&mut qm, &mut storage);
    assert!(
        qm.server_subscriptions.contains_key(&(quiet, QueryId(1))),
        "fixture: the registration ran"
    );
    assert_eq!(
        qm.stalled_units_for_test(),
        1,
        "fixture: the follow-up settle is class (i), stalled from birth"
    );
    qm.set_authorization_schema(users_posts_schema());
    let mut history = Vec::new();
    for round in 1..=6 {
        let query = expensive(&qm);
        push_query_subscription(&mut qm, loud, round, query);
        history.push(confirmed_pass(&mut qm, &mut storage));
        if pass_of(&history, quiet, QueryId(1)).is_some() {
            break;
        }
    }
    assert!(
        pass_of(&history, quiet, QueryId(1)).is_some_and(|pass| pass <= 2),
        "the recompiled subscription must be served as a live unit within two passes of the \
         schema arriving, not from the stalled tier behind the loud client (served in pass {:?})",
        pass_of(&history, quiet, QueryId(1))
    );
    assert_eq!(
        qm.stalled_units_for_test(),
        0,
        "nothing is left stalled once the schema is there"
    );
}

/// G6-13 (diff r3 S5). The env knob: unset, empty, `0` and noise all mean "no budget"
/// (today's behaviour); a number is milliseconds, and the conversion saturates.
#[test]
fn the_settle_budget_env_knob_maps_zero_and_noise_to_unbounded() {
    assert_eq!(QueryManager::settle_budget_micros_from_env(None), None);
    assert_eq!(
        QueryManager::settle_budget_micros_from_env(Some("  ")),
        None
    );
    assert_eq!(QueryManager::settle_budget_micros_from_env(Some("0")), None);
    assert_eq!(
        QueryManager::settle_budget_micros_from_env(Some("soon")),
        None
    );
    assert_eq!(
        QueryManager::settle_budget_micros_from_env(Some(" 250 ")),
        Some(250_000)
    );
    assert_eq!(
        QueryManager::settle_budget_micros_from_env(Some(&u64::MAX.to_string())),
        Some(u64::MAX),
        "milliseconds to microseconds saturates instead of wrapping"
    );
}

/// Offer a query to `peer`, settle to quiescence, and confirm only the FIRST `confirm` rows.
/// Returns the row ids left unconfirmed — what the peer is owed.
fn offer_and_partially_confirm(
    qm: &mut QueryManager,
    storage: &mut MemoryStorage,
    peer: ClientId,
    query: crate::query_manager::query::Query,
    confirm: usize,
) -> std::collections::BTreeSet<crate::object::ObjectId> {
    let mut owed = std::collections::BTreeSet::new();
    let mut confirmed_so_far = 0usize;
    push_query_subscription(qm, peer, 1, query);
    for _ in 0..40 {
        qm.process(storage);
        let outbox = qm.sync_manager_mut().take_outbox();
        let mut to_confirm = Vec::new();
        for entry in &outbox {
            let (Destination::Client(c), SyncPayload::RowBatchNeeded { row, .. }) =
                (&entry.destination, &entry.payload)
            else {
                continue;
            };
            if *c != peer {
                continue;
            }
            if confirmed_so_far < confirm {
                confirmed_so_far += 1;
                to_confirm.push((
                    *c,
                    row.row_id,
                    crate::object::BranchName::new(row.branch.as_str()),
                    row.batch_id,
                ));
                owed.remove(&row.row_id);
            } else {
                owed.insert(row.row_id);
            }
        }
        qm.sync_manager_mut().confirm_client_deliveries(&to_confirm);
    }
    owed
}

/// The rows `peer` is re-offered when it replays query id 1, over `passes` passes.
fn rows_re_offered(
    qm: &mut QueryManager,
    storage: &mut MemoryStorage,
    peer: ClientId,
    query: crate::query_manager::query::Query,
) -> std::collections::BTreeSet<crate::object::ObjectId> {
    push_query_subscription(qm, peer, 1, query);
    let mut seen = std::collections::BTreeSet::new();
    for _ in 0..40 {
        qm.process(storage);
        for entry in qm.sync_manager_mut().take_outbox() {
            if let (Destination::Client(c), SyncPayload::RowBatchNeeded { row, .. }) =
                (&entry.destination, &entry.payload)
                && c == &peer
            {
                seen.insert(row.row_id);
            }
        }
    }
    seen
}

/// G6-8 (v18 item 6, diff r27). A returning peer's UNCONFIRMED rows are re-offered under every
/// budget — and only those rows.
///
/// This closes a hole that was total. The one term that recovers a returning peer is `owed_rows`
/// on the registration path (`server_queries.rs`, feeding "re-offering rows the peer never
/// confirmed" in `sync_manager/mod.rs`); the settle path has no such term. And
/// `client_has_undelivered_payloads` appears in no file in this crate that sets a settle budget:
/// every gate above confirms every delivery on every pass, so `owed_rows` is false in all of
/// them, and `delivery_confirmation_differential.rs`, which does drop payloads, sets no budget.
/// The recovery path and the pool scheduler had never run together in any test — which is why a
/// 25 s integration timeout was the first signal that anything connected them.
///
/// Two assertions, and the second is the one with teeth: the re-offer must carry the owed rows
/// and NOT the whole scope, so a future "just re-offer everything on reconnect" cannot pass this
/// by being wasteful. The fixture confirms 3 of the scope's rows and drops the rest, and asserts
/// on the sets rather than on counts.
///
/// Internal on purpose: `client_has_undelivered_payloads` and the outbox are engine bookkeeping.
/// From outside, a peer that is re-offered its owed rows and one that is re-sent its whole scope
/// converge to the same store; they differ in what they cost, which is item 6's entire subject.
#[test]
fn a_returning_peer_is_re_offered_exactly_its_unconfirmed_rows_under_every_budget() {
    for budget in [None, Some(0u64), Some(50_000u64), Some(3_600_000_000u64)] {
        let (mut qm, mut storage) = seeded_server();
        let peer = client(&mut qm, &storage);
        qm.set_settle_budget_micros(budget);
        let query = qm.query("users").limit(8).build();

        let owed = offer_and_partially_confirm(&mut qm, &mut storage, peer, query.clone(), 3);
        assert!(
            qm.sync_manager().client_has_undelivered_payloads(peer),
            "budget {budget:?}: fixture precondition — the peer must end up owing something, or \
             this gate observes nothing and passes vacuously"
        );
        assert!(
            !owed.is_empty(),
            "budget {budget:?}: fixture precondition — some rows must be left unconfirmed"
        );

        let re_offered = rows_re_offered(&mut qm, &mut storage, peer, query);
        assert!(
            owed.is_subset(&re_offered),
            "budget {budget:?}: a returning peer was NOT re-offered rows it never confirmed. \
             owed {} rows, re-offered {} — missing {:?}",
            owed.len(),
            re_offered.len(),
            owed.difference(&re_offered).count()
        );
        assert_eq!(
            re_offered.len(),
            owed.len(),
            "budget {budget:?}: the re-offer must be NARROWED to the owed rows. It carried {} \
             rows for {} owed — re-sending the whole scope on every reconnect is the cost item 6 \
             exists to bound, and it would satisfy the subset assertion above",
            re_offered.len(),
            owed.len()
        );
    }
}

/// G6-9 (v18 item 6). Fairness INSIDE one client's slot: a subscription that is re-dirtied every
/// single pass must not starve its sibling.
///
/// Every fairness gate above this one is cross-CLIENT — twenty clients with one subscription
/// each, or a loud client against a quiet one. The rotation they pin is the rotation over
/// clients. Nothing pinned what happens between two subscriptions of the SAME client, and that
/// is the shape the app actually has: one phone holds thirteen standing subscriptions, and one
/// of them (`messages` on the open chat) is dirtied by every inbound message while the others
/// wait. With item 6's default now ON, a pool that always picks the same unit inside a slot
/// would leave those twelve permanently unsettled on a busy chat — and every existing gate would
/// stay green.
///
/// Fixture: one client, `posts`-including subscription 1 and an all-`users` subscription 2, both
/// settled and confirmed. A user insert dirties both. From then on a post is inserted before
/// every pass, which re-dirties subscription 1 and never touches subscription 2. Under a zero
/// budget exactly one unit runs per pass, so a pool with no rotation inside the slot would spend
/// every pass on subscription 1.
///
/// Internal on purpose: which of a client's subscriptions a pass settles is engine scheduling.
/// From outside, the starved subscription's rows simply never arrive, which is what the
/// assertion says in the currency the client sees.
#[test]
#[ignore = "open defect (v18 item 6), found by this gate 2026-09-06: the pool's rotation cursor is over SLOTS, not units. `server_units.sort()` then `push_back` (server_queries.rs:2805-2811) gives each client's slot the same (ClientId, QueryId) order every pass, and `rotation_cursor` (:2849-2853, advanced :2978) only moves between slots — so when a pass runs one unit, the same subscription of that client is served forever and its siblings never settle. Control: green under `None`; red under `Some(0)`; green under `Some(50_000)` in THIS fixture only because both units fit in one pass, and prod has single units of ~0.46 s that do not. This is why item 6's default is back to `None`. Un-ignore with the fix."]
fn a_re_dirtied_subscription_does_not_starve_its_sibling_on_the_same_client() {
    // `None` is the control: the unbounded path settles both, so a red below is the pool's, not
    // the fixture's. The two bounded arms are the shipping question — `Some(0)` is one unit per
    // pass and `Some(50_000)` is the default.
    let mut starved = Vec::new();
    for budget in [None, Some(0u64), Some(50_000u64)] {
        if let Some(hot) = starvation_probe(budget) {
            starved.push(format!("{budget:?} (sibling settled {hot}x, quiet 0x)"));
        }
    }
    assert!(
        starved.is_empty(),
        "the quiet subscription of the SAME client never settled in 12 passes, at: {}. Its \
         sibling was re-dirtied before every pass and nothing else changed. That is starvation \
         inside one client's slot, and no other gate in this file can see it — every fairness \
         gate above rotates over CLIENTS",
        starved.join(", ")
    );
}

/// `Some(hot_settles)` if the quiet subscription never settled; `None` if it did.
fn starvation_probe(budget: Option<u64>) -> Option<usize> {
    let _precise = crate::query_manager::precise_dirty::force_precise_dirty(true);
    let (mut qm, mut storage) = seeded_server();
    let peer = client(&mut qm, &storage);

    let hot = expensive(&qm);
    let quiet = qm.query("users").build();
    push_query_subscription(&mut qm, peer, 1, hot);
    push_query_subscription(&mut qm, peer, 2, quiet);
    let mut settled = 0;
    for _ in 0..40 {
        settled += confirmed_pass(&mut qm, &mut storage).len();
        if settled >= 2 {
            break;
        }
    }
    assert_eq!(
        settled, 2,
        "budget {budget:?} fixture: both subscriptions must settle once before the starvation \
         phase"
    );

    qm.set_settle_budget_micros(budget);
    // Dirties BOTH: subscription 2 reads `users`, subscription 1 reads it as the include's root.
    qm.insert(
        &mut storage,
        "users",
        &[Value::Integer(9_001), Value::Text("late-user".into())],
    )
    .unwrap();

    let mut quiet_settled_at = None;
    let mut hot_settles = 0usize;
    for pass_no in 1..=12 {
        // Re-dirty the hot subscription and nothing else: a post is outside subscription 2's
        // query entirely.
        qm.insert(
            &mut storage,
            "posts",
            &[
                Value::Integer(900_000 + pass_no),
                Value::Text(format!("hot-{pass_no}")),
                Value::Integer(1),
            ],
        )
        .unwrap();
        for (_, query_id) in confirmed_pass(&mut qm, &mut storage) {
            match query_id.0 {
                1 => hot_settles += 1,
                2 if quiet_settled_at.is_none() => quiet_settled_at = Some(pass_no),
                _ => {}
            }
        }
        if quiet_settled_at.is_some() {
            break;
        }
    }

    quiet_settled_at.is_none().then_some(hot_settles)
}

/// G6-10 (diff r28). The budget is in MICROseconds, and nothing else in the crate says so.
///
/// `set_settle_budget_micros` takes micros, `settle_budget_for_server` multiplies its
/// milliseconds by 1 000, and `SettleClock::new` turns the number into a `Duration` with
/// `Duration::from_micros` (`manager.rs:574`). Every other gate for this item passes either
/// `Some(0)` — identical under `from_micros` and `from_millis` — or a budget so large that no
/// pass could ever reach it. So `from_micros → from_millis` is a thousandfold change to prod's
/// pass bound that the whole suite reports as green: a 50 ms budget silently becomes 50 s, which
/// is "unbounded" at the scale of the passes this item exists to bound.
///
/// The margin is deliberately 1 000×: the clock is given 1 ms and then handed 20 ms of real
/// time, so the assertion is nowhere near a scheduling race in either direction.
///
/// Internal on purpose: the unit of an engine-internal deadline has no public surface, and from
/// outside a 50 ms and a 50 s pass bound differ only in how long a read waits.
#[test]
fn the_settle_budget_is_measured_in_microseconds() {
    use crate::query_manager::manager::SettleClock;

    let mut spent = SettleClock::new(Some(1_000));
    assert!(
        spent.may_run_unit(),
        "the first unit of a pass always runs, whatever the budget"
    );
    spent.note_unit_ran();
    std::thread::sleep(std::time::Duration::from_millis(20));
    assert!(
        !spent.may_run_unit(),
        "a budget of 1 000 is one MILLIsecond, and 20 ms have passed — this clock must be \
         spent. If it is not, the budget is being read as milliseconds and every configured \
         value is a thousand times larger than the operator asked for"
    );

    let mut roomy = SettleClock::new(Some(60_000_000));
    roomy.note_unit_ran();
    std::thread::sleep(std::time::Duration::from_millis(20));
    assert!(
        roomy.may_run_unit(),
        "and the other direction, so the gate cannot be satisfied by a clock that is simply \
         always spent: 60 s of budget is not exhausted by 20 ms of sleeping"
    );

    let mut unbounded = SettleClock::new(None);
    unbounded.note_unit_ran();
    assert!(
        unbounded.may_run_unit(),
        "`None` is unbounded and stays unbounded — that is prod's rollback"
    );
}
