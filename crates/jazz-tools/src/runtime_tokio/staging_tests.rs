//! Gates for the inbound staging area (v18 item 3, "transport off the lock").
//!
//! Internal on purpose: the notify chain between a refused pusher and the tick that drains
//! its client, and the per-connection order entries are parked in, are properties of the
//! staging structure under a held engine lock. No public builder exposes the instant a push
//! is refused or the order of parking; the black-box twins live in
//! `tests/transport_off_the_lock.rs` and `tests/transport_frames.rs`.
//!
//! ```text
//!   socket task ──push──▶ [staging] ◀──drain── jazz-tick thread ──park──▶ core (locked)
//!        ▲ refused (cap)      │ notify                ▲
//!        └────── waiter ◀─────┘                 pass thread holds the lock 0–20 ms
//! ```

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::{StagePush, StagingConfig, TokioRuntime};
use crate::query_manager::types::{ColumnType, Schema, SchemaBuilder, TableSchema};
use crate::schema_manager::{AppId, SchemaManager};
use crate::storage::MemoryStorage;
use crate::sync_manager::sync_tracer::SyncTracer;
use crate::sync_manager::types::{InboxEntry, QueryId, Source, SyncPayload};
use crate::sync_manager::{ClientId, SyncManager};

type Runtime = Arc<TokioRuntime<MemoryStorage>>;

fn schema() -> Schema {
    SchemaBuilder::new()
        .table(
            TableSchema::builder("users")
                .column("id", ColumnType::Uuid)
                .column("name", ColumnType::Text),
        )
        .build()
}

fn runtime(config: StagingConfig) -> Runtime {
    let schema_manager = SchemaManager::new(
        SyncManager::new(),
        schema(),
        AppId::from_name("staging-gates"),
        "dev",
        "main",
    )
    .expect("schema manager");
    Arc::new(TokioRuntime::new_with_staging(
        schema_manager,
        MemoryStorage::new(),
        |_| {},
        config,
    ))
}

/// A client frame the server accepts from any client and applies without side effects: an
/// unsubscription for a query nobody registered. The tag rides in the query id.
fn tagged(client: ClientId, tag: u64) -> InboxEntry {
    InboxEntry {
        source: Source::Client(client),
        payload: SyncPayload::QueryUnsubscription {
            query_id: QueryId(tag),
        },
    }
}

/// The stand-in for a long settle pass: a plain thread holds the engine lock until released.
struct LockHold {
    release: Option<std::sync::mpsc::Sender<()>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl LockHold {
    fn take(runtime: &Runtime) -> Self {
        let (release, released) = std::sync::mpsc::channel::<()>();
        let held = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&held);
        let runtime = Arc::clone(runtime);
        let thread = std::thread::spawn(move || {
            runtime.inspect_core_for_test(|_| {
                flag.store(true, Ordering::SeqCst);
                let _ = released.recv();
            });
        });
        while !held.load(Ordering::SeqCst) {
            std::thread::yield_now();
        }
        Self {
            release: Some(release),
            thread: Some(thread),
        }
    }

    fn release(mut self) {
        drop(self.release.take());
        if let Some(thread) = self.thread.take() {
            thread.join().expect("the pass thread ends");
        }
    }
}

async fn wait_until(what: &str, deadline: Duration, mut predicate: impl FnMut() -> bool) {
    let started = Instant::now();
    while !predicate() {
        assert!(
            started.elapsed() < deadline,
            "waited {deadline:?} for {what} without it happening"
        );
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
}

/// G-notify. A pusher refused while the lock is held counts as a waiter from that moment,
/// even before it starts waiting. Its client's entry — and the `Notify` on it — must
/// outlive it: the drain that empties the entry keeps it while waiters remain and leaves a
/// `notify_one` permit, so a waiter that only creates its `notified()` after that drain is
/// still woken. Disarm: remove the entry at count zero regardless of waiters → B waits on
/// a `Notify` no later drain signals → timeout.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_refused_pusher_that_starts_waiting_late_is_still_woken_by_the_drain() {
    let runtime = runtime(StagingConfig {
        max_entries_per_client: 1,
        max_bytes_per_client: 1 << 20,
        inflight_budget_bytes: 1 << 30,
    });
    let tracer = SyncTracer::new();
    runtime.set_sync_tracer(tracer.clone(), "server".to_string());
    let client = ClientId::new();
    tracer.register_client(client, "client");
    runtime
        .ensure_client_as_backend(client)
        .expect("client registered");

    let hold = LockHold::take(&runtime);
    // P0 fills the one-entry cap; A and B are refused behind it and count as waiters.
    assert!(matches!(
        runtime.stage_sync_inbox(client, vec![tagged(client, 0)], 16, None),
        StagePush::Staged
    ));
    let StagePush::Backpressure {
        entries: a_entries,
        permit: a_permit,
        waiter: a_waiter,
    } = runtime.stage_sync_inbox(client, vec![tagged(client, 1)], 16, None)
    else {
        panic!("A must be refused by the cap")
    };
    let StagePush::Backpressure {
        entries: b_entries,
        permit: b_permit,
        waiter: b_waiter,
    } = runtime.stage_sync_inbox(client, vec![tagged(client, 2)], 16, None)
    else {
        panic!("B must be refused by the cap")
    };
    assert_eq!(runtime.staging_stats_for_test(client).waiters, 2);

    hold.release();
    // Drain 1: the tick thread parks P0; the entry stays because two waiters remain.
    wait_until("drain 1", Duration::from_secs(5), || {
        let stats = runtime.staging_stats_for_test(client);
        stats.count == 0 && stats.staged_frames == 1
    })
    .await;
    assert_eq!(
        runtime.staging_live_clients_for_test(),
        1,
        "the entry must outlive its waiters"
    );
    // A wakes, re-pushes, and its frame drains: a second drain for the same client.
    a_waiter.notified().await;
    assert!(matches!(
        runtime.stage_sync_inbox_with_waiter(a_waiter, client, a_entries, 16, a_permit),
        StagePush::Staged
    ));
    wait_until("drain 2", Duration::from_secs(5), || {
        let stats = runtime.staging_stats_for_test(client);
        stats.count == 0 && stats.staged_frames == 2
    })
    .await;
    // Only now does B create its `notified()`: it slept through both drains. The permit the
    // second drain stored on the client's `Notify` must wake it.
    tokio::time::timeout(Duration::from_secs(2), b_waiter.notified())
        .await
        .expect("B must be woken by a drain that ran before it started waiting");
    assert!(matches!(
        runtime.stage_sync_inbox_with_waiter(b_waiter, client, b_entries, 16, b_permit),
        StagePush::Staged
    ));
    wait_until("drain 3", Duration::from_secs(5), || {
        let stats = runtime.staging_stats_for_test(client);
        stats.count == 0 && stats.staged_frames == 3
    })
    .await;

    // Every frame parked exactly once, in push order; nothing left behind.
    let tags: Vec<u64> = tracer
        .from("client")
        .iter()
        .filter_map(|message| message.query_id())
        .map(|query_id| query_id.0)
        .collect();
    assert_eq!(tags, vec![0, 1, 2]);
    assert_eq!(
        runtime.staging_live_clients_for_test(),
        0,
        "nothing staged, nobody waiting: the entry is gone"
    );
}

/// The differential oracle for the staging area: N connections over K clients push frames
/// of random sizes around the caps while a pass thread holds the lock at random, some
/// pushers drop their `notified()` mid-wait, and some sockets "close" with a frame on hold.
/// The model is a per-connection FIFO delivered exactly once:
///
/// - I1 per-connection FIFO: a connection's tags are parked in the order it pushed them;
/// - I2 exactly once: the multiset of parked tags equals the multiset of pushed tags;
/// - I3 quiescence: everything is parked with no test-side flush;
/// - I4 no leak: every counter is zero, no live entry, the budget is whole;
/// - I5 bounded staging: a client's peak never exceeds max(cap, its largest frame) plus
///   the frames its sockets staged uncapped on exit (the exit path stages a held frame
///   regardless of the cap — one frame per closing socket, never more). The bound counts
///   every exit frame of the run whether or not it was still staged at the peak: loose by
///   construction (the peak cannot be attributed to particular exit frames), tight for the
///   cap itself.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn staging_keeps_per_connection_order_and_delivers_exactly_once_under_random_holds() {
    const CONNECTIONS: usize = 6;
    const CLIENTS: usize = 2;
    const FRAMES_PER_CONNECTION: usize = 40;
    const ENTRIES_CAP: usize = 4;
    const BYTES_CAP: usize = ENTRIES_CAP * 1024;

    let seed = std::env::var("JAZZ_STAGING_SEED")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or_else(rand::random::<u64>);
    eprintln!("staging differential seed = {seed} (JAZZ_STAGING_SEED to replay)");

    let runtime = runtime(StagingConfig {
        max_entries_per_client: ENTRIES_CAP,
        max_bytes_per_client: BYTES_CAP,
        inflight_budget_bytes: 8 * 1024,
    });
    let tracer = SyncTracer::new();
    runtime.set_sync_tracer(tracer.clone(), "server".to_string());
    let clients: Vec<ClientId> = (0..CLIENTS)
        .map(|index| {
            let client = ClientId::new();
            tracer.register_client(client, format!("client{index}"));
            runtime
                .ensure_client_as_backend(client)
                .expect("client registered");
            client
        })
        .collect();
    let budget = runtime.inflight_budget();
    let whole_budget = budget.available_permits();
    let pushed: Arc<Mutex<Vec<(usize, u64)>>> = Arc::new(Mutex::new(Vec::new()));
    let abandoned = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    // Bytes each client staged through the exit path, for I5's bound.
    let uncapped: Arc<Mutex<std::collections::HashMap<ClientId, usize>>> =
        Arc::new(Mutex::new(std::collections::HashMap::new()));

    // The pass thread: random holds with random gaps until the pushers are done.
    let stop = Arc::new(AtomicBool::new(false));
    let pass = {
        let runtime = Arc::clone(&runtime);
        let stop = Arc::clone(&stop);
        let mut rng = StdRng::seed_from_u64(seed ^ 0xA5A5_5A5A);
        std::thread::spawn(move || {
            while !stop.load(Ordering::SeqCst) {
                let hold = rng.gen_range(0..=20);
                runtime.inspect_core_for_test(|_| {
                    std::thread::sleep(Duration::from_millis(hold));
                });
                std::thread::sleep(Duration::from_millis(rng.gen_range(0..=5)));
            }
        })
    };

    let mut pushers = Vec::new();
    for connection in 0..CONNECTIONS {
        let client = clients[connection % CLIENTS];
        let runtime = Arc::clone(&runtime);
        let budget = Arc::clone(&budget);
        let pushed = Arc::clone(&pushed);
        let uncapped = Arc::clone(&uncapped);
        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(connection as u64));
        // Odd connections drop their `notified()` mid-wait and create a new one.
        let drops_mid_wait = connection % 2 == 1;
        // Every third connection abandons its budget acquire mid-wait (a socket that
        // leaves while queued): the dropped acquire must return whatever permits it held.
        let abandons_acquires = connection % 3 == 2;
        // And every third from the other residue EXITS holding an admitted frame, on the first
        // refusal it meets: the socket goes away with a decoded frame it already charged, which
        // has to be staged uncapped and its waiter released. Deterministic, like the mode above
        // and unlike its predecessor — that was `last && rng.gen_bool(0.5)`, needing the LAST
        // frame to be refused AND a coin flip, and a full-suite run measured it never firing.
        let exits_holding = connection % 3 == 1;
        let abandoned = Arc::clone(&abandoned);
        pushers.push(tokio::spawn(async move {
            let mut mine = Vec::new();
            'frames: for index in 0..FRAMES_PER_CONNECTION {
                let entries_in_frame = rng.gen_range(1..=ENTRIES_CAP + 1);
                let entries: Vec<InboxEntry> = (0..entries_in_frame)
                    .map(|k| {
                        let tag =
                            ((connection as u64) << 20) | ((index * (ENTRIES_CAP + 1) + k) as u64);
                        mine.push((connection, tag));
                        tagged(client, tag)
                    })
                    .collect();
                let bytes = entries_in_frame * 1024;
                let permit = if abandons_acquires {
                    loop {
                        let patience = Duration::from_micros(rng.gen_range(50..=500));
                        match tokio::time::timeout(
                            patience,
                            Arc::clone(&budget).acquire_many_owned(entries_in_frame as u32),
                        )
                        .await
                        {
                            Ok(permit) => break permit.expect("the budget is open"),
                            Err(_) => {
                                abandoned.fetch_add(1, Ordering::SeqCst);
                                tokio::task::yield_now().await;
                            }
                        }
                    }
                } else {
                    Arc::clone(&budget)
                        .acquire_many_owned(entries_in_frame as u32)
                        .await
                        .expect("the budget is open")
                };
                let mut push = runtime.stage_sync_inbox(client, entries, bytes, Some(permit));
                loop {
                    match push {
                        StagePush::Staged => break,
                        StagePush::Backpressure {
                            entries,
                            permit,
                            waiter,
                        } => {
                            if exits_holding {
                                // The socket closed with a frame on hold: it goes in
                                // uncapped and still counts once. The connection is gone
                                // afterwards — it cannot push its remaining frames, which is
                                // why this leaves the frame loop and not just the retry loop.
                                runtime.stage_sync_inbox_uncapped(
                                    Some(waiter),
                                    client,
                                    entries,
                                    bytes,
                                    permit,
                                );
                                *uncapped
                                    .lock()
                                    .expect("uncapped")
                                    .entry(client)
                                    .or_insert(0) += bytes;
                                break 'frames;
                            }
                            if drops_mid_wait {
                                loop {
                                    let patience = Duration::from_millis(rng.gen_range(1..=3));
                                    if tokio::time::timeout(patience, waiter.notified())
                                        .await
                                        .is_ok()
                                    {
                                        break;
                                    }
                                }
                            } else {
                                waiter.notified().await;
                            }
                            push = runtime.stage_sync_inbox_with_waiter(
                                waiter, client, entries, bytes, permit,
                            );
                        }
                    }
                }
                if rng.gen_bool(0.3) {
                    tokio::task::yield_now().await;
                }
                if rng.gen_bool(0.1) {
                    tokio::time::sleep(Duration::from_millis(rng.gen_range(0..=2))).await;
                }
            }
            pushed.lock().expect("pushed").extend(mine);
        }));
    }
    for pusher in pushers {
        pusher.await.expect("a pusher panicked");
    }
    let mut expected = pushed.lock().expect("pushed").clone();
    let total = expected.len();

    // I3: quiescence without a test-side flush — the tick thread must drain everything.
    let parked = |tracer: &SyncTracer| -> Vec<(usize, u64)> {
        tracer
            .messages()
            .iter()
            .filter_map(|message| match message.payload {
                SyncPayload::QueryUnsubscription { query_id } => Some(query_id.0),
                _ => None,
            })
            .map(|tag| ((tag >> 20) as usize, tag))
            .collect()
    };
    wait_until(
        "every pushed frame to be parked",
        Duration::from_secs(20),
        || parked(&tracer).len() >= total,
    )
    .await;
    stop.store(true, Ordering::SeqCst);
    pass.join().expect("the pass thread ends");

    let observed = parked(&tracer);
    // I2: exactly once.
    let mut observed_sorted = observed.clone();
    observed_sorted.sort_unstable();
    expected.sort_unstable();
    assert_eq!(
        observed_sorted, expected,
        "seed {seed}: the parked tags are not the pushed tags exactly once"
    );
    // I1: per-connection FIFO (tags grow within a connection).
    for connection in 0..CONNECTIONS {
        let order: Vec<u64> = observed
            .iter()
            .filter(|(owner, _)| *owner == connection)
            .map(|(_, tag)| *tag)
            .collect();
        assert!(
            order.windows(2).all(|pair| pair[0] < pair[1]),
            "seed {seed}: connection {connection} was parked out of push order: {order:?}"
        );
    }
    // I4: no leak.
    for (index, client) in clients.iter().enumerate() {
        let stats = runtime.staging_stats_for_test(*client);
        assert_eq!(
            stats.count, 0,
            "seed {seed}: client{index} still has staged entries"
        );
        assert_eq!(
            stats.bytes, 0,
            "seed {seed}: client{index} still has staged bytes"
        );
        assert_eq!(
            stats.waiters, 0,
            "seed {seed}: client{index} still has waiters"
        );
        // I5: bounded staging.
        let exit_bytes = uncapped
            .lock()
            .expect("uncapped")
            .get(client)
            .copied()
            .unwrap_or(0);
        assert!(
            stats.peak_bytes <= BYTES_CAP.max(stats.largest_frame_bytes) + exit_bytes,
            "seed {seed}: client{index} staged {} bytes at peak, over max(cap {BYTES_CAP}, \
             largest frame {}) + {exit_bytes} staged on socket exit",
            stats.peak_bytes,
            stats.largest_frame_bytes
        );
    }
    assert_eq!(
        runtime.staging_live_clients_for_test(),
        0,
        "seed {seed}: a client entry survived quiescence"
    );
    assert_eq!(
        budget.available_permits(),
        whole_budget,
        "seed {seed}: the decoded-bytes budget leaked permits"
    );
    assert!(
        abandoned.load(Ordering::SeqCst) > 0,
        "seed {seed}: fixture — no acquire timed out mid-wait (holding permits or not), the mode did not run"
    );
    assert!(
        !uncapped.lock().expect("uncapped").is_empty(),
        "seed {seed}: fixture — no socket exited holding an admitted frame, so the uncapped exit \
         path did not run — which now means the mode itself is broken, since `exits_holding` \
         is deterministic. It was `last && rng.gen_bool(0.5)`, and this assertion caught that \
         form missing the branch entirely in a full-suite run"
    );
}

/// G-exit (v18 item 3). A socket that exits still holding an ADMITTED frame must stage that
/// frame — entries and all — regardless of the per-client caps.
///
/// This gate exists because chain row D7 came back GREEN — but NOT for the reason first written
/// here. The retracted claim was that `push_uncapped` had one call site and that no test reached
/// it; both were false when written. The staging differential has called it all along, at
/// `staging_tests.rs:347`, and in the production shape at that. Round 12 measured the actual
/// mechanism: that call sits behind `last && rng.gen_bool(0.5)` on a seed that is random by
/// default, and — unlike the differential's other stochastic mode, which asserts `abandoned > 0`
/// — nothing asserted the branch had run. In both measured runs it did not.
///
/// So the gap was stochastic coverage, not an unreached path, and item 3's "delivers exactly
/// once" claim rested on a coin flip for the one path where losing a frame is SILENT — the frame
/// is already admitted, its budget already charged, and the connection is gone, so there is no
/// socket left to notice the loss and no client to retry it. The missing precondition is now
/// asserted beside its twin; this gate makes the no-waiter shape deterministic; and
/// `the_exit_path_in_its_production_shape_...` below covers the shape production actually uses.
///
/// The cap is set to one entry and filled first, so this is the exit path in the state that
/// makes it *uncapped* rather than merely another push: the assertion that the count ends at
/// three under a cap of one is what proves the caps were bypassed on purpose.
///
/// Internal on purpose: the observable is the staging entry's count and byte total, which is
/// runtime state no client API exposes. From outside, a frame dropped on socket exit and a
/// frame the client never finished sending look identical.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_socket_that_exits_holding_an_admitted_frame_still_stages_it() {
    let runtime = runtime(StagingConfig {
        max_entries_per_client: 1,
        max_bytes_per_client: 1 << 20,
        inflight_budget_bytes: 1 << 30,
    });
    let tracer = SyncTracer::new();
    runtime.set_sync_tracer(tracer.clone(), "server".to_string());
    let client = ClientId::new();
    tracer.register_client(client, "client");
    runtime
        .ensure_client_as_backend(client)
        .expect("client registered");

    // The lock hold keeps a tick from draining the staging between the pushes; without it the
    // counts below would be measuring a race, not the exit path.
    let hold = LockHold::take(&runtime);
    assert!(
        matches!(
            runtime.stage_sync_inbox(client, vec![tagged(client, 0)], 16, None),
            StagePush::Staged
        ),
        "fixture precondition: the first push fills the one-entry cap"
    );

    let before = runtime.staging_stats_for_test(client);
    runtime.stage_sync_inbox_uncapped(
        None,
        client,
        vec![tagged(client, 1), tagged(client, 2)],
        32,
        None,
    );
    let after = runtime.staging_stats_for_test(client);
    hold.release();

    assert_eq!(
        after.count - before.count,
        2,
        "the exiting socket's frame must be staged with its entries: D7 empties them here"
    );
    assert_eq!(
        after.bytes - before.bytes,
        32,
        "the frame's decoded bytes must be accounted, or the drain returns the wrong budget"
    );
    assert_eq!(
        after.count, 3,
        "and staged REGARDLESS of the caps: three entries under a cap of one is the whole \
         point of the uncapped path — the connection is gone and cannot push more"
    );
}

/// G-exit-prod (v18 item 3). The exit path in the shape PRODUCTION uses: the waiter guard the
/// refusal handed back, and that refusal's permit.
///
/// G-exit above pins the `None, .., None` shape, which is nobody's. The only production call
/// site is `websocket.rs`'s loop-exit arm, reached only from `Pending::AwaitingStaging`, so it
/// always carries `Some(waiter)` and the permit the `Backpressure` returned. That means the
/// `Some(waiter)` arm of `push_uncapped` — `notify.take()`, `staging.take()`, `mem::forget`,
/// `release_waiter` — was reached only by the differential's stochastic exit branch, which
/// round 12 measured missing twice in a row.
///
/// What breaks when `release_waiter` is skipped there is not the frame — it is the entry. The
/// client's `waiters` never returns to zero, so `blocks_removal` answers `true` for the rest of
/// the process and `remove_client` refuses that client forever: a permanent per-client leak, on
/// the path where the connection is already gone and nothing is left to notice.
///
/// Internal on purpose: `waiters`, the live-entry count and the budget's permits are runtime
/// state no client API exposes. From outside, a client that can never be reaped looks exactly
/// like a client that is simply still connected.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_exit_path_in_its_production_shape_releases_the_waiter_and_the_permit() {
    let runtime = runtime(StagingConfig {
        max_entries_per_client: 1,
        max_bytes_per_client: 1 << 20,
        inflight_budget_bytes: 8 * 1024,
    });
    let tracer = SyncTracer::new();
    runtime.set_sync_tracer(tracer.clone(), "server".to_string());
    let client = ClientId::new();
    tracer.register_client(client, "client");
    runtime
        .ensure_client_as_backend(client)
        .expect("client registered");
    let budget = runtime.inflight_budget();
    let whole_budget = budget.available_permits();

    let hold = LockHold::take(&runtime);
    assert!(
        matches!(
            runtime.stage_sync_inbox(client, vec![tagged(client, 0)], 16, None),
            StagePush::Staged
        ),
        "fixture precondition: the first push fills the one-entry cap"
    );
    let charged = Arc::clone(&budget)
        .acquire_many_owned(1)
        .await
        .expect("the budget is open");
    let StagePush::Backpressure {
        entries,
        permit,
        waiter,
    } = runtime.stage_sync_inbox(client, vec![tagged(client, 1)], 16, Some(charged))
    else {
        panic!("fixture precondition: the second push must be refused by the one-entry cap")
    };
    assert_eq!(
        runtime.staging_stats_for_test(client).waiters,
        1,
        "fixture precondition: a refusal registers the socket as a waiter"
    );

    // The socket exits still holding the admitted frame — `websocket.rs`'s loop-exit arm, with
    // the waiter and the permit it owns at that moment.
    runtime.stage_sync_inbox_uncapped(Some(waiter), client, entries, 16, permit);

    let staged = runtime.staging_stats_for_test(client);
    assert_eq!(
        staged.waiters, 0,
        "the exiting socket's waiter guard must be released under the same lock that stages its \
         frame: leave it and `blocks_removal` answers true forever, so this client is never \
         reaped for the life of the process"
    );
    assert_eq!(
        staged.count, 2,
        "and its frame is staged with its entries, regardless of the one-entry cap"
    );

    hold.release();
    wait_until(
        "the exiting client's entry to drain and be reaped",
        Duration::from_secs(5),
        || runtime.staging_live_clients_for_test() == 0,
    )
    .await;
    assert_eq!(
        budget.available_permits(),
        whole_budget,
        "and the permit it carried returns to the budget: the drain releases what `store_locked` \
         forgot, so an exit neither leaks a permit nor releases one twice"
    );
}
