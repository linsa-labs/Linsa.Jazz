//! v18 item 3, gate G3′ (the security gate): a client's frames are ingested — and the
//! server's I/O stays alive — while another pass holds the engine lock.
//!
//! Today the socket task handling an inbound frame takes the core `std::sync::Mutex` inline
//! (`server/mod.rs` `process_ws_client_frame` → `runtime_tokio.rs` `push_sync_inbox`), so
//! for the length of any settle pass that task blocks its tokio worker thread — its own
//! outbound, its heartbeat, and with enough such tasks the whole runtime. This test runs on a
//! current-thread runtime on purpose: with one worker, one blocked socket task is observable
//! as a server that cannot answer `/health`.
#![cfg(feature = "test")]

mod support;

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use jazz_tools::JazzClient;
use jazz_tools::query_manager::query::QueryBuilder;
use jazz_tools::query_manager::types::{ColumnType, Schema, SchemaBuilder, TableSchema, Value};
use jazz_tools::runtime_tokio::StagingConfig;
use jazz_tools::server::JazzServer;
use jazz_tools::sync_manager::{DurabilityTier, QueryId, SyncPayload};
use jazz_tools::sync_tracer::SyncTracer;

fn notes_schema() -> Schema {
    SchemaBuilder::new()
        .table(TableSchema::builder("notes").column("body", ColumnType::Text))
        .build()
}

async fn user(server: &JazzServer, name: &str) -> JazzClient {
    JazzClient::connect(server.make_client_context_for_user(notes_schema(), name))
        .await
        .expect("user connects")
}

/// How long the "pass" holds the engine lock. Long enough that a socket task blocked on it
/// is unmistakable against the budget below, short enough to keep the test quick.
const PASS_HOLD: Duration = Duration::from_millis(1500);
/// What the whole "frame in flight + health probe" sequence may take when ingestion does not
/// wait for the lock.
const LIVE_BUDGET: Duration = Duration::from_millis(600);

/// G3′. A client's frame arrives while a settle pass holds the engine lock: the server keeps
/// answering `/health` inside `LIVE_BUDGET`, and the frame is applied once the pass ends.
/// Internal on purpose: holding the lock for a controlled time needs `server_state()` and
/// `inspect_core_for_test`; the public API cannot make a pass last `PASS_HOLD`.
///
/// ```text
///   test thread            engine lock            bob's socket task        /health
///   ───────────            ───────────            ─────────────────        ───────
///   hold_lock ──────────▶ HELD (PASS_HOLD)
///   bob.insert ─────────────────────────────────▶ frame decoded, staged
///                                                 (no lock taken)
///   GET /health ──────────────────────────────────────────────────────▶ 200 < LIVE_BUDGET
///   release ────────────▶ FREE ──▶ tick drains the staging ──▶ alice receives bob's row
/// ```
#[tokio::test]
async fn a_client_frame_in_flight_does_not_stall_the_server_while_a_pass_holds_the_lock() {
    let server = JazzServer::builder()
        .with_schema(notes_schema())
        .start()
        .await;
    let alice = user(&server, "alice").await;
    let bob = user(&server, "bob").await;
    let mut alice_stream = alice
        .subscribe(QueryBuilder::new("notes").build())
        .await
        .expect("alice subscribes");
    // Let the subscription settle before the pass starts, so the only inbound frame during
    // the hold is bob's write.
    tokio::time::sleep(Duration::from_millis(200)).await;

    // The pass: a plain thread holds the core lock, exactly as a long settle would.
    let state = server.server_state();
    let pass = std::thread::spawn(move || {
        state
            .runtime
            .inspect_core_for_test(|_| std::thread::sleep(PASS_HOLD));
    });
    // Give the thread the lock before the frame goes out.
    std::thread::sleep(Duration::from_millis(50));

    let started = Instant::now();
    let (_, _, batch) = bob
        .insert(
            "notes",
            HashMap::from([("body".to_string(), Value::Text("hello".to_string()))]),
        )
        .expect("bob writes locally");
    // Park the test so the I/O driver runs and the server's socket task receives the frame.
    tokio::time::sleep(Duration::from_millis(100)).await;
    let health = reqwest::Client::new()
        .get(format!("{}/health", server.base_url()))
        .send()
        .await
        .expect("health request completes");
    let elapsed = started.elapsed();
    assert!(
        health.status().is_success(),
        "health must answer: {:?}",
        health.status()
    );
    assert!(
        elapsed < LIVE_BUDGET,
        "with a client frame in flight the server took {elapsed:?} to answer /health while \
         a pass held the engine lock; the socket task must stage the frame and return, not \
         wait on the lock (budget {LIVE_BUDGET:?}, hold {PASS_HOLD:?})"
    );

    // The frame was staged, not lost: once the pass ends it is applied and fans out.
    pass.join().expect("the pass thread ends");
    tokio::time::timeout(
        Duration::from_secs(10),
        bob.wait_for_batch(batch, DurabilityTier::EdgeServer),
    )
    .await
    .expect("bob's write reaches the server within 10 s of the pass ending")
    .expect("bob's write reaches the server after the pass");
    let delivered = tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            match alice_stream.next().await {
                Some(delta) if !delta.is_empty() => break true,
                Some(_) => continue,
                None => break false,
            }
        }
    })
    .await
    .expect("alice receives bob's row after the pass");
    assert!(delivered, "alice's stream ended before the row arrived");
    server.shutdown().await;
}

/// The pass, held until released: a plain thread holds the engine lock exactly as a long
/// settle would, for as long as the gate needs.
struct LockHold {
    release: Option<std::sync::mpsc::Sender<()>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

/// A plain-thread watchdog for gates whose disarm freezes the test runtime itself: exits the
/// process (red) unless the gate finished first. Never fires when the mechanism is armed.
struct Watchdog {
    done: Arc<AtomicBool>,
}

impl Watchdog {
    fn arm(patience: Duration, gate: &'static str) -> Self {
        let done = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&done);
        std::thread::spawn(move || {
            std::thread::sleep(patience);
            if !flag.load(Ordering::SeqCst) {
                // Straight to the process's stderr: a thread spawned from a test inherits
                // libtest's output capture, and `eprintln!` would go into a buffer the exit
                // below never prints.
                use std::io::Write as _;
                let _ = writeln!(
                    std::io::stderr(),
                    "{gate}: the test runtime made no progress for {patience:?} — something \
                     waited on the engine lock ON the I/O worker"
                );
                std::process::exit(101);
            }
        });
        Self { done }
    }
}

impl Drop for Watchdog {
    fn drop(&mut self) {
        self.done.store(true, Ordering::SeqCst);
    }
}

fn hold_lock(server: &JazzServer) -> LockHold {
    let state = server.server_state();
    let (release, released) = std::sync::mpsc::channel::<()>();
    let held = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&held);
    let thread = std::thread::spawn(move || {
        state.runtime.inspect_core_for_test(|_| {
            flag.store(true, Ordering::SeqCst);
            let _ = released.recv();
        });
    });
    while !held.load(Ordering::SeqCst) {
        std::thread::sleep(Duration::from_millis(1));
    }
    LockHold {
        release: Some(release),
        thread: Some(thread),
    }
}

impl LockHold {
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
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

fn note(body: &str) -> HashMap<String, Value> {
    HashMap::from([("body".to_string(), Value::Text(body.to_string()))])
}

/// Read alice's stream until `rows` rows have arrived (or the deadline passes).
async fn receive_rows(
    stream: &mut jazz_tools::SubscriptionStream,
    rows: usize,
    deadline: Duration,
) -> usize {
    let mut received = 0;
    let _ = tokio::time::timeout(deadline, async {
        while received < rows {
            match stream.next().await {
                Some(delta) => received += delta.added.len(),
                None => break,
            }
        }
    })
    .await;
    received
}

/// G3′-handshake. A handshake registers the client under the engine lock; while a pass
/// holds it, the socket task of the newcomer must wait somewhere that is not the I/O
/// worker: another client's frame and `/health` stay inside the live budget, and the
/// newcomer's handshake completes once the pass ends.
///
/// ```text
///   carol ──handshake──▶ socket task ──register──▶ [lock: held by the pass]
///   bob   ──frame─────▶ socket task ──stage──▶ staging     (must not wait)
///   test  ──GET /health──────────────────────────────────▶ 200 within budget
/// ```
///
/// Internal on purpose: the pass is a thread holding the engine lock (`inspect_core_for_test`
/// through `hold_lock`); nothing public holds the lock for a chosen time. Under the disarm
/// (registration back on the I/O worker) this current-thread runtime freezes, timers
/// included, so the red is the watchdog's exit, not a timeout.
#[tokio::test]
async fn a_handshake_stuck_behind_the_lock_does_not_stall_the_server() {
    // Patience above the sum of the gate's own deadlines (10 s × 5 legs below): a slow but
    // green run must never trip it; only a frozen runtime does.
    let _watchdog = Watchdog::arm(Duration::from_secs(120), "G3′-handshake");
    let server = JazzServer::builder()
        .with_schema(notes_schema())
        .start()
        .await;
    let alice = user(&server, "alice").await;
    let bob = user(&server, "bob").await;
    let mut alice_stream = alice
        .subscribe(QueryBuilder::new("notes").build())
        .await
        .expect("alice subscribes");
    tokio::time::sleep(Duration::from_millis(200)).await;

    let hold = hold_lock(&server);
    // Carol's handshake goes out and its registration waits behind the pass.
    let carol_context = server.make_client_context_for_user(notes_schema(), "carol");
    let carol = tokio::spawn(JazzClient::connect(carol_context));
    tokio::time::sleep(Duration::from_millis(150)).await;

    let started = Instant::now();
    let (_, _, batch) = bob
        .insert("notes", note("while carol waits"))
        .expect("bob writes locally");
    tokio::time::sleep(Duration::from_millis(100)).await;
    let health = reqwest::Client::new()
        .get(format!("{}/health", server.base_url()))
        .send()
        .await
        .expect("health request completes");
    let elapsed = started.elapsed();
    assert!(health.status().is_success());
    assert!(
        elapsed < LIVE_BUDGET,
        "with a handshake stuck behind the lock the server took {elapsed:?} to answer \
         /health; the registration must wait off the I/O worker (budget {LIVE_BUDGET:?})"
    );

    hold.release();
    let carol = tokio::time::timeout(Duration::from_secs(10), carol)
        .await
        .expect("carol's connect completes after the pass")
        .expect("carol's task ends")
        .expect("carol connects");
    tokio::time::timeout(
        Duration::from_secs(10),
        bob.wait_for_batch(batch, DurabilityTier::EdgeServer),
    )
    .await
    .expect("bob's write reaches the server within 10 s of the pass ending")
    .expect("bob's write reaches the server after the pass");
    assert_eq!(
        receive_rows(&mut alice_stream, 1, Duration::from_secs(10)).await,
        1
    );
    // Carol is a registered client: her own subscription settles.
    let mut carol_stream = tokio::time::timeout(
        Duration::from_secs(10),
        carol.subscribe(QueryBuilder::new("notes").build()),
    )
    .await
    .expect("carol's subscribe answers within 10 s")
    .expect("carol subscribes after her late handshake");
    assert_eq!(
        receive_rows(&mut carol_stream, 1, Duration::from_secs(10)).await,
        1
    );
    server.shutdown().await;
}

/// G3″. A client that outruns its staging cap while the pass holds the lock is paused —
/// its socket stops reading — not disconnected: the server can still speak to it, every
/// frame is applied in push order once the pass ends, and its staged bytes never exceed
/// max(cap, one frame).
///
/// ```text
///   bob ──frame 1..n──▶ socket task ──push──▶ [staging: cap 8 entries / 8 KiB] ⟵ pass holds lock
///                              ▲ refused → waits (reads paused)
///   server ──QueryUnsubscription probe──▶ bob   (outbound must still flow)
/// ```
///
/// Internal on purpose: the lock hold, the staged-frame counters and the probe pushed
/// through `connection_event_hub` are engine state no client API exposes.
#[tokio::test]
async fn a_client_that_outruns_its_staging_cap_is_paused_not_dropped_and_still_hears_the_server() {
    const ENTRIES_CAP: usize = 8;
    const BYTES_CAP: usize = 8192;
    let server_tracer = SyncTracer::new();
    let server = JazzServer::builder()
        .with_schema(notes_schema())
        .with_staging_config(StagingConfig {
            max_entries_per_client: ENTRIES_CAP,
            max_bytes_per_client: BYTES_CAP,
            ..StagingConfig::default()
        })
        .with_tracer(server_tracer.clone())
        .start()
        .await;
    let alice = user(&server, "alice").await;
    let bob_tracer = SyncTracer::new();
    let mut bob_context = server.make_client_context_for_user(notes_schema(), "bob");
    bob_context.sync_tracer = Some((bob_tracer.clone(), "bob".to_string()));
    let bob = JazzClient::connect(bob_context)
        .await
        .expect("bob connects");
    let bob_id = bob.client_id().expect("bob has a wire client id");
    server_tracer.register_client(bob_id, "bob");
    let mut alice_stream = alice
        .subscribe(QueryBuilder::new("notes").build())
        .await
        .expect("alice subscribes");
    tokio::time::sleep(Duration::from_millis(200)).await;

    let state = server.server_state();
    let hold = hold_lock(&server);
    let mut batches = Vec::new();
    let paused = loop {
        if batches.len() >= 200 {
            break false;
        }
        let (_, _, batch) = bob
            .insert("notes", note(&format!("note {}", batches.len())))
            .expect("bob writes locally");
        batches.push(batch);
        tokio::time::sleep(Duration::from_millis(5)).await;
        let stats = state.runtime.staging_stats_for_test(bob_id);
        if stats.staged_frames >= 2 && stats.pauses >= 1 {
            break true;
        }
    };
    assert!(
        paused,
        "bob never hit the staging cap: {:?}",
        state.runtime.staging_stats_for_test(bob_id)
    );

    // Outbound still flows to a paused client: a probe the server can send any client.
    let probe = QueryId(0xDEAD_BEEF);
    state
        .connection_event_hub
        .dispatch_payload(bob_id, SyncPayload::QueryUnsubscription { query_id: probe });
    wait_until("the probe to reach bob while paused", Duration::from_millis(500), || {
        bob_tracer.to("bob").iter().any(|message| {
            matches!(message.payload, SyncPayload::QueryUnsubscription { query_id } if query_id == probe)
        })
    })
    .await;

    hold.release();
    // Every frame is applied…
    for batch in &batches {
        tokio::time::timeout(
            Duration::from_secs(10),
            bob.wait_for_batch(*batch, DurabilityTier::EdgeServer),
        )
        .await
        .expect("every staged write reaches the server within 10 s of the pass ending")
        .expect("every staged write reaches the server after the pass");
    }
    let received = receive_rows(&mut alice_stream, batches.len(), Duration::from_secs(15)).await;
    assert_eq!(
        received,
        batches.len(),
        "alice must receive every row bob wrote"
    );
    // …in push order…
    let mut parked_order = Vec::new();
    for message in server_tracer.from("bob") {
        for batch in message.batch_ids() {
            if !parked_order.contains(&batch) {
                parked_order.push(batch);
            }
        }
    }
    let inserted: Vec<_> = batches
        .iter()
        .filter(|batch| parked_order.contains(batch))
        .copied()
        .collect();
    assert_eq!(
        parked_order, inserted,
        "bob's frames must be parked in the order he pushed them"
    );
    // …and the staging never held more than the cap or one frame, and is empty now.
    let stats = state.runtime.staging_stats_for_test(bob_id);
    assert!(
        stats.peak_bytes <= BYTES_CAP.max(stats.largest_frame_bytes),
        "bob's staged bytes peaked at {} — over max(cap {BYTES_CAP}, largest frame {})",
        stats.peak_bytes,
        stats.largest_frame_bytes
    );
    // Bob keeps talking (delivery confirmations for what he receives), so the staging is
    // read at quiescence, not at an instant.
    wait_until("bob's staging to be empty", Duration::from_secs(5), || {
        let stats = state.runtime.staging_stats_for_test(bob_id);
        stats.count == 0 && stats.waiters == 0
    })
    .await;
    server.shutdown().await;
}

/// G-reap. A client whose frame is staged but not yet applied must not be reaped by the
/// disconnect sweep: its rows would be dropped as "unknown client" when the tick finally
/// parks them. The sweep skips it while the frame is staged and reaps it afterwards.
///
/// ```text
///   bob ──frame──▶ staging ──┐   bob disconnects; TTL 0; sweep → must skip bob
///                            └── pass ends → parked → alice gets the row → sweep reaps bob
/// ```
///
/// Internal on purpose: the lock hold, the staged-frame counter and a sweep run by hand
/// (`run_sweep_once`, TTL zero) — the sweep's timing is not observable from a client.
///
/// What is falsifiable here (diff r4 SF1, D2 and D2b): the 2 s timeout on the sweep IS the
/// observable of "the sweep consulted the lock instead of the staging" — both disarms red
/// there. The `!reaped.contains(&bob_id)` line after it is a belt no disarm reaches: while a
/// pass holds the lock the sweep cannot get past the timeout to reach it, and the fixture
/// needs the held lock to make the staged frame exist.
#[tokio::test]
async fn a_client_with_a_staged_frame_is_not_reaped_by_the_sweep() {
    let server = JazzServer::builder()
        .with_schema(notes_schema())
        .start()
        .await;
    let alice = user(&server, "alice").await;
    let bob = user(&server, "bob").await;
    let bob_id = bob.client_id().expect("bob has a wire client id");
    let mut alice_stream = alice
        .subscribe(QueryBuilder::new("notes").build())
        .await
        .expect("alice subscribes");
    tokio::time::sleep(Duration::from_millis(200)).await;

    let state = server.server_state();
    let hold = hold_lock(&server);
    bob.insert("notes", note("staged, then gone"))
        .expect("bob writes locally");
    wait_until("bob's frame to be staged", Duration::from_secs(2), || {
        state.runtime.staging_stats_for_test(bob_id).staged_frames >= 1
    })
    .await;
    bob.shutdown().await.expect("bob disconnects");
    let started = Instant::now();
    while server.disconnect_candidate_count().await == 0 {
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "bob's disconnect never became a sweep candidate"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    server.set_client_ttl(Duration::ZERO).await;
    let reaped = tokio::time::timeout(Duration::from_secs(2), server.run_sweep_once())
        .await
        .expect(
            "the sweep must answer without waiting on the engine lock while bob's frame is staged",
        );
    assert!(
        !reaped.contains(&bob_id),
        "the sweep reaped bob while his frame was still staged: {reaped:?}"
    );

    hold.release();
    assert_eq!(
        receive_rows(&mut alice_stream, 1, Duration::from_secs(10)).await,
        1,
        "bob's staged row must reach alice after the pass"
    );
    // With nothing staged the sweep reaps him.
    let started = Instant::now();
    loop {
        if server.run_sweep_once().await.contains(&bob_id) {
            break;
        }
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "the sweep never reaped bob once his frame was applied"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    server.shutdown().await;
}

/// G-reap-off-worker. A candidate whose removal must take the engine lock (nothing staged,
/// a clean disconnect) makes the sweep wait — on the blocking pool, not on the I/O worker:
/// while the pass holds the lock `/health` answers inside the live budget, and once the
/// pass ends the sweep returns with the candidate reaped. Red with `remove_client` inline
/// in the sweep task (the runtime's worker blocks; the watchdog exits).
///
/// ```text
///   carol disconnects cleanly; TTL 0; sweep → remove_client → [lock: held by the pass]
///   test ──GET /health──▶ 200 within budget          pass ends → sweep returns [carol]
/// ```
///
/// Internal on purpose: the lock hold and a sweep run by hand.
#[tokio::test]
async fn a_sweep_waiting_on_the_lock_does_not_stall_the_server() {
    let _watchdog = Watchdog::arm(Duration::from_secs(30), "G-reap-off-worker");
    let server = JazzServer::builder()
        .with_schema(notes_schema())
        .start()
        .await;
    let carol = user(&server, "carol").await;
    let carol_id = carol.client_id().expect("carol has a wire client id");
    carol.shutdown().await.expect("carol disconnects cleanly");
    let started = Instant::now();
    while server.disconnect_candidate_count().await == 0 {
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "carol's disconnect never became a sweep candidate"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    server.set_client_ttl(Duration::ZERO).await;

    let hold = hold_lock(&server);
    let sweep = {
        let state = server.server_state();
        tokio::spawn(async move { state.run_sweep_once().await })
    };
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(
        !sweep.is_finished(),
        "fixture: the sweep is waiting on the lock"
    );

    let started = Instant::now();
    let health = reqwest::Client::new()
        .get(format!("{}/health", server.base_url()))
        .send()
        .await
        .expect("health request completes");
    let elapsed = started.elapsed();
    assert!(health.status().is_success());
    assert!(
        elapsed < LIVE_BUDGET,
        "with the sweep waiting on the lock the server took {elapsed:?} to answer /health; \
         the removal must wait off the I/O worker (budget {LIVE_BUDGET:?})"
    );

    hold.release();
    let reaped = tokio::time::timeout(Duration::from_secs(5), sweep)
        .await
        .expect("the sweep returns once the pass ends")
        .expect("the sweep task ends");
    assert!(
        reaped.contains(&carol_id),
        "carol is reaped once the lock is free, got {reaped:?}"
    );
    server.shutdown().await;
}
