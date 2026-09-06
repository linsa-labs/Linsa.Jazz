//! Raw-frame gates for the WebSocket ingress (v18 item 3, "transport off the lock"): a
//! frame's declared size is checked against the cap before anything is allocated; a body
//! that does not match its declaration, or does not decode, closes the socket with a named
//! reason and returns its budget; frames beyond the decoded-bytes budget wait in arrival
//! order without losing their socket.
//!
//! ```text
//!   raw socket ──[len][declared][lz4]──▶ header check (cap) ──▶ budget (KiB permits)
//!                                             │ too large             │ wait, FIFO
//!                                             ▼                       ▼
//!                                  error frame + policy close     decode → stage
//! ```

use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use futures::{SinkExt as _, StreamExt as _};
use jazz_tools::query_manager::types::{ColumnType, Schema, SchemaBuilder, TableSchema};
use jazz_tools::runtime_tokio::StagingConfig;
use jazz_tools::server::{JazzServer, ServerBuilder, TestJwtIssuer};
use jazz_tools::sync_manager::types::{Destination, OutboxEntry, ServerId};
use jazz_tools::sync_manager::{ClientId, QueryId, SyncPayload};
use jazz_tools::sync_tracer::SyncTracer;
use jazz_tools::transport_manager::{
    AuthConfig, AuthHandshake, ConnectedResponse, SYNC_PROTOCOL_VERSION,
};
use jazz_tools::transport_protocol::{
    ErrorCode, ServerEvent, SyncBatchRequest, encode_outbox_entry_payload,
};
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream, connect_async};

type Socket = WebSocketStream<MaybeTlsStream<TcpStream>>;

fn notes_schema() -> Schema {
    SchemaBuilder::new()
        .table(TableSchema::builder("notes").column("body", ColumnType::Text))
        .build()
}

fn frame_encode(payload: &[u8]) -> Vec<u8> {
    let compressed = lz4_flex::compress_prepend_size(payload);
    let mut out = Vec::with_capacity(4 + compressed.len());
    out.extend_from_slice(&(compressed.len() as u32).to_be_bytes());
    out.extend_from_slice(&compressed);
    out
}

fn frame_decode(data: &[u8]) -> Option<Vec<u8>> {
    if data.len() < 4 {
        return None;
    }
    let len = u32::from_be_bytes(data[0..4].try_into().unwrap()) as usize;
    if data.len() < 4 + len {
        return None;
    }
    lz4_flex::decompress_size_prepended(&data[4..4 + len]).ok()
}

/// A well-formed frame whose lz4 header lies about the decoded size.
fn frame_declaring(payload: &[u8], declared: u32) -> Vec<u8> {
    let mut compressed = lz4_flex::compress_prepend_size(payload);
    compressed[0..4].copy_from_slice(&declared.to_le_bytes());
    let mut out = Vec::with_capacity(4 + compressed.len());
    out.extend_from_slice(&(compressed.len() as u32).to_be_bytes());
    out.extend_from_slice(&compressed);
    out
}

/// A frame whose body is not an lz4 block at all.
fn frame_with_garbage_body(declared: u32) -> Vec<u8> {
    let mut body = declared.to_le_bytes().to_vec();
    body.extend(std::iter::repeat(0xFFu8).take(64));
    let mut out = Vec::with_capacity(4 + body.len());
    out.extend_from_slice(&(body.len() as u32).to_be_bytes());
    out.extend_from_slice(&body);
    out
}

/// A client batch the server applies without side effects: unsubscriptions for queries
/// nobody registered, one per tag.
fn batch_frame(client_id: ClientId, tags: impl IntoIterator<Item = u64>) -> Vec<u8> {
    frame_encode(&batch_payload(client_id, tags))
}

/// The encoded (pre-compression) client batch for `tags`.
fn batch_payload(client_id: ClientId, tags: impl IntoIterator<Item = u64>) -> Vec<u8> {
    let mut payloads: Vec<SyncPayload> = tags
        .into_iter()
        .map(|tag| SyncPayload::QueryUnsubscription {
            query_id: QueryId(tag),
        })
        .collect();
    // The client encodes a lone payload as one outbox entry and several as a batch; the
    // server decodes in that order, so a raw sender must do the same.
    if payloads.len() == 1 {
        encode_outbox_entry_payload(&OutboxEntry {
            destination: Destination::Server(ServerId::new()),
            payload: payloads.pop().expect("one payload"),
        })
        .expect("entry encodes")
    } else {
        SyncBatchRequest {
            payloads,
            client_id,
        }
        .encode_payload()
        .expect("batch encodes")
    }
}

struct RawSocket {
    ws: Socket,
    client_id: ClientId,
}

/// A raw WebSocket client that completes the handshake as `user` and then sends whatever
/// bytes the gate hands it.
async fn raw_socket(server: &JazzServer, user: &str) -> RawSocket {
    let context = server.make_client_context_for_user(notes_schema(), user);
    let jwt = context
        .jwt_token
        .clone()
        .expect("a user context carries a jwt");
    let ws_url = format!(
        "ws://127.0.0.1:{}/apps/{}/ws",
        server.port(),
        server.app_id()
    );
    let (mut ws, _) = connect_async(&ws_url).await.expect("ws connect");
    let client_id = ClientId::new();
    let handshake = AuthHandshake {
        sync_protocol_version: SYNC_PROTOCOL_VERSION,
        client_id: client_id.to_string(),
        auth: AuthConfig {
            jwt_token: Some(jwt),
            ..Default::default()
        },
        catalogue_state_hash: None,
        declared_schema_hash: None,
        acks_deliveries: false,
    };
    let payload = serde_json::to_vec(&handshake).expect("serialize AuthHandshake");
    ws.send(Message::Binary(frame_encode(&payload).into()))
        .await
        .expect("handshake sent");
    match ws.next().await {
        Some(Ok(Message::Binary(bytes))) => {
            let inner = frame_decode(&bytes).expect("the handshake reply decodes");
            serde_json::from_slice::<ConnectedResponse>(&inner).unwrap_or_else(|error| {
                panic!(
                    "handshake rejected ({error}): {}",
                    String::from_utf8_lossy(&inner)
                )
            });
        }
        other => panic!("unexpected handshake reply: {other:?}"),
    }
    RawSocket { ws, client_id }
}

#[derive(Debug)]
enum Reply {
    Error {
        code: ErrorCode,
        message: String,
    },
    /// The variant name only. Read through `Debug` in the panics below and matched on as a
    /// "something arrived, but not the refusal" — since diff r9 B2 nothing branches on the
    /// name itself: identifying a specific frame is `expect_probe`'s job, and doing it by
    /// name is exactly the looseness that let a handshake frame pass for a probe.
    #[allow(dead_code)]
    Event(String),
    Close(Option<String>),
    Nothing,
}

async fn next_reply(ws: &mut Socket, patience: Duration) -> Reply {
    match tokio::time::timeout(patience, ws.next()).await {
        Err(_) => Reply::Nothing,
        Ok(None) => Reply::Close(None),
        Ok(Some(Ok(Message::Close(frame)))) => {
            Reply::Close(frame.map(|frame| frame.reason.to_string()))
        }
        Ok(Some(Ok(Message::Binary(bytes)))) => {
            let inner = frame_decode(&bytes).expect("a server frame decodes");
            match ServerEvent::decode_payload(&inner).expect("a server event decodes") {
                ServerEvent::Error { message, code } => Reply::Error { code, message },
                event => Reply::Event(event.variant_name().to_string()),
            }
        }
        Ok(Some(Ok(other))) => Reply::Event(format!("{other:?}")),
        Ok(Some(Err(error))) => Reply::Close(Some(format!("error: {error}"))),
    }
}

/// Wait for the probe — a `QueryUnsubscription` carrying `query_id` — discarding whatever the
/// server queued to this socket earlier.
///
/// diff r9 B2: the probe loop used to accept ANY event whose variant name started with
/// `SyncUpdate`, and a raw socket declares no catalogue state, so the server queues a burst of
/// catalogue `SyncUpdate`s to it at the handshake. A socket with one still buffered therefore
/// passed the probe WITHOUT hearing the server after the pause — measured, not argued: under
/// the D17 disarm, which stops outbound for every paused socket, the gate reached socket 5
/// before it reddened, so socket 4 had passed on a frame from before the pause. Matching the
/// id is what makes the assertion mean "the server can still reach a paused socket".
async fn expect_probe(
    // Generic since the fairness gate below splits its socket: it needs the sink half to keep
    // uploading while the stream half waits for the probe. `Socket` still satisfies this, so
    // every existing caller is unchanged.
    ws: &mut (
             impl futures::Stream<Item = Result<Message, tokio_tungstenite::tungstenite::Error>> + Unpin
         ),
    query_id: QueryId,
    patience: Duration,
) -> Result<(), String> {
    let deadline = Instant::now() + patience;
    loop {
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return Err("nothing".to_string());
        }
        match tokio::time::timeout(left, ws.next()).await {
            Err(_) => return Err("nothing".to_string()),
            Ok(None) => return Err("the socket closed".to_string()),
            Ok(Some(Ok(Message::Binary(bytes)))) => {
                let inner = frame_decode(&bytes).expect("a server frame decodes");
                match ServerEvent::decode_payload(&inner).expect("a server event decodes") {
                    ServerEvent::SyncUpdate { payload, .. } => {
                        if matches!(*payload, SyncPayload::QueryUnsubscription { query_id: id } if id == query_id)
                        {
                            return Ok(());
                        }
                    }
                    ServerEvent::SyncUpdateBatch { updates } => {
                        if updates.iter().any(|update| {
                            matches!(update.payload, SyncPayload::QueryUnsubscription { query_id: id } if id == query_id)
                        }) {
                            return Ok(());
                        }
                    }
                    ServerEvent::Error { message, code } => {
                        return Err(format!("an error frame ({code:?}): {message}"));
                    }
                    _ => {}
                }
            }
            Ok(Some(Ok(_))) => {}
            Ok(Some(Err(error))) => return Err(format!("a transport error: {error}")),
        }
    }
}

/// The server refused the frame: a `BadRequest` error frame whose message starts with the
/// reason, then a policy close naming the same reason.
///
/// Server-initiated events are drained on the way (diff r6 B1): a raw socket declares no
/// catalogue state, so the server queues catalogue `SyncUpdate`s to it at the handshake
/// and the refusal is not always the first frame on the wire (it was not in 11 of 245
/// runs). That burst is bounded — one `SyncUpdate` per catalogue entry at the handshake,
/// then nothing — so a refusal that never comes reads as `Nothing` once the deadline is
/// spent: a signature no server push can imitate, and what a disarmed check reds as. Each
/// phase has its own 4 s deadline and every read's patience is clipped to what remains
/// (diff r7 S1): the function is bounded by 8 s, a refusal is accepted only within 4 s of
/// the frame, and the close phase never inherits a spent budget. A close with no reason
/// is NOT accepted (diff r7 S2): the module's claim is a close naming the reason.
async fn expect_refusal(ws: &mut Socket, reason: &str) {
    let patience = |deadline: Instant| {
        Duration::from_secs(2).min(deadline.saturating_duration_since(Instant::now()))
    };
    let deadline = Instant::now() + Duration::from_secs(4);
    loop {
        match next_reply(ws, patience(deadline)).await {
            Reply::Error { code, message } => {
                assert_eq!(code, ErrorCode::BadRequest, "reason {reason}: {message}");
                assert!(
                    message.starts_with(reason),
                    "the error frame must name the reason {reason}: {message}"
                );
                break;
            }
            Reply::Event(_) if Instant::now() < deadline => continue,
            other => panic!("expected an error frame naming {reason}, got {other:?}"),
        }
    }
    let deadline = Instant::now() + Duration::from_secs(4);
    loop {
        match next_reply(ws, patience(deadline)).await {
            Reply::Close(Some(close_reason)) => {
                assert_eq!(close_reason, reason);
                break;
            }
            Reply::Event(_) if Instant::now() < deadline => continue,
            other => panic!("expected a close naming {reason}, got {other:?}"),
        }
    }
}

async fn health_ok(server: &JazzServer) {
    let health = reqwest::Client::new()
        .get(format!("{}/health", server.base_url()))
        .send()
        .await
        .expect("health request completes");
    assert!(health.status().is_success(), "{:?}", health.status());
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

/// G-frame. The declared size is checked on the lz4 header before allocation — a 4 GiB
/// declaration, or one byte over the cap, costs the server nothing but an error frame and
/// a close; a frame under the cap is applied; a body that does not match its declaration
/// or does not decode closes the socket by name and gives its budget back.
///
/// Internal on purpose: the budget's permit count (`inflight_budget`) is the only witness
/// that a refusal gave its permits back — no client API sees the server's budget.
#[tokio::test]
async fn frames_lying_about_their_size_are_refused_by_name_before_allocation() {
    let tracer = SyncTracer::new();
    let server = JazzServer::builder()
        .with_schema(notes_schema())
        .with_tracer(tracer.clone())
        .start()
        .await;
    let state = server.server_state();
    let budget = state.runtime.inflight_budget();
    let whole_budget = budget.available_permits();
    let cap = state.max_ws_decoded_frame_bytes;
    assert_eq!(
        cap,
        288 * 1024 * 1024,
        "the default cap the gate is written for"
    );

    // A 4 GiB declaration.
    let mut bomb = raw_socket(&server, "bomb").await;
    bomb.ws
        .send(Message::Binary(frame_declaring(b"tiny", u32::MAX).into()))
        .await
        .expect("sent");
    expect_refusal(&mut bomb.ws, "frame_too_large").await;
    health_ok(&server).await;
    assert_eq!(budget.available_permits(), whole_budget);

    // One byte over the cap.
    let mut over = raw_socket(&server, "over").await;
    over.ws
        .send(Message::Binary(
            frame_declaring(b"tiny", (cap + 1) as u32).into(),
        ))
        .await
        .expect("sent");
    expect_refusal(&mut over.ws, "frame_too_large").await;
    assert_eq!(budget.available_permits(), whole_budget);

    // A frame under the cap is applied: a thousand unsubscriptions reach the core.
    let mut fine = raw_socket(&server, "fine").await;
    tracer.register_client(fine.client_id, "fine");
    fine.ws
        .send(Message::Binary(
            batch_frame(fine.client_id, 1_000_000..1_001_000).into(),
        ))
        .await
        .expect("sent");
    wait_until(
        "the accepted frame to be applied",
        Duration::from_secs(5),
        || tracer.from("fine").len() >= 1000,
    )
    .await;
    let reply = next_reply(&mut fine.ws, Duration::from_millis(200)).await;
    assert!(
        matches!(reply, Reply::Nothing | Reply::Event(_)),
        "an accepted frame draws no error and no close, got {reply:?}"
    );
    assert_eq!(
        budget.available_permits(),
        whole_budget,
        "an applied frame returns its permits"
    );

    // Declares 8 MiB over a VALID one-payload body: lz4 decodes it happily to its true
    // size, so only the declaration check can refuse it — and it must, before the payload
    // is applied (a lying header is a free 8 MiB allocation per frame otherwise).
    let mut liar = raw_socket(&server, "liar").await;
    tracer.register_client(liar.client_id, "liar");
    liar.ws
        .send(Message::Binary(
            frame_declaring(&batch_payload(liar.client_id, [42]), 8 * 1024 * 1024).into(),
        ))
        .await
        .expect("sent");
    expect_refusal(&mut liar.ws, "frame_size_mismatch").await;
    wait_until(
        "the liar's permits to come back",
        Duration::from_secs(2),
        || budget.available_permits() == whole_budget,
    )
    .await;
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(
        tracer.from("liar").is_empty(),
        "a frame refused for its declaration must not be applied"
    );

    // A body that is not lz4 at all.
    let mut garbage = raw_socket(&server, "garbage").await;
    garbage
        .ws
        .send(Message::Binary(frame_with_garbage_body(1024).into()))
        .await
        .expect("sent");
    expect_refusal(&mut garbage.ws, "frame_corrupt").await;

    // Valid lz4 of bytes that are not a client batch.
    let mut noise = raw_socket(&server, "noise").await;
    noise
        .ws
        .send(Message::Binary(frame_encode(&[0xEEu8; 512]).into()))
        .await
        .expect("sent");
    expect_refusal(&mut noise.ws, "frame_corrupt").await;

    health_ok(&server).await;
    assert_eq!(
        budget.available_permits(),
        whole_budget,
        "no refusal leaked a permit"
    );
    server.shutdown().await;
}

/// G-budget. With a ONE-frame budget (1 KiB) and a 1 KiB cap, five sockets each send one
/// frame while the pass holds the lock: one is admitted, four wait for budget — without
/// losing their socket, still hearing the server — and once the pass ends all five are
/// applied in arrival order, with the budget whole and nobody waiting.
///
/// One frame of budget on purpose (design v7 § B1, restored in v11): the drain returns the
/// permits of every drained frame in one `add_permits`, and two permits wake two waiters that
/// race to stage — cross-connection order is undefined then (diff r3 measured 4/40 red with a
/// 2 KiB budget). With one permit released per drain a single waiter is woken and the
/// semaphore's FIFO makes the arrival order the admission order. Residual: the waiter-count
/// witness increments before the acquire registers on the semaphore; the sends are
/// sequential behind it, so the registration order can only invert if a socket task is
/// descheduled between the two — measured 0 red in the runs recorded in the log.
///
/// Three rounds with fresh sockets (diff r4 Blocking 2). Every number this doc carried before
/// diff r11 came from the 2026-09-05 18:32 run, which r10 invalidated, and every one of them
/// was wrong. Restated from the full chain run of 2026-09-05 23:06, which is the run on the
/// tree this ships as:
///
/// * The D8 disarm — an acquire re-created on every loop spin, losing its queue place — is
///   caught in **6 of 6** runs, and every one of them fails in **round 0**. The earlier claim
///   that "round 1 is the load-bearing one" is the opposite of what the data says.
/// * The permutation is not fixed but the detection is: two orders alternate across the runs,
///   `[1, 4, 3, 2, 5]` and `[1, 4, 5, 3, 2]`. Which one appears depends on where the drain's
///   single released permit sits when the spin happens. Neither is the pair this doc used to
///   name.
/// * **"6 of 6" is not a detection rate** (diff r11 SF2). It is one deterministic outcome
///   observed six times, and it bounds the flakiness — the row does not sometimes pass — and
///   nothing else. It does not license a per-round probability, and the arithmetic that once
///   pooled two such counts into "5 of 10" was combining two different things.
/// * Rounds 1 and 2 have never executed under the disarm, because the gate aborts at the first
///   failing round and round 0 has always been it. They are insurance, not measured coverage.
///
/// Three rounds are kept anyway: they cost 0.4 s, and a detector that has only ever been
/// exercised in its first round is one scheduling change away from needing the others. The
/// armed gate stays deterministic — every round must hold.
///
/// Internal on purpose: the pass is a thread holding the engine lock (`inspect_core_for_test`),
/// and the observables — staged frames per client, the waiter count, the permit count — are
/// engine state no client API exposes.
#[tokio::test]
async fn frames_beyond_the_decoded_budget_wait_in_arrival_order_without_losing_their_socket() {
    let tracer = SyncTracer::new();
    let server = JazzServer::builder()
        .with_schema(notes_schema())
        .with_tracer(tracer.clone())
        .with_max_ws_decoded_frame_bytes(1024)
        .with_staging_config(StagingConfig {
            inflight_budget_bytes: 1024,
            ..StagingConfig::default()
        })
        .start()
        .await;
    let state = server.server_state();
    let budget = state.runtime.inflight_budget();
    let whole_budget = budget.available_permits();
    assert_eq!(
        whole_budget, 1,
        "1 KiB of budget is one KiB permit: one frame"
    );

    let users = ["one", "two"];
    for round in 0..3u64 {
        let mut sockets = Vec::new();
        for index in 0..5 {
            let socket = raw_socket(&server, users[index % 2]).await;
            tracer.register_client(socket.client_id, format!("socket{}", index + 1));
            sockets.push(socket);
        }

        // The pass: a plain thread holds the engine lock until released.
        let (release, released) = std::sync::mpsc::channel::<()>();
        let pass = {
            let state = server.server_state();
            std::thread::spawn(move || {
                state.runtime.inspect_core_for_test(|_| {
                    let _ = released.recv();
                });
            })
        };
        std::thread::sleep(Duration::from_millis(50));

        for (index, socket) in sockets.iter_mut().enumerate() {
            let tag = round * 10 + (index + 1) as u64;
            socket
                .ws
                .send(Message::Binary(batch_frame(socket.client_id, [tag]).into()))
                .await
                .expect("sent");
            // Send the next only once this one is where the gate expects it: staged for the
            // first, waiting for budget for the rest.
            if index < 1 {
                let client_id = socket.client_id;
                wait_until("the frame to be staged", Duration::from_secs(2), || {
                    state
                        .runtime
                        .staging_stats_for_test(client_id)
                        .staged_frames
                        == 1
                })
                .await;
            } else {
                let waiting = index;
                wait_until(
                    "the frame to wait for budget",
                    Duration::from_secs(2),
                    || state.budget_waiters.load(Ordering::SeqCst) == waiting,
                )
                .await;
            }
        }
        assert_eq!(
            budget.available_permits(),
            0,
            "one frame holds the whole budget"
        );

        // Outbound still flows to every socket waiting for budget — probed in the order 4, 5,
        // 3, 2, each reply awaited: a socket's loop spins on its delivery, and an acquire that
        // lost its queue place on a spin (the D8 disarm) is admitted out of order (see the doc:
        // 5 of 10 single rounds pooled across v12 and this run; three rounds here).
        for index in [3, 4, 2, 1] {
            state.connection_event_hub.dispatch_payload(
                sockets[index].client_id,
                SyncPayload::QueryUnsubscription {
                    query_id: QueryId(0xBEEF),
                },
            );
            if let Err(what) = expect_probe(
                &mut sockets[index].ws,
                QueryId(0xBEEF),
                Duration::from_millis(500),
            )
            .await
            {
                panic!(
                    "socket {} waiting for budget must still hear the server, got {what}",
                    index + 1
                );
            }
        }

        drop(release);
        pass.join().expect("the pass thread ends");

        let tags_in_order = |tracer: &SyncTracer| -> Vec<u64> {
            tracer
                .messages()
                .iter()
                .filter(|message| message.from.0.starts_with("socket"))
                .filter_map(|message| match message.payload {
                    SyncPayload::QueryUnsubscription { query_id }
                        if query_id.0 / 10 == round && query_id.0 % 10 <= 5 =>
                    {
                        Some(query_id.0 % 10)
                    }
                    _ => None,
                })
                .collect()
        };
        wait_until(
            "all five frames to be applied",
            Duration::from_secs(10),
            || tags_in_order(&tracer).len() >= 5,
        )
        .await;
        assert_eq!(
            tags_in_order(&tracer),
            vec![1, 2, 3, 4, 5],
            "round {round}: frames must be admitted in arrival order, not in the order sockets \
         happened to wake"
        );
        assert_eq!(state.budget_waiters.load(Ordering::SeqCst), 0);
        assert_eq!(
            budget.available_permits(),
            whole_budget,
            "the budget is whole again"
        );
        for (index, socket) in sockets.iter_mut().enumerate() {
            let reply = next_reply(&mut socket.ws, Duration::from_millis(100)).await;
            assert!(
                matches!(reply, Reply::Nothing | Reply::Event(_)),
                "socket {} must keep its connection, got {reply:?}",
                index + 1
            );
        }
    }
    server.shutdown().await;
}

/// G-budget-exit. A socket that leaves while its frame waits for budget returns the permits
/// its acquire already held: with a 3 KiB budget and a 2 KiB cap, two 1 KiB frames are
/// staged, a third socket's 2 KiB frame takes the last permit and waits for one more, two
/// more sockets queue behind it; the third socket closes. Once the pass ends the other
/// four frames are applied in arrival order, the budget is whole (the partial permit came
/// back) and nobody waits. A paused socket is not read, so the exit is noticed at the
/// server's next write to it (the heartbeat, or a delivery as here). Red with the acquire
/// kept alive past the socket's exit: the returned permit never admits the next waiter
/// (the 2 s `wait_until` fires); the budget-whole clause below it is not reached under
/// that disarm (diff r6 SF1).
///
/// The arrival-order clause is deterministic here for a reason worth keeping (diff r3
/// SF3): the leaving socket returns ONE partial permit, admitting exactly one waiter at a
/// time — no two waiters are ever woken together.
///
/// Internal on purpose: same witnesses as G-budget (the lock hold, the waiter count, the
/// permit count).
#[tokio::test]
async fn a_socket_that_leaves_while_waiting_for_budget_returns_its_permits() {
    let tracer = SyncTracer::new();
    let server = JazzServer::builder()
        .with_schema(notes_schema())
        .with_tracer(tracer.clone())
        .with_max_ws_decoded_frame_bytes(2048)
        .with_staging_config(StagingConfig {
            inflight_budget_bytes: 3072,
            ..StagingConfig::default()
        })
        .start()
        .await;
    let state = server.server_state();
    let budget = state.runtime.inflight_budget();
    let whole_budget = budget.available_permits();
    assert_eq!(whole_budget, 3, "3 KiB of budget is three KiB permits");

    let users = ["one", "two"];
    let mut sockets = Vec::new();
    for index in 0..5 {
        let socket = raw_socket(&server, users[index % 2]).await;
        tracer.register_client(socket.client_id, format!("socket{}", index + 1));
        sockets.push(socket);
    }
    // A frame that decodes to more than one KiB and at most two: two permits.
    let two_kib_frame = |client_id: ClientId| -> Vec<u8> {
        let mut tags = 64u64;
        loop {
            let frame = batch_frame(client_id, 1000..1000 + tags);
            let decoded = frame_decode(&frame)
                .expect("a frame we built decodes")
                .len();
            if decoded > 1024 {
                assert!(
                    decoded <= 2048,
                    "fixture: the big frame must stay under the cap"
                );
                return frame;
            }
            tags += 16;
        }
    };

    let (release, released) = std::sync::mpsc::channel::<()>();
    let pass = {
        let state = server.server_state();
        std::thread::spawn(move || {
            state.runtime.inspect_core_for_test(|_| {
                let _ = released.recv();
            });
        })
    };
    std::thread::sleep(Duration::from_millis(50));

    for (index, socket) in sockets.iter_mut().enumerate() {
        let tag = (index + 1) as u64;
        let frame = if index == 2 {
            two_kib_frame(socket.client_id)
        } else {
            batch_frame(socket.client_id, [tag])
        };
        socket
            .ws
            .send(Message::Binary(frame.into()))
            .await
            .expect("sent");
        if index < 2 {
            let client_id = socket.client_id;
            wait_until("the frame to be staged", Duration::from_secs(2), || {
                state
                    .runtime
                    .staging_stats_for_test(client_id)
                    .staged_frames
                    == 1
            })
            .await;
        } else {
            let waiting = index - 1;
            wait_until(
                "the frame to wait for budget",
                Duration::from_secs(2),
                || state.budget_waiters.load(Ordering::SeqCst) == waiting,
            )
            .await;
        }
    }
    assert_eq!(
        budget.available_permits(),
        0,
        "two staged frames and the big frame's partial acquire hold the whole budget"
    );

    // The third socket leaves while its acquire holds one permit and waits for another. A
    // paused socket is not read, so the server notices the exit at its next write to it —
    // the 30 s heartbeat in production, a delivery here (the same exit path).
    let leaving = sockets.remove(2);
    let leaving_id = leaving.client_id;
    drop(leaving);
    // The server notices at its next write to the socket (the heartbeat in production, a
    // delivery here): the task ends and bob becomes a sweep candidate.
    let mut probes = 0;
    while server.disconnect_candidate_count().await == 0 {
        assert!(
            probes < 20,
            "the leaving socket's task did not end after {probes} deliveries"
        );
        state.connection_event_hub.dispatch_payload(
            leaving_id,
            SyncPayload::QueryUnsubscription {
                query_id: QueryId(0xBEEF),
            },
        );
        probes += 1;
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    // The dropped acquire returned its one permit, and that permit admitted the next
    // waiter at once: socket 4 is staged, socket 5 still waits.
    wait_until(
        "the returned permit to admit the next waiter",
        Duration::from_secs(2),
        || state.budget_waiters.load(Ordering::SeqCst) == 1,
    )
    .await;
    assert_eq!(
        budget.available_permits(),
        0,
        "socket 4 holds the returned permit"
    );

    drop(release);
    pass.join().expect("the pass thread ends");

    let tags_in_order = |tracer: &SyncTracer| -> Vec<u64> {
        tracer
            .messages()
            .iter()
            .filter(|message| message.from.0.starts_with("socket"))
            .filter_map(|message| match message.payload {
                SyncPayload::QueryUnsubscription { query_id } if query_id.0 <= 5 => {
                    Some(query_id.0)
                }
                _ => None,
            })
            .collect()
    };
    wait_until(
        "the four remaining frames to be applied",
        Duration::from_secs(10),
        || tags_in_order(&tracer).len() >= 4,
    )
    .await;
    assert_eq!(
        tags_in_order(&tracer),
        vec![1, 2, 4, 5],
        "the frames of the sockets that stayed are applied in arrival order"
    );
    assert_eq!(state.budget_waiters.load(Ordering::SeqCst), 0);
    assert_eq!(
        budget.available_permits(),
        whole_budget,
        "the budget is whole again: the leaving socket's partial permit came back"
    );
    server.shutdown().await;
}

/// G-builder (diff r5 SF1). A decoded-frame cap above the inflight budget is a frame no
/// admission could ever serve: the builder refuses the pair instead of starting a server
/// that would park a legal frame forever. Black-box: the public `ServerBuilder` and its
/// error text; the auth config mirrors the test server's.
#[tokio::test]
async fn a_frame_cap_above_the_inflight_budget_is_refused_by_the_builder() {
    let jwks = TestJwtIssuer::start().await;
    // The server-side config (the transport one imported above is the client's).
    let auth_config = jazz_tools::middleware::AuthConfig {
        jwks_url: Some(jwks.endpoint()),
        allow_local_first_auth: true,
        backend_secret: Some(JazzServer::BACKEND_SECRET.to_string()),
        admin_secret: Some(JazzServer::ADMIN_SECRET.to_string()),
        ..Default::default()
    };
    let refused = ServerBuilder::new(JazzServer::default_app_id())
        .with_auth_config(auth_config)
        .with_schema(notes_schema())
        .with_max_ws_decoded_frame_bytes(4096)
        .with_staging_config(StagingConfig {
            inflight_budget_bytes: 1024,
            ..StagingConfig::default()
        })
        .build()
        .await;
    let err = match refused {
        Ok(_) => panic!("a 4 KiB cap over a 1 KiB budget must not build"),
        Err(err) => err,
    };
    assert!(
        err.contains("could never be admitted"),
        "the refusal names the mechanism: {err}"
    );
}
