//! A client that comes back on a new socket is answered the reads it makes again.
//!
//! An app keeps its client id across launches and numbers its queries from zero each
//! time, and the server keeps a client's subscriptions for a while after its socket
//! closes. So the read a relaunched app makes first is, to the server, a subscription it
//! already holds and has already settled — and a read that asks for a durability tier
//! shows nothing until it is told the tier was reached.
//!
//! ```text
//!  app, first launch            server                  app, relaunched
//!    ──subscribe q0──────────►  settles q0
//!    ◄──────────settled q0────
//!    ✕ socket closes            keeps q0
//!                               ◄──────────subscribe q0──  (same client id, new socket)
//!                               ──settled q0────────────►  must be said again
//! ```
//!
//! What a client was told is known per socket, not per client: a frame still on its way
//! from the older socket must not use up the answer the newer one is owed.
//!
//! These run over real sockets against the route, so they cover the path from the
//! websocket handler to the subscription that the engine-level gates
//! (`tests/relaunched_client_read.rs`) do not: both ways a client authenticates, and both
//! shapes its frames take — one payload, or the batch a client sends when it has several.

use std::sync::Arc;
use std::time::Duration;

use futures::{SinkExt as _, StreamExt as _};
use tokio::net::TcpStream;
use tokio_tungstenite::{
    MaybeTlsStream, WebSocketStream, connect_async, tungstenite::Message as WsMessage,
};

use crate::jazz_transport::ServerEvent;
use crate::middleware::AuthConfig;
use crate::query_manager::query::Query;
use crate::query_manager::session::Session;
use crate::query_manager::types::{ColumnType, SchemaBuilder, TableSchema};
use crate::schema_manager::AppId;
use crate::server::{ServerBuilder, ServerState, StorageBackend};
use crate::sync_manager::types::{Destination, OutboxEntry, ServerId};
use crate::sync_manager::{ClientId, DurabilityTier, QueryId, SyncPayload};
use crate::transport_protocol::SyncBatchRequest;

type WsStream = WebSocketStream<MaybeTlsStream<TcpStream>>;

const BACKEND_SECRET: &str = "test-backend-secret";

/// Long enough for a loaded runner to answer; a read that is never answered waits it out.
const ANSWER_DEADLINE: Duration = Duration::from_secs(5);

/// How a client says who it is. An app's user arrives with a session; a backend with the
/// backend secret alone. The server sets the two up through different calls.
#[derive(Clone, Copy)]
enum As {
    Backend,
    User,
}

async fn server() -> (Arc<ServerState>, std::net::SocketAddr) {
    let schema = SchemaBuilder::new()
        .table(TableSchema::builder("todos").column("title", ColumnType::Text))
        .build();
    let state = ServerBuilder::new(AppId::from_name("test-app"))
        .with_auth_config(AuthConfig {
            backend_secret: Some(BACKEND_SECRET.to_string()),
            ..Default::default()
        })
        .with_storage(StorageBackend::InMemory)
        .with_schema(schema)
        .build()
        .await
        .expect("build test server state")
        .state;

    let app = super::create_router(state.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind test listener");
    let addr = listener.local_addr().expect("test listener addr");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve test app");
    });
    (state, addr)
}

/// Open a socket as `client_id` and wait until the server has registered it.
async fn open(
    addr: std::net::SocketAddr,
    state: &Arc<ServerState>,
    client_id: ClientId,
    who: As,
) -> WsStream {
    let url = format!("ws://{addr}/apps/{}/ws", state.app_id);
    let (mut ws, _) = connect_async(&url).await.expect("ws upgrade");

    let handshake = crate::transport_manager::AuthHandshake {
        acks_deliveries: true,
        sync_protocol_version: crate::transport_manager::SYNC_PROTOCOL_VERSION,
        client_id: client_id.to_string(),
        auth: crate::transport_manager::AuthConfig {
            backend_secret: Some(BACKEND_SECRET.to_string()),
            backend_session: match who {
                As::Backend => None,
                As::User => Some(
                    serde_json::to_value(Session::new("alice")).expect("serialize the session"),
                ),
            },
            ..Default::default()
        },
        catalogue_state_hash: None,
        declared_schema_hash: None,
    };
    let payload = serde_json::to_vec(&handshake).expect("serialize handshake");
    ws.send(WsMessage::Binary(crate::transport_manager::frame_encode(
        &payload,
    )))
    .await
    .expect("send handshake");

    let connected = tokio::time::timeout(ANSWER_DEADLINE, ws.next())
        .await
        .expect("the server answers the handshake")
        .expect("the socket stays open through the handshake")
        .expect("ws recv");
    let WsMessage::Binary(bytes) = connected else {
        panic!("unexpected answer to the handshake: {connected:?}");
    };
    let inner = crate::transport_manager::frame_decode(&bytes).expect("a framed answer");
    let _: crate::transport_manager::ConnectedResponse =
        serde_json::from_slice(&inner).expect("the handshake is accepted");
    ws
}

/// A read shown only once the edge tier has settled it. Every launch of an app numbers
/// its reads from zero.
fn read(query_id: u64) -> SyncPayload {
    SyncPayload::QuerySubscription {
        query_id: QueryId(query_id),
        query: Box::new(Query::new("todos")),
        session: None,
        required_tier: Some(DurabilityTier::EdgeServer),
        propagation: Default::default(),
        policy_context_tables: Vec::new(),
    }
}

/// The client makes the reads `query_ids` in one frame: a lone payload for one, and for
/// several the batch a client's transport sends them in.
async fn subscribe(ws: &mut WsStream, client_id: ClientId, query_ids: &[u64]) {
    let payload = match query_ids {
        [query_id] => crate::transport_protocol::encode_outbox_entry_payload(&OutboxEntry {
            destination: Destination::Server(ServerId::new()),
            payload: read(*query_id),
        }),
        several => SyncBatchRequest {
            payloads: several.iter().copied().map(read).collect(),
            client_id,
        }
        .encode_payload(),
    }
    .expect("encode the reads");
    ws.send(WsMessage::Binary(crate::transport_manager::frame_encode(
        &payload,
    )))
    .await
    .expect("send the reads");
}

/// The reads the socket is told have settled, in the order it is told, up to and including
/// `last`. Panics if `last` is not among them in time.
async fn told_settled_through(ws: &mut WsStream, last: u64, what: &str) -> Vec<u64> {
    let deadline = tokio::time::Instant::now() + ANSWER_DEADLINE;
    let mut told = Vec::new();
    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        let Ok(Some(Ok(message))) = tokio::time::timeout(remaining, ws.next()).await else {
            panic!("{what} (told so far: {told:?})");
        };
        let WsMessage::Binary(bytes) = message else {
            continue;
        };
        let payloads: Vec<SyncPayload> = match ServerEvent::decode_frame(&bytes) {
            Some(ServerEvent::SyncUpdate { payload, .. }) => vec![*payload],
            Some(ServerEvent::SyncUpdateBatch { updates }) => {
                updates.into_iter().map(|update| update.payload).collect()
            }
            _ => continue,
        };
        for payload in payloads {
            if let SyncPayload::QuerySettled { query_id, .. } = payload {
                told.push(query_id.0);
                if query_id.0 == last {
                    return told;
                }
            }
        }
    }
}

async fn a_client_that_comes_back_is_answered(who: As, reads: &[u64]) {
    let (state, addr) = server().await;
    let client_id = ClientId::new();
    let last = *reads.last().expect("a read");

    let mut first_launch = open(addr, &state, client_id, who).await;
    subscribe(&mut first_launch, client_id, reads).await;
    let told = told_settled_through(
        &mut first_launch,
        last,
        "the first launch's reads are answered",
    )
    .await;
    assert_eq!(told, reads);
    first_launch.close(None).await.expect("the app closes");
    drop(first_launch);

    let mut relaunched = open(addr, &state, client_id, who).await;
    subscribe(&mut relaunched, client_id, reads).await;
    let told = told_settled_through(
        &mut relaunched,
        last,
        "the reads the relaunched app makes again are never answered",
    )
    .await;
    assert_eq!(
        told, reads,
        "each read the relaunched app makes again is answered once",
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_backend_that_comes_back_on_a_new_socket_is_answered_the_read_it_made_before() {
    a_client_that_comes_back_is_answered(As::Backend, &[0]).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_user_that_comes_back_on_a_new_socket_is_answered_the_read_it_made_before() {
    a_client_that_comes_back_is_answered(As::User, &[0]).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_user_that_comes_back_is_answered_each_read_of_the_frame_it_makes_them_in() {
    a_client_that_comes_back_is_answered(As::User, &[0, 1, 2]).await;
}

/// The newer socket is open before the older one repeats its reads. The repeat is one the
/// client was already answered on that socket: it stays unanswered, and the first time the
/// newer socket asks it is answered. A read the older socket has not made before follows
/// its repeat on the same socket, so its answer marks the point by which an answer to the
/// repeat would have come: the server takes a socket's frames in order.
async fn frames_from_an_older_socket_do_not_take_the_newer_sockets_answers(who: As, reads: &[u64]) {
    let (state, addr) = server().await;
    let client_id = ClientId::new();
    let last = *reads.last().expect("a read");
    let marker = last + 1;

    let mut older = open(addr, &state, client_id, who).await;
    subscribe(&mut older, client_id, reads).await;
    told_settled_through(&mut older, last, "the older socket's reads are answered").await;

    let mut newer = open(addr, &state, client_id, who).await;
    subscribe(&mut older, client_id, reads).await;
    subscribe(&mut older, client_id, &[marker]).await;
    let told = told_settled_through(
        &mut newer,
        marker,
        "what the server tells a client reaches each socket it has open",
    )
    .await;
    assert_eq!(
        told,
        [marker],
        "reads repeated on the older socket were answered as if the newer one had made them",
    );

    subscribe(&mut newer, client_id, reads).await;
    let told = told_settled_through(
        &mut newer,
        last,
        "the newer socket's first reads are never answered",
    )
    .await;
    assert_eq!(told, reads);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_frame_from_an_older_socket_does_not_take_the_newer_sockets_answer() {
    frames_from_an_older_socket_do_not_take_the_newer_sockets_answers(As::Backend, &[0]).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_users_frame_from_an_older_socket_does_not_take_the_newer_sockets_answer() {
    frames_from_an_older_socket_do_not_take_the_newer_sockets_answers(As::User, &[0]).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_batch_from_an_older_socket_does_not_take_the_newer_sockets_answers() {
    frames_from_an_older_socket_do_not_take_the_newer_sockets_answers(As::User, &[0, 1, 2]).await;
}
