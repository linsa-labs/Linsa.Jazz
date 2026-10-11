#![cfg(feature = "test")]

//! A sync server that cannot fetch its JWT keys has not refused anyone.
//!
//! The sync server comes up before the authority that serves its keys on every stack start
//! (the authority waits for the sync server to be healthy), and stays without keys while the
//! authority is down for longer than the cache lasts. A client that handshakes then holds a
//! valid token; the server only cannot check it yet.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::{Json, Router, routing::get};
use base64::Engine;
use futures::{SinkExt as _, StreamExt as _};
use jazz_tools::server::{JazzServer, TestJwtIssuer};
use jazz_tools::sync_manager::ClientId;
use jazz_tools::transport_manager::{
    AuthConfig, AuthHandshake, ConnectedResponse, SYNC_PROTOCOL_VERSION,
};
use jazz_tools::transport_protocol::{ErrorCode, ServerEvent};
use jazz_tools::{
    AppContext, ClientStorage, ColumnType, DurabilityTier, JazzClient, Schema, SchemaBuilder,
    TableSchema, Value,
};
use serde_json::json;
use tempfile::TempDir;
use tokio_tungstenite::{connect_async, tungstenite::Message};

const JWT_KID: &str = "test-jwks-kid";
const JWT_SECRET: &str = "test-jwt-secret-for-integration";

/// The key authority: serves the keyset while up, refuses while down, answers each
/// request after `delay_ms`, and counts what it served and refused.
#[derive(Clone)]
struct Authority {
    up: Arc<AtomicBool>,
    refused: Arc<AtomicUsize>,
    served: Arc<AtomicUsize>,
    delay_ms: Arc<AtomicU64>,
}

struct AuthorityServer {
    authority: Authority,
    addr: SocketAddr,
    task: tokio::task::JoinHandle<()>,
}

impl AuthorityServer {
    async fn start() -> Self {
        let authority = Authority {
            up: Arc::new(AtomicBool::new(true)),
            refused: Arc::new(AtomicUsize::new(0)),
            served: Arc::new(AtomicUsize::new(0)),
            delay_ms: Arc::new(AtomicU64::new(0)),
        };
        let app = Router::new()
            .route("/jwks", get(keys))
            .with_state(authority.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind authority");
        let addr = listener.local_addr().expect("authority addr");
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.expect("serve authority");
        });
        Self {
            authority,
            addr,
            task,
        }
    }

    fn url(&self) -> String {
        format!("http://{}/jwks", self.addr)
    }

    fn set_up(&self, up: bool) {
        self.authority.up.store(up, Ordering::SeqCst);
    }

    fn set_delay(&self, delay: Duration) {
        self.authority
            .delay_ms
            .store(delay.as_millis() as u64, Ordering::SeqCst);
    }

    fn refused(&self) -> usize {
        self.authority.refused.load(Ordering::SeqCst)
    }

    fn served(&self) -> usize {
        self.authority.served.load(Ordering::SeqCst)
    }

    /// Waits until the authority has refused `count` key requests in all.
    async fn wait_refused(&self, count: usize, timeout: Duration) {
        let deadline = tokio::time::Instant::now() + timeout;
        while self.refused() < count {
            assert!(
                tokio::time::Instant::now() < deadline,
                "the authority was asked for keys {} times, expected {count}",
                self.refused()
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }
}

impl Drop for AuthorityServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn keys(State(authority): State<Authority>) -> Response {
    let delay = authority.delay_ms.load(Ordering::SeqCst);
    if delay > 0 {
        tokio::time::sleep(Duration::from_millis(delay)).await;
    }
    if !authority.up.load(Ordering::SeqCst) {
        authority.refused.fetch_add(1, Ordering::SeqCst);
        return StatusCode::SERVICE_UNAVAILABLE.into_response();
    }
    authority.served.fetch_add(1, Ordering::SeqCst);
    let encoded = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(JWT_SECRET.as_bytes());
    Json(json!({
        "keys": [{ "kty": "oct", "kid": JWT_KID, "alg": "HS256", "k": encoded }]
    }))
    .into_response()
}

fn test_schema() -> Schema {
    SchemaBuilder::new()
        .table(
            TableSchema::builder("todos")
                .column("title", ColumnType::Text)
                .column("completed", ColumnType::Boolean),
        )
        .build()
}

async fn start_server(authority: &AuthorityServer, dir: &TempDir, port: Option<u16>) -> JazzServer {
    let mut builder = JazzServer::builder()
        .with_schema(test_schema())
        .with_persistent_storage()
        .with_data_dir(dir.path())
        .with_jwks_url(authority.url());
    if let Some(port) = port {
        builder = builder.with_port(port);
    }
    builder.start().await
}

fn frame_encode(payload: &[u8]) -> Vec<u8> {
    let compressed = lz4_flex::compress_prepend_size(payload);
    let mut out = Vec::with_capacity(4 + compressed.len());
    out.extend_from_slice(&(compressed.len() as u32).to_be_bytes());
    out.extend_from_slice(&compressed);
    out
}

fn frame_decode(data: &[u8]) -> Option<Vec<u8>> {
    let len = u32::from_be_bytes(data.get(0..4)?.try_into().ok()?) as usize;
    lz4_flex::decompress_size_prepended(data.get(4..4 + len)?).ok()
}

enum Handshake {
    Connected,
    Error { code: ErrorCode, message: String },
}

/// One handshake with a valid user token; what the server answered.
async fn handshake(server: &JazzServer) -> Handshake {
    let ws_url = format!(
        "ws://127.0.0.1:{}/apps/{}/ws",
        server.port(),
        server.app_id()
    );
    let (mut ws, _) = connect_async(&ws_url).await.expect("ws connect");
    let frame = AuthHandshake {
        sync_protocol_version: SYNC_PROTOCOL_VERSION,
        client_id: ClientId::new().to_string(),
        auth: AuthConfig {
            jwt_token: Some(TestJwtIssuer::jwt_for_user("raw-handshake")),
            backend_secret: None,
            admin_secret: None,
            backend_session: None,
        },
        catalogue_state_hash: None,
        declared_schema_hash: None,
        acks_deliveries: false,
    };
    let payload = serde_json::to_vec(&frame).expect("serialize handshake");
    ws.send(Message::Binary(frame_encode(&payload).into()))
        .await
        .expect("send handshake");
    let Some(Ok(Message::Binary(bytes))) = ws.next().await else {
        panic!("the server answered the handshake with no frame");
    };
    let inner = frame_decode(&bytes).expect("handshake response frame");
    if serde_json::from_slice::<ConnectedResponse>(&inner).is_ok() {
        return Handshake::Connected;
    }
    match serde_json::from_slice::<ServerEvent>(&inner).expect("handshake response") {
        ServerEvent::Error { code, message } => Handshake::Error { code, message },
        other => panic!("unexpected handshake response: {}", other.variant_name()),
    }
}

// The refusal that parks a client is `Unauthorized`: every other code is retried. Without
// keys the server must not send it, and must not tell an unauthenticated peer where its
// keys live.
#[tokio::test]
async fn a_server_without_keys_does_not_refuse_the_token() {
    let authority = AuthorityServer::start().await;
    authority.set_up(false);
    let dir = TempDir::new().expect("server dir");
    let server = start_server(&authority, &dir, None).await;

    match handshake(&server).await {
        Handshake::Connected => panic!("a server without keys accepted a token"),
        Handshake::Error { code, message } => {
            assert_ne!(
                code,
                ErrorCode::Unauthorized,
                "a server that could not fetch its keys refused the token: {message}"
            );
            let authority_addr = authority.addr.to_string();
            assert!(
                !message.contains(&authority_addr) && !message.contains("http"),
                "the answer names the key authority: {message}"
            );
        }
    }

    // Retried as a client retries; it is accepted once the keys can be fetched.
    authority.set_up(true);
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    loop {
        match handshake(&server).await {
            Handshake::Connected => break,
            Handshake::Error { code, message } => {
                assert_ne!(
                    code,
                    ErrorCode::Unauthorized,
                    "the token was refused: {message}"
                );
                assert!(
                    tokio::time::Instant::now() < deadline,
                    "the same token was not accepted once the keys could be fetched: {message}"
                );
            }
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    server.shutdown().await;
}

// A connected client; the sync server restarts while the authority is down, the client's
// reconnects meet it without keys, then the authority comes back. The client must reach the
// server again with the token it already has, without a relaunch.
#[tokio::test]
async fn a_client_reconnects_once_the_restarted_server_has_keys() {
    let authority = AuthorityServer::start().await;
    let dir = TempDir::new().expect("server dir");
    let server = start_server(&authority, &dir, None).await;
    let port = server.port();
    let client_dir = TempDir::new().expect("client dir");
    let client = JazzClient::connect(AppContext {
        app_id: server.app_id(),
        client_id: None,
        schema: test_schema(),
        server_url: server.base_url(),
        data_dir: client_dir.path().to_path_buf(),
        storage: ClientStorage::Persistent,
        jwt_token: Some(TestJwtIssuer::jwt_for_user("keys-come-later")),
        backend_secret: None,
        admin_secret: None,
        sync_tracer: None,
    })
    .await
    .expect("connect client");

    authority.set_up(false);
    server.shutdown().await;
    let restarted = start_server(&authority, &dir, Some(port)).await;
    // The restarted server's warm-up asks once; anything after it is a handshake checked
    // without keys.
    authority.wait_refused(2, Duration::from_secs(15)).await;

    let (_, _, row) = client
        .insert(
            "todos",
            HashMap::from([
                (
                    "title".to_string(),
                    Value::Text("written without keys".to_string()),
                ),
                ("completed".to_string(), Value::Boolean(false)),
            ]),
        )
        .expect("write while the server has no keys");

    authority.set_up(true);
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    let mut last = None;
    while tokio::time::Instant::now() < deadline {
        match tokio::time::timeout(
            Duration::from_secs(5),
            client.wait_for_batch(row, DurabilityTier::EdgeServer),
        )
        .await
        {
            Ok(Ok(_)) => {
                client.shutdown().await.expect("shutdown client");
                restarted.shutdown().await;
                return;
            }
            other => last = Some(format!("{other:?}")),
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    panic!(
        "the client did not reach the server in 30 s after its keys came back; last: {}",
        last.unwrap_or_default()
    );
}

// Without keys every handshake asks the authority for them, and each request can wait out
// its own timeout. One request runs at a time, and a failed one answers for a second.
#[tokio::test]
async fn handshakes_without_keys_ask_the_authority_once() {
    let authority = AuthorityServer::start().await;
    authority.set_up(false);
    authority.set_delay(Duration::from_millis(200));
    let dir = TempDir::new().expect("server dir");
    let server = start_server(&authority, &dir, None).await;
    // The warm-up's refusal, then past its hold.
    authority.wait_refused(1, Duration::from_secs(10)).await;
    tokio::time::sleep(Duration::from_millis(1_200)).await;

    let before = authority.refused();
    let answers = futures::future::join_all((0..20).map(|_| handshake(&server))).await;
    assert!(
        answers.iter().all(|answer| matches!(
            answer,
            Handshake::Error { code, .. } if *code != ErrorCode::Unauthorized
        )),
        "a handshake without keys was accepted or refused"
    );
    let asked = authority.refused() - before;
    assert!(
        asked <= 2,
        "20 handshakes without keys asked the authority {asked} times"
    );

    // The hold runs out: the next handshake asks again.
    tokio::time::sleep(Duration::from_millis(1_200)).await;
    let before = authority.refused();
    let _ = handshake(&server).await;
    assert_eq!(
        authority.refused() - before,
        1,
        "the failed fetch was held for good"
    );

    // Keys again, nothing cached: a burst fetches them once and every handshake connects.
    authority.set_up(true);
    tokio::time::sleep(Duration::from_millis(1_200)).await;
    let answers = futures::future::join_all((0..20).map(|_| handshake(&server))).await;
    assert!(
        answers
            .iter()
            .all(|answer| matches!(answer, Handshake::Connected)),
        "a handshake was not accepted once the keys could be fetched"
    );
    assert_eq!(
        authority.served(),
        1,
        "a burst fetched the keys more than once"
    );
    server.shutdown().await;
}
