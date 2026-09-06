//! WebSocket handler — handshake authentication, connection lifecycle, and cleanup.

//! HTTP routes for the Jazz server.

use std::sync::Arc;

use axum::{
    extract::State,
    extract::ws::{CloseFrame, Message, WebSocket, WebSocketUpgrade, close_code},
    http::HeaderMap,
    response::{IntoResponse, Response},
};

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use futures::Stream as _;
use tokio::sync::{AcquireError, OwnedSemaphorePermit, Semaphore};

use super::utils::connection_schema_diagnostics_for_declared_hash;
use crate::middleware::auth::{extract_session, validate_admin_secret, validate_backend_secret};
use crate::runtime_tokio::{StagePush, WaiterGuard, kib_permits};
use crate::server::{ClientFrameError, ConnectionState, SequencedSyncUpdate, ServerState};
use crate::sync_manager::{ClientId, InboxEntry};
const MAX_WS_SYNC_UPDATES_PER_FRAME: usize = 256;

/// Generous ceiling on the decompressed size of the pre-auth handshake frame.
/// An `AuthHandshake` is a few KB of JSON, so this only rejects an obvious LZ4
/// decompression bomb sent by an unauthenticated peer before any auth runs.
const MAX_HANDSHAKE_DECOMPRESSED_BYTES: usize = 1024 * 1024;

/// Maximum time the server waits for a client to send its `AuthHandshake`
/// frame after the WS upgrade completes. Closes the slowloris pattern
/// where an attacker pins server-side state by opening upgrades and never
/// sending the first frame. See jaz0-a803.
pub(crate) const HANDSHAKE_READ_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// Maximum size of an inbound WebSocket message (the compressed bytes on the
/// wire). This is axum's existing default, set explicitly so the limit is
/// visible and can later be made configurable. It is *not* a bound on the
/// decompressed payload — see `MAX_HANDSHAKE_DECOMPRESSED_BYTES` and the
/// follow-up on bounded framing for that.
const MAX_WS_MESSAGE_BYTES: usize = 64 * 1024 * 1024;

pub(super) async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
) -> Response {
    if state.shutdown.is_shutting_down() {
        return (
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            axum::Json(crate::jazz_transport::ErrorResponse::internal(
                "server is shutting down".to_string(),
            )),
        )
            .into_response();
    }

    ws.max_message_size(MAX_WS_MESSAGE_BYTES)
        .on_upgrade(move |socket| handle_ws_connection(socket, state, headers))
}

/// Outcome of authenticating a WS handshake.
#[derive(Debug)]
pub(super) enum WsClientSetup {
    Backend,
    Session(crate::query_manager::session::Session),
}

/// Authenticate a WebSocket `AuthHandshake`.
///
/// Priority is:
/// 1. `admin_secret` valid → `WsClientSetup::Backend`
/// 2. `backend_secret` present + no session header → `WsClientSetup::Backend`
/// 3. Otherwise → `extract_session` → `WsClientSetup::Session`
///
/// Returns `Err(message)` on auth failure; the caller should send a
/// `ServerEvent::Error` frame before closing.
pub(super) async fn authenticate_ws_handshake(
    handshake: &crate::transport_manager::AuthHandshake,
    request_headers: &HeaderMap,
    state: &Arc<ServerState>,
) -> Result<WsClientSetup, String> {
    use axum::http::HeaderValue;
    use base64::Engine as _;

    let auth = &handshake.auth;

    // `admin_secret` is an explicit request to run this WS transport as the
    // backend. Validate it first and short-circuit all user-scoped auth.
    if let Some(admin_secret) = auth.admin_secret.as_deref() {
        validate_admin_secret(Some(admin_secret), &state.auth_config)
            .map_err(|(_, msg)| msg.to_string())?;
        return Ok(WsClientSetup::Backend);
    }

    if request_uses_cookie_auth(handshake, request_headers, &state.auth_config) {
        validate_ws_cookie_origin(request_headers)?;
    }

    // Build a synthetic HeaderMap from the handshake auth fields, layered on
    // top of the original upgrade request so cookie-based auth remains visible.
    let mut headers = request_headers.clone();

    if let Some(jwt) = &auth.jwt_token {
        let value = HeaderValue::from_str(&format!("Bearer {jwt}"))
            .map_err(|e| format!("invalid jwt_token header value: {e}"))?;
        headers.insert(axum::http::header::AUTHORIZATION, value);
    }
    if let Some(secret) = &auth.backend_secret {
        let value = HeaderValue::from_str(secret)
            .map_err(|e| format!("invalid backend_secret header value: {e}"))?;
        headers.insert("X-Jazz-Backend-Secret", value);
    }
    if let Some(session_val) = &auth.backend_session {
        let json = serde_json::to_string(session_val)
            .map_err(|e| format!("failed to serialise backend_session: {e}"))?;
        let b64 = base64::engine::general_purpose::STANDARD.encode(json.as_bytes());
        let value = HeaderValue::from_str(&b64)
            .map_err(|e| format!("invalid backend_session header value: {e}"))?;
        headers.insert("X-Jazz-Session", value);
    }

    let has_jwt = headers.get(axum::http::header::AUTHORIZATION).is_some();
    let has_session_header = headers.get("X-Jazz-Session").is_some();
    let backend_secret = headers
        .get("X-Jazz-Backend-Secret")
        .and_then(|v| v.to_str().ok());

    // 2. Backend secret — only when no user-scoped JWT is present.  Clients
    //    that carry both a backend_secret and a jwt_token (e.g. test helpers
    //    that mirror the full credential set) must be treated as users so the
    //    connection carries a session for row-level policy evaluation.
    if backend_secret.is_some() && !has_jwt && !has_session_header {
        validate_backend_secret(backend_secret, &state.auth_config)
            .map_err(|(_, msg)| msg.to_string())?;
        return Ok(WsClientSetup::Backend);
    }

    // 3. JWT / session-impersonation path.
    let session = extract_session(
        &headers,
        state.app_id,
        &state.auth_config,
        state.jwt_verifier.as_deref(),
    )
    .await
    .map_err(|e| serde_json::to_string(&e).unwrap_or_else(|_| "authentication failed".into()))?;

    let session =
        session.ok_or_else(|| "Session required. Provide JWT or backend secret.".to_string())?;

    Ok(WsClientSetup::Session(session))
}

fn request_uses_cookie_auth(
    handshake: &crate::transport_manager::AuthHandshake,
    request_headers: &HeaderMap,
    auth_config: &crate::middleware::AuthConfig,
) -> bool {
    let Some(cookie_name) = auth_config.auth_cookie_name.as_deref() else {
        return false;
    };

    let has_explicit_auth = handshake.auth.jwt_token.is_some()
        || handshake.auth.backend_secret.is_some()
        || handshake.auth.backend_session.is_some()
        || handshake.auth.admin_secret.is_some()
        || request_headers
            .get(axum::http::header::AUTHORIZATION)
            .is_some()
        || request_headers.get("X-Jazz-Backend-Secret").is_some()
        || request_headers.get("X-Jazz-Session").is_some()
        || request_headers.get("X-Jazz-Admin-Secret").is_some();

    if has_explicit_auth {
        return false;
    }

    request_cookie_value(request_headers, cookie_name).is_some()
}

fn request_cookie_value<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    let cookie_header = headers
        .get(axum::http::header::COOKIE)
        .and_then(|value| value.to_str().ok())?;

    cookie_header.split(';').find_map(|segment| {
        let trimmed = segment.trim();
        let (candidate_name, candidate_value) = trimmed.split_once('=')?;
        if candidate_name == name && !candidate_value.is_empty() {
            Some(candidate_value)
        } else {
            None
        }
    })
}

fn validate_ws_cookie_origin(headers: &HeaderMap) -> Result<(), String> {
    let host = headers
        .get("X-Forwarded-Host")
        .or_else(|| headers.get(axum::http::header::HOST))
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| "Cookie auth requires Host header".to_string())?;

    let origin = headers
        .get(axum::http::header::ORIGIN)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| "Cookie auth requires Origin header".to_string())?;

    let origin_uri: axum::http::Uri = origin
        .parse()
        .map_err(|_| "Cookie auth requires a valid Origin header".to_string())?;
    let origin_authority = origin_uri
        .authority()
        .map(|authority| authority.as_str())
        .ok_or_else(|| "Cookie auth requires an Origin authority".to_string())?;

    let is_allowed_origin = origin_authority.eq_ignore_ascii_case(host)
        || is_loopback_cookie_origin(origin_uri.scheme_str(), origin_authority, host)?;

    if is_allowed_origin {
        Ok(())
    } else {
        Err("Cookie auth Origin must match Host".to_string())
    }
}

fn is_loopback_cookie_origin(
    origin_scheme: Option<&str>,
    origin_authority: &str,
    host: &str,
) -> Result<bool, String> {
    if !matches!(origin_scheme, Some("http") | Some("https")) {
        return Ok(false);
    }

    let origin_authority: axum::http::uri::Authority = origin_authority
        .parse()
        .map_err(|_| "Cookie auth requires a valid Origin authority".to_string())?;
    let host_authority: axum::http::uri::Authority = host
        .parse()
        .map_err(|_| "Cookie auth requires a valid Host header".to_string())?;

    Ok(
        is_loopback_dev_host(origin_authority.host())
            && is_loopback_dev_host(host_authority.host()),
    )
}

fn is_loopback_dev_host(host: &str) -> bool {
    let host = host.trim_matches(['[', ']']);
    host.eq_ignore_ascii_case("localhost")
        || host.to_ascii_lowercase().ends_with(".localhost")
        || host == "127.0.0.1"
        || host == "::1"
}

/// Send a `ServerEvent::Error` frame on the socket, best-effort.
async fn send_ws_error(socket: &mut WebSocket, message: &str) {
    send_ws_error_with_code(
        socket,
        crate::jazz_transport::ErrorCode::Unauthorized,
        message,
    )
    .await;
}

/// Send a `ServerEvent::Error` frame on the socket, best-effort.
///
/// Uses JSON encoding so a not-yet-authenticated peer (which doesn't know
/// the post-handshake binary wire format yet) can still decode the error.
async fn send_ws_error_with_code(
    socket: &mut WebSocket,
    code: crate::jazz_transport::ErrorCode,
    message: &str,
) {
    let event = crate::jazz_transport::ServerEvent::Error {
        message: message.to_string(),
        code,
    };
    if let Ok(bytes) = serde_json::to_vec(&event) {
        let frame = crate::transport_manager::frame_encode(&bytes);
        let _ = socket.send(Message::Binary(frame)).await;
    }
}

/// Send a `ServerEvent::Error` frame using the post-handshake binary
/// wire format. Use this from any path that runs after the
/// `ConnectedResponse` has been sent — clients post-handshake parse
/// frames via `ServerEvent::decode_payload`, not JSON.
async fn send_ws_error_binary(
    socket: &mut WebSocket,
    code: crate::jazz_transport::ErrorCode,
    message: &str,
) {
    let event = crate::jazz_transport::ServerEvent::Error {
        message: message.to_string(),
        code,
    };
    if let Ok(bytes) = event.encode_payload() {
        let frame = crate::transport_manager::frame_encode(&bytes);
        let _ = socket.send(Message::Binary(frame)).await;
    }
}

async fn close_ws_with_protocol_reason(socket: &mut WebSocket, reason: &str) {
    let reason = reason.chars().take(123).collect::<String>();
    let _ = socket
        .send(Message::Close(Some(CloseFrame {
            code: close_code::PROTOCOL,
            reason: reason.into(),
        })))
        .await;
}

async fn close_ws_with_policy_reason(socket: &mut WebSocket, reason: &str) {
    let reason = reason.chars().take(123).collect::<String>();
    let _ = socket
        .send(Message::Close(Some(CloseFrame {
            code: close_code::POLICY,
            reason: reason.into(),
        })))
        .await;
}

async fn close_ws_for_shutdown(socket: &mut WebSocket) {
    let _ = socket
        .send(Message::Close(Some(CloseFrame {
            code: close_code::RESTART,
            reason: "server shutting down".into(),
        })))
        .await;
}

async fn handle_ws_connection(
    mut socket: WebSocket,
    state: Arc<ServerState>,
    request_headers: HeaderMap,
) {
    let mut shutdown_rx = state.shutdown.subscribe();
    let Some(_websocket_guard) = state.shutdown.try_enter_websocket() else {
        close_ws_for_shutdown(&mut socket).await;
        return;
    };
    if state.shutdown.is_shutting_down() {
        close_ws_for_shutdown(&mut socket).await;
        return;
    }

    // 1. Read the first binary frame — expected to be AuthHandshake.
    //    Bounded read so unauthenticated peers can't pin server-side
    //    resources by opening upgrades without sending a handshake.
    let first = tokio::select! {
        msg = socket.recv() => match msg {
            Some(Ok(Message::Binary(b))) => b,
            _ => {
                let _ = socket.close().await;
                return;
            }
        },
        changed = shutdown_rx.changed() => {
            if changed.is_ok() && state.shutdown.is_shutting_down() {
                close_ws_for_shutdown(&mut socket).await;
            } else {
                let _ = socket.close().await;
            }
            return;
        }
        _ = tokio::time::sleep(HANDSHAKE_READ_TIMEOUT) => {
            close_ws_with_policy_reason(&mut socket, "handshake timeout").await;
            return;
        }
    };
    let payload = match crate::transport_manager::frame_decode_capped(
        &first,
        MAX_HANDSHAKE_DECOMPRESSED_BYTES,
    ) {
        Some(payload) => payload,
        None => {
            let _ = socket.close().await;
            return;
        }
    };
    let handshake =
        match serde_json::from_slice::<crate::transport_manager::AuthHandshake>(&payload) {
            Ok(h) => h,
            Err(_) => {
                let _ = socket.close().await;
                return;
            }
        };

    // Older, pre-versioned clients deserialize as protocol version 0. Reject
    // them explicitly so developers see an actionable update prompt instead
    // of a dropped socket.
    if handshake.sync_protocol_version != crate::transport_manager::SYNC_PROTOCOL_VERSION {
        let message = format!(
            "Incompatible Jazz sync protocol: client sent {}, server requires {}. Please update Jazz.",
            handshake.sync_protocol_version,
            crate::transport_manager::SYNC_PROTOCOL_VERSION,
        );
        // Use BadRequest here so older clients that do not know newer error
        // codes can still deserialize and log the message.
        send_ws_error_with_code(
            &mut socket,
            crate::jazz_transport::ErrorCode::BadRequest,
            &message,
        )
        .await;
        close_ws_with_protocol_reason(&mut socket, &message).await;
        return;
    }

    // 2. Parse client_id.
    let client_id = match crate::sync_manager::ClientId::parse(&handshake.client_id) {
        Some(id) => id,
        None => {
            send_ws_error(&mut socket, "missing or invalid client_id").await;
            let _ = socket.close().await;
            return;
        }
    };

    // 3. Authenticate.
    let setup = tokio::select! {
        auth = authenticate_ws_handshake(&handshake, &request_headers, &state) => match auth {
            Ok(s) => s,
            Err(msg) => {
                send_ws_error(&mut socket, &msg).await;
                let _ = socket.close().await;
                return;
            }
        },
        changed = shutdown_rx.changed() => {
            if changed.is_ok() && state.shutdown.is_shutting_down() {
                close_ws_for_shutdown(&mut socket).await;
            } else {
                let _ = socket.close().await;
            }
            return;
        }
    };
    let role = match &setup {
        WsClientSetup::Backend => "backend",
        WsClientSetup::Session(_) => "session",
    };

    // 4. Register with ConnectionEventHub (mirrors events_handler).
    let connection_id = state
        .next_connection_id
        .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let crate::server::ConnectionRegistration {
        next_sync_seq,
        receiver: mut sync_rx,
        evicted: evicted_flag,
    } = state
        .connection_event_hub
        .register_connection(connection_id, client_id);
    {
        let mut connections = state.connections.write().await;
        connections.insert(connection_id, ConnectionState { client_id });
    }
    state.on_client_connected(client_id).await;

    // 5. Register the client with the runtime. Four calls under the engine lock, run on
    //    the blocking pool with owned values, so a pass holding the lock parks a pool
    //    thread, not this worker. `Connected` goes out after all four, so no client frame
    //    precedes `set_client_acks_deliveries`.
    let registration = {
        let blocking_state = Arc::clone(&state);
        let catalogue_state_hash = handshake.catalogue_state_hash.clone();
        let declared_schema_hash = handshake.declared_schema_hash();
        let acks_deliveries = handshake.acks_deliveries;
        let task = tokio::task::spawn_blocking(move || {
            register_ws_client(
                &blocking_state,
                client_id,
                setup,
                catalogue_state_hash.as_deref(),
                acks_deliveries,
                declared_schema_hash,
            )
        });
        tokio::select! {
            joined = task => match joined {
                Ok(registration) => registration,
                Err(error) => {
                    tracing::error!(%client_id, ?error, "ws client registration task failed");
                    ws_cleanup(&state, connection_id, client_id).await;
                    let _ = socket.close().await;
                    return;
                }
            },
            changed = shutdown_rx.changed() => {
                if changed.is_ok() && state.shutdown.is_shutting_down() {
                    close_ws_for_shutdown(&mut socket).await;
                } else {
                    let _ = socket.close().await;
                }
                ws_cleanup(&state, connection_id, client_id).await;
                return;
            }
        }
    };

    // 5b. Dispatch connection schema diagnostics if client sent a declared schema hash.
    match registration.diagnostics {
        Ok(Some(diagnostics)) => {
            state.connection_event_hub.dispatch_payload(
                client_id,
                crate::sync_manager::SyncPayload::ConnectionSchemaDiagnostics(diagnostics),
            );
        }
        Ok(None) => {}
        Err(err) => {
            tracing::error!(
                %client_id,
                declared_schema_hash = ?handshake.declared_schema_hash,
                "failed to compute connection schema diagnostics: {err}"
            );
        }
    }

    // 6. Send the Connected response.
    let resp = crate::transport_manager::ConnectedResponse {
        sync_protocol_version: crate::transport_manager::SYNC_PROTOCOL_VERSION,
        connection_id: connection_id.to_string(),
        client_id: client_id.to_string(),
        next_sync_seq: Some(next_sync_seq),
        catalogue_state_hash: registration.catalogue_state_hash,
        supports_delivery_acks: true,
    };
    let resp_bytes = match serde_json::to_vec(&resp) {
        Ok(b) => b,
        Err(_) => {
            ws_cleanup(&state, connection_id, client_id).await;
            let _ = socket.close().await;
            return;
        }
    };
    let connected_frame = crate::transport_manager::frame_encode(&resp_bytes);
    if socket.send(Message::Binary(connected_frame)).await.is_err() {
        ws_cleanup(&state, connection_id, client_id).await;
        return;
    }
    tracing::info!(connection_id, %client_id, role, "ws client connected");

    // 7. Bidirectional loop: inbound frames from client + outbound updates from hub.
    //    Also fires a periodic heartbeat so idle connections don't look half-open.
    //
    //    v18 item 3: an inbound frame is never handed to the engine lock from here. Its
    //    declared decoded size is checked against the cap and charged to the in-flight
    //    budget (the socket waits for permits, FIFO, keeping its queue place across
    //    iterations), then it is decoded and staged; a client at its staging cap pauses this
    //    socket's inbound side until the next drain. Outbound keeps flowing while inbound is
    //    paused; `sync_rx` and `recv` alternate their order every iteration so neither side
    //    starves the other at line rate. (It does not — see `poll_socket_or_hub`, which
    //    carries the code reading and the measurement. The flip is kept because removing it
    //    is also a behaviour change, and nothing measured asks for either.)
    let mut heartbeat = tokio::time::interval(std::time::Duration::from_secs(30));
    // Don't emit a heartbeat immediately after Connected — wait a full tick.
    heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    heartbeat.tick().await; // consume the immediate first tick
    let budget = state.runtime.inflight_budget();
    let max_decoded = state.max_ws_decoded_frame_bytes;
    let mut pending: Option<Pending> = None;
    let mut inbound_first = false;
    loop {
        let awaiting_budget = matches!(pending, Some(Pending::AwaitingBudget { .. }));
        let staging_notify = match &pending {
            Some(Pending::AwaitingStaging { waiter, .. }) => Some(waiter.notify_handle()),
            _ => None,
        };
        let inbound_enabled = pending.is_none();
        inbound_first = !inbound_first;
        tokio::select! {
            biased;
            changed = shutdown_rx.changed() => {
                if changed.is_ok() && state.shutdown.is_shutting_down() {
                    close_ws_for_shutdown(&mut socket).await;
                    break;
                }
            }
            // The budget arm polls the acquire future that lives in the pending state, so a
            // delivery or heartbeat winning the select does not cancel it and lose its
            // queue place.
            acquired = std::future::poll_fn(|cx| match pending.as_mut() {
                Some(Pending::AwaitingBudget { acquire, .. }) => acquire.as_mut().poll(cx),
                _ => Poll::Pending,
            }), if awaiting_budget => {
                let Some(Pending::AwaitingBudget { raw, declared, .. }) = pending.take() else {
                    unreachable!("the budget arm is enabled only while a frame awaits budget");
                };
                state.budget_waiters.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                let permit = acquired.expect("the in-flight budget semaphore is never closed");
                match admit_client_frame(&state, client_id, &mut socket, raw, declared, permit).await {
                    FrameOutcome::Done => {}
                    FrameOutcome::Paused(next) => pending = Some(next),
                    FrameOutcome::Closed => break,
                }
            }
            _ = async {
                match &staging_notify {
                    Some(notify) => notify.notified().await,
                    None => std::future::pending::<()>().await,
                }
            }, if staging_notify.is_some() => {
                let Some(Pending::AwaitingStaging { entries, bytes, permit, waiter }) = pending.take() else {
                    unreachable!("the staging arm is enabled only while a frame awaits staging");
                };
                match state
                    .runtime
                    .stage_sync_inbox_with_waiter(waiter, client_id, entries, bytes, permit)
                {
                    StagePush::Staged => {}
                    StagePush::Backpressure { entries, permit, waiter } => {
                        pending = Some(Pending::AwaitingStaging { entries, bytes, permit, waiter });
                    }
                }
            }
            event = std::future::poll_fn(|cx| {
                poll_socket_or_hub(&mut socket, &mut sync_rx, inbound_enabled, inbound_first, cx)
            }) => match event {
                SocketEvent::Inbound(msg) => match msg {
                    Some(Ok(Message::Binary(data))) => {
                        match receive_client_frame(
                            &state,
                            client_id,
                            &mut socket,
                            data,
                            &budget,
                            max_decoded,
                        )
                        .await
                        {
                            FrameOutcome::Done => {}
                            FrameOutcome::Paused(next) => pending = Some(next),
                            FrameOutcome::Closed => break,
                        }
                    }
                    Some(Ok(Message::Close(_))) | None => break,
                    _ => {}
                },
                SocketEvent::Outbound(update) => {
                    let Some(u) = update else {
                        // Distinguish per-client-cap eviction from a normal
                        // disconnect, so the evicted client gets a programmatic
                        // signal instead of an unexplained TCP close.
                        if evicted_flag.load(std::sync::atomic::Ordering::SeqCst) {
                            send_ws_error_binary(
                                &mut socket,
                                crate::jazz_transport::ErrorCode::RateLimited,
                                "per-client connection cap exceeded",
                            )
                            .await;
                            close_ws_with_policy_reason(
                                &mut socket,
                                "per-client connection cap exceeded",
                            )
                            .await;
                        }
                        break;
                    };
                    let mut updates = Vec::with_capacity(MAX_WS_SYNC_UPDATES_PER_FRAME);
                    updates.push(crate::jazz_transport::SequencedSyncPayload {
                        seq: Some(u.seq),
                        payload: u.payload,
                    });
                    while updates.len() < MAX_WS_SYNC_UPDATES_PER_FRAME {
                        match sync_rx.try_recv() {
                            Ok(u) => updates.push(crate::jazz_transport::SequencedSyncPayload {
                                seq: Some(u.seq),
                                payload: u.payload,
                            }),
                            Err(tokio::sync::mpsc::error::TryRecvError::Empty) => break,
                            Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => break,
                        }
                    }
                    let event = if updates.len() == 1 {
                        let update = updates.pop().expect("single update is present");
                        crate::jazz_transport::ServerEvent::SyncUpdate {
                            seq: update.seq,
                            payload: Box::new(update.payload),
                        }
                    } else {
                        crate::jazz_transport::ServerEvent::SyncUpdateBatch { updates }
                    };
                    let bytes = match event.encode_payload() {
                        Ok(b) => b,
                        Err(error) => {
                            // This payload is GONE, and the delivery bookkeeping has no way to
                            // learn that: silently continuing leaves the connection healthy and
                            // the row undelivered until the client happens to resubscribe.
                            // Tearing the connection down makes the client's transport
                            // reconnect, replay its subscriptions, and pick the row back up
                            // through the re-offer.
                            tracing::error!(
                                %client_id,
                                ?error,
                                "sync frame failed to encode; tearing the connection down so \
                                 the reconnect replay can re-offer what this frame carried"
                            );
                            break;
                        }
                    };
                    let frame = crate::transport_manager::frame_encode(&bytes);
                    if socket.send(Message::Binary(frame)).await.is_err() {
                        break;
                    }
                }
            },
            _ = heartbeat.tick() => {
                let event = crate::jazz_transport::ServerEvent::Heartbeat;
                let Ok(bytes) = event.encode_payload() else { continue };
                let frame = crate::transport_manager::frame_encode(&bytes);
                if socket.send(Message::Binary(frame)).await.is_err() {
                    break;
                }
            }
        }
    }
    // A decoded frame this socket admitted is staged on the way out, regardless of the
    // caps (the connection is gone and cannot push more) — BEFORE `ws_cleanup` inserts the
    // reap candidate, so a sweep never sees a gone client with an unstaged frame. A frame
    // still waiting for budget was never admitted: it is dropped, and dropping its acquire
    // returns whatever permits it held.
    match pending.take() {
        Some(Pending::AwaitingStaging {
            entries,
            bytes,
            permit,
            waiter,
        }) => {
            state.runtime.stage_sync_inbox_uncapped(
                Some(waiter),
                client_id,
                entries,
                bytes,
                permit,
            );
        }
        Some(Pending::AwaitingBudget { .. }) => {
            state
                .budget_waiters
                .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
        }
        None => {}
    }

    ws_cleanup(&state, connection_id, client_id).await;
    let _ = socket.close().await;
}

/// Disconnect cleanup: mirrors the drop path in `events_handler`.
async fn ws_cleanup(state: &Arc<ServerState>, connection_id: u64, client_id: ClientId) {
    {
        let mut connections = state.connections.write().await;
        connections.remove(&connection_id);
    }
    state
        .connection_event_hub
        .unregister_connection(connection_id);
    state.on_connection_closed(client_id).await;
}

/// What the blocking-pool registration of a client hands back to the handshake.
struct WsClientRegistration {
    diagnostics: Result<
        Option<crate::sync_manager::ConnectionSchemaDiagnostics>,
        crate::runtime_tokio::RuntimeError,
    >,
    catalogue_state_hash: Option<String>,
}

/// The handshake's four calls under the engine lock, in their original order.
fn register_ws_client(
    state: &ServerState,
    client_id: ClientId,
    setup: WsClientSetup,
    catalogue_state_hash: Option<&str>,
    acks_deliveries: bool,
    declared_schema_hash: Option<crate::query_manager::types::SchemaHash>,
) -> WsClientRegistration {
    // 5. Ensure the client state in the runtime.
    match setup {
        WsClientSetup::Backend => {
            let _ = state
                .runtime
                .ensure_client_as_backend_with_catalogue_state_hash(
                    client_id,
                    catalogue_state_hash,
                );
        }
        WsClientSetup::Session(session) => {
            let _ = state
                .runtime
                .ensure_client_with_session_and_catalogue_state_hash(
                    client_id,
                    session,
                    catalogue_state_hash,
                );
        }
    }
    // 5a. Record whether this client confirms what it applies. Known here, before it
    // subscribes, which is what the delivery bookkeeping needs: a client that does not
    // confirm keeps the old claim-at-queue behaviour rather than accumulating rows nothing
    // will ever clear.
    if let Err(error) = state
        .runtime
        .set_client_acks_deliveries(client_id, acks_deliveries)
    {
        tracing::warn!(%client_id, ?error, "could not record the client's delivery-ack capability");
    }
    let diagnostics = connection_schema_diagnostics_for_declared_hash(state, declared_schema_hash);
    let catalogue_state_hash = state.runtime.catalogue_state_hash().ok();
    WsClientRegistration {
        diagnostics,
        catalogue_state_hash,
    }
}

/// v18 item 3: a frame the socket has received but not yet handed to the runtime. One
/// socket holds at most one.
enum Pending {
    /// Waiting for the in-flight decoded-bytes budget. The raw (compressed) frame is kept,
    /// undecoded; the acquire future keeps its queue place across loop iterations.
    AwaitingBudget {
        raw: Vec<u8>,
        declared: usize,
        acquire: Pin<Box<dyn Future<Output = Result<OwnedSemaphorePermit, AcquireError>> + Send>>,
    },
    /// Refused by the client's staging cap. The decoded entries are kept with the permit
    /// they were charged for and the waiter's place in the staging.
    AwaitingStaging {
        entries: Vec<InboxEntry>,
        bytes: usize,
        permit: Option<OwnedSemaphorePermit>,
        waiter: WaiterGuard,
    },
}

enum FrameOutcome {
    /// Staged, or dropped on shutdown.
    Done,
    /// The socket waits; inbound is paused until the state resolves.
    Paused(Pending),
    /// The connection was closed with an error frame; the loop exits.
    Closed,
}

/// A per-poll stack value, destructured on the next line (diff r7 S5): boxing the large
/// variant would put a heap allocation on the per-update transport path, for `None` too.
#[allow(clippy::large_enum_variant)]
enum SocketEvent {
    Inbound(Option<Result<Message, axum::Error>>),
    Outbound(Option<SequencedSyncUpdate>),
}

/// Poll the socket's inbound side and the hub's outbound side in alternating order, so
/// neither starves the other at line rate. Inbound is not polled while a frame is pending.
///
/// THE ALTERNATION DOES NOT ALTERNATE, and the caller's "so neither side starves the other at
/// line rate" is not a property this code provides (v18 item 3, diff r11 B1). Verified in
/// code: `inbound_first` flips at the top of every loop iteration, but on the happy path an
/// admitted frame costs exactly TWO iterations — this arm takes the frame and
/// `receive_client_frame` returns `Paused(AwaitingBudget)`, then the budget arm (arm 2 under
/// `biased;`, ready as soon as a permit is free) takes the next iteration and
/// `admit_client_frame` returns `Done`. A two-iteration cycle against a one-iteration flip
/// means this function observes the SAME parity on every visit.
///
/// What r11 drew from that — that outbound therefore starves — does NOT reproduce, and the
/// measurement is on the record because the conclusion is the kind that looks obvious and is
/// wrong. Driving a client that uploads until its own `flush()` backpressures, so the server's
/// receive buffer is full and this socket is readable on essentially every poll, a
/// server-initiated update still reached that client in 0 ms. Replacing the inbound-first
/// branch's `hub.poll_recv` fall-through with `Poll::Pending` — the mechanism proposed as the
/// reason it is fine — did not change that either. Two candidate mechanisms, both refuted by
/// the same fixture, which means the fixture is not sensitive to this ordering at all.
///
/// That is the same answer chain rows D18 and D18b already gave: pinning the order to
/// outbound-first AND to inbound-first both leave every gate green. A deterministic gate would
/// mean moving the `inbound_first` flip out of the caller's loop and into this function so a
/// fake stream could drive it — which turns a per-ITERATION alternation into a per-POLL one,
/// since `select!` may poll this more than once per iteration. That is a behaviour change made
/// only to be testable, so it is not made here, and neither is a "fix" to an ordering with no
/// measured defect behind it.
///
/// Read the two branches as what they are: a bias, ungated, whose observed parity is constant
/// and whose effect on delivery latency measured zero. Not as a fairness guarantee.
/// Residual (diff r4 SF2): while a frame is paused (`pending.is_some()`) the socket is not
/// polled, so tungstenite's automatic PONG is not sent for the length of the pause — the
/// length of the lock hold. A proxy or client with a ping deadline shorter than a pass
/// would drop the connection. Item 6 bounds the pass only when `JAZZ_SETTLE_BUDGET_MS` is
/// set (default unset), and then to the budget plus one settle unit.
fn poll_socket_or_hub(
    socket: &mut WebSocket,
    hub: &mut tokio::sync::mpsc::UnboundedReceiver<SequencedSyncUpdate>,
    inbound_enabled: bool,
    inbound_first: bool,
    cx: &mut Context<'_>,
) -> Poll<SocketEvent> {
    if inbound_first {
        if inbound_enabled && let Poll::Ready(msg) = Pin::new(&mut *socket).poll_next(cx) {
            return Poll::Ready(SocketEvent::Inbound(msg));
        }
        hub.poll_recv(cx).map(SocketEvent::Outbound)
    } else {
        if let Poll::Ready(update) = hub.poll_recv(cx) {
            return Poll::Ready(SocketEvent::Outbound(update));
        }
        if inbound_enabled {
            Pin::new(&mut *socket)
                .poll_next(cx)
                .map(SocketEvent::Inbound)
        } else {
            Poll::Pending
        }
    }
}

/// A received binary frame: read its declared decoded size (before any allocation), refuse
/// it over the cap, and queue for the budget.
async fn receive_client_frame(
    state: &ServerState,
    client_id: ClientId,
    socket: &mut WebSocket,
    data: Vec<u8>,
    budget: &Arc<Semaphore>,
    max_decoded: usize,
) -> FrameOutcome {
    let Some(declared) = crate::transport_manager::frame_declared_size(&data) else {
        tracing::warn!(%client_id, len = data.len(), "malformed client frame header; closing");
        return reject_client_frame(
            socket,
            "frame_corrupt",
            "frame_corrupt: malformed frame header",
        )
        .await;
    };
    if declared > max_decoded {
        tracing::warn!(
            %client_id,
            declared,
            compressed = data.len(),
            cap = max_decoded,
            "client frame declares a decoded size over the cap; closing"
        );
        return reject_client_frame(
            socket,
            "frame_too_large",
            &format!("frame_too_large: declared {declared} bytes, cap {max_decoded}"),
        )
        .await;
    }
    let permits = frame_permits(declared, data.len(), max_decoded);
    let acquire = Box::pin(Arc::clone(budget).acquire_many_owned(permits));
    state
        .budget_waiters
        .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    FrameOutcome::Paused(Pending::AwaitingBudget {
        raw: data,
        declared,
        acquire,
    })
}

/// Permits an inbound frame is charged against the decoded-bytes budget.
///
/// `declared` is a client-supplied LE u32 read out of the lz4 header; `held` is what the frame
/// actually occupies, which the socket parks for the whole of the budget wait. Charging
/// `declared` alone let a frame declare 4 bytes over a 64 MiB body, cost one permit, and hold
/// 64 MiB — and the lie is caught only AFTER the wait, in `admit_client_frame`'s length check.
/// So the charge is the larger of the two, which makes the budget a bound on memory in flight
/// rather than on a number the client picks.
///
/// The cap is not decoration. `kib_permits(max_decoded)` is the number the builder's startup
/// invariant proves fits both the budget and u32, and an acquire for more permits than the
/// semaphore will ever hold does not fail — it waits forever. Clamping is what keeps a hostile
/// `held` from wedging the connection instead of being refused.
///
/// Residual, named rather than closed: memory held is still bounded by `MAX_WS_MESSAGE_BYTES`
/// and not by `max_decoded`, because refusing an under-declaring header outright needs an
/// lz4-expansion bound evaluated on the transport hot path.
fn frame_permits(declared: usize, held: usize, max_decoded: usize) -> u32 {
    kib_permits(declared.max(held).min(max_decoded)) as u32
}

/// A frame whose budget is acquired: decode, check the declaration, decode the payload,
/// stage. Every failure drops `permit` (released) and closes the connection.
async fn admit_client_frame(
    state: &ServerState,
    client_id: ClientId,
    socket: &mut WebSocket,
    raw: Vec<u8>,
    declared: usize,
    permit: OwnedSemaphorePermit,
) -> FrameOutcome {
    let Some(decoded) = crate::transport_manager::frame_decode_body(&raw) else {
        tracing::warn!(%client_id, declared, "client frame body does not decode; closing");
        return reject_client_frame(
            socket,
            "frame_corrupt",
            "frame_corrupt: lz4 body does not decode",
        )
        .await;
    };
    drop(raw);
    if decoded.len() != declared {
        // A lying header would pin `declared` bytes of budget for a smaller frame (and
        // `vec![0; declared]` for its decode). Every client we ship writes the exact length.
        tracing::warn!(
            %client_id,
            declared,
            decoded = decoded.len(),
            "client frame declaration does not match its decoded length; closing"
        );
        return reject_client_frame(
            socket,
            "frame_size_mismatch",
            &format!(
                "frame_size_mismatch: declared {declared} bytes, decoded {}",
                decoded.len()
            ),
        )
        .await;
    }
    let entries = match state.decode_client_frame(client_id, &decoded) {
        Ok(entries) => entries,
        Err(ClientFrameError::ShuttingDown) => {
            tracing::debug!(%client_id, "client frame dropped: server is shutting down");
            return FrameOutcome::Done;
        }
        Err(ClientFrameError::Invalid(message)) => {
            tracing::warn!(%client_id, %message, "client frame payload is invalid; closing");
            return reject_client_frame(
                socket,
                "frame_corrupt",
                &format!("frame_corrupt: {message}"),
            )
            .await;
        }
    };
    drop(decoded);
    match state
        .runtime
        .stage_sync_inbox(client_id, entries, declared, Some(permit))
    {
        StagePush::Staged => FrameOutcome::Done,
        StagePush::Backpressure {
            entries,
            permit,
            waiter,
        } => FrameOutcome::Paused(Pending::AwaitingStaging {
            entries,
            bytes: declared,
            permit,
            waiter,
        }),
    }
}

/// Refuse a frame: an error frame the client can decode (`BadRequest`, like the version
/// mismatch — an old client cannot decode a new `ErrorCode`), then a policy close whose
/// reason names the rule.
async fn reject_client_frame(socket: &mut WebSocket, reason: &str, message: &str) -> FrameOutcome {
    send_ws_error_binary(
        socket,
        crate::jazz_transport::ErrorCode::BadRequest,
        message,
    )
    .await;
    close_ws_with_policy_reason(socket, reason).await;
    FrameOutcome::Closed
}

#[cfg(test)]
mod frame_permits_gates {
    use super::{MAX_WS_MESSAGE_BYTES, frame_permits};

    /// G-charge (v18 item 3, round 12). The in-flight budget must bound the bytes actually
    /// held, not the number the client writes into its own header.
    ///
    /// Internal on purpose: the charge is an accounting decision between the socket task and
    /// the semaphore. No client API reports how many permits a frame cost, and from outside an
    /// under-charged frame and an honest one are indistinguishable until the budget saturates
    /// — which is exactly the condition the budget exists for.
    #[test]
    fn a_frame_is_charged_for_what_it_holds_not_for_what_it_claims() {
        let cap = 1 << 20;

        // An honest frame: the declared size dominates, and the charge is unchanged from
        // before this round. This is the control — if it moves, every client pays more.
        assert_eq!(
            frame_permits(8 * 1024, 2 * 1024, cap),
            8,
            "an honest frame must still be charged its declared size"
        );

        // The defect: four declared bytes over a 32 KiB body cost one permit and held 32 KiB.
        assert_eq!(
            frame_permits(4, 32 * 1024, cap),
            32,
            "a frame that under-declares must be charged for the bytes it actually parks: at              one permit it holds 32 KiB against a budget that thinks it lent 1 KiB"
        );

        // The clamp. Without it a hostile `held` asks the semaphore for more permits than it
        // will ever hold, and `acquire_many_owned` does not fail on that — it waits forever,
        // so the frame is wedged rather than refused.
        assert_eq!(
            frame_permits(4, MAX_WS_MESSAGE_BYTES, cap),
            1024,
            "the charge must never exceed `kib_permits(max_decoded)`, the number the builder's              startup invariant proves fits in the budget and in u32"
        );
    }
}
