use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use tokio::sync::RwLock;
use tracing::info;

use crate::middleware::AuthConfig;
use crate::middleware::auth::{
    JWKS_CACHE_TTL, JWKS_MAX_STALE, JwksCache, JwtVerifier, StaticJwtVerifier,
};
use crate::query_manager::types::Schema;
use crate::routes;
use crate::runtime_tokio::TokioRuntime;
use crate::schema_manager::{AppId, SchemaManager, rehydrate_schema_manager_from_catalogue};
use crate::server::{ConnectionEventHub, DynStorage, ServerState, ServerTopology};
#[cfg(feature = "rocksdb")]
use crate::storage::RocksDBStorage;
#[cfg(feature = "sqlite")]
use crate::storage::SqliteStorage;
use crate::storage::{MemoryStorage, Storage};
use crate::sync_manager::{Destination, DurabilityTier, SyncManager};
use crate::transport_manager::TransportRetryConfig;

#[cfg(feature = "rocksdb")]
const STORAGE_CACHE_SIZE_BYTES: usize = 64 * 1024 * 1024;
const DEFAULT_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(30);
const DEFAULT_CLIENT_TTL: Duration = Duration::from_secs(300);
const EDGE_UPSTREAM_CONNECT_ATTEMPT_TIMEOUT: Duration = Duration::from_secs(5);
const EDGE_UPSTREAM_AUTH_HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(5);

pub struct BuiltServer {
    #[cfg_attr(not(test), allow(dead_code))]
    pub state: Arc<ServerState>,
    pub app: Router,
}

#[cfg_attr(not(test), allow(dead_code))]
enum ServerSchemaMode {
    Dynamic,
    Fixed(Schema),
}

/// Storage backend selection for [`ServerBuilder::with_storage`].
///
/// `Persistent` picks the best available backend at compile time
/// (RocksDB > SQLite > in-memory). `Sqlite` and `RocksDb` pin the backend
/// regardless of which other storage features are enabled.
#[derive(Debug, Clone)]
pub enum StorageBackend {
    InMemory,
    Persistent {
        path: PathBuf,
    },
    #[cfg(feature = "sqlite")]
    Sqlite {
        path: PathBuf,
    },
    #[cfg(feature = "rocksdb")]
    RocksDb {
        path: PathBuf,
    },
}

pub struct ServerBuilder {
    app_id: AppId,
    auth_config: AuthConfig,
    schema_mode: ServerSchemaMode,
    storage_backend: StorageBackend,
    sync_tracer: Option<crate::sync_tracer::SyncTracer>,
    upstream_url: Option<String>,
    shutdown_timeout: Duration,
    client_ttl: Duration,
    subscription_caps: Option<crate::sync_manager::SubscriptionCaps>,
    /// v18 item 3: staging caps and the in-flight decoded-bytes budget. Absent:
    /// `StagingConfig::from_env()`.
    staging_config: Option<crate::runtime_tokio::StagingConfig>,
    /// v18 item 3: the decoded-frame cap. Absent: `JAZZ_MAX_WS_DECODED_FRAME_BYTES` or
    /// `DEFAULT_MAX_WS_DECODED_FRAME_BYTES`.
    max_ws_decoded_frame_bytes: Option<usize>,
}

/// The decoded size a post-handshake client frame may declare: 256 max-size payloads (a
/// BYTEA is at most 1 MiB, an old client coalesces up to 256 payloads per frame with no
/// byte bound) plus envelope headroom. Chosen for the APP envelope — the engine allows
/// unbounded text/array/row values; a deployment with larger rows raises the knob.
pub const DEFAULT_MAX_WS_DECODED_FRAME_BYTES: usize = 288 * 1024 * 1024;

fn max_ws_decoded_frame_bytes_from_env() -> usize {
    std::env::var("JAZZ_MAX_WS_DECODED_FRAME_BYTES")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(DEFAULT_MAX_WS_DECODED_FRAME_BYTES)
}

/// v18 item 6: how long one settle pass may spend on charged units before deferring the rest.
///
/// 50 ms, and every leg of that is measurable rather than taste:
///
/// * The tick thread sleeps a fixed 1 ms before taking the core lock, paid once per re-armed
///   continuation, so a small budget spends a large fraction of itself on scheduling. At 50 ms
///   that is under 2 %; at 5 ms it is 20 %.
/// * Prod's `loopMax` is 24 ms median, 38 ms p90 (`archive/v18/stand-before.txt`), so a 50 ms
///   hold is inside the stall the JS side already absorbs, while the tail it cuts is the 609 ms
///   maximum and the 1.2 s passes.
/// * The facade times a read out at 5 s, which leaves two orders of magnitude of queueing.
/// * It does not trip on ordinary passes, so the new scheduler behaves like the old one except
///   on the pathological ones — the lowest-regression operating point for a first default.
///
/// What it does NOT do, and the stand claim must not say otherwise: a unit is not preemptible,
/// so the budget bounds how many units a pass runs, not how long one takes. Prod's most
/// expensive observed unit is a single `messages` registration at ~0.46 s (REPRO.md row E).
/// The honest ceiling is therefore "worst hold ~ prologue + one unit", not "worst hold ~ 50 ms".
///
/// Unused while `SETTLE_BUDGET_DEFAULT` is `None`; kept with the reasoning that picked it so
/// that turning item 6 on is one edit rather than a rediscovery of why 50 ms.
#[allow(dead_code)]
const DEFAULT_SETTLE_BUDGET_MICROS: u64 = 50_000;

/// What an unconfigured server actually gets — `None`. Item 6 is OFF, and the history of this
/// one constant is worth keeping, because two of the three reasons given for it were wrong.
///
/// **Retracted reason 1 (mine).** This comment once said enabling the budget "broke delivery":
/// an edit made while a peer was offline never reaching it on return. Measured false, 8/8
/// deterministically at each budget (diff r27) — under a budget the owed row arrives EARLIER,
/// before the resubscribe, so the first delta carries it as `added` and no `updated` follows.
/// The peer's store is correct at every budget. The test asserted the delta KIND; it now asserts
/// the value in the peer's store and passes everywhere.
///
/// **Retracted reason 2.** "No test runs the pool at a non-zero budget" (diff r27's census) was
/// true and is now closed: the randomised differential replays every op sequence at
/// `Some(50_000)` as well as `None` and `Some(0)`, and G6-8 pins the returning-peer recovery at
/// four budgets. The full suite is green with this constant set to `Some(50_000)`: 2139 passed,
/// 2 failed, both being snapshot tests that already fail at the base commit `c9ec20fb0`.
///
/// **The reason it is still `None`, found by writing the gate the census implied.**
/// `a_re_dirtied_subscription_does_not_starve_its_sibling_on_the_same_client` (G6-9): the pool's
/// rotation is over SLOTS, and a slot is a CLIENT. `server_units.sort()` then `push_back`
/// (`server_queries.rs`) gives a client's slot the same `(ClientId, QueryId)` order on every
/// pass, and `rotation_cursor` only moves between slots — so a pass that runs one unit runs the
/// same subscription of that client forever:
///
/// ```text
///   None          both siblings settle          (control)
///   Some(0)       hot 12 settles, quiet 0       STARVED
///   Some(50_000)  both settle                   only because both units fit in one pass
/// ```
///
/// The 50 ms column is not reassurance. Prod's most expensive single unit is a `messages`
/// registration at ~0.46 s (`archive/prodrepro/REPRO.md` row E), an order of magnitude over the
/// budget, so "one unit and the pass is spent" is prod's normal case, not its extreme — and the
/// app's own shape is one client holding thirteen standing subscriptions with one of them
/// dirtied by every inbound message. Turning this on today would hand that phone twelve
/// subscriptions that never settle.
///
/// Every fairness gate written before G6-9 rotates over clients, so all of them stay green
/// through this. That is the gap, not the scheduler's alone.
///
/// To ship it: fix the rotation so the cursor is over units rather than slots, un-ignore G6-9,
/// then `Some(DEFAULT_SETTLE_BUDGET_MICROS)` here and update G6-7 below.
const SETTLE_BUDGET_DEFAULT: Option<u64> = None;

/// The server's settle budget from `JAZZ_SETTLE_BUDGET_MS`, or the default.
///
/// A pure function so it can be gated at all: inline it is one branch inside a constructor that
/// needs a storage backend and a schema manager to reach, and the only observable would be a
/// timing difference. Extracted, every case that matters is one assertion in G6-7 below.
fn settle_budget_for_server(raw: Option<&str>) -> Option<u64> {
    match raw.map(str::trim) {
        // Unset or empty -> `SETTLE_BUDGET_DEFAULT`, which is `None` TODAY and must not be
        // flipped until the pool has a test at a production-shaped budget.
        None | Some("") => SETTLE_BUDGET_DEFAULT,
        // An explicit zero is the documented opt-out, and the reason to keep it is operational:
        // it is prod's one-variable rollback out of the pool scheduler.
        Some("0") => None,
        // Anything else goes through the parser. Garbage falls back to the DEFAULT, not to
        // unbounded — a typo in a compose file must not silently switch the protection off,
        // which is exactly how this item came to be inert in the first place.
        other => crate::query_manager::manager::QueryManager::settle_budget_micros_from_env(other)
            .or(SETTLE_BUDGET_DEFAULT),
    }
}

impl ServerBuilder {
    pub fn new(app_id: AppId) -> Self {
        Self {
            app_id,
            auth_config: AuthConfig {
                allow_local_first_auth: true,
                ..Default::default()
            },
            schema_mode: ServerSchemaMode::Dynamic,
            storage_backend: StorageBackend::Persistent {
                path: PathBuf::from("./data"),
            },
            sync_tracer: None,
            upstream_url: None,
            shutdown_timeout: DEFAULT_SHUTDOWN_TIMEOUT,
            client_ttl: DEFAULT_CLIENT_TTL,
            subscription_caps: None,
            staging_config: None,
            max_ws_decoded_frame_bytes: None,
        }
    }

    /// v18 item 3: staging caps and the in-flight decoded-bytes budget (tests; production
    /// reads the environment).
    pub fn with_staging_config(mut self, config: crate::runtime_tokio::StagingConfig) -> Self {
        self.staging_config = Some(config);
        self
    }

    /// v18 item 3: the decoded-frame cap (tests; production reads the environment).
    pub fn with_max_ws_decoded_frame_bytes(mut self, bytes: usize) -> Self {
        self.max_ws_decoded_frame_bytes = Some(bytes);
        self
    }

    /// Caps for downstream registrations. Absent: `SubscriptionCaps::from_env()`.
    pub fn with_subscription_caps(mut self, caps: crate::sync_manager::SubscriptionCaps) -> Self {
        self.subscription_caps = Some(caps);
        self
    }

    pub fn with_sync_tracer(mut self, tracer: crate::sync_tracer::SyncTracer) -> Self {
        self.sync_tracer = Some(tracer);
        self
    }

    pub fn with_auth_config(mut self, auth_config: AuthConfig) -> Self {
        self.auth_config = auth_config;
        self
    }

    pub fn with_local_first_auth(mut self, enabled: bool) -> Self {
        self.auth_config.allow_local_first_auth = enabled;
        self
    }

    pub fn with_upstream_url(mut self, upstream_url: impl Into<String>) -> Self {
        self.upstream_url = Some(upstream_url.into());
        self
    }

    pub fn with_shutdown_timeout(mut self, timeout: Duration) -> Self {
        self.shutdown_timeout = timeout;
        self
    }

    /// How long a disconnected client's server-side state (subscriptions,
    /// outbox, sync bookkeeping) is kept for a possible reconnect before the
    /// sweep reaps it. Clients that mint a fresh client id on every launch
    /// never resume, so deployments dominated by such clients want this low.
    pub fn with_client_ttl(mut self, ttl: Duration) -> Self {
        self.client_ttl = ttl;
        self
    }

    pub fn with_storage(mut self, backend: StorageBackend) -> Self {
        self.storage_backend = backend;
        self
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn with_schema(mut self, schema: Schema) -> Self {
        self.schema_mode = ServerSchemaMode::Fixed(schema);
        self
    }

    pub async fn build(self) -> Result<BuiltServer, String> {
        let auth_config = self.auth_config.clone();
        let topology = if self.upstream_url.is_some() {
            ServerTopology::Edge
        } else {
            ServerTopology::Core
        };
        let upstream_ws_url = match self.upstream_url.as_deref() {
            Some(upstream_url) => Some(upstream_ws_url(upstream_url, self.app_id)?),
            None => None,
        };
        let upstream_http_url = match self.upstream_url.as_deref() {
            Some(upstream_url) => Some(upstream_http_url(upstream_url, self.app_id)?),
            None => None,
        };
        validate_server_config(&auth_config, topology)?;
        let jwt_verifier = build_jwt_verifier(&auth_config).await?;
        log_auth_config(&auth_config, topology);

        let staging_config = self
            .staging_config
            .clone()
            .unwrap_or_else(crate::runtime_tokio::StagingConfig::from_env);
        let max_ws_decoded_frame_bytes = self
            .max_ws_decoded_frame_bytes
            .unwrap_or_else(max_ws_decoded_frame_bytes_from_env);
        // A legal frame must be able to acquire the whole budget, or it waits forever.
        let cap_permits = crate::runtime_tokio::kib_permits(max_ws_decoded_frame_bytes);
        if cap_permits > staging_config.budget_permits() || cap_permits > u32::MAX as usize {
            return Err(format!(
                "JAZZ_MAX_WS_DECODED_FRAME_BYTES ({max_ws_decoded_frame_bytes} bytes = {cap_permits} \
                 KiB) exceeds JAZZ_MAX_INFLIGHT_DECODED_BYTES ({} bytes = {} KiB): a frame at the \
                 cap could never be admitted",
                staging_config.inflight_budget_bytes,
                staging_config.budget_permits()
            ));
        }
        let (runtime, connection_event_hub) = self.build_runtime(staging_config)?;
        if let Some(upstream_ws_url) = upstream_ws_url.clone() {
            start_upstream_sync(&runtime, upstream_ws_url, &auth_config)?;
        }
        let http_client = reqwest::Client::builder()
            .build()
            .map_err(|e| format!("failed to build HTTP client: {e}"))?;

        let state = Arc::new(ServerState {
            runtime,
            app_id: self.app_id,
            connections: RwLock::new(HashMap::new()),
            next_connection_id: std::sync::atomic::AtomicU64::new(1),
            connection_event_hub,
            auth_config,
            upstream_http_url,
            topology,
            jwt_verifier,
            http_client,
            disconnect_candidates: RwLock::new(HashMap::new()),
            client_ttl: RwLock::new(self.client_ttl),
            sync_tracer: self.sync_tracer.clone(),
            shutdown: crate::server::ShutdownController::new(self.shutdown_timeout),
            max_ws_decoded_frame_bytes,
            budget_waiters: std::sync::atomic::AtomicUsize::new(0),
        });

        // Spawn periodic client state sweep (uses Weak so the task exits
        // when all strong refs to ServerState are dropped, e.g. in tests).
        {
            let weak_state = Arc::downgrade(&state);
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(std::time::Duration::from_secs(30));
                loop {
                    interval.tick().await;
                    let Some(state) = weak_state.upgrade() else {
                        break;
                    };
                    let reaped = state.run_sweep_once().await;
                    if !reaped.is_empty() {
                        tracing::info!(count = reaped.len(), "reaped stale disconnected clients");
                    }
                }
            });
        }

        let app = routes::create_router(state.clone());
        Ok(BuiltServer { state, app })
    }

    #[allow(clippy::type_complexity)]
    fn build_runtime(
        &self,
        staging: crate::runtime_tokio::StagingConfig,
    ) -> Result<(TokioRuntime<DynStorage>, Arc<ConnectionEventHub>), String> {
        let connection_event_hub = Arc::new(ConnectionEventHub::default());
        let dispatch_hub = Arc::clone(&connection_event_hub);

        let storage = self.build_main_storage()?;
        let mut schema_manager = self.build_schema_manager(storage.as_ref())?;
        // v18 item 6: the settle budget per tick.
        //
        // It has a DEFAULT, and that is the fix. Until diff round 26 this read the environment
        // and did nothing when it was unset — and `JAZZ_SETTLE_BUDGET_MS` is set in no compose
        // file, on the stand or in prod, so the bound had never once executed outside a test.
        // An absent settle budget is not "a tunable at its default", it is the fix switched
        // off; every other knob this campaign added falls back to a real value
        // (`SubscriptionCaps::from_env` -> `Default`). This one did not.
        //
        // Here and not in `QueryManager::new`, deliberately: that constructor serves every host,
        // including jazz-rn on the phone, and a bounded pass takes a different scheduler
        // (`dispatch_unit_pool`) that has never run there. The admission caps draw the same
        // boundary — read by the SERVER BINARY, not by the constructor. The consequence, which
        // belongs in the open: item 6 does NOT bound the engine hold in jazz-napi or jazz-rn,
        // so it cannot answer the phone-side stall in REPRO.md row I.
        //
        // `0` still means unbounded. With the default in place, that is prod's one-variable
        // rollback out of the whole pool scheduler.
        if let Some(micros) =
            settle_budget_for_server(std::env::var("JAZZ_SETTLE_BUDGET_MS").ok().as_deref())
        {
            schema_manager
                .query_manager_mut()
                .set_settle_budget_micros(Some(micros));
        }
        let runtime = TokioRuntime::new_with_staging(
            schema_manager,
            storage,
            move |entry| {
                if let Destination::Client(client_id) = entry.destination {
                    dispatch_hub.dispatch_payload(client_id, entry.payload);
                }
            },
            staging,
        );
        if let Some(ref tracer) = self.sync_tracer {
            runtime.set_sync_tracer(tracer.clone(), "server".to_string());
        }

        Ok((runtime, connection_event_hub))
    }

    fn build_schema_manager(&self, storage: &dyn Storage) -> Result<SchemaManager, String> {
        let sync_manager = server_sync_manager(
            self.local_durability_tier(),
            self.subscription_caps
                .unwrap_or_else(crate::sync_manager::SubscriptionCaps::from_env),
        );
        match &self.schema_mode {
            ServerSchemaMode::Dynamic => {
                let mut schema_manager =
                    SchemaManager::new_server(sync_manager, self.app_id, "prod");
                rehydrate_schema_manager_from_catalogue(&mut schema_manager, storage, self.app_id)
                    .map_err(|e| format!("failed to rehydrate schema manager: {e}"))?;
                // Dynamic servers fail closed until an explicit permissions head
                // is available for the active app.
                schema_manager
                    .query_manager_mut()
                    .require_authorization_schema();
                Ok(schema_manager)
            }
            ServerSchemaMode::Fixed(schema) => {
                // Fixed pins the CURRENT schema; it does not mean "ignore what
                // the store already records". A server built this way over a
                // store spanning a migration must still read its catalogue, or
                // its branch universe holds only the pinned generation and every
                // row written under an earlier one is unreadable at every
                // durability tier — the same hole the Dynamic arm above closes,
                // and the one jazz-napi shipped to production.
                let mut schema_manager =
                    SchemaManager::new(sync_manager, schema.clone(), self.app_id, "prod", "main")
                        .map_err(|e| format!("failed to initialize schema manager: {e:?}"))?;
                rehydrate_schema_manager_from_catalogue(&mut schema_manager, storage, self.app_id)
                    .map_err(|e| format!("failed to rehydrate schema manager: {e}"))?;
                Ok(schema_manager)
            }
        }
    }

    fn build_main_storage(&self) -> Result<DynStorage, String> {
        match &self.storage_backend {
            StorageBackend::Persistent { path } => {
                std::fs::create_dir_all(path)
                    .map_err(|e| format!("failed to create data dir '{}': {e}", path.display()))?;

                #[cfg(feature = "rocksdb")]
                {
                    let db_path = path.join("jazz.rocksdb");
                    let storage = RocksDBStorage::open(&db_path, STORAGE_CACHE_SIZE_BYTES)
                        .map_err(|e| {
                            format!("failed to open storage '{}': {e:?}", db_path.display())
                        })?;
                    Ok(Box::new(storage))
                }
                #[cfg(all(feature = "sqlite", not(feature = "rocksdb")))]
                {
                    let db_path = path.join("jazz.sqlite");
                    let storage = SqliteStorage::open(&db_path).map_err(|e| {
                        format!("failed to open storage '{}': {e:?}", db_path.display())
                    })?;
                    Ok(Box::new(storage))
                }
                #[cfg(not(any(feature = "rocksdb", feature = "sqlite")))]
                {
                    Ok(Box::new(MemoryStorage::new()))
                }
            }
            #[cfg(feature = "sqlite")]
            StorageBackend::Sqlite { path } => {
                std::fs::create_dir_all(path)
                    .map_err(|e| format!("failed to create data dir '{}': {e}", path.display()))?;
                let db_path = path.join("jazz.sqlite");
                let storage = SqliteStorage::open(&db_path).map_err(|e| {
                    format!("failed to open storage '{}': {e:?}", db_path.display())
                })?;
                Ok(Box::new(storage))
            }
            #[cfg(feature = "rocksdb")]
            StorageBackend::RocksDb { path } => {
                std::fs::create_dir_all(path)
                    .map_err(|e| format!("failed to create data dir '{}': {e}", path.display()))?;
                let db_path = path.join("jazz.rocksdb");
                let storage =
                    RocksDBStorage::open(&db_path, STORAGE_CACHE_SIZE_BYTES).map_err(|e| {
                        format!("failed to open storage '{}': {e:?}", db_path.display())
                    })?;
                Ok(Box::new(storage))
            }
            StorageBackend::InMemory => Ok(Box::new(MemoryStorage::new())),
        }
    }

    fn local_durability_tier(&self) -> DurabilityTier {
        if self.upstream_url.is_some() {
            DurabilityTier::EdgeServer
        } else {
            DurabilityTier::GlobalServer
        }
    }
}

fn server_sync_manager(
    local_tier: DurabilityTier,
    subscription_caps: crate::sync_manager::SubscriptionCaps,
) -> SyncManager {
    let sync_manager = SyncManager::new()
        .with_durability_tier(local_tier)
        .with_subscription_caps(subscription_caps);
    if should_allow_unprivileged_schema_catalogue_writes() {
        sync_manager.with_unprivileged_schema_catalogue_writes()
    } else {
        sync_manager
    }
}

fn should_allow_unprivileged_schema_catalogue_writes() -> bool {
    !matches!(
        std::env::var("NODE_ENV"),
        Ok(value) if value.eq_ignore_ascii_case("production")
    )
}

async fn build_jwt_verifier(auth_config: &AuthConfig) -> Result<Option<Arc<JwtVerifier>>, String> {
    match (
        auth_config.jwks_url.as_ref(),
        auth_config.jwt_public_key.as_ref(),
    ) {
        (Some(_), Some(_)) => Err(
            "configure either --jwks-url / JAZZ_JWKS_URL or --jwt-public-key / JAZZ_JWT_PUBLIC_KEY, not both"
                .to_string(),
        ),
        (None, None) => Ok(None),
        (None, Some(public_key)) => {
            let verifier = StaticJwtVerifier::from_public_key(public_key)?;
            Ok(Some(Arc::new(JwtVerifier::Static(verifier))))
        }
        (Some(jwks_url), None) => {
            let jwks_ttl = std::env::var("JAZZ_JWKS_CACHE_TTL_SECS")
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
                .map(Duration::from_secs)
                .unwrap_or(JWKS_CACHE_TTL);
            let jwks_max_stale = std::env::var("JAZZ_JWKS_MAX_STALE_SECS")
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
                .map(Duration::from_secs)
                .unwrap_or(JWKS_MAX_STALE);

            let http_client = reqwest::Client::builder()
                .connect_timeout(Duration::from_secs(5))
                .timeout(Duration::from_secs(10))
                .build()
                .map_err(|e| format!("failed to build JWKS HTTP client: {e}"))?;

            let verifier = Arc::new(JwtVerifier::Jwks(JwksCache::new(
                jwks_url.clone(),
                http_client,
                jwks_ttl,
                jwks_max_stale,
            )));

            // Warm the cache in the background. The JWKS endpoint may not be
            // available yet (e.g. Jazz server starts during Next.js config resolution,
            // before the app is listening). First auth request will block on fetch
            // if the background warm hasn't completed.
            {
                let verifier = Arc::clone(&verifier);
                tokio::spawn(async move {
                    if let JwtVerifier::Jwks(cache) = verifier.as_ref()
                        && let Err(e) = cache.load(false).await
                    {
                        tracing::warn!(
                            "Background JWKS warm failed (will retry on first auth request): {e}"
                        );
                    }
                });
            }

            Ok(Some(verifier))
        }
    }
}

fn validate_server_config(
    auth_config: &AuthConfig,
    topology: ServerTopology,
) -> Result<(), String> {
    if topology.is_edge() && auth_config.admin_secret.is_none() {
        return Err("edge mode requires --admin-secret / JAZZ_ADMIN_SECRET when --upstream-url / JAZZ_UPSTREAM_URL is set".to_string());
    }

    Ok(())
}

fn log_auth_config(auth_config: &AuthConfig, topology: ServerTopology) {
    info!(
        "Auth configured: local_first={}, jwks={}, static_jwt_key={}, cookie={}, backend={}, admin={}, topology={:?}",
        auth_config.allow_local_first_auth,
        auth_config.jwks_url.is_some(),
        auth_config.jwt_public_key.is_some(),
        auth_config.auth_cookie_name.is_some(),
        auth_config.backend_secret.is_some(),
        auth_config.admin_secret.is_some(),
        topology
    );
}

pub fn upstream_ws_url(base_url: &str, app_id: AppId) -> Result<String, String> {
    let mut url = reqwest::Url::parse(base_url)
        .map_err(|err| format!("invalid upstream URL '{base_url}': {err}"))?;

    if url.query().is_some() || url.fragment().is_some() {
        return Err("upstream URL must not include query parameters or a fragment".to_string());
    }

    let scheme = match url.scheme() {
        "http" => "ws",
        "https" => "wss",
        "ws" => "ws",
        "wss" => "wss",
        other => {
            return Err(format!(
                "unsupported upstream URL scheme '{other}'; expected http, https, ws, or wss"
            ));
        }
    };
    url.set_scheme(scheme)
        .map_err(|_| format!("failed to set upstream URL scheme to {scheme}"))?;

    let app_ws_path = format!("/apps/{app_id}/ws");
    let normalized_path = url.path().trim_end_matches('/');
    if normalized_path == app_ws_path.trim_end_matches('/') {
        url.set_path(&app_ws_path);
    } else {
        let base_path = match normalized_path {
            "" | "/" => String::new(),
            path => path.to_string(),
        };
        url.set_path(&format!(
            "{}/{}",
            base_path.trim_end_matches('/'),
            app_ws_path.trim_start_matches('/')
        ));
    }

    Ok(url.to_string())
}

pub fn upstream_http_url(base_url: &str, app_id: AppId) -> Result<String, String> {
    let mut url = reqwest::Url::parse(base_url)
        .map_err(|err| format!("invalid upstream URL '{base_url}': {err}"))?;

    if url.query().is_some() || url.fragment().is_some() {
        return Err("upstream URL must not include query parameters or a fragment".to_string());
    }

    let scheme = match url.scheme() {
        "http" => "http",
        "https" => "https",
        "ws" => "http",
        "wss" => "https",
        other => {
            return Err(format!(
                "unsupported upstream URL scheme '{other}'; expected http, https, ws, or wss"
            ));
        }
    };
    url.set_scheme(scheme)
        .map_err(|_| format!("failed to set upstream URL scheme to {scheme}"))?;

    let app_ws_path = format!("/apps/{app_id}/ws");
    let normalized_path = url.path().trim_end_matches('/').to_string();
    if normalized_path == app_ws_path.trim_end_matches('/') {
        url.set_path("/");
    } else if let Some(prefix) = normalized_path.strip_suffix(&app_ws_path) {
        let prefix_path = if prefix.is_empty() {
            "/".to_string()
        } else {
            format!("{}/", prefix.trim_end_matches('/'))
        };
        url.set_path(&prefix_path);
    } else if normalized_path.is_empty() {
        url.set_path("/");
    }

    Ok(url.to_string())
}

fn start_upstream_sync(
    runtime: &TokioRuntime<DynStorage>,
    upstream_ws_url: String,
    auth_config: &AuthConfig,
) -> Result<(), String> {
    let admin_secret = auth_config
        .admin_secret
        .clone()
        .ok_or_else(|| "edge mode requires --admin-secret / JAZZ_ADMIN_SECRET".to_string())?;

    info!(
        local_tier = "edge",
        upstream_url = %upstream_ws_url,
        upstream_connected = false,
        "starting edge upstream sync"
    );

    let retry_config = TransportRetryConfig {
        connect_attempt_timeout: Some(EDGE_UPSTREAM_CONNECT_ATTEMPT_TIMEOUT),
        auth_handshake_timeout: Some(EDGE_UPSTREAM_AUTH_HANDSHAKE_TIMEOUT),
    };

    runtime.connect_with_retry_config(
        upstream_ws_url.clone(),
        crate::transport_manager::AuthConfig {
            admin_secret: Some(admin_secret),
            ..Default::default()
        },
        retry_config,
    );

    let wait_runtime = (*runtime).clone();
    tokio::spawn(async move {
        let connected = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            wait_runtime.transport_wait_until_connected(),
        )
        .await
        .unwrap_or(false);
        if connected {
            tracing::info!(
                local_tier = "edge",
                upstream_url = %upstream_ws_url,
                upstream_connected = true,
                "edge upstream sync connected"
            );
        } else {
            tracing::warn!(
                local_tier = "edge",
                upstream_url = %upstream_ws_url,
                upstream_connected = false,
                "edge upstream sync ended before first connection"
            );
        }
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema_manager::AppId;

    #[tokio::test]
    async fn builder_applies_configured_client_ttl() {
        let app_id =
            AppId::from_string("00000000-0000-0000-0000-000000000001").expect("parse app id");
        let built = ServerBuilder::new(app_id)
            .with_storage(StorageBackend::InMemory)
            .with_client_ttl(Duration::from_secs(7))
            .build()
            .await
            .expect("build server");
        assert_eq!(
            *built.state.client_ttl.read().await,
            Duration::from_secs(7),
            "configured client TTL must reach ServerState"
        );
    }

    #[test]
    fn upstream_url_conversion_maps_base_urls_to_app_ws_route() {
        let app_id =
            AppId::from_string("00000000-0000-0000-0000-000000000001").expect("parse app id");

        assert_eq!(
            upstream_ws_url("https://core.example.com", app_id).expect("https conversion"),
            "wss://core.example.com/apps/00000000-0000-0000-0000-000000000001/ws"
        );
        assert_eq!(
            upstream_ws_url("http://core.example.com/base/", app_id).expect("http conversion"),
            "ws://core.example.com/base/apps/00000000-0000-0000-0000-000000000001/ws"
        );
        assert_eq!(
            upstream_ws_url("ws://core.example.com", app_id).expect("ws conversion"),
            "ws://core.example.com/apps/00000000-0000-0000-0000-000000000001/ws"
        );
        assert_eq!(
            upstream_ws_url(
                "wss://core.example.com/apps/00000000-0000-0000-0000-000000000001/ws",
                app_id
            )
            .expect("already app-scoped ws URL"),
            "wss://core.example.com/apps/00000000-0000-0000-0000-000000000001/ws"
        );
    }

    #[test]
    fn upstream_url_conversion_rejects_query_and_fragment_urls() {
        let app_id =
            AppId::from_string("00000000-0000-0000-0000-000000000001").expect("parse app id");

        assert!(upstream_ws_url("https://core.example.com?token=abc", app_id).is_err());
        assert!(upstream_ws_url("https://core.example.com#cluster-a", app_id).is_err());
    }

    #[test]
    fn upstream_http_url_conversion_maps_base_urls_to_app_routes() {
        let app_id =
            AppId::from_string("00000000-0000-0000-0000-000000000001").expect("parse app id");

        assert_eq!(
            upstream_http_url("https://core.example.com", app_id).expect("https conversion"),
            "https://core.example.com/"
        );
        assert_eq!(
            upstream_http_url("http://core.example.com/base/", app_id).expect("http conversion"),
            "http://core.example.com/base/"
        );
        assert_eq!(
            upstream_http_url("ws://core.example.com", app_id).expect("ws conversion"),
            "http://core.example.com/"
        );
        assert_eq!(
            upstream_http_url(
                "wss://core.example.com/apps/00000000-0000-0000-0000-000000000001/ws",
                app_id,
            )
            .expect("wss conversion"),
            "https://core.example.com/"
        );
        assert_eq!(
            upstream_http_url(
                "wss://core.example.com/base/apps/00000000-0000-0000-0000-000000000001/ws",
                app_id,
            )
            .expect("prefixed wss conversion"),
            "https://core.example.com/base/"
        );
    }

    #[test]
    fn upstream_http_url_conversion_rejects_query_and_fragment_urls() {
        let app_id =
            AppId::from_string("00000000-0000-0000-0000-000000000001").expect("parse app id");

        assert!(upstream_http_url("https://core.example.com?token=abc", app_id).is_err());
        assert!(upstream_http_url("https://core.example.com#cluster-a", app_id).is_err());
    }

    #[tokio::test]
    async fn builder_requires_admin_secret_in_edge_mode() {
        let auth_config = AuthConfig {
            allow_local_first_auth: true,
            ..Default::default()
        };

        let result = ServerBuilder::new(AppId::from_name("test-app"))
            .with_auth_config(auth_config)
            .with_storage(StorageBackend::InMemory)
            .with_upstream_url("ws://127.0.0.1:9")
            .build()
            .await;
        let error = result
            .err()
            .expect("edge mode without admin secret should fail");

        assert!(error.contains("--admin-secret"));
        assert!(error.contains("--upstream-url"));
    }

    #[tokio::test]
    async fn builder_allows_edge_mode_with_admin_secret_only() {
        let built = ServerBuilder::new(AppId::from_name("edge-builder-admin-secret-only"))
            .with_storage(StorageBackend::InMemory)
            .with_auth_config(AuthConfig {
                admin_secret: Some("admin-secret".to_string()),
                ..Default::default()
            })
            .with_upstream_url("ws://127.0.0.1:9")
            .build()
            .await
            .expect("build edge server with admin secret only");

        let tiers = built
            .state
            .runtime
            .with_sync_manager(|sync| sync.local_durability_tiers())
            .expect("read sync manager");

        assert_eq!(
            tiers,
            std::collections::HashSet::from([DurabilityTier::EdgeServer])
        );
    }

    #[tokio::test]
    async fn builder_uses_global_tier_without_upstream() {
        let built = ServerBuilder::new(AppId::from_name("global-builder-tier"))
            .with_storage(StorageBackend::InMemory)
            .build()
            .await
            .expect("build global server");

        let tiers = built
            .state
            .runtime
            .with_sync_manager(|sync| sync.local_durability_tiers())
            .expect("read sync manager");

        assert_eq!(
            tiers,
            std::collections::HashSet::from([DurabilityTier::GlobalServer])
        );
    }

    #[tokio::test]
    async fn builder_uses_edge_tier_with_upstream() {
        let built = ServerBuilder::new(AppId::from_name("edge-builder-tier"))
            .with_storage(StorageBackend::InMemory)
            .with_auth_config(AuthConfig {
                admin_secret: Some("admin-secret".to_string()),
                ..Default::default()
            })
            .with_upstream_url("ws://127.0.0.1:9")
            .build()
            .await
            .expect("build edge server");

        let tiers = built
            .state
            .runtime
            .with_sync_manager(|sync| sync.local_durability_tiers())
            .expect("read sync manager");

        assert_eq!(
            tiers,
            std::collections::HashSet::from([DurabilityTier::EdgeServer])
        );
    }
}

#[cfg(test)]
mod settle_budget_default_gates {
    use super::settle_budget_for_server;

    /// G6-7 (v18 item 6, diff round 26). The settle budget must be ON without configuration.
    ///
    /// This is the gate whose absence let item 6 sit inert. Every other gate for this item sets
    /// the budget itself — `set_settle_budget_micros(Some(0))` — so not one of them could see
    /// that no deployment does. `JAZZ_SETTLE_BUDGET_MS` is set in no compose file, on the stand
    /// or in prod, and unset meant the unbounded pass the item exists to remove.
    ///
    /// Internal on purpose: which scheduler a pass takes is engine bookkeeping. From outside,
    /// a bounded and an unbounded pass differ only in how long a read waits behind them.
    #[test]
    fn the_server_bounds_its_settle_pass_without_being_configured() {
        // The LITERAL, not `SETTLE_BUDGET_DEFAULT`. Comparing the function's output against the
        // constant it returns is `X == X` (diff r28): it passes whatever that constant is set
        // to, so it could not notice the flip it exists to guard. Spelling the value here is
        // what makes flipping the default a red test rather than a silent behaviour change.
        assert_eq!(
            settle_budget_for_server(None),
            None,
            "an unconfigured server gets `SETTLE_BUDGET_DEFAULT`, and it is `None`: item 6 is \
             OFF, because the pool starves a client's other subscriptions when a pass runs one \
             unit (G6-9, ignored as an open defect). It is NOT off because delivery breaks — \
             diff r27 measured that it does not. This assertion exists because the item once \
             shipped INERT, with six green gates describing a knob nothing turned: switching it \
             on has to be an edit to a line that says what is still broken"
        );
        assert_eq!(
            settle_budget_for_server(Some("")),
            None,
            "and an empty value is unset, not an opt-out"
        );
        assert_eq!(
            settle_budget_for_server(Some("0")),
            None,
            "an EXPLICIT zero is the opt-out, and it stays: it is prod's one-variable rollback \
             out of the pool scheduler"
        );
        assert_eq!(
            settle_budget_for_server(Some("nonsense")),
            None,
            "and garbage falls back to the default, never to a value nobody chose — a typo in \
             a compose file must not decide the scheduler"
        );
        assert_eq!(
            settle_budget_for_server(Some("120")),
            Some(120_000),
            "an explicit value is still honoured, in milliseconds"
        );
    }
}
