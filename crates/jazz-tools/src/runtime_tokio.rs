//! Tokio runtime adapter for Jazz.
//!
//! Provides `TokioRuntime<S>` - a thin wrapper around
//! `RuntimeCore<S, TokioScheduler<S>>`
//! that handles async scheduling via `tokio::spawn`.
//!
//! # Architecture
//!
//! - `S: Storage + Send + 'static` provides synchronous storage
//! - `TokioScheduler<S>` implements `Scheduler` using tokio::spawn for batched ticks
//! - `CallbackSyncSender` implements `SyncSender` with a user-provided callback
//! - `TokioRuntime<S>` wraps `Arc<Mutex<RuntimeCore<...>>>`
//! - Methods grab the lock, call RuntimeCore, and return

use std::any::Any;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};

use futures::channel::oneshot;
mod staging;
#[cfg(test)]
mod staging_tests;
use crate::batch_fate::BatchMode;
use crate::object::ObjectId;
use crate::query_manager::query::Query;
use crate::query_manager::session::{Session, WriteContext};
use crate::query_manager::types::{Schema, SchemaHash, Value};
use crate::row_histories::BatchId;
pub use crate::runtime_core::SubscriptionHandle;
use crate::runtime_core::{
    QueryFuture, ReadDurabilityOptions, RuntimeCore, RuntimeError as CoreRuntimeError, Scheduler,
    SubscriptionDelta, SyncSender,
};
use crate::schema_manager::manager::{CurrentPermissionsSummary, PermissionsHeadSummary};
use crate::schema_manager::{Lens, SchemaManager};
use crate::storage::Storage;
use crate::sync_manager::{
    ClientId, DurabilityTier, InboxEntry, OutboxEntry, QueryPropagation, ServerId,
};
pub use staging::{
    ClientStagingStats, InboundStaging, StagePush, StagingConfig, WaiterGuard, kib_permits,
};

// ============================================================================
// TokioScheduler
// ============================================================================

/// Type alias for the concrete RuntimeCore used by TokioRuntime.
type TokioCoreType<S> = RuntimeCore<S, TokioScheduler<S>>;
type DirectInsertResult = (ObjectId, Vec<Value>, BatchId);

/// Scheduler implementation for Tokio.
///
/// Spawns a tokio task to call `batched_tick()` on the RuntimeCore.
/// Debounced: only one task is scheduled at a time.
pub struct TokioScheduler<S: Storage + Send + 'static> {
    /// Debounce flag for scheduled ticks.
    scheduled: Arc<AtomicBool>,
    /// Weak reference back to RuntimeCore for the tick thread.
    core_ref: Weak<Mutex<TokioCoreType<S>>>,
    /// v18 item 3: frames staged by socket tasks, drained by the tick thread under the core
    /// lock right before `batched_tick`.
    staging: Arc<InboundStaging>,
    /// Wakes the tick thread. The thread exits when every sender is gone.
    wake: std::sync::mpsc::Sender<()>,
    /// Held until `set_core_ref` spawns the thread.
    wake_rx: Arc<Mutex<Option<std::sync::mpsc::Receiver<()>>>>,
    /// Logged once: a wake that found no thread (it exited on a poisoned core mutex).
    wake_failed: Arc<AtomicBool>,
}

impl<S: Storage + Send + 'static> TokioScheduler<S> {
    /// Create a new TokioScheduler.
    ///
    /// Note: `core_ref` starts as empty and is set after RuntimeCore is created.
    fn new(staging: StagingConfig) -> Self {
        let (wake, wake_rx) = std::sync::mpsc::channel();
        Self {
            scheduled: Arc::new(AtomicBool::new(false)),
            core_ref: Weak::new(),
            staging: Arc::new(InboundStaging::new(staging)),
            wake,
            wake_rx: Arc::new(Mutex::new(Some(wake_rx))),
            wake_failed: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Set the core reference (called after RuntimeCore is wrapped in Arc<Mutex>) and start
    /// the tick thread.
    ///
    /// The tick runs on its own std thread (`jazz-tick`), not on tokio's blocking pool: it
    /// competes on the core mutex only, never on the pool's FIFO where handshake closures
    /// queue (512 handshakes parked on the lock during a reconnect storm would otherwise put
    /// the tick strictly last, keep the debounce flag high and stage every push to the cap).
    /// The thread has no tokio context; nothing reachable from `batched_tick` needs one.
    fn set_core_ref(&mut self, core_ref: Weak<Mutex<TokioCoreType<S>>>) {
        self.core_ref = core_ref.clone();
        let Some(wake_rx) = self
            .wake_rx
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
        else {
            return;
        };
        let scheduled = Arc::clone(&self.scheduled);
        let staging = Arc::clone(&self.staging);
        std::thread::Builder::new()
            .name("jazz-tick".to_string())
            .spawn(move || tick_thread(wake_rx, core_ref, scheduled, staging))
            .expect("spawn the jazz tick thread");
    }

    /// The staging shared with the runtime's push paths.
    fn staging(&self) -> &Arc<InboundStaging> {
        &self.staging
    }

    /// Check if a batched_tick is currently scheduled.
    pub fn is_scheduled(&self) -> bool {
        self.scheduled.load(Ordering::SeqCst)
    }
}

impl<S: Storage + Send + 'static> Scheduler for TokioScheduler<S> {
    fn schedule_batched_tick(&self) {
        // Debounce: only wake the thread if a tick is not already queued. The flag stays
        // high until the thread owns the core, so the queue is capped at one pending tick.
        if !self.scheduled.swap(true, Ordering::SeqCst) && self.wake.send(()).is_err() {
            // The thread is gone (it exits on a poisoned core mutex). The flag stays high
            // so `flush()` fails loudly after its bounded wait instead of ticking a runtime
            // whose mutex is poisoned.
            if !self.wake_failed.swap(true, Ordering::SeqCst) {
                tracing::error!(
                    "TokioScheduler: the tick thread is gone; no further batched tick will run"
                );
            }
        }
    }
}

/// The tick thread's body. Runs until the wake channel closes (every scheduler clone
/// dropped) or the core is gone.
fn tick_thread<S: Storage + Send + 'static>(
    wake_rx: std::sync::mpsc::Receiver<()>,
    core_ref: Weak<Mutex<TokioCoreType<S>>>,
    scheduled: Arc<AtomicBool>,
    staging: Arc<InboundStaging>,
) {
    while wake_rx.recv().is_ok() {
        // Give bursty transports (notably WebSocket frames emitted back-to-back) one
        // scheduler turn to enqueue related messages before the runtime drains. Without
        // this, a large subscription burst can be observed as many one-message ticks,
        // causing per-query result flushing and delayed tier-settled first deliveries.
        std::thread::sleep(std::time::Duration::from_millis(1));
        // Acquire the core lock FIRST, then clear the debounce flag, then drain the
        // staging, then tick. Clearing the flag before running the tick preserves the
        // lost-wakeup fix: a frame arriving while batched_tick executes finds
        // scheduled=false and wakes the thread again. Clearing it only after acquiring the
        // lock caps the queue at one pending tick.
        let Some(core_arc) = core_ref.upgrade() else {
            tracing::debug!("TokioScheduler: core dropped before the tick could run; exiting");
            return;
        };
        let Ok(mut core) = core_arc.lock() else {
            // Poisoned: ticking is over for this process. The flag is left high on purpose
            // (see `schedule_batched_tick`).
            tracing::error!(
                "TokioScheduler: core mutex poisoned; the tick thread exits and the runtime is unusable"
            );
            return;
        };
        scheduled.store(false, Ordering::SeqCst);
        drain_staging_into(&staging, &mut core);
        core.batched_tick();
    }
}

/// Park every staged frame under the core lock the caller holds, then return the frames'
/// permits to the budget. Shared by the tick thread and `flush()`.
fn drain_staging_into<S: Storage + Send + 'static>(
    staging: &InboundStaging,
    core: &mut TokioCoreType<S>,
) {
    let drained = staging.drain();
    if drained.is_empty() {
        return;
    }
    let released_kib = drained.released_kib();
    for entry in drained.into_entries() {
        core.park_staged_sync_message(entry);
    }
    staging.release_permits(released_kib);
}

// Manual Clone: `S` is not stored by value — the Arc and Weak clones
// are cheap pointer copies that share the underlying allocation.
impl<S: Storage + Send + 'static> Clone for TokioScheduler<S> {
    fn clone(&self) -> Self {
        Self {
            scheduled: Arc::clone(&self.scheduled),
            core_ref: Weak::clone(&self.core_ref),
            staging: Arc::clone(&self.staging),
            wake: self.wake.clone(),
            wake_rx: Arc::clone(&self.wake_rx),
            wake_failed: Arc::clone(&self.wake_failed),
        }
    }
}

// ============================================================================
// CallbackSyncSender
// ============================================================================

/// SyncSender implementation using a callback.
#[derive(Clone)]
pub struct CallbackSyncSender {
    callback: Arc<dyn Fn(OutboxEntry) + Send + Sync>,
}

impl CallbackSyncSender {
    fn new<F>(callback: F) -> Self
    where
        F: Fn(OutboxEntry) + Send + Sync + 'static,
    {
        Self {
            callback: Arc::new(callback),
        }
    }
}

impl SyncSender for CallbackSyncSender {
    fn send_sync_message(&self, message: OutboxEntry) {
        (self.callback)(message);
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ============================================================================
// Errors
// ============================================================================

/// Errors from runtime operations.
#[derive(Debug, Clone)]
pub enum RuntimeError {
    QueryError(String),
    WriteError(String),
    NotFound,
    LockError,
}

impl std::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RuntimeError::QueryError(s) => write!(f, "Query error: {}", s),
            RuntimeError::WriteError(s) => write!(f, "Write error: {}", s),
            RuntimeError::NotFound => write!(f, "Not found"),
            RuntimeError::LockError => write!(f, "Lock error"),
        }
    }
}

impl std::error::Error for RuntimeError {}

impl From<CoreRuntimeError> for RuntimeError {
    fn from(e: CoreRuntimeError) -> Self {
        match e {
            CoreRuntimeError::QueryError(s) => RuntimeError::QueryError(s),
            CoreRuntimeError::WriteError(s) => RuntimeError::WriteError(s),
            CoreRuntimeError::NotFound => RuntimeError::NotFound,
            CoreRuntimeError::QueryCancelled => {
                RuntimeError::QueryError("query cancelled by its deadline".to_string())
            }
            CoreRuntimeError::AnonymousWriteDenied { table, operation } => {
                RuntimeError::WriteError(format!(
                    "anonymous session cannot {} on table {}",
                    operation, table
                ))
            }
        }
    }
}

// ============================================================================
// TokioRuntime
// ============================================================================

/// Tokio runtime for Jazz, generic over storage backend.
///
/// Thin wrapper around `Arc<Mutex<RuntimeCore<S, TokioScheduler<S>>>>`.
/// All methods grab the lock, call RuntimeCore, and return.
/// Async scheduling happens via TokioScheduler.schedule_batched_tick().
pub struct TokioRuntime<S: Storage + Send + 'static> {
    core: Arc<Mutex<TokioCoreType<S>>>,
    /// Installed as `RuntimeCore::sync_sender` and retained here so the
    /// backing callback outlives the core Arc's lifetime.
    _sync_sender: CallbackSyncSender,
    /// Cloned handle to the scheduler (shares Arc-based state with the one inside core).
    /// Stored here so `connect()` can build a `NativeTickNotifier` without locking.
    scheduler: TokioScheduler<S>,
}
impl<S: Storage + Send + 'static> TokioRuntime<S> {
    /// Test hook: look at the core under its lock. For black-box gates that need a
    /// server-side count no wire message reports (e.g. admitted registrations).
    #[cfg(any(test, feature = "test"))]
    pub fn inspect_core_for_test<R>(&self, inspect: impl FnOnce(&TokioCoreType<S>) -> R) -> R {
        let core = self.core.lock().expect("core lock");
        inspect(&core)
    }
} // Manual Clone impl — only needs Arc::clone, not S: Clone
impl<S: Storage + Send + 'static> Clone for TokioRuntime<S> {
    fn clone(&self) -> Self {
        Self {
            core: Arc::clone(&self.core),
            _sync_sender: self._sync_sender.clone(),
            scheduler: self.scheduler.clone(),
        }
    }
}

impl<S: Storage + Send + 'static> TokioRuntime<S> {
    /// Create a new TokioRuntime with the given storage backend.
    ///
    /// # Arguments
    /// - `schema_manager` - The SchemaManager to wrap
    /// - `storage` - The storage backend (e.g., MemoryStorage, FjallStorage)
    /// - `sync_callback` - Called when sync messages need to be sent
    pub fn new<F>(schema_manager: SchemaManager, storage: S, sync_callback: F) -> Self
    where
        F: Fn(OutboxEntry) + Send + Sync + 'static,
    {
        Self::new_with_staging(
            schema_manager,
            storage,
            sync_callback,
            StagingConfig::from_env(),
        )
    }

    /// `new` with explicit staging and budget knobs (the server builder's override).
    pub fn new_with_staging<F>(
        schema_manager: SchemaManager,
        storage: S,
        sync_callback: F,
        staging: StagingConfig,
    ) -> Self
    where
        F: Fn(OutboxEntry) + Send + Sync + 'static,
    {
        let scheduler = TokioScheduler::new(staging);
        let sync_sender = CallbackSyncSender::new(sync_callback);

        // Create RuntimeCore
        let mut core = RuntimeCore::new(schema_manager, storage, scheduler);
        // Install the callback as the runtime's fallback outbox sink so
        // server-side fanout (or any code path without a TransportHandle)
        // still delivers OutboxEntries.
        core.set_sync_sender(Box::new(sync_sender.clone()));

        // Wrap in Arc<Mutex>
        let core_arc = Arc::new(Mutex::new(core));

        // Set the core_ref on the Scheduler
        {
            let mut core_guard = core_arc.lock().unwrap();
            core_guard
                .scheduler_mut()
                .set_core_ref(Arc::downgrade(&core_arc));
        }

        // Clone the scheduler AFTER set_core_ref so the clone shares the
        // Arc<AtomicBool> debounce flag and the Weak core reference.
        let scheduler_clone = {
            let core_guard = core_arc.lock().unwrap();
            (*core_guard.scheduler()).clone()
        };

        Self {
            core: core_arc,
            _sync_sender: sync_sender,
            scheduler: scheduler_clone,
        }
    }

    /// Persist the current schema to the catalogue for server sync.
    pub fn persist_schema(&self) -> Result<ObjectId, RuntimeError> {
        let mut core = self.core.lock().map_err(|_| RuntimeError::LockError)?;
        Ok(core.persist_schema())
    }

    /// Publish any schema object to the local catalogue and in-memory schema manager.
    pub fn publish_schema(&self, schema: Schema) -> Result<ObjectId, RuntimeError> {
        let mut core = self.core.lock().map_err(|_| RuntimeError::LockError)?;
        Ok(core.publish_schema(schema))
    }

    pub fn publish_permissions_bundle(
        &self,
        schema_hash: crate::query_manager::types::SchemaHash,
        permissions: std::collections::HashMap<
            crate::query_manager::types::TableName,
            crate::query_manager::types::TablePolicies,
        >,
        expected_parent_bundle_object_id: Option<ObjectId>,
    ) -> Result<Option<ObjectId>, RuntimeError> {
        let mut core = self.core.lock().map_err(|_| RuntimeError::LockError)?;
        core.publish_permissions_bundle(schema_hash, permissions, expected_parent_bundle_object_id)
            .map_err(|error| RuntimeError::WriteError(error.to_string()))
    }

    pub fn current_permissions_head(&self) -> Result<Option<PermissionsHeadSummary>, RuntimeError> {
        let core = self.core.lock().map_err(|_| RuntimeError::LockError)?;
        Ok(core.schema_manager().current_permissions_head())
    }

    pub fn current_permissions(&self) -> Result<Option<CurrentPermissionsSummary>, RuntimeError> {
        let core = self.core.lock().map_err(|_| RuntimeError::LockError)?;
        Ok(core.schema_manager().current_permissions())
    }

    /// Publish a reviewed lens edge to the local catalogue and active schema manager.
    pub fn publish_lens(&self, lens: &Lens) -> Result<ObjectId, RuntimeError> {
        let mut core = self.core.lock().map_err(|_| RuntimeError::LockError)?;
        Ok(core.publish_lens(lens)?)
    }

    // =========================================================================
    // CRUD Operations
    // =========================================================================

    /// Insert a row into a table.
    pub fn insert(
        &self,
        table: &str,
        values: HashMap<String, Value>,
        write_context: Option<&WriteContext>,
    ) -> Result<DirectInsertResult, RuntimeError> {
        self.insert_with_id(table, values, None, write_context)
    }

    /// Insert a row into a table with an optional external row id.
    pub fn insert_with_id(
        &self,
        table: &str,
        values: HashMap<String, Value>,
        object_id: Option<ObjectId>,
        write_context: Option<&WriteContext>,
    ) -> Result<DirectInsertResult, RuntimeError> {
        let mut core = self.core.lock().map_err(|_| RuntimeError::LockError)?;
        let ((row_id, row_values), batch_id) =
            core.insert_with_id(table, values, object_id, write_context)?;
        Ok((row_id, row_values, batch_id))
    }

    /// Update a row (partial update by column name).
    pub fn update(
        &self,
        object_id: ObjectId,
        values: Vec<(String, Value)>,
        write_context: Option<&WriteContext>,
    ) -> Result<BatchId, RuntimeError> {
        let mut core = self.core.lock().map_err(|_| RuntimeError::LockError)?;
        Ok(core.update(object_id, values, write_context)?)
    }

    /// Create or update a row with a caller-supplied external row id.
    pub fn upsert(
        &self,
        table: &str,
        object_id: ObjectId,
        values: HashMap<String, Value>,
        write_context: Option<&WriteContext>,
    ) -> Result<BatchId, RuntimeError> {
        let mut core = self.core.lock().map_err(|_| RuntimeError::LockError)?;
        Ok(core.upsert(table, object_id, values, write_context)?)
    }

    /// Delete a row.
    pub fn delete(
        &self,
        object_id: ObjectId,
        write_context: Option<&WriteContext>,
    ) -> Result<BatchId, RuntimeError> {
        let mut core = self.core.lock().map_err(|_| RuntimeError::LockError)?;
        Ok(core.delete(object_id, write_context)?)
    }

    /// Begin a batch or transaction and return its logical batch id.
    pub fn begin_batch(&self, batch_mode: BatchMode) -> Result<BatchId, RuntimeError> {
        let mut core = self.core.lock().map_err(|_| RuntimeError::LockError)?;
        Ok(core.begin_batch(batch_mode))
    }

    /// Commit an open batch or transaction.
    pub fn commit_batch(&self, batch_id: BatchId) -> Result<(), RuntimeError> {
        let mut core = self.core.lock().map_err(|_| RuntimeError::LockError)?;
        Ok(core.commit_batch(batch_id)?)
    }

    /// Roll back an open batch or transaction.
    pub fn rollback_batch(&self, batch_id: BatchId) -> Result<bool, RuntimeError> {
        let mut core = self.core.lock().map_err(|_| RuntimeError::LockError)?;
        Ok(core.rollback_batch(batch_id)?)
    }

    /// Wait for a batch to settle at the requested durability tier.
    pub fn wait_for_batch(
        &self,
        batch_id: BatchId,
        tier: DurabilityTier,
    ) -> Result<oneshot::Receiver<crate::runtime_core::PersistedWriteAck>, RuntimeError> {
        let mut core = self.core.lock().map_err(|_| RuntimeError::LockError)?;
        Ok(core.wait_for_batch(batch_id, tier)?)
    }

    /// Flush pending operations to storage.
    ///
    /// Call this after CRUD operations if you need to ensure data is persisted
    /// before continuing. Waits for any scheduled batched_tick to complete
    /// and then runs additional ticks until all storage is flushed.
    pub async fn flush(&self) -> Result<(), RuntimeError> {
        let mut attempts = 0;
        loop {
            // v18 item 4: a store that lost writes never flushes again until it is reopened;
            // say so at once, on every call, instead of ticking 200 times toward a barrier
            // that cannot succeed. Deliberate consequence (diff r21 SF3): the guard is the
            // FIRST statement of the loop, so on a lost store `flush()` no longer runs the
            // pending batched tick at all — a shutdown path that used `flush()` to drain
            // outbound sync loses that drain. The store cannot persist; draining it would
            // write nothing.
            {
                let core = self.core.lock().map_err(|_| RuntimeError::LockError)?;
                if let Some(error) = core.lost_writes() {
                    return Err(RuntimeError::WriteError(format!(
                        "storage flush or read-pass commit failed: {error}"
                    )));
                }
            }
            // Wait for any scheduled batched_tick to complete
            loop {
                let is_scheduled = {
                    let core = self.core.lock().map_err(|_| RuntimeError::LockError)?;
                    core.scheduler().is_scheduled()
                };

                if !is_scheduled {
                    break;
                }

                tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;

                attempts += 1;
                if attempts > 200 {
                    break;
                }
            }

            // Synchronous tick and check if more work was generated
            let has_more_work = {
                let mut core = self.core.lock().map_err(|_| RuntimeError::LockError)?;
                if core.has_storage_write_pending_flush()
                    && core.has_storage_flush_retry_scheduled()
                    && core.has_storage_flush_error()
                    && let Some(error) = core.take_storage_flush_error()
                {
                    return Err(RuntimeError::WriteError(format!(
                        "storage flush or read-pass commit failed: {error}"
                    )));
                }
                drain_staging_into(self.scheduler.staging(), &mut core);
                core.batched_tick();
                core.has_outbound()
                    || core.scheduler().is_scheduled()
                    || core.has_storage_write_pending_flush()
            };

            if !has_more_work {
                break;
            }

            attempts += 1;
            if attempts > 200 {
                break;
            }
        }

        let mut core = self.core.lock().map_err(|_| RuntimeError::LockError)?;
        if let Some(error) = core.take_storage_flush_error() {
            return Err(RuntimeError::WriteError(format!(
                "storage flush or read-pass commit failed: {error}"
            )));
        }
        if core.has_storage_write_pending_flush() {
            return Err(RuntimeError::WriteError(
                "storage WAL flush did not complete".to_string(),
            ));
        }

        Ok(())
    }

    // =========================================================================
    // Queries
    // =========================================================================

    /// Execute a one-shot query with durability options.
    ///
    /// When `batch_id` is provided, the query is scoped to that open local
    /// batch/transaction.
    pub fn query(
        &self,
        query: Query,
        session: Option<Session>,
        durability: ReadDurabilityOptions,
        batch_id: Option<BatchId>,
    ) -> Result<QueryFuture, RuntimeError> {
        let mut core = self.core.lock().map_err(|_| RuntimeError::LockError)?;
        Ok(core.query_with_local_batch(
            query,
            session,
            durability,
            QueryPropagation::Full,
            batch_id,
        )?)
    }

    // =========================================================================
    // Subscriptions
    // =========================================================================

    /// Subscribe to a query with a callback.
    pub fn subscribe<F>(
        &self,
        query: Query,
        callback: F,
        session: Option<Session>,
    ) -> Result<SubscriptionHandle, RuntimeError>
    where
        F: Fn(SubscriptionDelta) + Send + 'static,
    {
        let mut core = self.core.lock().map_err(|_| RuntimeError::LockError)?;
        core.subscribe(query, callback, session)
            .map_err(|e| RuntimeError::QueryError(e.to_string()))
    }

    /// Unsubscribe from a query.
    ///
    /// The core schedules a batched tick to send the unsubscription upstream; the tick runs
    /// on the `jazz-tick` thread, so no runtime context is needed here.
    pub fn unsubscribe(&self, handle: SubscriptionHandle) -> Result<(), RuntimeError> {
        let mut core = self.core.lock().map_err(|_| RuntimeError::LockError)?;
        core.unsubscribe(handle);
        Ok(())
    }

    // =========================================================================
    // Sync Operations
    // =========================================================================

    /// Push a sync message to the inbox (from network).
    pub fn push_sync_inbox(&self, entry: InboxEntry) -> Result<(), RuntimeError> {
        let mut core = self.core.lock().map_err(|_| RuntimeError::LockError)?;
        core.park_sync_message(entry);
        Ok(())
    }

    /// Push multiple sync messages to the inbox under a single core lock.
    pub fn push_sync_inbox_batch(&self, entries: Vec<InboxEntry>) -> Result<(), RuntimeError> {
        if entries.is_empty() {
            return Ok(());
        }
        let mut core = self.core.lock().map_err(|_| RuntimeError::LockError)?;
        for entry in entries {
            core.park_sync_message(entry);
        }
        Ok(())
    }

    /// v18 item 3: stage a client's decoded frame outside the engine lock. Returns
    /// `Staged` (a tick is scheduled) or `Backpressure` with the entries, the permit and a
    /// waiter guard; the caller waits on the guard and retries with
    /// `stage_sync_inbox_with_waiter`.
    pub fn stage_sync_inbox(
        &self,
        client_id: ClientId,
        entries: Vec<InboxEntry>,
        bytes: usize,
        permit: Option<tokio::sync::OwnedSemaphorePermit>,
    ) -> StagePush {
        let outcome = self
            .scheduler
            .staging()
            .push(client_id, entries, bytes, permit);
        if matches!(outcome, StagePush::Staged) {
            self.scheduler.schedule_batched_tick();
        }
        outcome
    }

    /// The retry of a refused `stage_sync_inbox`.
    pub fn stage_sync_inbox_with_waiter(
        &self,
        waiter: WaiterGuard,
        client_id: ClientId,
        entries: Vec<InboxEntry>,
        bytes: usize,
        permit: Option<tokio::sync::OwnedSemaphorePermit>,
    ) -> StagePush {
        let outcome = self
            .scheduler
            .staging()
            .push_with(waiter, client_id, entries, bytes, permit);
        if matches!(outcome, StagePush::Staged) {
            self.scheduler.schedule_batched_tick();
        }
        outcome
    }

    /// The exit path of a socket that still holds an admitted frame: staged regardless of
    /// the caps, then a tick is scheduled.
    pub fn stage_sync_inbox_uncapped(
        &self,
        waiter: Option<WaiterGuard>,
        client_id: ClientId,
        entries: Vec<InboxEntry>,
        bytes: usize,
        permit: Option<tokio::sync::OwnedSemaphorePermit>,
    ) {
        self.scheduler
            .staging()
            .push_uncapped(waiter, client_id, entries, bytes, permit);
        self.scheduler.schedule_batched_tick();
    }

    /// The in-flight decoded-bytes budget (KiB permits) a socket acquires before decoding.
    pub fn inflight_budget(&self) -> Arc<tokio::sync::Semaphore> {
        self.scheduler.staging().budget()
    }
    pub fn staging_config(&self) -> StagingConfig {
        self.scheduler.staging().config().clone()
    }

    /// Per-client staging counters. Internal on purpose: gates for the staging's own
    /// accounting, which no client API exposes.
    #[cfg(any(test, feature = "test"))]
    pub fn staging_stats_for_test(&self, client_id: ClientId) -> ClientStagingStats {
        self.scheduler.staging().stats(client_id)
    }

    /// Clients with a live staging entry (staged or waiting). Internal on purpose.
    #[cfg(any(test, feature = "test"))]
    pub fn staging_live_clients_for_test(&self) -> usize {
        self.scheduler.staging().live_clients()
    }

    /// Set the next expected stream sequence for a server.
    pub fn set_server_next_sequence(
        &self,
        server_id: ServerId,
        next_sequence: u64,
    ) -> Result<(), RuntimeError> {
        let mut core = self.core.lock().map_err(|_| RuntimeError::LockError)?;
        core.set_next_expected_server_sequence(server_id, next_sequence);
        Ok(())
    }

    /// Add a server connection.
    pub fn add_server(&self, server_id: ServerId) -> Result<(), RuntimeError> {
        self.add_server_with_catalogue_state_hash(server_id, None)
    }

    /// Add a server connection, optionally comparing the upstream catalogue
    /// digest first so unchanged catalogue objects are not replayed.
    pub fn add_server_with_catalogue_state_hash(
        &self,
        server_id: ServerId,
        remote_catalogue_state_hash: Option<&str>,
    ) -> Result<(), RuntimeError> {
        let mut core = self.core.lock().map_err(|_| RuntimeError::LockError)?;
        core.add_server_with_catalogue_state_hash(server_id, remote_catalogue_state_hash);
        Ok(())
    }

    /// Remove a server connection.
    pub fn remove_server(&self, server_id: ServerId) -> Result<(), RuntimeError> {
        let mut core = self.core.lock().map_err(|_| RuntimeError::LockError)?;
        core.remove_server(server_id);
        Ok(())
    }

    /// Add a client connection.
    pub fn add_client(
        &self,
        client_id: ClientId,
        session: Option<Session>,
    ) -> Result<(), RuntimeError> {
        let mut core = self.core.lock().map_err(|_| RuntimeError::LockError)?;
        core.add_client(client_id, session);
        Ok(())
    }

    /// Ensure a client exists with the given session.
    ///
    /// A session is always required — callers must authenticate before
    /// registering a client.
    pub fn ensure_client_with_session(
        &self,
        client_id: ClientId,
        session: Session,
    ) -> Result<(), RuntimeError> {
        let mut core = self.core.lock().map_err(|_| RuntimeError::LockError)?;
        core.ensure_client_with_session(client_id, session);
        Ok(())
    }

    pub fn ensure_client_with_session_and_catalogue_state_hash(
        &self,
        client_id: ClientId,
        session: Session,
        remote_catalogue_state_hash: Option<&str>,
    ) -> Result<(), RuntimeError> {
        let mut core = self.core.lock().map_err(|_| RuntimeError::LockError)?;
        core.ensure_client_with_session_and_catalogue_state_hash(
            client_id,
            session,
            remote_catalogue_state_hash,
        );
        Ok(())
    }

    /// Ensure a client exists and is marked as Backend without resetting state.
    pub fn ensure_client_as_backend(&self, client_id: ClientId) -> Result<(), RuntimeError> {
        let mut core = self.core.lock().map_err(|_| RuntimeError::LockError)?;
        core.ensure_client_as_backend(client_id);
        Ok(())
    }

    pub fn ensure_client_as_backend_with_catalogue_state_hash(
        &self,
        client_id: ClientId,
        remote_catalogue_state_hash: Option<&str>,
    ) -> Result<(), RuntimeError> {
        let mut core = self.core.lock().map_err(|_| RuntimeError::LockError)?;
        core.ensure_client_as_backend_with_catalogue_state_hash(
            client_id,
            remote_catalogue_state_hash,
        );
        Ok(())
    }

    /// Ensure a client exists and is marked as Peer without resetting state.
    pub fn ensure_client_as_peer(&self, client_id: ClientId) -> Result<(), RuntimeError> {
        let mut core = self.core.lock().map_err(|_| RuntimeError::LockError)?;
        core.ensure_client_as_peer(client_id);
        Ok(())
    }

    /// Ensure a peer client exists and replay catalogue only when its digest is stale.
    pub fn ensure_client_as_peer_with_catalogue_state_hash(
        &self,
        client_id: ClientId,
        remote_catalogue_state_hash: Option<&str>,
    ) -> Result<(), RuntimeError> {
        let mut core = self.core.lock().map_err(|_| RuntimeError::LockError)?;
        core.ensure_client_as_peer_with_catalogue_state_hash(
            client_id,
            remote_catalogue_state_hash,
        );
        Ok(())
    }

    /// Ensure a client exists and is marked as Admin without resetting state.
    pub fn ensure_client_as_admin(&self, client_id: ClientId) -> Result<(), RuntimeError> {
        let mut core = self.core.lock().map_err(|_| RuntimeError::LockError)?;
        core.ensure_client_as_admin(client_id);
        Ok(())
    }

    pub fn ensure_client_as_admin_with_catalogue_state_hash(
        &self,
        client_id: ClientId,
        remote_catalogue_state_hash: Option<&str>,
    ) -> Result<(), RuntimeError> {
        let mut core = self.core.lock().map_err(|_| RuntimeError::LockError)?;
        core.ensure_client_as_admin_with_catalogue_state_hash(
            client_id,
            remote_catalogue_state_hash,
        );
        Ok(())
    }

    /// Remove a client connection.
    ///
    /// Returns `Ok(true)` if removed, `Ok(false)` if skipped due to
    /// unprocessed inbox entries (caller should retry later).
    pub fn remove_client(&self, client_id: ClientId) -> Result<bool, RuntimeError> {
        // A client with a staged frame or a paused socket is not gone: its entries have not
        // been parked yet, and removing it now would make the drain log them as "unknown
        // client". This takes and releases the STAGING lock, before the core lock and never
        // under it — so the two are not atomic and this line alone does not enforce that
        // invariant. What enforces it is the ordering at both ends: the socket-exit path stages
        // its held frame BEFORE `ws_cleanup` inserts the reap candidate (`websocket.rs`), and
        // the sweep re-checks `has_connection` immediately before reaping (`server/mod.rs`).
        // This check is the cheap early out for the common case, not the guarantee.
        if self.scheduler.staging().blocks_removal(client_id) {
            return Ok(false);
        }
        let mut core = self.core.lock().map_err(|_| RuntimeError::LockError)?;
        Ok(core.remove_client(client_id))
    }

    /// Record whether a client confirms the rows it applies (negotiated in the handshake).
    pub fn set_client_acks_deliveries(
        &self,
        client_id: ClientId,
        acks: bool,
    ) -> Result<(), RuntimeError> {
        let mut core = self.core.lock().map_err(|_| RuntimeError::LockError)?;
        core.set_client_acks_deliveries(client_id, acks);
        Ok(())
    }

    /// Promote a client to Admin role (full access, no ReBAC).
    pub fn set_client_admin(&self, client_id: ClientId) -> Result<(), RuntimeError> {
        let mut core = self.core.lock().map_err(|_| RuntimeError::LockError)?;
        core.set_client_admin(client_id);
        Ok(())
    }

    /// Promote a client to Backend role (row access, no catalogue writes).
    pub fn set_client_backend(&self, client_id: ClientId) -> Result<(), RuntimeError> {
        let mut core = self.core.lock().map_err(|_| RuntimeError::LockError)?;
        core.set_client_backend(client_id);
        Ok(())
    }

    // =========================================================================
    // Schema Access
    // =========================================================================

    /// Get a clone of the current schema.
    pub fn current_schema(&self) -> Result<Schema, RuntimeError> {
        let core = self.core.lock().map_err(|_| RuntimeError::LockError)?;
        Ok(core.current_schema().clone())
    }

    /// Return all known schema hashes (for server mode).
    pub fn known_schema_hashes(&self) -> Result<Vec<SchemaHash>, RuntimeError> {
        let core = self.core.lock().map_err(|_| RuntimeError::LockError)?;
        Ok(core.schema_manager().known_schema_hashes())
    }

    /// Return a canonical digest of the runtime's catalogue state.
    pub fn catalogue_state_hash(&self) -> Result<String, RuntimeError> {
        let core = self.core.lock().map_err(|_| RuntimeError::LockError)?;
        Ok(core.schema_manager().catalogue_state_hash())
    }

    /// Get a known schema by hash from catalogue state.
    pub fn known_schema(&self, schema_hash: &SchemaHash) -> Result<Option<Schema>, RuntimeError> {
        let core = self.core.lock().map_err(|_| RuntimeError::LockError)?;
        Ok(core.schema_manager().get_known_schema(schema_hash).cloned())
    }

    /// Return the latest publish timestamp for a schema catalogue object.
    pub fn schema_published_at(
        &self,
        schema_hash: &SchemaHash,
    ) -> Result<Option<u64>, RuntimeError> {
        let core = self.core.lock().map_err(|_| RuntimeError::LockError)?;
        Ok(core.schema_manager().schema_published_at(schema_hash))
    }

    /// Seed an additional known schema into the in-memory schema manager.
    pub fn add_known_schema(&self, schema: Schema) -> Result<(), RuntimeError> {
        let mut core = self.core.lock().map_err(|_| RuntimeError::LockError)?;
        core.schema_manager_mut().add_known_schema(schema);
        Ok(())
    }

    /// Return grouped telemetry for active downstream server subscriptions.
    pub fn server_subscription_telemetry(
        &self,
    ) -> Result<Vec<crate::query_manager::manager::ServerSubscriptionTelemetryGroup>, RuntimeError>
    {
        let core = self.core.lock().map_err(|_| RuntimeError::LockError)?;
        Ok(core
            .schema_manager()
            .query_manager()
            .server_subscription_telemetry())
    }

    /// Access the underlying storage (for flushing, etc).
    ///
    /// The callback receives `&S` while holding the core lock.
    pub fn with_storage<R>(&self, f: impl FnOnce(&S) -> R) -> Result<R, RuntimeError> {
        let core = self.core.lock().map_err(|_| RuntimeError::LockError)?;
        Ok(f(core.storage()))
    }

    /// Run a closure with read access to the SyncManager (for testing/inspection).
    #[cfg(test)]
    pub(crate) fn with_sync_manager<R>(
        &self,
        f: impl FnOnce(&crate::sync_manager::SyncManager) -> R,
    ) -> Result<R, RuntimeError> {
        let core = self.core.lock().map_err(|_| RuntimeError::LockError)?;
        Ok(f(core.schema_manager().query_manager().sync_manager()))
    }

    /// Access the underlying schema manager while holding the core lock.
    pub fn with_schema_manager<R>(
        &self,
        f: impl FnOnce(&SchemaManager) -> R,
    ) -> Result<R, RuntimeError> {
        let core = self.core.lock().map_err(|_| RuntimeError::LockError)?;
        Ok(f(core.schema_manager()))
    }

    /// Return a reference to the scheduler stored on this runtime handle.
    ///
    /// The returned scheduler shares `Arc`-based state with the one inside
    /// `RuntimeCore` (same debounce flag, same `Weak` back-reference), so
    /// calling `schedule_batched_tick()` on it is equivalent to calling it
    /// from within the locked core.
    pub fn scheduler(&self) -> &TokioScheduler<S> {
        &self.scheduler
    }
}

// ============================================================================
// NativeTickNotifier
// ============================================================================

/// `TickNotifier` implementation for the native (Tokio) runtime.
///
/// Holds a clone of the `TokioScheduler` and calls
/// `schedule_batched_tick()` whenever the transport layer needs to wake
/// up `batched_tick` (on connect, on incoming sync frames, on disconnect).
#[derive(Clone)]
pub struct NativeTickNotifier<S: Storage + Send + 'static> {
    scheduler: TokioScheduler<S>,
}

impl<S: Storage + Send + 'static> crate::transport_manager::TickNotifier for NativeTickNotifier<S> {
    fn notify(&self) {
        self.scheduler.schedule_batched_tick();
    }
}

// ============================================================================
// TokioRuntime connect / disconnect (WebSocket transport)
// ============================================================================

#[cfg(feature = "transport-websocket")]
impl<S: Storage + Send + 'static> TokioRuntime<S> {
    /// Connect to a Jazz server over WebSocket.
    ///
    /// Creates a `TransportHandle` / `TransportManager` pair, wires the
    /// handle into `RuntimeCore`, and spawns the manager loop as a Tokio
    /// task. The manager drives the WebSocket connection, reconnecting on
    /// failure until the handle is dropped.
    pub fn connect(&self, url: String, auth: crate::transport_manager::AuthConfig) {
        self.connect_with_retry_config(
            url,
            auth,
            crate::transport_manager::TransportRetryConfig::default(),
        );
    }

    /// Connect to a Jazz server over WebSocket with explicit retry attempt
    /// deadlines.
    pub fn connect_with_retry_config(
        &self,
        url: String,
        auth: crate::transport_manager::AuthConfig,
        retry_config: crate::transport_manager::TransportRetryConfig,
    ) {
        let tick = NativeTickNotifier {
            scheduler: self.scheduler.clone(),
        };
        let manager = {
            let mut core = self.core.lock().unwrap();
            crate::runtime_core::install_transport_with_retry_config::<
                _,
                _,
                crate::ws_stream::NativeWsStream,
                _,
            >(&mut core, url, auth, tick, retry_config)
        };
        tokio::spawn(manager.run());
    }

    /// Disconnect from the Jazz server.
    ///
    /// Drops the `TransportHandle` from `RuntimeCore`. The spawned
    /// `TransportManager` task detects the dropped handle and exits cleanly.
    pub fn disconnect(&self) {
        self.core.lock().unwrap().clear_transport();
    }

    /// Returns `true` once the WebSocket transport has completed at least one
    /// successful handshake. Useful for callers that need to wait until the
    /// initial connection is established before proceeding.
    pub fn transport_ever_connected(&self) -> bool {
        self.core
            .lock()
            .ok()
            .and_then(|c| c.transport.as_ref().map(|h| h.has_ever_connected()))
            .unwrap_or(false)
    }

    /// Async wait that resolves once the WebSocket transport has completed
    /// its first successful handshake (or returns immediately if it already
    /// has). Returns `false` when no transport is installed — callers should
    /// either check `is_connected()`/`has_server` first or wrap this in a
    /// timeout.
    pub async fn transport_wait_until_connected(&self) -> bool {
        let Some(mut rx) = self
            .core
            .lock()
            .ok()
            .and_then(|c| c.transport.as_ref().map(|h| h.connected_rx.clone()))
        else {
            return false;
        };
        if *rx.borrow() {
            return true;
        }
        rx.wait_for(|connected| *connected).await.is_ok()
    }

    /// Returns the wire `ClientId` used by the active transport, if any.
    ///
    /// Tests use this to register the transport's client identity with a
    /// `SyncTracer` so server-originated messages resolve to human names.
    pub fn transport_client_id(&self) -> Option<ClientId> {
        self.core
            .lock()
            .ok()
            .and_then(|c| c.transport.as_ref().map(|h| h.client_id))
    }

    /// Attach a sync-message tracer to this runtime.
    pub fn set_sync_tracer(&self, tracer: crate::sync_tracer::SyncTracer, name: String) {
        if let Ok(mut core) = self.core.lock() {
            core.set_sync_tracer(tracer, name);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query_manager::types::{ColumnType, SchemaBuilder, TableSchema};
    use crate::schema_manager::AppId;
    use crate::storage::{MemoryStorage, StorageError};
    use crate::sync_manager::SyncManager;
    use std::sync::atomic::AtomicUsize;

    fn test_schema() -> Schema {
        SchemaBuilder::new()
            .table(
                TableSchema::builder("users")
                    .column("id", ColumnType::Uuid)
                    .column("name", ColumnType::Text),
            )
            .build()
    }

    fn user_row_values(id: ObjectId, name: &str) -> Vec<Value> {
        vec![Value::Uuid(id), Value::Text(name.to_string())]
    }

    fn user_insert_values(id: ObjectId, name: &str) -> HashMap<String, Value> {
        HashMap::from([
            ("id".to_string(), Value::Uuid(id)),
            ("name".to_string(), Value::Text(name.to_string())),
        ])
    }

    #[tokio::test]
    async fn test_runtime_insert_query() {
        let schema = test_schema();
        let app_id = AppId::from_name("test-app");
        let sync_manager = SyncManager::new();
        let schema_manager =
            SchemaManager::new(sync_manager, schema, app_id, "dev", "main").unwrap();

        let sync_count = Arc::new(AtomicUsize::new(0));
        let sync_count_clone = sync_count.clone();

        let runtime = TokioRuntime::new(schema_manager, MemoryStorage::new(), move |_msg| {
            sync_count_clone.fetch_add(1, Ordering::SeqCst);
        });

        // Insert a row
        let user_id = ObjectId::new();
        let expected_values = user_row_values(user_id, "Alice");
        let (object_id, row_values, _batch_id) = runtime
            .insert("users", user_insert_values(user_id, "Alice"), None)
            .unwrap();
        assert!(!object_id.0.is_nil());
        assert_eq!(row_values, expected_values);

        // Query
        let query = Query::new("users");
        let future = runtime
            .query(query, None, ReadDurabilityOptions::default(), None)
            .unwrap();
        let results = future.await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, object_id);
    }

    #[tokio::test]
    async fn test_runtime_update_delete() {
        let schema = test_schema();
        let app_id = AppId::from_name("test-crud");
        let sync_manager = SyncManager::new();
        let schema_manager =
            SchemaManager::new(sync_manager, schema, app_id, "dev", "main").unwrap();

        let runtime = TokioRuntime::new(schema_manager, MemoryStorage::new(), |_| {});

        // Insert
        let (object_id, _row_values, _batch_id) = runtime
            .insert("users", user_insert_values(ObjectId::new(), "Bob"), None)
            .unwrap();

        // Update
        let updates = vec![("name".to_string(), Value::Text("Charlie".to_string()))];
        runtime.update(object_id, updates, None).unwrap();

        // Verify update
        let query = Query::new("users");
        let future = runtime
            .query(query, None, ReadDurabilityOptions::default(), None)
            .unwrap();
        let results = future.await.unwrap();
        assert_eq!(results[0].1[1], Value::Text("Charlie".to_string()));

        // Delete
        runtime.delete(object_id, None).unwrap();

        // Verify deleted
        let query = Query::new("users");
        let future = runtime
            .query(query, None, ReadDurabilityOptions::default(), None)
            .unwrap();
        let results = future.await.unwrap();
        assert_eq!(results.len(), 0);
    }

    #[tokio::test]
    async fn flush_returns_error_when_wal_flush_fails() {
        let schema = test_schema();
        let app_id = AppId::from_name("test-wal-flush-failure");
        let sync_manager = SyncManager::new();
        let schema_manager =
            SchemaManager::new(sync_manager, schema, app_id, "dev", "main").unwrap();

        let storage = MemoryStorage::new().with_flush_wal_error(StorageError::IoError(
            "injected WAL flush failure".to_string(),
        ));
        let runtime = TokioRuntime::new(schema_manager, storage, |_| {});

        runtime
            .insert("users", user_insert_values(ObjectId::new(), "Alice"), None)
            .unwrap();

        let error = runtime
            .flush()
            .await
            .expect_err("explicit runtime flush should surface WAL flush failure");
        assert!(
            error.to_string().contains("injected WAL flush failure"),
            "unexpected error: {error}"
        );
        let flush_wal_calls = runtime
            .with_storage(|storage| storage.flush_wal_call_count())
            .expect("inspect storage calls");
        assert!(
            flush_wal_calls <= 2,
            "persistent WAL flush failures should not be retried in a tight loop, got {flush_wal_calls} attempts"
        );
    }

    /// G-C15. Once the store has lost writes, every later `flush()` returns THAT error at
    /// once — no 200-attempt spin, no ticks, and not the generic "storage WAL flush did not
    /// complete" the spin ends with. A flush is an explicit ask and deserves the true answer;
    /// the tick path stays once-only (G-C14), because a tick is a timer.
    ///
    /// Internal on purpose: the assertion is about how many `flush_wal` calls the spin made,
    /// which only the injected store counts.
    #[tokio::test]
    async fn flush_after_a_lost_write_returns_the_loss_without_spinning() {
        let schema = test_schema();
        let app_id = AppId::from_name("test-wal-lost-writes");
        let sync_manager = SyncManager::new();
        let schema_manager =
            SchemaManager::new(sync_manager, schema, app_id, "dev", "main").unwrap();
        let storage = MemoryStorage::new().with_flush_wal_error(StorageError::LostWrites {
            detail: "the write transaction was ended behind the store's back".to_string(),
        });
        let runtime = TokioRuntime::new(schema_manager, storage, |_| {});
        runtime
            .insert("users", user_insert_values(ObjectId::new(), "Alice"), None)
            .unwrap();
        for attempt in 1..=3 {
            let error = runtime
                .flush()
                .await
                .expect_err("a store that lost writes can never flush");
            assert!(
                error.to_string().contains("ended behind the store's back"),
                "attempt {attempt}: the flush must carry the loss itself, got: {error}"
            );
        }
        let flush_wal_calls = runtime
            .with_storage(|storage| storage.flush_wal_call_count())
            .expect("inspect storage calls");
        assert!(
            flush_wal_calls <= 2,
            "a lost store must not be flushed again on every call, got {flush_wal_calls} \
             attempts across three flushes"
        );
    }

    #[tokio::test]
    async fn flush_recovers_after_transient_wal_flush_failure() {
        let schema = test_schema();
        let app_id = AppId::from_name("test-transient-wal-flush-failure");
        let sync_manager = SyncManager::new();
        let schema_manager =
            SchemaManager::new(sync_manager, schema, app_id, "dev", "main").unwrap();

        let storage = MemoryStorage::new().with_transient_flush_wal_failures(
            StorageError::IoError("transient WAL flush failure".to_string()),
            1,
        );
        let runtime = TokioRuntime::new(schema_manager, storage, |_| {});

        runtime
            .insert("users", user_insert_values(ObjectId::new(), "Alice"), None)
            .unwrap();

        runtime
            .flush()
            .await
            .expect("explicit runtime flush should retry transient WAL flush failure");
    }

    #[tokio::test]
    async fn flush_retries_stored_wal_error_before_returning_failure() {
        let schema = test_schema();
        let app_id = AppId::from_name("test-stored-wal-flush-error");
        let sync_manager = SyncManager::new();
        let schema_manager =
            SchemaManager::new(sync_manager, schema, app_id, "dev", "main").unwrap();
        let runtime = TokioRuntime::new(schema_manager, MemoryStorage::new(), |_| {});

        {
            let mut core = runtime.core.lock().expect("lock runtime core");
            core.mark_storage_write_pending_flush();
            core.record_storage_flush_error(StorageError::IoError(
                "previous transient WAL flush failure".to_string(),
            ));
        }

        runtime
            .flush()
            .await
            .expect("explicit runtime flush should retry the pending WAL barrier");

        let flush_wal_calls = runtime
            .with_storage(|storage| storage.flush_wal_call_count())
            .expect("inspect storage calls");
        assert_eq!(flush_wal_calls, 1);
    }

    #[tokio::test]
    async fn test_subscription_callback() {
        use std::sync::Mutex;

        let schema = test_schema();
        let app_id = AppId::from_name("test-subscription");
        let sync_manager = SyncManager::new();
        let schema_manager =
            SchemaManager::new(sync_manager, schema, app_id, "dev", "main").unwrap();

        let runtime = TokioRuntime::new(schema_manager, MemoryStorage::new(), |_| {});

        // Track callback invocations
        let updates: Arc<Mutex<Vec<SubscriptionDelta>>> = Arc::new(Mutex::new(Vec::new()));
        let updates_clone = updates.clone();

        // Subscribe to users table
        let query = Query::new("users");
        let handle = runtime
            .subscribe(
                query,
                move |delta| {
                    updates_clone.lock().unwrap().push(delta);
                },
                None,
            )
            .unwrap();

        // Insert a row - this should trigger the subscription callback
        let (_object_id, _row_values, _batch_id) = runtime
            .insert("users", user_insert_values(ObjectId::new(), "Eve"), None)
            .unwrap();

        // Verify callback was invoked
        let updates_vec = updates.lock().unwrap();
        assert!(
            !updates_vec.is_empty(),
            "Subscription callback should have been invoked after insert"
        );
        assert_eq!(updates_vec[0].handle, handle);

        // Cleanup
        drop(updates_vec);
        runtime.unsubscribe(handle).unwrap();
    }
}
