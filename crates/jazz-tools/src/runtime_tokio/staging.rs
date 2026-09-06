//! v18 item 3 (half L): inbound frames are staged outside the engine lock.
//!
//! A socket task that receives a frame used to take the core `std::sync::Mutex` inline and
//! park the entries; for the length of any settle pass that task blocked its tokio worker.
//! Now the socket pushes into this staging (a small `std::sync::Mutex` never held across an
//! `.await` and never held while the core lock is taken), and the tick thread drains the
//! staging under the core lock right before `batched_tick`.
//!
//! Bounds (design 2026-09-v18-03 § v4–v7):
//! - per client, entries and decoded bytes are capped; a frame is accepted whole when the
//!   client has nothing staged, otherwise refused whole when it would cross either cap — the
//!   socket then pauses its inbound side and retries after the next drain (`WaiterGuard`);
//! - globally, decoded bytes in flight are bounded by a semaphore in KiB permits: the socket
//!   acquires the permits before decoding and hands the permit over with the frame; the drain
//!   returns the permits when it parks the entries.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use tokio::sync::{Notify, OwnedSemaphorePermit, Semaphore};

use crate::sync_manager::{ClientId, InboxEntry};

/// Staging and budget knobs. Read from the environment by `TokioRuntime::new`; the server
/// builder overrides them for tests.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StagingConfig {
    /// `JAZZ_MAX_STAGED_ENTRIES_PER_CLIENT`: entries a client may have staged before its
    /// sockets pause.
    pub max_entries_per_client: usize,
    /// `JAZZ_MAX_STAGED_BYTES_PER_CLIENT`: decoded bytes a client may have staged before
    /// its sockets pause.
    pub max_bytes_per_client: usize,
    /// `JAZZ_MAX_INFLIGHT_DECODED_BYTES`: decoded bytes admitted (staged or held by a paused
    /// socket) across every client. Floor-divided into KiB permits.
    pub inflight_budget_bytes: usize,
}

impl StagingConfig {
    pub const DEFAULT_MAX_ENTRIES_PER_CLIENT: usize = 1024;
    pub const DEFAULT_MAX_BYTES_PER_CLIENT: usize = 4 * 1024 * 1024;
    pub const DEFAULT_INFLIGHT_BUDGET_BYTES: usize = 1024 * 1024 * 1024;

    pub fn from_env() -> Self {
        Self {
            max_entries_per_client: env_usize(
                "JAZZ_MAX_STAGED_ENTRIES_PER_CLIENT",
                Self::DEFAULT_MAX_ENTRIES_PER_CLIENT,
            ),
            max_bytes_per_client: env_usize(
                "JAZZ_MAX_STAGED_BYTES_PER_CLIENT",
                Self::DEFAULT_MAX_BYTES_PER_CLIENT,
            ),
            inflight_budget_bytes: env_usize(
                "JAZZ_MAX_INFLIGHT_DECODED_BYTES",
                Self::DEFAULT_INFLIGHT_BUDGET_BYTES,
            ),
        }
    }

    /// The budget in permits (KiB, floored).
    pub fn budget_permits(&self) -> usize {
        self.inflight_budget_bytes / 1024
    }
}

impl Default for StagingConfig {
    fn default() -> Self {
        Self {
            max_entries_per_client: Self::DEFAULT_MAX_ENTRIES_PER_CLIENT,
            max_bytes_per_client: Self::DEFAULT_MAX_BYTES_PER_CLIENT,
            inflight_budget_bytes: Self::DEFAULT_INFLIGHT_BUDGET_BYTES,
        }
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(default)
}

/// Permits a decoded size of `bytes` needs (KiB, rounded up). A declared 0 rounds to 0
/// permits: `Ready` at once, `forget`/`add_permits(0)` no-ops.
pub fn kib_permits(bytes: usize) -> usize {
    bytes.div_ceil(1024)
}

struct ClientStaging {
    count: usize,
    bytes: usize,
    /// Sockets of this client that were refused and hold this entry's `notify`. The entry
    /// is never removed while this is non-zero, so every drain signals the handle they wait
    /// on.
    waiters: usize,
    notify: Arc<Notify>,
    stats: ClientStagingStats,
}

impl Default for ClientStaging {
    fn default() -> Self {
        Self {
            count: 0,
            bytes: 0,
            waiters: 0,
            notify: Arc::new(Notify::new()),
            stats: ClientStagingStats::default(),
        }
    }
}

/// Per-client counters for gates. Internal on purpose: no wire message reports them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClientStagingStats {
    /// Frames accepted into the staging (capped and uncapped).
    pub staged_frames: u64,
    /// Pushes refused by the caps.
    pub pauses: u64,
    /// The largest number of decoded bytes staged at once for the client.
    pub peak_bytes: usize,
    /// The largest single frame the client staged.
    pub largest_frame_bytes: usize,
    /// Current staged entries.
    pub count: usize,
    /// Current staged bytes.
    pub bytes: usize,
    /// Current waiters.
    pub waiters: usize,
}

struct StagedFrame {
    entries: Vec<InboxEntry>,
    /// Permits the frame's permit carried; returned to the budget when the frame is parked.
    kib: usize,
}

#[derive(Default)]
struct Staging {
    /// Arrival order across every client: a connection's frames are parked in the order its
    /// pushes returned.
    frames: Vec<StagedFrame>,
    per_client: HashMap<ClientId, ClientStaging>,
    /// Counters of entries that went away, kept for gates only.
    #[cfg(any(test, feature = "test"))]
    retired: HashMap<ClientId, ClientStagingStats>,
}

/// The shared staging of one runtime.
pub struct InboundStaging {
    inner: Mutex<Staging>,
    budget: Arc<Semaphore>,
    config: StagingConfig,
}

/// What `push` returns.
pub enum StagePush {
    /// The frame is staged; the caller schedules the tick.
    Staged,
    /// The client is at its cap. The entries and the permit come back with a guard that
    /// counts the socket as a waiter until it retries (`push_with`) or exits (drop).
    Backpressure {
        entries: Vec<InboxEntry>,
        permit: Option<OwnedSemaphorePermit>,
        waiter: WaiterGuard,
    },
}

/// Frames taken out of the staging by a drain; parked by the caller under the core lock.
pub struct DrainedFrames {
    frames: Vec<StagedFrame>,
    released_kib: usize,
}

impl DrainedFrames {
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// The entries in arrival order.
    pub fn into_entries(self) -> impl Iterator<Item = InboxEntry> {
        self.frames
            .into_iter()
            .flat_map(|frame| frame.entries.into_iter())
    }

    pub fn released_kib(&self) -> usize {
        self.released_kib
    }
}

/// A refused socket's place in the staging. Holds the client entry's `Notify` and counts the
/// socket as a waiter; `Drop` (the exit path) takes the staging lock itself, so it must
/// never run while the caller holds that lock — nothing but a hang would catch it.
pub struct WaiterGuard {
    staging: Option<Arc<InboundStaging>>,
    client: ClientId,
    notify: Option<Arc<Notify>>,
}

impl WaiterGuard {
    /// The `Notify` the socket waits on. Every drain calls `notify_waiters()` then
    /// `notify_one()` on it, so a `notified()` created after the refusal is woken by the next
    /// drain, and one created after that drain consumes the stored permit.
    pub fn notify_handle(&self) -> Arc<Notify> {
        Arc::clone(
            self.notify
                .as_ref()
                .expect("a live waiter guard holds its notify"),
        )
    }

    pub async fn notified(&self) {
        self.notify
            .as_ref()
            .expect("a live waiter guard holds its notify")
            .notified()
            .await;
    }
}

impl Drop for WaiterGuard {
    fn drop(&mut self) {
        if let Some(staging) = self.staging.take() {
            let mut inner = staging.lock();
            release_waiter(&mut inner, self.client);
        }
    }
}

/// The client's live entry, created on first use. Under test the counters of a retired
/// entry come back with it, so a gate can read a client's totals after its entry went away.
fn client_entry(inner: &mut Staging, client: ClientId) -> &mut ClientStaging {
    #[cfg(any(test, feature = "test"))]
    let retired = inner.retired.remove(&client);
    let entry = inner.per_client.entry(client).or_default();
    #[cfg(any(test, feature = "test"))]
    if let Some(stats) = retired {
        entry.stats = stats;
    }
    entry
}

/// Drop an idle entry (nothing staged, nobody waiting). Under test its counters are kept.
fn retire(inner: &mut Staging, client: ClientId) {
    #[cfg(any(test, feature = "test"))]
    if let Some(entry) = inner.per_client.remove(&client) {
        inner.retired.insert(client, entry.stats);
    }
    #[cfg(not(any(test, feature = "test")))]
    inner.per_client.remove(&client);
}

fn release_waiter(inner: &mut Staging, client: ClientId) {
    match inner.per_client.get_mut(&client) {
        Some(entry) => {
            debug_assert!(
                entry.waiters > 0,
                "a waiter guard released a client entry with no waiters — a double decrement \
                 would let a drain remove the entry another socket waits on"
            );
            entry.waiters = entry.waiters.saturating_sub(1);
            entry.stats.waiters = entry.waiters;
            if entry.count == 0 && entry.waiters == 0 {
                retire(inner, client);
            }
        }
        None => debug_assert!(
            false,
            "a waiter guard released a client entry that no longer exists — the entry must \
             outlive every waiter"
        ),
    }
}

impl InboundStaging {
    pub fn new(config: StagingConfig) -> Self {
        Self {
            inner: Mutex::new(Staging::default()),
            budget: Arc::new(Semaphore::new(config.budget_permits())),
            config,
        }
    }
    pub fn config(&self) -> &StagingConfig {
        &self.config
    }

    /// The in-flight decoded-bytes budget, in KiB permits.
    pub fn budget(&self) -> Arc<Semaphore> {
        Arc::clone(&self.budget)
    }
    fn lock(&self) -> std::sync::MutexGuard<'_, Staging> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Stage a frame for `client`. Accepted whole when the client has nothing staged;
    /// refused whole when it would cross the entries or bytes cap; never partial.
    pub fn push(
        self: &Arc<Self>,
        client: ClientId,
        entries: Vec<InboxEntry>,
        bytes: usize,
        permit: Option<OwnedSemaphorePermit>,
    ) -> StagePush {
        let mut inner = self.lock();
        self.push_locked(&mut inner, client, entries, bytes, permit)
    }

    /// The retry of a refused push: the waiter's decrement and the re-push happen under ONE
    /// staging lock, so a drain cannot remove the entry between them and mint a fresh
    /// `Notify`.
    pub fn push_with(
        self: &Arc<Self>,
        mut waiter: WaiterGuard,
        client: ClientId,
        entries: Vec<InboxEntry>,
        bytes: usize,
        permit: Option<OwnedSemaphorePermit>,
    ) -> StagePush {
        debug_assert_eq!(waiter.client, client, "a waiter retries for its own client");
        // Take the guard apart without running its `Drop` (which would take the lock again).
        let _notify = waiter.notify.take();
        let _staging = waiter.staging.take();
        std::mem::forget(waiter);
        let mut inner = self.lock();
        release_waiter(&mut inner, client);
        self.push_locked(&mut inner, client, entries, bytes, permit)
    }
    fn push_locked(
        self: &Arc<Self>,
        inner: &mut Staging,
        client: ClientId,
        entries: Vec<InboxEntry>,
        bytes: usize,
        permit: Option<OwnedSemaphorePermit>,
    ) -> StagePush {
        let entry = client_entry(&mut *inner, client);
        let accepted = entry.count == 0
            || (entry.count + entries.len() <= self.config.max_entries_per_client
                && entry.bytes + bytes <= self.config.max_bytes_per_client);
        if !accepted {
            entry.waiters += 1;
            entry.stats.pauses += 1;
            entry.stats.waiters = entry.waiters;
            let notify = Arc::clone(&entry.notify);
            return StagePush::Backpressure {
                entries,
                permit,
                waiter: WaiterGuard {
                    staging: Some(Arc::clone(self)),
                    client,
                    notify: Some(notify),
                },
            };
        }
        Self::store_locked(inner, client, entries, bytes, permit);
        StagePush::Staged
    }

    /// The loop-exit path with a decoded frame the socket already admitted: staged
    /// regardless of the caps (the connection is gone and cannot push more). A waiter guard
    /// the exiting socket still holds is released under the same lock.
    pub fn push_uncapped(
        &self,
        waiter: Option<WaiterGuard>,
        client: ClientId,
        entries: Vec<InboxEntry>,
        bytes: usize,
        permit: Option<OwnedSemaphorePermit>,
    ) {
        let mut inner = self.lock();
        if let Some(mut waiter) = waiter {
            let _notify = waiter.notify.take();
            let _staging = waiter.staging.take();
            std::mem::forget(waiter);
            release_waiter(&mut inner, client);
        }
        Self::store_locked(&mut inner, client, entries, bytes, permit);
    }
    fn store_locked(
        inner: &mut Staging,
        client: ClientId,
        entries: Vec<InboxEntry>,
        bytes: usize,
        permit: Option<OwnedSemaphorePermit>,
    ) {
        let entry = client_entry(&mut *inner, client);
        entry.count += entries.len();
        entry.bytes += bytes;
        entry.stats.staged_frames += 1;
        entry.stats.peak_bytes = entry.stats.peak_bytes.max(entry.bytes);
        entry.stats.largest_frame_bytes = entry.stats.largest_frame_bytes.max(bytes);
        entry.stats.count = entry.count;
        entry.stats.bytes = entry.bytes;
        // `kib` is what the permit carried, never recomputed; the permit is forgotten here,
        // under the lock, once the frame is stored, so no unwind can drop a permit the tick
        // will also release.
        let kib = permit.as_ref().map_or(0, |p| p.num_permits());
        if let Some(permit) = permit {
            permit.forget();
        }
        inner.frames.push(StagedFrame { entries, kib });
    }

    /// Take every staged frame. The per-client counters go to zero, every waiter is
    /// signalled (`notify_waiters` for those registered now, `notify_one` for one that
    /// registers after), and an entry is removed only when no socket waits on it.
    pub fn drain(&self) -> DrainedFrames {
        let mut inner = self.lock();
        let frames = std::mem::take(&mut inner.frames);
        let released_kib = frames.iter().map(|frame| frame.kib).sum();
        for entry in inner.per_client.values_mut() {
            entry.count = 0;
            entry.bytes = 0;
            entry.stats.count = 0;
            entry.stats.bytes = 0;
            entry.notify.notify_waiters();
            entry.notify.notify_one();
        }
        let idle: Vec<ClientId> = inner
            .per_client
            .iter()
            .filter(|(_, entry)| entry.waiters == 0)
            .map(|(client, _)| *client)
            .collect();
        for client in idle {
            retire(&mut inner, client);
        }
        DrainedFrames {
            frames,
            released_kib,
        }
    }

    /// Return the permits of parked frames to the budget. Called after the staging lock is
    /// dropped and after the entries are parked.
    pub fn release_permits(&self, kib: usize) {
        if kib > 0 {
            self.budget.add_permits(kib);
        }
    }

    /// `true` while the client has staged entries or paused sockets — a sweep must not
    /// remove such a client. Removes an idle entry on the way.
    pub fn blocks_removal(&self, client: ClientId) -> bool {
        let mut inner = self.lock();
        match inner.per_client.get(&client) {
            Some(entry) if entry.count > 0 || entry.waiters > 0 => true,
            Some(_) => {
                retire(&mut inner, client);
                false
            }
            None => false,
        }
    }

    /// Per-client counters; for gates.
    #[cfg(any(test, feature = "test"))]
    pub fn stats(&self, client: ClientId) -> ClientStagingStats {
        let inner = self.lock();
        if let Some(entry) = inner.per_client.get(&client) {
            return entry.stats;
        }
        #[cfg(any(test, feature = "test"))]
        if let Some(stats) = inner.retired.get(&client) {
            return *stats;
        }
        ClientStagingStats::default()
    }

    /// Clients with a live entry (staged or waiting). For gates.
    #[cfg(any(test, feature = "test"))]
    pub fn live_clients(&self) -> usize {
        self.lock().per_client.len()
    }
}
