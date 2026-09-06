//! SQLite-backed Storage implementation.
//!
//! Uses `rusqlite` with bundled SQLite. Single KV table on a WITHOUT ROWID
//! B-tree, WAL mode. Writes are batched into a lazy explicit transaction that
//! stays open across multiple calls and is committed on `flush()` / `close()`.
//! Per-operation SAVEPOINTs nested inside that transaction provide rollback
//! semantics for individual operations. Targets React Native / mobile.

use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

use rusqlite::OptionalExtension;

use super::{
    HistoryRowBytes, IndexMutation, OwnedHistoryRowBytes, OwnedVisibleRowBytes, RawTableMutation,
    Storage, StorageError, VisibleRowBytes, key_codec,
    storage_core::{
        append_history_region_row_bytes_core, raw_table_delete_core, raw_table_get_core,
        raw_table_put_core, raw_table_scan_prefix_core, raw_table_scan_prefix_keys_core,
        raw_table_scan_range_core, raw_table_scan_range_keys_core,
        upsert_visible_region_row_bytes_core,
    },
};
use crate::object::ObjectId;
use crate::row_histories::{HistoryScan, RowState, StoredRowBatch};
use crate::sync_manager::DurabilityTier;
/// v18 item 4: the transaction cells every write helper and pass boundary reads. `Cell`s so
/// that `with_savepoint` (an associated fn over `&Connection`) can mark them from inside a
/// closure that also borrows other fields of `SqliteInner` mutably (disjoint fields).
struct TxCells {
    /// At least one statement is committed into the current explicit transaction (set by
    /// `with_savepoint` from `total_changes()`, cleared by a successful COMMIT or a fresh
    /// BEGIN). Invariant: `dirty && conn.is_autocommit()` ⇔ those writes were rolled back
    /// behind the store's back (SQLite's own full rollback) — lost.
    dirty: Cell<bool>,
    /// The extended result code of the last failed SQLite call that went through
    /// `sqlite_error`. The BUSY_SNAPSHOT retry reads it (cleared at every `with_inner_mut`
    /// entry so a stale code can never decide a retry); tests observe it. The two discarded
    /// savepoint-unwind statements (`ROLLBACK TO` / `RELEASE` on the error path) never
    /// touch it: after a full rollback they answer "no such savepoint" and would overwrite
    /// the code that matters.
    last_sqlite_error: Cell<Option<i32>>,
    /// Test hook (`roll_back_after_next_write_for_test`): an ARMED ACTION, not an injected
    /// failure — after the next landed write, `ROLLBACK` the connection and report what
    /// SQLite's state then is (the state its own full rollback leaves). Always compiled:
    /// one bool beats a cfg'd signature.
    roll_back_after_next_write: Cell<bool>,
}

impl TxCells {
    fn new() -> Self {
        Self {
            dirty: Cell::new(false),
            last_sqlite_error: Cell::new(None),
            roll_back_after_next_write: Cell::new(false),
        }
    }
}

/// Map a rusqlite error, recording its extended result code first.
fn sqlite_error(last: &Cell<Option<i32>>, context: &str, e: rusqlite::Error) -> StorageError {
    if let rusqlite::Error::SqliteFailure(ffi, _) = &e {
        last.set(Some(ffi.extended_code));
    }
    StorageError::IoError(format!("{context}: {e}"))
}

const READ_ONLY_WRITE: &str = "sqlite store is read-only";
struct SqliteInner {
    conn: rusqlite::Connection,
    path: PathBuf,
    /// Whether an explicit `BEGIN` transaction is currently open (the store's memory; the
    /// connection's truth is `conn.is_autocommit()`, reconciled at every boundary).
    write_tx_open: bool,
    /// v18 item 4 (C1): nesting depth of settle passes; the 0→1 begin opens the pass's read
    /// transaction, the 1→0 end closes it if nothing was written.
    pass_depth: u32,
    /// v18 item 4: opened through `open_read_only` — no pass transaction, every write
    /// refused before `BEGIN`, no COMMIT/checkpoint at the barrier or at close.
    read_only: bool,
    /// The store-side latch: the first `LostWrites` logs `error!`, later ones `trace!`.
    lost_writes_reported: bool,
    /// A failed pass `BEGIN` is logged once per store (and counted every time).
    begin_failure_reported: bool,
    /// v18 item 8: whether the first checkpoint failure has been logged. A store whose
    /// checkpoints fail fails them on every barrier; the counter carries the rate, this keeps
    /// the log to one line.
    checkpoint_failure_reported: bool,
    tx: TxCells,
    ensured_raw_table_headers: HashSet<String>,
    visible_row_table_locators: HashMap<(String, ObjectId), super::ExactRowTableLocator>,
}

impl SqliteInner {
    /// v18 item 4 (design v7 § B1): the store's memory of its transaction against the
    /// connection's truth. `write_tx_open && conn.is_autocommit()` means SQLite ended the
    /// transaction behind the store's back (a full rollback on NOMEM/IOERR/INTERRUPT/FULL):
    /// with landed writes in it that is a loss, reported on every boundary until the store
    /// is reopened (strict, design v8 SF6 — the alternative, unconfirmed savepoints after
    /// a loss, hides it); with none it is a clean slate and the next write begins afresh.
    /// First statement of every transaction boundary AFTER the pass-depth bookkeeping, which
    /// reads no transaction flag (diff r21 SF5); nothing reads the transaction flags before it.
    fn reconcile_tx_state(&mut self) -> Result<(), StorageError> {
        if self.write_tx_open && self.conn.is_autocommit() {
            if self.tx.dirty.get() {
                let detail = "the write transaction was ended behind the store's back with \
                              landed writes in it (SQLite full rollback); reopen the store"
                    .to_string();
                if !self.lost_writes_reported {
                    self.lost_writes_reported = true;
                    tracing::error!(path = %self.path.display(), "sqlite lost writes: {detail}");
                } else {
                    // `trace`, not `debug` (diff r21 SF4): a dead store reconciles on both
                    // pass boundaries of every tick, so this arm repeats forever. The loss
                    // was logged once at `error!` above and the carrier holds it; v18 item 8
                    // is a campaign to keep `debug` readable, and this would undo it.
                    tracing::trace!(path = %self.path.display(), "sqlite lost writes: {detail}");
                }
                return Err(StorageError::LostWrites { detail });
            }
            self.write_tx_open = false;
        }
        Ok(())
    }

    /// Start a write transaction if one isn't already open. A read-only store refuses
    /// here — the one gate every write path enters before any statement.
    fn ensure_write_tx(&mut self) -> Result<(), StorageError> {
        if self.read_only {
            return Err(StorageError::IoError(READ_ONLY_WRITE.to_string()));
        }
        self.reconcile_tx_state()?;
        if !self.write_tx_open {
            self.conn
                .execute_batch("BEGIN")
                .map_err(|e| sqlite_error(&self.tx.last_sqlite_error, "sqlite begin", e))?;
            self.write_tx_open = true;
            self.tx.dirty.set(false);
        }
        Ok(())
    }

    /// Commit the open write transaction, if any. A successful COMMIT clears `dirty`.
    fn commit_write_tx(&mut self) -> Result<(), StorageError> {
        self.reconcile_tx_state()?;
        if self.write_tx_open {
            self.conn
                .execute_batch("COMMIT")
                .map_err(|e| sqlite_error(&self.tx.last_sqlite_error, "sqlite commit", e))?;
            self.write_tx_open = false;
            self.tx.dirty.set(false);
        }
        Ok(())
    }

    /// v18 item 4 (C1, design v11): `pass_depth += 1` FIRST, so the matching end stays
    /// balanced whatever happens below; reconcile SECOND — which is why the caller is
    /// `with_pass_boundary` and not `with_inner_mut` (diff r20 B1: an entry reconcile
    /// would return `Err(LostWrites)` before this line and pin the depth); on 0→1 open the
    /// pass's read
    /// transaction, unless one is already open (a foreign transaction — a test hook or a
    /// write transaction left for the barrier — must not make `BEGIN` fail on every tick).
    /// A `BEGIN` that fails is counted and logged once; the pass runs in autocommit.
    fn begin_pass(
        &mut self,
        begin_failures: &std::sync::atomic::AtomicU64,
    ) -> Result<(), StorageError> {
        if self.read_only {
            return Ok(());
        }
        self.pass_depth += 1;
        self.reconcile_tx_state()?;
        if self.pass_depth == 1 && !self.write_tx_open && self.conn.is_autocommit() {
            match self.conn.execute_batch("BEGIN DEFERRED") {
                Ok(()) => {
                    self.write_tx_open = true;
                    self.tx.dirty.set(false);
                }
                Err(e) => {
                    begin_failures.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    crate::query_manager::settle_cost::bump(
                        &crate::query_manager::settle_cost::READ_PASS_BEGIN_FAILURES,
                    );
                    let error = sqlite_error(&self.tx.last_sqlite_error, "sqlite pass begin", e);
                    if !self.begin_failure_reported {
                        self.begin_failure_reported = true;
                        tracing::warn!(
                            path = %self.path.display(),
                            "read pass could not begin a transaction; reads run autocommit: {error}"
                        );
                    }
                }
            }
        }
        Ok(())
    }

    /// v18 item 4 (design v10 SF5, v11): the depth comes down first, then reconcile (an
    /// `Err(LostWrites)` propagates with the depth already balanced — which only holds
    /// because `with_pass_boundary` does not reconcile at entry, diff r20 B1), `wrote` is
    /// the flag at EVERY depth, and only the 1→0 end of a clean transaction commits — a
    /// dirty one is the barrier's; a nested end commits nothing.
    fn end_pass(&mut self) -> Result<super::PassOutcome, StorageError> {
        if self.read_only {
            return Ok(super::PassOutcome { wrote: false });
        }
        debug_assert!(self.pass_depth > 0, "end_read_pass without a begin");
        self.pass_depth = self.pass_depth.saturating_sub(1);
        self.reconcile_tx_state()?;
        let wrote = self.tx.dirty.get();
        if self.pass_depth == 0 && self.write_tx_open && !wrote {
            self.commit_write_tx()?;
        }
        Ok(super::PassOutcome { wrote })
    }
}

pub struct SqliteStorage {
    cache_namespace: usize,
    inner: Mutex<Option<SqliteInner>>,
    /// Read statements this instance ran, and how many of them ran outside any transaction
    /// (SQLite's autocommit mode: a read transaction begun and ended per statement).
    read_statements: std::sync::atomic::AtomicU64,
    autocommit_read_statements: std::sync::atomic::AtomicU64,
    /// v18 item 8: explicit `PRAGMA wal_checkpoint(PASSIVE)` runs from `flush_wal`.
    checkpoints: std::sync::atomic::AtomicU64,
    /// v18 item 8: checkpoints that ran and drained nothing because a reader pinned the WAL.
    /// SQLite calls those a success (`sqlite3.c:67453-67457`), so they must be told apart by
    /// the frame columns or not at all.
    checkpoints_blocked: std::sync::atomic::AtomicU64,
    /// v18 item 8: the least time between two explicit checkpoints from `flush_wal`.
    /// Defaults to [`SqliteStorage::DEFAULT_CHECKPOINT_INTERVAL`] on every open, which is the
    /// whole of the fix — the first version of this item defaulted to `None` and offered a
    /// setter no shipping build called (diff r25 B1). `None` is kept, and still means "every
    /// barrier", but nothing in production asks for it. SQLite's own autocheckpoint bounds the
    /// WAL by size independently of this.
    checkpoint_interval: std::sync::Mutex<Option<std::time::Duration>>,
    /// v18 item 8: when the last explicit checkpoint ran (monotonic; the open counts as one).
    last_checkpoint: std::sync::Mutex<std::time::Instant>,
    /// v18 item 8: explicit checkpoints that returned an error (best-effort; the barrier is
    /// the COMMIT, a failed checkpoint is logged and counted, never a barrier failure).
    checkpoint_failures: std::sync::atomic::AtomicU64,
    /// v18 item 5: ladder walks this store served (the process-global is on the settle line).
    visible_ladder_recoveries: std::sync::atomic::AtomicU64,
    /// v18 item 5: recovered locators this store could not persist (counted, never a read
    /// failure).
    recovery_persist_failures: std::sync::atomic::AtomicU64,
    /// v18 item 4: first writes retried once after `SQLITE_BUSY_SNAPSHOT` on a clean pass
    /// transaction.
    busy_snapshot_retries: std::sync::atomic::AtomicU64,
    /// v18 item 4: pass `BEGIN`s that failed (the pass ran autocommit).
    read_pass_begin_failures: std::sync::atomic::AtomicU64,
}

impl SqliteStorage {
    fn store_has_any_rows(
        conn: &rusqlite::Connection,
        last: &Cell<Option<i32>>,
    ) -> Result<bool, StorageError> {
        conn.query_row("SELECT EXISTS(SELECT 1 FROM kv LIMIT 1)", [], |row| {
            row.get::<_, i64>(0)
        })
        .map(|exists| exists != 0)
        .map_err(|e| sqlite_error(last, "sqlite inspect store contents", e))
    }

    fn ensure_store_manifest(
        conn: &rusqlite::Connection,
        last: &Cell<Option<i32>>,
    ) -> Result<(), StorageError> {
        let expected = super::expected_store_manifest(super::SQLITE_STORE_KIND);
        let existing = conn
            .query_row(
                "SELECT value FROM kv WHERE key = ?1",
                rusqlite::params![super::STORE_MANIFEST_KEY.as_bytes()],
                |row| row.get::<_, Vec<u8>>(0),
            )
            .optional()
            .map_err(|e| sqlite_error(last, "sqlite read store manifest", e))?;
        match existing {
            Some(bytes) => {
                let actual = super::decode_store_manifest(&bytes)?;
                super::validate_store_manifest(&actual, &expected)
            }
            None => {
                if Self::store_has_any_rows(conn, last)? {
                    return Err(StorageError::IoError(
                        "missing store manifest for non-empty sqlite store".to_string(),
                    ));
                }
                let bytes = super::encode_store_manifest(&expected)?;
                conn.execute(
                    "INSERT INTO kv(key, value) VALUES (?1, ?2)",
                    rusqlite::params![super::STORE_MANIFEST_KEY.as_bytes(), bytes],
                )
                .map_err(|e| sqlite_error(last, "sqlite write store manifest", e))?;
                Ok(())
            }
        }
    }

    /// Compute the lexicographic successor of `prefix` for use as an
    /// exclusive upper bound. Same logic as RocksDB's `prefix_upper_bound`.
    fn prefix_upper_bound(prefix: &[u8]) -> Option<Vec<u8>> {
        let mut bound = prefix.to_vec();
        while let Some(last) = bound.last_mut() {
            if *last < 0xFF {
                *last += 1;
                return Some(bound);
            }
            bound.pop();
        }
        None
    }

    /// v18 item 8: how often `flush_wal` may run an explicit PASSIVE checkpoint. `None`
    /// checkpoints on every barrier. Durability does not depend on it: every barrier still
    /// commits, and a committed WAL transaction survives a process kill.
    pub fn set_checkpoint_interval(&self, interval: Option<std::time::Duration>) {
        *self
            .checkpoint_interval
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = interval;
    }
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StorageError> {
        let path = path.as_ref();
        let tx = TxCells::new();
        let conn = rusqlite::Connection::open(path)
            .map_err(|e| sqlite_error(&tx.last_sqlite_error, "sqlite open", e))?;
        conn.execute_batch(
            // `wal_autocheckpoint = 1000` restates SQLite's own compiled default
            // (`SQLITE_DEFAULT_WAL_AUTOCHECKPOINT`, `sqlite3.c:14172`) and so changes NOTHING at
            // runtime — diff r25 SF3, and r21 measured both halves green. It is here because
            // v18 item 8 defers the explicit checkpoint and this is the arm that bounds the WAL
            // in between: stating it makes the bound a decision in our code that a chain row can
            // disarm, instead of a vendored constant no gate covers.
            "PRAGMA journal_mode = WAL;
             PRAGMA wal_autocheckpoint = 1000;
             PRAGMA synchronous = NORMAL;
             PRAGMA cache_size = -65536;
             PRAGMA busy_timeout = 5000;
             PRAGMA foreign_keys = OFF;
             CREATE TABLE IF NOT EXISTS kv (
                 key   BLOB PRIMARY KEY,
                 value BLOB NOT NULL
             ) WITHOUT ROWID;",
        )
        .map_err(|e| sqlite_error(&tx.last_sqlite_error, "sqlite init", e))?;
        Self::ensure_store_manifest(&conn, &tx.last_sqlite_error)?;
        Ok(Self::over(conn, tx, path, false))
    }

    /// v18 item 8: how long a store waits between explicit WAL checkpoints.
    ///
    /// Arbitrary, within a region where the cost curve is flat: what item 8 removes is a checkpoint
    /// PER BARRIER, and at any real barrier rate an interval of a second already removes almost all
    /// of them, so 5 s, 30 s and 60 s are indistinguishable on cost. It is not derived from load
    /// because there is nothing to derive it from — this number does not bound the WAL. The size
    /// arm does (`PRAGMA wal_autocheckpoint`, set in `open`), and a derived interval would be the
    /// one thing no chain row could falsify.
    ///
    /// Note it is an `Instant` interval: 30 s of UPTIME, not of wall clock. A backgrounded iOS app
    /// resumes with the interval not elapsed. Bounded by the size arm and by `close`, which
    /// checkpoints unconditionally, so this is a property to know rather than a defect.
    const DEFAULT_CHECKPOINT_INTERVAL: std::time::Duration = std::time::Duration::from_secs(30);

    /// v18 item 4: the constructor for tools and reference readers (`SQLITE_OPEN_READ_ONLY`).
    /// No pass transaction is ever opened on it, every write is refused before `BEGIN`
    /// (the D2 recovery hook is a no-op, so a tool pointed at a live store never becomes
    /// its writer or pins its WAL), and neither the barrier nor `close()` commits or
    /// checkpoints. A read-only open of an EMPTY store fails at the manifest insert.
    pub fn open_read_only(path: impl AsRef<Path>) -> Result<Self, StorageError> {
        let path = path.as_ref();
        let tx = TxCells::new();
        let flags = rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY
            | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX
            | rusqlite::OpenFlags::SQLITE_OPEN_URI;
        let conn = rusqlite::Connection::open_with_flags(path, flags)
            .map_err(|e| sqlite_error(&tx.last_sqlite_error, "sqlite open read-only", e))?;
        conn.execute_batch("PRAGMA busy_timeout = 5000;")
            .map_err(|e| sqlite_error(&tx.last_sqlite_error, "sqlite init read-only", e))?;
        Self::ensure_store_manifest(&conn, &tx.last_sqlite_error)?;
        Ok(Self::over(conn, tx, path, true))
    }
    fn over(conn: rusqlite::Connection, tx: TxCells, path: &Path, read_only: bool) -> Self {
        Self {
            cache_namespace: super::next_storage_cache_namespace(),
            read_statements: std::sync::atomic::AtomicU64::new(0),
            autocommit_read_statements: std::sync::atomic::AtomicU64::new(0),
            checkpoints: std::sync::atomic::AtomicU64::new(0),
            checkpoints_blocked: std::sync::atomic::AtomicU64::new(0),
            // v18 item 8, diff r25 B1: the DEFAULT is the policy. The first version of this
            // item left it `None` here and offered `set_checkpoint_interval` — which six gates
            // called and nothing else did, so every shipping build kept checkpointing on every
            // barrier and the item changed nothing. A knob nobody turns is not a fix.
            checkpoint_interval: std::sync::Mutex::new(Some(Self::DEFAULT_CHECKPOINT_INTERVAL)),
            last_checkpoint: std::sync::Mutex::new(std::time::Instant::now()),
            checkpoint_failures: std::sync::atomic::AtomicU64::new(0),
            visible_ladder_recoveries: std::sync::atomic::AtomicU64::new(0),
            recovery_persist_failures: std::sync::atomic::AtomicU64::new(0),
            busy_snapshot_retries: std::sync::atomic::AtomicU64::new(0),
            read_pass_begin_failures: std::sync::atomic::AtomicU64::new(0),
            inner: Mutex::new(Some(SqliteInner {
                conn,
                path: path.to_path_buf(),
                write_tx_open: false,
                pass_depth: 0,
                read_only,
                lost_writes_reported: false,
                begin_failure_reported: false,
                checkpoint_failure_reported: false,
                tx,
                ensured_raw_table_headers: HashSet::new(),
                visible_row_table_locators: HashMap::new(),
            })),
        }
    }

    /// One read statement is about to run on `inner`'s connection.
    fn note_read(&self, inner: &SqliteInner) {
        use std::sync::atomic::Ordering::Relaxed;
        self.read_statements.fetch_add(1, Relaxed);
        if inner.conn.is_autocommit() {
            self.autocommit_read_statements.fetch_add(1, Relaxed);
            crate::query_manager::settle_cost::bump(
                &crate::query_manager::settle_cost::AUTOCOMMIT_READS,
            );
        }
    }

    /// Checkpoints this instance ran since it was opened.
    #[cfg(any(test, feature = "test"))]
    pub fn checkpoints_for_test(&self) -> u64 {
        self.checkpoints.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// v18 item 8: checkpoints that ran and drained nothing (a reader pinned the WAL).
    #[cfg(any(test, feature = "test"))]
    pub fn checkpoints_blocked_for_test(&self) -> u64 {
        self.checkpoints_blocked
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    #[cfg(any(test, feature = "test"))]
    pub fn checkpoint_failures_for_test(&self) -> u64 {
        self.checkpoint_failures
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Move the last explicit checkpoint into the past, so a gate can cross the interval
    /// without sleeping.
    #[cfg(any(test, feature = "test"))]
    pub fn age_last_checkpoint_for_test(&self, by: std::time::Duration) {
        let mut last = self
            .last_checkpoint
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        *last = last
            .checked_sub(by)
            .expect("the process has run long enough to age the checkpoint stamp");
    }

    /// SQLite's own WAL size trigger (`PRAGMA wal_autocheckpoint`, in pages; 0 disables
    /// it). Independent of the explicit interval; the two arms together bound the WAL.
    pub fn set_wal_autocheckpoint_pages(&self, pages: u32) -> Result<(), StorageError> {
        let inner = self.lock_inner()?;
        let Some(inner) = inner.as_ref() else {
            return Err(StorageError::IoError("sqlite storage closed".into()));
        };
        inner
            .conn
            .execute_batch(&format!("PRAGMA wal_autocheckpoint = {pages};"))
            .map_err(|e| sqlite_error(&inner.tx.last_sqlite_error, "sqlite wal_autocheckpoint", e))
    }

    /// Open a read transaction on the store's own connection and take a snapshot, the
    /// state in which a checkpoint on this connection returns SQLITE_LOCKED.
    #[cfg(any(test, feature = "test"))]
    pub fn hold_read_transaction_for_test(&self) -> Result<(), StorageError> {
        let inner = self.lock_inner()?;
        let Some(inner) = inner.as_ref() else {
            return Err(StorageError::IoError("sqlite storage closed".into()));
        };
        inner
            .conn
            .execute_batch("BEGIN DEFERRED; SELECT count(*) FROM kv;")
            .map_err(|e| sqlite_error(&inner.tx.last_sqlite_error, "sqlite hold read tx", e))
    }

    #[cfg(any(test, feature = "test"))]
    pub fn release_read_transaction_for_test(&self) -> Result<(), StorageError> {
        let inner = self.lock_inner()?;
        let Some(inner) = inner.as_ref() else {
            return Err(StorageError::IoError("sqlite storage closed".into()));
        };
        inner
            .conn
            .execute_batch("COMMIT")
            .map_err(|e| sqlite_error(&inner.tx.last_sqlite_error, "sqlite release read tx", e))
    }
    /// Read statements this instance ran since it was opened.
    #[cfg(any(test, feature = "test"))]
    pub fn read_statements_for_test(&self) -> u64 {
        self.read_statements
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Of those, the ones that ran in autocommit mode — each its own SQLite transaction.
    #[cfg(any(test, feature = "test"))]
    pub fn autocommit_read_statements_for_test(&self) -> u64 {
        self.autocommit_read_statements
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Whether the connection is outside any transaction right now.
    #[cfg(any(test, feature = "test"))]
    pub fn is_autocommit_for_test(&self) -> bool {
        self.with_inner(|inner| Ok(inner.conn.is_autocommit()))
            .expect("sqlite storage open")
    }

    /// v18 item 4: the store's `dirty` cell (a landed write in the current transaction).
    #[cfg(any(test, feature = "test"))]
    pub fn tx_dirty_for_test(&self) -> bool {
        self.with_inner(|inner| Ok(inner.tx.dirty.get()))
            .expect("sqlite storage open")
    }

    /// v18 item 4: the store's memory of an open explicit transaction (`write_tx_open`),
    /// distinct from the connection's `is_autocommit_for_test`.
    #[cfg(any(test, feature = "test"))]
    pub fn transaction_open_for_test(&self) -> bool {
        self.with_inner(|inner| Ok(inner.write_tx_open))
            .expect("sqlite storage open")
    }

    #[cfg(any(test, feature = "test"))]
    pub fn lost_writes_reported_for_test(&self) -> bool {
        self.with_inner(|inner| Ok(inner.lost_writes_reported))
            .expect("sqlite storage open")
    }

    /// The extended result code of the last SQLite failure mapped by this store.
    #[cfg(any(test, feature = "test"))]
    pub fn last_sqlite_error_for_test(&self) -> Option<i32> {
        self.with_inner(|inner| Ok(inner.tx.last_sqlite_error.get()))
            .expect("sqlite storage open")
    }

    #[cfg(any(test, feature = "test"))]
    pub fn busy_snapshot_retries_for_test(&self) -> u64 {
        self.busy_snapshot_retries
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    #[cfg(any(test, feature = "test"))]
    pub fn pass_depth_for_test(&self) -> u32 {
        self.with_inner(|inner| Ok(inner.pass_depth))
            .expect("sqlite storage open")
    }

    /// v18 item 5: ladder walks this store served (per store, unlike the process-global).
    #[cfg(any(test, feature = "test"))]
    pub fn visible_ladder_recoveries_for_test(&self) -> u64 {
        self.visible_ladder_recoveries
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    #[cfg(any(test, feature = "test"))]
    pub fn recovery_persist_failures_for_test(&self) -> u64 {
        self.recovery_persist_failures
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    #[cfg(any(test, feature = "test"))]
    pub fn read_pass_begin_failures_for_test(&self) -> u64 {
        self.read_pass_begin_failures
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// A bare `ROLLBACK` on the connection with the store's flags untouched — the state
    /// SQLite's own full rollback leaves behind the store's back.
    /// (`release_read_transaction_for_test`
    /// is a COMMIT and would persist the rows.)
    #[cfg(any(test, feature = "test"))]
    pub fn roll_back_transaction_for_test(&self) -> Result<(), StorageError> {
        let inner = self.lock_inner()?;
        let Some(inner) = inner.as_ref() else {
            return Err(StorageError::IoError("sqlite storage closed".into()));
        };
        inner
            .conn
            .execute_batch("ROLLBACK")
            .map_err(|e| sqlite_error(&inner.tx.last_sqlite_error, "sqlite rollback (test)", e))
    }

    /// Arm the store: after the next LANDED write (inside `with_savepoint`, after RELEASE
    /// and after `dirty` is marked) the connection is rolled back and that call returns
    /// `Err(LostWrites)`. An armed action, not an injected failure — see `TxCells`.
    #[cfg(any(test, feature = "test"))]
    pub fn roll_back_after_next_write_for_test(&self) {
        self.with_inner(|inner| {
            inner.tx.roll_back_after_next_write.set(true);
            Ok(())
        })
        .expect("sqlite storage open")
    }

    /// `PRAGMA max_page_count` on the store's own connection (per-connection, not
    /// persistent): the way to provoke `SQLITE_FULL` without a full disk. Clamping below
    /// the current page count clamps TO it.
    #[cfg(any(test, feature = "test"))]
    pub fn set_max_page_count_for_test(&self, pages: u32) -> Result<u32, StorageError> {
        let inner = self.lock_inner()?;
        let Some(inner) = inner.as_ref() else {
            return Err(StorageError::IoError("sqlite storage closed".into()));
        };
        inner
            .conn
            .query_row(&format!("PRAGMA max_page_count = {pages}"), [], |row| {
                row.get::<_, u32>(0)
            })
            .map_err(|e| sqlite_error(&inner.tx.last_sqlite_error, "sqlite max_page_count", e))
    }

    fn lock_inner(&self) -> Result<MutexGuard<'_, Option<SqliteInner>>, StorageError> {
        self.inner
            .lock()
            .map_err(|_| StorageError::IoError("sqlite storage mutex poisoned".to_string()))
    }

    fn with_inner<T>(
        &self,
        f: impl FnOnce(&SqliteInner) -> Result<T, StorageError>,
    ) -> Result<T, StorageError> {
        let inner = self.lock_inner()?;
        let inner = inner
            .as_ref()
            .ok_or_else(|| StorageError::IoError("sqlite storage already closed".to_string()))?;
        f(inner)
    }

    /// The pass boundaries only. Takes the lock WITHOUT `with_inner_mut`'s entry
    /// reconcile (diff r20 B1): a boundary must be ENTERED even on a store that has lost
    /// writes, because the boundary's own bookkeeping runs before its reconcile — the
    /// depth comes down first, then the loss is reported. Routed through `with_inner_mut`,
    /// a loss arriving between a successful begin and its end pre-empts `end_pass`, and
    /// `pass_depth` stays pinned at 1 for the life of the store (G-C11's fixture exactly).
    /// No BUSY_SNAPSHOT retry here: neither boundary writes a row.
    fn with_pass_boundary<T>(
        &self,
        f: impl FnOnce(&mut SqliteInner) -> Result<T, StorageError>,
    ) -> Result<T, StorageError> {
        let mut inner = self.lock_inner()?;
        let inner = inner
            .as_mut()
            .ok_or_else(|| StorageError::IoError("sqlite storage already closed".to_string()))?;
        inner.tx.last_sqlite_error.set(None);
        f(inner)
    }

    /// Every write path. Reconciles first (design v8 § B2: no flag is read before it), then
    /// runs `f`; a first write on a CLEAN pass transaction that finds another connection has
    /// committed fails at once with `SQLITE_BUSY_SNAPSHOT` (`walBeginWriteTransaction`; the
    /// busy handler is not consulted once a read snapshot is held) — `ROLLBACK`,
    /// `BEGIN DEFERRED`, and `f` runs once more. Nothing is lost because nothing was written;
    /// a dirty transaction already holds the write lock and cannot hit it. The retry ends the
    /// pass's read snapshot: reads served before it came from the earlier snapshot (design v8
    /// SF6 — the one C1 qualification G-C1 cannot see).
    ///
    /// NOT the pass boundaries: they take `with_pass_boundary`, and routing them back here
    /// pins `pass_depth` on the loss path (diff r20 B1, chain row E3).
    fn with_inner_mut<T>(
        &self,
        mut f: impl FnMut(&mut SqliteInner) -> Result<T, StorageError>,
    ) -> Result<T, StorageError> {
        let mut inner = self.lock_inner()?;
        let inner = inner
            .as_mut()
            .ok_or_else(|| StorageError::IoError("sqlite storage already closed".to_string()))?;
        inner.tx.last_sqlite_error.set(None);
        inner.reconcile_tx_state()?;
        let clean = inner.write_tx_open && !inner.tx.dirty.get();
        match f(inner) {
            Err(error)
                if clean
                    && inner.tx.last_sqlite_error.get()
                        == Some(rusqlite::ffi::SQLITE_BUSY_SNAPSHOT) =>
            {
                tracing::debug!(
                    "sqlite first write hit BUSY_SNAPSHOT on a clean pass transaction; retrying once: {error}"
                );
                // diff r20 SF9: a failing ROLLBACK/BEGIN here must not REPLACE the
                // BUSY_SNAPSHOT the caller needs to see. Log the secondary failure, return
                // the original. `write_tx_open` is left as it is, and both outcomes are safe
                // (diff r21 SF6): if the ROLLBACK took the transaction down, the next reconcile
                // finds autocommit with a CLEAN transaction and clears the flag; if it failed
                // with the transaction still alive, the reconcile finds `!is_autocommit()` and
                // correctly leaves the flag standing. `dirty` is false either way — the 517 can
                // only come from the pass transaction's FIRST write statement.
                if let Err(e) = inner.conn.execute_batch("ROLLBACK") {
                    let secondary =
                        sqlite_error(&inner.tx.last_sqlite_error, "sqlite snapshot rollback", e);
                    tracing::warn!("rollback after BUSY_SNAPSHOT failed: {secondary}");
                    return Err(error);
                }
                if let Err(e) = inner.conn.execute_batch("BEGIN DEFERRED") {
                    let secondary =
                        sqlite_error(&inner.tx.last_sqlite_error, "sqlite snapshot begin", e);
                    tracing::warn!("re-begin after BUSY_SNAPSHOT failed: {secondary}");
                    return Err(error);
                }
                inner.tx.dirty.set(false);
                inner.tx.last_sqlite_error.set(None);
                self.busy_snapshot_retries
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                f(inner)
            }
            other => other,
        }
    }

    /// Run `f` inside a SQLite SAVEPOINT. Releases on success, rolls back on error. Reads
    /// within `f` see uncommitted savepoint writes because all operations share the same
    /// connection.
    ///
    /// v18 item 4 (design v9 § B1/B2, v14, v16): `dirty` is marked from `total_changes()`
    /// BEFORE the RELEASE attempt (RELEASE only merges the savepoint and can itself fail
    /// with the body's statements already in the transaction); on the error path the
    /// flag is set only if `ROLLBACK TO` failed with the transaction still alive — after
    /// SQLite's own full rollback (NOMEM/IOERR/INTERRUPT/FULL) autocommit is back on,
    /// nothing partial remains, and `ROLLBACK TO` necessarily fails with "no such
    /// savepoint". Invariant for every body: no DDL and no PRAGMA inside it (the
    /// `total_changes` rule counts rows). A panic inside `f` leaves `jazz_sp` open — moot:
    /// no host runs another pass over a poisoned core. The armed test hook fires after a
    /// landed write; see `TxCells`.
    fn with_savepoint<T>(
        conn: &rusqlite::Connection,
        tx: &TxCells,
        f: impl FnOnce() -> Result<T, StorageError>,
    ) -> Result<T, StorageError> {
        let last = &tx.last_sqlite_error;
        conn.execute("SAVEPOINT jazz_sp", [])
            .map_err(|e| sqlite_error(last, "savepoint start", e))?;
        let changes_before = conn.total_changes();
        match f() {
            Ok(value) => {
                let landed = conn.total_changes() != changes_before;
                if landed {
                    tx.dirty.set(true);
                }
                conn.execute("RELEASE jazz_sp", [])
                    .map_err(|e| sqlite_error(last, "savepoint release", e))?;
                if landed && tx.roll_back_after_next_write.replace(false) {
                    return match conn.execute_batch("ROLLBACK") {
                        Ok(()) => Err(StorageError::LostWrites {
                            detail: "rolled back after the next write (test hook)".to_string(),
                        }),
                        Err(e) => Err(sqlite_error(
                            last,
                            "rollback after the next write (test hook)",
                            e,
                        )),
                    };
                }
                Ok(value)
            }
            Err(error) => {
                let rolled_back = conn.execute("ROLLBACK TO jazz_sp", []).is_ok();
                if !rolled_back && !conn.is_autocommit() {
                    tx.dirty.set(true);
                }
                let _ = conn.execute("RELEASE jazz_sp", []);
                Err(error)
            }
        }
    }

    fn get(
        conn: &rusqlite::Connection,
        tx: &TxCells,
        key: &str,
    ) -> Result<Option<Vec<u8>>, StorageError> {
        let last = &tx.last_sqlite_error;
        let mut stmt = conn
            .prepare_cached("SELECT value FROM kv WHERE key = ?1")
            .map_err(|e| sqlite_error(last, "sqlite prepare get", e))?;
        match stmt.query_row(rusqlite::params![key.as_bytes()], |row| {
            row.get::<_, Vec<u8>>(0)
        }) {
            Ok(v) => Ok(Some(v)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(sqlite_error(last, "sqlite get", e)),
        }
    }

    fn scan_prefix(
        conn: &rusqlite::Connection,
        tx: &TxCells,
        prefix: &str,
    ) -> Result<Vec<(String, Vec<u8>)>, StorageError> {
        let last = &tx.last_sqlite_error;
        let prefix_bytes = prefix.as_bytes();
        let upper = Self::prefix_upper_bound(prefix_bytes)
            .ok_or_else(|| StorageError::IoError("prefix upper bound overflow".to_string()))?;
        let mut stmt = conn
            .prepare_cached("SELECT key, value FROM kv WHERE key >= ?1 AND key < ?2 ORDER BY key")
            .map_err(|e| sqlite_error(last, "sqlite prepare scan_prefix", e))?;
        let rows = stmt
            .query_map(rusqlite::params![prefix_bytes, upper.as_slice()], |row| {
                Ok((row.get::<_, Vec<u8>>(0)?, row.get::<_, Vec<u8>>(1)?))
            })
            .map_err(|e| sqlite_error(last, "sqlite scan_prefix", e))?;
        let mut out = Vec::new();
        for row in rows {
            let (key_bytes, value) =
                row.map_err(|e| sqlite_error(last, "sqlite scan_prefix row", e))?;
            let key = String::from_utf8(key_bytes)
                .map_err(|e| StorageError::IoError(format!("sqlite key utf8: {e}")))?;
            out.push((key, value));
        }
        Ok(out)
    }

    fn scan_prefix_keys(
        conn: &rusqlite::Connection,
        tx: &TxCells,
        prefix: &str,
    ) -> Result<Vec<String>, StorageError> {
        let last = &tx.last_sqlite_error;
        let prefix_bytes = prefix.as_bytes();
        let upper = Self::prefix_upper_bound(prefix_bytes)
            .ok_or_else(|| StorageError::IoError("prefix upper bound overflow".to_string()))?;
        let mut stmt = conn
            .prepare_cached("SELECT key FROM kv WHERE key >= ?1 AND key < ?2 ORDER BY key")
            .map_err(|e| sqlite_error(last, "sqlite prepare scan_prefix_keys", e))?;
        let rows = stmt
            .query_map(rusqlite::params![prefix_bytes, upper.as_slice()], |row| {
                row.get::<_, Vec<u8>>(0)
            })
            .map_err(|e| sqlite_error(last, "sqlite scan_prefix_keys", e))?;
        let mut out = Vec::new();
        for row in rows {
            let key_bytes =
                row.map_err(|e| sqlite_error(last, "sqlite scan_prefix_keys row", e))?;
            let key = String::from_utf8(key_bytes)
                .map_err(|e| StorageError::IoError(format!("sqlite key utf8: {e}")))?;
            out.push(key);
        }
        Ok(out)
    }

    fn scan_range(
        conn: &rusqlite::Connection,
        tx: &TxCells,
        start: &str,
        end: &str,
    ) -> Result<Vec<(String, Vec<u8>)>, StorageError> {
        let last = &tx.last_sqlite_error;
        let mut stmt = conn
            .prepare_cached("SELECT key, value FROM kv WHERE key >= ?1 AND key < ?2 ORDER BY key")
            .map_err(|e| sqlite_error(last, "sqlite prepare scan_range", e))?;
        let rows = stmt
            .query_map(rusqlite::params![start.as_bytes(), end.as_bytes()], |row| {
                Ok((row.get::<_, Vec<u8>>(0)?, row.get::<_, Vec<u8>>(1)?))
            })
            .map_err(|e| sqlite_error(last, "sqlite scan_range", e))?;
        let mut out = Vec::new();
        for row in rows {
            let (key_bytes, value) =
                row.map_err(|e| sqlite_error(last, "sqlite scan_range row", e))?;
            let key = String::from_utf8(key_bytes)
                .map_err(|e| StorageError::IoError(format!("sqlite key utf8: {e}")))?;
            out.push((key, value));
        }
        Ok(out)
    }

    fn scan_range_keys(
        conn: &rusqlite::Connection,
        tx: &TxCells,
        start: &str,
        end: &str,
    ) -> Result<Vec<String>, StorageError> {
        let last = &tx.last_sqlite_error;
        let mut stmt = conn
            .prepare_cached("SELECT key FROM kv WHERE key >= ?1 AND key < ?2 ORDER BY key")
            .map_err(|e| sqlite_error(last, "sqlite prepare scan_range_keys", e))?;
        let rows = stmt
            .query_map(rusqlite::params![start.as_bytes(), end.as_bytes()], |row| {
                row.get::<_, Vec<u8>>(0)
            })
            .map_err(|e| sqlite_error(last, "sqlite scan_range_keys", e))?;
        let mut out = Vec::new();
        for row in rows {
            let key_bytes = row.map_err(|e| sqlite_error(last, "sqlite scan_range_keys row", e))?;
            let key = String::from_utf8(key_bytes)
                .map_err(|e| StorageError::IoError(format!("sqlite key utf8: {e}")))?;
            out.push(key);
        }
        Ok(out)
    }

    fn set(
        conn: &rusqlite::Connection,
        tx: &TxCells,
        key: &str,
        value: &[u8],
    ) -> Result<(), StorageError> {
        let last = &tx.last_sqlite_error;
        conn.prepare_cached("INSERT OR REPLACE INTO kv (key, value) VALUES (?1, ?2)")
            .map_err(|e| sqlite_error(last, "sqlite prepare set", e))?
            .execute(rusqlite::params![key.as_bytes(), value])
            .map(|_| ())
            .map_err(|e| sqlite_error(last, "sqlite set", e))
    }

    fn delete(conn: &rusqlite::Connection, tx: &TxCells, key: &str) -> Result<(), StorageError> {
        let last = &tx.last_sqlite_error;
        conn.prepare_cached("DELETE FROM kv WHERE key = ?1")
            .map_err(|e| sqlite_error(last, "sqlite prepare delete", e))?
            .execute(rusqlite::params![key.as_bytes()])
            .map(|_| ())
            .map_err(|e| sqlite_error(last, "sqlite delete", e))
    }
}

/// Whether a failed WAL checkpoint is the kind that will pass on its own.
///
/// v18 item 8. The checkpoint is housekeeping that runs after the commit which already made the
/// writes durable, so NEITHER answer fails the barrier — the difference is entirely how loudly
/// it is reported, and that difference matters: `SQLITE_FULL` and the `SQLITE_IOERR_*` family
/// are a disk that is filling or dying, and the checkpoint is where that shows up FIRST, while
/// commits keep appending to a WAL nothing is draining.
///
/// `SQLITE_BUSY` is in the transient list for completeness and cannot actually arrive here:
/// `OP_Checkpoint` converts it into the result row's busy column rather than an error
/// (`sqlite3.c:101414-101418`). `SQLITE_LOCKED` is the one transient code genuinely reachable —
/// `sqlite3BtreeCheckpoint` returns it when this connection has a transaction open — and after
/// `commit_write_tx` that should not happen either, which is why it is worth counting when it
/// does.
fn checkpoint_error_is_transient(error: &rusqlite::Error) -> bool {
    match error.sqlite_error_code() {
        Some(rusqlite::ErrorCode::DatabaseBusy) | Some(rusqlite::ErrorCode::DatabaseLocked) => true,
        // Anything else — full, I/O, corrupt, read-only, and every code not yet invented — is
        // reported at `error!`. Defaulting the UNKNOWN case to "serious" is the safe direction:
        // a new code misreported as transient is silent, and this store has one log line to
        // spend.
        _ => false,
    }
}

impl Storage for SqliteStorage {
    fn storage_cache_namespace(&self) -> usize {
        self.cache_namespace
    }

    fn raw_table_put(&mut self, table: &str, key: &str, value: &[u8]) -> Result<(), StorageError> {
        crate::query_manager::settle_cost::add(
            &crate::query_manager::settle_cost::STORAGE_WRITE_BYTES,
            value.len() as u64,
        );
        crate::query_manager::settle_cost::timed(
            &crate::query_manager::settle_cost::STORAGE_WRITE_MICROS,
            || {
                self.with_inner_mut(|inner| {
                    inner.ensure_write_tx()?;
                    Self::with_savepoint(&inner.conn, &inner.tx, || {
                        raw_table_put_core(table, key, value, |storage_key, bytes| {
                            Self::set(&inner.conn, &inner.tx, storage_key, bytes)
                        })
                    })
                })
            },
        )
    }

    fn raw_table_delete(&mut self, table: &str, key: &str) -> Result<(), StorageError> {
        crate::query_manager::settle_cost::timed(
            &crate::query_manager::settle_cost::STORAGE_WRITE_MICROS,
            || {
                self.with_inner_mut(|inner| {
                    inner.ensure_write_tx()?;
                    Self::with_savepoint(&inner.conn, &inner.tx, || {
                        raw_table_delete_core(table, key, |storage_key| {
                            Self::delete(&inner.conn, &inner.tx, storage_key)
                        })
                    })
                })
            },
        )
    }

    fn apply_raw_table_mutations(
        &mut self,
        mutations: &[RawTableMutation<'_>],
    ) -> Result<(), StorageError> {
        crate::query_manager::settle_cost::add(
            &crate::query_manager::settle_cost::STORAGE_WRITE_BYTES,
            mutations
                .iter()
                .map(|mutation| match mutation {
                    RawTableMutation::Put { value, .. } => value.len() as u64,
                    RawTableMutation::Delete { .. } => 0,
                })
                .sum(),
        );
        crate::query_manager::settle_cost::timed(
            &crate::query_manager::settle_cost::STORAGE_WRITE_MICROS,
            || {
                self.with_inner_mut(|inner| {
                    inner.ensure_write_tx()?;
                    Self::with_savepoint(&inner.conn, &inner.tx, || {
                        for mutation in mutations {
                            match mutation {
                                RawTableMutation::Put { table, key, value } => {
                                    raw_table_put_core(table, key, value, |storage_key, bytes| {
                                        Self::set(&inner.conn, &inner.tx, storage_key, bytes)
                                    })?;
                                }
                                RawTableMutation::Delete { table, key } => {
                                    raw_table_delete_core(table, key, |storage_key| {
                                        Self::delete(&inner.conn, &inner.tx, storage_key)
                                    })?;
                                }
                            }
                        }
                        Ok(())
                    })
                })
            },
        )
    }

    fn raw_table_get(&self, table: &str, key: &str) -> Result<Option<Vec<u8>>, StorageError> {
        crate::query_manager::settle_cost::bump(
            &crate::query_manager::settle_cost::STORAGE_READ_OPS,
        );
        crate::query_manager::settle_cost::timed(
            &crate::query_manager::settle_cost::STORAGE_READ_MICROS,
            || {
                self.with_inner(|inner| {
                    self.note_read(inner);
                    raw_table_get_core(table, key, |storage_key| {
                        Self::get(&inner.conn, &inner.tx, storage_key)
                    })
                })
                .inspect(|found| {
                    let bytes = found.as_ref().map_or(0, |bytes| bytes.len() as u64);
                    crate::query_manager::settle_cost::add(
                        &crate::query_manager::settle_cost::STORAGE_READ_BYTES,
                        bytes,
                    );
                })
            },
        )
    }

    fn raw_table_scan_prefix(
        &self,
        table: &str,
        prefix: &str,
    ) -> Result<super::RawTableRows, StorageError> {
        crate::query_manager::settle_cost::bump(
            &crate::query_manager::settle_cost::STORAGE_READ_OPS,
        );
        crate::query_manager::settle_cost::timed(
            &crate::query_manager::settle_cost::STORAGE_READ_MICROS,
            || {
                self.with_inner(|inner| {
                    self.note_read(inner);
                    raw_table_scan_prefix_core(table, prefix, |storage_prefix| {
                        Self::scan_prefix(&inner.conn, &inner.tx, storage_prefix)
                    })
                })
                .inspect(|rows| {
                    let bytes: u64 = rows.iter().map(|(_, bytes)| bytes.len() as u64).sum();
                    crate::query_manager::settle_cost::add(
                        &crate::query_manager::settle_cost::STORAGE_READ_BYTES,
                        bytes,
                    );
                })
            },
        )
    }

    fn raw_table_scan_prefix_keys(
        &self,
        table: &str,
        prefix: &str,
    ) -> Result<super::RawTableKeys, StorageError> {
        crate::query_manager::settle_cost::bump(
            &crate::query_manager::settle_cost::STORAGE_READ_OPS,
        );
        crate::query_manager::settle_cost::timed(
            &crate::query_manager::settle_cost::STORAGE_READ_MICROS,
            || {
                self.with_inner(|inner| {
                    self.note_read(inner);
                    raw_table_scan_prefix_keys_core(table, prefix, |storage_prefix| {
                        Self::scan_prefix_keys(&inner.conn, &inner.tx, storage_prefix)
                    })
                })
            },
        )
    }

    fn raw_table_scan_range(
        &self,
        table: &str,
        start: Option<&str>,
        end: Option<&str>,
    ) -> Result<super::RawTableRows, StorageError> {
        self.with_inner(|inner| {
            self.note_read(inner);
            raw_table_scan_range_core(table, start, end, |start_key, end_key| {
                Self::scan_range(&inner.conn, &inner.tx, start_key, end_key)
            })
        })
    }

    fn raw_table_scan_range_keys(
        &self,
        table: &str,
        start: Option<&str>,
        end: Option<&str>,
    ) -> Result<super::RawTableKeys, StorageError> {
        self.with_inner(|inner| {
            self.note_read(inner);
            raw_table_scan_range_keys_core(table, start, end, |start_key, end_key| {
                Self::scan_range_keys(&inner.conn, &inner.tx, start_key, end_key)
            })
        })
    }

    fn append_history_region_row_bytes(
        &mut self,
        table: &str,
        rows: &[HistoryRowBytes<'_>],
    ) -> Result<(), StorageError> {
        self.with_inner_mut(|inner| {
            inner.ensure_write_tx()?;
            Self::with_savepoint(&inner.conn, &inner.tx, || {
                append_history_region_row_bytes_core(table, rows, |key, bytes| {
                    Self::set(&inner.conn, &inner.tx, key, bytes)
                })
            })
        })
    }

    fn upsert_visible_region_row_bytes(
        &mut self,
        table: &str,
        rows: &[VisibleRowBytes<'_>],
    ) -> Result<(), StorageError> {
        self.with_inner_mut(|inner| {
            inner.ensure_write_tx()?;
            Self::with_savepoint(&inner.conn, &inner.tx, || {
                upsert_visible_region_row_bytes_core(table, rows, |key, bytes| {
                    Self::set(&inner.conn, &inner.tx, key, bytes)
                })
            })
        })
    }

    /// The write path dedups locator persists against `visible_row_table_locators`
    /// (see `apply_encoded_row_mutation`), so a direct write to this pointer must
    /// invalidate that cache or the next write will see its own value already
    /// cached and skip a persist the store actually needs. Defect 27's realign
    /// and repair sweep both write it directly.
    fn put_visible_row_table_locator(
        &mut self,
        branch: &str,
        row_id: ObjectId,
        locator: Option<&super::ExactRowTableLocator>,
    ) -> Result<(), StorageError> {
        let cache_key = (branch.to_string(), row_id);
        self.with_inner_mut(|inner| {
            inner.visible_row_table_locators.remove(&cache_key);
            Ok(())
        })?;
        super::storage_trait::put_visible_row_table_locator_default(self, branch, row_id, locator)
    }

    fn delete_visible_region_row(
        &mut self,
        table: &str,
        branch: &str,
        row_id: ObjectId,
    ) -> Result<(), StorageError> {
        // Evict the cached locator (it is a pointer, not a head), then delete
        // from every family that MEASURABLY holds the row. The cache is no longer
        // what decides where to delete: it named one family, and reaching one
        // family is exactly how a row survives its own delete.
        let cache_key = (branch.to_string(), row_id);
        self.with_inner_mut(|inner| Ok(inner.visible_row_table_locators.remove(&cache_key)))?;
        super::retire_index_entries_for_extra_visible_heads(self, table, branch, row_id)?;
        let raw_tables = super::visible_row_raw_tables_holding(self, table, branch, row_id)?;
        self.with_inner_mut(|inner| {
            inner.ensure_write_tx()?;
            Self::with_savepoint(&inner.conn, &inner.tx, || {
                let key = super::key_codec::visible_row_raw_table_key(branch, row_id);
                for raw_table in &raw_tables {
                    raw_table_delete_core(raw_table.as_str(), &key, |storage_key| {
                        Self::delete(&inner.conn, &inner.tx, storage_key)
                    })?;
                }
                raw_table_delete_core(
                    super::VISIBLE_ROW_TABLE_LOCATOR_TABLE,
                    &super::visible_row_table_locator_key(branch, row_id),
                    |storage_key| Self::delete(&inner.conn, &inner.tx, storage_key),
                )
            })
        })?;
        Ok(())
    }

    fn apply_encoded_row_mutation(
        &mut self,
        table: &str,
        encoded_history_rows: &[OwnedHistoryRowBytes],
        encoded_visible_rows: &[OwnedVisibleRowBytes],
        index_mutations: &[IndexMutation<'_>],
    ) -> Result<(), StorageError> {
        self.with_inner_mut(|inner| {
            inner.ensure_write_tx()?;
            // v18 item 4 (design v4 § B2(a)): names ensured inside the savepoint body are
            // merged into the caches only after RELEASE succeeded — a `ROLLBACK TO` must
            // not leave the set claiming a header that was never written (a retry would
            // then land rows in a header-less family, silently, for the process lifetime).
            // The locator dedup map likewise learns a pointer only once it is in the
            // transaction. Both merges sit here, at the call site, on the `Ok` path only.
            let (ensured, locators) = Self::with_savepoint(&inner.conn, &inner.tx, || {
                let mut ensured: Vec<String> = Vec::new();
                let mut locators: Vec<((String, ObjectId), super::ExactRowTableLocator)> =
                    Vec::new();
                let mut seen_row_raw_tables = std::collections::HashSet::new();
                // diff r20 SF4: the header is ENCODED lazily. The tree's `&&` chain
                // short-circuited on the ensured set; a two-statement rewrite would pay
                // `row_raw_table_header` + `encode_raw_table_header` per distinct raw
                // table per call, on the write path, for headers ensured long ago.
                let mut ensure_header = |name: &str,
                                         header: &dyn Fn() -> Result<Vec<u8>, StorageError>|
                 -> Result<(), StorageError> {
                    if inner.ensured_raw_table_headers.contains(name)
                        || ensured.iter().any(|seen| seen == name)
                    {
                        return Ok(());
                    }
                    let header = header()?;
                    raw_table_put_core(
                        super::RAW_TABLE_HEADER_TABLE,
                        name,
                        &header,
                        |storage_key, bytes| Self::set(&inner.conn, &inner.tx, storage_key, bytes),
                    )?;
                    ensured.push(name.to_string());
                    Ok(())
                };
                for row in encoded_history_rows {
                    if seen_row_raw_tables.insert(row.row_raw_table.clone()) {
                        ensure_header(row.row_raw_table.as_str(), &|| {
                            super::encode_raw_table_header(&super::row_raw_table_header(
                                &row.row_raw_table_id,
                                &row.user_descriptor,
                            ))
                        })?;
                    }
                }
                for row in encoded_visible_rows {
                    if seen_row_raw_tables.insert(row.row_raw_table.clone()) {
                        ensure_header(row.row_raw_table.as_str(), &|| {
                            super::encode_raw_table_header(&super::row_raw_table_header(
                                &row.row_raw_table_id,
                                &row.user_descriptor,
                            ))
                        })?;
                    }
                }
                if encoded_history_rows
                    .iter()
                    .any(|row| row.needs_exact_locator)
                {
                    ensure_header(super::HISTORY_ROW_BATCH_TABLE_LOCATOR_TABLE, &|| {
                        super::encode_raw_table_header(&super::RawTableHeader::system(
                            super::STORAGE_KIND_HISTORY_ROW_BATCH_TABLE_LOCATOR,
                            1,
                        ))
                    })?;
                }
                if encoded_visible_rows
                    .iter()
                    .any(|row| row.needs_exact_locator)
                {
                    ensure_header(super::VISIBLE_ROW_TABLE_LOCATOR_TABLE, &|| {
                        super::encode_raw_table_header(&super::RawTableHeader::system(
                            super::STORAGE_KIND_VISIBLE_ROW_TABLE_LOCATOR,
                            1,
                        ))
                    })?;
                }

                let borrowed_history_rows = encoded_history_rows
                    .iter()
                    .map(|row| HistoryRowBytes {
                        row_raw_table: row.row_raw_table.as_str(),
                        branch: row.branch.as_str(),
                        row_id: row.row_id,
                        batch_id: row.batch_id,
                        bytes: &row.bytes,
                    })
                    .collect::<Vec<_>>();
                append_history_region_row_bytes_core(
                    table,
                    &borrowed_history_rows,
                    |key, bytes| Self::set(&inner.conn, &inner.tx, key, bytes),
                )?;
                for row in encoded_history_rows {
                    if !row.needs_exact_locator {
                        continue;
                    }
                    let locator =
                        super::encode_exact_row_table_locator(&super::ExactRowTableLocator {
                            row_raw_table: row.row_raw_table.clone().into(),
                            table_name: row.row_raw_table_id.table_name.clone(),
                            schema_hash: row.row_raw_table_id.schema_hash,
                        })?;
                    raw_table_put_core(
                        super::HISTORY_ROW_BATCH_TABLE_LOCATOR_TABLE,
                        &super::history_row_batch_table_locator_key(
                            row.row_id,
                            row.branch.as_str(),
                            row.batch_id,
                        ),
                        &locator,
                        |storage_key, bytes| Self::set(&inner.conn, &inner.tx, storage_key, bytes),
                    )?;
                }

                let borrowed_visible_rows = encoded_visible_rows
                    .iter()
                    .map(|row| VisibleRowBytes {
                        row_raw_table: row.row_raw_table.as_str(),
                        branch: row.branch.as_str(),
                        row_id: row.row_id,
                        bytes: &row.bytes,
                    })
                    .collect::<Vec<_>>();
                upsert_visible_region_row_bytes_core(
                    table,
                    &borrowed_visible_rows,
                    |key, bytes| Self::set(&inner.conn, &inner.tx, key, bytes),
                )?;
                for row in encoded_visible_rows {
                    if !row.needs_exact_locator {
                        continue;
                    }
                    let locator = super::ExactRowTableLocator {
                        row_raw_table: row.row_raw_table.clone().into(),
                        table_name: row.row_raw_table_id.table_name.clone(),
                        schema_hash: row.row_raw_table_id.schema_hash,
                    };
                    let cache_key = (row.branch.clone(), row.row_id);
                    let already_written = inner.visible_row_table_locators.get(&cache_key)
                        == Some(&locator)
                        || locators
                            .iter()
                            .any(|(key, seen)| key == &cache_key && seen == &locator);
                    if !already_written {
                        let locator_bytes = super::encode_exact_row_table_locator(&locator)?;
                        raw_table_put_core(
                            super::VISIBLE_ROW_TABLE_LOCATOR_TABLE,
                            &super::visible_row_table_locator_key(row.branch.as_str(), row.row_id),
                            &locator_bytes,
                            |storage_key, bytes| {
                                Self::set(&inner.conn, &inner.tx, storage_key, bytes)
                            },
                        )?;
                        locators.push((cache_key, locator));
                    }
                }

                for mutation in index_mutations {
                    match mutation {
                        IndexMutation::Insert {
                            table,
                            column,
                            branch,
                            value,
                            row_id,
                        } => {
                            let raw_table = key_codec::index_raw_table(table, column, branch);
                            let key =
                                key_codec::index_entry_key(table, column, branch, value, *row_id)?;
                            raw_table_put_core(&raw_table, &key, &[0x01], |storage_key, bytes| {
                                Self::set(&inner.conn, &inner.tx, storage_key, bytes)
                            })?;
                        }
                        IndexMutation::Remove {
                            table,
                            column,
                            branch,
                            value,
                            row_id,
                        } => {
                            let key = match key_codec::index_entry_key(
                                table, column, branch, value, *row_id,
                            ) {
                                Ok(key) => key,
                                Err(StorageError::IndexKeyTooLarge { .. }) => continue,
                                Err(error) => return Err(error),
                            };
                            let raw_table = key_codec::index_raw_table(table, column, branch);
                            raw_table_delete_core(&raw_table, &key, |storage_key| {
                                Self::delete(&inner.conn, &inner.tx, storage_key)
                            })?;
                        }
                    }
                }
                Ok((ensured, locators))
            })?;
            inner.ensured_raw_table_headers.extend(ensured);
            inner.visible_row_table_locators.extend(locators);
            Ok(())
        })
    }

    fn patch_row_region_rows_by_batch(
        &mut self,
        table: &str,
        batch_id: crate::row_histories::BatchId,
        state: Option<RowState>,
        confirmed_tier: Option<DurabilityTier>,
    ) -> Result<(), StorageError> {
        super::patch_row_region_rows_by_batch_with_storage(
            self,
            table,
            batch_id,
            state,
            confirmed_tier,
        )
    }

    fn load_visible_region_row_bytes(
        &self,
        table: &str,
        branch: &str,
        row_id: ObjectId,
    ) -> Result<Option<Vec<u8>>, StorageError> {
        Ok(
            super::load_visible_region_row_bytes_with_storage(self, table, branch, row_id)?
                .map(|row| row.bytes),
        )
    }

    fn scan_visible_region_bytes(
        &self,
        table: &str,
        branch: &str,
    ) -> Result<Vec<Vec<u8>>, StorageError> {
        Ok(
            super::scan_visible_row_bytes_with_storage(self, table, branch)?
                .into_iter()
                .map(|row| row.bytes)
                .collect(),
        )
    }

    fn scan_visible_region_row_batches(
        &self,
        table: &str,
        row_id: ObjectId,
    ) -> Result<Vec<StoredRowBatch>, StorageError> {
        let branches =
            super::scan_visible_region_row_batch_branches_with_storage(self, table, row_id)?;

        let mut rows = Vec::new();
        for branch in branches {
            if let Some(row) = self.load_visible_region_row(table, &branch, row_id)? {
                rows.push(row);
            }
        }
        rows.sort_by_key(|row| row.branch.clone());
        Ok(rows)
    }

    fn load_history_row_batch_bytes(
        &self,
        table: &str,
        branch: &str,
        row_id: ObjectId,
        batch_id: crate::row_histories::BatchId,
    ) -> Result<Option<Vec<u8>>, StorageError> {
        Ok(super::load_history_row_batch_row_bytes_with_storage(
            self, table, branch, row_id, batch_id,
        )?
        .map(|row| row.bytes))
    }

    fn scan_history_region_bytes(
        &self,
        table: &str,
        scan: HistoryScan,
    ) -> Result<Vec<Vec<u8>>, StorageError> {
        Ok(
            super::scan_history_row_bytes_with_storage(self, table, scan)?
                .into_iter()
                .map(|row| row.bytes)
                .collect(),
        )
    }

    fn begin_read_pass(&self) -> Result<(), StorageError> {
        self.with_pass_boundary(|inner| inner.begin_pass(&self.read_pass_begin_failures))
    }
    fn end_read_pass(&self) -> Result<super::PassOutcome, StorageError> {
        self.with_pass_boundary(|inner| inner.end_pass())
    }

    /// v18 item 5 (D2): persist the exact locator the ladder recovered, inside the pass's
    /// transaction when one is open (the barrier commits it), in its own transaction when
    /// the read runs outside a pass — a direct `&self` read never commits an earlier pass's
    /// writes ahead of the barrier (`was_open`). The header is ensured the way the write
    /// path does it; its name joins the cache only after RELEASE. No dedup-map insert (the
    /// map is the write path's, and the helper puts unconditionally). Any failure is
    /// counted here and returned; the ladder never fails the read for it.
    fn record_visible_row_table_locator_recovery(
        &self,
        branch: &str,
        row_id: ObjectId,
        locator: &super::ExactRowTableLocator,
    ) -> Result<(), StorageError> {
        let result = self.with_inner_mut(|inner| {
            if inner.read_only {
                return Ok(());
            }
            let was_open = inner.write_tx_open;
            inner.ensure_write_tx()?;
            let needs_header = !inner
                .ensured_raw_table_headers
                .contains(super::VISIBLE_ROW_TABLE_LOCATOR_TABLE);
            // diff r23 SF6: encoded only when it is actually written. r20 SF4 made
            // `ensure_header` lazy in both stores so a header ensured long ago costs nothing per
            // call; the recovery hook is the one new write path that had not been given the
            // rule, and it runs on the read path, which is the path this item exists to make
            // cheaper.
            let header = if needs_header {
                Some(super::encode_raw_table_header(
                    &super::RawTableHeader::system(
                        super::STORAGE_KIND_VISIBLE_ROW_TABLE_LOCATOR,
                        super::EXACT_ROW_TABLE_LOCATOR_STORAGE_FORMAT_V1,
                    ),
                )?)
            } else {
                None
            };
            let locator_bytes = super::encode_exact_row_table_locator(locator)?;
            let key = super::visible_row_table_locator_key(branch, row_id);
            Self::with_savepoint(&inner.conn, &inner.tx, || {
                if let Some(header) = &header {
                    raw_table_put_core(
                        super::RAW_TABLE_HEADER_TABLE,
                        super::VISIBLE_ROW_TABLE_LOCATOR_TABLE,
                        header,
                        |storage_key, bytes| Self::set(&inner.conn, &inner.tx, storage_key, bytes),
                    )?;
                }
                raw_table_put_core(
                    super::VISIBLE_ROW_TABLE_LOCATOR_TABLE,
                    &key,
                    &locator_bytes,
                    |storage_key, bytes| Self::set(&inner.conn, &inner.tx, storage_key, bytes),
                )
            })?;
            if needs_header {
                inner
                    .ensured_raw_table_headers
                    .insert(super::VISIBLE_ROW_TABLE_LOCATOR_TABLE.to_string());
            }
            // v18 item 5 (diff r22 B2): this is a DIRECT write to the visible-locator pointer,
            // so it owes the same cache eviction the realign and the repair sweep pay — see
            // `put_visible_row_table_locator`'s doc, which states the rule with no exception.
            // Without it the write path can later see its own stale value in
            // `visible_row_table_locators`, skip a persist the store needs, and leave a pointer
            // that ladder arm 1 answers from directly — a hit there returns before the ladder
            // runs, so nothing heals it and the wrong family is served for good.
            inner
                .visible_row_table_locators
                .remove(&(branch.to_string(), row_id));
            if inner.pass_depth == 0 && !was_open {
                inner.commit_write_tx()?;
            }
            Ok(())
        });
        if result.is_err() {
            self.recovery_persist_failures
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        result
    }
    fn note_visible_locator_recovery(&self) {
        self.visible_ladder_recoveries
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    fn flush_wal(&self) -> Result<(), StorageError> {
        let mut inner = self.lock_inner()?;
        if let Some(inner) = inner.as_mut() {
            // v18 item 4: a read-only store has nothing to commit and must not checkpoint.
            if inner.read_only {
                return Ok(());
            }
            // Commit the open write transaction so writes land in the WAL
            // and survive a process crash. (`commit_write_tx` reconciles first: a lost
            // transaction is reported here, on the barrier, as `Err(LostWrites)`.)
            inner.commit_write_tx()?;
            // v18 item 8: the barrier is the COMMIT above — that, and only that, is what
            // makes a write survive a kill. The PASSIVE checkpoint below moves WAL pages
            // into the main database file; it is housekeeping, and running it on EVERY
            // barrier made a settle-heavy node pay a checkpoint per pass. It is now due
            // either on every barrier (interval `None`, the old behaviour, kept so a caller
            // can still ask for it) or once per interval, measured from the last checkpoint
            // that actually RAN. SQLite's own `wal_autocheckpoint` still bounds the WAL by
            // size in between, which is the arm that keeps deferring safe.
            let due = match *self
                .checkpoint_interval
                .lock()
                .unwrap_or_else(|e| e.into_inner())
            {
                None => true,
                Some(interval) => {
                    self.last_checkpoint
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .elapsed()
                        >= interval
                }
            };
            if due {
                // The result row is (busy, log, checkpointed). It has to be READ: the first
                // version of this ran `execute_batch`, which discards it, so a checkpoint that
                // moved nothing was counted as one that drained the log — r21 measured 96 such
                // "successes" against a WAL that never shrank.
                //
                // The discriminator is `checkpointed < log`, NOT `busy`. For a PASSIVE
                // checkpoint SQLite resets `SQLITE_BUSY` to OK on purpose, so as "not to report
                // a checkpoint failure just because there are active readers"
                // (`sqlite3.c:67453-67457`); `busy` is set only when another connection holds
                // the checkpointer lock (`sqlite3.c:101414-101418`), which this engine, with one
                // writer per store, never arranges. Reading `busy` would have measured zero
                // forever (diff r25 B1).
                //
                // A non-WAL database returns (0, -1, -1), which reads as drained. Correct: there
                // is no log to drain.
                let outcome = inner
                    .conn
                    .query_row("PRAGMA wal_checkpoint(PASSIVE)", [], |row| {
                        Ok((row.get::<_, i64>(1)?, row.get::<_, i64>(2)?))
                    });
                match outcome {
                    Ok((log, checkpointed)) => {
                        if checkpointed < log {
                            self.checkpoints_blocked
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            crate::query_manager::settle_cost::bump(
                                &crate::query_manager::settle_cost::CHECKPOINTS_BLOCKED,
                            );
                        } else {
                            self.checkpoints
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            crate::query_manager::settle_cost::bump(
                                &crate::query_manager::settle_cost::CHECKPOINTS,
                            );
                        }
                        // Renewed on BOTH arms. Gating the stamp on a full drain would hand a
                        // foreign reader control over this node's checkpoint rate: one probe or
                        // stray `sqlite3` shell holding a snapshot on the prod store would
                        // silently restore one PRAGMA per barrier — the defect. The counters
                        // stay pure observability and carry no control flow.
                        *self
                            .last_checkpoint
                            .lock()
                            .unwrap_or_else(|e| e.into_inner()) = std::time::Instant::now();
                    }
                    Err(e) => {
                        // A checkpoint that cannot run is NOT a barrier failure. The rows are
                        // committed; returning `Err` here would withhold this node's delivery
                        // confirmations for writes that are already durable — the exact loss
                        // the confirmation exists to prevent, from the other side. It would also
                        // re-arm the very loop item 8 exists to remove: the barrier's error path
                        // logs unlatched on every writing tick and skips
                        // `confirm_applied_rows_upstream`, and every new write clears the retry
                        // latch (diff r25 B5, and the reason its D4 is not taken).
                        //
                        // The stamp is NOT renewed: an error means the attempt did not happen,
                        // where a blocked checkpoint above did.
                        self.checkpoint_failures
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        crate::query_manager::settle_cost::bump(
                            &crate::query_manager::settle_cost::CHECKPOINT_FAILURES,
                        );
                        let transient = checkpoint_error_is_transient(&e);
                        let error =
                            sqlite_error(&inner.tx.last_sqlite_error, "sqlite wal checkpoint", e);
                        // Latched, like the two failure reports above it in this file: a store
                        // whose checkpoints fail fails them on every barrier, and an unlatched
                        // line here is one per writing tick forever. The COUNTER is what carries
                        // the ongoing rate; the log line only has to say it started.
                        if !inner.checkpoint_failure_reported {
                            inner.checkpoint_failure_reported = true;
                            if transient {
                                tracing::warn!(%error, "wal checkpoint skipped; the barrier stands");
                            } else {
                                // Not transient — a full or failing disk shows up HERE first,
                                // while commits keep appending to a WAL nothing drains.
                                tracing::error!(
                                    %error,
                                    "wal checkpoint failing for a non-transient reason; the \
                                     barrier stands but the WAL is no longer being drained"
                                );
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn flush(&self) -> Result<(), StorageError> {
        self.flush_wal()
    }

    fn close(&self) -> Result<(), StorageError> {
        let Some(mut inner) = self.lock_inner()?.take() else {
            return Ok(());
        };
        if inner.read_only {
            drop(inner);
            return Ok(());
        }
        // v18 item 4 (design v9 SF6, v11 SF4): reconcile first; a lost transaction is
        // logged by the reconcile and there is nothing left to commit — `close()` does not
        // fail for it; a live transaction (clean or dirty) is committed as before.
        match inner.reconcile_tx_state() {
            Ok(()) => inner.commit_write_tx()?,
            Err(StorageError::LostWrites { .. }) => {}
            Err(error) => return Err(error),
        }
        // Best-effort compaction before dropping the connection.
        let _ = inner.conn.execute_batch("PRAGMA wal_checkpoint(PASSIVE)");
        drop(inner);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn close_releases_lock_for_reopen() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.sqlite");
        let storage = SqliteStorage::open(&db_path).unwrap();
        storage.close().unwrap();
        let reopened = SqliteStorage::open(&db_path).unwrap();
        reopened.close().unwrap();
    }

    #[test]
    fn flush_does_not_panic() {
        use crate::object::ObjectId;
        use crate::query_manager::types::{SchemaBuilder, TableSchema};
        use crate::storage::RowLocator;

        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("test.sqlite");
        let mut storage = SqliteStorage::open(&path).unwrap();
        let schema_hash = crate::query_manager::types::SchemaHash::compute(
            &SchemaBuilder::new()
                .table(
                    TableSchema::builder("users")
                        .column("name", crate::query_manager::types::ColumnType::Text),
                )
                .build(),
        );

        for _ in 0..10 {
            let id = ObjectId::new();
            storage
                .put_row_locator(
                    id,
                    Some(&RowLocator {
                        table: "users".into(),
                        origin_schema_hash: Some(schema_hash),
                    }),
                )
                .unwrap();
        }

        // flush() should not panic or return an error for an open empty store.
        storage.flush().unwrap();
    }

    #[test]
    fn operations_fail_after_close() {
        use crate::object::ObjectId;

        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("test.sqlite");
        let storage = SqliteStorage::open(&path).unwrap();
        storage.close().unwrap();

        // Storage is closed but NOT yet dropped.
        // A real close() takes the inner; the next call must return Err, not succeed or panic.
        let result = storage.load_row_locator(ObjectId::new());
        assert!(
            result.is_err(),
            "load_row_locator should return Err after close, got Ok"
        );
    }

    #[test]
    fn storage_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<SqliteStorage>();
    }

    #[test]
    fn open_rejects_store_manifest_version_mismatch() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("test.sqlite");
        let storage = SqliteStorage::open(&path).unwrap();
        storage.close().unwrap();

        let conn = rusqlite::Connection::open(&path).unwrap();
        let bad_manifest = super::super::StoreManifest {
            store_kind: super::super::SQLITE_STORE_KIND.to_string(),
            store_format_version: 999,
        };
        let bytes = super::super::encode_store_manifest(&bad_manifest).unwrap();
        conn.execute(
            "UPDATE kv SET value = ?2 WHERE key = ?1",
            rusqlite::params![super::super::STORE_MANIFEST_KEY.as_bytes(), bytes],
        )
        .unwrap();

        let err = match SqliteStorage::open(&path) {
            Ok(_) => panic!("expected store manifest version mismatch"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("store manifest version mismatch"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn open_rejects_nonempty_store_without_manifest() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("legacy.sqlite");
        let conn = rusqlite::Connection::open(&path).unwrap();
        conn.execute_batch(
            "CREATE TABLE kv (
                 key   BLOB PRIMARY KEY,
                 value BLOB NOT NULL
             ) WITHOUT ROWID;",
        )
        .unwrap();
        conn.execute(
            "INSERT INTO kv(key, value) VALUES (?1, ?2)",
            rusqlite::params![b"raw:legacy:alice".as_slice(), b"hello".as_slice()],
        )
        .unwrap();
        drop(conn);

        let err = match SqliteStorage::open(&path) {
            Ok(_) => panic!("expected missing manifest rejection"),
            Err(err) => err,
        };
        assert!(
            err.to_string()
                .contains("missing store manifest for non-empty sqlite store"),
            "unexpected error: {err}"
        );
    }

    mod sqlite_conformance {
        use crate::storage::Storage;
        use crate::storage::sqlite::SqliteStorage;
        use crate::storage_conformance_tests_persistent;

        storage_conformance_tests_persistent!(
            sqlite,
            || {
                let dir = tempfile::TempDir::new().unwrap();
                let path = dir.path().join("test.sqlite");
                let storage = SqliteStorage::open(&path).unwrap();
                // Leak TempDir so the directory lives as long as the storage.
                std::mem::forget(dir);
                Box::new(storage) as Box<dyn Storage>
            },
            |path: &std::path::Path| {
                Box::new(SqliteStorage::open(path.join("test.sqlite")).unwrap()) as Box<dyn Storage>
            }
        );
    }

    /// v18 item 8: the durability barrier checkpoints by size and interval, not per tick.
    ///
    /// `flush_wal` runs on every batched tick that wrote (`runtime_core/ticks.rs`, the
    /// barrier), under the engine lock, and today it is `COMMIT` + `PRAGMA
    /// wal_checkpoint(PASSIVE)` every time — up to three fsyncs per writing tick (WAL header
    /// after the restart, WAL, main file) and a page copy into the main file (prod
    /// rpc-server: WAL flush 10–22 % of `batched_tick`). The `COMMIT` is what makes a write
    /// survive a kill; the checkpoint only bounds the WAL.
    ///
    /// v18 item 4 (C): the transaction gates. Internal on purpose, as a group: what they
    /// observe is whether an explicit transaction is open, whether the store believes a
    /// write landed in it, and which SQLite result code ended it — three pieces of
    /// connection state no client API exposes, and one of them (a second raw connection on
    /// the same file) is something no client can even construct. Each gate names its own
    /// observable below.
    mod read_pass_transactions {
        use super::super::{SqliteStorage, sqlite_error};
        use crate::object::ObjectId;
        use crate::query_manager::types::Value;
        use crate::storage::{IndexMutation, Storage, StorageError};
        fn open_in(dir: &tempfile::TempDir) -> (SqliteStorage, std::path::PathBuf) {
            let path = dir.path().join("pass.sqlite");
            (SqliteStorage::open(&path).unwrap(), path)
        }

        /// G-C7. Internal on purpose: the observable is `busy_snapshot_retries_for_test` — a
        /// per-store count of a retry that happens entirely inside one storage call — and
        /// the fixture opens a SECOND connection to the same file, which no client API can
        /// do. `SQLITE_BUSY_SNAPSHOT` is what a first write meets when another connection
        /// committed after this one's pass took its read snapshot; the busy handler is not
        /// consulted for it (`walBeginWriteTransaction`), so without the retry the write
        /// fails outright.
        #[test]
        fn a_first_write_after_a_foreign_commit_is_retried_once() {
            let dir = tempfile::TempDir::new().unwrap();
            let (mut store, path) = open_in(&dir);
            store.raw_table_put("__gate", "seed", b"seed").unwrap();
            store.flush_wal().unwrap();
            // The pass takes its read snapshot.
            store.begin_read_pass().unwrap();
            assert!(
                store.raw_table_get("__gate", "seed").unwrap().is_some(),
                "fixture: the pass must read, so the snapshot is taken"
            );
            // Another connection commits behind it.
            let mut other = SqliteStorage::open(&path).unwrap();
            other
                .raw_table_put("__gate", "foreign", b"foreign")
                .unwrap();
            other.flush_wal().unwrap();
            other.close().unwrap();
            // The pass's FIRST write: BUSY_SNAPSHOT, retried once.
            store
                .raw_table_put("__gate_new_family", "row", b"row")
                .expect("the first write of the pass must survive a foreign commit");
            assert_eq!(
                store.busy_snapshot_retries_for_test(),
                1,
                "the write must have been retried exactly once"
            );
            let outcome = store.end_read_pass().unwrap();
            assert!(outcome.wrote, "the retried write is still a write");
            store.flush_wal().unwrap();
            assert_eq!(
                store
                    .raw_table_get("__gate_new_family", "row")
                    .unwrap()
                    .as_deref(),
                Some(&b"row"[..]),
                "the retried write must be readable after the barrier"
            );
        }

        /// G-C8. Internal on purpose: `transaction_open_for_test` and
        /// `is_autocommit_for_test` are the store's memory and the connection's truth, and
        /// this gate is about them disagreeing — a state no client API can see or produce.
        /// A CLEAN pass transaction ended behind the store's back (SQLite's own rollback on
        /// NOMEM/IOERR) must not brick the store: nothing was lost, so the next write opens
        /// a fresh transaction and lands.
        #[test]
        fn a_clean_transaction_rolled_back_behind_the_store_does_not_brick_it() {
            let dir = tempfile::TempDir::new().unwrap();
            let (mut store, path) = open_in(&dir);
            store.begin_read_pass().unwrap();
            assert!(
                store.transaction_open_for_test() && !store.is_autocommit_for_test(),
                "fixture: the pass must hold an explicit transaction"
            );
            store.roll_back_transaction_for_test().unwrap();
            assert!(
                store.transaction_open_for_test() && store.is_autocommit_for_test(),
                "fixture: the store must still believe it holds the transaction SQLite just ended"
            );
            store
                .raw_table_put("__gate", "after", b"after")
                .expect("a write after a CLEAN rollback must land, not report a loss");
            assert!(
                !store.is_autocommit_for_test(),
                "the write must have opened a fresh explicit transaction"
            );
            assert!(!store.lost_writes_reported_for_test(), "nothing was lost");
            store.end_read_pass().unwrap();
            store.flush_wal().expect("the barrier must commit");
            store.close().unwrap();
            let reopened = SqliteStorage::open(&path).unwrap();
            assert_eq!(
                reopened
                    .raw_table_get("__gate", "after")
                    .unwrap()
                    .as_deref(),
                Some(&b"after"[..]),
                "the write after the clean rollback must be durable"
            );
        }

        /// G-C8'. The other half: a DIRTY transaction ended behind the store's back is a
        /// LOSS, and the store must be loud about it on every boundary until it is
        /// reopened. Internal on purpose: same instruments as G-C8, plus
        /// `lost_writes_reported_for_test` (the store's log-once latch).
        #[test]
        fn a_dirty_transaction_rolled_back_behind_the_store_is_loud() {
            let dir = tempfile::TempDir::new().unwrap();
            let (mut store, path) = open_in(&dir);
            store.begin_read_pass().unwrap();
            store.raw_table_put("__gate", "doomed", b"doomed").unwrap();
            assert!(
                store.tx_dirty_for_test(),
                "fixture: the write must have marked the transaction dirty"
            );
            store.roll_back_transaction_for_test().unwrap();
            let error = store
                .flush_wal()
                .expect_err("a barrier over a lost transaction must fail");
            assert!(
                matches!(error, StorageError::LostWrites { .. }),
                "the barrier must report the loss, got {error:?}"
            );
            assert!(store.lost_writes_reported_for_test());
            let again = store
                .flush_wal()
                .expect_err("every later barrier must report it too (strict)");
            assert!(matches!(again, StorageError::LostWrites { .. }));
            store.close().unwrap();
            let reopened = SqliteStorage::open(&path).unwrap();
            assert!(
                reopened
                    .raw_table_get("__gate", "doomed")
                    .unwrap()
                    .is_none(),
                "the rolled-back write must be absent — that is what makes it a loss"
            );
        }

        /// G-C9. A body that fails BEFORE any statement leaves the transaction clean, and
        /// the store keeps working. Hook-free: an index key over `INDEX_KEY_MAX_BYTES`
        /// fails in `key_codec::index_entry_key` with the row slices empty, so not one
        /// statement runs. Internal on purpose: `tx_dirty_for_test` is the store's own
        /// belief about a transaction, which no client API reports.
        #[test]
        fn a_body_that_fails_before_any_statement_leaves_the_transaction_clean() {
            let dir = tempfile::TempDir::new().unwrap();
            let (mut store, path) = open_in(&dir);
            store.begin_read_pass().unwrap();
            let oversize = "c".repeat(6 * 1024);
            let error = store
                .apply_encoded_row_mutation(
                    "docs",
                    &[],
                    &[],
                    &[IndexMutation::Insert {
                        table: "docs",
                        column: &oversize,
                        branch: "main",
                        value: Value::Text("x".to_string()),
                        row_id: ObjectId::new(),
                    }],
                )
                .expect_err("an index key over the limit must fail the call");
            assert!(
                matches!(error, StorageError::IndexKeyTooLarge { .. }),
                "unexpected error: {error:?}"
            );
            assert!(
                !store.tx_dirty_for_test(),
                "a body that ran no statement must not mark the transaction dirty"
            );
            store
                .raw_table_put("__gate", "after", b"after")
                .expect("the store must still take writes");
            store.end_read_pass().unwrap();
            store.flush_wal().unwrap();
            store.close().unwrap();
            let reopened = SqliteStorage::open(&path).unwrap();
            assert_eq!(
                reopened
                    .raw_table_get("__gate", "after")
                    .unwrap()
                    .as_deref(),
                Some(&b"after"[..]),
            );
        }

        /// G-C10. `SQLITE_FULL` rolls the whole transaction back itself — nothing partial
        /// remains, so the store must NOT call it a loss. Internal on purpose:
        /// `set_max_page_count_for_test` clamps this store's own connection (per-connection,
        /// non-persistent) because a full disk cannot be provoked through any client API,
        /// and `last_sqlite_error_for_test` is the extended result code the store recorded.
        #[test]
        fn a_full_rollback_inside_a_savepoint_does_not_mark_the_transaction_dirty() {
            let dir = tempfile::TempDir::new().unwrap();
            let (mut store, _path) = open_in(&dir);
            store.begin_read_pass().unwrap();
            store.set_max_page_count_for_test(1).unwrap();
            let error = store
                .raw_table_put("__gate", "big", &vec![0xAB; 64 * 1024])
                .expect_err("the put must fail with SQLITE_FULL");
            assert!(
                matches!(&error, StorageError::IoError(message) if message.contains("database or disk is full")),
                "unexpected error: {error:?}"
            );
            assert_eq!(
                store.last_sqlite_error_for_test(),
                Some(13),
                "SQLITE_FULL is 13"
            );
            assert!(
                store.is_autocommit_for_test(),
                "SQLITE_FULL ends the transaction itself"
            );
            assert!(
                !store.tx_dirty_for_test(),
                "a failure that landed nothing must not mark the transaction dirty"
            );
            let outcome = store
                .end_read_pass()
                .expect("nothing was lost: the failure was reported to the caller");
            assert!(!outcome.wrote);
            assert!(!store.lost_writes_reported_for_test());
        }

        /// G-RO. A read-only store never holds a transaction: no pass transaction is
        /// opened on it, every write is refused before `BEGIN`, and neither the barrier nor
        /// `close()` commits or checkpoints. This is what lets a tool read a live store
        /// without becoming its writer or pinning its WAL. Internal on purpose:
        /// `is_autocommit_for_test` and `pass_depth_for_test` are the only way to see that
        /// no transaction was opened; a client sees only that the reads worked.
        #[test]
        fn a_read_only_store_never_holds_a_transaction() {
            let dir = tempfile::TempDir::new().unwrap();
            let (mut writer, path) = open_in(&dir);
            writer.raw_table_put("__gate", "seed", b"seed").unwrap();
            writer.flush_wal().unwrap();
            writer.close().unwrap();
            let mut reader = SqliteStorage::open_read_only(&path).unwrap();
            reader.begin_read_pass().unwrap();
            assert_eq!(
                reader.raw_table_get("__gate", "seed").unwrap().as_deref(),
                Some(&b"seed"[..]),
                "a read-only store must still read"
            );
            let error = reader
                .raw_table_put("__gate", "denied", b"denied")
                .expect_err("a read-only store must refuse writes");
            assert!(
                matches!(&error, StorageError::IoError(message) if message.contains("read-only")),
                "unexpected error: {error:?}"
            );
            let outcome = reader.end_read_pass().unwrap();
            assert!(!outcome.wrote);
            assert!(
                reader.is_autocommit_for_test(),
                "no transaction may ever be open on a read-only store"
            );
            assert_eq!(reader.pass_depth_for_test(), 0);
            reader.flush_wal().expect("the barrier is a no-op here");
            assert!(reader.is_autocommit_for_test());
            reader.close().unwrap();
        }

        /// The mapper records the extended result code before it stringifies. Internal on
        /// purpose: `last_sqlite_error_for_test` reads a cell the store keeps for its own
        /// retry predicate.
        #[test]
        fn the_error_mapper_records_the_extended_result_code() {
            use std::cell::Cell;
            let last: Cell<Option<i32>> = Cell::new(None);
            let failure = rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_BUSY_SNAPSHOT),
                Some("busy snapshot".to_string()),
            );
            let mapped = sqlite_error(&last, "gate", failure);
            assert_eq!(last.get(), Some(rusqlite::ffi::SQLITE_BUSY_SNAPSHOT));
            assert!(
                matches!(&mapped, StorageError::IoError(message) if message.starts_with("gate: ")),
                "unexpected error: {mapped:?}"
            );
        }

        /// v18 items 4/5, methodology step 9: the differential oracle over the pass state machine.
        ///
        /// The three pieces of state the design added — `pass_depth`, `write_tx_open`, `dirty` —
        /// interact through six operations, and their ORDER is the whole of the design (v14 row 4:
        /// the depth moves before the reconcile; diff r20 B1: the boundaries must not take the
        /// entry reconcile). Hand-written gates cover the sequences we thought of. This covers the
        /// ones we did not: 200 randomised programs of up to 30 operations, checked against a model
        /// after every single step, with the seed printed on failure so any red is replayable.
        ///
        /// The model is deliberately written from the DESIGN's rules, not from the implementation:
        ///
        /// - `begin` raises the depth, then opens a transaction only at depth 1 with none open;
        /// - `end` lowers the depth, and at depth 0 commits a transaction that is CLEAN — a dirty
        ///   one is the durability barrier's to commit, never the pass's (design v7 § C2);
        /// - a write opens a transaction if none is open and marks it dirty, at any depth;
        /// - the barrier commits whatever is open and clears dirty;
        /// - autocommit is exactly "no transaction open".
        ///
        /// Internal on purpose: every observable here — `pass_depth_for_test`,
        /// `transaction_open_for_test`, `is_autocommit_for_test`, `tx_dirty_for_test` — is the
        /// store's private bookkeeping against the connection's truth, and the point of the gate is
        /// that they agree. No client API exposes any of them, and none could: a client cannot see
        /// a transaction boundary at all.
        #[test]
        fn the_pass_state_machine_matches_its_model_under_random_programs() {
            #[derive(Clone, Copy, Debug, PartialEq, Eq)]
            enum Op {
                Begin,
                End,
                Write,
                Read,
                Barrier,
            }

            /// The design's rules, with no reference to the implementation.
            #[derive(Debug, Default)]
            struct Model {
                depth: u32,
                tx_open: bool,
                dirty: bool,
            }
            impl Model {
                fn begin(&mut self) {
                    self.depth += 1;
                    if self.depth == 1 && !self.tx_open {
                        self.tx_open = true;
                        self.dirty = false;
                    }
                }
                fn end(&mut self) {
                    if self.depth == 0 {
                        return;
                        // the caller never does this; see the generator
                    }
                    self.depth -= 1;
                    if self.depth == 0 && self.tx_open && !self.dirty {
                        self.tx_open = false;
                    }
                }
                fn write(&mut self) {
                    self.tx_open = true;
                    self.dirty = true;
                }
                fn barrier(&mut self) {
                    self.tx_open = false;
                    self.dirty = false;
                }
            }
            // A tiny xorshift so the programs are reproducible from the seed alone; `rand` is not
            // a dependency of this crate and a differential that cannot be replayed is a rumour.
            struct Rng(u64);
            impl Rng {
                fn next(&mut self) -> u64 {
                    self.0 ^= self.0 << 13;
                    self.0 ^= self.0 >> 7;
                    self.0 ^= self.0 << 17;
                    self.0
                }
                fn below(&mut self, n: u64) -> u64 {
                    self.next() % n
                }
            }
            let dir = tempfile::TempDir::new().unwrap();
            for seed in 1..=200u64 {
                let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
                let path = dir.path().join(format!("diff-{seed}.sqlite"));
                let _ = std::fs::remove_file(&path);
                let mut store = SqliteStorage::open(&path).unwrap();
                let mut model = Model::default();
                let mut program: Vec<Op> = Vec::new();
                let mut writes = 0u32;
                let steps = 5 + rng.below(26);
                for _ in 0..steps {
                    // `end` only when a pass is open: an unbalanced end is a caller bug the
                    // `debug_assert!` in `end_pass` already gates (chain row E4), not a state the
                    // model is meant to describe.
                    let op = match rng.below(if model.depth > 0 { 5 } else { 4 }) {
                        0 => Op::Begin,
                        1 => Op::Write,
                        2 => Op::Read,
                        3 => Op::Barrier,
                        _ => Op::End,
                    };
                    program.push(op);
                    match op {
                        Op::Begin => {
                            store.begin_read_pass().unwrap();
                            model.begin();
                        }
                        Op::End => {
                            let outcome = store.end_read_pass().unwrap();
                            assert_eq!(
                                outcome.wrote, model.dirty,
                                "seed {seed} program {program:?}: the pass must report whether it \
                             wrote; the barrier decides what to do with a dirty transaction \
                             from this answer alone"
                            );
                            model.end();
                        }
                        Op::Write => {
                            writes += 1;
                            store
                                .raw_table_put("__diff_family", &format!("k{writes}"), b"payload")
                                .unwrap();
                            model.write();
                        }
                        Op::Read => {
                            // A read never moves the machine; it is here because a read inside a
                            // pass runs statements on the pass transaction, and a read outside one
                            // must not open a transaction of its own.
                            let _ = store.raw_table_get("__diff_absent", "missing").unwrap();
                        }
                        Op::Barrier => {
                            store.flush_wal().unwrap();
                            model.barrier();
                        }
                    }
                    assert_eq!(
                        store.pass_depth_for_test(),
                        model.depth,
                        "seed {seed} program {program:?}: pass depth"
                    );
                    assert_eq!(
                        store.transaction_open_for_test(),
                        model.tx_open,
                        "seed {seed} program {program:?}: the store's memory of its transaction"
                    );
                    assert_eq!(
                        store.is_autocommit_for_test(),
                        !model.tx_open,
                        "seed {seed} program {program:?}: autocommit is exactly \"no transaction \
                     open\"; a disagreement here is the defect-20 split forming"
                    );
                    assert_eq!(
                        store.tx_dirty_for_test(),
                        model.dirty,
                        "seed {seed} program {program:?}: the dirty flag decides who commits — the \
                     pass or the barrier"
                    );
                }
                // Whatever the program did, a balanced tail must leave the store quiescent: this is
                // the property the WAL depends on, and the reason C3 exists.
                while model.depth > 0 {
                    store.end_read_pass().unwrap();
                    model.end();
                }
                store.flush_wal().unwrap();
                model.barrier();
                assert!(
                    store.is_autocommit_for_test() && store.pass_depth_for_test() == 0,
                    "seed {seed} program {program:?}: a balanced program followed by a barrier must \
                 leave no transaction open and no depth outstanding; a reader left open pins \
                 the WAL for every checkpoint that follows"
                );
            }
        }
    }

    /// Internal on purpose: the observable is how many checkpoints a stream of barriers
    /// runs and whether the WAL stays bounded — neither is visible through a client API.
    mod checkpoint_policy {
        use super::super::SqliteStorage;
        use crate::storage::Storage;
        use std::time::Duration;
        fn open_in(dir: &tempfile::TempDir) -> (SqliteStorage, std::path::PathBuf) {
            let path = dir.path().join("store.sqlite");
            (SqliteStorage::open(&path).unwrap(), path)
        }
        /// Serialises this module against itself. `CheckpointCounts` is PROCESS-GLOBAL
        /// (`query_manager::settle_cost`), so G8-9 — the one gate that reads it — is measuring
        /// a counter every other test here also advances, on other threads, at the same time
        /// (diff r28). Without this it can go red on a sibling's checkpoint or, worse, green on
        /// one. Cargo runs tests in threads of ONE process, so a plain mutex is the whole fix;
        /// `serial_test` is not a dependency of this crate and is not worth becoming one for a
        /// single module. Poisoning is ignored on purpose: a failing sibling has already
        /// reported, and turning that into a cascade of unrelated failures hides it.
        fn exclusive() -> std::sync::MutexGuard<'static, ()> {
            static GATE: std::sync::Mutex<()> = std::sync::Mutex::new(());
            GATE.lock().unwrap_or_else(|e| e.into_inner())
        }

        fn barrier(store: &mut SqliteStorage, key: &str, bytes: usize) {
            store
                .raw_table_put("__gate_rows", key, &vec![0xAB; bytes])
                .unwrap();
            store.flush_wal().unwrap();
        }
        fn wal_len(path: &std::path::Path) -> u64 {
            let mut wal = path.as_os_str().to_owned();
            wal.push("-wal");
            std::fs::metadata(wal).map(|m| m.len()).unwrap_or(0)
        }

        /// G8-1. A hundred small barriers within the interval run no explicit checkpoint.
        /// Red today: one per barrier.
        #[test]
        fn small_barriers_within_the_interval_do_not_checkpoint() {
            let _exclusive = exclusive();
            let dir = tempfile::TempDir::new().unwrap();
            let (mut store, _) = open_in(&dir);
            store.set_checkpoint_interval(Some(Duration::from_secs(3600)));
            for i in 0..100 {
                barrier(&mut store, &format!("k{i}"), 100);
            }
            assert_eq!(
                store.checkpoints_for_test(),
                0,
                "a barrier is a COMMIT; the checkpoint waits for the interval or the WAL size"
            );
        }

        /// G8-2. Interval `None` keeps today's behaviour: every barrier checkpoints.
        #[test]
        fn without_an_interval_every_barrier_checkpoints() {
            let _exclusive = exclusive();
            let dir = tempfile::TempDir::new().unwrap();
            let (mut store, _) = open_in(&dir);
            store.set_checkpoint_interval(None);
            for i in 0..20 {
                barrier(&mut store, &format!("k{i}"), 100);
            }
            assert_eq!(store.checkpoints_for_test(), 20);
        }

        /// G8-3. The WAL stays bounded by size with no explicit checkpoint for an hour:
        /// SQLite's autocheckpoint (1 000 pages, 4 MiB) moves it at commit. Writes 24 MiB
        /// across barriers; the WAL must stay under three times the autocheckpoint size.
        /// Green today through the per-barrier checkpoint; must stay green.
        #[test]
        fn the_wal_is_bounded_by_size_between_explicit_checkpoints() {
            let _exclusive = exclusive();
            let dir = tempfile::TempDir::new().unwrap();
            let (mut store, path) = open_in(&dir);
            // The autocheckpoint is NOT set here any more. It was, and that made this gate
            // test its own fixture: `open` now states `PRAGMA wal_autocheckpoint = 1000`, so
            // the size arm this gate is about is production's, and a chain row can disarm it
            // (diff r25 B3 — deleting the pragma would only restore the identical compiled
            // default, so the row sets it to 0 instead).
            store.set_checkpoint_interval(Some(Duration::from_secs(3600)));
            for i in 0..96 {
                barrier(&mut store, &format!("big{i}"), 256 * 1024);
            }
            let wal = wal_len(&path);
            assert!(
                wal < 12 * 1024 * 1024,
                "the WAL grew to {wal} bytes with the explicit checkpoint deferred — the size \
                 bound must hold without it"
            );
        }

        /// G8-8 (v18 item 8, diff r25 B1/B2). A checkpoint that RUNS but drains nothing must be
        /// counted as blocked, not as a checkpoint.
        ///
        /// This is the assertion that stops the counter lying. The first version of this item
        /// ran the PRAGMA with `execute_batch`, which discards the result row, so every
        /// invocation counted as a drained log — r21 measured 96 of them against a WAL that
        /// never shrank, and `checkpoints` is a column in the stand's own report.
        ///
        /// The reader is a SECOND connection, because that is the only shape that reaches this
        /// arm. On the store's own connection an open transaction makes SQLite return
        /// `SQLITE_LOCKED` (the error path, which G8-6 covers); from another connection it is
        /// not an error at all — for a PASSIVE checkpoint SQLite deliberately resets
        /// `SQLITE_BUSY` to OK so as not to "report a checkpoint failure just because there are
        /// active readers" (`sqlite3.c:67453-67457`). Success is exactly what makes it
        /// dangerous, and the frame columns are the only thing that tells the two apart.
        ///
        /// Internal on purpose: no client API reports whether a checkpoint drained the log.
        /// From outside, a WAL held open by a forgotten `sqlite3` shell on the prod store and a
        /// WAL being drained normally look identical until the disk fills.
        #[test]
        fn a_checkpoint_a_reader_pins_is_counted_as_blocked_not_as_a_checkpoint() {
            let _exclusive = exclusive();
            let dir = tempfile::TempDir::new().unwrap();
            let (mut store, path) = open_in(&dir);
            store.set_checkpoint_interval(None);

            // Frames in the WAL for the checkpoint to have something to move.
            barrier(&mut store, "before-reader", 64 * 1024);
            let drained = store.checkpoints_for_test();
            assert!(
                drained > 0 && store.checkpoints_blocked_for_test() == 0,
                "fixture precondition: with no reader the checkpoint drains the log"
            );

            // A second connection takes a read snapshot and holds it. From here on the
            // checkpointer cannot copy past the frames this snapshot needs.
            let reader = rusqlite::Connection::open(&path).unwrap();
            reader
                .execute_batch("BEGIN DEFERRED; SELECT count(*) FROM kv;")
                .unwrap();

            for i in 0..8 {
                barrier(&mut store, &format!("pinned{i}"), 64 * 1024);
            }

            assert_eq!(
                store.checkpoints_for_test(),
                drained,
                "not one of those barriers drained the log — the reader's snapshot pins it — so \
                 not one of them may be counted as a checkpoint. Counting them is what made the \
                 stand's `checkpoints` column meaningless"
            );
            assert_eq!(
                store.checkpoints_blocked_for_test(),
                8,
                "and every one of them is counted as blocked: the PRAGMA ran and returned \
                 success each time, so a counter that does not read the frame columns cannot \
                 tell this apart from a drained log"
            );
            assert_eq!(
                store.checkpoint_failures_for_test(),
                0,
                "and none of it is an ERROR: SQLite resets BUSY to OK for a PASSIVE checkpoint \
                 with active readers on purpose (`sqlite3.c:67453-67457`)"
            );

            reader.execute_batch("COMMIT").unwrap();
            drop(store);
        }

        /// G8-7 (v18 item 8, diff r25 D5). The gate whose ABSENCE let item 8 ship as a no-op.
        ///
        /// Every other gate in this block configures the store first, so not one of them could
        /// see that nothing else does. The first version of this item left `over()` at `None` —
        /// "checkpoint on every barrier" — and offered `set_checkpoint_interval`, whose complete
        /// caller set was six gates. Six green assertions about a knob no shipping build turned.
        ///
        /// So this gate configures NOTHING. It opens the store the way `jazz-napi`, `jazz-rn`,
        /// the server builder and the client open it, and asks what the shipped default does.
        ///
        /// The three assertions BRACKET the default rather than restate it, and the numbers are
        /// hard-coded on purpose: a gate written against `DEFAULT_CHECKPOINT_INTERVAL` would
        /// follow the constant wherever it went and pin nothing. Retuning a shipped
        /// durability-adjacent default should cost a line that says what it used to be.
        ///
        /// Internal on purpose: the checkpoint count is engine bookkeeping. From outside, a
        /// store that checkpoints on every barrier and one that checkpoints twice a minute
        /// differ only in how much CPU they burn.
        #[test]
        fn a_store_opened_the_way_production_opens_it_does_not_checkpoint_every_barrier() {
            let _exclusive = exclusive();
            let dir = tempfile::TempDir::new().unwrap();
            let (mut store, _path) = open_in(&dir);
            for i in 0..20 {
                barrier(&mut store, &format!("row{i}"), 512);
            }
            assert_eq!(
                store.checkpoints_for_test(),
                0,
                "20 barriers inside the default interval must checkpoint NOTHING — this is the \
                 assertion that would have caught item 8 shipping inert"
            );

            store.age_last_checkpoint_for_test(Duration::from_secs(31));
            barrier(&mut store, "after-31s", 512);
            assert_eq!(
                store.checkpoints_for_test(),
                1,
                "past the interval, the next barrier checkpoints"
            );

            store.age_last_checkpoint_for_test(Duration::from_secs(29));
            barrier(&mut store, "after-29s", 512);
            assert_eq!(
                store.checkpoints_for_test(),
                1,
                "and 29 s does not: with the 31 s leg above, the shipped default is pinned to \
                 (29 s, 31 s] without this gate ever naming the constant"
            );
        }

        /// G8-9 (v18 item 8). The checkpoint must be readable from where the RUNTIME stands.
        ///
        /// Item 8 has now failed to reach a reader twice, by two different mechanisms. First the
        /// accessor: `checkpoints_for_test` is an inherent method on `SqliteStorage`, and the
        /// shipped storage is a `Box<dyn Storage>`, so the runtime cannot call it however public
        /// it is (diff r25 B4) — hence the process-global counters. Then the placement: the
        /// counters were printed on the settle line, but `SettlePass::begin()` closes in
        /// `QueryManager::process` and the barrier runs later in `batched_tick`, so the field was
        /// structurally always zero. Measured on the stand: 175 settle passes, `checkpoints=0`
        /// every time, while the policy underneath was working.
        ///
        /// This gate asserts the only thing that makes the fix legible: a barrier that
        /// checkpoints advances `CheckpointCounts`, the reader the barrier itself now uses, and a
        /// barrier that does not leaves it alone. `is_empty()` is what decides whether the
        /// barrier logs, so both directions are the emit decision.
        ///
        /// Internal on purpose: checkpoint accounting is engine bookkeeping with no public
        /// surface, and from outside the two barriers below are indistinguishable.
        #[test]
        fn the_barrier_can_read_its_own_checkpoint_where_the_settle_line_cannot() {
            let _exclusive = exclusive();
            use crate::query_manager::settle_cost::CheckpointCounts;

            let dir = tempfile::TempDir::new().unwrap();
            let (mut store, _path) = open_in(&dir);
            store.set_checkpoint_interval(Some(Duration::from_secs(3600)));

            let quiet_before = CheckpointCounts::read();
            barrier(&mut store, "no-checkpoint", 512);
            let quiet = CheckpointCounts::read().since(quiet_before);
            assert!(
                quiet.is_empty(),
                "a barrier inside the interval must leave the counters alone, or the runtime \
                 logs a line per tick and item 8 puts back in the log the cost it took off the \
                 disk; got {quiet:?}"
            );

            store.age_last_checkpoint_for_test(Duration::from_secs(3601));
            let ran_before = CheckpointCounts::read();
            barrier(&mut store, "checkpoint", 512);
            let ran = CheckpointCounts::read().since(ran_before);
            assert_eq!(
                ran.failures, 0,
                "the checkpoint must not have failed in this fixture, or the assertion below \
                 would be satisfied by the wrong counter"
            );
            assert_eq!(
                ran.checkpoints + ran.blocked,
                1,
                "the barrier that DID checkpoint has to be visible through the reader the \
                 barrier itself uses. `blocked` counts too: this store has no concurrent \
                 reader, but a machine that gives it one must still report a checkpoint \
                 attempt rather than silence; got {ran:?}"
            );
        }

        /// G8-4. A barrier without a checkpoint is still durable across a kill. A kill is
        /// modelled by copying the database and its WAL (not the shared-memory index) to a
        /// fresh directory and opening the copy: the wal-index is rebuilt from the WAL, as
        /// after a crash. Green today; must stay green.
        #[test]
        fn a_committed_barrier_survives_a_kill_without_a_checkpoint() {
            let _exclusive = exclusive();
            let dir = tempfile::TempDir::new().unwrap();
            let (mut store, path) = open_in(&dir);
            store.set_checkpoint_interval(Some(Duration::from_secs(3600)));
            barrier(&mut store, "survivor", 1000);
            assert_eq!(
                store.checkpoints_for_test(),
                0,
                "fixture precondition: the point of this gate is that the row survives with NO \
                 checkpoint. Without this line it passes whether one ran or not, and the chain \
                 row that forces `due` true comes back green (diff r25 D6)"
            );
            let crashed = tempfile::TempDir::new().unwrap();
            let copy = crashed.path().join("store.sqlite");
            std::fs::copy(&path, &copy).unwrap();
            let mut wal = path.as_os_str().to_owned();
            wal.push("-wal");
            let mut wal_copy = copy.as_os_str().to_owned();
            wal_copy.push("-wal");
            std::fs::copy(&wal, &wal_copy).expect("the WAL holds the committed row");
            // The original stays open: the copy has no live shared memory, so its open runs
            // WAL recovery, which is what a process restart after a kill does.
            let reopened = SqliteStorage::open(&copy).unwrap();
            let row = reopened.raw_table_get("__gate_rows", "survivor").unwrap();
            assert_eq!(
                row.map(|v| v.len()),
                Some(1000),
                "the committed row must be there"
            );
            drop(store);
        }

        /// G8-5. The interval is measured from the last checkpoint that RAN, not from the last
        /// barrier: once it has elapsed the next barrier checkpoints, and only that one; the
        /// clock is aged, not slept. Red today (every barrier checkpoints). Disarm "stamp on
        /// every barrier": the two 20 s ages never add up to the interval → red at the last
        /// assertion.
        #[test]
        fn the_interval_elapsed_checkpoints_exactly_the_next_barrier() {
            let _exclusive = exclusive();
            let dir = tempfile::TempDir::new().unwrap();
            let (mut store, _) = open_in(&dir);
            store.set_checkpoint_interval(Some(Duration::from_secs(30)));
            for i in 0..5 {
                barrier(&mut store, &format!("k{i}"), 100);
            }
            assert_eq!(
                store.checkpoints_for_test(),
                0,
                "nothing within the interval"
            );
            store.age_last_checkpoint_for_test(Duration::from_secs(31));
            barrier(&mut store, "after", 100);
            assert_eq!(
                store.checkpoints_for_test(),
                1,
                "the first barrier past the interval"
            );
            barrier(&mut store, "after2", 100);
            assert_eq!(store.checkpoints_for_test(), 1, "and not the one after it");
            // The stamp is the checkpoint's, not the barrier's: two barriers 20 s apart do
            // not each restart the clock.
            store.age_last_checkpoint_for_test(Duration::from_secs(20));
            barrier(&mut store, "twenty", 100);
            assert_eq!(
                store.checkpoints_for_test(),
                1,
                "20 s after the checkpoint: none"
            );
            store.age_last_checkpoint_for_test(Duration::from_secs(20));
            barrier(&mut store, "forty", 100);
            assert_eq!(
                store.checkpoints_for_test(),
                2,
                "40 s after the checkpoint the next barrier checkpoints, whatever barriers ran \
                 in between"
            );
        }

        /// G8-6. The barrier is the COMMIT; a checkpoint that cannot run (a read
        /// transaction open on this connection → SQLITE_LOCKED, or an I/O error in the copy)
        /// is logged and counted, never a barrier failure — a failed barrier withholds the
        /// node's delivery confirmations for rows that ARE committed. A failed checkpoint
        /// does not renew the stamp either: with a 30 s interval the next barrier after the
        /// release checkpoints at once instead of waiting another interval. Red today: the
        /// error propagates out of `flush_wal`.
        #[test]
        fn a_checkpoint_that_cannot_run_does_not_fail_the_barrier() {
            let _exclusive = exclusive();
            let dir = tempfile::TempDir::new().unwrap();
            let (mut store, _) = open_in(&dir);
            store.set_checkpoint_interval(Some(Duration::from_secs(30)));
            store.age_last_checkpoint_for_test(Duration::from_secs(31));
            barrier(&mut store, "first", 100);
            assert_eq!(
                store.checkpoints_for_test(),
                1,
                "fixture: the first barrier checkpointed"
            );
            store.hold_read_transaction_for_test().unwrap();
            store.age_last_checkpoint_for_test(Duration::from_secs(31));
            let outcome = store.flush_wal();
            assert!(
                outcome.is_ok(),
                "the barrier committed; the checkpoint's failure is not the barrier's: {outcome:?}"
            );
            assert_eq!(
                store.checkpoint_failures_for_test(),
                1,
                "the failure is counted"
            );
            assert_eq!(
                store.checkpoints_for_test(),
                1,
                "and not counted as a checkpoint"
            );
            store.release_read_transaction_for_test().unwrap();
            // The failure did not renew the stamp: the interval is still overdue, so the next
            // barrier checkpoints right away — a stamp taken on failure would make it wait
            // another 30 s and leave the count at 1.
            store.flush_wal().unwrap();
            assert_eq!(
                store.checkpoints_for_test(),
                2,
                "the connection checkpoints again"
            );
        }
    }
}
