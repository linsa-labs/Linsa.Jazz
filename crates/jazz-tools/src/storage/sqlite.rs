//! SQLite-backed Storage implementation.
//!
//! Uses `rusqlite` with bundled SQLite. Single KV table on a WITHOUT ROWID
//! B-tree, WAL mode. Writes are batched into a lazy explicit transaction that
//! stays open across multiple calls and is committed on `flush()` / `close()`.
//! Per-operation SAVEPOINTs nested inside that transaction provide rollback
//! semantics for individual operations. Targets React Native / mobile.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use rusqlite::OptionalExtension;

use super::{
    HistoryRowBytes, IndexMutation, OwnedHistoryRowBytes, OwnedVisibleRowBytes, RawTableMutation,
    Storage, StorageError, VisibleRowBytes, key_codec,
    storage_core::{
        append_history_region_row_bytes_core, raw_table_delete_core, raw_table_family_keys_core,
        raw_table_family_last_key_core, raw_table_get_core, raw_table_put_core,
        raw_table_scan_prefix_core, raw_table_scan_prefix_keys_core, raw_table_scan_range_core,
        raw_table_scan_range_keys_core, upsert_visible_region_row_bytes_core,
    },
};
use crate::object::ObjectId;
use crate::row_histories::{HistoryScan, RowState, StoredRowBatch};
use crate::sync_manager::DurabilityTier;

/// Row locators kept above this count are dropped wholesale on the next insert.
const MAX_MEMOIZED_ROW_LOCATORS: usize = 16_384;
/// Largest key set the read memo holds for one prefix.
const MAX_SMALL_PREFIX_KEYS: usize = 128;
/// Bytes of keys held over all prefixes above which the key sets are dropped wholesale.
/// Bytes, not keys: an index key carries the indexed value and can run to kilobytes.
const MAX_MEMOIZED_PREFIX_KEY_BYTES: usize = 4 * 1024 * 1024;
/// Prefixes the memo holds an answer for, whatever the answer weighs. The byte bound above
/// counts keys; a prefix remembered as empty, wide or merely non-empty holds none.
const MAX_MEMOIZED_PREFIXES: usize = 16_384;

/// What is known about the keys under one storage-key prefix.
enum PrefixKeys {
    /// Every storage key under the prefix, exactly. An empty set means the prefix is empty.
    Small(HashSet<String>),
    /// More keys than are worth holding; says nothing about any one of them. Deletes never
    /// turn it back into `Small` — "not known" is always a safe answer.
    Large,
    /// Held a key when it was asked whether it holds any, and was not read further. Says
    /// nothing; a question about one key reads the set.
    Unlisted,
}

/// How much of a prefix a question needs read.
#[derive(Clone, Copy, PartialEq, Eq)]
enum PrefixQuestion {
    /// Whether the prefix holds anything: one key answers it.
    Emptiness,
    /// Whether the prefix holds one given key: needs the whole set.
    Key,
}

/// Answers the read path would otherwise re-derive from the same unchanged keys.
///
/// Everything here is a function of the key space as THIS connection sees it. Two things
/// change that view, and each has its own guard:
///
/// - this connection's writes all pass through [`SqliteStorage::set`] /
///   [`SqliteStorage::delete`], which keep the memo exact; a rolled-back savepoint
///   un-writes keys without passing through either, so it drops the memo whole;
/// - another connection's commit is seen by no code of ours. Inside a transaction it is
///   not seen by SQLite either — the transaction reads one snapshot — so the memo is
///   consulted only inside a transaction, and every transaction starts by comparing
///   `PRAGMA data_version`, which moves exactly when another connection committed, and
///   drops the memo if it moved (see [`SqliteInner::note_transaction_began`]).
#[derive(Default)]
struct ReadMemo {
    /// Storage-key prefix (always ending in `:`) → the keys under it.
    prefix_keys: HashMap<String, PrefixKeys>,
    /// Bytes of the keys held by the `Small` sets of `prefix_keys`, in total.
    prefix_key_bytes: usize,
    /// `__row_locator` storage key → the decoded locator, `None` for "no locator".
    row_locators: HashMap<String, Option<super::RowLocator>>,
    /// Raw-table header prefix → the row raw tables registered under it.
    row_raw_table_ids: HashMap<String, Vec<super::RowRawTableId>>,
}

impl ReadMemo {
    fn clear(&mut self) {
        self.prefix_keys.clear();
        self.prefix_key_bytes = 0;
        self.row_locators.clear();
        self.row_raw_table_ids.clear();
    }

    fn touches_raw_table_headers(storage_key: &str) -> bool {
        key_codec::strip_raw_table_key(super::RAW_TABLE_HEADER_TABLE, storage_key).is_some()
    }

    /// Memoized prefixes end in `:`, so the ones a key falls under are exactly its
    /// `:`-terminated heads.
    fn for_each_memoized_prefix_of(
        prefix_keys: &mut HashMap<String, PrefixKeys>,
        storage_key: &str,
        mut f: impl FnMut(&mut PrefixKeys),
    ) {
        if prefix_keys.is_empty() {
            return;
        }
        for (index, byte) in storage_key.bytes().enumerate() {
            if byte == b':'
                && let Some(keys) = prefix_keys.get_mut(&storage_key[..=index])
            {
                f(keys);
            }
        }
    }

    fn key_bytes(keys: &HashSet<String>) -> usize {
        keys.iter().map(String::len).sum()
    }

    fn memoize_prefix(&mut self, storage_prefix: String, keys: PrefixKeys) {
        if self.prefix_keys.len() >= MAX_MEMOIZED_PREFIXES
            && !self.prefix_keys.contains_key(&storage_prefix)
        {
            self.prefix_keys.clear();
            self.prefix_key_bytes = 0;
        }
        if let PrefixKeys::Small(set) = &keys {
            let bytes = Self::key_bytes(set);
            if self.prefix_key_bytes + bytes > MAX_MEMOIZED_PREFIX_KEY_BYTES {
                self.prefix_keys.clear();
                self.prefix_key_bytes = 0;
            }
            self.prefix_key_bytes += bytes;
        }
        if let Some(PrefixKeys::Small(replaced)) = self.prefix_keys.insert(storage_prefix, keys) {
            self.prefix_key_bytes -= Self::key_bytes(&replaced);
        }
    }

    fn note_set(&mut self, storage_key: &str) {
        let bytes = &mut self.prefix_key_bytes;
        Self::for_each_memoized_prefix_of(&mut self.prefix_keys, storage_key, |keys| {
            if let PrefixKeys::Small(set) = keys {
                if set.insert(storage_key.to_string()) {
                    *bytes += storage_key.len();
                }
                if set.len() > MAX_SMALL_PREFIX_KEYS {
                    *bytes -= Self::key_bytes(set);
                    *keys = PrefixKeys::Large;
                }
            }
        });
        // Writes grow the sets as reads do, and a store being filled writes far more
        // than it reads: the bound holds on this side too.
        if self.prefix_key_bytes > MAX_MEMOIZED_PREFIX_KEY_BYTES {
            self.prefix_keys.clear();
            self.prefix_key_bytes = 0;
        }
        self.note_key_changed(storage_key);
    }

    fn note_delete(&mut self, storage_key: &str) {
        let bytes = &mut self.prefix_key_bytes;
        Self::for_each_memoized_prefix_of(&mut self.prefix_keys, storage_key, |keys| {
            if let PrefixKeys::Small(set) = keys
                && set.remove(storage_key)
            {
                *bytes -= storage_key.len();
            }
        });
        self.note_key_changed(storage_key);
    }

    /// The byte count is kept by hand on every path that touches a set; this recomputes
    /// it from the sets.
    #[cfg(test)]
    fn assert_accounting(&self) {
        let held: usize = self
            .prefix_keys
            .values()
            .map(|keys| match keys {
                PrefixKeys::Small(set) => Self::key_bytes(set),
                PrefixKeys::Large | PrefixKeys::Unlisted => 0,
            })
            .sum();
        assert_eq!(
            self.prefix_key_bytes, held,
            "the memo's byte count drifted from the keys it holds"
        );
        assert!(self.prefix_key_bytes <= MAX_MEMOIZED_PREFIX_KEY_BYTES);
        assert!(self.prefix_keys.len() <= MAX_MEMOIZED_PREFIXES);
        assert!(self.row_locators.len() <= MAX_MEMOIZED_ROW_LOCATORS);
    }

    fn note_key_changed(&mut self, storage_key: &str) {
        if !self.row_locators.is_empty() {
            self.row_locators.remove(storage_key);
        }
        if !self.row_raw_table_ids.is_empty() && Self::touches_raw_table_headers(storage_key) {
            self.row_raw_table_ids.clear();
        }
    }
}

/// Kill switches for the two read-path savings, both default ON.
///
/// `JAZZ_SQLITE_READ_MEMO=0` (or `false`) answers nothing from the read memo: every
/// probe reads the table, as before. `JAZZ_SQLITE_READ_SCOPE=0` opens no transaction
/// for a scope of reads: each read runs in autocommit, as before — and, with no
/// transaction to hold a snapshot, the memo is then consulted only while a write
/// transaction is open.
fn read_memo_enabled() -> bool {
    #[cfg(any(test, feature = "test"))]
    if let Some(forced) = read_path_override::forced_memo() {
        return forced;
    }
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| env_switch_on("JAZZ_SQLITE_READ_MEMO"))
}

fn read_scope_enabled() -> bool {
    #[cfg(any(test, feature = "test"))]
    if let Some(forced) = read_path_override::forced_scope() {
        return forced;
    }
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| env_switch_on("JAZZ_SQLITE_READ_SCOPE"))
}

fn env_switch_on(name: &str) -> bool {
    !matches!(std::env::var(name).as_deref(), Ok("0") | Ok("false"))
}

#[cfg(any(test, feature = "test"))]
mod read_path_override {
    use std::sync::atomic::{AtomicU8, Ordering};
    use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};

    const UNSET: u8 = 0;
    const FORCED_ON: u8 = 1;
    const FORCED_OFF: u8 = 2;

    static MEMO: AtomicU8 = AtomicU8::new(UNSET);
    static SCOPE: AtomicU8 = AtomicU8::new(UNSET);

    fn lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    /// Holds the SQLite read path in a forced mode for the guard's lifetime.
    pub struct SqliteReadPathMode {
        _serialised: MutexGuard<'static, ()>,
    }

    impl Drop for SqliteReadPathMode {
        fn drop(&mut self) {
            MEMO.store(UNSET, Ordering::SeqCst);
            SCOPE.store(UNSET, Ordering::SeqCst);
        }
    }

    /// Force the read memo and the read-scope transaction on or off for the returned
    /// guard's lifetime. Guards serialise: a second one waits for the first to drop.
    pub fn force_sqlite_read_path(memo: bool, scope: bool) -> SqliteReadPathMode {
        let guard = lock().lock().unwrap_or_else(PoisonError::into_inner);
        let state = |on: bool| if on { FORCED_ON } else { FORCED_OFF };
        MEMO.store(state(memo), Ordering::SeqCst);
        SCOPE.store(state(scope), Ordering::SeqCst);
        SqliteReadPathMode { _serialised: guard }
    }

    fn forced(state: &AtomicU8) -> Option<bool> {
        match state.load(Ordering::SeqCst) {
            FORCED_ON => Some(true),
            FORCED_OFF => Some(false),
            _ => None,
        }
    }

    pub(super) fn forced_memo() -> Option<bool> {
        forced(&MEMO)
    }

    pub(super) fn forced_scope() -> Option<bool> {
        forced(&SCOPE)
    }
}

#[cfg(any(test, feature = "test"))]
pub use read_path_override::{SqliteReadPathMode, force_sqlite_read_path};

struct SqliteInner {
    conn: rusqlite::Connection,
    #[allow(dead_code)]
    path: PathBuf,
    /// Whether an explicit transaction holding writes is currently open.
    write_tx_open: bool,
    /// Whether an explicit transaction opened for a read scope is currently open. It
    /// never holds a write: the first write closes it and opens the write transaction.
    read_scope_tx_open: bool,
    read_scope_depth: u32,
    /// `PRAGMA data_version` as of the last transaction this connection began.
    seen_data_version: Option<i64>,
    /// A transaction holding writes ended without committing them. The engine above has
    /// those writes in memory as stored, and nothing here can put them back, so from
    /// then on no flush reports success: the store has to be reopened.
    lost_writes: bool,
    /// `total_changes()` when the open write transaction began: a transaction lost with
    /// the count unmoved changed nothing, so it took nothing with it.
    write_tx_changes_at_begin: u64,
    /// The namespace the caches above this storage are keyed by, shared with the
    /// `SqliteStorage` that owns this connection. Those caches also remember what was
    /// written — a raw-table header, a catalogued schema — and a write transaction lost
    /// with changes in it makes every such memory suspect: it moves to a fresh namespace.
    cache_namespace: Arc<AtomicUsize>,
    /// Namespaces left behind that way, until the next read of the namespace purges
    /// what is cached under them. Not purged here: this runs under the connection's
    /// lock, and the caches have locks of their own.
    abandoned_cache_namespaces: Vec<usize>,
    ensured_raw_table_headers: HashSet<String>,
    visible_row_table_locators: HashMap<(String, ObjectId), super::ExactRowTableLocator>,
    memo: RefCell<ReadMemo>,
}

impl SqliteInner {
    /// Drops everything this connection remembers about which keys exist: the read memo
    /// and the two write-side caches that let a write skip a header or a locator it has
    /// already written.
    fn forget_key_assumptions(&mut self) {
        self.memo.borrow_mut().clear();
        self.ensured_raw_table_headers.clear();
        self.visible_row_table_locators.clear();
    }

    /// SQLite ends a transaction by itself when a statement fails hard enough (I/O error,
    /// out of memory, disk full), reads included, and the connection is back in
    /// autocommit with every pending write gone. Nothing reports that to us, so every
    /// place that is about to rely on "a transaction is open" asks the connection first.
    ///
    /// A read scope's transaction lost this way took nothing with it. A write
    /// transaction took its writes: that is recorded in `lost_writes`, and every flush
    /// from then on fails (see [`Self::commit_write_tx`]).
    fn reconcile_lost_transaction(&mut self) {
        if (self.write_tx_open || self.read_scope_tx_open) && self.conn.is_autocommit() {
            if self.write_tx_open && self.conn.total_changes() != self.write_tx_changes_at_begin {
                self.lost_writes = true;
                let abandoned = self
                    .cache_namespace
                    .swap(super::next_storage_cache_namespace(), Ordering::Relaxed);
                self.abandoned_cache_namespaces.push(abandoned);
            }
            self.write_tx_open = false;
            self.read_scope_tx_open = false;
            self.forget_key_assumptions();
        }
    }

    /// Runs first thing inside every transaction this connection begins. Reading the data
    /// version pins the transaction's snapshot, and the version read is that snapshot's:
    /// if it is not the one the previous transaction saw, another connection committed in
    /// between and what is remembered about the keys describes a store that is gone.
    fn note_transaction_began(&mut self) {
        let version = self
            .conn
            .prepare_cached("PRAGMA data_version")
            .and_then(|mut statement| statement.query_row([], |row| row.get::<_, i64>(0)))
            .ok();
        if version.is_none() || version != self.seen_data_version {
            self.forget_key_assumptions();
        }
        self.seen_data_version = version;
    }

    /// Whether the read memo may be consulted or filled: only inside a transaction, whose
    /// snapshot no other connection can change under it. Asks the connection, not the
    /// flags, so a transaction SQLite ended by itself counts as closed at once.
    fn memo_usable(&self) -> bool {
        (self.write_tx_open || self.read_scope_tx_open)
            && !self.conn.is_autocommit()
            && read_memo_enabled()
    }

    /// Start a write transaction if one isn't already open.
    fn ensure_write_tx(&mut self) -> Result<(), StorageError> {
        self.reconcile_lost_transaction();
        if self.read_scope_tx_open {
            // The scope's transaction reads a snapshot. A write that joined it would have
            // to upgrade that snapshot, and the upgrade fails outright — the busy handler
            // is not asked — once another connection has committed past it. It holds no
            // writes, so it is closed and the write gets a transaction of its own.
            self.conn
                .execute_batch("COMMIT")
                .map_err(|e| StorageError::IoError(format!("sqlite end read scope: {e}")))?;
            self.read_scope_tx_open = false;
        }
        if !self.write_tx_open {
            // IMMEDIATE: the writer's lock is taken here, through the busy handler, so the
            // data version read next is the one this transaction's writes land on.
            self.conn
                .execute_batch("BEGIN IMMEDIATE")
                .map_err(|e| StorageError::IoError(format!("sqlite begin: {e}")))?;
            self.write_tx_open = true;
            self.write_tx_changes_at_begin = self.conn.total_changes();
            self.note_transaction_began();
            if self.conn.is_autocommit() {
                // Reading the data version failed hard enough to end the transaction it
                // was read in. Nothing was written in it; a write that went on would run
                // in autocommit and be committed on its own, ahead of the flush.
                self.write_tx_open = false;
                self.forget_key_assumptions();
                return Err(StorageError::IoError(
                    "sqlite begin: the transaction did not stay open".to_string(),
                ));
            }
        }
        Ok(())
    }

    /// Commit the open transaction, if any.
    ///
    /// Succeeds only if every write made through this connection since it was opened is
    /// now committed. Once a write transaction has been lost — rolled back by SQLite on
    /// a failed statement or a failed commit — that can no longer be said: what is open
    /// is still committed, so later writes are not held back, but the flush fails, and
    /// keeps failing until the store is reopened. The caller's durability barrier stays
    /// shut, and nothing the lost writes covered is confirmed to whoever sent it.
    fn commit_write_tx(&mut self) -> Result<(), StorageError> {
        self.reconcile_lost_transaction();
        if self.write_tx_open || self.read_scope_tx_open {
            if let Err(error) = self.conn.execute_batch("COMMIT") {
                // A failed commit can leave the transaction rolled back, which un-writes
                // keys without passing through `set` / `delete`. If it is still open the
                // flags stand and the next flush tries again.
                self.forget_key_assumptions();
                self.reconcile_lost_transaction();
                return Err(StorageError::IoError(format!("sqlite commit: {error}")));
            }
            self.write_tx_open = false;
            self.read_scope_tx_open = false;
        }
        if self.lost_writes {
            return Err(StorageError::IoError(
                "sqlite lost a write transaction: writes made before it are not stored, \
                 reopen the store"
                    .to_string(),
            ));
        }
        Ok(())
    }

    /// A statement with no rows and no parameters, compiled once per connection. A read
    /// scope opens and closes on every tick, whether or not the tick reads anything.
    fn run_cached(&self, sql: &str) -> rusqlite::Result<()> {
        self.conn.prepare_cached(sql)?.execute([]).map(|_| ())
    }

    /// Autocommit takes and drops the WAL read lock around every statement; inside a
    /// transaction a statement pays neither. A scope of reads therefore runs in one
    /// transaction, which is closed when the scope ends — or sooner, by the first write
    /// in the scope, which opens the write transaction the rest of the scope then reads
    /// in. That one stays open for the next flush, exactly as if no scope were there.
    fn begin_read_scope(&mut self) {
        self.read_scope_depth += 1;
        if self.read_scope_depth != 1 {
            return;
        }
        self.reconcile_lost_transaction();
        if !self.write_tx_open
            && !self.read_scope_tx_open
            && read_scope_enabled()
            && self.run_cached("BEGIN").is_ok()
        {
            self.read_scope_tx_open = true;
            crate::query_manager::settle_cost::bump(
                &crate::query_manager::settle_cost::STORAGE_READ_SCOPES,
            );
            self.note_transaction_began();
            self.read_scope_tx_open = !self.conn.is_autocommit();
        }
    }

    fn end_read_scope(&mut self) {
        self.read_scope_depth = self.read_scope_depth.saturating_sub(1);
        if self.read_scope_depth != 0 || !self.read_scope_tx_open {
            return;
        }
        self.reconcile_lost_transaction();
        if !self.read_scope_tx_open {
            return;
        }
        if self.run_cached("COMMIT").is_err() && !self.conn.is_autocommit() {
            // It holds no writes: rolling it back loses nothing. If even that leaves it
            // open it stays what it is, a read scope's transaction, for the next write
            // or flush to close — never one a write may join.
            let _ = self.conn.execute_batch("ROLLBACK");
        }
        self.read_scope_tx_open = !self.conn.is_autocommit();
    }
}

pub struct SqliteStorage {
    cache_namespace: Arc<AtomicUsize>,
    /// The store was closed without everything written through it being committed. The
    /// connection that knew is gone; a flush asked afterwards must not answer for it.
    closed_with_lost_writes: AtomicBool,
    inner: Mutex<Option<SqliteInner>>,
}

impl SqliteStorage {
    fn store_has_any_rows(conn: &rusqlite::Connection) -> Result<bool, StorageError> {
        conn.query_row("SELECT EXISTS(SELECT 1 FROM kv LIMIT 1)", [], |row| {
            row.get::<_, i64>(0)
        })
        .map(|exists| exists != 0)
        .map_err(|e| StorageError::IoError(format!("sqlite inspect store contents: {e}")))
    }

    fn ensure_store_manifest(conn: &rusqlite::Connection) -> Result<(), StorageError> {
        let expected = super::expected_store_manifest(super::SQLITE_STORE_KIND);
        let existing = conn
            .query_row(
                "SELECT value FROM kv WHERE key = ?1",
                rusqlite::params![super::STORE_MANIFEST_KEY.as_bytes()],
                |row| row.get::<_, Vec<u8>>(0),
            )
            .optional()
            .map_err(|e| StorageError::IoError(format!("sqlite read store manifest: {e}")))?;

        match existing {
            Some(bytes) => {
                let actual = super::decode_store_manifest(&bytes)?;
                super::validate_store_manifest(&actual, &expected)
            }
            None => {
                if Self::store_has_any_rows(conn)? {
                    return Err(StorageError::IoError(
                        "missing store manifest for non-empty sqlite store".to_string(),
                    ));
                }
                let bytes = super::encode_store_manifest(&expected)?;
                conn.execute(
                    "INSERT INTO kv(key, value) VALUES (?1, ?2)",
                    rusqlite::params![super::STORE_MANIFEST_KEY.as_bytes(), bytes],
                )
                .map_err(|e| StorageError::IoError(format!("sqlite write store manifest: {e}")))?;
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

    pub fn open(path: impl AsRef<Path>) -> Result<Self, StorageError> {
        let path = path.as_ref();
        let conn = rusqlite::Connection::open(path)
            .map_err(|e| StorageError::IoError(format!("sqlite open: {e}")))?;

        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;
             PRAGMA cache_size = -65536;
             PRAGMA busy_timeout = 5000;
             PRAGMA foreign_keys = OFF;
             CREATE TABLE IF NOT EXISTS kv (
                 key   BLOB PRIMARY KEY,
                 value BLOB NOT NULL
             ) WITHOUT ROWID;",
        )
        .map_err(|e| StorageError::IoError(format!("sqlite init: {e}")))?;
        Self::ensure_store_manifest(&conn)?;

        let cache_namespace = Arc::new(AtomicUsize::new(super::next_storage_cache_namespace()));
        Ok(Self {
            cache_namespace: Arc::clone(&cache_namespace),
            closed_with_lost_writes: AtomicBool::new(false),
            inner: Mutex::new(Some(SqliteInner {
                conn,
                path: path.to_path_buf(),
                write_tx_open: false,
                read_scope_tx_open: false,
                read_scope_depth: 0,
                seen_data_version: None,
                lost_writes: false,
                write_tx_changes_at_begin: 0,
                cache_namespace,
                abandoned_cache_namespaces: Vec::new(),
                ensured_raw_table_headers: HashSet::new(),
                visible_row_table_locators: HashMap::new(),
                memo: RefCell::new(ReadMemo::default()),
            })),
        })
    }

    /// Answers `known` from the memoized key set of `prefix` in `table`, reading as much
    /// of the set as `question` needs if it has not been read that far. `false` whenever
    /// nothing is known — which is always the case outside a transaction.
    fn with_prefix_keys(
        &self,
        table: &str,
        prefix: &str,
        question: PrefixQuestion,
        known: impl FnOnce(&PrefixKeys) -> bool,
    ) -> bool {
        let storage_prefix = key_codec::raw_table_scan_prefix(table, prefix);
        // `ReadMemo` finds the prefixes a written key falls under by its `:`s.
        if !storage_prefix.ends_with(':') {
            return false;
        }
        self.with_inner(|inner| {
            if !inner.memo_usable() {
                return Ok(false);
            }
            match inner.memo.borrow().prefix_keys.get(&storage_prefix) {
                Some(PrefixKeys::Unlisted) if question == PrefixQuestion::Key => {}
                Some(keys) => {
                    #[cfg(debug_assertions)]
                    Self::debug_assert_prefix_keys_exact(&inner.conn, &storage_prefix, keys);
                    return Ok(known(keys));
                }
                None => {}
            }
            let mut end = storage_prefix.clone();
            key_codec::increment_string(&mut end);
            crate::query_manager::settle_cost::bump(
                &crate::query_manager::settle_cost::STORAGE_READ_OPS,
            );
            let limit = match question {
                PrefixQuestion::Emptiness => 1,
                PrefixQuestion::Key => MAX_SMALL_PREFIX_KEYS + 1,
            };
            let found =
                Self::scan_range_keys_limited(&inner.conn, &storage_prefix, &end, false, limit)?;
            let keys = match question {
                PrefixQuestion::Emptiness if !found.is_empty() => PrefixKeys::Unlisted,
                _ if found.len() > MAX_SMALL_PREFIX_KEYS => PrefixKeys::Large,
                _ => PrefixKeys::Small(found.into_iter().collect()),
            };
            let answer = known(&keys);
            inner.memo.borrow_mut().memoize_prefix(storage_prefix, keys);
            Ok(answer)
        })
        .unwrap_or(false)
    }

    /// Parity harness: in debug builds every answer served from a memoized key set is
    /// checked against the table, so the whole test suite keeps testing that `set` /
    /// `delete` and the transaction guards hold the memo exact. Reads through the
    /// uncounted primitive: the cost counters must not see the check. A store that cannot
    /// be read is not this check's business.
    #[cfg(debug_assertions)]
    fn debug_assert_prefix_keys_exact(
        conn: &rusqlite::Connection,
        storage_prefix: &str,
        keys: &PrefixKeys,
    ) {
        let PrefixKeys::Small(memoized) = keys else {
            return;
        };
        let mut end = storage_prefix.to_string();
        key_codec::increment_string(&mut end);
        if let Ok(stored) =
            Self::scan_range_keys_limited(conn, storage_prefix, &end, false, usize::MAX)
        {
            let stored: HashSet<String> = stored.into_iter().collect();
            debug_assert_eq!(
                &stored, memoized,
                "read memo diverged from the table for prefix {storage_prefix:?}"
            );
        }
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

    fn with_inner_mut<T>(
        &self,
        f: impl FnOnce(&mut SqliteInner) -> Result<T, StorageError>,
    ) -> Result<T, StorageError> {
        let mut inner = self.lock_inner()?;
        let inner = inner
            .as_mut()
            .ok_or_else(|| StorageError::IoError("sqlite storage already closed".to_string()))?;
        f(inner)
    }

    /// Run `f` inside a SQLite SAVEPOINT. Releases on success, rolls back on error.
    /// Reads within `f` see uncommitted savepoint writes because all operations
    /// share the same connection.
    fn with_savepoint<T>(
        conn: &rusqlite::Connection,
        memo: &RefCell<ReadMemo>,
        f: impl FnOnce() -> Result<T, StorageError>,
    ) -> Result<T, StorageError> {
        conn.execute("SAVEPOINT jazz_sp", [])
            .map_err(|e| StorageError::IoError(format!("savepoint start: {e}")))?;
        match f() {
            Ok(v) => {
                if let Err(error) = conn.execute("RELEASE jazz_sp", []) {
                    memo.borrow_mut().clear();
                    return Err(StorageError::IoError(format!("savepoint release: {error}")));
                }
                Ok(v)
            }
            Err(e) => {
                let _ = conn.execute("ROLLBACK TO jazz_sp", []);
                let _ = conn.execute("RELEASE jazz_sp", []);
                // The rollback un-wrote keys behind the memo's back.
                memo.borrow_mut().clear();
                Err(e)
            }
        }
    }

    fn get(conn: &rusqlite::Connection, key: &str) -> Result<Option<Vec<u8>>, StorageError> {
        let mut stmt = conn
            .prepare_cached("SELECT value FROM kv WHERE key = ?1")
            .map_err(|e| StorageError::IoError(format!("sqlite prepare get: {e}")))?;
        match stmt.query_row(rusqlite::params![key.as_bytes()], |row| {
            row.get::<_, Vec<u8>>(0)
        }) {
            Ok(v) => Ok(Some(v)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(StorageError::IoError(format!("sqlite get: {e}"))),
        }
    }

    fn scan_prefix(
        conn: &rusqlite::Connection,
        prefix: &str,
    ) -> Result<Vec<(String, Vec<u8>)>, StorageError> {
        let prefix_bytes = prefix.as_bytes();
        let upper = Self::prefix_upper_bound(prefix_bytes)
            .ok_or_else(|| StorageError::IoError("prefix upper bound overflow".to_string()))?;
        let mut stmt = conn
            .prepare_cached("SELECT key, value FROM kv WHERE key >= ?1 AND key < ?2 ORDER BY key")
            .map_err(|e| StorageError::IoError(format!("sqlite prepare scan_prefix: {e}")))?;
        let rows = stmt
            .query_map(rusqlite::params![prefix_bytes, upper.as_slice()], |row| {
                Ok((row.get::<_, Vec<u8>>(0)?, row.get::<_, Vec<u8>>(1)?))
            })
            .map_err(|e| StorageError::IoError(format!("sqlite scan_prefix: {e}")))?;
        let mut out = Vec::new();
        for row in rows {
            let (key_bytes, value) =
                row.map_err(|e| StorageError::IoError(format!("sqlite scan_prefix row: {e}")))?;
            let key = String::from_utf8(key_bytes)
                .map_err(|e| StorageError::IoError(format!("sqlite key utf8: {e}")))?;
            out.push((key, value));
        }
        Ok(out)
    }

    fn scan_prefix_keys(
        conn: &rusqlite::Connection,
        prefix: &str,
    ) -> Result<Vec<String>, StorageError> {
        let prefix_bytes = prefix.as_bytes();
        let upper = Self::prefix_upper_bound(prefix_bytes)
            .ok_or_else(|| StorageError::IoError("prefix upper bound overflow".to_string()))?;
        let mut stmt = conn
            .prepare_cached("SELECT key FROM kv WHERE key >= ?1 AND key < ?2 ORDER BY key")
            .map_err(|e| StorageError::IoError(format!("sqlite prepare scan_prefix_keys: {e}")))?;
        let rows = stmt
            .query_map(rusqlite::params![prefix_bytes, upper.as_slice()], |row| {
                row.get::<_, Vec<u8>>(0)
            })
            .map_err(|e| StorageError::IoError(format!("sqlite scan_prefix_keys: {e}")))?;
        let mut out = Vec::new();
        for row in rows {
            let key_bytes = row
                .map_err(|e| StorageError::IoError(format!("sqlite scan_prefix_keys row: {e}")))?;
            let key = String::from_utf8(key_bytes)
                .map_err(|e| StorageError::IoError(format!("sqlite key utf8: {e}")))?;
            out.push(key);
        }
        Ok(out)
    }

    fn scan_range(
        conn: &rusqlite::Connection,
        start: &str,
        end: &str,
    ) -> Result<Vec<(String, Vec<u8>)>, StorageError> {
        let mut stmt = conn
            .prepare_cached("SELECT key, value FROM kv WHERE key >= ?1 AND key < ?2 ORDER BY key")
            .map_err(|e| StorageError::IoError(format!("sqlite prepare scan_range: {e}")))?;
        let rows = stmt
            .query_map(rusqlite::params![start.as_bytes(), end.as_bytes()], |row| {
                Ok((row.get::<_, Vec<u8>>(0)?, row.get::<_, Vec<u8>>(1)?))
            })
            .map_err(|e| StorageError::IoError(format!("sqlite scan_range: {e}")))?;
        let mut out = Vec::new();
        for row in rows {
            let (key_bytes, value) =
                row.map_err(|e| StorageError::IoError(format!("sqlite scan_range row: {e}")))?;
            let key = String::from_utf8(key_bytes)
                .map_err(|e| StorageError::IoError(format!("sqlite key utf8: {e}")))?;
            out.push((key, value));
        }
        Ok(out)
    }

    fn scan_range_keys(
        conn: &rusqlite::Connection,
        start: &str,
        end: &str,
    ) -> Result<Vec<String>, StorageError> {
        let mut stmt = conn
            .prepare_cached("SELECT key FROM kv WHERE key >= ?1 AND key < ?2 ORDER BY key")
            .map_err(|e| StorageError::IoError(format!("sqlite prepare scan_range_keys: {e}")))?;
        let rows = stmt
            .query_map(rusqlite::params![start.as_bytes(), end.as_bytes()], |row| {
                row.get::<_, Vec<u8>>(0)
            })
            .map_err(|e| StorageError::IoError(format!("sqlite scan_range_keys: {e}")))?;
        let mut out = Vec::new();
        for row in rows {
            let key_bytes =
                row.map_err(|e| StorageError::IoError(format!("sqlite scan_range_keys row: {e}")))?;
            let key = String::from_utf8(key_bytes)
                .map_err(|e| StorageError::IoError(format!("sqlite key utf8: {e}")))?;
            out.push(key);
        }
        Ok(out)
    }

    fn scan_range_keys_limited(
        conn: &rusqlite::Connection,
        start: &str,
        end: &str,
        reverse: bool,
        limit: usize,
    ) -> Result<Vec<String>, StorageError> {
        let sql = if reverse {
            "SELECT key FROM kv WHERE key >= ?1 AND key < ?2 ORDER BY key DESC LIMIT ?3"
        } else {
            "SELECT key FROM kv WHERE key >= ?1 AND key < ?2 ORDER BY key LIMIT ?3"
        };
        let mut stmt = conn.prepare_cached(sql).map_err(|e| {
            StorageError::IoError(format!("sqlite prepare scan_range_keys_limited: {e}"))
        })?;
        let limit = i64::try_from(limit).unwrap_or(i64::MAX);
        let rows = stmt
            .query_map(
                rusqlite::params![start.as_bytes(), end.as_bytes(), limit],
                |row| row.get::<_, Vec<u8>>(0),
            )
            .map_err(|e| StorageError::IoError(format!("sqlite scan_range_keys_limited: {e}")))?;
        let mut out = Vec::new();
        for row in rows {
            let key_bytes = row.map_err(|e| {
                StorageError::IoError(format!("sqlite scan_range_keys_limited row: {e}"))
            })?;
            let key = String::from_utf8(key_bytes)
                .map_err(|e| StorageError::IoError(format!("sqlite key utf8: {e}")))?;
            out.push(key);
        }
        Ok(out)
    }

    fn set(
        conn: &rusqlite::Connection,
        memo: &RefCell<ReadMemo>,
        key: &str,
        value: &[u8],
    ) -> Result<(), StorageError> {
        conn.prepare_cached("INSERT OR REPLACE INTO kv (key, value) VALUES (?1, ?2)")
            .map_err(|e| StorageError::IoError(format!("sqlite prepare set: {e}")))?
            .execute(rusqlite::params![key.as_bytes(), value])
            .map_err(|e| StorageError::IoError(format!("sqlite set: {e}")))?;
        memo.borrow_mut().note_set(key);
        Ok(())
    }

    /// Writes index entries on the open connection, inside the caller's savepoint.
    fn write_index_mutations(
        conn: &rusqlite::Connection,
        memo: &RefCell<ReadMemo>,
        index_mutations: &[IndexMutation<'_>],
    ) -> Result<(), StorageError> {
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
                    let key = key_codec::index_entry_key(table, column, branch, value, *row_id)?;
                    raw_table_put_core(&raw_table, &key, &[0x01], |storage_key, bytes| {
                        Self::set(conn, memo, storage_key, bytes)
                    })?;
                }
                IndexMutation::Remove {
                    table,
                    column,
                    branch,
                    value,
                    row_id,
                } => {
                    let key =
                        match key_codec::index_entry_key(table, column, branch, value, *row_id) {
                            Ok(key) => key,
                            Err(StorageError::IndexKeyTooLarge { .. }) => continue,
                            Err(error) => return Err(error),
                        };
                    let raw_table = key_codec::index_raw_table(table, column, branch);
                    raw_table_delete_core(&raw_table, &key, |storage_key| {
                        Self::delete(conn, memo, storage_key)
                    })?;
                }
            }
        }
        Ok(())
    }

    fn delete(
        conn: &rusqlite::Connection,
        memo: &RefCell<ReadMemo>,
        key: &str,
    ) -> Result<(), StorageError> {
        conn.prepare_cached("DELETE FROM kv WHERE key = ?1")
            .map_err(|e| StorageError::IoError(format!("sqlite prepare delete: {e}")))?
            .execute(rusqlite::params![key.as_bytes()])
            .map_err(|e| StorageError::IoError(format!("sqlite delete: {e}")))?;
        memo.borrow_mut().note_delete(key);
        Ok(())
    }
}

impl Storage for SqliteStorage {
    fn storage_cache_namespace(&self) -> usize {
        // What is cached under the namespace is consulted BEFORE the write it spares
        // ("this table's header is stored"), so a transaction lost since the last
        // statement has to be noticed here: noticed by the write itself, the header
        // has already been skipped.
        let mut abandoned = Vec::new();
        if let Ok(mut inner) = self.inner.lock()
            && let Some(inner) = inner.as_mut()
        {
            inner.reconcile_lost_transaction();
            abandoned = std::mem::take(&mut inner.abandoned_cache_namespaces);
        }
        // Outside the connection's lock. Nothing asks for an abandoned namespace again,
        // so what is cached under it would otherwise stay for the life of the process —
        // one generation per lost transaction, and a commit that keeps failing loses
        // one on every flush.
        for namespace in abandoned {
            super::forget_storage_cache_namespace(namespace);
        }
        self.cache_namespace.load(Ordering::Relaxed)
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
                    Self::with_savepoint(&inner.conn, &inner.memo, || {
                        raw_table_put_core(table, key, value, |storage_key, bytes| {
                            Self::set(&inner.conn, &inner.memo, storage_key, bytes)
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
                    Self::with_savepoint(&inner.conn, &inner.memo, || {
                        raw_table_delete_core(table, key, |storage_key| {
                            Self::delete(&inner.conn, &inner.memo, storage_key)
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
                    Self::with_savepoint(&inner.conn, &inner.memo, || {
                        for mutation in mutations {
                            match mutation {
                                RawTableMutation::Put { table, key, value } => {
                                    raw_table_put_core(table, key, value, |storage_key, bytes| {
                                        Self::set(&inner.conn, &inner.memo, storage_key, bytes)
                                    })?;
                                }
                                RawTableMutation::Delete { table, key } => {
                                    raw_table_delete_core(table, key, |storage_key| {
                                        Self::delete(&inner.conn, &inner.memo, storage_key)
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

    /// One savepoint for the whole batch, as `apply_encoded_row_mutation` writes a row's
    /// entries. The trait's default writes entry by entry, and on SQLite every
    /// `raw_table_put` opens its own savepoint, whose first touch of each b-tree page
    /// copies that page to the statement journal.
    fn apply_index_mutations(
        &mut self,
        mutations: &[IndexMutation<'_>],
    ) -> Result<(), StorageError> {
        crate::query_manager::settle_cost::timed(
            &crate::query_manager::settle_cost::STORAGE_WRITE_MICROS,
            || {
                self.with_inner_mut(|inner| {
                    inner.ensure_write_tx()?;
                    Self::with_savepoint(&inner.conn, &inner.memo, || {
                        Self::write_index_mutations(&inner.conn, &inner.memo, mutations)
                    })
                })
            },
        )
    }

    fn load_row_locator(&self, id: ObjectId) -> Result<Option<super::RowLocator>, StorageError> {
        let storage_key =
            key_codec::raw_table_entry_key(super::ROW_LOCATOR_TABLE, &super::metadata_raw_key(id));
        let memoized = self.with_inner(|inner| {
            if !inner.memo_usable() {
                return Ok(None);
            }
            let memoized = inner.memo.borrow().row_locators.get(&storage_key).cloned();
            // Parity harness, as for the prefix keys: the stored bytes still decode to
            // what the memo holds.
            #[cfg(debug_assertions)]
            if let Some(memoized) = &memoized
                && let Ok(stored) = Self::get(&inner.conn, &storage_key)
                && let Ok(stored) = stored
                    .map(|bytes| super::decode_row_locator(&bytes))
                    .transpose()
            {
                debug_assert_eq!(
                    &stored, memoized,
                    "read memo diverged from the stored row locator of {id}"
                );
            }
            Ok(memoized)
        })?;
        if let Some(locator) = memoized {
            return Ok(locator);
        }
        let locator = super::storage_trait::load_row_locator_default(self, id)?;
        self.with_inner(|inner| {
            // Asked again: the read above ran under its own lock. This only says that a
            // transaction is open now — that it is the one the read ran in holds because
            // the storage is used from one thread at a time (the core's lock).
            if !inner.memo_usable() {
                return Ok(());
            }
            let mut memo = inner.memo.borrow_mut();
            if memo.row_locators.len() >= MAX_MEMOIZED_ROW_LOCATORS {
                memo.row_locators.clear();
            }
            memo.row_locators.insert(storage_key, locator.clone());
            Ok(())
        })?;
        Ok(locator)
    }

    fn begin_read_scope(&self) {
        if let Ok(mut inner) = self.lock_inner()
            && let Some(inner) = inner.as_mut()
        {
            inner.begin_read_scope();
        }
    }

    fn end_read_scope(&self) {
        if let Ok(mut inner) = self.lock_inner()
            && let Some(inner) = inner.as_mut()
        {
            inner.end_read_scope();
        }
    }

    fn raw_table_prefix_known_empty(&self, table: &str, prefix: &str) -> bool {
        self.with_prefix_keys(
            table,
            prefix,
            PrefixQuestion::Emptiness,
            |keys| matches!(keys, PrefixKeys::Small(set) if set.is_empty()),
        )
    }

    fn raw_table_key_known_absent(&self, table: &str, prefix: &str, key: &str) -> bool {
        if !key.starts_with(prefix) {
            return false;
        }
        self.with_prefix_keys(table, prefix, PrefixQuestion::Key, |keys| match keys {
            PrefixKeys::Small(set) => !set.contains(&key_codec::raw_table_entry_key(table, key)),
            PrefixKeys::Large | PrefixKeys::Unlisted => false,
        })
    }

    fn memoized_row_raw_table_ids(&self, header_prefix: &str) -> Option<Vec<super::RowRawTableId>> {
        self.with_inner(|inner| {
            if !inner.memo_usable() {
                return Ok(None);
            }
            let ids = inner
                .memo
                .borrow()
                .row_raw_table_ids
                .get(header_prefix)
                .cloned();
            // Parity harness: the ids are the header keys under the prefix.
            #[cfg(debug_assertions)]
            if let Some(ids) = &ids {
                let storage_prefix =
                    key_codec::raw_table_scan_prefix(super::RAW_TABLE_HEADER_TABLE, header_prefix);
                let mut end = storage_prefix.clone();
                key_codec::increment_string(&mut end);
                if let Ok(stored) = Self::scan_range_keys_limited(
                    &inner.conn,
                    &storage_prefix,
                    &end,
                    false,
                    usize::MAX,
                ) {
                    let mut stored: Vec<String> = stored
                        .iter()
                        .filter_map(|key| {
                            key_codec::strip_raw_table_key(super::RAW_TABLE_HEADER_TABLE, key)
                                .map(str::to_string)
                        })
                        .collect();
                    stored.sort();
                    let mut memoized: Vec<String> = ids
                        .iter()
                        .map(|id| id.raw_table_name().to_string())
                        .collect();
                    memoized.sort();
                    debug_assert_eq!(
                        stored, memoized,
                        "read memo diverged from the raw-table headers under {header_prefix:?}"
                    );
                }
            }
            Ok(ids)
        })
        .ok()
        .flatten()
    }

    fn memoize_row_raw_table_ids(&self, header_prefix: &str, ids: &[super::RowRawTableId]) {
        let _ = self.with_inner(|inner| {
            if inner.memo_usable() {
                inner
                    .memo
                    .borrow_mut()
                    .row_raw_table_ids
                    .insert(header_prefix.to_string(), ids.to_vec());
            }
            Ok(())
        });
    }

    fn raw_table_get(&self, table: &str, key: &str) -> Result<Option<Vec<u8>>, StorageError> {
        crate::query_manager::settle_cost::bump(
            &crate::query_manager::settle_cost::STORAGE_READ_OPS,
        );
        crate::query_manager::settle_cost::timed(
            &crate::query_manager::settle_cost::STORAGE_READ_MICROS,
            || {
                self.with_inner(|inner| {
                    raw_table_get_core(table, key, |storage_key| {
                        Self::get(&inner.conn, storage_key)
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
                    raw_table_scan_prefix_core(table, prefix, |storage_prefix| {
                        Self::scan_prefix(&inner.conn, storage_prefix)
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
                    raw_table_scan_prefix_keys_core(table, prefix, |storage_prefix| {
                        Self::scan_prefix_keys(&inner.conn, storage_prefix)
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
            raw_table_scan_range_core(table, start, end, |start_key, end_key| {
                Self::scan_range(&inner.conn, start_key, end_key)
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
            raw_table_scan_range_keys_core(table, start, end, |start_key, end_key| {
                Self::scan_range_keys(&inner.conn, start_key, end_key)
            })
        })
    }

    fn raw_table_scan_range_keys_limited(
        &self,
        table: &str,
        start: Option<&str>,
        end: Option<&str>,
        reverse: bool,
        limit: usize,
    ) -> Result<super::RawTableKeys, StorageError> {
        self.with_inner(|inner| {
            raw_table_scan_range_keys_core(table, start, end, |start_key, end_key| {
                Self::scan_range_keys_limited(&inner.conn, start_key, end_key, reverse, limit)
            })
        })
    }

    fn raw_table_family_keys(
        &self,
        name_prefix: &str,
        after: Option<&str>,
        limit: usize,
    ) -> Result<Vec<String>, StorageError> {
        self.with_inner(|inner| {
            raw_table_family_keys_core(name_prefix, after, limit, |start, end, limit| {
                Self::scan_range_keys_limited(&inner.conn, start, end, false, limit)
            })
        })
    }

    fn raw_table_family_last_key(&self, name_prefix: &str) -> Result<Option<String>, StorageError> {
        self.with_inner(|inner| {
            raw_table_family_last_key_core(name_prefix, |start, end| {
                Self::scan_range_keys_limited(&inner.conn, start, end, true, 1)
            })
        })
    }

    fn store_format_version(&self) -> Result<Option<i32>, StorageError> {
        self.with_inner(|inner| {
            Self::get(&inner.conn, super::STORE_MANIFEST_KEY)?
                .map(|bytes| super::decode_store_manifest(&bytes))
                .transpose()
                .map(|manifest| manifest.map(|manifest| manifest.store_format_version))
        })
    }

    fn set_store_format_version(&mut self, version: i32) -> Result<(), StorageError> {
        let bytes = super::encode_store_manifest(&super::StoreManifest {
            store_kind: super::SQLITE_STORE_KIND.to_string(),
            store_format_version: version,
        })?;
        self.with_inner_mut(|inner| {
            inner.ensure_write_tx()?;
            Self::set(&inner.conn, &inner.memo, super::STORE_MANIFEST_KEY, &bytes)
        })
    }

    fn append_history_region_row_bytes(
        &mut self,
        table: &str,
        rows: &[HistoryRowBytes<'_>],
    ) -> Result<(), StorageError> {
        self.with_inner_mut(|inner| {
            inner.ensure_write_tx()?;
            Self::with_savepoint(&inner.conn, &inner.memo, || {
                append_history_region_row_bytes_core(table, rows, |key, bytes| {
                    Self::set(&inner.conn, &inner.memo, key, bytes)
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
            Self::with_savepoint(&inner.conn, &inner.memo, || {
                upsert_visible_region_row_bytes_core(table, rows, |key, bytes| {
                    Self::set(&inner.conn, &inner.memo, key, bytes)
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
            Self::with_savepoint(&inner.conn, &inner.memo, || {
                let key = super::key_codec::visible_row_raw_table_key(branch, row_id);
                for raw_table in &raw_tables {
                    raw_table_delete_core(raw_table.as_str(), &key, |storage_key| {
                        Self::delete(&inner.conn, &inner.memo, storage_key)
                    })?;
                }
                raw_table_delete_core(
                    super::VISIBLE_ROW_TABLE_LOCATOR_TABLE,
                    &super::visible_row_table_locator_key(branch, row_id),
                    |storage_key| Self::delete(&inner.conn, &inner.memo, storage_key),
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
            let applied = Self::with_savepoint(&inner.conn, &inner.memo, || {
                let mut seen_row_raw_tables = std::collections::HashSet::new();
                for row in encoded_history_rows {
                    if seen_row_raw_tables.insert(row.row_raw_table.clone())
                        && inner
                            .ensured_raw_table_headers
                            .insert(row.row_raw_table.clone())
                    {
                        let header = super::encode_raw_table_header(&super::row_raw_table_header(
                            &row.row_raw_table_id,
                            &row.user_descriptor,
                        ))?;
                        raw_table_put_core(
                            super::RAW_TABLE_HEADER_TABLE,
                            row.row_raw_table.as_str(),
                            &header,
                            |storage_key, bytes| {
                                Self::set(&inner.conn, &inner.memo, storage_key, bytes)
                            },
                        )?;
                    }
                }
                for row in encoded_visible_rows {
                    if seen_row_raw_tables.insert(row.row_raw_table.clone())
                        && inner
                            .ensured_raw_table_headers
                            .insert(row.row_raw_table.clone())
                    {
                        let header = super::encode_raw_table_header(&super::row_raw_table_header(
                            &row.row_raw_table_id,
                            &row.user_descriptor,
                        ))?;
                        raw_table_put_core(
                            super::RAW_TABLE_HEADER_TABLE,
                            row.row_raw_table.as_str(),
                            &header,
                            |storage_key, bytes| {
                                Self::set(&inner.conn, &inner.memo, storage_key, bytes)
                            },
                        )?;
                    }
                }
                if encoded_history_rows
                    .iter()
                    .any(|row| row.needs_exact_locator)
                    && inner
                        .ensured_raw_table_headers
                        .insert(super::HISTORY_ROW_BATCH_TABLE_LOCATOR_TABLE.to_string())
                {
                    let header = super::encode_raw_table_header(&super::RawTableHeader::system(
                        super::STORAGE_KIND_HISTORY_ROW_BATCH_TABLE_LOCATOR,
                        1,
                    ))?;
                    raw_table_put_core(
                        super::RAW_TABLE_HEADER_TABLE,
                        super::HISTORY_ROW_BATCH_TABLE_LOCATOR_TABLE,
                        &header,
                        |storage_key, bytes| {
                            Self::set(&inner.conn, &inner.memo, storage_key, bytes)
                        },
                    )?;
                }
                if encoded_visible_rows
                    .iter()
                    .any(|row| row.needs_exact_locator)
                    && inner
                        .ensured_raw_table_headers
                        .insert(super::VISIBLE_ROW_TABLE_LOCATOR_TABLE.to_string())
                {
                    let header = super::encode_raw_table_header(&super::RawTableHeader::system(
                        super::STORAGE_KIND_VISIBLE_ROW_TABLE_LOCATOR,
                        1,
                    ))?;
                    raw_table_put_core(
                        super::RAW_TABLE_HEADER_TABLE,
                        super::VISIBLE_ROW_TABLE_LOCATOR_TABLE,
                        &header,
                        |storage_key, bytes| {
                            Self::set(&inner.conn, &inner.memo, storage_key, bytes)
                        },
                    )?;
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
                    |key, bytes| Self::set(&inner.conn, &inner.memo, key, bytes),
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
                        |storage_key, bytes| {
                            Self::set(&inner.conn, &inner.memo, storage_key, bytes)
                        },
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
                    |key, bytes| Self::set(&inner.conn, &inner.memo, key, bytes),
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
                    if inner.visible_row_table_locators.get(&cache_key) != Some(&locator) {
                        let locator_bytes = super::encode_exact_row_table_locator(&locator)?;
                        raw_table_put_core(
                            super::VISIBLE_ROW_TABLE_LOCATOR_TABLE,
                            &super::visible_row_table_locator_key(row.branch.as_str(), row.row_id),
                            &locator_bytes,
                            |storage_key, bytes| {
                                Self::set(&inner.conn, &inner.memo, storage_key, bytes)
                            },
                        )?;
                        inner.visible_row_table_locators.insert(cache_key, locator);
                    }
                }

                Self::write_index_mutations(&inner.conn, &inner.memo, index_mutations)
            });
            if applied.is_err() {
                // The rollback took back the headers and the locators this mutation had
                // marked as written; left marked, the next mutation would skip writing
                // them and its rows would sit in a family with no header.
                inner.ensured_raw_table_headers.clear();
                inner.visible_row_table_locators.clear();
            }
            applied
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

    fn flush_wal(&self) -> Result<(), StorageError> {
        let mut inner = self.lock_inner()?;
        if let Some(inner) = inner.as_mut() {
            // Commit the open write transaction so writes land in the WAL
            // and survive a process crash.
            inner.commit_write_tx()?;
            // PASSIVE checkpoint: moves WAL pages into the main db file without
            // blocking concurrent readers.
            inner
                .conn
                .execute_batch("PRAGMA wal_checkpoint(PASSIVE)")
                .map_err(|e| StorageError::IoError(format!("sqlite wal checkpoint: {e}")))?;
        } else if self.closed_with_lost_writes.load(Ordering::Relaxed) {
            return Err(StorageError::IoError(
                "sqlite storage was closed with writes that are not stored".to_string(),
            ));
        }
        Ok(())
    }

    fn flush(&self) -> Result<(), StorageError> {
        self.flush_wal()
    }

    fn close(&self) -> Result<(), StorageError> {
        let Some(mut inner) = self.lock_inner()?.take() else {
            // Closed before. If that close lost writes, this one does not report them
            // stored either.
            return if self.closed_with_lost_writes.load(Ordering::Relaxed) {
                Err(StorageError::IoError(
                    "sqlite storage was closed with writes that are not stored".to_string(),
                ))
            } else {
                Ok(())
            };
        };
        // Commit any pending writes before closing. The connection is dropped either
        // way, and what it could not commit goes with it: later flushes say so.
        if let Err(error) = inner.commit_write_tx() {
            self.closed_with_lost_writes.store(true, Ordering::Relaxed);
            return Err(error);
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

    /// A batch of index mutations goes through one savepoint, as a row's entries do in
    /// `apply_encoded_row_mutation`: an entry that cannot be written takes the batch's
    /// earlier entries back with it. Written entry by entry, each under its own savepoint,
    /// the entries before the failure stayed, and filling a declared index over a
    /// 100k-message table cost 5.5 s of CPU instead of 2.1.
    #[test]
    fn a_batch_of_index_mutations_applies_as_one_unit() {
        use crate::object::ObjectId;
        use crate::query_manager::types::Value;

        let dir = tempfile::TempDir::new().unwrap();
        let mut storage = SqliteStorage::open(dir.path().join("test.sqlite")).unwrap();
        let row_id = ObjectId::new();
        let written = Value::Integer(1);
        // A long value only overflows into a hashed segment; a key too large to write
        // takes a column name longer than the key limit.
        let unkeyable_column = "c".repeat(6 * 1024);
        let mutations = [
            IndexMutation::Insert {
                table: "t",
                column: "c",
                branch: "main",
                value: written.clone(),
                row_id,
            },
            IndexMutation::Insert {
                table: "t",
                column: &unkeyable_column,
                branch: "main",
                value: Value::Integer(2),
                row_id,
            },
        ];

        assert!(matches!(
            storage.apply_index_mutations(&mutations),
            Err(StorageError::IndexKeyTooLarge { .. })
        ));
        assert!(
            !storage
                .index_contains("t", "c", "main", &written, row_id)
                .unwrap(),
            "the entry before the failed one outlived its batch"
        );
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

    fn memoized_small(memo: &ReadMemo, prefix: &str) -> Option<Vec<String>> {
        match memo.prefix_keys.get(prefix)? {
            PrefixKeys::Small(keys) => {
                let mut keys: Vec<String> = keys.iter().cloned().collect();
                keys.sort();
                Some(keys)
            }
            PrefixKeys::Large | PrefixKeys::Unlisted => None,
        }
    }

    /// A written key lands in the key set of every memoized prefix it falls under, and
    /// in no other.
    #[test]
    fn read_memo_follows_a_key_under_each_of_its_memoized_heads() {
        let mut memo = ReadMemo::default();
        for prefix in ["raw:t:", "raw:t:main:", "raw:t:dev:", "raw:u:"] {
            memo.prefix_keys
                .insert(prefix.to_string(), PrefixKeys::Small(HashSet::new()));
        }

        memo.note_set("raw:t:main:row");
        assert_eq!(
            memoized_small(&memo, "raw:t:"),
            Some(vec!["raw:t:main:row".to_string()])
        );
        assert_eq!(
            memoized_small(&memo, "raw:t:main:"),
            Some(vec!["raw:t:main:row".to_string()])
        );
        assert_eq!(memoized_small(&memo, "raw:t:dev:"), Some(vec![]));
        assert_eq!(memoized_small(&memo, "raw:u:"), Some(vec![]));

        assert_eq!(memo.prefix_key_bytes, 2 * "raw:t:main:row".len());

        memo.note_delete("raw:t:main:row");
        assert_eq!(memoized_small(&memo, "raw:t:"), Some(vec![]));
        assert_eq!(memoized_small(&memo, "raw:t:main:"), Some(vec![]));
        assert_eq!(memo.prefix_key_bytes, 0);
    }

    /// Past the limit a key set stops being held, and no later delete brings it back:
    /// the set no longer knows which keys it lost track of.
    #[test]
    fn read_memo_gives_a_prefix_up_past_the_limit_for_good() {
        let mut memo = ReadMemo::default();
        memo.prefix_keys
            .insert("raw:t:".to_string(), PrefixKeys::Small(HashSet::new()));

        for index in 0..MAX_SMALL_PREFIX_KEYS {
            memo.note_set(&format!("raw:t:{index}"));
        }
        assert_eq!(
            memoized_small(&memo, "raw:t:").map(|keys| keys.len()),
            Some(MAX_SMALL_PREFIX_KEYS),
            "a set at the limit is still held exactly"
        );
        // Rewriting a key it already holds does not grow the set.
        memo.note_set("raw:t:0");
        assert!(memoized_small(&memo, "raw:t:").is_some());

        assert_eq!(
            memo.prefix_key_bytes,
            (0..MAX_SMALL_PREFIX_KEYS)
                .map(|index| format!("raw:t:{index}").len())
                .sum::<usize>()
        );

        memo.note_set("raw:t:one-more");
        assert!(matches!(
            memo.prefix_keys.get("raw:t:"),
            Some(PrefixKeys::Large)
        ));
        assert_eq!(
            memo.prefix_key_bytes, 0,
            "a set given up is no longer counted"
        );
        for index in 0..MAX_SMALL_PREFIX_KEYS {
            memo.note_delete(&format!("raw:t:{index}"));
        }
        memo.note_delete("raw:t:one-more");
        assert!(matches!(
            memo.prefix_keys.get("raw:t:"),
            Some(PrefixKeys::Large)
        ));
    }

    /// The key sets are dropped together once the keys they hold between them take too
    /// many bytes — whether a read filled the last set or a write grew it — and a prefix
    /// that was only asked whether it holds anything holds no keys at all.
    #[test]
    fn read_memo_bounds_the_keys_it_holds() {
        let mut memo = ReadMemo::default();
        let per_prefix = MAX_SMALL_PREFIX_KEYS / 2;
        // Fixed-width keys, so the bytes held are a plain product — of a width that does
        // not divide the bound, so the sets filled below stop short of it.
        let key_of = |prefix: usize, key: usize| format!("raw:t{prefix:06}:{key:05}");
        let per_prefix_bytes = per_prefix * key_of(0, 0).len();
        let prefixes = MAX_MEMOIZED_PREFIX_KEY_BYTES / per_prefix_bytes;
        let keys_of = |prefix: usize| -> HashSet<String> {
            (0..per_prefix).map(|key| key_of(prefix, key)).collect()
        };
        let fill = |memo: &mut ReadMemo| {
            for prefix in 0..prefixes {
                memo.memoize_prefix(
                    format!("raw:t{prefix:06}:"),
                    PrefixKeys::Small(keys_of(prefix)),
                );
            }
            assert_eq!(memo.prefix_keys.len(), prefixes);
            assert_eq!(memo.prefix_key_bytes, prefixes * per_prefix_bytes);
        };
        fill(&mut memo);

        memo.memoize_prefix("raw:unlisted:".to_string(), PrefixKeys::Unlisted);
        memo.memoize_prefix("raw:large:".to_string(), PrefixKeys::Large);
        assert_eq!(memo.prefix_key_bytes, prefixes * per_prefix_bytes);

        memo.memoize_prefix(
            format!("raw:t{prefixes:06}:"),
            PrefixKeys::Small(keys_of(prefixes)),
        );
        assert_eq!(
            memo.prefix_keys.len(),
            1,
            "everything held before was dropped"
        );
        assert_eq!(memo.prefix_key_bytes, per_prefix_bytes);

        // Re-reading a prefix replaces its set instead of counting it twice.
        memo.memoize_prefix(
            format!("raw:t{prefixes:06}:"),
            PrefixKeys::Small(HashSet::new()),
        );
        assert_eq!(memo.prefix_key_bytes, 0);

        // Writes grow the sets too: filled to the brim by reads, the memo is pushed over
        // by keys written under prefixes it already holds.
        memo.memoize_prefix("raw:held:".to_string(), PrefixKeys::Small(keys_of(0)));
        assert_ne!(memo.prefix_key_bytes, 0);
        memo.clear();
        assert_eq!(memo.prefix_key_bytes, 0, "a cleared memo holds no bytes");
        fill(&mut memo);
        memo.assert_accounting();
        let room = MAX_MEMOIZED_PREFIX_KEY_BYTES - memo.prefix_key_bytes;
        let written = room / key_of(0, 0).len() + 1;
        assert!(
            (2..per_prefix).contains(&written),
            "several writes fit before the bound, and all of them fit the set they land in"
        );
        for key in 0..written {
            assert_eq!(
                memo.prefix_keys.len(),
                prefixes,
                "dropped before the bound was passed"
            );
            memo.note_set(&key_of(0, per_prefix + key));
        }
        assert!(
            memo.prefix_keys.is_empty(),
            "writes took the memo past its bound"
        );
        assert_eq!(memo.prefix_key_bytes, 0);

        // A prefix remembered as empty, wide or merely non-empty holds no key bytes: how
        // many prefixes are remembered is bounded on its own.
        for prefix in 0..MAX_MEMOIZED_PREFIXES {
            let keys = match prefix % 3 {
                0 => PrefixKeys::Unlisted,
                1 => PrefixKeys::Large,
                // One of them holds a key: giving the prefixes up gives its bytes up too.
                _ if prefix == 2 => {
                    PrefixKeys::Small([format!("raw:u{prefix}:key")].into_iter().collect())
                }
                _ => PrefixKeys::Small(HashSet::new()),
            };
            memo.memoize_prefix(format!("raw:u{prefix}:"), keys);
        }
        assert_eq!(memo.prefix_keys.len(), MAX_MEMOIZED_PREFIXES);
        assert!(memo.prefix_key_bytes > 0);
        // An answer for a prefix already held replaces it.
        memo.memoize_prefix("raw:u0:".to_string(), PrefixKeys::Large);
        assert_eq!(memo.prefix_keys.len(), MAX_MEMOIZED_PREFIXES);
        memo.memoize_prefix("raw:one-more:".to_string(), PrefixKeys::Unlisted);
        assert_eq!(
            memo.prefix_keys.len(),
            1,
            "prefixes holding no keys were remembered without bound"
        );
        assert_eq!(memo.prefix_key_bytes, 0);
        memo.assert_accounting();
    }

    /// A write to a row locator drops that locator from the memo; a write to any raw-table
    /// header drops every memoized list of row raw tables; other writes touch neither.
    #[test]
    fn read_memo_drops_what_a_written_key_decides() {
        let locator_key = key_codec::raw_table_entry_key(super::super::ROW_LOCATOR_TABLE, "row");
        let other_locator_key =
            key_codec::raw_table_entry_key(super::super::ROW_LOCATOR_TABLE, "other");
        let header_key =
            key_codec::raw_table_entry_key(super::super::RAW_TABLE_HEADER_TABLE, "some-family");

        let mut memo = ReadMemo::default();
        memo.row_locators.insert(locator_key.clone(), None);
        memo.row_locators.insert(other_locator_key.clone(), None);
        memo.row_raw_table_ids
            .insert("prefix".to_string(), Vec::new());

        memo.note_set("raw:unrelated:key");
        assert_eq!(memo.row_locators.len(), 2);
        assert_eq!(memo.row_raw_table_ids.len(), 1);

        memo.note_set(&locator_key);
        assert!(!memo.row_locators.contains_key(&locator_key));
        assert!(memo.row_locators.contains_key(&other_locator_key));
        assert_eq!(memo.row_raw_table_ids.len(), 1);

        memo.note_delete(&other_locator_key);
        assert!(memo.row_locators.is_empty());

        memo.note_delete(&header_key);
        assert!(memo.row_raw_table_ids.is_empty());
    }

    /// What the memo answers stays what the table holds through puts and deletes, with the
    /// answers asked for BEFORE the writes, so they are served from the memo afterwards.
    #[test]
    fn a_memoized_prefix_follows_the_writes_under_it() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut storage = SqliteStorage::open(dir.path().join("test.sqlite")).unwrap();

        // Outside a transaction nothing is known, however empty the store is.
        assert!(!storage.raw_table_prefix_known_empty("family", "main:"));
        assert!(!storage.raw_table_key_known_absent("family", "main:", "main:row"));

        storage.begin_read_scope();
        assert!(storage.raw_table_prefix_known_empty("family", "main:"));
        assert!(storage.raw_table_key_known_absent("family", "main:", "main:row"));

        storage.raw_table_put("family", "main:row", b"v").unwrap();
        assert!(!storage.raw_table_prefix_known_empty("family", "main:"));
        assert!(!storage.raw_table_key_known_absent("family", "main:", "main:row"));
        assert!(storage.raw_table_key_known_absent("family", "main:", "main:other"));
        // Another branch of the same family was never asked about and is read now.
        assert!(storage.raw_table_prefix_known_empty("family", "dev:"));

        storage.raw_table_delete("family", "main:row").unwrap();
        assert!(storage.raw_table_prefix_known_empty("family", "main:"));
        assert!(storage.raw_table_key_known_absent("family", "main:", "main:row"));

        // A key outside the prefix is not something the prefix's keys can speak for.
        assert!(!storage.raw_table_key_known_absent("family", "main:", "dev:row"));
        storage.end_read_scope();

        // The write made in the scope keeps its transaction open until the flush, and
        // the memo with it; after the flush nothing is known again.
        assert!(storage.raw_table_key_known_absent("family", "main:", "main:row"));
        storage.flush().unwrap();
        assert!(!storage.raw_table_key_known_absent("family", "main:", "main:row"));
    }

    /// Asking whether a prefix holds anything reads one key of it and keeps none; asking
    /// about one key afterwards reads the set.
    #[test]
    fn an_emptiness_question_does_not_hold_the_keys() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut storage = SqliteStorage::open(dir.path().join("test.sqlite")).unwrap();
        for index in 0..10 {
            storage
                .raw_table_put("family", &format!("main:{index}"), b"v")
                .unwrap();
        }
        let held = |storage: &SqliteStorage| {
            storage
                .with_inner(|inner| Ok(inner.memo.borrow().prefix_key_bytes))
                .unwrap()
        };

        assert!(!storage.raw_table_prefix_known_empty("family", "main:"));
        assert_eq!(held(&storage), 0);
        // Still not known to be empty once every key is gone: it was never listed.
        for index in 0..10 {
            storage
                .raw_table_delete("family", &format!("main:{index}"))
                .unwrap();
        }
        assert!(!storage.raw_table_prefix_known_empty("family", "main:"));

        storage.raw_table_put("family", "main:row", b"v").unwrap();
        assert!(storage.raw_table_key_known_absent("family", "main:", "main:other"));
        assert!(!storage.raw_table_key_known_absent("family", "main:", "main:row"));
        assert_eq!(
            held(&storage),
            key_codec::raw_table_entry_key("family", "main:row").len()
        );
        storage.raw_table_delete("family", "main:row").unwrap();
        assert!(storage.raw_table_prefix_known_empty("family", "main:"));
    }

    /// A prefix holding more keys than the memo keeps answers "not known" for every key,
    /// present or not, whether it was read that large or grew that large afterwards.
    #[test]
    fn a_large_prefix_is_never_answered_from_the_memo() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut storage = SqliteStorage::open(dir.path().join("test.sqlite")).unwrap();

        for index in 0..=MAX_SMALL_PREFIX_KEYS {
            storage
                .raw_table_put("read-large", &format!("main:{index}"), b"v")
                .unwrap();
        }
        assert!(in_transaction(&storage));
        assert!(!storage.raw_table_key_known_absent("read-large", "main:", "main:absent"));
        assert!(!storage.raw_table_prefix_known_empty("read-large", "main:"));

        assert!(storage.raw_table_prefix_known_empty("grown-large", "main:"));
        for index in 0..=MAX_SMALL_PREFIX_KEYS {
            storage
                .raw_table_put("grown-large", &format!("main:{index}"), b"v")
                .unwrap();
        }
        assert!(!storage.raw_table_key_known_absent("grown-large", "main:", "main:absent"));
        for index in 0..=MAX_SMALL_PREFIX_KEYS {
            storage
                .raw_table_delete("grown-large", &format!("main:{index}"))
                .unwrap();
        }
        assert!(
            !storage.raw_table_prefix_known_empty("grown-large", "main:"),
            "a prefix that outgrew the memo is not known again until the memo is rebuilt"
        );
        assert_eq!(
            storage
                .raw_table_scan_prefix_keys("grown-large", "main:")
                .unwrap()
                .len(),
            0
        );
    }

    /// A savepoint that rolls back puts a deleted key back without a write the memo sees.
    /// The memo must not go on answering that the key is gone.
    #[test]
    fn a_rolled_back_delete_is_not_remembered_as_a_delete() {
        use crate::object::ObjectId;
        use crate::query_manager::types::Value;

        let dir = tempfile::TempDir::new().unwrap();
        let mut storage = SqliteStorage::open(dir.path().join("test.sqlite")).unwrap();
        let row_id = ObjectId::new();
        let value = Value::Integer(1);
        let raw_table = key_codec::index_raw_table("t", "c", "main");
        let entry_key = key_codec::index_entry_key("t", "c", "main", &value, row_id).unwrap();

        storage
            .apply_index_mutations(&[IndexMutation::Insert {
                table: "t",
                column: "c",
                branch: "main",
                value: value.clone(),
                row_id,
            }])
            .unwrap();
        // Memoize the index's keys while the entry is there.
        assert!(!storage.raw_table_key_known_absent(&raw_table, "", &entry_key));

        let unkeyable_column = "c".repeat(6 * 1024);
        let failed = storage.apply_index_mutations(&[
            IndexMutation::Remove {
                table: "t",
                column: "c",
                branch: "main",
                value: value.clone(),
                row_id,
            },
            IndexMutation::Insert {
                table: "t",
                column: &unkeyable_column,
                branch: "main",
                value: Value::Integer(2),
                row_id,
            },
        ]);
        assert!(matches!(failed, Err(StorageError::IndexKeyTooLarge { .. })));

        assert!(
            storage
                .index_contains("t", "c", "main", &value, row_id)
                .unwrap(),
            "the rollback kept the entry"
        );
        assert!(
            !storage.raw_table_key_known_absent(&raw_table, "", &entry_key),
            "the memo still answers that an entry the rollback put back is absent"
        );
        assert!(!storage.raw_table_prefix_known_empty(&raw_table, ""));
        assert_eq!(storage.index_lookup("t", "c", "main", &value), vec![row_id]);
    }

    /// The same for a rolled-back insert: an index the memo knew to be empty is still
    /// known to be empty, not "holding" an entry that was taken back.
    #[test]
    fn a_rolled_back_insert_is_not_remembered_as_an_insert() {
        use crate::object::ObjectId;
        use crate::query_manager::types::Value;

        let dir = tempfile::TempDir::new().unwrap();
        let mut storage = SqliteStorage::open(dir.path().join("test.sqlite")).unwrap();
        let row_id = ObjectId::new();
        let raw_table = key_codec::index_raw_table("t", "c", "main");
        storage.begin_read_scope();
        assert!(storage.raw_table_prefix_known_empty(&raw_table, ""));

        let unkeyable_column = "c".repeat(6 * 1024);
        let failed = storage.apply_index_mutations(&[
            IndexMutation::Insert {
                table: "t",
                column: "c",
                branch: "main",
                value: Value::Integer(1),
                row_id,
            },
            IndexMutation::Insert {
                table: "t",
                column: &unkeyable_column,
                branch: "main",
                value: Value::Integer(2),
                row_id,
            },
        ]);
        assert!(matches!(failed, Err(StorageError::IndexKeyTooLarge { .. })));
        assert!(storage.raw_table_prefix_known_empty(&raw_table, ""));
        storage.end_read_scope();
    }

    /// A row locator is answered from the memo until it is rewritten or removed, and then
    /// from the table again.
    #[test]
    fn a_memoized_row_locator_follows_its_writes() {
        use crate::object::ObjectId;
        use crate::storage::RowLocator;

        let dir = tempfile::TempDir::new().unwrap();
        let mut storage = SqliteStorage::open(dir.path().join("test.sqlite")).unwrap();
        let id = ObjectId::new();
        let locator = |table: &str| RowLocator {
            table: table.into(),
            origin_schema_hash: None,
        };

        assert_eq!(storage.load_row_locator(id).unwrap(), None);
        storage
            .put_row_locator(id, Some(&locator("users")))
            .unwrap();
        assert_eq!(
            storage.load_row_locator(id).unwrap(),
            Some(locator("users"))
        );
        storage
            .put_row_locator(id, Some(&locator("chats")))
            .unwrap();
        assert_eq!(
            storage.load_row_locator(id).unwrap(),
            Some(locator("chats"))
        );
        storage.put_row_locator(id, None).unwrap();
        assert_eq!(storage.load_row_locator(id).unwrap(), None);
    }

    fn in_transaction(storage: &SqliteStorage) -> bool {
        storage
            .with_inner(|inner| Ok(!inner.conn.is_autocommit()))
            .unwrap()
    }

    fn visible_to_another_connection(path: &Path, table: &str, key: &str) -> bool {
        let conn = rusqlite::Connection::open(path).unwrap();
        let storage_key = key_codec::raw_table_entry_key(table, key);
        conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM kv WHERE key = ?1)",
            rusqlite::params![storage_key.as_bytes()],
            |row| row.get::<_, i64>(0),
        )
        .unwrap()
            != 0
    }

    /// A scope of reads holds one transaction for as long as the outermost scope is open,
    /// and leaves none behind.
    #[test]
    fn a_read_scope_holds_one_transaction_and_closes_it() {
        use crate::query_manager::settle_cost::{STORAGE_READ_SCOPES, SettleCounts};
        let _ = &STORAGE_READ_SCOPES;

        let dir = tempfile::TempDir::new().unwrap();
        let storage = SqliteStorage::open(dir.path().join("test.sqlite")).unwrap();
        assert!(!in_transaction(&storage));

        let before = SettleCounts::snapshot();
        storage.begin_read_scope();
        assert!(in_transaction(&storage));
        storage.begin_read_scope();
        assert_eq!(storage.raw_table_get("family", "main:row").unwrap(), None);
        storage.end_read_scope();
        assert!(in_transaction(&storage), "the outer scope is still open");
        storage.end_read_scope();
        assert!(!in_transaction(&storage));
        // Other tests of this process open scopes too; this one opened at least its own.
        assert!(SettleCounts::snapshot().since(before).storage_read_scopes >= 1);

        // An unmatched end is ignored, and the next scope still opens.
        storage.end_read_scope();
        storage.begin_read_scope();
        assert!(in_transaction(&storage));
        storage.end_read_scope();
        assert!(!in_transaction(&storage));

        // Nothing is left for a flush to trip over.
        storage.flush().unwrap();
    }

    /// A write made inside a read scope is durable exactly when it would have been
    /// without the scope: at the next flush, not when the scope ends.
    #[test]
    fn a_write_inside_a_read_scope_waits_for_the_flush() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("test.sqlite");
        let mut storage = SqliteStorage::open(&path).unwrap();

        storage.begin_read_scope();
        storage.raw_table_put("family", "main:row", b"v").unwrap();
        assert_eq!(
            storage.raw_table_get("family", "main:row").unwrap(),
            Some(b"v".to_vec()),
            "a read in the scope sees the scope's own write"
        );
        storage.end_read_scope();

        assert!(in_transaction(&storage), "the write is still uncommitted");
        assert!(!visible_to_another_connection(&path, "family", "main:row"));
        storage.flush().unwrap();
        assert!(!in_transaction(&storage));
        assert!(visible_to_another_connection(&path, "family", "main:row"));

        // A scope opened while a write is pending reads in that transaction and leaves it
        // open.
        storage
            .raw_table_put("family", "main:second", b"v")
            .unwrap();
        storage.begin_read_scope();
        storage.end_read_scope();
        assert!(in_transaction(&storage));
        assert!(!visible_to_another_connection(
            &path,
            "family",
            "main:second"
        ));
        storage.flush().unwrap();
        assert!(visible_to_another_connection(
            &path,
            "family",
            "main:second"
        ));
    }

    /// A write that lands while a read scope is open and is never flushed is lost with
    /// the connection, as an unflushed write outside a scope is.
    #[test]
    fn an_unflushed_write_inside_a_read_scope_does_not_outlive_the_connection() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("test.sqlite");
        let mut storage = SqliteStorage::open(&path).unwrap();
        storage
            .raw_table_put("family", "main:flushed", b"v")
            .unwrap();
        storage.flush().unwrap();

        storage.begin_read_scope();
        storage
            .raw_table_put("family", "main:unflushed", b"v")
            .unwrap();
        storage.end_read_scope();
        // Dropped without a flush or a close: the open transaction rolls back.
        drop(storage);

        let reopened = SqliteStorage::open(&path).unwrap();
        assert!(
            reopened
                .raw_table_get("family", "main:flushed")
                .unwrap()
                .is_some()
        );
        assert_eq!(
            reopened.raw_table_get("family", "main:unflushed").unwrap(),
            None
        );
    }

    /// Another connection's commit passes through no code of this one. What this one
    /// remembered about the keys must not outlive it: the app has run several runtimes on
    /// one store file, and a row one of them wrote was then "known absent" to the others.
    #[test]
    fn another_connections_commit_is_seen_past_the_memo() {
        use crate::object::ObjectId;
        use crate::query_manager::types::Value;
        use crate::storage::RowLocator;

        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("test.sqlite");
        let mut ours = SqliteStorage::open(&path).unwrap();
        let mut theirs = SqliteStorage::open(&path).unwrap();
        let row_id = ObjectId::new();
        let value = Value::Integer(1);
        let index_raw_table = key_codec::index_raw_table("t", "c", "main");

        // Everything the memo holds, learned while the store is empty.
        ours.begin_read_scope();
        assert!(ours.raw_table_key_known_absent("family", "main:", "main:row"));
        assert!(ours.raw_table_prefix_known_empty(&index_raw_table, ""));
        assert_eq!(ours.load_row_locator(row_id).unwrap(), None);
        ours.end_read_scope();

        theirs.raw_table_put("family", "main:row", b"v").unwrap();
        theirs
            .apply_index_mutations(&[IndexMutation::Insert {
                table: "t",
                column: "c",
                branch: "main",
                value: value.clone(),
                row_id,
            }])
            .unwrap();
        let locator = RowLocator {
            table: "users".into(),
            origin_schema_hash: None,
        };
        theirs.put_row_locator(row_id, Some(&locator)).unwrap();
        theirs.flush().unwrap();

        // Outside a transaction the memo is not asked at all.
        assert!(!ours.raw_table_key_known_absent("family", "main:", "main:row"));
        assert_eq!(ours.index_lookup("t", "c", "main", &value), vec![row_id]);
        assert_eq!(
            ours.load_row_locator(row_id).unwrap(),
            Some(locator.clone())
        );

        // Inside the next one it is, and it has been dropped.
        ours.begin_read_scope();
        assert!(
            !ours.raw_table_key_known_absent("family", "main:", "main:row"),
            "a key another connection committed is still remembered as absent"
        );
        assert!(!ours.raw_table_prefix_known_empty(&index_raw_table, ""));
        assert_eq!(ours.index_lookup("t", "c", "main", &value), vec![row_id]);
        assert_eq!(ours.load_row_locator(row_id).unwrap(), Some(locator));
        assert!(ours.raw_table_get("family", "main:row").unwrap().is_some());
        ours.end_read_scope();

        // The same through a write transaction: it starts on the other's commit too.
        theirs.raw_table_delete("family", "main:row").unwrap();
        theirs.raw_table_put("family", "main:second", b"v").unwrap();
        theirs.flush().unwrap();
        ours.raw_table_put("family", "main:third", b"v").unwrap();
        assert!(ours.raw_table_key_known_absent("family", "main:", "main:row"));
        assert!(!ours.raw_table_key_known_absent("family", "main:", "main:second"));
        assert!(!ours.raw_table_key_known_absent("family", "main:", "main:third"));
        ours.flush().unwrap();
    }

    /// Inside a transaction the snapshot is fixed, so the memo and the table agree there
    /// even while another connection commits: neither sees the commit until the
    /// transaction ends.
    #[test]
    fn a_read_scope_reads_one_snapshot() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("test.sqlite");
        let ours = SqliteStorage::open(&path).unwrap();
        let mut theirs = SqliteStorage::open(&path).unwrap();

        ours.begin_read_scope();
        assert!(ours.raw_table_key_known_absent("family", "main:", "main:row"));
        theirs.raw_table_put("family", "main:row", b"v").unwrap();
        theirs.flush().unwrap();
        assert_eq!(ours.raw_table_get("family", "main:row").unwrap(), None);
        assert!(ours.raw_table_key_known_absent("family", "main:", "main:row"));
        ours.end_read_scope();

        assert!(ours.raw_table_get("family", "main:row").unwrap().is_some());
    }

    /// A write inside a read scope starts its own transaction instead of upgrading the
    /// scope's snapshot, which SQLite refuses outright — without asking the busy handler —
    /// once another connection has committed past that snapshot.
    #[test]
    fn a_write_inside_a_read_scope_survives_another_connections_commit() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("test.sqlite");
        let mut ours = SqliteStorage::open(&path).unwrap();
        let mut theirs = SqliteStorage::open(&path).unwrap();

        ours.begin_read_scope();
        assert_eq!(ours.raw_table_get("family", "main:theirs").unwrap(), None);
        theirs.raw_table_put("family", "main:theirs", b"v").unwrap();
        theirs.flush().unwrap();

        ours.raw_table_put("family", "main:ours", b"v")
            .expect("the write starts on the current state of the store");
        // The rest of the scope reads in the write's transaction, the other's commit
        // included.
        assert!(
            ours.raw_table_get("family", "main:theirs")
                .unwrap()
                .is_some()
        );
        assert!(!ours.raw_table_key_known_absent("family", "main:", "main:theirs"));
        ours.end_read_scope();
        ours.flush().unwrap();

        assert!(visible_to_another_connection(&path, "family", "main:ours"));
        assert!(visible_to_another_connection(
            &path,
            "family",
            "main:theirs"
        ));
    }

    fn assert_memo_accounting(storage: &SqliteStorage) {
        storage
            .with_inner(|inner| {
                inner.memo.borrow().assert_accounting();
                Ok(())
            })
            .unwrap();
    }

    fn holds_an_unlisted_prefix(storage: &SqliteStorage) -> bool {
        storage
            .with_inner(|inner| {
                Ok(inner
                    .memo
                    .borrow()
                    .prefix_keys
                    .values()
                    .any(|keys| matches!(keys, PrefixKeys::Unlisted)))
            })
            .unwrap()
    }

    fn remembers_any_prefix(storage: &SqliteStorage) -> bool {
        storage
            .with_inner(|inner| Ok(!inner.memo.borrow().prefix_keys.is_empty()))
            .unwrap()
    }

    /// What an I/O error inside a statement does to the open transaction: SQLite rolls it
    /// back and the connection is in autocommit again, with nothing reported.
    fn lose_the_open_transaction(storage: &SqliteStorage) {
        storage
            .with_inner(|inner| {
                inner.conn.execute_batch("ROLLBACK").unwrap();
                Ok(())
            })
            .unwrap();
    }

    /// SQLite can end a transaction by itself, taking its writes with it. The storage
    /// must notice and stop answering from what it remembered about those writes — and
    /// it must never again report a flush as done: the engine holds the lost writes in
    /// memory as stored, and a flush that succeeded would open the durability barrier
    /// and confirm them to whoever sent them. Only a reopened store flushes again.
    #[test]
    fn a_lost_write_transaction_fails_every_flush_after_it() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("test.sqlite");
        let mut storage = SqliteStorage::open(&path).unwrap();
        storage.raw_table_put("family", "main:kept", b"v").unwrap();
        storage.flush().unwrap();

        storage.raw_table_delete("family", "main:kept").unwrap();
        storage.raw_table_put("family", "main:lost", b"v").unwrap();
        assert!(storage.raw_table_key_known_absent("family", "main:", "main:kept"));
        lose_the_open_transaction(&storage);

        assert!(
            !storage.raw_table_key_known_absent("family", "main:", "main:kept"),
            "a delete that was rolled back is still remembered"
        );
        assert!(
            storage
                .raw_table_get("family", "main:kept")
                .unwrap()
                .is_some()
        );
        assert_eq!(storage.raw_table_get("family", "main:lost").unwrap(), None);

        // Writes go on, and reach the file at the next flush like any others.
        storage.raw_table_put("family", "main:after", b"v").unwrap();
        assert!(!storage.raw_table_key_known_absent("family", "main:", "main:kept"));
        assert!(storage.raw_table_key_known_absent("family", "main:", "main:lost"));
        for attempt in 0..3 {
            assert!(
                storage.flush().is_err(),
                "flush {attempt} after a lost write transaction reported success"
            );
        }
        assert!(visible_to_another_connection(&path, "family", "main:after"));
        assert!(visible_to_another_connection(&path, "family", "main:kept"));
        assert!(!visible_to_another_connection(&path, "family", "main:lost"));
        storage.raw_table_put("family", "main:later", b"v").unwrap();
        assert!(storage.flush().is_err());
        assert!(visible_to_another_connection(&path, "family", "main:later"));

        for attempt in 0..2 {
            assert!(
                storage.close().is_err(),
                "close {attempt} reported the lost writes as stored"
            );
        }
        // The connection that knew about the loss is gone; the storage still does.
        for attempt in 0..2 {
            assert!(
                storage.flush().is_err(),
                "flush {attempt} after the close reported the lost writes as stored"
            );
        }
        let reopened = SqliteStorage::open(&path).unwrap();
        reopened.flush().expect("a reopened store flushes");
        assert!(
            reopened
                .raw_table_get("family", "main:later")
                .unwrap()
                .is_some()
        );
        reopened.close().unwrap();
        reopened
            .flush()
            .expect("a store closed with everything committed has nothing left to flush");
    }

    /// What a lost transaction took is judged against the count of changes when the
    /// TRANSACTION began, not when its last write did: a write that changes nothing —
    /// a delete of a key that is not there — coming after one that did must not make
    /// the transaction look empty.
    #[test]
    fn a_write_that_changed_nothing_does_not_hide_the_ones_lost_with_it() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("test.sqlite");
        let mut storage = SqliteStorage::open(&path).unwrap();
        storage.raw_table_put("family", "main:kept", b"v").unwrap();
        storage.flush().unwrap();

        storage.raw_table_put("family", "main:lost", b"v").unwrap();
        storage.raw_table_delete("family", "main:absent").unwrap();
        lose_the_open_transaction(&storage);

        assert!(
            storage.flush().is_err(),
            "a flush reported a lost write as stored"
        );
        assert!(!visible_to_another_connection(&path, "family", "main:lost"));
        assert!(storage.close().is_err());
    }

    /// A write transaction lost before it changed anything took nothing with it: the
    /// operation that failed said so itself, and no write reported as done is missing.
    /// Refusing every flush from then on would brick a store that lost nothing.
    #[test]
    fn a_write_transaction_lost_before_it_changed_anything_loses_nothing() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("test.sqlite");
        let mut storage = SqliteStorage::open(&path).unwrap();
        storage.raw_table_put("family", "main:kept", b"v").unwrap();
        storage.flush().unwrap();

        // Opens the write transaction and changes nothing: there is no such key.
        let namespace = storage.storage_cache_namespace();
        storage.raw_table_delete("family", "main:absent").unwrap();
        assert!(in_transaction(&storage));
        lose_the_open_transaction(&storage);

        storage.raw_table_put("family", "main:after", b"v").unwrap();
        assert_eq!(
            storage.storage_cache_namespace(),
            namespace,
            "a transaction that wrote nothing made the caches above the storage start over"
        );
        storage.flush().expect("nothing was lost");
        assert!(visible_to_another_connection(&path, "family", "main:after"));
        storage.close().expect("nothing was lost");
    }

    /// The caches above the storage also remember what was WRITTEN through it — a
    /// raw-table header, a catalogued schema. A write transaction lost with such a write in
    /// it takes the stored half away, and the next write must store it again: a header
    /// skipped because "it is there" leaves the reopened store with rows in a table that
    /// has no header, which no read and no further write of that table gets past.
    #[test]
    fn a_header_written_in_a_lost_transaction_is_written_again() {
        use crate::object::ObjectId;
        use crate::storage::RowLocator;

        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("test.sqlite");
        let mut storage = SqliteStorage::open(&path).unwrap();
        let locator = RowLocator {
            table: "users".into(),
            origin_schema_hash: None,
        };
        let (lost, kept) = (ObjectId::new(), ObjectId::new());

        storage.put_row_locator(lost, Some(&locator)).unwrap();
        let abandoned = storage.storage_cache_namespace();
        assert!(super::super::storage_cache_entries_in_namespace(abandoned)[0] > 0);
        lose_the_open_transaction(&storage);
        storage.put_row_locator(kept, Some(&locator)).unwrap();
        assert_ne!(storage.storage_cache_namespace(), abandoned);
        assert_eq!(
            super::super::storage_cache_entries_in_namespace(abandoned),
            [0; 5],
            "what was cached for the store before the loss is kept for good"
        );
        assert_eq!(
            storage.load_row_locator(kept).unwrap(),
            Some(locator.clone())
        );
        assert!(storage.flush().is_err());
        assert!(storage.close().is_err());

        let mut reopened = SqliteStorage::open(&path).unwrap();
        assert_eq!(reopened.load_row_locator(lost).unwrap(), None);
        assert_eq!(
            reopened
                .load_row_locator(kept)
                .expect("the locator table lost its header"),
            Some(locator.clone())
        );
        reopened
            .put_row_locator(ObjectId::new(), Some(&locator))
            .expect("the locator table takes no further rows");
        reopened.flush().unwrap();
    }

    /// A commit that keeps failing — a full disk — loses a transaction on every flush,
    /// and each loss leaves a cache namespace behind. Nothing asks for one again, so what
    /// is cached under it has to go with it, or the process keeps one generation of
    /// headers and descriptors per failed flush.
    #[test]
    fn transactions_lost_one_after_another_leave_nothing_cached_behind() {
        use crate::object::ObjectId;
        use crate::storage::RowLocator;

        let dir = tempfile::TempDir::new().unwrap();
        let mut storage = SqliteStorage::open(dir.path().join("test.sqlite")).unwrap();
        let locator = RowLocator {
            table: "users".into(),
            origin_schema_hash: None,
        };

        // Something under the namespace in each of the five caches: a header (the
        // write), a validated table (the read back), and the three catalogue lookups.
        let fill = |storage: &mut SqliteStorage| {
            let id = ObjectId::new();
            storage.put_row_locator(id, Some(&locator)).unwrap();
            assert_eq!(storage.load_row_locator(id).unwrap(), Some(locator.clone()));
            super::super::cache_catalogue_user_descriptor_with_storage(
                storage,
                "users",
                crate::query_manager::types::SchemaHash::from_bytes([7; 32]),
                Arc::new(crate::query_manager::types::RowDescriptor::new(Vec::new())),
            );
            super::super::schema_hashes_matching_branch(storage, "main").unwrap();
            super::super::catalogue_row_descriptors_for_table(storage, "users").unwrap();
            let namespace = storage.storage_cache_namespace();
            let held = super::super::storage_cache_entries_in_namespace(namespace);
            assert!(
                held.iter().all(|entries| *entries > 0),
                "a cache holds nothing for the storage: {held:?}"
            );
            namespace
        };

        let mut left = Vec::new();
        for _ in 0..16 {
            left.push(fill(&mut storage));
            lose_the_open_transaction(&storage);
        }
        let current = fill(&mut storage);
        left.dedup();
        assert_eq!(left.len(), 16, "a loss kept the namespace it happened in");
        for namespace in left {
            assert_ne!(namespace, current);
            assert_eq!(
                super::super::storage_cache_entries_in_namespace(namespace),
                [0; 5]
            );
        }

        // Two losses with no read of the namespace between them: both are left behind.
        // Raw writes ask for no namespace, so nothing purges the first before the second.
        let first = fill(&mut storage);
        lose_the_open_transaction(&storage);
        storage.raw_table_put("family", "main:a", b"v").unwrap();
        lose_the_open_transaction(&storage);
        storage.raw_table_put("family", "main:b", b"v").unwrap();
        let pending = storage
            .inner
            .lock()
            .unwrap()
            .as_ref()
            .map(|inner| inner.abandoned_cache_namespaces.clone())
            .unwrap();
        assert_eq!(pending.len(), 2, "a namespace read came between the losses");
        assert_eq!(pending[0], first);
        let second = storage.storage_cache_namespace();
        assert_ne!(first, second);
        assert_eq!(
            super::super::storage_cache_entries_in_namespace(first),
            [0; 5]
        );
    }

    /// The same loss discovered by the flush itself: a COMMIT that fails and leaves the
    /// transaction rolled back, which is what a full disk does. That flush fails, and so
    /// does every one after it — the retry finds no transaction open and would otherwise
    /// have nothing to fail on.
    #[test]
    fn a_commit_that_fails_and_rolls_back_fails_every_flush_after_it() {
        // SQLite turns a COMMIT into a ROLLBACK when its commit hook returns non-zero.
        unsafe extern "C" fn refuse(_: *mut std::ffi::c_void) -> std::ffi::c_int {
            1
        }
        fn refuse_commits(storage: &SqliteStorage, refuse_them: bool) {
            storage
                .with_inner(|inner| {
                    // SAFETY: the handle is this connection's, live for the call; the
                    // hook is a plain function taking no state.
                    unsafe {
                        rusqlite::ffi::sqlite3_commit_hook(
                            inner.conn.handle(),
                            refuse_them.then_some(refuse as unsafe extern "C" fn(_) -> _),
                            std::ptr::null_mut(),
                        );
                    }
                    Ok(())
                })
                .unwrap();
        }

        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("test.sqlite");
        let mut storage = SqliteStorage::open(&path).unwrap();
        storage.raw_table_put("family", "main:kept", b"v").unwrap();
        storage.flush().unwrap();

        storage.raw_table_put("family", "main:lost", b"v").unwrap();
        assert!(!storage.raw_table_key_known_absent("family", "main:", "main:lost"));
        refuse_commits(&storage, true);
        assert!(
            storage.flush().is_err(),
            "the refused commit was reported as done"
        );
        refuse_commits(&storage, false);

        assert!(!in_transaction(&storage));
        assert_eq!(storage.raw_table_get("family", "main:lost").unwrap(), None);
        for attempt in 0..3 {
            assert!(
                storage.flush().is_err(),
                "retry {attempt} of a flush whose writes were rolled back reported success"
            );
        }
        // What it remembered of the write went with the write.
        storage.raw_table_put("family", "main:after", b"v").unwrap();
        assert!(storage.raw_table_key_known_absent("family", "main:", "main:lost"));
        assert!(storage.flush().is_err());
        assert!(visible_to_another_connection(&path, "family", "main:after"));
    }

    /// A read scope's transaction lost the same way took no write with it: nothing is
    /// owed, and flushes go on succeeding.
    #[test]
    fn a_lost_read_scope_transaction_loses_nothing() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("test.sqlite");
        let mut storage = SqliteStorage::open(&path).unwrap();
        storage.raw_table_put("family", "main:kept", b"v").unwrap();
        storage.flush().unwrap();

        storage.begin_read_scope();
        assert!(storage.raw_table_key_known_absent("family", "main:", "main:other"));
        lose_the_open_transaction(&storage);
        assert!(
            !storage.raw_table_key_known_absent("family", "main:", "main:other"),
            "the memo answered outside a transaction"
        );
        storage.end_read_scope();
        assert!(!in_transaction(&storage));

        storage.raw_table_put("family", "main:last", b"v").unwrap();
        storage.flush().expect("a flush after a lost read scope");
        assert!(visible_to_another_connection(&path, "family", "main:last"));
        storage.flush().expect("and the one after it");
    }

    /// A scope begun on an unchanged store keeps what the previous one remembered, and
    /// its snapshot is taken right there — by the data-version read, before any read of
    /// ours. A commit that lands after that and before our first read is in neither the
    /// memo nor the snapshot, so the two agree; were the snapshot taken at the first read
    /// instead, the table would show a key the memo still calls absent.
    #[test]
    fn a_kept_memo_and_the_scopes_snapshot_are_of_the_same_moment() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("test.sqlite");
        let ours = SqliteStorage::open(&path).unwrap();
        let mut theirs = SqliteStorage::open(&path).unwrap();

        ours.begin_read_scope();
        assert!(ours.raw_table_key_known_absent("family", "main:", "main:row"));
        ours.end_read_scope();

        ours.begin_read_scope();
        assert!(
            remembers_any_prefix(&ours),
            "a scope begun on an unchanged store starts with nothing remembered"
        );
        theirs.raw_table_put("family", "main:row", b"v").unwrap();
        theirs.flush().unwrap();
        // Nothing has been read through `ours` in this scope yet.
        assert_eq!(
            ours.raw_table_get("family", "main:row").unwrap(),
            None,
            "the scope's snapshot was taken at its first read, after the memo was kept"
        );
        assert!(ours.raw_table_key_known_absent("family", "main:", "main:row"));
        ours.end_read_scope();

        ours.begin_read_scope();
        assert!(
            !remembers_any_prefix(&ours),
            "another connection's commit left the memo standing"
        );
        assert!(!ours.raw_table_key_known_absent("family", "main:", "main:row"));
        assert!(ours.raw_table_get("family", "main:row").unwrap().is_some());
        ours.end_read_scope();
    }

    /// The row-locator memo is bounded: past its limit it starts over.
    #[test]
    fn the_row_locator_memo_is_bounded() {
        use crate::object::ObjectId;

        let dir = tempfile::TempDir::new().unwrap();
        let storage = SqliteStorage::open(dir.path().join("test.sqlite")).unwrap();
        let held = |storage: &SqliteStorage| {
            storage
                .with_inner(|inner| Ok(inner.memo.borrow().row_locators.len()))
                .unwrap()
        };

        storage.begin_read_scope();
        for _ in 0..MAX_MEMOIZED_ROW_LOCATORS {
            assert_eq!(storage.load_row_locator(ObjectId::new()).unwrap(), None);
        }
        assert_eq!(held(&storage), MAX_MEMOIZED_ROW_LOCATORS);
        assert_eq!(storage.load_row_locator(ObjectId::new()).unwrap(), None);
        assert_eq!(held(&storage), 1);
        storage.end_read_scope();
    }

    /// A row mutation that rolls back takes its raw-table header and its exact locator
    /// back with it. The next mutation must write both again rather than trust that the
    /// failed one did.
    #[test]
    fn a_rolled_back_row_mutation_does_not_leave_its_header_marked_written() {
        use crate::object::ObjectId;
        use crate::query_manager::types::{RowDescriptor, SchemaBuilder, TableSchema, Value};
        use crate::storage::{RowRawTableId, RowRawTableKind};
        use std::sync::Arc;

        let dir = tempfile::TempDir::new().unwrap();
        let mut storage = SqliteStorage::open(dir.path().join("test.sqlite")).unwrap();
        let schema_hash = crate::query_manager::types::SchemaHash::compute(
            &SchemaBuilder::new()
                .table(
                    TableSchema::builder("t")
                        .column("c", crate::query_manager::types::ColumnType::Integer),
                )
                .build(),
        );
        let row_id = ObjectId::new();
        let row_raw_table_id = RowRawTableId::new(RowRawTableKind::Visible, "t", schema_hash);
        let visible_row = OwnedVisibleRowBytes {
            row_raw_table: row_raw_table_id.raw_table_name().to_string(),
            row_raw_table_id: row_raw_table_id.clone(),
            user_descriptor: Arc::new(RowDescriptor::new(Vec::new())),
            branch: "main".to_string(),
            row_id,
            needs_exact_locator: true,
            bytes: vec![1],
        };
        let unkeyable_column = "c".repeat(6 * 1024);

        let failed = storage.apply_encoded_row_mutation(
            "t",
            &[],
            std::slice::from_ref(&visible_row),
            &[IndexMutation::Insert {
                table: "t",
                column: &unkeyable_column,
                branch: "main",
                value: Value::Integer(1),
                row_id,
            }],
        );
        assert!(matches!(failed, Err(StorageError::IndexKeyTooLarge { .. })));
        assert!(
            storage
                .load_raw_table_header(row_raw_table_id.raw_table_name())
                .unwrap()
                .is_none(),
            "the rollback took the header back"
        );

        storage
            .apply_encoded_row_mutation("t", &[], std::slice::from_ref(&visible_row), &[])
            .unwrap();
        assert!(
            storage
                .load_raw_table_header(row_raw_table_id.raw_table_name())
                .unwrap()
                .is_some(),
            "the family holds a row and has no header"
        );
        assert!(
            storage
                .load_visible_row_table_locator("main", row_id)
                .unwrap()
                .is_some(),
            "the row's exact locator was not written"
        );
    }

    /// A lost transaction takes back the raw-table headers and exact locators written in
    /// it, exactly as a rolled-back mutation does; the mutation after it writes them again.
    #[test]
    fn a_lost_transaction_does_not_leave_its_header_marked_written() {
        use crate::object::ObjectId;
        use crate::query_manager::types::{RowDescriptor, SchemaBuilder, TableSchema};
        use crate::storage::{RowRawTableId, RowRawTableKind};
        use std::sync::Arc;

        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("test.sqlite");
        let mut storage = SqliteStorage::open(&path).unwrap();
        let schema_hash = crate::query_manager::types::SchemaHash::compute(
            &SchemaBuilder::new()
                .table(
                    TableSchema::builder("t")
                        .column("c", crate::query_manager::types::ColumnType::Integer),
                )
                .build(),
        );
        let row_id = ObjectId::new();
        let row_raw_table_id = RowRawTableId::new(RowRawTableKind::Visible, "t", schema_hash);
        let visible_row = OwnedVisibleRowBytes {
            row_raw_table: row_raw_table_id.raw_table_name().to_string(),
            row_raw_table_id: row_raw_table_id.clone(),
            user_descriptor: Arc::new(RowDescriptor::new(Vec::new())),
            branch: "main".to_string(),
            row_id,
            needs_exact_locator: true,
            bytes: vec![1],
        };

        // The header is asked for through another connection: this one keeps headers it
        // has read in a cache of its own, which is not what is under test.
        let header_is_stored = || {
            visible_to_another_connection(
                &path,
                crate::storage::RAW_TABLE_HEADER_TABLE,
                row_raw_table_id.raw_table_name(),
            )
        };

        storage
            .apply_encoded_row_mutation("t", &[], std::slice::from_ref(&visible_row), &[])
            .unwrap();
        lose_the_open_transaction(&storage);
        assert!(
            !header_is_stored(),
            "the lost transaction took the header back"
        );

        storage
            .apply_encoded_row_mutation("t", &[], std::slice::from_ref(&visible_row), &[])
            .unwrap();
        assert!(storage.flush().is_err(), "a write transaction was lost");
        assert!(
            header_is_stored(),
            "the family holds a row and has no header"
        );
        assert!(
            storage
                .load_visible_row_table_locator("main", row_id)
                .unwrap()
                .is_some(),
            "the row's exact locator was not written"
        );
    }

    /// The memo against a model of the table, over a random run of everything that
    /// changes the key space: single writes, batches, batches that roll back, flushes,
    /// reopens and read scopes. Small prefixes must be answered EXACTLY — a memo that
    /// only ever said "not known" would pass a soundness check and save nothing — and a
    /// prefix that outgrew the memo must never be answered wrongly.
    #[test]
    fn read_memo_differential_against_a_model_of_the_table() {
        use crate::object::ObjectId;
        use crate::query_manager::types::Value;
        use crate::storage::RowLocator;
        use std::collections::{BTreeMap, BTreeSet};

        struct Rng(u64);
        impl Rng {
            fn next(&mut self) -> u64 {
                self.0 ^= self.0 << 13;
                self.0 ^= self.0 >> 7;
                self.0 ^= self.0 << 17;
                self.0
            }
            fn below(&mut self, bound: usize) -> usize {
                (self.next() % bound as u64) as usize
            }
            fn chance(&mut self, one_in: usize) -> bool {
                self.below(one_in) == 0
            }
        }

        const FAMILIES: [&str; 2] = ["fam_a", "fam_b"];
        const BRANCHES: [&str; 2] = ["main", "dev"];
        const COLUMNS: [&str; 2] = ["c", "d"];
        const SMALL_KEYS: usize = 10;
        // One family/branch pair grows past the memo's limit and shrinks back.
        const WIDE_FAMILY: &str = "fam_wide";
        const WIDE_KEYS: usize = MAX_SMALL_PREFIX_KEYS + 40;
        const STEPS: usize = 2_500;

        for seed in [
            0x9E37_79B9_7F4A_7C15u64,
            0xD1B5_4A32_D192_ED03,
            0x2545_F491_4F6C_DD1D,
        ] {
            let mut rng = Rng(seed);
            let dir = tempfile::TempDir::new().unwrap();
            let path = dir.path().join("test.sqlite");
            let mut storage = SqliteStorage::open(&path).unwrap();
            let row_ids: Vec<ObjectId> = (0..5).map(|_| ObjectId::new()).collect();
            let unkeyable_column = "c".repeat(6 * 1024);

            // The model: every (raw table, key) present, and every row locator.
            let mut present: BTreeSet<(String, String)> = BTreeSet::new();
            let mut locators: BTreeMap<usize, Option<RowLocator>> = BTreeMap::new();
            let mut exact_answers = 0usize;
            let mut wide_answers = 0usize;
            let mut locator_answers = 0usize;
            let mut rollbacks = 0usize;
            let mut open_scopes = 0usize;
            let mut unlisted_prefixes = 0usize;

            let family_key =
                |rng: &mut Rng| format!("{}:{}", BRANCHES[rng.below(2)], rng.below(SMALL_KEYS));

            for step in 0..STEPS {
                let context = |what: &str| format!("seed {seed:#x} step {step}: {what}");
                match rng.below(16) {
                    0..=2 => {
                        let family = FAMILIES[rng.below(2)];
                        let key = family_key(&mut rng);
                        storage.raw_table_put(family, &key, b"v").unwrap();
                        present.insert((family.to_string(), key));
                    }
                    3..=4 => {
                        let family = FAMILIES[rng.below(2)];
                        let key = family_key(&mut rng);
                        storage.raw_table_delete(family, &key).unwrap();
                        present.remove(&(family.to_string(), key));
                    }
                    5 => {
                        // A batch of puts and deletes under one savepoint.
                        let owned: Vec<(bool, &str, String)> = (0..rng.below(6) + 1)
                            .map(|_| (rng.chance(2), FAMILIES[rng.below(2)], family_key(&mut rng)))
                            .collect();
                        let mutations: Vec<RawTableMutation<'_>> = owned
                            .iter()
                            .map(|(put, family, key)| {
                                if *put {
                                    RawTableMutation::Put {
                                        table: family,
                                        key,
                                        value: b"v",
                                    }
                                } else {
                                    RawTableMutation::Delete { table: family, key }
                                }
                            })
                            .collect();
                        storage.apply_raw_table_mutations(&mutations).unwrap();
                        for (put, family, key) in owned {
                            if put {
                                present.insert((family.to_string(), key));
                            } else {
                                present.remove(&(family.to_string(), key));
                            }
                        }
                    }
                    6..=7 => {
                        // The wide prefix: runs of puts, runs of deletes.
                        let put = rng.chance(2);
                        for _ in 0..rng.below(60) + 1 {
                            let key = format!("main:{}", rng.below(WIDE_KEYS));
                            if put {
                                storage.raw_table_put(WIDE_FAMILY, &key, b"v").unwrap();
                                present.insert((WIDE_FAMILY.to_string(), key));
                            } else {
                                storage.raw_table_delete(WIDE_FAMILY, &key).unwrap();
                                present.remove(&(WIDE_FAMILY.to_string(), key));
                            }
                        }
                    }
                    8..=9 => {
                        // Index entries, one savepoint per batch; one batch in three ends
                        // with an entry that cannot be written and takes the batch back.
                        let fails = rng.chance(3);
                        let planned: Vec<(bool, &str, &str, i32, ObjectId)> = (0..rng.below(4) + 1)
                            .map(|_| {
                                (
                                    rng.chance(2),
                                    COLUMNS[rng.below(2)],
                                    BRANCHES[rng.below(2)],
                                    rng.below(3) as i32,
                                    row_ids[rng.below(row_ids.len())],
                                )
                            })
                            .collect();
                        let mut mutations: Vec<IndexMutation<'_>> = planned
                            .iter()
                            .map(|&(insert, column, branch, value, row_id)| {
                                let value = Value::Integer(value);
                                if insert {
                                    IndexMutation::Insert {
                                        table: "t",
                                        column,
                                        branch,
                                        value,
                                        row_id,
                                    }
                                } else {
                                    IndexMutation::Remove {
                                        table: "t",
                                        column,
                                        branch,
                                        value,
                                        row_id,
                                    }
                                }
                            })
                            .collect();
                        if fails {
                            mutations.push(IndexMutation::Insert {
                                table: "t",
                                column: &unkeyable_column,
                                branch: "main",
                                value: Value::Integer(0),
                                row_id: row_ids[0],
                            });
                        }
                        let result = storage.apply_index_mutations(&mutations);
                        if fails {
                            assert!(
                                matches!(result, Err(StorageError::IndexKeyTooLarge { .. })),
                                "{}",
                                context("the unkeyable entry was written")
                            );
                            rollbacks += 1;
                        } else {
                            result.unwrap();
                            for (insert, column, branch, value, row_id) in planned {
                                let entry = (
                                    key_codec::index_raw_table("t", column, branch),
                                    key_codec::index_entry_key(
                                        "t",
                                        column,
                                        branch,
                                        &Value::Integer(value),
                                        row_id,
                                    )
                                    .unwrap(),
                                );
                                if insert {
                                    present.insert(entry);
                                } else {
                                    present.remove(&entry);
                                }
                            }
                        }
                    }
                    10 => {
                        let index = rng.below(row_ids.len());
                        let locator = (!rng.chance(3)).then(|| RowLocator {
                            table: ["users", "chats"][rng.below(2)].into(),
                            origin_schema_hash: None,
                        });
                        storage
                            .put_row_locator(row_ids[index], locator.as_ref())
                            .unwrap();
                        locators.insert(index, locator);
                    }
                    11 => {
                        if open_scopes > 0 && rng.chance(2) {
                            storage.end_read_scope();
                            open_scopes -= 1;
                        } else if open_scopes < 2 {
                            storage.begin_read_scope();
                            open_scopes += 1;
                        }
                    }
                    12 => {
                        if rng.chance(4) {
                            while open_scopes > 0 {
                                storage.end_read_scope();
                                open_scopes -= 1;
                            }
                            storage.flush().unwrap();
                            if rng.chance(3) {
                                storage.close().unwrap();
                                storage = SqliteStorage::open(&path).unwrap();
                            }
                        }
                    }
                    _ => {}
                }

                // Ask the memo about a few things after every step, and hold each answer
                // against the model.
                for _ in 0..3 {
                    match rng.below(4) {
                        0 => {
                            let family = FAMILIES[rng.below(2)];
                            let branch = BRANCHES[rng.below(2)];
                            let prefix = format!("{branch}:");
                            let key = format!("{prefix}{}", rng.below(SMALL_KEYS + 2));
                            // The memo answers inside a transaction, and exactly there.
                            let usable = in_transaction(&storage);
                            let absent =
                                usable && !present.contains(&(family.to_string(), key.clone()));
                            let empty = usable
                                && !present.iter().any(|(table, stored)| {
                                    table == family && stored.starts_with(&prefix)
                                });
                            // Either question may come first: emptiness asked of a prefix
                            // not read yet reads one key of it, and the key question that
                            // follows has to read the rest.
                            let emptiness_first = rng.chance(2);
                            if emptiness_first {
                                assert_eq!(
                                    storage.raw_table_prefix_known_empty(family, &prefix),
                                    empty,
                                    "{}",
                                    context("a small prefix's emptiness was not answered exactly")
                                );
                                unlisted_prefixes +=
                                    usize::from(holds_an_unlisted_prefix(&storage));
                            }
                            assert_eq!(
                                storage.raw_table_key_known_absent(family, &prefix, &key),
                                absent,
                                "{}",
                                context("a small prefix was not answered exactly")
                            );
                            if !emptiness_first {
                                assert_eq!(
                                    storage.raw_table_prefix_known_empty(family, &prefix),
                                    empty,
                                    "{}",
                                    context("a small prefix's emptiness was not answered exactly")
                                );
                            }
                            exact_answers += usize::from(usable) * 2;
                        }
                        1 => {
                            let column = COLUMNS[rng.below(2)];
                            let branch = BRANCHES[rng.below(2)];
                            let value = Value::Integer(rng.below(3) as i32);
                            let row_id = row_ids[rng.below(row_ids.len())];
                            let raw_table = key_codec::index_raw_table("t", column, branch);
                            let key =
                                key_codec::index_entry_key("t", column, branch, &value, row_id)
                                    .unwrap();
                            let usable = in_transaction(&storage);
                            let absent = !present.contains(&(raw_table.clone(), key.clone()));
                            assert_eq!(
                                storage.raw_table_key_known_absent(&raw_table, "", &key),
                                usable && absent,
                                "{}",
                                context("an index entry was not answered exactly")
                            );
                            assert_eq!(
                                storage
                                    .index_contains("t", column, branch, &value, row_id)
                                    .unwrap(),
                                !absent,
                                "{}",
                                context("the table disagrees with the model")
                            );
                            let mut expected: Vec<ObjectId> = row_ids
                                .iter()
                                .copied()
                                .filter(|&candidate| {
                                    present.contains(&(
                                        raw_table.clone(),
                                        key_codec::index_entry_key(
                                            "t", column, branch, &value, candidate,
                                        )
                                        .unwrap(),
                                    ))
                                })
                                .collect();
                            expected.sort();
                            let mut found = storage.index_lookup("t", column, branch, &value);
                            found.sort();
                            assert_eq!(found, expected, "{}", context("index lookup"));
                            exact_answers += usize::from(usable);
                        }
                        2 => {
                            let key = format!("main:{}", rng.below(WIDE_KEYS));
                            let is_present =
                                present.contains(&(WIDE_FAMILY.to_string(), key.clone()));
                            if storage.raw_table_key_known_absent(WIDE_FAMILY, "main:", &key) {
                                assert!(
                                    !is_present,
                                    "{}",
                                    context("a present key of the wide prefix was called absent")
                                );
                                wide_answers += 1;
                            }
                            if storage.raw_table_prefix_known_empty(WIDE_FAMILY, "main:") {
                                assert!(
                                    !present.iter().any(|(table, _)| table == WIDE_FAMILY),
                                    "{}",
                                    context("the wide prefix was called empty")
                                );
                            }
                            assert_eq!(
                                storage.raw_table_get(WIDE_FAMILY, &key).unwrap().is_some(),
                                is_present,
                                "{}",
                                context("the table disagrees with the model")
                            );
                        }
                        _ => {
                            let index = rng.below(row_ids.len());
                            assert_eq!(
                                storage.load_row_locator(row_ids[index]).unwrap(),
                                locators.get(&index).cloned().flatten(),
                                "{}",
                                context("row locator")
                            );
                            locator_answers += 1;
                        }
                    }
                }
                assert_memo_accounting(&storage);
            }

            assert!(
                exact_answers > STEPS,
                "seed {seed:#x}: {exact_answers} exact answers"
            );
            assert!(
                unlisted_prefixes > 20,
                "seed {seed:#x}: emptiness came first for a prefix holding keys \
                 {unlisted_prefixes} times"
            );
            assert!(
                wide_answers > 20,
                "seed {seed:#x}: {wide_answers} wide answers"
            );
            assert!(
                locator_answers > STEPS / 4,
                "seed {seed:#x}: {locator_answers}"
            );
            assert!(rollbacks > 50, "seed {seed:#x}: {rollbacks} rollbacks");
        }
    }

    /// Two connections on one store file, the way the app has run two runtimes on one
    /// store: `ours` is the one under test, `theirs` commits behind its back. The model
    /// is what SQLite promises `ours` — outside a transaction it sees every commit, inside
    /// one it sees the snapshot the transaction began on plus its own writes — and both
    /// the table and the memo are held against it after every step. Flushes land inside
    /// open scopes, transactions are lost, and a commit of theirs is placed between the
    /// start of a scope and its first read.
    #[test]
    fn read_memo_differential_with_a_second_connection() {
        use std::collections::BTreeSet;

        struct Rng(u64);
        impl Rng {
            fn next(&mut self) -> u64 {
                self.0 ^= self.0 << 13;
                self.0 ^= self.0 >> 7;
                self.0 ^= self.0 << 17;
                self.0
            }
            fn below(&mut self, bound: usize) -> usize {
                (self.next() % bound as u64) as usize
            }
            fn chance(&mut self, one_in: usize) -> bool {
                self.below(one_in) == 0
            }
        }

        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        enum Tx {
            None,
            Read,
            Write,
        }

        const FAMILIES: [&str; 2] = ["fam_a", "fam_b"];
        const BRANCHES: [&str; 2] = ["main", "dev"];
        const KEYS: usize = 8;
        const STEPS: usize = 3_000;

        for seed in [
            0x9E37_79B9_7F4A_7C15u64,
            0xD1B5_4A32_D192_ED03,
            0x2545_F491_4F6C_DD1D,
        ] {
            let mut rng = Rng(seed);
            let dir = tempfile::TempDir::new().unwrap();
            let path = dir.path().join("test.sqlite");
            let mut ours = SqliteStorage::open(&path).unwrap();
            let mut theirs = SqliteStorage::open(&path).unwrap();

            // What the file holds for a transaction that begins now, and what `ours` sees.
            let mut committed: BTreeSet<(String, String)> = BTreeSet::new();
            let mut view: BTreeSet<(String, String)> = BTreeSet::new();
            let mut tx = Tx::None;
            let mut scopes = 0usize;
            let mut lost_writes = false;
            // Whether the open write transaction changed any key.
            let mut tx_changed = false;
            let mut refused_flushes = 0usize;
            let mut lost_with_writes = 0usize;
            let mut lost_unchanged = 0usize;

            let mut exact_answers = 0usize;
            let mut snapshots_held = 0usize;
            let mut foreign_commits_before_first_read = 0usize;
            let mut flushes_in_scope = 0usize;
            let mut lost = 0usize;

            let entry = |rng: &mut Rng| {
                (
                    FAMILIES[rng.below(2)].to_string(),
                    format!("{}:{}", BRANCHES[rng.below(2)], rng.below(KEYS)),
                )
            };
            // A commit through the other connection. It cannot land while `ours` holds
            // the writer's lock, so the caller only asks for one outside a write.
            let foreign_commit =
                |rng: &mut Rng,
                 theirs: &mut SqliteStorage,
                 committed: &mut BTreeSet<(String, String)>| {
                    for _ in 0..rng.below(3) + 1 {
                        let (family, key) = entry(rng);
                        if rng.chance(3) {
                            theirs.raw_table_delete(&family, &key).unwrap();
                            committed.remove(&(family, key));
                        } else {
                            theirs.raw_table_put(&family, &key, b"v").unwrap();
                            committed.insert((family, key));
                        }
                    }
                    theirs.flush().unwrap();
                };

            for step in 0..STEPS {
                let context = |what: &str| format!("seed {seed:#x} step {step}: {what}");
                match rng.below(16) {
                    0..=1 => {
                        // A write of ours: it runs in a write transaction, begun on what
                        // is committed now if none is open.
                        if tx != Tx::Write {
                            view = committed.clone();
                            tx = Tx::Write;
                            tx_changed = false;
                        }
                        let (family, key) = entry(&mut rng);
                        if rng.chance(3) {
                            ours.raw_table_delete(&family, &key).unwrap();
                            // Deleting a key that is not there changes nothing, and a
                            // transaction that changed nothing has nothing to lose.
                            tx_changed |= view.remove(&(family, key));
                        } else {
                            ours.raw_table_put(&family, &key, b"v").unwrap();
                            view.insert((family, key));
                            tx_changed = true;
                        }
                    }
                    2..=5 => {
                        if tx != Tx::Write {
                            foreign_commit(&mut rng, &mut theirs, &mut committed);
                            if tx == Tx::None {
                                view = committed.clone();
                            } else {
                                snapshots_held += 1;
                            }
                        }
                    }
                    6..=8 => {
                        if scopes > 0 && rng.chance(2) {
                            ours.end_read_scope();
                            scopes -= 1;
                            if scopes == 0 && tx == Tx::Read {
                                tx = Tx::None;
                                view = committed.clone();
                            }
                        } else if scopes < 2 {
                            ours.begin_read_scope();
                            scopes += 1;
                            if scopes == 1 && tx == Tx::None {
                                tx = Tx::Read;
                                // The scope's snapshot is taken as it begins: a commit
                                // that lands before its first read is not in it.
                                if rng.chance(2) {
                                    foreign_commit(&mut rng, &mut theirs, &mut committed);
                                    foreign_commits_before_first_read += 1;
                                }
                            }
                        }
                    }
                    9..=11 => {
                        // A flush, inside a scope as often as not. It commits whatever
                        // transaction is open, a scope's included.
                        let flushed = ours.flush();
                        assert_eq!(
                            flushed.is_err(),
                            lost_writes,
                            "{}",
                            context("flush after a lost write transaction")
                        );
                        refused_flushes += usize::from(lost_writes);
                        if tx == Tx::Write {
                            committed = view.clone();
                        }
                        if scopes > 0 && tx != Tx::None {
                            flushes_in_scope += 1;
                        }
                        tx = Tx::None;
                        view = committed.clone();
                    }
                    12 => {
                        if tx != Tx::None && rng.chance(3) {
                            lose_the_open_transaction(&ours);
                            if tx == Tx::Write {
                                lost_writes |= tx_changed;
                                lost_with_writes += usize::from(tx_changed);
                                lost_unchanged += usize::from(!tx_changed);
                            }
                            lost += 1;
                            tx = Tx::None;
                            view = committed.clone();
                        }
                    }
                    13 => {
                        if rng.chance(6) {
                            while scopes > 0 {
                                ours.end_read_scope();
                                scopes -= 1;
                            }
                            let closed = ours.close();
                            assert_eq!(closed.is_err(), lost_writes, "{}", context("close"));
                            if tx == Tx::Write {
                                committed = view.clone();
                            }
                            ours = SqliteStorage::open(&path).unwrap();
                            lost_writes = false;
                            tx = Tx::None;
                            view = committed.clone();
                        }
                    }
                    _ => {}
                }

                assert_eq!(
                    in_transaction(&ours),
                    tx != Tx::None,
                    "{}",
                    context(&format!("the model says {tx:?}"))
                );
                for _ in 0..3 {
                    let family = FAMILIES[rng.below(2)];
                    let branch = BRANCHES[rng.below(2)];
                    let prefix = format!("{branch}:");
                    let key = format!("{prefix}{}", rng.below(KEYS + 2));
                    let held = view.contains(&(family.to_string(), key.clone()));
                    let usable = tx != Tx::None;
                    assert_eq!(
                        ours.raw_table_key_known_absent(family, &prefix, &key),
                        usable && !held,
                        "{}",
                        context("the memo's answer about a key")
                    );
                    assert_eq!(
                        ours.raw_table_prefix_known_empty(family, &prefix),
                        usable
                            && !view
                                .iter()
                                .any(|(table, stored)| table == family
                                    && stored.starts_with(&prefix)),
                        "{}",
                        context("the memo's answer about a prefix")
                    );
                    assert_eq!(
                        ours.raw_table_get(family, &key).unwrap().is_some(),
                        held,
                        "{}",
                        context("the table as this connection reads it")
                    );
                    exact_answers += usize::from(usable) * 2;
                }
                assert_memo_accounting(&ours);
            }

            assert!(
                exact_answers > STEPS,
                "seed {seed:#x}: {exact_answers} exact answers"
            );
            assert!(
                lost_with_writes > 3 && refused_flushes > 3,
                "seed {seed:#x}: {lost_with_writes} transactions lost with writes in them \
                 ({lost_unchanged} lost having changed nothing), {refused_flushes} flushes \
                 refused"
            );
            assert!(
                snapshots_held > 30,
                "seed {seed:#x}: {snapshots_held} held snapshots"
            );
            assert!(
                foreign_commits_before_first_read > 20,
                "seed {seed:#x}: {foreign_commits_before_first_read} commits before a first read"
            );
            assert!(
                flushes_in_scope > 50,
                "seed {seed:#x}: {flushes_in_scope} flushes in a scope"
            );
            assert!(lost > 10, "seed {seed:#x}: {lost} lost transactions");
        }
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
}
