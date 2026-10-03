//! RocksDB-backed Storage implementation.
//!
//! Uses `TransactionDB` (pessimistic transactions) for write operations and
//! direct DB access for read-only operations. Follows the same structural
//! pattern as FjallStorage, delegating all logic to `storage_core` callbacks.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::path::Path;

use rocksdb::{
    BlockBasedOptions, Cache, IteratorMode, Options, ReadOptions, TransactionDB,
    TransactionDBOptions,
};

use super::{
    HistoryRowBytes, IndexMutation, RawTableMutation, Storage, StorageError, VisibleRowBytes,
    key_codec,
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
/// Largest key set of a branch's exact row-table locators the memo holds; a branch with
/// more is not answered from memory.
const MAX_MEMOIZED_EXACT_LOCATOR_KEYS: usize = 128;
/// Most branch prefixes whose exact row-table locator keys the memo holds; past it a branch
/// is forgotten to make room for the next, one at a time, so that a store with more branches
/// than this still answers most of them.
const MAX_MEMOIZED_EXACT_LOCATOR_PREFIXES: usize = 4096;
/// Most keys the memo holds over all branches; past it a branch that holds keys is forgotten,
/// one at a time, and the branches holding none — almost all of them — stay.
const MAX_MEMOIZED_EXACT_LOCATOR_KEYS_HELD: usize = 16_384;

/// The row locators a store remembers: `__row_locator` storage key → the decoded locator,
/// `None` for "no locator".
type RowLocatorMemo = RefCell<HashMap<String, Option<super::RowLocator>>>;

/// The exact row-table locator keys a store remembers: storage prefix of a branch → its
/// keys, `None` for more than the memo holds of a branch; and how many keys it holds over all
/// branches.
#[derive(Default)]
struct ExactLocatorKeys {
    by_prefix: HashMap<String, Option<HashSet<String>>>,
    keys_held: usize,
}

type ExactLocatorMemo = RefCell<ExactLocatorKeys>;

impl ExactLocatorKeys {
    /// Whether `storage_key` is known absent under `storage_prefix`; `None` if the branch is
    /// not remembered.
    fn known_absent(&self, storage_prefix: &str, storage_key: &str) -> Option<bool> {
        self.by_prefix.get(storage_prefix).map(|keys| {
            keys.as_ref()
                .is_some_and(|keys| !keys.contains(storage_key))
        })
    }

    /// Remembers a branch's keys, forgetting other branches one at a time to make room.
    fn remember(&mut self, storage_prefix: String, keys: Option<HashSet<String>>) {
        self.forget(&storage_prefix);
        let adding = keys.as_ref().map_or(0, HashSet::len);
        while self.by_prefix.len() >= MAX_MEMOIZED_EXACT_LOCATOR_PREFIXES {
            let Some(victim) = self.by_prefix.keys().next().cloned() else {
                break;
            };
            self.forget(&victim);
        }
        while self.keys_held + adding > MAX_MEMOIZED_EXACT_LOCATOR_KEYS_HELD
            && self.forget_one_holding_keys()
        {}
        self.keys_held += adding;
        self.by_prefix.insert(storage_prefix, keys);
    }

    fn forget(&mut self, storage_prefix: &str) {
        if let Some(keys) = self.by_prefix.remove(storage_prefix) {
            self.keys_held -= keys.map_or(0, |keys| keys.len());
        }
    }

    /// Forgets a branch that holds keys; false if none does.
    fn forget_one_holding_keys(&mut self) -> bool {
        let Some(victim) = self
            .by_prefix
            .iter()
            .find(|(_, keys)| keys.as_ref().is_some_and(|keys| !keys.is_empty()))
            .map(|(storage_prefix, _)| storage_prefix.clone())
        else {
            return false;
        };
        self.forget(&victim);
        true
    }

    fn clear(&mut self) {
        self.by_prefix.clear();
        self.keys_held = 0;
    }

    /// A committed put or delete of `storage_key`, put into or taken out of every branch it
    /// falls under. A branch that grows past what the memo holds of one stays unanswered: a
    /// delete does not say how many are left.
    fn apply(&mut self, storage_key: &str, put: bool) {
        for (storage_prefix, keys) in self.by_prefix.iter_mut() {
            if !storage_key.starts_with(storage_prefix.as_str()) {
                continue;
            }
            let Some(held) = keys else {
                continue;
            };
            if put {
                if held.insert(storage_key.to_string()) {
                    self.keys_held += 1;
                }
                if held.len() > MAX_MEMOIZED_EXACT_LOCATOR_KEYS {
                    self.keys_held -= held.len();
                    *keys = None;
                }
            } else if held.remove(storage_key) {
                self.keys_held -= 1;
            }
        }
        while self.keys_held > MAX_MEMOIZED_EXACT_LOCATOR_KEYS_HELD
            && self.forget_one_holding_keys()
        {}
    }
}

/// The store's database and its write transactions, apart so that nothing else in this file
/// can write a key of an open store: the store's memos are kept by what `StoreTxn` notes,
/// and a write past it would leave them serving what it overwrote.
mod store_db {
    use std::cell::{Cell, RefCell};

    use rocksdb::{
        DBIteratorWithThreadMode, DBRawIteratorWithThreadMode, IteratorMode, ReadOptions,
        Transaction, TransactionDB,
    };

    use super::{ExactLocatorMemo, RowLocatorMemo, StorageError, key_codec};

    #[cfg(test)]
    thread_local! {
        /// Whether the next `StoreTxn::commit` of this thread fails, rolled back.
        static FAIL_NEXT_COMMIT: Cell<bool> = const { Cell::new(false) };
    }

    /// Makes the next commit of this thread fail, rolled back.
    #[cfg(test)]
    pub(super) fn fail_next_commit() {
        FAIL_NEXT_COMMIT.with(|fails| fails.set(true));
    }

    /// An open store's database: read here, written through a `StoreTxn` only.
    pub(super) struct StoreDb(TransactionDB);

    impl StoreDb {
        /// Takes the database once its manifest is in place, the one key written past a
        /// `StoreTxn`.
        pub(super) fn new(db: TransactionDB) -> Self {
            Self(db)
        }

        pub(super) fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, rocksdb::Error> {
            self.0.get(key)
        }

        pub(super) fn iterator_opt(
            &self,
            mode: IteratorMode<'_>,
            read_opts: ReadOptions,
        ) -> DBIteratorWithThreadMode<'_, TransactionDB> {
            self.0.iterator_opt(mode, read_opts)
        }

        pub(super) fn raw_iterator_opt(
            &self,
            read_opts: ReadOptions,
        ) -> DBRawIteratorWithThreadMode<'_, TransactionDB> {
            self.0.raw_iterator_opt(read_opts)
        }

        pub(super) fn flush(&self) -> Result<(), rocksdb::Error> {
            self.0.flush()
        }

        pub(super) fn flush_wal(&self, sync: bool) -> Result<(), rocksdb::Error> {
            self.0.flush_wal(sync)
        }
    }

    /// A write transaction of one store, and the only way a key of an open store is written
    /// (the manifest is put while the store opens): the puts and deletes go through it, so a
    /// transaction that wrote a raw-table header moves the store's `header_epoch` here and
    /// nowhere else.
    pub(super) struct StoreTxn<'a> {
        txn: Transaction<'a, TransactionDB>,
        header_epoch: &'a Cell<u64>,
        wrote_header: Cell<bool>,
        row_locators: &'a RowLocatorMemo,
        written_row_locators: RefCell<Vec<String>>,
        /// Whether a locator table's header was written: what was read under the old one goes.
        wrote_row_locator_header: Cell<bool>,
        wrote_exact_locator_header: Cell<bool>,
        exact_locators: &'a ExactLocatorMemo,
        /// The exact row-table locator keys written, in order, each with whether it was put.
        written_exact_locators: RefCell<Vec<(String, bool)>>,
    }

    impl<'a> StoreTxn<'a> {
        pub(super) fn begin(
            db: &'a StoreDb,
            header_epoch: &'a Cell<u64>,
            row_locators: &'a RowLocatorMemo,
            exact_locators: &'a ExactLocatorMemo,
        ) -> Self {
            Self {
                txn: db.0.transaction(),
                header_epoch,
                wrote_header: Cell::new(false),
                row_locators,
                written_row_locators: RefCell::new(Vec::new()),
                wrote_row_locator_header: Cell::new(false),
                wrote_exact_locator_header: Cell::new(false),
                exact_locators,
                written_exact_locators: RefCell::new(Vec::new()),
            }
        }

        pub(super) fn put(&self, key: &str, value: &[u8]) -> Result<(), StorageError> {
            self.note_write(key, true);
            self.txn
                .put(key.as_bytes(), value)
                .map_err(|e| StorageError::IoError(format!("rocksdb txn put: {e}")))
        }

        pub(super) fn delete(&self, key: &str) -> Result<(), StorageError> {
            self.note_write(key, false);
            self.txn
                .delete(key.as_bytes())
                .map_err(|e| StorageError::IoError(format!("rocksdb txn delete: {e}")))
        }

        /// A header or a locator written here is readable once this commits, and not before:
        /// the header epoch moves, the written row locators leave the memo, and the written
        /// exact locator keys are put into or taken out of every branch's keys they fall
        /// under here, so that nothing read before — the write or the commit — is served
        /// after. A commit that fails leaves the store unknown: the exact locator keys are
        /// all dropped.
        pub(super) fn commit(self) -> Result<(), StorageError> {
            #[cfg(test)]
            let fails = FAIL_NEXT_COMMIT.with(|fails| fails.replace(false));
            #[cfg(not(test))]
            let fails = false;
            let committed = if fails {
                // Dropped uncommitted: rolled back.
                drop(self.txn);
                Err(StorageError::IoError(
                    "rocksdb txn commit: failed by a test".to_string(),
                ))
            } else {
                self.txn
                    .commit()
                    .map_err(|e| StorageError::IoError(format!("rocksdb txn commit: {e}")))
            };
            if self.wrote_header.get() {
                self.header_epoch.set(self.header_epoch.get() + 1);
            }
            let written = self.written_row_locators.take();
            if self.wrote_row_locator_header.get() {
                self.row_locators.borrow_mut().clear();
            } else if !written.is_empty() {
                let mut memo = self.row_locators.borrow_mut();
                for storage_key in &written {
                    memo.remove(storage_key);
                }
            }
            let written = self.written_exact_locators.take();
            if self.wrote_exact_locator_header.get() {
                self.exact_locators.borrow_mut().clear();
            } else if !written.is_empty() {
                let mut memo = self.exact_locators.borrow_mut();
                if committed.is_err() {
                    memo.clear();
                }
                for (storage_key, put) in &written {
                    memo.apply(storage_key, *put);
                }
            }
            committed
        }

        fn note_write(&self, storage_key: &str, put: bool) {
            if let Some(raw_table) =
                key_codec::strip_raw_table_key(super::super::RAW_TABLE_HEADER_TABLE, storage_key)
            {
                self.wrote_header.set(true);
                if raw_table == super::super::ROW_LOCATOR_TABLE {
                    self.wrote_row_locator_header.set(true);
                } else if raw_table == super::super::VISIBLE_ROW_TABLE_LOCATOR_TABLE {
                    self.wrote_exact_locator_header.set(true);
                }
            } else if key_codec::strip_raw_table_key(super::super::ROW_LOCATOR_TABLE, storage_key)
                .is_some()
            {
                self.written_row_locators
                    .borrow_mut()
                    .push(storage_key.to_string());
            } else if key_codec::strip_raw_table_key(
                super::super::VISIBLE_ROW_TABLE_LOCATOR_TABLE,
                storage_key,
            )
            .is_some()
            {
                self.written_exact_locators
                    .borrow_mut()
                    .push((storage_key.to_string(), put));
            }
        }
    }
}

use store_db::{StoreDb, StoreTxn};

struct RocksDBInner {
    db: StoreDb,
    ensured_raw_table_headers: HashSet<String>,
    visible_row_table_locators: HashMap<(String, ObjectId), super::ExactRowTableLocator>,
}

pub struct RocksDBStorage {
    cache_namespace: usize,
    inner: RefCell<Option<RocksDBInner>>,
    /// Moved by every transaction that wrote a raw-table header of this store
    /// (`StoreTxn`). A header is written when a table gets a new raw table — a first
    /// write, a new schema generation — so this hardly ever moves.
    header_epoch: std::cell::Cell<u64>,
    /// `row_raw_table_ids_for_table` by header prefix, and the `header_epoch` it was
    /// read under. Without it every row load scans the header table for the raw tables
    /// of its table.
    row_raw_table_ids: RefCell<(u64, HashMap<String, Vec<super::RowRawTableId>>)>,
    /// `load_row_locator` by storage key. A committed write of a locator removes it.
    row_locators: RowLocatorMemo,
    /// The exact row-table locator keys under a branch prefix: every row load asks whether
    /// its row has one, and almost none has. A committed write of one is applied to every
    /// prefix it falls under.
    exact_locator_keys: ExactLocatorMemo,
    /// Prefix scans issued against the store. A whole-history read is one of
    /// these; the existence probe that replaced it is a seek. A gate can pin
    /// that the probe's cost does not track how deep the history is.
    #[cfg(test)]
    prefix_scans: std::cell::Cell<usize>,
    /// Point reads of the store through `raw_table_get`, by raw table.
    #[cfg(test)]
    table_gets: RefCell<HashMap<String, usize>>,
    /// Reads of a branch's exact row-table locator keys, to answer from memory.
    #[cfg(test)]
    exact_locator_scans: std::cell::Cell<usize>,
}

impl RocksDBStorage {
    fn store_has_any_rows(db: &TransactionDB) -> Result<bool, StorageError> {
        let mut iter = db.iterator(IteratorMode::Start);
        match iter.next() {
            Some(Ok(_)) => Ok(true),
            Some(Err(e)) => Err(StorageError::IoError(format!(
                "rocksdb inspect store contents: {e}"
            ))),
            None => Ok(false),
        }
    }

    fn ensure_store_manifest(db: &TransactionDB) -> Result<(), StorageError> {
        let expected = super::expected_store_manifest(super::ROCKSDB_STORE_KIND);
        match db
            .get(super::STORE_MANIFEST_KEY.as_bytes())
            .map_err(|e| StorageError::IoError(format!("rocksdb read store manifest: {e}")))?
        {
            Some(bytes) => {
                let actual = super::decode_store_manifest(&bytes)?;
                super::validate_store_manifest(&actual, &expected)
            }
            None => {
                if Self::store_has_any_rows(db)? {
                    return Err(StorageError::IoError(
                        "missing store manifest for non-empty rocksdb store".to_string(),
                    ));
                }
                let bytes = super::encode_store_manifest(&expected)?;
                db.put(super::STORE_MANIFEST_KEY.as_bytes(), bytes)
                    .map_err(|e| {
                        StorageError::IoError(format!("rocksdb write store manifest: {e}"))
                    })
            }
        }
    }

    pub fn open(path: impl AsRef<Path>, cache_size_bytes: usize) -> Result<Self, StorageError> {
        let mut block_opts = BlockBasedOptions::default();
        block_opts.set_bloom_filter(10.0, false);
        let cache = Cache::new_lru_cache(cache_size_bytes);
        block_opts.set_block_cache(&cache);

        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.set_block_based_table_factory(&block_opts);
        // LZ4 for L0-L2 (fast), Zstd for deeper levels (compact)
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_bottommost_compression_type(rocksdb::DBCompressionType::Zstd);
        // Compact a file once it is mostly deletions.
        //
        // The engine's write pattern puts a key and then deletes it on the same narrow
        // keyspaces — every unbatched direct write seals its own batch, writing a sealed
        // submission that is deleted again the moment the batch settles. The live set stays
        // tiny while the tombstones accumulate, and every prefix iteration over that
        // keyspace has to step over all of them.
        //
        // Measured on a store shaped like production: iterating the ~1284 live
        // sealed-submission keys cost 350 µs with no churn and 2.87 ms behind ~50k
        // tombstones, and that iteration runs at the top of every tick. Reclaiming it needs
        // a compaction that RocksDB will not schedule on its own, because nothing here
        // triggers its usual heuristics — the files are small and the level is bottommost.
        //
        // 10000 keys in a sliding window, 5000 of them deletions: a file that is half
        // tombstones over any such window is marked for compaction as it is written.
        //
        // This does not retroactively clean a store that already accrued them — a forced
        // range compaction would, but `TransactionDB` exposes no compaction call in this
        // binding. It does not need to: the sealed-submission keyspace is rewritten by
        // every direct write, so new files are produced, marked, and compacted against the
        // old ones continuously. The tombstones drain with use rather than at open.
        opts.add_compact_on_deletion_collector_factory(10_000, 5_000, 0.0);

        let txdb_opts = TransactionDBOptions::default();
        let db = TransactionDB::open(&opts, &txdb_opts, path.as_ref())
            .map_err(|e| StorageError::IoError(format!("rocksdb open: {e}")))?;
        Self::ensure_store_manifest(&db)?;

        Ok(Self {
            cache_namespace: super::next_storage_cache_namespace(),
            header_epoch: std::cell::Cell::new(0),
            row_raw_table_ids: RefCell::new((0, HashMap::new())),
            row_locators: RefCell::new(HashMap::new()),
            exact_locator_keys: RefCell::new(ExactLocatorKeys::default()),
            #[cfg(test)]
            prefix_scans: std::cell::Cell::new(0),
            #[cfg(test)]
            table_gets: RefCell::new(HashMap::new()),
            #[cfg(test)]
            exact_locator_scans: std::cell::Cell::new(0),
            inner: RefCell::new(Some(RocksDBInner {
                db: StoreDb::new(db),
                ensured_raw_table_headers: HashSet::new(),
                visible_row_table_locators: HashMap::new(),
            })),
        })
    }

    fn with_inner<T>(
        &self,
        f: impl FnOnce(&RocksDBInner) -> Result<T, StorageError>,
    ) -> Result<T, StorageError> {
        let inner = self.inner.borrow();
        let inner = inner
            .as_ref()
            .ok_or_else(|| StorageError::IoError("rocksdb storage already closed".to_string()))?;
        f(inner)
    }

    fn with_inner_mut<T>(
        &self,
        f: impl FnOnce(&mut RocksDBInner) -> Result<T, StorageError>,
    ) -> Result<T, StorageError> {
        let mut inner = self.inner.borrow_mut();
        let inner = inner
            .as_mut()
            .ok_or_else(|| StorageError::IoError("rocksdb storage already closed".to_string()))?;
        f(inner)
    }

    /// Compute the lexicographic successor of a byte prefix for use as an
    /// exclusive upper bound. Returns `None` when the prefix is all `0xFF`
    /// bytes (practically never for our key scheme).
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

    // ---- read helpers (direct DB, no transaction) ----

    fn get_from_db(db: &StoreDb, key: &str) -> Result<Option<Vec<u8>>, StorageError> {
        db.get(key.as_bytes())
            .map_err(|e| StorageError::IoError(format!("rocksdb get: {e}")))
    }

    fn scan_prefix_from_db(
        db: &StoreDb,
        prefix: &str,
    ) -> Result<Vec<(String, Vec<u8>)>, StorageError> {
        let prefix_bytes = prefix.as_bytes();
        let mut read_opts = ReadOptions::default();
        if let Some(ub) = Self::prefix_upper_bound(prefix_bytes) {
            read_opts.set_iterate_upper_bound(ub);
        }
        let mut out = Vec::new();
        let iter = db.iterator_opt(
            IteratorMode::From(prefix_bytes, rocksdb::Direction::Forward),
            read_opts,
        );
        for item in iter {
            let (key, value) =
                item.map_err(|e| StorageError::IoError(format!("rocksdb iter: {e}")))?;
            let key_str = String::from_utf8(key.to_vec())
                .map_err(|e| StorageError::IoError(format!("rocksdb invalid key utf8: {e}")))?;
            out.push((key_str, value.to_vec()));
        }
        Ok(out)
    }

    /// The first key under `prefix`, without reading the rest.
    ///
    /// Same seek as the prefix scan, stopped after one step. The callers use it
    /// to ask whether anything exists, and they are the ones replacing reads of
    /// a row's whole history — materialising the answer set instead would trade
    /// one unbounded read for another.
    /// Is there a key for this row past the end of `branch_prefix`?
    ///
    /// One seek to the successor of the branch range, bounded by the end of the
    /// row's range, so it stops at the first key rather than reading the rest.
    fn first_row_key_after_branch(
        &self,
        table: &str,
        row_prefix: &str,
        branch_prefix: &str,
    ) -> Result<bool, StorageError> {
        let Some(after_branch) = Self::prefix_upper_bound(branch_prefix.as_bytes()) else {
            return Ok(false);
        };
        let Ok(after_branch) = String::from_utf8(after_branch) else {
            // The successor is not valid utf8, so it cannot be compared against
            // the string keys this store uses. Fall back to the honest answer.
            return Ok(self
                .raw_table_scan_prefix_keys(table, row_prefix)?
                .into_iter()
                .any(|key| !key.starts_with(branch_prefix)));
        };
        let end = Self::prefix_upper_bound(row_prefix.as_bytes())
            .and_then(|bytes| String::from_utf8(bytes).ok());
        Ok(self
            .raw_table_scan_range_keys(table, Some(&after_branch), end.as_deref())?
            .into_iter()
            .any(|key| key.starts_with(row_prefix)))
    }

    fn first_key_with_prefix_from_db(
        db: &StoreDb,
        prefix: &str,
    ) -> Result<Option<String>, StorageError> {
        let prefix_bytes = prefix.as_bytes();
        let mut read_opts = ReadOptions::default();
        if let Some(ub) = Self::prefix_upper_bound(prefix_bytes) {
            read_opts.set_iterate_upper_bound(ub);
        }
        let mut iter = db.iterator_opt(
            IteratorMode::From(prefix_bytes, rocksdb::Direction::Forward),
            read_opts,
        );
        match iter.next() {
            None => Ok(None),
            Some(item) => {
                let (key, _) =
                    item.map_err(|e| StorageError::IoError(format!("rocksdb iter: {e}")))?;
                let key_str = String::from_utf8(key.to_vec())
                    .map_err(|e| StorageError::IoError(format!("rocksdb key is not utf8: {e}")))?;
                Ok(key_str.starts_with(prefix).then_some(key_str))
            }
        }
    }

    fn scan_prefix_keys_from_db(db: &StoreDb, prefix: &str) -> Result<Vec<String>, StorageError> {
        let prefix_bytes = prefix.as_bytes();
        let mut read_opts = ReadOptions::default();
        if let Some(ub) = Self::prefix_upper_bound(prefix_bytes) {
            read_opts.set_iterate_upper_bound(ub);
        }
        // A raw iterator, not `iterator_opt`: the latter yields `(key, value)` and copies
        // BOTH out of the block for every row, so a "keys only" scan still paid for every
        // value. This is the scan the per-tick sealed-batch recovery leans on, and on a
        // store holding a thousand-odd retained submissions the copies were most of its
        // cost.
        let mut out = Vec::new();
        let mut iter = db.raw_iterator_opt(read_opts);
        iter.seek(prefix_bytes);
        while iter.valid() {
            let Some(key) = iter.key() else { break };
            let key_str = String::from_utf8(key.to_vec())
                .map_err(|e| StorageError::IoError(format!("rocksdb invalid key utf8: {e}")))?;
            out.push(key_str);
            iter.next();
        }
        iter.status()
            .map_err(|e| StorageError::IoError(format!("rocksdb iter: {e}")))?;
        Ok(out)
    }

    fn scan_range_from_db(
        db: &StoreDb,
        start: &str,
        end: &str,
    ) -> Result<Vec<(String, Vec<u8>)>, StorageError> {
        let start_bytes = start.as_bytes();
        let mut read_opts = ReadOptions::default();
        read_opts.set_iterate_upper_bound(end.as_bytes().to_vec());
        let mut out = Vec::new();
        let iter = db.iterator_opt(
            IteratorMode::From(start_bytes, rocksdb::Direction::Forward),
            read_opts,
        );
        for item in iter {
            let (key, value) =
                item.map_err(|e| StorageError::IoError(format!("rocksdb iter: {e}")))?;
            let key_str = String::from_utf8(key.to_vec())
                .map_err(|e| StorageError::IoError(format!("rocksdb invalid key utf8: {e}")))?;
            out.push((key_str, value.to_vec()));
        }
        Ok(out)
    }

    fn scan_range_keys_from_db(
        db: &StoreDb,
        start: &str,
        end: &str,
    ) -> Result<Vec<String>, StorageError> {
        let start_bytes = start.as_bytes();
        let mut read_opts = ReadOptions::default();
        read_opts.set_iterate_upper_bound(end.as_bytes().to_vec());
        let mut out = Vec::new();
        let iter = db.iterator_opt(
            IteratorMode::From(start_bytes, rocksdb::Direction::Forward),
            read_opts,
        );
        for item in iter {
            let (key, _) = item.map_err(|e| StorageError::IoError(format!("rocksdb iter: {e}")))?;
            let key_str = String::from_utf8(key.to_vec())
                .map_err(|e| StorageError::IoError(format!("rocksdb invalid key utf8: {e}")))?;
            out.push(key_str);
        }
        Ok(out)
    }

    fn scan_range_keys_limited_from_db(
        db: &StoreDb,
        start: &str,
        end: &str,
        reverse: bool,
        limit: usize,
    ) -> Result<Vec<String>, StorageError> {
        let mut read_opts = ReadOptions::default();
        read_opts.set_iterate_lower_bound(start.as_bytes().to_vec());
        read_opts.set_iterate_upper_bound(end.as_bytes().to_vec());
        // A raw iterator for the same reason as `scan_prefix_keys_from_db`: keys only.
        let mut out = Vec::new();
        let mut iter = db.raw_iterator_opt(read_opts);
        if reverse {
            iter.seek_to_last();
        } else {
            iter.seek(start.as_bytes());
        }
        while iter.valid() && out.len() < limit {
            let Some(key) = iter.key() else { break };
            let key_str = String::from_utf8(key.to_vec())
                .map_err(|e| StorageError::IoError(format!("rocksdb invalid key utf8: {e}")))?;
            out.push(key_str);
            if reverse {
                iter.prev();
            } else {
                iter.next();
            }
        }
        iter.status()
            .map_err(|e| StorageError::IoError(format!("rocksdb iter: {e}")))?;
        Ok(out)
    }

    // ---- transaction helpers ----

    fn begin<'a>(&'a self, db: &'a StoreDb) -> StoreTxn<'a> {
        StoreTxn::begin(
            db,
            &self.header_epoch,
            &self.row_locators,
            &self.exact_locator_keys,
        )
    }

    fn apply_index_mutations_on_txn(
        txn: &StoreTxn<'_>,
        mutations: &[IndexMutation<'_>],
    ) -> Result<(), StorageError> {
        for mutation in mutations {
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
                        txn.put(storage_key, bytes)
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
                    raw_table_delete_core(&raw_table, &key, |storage_key| txn.delete(storage_key))?;
                }
            }
        }
        Ok(())
    }
}

impl Storage for RocksDBStorage {
    fn storage_cache_namespace(&self) -> usize {
        self.cache_namespace
    }

    fn memoized_row_raw_table_ids(&self, header_prefix: &str) -> Option<Vec<super::RowRawTableId>> {
        let memo = self.row_raw_table_ids.borrow();
        if memo.0 != self.header_epoch.get() {
            return None;
        }
        let ids = memo.1.get(header_prefix).cloned();
        // Parity harness: the ids are the header keys under the prefix.
        #[cfg(debug_assertions)]
        if let Some(ids) = &ids {
            let stored = self.with_inner(|inner| {
                Self::scan_prefix_keys_from_db(
                    &inner.db,
                    &key_codec::raw_table_scan_prefix(super::RAW_TABLE_HEADER_TABLE, header_prefix),
                )
            });
            if let Ok(stored) = stored {
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
                    "the memo diverged from the raw-table headers under {header_prefix:?}"
                );
            }
        }
        ids
    }

    fn load_row_locator(&self, id: ObjectId) -> Result<Option<super::RowLocator>, StorageError> {
        let storage_key =
            key_codec::raw_table_entry_key(super::ROW_LOCATOR_TABLE, &super::metadata_raw_key(id));
        let memoized = self.row_locators.borrow().get(&storage_key).cloned();
        if let Some(locator) = memoized {
            // Parity harness: the stored bytes still decode to what the memo holds.
            #[cfg(debug_assertions)]
            if let Ok(stored) = self.with_inner(|inner| Self::get_from_db(&inner.db, &storage_key))
                && let Ok(stored) = stored
                    .map(|bytes| super::decode_row_locator(&bytes))
                    .transpose()
            {
                debug_assert_eq!(
                    stored, locator,
                    "the memo diverged from the stored row locator of {id}"
                );
            }
            return Ok(locator);
        }
        let locator = super::storage_trait::load_row_locator_default(self, id)?;
        let mut memo = self.row_locators.borrow_mut();
        if memo.len() >= MAX_MEMOIZED_ROW_LOCATORS {
            memo.clear();
        }
        memo.insert(storage_key, locator.clone());
        Ok(locator)
    }

    fn raw_table_key_known_absent(&self, table: &str, prefix: &str, key: &str) -> bool {
        // Only the exact row-table locators: written across schema generations alone, so
        // a branch holds a handful of them against all of its rows.
        if table != super::VISIBLE_ROW_TABLE_LOCATOR_TABLE || !key.starts_with(prefix) {
            return false;
        }
        let storage_prefix = key_codec::raw_table_scan_prefix(table, prefix);
        let storage_key = key_codec::raw_table_entry_key(table, key);
        let memoized = self
            .exact_locator_keys
            .borrow()
            .known_absent(&storage_prefix, &storage_key);
        let absent = match memoized {
            Some(absent) => absent,
            None => {
                let mut end = storage_prefix.clone();
                key_codec::increment_string(&mut end);
                // No bound past a prefix that ends where UTF-8 cannot be incremented: an
                // empty one would read no keys and call every key absent.
                if end.is_empty() {
                    return false;
                }
                #[cfg(test)]
                self.exact_locator_scans
                    .set(self.exact_locator_scans.get() + 1);
                crate::query_manager::settle_cost::bump(
                    &crate::query_manager::settle_cost::STORAGE_READ_OPS,
                );
                let keys = crate::query_manager::settle_cost::timed(
                    &crate::query_manager::settle_cost::STORAGE_READ_MICROS,
                    || {
                        self.with_inner(|inner| {
                            Self::scan_range_keys_limited_from_db(
                                &inner.db,
                                &storage_prefix,
                                &end,
                                false,
                                MAX_MEMOIZED_EXACT_LOCATOR_KEYS + 1,
                            )
                        })
                    },
                );
                let Ok(keys) = keys else {
                    return false;
                };
                let keys = (keys.len() <= MAX_MEMOIZED_EXACT_LOCATOR_KEYS)
                    .then(|| keys.into_iter().collect::<HashSet<String>>());
                let absent = keys
                    .as_ref()
                    .is_some_and(|keys| !keys.contains(&storage_key));
                self.exact_locator_keys
                    .borrow_mut()
                    .remember(storage_prefix, keys);
                absent
            }
        };
        // Parity harness: a key the memo calls absent is not stored.
        #[cfg(debug_assertions)]
        if absent
            && let Ok(stored) = self.with_inner(|inner| Self::get_from_db(&inner.db, &storage_key))
        {
            debug_assert!(
                stored.is_none(),
                "the memo called a stored exact row-table locator absent: {storage_key:?}"
            );
        }
        absent
    }

    fn memoize_row_raw_table_ids(&self, header_prefix: &str, ids: &[super::RowRawTableId]) {
        let epoch = self.header_epoch.get();
        let mut memo = self.row_raw_table_ids.borrow_mut();
        if memo.0 != epoch {
            memo.1.clear();
            memo.0 = epoch;
        }
        memo.1.insert(header_prefix.to_string(), ids.to_vec());
    }

    fn raw_table_put(&mut self, table: &str, key: &str, value: &[u8]) -> Result<(), StorageError> {
        self.with_inner(|inner| {
            let txn = self.begin(&inner.db);
            raw_table_put_core(table, key, value, |storage_key, bytes| {
                txn.put(storage_key, bytes)
            })?;
            txn.commit()
        })
    }

    fn raw_table_delete(&mut self, table: &str, key: &str) -> Result<(), StorageError> {
        self.with_inner(|inner| {
            let txn = self.begin(&inner.db);
            raw_table_delete_core(table, key, |storage_key| txn.delete(storage_key))?;
            txn.commit()
        })
    }

    fn apply_raw_table_mutations(
        &mut self,
        mutations: &[RawTableMutation<'_>],
    ) -> Result<(), StorageError> {
        self.with_inner(|inner| {
            let txn = self.begin(&inner.db);
            for mutation in mutations {
                match mutation {
                    RawTableMutation::Put { table, key, value } => {
                        raw_table_put_core(table, key, value, |storage_key, bytes| {
                            txn.put(storage_key, bytes)
                        })?;
                    }
                    RawTableMutation::Delete { table, key } => {
                        raw_table_delete_core(table, key, |storage_key| txn.delete(storage_key))?;
                    }
                }
            }
            txn.commit()
        })
    }

    fn raw_table_get(&self, table: &str, key: &str) -> Result<Option<Vec<u8>>, StorageError> {
        #[cfg(test)]
        {
            *self
                .table_gets
                .borrow_mut()
                .entry(table.to_string())
                .or_default() += 1;
        }
        self.with_inner(|inner| {
            raw_table_get_core(table, key, |storage_key| {
                Self::get_from_db(&inner.db, storage_key)
            })
        })
    }

    fn apply_index_mutations(
        &mut self,
        mutations: &[IndexMutation<'_>],
    ) -> Result<(), StorageError> {
        if mutations.is_empty() {
            return Ok(());
        }

        self.with_inner(|inner| {
            let txn = self.begin(&inner.db);
            Self::apply_index_mutations_on_txn(&txn, mutations)?;
            txn.commit()
        })
    }

    fn raw_table_scan_prefix(
        &self,
        table: &str,
        prefix: &str,
    ) -> Result<super::RawTableRows, StorageError> {
        #[cfg(test)]
        self.prefix_scans.set(self.prefix_scans.get() + 1);
        self.with_inner(|inner| {
            raw_table_scan_prefix_core(table, prefix, |storage_prefix| {
                Self::scan_prefix_from_db(&inner.db, storage_prefix)
            })
        })
    }

    fn raw_table_scan_prefix_keys(
        &self,
        table: &str,
        prefix: &str,
    ) -> Result<super::RawTableKeys, StorageError> {
        #[cfg(test)]
        self.prefix_scans.set(self.prefix_scans.get() + 1);
        crate::query_manager::settle_cost::bump(
            &crate::query_manager::settle_cost::STORAGE_READ_OPS,
        );
        crate::query_manager::settle_cost::timed(
            &crate::query_manager::settle_cost::STORAGE_READ_MICROS,
            || {
                self.with_inner(|inner| {
                    raw_table_scan_prefix_keys_core(table, prefix, |storage_prefix| {
                        Self::scan_prefix_keys_from_db(&inner.db, storage_prefix)
                    })
                })
            },
        )
    }

    fn raw_table_first_key_with_prefix(
        &self,
        table: &str,
        prefix: &str,
    ) -> Result<Option<String>, StorageError> {
        self.with_inner(|inner| {
            raw_table_scan_prefix_keys_core(table, prefix, |storage_prefix| {
                Ok(
                    Self::first_key_with_prefix_from_db(&inner.db, storage_prefix)?
                        .into_iter()
                        .collect(),
                )
            })
        })
        .map(|keys| keys.into_iter().next())
    }

    /// Answered by two seeks, never a walk.
    ///
    /// History keys are `<row_id>:<branch>:<batch_id>`, so a row's versions are
    /// contiguous and grouped by branch. The first key under `<row_id>:` settles
    /// it when it belongs to another branch; otherwise one seek past the end of
    /// this branch's range says whether anything of this row remains. Walking
    /// forward from the first key instead would step through every version on the
    /// incoming branch — thousands, for the row this exists to stop reading.
    ///
    /// Every schema-hash table of the logical table is asked: a row's versions
    /// from before a schema deployment live under the older one.
    fn row_has_history_outside_branch(
        &self,
        table: &str,
        row_id: ObjectId,
        branch: &str,
    ) -> Result<bool, StorageError> {
        let row_prefix = super::key_codec::history_row_raw_table_prefix(Some(row_id));
        let branch_prefix = super::key_codec::history_row_raw_table_branch_prefix(row_id, branch);
        for resolved in
            super::resolved_row_tables_for_table(self, super::RowRawTableKind::History, table)?
        {
            let raw_table = resolved.row_raw_table.as_str();
            let Some(first) = self.raw_table_first_key_with_prefix(raw_table, &row_prefix)? else {
                continue;
            };
            if !first.starts_with(&branch_prefix) {
                return Ok(true);
            }
            if self.first_row_key_after_branch(raw_table, &row_prefix, &branch_prefix)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn raw_table_scan_range(
        &self,
        table: &str,
        start: Option<&str>,
        end: Option<&str>,
    ) -> Result<super::RawTableRows, StorageError> {
        self.with_inner(|inner| {
            raw_table_scan_range_core(table, start, end, |start_key, end_key| {
                Self::scan_range_from_db(&inner.db, start_key, end_key)
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
                Self::scan_range_keys_from_db(&inner.db, start_key, end_key)
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
                Self::scan_range_keys_limited_from_db(&inner.db, start_key, end_key, reverse, limit)
            })
        })
    }

    fn limited_range_scans_are_bounded(&self) -> bool {
        true
    }

    fn raw_table_family_keys(
        &self,
        name_prefix: &str,
        after: Option<&str>,
        limit: usize,
    ) -> Result<Vec<String>, StorageError> {
        self.with_inner(|inner| {
            raw_table_family_keys_core(name_prefix, after, limit, |start, end, limit| {
                Self::scan_range_keys_limited_from_db(&inner.db, start, end, false, limit)
            })
        })
    }

    fn raw_table_family_last_key(&self, name_prefix: &str) -> Result<Option<String>, StorageError> {
        self.with_inner(|inner| {
            raw_table_family_last_key_core(name_prefix, |start, end| {
                Self::scan_range_keys_limited_from_db(&inner.db, start, end, true, 1)
            })
        })
    }

    fn store_format_version(&self) -> Result<Option<i32>, StorageError> {
        self.with_inner(|inner| {
            Self::get_from_db(&inner.db, super::STORE_MANIFEST_KEY)?
                .map(|bytes| super::decode_store_manifest(&bytes))
                .transpose()
                .map(|manifest| manifest.map(|manifest| manifest.store_format_version))
        })
    }

    fn set_store_format_version(&mut self, version: i32) -> Result<(), StorageError> {
        let bytes = super::encode_store_manifest(&super::StoreManifest {
            store_kind: super::ROCKSDB_STORE_KIND.to_string(),
            store_format_version: version,
        })?;
        self.with_inner(|inner| {
            let txn = self.begin(&inner.db);
            txn.put(super::STORE_MANIFEST_KEY, &bytes)?;
            txn.commit()
        })
    }

    fn append_history_region_row_bytes(
        &mut self,
        table: &str,
        rows: &[HistoryRowBytes<'_>],
    ) -> Result<(), StorageError> {
        self.with_inner(|inner| {
            let txn = self.begin(&inner.db);
            append_history_region_row_bytes_core(table, rows, |key, bytes| txn.put(key, bytes))?;
            txn.commit()
        })
    }

    fn upsert_visible_region_row_bytes(
        &mut self,
        table: &str,
        rows: &[VisibleRowBytes<'_>],
    ) -> Result<(), StorageError> {
        self.with_inner(|inner| {
            let txn = self.begin(&inner.db);
            upsert_visible_region_row_bytes_core(table, rows, |key, bytes| txn.put(key, bytes))?;
            txn.commit()
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
            let txn = self.begin(&inner.db);
            let key = super::key_codec::visible_row_raw_table_key(branch, row_id);
            for raw_table in &raw_tables {
                raw_table_delete_core(raw_table.as_str(), &key, |storage_key| {
                    txn.delete(storage_key)
                })?;
            }
            raw_table_delete_core(
                super::VISIBLE_ROW_TABLE_LOCATOR_TABLE,
                &super::visible_row_table_locator_key(branch, row_id),
                |storage_key| txn.delete(storage_key),
            )?;
            txn.commit()
        })
    }

    fn apply_encoded_row_mutation(
        &mut self,
        table: &str,
        encoded_history_rows: &[super::OwnedHistoryRowBytes],
        encoded_visible_rows: &[super::OwnedVisibleRowBytes],
        index_mutations: &[IndexMutation<'_>],
    ) -> Result<(), StorageError> {
        self.with_inner_mut(|inner| {
            let txn = self.begin(&inner.db);
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
                        |storage_key, bytes| txn.put(storage_key, bytes),
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
                        |storage_key, bytes| txn.put(storage_key, bytes),
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
                    |storage_key, bytes| txn.put(storage_key, bytes),
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
                    |storage_key, bytes| txn.put(storage_key, bytes),
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
            append_history_region_row_bytes_core(table, &borrowed_history_rows, |key, bytes| {
                txn.put(key, bytes)
            })?;
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
                    |storage_key, bytes| txn.put(storage_key, bytes),
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
            upsert_visible_region_row_bytes_core(table, &borrowed_visible_rows, |key, bytes| {
                txn.put(key, bytes)
            })?;
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
                        |storage_key, bytes| txn.put(storage_key, bytes),
                    )?;
                    inner.visible_row_table_locators.insert(cache_key, locator);
                }
            }
            Self::apply_index_mutations_on_txn(&txn, index_mutations)?;
            txn.commit()
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

    fn flush(&self) -> Result<(), StorageError> {
        self.with_inner(|inner| {
            inner
                .db
                .flush()
                .map_err(|e| StorageError::IoError(format!("rocksdb flush: {e}")))
        })
    }

    fn flush_wal(&self) -> Result<(), StorageError> {
        self.with_inner(|inner| {
            inner
                .db
                .flush_wal(true)
                .map_err(|e| StorageError::IoError(format!("rocksdb flush_wal: {e}")))
        })
    }

    fn close(&self) -> Result<(), StorageError> {
        let Some(inner) = self.inner.borrow_mut().take() else {
            return Ok(());
        };
        drop(inner);
        // A closed store answers nothing, from memory either.
        self.row_raw_table_ids.borrow_mut().1.clear();
        self.row_locators.borrow_mut().clear();
        self.exact_locator_keys.borrow_mut().clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query_manager::types::{
        ColumnDescriptor, ColumnType, RowDescriptor, SchemaHash, Value,
    };
    use crate::storage::{RowRawTableId, RowRawTableKind};

    fn temp_store() -> (tempfile::TempDir, RocksDBStorage) {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let storage =
            RocksDBStorage::open(temp_dir.path().join("test.rocksdb"), 8 * 1024 * 1024).unwrap();
        (temp_dir, storage)
    }

    fn visible_table(table: &str, generation: u8) -> (RowRawTableId, Vec<u8>) {
        let id = RowRawTableId::new(
            RowRawTableKind::Visible,
            table,
            SchemaHash::from_bytes([generation; 32]),
        );
        let descriptor = RowDescriptor::new(vec![ColumnDescriptor::new("title", ColumnType::Text)]);
        let header = super::super::encode_raw_table_header(&super::super::row_raw_table_header(
            &id,
            &descriptor,
        ))
        .unwrap();
        (id, header)
    }

    fn put_header(storage: &mut RocksDBStorage, table: &str, generation: u8) -> RowRawTableId {
        let (id, header) = visible_table(table, generation);
        storage
            .raw_table_put(
                super::super::RAW_TABLE_HEADER_TABLE,
                id.raw_table_name(),
                &header,
            )
            .unwrap();
        id
    }

    fn raw_tables_of(storage: &RocksDBStorage, table: &str) -> Vec<String> {
        super::super::row_raw_table_ids_for_table(storage, RowRawTableKind::Visible, table)
            .unwrap()
            .iter()
            .map(|id| id.raw_table_name().to_string())
            .collect()
    }

    /// A row load asks which raw tables its table has. That was a scan of the header
    /// table per row; it is one per table until a header is written.
    #[test]
    fn the_raw_tables_of_a_table_are_read_from_the_headers_once() {
        let (_dir, mut storage) = temp_store();
        let first = put_header(&mut storage, "notes", 0x11);
        let second = put_header(&mut storage, "notes", 0x22);
        put_header(&mut storage, "tasks", 0x11);

        let before = storage.prefix_scans.get();
        for _ in 0..50 {
            assert_eq!(
                raw_tables_of(&storage, "notes"),
                vec![
                    first.raw_table_name().to_string(),
                    second.raw_table_name().to_string()
                ]
            );
        }
        assert_eq!(
            storage.prefix_scans.get() - before,
            1,
            "fifty reads of one table's raw tables"
        );
        assert_eq!(raw_tables_of(&storage, "tasks").len(), 1);
        assert_eq!(raw_tables_of(&storage, "tasks").len(), 1);
        assert_eq!(
            storage.prefix_scans.get() - before,
            2,
            "another table is another read, once"
        );
    }

    /// A table gets a raw table when a row of a new schema generation is written, and
    /// loses one when a header is deleted: what was read before is not served after,
    /// whichever way the header was written.
    #[test]
    fn a_header_written_or_deleted_after_the_raw_tables_were_read_is_seen() {
        let (_dir, mut storage) = temp_store();
        let first = put_header(&mut storage, "notes", 0x11);
        assert_eq!(raw_tables_of(&storage, "notes").len(), 1);

        let second = put_header(&mut storage, "notes", 0x22);
        assert_eq!(
            raw_tables_of(&storage, "notes"),
            vec![
                first.raw_table_name().to_string(),
                second.raw_table_name().to_string()
            ],
            "a header put on its own"
        );

        let (third, header) = visible_table("notes", 0x33);
        storage
            .apply_raw_table_mutations(&[RawTableMutation::Put {
                table: super::super::RAW_TABLE_HEADER_TABLE,
                key: third.raw_table_name(),
                value: &header,
            }])
            .unwrap();
        assert_eq!(
            raw_tables_of(&storage, "notes").len(),
            3,
            "a header put among other mutations"
        );

        storage
            .raw_table_delete(
                super::super::RAW_TABLE_HEADER_TABLE,
                second.raw_table_name(),
            )
            .unwrap();
        assert_eq!(
            raw_tables_of(&storage, "notes"),
            vec![
                first.raw_table_name().to_string(),
                third.raw_table_name().to_string()
            ],
            "a header deleted"
        );
    }

    fn row_locator(table: &str) -> super::super::RowLocator {
        super::super::RowLocator {
            table: table.into(),
            origin_schema_hash: Some(SchemaHash::from_bytes([0x11; 32])),
        }
    }

    fn exact_locator(table: &str) -> super::super::ExactRowTableLocator {
        super::super::ExactRowTableLocator {
            row_raw_table: format!("rowtable:visible:{table}:11").into(),
            table_name: table.into(),
            schema_hash: SchemaHash::from_bytes([0x11; 32]),
        }
    }

    fn gets_of(storage: &RocksDBStorage, table: &str) -> usize {
        storage.table_gets.borrow().get(table).copied().unwrap_or(0)
    }

    fn row_locator_storage_key(id: ObjectId) -> String {
        key_codec::raw_table_entry_key(
            super::super::ROW_LOCATOR_TABLE,
            &super::super::metadata_raw_key(id),
        )
    }

    fn exact_locator_storage_key(branch: &str, id: ObjectId) -> String {
        key_codec::raw_table_entry_key(
            super::super::VISIBLE_ROW_TABLE_LOCATOR_TABLE,
            &key_codec::visible_row_raw_table_key(branch, id),
        )
    }

    fn exact_locator_known_absent(storage: &RocksDBStorage, branch: &str, id: ObjectId) -> bool {
        storage.raw_table_key_known_absent(
            super::super::VISIBLE_ROW_TABLE_LOCATOR_TABLE,
            &key_codec::visible_row_raw_table_prefix(branch),
            &key_codec::visible_row_raw_table_key(branch, id),
        )
    }

    /// A row load asks for the row's locator and whether the row has an exact row-table
    /// locator. Read twice, twenty rows read their locators from the store once, and the
    /// exact locators never: the branch holds none, and one read of its keys says so.
    #[test]
    fn a_row_load_reads_its_locator_once_and_no_exact_locator() {
        use crate::metadata::RowProvenance;
        use crate::query_manager::types::{SchemaBuilder, TableSchema};
        use crate::row_histories::VisibleRowEntry;

        let (_dir, mut storage) = temp_store();
        let schema = SchemaBuilder::new()
            .table(TableSchema::builder("users").column("value", ColumnType::Text))
            .build();
        let schema_hash = crate::test_support::persist_test_schema(&mut storage, &schema);
        let descriptor = RowDescriptor::new(vec![ColumnDescriptor::new("value", ColumnType::Text)]);
        let mut rows = Vec::new();
        for index in 0..20u64 {
            let row_id = ObjectId::new();
            storage
                .put_row_locator(
                    row_id,
                    Some(&super::super::RowLocator {
                        table: "users".into(),
                        origin_schema_hash: Some(schema_hash),
                    }),
                )
                .unwrap();
            let version = StoredRowBatch::new(
                row_id,
                "main",
                Vec::new(),
                crate::row_format::encode_row(&descriptor, &[Value::Text(format!("row {index}"))])
                    .unwrap(),
                RowProvenance::for_insert(row_id.to_string(), 10 + index),
                HashMap::new(),
                RowState::VisibleDirect,
                None,
            );
            storage
                .append_history_region_rows("users", std::slice::from_ref(&version))
                .unwrap();
            storage
                .upsert_visible_region_rows(
                    "users",
                    std::slice::from_ref(&VisibleRowEntry::rebuild(
                        version.clone(),
                        std::slice::from_ref(&version),
                    )),
                )
                .unwrap();
            rows.push(version);
        }

        let locators_before = gets_of(&storage, super::super::ROW_LOCATOR_TABLE);
        let exact_before = gets_of(&storage, super::super::VISIBLE_ROW_TABLE_LOCATOR_TABLE);
        for row in &rows {
            assert_eq!(
                storage
                    .load_visible_region_row("users", "main", row.row_id)
                    .unwrap()
                    .as_ref(),
                Some(row)
            );
        }
        let locators_first = gets_of(&storage, super::super::ROW_LOCATOR_TABLE) - locators_before;
        assert!(
            locators_first <= rows.len(),
            "at most a read of each row's locator: {locators_first}"
        );
        for row in &rows {
            assert_eq!(
                storage
                    .load_visible_region_row("users", "main", row.row_id)
                    .unwrap()
                    .as_ref(),
                Some(row)
            );
        }
        assert_eq!(
            gets_of(&storage, super::super::ROW_LOCATOR_TABLE) - locators_before,
            locators_first,
            "the second load of each row reads no locator"
        );
        assert_eq!(
            gets_of(&storage, super::super::VISIBLE_ROW_TABLE_LOCATOR_TABLE) - exact_before,
            0,
            "no row has an exact locator, and none is read"
        );
        assert_eq!(
            storage.exact_locator_scans.get(),
            1,
            "one read of the branch"
        );
    }

    /// A locator is written, read, written again, deleted, written: every read after a
    /// committed write sees it.
    #[test]
    fn a_locator_written_or_deleted_after_it_was_read_is_seen() {
        let (_dir, mut storage) = temp_store();
        let id = ObjectId::new();
        assert_eq!(storage.load_row_locator(id).unwrap(), None);
        storage
            .put_row_locator(id, Some(&row_locator("notes")))
            .unwrap();
        assert_eq!(
            storage.load_row_locator(id).unwrap(),
            Some(row_locator("notes"))
        );
        storage
            .put_row_locator(id, Some(&row_locator("todos")))
            .unwrap();
        assert_eq!(
            storage.load_row_locator(id).unwrap(),
            Some(row_locator("todos"))
        );
        storage.put_row_locator(id, None).unwrap();
        assert_eq!(storage.load_row_locator(id).unwrap(), None);
        storage
            .put_row_locator(id, Some(&row_locator("notes")))
            .unwrap();
        assert_eq!(
            storage.load_row_locator(id).unwrap(),
            Some(row_locator("notes"))
        );

        let before = gets_of(&storage, super::super::ROW_LOCATOR_TABLE);
        for _ in 0..10 {
            assert_eq!(
                storage.load_row_locator(id).unwrap(),
                Some(row_locator("notes"))
            );
        }
        assert_eq!(
            gets_of(&storage, super::super::ROW_LOCATOR_TABLE) - before,
            0,
            "an unchanged locator is answered from memory"
        );
    }

    /// A locator put on a transaction is not readable until it commits: what a read in
    /// between finds is not served after the commit, and a transaction dropped before it
    /// commits changes nothing.
    #[test]
    fn what_is_read_of_a_locator_before_its_write_commits_does_not_outlive_the_commit() {
        let (_dir, mut storage) = temp_store();
        let id = ObjectId::new();
        storage
            .put_row_locator(id, Some(&row_locator("notes")))
            .unwrap();
        assert_eq!(
            storage.load_row_locator(id).unwrap(),
            Some(row_locator("notes"))
        );

        let bytes = super::super::encode_row_locator(&row_locator("todos")).unwrap();
        storage
            .with_inner(|inner| {
                let txn = storage.begin(&inner.db);
                txn.put(&row_locator_storage_key(id), &bytes)?;
                assert_eq!(
                    storage.load_row_locator(id).unwrap(),
                    Some(row_locator("notes")),
                    "not committed yet"
                );
                Ok(())
            })
            .unwrap();
        assert_eq!(
            storage.load_row_locator(id).unwrap(),
            Some(row_locator("notes")),
            "the transaction was dropped"
        );

        storage
            .with_inner(|inner| {
                let txn = storage.begin(&inner.db);
                txn.put(&row_locator_storage_key(id), &bytes)?;
                assert_eq!(
                    storage.load_row_locator(id).unwrap(),
                    Some(row_locator("notes")),
                    "not committed yet"
                );
                txn.commit()
            })
            .unwrap();
        assert_eq!(
            storage.load_row_locator(id).unwrap(),
            Some(row_locator("todos")),
            "committed"
        );
    }

    /// A closed store answers no locator, from memory either: no row locator, and no
    /// exact locator known absent.
    #[test]
    fn the_row_locators_of_a_closed_store_are_not_answered_from_memory() {
        let (_dir, mut storage) = temp_store();
        let id = ObjectId::new();
        storage
            .put_row_locator(id, Some(&row_locator("notes")))
            .unwrap();
        assert_eq!(
            storage.load_row_locator(id).unwrap(),
            Some(row_locator("notes"))
        );
        assert!(exact_locator_known_absent(&storage, "main", id));
        storage.close().unwrap();
        assert!(
            !exact_locator_known_absent(&storage, "main", id),
            "a closed store knows nothing absent"
        );
        assert!(
            storage.load_row_locator(id).is_err(),
            "a read of a closed store"
        );
    }

    /// The memo of row locators holds a bounded number of them.
    #[test]
    fn the_memo_of_row_locators_is_bounded() {
        let (_dir, storage) = temp_store();
        for _ in 0..MAX_MEMOIZED_ROW_LOCATORS + 5 {
            assert_eq!(storage.load_row_locator(ObjectId::new()).unwrap(), None);
            assert!(storage.row_locators.borrow().len() <= MAX_MEMOIZED_ROW_LOCATORS);
        }
    }

    /// Whether a row has an exact row-table locator is answered from one read of its
    /// branch's keys; a committed write of one is put into or taken out of every branch's
    /// keys it falls under, and no branch is read again for it.
    #[test]
    fn an_exact_locator_is_known_absent_from_one_read_of_its_branch() {
        let (_dir, mut storage) = temp_store();
        let (a, b) = (ObjectId::new(), ObjectId::new());
        storage
            .put_visible_row_table_locator("main", a, Some(&exact_locator("notes")))
            .unwrap();
        for _ in 0..10 {
            assert!(!exact_locator_known_absent(&storage, "main", a));
            assert!(exact_locator_known_absent(&storage, "main", b));
        }
        assert_eq!(storage.exact_locator_scans.get(), 1, "one read of main");
        assert!(exact_locator_known_absent(&storage, "draft", a));
        assert_eq!(storage.exact_locator_scans.get(), 2, "and one of draft");
        storage
            .put_visible_row_table_locator("draft", b, Some(&exact_locator("notes")))
            .unwrap();
        assert!(
            !storage.raw_table_key_known_absent(
                super::super::VISIBLE_ROW_TABLE_LOCATOR_TABLE,
                &key_codec::visible_row_raw_table_prefix("main"),
                &key_codec::visible_row_raw_table_key("draft", b),
            ),
            "a key is not answered under a prefix it is not under"
        );
        assert!(exact_locator_known_absent(&storage, "main", b));
        assert!(
            !exact_locator_known_absent(&storage, "draft", b),
            "b's on draft"
        );
        assert!(
            !storage.raw_table_key_known_absent(
                super::super::ROW_LOCATOR_TABLE,
                "",
                &super::super::metadata_raw_key(b),
            ),
            "no other table is answered"
        );

        storage
            .put_visible_row_table_locator("main", b, Some(&exact_locator("notes")))
            .unwrap();
        assert!(
            !exact_locator_known_absent(&storage, "main", b),
            "b's is written"
        );
        storage
            .put_visible_row_table_locator("main", a, None)
            .unwrap();
        assert!(
            exact_locator_known_absent(&storage, "main", a),
            "a's is deleted"
        );
        assert!(!exact_locator_known_absent(&storage, "main", b));
        assert_eq!(
            storage.exact_locator_scans.get(),
            2,
            "no write made a branch be read again"
        );
    }

    /// A branch that holds more exact locators than the memo does is read once, and a write
    /// does not make it be read again; one that grows past the memo by a write is not
    /// answered from then on, and not read again either.
    #[test]
    fn writes_to_a_branch_read_once_do_not_make_it_be_read_again() {
        let (_dir, mut storage) = temp_store();
        for _ in 0..MAX_MEMOIZED_EXACT_LOCATOR_KEYS + 2 {
            storage
                .put_visible_row_table_locator(
                    "big",
                    ObjectId::new(),
                    Some(&exact_locator("notes")),
                )
                .unwrap();
        }
        let mut held = Vec::new();
        for _ in 0..MAX_MEMOIZED_EXACT_LOCATOR_KEYS {
            let id = ObjectId::new();
            storage
                .put_visible_row_table_locator("full", id, Some(&exact_locator("notes")))
                .unwrap();
            held.push(id);
        }
        assert!(!exact_locator_known_absent(
            &storage,
            "big",
            ObjectId::new()
        ));
        assert!(exact_locator_known_absent(
            &storage,
            "full",
            ObjectId::new()
        ));
        assert_eq!(storage.exact_locator_scans.get(), 2);
        for _ in 0..20 {
            let id = ObjectId::new();
            storage
                .put_visible_row_table_locator("big", id, Some(&exact_locator("notes")))
                .unwrap();
            storage
                .put_visible_row_table_locator("big", id, None)
                .unwrap();
            storage
                .put_visible_row_table_locator("full", id, None)
                .unwrap();
            assert!(!exact_locator_known_absent(&storage, "big", id));
            assert!(exact_locator_known_absent(&storage, "full", id));
        }
        assert!(!exact_locator_known_absent(&storage, "full", held[0]));
        let id = ObjectId::new();
        storage
            .put_visible_row_table_locator("full", id, Some(&exact_locator("notes")))
            .unwrap();
        assert!(
            !exact_locator_known_absent(&storage, "full", ObjectId::new()),
            "past the memo"
        );
        storage
            .put_visible_row_table_locator("full", id, None)
            .unwrap();
        assert!(
            !exact_locator_known_absent(&storage, "full", ObjectId::new()),
            "a delete does not say how many are left"
        );
        assert_eq!(
            storage.exact_locator_scans.get(),
            2,
            "no write made a branch be read again"
        );
    }

    /// A prefix past which no UTF-8 string bounds a read is not answered: a read with no
    /// keys would call every key absent.
    #[test]
    fn a_prefix_with_no_bound_past_it_is_not_answered() {
        let (_dir, storage) = temp_store();
        assert!(!storage.raw_table_key_known_absent(
            super::super::VISIBLE_ROW_TABLE_LOCATOR_TABLE,
            "\u{43f}",
            "\u{43f}:1",
        ));
        assert_eq!(storage.exact_locator_scans.get(), 0);
    }

    /// The keys the memo holds, counted, against its own count.
    fn exact_locator_keys_held(storage: &RocksDBStorage) -> usize {
        let memo = storage.exact_locator_keys.borrow();
        let counted = memo
            .by_prefix
            .values()
            .map(|keys| keys.as_ref().map_or(0, HashSet::len))
            .sum::<usize>();
        assert_eq!(
            memo.keys_held, counted,
            "the memo's count of the keys it holds"
        );
        counted
    }

    /// The memo holds the keys of a bounded number of branches.
    #[test]
    fn the_memo_of_exact_locator_keys_holds_a_bounded_number_of_branches() {
        let (_dir, storage) = temp_store();
        for branch in 0..MAX_MEMOIZED_EXACT_LOCATOR_PREFIXES + 5 {
            assert!(exact_locator_known_absent(
                &storage,
                &format!("branch{branch}"),
                ObjectId::new()
            ));
            assert!(
                storage.exact_locator_keys.borrow().by_prefix.len()
                    <= MAX_MEMOIZED_EXACT_LOCATOR_PREFIXES
            );
        }
        assert_eq!(
            storage.exact_locator_keys.borrow().by_prefix.len(),
            MAX_MEMOIZED_EXACT_LOCATOR_PREFIXES,
            "a branch past the bound makes room for itself alone, not by dropping the rest"
        );
    }

    /// The memo holds a bounded number of keys over all branches, by its own count and by
    /// the sets it holds, whether the keys come in with a branch's read or with writes after.
    #[test]
    fn the_memo_of_exact_locator_keys_holds_a_bounded_number_of_keys() {
        let (_dir, mut storage) = temp_store();
        // The table's header first: writing it drops whatever was read of the table.
        storage
            .put_visible_row_table_locator("seed", ObjectId::new(), Some(&exact_locator("notes")))
            .unwrap();
        // Branches that free nothing when forgotten: ones holding no key, and ones past what
        // the memo holds of a branch. Making room for keys must forget neither.
        let empty_branches = 1000;
        let wide_branches = 20;
        for branch in 0..wide_branches {
            for _ in 0..=MAX_MEMOIZED_EXACT_LOCATOR_KEYS {
                storage
                    .put_visible_row_table_locator(
                        &format!("wide{branch}"),
                        ObjectId::new(),
                        Some(&exact_locator("notes")),
                    )
                    .unwrap();
            }
        }
        let read_again = |storage: &RocksDBStorage| {
            let scans = storage.exact_locator_scans.get();
            for branch in 0..empty_branches {
                assert!(exact_locator_known_absent(
                    storage,
                    &format!("empty{branch}"),
                    ObjectId::new()
                ));
            }
            for branch in 0..wide_branches {
                assert!(!exact_locator_known_absent(
                    storage,
                    &format!("wide{branch}"),
                    ObjectId::new()
                ));
            }
            storage.exact_locator_scans.get() - scans
        };
        assert_eq!(
            read_again(&storage),
            empty_branches + wide_branches,
            "each branch is read once"
        );
        let branches = MAX_MEMOIZED_EXACT_LOCATOR_KEYS_HELD / 100 + 5;
        for branch in 0..branches {
            let branch = format!("branch{branch}");
            for _ in 0..100 {
                storage
                    .put_visible_row_table_locator(
                        &branch,
                        ObjectId::new(),
                        Some(&exact_locator("notes")),
                    )
                    .unwrap();
            }
            assert!(exact_locator_known_absent(
                &storage,
                &branch,
                ObjectId::new()
            ));
            assert!(exact_locator_keys_held(&storage) <= MAX_MEMOIZED_EXACT_LOCATOR_KEYS_HELD);
        }
        assert!(
            exact_locator_keys_held(&storage) > MAX_MEMOIZED_EXACT_LOCATOR_KEYS_HELD - 200,
            "a branch read past the bound makes room for itself alone, not by dropping the rest"
        );
        assert_eq!(
            read_again(&storage),
            0,
            "making room for a branch read forgets a branch that holds keys"
        );
        for branch in 0..branches {
            let branch = format!("branch{branch}");
            for _ in 0..20 {
                storage
                    .put_visible_row_table_locator(
                        &branch,
                        ObjectId::new(),
                        Some(&exact_locator("notes")),
                    )
                    .unwrap();
                assert!(exact_locator_keys_held(&storage) <= MAX_MEMOIZED_EXACT_LOCATOR_KEYS_HELD);
            }
        }
        assert!(
            exact_locator_keys_held(&storage) > MAX_MEMOIZED_EXACT_LOCATOR_KEYS_HELD - 200,
            "writes past the bound make room by dropping a branch, not every branch"
        );
        assert_eq!(
            read_again(&storage),
            0,
            "making room for written keys forgets a branch that holds keys"
        );
    }

    /// Making room for keys forgets a branch that holds keys: one past what the memo holds of
    /// a branch, or one holding none, frees nothing.
    #[test]
    fn making_room_for_keys_forgets_a_branch_that_holds_keys() {
        let mut memo = ExactLocatorKeys::default();
        memo.remember("past:".to_string(), None);
        memo.remember("none:".to_string(), Some(HashSet::new()));
        memo.remember(
            "held:".to_string(),
            Some(HashSet::from(["held:k".to_string()])),
        );
        assert!(memo.forget_one_holding_keys());
        assert_eq!(memo.keys_held, 0);
        let mut left: Vec<&str> = memo.by_prefix.keys().map(String::as_str).collect();
        left.sort();
        assert_eq!(left, ["none:", "past:"]);
        assert!(
            !memo.forget_one_holding_keys(),
            "nothing left that holds keys"
        );
        assert_eq!(memo.by_prefix.len(), 2);
    }

    /// More branches than a handful, read over and over in turn, are each read from the
    /// store once.
    #[test]
    fn branches_read_in_turn_are_each_read_once() {
        let (_dir, storage) = temp_store();
        for _ in 0..3 {
            for branch in 0..70 {
                assert!(exact_locator_known_absent(
                    &storage,
                    &format!("branch{branch}"),
                    ObjectId::new()
                ));
            }
        }
        assert_eq!(storage.exact_locator_scans.get(), 70);
    }

    /// A commit that fails leaves the store as it was: a delete of an exact locator it
    /// carried is not taken for done.
    #[test]
    fn a_commit_that_fails_takes_no_exact_locator_out() {
        let (_dir, mut storage) = temp_store();
        let id = ObjectId::new();
        storage
            .put_visible_row_table_locator("main", id, Some(&exact_locator("notes")))
            .unwrap();
        assert!(!exact_locator_known_absent(&storage, "main", id));
        let failed = storage.with_inner(|inner| {
            let txn = storage.begin(&inner.db);
            txn.delete(&exact_locator_storage_key("main", id))?;
            store_db::fail_next_commit();
            txn.commit()
        });
        assert!(
            matches!(&failed, Err(StorageError::IoError(message)) if message.contains("failed by a test")),
            "the commit fails, by the hook: {failed:?}"
        );
        assert!(
            !exact_locator_known_absent(&storage, "main", id),
            "the locator is still stored"
        );
        exact_locator_keys_held(&storage);
    }

    /// A locator table's header rewritten: what was read under the old one is not
    /// answered after the commit.
    #[test]
    fn a_locator_table_header_written_drops_what_was_read_of_the_table() {
        let (_dir, mut storage) = temp_store();
        let id = ObjectId::new();
        storage
            .put_row_locator(id, Some(&row_locator("notes")))
            .unwrap();
        storage
            .put_visible_row_table_locator("main", id, Some(&exact_locator("notes")))
            .unwrap();
        assert!(storage.load_row_locator(id).unwrap().is_some());
        assert!(!exact_locator_known_absent(&storage, "main", id));
        for (table, row_locators_left, exact_locator_branches_left) in [
            (super::super::ROW_LOCATOR_TABLE, 0, 1),
            (super::super::VISIBLE_ROW_TABLE_LOCATOR_TABLE, 0, 0),
        ] {
            let header_key =
                key_codec::raw_table_entry_key(super::super::RAW_TABLE_HEADER_TABLE, table);
            let header = storage
                .with_inner(|inner| RocksDBStorage::get_from_db(&inner.db, &header_key))
                .unwrap()
                .expect("the table has a header");
            storage
                .with_inner(|inner| {
                    let txn = storage.begin(&inner.db);
                    txn.put(&header_key, &header)?;
                    txn.commit()
                })
                .unwrap();
            assert_eq!(
                storage.row_locators.borrow().len(),
                row_locators_left,
                "{table}"
            );
            assert_eq!(
                storage.exact_locator_keys.borrow().by_prefix.len(),
                exact_locator_branches_left,
                "{table}"
            );
        }
    }

    /// An exact locator put on a transaction is not there until it commits: what a read
    /// in between finds is not what a read after the commit is answered.
    #[test]
    fn an_exact_locator_read_before_its_commit_does_not_outlive_it() {
        let (_dir, mut storage) = temp_store();
        let (a, b) = (ObjectId::new(), ObjectId::new());
        storage
            .put_visible_row_table_locator("main", a, Some(&exact_locator("notes")))
            .unwrap();
        let bytes = super::super::encode_exact_row_table_locator(&exact_locator("notes")).unwrap();
        storage
            .with_inner(|inner| {
                let txn = storage.begin(&inner.db);
                txn.put(&exact_locator_storage_key("main", b), &bytes)?;
                assert!(
                    exact_locator_known_absent(&storage, "main", b),
                    "not committed yet"
                );
                txn.commit()
            })
            .unwrap();
        assert!(
            !exact_locator_known_absent(&storage, "main", b),
            "committed"
        );
    }

    /// A branch with more exact locators than the memo holds is not answered from memory;
    /// one with as many is.
    #[test]
    fn a_branch_with_more_exact_locators_than_the_memo_holds_is_not_answered() {
        let (_dir, mut storage) = temp_store();
        for _ in 0..MAX_MEMOIZED_EXACT_LOCATOR_KEYS {
            storage
                .put_visible_row_table_locator(
                    "main",
                    ObjectId::new(),
                    Some(&exact_locator("notes")),
                )
                .unwrap();
        }
        assert!(exact_locator_known_absent(
            &storage,
            "main",
            ObjectId::new()
        ));
        storage
            .put_visible_row_table_locator("main", ObjectId::new(), Some(&exact_locator("notes")))
            .unwrap();
        assert!(
            !exact_locator_known_absent(&storage, "main", ObjectId::new()),
            "one more than the memo holds"
        );
    }

    /// Random writes of row locators and exact locators — committed, or dropped before
    /// they commit with a read in between — on two branches one of whose names starts the
    /// other's ("a" and "a:b"), with the store closed and reopened now and then, against a
    /// model of what is committed: every locator read equals the model's, and a key is
    /// called absent exactly when the model holds no such key and its branch has held no
    /// more keys than the memo does since the branch was first read.
    #[test]
    fn locator_reads_match_a_model_under_random_writes() {
        struct Rng(u64);
        impl Rng {
            fn below(&mut self, bound: u64) -> u64 {
                self.0 ^= self.0 << 13;
                self.0 ^= self.0 >> 7;
                self.0 ^= self.0 << 17;
                self.0 % bound
            }
        }
        fn storage_prefix(branch: &str) -> String {
            key_codec::raw_table_scan_prefix(
                super::super::VISIBLE_ROW_TABLE_LOCATOR_TABLE,
                &key_codec::visible_row_raw_table_prefix(branch),
            )
        }
        let branches = ["a", "a:b"];
        let held = |exact: &HashSet<String>, branch: usize| {
            let prefix = storage_prefix(branches[branch]);
            exact.iter().filter(|key| key.starts_with(&prefix)).count()
        };
        let (mut answered, mut too_many, mut reopened) = (0usize, 0usize, 0usize);
        // Two seeds start "a" with almost as many exact locators as the memo holds, for the
        // writes to take it past that.
        for (seed, ids_per_branch, prefilled) in [
            (1u64, 24usize, 0usize),
            (2, 24, 0),
            (3, 160, 124),
            (4, 24, 0),
            (5, 160, 124),
        ] {
            let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
            let dir = tempfile::TempDir::new().unwrap();
            let path = dir.path().join("test.rocksdb");
            let mut storage = RocksDBStorage::open(&path, 8 * 1024 * 1024).unwrap();
            let ids: Vec<ObjectId> = (0..ids_per_branch).map(|_| ObjectId::new()).collect();
            let tables = ["notes", "todos", "users"];
            let mut locators: HashMap<ObjectId, super::super::RowLocator> = HashMap::new();
            // The committed exact locator keys, as stored.
            let mut exact: HashSet<String> = HashSet::new();
            // Per branch: `None` until it is read after the store opened, then whether it
            // has held more keys than the memo does since.
            let mut past_the_memo: [Option<bool>; 2] = [None, None];
            // The raw writes below go past the header a table's first write puts.
            let first = ObjectId::new();
            storage
                .put_row_locator(first, Some(&row_locator("notes")))
                .unwrap();
            storage.put_row_locator(first, None).unwrap();
            storage
                .put_visible_row_table_locator("a", first, Some(&exact_locator("notes")))
                .unwrap();
            storage
                .put_visible_row_table_locator("a", first, None)
                .unwrap();
            for id in &ids[..prefilled] {
                storage
                    .put_visible_row_table_locator("a", *id, Some(&exact_locator("notes")))
                    .unwrap();
                exact.insert(exact_locator_storage_key("a", *id));
            }
            let put_exact = |exact: &mut HashSet<String>,
                             past_the_memo: &mut [Option<bool>; 2],
                             key: String| {
                exact.insert(key.clone());
                for branch in 0..2 {
                    if past_the_memo[branch] == Some(false)
                        && key.starts_with(&storage_prefix(branches[branch]))
                        && held(exact, branch) > MAX_MEMOIZED_EXACT_LOCATOR_KEYS
                    {
                        past_the_memo[branch] = Some(true);
                    }
                }
            };
            for step in 0..1500 {
                let id = ids[rng.below(ids.len() as u64) as usize];
                let branch = rng.below(2) as usize;
                match rng.below(10) {
                    0 | 1 => {
                        if rng.below(4) == 0 {
                            storage.put_row_locator(id, None).unwrap();
                            locators.remove(&id);
                        } else {
                            let locator = row_locator(tables[rng.below(3) as usize]);
                            storage.put_row_locator(id, Some(&locator)).unwrap();
                            locators.insert(id, locator);
                        }
                    }
                    2 | 3 => {
                        if rng.below(4) == 0 {
                            storage
                                .put_visible_row_table_locator(branches[branch], id, None)
                                .unwrap();
                            exact.remove(&exact_locator_storage_key(branches[branch], id));
                        } else {
                            storage
                                .put_visible_row_table_locator(
                                    branches[branch],
                                    id,
                                    Some(&exact_locator(tables[rng.below(3) as usize])),
                                )
                                .unwrap();
                            put_exact(
                                &mut exact,
                                &mut past_the_memo,
                                exact_locator_storage_key(branches[branch], id),
                            );
                        }
                    }
                    4 => {
                        // A write that commits, or not, with reads in between.
                        let commit = rng.below(2) == 0;
                        let locator = row_locator(tables[rng.below(3) as usize]);
                        let bytes = super::super::encode_row_locator(&locator).unwrap();
                        let exact_bytes =
                            super::super::encode_exact_row_table_locator(&exact_locator("notes"))
                                .unwrap();
                        let exact_key = exact_locator_storage_key(branches[branch], id);
                        let model_locators = &locators;
                        let model_exact = &exact;
                        storage
                            .with_inner(|inner| {
                                let txn = storage.begin(&inner.db);
                                txn.put(&row_locator_storage_key(id), &bytes)?;
                                txn.put(&exact_key, &exact_bytes)?;
                                assert_eq!(
                                    storage.load_row_locator(id).unwrap().as_ref(),
                                    model_locators.get(&id),
                                    "seed {seed}, step {step}: before the commit"
                                );
                                let absent =
                                    exact_locator_known_absent(&storage, branches[branch], id);
                                assert!(
                                    !absent || !model_exact.contains(&exact_key),
                                    "seed {seed}, step {step}: a stored exact locator called absent"
                                );
                                if commit { txn.commit() } else { Ok(()) }
                            })
                            .unwrap();
                        // The read in between may have been the branch's first.
                        if past_the_memo[branch].is_none() {
                            past_the_memo[branch] =
                                Some(held(&exact, branch) > MAX_MEMOIZED_EXACT_LOCATOR_KEYS);
                        }
                        if commit {
                            locators.insert(id, locator);
                            put_exact(&mut exact, &mut past_the_memo, exact_key);
                        }
                    }
                    5 => {
                        if rng.below(20) == 0 {
                            // Closed and opened again: nothing is remembered.
                            storage.close().unwrap();
                            drop(storage);
                            storage = RocksDBStorage::open(&path, 8 * 1024 * 1024).unwrap();
                            past_the_memo = [None, None];
                            reopened += 1;
                        } else {
                            // An unrelated write.
                            storage
                                .raw_table_put("anything", &step.to_string(), b"v")
                                .unwrap();
                        }
                    }
                    _ => {
                        assert_eq!(
                            storage.load_row_locator(id).unwrap().as_ref(),
                            locators.get(&id),
                            "seed {seed}, step {step}: the locator of a row"
                        );
                        let absent = exact_locator_known_absent(&storage, branches[branch], id);
                        let past = *past_the_memo[branch].get_or_insert_with(|| {
                            held(&exact, branch) > MAX_MEMOIZED_EXACT_LOCATOR_KEYS
                        });
                        if past {
                            too_many += 1;
                        } else {
                            answered += 1;
                        }
                        let stored =
                            exact.contains(&exact_locator_storage_key(branches[branch], id));
                        assert_eq!(
                            absent,
                            !stored && !past,
                            "seed {seed}, step {step}: absent, with {} keys under the branch",
                            held(&exact, branch)
                        );
                        exact_locator_keys_held(&storage);
                    }
                }
            }
        }
        assert!(
            answered > 500 && too_many > 100 && reopened > 10,
            "checks answered from memory: {answered}, past the memo: {too_many}; reopened {reopened} times"
        );
    }

    /// A header put on a transaction is not readable until the transaction commits: a
    /// read in between finds the table without it, and must not be what a read after
    /// the commit is served.
    #[test]
    fn what_is_read_before_a_header_is_committed_does_not_outlive_the_commit() {
        let (_dir, mut storage) = temp_store();
        put_header(&mut storage, "notes", 0x11);
        assert_eq!(raw_tables_of(&storage, "notes").len(), 1);

        let (second, header) = visible_table("notes", 0x22);
        let storage_key = key_codec::raw_table_entry_key(
            super::super::RAW_TABLE_HEADER_TABLE,
            second.raw_table_name(),
        );
        storage
            .with_inner(|inner| {
                let txn = storage.begin(&inner.db);
                txn.put(&storage_key, &header)?;
                assert_eq!(
                    raw_tables_of(&storage, "notes").len(),
                    1,
                    "the header is not committed yet"
                );
                txn.commit()
            })
            .unwrap();
        assert_eq!(
            raw_tables_of(&storage, "notes").len(),
            2,
            "the header is committed"
        );
    }

    /// A transaction that put a header and was dropped — an error between the put and
    /// the commit — wrote nothing: the table has the raw tables it had, before and after
    /// the next write that does commit.
    #[test]
    fn a_header_put_that_never_commits_is_not_remembered() {
        let (_dir, mut storage) = temp_store();
        put_header(&mut storage, "notes", 0x11);
        assert_eq!(raw_tables_of(&storage, "notes").len(), 1);

        let (second, header) = visible_table("notes", 0x22);
        let storage_key = key_codec::raw_table_entry_key(
            super::super::RAW_TABLE_HEADER_TABLE,
            second.raw_table_name(),
        );
        storage
            .with_inner(|inner| {
                let txn = storage.begin(&inner.db);
                txn.put(&storage_key, &header)?;
                assert_eq!(raw_tables_of(&storage, "notes").len(), 1);
                Ok(())
            })
            .unwrap();
        assert_eq!(
            raw_tables_of(&storage, "notes").len(),
            1,
            "the transaction was dropped"
        );
        storage.raw_table_put("anything", "key", b"value").unwrap();
        assert_eq!(
            raw_tables_of(&storage, "notes").len(),
            1,
            "a later commit does not bring the dropped header in"
        );
        put_header(&mut storage, "notes", 0x22);
        assert_eq!(raw_tables_of(&storage, "notes").len(), 2);
    }

    /// A commit that wrote no header keeps what is remembered: a row put, an index
    /// mutation and a delete later, the raw tables of a table are still answered from
    /// memory.
    #[test]
    fn a_write_that_is_not_a_header_keeps_what_is_remembered() {
        let (_dir, mut storage) = temp_store();
        put_header(&mut storage, "notes", 0x11);
        assert_eq!(raw_tables_of(&storage, "notes").len(), 1);
        let before = storage.prefix_scans.get();

        storage.raw_table_put("anything", "key", b"value").unwrap();
        storage
            .apply_index_mutations(&[IndexMutation::Insert {
                table: "notes",
                column: "title",
                branch: "main",
                value: Value::Text("a".to_string()),
                row_id: ObjectId::new(),
            }])
            .unwrap();
        storage.raw_table_delete("anything", "key").unwrap();
        assert_eq!(raw_tables_of(&storage, "notes").len(), 1);
        assert_eq!(
            storage.prefix_scans.get(),
            before,
            "no header was written: the headers are not read again"
        );
    }

    /// A header write drops what is remembered of every table, not of the one read
    /// next: the other tables' raw tables are read again too.
    #[test]
    fn a_header_write_drops_what_is_remembered_of_every_table() {
        let (_dir, mut storage) = temp_store();
        put_header(&mut storage, "notes", 0x11);
        put_header(&mut storage, "tasks", 0x11);
        assert_eq!(raw_tables_of(&storage, "notes").len(), 1);
        assert_eq!(raw_tables_of(&storage, "tasks").len(), 1);

        put_header(&mut storage, "notes", 0x22);
        assert_eq!(raw_tables_of(&storage, "tasks").len(), 1);
        assert_eq!(
            raw_tables_of(&storage, "notes").len(),
            2,
            "read after another table was"
        );
    }

    /// A closed store answers nothing, and not from what it remembers either.
    #[test]
    fn the_raw_tables_of_a_closed_store_are_not_answered_from_memory() {
        let (_dir, mut storage) = temp_store();
        put_header(&mut storage, "notes", 0x11);
        assert_eq!(raw_tables_of(&storage, "notes").len(), 1);
        storage.close().unwrap();
        assert!(
            super::super::row_raw_table_ids_for_table(&storage, RowRawTableKind::Visible, "notes")
                .is_err(),
            "a read of a closed store"
        );
    }

    /// What one store remembers of its headers is not what another is served, and a
    /// header written to one does not make the other read its own again.
    #[test]
    fn two_stores_do_not_share_what_they_remember_of_their_headers() {
        let (_dir_a, mut store_a) = temp_store();
        let (_dir_b, mut store_b) = temp_store();
        put_header(&mut store_a, "notes", 0x11);
        put_header(&mut store_b, "notes", 0x11);
        put_header(&mut store_b, "notes", 0x22);
        assert_eq!(raw_tables_of(&store_a, "notes").len(), 1);
        assert_eq!(raw_tables_of(&store_b, "notes").len(), 2);
        let before = store_a.prefix_scans.get();
        put_header(&mut store_b, "notes", 0x33);
        assert_eq!(raw_tables_of(&store_a, "notes").len(), 1);
        assert_eq!(raw_tables_of(&store_b, "notes").len(), 3);
        assert_eq!(
            store_a.prefix_scans.get(),
            before,
            "a header written to another store"
        );
    }

    #[test]
    fn open_and_close() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.rocksdb");
        let storage = RocksDBStorage::open(&db_path, 8 * 1024 * 1024).unwrap();
        storage.close().unwrap();
        let reopened = RocksDBStorage::open(&db_path, 8 * 1024 * 1024).unwrap();
        reopened.close().unwrap();
    }

    #[test]
    fn open_rejects_store_manifest_version_mismatch() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.rocksdb");
        let storage = RocksDBStorage::open(&db_path, 8 * 1024 * 1024).unwrap();
        storage.close().unwrap();

        let db = TransactionDB::<rocksdb::SingleThreaded>::open(
            &Options::default(),
            &TransactionDBOptions::default(),
            &db_path,
        )
        .unwrap();
        let bad_manifest = super::super::StoreManifest {
            store_kind: super::super::ROCKSDB_STORE_KIND.to_string(),
            store_format_version: 999,
        };
        let bytes = super::super::encode_store_manifest(&bad_manifest).unwrap();
        db.put(super::super::STORE_MANIFEST_KEY.as_bytes(), bytes)
            .unwrap();
        drop(db);

        let err = match RocksDBStorage::open(&db_path, 8 * 1024 * 1024) {
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
        let temp_dir = tempfile::TempDir::new().unwrap();
        let db_path = temp_dir.path().join("legacy.rocksdb");
        let mut opts = Options::default();
        opts.create_if_missing(true);
        let db = TransactionDB::<rocksdb::SingleThreaded>::open(
            &opts,
            &TransactionDBOptions::default(),
            &db_path,
        )
        .unwrap();
        db.put(b"raw:legacy:alice", b"hello").unwrap();
        drop(db);

        let err = match RocksDBStorage::open(&db_path, 8 * 1024 * 1024) {
            Ok(_) => panic!("expected missing manifest rejection"),
            Err(err) => err,
        };
        assert!(
            err.to_string()
                .contains("missing store manifest for non-empty rocksdb store"),
            "unexpected error: {err}"
        );
    }

    mod rocksdb_conformance {
        use crate::storage::Storage;
        use crate::storage::rocksdb::RocksDBStorage;
        use crate::storage_conformance_tests_persistent;

        storage_conformance_tests_persistent!(
            rocksdb,
            || {
                let dir = tempfile::TempDir::new().unwrap();
                let path = dir.path().join("test.rocksdb");
                let storage = RocksDBStorage::open(&path, 8 * 1024 * 1024).unwrap();
                std::mem::forget(dir);
                Box::new(storage) as Box<dyn Storage>
            },
            |path: &std::path::Path| {
                Box::new(RocksDBStorage::open(path, 8 * 1024 * 1024).unwrap()) as Box<dyn Storage>
            }
        );
    }
}
