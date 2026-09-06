//! RocksDB-backed Storage implementation.
//!
//! Uses `TransactionDB` (pessimistic transactions) for write operations and
//! direct DB access for read-only operations. Follows the same structural
//! pattern as FjallStorage, delegating all logic to `storage_core` callbacks.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::path::Path;

use rocksdb::{
    BlockBasedOptions, Cache, IteratorMode, Options, ReadOptions, Transaction, TransactionDB,
    TransactionDBOptions,
};

use super::{
    HistoryRowBytes, IndexMutation, RawTableMutation, Storage, StorageError, VisibleRowBytes,
    key_codec,
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

struct RocksDBInner {
    db: TransactionDB,
    ensured_raw_table_headers: HashSet<String>,
    visible_row_table_locators: HashMap<(String, ObjectId), super::ExactRowTableLocator>,
}

pub struct RocksDBStorage {
    cache_namespace: usize,
    inner: RefCell<Option<RocksDBInner>>,
    /// Prefix scans issued against the store. A whole-history read is one of
    /// these; the existence probe that replaced it is a seek. A gate can pin
    /// that the probe's cost does not track how deep the history is.
    #[cfg(test)]
    prefix_scans: std::cell::Cell<usize>,
    /// v18 item 5: ladder walks this store served (bumped by `note_visible_locator_recovery`
    /// from the read ladder; unconditional, like `SqliteStorage`'s counters).
    visible_ladder_recoveries: std::sync::atomic::AtomicU64,
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
            #[cfg(test)]
            prefix_scans: std::cell::Cell::new(0),
            visible_ladder_recoveries: std::sync::atomic::AtomicU64::new(0),
            inner: RefCell::new(Some(RocksDBInner {
                db,
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

    /// v18 item 5: ladder walks this store served (per store, unlike the process-global).
    #[cfg(any(test, feature = "test"))]
    pub fn visible_ladder_recoveries_for_test(&self) -> u64 {
        self.visible_ladder_recoveries
            .load(std::sync::atomic::Ordering::Relaxed)
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

    fn get_from_db(db: &TransactionDB, key: &str) -> Result<Option<Vec<u8>>, StorageError> {
        db.get(key.as_bytes())
            .map_err(|e| StorageError::IoError(format!("rocksdb get: {e}")))
    }

    fn scan_prefix_from_db(
        db: &TransactionDB,
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
        db: &TransactionDB,
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

    fn scan_prefix_keys_from_db(
        db: &TransactionDB,
        prefix: &str,
    ) -> Result<Vec<String>, StorageError> {
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
        db: &TransactionDB,
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
        db: &TransactionDB,
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

    // ---- transaction helpers ----

    fn put_on_txn<'a>(
        txn: &Transaction<'a, TransactionDB>,
        key: &str,
        value: &[u8],
    ) -> Result<(), StorageError> {
        txn.put(key.as_bytes(), value)
            .map_err(|e| StorageError::IoError(format!("rocksdb txn put: {e}")))
    }

    fn put_on_txn_cell<'a>(
        txn: &RefCell<Transaction<'a, TransactionDB>>,
        key: &str,
        value: &[u8],
    ) -> Result<(), StorageError> {
        Self::put_on_txn(&txn.borrow(), key, value)
    }

    fn delete_on_txn<'a>(
        txn: &Transaction<'a, TransactionDB>,
        key: &str,
    ) -> Result<(), StorageError> {
        txn.delete(key.as_bytes())
            .map_err(|e| StorageError::IoError(format!("rocksdb txn delete: {e}")))
    }

    fn delete_on_txn_cell<'a>(
        txn: &RefCell<Transaction<'a, TransactionDB>>,
        key: &str,
    ) -> Result<(), StorageError> {
        Self::delete_on_txn(&txn.borrow(), key)
    }

    fn commit_txn(txn: Transaction<'_, TransactionDB>) -> Result<(), StorageError> {
        txn.commit()
            .map_err(|e| StorageError::IoError(format!("rocksdb txn commit: {e}")))
    }

    fn apply_index_mutations_on_txn<'a>(
        txn: &RefCell<Transaction<'a, TransactionDB>>,
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
                        Self::put_on_txn_cell(txn, storage_key, bytes)
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
                        Self::delete_on_txn_cell(txn, storage_key)
                    })?;
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

    fn raw_table_put(&mut self, table: &str, key: &str, value: &[u8]) -> Result<(), StorageError> {
        self.with_inner(|inner| {
            let txn = RefCell::new(inner.db.transaction());
            raw_table_put_core(table, key, value, |storage_key, bytes| {
                Self::put_on_txn_cell(&txn, storage_key, bytes)
            })?;
            Self::commit_txn(txn.into_inner())
        })
    }

    fn raw_table_delete(&mut self, table: &str, key: &str) -> Result<(), StorageError> {
        self.with_inner(|inner| {
            let txn = RefCell::new(inner.db.transaction());
            raw_table_delete_core(table, key, |storage_key| {
                Self::delete_on_txn_cell(&txn, storage_key)
            })?;
            Self::commit_txn(txn.into_inner())
        })
    }

    fn apply_raw_table_mutations(
        &mut self,
        mutations: &[RawTableMutation<'_>],
    ) -> Result<(), StorageError> {
        self.with_inner(|inner| {
            let txn = RefCell::new(inner.db.transaction());
            for mutation in mutations {
                match mutation {
                    RawTableMutation::Put { table, key, value } => {
                        raw_table_put_core(table, key, value, |storage_key, bytes| {
                            Self::put_on_txn_cell(&txn, storage_key, bytes)
                        })?;
                    }
                    RawTableMutation::Delete { table, key } => {
                        raw_table_delete_core(table, key, |storage_key| {
                            Self::delete_on_txn_cell(&txn, storage_key)
                        })?;
                    }
                }
            }
            Self::commit_txn(txn.into_inner())
        })
    }

    fn raw_table_get(&self, table: &str, key: &str) -> Result<Option<Vec<u8>>, StorageError> {
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
            let txn = RefCell::new(inner.db.transaction());
            Self::apply_index_mutations_on_txn(&txn, mutations)?;
            Self::commit_txn(txn.into_inner())
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

    fn append_history_region_row_bytes(
        &mut self,
        table: &str,
        rows: &[HistoryRowBytes<'_>],
    ) -> Result<(), StorageError> {
        self.with_inner(|inner| {
            let txn = RefCell::new(inner.db.transaction());
            append_history_region_row_bytes_core(table, rows, |key, bytes| {
                Self::put_on_txn_cell(&txn, key, bytes)
            })?;
            Self::commit_txn(txn.into_inner())
        })
    }

    fn upsert_visible_region_row_bytes(
        &mut self,
        table: &str,
        rows: &[VisibleRowBytes<'_>],
    ) -> Result<(), StorageError> {
        self.with_inner(|inner| {
            let txn = RefCell::new(inner.db.transaction());
            upsert_visible_region_row_bytes_core(table, rows, |key, bytes| {
                Self::put_on_txn_cell(&txn, key, bytes)
            })?;
            Self::commit_txn(txn.into_inner())
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
            let txn = RefCell::new(inner.db.transaction());
            let key = super::key_codec::visible_row_raw_table_key(branch, row_id);
            for raw_table in &raw_tables {
                raw_table_delete_core(raw_table.as_str(), &key, |storage_key| {
                    Self::delete_on_txn_cell(&txn, storage_key)
                })?;
            }
            raw_table_delete_core(
                super::VISIBLE_ROW_TABLE_LOCATOR_TABLE,
                &super::visible_row_table_locator_key(branch, row_id),
                |storage_key| Self::delete_on_txn_cell(&txn, storage_key),
            )?;
            Self::commit_txn(txn.into_inner())
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
            // v18 item 4 (design v4 § B2(a)): header names and locator pointers written by
            let txn = RefCell::new(inner.db.transaction());
            // this transaction join the caches only after `commit_txn` succeeded — a failed
            // commit must not leave the set claiming a header that was never written.
            let mut ensured: Vec<String> = Vec::new();
            let mut locators: Vec<((String, ObjectId), super::ExactRowTableLocator)> = Vec::new();
            let mut seen_row_raw_tables = std::collections::HashSet::new();
            // diff r21 B2: the header is ENCODED lazily here too. The tree's
            // `seen.insert(..) && ensured.insert(..)` short-circuited before the encode, and
            // the rewrite must not put a `row_raw_table_header` + `encode_raw_table_header` on
            // every write call per raw table — least of all on RocksDB, the store the stand
            // measures.
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
                    |storage_key, bytes| Self::put_on_txn_cell(&txn, storage_key, bytes),
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
            append_history_region_row_bytes_core(table, &borrowed_history_rows, |key, bytes| {
                Self::put_on_txn_cell(&txn, key, bytes)
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
                    |storage_key, bytes| Self::put_on_txn_cell(&txn, storage_key, bytes),
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
                Self::put_on_txn_cell(&txn, key, bytes)
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
                        |storage_key, bytes| Self::put_on_txn_cell(&txn, storage_key, bytes),
                    )?;
                    locators.push((cache_key, locator));
                }
            }
            Self::apply_index_mutations_on_txn(&txn, index_mutations)?;
            Self::commit_txn(txn.into_inner())?;
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

    /// v18 item 5 (D2): persist the exact locator the ladder recovered, in its own
    /// transaction (RocksDB has no pass transaction: every call commits). The header is
    /// ensured the way the write path does it; its name joins the cache after the commit.
    fn record_visible_row_table_locator_recovery(
        &self,
        branch: &str,
        row_id: ObjectId,
        locator: &super::ExactRowTableLocator,
    ) -> Result<(), StorageError> {
        self.with_inner_mut(|inner| {
            let txn = RefCell::new(inner.db.transaction());
            let needs_header = !inner
                .ensured_raw_table_headers
                .contains(super::VISIBLE_ROW_TABLE_LOCATOR_TABLE);
            if needs_header {
                let header = super::encode_raw_table_header(&super::RawTableHeader::system(
                    super::STORAGE_KIND_VISIBLE_ROW_TABLE_LOCATOR,
                    super::EXACT_ROW_TABLE_LOCATOR_STORAGE_FORMAT_V1,
                ))?;
                raw_table_put_core(
                    super::RAW_TABLE_HEADER_TABLE,
                    super::VISIBLE_ROW_TABLE_LOCATOR_TABLE,
                    &header,
                    |storage_key, bytes| Self::put_on_txn_cell(&txn, storage_key, bytes),
                )?;
            }
            let locator_bytes = super::encode_exact_row_table_locator(locator)?;
            raw_table_put_core(
                super::VISIBLE_ROW_TABLE_LOCATOR_TABLE,
                &super::visible_row_table_locator_key(branch, row_id),
                &locator_bytes,
                |storage_key, bytes| Self::put_on_txn_cell(&txn, storage_key, bytes),
            )?;
            Self::commit_txn(txn.into_inner())?;
            // v18 item 5 (diff r22 B2): the same eviction the SQLite hook pays, for the same
            // reason — this is a direct write to the pointer the write path dedups against.
            inner
                .visible_row_table_locators
                .remove(&(branch.to_string(), row_id));
            if needs_header {
                inner
                    .ensured_raw_table_headers
                    .insert(super::VISIBLE_ROW_TABLE_LOCATOR_TABLE.to_string());
            }
            Ok(())
        })
    }

    fn note_visible_locator_recovery(&self) {
        self.visible_ladder_recoveries
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// G-D1r (v18 item 5). Internal on purpose: whether the ladder walked is a count only
    /// the store keeps (`visible_ladder_recoveries_for_test`), and whether the recovered
    /// pointer was persisted is a question to the store's locator table; no client API
    /// exposes either. The RocksDB twin of the SQLite gates — the stand (jazz-sync) runs on
    /// RocksDB, and its "216 walks per pass" is the number this pins at the unit level.
    #[test]
    #[cfg(feature = "test-utils")] // the shared fixture lives in `crate::test_support` (diff r21 SF8)
    fn a_recovered_visible_locator_is_persisted_and_survives_a_reopen() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let db_path = temp_dir.path().join("recover.rocksdb");
        let mut storage = RocksDBStorage::open(&db_path, 8 * 1024 * 1024).unwrap();
        let (branch, row_id, schema_hash) = crate::test_support::poisoned_split_row(&mut storage);
        let honest_family = super::super::visible_row_raw_table_id("users", schema_hash)
            .raw_table_name()
            .to_string();

        let first = storage
            .load_visible_region_row_bytes("users", branch.as_str(), row_id)
            .expect("read succeeds")
            .expect("fixture precondition: the poisoned row must be readable through the ladder");
        assert_eq!(
            storage.visible_ladder_recoveries_for_test(),
            1,
            "fixture precondition: the first read must walk the ladder exactly once"
        );
        let pointer = storage
            .load_visible_row_table_locator(branch.as_str(), row_id)
            .expect("locator readable")
            .expect("the ladder must persist the exact locator it recovered");
        assert_eq!(
            pointer.row_raw_table.to_string(),
            honest_family,
            "the persisted pointer must name the family that holds the bytes"
        );

        storage.close().unwrap();
        let reopened = RocksDBStorage::open(&db_path, 8 * 1024 * 1024).unwrap();
        let again = reopened
            .load_visible_region_row_bytes("users", branch.as_str(), row_id)
            .expect("read succeeds")
            .expect("the row must still be readable after the reopen");
        assert_eq!(
            again, first,
            "the same bytes must be served after the reopen"
        );
        assert_eq!(
            reopened.visible_ladder_recoveries_for_test(),
            0,
            "after a reopen the read must answer from the persisted pointer, not walk the \
             families again"
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
