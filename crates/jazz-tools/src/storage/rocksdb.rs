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
        append_history_region_row_bytes_core, raw_table_delete_core, raw_table_family_keys_core,
        raw_table_family_last_key_core, raw_table_get_core, raw_table_put_core,
        raw_table_scan_prefix_core, raw_table_scan_prefix_keys_core, raw_table_scan_range_core,
        raw_table_scan_range_keys_core, upsert_visible_region_row_bytes_core,
    },
};
use crate::object::ObjectId;
use crate::row_histories::{HistoryScan, RowState, StoredRowBatch};
use crate::sync_manager::DurabilityTier;

/// A write transaction of one store, and the only way a key of an open store is written
/// (the manifest is put while the store opens): the puts and deletes go through it, so a
/// transaction that wrote a raw-table header moves the store's `header_epoch` here and
/// nowhere else.
struct StoreTxn<'a> {
    txn: Transaction<'a, TransactionDB>,
    header_epoch: &'a std::cell::Cell<u64>,
    wrote_header: std::cell::Cell<bool>,
}

impl StoreTxn<'_> {
    fn put(&self, key: &str, value: &[u8]) -> Result<(), StorageError> {
        self.note_header_write(key);
        self.txn
            .put(key.as_bytes(), value)
            .map_err(|e| StorageError::IoError(format!("rocksdb txn put: {e}")))
    }

    fn delete(&self, key: &str) -> Result<(), StorageError> {
        self.note_header_write(key);
        self.txn
            .delete(key.as_bytes())
            .map_err(|e| StorageError::IoError(format!("rocksdb txn delete: {e}")))
    }

    /// A header written here is readable once this commits, and not before: the epoch
    /// moves here, so that nothing read before — the write or the commit — is served
    /// after.
    fn commit(self) -> Result<(), StorageError> {
        let committed = self
            .txn
            .commit()
            .map_err(|e| StorageError::IoError(format!("rocksdb txn commit: {e}")));
        if self.wrote_header.get() {
            self.header_epoch.set(self.header_epoch.get() + 1);
        }
        committed
    }

    fn note_header_write(&self, storage_key: &str) {
        if key_codec::strip_raw_table_key(super::RAW_TABLE_HEADER_TABLE, storage_key).is_some() {
            self.wrote_header.set(true);
        }
    }
}

struct RocksDBInner {
    db: TransactionDB,
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
    /// Prefix scans issued against the store. A whole-history read is one of
    /// these; the existence probe that replaced it is a seek. A gate can pin
    /// that the probe's cost does not track how deep the history is.
    #[cfg(test)]
    prefix_scans: std::cell::Cell<usize>,
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
            #[cfg(test)]
            prefix_scans: std::cell::Cell::new(0),
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

    fn scan_range_keys_limited_from_db(
        db: &TransactionDB,
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

    fn begin<'a>(&'a self, db: &'a TransactionDB) -> StoreTxn<'a> {
        StoreTxn {
            txn: db.transaction(),
            header_epoch: &self.header_epoch,
            wrote_header: std::cell::Cell::new(false),
        }
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
