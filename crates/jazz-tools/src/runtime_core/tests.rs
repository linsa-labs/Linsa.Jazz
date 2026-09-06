use super::*;
use crate::batch_fate::{CapturedFrontierMember, SealedBatchMember, SealedBatchSubmission};
use crate::query_manager::policy::PolicyExpr;
use crate::query_manager::query::QueryBuilder;
use crate::query_manager::session::WriteContext;
use crate::query_manager::types::{
    ColumnType, SchemaBuilder, SchemaHash, TableName, TablePolicies, TableSchema,
};
use crate::row_format::encode_row;
use crate::row_histories::BatchId;
use crate::schema_manager::AppId;
use crate::storage::{
    MemoryStorage, RawTableKeys, RawTableRows, RowLocator, Storage, StorageError,
};
use crate::sync_manager::{
    ClientId, ClientRole, Destination, DurabilityTier, InboxEntry, OutboxEntry, ServerId, Source,
    SyncError, SyncManager, SyncPayload,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

type TestCore = RuntimeCore<MemoryStorage, NoopScheduler>;
type BoxedStorageTestCore = RuntimeCore<Box<dyn Storage>, NoopScheduler>;

fn new_test_core<S: Storage, Sch: Scheduler>(
    schema_manager: SchemaManager,
    storage: S,
    scheduler: Sch,
) -> RuntimeCore<S, Sch> {
    let mut core = RuntimeCore::new(schema_manager, storage, scheduler);
    core.set_sync_sender(Box::new(VecSyncSender::new()));
    core
}

struct RowRegionReadFailingStorage {
    inner: MemoryStorage,
    fail_visible_row_reads: bool,
    fail_row_locator_scans: bool,
    fail_sealed_submission_upserts: Arc<Mutex<bool>>,
    fail_prepared_row_mutations: Arc<Mutex<bool>>,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct LegacyStorageCallCounts;

struct LegacyPersistenceObservingStorage {
    inner: MemoryStorage,
    _calls: Arc<Mutex<LegacyStorageCallCounts>>,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct RowMutationCallCounts {
    row_mutation_calls: usize,
    separate_index_mutation_calls: usize,
    flush_wal_calls: usize,
    local_batch_record_get_calls: usize,
}

/// What one tick's sealed-batch recovery sweep read. Separate from
/// [`RowMutationCallCounts`] because that struct is compared exhaustively by tests about
/// row mutations, which have no business knowing what the sweep costs.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct SweepCallCounts {
    /// Full prefix scans of the sealed-submission table. The sweep does one per tick,
    /// unconditionally, so this answers "did this tick walk the whole table".
    sealed_submission_scans: usize,
    /// Point reads of an authoritative batch fate. The sweep does one PER retained
    /// submission, which is what makes its cost track the table instead of the work.
    authoritative_fate_gets: usize,
    /// Reads of a sealed submission ROW. Every one of these carries a decode, and each
    /// decode resolves a branch name by ord — the expensive half of the sweep.
    submission_row_reads: usize,
    /// Point reads resolving a branch ord back to its name. Only submission decoding does
    /// this on the sweep's path, so it tracks how many rows the sweep decoded.
    branch_name_gets: usize,
}

struct RowMutationObservingStorage {
    inner: MemoryStorage,
    calls: Arc<Mutex<RowMutationCallCounts>>,
    sweep: Arc<Mutex<SweepCallCounts>>,
}

#[derive(Clone, Default)]
struct CountingScheduler {
    schedule_calls: Arc<Mutex<usize>>,
}

impl RowRegionReadFailingStorage {
    fn new() -> Self {
        Self {
            inner: MemoryStorage::new(),
            fail_visible_row_reads: true,
            fail_row_locator_scans: false,
            fail_sealed_submission_upserts: Arc::new(Mutex::new(false)),
            fail_prepared_row_mutations: Arc::new(Mutex::new(false)),
        }
    }

    fn with_row_locator_scan_failure() -> Self {
        Self {
            inner: MemoryStorage::new(),
            fail_visible_row_reads: false,
            fail_row_locator_scans: true,
            fail_sealed_submission_upserts: Arc::new(Mutex::new(false)),
            fail_prepared_row_mutations: Arc::new(Mutex::new(false)),
        }
    }

    fn with_sealed_submission_upsert_failure(
        fail_sealed_submission_upserts: Arc<Mutex<bool>>,
    ) -> Self {
        Self {
            inner: MemoryStorage::new(),
            fail_visible_row_reads: false,
            fail_row_locator_scans: false,
            fail_sealed_submission_upserts,
            fail_prepared_row_mutations: Arc::new(Mutex::new(false)),
        }
    }

    fn with_prepared_row_mutation_failure(fail_prepared_row_mutations: Arc<Mutex<bool>>) -> Self {
        Self {
            inner: MemoryStorage::new(),
            fail_visible_row_reads: false,
            fail_row_locator_scans: false,
            fail_sealed_submission_upserts: Arc::new(Mutex::new(false)),
            fail_prepared_row_mutations,
        }
    }
}

impl LegacyPersistenceObservingStorage {
    fn new(calls: Arc<Mutex<LegacyStorageCallCounts>>) -> Self {
        Self {
            inner: MemoryStorage::new(),
            _calls: calls,
        }
    }
}

impl RowMutationObservingStorage {
    fn new(calls: Arc<Mutex<RowMutationCallCounts>>) -> Self {
        Self {
            inner: MemoryStorage::new(),
            calls,
            sweep: Arc::new(Mutex::new(SweepCallCounts::default())),
        }
    }

    fn observing_sweep(sweep: Arc<Mutex<SweepCallCounts>>) -> Self {
        Self {
            inner: MemoryStorage::new(),
            calls: Arc::new(Mutex::new(RowMutationCallCounts::default())),
            sweep,
        }
    }
}

impl CountingScheduler {
    fn schedule_count(&self) -> usize {
        *self.schedule_calls.lock().unwrap()
    }
}

impl Scheduler for CountingScheduler {
    fn schedule_batched_tick(&self) {
        *self.schedule_calls.lock().unwrap() += 1;
    }
}

impl Storage for RowRegionReadFailingStorage {
    fn apply_encoded_row_mutation(
        &mut self,
        table: &str,
        history_rows: &[crate::storage::OwnedHistoryRowBytes],
        visible_rows: &[crate::storage::OwnedVisibleRowBytes],
        index_mutations: &[crate::storage::IndexMutation<'_>],
    ) -> Result<(), StorageError> {
        self.inner
            .apply_encoded_row_mutation(table, history_rows, visible_rows, index_mutations)
    }

    fn apply_prepared_row_mutation(
        &mut self,
        table: &str,
        history_rows: &[crate::row_histories::StoredRowBatch],
        visible_entries: &[crate::row_histories::VisibleRowEntry],
        encoded_history_rows: &[crate::storage::OwnedHistoryRowBytes],
        encoded_visible_rows: &[crate::storage::OwnedVisibleRowBytes],
        index_mutations: &[crate::storage::IndexMutation<'_>],
    ) -> Result<(), StorageError> {
        if *self.fail_prepared_row_mutations.lock().unwrap() {
            return Err(StorageError::IoError(
                "prepared row mutations deliberately disabled in this test".to_string(),
            ));
        }
        self.inner.apply_prepared_row_mutation(
            table,
            history_rows,
            visible_entries,
            encoded_history_rows,
            encoded_visible_rows,
            index_mutations,
        )
    }

    fn scan_row_locators(&self) -> Result<crate::storage::RowLocatorRows, StorageError> {
        if self.fail_row_locator_scans {
            return Err(StorageError::IoError(
                "row-locator scans deliberately disabled in this test".to_string(),
            ));
        }
        self.inner.scan_row_locators()
    }

    fn load_row_locator(
        &self,
        id: ObjectId,
    ) -> Result<Option<crate::storage::RowLocator>, StorageError> {
        self.inner.load_row_locator(id)
    }

    fn put_row_locator(
        &mut self,
        id: ObjectId,
        locator: Option<&crate::storage::RowLocator>,
    ) -> Result<(), StorageError> {
        self.inner.put_row_locator(id, locator)
    }

    fn upsert_sealed_batch_submission(
        &mut self,
        submission: &SealedBatchSubmission,
    ) -> Result<(), StorageError> {
        if *self.fail_sealed_submission_upserts.lock().unwrap() {
            return Err(StorageError::IoError(
                "sealed submission upserts deliberately disabled in this test".to_string(),
            ));
        }
        self.inner.upsert_sealed_batch_submission(submission)
    }

    fn raw_table_put(&mut self, table: &str, key: &str, value: &[u8]) -> Result<(), StorageError> {
        self.inner.raw_table_put(table, key, value)
    }

    fn raw_table_delete(&mut self, table: &str, key: &str) -> Result<(), StorageError> {
        self.inner.raw_table_delete(table, key)
    }

    fn raw_table_get(&self, table: &str, key: &str) -> Result<Option<Vec<u8>>, StorageError> {
        self.inner.raw_table_get(table, key)
    }

    fn raw_table_scan_prefix(
        &self,
        table: &str,
        prefix: &str,
    ) -> Result<RawTableRows, StorageError> {
        self.inner.raw_table_scan_prefix(table, prefix)
    }

    fn raw_table_scan_prefix_keys(
        &self,
        table: &str,
        prefix: &str,
    ) -> Result<RawTableKeys, StorageError> {
        self.inner.raw_table_scan_prefix_keys(table, prefix)
    }

    fn raw_table_scan_range(
        &self,
        table: &str,
        start: Option<&str>,
        end: Option<&str>,
    ) -> Result<RawTableRows, StorageError> {
        self.inner.raw_table_scan_range(table, start, end)
    }

    fn raw_table_scan_range_keys(
        &self,
        table: &str,
        start: Option<&str>,
        end: Option<&str>,
    ) -> Result<RawTableKeys, StorageError> {
        self.inner.raw_table_scan_range_keys(table, start, end)
    }

    fn append_history_region_rows(
        &mut self,
        table: &str,
        rows: &[crate::row_histories::StoredRowBatch],
    ) -> Result<(), StorageError> {
        self.inner.append_history_region_rows(table, rows)
    }

    fn append_history_region_row_bytes(
        &mut self,
        table: &str,
        rows: &[crate::storage::HistoryRowBytes<'_>],
    ) -> Result<(), StorageError> {
        self.inner.append_history_region_row_bytes(table, rows)
    }

    fn upsert_visible_region_rows(
        &mut self,
        table: &str,
        entries: &[crate::row_histories::VisibleRowEntry],
    ) -> Result<(), StorageError> {
        self.inner.upsert_visible_region_rows(table, entries)
    }

    fn delete_visible_region_row(
        &mut self,
        table: &str,
        branch: &str,
        row_id: ObjectId,
    ) -> Result<(), StorageError> {
        self.inner.delete_visible_region_row(table, branch, row_id)
    }

    fn patch_row_region_rows_by_batch(
        &mut self,
        table: &str,
        batch_id: crate::row_histories::BatchId,
        state: Option<crate::row_histories::RowState>,
        confirmed_tier: Option<DurabilityTier>,
    ) -> Result<(), StorageError> {
        self.inner
            .patch_row_region_rows_by_batch(table, batch_id, state, confirmed_tier)
    }

    fn patch_exact_row_batch(
        &mut self,
        table: &str,
        branch: &str,
        row_id: ObjectId,
        batch_id: crate::row_histories::BatchId,
        state: Option<crate::row_histories::RowState>,
        confirmed_tier: Option<DurabilityTier>,
    ) -> Result<bool, StorageError> {
        self.inner
            .patch_exact_row_batch(table, branch, row_id, batch_id, state, confirmed_tier)
    }

    fn patch_exact_row_batch_for_schema_hash(
        &mut self,
        table: &str,
        schema_hash: crate::query_manager::types::SchemaHash,
        branch: &str,
        row_id: ObjectId,
        batch_id: crate::row_histories::BatchId,
        state: Option<crate::row_histories::RowState>,
        confirmed_tier: Option<DurabilityTier>,
    ) -> Result<bool, StorageError> {
        self.inner.patch_exact_row_batch_for_schema_hash(
            table,
            schema_hash,
            branch,
            row_id,
            batch_id,
            state,
            confirmed_tier,
        )
    }

    fn scan_visible_region(
        &self,
        table: &str,
        branch: &str,
    ) -> Result<Vec<crate::row_histories::StoredRowBatch>, StorageError> {
        self.inner.scan_visible_region(table, branch)
    }

    fn load_visible_region_row(
        &self,
        table: &str,
        branch: &str,
        row_id: ObjectId,
    ) -> Result<Option<crate::row_histories::StoredRowBatch>, StorageError> {
        if self.fail_visible_row_reads {
            return Err(StorageError::IoError(
                "row-history reads deliberately disabled in this test".to_string(),
            ));
        }
        self.inner.load_visible_region_row(table, branch, row_id)
    }

    fn load_visible_region_frontier(
        &self,
        table: &str,
        branch: &str,
        row_id: ObjectId,
    ) -> Result<Option<Vec<crate::row_histories::BatchId>>, StorageError> {
        self.inner
            .load_visible_region_frontier(table, branch, row_id)
    }

    fn capture_family_visible_frontier(
        &self,
        target_branch_name: crate::object::BranchName,
    ) -> Result<Vec<crate::batch_fate::CapturedFrontierMember>, StorageError> {
        self.inner
            .capture_family_visible_frontier(target_branch_name)
    }

    fn scan_visible_region_row_batches(
        &self,
        table: &str,
        row_id: ObjectId,
    ) -> Result<Vec<crate::row_histories::StoredRowBatch>, StorageError> {
        self.inner.scan_visible_region_row_batches(table, row_id)
    }

    fn scan_history_row_batches(
        &self,
        table: &str,
        row_id: ObjectId,
    ) -> Result<Vec<crate::row_histories::StoredRowBatch>, StorageError> {
        self.inner.scan_history_row_batches(table, row_id)
    }

    fn load_history_row_batch(
        &self,
        table: &str,
        branch: &str,
        row_id: ObjectId,
        batch_id: crate::row_histories::BatchId,
    ) -> Result<Option<crate::row_histories::StoredRowBatch>, StorageError> {
        self.inner
            .load_history_row_batch(table, branch, row_id, batch_id)
    }

    fn load_history_query_row_batch(
        &self,
        table: &str,
        branch: &str,
        row_id: ObjectId,
        batch_id: crate::row_histories::BatchId,
    ) -> Result<Option<crate::row_histories::QueryRowBatch>, StorageError> {
        self.inner
            .load_history_query_row_batch(table, branch, row_id, batch_id)
    }

    fn load_history_row_batch_for_schema_hash(
        &self,
        table: &str,
        schema_hash: crate::query_manager::types::SchemaHash,
        branch: &str,
        row_id: ObjectId,
        batch_id: crate::row_histories::BatchId,
    ) -> Result<Option<crate::row_histories::StoredRowBatch>, StorageError> {
        self.inner.load_history_row_batch_for_schema_hash(
            table,
            schema_hash,
            branch,
            row_id,
            batch_id,
        )
    }

    fn load_history_row_batch_any_branch(
        &self,
        table: &str,
        row_id: ObjectId,
        batch_id: crate::row_histories::BatchId,
    ) -> Result<Option<crate::row_histories::StoredRowBatch>, StorageError> {
        self.inner
            .load_history_row_batch_any_branch(table, row_id, batch_id)
    }

    fn load_history_query_row_batch_any_branch(
        &self,
        table: &str,
        row_id: ObjectId,
        batch_id: crate::row_histories::BatchId,
    ) -> Result<Option<crate::row_histories::QueryRowBatch>, StorageError> {
        self.inner
            .load_history_query_row_batch_any_branch(table, row_id, batch_id)
    }

    fn row_batch_exists(
        &self,
        table: &str,
        branch: &str,
        row_id: ObjectId,
        batch_id: crate::row_histories::BatchId,
    ) -> Result<bool, StorageError> {
        self.inner.row_batch_exists(table, branch, row_id, batch_id)
    }

    fn scan_row_branch_tip_ids(
        &self,
        table: &str,
        branch: &str,
        row_id: ObjectId,
    ) -> Result<Vec<crate::row_histories::BatchId>, StorageError> {
        self.inner.scan_row_branch_tip_ids(table, branch, row_id)
    }

    fn load_history_row_batch_bytes(
        &self,
        table: &str,
        branch: &str,
        row_id: ObjectId,
        batch_id: crate::row_histories::BatchId,
    ) -> Result<Option<Vec<u8>>, StorageError> {
        self.inner
            .load_history_row_batch_bytes(table, branch, row_id, batch_id)
    }

    fn scan_history_region_bytes(
        &self,
        table: &str,
        scan: crate::row_histories::HistoryScan,
    ) -> Result<Vec<Vec<u8>>, StorageError> {
        self.inner.scan_history_region_bytes(table, scan)
    }

    fn scan_history_region(
        &self,
        table: &str,
        branch: &str,
        scan: crate::row_histories::HistoryScan,
    ) -> Result<Vec<crate::row_histories::StoredRowBatch>, StorageError> {
        self.inner.scan_history_region(table, branch, scan)
    }

    fn index_insert(
        &mut self,
        table: &str,
        column: &str,
        branch: &str,
        value: &Value,
        row_id: ObjectId,
    ) -> Result<(), StorageError> {
        self.inner
            .index_insert(table, column, branch, value, row_id)
    }

    fn index_remove(
        &mut self,
        table: &str,
        column: &str,
        branch: &str,
        value: &Value,
        row_id: ObjectId,
    ) -> Result<(), StorageError> {
        self.inner
            .index_remove(table, column, branch, value, row_id)
    }

    fn index_lookup(
        &self,
        table: &str,
        column: &str,
        branch: &str,
        value: &Value,
    ) -> Vec<ObjectId> {
        self.inner.index_lookup(table, column, branch, value)
    }

    fn index_range(
        &self,
        table: &str,
        column: &str,
        branch: &str,
        start: std::ops::Bound<&Value>,
        end: std::ops::Bound<&Value>,
    ) -> Vec<ObjectId> {
        self.inner.index_range(table, column, branch, start, end)
    }

    fn index_scan_all(&self, table: &str, column: &str, branch: &str) -> Vec<ObjectId> {
        self.inner.index_scan_all(table, column, branch)
    }

    fn flush(&self) -> Result<(), StorageError> {
        self.inner.flush()
    }

    fn flush_wal(&self) -> Result<(), StorageError> {
        self.inner.flush_wal()
    }

    fn close(&self) -> Result<(), StorageError> {
        self.inner.close()
    }
}

impl Storage for LegacyPersistenceObservingStorage {
    fn apply_encoded_row_mutation(
        &mut self,
        table: &str,
        history_rows: &[crate::storage::OwnedHistoryRowBytes],
        visible_rows: &[crate::storage::OwnedVisibleRowBytes],
        index_mutations: &[crate::storage::IndexMutation<'_>],
    ) -> Result<(), StorageError> {
        self.inner
            .apply_encoded_row_mutation(table, history_rows, visible_rows, index_mutations)
    }

    fn apply_prepared_row_mutation(
        &mut self,
        table: &str,
        history_rows: &[crate::row_histories::StoredRowBatch],
        visible_entries: &[crate::row_histories::VisibleRowEntry],
        encoded_history_rows: &[crate::storage::OwnedHistoryRowBytes],
        encoded_visible_rows: &[crate::storage::OwnedVisibleRowBytes],
        index_mutations: &[crate::storage::IndexMutation<'_>],
    ) -> Result<(), StorageError> {
        self.inner.apply_prepared_row_mutation(
            table,
            history_rows,
            visible_entries,
            encoded_history_rows,
            encoded_visible_rows,
            index_mutations,
        )
    }

    fn scan_row_locators(&self) -> Result<crate::storage::RowLocatorRows, StorageError> {
        self.inner.scan_row_locators()
    }

    fn load_row_locator(
        &self,
        id: ObjectId,
    ) -> Result<Option<crate::storage::RowLocator>, StorageError> {
        self.inner.load_row_locator(id)
    }

    fn put_row_locator(
        &mut self,
        id: ObjectId,
        locator: Option<&crate::storage::RowLocator>,
    ) -> Result<(), StorageError> {
        self.inner.put_row_locator(id, locator)
    }

    fn raw_table_put(&mut self, table: &str, key: &str, value: &[u8]) -> Result<(), StorageError> {
        self.inner.raw_table_put(table, key, value)
    }

    fn raw_table_delete(&mut self, table: &str, key: &str) -> Result<(), StorageError> {
        self.inner.raw_table_delete(table, key)
    }

    fn raw_table_get(&self, table: &str, key: &str) -> Result<Option<Vec<u8>>, StorageError> {
        self.inner.raw_table_get(table, key)
    }

    fn raw_table_scan_prefix(
        &self,
        table: &str,
        prefix: &str,
    ) -> Result<RawTableRows, StorageError> {
        self.inner.raw_table_scan_prefix(table, prefix)
    }

    fn raw_table_scan_prefix_keys(
        &self,
        table: &str,
        prefix: &str,
    ) -> Result<RawTableKeys, StorageError> {
        self.inner.raw_table_scan_prefix_keys(table, prefix)
    }

    fn raw_table_scan_range(
        &self,
        table: &str,
        start: Option<&str>,
        end: Option<&str>,
    ) -> Result<RawTableRows, StorageError> {
        self.inner.raw_table_scan_range(table, start, end)
    }

    fn raw_table_scan_range_keys(
        &self,
        table: &str,
        start: Option<&str>,
        end: Option<&str>,
    ) -> Result<RawTableKeys, StorageError> {
        self.inner.raw_table_scan_range_keys(table, start, end)
    }

    fn append_history_region_rows(
        &mut self,
        table: &str,
        rows: &[crate::row_histories::StoredRowBatch],
    ) -> Result<(), StorageError> {
        self.inner.append_history_region_rows(table, rows)
    }

    fn append_history_region_row_bytes(
        &mut self,
        table: &str,
        rows: &[crate::storage::HistoryRowBytes<'_>],
    ) -> Result<(), StorageError> {
        self.inner.append_history_region_row_bytes(table, rows)
    }

    fn upsert_visible_region_rows(
        &mut self,
        table: &str,
        entries: &[crate::row_histories::VisibleRowEntry],
    ) -> Result<(), StorageError> {
        self.inner.upsert_visible_region_rows(table, entries)
    }

    fn delete_visible_region_row(
        &mut self,
        table: &str,
        branch: &str,
        row_id: ObjectId,
    ) -> Result<(), StorageError> {
        self.inner.delete_visible_region_row(table, branch, row_id)
    }

    fn patch_row_region_rows_by_batch(
        &mut self,
        table: &str,
        batch_id: crate::row_histories::BatchId,
        state: Option<crate::row_histories::RowState>,
        confirmed_tier: Option<DurabilityTier>,
    ) -> Result<(), StorageError> {
        self.inner
            .patch_row_region_rows_by_batch(table, batch_id, state, confirmed_tier)
    }

    fn patch_exact_row_batch(
        &mut self,
        table: &str,
        branch: &str,
        row_id: ObjectId,
        batch_id: crate::row_histories::BatchId,
        state: Option<crate::row_histories::RowState>,
        confirmed_tier: Option<DurabilityTier>,
    ) -> Result<bool, StorageError> {
        self.inner
            .patch_exact_row_batch(table, branch, row_id, batch_id, state, confirmed_tier)
    }

    fn patch_exact_row_batch_for_schema_hash(
        &mut self,
        table: &str,
        schema_hash: crate::query_manager::types::SchemaHash,
        branch: &str,
        row_id: ObjectId,
        batch_id: crate::row_histories::BatchId,
        state: Option<crate::row_histories::RowState>,
        confirmed_tier: Option<DurabilityTier>,
    ) -> Result<bool, StorageError> {
        self.inner.patch_exact_row_batch_for_schema_hash(
            table,
            schema_hash,
            branch,
            row_id,
            batch_id,
            state,
            confirmed_tier,
        )
    }

    fn scan_visible_region(
        &self,
        table: &str,
        branch: &str,
    ) -> Result<Vec<crate::row_histories::StoredRowBatch>, StorageError> {
        self.inner.scan_visible_region(table, branch)
    }

    fn load_visible_region_row(
        &self,
        table: &str,
        branch: &str,
        row_id: ObjectId,
    ) -> Result<Option<crate::row_histories::StoredRowBatch>, StorageError> {
        self.inner.load_visible_region_row(table, branch, row_id)
    }

    fn load_visible_region_frontier(
        &self,
        table: &str,
        branch: &str,
        row_id: ObjectId,
    ) -> Result<Option<Vec<crate::row_histories::BatchId>>, StorageError> {
        self.inner
            .load_visible_region_frontier(table, branch, row_id)
    }

    fn capture_family_visible_frontier(
        &self,
        target_branch_name: crate::object::BranchName,
    ) -> Result<Vec<crate::batch_fate::CapturedFrontierMember>, StorageError> {
        self.inner
            .capture_family_visible_frontier(target_branch_name)
    }

    fn scan_visible_region_row_batches(
        &self,
        table: &str,
        row_id: ObjectId,
    ) -> Result<Vec<crate::row_histories::StoredRowBatch>, StorageError> {
        self.inner.scan_visible_region_row_batches(table, row_id)
    }

    fn scan_history_row_batches(
        &self,
        table: &str,
        row_id: ObjectId,
    ) -> Result<Vec<crate::row_histories::StoredRowBatch>, StorageError> {
        self.inner.scan_history_row_batches(table, row_id)
    }

    fn load_history_row_batch(
        &self,
        table: &str,
        branch: &str,
        row_id: ObjectId,
        batch_id: crate::row_histories::BatchId,
    ) -> Result<Option<crate::row_histories::StoredRowBatch>, StorageError> {
        self.inner
            .load_history_row_batch(table, branch, row_id, batch_id)
    }

    fn load_history_query_row_batch(
        &self,
        table: &str,
        branch: &str,
        row_id: ObjectId,
        batch_id: crate::row_histories::BatchId,
    ) -> Result<Option<crate::row_histories::QueryRowBatch>, StorageError> {
        self.inner
            .load_history_query_row_batch(table, branch, row_id, batch_id)
    }

    fn load_history_row_batch_for_schema_hash(
        &self,
        table: &str,
        schema_hash: crate::query_manager::types::SchemaHash,
        branch: &str,
        row_id: ObjectId,
        batch_id: crate::row_histories::BatchId,
    ) -> Result<Option<crate::row_histories::StoredRowBatch>, StorageError> {
        self.inner.load_history_row_batch_for_schema_hash(
            table,
            schema_hash,
            branch,
            row_id,
            batch_id,
        )
    }

    fn load_history_row_batch_any_branch(
        &self,
        table: &str,
        row_id: ObjectId,
        batch_id: crate::row_histories::BatchId,
    ) -> Result<Option<crate::row_histories::StoredRowBatch>, StorageError> {
        self.inner
            .load_history_row_batch_any_branch(table, row_id, batch_id)
    }

    fn load_history_query_row_batch_any_branch(
        &self,
        table: &str,
        row_id: ObjectId,
        batch_id: crate::row_histories::BatchId,
    ) -> Result<Option<crate::row_histories::QueryRowBatch>, StorageError> {
        self.inner
            .load_history_query_row_batch_any_branch(table, row_id, batch_id)
    }

    fn row_batch_exists(
        &self,
        table: &str,
        branch: &str,
        row_id: ObjectId,
        batch_id: crate::row_histories::BatchId,
    ) -> Result<bool, StorageError> {
        self.inner.row_batch_exists(table, branch, row_id, batch_id)
    }

    fn scan_row_branch_tip_ids(
        &self,
        table: &str,
        branch: &str,
        row_id: ObjectId,
    ) -> Result<Vec<crate::row_histories::BatchId>, StorageError> {
        self.inner.scan_row_branch_tip_ids(table, branch, row_id)
    }

    fn load_history_row_batch_bytes(
        &self,
        table: &str,
        branch: &str,
        row_id: ObjectId,
        batch_id: crate::row_histories::BatchId,
    ) -> Result<Option<Vec<u8>>, StorageError> {
        self.inner
            .load_history_row_batch_bytes(table, branch, row_id, batch_id)
    }

    fn scan_history_region_bytes(
        &self,
        table: &str,
        scan: crate::row_histories::HistoryScan,
    ) -> Result<Vec<Vec<u8>>, StorageError> {
        self.inner.scan_history_region_bytes(table, scan)
    }

    fn scan_history_region(
        &self,
        table: &str,
        branch: &str,
        scan: crate::row_histories::HistoryScan,
    ) -> Result<Vec<crate::row_histories::StoredRowBatch>, StorageError> {
        self.inner.scan_history_region(table, branch, scan)
    }

    fn index_insert(
        &mut self,
        table: &str,
        column: &str,
        branch: &str,
        value: &Value,
        row_id: ObjectId,
    ) -> Result<(), StorageError> {
        self.inner
            .index_insert(table, column, branch, value, row_id)
    }

    fn index_remove(
        &mut self,
        table: &str,
        column: &str,
        branch: &str,
        value: &Value,
        row_id: ObjectId,
    ) -> Result<(), StorageError> {
        self.inner
            .index_remove(table, column, branch, value, row_id)
    }

    fn index_lookup(
        &self,
        table: &str,
        column: &str,
        branch: &str,
        value: &Value,
    ) -> Vec<ObjectId> {
        self.inner.index_lookup(table, column, branch, value)
    }

    fn index_range(
        &self,
        table: &str,
        column: &str,
        branch: &str,
        start: std::ops::Bound<&Value>,
        end: std::ops::Bound<&Value>,
    ) -> Vec<ObjectId> {
        self.inner.index_range(table, column, branch, start, end)
    }

    fn index_scan_all(&self, table: &str, column: &str, branch: &str) -> Vec<ObjectId> {
        self.inner.index_scan_all(table, column, branch)
    }

    fn flush(&self) -> Result<(), StorageError> {
        self.inner.flush()
    }

    fn flush_wal(&self) -> Result<(), StorageError> {
        self.inner.flush_wal()
    }

    fn close(&self) -> Result<(), StorageError> {
        self.inner.close()
    }
}

impl Storage for RowMutationObservingStorage {
    fn apply_encoded_row_mutation(
        &mut self,
        table: &str,
        history_rows: &[crate::storage::OwnedHistoryRowBytes],
        visible_rows: &[crate::storage::OwnedVisibleRowBytes],
        index_mutations: &[crate::storage::IndexMutation<'_>],
    ) -> Result<(), StorageError> {
        self.calls.lock().unwrap().row_mutation_calls += 1;
        self.inner
            .apply_encoded_row_mutation(table, history_rows, visible_rows, index_mutations)
    }

    fn apply_prepared_row_mutation(
        &mut self,
        table: &str,
        history_rows: &[crate::row_histories::StoredRowBatch],
        visible_entries: &[crate::row_histories::VisibleRowEntry],
        encoded_history_rows: &[crate::storage::OwnedHistoryRowBytes],
        encoded_visible_rows: &[crate::storage::OwnedVisibleRowBytes],
        index_mutations: &[crate::storage::IndexMutation<'_>],
    ) -> Result<(), StorageError> {
        self.calls.lock().unwrap().row_mutation_calls += 1;
        self.inner.apply_prepared_row_mutation(
            table,
            history_rows,
            visible_entries,
            encoded_history_rows,
            encoded_visible_rows,
            index_mutations,
        )
    }

    fn scan_row_locators(&self) -> Result<crate::storage::RowLocatorRows, StorageError> {
        self.inner.scan_row_locators()
    }

    fn load_row_locator(
        &self,
        id: ObjectId,
    ) -> Result<Option<crate::storage::RowLocator>, StorageError> {
        self.inner.load_row_locator(id)
    }

    fn put_row_locator(
        &mut self,
        id: ObjectId,
        locator: Option<&crate::storage::RowLocator>,
    ) -> Result<(), StorageError> {
        self.inner.put_row_locator(id, locator)
    }

    fn raw_table_put(&mut self, table: &str, key: &str, value: &[u8]) -> Result<(), StorageError> {
        self.inner.raw_table_put(table, key, value)
    }

    fn raw_table_delete(&mut self, table: &str, key: &str) -> Result<(), StorageError> {
        self.inner.raw_table_delete(table, key)
    }

    fn raw_table_get(&self, table: &str, key: &str) -> Result<Option<Vec<u8>>, StorageError> {
        if table == "__local_batch_record" && key.starts_with("batch:") {
            self.calls.lock().unwrap().local_batch_record_get_calls += 1;
        }
        // This wrapper deliberately does NOT override `load_authoritative_batch_fate` or
        // `scan_sealed_batch_submissions`: their default trait bodies run here and route
        // through the raw table, which is what makes the count observable — and it also
        // bypasses `MemoryStorage`'s own fate map, so for these two reads the wrapper
        // behaves like a real backend rather than like memory.
        if table == "__authoritative_batch_settlement" && key.starts_with("batch:") {
            self.sweep.lock().unwrap().authoritative_fate_gets += 1;
        }
        if table == "__sealed_batch_submission" && key.starts_with("batch:") {
            self.sweep.lock().unwrap().submission_row_reads += 1;
        }
        if table == "__branch_name_by_ord" {
            self.sweep.lock().unwrap().branch_name_gets += 1;
        }
        self.inner.raw_table_get(table, key)
    }

    fn raw_table_scan_prefix(
        &self,
        table: &str,
        prefix: &str,
    ) -> Result<RawTableRows, StorageError> {
        if table == "__sealed_batch_submission" {
            self.sweep.lock().unwrap().sealed_submission_scans += 1;
        }
        self.inner.raw_table_scan_prefix(table, prefix)
    }

    fn raw_table_scan_prefix_keys(
        &self,
        table: &str,
        prefix: &str,
    ) -> Result<RawTableKeys, StorageError> {
        self.inner.raw_table_scan_prefix_keys(table, prefix)
    }

    fn raw_table_scan_range(
        &self,
        table: &str,
        start: Option<&str>,
        end: Option<&str>,
    ) -> Result<RawTableRows, StorageError> {
        self.inner.raw_table_scan_range(table, start, end)
    }

    fn raw_table_scan_range_keys(
        &self,
        table: &str,
        start: Option<&str>,
        end: Option<&str>,
    ) -> Result<RawTableKeys, StorageError> {
        self.inner.raw_table_scan_range_keys(table, start, end)
    }

    fn append_history_region_rows(
        &mut self,
        table: &str,
        rows: &[crate::row_histories::StoredRowBatch],
    ) -> Result<(), StorageError> {
        self.inner.append_history_region_rows(table, rows)
    }

    fn append_history_region_row_bytes(
        &mut self,
        table: &str,
        rows: &[crate::storage::HistoryRowBytes<'_>],
    ) -> Result<(), StorageError> {
        self.inner.append_history_region_row_bytes(table, rows)
    }

    fn upsert_visible_region_rows(
        &mut self,
        table: &str,
        entries: &[crate::row_histories::VisibleRowEntry],
    ) -> Result<(), StorageError> {
        self.inner.upsert_visible_region_rows(table, entries)
    }

    fn delete_visible_region_row(
        &mut self,
        table: &str,
        branch: &str,
        row_id: ObjectId,
    ) -> Result<(), StorageError> {
        self.inner.delete_visible_region_row(table, branch, row_id)
    }

    fn apply_row_mutation(
        &mut self,
        table: &str,
        history_rows: &[crate::row_histories::StoredRowBatch],
        visible_entries: &[crate::row_histories::VisibleRowEntry],
        index_mutations: &[crate::storage::IndexMutation<'_>],
    ) -> Result<(), StorageError> {
        self.calls.lock().unwrap().row_mutation_calls += 1;
        self.inner
            .apply_row_mutation(table, history_rows, visible_entries, index_mutations)
    }

    fn patch_row_region_rows_by_batch(
        &mut self,
        table: &str,
        batch_id: crate::row_histories::BatchId,
        state: Option<crate::row_histories::RowState>,
        confirmed_tier: Option<DurabilityTier>,
    ) -> Result<(), StorageError> {
        self.inner
            .patch_row_region_rows_by_batch(table, batch_id, state, confirmed_tier)
    }

    fn patch_exact_row_batch(
        &mut self,
        table: &str,
        branch: &str,
        row_id: ObjectId,
        batch_id: crate::row_histories::BatchId,
        state: Option<crate::row_histories::RowState>,
        confirmed_tier: Option<DurabilityTier>,
    ) -> Result<bool, StorageError> {
        self.inner
            .patch_exact_row_batch(table, branch, row_id, batch_id, state, confirmed_tier)
    }

    fn patch_exact_row_batch_for_schema_hash(
        &mut self,
        table: &str,
        schema_hash: crate::query_manager::types::SchemaHash,
        branch: &str,
        row_id: ObjectId,
        batch_id: crate::row_histories::BatchId,
        state: Option<crate::row_histories::RowState>,
        confirmed_tier: Option<DurabilityTier>,
    ) -> Result<bool, StorageError> {
        self.inner.patch_exact_row_batch_for_schema_hash(
            table,
            schema_hash,
            branch,
            row_id,
            batch_id,
            state,
            confirmed_tier,
        )
    }

    fn scan_visible_region(
        &self,
        table: &str,
        branch: &str,
    ) -> Result<Vec<crate::row_histories::StoredRowBatch>, StorageError> {
        self.inner.scan_visible_region(table, branch)
    }

    fn load_visible_region_row(
        &self,
        table: &str,
        branch: &str,
        row_id: ObjectId,
    ) -> Result<Option<crate::row_histories::StoredRowBatch>, StorageError> {
        self.inner.load_visible_region_row(table, branch, row_id)
    }

    fn load_visible_region_frontier(
        &self,
        table: &str,
        branch: &str,
        row_id: ObjectId,
    ) -> Result<Option<Vec<crate::row_histories::BatchId>>, StorageError> {
        self.inner
            .load_visible_region_frontier(table, branch, row_id)
    }

    fn capture_family_visible_frontier(
        &self,
        target_branch_name: crate::object::BranchName,
    ) -> Result<Vec<crate::batch_fate::CapturedFrontierMember>, StorageError> {
        self.inner
            .capture_family_visible_frontier(target_branch_name)
    }

    fn scan_visible_region_row_batches(
        &self,
        table: &str,
        row_id: ObjectId,
    ) -> Result<Vec<crate::row_histories::StoredRowBatch>, StorageError> {
        self.inner.scan_visible_region_row_batches(table, row_id)
    }

    fn scan_history_row_batches(
        &self,
        table: &str,
        row_id: ObjectId,
    ) -> Result<Vec<crate::row_histories::StoredRowBatch>, StorageError> {
        self.inner.scan_history_row_batches(table, row_id)
    }

    fn load_history_row_batch(
        &self,
        table: &str,
        branch: &str,
        row_id: ObjectId,
        batch_id: crate::row_histories::BatchId,
    ) -> Result<Option<crate::row_histories::StoredRowBatch>, StorageError> {
        self.inner
            .load_history_row_batch(table, branch, row_id, batch_id)
    }

    fn load_history_query_row_batch(
        &self,
        table: &str,
        branch: &str,
        row_id: ObjectId,
        batch_id: crate::row_histories::BatchId,
    ) -> Result<Option<crate::row_histories::QueryRowBatch>, StorageError> {
        self.inner
            .load_history_query_row_batch(table, branch, row_id, batch_id)
    }

    fn load_history_row_batch_for_schema_hash(
        &self,
        table: &str,
        schema_hash: crate::query_manager::types::SchemaHash,
        branch: &str,
        row_id: ObjectId,
        batch_id: crate::row_histories::BatchId,
    ) -> Result<Option<crate::row_histories::StoredRowBatch>, StorageError> {
        self.inner.load_history_row_batch_for_schema_hash(
            table,
            schema_hash,
            branch,
            row_id,
            batch_id,
        )
    }

    fn load_history_row_batch_any_branch(
        &self,
        table: &str,
        row_id: ObjectId,
        batch_id: crate::row_histories::BatchId,
    ) -> Result<Option<crate::row_histories::StoredRowBatch>, StorageError> {
        self.inner
            .load_history_row_batch_any_branch(table, row_id, batch_id)
    }

    fn load_history_query_row_batch_any_branch(
        &self,
        table: &str,
        row_id: ObjectId,
        batch_id: crate::row_histories::BatchId,
    ) -> Result<Option<crate::row_histories::QueryRowBatch>, StorageError> {
        self.inner
            .load_history_query_row_batch_any_branch(table, row_id, batch_id)
    }

    fn row_batch_exists(
        &self,
        table: &str,
        branch: &str,
        row_id: ObjectId,
        batch_id: crate::row_histories::BatchId,
    ) -> Result<bool, StorageError> {
        self.inner.row_batch_exists(table, branch, row_id, batch_id)
    }

    fn scan_row_branch_tip_ids(
        &self,
        table: &str,
        branch: &str,
        row_id: ObjectId,
    ) -> Result<Vec<crate::row_histories::BatchId>, StorageError> {
        self.inner.scan_row_branch_tip_ids(table, branch, row_id)
    }

    fn load_history_row_batch_bytes(
        &self,
        table: &str,
        branch: &str,
        row_id: ObjectId,
        batch_id: crate::row_histories::BatchId,
    ) -> Result<Option<Vec<u8>>, StorageError> {
        self.inner
            .load_history_row_batch_bytes(table, branch, row_id, batch_id)
    }

    fn scan_history_region_bytes(
        &self,
        table: &str,
        scan: crate::row_histories::HistoryScan,
    ) -> Result<Vec<Vec<u8>>, StorageError> {
        self.inner.scan_history_region_bytes(table, scan)
    }

    fn scan_history_region(
        &self,
        table: &str,
        branch: &str,
        scan: crate::row_histories::HistoryScan,
    ) -> Result<Vec<crate::row_histories::StoredRowBatch>, StorageError> {
        self.inner.scan_history_region(table, branch, scan)
    }

    fn index_insert(
        &mut self,
        table: &str,
        column: &str,
        branch: &str,
        value: &Value,
        row_id: ObjectId,
    ) -> Result<(), StorageError> {
        self.inner
            .index_insert(table, column, branch, value, row_id)
    }

    fn index_remove(
        &mut self,
        table: &str,
        column: &str,
        branch: &str,
        value: &Value,
        row_id: ObjectId,
    ) -> Result<(), StorageError> {
        self.inner
            .index_remove(table, column, branch, value, row_id)
    }

    fn apply_index_mutations(
        &mut self,
        mutations: &[crate::storage::IndexMutation<'_>],
    ) -> Result<(), StorageError> {
        self.calls.lock().unwrap().separate_index_mutation_calls += 1;
        self.inner.apply_index_mutations(mutations)
    }

    fn index_lookup(
        &self,
        table: &str,
        column: &str,
        branch: &str,
        value: &Value,
    ) -> Vec<ObjectId> {
        self.inner.index_lookup(table, column, branch, value)
    }

    fn index_range(
        &self,
        table: &str,
        column: &str,
        branch: &str,
        start: std::ops::Bound<&Value>,
        end: std::ops::Bound<&Value>,
    ) -> Vec<ObjectId> {
        self.inner.index_range(table, column, branch, start, end)
    }

    fn index_scan_all(&self, table: &str, column: &str, branch: &str) -> Vec<ObjectId> {
        self.inner.index_scan_all(table, column, branch)
    }

    fn flush(&self) -> Result<(), StorageError> {
        self.inner.flush()
    }

    fn flush_wal(&self) -> Result<(), StorageError> {
        self.calls.lock().unwrap().flush_wal_calls += 1;
        self.inner.flush_wal()
    }

    fn close(&self) -> Result<(), StorageError> {
        self.inner.close()
    }
}

fn test_schema() -> Schema {
    SchemaBuilder::new()
        .table(
            TableSchema::builder("users")
                .column("id", ColumnType::Uuid)
                .column("name", ColumnType::Text),
        )
        .build()
}

fn schema_evolution_v1() -> Schema {
    test_schema()
}

fn schema_evolution_v2() -> Schema {
    SchemaBuilder::new()
        .table(
            TableSchema::builder("users")
                .column("id", ColumnType::Uuid)
                .column("name", ColumnType::Text)
                .column("email", ColumnType::Text),
        )
        .build()
}

fn protected_documents_schema() -> Schema {
    let policies = TablePolicies::new()
        .with_select(PolicyExpr::eq_session("owner_id", vec!["user_id".into()]))
        .with_insert(PolicyExpr::eq_session("owner_id", vec!["user_id".into()]));

    SchemaBuilder::new()
        .table(
            TableSchema::builder("documents")
                .column("owner_id", ColumnType::Text)
                .column("title", ColumnType::Text)
                .policies(policies),
        )
        .build()
}

fn session_exists_rel_teams_schema() -> Schema {
    use crate::query_manager::relation_ir::{
        ColumnRef, JoinCondition, JoinKind, PredicateCmpOp, PredicateExpr, RelExpr, RowIdRef,
        ValueRef,
    };

    let team_select_policy = PolicyExpr::ExistsRel {
        rel: RelExpr::Filter {
            input: Box::new(RelExpr::Join {
                left: Box::new(RelExpr::TableScan {
                    table: TableName::new("user_team_edges"),
                }),
                right: Box::new(RelExpr::TableScan {
                    table: TableName::new("teams"),
                }),
                on: vec![JoinCondition {
                    left: ColumnRef::scoped("user_team_edges", "team_id"),
                    right: ColumnRef::scoped("__join_0", "id"),
                }],
                join_kind: JoinKind::Inner,
            }),
            predicate: PredicateExpr::And(vec![
                PredicateExpr::Cmp {
                    left: ColumnRef::scoped("user_team_edges", "user_id"),
                    op: PredicateCmpOp::Eq,
                    right: ValueRef::SessionRef(vec!["user_id".into()]),
                },
                PredicateExpr::Cmp {
                    left: ColumnRef::scoped("__join_0", "id"),
                    op: PredicateCmpOp::Eq,
                    right: ValueRef::RowId(RowIdRef::Outer),
                },
            ]),
        },
    };

    SchemaBuilder::new()
        .table(
            TableSchema::builder("teams")
                .column("name", ColumnType::Text)
                .policies(
                    TablePolicies::new()
                        .with_select(team_select_policy)
                        .with_insert(PolicyExpr::True),
                ),
        )
        .table(
            TableSchema::builder("user_team_edges")
                .column("user_id", ColumnType::Text)
                .column("team_id", ColumnType::Uuid)
                .policies(TablePolicies::new().with_insert(PolicyExpr::True)),
        )
        .build()
}

fn structural_session_exists_rel_teams_schema() -> Schema {
    SchemaBuilder::new()
        .table(TableSchema::builder("teams").column("name", ColumnType::Text))
        .table(
            TableSchema::builder("user_team_edges")
                .column("user_id", ColumnType::Text)
                .column("team_id", ColumnType::Uuid),
        )
        .build()
}

fn users_insert_denied_authorization_schema() -> Schema {
    SchemaBuilder::new()
        .table(
            TableSchema::builder("users")
                .column("id", ColumnType::Uuid)
                .column("name", ColumnType::Text)
                .policies(TablePolicies::new().with_insert(PolicyExpr::False)),
        )
        .build()
}

fn defaulted_todos_schema() -> Schema {
    SchemaBuilder::new()
        .table(
            TableSchema::builder("todos")
                .column("title", ColumnType::Text)
                .column_with_default("done", ColumnType::Boolean, Value::Boolean(false)),
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

fn insert_and_wait_for_batch<S: Storage, Sch: Scheduler>(
    core: &mut RuntimeCore<S, Sch>,
    table: &str,
    values: HashMap<String, Value>,
    write_context: Option<&WriteContext>,
    tier: DurabilityTier,
) -> std::result::Result<
    (
        InsertedRow,
        futures::channel::oneshot::Receiver<PersistedWriteAck>,
    ),
    RuntimeError,
> {
    let (row, batch_id) = core.insert(table, values, write_context)?;
    let receiver = core.wait_for_batch(batch_id, tier)?;
    Ok((row, receiver))
}

fn delete_and_wait_for_batch<S: Storage, Sch: Scheduler>(
    core: &mut RuntimeCore<S, Sch>,
    object_id: ObjectId,
    write_context: Option<&WriteContext>,
    tier: DurabilityTier,
) -> std::result::Result<futures::channel::oneshot::Receiver<PersistedWriteAck>, RuntimeError> {
    let batch_id = core.delete(object_id, write_context)?;
    core.wait_for_batch(batch_id, tier)
}

fn staged_user_row(
    row_id: ObjectId,
    batch_id: BatchId,
    updated_at: u64,
    name: &str,
) -> crate::row_histories::StoredRowBatch {
    crate::row_histories::StoredRowBatch::new_with_batch_id(
        batch_id,
        row_id,
        "main",
        Vec::<BatchId>::new(),
        encode_row(
            &test_schema()[&TableName::new("users")].columns,
            &user_row_values(row_id, name),
        )
        .expect("user test row should encode"),
        crate::metadata::RowProvenance::for_insert(row_id.to_string(), updated_at),
        HashMap::new(),
        crate::row_histories::RowState::StagingPending,
        None,
    )
}

fn document_insert_values(owner_id: &str, title: &str) -> HashMap<String, Value> {
    HashMap::from([
        ("owner_id".to_string(), Value::Text(owner_id.to_string())),
        ("title".to_string(), Value::Text(title.to_string())),
    ])
}

fn project_insert_values(name: &str, owner_id: &str) -> HashMap<String, Value> {
    HashMap::from([
        ("name".to_string(), Value::Text(name.to_string())),
        ("owner_id".to_string(), Value::Text(owner_id.to_string())),
    ])
}

fn todo_insert_values(
    title: &str,
    done: bool,
    description: Value,
    owner_id: &str,
    project: Value,
) -> HashMap<String, Value> {
    HashMap::from([
        ("title".to_string(), Value::Text(title.to_string())),
        ("done".to_string(), Value::Boolean(done)),
        ("description".to_string(), description),
        ("owner_id".to_string(), Value::Text(owner_id.to_string())),
        ("project".to_string(), project),
    ])
}

fn create_runtime_with_schema_and_sync_manager(
    schema: Schema,
    app_name: &str,
    sync_manager: SyncManager,
) -> TestCore {
    let app_id = AppId::from_name(app_name);
    let schema_manager = SchemaManager::new(sync_manager, schema, app_id, "dev", "main").unwrap();
    let mut core = new_test_core(schema_manager, MemoryStorage::new(), NoopScheduler);
    core.immediate_tick();
    core
}

fn create_runtime_with_schema(schema: Schema, app_name: &str) -> TestCore {
    create_runtime_with_schema_and_sync_manager(schema, app_name, SyncManager::new())
}

fn create_runtime_with_storage(schema: Schema, app_name: &str, storage: MemoryStorage) -> TestCore {
    create_runtime_with_storage_and_sync_manager(schema, app_name, storage, SyncManager::new())
}

fn create_runtime_with_storage_and_sync_manager(
    schema: Schema,
    app_name: &str,
    storage: MemoryStorage,
    sync_manager: SyncManager,
) -> TestCore {
    let app_id = AppId::from_name(app_name);
    let schema_manager = SchemaManager::new(sync_manager, schema, app_id, "dev", "main").unwrap();
    let mut core = new_test_core(schema_manager, storage, NoopScheduler);
    core.immediate_tick();
    core
}

fn create_runtime_with_boxed_storage(
    schema: Schema,
    app_name: &str,
    storage: Box<dyn Storage>,
) -> BoxedStorageTestCore {
    let app_id = AppId::from_name(app_name);
    let schema_manager =
        SchemaManager::new(SyncManager::new(), schema, app_id, "dev", "main").unwrap();
    let mut core = new_test_core(schema_manager, storage, NoopScheduler);
    core.immediate_tick();
    core
}

fn create_test_runtime() -> TestCore {
    create_runtime_with_schema(test_schema(), "test-app")
}

fn documents_query_by_title(title: &str) -> Query {
    QueryBuilder::new("documents")
        .filter_eq("title", Value::Text(title.into()))
        .build()
}

fn column_index(schema: &Schema, table: &str, column: &str) -> usize {
    schema
        .get(&TableName::new(table))
        .unwrap_or_else(|| panic!("table '{table}' should exist"))
        .columns
        .column_index(column)
        .unwrap_or_else(|| panic!("column '{column}' should exist on table '{table}'"))
}

/// Helper to execute a query synchronously via subscribe/tick/unsubscribe.
fn execute_query(core: &mut TestCore, query: Query) -> Vec<(ObjectId, Vec<Value>)> {
    let sub_id = core
        .schema_manager_mut()
        .query_manager_mut()
        .subscribe(query)
        .unwrap();
    core.immediate_tick();
    let results = core
        .schema_manager_mut()
        .query_manager_mut()
        .get_subscription_results(sub_id);
    core.schema_manager_mut()
        .query_manager_mut()
        .unsubscribe_with_sync(sub_id);
    results
}

fn execute_runtime_query(
    core: &mut TestCore,
    query: Query,
    session: Option<Session>,
) -> Vec<(ObjectId, Vec<Value>)> {
    execute_runtime_query_with_propagation(
        core,
        query,
        session,
        crate::sync_manager::QueryPropagation::Full,
    )
}

fn execute_local_runtime_query(
    core: &mut TestCore,
    query: Query,
    session: Option<Session>,
) -> Vec<(ObjectId, Vec<Value>)> {
    execute_runtime_query_with_propagation(
        core,
        query,
        session,
        crate::sync_manager::QueryPropagation::LocalOnly,
    )
}

fn execute_runtime_query_with_propagation(
    core: &mut TestCore,
    query: Query,
    session: Option<Session>,
    propagation: crate::sync_manager::QueryPropagation,
) -> Vec<(ObjectId, Vec<Value>)> {
    execute_runtime_query_with_durability_and_propagation(
        core,
        query,
        session,
        ReadDurabilityOptions::default(),
        propagation,
    )
}

fn execute_runtime_query_with_durability_and_propagation(
    core: &mut TestCore,
    query: Query,
    session: Option<Session>,
    durability: ReadDurabilityOptions,
    propagation: crate::sync_manager::QueryPropagation,
) -> Vec<(ObjectId, Vec<Value>)> {
    let waker = noop_waker();
    let mut cx = std::task::Context::from_waker(&waker);

    let mut future = core.query_with_propagation(query, session, durability, propagation);

    match Pin::new(&mut future).poll(&mut cx) {
        Poll::Ready(Ok(results)) => results,
        Poll::Ready(Err(err)) => panic!("query should succeed: {err:?}"),
        Poll::Pending => panic!("query should resolve immediately"),
    }
}

fn execute_runtime_query_with_local_overlay(
    core: &mut TestCore,
    query: Query,
    session: Option<Session>,
    durability: ReadDurabilityOptions,
    propagation: crate::sync_manager::QueryPropagation,
    overlay: QueryLocalOverlay,
) -> Vec<(ObjectId, Vec<Value>)> {
    let waker = noop_waker();
    let mut cx = std::task::Context::from_waker(&waker);

    let mut future =
        core.query_with_local_overlay(query, session, durability, propagation, overlay);

    match Pin::new(&mut future).poll(&mut cx) {
        Poll::Ready(Ok(results)) => results,
        Poll::Ready(Err(err)) => panic!("query should succeed: {err:?}"),
        Poll::Pending => panic!("query should resolve immediately"),
    }
}

fn decode_added_rows(delta: &SubscriptionDelta) -> Vec<(ObjectId, Vec<Value>)> {
    delta
        .ordered_delta
        .added
        .iter()
        .map(|row| {
            let values = decode_row(&delta.descriptor, &row.row.data).unwrap_or_else(|err| {
                panic!(
                    "subscription row {:?} should decode successfully: {err:?}",
                    row.row.id
                )
            });
            (row.row.id, values)
        })
        .collect()
}

fn pump_client_messages_to_server(
    client: &mut TestCore,
    server: &mut TestCore,
    server_id: ServerId,
    client_id: ClientId,
) -> bool {
    let mut any_messages = false;

    client.batched_tick();
    for entry in client.sync_sender().take() {
        if entry.destination == Destination::Server(server_id) {
            any_messages = true;
            server.park_sync_message(InboxEntry {
                source: Source::Client(client_id),
                payload: entry.payload,
            });
        }
    }
    server.batched_tick();
    server.immediate_tick();

    any_messages
}

struct ClientForServer<'a> {
    core: &'a mut TestCore,
    server_id: ServerId,
    client_id: ClientId,
}

fn pump_server_messages_to_clients(
    server: &mut TestCore,
    clients: &mut [ClientForServer<'_>],
    server_outputs: &mut Vec<OutboxEntry>,
) -> bool {
    let mut any_messages = false;

    server.batched_tick();

    let server_out = server.sync_sender().take();
    server_outputs.extend(server_out.iter().cloned());
    for entry in server_out {
        let Destination::Client(destination_client_id) = entry.destination else {
            continue;
        };

        if let Some(client) = clients
            .iter_mut()
            .find(|client| client.client_id == destination_client_id)
        {
            any_messages = true;
            client.core.park_sync_message(InboxEntry {
                source: Source::Server(client.server_id),
                payload: entry.payload,
            });
        }
    }

    any_messages
}

fn sync_server_with_clients(
    server: &mut TestCore,
    clients: &mut [ClientForServer<'_>],
) -> Vec<OutboxEntry> {
    let mut server_outputs = Vec::new();

    for _ in 0..10 {
        let mut any_messages = false;

        for client in clients.iter_mut() {
            any_messages |= pump_client_messages_to_server(
                client.core,
                server,
                client.server_id,
                client.client_id,
            );
        }

        any_messages |= pump_server_messages_to_clients(server, clients, &mut server_outputs);

        for client in clients.iter_mut() {
            client.core.batched_tick();
            client.core.immediate_tick();
        }

        if !any_messages {
            break;
        }
    }

    server_outputs
}

fn outbox_has_object_update_for_client(
    entries: &[OutboxEntry],
    client_id: ClientId,
    object_id: ObjectId,
) -> bool {
    entries.iter().any(|entry| {
        matches!(
            &entry.destination,
            Destination::Client(dest_client_id) if *dest_client_id == client_id
        ) && match &entry.payload {
            SyncPayload::RowBatchNeeded { row, .. } | SyncPayload::RowBatchCreated { row, .. } => {
                row.row_id == object_id
            }
            _ => false,
        }
    })
}

/// Three-tier RuntimeCore setup for durability tests.
struct ThreeTierRC {
    a: TestCore,
    b: TestCore,
    c: TestCore,
    a_client_of_b: ClientId,
    b_server_for_a: ServerId,
    b_client_of_c: ClientId,
    c_server_for_b: ServerId,
}

fn create_3tier_rc() -> ThreeTierRC {
    let schema = test_schema();
    create_3tier_rc_with_schema(schema)
}

fn create_3tier_rc_with_schema(schema: Schema) -> ThreeTierRC {
    let app_id = AppId::from_name("durability-test");

    // A = client (no tier)
    let sm_a = SyncManager::new();
    let mgr_a = SchemaManager::new(sm_a, schema.clone(), app_id, "dev", "main").unwrap();
    let mut a = new_test_core(mgr_a, MemoryStorage::new(), NoopScheduler);

    // B = Worker server
    let sm_b = SyncManager::new().with_durability_tier(DurabilityTier::Local);
    let mgr_b = SchemaManager::new(sm_b, schema.clone(), app_id, "dev", "main").unwrap();
    let mut b = new_test_core(mgr_b, MemoryStorage::new(), NoopScheduler);

    // C = EdgeServer
    let sm_c = SyncManager::new().with_durability_tier(DurabilityTier::EdgeServer);
    let mgr_c = SchemaManager::new(sm_c, schema, app_id, "dev", "main").unwrap();
    let mut c = new_test_core(mgr_c, MemoryStorage::new(), NoopScheduler);

    let a_client_of_b = ClientId::new();
    let b_server_for_a = ServerId::new();
    let b_client_of_c = ClientId::new();
    let c_server_for_b = ServerId::new();

    // Topology: A ↔ B ↔ C
    {
        b.add_client(a_client_of_b, None);
        b.schema_manager_mut()
            .query_manager_mut()
            .sync_manager_mut()
            .set_client_role(a_client_of_b, ClientRole::Peer);
    }
    a.add_server(b_server_for_a);

    {
        c.add_client(b_client_of_c, None);
        c.schema_manager_mut()
            .query_manager_mut()
            .sync_manager_mut()
            .set_client_role(b_client_of_c, ClientRole::Peer);
    }
    b.add_server(c_server_for_b);

    // Initial tick + clear initial sync messages
    a.immediate_tick();
    b.immediate_tick();
    c.immediate_tick();
    a.batched_tick();
    b.batched_tick();
    c.batched_tick();
    a.sync_sender().take();
    b.sync_sender().take();
    c.sync_sender().take();

    ThreeTierRC {
        a,
        b,
        c,
        a_client_of_b,
        b_server_for_a,
        b_client_of_c,
        c_server_for_b,
    }
}

/// Pump all messages between 3 RuntimeCore nodes until quiescent.
fn pump_3tier(s: &mut ThreeTierRC) {
    for _ in 0..10 {
        let mut any_messages = false;

        // A outbox → B
        s.a.batched_tick();
        let a_out = s.a.sync_sender().take();
        for entry in a_out {
            if entry.destination == Destination::Server(s.b_server_for_a) {
                any_messages = true;
                s.b.park_sync_message(InboxEntry {
                    source: Source::Client(s.a_client_of_b),
                    payload: entry.payload,
                });
            }
        }

        // B process, then route outbox to A or C
        s.b.batched_tick();
        s.b.immediate_tick();
        s.b.batched_tick();
        let b_out = s.b.sync_sender().take();
        for entry in b_out {
            match &entry.destination {
                Destination::Client(cid) if *cid == s.a_client_of_b => {
                    any_messages = true;
                    s.a.park_sync_message(InboxEntry {
                        source: Source::Server(s.b_server_for_a),
                        payload: entry.payload,
                    });
                }
                Destination::Server(sid) if *sid == s.c_server_for_b => {
                    any_messages = true;
                    s.c.park_sync_message(InboxEntry {
                        source: Source::Client(s.b_client_of_c),
                        payload: entry.payload,
                    });
                }
                _ => {}
            }
        }

        // C process, then route outbox to B
        s.c.batched_tick();
        s.c.immediate_tick();
        s.c.batched_tick();
        let c_out = s.c.sync_sender().take();
        for entry in c_out {
            if entry.destination == Destination::Client(s.b_client_of_c) {
                any_messages = true;
                s.b.park_sync_message(InboxEntry {
                    source: Source::Server(s.c_server_for_b),
                    payload: entry.payload,
                });
            }
        }

        // A processes incoming
        s.a.batched_tick();
        s.a.immediate_tick();

        if !any_messages {
            break;
        }
    }
}

/// Pump only A → B (one hop, no C).
fn pump_a_to_b(s: &mut ThreeTierRC) {
    s.a.batched_tick();
    let a_out = s.a.sync_sender().take();
    for entry in a_out {
        if entry.destination == Destination::Server(s.b_server_for_a) {
            s.b.park_sync_message(InboxEntry {
                source: Source::Client(s.a_client_of_b),
                payload: entry.payload,
            });
        }
    }
    s.b.batched_tick();
    s.b.immediate_tick();
}

/// Route B's outbox to both A and C as appropriate.
fn route_b_outbox(s: &mut ThreeTierRC) {
    s.b.batched_tick();
    let b_out = s.b.sync_sender().take();
    for entry in b_out {
        match &entry.destination {
            Destination::Client(cid) if *cid == s.a_client_of_b => {
                s.a.park_sync_message(InboxEntry {
                    source: Source::Server(s.b_server_for_a),
                    payload: entry.payload,
                });
            }
            Destination::Server(sid) if *sid == s.c_server_for_b => {
                s.c.park_sync_message(InboxEntry {
                    source: Source::Client(s.b_client_of_c),
                    payload: entry.payload,
                });
            }
            _ => {}
        }
    }
}

/// Pump B → A (acks back).
fn pump_b_to_a(s: &mut ThreeTierRC) {
    route_b_outbox(s);
    s.a.batched_tick();
    s.a.immediate_tick();
}

/// Pump B → C (forward to edge).
fn pump_b_to_c(s: &mut ThreeTierRC) {
    route_b_outbox(s);
    s.c.batched_tick();
    s.c.immediate_tick();
}

/// Pump C → B → A (edge ack relay).
fn pump_c_to_b_to_a(s: &mut ThreeTierRC) {
    // C → B
    s.c.batched_tick();
    let c_out = s.c.sync_sender().take();
    for entry in c_out {
        if entry.destination == Destination::Client(s.b_client_of_c) {
            s.b.park_sync_message(InboxEntry {
                source: Source::Server(s.c_server_for_b),
                payload: entry.payload,
            });
        }
    }
    s.b.batched_tick();
    s.b.immediate_tick();

    // B → A
    pump_b_to_a(s);
}

fn count_query_subscriptions_to_server(entries: &[OutboxEntry], server_id: ServerId) -> usize {
    entries
        .iter()
        .filter(|entry| {
            matches!(
                &entry.destination,
                Destination::Server(dest_server_id) if *dest_server_id == server_id
            ) && matches!(&entry.payload, SyncPayload::QuerySubscription { .. })
        })
        .count()
}

fn noop_waker() -> std::task::Waker {
    fn noop(_: *const ()) {}
    fn clone(_: *const ()) -> std::task::RawWaker {
        std::task::RawWaker::new(std::ptr::null(), &VTABLE)
    }
    static VTABLE: std::task::RawWakerVTable =
        std::task::RawWakerVTable::new(clone, noop, noop, noop);
    unsafe { std::task::Waker::from_raw(std::task::RawWaker::new(std::ptr::null(), &VTABLE)) }
}

mod accepted_batch_downgrade;
mod authz_cache_runtime;
mod basic;
mod batch_fate_offer_differential;
mod batched_tick_parked_drain;
mod cold_boot_generation_universe;
mod cross_generation_backref_read;
mod cross_generation_differential;
mod cross_generation_local_write;
mod cross_generation_policy;
mod delivery_confirmation;
mod delivery_convergence_differential;
mod fk_remove_error;
mod incremental_scan;
mod install_transport_tests;
mod locator_ladder_heal;
mod locator_persistence_differential;
mod locator_warmth;
mod query_subscription;
mod read_pass;
mod rejected_write_retires_tracking;
mod schema_catalogue;
mod sealed_batch_cost;
mod settle_budget_ticks;
mod subscription_fanout_cost;
mod subscription_registration_cost;
mod support;
mod sync_replay;
mod unappliable_row_logging;
mod write_batch;

/// A wiped upstream (fresh store behind the same endpoint) RELEARNS settled
/// rows on reconnect: the reconnect replay re-offers the client's durable
/// batches (`RowBatchCreated`), and the amnesiac server accepts them and
/// forwards them to its own upstream. This is the recovery property the
/// "full wipe" story leans on — if it ever regresses, a wipe permanently
/// strands every deterministic-id singleton (Saved Messages, assistant chat).
///
/// Also documented here: the ENGINE has no column-based duplicate refusal —
/// re-inserting the same column values mints a fresh row ObjectId. The
/// "row already exists" behavior the app sees for deterministic ids lives in
/// the TS binding (which maps the caller-provided id onto the engine row id
/// and checks the local store), not in the core.
#[test]
fn a_wiped_upstream_relearns_settled_rows_on_reconnect() {
    let mut s = create_3tier_rc();

    let fixed_id = ObjectId::new();
    let deterministic_row = || {
        HashMap::from([
            ("id".to_string(), Value::Uuid(fixed_id)),
            (
                "name".to_string(),
                Value::Text("Saved Messages".to_string()),
            ),
        ])
    };

    // The app always holds live queries; reconnect replays exactly these.
    let _sub =
        s.a.subscribe(Query::new("users"), |_delta| {}, None)
            .unwrap();

    let ((created_row_id, _values), _ack) = insert_and_wait_for_batch(
        &mut s.a,
        "users",
        deterministic_row(),
        None,
        DurabilityTier::Local,
    )
    .expect("the first create must succeed");

    // Positive control at the emission level: the client offers the batch upstream.
    s.a.batched_tick();
    let first_out = s.a.sync_sender().take();
    assert!(
        first_out.iter().any(|e| matches!(
            &e.payload,
            SyncPayload::RowBatchCreated { row, .. } if row.row_id == created_row_id
        )),
        "the fresh insert must be offered to the server"
    );
    for e in first_out {
        if e.destination == Destination::Server(s.b_server_for_a) {
            s.b.park_sync_message(InboxEntry {
                source: Source::Client(s.a_client_of_b),
                payload: e.payload,
            });
        }
    }
    pump_3tier(&mut s);

    // The wipe: same endpoint from the client's point of view, brand-new server
    // state behind it (the client keeps its own store and settled state).
    let app_id = AppId::from_name("durability-test");
    let sm_b2 = SyncManager::new().with_durability_tier(DurabilityTier::Local);
    let mgr_b2 = SchemaManager::new(sm_b2, test_schema(), app_id, "dev", "main").unwrap();
    let mut b2 = new_test_core(mgr_b2, MemoryStorage::new(), NoopScheduler);
    b2.add_client(s.a_client_of_b, None);
    b2.schema_manager_mut()
        .query_manager_mut()
        .sync_manager_mut()
        .set_client_role(s.a_client_of_b, ClientRole::Peer);
    b2.add_server(s.c_server_for_b);
    b2.immediate_tick();
    b2.batched_tick();
    b2.sync_sender().take();
    s.b = b2;

    // Reconnect semantics: drop and re-add the upstream — this is what replays
    // active query subscriptions (rc_replays_active_queries_on_upstream_reconnect).
    s.a.remove_server(s.b_server_for_a);
    s.a.add_server(s.b_server_for_a);

    // Pump manually so every payload the client sends after the reconnect is
    // recorded before being delivered.
    let mut reoffered_fixed_row = false;
    let mut reoffer_kinds: Vec<&'static str> = Vec::new();
    let mut b2_forwarded_to_c = false;
    for _ in 0..10 {
        let mut any_messages = false;

        s.a.batched_tick();
        for entry in s.a.sync_sender().take() {
            if entry.destination == Destination::Server(s.b_server_for_a) {
                any_messages = true;
                match &entry.payload {
                    SyncPayload::RowBatchCreated { row, .. } if row.row_id == created_row_id => {
                        reoffered_fixed_row = true;
                        reoffer_kinds.push("RowBatchCreated");
                    }
                    SyncPayload::RowBatchNeeded { row, .. } if row.row_id == created_row_id => {
                        reoffered_fixed_row = true;
                        reoffer_kinds.push("RowBatchNeeded");
                    }
                    _ => {}
                }
                s.b.park_sync_message(InboxEntry {
                    source: Source::Client(s.a_client_of_b),
                    payload: entry.payload,
                });
            }
        }

        s.b.batched_tick();
        s.b.immediate_tick();
        s.b.batched_tick();
        for entry in s.b.sync_sender().take() {
            match &entry.destination {
                Destination::Client(cid) if *cid == s.a_client_of_b => {
                    any_messages = true;
                    s.a.park_sync_message(InboxEntry {
                        source: Source::Server(s.b_server_for_a),
                        payload: entry.payload,
                    });
                }
                Destination::Server(sid) if *sid == s.c_server_for_b => {
                    any_messages = true;
                    if matches!(
                        &entry.payload,
                        SyncPayload::RowBatchCreated { row, .. } if row.row_id == created_row_id
                    ) {
                        b2_forwarded_to_c = true;
                    }
                    s.c.park_sync_message(InboxEntry {
                        source: Source::Client(s.b_client_of_c),
                        payload: entry.payload,
                    });
                }
                _ => {}
            }
        }

        s.c.batched_tick();
        s.c.immediate_tick();
        for entry in s.c.sync_sender().take() {
            if entry.destination == Destination::Client(s.b_client_of_c) {
                any_messages = true;
                s.b.park_sync_message(InboxEntry {
                    source: Source::Server(s.c_server_for_b),
                    payload: entry.payload,
                });
            }
        }

        s.a.batched_tick();
        s.a.immediate_tick();

        if !any_messages {
            break;
        }
    }

    let duplicate = s.a.insert("users", deterministic_row(), None);

    assert!(
        reoffered_fixed_row && reoffer_kinds.contains(&"RowBatchCreated"),
        "reconnect must re-offer the settled row to the wiped upstream \
         (got kinds {reoffer_kinds:?}) — without this a wipe strands every \
         deterministic-id singleton"
    );
    assert!(
        b2_forwarded_to_c,
        "the wiped upstream must accept the re-offered row and forward it to its own upstream"
    );
    let ((second_row_id, _), _) =
        duplicate.expect("engine-level insert has no column-id dedupe; the TS binding owns that");
    assert_ne!(
        second_row_id, created_row_id,
        "engine identity is the row ObjectId, not the id column — a re-insert mints a new row"
    );
}

/// RED GATE (prod incident 2026-08-09, UPSTREAM-DEFECTS #11): a seal that
/// arrives for a batch whose ROWS never made it (they rode a dying connection)
/// must not become an eternal full-store-scan loop on the server. In the field
/// this pinned a core: every pass scanned the whole history region (~3.5s on
/// the production store), made no progress, and repeated on the next tick,
/// starving sync entirely.
///
/// The contract this gate encodes for the fix:
///   1. the full-store fallback runs AT MOST ONCE per stuck batch while no new
///      writes land (negative cache), and
///   2. the sealer is answered with `BatchFate::Missing` — whose client-side
///      handler already retransmits the batch rows — so the two-phase loop
///      closes instead of spinning.
#[test]
fn a_seal_without_rows_must_not_loop_full_store_scans() {
    // Two nodes, prod-shaped: the server is the AUTHORITY tier (EdgeServer,
    // like production jazz-sync). Client fates recorded here become
    // authoritative — the ingredient the 3-tier middle node lacks.
    let app_id = AppId::from_name("durability-test");
    let schema = test_schema();
    let sm_a = SyncManager::new();
    let mgr_a = SchemaManager::new(sm_a, schema.clone(), app_id, "dev", "main").unwrap();
    let mut a = new_test_core(mgr_a, MemoryStorage::new(), NoopScheduler);
    let sm_e = SyncManager::new().with_durability_tier(DurabilityTier::EdgeServer);
    let mgr_e = SchemaManager::new(sm_e, schema, app_id, "dev", "main").unwrap();
    let mut e = new_test_core(mgr_e, MemoryStorage::new(), NoopScheduler);
    let a_client_of_e = ClientId::new();
    let e_server_for_a = ServerId::new();
    e.add_client(a_client_of_e, None);
    e.schema_manager_mut()
        .query_manager_mut()
        .sync_manager_mut()
        .set_client_role(a_client_of_e, ClientRole::Peer);
    a.add_server(e_server_for_a);
    a.immediate_tick();
    e.immediate_tick();
    a.batched_tick();
    e.batched_tick();
    a.sync_sender().take();
    e.sync_sender().take();
    struct Pair {
        a: TestCore,
        b: TestCore,
        a_client_of_b: ClientId,
        b_server_for_a: ServerId,
    }
    let mut s = Pair {
        a,
        b: e,
        a_client_of_b: a_client_of_e,
        b_server_for_a: e_server_for_a,
    };

    let ((row_id, _values), _ack) = insert_and_wait_for_batch(
        &mut s.a,
        "users",
        HashMap::from([
            ("id".to_string(), Value::Uuid(ObjectId::new())),
            ("name".to_string(), Value::Text("lost rows".to_string())),
        ]),
        None,
        DurabilityTier::EdgeServer,
    )
    .expect("client-side insert");

    // Capture the client's upload and deliver everything EXCEPT the row
    // payload itself: the rows died with the first connection, the seal
    // arrived on the second.
    s.a.batched_tick();
    let mut dropped_row_payloads = 0usize;
    let mut retried_payloads = Vec::new();
    for entry in s.a.sync_sender().take() {
        if entry.destination != Destination::Server(s.b_server_for_a) {
            continue;
        }
        match &entry.payload {
            SyncPayload::RowBatchCreated { row, .. } if row.row_id == row_id => {
                dropped_row_payloads += 1;
            }
            _ => retried_payloads.push(entry.payload),
        }
    }
    assert!(
        dropped_row_payloads > 0,
        "the scenario requires actually dropping the row payload"
    );
    assert!(
        !retried_payloads.is_empty(),
        "the seal/durability payloads must survive to be retried"
    );
    // Deliver the seal once and check what the server persisted — production
    // stores an orphan sealed submission for every such batch (609 of them in
    // the incident store). If this harness delivery does not persist one,
    // seed it exactly as production holds it: from the client's own payload.
    for payload in retried_payloads.clone() {
        s.b.park_sync_message(InboxEntry {
            source: Source::Client(s.a_client_of_b),
            payload,
        });
    }
    s.b.batched_tick();
    s.b.immediate_tick();
    {
        use crate::storage::Storage as _;
        let batch_id = retried_payloads
            .iter()
            .find_map(|p| match p {
                SyncPayload::SealBatch { submission } => Some(submission.batch_id),
                _ => None,
            })
            .expect("a SealBatch payload");
        let persisted =
            s.b.storage()
                .load_sealed_batch_submission(batch_id)
                .unwrap();
        eprintln!(
            "server persisted orphan submission after seal: {}",
            persisted.is_some()
        );
        if persisted.is_none() {
            if let Some(SyncPayload::SealBatch { submission }) = retried_payloads
                .iter()
                .find(|p| matches!(p, SyncPayload::SealBatch { .. }))
            {
                s.b.storage_mut()
                    .upsert_sealed_batch_submission(submission)
                    .expect("seed the production-shaped orphan submission");
                eprintln!("seeded orphan submission (production shape)");
            }
        }
    }

    // Let the server process the orphan seal across several ticks, re-parking
    // whatever the client keeps retrying (the field client retried its seal on
    // a live connection for 38+ minutes).
    let scans_before =
        crate::runtime_core::LOCAL_BATCH_FULL_SCANS.load(std::sync::atomic::Ordering::Relaxed);
    let _orphan_batch_id = retried_payloads
        .iter()
        .find_map(|p| match p {
            SyncPayload::SealBatch { submission } => Some(submission.batch_id),
            _ => None,
        })
        .expect("a SealBatch payload");
    let mut missing_answered = false;
    for round in 0..5 {
        // The field driver, named by the production stack dump: a DIVERGED
        // client keeps uploading rows whose parents this store never had
        // (ParentNotFound). Each one the server rejects walks
        // apply_received_batch_fate -> mark_local_batch_rows_rejected ->
        // local_batch_rows -> full-store scan.
        let diverged_parent = crate::row_histories::BatchId(*ObjectId::new().uuid().as_bytes());
        let diverged_batch_id = crate::row_histories::BatchId(*ObjectId::new().uuid().as_bytes());
        let diverged_row_id = ObjectId::new();
        let diverged_row = crate::row_histories::StoredRowBatch::new_with_batch_id(
            diverged_batch_id,
            diverged_row_id,
            "main",
            vec![diverged_parent],
            encode_row(
                &test_schema()[&TableName::new("users")].columns,
                &user_row_values(diverged_row_id, &format!("diverged-{round}")),
            )
            .expect("diverged row encodes"),
            crate::metadata::RowProvenance::for_insert(diverged_row_id.to_string(), 1),
            HashMap::new(),
            crate::row_histories::RowState::StagingPending,
            None,
        );
        s.b.park_sync_message(InboxEntry {
            source: Source::Client(s.a_client_of_b),
            payload: SyncPayload::RowBatchCreated {
                metadata: None,
                row: diverged_row,
            },
        });
        // Plus the seal retries the same client kept sending for 38 minutes.
        for payload in retried_payloads.clone() {
            s.b.park_sync_message(InboxEntry {
                source: Source::Client(s.a_client_of_b),
                payload,
            });
        }
        // Production is never quiet: unrelated writes land continuously
        // (presence heartbeats, tokens). Feed one per round so any
        // "re-scan on new input" behavior surfaces.
        let (_, _bg_ack) = insert_and_wait_for_batch(
            &mut s.a,
            "users",
            HashMap::from([
                ("id".to_string(), Value::Uuid(ObjectId::new())),
                (
                    "name".to_string(),
                    Value::Text(format!("background-{round}")),
                ),
            ]),
            None,
            DurabilityTier::EdgeServer,
        )
        .expect("background write");
        s.a.batched_tick();
        for entry in s.a.sync_sender().take() {
            if entry.destination == Destination::Server(s.b_server_for_a) {
                s.b.park_sync_message(InboxEntry {
                    source: Source::Client(s.a_client_of_b),
                    payload: entry.payload,
                });
            }
        }
        s.b.batched_tick();
        s.b.immediate_tick();
        for entry in s.b.sync_sender().take() {
            if let Destination::Client(cid) = &entry.destination {
                if *cid == s.a_client_of_b
                    && matches!(
                        &entry.payload,
                        SyncPayload::BatchFate {
                            fate: crate::batch_fate::BatchFate::Missing { .. },
                            ..
                        }
                    )
                {
                    missing_answered = true;
                }
            }
        }
    }
    let scans = crate::runtime_core::LOCAL_BATCH_FULL_SCANS
        .load(std::sync::atomic::Ordering::Relaxed)
        - scans_before;

    eprintln!("gate observation: scans={scans} missing_answered={missing_answered}");
    // The cost contract, driver-agnostic: ANY path deriving the pending set
    // over the persisted orphan pays local_batch_rows; with the orphan seeded,
    // repeated derivations must answer from the first scan's result.
    let scans_p0 = s.b.local_batch_full_scan_count();
    let pending_first = s.b.pending_batch_ids_needing_reconciliation_for_test();
    let pending_second = s.b.pending_batch_ids_needing_reconciliation_for_test();
    let derivation_scans = crate::runtime_core::LOCAL_BATCH_FULL_SCANS
        .load(std::sync::atomic::Ordering::Relaxed)
        - scans_p0;
    eprintln!(
        "pending derivations: first={} second={} scans={derivation_scans}",
        pending_first.len(),
        pending_second.len()
    );

    assert!(
        scans <= 1,
        "a stuck seal must cost at most one full-store scan while nothing changes; \
         got {scans} scans across 5 ticks — the production CPU-pin loop"
    );
    assert!(
        derivation_scans <= 1,
        "two pending-set derivations over one persisted orphan cost {derivation_scans} \
         full-store scans — repeated derivations must reuse the first scan's answer"
    );
    assert!(
        missing_answered,
        "the sealer must be told the batch is Missing so it retransmits the rows"
    );
}

/// RED GATE #2 (prod incident 2026-08-09, the "conveyor" half): settled
/// history must not cost full-store scans when the pending set is derived.
///
/// Production holds tens of thousands of settled batches (heartbeats, every
/// row ever written) whose batchId->rows index was cleared at settlement while
/// their fates/records persist. Deriving the pending-reconciliation set walks
/// those and, for every one whose fate still reads as unsettled, pays the
/// full-store fallback: one ~0.45s scan per batch, hours of pinned CPU after
/// every server restart, with the tick loop blocked the whole time.
#[test]
fn settled_history_must_not_cost_full_store_scans_on_reconciliation() {
    let app_id = AppId::from_name("durability-test");
    let schema = test_schema();
    let sm_a = SyncManager::new();
    let mgr_a = SchemaManager::new(sm_a, schema.clone(), app_id, "dev", "main").unwrap();
    let mut a = new_test_core(mgr_a, MemoryStorage::new(), NoopScheduler);
    let sm_e = SyncManager::new().with_durability_tier(DurabilityTier::EdgeServer);
    let mgr_e = SchemaManager::new(sm_e, schema, app_id, "dev", "main").unwrap();
    let mut e = new_test_core(mgr_e, MemoryStorage::new(), NoopScheduler);
    let a_client_of_e = ClientId::new();
    let e_server_for_a = ServerId::new();
    e.add_client(a_client_of_e, None);
    e.schema_manager_mut()
        .query_manager_mut()
        .sync_manager_mut()
        .set_client_role(a_client_of_e, ClientRole::Backend);
    a.add_server(e_server_for_a);
    a.immediate_tick();
    e.immediate_tick();
    a.batched_tick();
    e.batched_tick();
    a.sync_sender().take();
    e.sync_sender().take();

    // The "heartbeat backlog": a run of ordinary writes, fully settled.
    for i in 0..8 {
        let (_, _ack) = insert_and_wait_for_batch(
            &mut a,
            "users",
            HashMap::from([
                ("id".to_string(), Value::Uuid(ObjectId::new())),
                ("name".to_string(), Value::Text(format!("heartbeat-{i}"))),
            ]),
            None,
            DurabilityTier::EdgeServer,
        )
        .expect("insert");
        // Full roundtrip: client -> server, server acks -> client.
        for _ in 0..4 {
            a.batched_tick();
            for entry in a.sync_sender().take() {
                if entry.destination == Destination::Server(e_server_for_a) {
                    e.park_sync_message(InboxEntry {
                        source: Source::Client(a_client_of_e),
                        payload: entry.payload,
                    });
                }
            }
            e.batched_tick();
            e.immediate_tick();
            for entry in e.sync_sender().take() {
                if let Destination::Client(cid) = &entry.destination {
                    if *cid == a_client_of_e {
                        a.park_sync_message(InboxEntry {
                            source: Source::Server(e_server_for_a),
                            payload: entry.payload,
                        });
                    }
                }
            }
            a.batched_tick();
            a.immediate_tick();
        }
    }

    // The server-side leftovers a restart sweep will walk.
    {
        use crate::storage::Storage as _;
        let fates = e.storage().scan_authoritative_batch_fates().unwrap();
        let submissions = e.storage().scan_sealed_batch_submissions().unwrap();
        eprintln!(
            "server leftovers: fates={} submissions={}",
            fates.len(),
            submissions.len()
        );
    }

    let scans_before =
        crate::runtime_core::LOCAL_BATCH_FULL_SCANS.load(std::sync::atomic::Ordering::Relaxed);
    let pending = e.pending_batch_ids_needing_reconciliation_for_test();
    let scans = crate::runtime_core::LOCAL_BATCH_FULL_SCANS
        .load(std::sync::atomic::Ordering::Relaxed)
        - scans_before;
    eprintln!(
        "pending={} full_scans={} (settled history must answer from records, not scans)",
        pending.len(),
        scans
    );
    assert_eq!(
        scans, 0,
        "deriving the pending set over settled history cost {scans} full-store scans — \
         the production restart conveyor"
    );
}

/// RED GATE (prod 2026-08-10, the v16.1 follow-up): a POLICY-REJECTED write
/// must not cost a full-store history scan.
///
/// Named by the production store's own fates, not by inference: 934
/// `Insert denied by policy on table users` and 114 `Update denied by USING
/// policy on table users - no old content`. A diverged client — one whose
/// chains predate this store — writes presence heartbeats onto a `users` row
/// whose old content is absent here, so policy denies every one. Each denial
/// records a Rejected fate, and `apply_received_batch_fate` then calls
/// `mark_local_batch_rows_rejected`, which walks `local_batch_rows`: all four
/// point-lookup member sources miss (the rejected row never landed), and the
/// "last-resort" full-store history scan runs. Marking rows rejected is
/// best-effort bookkeeping over rows we already track; with no bookkeeping
/// there is nothing to mark, and rediscovering that by scanning every table's
/// history is unbounded work bought with peer input.
///
/// Measured in production: 264 distinct such batches in 26 minutes, the core
/// pinned at 100%. v16.1's negative cache capped REPEATS only (680 warns over
/// those 264 ids) and could not help the first scan of each new id.
#[test]
fn a_policy_rejected_write_costs_no_full_store_scan() {
    let schema = protected_documents_schema();
    let mut client = create_runtime_with_schema(schema.clone(), "policy-reject-scan-test");
    let mut server = create_runtime_with_schema(schema, "policy-reject-scan-test");

    let client_id = ClientId::new();
    let server_id = ServerId::new();
    // The server knows this connection as mallory; the client writes alice's
    // rows. Locally the write satisfies alice's own policy, and on the server
    // it is denied — the shape the field client hits, where its writes pass at
    // home and are refused here.
    server.add_client(client_id, Some(Session::new("mallory")));
    client.add_server(server_id);
    let alice_session = Session::new("alice");

    // Ordinary history on the server, so a full scan has something to walk.
    for index in 0..12 {
        server
            .insert(
                "documents",
                document_insert_values("resident", &format!("doc-{index}")),
                None,
            )
            .expect("seed history");
    }
    client.batched_tick();
    server.batched_tick();
    server.immediate_tick();
    client.sync_sender().take();
    server.sync_sender().take();

    // Per-runtime scans, measured as a DELTA between two identical halves: the
    // absolute count is not the contract (an unrelated caller may legitimately
    // scan once), but denials themselves must buy none — before the fix this
    // delta was one scan per denial.
    let mut scans_before = server.local_batch_full_scan_count();
    let mut first_half_scans = 0u64;
    let mut rejected_fates = 0usize;

    // Twelve denied writes in two identical halves — the cadence of a client
    // that keeps coming back.
    for round in 0..12 {
        if round == 6 {
            first_half_scans = server.local_batch_full_scan_count() - scans_before;
            scans_before = server.local_batch_full_scan_count();
        }
        client
            .insert(
                "documents",
                document_insert_values("alice", &format!("denied-{round}")),
                Some(&WriteContext::from_session(alice_session.clone())),
            )
            .expect("the write satisfies the client's own policy");
        pump_client_messages_to_server(&mut client, &mut server, server_id, client_id);
        server.batched_tick();
        server.immediate_tick();
        server.batched_tick();
        for entry in server.sync_sender().take() {
            if let SyncPayload::BatchFate { fate } = &entry.payload
                && matches!(fate, crate::batch_fate::BatchFate::Rejected { .. })
            {
                rejected_fates += 1;
            }
        }
    }

    let second_half_scans = server.local_batch_full_scan_count() - scans_before;
    eprintln!(
        "policy-denied writes: rejected_fates={rejected_fates} \
         scans_first_half={first_half_scans} scans_second_half={second_half_scans}"
    );
    assert!(
        rejected_fates >= 12,
        "the scenario must produce a rejected fate per write, else it gates nothing"
    );
    assert_eq!(
        second_half_scans, 0,
        "six more policy-denied writes cost {second_half_scans} full-store history scans — \
         a denial must not buy O(store) work"
    );
}

/// A parentless write must not read the row's whole history.
///
/// `pre_batch_visible_row` prepares the pre-batch content every incoming write
/// is policy-checked against. A batch with ONE parent takes a point-lookup
/// fast path there. A PARENTLESS batch does not: it falls through to
/// `scan_history_row_batches`, which reads every version the row has.
///
/// That is the shape a diverged client sends — the server classifies a write
/// with no parents and no visible old content as an insert — and production
/// 2026-08-10 collected 1442 `Insert denied by policy on table users` from it.
/// The row those landed on had grown to 2541 versions on presence heartbeats,
/// so each attempt read all 2541, with the runtime mutex held; a stack dump
/// caught the server there repeatedly.
///
/// A parentless batch has no ancestry to resolve. The read only decides
/// whether every visible version sits on the incoming branch, and when it does
/// — the single-branch case, which is what production is — the answer is
/// always `None`, so the whole read was spent proving that.
#[test]
fn a_parentless_write_does_not_read_the_whole_row_history() {
    let schema = test_schema();
    let mut client = create_runtime_with_schema(schema.clone(), "parentless-history-cost");
    let mut server = create_runtime_with_schema(schema, "parentless-history-cost");

    let client_id = ClientId::new();
    let server_id = ServerId::new();
    server.add_client(client_id, Some(Session::new("writer")));
    client.add_server(server_id);

    // One row, many versions — a presence row's shape.
    let row_id = ObjectId::new();
    let ((server_row_id, _), _) = insert_and_wait_for_batch(
        &mut server,
        "users",
        HashMap::from([
            ("id".to_string(), Value::Uuid(row_id)),
            ("name".to_string(), Value::Text("v0".to_string())),
        ]),
        None,
        DurabilityTier::Local,
    )
    .expect("seed the row");
    for version in 1..40 {
        server
            .update(
                server_row_id,
                vec![("name".to_string(), Value::Text(format!("v{version}")))],
                None,
            )
            .expect("grow the history");
    }
    server.batched_tick();
    server.immediate_tick();
    client.batched_tick();
    client.sync_sender().take();
    server.sync_sender().take();

    // The diverged shape: a write for that row carrying no parents, on the
    // branch rows actually live on (env + scope + user branch, composed).
    let live_branch = crate::storage::sole_branch_name(server.storage())
        .expect("branch registry readable")
        .expect("the seeded rows registered a branch");
    let parentless = crate::row_histories::StoredRowBatch::new(
        server_row_id,
        live_branch.as_str(),
        Vec::<crate::row_histories::BatchId>::new(),
        encode_row(
            &test_schema()[&TableName::new("users")].columns,
            &user_row_values(server_row_id, "from a diverged client"),
        )
        .expect("row encodes"),
        crate::metadata::RowProvenance::for_insert(server_row_id.to_string(), 9_999),
        HashMap::new(),
        crate::row_histories::RowState::VisibleDirect,
        None,
    );

    server.storage().reset_history_scans();
    server.park_sync_message(InboxEntry {
        source: Source::Client(client_id),
        payload: SyncPayload::RowBatchCreated {
            metadata: None,
            row: parentless,
        },
    });
    server.batched_tick();
    server.immediate_tick();
    let history_reads = server.storage().history_scans();

    eprintln!("whole-history reads for one parentless write: {history_reads}");
    assert_eq!(
        history_reads, 0,
        "one parentless write cost {history_reads} reads of the row's entire history — a \
         batch with no ancestry has nothing to resolve, and a client can send these as fast \
         as it likes"
    );
}

/// An exact replay must be recognised before the work it does not need.
///
/// A peer that retransmits rows it already sent — what every reconnect and
/// every `Missing` answer produces — sends batches the authority holds byte
/// for byte. `process_from_client` recognises that
/// (`matches_replayed_row_batch`) and short-circuits, but the preparation for
/// the policy check that will never run used to happen FIRST: another history
/// row read and decode, or a walk of the whole history for a parentless row.
/// The decision needs only the row the first read already fetched.
///
/// Production 2026-08-10: a core pinned with an empty log, the runtime
/// absorbing a peer's replays — silently, because the short-circuit logs
/// nothing.
#[test]
fn an_exact_replay_costs_one_history_read() {
    let schema = test_schema();
    let mut server = create_runtime_with_schema(schema, "replay-cost");
    let client_id = ClientId::new();
    server.add_client(client_id, Some(Session::new("writer")));

    // A row with history, so a replay has a parent that could be looked up.
    let row_id = ObjectId::new();
    let ((server_row_id, _), _) = insert_and_wait_for_batch(
        &mut server,
        "users",
        HashMap::from([
            ("id".to_string(), Value::Uuid(row_id)),
            ("name".to_string(), Value::Text("v0".to_string())),
        ]),
        None,
        DurabilityTier::Local,
    )
    .expect("seed the row");
    server.batched_tick();
    server.immediate_tick();
    server.sync_sender().take();

    let live_branch = crate::storage::sole_branch_name(server.storage())
        .expect("branch registry readable")
        .expect("the seeded row registered a branch");
    let history = server
        .storage()
        .scan_history_row_batches("users", server_row_id)
        .expect("history readable");
    let root = history.first().expect("the row has a version").clone();

    // What a peer sends: a child of what it holds.
    let child = crate::row_histories::StoredRowBatch::new(
        server_row_id,
        live_branch.as_str(),
        vec![root.batch_id],
        encode_row(
            &test_schema()[&TableName::new("users")].columns,
            &user_row_values(server_row_id, "from the peer"),
        )
        .expect("row encodes"),
        crate::metadata::RowProvenance::for_insert(server_row_id.to_string(), 5_000),
        HashMap::new(),
        crate::row_histories::RowState::VisibleDirect,
        None,
    );
    let deliver = |server: &mut TestCore, row: crate::row_histories::StoredRowBatch| {
        server.park_sync_message(InboxEntry {
            source: Source::Client(client_id),
            payload: SyncPayload::RowBatchCreated {
                metadata: None,
                row,
            },
        });
        server.batched_tick();
        server.immediate_tick();
    };
    deliver(&mut server, child.clone());

    let stored = server
        .storage()
        .load_history_row_batch("users", live_branch.as_str(), server_row_id, child.batch_id)
        .expect("history lookup");
    assert!(
        stored.is_some(),
        "the first send must land, else the second is not a replay and this gates nothing"
    );

    // The same batch again, byte for byte.
    server.storage().reset_history_row_lookups();
    server.storage().reset_history_scans();
    deliver(&mut server, child);
    let lookups = server.storage().history_row_lookups();
    let scans = server.storage().history_scans();

    eprintln!("exact replay cost: history_row_lookups={lookups} history_scans={scans}");
    assert_eq!(
        scans, 0,
        "an exact replay walked the row's whole history {scans} time(s)"
    );
    assert!(
        lookups <= 1,
        "an exact replay cost {lookups} history row reads; one fetches the row the \
         short-circuit is decided from, and everything past that decision prepares a \
         policy check that never runs"
    );
}

/// The registration a real handshake goes through must re-arm the `Missing`
/// answers this authority gave up on.
///
/// `may_tell_client_a_batch_is_missing` stops answering a batch that never
/// completes, which is only a deferral because a reconnect asks again. The
/// SyncManager-level gate proves the hook clears the budget; this one proves the
/// path a socket actually takes reaches the hook — `handle_ws_connection` step 5
/// calls exactly this, once per connection, before any frame is processed. The
/// two are separate claims, and the second is the one that broke twice: first
/// because `add_client` is skipped for an existing client, then because the same
/// user reconnects with an equal session.
#[test]
fn the_registration_a_handshake_uses_re_arms_the_missing_answers() {
    let app_id = AppId::from_name("missing-answer-rearm");
    let sm = SyncManager::new().with_durability_tier(DurabilityTier::Local);
    let mgr = SchemaManager::new(sm, test_schema(), app_id, "dev", "main").unwrap();
    let mut core = new_test_core(mgr, MemoryStorage::new(), NoopScheduler);

    let client_id = ClientId::new();
    let session = crate::query_manager::session::Session::new("alice");
    let batch_id = crate::row_histories::BatchId::new();
    core.ensure_client_with_session_and_catalogue_state_hash(client_id, session.clone(), None);

    let sm = core
        .schema_manager_mut()
        .query_manager_mut()
        .sync_manager_mut();
    sm.missing_answers
        .entry(client_id)
        .or_default()
        .budgets
        .insert(
            batch_id,
            crate::sync_manager::MissingAnswerBudget {
                answers: crate::sync_manager::MAX_MISSING_ANSWERS,
                first_answered_at: 1,
                last_answered_at: 1,
                silenced: true,
            },
        );

    // The same client, the same session: a reconnect on a fresh socket, which is
    // all the server can tell us and all it needs to.
    core.ensure_client_with_session_and_catalogue_state_hash(client_id, session, None);

    let tracked = core
        .schema_manager()
        .query_manager()
        .sync_manager()
        .missing_answers
        .get(&client_id)
        .map(|tracked| tracked.budgets.len())
        .unwrap_or(0);
    assert_eq!(
        tracked, 0,
        "the registration every connection goes through left {tracked} given-up batches in \
         place; silence that survives a reconnect is not a deferral, it is a loss"
    );
}

/// A second branch in the store must not put every parentless write back on the
/// whole-history read.
///
/// The parentless fast path asks `sole_branch_name`, which answers only when the
/// registry holds exactly one branch. That was written as the cheap way to ask
/// "does this row live anywhere but here", and it is exact — for a store that
/// never grew a second branch. Production 2026-08-14: a second branch appeared,
/// `sole_branch_name` went quiet for the whole store, and a diverged client's
/// writes went back to reading a `users` row's entire history — 21,960 refused
/// attempts against one row, one core pinned, the log silent for an hour.
///
/// The question the walk actually answers for a parentless row is per-ROW, not
/// per-store: does THIS row have a version on some other branch? History keys
/// are `<row_id>:<branch>:<batch_id>`, so that is an existence probe per
/// registered branch — a handful of point reads — not a walk of one row's
/// thousands of versions.
#[test]
fn a_parentless_write_does_not_read_the_whole_row_history_when_the_store_has_two_branches() {
    let schema = test_schema();
    let mut server = create_runtime_with_schema(schema, "parentless-two-branch-cost");
    let client_id = ClientId::new();
    server.add_client(client_id, Some(Session::new("writer")));

    // One row, many versions — a presence row's shape.
    let row_id = ObjectId::new();
    let ((server_row_id, _), _) = insert_and_wait_for_batch(
        &mut server,
        "users",
        HashMap::from([
            ("id".to_string(), Value::Uuid(row_id)),
            ("name".to_string(), Value::Text("v0".to_string())),
        ]),
        None,
        DurabilityTier::Local,
    )
    .expect("seed the row");
    for version in 1..40 {
        server
            .update(
                server_row_id,
                vec![("name".to_string(), Value::Text(format!("v{version}")))],
                None,
            )
            .expect("grow the history");
    }
    server.batched_tick();
    server.immediate_tick();

    let live_branch = crate::storage::sole_branch_name(server.storage())
        .expect("branch registry readable")
        .expect("the seeded rows registered a branch");

    // A second branch enters the store — another app scope, an environment, a
    // branch someone opened once. It has nothing to do with the row above.
    let other_branch = BranchName::new("dev-ffffffffffff-main");
    assert_ne!(other_branch.as_str(), live_branch.as_str());
    server
        .storage_mut()
        .resolve_or_alloc_branch_ord(other_branch)
        .expect("register a second branch, as sealing a batch on one does");

    // The precondition, asserted rather than assumed: with two branches
    // registered the store-wide question can no longer be answered, which is
    // exactly the state production was in.
    assert!(
        crate::storage::sole_branch_name(server.storage())
            .expect("branch registry readable")
            .is_none(),
        "the store must hold more than one branch, or this gate measures the single-branch \
         path that is already fast"
    );

    // The diverged shape: a write for the long-history row carrying no parents.
    let parentless = crate::row_histories::StoredRowBatch::new(
        server_row_id,
        live_branch.as_str(),
        Vec::<crate::row_histories::BatchId>::new(),
        encode_row(
            &test_schema()[&TableName::new("users")].columns,
            &user_row_values(server_row_id, "from a diverged client"),
        )
        .expect("row encodes"),
        crate::metadata::RowProvenance::for_insert(server_row_id.to_string(), 9_999),
        HashMap::new(),
        crate::row_histories::RowState::VisibleDirect,
        None,
    );

    server.storage().reset_history_scans();
    server.park_sync_message(InboxEntry {
        source: Source::Client(client_id),
        payload: SyncPayload::RowBatchCreated {
            metadata: None,
            row: parentless,
        },
    });
    server.batched_tick();
    server.immediate_tick();
    let history_reads = server.storage().history_scans();

    eprintln!("whole-history reads with a second branch present: {history_reads}");
    assert_eq!(
        history_reads, 0,
        "one parentless write cost {history_reads} reads of the row's entire history because \
         some unrelated branch exists; the row itself lives on one branch, and a client can \
         send these as fast as it likes"
    );
}

/// The cheap answer must stay the true answer.
///
/// The gate above would also pass if the parentless path simply returned `None`
/// and stopped asking. It must not: when the row genuinely has versions on
/// another branch, the walk is what resolves what was visible before the batch,
/// and skipping it would hand the permission check a wrong `old_content`.
#[test]
fn a_parentless_write_still_resolves_a_row_that_spans_branches() {
    let schema = test_schema();
    let mut server = create_runtime_with_schema(schema, "parentless-spanning-row");
    let client_id = ClientId::new();
    server.add_client(client_id, Some(Session::new("writer")));

    let row_id = ObjectId::new();
    let ((server_row_id, _), _) = insert_and_wait_for_batch(
        &mut server,
        "users",
        HashMap::from([
            ("id".to_string(), Value::Uuid(row_id)),
            ("name".to_string(), Value::Text("v0".to_string())),
        ]),
        None,
        DurabilityTier::Local,
    )
    .expect("seed the row");
    server.batched_tick();
    server.immediate_tick();

    let live_branch = crate::storage::sole_branch_name(server.storage())
        .expect("branch registry readable")
        .expect("the seeded row registered a branch");

    // THIS row also exists on another branch.
    let other_branch = BranchName::new("dev-ffffffffffff-main");
    server
        .storage_mut()
        .resolve_or_alloc_branch_ord(other_branch)
        .expect("register the other branch");
    server
        .storage_mut()
        .append_history_region_rows(
            "users",
            std::slice::from_ref(&crate::row_histories::StoredRowBatch::new(
                server_row_id,
                other_branch.as_str(),
                Vec::<crate::row_histories::BatchId>::new(),
                encode_row(
                    &test_schema()[&TableName::new("users")].columns,
                    &user_row_values(server_row_id, "the same row, elsewhere"),
                )
                .expect("row encodes"),
                crate::metadata::RowProvenance::for_insert(server_row_id.to_string(), 2_000),
                HashMap::new(),
                crate::row_histories::RowState::VisibleDirect,
                None,
            )),
        )
        .expect("the row also has a version on the other branch");

    let parentless = crate::row_histories::StoredRowBatch::new(
        server_row_id,
        live_branch.as_str(),
        Vec::<crate::row_histories::BatchId>::new(),
        encode_row(
            &test_schema()[&TableName::new("users")].columns,
            &user_row_values(server_row_id, "from a diverged client"),
        )
        .expect("row encodes"),
        crate::metadata::RowProvenance::for_insert(server_row_id.to_string(), 9_999),
        HashMap::new(),
        crate::row_histories::RowState::VisibleDirect,
        None,
    );

    server.storage().reset_history_scans();
    server.park_sync_message(InboxEntry {
        source: Source::Client(client_id),
        payload: SyncPayload::RowBatchCreated {
            metadata: None,
            row: parentless,
        },
    });
    server.batched_tick();
    server.immediate_tick();

    assert!(
        server.storage().history_scans() > 0,
        "a row that really does span branches must still be resolved by the walk; answering \
         `None` without asking would feed the permission check a wrong previous value"
    );
}

/// A batch whose parents this authority does not hold must cost nothing to refuse.
///
/// `apply_row_batch` refuses such a batch with `ParentNotFound` — it checks every
/// declared parent by point lookup and the sync path always checks (the
/// known-new escape asserts an empty parent list). Nothing the policy decides
/// can change that outcome. Yet the inbox pays for the policy check's inputs
/// first: `pre_batch_visible_row`, which for a row with two or more parents has
/// no fast path at all and reads the row's ENTIRE history, and then the policy
/// check itself.
///
/// Production 2026-08-14: one diverged client, 21,960 refusals against a single
/// `users` row grown to thousands of versions by presence heartbeats, its
/// declared parent count climbing with every retry (1337 at `parents=1`, then
/// 805, 697, 574, 477, 394 at two through six). Every one of the multi-parent
/// attempts read the whole history, under the runtime mutex, before being
/// refused for a reason two point lookups would have given. One core pinned at
/// 98%, the log silent for an hour.
/// A tombstone whose parents are absent must PARK, never be rejected.
///
/// The refusal-cost shortcut must not touch deletes, and the reason is asymmetric in a way
/// that is invisible from the sync layer. `inbox.rs` classifies on `is_deleted` BEFORE it
/// looks at parents, so a tombstone with parents is `Operation::Delete`. The Update arm of
/// the policy refills empty old content from the row's visible version
/// (`server_queries.rs`, `evaluate_update_permission`); the Delete arm has no such
/// recovery — with no old content it calls `reject_permission_check`, and that PERSISTS a
/// `Rejected` fate. From then on every arrival of that batch id is forced to `Rejected` on
/// sight, so handing over the missing parent afterwards cannot heal it.
///
/// So for a tombstone the difference between reading the row's history and not reading it
/// is the difference between a recoverable park and the permanent destruction of a
/// client's delete. The cheap path buys CPU on updates, which is the traffic that caused
/// the incident; deletes keep paying, and that is the right trade until Delete is given
/// the recovery Update already has.
#[test]
fn a_tombstone_with_absent_parents_parks_instead_of_being_rejected() {
    let schema = owned_documents_schema_v1();
    let mut server = create_runtime_with_schema(schema, "tombstone-absent-parents");
    let client_id = ClientId::new();
    let alice = Session::new("alice");
    server.add_client(client_id, Some(alice.clone()));

    let ((row_id, _), _) = server
        .insert(
            "documents",
            document_insert_values("alice", "doc"),
            Some(&WriteContext::from_session(alice.clone())),
        )
        .expect("seed a row alice owns");
    server.batched_tick();
    server.immediate_tick();

    let live_branch = crate::storage::sole_branch_name(server.storage())
        .expect("branch registry readable")
        .expect("the seeded row registered a branch");

    // The same row on another branch, so the fallback the tombstone depends on is the
    // expensive arm — the shape a schema deployment leaves behind.
    let other_branch = format!("{}-previous-generation", live_branch.as_str());
    crate::test_support::apply_test_row_batch(
        server.storage_mut(),
        row_id,
        &other_branch,
        crate::row_histories::StoredRowBatch::new(
            row_id,
            other_branch.as_str(),
            Vec::new(),
            encode_row(
                &owned_documents_schema_v1()[&TableName::new("documents")].columns,
                &[Value::Text("alice".into()), Value::Text("older".into())],
            )
            .expect("row encodes"),
            crate::metadata::RowProvenance::for_insert(row_id.to_string(), 1),
            HashMap::new(),
            crate::row_histories::RowState::VisibleDirect,
            None,
        ),
    )
    .expect("the older generation's version applies on its own branch");
    server.sync_sender().take();

    let mut tombstone = crate::row_histories::StoredRowBatch::new(
        row_id,
        live_branch.as_str(),
        vec![
            crate::row_histories::BatchId::new(),
            crate::row_histories::BatchId::new(),
        ],
        encode_row(
            &owned_documents_schema_v1()[&TableName::new("documents")].columns,
            &[Value::Text("alice".into()), Value::Text("doc".into())],
        )
        .expect("row encodes"),
        crate::metadata::RowProvenance::for_insert(row_id.to_string(), 9_999),
        HashMap::new(),
        crate::row_histories::RowState::VisibleDirect,
        None,
    );
    tombstone.is_deleted = true;
    let tombstone_batch_id = tombstone.batch_id;

    server.park_sync_message(InboxEntry {
        source: Source::Client(client_id),
        payload: SyncPayload::RowBatchCreated {
            metadata: None,
            row: tombstone,
        },
    });
    server.batched_tick();
    server.immediate_tick();

    let rejected = server.sync_sender().take().into_iter().any(|entry| {
        matches!(
            &entry.payload,
            SyncPayload::BatchFate { fate }
                if fate.batch_id() == tombstone_batch_id && fate.is_rejected()
        )
    });
    assert!(
        !rejected,
        "a delete whose declared parents are absent must park and wait for them, not be \
         rejected. A rejection here is persisted as an authoritative fate, every later \
         arrival of the batch is forced to `Rejected` on sight, and the client's delete is \
         gone for good — supplying the missing parent afterwards cannot bring it back. The \
         apply would have refused this batch with `ParentNotFound` either way; the only \
         thing that changed is whether the policy was handed the old content it needs."
    );
}

/// A parent and its child arriving together must both land, and the child must not be
/// refused for a parent that is in the same delivery.
///
/// This is the one behaviour the cheap refusal predicate can reach. `pre_batch_visible_row`
/// runs while the inbox is drained; the apply that needs the parent runs later in the same
/// settle pass. So when parent and child arrive together, the child's policy inputs are
/// computed at a moment when the parent is genuinely absent from storage — and the walk now
/// stops there rather than reading the row's history. What must NOT follow is a refusal:
/// the parent lands before the apply, the child applies on top of it, and nothing asks the
/// peer to retransmit anything.
///
/// The distinction is why the predicate lives in the walk and not higher up. Hoisting an
/// unappliability check to the top of the inbox would refuse this child outright, turning
/// ordinary ordered delivery into a retransmit round-trip — amplifying the very loop this
/// work exists to quiet.
#[test]
fn a_parent_and_child_in_one_delivery_both_apply() {
    let schema = test_schema();
    let mut server = create_runtime_with_schema(schema, "parent-child-same-delivery");
    let client_id = ClientId::new();
    server.add_client(client_id, Some(Session::new("writer")));
    server.batched_tick();
    server.immediate_tick();
    server.sync_sender().take();

    // One local write first, so the branch registry names the composed branch this
    // authority actually writes on. An empty store has no branch to read.
    let _ = insert_and_wait_for_batch(
        &mut server,
        "users",
        HashMap::from([
            ("id".to_string(), Value::Uuid(ObjectId::new())),
            ("name".to_string(), Value::Text("seed".to_string())),
        ]),
        None,
        DurabilityTier::Local,
    )
    .expect("seed a row so the branch exists");
    server.batched_tick();
    server.immediate_tick();
    server.sync_sender().take();

    let row_id = ObjectId::new();
    let columns = &test_schema()[&TableName::new("users")].columns;
    let branch = crate::storage::sole_branch_name(server.storage())
        .expect("branch registry readable")
        .expect("the seeded row registered a branch")
        .as_str()
        .to_string();

    // TWO parents, because one takes a different route: a single-parent batch is answered
    // by a point read and returns before the ancestor walk, so a one-parent child would
    // never reach the code this guards.
    let first_parent = crate::row_histories::StoredRowBatch::new(
        row_id,
        branch.as_str(),
        Vec::new(),
        encode_row(columns, &user_row_values(row_id, "parent-a")).expect("row encodes"),
        crate::metadata::RowProvenance::for_insert(row_id.to_string(), 1_000),
        HashMap::new(),
        crate::row_histories::RowState::VisibleDirect,
        None,
    );
    let second_parent = crate::row_histories::StoredRowBatch::new(
        row_id,
        branch.as_str(),
        vec![first_parent.batch_id],
        encode_row(columns, &user_row_values(row_id, "parent-b")).expect("row encodes"),
        crate::metadata::RowProvenance::for_insert(row_id.to_string(), 2_000),
        HashMap::new(),
        crate::row_histories::RowState::VisibleDirect,
        None,
    );
    let child = crate::row_histories::StoredRowBatch::new(
        row_id,
        branch.as_str(),
        vec![first_parent.batch_id, second_parent.batch_id],
        encode_row(columns, &user_row_values(row_id, "child")).expect("row encodes"),
        crate::metadata::RowProvenance::for_insert(row_id.to_string(), 3_000),
        HashMap::new(),
        crate::row_histories::RowState::VisibleDirect,
        None,
    );

    // All in ONE drain, child last — the ordering a client produces when it writes
    // several times before the socket flushes.
    for row in [first_parent, second_parent, child] {
        server.park_sync_message(InboxEntry {
            source: Source::Client(client_id),
            payload: SyncPayload::RowBatchCreated {
                // The row is new to this authority, so the table has to ride with it —
                // without it there is no locator and nothing can be applied at all.
                metadata: Some(crate::sync_manager::RowMetadata {
                    id: row_id,
                    metadata: HashMap::from([(
                        crate::metadata::MetadataKey::Table.as_str().to_string(),
                        "users".to_string(),
                    )]),
                }),
                row,
            },
        });
    }
    server.batched_tick();
    server.immediate_tick();

    let asked_for_a_retransmit = server.sync_sender().take().into_iter().any(|entry| {
        matches!(
            &entry.payload,
            SyncPayload::BatchFate { fate } if matches!(fate, crate::batch_fate::BatchFate::Missing { .. })
        )
    });
    assert!(
        !asked_for_a_retransmit,
        "a child whose parent is in the same delivery must not be answered with \
         `BatchFate::Missing`: the parent was never lost, and asking for it converts \
         ordered delivery into a retransmit round-trip"
    );

    let visible = server
        .storage()
        .load_visible_region_row("users", branch.as_str(), row_id)
        .expect("visible row readable")
        .expect("the row must exist after both batches applied");
    let values = decode_row(columns, visible.data.as_ref()).expect("row decodes");
    assert_eq!(
        values[1],
        Value::Text("child".into()),
        "the child must be the visible version — it applied on top of the parent that \
         arrived with it. Skipping the pre-batch read decides only what the policy sees, \
         never whether the write lands."
    );
}

/// The same refusal, for a row that also exists on another branch — the shape a schema
/// deployment leaves behind, and the one the 2026-08-14 fix did not reach.
///
/// `pre_batch_visible_row` walks the batch's declared ancestors by point lookup, and when
/// none resolves it asks one question before giving up: does this row have history on
/// another branch. On `false` it returns cheaply, which is what
/// `a_write_whose_parents_are_missing_costs_no_history_read` above gates — that fixture
/// builds its row through `sole_branch_name`, so the probe is always false there and only
/// the cheap arm is ever exercised. On `true` it still reads the row's ENTIRE history, and
/// `apply_row_batch` then refuses the batch with `ParentNotFound` anyway. The scan buys
/// nothing, and the refusal was decided by the parents alone.
///
/// The gate turns on the BRANCH because that is the question the code asks. In production
/// the two branches are two schema generations — composed branch names carry the schema
/// hash (`<env>-<hash12>-<branch>`), so a deployment splits every touched row across two of
/// them — but nothing on this path reads a generation, and modelling one would test a
/// coincidence rather than the condition.
///
/// MEASURED in production 2026-08-18: 130,634 refusals in three hours, peaking at 49,606
/// per minute, every one of them the SAME row on the older generation's branch with
/// `source="permission_approval"`, its declared parent count climbing 1, 2, 3 … 46 as the
/// sender retried. The settle passes that carried them ran 15.4 s, 25.0 s, 29.3 s and
/// 37.6 s with `subscriptions=0` and `rows_emitted=0`, spending 205–663 history scans over
/// as many as 2,467,082 history entries and 412 MB. One core, held by refusals.
#[test]
fn a_write_whose_parents_are_missing_costs_no_history_read_across_branches() {
    let schema = test_schema();
    let mut server = create_runtime_with_schema(schema, "missing-parent-cost-cross-branch");
    let client_id = ClientId::new();
    server.add_client(client_id, Some(Session::new("writer")));

    // One row, many versions — a presence row's shape, and the reason a whole-history read
    // is not a rounding error.
    let row_id = ObjectId::new();
    let ((server_row_id, _), _) = insert_and_wait_for_batch(
        &mut server,
        "users",
        HashMap::from([
            ("id".to_string(), Value::Uuid(row_id)),
            ("name".to_string(), Value::Text("v0".to_string())),
        ]),
        None,
        DurabilityTier::Local,
    )
    .expect("seed the row");
    for version in 1..40 {
        server
            .update(
                server_row_id,
                vec![("name".to_string(), Value::Text(format!("v{version}")))],
                None,
            )
            .expect("grow the history");
    }
    server.batched_tick();
    server.immediate_tick();

    let live_branch = crate::storage::sole_branch_name(server.storage())
        .expect("branch registry readable")
        .expect("the seeded rows registered a branch");

    // THE ONE DIFFERENCE from the single-branch gate: the same row also has history on
    // another branch, which is what a schema deployment leaves behind and what flips
    // `row_has_history_outside_branch` to true.
    let other_branch = format!("{}-previous-generation", live_branch.as_str());
    crate::test_support::apply_test_row_batch(
        server.storage_mut(),
        server_row_id,
        &other_branch,
        crate::row_histories::StoredRowBatch::new(
            server_row_id,
            other_branch.as_str(),
            Vec::new(),
            encode_row(
                &test_schema()[&TableName::new("users")].columns,
                &user_row_values(server_row_id, "authored under the older generation"),
            )
            .expect("row encodes"),
            crate::metadata::RowProvenance::for_insert(server_row_id.to_string(), 1),
            HashMap::new(),
            crate::row_histories::RowState::VisibleDirect,
            None,
        ),
    )
    .expect("the older generation's version applies on its own branch");
    assert!(
        crate::storage::row_has_history_on_another_branch(
            server.storage(),
            "users",
            server_row_id,
            live_branch.as_str(),
        )
        .expect("branch probe readable"),
        "fixture precondition: the row must have history outside the incoming branch, else \
         this gates the same cheap arm the single-branch test already covers"
    );
    server.sync_sender().take();

    // The diverged shape: two parents, neither of which this authority holds.
    let orphaned = crate::row_histories::StoredRowBatch::new(
        server_row_id,
        live_branch.as_str(),
        vec![
            crate::row_histories::BatchId::new(),
            crate::row_histories::BatchId::new(),
        ],
        encode_row(
            &test_schema()[&TableName::new("users")].columns,
            &user_row_values(server_row_id, "from a diverged client"),
        )
        .expect("row encodes"),
        crate::metadata::RowProvenance::for_insert(server_row_id.to_string(), 9_999),
        HashMap::new(),
        crate::row_histories::RowState::VisibleDirect,
        None,
    );

    server.storage().reset_history_scans();
    server.park_sync_message(InboxEntry {
        source: Source::Client(client_id),
        payload: SyncPayload::RowBatchCreated {
            metadata: None,
            row: orphaned,
        },
    });
    server.batched_tick();
    server.immediate_tick();
    let history_reads = server.storage().history_scans();

    eprintln!("whole-history reads for one unappliable cross-branch write: {history_reads}");
    assert_eq!(
        history_reads, 0,
        "one write with absent parents cost {history_reads} reads of the row's entire \
         history before refusing it. The refusal is decided by the declared parents alone — \
         point lookups that cannot be changed by anything the policy inputs say — so the \
         read is paid for an answer that was already fixed. A row that exists on two \
         branches is not an edge case: a schema deployment splits every touched row across \
         two of them, and a client left on the older one can send these as fast as it likes."
    );
}

#[test]
fn a_write_whose_parents_are_missing_costs_no_history_read() {
    let schema = test_schema();
    let mut server = create_runtime_with_schema(schema, "missing-parent-cost");
    let client_id = ClientId::new();
    server.add_client(client_id, Some(Session::new("writer")));

    // One row, many versions — a presence row's shape.
    let row_id = ObjectId::new();
    let ((server_row_id, _), _) = insert_and_wait_for_batch(
        &mut server,
        "users",
        HashMap::from([
            ("id".to_string(), Value::Uuid(row_id)),
            ("name".to_string(), Value::Text("v0".to_string())),
        ]),
        None,
        DurabilityTier::Local,
    )
    .expect("seed the row");
    for version in 1..40 {
        server
            .update(
                server_row_id,
                vec![("name".to_string(), Value::Text(format!("v{version}")))],
                None,
            )
            .expect("grow the history");
    }
    server.batched_tick();
    server.immediate_tick();
    let live_branch = crate::storage::sole_branch_name(server.storage())
        .expect("branch registry readable")
        .expect("the seeded rows registered a branch");
    let last_seeded_batch_id = server
        .storage()
        .load_visible_region_row("users", live_branch.as_str(), server_row_id)
        .expect("visible row readable")
        .expect("the seeded row is visible")
        .batch_id();
    server.sync_sender().take();

    // The diverged shape: two parents, neither of which this authority holds.
    // Two rather than one because a single parent already takes a point lookup;
    // the population that hurt production declared two and more.
    let orphaned = crate::row_histories::StoredRowBatch::new(
        server_row_id,
        live_branch.as_str(),
        vec![
            crate::row_histories::BatchId::new(),
            crate::row_histories::BatchId::new(),
        ],
        encode_row(
            &test_schema()[&TableName::new("users")].columns,
            &user_row_values(server_row_id, "from a diverged client"),
        )
        .expect("row encodes"),
        crate::metadata::RowProvenance::for_insert(server_row_id.to_string(), 9_999),
        HashMap::new(),
        crate::row_histories::RowState::VisibleDirect,
        None,
    );

    server.storage().reset_history_scans();
    server.park_sync_message(InboxEntry {
        source: Source::Client(client_id),
        payload: SyncPayload::RowBatchCreated {
            metadata: None,
            row: orphaned,
        },
    });
    server.batched_tick();
    server.immediate_tick();
    let history_reads = server.storage().history_scans();

    eprintln!("whole-history reads for one unappliable write: {history_reads}");
    assert_eq!(
        history_reads, 0,
        "one write with absent parents cost {history_reads} reads of the row's entire \
         history before refusing it; the refusal was decided by the parents alone, and a \
         client can send these as fast as it likes"
    );

    // The refusal itself must not change: same outcome, reached cheaper. The
    // seeded row's last version is what stays visible — an unappliable write
    // must not have been applied along the way.
    let visible = server
        .storage()
        .load_visible_region_row("users", live_branch.as_str(), server_row_id)
        .expect("visible row readable")
        .expect("the seeded row stays visible");
    assert_eq!(
        visible.batch_id(),
        last_seeded_batch_id,
        "the refused write must not have become the visible row"
    );
}

/// Rebuild a runtime over an existing store the way every production
/// construction does: rehydrate the schema manager from the persisted
/// catalogue, so the new runtime knows every schema the store has lived under.
/// Mirrors `server/builder.rs` (and the client + jazz-rn constructions).
fn recreate_runtime_rehydrated(schema: Schema, app_name: &str, storage: MemoryStorage) -> TestCore {
    let app_id = AppId::from_name(app_name);
    let mut schema_manager =
        SchemaManager::new(SyncManager::new(), schema, app_id, "dev", "main").unwrap();
    crate::schema_manager::rehydrate_schema_manager_from_catalogue(
        &mut schema_manager,
        &storage,
        app_id,
    )
    .expect("rehydrate from the persisted catalogue");
    let mut core = new_test_core(schema_manager, storage, NoopScheduler);
    core.immediate_tick();
    core
}

/// A row written under the old schema must survive the migration — visible to
/// the new schema's queries and writable from the new schema's runtime.
///
/// Production shape, 2026-08-14: every schema deployment mints a new composed
/// branch (`dev-<hash>-main`), old rows keep their history under the old one,
/// and every client that upgrades queries under the new one. The unit-level
/// lens machinery is well covered; NOTHING covered the crossing itself — the
/// helpers `schema_evolution_v1/v2` below this suite were defined and never
/// used. Meanwhile production spent a week rediscovering the seam one incident
/// at a time.
#[test]
fn a_row_written_under_the_old_schema_is_served_after_the_migration() {
    // Life under the old schema: a server, one row, a little history.
    let mut core = create_runtime_with_storage_and_sync_manager(
        schema_evolution_v1(),
        "schema-crossing",
        MemoryStorage::new(),
        SyncManager::new(),
    );
    let row_id = ObjectId::new();
    let ((server_row_id, _), _) = insert_and_wait_for_batch(
        &mut core,
        "users",
        HashMap::from([
            ("id".to_string(), Value::Uuid(row_id)),
            ("name".to_string(), Value::Text("born under v1".to_string())),
        ]),
        None,
        DurabilityTier::Local,
    )
    .expect("seed the row under v1");
    for version in 1..4 {
        core.update(
            server_row_id,
            vec![(
                "name".to_string(),
                Value::Text(format!("v1 edit {version}")),
            )],
            None,
        )
        .expect("grow v1 history");
    }
    core.batched_tick();
    core.immediate_tick();
    let old_branch = crate::storage::sole_branch_name(core.storage())
        .expect("branch registry readable")
        .expect("v1 registered its branch");

    // The redeploy: same store, new schema, the migration lens published — the
    // engine-level mirror of `migrations push` + restarting the server.
    let storage = core.into_storage();
    // The v1 runtime must have persisted its catalogue, or the rehydrate below
    // has nothing to read and this gate would fail for a reason that is not
    // the defect.
    assert!(
        !storage
            .scan_catalogue_entries()
            .expect("catalogue readable")
            .is_empty(),
        "the v1 runtime persisted no catalogue entries; the crossing cannot even begin"
    );
    let mut core = recreate_runtime_rehydrated(schema_evolution_v2(), "schema-crossing", storage);
    let lens = crate::schema_manager::auto_lens::generate_lens(
        &schema_evolution_v1(),
        &schema_evolution_v2(),
    );
    core.publish_lens(&lens).expect("publish the v1->v2 lens");
    core.immediate_tick();

    // The new runtime's world is a different composed branch than the row's.
    // No branch precondition here on purpose: branches are registered by the
    // first WRITE (seal), and this scenario never writes. The crossing under
    // test is logical — the v2 runtime queries under its own composed branch
    // while the row's history sits under `old_branch` — and the assertion that
    // matters is the served result below. (An earlier precondition here was
    // vacuous in one direction and wrong in the other.)

    // Axis check: the LOCAL query path first — the passing evolution tests read
    // this way. If this sees the row while the subscription below serves
    // nothing, the defect is in serving, not in schema resolution.
    // Both faces of the same store: the local path is the positive control
    // proving the row is resolvable at all, so the serving assertion below can
    // only fail for serving reasons.
    let local = execute_runtime_query(&mut core, Query::new("users"), None);
    assert_eq!(
        local.len(),
        1,
        "the local query path must resolve the v1 row"
    );
    let _ = old_branch;

    // A v2 client subscribes to the table, exactly as an upgraded app does.
    let client_id = ClientId::new();
    core.add_client(client_id, Some(Session::new("reader")));
    core.sync_sender().take();
    let query = core
        .schema_manager_mut()
        .query_manager_mut()
        .query("users")
        .build();
    core.park_sync_message(InboxEntry {
        source: Source::Client(client_id),
        payload: SyncPayload::QuerySubscription {
            query_id: crate::sync_manager::QueryId(1),
            query: Box::new(query),
            session: Some(Session::new("reader")),
            required_tier: None,
            propagation: crate::sync_manager::QueryPropagation::Full,
            policy_context_tables: vec![],
        },
    });
    core.batched_tick();
    core.immediate_tick();

    // Both variants carry the row: `RowBatchCreated` is the direct push,
    // `RowBatchNeeded` the confirm-me delivery every real client answers.
    let served: Vec<_> =
        core.sync_sender()
            .take()
            .into_iter()
            .filter_map(|entry| match entry {
                OutboxEntry {
                    destination: Destination::Client(id),
                    payload:
                        SyncPayload::RowBatchCreated { row, .. }
                        | SyncPayload::RowBatchNeeded { row, .. },
                } if id == client_id && row.row_id == server_row_id => Some(row),
                _ => None,
            })
            .collect();
    assert!(
        !served.is_empty(),
        "a row written under the old schema was not served to the new schema's \
         subscription; every upgraded client sees an empty world"
    );
}

/// A write from the new schema's runtime onto a row whose history lives under
/// the old branch must apply — this is what every upgraded client does within
/// seconds of connecting (presence, tokens, read markers).
#[test]
fn a_write_after_the_migration_applies_onto_old_schema_history() {
    let mut core = create_runtime_with_storage_and_sync_manager(
        schema_evolution_v1(),
        "schema-crossing-write",
        MemoryStorage::new(),
        SyncManager::new(),
    );
    let row_id = ObjectId::new();
    let ((server_row_id, _), _) = insert_and_wait_for_batch(
        &mut core,
        "users",
        HashMap::from([
            ("id".to_string(), Value::Uuid(row_id)),
            ("name".to_string(), Value::Text("born under v1".to_string())),
        ]),
        None,
        DurabilityTier::Local,
    )
    .expect("seed the row under v1");
    core.batched_tick();
    core.immediate_tick();

    let storage = core.into_storage();
    // The v1 runtime must have persisted its catalogue, or the rehydrate below
    // has nothing to read and this gate would fail for a reason that is not
    // the defect.
    assert!(
        !storage
            .scan_catalogue_entries()
            .expect("catalogue readable")
            .is_empty(),
        "the v1 runtime persisted no catalogue entries; the crossing cannot even begin"
    );
    let mut core =
        recreate_runtime_rehydrated(schema_evolution_v2(), "schema-crossing-write", storage);
    let lens = crate::schema_manager::auto_lens::generate_lens(
        &schema_evolution_v1(),
        &schema_evolution_v2(),
    );
    core.publish_lens(&lens).expect("publish the v1->v2 lens");
    core.immediate_tick();

    core.update(
        server_row_id,
        vec![(
            "name".to_string(),
            Value::Text("edited under v2".to_string()),
        )],
        None,
    )
    .expect("a write from the upgraded runtime must apply onto the old history");
    core.batched_tick();
    core.immediate_tick();
}

/// Shared scaffold for the migration-family gates: a store whose row was
/// written under schema v1, rebuilt the way production rebuilds (rehydrated),
/// with the v1→v2 lens published.
fn two_shelf_world(app: &str) -> (TestCore, ObjectId, crate::object::BranchName) {
    let mut core = create_runtime_with_storage_and_sync_manager(
        schema_evolution_v1(),
        app,
        MemoryStorage::new(),
        SyncManager::new(),
    );
    let row_id = ObjectId::new();
    let ((server_row_id, _), _) = insert_and_wait_for_batch(
        &mut core,
        "users",
        HashMap::from([
            ("id".to_string(), Value::Uuid(row_id)),
            ("name".to_string(), Value::Text("Alice".to_string())),
        ]),
        None,
        DurabilityTier::Local,
    )
    .expect("seed the v1 row");
    core.batched_tick();
    core.immediate_tick();
    let old_branch = crate::storage::sole_branch_name(core.storage())
        .expect("branch registry readable")
        .expect("the v1 write registered its branch");
    let storage = core.into_storage();
    let mut core = recreate_runtime_rehydrated(schema_evolution_v2(), app, storage);
    let lens = crate::schema_manager::auto_lens::generate_lens(
        &schema_evolution_v1(),
        &schema_evolution_v2(),
    );
    core.publish_lens(&lens).expect("lens publishes");
    core.immediate_tick();
    (core, server_row_id, old_branch)
}

fn served_rows_for(core: &mut TestCore, client_id: ClientId, row_id: ObjectId) -> usize {
    core.sync_sender()
        .take()
        .into_iter()
        .filter(|entry| {
            matches!(
                entry,
                OutboxEntry {
                    destination: Destination::Client(id),
                    payload: SyncPayload::RowBatchCreated { row, .. }
                        | SyncPayload::RowBatchNeeded { row, .. },
                } if *id == client_id && row.row_id == row_id
            )
        })
        .count()
}

/// An old-schema subscriber must not silently freeze when the row moves shelves.
///
/// Production 2026-08-14: the backend moved to the new schema and its presence
/// writes land under the new branch; the TestFlight app still subscribes under
/// the old one. The user watched last-online freeze on his phone — the write
/// crossed shelves and the old-branch subscription was never told anything.
#[test]
fn an_old_branch_subscriber_hears_about_a_write_that_moves_the_row() {
    let (mut core, server_row_id, old_branch) = two_shelf_world("stale-subscriber");

    // The old-schema client: subscribed under the row's own (old) branch.
    let client_id = ClientId::new();
    core.add_client(client_id, Some(Session::new("reader")));
    core.sync_sender().take();
    let query = core
        .schema_manager_mut()
        .query_manager_mut()
        .query("users")
        .branch(old_branch.as_str())
        .build();
    core.park_sync_message(InboxEntry {
        source: Source::Client(client_id),
        payload: SyncPayload::QuerySubscription {
            query_id: crate::sync_manager::QueryId(7),
            query: Box::new(query),
            session: Some(Session::new("reader")),
            required_tier: None,
            propagation: crate::sync_manager::QueryPropagation::Full,
            policy_context_tables: vec![],
        },
    });
    core.batched_tick();
    core.immediate_tick();
    assert!(
        served_rows_for(&mut core, client_id, server_row_id) > 0,
        "the old-branch subscription must serve the old-shelf row at all, or this gates \
         nothing"
    );

    // A new-schema write moves the row's current to the new shelf.
    core.update(
        server_row_id,
        vec![("name".to_string(), Value::Text("Alice moved".to_string()))],
        None,
    )
    .expect("the v2 write applies");
    core.batched_tick();
    core.immediate_tick();

    assert!(
        served_rows_for(&mut core, client_id, server_row_id) > 0,
        "the row moved shelves and the old-branch subscriber was told nothing; every \
         old-schema client silently freezes on stale data"
    );
}

/// A client write onto an old-shelf row must be classified as the UPDATE it is.
///
/// The policy check's inputs come from `pre_batch_visible_row`. If the previous
/// content cannot be resolved across shelves, the check runs as an INSERT with
/// no old content — evaluating the wrong policy against the wrong shape.
#[test]
#[ignore = "harness gap: the evolution schemas carry no policy bundle, so client writes \
bypass the permission queue entirely; needs a policy-bearing v1/v2 schema pair before \
this can assert anything"]
fn a_write_onto_an_old_shelf_row_is_permission_checked_as_an_update() {
    let (mut core, server_row_id, old_branch) = two_shelf_world("cross-shelf-permission");
    let v2_branch = core.schema_manager().branch_name();
    assert_ne!(v2_branch.as_str(), old_branch.as_str());

    let client_id = ClientId::new();
    core.add_client(client_id, Some(Session::new("writer")));
    core.sync_sender().take();

    // The upgraded client's write: its branch, no resolvable parents — the
    // diverged-but-legitimate shape every reconnect produces.
    let incoming = crate::row_histories::StoredRowBatch::new(
        server_row_id,
        v2_branch.as_str(),
        Vec::<crate::row_histories::BatchId>::new(),
        encode_row(
            &schema_evolution_v2()[&TableName::new("users")].columns,
            &vec![
                Value::Uuid(server_row_id),
                Value::Text("renamed".to_string()),
                Value::Text(String::new()),
            ],
        )
        .expect("row encodes"),
        crate::metadata::RowProvenance::for_insert(server_row_id.to_string(), 9_999),
        HashMap::new(),
        crate::row_histories::RowState::VisibleDirect,
        None,
    );
    core.park_sync_message(InboxEntry {
        source: Source::Client(client_id),
        payload: SyncPayload::RowBatchCreated {
            metadata: None,
            row: incoming,
        },
    });
    core.batched_tick();
    core.immediate_tick();

    let checks = core
        .schema_manager_mut()
        .query_manager_mut()
        .sync_manager_mut()
        .take_pending_permission_checks();
    let Some(check) = checks.iter().find(|check| check.client_id == client_id) else {
        panic!("the write must reach the permission queue, or this gates nothing");
    };
    assert_eq!(
        check.operation,
        crate::query_manager::policy::Operation::Update,
        "a write onto a row that exists on the old shelf was classified as {:?}; the \
         wrong policy evaluates and old content is invisible to it",
        check.operation
    );
    assert!(
        check.old_content.is_some(),
        "the permission check sees no previous content for a row that has 1 visible \
         version on the old shelf"
    );
}

/// Differential gate: whatever the local query path answers, serving must
/// answer the same — for every query shape.
///
/// The incident's defect was exactly a divergence between these two faces, and
/// a bare-table subscription was the shape that caught it. This pins the next
/// shapes before they catch us: a filtered query (which may take the indexed
/// path) must serve the same rows it answers locally.
#[test]
fn serving_agrees_with_the_local_query_for_a_filtered_cross_shelf_query() {
    let (mut core, server_row_id, _old_branch) = two_shelf_world("differential-filtered");

    let local_query = crate::query_manager::query::QueryBuilder::new("users")
        .filter_eq("name", Value::Text("Alice".to_string()))
        .build();
    let local = execute_runtime_query(&mut core, local_query, None);

    let client_id = ClientId::new();
    core.add_client(client_id, Some(Session::new("reader")));
    core.sync_sender().take();
    let query = core
        .schema_manager_mut()
        .query_manager_mut()
        .query("users")
        .filter_eq("name", Value::Text("Alice".to_string()))
        .build();
    core.park_sync_message(InboxEntry {
        source: Source::Client(client_id),
        payload: SyncPayload::QuerySubscription {
            query_id: crate::sync_manager::QueryId(9),
            query: Box::new(query),
            session: Some(Session::new("reader")),
            required_tier: None,
            propagation: crate::sync_manager::QueryPropagation::Full,
            policy_context_tables: vec![],
        },
    });
    core.batched_tick();
    core.immediate_tick();
    let served = served_rows_for(&mut core, client_id, server_row_id);

    assert_eq!(
        (local.len(), served > 0),
        (1, true),
        "local sees {} row(s), serving delivered {}; the two faces of the same query \
         must not disagree",
        local.len(),
        served
    );
}

/// Policy-free structural twins of the owned-documents schemas: the explicit
/// authorization path runs only when the AUTH schema differs from the runtime
/// schema, so the runtime carries these and the policy-bearing pair rides as
/// the authorization schema.
fn owned_documents_structural_v1() -> Schema {
    SchemaBuilder::new()
        .table(
            TableSchema::builder("documents")
                .column("owner_id", ColumnType::Text)
                .column("title", ColumnType::Text),
        )
        .build()
}

fn owned_documents_structural_v2() -> Schema {
    SchemaBuilder::new()
        .table(
            TableSchema::builder("documents")
                .column("owner_id", ColumnType::Text)
                .column("title", ColumnType::Text)
                .column("note", ColumnType::Text),
        )
        .build()
}

/// A USING policy nobody satisfies, for proving the arm actually runs.
fn owned_documents_auth_denying_everyone() -> Schema {
    let policies = TablePolicies::new()
        .with_insert(PolicyExpr::eq_session("owner_id", vec!["user_id".into()]))
        .with_update(
            Some(PolicyExpr::eq_session(
                "owner_id",
                vec!["nonexistent_claim".into()],
            )),
            PolicyExpr::eq_session("owner_id", vec!["nonexistent_claim".into()]),
        );
    SchemaBuilder::new()
        .table(
            TableSchema::builder("documents")
                .column("owner_id", ColumnType::Text)
                .column("title", ColumnType::Text)
                .policies(policies),
        )
        .build()
}

/// v1 of the update-policy world: documents owned via session, with an
/// explicit UPDATE policy whose USING half reads the OLD row.
fn owned_documents_schema_v1() -> Schema {
    let policies = TablePolicies::new()
        .with_select(PolicyExpr::eq_session("owner_id", vec!["user_id".into()]))
        .with_insert(PolicyExpr::eq_session("owner_id", vec!["user_id".into()]))
        .with_update(
            Some(PolicyExpr::eq_session("owner_id", vec!["user_id".into()])),
            PolicyExpr::eq_session("owner_id", vec!["user_id".into()]),
        )
        .with_delete(PolicyExpr::eq_session("owner_id", vec!["user_id".into()]));
    SchemaBuilder::new()
        .table(
            TableSchema::builder("documents")
                .column("owner_id", ColumnType::Text)
                .column("title", ColumnType::Text)
                .policies(policies),
        )
        .build()
}

/// v2 adds one column, which is all a schema needs to mint a new shelf.
fn owned_documents_schema_v2() -> Schema {
    let policies = TablePolicies::new()
        .with_select(PolicyExpr::eq_session("owner_id", vec!["user_id".into()]))
        .with_insert(PolicyExpr::eq_session("owner_id", vec!["user_id".into()]))
        .with_update(
            Some(PolicyExpr::eq_session("owner_id", vec!["user_id".into()])),
            PolicyExpr::eq_session("owner_id", vec!["user_id".into()]),
        )
        .with_delete(PolicyExpr::eq_session("owner_id", vec!["user_id".into()]));
    SchemaBuilder::new()
        .table(
            TableSchema::builder("documents")
                .column("owner_id", ColumnType::Text)
                .column("title", ColumnType::Text)
                .column("note", ColumnType::Text)
                .policies(policies),
        )
        .build()
}

/// An owner's UPDATE must survive its USING policy after the schema moves on.
///
/// The USING half of an update policy evaluates against the row's OLD content.
/// That old content is resolved for the permission check by reads that carry a
/// branch — and after a schema deployment the row's only versions sit under
/// the previous schema's branch. If that resolution cannot cross shelves, the
/// check either misclassifies the write as an INSERT or rejects it with
/// "no old content" — production 2026-08-10 recorded 114 of exactly that
/// rejection string. Either way the owner loses the write for reasons that
/// have nothing to do with policy.
///
/// The same-shelf half runs first as the positive control: it proves the whole
/// observation channel — client write, permission queue, policy evaluation,
/// fate, server state — before the crossing is asserted, so the second half
/// can only fail for crossing reasons.
#[test]
fn an_update_onto_an_old_shelf_row_survives_its_using_policy() {
    // Precondition on the fixture, not the system: the v1/v2 pair must
    // actually MISREAD owner_id when v1 bytes are decoded with the v2
    // descriptor — otherwise "a wrong-shape decode denies the owner" is
    // vacuously satisfied, the transform never has to run, and this gate
    // stops discriminating the moment someone edits the schema pair.
    {
        let v1_bytes = encode_row(
            &owned_documents_schema_v1()[&TableName::new("documents")].columns,
            &vec![
                Value::Text("alice".to_string()),
                Value::Text("draft".to_string()),
            ],
        )
        .expect("v1 row encodes");
        let misread = crate::row_format::decode_row(
            &owned_documents_schema_v2()[&TableName::new("documents")].columns,
            &v1_bytes,
        );
        let still_alice = matches!(
            misread.as_ref().map(|values| values.first()),
            Ok(Some(Value::Text(owner))) if owner == "alice"
        );
        assert!(
            !still_alice,
            "the schema pair no longer discriminates: v1 bytes decode to the right owner \
             under the v2 descriptor, so nothing here would catch a missing transform"
        );
    }

    let v1 = owned_documents_schema_v1();
    let mut client = create_runtime_with_schema(v1.clone(), "cross-shelf-using-policy");
    let mut server = create_runtime_with_schema(v1, "cross-shelf-using-policy");

    let client_id = ClientId::new();
    let server_id = ServerId::new();
    let alice = Session::new("alice");
    server.add_client(client_id, Some(alice.clone()));
    client.add_server(server_id);

    let ((row_id, _), _) = client
        .insert(
            "documents",
            document_insert_values("alice", "draft"),
            Some(&WriteContext::from_session(alice.clone())),
        )
        .expect("the owner's insert satisfies her own policy");
    let ((untouched_row, _), _) = client
        .insert(
            "documents",
            document_insert_values("alice", "second-draft"),
            Some(&WriteContext::from_session(alice.clone())),
        )
        .expect("the second insert satisfies the policy too");
    pump_client_messages_to_server(&mut client, &mut server, server_id, client_id);
    server.batched_tick();
    server.immediate_tick();
    client.batched_tick();
    server.sync_sender().take();
    client.sync_sender().take();

    let rejected_reasons = |server: &mut TestCore| -> Vec<String> {
        server
            .sync_sender()
            .take()
            .into_iter()
            .filter_map(|entry| match entry.payload {
                SyncPayload::BatchFate {
                    fate: crate::batch_fate::BatchFate::Rejected { reason, .. },
                } => Some(reason),
                _ => None,
            })
            .collect()
    };

    // Positive control: the same-shelf update is approved and lands.
    client
        .update(
            row_id,
            vec![("title".to_string(), Value::Text("same-shelf".to_string()))],
            Some(&WriteContext::from_session(alice.clone())),
        )
        .expect("the local update applies");
    pump_client_messages_to_server(&mut client, &mut server, server_id, client_id);
    server.batched_tick();
    server.immediate_tick();
    let rejections = rejected_reasons(&mut server);
    assert!(
        rejections.is_empty(),
        "the same-shelf control update was rejected ({rejections:?}); the observation \
         channel itself is broken and the crossing half below would prove nothing"
    );

    // The deployment: both sides move to v2, rebuilt the way production
    // rebuilds, with the lens published server-side.
    let server_storage = server.into_storage();
    let mut server = recreate_runtime_rehydrated(
        owned_documents_schema_v2(),
        "cross-shelf-using-policy",
        server_storage,
    );
    let lens = crate::schema_manager::auto_lens::generate_lens(
        &owned_documents_schema_v1(),
        &owned_documents_schema_v2(),
    );
    server.publish_lens(&lens).expect("lens publishes");
    let client_storage = client.into_storage();
    let mut client = recreate_runtime_rehydrated(
        owned_documents_schema_v2(),
        "cross-shelf-using-policy",
        client_storage,
    );
    server.add_client(client_id, Some(alice.clone()));
    client.add_server(server_id);
    server.immediate_tick();
    client.immediate_tick();
    // The reconnect handshake in both directions: the client re-uploads its
    // world, and the server's catalogue — including the freshly published
    // lens — reaches the client. Dropping the server outbox here would leave
    // the client without the lens and fail the write for harness reasons.
    pump_client_messages_to_server(&mut client, &mut server, server_id, client_id);
    server.batched_tick();
    server.immediate_tick();
    let mut server_outputs = Vec::new();
    pump_server_messages_to_clients(
        &mut server,
        &mut [ClientForServer {
            core: &mut client,
            server_id,
            client_id,
        }],
        &mut server_outputs,
    );
    client.batched_tick();
    client.immediate_tick();
    server.sync_sender().take();
    client.sync_sender().take();

    // The crossing: the same owner updates the same row from the new schema.
    client
        .update(
            row_id,
            vec![("title".to_string(), Value::Text("cross-shelf".to_string()))],
            Some(&WriteContext::from_session(alice.clone())),
        )
        .expect("the upgraded client's local update applies");
    pump_client_messages_to_server(&mut client, &mut server, server_id, client_id);
    server.batched_tick();
    server.immediate_tick();
    server.batched_tick();
    let rejections = rejected_reasons(&mut server);
    assert!(
        rejections.is_empty(),
        "the owner's update was rejected after the schema moved on: {rejections:?}"
    );

    // Negative control: the fix loosens how old content is DECODED, and must
    // not loosen what the policy DECIDES. A non-owner's cross-shelf update has
    // to stay denied — "always approve" passes every assertion above. Sent
    // straight to the server as a raw batch, the way a client whose local
    // policy engine cannot be trusted would send it.
    let mallory = Session::new("mallory");
    let mallory_client_id = ClientId::new();
    server.add_client(mallory_client_id, Some(mallory));
    server.sync_sender().take();
    let current = server
        .storage()
        .load_visible_region_row(
            "documents",
            server.schema_manager().branch_name().as_str(),
            row_id,
        )
        .expect("visible row readable")
        .expect("the row is visible after the owner's update");
    let stolen = crate::row_histories::StoredRowBatch::new(
        row_id,
        current.branch.as_str(),
        vec![current.batch_id()],
        encode_row(
            &owned_documents_schema_v2()[&TableName::new("documents")].columns,
            &vec![
                Value::Text("alice".to_string()),
                Value::Text("stolen".to_string()),
                Value::Text(String::new()),
            ],
        )
        .expect("row encodes"),
        crate::metadata::RowProvenance::for_insert(row_id.to_string(), 9_999),
        HashMap::new(),
        crate::row_histories::RowState::VisibleDirect,
        None,
    );
    server.park_sync_message(InboxEntry {
        source: Source::Client(mallory_client_id),
        payload: SyncPayload::RowBatchCreated {
            metadata: Some(crate::sync_manager::RowMetadata {
                id: row_id,
                metadata: HashMap::from([(
                    crate::metadata::MetadataKey::Table.as_str().to_string(),
                    "documents".to_string(),
                )]),
            }),
            row: stolen,
        },
    });
    server.batched_tick();
    server.immediate_tick();
    server.batched_tick();
    let rejections = rejected_reasons(&mut server);
    assert!(
        !rejections.is_empty(),
        "a non-owner's cross-shelf update was APPROVED; the decode fix must not have \
         loosened the decision"
    );

    // The Delete arm evaluates OLD content through a different construction
    // site than the update USING arm — the same defect wears a different
    // `None` there. The second row was never touched after the migration, so
    // its only content is v1-shaped: the owner's delete must survive.
    client
        .delete(
            untouched_row,
            Some(&WriteContext::from_session(alice.clone())),
        )
        .expect("the owner's local delete applies");
    pump_client_messages_to_server(&mut client, &mut server, server_id, client_id);
    server.batched_tick();
    server.immediate_tick();
    server.batched_tick();
    let rejections = rejected_reasons(&mut server);
    assert!(
        rejections.is_empty(),
        "the owner's cross-shelf DELETE was rejected: {rejections:?}"
    );
}

/// A client write racing delivery fails cleanly and recovers — at THIS level.
///
/// GREEN, and its green is a boundary marker, not an all-clear: with the
/// memory backend and synchronous ticks the racing write fails with a clean
/// `object not found`, the retry after delivery applies, and unrelated writes
/// are untouched. The simulator reproduction of 2026-08-14 showed MORE than
/// this — an uncaught `missing row-history parent`, then `object not found`
/// persisting, then the account query settling empty — which this harness
/// does not reproduce. Whatever poisons the row in the field therefore lives
/// below this level: the sqlite backend, the jazz-rn actor pipeline, or the
/// binding's error propagation. That is where the next gate belongs.
///
/// Reproduced live on the simulator, 2026-08-14: an app on a freshly wiped
/// store entered the main screen, delivery of its rows was still in flight,
/// and its first writes threw uncaught `missing row-history parent` followed
/// by `object not found` for the SAME object — after which the row was gone
/// locally and the account query settled empty. The store inspected a minute
/// later held the delivered batch; the write had simply raced the persistence.
///
/// Two halves:
/// - positive control: delivery applied, then the write — must succeed (the
///   channel works when nothing races);
/// - the race: the write issued before the delivery is processed — the write
///   itself may fail (the row is genuinely not there yet), but a RETRY after
///   the delivery lands must succeed, and the failed attempt must not have
///   left the object unusable.
#[test]
fn a_client_write_racing_delivery_recovers_once_the_delivery_lands() {
    let schema = test_schema();
    let mut server = create_runtime_with_schema(schema.clone(), "cold-store-race");
    let mut client = create_runtime_with_schema(schema, "cold-store-race");
    let client_id = ClientId::new();
    let server_id = ServerId::new();
    server.add_client(client_id, Some(Session::new("writer")));
    client.add_server(server_id);

    // The server-side row the cold client is about to receive.
    let row_id = ObjectId::new();
    let ((server_row_id, _), _) = insert_and_wait_for_batch(
        &mut server,
        "users",
        HashMap::from([
            ("id".to_string(), Value::Uuid(row_id)),
            ("name".to_string(), Value::Text("v0".to_string())),
        ]),
        None,
        DurabilityTier::Local,
    )
    .expect("seed the row");
    server.batched_tick();
    server.immediate_tick();

    // Deliver it to the cold client, but do NOT process it yet: the message
    // sits in the inbox the way in-flight hydration sits in a real app.
    let delivered = server
        .storage()
        .load_visible_region_row(
            "users",
            crate::storage::sole_branch_name(server.storage())
                .expect("registry readable")
                .expect("branch registered")
                .as_str(),
            server_row_id,
        )
        .expect("visible row readable")
        .expect("the seeded row is visible");
    client.park_sync_message(InboxEntry {
        source: Source::Server(server_id),
        payload: SyncPayload::RowBatchCreated {
            metadata: Some(crate::sync_manager::RowMetadata {
                id: server_row_id,
                metadata: HashMap::from([(
                    crate::metadata::MetadataKey::Table.as_str().to_string(),
                    "users".to_string(),
                )]),
            }),
            row: delivered,
        },
    });

    // The race: the app writes before the delivery is processed. Today this
    // fails — the row is genuinely not applied yet — and that failure is
    // tolerable ONLY if it is clean.
    let raced = client.update(
        server_row_id,
        vec![("name".to_string(), Value::Text("raced".to_string()))],
        None,
    );
    eprintln!("racing write: {:?}", raced.as_ref().map(|_| ()));

    // The delivery lands.
    client.batched_tick();
    client.immediate_tick();

    // Quiescence first: "not yet flushed" and "lost" look identical at the
    // wrong moment.
    client.batched_tick();
    client.immediate_tick();

    // Inverse control: a write to a DIFFERENT, locally-authored row must be
    // fine — separating "cold-store writes are broken" from "this race is
    // broken".
    let ((other_row, _), _) = insert_and_wait_for_batch(
        &mut client,
        "users",
        HashMap::from([
            ("id".to_string(), Value::Uuid(ObjectId::new())),
            ("name".to_string(), Value::Text("local".to_string())),
        ]),
        None,
        DurabilityTier::Local,
    )
    .expect("a local insert on the cold store applies");
    client
        .update(
            other_row,
            vec![("name".to_string(), Value::Text("local2".to_string()))],
            None,
        )
        .expect("an unrelated write must be unaffected by the race");

    // The retry must succeed, and the row must not have been poisoned by the
    // failed attempt.
    client
        .update(
            server_row_id,
            vec![(
                "name".to_string(),
                Value::Text("after delivery".to_string()),
            )],
            None,
        )
        .expect(
            "the retry after the delivery landed must apply; a failed racing write must not \
             leave the object unusable",
        );
}

/// The cold-store race on the backend the field actually runs.
///
/// The engine-level race gate is green on the memory backend; the simulator's
/// poisoning — a failed racing write leaving the row unusable — was observed on
/// SQLITE, through the jazz-rn pipeline. This is the same scenario over
/// `SqliteStorage`: if it stays green, the backend is exonerated too and the
/// remaining suspects are the jazz-rn actor pipeline and the binding's error
/// propagation; if it goes red, the backend is the address.
#[test]
#[cfg(feature = "sqlite")]
fn a_client_write_racing_delivery_recovers_on_sqlite_too() {
    let temp_dir = tempfile::TempDir::new().unwrap();
    let schema = test_schema();
    let mut server = create_runtime_with_schema(schema.clone(), "cold-store-race-sqlite");
    let client_storage: Box<dyn Storage> = Box::new(
        crate::storage::SqliteStorage::open(&temp_dir.path().join("client.sqlite"))
            .expect("open sqlite client store"),
    );
    let app_id = AppId::from_name("cold-store-race-sqlite");
    let client_manager =
        SchemaManager::new(SyncManager::new(), test_schema(), app_id, "dev", "main").unwrap();
    let mut client = new_test_core(client_manager, client_storage, NoopScheduler);
    client.immediate_tick();

    let client_id = ClientId::new();
    let server_id = ServerId::new();
    server.add_client(client_id, Some(Session::new("writer")));
    client.add_server(server_id);

    let row_id = ObjectId::new();
    let ((server_row_id, _), _) = insert_and_wait_for_batch(
        &mut server,
        "users",
        HashMap::from([
            ("id".to_string(), Value::Uuid(row_id)),
            ("name".to_string(), Value::Text("v0".to_string())),
        ]),
        None,
        DurabilityTier::Local,
    )
    .expect("seed the row");
    server.batched_tick();
    server.immediate_tick();

    let delivered = server
        .storage()
        .load_visible_region_row(
            "users",
            crate::storage::sole_branch_name(server.storage())
                .expect("registry readable")
                .expect("branch registered")
                .as_str(),
            server_row_id,
        )
        .expect("visible row readable")
        .expect("the seeded row is visible");
    client.park_sync_message(InboxEntry {
        source: Source::Server(server_id),
        payload: SyncPayload::RowBatchCreated {
            metadata: Some(crate::sync_manager::RowMetadata {
                id: server_row_id,
                metadata: HashMap::from([(
                    crate::metadata::MetadataKey::Table.as_str().to_string(),
                    "users".to_string(),
                )]),
            }),
            row: delivered,
        },
    });

    let raced = client.update(
        server_row_id,
        vec![("name".to_string(), Value::Text("raced".to_string()))],
        None,
    );
    eprintln!("racing write on sqlite: {:?}", raced.as_ref().map(|_| ()));

    client.batched_tick();
    client.immediate_tick();
    client.batched_tick();
    client.immediate_tick();

    client
        .update(
            server_row_id,
            vec![(
                "name".to_string(),
                Value::Text("after delivery".to_string()),
            )],
            None,
        )
        .expect(
            "the retry after the delivery landed must apply on sqlite; a failed racing \
             write must not leave the object unusable",
        );
}

/// Presence proof for the explicit-auth USING arm: with a policy nobody
/// satisfies, even the owner's same-shelf update must be denied. Without this,
/// the twin gate below could go green with the arm never running — which is
/// exactly what its first version did (the auth schema equalled the runtime
/// schema, and the explicit path requires them to DIFFER).
#[test]
fn the_explicit_auth_using_arm_is_reachable() {
    let mut core =
        create_runtime_with_schema(owned_documents_structural_v1(), "local-using-presence");
    core.schema_manager_mut()
        .query_manager_mut()
        .set_authorization_schema(owned_documents_auth_denying_everyone());
    let alice = Session::new("alice");
    let ((row_id, _), _) = core
        .insert(
            "documents",
            document_insert_values("alice", "draft"),
            Some(&WriteContext::from_session(alice.clone())),
        )
        .expect("the permissive insert policy admits the owner");
    core.batched_tick();
    core.immediate_tick();
    let denied = core.update(
        row_id,
        vec![("title".to_string(), Value::Text("nope".to_string()))],
        Some(&WriteContext::from_session(alice)),
    );
    assert!(
        denied.is_err(),
        "a USING policy nobody satisfies approved an update; the explicit-auth arm is not \
         being reached and nothing downstream of it can be tested"
    );
}

/// The client-side twin of the USING decode: the LOCAL write path, explicit
/// authorization schema set, old content on the previous schema's shelf.
///
/// The local update path evaluates `update_using_policy` over the OLD row via
/// an `AuthorizationPolicyRequest` that passes `content_schema_hash: None` —
/// the same branch-keyed decode the server-side fix removed. It runs only when
/// the authorization schema DIFFERS from the runtime schema; the presence gate
/// above proves this harness reaches it.
#[test]
fn a_local_update_onto_an_old_shelf_row_survives_its_using_policy() {
    let mut core =
        create_runtime_with_schema(owned_documents_structural_v1(), "local-cross-shelf-using");
    core.schema_manager_mut()
        .query_manager_mut()
        .set_authorization_schema(owned_documents_schema_v1());
    let alice = Session::new("alice");

    let ((row_id, _), _) = core
        .insert(
            "documents",
            document_insert_values("alice", "draft"),
            Some(&WriteContext::from_session(alice.clone())),
        )
        .expect("the owner's insert satisfies her own policy");
    core.batched_tick();
    core.immediate_tick();

    // Positive control: same shelf, through the explicit-auth path.
    core.update(
        row_id,
        vec![("title".to_string(), Value::Text("same-shelf".to_string()))],
        Some(&WriteContext::from_session(alice.clone())),
    )
    .expect("the same-shelf control update must pass, or the channel proves nothing");

    // The deployment.
    let storage = core.into_storage();
    let mut core = recreate_runtime_rehydrated(
        owned_documents_structural_v2(),
        "local-cross-shelf-using",
        storage,
    );
    core.schema_manager_mut()
        .query_manager_mut()
        .set_authorization_schema(owned_documents_schema_v2());
    let lens = crate::schema_manager::auto_lens::generate_lens(
        &owned_documents_structural_v1(),
        &owned_documents_structural_v2(),
    );
    core.publish_lens(&lens).expect("lens publishes");
    core.immediate_tick();

    // The crossing, locally.
    core.update(
        row_id,
        vec![("title".to_string(), Value::Text("cross-shelf".to_string()))],
        Some(&WriteContext::from_session(alice.clone())),
    )
    .expect("the owner's LOCAL update after the schema moved on must survive its own policy");

    // Negative control: a non-owner's local update must stay denied.
    let denied = core.update(
        row_id,
        vec![("title".to_string(), Value::Text("stolen".to_string()))],
        Some(&WriteContext::from_session(Session::new("mallory"))),
    );
    assert!(
        denied.is_err(),
        "a non-owner's local update was approved; the decode must not loosen the decision"
    );
}

/// A SELECT policy must still see the owner's row after the schema moves on.
///
/// The write-side decode fix (defect 17) covered the UPDATE/DELETE arms; the
/// SELECT arm evaluates each candidate row's content through the same
/// authorization request, whose schema is derived from the branch when no
/// authored hash is supplied. If that decode misreads a v1 row under the v2
/// descriptor, the select policy quietly evaluates to invisible and the row
/// vanishes from the owner's own query — the "empty world" through the policy
/// door rather than the serving door.
///
/// Same discipline: the pre-migration half is the positive control proving
/// the sessioned subscription channel, so the post-migration assertion can
/// only fail for crossing reasons.
#[test]
fn a_select_policy_still_sees_the_owners_row_after_the_migration() {
    let mut core = create_runtime_with_schema(owned_documents_schema_v1(), "cross-shelf-select");
    let alice = Session::new("alice");
    let ((row_id, _), _) = core
        .insert(
            "documents",
            document_insert_values("alice", "draft"),
            Some(&WriteContext::from_session(alice.clone())),
        )
        .expect("the owner's insert satisfies her own policy");
    core.batched_tick();
    core.immediate_tick();

    // Positive control: the sessioned subscription sees the row pre-migration.
    let sub = core
        .schema_manager_mut()
        .query_manager_mut()
        .subscribe_with_session(Query::new("documents"), Some(alice.clone()), None)
        .expect("subscription registers");
    core.immediate_tick();
    let pre: Vec<_> = core
        .schema_manager_mut()
        .query_manager_mut()
        .get_subscription_results(sub)
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    assert_eq!(
        pre,
        vec![row_id],
        "the sessioned subscription must see the row before the migration, or the channel \
         proves nothing"
    );

    // The deployment.
    let storage = core.into_storage();
    let mut core =
        recreate_runtime_rehydrated(owned_documents_schema_v2(), "cross-shelf-select", storage);
    let lens = crate::schema_manager::auto_lens::generate_lens(
        &owned_documents_schema_v1(),
        &owned_documents_schema_v2(),
    );
    core.publish_lens(&lens).expect("lens publishes");
    core.immediate_tick();

    // The owner's own query after the crossing.
    let sub = core
        .schema_manager_mut()
        .query_manager_mut()
        .subscribe_with_session(Query::new("documents"), Some(alice), None)
        .expect("subscription registers");
    core.immediate_tick();
    core.batched_tick();
    core.immediate_tick();
    let post: Vec<_> = core
        .schema_manager_mut()
        .query_manager_mut()
        .get_subscription_results(sub)
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    assert_eq!(
        post,
        vec![row_id],
        "the owner's row vanished from her own query after the schema moved on — the \
         select policy cannot see the old-shelf row"
    );
}

/// Schema pair for the include crossing — the mobile account query's shape:
/// a parent `users` row read together with related rows as include arrays.
/// v2 adds one column to the parent, which is all a schema needs to mint a
/// new shelf.
fn account_include_schema_v1() -> Schema {
    SchemaBuilder::new()
        .table(
            TableSchema::builder("users")
                .column("id", ColumnType::Uuid)
                .column("name", ColumnType::Text),
        )
        .table(
            TableSchema::builder("user_emails")
                .column("email", ColumnType::Text)
                .fk_column("user_id", "users"),
        )
        .table(
            TableSchema::builder("unique_names")
                .column("handle", ColumnType::Text)
                .fk_column("user_id", "users"),
        )
        .build()
}

fn account_include_schema_v2() -> Schema {
    SchemaBuilder::new()
        .table(
            TableSchema::builder("users")
                .column("id", ColumnType::Uuid)
                .column("name", ColumnType::Text)
                .column("bio", ColumnType::Text),
        )
        .table(
            TableSchema::builder("user_emails")
                .column("email", ColumnType::Text)
                .fk_column("user_id", "users"),
        )
        .table(
            TableSchema::builder("unique_names")
                .column("handle", ColumnType::Text)
                .fk_column("user_id", "users"),
        )
        .build()
}

/// Row ids a client was served, both delivery variants, drained once.
fn served_row_ids_for(core: &mut TestCore, client_id: ClientId) -> Vec<ObjectId> {
    core.sync_sender()
        .take()
        .into_iter()
        .filter_map(|entry| match entry {
            OutboxEntry {
                destination: Destination::Client(id),
                payload:
                    SyncPayload::RowBatchCreated { row, .. }
                    | SyncPayload::RowBatchNeeded { row, .. },
            } if id == client_id => Some(row.row_id),
            _ => None,
        })
        .collect()
}

/// An include-shaped subscription must serve the parent AND its relations
/// across the crossing.
///
/// Production 2026-08-15: after the schema deployment the mobile app's account
/// query — `users` by id with backref includes — settles empty for an identity
/// whose rows predate the migration, and the rpc-server's own hydration
/// samples count whole identity tables at zero. Every flat-table crossing gate
/// above is green, so if this gate is red the disease lives in the include
/// path; if it is green the hunt moves off this layer.
///
/// No serving assertion for an include-shaped query exists anywhere else in
/// the crate — migrated or not — so the flat-table gates cannot stand in for
/// this one. The pre-migration subscription runs first as the positive
/// control: it proves the harness observes include serving at all (parent and
/// children as row batches), so the post-migration assertion can only fail
/// for crossing reasons.
#[test]
fn an_include_query_serves_the_parent_and_its_relations_across_the_migration() {
    let mut core = create_runtime_with_storage_and_sync_manager(
        account_include_schema_v1(),
        "include-crossing",
        MemoryStorage::new(),
        SyncManager::new(),
    );
    let user_id = ObjectId::new();
    let ((user_row_id, _), _) = insert_and_wait_for_batch(
        &mut core,
        "users",
        HashMap::from([
            ("id".to_string(), Value::Uuid(user_id)),
            ("name".to_string(), Value::Text("Owner".to_string())),
        ]),
        None,
        DurabilityTier::Local,
    )
    .expect("seed the v1 user");
    let ((email_row_id, _), _) = insert_and_wait_for_batch(
        &mut core,
        "user_emails",
        HashMap::from([
            ("user_id".to_string(), Value::Uuid(user_id)),
            (
                "email".to_string(),
                Value::Text("owner@example.test".to_string()),
            ),
        ]),
        None,
        DurabilityTier::Local,
    )
    .expect("seed the v1 email");
    let ((name_row_id, _), _) = insert_and_wait_for_batch(
        &mut core,
        "unique_names",
        HashMap::from([
            ("user_id".to_string(), Value::Uuid(user_id)),
            ("handle".to_string(), Value::Text("owner".to_string())),
        ]),
        None,
        DurabilityTier::Local,
    )
    .expect("seed the v1 handle");
    core.batched_tick();
    core.immediate_tick();

    let include_query = |core: &mut TestCore| {
        core.schema_manager_mut()
            .query_manager_mut()
            .query("users")
            .with_array("user_emails", |sub| {
                sub.from("user_emails").correlate("user_id", "users.id")
            })
            .with_array("unique_names", |sub| {
                sub.from("unique_names").correlate("user_id", "users.id")
            })
            .build()
    };

    // Positive control: pre-migration, the include subscription must serve
    // all three rows, or the observation channel proves nothing.
    let pre_client = ClientId::new();
    core.add_client(pre_client, Some(Session::new("reader")));
    core.sync_sender().take();
    let query = include_query(&mut core);
    core.park_sync_message(InboxEntry {
        source: Source::Client(pre_client),
        payload: SyncPayload::QuerySubscription {
            query_id: crate::sync_manager::QueryId(21),
            query: Box::new(query),
            session: Some(Session::new("reader")),
            required_tier: None,
            propagation: crate::sync_manager::QueryPropagation::Full,
            policy_context_tables: vec![],
        },
    });
    core.batched_tick();
    core.immediate_tick();
    let pre_served = served_row_ids_for(&mut core, pre_client);
    for (label, id) in [
        ("parent users row", user_row_id),
        ("included user_emails row", email_row_id),
        ("included unique_names row", name_row_id),
    ] {
        assert!(
            pre_served.contains(&id),
            "pre-migration include subscription did not serve the {label}; the harness \
             cannot observe include serving, so the crossing assertion below would be \
             vacuous (served: {pre_served:?})"
        );
    }

    // The deployment: same store, new schema, migration lens published.
    let storage = core.into_storage();
    assert!(
        !storage
            .scan_catalogue_entries()
            .expect("catalogue readable")
            .is_empty(),
        "the v1 runtime persisted no catalogue entries; the crossing cannot even begin"
    );
    let mut core =
        recreate_runtime_rehydrated(account_include_schema_v2(), "include-crossing", storage);
    let lens = crate::schema_manager::auto_lens::generate_lens(
        &account_include_schema_v1(),
        &account_include_schema_v2(),
    );
    core.publish_lens(&lens).expect("lens publishes");
    core.immediate_tick();

    // Axis check: the LOCAL include query first. If this sees the rows while
    // the subscription below serves nothing, the defect is in serving, not in
    // schema resolution.
    let local_query = include_query(&mut core);
    let local = execute_runtime_query(&mut core, local_query, None);
    assert_eq!(
        local.len(),
        1,
        "the local include query must resolve the v1 parent row after the migration"
    );
    let array_lens: Vec<usize> = local[0]
        .1
        .iter()
        .filter_map(|value| value.as_array().map(|rows| rows.len()))
        .collect();
    assert_eq!(
        array_lens,
        vec![1, 1],
        "the local include query must carry both related rows across the migration"
    );

    // A v2 client subscribes with the same include query, exactly as the
    // upgraded app does.
    let post_client = ClientId::new();
    core.add_client(post_client, Some(Session::new("reader")));
    core.sync_sender().take();
    let query = include_query(&mut core);
    core.park_sync_message(InboxEntry {
        source: Source::Client(post_client),
        payload: SyncPayload::QuerySubscription {
            query_id: crate::sync_manager::QueryId(22),
            query: Box::new(query),
            session: Some(Session::new("reader")),
            required_tier: None,
            propagation: crate::sync_manager::QueryPropagation::Full,
            policy_context_tables: vec![],
        },
    });
    core.batched_tick();
    core.immediate_tick();
    let post_served = served_row_ids_for(&mut core, post_client);
    for (label, id) in [
        ("parent users row", user_row_id),
        ("included user_emails row", email_row_id),
        ("included unique_names row", name_row_id),
    ] {
        assert!(
            post_served.contains(&id),
            "the {label} vanished from the include subscription after the schema moved \
             on (served: {post_served:?})"
        );
    }
}

/// The policied variant of the include schema pair: every table owner-read,
/// the mobile `users` shape (allowRead: own row only).
fn policied_include_schema(with_extra_column: bool) -> Schema {
    let owner_read = || {
        TablePolicies::new()
            .with_select(PolicyExpr::eq_session("owner_id", vec!["user_id".into()]))
            .with_insert(PolicyExpr::eq_session("owner_id", vec!["user_id".into()]))
    };
    let users = TableSchema::builder("users")
        .column("id", ColumnType::Uuid)
        .column("owner_id", ColumnType::Text)
        .column("name", ColumnType::Text);
    let users = if with_extra_column {
        users.column("bio", ColumnType::Text)
    } else {
        users
    };
    SchemaBuilder::new()
        .table(users.policies(owner_read()))
        .table(
            TableSchema::builder("user_emails")
                .column("owner_id", ColumnType::Text)
                .column("email", ColumnType::Text)
                .fk_column("user_id", "users")
                .policies(owner_read()),
        )
        .table(
            TableSchema::builder("unique_names")
                .column("owner_id", ColumnType::Text)
                .column("handle", ColumnType::Text)
                .fk_column("user_id", "users")
                .policies(owner_read()),
        )
        .build()
}

/// A POLICIED include subscription must survive the crossing.
///
/// The flat select-policy gate and the policy-free include gate above are both
/// green; this is their cross product — the mobile account query exactly:
/// owner-read tables, a parent row with related rows, all written before the
/// migration, subscribed with the owner's session after it. The mallory
/// subscription is the fixture-discrimination control: if she is served the
/// rows too, the policy never gated anything and green here proves nothing.
#[test]
fn a_policied_include_subscription_survives_the_crossing() {
    let mut core = create_runtime_with_storage_and_sync_manager(
        policied_include_schema(false),
        "policied-include-crossing",
        MemoryStorage::new(),
        SyncManager::new(),
    );
    let alice = Session::new("alice");
    let alice_ctx = WriteContext::from_session(alice.clone());
    let user_id = ObjectId::new();
    let ((user_row_id, _), _) = insert_and_wait_for_batch(
        &mut core,
        "users",
        HashMap::from([
            ("id".to_string(), Value::Uuid(user_id)),
            ("owner_id".to_string(), Value::Text("alice".to_string())),
            ("name".to_string(), Value::Text("Alice".to_string())),
        ]),
        Some(&alice_ctx),
        DurabilityTier::Local,
    )
    .expect("the owner's users insert satisfies her policy");
    let ((email_row_id, _), _) = insert_and_wait_for_batch(
        &mut core,
        "user_emails",
        HashMap::from([
            ("owner_id".to_string(), Value::Text("alice".to_string())),
            ("user_id".to_string(), Value::Uuid(user_id)),
            (
                "email".to_string(),
                Value::Text("alice@example.test".to_string()),
            ),
        ]),
        Some(&alice_ctx),
        DurabilityTier::Local,
    )
    .expect("the owner's email insert satisfies her policy");
    let ((name_row_id, _), _) = insert_and_wait_for_batch(
        &mut core,
        "unique_names",
        HashMap::from([
            ("owner_id".to_string(), Value::Text("alice".to_string())),
            ("user_id".to_string(), Value::Uuid(user_id)),
            ("handle".to_string(), Value::Text("alice".to_string())),
        ]),
        Some(&alice_ctx),
        DurabilityTier::Local,
    )
    .expect("the owner's handle insert satisfies her policy");
    core.batched_tick();
    core.immediate_tick();

    let include_query = |core: &mut TestCore| {
        core.schema_manager_mut()
            .query_manager_mut()
            .query("users")
            .with_array("user_emails", |sub| {
                sub.from("user_emails").correlate("user_id", "users.id")
            })
            .with_array("unique_names", |sub| {
                sub.from("unique_names").correlate("user_id", "users.id")
            })
            .build()
    };
    let subscribe = |core: &mut TestCore, session: &Session, query_id: u64| {
        let client_id = ClientId::new();
        core.add_client(client_id, Some(session.clone()));
        core.sync_sender().take();
        let query = include_query(core);
        core.park_sync_message(InboxEntry {
            source: Source::Client(client_id),
            payload: SyncPayload::QuerySubscription {
                query_id: crate::sync_manager::QueryId(query_id),
                query: Box::new(query),
                session: Some(session.clone()),
                required_tier: None,
                propagation: crate::sync_manager::QueryPropagation::Full,
                policy_context_tables: vec![],
            },
        });
        core.batched_tick();
        core.immediate_tick();
        client_id
    };

    // Positive control: pre-migration the owner is served all three rows.
    let pre_client = subscribe(&mut core, &alice, 31);
    let pre_served = served_row_ids_for(&mut core, pre_client);
    for (label, id) in [
        ("parent users row", user_row_id),
        ("included user_emails row", email_row_id),
        ("included unique_names row", name_row_id),
    ] {
        assert!(
            pre_served.contains(&id),
            "pre-migration the owner's policied include subscription did not serve the \
             {label}; the channel proves nothing (served: {pre_served:?})"
        );
    }

    // The deployment.
    let storage = core.into_storage();
    assert!(
        !storage
            .scan_catalogue_entries()
            .expect("catalogue readable")
            .is_empty(),
        "the v1 runtime persisted no catalogue entries; the crossing cannot even begin"
    );
    let mut core = recreate_runtime_rehydrated(
        policied_include_schema(true),
        "policied-include-crossing",
        storage,
    );
    let lens = crate::schema_manager::auto_lens::generate_lens(
        &policied_include_schema(false),
        &policied_include_schema(true),
    );
    core.publish_lens(&lens).expect("lens publishes");
    core.immediate_tick();

    // Fixture discrimination: mallory subscribes after the crossing and must be
    // served NONE of the rows, or the policy never gated and green below is
    // vacuous.
    let mallory = Session::new("mallory");
    let mallory_client = subscribe(&mut core, &mallory, 32);
    let mallory_served = served_row_ids_for(&mut core, mallory_client);
    assert!(
        !mallory_served.contains(&user_row_id)
            && !mallory_served.contains(&email_row_id)
            && !mallory_served.contains(&name_row_id),
        "mallory was served the owner's rows across the crossing — the policy gates \
         nothing here (served: {mallory_served:?})"
    );

    // The owner after the crossing.
    let post_client = subscribe(&mut core, &alice, 33);
    let post_served = served_row_ids_for(&mut core, post_client);
    for (label, id) in [
        ("parent users row", user_row_id),
        ("included user_emails row", email_row_id),
        ("included unique_names row", name_row_id),
    ] {
        assert!(
            post_served.contains(&id),
            "the owner's {label} vanished from her policied include subscription after \
             the schema moved on (served: {post_served:?})"
        );
    }
}

/// Schema pair for the lensless crossing: the migration only removes an
/// unrelated table; `users` is byte-identical in both versions.
fn users_with_doomed_table_schema() -> Schema {
    SchemaBuilder::new()
        .table(
            TableSchema::builder("users")
                .column("id", ColumnType::Uuid)
                .column("name", ColumnType::Text),
        )
        .table(TableSchema::builder("chat_activities").column("kind", ColumnType::Text))
        .build()
}

fn users_survivor_schema() -> Schema {
    SchemaBuilder::new()
        .table(
            TableSchema::builder("users")
                .column("id", ColumnType::Uuid)
                .column("name", ColumnType::Text),
        )
        .build()
}

/// A row of an untouched table must be served across a migration that has NO
/// published lens.
///
/// Production 2026-08-15: the deployed migration removed one table; nobody
/// pushed a lens covering the crossing for the 40 untouched tables (the only
/// pushed edge covered the removed table, in the other direction). Every
/// crossing gate above publishes a lens first, so none of them could see this
/// starvation. This gate is that missing world: same store, new schema, no
/// `publish_lens` at all — the untouched table's rows must still be resolvable
/// locally and served to a subscriber.
#[test]
fn a_row_of_an_untouched_table_is_served_across_a_lensless_migration() {
    let mut core = create_runtime_with_storage_and_sync_manager(
        users_with_doomed_table_schema(),
        "lensless-crossing",
        MemoryStorage::new(),
        SyncManager::new(),
    );
    let row_id = ObjectId::new();
    let ((server_row_id, _), _) = insert_and_wait_for_batch(
        &mut core,
        "users",
        HashMap::from([
            ("id".to_string(), Value::Uuid(row_id)),
            ("name".to_string(), Value::Text("born before".to_string())),
        ]),
        None,
        DurabilityTier::Local,
    )
    .expect("seed the row under the old schema");
    core.batched_tick();
    core.immediate_tick();

    // The redeploy: same store, new schema, NO lens published.
    let storage = core.into_storage();
    assert!(
        !storage
            .scan_catalogue_entries()
            .expect("catalogue readable")
            .is_empty(),
        "the old runtime persisted no catalogue entries; the crossing cannot even begin"
    );
    let mut core =
        recreate_runtime_rehydrated(users_survivor_schema(), "lensless-crossing", storage);
    core.immediate_tick();

    // Axis check: the local query path resolves the row without any lens.
    let local = execute_runtime_query(&mut core, Query::new("users"), None);
    assert_eq!(
        local.len(),
        1,
        "the local query path must resolve the untouched table's row without a lens"
    );

    // A subscriber on the new schema hears about it too.
    let client_id = ClientId::new();
    core.add_client(client_id, Some(Session::new("reader")));
    core.sync_sender().take();
    let query = core
        .schema_manager_mut()
        .query_manager_mut()
        .query("users")
        .build();
    core.park_sync_message(InboxEntry {
        source: Source::Client(client_id),
        payload: SyncPayload::QuerySubscription {
            query_id: crate::sync_manager::QueryId(41),
            query: Box::new(query),
            session: Some(Session::new("reader")),
            required_tier: None,
            propagation: crate::sync_manager::QueryPropagation::Full,
            policy_context_tables: vec![],
        },
    });
    core.batched_tick();
    core.immediate_tick();
    assert!(
        served_rows_for(&mut core, client_id, server_row_id) > 0,
        "the untouched table's row vanished from serving across a lensless migration"
    );
}

/// A delivered old-branch row must land READABLY in a fresh store.
///
/// The app's measured state, production 2026-08-15: a fresh client store
/// (post-wipe), catalogue carrying both schemas, receives the user's own
/// pre-migration `users` row from the server (confirm-me delivery, receiver
/// confirmed). The history batch, the id index entry and a visible row all
/// land — but the visible row sits in the raw table suffixed with one schema
/// hash while the row locator names the other, so every read path misses it
/// and the account query stays empty forever. This gate is that exact flow:
/// deliver an old-branch row into a fresh runtime that knows both schemas,
/// then require the local query to resolve it.
#[test]
#[ignore = "defect 20, open: red by design until the split-hash ingest fix lands — \
a row delivered BEFORE the catalogue knows its origin schema is placed in the \
current schema's raw table while the locator names the origin, and every read \
misses it. See UPSTREAM-DEFECTS.md entry 20."]
fn a_delivered_old_branch_row_lands_readably_in_a_fresh_store() {
    let old_schema = users_with_doomed_table_schema();
    let old_hash = SchemaHash::compute(&old_schema);
    let mut core = create_runtime_with_storage_and_sync_manager(
        users_survivor_schema(),
        "fresh-client-ingest",
        MemoryStorage::new(),
        SyncManager::new(),
    );
    let old_branch = crate::query_manager::types::ComposedBranchName::new("dev", old_hash, "main")
        .to_branch_name();
    let row_id = ObjectId::new();
    let descriptor = &old_schema
        .get(&TableName::new("users"))
        .expect("users exists in the old schema")
        .columns;
    let delivered = crate::row_histories::StoredRowBatch::new(
        row_id,
        old_branch.as_str(),
        Vec::new(),
        encode_row(
            descriptor,
            &[Value::Uuid(row_id), Value::Text("born before".to_string())],
        )
        .expect("row encodes under the shared descriptor"),
        crate::metadata::RowProvenance::for_insert(row_id.to_string(), 1_000),
        HashMap::new(),
        crate::row_histories::RowState::VisibleDirect,
        None,
    );
    let upstream = ClientId::new();
    core.add_client(upstream, Some(Session::new("upstream")));
    core.sync_sender().take();
    core.park_sync_message(InboxEntry {
        source: Source::Client(upstream),
        payload: SyncPayload::RowBatchCreated {
            metadata: Some(crate::sync_manager::RowMetadata {
                id: row_id,
                metadata: HashMap::from([
                    (
                        crate::metadata::MetadataKey::Table.as_str().to_string(),
                        "users".to_string(),
                    ),
                    (
                        crate::metadata::MetadataKey::OriginSchemaHash
                            .as_str()
                            .to_string(),
                        old_hash.to_string(),
                    ),
                ]),
            }),
            row: delivered,
        },
    });
    core.batched_tick();
    core.immediate_tick();

    // The catalogue learns the old schema AFTER the row landed — the app's
    // measured race. Identity activation makes the old branch queryable.
    core.schema_manager_mut()
        .query_manager_mut()
        .add_live_schema(old_schema.clone());
    core.batched_tick();
    core.immediate_tick();

    let local = execute_runtime_query(&mut core, Query::new("users"), None);
    assert_eq!(
        local.len(),
        1,
        "a delivered old-branch row is unreadable in a fresh store — the ingest \
         placed it where no read path looks"
    );
}

/// The app's chat-list shape across the crossing: memberships with a PARENT
/// ref-include, then the parent touched on the new branch.
///
/// `chat_members.include({ chat })` compiles the singular ref as a correlated
/// subquery from the child to the parent table. Two faces must hold: the
/// whole family served across a lensless identity crossing (the list after an
/// upgrade), and the family SPLIT — the parent's newest version written under
/// the new schema while members stay on the old branch (the second device's
/// touch). The app drops memberships whose `chat` include is empty, so either
/// failure erases the chat from every device's list while the rows sit
/// intact server-side (live incident 2026-08-15).
#[test]
fn a_membership_list_keeps_its_chats_across_the_crossing_and_a_split() {
    let old_schema = || {
        SchemaBuilder::new()
            .table(
                TableSchema::builder("chats")
                    .column("id", ColumnType::Uuid)
                    .column("title", ColumnType::Text),
            )
            .table(
                TableSchema::builder("chat_members")
                    .column("chatId", ColumnType::Uuid)
                    .column("name", ColumnType::Text),
            )
            .table(TableSchema::builder("chat_activities").column("kind", ColumnType::Text))
            .build()
    };
    let new_schema = || {
        SchemaBuilder::new()
            .table(
                TableSchema::builder("chats")
                    .column("id", ColumnType::Uuid)
                    .column("title", ColumnType::Text),
            )
            .table(
                TableSchema::builder("chat_members")
                    .column("chatId", ColumnType::Uuid)
                    .column("name", ColumnType::Text),
            )
            .build()
    };

    let mut core = create_runtime_with_storage_and_sync_manager(
        old_schema(),
        "list-family-crossing",
        MemoryStorage::new(),
        SyncManager::new(),
    );
    let chat_uuid = ObjectId::new();
    let ((chat_row_id, _), _) = insert_and_wait_for_batch(
        &mut core,
        "chats",
        HashMap::from([
            ("id".to_string(), Value::Uuid(chat_uuid)),
            ("title".to_string(), Value::Text("family chat".to_string())),
        ]),
        None,
        DurabilityTier::Local,
    )
    .expect("seed the chat under the old schema");
    for name in ["alice", "bob"] {
        let (_, _confirmation) = insert_and_wait_for_batch(
            &mut core,
            "chat_members",
            HashMap::from([
                ("chatId".to_string(), Value::Uuid(chat_uuid)),
                ("name".to_string(), Value::Text(name.to_string())),
            ]),
            None,
            DurabilityTier::Local,
        )
        .expect("seed a membership under the old schema");
    }
    core.batched_tick();
    core.immediate_tick();

    // The upgrade: same store, new schema, NO lens (identity crossing).
    let storage = core.into_storage();
    let mut core = recreate_runtime_rehydrated(new_schema(), "list-family-crossing", storage);
    core.immediate_tick();

    let list_query = |core: &mut TestCore| {
        core.schema_manager_mut()
            .query_manager_mut()
            .query("chat_members")
            .with_array("chat", |sub| {
                sub.from("chats").correlate("id", "chat_members.chatId")
            })
            .build()
    };
    let chat_of = |values: &Vec<Value>| -> usize {
        values
            .iter()
            .filter_map(|value| value.as_array().map(|rows| rows.len()))
            .next_back()
            .unwrap_or(0)
    };

    let query = list_query(&mut core);
    let baseline = execute_runtime_query(&mut core, query, None);
    assert_eq!(
        baseline.len(),
        2,
        "both memberships must resolve across the crossing"
    );
    assert!(
        baseline.iter().all(|(_, values)| chat_of(values) == 1),
        "every membership must carry its chat across the crossing, or the split \
         assertion below gates nothing"
    );

    // The second device's touch: the chat updated under the NEW schema — the
    // family splits across branches.
    core.update(
        chat_row_id,
        vec![(
            "title".to_string(),
            Value::Text("family chat renamed".to_string()),
        )],
        None,
    )
    .expect("the new-schema update applies");
    core.batched_tick();
    core.immediate_tick();

    let query = list_query(&mut core);
    let after = execute_runtime_query(&mut core, query, None);
    assert_eq!(
        after.len(),
        2,
        "memberships vanished after the parent's cross-branch touch"
    );
    assert!(
        after.iter().all(|(_, values)| chat_of(values) == 1),
        "a membership's parent ref-include came back empty after the family \
         split — the app drops such rows and the chat vanishes from the list"
    );
}

/// Schema pair for the policied crossing: `chats` readable only by its
/// members (an EXISTS policy over `chat_members`), the migration merely
/// dropping an unrelated table so both tables are identity-compatible
/// across the crossing.
fn membership_policied_chat_schema(with_doomed_table: bool) -> Schema {
    use crate::query_manager::policy::{CmpOp, OUTER_ROW_SESSION_PREFIX, PolicyValue};
    let member_can_read_chat = PolicyExpr::Exists {
        table: "chat_members".into(),
        condition: Box::new(PolicyExpr::And(vec![
            PolicyExpr::Cmp {
                column: "chatId".into(),
                op: CmpOp::Eq,
                value: PolicyValue::SessionRef(vec![OUTER_ROW_SESSION_PREFIX.into(), "id".into()]),
            },
            PolicyExpr::eq_session("userId", vec!["user_id".into()]),
        ])),
    };
    let builder = SchemaBuilder::new()
        .table(
            TableSchema::builder("chats")
                .column("id", ColumnType::Uuid)
                .column("title", ColumnType::Text)
                .policies(
                    TablePolicies::new()
                        .with_select(member_can_read_chat)
                        .with_insert(PolicyExpr::True),
                ),
        )
        .table(
            TableSchema::builder("chat_members")
                .column("chatId", ColumnType::Uuid)
                .column("userId", ColumnType::Text)
                .policies(
                    // Row-local select policy on purpose: the membership row
                    // must survive on its own so the gate's final assertion
                    // isolates the chat's cross-branch EXISTS arm.
                    TablePolicies::new()
                        .with_select(PolicyExpr::eq_session("userId", vec!["user_id".into()]))
                        .with_insert(PolicyExpr::True),
                ),
        );
    if with_doomed_table {
        builder
            .table(TableSchema::builder("chat_activities").column("kind", ColumnType::Text))
            .build()
    } else {
        builder.build()
    }
}

/// A POLICIED chat list must survive the crossing for its member.
///
/// The 2026-08-15 incident, fourth pass (defect 23): the app's chat list —
/// `chat_members` with a parent ref-include of `chats`, `chats` readable only
/// by its members — served every row without a session but dropped every
/// OLD-branch row under the owner's session. Reads were taught to union over
/// all queried branches (defects 19/20/22); policy was not: its EXISTS arm
/// scanned only `branches.first()`, so a chat whose supporting membership
/// row lives on the pre-migration branch failed the arm and vanished.
///
/// The mallory subscription is the fixture-discrimination AND the
/// don't-weaken control: she has no membership on ANY branch, so if she is
/// ever served the chat, policy evaluation was broadened past the sanctioned
/// row universe rather than aligned with it.
#[test]
fn a_policied_chat_list_serves_the_old_branch_chat_to_its_member() {
    let mut core = create_runtime_with_storage_and_sync_manager(
        membership_policied_chat_schema(true),
        "policied-chat-crossing",
        MemoryStorage::new(),
        SyncManager::new(),
    );
    let alice = Session::new("alice");
    let alice_ctx = WriteContext::from_session(alice.clone());
    let chat_uuid = ObjectId::new();
    let ((chat_row_id, _), _) = insert_and_wait_for_batch(
        &mut core,
        "chats",
        HashMap::from([
            ("id".to_string(), Value::Uuid(chat_uuid)),
            ("title".to_string(), Value::Text("family chat".to_string())),
        ]),
        Some(&alice_ctx),
        DurabilityTier::Local,
    )
    .expect("the chat inserts under the old schema");
    let ((member_row_id, _), _) = insert_and_wait_for_batch(
        &mut core,
        "chat_members",
        HashMap::from([
            ("chatId".to_string(), Value::Uuid(chat_uuid)),
            ("userId".to_string(), Value::Text("alice".to_string())),
        ]),
        Some(&alice_ctx),
        DurabilityTier::Local,
    )
    .expect("the membership inserts under the old schema");
    core.batched_tick();
    core.immediate_tick();

    let list_query = |core: &mut TestCore| {
        core.schema_manager_mut()
            .query_manager_mut()
            .query("chat_members")
            .with_array("chat", |sub| {
                sub.from("chats").correlate("id", "chat_members.chatId")
            })
            .build()
    };
    let subscribe = |core: &mut TestCore, session: &Session, query_id: u64| {
        let client_id = ClientId::new();
        core.add_client(client_id, Some(session.clone()));
        core.sync_sender().take();
        let query = list_query(core);
        core.park_sync_message(InboxEntry {
            source: Source::Client(client_id),
            payload: SyncPayload::QuerySubscription {
                query_id: crate::sync_manager::QueryId(query_id),
                query: Box::new(query),
                session: Some(session.clone()),
                required_tier: None,
                propagation: crate::sync_manager::QueryPropagation::Full,
                policy_context_tables: vec![],
            },
        });
        core.batched_tick();
        core.immediate_tick();
        client_id
    };

    // Positive control: pre-migration the member is served the chat through
    // her policied list subscription, so the channel observes include serving.
    let pre_client = subscribe(&mut core, &alice, 61);
    let pre_served = served_row_ids_for(&mut core, pre_client);
    assert!(
        pre_served.contains(&member_row_id) && pre_served.contains(&chat_row_id),
        "pre-migration the member's policied list did not serve her chat; the \
         channel proves nothing (served: {pre_served:?})"
    );

    // The deployment: same store, new identity-compatible schema, no lens.
    let storage = core.into_storage();
    assert!(
        !storage
            .scan_catalogue_entries()
            .expect("catalogue readable")
            .is_empty(),
        "the old runtime persisted no catalogue entries; the crossing cannot even begin"
    );
    let mut core = recreate_runtime_rehydrated(
        membership_policied_chat_schema(false),
        "policied-chat-crossing",
        storage,
    );
    core.immediate_tick();

    // Negative control: a session with no membership on ANY branch is served
    // no chat — before and after the fix.
    let mallory = Session::new("mallory");
    let mallory_client = subscribe(&mut core, &mallory, 62);
    let mallory_served = served_row_ids_for(&mut core, mallory_client);
    assert!(
        !mallory_served.contains(&chat_row_id),
        "mallory was served the members-only chat across the crossing — policy \
         evaluation grants beyond the sanctioned branch universe (served: \
         {mallory_served:?})"
    );

    // THE gate: after the crossing the member's session must still be served
    // the old-branch chat — its supporting membership row lives on the OLD
    // branch while the query's first branch is the NEW one.
    let post_client = subscribe(&mut core, &alice, 63);
    let post_served = served_row_ids_for(&mut core, post_client);
    assert!(
        post_served.contains(&member_row_id),
        "the membership row itself vanished across the crossing — the plain \
         read union regressed, this is upstream of the policy defect (served: \
         {post_served:?})"
    );
    assert!(
        post_served.contains(&chat_row_id),
        "defect 23: the member's chat vanished from her policied list after the \
         schema moved on — the SELECT policy's EXISTS arm consulted only the \
         first queried branch and never saw the old-branch membership (served: \
         {post_served:?})"
    );
}

/// Structural twin of the schema pair above: the runtime schema carries NO
/// policies (catalogue schemas are policy-stripped in production); the
/// permissions live in a separate authorization schema, the app's real shape.
fn membership_chat_structural_schema(with_doomed_table: bool) -> Schema {
    let builder = SchemaBuilder::new()
        .table(
            TableSchema::builder("chats")
                .column("id", ColumnType::Uuid)
                .column("title", ColumnType::Text),
        )
        .table(
            TableSchema::builder("chat_members")
                .column("chatId", ColumnType::Uuid)
                .column("userId", ColumnType::Text),
        );
    if with_doomed_table {
        builder
            .table(TableSchema::builder("chat_activities").column("kind", ColumnType::Text))
            .build()
    } else {
        builder.build()
    }
}

/// The permissions head: the same tables WITH policies.
fn membership_chat_auth_schema() -> Schema {
    membership_policied_chat_schema(false)
}

/// The incident's real door, runtime surface: policies come from an EXPLICIT
/// authorization schema (the permissions head), not from the runtime schema —
/// with-session visibility then flows through per-row authorization, which
/// must transform each provenance row into the authorization schema's world
/// (measured live 2026-08-15: no-session 12 memberships, with-session 2 —
/// only the current-branch pair).
///
/// SURFACE CONTROL, not the red witness: this runtime harness ticks
/// `SchemaManager::process`, whose `known_schemas` sync happens to heal the
/// authorization context here, so this gate was green even before the fix.
/// It pins that healed surface against regression. The red→green witness for
/// the defect — the authorization context blind to the old generation on a
/// surface that drives the QueryManager directly — is
/// `manager_tests::policies::an_explicitly_authorized_session_still_sees_an_old_branch_row`.
#[test]
fn an_explicitly_authorized_chat_list_survives_the_crossing_for_its_member() {
    let mut core = create_runtime_with_storage_and_sync_manager(
        membership_chat_structural_schema(true),
        "authorized-chat-crossing",
        MemoryStorage::new(),
        SyncManager::new(),
    );
    core.schema_manager_mut()
        .query_manager_mut()
        .set_authorization_schema(membership_chat_auth_schema());
    let alice = Session::new("alice");
    let alice_ctx = WriteContext::from_session(alice.clone());
    let chat_uuid = ObjectId::new();
    let ((chat_row_id, _), _) = insert_and_wait_for_batch(
        &mut core,
        "chats",
        HashMap::from([
            ("id".to_string(), Value::Uuid(chat_uuid)),
            ("title".to_string(), Value::Text("family chat".to_string())),
        ]),
        Some(&alice_ctx),
        DurabilityTier::Local,
    )
    .expect("the chat inserts under the old schema");
    let ((member_row_id, _), _) = insert_and_wait_for_batch(
        &mut core,
        "chat_members",
        HashMap::from([
            ("chatId".to_string(), Value::Uuid(chat_uuid)),
            ("userId".to_string(), Value::Text("alice".to_string())),
        ]),
        Some(&alice_ctx),
        DurabilityTier::Local,
    )
    .expect("the membership inserts under the old schema");
    core.batched_tick();
    core.immediate_tick();

    let list_query = |core: &mut TestCore| {
        core.schema_manager_mut()
            .query_manager_mut()
            .query("chat_members")
            .with_array("chat", |sub| {
                sub.from("chats").correlate("id", "chat_members.chatId")
            })
            .build()
    };
    let visible_ids = |core: &mut TestCore, session: &Session| -> Vec<ObjectId> {
        let query = list_query(core);
        let sub = core
            .schema_manager_mut()
            .query_manager_mut()
            .subscribe_with_session(query, Some(session.clone()), None)
            .expect("subscription registers");
        core.batched_tick();
        core.immediate_tick();
        let results = core
            .schema_manager_mut()
            .query_manager_mut()
            .get_subscription_results(sub);
        let mut ids = Vec::new();
        for (id, values) in results {
            ids.push(id);
            // Surface every include-resolved chat row id as well.
            for value in values {
                if let Some(rows) = value.as_array() {
                    if !rows.is_empty() {
                        ids.push(chat_row_id);
                    }
                }
            }
        }
        ids
    };

    // Positive control: pre-migration the member's explicitly-authorized list
    // resolves the membership and its chat include.
    let pre = visible_ids(&mut core, &alice);
    assert!(
        pre.contains(&member_row_id) && pre.contains(&chat_row_id),
        "pre-migration the member's authorized list is broken; the channel proves \
         nothing (visible: {pre:?})"
    );

    // The deployment: same store, new identity-compatible schema, no lens.
    let storage = core.into_storage();
    let mut core = recreate_runtime_rehydrated(
        membership_chat_structural_schema(false),
        "authorized-chat-crossing",
        storage,
    );
    core.schema_manager_mut()
        .query_manager_mut()
        .set_authorization_schema(membership_chat_auth_schema());
    core.immediate_tick();

    // Negative control: no membership on ANY branch, nothing served.
    let mallory = visible_ids(&mut core, &Session::new("mallory"));
    assert!(
        !mallory.contains(&member_row_id) && !mallory.contains(&chat_row_id),
        "mallory sees the members-only rows across the crossing (visible: {mallory:?})"
    );

    // THE gate: the member still sees her OLD-branch membership and its chat.
    let post = visible_ids(&mut core, &alice);
    assert!(
        post.contains(&member_row_id),
        "defect 23 (authorization door): the member's own OLD-branch membership \
         vanished under her session — per-row authorization could not transform \
         the old-generation row into the authorization schema's world and failed \
         closed (visible: {post:?})"
    );
    assert!(
        post.contains(&chat_row_id),
        "defect 23 (authorization door): the member's chat include came back \
         empty under her session after the crossing (visible: {post:?})"
    );
}

/// Structural schema for the NESTED-include incident shape (defect 24):
/// `users` referenced from `chat_members.userId`, no policies here — the
/// permissions live in the explicit authorization schema, the app's real door.
fn nested_user_chat_structural_schema(with_doomed_table: bool) -> Schema {
    let builder = SchemaBuilder::new()
        .table(TableSchema::builder("users").column("name", ColumnType::Text))
        .table(
            TableSchema::builder("chats")
                .column("id", ColumnType::Uuid)
                .column("title", ColumnType::Text),
        )
        .table(
            TableSchema::builder("chat_members")
                .column("chatId", ColumnType::Uuid)
                .fk_column("userId", "users"),
        );
    if with_doomed_table {
        builder
            .table(TableSchema::builder("chat_activities").column("kind", ColumnType::Text))
            .build()
    } else {
        builder.build()
    }
}

/// The permissions head for the nested-include incident shape: everything
/// allowRead-always EXCEPT `users`, which is policied — a self arm, plus
/// (when `co_membership_arm`) the app's readReferencing co-membership arm.
fn nested_user_chat_auth_schema(co_membership_arm: bool) -> Schema {
    use crate::query_manager::types::policy_expr;
    let mut users_select = PolicyExpr::eq_session("name", vec!["user_id".into()]);
    if co_membership_arm {
        users_select = PolicyExpr::or(vec![
            users_select,
            policy_expr::allowed_to_read_referencing("chat_members", "userId"),
        ]);
    }
    SchemaBuilder::new()
        .table(
            TableSchema::builder("users")
                .column("name", ColumnType::Text)
                .policies(
                    TablePolicies::new()
                        .with_select(users_select)
                        .with_insert(PolicyExpr::True),
                ),
        )
        .table(
            TableSchema::builder("chats")
                .column("id", ColumnType::Uuid)
                .column("title", ColumnType::Text)
                .policies(
                    TablePolicies::new()
                        .with_select(PolicyExpr::True)
                        .with_insert(PolicyExpr::True),
                ),
        )
        .table(
            TableSchema::builder("chat_members")
                .column("chatId", ColumnType::Uuid)
                .fk_column("userId", "users")
                .policies(
                    TablePolicies::new()
                        .with_select(PolicyExpr::True)
                        .with_insert(PolicyExpr::True),
                ),
        )
        .build()
}

/// The app's chat-list shape, three include levels deep:
/// `chat_members` (mine) → `chat` → its member set → each member's `user`.
fn nested_user_list_query(core: &mut TestCore, my_user_id: ObjectId) -> Query {
    core.schema_manager_mut()
        .query_manager_mut()
        .query("chat_members")
        .filter_eq("userId", Value::Uuid(my_user_id))
        .with_array("chat", |sub| {
            sub.from("chats")
                .correlate("id", "chat_members.chatId")
                .with_array("chat_membersViaChat", |sub| {
                    sub.from("chat_members")
                        .correlate("chatId", "chats.id")
                        .with_array("user", |sub| {
                            sub.from("users").correlate("id", "chat_members.userId")
                        })
                })
        })
        .build()
}

/// For one outer membership row's decoded values, walk chat → members → user
/// and return each member element as (its `userId` column, its user-include
/// length).
fn member_user_includes(values: &[Value]) -> Vec<(Value, usize)> {
    let mut out = Vec::new();
    for value in values {
        let Value::Array(chats) = value else { continue };
        for chat in chats {
            let Value::Row {
                values: chat_values,
                ..
            } = chat
            else {
                continue;
            };
            for chat_value in chat_values {
                let Value::Array(members) = chat_value else {
                    continue;
                };
                for member in members {
                    let Value::Row {
                        values: member_values,
                        ..
                    } = member
                    else {
                        continue;
                    };
                    let user_id = member_values.get(1).cloned().unwrap_or(Value::Null);
                    let users_len = member_values
                        .iter()
                        .filter_map(|value| match value {
                            Value::Array(users) => Some(users.len()),
                            _ => None,
                        })
                        .next_back()
                        .unwrap_or(0);
                    out.push((user_id, users_len));
                }
            }
        }
    }
    out
}

/// Defect 24, face B: a co-member whose `users` row moved to the NEW world
/// while the shared chat's family stayed OLD must not kill the outer
/// membership row of everyone who shares a chat with them.
///
/// The 2026-08-15 incident, fifth pass. The app's list nests three include
/// levels (membership → chat → member set → user); `users` is the first
/// POLICIED table in the shape (self arm OR readReferencing co-membership).
/// One member of two old-world chats had their `users` row touched after the
/// migration, so its tip lives on the NEW branch while every supporting
/// `chat_members` row lives on the OLD branch. Per-row authorization pinned
/// the policy's branch universe to the row's OWN branch, the readReferencing
/// arm never saw the old-world referencing rows, the user was denied — and
/// the output filter killed the whole OUTER membership tuple, erasing the
/// chat from the member's list (measured: 5 memberships without the user
/// level, 3 with it).
#[test]
fn a_co_members_new_world_user_row_keeps_the_old_chat_in_the_list() {
    let mut core = create_runtime_with_storage_and_sync_manager(
        nested_user_chat_structural_schema(true),
        "nested-user-crossing",
        MemoryStorage::new(),
        SyncManager::new(),
    );
    core.schema_manager_mut()
        .query_manager_mut()
        .set_authorization_schema(nested_user_chat_auth_schema(true));
    let alice = Session::new("alice");
    let alice_ctx = WriteContext::from_session(alice.clone());

    let ((alice_user_id, _), _) = insert_and_wait_for_batch(
        &mut core,
        "users",
        HashMap::from([("name".to_string(), Value::Text("alice".to_string()))]),
        Some(&alice_ctx),
        DurabilityTier::Local,
    )
    .expect("alice's user row inserts under the old schema");
    let ((bob_user_id, _), _) = insert_and_wait_for_batch(
        &mut core,
        "users",
        HashMap::from([("name".to_string(), Value::Text("bob".to_string()))]),
        Some(&alice_ctx),
        DurabilityTier::Local,
    )
    .expect("bob's user row inserts under the old schema");
    let chat_uuid = ObjectId::new();
    let (_, _confirmation) = insert_and_wait_for_batch(
        &mut core,
        "chats",
        HashMap::from([
            ("id".to_string(), Value::Uuid(chat_uuid)),
            ("title".to_string(), Value::Text("old chat".to_string())),
        ]),
        Some(&alice_ctx),
        DurabilityTier::Local,
    )
    .expect("the chat inserts under the old schema");
    let ((alice_member_id, _), _) = insert_and_wait_for_batch(
        &mut core,
        "chat_members",
        HashMap::from([
            ("chatId".to_string(), Value::Uuid(chat_uuid)),
            ("userId".to_string(), Value::Uuid(alice_user_id)),
        ]),
        Some(&alice_ctx),
        DurabilityTier::Local,
    )
    .expect("alice's membership inserts under the old schema");
    let (_, _confirmation) = insert_and_wait_for_batch(
        &mut core,
        "chat_members",
        HashMap::from([
            ("chatId".to_string(), Value::Uuid(chat_uuid)),
            ("userId".to_string(), Value::Uuid(bob_user_id)),
        ]),
        Some(&alice_ctx),
        DurabilityTier::Local,
    )
    .expect("bob's membership inserts under the old schema");
    core.batched_tick();
    core.immediate_tick();

    // The deployment: same store, new identity-compatible schema, no lens.
    let storage = core.into_storage();
    let mut core = recreate_runtime_rehydrated(
        nested_user_chat_structural_schema(false),
        "nested-user-crossing",
        storage,
    );
    core.schema_manager_mut()
        .query_manager_mut()
        .set_authorization_schema(nested_user_chat_auth_schema(true));
    core.immediate_tick();

    // The split: bob's user row is touched AFTER the migration — its tip
    // moves to the NEW branch while his membership stays on the OLD one.
    core.update(
        bob_user_id,
        vec![("name".to_string(), Value::Text("bob renamed".to_string()))],
        None,
    )
    .expect("bob's post-migration touch applies");
    core.batched_tick();
    core.immediate_tick();

    let query = nested_user_list_query(&mut core, alice_user_id);
    let sub = core
        .schema_manager_mut()
        .query_manager_mut()
        .subscribe_with_session(query, Some(alice.clone()), None)
        .expect("alice's nested list subscription registers");
    core.batched_tick();
    core.immediate_tick();
    let results = core
        .schema_manager_mut()
        .query_manager_mut()
        .get_subscription_results(sub);

    // THE gate: the outer membership row survives its co-member's user row
    // straddling the schema crossing.
    let outer_ids: Vec<ObjectId> = results.iter().map(|(id, _)| *id).collect();
    assert!(
        outer_ids.contains(&alice_member_id) && results.len() == 1,
        "defect 24: the nested user include's policy verdict killed the OUTER \
         membership row — the chat vanished from the member's list (served \
         outer rows: {outer_ids:?})"
    );

    // Both member elements resolve their user: alice through the self arm,
    // bob through the co-membership arm across the crossing.
    let members = member_user_includes(&results[0].1);
    assert_eq!(
        members.len(),
        2,
        "the chat's member set must serve both memberships (got {members:?})"
    );
    for (user_id, users_len) in &members {
        assert_eq!(
            *users_len, 1,
            "a granted nested user include came back empty for member \
             {user_id:?} (members: {members:?})"
        );
    }
}

/// Defect 24, face A: a nested include element whose row the session may NOT
/// read is clipped at its OWN level — the outer row and every ancestor level
/// are governed by their own tables' policies, not the leaf's.
///
/// No crossing here at all: one world, `users` readable only by the self
/// arm. Bob's user row is legitimately denied to alice — that must empty
/// bob's `user` include element, not erase alice's membership row.
#[test]
fn a_denied_nested_user_include_clips_its_element_not_the_outer_row() {
    let mut core = create_runtime_with_storage_and_sync_manager(
        nested_user_chat_structural_schema(false),
        "nested-user-clip",
        MemoryStorage::new(),
        SyncManager::new(),
    );
    core.schema_manager_mut()
        .query_manager_mut()
        .set_authorization_schema(nested_user_chat_auth_schema(false));
    let alice = Session::new("alice");
    let alice_ctx = WriteContext::from_session(alice.clone());

    let ((alice_user_id, _), _) = insert_and_wait_for_batch(
        &mut core,
        "users",
        HashMap::from([("name".to_string(), Value::Text("alice".to_string()))]),
        Some(&alice_ctx),
        DurabilityTier::Local,
    )
    .expect("alice's user row inserts");
    let ((bob_user_id, _), _) = insert_and_wait_for_batch(
        &mut core,
        "users",
        HashMap::from([("name".to_string(), Value::Text("bob".to_string()))]),
        Some(&alice_ctx),
        DurabilityTier::Local,
    )
    .expect("bob's user row inserts");
    let chat_uuid = ObjectId::new();
    let (_, _confirmation) = insert_and_wait_for_batch(
        &mut core,
        "chats",
        HashMap::from([
            ("id".to_string(), Value::Uuid(chat_uuid)),
            ("title".to_string(), Value::Text("shared chat".to_string())),
        ]),
        Some(&alice_ctx),
        DurabilityTier::Local,
    )
    .expect("the chat inserts");
    let ((alice_member_id, _), _) = insert_and_wait_for_batch(
        &mut core,
        "chat_members",
        HashMap::from([
            ("chatId".to_string(), Value::Uuid(chat_uuid)),
            ("userId".to_string(), Value::Uuid(alice_user_id)),
        ]),
        Some(&alice_ctx),
        DurabilityTier::Local,
    )
    .expect("alice's membership inserts");
    let (_, _confirmation) = insert_and_wait_for_batch(
        &mut core,
        "chat_members",
        HashMap::from([
            ("chatId".to_string(), Value::Uuid(chat_uuid)),
            ("userId".to_string(), Value::Uuid(bob_user_id)),
        ]),
        Some(&alice_ctx),
        DurabilityTier::Local,
    )
    .expect("bob's membership inserts");
    core.batched_tick();
    core.immediate_tick();

    let query = nested_user_list_query(&mut core, alice_user_id);
    let sub = core
        .schema_manager_mut()
        .query_manager_mut()
        .subscribe_with_session(query, Some(alice.clone()), None)
        .expect("alice's nested list subscription registers");
    core.batched_tick();
    core.immediate_tick();
    let results = core
        .schema_manager_mut()
        .query_manager_mut()
        .get_subscription_results(sub);

    // THE gate: the denied leaf clips its own element only.
    let outer_ids: Vec<ObjectId> = results.iter().map(|(id, _)| *id).collect();
    assert!(
        outer_ids.contains(&alice_member_id) && results.len() == 1,
        "defect 24 (clip face): a legitimately denied nested user element \
         erased the OUTER membership row instead of clipping at its own level \
         (served outer rows: {outer_ids:?})"
    );

    let members = member_user_includes(&results[0].1);
    assert_eq!(
        members.len(),
        2,
        "the always-readable member set must survive the leaf denial (got {members:?})"
    );
    let user_len_for = |user_id: ObjectId| -> Option<usize> {
        members
            .iter()
            .find(|(id, _)| *id == Value::Uuid(user_id))
            .map(|(_, len)| *len)
    };
    assert_eq!(
        user_len_for(alice_user_id),
        Some(1),
        "alice's own user element must resolve through the self arm (members: {members:?})"
    );
    assert_eq!(
        user_len_for(bob_user_id),
        Some(0),
        "bob's denied user element must be EMPTY, not populated and not \
         fatal to any ancestor (members: {members:?})"
    );

    // The denial must actually clip the DATA: bob's row content may not
    // appear anywhere in the served tree.
    let serialized = format!("{results:?}");
    assert!(
        !serialized.contains("bob"),
        "the denied user's content leaked into the served result: {serialized}"
    );
}

/// Structural presence schema for the defect-25 gate: policy-stripped, the
/// way the app's runtime schema arrives; enforcement comes from the
/// permissions head below.
fn presence_users_structural_schema(with_doomed_table: bool) -> Schema {
    let builder = SchemaBuilder::new().table(
        TableSchema::builder("users")
            .column("name", ColumnType::Text)
            .column("onlineTimeUpdatedAtMs", ColumnType::Timestamp),
    );
    if with_doomed_table {
        builder
            .table(TableSchema::builder("user_emails").column("email", ColumnType::Text))
            .build()
    } else {
        builder.build()
    }
}

/// The permissions head for the defect-25 gate: the app's `users` shape —
/// UPDATE gated by whereOld AND whereNew (`USING` + `WITH CHECK`), both
/// requiring the row to be the session user's own.
fn presence_users_auth_schema() -> Schema {
    SchemaBuilder::new()
        .table(
            TableSchema::builder("users")
                .column("name", ColumnType::Text)
                .column("onlineTimeUpdatedAtMs", ColumnType::Timestamp)
                .policies(
                    TablePolicies::new()
                        .with_select(PolicyExpr::True)
                        .with_insert(PolicyExpr::True)
                        .with_update(
                            Some(PolicyExpr::eq_session("name", vec!["user_id".into()])),
                            PolicyExpr::eq_session("name", vec!["user_id".into()]),
                        ),
                ),
        )
        .build()
}

/// Defect-25 gate: a session UPDATE to the user's OWN row whose family is
/// SPLIT across the schema crossing, under a permissions head with
/// whereOld + whereNew, MUST apply — and land on the writer's branch.
///
/// The whereOld arm forces write authorization to read the row's OLD
/// content; across the crossing that content's tip lives on the
/// pre-migration branch (first update) and then the family is split
/// (second update). Investigated as the suspected mechanism of the
/// 2026-08-15 frozen-presence incident; the engine handled this shape at
/// every probed level (the incident traced to the store file itself —
/// see UPSTREAM-DEFECTS.md entry 25), and this pins the behavior against
/// an actual regression.
///
/// Controls, don't-weaken: whereOld still denies the session another
/// user's row; whereNew still denies rewriting the row to another owner;
/// a row born in the NEW world updates the same way (v16.12-era shape).
#[test]
fn an_owner_updates_their_split_family_row_under_the_permissions_head() {
    let mut core = create_runtime_with_storage_and_sync_manager(
        presence_users_structural_schema(true),
        "defect25-split-update",
        MemoryStorage::new(),
        SyncManager::new(),
    );
    core.schema_manager_mut()
        .query_manager_mut()
        .set_authorization_schema(presence_users_auth_schema());
    let alice = Session::new("alice");
    let alice_ctx = WriteContext::from_session(alice.clone());
    let bob_ctx = WriteContext::from_session(Session::new("bob"));

    // Rows born in the OLD world.
    let ((alice_row, _), _) = insert_and_wait_for_batch(
        &mut core,
        "users",
        HashMap::from([
            ("name".to_string(), Value::Text("alice".to_string())),
            ("onlineTimeUpdatedAtMs".to_string(), Value::Timestamp(1_000)),
        ]),
        Some(&alice_ctx),
        DurabilityTier::Local,
    )
    .expect("alice's row inserts under the old schema");
    let ((bob_row, _), _) = insert_and_wait_for_batch(
        &mut core,
        "users",
        HashMap::from([
            ("name".to_string(), Value::Text("bob".to_string())),
            ("onlineTimeUpdatedAtMs".to_string(), Value::Timestamp(1_000)),
        ]),
        Some(&bob_ctx),
        DurabilityTier::Local,
    )
    .expect("bob's row inserts under the old schema");
    core.batched_tick();
    core.immediate_tick();

    // The crossing: same store, identity-compatible schema minus a table.
    let storage = core.into_storage();
    let mut core = recreate_runtime_rehydrated(
        presence_users_structural_schema(false),
        "defect25-split-update",
        storage,
    );
    core.schema_manager_mut()
        .query_manager_mut()
        .set_authorization_schema(presence_users_auth_schema());
    core.immediate_tick();

    // First post-crossing heartbeat: the row's tip still lives on the OLD
    // branch, so whereOld must read across the crossing.
    core.update(
        alice_row,
        vec![("onlineTimeUpdatedAtMs".to_string(), Value::Timestamp(2_000))],
        Some(&alice_ctx),
    )
    .expect("the owner's update across the crossing applies");
    core.batched_tick();
    core.immediate_tick();

    // Second heartbeat: the family is now SPLIT — old-branch history plus a
    // new-branch tip. The incident shape.
    core.update(
        alice_row,
        vec![("onlineTimeUpdatedAtMs".to_string(), Value::Timestamp(3_000))],
        Some(&alice_ctx),
    )
    .expect("the owner's update to the split family applies");
    core.batched_tick();
    core.immediate_tick();

    // The write is visible fresh under the owner's session.
    let online_index = column_index(
        &presence_users_structural_schema(false),
        "users",
        "onlineTimeUpdatedAtMs",
    );
    let alice_query = core
        .schema_manager_mut()
        .query_manager_mut()
        .query("users")
        .filter_eq("name", Value::Text("alice".to_string()))
        .build();
    let served = execute_runtime_query(&mut core, alice_query, Some(alice.clone()));
    assert_eq!(
        served.len(),
        1,
        "alice reads exactly her row (served: {served:?})"
    );
    assert_eq!(
        served[0].1[online_index],
        Value::Timestamp(3_000),
        "the split-family update's content must be what reads serve"
    );

    // Don't-weaken: whereOld still denies another user's row.
    let foreign = core.update(
        bob_row,
        vec![("onlineTimeUpdatedAtMs".to_string(), Value::Timestamp(4_000))],
        Some(&alice_ctx),
    );
    assert!(
        foreign.is_err(),
        "alice updated bob's row — whereOld no longer bites: {foreign:?}"
    );

    // Don't-weaken: whereNew still denies rewriting the row to another owner.
    let stolen = core.update(
        alice_row,
        vec![("name".to_string(), Value::Text("mallory".to_string()))],
        Some(&alice_ctx),
    );
    assert!(
        stolen.is_err(),
        "alice renamed her row to another owner — whereNew no longer bites: {stolen:?}"
    );

    // v16.12-era control: a row born in the NEW world updates the same way.
    let carol_ctx = WriteContext::from_session(Session::new("carol"));
    let ((carol_row, _), _) = insert_and_wait_for_batch(
        &mut core,
        "users",
        HashMap::from([
            ("name".to_string(), Value::Text("carol".to_string())),
            ("onlineTimeUpdatedAtMs".to_string(), Value::Timestamp(1_000)),
        ]),
        Some(&carol_ctx),
        DurabilityTier::Local,
    )
    .expect("carol's row inserts under the new schema");
    core.update(
        carol_row,
        vec![("onlineTimeUpdatedAtMs".to_string(), Value::Timestamp(5_000))],
        Some(&carol_ctx),
    )
    .expect("a same-world update still applies");
}

/// Gate G5 (linsa-v18, item 1): a one-shot read whose caller gave up must be
/// cancellable, and cancelling must release the engine's subscription and tell the
/// server, so an abandoned read stops costing settle passes on both engines.
///
/// Internal-level test, on purpose: the observable is "the engine no longer holds a
/// subscription for a read the consumer abandoned", which no public client API exposes
/// (there is no live-subscription count on `JazzClient`, and the black-box variant would
/// need a server with blocked messages plus a Rust-level query deadline that does not
/// exist).
#[test]
fn cancelled_one_shot_query_releases_its_subscription_and_unsubscribes_upstream() {
    use crate::query_manager::manager::LocalUpdates;
    use crate::sync_manager::{QueryId, QueryPropagation};
    use std::pin::Pin;
    use std::task::{Context, Poll};
    let app_id = AppId::from_name("cancel-one-shot");
    let schema_manager =
        SchemaManager::new(SyncManager::new(), test_schema(), app_id, "dev", "main").unwrap();
    let mut core = new_test_core(schema_manager, MemoryStorage::new(), NoopScheduler);
    let server_id = ServerId::new();
    core.add_server(server_id);
    core.immediate_tick();
    core.batched_tick();
    core.sync_sender().take();
    let baseline = core.schema_manager().query_manager().subscription_count();
    // A tiered read: it resolves only on the server's QuerySettled, which never comes
    // here — exactly the read a facade times out on.
    let (handle, mut future) = core
        .query_with_local_batch_tracked(
            Query::new("users"),
            None,
            ReadDurabilityOptions {
                tier: Some(DurabilityTier::EdgeServer),
                local_updates: LocalUpdates::Immediate,
            },
            QueryPropagation::Full,
            None,
        )
        .expect("query setup");
    core.batched_tick();
    let registered = core.sync_sender().take();
    assert_eq!(
        count_query_subscriptions_to_server(&registered, server_id),
        1,
        "the one-shot read registers a subscription at the server"
    );
    let registered_query_id = registered
        .iter()
        .find_map(|entry| match &entry.payload {
            SyncPayload::QuerySubscription { query_id, .. } => Some(*query_id),
            _ => None,
        })
        .expect("registration carries the query id");
    let waker = noop_waker();
    let mut cx = Context::from_waker(&waker);
    assert!(
        matches!(Pin::new(&mut future).poll(&mut cx), Poll::Pending),
        "without the server's settle the read is still pending"
    );
    assert_eq!(
        core.schema_manager().query_manager().subscription_count(),
        baseline + 1
    );
    assert_eq!(core.pending_one_shot_query_count(), 1);
    // The caller's deadline fires.
    let released = core.cancel_one_shot_query(handle);
    assert_eq!(
        core.schema_manager().query_manager().subscription_count(),
        baseline,
        "cancelling releases the engine's subscription for the abandoned read"
    );
    assert!(
        released,
        "a pending one-shot query reports that it was released"
    );
    assert_eq!(core.pending_one_shot_query_count(), 0);
    match Pin::new(&mut future).poll(&mut cx) {
        Poll::Ready(Err(RuntimeError::QueryCancelled)) => {}
        other => panic!("the abandoned read must resolve as cancelled, got {other:?}"),
    }
    core.batched_tick();
    let after = core.sync_sender().take();
    let unsubscribed: Vec<QueryId> = after
        .iter()
        .filter(|entry| entry.destination == Destination::Server(server_id))
        .filter_map(|entry| match &entry.payload {
            SyncPayload::QueryUnsubscription { query_id } => Some(*query_id),
            _ => None,
        })
        .collect();
    assert_eq!(
        unsubscribed,
        vec![registered_query_id],
        "the server is told to drop exactly that registration, once: {after:?}"
    );
    assert!(
        !core.cancel_one_shot_query(handle),
        "a second cancel finds nothing to release"
    );
}

/// Gate G5b: the deadline can fire before the registration ever left the node. What
/// leaves then is the subscribe followed by its unsubscribe, in that order, on one tick —
/// the pair the server must collapse (see `an_unsubscription_withdraws_a_registration_parked_in_the_same_pass`).
#[test]
fn cancelling_before_the_registration_left_sends_subscribe_then_unsubscribe_in_order() {
    // Internal on purpose: the observable is the ORDER of two frames on the sync sender
    // within one tick, which no public surface exposes.
    use crate::query_manager::manager::LocalUpdates;
    use crate::sync_manager::QueryPropagation;
    let app_id = AppId::from_name("cancel-one-shot-early");
    let schema_manager =
        SchemaManager::new(SyncManager::new(), test_schema(), app_id, "dev", "main").unwrap();
    let mut core = new_test_core(schema_manager, MemoryStorage::new(), NoopScheduler);
    let server_id = ServerId::new();
    core.add_server(server_id);
    core.immediate_tick();
    core.batched_tick();
    core.sync_sender().take();
    let baseline = core.schema_manager().query_manager().subscription_count();
    let (handle, _future) = core
        .query_with_local_batch_tracked(
            Query::new("users"),
            None,
            ReadDurabilityOptions {
                tier: Some(DurabilityTier::EdgeServer),
                local_updates: LocalUpdates::Immediate,
            },
            QueryPropagation::Full,
            None,
        )
        .expect("query setup");
    assert!(core.cancel_one_shot_query(handle));
    assert_eq!(
        core.schema_manager().query_manager().subscription_count(),
        baseline
    );
    core.batched_tick();
    let kinds: Vec<&'static str> = core
        .sync_sender()
        .take()
        .iter()
        .filter(|entry| entry.destination == Destination::Server(server_id))
        .filter_map(|entry| match &entry.payload {
            SyncPayload::QuerySubscription { .. } => Some("subscribe"),
            SyncPayload::QueryUnsubscription { .. } => Some("unsubscribe"),
            _ => None,
        })
        .collect();
    assert_eq!(kinds, vec!["subscribe", "unsubscribe"]);
}

/// Server-side half of gate G5: a subscribe and its unsubscribe parked in the same pass
/// must leave no server subscription behind. The query manager drains unsubscriptions
/// before subscriptions, so without the withdrawal in the inbox the registration would be
/// created AFTER its own unsubscription and live until the client disconnects.
#[test]
fn an_unsubscription_withdraws_a_registration_parked_in_the_same_pass() {
    let app_id = AppId::from_name("withdraw-parked-registration");
    let sync_manager = SyncManager::new().with_durability_tier(DurabilityTier::EdgeServer);
    let schema_manager =
        SchemaManager::new(sync_manager, test_schema(), app_id, "dev", "main").unwrap();
    let mut server = new_test_core(schema_manager, MemoryStorage::new(), NoopScheduler);
    server.immediate_tick();
    let client_id = ClientId::new();
    server.add_client(client_id, Some(Session::new("reader")));
    server.batched_tick();
    server.sync_sender().take();
    let query = server
        .schema_manager_mut()
        .query_manager_mut()
        .query("users")
        .build();
    let query_id = crate::sync_manager::QueryId(7);
    server.park_sync_message(InboxEntry {
        source: Source::Client(client_id),
        payload: SyncPayload::QuerySubscription {
            query_id,
            query: Box::new(query),
            session: Some(Session::new("reader")),
            required_tier: None,
            propagation: crate::sync_manager::QueryPropagation::Full,
            policy_context_tables: vec![],
        },
    });
    server.park_sync_message(InboxEntry {
        source: Source::Client(client_id),
        payload: SyncPayload::QueryUnsubscription { query_id },
    });
    server.batched_tick();
    server.immediate_tick();
    server.batched_tick();
    assert_eq!(
        server
            .schema_manager()
            .query_manager()
            .server_subscription_count(),
        0,
        "a registration withdrawn in the same pass must not outlive its unsubscription"
    );
}

/// Gate G5c: the deadline fires while the upstream is still pending — and has been for
/// longer than `PENDING_SERVER_TIMEOUT`. The registration was handed to the transport while
/// the upstream was young enough to write to, and the transport delivers it on the next
/// connection however late that is; the unsubscription must reach that server too, or the
/// registration outlives everything local that could withdraw it. This is the incident
/// shape: jazz-sync stalls, the socket drops, reads time out, the socket comes back.
#[test]
fn cancelling_a_read_registered_at_an_upstream_whose_connection_attempt_failed_still_unsubscribes_it()
 {
    use crate::query_manager::manager::LocalUpdates;
    use crate::sync_manager::{QueryId, QueryPropagation};
    // Internal on purpose: `remove_pending_server` is the core's own arm for `ConnectFailed`
    // and the registration marker it must forget is core state.
    // G5e. The transport buffers a registration pushed at a pending upstream and keeps it
    // across failed connection attempts; `ConnectFailed` reaches the core as
    // `remove_pending_server`. If that call forgets the registration marker, a cancellation
    // in the backoff window sends no unsubscription, and the registration lands at the
    // server on the next attempt with nothing to withdraw it.
    let app_id = AppId::from_name("cancel-one-shot-connect-failed");
    let schema_manager =
        SchemaManager::new(SyncManager::new(), test_schema(), app_id, "dev", "main").unwrap();
    let mut core = new_test_core(schema_manager, MemoryStorage::new(), NoopScheduler);
    let server_id = ServerId::new();
    core.schema_manager_mut()
        .query_manager_mut()
        .sync_manager_mut()
        .add_pending_server(server_id);
    core.immediate_tick();
    core.batched_tick();
    core.sync_sender().take();
    let (handle, _future) = core
        .query_with_local_batch_tracked(
            Query::new("users"),
            None,
            ReadDurabilityOptions {
                tier: Some(DurabilityTier::EdgeServer),
                local_updates: LocalUpdates::Immediate,
            },
            QueryPropagation::Full,
            None,
        )
        .expect("query setup");
    core.batched_tick();
    let registered = core.sync_sender().take();
    let registered_query_id = registered
        .iter()
        .find_map(|entry| match (&entry.destination, &entry.payload) {
            (Destination::Server(id), SyncPayload::QuerySubscription { query_id, .. })
                if *id == server_id =>
            {
                Some(*query_id)
            }
            _ => None,
        })
        .expect("a young pending upstream still receives the registration");
    // The connection attempt fails; the transport still holds the registration.
    core.schema_manager_mut()
        .query_manager_mut()
        .sync_manager_mut()
        .remove_pending_server(server_id);
    assert!(core.cancel_one_shot_query(handle));
    core.batched_tick();
    let unsubscribed: Vec<QueryId> = core
        .sync_sender()
        .take()
        .iter()
        .filter(|entry| entry.destination == Destination::Server(server_id))
        .filter_map(|entry| match &entry.payload {
            SyncPayload::QueryUnsubscription { query_id } => Some(*query_id),
            _ => None,
        })
        .collect();
    assert_eq!(
        unsubscribed,
        vec![registered_query_id],
        "the registration the transport still buffers for the failed upstream must be \
         withdrawn by an unsubscription addressed to that same upstream"
    );
    assert!(
        core.schema_manager()
            .query_manager()
            .sync_manager()
            .pending_server_query_subscription_count_for_test()
            == 0,
        "no registration marker may survive the cancellation"
    );
}

#[test]
fn a_registration_that_fails_to_compile_is_released_from_admission() {
    // Internal on purpose: a pending upstream (`add_pending_server`) and the frames the core
    // buffers for it are core state with no public observer.
    use crate::sync_manager::{QueryId, QueryPropagation, SubscriptionCaps};
    // Admission charges the principal when the frame is accepted, before the pass compiles
    // the query; a compile failure must give the charge back, or a client could be walled off
    // by its own typos (and an attacker could fill the ceiling with queries that cost nothing).
    // Internal on purpose: the observable is the admitted count on the sync manager, which no
    // public surface exposes — a client only sees its own rejection frame.
    let app_id = AppId::from_name("admission-compile-failure");
    let sync = SyncManager::new()
        .with_durability_tier(DurabilityTier::EdgeServer)
        .with_subscription_caps(SubscriptionCaps::default());
    let schema_manager = SchemaManager::new(sync, test_schema(), app_id, "dev", "main").unwrap();
    let mut server = new_test_core(schema_manager, MemoryStorage::new(), NoopScheduler);
    server.immediate_tick();
    let client_id = ClientId::new();
    server.add_client(client_id, Some(Session::new("reader")));
    server.batched_tick();
    let query_id = QueryId(11);
    server.park_sync_message(InboxEntry {
        source: Source::Client(client_id),
        payload: SyncPayload::QuerySubscription {
            query_id,
            query: Box::new(Query::new("no_such_table")),
            session: Some(Session::new("reader")),
            required_tier: None,
            propagation: QueryPropagation::Full,
            policy_context_tables: vec![],
        },
    });
    server.batched_tick();
    server.immediate_tick();
    server.batched_tick();
    let rejected = server.sync_sender().take().iter().any(|entry| {
        matches!(
            &entry.payload,
            SyncPayload::Error(crate::sync_manager::SyncError::QuerySubscriptionRejected { query_id: id, .. }) if *id == query_id
        )
    });
    assert!(rejected, "a query on an unknown table is rejected");
    assert_eq!(
        server
            .schema_manager()
            .query_manager()
            .sync_manager()
            .total_admitted_subscriptions(),
        0,
        "the charge for a registration that never compiled must be released"
    );
}

#[test]
fn a_tiered_read_hands_its_registration_to_the_transport_inside_the_query_call() {
    use crate::query_manager::manager::LocalUpdates;
    use crate::sync_manager::QueryPropagation;
    // GB (v18 item 3, half B). Internal on purpose: the observable is "the sync sender
    // received the registration before any tick ran", which only the captured sender can
    // see. Today the registration sits in the outbox until the next batched tick, and that
    // tick must win the engine lock behind whatever pass is running.
    let app_id = AppId::from_name("registration-leaves-in-the-query-call");
    let schema_manager =
        SchemaManager::new(SyncManager::new(), test_schema(), app_id, "dev", "main").unwrap();
    let mut core = new_test_core(schema_manager, MemoryStorage::new(), NoopScheduler);
    let server_id = ServerId::new();
    core.schema_manager_mut()
        .query_manager_mut()
        .sync_manager_mut()
        .add_pending_server(server_id);
    core.immediate_tick();
    core.batched_tick();
    core.sync_sender().take();
    let (_handle, _future) = core
        .query_with_local_batch_tracked(
            Query::new("users"),
            None,
            ReadDurabilityOptions {
                tier: Some(DurabilityTier::EdgeServer),
                local_updates: LocalUpdates::Immediate,
            },
            QueryPropagation::Full,
            None,
        )
        .expect("query setup");
    // No tick of any kind between the query call and this assertion.
    let registered = core.sync_sender().take().iter().any(|entry| {
        entry.destination == Destination::Server(server_id)
            && matches!(entry.payload, SyncPayload::QuerySubscription { .. })
    });
    assert!(
        registered,
        "the registration must reach the sync sender inside the query call, not on the \
         next batched tick"
    );
}

#[test]
fn a_tiered_subscription_hands_its_registration_to_the_transport_inside_the_subscribe_call() {
    use crate::query_manager::manager::LocalUpdates;
    use crate::sync_manager::QueryPropagation;
    // GB twin (v18 item 3, half B), the subscribe path. `create_subscription` +
    // `execute_subscription` flush the registration at a different site than the query call
    // does, so a disarm of either site is visible to exactly one of the two gates (the
    // falsification of the query-call gate alone left the subscribe site uncovered).
    // Internal on purpose: same observable as the query-call gate above.
    let app_id = AppId::from_name("registration-leaves-in-the-subscribe-call");
    let schema_manager =
        SchemaManager::new(SyncManager::new(), test_schema(), app_id, "dev", "main").unwrap();
    let mut core = new_test_core(schema_manager, MemoryStorage::new(), NoopScheduler);
    let server_id = ServerId::new();
    core.schema_manager_mut()
        .query_manager_mut()
        .sync_manager_mut()
        .add_pending_server(server_id);
    core.immediate_tick();
    core.batched_tick();
    core.sync_sender().take();
    let handle = core.create_subscription(
        Query::new("users"),
        None,
        ReadDurabilityOptions {
            tier: Some(DurabilityTier::EdgeServer),
            local_updates: LocalUpdates::Immediate,
        },
        QueryPropagation::Full,
    );
    core.execute_subscription(handle, |_delta| {})
        .expect("subscription setup");
    // No tick of any kind between the execute call and this assertion.
    let registered = core.sync_sender().take().iter().any(|entry| {
        entry.destination == Destination::Server(server_id)
            && matches!(entry.payload, SyncPayload::QuerySubscription { .. })
    });
    assert!(
        registered,
        "the registration must reach the sync sender inside the subscribe call, not on the \
         next batched tick"
    );
}

#[test]
fn a_failed_re_registration_of_a_live_query_ends_the_old_subscription() {
    use crate::sync_manager::{QueryId, QueryPropagation, SubscriptionCaps};
    // A re-registration of a held id is admitted without a charge (replays are free), so it
    // reaches compile; if it fails there, the rejection tells the client the id is dead and
    // the charge is released. The old subscription under that id must end with it —
    // otherwise a client can grow live, uncounted server subscriptions one failed
    // re-registration at a time, past every cap. The registration this server forwarded
    // upstream for the replaced query must be withdrawn there too, or it settles at the
    // upstream, charged to this server, for the life of the connection.
    // Internal on purpose: live server subscriptions, admitted charges and the upstream
    // outbox are engine state no public surface exposes.
    let app_id = AppId::from_name("admission-failed-re-registration");
    let sync = SyncManager::new()
        .with_durability_tier(DurabilityTier::EdgeServer)
        .with_subscription_caps(SubscriptionCaps::default());
    let schema_manager = SchemaManager::new(sync, test_schema(), app_id, "dev", "main").unwrap();
    let mut server = new_test_core(schema_manager, MemoryStorage::new(), NoopScheduler);
    server.immediate_tick();
    let client_id = ClientId::new();
    server.add_client(client_id, Some(Session::new("reader")));
    let upstream = ServerId::new();
    server
        .schema_manager_mut()
        .query_manager_mut()
        .sync_manager_mut()
        .add_pending_server(upstream);
    server.batched_tick();
    server.sync_sender().take();
    let query_id = QueryId(21);
    let register = |server: &mut RuntimeCore<MemoryStorage, NoopScheduler>, table: &str| {
        server.park_sync_message(InboxEntry {
            source: Source::Client(client_id),
            payload: SyncPayload::QuerySubscription {
                query_id,
                query: Box::new(Query::new(table)),
                session: Some(Session::new("reader")),
                required_tier: None,
                propagation: QueryPropagation::Full,
                policy_context_tables: vec![],
            },
        });
        server.batched_tick();
        server.immediate_tick();
        server.batched_tick();
    };
    let counts = |server: &RuntimeCore<MemoryStorage, NoopScheduler>| {
        let qm = server.schema_manager().query_manager();
        (
            qm.server_subscription_count(),
            qm.sync_manager().total_admitted_subscriptions(),
        )
    };
    register(&mut server, "users");
    assert_eq!(
        counts(&server),
        (1, 1),
        "the first registration is live and counted"
    );
    let forwarded = server.sync_sender().take().iter().any(|entry| {
        matches!(
            (&entry.destination, &entry.payload),
            (Destination::Server(id), SyncPayload::QuerySubscription { query_id: q, .. }) if *id == upstream && *q == query_id
        )
    });
    assert!(
        forwarded,
        "fixture precondition: a full-propagation registration is forwarded to the upstream"
    );
    register(&mut server, "no_such_table");
    assert_eq!(
        counts(&server),
        (0, 0),
        "a re-registration that fails to compile ends the subscription it replaced: live \
         server subscriptions and admitted charges must both be zero"
    );
    let withdrawn = server.sync_sender().take().iter().any(|entry| {
        matches!(
            (&entry.destination, &entry.payload),
            (Destination::Server(id), SyncPayload::QueryUnsubscription { query_id: q }) if *id == upstream && *q == query_id
        )
    });
    assert!(
        withdrawn,
        "the registration forwarded upstream for the replaced query must be withdrawn there"
    );
}

#[test]
fn cancelling_a_read_registered_at_a_long_pending_upstream_still_unsubscribes_it() {
    use crate::query_manager::manager::LocalUpdates;
    use crate::sync_manager::{PENDING_SERVER_TIMEOUT, QueryId, QueryPropagation};
    let app_id = AppId::from_name("cancel-one-shot-pending-upstream");
    let schema_manager =
        SchemaManager::new(SyncManager::new(), test_schema(), app_id, "dev", "main").unwrap();
    let mut core = new_test_core(schema_manager, MemoryStorage::new(), NoopScheduler);
    let server_id = ServerId::new();
    core.schema_manager_mut()
        .query_manager_mut()
        .sync_manager_mut()
        .add_pending_server(server_id);
    core.immediate_tick();
    core.batched_tick();
    core.sync_sender().take();
    let (handle, _future) = core
        .query_with_local_batch_tracked(
            Query::new("users"),
            None,
            ReadDurabilityOptions {
                tier: Some(DurabilityTier::EdgeServer),
                local_updates: LocalUpdates::Immediate,
            },
            QueryPropagation::Full,
            None,
        )
        .expect("query setup");
    core.batched_tick();
    let registered = core.sync_sender().take();
    let registered_query_id = registered
        .iter()
        .find_map(|entry| match (&entry.destination, &entry.payload) {
            (Destination::Server(id), SyncPayload::QuerySubscription { query_id, .. })
                if *id == server_id =>
            {
                Some(*query_id)
            }
            _ => None,
        })
        .expect("a young pending upstream still receives the registration");
    // The upstream stays pending past the point where outbound writes stop targeting it.
    core.schema_manager_mut()
        .query_manager_mut()
        .sync_manager_mut()
        .age_pending_server_for_test(server_id, PENDING_SERVER_TIMEOUT * 3);
    assert!(core.cancel_one_shot_query(handle));
    core.batched_tick();
    let unsubscribed: Vec<QueryId> = core
        .sync_sender()
        .take()
        .iter()
        .filter(|entry| entry.destination == Destination::Server(server_id))
        .filter_map(|entry| match &entry.payload {
            SyncPayload::QueryUnsubscription { query_id } => Some(*query_id),
            _ => None,
        })
        .collect();
    assert_eq!(
        unsubscribed,
        vec![registered_query_id],
        "the unsubscription must follow the registration to the pending upstream"
    );
}
