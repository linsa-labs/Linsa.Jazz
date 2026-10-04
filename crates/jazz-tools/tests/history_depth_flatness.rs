//! Depth-flatness perf gates for the row-history hot paths (v13-1, §6.3 of
//! the include-plan-sharing design; workload shapes from the history-fastpaths
//! design §4/§5).
//!
//! Deterministic, counter-based, no wall-clock: WORK per 500-op window is
//! measured as (storage reads) + (TrackingAllocator bytes) on ONE hot row at
//! history depth ~500 vs ~8000, and the gate asserts
//! `work(8000) <= work(500) * 1.2` per verb:
//!
//! - serial applies (`apply_row_batch`, unconfirmed tips) — flat via the
//!   Fix A serial fast path;
//! - append + wait-confirm-per-write at `EdgeServer` (the external agent's
//!   workload: every write is immediately re-applied through
//!   `accepted_transaction_output(EdgeServer)`) — flat since v13-2 removed
//!   the B3 tier-pointer wall (causal-domination tier frontiers +
//!   unconditional tier-entry pointer collapse, `row_histories/fastpath.rs`);
//! - visible reads (`load_visible_query_row`, the query-serving point read) —
//!   flat today via Fix C frontier-first reads.
//!
//! "Storage reads" counts every low-level read the storage sees: point gets,
//! scan calls, and the rows those scans return. The counting wrapper forwards
//! only the raw-table byte ops (plus the write entry points whose trait
//! defaults are unimplemented), so every higher-level read decomposes into
//! counted raw operations — the same interception idea as `DfsProbeStorage`
//! in `src/sync_manager/tests/frontier_pruning.rs`, widened from two probes to
//! total read work. Allocator bytes are CUMULATIVE allocations (churn), not
//! live bytes: an O(depth) history scan shows up as O(depth) allocation even
//! when everything is freed again.
//!
//! The three tests share one process-global allocator, so they serialise on a
//! mutex; each window's counters are read only while the lock is held.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};

use uuid::Uuid;

use jazz_tools::batch_fate::BatchFate;
use jazz_tools::metadata::RowProvenance;
use jazz_tools::object::{BranchName, ObjectId};
use jazz_tools::query_manager::types::{
    ColumnDescriptor, ColumnType, RowDescriptor, Schema, SchemaHash, TableName, TableSchema, Value,
};
use jazz_tools::row_format::encode_row;
use jazz_tools::row_histories::{
    BatchId, HistoryScan, QueryRowBatch, RowState, StoredRowBatch, VisibleRowEntry, apply_row_batch,
};
use jazz_tools::storage::{
    HistoryRowBytes, IndexMutation, MemoryStorage, OwnedHistoryRowBytes, OwnedVisibleRowBytes,
    RawTableMutation, RawTableRows, RowLocator, Storage, StorageError, VisibleRowBytes,
};
use jazz_tools::sync_manager::DurabilityTier;
use jazz_tools::test_support::persist_test_schema;

// ============================================================================
// Tracking allocator: cumulative allocated bytes (churn), monotone.
// ============================================================================

struct TrackingAllocator;

static TOTAL_ALLOCATED: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            TOTAL_ALLOCATED.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            TOTAL_ALLOCATED.fetch_add(new_size as u64, Ordering::Relaxed);
        }
        new_ptr
    }
}

#[global_allocator]
static GLOBAL: TrackingAllocator = TrackingAllocator;

fn total_allocated() -> u64 {
    TOTAL_ALLOCATED.load(Ordering::Relaxed)
}

/// Serialises the measuring tests: the allocator counter is process-global.
fn measure_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
}

// ============================================================================
// Counting storage: total low-level read work over a MemoryStorage.
// ============================================================================

/// Forwards `MemoryStorage`'s native read and write surface, counting every
/// read: point loads count 1, scans count the call plus every row they
/// return. `MemoryStorage` keeps its row regions in native maps (its raw
/// tables carry only headers/locators/catalogue), so the wrapper must forward
/// the structured methods natively rather than letting trait defaults
/// decompose them onto raw tables the native writes never populate.
struct CountingStorage {
    inner: MemoryStorage,
    point_gets: Cell<u64>,
    scan_calls: Cell<u64>,
    scanned_rows: Cell<u64>,
}

impl CountingStorage {
    fn new(inner: MemoryStorage) -> Self {
        Self {
            inner,
            point_gets: Cell::new(0),
            scan_calls: Cell::new(0),
            scanned_rows: Cell::new(0),
        }
    }

    fn reset(&self) {
        self.point_gets.set(0);
        self.scan_calls.set(0);
        self.scanned_rows.set(0);
    }

    /// Total read work: point gets + scan calls + rows yielded by scans.
    fn reads(&self) -> u64 {
        self.point_gets.get() + self.scan_calls.get() + self.scanned_rows.get()
    }

    fn count_point(&self) {
        self.point_gets.set(self.point_gets.get() + 1);
    }

    fn count_scan(&self, rows: usize) {
        self.scan_calls.set(self.scan_calls.get() + 1);
        self.scanned_rows.set(self.scanned_rows.get() + rows as u64);
    }
}

impl Storage for CountingStorage {
    // ---- raw byte ops -----------------------------------------------------

    fn raw_table_put(&mut self, table: &str, key: &str, value: &[u8]) -> Result<(), StorageError> {
        self.inner.raw_table_put(table, key, value)
    }

    fn raw_table_delete(&mut self, table: &str, key: &str) -> Result<(), StorageError> {
        self.inner.raw_table_delete(table, key)
    }

    fn apply_raw_table_mutations(
        &mut self,
        mutations: &[RawTableMutation<'_>],
    ) -> Result<(), StorageError> {
        self.inner.apply_raw_table_mutations(mutations)
    }

    fn raw_table_get(&self, table: &str, key: &str) -> Result<Option<Vec<u8>>, StorageError> {
        self.count_point();
        self.inner.raw_table_get(table, key)
    }

    fn raw_table_scan_prefix(
        &self,
        table: &str,
        prefix: &str,
    ) -> Result<RawTableRows, StorageError> {
        let rows = self.inner.raw_table_scan_prefix(table, prefix)?;
        self.count_scan(rows.len());
        Ok(rows)
    }

    fn raw_table_scan_range(
        &self,
        table: &str,
        start: Option<&str>,
        end: Option<&str>,
    ) -> Result<RawTableRows, StorageError> {
        let rows = self.inner.raw_table_scan_range(table, start, end)?;
        self.count_scan(rows.len());
        Ok(rows)
    }

    // ---- locators ---------------------------------------------------------

    fn put_row_locator(
        &mut self,
        id: ObjectId,
        locator: Option<&RowLocator>,
    ) -> Result<(), StorageError> {
        self.inner.put_row_locator(id, locator)
    }

    fn load_row_locator(&self, id: ObjectId) -> Result<Option<RowLocator>, StorageError> {
        self.count_point();
        self.inner.load_row_locator(id)
    }

    fn storage_cache_namespace(&self) -> usize {
        self.inner.storage_cache_namespace()
    }

    // ---- row-region writes (uncounted; the gates measure read work) -------

    fn apply_prepared_row_mutation(
        &mut self,
        table: &str,
        history_rows: &[StoredRowBatch],
        visible_entries: &[VisibleRowEntry],
        encoded_history_rows: &[OwnedHistoryRowBytes],
        encoded_visible_rows: &[OwnedVisibleRowBytes],
        index_mutations: &[IndexMutation<'_>],
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

    fn apply_encoded_row_mutation(
        &mut self,
        table: &str,
        history_rows: &[OwnedHistoryRowBytes],
        visible_rows: &[OwnedVisibleRowBytes],
        index_mutations: &[IndexMutation<'_>],
    ) -> Result<(), StorageError> {
        self.inner
            .apply_encoded_row_mutation(table, history_rows, visible_rows, index_mutations)
    }

    fn append_history_region_rows(
        &mut self,
        table: &str,
        rows: &[StoredRowBatch],
    ) -> Result<(), StorageError> {
        self.inner.append_history_region_rows(table, rows)
    }

    fn append_history_region_row_bytes(
        &mut self,
        table: &str,
        rows: &[HistoryRowBytes<'_>],
    ) -> Result<(), StorageError> {
        self.inner.append_history_region_row_bytes(table, rows)
    }

    fn upsert_visible_region_rows(
        &mut self,
        table: &str,
        entries: &[VisibleRowEntry],
    ) -> Result<(), StorageError> {
        self.inner.upsert_visible_region_rows(table, entries)
    }

    fn upsert_visible_region_row_bytes(
        &mut self,
        table: &str,
        rows: &[VisibleRowBytes<'_>],
    ) -> Result<(), StorageError> {
        self.inner.upsert_visible_region_row_bytes(table, rows)
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
        batch_id: BatchId,
        state: Option<RowState>,
        confirmed_tier: Option<DurabilityTier>,
    ) -> Result<(), StorageError> {
        self.inner
            .patch_row_region_rows_by_batch(table, batch_id, state, confirmed_tier)
    }

    fn patch_exact_row_batch_for_schema_hash(
        &mut self,
        table: &str,
        schema_hash: SchemaHash,
        branch: &str,
        row_id: ObjectId,
        batch_id: BatchId,
        state: Option<RowState>,
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

    fn upsert_authoritative_batch_fate(&mut self, fate: &BatchFate) -> Result<(), StorageError> {
        self.inner.upsert_authoritative_batch_fate(fate)
    }

    // ---- point reads ------------------------------------------------------

    fn load_authoritative_batch_fate(
        &self,
        batch_id: BatchId,
    ) -> Result<Option<BatchFate>, StorageError> {
        self.count_point();
        self.inner.load_authoritative_batch_fate(batch_id)
    }

    fn load_visible_region_row(
        &self,
        table: &str,
        branch: &str,
        row_id: ObjectId,
    ) -> Result<Option<StoredRowBatch>, StorageError> {
        self.count_point();
        self.inner.load_visible_region_row(table, branch, row_id)
    }

    fn load_visible_query_row(
        &self,
        table: &str,
        branch: &str,
        row_id: ObjectId,
    ) -> Result<Option<QueryRowBatch>, StorageError> {
        self.count_point();
        self.inner.load_visible_query_row(table, branch, row_id)
    }

    fn load_visible_region_entry(
        &self,
        table: &str,
        branch: &str,
        row_id: ObjectId,
    ) -> Result<Option<VisibleRowEntry>, StorageError> {
        self.count_point();
        self.inner.load_visible_region_entry(table, branch, row_id)
    }

    fn load_visible_region_row_bytes(
        &self,
        table: &str,
        branch: &str,
        row_id: ObjectId,
    ) -> Result<Option<Vec<u8>>, StorageError> {
        self.count_point();
        self.inner
            .load_visible_region_row_bytes(table, branch, row_id)
    }

    fn load_visible_region_frontier(
        &self,
        table: &str,
        branch: &str,
        row_id: ObjectId,
    ) -> Result<Option<Vec<BatchId>>, StorageError> {
        self.count_point();
        self.inner
            .load_visible_region_frontier(table, branch, row_id)
    }

    fn load_history_row_batch(
        &self,
        table: &str,
        branch: &str,
        row_id: ObjectId,
        batch_id: BatchId,
    ) -> Result<Option<StoredRowBatch>, StorageError> {
        self.count_point();
        self.inner
            .load_history_row_batch(table, branch, row_id, batch_id)
    }

    fn load_history_row_batch_bytes(
        &self,
        table: &str,
        branch: &str,
        row_id: ObjectId,
        batch_id: BatchId,
    ) -> Result<Option<Vec<u8>>, StorageError> {
        self.count_point();
        self.inner
            .load_history_row_batch_bytes(table, branch, row_id, batch_id)
    }

    fn load_history_row_batch_for_schema_hash(
        &self,
        table: &str,
        schema_hash: SchemaHash,
        branch: &str,
        row_id: ObjectId,
        batch_id: BatchId,
    ) -> Result<Option<StoredRowBatch>, StorageError> {
        self.count_point();
        self.inner.load_history_row_batch_for_schema_hash(
            table,
            schema_hash,
            branch,
            row_id,
            batch_id,
        )
    }

    fn load_history_query_row_batch(
        &self,
        table: &str,
        branch: &str,
        row_id: ObjectId,
        batch_id: BatchId,
    ) -> Result<Option<QueryRowBatch>, StorageError> {
        self.count_point();
        self.inner
            .load_history_query_row_batch(table, branch, row_id, batch_id)
    }

    // ---- scans ------------------------------------------------------------

    fn scan_history_region(
        &self,
        table: &str,
        branch: &str,
        scan: HistoryScan,
    ) -> Result<Vec<StoredRowBatch>, StorageError> {
        let rows = self.inner.scan_history_region(table, branch, scan)?;
        self.count_scan(rows.len());
        Ok(rows)
    }

    fn scan_history_row_batches(
        &self,
        table: &str,
        row_id: ObjectId,
    ) -> Result<Vec<StoredRowBatch>, StorageError> {
        let rows = self.inner.scan_history_row_batches(table, row_id)?;
        self.count_scan(rows.len());
        Ok(rows)
    }

    fn scan_history_region_bytes(
        &self,
        table: &str,
        scan: HistoryScan,
    ) -> Result<Vec<Vec<u8>>, StorageError> {
        let rows = self.inner.scan_history_region_bytes(table, scan)?;
        self.count_scan(rows.len());
        Ok(rows)
    }

    fn scan_visible_region(
        &self,
        table: &str,
        branch: &str,
    ) -> Result<Vec<StoredRowBatch>, StorageError> {
        let rows = self.inner.scan_visible_region(table, branch)?;
        self.count_scan(rows.len());
        Ok(rows)
    }

    fn scan_visible_region_row_batches(
        &self,
        table: &str,
        row_id: ObjectId,
    ) -> Result<Vec<StoredRowBatch>, StorageError> {
        let rows = self.inner.scan_visible_region_row_batches(table, row_id)?;
        self.count_scan(rows.len());
        Ok(rows)
    }

    fn scan_visible_region_bytes(
        &self,
        table: &str,
        branch: &str,
    ) -> Result<Vec<Vec<u8>>, StorageError> {
        let rows = self.inner.scan_visible_region_bytes(table, branch)?;
        self.count_scan(rows.len());
        Ok(rows)
    }
}

// ============================================================================
// Hot-row driver.
// ============================================================================

const TABLE: &str = "hot_rows";
const BRANCH: &str = "main";
const AUTHOR: &str = "hot-author";

const DEPTH_LOW: usize = 500;
const DEPTH_HIGH: usize = 8_000;
const WINDOW_OPS: usize = 500;

fn hot_row_descriptor() -> RowDescriptor {
    RowDescriptor::new(vec![
        ColumnDescriptor::new("label", ColumnType::Text),
        ColumnDescriptor::new("seq", ColumnType::Integer),
    ])
}

fn hot_row_schema() -> Schema {
    let mut schema = Schema::new();
    schema.insert(
        TableName::new(TABLE),
        TableSchema::new(hot_row_descriptor()),
    );
    schema
}

struct HotRow {
    storage: CountingStorage,
    row_id: ObjectId,
    branch: BranchName,
    descriptor: RowDescriptor,
    /// The current sole tip as last applied (content, state, tier included).
    tip: Option<StoredRowBatch>,
    depth: usize,
    next_ts: u64,
    mint_counter: u128,
}

impl HotRow {
    fn new() -> Self {
        let schema = hot_row_schema();
        let mut storage = CountingStorage::new(MemoryStorage::new());
        let schema_hash = persist_test_schema(&mut storage, &schema);
        let row_id = ObjectId::from_uuid(Uuid::from_u128(0x0507_0000_0000_0001));
        storage
            .put_row_locator(
                row_id,
                Some(&RowLocator {
                    table: TABLE.into(),
                    origin_schema_hash: Some(schema_hash),
                }),
            )
            .expect("hot row locator should persist");
        Self {
            storage,
            row_id,
            branch: BranchName::new(BRANCH),
            descriptor: hot_row_descriptor(),
            tip: None,
            depth: 0,
            next_ts: 1_000,
            mint_counter: 0,
        }
    }

    fn next_batch_id(&mut self) -> BatchId {
        self.mint_counter += 1;
        BatchId::from_uuid(Uuid::from_u128(
            0x0B0B_0000_0000_0000_0000 + self.mint_counter,
        ))
    }

    /// One serial append: a fresh unconfirmed visible batch on top of the tip
    /// (the production write shape — every client publish stores
    /// `confirmed_tier: None`).
    fn append(&mut self) {
        let parents: Vec<BatchId> = self.tip.iter().map(StoredRowBatch::batch_id).collect();
        let batch_id = self.next_batch_id();
        let ts = self.next_ts;
        self.next_ts += 10;
        let provenance = if self.depth == 0 {
            RowProvenance::for_insert(AUTHOR.to_string(), ts)
        } else {
            RowProvenance {
                created_by: AUTHOR.into(),
                created_at: 1_000,
                updated_by: AUTHOR.into(),
                updated_at: ts,
            }
        };
        // Fixed-size payload so per-op allocation is comparable across depths.
        let values = vec![
            Value::Text("hot".into()),
            Value::Integer((self.depth % 1_000) as i32),
        ];
        let data = encode_row(&self.descriptor, &values).expect("hot row values should encode");
        let batch = StoredRowBatch::new_with_batch_id(
            batch_id,
            self.row_id,
            BRANCH,
            parents,
            data,
            provenance,
            HashMap::new(),
            RowState::VisibleDirect,
            None,
        );
        apply_row_batch(
            &mut self.storage,
            self.row_id,
            &self.branch,
            batch.clone(),
            &[],
        )
        .unwrap_or_else(|err| panic!("serial append at depth {} failed: {err:?}", self.depth));
        self.tip = Some(batch);
        self.depth += 1;
    }

    /// Confirm the current tip at `EdgeServer`: the sync inbox's
    /// `BatchFate::AcceptedTransaction` shape — the SAME batch id re-applied
    /// through `accepted_transaction_output` with a raised confirmed tier
    /// (`sync_manager/inbox.rs`). This is what "append then wait for the write
    /// to confirm" costs on the author's storage for every single write.
    fn confirm_at_edge(&mut self) {
        let tip = self
            .tip
            .as_ref()
            .expect("confirm requires an appended tip")
            .clone();
        let confirmed = tip.accepted_transaction_output(DurabilityTier::EdgeServer);
        apply_row_batch(
            &mut self.storage,
            self.row_id,
            &self.branch,
            confirmed.clone(),
            &[],
        )
        .unwrap_or_else(|err| panic!("edge confirm at depth {} failed: {err:?}", self.depth));
        self.tip = Some(confirmed);
    }

    /// One DELIVERED version: the same visible batch a peer sends, which is to say with
    /// its parents stripped — `scope_delivery_row` clears them for every visible row it
    /// puts on the wire.
    ///
    /// On the receiver a parentless visible row is a frontier ROOT, so each delivery adds a
    /// tip instead of advancing one. `parents_cover_frontier_exactly` rejects empty parents
    /// outright, so the O(1) construction never applies and every delivery falls to a full
    /// history read plus a rebuild — and the frontier it rebuilds grows by one every time.
    fn deliver(&mut self) {
        let batch_id = self.next_batch_id();
        let ts = self.next_ts;
        self.next_ts += 10;
        let provenance = if self.depth == 0 {
            RowProvenance::for_insert(AUTHOR.to_string(), ts)
        } else {
            RowProvenance {
                created_by: AUTHOR.into(),
                created_at: 1_000,
                updated_by: AUTHOR.into(),
                updated_at: ts,
            }
        };
        let values = vec![
            Value::Text("hot".into()),
            Value::Integer((self.depth % 1_000) as i32),
        ];
        let data = encode_row(&self.descriptor, &values).expect("hot row values should encode");
        let batch = StoredRowBatch::new_with_batch_id(
            batch_id,
            self.row_id,
            BRANCH,
            // Stripped on the wire. This is the whole point of the gate.
            Vec::new(),
            data,
            provenance,
            HashMap::new(),
            RowState::VisibleDirect,
            None,
        );
        apply_row_batch(
            &mut self.storage,
            self.row_id,
            &self.branch,
            batch.clone(),
            &[],
        )
        .unwrap_or_else(|err| panic!("delivery at depth {} failed: {err:?}", self.depth));
        self.tip = Some(batch);
        self.depth += 1;
    }

    /// Grow the history the way a receiver's history actually grows: by delivery.
    fn grow_delivered_to(&mut self, depth: usize) {
        while self.depth < depth {
            self.deliver();
        }
    }

    /// The query-serving visible point read.
    fn read_visible(&self) {
        let row = self
            .storage
            .load_visible_query_row(TABLE, BRANCH, self.row_id)
            .expect("visible read should succeed");
        assert!(row.is_some(), "hot row must stay visible");
    }

    /// Grow the history with cheap serial appends (the workload under test
    /// runs only inside measured windows).
    fn grow_to(&mut self, depth: usize) {
        while self.depth < depth {
            self.append();
        }
    }
}

// ============================================================================
// Measurement + gate.
// ============================================================================

#[derive(Debug, Clone, Copy)]
struct WorkSample {
    reads: u64,
    alloc_bytes: u64,
}

fn measure(hot: &mut HotRow, ops: usize, mut op: impl FnMut(&mut HotRow)) -> WorkSample {
    hot.storage.reset();
    let alloc_before = total_allocated();
    for _ in 0..ops {
        op(hot);
    }
    WorkSample {
        reads: hot.storage.reads(),
        alloc_bytes: total_allocated() - alloc_before,
    }
}

/// Absolute slack for the allocation-churn comparison: a fully Arc-shared
/// read path can allocate zero bytes per window, and a strict ratio over a
/// zero baseline would be meaningless. 16 KiB per 500-op window is noise
/// (~32 B/op), far below any O(depth) regression this gate exists to catch.
const ALLOC_SLACK_BYTES: u64 = 16 * 1024;

/// The gate: work per window at depth ~8000 must stay within 1.2x of the
/// window at depth ~500, for both storage reads and allocated bytes.
fn assert_flat(verb: &str, low: WorkSample, high: WorkSample) {
    eprintln!(
        "depth-flatness [{verb}]: depth {DEPTH_LOW} -> reads {} allocs {}B; \
         depth {DEPTH_HIGH} -> reads {} allocs {}B (window {WINDOW_OPS} ops)",
        low.reads, low.alloc_bytes, high.reads, high.alloc_bytes
    );
    assert!(
        low.reads > 0,
        "{verb}: low window performed no storage reads"
    );
    assert!(
        high.reads * 10 <= low.reads * 12,
        "{verb}: storage reads are not flat in history depth: \
         {} reads at depth {DEPTH_HIGH} vs {} at depth {DEPTH_LOW} (limit 1.2x)",
        high.reads,
        low.reads,
    );
    assert!(
        high.alloc_bytes * 10 <= low.alloc_bytes * 12 + ALLOC_SLACK_BYTES * 10,
        "{verb}: allocation churn is not flat in history depth: \
         {} bytes at depth {DEPTH_HIGH} vs {} at depth {DEPTH_LOW} (limit 1.2x + {ALLOC_SLACK_BYTES}B slack)",
        high.alloc_bytes,
        low.alloc_bytes,
    );
}

/// (a) Serial applies must be flat — the Fix A serial-write fast path.
#[test]
fn serial_apply_work_is_flat_in_history_depth() {
    let _lock = measure_lock();
    let mut hot = HotRow::new();
    hot.grow_to(DEPTH_LOW);
    let low = measure(&mut hot, WINDOW_OPS, HotRow::append);
    hot.grow_to(DEPTH_HIGH);
    let high = measure(&mut hot, WINDOW_OPS, HotRow::append);
    assert_flat("serial applies", low, high);
}

/// (b) Append + wait-confirm-per-write at `EdgeServer` — the external agent's
/// workload. One window op = one append + one edge confirm of that batch.
///
/// This was the B3 tier-pointer wall (history-fastpaths design §5, "pure
/// confirmed_tier bump" row; include-plan-sharing design §9): the confirm is
/// an in-place tip update whose EdgeServer tier pointer is `Some(previous
/// confirmed batch)`, and under one-step tier frontiers `in_place_tier_pointer`
/// could only prove the `(false -> true)` tier entry when the pointer was
/// `None` — every confirm after the first declined to an O(depth) rebuild.
/// v13-2 switched the tier-filtered frontier to causal domination through the
/// full history DAG (`row_histories/resolution.rs`), under which a confirmed
/// sole tip provably dominates every tier set it enters — the tier-entry
/// transition is O(1) unconditionally, holes in the unconfirmed prefix
/// included (this test's growth phase creates exactly such a hole).
#[test]
fn append_confirm_per_write_at_edge_server_is_flat_in_history_depth() {
    let _lock = measure_lock();
    let mut hot = HotRow::new();
    hot.grow_to(DEPTH_LOW);
    let low = measure(&mut hot, WINDOW_OPS, |hot| {
        hot.append();
        hot.confirm_at_edge();
    });
    hot.grow_to(DEPTH_HIGH);
    let high = measure(&mut hot, WINDOW_OPS, |hot| {
        hot.append();
        hot.confirm_at_edge();
    });
    assert_flat("append+confirm@edge", low, high);
}

/// (c) Visible reads must be flat — Fix C frontier-first point reads.
#[test]
fn visible_read_work_is_flat_in_history_depth() {
    let _lock = measure_lock();
    let mut hot = HotRow::new();
    hot.grow_to(DEPTH_LOW);
    let low = measure(&mut hot, WINDOW_OPS, |hot| hot.read_visible());
    hot.grow_to(DEPTH_HIGH);
    let high = measure(&mut hot, WINDOW_OPS, |hot| hot.read_visible());
    assert_flat("visible reads", low, high);
}

/// (d) A row whose versions ARRIVE — the receiving side of every chat.
///
/// Delivery strips parents from visible rows (`sync_manager::sync_logic::scope_delivery_row`),
/// so on the receiver every delivered version is a frontier root. The frontier therefore
/// grows by one per delivery, `parents_cover_frontier_exactly` rejects the empty parent set
/// before it compares anything, and the O(1) construction is unreachable for the life of the
/// row: each arrival reads the whole history and rebuilds.
///
/// MEASURED on a device store, 2026-08-20: `users` row 1234c757 — 460 frontier tips at
/// history depth 460; `chat_activities` row 92d67d1d — 145 tips at depth 145. Every version
/// a tip, none naming a parent. The phone showed a CPU spike on every presence beat, peaking
/// at 107% of a core, and 139% sustained while a draft was typed.
///
/// Rows the device writes ITSELF are single-tip and take the fast path — the same store had
/// `users` 2b1452ba at depth 11956 with one tip. The cost is not history depth; it is being
/// the receiver.
#[test]
fn delivered_apply_work_is_flat_in_history_depth() {
    let _lock = measure_lock();
    let mut hot = HotRow::new();
    hot.grow_delivered_to(DEPTH_LOW);
    let low = measure(&mut hot, WINDOW_OPS, HotRow::deliver);
    hot.grow_delivered_to(DEPTH_HIGH);
    let high = measure(&mut hot, WINDOW_OPS, HotRow::deliver);
    assert_flat("delivered applies", low, high);
}
