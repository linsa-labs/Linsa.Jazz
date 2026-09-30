//! Instance-count flatness gate for include (array-subquery) dirtiness.
//!
//! The depth gates in `history_depth_flatness.rs` measure work against history
//! DEPTH. They are blind to the axis that produced the v13-2 CPU regression on
//! the live server: work against the number of CACHED SUBGRAPH INSTANCES an
//! `ArraySubqueryNode` holds. Precise dirtiness threaded row-precise marks into
//! every cached instance EAGERLY, at write time, so one write into an include's
//! inner table cost O(live outer rows) — on the real store that turned a
//! ten-second presence heartbeat into a CPU burst that scaled with the size of
//! the subscribed result set.
//!
//! This gate pins the axis: the SAME single-row write into the include's inner
//! table, measured with 50 outer rows instantiated and again with 1000, must
//! cost the same work. WORK is (storage reads) + (TrackingAllocator bytes)
//! across the write plus the settle it triggers, the same counter pair the
//! depth gates use; the counting storage is the same wrapper idea, forwarding
//! `MemoryStorage`'s native surface so every read decomposes into a counted
//! low-level operation.
//!
//! ONE VARIABLE AT A TIME (v14). The outer query is windowed to a single
//! delivered row, so the instance count is the only thing that moves between
//! the two runs. Without the window the fixture also varied the DELIVERED
//! result set, and delivery carries an O(delivered rows) cost of its own: the
//! manager re-derives its client mirror from the whole visible tuple set on
//! every settle. That term is present with no include in the graph at all —
//! `unwindowed_write_measures_the_delivery_diff_not_the_include` measures both
//! and reports the two slopes side by side, so the window here is a measured
//! decision, not a way to make the number smaller.
//!
//! Run with precise dirtiness FORCED ON — the legacy path is flat here only
//! because it is coarse (and stale, see `subscription_output_oracle.rs`), so
//! measuring the default-off path would prove nothing.

#![cfg(feature = "test")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};

use jazz_tools::batch_fate::BatchFate;
use jazz_tools::object::ObjectId;
use jazz_tools::query_manager::manager::QueryManager;
use jazz_tools::query_manager::precise_dirty::force_precise_dirty;
use jazz_tools::query_manager::settle_cost::SettleCounts;
use jazz_tools::query_manager::types::{
    ColumnDescriptor, ColumnType, RowDescriptor, Schema, SchemaHash, TableName, Value,
};
use jazz_tools::row_histories::{
    BatchId, HistoryScan, QueryRowBatch, RowState, StoredRowBatch, VisibleRowEntry,
};
use jazz_tools::storage::{
    HistoryRowBytes, IndexMutation, MemoryStorage, OwnedHistoryRowBytes, OwnedVisibleRowBytes,
    RawTableMutation, RawTableRows, RowLocator, Storage, StorageError, VisibleRowBytes,
};
use jazz_tools::sync_manager::{DurabilityTier, SyncManager};
use jazz_tools::test_support::seeded_memory_storage;

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
// Include workload: `users`, each carrying an include of their `posts`.
// ============================================================================

/// Outer-row counts. Both sit below `MAX_CACHED_SUBGRAPHS` (2048), so the whole
/// result set is instance-cached in both runs — the gate measures the eager
/// walk, not the cache's eviction backstop.
const OUTER_LOW: usize = 50;
const OUTER_HIGH: usize = 1_000;

/// Allowed growth from 20x the instances. Same slack as the depth gates.
const FLATNESS_TOLERANCE_PERCENT: u64 = 120;

fn users_posts_schema() -> Schema {
    let mut schema = Schema::new();
    schema.insert(
        TableName::new("users"),
        RowDescriptor::new(vec![
            ColumnDescriptor::new("id", ColumnType::Integer),
            ColumnDescriptor::new("name", ColumnType::Text),
        ])
        .into(),
    );
    schema.insert(
        TableName::new("posts"),
        RowDescriptor::new(vec![
            ColumnDescriptor::new("id", ColumnType::Integer),
            ColumnDescriptor::new("title", ColumnType::Text),
            ColumnDescriptor::new("author_id", ColumnType::Integer),
        ])
        .into(),
    );
    schema
}

/// Work on the two axes the gates measure, kept apart on purpose: a sum lets a
/// flat read count hide a linear allocation.
#[derive(Clone, Copy)]
struct Work {
    reads: u64,
    bytes: u64,
}

fn insert_row(
    qm: &mut QueryManager,
    storage: &mut CountingStorage,
    schema: &Schema,
    branch: &str,
    table: &str,
    values: &[Value],
) {
    qm.insert_on_branch_with_schema_and_write_context_and_id(
        storage, table, branch, values, None, schema, None, true,
    )
    .unwrap_or_else(|error| panic!("insert into {table} failed: {error:?}"));
}

/// The two halves of one write, measured apart because they regressed apart.
///
/// `write` is the MARKING path — everything `insert` does before control
/// returns, including routing dirt at the graph nodes. `settle` is the
/// `process` pass that turns that dirt into deltas. v13-2 blew up the first;
/// the second is where the instance walk still lives.
#[derive(Clone, Copy)]
struct Measured {
    write: Work,
    settle: Work,
}

impl Measured {
    fn total(&self) -> Work {
        Work {
            reads: self.write.reads + self.settle.reads,
            bytes: self.write.bytes + self.settle.bytes,
        }
    }
}

/// Assert one measured phase does not scale with the instance count, on BOTH
/// axes independently — a sum would let a flat read count hide a linear
/// allocation, which is exactly how the deep clone in `reevaluate_all`
/// survived every existing gate.
fn assert_flat(phase: &str, low: Work, high: Work) {
    let read_budget = low.reads * FLATNESS_TOLERANCE_PERCENT / 100;
    let byte_budget = low.bytes * FLATNESS_TOLERANCE_PERCENT / 100;
    assert!(
        high.reads <= read_budget,
        "{phase}: one single-row write into an include's inner table costs {} storage reads \
         with {OUTER_HIGH} cached instances but {} with {OUTER_LOW} (budget {read_budget}) — \
         work scales with the number of cached subgraph instances, not with the change",
        high.reads,
        low.reads,
    );
    assert!(
        high.bytes <= byte_budget,
        "{phase}: one single-row write into an include's inner table allocates {} bytes \
         with {OUTER_HIGH} cached instances but {} with {OUTER_LOW} (budget {byte_budget}) — \
         work scales with the number of cached subgraph instances, not with the change",
        high.bytes,
        low.bytes,
    );
}

fn report(label: &str, outer_rows: usize, measured: Measured) {
    let total = measured.total();
    eprintln!(
        "{label} @ {outer_rows:>4} instances | write {:>5} reads {:>9} bytes | \
         settle {:>5} reads {:>9} bytes | total {:>5} reads {:>9} bytes",
        measured.write.reads,
        measured.write.bytes,
        measured.settle.reads,
        measured.settle.bytes,
        total.reads,
        total.bytes,
    );
}

/// Which subscription shape a measurement runs against.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Shape {
    /// The gate's shape: a required include, with the OUTER query windowed
    /// to one row. See [`include_inner_write_is_flat_in_cached_instance_count`] for
    /// why the window is what isolates the axis.
    Windowed,
    /// The include with no window: every outer row is also a DELIVERED row.
    /// Reported, never asserted — see
    /// [`unwindowed_write_measures_the_delivery_diff_not_the_include`].
    Unwindowed,
    /// The same unwindowed subscription with NO include at all, and the write
    /// aimed at the outer table so one delivered row still moves. The control
    /// that attributes `Unwindowed`'s residual slope.
    UnwindowedWithoutInclude,
}

/// Instantiate `outer_rows` include instances, then measure ONE single-row
/// write into the include's inner table plus the settle it triggers.
fn measure_single_inner_write(outer_rows: usize) -> Measured {
    measure_shape(outer_rows, Shape::Windowed)
}

fn measure_shape(outer_rows: usize, shape: Shape) -> Measured {
    let mut qm = QueryManager::new(SyncManager::new());
    qm.set_current_schema(users_posts_schema(), "dev", "main");
    let schema = qm.schema_context().current_schema.clone();
    let branch = qm.schema_context().branch_name().as_str().to_string();
    let mut storage = CountingStorage::new(seeded_memory_storage(&schema));

    // One user per outer row, each already carrying one post: every outer row
    // gets a live, settled subgraph instance holding a row.
    for index in 0..outer_rows {
        let user_id = index as i32 + 1;
        insert_row(
            &mut qm,
            &mut storage,
            &schema,
            &branch,
            "users",
            &[
                Value::Integer(user_id),
                Value::Text(format!("user-{index}")),
            ],
        );
        insert_row(
            &mut qm,
            &mut storage,
            &schema,
            &branch,
            "posts",
            &[
                Value::Integer(1_000_000 + user_id),
                Value::Text(format!("seed post {index}")),
                Value::Integer(user_id),
            ],
        );
    }

    let builder = qm.query("users");
    let builder = match shape {
        Shape::UnwindowedWithoutInclude => builder,
        // The windowed include is REQUIRED: whether an outer row survives
        // depends on its include, so the plan has to build one for every
        // outer row before it can take the window. An optional include under
        // a limit is built for the window's rows only (`compile.rs` attaches
        // it after the limit), which would leave one instance for any
        // `outer_rows`. Every seeded user carries a post, so the requirement
        // filters nothing out.
        Shape::Windowed => builder.with_array("posts", |sub| {
            sub.from("posts")
                .correlate("author_id", "users.id")
                .require_result()
        }),
        Shape::Unwindowed => builder.with_array("posts", |sub| {
            sub.from("posts").correlate("author_id", "users.id")
        }),
    };
    let query = match shape {
        // `order_by` + `limit` land ABOVE a required include in the compiled
        // plan, so every outer row still gets its own cached subgraph instance
        // while only one row is delivered. The live-instance assertion below is
        // what keeps that true if the plan ever changes.
        Shape::Windowed => builder.order_by("id").limit(1).build(),
        Shape::Unwindowed | Shape::UnwindowedWithoutInclude => builder.build(),
    };
    qm.subscribe(query).expect("subscribe to the include query");
    qm.process(&mut storage);
    let updates = qm.take_updates();
    let delivered: usize = updates.iter().map(|update| update.delta.added.len()).sum();
    let expected_delivered = match shape {
        Shape::Windowed => 1,
        Shape::Unwindowed | Shape::UnwindowedWithoutInclude => outer_rows,
    };
    assert_eq!(
        delivered, expected_delivered,
        "the subscription must carry its whole window before measuring"
    );
    if shape != Shape::UnwindowedWithoutInclude {
        assert_eq!(
            SettleCounts::snapshot().live_instances,
            outer_rows as u64,
            "every outer row must hold a cached subgraph instance — the axis this gate \
             measures is the INSTANCE count, and a plan that instantiated only the delivered \
             window would make it vacuous"
        );
    }

    // ── measured window: one row write + the settle it causes ──
    storage.reset();
    let bytes_before = total_allocated();

    match shape {
        // The control has no include, so it writes the OUTER table instead:
        // one delivered row still moves, and everything the settle then pays
        // is the manager's delivery diff.
        Shape::UnwindowedWithoutInclude => insert_row(
            &mut qm,
            &mut storage,
            &schema,
            &branch,
            "users",
            &[
                Value::Integer(3_000_000),
                Value::Text("the one measured user".into()),
            ],
        ),
        Shape::Windowed | Shape::Unwindowed => insert_row(
            &mut qm,
            &mut storage,
            &schema,
            &branch,
            "posts",
            &[
                Value::Integer(2_000_000),
                Value::Text("the one measured post".into()),
                Value::Integer(1), // author_id = the first user only
            ],
        ),
    }
    let write_work = Work {
        reads: storage.reads(),
        bytes: total_allocated() - bytes_before,
    };
    let settle_start_reads = storage.reads();
    let settle_start_bytes = total_allocated();
    qm.process(&mut storage);
    let settle_work = Work {
        reads: storage.reads() - settle_start_reads,
        bytes: total_allocated() - settle_start_bytes,
    };

    // The write must actually have reached the output — a gate over a settle
    // that did nothing would be flat for the wrong reason. Exactly one outer
    // row gained a post, so exactly one output row may move, at BOTH instance
    // counts: the delivered work is flat by construction, which is what makes
    // any growth in the measured work pure overhead.
    let updates = qm.take_updates();
    let changed: usize = updates
        .iter()
        .map(|update| update.delta.added.len() + update.delta.updated.len())
        .sum();
    assert_eq!(
        changed, 1,
        "exactly one output row must move, at both instance counts"
    );

    Measured {
        write: write_work,
        settle: settle_work,
    }
}

/// The marking half: what the write itself pays before it returns.
///
/// This is the axis v13-2 broke — every mark was threaded into every cached
/// subgraph instance inline, so a write paid O(live outer rows) before it
/// returned, for every table the subscription touched. The buffer in
/// `PendingInnerDirt` plus the table scoping in `QueryGraph::mark_rows_*`
/// makes it O(1) in the instance count, and this gate is what holds it there.
#[test]
fn include_inner_write_marking_is_flat_in_cached_instance_count() {
    let _serialised = measure_lock();
    let _precise = force_precise_dirty(true);

    let low = measure_single_inner_write(OUTER_LOW);
    let high = measure_single_inner_write(OUTER_HIGH);
    report("marking", OUTER_LOW, low);
    report("marking", OUTER_HIGH, high);

    assert_flat("write-phase (marking)", low.write, high.write);
}

/// The whole write: marking plus the settle it triggers.
///
/// One row entered one instance's array; 999 of the 1000 cached instances
/// cannot contain it. The work to establish that must not grow with the
/// instance count — which is what correlation-routed dirt buys (v14 L1): the
/// changed row's correlate is resolved once per settle against the instance
/// bindings, and only the instances that can hold it are marked, so the rest
/// stay clean and `reevaluate_all` skips them.
///
/// WHY THE OUTER QUERY IS WINDOWED. The fixture used to subscribe to all
/// `outer_rows` and therefore varied TWO axes at once: the cached instance
/// count AND the delivered result set. The second one carries a cost of its
/// own — the manager re-derives its client mirror from the full visible tuple
/// set on every settle (`rows_from_tuples` + the row-by-row diff in
/// `row_delta_from_rows`), which is O(delivered rows) whether or not an
/// include exists at all. `unwindowed_write_measures_the_delivery_diff_not_the_include`
/// measures exactly that, WITHOUT an include in the graph, and it is the same
/// slope. Holding the window at one row leaves the instance count as the only
/// variable, which is what this gate's own doc header says it measures.
#[test]
fn include_inner_write_is_flat_in_cached_instance_count() {
    let _serialised = measure_lock();
    let _precise = force_precise_dirty(true);

    let low = measure_single_inner_write(OUTER_LOW);
    let high = measure_single_inner_write(OUTER_HIGH);
    report("precise", OUTER_LOW, low);
    report("precise", OUTER_HIGH, high);

    assert_flat("settle-phase", low.settle, high.settle);
    assert_flat("write + settle", low.total(), high.total());
}

/// The same measurement with precise dirtiness off, for comparison only.
///
/// Legacy is coarse — one inner-table write re-evaluates every instance
/// unconditionally — so it is expected to scale here and carries no flatness
/// assertion. It is measured so a regression report can name all three states
/// (legacy / precise-marking / precise-settle) from one run.
#[test]
fn legacy_dirtiness_instance_axis_baseline() {
    let _serialised = measure_lock();
    let _precise = force_precise_dirty(false);

    let low = measure_single_inner_write(OUTER_LOW);
    let high = measure_single_inner_write(OUTER_HIGH);
    report("legacy ", OUTER_LOW, low);
    report("legacy ", OUTER_HIGH, high);
}

/// Attribution for the residual, so the window in the gate above is a measured
/// decision rather than a convenient one.
///
/// Same write, same instance counts, but the outer query is UNWINDOWED, so
/// every outer row is also a delivered row. Then the control: the identical
/// unwindowed subscription with NO include in the graph, writing the outer
/// table so one delivered row still moves. If the residual per-row cost were
/// include work, the control would be flat. It is not — the two slopes match,
/// because both are the manager's per-settle delivery diff over the visible
/// set. Reported, never asserted: making THAT flat is incremental delivery,
/// a different lever from correlation routing.
#[test]
fn unwindowed_write_measures_the_delivery_diff_not_the_include() {
    let _serialised = measure_lock();
    let _precise = force_precise_dirty(true);

    let slope = |low: Measured, high: Measured| -> f64 {
        (high.settle.bytes as f64 - low.settle.bytes as f64) / (OUTER_HIGH - OUTER_LOW) as f64
    };

    let include_low = measure_shape(OUTER_LOW, Shape::Unwindowed);
    let include_high = measure_shape(OUTER_HIGH, Shape::Unwindowed);
    report("unwindowed", OUTER_LOW, include_low);
    report("unwindowed", OUTER_HIGH, include_high);

    let control_low = measure_shape(OUTER_LOW, Shape::UnwindowedWithoutInclude);
    let control_high = measure_shape(OUTER_HIGH, Shape::UnwindowedWithoutInclude);
    report("no include", OUTER_LOW, control_low);
    report("no include", OUTER_HIGH, control_high);

    eprintln!(
        "settle byte slope per delivered row: with include {:.0} B, without include {:.0} B \
         — the residual is the delivery diff, not the include",
        slope(include_low, include_high),
        slope(control_low, control_high),
    );
}
