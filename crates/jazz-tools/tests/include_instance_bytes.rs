//! What ONE include (array-subquery) instance costs, in bytes, split by part.
//!
//! # Why this exists
//!
//! The v14 design (`2026-08-v14-include-cost.md` §1) sizes the include problem
//! from two production readings: a settle pass reports ~23 800 cached-subquery
//! EVALUATIONS, and the server holds ~1.5 GB. Dividing one by the other gives
//! "~60 KB per instance" — but only if evaluations and live instances are the
//! same population, which is an assumption, and only if the whole 1.5 GB is
//! instances, which is another. Both halves of the decision to build shared
//! compiled plans (L2) rest on that division.
//!
//! This is the measurement that replaces it. The live-instance half is the
//! `live_instances` gauge now carried in the `jazz::settle_cost` line; this
//! file is the bytes half: the allocator delta of building ONE subgraph
//! instance and of evaluating it once, on the REAL production schema fixture,
//! for a leaf include and for one that nests another include inside it.
//!
//! # What is measured, and what each part means for L2
//!
//! | part | measured as | can shared plans remove it? |
//! |---|---|---|
//! | bound query | building the correlated `Query` | CHURN, not residency — see below |
//! | compiled plan | `QueryGraph::try_compile_*` on it | partly; the node array stays |
//! | ↳ node array | `nodes.capacity() * size_of::<CompactNode>()` | only its shape half, not its state half |
//! | ↳ descriptors | extra handle vs deep copy | already shared behind `Arc` |
//! | populated state | the instance's first full settle | NO — this is the floor |
//!
//! The bound query is built inside `instantiate` and dropped before it
//! returns, so it costs allocator churn per instantiation and NOTHING in
//! residency. `instantiate` therefore retains exactly the compiled plan, and
//! this file asserts that equality: it is what says the split is complete
//! rather than convenient.
//!
//! # This is an instrument, not a gate
//!
//! The only assertions are non-zero-ness and internal consistency. A byte
//! budget here would be wrong: the numbers exist to be read and compared
//! against production, and pinning them would turn every legitimate change to
//! the compiler into a failure with no diagnosis attached.
//!
//! # Known lower bound
//!
//! The fixture carries columns only, no policy bundle (same caveat as
//! `linsa_schema_profile.rs`), so instances compile with no `PolicyFilter` or
//! `MagicColumns` nodes and no session. Production compiles those in. Every
//! number below is therefore a FLOOR for the production per-instance cost.

#![cfg(feature = "test")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};

use jazz_tools::object::{BranchName, ObjectId};
use jazz_tools::query_manager::graph::{CompactNode, GraphNode, QueryGraph};
use jazz_tools::query_manager::graph_nodes::array_subquery::ArraySubqueryNode;
use jazz_tools::query_manager::graph_nodes::subgraph::{SubgraphInstance, SubgraphTemplate};
use jazz_tools::query_manager::index_declarations::IndexDeclarations;
use jazz_tools::query_manager::manager::QueryManager;
use jazz_tools::query_manager::query::{Query, QueryBuilder};
use jazz_tools::query_manager::settle_cost::{LIVE_SUBQUERY_INSTANCES, LIVE_SUBQUERY_NODES};
use jazz_tools::query_manager::types::{
    ColumnType, LoadedRow, RowDescriptor, RowPolicyMode, Schema, TableName, TupleProvenance, Value,
};
use jazz_tools::schema_manager::SchemaContext;
use jazz_tools::storage::{MemoryStorage, Storage};
use jazz_tools::sync_manager::SyncManager;
use jazz_tools::test_support::seeded_memory_storage;

// ============================================================================
// Tracking allocator: LIVE (retained) bytes, so a delta is residency
// ============================================================================

struct TrackingAllocator;

static LIVE_BYTES: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            LIVE_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        LIVE_BYTES.fetch_sub(layout.size() as u64, Ordering::Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            LIVE_BYTES.fetch_sub(layout.size() as u64, Ordering::Relaxed);
            LIVE_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        }
        new_ptr
    }
}

#[global_allocator]
static GLOBAL: TrackingAllocator = TrackingAllocator;

fn live_bytes() -> i64 {
    LIVE_BYTES.load(Ordering::Relaxed) as i64
}

/// Bytes still held after `body` returned, with its result kept alive.
///
/// Transient allocations freed inside `body` do not count, which is the point:
/// the question is what an instance WEIGHS, not what building it churned.
fn retained<T>(body: impl FnOnce() -> T) -> (i64, T) {
    let before = live_bytes();
    let value = body();
    (live_bytes() - before, value)
}

/// Serialises the measuring tests: the allocator counter and the instance
/// gauge are both process-global.
fn measure_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
}

// ============================================================================
// The production schema fixture and a store seeded through it
// ============================================================================

/// The production wire schema, same fixture and same envelope path as
/// `linsa_schema_profile.rs`.
fn linsa_schema() -> Schema {
    let envelope: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/linsa-schema.json"))
            .expect("envelope must parse");
    serde_json::from_value(envelope["schema"]["users"]["_schema"].clone())
        .expect("production wire schema must parse")
}

/// Messages seeded; instance 0 is the warm-up, instance 1 is measured.
const MESSAGES: usize = 3;
/// Attachments per message — the fan-out of the level-1 include, hence the
/// number of NESTED instances one level-1 instance spawns.
const ATTACHMENTS_PER_MESSAGE: usize = 2;
/// Variant rows per media asset — the rows a leaf instance materialises.
const VARIANTS_PER_ASSET: usize = 3;

struct Fixture {
    storage: MemoryStorage,
    schema: Arc<Schema>,
    schema_context: Arc<SchemaContext>,
    /// One shared set, as a query manager holds it: an allocation inside a measured
    /// compile would be charged to the plan.
    declarations: Arc<IndexDeclarations>,
    branch: String,
    message_ids: Vec<ObjectId>,
    asset_ids: Vec<ObjectId>,
}

impl Fixture {
    fn seed() -> Self {
        let mut query_manager = QueryManager::new(SyncManager::new());
        query_manager.set_current_schema(linsa_schema(), "dev", "main");
        let schema_context = query_manager.schema_context().clone();
        let schema = schema_context.current_schema.clone();
        let branch = schema_context.branch_name().as_str().to_string();
        let mut storage = seeded_memory_storage(&schema);

        let chat_id = ObjectId::new();
        let owner_id = ObjectId::new();
        let mut message_ids = Vec::new();
        let mut asset_ids = Vec::new();

        for message_index in 0..MESSAGES {
            let message_id = insert(
                &mut query_manager,
                &mut storage,
                &schema,
                &branch,
                "messages",
                &[
                    ("chatId", Value::Uuid(chat_id)),
                    ("senderKind", Value::Text("user".into())),
                    ("createdAtMs", Value::Timestamp(1000 + message_index as u64)),
                    ("isDeleted", Value::Boolean(false)),
                ],
            );
            message_ids.push(message_id);

            for position in 0..ATTACHMENTS_PER_MESSAGE {
                let asset_id = insert(
                    &mut query_manager,
                    &mut storage,
                    &schema,
                    &branch,
                    "media_assets",
                    &[("ownerUserId", Value::Uuid(owner_id))],
                );
                insert(
                    &mut query_manager,
                    &mut storage,
                    &schema,
                    &branch,
                    "message_attachments",
                    &[
                        ("messageId", Value::Uuid(message_id)),
                        ("mediaAssetId", Value::Uuid(asset_id)),
                        ("position", Value::Integer(position as i32)),
                    ],
                );
                for variant in 0..VARIANTS_PER_ASSET {
                    insert(
                        &mut query_manager,
                        &mut storage,
                        &schema,
                        &branch,
                        "media_asset_variants",
                        &[
                            ("mediaAssetId", Value::Uuid(asset_id)),
                            ("createdAtMs", Value::Timestamp(2000 + variant as u64)),
                        ],
                    );
                }
                asset_ids.push(asset_id);
            }
        }

        Self {
            storage,
            schema: Arc::new(schema),
            schema_context: Arc::new(schema_context),
            declarations: Arc::new(IndexDeclarations::empty()),
            branch,
            message_ids,
            asset_ids,
        }
    }

    /// The row loader a subgraph settle needs, in the minimal form the
    /// production loader reduces to for a single-branch local store.
    fn row_loader(&self) -> impl FnMut(ObjectId, Option<TableName>) -> Option<LoadedRow> + '_ {
        move |row_id, table_hint| {
            let table = match table_hint {
                Some(table) => table.as_str().to_string(),
                None => self
                    .storage
                    .load_row_locator(row_id)
                    .ok()
                    .flatten()?
                    .table
                    .to_string(),
            };
            let row = self
                .storage
                .load_visible_query_row(&table, &self.branch, row_id)
                .ok()
                .flatten()?;
            if !row.state.is_visible() {
                return None;
            }
            let mut provenance = TupleProvenance::default();
            provenance.insert((row_id, BranchName::new(self.branch.clone())));
            Some(LoadedRow::new(
                row.data.clone(),
                row.row_provenance(),
                provenance,
                row.batch_id,
            ))
        }
    }
}

/// Insert one row, filling every column the caller did not name with a
/// type-appropriate placeholder (`Null` where the schema allows it).
fn insert(
    query_manager: &mut QueryManager,
    storage: &mut MemoryStorage,
    schema: &Schema,
    branch: &str,
    table: &str,
    provided: &[(&str, Value)],
) -> ObjectId {
    let descriptor = schema
        .get(&TableName::new(table))
        .unwrap_or_else(|| panic!("table {table} in the production schema"))
        .columns
        .clone();
    let values: Vec<Value> = descriptor
        .columns
        .iter()
        .map(|column| {
            if let Some((_, value)) = provided
                .iter()
                .find(|(name, _)| *name == column.name.as_str())
            {
                return value.clone();
            }
            if column.nullable {
                return Value::Null;
            }
            match &column.column_type {
                ColumnType::Text => Value::Text("mock".into()),
                ColumnType::Enum { variants } => {
                    Value::Text(variants.first().cloned().unwrap_or_default())
                }
                ColumnType::Integer => Value::Integer(1),
                ColumnType::BigInt => Value::BigInt(1),
                ColumnType::Double => Value::Double(1.0),
                ColumnType::Boolean => Value::Boolean(false),
                ColumnType::Timestamp => Value::Timestamp(1),
                ColumnType::Uuid => Value::Uuid(ObjectId::new()),
                ColumnType::Bytea => Value::Bytea(vec![0u8; 4]),
                ColumnType::Array { .. } => Value::Array(Vec::new()),
                other => panic!(
                    "unhandled required column {table}.{}: {other:?}",
                    column.name
                ),
            }
        })
        .collect();

    query_manager
        .insert_on_branch_with_schema_and_write_context_and_id(
            storage, table, branch, &values, None, schema, None, true,
        )
        .unwrap_or_else(|error| panic!("insert into {table} failed: {error:?}"))
        .row_id
}

// ============================================================================
// The two include shapes, taken from the production thread view
// ============================================================================

/// One include shape: how to build its template and how to build the same
/// bound query by hand, so `instantiate` can be decomposed.
struct Shape {
    label: &'static str,
    table: &'static str,
    inner_column: &'static str,
    /// True when the shape carries another include inside it.
    nested: bool,
}

/// The leaf of the production thread view: an attachment's media variants.
const LEAF: Shape = Shape {
    label: "leaf   (media_asset_variants)",
    table: "media_asset_variants",
    inner_column: "mediaAssetId",
    nested: false,
};

/// One level up: a message's attachments, each carrying its variants include.
const NESTING: Shape = Shape {
    label: "nesting(message_attachments + variants)",
    table: "message_attachments",
    inner_column: "messageId",
    nested: true,
};

impl Shape {
    /// The base query a `SubgraphTemplate` is built from — the inner query
    /// WITHOUT the correlation filter, exactly as the compiler produces it for
    /// the corresponding `with_array` in the thread subscription.
    fn base_query(&self, branch: &str) -> Query {
        let builder = QueryBuilder::new(self.table).branches(&[branch]);
        if self.nested {
            builder
                .order_by("position")
                .with_array("variants", |variants| {
                    variants
                        .from("media_asset_variants")
                        .correlate("mediaAssetId", "message_attachments.mediaAssetId")
                        .order_by("createdAtMs")
                })
                .build()
        } else {
            builder.order_by("createdAtMs").build()
        }
    }

    /// The query `SubgraphTemplate::instantiate` builds internally, mirrored
    /// here so its cost can be measured apart from the compile. The mirror is
    /// validated by `instantiate ≈ bound query + compile` in the report.
    fn bound_query(&self, branch: &str, correlation: &Value) -> Query {
        let builder = QueryBuilder::new(self.table)
            .branches(&[branch])
            .filter_eq(self.inner_column, correlation.clone());
        if self.nested {
            builder
                .order_by("position")
                .with_array("variants", |variants| {
                    variants
                        .from("media_asset_variants")
                        .correlate("mediaAssetId", "message_attachments.mediaAssetId")
                        .order_by("createdAtMs")
                })
                .build()
        } else {
            builder.order_by("createdAtMs").build()
        }
    }

    fn template(&self, fixture: &Fixture) -> SubgraphTemplate {
        let output_descriptor = fixture
            .schema
            .get(&TableName::new(self.table))
            .expect("include table in the production schema")
            .columns
            .clone();
        SubgraphTemplate::new(
            self.base_query(&fixture.branch),
            self.inner_column.to_string(),
            Vec::new(),
            output_descriptor,
            Arc::clone(&fixture.schema_context),
            None,
            RowPolicyMode::PermissiveLocal,
            Arc::clone(&fixture.declarations),
        )
    }
}

// ============================================================================
// One shape's measurement
// ============================================================================

#[derive(Debug)]
struct Split {
    /// The correlated `Query` value: relation IR, filters, nested specs.
    /// Transient — `instantiate` drops it before returning.
    bound_query: i64,
    /// The compiled `QueryGraph` built from it: nodes, plans, descriptors.
    compiled_plan: i64,
    /// `SubgraphTemplate::instantiate` end to end — must equal the compiled
    /// plan, since that is all a `SubgraphInstance` retains.
    instantiate: i64,
    /// The dense node array inside the compiled plan. Per-instance by
    /// construction: it holds each node's mutable state inline.
    node_array: i64,
    /// What one extra handle on the graph's descriptors costs TODAY.
    descriptor_handle: i64,
    /// What those same descriptors would cost unshared.
    descriptor_deep_copy: i64,
    /// Retained after the instance's first full evaluation.
    populated_state: i64,
    nodes: usize,
    output_rows: usize,
    /// Cached subgraph instances the evaluation left live INSIDE this one.
    nested_instances: u64,
}

impl Split {
    fn total(&self) -> i64 {
        self.instantiate + self.populated_state
    }
}

fn measure(shape: &Shape, fixture: &Fixture, correlation: Value) -> Split {
    let template = shape.template(fixture);

    // Two warm-ups, both dropped before the measured window opens.
    //
    // The first is on a DIFFERENT binding: it pays for whatever the first
    // instantiation of this shape leaks into process-wide state (string
    // interning, lazy statics, one-shot caches).
    //
    // The second is on the SAME binding that will be measured, and it is the
    // one that matters: anything the store itself retains for these particular
    // rows — decoded row caches, index structures — is created by a settle
    // over them, and would otherwise be counted as instance residency. After
    // it, the measured settle allocates only what the instance keeps.
    for binding in [warm_up_binding(shape, fixture), correlation.clone()] {
        if let Some(mut warm) = template.instantiate(binding, &fixture.schema) {
            let mut loader = fixture.row_loader();
            warm.graph.settle(&fixture.storage, &mut loader);
        }
    }

    let (bound_query_bytes, bound_query) =
        retained(|| shape.bound_query(&fixture.branch, &correlation));
    let (compile_bytes, graph) = retained(|| {
        QueryGraph::try_compile_with_schema_context_shared(
            &bound_query,
            &fixture.schema,
            None,
            &fixture.schema_context,
            RowPolicyMode::PermissiveLocal,
            &fixture.declarations,
        )
        .expect("the production include shape must compile")
    });

    let nodes = graph.nodes.len();
    let node_array = (graph.nodes.capacity() * std::mem::size_of::<CompactNode>()) as i64;
    let (descriptor_handle, handle) = retained(|| {
        (
            graph.table_descriptors.clone(),
            graph.combined_descriptor.clone(),
        )
    });
    drop(handle);
    let (descriptor_deep_copy, deep) = retained(|| {
        let tables: Vec<RowDescriptor> = graph
            .table_descriptors
            .iter()
            .map(|descriptor| RowDescriptor::new(descriptor.columns.to_vec()))
            .collect();
        (
            tables,
            RowDescriptor::new(graph.combined_descriptor.columns.to_vec()),
        )
    });
    drop(deep);
    drop(graph);
    drop(bound_query);

    // The real API, end to end, on the same binding.
    let instances_before = LIVE_SUBQUERY_INSTANCES.load(Ordering::Relaxed);
    let (instantiate_bytes, instance) = retained(|| {
        template
            .instantiate(correlation.clone(), &fixture.schema)
            .expect("the production include shape must instantiate")
    });
    let mut instance: SubgraphInstance = instance;

    let (populated_state, _) = retained(|| {
        let mut loader = fixture.row_loader();
        instance.graph.settle(&fixture.storage, &mut loader);
    });
    let output_rows = instance.graph.current_output_tuples_ref().len();
    let nested_instances = LIVE_SUBQUERY_INSTANCES.load(Ordering::Relaxed) - instances_before;

    Split {
        bound_query: bound_query_bytes,
        compiled_plan: compile_bytes,
        instantiate: instantiate_bytes,
        node_array,
        descriptor_handle,
        descriptor_deep_copy,
        populated_state,
        nodes,
        output_rows,
        nested_instances,
    }
}

fn warm_up_binding(shape: &Shape, fixture: &Fixture) -> Value {
    if shape.nested {
        Value::Uuid(fixture.message_ids[0])
    } else {
        Value::Uuid(fixture.asset_ids[0])
    }
}

fn measured_binding(shape: &Shape, fixture: &Fixture) -> Value {
    if shape.nested {
        Value::Uuid(fixture.message_ids[1])
    } else {
        Value::Uuid(fixture.asset_ids[2])
    }
}

fn report(shape: &Shape, split: &Split) {
    eprintln!("{}", shape.label);
    eprintln!(
        "  compiled plan      {:>9} B  ({} nodes; instantiate retains {} B — the bound query, \
         {} B, is churn)",
        split.compiled_plan, split.nodes, split.instantiate, split.bound_query,
    );
    eprintln!(
        "  ↳ node array       {:>9} B  ({:.0} % of the plan; descriptors: extra handle {} B, \
         unshared deep copy {} B)",
        split.node_array,
        100.0 * split.node_array as f64 / split.compiled_plan as f64,
        split.descriptor_handle,
        split.descriptor_deep_copy,
    );
    eprintln!(
        "  populated state    {:>9} B  ({} output rows, {} nested instances)",
        split.populated_state, split.output_rows, split.nested_instances,
    );
    eprintln!(
        "  TOTAL per instance {:>9} B  = {:.1} KiB, of which state is {:.0} %",
        split.total(),
        split.total() as f64 / 1024.0,
        100.0 * split.populated_state as f64 / split.total() as f64,
    );
}

// ============================================================================
// The measurement
// ============================================================================

/// Print the per-instance byte split for a leaf include and for one that nests
/// another include inside it, on the real production schema.
#[test]
fn include_instance_byte_split_on_the_production_schema() {
    let _serialised = measure_lock();
    let fixture = Fixture::seed();

    let leaf = measure(&LEAF, &fixture, measured_binding(&LEAF, &fixture));
    let nesting = measure(&NESTING, &fixture, measured_binding(&NESTING, &fixture));

    eprintln!("\n=== per-include-instance bytes, production schema, no policy bundle ===");
    // `GraphNode` is an enum stored inline in a dense `Vec`, so EVERY node slot
    // is sized for the largest variant. That is why the node array is the bulk
    // of a compiled plan, and it is a different lever from plan sharing: most
    // of those bytes are slack, not per-instance data.
    eprintln!(
        "node slot: CompactNode {} B (GraphNode {} B, largest variant ArraySubqueryNode {} B)",
        std::mem::size_of::<CompactNode>(),
        std::mem::size_of::<GraphNode>(),
        std::mem::size_of::<ArraySubqueryNode>(),
    );
    report(&LEAF, &leaf);
    report(&NESTING, &nesting);

    // What a level-1 instance costs including everything it drags in, versus
    // what it would cost if its nested instances were free. The share is a
    // function of the include's fan-out — here `ATTACHMENTS_PER_MESSAGE` — so
    // it is reported with the fan-out that produced it, not as a constant.
    let nested_share = leaf.total() * nesting.nested_instances as i64;
    eprintln!(
        "\nnested share at fan-out {}: {} nested x {} B = {} B of {} B ({:.0} %); \
         the level-1 instance's own cost is {} B",
        nesting.nested_instances,
        nesting.nested_instances,
        leaf.total(),
        nested_share,
        nesting.total(),
        100.0 * nested_share as f64 / nesting.total() as f64,
        nesting.total() - nested_share,
    );
    eprintln!(
        "L2 ceiling — plan bytes outside the per-instance node array: leaf {} B ({:.0} % of the \
         instance), nesting {} B ({:.0} %); plus {} B / {} B of per-instantiation query churn",
        leaf.compiled_plan - leaf.node_array,
        100.0 * (leaf.compiled_plan - leaf.node_array) as f64 / leaf.total() as f64,
        nesting.compiled_plan - nesting.node_array,
        100.0 * (nesting.compiled_plan - nesting.node_array) as f64 / nesting.total() as f64,
        leaf.bound_query,
        nesting.bound_query,
    );
    eprintln!(
        "L2 floor — populated state alone: leaf {} B ({:.0} %), nesting {} B ({:.0} %); the node \
         array ({} B / {} B) is part shape, part state and splits between the two\n",
        leaf.populated_state,
        100.0 * leaf.populated_state as f64 / leaf.total() as f64,
        nesting.populated_state,
        100.0 * nesting.populated_state as f64 / nesting.total() as f64,
        leaf.node_array,
        nesting.node_array,
    );

    // ── consistency, not budgets ────────────────────────────────────────────
    for (shape, split) in [(&LEAF, &leaf), (&NESTING, &nesting)] {
        assert!(
            split.bound_query > 0 && split.compiled_plan > 0 && split.instantiate > 0,
            "{}: building an instance must retain bytes: {split:?}",
            shape.label,
        );
        assert!(
            split.populated_state > 0,
            "{}: evaluating an instance that materialises {} rows must retain state — \
             a zero here means the settle found nothing and the measurement is empty",
            shape.label,
            split.output_rows,
        );
        assert!(
            split.output_rows > 0,
            "{}: the measured instance materialised no rows",
            shape.label,
        );
        // A `SubgraphInstance` retains its compiled graph and nothing else, so
        // the separately measured compile must BE the instantiation. Any drift
        // is residency the split failed to name.
        assert_eq!(
            split.compiled_plan, split.instantiate,
            "{}: instantiate retains {} B but compiling the same bound query retains {} B — \
             the split is missing whatever the difference is",
            shape.label, split.instantiate, split.compiled_plan,
        );
        assert!(
            split.node_array > 0 && split.node_array <= split.compiled_plan,
            "{}: the node array is part of the compiled plan: {split:?}",
            shape.label,
        );
        assert!(
            split.descriptor_handle <= split.descriptor_deep_copy,
            "{}: an extra descriptor handle cannot cost more than an unshared copy",
            shape.label,
        );
    }

    assert_eq!(
        leaf.nested_instances, 0,
        "a leaf include has nothing to nest"
    );
    assert_eq!(
        nesting.nested_instances, ATTACHMENTS_PER_MESSAGE as u64,
        "one nested instance per attachment row the level-1 instance holds"
    );
    assert_eq!(
        leaf.output_rows, VARIANTS_PER_ASSET,
        "the leaf instance holds every variant of its asset"
    );
    assert_eq!(
        nesting.output_rows, ATTACHMENTS_PER_MESSAGE,
        "the level-1 instance holds every attachment of its message"
    );
}

/// The gauge counts what it says it counts: instances appear as includes are
/// instantiated and are gone when the holder is dropped.
///
/// A gauge that only leaks upward would still look plausible in a production
/// log line — it would just quietly overstate the instance population, which is
/// the exact number the v14 sizing rests on.
#[test]
fn live_instance_gauge_returns_to_its_baseline() {
    let _serialised = measure_lock();
    let fixture = Fixture::seed();

    let instances_before = LIVE_SUBQUERY_INSTANCES.load(Ordering::Relaxed);
    let nodes_before = LIVE_SUBQUERY_NODES.load(Ordering::Relaxed);

    {
        let template = NESTING.template(&fixture);
        let mut instance = template
            .instantiate(Value::Uuid(fixture.message_ids[0]), &fixture.schema)
            .expect("instantiate the nesting shape");
        assert_eq!(
            LIVE_SUBQUERY_NODES.load(Ordering::Relaxed) - nodes_before,
            1,
            "the compiled plan of a nesting include carries exactly one include node"
        );

        let mut loader = fixture.row_loader();
        instance.graph.settle(&fixture.storage, &mut loader);
        assert_eq!(
            LIVE_SUBQUERY_INSTANCES.load(Ordering::Relaxed) - instances_before,
            ATTACHMENTS_PER_MESSAGE as u64,
            "evaluating it caches one nested instance per attachment row"
        );
    }

    assert_eq!(
        LIVE_SUBQUERY_INSTANCES.load(Ordering::Relaxed),
        instances_before,
        "dropping the holder must release every instance it counted"
    );
    assert_eq!(
        LIVE_SUBQUERY_NODES.load(Ordering::Relaxed),
        nodes_before,
        "dropping the holder must release its include nodes"
    );
}
