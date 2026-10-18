//! Storage conformance test suite for the row-history/raw-table storage model.

use std::collections::HashMap;

use std::ops::Bound;

use crate::batch_fate::{
    BatchFate, BatchMode, CapturedFrontierMember, LocalBatchMember, LocalBatchRecord,
    SealedBatchMember, SealedBatchSubmission,
};
use crate::catalogue::CatalogueEntry;
use crate::digest::Digest32;
use crate::metadata::{MetadataKey, ObjectType, RowProvenance};
use crate::object::ObjectId;
use crate::query_manager::types::{
    ColumnType, RowDescriptor, SchemaBuilder, SchemaHash, TableSchema, Value,
};
use crate::row_format::encode_row;
use crate::row_histories::{
    BatchId, HistoryScan, RowState, StoredRowBatch, VisibleRowEntry, decode_flat_history_row,
    decode_flat_visible_row_entry,
};
use crate::schema_manager::encoding::encode_schema;
use crate::storage::{
    IndexMutation, RawTableHeader, RowLocator, RowRawTableId, RowRawTableKind, Storage,
    StorageError, scan_row_raw_table_headers_with_storage,
};
use crate::sync_manager::DurabilityTier;
use crate::test_support::persist_test_schema;

/// Factory type for persistence tests that reopen storage at a given path.
pub type PersistentStorageFactory = dyn Fn(&std::path::Path) -> Box<dyn Storage>;

fn row_history_user_descriptor() -> RowDescriptor {
    RowDescriptor::new(vec![crate::query_manager::types::ColumnDescriptor::new(
        "value",
        ColumnType::Text,
    )])
}

fn row_history_test_schema(table: &str) -> crate::query_manager::types::Schema {
    SchemaBuilder::new()
        .table(TableSchema::builder(table).column("value", ColumnType::Text))
        .build()
}

fn seed_row_history_table(storage: &mut dyn Storage, table: &str) -> SchemaHash {
    let schema = row_history_test_schema(table);
    let schema_hash = persist_test_schema(storage, &schema);
    schema_hash
}

fn seed_row_history_locator(
    storage: &mut dyn Storage,
    table: &str,
    row_id: ObjectId,
    schema_hash: SchemaHash,
) {
    storage
        .put_row_locator(
            row_id,
            Some(&RowLocator {
                table: table.into(),
                origin_schema_hash: Some(schema_hash),
            }),
        )
        .unwrap();
}

fn make_row_batch(row_id: ObjectId, branch: &str, updated_at: u64, value: &str) -> StoredRowBatch {
    StoredRowBatch::new(
        row_id,
        branch,
        Vec::new(),
        encode_row(
            &row_history_user_descriptor(),
            &[Value::Text(value.to_string())],
        )
        .unwrap(),
        RowProvenance::for_insert(row_id.to_string(), updated_at),
        HashMap::new(),
        RowState::VisibleDirect,
        None,
    )
}

fn make_visible_entry(
    current_row: StoredRowBatch,
    history_rows: &[StoredRowBatch],
) -> VisibleRowEntry {
    VisibleRowEntry::rebuild(current_row, history_rows)
}

// ============================================================================
// Branch ord tests
// ============================================================================

pub fn test_row_raw_table_header_round_trip(factory: &dyn Fn() -> Box<dyn Storage>) {
    let mut storage = factory();
    let schema = row_history_test_schema("todos");
    let schema_hash = SchemaHash::compute(&schema);
    let user_descriptor = schema[&"todos".into()].columns.clone();
    let row_raw_table_id = RowRawTableId::new(RowRawTableKind::Visible, "todos", schema_hash);
    let header = RawTableHeader::row_raw_table(
        RowRawTableKind::Visible,
        "todos",
        schema_hash,
        &user_descriptor,
    );

    storage
        .upsert_raw_table_header(&row_raw_table_id.raw_table_name(), &header)
        .expect("row raw table header should persist");

    assert_eq!(
        storage
            .load_raw_table_header(&row_raw_table_id.raw_table_name())
            .unwrap(),
        Some(header.clone())
    );
    assert_eq!(
        scan_row_raw_table_headers_with_storage(&*storage).unwrap(),
        vec![(row_raw_table_id, header)]
    );
}

pub fn test_branch_ord_round_trip(factory: &dyn Fn() -> Box<dyn Storage>) {
    let mut storage = factory();
    let main = crate::object::BranchName::new("dev-aaaaaaaaaaaa-main");
    let draft = crate::object::BranchName::new("dev-bbbbbbbbbbbb-main");

    let main_ord = storage.resolve_or_alloc_branch_ord(main).unwrap();
    let draft_ord = storage.resolve_or_alloc_branch_ord(draft).unwrap();

    assert_eq!(storage.resolve_or_alloc_branch_ord(main).unwrap(), main_ord);
    assert_ne!(main_ord, draft_ord);
    assert_eq!(storage.load_branch_ord(main).unwrap(), Some(main_ord));
    assert_eq!(storage.load_branch_ord(draft).unwrap(), Some(draft_ord));
    assert_eq!(
        storage.load_branch_name_by_ord(main_ord).unwrap(),
        Some(main)
    );
    assert_eq!(
        storage.load_branch_name_by_ord(draft_ord).unwrap(),
        Some(draft)
    );
}

// ============================================================================
// Row locator tests
// ============================================================================

pub fn test_object_create_and_load_row_locator(factory: &dyn Fn() -> Box<dyn Storage>) {
    let mut storage = factory();
    let object_id = ObjectId::new();
    let schema_hash = SchemaHash::compute(&row_history_test_schema("users"));
    let locator = RowLocator {
        table: "users".into(),
        origin_schema_hash: Some(schema_hash),
    };

    storage.put_row_locator(object_id, Some(&locator)).unwrap();

    assert_eq!(storage.load_row_locator(object_id).unwrap(), Some(locator));
}

pub fn test_row_locator_load_nonexistent_returns_none(factory: &dyn Fn() -> Box<dyn Storage>) {
    let storage = factory();
    assert!(storage.load_row_locator(ObjectId::new()).unwrap().is_none());
}

pub fn test_row_locator_isolation(factory: &dyn Fn() -> Box<dyn Storage>) {
    let mut storage = factory();
    let alice = ObjectId::new();
    let bob = ObjectId::new();
    let schema_hash = SchemaHash::compute(&row_history_test_schema("users"));

    storage
        .put_row_locator(
            alice,
            Some(&RowLocator {
                table: "users".into(),
                origin_schema_hash: Some(schema_hash),
            }),
        )
        .unwrap();
    storage
        .put_row_locator(
            bob,
            Some(&RowLocator {
                table: "posts".into(),
                origin_schema_hash: Some(schema_hash),
            }),
        )
        .unwrap();

    assert_eq!(
        storage
            .load_row_locator(alice)
            .unwrap()
            .unwrap()
            .table
            .as_str(),
        "users"
    );
    assert_eq!(
        storage
            .load_row_locator(bob)
            .unwrap()
            .unwrap()
            .table
            .as_str(),
        "posts"
    );
}

// ============================================================================
// Ordered raw-table tests
// ============================================================================

pub fn test_raw_table_round_trip(factory: &dyn Fn() -> Box<dyn Storage>) {
    let mut storage = factory();
    storage.raw_table_put("users", "alice", b"hello").unwrap();
    assert_eq!(
        storage.raw_table_get("users", "alice").unwrap(),
        Some(b"hello".to_vec())
    );

    storage.raw_table_delete("users", "alice").unwrap();
    assert_eq!(storage.raw_table_get("users", "alice").unwrap(), None);
}

pub fn test_raw_table_scan_prefix(factory: &dyn Fn() -> Box<dyn Storage>) {
    let mut storage = factory();
    storage.raw_table_put("users", "alice/1", b"a").unwrap();
    storage.raw_table_put("users", "alice/2", b"b").unwrap();
    storage.raw_table_put("users", "bob/1", b"c").unwrap();

    let rows = storage.raw_table_scan_prefix("users", "alice/").unwrap();
    assert_eq!(
        rows,
        vec![
            ("alice/1".to_string(), b"a".to_vec()),
            ("alice/2".to_string(), b"b".to_vec()),
        ]
    );
}

pub fn test_raw_table_scan_range(factory: &dyn Fn() -> Box<dyn Storage>) {
    let mut storage = factory();
    storage.raw_table_put("users", "01", b"a").unwrap();
    storage.raw_table_put("users", "02", b"b").unwrap();
    storage.raw_table_put("users", "03", b"c").unwrap();

    let rows = storage
        .raw_table_scan_range("users", Some("02"), Some("04"))
        .unwrap();
    assert_eq!(
        rows,
        vec![
            ("02".to_string(), b"b".to_vec()),
            ("03".to_string(), b"c".to_vec()),
        ]
    );
}

// ============================================================================
// Index tests
// ============================================================================

pub fn test_index_insert_and_exact_lookup(factory: &dyn Fn() -> Box<dyn Storage>) {
    let mut storage = factory();
    let row_id = ObjectId::new();

    storage
        .index_insert("users", "age", "main", &Value::Integer(25), row_id)
        .unwrap();

    assert_eq!(
        storage.index_lookup("users", "age", "main", &Value::Integer(25)),
        vec![row_id]
    );
}

pub fn test_index_duplicate_values(factory: &dyn Fn() -> Box<dyn Storage>) {
    let mut storage = factory();
    let alice = ObjectId::new();
    let bob = ObjectId::new();

    storage
        .index_insert("users", "age", "main", &Value::Integer(25), alice)
        .unwrap();
    storage
        .index_insert("users", "age", "main", &Value::Integer(25), bob)
        .unwrap();

    let mut rows = storage.index_lookup("users", "age", "main", &Value::Integer(25));
    rows.sort();
    let mut expected = vec![alice, bob];
    expected.sort();
    assert_eq!(rows, expected);
}

pub fn test_index_remove(factory: &dyn Fn() -> Box<dyn Storage>) {
    let mut storage = factory();
    let alice = ObjectId::new();
    let bob = ObjectId::new();

    storage
        .index_insert("users", "age", "main", &Value::Integer(25), alice)
        .unwrap();
    storage
        .index_insert("users", "age", "main", &Value::Integer(25), bob)
        .unwrap();
    storage
        .index_remove("users", "age", "main", &Value::Integer(25), alice)
        .unwrap();

    assert_eq!(
        storage.index_lookup("users", "age", "main", &Value::Integer(25)),
        vec![bob]
    );
}

pub fn test_index_range(factory: &dyn Fn() -> Box<dyn Storage>) {
    let mut storage = factory();
    let row1 = ObjectId::new();
    let row2 = ObjectId::new();
    let row3 = ObjectId::new();

    storage
        .index_insert("users", "age", "main", &Value::Integer(20), row1)
        .unwrap();
    storage
        .index_insert("users", "age", "main", &Value::Integer(25), row2)
        .unwrap();
    storage
        .index_insert("users", "age", "main", &Value::Integer(30), row3)
        .unwrap();

    assert_eq!(
        storage.index_range(
            "users",
            "age",
            "main",
            Bound::Included(&Value::Integer(25)),
            Bound::Unbounded,
        ),
        vec![row2, row3]
    );
}

pub fn test_index_cross_branch_isolation(factory: &dyn Fn() -> Box<dyn Storage>) {
    let mut storage = factory();
    let main_row = ObjectId::new();
    let draft_row = ObjectId::new();

    storage
        .index_insert("users", "age", "main", &Value::Integer(25), main_row)
        .unwrap();
    storage
        .index_insert("users", "age", "draft", &Value::Integer(25), draft_row)
        .unwrap();

    assert_eq!(
        storage.index_lookup("users", "age", "main", &Value::Integer(25)),
        vec![main_row]
    );
    assert_eq!(
        storage.index_lookup("users", "age", "draft", &Value::Integer(25)),
        vec![draft_row]
    );
}

// ============================================================================
// Row-region tests
// ============================================================================

pub fn test_row_region_round_trip(factory: &dyn Fn() -> Box<dyn Storage>) {
    let mut storage = factory();
    let schema_hash = seed_row_history_table(storage.as_mut(), "users");
    let row_id = ObjectId::new();
    seed_row_history_locator(storage.as_mut(), "users", row_id, schema_hash);
    let version = make_row_batch(row_id, "main", 10, "alice");
    let batch_id = version.batch_id();

    storage
        .append_history_region_rows("users", std::slice::from_ref(&version))
        .unwrap();
    storage
        .upsert_visible_region_rows(
            "users",
            std::slice::from_ref(&make_visible_entry(
                version.clone(),
                std::slice::from_ref(&version),
            )),
        )
        .unwrap();

    assert_eq!(
        storage
            .load_visible_region_row("users", "main", row_id)
            .unwrap(),
        Some(version.clone())
    );
    assert_eq!(
        storage.scan_visible_region("users", "main").unwrap(),
        vec![version.clone()]
    );
    assert_eq!(
        storage
            .scan_history_region("users", "main", HistoryScan::Row { row_id })
            .unwrap(),
        vec![version]
    );
    assert_eq!(
        storage
            .load_visible_region_frontier("users", "main", row_id)
            .unwrap(),
        Some(vec![batch_id])
    );
}

pub fn test_row_region_keeps_same_batch_id_distinct_across_branches(
    factory: &dyn Fn() -> Box<dyn Storage>,
) {
    let mut storage = factory();
    let schema_hash = seed_row_history_table(storage.as_mut(), "users");
    let row_id = ObjectId::new();
    seed_row_history_locator(storage.as_mut(), "users", row_id, schema_hash);

    let shared_batch_id = BatchId([0x5a; 16]);
    let mut main = make_row_batch(row_id, "dev/main", 10, "alice");
    main.batch_id = shared_batch_id;
    let mut draft = make_row_batch(row_id, "dev/draft", 20, "alice draft");
    draft.batch_id = shared_batch_id;

    storage
        .append_history_region_rows("users", &[main.clone(), draft.clone()])
        .unwrap();

    let history_by_row = storage.scan_history_row_batches("users", row_id).unwrap();
    let main_history = storage
        .scan_history_region("users", "dev/main", HistoryScan::Row { row_id })
        .unwrap();
    let draft_history = storage
        .scan_history_region("users", "dev/draft", HistoryScan::Row { row_id })
        .unwrap();
    let main_loaded = storage
        .load_history_row_batch("users", "dev/main", row_id, shared_batch_id)
        .unwrap();
    let draft_loaded = storage
        .load_history_row_batch("users", "dev/draft", row_id, shared_batch_id)
        .unwrap();
    let ambiguous_lookup =
        storage.load_history_row_batch_any_branch("users", row_id, shared_batch_id);

    assert_eq!(history_by_row, vec![draft.clone(), main.clone()]);
    assert_eq!(main_history, vec![main]);
    assert_eq!(draft_history, vec![draft]);
    assert_eq!(main_loaded, Some(main_history[0].clone()));
    assert_eq!(draft_loaded, Some(draft_history[0].clone()));
    assert!(
        matches!(ambiguous_lookup, Err(StorageError::IoError(ref message)) if message.contains("ambiguous row history version")),
        "expected ambiguous unscoped lookup, got {ambiguous_lookup:?}"
    );
}

pub fn test_row_region_uses_flat_history_bytes_when_schema_known(
    factory: &dyn Fn() -> Box<dyn Storage>,
) {
    let mut storage = factory();
    let schema = SchemaBuilder::new()
        .table(
            TableSchema::builder("tasks")
                .column("title", ColumnType::Text)
                .nullable_column("done", ColumnType::Boolean),
        )
        .build();
    let schema_hash = SchemaHash::compute(&schema);
    let descriptor = schema[&"tasks".into()].columns.clone();
    let row_id = ObjectId::new();
    let row = StoredRowBatch::new(
        row_id,
        "main",
        Vec::new(),
        encode_row(
            &descriptor,
            &[Value::Text("Ship flat rows".into()), Value::Boolean(false)],
        )
        .unwrap(),
        RowProvenance::for_insert("alice".to_string(), 100),
        HashMap::new(),
        RowState::VisibleDirect,
        None,
    );

    storage
        .upsert_catalogue_entry(&CatalogueEntry {
            object_id: schema_hash.to_object_id(),
            metadata: HashMap::from([(
                MetadataKey::Type.to_string(),
                ObjectType::CatalogueSchema.to_string(),
            )]),
            content: encode_schema(&schema),
        })
        .unwrap();
    storage
        .put_row_locator(
            row_id,
            Some(&crate::storage::RowLocator {
                table: "tasks".into(),
                origin_schema_hash: Some(schema_hash),
            }),
        )
        .unwrap();
    storage
        .append_history_region_rows("tasks", std::slice::from_ref(&row))
        .unwrap();

    let encoded = storage
        .load_history_row_batch_bytes("tasks", row.branch.as_str(), row_id, row.batch_id())
        .unwrap()
        .expect("history row should persist");

    assert_eq!(
        decode_flat_history_row(
            &descriptor,
            row_id,
            row.branch.as_str(),
            row.batch_id(),
            &encoded,
        )
        .unwrap(),
        row
    );
}

pub fn test_visible_region_uses_flat_bytes_when_schema_known(
    factory: &dyn Fn() -> Box<dyn Storage>,
) {
    let mut storage = factory();
    let schema = SchemaBuilder::new()
        .table(
            TableSchema::builder("tasks")
                .column("title", ColumnType::Text)
                .nullable_column("done", ColumnType::Boolean),
        )
        .build();
    let schema_hash = SchemaHash::compute(&schema);
    let descriptor = schema[&"tasks".into()].columns.clone();
    let row_id = ObjectId::new();
    let row = StoredRowBatch::new(
        row_id,
        "main",
        Vec::new(),
        encode_row(
            &descriptor,
            &[
                Value::Text("Ship visible rows".into()),
                Value::Boolean(false),
            ],
        )
        .unwrap(),
        RowProvenance::for_insert("alice".to_string(), 100),
        HashMap::new(),
        RowState::VisibleDirect,
        Some(DurabilityTier::Local),
    );
    let entry = VisibleRowEntry::rebuild(row.clone(), std::slice::from_ref(&row));

    storage
        .upsert_catalogue_entry(&CatalogueEntry {
            object_id: schema_hash.to_object_id(),
            metadata: HashMap::from([(
                MetadataKey::Type.to_string(),
                ObjectType::CatalogueSchema.to_string(),
            )]),
            content: encode_schema(&schema),
        })
        .unwrap();
    storage
        .put_row_locator(
            row_id,
            Some(&crate::storage::RowLocator {
                table: "tasks".into(),
                origin_schema_hash: Some(schema_hash),
            }),
        )
        .unwrap();
    storage
        .upsert_visible_region_rows("tasks", std::slice::from_ref(&entry))
        .unwrap();

    let encoded = storage
        .load_visible_region_row_bytes("tasks", "main", row_id)
        .unwrap()
        .expect("visible row should persist");

    assert_eq!(
        decode_flat_visible_row_entry(&descriptor, row_id, "main", &encoded).unwrap(),
        entry
    );
}

pub fn test_visible_region_does_not_write_separate_batch_side_index(
    factory: &dyn Fn() -> Box<dyn Storage>,
) {
    let mut storage = factory();
    let schema_hash = seed_row_history_table(storage.as_mut(), "tasks");
    let schema = row_history_test_schema("tasks");
    let descriptor = row_history_user_descriptor();
    let row_id = ObjectId::new();
    seed_row_history_locator(storage.as_mut(), "tasks", row_id, schema_hash);

    let row = StoredRowBatch::new(
        row_id,
        "main",
        Vec::new(),
        encode_row(&descriptor, &[Value::Text("ship".to_string())]).unwrap(),
        RowProvenance::for_insert("alice".to_string(), 100),
        HashMap::new(),
        RowState::VisibleDirect,
        Some(DurabilityTier::Local),
    );
    let entry = VisibleRowEntry::rebuild(row.clone(), std::slice::from_ref(&row));

    storage
        .upsert_catalogue_entry(&CatalogueEntry {
            object_id: schema_hash.to_object_id(),
            metadata: HashMap::from([(
                MetadataKey::Type.to_string(),
                ObjectType::CatalogueSchema.to_string(),
            )]),
            content: encode_schema(&schema),
        })
        .unwrap();
    storage
        .put_row_locator(
            row_id,
            Some(&crate::storage::RowLocator {
                table: "tasks".into(),
                origin_schema_hash: Some(schema_hash),
            }),
        )
        .unwrap();
    storage
        .upsert_visible_region_rows("tasks", std::slice::from_ref(&entry))
        .unwrap();

    assert_eq!(
        storage
            .raw_table_scan_prefix("tasks", "row:tasks:2:")
            .unwrap(),
        Vec::new(),
        "visible rows should not need a separate batch-id side index once current batch identity lives in the flat payload"
    );
}

pub fn test_row_region_patch_state_monotonic(factory: &dyn Fn() -> Box<dyn Storage>) {
    let mut storage = factory();
    let schema_hash = seed_row_history_table(storage.as_mut(), "users");
    let row_id = ObjectId::new();
    seed_row_history_locator(storage.as_mut(), "users", row_id, schema_hash);
    let version = make_row_batch(row_id, "main", 10, "alice");

    storage
        .append_history_region_rows("users", std::slice::from_ref(&version))
        .unwrap();
    storage
        .upsert_visible_region_rows(
            "users",
            std::slice::from_ref(&make_visible_entry(
                version.clone(),
                std::slice::from_ref(&version),
            )),
        )
        .unwrap();

    storage
        .patch_row_region_rows_by_batch(
            "users",
            version.batch_id,
            None,
            Some(DurabilityTier::EdgeServer),
        )
        .unwrap();
    storage
        .patch_row_region_rows_by_batch(
            "users",
            version.batch_id,
            None,
            Some(DurabilityTier::Local),
        )
        .unwrap();

    let visible = storage
        .load_visible_region_row("users", "main", row_id)
        .unwrap();
    let history = storage
        .scan_history_region("users", "main", HistoryScan::Row { row_id })
        .unwrap();

    assert_eq!(
        visible.and_then(|row| row.confirmed_tier),
        Some(DurabilityTier::EdgeServer)
    );
    assert_eq!(history[0].confirmed_tier, Some(DurabilityTier::EdgeServer));
}

pub fn test_row_region_branch_isolation(factory: &dyn Fn() -> Box<dyn Storage>) {
    let mut storage = factory();
    let schema_hash = seed_row_history_table(storage.as_mut(), "users");
    let main_row_id = ObjectId::new();
    let draft_row_id = ObjectId::new();
    seed_row_history_locator(storage.as_mut(), "users", main_row_id, schema_hash);
    seed_row_history_locator(storage.as_mut(), "users", draft_row_id, schema_hash);
    let main_row = make_row_batch(main_row_id, "main", 10, "main");
    let draft_row = make_row_batch(draft_row_id, "draft", 20, "draft");

    storage
        .append_history_region_rows("users", &[main_row.clone(), draft_row.clone()])
        .unwrap();
    storage
        .upsert_visible_region_rows(
            "users",
            &[
                make_visible_entry(main_row.clone(), std::slice::from_ref(&main_row)),
                make_visible_entry(draft_row.clone(), std::slice::from_ref(&draft_row)),
            ],
        )
        .unwrap();

    assert_eq!(
        storage.scan_visible_region("users", "main").unwrap(),
        vec![main_row]
    );
    assert_eq!(
        storage.scan_visible_region("users", "draft").unwrap(),
        vec![draft_row]
    );
}

pub fn test_row_region_cross_branch_visible_heads(factory: &dyn Fn() -> Box<dyn Storage>) {
    let mut storage = factory();
    let schema_hash = seed_row_history_table(storage.as_mut(), "users");
    let row_id = ObjectId::new();
    seed_row_history_locator(storage.as_mut(), "users", row_id, schema_hash);
    let main_row = make_row_batch(row_id, "main", 10, "main");
    let draft_row = make_row_batch(row_id, "draft", 20, "draft");

    storage
        .append_history_region_rows("users", &[main_row.clone(), draft_row.clone()])
        .unwrap();
    storage
        .upsert_visible_region_rows(
            "users",
            &[
                make_visible_entry(main_row.clone(), std::slice::from_ref(&main_row)),
                make_visible_entry(draft_row.clone(), std::slice::from_ref(&draft_row)),
            ],
        )
        .unwrap();

    assert_eq!(
        storage
            .scan_visible_region_row_batches("users", row_id)
            .unwrap(),
        vec![draft_row, main_row]
    );
}

pub fn test_apply_row_mutation_combines_row_and_index_effects(
    factory: &dyn Fn() -> Box<dyn Storage>,
) {
    let mut storage = factory();
    let schema_hash = seed_row_history_table(storage.as_mut(), "users");
    let row_id = ObjectId::new();
    seed_row_history_locator(storage.as_mut(), "users", row_id, schema_hash);
    let version = make_row_batch(row_id, "main", 10, "alice");
    let visible_entry = make_visible_entry(version.clone(), std::slice::from_ref(&version));
    let index_mutations = [IndexMutation::Insert {
        table: "users",
        column: "name",
        branch: "main",
        value: Value::Text("alice".to_string()),
        row_id,
    }];

    storage
        .apply_row_mutation(
            "users",
            std::slice::from_ref(&version),
            std::slice::from_ref(&visible_entry),
            &index_mutations,
        )
        .unwrap();

    assert_eq!(
        storage
            .load_visible_region_row("users", "main", row_id)
            .unwrap(),
        Some(version.clone())
    );
    assert_eq!(
        storage
            .load_history_row_batch("users", "main", row_id, version.batch_id())
            .unwrap(),
        Some(version.clone())
    );
    assert_eq!(
        storage.index_lookup("users", "name", "main", &Value::Text("alice".to_string())),
        vec![row_id]
    );
}

// ============================================================================
// Catalogue tests
// ============================================================================

pub fn test_catalogue_entry_round_trip(factory: &dyn Fn() -> Box<dyn Storage>) {
    let mut storage = factory();
    let object_id = ObjectId::new();
    let entry = CatalogueEntry {
        object_id,
        metadata: HashMap::from([
            (
                MetadataKey::Type.to_string(),
                ObjectType::CatalogueSchema.to_string(),
            ),
            ("schema_hash".to_string(), "abc123".to_string()),
        ]),
        content: br#"{"tables":["users"]}"#.to_vec(),
    };

    storage.upsert_catalogue_entry(&entry).unwrap();
    assert_eq!(
        storage.load_catalogue_entry(object_id).unwrap(),
        Some(entry)
    );
}

pub fn test_catalogue_entry_scan_returns_sorted_entries(factory: &dyn Fn() -> Box<dyn Storage>) {
    let mut storage = factory();
    let low_id = ObjectId::from_uuid(uuid::Uuid::nil());
    let high_id = ObjectId::new();

    storage
        .upsert_catalogue_entry(&CatalogueEntry {
            object_id: high_id,
            metadata: HashMap::from([(
                MetadataKey::Type.to_string(),
                ObjectType::CatalogueLens.to_string(),
            )]),
            content: b"lens".to_vec(),
        })
        .unwrap();
    storage
        .upsert_catalogue_entry(&CatalogueEntry {
            object_id: low_id,
            metadata: HashMap::from([(
                MetadataKey::Type.to_string(),
                ObjectType::CatalogueSchema.to_string(),
            )]),
            content: b"schema".to_vec(),
        })
        .unwrap();

    let scanned = storage.scan_catalogue_entries().unwrap();
    assert_eq!(scanned.len(), 2);
    assert_eq!(scanned[0].object_id, low_id);
    assert_eq!(scanned[1].object_id, high_id);
}

pub fn test_catalogue_entry_upsert_replaces_existing(factory: &dyn Fn() -> Box<dyn Storage>) {
    let mut storage = factory();
    let object_id = ObjectId::new();
    let first = CatalogueEntry {
        object_id,
        metadata: HashMap::from([(
            MetadataKey::Type.to_string(),
            ObjectType::CataloguePermissionsHead.to_string(),
        )]),
        content: b"v1".to_vec(),
    };
    let second = CatalogueEntry {
        object_id,
        metadata: HashMap::from([
            (
                MetadataKey::Type.to_string(),
                ObjectType::CataloguePermissionsHead.to_string(),
            ),
            ("note".to_string(), "updated".to_string()),
        ]),
        content: b"v2".to_vec(),
    };

    storage.upsert_catalogue_entry(&first).unwrap();
    storage.upsert_catalogue_entry(&second).unwrap();

    assert_eq!(
        storage.load_catalogue_entry(object_id).unwrap(),
        Some(second)
    );
}

pub fn test_catalogue_entry_nonexistent_returns_none(factory: &dyn Fn() -> Box<dyn Storage>) {
    let storage = factory();
    assert!(
        storage
            .load_catalogue_entry(ObjectId::new())
            .unwrap()
            .is_none()
    );
}

pub fn test_local_batch_record_round_trip(factory: &dyn Fn() -> Box<dyn Storage>) {
    let mut storage = factory();
    let batch_id = crate::row_histories::BatchId::new();
    let mut record = LocalBatchRecord::new(
        batch_id,
        BatchMode::Direct,
        true,
        Some(BatchFate::DurableDirect {
            batch_id,
            confirmed_tier: DurabilityTier::Local,
        }),
    );
    record.upsert_member(LocalBatchMember {
        object_id: ObjectId::from_uuid(uuid::Uuid::from_u128(94)),
        table_name: "users".to_string(),
        branch_name: crate::object::BranchName::new("dev-aaaaaaaaaaaa-main"),
        schema_hash: SchemaHash::from_bytes([0xaa; 32]),
        row_digest: Digest32([11; 32]),
    });
    record.upsert_member(LocalBatchMember {
        object_id: ObjectId::from_uuid(uuid::Uuid::from_u128(95)),
        table_name: "users".to_string(),
        branch_name: crate::object::BranchName::new("dev-aaaaaaaaaaaa-main"),
        schema_hash: SchemaHash::from_bytes([0xaa; 32]),
        row_digest: Digest32([12; 32]),
    });
    record.sealed_submission = Some(SealedBatchSubmission::new(
        batch_id,
        crate::batch_fate::BatchMode::Direct,
        crate::object::BranchName::new("dev-aaaaaaaaaaaa-main"),
        vec![SealedBatchMember {
            object_id: ObjectId::from_uuid(uuid::Uuid::from_u128(92)),
            row_digest: Digest32([9; 32]),
        }],
        // Compatibility payload only: storage must preserve it until the next
        // format break, but transaction validation no longer reads it.
        vec![CapturedFrontierMember {
            object_id: ObjectId::from_uuid(uuid::Uuid::from_u128(93)),
            branch_name: crate::object::BranchName::new("dev-bbbbbbbbbbbb-main"),
            batch_id: BatchId([10; 16]),
        }],
    ));

    storage.upsert_local_batch_record(&record).unwrap();

    assert_eq!(
        storage.load_local_batch_record(batch_id).unwrap(),
        Some(record)
    );
    assert!(
        storage
            .load_branch_ord(crate::object::BranchName::new("dev-aaaaaaaaaaaa-main"))
            .unwrap()
            .is_some()
    );
    assert!(
        storage
            .load_branch_ord(crate::object::BranchName::new("dev-bbbbbbbbbbbb-main"))
            .unwrap()
            .is_some()
    );
}

pub fn test_local_batch_record_scan_returns_sorted_entries(factory: &dyn Fn() -> Box<dyn Storage>) {
    let mut storage = factory();
    let low = crate::row_histories::BatchId::from_uuid(uuid::Uuid::from_u128(1));
    let high = crate::row_histories::BatchId::from_uuid(uuid::Uuid::from_u128(2));

    storage
        .upsert_local_batch_record(&LocalBatchRecord::new(high, BatchMode::Direct, true, None))
        .unwrap();
    storage
        .upsert_local_batch_record(&LocalBatchRecord::new(
            low,
            BatchMode::Transactional,
            false,
            Some(BatchFate::Missing { batch_id: low }),
        ))
        .unwrap();

    let records = storage.scan_local_batch_records().unwrap();
    assert_eq!(records.len(), 2);
    assert_eq!(records[0].batch_id, low);
    assert_eq!(records[1].batch_id, high);
}

pub fn test_local_batch_record_delete_removes_record(factory: &dyn Fn() -> Box<dyn Storage>) {
    let mut storage = factory();
    let batch_id = crate::row_histories::BatchId::new();
    let record = LocalBatchRecord::new(
        batch_id,
        BatchMode::Transactional,
        false,
        Some(BatchFate::Rejected {
            batch_id,
            code: "permission_denied".to_string(),
            reason: "writer lacks publish rights".to_string(),
        }),
    );

    storage.upsert_local_batch_record(&record).unwrap();
    assert_eq!(
        storage.load_local_batch_record(batch_id).unwrap(),
        Some(record)
    );

    storage.delete_local_batch_record(batch_id).unwrap();
    assert_eq!(storage.load_local_batch_record(batch_id).unwrap(), None);
}

pub fn test_authoritative_batch_fate_round_trip(factory: &dyn Fn() -> Box<dyn Storage>) {
    let mut storage = factory();
    let batch_id = crate::row_histories::BatchId::new();
    let settlement = BatchFate::Rejected {
        batch_id,
        code: "permission_denied".to_string(),
        reason: "writer lacks publish rights".to_string(),
    };

    storage
        .upsert_authoritative_batch_fate(&settlement)
        .unwrap();

    assert_eq!(
        storage.load_authoritative_batch_fate(batch_id).unwrap(),
        Some(settlement)
    );
}

pub fn test_sealed_batch_submission_round_trip(factory: &dyn Fn() -> Box<dyn Storage>) {
    let mut storage = factory();
    let batch_id = crate::row_histories::BatchId::new();
    let alice = ObjectId::from_uuid(uuid::Uuid::from_u128(301));
    let bob = ObjectId::from_uuid(uuid::Uuid::from_u128(302));
    let submission = SealedBatchSubmission::new(
        batch_id,
        crate::batch_fate::BatchMode::Transactional,
        crate::object::BranchName::new("main"),
        vec![
            SealedBatchMember {
                object_id: alice,
                row_digest: Digest32([1; 32]),
            },
            SealedBatchMember {
                object_id: bob,
                row_digest: Digest32([2; 32]),
            },
            SealedBatchMember {
                object_id: alice,
                row_digest: Digest32([1; 32]),
            },
        ],
        vec![CapturedFrontierMember {
            object_id: bob,
            branch_name: crate::object::BranchName::new("dev-aaaaaaaaaaaa-main"),
            batch_id: BatchId([3; 16]),
        }],
    );

    storage.upsert_sealed_batch_submission(&submission).unwrap();

    assert_eq!(
        storage.load_sealed_batch_submission(batch_id).unwrap(),
        Some(SealedBatchSubmission::new(
            batch_id,
            crate::batch_fate::BatchMode::Transactional,
            crate::object::BranchName::new("main"),
            vec![
                SealedBatchMember {
                    object_id: alice,
                    row_digest: Digest32([1; 32]),
                },
                SealedBatchMember {
                    object_id: bob,
                    row_digest: Digest32([2; 32]),
                },
            ],
            vec![CapturedFrontierMember {
                object_id: bob,
                branch_name: crate::object::BranchName::new("dev-aaaaaaaaaaaa-main"),
                batch_id: BatchId([3; 16]),
            }],
        ))
    );

    assert!(
        storage
            .load_branch_ord(crate::object::BranchName::new("main"))
            .unwrap()
            .is_some()
    );
    assert!(
        storage
            .load_branch_ord(crate::object::BranchName::new("dev-aaaaaaaaaaaa-main"))
            .unwrap()
            .is_some()
    );
}

pub fn test_sealed_batch_submission_delete_removes_record(factory: &dyn Fn() -> Box<dyn Storage>) {
    let mut storage = factory();
    let batch_id = crate::row_histories::BatchId::new();
    let submission = SealedBatchSubmission::new(
        batch_id,
        crate::batch_fate::BatchMode::Direct,
        crate::object::BranchName::new("main"),
        vec![SealedBatchMember {
            object_id: ObjectId::from_uuid(uuid::Uuid::from_u128(401)),
            row_digest: Digest32([4; 32]),
        }],
        Vec::new(),
    );

    storage.upsert_sealed_batch_submission(&submission).unwrap();
    assert_eq!(
        storage.load_sealed_batch_submission(batch_id).unwrap(),
        Some(submission)
    );

    storage.delete_sealed_batch_submission(batch_id).unwrap();
    assert_eq!(
        storage.load_sealed_batch_submission(batch_id).unwrap(),
        None
    );
}

// ============================================================================
// Persistence tests
// ============================================================================

pub fn test_persistence_survives_close_reopen(factory: &PersistentStorageFactory) {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path();

    let row_id = ObjectId::new();
    let schema_hash = SchemaHash::compute(&row_history_test_schema("users"));
    let version = make_row_batch(row_id, "main", 10, "alice");

    {
        let mut storage = factory(path);
        seed_row_history_table(storage.as_mut(), "users");
        seed_row_history_locator(storage.as_mut(), "users", row_id, schema_hash);
        storage.raw_table_put("users", "alice", b"hello").unwrap();
        storage
            .append_history_region_rows("users", std::slice::from_ref(&version))
            .unwrap();
        storage
            .upsert_visible_region_rows(
                "users",
                std::slice::from_ref(&make_visible_entry(
                    version.clone(),
                    std::slice::from_ref(&version),
                )),
            )
            .unwrap();
        storage
            .index_insert(
                "users",
                "name",
                "main",
                &Value::Text("alice".to_string()),
                row_id,
            )
            .unwrap();
        storage.flush().unwrap();
        storage.close().unwrap();
    }

    {
        let storage = factory(path);
        assert_eq!(
            storage.raw_table_get("users", "alice").unwrap(),
            Some(b"hello".to_vec())
        );
        assert_eq!(
            storage
                .load_visible_region_row("users", "main", row_id)
                .unwrap(),
            Some(version.clone())
        );
        assert_eq!(
            storage.index_lookup("users", "name", "main", &Value::Text("alice".to_string())),
            vec![row_id]
        );
    }
}

pub fn test_close_releases_resources_for_reopen(factory: &PersistentStorageFactory) {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path();

    factory(path).close().unwrap();
    factory(path).close().unwrap();
}

pub fn test_local_batch_record_survives_close_reopen(factory: &PersistentStorageFactory) {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path();
    let batch_id = crate::row_histories::BatchId::new();
    let mut record = LocalBatchRecord::new(
        batch_id,
        BatchMode::Direct,
        true,
        Some(BatchFate::DurableDirect {
            batch_id,
            confirmed_tier: DurabilityTier::Local,
        }),
    );
    record.sealed_submission = Some(SealedBatchSubmission::new(
        batch_id,
        crate::batch_fate::BatchMode::Direct,
        crate::object::BranchName::new("dev-aaaaaaaaaaaa-main"),
        vec![SealedBatchMember {
            object_id: ObjectId::from_uuid(uuid::Uuid::from_u128(112)),
            row_digest: Digest32([11; 32]),
        }],
        vec![CapturedFrontierMember {
            object_id: ObjectId::from_uuid(uuid::Uuid::from_u128(113)),
            branch_name: crate::object::BranchName::new("dev-bbbbbbbbbbbb-main"),
            batch_id: BatchId([12; 16]),
        }],
    ));

    {
        let mut storage = factory(path);
        storage.upsert_local_batch_record(&record).unwrap();
        storage.flush().unwrap();
        storage.close().unwrap();
    }

    {
        let storage = factory(path);
        assert_eq!(
            storage.load_local_batch_record(batch_id).unwrap(),
            Some(record)
        );
        assert!(
            storage
                .load_branch_ord(crate::object::BranchName::new("dev-aaaaaaaaaaaa-main"))
                .unwrap()
                .is_some()
        );
        assert!(
            storage
                .load_branch_ord(crate::object::BranchName::new("dev-bbbbbbbbbbbb-main"))
                .unwrap()
                .is_some()
        );
    }
}

pub fn test_authoritative_batch_fate_survives_close_reopen(factory: &PersistentStorageFactory) {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path();
    let batch_id = crate::row_histories::BatchId::new();
    let settlement = BatchFate::Rejected {
        batch_id,
        code: "session_required".to_string(),
        reason: "transaction needs an authenticated session".to_string(),
    };

    {
        let mut storage = factory(path);
        storage
            .upsert_authoritative_batch_fate(&settlement)
            .unwrap();
        storage.flush().unwrap();
        storage.close().unwrap();
    }

    {
        let storage = factory(path);
        assert_eq!(
            storage.load_authoritative_batch_fate(batch_id).unwrap(),
            Some(settlement)
        );
    }
}

pub fn test_sealed_batch_submission_survives_close_reopen(factory: &PersistentStorageFactory) {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path();
    let batch_id = crate::row_histories::BatchId::new();
    let submission = SealedBatchSubmission::new(
        batch_id,
        crate::batch_fate::BatchMode::Direct,
        crate::object::BranchName::new("main"),
        vec![
            SealedBatchMember {
                object_id: ObjectId::from_uuid(uuid::Uuid::from_u128(501)),
                row_digest: Digest32([5; 32]),
            },
            SealedBatchMember {
                object_id: ObjectId::from_uuid(uuid::Uuid::from_u128(502)),
                row_digest: Digest32([6; 32]),
            },
        ],
        vec![CapturedFrontierMember {
            object_id: ObjectId::from_uuid(uuid::Uuid::from_u128(503)),
            branch_name: crate::object::BranchName::new("dev-aaaaaaaaaaaa-main"),
            batch_id: BatchId([7; 16]),
        }],
    );

    {
        let mut storage = factory(path);
        storage.upsert_sealed_batch_submission(&submission).unwrap();
        storage.flush().unwrap();
        storage.close().unwrap();
    }

    {
        let storage = factory(path);
        assert_eq!(
            storage.load_sealed_batch_submission(batch_id).unwrap(),
            Some(submission)
        );
        assert!(
            storage
                .load_branch_ord(crate::object::BranchName::new("main"))
                .unwrap()
                .is_some()
        );
        assert!(
            storage
                .load_branch_ord(crate::object::BranchName::new("dev-aaaaaaaaaaaa-main"))
                .unwrap()
                .is_some()
        );
    }
}

pub fn test_branch_ord_survives_close_reopen(factory: &PersistentStorageFactory) {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path();
    let main = crate::object::BranchName::new("dev-aaaaaaaaaaaa-main");
    let draft = crate::object::BranchName::new("dev-bbbbbbbbbbbb-main");

    let (main_ord, draft_ord) = {
        let mut storage = factory(path);
        let main_ord = storage.resolve_or_alloc_branch_ord(main).unwrap();
        let draft_ord = storage.resolve_or_alloc_branch_ord(draft).unwrap();
        storage.flush().unwrap();
        storage.close().unwrap();
        (main_ord, draft_ord)
    };

    {
        let mut storage = factory(path);
        assert_eq!(storage.load_branch_ord(main).unwrap(), Some(main_ord));
        assert_eq!(storage.load_branch_ord(draft).unwrap(), Some(draft_ord));
        assert_eq!(
            storage.load_branch_name_by_ord(main_ord).unwrap(),
            Some(main)
        );
        assert_eq!(
            storage.load_branch_name_by_ord(draft_ord).unwrap(),
            Some(draft)
        );
        let feature = crate::object::BranchName::new("dev-cccccccccccc-main");
        let feature_ord = storage.resolve_or_alloc_branch_ord(feature).unwrap();
        assert!(feature_ord > draft_ord);
    }
}

// ============================================================================
// Multi-actor test
// ============================================================================

pub fn test_alice_bob_branch_isolation(factory: &dyn Fn() -> Box<dyn Storage>) {
    let mut storage = factory();
    let schema_hash = seed_row_history_table(storage.as_mut(), "users");
    let main_row_id = ObjectId::new();
    let draft_row_id = ObjectId::new();
    seed_row_history_locator(storage.as_mut(), "users", main_row_id, schema_hash);
    seed_row_history_locator(storage.as_mut(), "users", draft_row_id, schema_hash);
    let main_row = make_row_batch(main_row_id, "main", 10, "alice");
    let draft_row = make_row_batch(draft_row_id, "draft", 20, "bob");

    storage
        .append_history_region_rows("users", &[main_row.clone(), draft_row.clone()])
        .unwrap();
    storage
        .upsert_visible_region_rows(
            "users",
            &[
                make_visible_entry(main_row.clone(), std::slice::from_ref(&main_row)),
                make_visible_entry(draft_row.clone(), std::slice::from_ref(&draft_row)),
            ],
        )
        .unwrap();
    storage
        .index_insert(
            "users",
            "name",
            "main",
            &Value::Text("alice".to_string()),
            main_row.row_id,
        )
        .unwrap();
    storage
        .index_insert(
            "users",
            "name",
            "draft",
            &Value::Text("bob".to_string()),
            draft_row.row_id,
        )
        .unwrap();

    assert_eq!(
        storage.scan_visible_region("users", "main").unwrap(),
        vec![main_row.clone()]
    );
    assert_eq!(
        storage.scan_visible_region("users", "draft").unwrap(),
        vec![draft_row.clone()]
    );
    assert_eq!(
        storage.index_lookup("users", "name", "main", &Value::Text("alice".to_string())),
        vec![main_row.row_id]
    );
    assert_eq!(
        storage.index_lookup("users", "name", "draft", &Value::Text("bob".to_string())),
        vec![draft_row.row_id]
    );
}

// ============================================================================
// Macros
// ============================================================================

#[macro_export]
macro_rules! storage_conformance_tests {
    ($prefix:ident, $factory:expr) => {
        mod $prefix {
            use super::*;
            use $crate::storage::conformance;

            #[test]
            fn object_create_and_load_row_locator() {
                conformance::test_object_create_and_load_row_locator(&$factory);
            }

            #[test]
            fn row_locator_load_nonexistent_returns_none() {
                conformance::test_row_locator_load_nonexistent_returns_none(&$factory);
            }

            #[test]
            fn row_locator_isolation() {
                conformance::test_row_locator_isolation(&$factory);
            }

            #[test]
            fn raw_table_round_trip() {
                conformance::test_raw_table_round_trip(&$factory);
            }

            #[test]
            fn raw_table_scan_prefix() {
                conformance::test_raw_table_scan_prefix(&$factory);
            }

            #[test]
            fn raw_table_first_key_with_prefix() {
                conformance::test_raw_table_first_key_with_prefix(&$factory);
            }

            #[test]
            fn row_has_history_outside_branch() {
                conformance::test_history_row_exists_on_branch(&$factory);
            }

            #[test]
            fn history_row_exists_on_an_unregistered_branch() {
                conformance::test_history_row_exists_on_an_unregistered_branch(&$factory);
            }

            #[test]
            fn history_row_batch_family_is_the_table_that_holds_it() {
                conformance::test_history_row_batch_family_is_the_table_that_holds_it(&$factory);
            }

            #[test]
            fn raw_table_scan_range() {
                conformance::test_raw_table_scan_range(&$factory);
            }

            #[test]
            fn index_insert_and_exact_lookup() {
                conformance::test_index_insert_and_exact_lookup(&$factory);
            }

            #[test]
            fn index_duplicate_values() {
                conformance::test_index_duplicate_values(&$factory);
            }

            #[test]
            fn index_remove() {
                conformance::test_index_remove(&$factory);
            }

            #[test]
            fn index_range() {
                conformance::test_index_range(&$factory);
            }

            #[test]
            fn index_cross_branch_isolation() {
                conformance::test_index_cross_branch_isolation(&$factory);
            }

            #[test]
            fn row_region_round_trip() {
                conformance::test_row_region_round_trip(&$factory);
            }

            #[test]
            fn row_region_keeps_same_batch_id_distinct_across_branches() {
                conformance::test_row_region_keeps_same_batch_id_distinct_across_branches(
                    &$factory,
                );
            }

            #[test]
            fn row_region_uses_flat_history_bytes_when_schema_known() {
                conformance::test_row_region_uses_flat_history_bytes_when_schema_known(&$factory);
            }

            #[test]
            fn visible_region_uses_flat_bytes_when_schema_known() {
                conformance::test_visible_region_uses_flat_bytes_when_schema_known(&$factory);
            }

            #[test]
            fn visible_region_does_not_write_separate_batch_side_index() {
                conformance::test_visible_region_does_not_write_separate_batch_side_index(
                    &$factory,
                );
            }

            #[test]
            fn row_region_patch_state_monotonic() {
                conformance::test_row_region_patch_state_monotonic(&$factory);
            }

            #[test]
            fn row_region_branch_isolation() {
                conformance::test_row_region_branch_isolation(&$factory);
            }

            #[test]
            fn row_region_cross_branch_visible_heads() {
                conformance::test_row_region_cross_branch_visible_heads(&$factory);
            }

            #[test]
            fn apply_row_mutation_combines_row_and_index_effects() {
                conformance::test_apply_row_mutation_combines_row_and_index_effects(&$factory);
            }

            #[test]
            fn catalogue_entry_round_trip() {
                conformance::test_catalogue_entry_round_trip(&$factory);
            }

            #[test]
            fn catalogue_entry_scan_returns_sorted_entries() {
                conformance::test_catalogue_entry_scan_returns_sorted_entries(&$factory);
            }

            #[test]
            fn catalogue_entry_upsert_replaces_existing() {
                conformance::test_catalogue_entry_upsert_replaces_existing(&$factory);
            }

            #[test]
            fn catalogue_entry_nonexistent_returns_none() {
                conformance::test_catalogue_entry_nonexistent_returns_none(&$factory);
            }

            #[test]
            fn row_raw_table_header_round_trip() {
                conformance::test_row_raw_table_header_round_trip(&$factory);
            }

            #[test]
            fn branch_ord_round_trip() {
                conformance::test_branch_ord_round_trip(&$factory);
            }

            #[test]
            fn local_batch_record_round_trip() {
                conformance::test_local_batch_record_round_trip(&$factory);
            }

            #[test]
            fn local_batch_record_scan_returns_sorted_entries() {
                conformance::test_local_batch_record_scan_returns_sorted_entries(&$factory);
            }

            #[test]
            fn local_batch_record_delete_removes_record() {
                conformance::test_local_batch_record_delete_removes_record(&$factory);
            }

            #[test]
            fn authoritative_batch_fate_round_trip() {
                conformance::test_authoritative_batch_fate_round_trip(&$factory);
            }

            #[test]
            fn sealed_batch_submission_round_trip() {
                conformance::test_sealed_batch_submission_round_trip(&$factory);
            }

            #[test]
            fn sealed_batch_submission_delete_removes_record() {
                conformance::test_sealed_batch_submission_delete_removes_record(&$factory);
            }

            #[test]
            fn alice_bob_branch_isolation() {
                conformance::test_alice_bob_branch_isolation(&$factory);
            }

            #[test]
            fn visible_entry_differential_random_ops() {
                $crate::storage::conformance_differential::test_visible_entry_differential_random_ops(
                    &$factory,
                );
            }
        }
    };
}

#[macro_export]
macro_rules! storage_conformance_tests_persistent {
    ($prefix:ident, $factory:expr, $reopen_factory:expr) => {
        $crate::storage_conformance_tests!($prefix, $factory);

        mod paste_persistent {
            use super::*;
            use $crate::storage::conformance;

            #[test]
            fn branch_ord_survives_close_reopen() {
                conformance::test_branch_ord_survives_close_reopen(&$reopen_factory);
            }

            #[test]
            fn persistence_survives_close_reopen() {
                conformance::test_persistence_survives_close_reopen(&$reopen_factory);
            }

            #[test]
            fn close_releases_resources_for_reopen() {
                conformance::test_close_releases_resources_for_reopen(&$reopen_factory);
            }

            #[test]
            fn local_batch_record_survives_close_reopen() {
                conformance::test_local_batch_record_survives_close_reopen(&$reopen_factory);
            }

            #[test]
            fn authoritative_batch_fate_survives_close_reopen() {
                conformance::test_authoritative_batch_fate_survives_close_reopen(&$reopen_factory);
            }

            #[test]
            fn sealed_batch_submission_survives_close_reopen() {
                conformance::test_sealed_batch_submission_survives_close_reopen(&$reopen_factory);
            }
        }
    };
}

/// A backend that seeks must find exactly what a backend that scans finds.
///
/// `raw_table_first_key_with_prefix` has a correct-everywhere default and a seek
/// override where the backend can do better. The two disagreeing is the failure
/// mode that only production notices, because the callers use it to decide
/// whether to read a row's whole history at all.
pub fn test_raw_table_first_key_with_prefix(factory: &dyn Fn() -> Box<dyn Storage>) {
    let mut storage = factory();
    storage.raw_table_put("users", "alice/1", b"a").unwrap();
    storage.raw_table_put("users", "alice/2", b"b").unwrap();
    storage.raw_table_put("users", "bob/1", b"c").unwrap();

    for prefix in ["alice/", "bob/", "carol/", "alice/1", ""] {
        let sought = storage
            .raw_table_first_key_with_prefix("users", prefix)
            .unwrap();
        let scanned = storage
            .raw_table_scan_prefix_keys("users", prefix)
            .unwrap()
            .into_iter()
            .next();
        assert_eq!(
            sought, scanned,
            "seeking and scanning disagree on the first key under {prefix:?}"
        );
    }
}

/// The branch question must have one answer, whatever the backend.
///
/// `history_row_exists_on_branch` is what a write asks instead of reading a
/// row's whole history, and backends answer it differently — from their own
/// index, or by seeking history keys, or by the scanning default. A backend that
/// answers "no" where the rows are kept somewhere it did not look sends the
/// caller down the cheap path with the wrong answer, and only production
/// notices.
pub fn test_history_row_exists_on_branch(factory: &dyn Fn() -> Box<dyn Storage>) {
    let mut storage = factory();
    let schema_hash = seed_row_history_table(storage.as_mut(), "users");
    let row_id = ObjectId::new();
    seed_row_history_locator(storage.as_mut(), "users", row_id, schema_hash);

    let main = make_row_batch(row_id, "dev/main", 10, "alice");
    storage
        .append_history_region_rows("users", std::slice::from_ref(&main))
        .unwrap();

    assert!(
        storage
            .row_has_history_outside_branch("users", row_id, "dev/draft")
            .unwrap(),
        "the branch the row was written on must be seen from another one"
    );
    assert!(
        !storage
            .row_has_history_outside_branch("users", row_id, "dev/main")
            .unwrap(),
        "with every version on this branch there is nothing outside it"
    );
    assert!(
        !storage
            .row_has_history_outside_branch("users", ObjectId::new(), "dev/main")
            .unwrap(),
        "another row's versions must not answer for this one"
    );

    let draft = make_row_batch(row_id, "dev/draft", 20, "alice draft");
    storage
        .append_history_region_rows("users", std::slice::from_ref(&draft))
        .unwrap();
    assert!(
        storage
            .row_has_history_outside_branch("users", row_id, "dev/main")
            .unwrap(),
        "a version added on another branch must be found from this one"
    );

    // The answer must match the one the scanning default would give.
    for branch in ["dev/main", "dev/draft", "dev/nowhere"] {
        let scanned = storage
            .scan_history_row_batches("users", row_id)
            .unwrap()
            .into_iter()
            .any(|candidate| candidate.branch.as_str() != branch);
        assert_eq!(
            storage
                .row_has_history_outside_branch("users", row_id, branch)
                .unwrap(),
            scanned,
            "the fast answer and the scan disagree about what lies outside {branch}"
        );
    }
}

/// The family `load_history_row_batch_with_family` names is the raw table that HOLDS the
/// batch, whichever step of the history ladder found it. Asked through `Box<dyn Storage>`,
/// the shape every server and client holds its store in: a backend override the box does
/// not forward would answer `None` there and nowhere a concrete store is tested.
///
/// A permission check decodes a write's old row with this family. The row locator is no
/// substitute: it names whichever generation last moved a head, and after a deployment a
/// batch on an older generation's branch sits in that generation's family (local stack
/// 2026-10-18: a delete's tombstone on the first generation's branch was judged against
/// first-generation bytes read as the second generation's, and refused).
///
/// A backend that keeps no families — the bytes are not in any family's raw table — may
/// answer `None`; one that keeps them must name the table they are in.
pub fn test_history_row_batch_family_is_the_table_that_holds_it(
    factory: &dyn Fn() -> Box<dyn Storage>,
) {
    let mut storage = factory();
    let first = seed_row_history_table(storage.as_mut(), "users");
    let second = persist_test_schema(
        storage.as_mut(),
        &SchemaBuilder::new()
            .table(
                TableSchema::builder("users")
                    .column("value", ColumnType::Text)
                    .column("note", ColumnType::Text),
            )
            .build(),
    );
    assert_ne!(
        first, second,
        "control: the two generations must be distinct"
    );
    let row_id = ObjectId::new();
    seed_row_history_locator(storage.as_mut(), "users", row_id, first);
    let batch = make_row_batch(row_id, "dev/main", 10, "alice");
    storage
        .append_history_region_rows("users", std::slice::from_ref(&batch))
        .unwrap();

    let first_table = RowRawTableId::new(RowRawTableKind::History, "users", first)
        .raw_table_name()
        .to_string();
    let key = super::key_codec::history_row_raw_table_key(row_id, "dev/main", batch.batch_id());
    // Where the bytes physically are: every history family of the table, probed for the key.
    // A backend that keeps no families holds them in none, and may answer `None`.
    let holding: Vec<SchemaHash> =
        super::row_raw_table_ids_for_table(storage.as_ref(), RowRawTableKind::History, "users")
            .unwrap()
            .into_iter()
            .filter(|id| {
                storage
                    .raw_table_get(id.raw_table_name(), &key)
                    .unwrap()
                    .is_some()
            })
            .map(|id| id.schema_hash)
            .collect();
    assert!(
        holding.len() <= 1,
        "a batch is held in one family: {holding:?}"
    );
    let expected = holding.first().copied();
    if let Some(family) = expected {
        assert_eq!(
            family, first,
            "control: appended while the locator named the first generation, the batch is in \
             the first generation's family"
        );
    }

    // Generic over the storage, and handed the BOX: `H = Box<dyn Storage>` dispatches
    // through `impl Storage for Box<T>`, as the server's generic code does. Handing it
    // `storage.as_ref()` would call the backend directly and never meet the forward.
    fn family_of<H: Storage + ?Sized>(
        storage: &H,
        row_id: ObjectId,
        batch_id: BatchId,
    ) -> Option<SchemaHash> {
        let (row, family) = storage
            .load_history_row_batch_with_family("users", "dev/main", row_id, batch_id)
            .unwrap()
            .expect("the batch must load");
        assert_eq!(
            Some(row),
            storage
                .load_history_row_batch("users", "dev/main", row_id, batch_id)
                .unwrap(),
            "the row must be the one `load_history_row_batch` returns"
        );
        family
    }

    assert_eq!(
        family_of(&storage, row_id, batch.batch_id()),
        expected,
        "the locator names the family that holds the batch: that family"
    );

    // The next generation moves the row's head, and with it the locator. The batch did
    // not move: the history ladder finds it in its own family.
    seed_row_history_locator(storage.as_mut(), "users", row_id, second);
    assert_eq!(
        family_of(&storage, row_id, batch.batch_id()),
        expected,
        "the locator names another generation: the family is still the one holding the batch"
    );

    if expected.is_some() {
        storage
            .put_history_row_batch_table_locator(
                "dev/main",
                row_id,
                batch.batch_id(),
                Some(&super::ExactRowTableLocator {
                    row_raw_table: first_table.clone().into(),
                    table_name: "users".into(),
                    schema_hash: first,
                }),
            )
            .unwrap();
        assert_eq!(
            family_of(&storage, row_id, batch.batch_id()),
            Some(first),
            "the batch's exact locator names its family: that family"
        );
    }
}

/// A branch that only ever had history written on it must still be found.
///
/// The probe enumerates the branch-ord registry, and the registry is written by
/// seal and local-batch-record persistence — not by history application. If a
/// branch can carry a row's versions without ever being registered, the probe
/// answers "nowhere else" for a row that does live elsewhere, and its caller
/// takes the cheap path with the wrong answer. That is the one direction that
/// changes an answer rather than a cost, so it is pinned here rather than
/// assumed.
pub fn test_history_row_exists_on_an_unregistered_branch(factory: &dyn Fn() -> Box<dyn Storage>) {
    let mut storage = factory();
    let schema_hash = seed_row_history_table(storage.as_mut(), "users");
    let row_id = ObjectId::new();
    seed_row_history_locator(storage.as_mut(), "users", row_id, schema_hash);

    // Written straight into history: no seal, no local batch record, so nothing
    // has taken a branch ord for it.
    let stray = make_row_batch(row_id, "dev/never-sealed", 30, "alice elsewhere");
    storage
        .append_history_region_rows("users", std::slice::from_ref(&stray))
        .unwrap();

    assert!(
        storage
            .row_has_history_outside_branch("users", row_id, "dev/main")
            .unwrap(),
        "a branch that only ever had history written on it must still be seen from another"
    );
    assert!(
        crate::storage::row_has_history_on_another_branch(&storage, "users", row_id, "dev/main",)
            .unwrap(),
        "the row does live on another branch; answering otherwise sends the caller down the \
         cheap path with the wrong answer"
    );
}
