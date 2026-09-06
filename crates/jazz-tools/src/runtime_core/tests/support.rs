//! Fixtures shared by the read-pass and locator gates (v18 items 4/5, design v21 step 4).
//!
//! `split_store` was `locator_warmth.rs`'s; `runtime_over`, `docs_v1` and `docs_v2` existed
//! twice (there and in `locator_ladder_heal.rs`); `read_on` was concrete and is now generic
//! over the storage, because G-C5 drives the same read through `Box<SqliteStorage>`.

use super::*;
use crate::query_manager::manager::LocalUpdates;
use crate::storage::SqliteStorage;
use crate::sync_manager::QueryPropagation;

pub(super) fn docs_v1() -> Schema {
    SchemaBuilder::new()
        .table(
            TableSchema::builder("docs")
                .column("owner", ColumnType::Text)
                .column("body", ColumnType::Text),
        )
        .build()
}

pub(super) fn docs_v2() -> Schema {
    SchemaBuilder::new()
        .table(
            TableSchema::builder("docs")
                .column("owner", ColumnType::Text)
                .column("body", ColumnType::Text),
        )
        .table(TableSchema::builder("tags").column("label", ColumnType::Text))
        .build()
}

pub(super) fn runtime_over(
    schema: Schema,
    app_name: &str,
    storage: SqliteStorage,
) -> RuntimeCore<SqliteStorage, NoopScheduler> {
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

/// The crossing: the core over generation B, the row, and branch A (the branch whose head
/// the derived locator no longer names). The barrier has run: nothing is pending and no
/// transaction is open.
pub(super) fn split_store(
    path: &std::path::Path,
    app_name: &str,
) -> (
    RuntimeCore<SqliteStorage, NoopScheduler>,
    ObjectId,
    BranchName,
) {
    let storage = SqliteStorage::open(path).expect("sqlite storage should open");
    let mut core = runtime_over(docs_v1(), app_name, storage);
    let alice = WriteContext::from_session(Session::new("alice"));
    let ((row_id, _), _) = insert_and_wait_for_batch(
        &mut core,
        "docs",
        HashMap::from([
            ("owner".to_string(), Value::Text("alice".to_string())),
            ("body".to_string(), Value::Text("draft".to_string())),
        ]),
        Some(&alice),
        DurabilityTier::Local,
    )
    .expect("the row inserts under generation A");
    core.batched_tick();
    core.immediate_tick();
    let branch_a = crate::storage::sole_branch_name(core.storage())
        .expect("branch registry readable")
        .expect("the seeded row registered a branch");

    let storage = core.into_storage();
    let mut core = runtime_over(docs_v2(), app_name, storage);
    core.update(
        row_id,
        vec![("body".to_string(), Value::Text("edited".to_string()))],
        Some(&alice),
    )
    .expect("the update across the crossing applies");
    core.batched_tick();
    core.immediate_tick();
    core.batched_tick();
    assert!(
        !core.has_storage_write_pending_flush(),
        "fixture: nothing may be pending for the barrier after the crossing"
    );
    (core, row_id, branch_a)
}

/// One local one-shot read on `branch`: the pass is the `immediate_tick` under `query()`.
/// Generic over BOTH parameters: over the storage so G-C5 can drive it through
/// `Box<SqliteStorage>`, and over the scheduler because G-C4/G-C4′/G-C11 count re-arms and
/// therefore run over `CountingScheduler`, not `NoopScheduler` (found by the compiler when the
/// draft was applied — the dry run only formats).
pub(super) fn read_on<S: Storage, Sch: Scheduler>(
    core: &mut RuntimeCore<S, Sch>,
    branch: &BranchName,
) {
    let (_handle, _future) = core
        .query_with_local_batch_tracked(
            QueryBuilder::new("docs").branch(branch.as_str()).build(),
            None,
            ReadDurabilityOptions {
                tier: None,
                local_updates: LocalUpdates::Immediate,
            },
            QueryPropagation::LocalOnly,
            None,
        )
        .expect("query setup");
}
