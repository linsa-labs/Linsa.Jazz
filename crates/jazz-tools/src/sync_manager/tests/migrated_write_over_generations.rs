//! Gate: the first write a client makes after a schema deployment, onto a row whose
//! history spans two OLDER generations, must be judged as the update it is.
//!
//! Measured live (local stack 2026-10-18, engine linsa-v27). A `users` presence row has
//! visible history on three generation branches: `dev-39b38b18836a-main` and
//! `dev-b12c5d6a3777-main` (nine columns, before `timeZone`) and `dev-97780341566d-main`
//! (ten). After the next deployment every heartbeat is a copy-on-write onto the new
//! branch, so it names no parents. The server answered every one with
//! `permission_denied: Insert denied by policy on table users` — 22 468 beats over a
//! week, and the row's presence froze at the last beat before the deployment.
//!
//! The classification in `inbox.rs`: no parents and no pre-batch row means INSERT. The
//! pre-batch read finds the row on the other branches, but it hands the tips of ALL of
//! them to `visible_row_preview_from_history_rows` as one frontier. No parent links
//! cross a generation boundary, so each branch's tip is a "concurrent" tip, and the
//! merge decodes every tip with ONE descriptor — the incoming write's. The nine-column
//! tips fail with `MalformedData: variable column offset out of bounds`, `.ok().flatten()`
//! reads that as "no old row", and the update is judged as an insert. `users` allows
//! inserts to the backend only, so every beat is refused; tables whose insert policy is
//! as open as their update policy accept the misjudged write, which is why only presence
//! broke.
//!
//! The old row has to be the one READERS serve — one head per branch, each from its own
//! generation's family, a tombstone first, then the newest `(updated_at, batch_id)`. The
//! inbox does not know which branches readers see or how each generation's bytes are
//! shaped, so for a write with no visible ancestor on its own branch over a row that lives
//! on another one, it DEFERS: the check is queued with `OldRowSource::AcrossBranches`, and
//! the query layer reads the old row the way readers do at check time. These gates pin the
//! inbox's half of that contract; the outcome through the real policy path is pinned in
//! `runtime_core/tests/migrated_write_policy.rs`.
//!
//! A write with a parent on its own branch is not deferred — its old row is that parent —
//! but the shape stamped with it has to be the shape the parent's bytes were written in.
//! The locator cannot supply that: it follows whichever generation last moved a head, so
//! one late write from an old client points it at a generation the parent is not in.

use super::cross_generation_visible_split::{send_and_approve, users_next_generation_schema};
use super::*;
use crate::metadata::DeleteKind;
use crate::query_manager::types::branch::ComposedBranchName;
use crate::storage::SqliteStorage;

/// Generation C: generation B's `users` plus an unrelated table — the deployment that
/// does not touch `users` at all (`9778 → 57c2` added the agent-runtime tables) and still
/// moves every row to a new branch.
fn third_generation_schema() -> crate::query_manager::types::Schema {
    SchemaBuilder::new()
        .table(
            TableSchema::builder("users")
                .column("value", ColumnType::Text)
                .column("presence", ColumnType::Text),
        )
        .table(TableSchema::builder("agent_sessions").column("state", ColumnType::Text))
        .build()
}

fn branch_of(schema: &crate::query_manager::types::Schema) -> String {
    ComposedBranchName::new("dev", SchemaHash::compute(schema), "main")
        .to_branch_name()
        .as_str()
        .to_string()
}

fn metadata_of(schema: &crate::query_manager::types::Schema) -> HashMap<String, String> {
    HashMap::from([
        (MetadataKey::Table.to_string(), "users".to_string()),
        (
            MetadataKey::OriginSchemaHash.to_string(),
            SchemaHash::compute(schema).to_string(),
        ),
    ])
}

/// One version of the row, encoded under `schema`'s `users` descriptor. `created_at` is
/// the row's creation on every version, as `RowProvenance::for_update` keeps it.
fn version(
    schema: &crate::query_manager::types::Schema,
    row_id: ObjectId,
    parents: Vec<BatchId>,
    updated_at: u64,
    values: &[&str],
) -> StoredRowBatch {
    let creation = RowProvenance::for_insert(row_id.to_string(), 1_000);
    StoredRowBatch::new(
        row_id,
        branch_of(schema).as_str(),
        parents,
        encode_row(
            &schema[&"users".into()].columns,
            &values
                .iter()
                .map(|value| Value::Text(value.to_string()))
                .collect::<Vec<_>>(),
        )
        .expect("the version should encode under its own generation"),
        RowProvenance::for_update(&creation, row_id.to_string(), updated_at),
        HashMap::new(),
        crate::row_histories::RowState::VisibleDirect,
        None,
    )
}

/// The same version, retired: a soft delete keeps the last content, as the client writes it.
fn soft_deleted(mut row: StoredRowBatch) -> StoredRowBatch {
    row.delete_kind = Some(DeleteKind::Soft);
    row.is_deleted = true;
    row
}

struct Stand {
    io: SqliteStorage,
    sm: SyncManager,
    client_id: ClientId,
    row_id: ObjectId,
    generation_a: crate::query_manager::types::Schema,
    generation_b: crate::query_manager::types::Schema,
    generation_c: crate::query_manager::types::Schema,
    a_beat: StoredRowBatch,
    b_beat: StoredRowBatch,
}

impl Stand {
    /// Generation A creates the row and beats once; generation B copies it onto its own
    /// branch (no parents) and beats once. B's beat is the head readers serve.
    fn seeded() -> Self {
        let generation_a = users_test_schema();
        let generation_b = users_next_generation_schema();
        let generation_c = third_generation_schema();

        let mut io = SqliteStorage::open(":memory:").expect("in-memory sqlite storage should open");
        for schema in [&generation_a, &generation_b, &generation_c] {
            crate::test_support::persist_test_schema(&mut io, schema);
        }

        let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::GlobalServer);
        let client_id = ClientId::new();
        sm.add_client_with_storage(&io, client_id);
        sm.set_client_acks_deliveries(client_id, true);
        sm.set_client_role(client_id, ClientRole::User);
        sm.set_client_session(
            client_id,
            crate::query_manager::session::Session::new("alice"),
        );
        sm.take_outbox();

        let row_id = ObjectId::new();

        let a_created = StoredRowBatch::new(
            row_id,
            branch_of(&generation_a).as_str(),
            Vec::new(),
            encode_row(
                &generation_a[&"users".into()].columns,
                &[Value::Text("a0".to_string())],
            )
            .expect("generation-A row should encode"),
            RowProvenance::for_insert(row_id.to_string(), 1_000),
            HashMap::new(),
            crate::row_histories::RowState::VisibleDirect,
            None,
        );
        send_and_approve(
            &mut sm,
            &mut io,
            client_id,
            metadata_of(&generation_a),
            &a_created,
        );
        let a_beat = version(
            &generation_a,
            row_id,
            vec![a_created.batch_id],
            1_500,
            &["a1"],
        );
        send_and_approve(
            &mut sm,
            &mut io,
            client_id,
            metadata_of(&generation_a),
            &a_beat,
        );

        let b_copied = version(&generation_b, row_id, Vec::new(), 2_000, &["b0", "online"]);
        send_and_approve(
            &mut sm,
            &mut io,
            client_id,
            metadata_of(&generation_b),
            &b_copied,
        );
        let b_beat = version(
            &generation_b,
            row_id,
            vec![b_copied.batch_id],
            2_500,
            &["b1", "online"],
        );
        send_and_approve(
            &mut sm,
            &mut io,
            client_id,
            metadata_of(&generation_b),
            &b_beat,
        );

        let b_served = io
            .load_visible_region_row("users", branch_of(&generation_b).as_str(), row_id)
            .expect("visible read should succeed")
            .expect("control: generation B's beat must be visible on its branch");
        assert_eq!(
            b_served.batch_id(),
            b_beat.batch_id,
            "control: the seed must leave generation B's beat as the row's visible head"
        );

        Self {
            io,
            sm,
            client_id,
            row_id,
            generation_a,
            generation_b,
            generation_c,
            a_beat,
            b_beat,
        }
    }

    /// A client still on generation A beats once more, with a clock OLDER than B's beat.
    /// It moves A's head, so the locator now names generation A, while the head readers
    /// serve is still B's beat.
    fn late_old_generation_beat(&mut self) {
        let late = version(
            &self.generation_a,
            self.row_id,
            vec![self.a_beat.batch_id],
            1_800,
            &["a2"],
        );
        send_and_approve(
            &mut self.sm,
            &mut self.io,
            self.client_id,
            metadata_of(&self.generation_a),
            &late,
        );
        let locator = self
            .io
            .load_row_locator(self.row_id)
            .expect("locator read should succeed")
            .expect("control: the row must have a locator");
        assert_eq!(
            locator.origin_schema_hash,
            Some(SchemaHash::compute(&self.generation_a)),
            "control: the late generation-A write must leave the locator on generation A"
        );
    }

    /// Push `row` as the client's next write and return the permission check it queued.
    fn check_for(
        &mut self,
        schema: &crate::query_manager::types::Schema,
        row: &StoredRowBatch,
    ) -> PendingPermissionCheck {
        self.sm.push_inbox(InboxEntry {
            source: Source::Client(self.client_id),
            payload: SyncPayload::RowBatchCreated {
                metadata: Some(RowMetadata {
                    id: self.row_id,
                    metadata: metadata_of(schema),
                }),
                row: row.clone(),
            },
        });
        self.sm.process_inbox(&mut self.io);
        self.sm
            .take_pending_permission_checks()
            .into_iter()
            .find(|check| {
                matches!(&check.payload, SyncPayload::RowBatchCreated { row: queued, .. }
                    if queued.batch_id == row.batch_id)
            })
            .expect("the write must be queued for a permission check")
    }
}

fn assert_old_row_is(
    check: &PendingPermissionCheck,
    operation: Operation,
    old: &StoredRowBatch,
    old_schema: &crate::query_manager::types::Schema,
) {
    assert_eq!(
        check.operation, operation,
        "a write onto an existing row is an update (or a delete), whatever branch its \
         history sits on: judged as an insert, a table that allows inserts to the backend \
         only refuses it"
    );
    assert_eq!(
        check.old_content.as_deref(),
        Some(old.data.as_ref()),
        "the old row is the one readers see — the newest head across the generations' \
         branches — not a merge of every generation's tip"
    );
    assert_eq!(
        check.old_content_schema_hash,
        Some(SchemaHash::compute(old_schema)),
        "the old row's shape is the generation its bytes were written in, not the one the \
         locator names"
    );
}

/// The inbox's answer for a write whose old row lives only on other branches: no guess at
/// the old row, a deferral to the query layer, and the operation the write is unless that
/// read finds nothing.
fn assert_deferred(check: &PendingPermissionCheck, operation: Operation) {
    assert_eq!(
        check.old_row,
        OldRowSource::AcrossBranches,
        "a write with no visible ancestor on its branch, over a row that lives on another \
         branch, defers its old row to the query layer, which reads it as readers do"
    );
    assert_eq!(
        check.operation, operation,
        "a deferred write is queued as the update (or delete) it is: only the query layer's \
         read, finding no row where readers look, may lower it to an insert"
    );
    assert!(
        check.old_content.is_none(),
        "the inbox must not guess the old row: merging every generation's tips decoded them \
         with one descriptor, and the guess read as \"no old row\": {:?}",
        check.old_content
    );
}

#[test]
fn a_migrated_write_over_two_older_generations_is_deferred_as_an_update() {
    let mut stand = Stand::seeded();
    let c_copied = version(
        &stand.generation_c,
        stand.row_id,
        Vec::new(),
        3_000,
        &["c0", "online"],
    );
    let generation_c = stand.generation_c.clone();
    let check = stand.check_for(&generation_c, &c_copied);
    assert_deferred(&check, Operation::Update);
}

#[test]
fn a_migrated_write_after_a_late_old_generation_beat_is_deferred_as_an_update() {
    let mut stand = Stand::seeded();
    stand.late_old_generation_beat();
    let c_copied = version(
        &stand.generation_c,
        stand.row_id,
        Vec::new(),
        3_000,
        &["c0", "online"],
    );
    let generation_c = stand.generation_c.clone();
    let check = stand.check_for(&generation_c, &c_copied);
    assert_deferred(&check, Operation::Update);
}

#[test]
fn a_migrated_delete_over_two_older_generations_defers_the_old_row() {
    let mut stand = Stand::seeded();
    let c_deleted = soft_deleted(version(
        &stand.generation_c,
        stand.row_id,
        Vec::new(),
        3_000,
        &["b1", "online"],
    ));
    let generation_c = stand.generation_c.clone();
    let check = stand.check_for(&generation_c, &c_deleted);
    assert_deferred(&check, Operation::Delete);
}

#[test]
fn a_parented_update_is_stamped_with_its_parents_generation_not_the_locators() {
    let mut stand = Stand::seeded();
    stand.late_old_generation_beat();
    let b_next = version(
        &stand.generation_b,
        stand.row_id,
        vec![stand.b_beat.batch_id],
        2_600,
        &["b2", "online"],
    );
    let generation_b = stand.generation_b.clone();
    let check = stand.check_for(&generation_b, &b_next);
    assert_old_row_is(
        &check,
        Operation::Update,
        &stand.b_beat,
        &stand.generation_b,
    );
}

/// The same write naming both versions on its branch — the copy and the beat over it — as a
/// device holding both does. The read from the parents finds the beat standing above the
/// copy and returns it as loaded, so its shape is the family it was loaded from too.
#[test]
fn a_write_over_a_version_and_the_one_under_it_is_stamped_with_their_generation() {
    let mut stand = Stand::seeded();
    stand.late_old_generation_beat();
    let b_copied = stand.b_beat.parents[0];
    let b_next = version(
        &stand.generation_b,
        stand.row_id,
        vec![b_copied, stand.b_beat.batch_id],
        2_600,
        &["b2", "online"],
    );
    let generation_b = stand.generation_b.clone();
    let check = stand.check_for(&generation_b, &b_next);
    assert_old_row_is(
        &check,
        Operation::Update,
        &stand.b_beat,
        &stand.generation_b,
    );
}

/// The walk's answer for that write, with the read from the parents switched off: one tip
/// comes back as it was loaded, and its shape is the family it was loaded from.
#[test]
fn the_walk_stamps_a_single_tip_with_the_generation_it_was_loaded_from() {
    let mut stand = Stand::seeded();
    stand.late_old_generation_beat();
    let b_copied = stand.b_beat.parents[0];
    let b_next = version(
        &stand.generation_b,
        stand.row_id,
        vec![b_copied, stand.b_beat.batch_id],
        2_600,
        &["b2", "online"],
    );
    match stand
        .sm
        .pre_batch_row_read(&stand.io, "users", &b_next, false)
    {
        crate::sync_manager::inbox::PreBatchRow::Row { row, shape } => {
            assert_eq!(
                row.batch_id, stand.b_beat.batch_id,
                "control: the walk must answer with the beat the write was made over"
            );
            assert_eq!(
                shape,
                Some(SchemaHash::compute(&stand.generation_b)),
                "the walk returned the beat as loaded, so its shape is the family it was \
                 loaded from, not the generation the locator names"
            );
        }
        other => panic!("the walk must find the beat the write was made over: {other:?}"),
    }
}

/// The cheap answer must stay the true answer: a parentless write over a row that ALSO has
/// a version on another branch is not answered "no old row" from its own branch. It defers,
/// and the query layer decides from the branches readers see (moved here from
/// `runtime_core/tests.rs`, where it asserted that the inbox read the row's whole history —
/// the read the deferral exists to remove, and whose deferral the runtime cannot observe:
/// the query layer drains the inbox and the permission queue in one `process`).
#[test]
fn a_parentless_write_over_a_row_that_also_lives_on_another_branch_is_deferred() {
    let schema = users_test_schema();
    let mut io = MemoryStorage::new();
    crate::test_support::persist_test_schema(&mut io, &schema);

    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::GlobalServer);
    let client_id = ClientId::new();
    sm.add_client_with_storage(&io, client_id);
    sm.set_client_acks_deliveries(client_id, true);
    sm.set_client_role(client_id, ClientRole::User);
    sm.set_client_session(
        client_id,
        crate::query_manager::session::Session::new("alice"),
    );
    sm.take_outbox();

    let row_id = ObjectId::new();
    let created = StoredRowBatch::new(
        row_id,
        branch_of(&schema).as_str(),
        Vec::new(),
        encode_row(
            &schema[&"users".into()].columns,
            &[Value::Text("v0".to_string())],
        )
        .expect("the row should encode"),
        RowProvenance::for_insert(row_id.to_string(), 1_000),
        HashMap::new(),
        crate::row_histories::RowState::VisibleDirect,
        None,
    );
    send_and_approve(&mut sm, &mut io, client_id, metadata_of(&schema), &created);

    // THIS row also exists on another branch.
    let other_branch = crate::object::BranchName::new("dev-ffffffffffff-main");
    io.resolve_or_alloc_branch_ord(other_branch)
        .expect("register the other branch");
    let mut elsewhere = version(
        &schema,
        row_id,
        Vec::new(),
        2_000,
        &["the same row, elsewhere"],
    );
    elsewhere.branch = other_branch.as_str().into();
    io.append_history_region_rows("users", std::slice::from_ref(&elsewhere))
        .expect("the row also has a version on the other branch");

    let parentless = version(
        &schema,
        row_id,
        Vec::new(),
        9_999,
        &["from a diverged client"],
    );
    sm.push_inbox(InboxEntry {
        source: Source::Client(client_id),
        payload: SyncPayload::RowBatchCreated {
            metadata: Some(RowMetadata {
                id: row_id,
                metadata: metadata_of(&schema),
            }),
            row: parentless.clone(),
        },
    });
    sm.process_inbox(&mut io);
    let check = sm
        .take_pending_permission_checks()
        .into_iter()
        .find(|check| {
            matches!(&check.payload, SyncPayload::RowBatchCreated { row, .. }
                if row.batch_id == parentless.batch_id)
        })
        .expect("the write must be queued for a permission check");
    assert_deferred(&check, Operation::Update);
}
