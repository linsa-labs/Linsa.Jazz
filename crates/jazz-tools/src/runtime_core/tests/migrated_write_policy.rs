//! Gate (local stack 2026-10-18, engine linsa-v27): the owner's update must survive a
//! SECOND schema deployment, on a table whose inserts belong to the backend.
//!
//! `users` is that table: `allowInsert` is backend-only, `allowUpdate` is the owner's.
//! After the deployment that added the agent-runtime tables (`9778 → 57c2`, which did not
//! touch `users`), every presence beat from every upgraded device was refused with
//! `Insert denied by policy on table users` — 22 468 over a week — and every user's
//! "last seen" froze at the deployment.
//!
//! The row's history sat on two older generations' branches with different shapes. The
//! upgraded device's update is a copy-on-write onto its own branch, so it names no
//! parents; the server derives the old row from the other branches' tips, merged as one
//! frontier and decoded with one descriptor. The older shape does not decode under it,
//! the error reads as "no old row", and a parentless write with no old row is an INSERT.
//! After only ONE deployment the bug is invisible: every tip then has one shape — which
//! is why `an_update_onto_an_old_shelf_row_survives_its_using_policy` stays green.
//!
//! Real runtimes on both ends and the real policy path; the verdict is the server's own
//! fate. SqliteStorage, because `MemoryStorage` keeps visible rows as structs and never
//! runs the raw-table family reads this crosses.

use super::*;

use crate::batch_fate::BatchFate;
use crate::metadata::{MetadataKey, RowProvenance};
use crate::query_manager::types::branch::ComposedBranchName;
use crate::row_histories::{RowState, StoredRowBatch};
use crate::storage::{MemoryStorage, SqliteStorage};
use crate::sync_manager::RowMetadata;

const APP: &str = "migrated-write-policy";
const TABLE: &str = "documents";

fn owner_policies() -> TablePolicies {
    let owner = || PolicyExpr::eq_session("owner_id", vec!["user_id".into()]);
    TablePolicies::new()
        .with_select(owner())
        .with_insert(owner())
        .with_update(Some(owner()), owner())
        .with_delete(owner())
}

/// The `users` shape: the owner may update and delete, nobody but the backend inserts.
fn backend_insert_policies() -> TablePolicies {
    let owner = || PolicyExpr::eq_session("owner_id", vec!["user_id".into()]);
    TablePolicies::new()
        .with_select(owner())
        .with_insert(PolicyExpr::False)
        .with_update(Some(owner()), owner())
        .with_delete(owner())
}

fn v1() -> Schema {
    SchemaBuilder::new()
        .table(
            TableSchema::builder(TABLE)
                .column("owner_id", ColumnType::Text)
                .column("title", ColumnType::Text),
        )
        .build()
}

/// The first deployment adds a column: a second shape.
fn v2() -> Schema {
    SchemaBuilder::new()
        .table(
            TableSchema::builder(TABLE)
                .column("owner_id", ColumnType::Text)
                .column("title", ColumnType::Text)
                .column("note", ColumnType::Text),
        )
        .build()
}

/// The second deployment leaves `documents`' shape alone and adds a table, as
/// `9778 → 57c2` did.
fn v3() -> Schema {
    SchemaBuilder::new()
        .table(
            TableSchema::builder(TABLE)
                .column("owner_id", ColumnType::Text)
                .column("title", ColumnType::Text)
                .column("note", ColumnType::Text),
        )
        .table(TableSchema::builder("agent_sessions").column("state", ColumnType::Text))
        .build()
}

/// The permissions head the server enforces for a structural schema: the same tables
/// carrying row policies. It must DIFFER from the runtime schema, or the explicit
/// authorization path never runs (`the_explicit_auth_using_arm_is_reachable`). Before
/// the second deployment (`v1`, `v2`) the owner inserts; from it on — `v3`, and every
/// generation the differential below mints — `documents` has `users`' policy, where
/// inserts are the backend's.
fn permissions_for(schema: &Schema) -> Schema {
    let hash = SchemaHash::compute(schema);
    let policies = if hash == SchemaHash::compute(&v1()) || hash == SchemaHash::compute(&v2()) {
        owner_policies()
    } else {
        backend_insert_policies()
    };
    let mut head = schema.clone();
    for (name, table) in head.iter_mut() {
        if name.as_str() == TABLE {
            table.policies = policies.clone();
        } else {
            table.policies = TablePolicies::new()
                .with_select(PolicyExpr::True)
                .with_insert(PolicyExpr::True)
                .with_update(Some(PolicyExpr::True), PolicyExpr::True)
                .with_delete(PolicyExpr::True);
        }
    }
    head
}

type Core<S> = RuntimeCore<S, NoopScheduler>;

fn sqlite() -> SqliteStorage {
    SqliteStorage::open(":memory:").expect("in-memory sqlite storage should open")
}

/// `MemoryStorage` keeps rows per logical table and overrides the visible reads, so the
/// raw-table family machinery never runs there; the old row has to be found on it too.
fn memory() -> MemoryStorage {
    MemoryStorage::new()
}

fn server_runtime<S: Storage>(schema: Schema, storage: S) -> Core<S> {
    let mut core = runtime(schema.clone(), storage);
    core.schema_manager_mut()
        .query_manager_mut()
        .set_authorization_schema(permissions_for(&schema));
    core.schema_manager_mut()
        .query_manager_mut()
        .require_authorization_schema();
    core.immediate_tick();
    core
}

fn runtime<S: Storage>(schema: Schema, storage: S) -> Core<S> {
    let app_id = AppId::from_name(APP);
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

/// One-shot bidirectional pump (as `cross_generation_policy::pump`).
fn pump<S: Storage>(
    server: &mut Core<S>,
    server_id: ServerId,
    client: &mut Core<S>,
    client_id: ClientId,
) {
    for _ in 0..12 {
        let mut any = false;
        client.batched_tick();
        for entry in client.sync_sender().take() {
            if entry.destination == Destination::Server(server_id) {
                any = true;
                server.park_sync_message(InboxEntry {
                    source: Source::Client(client_id),
                    payload: entry.payload,
                });
            }
        }
        server.batched_tick();
        server.immediate_tick();
        server.batched_tick();
        for entry in server.sync_sender().take() {
            if entry.destination == Destination::Client(client_id) {
                any = true;
                client.park_sync_message(InboxEntry {
                    source: Source::Server(server_id),
                    payload: entry.payload,
                });
            }
        }
        client.batched_tick();
        client.immediate_tick();
        if !any {
            break;
        }
    }
}

struct World<S: Storage> {
    fresh: fn() -> S,
    server: Core<S>,
    client: Core<S>,
    server_id: ServerId,
    client_id: ClientId,
    row_id: ObjectId,
    schema: Schema,
    /// Branches `copy_on_another_user_branch` put a copy of the row on: beside the
    /// readers by design, and the only ones the model's controls may skip.
    beside_readers: Vec<String>,
}

impl<S: Storage> World<S> {
    fn alice() -> Session {
        Session::new("alice")
    }

    fn new(fresh: fn() -> S) -> Self {
        let mut server = server_runtime(v1(), fresh());
        let mut client = runtime(v1(), fresh());
        let server_id = ServerId::new();
        let client_id = ClientId::new();
        server.add_client(client_id, Some(Self::alice()));
        client.add_server(server_id);

        let ((row_id, _), _) = client
            .insert(
                TABLE,
                HashMap::from([
                    ("owner_id".to_string(), Value::Text("alice".to_string())),
                    ("title".to_string(), Value::Text("draft".to_string())),
                ]),
                Some(&WriteContext::from_session(Self::alice())),
            )
            .expect("the owner's insert satisfies her own policy");
        let mut world = Self {
            fresh,
            server,
            client,
            server_id,
            client_id,
            row_id,
            schema: v1(),
            beside_readers: Vec::new(),
        };
        world.sync();
        let rejections = world.rejections();
        assert!(
            rejections.is_empty(),
            "control: the seed insert was rejected ({rejections:?})"
        );
        world
    }

    fn sync(&mut self) {
        pump(
            &mut self.server,
            self.server_id,
            &mut self.client,
            self.client_id,
        );
    }

    fn rejections(&mut self) -> Vec<String> {
        self.server
            .sync_sender()
            .take()
            .into_iter()
            .filter_map(|entry| match entry.payload {
                SyncPayload::BatchFate {
                    fate: BatchFate::Rejected { reason, .. },
                } => Some(reason),
                _ => None,
            })
            .collect()
    }

    /// A deployment: both ends rebuilt on `next` from their own stores, the lens
    /// published server-side, and the reconnect handshake in both directions.
    fn deploy(&mut self, next: Schema) {
        let lens = crate::schema_manager::auto_lens::generate_lens(&self.schema, &next);
        let server_storage =
            std::mem::replace(&mut self.server, runtime(v1(), (self.fresh)())).into_storage();
        let client_storage =
            std::mem::replace(&mut self.client, runtime(v1(), (self.fresh)())).into_storage();
        self.server = server_runtime(next.clone(), server_storage);
        self.server.publish_lens(&lens).expect("lens publishes");
        self.client = runtime(next.clone(), client_storage);
        self.server.add_client(self.client_id, Some(Self::alice()));
        self.client.add_server(self.server_id);
        self.server.immediate_tick();
        self.client.immediate_tick();
        self.schema = next;
        self.sync();
        self.server.sync_sender().take();
        self.client.sync_sender().take();
    }

    /// The owner retitles her row from her device; returns the server's refusals.
    fn retitle(&mut self, title: &str) -> Vec<String> {
        self.client
            .update(
                self.row_id,
                vec![("title".to_string(), Value::Text(title.to_string()))],
                Some(&WriteContext::from_session(Self::alice())),
            )
            .expect("the owner's update applies locally");
        self.push_client_writes()
    }

    /// The owner deletes her row from her device; returns the server's refusals.
    fn remove(&mut self) -> Vec<String> {
        self.client
            .delete(
                self.row_id,
                Some(&WriteContext::from_session(Self::alice())),
            )
            .expect("the owner's delete applies locally");
        self.push_client_writes()
    }

    /// The device's queued frames, delivered to the server once; the server's refusals.
    fn push_client_writes(&mut self) -> Vec<String> {
        self.client.batched_tick();
        for entry in self.client.sync_sender().take() {
            if entry.destination == Destination::Server(self.server_id) {
                self.server.park_sync_message(InboxEntry {
                    source: Source::Client(self.client_id),
                    payload: entry.payload,
                });
            }
        }
        self.server.batched_tick();
        self.server.immediate_tick();
        self.server.batched_tick();
        self.rejections()
    }

    /// A frame from a connection the server knows as `session`, with `role`; the
    /// server's refusals.
    fn push_raw(
        &mut self,
        session: &str,
        role: ClientRole,
        schema: &Schema,
        row: StoredRowBatch,
    ) -> Vec<String> {
        self.push_raw_answers(session, role, schema, row).0
    }

    /// [`Self::push_raw`], with the batches the server asked that connection to resend
    /// (`BatchFate::Missing`): the parents of a batch it parked on `ParentNotFound`.
    fn push_raw_answers(
        &mut self,
        session: &str,
        role: ClientRole,
        schema: &Schema,
        row: StoredRowBatch,
    ) -> (Vec<String>, Vec<BatchId>) {
        let connection = ClientId::new();
        self.server
            .add_client(connection, Some(Session::new(session)));
        self.server
            .schema_manager_mut()
            .query_manager_mut()
            .sync_manager_mut()
            .set_client_role(connection, role);
        self.server.sync_sender().take();
        self.server.park_sync_message(InboxEntry {
            source: Source::Client(connection),
            payload: SyncPayload::RowBatchCreated {
                metadata: Some(RowMetadata {
                    id: row.row_id,
                    metadata: HashMap::from([
                        (MetadataKey::Table.to_string(), TABLE.to_string()),
                        (
                            MetadataKey::OriginSchemaHash.to_string(),
                            SchemaHash::compute(schema).to_string(),
                        ),
                    ]),
                }),
                row,
            },
        });
        self.server.batched_tick();
        self.server.immediate_tick();
        self.server.batched_tick();
        let mut refusals = Vec::new();
        let mut asked = Vec::new();
        for entry in self.server.sync_sender().take() {
            match entry.payload {
                SyncPayload::BatchFate {
                    fate: BatchFate::Rejected { reason, .. },
                } => refusals.push(reason),
                SyncPayload::BatchFate {
                    fate: BatchFate::Missing { batch_id },
                } if entry.destination == Destination::Client(connection) => asked.push(batch_id),
                _ => {}
            }
        }
        (refusals, asked)
    }

    /// The server's newest PARENTED version of the row on `branch`: that generation's
    /// last beat.
    fn last_beat_on(&self, branch: &str) -> StoredRowBatch {
        self.server
            .storage()
            .scan_history_row_batches(TABLE, self.row_id)
            .expect("history scan should succeed")
            .into_iter()
            .filter(|row| row.branch.as_str() == branch && !row.parents.is_empty())
            .max_by_key(|row| (row.updated_at, row.batch_id()))
            .expect("the generation has a beat on its branch")
    }

    /// The row with one generation behind it and a beat on each: the presence row's
    /// shape just before the deployment that broke it.
    fn through_two_generations(fresh: fn() -> S) -> (Self, String, String) {
        let mut world = Self::new(fresh);
        // Every generation's history ends in a PARENTED version, as a presence row's
        // does: a beat on top of whatever started that generation's chain. Without it
        // each older generation's only version is its parentless copy,
        // `elided_snapshot_dominator` collapses the frontier to the newest copy, nothing
        // is decoded, and the gates pass for a reason that is not the fix.
        let beat = world.retitle("beat on the first generation");
        assert!(
            beat.is_empty(),
            "control: a same-generation update was refused ({beat:?}); the observation \
             channel is broken"
        );
        let v1_branch = world.branch();

        world.deploy(v2());
        let after_first = world.retitle("after the first deployment");
        assert!(
            after_first.is_empty(),
            "the copy-on-write over ONE older generation was refused ({after_first:?}). \
             Not only a control: its old row lives on another branch, so it is deferred and \
             read by the query layer like every first write after a deployment (v27 got it \
             right only because the snapshot dominator collapsed the frontier)"
        );
        let beat = world.retitle("beat on the second generation");
        assert!(
            beat.is_empty(),
            "control: a same-generation update after the first deployment was refused \
             ({beat:?})"
        );
        let v2_branch = world.branch();
        (world, v1_branch, v2_branch)
    }

    fn deploy_the_second_time(&mut self, v2_branch: &str) {
        self.deploy(v3());
        assert_ne!(
            self.branch(),
            v2_branch,
            "control: the second deployment must mint a new branch, or nothing crosses"
        );
    }

    fn branch(&self) -> String {
        self.server
            .schema_manager()
            .branch_name()
            .as_str()
            .to_string()
    }

    /// The title the server serves on its current branch, decoded in its own shape.
    fn served_title(&self) -> Option<String> {
        let row = self
            .server
            .storage()
            .load_visible_region_row(TABLE, &self.branch(), self.row_id)
            .expect("visible read should succeed")?;
        let values =
            crate::row_format::decode_row(&self.schema[&TableName::new(TABLE)].columns, &row.data)
                .expect("the served row decodes in the server's current shape");
        match values.get(1) {
            Some(Value::Text(title)) => Some(title.clone()),
            _ => None,
        }
    }
}

#[test]
fn the_owners_update_survives_a_second_deployment_on_a_backend_insert_table() {
    owners_update_survives_a_second_deployment(sqlite);
}

#[test]
#[ignore = "separate defect, observed 2026-10-18 and not yet diagnosed: on MemoryStorage the \
            CONTROL fails first, a same-generation update after ONE deployment is refused with \
            'Update denied by USING policy - cannot see old row'"]
fn the_owners_update_survives_a_second_deployment_on_memory_storage() {
    owners_update_survives_a_second_deployment(memory);
}

fn owners_update_survives_a_second_deployment<S: Storage>(fresh: fn() -> S) {
    let (mut world, _v1_branch, v2_branch) = World::through_two_generations(fresh);
    world.deploy_the_second_time(&v2_branch);
    let after_second = world.retitle("after the second deployment");
    assert!(
        after_second.is_empty(),
        "the owner's update after a second deployment was refused: {after_second:?} — \
         the old row spans two shapes, the server could not read it, and judged the \
         update an insert, which only the backend may do"
    );
    assert_eq!(
        world.served_title().as_deref(),
        Some("after the second deployment"),
        "the approved update must be what the server serves"
    );

    // Negative control: a non-owner's parentless write over the same row must stay
    // refused — "approve whatever has history somewhere" passes everything above.
    let mallory_client_id = ClientId::new();
    world
        .server
        .add_client(mallory_client_id, Some(Session::new("mallory")));
    world.server.sync_sender().take();
    let stolen = StoredRowBatch::new(
        world.row_id,
        world.branch().as_str(),
        Vec::new(),
        encode_row(
            &v3()[&TableName::new(TABLE)].columns,
            &[
                Value::Text("alice".to_string()),
                Value::Text("stolen".to_string()),
                Value::Text(String::new()),
            ],
        )
        .expect("row encodes"),
        RowProvenance::for_insert(world.row_id.to_string(), 9_999),
        HashMap::new(),
        RowState::VisibleDirect,
        None,
    );
    world.server.park_sync_message(InboxEntry {
        source: Source::Client(mallory_client_id),
        payload: SyncPayload::RowBatchCreated {
            metadata: Some(RowMetadata {
                id: world.row_id,
                metadata: HashMap::from([
                    (MetadataKey::Table.to_string(), TABLE.to_string()),
                    (
                        MetadataKey::OriginSchemaHash.to_string(),
                        SchemaHash::compute(&v3()).to_string(),
                    ),
                ]),
            }),
            row: stolen,
        },
    });
    world.server.batched_tick();
    world.server.immediate_tick();
    world.server.batched_tick();
    let refused = world.rejections();
    assert_eq!(
        refused.len(),
        1,
        "mallory's write over alice's row was not refused: {refused:?}"
    );
    assert_eq!(
        world.served_title().as_deref(),
        Some("after the second deployment"),
        "the refused write landed on the server's row anyway"
    );
}

/// The locator follows whichever generation last moved a head, so one late beat from a
/// device still on the FIRST generation points it back there — while the head every
/// reader serves is still the second generation's newer beat. That late beat also hands
/// the row to mallory, so an old row taken from the locator's generation (or a merge
/// that lets it win) fails the owner's USING check; only the reader's winner passes it.
#[test]
fn the_owners_update_after_a_late_first_generation_beat_still_sees_the_newest_head() {
    let (mut world, v1_branch, v2_branch) = World::through_two_generations(sqlite);

    let first_beat = world.last_beat_on(&v1_branch);
    let second_beat = world.last_beat_on(&v2_branch);
    let late_at = first_beat.updated_at + 1;
    assert!(
        late_at < second_beat.updated_at,
        "control: the late first-generation beat must be OLDER than the second \
         generation's, or it is the rightful head"
    );
    let late = StoredRowBatch::new(
        world.row_id,
        v1_branch.as_str(),
        vec![first_beat.batch_id()],
        encode_row(
            &v1()[&TableName::new(TABLE)].columns,
            &[
                Value::Text("mallory".to_string()),
                Value::Text("late first-generation beat".to_string()),
            ],
        )
        .expect("row encodes"),
        RowProvenance {
            created_by: first_beat.created_by.clone(),
            created_at: first_beat.created_at,
            updated_by: "backend".into(),
            updated_at: late_at,
        },
        HashMap::new(),
        RowState::VisibleDirect,
        None,
    );
    let refused = world.push_raw("backend", ClientRole::Backend, &v1(), late);
    assert!(
        refused.is_empty(),
        "control: the backend's late first-generation write was refused ({refused:?})"
    );
    assert_eq!(
        world
            .server
            .storage()
            .load_row_locator(world.row_id)
            .expect("locator readable")
            .and_then(|locator| locator.origin_schema_hash),
        Some(SchemaHash::compute(&v1())),
        "control: the late write must leave the locator on the first generation"
    );

    world.deploy_the_second_time(&v2_branch);
    let refused = world.retitle("after the second deployment");
    assert!(
        refused.is_empty(),
        "the owner's update was refused: {refused:?} — the old row it was judged against \
         is not the newest head readers serve (the second generation's beat, owned by \
         alice), but the late first-generation copy that names mallory"
    );
    assert_eq!(
        world.served_title().as_deref(),
        Some("after the second deployment"),
        "the approved update must be what the server serves"
    );
}

/// ONE deployment, on a table whose inserts are the backend's: the owner's first write on
/// the new branch is a copy-on-write over a row whose history is all on the previous
/// generation's branch. v27 judged it right only because the snapshot dominator collapsed
/// the old branch's frontier; with the fix it is deferred like every first write after a
/// deployment, and the query layer reads the old row across the generations' branches.
#[test]
fn the_owners_first_write_after_one_deployment_is_judged_against_the_previous_generation() {
    let mut world = World::new(sqlite);
    let beat = world.retitle("beat on the first generation");
    assert!(
        beat.is_empty(),
        "control: a same-generation update was refused ({beat:?})"
    );
    let next = next_generation(&v1(), 1, true);
    world.deploy(next);
    let refused = world.retitle("after one deployment");
    assert!(
        refused.is_empty(),
        "the owner's first write after one deployment was refused: {refused:?}"
    );
    assert_eq!(
        world.served_title().as_deref(),
        Some("after one deployment"),
        "the approved update must be what the server serves"
    );
}

/// A delete has no recovery at all when the old row is wrong: the DELETE arm judges the
/// old row, and its refusal is a persisted fate.
///
/// The device's delete after two deployments is not one write. It retires the row on every
/// generation's branch: a parentless tombstone on the current branch, and on each OLDER
/// branch a tombstone over that branch's own head, with that head as its single parent. The
/// older ones are judged against their parent — bytes in their own generation's shape — and
/// stamped with the shape the row LOCATOR names, which is the newest generation that moved a
/// head. Measured on this gate (10-18): the first generation's tombstone reached the policy
/// with first-generation bytes stamped as the second generation's, decoded into the wrong
/// owner, and was refused.
#[test]
fn the_owners_delete_survives_a_second_deployment() {
    let (mut world, _v1_branch, v2_branch) = World::through_two_generations(sqlite);
    world.deploy_the_second_time(&v2_branch);

    let refused = world.remove();
    assert!(
        refused.is_empty(),
        "the owner's delete after a second deployment was refused: {refused:?}"
    );
    let served = world
        .server
        .storage()
        .load_visible_region_row(TABLE, &world.branch(), world.row_id)
        .expect("visible read should succeed");
    assert!(
        served.as_ref().is_none_or(|row| row.is_deleted),
        "the approved delete must retire the row on the server's branch: {served:?}"
    );
}

/// The old row is the one READERS serve, and readers of `main` never see another user
/// branch: `all_branch_names()` is one `(env, user_branch)` across the live generations.
/// A row that exists only on `feature` does not exist for a writer on `main`, so a
/// parentless write of that id there is an insert — and on a table whose inserts are the
/// backend's, it is refused, whoever owns the copy on `feature`.
#[test]
fn a_row_that_lives_only_on_another_user_branch_is_not_an_old_row_on_main() {
    let (mut world, _v1_branch, v2_branch) = World::through_two_generations(sqlite);
    world.deploy_the_second_time(&v2_branch);

    let row_id = ObjectId::new();
    let feature = ComposedBranchName::new("dev", SchemaHash::compute(&v3()), "feature")
        .to_branch_name()
        .as_str()
        .to_string();
    let encode = |title: &str| {
        encode_row(
            &v3()[&TableName::new(TABLE)].columns,
            &[
                Value::Text("alice".to_string()),
                Value::Text(title.to_string()),
                Value::Text(String::new()),
            ],
        )
        .expect("row encodes")
    };
    let elsewhere = StoredRowBatch::new(
        row_id,
        feature.as_str(),
        Vec::new(),
        encode("on another user branch"),
        RowProvenance::for_insert("alice".to_string(), 1_000),
        HashMap::new(),
        RowState::VisibleDirect,
        None,
    );
    let refused = world.push_raw("backend", ClientRole::Backend, &v3(), elsewhere);
    assert!(
        refused.is_empty(),
        "control: the backend's write on the other user branch was refused ({refused:?})"
    );

    let on_main = StoredRowBatch::new(
        row_id,
        world.branch().as_str(),
        Vec::new(),
        encode("on main"),
        RowProvenance::for_insert("alice".to_string(), 2_000),
        HashMap::new(),
        RowState::VisibleDirect,
        None,
    );
    let refused = world.push_raw("alice", ClientRole::User, &v3(), on_main);
    assert!(
        refused
            .iter()
            .any(|reason| reason.contains("Insert denied")),
        "a write on `main` was judged against a row only `feature` holds — a row no reader \
         of `main` serves: {refused:?}"
    );
}

impl World<SqliteStorage> {
    /// A version of the row on another user branch of the current generation: history the
    /// inbox sees on "another branch", and a branch no reader of `main` serves. Old rows
    /// read across branches find nothing outside `main` then.
    fn copy_on_another_user_branch(&mut self) {
        let schema = self.schema.clone();
        self.copy_on_another_user_branch_of(&schema);
    }

    /// [`Self::copy_on_another_user_branch`] on `schema`'s generation, in its shape.
    fn copy_on_another_user_branch_of(&mut self, schema: &Schema) {
        let feature = ComposedBranchName::new("dev", SchemaHash::compute(schema), "feature")
            .to_branch_name()
            .as_str()
            .to_string();
        let columns = &schema[&TableName::new(TABLE)].columns;
        let mut values = vec![
            Value::Text("alice".to_string()),
            Value::Text("on another user branch".to_string()),
        ];
        values.resize(columns.columns.len(), Value::Text(String::new()));
        let copy = StoredRowBatch::new(
            self.row_id,
            feature.as_str(),
            Vec::new(),
            encode_row(columns, &values).expect("row encodes"),
            RowProvenance::for_insert("alice".to_string(), 1_000),
            HashMap::new(),
            RowState::VisibleDirect,
            None,
        );
        let refused = self.push_raw("backend", ClientRole::Backend, schema, copy);
        assert!(
            refused.is_empty(),
            "control: the backend's copy on another user branch was refused ({refused:?})"
        );
        self.beside_readers.push(feature);
    }

    fn encode(&self, owner: &str, title: &str) -> Vec<u8> {
        let columns = &self.schema[&TableName::new(TABLE)].columns;
        let mut values = vec![
            Value::Text(owner.to_string()),
            Value::Text(title.to_string()),
        ];
        values.resize(columns.columns.len(), Value::Text(String::new()));
        encode_row(columns, &values).expect("row encodes")
    }

    /// The head the server serves on its current branch.
    fn served_head(&self) -> StoredRowBatch {
        self.server
            .storage()
            .load_visible_region_row(TABLE, &self.branch(), self.row_id)
            .expect("visible read should succeed")
            .expect("the row is served on the current branch")
    }

    /// A write of alice's on the current branch naming `parents`, none of them a version
    /// the server serves; the server's refusals.
    fn alices_write_over(&mut self, parents: Vec<BatchId>, deleted: bool) -> Vec<String> {
        self.alices_write_over_answers(parents, deleted).0
    }

    /// [`Self::alices_write_over`], with the parents the server asked her to resend.
    fn alices_write_over_answers(
        &mut self,
        parents: Vec<BatchId>,
        deleted: bool,
    ) -> (Vec<String>, Vec<BatchId>) {
        let head = self.served_head();
        let mut row = StoredRowBatch::new(
            self.row_id,
            self.branch().as_str(),
            parents,
            self.encode("alice", "over versions the server does not serve"),
            RowProvenance {
                created_by: head.created_by.clone(),
                created_at: head.created_at,
                updated_by: "alice".into(),
                updated_at: head.updated_at + 1_000,
            },
            HashMap::new(),
            RowState::VisibleDirect,
            None,
        );
        row.is_deleted = deleted;
        let schema = self.schema.clone();
        self.push_raw_answers("alice", ClientRole::User, &schema, row)
    }
}

/// A tombstone whose parents have not arrived must PARK, judged against the head readers
/// serve — here the incoming branch's own, because the only other copy of the row is on
/// a branch no reader of `main` serves. The inbox defers it (history on another branch);
/// reading only OUTSIDE the incoming branch finds nothing, the delete reaches its arm
/// without old content, and that refusal is a persisted fate: the delete is gone for good,
/// whatever arrives after it. (Revision 6 — `a_tombstone_with_absent_parents_…` in
/// `runtime_core/tests.rs` is the same shape on MemoryStorage.)
#[test]
fn a_tombstone_whose_parents_have_not_arrived_is_judged_against_the_head_readers_serve() {
    let mut world = World::new(sqlite);
    world.copy_on_another_user_branch();
    let head = world.served_head();

    let parents = vec![BatchId::new(), BatchId::new()];
    let (refused, asked) = world.alices_write_over_answers(parents.clone(), true);
    assert!(
        refused.is_empty(),
        "alice's delete of her own row was refused before its parents arrived: {refused:?}"
    );
    assert!(
        asked.iter().any(|batch_id| parents.contains(batch_id)),
        "the delete must be PARKED on its absent parents — the server asks for them — not \
         left undecided: asked for {asked:?}, parents {parents:?}"
    );
    let served = world.served_head();
    assert!(
        served.batch_id() == head.batch_id() && !served.is_deleted,
        "control: a tombstone whose parents are absent must park, not apply: {served:?}"
    );
}

/// The negative of the gate above: the head is judged, not waved through. A backend beat
/// hands the row to mallory on `main`; alice's tombstone over parents the server does not
/// hold is then refused BY POLICY, against mallory's head — not for missing old content.
#[test]
fn a_tombstone_whose_parents_have_not_arrived_is_refused_over_someone_elses_head() {
    let mut world = World::new(sqlite);
    world.copy_on_another_user_branch();
    let head = world.served_head();
    let handed_over = StoredRowBatch::new(
        world.row_id,
        world.branch().as_str(),
        vec![head.batch_id()],
        world.encode("mallory", "handed to mallory"),
        RowProvenance {
            created_by: head.created_by.clone(),
            created_at: head.created_at,
            updated_by: "backend".into(),
            updated_at: head.updated_at + 1,
        },
        HashMap::new(),
        RowState::VisibleDirect,
        None,
    );
    let refused = world.push_raw("backend", ClientRole::Backend, &v1(), handed_over);
    assert!(
        refused.is_empty(),
        "control: the backend's beat on main was refused ({refused:?})"
    );

    let refused = world.alices_write_over(vec![BatchId::new(), BatchId::new()], true);
    assert!(
        refused
            .iter()
            .any(|reason| reason.contains("Delete denied by policy")),
        "alice's delete over mallory's head must be refused by the policy, judged against \
         that head: {refused:?}"
    );
}

/// An update naming two versions the server holds but does not serve (refused ones, with
/// no visible ancestor) is judged against the head readers serve, in that head's shape.
///
/// The row is born on the SECOND generation, so no other reader branch holds it, and a
/// backend copy in the FIRST generation's shape on a branch no reader serves moves the row
/// locator to the first generation. Reading only outside the incoming branch finds
/// nothing; the update arm then refills its old content from the incoming branch's head —
/// the same bytes — but stamps them with the locator's generation, and the second
/// generation's bytes are decoded as the first's. Judged against the head readers serve,
/// stamped with the branch it was served from, alice's update over her own row passes.
#[test]
fn an_update_over_two_refused_versions_is_judged_against_the_head_readers_serve() {
    let mut world = World::new(sqlite);
    world.deploy(v2());
    let ((row_id, _), _) = world
        .client
        .insert(
            TABLE,
            HashMap::from([
                ("owner_id".to_string(), Value::Text("alice".to_string())),
                (
                    "title".to_string(),
                    Value::Text("born on the second generation".to_string()),
                ),
                (
                    "note".to_string(),
                    Value::Text("a note only this shape has".to_string()),
                ),
            ]),
            Some(&WriteContext::from_session(World::<SqliteStorage>::alice())),
        )
        .expect("the owner's insert satisfies her own policy");
    world.row_id = row_id;
    world.sync();
    let refused = world.rejections();
    assert!(
        refused.is_empty(),
        "control: the second-generation insert was refused ({refused:?})"
    );
    world.copy_on_another_user_branch_of(&v1());
    assert_eq!(
        world
            .server
            .storage()
            .load_row_locator(world.row_id)
            .expect("locator readable")
            .and_then(|locator| locator.origin_schema_hash),
        Some(SchemaHash::compute(&v1())),
        "control: the first-generation copy must move the locator to the first generation, \
         or the refill's stamp is right by coincidence and this gates nothing"
    );
    let head = world.served_head();
    let branch = world.branch();
    let refused_versions: Vec<BatchId> = (0..2)
        .map(|n| {
            let row = StoredRowBatch::new(
                world.row_id,
                branch.as_str(),
                Vec::new(),
                world.encode("mallory", &format!("refused {n}")),
                RowProvenance::for_insert("mallory".to_string(), head.updated_at + 1 + n),
                HashMap::new(),
                RowState::Rejected,
                None,
            );
            let batch_id = row.batch_id();
            crate::test_support::apply_test_row_batch(
                world.server.storage_mut(),
                world.row_id,
                &branch,
                row,
            )
            .expect("a refused version is held");
            batch_id
        })
        .collect();
    for batch_id in &refused_versions {
        let held = world
            .server
            .storage()
            .load_history_row_batch(TABLE, &branch, world.row_id, *batch_id)
            .expect("history readable");
        assert!(
            held.is_some_and(|row| !row.state.is_visible()),
            "control: the refused version must be held and not visible"
        );
    }
    assert_eq!(
        world.served_head().batch_id(),
        head.batch_id(),
        "control: refused versions must not change the served head"
    );

    let refused = world.alices_write_over(refused_versions, false);
    assert!(
        refused.is_empty(),
        "alice's update over two refused versions of her own row was refused: {refused:?}"
    );
}

/// A delete sent in the same delivery as the insert it retires: its one parent is in that
/// delivery, not yet in storage when the policy inputs are read. The single-parent read
/// finds nothing, and a delete has no refill — so this must not end in a persisted
/// refusal of a delete whose parent lands a moment later.
#[test]
#[ignore = "separate defect, pre-existing: red on linsa-v27 (+ JWKS fix ef89f266b2) with the \
            same reason, 'Delete denied on table documents - missing row content', measured \
            2026-10-18. Code-verified: the single-parent read answers Absent and a Delete with \
            no old content is persisted Rejected. NOT verified in code: that the parent is \
            not yet in storage when the inputs are read (inferred from the comment on \
            a_parent_and_child_in_one_delivery_both_apply). Not on the presence symptom path"]
fn a_delete_sent_with_the_insert_it_retires_is_not_refused() {
    let mut world = World::new(sqlite);
    let ((row_id, _), _) = world
        .client
        .insert(
            TABLE,
            HashMap::from([
                ("owner_id".to_string(), Value::Text("alice".to_string())),
                ("title".to_string(), Value::Text("short-lived".to_string())),
            ]),
            Some(&WriteContext::from_session(World::<SqliteStorage>::alice())),
        )
        .expect("the owner's insert satisfies her own policy");
    world
        .client
        .delete(
            row_id,
            Some(&WriteContext::from_session(World::<SqliteStorage>::alice())),
        )
        .expect("the owner's delete applies locally");
    world.row_id = row_id;
    let frames = world.push_client_frames();
    assert!(
        frames
            .iter()
            .any(|(frame, _)| frame.is_deleted && frame.parents.len() == 1),
        "control: the delete must travel as its own single-parent tombstone, in the same \
         delivery as the insert: {:?}",
        frames
            .iter()
            .map(|(frame, _)| (frame.is_deleted, frame.parents.len()))
            .collect::<Vec<_>>()
    );
    let refused: Vec<_> = frames
        .iter()
        .filter_map(|(_, refusal)| refusal.clone())
        .collect();
    assert!(
        refused.is_empty(),
        "a delete sent with the insert it retires was refused: {refused:?}"
    );
    let served = world
        .server
        .storage()
        .load_visible_region_row(TABLE, &world.branch(), row_id)
        .expect("visible read should succeed");
    assert!(
        served.as_ref().is_none_or(|row| row.is_deleted),
        "the delete must retire the row on the server, not leave it undecided: {served:?}"
    );
}

/// The first write after a deployment, over a row grown long on the previous generation,
/// must not read that row's history. This is the path every presence beat takes after a
/// deployment (the local stack's `users` row had 40 802 versions), and the shape of the
/// 2026-08-10 incident — a 2541-version row read whole on every write — that
/// `visible_ancestor_cost.rs` gated before the sieve it covered went with the fallback it
/// served. The inbox's probe for history on another branch and the query layer's reader
/// both answer from point reads now.
///
/// MemoryStorage counts whole-history reads and, like RocksDB, answers the other-branch
/// probe with a seek. SQLite has no such probe override: its default scans the row, once
/// per deferred write (recorded, not gated here).
#[test]
fn the_first_write_after_a_deployment_does_not_read_the_rows_history() {
    let mut world = World::new(memory);
    for beat in 0..200 {
        let refused = world.retitle(&format!("beat {beat}"));
        assert!(
            refused.is_empty(),
            "control: beat {beat} was refused ({refused:?})"
        );
    }
    let versions = world
        .server
        .storage()
        .scan_history_row_batches(TABLE, world.row_id)
        .expect("history scan should succeed")
        .len();
    assert!(
        versions > 200,
        "control: the row must be long ({versions} versions)"
    );
    world.deploy(v2());

    world.server.storage().reset_history_scans();
    let refused = world.retitle("after the deployment");
    let reads = world.server.storage().history_scans();
    assert!(
        refused.is_empty(),
        "control: the first write after the deployment was refused ({refused:?})"
    );
    assert_eq!(
        reads, 0,
        "the first write after a deployment read the {versions}-version row's whole history \
         {reads} time(s); every presence beat after a deployment takes this path"
    );
}

// ============================================================================
// Differential oracle
// ============================================================================

/// SplitMix64 — deterministic, dependency-free, reproducible from the seed alone.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// One version of the row the server holds visible, as the model reads it: its owner is
/// decoded in its own branch's generation, which is the shape it was written in.
#[derive(Debug, Clone)]
struct ModelVersion {
    branch: String,
    parents: Vec<BatchId>,
    owner: String,
    updated_at: u64,
    batch_id: BatchId,
    deleted: bool,
}

/// The intended semantics, independent of how the engine derives the old row: a write
/// is judged against the version it was made over.
///
/// - Naming one parent, that parent.
/// - Naming none — the first write on a generation's branch, a copy-on-write over the
///   row's older history — the row readers serve across the OTHER generations' branches:
///   one head per branch, a tombstone first, then the newest `(updated_at, batch_id)`.
///   With no such row it is an insert, and inserts are the backend's.
///
/// The owner may update and delete what she owns; nobody else's row is hers to touch.
struct Model {
    versions: HashMap<BatchId, ModelVersion>,
}

/// The owner a version names, decoded in its own branch's generation. An `Err` is a
/// control failure, reported with the seed like a verdict mismatch.
fn owner_of(generations: &[(Schema, String)], branch: &str, data: &[u8]) -> Result<String, String> {
    let (schema, _) = generations
        .iter()
        .find(|(_, candidate)| candidate == branch)
        .ok_or_else(|| format!("control: a version is on {branch}, no generation's branch"))?;
    let values = crate::row_format::decode_row(&schema[&TableName::new(TABLE)].columns, data)
        .map_err(|error| {
            format!("control: a version on {branch} does not decode in its generation: {error:?}")
        })?;
    match values.first() {
        Some(Value::Text(owner)) => Ok(owner.clone()),
        other => Err(format!(
            "control: owner_id on {branch} is not text: {other:?}"
        )),
    }
}

/// What the model takes for granted, checked against the server before every verdict:
///
/// - every branch holding a visible version is one the server's readers serve (the model
///   counts every branch holding a version; the engine reads `all_branch_names()`, equal
///   only while every generation activates in the context);
/// - every version's bytes are in its branch's generation, which is the shape the model
///   decodes them in (writes without metadata are placed by the locator, and backend
///   writes on older branches move it).
fn model_controls(world: &World<SqliteStorage>, generations: &[(Schema, String)]) -> Vec<String> {
    let readers: Vec<String> = world
        .server
        .schema_manager()
        .context()
        .all_branch_names()
        .into_iter()
        .map(|branch| branch.as_str().to_string())
        .collect();
    let mut failures = Vec::new();
    let versions = world
        .server
        .storage()
        .scan_history_row_batches(TABLE, world.row_id)
        .expect("history scan should succeed");
    for row in versions.iter().filter(|row| row.state.is_visible()) {
        let branch = row.branch.as_str();
        // The copy `copy_on_another_user_branch` placed beside the readers, by design.
        if world.beside_readers.iter().any(|beside| beside == branch) {
            continue;
        }
        if !readers.iter().any(|reader| reader == branch) {
            failures.push(format!(
                "control: {branch} holds a visible version but is not a reader branch ({readers:?})"
            ));
        }
        let Some((schema, _)) = generations
            .iter()
            .find(|(_, candidate)| candidate == branch)
        else {
            continue;
        };
        let family = world
            .server
            .storage()
            .load_history_row_batch_with_family(TABLE, branch, world.row_id, row.batch_id())
            .expect("history readable")
            .and_then(|(_, family)| family);
        if family != Some(SchemaHash::compute(schema)) {
            failures.push(format!(
                "control: a version on {branch} is held in family {family:?}, not its \
                 generation's"
            ));
        }
    }
    failures
}

impl Model {
    fn read(
        world: &World<SqliteStorage>,
        generations: &[(Schema, String)],
    ) -> Result<Self, String> {
        let versions = world
            .server
            .storage()
            .scan_history_row_batches(TABLE, world.row_id)
            .expect("history scan should succeed")
            .into_iter()
            .filter(|row| row.state.is_visible())
            // Readers serve generation branches only; a copy beside them is not a version.
            .filter(|row| {
                generations
                    .iter()
                    .any(|(_, branch)| branch == row.branch.as_str())
            })
            .map(|row| {
                let owner = if row.is_deleted {
                    String::new()
                } else {
                    owner_of(generations, row.branch.as_str(), &row.data)?
                };
                Ok((
                    row.batch_id(),
                    ModelVersion {
                        branch: row.branch.as_str().to_string(),
                        parents: row.parents.to_vec(),
                        owner,
                        updated_at: row.updated_at,
                        batch_id: row.batch_id(),
                        deleted: row.is_deleted,
                    },
                ))
            })
            .collect::<Result<_, String>>()?;
        Ok(Self { versions })
    }

    /// The branch's head: the one version no other version on the branch names.
    fn head_on(&self, branch: &str) -> Option<&ModelVersion> {
        let on_branch: Vec<_> = self
            .versions
            .values()
            .filter(|version| version.branch == branch)
            .collect();
        let heads: Vec<_> = on_branch
            .iter()
            .copied()
            .filter(|candidate| {
                !on_branch
                    .iter()
                    .any(|other| other.parents.contains(&candidate.batch_id))
            })
            .collect();
        assert!(
            heads.len() <= 1,
            "model: the driver forked {branch} ({} heads); it only ever extends a branch's head",
            heads.len()
        );
        heads.first().copied()
    }

    /// The readers' rule over the heads of every branch holding a version, except
    /// `excluded`: a tombstone first, then the newest `(updated_at, batch_id)`.
    fn readers_winner(&self, excluded: Option<&str>) -> Option<&ModelVersion> {
        let mut branches: Vec<&str> = self
            .versions
            .values()
            .map(|version| version.branch.as_str())
            .filter(|other| Some(*other) != excluded)
            .collect();
        branches.sort_unstable();
        branches.dedup();
        branches
            .into_iter()
            .filter_map(|other| self.head_on(other))
            .max_by_key(|head| (head.deleted, head.updated_at, head.batch_id))
    }

    /// The old row of a write the server defers: a parentless one must find the row
    /// OUTSIDE its branch (else it is an insert), then competes with its branch's head
    /// too; one naming parents it does not serve is judged against every branch's heads.
    fn deferred_old_row(&self, branch: &str, parentless: bool) -> Option<&ModelVersion> {
        if parentless {
            self.readers_winner(Some(branch))?;
        }
        self.readers_winner(None)
    }

    /// `Some(accepted)` where the semantics decide; `None` where the driver does not
    /// claim to know (a tombstone the same delete races, an unknown parent). A write that
    /// is not a delete must also leave the row hers: `new_owner` is what it names.
    fn verdict(&self, frame: &StoredRowBatch, same_push: usize, new_owner: &str) -> Option<bool> {
        let old_row_allows = self.old_row_allows(frame, same_push)?;
        Some(old_row_allows && (frame.is_deleted || new_owner == "alice"))
    }

    fn old_row_allows(&self, frame: &StoredRowBatch, same_push: usize) -> Option<bool> {
        match frame.parents.as_slice() {
            [parent] => {
                let old = self.versions.get(parent)?;
                Some(old.owner == "alice" && !old.deleted)
            }
            [] if frame.is_deleted && same_push > 1 => None,
            [] => match self.deferred_old_row(frame.branch.as_str(), true) {
                None => Some(false),
                Some(winner) if winner.deleted => None,
                Some(winner) => Some(winner.owner == "alice"),
            },
            _ => None,
        }
    }
}

impl World<SqliteStorage> {
    /// [`Self::push_client_writes`], keeping the frames sent and each one's refusal.
    fn push_client_frames(&mut self) -> Vec<(StoredRowBatch, Option<String>)> {
        self.client.batched_tick();
        let mut frames = Vec::new();
        for entry in self.client.sync_sender().take() {
            if entry.destination == Destination::Server(self.server_id) {
                if let SyncPayload::RowBatchCreated { row, .. } = &entry.payload
                    && row.row_id == self.row_id
                {
                    frames.push(row.clone());
                }
                self.server.park_sync_message(InboxEntry {
                    source: Source::Client(self.client_id),
                    payload: entry.payload,
                });
            }
        }
        self.server.batched_tick();
        self.server.immediate_tick();
        self.server.batched_tick();
        let refused: HashMap<BatchId, String> = self
            .server
            .sync_sender()
            .take()
            .into_iter()
            .filter_map(|entry| match entry.payload {
                SyncPayload::BatchFate {
                    fate:
                        BatchFate::Rejected {
                            batch_id, reason, ..
                        },
                } => Some((batch_id, reason)),
                _ => None,
            })
            .collect();
        frames
            .into_iter()
            .map(|frame| {
                let refusal = refused.get(&frame.batch_id()).cloned();
                (frame, refusal)
            })
            .collect()
    }
}

/// A generation after `previous`: a new column (a new shape) or a new table (the shape
/// kept, a new branch all the same, as `9778 → 57c2`).
fn next_generation(previous: &Schema, ordinal: usize, new_shape: bool) -> Schema {
    let mut builder = SchemaBuilder::new();
    for (name, table) in previous.iter() {
        let mut table_builder = TableSchema::builder(name.as_str());
        for column in table.columns.columns.iter() {
            table_builder = table_builder.column(column.name.as_str(), column.column_type.clone());
        }
        if new_shape && name.as_str() == TABLE {
            table_builder = table_builder.column(&format!("note_{ordinal}"), ColumnType::Text);
        }
        builder = builder.table(table_builder);
    }
    if !new_shape {
        builder = builder.table(
            TableSchema::builder(&format!("extra_{ordinal}")).column("state", ColumnType::Text),
        );
    }
    builder.build()
}

/// What one differential run actually exercised, so a green run cannot be a vacuous one.
#[derive(Debug, Default)]
struct Coverage {
    judged_accept: usize,
    judged_refuse: usize,
    unknown: usize,
    parentless_judged: usize,
    parented_judged: usize,
    deletes_judged: usize,
    absent_parent_deletes_judged: usize,
    absent_parent_deletes_over_own_head_only: usize,
    absent_parent_parks_over_own_head_only: usize,
    deploys: usize,
    backend_beats: usize,
    runs_past_two_deployments: usize,
}

fn run_migrated_write_differential(
    seed: u64,
    steps: usize,
    coverage: &mut Coverage,
) -> Vec<String> {
    let mut rng = Rng(seed);
    let mut world = World::new(sqlite);
    // Half the runs keep a copy of the row on a branch no reader of `main` serves: history
    // the inbox counts as "another branch" (it defers), which the old-row read must not.
    // Only there does a write that names parents differ between reading every branch and
    // reading outside its own first.
    let beside_readers = rng.below(2) == 0;
    if beside_readers {
        world.copy_on_another_user_branch();
    }
    let mut generations: Vec<(Schema, String)> = vec![(v1(), world.branch())];
    let mut log = Vec::new();
    let mut mismatches = Vec::new();

    for step in 0..steps {
        let older: Vec<(Schema, String)> = generations[..generations.len() - 1].to_vec();
        match rng.below(100) {
            0..=39 => {
                let title = format!("t{step}");
                world
                    .client
                    .update(
                        world.row_id,
                        vec![("title".to_string(), Value::Text(title.clone()))],
                        Some(&WriteContext::from_session(World::<SqliteStorage>::alice())),
                    )
                    .expect("the owner's update applies locally");
                log.push(format!("retitle@g{}", generations.len() - 1));
                mismatches.extend(judge(
                    &mut world,
                    &generations,
                    &log,
                    seed,
                    step,
                    "retitle",
                    coverage,
                ));
            }
            40..=59 if generations.len() < 6 => {
                let new_shape = rng.below(2) == 0;
                let (previous, _) = generations.last().expect("a generation").clone();
                let next = next_generation(&previous, generations.len(), new_shape);
                world.deploy(next.clone());
                generations.push((next, world.branch()));
                coverage.deploys += 1;
                log.push(format!(
                    "deploy g{} ({})",
                    generations.len() - 1,
                    if new_shape { "column" } else { "table" }
                ));
            }
            60..=89 if !older.is_empty() => {
                let (schema, branch) = &older[rng.below(older.len())];
                let model = match Model::read(&world, &generations) {
                    Ok(model) => model,
                    Err(failure) => {
                        mismatches.push(format!(
                            "seed {seed:#x} step {step}: {failure}\n  ops: {}",
                            log.join(" → ")
                        ));
                        continue;
                    }
                };
                let Some(head) = model.head_on(branch).cloned() else {
                    continue;
                };
                let newest = model
                    .versions
                    .values()
                    .map(|version| version.updated_at)
                    .max()
                    .expect("the row has versions");
                let newer = rng.below(2) == 0 || head.updated_at + 1 >= newest;
                let updated_at = if newer {
                    newest + 1_000
                } else {
                    head.updated_at + 1
                };
                let owner = if rng.below(2) == 0 {
                    "alice"
                } else {
                    "mallory"
                };
                let columns = &schema[&TableName::new(TABLE)].columns;
                let mut values = vec![
                    Value::Text(owner.to_string()),
                    Value::Text(format!("backend {step}")),
                ];
                values.resize(columns.columns.len(), Value::Text(String::new()));
                let parent = world
                    .server
                    .storage()
                    .load_history_row_batch(TABLE, branch, world.row_id, head.batch_id)
                    .expect("history readable")
                    .expect("the head is held");
                let row = StoredRowBatch::new(
                    world.row_id,
                    branch.as_str(),
                    vec![head.batch_id],
                    encode_row(columns, &values).expect("row encodes"),
                    RowProvenance {
                        created_by: parent.created_by.clone(),
                        created_at: parent.created_at,
                        updated_by: "backend".into(),
                        updated_at,
                    },
                    HashMap::new(),
                    RowState::VisibleDirect,
                    None,
                );
                let refused = world.push_raw("backend", ClientRole::Backend, schema, row);
                coverage.backend_beats += 1;
                log.push(format!(
                    "backend {owner} {} on {}",
                    if newer { "newest" } else { "late" },
                    &branch[..16.min(branch.len())]
                ));
                if !refused.is_empty() {
                    mismatches.push(format!(
                        "seed {seed:#x} step {step}: control — the backend's write was refused: \
                         {refused:?}\n  ops: {}",
                        log.join(" → ")
                    ));
                }
            }
            90..=99 => {
                let (schema, branch) = generations[rng.below(generations.len())].clone();
                let context = |what: &str| {
                    format!("seed {seed:#x} step {step} absent-parent delete on {branch}: {what}")
                };
                let failures = model_controls(&world, &generations);
                if !failures.is_empty() {
                    mismatches.extend(failures.iter().map(|failure| context(failure)));
                    continue;
                }
                let model = match Model::read(&world, &generations) {
                    Ok(model) => model,
                    Err(failure) => {
                        mismatches.push(context(&failure));
                        continue;
                    }
                };
                // Deferred only while the row has history on another branch; without it a
                // tombstone with absent parents is refused for missing content, as on v27
                // (recorded separately, not this model's claim).
                let elsewhere_for_readers = model
                    .versions
                    .values()
                    .any(|version| version.branch != branch);
                let elsewhere = elsewhere_for_readers || beside_readers;
                let expected = match model.deferred_old_row(&branch, false) {
                    _ if !elsewhere => None,
                    Some(winner) if !winner.deleted => Some(winner.owner == "alice"),
                    _ => None,
                };
                let columns = &schema[&TableName::new(TABLE)].columns;
                let mut values = vec![
                    Value::Text("alice".to_string()),
                    Value::Text(format!("delete {step}")),
                ];
                values.resize(columns.columns.len(), Value::Text(String::new()));
                let mut tombstone = StoredRowBatch::new(
                    world.row_id,
                    branch.as_str(),
                    vec![BatchId::new(), BatchId::new()],
                    encode_row(columns, &values).expect("row encodes"),
                    RowProvenance {
                        created_by: "alice".into(),
                        created_at: 1,
                        updated_by: "alice".into(),
                        updated_at: 2,
                    },
                    HashMap::new(),
                    RowState::VisibleDirect,
                    None,
                );
                tombstone.is_deleted = true;
                let parents = tombstone.parents.to_vec();
                let (refused, asked) =
                    world.push_raw_answers("alice", ClientRole::User, &schema, tombstone);
                log.push(format!(
                    "absent-parent delete on {}",
                    &branch[..16.min(branch.len())]
                ));
                let Some(expected) = expected else {
                    coverage.unknown += 1;
                    continue;
                };
                coverage.absent_parent_deletes_judged += 1;
                if !elsewhere_for_readers {
                    coverage.absent_parent_deletes_over_own_head_only += 1;
                    if expected {
                        coverage.absent_parent_parks_over_own_head_only += 1;
                    }
                }
                if expected {
                    coverage.judged_accept += 1;
                } else {
                    coverage.judged_refuse += 1;
                }
                // Parked means asked for a declared parent; refused means BY POLICY, judged
                // against the head — a refusal for missing content is the wrong refusal.
                let parked =
                    refused.is_empty() && asked.iter().any(|batch_id| parents.contains(batch_id));
                let refused_by_policy = refused
                    .iter()
                    .any(|reason| reason.contains("Delete denied by policy"));
                if (expected && !parked) || (!expected && !refused_by_policy) {
                    mismatches.push(format!(
                        "{} — model says {}, server {}\n  ops: {}",
                        context("verdict"),
                        if expected { "park" } else { "refuse by policy" },
                        if parked {
                            "parked".to_string()
                        } else if refused.is_empty() {
                            format!("neither refused nor asked for a parent (asked {asked:?})")
                        } else {
                            format!("refused: {refused:?}")
                        },
                        log.join(" → "),
                    ));
                }
            }
            _ => {}
        }
    }

    world
        .client
        .delete(
            world.row_id,
            Some(&WriteContext::from_session(World::<SqliteStorage>::alice())),
        )
        .expect("the owner's delete applies locally");
    log.push("delete".to_string());
    mismatches.extend(judge(
        &mut world,
        &generations,
        &log,
        seed,
        steps,
        "delete",
        coverage,
    ));
    if generations.len() > 3 {
        coverage.runs_past_two_deployments += 1;
    }
    mismatches
}

/// Push the owner's queued frames and compare each verdict with the model's, read from the
/// server's visible versions BEFORE the push.
fn judge(
    world: &mut World<SqliteStorage>,
    generations: &[(Schema, String)],
    log: &[String],
    seed: u64,
    step: usize,
    op: &str,
    coverage: &mut Coverage,
) -> Vec<String> {
    let context = |what: &str| {
        format!(
            "seed {seed:#x} step {step} {op}: {what}\n  ops: {}",
            log.join(" → ")
        )
    };
    let mut mismatches: Vec<String> = model_controls(world, generations)
        .iter()
        .map(|failure| context(failure))
        .collect();
    let model = match Model::read(world, generations) {
        Ok(model) => model,
        Err(failure) => {
            mismatches.push(context(&failure));
            world.push_client_frames();
            return mismatches;
        }
    };
    let frames = world.push_client_frames();
    let same_push = frames.len();
    for (frame, refusal) in frames {
        let new_owner = if frame.is_deleted {
            String::new()
        } else {
            match owner_of(generations, frame.branch.as_str(), &frame.data) {
                Ok(owner) => owner,
                Err(failure) => {
                    mismatches.push(context(&failure));
                    continue;
                }
            }
        };
        let Some(expected) = model.verdict(&frame, same_push, &new_owner) else {
            coverage.unknown += 1;
            continue;
        };
        if expected {
            coverage.judged_accept += 1;
        } else {
            coverage.judged_refuse += 1;
        }
        if frame.parents.is_empty() {
            coverage.parentless_judged += 1;
        } else {
            coverage.parented_judged += 1;
        }
        if frame.is_deleted {
            coverage.deletes_judged += 1;
        }
        if expected != refusal.is_none() {
            mismatches.push(format!(
                "seed {seed:#x} step {step} {op}: frame on {} with {} parent(s), deleted={} — \
                 model says {}, server {}\n  ops: {}",
                frame.branch,
                frame.parents.len(),
                frame.is_deleted,
                if expected { "accept" } else { "refuse" },
                refusal
                    .as_deref()
                    .map_or("accepted".to_string(), |reason| format!(
                        "refused: {reason}"
                    )),
                log.join(" → "),
            ));
        }
    }
    mismatches
}

/// Differential: across random deployments (new shapes and shape-keeping ones), the
/// owner's writes and the backend's late writes on older generations, the server's verdict
/// on every frame the owner's device sends must be the one the model's semantics give.
#[test]
fn migrated_writes_are_judged_against_the_version_they_were_made_over() {
    let seeds: Vec<u64> = match std::env::var("JAZZ_MIGRATED_WRITE_SEED") {
        Ok(seed) => vec![u64::from_str_radix(seed.trim_start_matches("0x"), 16).expect("hex seed")],
        Err(_) => (0..24u64).map(|n| 0x2764_D1FF_C0DE_0000 + n).collect(),
    };
    let mut mismatches = Vec::new();
    let mut coverage = Coverage::default();
    let several = seeds.len() > 1;
    for seed in seeds {
        mismatches.extend(run_migrated_write_differential(seed, 28, &mut coverage));
    }
    eprintln!("migrated-write differential coverage: {coverage:?}");
    if several {
        assert!(
            coverage.judged_accept > 0
                && coverage.judged_refuse > 0
                && coverage.parentless_judged > 0
                && coverage.parented_judged > 0
                && coverage.deletes_judged > 0
                && coverage.absent_parent_deletes_judged > 0
                && coverage.absent_parent_deletes_over_own_head_only > 0
                && coverage.absent_parent_parks_over_own_head_only > 0
                && coverage.backend_beats > 0
                && coverage.runs_past_two_deployments > 0,
            "the differential exercised too little to mean anything: {coverage:?}"
        );
    }
    assert!(
        mismatches.is_empty(),
        "{} verdict(s) differ from the model:\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
}
