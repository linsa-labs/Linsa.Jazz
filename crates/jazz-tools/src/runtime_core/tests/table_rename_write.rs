//! An owner's writes to a row after its table is renamed.
//!
//! The row is created under the old name. A client on the new generation names the table
//! as that generation calls it; the server must judge and apply the write as the same
//! row, however the row's locator was stamped by the writes before it.

use super::*;

use crate::batch_fate::BatchFate;
use crate::schema_manager::{Lens, LensOp, LensTransform};
use crate::storage::SqliteStorage;

const APP: &str = "table-rename-write";

fn owner_policies() -> TablePolicies {
    let owner = || PolicyExpr::eq_session("owner_id", vec!["user_id".into()]);
    TablePolicies::new()
        .with_select(owner())
        .with_insert(owner())
        .with_update(Some(owner()), owner())
        .with_delete(owner())
}

fn table_builder(name: &str) -> crate::query_manager::types::TableSchemaBuilder {
    TableSchema::builder(name)
        .column("owner_id", ColumnType::Text)
        .column("title", ColumnType::Text)
}

fn schema_of(names: &[&str]) -> Schema {
    names
        .iter()
        .fold(SchemaBuilder::new(), |builder, name| {
            builder.table(table_builder(name))
        })
        .build()
}

/// `documents` before the rename.
fn v1() -> Schema {
    schema_of(&["documents"])
}

fn lens(from: &Schema, to: &Schema, ops: Vec<LensOp>) -> Lens {
    let mut transform = LensTransform::new();
    for op in ops {
        transform.push(op, false);
    }
    Lens::new(
        SchemaHash::compute(from),
        SchemaHash::compute(to),
        transform,
    )
}

fn rename(old_name: &str, new_name: &str) -> LensOp {
    LensOp::RenameTable {
        old_name: old_name.to_string(),
        new_name: new_name.to_string(),
    }
}

fn permissions_for(schema: &Schema) -> Schema {
    let mut head = schema.clone();
    for (_, table) in head.iter_mut() {
        table.policies = owner_policies();
    }
    head
}

type Core = RuntimeCore<SqliteStorage, NoopScheduler>;

fn sqlite() -> SqliteStorage {
    SqliteStorage::open(":memory:").expect("in-memory sqlite storage should open")
}

fn server_runtime(schema: Schema, storage: SqliteStorage) -> Core {
    let mut core = server_core(schema.clone(), storage);
    core.schema_manager_mut()
        .query_manager_mut()
        .set_authorization_schema(permissions_for(&schema));
    core.schema_manager_mut()
        .query_manager_mut()
        .require_authorization_schema();
    core.immediate_tick();
    core
}

fn runtime(schema: Schema, storage: SqliteStorage) -> Core {
    runtime_with(schema, storage, SyncManager::new())
}

/// A server at the tier a deployed one has with no upstream (`server::builder`): without
/// one it never stores a fate, and every batch stays in the window before it.
fn server_core(schema: Schema, storage: SqliteStorage) -> Core {
    runtime_with(
        schema,
        storage,
        SyncManager::new().with_durability_tier(crate::sync_manager::DurabilityTier::GlobalServer),
    )
}

fn runtime_with(schema: Schema, storage: SqliteStorage, sync_manager: SyncManager) -> Core {
    let app_id = AppId::from_name(APP);
    let mut schema_manager =
        SchemaManager::new(sync_manager, schema, app_id, "dev", "main").unwrap();
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

struct World {
    server: Core,
    client: Core,
    server_id: ServerId,
    client_id: ClientId,
    row_id: ObjectId,
    /// Every `Rejected` reason the server sent.
    rejected: Vec<String>,
}

impl World {
    fn alice() -> WriteContext {
        WriteContext::from_session(Session::new("alice"))
    }

    /// Alice's document, written and accepted under `v1`.
    fn new() -> Self {
        let mut server = server_runtime(v1(), sqlite());
        let mut client = runtime(v1(), sqlite());
        let server_id = ServerId::new();
        let client_id = ClientId::new();
        server.add_client(client_id, Some(Session::new("alice")));
        client.add_server(server_id);
        let ((row_id, _), _) = client
            .insert(
                "documents",
                HashMap::from([
                    ("owner_id".to_string(), Value::Text("alice".to_string())),
                    ("title".to_string(), Value::Text("draft".to_string())),
                ]),
                Some(&Self::alice()),
            )
            .expect("the owner's insert applies locally");
        let mut world = Self {
            server,
            client,
            server_id,
            client_id,
            row_id,
            rejected: Vec::new(),
        };
        world.sync();
        assert!(
            world.rejected.is_empty(),
            "control: the seed insert was rejected"
        );
        world
    }

    /// Both directions until nothing moves; the server's refusals are recorded.
    fn sync(&mut self) {
        for _ in 0..12 {
            let mut moved = false;
            self.client.batched_tick();
            for entry in self.client.sync_sender().take() {
                if entry.destination == Destination::Server(self.server_id) {
                    moved = true;
                    self.server.park_sync_message(InboxEntry {
                        source: Source::Client(self.client_id),
                        payload: entry.payload,
                    });
                }
            }
            self.server.batched_tick();
            self.server.immediate_tick();
            self.server.batched_tick();
            for entry in self.server.sync_sender().take() {
                if entry.destination != Destination::Client(self.client_id) {
                    continue;
                }
                if let SyncPayload::BatchFate {
                    fate: BatchFate::Rejected { reason, .. },
                } = &entry.payload
                {
                    self.rejected.push(reason.clone());
                }
                moved = true;
                self.client.park_sync_message(InboxEntry {
                    source: Source::Server(self.server_id),
                    payload: entry.payload,
                });
            }
            self.client.batched_tick();
            self.client.immediate_tick();
            if !moved {
                break;
            }
        }
    }

    /// A deployment: both ends rebuilt on `next` from their own stores, `lens` published
    /// server-side, and the reconnect handshake in both directions.
    fn deploy(&mut self, next: Schema, lens: Lens) {
        let server_storage =
            std::mem::replace(&mut self.server, runtime(v1(), sqlite())).into_storage();
        let client_storage =
            std::mem::replace(&mut self.client, runtime(v1(), sqlite())).into_storage();
        self.server = server_runtime(next.clone(), server_storage);
        self.server.publish_lens(&lens).expect("lens publishes");
        self.client = runtime(next, client_storage);
        self.client
            .publish_lens(&lens)
            .expect("lens publishes on the client");
        self.server
            .add_client(self.client_id, Some(Session::new("alice")));
        self.client.add_server(self.server_id);
        self.server.immediate_tick();
        self.client.immediate_tick();
        self.sync();
    }

    /// The title the server serves for alice's row in `table`.
    fn served_title(&mut self, table: &str) -> Option<Value> {
        let sub = self
            .server
            .schema_manager_mut()
            .query_manager_mut()
            .subscribe_with_session(Query::new(table), None, None)
            .unwrap();
        self.server.immediate_tick();
        let title = self
            .server
            .schema_manager()
            .query_manager()
            .get_subscription_results(sub)
            .into_iter()
            .find(|(id, _)| *id == self.row_id)
            .map(|(_, values)| values[1].clone());
        self.server
            .schema_manager_mut()
            .query_manager_mut()
            .unsubscribe_with_sync(sub);
        title
    }
}

#[test]
fn the_owners_update_after_a_rename_is_accepted_and_served() {
    let mut world = World::new();
    let v2 = schema_of(&["papers"]);
    let lens = lens(&v1(), &v2, vec![rename("documents", "papers")]);
    world.deploy(v2, lens);

    world
        .client
        .update(
            world.row_id,
            vec![("title".to_string(), Value::Text("final".to_string()))],
            Some(&World::alice()),
        )
        .expect("the owner's update applies locally");
    world.sync();

    assert_eq!(
        world.rejected,
        Vec::<String>::new(),
        "the update is accepted"
    );
    assert_eq!(
        world.served_title("papers"),
        Some(Value::Text("final".to_string())),
        "the renamed table serves the update"
    );
}

#[test]
fn the_owners_updates_and_delete_after_a_rename_are_accepted() {
    let mut world = World::new();
    let v2 = schema_of(&["papers"]);
    let lens = lens(&v1(), &v2, vec![rename("documents", "papers")]);
    world.deploy(v2, lens);
    for title in ["second", "third"] {
        world
            .client
            .update(
                world.row_id,
                vec![("title".to_string(), Value::Text(title.to_string()))],
                Some(&World::alice()),
            )
            .expect("the owner's update applies locally");
        world.sync();
        assert!(world.rejected.is_empty(), "{title}: {:?}", world.rejected);
        assert_eq!(
            world.served_title("papers"),
            Some(Value::Text(title.to_string()))
        );
    }
    world
        .client
        .delete(world.row_id, Some(&World::alice()))
        .expect("the owner's delete applies locally");
    world.sync();
    assert!(world.rejected.is_empty(), "{:?}", world.rejected);
    assert_eq!(world.served_title("papers"), None, "the delete is served");
}
