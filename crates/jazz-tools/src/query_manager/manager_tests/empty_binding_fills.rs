//! Ways an empty include binding gets filled that `include_routing_liveness` does not
//! reach.
//!
//! That matrix moves children under a reverse include on `MemoryStorage`. An empty
//! binding is also what a forward include holds for a reference whose target has not
//! arrived, what keeps an outer row OUT of a result when the include is required, and
//! what a hard delete or a row arriving from a server lands on — and on SQLite the
//! question "does the correlation index hold anything" is answered through the read
//! memo, inside a read scope.
//!
//! A binding that is not probed again when its first row arrives raises nothing: the
//! subscriber keeps an empty array, or never sees the outer row at all. So every case
//! here writes ONE change, settles ONCE, and reads a mirror built from the emitted
//! deltas — on both storages, with empty bindings and with an instance per binding. The
//! runs with empty bindings also compile each spared instance and require it to be
//! empty (`check_empty_bindings_against_instances`).

use std::collections::HashMap;

use super::*;
use crate::query_manager::graph_nodes::include_routing::{
    check_empty_bindings_against_instances, empty_bindings_bound_on_this_thread,
    force_include_empty_probe, force_include_routing,
};
use crate::query_manager::manager::QueryUpdate;
use crate::query_manager::precise_dirty::force_precise_dirty;
use crate::query_manager::types::TableSchemaBuilder;
use crate::storage::SqliteStorage;
use crate::test_support::persist_test_schema;

/// Outer rows per case. Above `MIN_ROUTABLE_INSTANCES`: at or below it a node
/// broadcasts, and a fill that only works by broadcast would pass here unnoticed.
const OUTER_ROWS: usize = 5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Backend {
    Memory,
    Sqlite,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Empty {
    Bindings,
    Instances,
}

fn fills_schema() -> Schema {
    let mut schema = Schema::new();
    let (name, table) = TableSchemaBuilder::new("chats")
        .column("name", ColumnType::Text)
        .build_named();
    schema.insert(name, table);
    let (name, table) = TableSchemaBuilder::new("members")
        .column("name", ColumnType::Text)
        .fk_column("chat", "chats")
        .build_named();
    schema.insert(name, table);
    let (name, table) = TableSchemaBuilder::new("notes")
        .column("title", ColumnType::Text)
        .fk_column("chat", "chats")
        .build_named();
    schema.insert(name, table);
    schema
}

/// One engine, one subscription, and the mirror its deltas build.
struct Live {
    qm: QueryManager,
    storage: Box<dyn Storage>,
    sub_id: Option<crate::query_manager::QuerySubscriptionId>,
    rows: HashMap<ObjectId, Vec<Value>>,
    backend: Backend,
    empty: Empty,
    bound_at_start: u64,
}

impl Live {
    fn open(backend: Backend, empty: Empty) -> Self {
        let mut qm = QueryManager::new(SyncManager::new());
        qm.set_current_schema(fills_schema(), "dev", "main");
        let schema = qm.schema_context().current_schema.clone();
        let mut storage: Box<dyn Storage> = match backend {
            Backend::Memory => Box::new(MemoryStorage::new()),
            Backend::Sqlite => {
                Box::new(SqliteStorage::open(":memory:").expect("in-memory sqlite storage"))
            }
        };
        persist_test_schema(&mut storage, &schema);
        Self {
            qm,
            storage,
            sub_id: None,
            rows: HashMap::new(),
            backend,
            empty,
            bound_at_start: empty_bindings_bound_on_this_thread(),
        }
    }

    fn insert(&mut self, table: &str, values: &[Value]) -> ObjectId {
        self.qm
            .insert(&mut self.storage, table, values)
            .expect("insert")
            .row_id
    }

    fn insert_with_id(&mut self, table: &str, id: ObjectId, values: &[Value]) {
        let branch = get_branch(&self.qm);
        let schema = self.qm.schema_context().current_schema.clone();
        self.qm
            .insert_on_branch_with_schema_and_write_context_and_id(
                &mut self.storage,
                table,
                &branch,
                values,
                Some(id),
                &schema,
                None,
                true,
            )
            .expect("insert with a chosen id");
    }

    fn subscribe(&mut self, query: Query) {
        self.sub_id = Some(self.qm.subscribe(query).expect("subscribe"));
        self.settle_once();
    }

    /// One settle, the way a runtime tick runs it: inside a read scope, with the
    /// writes that preceded it still in the storage's open write transaction.
    fn settle_once(&mut self) {
        self.storage.begin_read_scope();
        self.qm.process(&mut self.storage);
        self.storage.end_read_scope();
        let sub_id = self.sub_id.expect("a subscription");
        let updates: Vec<QueryUpdate> = self
            .qm
            .take_updates()
            .into_iter()
            .filter(|update| update.subscription_id == sub_id)
            .collect();
        for update in updates {
            for row in &update.delta.removed {
                self.rows.remove(&row.id);
            }
            for row in update
                .delta
                .updated
                .iter()
                .map(|(_, new)| new)
                .chain(&update.delta.added)
            {
                let values = decode_row(&update.descriptor, &row.data).expect("decode a row");
                self.rows.insert(row.id, values);
            }
        }
    }

    /// The subscriber's include array for `outer`: `None` when the subscriber does
    /// not hold the outer row at all.
    fn included(&self, outer: ObjectId) -> Option<usize> {
        self.rows.get(&outer).map(|values| {
            values
                .last()
                .and_then(Value::as_array)
                .expect("the include is the last column")
                .len()
        })
    }

    fn empty_bindings_bound(&self) -> u64 {
        empty_bindings_bound_on_this_thread() - self.bound_at_start
    }

    /// The case says nothing about empty bindings unless the engine made some — and
    /// the kill switch has to make none.
    fn assert_bindings_were_made(&self, at_least: u64) {
        let bound = self.empty_bindings_bound();
        match self.empty {
            Empty::Bindings => assert!(
                bound >= at_least,
                "{:?}: {bound} empty bindings, expected at least {at_least} — the include \
                 holds instances, so this run says nothing about empty bindings",
                self.backend
            ),
            Empty::Instances => {
                assert_eq!(bound, 0, "the kill switch still made empty bindings")
            }
        }
    }

    fn what(&self) -> String {
        format!("{:?}/{:?}", self.backend, self.empty)
    }
}

/// Run a case on both storages, with empty bindings and with an instance per binding.
fn each_way(case: impl Fn(&mut Live)) {
    for backend in [Backend::Memory, Backend::Sqlite] {
        for empty in [Empty::Bindings, Empty::Instances] {
            let _precise = force_precise_dirty(true);
            let _routing = force_include_routing(true);
            let _empty = force_include_empty_probe(matches!(empty, Empty::Bindings));
            let _parity = check_empty_bindings_against_instances();
            let mut live = Live::open(backend, empty);
            case(&mut live);
        }
    }
}

fn forward_include(qm: &QueryManager) -> Query {
    qm.query("members")
        .with_array("chat", |sub| {
            sub.from("chats").correlate("id", "members.chat")
        })
        .build()
}

fn reverse_include(qm: &QueryManager, required: bool) -> Query {
    qm.query("chats")
        .with_array("notes", |sub| {
            let sub = sub.from("notes").correlate("chat", "chats.id");
            if required { sub.require_result() } else { sub }
        })
        .build()
}

fn text(value: &str) -> Value {
    Value::Text(value.into())
}

/// FORWARD include over a reference whose target has not arrived — a reply to a
/// message outside the loaded window, a member of a chat still on its way. The binding
/// is the missing row's id, and the only thing that can fill it is that row's insert.
#[test]
fn a_dangling_reference_is_filled_when_its_target_arrives() {
    each_way(|live| {
        let ghosts: Vec<ObjectId> = (0..OUTER_ROWS).map(|_| ObjectId::new()).collect();
        let members: Vec<ObjectId> = ghosts
            .iter()
            .enumerate()
            .map(|(index, ghost)| {
                live.insert(
                    "members",
                    &[text(&format!("member-{index}")), Value::Uuid(*ghost)],
                )
            })
            .collect();
        live.subscribe(forward_include(&live.qm));
        for member in &members {
            assert_eq!(live.included(*member), Some(0), "{}", live.what());
        }
        live.assert_bindings_were_made((OUTER_ROWS - 1) as u64);

        live.insert_with_id("chats", ghosts[2], &[text("arrived")]);
        live.settle_once();
        for (index, member) in members.iter().enumerate() {
            assert_eq!(
                live.included(*member),
                Some(usize::from(index == 2)),
                "{}: member {index} after its chat arrived",
                live.what()
            );
        }

        // And out again, each way a row can go, with the binding filled in between.
        live.qm
            .delete(&mut live.storage, ghosts[2])
            .expect("soft delete");
        live.settle_once();
        assert_eq!(live.included(members[2]), Some(0), "{}", live.what());

        live.qm
            .restore(&mut live.storage, ghosts[2], &[text("restored")])
            .expect("restore");
        live.settle_once();
        assert_eq!(live.included(members[2]), Some(1), "{}", live.what());

        live.qm
            .hard_delete(&mut live.storage, ghosts[2])
            .expect("hard delete");
        live.settle_once();
        assert_eq!(live.included(members[2]), Some(0), "{}", live.what());
    });
}

/// The outer row's reference moves from one missing target to another. The old
/// binding must stop answering for this row and the new one start: a target arriving
/// for the OLD reference changes nothing, one arriving for the new reference fills it.
#[test]
fn an_outer_row_that_moves_between_empty_bindings_follows_its_new_reference() {
    each_way(|live| {
        let first: Vec<ObjectId> = (0..OUTER_ROWS).map(|_| ObjectId::new()).collect();
        let members: Vec<ObjectId> = first
            .iter()
            .enumerate()
            .map(|(index, ghost)| {
                live.insert(
                    "members",
                    &[text(&format!("member-{index}")), Value::Uuid(*ghost)],
                )
            })
            .collect();
        live.subscribe(forward_include(&live.qm));
        live.assert_bindings_were_made((OUTER_ROWS - 1) as u64);

        let second = ObjectId::new();
        live.qm
            .update(
                &mut live.storage,
                members[1],
                &[text("member-1"), Value::Uuid(second)],
            )
            .expect("re-point the member");
        live.settle_once();
        assert_eq!(live.included(members[1]), Some(0), "{}", live.what());

        live.insert_with_id("chats", first[1], &[text("the old reference")]);
        live.settle_once();
        assert_eq!(
            live.included(members[1]),
            Some(0),
            "{}: the member no longer points at this chat",
            live.what()
        );

        live.insert_with_id("chats", second, &[text("the new reference")]);
        live.settle_once();
        assert_eq!(
            live.included(members[1]),
            Some(1),
            "{}: the chat the member now points at arrived",
            live.what()
        );
        for (index, member) in members.iter().enumerate() {
            if index != 1 {
                assert_eq!(live.included(*member), Some(0), "{}", live.what());
            }
        }
    });
}

/// Every outer row moves to a new missing target, so the one instance the node compiled
/// to learn its probe goes with the binding it stood for: the node is left holding empty
/// bindings and nothing else. A target arriving then has only bindings to reach, and a
/// node that took "no instance" for "nothing to tell" would drop the news.
#[test]
fn a_node_left_with_only_empty_bindings_is_still_filled() {
    each_way(|live| {
        let members: Vec<ObjectId> = (0..OUTER_ROWS)
            .map(|index| {
                live.insert(
                    "members",
                    &[
                        text(&format!("member-{index}")),
                        Value::Uuid(ObjectId::new()),
                    ],
                )
            })
            .collect();
        live.subscribe(forward_include(&live.qm));
        live.assert_bindings_were_made((OUTER_ROWS - 1) as u64);

        let targets: Vec<ObjectId> = members
            .iter()
            .enumerate()
            .map(|(index, member)| {
                let target = ObjectId::new();
                live.qm
                    .update(
                        &mut live.storage,
                        *member,
                        &[text(&format!("member-{index}")), Value::Uuid(target)],
                    )
                    .expect("re-point the member");
                target
            })
            .collect();
        live.settle_once();
        for member in &members {
            assert_eq!(live.included(*member), Some(0), "{}", live.what());
        }
        // Every slot was bound again, the learning instance's included.
        live.assert_bindings_were_made((2 * OUTER_ROWS - 1) as u64);

        for (index, target) in targets.iter().enumerate() {
            live.insert_with_id("chats", *target, &[text("arrived")]);
            live.settle_once();
            for (other, member) in members.iter().enumerate() {
                assert_eq!(
                    live.included(*member),
                    Some(usize::from(other <= index)),
                    "{}: member {other} after the chat of member {index} arrived",
                    live.what()
                );
            }
        }
    });
}

/// A REQUIRED include: while its binding is empty the outer row is not in the result
/// at all, so the fill has to make the OUTER row appear — there is no array on the
/// subscriber's side to be stale, only a row that never shows.
#[test]
fn a_required_include_filled_from_an_empty_binding_brings_its_outer_row() {
    each_way(|live| {
        let chats: Vec<ObjectId> = (0..OUTER_ROWS)
            .map(|index| live.insert("chats", &[text(&format!("chat-{index}"))]))
            .collect();
        live.subscribe(reverse_include(&live.qm, true));
        assert!(live.rows.is_empty(), "{}: {:?}", live.what(), live.rows);
        live.assert_bindings_were_made((OUTER_ROWS - 1) as u64);

        let note = live.insert("notes", &[text("first"), Value::Uuid(chats[3])]);
        live.settle_once();
        assert_eq!(live.included(chats[3]), Some(1), "{}", live.what());
        assert_eq!(live.rows.len(), 1, "{}", live.what());

        live.qm.delete(&mut live.storage, note).expect("delete");
        live.settle_once();
        assert_eq!(live.included(chats[3]), None, "{}", live.what());

        // Two fills in ONE settle, under bindings emptied at different times.
        live.insert("notes", &[text("again"), Value::Uuid(chats[3])]);
        live.insert("notes", &[text("elsewhere"), Value::Uuid(chats[0])]);
        live.settle_once();
        assert_eq!(live.included(chats[3]), Some(1), "{}", live.what());
        assert_eq!(live.included(chats[0]), Some(1), "{}", live.what());
        assert_eq!(live.rows.len(), 2, "{}", live.what());
    });
}

/// A filled binding emptied by a HARD delete, a soft delete undone by a restore, and
/// every outer row filled in a single settle.
#[test]
fn a_binding_emptied_by_any_delete_fills_again() {
    each_way(|live| {
        let chats: Vec<ObjectId> = (0..OUTER_ROWS)
            .map(|index| live.insert("chats", &[text(&format!("chat-{index}"))]))
            .collect();
        live.subscribe(reverse_include(&live.qm, false));
        live.assert_bindings_were_made((OUTER_ROWS - 1) as u64);

        let note = live.insert("notes", &[text("first"), Value::Uuid(chats[1])]);
        live.settle_once();
        assert_eq!(live.included(chats[1]), Some(1), "{}", live.what());

        live.qm
            .hard_delete(&mut live.storage, note)
            .expect("hard delete");
        live.settle_once();
        assert_eq!(live.included(chats[1]), Some(0), "{}", live.what());

        let second = live.insert("notes", &[text("second"), Value::Uuid(chats[1])]);
        live.settle_once();
        assert_eq!(live.included(chats[1]), Some(1), "{}", live.what());

        live.qm.delete(&mut live.storage, second).expect("delete");
        live.settle_once();
        assert_eq!(live.included(chats[1]), Some(0), "{}", live.what());

        live.qm
            .restore(
                &mut live.storage,
                second,
                &[text("second"), Value::Uuid(chats[1])],
            )
            .expect("restore");
        live.settle_once();
        assert_eq!(live.included(chats[1]), Some(1), "{}", live.what());

        // Every binding at once: one batch of writes, one settle.
        for chat in &chats {
            live.insert("notes", &[text("everywhere"), Value::Uuid(*chat)]);
        }
        live.settle_once();
        for (index, chat) in chats.iter().enumerate() {
            assert_eq!(
                live.included(*chat),
                Some(if index == 1 { 2 } else { 1 }),
                "{}: chat {index}",
                live.what()
            );
        }
    });
}

/// The fill arrives from a SERVER: the row is applied by the sync manager, not written
/// through this engine's own write path, and the binding it lands on was made while
/// the client held nothing for it.
#[test]
fn a_binding_is_filled_by_a_row_arriving_from_the_server() {
    use crate::sync_manager::{ClientId, ServerId};
    use uuid::Uuid;

    use crate::sync_manager::DurabilityTier;

    // With no tier asked for, and asking for rows the edge has confirmed: a tier
    // filter sits over the instance's rows, and an empty binding has none to filter.
    let ways = [Empty::Bindings, Empty::Instances]
        .into_iter()
        .flat_map(|empty| [None, Some(DurabilityTier::EdgeServer)].map(|tier| (empty, tier)));
    for (empty, tier) in ways {
        let _precise = force_precise_dirty(true);
        let _routing = force_include_routing(true);
        let _empty = force_include_empty_probe(matches!(empty, Empty::Bindings));
        let _parity = check_empty_bindings_against_instances();

        let (mut server, mut server_io) = create_query_manager(
            SyncManager::new().with_durability_tier(DurabilityTier::EdgeServer),
            fills_schema(),
        );
        let chats: Vec<ObjectId> = (0..OUTER_ROWS)
            .map(|index| {
                server
                    .insert(&mut server_io, "chats", &[text(&format!("chat-{index}"))])
                    .expect("insert chat")
                    .row_id
            })
            .collect();
        server.process(&mut server_io);

        let (mut client, mut client_io) = create_query_manager(SyncManager::new(), fills_schema());
        let server_id = ServerId(Uuid::new_v7(uuid::Timestamp::now(uuid::NoContext)));
        let client_id = ClientId(Uuid::new_v7(uuid::Timestamp::now(uuid::NoContext)));
        connect_server(&mut client, &client_io, server_id);
        connect_client(&mut server, &server_io, client_id);
        let _ = client.sync_manager_mut().take_outbox();

        let bound_before = empty_bindings_bound_on_this_thread();
        let query = reverse_include(&client, false);
        let sub_id = client.subscribe_with_sync(query, None, tier).unwrap();
        pump_messages(
            &mut client,
            &mut server,
            &mut client_io,
            &mut server_io,
            client_id,
            server_id,
        );
        let notes_per_chat = |client: &QueryManager| -> HashMap<ObjectId, usize> {
            client
                .get_subscription_results(sub_id)
                .iter()
                .map(|(id, values)| {
                    (
                        *id,
                        values
                            .last()
                            .and_then(Value::as_array)
                            .expect("notes include")
                            .len(),
                    )
                })
                .collect()
        };
        let held = notes_per_chat(&client);
        assert_eq!(held.len(), OUTER_ROWS, "{empty:?}/{tier:?}");
        assert_eq!(held.values().sum::<usize>(), 0, "{empty:?}/{tier:?}");
        let bound = empty_bindings_bound_on_this_thread() - bound_before;
        match empty {
            Empty::Bindings => assert!(bound >= (OUTER_ROWS - 1) as u64, "{bound}"),
            Empty::Instances => assert_eq!(bound, 0),
        }

        server
            .insert(
                &mut server_io,
                "notes",
                &[text("from the server"), Value::Uuid(chats[4])],
            )
            .expect("insert note");
        pump_messages(
            &mut client,
            &mut server,
            &mut client_io,
            &mut server_io,
            client_id,
            server_id,
        );
        let held = notes_per_chat(&client);
        assert_eq!(
            held[&chats[4]], 1,
            "{empty:?}/{tier:?}: the note the server sent is missing from its chat"
        );
        assert_eq!(held.values().sum::<usize>(), 1, "{empty:?}/{tier:?}");
    }
}

/// Deliver a row written under another schema generation, the way a server delivers
/// it: on that generation's branch, encoded with that generation's descriptor.
fn receive_on_branch(
    live: &mut Live,
    table: &str,
    branch: &str,
    descriptor: &RowDescriptor,
    values: &[Value],
) -> ObjectId {
    use crate::query_manager::encoding::encode_row;

    let row_id = ObjectId::new();
    put_test_row_metadata(
        &mut live.storage,
        row_id,
        HashMap::from([(MetadataKey::Table.to_string(), table.to_string())]),
    );
    let commit = stored_row_commit(
        smallvec![],
        encode_row(descriptor, values).expect("encode under the old descriptor"),
        1000,
        row_id.to_string(),
    );
    let row = commit.to_row(row_id, branch, RowState::VisibleDirect);
    live.qm.sync_manager_mut().push_inbox(InboxEntry {
        source: Source::Server(ServerId::new()),
        payload: SyncPayload::RowBatchCreated {
            metadata: None,
            row,
        },
    });
    row_id
}

fn other_branch(qm: &QueryManager) -> String {
    let current = get_branch(qm);
    qm.all_query_branches()
        .into_iter()
        .find(|branch| *branch != current)
        .expect("the old generation has a branch")
}

/// The fill arrives on ANOTHER schema generation's branch — a client still on the
/// previous build writes a note under a chat this engine holds an empty binding for.
/// The binding's "nothing here" was read off this generation's index AND that one's;
/// a probe that only asked the current branch would never be asked again.
#[test]
fn a_binding_is_filled_by_a_row_of_another_schema_generation() {
    each_way(|live| {
        // The old world: every table identical, one more table beside them — which
        // is all it takes for the schema to hash differently and get its own branch.
        let mut old_schema = fills_schema();
        let (name, table) = TableSchemaBuilder::new("retired")
            .column("kind", ColumnType::Text)
            .build_named();
        old_schema.insert(name, table);
        let old_notes = old_schema[&TableName::new("notes")].columns.clone();
        live.qm.add_live_schema(old_schema);
        let old_branch = other_branch(&live.qm);

        let chats: Vec<ObjectId> = (0..OUTER_ROWS)
            .map(|index| live.insert("chats", &[text(&format!("chat-{index}"))]))
            .collect();
        live.subscribe(reverse_include(&live.qm, false));
        for chat in &chats {
            assert_eq!(live.included(*chat), Some(0), "{}", live.what());
        }
        live.assert_bindings_were_made((OUTER_ROWS - 1) as u64);

        receive_on_branch(
            live,
            "notes",
            &old_branch,
            &old_notes,
            &[text("from the old build"), Value::Uuid(chats[2])],
        );
        live.settle_once();
        for (index, chat) in chats.iter().enumerate() {
            assert_eq!(
                live.included(*chat),
                Some(usize::from(index == 2)),
                "{}: chat {index} after a note arrived on the old generation's branch",
                live.what()
            );
        }
    });
}

/// The other generation goes live AFTER the bindings were made. They were read off the
/// branches there were then; the new branch is one more index nobody has asked. A
/// schema going live replaces the subscription's graph — node, probe and bindings with
/// it — and that replacement is the only thing that ever asks the new branch.
#[test]
fn a_binding_made_before_a_schema_generation_went_live_is_filled_from_it() {
    each_way(|live| {
        let chats: Vec<ObjectId> = (0..OUTER_ROWS)
            .map(|index| live.insert("chats", &[text(&format!("chat-{index}"))]))
            .collect();
        live.subscribe(reverse_include(&live.qm, false));
        live.assert_bindings_were_made((OUTER_ROWS - 1) as u64);

        let mut old_schema = fills_schema();
        let (name, table) = TableSchemaBuilder::new("retired")
            .column("kind", ColumnType::Text)
            .build_named();
        old_schema.insert(name, table);
        let old_notes = old_schema[&TableName::new("notes")].columns.clone();
        live.qm.add_live_schema(old_schema);
        let old_branch = other_branch(&live.qm);
        live.settle_once();
        for chat in &chats {
            assert_eq!(live.included(*chat), Some(0), "{}", live.what());
        }
        // The graph that replaced the first one answers from the index too, over both
        // branches now: the fill below is a fill of empty bindings, not of instances.
        live.assert_bindings_were_made(2 * (OUTER_ROWS - 1) as u64);

        receive_on_branch(
            live,
            "notes",
            &old_branch,
            &old_notes,
            &[text("from the old build"), Value::Uuid(chats[2])],
        );
        live.settle_once();
        for (index, chat) in chats.iter().enumerate() {
            assert_eq!(
                live.included(*chat),
                Some(usize::from(index == 2)),
                "{}: chat {index} after a note arrived on a branch that went live \
                 after the bindings were made",
                live.what()
            );
        }
    });
}

/// The same, across a lens that RENAMES the correlation column: the old generation
/// indexes its notes under `room`, the current one under `chat`. An instance scans
/// both, and the old generation's note has to reach its chat.
#[test]
fn a_binding_is_filled_across_a_lens_that_renames_the_correlation_column() {
    use crate::query_manager::types::SchemaHash;
    use crate::schema_manager::lens::{Lens, LensOp, LensTransform};

    each_way(|live| {
        let mut old_schema = fills_schema();
        let (name, table) = TableSchemaBuilder::new("notes")
            .column("title", ColumnType::Text)
            .fk_column("room", "chats")
            .build_named();
        old_schema.insert(name, table);
        let old_notes = old_schema[&TableName::new("notes")].columns.clone();
        let lens = Lens::new(
            SchemaHash::compute(&old_schema),
            SchemaHash::compute(&fills_schema()),
            LensTransform::with_ops(vec![LensOp::RenameColumn {
                table: "notes".to_string(),
                old_name: "room".to_string(),
                new_name: "chat".to_string(),
            }]),
        );
        live.qm.add_live_schema(old_schema);
        live.qm.register_lens(lens);
        let old_branch = other_branch(&live.qm);

        let chats: Vec<ObjectId> = (0..OUTER_ROWS)
            .map(|index| live.insert("chats", &[text(&format!("chat-{index}"))]))
            .collect();
        live.subscribe(reverse_include(&live.qm, false));
        for chat in &chats {
            assert_eq!(live.included(*chat), Some(0), "{}", live.what());
        }
        // The old generation's scan is by `room`, a column the current schema does
        // not have: the include cannot tell that scan is by its correlation, so it
        // answers nothing from the index and keeps an instance per binding. Making
        // this shape eligible means teaching the probe the lens — not loosening the
        // check that says which scans it may trust.
        assert_eq!(
            live.empty_bindings_bound(),
            0,
            "{}: an include across a renaming lens answered from the index",
            live.what()
        );

        receive_on_branch(
            live,
            "notes",
            &old_branch,
            &old_notes,
            &[text("from the old build"), Value::Uuid(chats[2])],
        );
        live.settle_once();
        // And one written here, under another chat, on the current generation.
        live.insert("notes", &[text("from this build"), Value::Uuid(chats[4])]);
        live.settle_once();
        for (index, chat) in chats.iter().enumerate() {
            assert_eq!(
                live.included(*chat),
                Some(usize::from(index == 2 || index == 4)),
                "{}: chat {index} with notes on both generations",
                live.what()
            );
        }
    });
}
