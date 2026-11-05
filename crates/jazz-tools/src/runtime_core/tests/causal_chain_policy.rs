//! Causal chains of client writes under strict permissions (Linsa.Server.Schema#22).
//!
//! A client writes a parent and then a child whose insert policy reads the parent through
//! `exists`: a chat, then its membership, then a message, then the message's attachment.
//! Offline, the client cannot wait for the server between the links; on reconnect the
//! server must accept the chain in the client's order.
//!
//! The policies are the shapes `definePermissions` compiles the app's rules to:
//! `policy.<t>.exists.where({ id: row.chatId, … })` becomes `Exists` over `<t>` with the
//! outer row's columns as `__jazz_outer_row` session refs, and the id guards
//! (`idIsFree` / `idIsFreeOrOwn`) are `Not(Exists)` over the written table itself.
//!
//! Real runtimes on both ends, a separate permissions head, SqliteStorage.

use super::*;

use crate::batch_fate::{BatchFate, BatchMode};
use crate::query_manager::policy::{CmpOp, OUTER_ROW_SESSION_PREFIX, PolicyValue};
use crate::row_histories::{BatchId, RowState};
use crate::storage::SqliteStorage;

const APP: &str = "causal-chain-policy";
const ME: &str = "alice";

type Core = RuntimeCore<SqliteStorage, NoopScheduler>;

fn outer(column: &str) -> PolicyValue {
    PolicyValue::SessionRef(vec![OUTER_ROW_SESSION_PREFIX.into(), column.into()])
}

fn me() -> PolicyValue {
    PolicyValue::SessionRef(vec!["user_id".into()])
}

fn cmp(column: &str, value: PolicyValue) -> PolicyExpr {
    PolicyExpr::Cmp {
        column: column.into(),
        op: CmpOp::Eq,
        value,
    }
}

fn exists(table: &str, conditions: Vec<PolicyExpr>) -> PolicyExpr {
    PolicyExpr::Exists {
        table: table.into(),
        condition: Box::new(PolicyExpr::and(conditions)),
    }
}

/// `idIsFree(table)`: no row with the written id.
fn id_is_free(table: &str) -> PolicyExpr {
    PolicyExpr::not(exists(table, vec![cmp("id", outer("id"))]))
}

/// `idIsFreeOrOwn(table, owner, frozen)`: no row with the written id that someone else
/// owns, or whose frozen columns differ from the written ones.
fn id_is_free_or_own(table: &str, owner: &str, frozen: &[&str]) -> PolicyExpr {
    let mut taken = vec![
        PolicyExpr::IsNull {
            column: owner.into(),
        },
        PolicyExpr::not(cmp(owner, me())),
    ];
    taken.extend(
        frozen
            .iter()
            .map(|column| PolicyExpr::not(cmp(column, outer(column)))),
    );
    PolicyExpr::not(exists(
        table,
        vec![cmp("id", outer("id")), PolicyExpr::or(taken)],
    ))
}

fn structural_schema() -> Schema {
    SchemaBuilder::new()
        .table(
            TableSchema::builder("chats")
                .column("created_by", ColumnType::Text)
                .column("kind", ColumnType::Text),
        )
        .table(
            TableSchema::builder("chat_members")
                .column("chat_id", ColumnType::Uuid)
                .column("user_id", ColumnType::Text)
                .column("is_banned", ColumnType::Boolean),
        )
        .table(
            TableSchema::builder("messages")
                .column("chat_id", ColumnType::Uuid)
                .column("sender_id", ColumnType::Text)
                .column("body", ColumnType::Text),
        )
        .table(
            TableSchema::builder("message_attachments")
                .column("message_id", ColumnType::Uuid)
                .column("kind", ColumnType::Text),
        )
        .table(TableSchema::builder("agents").column("owner_id", ColumnType::Text))
        .table(
            TableSchema::builder("chat_agents")
                .column("chat_id", ColumnType::Uuid)
                .column("agent_id", ColumnType::Uuid),
        )
        .table(
            TableSchema::builder("chat_summaries")
                .column("chat_id", ColumnType::Uuid)
                .column("body", ColumnType::Text),
        )
        .build()
}

fn policies(insert: PolicyExpr) -> TablePolicies {
    TablePolicies::new()
        .with_select(PolicyExpr::True)
        .with_insert(insert)
        .with_update(Some(PolicyExpr::True), PolicyExpr::True)
        .with_delete(PolicyExpr::True)
}

/// [`policies`] whose update and delete read the row: only its `owner` may change it.
fn owned_policies(insert: PolicyExpr, owner: &str) -> TablePolicies {
    TablePolicies::new()
        .with_select(PolicyExpr::True)
        .with_insert(insert)
        .with_update(Some(cmp(owner, me())), cmp(owner, me()))
        .with_delete(cmp(owner, me()))
}

/// `memberOfChat(row)`: I am an unbanned member of the row's chat.
fn member_of_chat() -> PolicyExpr {
    exists(
        "chat_members",
        vec![
            cmp("chat_id", outer("chat_id")),
            cmp("user_id", me()),
            cmp("is_banned", PolicyValue::Literal(Value::Boolean(false))),
        ],
    )
}

/// The Schema#22 insert rules for the chain, one table each.
fn permissions() -> Schema {
    let mut head = structural_schema();
    for (name, table) in head.iter_mut() {
        table.policies = match name.as_str() {
            // chats.allowInsert: mine, of a client kind, and the id free or my own.
            "chats" => owned_policies(
                PolicyExpr::and(vec![
                    cmp("created_by", me()),
                    PolicyExpr::InList {
                        column: "kind".into(),
                        values: ["assistant", "savedMessages", "oneToOne"]
                            .into_iter()
                            .map(|kind| PolicyValue::Literal(Value::Text(kind.into())))
                            .collect(),
                    },
                    id_is_free_or_own("chats", "created_by", &["kind"]),
                ]),
                "created_by",
            ),
            // chat_members.allowInsert: my own membership of a chat I created.
            "chat_members" => policies(PolicyExpr::and(vec![
                cmp("user_id", me()),
                exists(
                    "chats",
                    vec![cmp("id", outer("chat_id")), cmp("created_by", me())],
                ),
                id_is_free_or_own("chat_members", "user_id", &["chat_id"]),
            ])),
            // messages.allowInsert: sent by me into a chat I am an unbanned member of.
            "messages" => owned_policies(
                PolicyExpr::and(vec![
                    cmp("sender_id", me()),
                    member_of_chat(),
                    id_is_free_or_own("messages", "sender_id", &["chat_id"]),
                ]),
                "sender_id",
            ),
            // message_attachments.allowInsert: on a message I sent, at a free id.
            "message_attachments" => policies(PolicyExpr::and(vec![
                exists(
                    "messages",
                    vec![cmp("id", outer("message_id")), cmp("sender_id", me())],
                ),
                id_is_free("message_attachments"),
            ])),
            // agents.allowInsert: my own agent.
            "agents" => policies(cmp("owner_id", me())),
            // chat_agents.allowInsert: my agent, on an assistant chat I created.
            "chat_agents" => policies(PolicyExpr::and(vec![
                exists(
                    "chats",
                    vec![
                        cmp("id", outer("chat_id")),
                        cmp("created_by", me()),
                        cmp("kind", PolicyValue::Literal(text("assistant"))),
                    ],
                ),
                exists(
                    "agents",
                    vec![cmp("id", outer("agent_id")), cmp("owner_id", me())],
                ),
            ])),
            // chat_summaries.allowInsert: memberOfChat.
            "chat_summaries" => policies(member_of_chat()),
            other => panic!("no rule for {other}"),
        };
    }
    head
}

fn runtime(storage: SqliteStorage) -> Core {
    runtime_with(storage, SyncManager::new())
}

fn runtime_with(storage: SqliteStorage, sync_manager: SyncManager) -> Core {
    let app_id = AppId::from_name(APP);
    let mut schema_manager =
        SchemaManager::new(sync_manager, structural_schema(), app_id, "dev", "main").unwrap();
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

fn sqlite() -> SqliteStorage {
    SqliteStorage::open(":memory:").expect("in-memory sqlite storage should open")
}

/// The server, at the tier a deployed one has with no upstream (`server::builder`). Without
/// a tier a node never stores a fate, and what a gate would read as accepted is only the
/// client's own local fate.
fn server_runtime() -> Core {
    let mut core = runtime_with(
        sqlite(),
        SyncManager::new().with_durability_tier(crate::sync_manager::DurabilityTier::GlobalServer),
    );
    core.schema_manager_mut()
        .query_manager_mut()
        .set_authorization_schema(permissions());
    core.schema_manager_mut()
        .query_manager_mut()
        .require_authorization_schema();
    core.immediate_tick();
    core
}

fn text(value: &str) -> Value {
    Value::Text(value.into())
}

struct World {
    server: Core,
    client: Core,
    server_id: ServerId,
    client_id: ClientId,
    connected: bool,
    /// Every `Rejected` fate the server sent the client, with its reason.
    rejected: HashMap<BatchId, String>,
    /// Every payload the client sent the server.
    sent: Vec<SyncPayload>,
    /// Other users' clients, always connected.
    peers: Vec<(Core, ClientId)>,
}

impl World {
    fn new(connected: bool) -> Self {
        let mut server = server_runtime();
        let mut client = runtime(sqlite());
        let server_id = ServerId::new();
        let client_id = ClientId::new();
        server.add_client(client_id, Some(Session::new(ME)));
        if connected {
            client.add_server(server_id);
        }
        let mut world = Self {
            server,
            client,
            server_id,
            client_id,
            connected,
            rejected: HashMap::new(),
            sent: Vec::new(),
            peers: Vec::new(),
        };
        world.sync();
        world
    }

    fn mine() -> WriteContext {
        WriteContext::from_session(Session::new(ME))
    }

    /// Another user's client, connected; returns its index in `peers`.
    fn add_peer(&mut self, user: &str) -> usize {
        let mut peer = runtime(sqlite());
        let peer_id = ClientId::new();
        self.server.add_client(peer_id, Some(Session::new(user)));
        peer.add_server(self.server_id);
        self.peers.push((peer, peer_id));
        self.sync();
        self.peers.len() - 1
    }

    fn connect(&mut self) {
        assert!(!self.connected);
        self.client.add_server(self.server_id);
        self.connected = true;
        self.sync();
    }

    /// Both directions until nothing moves; the server's refusals are recorded.
    fn sync(&mut self) {
        for _ in 0..16 {
            let mut moved = false;
            self.client.batched_tick();
            for entry in self.client.sync_sender().take() {
                if self.connected && entry.destination == Destination::Server(self.server_id) {
                    moved = true;
                    self.sent.push(entry.payload.clone());
                    self.server.park_sync_message(InboxEntry {
                        source: Source::Client(self.client_id),
                        payload: entry.payload,
                    });
                }
            }
            for (peer, peer_id) in &mut self.peers {
                peer.batched_tick();
                for entry in peer.sync_sender().take() {
                    if entry.destination == Destination::Server(self.server_id) {
                        moved = true;
                        self.server.park_sync_message(InboxEntry {
                            source: Source::Client(*peer_id),
                            payload: entry.payload,
                        });
                    }
                }
            }
            self.server.batched_tick();
            self.server.immediate_tick();
            self.server.batched_tick();
            for entry in self.server.sync_sender().take() {
                if let Some((peer, _)) = self
                    .peers
                    .iter_mut()
                    .find(|(_, peer_id)| entry.destination == Destination::Client(*peer_id))
                {
                    moved = true;
                    peer.park_sync_message(InboxEntry {
                        source: Source::Server(self.server_id),
                        payload: entry.payload,
                    });
                    continue;
                }
                if entry.destination != Destination::Client(self.client_id) {
                    continue;
                }
                if let SyncPayload::BatchFate {
                    fate:
                        BatchFate::Rejected {
                            batch_id, reason, ..
                        },
                } = &entry.payload
                {
                    self.rejected.insert(*batch_id, reason.clone());
                }
                if self.connected {
                    moved = true;
                    self.client.park_sync_message(InboxEntry {
                        source: Source::Server(self.server_id),
                        payload: entry.payload,
                    });
                }
            }
            self.client.batched_tick();
            self.client.immediate_tick();
            for (peer, _) in &mut self.peers {
                peer.batched_tick();
                peer.immediate_tick();
            }
            if !moved {
                break;
            }
        }
    }

    fn insert(&mut self, table: &str, values: Vec<(&str, Value)>) -> (ObjectId, BatchId) {
        self.insert_in(table, values, None, Self::mine())
    }

    fn insert_in(
        &mut self,
        table: &str,
        values: Vec<(&str, Value)>,
        id: Option<ObjectId>,
        context: WriteContext,
    ) -> (ObjectId, BatchId) {
        let values = values
            .into_iter()
            .map(|(column, value)| (column.to_string(), value))
            .collect();
        let ((row, _), batch) = self
            .client
            .insert_with_id(table, values, id, Some(&context))
            .unwrap_or_else(|error| panic!("the {table} insert applies locally: {error:?}"));
        (row, batch)
    }

    fn chat(&mut self) -> (ObjectId, BatchId) {
        self.insert(
            "chats",
            vec![("created_by", text(ME)), ("kind", text("assistant"))],
        )
    }

    fn membership(&mut self, chat: ObjectId) -> (ObjectId, BatchId) {
        self.insert(
            "chat_members",
            vec![
                ("chat_id", Value::Uuid(chat)),
                ("user_id", text(ME)),
                ("is_banned", Value::Boolean(false)),
            ],
        )
    }

    fn message(&mut self, chat: ObjectId, id: Option<ObjectId>) -> (ObjectId, BatchId) {
        self.insert_in(
            "messages",
            vec![
                ("chat_id", Value::Uuid(chat)),
                ("sender_id", text(ME)),
                ("body", text("hello")),
            ],
            id,
            Self::mine(),
        )
    }

    fn attachment(&mut self, message: ObjectId) -> (ObjectId, BatchId) {
        self.insert(
            "message_attachments",
            vec![
                ("message_id", Value::Uuid(message)),
                ("kind", text("photo")),
            ],
        )
    }

    /// The batch's fate as the SERVER stored it. A client's own `DurableDirect{Local}` is
    /// not an acceptance, so a batch only the client has settled reads "local only"; and a
    /// client that disagrees with the server's stored fate fails the gate on the spot.
    fn fate(&self, batch: BatchId) -> String {
        fn kind(fate: &BatchFate) -> &'static str {
            match fate {
                BatchFate::Rejected { .. } => "rejected",
                BatchFate::DurableDirect { .. } | BatchFate::AcceptedTransaction { .. } => {
                    "accepted"
                }
                BatchFate::Missing { .. } => "missing",
            }
        }
        let client = self.client.batch_fate(batch).expect("load batch fate");
        let server = self
            .server
            .storage()
            .load_authoritative_batch_fate(batch)
            .expect("load the server's batch fate");
        match (&server, &client) {
            (Some(server_fate), Some(client_fate))
                if self.connected && kind(server_fate) != kind(client_fate) =>
            {
                panic!(
                    "client and server disagree on {batch:?}: server {server_fate:?}, client {client_fate:?}"
                )
            }
            (Some(server_fate), _) => kind(server_fate).into(),
            (None, None) => "none".into(),
            (None, Some(_)) => "local only".into(),
        }
    }

    /// `(label, fate, server reason)` for each named batch.
    fn report(&self, batches: &[(&str, BatchId)]) -> Vec<String> {
        batches
            .iter()
            .map(|(label, batch)| {
                let reason = self
                    .rejected
                    .get(batch)
                    .map(|reason| format!(" ({reason})"))
                    .unwrap_or_default();
                format!("{label}: {}{reason}", self.fate(*batch))
            })
            .collect()
    }

    fn assert_all_accepted(&self, batches: &[(&str, BatchId)]) {
        let report = self.report(batches);
        assert!(
            batches
                .iter()
                .all(|(_, batch)| self.fate(*batch) == "accepted"),
            "every link of the chain must be accepted in the client's order: {report:#?}"
        );
    }

    /// The server serves every named row: `(label, table, id)`.
    fn assert_served(&mut self, rows: &[(&str, &str, ObjectId)]) {
        let missing: Vec<&str> = rows
            .iter()
            .filter(|(_, table, id)| !self.served(table).contains(id))
            .map(|(label, _, _)| *label)
            .collect();
        assert!(
            missing.is_empty(),
            "the server serves every link; missing: {missing:?}"
        );
    }

    /// Ids of the rows the server serves from `table`.
    fn served(&mut self, table: &str) -> Vec<ObjectId> {
        let sub = self
            .server
            .schema_manager_mut()
            .query_manager_mut()
            .subscribe_with_session(Query::new(table), None, None)
            .unwrap();
        self.server.immediate_tick();
        let ids = self
            .server
            .schema_manager()
            .query_manager()
            .get_subscription_results(sub)
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        self.server
            .schema_manager_mut()
            .query_manager_mut()
            .unsubscribe_with_sync(sub);
        ids
    }

    /// The values the server serves for row `id` of `table`, in column order.
    fn served_row(&mut self, table: &str, id: ObjectId) -> Option<Vec<Value>> {
        let sub = self
            .server
            .schema_manager_mut()
            .query_manager_mut()
            .subscribe_with_session(Query::new(table), None, None)
            .unwrap();
        self.server.immediate_tick();
        let row = self
            .server
            .schema_manager()
            .query_manager()
            .get_subscription_results(sub)
            .into_iter()
            .find(|(row_id, _)| *row_id == id)
            .map(|(_, values)| values);
        self.server
            .schema_manager_mut()
            .query_manager_mut()
            .unsubscribe_with_sync(sub);
        row
    }

    /// Ids of the rows the client shows in `table`.
    fn shown(&mut self, table: &str) -> Vec<ObjectId> {
        let sub = self
            .client
            .schema_manager_mut()
            .query_manager_mut()
            .subscribe_with_session(Query::new(table), None, None)
            .unwrap();
        self.client.immediate_tick();
        let ids = self
            .client
            .schema_manager()
            .query_manager()
            .get_subscription_results(sub)
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        self.client
            .schema_manager_mut()
            .query_manager_mut()
            .unsubscribe_with_sync(sub);
        ids
    }
}

fn chain_rows(
    chat: ObjectId,
    member: ObjectId,
    message: ObjectId,
    attachment: ObjectId,
) -> [(&'static str, &'static str, ObjectId); 4] {
    [
        ("chat", "chats", chat),
        ("membership", "chat_members", member),
        ("message", "messages", message),
        ("attachment", "message_attachments", attachment),
    ]
}

/// Control: with a full round trip between the links, the rules admit the chain.
#[test]
fn the_chain_is_accepted_when_each_link_waits_for_the_server() {
    let mut world = World::new(true);
    let (chat, chat_batch) = world.chat();
    world.sync();
    let (member, member_batch) = world.membership(chat);
    world.sync();
    let (message, message_batch) = world.message(chat, None);
    world.sync();
    let (attachment, attachment_batch) = world.attachment(message);
    world.sync();
    world.assert_all_accepted(&[
        ("chat", chat_batch),
        ("membership", member_batch),
        ("message", message_batch),
        ("attachment", attachment_batch),
    ]);
    world.assert_served(&chain_rows(chat, member, message, attachment));
}

/// Connected, the links written back to back: each its own batch, none waits.
#[test]
fn the_chain_written_back_to_back_is_accepted_in_order() {
    let mut world = World::new(true);
    let (chat, chat_batch) = world.chat();
    let (member, member_batch) = world.membership(chat);
    let (message, message_batch) = world.message(chat, None);
    let (attachment, attachment_batch) = world.attachment(message);
    world.sync();
    world.assert_all_accepted(&[
        ("chat", chat_batch),
        ("membership", member_batch),
        ("message", message_batch),
        ("attachment", attachment_batch),
    ]);
    world.assert_served(&chain_rows(chat, member, message, attachment));
}

/// Offline, the links written back to back; then the client connects.
#[test]
fn the_chain_written_offline_is_accepted_in_order_on_reconnect() {
    let mut world = World::new(false);
    let (chat, chat_batch) = world.chat();
    let (member, member_batch) = world.membership(chat);
    let (message, message_batch) = world.message(chat, None);
    let (attachment, attachment_batch) = world.attachment(message);
    world.connect();
    world.assert_all_accepted(&[
        ("chat", chat_batch),
        ("membership", member_batch),
        ("message", message_batch),
        ("attachment", attachment_batch),
    ]);
    world.assert_served(&chain_rows(chat, member, message, attachment));
}

/// A rejected insert must leave its id free: the client takes the row back, and once the
/// rule admits it, the same id inserts again.
#[test]
fn a_rejected_insert_leaves_its_id_free_for_a_retry() {
    let mut world = World::new(true);
    let (chat, chat_batch) = world.chat();
    world.sync();
    // No membership yet: the message is refused.
    let (message, refused_batch) = world.message(chat, None);
    world.sync();
    assert_eq!(
        world.fate(refused_batch),
        "rejected",
        "control: a message without a membership is refused: {:?}",
        world.report(&[("refused message", refused_batch)])
    );
    let shown_after_refusal = world.shown("messages");

    let (_, member_batch) = world.membership(chat);
    world.sync();
    let (retried, retry_batch) = world.message(chat, Some(message));
    assert_eq!(retried, message);
    world.sync();
    let served = world.served("messages");
    assert!(
        !shown_after_refusal.contains(&message),
        "the refused message must be taken back locally"
    );
    world.assert_all_accepted(&[
        ("chat", chat_batch),
        ("membership", member_batch),
        ("retried message", retry_batch),
    ]);
    assert!(
        served.contains(&message),
        "the server serves the retried message"
    );
}

/// A Lin session written offline: the agent, the chat, my membership, the agent on the chat,
/// the chat's summary and the first message.
#[test]
fn the_lin_session_written_offline_is_accepted_in_order_on_reconnect() {
    let mut world = World::new(false);
    let (agent, agent_batch) = world.insert("agents", vec![("owner_id", text(ME))]);
    let (chat, chat_batch) = world.chat();
    let (member, member_batch) = world.membership(chat);
    let (chat_agent, chat_agent_batch) = world.insert(
        "chat_agents",
        vec![
            ("chat_id", Value::Uuid(chat)),
            ("agent_id", Value::Uuid(agent)),
        ],
    );
    let (summary, summary_batch) = world.insert(
        "chat_summaries",
        vec![("chat_id", Value::Uuid(chat)), ("body", text(""))],
    );
    let (message, message_batch) = world.message(chat, None);
    world.connect();
    world.assert_all_accepted(&[
        ("agent", agent_batch),
        ("chat", chat_batch),
        ("membership", member_batch),
        ("chat agent", chat_agent_batch),
        ("summary", summary_batch),
        ("message", message_batch),
    ]);
    world.assert_served(&[
        ("agent", "agents", agent),
        ("chat", "chats", chat),
        ("membership", "chat_members", member),
        ("chat agent", "chat_agents", chat_agent),
        ("summary", "chat_summaries", summary),
        ("message", "messages", message),
    ]);
}

/// A refused parent takes the chain written on it along: no link is admitted by a row the
/// server never accepted.
#[test]
fn a_refused_parent_refuses_the_chain_written_on_it() {
    let mut world = World::new(false);
    // Control: a chat of a client kind, written in the same pass, is accepted.
    let (kept, kept_batch) = world.chat();
    // A group chat is not a kind a client may create.
    let (chat, chat_batch) = world.insert(
        "chats",
        vec![("created_by", text(ME)), ("kind", text("group"))],
    );
    let (_, member_batch) = world.membership(chat);
    let (message, message_batch) = world.message(chat, None);
    let (_, attachment_batch) = world.attachment(message);
    world.connect();
    let batches = [
        ("chat", chat_batch),
        ("membership", member_batch),
        ("message", message_batch),
        ("attachment", attachment_batch),
    ];
    let report = world.report(&batches);
    assert!(
        batches
            .iter()
            .all(|(_, batch)| world.fate(*batch) == "rejected"),
        "every link written on a refused parent is refused: {report:#?}"
    );
    assert_eq!(world.fate(kept_batch), "accepted", "control");
    assert_eq!(
        world.served("chats"),
        vec![kept],
        "only the control chat is served"
    );
    for table in ["chat_members", "messages", "message_attachments"] {
        assert!(
            world.served(table).is_empty(),
            "the server serves no {table}"
        );
    }
}

/// Written, edited and deleted offline: the edit and the delete of a row written in the same
/// pass are decided on that row.
#[test]
fn a_row_written_edited_and_deleted_offline_is_deleted_on_reconnect() {
    let mut world = World::new(false);
    let (chat, chat_batch) = world.chat();
    let (_, member_batch) = world.membership(chat);
    let (message, message_batch) = world.message(chat, None);
    let edit_batch = world
        .client
        .update(
            message,
            vec![("body".into(), text("edited"))],
            Some(&World::mine()),
        )
        .expect("the edit applies locally");
    let delete_batch = world
        .client
        .delete(message, Some(&World::mine()))
        .expect("the delete applies locally");
    world.connect();
    world.assert_all_accepted(&[
        ("chat", chat_batch),
        ("membership", member_batch),
        ("message", message_batch),
        ("edit", edit_batch),
        ("delete", delete_batch),
    ]);
    assert!(
        !world.served("messages").contains(&message),
        "the server no longer serves the deleted message"
    );
}

/// Two users write the same derived chat id in one pass: the chat is the first one's, and
/// the second meets it exactly as it would one pass later.
#[test]
fn the_same_chat_id_written_by_two_users_in_one_pass_goes_to_the_first() {
    let mut world = World::new(true);
    let bob = world.add_peer("bob");
    let chat = ObjectId::new();
    let (_, alice_batch) = world.insert_in(
        "chats",
        vec![("created_by", text(ME)), ("kind", text("oneToOne"))],
        Some(chat),
        World::mine(),
    );
    let bob_context = WriteContext::from_session(Session::new("bob"));
    let values = HashMap::from([
        ("created_by".to_string(), text("bob")),
        ("kind".to_string(), text("oneToOne")),
    ]);
    let (_, bob_batch) = world.peers[bob]
        .0
        .insert_with_id("chats", values, Some(chat), Some(&bob_context))
        .expect("bob's insert applies locally");
    world.sync();
    assert_eq!(world.fate(alice_batch), "accepted");
    assert!(
        matches!(
            world.peers[bob]
                .0
                .batch_fate(bob_batch)
                .expect("load bob's fate"),
            Some(BatchFate::Rejected { .. })
        ),
        "the second writer of the id is refused"
    );
    assert_eq!(
        world
            .served_row("chats", chat)
            .map(|values| values[0].clone()),
        Some(text(ME)),
        "the chat stays the first writer's"
    );
}

/// A late copy of the old refusal reaches the client after the retry was accepted: the
/// retried row stays.
#[test]
fn a_late_copy_of_the_old_refusal_leaves_the_retried_row() {
    let mut world = World::new(true);
    let (chat, _) = world.chat();
    world.sync();
    let (message, refused_batch) = world.message(chat, None);
    world.sync();
    assert_eq!(world.fate(refused_batch), "rejected", "control");
    world.membership(chat);
    world.sync();
    let (_, retry_batch) = world.message(chat, Some(message));
    world.sync();
    assert_eq!(world.fate(retry_batch), "accepted", "control");

    let refusal = BatchFate::Rejected {
        batch_id: refused_batch,
        code: "permission_denied".into(),
        reason: world.rejected[&refused_batch].clone(),
    };
    world.client.park_sync_message(InboxEntry {
        source: Source::Server(world.server_id),
        payload: SyncPayload::BatchFate { fate: refusal },
    });
    world.client.batched_tick();
    world.client.immediate_tick();
    assert!(
        world.shown("messages").contains(&message),
        "the old refusal does not take the retried message back"
    );
    assert_eq!(world.fate(retry_batch), "accepted");
}

/// The server stored a member of a batch it then refused: a retry at that member's id is
/// still a plain insert.
#[test]
fn a_row_of_a_refused_batch_inserts_again_at_its_id() {
    let mut world = World::new(true);
    let (chat, chat_batch) = world.chat();
    world.sync();
    let batch = world.client.begin_batch(BatchMode::Direct);
    let in_batch = World::mine()
        .with_batch_mode(BatchMode::Direct)
        .with_batch_id(batch);
    // Written first, so its id sorts first and the server stores it before the refusal.
    let membership = vec![
        ("chat_id", Value::Uuid(chat)),
        ("user_id", text(ME)),
        ("is_banned", Value::Boolean(false)),
    ];
    let (member, _) = world.insert_in("chat_members", membership.clone(), None, in_batch.clone());
    // Someone else's message: refused, and the batch with it.
    world.insert_in(
        "messages",
        vec![
            ("chat_id", Value::Uuid(chat)),
            ("sender_id", text("bob")),
            ("body", text("hello")),
        ],
        None,
        in_batch,
    );
    world.client.commit_batch(batch).expect("the batch commits");
    world.sync();
    assert_eq!(world.fate(batch), "rejected", "control");
    let server_versions = world
        .server
        .storage()
        .scan_history_row_batches("chat_members", member)
        .expect("scan the server's history of the member");
    assert!(
        !server_versions.is_empty()
            && server_versions
                .iter()
                .all(|version| version.state == RowState::Rejected),
        "control: the server stored the member and then refused it: {:?}",
        server_versions
            .iter()
            .map(|version| version.state)
            .collect::<Vec<_>>()
    );

    let (retried, retry_batch) =
        world.insert_in("chat_members", membership, Some(member), World::mine());
    assert_eq!(retried, member);
    world.sync();
    world.assert_all_accepted(&[("chat", chat_batch), ("retried membership", retry_batch)]);
    assert!(world.served("chat_members").contains(&member));
}

// ---- Differential oracle: random chains from two users against a model of the rules.

const BOB: &str = "bob";

/// A small deterministic generator, so a failing seed replays.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn pick<T: Copy>(&mut self, items: &[T]) -> Option<T> {
        (!items.is_empty()).then(|| items[self.below(items.len())])
    }
}

/// One write. `user` is the user its row names (its owner, member or sender); the writer
/// is said separately.
#[derive(Clone, Copy, Debug)]
enum Write {
    Chat {
        id: ObjectId,
        user: &'static str,
        kind: &'static str,
    },
    Member {
        id: ObjectId,
        chat: ObjectId,
        user: &'static str,
    },
    Message {
        id: ObjectId,
        chat: ObjectId,
        user: &'static str,
    },
    Attachment {
        id: ObjectId,
        message: ObjectId,
    },
    Delete {
        message: ObjectId,
    },
}

/// What the rules decide, write by write in the order the server receives them, over the
/// rows accepted before it.
#[derive(Default)]
struct Model {
    chats: HashMap<ObjectId, (&'static str, &'static str)>,
    members: HashMap<ObjectId, (ObjectId, &'static str)>,
    messages: HashMap<ObjectId, (ObjectId, &'static str)>,
    attachments: HashMap<ObjectId, ObjectId>,
}

impl Model {
    fn apply(&mut self, writer: &'static str, write: Write) -> bool {
        match write {
            Write::Chat { id, user, kind } => {
                // A taken id is someone else's, or mine of another kind: refused both as an
                // insert (`idIsFreeOrOwn`) and as an update of the row there (`created_by`).
                let admitted = user == writer
                    && ["assistant", "savedMessages", "oneToOne"].contains(&kind)
                    && self
                        .chats
                        .get(&id)
                        .is_none_or(|(owner, held)| *owner == writer && *held == kind);
                if admitted {
                    self.chats.insert(id, (writer, kind));
                }
                admitted
            }
            Write::Member { id, chat, user } => {
                let admitted = user == writer
                    && self
                        .chats
                        .get(&chat)
                        .is_some_and(|(owner, _)| *owner == writer);
                if admitted {
                    self.members.insert(id, (chat, writer));
                }
                admitted
            }
            Write::Message { id, chat, user } => {
                let admitted = user == writer
                    && self
                        .members
                        .values()
                        .any(|(member_chat, member)| *member_chat == chat && *member == writer);
                if admitted {
                    self.messages.insert(id, (chat, writer));
                }
                admitted
            }
            Write::Attachment { id, message } => {
                let admitted = self
                    .messages
                    .get(&message)
                    .is_some_and(|(_, sender)| *sender == writer);
                if admitted {
                    self.attachments.insert(id, message);
                }
                admitted
            }
            Write::Delete { message } => {
                let admitted = self
                    .messages
                    .get(&message)
                    .is_some_and(|(_, sender)| *sender == writer);
                if admitted {
                    self.messages.remove(&message);
                }
                admitted
            }
        }
    }
}

impl World {
    /// Make `write` locally on `writer`'s client; the error when that client refuses it.
    fn make(&mut self, writer: &'static str, write: Write) -> Result<BatchId, String> {
        let context = WriteContext::from_session(Session::new(writer));
        let core = if writer == ME {
            &mut self.client
        } else {
            &mut self.peers[0].0
        };
        let mut insert = |table: &str, id: ObjectId, row: Vec<(&str, Value)>| {
            let values = row
                .into_iter()
                .map(|(column, value)| (column.to_string(), value))
                .collect();
            core.insert_with_id(table, values, Some(id), Some(&context))
                .map(|(_, batch)| batch)
                .map_err(|error| format!("{error:?}"))
        };
        match write {
            Write::Chat { id, user, kind } => insert(
                "chats",
                id,
                vec![("created_by", text(user)), ("kind", text(kind))],
            ),
            Write::Member { id, chat, user } => insert(
                "chat_members",
                id,
                vec![
                    ("chat_id", Value::Uuid(chat)),
                    ("user_id", text(user)),
                    ("is_banned", Value::Boolean(false)),
                ],
            ),
            Write::Message { id, chat, user } => insert(
                "messages",
                id,
                vec![
                    ("chat_id", Value::Uuid(chat)),
                    ("sender_id", text(user)),
                    ("body", text("hello")),
                ],
            ),
            Write::Attachment { id, message } => insert(
                "message_attachments",
                id,
                vec![
                    ("message_id", Value::Uuid(message)),
                    ("kind", text("photo")),
                ],
            ),
            Write::Delete { message } => core
                .delete(message, Some(&context))
                .map_err(|error| format!("{error:?}")),
        }
    }

    fn fate_of(&self, writer: &str, batch: BatchId) -> String {
        if writer == ME {
            return self.fate(batch);
        }
        match self.peers[0].0.batch_fate(batch).expect("load batch fate") {
            Some(BatchFate::Rejected { .. }) => "rejected".into(),
            Some(BatchFate::DurableDirect { .. } | BatchFate::AcceptedTransaction { .. }) => {
                "accepted".into()
            }
            Some(BatchFate::Missing { .. }) => "missing".into(),
            None => "none".into(),
        }
    }
}

/// The rows written so far: everyone's, and the ones the writer wrote as the app would
/// (naming itself, on a chat of its own), whatever the server made of them.
struct Written<'a> {
    chats: &'a [ObjectId],
    own_chats: &'a [ObjectId],
    joined_chats: &'a [ObjectId],
    messages: &'a [ObjectId],
    own_messages: &'a [ObjectId],
}

/// The rows a user wrote as the app would, by kind.
#[derive(Default)]
struct OwnRows {
    chats: Vec<ObjectId>,
    joined_chats: Vec<ObjectId>,
    messages: Vec<ObjectId>,
}

fn random_write(rng: &mut Rng, writer: &'static str, written: &Written) -> Write {
    // Most rows name their writer and hang on a row the writer wrote before, so most
    // chains are long; some name the other user or hang on the other user's rows, so
    // refusals, and the chains written on them, happen too. Chat ids are sometimes taken
    // from the chats already written, by either user: a derived id two users both write.
    let user = if rng.below(6) == 0 {
        if writer == ME { BOB } else { ME }
    } else {
        writer
    };
    let fresh_or = |rng: &mut Rng, ids: &[ObjectId]| rng.pick(ids).unwrap_or_default();
    let (chats, joined_chats, messages) = if rng.below(4) == 0 {
        (written.chats, written.chats, written.messages)
    } else {
        (
            written.own_chats,
            written.joined_chats,
            written.own_messages,
        )
    };
    match rng.below(10) {
        0..=1 => Write::Chat {
            id: if rng.below(4) == 0 {
                fresh_or(rng, chats)
            } else {
                ObjectId::new()
            },
            user,
            kind: ["assistant", "oneToOne", "savedMessages", "group"][rng.below(4)],
        },
        2..=3 => Write::Member {
            id: ObjectId::new(),
            chat: fresh_or(rng, chats),
            user,
        },
        4..=6 => Write::Message {
            id: ObjectId::new(),
            chat: fresh_or(rng, joined_chats),
            user,
        },
        7..=8 => Write::Attachment {
            id: ObjectId::new(),
            message: fresh_or(rng, messages),
        },
        _ => Write::Delete {
            message: fresh_or(rng, messages),
        },
    }
}

/// Random chains of dependent writes from two users, each write its own batch. Alice is
/// offline and syncs once at the end, or online with syncs at random points; bob is online.
/// Every write's fate, and what the server serves, match the model of the rules applied in
/// the order the server receives the writes: per sync, alice's then bob's.
#[test]
fn random_chains_match_the_model_of_the_rules() {
    for seed in 1..=160u64 {
        let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let alice_offline = seed % 2 == 0;
        let mut world = World::new(!alice_offline);
        world.add_peer(BOB);
        let mut model = Model::default();
        let mut chats = Vec::new();
        let mut messages = Vec::new();
        let mut own: HashMap<&str, OwnRows> = HashMap::new();
        let mut unsent: HashMap<&str, Vec<(usize, Write, BatchId)>> = HashMap::new();
        let mut fates = Vec::new();
        let flush = |model: &mut Model,
                     unsent: &mut HashMap<&str, Vec<(usize, Write, BatchId)>>,
                     fates: &mut Vec<(usize, &'static str, Write, BatchId, bool)>,
                     users: &[&'static str]| {
            for user in users {
                for (step, write, batch) in unsent.remove(user).unwrap_or_default() {
                    fates.push((step, *user, write, batch, model.apply(user, write)));
                }
            }
        };
        for step in 0..36 {
            let writer = if rng.below(3) == 0 { BOB } else { ME };
            let OwnRows {
                chats: own_chats,
                joined_chats,
                messages: own_messages,
            } = own.entry(writer).or_default();
            let write = random_write(
                &mut rng,
                writer,
                &Written {
                    chats: &chats,
                    own_chats,
                    joined_chats,
                    messages: &messages,
                    own_messages,
                },
            );
            let batch = match world.make(writer, write) {
                Ok(batch) => batch,
                Err(reason) => {
                    // The client refuses only what the server could not have accepted
                    // either: a chat id that is held, a message it cannot delete. Anything
                    // else is a client-side regression, never lost coverage.
                    let pending = unsent.get(writer).map(Vec::as_slice).unwrap_or_default();
                    let held = |chat: ObjectId| {
                        model.chats.contains_key(&chat)
                            || pending
                                .iter()
                                .any(|(_, write, _)| matches!(write, Write::Chat { id, .. } if *id == chat))
                    };
                    let deletable = |message: ObjectId| {
                        let deleted = pending.iter().any(
                            |(_, write, _)| matches!(write, Write::Delete { message: m } if *m == message),
                        );
                        let written = model
                            .messages
                            .get(&message)
                            .is_some_and(|(_, sender)| *sender == writer)
                            || pending.iter().any(|(_, write, _)| {
                                matches!(write, Write::Message { id, user, .. } if *id == message && *user == writer)
                            });
                        written && !deleted
                    };
                    let expected = match write {
                        Write::Chat { id, .. } => {
                            reason.contains("object already exists") && held(id)
                        }
                        Write::Delete { message } => {
                            (reason.contains("object not found")
                                || reason.contains("row already deleted"))
                                && !deletable(message)
                        }
                        _ => false,
                    };
                    assert!(
                        expected,
                        "seed {seed} step {step}, {writer}: {write:?} refused locally: {reason}"
                    );
                    continue;
                }
            };
            match write {
                Write::Chat { id, user, kind } => {
                    chats.push(id);
                    if user == writer && kind != "group" {
                        own_chats.push(id);
                    }
                }
                Write::Member { chat, user, .. } => {
                    if user == writer && own_chats.contains(&chat) {
                        joined_chats.push(chat);
                    }
                }
                Write::Message { id, chat, user } => {
                    messages.push(id);
                    if user == writer && joined_chats.contains(&chat) {
                        own_messages.push(id);
                    }
                }
                _ => {}
            }
            unsent.entry(writer).or_default().push((step, write, batch));
            if rng.below(4) == 0 {
                world.sync();
                let users: &[&'static str] = if alice_offline { &[BOB] } else { &[ME, BOB] };
                flush(&mut model, &mut unsent, &mut fates, users);
            }
        }
        if alice_offline {
            world.connect();
        } else {
            world.sync();
        }
        flush(&mut model, &mut unsent, &mut fates, &[ME, BOB]);
        world.sync();

        let mode = if alice_offline {
            "alice offline"
        } else {
            "online"
        };
        for (step, writer, write, batch, admitted) in &fates {
            let expected = if *admitted { "accepted" } else { "rejected" };
            assert_eq!(
                world.fate_of(writer, *batch),
                expected,
                "seed {seed} ({mode}) step {step}, {writer}: {write:?} — {:?}",
                world.rejected.get(batch)
            );
        }
        let sorted = |mut ids: Vec<ObjectId>| {
            ids.sort();
            ids
        };
        for (table, expected) in [
            ("chats", model.chats.keys().copied().collect::<Vec<_>>()),
            ("chat_members", model.members.keys().copied().collect()),
            ("messages", model.messages.keys().copied().collect()),
            (
                "message_attachments",
                model.attachments.keys().copied().collect(),
            ),
        ] {
            assert_eq!(
                sorted(world.served(table)),
                sorted(expected),
                "seed {seed} ({mode}): served {table}"
            );
        }
    }
}
