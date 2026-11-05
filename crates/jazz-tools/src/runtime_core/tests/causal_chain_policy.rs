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

use crate::batch_fate::{BatchFate, BatchMode, SealedBatchSubmission};
use crate::query_manager::policy::{CmpOp, OUTER_ROW_SESSION_PREFIX, PolicyValue};
use crate::row_histories::{BatchId, RowState};
use crate::storage::SqliteStorage;
use std::collections::{BTreeMap, HashSet};

const APP: &str = "causal-chain-policy";
const ME: &str = "alice";

type Core = RuntimeCore<SqliteStorage, NoopScheduler>;
/// A row of a chain: what it stands for, its table, its id and its values.
type ChainRow<'a> = (&'a str, &'a str, ObjectId, Vec<(&'a str, Value)>);

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
            TableSchema::builder("notes")
                .column("created_by", ColumnType::Text)
                .column("kind", ColumnType::Text),
        )
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
            "notes" => policies(cmp("created_by", me())),
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

/// The app a world runs: its structural schema and its permissions head.
#[derive(Clone, Copy)]
struct Rules {
    structural: fn() -> Schema,
    permissions: fn() -> Schema,
}

impl Rules {
    /// The Schema#22 chain.
    const LINSA: Rules = Rules {
        structural: structural_schema,
        permissions,
    };
}

fn runtime_for(storage: SqliteStorage, sync_manager: SyncManager, rules: Rules) -> Core {
    let app_id = AppId::from_name(APP);
    let mut schema_manager =
        SchemaManager::new(sync_manager, (rules.structural)(), app_id, "dev", "main").unwrap();
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

/// The server, at `tier` as a deployed one always is (`server::builder`), or with none: a
/// node with no tier never settles a batch, so it holds every batch in the window before
/// its fate.
fn server_runtime_at(tier: Option<crate::sync_manager::DurabilityTier>) -> Core {
    server_runtime_over(sqlite(), tier)
}

/// [`server_runtime_at`] over a store it did not create: a restart.
fn server_runtime_over(
    storage: SqliteStorage,
    tier: Option<crate::sync_manager::DurabilityTier>,
) -> Core {
    server_runtime_for(storage, tier, Rules::LINSA)
}

fn server_runtime_for(
    storage: SqliteStorage,
    tier: Option<crate::sync_manager::DurabilityTier>,
    rules: Rules,
) -> Core {
    let sync_manager = match tier {
        Some(tier) => SyncManager::new().with_durability_tier(tier),
        None => SyncManager::new(),
    };
    let mut core = runtime_for(storage, sync_manager, rules);
    core.schema_manager_mut()
        .query_manager_mut()
        .set_authorization_schema((rules.permissions)());
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
    rules: Rules,
    /// The batches the server answered `Missing` for, once per answer.
    missing: Vec<BatchId>,
    /// Rows of the client that never reach the server, however often they are sent.
    lost_rows: HashSet<ObjectId>,
    /// The last send of each lost row, for `deliver_held_back`.
    held_back: BTreeMap<ObjectId, SyncPayload>,
}

impl World {
    /// On a server at the tier a deployed one has with no upstream (`server::builder`).
    fn new(connected: bool) -> Self {
        Self::with_server_tier(
            connected,
            Some(crate::sync_manager::DurabilityTier::GlobalServer),
        )
    }

    fn with_server_tier(
        connected: bool,
        tier: Option<crate::sync_manager::DurabilityTier>,
    ) -> Self {
        Self::with_rules_at(connected, tier, Rules::LINSA)
    }

    /// [`Self::new`], running another app.
    fn with_rules(connected: bool, rules: Rules) -> Self {
        Self::with_rules_at(
            connected,
            Some(crate::sync_manager::DurabilityTier::GlobalServer),
            rules,
        )
    }

    fn with_rules_at(
        connected: bool,
        tier: Option<crate::sync_manager::DurabilityTier>,
        rules: Rules,
    ) -> Self {
        let mut server = server_runtime_for(sqlite(), tier, rules);
        let mut client = runtime_for(sqlite(), SyncManager::new(), rules);
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
            rules,
            missing: Vec::new(),
            lost_rows: HashSet::new(),
            held_back: BTreeMap::new(),
        };
        world.sync();
        world
    }

    fn mine() -> WriteContext {
        WriteContext::from_session(Session::new(ME))
    }

    /// Another user's client, connected; returns its index in `peers`.
    fn add_peer(&mut self, user: &str) -> usize {
        let mut peer = runtime_for(sqlite(), SyncManager::new(), self.rules);
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
                if let SyncPayload::RowBatchCreated { row, .. } = &entry.payload
                    && self.lost_rows.contains(&row.row_id)
                {
                    self.held_back.insert(row.row_id, entry.payload.clone());
                    continue;
                }
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
                if let SyncPayload::BatchFate {
                    fate: BatchFate::Missing { batch_id },
                } = &entry.payload
                {
                    self.missing.push(*batch_id);
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
#[derive(Default, Clone)]
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
        self.make_in(
            writer,
            write,
            WriteContext::from_session(Session::new(writer)),
        )
    }

    /// [`Self::make`] in `context`: an explicit batch, for one.
    fn make_in(
        &mut self,
        writer: &'static str,
        write: Write,
        context: WriteContext,
    ) -> Result<BatchId, String> {
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

// ---- Differential oracle: random chains, some of their links written in one batch.

impl Write {
    fn id(&self) -> ObjectId {
        match *self {
            Write::Chat { id, .. }
            | Write::Member { id, .. }
            | Write::Message { id, .. }
            | Write::Attachment { id, .. } => id,
            Write::Delete { message } => message,
        }
    }
}

impl Model {
    /// A batch of `writer`'s rows, as the server decides it. Each row is first checked
    /// alone, in the order it arrives, against the rows accepted before the batch. The
    /// rows refused then are checked again, in id order and up to eight rounds, each
    /// against those rows plus the batch's rows accepted so far. The batch is accepted
    /// when every row is, and then all of them are; otherwise none.
    fn apply_batch(&mut self, writer: &'static str, arriving: &[Write]) -> bool {
        let mut state = self.clone();
        let mut waiting = Vec::new();
        for write in arriving {
            if self.clone().apply(writer, *write) {
                assert!(
                    state.apply(writer, *write),
                    "a row allowed alone is allowed with its batch's accepted rows"
                );
            } else {
                waiting.push(*write);
            }
        }
        waiting.sort_by_key(Write::id);
        for _ in 0..8 {
            let before = waiting.len();
            waiting.retain(|write| !state.apply(writer, *write));
            if waiting.is_empty() || waiting.len() == before {
                break;
            }
        }
        let accepted = waiting.is_empty();
        if accepted {
            *self = state;
        }
        accepted
    }
}

/// One of alice's units: a write of its own batch, or several in one explicit batch.
#[derive(Clone, Debug)]
enum Unit {
    Single(Write),
    /// The model decides both modes alike: the mode is there for the failure message.
    Batch(#[allow(dead_code)] BatchMode, Vec<Write>),
}

/// Random chains from two users, where alice sometimes writes two to five links in one
/// explicit batch of a random mode. Alice is offline and syncs once at the end, or online
/// with syncs between her units; bob is online and writes one batch per write. Every
/// batch's fate, and what the server serves, match the model: alice's units in order,
/// each batch decided as `Model::apply_batch` says.
#[test]
fn random_batches_match_the_model_of_the_rules() {
    for seed in 1..=120u64 {
        let mut rng = Rng(seed.wrapping_mul(0xD1B5_4A32_D192_ED03) | 1);
        let alice_offline = seed % 2 == 0;
        let mut world = World::new(!alice_offline);
        world.add_peer(BOB);
        let mut model = Model::default();
        let mut chats = Vec::new();
        let mut messages = Vec::new();
        let mut own: HashMap<&str, OwnRows> = HashMap::new();
        let mut unsent: HashMap<&str, Vec<(usize, Unit, BatchId)>> = HashMap::new();
        let mut fates: Vec<(usize, &'static str, Unit, BatchId, bool)> = Vec::new();
        let flush = |model: &mut Model,
                     unsent: &mut HashMap<&str, Vec<(usize, Unit, BatchId)>>,
                     fates: &mut Vec<(usize, &'static str, Unit, BatchId, bool)>,
                     users: &[&'static str],
                     sent_on_reconnect: bool| {
            for user in users {
                for (step, unit, batch) in unsent.remove(user).unwrap_or_default() {
                    let admitted = match &unit {
                        Unit::Single(write) => model.apply(user, *write),
                        Unit::Batch(_, writes) => {
                            // Sent as written; a resend goes in id order.
                            let mut arriving = writes.clone();
                            if sent_on_reconnect {
                                arriving.sort_by_key(Write::id);
                            }
                            model.apply_batch(user, &arriving)
                        }
                    };
                    fates.push((step, *user, unit, batch, admitted));
                }
            }
        };
        for step in 0..30 {
            let writer = if rng.below(3) == 0 { BOB } else { ME };
            let in_one_batch = writer == ME && rng.below(3) == 0;
            let links = if in_one_batch { 2 + rng.below(4) } else { 1 };
            let mode = if rng.below(2) == 0 {
                BatchMode::Direct
            } else {
                BatchMode::Transactional
            };
            let batch_context = in_one_batch.then(|| {
                let batch = world.client.begin_batch(mode);
                (batch, in_batch(batch, mode))
            });
            let mut written = Vec::new();
            let mut last_batch = None;
            for _ in 0..links {
                let OwnRows {
                    chats: own_chats,
                    joined_chats,
                    messages: own_messages,
                } = own.entry(writer).or_default();
                let mut write = random_write(
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
                if batch_context.is_some() {
                    // In a batch: no deletes, and every chat at a fresh id. Both are
                    // decided outside what this oracle models (a staged row's own id).
                    write = match write {
                        Write::Delete { .. } => Write::Message {
                            id: ObjectId::new(),
                            chat: rng.pick(joined_chats).unwrap_or_default(),
                            user: writer,
                        },
                        Write::Chat { user, kind, .. } => Write::Chat {
                            id: ObjectId::new(),
                            user,
                            kind,
                        },
                        other => other,
                    };
                }
                let made = match &batch_context {
                    Some((_, context)) => world.make_in(writer, write, context.clone()),
                    None => world.make(writer, write),
                };
                let batch = match made {
                    Ok(batch) => batch,
                    Err(reason) => {
                        assert!(
                            batch_context.is_none()
                                && match write {
                                    Write::Chat { .. } => reason.contains("object already exists"),
                                    Write::Delete { .. } => {
                                        reason.contains("object not found")
                                            || reason.contains("row already deleted")
                                    }
                                    _ => false,
                                },
                            "seed {seed} step {step}, {writer}: {write:?} refused locally: {reason}"
                        );
                        continue;
                    }
                };
                last_batch = Some(batch);
                match write {
                    Write::Chat { id, user, kind } => {
                        // A chat staged in a batch is invisible to the other user's id
                        // check until the batch is decided, a race this oracle does not
                        // model: its id is never offered to the other user.
                        if batch_context.is_none() {
                            chats.push(id);
                        }
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
                written.push(write);
            }
            let unit = match &batch_context {
                Some((batch, _)) => {
                    world
                        .client
                        .commit_batch(*batch)
                        .expect("the batch commits");
                    if written.is_empty() {
                        continue;
                    }
                    (Unit::Batch(mode, written), *batch)
                }
                None => match (written.pop(), last_batch) {
                    (Some(write), Some(batch)) => (Unit::Single(write), batch),
                    _ => continue,
                },
            };
            unsent
                .entry(writer)
                .or_default()
                .push((step, unit.0, unit.1));
            if rng.below(4) == 0 {
                world.sync();
                let users: &[&'static str] = if alice_offline { &[BOB] } else { &[ME, BOB] };
                flush(&mut model, &mut unsent, &mut fates, users, false);
            }
        }
        if alice_offline {
            world.connect();
        } else {
            world.sync();
        }
        flush(&mut model, &mut unsent, &mut fates, &[ME], alice_offline);
        flush(&mut model, &mut unsent, &mut fates, &[BOB], false);
        world.sync();

        let mode = if alice_offline {
            "alice offline"
        } else {
            "online"
        };
        for (step, writer, unit, batch, admitted) in &fates {
            let expected = if *admitted { "accepted" } else { "rejected" };
            assert_eq!(
                world.fate_of(writer, *batch),
                expected,
                "seed {seed} ({mode}) step {step}, {writer}: {unit:?} — {:?}",
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
        assert!(
            world.missing.is_empty(),
            "seed {seed} ({mode}): {} Missing answers",
            world.missing.len()
        );
    }
}

// ---- A parent and its child in one batch.
//
// An approved row of an explicit batch stays out of view until the batch settles, so a
// child's `exists` cannot see a parent written in the same batch; and a batch's rows travel
// in id order, so the child may reach the server first.

/// Two fresh ids `(parent, child)`, the child's sorting first: a batch's rows travel in id
/// order, so the child reaches the server before its parent.
fn ids_child_first() -> (ObjectId, ObjectId) {
    let (a, b) = (ObjectId::new(), ObjectId::new());
    if a > b { (a, b) } else { (b, a) }
}

/// `n` fresh ids, largest first: written in this order, each row sorts before the rows
/// written before it, so every child travels ahead of its parents.
fn ids_descending(n: usize) -> Vec<ObjectId> {
    let mut ids: Vec<ObjectId> = (0..n).map(|_| ObjectId::new()).collect();
    ids.sort();
    ids.reverse();
    ids
}

fn label(text: String) -> &'static str {
    Box::leak(text.into_boxed_str())
}

fn in_batch(batch: BatchId, mode: BatchMode) -> WriteContext {
    World::mine().with_batch_mode(mode).with_batch_id(batch)
}

impl World {
    /// The server stops and starts again over its store: whatever it held only in memory
    /// is gone, and the client connects again and retransmits what is unsettled.
    fn restart_server(&mut self) {
        let stopped = std::mem::replace(&mut self.server, server_runtime_at(None));
        self.server = server_runtime_for(
            stopped.into_storage(),
            Some(crate::sync_manager::DurabilityTier::GlobalServer),
            self.rules,
        );
        self.server
            .add_client(self.client_id, Some(Session::new(ME)));
        self.client.remove_server(self.server_id);
        self.client.add_server(self.server_id);
        self.sync();
    }

    /// The connection drops and comes back: the server forgets the client, the client
    /// retransmits what is unsettled.
    fn reconnect(&mut self) {
        assert!(
            self.server.remove_client(self.client_id),
            "the server lets the client go"
        );
        self.server
            .add_client(self.client_id, Some(Session::new(ME)));
        self.client.remove_server(self.server_id);
        self.client.add_server(self.server_id);
        self.sync();
    }

    /// Of what the client has to send, only row `row` reaches the server, which takes it
    /// in; the rest is lost on the way.
    fn deliver_only(&mut self, row: ObjectId) {
        self.client.batched_tick();
        for entry in self.client.sync_sender().take() {
            if let SyncPayload::RowBatchCreated { row: sent, .. } = &entry.payload
                && sent.row_id == row
            {
                self.server.park_sync_message(InboxEntry {
                    source: Source::Client(self.client_id),
                    payload: entry.payload,
                });
            }
        }
        self.server.batched_tick();
        self.server.immediate_tick();
        self.server.batched_tick();
        self.server.sync_sender().take();
    }

    /// The lost rows reach the server after all, each as last sent, and nothing else is
    /// sent again.
    fn deliver_held_back(&mut self) {
        self.lost_rows.clear();
        for (_, payload) in std::mem::take(&mut self.held_back) {
            self.sent.push(payload.clone());
            self.server.park_sync_message(InboxEntry {
                source: Source::Client(self.client_id),
                payload,
            });
        }
        self.sync();
    }

    /// Whether a row of `batch` reached the server before `commit_batch`.
    fn sent_rows_of(&self, batch: BatchId) -> usize {
        self.sent
            .iter()
            .filter(|payload| {
                matches!(payload, SyncPayload::RowBatchCreated { row, .. } if row.batch_id == batch)
            })
            .count()
    }
}

#[derive(Clone, Copy, Debug)]
enum Pair {
    MembershipAndMessage,
    MessageAndAttachment,
}

/// How a batch's rows reach the server.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Delivery {
    /// Online, the rows and the seal in one exchange.
    Together,
    /// Online, each row sent as it is written, the seal at commit.
    Streamed,
    /// Written offline, sent on reconnect.
    Offline,
}

/// The pair in one explicit batch, its other links in batches of their own before it;
/// sent as written, streamed, or offline and on reconnect.
fn one_batch_pair(mode: BatchMode, pair: Pair) {
    for child_first in [true, false] {
        for delivery in [Delivery::Together, Delivery::Streamed, Delivery::Offline] {
            let connected = delivery != Delivery::Offline;
            let streamed = delivery == Delivery::Streamed;
            let case = format!(
                "{pair:?}, {mode:?}, {}, {delivery:?}",
                if child_first {
                    "child first"
                } else {
                    "parent first"
                },
            );
            let mut world = World::new(connected);
            let (chat, chat_batch) = world.chat();
            let mut batches = vec![("chat", chat_batch)];
            let mut rows = vec![("chat", "chats", chat)];
            if let Pair::MessageAndAttachment = pair {
                let (member, member_batch) = world.membership(chat);
                batches.push(("membership", member_batch));
                rows.push(("membership", "chat_members", member));
            }
            let (parent, child) = match ids_child_first() {
                ids if child_first => ids,
                (later, earlier) => (earlier, later),
            };
            let batch = world.client.begin_batch(mode);
            let (parent_table, child_table) = match pair {
                Pair::MembershipAndMessage => {
                    world.insert_in(
                        "chat_members",
                        vec![
                            ("chat_id", Value::Uuid(chat)),
                            ("user_id", text(ME)),
                            ("is_banned", Value::Boolean(false)),
                        ],
                        Some(parent),
                        in_batch(batch, mode),
                    );
                    if streamed {
                        world.sync();
                    }
                    world.insert_in(
                        "messages",
                        vec![
                            ("chat_id", Value::Uuid(chat)),
                            ("sender_id", text(ME)),
                            ("body", text("hello")),
                        ],
                        Some(child),
                        in_batch(batch, mode),
                    );
                    if streamed {
                        world.sync();
                    }
                    ("chat_members", "messages")
                }
                Pair::MessageAndAttachment => {
                    world.insert_in(
                        "messages",
                        vec![
                            ("chat_id", Value::Uuid(chat)),
                            ("sender_id", text(ME)),
                            ("body", text("hello")),
                        ],
                        Some(parent),
                        in_batch(batch, mode),
                    );
                    if streamed {
                        world.sync();
                    }
                    world.insert_in(
                        "message_attachments",
                        vec![("message_id", Value::Uuid(parent)), ("kind", text("photo"))],
                        Some(child),
                        in_batch(batch, mode),
                    );
                    if streamed {
                        world.sync();
                    }
                    ("messages", "message_attachments")
                }
            };
            world.client.commit_batch(batch).expect("the batch commits");
            if connected {
                world.sync();
            } else {
                world.connect();
            }
            batches.push((label(format!("the one batch ({case})")), batch));
            world.assert_all_accepted(&batches);
            rows.push((label(format!("parent ({case})")), parent_table, parent));
            rows.push((label(format!("child ({case})")), child_table, child));
            world.assert_served(&rows);
            if delivery == Delivery::Offline && mode == BatchMode::Direct {
                // A direct batch is published at commit, so its resend carries it visible:
                // the server must still let the child wait for its parent.
                assert!(
                    world.sent.iter().all(|payload| !matches!(
                        payload,
                        SyncPayload::RowBatchCreated { row, .. }
                            if row.batch_id == batch && row.state != RowState::VisibleDirect
                    )),
                    "{case}: the resent rows are visible"
                );
            }
            let asked_again = world
                .missing
                .iter()
                .map(|missing| {
                    batches
                        .iter()
                        .find(|(_, batch)| batch == missing)
                        .map_or("another batch", |(label, _)| *label)
                })
                .collect::<Vec<_>>();
            assert!(
                asked_again.is_empty(),
                "{case}: rows that are all here are never asked for again: {asked_again:?}"
            );
        }
    }
}

#[test]
fn a_membership_and_its_message_in_one_direct_batch_are_accepted() {
    one_batch_pair(BatchMode::Direct, Pair::MembershipAndMessage);
}

#[test]
fn a_membership_and_its_message_in_one_transactional_batch_are_accepted() {
    one_batch_pair(BatchMode::Transactional, Pair::MembershipAndMessage);
}

#[test]
fn a_message_and_its_attachment_in_one_direct_batch_are_accepted() {
    one_batch_pair(BatchMode::Direct, Pair::MessageAndAttachment);
}

#[test]
fn a_message_and_its_attachment_in_one_transactional_batch_are_accepted() {
    one_batch_pair(BatchMode::Transactional, Pair::MessageAndAttachment);
}

/// A whole Lin session in one batch: four levels, each link hanging on one before it, every
/// child travelling ahead of its parents.
#[test]
fn the_lin_session_in_one_batch_is_accepted() {
    for mode in [BatchMode::Direct, BatchMode::Transactional] {
        for connected in [true, false] {
            let case = format!("{mode:?}, {}", if connected { "online" } else { "offline" });
            let mut world = World::new(connected);
            let (agent, agent_batch) = world.insert("agents", vec![("owner_id", text(ME))]);
            let ids = ids_descending(6);
            let (chat, member, chat_agent, summary, message, attachment) =
                (ids[0], ids[1], ids[2], ids[3], ids[4], ids[5]);
            let batch = world.client.begin_batch(mode);
            let rows: [ChainRow; 6] = [
                (
                    "chat",
                    "chats",
                    chat,
                    vec![("created_by", text(ME)), ("kind", text("assistant"))],
                ),
                (
                    "membership",
                    "chat_members",
                    member,
                    vec![
                        ("chat_id", Value::Uuid(chat)),
                        ("user_id", text(ME)),
                        ("is_banned", Value::Boolean(false)),
                    ],
                ),
                (
                    "chat agent",
                    "chat_agents",
                    chat_agent,
                    vec![
                        ("chat_id", Value::Uuid(chat)),
                        ("agent_id", Value::Uuid(agent)),
                    ],
                ),
                (
                    "summary",
                    "chat_summaries",
                    summary,
                    vec![("chat_id", Value::Uuid(chat)), ("body", text(""))],
                ),
                (
                    "message",
                    "messages",
                    message,
                    vec![
                        ("chat_id", Value::Uuid(chat)),
                        ("sender_id", text(ME)),
                        ("body", text("hello")),
                    ],
                ),
                (
                    "attachment",
                    "message_attachments",
                    attachment,
                    vec![
                        ("message_id", Value::Uuid(message)),
                        ("kind", text("photo")),
                    ],
                ),
            ];
            let mut served = vec![("agent", "agents", agent)];
            for (name, table, id, values) in rows {
                world.insert_in(table, values, Some(id), in_batch(batch, mode));
                served.push((label(format!("{name} ({case})")), table, id));
            }
            world.client.commit_batch(batch).expect("the batch commits");
            if connected {
                world.sync();
            } else {
                world.connect();
            }
            world.assert_all_accepted(&[
                ("agent", agent_batch),
                (label(format!("the session ({case})")), batch),
            ]);
            world.assert_served(&served);
        }
    }
}

/// A parent its own rule refuses takes its batch down: no sibling vouches for a row the
/// rules deny.
#[test]
fn a_parent_its_rule_refuses_refuses_its_batch() {
    for mode in [BatchMode::Direct, BatchMode::Transactional] {
        let mut world = World::new(true);
        let (chat, chat_batch) = world.chat();
        let (_, member_batch) = world.membership(chat);
        world.sync();
        let (message, attachment) = ids_child_first();
        let batch = world.client.begin_batch(mode);
        // Alice may not write a message that names bob as its sender.
        world.insert_in(
            "messages",
            vec![
                ("chat_id", Value::Uuid(chat)),
                ("sender_id", text(BOB)),
                ("body", text("hello")),
            ],
            Some(message),
            in_batch(batch, mode),
        );
        world.insert_in(
            "message_attachments",
            vec![
                ("message_id", Value::Uuid(message)),
                ("kind", text("photo")),
            ],
            Some(attachment),
            in_batch(batch, mode),
        );
        world.client.commit_batch(batch).expect("the batch commits");
        world.sync();
        let report = world.report(&[("the batch", batch)]);
        assert_eq!(world.fate(batch), "rejected", "{mode:?}: {report:#?}");
        world.assert_all_accepted(&[("chat", chat_batch), ("membership", member_batch)]);
        assert!(
            world.served("messages").is_empty(),
            "{mode:?}: no message served"
        );
        assert!(
            world.served("message_attachments").is_empty(),
            "{mode:?}: no attachment served"
        );
    }
}

/// A batch whose child waits for its parent is decided before the writes the client made
/// after it, which may hang on its rows.
#[test]
fn a_waiting_batch_is_decided_before_the_writes_after_it() {
    for refused in [false, true] {
        let mut world = World::new(false);
        let (chat, chat_batch) = world.chat();
        let (_, member_batch) = world.membership(chat);
        let (message, attachment) = ids_child_first();
        let batch = world.client.begin_batch(BatchMode::Direct);
        world.insert_in(
            "messages",
            vec![
                ("chat_id", Value::Uuid(chat)),
                ("sender_id", text(if refused { BOB } else { ME })),
                ("body", text("hello")),
            ],
            Some(message),
            in_batch(batch, BatchMode::Direct),
        );
        world.insert_in(
            "message_attachments",
            vec![
                ("message_id", Value::Uuid(message)),
                ("kind", text("photo")),
            ],
            Some(attachment),
            in_batch(batch, BatchMode::Direct),
        );
        world.client.commit_batch(batch).expect("the batch commits");
        let (later, later_batch) = world.attachment(message);
        world.connect();
        world.assert_all_accepted(&[("chat", chat_batch), ("membership", member_batch)]);
        let batches = [
            ("the batch", batch),
            ("the attachment after it", later_batch),
        ];
        let report = world.report(&batches);
        let expected = if refused { "rejected" } else { "accepted" };
        assert!(
            batches
                .iter()
                .all(|(_, batch)| world.fate(*batch) == expected),
            "refused parent: {refused}; every write {expected}: {report:#?}"
        );
        let attachments = world.served("message_attachments");
        assert_eq!(
            [attachment, later].map(|id| attachments.contains(&id)),
            [!refused; 2],
            "refused parent: {refused}"
        );
    }
}

/// While a batch is open on the client, its writer's other writes and everyone else's go
/// through; the batch is served once it is committed.
#[test]
fn an_open_batch_holds_back_nobody() {
    let mut world = World::new(true);
    let bob = world.add_peer(BOB);
    let (chat, _) = world.chat();
    world.membership(chat);
    world.sync();
    let (message, attachment) = ids_child_first();
    let batch = world.client.begin_batch(BatchMode::Direct);
    world.insert_in(
        "messages",
        vec![
            ("chat_id", Value::Uuid(chat)),
            ("sender_id", text(ME)),
            ("body", text("hello")),
        ],
        Some(message),
        in_batch(batch, BatchMode::Direct),
    );
    world.insert_in(
        "message_attachments",
        vec![
            ("message_id", Value::Uuid(message)),
            ("kind", text("photo")),
        ],
        Some(attachment),
        in_batch(batch, BatchMode::Direct),
    );
    world.sync();
    let sent_before_commit = world.sent_rows_of(batch);

    let (_, alices_other) = world.chat();
    let bob_context = WriteContext::from_session(Session::new(BOB));
    let (_, bobs) = world.peers[bob]
        .0
        .insert_with_id(
            "chats",
            HashMap::from([
                ("created_by".to_string(), text(BOB)),
                ("kind".to_string(), text("assistant")),
            ]),
            None,
            Some(&bob_context),
        )
        .expect("bob's chat applies locally");
    world.sync();
    assert_eq!(
        world.fate(alices_other),
        "accepted",
        "alice writes on while her batch is open ({sent_before_commit} of its rows sent)"
    );
    assert_eq!(world.fate_of(BOB, bobs), "accepted", "bob is not held back");
    assert!(
        !world.served("messages").contains(&message),
        "nothing of an open batch is served"
    );

    world.client.commit_batch(batch).expect("the batch commits");
    world.sync();
    world.assert_all_accepted(&[("the batch", batch)]);
    world.assert_served(&[
        ("message", "messages", message),
        ("attachment", "message_attachments", attachment),
    ]);
}

/// The server restarts while a child waits for its parent: what it held in memory is gone,
/// and the client's retransmit decides the batch.
#[test]
fn a_waiting_child_survives_a_server_restart() {
    let mut world = World::new(true);
    let (chat, _) = world.chat();
    world.membership(chat);
    world.sync();
    let (message, attachment) = ids_child_first();
    let batch = world.client.begin_batch(BatchMode::Direct);
    world.insert_in(
        "messages",
        vec![
            ("chat_id", Value::Uuid(chat)),
            ("sender_id", text(ME)),
            ("body", text("hello")),
        ],
        Some(message),
        in_batch(batch, BatchMode::Direct),
    );
    world.insert_in(
        "message_attachments",
        vec![
            ("message_id", Value::Uuid(message)),
            ("kind", text("photo")),
        ],
        Some(attachment),
        in_batch(batch, BatchMode::Direct),
    );
    world.client.commit_batch(batch).expect("the batch commits");
    world.deliver_only(attachment);
    world.restart_server();
    world.assert_all_accepted(&[("the batch", batch)]);
    world.assert_served(&[
        ("message", "messages", message),
        ("attachment", "message_attachments", attachment),
    ]);
}

/// The connection drops while a child waits for its parent; the reconnect decides the batch.
#[test]
fn a_waiting_child_survives_a_reconnect() {
    let mut world = World::new(true);
    let (chat, _) = world.chat();
    world.membership(chat);
    world.sync();
    let (message, attachment) = ids_child_first();
    let batch = world.client.begin_batch(BatchMode::Transactional);
    world.insert_in(
        "messages",
        vec![
            ("chat_id", Value::Uuid(chat)),
            ("sender_id", text(ME)),
            ("body", text("hello")),
        ],
        Some(message),
        in_batch(batch, BatchMode::Transactional),
    );
    world.insert_in(
        "message_attachments",
        vec![
            ("message_id", Value::Uuid(message)),
            ("kind", text("photo")),
        ],
        Some(attachment),
        in_batch(batch, BatchMode::Transactional),
    );
    world.client.commit_batch(batch).expect("the batch commits");
    world.deliver_only(attachment);
    world.reconnect();
    world.assert_all_accepted(&[("the batch", batch)]);
    world.assert_served(&[
        ("message", "messages", message),
        ("attachment", "message_attachments", attachment),
    ]);
}

// ---- Siblings that vouch for each other, and an update that leans on a sibling.

/// A `b` may stand alone as a root, or hang on an `a`; an `a` always hangs on a `b`; a `c`
/// may point at a `b` only.
fn sibling_schema() -> Schema {
    SchemaBuilder::new()
        .table(TableSchema::builder("a").column("peer", ColumnType::Uuid))
        .table(
            TableSchema::builder("b")
                .column("kind", ColumnType::Text)
                .column("peer", ColumnType::Uuid),
        )
        .table(TableSchema::builder("c").column("peer", ColumnType::Uuid))
        .table(TableSchema::builder("d").column("peer", ColumnType::Uuid))
        .table(TableSchema::builder("e").column("peer", ColumnType::Uuid))
        .build()
}

fn sibling_permissions() -> Schema {
    let points_at_a_b = exists("b", vec![cmp("id", outer("peer"))]);
    let mut head = sibling_schema();
    for (name, table) in head.iter_mut() {
        table.policies = match name.as_str() {
            "a" => policies(points_at_a_b.clone()),
            "b" => policies(PolicyExpr::or(vec![
                cmp("kind", PolicyValue::Literal(text("root"))),
                exists("a", vec![cmp("id", outer("peer"))]),
            ])),
            "c" => TablePolicies::new()
                .with_select(PolicyExpr::True)
                .with_insert(PolicyExpr::True)
                .with_update(Some(PolicyExpr::True), points_at_a_b.clone())
                .with_delete(PolicyExpr::True),
            // A `d` points at a `b`, while any `c` exists: a read nested in the `exists`.
            "d" => policies(exists(
                "b",
                vec![
                    cmp("id", outer("peer")),
                    exists("c", vec![PolicyExpr::True]),
                ],
            )),
            // An `e` may be changed while it points at a `b`, to anything no `d` points at
            // while a `c` exists: its WITH CHECK reads where the batch cannot show its rows.
            "e" => TablePolicies::new()
                .with_select(PolicyExpr::True)
                .with_insert(PolicyExpr::True)
                .with_update(
                    Some(points_at_a_b.clone()),
                    PolicyExpr::Not(Box::new(exists(
                        "d",
                        vec![
                            cmp("id", outer("peer")),
                            exists("c", vec![PolicyExpr::True]),
                        ],
                    ))),
                )
                .with_delete(PolicyExpr::True),
            other => panic!("no rule for {other}"),
        };
    }
    head
}

const SIBLINGS: Rules = Rules {
    structural: sibling_schema,
    permissions: sibling_permissions,
};

/// Two rows of one batch that each pass only if the other is there do not let each other
/// in: a batch is admitted row by row from what is already allowed, never by a loop.
#[test]
fn rows_that_only_vouch_for_each_other_are_refused() {
    for mode in [BatchMode::Direct, BatchMode::Transactional] {
        for a_first in [true, false] {
            let case = format!("{mode:?}, {}", if a_first { "a first" } else { "b first" });
            let mut world = World::with_rules(true, SIBLINGS);
            let (first, second) = ids_child_first();
            let (a, b) = if a_first {
                (second, first)
            } else {
                (first, second)
            };
            let batch = world.client.begin_batch(mode);
            world.insert_in(
                "a",
                vec![("peer", Value::Uuid(b))],
                Some(a),
                in_batch(batch, mode),
            );
            world.insert_in(
                "b",
                vec![("kind", text("leaf")), ("peer", Value::Uuid(a))],
                Some(b),
                in_batch(batch, mode),
            );
            world.client.commit_batch(batch).expect("the batch commits");
            world.sync();
            let report = world.report(&[("the loop", batch)]);
            assert_eq!(world.fate(batch), "rejected", "{case}: {report:#?}");
            assert!(world.served("a").is_empty(), "{case}: no a served");
            assert!(world.served("b").is_empty(), "{case}: no b served");

            // Control: a root `b` and the `a` on it, in one batch.
            let (root, leaf) = ids_child_first();
            let control = world.client.begin_batch(mode);
            world.insert_in(
                "b",
                vec![
                    ("kind", text("root")),
                    ("peer", Value::Uuid(ObjectId::new())),
                ],
                Some(root),
                in_batch(control, mode),
            );
            world.insert_in(
                "a",
                vec![("peer", Value::Uuid(root))],
                Some(leaf),
                in_batch(control, mode),
            );
            world
                .client
                .commit_batch(control)
                .expect("the batch commits");
            world.sync();
            world.assert_all_accepted(&[(label(format!("control ({case})")), control)]);
            world.assert_served(&[("root", "b", root), ("leaf", "a", leaf)]);
        }
    }
}

/// An update's WITH CHECK reads a row written in the same batch.
#[test]
fn an_update_that_leans_on_a_row_of_its_batch_is_accepted() {
    for mode in [BatchMode::Direct, BatchMode::Transactional] {
        let mut world = World::with_rules(true, SIBLINGS);
        let (c, c_batch) = world.insert("c", vec![("peer", Value::Uuid(ObjectId::new()))]);
        world.sync();
        world.assert_all_accepted(&[("c", c_batch)]);
        let root = ObjectId::new();
        let batch = world.client.begin_batch(mode);
        world.insert_in(
            "b",
            vec![
                ("kind", text("root")),
                ("peer", Value::Uuid(ObjectId::new())),
            ],
            Some(root),
            in_batch(batch, mode),
        );
        world
            .client
            .update(
                c,
                vec![("peer".to_string(), Value::Uuid(root))],
                Some(&in_batch(batch, mode)),
            )
            .expect("the update applies locally");
        world.client.commit_batch(batch).expect("the batch commits");
        world.sync();
        world.assert_all_accepted(&[(label(format!("the batch ({mode:?})")), batch)]);
        assert_eq!(
            world.served_row("c", c).map(|values| values[0].clone()),
            Some(Value::Uuid(root)),
            "{mode:?}: c points at the root written with it"
        );
    }
}

// ---- How long a batch waits, and whom it holds.

impl World {
    fn set_parking(
        &mut self,
        change: impl FnOnce(&mut crate::sync_manager::parked_checks::ParkingBounds),
    ) {
        let sync_manager = self
            .server
            .schema_manager
            .query_manager_mut()
            .sync_manager_mut();
        let mut bounds = sync_manager.parking_bounds();
        change(&mut bounds);
        sync_manager.set_parking_bounds(bounds);
    }

    fn server_parks(&mut self, batch: BatchId) -> bool {
        self.server
            .schema_manager
            .query_manager_mut()
            .sync_manager_mut()
            .has_parked_batch(batch)
    }

    /// A message and its attachment in one direct batch, the attachment first in id order.
    fn message_with_attachment(&mut self, chat: ObjectId) -> (BatchId, ObjectId, ObjectId) {
        let (message, attachment) = ids_child_first();
        let batch = self.client.begin_batch(BatchMode::Direct);
        self.insert_in(
            "messages",
            vec![
                ("chat_id", Value::Uuid(chat)),
                ("sender_id", text(ME)),
                ("body", text("hello")),
            ],
            Some(message),
            in_batch(batch, BatchMode::Direct),
        );
        self.insert_in(
            "message_attachments",
            vec![
                ("message_id", Value::Uuid(message)),
                ("kind", text("photo")),
            ],
            Some(attachment),
            in_batch(batch, BatchMode::Direct),
        );
        (batch, message, attachment)
    }
}

/// A sealed batch whose parent has not arrived holds its writer's later writes until its
/// deadline. Past it her writes go on, and the batch waits on undecided, never rejected,
/// with what it parked: the parent may be on its way (a resend after a restart can take
/// longer than any deadline), and its arrival alone decides the batch.
#[test]
fn a_sealed_batch_missing_a_row_lets_its_writer_go_at_its_deadline_and_waits_on() {
    let mut world = World::new(true);
    let (chat, _) = world.chat();
    world.membership(chat);
    world.sync();
    let (batch, message, attachment) = world.message_with_attachment(chat);
    world.lost_rows.insert(message);
    world.client.commit_batch(batch).expect("the batch commits");
    world.sync();
    assert!(
        world.server_parks(batch),
        "the attachment waits for its message"
    );
    let (_, later) = world.chat();
    world.sync();
    assert!(
        !matches!(world.fate(later).as_str(), "accepted" | "rejected"),
        "a write after the batch waits for it: {}",
        world.fate(later)
    );

    world.set_parking(|bounds| bounds.sealed_deadline_micros = 0);
    world.sync();
    assert_eq!(
        world.fate(later),
        "accepted",
        "the deadline releases the writer"
    );
    assert!(
        !matches!(world.fate(batch).as_str(), "accepted" | "rejected"),
        "the batch is undecided: {}",
        world.fate(batch)
    );
    assert!(
        world.server_parks(batch),
        "the batch keeps the attachment it parked"
    );
    assert!(!world.served("message_attachments").contains(&attachment));

    let asked = world.missing.len();
    world.deliver_held_back();
    world.assert_all_accepted(&[("the batch", batch)]);
    world.assert_served(&[
        ("message", "messages", message),
        ("attachment", "message_attachments", attachment),
    ]);
    assert_eq!(
        world.missing.len(),
        asked,
        "the late parent completes the batch without asking for it again"
    );
}

/// A sealed batch whose rows stop coming is let go, and its writer is asked for it again:
/// nothing else would ever complete it.
#[test]
fn a_sealed_batch_whose_rows_stop_coming_is_asked_for_again() {
    let mut world = World::new(true);
    let (chat, _) = world.chat();
    world.membership(chat);
    world.sync();
    let (batch, message, attachment) = world.message_with_attachment(chat);
    world.lost_rows.insert(message);
    world.client.commit_batch(batch).expect("the batch commits");
    world.sync();
    assert!(world.server_parks(batch));

    let asked = world
        .missing
        .iter()
        .filter(|asked| **asked == batch)
        .count();
    world.lost_rows.clear();
    world.held_back.clear();
    // The rows stop for longer than `Missing` is rate-limited for.
    world
        .server
        .schema_manager
        .query_manager_mut()
        .sync_manager_mut()
        .forget_missing_answers();
    world.set_parking(|bounds| bounds.idle_ttl_micros = 0);
    world.sync();
    assert!(
        world
            .missing
            .iter()
            .filter(|asked| **asked == batch)
            .count()
            > asked,
        "the writer is asked for the batch again"
    );
    world.set_parking(|bounds| *bounds = Default::default());
    world.sync();
    world.assert_all_accepted(&[("the batch", batch)]);
    world.assert_served(&[
        ("message", "messages", message),
        ("attachment", "message_attachments", attachment),
    ]);
}

/// A batch rolled back while its child waits holds nothing back, and its rows are let go.
#[test]
fn a_rolled_back_batch_holds_back_nothing() {
    let mut world = World::new(true);
    let (chat, _) = world.chat();
    world.membership(chat);
    world.sync();
    let (message, attachment) = ids_child_first();
    let batch = world.client.begin_batch(BatchMode::Direct);
    world.insert_in(
        "messages",
        vec![
            ("chat_id", Value::Uuid(chat)),
            ("sender_id", text(ME)),
            ("body", text("hello")),
        ],
        Some(message),
        in_batch(batch, BatchMode::Direct),
    );
    world.sync();
    world.insert_in(
        "message_attachments",
        vec![
            ("message_id", Value::Uuid(message)),
            ("kind", text("photo")),
        ],
        Some(attachment),
        in_batch(batch, BatchMode::Direct),
    );
    world.sync();
    assert!(
        world.server_parks(batch),
        "the attachment waits while the batch is open"
    );
    world
        .client
        .rollback_batch(batch)
        .expect("the batch rolls back");
    let (_, later) = world.chat();
    world.sync();
    assert_eq!(
        world.fate(later),
        "accepted",
        "nothing waits for a batch never sealed"
    );

    world.set_parking(|bounds| bounds.idle_ttl_micros = 0);
    world.sync();
    assert!(
        !world.server_parks(batch),
        "the rows of a batch never sealed are let go"
    );
    assert!(world.served("message_attachments").is_empty());
    assert!(!world.served("messages").contains(&message));
}

/// A waiting batch holds only its own connection: the same user's other device writes on.
#[test]
fn a_waiting_batch_holds_back_only_its_own_connection() {
    let mut world = World::new(true);
    let other_device = world.add_peer(ME);
    let (chat, _) = world.chat();
    world.membership(chat);
    world.sync();
    let (batch, message, _) = world.message_with_attachment(chat);
    world.lost_rows.insert(message);
    world.client.commit_batch(batch).expect("the batch commits");
    world.sync();
    assert!(world.server_parks(batch));

    let (_, on_the_other_device) = world.peers[other_device]
        .0
        .insert_with_id(
            "chats",
            HashMap::from([
                ("created_by".to_string(), text(ME)),
                ("kind".to_string(), text("assistant")),
            ]),
            None,
            Some(&World::mine()),
        )
        .expect("the chat applies locally");
    world.sync();
    assert!(
        matches!(
            world.peers[other_device].0.batch_fate(on_the_other_device),
            Ok(Some(BatchFate::DurableDirect { .. }))
        ),
        "the other device is not held back"
    );
}

/// A child whose rule reads a sibling through a nested `exists` cannot wait: the nested
/// read does not see the batch, so it is refused as before, whatever its siblings.
#[test]
fn a_rule_the_batch_cannot_show_its_rows_to_is_refused_as_before() {
    let mut world = World::with_rules(true, SIBLINGS);
    let (_, c_batch) = world.insert("c", vec![("peer", Value::Uuid(ObjectId::new()))]);
    world.sync();
    world.assert_all_accepted(&[("c", c_batch)]);

    let (d, root) = ids_child_first();
    let batch = world.client.begin_batch(BatchMode::Direct);
    world.insert_in(
        "b",
        vec![
            ("kind", text("root")),
            ("peer", Value::Uuid(ObjectId::new())),
        ],
        Some(root),
        in_batch(batch, BatchMode::Direct),
    );
    world.insert_in(
        "d",
        vec![("peer", Value::Uuid(root))],
        Some(d),
        in_batch(batch, BatchMode::Direct),
    );
    world.client.commit_batch(batch).expect("the batch commits");
    world.sync();
    assert_eq!(
        world.fate(batch),
        "rejected",
        "{:#?}",
        world.report(&[("the batch", batch)])
    );

    // Control: the same rows one batch each.
    let (root, root_batch) = world.insert(
        "b",
        vec![
            ("kind", text("root")),
            ("peer", Value::Uuid(ObjectId::new())),
        ],
    );
    world.sync();
    let (_, d_batch) = world.insert("d", vec![("peer", Value::Uuid(root))]);
    world.sync();
    world.assert_all_accepted(&[("root", root_batch), ("d", d_batch)]);
}

/// An update whose USING leans on a row of its batch cannot wait when its WITH CHECK reads
/// where the batch cannot show its rows: judged again, it would be judged in two stores at
/// once. It is refused as before.
#[test]
fn an_update_whose_other_rule_cannot_see_the_batch_is_refused_as_before() {
    let mut world = World::with_rules(true, SIBLINGS);
    let root = ObjectId::new();
    let (e, e_batch) = world.insert("e", vec![("peer", Value::Uuid(root))]);
    world.sync();
    world.assert_all_accepted(&[("e", e_batch)]);

    let batch = world.client.begin_batch(BatchMode::Direct);
    world.insert_in(
        "b",
        vec![
            ("kind", text("root")),
            ("peer", Value::Uuid(ObjectId::new())),
        ],
        Some(root),
        in_batch(batch, BatchMode::Direct),
    );
    world
        .client
        .update(
            e,
            vec![("peer".to_string(), Value::Uuid(ObjectId::new()))],
            Some(&in_batch(batch, BatchMode::Direct)),
        )
        .expect("the update applies locally");
    world.client.commit_batch(batch).expect("the batch commits");
    world.sync();
    assert_eq!(
        world.fate(batch),
        "rejected",
        "{:#?}",
        world.report(&[("the batch", batch)])
    );

    // Control: the same rows one batch each.
    let other_root = ObjectId::new();
    let (other, other_batch) = world.insert("e", vec![("peer", Value::Uuid(other_root))]);
    world.sync();
    let root_batch = world.client.begin_batch(BatchMode::Direct);
    world.insert_in(
        "b",
        vec![
            ("kind", text("root")),
            ("peer", Value::Uuid(ObjectId::new())),
        ],
        Some(other_root),
        in_batch(root_batch, BatchMode::Direct),
    );
    world
        .client
        .commit_batch(root_batch)
        .expect("the batch commits");
    world.sync();
    let update_batch = world.client.begin_batch(BatchMode::Direct);
    world
        .client
        .update(
            other,
            vec![("peer".to_string(), Value::Uuid(ObjectId::new()))],
            Some(&in_batch(update_batch, BatchMode::Direct)),
        )
        .expect("the update applies locally");
    world
        .client
        .commit_batch(update_batch)
        .expect("the batch commits");
    world.sync();
    world.assert_all_accepted(&[
        ("other e", other_batch),
        ("root", root_batch),
        ("update", update_batch),
    ]);
}

/// A seal sent again without the row a sibling leaned on does not let the sibling in: only
/// the rows the seal declares are the batch, and a staged row it drops is never served, so
/// it is no parent. A direct batch's rows sent again after its commit are visible on their
/// own, declared or not, and need no seal to be a parent.
#[test]
fn a_seal_that_drops_the_parent_does_not_let_the_child_in() {
    for mode in [BatchMode::Direct, BatchMode::Transactional] {
        let mut world = World::new(true);
        let (chat, _) = world.chat();
        world.membership(chat);
        world.sync();
        let (message, attachment) = ids_child_first();
        let lost = ObjectId::new();
        let batch = world.client.begin_batch(mode);
        world.insert_in(
            "messages",
            vec![
                ("chat_id", Value::Uuid(chat)),
                ("sender_id", text(ME)),
                ("body", text("hello")),
            ],
            Some(message),
            in_batch(batch, mode),
        );
        world.insert_in(
            "message_attachments",
            vec![
                ("message_id", Value::Uuid(message)),
                ("kind", text("photo")),
            ],
            Some(attachment),
            in_batch(batch, mode),
        );
        world.insert_in(
            "messages",
            vec![
                ("chat_id", Value::Uuid(chat)),
                ("sender_id", text(ME)),
                ("body", text("on its way")),
            ],
            Some(lost),
            in_batch(batch, mode),
        );
        world.lost_rows.insert(lost);
        // The rows go while the batch is open, staged.
        world.sync();
        world.client.commit_batch(batch).expect("the batch commits");
        world.sync();
        assert!(
            world.server_parks(batch),
            "{mode:?}: the attachment waits for the batch to be complete"
        );

        let mut seal = world
            .sent
            .iter()
            .find_map(|payload| match payload {
                SyncPayload::SealBatch { submission } if submission.batch_id == batch => {
                    Some(submission.clone())
                }
                _ => None,
            })
            .expect("the client sealed the batch");
        seal.members.retain(|member| member.object_id == attachment);
        seal.batch_digest = SealedBatchSubmission::compute_batch_digest(&seal.members);
        world.server.park_sync_message(InboxEntry {
            source: Source::Client(world.client_id),
            payload: SyncPayload::SealBatch { submission: seal },
        });
        world.sync();
        let attachment_served = world.served("message_attachments").contains(&attachment);
        let message_served = world.served("messages").contains(&message);
        match mode {
            BatchMode::Transactional => {
                assert_eq!(
                    world.fate(batch),
                    "rejected",
                    "{:#?}",
                    world.report(&[("the batch", batch)])
                );
                assert!(!attachment_served, "the attachment is not served");
                assert!(!message_served, "the message is not served");
            }
            // The `Missing` for the lost row brought the message back visible, as a direct
            // batch's rows are once committed: it stands on its own, and the attachment
            // may stand on it.
            BatchMode::Direct => assert!(
                !attachment_served || message_served,
                "no attachment without its message"
            ),
        }
    }
}

/// A row of another table at the parent's id does not stand in for the parent: the batch
/// shows each of its rows in its own table only.
#[test]
fn a_row_of_another_table_at_the_parents_id_is_no_parent() {
    for mode in [BatchMode::Direct, BatchMode::Transactional] {
        let mut world = World::with_rules(false, SIBLINGS);
        let (leaf, impostor) = ids_child_first();
        let batch = world.client.begin_batch(mode);
        world.insert_in(
            "c",
            vec![("peer", Value::Uuid(ObjectId::new()))],
            Some(impostor),
            in_batch(batch, mode),
        );
        world.insert_in(
            "a",
            vec![("peer", Value::Uuid(impostor))],
            Some(leaf),
            in_batch(batch, mode),
        );
        world.client.commit_batch(batch).expect("the batch commits");
        world.connect();
        assert_eq!(
            world.fate(batch),
            "rejected",
            "{mode:?}: {:#?}",
            world.report(&[("the batch", batch)])
        );
        assert!(world.served("a").is_empty(), "{mode:?}: no a served");
    }
}

/// A row corrected inside its open batch, whose first version waited for a sibling that
/// never came and whose correction stands on its own: the seal declares the correction, so
/// the version that waited is superseded, not a denial of the batch. A second child that
/// waits for its parent keeps the batch from being complete before its decision.
#[test]
fn a_correction_accepted_on_its_own_supersedes_the_version_that_waited() {
    let mut world = World::with_rules(true, SIBLINGS);
    let (root, root_batch) = world.insert(
        "b",
        vec![
            ("kind", text("root")),
            ("peer", Value::Uuid(ObjectId::new())),
        ],
    );
    world.sync();
    world.assert_all_accepted(&[("root", root_batch)]);

    let leaf = ObjectId::new();
    let batch = world.client.begin_batch(BatchMode::Direct);
    world.insert_in(
        "a",
        vec![("peer", Value::Uuid(ObjectId::new()))],
        Some(leaf),
        in_batch(batch, BatchMode::Direct),
    );
    world.sync();
    assert!(
        world.server_parks(batch),
        "the first version waits for a sibling"
    );
    world
        .client
        .update(
            leaf,
            vec![("peer".to_string(), Value::Uuid(root))],
            Some(&in_batch(batch, BatchMode::Direct)),
        )
        .expect("the correction applies locally");
    world.sync();
    let (child, parent) = ids_child_first();
    world.insert_in(
        "a",
        vec![("peer", Value::Uuid(parent))],
        Some(child),
        in_batch(batch, BatchMode::Direct),
    );
    world.insert_in(
        "b",
        vec![
            ("kind", text("root")),
            ("peer", Value::Uuid(ObjectId::new())),
        ],
        Some(parent),
        in_batch(batch, BatchMode::Direct),
    );
    world.client.commit_batch(batch).expect("the batch commits");
    world.sync();
    world.assert_all_accepted(&[("the batch", batch)]);
    assert_eq!(
        world.served_row("a", leaf).map(|values| values[0].clone()),
        Some(Value::Uuid(root)),
        "the correction is served"
    );
    world.assert_served(&[("parent", "b", parent), ("child", "a", child)]);
}

/// A denied row the seal leaves out still refuses its batch, whichever order the seal
/// and the declared rows arrive in: the batch is decided with its waiting rows, never
/// accepted around them.
#[test]
fn a_denied_row_the_seal_leaves_out_still_refuses_its_batch() {
    let mode = BatchMode::Transactional;
    let mut world = World::new(true);
    let (chat, _) = world.chat();
    world.membership(chat);
    world.sync();
    let (message, attachment) = ids_child_first();
    let lost = ObjectId::new();
    let batch = world.client.begin_batch(mode);
    world.insert_in(
        "messages",
        vec![
            ("chat_id", Value::Uuid(chat)),
            ("sender_id", text(ME)),
            ("body", text("hello")),
        ],
        Some(message),
        in_batch(batch, mode),
    );
    world.insert_in(
        "message_attachments",
        vec![
            ("message_id", Value::Uuid(message)),
            ("kind", text("photo")),
        ],
        Some(attachment),
        in_batch(batch, mode),
    );
    world.insert_in(
        "messages",
        vec![
            ("chat_id", Value::Uuid(chat)),
            ("sender_id", text(ME)),
            ("body", text("on its way")),
        ],
        Some(lost),
        in_batch(batch, mode),
    );
    world.lost_rows.insert(lost);
    world.sync();
    world.client.commit_batch(batch).expect("the batch commits");
    world.sync();
    assert!(world.server_parks(batch), "the attachment waits");

    let mut seal = world
        .sent
        .iter()
        .find_map(|payload| match payload {
            SyncPayload::SealBatch { submission } if submission.batch_id == batch => {
                Some(submission.clone())
            }
            _ => None,
        })
        .expect("the client sealed the batch");
    seal.members.retain(|member| member.object_id == message);
    seal.batch_digest = SealedBatchSubmission::compute_batch_digest(&seal.members);
    world.server.park_sync_message(InboxEntry {
        source: Source::Client(world.client_id),
        payload: SyncPayload::SealBatch { submission: seal },
    });
    world.sync();
    assert_eq!(
        world.fate(batch),
        "rejected",
        "{:#?}",
        world.report(&[("the batch", batch)])
    );
    assert!(!world.served("messages").contains(&message));
    assert!(!world.served("message_attachments").contains(&attachment));
}

/// A parent the batch writes and deletes again is no parent: the child that leans on it is
/// refused, as it is when the two are written one batch each.
#[test]
fn a_parent_deleted_in_its_batch_does_not_let_the_child_in() {
    for mode in [BatchMode::Direct, BatchMode::Transactional] {
        let mut world = World::with_rules(true, SIBLINGS);
        let (leaf, root) = ids_child_first();
        let batch = world.client.begin_batch(mode);
        world.insert_in(
            "b",
            vec![
                ("kind", text("root")),
                ("peer", Value::Uuid(ObjectId::new())),
            ],
            Some(root),
            in_batch(batch, mode),
        );
        world
            .client
            .delete(root, Some(&in_batch(batch, mode)))
            .expect("the delete applies locally");
        world.insert_in(
            "a",
            vec![("peer", Value::Uuid(root))],
            Some(leaf),
            in_batch(batch, mode),
        );
        world.client.commit_batch(batch).expect("the batch commits");
        world.sync();
        assert_eq!(
            world.fate(batch),
            "rejected",
            "{mode:?}: {:#?}",
            world.report(&[("the batch", batch)])
        );
        assert!(world.served("a").is_empty(), "{mode:?}: no a served");
    }
}

/// A chain in one batch is decided in at most eight rounds: one sent leaf first needs a
/// round per link. Six links pass; a chain past the bound is refused as before, at once.
#[test]
fn a_chain_in_one_batch_is_decided_within_the_bound() {
    for (links, expected) in [(6, "accepted"), (12, "rejected")] {
        let mut world = World::with_rules(false, SIBLINGS);
        let ids = ids_descending(links);
        let batch = world.client.begin_batch(BatchMode::Direct);
        for (index, id) in ids.iter().enumerate() {
            let (table, values) = if index == 0 {
                (
                    "b",
                    vec![
                        ("kind", text("root")),
                        ("peer", Value::Uuid(ObjectId::new())),
                    ],
                )
            } else if index % 2 == 1 {
                ("a", vec![("peer", Value::Uuid(ids[index - 1]))])
            } else {
                (
                    "b",
                    vec![
                        ("kind", text("leaf")),
                        ("peer", Value::Uuid(ids[index - 1])),
                    ],
                )
            };
            world.insert_in(table, values, Some(*id), in_batch(batch, BatchMode::Direct));
        }
        world.client.commit_batch(batch).expect("the batch commits");
        world.connect();
        assert_eq!(
            world.fate(batch),
            expected,
            "{links} links: {:#?}",
            world.report(&[("the chain", batch)])
        );
    }
}
