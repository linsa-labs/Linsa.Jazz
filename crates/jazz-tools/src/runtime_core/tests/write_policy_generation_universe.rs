//! Gate (tester report 2026-11-06, engine linsa-v29.3): a write whose policy reads another
//! table through `exists` must see the rows readers serve from older schema generations.
//!
//! In chats created before a deployment the phone refused every message, even plain text;
//! in chats created after it the messages went through, and an old membership the backend
//! rewrote started to work too. The messages rule asks for a membership —
//! `policy.chat_members.exists.where({ chatId: row.chatId, userId: session.user_id })` —
//! and readers serve the old membership: a read unions every branch of the schema family
//! and lenses each row into the current shape. A write's policy scanned only the write's
//! own branch, and a branch is named by the WHOLE schema's hash, so any deployment, even
//! one that touches no table the rule reads, leaves every row written before it where no
//! write check looks. A rewrite copies the row onto the new branch, which is why it seemed
//! to fix the chat. This is the write side of defect 24, whose fix gave the read side the
//! family and kept writes on their own branch.
//!
//! Seeing the family is not enough on its own: each generation keeps its own copy of a row
//! (a copy-on-write update after a deployment leaves the older generation its head, and a
//! delete inside a batch retires no older copy), so a write's policy must judge the row
//! READERS serve, never a generation's own copy of it — the negative controls below.
//!
//! Verdicts are the phone's own local refusal and the server's own fate: real runtimes on
//! both ends, a separate permissions head, SqliteStorage (`MemoryStorage` overrides the
//! visible reads and never runs the raw-table family reads this crosses).

use super::*;

use crate::batch_fate::{BatchFate, BatchMode};
use crate::metadata::{MetadataKey, RowProvenance};
use crate::query_manager::manager::VISIBLE_CANDIDATE_LOADS;
use crate::query_manager::policy::{CmpOp, OUTER_ROW_SESSION_PREFIX, PolicyValue};
use crate::query_manager::relation_ir::{
    ColumnRef, PredicateCmpOp, PredicateExpr, RelExpr, ValueRef,
};
use crate::row_histories::{RowState, StoredRowBatch};
use crate::schema_manager::lens::{Lens, LensOp, LensTransform};
use crate::storage::SqliteStorage;
use crate::sync_manager::RowMetadata;

mod differential;
mod narrowing;

const APP: &str = "write-policy-generation-universe";
const MEMBERS: &str = "chat_members";
const MESSAGES: &str = "messages";
const NOTES: &str = "notes";
const CHATS: &str = "chats";
const HANDLES: &str = "handles";
const BOARDS: &str = "boards";
const INVITES: &str = "invites";
const REDEMPTIONS: &str = "redemptions";
const CODES: &str = "codes";
const CLAIMS: &str = "claims";
const ME: &str = "alice";

type Core = RuntimeCore<SqliteStorage, NoopScheduler>;

fn sqlite() -> SqliteStorage {
    SqliteStorage::open(":memory:").expect("in-memory sqlite storage should open")
}

fn text(value: &str) -> Value {
    Value::Text(value.to_string())
}

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

/// What the deployment changes.
#[derive(Clone, Copy, Debug)]
enum Deployment {
    /// A table the rules never read: the shapes of every other table stay as they were.
    AnotherTable,
    /// `chat_members` gains `status`, `'active'` for the rows written before it, and the
    /// messages rule asks for an active membership: the tester's own schema change.
    MemberStatus,
}

/// Where a write's rules are judged first.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Device {
    /// The device holds no rules; every write reaches the server, whose verdict is the one
    /// measured.
    ServerJudges,
    /// The device holds the permissions head, as the app's phone does, and refuses a write
    /// its rules deny before sending it.
    ExplicitHead,
    /// The device's own schema carries the rules (no separate head): the other local path.
    EmbeddedPolicies,
}

fn members_table(with_status: bool) -> crate::query_manager::types::TableSchemaBuilder {
    let table = TableSchema::builder(MEMBERS)
        .column("chatId", ColumnType::Text)
        .column("userId", ColumnType::Text);
    if with_status {
        table.column("status", ColumnType::Text)
    } else {
        table
    }
}

fn schema_with(with_status: bool, extra_table: bool) -> Schema {
    let builder = SchemaBuilder::new()
        .table(members_table(with_status))
        .table(
            TableSchema::builder(MESSAGES)
                .column("chatId", ColumnType::Text)
                .column("senderUserId", ColumnType::Text)
                .column("text", ColumnType::Text),
        )
        .table(
            TableSchema::builder(NOTES)
                .column("ownerUserId", ColumnType::Text)
                .column("body", ColumnType::Text),
        )
        .table(
            TableSchema::builder(CHATS)
                .column("createdByUserId", ColumnType::Text)
                .column("kind", ColumnType::Text)
                .column("title", ColumnType::Text),
        )
        .table(
            TableSchema::builder(HANDLES)
                .column("ownerId", ColumnType::Text)
                .column("name", ColumnType::Text),
        )
        .table(
            TableSchema::builder(BOARDS)
                .column("ownerId", ColumnType::Text)
                .column("name", ColumnType::Text),
        )
        .table(
            TableSchema::builder(INVITES)
                .nullable_column("code", ColumnType::Text)
                .column("ownerId", ColumnType::Text),
        )
        .table(
            TableSchema::builder(REDEMPTIONS)
                .nullable_column("code", ColumnType::Text)
                .column("ownerId", ColumnType::Text),
        )
        .table(
            TableSchema::builder(CODES)
                .column("id", ColumnType::Text)
                .column("ownerId", ColumnType::Text),
        )
        .table(
            TableSchema::builder(CLAIMS)
                .column("code", ColumnType::Text)
                .column("ownerId", ColumnType::Text),
        );
    if extra_table {
        builder
            .table(TableSchema::builder("agent_sessions").column("state", ColumnType::Text))
            .build()
    } else {
        builder.build()
    }
}

fn first_generation() -> Schema {
    schema_with(false, false)
}

fn second_generation(deployment: Deployment) -> Schema {
    match deployment {
        Deployment::AnotherTable => schema_with(false, true),
        Deployment::MemberStatus => schema_with(true, false),
    }
}

/// The migration as the app declares it: `s.add.string({ default: 'active' })` for the
/// status deployment, and the generated lens (a table added) for the other.
fn migration(old: &Schema, new: &Schema, deployment: Deployment) -> Lens {
    match deployment {
        Deployment::AnotherTable => crate::schema_manager::auto_lens::generate_lens(old, new),
        Deployment::MemberStatus => Lens::new(
            SchemaHash::compute(old),
            SchemaHash::compute(new),
            LensTransform::with_ops(vec![LensOp::AddColumn {
                table: MEMBERS.to_string(),
                column: "status".to_string(),
                column_type: ColumnType::Text,
                default: text("active"),
            }]),
        ),
    }
}

/// The permissions head for a structural schema, in the shapes `definePermissions` compiles
/// the app's rules to:
/// - memberships and handles open to everyone (the app's backend writes memberships; the
///   gate seeds them from the device);
/// - a message only from a member of its chat, an active one wherever memberships carry a
///   status (`policy.chat_members.exists.where(…)` → `Exists`);
/// - a note only under an id no note holds (`idIsFree`);
/// - a chat updated only by its creator, and never out of its creator or kind
///   (`whereNew(row => policy.chats.exists.where({ id, createdByUserId, kind }))`);
/// - a board only by someone who owns a handle (`policy.exists(handles.where(…))` →
///   `ExistsRel` over one column);
/// - a message updated only by its sender while still a member of its chat;
/// - a redemption only of an invite of one's own with the same code — a null code too,
///   which the policy's equality matches (`{ code: row.code }`);
/// - a claim only of a code of one's own, where `codes` declares its own `id` column.
fn permissions_for(schema: &Schema) -> Schema {
    let has_status = schema[&TableName::new(MEMBERS)]
        .columns
        .column_index("status")
        .is_some();
    let mut membership = vec![cmp("chatId", outer("chatId")), cmp("userId", me())];
    if has_status {
        membership.push(cmp("status", PolicyValue::Literal(text("active"))));
    }
    let open = || {
        TablePolicies::new()
            .with_select(PolicyExpr::True)
            .with_insert(PolicyExpr::True)
            .with_update(Some(PolicyExpr::True), PolicyExpr::True)
            .with_delete(PolicyExpr::True)
    };
    let owns = |column: &str| cmp(column, me());
    let owns_a_handle = PolicyExpr::ExistsRel {
        rel: RelExpr::Filter {
            input: Box::new(RelExpr::TableScan {
                table: TableName::new(HANDLES),
            }),
            predicate: PredicateExpr::Cmp {
                left: ColumnRef::scoped(HANDLES, "ownerId"),
                op: PredicateCmpOp::Eq,
                right: ValueRef::SessionRef(vec!["user_id".into()]),
            },
        },
    };
    let mut head = schema.clone();
    for (name, table) in head.iter_mut() {
        table.policies = match name.as_str() {
            MESSAGES => {
                let sender_and_member = PolicyExpr::and(vec![
                    owns("senderUserId"),
                    exists(MEMBERS, membership.clone()),
                ]);
                TablePolicies::new()
                    .with_select(exists(MEMBERS, membership.clone()))
                    .with_insert(sender_and_member.clone())
                    .with_update(Some(sender_and_member.clone()), sender_and_member)
            }
            REDEMPTIONS => TablePolicies::new()
                .with_select(PolicyExpr::True)
                .with_insert(PolicyExpr::and(vec![
                    owns("ownerId"),
                    exists(
                        INVITES,
                        vec![cmp("code", outer("code")), cmp("ownerId", me())],
                    ),
                ])),
            CLAIMS => TablePolicies::new()
                .with_select(PolicyExpr::True)
                .with_insert(PolicyExpr::and(vec![
                    owns("ownerId"),
                    exists(CODES, vec![cmp("id", outer("code")), cmp("ownerId", me())]),
                ])),
            NOTES => TablePolicies::new()
                .with_select(PolicyExpr::True)
                .with_insert(PolicyExpr::and(vec![
                    owns("ownerUserId"),
                    PolicyExpr::not(exists(NOTES, vec![cmp("id", outer("id"))])),
                ]))
                .with_update(Some(owns("ownerUserId")), owns("ownerUserId"))
                .with_delete(owns("ownerUserId")),
            CHATS => TablePolicies::new()
                .with_select(PolicyExpr::True)
                .with_insert(owns("createdByUserId"))
                .with_update(
                    Some(owns("createdByUserId")),
                    exists(
                        CHATS,
                        vec![
                            cmp("id", outer("id")),
                            cmp("createdByUserId", outer("createdByUserId")),
                            cmp("kind", outer("kind")),
                        ],
                    ),
                ),
            BOARDS => TablePolicies::new()
                .with_select(PolicyExpr::True)
                .with_insert(PolicyExpr::and(vec![
                    owns("ownerId"),
                    owns_a_handle.clone(),
                ])),
            _ => open(),
        };
    }
    head
}

/// A node at `tier`: a deployed server at the global tier (`server::builder`), an app's
/// device at the local one. A node with no tier never settles a batch.
fn runtime(schema: Schema, storage: SqliteStorage, tier: DurabilityTier) -> Core {
    let app_id = AppId::from_name(APP);
    let sync_manager = SyncManager::new().with_durability_tier(tier);
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

fn server_runtime(schema: Schema, storage: SqliteStorage) -> Core {
    let mut core = runtime(schema.clone(), storage, DurabilityTier::GlobalServer);
    core.schema_manager_mut()
        .query_manager_mut()
        .set_authorization_schema(permissions_for(&schema));
    core.schema_manager_mut()
        .query_manager_mut()
        .require_authorization_schema();
    core.immediate_tick();
    core
}

fn device_runtime(schema: Schema, storage: SqliteStorage) -> Core {
    runtime(schema, storage, DurabilityTier::Local)
}

fn client_runtime(schema: Schema, storage: SqliteStorage, device: Device) -> Core {
    match device {
        Device::ServerJudges => device_runtime(schema, storage),
        Device::ExplicitHead => {
            let mut core = device_runtime(schema.clone(), storage);
            core.schema_manager_mut()
                .query_manager_mut()
                .set_authorization_schema(permissions_for(&schema));
            core.immediate_tick();
            core
        }
        Device::EmbeddedPolicies => device_runtime(permissions_for(&schema), storage),
    }
}

/// One-shot bidirectional pump (as `migrated_write_policy::pump`).
fn pump(server: &mut Core, server_id: ServerId, client: &mut Core, client_id: ClientId) {
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

/// The device's queued frames, delivered to the server once; the server's refusals.
fn deliver(
    server: &mut Core,
    server_id: ServerId,
    client: &mut Core,
    client_id: ClientId,
) -> Vec<String> {
    client.batched_tick();
    for entry in client.sync_sender().take() {
        if entry.destination == Destination::Server(server_id) {
            server.park_sync_message(InboxEntry {
                source: Source::Client(client_id),
                payload: entry.payload,
            });
        }
    }
    server.batched_tick();
    server.immediate_tick();
    server.batched_tick();
    rejections(server)
}

fn rejections(server: &mut Core) -> Vec<String> {
    server
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

/// What became of a write.
#[derive(Debug, PartialEq)]
enum Verdict {
    Sent,
    RefusedLocally(String),
    RejectedByServer(Vec<String>),
}

fn verdict_of(
    written: Result<(), String>,
    server_refusals: impl FnOnce() -> Vec<String>,
) -> Verdict {
    if let Err(error) = written {
        return Verdict::RefusedLocally(error);
    }
    let refusals = server_refusals();
    if refusals.is_empty() {
        Verdict::Sent
    } else {
        Verdict::RejectedByServer(refusals)
    }
}

fn message(chat: &str) -> HashMap<String, Value> {
    HashMap::from([
        ("chatId".to_string(), text(chat)),
        ("senderUserId".to_string(), text(ME)),
        ("text".to_string(), text(&format!("hello {chat}"))),
    ])
}

struct World {
    deployment: Deployment,
    device: Device,
    server: Core,
    client: Core,
    server_id: ServerId,
    client_id: ClientId,
    schema: Schema,
    /// The server's branch before the deployment.
    first_branch: String,
    /// Memberships by chat, as written.
    members: HashMap<&'static str, ObjectId>,
    /// Alice's note, written before the deployment.
    note: ObjectId,
    /// Alice's chats, written before the deployment: one she renames, one she tries to
    /// turn into another kind.
    renamed_chat: ObjectId,
    rekinded_chat: ObjectId,
    /// Alice's handle, written before the deployment.
    handle: ObjectId,
}

impl World {
    fn alice() -> WriteContext {
        WriteContext::from_session(Session::new(ME))
    }

    /// Before the deployment: alice is a member of `chat-old`, `chat-rewritten`,
    /// `chat-left`, `chat-left-in-batch` and `chat-handed`, bob of `chat-bob`; alice owns
    /// a note, two chats and a handle.
    fn new(deployment: Deployment, device: Device) -> Self {
        let schema = first_generation();
        let mut server = server_runtime(schema.clone(), sqlite());
        let client = client_runtime(schema.clone(), sqlite(), device);
        let server_id = ServerId::new();
        let client_id = ClientId::new();
        server.add_client(client_id, Some(Session::new(ME)));
        let first_branch = server.schema_manager().branch_name().as_str().to_string();
        let mut world = Self {
            deployment,
            device,
            server,
            client,
            server_id,
            client_id,
            schema,
            first_branch,
            members: HashMap::new(),
            note: ObjectId::new(),
            renamed_chat: ObjectId::new(),
            rekinded_chat: ObjectId::new(),
            handle: ObjectId::new(),
        };
        world.client.add_server(server_id);
        world.client.immediate_tick();
        for (chat, user) in [
            ("chat-old", ME),
            ("chat-rewritten", ME),
            ("chat-left", ME),
            ("chat-left-in-batch", ME),
            ("chat-handed", ME),
            ("chat-bob", "bob"),
        ] {
            world.join(chat, user);
        }
        // A device with embedded rules refuses every `idIsFree` insert, deployment or not
        // (a separate defect of that path, observed 2026-11-06); the note serves only the
        // id gate, which runs with the server judging.
        if device != Device::EmbeddedPolicies {
            world.note = world.seed(
                NOTES,
                vec![("ownerUserId", text(ME)), ("body", text("mine"))],
            );
        }
        for chat in ["renamed_chat", "rekinded_chat"] {
            let id = world.seed(
                CHATS,
                vec![
                    ("createdByUserId", text(ME)),
                    ("kind", text("group")),
                    ("title", text("before")),
                ],
            );
            match chat {
                "renamed_chat" => world.renamed_chat = id,
                _ => world.rekinded_chat = id,
            }
        }
        world.handle = world.seed(HANDLES, vec![("ownerId", text(ME)), ("name", text("al"))]);
        world.seed(INVITES, vec![("code", Value::Null), ("ownerId", text(ME))]);
        world.seed(CODES, vec![("id", text("C1")), ("ownerId", text(ME))]);
        world.sync();
        let refusals = rejections(&mut world.server);
        assert!(
            refusals.is_empty(),
            "control: a seed write was rejected ({refusals:?})"
        );
        world
    }

    fn seed(&mut self, table: &str, values: Vec<(&str, Value)>) -> ObjectId {
        let values = values
            .into_iter()
            .map(|(column, value)| (column.to_string(), value))
            .collect();
        let ((id, _), _) = self
            .client
            .insert(table, values, Some(&Self::alice()))
            .unwrap_or_else(|error| panic!("control: seeding {table} was refused: {error:?}"));
        self.sync();
        id
    }

    fn membership_values(&self, chat: &str, user: &str) -> HashMap<String, Value> {
        let mut values = HashMap::from([
            ("chatId".to_string(), text(chat)),
            ("userId".to_string(), text(user)),
        ]);
        if self.schema[&TableName::new(MEMBERS)]
            .columns
            .column_index("status")
            .is_some()
        {
            values.insert("status".to_string(), text("active"));
        }
        values
    }

    fn join(&mut self, chat: &'static str, user: &str) {
        let member = self.add_member(chat, user);
        self.members.insert(chat, member);
    }

    fn add_member(&mut self, chat: &str, user: &str) -> ObjectId {
        let ((member, _), _) = self
            .client
            .insert(
                MEMBERS,
                self.membership_values(chat, user),
                Some(&Self::alice()),
            )
            .expect("memberships are open");
        self.sync();
        member
    }

    /// `count` memberships of other people in other chats.
    fn crowd(&mut self, count: usize, tag: &str) {
        for index in 0..count {
            self.add_member(
                &format!("crowd-{tag}-{index}"),
                &format!("user-{tag}-{index}"),
            );
        }
    }

    /// What the backend's rewrite of a membership does: a new version of the row, written
    /// under the current schema.
    fn rewrite(&mut self, chat: &str) {
        self.hand_to(chat, ME);
    }

    /// The membership given to `user`: alice's no longer.
    fn hand_to(&mut self, chat: &str, user: &str) {
        let member = self.members[chat];
        self.client
            .update(
                member,
                vec![("userId".to_string(), text(user))],
                Some(&Self::alice()),
            )
            .expect("memberships are open");
        self.sync();
    }

    /// The membership deleted: alice left the chat.
    fn leave(&mut self, chat: &str) {
        let member = self.members[chat];
        self.client
            .delete(member, Some(&Self::alice()))
            .expect("memberships are open");
        self.sync();
    }

    /// Alice leaves `chat` inside a batch — and, with `goodbye`, posts in it in the same
    /// batch. A delete in an open batch retires no older generation's copy of the row.
    fn leave_in_batch(&mut self, chat: &str, goodbye: bool) -> Verdict {
        let member = self.members[chat];
        let batch = self.client.begin_batch(BatchMode::Direct);
        let in_batch = Self::alice()
            .with_batch_mode(BatchMode::Direct)
            .with_batch_id(batch);
        self.client
            .delete(member, Some(&in_batch))
            .expect("memberships are open");
        let posted = if goodbye {
            self.client
                .insert(MESSAGES, message(chat), Some(&in_batch))
                .map(|_| ())
                .map_err(|error| format!("{error:?}"))
        } else {
            Ok(())
        };
        self.client.commit_batch(batch).expect("the batch commits");
        let verdict = verdict_of(posted, || {
            deliver(
                &mut self.server,
                self.server_id,
                &mut self.client,
                self.client_id,
            )
        });
        self.sync();
        verdict
    }

    /// A deployment: both ends rebuilt on the next generation from their own stores, the
    /// migration published server-side, and the reconnect handshake in both directions.
    fn deploy(&mut self) {
        let before = self.branch();
        let next = second_generation(self.deployment);
        let lens = migration(&self.schema, &next, self.deployment);
        let server_storage = std::mem::replace(
            &mut self.server,
            device_runtime(first_generation(), sqlite()),
        )
        .into_storage();
        let client_storage = std::mem::replace(
            &mut self.client,
            device_runtime(first_generation(), sqlite()),
        )
        .into_storage();
        self.server = server_runtime(next.clone(), server_storage);
        self.server.publish_lens(&lens).expect("lens publishes");
        self.client = client_runtime(next.clone(), client_storage, self.device);
        self.server
            .add_client(self.client_id, Some(Session::new(ME)));
        self.client.add_server(self.server_id);
        self.server.immediate_tick();
        self.client.immediate_tick();
        self.schema = next;
        self.sync();
        self.server.sync_sender().take();
        self.client.sync_sender().take();
        assert_ne!(
            self.branch(),
            before,
            "control: the deployment must mint a new branch, or nothing crosses"
        );
    }

    fn branch(&self) -> String {
        self.server
            .schema_manager()
            .branch_name()
            .as_str()
            .to_string()
    }

    /// Alice sends a message to `chat` from her device.
    fn send(&mut self, chat: &str) -> Verdict {
        let written = self
            .client
            .insert(MESSAGES, message(chat), Some(&Self::alice()))
            .map(|_| ())
            .map_err(|error| format!("{error:?}"));
        verdict_of(written, || {
            deliver(
                &mut self.server,
                self.server_id,
                &mut self.client,
                self.client_id,
            )
        })
    }

    /// Alice updates one of her chats from her device.
    fn update_chat(&mut self, chat: ObjectId, column: &str, value: &str) -> Verdict {
        let written = self
            .client
            .update(
                chat,
                vec![(column.to_string(), text(value))],
                Some(&Self::alice()),
            )
            .map(|_| ())
            .map_err(|error| format!("{error:?}"));
        verdict_of(written, || {
            deliver(
                &mut self.server,
                self.server_id,
                &mut self.client,
                self.client_id,
            )
        })
    }

    /// Alice creates a board from her device.
    fn create_board(&mut self, name: &str) -> Verdict {
        let written = self
            .client
            .insert(
                BOARDS,
                HashMap::from([
                    ("ownerId".to_string(), text(ME)),
                    ("name".to_string(), text(name)),
                ]),
                Some(&Self::alice()),
            )
            .map(|_| ())
            .map_err(|error| format!("{error:?}"));
        verdict_of(written, || {
            deliver(
                &mut self.server,
                self.server_id,
                &mut self.client,
                self.client_id,
            )
        })
    }

    /// Alice's handle given to bob.
    fn hand_the_handle_to_bob(&mut self) {
        self.client
            .update(
                self.handle,
                vec![("ownerId".to_string(), text("bob"))],
                Some(&Self::alice()),
            )
            .expect("handles are open");
        self.sync();
    }

    /// A second device of alice's, still on the schema from before the deployment, with no
    /// rules of its own: the server judges what it sends.
    fn old_device(&mut self) -> (Core, ClientId) {
        let mut device = device_runtime(first_generation(), sqlite());
        let device_id = ClientId::new();
        self.server.add_client(device_id, Some(Session::new(ME)));
        device.add_server(self.server_id);
        device.immediate_tick();
        pump(&mut self.server, self.server_id, &mut device, device_id);
        rejections(&mut self.server);
        (device, device_id)
    }

    /// The old device back on the server after the deployment rebuilt it.
    fn reconnect(&mut self, device: &mut Core, device_id: ClientId) {
        device.remove_server(self.server_id);
        self.server.add_client(device_id, Some(Session::new(ME)));
        device.add_server(self.server_id);
        device.immediate_tick();
        pump(&mut self.server, self.server_id, device, device_id);
        rejections(&mut self.server);
    }

    /// Alice writes a row of `table` from her device; how it went.
    fn write_as_alice(&mut self, table: &str, values: Vec<(&str, Value)>) -> Verdict {
        let values = values
            .into_iter()
            .map(|(column, value)| (column.to_string(), value))
            .collect();
        let written = self
            .client
            .insert(table, values, Some(&Self::alice()))
            .map(|_| ())
            .map_err(|error| format!("{error:?}"));
        verdict_of(written, || {
            deliver(
                &mut self.server,
                self.server_id,
                &mut self.client,
                self.client_id,
            )
        })
    }

    /// Bob's message in `chat`, written by the app's backend on the server.
    fn backend_post(&mut self, chat: &str) {
        self.server
            .insert(
                MESSAGES,
                HashMap::from([
                    ("chatId".to_string(), text(chat)),
                    ("senderUserId".to_string(), text("bob")),
                    ("text".to_string(), text("from bob")),
                ]),
                None,
            )
            .expect("the backend writes unchecked");
        self.server.immediate_tick();
        self.server.sync_sender().take();
    }

    /// The messages of `chat` the server authorizes alice to read.
    fn alice_reads_messages(&mut self, chat: &str) -> Vec<ObjectId> {
        let query = self
            .server
            .schema_manager_mut()
            .query_manager_mut()
            .query(MESSAGES)
            .filter_eq("chatId", text(chat))
            .build();
        let waker = noop_waker();
        let mut cx = std::task::Context::from_waker(&waker);
        let mut future = self.server.query_with_propagation(
            query,
            Some(Session::new(ME)),
            ReadDurabilityOptions::default(),
            crate::sync_manager::QueryPropagation::LocalOnly,
        );
        match Pin::new(&mut future).poll(&mut cx) {
            Poll::Ready(Ok(rows)) => rows.into_iter().map(|(id, _)| id).collect(),
            Poll::Ready(Err(error)) => panic!("the read should succeed: {error:?}"),
            Poll::Pending => panic!("a local read should resolve at once"),
        }
    }

    fn send_from(&mut self, device: &mut Core, device_id: ClientId, chat: &str) -> Verdict {
        let written = device
            .insert(MESSAGES, message(chat), Some(&Self::alice()))
            .map(|_| ())
            .map_err(|error| format!("{error:?}"));
        verdict_of(written, || {
            deliver(&mut self.server, self.server_id, device, device_id)
        })
    }

    /// A parentless note `mallory` sends under `id` from a connection of her own; the
    /// server's refusals.
    fn mallory_writes_note(&mut self, id: ObjectId) -> Vec<String> {
        let connection = ClientId::new();
        self.server
            .add_client(connection, Some(Session::new("mallory")));
        self.server.sync_sender().take();
        let row = StoredRowBatch::new(
            id,
            self.branch().as_str(),
            Vec::new(),
            encode_row(
                &self.schema[&TableName::new(NOTES)].columns,
                &[text("mallory"), text("taken")],
            )
            .expect("row encodes"),
            RowProvenance::for_insert("mallory".to_string(), 9_999),
            HashMap::new(),
            RowState::VisibleDirect,
            None,
        );
        self.server.park_sync_message(InboxEntry {
            source: Source::Client(connection),
            payload: SyncPayload::RowBatchCreated {
                metadata: Some(RowMetadata {
                    id,
                    metadata: HashMap::from([
                        (MetadataKey::Table.to_string(), NOTES.to_string()),
                        (
                            MetadataKey::OriginSchemaHash.to_string(),
                            SchemaHash::compute(&self.schema).to_string(),
                        ),
                    ]),
                }),
                row,
            },
        });
        self.server.batched_tick();
        self.server.immediate_tick();
        self.server.batched_tick();
        rejections(&mut self.server)
    }

    /// The server's own visible head of a row on `branch`, if it is live.
    fn live_on(&self, table: &str, branch: &str, id: ObjectId) -> bool {
        self.server
            .storage()
            .load_visible_region_row(table, branch, id)
            .expect("visible read should succeed")
            .is_some_and(|row| !row.is_deleted)
    }

    fn sync(&mut self) {
        pump(
            &mut self.server,
            self.server_id,
            &mut self.client,
            self.client_id,
        );
    }

    fn refused(&self, verdict: &Verdict) -> bool {
        match verdict {
            Verdict::Sent => false,
            Verdict::RefusedLocally(_) => self.device != Device::ServerJudges,
            Verdict::RejectedByServer(_) => self.device == Device::ServerJudges,
        }
    }
}

fn a_member_from_before_the_deployment_may_send(deployment: Deployment, device: Device) {
    let mut world = World::new(deployment, device);
    assert_eq!(
        world.send("chat-old"),
        Verdict::Sent,
        "control: a member's message in the generation her membership was written in"
    );

    world.deploy();
    world.join("chat-new", ME);
    world.rewrite("chat-rewritten");
    world.leave("chat-left");
    world.hand_to("chat-handed", "bob");

    assert_eq!(
        world.send("chat-new"),
        Verdict::Sent,
        "control: a membership written after the deployment"
    );
    assert_eq!(
        world.send("chat-rewritten"),
        Verdict::Sent,
        "control: an old membership rewritten after the deployment"
    );
    // Negative controls: a fix that approves every message passes the first two; one that
    // judges each generation's own copy of a row, instead of the row readers serve, passes
    // the last — the old generation still holds alice's membership as it was before it was
    // handed to bob.
    for chat in ["chat-none", "chat-bob", "chat-left", "chat-handed"] {
        let verdict = world.send(chat);
        assert!(
            world.refused(&verdict),
            "a message to {chat}, where alice is no member, was not refused where it is \
             judged ({device:?}): {verdict:?}"
        );
    }

    assert_eq!(
        world.send("chat-old"),
        Verdict::Sent,
        "alice's membership in chat-old was written before the deployment ({deployment:?}); \
         readers serve it, and the write's policy must see it too"
    );
}

#[test]
fn a_member_from_before_a_deployment_of_another_table_may_send() {
    a_member_from_before_the_deployment_may_send(Deployment::AnotherTable, Device::ServerJudges);
}

#[test]
fn a_member_from_before_a_deployment_of_another_table_may_send_from_the_phone() {
    a_member_from_before_the_deployment_may_send(Deployment::AnotherTable, Device::ExplicitHead);
}

#[test]
#[ignore = "the device-local legacy path (rules embedded in the schema) needs the family in \
            its own Inherits/InheritsReferencing scans and a lensing loader; a separate fix"]
fn a_member_from_before_a_deployment_may_send_from_a_device_with_embedded_rules() {
    a_member_from_before_the_deployment_may_send(
        Deployment::AnotherTable,
        Device::EmbeddedPolicies,
    );
}

#[test]
fn a_member_the_deployment_marked_active_may_send() {
    a_member_from_before_the_deployment_may_send(Deployment::MemberStatus, Device::ServerJudges);
}

#[test]
fn a_member_the_deployment_marked_active_may_send_from_the_phone() {
    a_member_from_before_the_deployment_may_send(Deployment::MemberStatus, Device::ExplicitHead);
}

/// A delete inside an open batch retires no older generation's copy of the row, so after
/// it the old generation still holds alice's membership live while the generation readers
/// serve holds its tombstone — and the tombstone keeps the row's data. A check that lets
/// the old copy's index entry in and then judges the tombstone's data as a row grants the
/// message.
fn a_member_who_left_in_a_batch_may_not_send(device: Device) {
    let mut world = World::new(Deployment::AnotherTable, device);
    world.deploy();
    assert_eq!(
        world.leave_in_batch("chat-left-in-batch", false),
        Verdict::Sent,
        "control: leaving a chat is open"
    );
    let member = world.members["chat-left-in-batch"];
    let first_branch = world.first_branch.clone();
    assert!(
        world.live_on(MEMBERS, &first_branch, member),
        "control: the older generation must still hold the membership live, or this gate \
         tests nothing"
    );

    let verdict = world.send("chat-left-in-batch");
    assert!(
        world.refused(&verdict),
        "alice left chat-left-in-batch; readers serve no membership, yet her message was not \
         refused where it is judged ({device:?}): {verdict:?}"
    );
}

#[test]
fn a_member_who_left_in_a_batch_after_a_deployment_may_not_send() {
    a_member_who_left_in_a_batch_may_not_send(Device::ServerJudges);
}

#[test]
fn a_member_who_left_in_a_batch_after_a_deployment_may_not_send_from_the_phone() {
    a_member_who_left_in_a_batch_may_not_send(Device::ExplicitHead);
}

/// A batch is judged as it would be with no deployment: the generation it crosses decides
/// nothing. Leaving a chat and posting in it in one batch comes out the same on both sides
/// of a deployment.
fn a_batch_is_judged_the_same_across_a_deployment(device: Device) {
    let leave_and_post = |deploy: bool| {
        let mut world = World::new(Deployment::AnotherTable, device);
        if deploy {
            world.deploy();
        }
        let verdict = world.leave_in_batch("chat-left-in-batch", true);
        world.refused(&verdict)
    };
    assert_eq!(
        leave_and_post(true),
        leave_and_post(false),
        "leaving chat-left-in-batch and posting in it in one batch was judged differently \
         after a deployment ({device:?})"
    );
}

#[test]
fn a_batch_is_judged_the_same_across_a_deployment_by_the_server() {
    a_batch_is_judged_the_same_across_a_deployment(Device::ServerJudges);
}

#[test]
fn a_batch_is_judged_the_same_across_a_deployment_on_the_phone() {
    a_batch_is_judged_the_same_across_a_deployment(Device::ExplicitHead);
}

/// The batch's own rows count as written when its other rows are judged: leaving a chat
/// and posting in it in one batch must be refused.
#[test]
#[ignore = "pre-existing, no deployment needed (measured on v29.3 2026-11-06): a check that \
            passes on its first evaluation is final, so a batch's own delete of the row it \
            leans on is never seen by it; only denials are parked and read again with the batch"]
fn leaving_and_posting_in_one_batch_is_refused() {
    for device in [Device::ServerJudges, Device::ExplicitHead] {
        let mut world = World::new(Deployment::AnotherTable, device);
        let verdict = world.leave_in_batch("chat-left-in-batch", true);
        assert!(
            world.refused(&verdict),
            "alice left chat-left-in-batch and posted in it in the same batch; the post was \
             not refused where it is judged ({device:?}): {verdict:?}"
        );
    }
}

/// The second face: a rule that reads the row's OWN table. A chat's WITH CHECK asks for a
/// stored chat with the same id, creator and kind (the frozen-column idiom); before the
/// first update after a deployment the stored chat lives only on the older generation.
fn first_update_keeps_frozen_columns(device: Device) {
    let mut world = World::new(Deployment::AnotherTable, device);
    world.deploy();

    let rekinded_chat = world.rekinded_chat;
    let rekinded = world.update_chat(rekinded_chat, "kind", "oneToOne");
    assert!(
        world.refused(&rekinded),
        "negative control: a chat turned into another kind was not refused ({device:?}): \
         {rekinded:?}"
    );

    let renamed_chat = world.renamed_chat;
    assert_eq!(
        world.update_chat(renamed_chat, "title", "after"),
        Verdict::Sent,
        "alice renamed her chat from before the deployment, keeping its creator and kind \
         ({device:?})"
    );
}

#[test]
fn the_first_update_after_a_deployment_keeps_its_frozen_columns() {
    first_update_keeps_frozen_columns(Device::ServerJudges);
}

#[test]
fn the_first_update_after_a_deployment_keeps_its_frozen_columns_on_the_phone() {
    first_update_keeps_frozen_columns(Device::ExplicitHead);
}

/// A device that stayed on the older schema writes onto the older generation, whose own
/// copy of a membership is stale once the membership was handed away on the newer one.
#[test]
fn a_device_on_the_old_schema_is_judged_by_the_rows_readers_serve() {
    let mut world = World::new(Deployment::AnotherTable, Device::ServerJudges);
    world.deploy();
    world.leave("chat-left");
    world.hand_to("chat-handed", "bob");
    let (mut old, old_id) = world.old_device();

    assert_eq!(
        world.send_from(&mut old, old_id, "chat-old"),
        Verdict::Sent,
        "control: the old device sends to a chat alice is a member of"
    );
    for chat in ["chat-none", "chat-left", "chat-handed"] {
        let verdict = world.send_from(&mut old, old_id, chat);
        assert!(
            matches!(verdict, Verdict::RejectedByServer(_)),
            "the old device's message to {chat}, where readers serve no membership of \
             alice's, was not rejected: {verdict:?}"
        );
    }
}

/// `policy.exists(handles.where({ ownerId: session.user_id }))` compiles to an index scan on
/// one column per branch; the older generation's copy of a handed-away handle keeps its
/// index entry under alice.
#[test]
fn a_handle_handed_away_after_a_deployment_no_longer_lets_its_old_owner_create_a_board() {
    let mut world = World::new(Deployment::AnotherTable, Device::ServerJudges);
    world.deploy();
    assert_eq!(
        world.create_board("first"),
        Verdict::Sent,
        "control (red while the write sees its own generation only): alice owns a handle"
    );

    world.hand_the_handle_to_bob();
    let handle = world.handle;
    let first_branch = world.first_branch.clone();
    assert!(
        world.live_on(HANDLES, &first_branch, handle),
        "control: the older generation must still hold alice's copy of the handle, or this \
         gate tests nothing"
    );
    let verdict = world.create_board("second");
    assert!(
        matches!(verdict, Verdict::RejectedByServer(_)),
        "alice's handle is bob's now; readers serve no handle of hers, yet her board was not \
         rejected: {verdict:?}"
    );
}

/// What a write check pays must not grow with rows the rule cannot match: the messages
/// rule names one chat and one user, and other people's memberships in other chats are
/// nothing to it.
#[test]
fn a_write_checks_reads_do_not_grow_with_unrelated_memberships() {
    let mut world = World::new(Deployment::AnotherTable, Device::ServerJudges);
    world.crowd(20, "first");
    world.deploy();
    world.join("chat-new", ME);
    world.crowd(20, "second");

    let reads_of = |world: &mut World| {
        let before = VISIBLE_CANDIDATE_LOADS.with(|loads| loads.get());
        let verdict = world.send("chat-new");
        assert_eq!(
            verdict,
            Verdict::Sent,
            "control: alice is a member of chat-new"
        );
        VISIBLE_CANDIDATE_LOADS.with(|loads| loads.get()) - before
    };
    let with_forty = reads_of(&mut world);
    world.crowd(40, "more");
    let with_eighty = reads_of(&mut world);
    eprintln!(
        "visible candidate loads per message check: {with_forty} with 40 unrelated \
         memberships, {with_eighty} with 80"
    );
    assert_eq!(
        with_eighty, with_forty,
        "a message check read more rows once other people joined other chats"
    );
}

/// The twin of the membership arm: `idIsFree` is `Not(Exists)` over the written table, so
/// a scan that misses the older generations would call their ids free and let mallory
/// "insert" a note under the id of alice's note from before the deployment. Measured on
/// v29.3: the server does not judge that write as an insert at all — it finds the old row
/// across branches and refuses it as an update ("cannot see old row"). A control the fix
/// must keep: widening the id guard's scan can only make it stricter.
#[test]
fn an_id_from_before_a_deployment_is_not_free() {
    let mut world = World::new(Deployment::AnotherTable, Device::ServerJudges);
    world.deploy();

    let fresh = world.mallory_writes_note(ObjectId::new());
    assert!(
        fresh.is_empty(),
        "control: mallory's note under a free id was refused ({fresh:?}); the channel is \
         broken"
    );

    let note = world.note;
    let taken = world.mallory_writes_note(note);
    assert_eq!(
        taken.len(),
        1,
        "mallory's note under the id of alice's note from before the deployment was not \
         refused: {taken:?}"
    );
    let branch = world.branch();
    assert!(
        !world.live_on(NOTES, &branch, note),
        "the refused note landed on the current branch anyway"
    );
}

/// The read side of the handle gate (hypothesis from the write review): a plain read
/// filtered by one column plans an index scan per branch, and the older generation's copy
/// of a handle handed away keeps its index entry under alice. A read must serve the handle
/// as it is now — bob's — and so not at all under alice.
#[test]
#[ignore = "the general read path's own defect: a plan covered by one index scan places no \
            residual filter, so an older generation's copy listed by its value is served as \
            the newest version; fixed apart from the write checks"]
fn a_read_by_one_column_serves_no_copy_a_newer_generation_moved_on() {
    let mut world = World::new(Deployment::AnotherTable, Device::ServerJudges);
    world.deploy();
    world.hand_the_handle_to_bob();
    let handle = world.handle;
    let first_branch = world.first_branch.clone();
    assert!(
        world.live_on(HANDLES, &first_branch, handle),
        "control: the older generation must still hold alice's copy of the handle"
    );

    let read = |core: &mut Core, owner: &str| -> Vec<ObjectId> {
        let query = core
            .schema_manager_mut()
            .query_manager_mut()
            .query(HANDLES)
            .filter_eq("ownerId", text(owner))
            .build();
        let sub_id = core
            .schema_manager_mut()
            .query_manager_mut()
            .subscribe(query)
            .expect("the read subscribes");
        core.immediate_tick();
        let rows = core
            .schema_manager_mut()
            .query_manager_mut()
            .get_subscription_results(sub_id);
        core.schema_manager_mut()
            .query_manager_mut()
            .unsubscribe_with_sync(sub_id);
        rows.into_iter().map(|(id, _)| id).collect()
    };
    let served: Vec<(&str, Vec<ObjectId>, Vec<ObjectId>)> =
        [("server", &mut world.server), ("device", &mut world.client)]
            .into_iter()
            .map(|(end, core)| (end, read(core, "bob"), read(core, ME)))
            .collect();
    eprintln!("handle reads (end, under bob, under alice): {served:?}");
    for (end, under_bob, under_alice) in served {
        assert_eq!(
            under_bob,
            vec![handle],
            "control: the {end} serves the handle under its owner now"
        );
        assert_eq!(
            under_alice,
            Vec::<ObjectId>::new(),
            "the {end} served alice's handle under alice after it was handed to bob"
        );
    }
}

/// The read twin of the batch-left gate: the messages rule's SELECT arm reads the same
/// membership, so once alice left in a batch the server must not authorize the chat's new
/// messages to her, though the older generation still holds her copy.
#[test]
fn a_member_who_left_in_a_batch_is_not_served_the_chats_new_messages() {
    let mut world = World::new(Deployment::AnotherTable, Device::ServerJudges);
    world.deploy();
    assert_eq!(
        world.leave_in_batch("chat-left-in-batch", false),
        Verdict::Sent,
        "control: leaving a chat is open"
    );
    let member = world.members["chat-left-in-batch"];
    let first_branch = world.first_branch.clone();
    assert!(
        world.live_on(MEMBERS, &first_branch, member),
        "control: the older generation must still hold the membership live"
    );
    world.backend_post("chat-old");
    world.backend_post("chat-left-in-batch");

    assert_eq!(
        world.alice_reads_messages("chat-old").len(),
        1,
        "control: alice reads bob's message in a chat she is still a member of"
    );
    let served = world.alice_reads_messages("chat-left-in-batch");
    assert!(
        served.is_empty(),
        "alice left chat-left-in-batch, yet the server authorizes its new message to her: \
         {served:?}"
    );
}

/// A device that stayed on the older schema joins a chat and posts in it in one batch
/// after a deployment that changed `chat_members`' layout: the batch's own membership is
/// read in the older layout, and must still vouch for the post.
#[test]
fn an_old_device_joins_and_posts_in_one_batch_after_a_members_layout_change() {
    let mut world = World::new(Deployment::MemberStatus, Device::ServerJudges);
    world.deploy();
    let (mut old, old_id) = world.old_device();
    let batch = old.begin_batch(BatchMode::Direct);
    let in_batch = World::alice()
        .with_batch_mode(BatchMode::Direct)
        .with_batch_id(batch);
    old.insert(
        MEMBERS,
        HashMap::from([
            ("chatId".to_string(), text("chat-fresh")),
            ("userId".to_string(), text(ME)),
        ]),
        Some(&in_batch),
    )
    .expect("memberships are open");
    old.insert(MESSAGES, message("chat-fresh"), Some(&in_batch))
        .expect("the old device holds no rules");
    old.commit_batch(batch).expect("the batch commits");
    let refusals = deliver(&mut world.server, world.server_id, &mut old, old_id);
    assert!(
        refusals.is_empty(),
        "the old device's join-and-post batch was rejected: {refusals:?}"
    );
}

/// The policy's equality matches a null to a null (`{ code: row.code }` with no code);
/// an index holds no entry for a null, so a scan narrowed to the value finds nothing.
#[test]
fn a_bound_null_still_matches_after_a_deployment() {
    let mut world = World::new(Deployment::AnotherTable, Device::ServerJudges);
    let redeem = |world: &mut World| {
        world.write_as_alice(
            REDEMPTIONS,
            vec![("code", Value::Null), ("ownerId", text(ME))],
        )
    };
    assert_eq!(
        redeem(&mut world),
        Verdict::Sent,
        "control: alice redeems her own invite with no code before the deployment"
    );
    world.deploy();
    assert_eq!(
        redeem(&mut world),
        Verdict::Sent,
        "alice's invite with no code is from before the deployment; readers serve it"
    );
}

/// `codes` declares its own `id` column, which the rule matches; the row id is another
/// thing.
#[test]
fn a_table_with_its_own_id_column_is_matched_by_that_column() {
    let mut world = World::new(Deployment::AnotherTable, Device::ServerJudges);
    let claim = |world: &mut World| {
        world.write_as_alice(CLAIMS, vec![("code", text("C1")), ("ownerId", text(ME))])
    };
    assert_eq!(
        claim(&mut world),
        Verdict::Sent,
        "control: alice claims her own code before the deployment"
    );
    world.deploy();
    assert_eq!(
        claim(&mut world),
        Verdict::Sent,
        "alice's code is from before the deployment; readers serve it"
    );
}

/// The update face of the old-device gate: the old device edits alice's message in a chat
/// whose membership was handed to bob on the newer generation.
#[test]
fn a_device_on_the_old_schema_may_not_edit_in_a_chat_it_was_moved_out_of() {
    let mut world = World::new(Deployment::AnotherTable, Device::ServerJudges);
    let (mut old, old_id) = world.old_device();
    let mut post = |world: &mut World, chat: &str| {
        let ((id, _), _) = old
            .insert(MESSAGES, message(chat), Some(&World::alice()))
            .expect("the old device holds no rules");
        let refusals = deliver(&mut world.server, world.server_id, &mut old, old_id);
        assert!(
            refusals.is_empty(),
            "control: a member's post ({refusals:?})"
        );
        id
    };
    let in_old = post(&mut world, "chat-old");
    let in_handed = post(&mut world, "chat-handed");
    world.deploy();
    world.reconnect(&mut old, old_id);
    world.hand_to("chat-handed", "bob");

    let mut edit = |world: &mut World, id: ObjectId| {
        old.update(
            id,
            vec![("text".to_string(), text("edited"))],
            Some(&World::alice()),
        )
        .expect("the old device holds no rules");
        deliver(&mut world.server, world.server_id, &mut old, old_id)
    };
    let edited_old = edit(&mut world, in_old);
    assert!(
        edited_old.is_empty(),
        "control: the old device edits alice's message in a chat she is a member of \
         ({edited_old:?})"
    );
    let edited_handed = edit(&mut world, in_handed);
    assert_eq!(
        edited_handed.len(),
        1,
        "the old device edited alice's message in chat-handed, whose membership readers \
         serve as bob's: {edited_handed:?}"
    );
}
