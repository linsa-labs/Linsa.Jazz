//! What a write check's EXISTS scans on each generation, gated where the main gates reach it
//! only through the cost gate: the narrowing of a branch's scan to one equality, the residual
//! filter a covered scan still needs, and the table name an older generation stores rows
//! under. Each gate models a schema history the narrowing has to survive and is red at its
//! own assertion when the piece that keeps it sound is removed.

use super::*;

use crate::query_manager::relation_ir::{
    JoinCondition, JoinKind, KeyRef, ProjectColumn, ProjectExpr, RowIdRef,
};

const TEAMS: &str = "teams";
const TEAM_EDGES: &str = "team_edges";
const GRANTS: &str = "grants";
const DOCS: &str = "docs";
const RENAMED_MEMBERS: &str = "memberships";

/// One generation of a world: its structural schema, its permissions head, and the lens that
/// leads into it from the one before.
struct Generation {
    schema: Schema,
    permissions: Schema,
    lens: Option<Lens>,
}

fn open() -> TablePolicies {
    TablePolicies::new()
        .with_select(PolicyExpr::True)
        .with_insert(PolicyExpr::True)
        .with_update(Some(PolicyExpr::True), PolicyExpr::True)
        .with_delete(PolicyExpr::True)
}

/// `schema` with `policies(table)` on every table, open where it gives none.
fn head(schema: &Schema, policies: impl Fn(&str) -> Option<TablePolicies>) -> Schema {
    let mut head = schema.clone();
    for (name, table) in head.iter_mut() {
        table.policies = policies(name.as_str()).unwrap_or_else(open);
    }
    head
}

fn insert_only(rule: PolicyExpr) -> TablePolicies {
    TablePolicies::new()
        .with_select(PolicyExpr::True)
        .with_insert(rule)
}

fn add_column(table: &str, column: &str, default: &str) -> LensOp {
    LensOp::AddColumn {
        table: table.to_string(),
        column: column.to_string(),
        column_type: ColumnType::Text,
        default: text(default),
    }
}

fn lens(old: &Schema, new: &Schema, ops: Vec<LensOp>) -> Lens {
    Lens::new(
        SchemaHash::compute(old),
        SchemaHash::compute(new),
        LensTransform::with_ops(ops),
    )
}

/// A server and alice's device, rebuilt on each generation from their own stores, as
/// `World` does for the two generations of the main gates.
struct Stand {
    /// The session's user id: `alice`, or a uuid where rows refer to her by one.
    user: String,
    device: Device,
    server: Core,
    client: Core,
    server_id: ServerId,
    client_id: ClientId,
}

fn server_on(generation: &Generation, storage: SqliteStorage) -> Core {
    let mut core = runtime(
        generation.schema.clone(),
        storage,
        DurabilityTier::GlobalServer,
    );
    let manager = core.schema_manager_mut().query_manager_mut();
    manager.set_authorization_schema(generation.permissions.clone());
    manager.require_authorization_schema();
    core.immediate_tick();
    core
}

fn device_on(generation: &Generation, storage: SqliteStorage, device: Device) -> Core {
    let mut core = device_runtime(generation.schema.clone(), storage);
    if device == Device::ExplicitHead {
        core.schema_manager_mut()
            .query_manager_mut()
            .set_authorization_schema(generation.permissions.clone());
        core.immediate_tick();
    }
    core
}

impl Stand {
    fn new(first: &Generation, device: Device) -> Self {
        Self::new_as(first, device, ME)
    }

    fn new_as(first: &Generation, device: Device, user: &str) -> Self {
        let mut stand = Self {
            user: user.to_string(),
            device,
            server: server_on(first, sqlite()),
            client: device_on(first, sqlite(), device),
            server_id: ServerId::new(),
            client_id: ClientId::new(),
        };
        stand
            .server
            .add_client(stand.client_id, Some(Session::new(user)));
        stand.client.add_server(stand.server_id);
        stand.client.immediate_tick();
        stand.sync();
        stand
    }

    fn sync(&mut self) {
        pump(
            &mut self.server,
            self.server_id,
            &mut self.client,
            self.client_id,
        );
        let refusals = rejections(&mut self.server);
        assert!(
            refusals.is_empty(),
            "control: a seed write was refused ({refusals:?})"
        );
    }

    fn deploy(&mut self, next: &Generation) {
        let placeholder = || device_runtime(first_generation(), sqlite());
        let server_storage = std::mem::replace(&mut self.server, placeholder()).into_storage();
        let client_storage = std::mem::replace(&mut self.client, placeholder()).into_storage();
        self.server = server_on(next, server_storage);
        if let Some(lens) = &next.lens {
            self.server.publish_lens(lens).expect("lens publishes");
        }
        self.client = device_on(next, client_storage, self.device);
        self.server
            .add_client(self.client_id, Some(Session::new(&self.user)));
        self.client.add_server(self.server_id);
        self.server.immediate_tick();
        self.client.immediate_tick();
        self.sync();
        self.server.sync_sender().take();
        self.client.sync_sender().take();
    }

    fn alice(&self) -> WriteContext {
        WriteContext::from_session(Session::new(&self.user))
    }

    fn seed(&mut self, table: &str, values: Vec<(&str, Value)>) -> ObjectId {
        self.seed_as(table, None, values)
    }

    /// A row of `table` under `id` where one is given.
    fn seed_as(
        &mut self,
        table: &str,
        id: Option<ObjectId>,
        values: Vec<(&str, Value)>,
    ) -> ObjectId {
        let values = values
            .into_iter()
            .map(|(column, value)| (column.to_string(), value))
            .collect();
        let alice = self.alice();
        let ((id, _), _) = self
            .client
            .insert_with_id(table, values, id, Some(&alice))
            .unwrap_or_else(|error| panic!("control: seeding {table} was refused: {error:?}"));
        self.sync();
        id
    }

    fn update(&mut self, id: ObjectId, column: &str, value: Value) {
        let alice = self.alice();
        self.client
            .update(id, vec![(column.to_string(), value)], Some(&alice))
            .expect("control: the table is open");
        self.sync();
    }

    fn write(&mut self, table: &str, values: Vec<(&str, Value)>) -> Verdict {
        let values = values
            .into_iter()
            .map(|(column, value)| (column.to_string(), value))
            .collect();
        let alice = self.alice();
        let written = self
            .client
            .insert(table, values, Some(&alice))
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

    fn post(&mut self, chat: &str) -> Verdict {
        self.write(
            MESSAGES,
            vec![
                ("chatId", text(chat)),
                ("senderUserId", text(&self.user)),
                ("text", text("hello")),
            ],
        )
    }

    /// The rows of `table` the server serves alice, in the current layout.
    fn served(&mut self, table: &str) -> Vec<Vec<Value>> {
        let manager = self.server.schema_manager_mut().query_manager_mut();
        let sub = manager
            .subscribe_with_session(Query::new(table), Some(Session::new(&self.user)), None)
            .expect("the read subscribes");
        self.server.immediate_tick();
        let rows = self
            .server
            .schema_manager()
            .query_manager()
            .get_subscription_results(sub);
        self.server
            .schema_manager_mut()
            .query_manager_mut()
            .unsubscribe_with_sync(sub);
        self.server.sync_sender().take();
        rows.into_iter().map(|(_, values)| values).collect()
    }

    fn refused(&self, verdict: &Verdict) -> bool {
        match verdict {
            Verdict::Sent => false,
            Verdict::RefusedLocally(_) => self.device != Device::ServerJudges,
            Verdict::RejectedByServer(_) => self.device == Device::ServerJudges,
        }
    }
}

fn chat_tables(members: crate::query_manager::types::TableSchemaBuilder) -> Schema {
    SchemaBuilder::new()
        .table(members)
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
        .build()
}

fn members_with(columns: &[&str]) -> crate::query_manager::types::TableSchemaBuilder {
    columns.iter().fold(
        TableSchema::builder(MEMBERS)
            .column("chatId", ColumnType::Text)
            .column("userId", ColumnType::Text),
        |table, column| table.column(column, ColumnType::Text),
    )
}

/// The rules with `marker = value` FIRST in the membership condition, where the generation
/// has the column: a message only from a member of its chat; a note only from someone who
/// is no member of `banned` (an EXISTS under a NOT, as `idIsFree` and ban rules compile).
fn chat_rules(schema: &Schema, marker: Option<(&str, &str)>) -> Schema {
    let membership = |chat: PolicyValue| {
        let mut conditions = Vec::new();
        if let Some((column, value)) = marker {
            conditions.push(cmp(column, PolicyValue::Literal(text(value))));
        }
        conditions.push(cmp("chatId", chat));
        conditions.push(cmp("userId", me()));
        exists(MEMBERS, conditions)
    };
    head(schema, |table| match table {
        MESSAGES => Some(insert_only(PolicyExpr::and(vec![
            cmp("senderUserId", me()),
            membership(outer("chatId")),
        ]))),
        NOTES => Some(insert_only(PolicyExpr::and(vec![
            cmp("ownerUserId", me()),
            PolicyExpr::not(membership(PolicyValue::Literal(text("banned")))),
        ]))),
        _ => None,
    })
}

fn note() -> Vec<(&'static str, Value)> {
    vec![("ownerUserId", text(ME)), ("body", text("mine"))]
}

/// `status` dropped by one deployment and added back, `'active'` by default, by the next:
/// the third generation serves every membership written on the first as active, whatever
/// the first held. The first generation's index still lists them under their old status.
fn a_column_removed_and_added_back(device: Device) {
    let first = chat_tables(members_with(&["status"]));
    let second = chat_tables(members_with(&[]));
    // Not the first generation's layout to the byte: an identical schema is the first
    // generation's own branch again, which holds the rows as written.
    let mut third = first.clone();
    third.insert(
        TableName::new("agent_sessions"),
        TableSchema::builder("agent_sessions")
            .column("state", ColumnType::Text)
            .build(),
    );
    let generations = [
        Generation {
            permissions: chat_rules(&first, Some(("status", "active"))),
            schema: first.clone(),
            lens: None,
        },
        Generation {
            permissions: chat_rules(&second, None),
            lens: Some(lens(
                &first,
                &second,
                vec![LensOp::RemoveColumn {
                    table: MEMBERS.to_string(),
                    column: "status".to_string(),
                    column_type: ColumnType::Text,
                    default: text("active"),
                }],
            )),
            schema: second.clone(),
        },
        Generation {
            permissions: chat_rules(&third, Some(("status", "active"))),
            lens: Some(lens(
                &second,
                &third,
                vec![
                    add_column(MEMBERS, "status", "active"),
                    LensOp::AddTable {
                        table: "agent_sessions".to_string(),
                        schema: third[&TableName::new("agent_sessions")].clone(),
                    },
                ],
            )),
            schema: third,
        },
    ];
    let mut stand = Stand::new(&generations[0], device);
    for chat in ["chat-x", "banned"] {
        stand.seed(
            MEMBERS,
            vec![
                ("chatId", text(chat)),
                ("userId", text(ME)),
                ("status", text("inactive")),
            ],
        );
    }
    stand.deploy(&generations[1]);
    stand.deploy(&generations[2]);
    let served = stand.served(MEMBERS);
    assert!(
        served.len() == 2 && served.iter().all(|row| row.contains(&text("active"))),
        "control: the third generation must serve both memberships as active: {served:?}"
    );

    assert_eq!(
        stand.post("chat-x"),
        Verdict::Sent,
        "readers serve alice's membership of chat-x as active, yet her message was refused \
         ({device:?})"
    );
    let verdict = stand.write(NOTES, note());
    assert!(
        stand.refused(&verdict),
        "readers serve alice as an active member of `banned`, yet her note was not refused \
         ({device:?}): {verdict:?}"
    );
}

#[test]
fn a_column_removed_and_added_back_is_judged_by_its_new_value() {
    a_column_removed_and_added_back(Device::ServerJudges);
}

#[test]
fn a_column_removed_and_added_back_is_judged_by_its_new_value_on_the_phone() {
    a_column_removed_and_added_back(Device::ExplicitHead);
}

/// One deployment renames `kind` to `role` and adds a new `kind`, `'member'` by default: on
/// the first generation a column of that name holds the old values under another meaning.
#[test]
fn a_column_renamed_and_its_name_reused_is_judged_by_the_new_column() {
    let first = chat_tables(members_with(&["kind"]));
    let second = chat_tables(members_with(&["role", "kind"]));
    let generations = [
        Generation {
            permissions: chat_rules(&first, None),
            schema: first.clone(),
            lens: None,
        },
        Generation {
            permissions: chat_rules(&second, Some(("kind", "member"))),
            lens: Some(lens(
                &first,
                &second,
                vec![
                    LensOp::RenameColumn {
                        table: MEMBERS.to_string(),
                        old_name: "kind".to_string(),
                        new_name: "role".to_string(),
                    },
                    add_column(MEMBERS, "kind", "member"),
                ],
            )),
            schema: second,
        },
    ];
    let mut stand = Stand::new(&generations[0], Device::ServerJudges);
    stand.seed(
        MEMBERS,
        vec![
            ("chatId", text("chat-x")),
            ("userId", text(ME)),
            ("kind", text("owner")),
        ],
    );
    stand.deploy(&generations[1]);
    let served = stand.served(MEMBERS);
    assert!(
        served.len() == 1
            && served[0].contains(&text("member"))
            && served[0].contains(&text("owner")),
        "control: the second generation must serve kind 'member' and role 'owner': {served:?}"
    );

    assert_eq!(
        stand.post("chat-x"),
        Verdict::Sent,
        "readers serve alice's membership with kind 'member', yet her message was refused"
    );
}

/// The column a deployment added, first in the rule: the older generation has no such
/// column to narrow by, and its memberships still count.
#[test]
fn a_column_a_deployment_added_first_in_the_rule_hides_no_older_member() {
    let first = chat_tables(members_with(&[]));
    let second = chat_tables(members_with(&["status"]));
    let generations = [
        Generation {
            permissions: chat_rules(&first, None),
            schema: first.clone(),
            lens: None,
        },
        Generation {
            permissions: chat_rules(&second, Some(("status", "active"))),
            lens: Some(lens(
                &first,
                &second,
                vec![add_column(MEMBERS, "status", "active")],
            )),
            schema: second,
        },
    ];
    let mut stand = Stand::new(&generations[0], Device::ServerJudges);
    stand.seed(
        MEMBERS,
        vec![("chatId", text("chat-x")), ("userId", text(ME))],
    );
    stand.deploy(&generations[1]);

    assert_eq!(
        stand.post("chat-x"),
        Verdict::Sent,
        "the deployment marked alice's membership active, yet her message was refused"
    );
}

/// Memberships indexed by `userId` alone, `chatId` first in the rule: a branch is narrowed
/// by a column its index holds, or not at all.
#[test]
fn a_column_the_table_does_not_index_is_not_narrowed_by() {
    let members = || members_with(&[]).index_only(["userId"]);
    let first = chat_tables(members());
    let mut second = chat_tables(members());
    second.insert(
        TableName::new("agent_sessions"),
        TableSchema::builder("agent_sessions")
            .column("state", ColumnType::Text)
            .build(),
    );
    let generations = [
        Generation {
            permissions: chat_rules(&first, None),
            schema: first.clone(),
            lens: None,
        },
        Generation {
            permissions: chat_rules(&second, None),
            lens: Some(crate::schema_manager::auto_lens::generate_lens(
                &first, &second,
            )),
            schema: second,
        },
    ];
    let mut stand = Stand::new(&generations[0], Device::ServerJudges);
    stand.seed(
        MEMBERS,
        vec![("chatId", text("chat-x")), ("userId", text(ME))],
    );
    stand.deploy(&generations[1]);

    assert_eq!(
        stand.post("chat-x"),
        Verdict::Sent,
        "alice is a member of chat-x, yet her message was refused"
    );
}

/// `chat_members` renamed to `memberships`: the older generation stores the rows under the
/// old name. A rule with an equality to narrow by and one without must see the same rows.
fn a_renamed_table(narrowable: bool) {
    let first = chat_tables(members_with(&[]));
    let renamed = |schema: &Schema| {
        let mut schema = schema.clone();
        let table = schema
            .remove(&TableName::new(MEMBERS))
            .expect("the first generation has memberships");
        schema.insert(TableName::new(RENAMED_MEMBERS), table);
        schema
    };
    let second = renamed(&chat_tables(members_with(&[])));
    let membership = if narrowable {
        PolicyExpr::Exists {
            table: RENAMED_MEMBERS.into(),
            condition: Box::new(PolicyExpr::and(vec![
                cmp("chatId", outer("chatId")),
                cmp("userId", me()),
            ])),
        }
    } else {
        // A disjunction at the top: nothing to narrow by, every branch scanned whole.
        PolicyExpr::Exists {
            table: RENAMED_MEMBERS.into(),
            condition: Box::new(PolicyExpr::or(vec![
                PolicyExpr::and(vec![cmp("chatId", outer("chatId")), cmp("userId", me())]),
                PolicyExpr::and(vec![
                    cmp("chatId", PolicyValue::Literal(text("lobby"))),
                    cmp("userId", me()),
                ]),
            ])),
        }
    };
    let generations = [
        Generation {
            permissions: chat_rules(&first, None),
            schema: first.clone(),
            lens: None,
        },
        Generation {
            permissions: head(&second, |table| {
                (table == MESSAGES).then(|| {
                    insert_only(PolicyExpr::and(vec![
                        cmp("senderUserId", me()),
                        membership.clone(),
                    ]))
                })
            }),
            lens: Some(lens(
                &first,
                &second,
                vec![LensOp::RenameTable {
                    old_name: MEMBERS.to_string(),
                    new_name: RENAMED_MEMBERS.to_string(),
                }],
            )),
            schema: second,
        },
    ];
    let mut stand = Stand::new(&generations[0], Device::ServerJudges);
    stand.seed(
        MEMBERS,
        vec![("chatId", text("chat-x")), ("userId", text(ME))],
    );
    stand.deploy(&generations[1]);
    assert_eq!(
        stand.served(RENAMED_MEMBERS).len(),
        1,
        "control: readers serve the membership under the table's new name"
    );

    assert_eq!(
        stand.post("chat-x"),
        Verdict::Sent,
        "readers serve alice's membership under the new table name, yet her message was \
         refused (a rule {} an equality to narrow by)",
        if narrowable { "with" } else { "without" }
    );
}

#[test]
fn a_renamed_table_serves_older_rows_to_a_rule_it_narrows() {
    a_renamed_table(true);
}

#[test]
fn a_renamed_table_serves_older_rows_to_a_rule_it_scans_whole() {
    a_renamed_table(false);
}

fn team_tables(extra_table: bool) -> Schema {
    let builder = SchemaBuilder::new()
        .table(TableSchema::builder(TEAMS).column("name", ColumnType::Text))
        .table(
            TableSchema::builder(TEAM_EDGES)
                .column("child_team", ColumnType::Uuid)
                .column("parent_team", ColumnType::Uuid),
        )
        .table(
            TableSchema::builder(GRANTS)
                .column("team", ColumnType::Uuid)
                .column("userId", ColumnType::Text),
        )
        .table(
            TableSchema::builder(DOCS)
                .column("ownerId", ColumnType::Text)
                .column("title", ColumnType::Text),
        );
    if extra_table {
        builder
            .table(TableSchema::builder("agent_sessions").column("state", ColumnType::Text))
            .build()
    } else {
        builder.build()
    }
}

/// A doc only from someone granted a team the `home` team reaches through its parent
/// edges: `policy.exists(teams.where({ name: 'home' }).gather(parents).join(grants…))`,
/// the shape a recursive permission compiles to.
fn team_rules(schema: &Schema) -> Schema {
    let reachable_grant = reachable_grant();
    head(schema, |table| {
        (table == DOCS).then(|| {
            insert_only(PolicyExpr::and(vec![
                cmp("ownerId", me()),
                PolicyExpr::ExistsRel {
                    rel: reachable_grant.clone(),
                },
            ]))
        })
    })
}

fn reachable_grant() -> RelExpr {
    RelExpr::Project {
        input: Box::new(RelExpr::Filter {
            input: Box::new(RelExpr::Join {
                left: Box::new(RelExpr::Gather {
                    seed: Box::new(RelExpr::Filter {
                        input: Box::new(RelExpr::TableScan {
                            table: TableName::new(TEAMS),
                        }),
                        predicate: PredicateExpr::Cmp {
                            left: ColumnRef::scoped(TEAMS, "name"),
                            op: PredicateCmpOp::Eq,
                            right: ValueRef::Literal(text("home")),
                        },
                    }),
                    step: Box::new(RelExpr::Project {
                        input: Box::new(RelExpr::Join {
                            left: Box::new(RelExpr::Filter {
                                input: Box::new(RelExpr::TableScan {
                                    table: TableName::new(TEAM_EDGES),
                                }),
                                predicate: PredicateExpr::Cmp {
                                    left: ColumnRef::scoped(TEAM_EDGES, "child_team"),
                                    op: PredicateCmpOp::Eq,
                                    right: ValueRef::RowId(RowIdRef::Frontier),
                                },
                            }),
                            right: Box::new(RelExpr::TableScan {
                                table: TableName::new(TEAMS),
                            }),
                            on: vec![JoinCondition {
                                left: ColumnRef::scoped(TEAM_EDGES, "parent_team"),
                                right: ColumnRef::scoped("__recursive_hop_0", "id"),
                            }],
                            join_kind: JoinKind::Inner,
                        }),
                        columns: vec![ProjectColumn {
                            alias: "id".to_string(),
                            expr: ProjectExpr::Column(ColumnRef::scoped("__recursive_hop_0", "id")),
                        }],
                    }),
                    frontier_key: KeyRef::RowId(RowIdRef::Current),
                    max_depth: 5,
                    dedupe_key: vec![KeyRef::RowId(RowIdRef::Current)],
                }),
                right: Box::new(RelExpr::TableScan {
                    table: TableName::new(GRANTS),
                }),
                on: vec![JoinCondition {
                    left: ColumnRef::scoped(TEAMS, "id"),
                    right: ColumnRef::scoped("__hop_0", "team"),
                }],
                join_kind: JoinKind::Inner,
            }),
            predicate: PredicateExpr::Cmp {
                left: ColumnRef::scoped("__hop_0", "userId"),
                op: PredicateCmpOp::Eq,
                right: ValueRef::SessionRef(vec!["user_id".into()]),
            },
        }),
        columns: vec![ProjectColumn {
            alias: "id".to_string(),
            expr: ProjectExpr::Column(ColumnRef::scoped("__hop_0", "id")),
        }],
    }
}

/// The `home → granted` edge moved off `home` after a deployment: its first-generation copy
/// is still listed under `home` in that generation's index, and readers serve the edge as
/// leaving from elsewhere.
fn an_edge_moved_off_the_seed(deploy: bool, device: Device) {
    let first = team_tables(false);
    let second = team_tables(true);
    let generations = [
        Generation {
            permissions: team_rules(&first),
            schema: first.clone(),
            lens: None,
        },
        Generation {
            permissions: team_rules(&second),
            lens: Some(crate::schema_manager::auto_lens::generate_lens(
                &first, &second,
            )),
            schema: second,
        },
    ];
    let mut stand = Stand::new(&generations[0], device);
    let team = |stand: &mut Stand, name: &str| stand.seed(TEAMS, vec![("name", text(name))]);
    let home = team(&mut stand, "home");
    let granted = team(&mut stand, "granted");
    let elsewhere = team(&mut stand, "elsewhere");
    let edge = stand.seed(
        TEAM_EDGES,
        vec![
            ("child_team", Value::Uuid(home)),
            ("parent_team", Value::Uuid(granted)),
        ],
    );
    stand.seed(
        GRANTS,
        vec![("team", Value::Uuid(granted)), ("userId", text(ME))],
    );
    let doc = |title: &str| vec![("ownerId", text(ME)), ("title", text(title))];
    if deploy {
        stand.deploy(&generations[1]);
    }
    assert_eq!(
        stand.write(DOCS, doc("first")),
        Verdict::Sent,
        "control: home reaches the granted team ({device:?}, deployed: {deploy})"
    );

    stand.update(edge, "child_team", Value::Uuid(elsewhere));
    let verdict = stand.write(DOCS, doc("second"));
    assert!(
        stand.refused(&verdict),
        "the edge leaves from another team now and home reaches no grant, yet alice's doc \
         was not refused ({device:?}, deployed: {deploy}): {verdict:?}"
    );
}

#[test]
#[ignore = "a recursive EXISTS grants nothing even on one generation (the control is red \
            with no deployment); the residual its step needs across generations (R1/R2) waits \
            on that"]
fn an_edge_moved_off_the_seed_team_grants_nothing() {
    an_edge_moved_off_the_seed(false, Device::ServerJudges);
}

#[test]
#[ignore = "a recursive EXISTS grants nothing even on one generation (the control is red \
            with no deployment); the residual its step needs across generations (R1/R2) waits \
            on that"]
fn an_edge_moved_off_the_seed_team_after_a_deployment_grants_nothing() {
    an_edge_moved_off_the_seed(true, Device::ServerJudges);
}

#[test]
#[ignore = "a recursive EXISTS grants nothing even on one generation (the control is red \
            with no deployment); the residual its step needs across generations (R1/R2) waits \
            on that"]
fn an_edge_moved_off_the_seed_team_after_a_deployment_grants_nothing_on_the_phone() {
    an_edge_moved_off_the_seed(true, Device::ExplicitHead);
}

const ADMINS: &str = "admins";
const AGENT_SESSIONS: &str = "agent_sessions";

/// A user id as the app's sessions carry one: a uuid, which rows that refer to the user hold
/// as a uuid value. The session carries it as text all the same.
fn uuid_user() -> (String, ObjectId) {
    let id = ObjectId::new();
    (id.to_string(), id)
}

/// `schema` with one more table: a deployment that leaves the tables of the rules as they are.
fn with_agent_sessions(schema: &Schema) -> Schema {
    let mut schema = schema.clone();
    schema.insert(
        TableName::new(AGENT_SESSIONS),
        TableSchema::builder(AGENT_SESSIONS)
            .column("state", ColumnType::Text)
            .build(),
    );
    schema
}

fn add_agent_sessions(schema: &Schema) -> LensOp {
    LensOp::AddTable {
        table: AGENT_SESSIONS.to_string(),
        schema: schema[&TableName::new(AGENT_SESSIONS)].clone(),
    }
}

/// The two generations of `first` and `first` with one more table, each under `rules`.
fn another_table(first: Schema, rules: impl Fn(&Schema) -> Schema) -> [Generation; 2] {
    let second = with_agent_sessions(&first);
    [
        Generation {
            permissions: rules(&first),
            schema: first.clone(),
            lens: None,
        },
        Generation {
            permissions: rules(&second),
            lens: Some(lens(&first, &second, vec![add_agent_sessions(&second)])),
            schema: second,
        },
    ]
}

/// Memberships whose `userId` is a uuid column, as `schema.ref('users')` declares it.
fn ref_tables() -> Schema {
    chat_tables(
        TableSchema::builder(MEMBERS)
            .column("chatId", ColumnType::Text)
            .column("userId", ColumnType::Uuid),
    )
}

/// The rules of `chat_rules`, the user FIRST in the membership condition: what the scan is
/// narrowed by is the session's user id, text, against a uuid column.
fn user_first_rules(schema: &Schema) -> Schema {
    let membership =
        |chat: PolicyValue| exists(MEMBERS, vec![cmp("userId", me()), cmp("chatId", chat)]);
    head(schema, |table| match table {
        MESSAGES => Some(insert_only(PolicyExpr::and(vec![
            cmp("senderUserId", me()),
            membership(outer("chatId")),
        ]))),
        NOTES => Some(insert_only(PolicyExpr::and(vec![
            cmp("ownerUserId", me()),
            PolicyExpr::not(membership(PolicyValue::Literal(text("banned")))),
        ]))),
        _ => None,
    })
}

/// Alice's memberships of chat-x and `banned`, referring to her by her uuid, written before
/// the deployment where there is one.
fn a_member_by_uuid(deploy: bool, device: Device) -> Stand {
    let (user, id) = uuid_user();
    let generations = another_table(ref_tables(), user_first_rules);
    let mut stand = Stand::new_as(&generations[0], device, &user);
    for chat in ["chat-x", "banned"] {
        stand.seed(
            MEMBERS,
            vec![("chatId", text(chat)), ("userId", Value::Uuid(id))],
        );
    }
    if deploy {
        stand.deploy(&generations[1]);
    }
    let verdict = stand.post("chat-none");
    assert!(
        stand.refused(&verdict),
        "control: alice is no member of chat-none, yet her message was not refused \
         ({device:?}): {verdict:?}"
    );
    stand
}

fn a_ref_column_admits(deploy: bool, device: Device) {
    let mut stand = a_member_by_uuid(deploy, device);
    assert_eq!(
        stand.post("chat-x"),
        Verdict::Sent,
        "alice's membership of chat-x refers to her by her uuid, yet her message was refused \
         ({device:?}, {})",
        if deploy { "deployed" } else { "one generation" }
    );
}

fn a_ref_column_bans(deploy: bool, device: Device) {
    let mut stand = a_member_by_uuid(deploy, device);
    let owner = stand.user.clone();
    let verdict = stand.write(
        NOTES,
        vec![("ownerUserId", text(&owner)), ("body", text("mine"))],
    );
    assert!(
        stand.refused(&verdict),
        "alice is a member of `banned` by her uuid, yet her note was not refused ({device:?}, \
         {}): {verdict:?}",
        if deploy { "deployed" } else { "one generation" }
    );
}

#[test]
fn a_ref_column_admits_the_session_user_it_names_by_uuid() {
    a_ref_column_admits(false, Device::ServerJudges);
}

#[test]
fn a_ref_column_admits_the_session_user_it_names_by_uuid_after_a_deployment() {
    a_ref_column_admits(true, Device::ServerJudges);
}

#[test]
fn a_ref_column_admits_the_session_user_it_names_by_uuid_after_a_deployment_on_the_phone() {
    a_ref_column_admits(true, Device::ExplicitHead);
}

#[test]
fn a_ref_column_bans_the_session_user_it_names_by_uuid() {
    a_ref_column_bans(false, Device::ServerJudges);
}

#[test]
fn a_ref_column_bans_the_session_user_it_names_by_uuid_after_a_deployment() {
    a_ref_column_bans(true, Device::ServerJudges);
}

/// A rule over the row id a table does not declare: an admin is a row of `admins` under the
/// session user's own id.
fn the_row_id(deploy: bool) {
    let (user, id) = uuid_user();
    let tables = || {
        let mut schema = chat_tables(members_with(&[]));
        schema.insert(
            TableName::new(ADMINS),
            TableSchema::builder(ADMINS)
                .column("label", ColumnType::Text)
                .build(),
        );
        schema
    };
    let rules = |schema: &Schema| {
        head(schema, |table| {
            (table == MESSAGES).then(|| {
                insert_only(PolicyExpr::and(vec![
                    cmp("senderUserId", me()),
                    exists(ADMINS, vec![cmp("id", me())]),
                ]))
            })
        })
    };
    let generations = another_table(tables(), rules);
    let mut stand = Stand::new_as(&generations[0], Device::ServerJudges, &user);
    stand.seed_as(
        ADMINS,
        Some(ObjectId::new()),
        vec![("label", text("someone"))],
    );
    let verdict = stand.post("chat-x");
    assert!(
        stand.refused(&verdict),
        "control: alice is no admin, yet her message was not refused: {verdict:?}"
    );
    stand.seed_as(ADMINS, Some(id), vec![("label", text("alice"))]);
    if deploy {
        stand.deploy(&generations[1]);
    }

    assert_eq!(
        stand.post("chat-x"),
        Verdict::Sent,
        "alice's admin row has her uuid for its id, yet her message was refused ({})",
        if deploy { "deployed" } else { "one generation" }
    );
}

#[test]
fn the_row_id_admits_the_session_user_it_names() {
    the_row_id(false);
}

#[test]
fn the_row_id_admits_the_session_user_it_names_after_a_deployment() {
    the_row_id(true);
}

/// A deployment that changes `userId` from text to a uuid column, by a lens with no step for
/// it: a write check grants a membership exactly where readers serve one.
#[test]
fn a_column_whose_type_a_deployment_changed_is_judged_as_readers_serve_it() {
    let (user, id) = uuid_user();
    let first = chat_tables(members_with(&[]));
    let second = with_agent_sessions(&ref_tables());
    let generations = [
        Generation {
            permissions: user_first_rules(&first),
            schema: first.clone(),
            lens: None,
        },
        Generation {
            permissions: user_first_rules(&second),
            lens: Some(lens(&first, &second, vec![add_agent_sessions(&second)])),
            schema: second,
        },
    ];
    let mut stand = Stand::new_as(&generations[0], Device::ServerJudges, &user);
    stand.seed(
        MEMBERS,
        vec![("chatId", text("chat-x")), ("userId", text(&user))],
    );
    stand.deploy(&generations[1]);
    let served = stand.served(MEMBERS);
    eprintln!("a membership written as text, served after the type change: {served:?}");
    let member = served.iter().any(|row| {
        row.contains(&text("chat-x"))
            && (row.contains(&Value::Uuid(id)) || row.contains(&text(&user)))
    });

    let verdict = stand.post("chat-x");
    assert_eq!(
        verdict == Verdict::Sent,
        member,
        "readers {} alice's membership of chat-x, yet her message was {verdict:?}",
        if member { "serve" } else { "do not serve" }
    );
}

/// `chat_members` renamed to `memberships`, and a new `chat_members` added under the old
/// name, in the old table's layout or with one more column: the older generation's
/// `chat_members` is the renamed table, whose rows readers serve as memberships and never
/// as rows of the new table.
fn a_table_name_a_rename_freed(same_layout: bool) {
    let new_columns: &[&str] = if same_layout { &[] } else { &["invitedBy"] };
    let first = chat_tables(members_with(&[]));
    let mut second = first.clone();
    let renamed = second
        .remove(&TableName::new(MEMBERS))
        .expect("the first generation has memberships");
    second.insert(TableName::new(RENAMED_MEMBERS), renamed);
    second.insert(TableName::new(MEMBERS), members_with(new_columns).build());
    let generations = [
        Generation {
            permissions: chat_rules(&first, None),
            schema: first.clone(),
            lens: None,
        },
        Generation {
            permissions: chat_rules(&second, None),
            lens: Some(lens(
                &first,
                &second,
                vec![
                    LensOp::RenameTable {
                        old_name: MEMBERS.to_string(),
                        new_name: RENAMED_MEMBERS.to_string(),
                    },
                    LensOp::AddTable {
                        table: MEMBERS.to_string(),
                        schema: second[&TableName::new(MEMBERS)].clone(),
                    },
                ],
            )),
            schema: second,
        },
    ];
    let mut stand = Stand::new(&generations[0], Device::ServerJudges);
    stand.seed(
        MEMBERS,
        vec![("chatId", text("chat-x")), ("userId", text(ME))],
    );
    stand.deploy(&generations[1]);
    assert!(
        stand.served(RENAMED_MEMBERS).len() == 1 && stand.served(MEMBERS).is_empty(),
        "control: readers serve the old membership as a row of `memberships` alone"
    );

    let verdict = stand.post("chat-x");
    assert!(
        stand.refused(&verdict),
        "readers serve no row of the new `chat_members`, yet alice's message was not \
         refused ({} layout): {verdict:?}",
        if same_layout {
            "the old table's"
        } else {
            "another"
        }
    );
    let mut joined = vec![("chatId", text("chat-x")), ("userId", text(ME))];
    if !same_layout {
        joined.push(("invitedBy", text("bob")));
    }
    stand.seed(MEMBERS, joined);
    assert_eq!(
        stand.post("chat-x"),
        Verdict::Sent,
        "control: alice joined the new `chat_members`, yet her message was refused"
    );
}

#[test]
fn a_table_name_a_rename_freed_and_a_new_table_took_holds_no_older_rows() {
    a_table_name_a_rename_freed(true);
}

#[test]
fn a_table_name_a_rename_freed_and_a_new_table_took_in_another_layout_holds_no_older_rows() {
    a_table_name_a_rename_freed(false);
}
