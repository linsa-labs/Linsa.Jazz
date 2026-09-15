//! Randomized differential for subscriptions filtered by explicit authorization whose policy
//! reads a table their graph never names, and for subscriptions left clean once nothing changes.
//!
//! The select policy of `teams` reads `user_team_edges`. Every path that changes that table has
//! to mark such subscriptions for a visibility recompute — a local write, a pending overlay, the
//! retraction of a write the server rejected — or the subscription keeps a verdict the data no
//! longer supports. Until 2026-09-15 a `has_pending_local_updates` flag that an empty delta never
//! cleared re-settled most of these subscriptions on every pass. That hid the missing marks, and
//! on the phone it cost a re-authorization of every such subscription per keystroke.
//!
//! Hand-written gates pin the sequences we already thought of; this exists for the rest. Three
//! properties are asserted whenever the link has gone quiet:
//!
//! * **Incremental equals fresh** — every long-lived subscription holds exactly the rows a
//!   subscription created now, over the same query and session, settles to.
//! * **Model** — alice sees the teams her surviving edges point at; bob, whose every edge the
//!   server rejects, sees none.
//! * **Clean** — no subscription is still waiting for a settle. A leftover re-settles and
//!   re-authorizes on every later `process()` for nothing.
//! * **Server scope** — the scope the server derives from alice's synced subscriptions holds
//!   exactly the teams she sees. A server subscription re-derives it only when marked, and a
//!   grant through `user_team_edges` touches nothing its graph reads.
//!
//! Alice's subscriptions go upstream, as the app's do; bob's stay local, because the server
//! knows the connection as alice.
//!
//! The server knows the connection as alice. The client writes edges as alice (accepted) and as
//! bob (accepted locally, rejected upstream, retracted), moves and deletes alice's edges, deletes
//! bob's pending edges, renames teams, and writes a table no policy reads. Network steps
//! interleave with the writes, so fates arrive while other local writes are still pending.
//!
//! One sequence is excluded, and not by preference: a delete that goes up in the same frame as an
//! earlier write to its row is refused upstream for "missing row content" — the server reads the
//! delete's old content when it queues the check, before the earlier write's own check has
//! applied it — and the row comes back on both nodes. That defect predates this oracle and is
//! tracked on its own; deletes here only target rows whose last write has already gone up.
//!
//! And one blind spot is deliberate. A query the client withdraws keeps its scope on the server:
//! `process_pending_query_unsubscriptions` removes the server subscription but never calls
//! `drop_client_query_subscription`, so the scope of every fresh subscription a check opens and
//! closes stays behind, frozen. The server-scope property therefore looks only at the queries of
//! live subscriptions. That defect predates this oracle too and is tracked on its own.
//!
//! Backend: `SqliteStorage` on both nodes, as on the phone.

use super::*;
use crate::batch_fate::BatchFate;
use crate::query_manager::relation_ir::{
    ColumnRef, PredicateCmpOp, PredicateExpr, RelExpr, RowIdRef, ValueRef,
};
use crate::storage::SqliteStorage;
use crate::sync_manager::QueryId;
use std::collections::{BTreeMap, BTreeSet, HashSet};

type Node = RuntimeCore<SqliteStorage, NoopScheduler>;

const SEEDS: [u64; 6] = [
    0x9011_C7DE_0000_0001,
    0x9011_C7DE_0000_0002,
    0x9011_C7DE_0000_0003,
    0x9011_C7DE_0000_0004,
    0x9011_C7DE_0000_0005,
    0x9011_C7DE_0000_0006,
];

const OPS_PER_SEED: usize = 120;
const CHECK_EVERY: usize = 20;
const TEAM_NAMES: [&str; 3] = ["red", "green", "blue"];

struct Xorshift(u64);

impl Xorshift {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }
}

fn structural_schema() -> Schema {
    SchemaBuilder::new()
        .table(TableSchema::builder("teams").column("name", ColumnType::Text))
        .table(
            TableSchema::builder("user_team_edges")
                .column("user_id", ColumnType::Text)
                .column("team_id", ColumnType::Uuid),
        )
        .table(TableSchema::builder("notes").column("body", ColumnType::Text))
        .build()
}

fn own_edge() -> PolicyExpr {
    PolicyExpr::eq_session("user_id", vec!["user_id".into()])
}

/// A team is visible when an edge for the session's user points at it. Edges are written,
/// moved and deleted only by their own user. Nothing reads `notes`.
fn authorization_schema() -> Schema {
    let team_select_policy = PolicyExpr::ExistsRel {
        rel: RelExpr::Filter {
            input: Box::new(RelExpr::TableScan {
                table: TableName::new("user_team_edges"),
            }),
            predicate: PredicateExpr::And(vec![
                PredicateExpr::Cmp {
                    left: ColumnRef::scoped("user_team_edges", "user_id"),
                    op: PredicateCmpOp::Eq,
                    right: ValueRef::SessionRef(vec!["user_id".into()]),
                },
                PredicateExpr::Cmp {
                    left: ColumnRef::scoped("user_team_edges", "team_id"),
                    op: PredicateCmpOp::Eq,
                    right: ValueRef::RowId(RowIdRef::Outer),
                },
            ]),
        },
    };

    SchemaBuilder::new()
        .table(
            TableSchema::builder("teams")
                .column("name", ColumnType::Text)
                .policies(
                    TablePolicies::new()
                        .with_select(team_select_policy)
                        .with_insert(PolicyExpr::True)
                        .with_update(Some(PolicyExpr::True), PolicyExpr::True),
                ),
        )
        .table(
            TableSchema::builder("user_team_edges")
                .column("user_id", ColumnType::Text)
                .column("team_id", ColumnType::Uuid)
                .policies(
                    TablePolicies::new()
                        .with_insert(own_edge())
                        .with_update(Some(own_edge()), own_edge())
                        .with_delete(own_edge()),
                ),
        )
        .table(
            TableSchema::builder("notes")
                .column("body", ColumnType::Text)
                .policies(
                    TablePolicies::new()
                        .with_select(PolicyExpr::True)
                        .with_insert(PolicyExpr::True)
                        .with_update(Some(PolicyExpr::True), PolicyExpr::True),
                ),
        )
        .build()
}

fn node(role: &str, seed: u64, paths: &mut Vec<std::path::PathBuf>) -> Node {
    let path = std::env::temp_dir().join(format!(
        "jazz-policy-dependency-{role}-{seed:x}-{}.sqlite",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);
    let storage = SqliteStorage::open(&path).expect("sqlite storage should open");
    paths.push(path);

    let app_id = AppId::from_name("policy-dependency-differential");
    let schema_manager = SchemaManager::new(
        SyncManager::new(),
        structural_schema(),
        app_id,
        "dev",
        "main",
    )
    .unwrap();
    let mut core = new_test_core(schema_manager, storage, NoopScheduler);
    core.immediate_tick();
    core.schema_manager_mut()
        .query_manager_mut()
        .set_authorization_schema(authorization_schema());
    core
}

struct Link {
    client: Node,
    server: Node,
    client_id: ClientId,
    server_id: ServerId,
    rejected_fates: usize,
    /// `code: reason` of every rejection the server sent, with counts.
    rejections: BTreeMap<String, usize>,
    /// How many times the client's queue went up; a write made before the n-th `up` shipped
    /// with it.
    ups: usize,
}

impl Link {
    /// Hands the server everything the client has queued and lets the server act on it.
    fn up(&mut self) -> bool {
        self.client.batched_tick();
        let mut any = false;
        for entry in self.client.sync_sender().take() {
            if entry.destination == Destination::Server(self.server_id) {
                any = true;
                self.server.park_sync_message(InboxEntry {
                    source: Source::Client(self.client_id),
                    payload: entry.payload,
                });
            }
        }
        self.server.batched_tick();
        self.server.immediate_tick();
        self.ups += 1;
        any
    }

    /// Parks everything the server has queued for the client; the client acts on it at its
    /// next tick, which may come after further local writes.
    fn down(&mut self) -> bool {
        self.server.batched_tick();
        let mut any = false;
        for entry in self.server.sync_sender().take() {
            if entry.destination != Destination::Client(self.client_id) {
                continue;
            }
            if let SyncPayload::BatchFate {
                fate: BatchFate::Rejected { code, reason, .. },
            } = &entry.payload
            {
                self.rejected_fates += 1;
                *self
                    .rejections
                    .entry(format!("{code}: {reason}"))
                    .or_default() += 1;
            }
            any = true;
            self.client.park_sync_message(InboxEntry {
                source: Source::Server(self.server_id),
                payload: entry.payload,
            });
        }
        any
    }

    fn settle(&mut self) {
        let mut quiet_rounds = 0;
        for _ in 0..60 {
            let up = self.up();
            let down = self.down();
            self.client.batched_tick();
            self.client.immediate_tick();
            quiet_rounds = if up || down { 0 } else { quiet_rounds + 1 };
            if quiet_rounds == 2 {
                return;
            }
        }
        panic!("the link never went quiet");
    }
}

#[derive(Default)]
struct Model {
    teams: Vec<(ObjectId, &'static str)>,
    /// `(edge, team)`, accepted upstream.
    alice_edges: Vec<(ObjectId, ObjectId)>,
    /// Accepted locally, rejected upstream.
    bob_edges: Vec<ObjectId>,
    notes: Vec<ObjectId>,
    /// `Link::ups` at each edge's last write; the write has shipped once `ups` moved past it.
    last_write: HashMap<ObjectId, usize>,
    /// `team#N`, `a#N`, `b#N` — stable names for failure reports.
    labels: HashMap<ObjectId, String>,
}

impl Model {
    fn label(&self, id: ObjectId) -> String {
        self.labels
            .get(&id)
            .cloned()
            .unwrap_or_else(|| "?".to_string())
    }

    fn name(&mut self, id: ObjectId, prefix: &str) -> String {
        let label = format!("{prefix}#{}", self.labels.len());
        self.labels.insert(id, label.clone());
        label
    }

    fn visible(&self, session: &str, name: Option<&str>) -> Vec<ObjectId> {
        if session != "alice" {
            return Vec::new();
        }
        let mut ids: Vec<ObjectId> = self
            .teams
            .iter()
            .filter(|(team, team_name)| {
                name.is_none_or(|name| name == *team_name)
                    && self.alice_edges.iter().any(|(_, granted)| granted == team)
            })
            .map(|(team, _)| *team)
            .collect();
        ids.sort();
        ids
    }
}

struct Watch {
    session: &'static str,
    name: Option<&'static str>,
    synced: bool,
    sub: QuerySubscriptionId,
}

fn teams_query(name: Option<&str>) -> Query {
    let builder = QueryBuilder::new("teams");
    match name {
        Some(name) => builder.filter_eq("name", Value::Text(name.into())).build(),
        None => builder.build(),
    }
}

fn subscribe(
    client: &mut Node,
    session: &str,
    name: Option<&str>,
    synced: bool,
) -> QuerySubscriptionId {
    let query_manager = client.schema_manager_mut().query_manager_mut();
    let session = Some(Session::new(session));
    let subscribed = if synced {
        query_manager.subscribe_with_sync(teams_query(name), session, None)
    } else {
        query_manager.subscribe_with_session(teams_query(name), session, None)
    };
    subscribed.expect("a session may subscribe to teams")
}

/// Whether any of `queries` holds `row` in the scope the server keeps for the client.
fn in_live_server_scope(
    server: &Node,
    client_id: ClientId,
    queries: &[QueryId],
    row: ObjectId,
) -> bool {
    let branch = server.schema_manager().branch_name();
    server
        .schema_manager()
        .query_manager()
        .sync_manager()
        .get_client(client_id)
        .is_some_and(|client| {
            queries.iter().any(|query_id| {
                client
                    .queries
                    .get(query_id)
                    .is_some_and(|query| query.scope.contains(&(row, branch)))
            })
        })
}

fn ids(client: &Node, sub: QuerySubscriptionId) -> Vec<ObjectId> {
    let mut ids: Vec<ObjectId> = client
        .schema_manager()
        .query_manager()
        .get_subscription_results(sub)
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    ids.sort();
    ids
}

fn text(value: &str) -> Value {
    Value::Text(value.into())
}

fn apply_op(link: &mut Link, model: &mut Model, rng: &mut Xorshift, log: &mut Vec<String>) {
    let alice = WriteContext::from_session(Session::new("alice"));
    let bob = WriteContext::from_session(Session::new("bob"));
    let client = &mut link.client;
    let roll = rng.below(100);

    if model.teams.is_empty() || roll < 12 {
        let name = TEAM_NAMES[rng.below(TEAM_NAMES.len())];
        let ((team, _), _) = client
            .insert(
                "teams",
                HashMap::from([("name".to_string(), text(name))]),
                Some(&alice),
            )
            .expect("alice may create a team");
        model.teams.push((team, name));
        let label = model.name(team, "team");
        log.push(format!("insert {label} {name}"));
        return;
    }

    let team_index = rng.below(model.teams.len());
    let team = model.teams[team_index].0;
    let roll = if model.alice_edges.is_empty() && (57..=76).contains(&roll) {
        22
    } else {
        roll
    };

    match roll {
        12..=21 => {
            let name = TEAM_NAMES[rng.below(TEAM_NAMES.len())];
            client
                .update(team, vec![("name".to_string(), text(name))], Some(&alice))
                .expect("alice may rename a team");
            model.teams[team_index].1 = name;
            log.push(format!("rename {} {name}", model.label(team)));
        }
        22..=41 => {
            let ((edge, _), _) = client
                .insert(
                    "user_team_edges",
                    HashMap::from([
                        ("user_id".to_string(), text("alice")),
                        ("team_id".to_string(), Value::Uuid(team)),
                    ]),
                    Some(&alice),
                )
                .expect("alice may grant herself a team");
            model.alice_edges.push((edge, team));
            model.last_write.insert(edge, link.ups);
            let label = model.name(edge, "a");
            log.push(format!("insert {label} alice -> {}", model.label(team)));
        }
        42..=56 => {
            let ((edge, _), _) = client
                .insert(
                    "user_team_edges",
                    HashMap::from([
                        ("user_id".to_string(), text("bob")),
                        ("team_id".to_string(), Value::Uuid(team)),
                    ]),
                    Some(&bob),
                )
                .expect("bob's own policy admits his edge locally");
            model.bob_edges.push(edge);
            let label = model.name(edge, "b");
            log.push(format!("insert {label} bob -> {}", model.label(team)));
        }
        57..=66 => {
            // Only rows whose last write already went up; see the module note.
            let shipped: Vec<usize> = (0..model.alice_edges.len())
                .filter(|index| model.last_write[&model.alice_edges[*index].0] < link.ups)
                .collect();
            if shipped.is_empty() {
                log.push("delete skipped: every alice edge has a write still queued".to_string());
                return;
            }
            let index = shipped[rng.below(shipped.len())];
            let (edge, _) = model.alice_edges.remove(index);
            client
                .delete(edge, Some(&alice))
                .expect("alice may delete her own edge");
            log.push(format!("delete {}", model.label(edge)));
        }
        67..=76 => {
            let index = rng.below(model.alice_edges.len());
            let edge = model.alice_edges[index].0;
            client
                .update(
                    edge,
                    vec![("team_id".to_string(), Value::Uuid(team))],
                    Some(&alice),
                )
                .expect("alice may move her own edge");
            let from = model.label(model.alice_edges[index].1);
            model.alice_edges[index].1 = team;
            model.last_write.insert(edge, link.ups);
            log.push(format!(
                "move {} {from} -> {}",
                model.label(edge),
                model.label(team)
            ));
        }
        77..=84 if !model.bob_edges.is_empty() => {
            let index = rng.below(model.bob_edges.len());
            let edge = model.bob_edges.remove(index);
            // The edge may already be retracted, and then there is nothing to delete.
            let outcome = client.delete(edge, Some(&bob));
            log.push(format!(
                "delete {}: {}",
                model.label(edge),
                if outcome.is_ok() {
                    "written"
                } else {
                    "refused"
                }
            ));
        }
        _ => {
            let body = format!("beat {}", rng.below(1000));
            if model.notes.is_empty() || rng.below(4) == 0 {
                let ((note, _), _) = client
                    .insert(
                        "notes",
                        HashMap::from([("body".to_string(), text(&body))]),
                        Some(&alice),
                    )
                    .expect("anyone may write a note");
                model.notes.push(note);
                log.push("insert note".to_string());
            } else {
                let note = model.notes[rng.below(model.notes.len())];
                client
                    .update(note, vec![("body".to_string(), text(&body))], Some(&alice))
                    .expect("anyone may update a note");
                log.push("update note".to_string());
            }
        }
    }
}

fn network_step(link: &mut Link, rng: &mut Xorshift, log: &mut Vec<String>) {
    let step = match rng.below(8) {
        0 => {
            link.up();
            "up"
        }
        1 => {
            link.down();
            "down"
        }
        2 => {
            link.client.immediate_tick();
            "client immediate_tick"
        }
        3 => {
            link.client.batched_tick();
            "client batched_tick"
        }
        4 => {
            link.up();
            link.down();
            link.client.batched_tick();
            link.client.immediate_tick();
            "round trip"
        }
        _ => return,
    };
    log.push(format!("  net: {step}"));
}

fn check(
    link: &mut Link,
    model: &Model,
    watches: &[Watch],
    seed: u64,
    step: usize,
    log: &[String],
) {
    link.settle();
    let context = || {
        format!(
            "seed {seed:#x}, step {step}; last ops:\n    {}",
            log[log.len().saturating_sub(16)..].join("\n    ")
        )
    };
    // Diagnostic knob: assert the model alone, e.g. with the fix disarmed, to tell a defect the
    // fix introduced from one it merely stopped hiding.
    let model_alone = std::env::var_os("POLICY_DIFF_MODEL_ONLY").is_some();

    let waiting = link
        .client
        .schema_manager()
        .query_manager()
        .subscriptions_awaiting_settle();
    assert!(
        model_alone || waiting.is_empty(),
        "{} subscription(s) still wait for a settle on a quiet link, so every later process() \
         re-settles and re-authorizes them for nothing ({})",
        waiting.len(),
        context()
    );

    // The server keeps a scope per query. Over alice's live synced subscriptions — all teams and
    // the red ones — their union is everything alice sees. Queries withdrawn earlier are left
    // out on purpose; see the module note.
    let alice_visible = model.visible("alice", None);
    let live_queries: Vec<QueryId> = watches
        .iter()
        .filter(|watch| watch.synced)
        .map(|watch| QueryId(watch.sub.0))
        .collect();
    for (team, _) in &model.teams {
        let in_scope = in_live_server_scope(&link.server, link.client_id, &live_queries, *team);
        if !model_alone && in_scope != alice_visible.contains(team) {
            report_scope_mismatch(link, model, watches, *team, in_scope, log, context());
        }
    }

    for watch in watches {
        let held = ids(&link.client, watch.sub);
        let fresh_sub = subscribe(&mut link.client, watch.session, watch.name, watch.synced);
        if watch.synced {
            link.settle();
        } else {
            link.client.immediate_tick();
        }
        let fresh = ids(&link.client, fresh_sub);
        link.client
            .schema_manager_mut()
            .query_manager_mut()
            .unsubscribe_with_sync(fresh_sub);

        if !model_alone {
            assert_eq!(
                held,
                fresh,
                "{} {:?}: the long-lived subscription disagrees with one created now over the \
                 same data, so some change never marked it ({})",
                watch.session,
                watch.name,
                context()
            );
        }

        let expected = model.visible(watch.session, watch.name);
        if fresh != expected {
            report_model_mismatch(link, model, watch, &fresh, &expected, log, context());
        }
    }
    link.settle();
}

fn report_model_mismatch(
    link: &mut Link,
    model: &Model,
    watch: &Watch,
    fresh: &[ObjectId],
    expected: &[ObjectId],
    log: &[String],
    context: String,
) -> ! {
    let engine_only: Vec<String> = fresh
        .iter()
        .filter(|id| !expected.contains(id))
        .map(|id| model.label(*id))
        .collect();
    let model_only: Vec<String> = expected
        .iter()
        .filter(|id| !fresh.contains(id))
        .map(|id| model.label(*id))
        .collect();
    let client_edges = edges(&mut link.client, model);
    let server_edges = edges(&mut link.server, model);

    let mut involved: HashSet<String> = engine_only.iter().chain(&model_only).cloned().collect();
    for (edge, _, team) in client_edges.iter().chain(&server_edges) {
        if engine_only.contains(team) || model_only.contains(team) {
            involved.insert(edge.clone());
        }
    }
    let history = history(log, &involved);
    let render = |rows: &[(String, String, String)]| {
        rows.iter()
            .map(|(edge, user, team)| format!("{edge}({user}->{team})"))
            .collect::<Vec<_>>()
            .join(" ")
    };
    let model_edges = model
        .alice_edges
        .iter()
        .map(|(edge, team)| format!("{}->{}", model.label(*edge), model.label(*team)))
        .collect::<Vec<_>>()
        .join(" ");

    panic!(
        "{} {:?}: a fresh subscription disagrees with the model\n  engine only: {engine_only:?}\n  \
         model only: {model_only:?}\n  client edges: {}\n  server edges: {}\n  \
         model alice edges: {model_edges}\n  rejected fates: {} {:#?}\n  history of {involved:?}:\n    \
         {history}\n({context})",
        watch.session,
        watch.name,
        render(&client_edges),
        render(&server_edges),
        link.rejected_fates,
        link.rejections,
    );
}

fn report_scope_mismatch(
    link: &mut Link,
    model: &Model,
    watches: &[Watch],
    team: ObjectId,
    in_scope: bool,
    log: &[String],
    context: String,
) -> ! {
    let branch = link.server.schema_manager().branch_name();
    let queries = link
        .server
        .schema_manager()
        .query_manager()
        .sync_manager()
        .get_client(link.client_id)
        .map(|client| {
            let mut queries: Vec<String> = client
                .queries
                .iter()
                .map(|(query_id, scope)| {
                    format!(
                        "{query_id:?}: {} rows, holds it: {}",
                        scope.scope.len(),
                        scope.scope.contains(&(team, branch))
                    )
                })
                .collect();
            queries.sort();
            queries.join("; ")
        })
        .unwrap_or_else(|| "the server does not know the client".to_string());
    let server_subscriptions = link
        .server
        .schema_manager()
        .query_manager()
        .server_subscription_telemetry()
        .iter()
        .map(|group| format!("{} x{}", group.query, group.count))
        .collect::<Vec<_>>()
        .join("; ");
    let held = watches
        .iter()
        .map(|watch| {
            format!(
                "{} {:?} {:?} synced={}: holds it {}",
                watch.session,
                watch.name,
                watch.sub,
                watch.synced,
                ids(&link.client, watch.sub).contains(&team)
            )
        })
        .collect::<Vec<_>>()
        .join("\n    ");
    let label = model.label(team);
    let client_edges = edges(&mut link.client, model);
    let server_edges = edges(&mut link.server, model);
    let mut involved: HashSet<String> = HashSet::from([label.clone()]);
    for (edge, _, edge_team) in client_edges.iter().chain(&server_edges) {
        if *edge_team == label {
            involved.insert(edge.clone());
        }
    }
    let render = |rows: &[(String, String, String)]| {
        rows.iter()
            .filter(|(_, _, edge_team)| *edge_team == label)
            .map(|(edge, user, _)| format!("{edge}({user})"))
            .collect::<Vec<_>>()
            .join(" ")
    };
    panic!(
        "{label}: alice's server scope {} it, but the model has it {}\n  server queries of the \
         client: {queries}\n  server subscriptions: {server_subscriptions}\n  client watches:\n    \
         {held}\n  edges to {label} on the client: {}\n  edges to {label} on the server: {}\n  \
         history of {involved:?}:\n    {}\n({context})",
        if in_scope { "holds" } else { "lacks" },
        if in_scope { "hidden" } else { "visible" },
        render(&client_edges),
        render(&server_edges),
        history(log, &involved),
    );
}

/// The log lines that name any of `involved`, each with two lines before and four after.
fn history(log: &[String], involved: &HashSet<String>) -> String {
    let mut shown = BTreeSet::new();
    for (index, entry) in log.iter().enumerate() {
        if entry
            .split_whitespace()
            .any(|token| involved.contains(token.trim_end_matches(':')))
        {
            shown.extend(index.saturating_sub(2)..(index + 5).min(log.len()));
        }
    }
    shown
        .into_iter()
        .map(|index| log[index].as_str())
        .collect::<Vec<_>>()
        .join("\n    ")
}

/// Every edge a node holds, read without a session, as `(edge, user, team)` labels.
fn edges(core: &mut Node, model: &Model) -> Vec<(String, String, String)> {
    let sub = core
        .schema_manager_mut()
        .query_manager_mut()
        .subscribe_with_session(QueryBuilder::new("user_team_edges").build(), None, None)
        .expect("a sessionless read of edges");
    core.immediate_tick();
    let rows = core
        .schema_manager()
        .query_manager()
        .get_subscription_results(sub);
    core.schema_manager_mut()
        .query_manager_mut()
        .unsubscribe_with_sync(sub);
    rows.into_iter()
        .map(|(id, values)| {
            let user = match values.first() {
                Some(Value::Text(user)) => user.clone(),
                other => format!("{other:?}"),
            };
            let team = match values.get(1) {
                Some(Value::Uuid(team)) => model.label(*team),
                other => format!("{other:?}"),
            };
            (model.label(id), user, team)
        })
        .collect()
}

#[test]
fn policy_dependent_subscriptions_match_a_fresh_read_and_stay_clean() {
    if let Some(seed) = std::env::var("POLICY_DIFF_SEED")
        .ok()
        .and_then(|value| u64::from_str_radix(value.trim_start_matches("0x"), 16).ok())
    {
        run_seed(seed);
        return;
    }
    let extra = std::env::var("POLICY_DIFF_EXTRA_SEEDS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(0);
    for seed in SEEDS
        .into_iter()
        .chain((1..=extra).map(|index| 0x9011_C7DE_1000_0000 + index))
    {
        run_seed(seed);
    }
}

fn run_seed(seed: u64) {
    let mut rng = Xorshift(seed);
    let mut paths = Vec::new();
    let mut client = node("client", seed, &mut paths);
    let mut server = node("server", seed, &mut paths);
    let client_id = ClientId::new();
    let server_id = ServerId::new();
    server.add_client(client_id, Some(Session::new("alice")));
    client.add_server(server_id);
    let mut link = Link {
        client,
        server,
        client_id,
        server_id,
        rejected_fates: 0,
        rejections: BTreeMap::new(),
        ups: 0,
    };

    let mut watches = Vec::new();
    for (session, name, synced) in [
        ("alice", None, true),
        ("alice", Some("red"), true),
        ("bob", None, false),
    ] {
        let sub = subscribe(&mut link.client, session, name, synced);
        watches.push(Watch {
            session,
            name,
            synced,
            sub,
        });
    }
    link.client.immediate_tick();

    let mut model = Model::default();
    let mut log = Vec::new();
    let mut alice_saw_a_team = false;
    for step in 1..=OPS_PER_SEED {
        let first_entry = log.len();
        apply_op(&mut link, &mut model, &mut rng, &mut log);
        network_step(&mut link, &mut rng, &mut log);
        if step == OPS_PER_SEED / 2 {
            // Born mid-run, over pending writes and in-flight fates.
            for (session, name, synced) in [("alice", None, true), ("bob", Some("blue"), false)] {
                let sub = subscribe(&mut link.client, session, name, synced);
                watches.push(Watch {
                    session,
                    name,
                    synced,
                    sub,
                });
            }
            log.push("  subscribe alice all, bob blue".to_string());
        }
        for entry in &mut log[first_entry..] {
            entry.insert_str(0, &format!("{step:>3} "));
        }
        if step % CHECK_EVERY == 0 {
            check(&mut link, &model, &watches, seed, step, &log);
            log.push(format!("{step:>3}   -- check passed, link settled"));
            alice_saw_a_team |= !model.visible("alice", None).is_empty();
        }
    }

    assert!(
        link.rejected_fates > 0,
        "seed {seed:#x}: fixture precondition — nothing was rejected upstream, so the retraction \
         path went unexercised"
    );
    assert!(
        alice_saw_a_team,
        "seed {seed:#x}: fixture precondition — alice never saw a team at a check"
    );

    drop(link);
    for path in paths {
        for suffix in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{}{suffix}", path.display()));
        }
    }
}
