//! Randomized differential for authorization verdicts that are kept instead of recomputed.
//!
//! Two kinds of verdict are reused (`query_manager::authz_cache`): one reached in a settle
//! pass, by the other subscriptions of that pass; and an allowed one of a table whose select
//! policy is constant, until the row changes. Both are right only as long as nothing they
//! rest on moves without the cache hearing of it, and the sequences that could do that are
//! the ones nobody wrote a gate for: a write the server refuses and the client takes back, a
//! row that changes hands while another write to it is still on its way, a delete that lands
//! between two settles, a subscription born over all of that.
//!
//! `notes` is read by everyone and written by its author; `docs` is read by its owner and
//! written by anyone. The client writes as alice, whom the server knows the connection as,
//! and as bob, whose notes the server therefore refuses. A second client only reads, as
//! carol: on the server her subscriptions are settled in the same passes as alice's, over
//! the same rows, and are served the verdicts alice's reached. Several subscriptions watch
//! each table — the same query more than once, so that a pass has verdicts to share.
//!
//! Whenever the link has gone quiet:
//!
//! * **Incremental equals fresh** — every long-lived subscription holds exactly the rows a
//!   subscription created now, over the same query and session, settles to with no verdict
//!   served or kept on either node: every one of its rows is checked from storage.
//! * **Model** — everyone reads alice's surviving notes and nobody reads bob's; each reads
//!   the docs they own.
//! * **Clean** — no subscription is still waiting for a settle.
//! * **Server scope** — the scope the server keeps for each client's synced subscriptions
//!   holds alice's surviving notes and the docs that client's session owns; and it holds
//!   no note the server refused or that was deleted, and no doc of anyone else's. The
//!   scope of each long-lived subscription is also the scope of the one created now.
//!
//! And all the way through, in a debug build, every kept verdict that is served is compared
//! with a fresh evaluation where it is served.
//!
//! As in `policy_dependency_differential`, a delete only targets a row whose last write has
//! already gone up: a delete sent in the same frame as an earlier write to its row is
//! refused upstream, a defect that predates this oracle.
//!
//! Backend: `SqliteStorage` on both nodes, as on the phone.

use super::*;
use crate::batch_fate::BatchFate;
use crate::storage::SqliteStorage;
use crate::sync_manager::QueryId;

type Node = RuntimeCore<SqliteStorage, NoopScheduler>;

const SEEDS: [u64; 6] = [
    0x4B3D_7E2D_0000_0001,
    0x4B3D_7E2D_0000_0002,
    0x4B3D_7E2D_0000_0003,
    0x4B3D_7E2D_0000_0004,
    0x4B3D_7E2D_0000_0005,
    0x4B3D_7E2D_0000_0006,
];

const OPS_PER_SEED: usize = 160;
const CHECK_EVERY: usize = 20;

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
        .table(
            TableSchema::builder("notes")
                .column("author", ColumnType::Text)
                .column("body", ColumnType::Text),
        )
        .table(
            TableSchema::builder("docs")
                .column("owner", ColumnType::Text)
                .column("body", ColumnType::Text),
        )
        .build()
}

fn own(column: &str) -> PolicyExpr {
    PolicyExpr::eq_session(column, vec!["user_id".into()])
}

fn authorization_schema() -> Schema {
    SchemaBuilder::new()
        .table(
            TableSchema::builder("notes")
                .column("author", ColumnType::Text)
                .column("body", ColumnType::Text)
                .policies(
                    TablePolicies::new()
                        .with_select(PolicyExpr::True)
                        .with_insert(own("author"))
                        .with_update(Some(own("author")), own("author"))
                        .with_delete(own("author")),
                ),
        )
        .table(
            TableSchema::builder("docs")
                .column("owner", ColumnType::Text)
                .column("body", ColumnType::Text)
                .policies(
                    TablePolicies::new()
                        .with_select(own("owner"))
                        .with_insert(PolicyExpr::True)
                        .with_update(Some(PolicyExpr::True), PolicyExpr::True)
                        .with_delete(PolicyExpr::True),
                ),
        )
        .build()
}

fn node(role: &str, seed: u64, paths: &mut Vec<std::path::PathBuf>) -> Node {
    let path = std::env::temp_dir().join(format!(
        "jazz-kept-verdict-{role}-{seed:x}-{}.sqlite",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);
    let storage = SqliteStorage::open(&path).expect("sqlite storage should open");
    paths.push(path);
    node_over(storage)
}

/// A runtime over a store: a fresh one, or the one a runtime before it left behind.
fn node_over(storage: SqliteStorage) -> Node {
    let app_id = AppId::from_name("kept-verdict-differential");
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

struct Peer {
    node: Node,
    id: ClientId,
}

struct Link {
    /// The first one writes, as alice and as bob; the rest only read.
    peers: Vec<Peer>,
    server: Node,
    server_id: ServerId,
    rejected_fates: usize,
    /// How many times the clients' queues went up; a write made before the n-th `up`
    /// shipped with it.
    ups: usize,
    /// Every verdict is evaluated, on every node, all run long: what a run with kept
    /// verdicts is compared with.
    evaluates_everything: bool,
}

impl Link {
    fn client(&mut self) -> &mut Node {
        &mut self.peers[0].node
    }

    fn up(&mut self) -> bool {
        let mut any = false;
        for peer in &mut self.peers {
            peer.node.batched_tick();
            for entry in peer.node.sync_sender().take() {
                if entry.destination == Destination::Server(self.server_id) {
                    any = true;
                    self.server.park_sync_message(InboxEntry {
                        source: Source::Client(peer.id),
                        payload: entry.payload,
                    });
                }
            }
        }
        self.server.batched_tick();
        self.server.immediate_tick();
        self.ups += 1;
        any
    }

    fn down(&mut self) -> bool {
        self.server.batched_tick();
        let mut any = false;
        for entry in self.server.sync_sender().take() {
            let Destination::Client(to) = entry.destination else {
                continue;
            };
            let Some(index) = self.peers.iter().position(|peer| peer.id == to) else {
                continue;
            };
            if index == 0
                && let SyncPayload::BatchFate {
                    fate: BatchFate::Rejected { .. },
                } = &entry.payload
            {
                self.rejected_fates += 1;
            }
            any = true;
            self.peers[index].node.park_sync_message(InboxEntry {
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
            for peer in &mut self.peers {
                peer.node.batched_tick();
                peer.node.immediate_tick();
            }
            quiet_rounds = if up || down { 0 } else { quiet_rounds + 1 };
            if quiet_rounds == 2 {
                return;
            }
        }
        panic!("the link never went quiet");
    }

    /// Whether verdicts are served and kept from here on; never, in a run that evaluates
    /// everything.
    fn bypass_kept_verdicts(&mut self, bypassed: bool) {
        let bypassed = bypassed || self.evaluates_everything;
        for node in self
            .peers
            .iter_mut()
            .map(|peer| &mut peer.node)
            .chain([&mut self.server])
        {
            node.schema_manager_mut()
                .query_manager_mut()
                .bypass_authz_verdicts_for_tests(bypassed);
        }
    }
}

#[derive(Default)]
struct Model {
    /// Alice's notes, accepted upstream.
    alice_notes: Vec<ObjectId>,
    /// Bob's notes: accepted locally, refused upstream.
    bob_notes: Vec<ObjectId>,
    /// Every note nobody may be served: bob's, whatever became of them, and the ones
    /// alice deleted.
    gone_notes: Vec<ObjectId>,
    /// Deleted docs.
    gone_docs: Vec<ObjectId>,
    /// `(doc, owner)`.
    docs: Vec<(ObjectId, &'static str)>,
    /// `Link::ups` at each row's last write; the write has shipped once `ups` moved past it.
    last_write: HashMap<ObjectId, usize>,
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

    fn shipped(&self, id: ObjectId, ups: usize) -> bool {
        self.last_write[&id] < ups
    }

    fn visible(&self, table: &str, session: &str) -> Vec<ObjectId> {
        let mut ids: Vec<ObjectId> = match table {
            "notes" => self.alice_notes.clone(),
            _ => self
                .docs
                .iter()
                .filter(|(_, owner)| *owner == session)
                .map(|(doc, _)| *doc)
                .collect(),
        };
        ids.sort();
        ids
    }
}

struct Watch {
    /// Which of `Link::peers` holds it.
    peer: usize,
    table: &'static str,
    session: &'static str,
    synced: bool,
    sub: QuerySubscriptionId,
}

fn subscribe(client: &mut Node, table: &str, session: &str, synced: bool) -> QuerySubscriptionId {
    let query_manager = client.schema_manager_mut().query_manager_mut();
    let query = QueryBuilder::new(table).build();
    let session = Some(Session::new(session));
    let subscribed = if synced {
        query_manager.subscribe_with_sync(query, session, None)
    } else {
        query_manager.subscribe_with_session(query, session, None)
    };
    subscribed.expect("a session may subscribe")
}

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

/// The rows of the scope the server keeps for one query of a client.
fn server_scope(server: &Node, client_id: ClientId, query: QueryId) -> Vec<ObjectId> {
    let branch = server.schema_manager().branch_name();
    let mut rows: Vec<ObjectId> = server
        .schema_manager()
        .query_manager()
        .sync_manager()
        .get_client(client_id)
        .and_then(|client| client.queries.get(&query))
        .map(|query| {
            query
                .scope
                .iter()
                .filter(|(_, row_branch)| *row_branch == branch)
                .map(|(row, _)| *row)
                .collect()
        })
        .unwrap_or_default();
    rows.sort();
    rows
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
    let ups = link.ups;
    let client = link.client();
    let body = format!("body {}", rng.below(1000));

    match rng.below(100) {
        0..=13 => {
            let ((note, _), _) = client
                .insert(
                    "notes",
                    HashMap::from([
                        ("author".to_string(), text("alice")),
                        ("body".to_string(), text(&body)),
                    ]),
                    Some(&alice),
                )
                .expect("alice may write her note");
            model.alice_notes.push(note);
            model.last_write.insert(note, ups);
            let label = model.name(note, "an");
            log.push(format!("insert {label}"));
        }
        14..=25 => {
            let ((note, _), _) = client
                .insert(
                    "notes",
                    HashMap::from([
                        ("author".to_string(), text("bob")),
                        ("body".to_string(), text(&body)),
                    ]),
                    Some(&bob),
                )
                .expect("bob's own policy admits his note locally");
            model.bob_notes.push(note);
            model.gone_notes.push(note);
            let label = model.name(note, "bn");
            log.push(format!("insert {label}"));
        }
        26..=37 if !model.alice_notes.is_empty() => {
            let note = model.alice_notes[rng.below(model.alice_notes.len())];
            client
                .update(note, vec![("body".to_string(), text(&body))], Some(&alice))
                .expect("alice may rewrite her note");
            model.last_write.insert(note, ups);
            log.push(format!("rewrite {}", model.label(note)));
        }
        38..=45 => {
            let shipped: Vec<usize> = (0..model.alice_notes.len())
                .filter(|index| model.shipped(model.alice_notes[*index], ups))
                .collect();
            if shipped.is_empty() {
                log.push("delete note skipped: nothing shipped".to_string());
                return;
            }
            let note = model.alice_notes.remove(shipped[rng.below(shipped.len())]);
            model.gone_notes.push(note);
            client
                .delete(note, Some(&alice))
                .expect("alice may delete her note");
            log.push(format!("delete {}", model.label(note)));
        }
        46..=53 if !model.bob_notes.is_empty() => {
            // The note may already have been taken back, and then there is nothing to write.
            let index = rng.below(model.bob_notes.len());
            let note = model.bob_notes[index];
            let outcome = if rng.below(2) == 0 {
                model.bob_notes.remove(index);
                client.delete(note, Some(&bob)).map(|_| ())
            } else {
                client
                    .update(note, vec![("body".to_string(), text(&body))], Some(&bob))
                    .map(|_| ())
            };
            log.push(format!(
                "bob touches {}: {}",
                model.label(note),
                if outcome.is_ok() {
                    "written"
                } else {
                    "refused"
                }
            ));
        }
        54..=67 => {
            let owner = if rng.below(2) == 0 { "alice" } else { "bob" };
            let ((doc, _), _) = client
                .insert(
                    "docs",
                    HashMap::from([
                        ("owner".to_string(), text(owner)),
                        ("body".to_string(), text(&body)),
                    ]),
                    Some(&alice),
                )
                .expect("anyone may write a doc");
            model.docs.push((doc, owner));
            model.last_write.insert(doc, ups);
            let label = model.name(doc, "doc");
            log.push(format!("insert {label} of {owner}"));
        }
        68..=81 if !model.docs.is_empty() => {
            let index = rng.below(model.docs.len());
            let (doc, owner) = model.docs[index];
            let to = if owner == "alice" { "bob" } else { "alice" };
            client
                .update(doc, vec![("owner".to_string(), text(to))], Some(&alice))
                .expect("anyone may hand a doc over");
            model.docs[index].1 = to;
            model.last_write.insert(doc, ups);
            log.push(format!("hand {} {owner} -> {to}", model.label(doc)));
        }
        82..=91 if !model.docs.is_empty() => {
            let doc = model.docs[rng.below(model.docs.len())].0;
            client
                .update(doc, vec![("body".to_string(), text(&body))], Some(&alice))
                .expect("anyone may rewrite a doc");
            model.last_write.insert(doc, ups);
            log.push(format!("rewrite {}", model.label(doc)));
        }
        _ => {
            let shipped: Vec<usize> = (0..model.docs.len())
                .filter(|index| model.shipped(model.docs[*index].0, ups))
                .collect();
            if shipped.is_empty() {
                log.push("delete doc skipped: nothing shipped".to_string());
                return;
            }
            let (doc, _) = model.docs.remove(shipped[rng.below(shipped.len())]);
            model.gone_docs.push(doc);
            client
                .delete(doc, Some(&alice))
                .expect("anyone may delete a doc");
            log.push(format!("delete {}", model.label(doc)));
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
            link.client().immediate_tick();
            "client immediate_tick"
        }
        3 => {
            link.client().batched_tick();
            "client batched_tick"
        }
        4 => {
            link.up();
            link.down();
            for peer in &mut link.peers {
                peer.node.batched_tick();
                peer.node.immediate_tick();
            }
            "round trip"
        }
        _ => return,
    };
    log.push(format!("  net: {step}"));
}

fn labels(model: &Model, ids: &[ObjectId]) -> Vec<String> {
    ids.iter().map(|id| model.label(*id)).collect()
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
            log[log.len().saturating_sub(20)..].join("\n    ")
        )
    };

    for peer in &link.peers {
        let waiting = peer
            .node
            .schema_manager()
            .query_manager()
            .subscriptions_awaiting_settle();
        assert!(
            waiting.is_empty(),
            "{} subscription(s) still wait for a settle on a quiet link ({})",
            waiting.len(),
            context()
        );
    }

    for (index, session) in READERS.into_iter().enumerate() {
        let client_id = link.peers[index].id;
        let live_queries: Vec<QueryId> = watches
            .iter()
            .filter(|watch| watch.synced && watch.peer == index)
            .map(|watch| QueryId(watch.sub.0))
            .collect();
        for note in &model.alice_notes {
            assert!(
                in_live_server_scope(&link.server, client_id, &live_queries, *note),
                "the server's scope for {session} lacks alice's note {} ({})",
                model.label(*note),
                context()
            );
        }
        for gone in model.gone_notes.iter().chain(&model.gone_docs) {
            assert!(
                !in_live_server_scope(&link.server, client_id, &live_queries, *gone),
                "the server's scope for {session} holds {}, which nobody may be served ({})",
                model.label(*gone),
                context()
            );
        }
        for (doc, owner) in &model.docs {
            assert_eq!(
                in_live_server_scope(&link.server, client_id, &live_queries, *doc),
                *owner == session,
                "{} of {owner} in the server's scope for {session} ({})",
                model.label(*doc),
                context()
            );
        }
    }

    for watch in watches {
        let client_id = link.peers[watch.peer].id;
        let held = ids(&link.peers[watch.peer].node, watch.sub);
        // The comparison is made without the thing compared: on every node the fresh
        // subscription settles with every verdict evaluated from storage.
        link.bypass_kept_verdicts(true);
        let fresh_sub = subscribe(
            &mut link.peers[watch.peer].node,
            watch.table,
            watch.session,
            watch.synced,
        );
        if watch.synced {
            link.settle();
        } else {
            link.peers[watch.peer].node.immediate_tick();
        }
        link.bypass_kept_verdicts(false);
        let fresh = ids(&link.peers[watch.peer].node, fresh_sub);
        if watch.synced {
            assert_eq!(
                labels(
                    model,
                    &server_scope(&link.server, client_id, QueryId(watch.sub.0))
                ),
                labels(
                    model,
                    &server_scope(&link.server, client_id, QueryId(fresh_sub.0))
                ),
                "{} of {}: the scope the server keeps for the long-lived subscription is not \
                 the scope of one created now ({})",
                watch.table,
                watch.session,
                context()
            );
        }
        link.peers[watch.peer]
            .node
            .schema_manager_mut()
            .query_manager_mut()
            .unsubscribe_with_sync(fresh_sub);

        assert_eq!(
            labels(model, &held),
            labels(model, &fresh),
            "{} of {}: the long-lived subscription disagrees with one created now over the same \
             data ({})",
            watch.table,
            watch.session,
            context()
        );
        assert_eq!(
            labels(model, &fresh),
            labels(model, &model.visible(watch.table, watch.session)),
            "{} of {}: a subscription created now disagrees with the model ({})",
            watch.table,
            watch.session,
            context()
        );
    }
    link.settle();
}

#[test]
fn kept_verdicts_match_a_fresh_read_and_the_model() {
    if let Some(seed) = std::env::var("KEPT_VERDICT_SEED")
        .ok()
        .and_then(|value| u64::from_str_radix(value.trim_start_matches("0x"), 16).ok())
    {
        run_seed(seed);
        return;
    }
    let extra = std::env::var("KEPT_VERDICT_EXTRA_SEEDS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(0);
    for seed in SEEDS
        .into_iter()
        .chain((1..=extra).map(|index| 0x4B3D_7E2D_1000_0000 + index))
    {
        run_seed(seed);
    }
}

/// Who each of `Link::peers` is to the server, in their order.
const READERS: [&str; 2] = ["alice", "carol"];

fn run_seed(seed: u64) {
    let mut rng = Xorshift(seed);
    let mut paths = Vec::new();
    let mut server = node("server", seed, &mut paths);
    let server_id = ServerId::new();
    let peers = READERS
        .into_iter()
        .map(|session| {
            let mut node = node(session, seed, &mut paths);
            let id = ClientId::new();
            server.add_client(id, Some(Session::new(session)));
            node.add_server(server_id);
            Peer { node, id }
        })
        .collect();
    let mut link = Link {
        peers,
        server,
        server_id,
        rejected_fates: 0,
        ups: 0,
        evaluates_everything: std::env::var_os("KEPT_VERDICT_EVALUATE_EVERYTHING").is_some(),
    };
    link.bypass_kept_verdicts(false);

    let mut watches = Vec::new();
    for (peer, table, session, synced) in [
        (0, "notes", "alice", true),
        (0, "notes", "alice", true),
        (0, "notes", "bob", false),
        (0, "docs", "alice", true),
        (0, "docs", "alice", false),
        (0, "docs", "bob", false),
        (1, "notes", "carol", true),
        (1, "notes", "carol", true),
        (1, "docs", "carol", true),
    ] {
        let sub = subscribe(&mut link.peers[peer].node, table, session, synced);
        watches.push(Watch {
            peer,
            table,
            session,
            synced,
            sub,
        });
    }
    for peer in &mut link.peers {
        peer.node.immediate_tick();
    }

    let hits = |link: &Link| -> u64 {
        link.peers
            .iter()
            .map(|peer| &peer.node)
            .chain([&link.server])
            .map(|node| {
                node.schema_manager()
                    .query_manager()
                    .authz_cache_hit_count()
            })
            .sum()
    };
    let (client_hits_before, server_hits_before) = (
        link.peers[0]
            .node
            .schema_manager()
            .query_manager()
            .authz_cache_hit_count(),
        link.server
            .schema_manager()
            .query_manager()
            .authz_cache_hit_count(),
    );
    let mut model = Model::default();
    let mut log = Vec::new();
    for step in 1..=OPS_PER_SEED {
        let first_entry = log.len();
        apply_op(&mut link, &mut model, &mut rng, &mut log);
        network_step(&mut link, &mut rng, &mut log);
        if step == OPS_PER_SEED / 2 {
            // Born mid-run, over pending writes and fates on their way.
            for (table, session, synced) in [("notes", "bob", false), ("docs", "bob", false)] {
                let sub = subscribe(link.client(), table, session, synced);
                watches.push(Watch {
                    peer: 0,
                    table,
                    session,
                    synced,
                    sub,
                });
            }
            log.push("  subscribe bob notes, bob docs".to_string());
        }
        for entry in &mut log[first_entry..] {
            entry.insert_str(0, &format!("{step:>3} "));
        }
        if step % CHECK_EVERY == 0 {
            check(&mut link, &model, &watches, seed, step, &log);
            log.push(format!("{step:>3}   -- check passed, link settled"));
        }
    }

    assert!(
        link.rejected_fates > 0,
        "seed {seed:#x}: fixture precondition — nothing was refused upstream, so no write was \
         taken back"
    );
    if link.evaluates_everything {
        assert_eq!(hits(&link), 0, "fixture: a verdict was served");
    } else {
        let client_hits = link.peers[0]
            .node
            .schema_manager()
            .query_manager()
            .authz_cache_hit_count()
            - client_hits_before;
        let server_hits = link
            .server
            .schema_manager()
            .query_manager()
            .authz_cache_hit_count()
            - server_hits_before;
        assert!(
            client_hits > 0 && server_hits > 0,
            "seed {seed:#x}: fixture precondition — no kept verdict was ever served \
             (client {client_hits}, server {server_hits})"
        );
    }

    drop(link);
    for path in paths {
        for suffix in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{}{suffix}", path.display()));
        }
    }
}

/// Where one row is served: by the server's own subscription, in the scope the server
/// keeps for each reading client, and by each reading client's subscription.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Served {
    by_the_server: bool,
    in_carols_scope: bool,
    in_daves_scope: bool,
    to_carol: bool,
    to_dave: bool,
}

impl Served {
    const EVERYWHERE: Self = Self {
        by_the_server: true,
        in_carols_scope: true,
        in_daves_scope: true,
        to_carol: true,
        to_dave: true,
    };
    const NOWHERE: Self = Self {
        by_the_server: false,
        in_carols_scope: false,
        in_daves_scope: false,
        to_carol: false,
        to_dave: false,
    };
}

/// Where the notes of a refused batch were served, as it went.
#[derive(Debug)]
struct Refusal {
    /// Her note, once the server had taken it.
    before: Served,
    /// Her note, once the server had refused the batch and the link had gone quiet.
    just_after: Served,
    /// Her note, once another note had been written and every reader had settled.
    after: Served,
    bobs_after: Served,
    /// How many verdicts the server served from what it kept, from the refusal on.
    verdicts_served: u64,
}

/// A batch of two notes goes up from alice's client: one of hers, one of bob's. The
/// server takes hers first and serves it — to its own subscription and to two clients
/// that read `notes` — and then meets bob's, refuses it, and with it the batch: her note
/// leaves the store with nothing in its place.
fn a_batch_is_refused_after_one_of_its_rows_was_served(
    tag: u64,
    evaluates_everything: bool,
    server_starts_again: bool,
) -> Refusal {
    let mut paths = Vec::new();
    let mut server = node("server", tag, &mut paths);
    let server_id = ServerId::new();
    let mut writer = node("alice", tag, &mut paths);
    let writer_id = ClientId::new();
    server.add_client(writer_id, Some(Session::new("alice")));
    writer.add_server(server_id);
    let mut link = Link {
        peers: vec![Peer {
            node: writer,
            id: writer_id,
        }],
        server,
        server_id,
        rejected_fates: 0,
        ups: 0,
        evaluates_everything,
    };
    link.bypass_kept_verdicts(false);
    link.settle();

    let batch = link.client().begin_batch(BatchMode::Direct);
    let in_the_batch = |session: &str| {
        WriteContext::from_session(Session::new(session))
            .with_batch_mode(BatchMode::Direct)
            .with_batch_id(batch)
    };
    let mut write = |author: &str| {
        let ((note, _), _) = link
            .client()
            .insert(
                "notes",
                HashMap::from([
                    ("author".to_string(), text(author)),
                    ("body".to_string(), text("in one batch")),
                ]),
                Some(&in_the_batch(author)),
            )
            .expect("each author's own policy admits their note locally");
        note
    };
    let hers = write("alice");
    let bobs = write("bob");
    link.client()
        .commit_batch(batch)
        .expect("the batch commits locally");

    // The connection drops before anything of the batch gets through. Connected again,
    // the client sends what the server never acknowledged: each row as the visible row
    // it is locally, and then the seal.
    link.client().batched_tick();
    link.client().immediate_tick();
    link.client().sync_sender().take();
    link.client().remove_server(server_id);
    link.client().add_server(server_id);
    link.client().batched_tick();
    // Her note arrives in one frame; bob's, and whatever follows it, in the next.
    let mut going_up: Vec<SyncPayload> = link
        .client()
        .sync_sender()
        .take()
        .into_iter()
        .filter(|entry| entry.destination == Destination::Server(server_id))
        .map(|entry| entry.payload)
        .collect();
    let first_of_bobs = going_up
        .iter()
        .position(
            |payload| matches!(payload, SyncPayload::RowBatchCreated { row, .. } if row.row_id == bobs),
        )
        .expect("fixture: bob's note is on its way up");
    assert!(
        going_up[..first_of_bobs].iter().any(
            |payload| matches!(payload, SyncPayload::RowBatchCreated { row, .. } if row.row_id == hers)
        ),
        "fixture: her note goes up before bob's"
    );
    let held_back = going_up.split_off(first_of_bobs);
    for payload in going_up {
        link.server.park_sync_message(InboxEntry {
            source: Source::Client(writer_id),
            payload,
        });
    }
    link.server.batched_tick();
    link.server.immediate_tick();

    assert!(
        link.server
            .storage()
            .load_visible_region_row(
                "notes",
                link.server.schema_manager().branch_name().as_str(),
                hers
            )
            .expect("read the visible region")
            .is_some(),
        "fixture: the server took her note and made it visible"
    );

    if server_starts_again {
        // Nothing the server held in memory about the batch is there any more; the
        // store is.
        let Link { peers, server, .. } = link;
        let mut server = node_over(server.into_storage());
        server.add_client(writer_id, Some(Session::new("alice")));
        link = Link {
            peers,
            server,
            server_id,
            rejected_fates: 0,
            ups: 0,
            evaluates_everything,
        };
        link.bypass_kept_verdicts(false);
    }

    let mut readers = Vec::new();
    for session in ["carol", "dave"] {
        let mut node = node(session, tag, &mut paths);
        let id = ClientId::new();
        link.server.add_client(id, Some(Session::new(session)));
        node.add_server(server_id);
        link.peers.push(Peer { node, id });
        link.bypass_kept_verdicts(false);
        let peer = link.peers.len() - 1;
        let sub = subscribe(&mut link.peers[peer].node, "notes", session, true);
        readers.push((peer, sub));
    }
    let on_the_server = link
        .server
        .schema_manager_mut()
        .query_manager_mut()
        .subscribe_with_session(
            QueryBuilder::new("notes").build(),
            Some(Session::new("erin")),
            None,
        )
        .expect("a session may subscribe on the server");
    link.settle();

    let served = |link: &Link, note: ObjectId| {
        let scope = |(peer, sub): (usize, QuerySubscriptionId)| {
            server_scope(&link.server, link.peers[peer].id, QueryId(sub.0)).contains(&note)
        };
        let held = |(peer, sub): (usize, QuerySubscriptionId)| {
            ids(&link.peers[peer].node, sub).contains(&note)
        };
        Served {
            by_the_server: ids(&link.server, on_the_server).contains(&note),
            in_carols_scope: scope(readers[0]),
            in_daves_scope: scope(readers[1]),
            to_carol: held(readers[0]),
            to_dave: held(readers[1]),
        }
    };
    let before = served(&link, hers);
    let hits_before = link
        .server
        .schema_manager()
        .query_manager()
        .authz_cache_hit_count();

    for payload in held_back {
        link.server.park_sync_message(InboxEntry {
            source: Source::Client(writer_id),
            payload,
        });
    }
    link.server.batched_tick();
    link.server.immediate_tick();
    link.settle();
    assert!(
        link.rejected_fates > 0,
        "fixture: the server refused the batch and told its writer"
    );
    assert!(
        link.server
            .storage()
            .load_visible_region_row(
                "notes",
                link.server.schema_manager().branch_name().as_str(),
                hers
            )
            .expect("read the visible region")
            .is_none(),
        "fixture: the refusal took her note out of the server's store"
    );

    let just_after = served(&link, hers);
    // Something else is written into the table: every subscription that reads it settles.
    let ((another, _), _) = link
        .client()
        .insert(
            "notes",
            HashMap::from([
                ("author".to_string(), text("alice")),
                ("body".to_string(), text("on its own")),
            ]),
            Some(&WriteContext::from_session(Session::new("alice"))),
        )
        .expect("alice may write her note");
    link.settle();
    assert_eq!(
        served(&link, another),
        Served::EVERYWHERE,
        "fixture: a note written after the refusal reaches every reader"
    );

    let after = served(&link, hers);
    let bobs_after = served(&link, bobs);
    let hits = link
        .server
        .schema_manager()
        .query_manager()
        .authz_cache_hit_count()
        - hits_before;

    drop(link);
    for path in paths {
        for suffix in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{}{suffix}", path.display()));
        }
    }
    Refusal {
        before,
        just_after,
        after,
        bobs_after,
        verdicts_served: hits,
    }
}

/// A refused batch takes a row it had made visible out of the store without a row to put
/// in its place, so no visibility change says so. With every verdict evaluated, loading
/// the row to authorize it is what finds it gone; a kept verdict must not keep it served —
/// not to the server's own subscription, not in the scope it keeps for a client, and so
/// not to any client. Driven through the server's own permission check, in its own tick.
#[test]
fn a_row_a_refused_batch_takes_back_leaves_everyone_it_was_served_to() {
    for (tag, server_starts_again) in [(0xE2E0_u64, false), (0xE2E2, true)] {
        let arm = format!("server starts again: {server_starts_again}");
        // With every verdict evaluated the row leaves: that is what must not change.
        let evaluated =
            a_batch_is_refused_after_one_of_its_rows_was_served(tag, true, server_starts_again);
        assert_eq!(evaluated.before, Served::EVERYWHERE, "fixture, {arm}");
        assert_eq!(evaluated.after, Served::NOWHERE, "fixture, {arm}");
        assert_eq!(evaluated.bobs_after, Served::NOWHERE, "fixture, {arm}");
        assert_eq!(evaluated.verdicts_served, 0, "fixture, {arm}");

        let kept = a_batch_is_refused_after_one_of_its_rows_was_served(
            tag + 1,
            false,
            server_starts_again,
        );
        assert_eq!(kept.before, Served::EVERYWHERE, "fixture, {arm}");
        assert!(
            kept.verdicts_served > 0,
            "fixture, {arm}: the server served no verdict it had kept"
        );
        assert_eq!(
            kept.after,
            Served::NOWHERE,
            "a row that left the store is still served ({arm}): its verdict was kept"
        );
        assert_eq!(kept.bobs_after, Served::NOWHERE, "{arm}");
        // Until something makes a client's subscription settle on the server, the scope
        // kept for it names the row whether verdicts are kept or not: a server started
        // again holds nothing that ties the refused batch to the subscriptions over it.
        assert_eq!(
            kept.just_after, evaluated.just_after,
            "{arm}: keeping verdicts changed what is served between the refusal and the \
             next settle"
        );
    }
}

/// A client that learns one of its batches was rejected, or rolls one back, patches the
/// batch's rows by the batch id, table by table: every row of the batch in the table, on
/// any branch, whether the client tracks it for the batch or not. Which rows the patch
/// touched it does not say, so nothing kept is served after it.
///
/// Three ways in: a direct write the upstream refused, whose row was visible here; a
/// committed transaction it refused, whose row was only staged; and a transaction rolled
/// back while still open, whose staged row the batch's index names.
#[test]
fn a_refused_direct_batch_costs_what_was_kept() {
    a_local_batch_is_taken_back(TakenBack::RefusedDirect);
}

#[test]
fn a_refused_transaction_costs_what_was_kept() {
    a_local_batch_is_taken_back(TakenBack::RefusedTransaction);
}

#[test]
fn a_rolled_back_transaction_costs_what_was_kept() {
    a_local_batch_is_taken_back(TakenBack::RolledBack);
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum TakenBack {
    RefusedDirect,
    RefusedTransaction,
    RolledBack,
}

fn a_local_batch_is_taken_back(how: TakenBack) {
    const NOTES: u64 = 5;
    let mut paths = Vec::new();
    // One store per way in: the three run side by side.
    let mut client = node("client", 0xE2E4 + how as u64, &mut paths);
    let alice = WriteContext::from_session(Session::new("alice"));
    let mut hers = Vec::new();
    for index in 0..NOTES {
        let ((note, _), _) = client
            .insert(
                "notes",
                HashMap::from([
                    ("author".to_string(), text("alice")),
                    ("body".to_string(), text(&format!("note {index}"))),
                ]),
                Some(&alice),
            )
            .expect("alice may write her note");
        hers.push(note);
    }
    let taken_back = if how == TakenBack::RefusedDirect {
        let ((_, _), refused) = client
            .insert(
                "notes",
                HashMap::from([
                    ("author".to_string(), text("bob")),
                    ("body".to_string(), text("refused upstream")),
                ]),
                Some(&WriteContext::from_session(Session::new("bob"))),
            )
            .expect("bob's own policy admits his note locally");
        refused
    } else {
        let batch = client.begin_batch(BatchMode::Transactional);
        client
            .insert(
                "notes",
                HashMap::from([
                    ("author".to_string(), text("alice")),
                    ("body".to_string(), text("staged")),
                ]),
                Some(
                    &WriteContext::from_session(Session::new("alice"))
                        .with_batch_mode(BatchMode::Transactional)
                        .with_batch_id(batch),
                ),
            )
            .expect("alice may stage a note");
        if how == TakenBack::RefusedTransaction {
            client
                .commit_batch(batch)
                .expect("the transaction is sealed and waits for the upstream");
        }
        batch
    };
    let sub = subscribe(&mut client, "notes", "alice", false);
    client.immediate_tick();
    let evaluated = |client: &Node| {
        client
            .schema_manager()
            .query_manager()
            .authz_cache_miss_count()
    };
    let rewrite = |client: &mut Node, body: &str| {
        let before = evaluated(client);
        client
            .update(
                hers[0],
                vec![("body".to_string(), text(body))],
                Some(&alice),
            )
            .expect("alice may rewrite her note");
        client.immediate_tick();
        evaluated(client) - before
    };
    assert_eq!(
        rewrite(&mut client, "rewritten"),
        1,
        "fixture ({how:?}): a write checks the row it changed and no other"
    );

    let checked_again = if how == TakenBack::RolledBack {
        // The rollback settles what it dirtied before it returns.
        let before = evaluated(&client);
        client
            .rollback_batch(taken_back)
            .expect("an open transaction rolls back");
        evaluated(&client) - before
    } else {
        client.mark_local_batch_rows_rejected(taken_back);
        rewrite(&mut client, "rewritten again")
    };
    assert_eq!(
        checked_again, NOTES,
        "{how:?}: kept verdicts were served after a patch went over rows by batch id"
    );
    assert_eq!(ids(&client, sub).len() as u64, NOTES);
    assert_eq!(rewrite(&mut client, "and again"), 1);
    drop(client);
    for path in paths {
        for suffix in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{}{suffix}", path.display()));
        }
    }
}
