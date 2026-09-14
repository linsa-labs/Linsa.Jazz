//! Randomized differential: after any stream of deliveries, every node that holds a row must
//! hold the SAME row, and no node may accumulate frontier tips.
//!
//! `SyncManager::scope_delivery_row` clears `parents` on every visible row before it goes out
//! to a client, and it still does — this fix changed no wire format. What changed is the
//! receiver's reading of the result. The frontier rule is "a visible row is a non-tip iff some
//! visible row names it as a parent", so a parentless arrival named nobody, superseded nothing,
//! and stood as its own branch. Tips accumulated one per delivery: measured in a real client
//! store at 460 on a single `users` row.
//!
//! `parents == []` was carrying two meanings — "this is the row's creation" and "my ancestry
//! was elided in transit" — and `row_histories::resolution::elided_snapshot_dominator` now
//! separates them by provenance: `for_insert` sets `created_at == updated_at`, `for_update`
//! copies `created_at` verbatim, and the clock never repeats, so a parentless row with
//! `created_at != updated_at` is provably not a creation. Equal `created_at` marks one lineage,
//! and within a lineage the newest snapshot supersedes the rest.
//!
//! Hand-written gates pin the sequences we already thought of; this exists for the rest. Two
//! properties are asserted after every operation, once the network has gone quiet:
//!
//! * **Convergence** — every node's visible `docs` set is identical, per column.
//! * **Width** — no node exceeds one un-superseded tip per author. This is the one that can
//!   tell the fix from its absence: every column here is LWW, and the merged preview of N
//!   parentless roots is already byte-equal to the newest of them, so the value assertions stay
//!   green either way. Measured: peak width 4 with the rule armed, 23 with it disarmed.
//!
//! The network is deliberately hostile within what the transport permits: it reorders, delays
//! across operations, and duplicates. Ordering defects do not show up on a network that
//! delivers in order, and this one is an ordering defect.
//!
//! Backend: `SqliteStorage`, not `MemoryStorage`, and not by preference. `MemoryStorage`
//! overrides the visible-row reads and answers from live maps, so on it an arrival never
//! travels the storage path whose fast-path miss is the defect. Defect 27 was invisible to the
//! memory backend for exactly this reason.
//!
//! Two known blind spots, both worth closing before this oracle is trusted alone. Every column
//! is strategy-less, so the model is LWW and collapsing a frontier to its `(updated_at,
//! batch_id)` maximum is what LWW says should happen — the oracle can therefore see
//! accumulation but not OVER-collapse; a `Counter` or `GSet` column and a matching model arm
//! would fix that. And delete/restore are excluded from the alphabet (a single delete is pinned
//! by `a_sender_delete_reaches_a_subscribed_peer`), so the delete-winner corruption is outside
//! the generated space.

use super::*;
use crate::batch_fate::{BatchFate, BatchMode};
use crate::storage::SqliteStorage;

type Node = RuntimeCore<SqliteStorage, NoopScheduler>;

const SEEDS: [u64; 6] = [
    0xDE11_0F5E_0000_0001,
    0xDE11_0F5E_0000_0002,
    0xDE11_0F5E_0000_0003,
    0xDE11_0F5E_0000_0004,
    0xDE11_0F5E_0000_0005,
    0xDE11_0F5E_0000_0006,
];

const OPS_PER_SEED: usize = 90;
const ROW_POOL: usize = 6;

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

fn docs_schema() -> Schema {
    SchemaBuilder::new()
        .table(
            TableSchema::builder("docs")
                .column("owner", ColumnType::Text)
                .column("body", ColumnType::Text)
                .column("hits", ColumnType::BigInt),
        )
        .build()
}

/// What the stream MEANT, independent of how it was delivered. Last write wins per row, a
/// delete hides the row until a restore brings it back.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ModelRow {
    owner: String,
    body: String,
    hits: i64,
    deleted: bool,
}

/// One node's view of the shared table, in a shape that compares cleanly.
type Snapshot = Vec<(ObjectId, String, String, i64)>;

fn visible_docs(core: &mut Node) -> Snapshot {
    let query = core
        .schema_manager_mut()
        .query_manager_mut()
        .query("docs")
        .build();
    let waker = noop_waker();
    let mut cx = std::task::Context::from_waker(&waker);
    // LocalOnly: this is a measurement, and a measurement must not itself move the scope it
    // is measuring.
    let mut future = core.query_with_propagation(
        query,
        None,
        ReadDurabilityOptions::default(),
        crate::sync_manager::QueryPropagation::LocalOnly,
    );
    let rows = match Pin::new(&mut future).poll(&mut cx) {
        Poll::Ready(Ok(results)) => results,
        Poll::Ready(Err(err)) => panic!("local query should succeed: {err:?}"),
        Poll::Pending => panic!("a local-only query must resolve immediately"),
    };
    let mut out: Snapshot = rows
        .into_iter()
        .map(|(id, values)| {
            let owner = match values.first() {
                Some(Value::Text(text)) => text.clone(),
                other => panic!("owner should be text, got {other:?}"),
            };
            let body = match values.get(1) {
                Some(Value::Text(text)) => text.clone(),
                other => panic!("body should be text, got {other:?}"),
            };
            let hits = match values.get(2) {
                Some(Value::BigInt(count)) => *count,
                other => panic!("hits should be a bigint, got {other:?}"),
            };
            (id, owner, body, hits)
        })
        .collect();
    out.sort();
    out
}

/// How many tips each node believes the row has.
///
/// Convergence of visible VALUES is not enough to pin this fix. Every column here is LWW, and
/// the merged preview of N parentless roots is already byte-equal to the newest of them — so a
/// node can agree with its peer on every column while holding N tips, and the assertion that
/// really matters is invisible. Frontier width is the thing the defect moves, and it is what
/// closes the O(1) apply path: `parents_cover_frontier_exactly` needs
/// `parents.len() == frontier.len()`, so a single-parent arrival against a wide frontier can
/// never qualify, for the life of the row.
fn tip_count(core: &mut Node, row_id: ObjectId, branch: &str) -> usize {
    core.storage()
        .scan_row_branch_tip_ids("docs", branch, row_id)
        .expect("tip ids should read")
        .len()
}

fn expected_snapshot(model: &BTreeMap<ObjectId, ModelRow>) -> Snapshot {
    let mut out: Snapshot = model
        .iter()
        .filter(|(_, row)| !row.deleted)
        .map(|(id, row)| (*id, row.owner.clone(), row.body.clone(), row.hits))
        .collect();
    out.sort();
    out
}

/// A message in flight. The transport may hold it, reorder it, or hand it over twice.
struct InFlight {
    to_server: bool,
    peer_index: usize,
    payload: SyncPayload,
}

struct Peer {
    core: Node,
    server_id: ServerId,
    client_id: ClientId,
    /// Held, not dropped. A query future unsubscribes when it falls out of scope, so polling
    /// one and letting it go produces a subscription and an immediate unsubscription — the
    /// peer would then receive nothing and the oracle would pass by testing nothing. A real
    /// client keeps a standing subscription; so does this one.
    subscription: Option<SubscriptionHandle>,
}

/// A network that is allowed to be unkind: it buffers, reorders, delays across operations and
/// occasionally duplicates. Every one of those is something a real socket does, and each is a
/// shape under which "relate this arrival to the history I hold" has to keep working.
struct Network {
    wire: Vec<InFlight>,
}

impl Network {
    fn new() -> Self {
        Self { wire: Vec::new() }
    }

    fn collect(&mut self, server: &mut Node, peers: &mut [Peer]) {
        server.batched_tick();
        for entry in server.sync_sender().take() {
            let Destination::Client(destination) = entry.destination else {
                continue;
            };
            if let Some(index) = peers.iter().position(|peer| peer.client_id == destination) {
                self.wire.push(InFlight {
                    to_server: false,
                    peer_index: index,
                    payload: entry.payload,
                });
            }
        }
        for (index, peer) in peers.iter_mut().enumerate() {
            peer.core.batched_tick();
            for entry in peer.core.sync_sender().take() {
                if entry.destination == Destination::Server(peer.server_id) {
                    self.wire.push(InFlight {
                        to_server: true,
                        peer_index: index,
                        payload: entry.payload,
                    });
                }
            }
        }
    }

    /// Hand over some of what is in flight, chosen out of order. Returns whether anything moved.
    fn deliver_some(
        &mut self,
        server: &mut Node,
        peers: &mut [Peer],
        rng: &mut Xorshift,
        drain_everything: bool,
    ) -> bool {
        if self.wire.is_empty() {
            return false;
        }
        let count = if drain_everything {
            self.wire.len()
        } else {
            1 + rng.below(self.wire.len())
        };
        let mut moved = false;
        for _ in 0..count {
            if self.wire.is_empty() {
                break;
            }
            let index = rng.below(self.wire.len());
            let message = self.wire.remove(index);
            // A duplicate is not a corruption of the stream, it is the stream: retransmission
            // is normal, and an idempotent receiver is the contract.
            let duplicate = !drain_everything && rng.below(16) == 0;
            let peer = &mut peers[message.peer_index];
            for _ in 0..(if duplicate { 2 } else { 1 }) {
                if message.to_server {
                    if std::env::var("DIFF_TRACE").is_ok() {
                        let text = format!("{:?}", message.payload);
                        eprintln!("  -> server: {}", &text[..text.len().min(90)]);
                    }
                    server.park_sync_message(InboxEntry {
                        source: Source::Client(peer.client_id),
                        payload: message.payload.clone(),
                    });
                } else {
                    if std::env::var("DIFF_TRACE").is_ok()
                        && let SyncPayload::RowBatchNeeded { row, .. }
                        | SyncPayload::RowBatchCreated { row, .. } = &message.payload
                    {
                        eprintln!(
                            "  <- peer{}: row {} deleted={} kind={:?} state={:?} parents={}",
                            message.peer_index,
                            row.row_id,
                            row.is_deleted,
                            row.delete_kind,
                            row.state,
                            row.parents.len()
                        );
                    }
                    peer.core.park_sync_message(InboxEntry {
                        source: Source::Server(peer.server_id),
                        payload: message.payload.clone(),
                    });
                }
            }
            moved = true;
        }
        server.batched_tick();
        server.immediate_tick();
        for peer in peers.iter_mut() {
            peer.core.batched_tick();
            peer.core.immediate_tick();
        }
        moved
    }

    /// Run until nothing is left in flight and nobody has anything more to say. Convergence is
    /// only meaningful at rest.
    fn settle(&mut self, server: &mut Node, peers: &mut [Peer], rng: &mut Xorshift) {
        for _ in 0..64 {
            self.collect(server, peers);
            let moved = self.deliver_some(server, peers, rng, true);
            if !moved && self.wire.is_empty() {
                self.collect(server, peers);
                if self.wire.is_empty() {
                    return;
                }
            }
        }
        panic!(
            "the network never went quiet: {} still in flight",
            self.wire.len()
        );
    }
}

fn open_store(tag: &str, seed: u64, node: usize) -> (SqliteStorage, std::path::PathBuf) {
    let path = std::env::temp_dir().join(format!(
        "jazz-delivery-convergence-{tag}-{seed:x}-{node}-{}.sqlite",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);
    let storage = SqliteStorage::open(&path).expect("sqlite storage should open");
    (storage, path)
}

fn runtime_over(
    schema: Schema,
    app_name: &str,
    storage: SqliteStorage,
    tier: Option<DurabilityTier>,
) -> Node {
    let app_id = AppId::from_name(app_name);
    let sync_manager = match tier {
        Some(tier) => SyncManager::new().with_durability_tier(tier),
        None => SyncManager::new(),
    };
    let schema_manager = SchemaManager::new(sync_manager, schema, app_id, "dev", "main").unwrap();
    let mut core = new_test_core(schema_manager, storage, NoopScheduler);
    core.immediate_tick();
    core
}

/// Give a peer a standing subscription, so the server has somewhere to deliver to.
fn subscribe(peer: &mut Peer) {
    if let Some(previous) = peer.subscription.take() {
        peer.core.unsubscribe(previous);
    }
    let query = peer
        .core
        .schema_manager_mut()
        .query_manager_mut()
        .query("docs")
        .build();
    let handle = peer
        .core
        .subscribe_with_durability_and_propagation(
            query,
            |_delta| {},
            None,
            ReadDurabilityOptions::default(),
            crate::sync_manager::QueryPropagation::Full,
        )
        .expect("a peer may subscribe to docs");
    peer.subscription = Some(handle);
    peer.core.batched_tick();
}

#[test]
fn delivered_rows_converge_across_a_reordering_network() {
    for seed in SEEDS {
        run_seed(seed);
    }
}

fn run_seed(seed: u64) {
    let mut rng = Xorshift(seed);
    let schema = docs_schema();
    let app = "delivery-convergence-differential";

    let (server_storage, server_path) = open_store("server", seed, 0);
    let mut server = runtime_over(
        schema.clone(),
        app,
        server_storage,
        Some(DurabilityTier::EdgeServer),
    );

    let mut paths = vec![server_path];
    let mut peers = Vec::new();
    for index in 0..2 {
        let (storage, path) = open_store("peer", seed, index + 1);
        paths.push(path);
        let core = runtime_over(schema.clone(), app, storage, None);
        let client_id = ClientId::new();
        let server_id = ServerId::new();
        let mut peer = Peer {
            core,
            server_id,
            client_id,
            subscription: None,
        };
        // A `ClientRole::User` connection with no session is refused at the row-apply arm with
        // `SessionRequired` and never applies anything (inbox.rs, the User branch of
        // `process_from_client`). Passing None here silently turns every peer write into a
        // no-op, which is what made this harness accuse the engine of losing writes.
        server.add_client(client_id, Some(Session::new("alice")));
        peer.core.add_server(server_id);
        subscribe(&mut peer);
        peers.push(peer);
    }

    let mut network = Network::new();
    network.settle(&mut server, &mut peers, &mut rng);

    // The composed branch name (`<env>-<schemahash>-<userbranch>`), which is what the raw
    // tip-id read needs — "main" is not the on-disk branch.
    let server_branch = server.schema_manager().branch_name().to_string();

    let mut model: BTreeMap<ObjectId, ModelRow> = BTreeMap::new();
    let mut pool: Vec<ObjectId> = Vec::new();
    let mut history: Vec<String> = Vec::new();

    for op_index in 0..OPS_PER_SEED {
        let op = choose_and_apply(
            &mut rng,
            &mut server,
            &mut peers,
            &mut model,
            &mut pool,
            op_index,
        );
        history.push(op);

        // Let the hostile network run for a while before demanding agreement, so reordering
        // and delay have somewhere to accumulate.
        network.collect(&mut server, &mut peers);
        network.deliver_some(&mut server, &mut peers, &mut rng, false);
        network.settle(&mut server, &mut peers, &mut rng);

        // Convergence first, because it needs no model to be true: whatever the operations
        // meant, the nodes must at least mean the same thing by them. Checking intent first
        // would let a lost write masquerade as a modelling mistake.
        let server_view = visible_docs(&mut server);
        for (index, peer) in peers.iter_mut().enumerate() {
            let peer_view = visible_docs(&mut peer.core);
            assert_eq!(
                peer_view,
                server_view,
                "seed {seed:#x} op {op_index}: peer {index} did not converge with the sender. \
                 A receiver that cannot relate an arrival to the history it already holds \
                 treats every delivery as a new root: tips accumulate instead of collapsing, \
                 counters sum instead of superseding, and a soft delete has nothing left that \
                 can supersede it.\nhistory:\n  {}",
                history.join("\n  ")
            );
        }

        // Width, not just value. Disarming the frontier rule leaves every value assertion
        // above green — that is measured, not assumed, because every column here is LWW and the
        // merged preview of N parentless roots is already byte-equal to the newest of them. So
        // width is the only thing that can tell this fix from its absence.
        //
        // The bound is one un-superseded tip per author, and each half of it is load-bearing.
        // Genuine concurrency must survive: two peers writing at once really are two branches,
        // and a peer's own local write is one the authority can never name back at it, because
        // the server does not echo a writer its own batch and its later snapshots arrive with
        // their ancestry stripped. Collapsing either of those would destroy a real write. What
        // must never happen is ACCUMULATION — one more tip for every delivery, unbounded, which
        // is the defect: measured in a real client store at 460 tips on one `users` row.
        // One un-superseded tip per author, plus one authority snapshot that no local write has
        // named yet, plus a tip of headroom. Measured across these six seeds: peak 4 with the
        // rule armed, 23 with it disarmed; the field reached 460, because without the rule
        // nothing bounds the count at all. The headroom is deliberate — a bound sitting exactly
        // on the observed peak is not a gate, it is a coincidence waiting for the next seed —
        // and it costs nothing here, since the value it must separate is five times larger.
        let author_bound = peers.len() + 3;
        for (row_id, _, _, _) in &server_view {
            let branch = server_branch.clone();
            let server_tips = tip_count(&mut server, *row_id, &branch);
            assert!(
                server_tips <= author_bound,
                "seed {seed:#x} op {op_index}: the sender holds {server_tips} tips for row \
                 {row_id}, above the one-per-author bound of {author_bound}.\nhistory:\n  {}",
                history.join("\n  ")
            );
            for (index, peer) in peers.iter_mut().enumerate() {
                let peer_tips = tip_count(&mut peer.core, *row_id, &branch);
                assert!(
                    peer_tips <= author_bound,
                    "seed {seed:#x} op {op_index}: peer {index} holds {peer_tips} tips for row \
                     {row_id}, above the one-per-author bound of {author_bound}. Delivery strips \
                     ancestry, so each arrival looks like a row creation and supersedes nothing; \
                     tips then accumulate one per delivery and the O(1) apply path is shut for \
                     the life of the row — `parents_cover_frontier_exactly` needs \
                     parents.len() == frontier.len(), which a one-parent arrival can never \
                     satisfy against a wide frontier.\nhistory:\n  {}",
                    history.join("\n  ")
                );
            }
        }

        let expected = expected_snapshot(&model);
        assert_eq!(
            server_view,
            expected,
            "seed {seed:#x} op {op_index}: every node agrees, and they agree on something the \
             operations did not ask for.\nhistory:\n  {}",
            history.join("\n  ")
        );
    }

    for path in paths {
        let _ = std::fs::remove_file(path);
    }
}

fn choose_and_apply(
    rng: &mut Xorshift,
    server: &mut Node,
    peers: &mut [Peer],
    model: &mut BTreeMap<ObjectId, ModelRow>,
    pool: &mut Vec<ObjectId>,
    op_index: usize,
) -> String {
    // Keep a row pool alive before anything else can act on it.
    if pool.len() < ROW_POOL && (pool.is_empty() || rng.below(4) == 0) {
        let owner = format!("owner{}", rng.below(3));
        let body = format!("b{op_index}");
        let values = HashMap::from([
            ("owner".to_string(), Value::Text(owner.clone())),
            ("body".to_string(), Value::Text(body.clone())),
            ("hits".to_string(), Value::BigInt(0)),
        ]);
        let ((row_id, _), _) = server
            .insert("docs", values, None)
            .expect("the server may insert");
        pool.push(row_id);
        model.insert(
            row_id,
            ModelRow {
                owner,
                body,
                hits: 0,
                deleted: false,
            },
        );
        return format!("insert(server, {row_id})");
    }

    let row_id = pool[rng.below(pool.len())];
    let entry = model.get_mut(&row_id).expect("pooled rows are modelled");

    match rng.below(10) {
        // Sender-side update — the plain delivery case.
        0..=2 => {
            let body = format!("s{op_index}");
            entry.body = body.clone();
            entry.hits += 1;
            let hits = entry.hits;
            let deleted = entry.deleted;
            if deleted {
                return format!("update(server, {row_id}) skipped: row is deleted");
            }
            server
                .update(
                    row_id,
                    vec![
                        ("body".to_string(), Value::Text(body)),
                        ("hits".to_string(), Value::BigInt(hits)),
                    ],
                    None,
                )
                .expect("the server may update a live row");
            format!("update(server, {row_id}, hits={hits})")
        }
        // A receiver writes on top of a row it was delivered, and that write travels back.
        3..=5 => {
            let peer_index = rng.below(peers.len());
            let body = format!("p{peer_index}-{op_index}");
            if entry.deleted {
                return format!("update(peer {peer_index}, {row_id}) skipped: row is deleted");
            }
            entry.body = body.clone();
            entry.hits += 1;
            let hits = entry.hits;
            peers[peer_index]
                .core
                .update(
                    row_id,
                    vec![
                        ("body".to_string(), Value::Text(body)),
                        ("hits".to_string(), Value::BigInt(hits)),
                    ],
                    None,
                )
                .expect("a peer may update a row it holds");
            format!("update(peer {peer_index}, {row_id}, hits={hits})")
        }
        // Delete and restore are not in this alphabet yet. They were kept out while a server-side
        // delete never reached a subscribed peer; `a_sender_delete_reaches_a_subscribed_peer`
        // below now pins that it does, and adding them here is the next step for this oracle.
        6..=8 => {
            let body = format!("s{op_index}-b");
            entry.body = body.clone();
            entry.hits += 1;
            let hits = entry.hits;
            server
                .update(
                    row_id,
                    vec![
                        ("body".to_string(), Value::Text(body)),
                        ("hits".to_string(), Value::BigInt(hits)),
                    ],
                    None,
                )
                .expect("the server may update a live row");
            format!("update(server, {row_id}, hits={hits})")
        }
        // Reconnect. The delivered-frontier cursor is per-peer, so a reconnect is exactly the
        // state where stamping must fall back to clearing rather than claim from memory.
        _ => {
            let peer_index = rng.below(peers.len());
            let client_id = peers[peer_index].client_id;
            // A `ClientRole::User` connection with no session is refused at the row-apply arm with
            // `SessionRequired` and never applies anything (inbox.rs, the User branch of
            // `process_from_client`). Passing None here silently turns every peer write into a
            // no-op, which is what made this harness accuse the engine of losing writes.
            server.add_client(client_id, Some(Session::new("alice")));
            subscribe(&mut peers[peer_index]);
            format!("reconnect(peer {peer_index})")
        }
    }
}

/// The smallest shape the randomized stream keeps failing on, written out by hand so the
/// failure is one page instead of a seed: the server owns a row, a peer receives it, the peer
/// writes on it, and the write travels back. Nothing here is exotic — it is what a second
/// device does the moment it touches a row it did not create.
/// This started life as an accusation against the engine and turned out to be an accusation
/// against its own fixture: the peer had no session, so every write it made was refused at the
/// row-apply arm and the harness measured a no-op. It stays as a gate because that failure was
/// silent — the batch fate still came back `DurableDirect`, so nothing in the transcript said
/// the write had been dropped.
#[test]
fn a_peer_write_on_a_delivered_row_reaches_the_sender() {
    let mut rng = Xorshift(0x5EED_0001);
    let schema = docs_schema();
    let app = "delivery-convergence-minimal";

    let (server_storage, server_path) = open_store("min-server", 0, 0);
    let mut server = runtime_over(
        schema.clone(),
        app,
        server_storage,
        Some(DurabilityTier::EdgeServer),
    );

    let (peer_storage, peer_path) = open_store("min-peer", 0, 1);
    let core = runtime_over(schema, app, peer_storage, None);
    let client_id = ClientId::new();
    let server_id = ServerId::new();
    let mut peer = Peer {
        core,
        server_id,
        client_id,
        subscription: None,
    };
    // A `ClientRole::User` connection with no session is refused at the row-apply arm with
    // `SessionRequired` and never applies anything (inbox.rs, the User branch of
    // `process_from_client`). Passing None here silently turns every peer write into a
    // no-op, which is what made this harness accuse the engine of losing writes.
    server.add_client(client_id, Some(Session::new("alice")));
    peer.core.add_server(server_id);
    subscribe(&mut peer);

    let mut peers = vec![peer];
    let mut network = Network::new();
    network.settle(&mut server, &mut peers, &mut rng);

    let ((row_id, _), _) = server
        .insert(
            "docs",
            HashMap::from([
                ("owner".to_string(), Value::Text("owner".into())),
                ("body".to_string(), Value::Text("from-server".into())),
                ("hits".to_string(), Value::BigInt(0)),
            ]),
            None,
        )
        .expect("the server may insert");
    network.settle(&mut server, &mut peers, &mut rng);

    assert_eq!(
        visible_docs(&mut peers[0].core),
        visible_docs(&mut server),
        "fixture precondition: the peer must have received the row before it can write on it"
    );

    peers[0]
        .core
        .update(
            row_id,
            vec![
                ("body".to_string(), Value::Text("from-peer".into())),
                ("hits".to_string(), Value::BigInt(7)),
            ],
            None,
        )
        .expect("a peer may update a row it holds");
    network.settle(&mut server, &mut peers, &mut rng);

    let expected: Snapshot = vec![(row_id, "owner".to_string(), "from-peer".to_string(), 7)];
    assert_eq!(
        visible_docs(&mut peers[0].core),
        expected,
        "the peer must hold its own write"
    );
    assert_eq!(
        visible_docs(&mut server),
        expected,
        "the sender must end up holding the peer's write. The server acknowledged the batch \
         as DurableDirect, so this is not a rejection — the batch is stored and simply never \
         becomes the visible row."
    );

    let _ = std::fs::remove_file(server_path);
    let _ = std::fs::remove_file(peer_path);
}

/// A sender on EdgeServer with one subscribed peer, settled, the peer holding a row the sender
/// inserted.
///
/// Peers are read through a local-only query: a query that went to the sender would register there
/// and replay the row, which is the delivery these gates are about.
struct SenderAndPeer {
    server: Node,
    peers: Vec<Peer>,
    network: Network,
    rng: Xorshift,
    row_id: ObjectId,
    app: String,
    tag: String,
    paths: Vec<std::path::PathBuf>,
}

impl SenderAndPeer {
    fn new(tag: &str, seed: u64) -> Self {
        let mut rng = Xorshift(seed);
        let schema = docs_schema();
        let app = format!("delivery-sender-{tag}");

        let (server_storage, server_path) = open_store(&format!("{tag}-server"), 0, 0);
        let mut server = runtime_over(
            schema.clone(),
            &app,
            server_storage,
            Some(DurabilityTier::EdgeServer),
        );

        let (peer_storage, peer_path) = open_store(&format!("{tag}-peer"), 0, 1);
        let core = runtime_over(schema, &app, peer_storage, None);
        let client_id = ClientId::new();
        let server_id = ServerId::new();
        let mut peer = Peer {
            core,
            server_id,
            client_id,
            subscription: None,
        };
        server.add_client(client_id, Some(Session::new("alice")));
        peer.core.add_server(server_id);
        subscribe(&mut peer);

        let mut peers = vec![peer];
        let mut network = Network::new();
        network.settle(&mut server, &mut peers, &mut rng);

        let ((row_id, _), _) = server
            .insert(
                "docs",
                HashMap::from([
                    ("owner".to_string(), Value::Text("owner".into())),
                    ("body".to_string(), Value::Text("from-server".into())),
                    ("hits".to_string(), Value::BigInt(0)),
                ]),
                None,
            )
            .expect("the server may insert");
        network.settle(&mut server, &mut peers, &mut rng);

        let branch = server.schema_manager().branch_name().to_string();
        assert_eq!(
            tip_count(&mut peers[0].core, row_id, &branch),
            1,
            "fixture precondition: the peer must hold the inserted row before it can miss a later \
             version of it"
        );

        Self {
            server,
            peers,
            network,
            rng,
            row_id,
            app,
            tag: tag.to_string(),
            paths: vec![server_path, peer_path],
        }
    }

    /// Connect and subscribe one more peer, not yet settled; returns its index.
    fn add_peer(&mut self) -> usize {
        let index = self.peers.len();
        let (storage, path) = open_store(&format!("{}-peer{index}", self.tag), 0, index + 1);
        self.paths.push(path);
        let core = runtime_over(docs_schema(), &self.app, storage, None);
        let mut peer = Peer {
            core,
            server_id: ServerId::new(),
            client_id: ClientId::new(),
            subscription: None,
        };
        self.server
            .add_client(peer.client_id, Some(Session::new("alice")));
        peer.core.add_server(peer.server_id);
        subscribe(&mut peer);
        self.peers.push(peer);
        index
    }

    /// Deliver an upstream's rejection of `batch_id` to the sender and let it act on it.
    fn reject_from_upstream(
        &mut self,
        upstream_id: ServerId,
        batch_id: crate::row_histories::BatchId,
    ) {
        self.server.park_sync_message(InboxEntry {
            source: Source::Server(upstream_id),
            payload: SyncPayload::BatchFate {
                fate: BatchFate::Rejected {
                    batch_id,
                    code: "permission_denied".to_string(),
                    reason: "upstream refused the write".to_string(),
                },
            },
        });
        self.server.batched_tick();
        self.server.immediate_tick();
    }

    fn settle(&mut self) {
        self.network
            .settle(&mut self.server, &mut self.peers, &mut self.rng);
    }

    fn update_values() -> Vec<(String, Value)> {
        vec![
            ("body".to_string(), Value::Text("updated".into())),
            ("hits".to_string(), Value::BigInt(1)),
        ]
    }

    fn row(&self, body: &str, hits: i64) -> Snapshot {
        vec![(self.row_id, "owner".to_string(), body.to_string(), hits)]
    }
}

fn batch_fate_interest_len(node: &mut Node) -> usize {
    node.schema_manager_mut()
        .query_manager_mut()
        .sync_manager()
        .batch_fate_interest_len()
}

fn batch_fate_interest_clients(node: &mut Node, batch_id: crate::row_histories::BatchId) -> usize {
    node.schema_manager_mut()
        .query_manager_mut()
        .sync_manager()
        .batch_fate_interest_clients(batch_id)
}

impl Drop for SenderAndPeer {
    fn drop(&mut self) {
        for path in &self.paths {
            let _ = std::fs::remove_file(path);
        }
    }
}

/// A sender's own update to a row a subscribed peer already holds reaches that peer, with its
/// fate.
///
/// A row reaches a client by one of two paths. A change arriving from another node is forwarded
/// to every client whose scope holds the row (`forward_update_to_clients_*`, from the inbox), and
/// a row entering a client's scope is replayed when the settle pass moves that scope. A node's
/// own write that leaves membership unchanged took neither: `finish_local_row_history_write`
/// forwarded to servers only, and `settle_server_subscriptions` sends rows only when the scope set
/// changes. The peer kept the version it was first given.
///
/// The randomized oracle above hid this. A peer without a durability tier sends even its
/// local-only reads to the server, where each one was registered and never removed, and every
/// such registration replayed the row at its current version. Once that leak was closed the
/// oracle went red on nearly every seed.
#[test]
fn a_sender_update_on_a_delivered_row_reaches_a_subscribed_peer() {
    let mut f = SenderAndPeer::new("upd", 0x5EED_0003);

    let batch_id = f
        .server
        .update(f.row_id, SenderAndPeer::update_values(), None)
        .expect("the server may update a live row");
    f.settle();

    let expected = f.row("updated", 1);
    assert_eq!(
        visible_docs(&mut f.server),
        expected,
        "fixture precondition: the sender must hold its own update"
    );
    assert_eq!(
        visible_docs(&mut f.peers[0].core),
        expected,
        "the peer is subscribed to docs and holds this row, so the sender's update must reach \
         it. The write changed no membership, so no scope moved and nothing was replayed."
    );
    assert_eq!(
        f.peers[0]
            .core
            .storage()
            .load_authoritative_batch_fate(batch_id)
            .expect("the peer's fate should read"),
        Some(BatchFate::DurableDirect {
            batch_id,
            confirmed_tier: DurabilityTier::EdgeServer,
        }),
        "the peer holds the update, so it must hold its fate too. A local write's fate is \
         recorded after the write itself, so a forward made inside the write goes out before \
         there is a fate to send, and nothing sends it later: the sweep leaves a batch confirmed \
         at the node's own tier alone."
    );
}

/// A sender's update inside an explicit direct batch reaches a subscribed peer when the batch
/// commits.
///
/// A write into an open batch is staged, so the write changes no visibility and forwards nothing.
/// Its rows become visible at commit, in `publish_direct_batch_rows`, which is a second place a
/// node makes its own version visible.
#[test]
fn a_sender_direct_batch_update_reaches_a_subscribed_peer_on_commit() {
    let mut f = SenderAndPeer::new("batch", 0x5EED_0004);

    let batch_id = f.server.begin_batch(BatchMode::Direct);
    let write_context = WriteContext {
        session: None,
        attribution: None,
        updated_at: None,
        batch_mode: None,
        batch_id: Some(batch_id),
        target_branch_name: None,
    };
    f.server
        .update(
            f.row_id,
            SenderAndPeer::update_values(),
            Some(&write_context),
        )
        .expect("the server may update inside its batch");
    f.server
        .commit_batch(batch_id)
        .expect("the server may commit its batch");
    f.settle();

    let expected = f.row("updated", 1);
    assert_eq!(
        visible_docs(&mut f.server),
        expected,
        "fixture precondition: the sender must hold its committed update"
    );
    assert_eq!(
        visible_docs(&mut f.peers[0].core),
        expected,
        "the batch committed and its update is visible on the sender, so the subscribed peer \
         holding the row must receive it"
    );
}

/// A sender's update that its upstream rejects is withdrawn from a subscribed peer.
///
/// Once a node's own update reaches its clients, a rejection from upstream has to follow it:
/// `mark_local_batch_rows_rejected` rolls the sender back to the version before, and a peer left
/// holding the rejected version must be told, as clients are for a rejected client write.
#[test]
fn a_sender_update_rejected_upstream_is_withdrawn_from_a_subscribed_peer() {
    let mut f = SenderAndPeer::new("reject", 0x5EED_0005);
    let upstream_id = ServerId::new();
    f.server.add_server(upstream_id);
    f.settle();

    let batch_id = f
        .server
        .update(f.row_id, SenderAndPeer::update_values(), None)
        .expect("the server may update a live row");
    f.settle();
    assert_eq!(
        visible_docs(&mut f.peers[0].core),
        f.row("updated", 1),
        "fixture precondition: the peer must hold the update before its rejection can be missed"
    );

    f.reject_from_upstream(upstream_id, batch_id);
    f.settle();

    let expected = f.row("from-server", 0);
    assert_eq!(
        visible_docs(&mut f.server),
        expected,
        "fixture precondition: the rejection must roll the sender back"
    );
    assert_eq!(
        visible_docs(&mut f.peers[0].core),
        expected,
        "the upstream rejected the update this peer was sent, so the peer must drop it as the \
         sender did"
    );
}

/// A sender's delete that took a row out of a subscribed peer's scope, then was rejected upstream,
/// is undone on that peer as well.
///
/// The delete reaches the peer while the row is still in its scope. The next settle pass drops the
/// row from that scope, and `prune_client_scope_tracking` drops with it the record that the peer
/// was sent the delete's batch. The rejection comes after that: the fate relay finds no client
/// interested in the batch, and the restored row re-enters the peer's scope as a version the peer
/// already holds, under a tombstone the peer was never told is void.
#[test]
fn a_sender_delete_rejected_upstream_after_leaving_scope_is_undone_on_a_subscribed_peer() {
    let mut f = SenderAndPeer::new("reject-delete", 0x5EED_0006);
    let upstream_id = ServerId::new();
    f.server.add_server(upstream_id);
    f.settle();

    let batch_id = f
        .server
        .delete(f.row_id, None)
        .expect("the server may delete a live row");
    f.settle();
    assert!(
        visible_docs(&mut f.peers[0].core).is_empty(),
        "fixture precondition: the peer must hold the delete before its rejection can be missed"
    );

    f.reject_from_upstream(upstream_id, batch_id);
    f.settle();

    let expected = f.row("from-server", 0);
    assert_eq!(
        visible_docs(&mut f.server),
        expected,
        "fixture precondition: the rejection must restore the row on the sender"
    );
    assert_eq!(
        visible_docs(&mut f.peers[0].core),
        expected,
        "the upstream rejected the delete this peer was sent, so the peer must show the row again \
         as the sender does"
    );
    assert_eq!(
        batch_fate_interest_clients(&mut f.server, batch_id),
        0,
        "a rejection is the batch's last fate, so the sender must stop keeping the peer on it"
    );
}

/// The same loss with a peer that confirms deliveries: the confirmed delete is recorded as sent
/// only when the confirmation arrives, which here is after the settle pass has dropped the row
/// from the peer's scope, so the delete's interest outlives that prune and the rejection gets
/// through without help. This pins that the kept interest changes nothing for such a peer.
#[test]
fn a_sender_delete_rejected_upstream_after_leaving_scope_is_undone_on_a_confirming_peer() {
    let mut f = SenderAndPeer::new("reject-delete-acks", 0x5EED_0009);
    let client_id = f.peers[0].client_id;
    f.server.set_client_acks_deliveries(client_id, true);
    f.peers[0].core.set_upstream_supports_delivery_acks(true);
    let upstream_id = ServerId::new();
    f.server.add_server(upstream_id);
    f.settle();

    let batch_id = f
        .server
        .delete(f.row_id, None)
        .expect("the server may delete a live row");
    f.settle();
    assert!(
        visible_docs(&mut f.peers[0].core).is_empty(),
        "fixture precondition: the peer must hold the delete before its rejection can be missed"
    );

    f.reject_from_upstream(upstream_id, batch_id);
    f.settle();

    let expected = f.row("from-server", 0);
    assert_eq!(
        visible_docs(&mut f.server),
        expected,
        "fixture precondition: the rejection must restore the row on the sender"
    );
    assert_eq!(
        visible_docs(&mut f.peers[0].core),
        expected,
        "the upstream rejected the delete this confirming peer was sent, so the peer must show \
         the row again as the sender does"
    );
    assert_eq!(batch_fate_interest_clients(&mut f.server, batch_id), 0);
}

/// A peer's delete relayed by the sender to another subscribed peer, then rejected upstream after
/// the row left that peer's scope, is undone there as well.
///
/// The same loss as for the sender's own delete, on the path that forwarded changes to clients
/// before local writes were: an arrival is forwarded while the row is in scope, and the scope
/// prune forgets that it was.
#[test]
fn a_relayed_delete_rejected_upstream_after_leaving_scope_is_undone_on_another_peer() {
    let mut f = SenderAndPeer::new("reject-relayed", 0x5EED_0008);
    let upstream_id = ServerId::new();
    f.server.add_server(upstream_id);
    let writer = f.add_peer();
    f.settle();
    assert_eq!(
        visible_docs(&mut f.peers[writer].core),
        f.row("from-server", 0),
        "fixture precondition: the writer must hold the row before it can delete it"
    );

    let batch_id = f.peers[writer]
        .core
        .delete(f.row_id, None)
        .expect("the writer may delete a live row");
    f.settle();
    assert!(
        visible_docs(&mut f.server).is_empty() && visible_docs(&mut f.peers[0].core).is_empty(),
        "fixture precondition: the writer's delete must reach the sender and the other peer"
    );

    f.reject_from_upstream(upstream_id, batch_id);
    f.settle();

    let expected = f.row("from-server", 0);
    assert_eq!(
        visible_docs(&mut f.server),
        expected,
        "fixture precondition: the rejection must restore the row on the sender"
    );
    assert_eq!(
        visible_docs(&mut f.peers[0].core),
        expected,
        "the upstream rejected the delete this peer was relayed, so the peer must show the row \
         again as the sender does"
    );
    assert_eq!(
        batch_fate_interest_clients(&mut f.server, batch_id),
        0,
        "a rejection is the batch's last fate, so the sender must stop keeping the peer on it"
    );
}

/// A peer that subscribed after a sender's update holds only that version; when the update is
/// rejected upstream, the peer is sent the version that stands again.
///
/// A scope replay sends a row's current version with its parents stripped, so this peer's history
/// for the row is the rejected version alone. Rolling it back leaves the peer nothing to fall back
/// to: the rejected version reads as the row's insert, and the row goes. The peer's scope on the
/// sender does not change, because the restored version matches its query as the rejected one did,
/// so no settle pass sends the row again. The sender has to send it when it withdraws the version.
#[test]
fn a_sender_update_rejected_upstream_is_restored_on_a_peer_that_subscribed_after_it() {
    let mut f = SenderAndPeer::new("reject-late-peer", 0x5EED_0007);
    let upstream_id = ServerId::new();
    f.server.add_server(upstream_id);
    f.settle();

    let batch_id = f
        .server
        .update(f.row_id, SenderAndPeer::update_values(), None)
        .expect("the server may update a live row");
    f.settle();
    let late = f.add_peer();
    f.settle();
    assert_eq!(
        visible_docs(&mut f.peers[late].core),
        f.row("updated", 1),
        "fixture precondition: the late peer must hold the update"
    );
    assert_eq!(
        f.peers[late]
            .core
            .storage()
            .scan_history_row_batches("docs", f.row_id)
            .expect("history should read")
            .len(),
        1,
        "fixture precondition: the late peer must hold the update alone, as a scope replay sends it"
    );

    f.reject_from_upstream(upstream_id, batch_id);
    f.settle();

    let expected = f.row("from-server", 0);
    assert_eq!(
        visible_docs(&mut f.server),
        expected,
        "fixture precondition: the rejection must roll the sender back"
    );
    assert_eq!(
        visible_docs(&mut f.peers[late].core),
        expected,
        "the late peer was sent only the rejected update, so it must be sent the version that \
         stands again"
    );
}

/// A sender's delete of a row a subscribed peer holds reaches that peer.
///
/// This stood as an open finding blamed on the scope gate in `queue_row_to_client`: a deleted row
/// leaves the subscription's scope, so its tombstone would be dropped for the very reason that it
/// is gone. That was not the mechanism. A delete arriving through the inbox is forwarded on
/// arrival, while the row is still in scope, and a delete the sender made itself was never
/// forwarded to clients at all, like every other local write. Now that local writes are
/// forwarded, the tombstone goes out before the settle pass drops the row from scope.
#[test]
fn a_sender_delete_reaches_a_subscribed_peer() {
    let mut rng = Xorshift(0x5EED_0002);
    let schema = docs_schema();
    let app = "delivery-delete-propagation";

    let (server_storage, server_path) = open_store("del-server", 0, 0);
    let mut server = runtime_over(
        schema.clone(),
        app,
        server_storage,
        Some(DurabilityTier::EdgeServer),
    );

    let (peer_storage, peer_path) = open_store("del-peer", 0, 1);
    let core = runtime_over(schema, app, peer_storage, None);
    let client_id = ClientId::new();
    let server_id = ServerId::new();
    let mut peer = Peer {
        core,
        server_id,
        client_id,
        subscription: None,
    };
    server.add_client(client_id, Some(Session::new("alice")));
    peer.core.add_server(server_id);
    subscribe(&mut peer);

    let mut peers = vec![peer];
    let mut network = Network::new();
    network.settle(&mut server, &mut peers, &mut rng);

    let ((row_id, _), _) = server
        .insert(
            "docs",
            HashMap::from([
                ("owner".to_string(), Value::Text("owner".into())),
                ("body".to_string(), Value::Text("doomed".into())),
                ("hits".to_string(), Value::BigInt(0)),
            ]),
            None,
        )
        .expect("the server may insert");
    network.settle(&mut server, &mut peers, &mut rng);

    assert_eq!(
        visible_docs(&mut peers[0].core).len(),
        1,
        "fixture precondition: the peer must hold the row before the delete can be missed"
    );

    server.delete(row_id, None).expect("the server may delete");
    network.settle(&mut server, &mut peers, &mut rng);

    assert!(
        visible_docs(&mut server).is_empty(),
        "fixture precondition: the server must have dropped the row itself"
    );
    assert!(
        visible_docs(&mut peers[0].core).is_empty(),
        "a peer that was subscribed to this row when it was deleted must learn that it was \
         deleted. It cannot ask: nothing in the client direction requests a batch. So the one \
         payload carrying the tombstone is the delivery, and the scope gate drops it because \
         the deletion removed the row from the scope that would have carried it. The peer \
         serves a deleted row as live indefinitely."
    );
    assert_eq!(
        batch_fate_interest_len(&mut server),
        0,
        "with no upstream nothing can reject the delete later, so the row leaving the peer's \
         scope must not keep the peer waiting for its fate"
    );

    let _ = std::fs::remove_file(server_path);
    let _ = std::fs::remove_file(peer_path);
}
