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
//! would fix that. And delete/restore are excluded from the alphabet (see
//! `a_deleted_row_never_reaches_a_subscribed_peer`), so the delete-winner corruption is outside
//! the generated space.

use super::*;
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
/// The peer that plays the sender: every "sender-side" write is issued here and reaches the
/// server through its inbox, which is the only path a write takes in production. A write made
/// on the server node itself is sealed upstream and is NOT forwarded to the node's own clients
/// (`forward_update_to_clients*` in `sync_manager/forwarding.rs` is called only from the inbox).
/// An insert still arrives at the peers, because the next settle of their standing subscription
/// sees the row enter its scope and offers it as a scope growth; an update of a row already in
/// scope does not, since nothing re-offers a row the scope already holds. Until linsa-v18 this
/// oracle converged on such updates anyway, because its own measurement reads (then one-shot
/// queries, which an identity-less node registers upstream even as LocalOnly) left a zombie
/// subscription at the server on every op, and each zombie's first settle re-offered every
/// row. Withdrawing a parked registration on its own unsubscription (v18 item 1) halved the
/// zombies and exposed this; the measurement now reads storage and registers nothing.
const WRITER: usize = 0;
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
    // A measurement must not move what it measures: a query, even LocalOnly, is a subscription
    // that an identity-less node registers upstream, and its first settle at the server
    // re-offers rows. So read the visible rows straight from storage, on the composed branch.
    let branch = core.schema_manager().branch_name().to_string();
    let descriptor = docs_schema()
        .get(&"docs".into())
        .expect("docs table")
        .columns
        .clone();
    let rows = core
        .storage()
        .scan_visible_region("docs", &branch)
        .expect("visible docs should scan");
    let mut out: Snapshot = rows
        .into_iter()
        .filter(|row| row.state.is_visible() && !row.is_deleted)
        .map(|row| {
            let values = crate::row_format::decode_row(&descriptor, &row.data)
                .expect("a stored docs row should decode");
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
            (row.row_id, owner, body, hits)
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
    let ops = std::env::var("DIFF_OPS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(OPS_PER_SEED);
    for op_index in 0..ops {
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
                "seed {seed:#x} op {op_index}: peer {index} did not converge with the server. \
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
        // One un-superseded tip per writing author — every peer writes here, the writer peer
        // for the sender-side arms and any peer for the receiver arm; the server authors
        // nothing — plus one authority snapshot that no local write has named yet, plus
        // headroom. Measured across these six seeds with the writes routed through the peers
        // and the measurement reading storage (v18): peak 6 (seed 1), 5 and 4 on the others,
        // at `OPS_PER_SEED` = 90, two peers, so the bound is 7. With the rule disarmed the count is unbounded: 23 within 90 ops on the
        // original harness, 460 in the field. The headroom is deliberate — a bound sitting
        // exactly on the observed peak is not a gate, it is a coincidence waiting for the next
        // seed.
        //
        // Open finding, recorded and not gated here: a 300-op run reaches 9 tips on the server
        // (seeds 1–6: 7, 6, 8, 9, 8, 6; peers stay at ≤ 6), and the base engine at c9ec20fb0
        // gives the same numbers with this harness, so the drift predates v18. It was invisible
        // while the measurement itself re-delivered every row through a zombie subscription
        // each op. Slow growth is not the accumulation-per-delivery this oracle was built for,
        // but it is growth; see `archive/v18/PLAN.md` (decisions log, item 1 round 2). Probe it
        // with `DIFF_OPS=300 DIFF_TIP_SERIES=1 … -- delivered_rows_converge --nocapture`,
        // which lifts the bound and prints the peak after every op instead of asserting.
        let writing_authors = peers.len();
        let probe = std::env::var("DIFF_TIP_SERIES").is_ok();
        let author_bound = if probe {
            usize::MAX
        } else {
            writing_authors + 5
        };
        for (row_id, _, _, _) in &server_view {
            let branch = server_branch.clone();
            let server_tips = tip_count(&mut server, *row_id, &branch);
            if probe {
                eprintln!(
                    "tip-series seed={seed:#x} op={op_index} row={row_id} server_tips={server_tips}"
                );
            }
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
        let ((row_id, _), _) = peers[WRITER]
            .core
            .insert("docs", values, None)
            .expect("the writer may insert");
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
        return format!("insert(writer, {row_id})");
    }

    let row_id = pool[rng.below(pool.len())];
    let entry = model.get_mut(&row_id).expect("pooled rows are modelled");

    match rng.below(10) {
        // Sender-side update — the plain delivery case: the writer's batch reaches the server
        // and is forwarded to every other client whose scope holds the row.
        0..=2 => {
            let body = format!("s{op_index}");
            entry.body = body.clone();
            entry.hits += 1;
            let hits = entry.hits;
            let deleted = entry.deleted;
            if deleted {
                return format!("update(writer, {row_id}) skipped: row is deleted");
            }
            peers[WRITER]
                .core
                .update(
                    row_id,
                    vec![
                        ("body".to_string(), Value::Text(body)),
                        ("hits".to_string(), Value::BigInt(hits)),
                    ],
                    None,
                )
                .expect("the writer may update a live row");
            format!("update(writer, {row_id}, hits={hits})")
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
        // Delete and restore are deliberately NOT in this alphabet. A server-side delete never
        // reaches a subscribed peer at all — see
        // `a_deleted_row_never_reaches_a_subscribed_peer` below, which pins that as its own
        // measured defect. Including an operation whose outcome is governed by a different,
        // still-open defect would keep this oracle permanently red and mask everything it was
        // built to find.
        6..=8 => {
            let body = format!("s{op_index}-b");
            entry.body = body.clone();
            entry.hits += 1;
            let hits = entry.hits;
            peers[WRITER]
                .core
                .update(
                    row_id,
                    vec![
                        ("body".to_string(), Value::Text(body)),
                        ("hits".to_string(), Value::BigInt(hits)),
                    ],
                    None,
                )
                .expect("the writer may update a live row");
            format!("update(writer, {row_id}, hits={hits})")
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

/// A server-side delete never reaches a subscribed peer.
///
/// MEASURED, not inferred: across a full randomized run of the oracle above — 90 row
/// deliveries traced payload by payload — not one delivery carried `is_deleted`. The peer goes
/// on serving the row as live forever.
///
/// Mechanism, verified as far as the code goes: deletion is not a `RowState`. A delete is a
/// `VisibleDirect` batch carrying `is_deleted`/`delete_kind`
/// (`row_histories/types.rs:111-123`), so `load_current_row_from_storage`
/// (`sync_manager/forwarding.rs:120-145`) loads the tombstone perfectly well — both its
/// visible-region branch and its history fallback accept it. What refuses it is the scope gate
/// in `queue_row_to_client`: `if !in_scope { return; }`. A deleted row no longer matches the
/// subscription that put it in scope, so the one payload that would tell the peer it is gone is
/// dropped for the precise reason that it is gone. `prune_client_scope_tracking`
/// (`sync_manager/mod.rs:1233-1269`) then clears the bookkeeping and sends the client nothing.
///
/// The step not directly instrumented is the scope recomputation itself — that the delete is
/// what removes the row from the client's scope, rather than some earlier refusal. Everything
/// either side of it is measured.
///
/// This is NOT the parent-stripping defect and is not fixed by the elided-snapshot frontier
/// rule; it reproduces identically with that rule armed. It is recorded here rather than folded
/// into the oracle above, because an alphabet containing an operation governed by a separate
/// open defect can never go green and would mask every other finding.
#[ignore = "open finding: a delete is refused by the scope gate that its own effect triggers"]
#[test]
fn a_deleted_row_never_reaches_a_subscribed_peer() {
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

    let _ = std::fs::remove_file(server_path);
    let _ = std::fs::remove_file(peer_path);
}
