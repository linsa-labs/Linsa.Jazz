//! Include routing and settle skips on the phone's own read path.
//!
//! The RN client subscribes sync-backed at `edge`, `Immediate`. On that path the row loader
//! depends on subscription state the routing and skip gates never ran under: the read tier,
//! `pending_local_row_ids`, the source overlay of the author's own unconfirmed writes (under
//! which a clean scan still rescans), and the server's scope answers. The measured wins of the
//! routed engine — a contact's presence beat, the author's own confirmation pass — are on this
//! path, so it is where routed-with-skips has to be shown to serve exactly what the reference
//! engine (broadcast, nothing skipped) serves.
//!
//! Two worlds receive the same operations in lockstep: an author subscribed like the app to
//! `chats -> members -> person` (all chats, and a three-chat window), a sync server, and a
//! contact writing the `people` rows the author's members name. The author's writes land in the
//! overlay; the server's answers are delivered, held back across later writes, or lose their
//! fates on the way. After every operation both worlds must have SENT their subscribers the
//! same trees. SQLite, the phone's backend.

use std::collections::BTreeMap;

use super::*;
use crate::query_manager::graph_nodes::include_routing::{
    IncludeEmptyProbeMode, IncludeRoutingMode, force_include_empty_probe, force_include_routing,
};
use crate::query_manager::manager::LocalUpdates;
use crate::query_manager::precise_dirty::{PreciseDirtyMode, force_precise_dirty};
use crate::query_manager::settle_skips::{SettleSkipsMode, force_settle_skips};
use crate::row_format::decode_row;
use crate::storage::SqliteStorage;
use crate::sync_manager::QueryPropagation;

type SqliteCore = RuntimeCore<SqliteStorage, NoopScheduler>;

const CHATS: &str = "chats";
const MEMBERS: &str = "members";
const PEOPLE: &str = "people";
const WINDOW: usize = 3;

fn schema() -> Schema {
    SchemaBuilder::new()
        .table(TableSchema::builder(CHATS).column("title", ColumnType::Text))
        .table(
            TableSchema::builder(MEMBERS)
                .column("chat_id", ColumnType::Uuid)
                .column("person_id", ColumnType::Uuid),
        )
        .table(TableSchema::builder(PEOPLE).column("name", ColumnType::Text))
        .build()
}

fn core_in(dirs: &mut Vec<tempfile::TempDir>, sync_manager: SyncManager) -> SqliteCore {
    let dir = tempfile::TempDir::new().unwrap();
    let schema_manager = SchemaManager::new(
        sync_manager,
        schema(),
        AppId::from_name("settle-skips-tiered"),
        "dev",
        "main",
    )
    .unwrap();
    let storage = SqliteStorage::open(dir.path().join("jazz.sqlite")).unwrap();
    let mut core = new_test_core(schema_manager, storage, NoopScheduler);
    core.immediate_tick();
    dirs.push(dir);
    core
}

fn settle(core: &mut SqliteCore) {
    core.batched_tick();
    core.immediate_tick();
    core.batched_tick();
}

/// The engine paths a world runs on, pinned while it runs: the routed engine with real settle
/// skips, or the reference — broadcast, nothing skipped. Taken in the order the other gates
/// take them, so none deadlock.
struct Engaged {
    _precise: PreciseDirtyMode,
    _routing: IncludeRoutingMode,
    _empty_probe: IncludeEmptyProbeMode,
    _skips: SettleSkipsMode,
}

fn engage(routed: bool) -> Engaged {
    let precise = force_precise_dirty(true);
    let routing = force_include_routing(routed);
    let empty_probe = force_include_empty_probe(true);
    Engaged {
        _precise: precise,
        _routing: routing,
        _empty_probe: empty_probe,
        _skips: force_settle_skips(routed),
    }
}

/// What a subscriber has been SENT: chat id -> its decoded values, and the chats in the order
/// the positions it was sent put them.
#[derive(Clone, Default)]
struct Mirror(
    Arc<Mutex<HashMap<ObjectId, Vec<Value>>>>,
    Arc<Mutex<Vec<ObjectId>>>,
);

/// chat -> (title, [(member, [(person, name)])]), members by id.
type Tree = BTreeMap<ObjectId, (String, Vec<(ObjectId, Vec<(ObjectId, String)>)>)>;

impl Mirror {
    fn apply(&self, delta: SubscriptionDelta) {
        let decode = |data: &[u8]| decode_row(&delta.descriptor, data).expect("a sent row decodes");
        let mut rows = self.0.lock().unwrap();
        for removed in &delta.ordered_delta.removed {
            rows.remove(&removed.id);
        }
        for added in &delta.ordered_delta.added {
            rows.insert(added.id, decode(added.row.data.as_ref()));
        }
        for updated in &delta.ordered_delta.updated {
            if let Some(row) = &updated.row {
                rows.insert(updated.id, decode(row.data.as_ref()));
            }
        }
        // Positions: what leaves or moves comes out, then what arrives or moved goes in at its
        // new index, lowest first.
        let mut order = self.1.lock().unwrap();
        let leaving: std::collections::HashSet<ObjectId> = delta
            .ordered_delta
            .removed
            .iter()
            .map(|removed| removed.id)
            .chain(
                delta
                    .ordered_delta
                    .updated
                    .iter()
                    .filter(|updated| updated.old_index != updated.new_index)
                    .map(|updated| updated.id),
            )
            .collect();
        order.retain(|id| !leaving.contains(id));
        let mut arriving: Vec<(usize, ObjectId)> = delta
            .ordered_delta
            .added
            .iter()
            .map(|added| (added.index, added.id))
            .chain(
                delta
                    .ordered_delta
                    .updated
                    .iter()
                    .filter(|updated| updated.old_index != updated.new_index)
                    .map(|updated| (updated.new_index, updated.id)),
            )
            .collect();
        arriving.sort();
        for (index, id) in arriving {
            let at = index.min(order.len());
            order.insert(at, id);
        }
    }

    fn order(&self) -> Vec<ObjectId> {
        self.1.lock().unwrap().clone()
    }

    fn tree(&self) -> Tree {
        let text = |value: Option<&Value>| match value {
            Some(Value::Text(text)) => text.clone(),
            other => panic!("expected text, got {other:?}"),
        };
        let array = |value: Option<&Value>| match value {
            Some(Value::Array(items)) => items.clone(),
            other => panic!("expected an include array, got {other:?}"),
        };
        let row = |value: &Value| match value {
            Value::Row {
                id: Some(id),
                values,
            } => (*id, values.clone()),
            other => panic!("expected an included row with its id, got {other:?}"),
        };
        self.0
            .lock()
            .unwrap()
            .iter()
            .map(|(chat, values)| {
                let mut members: Vec<_> = array(values.get(1))
                    .iter()
                    .map(|member| {
                        let (member_id, member_values) = row(member);
                        let mut person: Vec<_> = array(member_values.get(2))
                            .iter()
                            .map(|person| {
                                let (person_id, person_values) = row(person);
                                (person_id, text(person_values.first()))
                            })
                            .collect();
                        person.sort();
                        (member_id, person)
                    })
                    .collect();
                members.sort();
                (*chat, (text(values.first()), members))
            })
            .collect()
    }
}

#[derive(Clone, Copy, Debug)]
enum Delivery {
    /// Everything held, then this round's answers, delivered.
    All,
    /// This round's answers held back.
    Hold,
    /// This round's answers delivered without their fates.
    LoseFates,
}

#[derive(Clone, Debug)]
enum Op {
    InsertChat(ObjectId, String),
    RetitleChat(ObjectId, String),
    InsertMember(ObjectId, ObjectId, ObjectId),
    RepointMember(ObjectId, ObjectId),
    DeleteMember(ObjectId),
    /// The author writes a person row of its own: an overlay row of the included table.
    AuthorPerson(ObjectId, String),
    /// A contact's row, written by the peer: remote for the author, with its fate.
    ContactInsert(ObjectId, String),
    /// A contact's presence beat.
    ContactRename(ObjectId, String),
    ContactDelete(ObjectId),
    Exchange(Delivery),
}

struct World {
    routed: bool,
    author: SqliteCore,
    server: SqliteCore,
    peer: SqliteCore,
    author_client: ClientId,
    author_server: ServerId,
    peer_client: ClientId,
    peer_server: ServerId,
    all: Mirror,
    window: Mirror,
    held: Vec<SyncPayload>,
    _dirs: Vec<tempfile::TempDir>,
}

fn chats_query(window: Option<usize>) -> Query {
    let builder = QueryBuilder::new(CHATS)
        .with_array("members", |sub| {
            sub.from(MEMBERS)
                .correlate("chat_id", "chats.id")
                .with_array("person", |nested| {
                    nested.from(PEOPLE).correlate("id", "members.person_id")
                })
        })
        .order_by("title");
    match window {
        Some(window) => builder.limit(window).build(),
        None => builder.build(),
    }
}

impl World {
    fn new(routed: bool) -> Self {
        let _engaged = engage(routed);
        let mut dirs = Vec::new();
        let mut author = core_in(&mut dirs, SyncManager::new());
        let mut server = core_in(
            &mut dirs,
            SyncManager::new().with_durability_tier(DurabilityTier::GlobalServer),
        );
        let mut peer = core_in(&mut dirs, SyncManager::new());
        let (author_client, author_server) = (ClientId::new(), ServerId::new());
        let (peer_client, peer_server) = (ClientId::new(), ServerId::new());
        server.add_client(author_client, Some(Session::new("alice")));
        server.add_client(peer_client, Some(Session::new("bob")));
        author.add_server(author_server);
        peer.add_server(peer_server);
        let mut world = Self {
            routed,
            author,
            server,
            peer,
            author_client,
            author_server,
            peer_client,
            peer_server,
            all: Mirror::default(),
            window: Mirror::default(),
            held: Vec::new(),
            _dirs: dirs,
        };
        for (window, mirror) in [
            (None, world.all.clone()),
            (Some(WINDOW), world.window.clone()),
        ] {
            world
                .author
                .subscribe_with_durability_and_propagation(
                    chats_query(window),
                    move |delta| mirror.apply(delta),
                    None,
                    ReadDurabilityOptions {
                        tier: Some(DurabilityTier::EdgeServer),
                        local_updates: LocalUpdates::Immediate,
                    },
                    QueryPropagation::Full,
                )
                .expect("subscribe like the app");
        }
        world.author.immediate_tick();
        world.exchange(Delivery::All);
        world
    }

    fn apply(&mut self, op: &Op) {
        let _engaged = engage(self.routed);
        let text = |value: &str| Value::Text(value.to_string());
        match op {
            Op::InsertChat(id, title) => {
                self.author
                    .insert_with_id(
                        CHATS,
                        HashMap::from([("title".to_string(), text(title))]),
                        Some(*id),
                        None,
                    )
                    .expect("insert chat");
            }
            Op::RetitleChat(id, title) => {
                self.author
                    .update(*id, vec![("title".to_string(), text(title))], None)
                    .expect("retitle chat");
            }
            Op::InsertMember(id, chat, person) => {
                self.author
                    .insert_with_id(
                        MEMBERS,
                        HashMap::from([
                            ("chat_id".to_string(), Value::Uuid(*chat)),
                            ("person_id".to_string(), Value::Uuid(*person)),
                        ]),
                        Some(*id),
                        None,
                    )
                    .expect("insert member");
            }
            Op::RepointMember(id, person) => {
                self.author
                    .update(
                        *id,
                        vec![("person_id".to_string(), Value::Uuid(*person))],
                        None,
                    )
                    .expect("re-point member");
            }
            Op::DeleteMember(id) => {
                self.author.delete(*id, None).expect("delete member");
            }
            Op::AuthorPerson(id, name) => {
                self.author
                    .insert_with_id(
                        PEOPLE,
                        HashMap::from([("name".to_string(), text(name))]),
                        Some(*id),
                        None,
                    )
                    .expect("insert own person");
            }
            Op::ContactInsert(id, name) => {
                self.peer
                    .insert_with_id(
                        PEOPLE,
                        HashMap::from([("name".to_string(), text(name))]),
                        Some(*id),
                        None,
                    )
                    .expect("insert contact");
            }
            Op::ContactRename(id, name) => {
                self.peer
                    .update(*id, vec![("name".to_string(), text(name))], None)
                    .expect("rename contact");
            }
            Op::ContactDelete(id) => {
                self.peer.delete(*id, None).expect("delete contact");
            }
            Op::Exchange(delivery) => {
                self.exchange(*delivery);
                return;
            }
        }
        self.author.immediate_tick();
        self.peer.immediate_tick();
    }

    /// One round: both clients' outboxes to the server, the server settles, the peer gets its
    /// answers, the author's are delivered as `delivery` says.
    fn exchange(&mut self, delivery: Delivery) {
        for (client, server_id, client_id) in [
            (&mut self.author, self.author_server, self.author_client),
            (&mut self.peer, self.peer_server, self.peer_client),
        ] {
            client.batched_tick();
            for entry in client.sync_sender().take() {
                if entry.destination == Destination::Server(server_id) {
                    self.server.park_sync_message(InboxEntry {
                        source: Source::Client(client_id),
                        payload: entry.payload,
                    });
                }
            }
        }
        settle(&mut self.server);
        let mut answers = Vec::new();
        for entry in self.server.sync_sender().take() {
            if entry.destination == Destination::Client(self.author_client) {
                answers.push(entry.payload);
            } else if entry.destination == Destination::Client(self.peer_client) {
                self.peer.park_sync_message(InboxEntry {
                    source: Source::Server(self.peer_server),
                    payload: entry.payload,
                });
            }
        }
        settle(&mut self.peer);
        let deliver: Vec<SyncPayload> = match delivery {
            Delivery::All => {
                let mut all = std::mem::take(&mut self.held);
                all.extend(answers);
                all
            }
            Delivery::Hold => {
                self.held.extend(answers);
                return;
            }
            Delivery::LoseFates => answers
                .into_iter()
                .filter(|payload| !matches!(payload, SyncPayload::BatchFate { .. }))
                .collect(),
        };
        for payload in deliver {
            self.author.park_sync_message(InboxEntry {
                source: Source::Server(self.author_server),
                payload,
            });
        }
        self.author.batched_tick();
        self.author.immediate_tick();
    }

    fn overlay_rows(&self) -> usize {
        self.author
            .schema_manager()
            .query_manager()
            .pending_local_row_count()
    }
}

struct Prng(u64);

impl Prng {
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

    fn pick(&mut self, items: &[ObjectId]) -> Option<ObjectId> {
        (!items.is_empty()).then(|| items[self.below(items.len())])
    }
}

fn run(seed: u64) {
    let mut rng = Prng(seed | 1);
    let mut counter = 0u64;
    let mut mint = || {
        counter += 1;
        ObjectId::from_uuid(uuid::Uuid::from_u128(
            (u128::from(seed) << 64) | u128::from(counter),
        ))
    };
    let mut worlds = [World::new(true), World::new(false)];
    let mut chats: Vec<ObjectId> = Vec::new();
    let mut members: Vec<ObjectId> = Vec::new();
    let mut contacts: Vec<ObjectId> = Vec::new();
    let mut own_people: Vec<ObjectId> = Vec::new();
    let mut dangling: Vec<ObjectId> = Vec::new();
    let mut steps_with_overlay = 0;
    let mut steps_with_a_named_person = 0;

    for step in 0..STEPS {
        let roll = rng.below(100);
        let op = match roll {
            0..=7 => {
                let id = mint();
                chats.push(id);
                Op::InsertChat(id, format!("chat-{:03}-{step:03}", rng.below(1000)))
            }
            8..=11 if !chats.is_empty() => Op::RetitleChat(
                rng.pick(&chats).unwrap(),
                format!("chat-{:03}-{step:03}", rng.below(1000)),
            ),
            12..=27 if !chats.is_empty() => {
                let person = match rng.below(4) {
                    0 => rng.pick(&own_people),
                    1 => rng.pick(&dangling),
                    _ => rng.pick(&contacts),
                }
                .unwrap_or_else(|| {
                    let fresh = mint();
                    dangling.push(fresh);
                    fresh
                });
                let id = mint();
                members.push(id);
                Op::InsertMember(id, rng.pick(&chats).unwrap(), person)
            }
            28..=33 if !members.is_empty() => {
                let person = if rng.below(2) == 0 {
                    rng.pick(&contacts)
                } else {
                    rng.pick(&dangling)
                }
                .unwrap_or_else(|| {
                    let fresh = mint();
                    dangling.push(fresh);
                    fresh
                });
                Op::RepointMember(rng.pick(&members).unwrap(), person)
            }
            34..=36 if !members.is_empty() => {
                let id = members.swap_remove(rng.below(members.len()));
                Op::DeleteMember(id)
            }
            37..=40 => {
                // An own person, sometimes the one a member is already waiting for.
                let id = if !dangling.is_empty() && rng.below(2) == 0 {
                    dangling.swap_remove(rng.below(dangling.len()))
                } else {
                    mint()
                };
                own_people.push(id);
                Op::AuthorPerson(id, format!("own-{step}"))
            }
            41..=48 => {
                let id = if !dangling.is_empty() && rng.below(2) == 0 {
                    dangling.swap_remove(rng.below(dangling.len()))
                } else {
                    mint()
                };
                contacts.push(id);
                Op::ContactInsert(id, format!("contact-{step}"))
            }
            49..=68 if !contacts.is_empty() => {
                Op::ContactRename(rng.pick(&contacts).unwrap(), format!("beat-{step}"))
            }
            69..=70 if !contacts.is_empty() => {
                let id = contacts.swap_remove(rng.below(contacts.len()));
                Op::ContactDelete(id)
            }
            71..=84 => Op::Exchange(Delivery::All),
            85..=93 => Op::Exchange(Delivery::Hold),
            94..=99 => Op::Exchange(Delivery::LoseFates),
            _ => Op::Exchange(Delivery::All),
        };
        for world in &mut worlds {
            world.apply(&op);
        }
        let [routed, reference] = &worlds;
        let (routed_all, reference_all) = (routed.all.tree(), reference.all.tree());
        assert_eq!(
            routed_all, reference_all,
            "seed {seed:#x} step {step} after {op:?}: the routed engine with settle skips must \
             send the app-shaped subscription exactly what the reference engine sends"
        );
        assert_eq!(
            routed.window.tree(),
            reference.window.tree(),
            "seed {seed:#x} step {step} after {op:?}: the three-chat window must match too"
        );
        assert_eq!(
            (routed.all.order(), routed.window.order()),
            (reference.all.order(), reference.window.order()),
            "seed {seed:#x} step {step} after {op:?}: the chats must be sent in the same order"
        );
        let by_title = |tree: &Tree| {
            let mut keyed: Vec<(String, ObjectId)> = tree
                .iter()
                .map(|(id, (title, _))| (title.clone(), *id))
                .collect();
            keyed.sort();
            keyed.into_iter().map(|(_, id)| id).collect::<Vec<_>>()
        };
        assert_eq!(
            reference.all.order(),
            by_title(&reference_all),
            "seed {seed:#x} step {step}: fixture precondition: the mirror puts the chats where              the positions sent say, which for the whole list is by title"
        );
        if routed.overlay_rows() > 0 {
            steps_with_overlay += 1;
        }
        if routed_all
            .values()
            .any(|(_, members)| members.iter().any(|(_, person)| !person.is_empty()))
        {
            steps_with_a_named_person += 1;
        }
    }
    assert!(
        steps_with_overlay > STEPS / 4 && steps_with_a_named_person > STEPS / 4,
        "seed {seed:#x}: fixture precondition: the run must spend its time with own writes in \
         the overlay ({steps_with_overlay}) and members showing a person \
         ({steps_with_a_named_person})"
    );
}

const STEPS: usize = 200;

#[cfg(feature = "sqlite")]
#[test]
fn routed_settle_with_skips_serves_the_app_path_exactly_as_the_reference_engine() {
    for seed in [0x5c1b_0001_u64, 0x5c1b_0002, 0x5c1b_0003, 0x5c1b_0004] {
        run(seed);
    }
}
