//! A read made again by an app that was closed and opened.
//!
//! An app keeps its client id across launches and numbers its queries from the start in
//! each one, so the first reads of a launch reach the server under ids the server still
//! holds subscriptions for — the last launch's, kept until the client is reaped. When the
//! query is the same one too, the server takes the subscription for a replay by a peer that
//! already has its answer, and says nothing. The engine that asked is a new one: it holds no
//! answer, and a read that waits for the server's tier waits for ever.
//!
//! Measured on the app (2026-10-01): a chat reopened at the position the last launch saved
//! made the page query it had made before; the server logged `fast path: nothing
//! re-derived` and sent nothing; the room loaded no history for the rest of the run.
//!
//! ```text
//! launch 1   subscribe(id 0, Q, tier) ─▶ server: derive, rows, QuerySettled ─▶ answered
//!            (closed; nothing is withdrawn)
//! launch 2   subscribe(id 0, Q, tier) ─▶ server: "equivalent, settled"       ─▶ ?
//! ```

#![cfg(feature = "test")]

use std::any::Any;
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::{Arc, Mutex, PoisonError};

use jazz_tools::object::ObjectId;
use jazz_tools::query_manager::manager::LocalUpdates;
use jazz_tools::query_manager::policy::PolicyExpr;
use jazz_tools::query_manager::session::Session;
use jazz_tools::query_manager::settle_cost::SettleCounts;
use jazz_tools::query_manager::types::{
    ColumnType, Schema, SchemaBuilder, TablePolicies, TableSchema, Value,
};
use jazz_tools::runtime_core::{
    NoopScheduler, ReadDurabilityOptions, RuntimeCore, SubscriptionHandle, SyncSender,
};
use jazz_tools::schema_manager::{AppId, SchemaManager};
use jazz_tools::storage::SqliteStorage;
use jazz_tools::sync_manager::{
    ClientId, Destination, DurabilityTier, InboxEntry, OutboxEntry, QueryId, QueryPropagation,
    ServerId, Source, SyncManager, SyncPayload,
};

/// The settle counters are the process's, and every test here moves them.
static COUNTERS: Mutex<()> = Mutex::new(());

fn schema() -> Schema {
    SchemaBuilder::new()
        .table(
            TableSchema::builder("messages")
                .column("text", ColumnType::Text)
                .column("sent_at", ColumnType::Timestamp)
                .policies(
                    TablePolicies::new()
                        .with_select(PolicyExpr::True)
                        .with_insert(PolicyExpr::True)
                        .with_update(Some(PolicyExpr::True), PolicyExpr::True),
                ),
        )
        .build()
}

#[derive(Clone, Default)]
struct Outbox(Arc<Mutex<Vec<OutboxEntry>>>);

impl Outbox {
    fn take(&self) -> Vec<OutboxEntry> {
        std::mem::take(&mut self.0.lock().unwrap_or_else(PoisonError::into_inner))
    }
}

impl SyncSender for Outbox {
    fn send_sync_message(&self, message: OutboxEntry) {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(message);
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn core(path: &Path, tier: DurabilityTier) -> (RuntimeCore<SqliteStorage, NoopScheduler>, Outbox) {
    let schema_manager = SchemaManager::new(
        SyncManager::new().with_durability_tier(tier),
        schema(),
        AppId::from_name("relaunched-client-read"),
        "dev",
        "main",
    )
    .expect("schema manager");
    let storage = SqliteStorage::open(path).expect("open sqlite");
    let mut core = RuntimeCore::new(schema_manager, storage, NoopScheduler);
    let outbox = Outbox::default();
    core.set_sync_sender(Box::new(outbox.clone()));
    for _ in 0..8 {
        core.immediate_tick();
        core.batched_tick();
    }
    (core, outbox)
}

/// What a read has handed to whoever made it: how many times, and the rows it shows now.
#[derive(Clone, Default)]
struct Answers(Arc<Mutex<(usize, BTreeSet<ObjectId>)>>);

impl Answers {
    fn count(&self) -> usize {
        self.0.lock().unwrap_or_else(PoisonError::into_inner).0
    }

    fn rows(&self) -> BTreeSet<ObjectId> {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .1
            .clone()
    }
}

/// A read the app has open: a page of messages sent after `after`.
struct Read {
    after: u64,
    handle: SubscriptionHandle,
    answers: Answers,
}

/// An app and the server it syncs with. The app's client id is its own for as long as it
/// is installed; the server keeps what it knows of a client until it reaps it, which a
/// client that comes back in time never lets happen.
struct Pair {
    dir: tempfile::TempDir,
    session: Session,
    app: RuntimeCore<SqliteStorage, NoopScheduler>,
    app_outbox: Outbox,
    client_id: ClientId,
    server: RuntimeCore<SqliteStorage, NoopScheduler>,
    server_outbox: Outbox,
    server_id: ServerId,
    to_app: Vec<OutboxEntry>,
    /// Ids of the subscriptions the app sent, in order.
    asked: Vec<QueryId>,
    /// Ids the server said are settled, in order.
    settled: Vec<QueryId>,
}

impl Pair {
    fn new() -> Self {
        let dir = tempfile::tempdir().expect("tempdir");
        let (app, app_outbox) = core(&dir.path().join("app.sqlite"), DurabilityTier::Local);
        let (server, server_outbox) = core(
            &dir.path().join("server.sqlite"),
            DurabilityTier::GlobalServer,
        );
        let mut pair = Pair {
            dir,
            session: Session::new("alice"),
            app,
            app_outbox,
            client_id: ClientId::new(),
            server,
            server_outbox,
            server_id: ServerId::new(),
            to_app: Vec::new(),
            asked: Vec::new(),
            settled: Vec::new(),
        };
        pair.connect();
        pair
    }

    /// A connection opens: the server meets the client, or meets it again. As the shipped
    /// clients do, this one confirms what it applies, and the server counts a row as
    /// delivered only once it has.
    fn connect(&mut self) {
        self.server
            .ensure_client_with_session(self.client_id, self.session.clone());
        self.server.set_client_acks_deliveries(self.client_id, true);
        self.app.add_server(self.server_id);
        self.app.set_upstream_supports_delivery_acks(true);
        self.exchange();
    }

    /// The app is closed and opened: a new engine over the same store under the same client
    /// id. The old engine withdrew nothing, and what was on its way to it is gone.
    fn relaunch(&mut self) {
        // The old process ends before the new one starts: what it wrote is on disk, and
        // its hold on the store is gone. (Two engines open on one store is not a relaunch,
        // and the second would wait on the first's open transaction.)
        self.app.batched_tick();
        let (closed, _) = core(
            &self.dir.path().join("closed.sqlite"),
            DurabilityTier::Local,
        );
        drop(std::mem::replace(&mut self.app, closed));
        self.app_outbox.take();
        self.server_outbox.take();
        self.to_app.clear();
        let (app, app_outbox) = core(&self.dir.path().join("app.sqlite"), DurabilityTier::Local);
        self.app = app;
        self.app_outbox = app_outbox;
        self.connect();
    }

    /// The connection drops and comes back under the same engine.
    fn reconnect(&mut self) {
        self.app_outbox.take();
        self.server_outbox.take();
        self.to_app.clear();
        self.app.remove_server(self.server_id);
        self.connect();
    }

    fn round(&mut self) -> bool {
        for entry in self.to_app.drain(..) {
            if let SyncPayload::QuerySettled { query_id, .. } = &entry.payload {
                self.settled.push(*query_id);
            }
            self.app.park_sync_message(InboxEntry {
                source: Source::Server(self.server_id),
                payload: entry.payload,
            });
        }
        self.app.batched_tick();
        self.app.immediate_tick();

        let mut moved = false;
        for entry in self.app_outbox.take() {
            moved = true;
            if entry.destination == Destination::Server(self.server_id) {
                if let SyncPayload::QuerySubscription { query_id, .. } = &entry.payload {
                    self.asked.push(*query_id);
                }
                self.server.park_sync_message(InboxEntry {
                    source: Source::Client(self.client_id),
                    payload: entry.payload,
                });
            }
        }
        self.server.batched_tick();
        self.server.immediate_tick();
        for entry in self.server_outbox.take() {
            moved = true;
            if entry.destination == Destination::Client(self.client_id) {
                self.to_app.push(entry);
            }
        }
        moved
    }

    fn exchange(&mut self) {
        for _ in 0..10_000 {
            if !self.round() && self.to_app.is_empty() {
                return;
            }
        }
        panic!("the pair never went quiet");
    }

    fn values(text: &str, at: u64) -> std::collections::HashMap<String, Value> {
        [
            ("text".to_string(), Value::Text(text.into())),
            ("sent_at".to_string(), Value::Timestamp(at)),
        ]
        .into()
    }

    /// The app's user sends a message.
    fn message(&mut self, text: &str, at: u64) -> ObjectId {
        let ((id, _), _) = self
            .app
            .insert("messages", Self::values(text, at), None)
            .expect("write a message");
        self.app.immediate_tick();
        self.exchange();
        id
    }

    /// Somebody else's message reaches the server: the app holds it only if the server
    /// sends it.
    fn message_from_elsewhere(&mut self, text: &str, at: u64) -> ObjectId {
        let ((id, _), _) = self
            .server
            .insert("messages", Self::values(text, at), None)
            .expect("write a message on the server");
        self.server.immediate_tick();
        self.exchange();
        id
    }

    /// The app reads messages sent after `after` and wants the server's word on them, as a
    /// chat reads a page of its history.
    fn read_page(&mut self, after: u64) -> Read {
        let query = self
            .app
            .schema_manager_mut()
            .query_manager_mut()
            .query("messages")
            .filter_gt("sent_at", Value::Timestamp(after))
            .build();
        let answers = Answers::default();
        let told = answers.clone();
        let handle = self
            .app
            .subscribe_with_durability_and_propagation(
                query,
                move |delta| {
                    let mut told = told.0.lock().unwrap_or_else(PoisonError::into_inner);
                    told.0 += 1;
                    for removed in &delta.ordered_delta.removed {
                        told.1.remove(&removed.id);
                    }
                    for added in &delta.ordered_delta.added {
                        told.1.insert(added.id);
                    }
                },
                Some(self.session.clone()),
                ReadDurabilityOptions {
                    tier: Some(DurabilityTier::GlobalServer),
                    local_updates: LocalUpdates::Immediate,
                },
                QueryPropagation::Full,
            )
            .expect("subscribe");
        self.app.immediate_tick();
        self.exchange();
        Read {
            after,
            handle,
            answers,
        }
    }

    fn close(&mut self, read: Read) {
        self.app.unsubscribe(read.handle);
        self.app.immediate_tick();
        self.exchange();
    }
}

/// The room and its first read, answered: what every test below starts from. One message
/// is the app's own and one came from elsewhere.
fn a_room_read_once() -> (Pair, BTreeSet<ObjectId>) {
    let mut pair = Pair::new();
    let mine = pair.message("one", 10);
    let theirs = pair.message_from_elsewhere("two", 20);
    let room = BTreeSet::from([mine, theirs]);
    let first = pair.read_page(0);
    assert_eq!(
        (first.answers.count() > 0, first.answers.rows()),
        (true, room.clone()),
        "the first launch's read is not answered in full, so nothing below measures a relaunch"
    );
    (pair, room)
}

#[test]
fn a_relaunched_app_is_answered_the_read_it_made_before() {
    let _counters = COUNTERS.lock().unwrap_or_else(PoisonError::into_inner);
    let (mut pair, room) = a_room_read_once();

    pair.relaunch();
    pair.asked.clear();
    pair.settled.clear();
    let again = pair.read_page(0);

    assert_eq!(
        pair.asked,
        vec![QueryId(0)],
        "the new engine numbers its queries from the start, so the server is asked under \
         the id it holds the last launch's subscription for; if not, this is another case"
    );
    assert_eq!(
        again.answers.count(),
        1,
        "the app was closed and opened, and the read it makes again never returns"
    );
    assert_eq!(
        again.answers.rows(),
        room,
        "the read returns, and not with the room's messages"
    );
}

#[test]
fn a_relaunched_app_is_answered_a_read_it_did_not_make_before() {
    let _counters = COUNTERS.lock().unwrap_or_else(PoisonError::into_inner);
    let (mut pair, _room) = a_room_read_once();

    pair.relaunch();
    let other = pair.read_page(10);

    assert_eq!(other.answers.count(), 1);
    assert_eq!(other.answers.rows().len(), 1);
}

#[test]
fn a_connection_that_comes_back_is_told_once_more_and_derives_nothing() {
    let _counters = COUNTERS.lock().unwrap_or_else(PoisonError::into_inner);
    let (mut pair, _room) = a_room_read_once();

    pair.settled.clear();
    let before = SettleCounts::snapshot();
    pair.reconnect();
    let cost = SettleCounts::snapshot().since(before);

    assert_eq!(
        pair.settled,
        vec![QueryId(0)],
        "the subscription the engine replays on a new connection is answered once: the \
         server cannot tell this engine from one that holds nothing"
    );
    assert_eq!(
        cost.plan_compiles, 0,
        "the answer is the scope the server holds; no query is built to give it"
    );
}

#[test]
fn a_subscription_sent_twice_on_one_connection_is_answered_once() {
    let _counters = COUNTERS.lock().unwrap_or_else(PoisonError::into_inner);
    let (mut pair, _room) = a_room_read_once();
    let before = pair.settled.len();

    // The same engine, the same connection: the server has told it, and it holds the answer.
    pair.app.remove_server(pair.server_id);
    pair.app.add_server(pair.server_id);
    pair.exchange();

    assert_eq!(
        pair.settled.len(),
        before,
        "a peer that never left is not told again what it was told"
    );
}

struct Prng(u64);

impl Prng {
    fn below(&mut self, bound: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % bound as u64) as usize
    }
}

/// The gates above are the cases thought of. This is the rest: messages from the app and
/// from elsewhere, pages opened and closed, the app closed and opened, the connection
/// dropped and back, in an order nobody chose — against a model of one sentence.
///
/// Every read the app has open has been answered, and shows the messages sent after the
/// time it asked for: all of them, and no others.
#[test]
fn random_reads_relaunches_and_reconnects_are_answered_with_the_rooms_messages() {
    const STEPS: usize = 160;
    const PAGES: [u64; 3] = [0, 40, 90];

    let _counters = COUNTERS.lock().unwrap_or_else(PoisonError::into_inner);
    for seed in [0x5eed_0101_u64, 0x5eed_0102, 0x5eed_0103, 0x5eed_0104] {
        let mut prng = Prng(seed);
        let mut pair = Pair::new();
        let mut clock = 0_u64;
        let mut room: Vec<(ObjectId, u64)> = Vec::new();
        let mut reads: Vec<Read> = Vec::new();
        let mut trail: Vec<String> = Vec::new();

        for step in 0..STEPS {
            let what = match prng.below(100) {
                0..=24 => {
                    clock += 1 + prng.below(4) as u64;
                    room.push((pair.message("mine", clock), clock));
                    "message".to_string()
                }
                25..=44 => {
                    clock += 1 + prng.below(4) as u64;
                    room.push((pair.message_from_elsewhere("theirs", clock), clock));
                    "message from elsewhere".to_string()
                }
                45..=69 => {
                    let after = PAGES[prng.below(PAGES.len())];
                    reads.push(pair.read_page(after));
                    format!("read after {after}")
                }
                70..=79 => {
                    if reads.is_empty() {
                        "close (nothing open)".to_string()
                    } else {
                        let read = reads.remove(prng.below(reads.len()));
                        let after = read.after;
                        pair.close(read);
                        format!("close after {after}")
                    }
                }
                80..=91 => {
                    // The new engine opens nothing by itself: what was open is gone.
                    reads.clear();
                    pair.relaunch();
                    "relaunch".to_string()
                }
                _ => {
                    pair.reconnect();
                    "reconnect".to_string()
                }
            };
            trail.push(what);
            for read in &reads {
                let context = format!(
                    "seed {seed:#x} step {step}: the read after {} at the end of {:?}",
                    read.after,
                    &trail[trail.len().saturating_sub(8)..]
                );
                assert!(read.answers.count() > 0, "{context} was never answered");
                let expected: BTreeSet<ObjectId> = room
                    .iter()
                    .filter(|(_, sent_at)| *sent_at > read.after)
                    .map(|(id, _)| *id)
                    .collect();
                assert_eq!(
                    read.answers.rows(),
                    expected,
                    "{context} does not show the room's messages"
                );
            }
        }
    }
}
