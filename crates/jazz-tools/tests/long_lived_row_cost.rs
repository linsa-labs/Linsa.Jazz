//! What one more write costs a row that has been written thousands of times, end to end:
//! a client engine on SQLite, a server engine on SQLite, and the messages between them.
//!
//! The product path is the presence beat: every ten seconds a device writes one column of
//! its user's row. Measured on the app (simulator against the local sync server,
//! 2026-10-01), a row that had gathered 17 000 of those was sent to the server again in
//! full — 17 189 batches for one beat — and the server spent 18.8 s answering, holding its
//! core while chat history pages waited behind it. Two things had to go wrong, one after
//! the other, and each has a gate here:
//!
//! - the server delivers a device the newest batch of a row the device subscribes to, as a
//!   copy without parents. When that batch is the device's own, the copy replaced the one
//!   it held: the batch lost its parent, the parent became a tip again, and the next beat
//!   merged two states that were never concurrent;
//! - sending a batch sends the ancestors the server is not known to hold, and "known" was
//!   what this connection had sent. The parent of a merge is not in that set, and neither
//!   is anything on the first write after a restart, so the walk went to the row's creation.
//!
//! The first is cured here. The second is not: the first write a connection carries for a
//! row still takes with it every ancestor that connection has not sent, and what a server
//! that lost its store, or was put back to an older copy, needs to be whole again is exactly
//! that — the gates for it are here too, so that a cure for the second cannot take it away.
//!
//! Counts, not times. The counters are process-global, so the tests take one lock.
//!
//! ```text
//! cargo test -p jazz-tools --features test --test long_lived_row_cost -- --nocapture
//! ```

#![cfg(feature = "test")]

use std::any::Any;
use std::collections::HashSet;
use std::path::Path;
use std::sync::{Arc, Mutex, PoisonError};

use jazz_tools::ObjectId;
use jazz_tools::batch_fate::BatchFate;
use jazz_tools::query_manager::manager::LocalUpdates;
use jazz_tools::query_manager::policy::PolicyExpr;
use jazz_tools::query_manager::session::Session;
use jazz_tools::query_manager::settle_cost::SettleCounts;
use jazz_tools::query_manager::types::{
    ColumnType, Schema, SchemaBuilder, TablePolicies, TableSchema, Value,
};
use jazz_tools::row_histories::BatchId;
use jazz_tools::runtime_core::{NoopScheduler, ReadDurabilityOptions, RuntimeCore, SyncSender};
use jazz_tools::schema_manager::{AppId, SchemaManager};
use jazz_tools::storage::{SqliteStorage, Storage};
use jazz_tools::sync_manager::{
    ClientId, Destination, DurabilityTier, InboxEntry, OutboxEntry, QueryPropagation, ServerId,
    Source, SyncManager, SyncPayload,
};

/// Beats the row has taken before anything is counted. Far above every count asserted
/// below, so a cost that follows the row's history cannot hide inside a ceiling.
const DEPTH: usize = 200;

static COUNTERS: Mutex<()> = Mutex::new(());

/// A `users` table with the shape of policy the app's own has: a write is checked against
/// the row it replaces, so the server prepares that row for every write it is sent.
fn schema() -> Schema {
    SchemaBuilder::new()
        .table(
            TableSchema::builder("users")
                .column("name", ColumnType::Text)
                .column("online_at", ColumnType::Timestamp)
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
        AppId::from_name("long-lived-row-cost"),
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

/// One row batch as it crossed the wire.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Sent {
    batch: BatchId,
    parents: usize,
}

fn row_batch(payload: &SyncPayload) -> Option<Sent> {
    match payload {
        SyncPayload::RowBatchCreated { row, .. } | SyncPayload::RowBatchNeeded { row, .. } => {
            Some(Sent {
                batch: row.batch_id(),
                parents: row.parents.len(),
            })
        }
        _ => None,
    }
}

fn copy_store(from: &Path, to: &Path) {
    for suffix in ["", "-wal", "-shm"] {
        let source = format!("{}{suffix}", from.display());
        if Path::new(&source).exists() {
            std::fs::copy(&source, format!("{}{suffix}", to.display()))
                .expect("copy the server store");
        }
    }
}

/// A device and the server it syncs with. The server confirms at the global tier, as the
/// app's does: a batch it has answered for is settled, and nothing is owed for it again.
struct Pair {
    dir: tempfile::TempDir,
    session: Session,
    device: RuntimeCore<SqliteStorage, NoopScheduler>,
    device_outbox: Outbox,
    client_id: ClientId,
    server: RuntimeCore<SqliteStorage, NoopScheduler>,
    server_outbox: Outbox,
    server_id: ServerId,
    to_device: Vec<OutboxEntry>,
    /// Row batches the device sent, in order, since the last `reset`.
    up: Vec<Sent>,
    /// Row batches the server sent the device since the last `reset`.
    down: Vec<Sent>,
    /// Times the server told the device it lacks a batch.
    missing: usize,
    /// The batches it said it lacks, in order.
    asked_for: Vec<BatchId>,
    /// The server's reasons for the batches it refused.
    refusals: Vec<String>,
    /// The store the server runs on now.
    server_store: std::path::PathBuf,
    /// Stores made so far, for names that do not repeat.
    stores: usize,
    device_history_scans: u64,
    server_history_scans: u64,
}

impl Pair {
    fn new() -> Self {
        let dir = tempfile::tempdir().expect("tempdir");
        let (device, device_outbox) =
            core(&dir.path().join("device.sqlite"), DurabilityTier::Local);
        let (server, server_outbox) = core(
            &dir.path().join("server.sqlite"),
            DurabilityTier::GlobalServer,
        );
        let mut pair = Pair {
            dir,
            session: Session::new("alice"),
            device,
            device_outbox,
            client_id: ClientId::new(),
            server,
            server_outbox,
            server_id: ServerId::new(),
            to_device: Vec::new(),
            up: Vec::new(),
            down: Vec::new(),
            missing: 0,
            asked_for: Vec::new(),
            refusals: Vec::new(),
            server_store: std::path::PathBuf::new(),
            stores: 0,
            device_history_scans: 0,
            server_history_scans: 0,
        };
        pair.server_store = pair.dir.path().join("server.sqlite");
        pair.connect();
        pair
    }

    fn connect(&mut self) {
        self.server
            .add_client(self.client_id, Some(self.session.clone()));
        self.device.add_server(self.server_id);
        self.exchange();
    }

    /// The app is closed and opened again: a new engine over the same store, and a new
    /// connection. Nothing the old engine kept in memory survives.
    fn restart_device(&mut self) {
        self.server.remove_client(self.client_id);
        let (device, device_outbox) = core(
            &self.dir.path().join("device.sqlite"),
            DurabilityTier::Local,
        );
        drop(std::mem::replace(&mut self.device, device));
        self.device_outbox = device_outbox;
        self.client_id = ClientId::new();
        self.to_device.clear();
        self.connect();
    }

    /// The server is stopped and started over the store at `path`, and the device connects
    /// to it anew. The device keeps everything the server before told it.
    fn serve_from(&mut self, path: &Path) {
        self.device.remove_server(self.server_id);
        let (server, server_outbox) = core(path, DurabilityTier::GlobalServer);
        drop(std::mem::replace(&mut self.server, server));
        self.server_outbox = server_outbox;
        self.server_store = path.to_path_buf();
        self.server_id = ServerId::new();
        self.to_device.clear();
        self.connect();
    }

    fn new_store(&mut self, what: &str) -> std::path::PathBuf {
        self.stores += 1;
        self.dir
            .path()
            .join(format!("server-{what}-{}.sqlite", self.stores))
    }

    /// The server comes back with nothing: its store was lost and it starts an empty one.
    fn lose_server(&mut self) {
        let empty = self.new_store("empty");
        self.serve_from(&empty);
    }

    /// A copy of the server's store as it is now, to restore the server from later.
    fn back_up_server(&mut self) -> std::path::PathBuf {
        let live = self.server_store.clone();
        let backup = self.new_store("backup");
        // Stop the server while its files are copied, as a backup of a database is taken.
        let elsewhere = self.new_store("elsewhere");
        self.serve_from(&elsewhere);
        copy_store(&live, &backup);
        self.serve_from(&live);
        backup
    }

    /// The server is put back to a backup. The backup itself is left as it was taken.
    fn restore_server(&mut self, backup: &Path) {
        let restored = self.new_store("restored");
        copy_store(backup, &restored);
        self.serve_from(&restored);
    }

    fn server_versions(&self, row: ObjectId) -> usize {
        self.server
            .storage()
            .scan_history_row_batches("users", row)
            .expect("server history")
            .len()
    }

    /// What is on its way to the device reaches it.
    fn deliver_to_device(&mut self) {
        for entry in self.to_device.drain(..) {
            self.down.extend(row_batch(&entry.payload));
            if let SyncPayload::BatchFate {
                fate: BatchFate::Missing { batch_id },
            } = &entry.payload
            {
                self.missing += 1;
                self.asked_for.push(*batch_id);
            }
            if let SyncPayload::BatchFate {
                fate: BatchFate::Rejected { reason, .. },
            } = &entry.payload
            {
                self.refusals.push(reason.clone());
            }
            self.device.park_sync_message(InboxEntry {
                source: Source::Server(self.server_id),
                payload: entry.payload,
            });
        }
    }

    /// The device works through what it has, the server through what the device sent it,
    /// and the server's answers are on their way: not yet with the device.
    fn up(&mut self) -> bool {
        let before = SettleCounts::snapshot();
        self.device.batched_tick();
        self.device.immediate_tick();
        self.device_history_scans += SettleCounts::snapshot().since(before).history_scans;

        let mut moved = false;
        for entry in self.device_outbox.take() {
            moved = true;
            if entry.destination == Destination::Server(self.server_id) {
                self.up.extend(row_batch(&entry.payload));
                self.server.park_sync_message(InboxEntry {
                    source: Source::Client(self.client_id),
                    payload: entry.payload,
                });
            }
        }
        let before = SettleCounts::snapshot();
        self.server.batched_tick();
        self.server.immediate_tick();
        self.server_history_scans += SettleCounts::snapshot().since(before).history_scans;
        for entry in self.server_outbox.take() {
            moved = true;
            if entry.destination == Destination::Client(self.client_id) {
                self.to_device.push(entry);
            }
        }
        moved
    }

    fn round(&mut self) -> bool {
        self.deliver_to_device();
        self.up()
    }

    fn exchange(&mut self) {
        for _ in 0..10_000 {
            if !self.round() && self.to_device.is_empty() {
                return;
            }
        }
        panic!("the pair never went quiet");
    }

    fn reset(&mut self) {
        self.up.clear();
        self.down.clear();
        self.missing = 0;
        self.asked_for.clear();
        self.refusals.clear();
        self.device_history_scans = 0;
        self.server_history_scans = 0;
    }

    /// The device creates its user's row and beats `DEPTH` times, each beat confirmed.
    fn row_with_history(&mut self) -> ObjectId {
        let ((row, _), _) = self
            .device
            .insert(
                "users",
                [
                    ("name".to_string(), Value::Text("me".into())),
                    ("online_at".to_string(), Value::Timestamp(1)),
                ]
                .into(),
                None,
            )
            .expect("create the row");
        self.exchange();
        for beat in 0..DEPTH {
            self.beat(row, 2 + beat as u64);
        }
        assert_eq!(
            self.server
                .storage()
                .scan_history_row_batches("users", row)
                .expect("server history")
                .len(),
            DEPTH + 1,
            "the server does not hold the row's history, so nothing below is measured \
             against a confirmed chain"
        );
        row
    }

    fn beat(&mut self, row: ObjectId, at: u64) -> BatchId {
        let batch = self
            .device
            .update(
                row,
                vec![("online_at".to_string(), Value::Timestamp(at))],
                None,
            )
            .expect("beat");
        self.device.immediate_tick();
        self.exchange();
        batch
    }

    /// The device makes a write and no message moves: what it would send is still with it.
    fn write_only(&mut self, row: ObjectId, at: u64) -> BatchId {
        let batch = self
            .device
            .update(
                row,
                vec![("online_at".to_string(), Value::Timestamp(at))],
                None,
            )
            .expect("write");
        self.device.immediate_tick();
        self.device.batched_tick();
        batch
    }

    /// The connection drops and comes back: the same engines over the same stores, and
    /// whatever was on its way in either direction is lost.
    fn reconnect(&mut self) {
        self.device_outbox.take();
        self.server_outbox.take();
        self.to_device.clear();
        self.server.remove_client(self.client_id);
        self.device.remove_server(self.server_id);
        self.client_id = ClientId::new();
        self.server_id = ServerId::new();
        self.connect();
    }

    /// The device subscribes to the table, as opening a chat subscribes to its members.
    fn subscribe(&mut self) {
        let query = self
            .device
            .schema_manager_mut()
            .query_manager_mut()
            .query("users")
            .build();
        self.device
            .subscribe_with_durability_and_propagation(
                query,
                |_delta| {},
                Some(self.session.clone()),
                ReadDurabilityOptions {
                    tier: Some(DurabilityTier::GlobalServer),
                    local_updates: LocalUpdates::Immediate,
                },
                QueryPropagation::Full,
            )
            .expect("subscribe");
        self.device.immediate_tick();
        self.exchange();
    }

    fn device_tips(&self, row: ObjectId) -> Vec<BatchId> {
        let branch = self
            .device
            .storage()
            .scan_history_row_batches("users", row)
            .expect("device history")[0]
            .branch
            .to_string();
        self.device
            .storage()
            .load_visible_region_frontier("users", &branch, row)
            .expect("read the frontier")
            .expect("the row has a visible entry")
    }

    fn assert_server_holds_the_history(&self, row: ObjectId, versions: usize, newest: BatchId) {
        let held: HashSet<BatchId> = self
            .server
            .storage()
            .scan_history_row_batches("users", row)
            .expect("server history")
            .iter()
            .map(|version| version.batch_id())
            .collect();
        assert_eq!(
            held.len(),
            versions,
            "the server holds {} of the row's {versions} versions after the write",
            held.len()
        );
        assert!(held.contains(&newest), "the write itself did not arrive");
        let visible = self
            .server
            .storage()
            .load_visible_region_row("users", self.branch(row).as_str(), row)
            .expect("read the server's row")
            .expect("the server shows no row");
        assert_eq!(
            visible.batch_id(),
            newest,
            "the write is stored and is not what the server shows"
        );
    }

    fn branch(&self, row: ObjectId) -> String {
        self.device
            .storage()
            .scan_history_row_batches("users", row)
            .expect("device history")[0]
            .branch
            .to_string()
    }

    /// Batches the server said it lacks, besides `write` itself. The server answers
    /// "missing" for every write once — its seal is handled while the row still waits for
    /// its policy check — and the device sends the write a second time. That is as it was
    /// before these gates and is not what they count.
    fn asked_for_besides(&self, write: BatchId) -> Vec<BatchId> {
        self.asked_for
            .iter()
            .copied()
            .filter(|asked| *asked != write)
            .collect()
    }

    /// Distinct batches among what the device sent. A batch the server asks for again is
    /// sent again, and that is not what these gates count.
    fn batches_sent(&self) -> HashSet<BatchId> {
        self.up.iter().map(|sent| sent.batch).collect()
    }
}

#[test]
fn a_batch_delivered_back_to_its_writer_changes_nothing() {
    let _counters = COUNTERS.lock().unwrap_or_else(PoisonError::into_inner);
    let mut pair = Pair::new();
    let row = pair.row_with_history();
    let newest = pair.beat(row, 10_000);

    pair.reset();
    pair.subscribe();
    assert!(
        pair.down
            .iter()
            .any(|sent| sent.batch == newest && sent.parents == 0),
        "the server did not deliver the device its own newest batch without parents, so \
         this run measured nothing: {:?}",
        pair.down
    );
    assert_eq!(
        pair.device_history_scans, 0,
        "the device read the row's history to store a copy of a batch it already held"
    );
    assert_eq!(
        pair.device_tips(row),
        vec![newest],
        "the copy cut the batch off its parent, and the parent is a tip again"
    );

    pair.reset();
    let beat = pair.beat(row, 10_001);
    assert_eq!(
        pair.up.first().copied(),
        Some(Sent {
            batch: beat,
            parents: 1
        }),
        "the beat after the delivery is not a write on top of the newest batch: {:?}",
        pair.up.first()
    );
    assert_eq!(
        pair.batches_sent(),
        HashSet::from([beat]),
        "the beat sent {} batches besides itself",
        pair.batches_sent().len() - 1
    );
    assert_eq!(
        pair.server_history_scans, 0,
        "the server read the row's history to accept one beat"
    );
}

/// A stored confirmation says the server held the batch once. When it no longer does, the
/// row must still reach it, and with one write: the first write a connection carries for
/// a row takes the row's ancestors with it.
///
/// Here the server lost everything. Sent the update alone it would have no row to judge
/// it against and would refuse it — a refusal is recorded, and the history arriving later
/// does not undo it.
#[test]
fn a_server_that_lost_its_store_gets_a_confirmed_history_back_with_one_write() {
    let _counters = COUNTERS.lock().unwrap_or_else(PoisonError::into_inner);
    let mut pair = Pair::new();
    let row = pair.row_with_history();

    pair.lose_server();
    assert!(
        pair.server
            .storage()
            .scan_history_row_batches("users", row)
            .expect("server history")
            .is_empty(),
        "the server still holds the row, so this run measured nothing"
    );

    pair.reset();
    let beat = pair.beat(row, 10_000);
    pair.assert_server_holds_the_history(row, DEPTH + 2, beat);
    assert!(
        pair.refusals.is_empty(),
        "the server refused the write: {:?}",
        pair.refusals
    );

    pair.reset();
    let next = pair.beat(row, 10_001);
    assert_eq!(
        pair.batches_sent(),
        HashSet::from([next]),
        "the history is back, and the next write still sent {} batches besides itself",
        pair.batches_sent().len() - 1
    );
}

/// The server was restored from a backup: it holds the row, up to a batch the device has
/// long since written past. Sent the update alone it could judge it, against a row that
/// is not the one the device wrote over, and could not store it for want of the parent.
#[test]
fn a_server_restored_from_a_backup_gets_the_rest_of_a_confirmed_history_with_one_write() {
    let _counters = COUNTERS.lock().unwrap_or_else(PoisonError::into_inner);
    let mut pair = Pair::new();
    let row = pair.row_with_history();
    let backup = pair.back_up_server();
    for beat in 0..DEPTH {
        pair.beat(row, 5_000 + beat as u64);
    }

    pair.restore_server(&backup);
    assert_eq!(
        pair.server_versions(row),
        DEPTH + 1,
        "the server was not put back to the backup, so this run measured nothing"
    );

    pair.reset();
    let beat = pair.beat(row, 10_000);
    pair.assert_server_holds_the_history(row, 2 * DEPTH + 2, beat);
    assert!(
        pair.refusals.is_empty(),
        "the server refused the write: {:?}",
        pair.refusals
    );

    pair.reset();
    let next = pair.beat(row, 10_001);
    assert_eq!(
        pair.batches_sent(),
        HashSet::from([next]),
        "the history is back, and the next write still sent {} batches besides itself",
        pair.batches_sent().len() - 1
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

/// The gates above are the cases thought of. This is the rest: writes to several rows,
/// app restarts, dropped connections, writes a connection drops under, subscriptions that
/// deliver a row back to its writer, backups, restores and lost stores, in an order nobody
/// chose, against a model of two sentences.
///
/// - After a write to a row, the server holds every version the device holds of it and
///   shows the newest, and it refused nothing.
/// - A write to a row this connection has already carried a write of sends that write and
///   nothing else, and the server asks for nothing — whatever was delivered back to the
///   device in between.
#[test]
fn random_writes_restarts_deliveries_and_server_losses_keep_the_server_whole() {
    const ROWS: usize = 3;
    const STEPS: usize = 220;

    struct Row {
        id: ObjectId,
        versions: usize,
        server_holds_all: bool,
        /// This connection has carried a write of the row to a server that held it whole:
        /// the next write goes alone.
        carried: bool,
    }

    let _counters = COUNTERS.lock().unwrap_or_else(PoisonError::into_inner);
    // Writes the second sentence of the model was held to.
    let mut went_alone = 0_usize;
    for seed in [0x5eed_0001_u64, 0x5eed_0002, 0x5eed_0003, 0x5eed_0004] {
        let mut prng = Prng(seed);
        let mut pair = Pair::new();
        let mut clock = 10_u64;
        let mut rows: Vec<Row> = (0..ROWS)
            .map(|index| {
                let ((id, _), _) = pair
                    .device
                    .insert(
                        "users",
                        [
                            ("name".to_string(), Value::Text(format!("user {index}"))),
                            ("online_at".to_string(), Value::Timestamp(1)),
                        ]
                        .into(),
                        None,
                    )
                    .expect("create the row");
                pair.exchange();
                Row {
                    id,
                    versions: 1,
                    server_holds_all: true,
                    carried: true,
                }
            })
            .collect();
        let mut backup: Option<std::path::PathBuf> = None;
        let mut trail: Vec<String> = Vec::new();

        for step in 0..STEPS {
            let roll = prng.below(100);
            let what = match roll {
                0..=61 => {
                    let index = prng.below(ROWS);
                    clock += 1;
                    pair.reset();
                    let beat = pair.beat(rows[index].id, clock);
                    rows[index].versions += 1;
                    let context = format!(
                        "seed {seed:#x} step {step}: write to row {index} after {:?}",
                        &trail[trail.len().saturating_sub(6)..]
                    );
                    assert!(
                        pair.refusals.is_empty(),
                        "{context}: refused {:?}",
                        pair.refusals
                    );
                    if rows[index].carried {
                        went_alone += 1;
                        assert_eq!(
                            pair.batches_sent(),
                            HashSet::from([beat]),
                            "{context}: the connection had carried the row and now sent {} \
                             batches",
                            pair.batches_sent().len()
                        );
                        assert_eq!(
                            pair.asked_for_besides(beat),
                            Vec::new(),
                            "{context}: the server said it lacked a batch"
                        );
                    }
                    assert_eq!(
                        pair.server_versions(rows[index].id),
                        rows[index].versions,
                        "{context}: the server does not hold the row's history"
                    );
                    let branch = pair.branch(rows[index].id);
                    let shown = pair
                        .server
                        .storage()
                        .load_visible_region_row("users", branch.as_str(), rows[index].id)
                        .expect("read the server's row")
                        .expect("the server shows no row");
                    assert_eq!(
                        shown.batch_id(),
                        beat,
                        "{context}: the server does not show the write"
                    );
                    rows[index].server_holds_all = true;
                    rows[index].carried = true;
                    format!("write {index}")
                }
                62..=67 => {
                    pair.restart_device();
                    rows.iter_mut().for_each(|row| row.carried = false);
                    "restart".to_string()
                }
                68..=69 => {
                    pair.reconnect();
                    rows.iter_mut().for_each(|row| row.carried = false);
                    "reconnect".to_string()
                }
                70..=71 => {
                    // A write the connection drops under, before any message has moved.
                    // Only on a row the server holds: a write offered again by a new
                    // connection goes alone, as it always has.
                    let index = prng.below(ROWS);
                    if !rows[index].server_holds_all {
                        "dropped write (skipped)".to_string()
                    } else {
                        clock += 1;
                        pair.reset();
                        let write = pair.write_only(rows[index].id, clock);
                        if prng.below(2) == 0 {
                            pair.reconnect();
                        } else {
                            pair.device_outbox.take();
                            pair.restart_device();
                        }
                        rows[index].versions += 1;
                        rows.iter_mut().for_each(|row| row.carried = false);
                        let context = format!(
                            "seed {seed:#x} step {step}: dropped write to row {index} after {:?}",
                            &trail[trail.len().saturating_sub(6)..]
                        );
                        assert!(
                            pair.refusals.is_empty(),
                            "{context}: refused {:?}",
                            pair.refusals
                        );
                        pair.assert_server_holds_the_history(
                            rows[index].id,
                            rows[index].versions,
                            write,
                        );
                        format!("dropped write {index}")
                    }
                }
                72..=78 => {
                    pair.subscribe();
                    "subscribe".to_string()
                }
                79..=81 => {
                    // Several writes before any answer, to rows chosen freely — the same
                    // one twice, too.
                    let writes = 1 + prng.below(3);
                    let touched: Vec<usize> = (0..writes).map(|_| prng.below(ROWS)).collect();
                    let carried = touched.iter().all(|index| rows[*index].carried);
                    pair.reset();
                    let mut written = Vec::new();
                    for index in &touched {
                        clock += 1;
                        written.push(pair.write_only(rows[*index].id, clock));
                        rows[*index].versions += 1;
                    }
                    pair.exchange();
                    let context = format!(
                        "seed {seed:#x} step {step}: writes to rows {touched:?} after {:?}",
                        &trail[trail.len().saturating_sub(6)..]
                    );
                    assert!(
                        pair.refusals.is_empty(),
                        "{context}: refused {:?}",
                        pair.refusals
                    );
                    if carried {
                        assert_eq!(
                            pair.batches_sent(),
                            written.iter().copied().collect::<HashSet<_>>(),
                            "{context}: the connection had carried the rows and now sent {} \
                             batches",
                            pair.batches_sent().len()
                        );
                    }
                    for index in &touched {
                        assert_eq!(
                            pair.server_versions(rows[*index].id),
                            rows[*index].versions,
                            "{context}: the server does not hold the history of row {index}"
                        );
                        rows[*index].carried = true;
                    }
                    format!("burst {touched:?}")
                }
                82..=87 => {
                    backup = Some(pair.back_up_server());
                    rows.iter_mut().for_each(|row| row.carried = false);
                    "backup".to_string()
                }
                88..=93 => match backup.clone() {
                    Some(backup) => {
                        pair.restore_server(&backup);
                        rows.iter_mut().for_each(|row| row.carried = false);
                        "restore".to_string()
                    }
                    None => "restore (no backup yet)".to_string(),
                },
                _ => {
                    pair.lose_server();
                    rows.iter_mut().for_each(|row| row.carried = false);
                    "lose".to_string()
                }
            };
            for row in &mut rows {
                let held = pair.server_versions(row.id);
                assert!(
                    held <= row.versions,
                    "seed {seed:#x} step {step} [{what}]: the server holds {held} versions of \
                     a row written {} times",
                    row.versions
                );
                row.server_holds_all = held == row.versions;
            }
            trail.push(what);
        }
    }
    println!("writes held to going alone: {went_alone}");
    assert!(
        went_alone > 200,
        "only {went_alone} writes were held to going alone: the model's second sentence \
         is barely exercised"
    );
}
