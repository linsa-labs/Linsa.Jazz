//! What a write costs a row whose history has more than one tip, end to end: devices on
//! SQLite, a server on SQLite, and the messages between them.
//!
//! A row's history is kept for what it is: a record. Nothing a device or a server does to
//! the row — a write, a delivery, a merge — may cost more because the row has been written
//! twenty thousand times before. A row with one tip already holds to that: the row readers
//! see is carried forward from the tip. A row with two tips did not. Measured on the local
//! sync server (2026-10-02): a `users` row with 20 000 versions and a tip no device writes
//! over any more was rebuilt from its whole history on every presence beat — 215 ms a
//! beat, 27–40 % of the server's CPU, with chat pages waiting behind it.
//!
//! Two things a device does leave a tip behind, and both are ordinary. A device put back
//! to an older copy of its store writes over a version the server has moved past. And two
//! devices of one user write at the same moment, each over what it holds. Either way the
//! server shows each device the row it resolves, under the newest tip's id, the devices go
//! on from that id, and the other tip is never named again.
//!
//! Counts, not times. The counters are process-global, so the tests take one lock.
//!
//! ```text
//! cargo test -p jazz-tools --features test --test forked_row_cost -- --nocapture
//! ```

#![cfg(feature = "test")]

use std::any::Any;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};

use jazz_tools::ObjectId;
use jazz_tools::query_manager::manager::LocalUpdates;
use jazz_tools::query_manager::policy::PolicyExpr;
use jazz_tools::query_manager::session::Session;
use jazz_tools::query_manager::settle_cost::SettleCounts;
use jazz_tools::query_manager::types::{
    ColumnType, Schema, SchemaBuilder, TablePolicies, TableSchema, Value,
};
use jazz_tools::row_histories::{BatchId, HistoryScan, VisibleRowEntry};
use jazz_tools::runtime_core::{NoopScheduler, ReadDurabilityOptions, RuntimeCore, SyncSender};
use jazz_tools::schema_manager::{AppId, SchemaManager};
use jazz_tools::storage::{SqliteStorage, Storage};
use jazz_tools::sync_manager::{
    ClientId, Destination, DurabilityTier, InboxEntry, OutboxEntry, QueryPropagation, ServerId,
    Source, SyncManager,
};

/// Beats the row has taken before anything is counted. Far above every count asserted
/// below, so a cost that follows the row's history cannot hide inside a ceiling.
const DEPTH: usize = 300;

/// Versions of a row one write may have its server read: the tips, what they were
/// written over, the version they all descend from — each by its own id. Measured at 9
/// to 15. A read of the row's history is `DEPTH` and more.
const BY_ID: u64 = 32;

static COUNTERS: Mutex<()> = Mutex::new(());

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

type Core = RuntimeCore<SqliteStorage, NoopScheduler>;

fn core(path: &Path, tier: DurabilityTier) -> (Core, Outbox) {
    let schema_manager = SchemaManager::new(
        SyncManager::new().with_durability_tier(tier),
        schema(),
        AppId::from_name("forked-row-cost"),
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

fn copy_store(from: &Path, to: &Path) {
    for suffix in ["", "-wal", "-shm"] {
        let target = format!("{}{suffix}", to.display());
        let _ = std::fs::remove_file(&target);
        let source = format!("{}{suffix}", from.display());
        if Path::new(&source).exists() {
            std::fs::copy(&source, &target).expect("copy a store");
        }
    }
}

/// History read while one side worked: how many times it went to a row's history, and how
/// many versions it was handed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Read {
    scans: u64,
    versions: u64,
}

impl Read {
    fn add(&mut self, cost: SettleCounts) {
        self.scans += cost.history_scans;
        self.versions += cost.history_entries;
    }
}

struct Device {
    core: Core,
    outbox: Outbox,
    client_id: ClientId,
    store: PathBuf,
    inbound: Vec<OutboxEntry>,
    read: Read,
}

/// One user's devices and the server they sync with.
struct World {
    dir: tempfile::TempDir,
    session: Session,
    server: Core,
    server_outbox: Outbox,
    server_id: ServerId,
    devices: Vec<Device>,
    server_read: Read,
    stores: usize,
}

impl World {
    fn new(devices: usize) -> Self {
        let dir = tempfile::tempdir().expect("tempdir");
        let (server, server_outbox) = core(
            &dir.path().join("server.sqlite"),
            DurabilityTier::GlobalServer,
        );
        let mut world = World {
            dir,
            session: Session::new("alice"),
            server,
            server_outbox,
            server_id: ServerId::new(),
            devices: Vec::new(),
            server_read: Read::default(),
            stores: 0,
        };
        for index in 0..devices {
            let store = world.dir.path().join(format!("device-{index}.sqlite"));
            let (core, outbox) = core(&store, DurabilityTier::Local);
            world.devices.push(Device {
                core,
                outbox,
                client_id: ClientId::new(),
                store,
                inbound: Vec::new(),
                read: Read::default(),
            });
            world.connect(index);
        }
        world
    }

    fn connect(&mut self, device: usize) {
        let client_id = self.devices[device].client_id;
        self.server
            .add_client(client_id, Some(self.session.clone()));
        self.devices[device].core.add_server(self.server_id);
        self.exchange();
    }

    fn round(&mut self) -> bool {
        let mut moved = false;
        for device in &mut self.devices {
            for entry in device.inbound.drain(..) {
                device.core.park_sync_message(InboxEntry {
                    source: Source::Server(self.server_id),
                    payload: entry.payload,
                });
            }
            let before = SettleCounts::snapshot();
            device.core.batched_tick();
            device.core.immediate_tick();
            device.read.add(SettleCounts::snapshot().since(before));
            for entry in device.outbox.take() {
                moved = true;
                if entry.destination == Destination::Server(self.server_id) {
                    self.server.park_sync_message(InboxEntry {
                        source: Source::Client(device.client_id),
                        payload: entry.payload,
                    });
                }
            }
        }
        let before = SettleCounts::snapshot();
        self.server.batched_tick();
        self.server.immediate_tick();
        self.server_read.add(SettleCounts::snapshot().since(before));
        for entry in self.server_outbox.take() {
            moved = true;
            if let Some(device) = self
                .devices
                .iter_mut()
                .find(|device| entry.destination == Destination::Client(device.client_id))
            {
                device.inbound.push(entry);
            }
        }
        moved
    }

    fn exchange(&mut self) {
        for _ in 0..10_000 {
            if !self.round() && self.devices.iter().all(|device| device.inbound.is_empty()) {
                return;
            }
        }
        panic!("the devices and the server never went quiet");
    }

    fn reset(&mut self) {
        self.server_read = Read::default();
        for device in &mut self.devices {
            device.read = Read::default();
        }
    }

    /// A device writes its user's presence, and nothing moves yet.
    fn write(&mut self, device: usize, row: ObjectId, at: u64) -> BatchId {
        let device = &mut self.devices[device];
        let before = SettleCounts::snapshot();
        let batch = device
            .core
            .update(
                row,
                vec![("online_at".to_string(), Value::Timestamp(at))],
                None,
            )
            .expect("beat");
        device.core.immediate_tick();
        device.read.add(SettleCounts::snapshot().since(before));
        batch
    }

    /// A device writes its user's presence, and everybody hears of it.
    fn beat(&mut self, device: usize, row: ObjectId, at: u64) -> BatchId {
        let batch = self.write(device, row, at);
        self.exchange();
        batch
    }

    /// The first device creates the user's row and beats `DEPTH` times.
    fn row_with_history(&mut self) -> ObjectId {
        self.row_with_history_of(DEPTH)
    }

    fn row_with_history_of(&mut self, depth: usize) -> ObjectId {
        let ((row, _), _) = self.devices[0]
            .core
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
        for beat in 0..depth {
            self.beat(0, row, 2 + beat as u64);
        }
        assert_eq!(self.server_versions(row), depth + 1);
        row
    }

    /// A device subscribes to the table, as the app does to its own user.
    fn subscribe(&mut self, device: usize) {
        let session = self.session.clone();
        let device = &mut self.devices[device];
        let query = device
            .core
            .schema_manager_mut()
            .query_manager_mut()
            .query("users")
            .build();
        device
            .core
            .subscribe_with_durability_and_propagation(
                query,
                |_delta| {},
                Some(session),
                ReadDurabilityOptions {
                    tier: Some(DurabilityTier::GlobalServer),
                    local_updates: LocalUpdates::Immediate,
                },
                QueryPropagation::Full,
            )
            .expect("subscribe");
        device.core.immediate_tick();
        self.exchange();
    }

    /// The device's engine is stopped, as an app is closed.
    fn stop(&mut self, device: usize) {
        self.stores += 1;
        let elsewhere = self
            .dir
            .path()
            .join(format!("closed-{}.sqlite", self.stores));
        let client_id = self.devices[device].client_id;
        self.server.remove_client(client_id);
        self.devices[device].core.batched_tick();
        let (closed, outbox) = core(&elsewhere, DurabilityTier::Local);
        drop(std::mem::replace(&mut self.devices[device].core, closed));
        self.devices[device].outbox = outbox;
        self.devices[device].inbound.clear();
    }

    /// The device's engine is started over its store, on a new connection.
    fn start(&mut self, device: usize) {
        let (core, outbox) = core(&self.devices[device].store.clone(), DurabilityTier::Local);
        self.devices[device].core = core;
        self.devices[device].outbox = outbox;
        self.devices[device].client_id = ClientId::new();
        self.connect(device);
    }

    /// A copy of a device's store as it is now.
    fn back_up(&mut self, device: usize) -> PathBuf {
        self.stores += 1;
        let backup = self
            .dir
            .path()
            .join(format!("backup-{}.sqlite", self.stores));
        self.stop(device);
        copy_store(&self.devices[device].store.clone(), &backup);
        self.start(device);
        backup
    }

    /// A device is put back to an older copy of its store, as a phone restored from a
    /// backup is: what it wrote since is gone from it, and the server still holds it. It
    /// writes as soon as it is opened, before the server has told it anything.
    fn put_back_and_write(&mut self, device: usize, backup: &Path, row: ObjectId, at: u64) {
        self.stop(device);
        copy_store(backup, &self.devices[device].store.clone());
        let (core, outbox) = core(&self.devices[device].store.clone(), DurabilityTier::Local);
        self.devices[device].core = core;
        self.devices[device].outbox = outbox;
        self.devices[device].client_id = ClientId::new();
        self.write(device, row, at);
        self.connect(device);
    }

    fn branch(&self, row: ObjectId) -> String {
        self.server
            .storage()
            .scan_history_row_batches("users", row)
            .expect("server history")[0]
            .branch
            .to_string()
    }

    fn server_versions(&self, row: ObjectId) -> usize {
        self.server
            .storage()
            .scan_history_row_batches("users", row)
            .expect("server history")
            .len()
    }

    fn server_tips(&self, row: ObjectId) -> Vec<BatchId> {
        let branch = self.branch(row);
        self.server
            .storage()
            .load_visible_region_frontier("users", &branch, row)
            .expect("read the frontier")
            .expect("the row has a visible entry")
    }

    fn device_tips(&self, device: usize, row: ObjectId) -> Vec<BatchId> {
        let branch = self.branch(row);
        self.devices[device]
            .core
            .storage()
            .load_visible_region_frontier("users", &branch, row)
            .expect("read the frontier")
            .expect("the row has a visible entry")
    }

    /// The row the server keeps for its readers is the row its history resolves to: the
    /// entry is carried from write to write, and a carried entry that drifted from the
    /// history would be a row nobody wrote.
    fn assert_server_entry_is_its_history(&self, row: ObjectId, when: &str) {
        let branch = self.branch(row);
        let storage = self.server.storage();
        let history = storage
            .scan_history_region("users", &branch, HistoryScan::Row { row_id: row })
            .expect("server history");
        let descriptor = schema()
            .get(&jazz_tools::query_manager::types::TableName::new("users"))
            .expect("users table")
            .columns
            .clone();
        let resolved = VisibleRowEntry::rebuild_with_descriptor(&descriptor, &history)
            .expect("resolve the history")
            .expect("the history resolves to a row");
        let kept = storage
            .load_visible_region_entry("users", &branch, row)
            .expect("read the entry")
            .expect("the server keeps an entry");
        assert_eq!(
            kept.branch_frontier, resolved.branch_frontier,
            "{when}: the tips the server keeps are not the tips of its history"
        );
        assert_eq!(
            kept.current_row.data, resolved.current_row.data,
            "{when}: the row the server shows is not the row its history resolves to"
        );
        assert_eq!(
            kept.current_row.batch_id(),
            resolved.current_row.batch_id(),
            "{when}: the server shows its row under another version's id"
        );
        assert_eq!(
            (
                &kept.winner_batch_pool,
                &kept.current_winner_ordinals,
                &kept.merge_artifacts
            ),
            (
                &resolved.winner_batch_pool,
                &resolved.current_winner_ordinals,
                &resolved.merge_artifacts
            ),
            "{when}: the server's record of which version won which column drifted"
        );
    }

    fn server_shows(&self, row: ObjectId) -> u64 {
        let branch = self.branch(row);
        let visible = self
            .server
            .storage()
            .load_visible_region_row("users", &branch, row)
            .expect("read the server's row")
            .expect("the server shows no row");
        let values = jazz_tools::row_format::decode_row(
            &schema()
                .get(&jazz_tools::query_manager::types::TableName::new("users"))
                .expect("users table")
                .columns,
            &visible.data,
        )
        .expect("decode the server's row");
        match values[1] {
            Value::Timestamp(at) => at,
            ref other => panic!("online_at is {other:?}"),
        }
    }
}

/// A device restored from a backup writes over a version the server has moved past. From
/// then on the server's row has a tip that no device will write over again.
fn a_row_with_a_tip_nobody_merges() -> (World, ObjectId) {
    let mut world = World::new(1);
    world.subscribe(0);
    let row = world.row_with_history();
    let backup = world.back_up(0);
    for beat in 0..5 {
        world.beat(0, row, 10_000 + beat);
    }
    world.put_back_and_write(0, &backup, row, 20_000);
    world.subscribe(0);
    (world, row)
}

/// The measured defect. The server's row has two tips and the device writes over one of
/// them, every ten seconds, for good. Each of those writes read the row's whole history.
#[test]
fn a_tip_nobody_writes_over_costs_the_server_no_history() {
    let _counters = COUNTERS.lock().unwrap_or_else(PoisonError::into_inner);
    let (mut world, row) = a_row_with_a_tip_nobody_merges();
    assert_eq!(
        world.server_tips(row).len(),
        2,
        "the scenario did not leave the server a second tip"
    );
    assert_eq!(world.device_tips(0, row).len(), 1);

    for beat in 0..6u64 {
        world.reset();
        world.beat(0, row, 30_000 + beat);
        let when = format!("beat {beat} over a row with a tip nobody merges");
        assert_eq!(
            world.server_read.scans, 0,
            "{when}: the server read the row's whole history to take one write"
        );
        assert!(
            world.server_read.versions <= BY_ID,
            "{when}: the server read {} versions of a row to take one write",
            world.server_read.versions
        );
        assert_eq!(
            world.devices[0].read.scans, 0,
            "{when}: the device read its history"
        );
        assert_eq!(
            world.server_shows(row),
            30_000 + beat,
            "{when}: the server does not show the beat"
        );
        assert_eq!(
            world.server_tips(row).len(),
            2,
            "{when}: the other tip is still there, and still nobody's to write over"
        );
        world.assert_server_entry_is_its_history(row, &when);
    }
}

/// Two devices of one user. They write in turn and the row keeps one tip. They write at
/// the same moment once, and from then on the server's row has two.
fn two_devices_that_wrote_at_once() -> (World, ObjectId) {
    let mut world = World::new(2);
    world.subscribe(0);
    let row = world.row_with_history();
    world.subscribe(1);
    for beat in 0..4u64 {
        world.beat((beat % 2) as usize, row, 40_000 + beat);
    }
    assert_eq!(
        world.server_tips(row).len(),
        1,
        "devices writing in turn left the server more than one tip"
    );
    world.write(0, row, 50_000);
    world.write(1, row, 50_001);
    world.exchange();
    assert_eq!(
        world.server_tips(row).len(),
        2,
        "two writes over the same version left the server one tip"
    );
    // Each device writes once more, over everything it holds. A device that came out of
    // the round holding tips of its own names them all here, the server resolves that one
    // write from the row's history, and what is counted below starts after it.
    world.beat(0, row, 50_002);
    world.beat(1, row, 50_003);
    (world, row)
}

#[test]
fn two_devices_that_once_wrote_at_the_same_moment_cost_the_server_no_history() {
    let _counters = COUNTERS.lock().unwrap_or_else(PoisonError::into_inner);
    let (mut world, row) = two_devices_that_wrote_at_once();
    assert_eq!(world.server_shows(row), 50_003);
    assert_eq!(
        world.server_tips(row).len(),
        2,
        "the tip neither device holds is still a tip of the server's row"
    );
    world.assert_server_entry_is_its_history(row, "after both wrote at once");

    for beat in 0..8u64 {
        let device = (beat % 2) as usize;
        world.reset();
        world.beat(device, row, 60_000 + beat);
        let when = format!("beat {beat} by device {device} after both wrote at once");
        assert_eq!(
            world.server_read.scans, 0,
            "{when}: the server read the row's whole history to take one write"
        );
        assert!(
            world.server_read.versions <= BY_ID,
            "{when}: the server read {} versions of a row to take one write",
            world.server_read.versions
        );
        for (index, device) in world.devices.iter().enumerate() {
            assert_eq!(
                device.read.scans, 0,
                "{when}: device {index} read the row's whole history"
            );
        }
        assert_eq!(
            world.server_shows(row),
            60_000 + beat,
            "{when}: the server does not show the beat"
        );
        world.assert_server_entry_is_its_history(row, &when);
    }
}

/// What a write costs its server is the same on a row written three hundred times and on
/// a row written three thousand times.
#[test]
fn a_write_costs_the_same_however_long_the_history() {
    let _counters = COUNTERS.lock().unwrap_or_else(PoisonError::into_inner);
    let mut read = Vec::new();
    for depth in [DEPTH, DEPTH * 10] {
        let mut world = World::new(2);
        world.subscribe(0);
        let row = world.row_with_history_of(depth);
        world.subscribe(1);
        world.write(0, row, 50_000_000);
        world.write(1, row, 50_000_001);
        world.exchange();
        world.beat(0, row, 50_000_002);
        world.beat(1, row, 50_000_003);
        assert_eq!(world.server_tips(row).len(), 2);
        let mut total = Read::default();
        for beat in 0..8u64 {
            world.reset();
            world.beat((beat % 2) as usize, row, 60_000_000 + beat);
            total.scans += world.server_read.scans;
            total.versions += world.server_read.versions;
        }
        world.assert_server_entry_is_its_history(row, &format!("history of {depth}"));
        read.push(total);
    }
    for (depth, read) in [DEPTH, DEPTH * 10].into_iter().zip(&read) {
        assert_eq!(
            read.scans, 0,
            "eight beats over a row of {depth} versions: the server read the row's history"
        );
        assert!(
            read.versions <= 8 * BY_ID,
            "eight beats over a row of {depth} versions had the server read {} of them",
            read.versions
        );
    }
}

/// OPEN DEFECT, not cured by building the entry from the tips: the tips themselves grow.
///
/// Two devices of one user that write at the same moment leave the server two tips, and
/// the device whose write was the older of the two stops naming it — the copy of the newer
/// write the server delivers replaces the device's own tip in its store
/// (`fastpath::snapshot_dominates_frontier` cannot see that the tip has parents: the flat
/// visible codec drops them). Nobody ever writes over that tip again. Each such moment
/// adds a tip, a write to the row reads every tip, and a device that later rebuilds its
/// entry finds all of its own forgotten writes and names them in one write the server
/// reads the row's whole history for.
///
/// Measured 2026-10-02: 14 tips after 24 such moments here; 41 tips on the busiest `users`
/// row of the local stand's store.
#[test]
#[ignore = "open defect: every moment two devices write at once leaves the server one more tip"]
fn devices_that_keep_writing_at_the_same_moment_leave_the_server_two_tips() {
    let _counters = COUNTERS.lock().unwrap_or_else(PoisonError::into_inner);
    let mut world = World::new(2);
    world.subscribe(0);
    let row = world.row_with_history();
    world.subscribe(1);
    let mut at = 40_000u64;
    for round in 0..24u64 {
        at += 10;
        world.write((round % 2) as usize, row, at);
        world.write(((round + 1) % 2) as usize, row, at + 1);
        world.exchange();
        for beat in 0..2u64 {
            at += 10;
            world.reset();
            world.beat(((round + beat) % 2) as usize, row, at);
            assert_eq!(
                world.server_read.scans, 0,
                "round {round} beat {beat}: the server read the row's whole history to take \
                 one write"
            );
        }
        let tips = world.server_tips(row).len();
        assert!(
            tips <= 2,
            "round {round}: the server's row has {tips} tips for two devices"
        );
    }
}
