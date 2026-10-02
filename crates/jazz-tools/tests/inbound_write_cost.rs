//! What a row costs a client engine to store, on SQLite: a message the server delivers,
//! and a message written locally.
//!
//! The product path: a search in a chat brings some 1 500 hit rows from the server in one
//! tick, and the engine applies them on its own thread — on a phone, the thread the UI
//! runs on. Measured on the app (simulator, 100k-message store, 2026-10-01) that tick
//! blocked the thread for 2.4 s, and two counts here are what made up the part of it
//! that was not the b-tree itself:
//!
//! - **savepoints**: a delivered row is indexed entry by entry, and every entry — some
//!   seventy for a chat message, most of them trigrams — opened a savepoint of its own,
//!   whose first touch of each b-tree page copies the page to the statement journal. An
//!   operation of one statement now opens none;
//! - **checkpoints**: the flush at the end of the tick moved the log into the database
//!   file before it returned — every page the tick touched written a second time, and
//!   two syncs — on the engine's thread. The flush now commits and syncs the log; a
//!   thread of the store's own moves it.
//!
//! The counts are ceilings the fixture fixes; a ratio alone would pass with both sides
//! at zero. One test, because the counters are process-global: nothing else may run
//! beside it.
//!
//! ```text
//! cargo test -p jazz-tools --features test --test inbound_write_cost -- --nocapture
//! ```

#![cfg(feature = "test")]

use std::any::Any;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use jazz_tools::ObjectId;
use jazz_tools::query_manager::index_declarations::IndexDeclarations;
use jazz_tools::query_manager::manager::LocalUpdates;
use jazz_tools::query_manager::session::Session;
use jazz_tools::query_manager::settle_cost::SettleCounts;
use jazz_tools::query_manager::types::{ColumnType, Schema, SchemaBuilder, TableSchema, Value};
use jazz_tools::runtime_core::{NoopScheduler, ReadDurabilityOptions, RuntimeCore, SyncSender};
use jazz_tools::schema_manager::{AppId, SchemaManager};
use jazz_tools::storage::{MemoryStorage, SqliteStorage, Storage};
use jazz_tools::sync_manager::{
    ClientId, Destination, DurabilityTier, InboxEntry, OutboxEntry, QueryPropagation, ServerId,
    Source, SyncManager,
};

const BASE_MS: u64 = 1_750_000_000_000;

const PHRASES: &[&str] = &[
    "Can you remind me what time the concert starts tonight?",
    "The presentation went well, they liked the demo a lot.",
    "Not sure about the jacket, the color looked different online.",
    "Honestly the second half of the book was way better than the first.",
    "I finally finished the project I have been complaining about for months.",
    "We need to decide on the hotel by Friday, the prices keep going up.",
    "Just saw the forecast, it is going to rain the whole weekend.",
    "Remember that bookstore we found last summer? It is closing down.",
];

fn schema() -> Schema {
    SchemaBuilder::new()
        .table(TableSchema::builder("users").column("name", ColumnType::Text))
        .table(
            TableSchema::builder("messages")
                .column("chat", ColumnType::Uuid)
                .fk_column("sender", "users")
                .nullable_fk_column("reply_to", "messages")
                .column("text", ColumnType::Text)
                .column("created_at", ColumnType::Timestamp)
                .column("sent_at", ColumnType::Timestamp)
                .column("sender_kind", ColumnType::Text)
                .column("primary_kind", ColumnType::Text)
                .column("is_deleted", ColumnType::Boolean)
                .column("is_streaming", ColumnType::Boolean),
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

fn core<S: Storage>(storage: S, tier: DurabilityTier) -> (RuntimeCore<S, NoopScheduler>, Outbox) {
    let schema_manager = SchemaManager::new(
        SyncManager::new().with_durability_tier(tier),
        schema(),
        AppId::from_name("inbound-write-cost"),
        "dev",
        "main",
    )
    .expect("schema manager");
    let mut core = RuntimeCore::new(schema_manager, storage, NoopScheduler);
    let outbox = Outbox::default();
    core.set_sync_sender(Box::new(outbox.clone()));
    core.schema_manager_mut()
        .query_manager_mut()
        .propose_index_declarations(
            IndexDeclarations::empty()
                .with_composite("messages", "chat", "created_at")
                .expect("declaration")
                .with_trigram("messages", "chat", "text")
                .expect("declaration"),
        );
    for _ in 0..8 {
        core.immediate_tick();
        core.batched_tick();
    }
    (core, outbox)
}

fn message(chat: ObjectId, sender: ObjectId, index: usize) -> HashMap<String, Value> {
    let text = format!(
        "{} #{index} {}",
        PHRASES[index % PHRASES.len()],
        PHRASES[(index / PHRASES.len()) % PHRASES.len()]
            .split(' ')
            .nth(index % 5)
            .unwrap_or("ok")
    );
    [
        ("chat", Value::Uuid(chat)),
        ("sender", Value::Uuid(sender)),
        ("text", Value::Text(text)),
        (
            "created_at",
            Value::Timestamp(BASE_MS + index as u64 * 60_000),
        ),
        ("sent_at", Value::Timestamp(BASE_MS + index as u64 * 60_000)),
        ("sender_kind", Value::Text("user".into())),
        ("primary_kind", Value::Text("text".into())),
        ("is_deleted", Value::Boolean(false)),
        ("is_streaming", Value::Boolean(false)),
    ]
    .into_iter()
    .map(|(column, value)| (column.to_string(), value))
    .collect()
}

struct Pair {
    client: RuntimeCore<SqliteStorage, NoopScheduler>,
    client_outbox: Outbox,
    server: RuntimeCore<MemoryStorage, NoopScheduler>,
    server_outbox: Outbox,
    client_id: ClientId,
    server_id: ServerId,
    /// Server messages not handed to the client yet.
    to_client: Vec<OutboxEntry>,
    client_ticks: usize,
}

impl Pair {
    fn client_to_server(&mut self) -> bool {
        let mut moved = false;
        for entry in self.client_outbox.take() {
            moved = true;
            if entry.destination == Destination::Server(self.server_id) {
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
                self.to_client.push(entry);
            }
        }
        moved
    }

    /// Hands the client every pending server message and runs its tick.
    fn client_tick(&mut self) {
        for entry in self.to_client.drain(..) {
            self.client.park_sync_message(InboxEntry {
                source: Source::Server(self.server_id),
                payload: entry.payload,
            });
        }
        self.client.batched_tick();
        self.client.immediate_tick();
        self.client_ticks += 1;
    }

    fn exchange(&mut self) {
        for _ in 0..10_000 {
            let moved = self.client_to_server();
            let pending = !self.to_client.is_empty();
            self.client_tick();
            if !moved && !pending {
                return;
            }
        }
        panic!("the pair never went quiet");
    }
}

/// Rows synced before anything is counted, so the indexes the counted rows land in are
/// not empty.
const SEED: usize = 400;
/// Rows delivered in the counted tick.
const ROWS: usize = 200;
/// Rows written locally in the counted run, one commit each. A commit of one message
/// leaves some 170 frames in the log, and commits with no pause between them never let
/// the log restart, so the run is kept short of the log's bound: past it the writer
/// checkpoints by design, which `storage::sqlite`'s own tests cover.
const LOCAL_ROWS: usize = 40;

#[test]
fn what_a_row_costs_the_client_to_store() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (client, client_outbox) = core(
        SqliteStorage::open(dir.path().join("client.sqlite")).expect("open sqlite"),
        DurabilityTier::Local,
    );
    let (server, server_outbox) = core(MemoryStorage::new(), DurabilityTier::EdgeServer);
    let mut pair = Pair {
        client,
        client_outbox,
        server,
        server_outbox,
        client_id: ClientId::new(),
        server_id: ServerId::new(),
        to_client: Vec::new(),
        client_ticks: 0,
    };
    let alice = Session::new("alice");
    pair.server.add_client(pair.client_id, Some(alice.clone()));
    pair.client.add_server(pair.server_id);

    let ((sender, _), _) = pair
        .server
        .insert(
            "users",
            [("name".to_string(), Value::Text("me".into()))].into(),
            None,
        )
        .expect("user");
    let chat = ObjectId::new();
    for index in 0..SEED {
        pair.server
            .insert("messages", message(chat, sender, index), None)
            .expect("seed");
    }

    let delivered = Arc::new(AtomicUsize::new(0));
    let query = pair
        .client
        .schema_manager_mut()
        .query_manager_mut()
        .query("messages")
        .build();
    let sink = Arc::clone(&delivered);
    pair.client
        .subscribe_with_durability_and_propagation(
            query,
            move |delta| {
                sink.fetch_add(delta.ordered_delta.added.len(), Ordering::Relaxed);
            },
            Some(alice),
            ReadDurabilityOptions {
                tier: Some(DurabilityTier::Local),
                local_updates: LocalUpdates::Immediate,
            },
            QueryPropagation::Full,
        )
        .expect("subscribe");
    pair.client.immediate_tick();
    pair.exchange();
    assert_eq!(
        delivered.load(Ordering::Relaxed),
        SEED,
        "the seed did not sync"
    );

    // Delivered: the server's rows, all of them in the client's next tick.
    for index in SEED..SEED + ROWS {
        pair.server
            .insert("messages", message(chat, sender, index), None)
            .expect("insert");
    }
    pair.client_ticks = 0;
    let before = SettleCounts::snapshot();
    pair.exchange();
    let inbound = SettleCounts::snapshot().since(before);
    assert_eq!(
        delivered.load(Ordering::Relaxed),
        SEED + ROWS,
        "the delivered rows did not all arrive"
    );
    println!(
        "delivered: {ROWS} rows over {} ticks: {} savepoints ({:.2}/row), {} checkpoints on \
         the engine's thread, storage write {:.1} us/row",
        pair.client_ticks,
        inbound.storage_savepoints,
        inbound.storage_savepoints as f64 / ROWS as f64,
        inbound.storage_checkpoints,
        inbound.storage_write_micros as f64 / ROWS as f64,
    );

    // Written locally: one row a tick, as a message is sent.
    let before = SettleCounts::snapshot();
    for index in SEED + ROWS..SEED + ROWS + LOCAL_ROWS {
        pair.client
            .insert("messages", message(chat, sender, index), None)
            .expect("local insert");
        pair.client.immediate_tick();
        pair.client.batched_tick();
    }
    let local = SettleCounts::snapshot().since(before);
    println!(
        "written locally: {LOCAL_ROWS} rows: {} savepoints ({:.2}/row), {} checkpoints on the \
         engine's thread",
        local.storage_savepoints,
        local.storage_savepoints as f64 / LOCAL_ROWS as f64,
        local.storage_checkpoints,
    );

    // One savepoint a row either way: the row's own mutation, which is several
    // statements. Before, a delivered row opened 87 and a local one 11.
    assert!(
        inbound.storage_savepoints >= ROWS as u64 && inbound.storage_savepoints <= 2 * ROWS as u64,
        "{} savepoints for {ROWS} delivered rows",
        inbound.storage_savepoints
    );
    assert!(
        local.storage_savepoints >= LOCAL_ROWS as u64
            && local.storage_savepoints <= 2 * LOCAL_ROWS as u64,
        "{} savepoints for {LOCAL_ROWS} local rows",
        local.storage_savepoints
    );
    // Every tick that wrote flushed, and none of the flushes moved the log itself.
    assert_eq!(
        inbound.storage_checkpoints, 0,
        "a tick that stored delivered rows checkpointed on the engine's thread"
    );
    assert_eq!(
        local.storage_checkpoints, 0,
        "a tick that stored a local row checkpointed on the engine's thread"
    );
}
