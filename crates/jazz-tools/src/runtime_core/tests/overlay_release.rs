//! A direct write the top tier has confirmed must leave the source overlay.
//!
//! `QueryManager::pending_local_row_batches` is the source overlay: while it holds an entry,
//! every subscription that is sync-backed, tiered and `Immediate` settles with it, and every
//! IndexScan in its order rescans in full — the clean-scan shortcut in the `execute.rs`
//! IndexScan arm and the incremental path in `index_scan.rs` both require an empty overlay —
//! and loads every overlay row once per scan. The map is table-blind, so one pending row of
//! any table does it to every scan of every such subscription.
//!
//! That is the RN client's subscription: sync-backed, `edge`, `Immediate`. And its direct
//! writes never left the map. A row leaves it on a rejection, on a remote version of the same
//! row from another batch, or on a visibility change carrying `GlobalServer` — which comes only
//! for a transactional batch, whose rows its fate patches. A `DurableDirect` fate only parked the
//! scope exemption, and no version of the row came back: the server forwards a write to every
//! client except its author, and the author's own re-sent copy is a no-op. MEASURED on the app
//! (simulator, 2026-10-03): 3 entries from the first pass after launch, 5 after seven keystrokes
//! in the composer (draft upserts), never 0 across the session.
//!
//! Releasing on the fate is sound because the tier-gated loader takes a row's tier from its
//! authoritative fate (`row_confirmed_tier_with_batch_fate`): once the batch is confirmed at
//! `GlobalServer`, a read at any tier serves the stored version without the overlay. The gates
//! assert on what subscribers are SENT, and two of them read the released row at a tier rather
//! than through ids seeded before the release: a tiered include of a remote row, and a reader
//! recompiled after the release. A sync-backed `Immediate` reader opened after the release does
//! not: until its own `QuerySettled` it reads at no tier, so it never reaches the fate.
//!
//! Two topologies: ours, where the sync server has no upstream and confirms at `GlobalServer`,
//! and an edge confirming at `EdgeServer` in front of a `GlobalServer` upstream, with the link
//! between them cut at will — the window in which only the edge has confirmed a write.
//!
//! Runtime level, because the fate must be the server's own answer to a sealed batch, and
//! SQLite, the phone's backend: `MemoryStorage` overrides the tier-gated visible-row read, so it
//! would not exercise the path the release depends on.

use super::*;
use crate::batch_fate::{BatchFate, BatchMode};
use crate::query_manager::manager::LocalUpdates;
use crate::row_format::decode_row;
use crate::storage::SqliteStorage;
use crate::sync_manager::QueryPropagation;

type SqliteCore = RuntimeCore<SqliteStorage, NoopScheduler>;

fn schema() -> Schema {
    SchemaBuilder::new()
        .table(
            TableSchema::builder("users")
                .column("name", ColumnType::Text)
                .column("score", ColumnType::Integer),
        )
        .table(
            TableSchema::builder("notes")
                .column("title", ColumnType::Text)
                .column("author_id", ColumnType::Uuid),
        )
        // The history window's shape: a page of one owner's rows in `at` order, read over the
        // declared composite index `(author_id, at)` (`messages (chatId, createdAtMs)` in the app).
        .table(
            TableSchema::builder("entries")
                .column("title", ColumnType::Text)
                .column("author_id", ColumnType::Uuid)
                .column("at", ColumnType::Integer),
        )
        .build()
}

fn core_in(dirs: &mut Vec<tempfile::TempDir>, sync_manager: SyncManager) -> SqliteCore {
    let dir = tempfile::TempDir::new().unwrap();
    let schema_manager = SchemaManager::new(
        sync_manager,
        schema(),
        AppId::from_name("overlay-release"),
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

/// What one subscriber has been SENT, rebuilt from the deltas the runtime delivered:
/// row id -> the row's decoded values.
#[derive(Clone, Default)]
struct Delivered(
    Arc<Mutex<HashMap<ObjectId, Vec<Value>>>>,
    /// Every removal the subscriber was sent, per row: a remove followed by a re-add between
    /// two checks leaves the rows as they were, and only this sees it.
    Arc<Mutex<HashMap<ObjectId, usize>>>,
);

impl Delivered {
    fn apply(&self, delta: SubscriptionDelta) {
        let decode =
            |data: &[u8]| decode_row(&delta.descriptor, data).expect("a delivered row decodes");
        let mut rows = self.0.lock().unwrap();
        let mut removals = self.1.lock().unwrap();
        for removed in &delta.ordered_delta.removed {
            rows.remove(&removed.id);
            *removals.entry(removed.id).or_default() += 1;
        }
        for added in &delta.ordered_delta.added {
            rows.insert(added.id, decode(added.row.data.as_ref()));
        }
        for updated in &delta.ordered_delta.updated {
            if let Some(row) = &updated.row {
                rows.insert(updated.id, decode(row.data.as_ref()));
            }
        }
    }

    /// How many times the subscriber was told the row left.
    fn removals(&self, id: ObjectId) -> usize {
        self.1.lock().unwrap().get(&id).copied().unwrap_or(0)
    }

    /// The row's first column (`name` for users, `title` for notes).
    fn name(&self, id: ObjectId) -> Option<String> {
        self.0
            .lock()
            .unwrap()
            .get(&id)
            .map(|values| match values.first() {
                Some(Value::Text(text)) => text.clone(),
                other => panic!("a delivered row starts with a text column, got {other:?}"),
            })
    }

    /// The first column of every row an include array of the row carries.
    fn included_names(&self, id: ObjectId, column: usize) -> Option<Vec<String>> {
        self.0
            .lock()
            .unwrap()
            .get(&id)
            .map(|values| match values.get(column) {
                Some(Value::Array(items)) => items
                    .iter()
                    .map(|item| match item {
                        Value::Row { values, .. } => match values.first() {
                            Some(Value::Text(text)) => text.clone(),
                            other => {
                                panic!("an included row starts with a text column, got {other:?}")
                            }
                        },
                        other => panic!("expected an included row, got {other:?}"),
                    })
                    .collect(),
                other => panic!("expected an include array at column {column}, got {other:?}"),
            })
    }
}

/// Where the author's sync server sits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Topology {
    /// Ours: the sync server has no upstream and declares `GlobalServer` (`server/builder.rs`).
    Direct,
    /// An edge declaring `EdgeServer` between the author and a `GlobalServer` upstream.
    BehindEdge,
}

/// The edge's upstream and the link to it. While the link is down the edge's upstream traffic
/// is held, not lost.
struct Upstream {
    global: SqliteCore,
    edge_as_client: ClientId,
    global_for_edge: ServerId,
    link_up: bool,
    held: Vec<SyncPayload>,
}

/// A second client of the author's sync server, writing the same rows.
struct Peer {
    core: SqliteCore,
    client_id: ClientId,
    server_id: ServerId,
}

struct Nodes {
    author: SqliteCore,
    server: SqliteCore,
    client_id: ClientId,
    server_id: ServerId,
    upstream: Option<Upstream>,
    peer: Option<Peer>,
    _dirs: Vec<tempfile::TempDir>,
}

fn nodes_in(topology: Topology, with_peer: bool) -> Nodes {
    let mut dirs = Vec::new();
    let server_tier = match topology {
        Topology::Direct => DurabilityTier::GlobalServer,
        Topology::BehindEdge => DurabilityTier::EdgeServer,
    };
    let mut author = core_in(&mut dirs, SyncManager::new());
    let mut server = core_in(
        &mut dirs,
        SyncManager::new().with_durability_tier(server_tier),
    );
    let client_id = ClientId::new();
    let server_id = ServerId::new();
    // A client connection needs a session for the server to take its writes.
    server.add_client(client_id, Some(Session::new("alice")));
    author.add_server(server_id);
    let peer = with_peer.then(|| {
        let mut core = core_in(&mut dirs, SyncManager::new());
        let client_id = ClientId::new();
        let server_id = ServerId::new();
        server.add_client(client_id, Some(Session::new("bob")));
        core.add_server(server_id);
        Peer {
            core,
            client_id,
            server_id,
        }
    });
    let upstream = (topology == Topology::BehindEdge).then(|| {
        let mut global = core_in(
            &mut dirs,
            SyncManager::new().with_durability_tier(DurabilityTier::GlobalServer),
        );
        let edge_as_client = ClientId::new();
        let global_for_edge = ServerId::new();
        global.add_client(edge_as_client, None);
        global
            .schema_manager_mut()
            .query_manager_mut()
            .sync_manager_mut()
            .set_client_role(edge_as_client, ClientRole::Peer);
        server.add_server(global_for_edge);
        Upstream {
            global,
            edge_as_client,
            global_for_edge,
            link_up: true,
            held: Vec::new(),
        }
    });
    Nodes {
        author,
        server,
        client_id,
        server_id,
        upstream,
        peer,
        _dirs: dirs,
    }
}

fn nodes() -> Nodes {
    nodes_in(Topology::Direct, false)
}

impl Nodes {
    /// Open a runtime subscription at `tier` with `local_updates`, without exchanging anything;
    /// the app subscribes at `EdgeServer`, `Immediate`.
    fn open(
        &mut self,
        table: &str,
        tier: DurabilityTier,
        local_updates: LocalUpdates,
    ) -> Delivered {
        let delivered = Delivered::default();
        let sink = delivered.clone();
        self.author
            .subscribe_with_durability_and_propagation(
                QueryBuilder::new(table).build(),
                move |delta| sink.apply(delta),
                None,
                ReadDurabilityOptions {
                    tier: Some(tier),
                    local_updates,
                },
                QueryPropagation::Full,
            )
            .expect("subscribe");
        self.author.immediate_tick();
        delivered
    }

    fn subscribe(
        &mut self,
        table: &str,
        tier: DurabilityTier,
        local_updates: LocalUpdates,
    ) -> Delivered {
        let delivered = self.open(table, tier, local_updates);
        self.pump(|_| true);
        delivered
    }

    fn subscribe_like_the_app(&mut self) -> Delivered {
        self.subscribe("users", DurabilityTier::EdgeServer, LocalUpdates::Immediate)
    }

    fn write_with(&mut self, name: &str, context: Option<&WriteContext>) -> (ObjectId, BatchId) {
        let ((row_id, _values), batch_id) = self
            .author
            .insert(
                "users",
                HashMap::from([
                    ("name".to_string(), Value::Text(name.into())),
                    ("score".to_string(), Value::Integer(1)),
                ]),
                context,
            )
            .expect("the author writes");
        self.author.immediate_tick();
        (row_id, batch_id)
    }

    fn write(&mut self, name: &str) -> (ObjectId, BatchId) {
        self.write_with(name, None)
    }

    fn update(&mut self, row_id: ObjectId, name: &str) -> BatchId {
        let batch_id = self
            .author
            .update(
                row_id,
                vec![("name".to_string(), Value::Text(name.into()))],
                None,
            )
            .expect("the author updates");
        self.author.immediate_tick();
        batch_id
    }

    /// A reader on the peer, like the app's.
    fn open_on_peer(&mut self, table: &str) -> Delivered {
        let delivered = Delivered::default();
        let sink = delivered.clone();
        let peer = self.peer.as_mut().expect("the peer");
        peer.core
            .subscribe_with_durability_and_propagation(
                QueryBuilder::new(table).build(),
                move |delta| sink.apply(delta),
                None,
                ReadDurabilityOptions {
                    tier: Some(DurabilityTier::EdgeServer),
                    local_updates: LocalUpdates::Immediate,
                },
                QueryPropagation::Full,
            )
            .expect("the peer subscribes");
        peer.core.immediate_tick();
        delivered
    }

    /// The peer writes over a row of the author's, once the row has reached it.
    fn peer_update(&mut self, row_id: ObjectId, name: &str) -> Option<BatchId> {
        let peer = self.peer.as_mut().expect("the peer");
        let written = peer
            .core
            .update(
                row_id,
                vec![("name".to_string(), Value::Text(name.into()))],
                None,
            )
            .ok();
        peer.core.immediate_tick();
        written
    }

    fn link_up(&self) -> bool {
        self.upstream
            .as_ref()
            .is_some_and(|upstream| upstream.link_up)
    }

    fn set_link(&mut self, up: bool) {
        self.upstream.as_mut().expect("an edge topology").link_up = up;
    }

    /// One round: the clients' outboxes to the server, the server's upstream traffic across the
    /// link while it is up, and the server's answers routed — the author's returned undelivered,
    /// the peer's delivered.
    fn to_server(&mut self) -> Vec<SyncPayload> {
        self.author.batched_tick();
        for entry in self.author.sync_sender().take() {
            if entry.destination == Destination::Server(self.server_id) {
                self.server.park_sync_message(InboxEntry {
                    source: Source::Client(self.client_id),
                    payload: entry.payload,
                });
            }
        }
        if let Some(peer) = &mut self.peer {
            peer.core.batched_tick();
            for entry in peer.core.sync_sender().take() {
                if entry.destination == Destination::Server(peer.server_id) {
                    self.server.park_sync_message(InboxEntry {
                        source: Source::Client(peer.client_id),
                        payload: entry.payload,
                    });
                }
            }
        }
        settle(&mut self.server);
        let mut for_author = Vec::new();
        self.route_server_outbox(&mut for_author);

        let mut crossed = false;
        if let Some(upstream) = &mut self.upstream
            && upstream.link_up
            && !upstream.held.is_empty()
        {
            for payload in upstream.held.drain(..) {
                upstream.global.park_sync_message(InboxEntry {
                    source: Source::Client(upstream.edge_as_client),
                    payload,
                });
            }
            settle(&mut upstream.global);
            for entry in upstream.global.sync_sender().take() {
                if entry.destination == Destination::Client(upstream.edge_as_client) {
                    self.server.park_sync_message(InboxEntry {
                        source: Source::Server(upstream.global_for_edge),
                        payload: entry.payload,
                    });
                }
            }
            settle(&mut self.server);
            crossed = true;
        }
        if crossed {
            self.route_server_outbox(&mut for_author);
        }

        if let Some(peer) = &mut self.peer {
            peer.core.batched_tick();
            peer.core.immediate_tick();
        }
        for_author
    }

    fn route_server_outbox(&mut self, for_author: &mut Vec<SyncPayload>) {
        for entry in self.server.sync_sender().take() {
            match entry.destination {
                Destination::Client(id) if id == self.client_id => for_author.push(entry.payload),
                Destination::Client(id)
                    if self.peer.as_ref().is_some_and(|peer| peer.client_id == id) =>
                {
                    let peer = self.peer.as_mut().expect("the peer");
                    peer.core.park_sync_message(InboxEntry {
                        source: Source::Server(peer.server_id),
                        payload: entry.payload,
                    });
                }
                Destination::Server(id)
                    if self
                        .upstream
                        .as_ref()
                        .is_some_and(|upstream| upstream.global_for_edge == id) =>
                {
                    self.upstream
                        .as_mut()
                        .expect("the upstream")
                        .held
                        .push(entry.payload);
                }
                _ => {}
            }
        }
    }

    fn upstream_backlog(&self) -> bool {
        self.upstream
            .as_ref()
            .is_some_and(|upstream| upstream.link_up && !upstream.held.is_empty())
    }

    fn deliver(&mut self, payloads: Vec<SyncPayload>) {
        for payload in payloads {
            self.author.park_sync_message(InboxEntry {
                source: Source::Server(self.server_id),
                payload,
            });
        }
        self.author.batched_tick();
        self.author.immediate_tick();
    }

    /// Exchange until quiet, delivering only the server payloads `keep` admits. Returns every
    /// fate the author was handed, and the payloads `keep` held back, undelivered, in order.
    fn pump_holding(
        &mut self,
        keep: impl Fn(&SyncPayload) -> bool,
    ) -> (Vec<BatchFate>, Vec<SyncPayload>) {
        let mut fates = Vec::new();
        let mut held = Vec::new();
        let mut quiet_rounds = 0;
        for _ in 0..24 {
            let (delivered, withheld): (Vec<_>, Vec<_>) = self
                .to_server()
                .into_iter()
                .partition(|payload| keep(payload));
            held.extend(withheld);
            if delivered.is_empty() && !self.upstream_backlog() {
                quiet_rounds += 1;
                if quiet_rounds == 2 {
                    break;
                }
                continue;
            }
            quiet_rounds = 0;
            fates.extend(fates_in(&delivered));
            self.deliver(delivered);
        }
        (fates, held)
    }

    fn pump(&mut self, keep: impl Fn(&SyncPayload) -> bool) -> Vec<BatchFate> {
        self.pump_holding(keep).0
    }

    fn tracking(&self, row_id: ObjectId) -> crate::query_manager::manager::LocalRowTracking {
        self.author
            .schema_manager()
            .query_manager()
            .local_row_tracking(row_id)
    }

    fn in_overlay(&self, row_id: ObjectId) -> bool {
        self.tracking(row_id).pending_batch
    }

    fn unflagged_with_unbacked(&self) -> usize {
        self.author
            .schema_manager()
            .query_manager()
            .pending_local_row_id_census()
            .unflagged_with_unbacked
    }
}

fn fates_in(payloads: &[SyncPayload]) -> Vec<BatchFate> {
    payloads
        .iter()
        .filter_map(|payload| match payload {
            SyncPayload::BatchFate { fate } => Some(fate.clone()),
            _ => None,
        })
        .collect()
}

fn confirmed_at(fates: &[BatchFate], batch_id: BatchId, tier: DurabilityTier) -> bool {
    fates
        .iter()
        .any(|fate| fate.batch_id() == batch_id && fate.confirmed_tier() == Some(tier))
}

fn confirmed_at_global(fates: &[BatchFate], batch_id: BatchId) -> bool {
    confirmed_at(fates, batch_id, DurabilityTier::GlobalServer)
}

fn is_scope_answer(payload: &SyncPayload) -> bool {
    matches!(payload, SyncPayload::QuerySettled { .. })
}

/// THE GATE. The write reaches the top tier and the author is told so: the row must leave the
/// overlay with its exemption parked, every subscription holding the id must be flagged —
/// including one on another table, whose ids were seeded from the whole map — and the author
/// must go on being sent its write.
#[cfg(feature = "sqlite")]
#[test]
fn a_direct_write_confirmed_at_the_top_tier_leaves_the_source_overlay() {
    let mut nodes = nodes();
    let users = nodes.subscribe_like_the_app();

    let (row_id, batch_id) = nodes.write("Alice");
    assert!(
        nodes.in_overlay(row_id),
        "fixture precondition: a fresh local write is in the source overlay"
    );
    // Opened while the row is pending: its pending ids are seeded from the whole table-blind
    // map, so it holds an id of a table it does not read.
    let _notes = nodes.open("notes", DurabilityTier::EdgeServer, LocalUpdates::Immediate);

    // The scope answers are held back, so the exemption's own retirement can be seen apart.
    let (fates, held) = nodes.pump_holding(|payload| !is_scope_answer(payload));
    assert!(
        confirmed_at_global(&fates, batch_id),
        "fixture precondition: the server must confirm the write at `GlobalServer`, else this \
         gates nothing — got {fates:?}"
    );

    let tracking = nodes.tracking(row_id);
    assert!(
        !tracking.pending_batch,
        "a direct write confirmed at `GlobalServer` must leave the source overlay: the tier-gated \
         loader reads its tier from the fate, so nothing needs the overlay for it any more, and \
         while it stays every `Immediate` tiered subscription rescans every scan on every settle \
         for the rest of the process (measured on the app: never drained). Got {tracking:?}"
    );
    assert!(
        !tracking.scope_exempt && tracking.awaiting_scope,
        "the exemption must have been parked from the overlay entry before it was released, \
         else nothing retires it any more. Got {tracking:?}"
    );
    assert_eq!(
        nodes.unflagged_with_unbacked(),
        0,
        "every subscription holding the released id must be flagged for the retain that drops \
         it, on every table"
    );
    assert_eq!(
        users.name(row_id).as_deref(),
        Some("Alice"),
        "the author must go on being sent its confirmed write"
    );

    nodes.deliver(held);
    nodes.pump(|_| true);
    let tracking = nodes.tracking(row_id);
    assert!(
        tracking.is_retired(),
        "a scope answer after the parking retires the parked exemption — got {tracking:?}"
    );
    assert_eq!(users.name(row_id).as_deref(), Some("Alice"));

    // From now on the row is read from storage: a confirmed update must be sent and stay.
    let updated = nodes.update(row_id, "Alicia");
    let fates = nodes.pump(|_| true);
    assert!(
        confirmed_at_global(&fates, updated),
        "fixture precondition: the update is confirmed too — got {fates:?}"
    );
    assert!(
        !nodes.in_overlay(row_id),
        "the confirmed update leaves the overlay too"
    );
    assert_eq!(users.name(row_id).as_deref(), Some("Alicia"));
    assert_eq!(
        users.removals(row_id),
        0,
        "the release reloads the row at the reader's tier: the author must never be told it \
         left, not even for one pass"
    );
}

/// The argument the release rests on, run: readers opened AFTER the release are not seeded
/// with the id, so they read the row at their own tier — and the confirmed write must be sent
/// to each of them, which only the fate-aware tier read can do.
#[cfg(feature = "sqlite")]
#[test]
fn a_reader_opened_after_the_release_reads_the_confirmed_write_at_its_tier() {
    let mut nodes = nodes();
    let _users = nodes.subscribe_like_the_app();
    let (row_id, batch_id) = nodes.write("Alice");
    let fates = nodes.pump(|_| true);
    assert!(
        confirmed_at_global(&fates, batch_id),
        "precondition — got {fates:?}"
    );
    assert!(
        !nodes.in_overlay(row_id),
        "precondition: the row was released"
    );

    for (tier, local_updates) in [
        (DurabilityTier::EdgeServer, LocalUpdates::Immediate),
        (DurabilityTier::GlobalServer, LocalUpdates::Immediate),
        (DurabilityTier::EdgeServer, LocalUpdates::Deferred),
        (DurabilityTier::GlobalServer, LocalUpdates::Deferred),
    ] {
        let reader = nodes.subscribe("users", tier, local_updates);
        assert_eq!(
            reader.name(row_id).as_deref(),
            Some("Alice"),
            "a reader at {tier:?} / {local_updates:?} opened after the release must be sent the \
             confirmed write: it reads at its tier, and the stored version's tier comes from \
             the fate"
        );
        assert_eq!(
            reader.removals(row_id),
            0,
            "{tier:?} / {local_updates:?}: told the row left"
        );
    }
}

/// The app reads its own rows through includes of other people's rows — a chat's members, each
/// with its user. After the release, an include instance evaluated for a REMOTE row loads the
/// released row through the subscription's row loader, at the subscription's tier: only the
/// fate says that version is there.
#[cfg(feature = "sqlite")]
#[test]
fn a_released_row_reached_through_an_include_of_a_remote_row_is_read_at_the_tier() {
    let mut nodes = nodes_in(Topology::Direct, true);
    let notes = Delivered::default();
    {
        let sink = notes.clone();
        nodes
            .author
            .subscribe_with_durability_and_propagation(
                QueryBuilder::new("notes")
                    .with_array("author", |sub| {
                        sub.from("users").correlate("id", "notes.author_id")
                    })
                    .build(),
                move |delta| sink.apply(delta),
                None,
                ReadDurabilityOptions {
                    tier: Some(DurabilityTier::EdgeServer),
                    local_updates: LocalUpdates::Immediate,
                },
                QueryPropagation::Full,
            )
            .expect("subscribe to notes with their authors, like the app");
        nodes.author.immediate_tick();
    }
    let users = nodes.subscribe_like_the_app();

    let (row_id, batch_id) = nodes.write("Alice");
    let fates = nodes.pump(|_| true);
    assert!(
        confirmed_at_global(&fates, batch_id),
        "precondition — got {fates:?}"
    );
    assert!(
        !nodes.in_overlay(row_id),
        "precondition: the row was released"
    );

    let note = ObjectId::new();
    {
        let peer = nodes.peer.as_mut().expect("the peer");
        peer.core
            .insert_with_id(
                "notes",
                HashMap::from([
                    ("title".to_string(), Value::Text("hello".into())),
                    ("author_id".to_string(), Value::Uuid(row_id)),
                ]),
                Some(note),
                None,
            )
            .expect("the peer writes a note naming the author");
        peer.core.immediate_tick();
    }
    nodes.pump(|_| true);

    assert_eq!(
        notes.name(note).as_deref(),
        Some("hello"),
        "fixture precondition: the peer's note reached the author"
    );
    assert_eq!(
        notes.included_names(note, 2),
        Some(vec!["Alice".to_string()]),
        "the include instance of a remote note must load the author's released row at the \
         subscription's tier — the stored version is confirmed there only by its fate"
    );
    assert_eq!(
        notes.removals(note),
        0,
        "the notes reader was told the note left"
    );
    assert_eq!(
        users.removals(row_id),
        0,
        "the users reader was told the row left"
    );
}

/// Change the app's declared indexes, as a new permissions head does, and let the runtime's
/// batched ticks apply it. A change of the declarations recompiles every subscription
/// (`adopt_declared_indexes`). Validated against the schema, as published declarations are.
fn change_declared_indexes(core: &mut SqliteCore) {
    let declarations = crate::query_manager::index_declarations::IndexDeclarations::empty()
        .with_composite("entries", "author_id", "at")
        .expect("composite declaration");
    declarations
        .validate_against(&schema())
        .expect("the declaration fits the schema");
    core.schema_manager_mut()
        .query_manager_mut()
        .propose_index_declarations(declarations);
    for _ in 0..64 {
        core.batched_tick();
        core.immediate_tick();
        if !core
            .schema_manager()
            .query_manager()
            .has_declared_index_work()
        {
            return;
        }
    }
    panic!("the declared index never finished its fill");
}

/// THE SECOND RELOAD AT THE TIER. A sync-backed `Immediate` reader reads at no tier until its
/// own scope answer lands (`lacks_authoritative_remote_scope`), and a row it loaded then is not
/// read again by itself — so a reader opened after the release reads the released row at its
/// tier only when something reloads it after that answer. A recompile does, for every row of
/// every subscription: the app publishing new declared indexes (or a schema) while it runs.
/// The released row is confirmed at `edge` only by its fate, and the reader must keep it.
#[cfg(feature = "sqlite")]
#[test]
fn a_released_row_survives_a_recompile_of_a_reader_opened_after_the_release() {
    let mut nodes = nodes();
    let (row_id, batch_id) = nodes.write("Alice");
    let fates = nodes.pump(|_| true);
    assert!(
        confirmed_at_global(&fates, batch_id),
        "precondition — got {fates:?}"
    );
    assert!(
        !nodes.in_overlay(row_id),
        "precondition: the row was released"
    );

    let users = nodes.subscribe_like_the_app();
    assert_eq!(
        users.name(row_id).as_deref(),
        Some("Alice"),
        "fixture precondition: the reader was sent the row before its scope answer"
    );

    change_declared_indexes(&mut nodes.author);
    nodes.pump(|_| true);

    assert_eq!(
        users.removals(row_id),
        0,
        "the recompile reloaded the released row at `edge`, where its fate alone confirms it: \
         the reader must not be told it left"
    );
    assert_eq!(users.name(row_id).as_deref(), Some("Alice"));
}

/// THE GUARD. Until the fate arrives the write is not durable anywhere the author can read at
/// `edge`, and the overlay is what serves it.
#[cfg(feature = "sqlite")]
#[test]
fn a_write_the_top_tier_has_not_confirmed_stays_in_the_overlay() {
    let mut nodes = nodes();
    let users = nodes.subscribe_like_the_app();

    let (row_id, _) = nodes.write("Alice");
    let fates = nodes.pump(|payload| !matches!(payload, SyncPayload::BatchFate { .. }));
    assert!(
        fates.is_empty(),
        "fixture precondition: every fate is withheld"
    );

    assert!(
        nodes.in_overlay(row_id),
        "a write with no confirmation must stay in the source overlay"
    );
    assert_eq!(
        users.name(row_id).as_deref(),
        Some("Alice"),
        "an unconfirmed write must stay visible to its author"
    );
}

/// THE TIER GUARD. Behind an edge, a write is confirmed at `EdgeServer` before the top tier has
/// it. The overlay is shared by readers of every tier, and a version below the top tier is
/// invisible to a top-tier reader without it: the row must stay until the top tier's own
/// confirmation arrives, and a `GlobalServer` reader must be sent the write throughout.
#[cfg(feature = "sqlite")]
#[test]
fn a_write_confirmed_only_below_the_top_tier_stays_in_the_overlay() {
    let mut nodes = nodes_in(Topology::BehindEdge, false);
    let global_reader = nodes.subscribe(
        "users",
        DurabilityTier::GlobalServer,
        LocalUpdates::Immediate,
    );
    let edge_reader = nodes.subscribe_like_the_app();

    nodes.set_link(false);
    let (row_id, batch_id) = nodes.write("Alice");
    let fates = nodes.pump(|_| true);
    assert!(
        confirmed_at(&fates, batch_id, DurabilityTier::EdgeServer)
            && !confirmed_at_global(&fates, batch_id),
        "fixture precondition: with the upstream cut, the edge confirms the write at \
         `EdgeServer` only — got {fates:?}"
    );
    assert!(
        nodes.in_overlay(row_id),
        "a write confirmed only at `EdgeServer` must stay in the source overlay"
    );
    assert_eq!(
        global_reader.name(row_id).as_deref(),
        Some("Alice"),
        "a `GlobalServer` `Immediate` reader must keep being sent the author's write"
    );
    assert_eq!(edge_reader.name(row_id).as_deref(), Some("Alice"));

    nodes.set_link(true);
    let fates = nodes.pump(|_| true);
    assert!(
        confirmed_at_global(&fates, batch_id),
        "fixture precondition: once the link is back, the top tier's confirmation reaches the \
         author — got {fates:?}"
    );
    assert!(
        !nodes.in_overlay(row_id),
        "the top tier's confirmation, relayed by the edge, releases the row"
    );
    assert_eq!(global_reader.name(row_id).as_deref(), Some("Alice"));
    assert_eq!(edge_reader.name(row_id).as_deref(), Some("Alice"));
    assert_eq!(
        (global_reader.removals(row_id), edge_reader.removals(row_id)),
        (0, 0),
        "neither reader may be told the row left, across the edge-only window and the release"
    );
}

/// THE TRANSACTIONAL GUARD. A transactional write leaves the overlay by its own path — the
/// visibility change its fate produces — and must stay sent to its author.
#[cfg(feature = "sqlite")]
#[test]
fn a_transactional_write_leaves_the_overlay_by_its_own_path() {
    let mut nodes = nodes();
    let users = nodes.subscribe_like_the_app();

    let transactional = WriteContext {
        session: None,
        attribution: None,
        updated_at: None,
        batch_mode: Some(BatchMode::Transactional),
        batch_id: None,
        target_branch_name: None,
    };
    let (row_id, batch_id) = nodes.write_with("Alice", Some(&transactional));
    nodes
        .author
        .commit_batch(batch_id)
        .expect("commit the transaction");
    nodes.author.immediate_tick();

    let fates = nodes.pump(|_| true);
    assert!(
        fates.iter().any(|fate| matches!(
            fate,
            BatchFate::AcceptedTransaction {
                batch_id: id,
                confirmed_tier: DurabilityTier::GlobalServer,
            } if *id == batch_id
        )),
        "fixture precondition: the server accepts the transaction at `GlobalServer` — got \
         {fates:?}"
    );
    assert!(
        !nodes.in_overlay(row_id),
        "an accepted transaction leaves the overlay"
    );
    assert_eq!(users.name(row_id).as_deref(), Some("Alice"));
}

/// THE ORDERING GUARD. A fate names a batch, the overlay entry names the row's NEWEST pending
/// batch: confirming an older write of the row must not release a newer one still in flight.
#[cfg(feature = "sqlite")]
#[test]
fn confirming_an_older_write_keeps_a_newer_pending_write_of_the_row() {
    let mut nodes = nodes();
    let users = nodes.subscribe_like_the_app();

    let (row_id, first) = nodes.write("Alice");
    let fates = nodes.pump(|_| true);
    assert!(
        confirmed_at_global(&fates, first),
        "fixture precondition: the first write is confirmed — got {fates:?}"
    );

    // The second write reaches the server and its answers are held back; a third write is
    // made; only then do the second write's answers arrive.
    let second = nodes.update(row_id, "Alicia");
    let held = nodes.to_server();
    assert!(
        held.iter().any(|payload| matches!(
            payload,
            SyncPayload::BatchFate { fate } if fate.batch_id() == second
        )),
        "fixture precondition: the server answers the second write with its fate"
    );
    let third = nodes.update(row_id, "Alix");
    assert_ne!(second, third);
    nodes.deliver(held);

    assert!(
        nodes.in_overlay(row_id),
        "the second write's fate must not release the row while the third write is pending"
    );
    assert_eq!(
        users.name(row_id).as_deref(),
        Some("Alix"),
        "the author must keep being sent its newest, unconfirmed write"
    );
}

/// A RELEASED ROW'S EXEMPTION STILL RETIRES ON AN UPSTREAM VERSION. A version of the row from
/// another writer, arriving through the server, retires the tracking of a row still in the
/// overlay; a released row keeps only its parked exemption, and that must retire the same way
/// rather than wait for a scope answer that may be long in coming.
#[cfg(feature = "sqlite")]
#[test]
fn an_upstream_version_of_a_released_row_retires_its_parked_exemption() {
    let mut nodes = nodes_in(Topology::Direct, true);
    let users = nodes.subscribe_like_the_app();
    {
        let peer = nodes.peer.as_mut().expect("the peer");
        peer.core
            .subscribe(QueryBuilder::new("users").build(), |_| {}, None)
            .expect("the peer reads users");
        peer.core.immediate_tick();
    }
    nodes.pump(|_| true);

    // Every scope answer is held back from here on, so the parked exemption cannot drain by its
    // usual path.
    let (row_id, batch_id) = nodes.write("Alice");
    let (fates, mut held) = nodes.pump_holding(|payload| !is_scope_answer(payload));
    assert!(
        confirmed_at_global(&fates, batch_id),
        "precondition — got {fates:?}"
    );
    let tracking = nodes.tracking(row_id);
    assert!(
        !tracking.pending_batch && tracking.awaiting_scope,
        "fixture precondition: released, exemption parked — got {tracking:?}"
    );
    assert_eq!(
        users.name(row_id).as_deref(),
        Some("Alice"),
        "the parked exemption keeps the released row served until a scope answer accounts for it"
    );

    {
        let peer = nodes.peer.as_mut().expect("the peer");
        peer.core
            .update(
                row_id,
                vec![("name".to_string(), Value::Text("Bob was here".into()))],
                None,
            )
            .expect("the peer updates the author's row");
        peer.core.immediate_tick();
    }
    let (_, more) = nodes.pump_holding(|payload| !is_scope_answer(payload));
    held.extend(more);
    let tracking = nodes.tracking(row_id);
    assert!(
        tracking.is_retired(),
        "an upstream version of a released row must retire its parked exemption, as it retires \
         the tracking of a row still in the overlay — got {tracking:?}"
    );

    nodes.deliver(held);
    nodes.pump(|_| true);
    assert_eq!(
        users.name(row_id).as_deref(),
        Some("Bob was here"),
        "once the server's scope answers land, the author is sent the upstream version"
    );
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// THE DIFFERENTIAL. Random writes, updates and exchanges — answers delivered at once, fates
/// withheld, answers held back across later writes and delivered late, the edge's upstream link
/// cut and restored, readers of every tier and mode opened mid-run, a peer writing over the
/// author's rows — against a model of the intended semantics, in both topologies. For a row the
/// peer has not written: it is in the source overlay exactly while its NEWEST local batch has no
/// `GlobalServer` fate on this node, every `Immediate` reader is sent the newest value the
/// author wrote, and a `Deferred` `GlobalServer` reader the newest value confirmed there. For a
/// row the peer has written, whose winner is the merge rule's: the overlay never holds a batch
/// confirmed at `GlobalServer`, every reader is sent some value written to the row, and once
/// everything has been exchanged the author's readers are sent what the peer's reader is. The
/// peer writes only rows at odd positions, so the strict model holds on the others for the whole
/// run. No row is ever deleted, so no reader — `Immediate`, `Deferred` or the peer's — may ever
/// be told one left; a removal followed by a re-add between two checks is counted, not hidden.
///
/// Answers arrive in the order the server sent them: what was held back always lands before a
/// later round (a link delivers in order), so a scope snapshot is never overwritten by an older one.
///
/// One exception, on base as with the release, and only behind an edge: a row the peer has
/// written may be lost to an `Immediate` reader while no version of the row, the author's or the
/// peer's, is known on this node to be confirmed at the reader's tier — "known" counting the fates
/// already on their way to it (held). A step that delivers them (ops 3 and 5, the final settle)
/// hands them over before the node ticks; ops 6 and 7 open a reader, which ticks, before they
/// deliver, and the other ops tick without delivering. In those windows the node knows less than
/// the allowance assumes, so the allowance is stricter than one that ignored the held fates: it
/// can raise a false red there. It is taken once per step, though, so a fate that arrives in the
/// step's own exchange and closes the gap does not take the allowance back: a removal and re-add
/// inside that exchange, caused by the release it triggers, is allowed behind an edge. A removal
/// that lasts is still caught, by the value check after the exchange. The
/// peer's version, a version of another batch on the same branch, retires the author's tracking of
/// the row (`prepare_row_update_with_origin`) — which ends the row's exemption from the reader's
/// scope filter, so a row the reader's server scope does not name yet is filtered out
/// (`filter_synced_query_scope_tuples`), and later reloads read the row at the reader's tier and
/// find no version there (`load_visible_region_row_for_tier`). Behind an edge it happens while the
/// upstream link is down: the top tier has seen no version of the row. Every removal counted by the
/// end of a step that started with that condition is allowed; any other is not. While the condition
/// holds the reader is only required to be sent some value written to the row once it has one.
///
/// Whether the same mechanism exists in `Direct` is a hypothesis. A fate lost for good (op 5,
/// never re-sent by the server) reaches the same state there — no version of a peer-touched row
/// known confirmed at the reader's tier; a probe outside this test saw it at seed 0x51ce, `rows[7]`,
/// from loop iteration 108 — and no removal followed, so `Direct` would need something more, and
/// it stays strict. A red's message says whether the row was peer-touched (as the step ends) and
/// whether no version of it was known at the reader's tier (as the step started, held fates left
/// out); in `Direct` the second cannot tell a fate lost for good from one still on its way.
#[cfg(feature = "sqlite")]
#[test]
fn random_write_and_confirmation_orders_keep_the_overlay_exactly_the_unconfirmed_writes() {
    for topology in [Topology::Direct, Topology::BehindEdge] {
        let (converged, converged_peer_touched) = [0x0f3a_u64, 0x51ce, 0xbeef, 0x7001]
            .into_iter()
            .map(|seed| run_differential(topology, seed))
            .fold((0, 0), |(all, peer), (a, p)| (all + a, peer + p));
        assert!(
            converged > 0 && converged_peer_touched > 0,
            "{topology:?}: fixture precondition: rows, the peer's among them, were checked for \
             convergence before the last writes ({converged}, {converged_peer_touched})"
        );
    }
}

/// Returns how many rows were checked for convergence before the last writes, and how many of
/// them the peer had written.
fn run_differential(topology: Topology, seed: u64) -> (usize, usize) {
    let mut nodes = nodes_in(topology, true);
    let peer_reader = nodes.open_on_peer("users");
    let mut immediate_readers = vec![nodes.subscribe_like_the_app()];
    // Rows the peer has written over, and every value ever written to each row.
    let mut peer_touched: std::collections::HashSet<ObjectId> = std::collections::HashSet::new();
    // The peer's batches per row, and every fate the author lost for good (step 5).
    let mut peer_batches: HashMap<ObjectId, Vec<BatchId>> = HashMap::new();
    let mut lost: std::collections::HashSet<BatchId> = std::collections::HashSet::new();
    let mut written: HashMap<ObjectId, std::collections::HashSet<String>> = HashMap::new();
    let mut peer_writes = 0;
    // Row-steps at which a strict row was out of the overlay on a `GlobalServer` fate.
    let mut strict_released = 0usize;
    let mut deferred_global_readers: Vec<Delivered> = Vec::new();
    // The tier each `Immediate` reader reads at, and the removals each was allowed per row.
    let mut reader_tiers = vec![DurabilityTier::EdgeServer];
    let mut allowed_removals: HashMap<(usize, ObjectId), usize> = HashMap::new();
    // What the exception absorbed, per reader tier: removals, and "sent nothing" skips.
    let mut absorbed: HashMap<DurabilityTier, (usize, usize)> = HashMap::new();
    let mut rng = Lcg(seed);
    // Model: per row, every version the author wrote, oldest first: (batch, value).
    let mut rows: Vec<(ObjectId, Vec<(BatchId, String)>)> = Vec::new();
    let mut global: std::collections::HashSet<BatchId> = std::collections::HashSet::new();
    // Batches known confirmed at `EdgeServer` or above.
    let mut edge: std::collections::HashSet<BatchId> = std::collections::HashSet::new();
    let mut held: Vec<SyncPayload> = Vec::new();
    let note = |fates: Vec<BatchFate>,
                global: &mut std::collections::HashSet<BatchId>,
                edge: &mut std::collections::HashSet<BatchId>| {
        for fate in fates {
            if fate.confirmed_tier() == Some(DurabilityTier::GlobalServer) {
                global.insert(fate.batch_id());
            }
            if fate.confirmed_tier() >= Some(DurabilityTier::EdgeServer) {
                edge.insert(fate.batch_id());
            }
        }
    };

    let no_version_at = |tier: DurabilityTier,
                         row_id: &ObjectId,
                         versions: &[(BatchId, String)],
                         global: &std::collections::HashSet<BatchId>,
                         edge: &std::collections::HashSet<BatchId>,
                         peer_batches: &HashMap<ObjectId, Vec<BatchId>>| {
        let known = if tier == DurabilityTier::GlobalServer {
            global
        } else {
            edge
        };
        versions.iter().all(|(batch, _)| !known.contains(batch))
            && peer_batches
                .get(row_id)
                .is_none_or(|batches| batches.iter().all(|batch| !known.contains(batch)))
    };
    let gaps = |rows: &[(ObjectId, Vec<(BatchId, String)>)],
                global: &std::collections::HashSet<BatchId>,
                edge: &std::collections::HashSet<BatchId>,
                peer_batches: &HashMap<ObjectId, Vec<BatchId>>| {
        let mut gaps = std::collections::HashSet::new();
        for (row_id, versions) in rows {
            for tier in [DurabilityTier::EdgeServer, DurabilityTier::GlobalServer] {
                if no_version_at(tier, row_id, versions, global, edge, peer_batches) {
                    gaps.insert((*row_id, tier));
                }
            }
        }
        gaps
    };
    let gaps_with_held = |rows: &[(ObjectId, Vec<(BatchId, String)>)],
                          held: &[SyncPayload],
                          global: &std::collections::HashSet<BatchId>,
                          edge: &std::collections::HashSet<BatchId>,
                          peer_batches: &HashMap<ObjectId, Vec<BatchId>>| {
        if topology != Topology::BehindEdge {
            return std::collections::HashSet::new();
        }
        let (mut global, mut edge) = (global.clone(), edge.clone());
        note(fates_in(held), &mut global, &mut edge);
        gaps(rows, &global, &edge, peer_batches)
    };
    for step in 0..120 {
        // The exception's condition as the step starts: a removal inside the step is allowed only
        // if the reader had no version of the row at its tier to fall back to — counting the fates
        // held on the way, which ops 3 and 5 deliver before the node ticks (ops 6 and 7 tick first).
        let gap_at_start = gaps_with_held(&rows, &held, &global, &edge, &peer_batches);
        // For a red's message only: the same state with the held fates left out, in any topology.
        let gap_known = gaps(&rows, &global, &edge, &peer_batches);
        match rng.below(10) {
            0 | 1 if rows.len() < 6 || rng.below(4) == 0 => {
                let name = format!("n{step}");
                let (row_id, batch_id) = nodes.write(&name);
                written.entry(row_id).or_default().insert(name.clone());
                rows.push((row_id, vec![(batch_id, name)]));
            }
            0..=2 if !rows.is_empty() => {
                let index = rng.below(rows.len());
                let name = format!("u{step}");
                let batch_id = nodes.update(rows[index].0, &name);
                written
                    .entry(rows[index].0)
                    .or_default()
                    .insert(name.clone());
                rows[index].1.push((batch_id, name));
            }
            9 if rows.len() > 1 => {
                // Odd positions only: the even ones stay under the strict model.
                let row_id = rows[2 * rng.below(rows.len() / 2) + 1].0;
                let name = format!("p{step}");
                if let Some(batch_id) = nodes.peer_update(row_id, &name) {
                    peer_batches.entry(row_id).or_default().push(batch_id);
                    peer_touched.insert(row_id);
                    written.entry(row_id).or_default().insert(name);
                    peer_writes += 1;
                }
            }
            3 => {
                // Everything in flight, including what was held, arrives now.
                let late = std::mem::take(&mut held);
                note(fates_in(&late), &mut global, &mut edge);
                nodes.deliver(late);
                note(nodes.pump(|_| true), &mut global, &mut edge);
            }
            4 => {
                // The server answers, the answers are held back.
                held.extend(nodes.to_server());
            }
            5 => {
                // What was held arrives first — a link delivers in order — then the server
                // answers, and every fate of this round is lost on the way.
                let late = std::mem::take(&mut held);
                note(fates_in(&late), &mut global, &mut edge);
                nodes.deliver(late);
                let (dropped, answers): (Vec<SyncPayload>, Vec<SyncPayload>) = nodes
                    .to_server()
                    .into_iter()
                    .partition(|payload| matches!(payload, SyncPayload::BatchFate { .. }));
                lost.extend(fates_in(&dropped).iter().map(BatchFate::batch_id));
                nodes.deliver(answers);
            }
            6 if immediate_readers.len() < 4 => {
                let tier = if rng.below(2) == 0 {
                    DurabilityTier::EdgeServer
                } else {
                    DurabilityTier::GlobalServer
                };
                // A top-tier reader settles only through the top tier.
                if topology == Topology::BehindEdge {
                    nodes.set_link(true);
                }
                immediate_readers.push(nodes.open("users", tier, LocalUpdates::Immediate));
                reader_tiers.push(tier);
                let late = std::mem::take(&mut held);
                note(fates_in(&late), &mut global, &mut edge);
                nodes.deliver(late);
                note(nodes.pump(|_| true), &mut global, &mut edge);
            }
            7 if deferred_global_readers.len() < 2 => {
                if topology == Topology::BehindEdge {
                    nodes.set_link(true);
                }
                deferred_global_readers.push(nodes.open(
                    "users",
                    DurabilityTier::GlobalServer,
                    LocalUpdates::Deferred,
                ));
                let late = std::mem::take(&mut held);
                note(fates_in(&late), &mut global, &mut edge);
                nodes.deliver(late);
                note(nodes.pump(|_| true), &mut global, &mut edge);
            }
            8 if topology == Topology::BehindEdge => {
                let up = nodes.link_up();
                nodes.set_link(!up);
            }
            _ => {}
        }

        for (row_id, versions) in &rows {
            let (newest_batch, newest_name) = versions.last().expect("a row has a version");
            // The exception above, for this row and step.
            let may_lose = |index: usize| {
                peer_touched.contains(row_id)
                    && gap_at_start.contains(&(*row_id, reader_tiers[index]))
            };
            for (index, reader) in immediate_readers.iter().enumerate() {
                let allowed = allowed_removals.entry((index, *row_id)).or_default();
                if may_lose(index) && reader.removals(*row_id) > *allowed {
                    absorbed.entry(reader_tiers[index]).or_default().0 +=
                        reader.removals(*row_id) - *allowed;
                    *allowed = reader.removals(*row_id);
                }
                assert_eq!(
                    reader.removals(*row_id),
                    *allowed,
                    "{topology:?} seed {seed:#x} step {step}: `Immediate` reader {index} was told \
                     a row left, and no row is ever deleted (peer-touched: {}, no version known \
                     at the reader's tier: {})",
                    peer_touched.contains(row_id),
                    gap_known.contains(&(*row_id, reader_tiers[index]))
                );
            }
            for (index, reader) in deferred_global_readers.iter().enumerate() {
                assert_eq!(
                    reader.removals(*row_id),
                    0,
                    "{topology:?} seed {seed:#x} step {step}: `Deferred` reader {index} was told \
                     a row left"
                );
            }
            assert_eq!(
                peer_reader.removals(*row_id),
                0,
                "{topology:?} seed {seed:#x} step {step}: the peer's reader was told a row left"
            );
            if peer_touched.contains(row_id) {
                assert!(
                    !(nodes.in_overlay(*row_id) && global.contains(newest_batch)),
                    "{topology:?} seed {seed:#x} step {step}: the overlay holds a batch confirmed \
                     at `GlobalServer`"
                );
                for (index, reader) in immediate_readers.iter().enumerate() {
                    let sent = reader.name(*row_id);
                    if sent.is_none()
                        && topology == Topology::BehindEdge
                        && no_version_at(
                            reader_tiers[index],
                            row_id,
                            versions,
                            &global,
                            &edge,
                            &peer_batches,
                        )
                    {
                        absorbed.entry(reader_tiers[index]).or_default().1 += 1;
                        continue;
                    }
                    assert!(
                        sent.as_ref()
                            .is_some_and(|name| written[row_id].contains(name)),
                        "{topology:?} seed {seed:#x} step {step}: `Immediate` reader {index} must \
                         be sent a value written to the row, got {sent:?}"
                    );
                }
                continue;
            }
            assert_eq!(
                nodes.in_overlay(*row_id),
                !global.contains(newest_batch),
                "{topology:?} seed {seed:#x} step {step}: a row must be in the source overlay \
                 exactly while its newest local batch has no `GlobalServer` fate on this node"
            );
            if !nodes.in_overlay(*row_id) {
                strict_released += 1;
            }
            for (index, reader) in immediate_readers.iter().enumerate() {
                assert_eq!(
                    reader.name(*row_id).as_deref(),
                    Some(newest_name.as_str()),
                    "{topology:?} seed {seed:#x} step {step}: `Immediate` reader {index} must be \
                     sent the newest write"
                );
            }
        }
        assert_eq!(
            nodes.unflagged_with_unbacked(),
            0,
            "{topology:?} seed {seed:#x} step {step}: an unbacked pending id must be flagged"
        );
    }

    // Settle everything; then a `Deferred` `GlobalServer` reader must have been sent each row's
    // newest version confirmed at `GlobalServer` on this node. Removals in the settle are checked
    // against the exception's condition as it starts; the last writes after it may remove nothing.
    let gap_at_start = gaps_with_held(&rows, &held, &global, &edge, &peer_batches);
    let gap_known = gaps(&rows, &global, &edge, &peer_batches);
    if topology == Topology::BehindEdge {
        nodes.set_link(true);
    }
    let late = std::mem::take(&mut held);
    note(fates_in(&late), &mut global, &mut edge);
    nodes.deliver(late);
    note(nodes.pump(|_| true), &mut global, &mut edge);
    // A second round lets the peer's own answers settle after everything above.
    note(nodes.pump(|_| true), &mut global, &mut edge);
    for (row_id, _) in &rows {
        for (index, reader) in immediate_readers.iter().enumerate() {
            let allowed = allowed_removals.entry((index, *row_id)).or_default();
            if peer_touched.contains(row_id)
                && gap_at_start.contains(&(*row_id, reader_tiers[index]))
                && reader.removals(*row_id) > *allowed
            {
                absorbed.entry(reader_tiers[index]).or_default().0 +=
                    reader.removals(*row_id) - *allowed;
                *allowed = reader.removals(*row_id);
            }
            assert_eq!(
                reader.removals(*row_id),
                *allowed,
                "{topology:?} seed {seed:#x}: in the final settle, `Immediate` reader {index} was \
                 told a row left (peer-touched: {}, no version known at the reader's tier: {})",
                peer_touched.contains(row_id),
                gap_known.contains(&(*row_id, reader_tiers[index]))
            );
        }
    }
    // Convergence, for the rows every version of which the author has heard confirmed. A fate
    // lost for good (step 5) leaves that version unconfirmed on this node, though the server
    // and the writer hold it, and a tiered reader rightly stays on the newest version it knows
    // confirmed — the author's own older one, or the peer's. (On base the settle that loads the
    // peer's version still holds the row's pending id, so it loads that version at NO tier and
    // the reader shows it, unconfirmed, from then on.)
    // An older version of the author's whose fate was lost does not matter once a newer one is
    // confirmed: the author's versions are one chain, and the newest confirmed one is read.
    let mut converged = 0;
    let mut converged_peer_touched = 0;
    for (row_id, versions) in &rows {
        let confirmed_here = versions
            .last()
            .is_some_and(|(batch, _)| !lost.contains(batch))
            && peer_batches
                .get(row_id)
                .is_none_or(|batches| batches.iter().all(|batch| !lost.contains(batch)));
        if !confirmed_here {
            continue;
        }
        converged += 1;
        if peer_touched.contains(row_id) {
            converged_peer_touched += 1;
        }
        assert!(
            peer_reader.name(*row_id).is_some(),
            "{topology:?} seed {seed:#x}: once everything is exchanged, the peer's reader must \
             be sent the row"
        );
        for (index, reader) in immediate_readers.iter().enumerate() {
            assert_eq!(
                reader.name(*row_id),
                peer_reader.name(*row_id),
                "{topology:?} seed {seed:#x}: once everything is exchanged, `Immediate` reader \
                 {index} must be sent what the peer's reader is"
            );
        }
    }
    // A version whose fate was lost stays behind until the row is written again — the only way
    // back (the server never re-sends a lost fate). So the author writes every row once more, the
    // peer's included. At the author and the servers that write descends from every version the
    // author holds; at the peer, which got the rows through scope delivery without parents, it
    // wins by last-writer-wins — `(updated_at, batch_id)` on the one clock this process shares,
    // and it is the newest write. Everything is exchanged, and each row must converge on it.
    let removals_before_the_last_writes: Vec<Vec<usize>> = rows
        .iter()
        .map(|(row_id, _)| {
            immediate_readers
                .iter()
                .map(|reader| reader.removals(*row_id))
                .collect()
        })
        .collect();
    for (index, (row_id, versions)) in rows.iter_mut().enumerate() {
        let name = format!("f{index}");
        let batch_id = nodes.update(*row_id, &name);
        written.entry(*row_id).or_default().insert(name.clone());
        versions.push((batch_id, name));
    }
    note(nodes.pump(|_| true), &mut global, &mut edge);
    note(nodes.pump(|_| true), &mut global, &mut edge);
    for (row_index, (row_id, versions)) in rows.iter().enumerate() {
        let (_, newest_name) = versions.last().expect("a row has a version");
        for (index, reader) in immediate_readers.iter().enumerate() {
            assert_eq!(
                reader.name(*row_id).as_deref(),
                Some(newest_name.as_str()),
                "{topology:?} seed {seed:#x}: after a last write and a full exchange, `Immediate` \
                 reader {index} must be sent it"
            );
        }
        assert_eq!(
            peer_reader.name(*row_id).as_deref(),
            Some(newest_name.as_str()),
            "{topology:?} seed {seed:#x}: after a last write and a full exchange, the peer's \
             reader must be sent it"
        );
        assert!(
            !nodes.in_overlay(*row_id),
            "{topology:?} seed {seed:#x}: after a full exchange no row is left in the overlay"
        );
        for (index, reader) in immediate_readers.iter().enumerate() {
            assert_eq!(
                reader.removals(*row_id),
                removals_before_the_last_writes[row_index][index],
                "{topology:?} seed {seed:#x}: after the last writes, `Immediate` reader {index} \
                 was told a row left (peer-touched: {})",
                peer_touched.contains(row_id)
            );
        }
        for (index, reader) in deferred_global_readers.iter().enumerate() {
            assert_eq!(
                reader.removals(*row_id),
                0,
                "{topology:?} seed {seed:#x}: after the last writes, `Deferred` reader {index} was \
                 told a row left"
            );
        }
        assert_eq!(
            peer_reader.removals(*row_id),
            0,
            "{topology:?} seed {seed:#x}: after the last writes, the peer's reader was told a row \
             left"
        );
    }
    for (index, reader) in deferred_global_readers.iter().enumerate() {
        for (row_id, versions) in &rows {
            let expected = versions
                .iter()
                .rev()
                .find(|(batch, _)| global.contains(batch))
                .map(|(_, name)| name.clone());
            assert_eq!(
                reader.name(*row_id),
                expected,
                "{topology:?} seed {seed:#x}: `Deferred` `GlobalServer` reader {index} must be \
                 sent the newest version confirmed at `GlobalServer`"
            );
        }
    }
    assert!(
        !global.is_empty(),
        "{topology:?} seed {seed:#x}: fixture precondition: some writes were confirmed"
    );
    assert!(
        peer_writes > 0,
        "{topology:?} seed {seed:#x}: fixture precondition: the peer wrote over some row"
    );
    assert!(
        strict_released > 0,
        "{topology:?} seed {seed:#x}: fixture precondition: a row the peer never wrote was \
         released on a `GlobalServer` fate"
    );
    eprintln!(
        "overlay differential {topology:?} seed {seed:#x}: exception absorbed (removals, sent-nothing \
         skips) per reader tier: {absorbed:?}"
    );
    (converged, converged_peer_touched)
}
