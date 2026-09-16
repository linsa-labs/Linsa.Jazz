//! What a confirmation leaves behind in subscriptions that do not hold the confirmed row.
//!
//! A confirmation marks its rows through `mark_local_row_updated_in_subscriptions`. That puts
//! each row's id into `pending_local_row_ids` of every subscription on the table, and dirties
//! only the graphs that hold or track the row. The row loader of an `Immediate` subscription
//! reads an id in that set at no durability tier. The delivery paths of a settle drop the ids
//! no local batch backs any more, but only after they load.
//!
//! A tier-wide sweep used to run on every confirmation and made every subscription at or below
//! the confirmed tier settle, which dropped those ids as a side effect. It was removed because
//! the same settle re-authorized every row of every such subscription on each confirmation.
//! The settle loop in `QueryManager::process` now drops unbacked ids in the subscriptions it
//! skips, where a flag says one may have appeared. Without that, on a node that has no
//! durability tier of its own:
//!
//! 1. B subscribes to `notes` and to `notes where tag = 'x'` (`tagged`), both at tier edge and
//!    `Immediate`, as the app's subscriptions are.
//! 2. Client A inserts R with tag `y`. The server confirms it at edge and B learns that.
//!    `tagged` does not hold R, but the confirmation leaves R's id in its pending set.
//! 3. A second `notes where tag = 'x'` subscription (`tagged_late`) opens after that.
//! 4. A moves R to tag `x`. B receives the new version, but its fate is held back.
//! 5. `tagged` loads R at no tier and serves the new version; `tagged_late` does not. Two
//!    identical subscriptions on one node disagree.
//!
//! Next to that scenario: its control, the reveal of another client's row on its confirmation,
//! the bookkeeping after a write of B's own is superseded, and a randomized differential that
//! states same-query agreement and the flag's invariant over random histories, with a variant
//! in which each client also updates the other's rows and both also subscribe to another table.
//! Every test runs on `MemoryStorage` and on `SqliteStorage`: the memory store overrides the
//! visible-row reads, so it alone would never execute the storage path production nodes use.
//!
//! Findings linsa-v21 fails as well are pinned as ignored tests. On a `Local`-tier node an
//! edge-tier subscription serves a received version before the node learns it is edge-confirmed:
//! a fixed scenario, red on linsa-v21 and with the change alike. On a tierless node same-query
//! subscriptions still disagree in some randomized histories, with and without cross writes:
//! the change added no failing seed to either differential in repeated runs.

#![cfg(feature = "test")]

use std::any::Any;
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use jazz_tools::ObjectId;
use jazz_tools::query_manager::manager::LocalUpdates;
use jazz_tools::query_manager::session::{Session, WriteContext};
use jazz_tools::query_manager::types::{ColumnType, Schema, SchemaBuilder, TableSchema, Value};
use jazz_tools::row_format::decode_row;
use jazz_tools::runtime_core::{
    NoopScheduler, ReadDurabilityOptions, RuntimeCore, SubscriptionDelta, SyncSender,
};
use jazz_tools::schema_manager::{AppId, SchemaManager};
use jazz_tools::storage::{MemoryStorage, SqliteStorage, Storage};
use jazz_tools::sync_manager::{
    ClientId, Destination, DurabilityTier, InboxEntry, OutboxEntry, QueryPropagation, ServerId,
    Source, SyncManager, SyncPayload,
};
use jazz_tools::{BatchId, QueryBuilder};

type Core<S> = RuntimeCore<S, NoopScheduler>;

/// A fresh, empty store for one node. Every gate here runs on both backends: `MemoryStorage`
/// overrides the visible-row reads, so a gate that only ran on it would never execute the
/// storage path a production node uses. The SQLite store's directory lives as long as the
/// network that owns it.
trait Store: Storage + Sized + 'static {
    fn fresh() -> (Self, Option<tempfile::TempDir>);
}

impl Store for MemoryStorage {
    fn fresh() -> (Self, Option<tempfile::TempDir>) {
        (MemoryStorage::new(), None)
    }
}

impl Store for SqliteStorage {
    fn fresh() -> (Self, Option<tempfile::TempDir>) {
        let dir = tempfile::tempdir().expect("temp dir");
        let storage = SqliteStorage::open(dir.path().join("node.sqlite")).expect("open sqlite");
        (storage, Some(dir))
    }
}

const MAX_ROUNDS: usize = 200;

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

fn schema() -> Schema {
    SchemaBuilder::new()
        .table(
            TableSchema::builder("notes")
                .column("tag", ColumnType::Text)
                .column("body", ColumnType::Text),
        )
        .table(TableSchema::builder("other").column("body", ColumnType::Text))
        .build()
}

fn core<S: Store>(tier: Option<DurabilityTier>) -> (Core<S>, Outbox, Option<tempfile::TempDir>) {
    let sync_manager = match tier {
        Some(tier) => SyncManager::new().with_durability_tier(tier),
        None => SyncManager::new(),
    };
    let schema_manager = SchemaManager::new(
        sync_manager,
        schema(),
        AppId::from_name("confirmation-pending-row-ids"),
        "dev",
        "main",
    )
    .expect("schema manager");
    let (storage, dir) = S::fresh();
    let mut core = RuntimeCore::new(schema_manager, storage, NoopScheduler);
    let outbox = Outbox::default();
    core.set_sync_sender(Box::new(outbox.clone()));
    core.immediate_tick();
    (core, outbox, dir)
}

/// What a subscription currently shows: row id -> (tag, body).
type View = Arc<Mutex<BTreeMap<ObjectId, (String, String)>>>;

fn text_at(delta: &SubscriptionDelta, values: &[Value], column: &str) -> String {
    let index = delta
        .descriptor
        .columns
        .iter()
        .position(|c| c.name.as_str() == column)
        .unwrap_or_else(|| panic!("column {column} in output descriptor"));
    match &values[index] {
        Value::Text(text) => text.clone(),
        other => format!("{other:?}"),
    }
}

fn subscribe_view<S: Store>(
    core: &mut Core<S>,
    query_tag: Option<&str>,
    session: &Session,
) -> View {
    subscribe_counted_view(core, query_tag, session, Arc::default())
}

/// `deliveries` counts the callbacks, so a caller can tell a view that has not had its first
/// delivery yet from one that is genuinely empty.
fn subscribe_counted_view<S: Store>(
    core: &mut Core<S>,
    query_tag: Option<&str>,
    session: &Session,
    deliveries: Arc<AtomicUsize>,
) -> View {
    let view: View = Arc::default();
    let sink = Arc::clone(&view);
    let mut builder = QueryBuilder::new("notes");
    if let Some(tag) = query_tag {
        builder = builder.filter_eq("tag", Value::Text(tag.to_string()));
    }
    core.subscribe_with_durability_and_propagation(
        builder.build(),
        move |delta| {
            deliveries.fetch_add(1, Ordering::Relaxed);
            let mut view = sink.lock().unwrap_or_else(PoisonError::into_inner);
            for removed in &delta.ordered_delta.removed {
                view.remove(&removed.id);
            }
            for added in &delta.ordered_delta.added {
                let values = decode_row(&delta.descriptor, &added.row.data).expect("decode");
                view.insert(
                    added.id,
                    (
                        text_at(&delta, &values, "tag"),
                        text_at(&delta, &values, "body"),
                    ),
                );
            }
            for updated in &delta.ordered_delta.updated {
                if let Some(row) = &updated.row {
                    let values = decode_row(&delta.descriptor, &row.data).expect("decode");
                    view.insert(
                        updated.id,
                        (
                            text_at(&delta, &values, "tag"),
                            text_at(&delta, &values, "body"),
                        ),
                    );
                }
            }
        },
        Some(session.clone()),
        ReadDurabilityOptions {
            tier: Some(DurabilityTier::EdgeServer),
            local_updates: LocalUpdates::Immediate,
        },
        QueryPropagation::Full,
    )
    .expect("subscribe");
    view
}

fn snapshot(view: &View) -> BTreeMap<ObjectId, (String, String)> {
    view.lock().unwrap_or_else(PoisonError::into_inner).clone()
}

struct Net<S: Store> {
    server: Core<S>,
    server_outbox: Outbox,
    server_id: ServerId,
    a_id: ClientId,
    a: Core<S>,
    a_outbox: Outbox,
    b_id: ClientId,
    b: Core<S>,
    b_outbox: Outbox,
    /// Every batch fate that reached B, in arrival order.
    b_fates: Vec<(BatchId, Option<DurabilityTier>)>,
    _stores: Vec<tempfile::TempDir>,
}

fn record_fate(fates: &mut Vec<(BatchId, Option<DurabilityTier>)>, payload: &SyncPayload) {
    if let SyncPayload::BatchFate { fate } = payload {
        fates.push((fate.batch_id(), fate.confirmed_tier()));
    }
}

impl<S: Store> Net<S> {
    /// Pump until quiet. `hold_for_b` returns true for a server->B payload that must be held
    /// back; held payloads are returned so the caller can release them later.
    fn pump(&mut self, hold_for_b: &dyn Fn(&SyncPayload) -> bool) -> Vec<SyncPayload> {
        let mut held = Vec::new();
        let server_id = self.server_id;
        for _ in 0..MAX_ROUNDS {
            let mut moved = false;
            for (client_id, client, outbox) in [
                (self.a_id, &mut self.a, &self.a_outbox),
                (self.b_id, &mut self.b, &self.b_outbox),
            ] {
                client.batched_tick();
                for entry in outbox.take() {
                    if entry.destination == Destination::Server(server_id) {
                        moved = true;
                        self.server.park_sync_message(InboxEntry {
                            source: Source::Client(client_id),
                            payload: entry.payload,
                        });
                    }
                }
            }
            self.server.batched_tick();
            self.server.immediate_tick();
            for entry in self.server_outbox.take() {
                let Destination::Client(target) = entry.destination else {
                    continue;
                };
                moved = true;
                if target == self.a_id {
                    self.a.park_sync_message(InboxEntry {
                        source: Source::Server(server_id),
                        payload: entry.payload,
                    });
                } else if target == self.b_id {
                    if hold_for_b(&entry.payload) {
                        held.push(entry.payload);
                    } else {
                        record_fate(&mut self.b_fates, &entry.payload);
                        self.b.park_sync_message(InboxEntry {
                            source: Source::Server(server_id),
                            payload: entry.payload,
                        });
                    }
                }
            }
            self.a.batched_tick();
            self.a.immediate_tick();
            self.b.batched_tick();
            self.b.immediate_tick();
            if !moved {
                return held;
            }
        }
        panic!("network never went quiet");
    }

    fn release_to_b(&mut self, payloads: Vec<SyncPayload>) {
        for payload in payloads {
            record_fate(&mut self.b_fates, &payload);
            self.b.park_sync_message(InboxEntry {
                source: Source::Server(self.server_id),
                payload,
            });
        }
        self.b.batched_tick();
        self.b.immediate_tick();
    }
}

fn fate_tier_on<S: Store>(core: &Core<S>, batch_id: BatchId) -> Option<DurabilityTier> {
    core.storage()
        .load_authoritative_batch_fate(batch_id)
        .ok()
        .flatten()
        .and_then(|fate| fate.confirmed_tier())
}

/// What the three subscriptions on B showed while R's second version was held below edge.
struct Observed {
    b_tier_of_second_version: Option<DurabilityTier>,
    all_notes: BTreeMap<ObjectId, (String, String)>,
    tagged: BTreeMap<ObjectId, (String, String)>,
    tagged_late: BTreeMap<ObjectId, (String, String)>,
    row_id: ObjectId,
    after_release_tagged: BTreeMap<ObjectId, (String, String)>,
    after_release_tagged_late: BTreeMap<ObjectId, (String, String)>,
}

/// Server E at `EdgeServer`, client A at `Local`, client B at `b_node_tier`, all connected.
fn network<S: Store>(b_node_tier: Option<DurabilityTier>) -> (Net<S>, Session, Session) {
    let (server, server_outbox, server_store) = core::<S>(Some(DurabilityTier::EdgeServer));
    let (a, a_outbox, a_store) = core::<S>(Some(DurabilityTier::Local));
    let (b, b_outbox, b_store) = core::<S>(b_node_tier);
    let mut net = Net {
        server,
        server_outbox,
        server_id: ServerId::new(),
        a_id: ClientId::new(),
        a,
        a_outbox,
        b_id: ClientId::new(),
        b,
        b_outbox,
        b_fates: Vec::new(),
        _stores: [server_store, a_store, b_store]
            .into_iter()
            .flatten()
            .collect(),
    };
    let alice = Session::new("alice");
    let bob = Session::new("bob");
    net.server.add_client(net.a_id, Some(alice.clone()));
    net.server.add_client(net.b_id, Some(bob.clone()));
    let server_id = net.server_id;
    net.a.add_server(server_id);
    net.b.add_server(server_id);
    (net, alice, bob)
}

fn run_scenario<S: Store>(
    b_node_tier: Option<DurabilityTier>,
    settle_filtered_between: bool,
) -> Observed {
    let (mut net, alice, bob) = network::<S>(b_node_tier);
    let no_hold = |_: &SyncPayload| false;

    let all_notes = subscribe_view(&mut net.b, None, &bob);
    let tagged = subscribe_view(&mut net.b, Some("x"), &bob);
    net.pump(&no_hold);

    // Step 2: R is written with tag y and confirmed at edge.
    let as_alice = WriteContext::from_session(alice.clone());
    let ((row_id, _), first_batch) = net
        .a
        .insert(
            "notes",
            HashMap::from([
                ("tag".to_string(), Value::Text("y".into())),
                ("body".to_string(), Value::Text("v1".into())),
            ]),
            Some(&as_alice),
        )
        .expect("insert R");
    net.pump(&no_hold);

    assert_eq!(
        snapshot(&all_notes).get(&row_id),
        Some(&("y".to_string(), "v1".to_string())),
        "fixture precondition: B holds R through the unfiltered edge-tier subscription"
    );
    assert!(
        fate_tier_on(&net.b, first_batch).is_some_and(|tier| tier >= DurabilityTier::EdgeServer),
        "fixture precondition: B learned that R's first version is confirmed at edge \
         (stored fate tier {:?})",
        fate_tier_on(&net.b, first_batch)
    );
    assert!(
        snapshot(&tagged).is_empty(),
        "fixture precondition: the filtered subscription does not hold R"
    );

    // Step 3: the same query and tier, created after R's confirmation, so it carries no
    // per-subscription history with R.
    let tagged_late = subscribe_view(&mut net.b, Some("x"), &bob);
    net.pump(&no_hold);
    assert!(snapshot(&tagged_late).is_empty());

    if settle_filtered_between {
        // Control arm: give every `notes` subscription on B one ordinary settle between R's
        // first confirmation and its second version — what the tier-wide sweep used to do
        // on that confirmation. A local write of an unrelated row that matches neither filter
        // dirties the table's scans and nothing else visible.
        net.b
            .insert(
                "notes",
                HashMap::from([
                    ("tag".to_string(), Value::Text("z".into())),
                    ("body".to_string(), Value::Text("unrelated".into())),
                ]),
                Some(&WriteContext::from_session(bob.clone())),
            )
            .expect("unrelated local write on B");
        net.pump(&no_hold);
        assert!(snapshot(&tagged).is_empty() && snapshot(&tagged_late).is_empty());
    }

    // Step 4: R moves into the filter; B receives the new version but not its fate.
    let second_batch = net
        .a
        .update(
            row_id,
            vec![
                ("tag".to_string(), Value::Text("x".into())),
                ("body".to_string(), Value::Text("v2".into())),
            ],
            Some(&as_alice),
        )
        .expect("update R");
    let hold_second_fate = move |payload: &SyncPayload| matches!(payload, SyncPayload::BatchFate { fate } if fate.batch_id() == second_batch);
    let held = net.pump(&hold_second_fate);

    assert!(
        !held.is_empty(),
        "fixture precondition: the server sent B the fate of R's second version, and it was held"
    );
    assert!(
        fate_tier_on(&net.server, second_batch)
            .is_some_and(|tier| tier >= DurabilityTier::EdgeServer),
        "fixture precondition: the server confirmed R's second version at edge"
    );
    let b_tier_of_second_version = fate_tier_on(&net.b, second_batch);
    assert!(
        b_tier_of_second_version.is_none_or(|tier| tier < DurabilityTier::EdgeServer),
        "fixture precondition: on B, R's second version is below edge while its fate is held \
         (stored fate tier {b_tier_of_second_version:?})"
    );

    let observed_all = snapshot(&all_notes);
    let observed_tagged = snapshot(&tagged);
    let observed_late = snapshot(&tagged_late);

    net.release_to_b(held);
    net.pump(&no_hold);

    Observed {
        b_tier_of_second_version,
        all_notes: observed_all,
        tagged: observed_tagged,
        tagged_late: observed_late,
        row_id,
        after_release_tagged: snapshot(&tagged),
        after_release_tagged_late: snapshot(&tagged_late),
    }
}

/// The scenario in the module doc. B has no durability tier of its own, so it confirms nothing
/// itself: the fates that mark R on B are the server's.
///
/// `tagged` and `tagged_late` are the same query at the same tier on the same node and may only
/// differ by history: `tagged` was live when R's first version was confirmed while R was outside
/// its filter.
fn check_a_subscription_opened_before_a_confirmation_agrees_with_one_opened_after<S: Store>() {
    let observed = run_scenario::<S>(None, false);
    eprintln!(
        "tierless B: second version tier on B = {:?}\n  all_notes = {:?}\n  tagged = {:?}\n  tagged_late = {:?}",
        observed.b_tier_of_second_version,
        observed.all_notes,
        observed.tagged,
        observed.tagged_late
    );
    assert_eq!(
        observed.tagged, observed.tagged_late,
        "two subscriptions over `notes where tag = 'x'` at tier edge on the same node disagree \
         while R's second version is only {:?} on B. `tagged` was subscribed before R's first \
         confirmation, `tagged_late` after it. The one serving body `v2` is reading R at no \
         tier: R's id was inserted into its `pending_local_row_ids` by the confirmation marking \
         while R was outside its filter, and no settle retired it",
        observed.b_tier_of_second_version,
    );
    assert!(
        observed
            .tagged
            .get(&observed.row_id)
            .is_none_or(|(_, body)| body != "v2"),
        "a subscription at tier edge served R's second version while B holds it below edge \
         ({:?})",
        observed.b_tier_of_second_version,
    );
    assert_eq!(
        observed.after_release_tagged.get(&observed.row_id),
        Some(&("x".to_string(), "v2".to_string())),
        "once the fate arrives the filtered subscription must show R's second version"
    );
    assert_eq!(
        observed.after_release_tagged,
        observed.after_release_tagged_late
    );
}

/// Pre-existing on linsa-v21. On a node with its own `Local` tier, applying a row received
/// from a server records `DurableDirect { Local }` for it (`AcceptedByLocalAuthority`), which
/// is a confirmed fate; both fate paths then call `mark_local_row_updated_in_subscriptions`,
/// which inserts the row into `pending_local_row_ids` of every subscription on the table, and
/// the loader reads those ids at no tier. An edge-tier `Immediate` subscription therefore
/// serves another client's version before it is edge-confirmed on this node.
fn check_an_edge_tier_subscription_on_a_local_tier_node_serves_a_version_below_edge<S: Store>() {
    let observed = run_scenario::<S>(Some(DurabilityTier::Local), false);
    eprintln!(
        "local-tier B: second version tier on B = {:?}\n  all_notes = {:?}\n  tagged = {:?}\n  tagged_late = {:?}",
        observed.b_tier_of_second_version,
        observed.all_notes,
        observed.tagged,
        observed.tagged_late
    );
    assert_eq!(
        observed
            .all_notes
            .get(&observed.row_id)
            .map(|(_, body)| body.as_str()),
        Some("v1"),
        "the unfiltered edge-tier subscription served R's second version while B holds it at \
         {:?}, below edge",
        observed.b_tier_of_second_version,
    );
}

/// Control for the scenario: identical, except that one ordinary settle of B's `notes`
/// subscriptions happens between R's first confirmation and its second version. It passes with
/// or without the skip-branch retain, which pins the disagreement on per-subscription state
/// that only a settle used to retire.
fn check_one_settle_between_confirmation_and_move_restores_agreement<S: Store>() {
    let observed = run_scenario::<S>(None, true);
    eprintln!(
        "control (tierless B, settle between): tagged = {:?}\n  tagged_late = {:?}",
        observed.tagged, observed.tagged_late
    );
    assert_eq!(observed.tagged, observed.tagged_late);
    assert!(
        observed
            .tagged
            .get(&observed.row_id)
            .is_none_or(|(_, body)| body != "v2")
    );
}

#[test]
fn a_subscription_opened_before_a_confirmation_agrees_with_one_opened_after() {
    check_a_subscription_opened_before_a_confirmation_agrees_with_one_opened_after::<MemoryStorage>(
    );
}

#[test]
fn a_subscription_opened_before_a_confirmation_agrees_with_one_opened_after_sqlite() {
    check_a_subscription_opened_before_a_confirmation_agrees_with_one_opened_after::<SqliteStorage>(
    );
}

#[test]
#[ignore = "pre-existing on linsa-v21: a Local-tier node records a received row as locally confirmed, marks it as a pending local row, and edge-tier Immediate subscriptions load it at no tier"]
fn an_edge_tier_subscription_on_a_local_tier_node_serves_a_version_below_edge() {
    check_an_edge_tier_subscription_on_a_local_tier_node_serves_a_version_below_edge::<MemoryStorage>(
    );
}

#[test]
#[ignore = "pre-existing on linsa-v21: a Local-tier node records a received row as locally confirmed, marks it as a pending local row, and edge-tier Immediate subscriptions load it at no tier"]
fn an_edge_tier_subscription_on_a_local_tier_node_serves_a_version_below_edge_sqlite() {
    check_an_edge_tier_subscription_on_a_local_tier_node_serves_a_version_below_edge::<SqliteStorage>(
    );
}

#[test]
fn one_settle_between_confirmation_and_move_restores_agreement() {
    check_one_settle_between_confirmation_and_move_restores_agreement::<MemoryStorage>();
}

#[test]
fn one_settle_between_confirmation_and_move_restores_agreement_sqlite() {
    check_one_settle_between_confirmation_and_move_restores_agreement::<SqliteStorage>();
}

fn note(tag: &str, body: &str) -> HashMap<String, Value> {
    HashMap::from([
        ("tag".to_string(), Value::Text(tag.into())),
        ("body".to_string(), Value::Text(body.into())),
    ])
}

fn note_update(tag: &str, body: &str) -> Vec<(String, Value)> {
    vec![
        ("tag".to_string(), Value::Text(tag.into())),
        ("body".to_string(), Value::Text(body.into())),
    ]
}

fn row(tag: &str, body: &str) -> (String, String) {
    (tag.to_string(), body.to_string())
}

/// A subscription that demands the edge tier is shown another client's row once the fate
/// confirming it at that tier arrives, and not before. B has no tier of its own and confirms
/// nothing itself; the precondition checks that the row is below edge on B while the server's
/// fate is held.
fn check_a_confirmation_reveals_a_row_another_client_wrote<S: Store>() {
    let (mut net, alice, bob) = network::<S>(None);
    let no_hold = |_: &SyncPayload| false;
    let all_notes = subscribe_view(&mut net.b, None, &bob);
    net.pump(&no_hold);

    let ((row_id, _), batch) = net
        .a
        .insert(
            "notes",
            note("y", "v1"),
            Some(&WriteContext::from_session(alice.clone())),
        )
        .expect("A inserts R");
    let hold_its_fate = move |payload: &SyncPayload| matches!(payload, SyncPayload::BatchFate { fate } if fate.batch_id() == batch);
    let held = net.pump(&hold_its_fate);
    assert!(
        !held.is_empty(),
        "fixture precondition: the server sent B the fate of R, and it was held"
    );
    assert!(
        fate_tier_on(&net.b, batch).is_none_or(|tier| tier < DurabilityTier::EdgeServer),
        "fixture precondition: on B, R is below edge while its fate is held"
    );
    assert_eq!(
        snapshot(&all_notes).get(&row_id),
        None,
        "an edge-tier subscription showed another client's row while it is below edge on B"
    );

    net.release_to_b(held);
    net.pump(&no_hold);
    assert_eq!(
        snapshot(&all_notes).get(&row_id),
        Some(&row("y", "v1")),
        "an edge-tier subscription must show another client's row once the fate confirming it \
         at edge arrives"
    );
}

/// B stops tracking its own write when another client's version of the row arrives from
/// another batch (`retire_local_row_tracking`). A subscription dirtied by that version settles
/// and drops the id itself; one that holds the id without reading the table — it copied every
/// tracked id when it opened — is not dirtied, keeps an id no local batch backs, and only a
/// flagged retain drops it. Nothing such a subscription serves can show the stale id, so this
/// pins the bookkeeping instead of a view.
fn check_a_retired_write_leaves_no_unflagged_pending_id<S: Store>() {
    let (mut net, alice, bob) = network::<S>(None);
    let no_hold = |_: &SyncPayload| false;
    let a_notes = subscribe_view(&mut net.a, None, &alice);
    let _b_notes = subscribe_view(&mut net.b, None, &bob);
    net.pump(&no_hold);

    let ((row_id, _), _) = net
        .b
        .insert(
            "notes",
            note("y", "b1"),
            Some(&WriteContext::from_session(bob.clone())),
        )
        .expect("B inserts R");
    net.pump(&no_hold);
    assert_eq!(
        snapshot(&a_notes).get(&row_id),
        Some(&row("y", "b1")),
        "fixture precondition: A holds R, so it can update it"
    );

    net.b
        .subscribe_with_durability_and_propagation(
            QueryBuilder::new("other").build(),
            |_| {},
            Some(bob.clone()),
            ReadDurabilityOptions {
                tier: Some(DurabilityTier::EdgeServer),
                local_updates: LocalUpdates::Immediate,
            },
            QueryPropagation::Full,
        )
        .expect("subscribe to another table");
    net.pump(&no_hold);
    assert_eq!(
        net.b
            .schema_manager()
            .query_manager()
            .pending_local_row_id_holders(row_id),
        2,
        "fixture precondition: B's `notes` subscription and the `other` subscription both hold \
         R's id while B still tracks its write"
    );

    net.a
        .update(
            row_id,
            note_update("y", "a1"),
            Some(&WriteContext::from_session(alice.clone())),
        )
        .expect("A updates R");
    net.pump(&no_hold);

    let census = net
        .b
        .schema_manager()
        .query_manager()
        .pending_local_row_id_census();
    assert_eq!(
        census.unflagged_with_unbacked, 0,
        "after A's version of R retired B's own write, a subscription on B holds R's id, no \
         local batch backs it, and nothing flags the subscription for the retain that drops it"
    );
    assert_eq!(
        net.b
            .schema_manager()
            .query_manager()
            .pending_local_row_id_holders(row_id),
        0,
        "once B no longer tracks its write to R, no subscription may keep R's id"
    );
}

#[test]
fn a_confirmation_reveals_a_row_another_client_wrote() {
    check_a_confirmation_reveals_a_row_another_client_wrote::<MemoryStorage>();
}

#[test]
fn a_confirmation_reveals_a_row_another_client_wrote_sqlite() {
    check_a_confirmation_reveals_a_row_another_client_wrote::<SqliteStorage>();
}

#[test]
fn a_retired_write_leaves_no_unflagged_pending_id() {
    check_a_retired_write_leaves_no_unflagged_pending_id::<MemoryStorage>();
}

#[test]
fn a_retired_write_leaves_no_unflagged_pending_id_sqlite() {
    check_a_retired_write_leaves_no_unflagged_pending_id::<SqliteStorage>();
}

/// SplitMix64: deterministic and dependency-free, so a failing seed replays exactly.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

const QUERIES: [Option<&str>; 3] = [None, Some("x"), Some("y")];
const TAGS: [&str; 3] = ["x", "y", "z"];
const MAX_ROWS_PER_WRITER: usize = 3;

struct LiveView {
    query: usize,
    view: View,
    deliveries: Arc<AtomicUsize>,
}

struct DifferentialRun {
    /// One line per quiet point: the operation, then every delivered view with row ids
    /// replaced by stable labels. Traces of one build differ between runs on `MemoryStorage`,
    /// so compare builds by the set of seeds that fail, not by their traces.
    trace: String,
    /// The first quiet point at which two delivered views of the same query disagreed.
    first_disagreement: Option<String>,
    /// The first quiet point at which a delivered view differed from the fate model: B's own
    /// rows at their latest version, A's rows at the latest version whose edge fate reached B.
    /// Reported, not asserted: on a tierless node an edge-tier view already shows versions
    /// before their fate arrives on linsa-v21.
    first_model_divergence: Option<String>,
    /// The first step after which a subscription on A or B held a pending local row id no local
    /// batch backs without being flagged for the retain that drops it.
    first_unflagged_unbacked_id: Option<String>,
}

fn render(view: &View, labels: &HashMap<ObjectId, String>) -> String {
    let mut rows: Vec<String> = snapshot(view)
        .into_iter()
        .map(|(id, (tag, body))| {
            let label = labels.get(&id).cloned().unwrap_or_else(|| "?".to_string());
            format!("{label}={tag}/{body}")
        })
        .collect();
    rows.sort();
    format!("{{{}}}", rows.join(","))
}

/// Random writes by A and B that move rows between tags, subscriptions opened on B at random
/// moments, and a network that sometimes holds every batch fate on its way to B and releases
/// them later, oldest first.
///
/// Invariant: at every quiet point, all delivered views of the same query at the same tier on
/// B show the same rows. They differ only in when they were opened, and what a subscription
/// shows must not depend on its own history. At the end, with every fate released, they must
/// also match a view opened last.
fn run_differential<S: Store>(
    b_node_tier: Option<DurabilityTier>,
    seed: u64,
    steps: usize,
    cross_writes: bool,
) -> DifferentialRun {
    let (mut net, alice, bob) = network::<S>(b_node_tier);
    // A must hold B's rows to update them.
    let _a_notes = cross_writes.then(|| subscribe_view(&mut net.a, None, &alice));
    let mut rng = Rng(seed);
    let no_hold = |_: &SyncPayload| false;
    let hold_fates = |payload: &SyncPayload| matches!(payload, SyncPayload::BatchFate { .. });
    let as_alice = WriteContext::from_session(alice.clone());
    let as_bob = WriteContext::from_session(bob.clone());

    let mut views: Vec<LiveView> = Vec::new();
    let open = |core: &mut Core<S>, views: &mut Vec<LiveView>, query: usize| {
        let deliveries = Arc::new(AtomicUsize::new(0));
        let view = subscribe_counted_view(core, QUERIES[query], &bob, Arc::clone(&deliveries));
        views.push(LiveView {
            query,
            view,
            deliveries,
        });
    };
    for query in 0..QUERIES.len() {
        open(&mut net.b, &mut views, query);
    }
    net.pump(&no_hold);

    let mut labels: HashMap<ObjectId, String> = HashMap::new();
    let mut a_rows: Vec<ObjectId> = Vec::new();
    let mut b_rows: Vec<ObjectId> = Vec::new();
    // label -> every version written, oldest first: (batch, tag, body)
    let mut history: HashMap<String, Vec<(BatchId, &str, String)>> = HashMap::new();
    let mut held: Vec<SyncPayload> = Vec::new();
    let mut version = 0u64;
    let mut trace = String::new();
    let mut first_disagreement = None;
    let mut first_model_divergence = None;
    let mut first_unflagged_unbacked_id = None;

    for step in 0..=steps {
        let last = step == steps;
        let op = if last {
            "release all".to_string()
        } else {
            match rng.below(7) {
                writer @ (0 | 1 | 2) => {
                    let by_a = writer < 2;
                    let others_rows = if by_a { b_rows.clone() } else { a_rows.clone() };
                    let (core, rows, ctx, prefix) = if by_a {
                        (&mut net.a, &mut a_rows, &as_alice, "a")
                    } else {
                        (&mut net.b, &mut b_rows, &as_bob, "b")
                    };
                    let tag = TAGS[rng.below(TAGS.len())];
                    version += 1;
                    let body = format!("v{version}");
                    if rows.len() < MAX_ROWS_PER_WRITER && (rows.is_empty() || rng.below(3) == 0) {
                        let ((id, _), batch) = core
                            .insert(
                                "notes",
                                HashMap::from([
                                    ("tag".to_string(), Value::Text(tag.into())),
                                    ("body".to_string(), Value::Text(body.clone())),
                                ]),
                                Some(ctx),
                            )
                            .expect("insert");
                        let label = format!("{prefix}{}", rows.len());
                        labels.insert(id, label.clone());
                        history
                            .entry(label.clone())
                            .or_default()
                            .push((batch, tag, body.clone()));
                        rows.push(id);
                        format!("{label} insert {tag}/{body}")
                    } else if cross_writes && !others_rows.is_empty() && rng.below(2) == 0 {
                        // The other client's row: its author stops tracking its own write when
                        // this version reaches it.
                        let id = others_rows[rng.below(others_rows.len())];
                        let label = labels.get(&id).cloned().unwrap_or_default();
                        match core.update(
                            id,
                            vec![
                                ("tag".to_string(), Value::Text(tag.into())),
                                ("body".to_string(), Value::Text(body.clone())),
                            ],
                            Some(ctx),
                        ) {
                            Ok(batch) => {
                                history.entry(label.clone()).or_default().push((
                                    batch,
                                    tag,
                                    body.clone(),
                                ));
                                format!("{prefix} updates {label} {tag}/{body}")
                            }
                            Err(_) => format!("{prefix} cannot update {label} yet"),
                        }
                    } else {
                        let index = rng.below(rows.len());
                        let batch = core
                            .update(
                                rows[index],
                                vec![
                                    ("tag".to_string(), Value::Text(tag.into())),
                                    ("body".to_string(), Value::Text(body.clone())),
                                ],
                                Some(ctx),
                            )
                            .expect("update");
                        history
                            .entry(format!("{prefix}{index}"))
                            .or_default()
                            .push((batch, tag, body.clone()));
                        format!("{prefix}{index} update {tag}/{body}")
                    }
                }
                3 => {
                    // With cross writes both clients also open subscriptions on another table.
                    // Such a subscription copies every id its client still tracks and settles
                    // again only when that table changes, while a remote version of a `notes`
                    // row dirties every `notes` subscription and so lets it drop the id on its
                    // own.
                    let query = rng.below(QUERIES.len() + usize::from(cross_writes));
                    if query == QUERIES.len() {
                        for (core, session) in [(&mut net.b, &bob), (&mut net.a, &alice)] {
                            core.subscribe_with_durability_and_propagation(
                                QueryBuilder::new("other").build(),
                                |_| {},
                                Some(session.clone()),
                                ReadDurabilityOptions {
                                    tier: Some(DurabilityTier::EdgeServer),
                                    local_updates: LocalUpdates::Immediate,
                                },
                                QueryPropagation::Full,
                            )
                            .expect("subscribe to another table");
                        }
                        "open other".to_string()
                    } else {
                        open(&mut net.b, &mut views, query);
                        format!("open {:?}", QUERIES[query])
                    }
                }
                4 => {
                    held.extend(net.pump(&hold_fates));
                    format!("pump holding fates ({} held)", held.len())
                }
                5 => {
                    held.extend(net.pump(&no_hold));
                    "pump".to_string()
                }
                _ => {
                    let count = rng.below(held.len() + 1);
                    let released: Vec<SyncPayload> = held.drain(..count).collect();
                    net.release_to_b(released);
                    held.extend(net.pump(&no_hold));
                    format!("release {count} ({} still held)", held.len())
                }
            }
        };
        // Both clients: with cross writes B's updates also retire A's tracking of its own rows.
        let b_census = net
            .b
            .schema_manager()
            .query_manager()
            .pending_local_row_id_census();
        let a_census = net
            .a
            .schema_manager()
            .query_manager()
            .pending_local_row_id_census();
        for (node, census) in [("B", b_census), ("A", a_census)] {
            if first_unflagged_unbacked_id.is_none() && census.unflagged_with_unbacked > 0 {
                first_unflagged_unbacked_id = Some(format!(
                    "seed {seed}, step {step} ({op}): {} subscriptions on {node}",
                    census.unflagged_with_unbacked
                ));
            }
        }
        if last {
            let released = std::mem::take(&mut held);
            net.release_to_b(released);
            net.pump(&no_hold);
            for query in 0..QUERIES.len() {
                open(&mut net.b, &mut views, query);
            }
            net.pump(&no_hold);
        } else if !op.starts_with("pump") && !op.starts_with("release") {
            // Writes and new subscriptions are only checked once the network has carried
            // them: a quiet point is where the invariant is stated.
            trace.push_str(&format!("{step}: {op}\n"));
            continue;
        }

        // The model: B's own rows show their latest version (the subscriptions are
        // `Immediate`); a row A wrote shows its latest version whose batch B has been told is
        // confirmed at edge or above.
        let confirmed: std::collections::HashSet<BatchId> = net
            .b_fates
            .iter()
            .filter(|(_, tier)| tier.is_some_and(|tier| tier >= DurabilityTier::EdgeServer))
            .map(|(batch, _)| *batch)
            .collect();
        let mut line = format!("{step}: {op} |");
        for query in 0..QUERIES.len() {
            let mut expected_rows: Vec<String> = history
                .iter()
                .filter_map(|(label, versions)| {
                    let (_, tag, body) = if label.starts_with('b') {
                        versions.last()
                    } else {
                        versions
                            .iter()
                            .rev()
                            .find(|(batch, _, _)| confirmed.contains(batch))
                    }?;
                    QUERIES[query]
                        .is_none_or(|wanted| wanted == *tag)
                        .then(|| format!("{label}={tag}/{body}"))
                })
                .collect();
            expected_rows.sort();
            let expected = format!("{{{}}}", expected_rows.join(","));
            let delivered: Vec<&LiveView> = views
                .iter()
                .filter(|v| v.query == query && v.deliveries.load(Ordering::Relaxed) > 0)
                .collect();
            let rendered: Vec<String> =
                delivered.iter().map(|v| render(&v.view, &labels)).collect();
            line.push_str(&format!(
                " {:?}: {} model {expected}",
                QUERIES[query],
                rendered.join(" ")
            ));
            if first_disagreement.is_none() && rendered.windows(2).any(|w| w[0] != w[1]) {
                first_disagreement = Some(format!(
                    "seed {seed}, step {step} ({op}): views of {:?} disagree: {}",
                    QUERIES[query],
                    rendered.join(" vs ")
                ));
            }
            if first_model_divergence.is_none() && rendered.iter().any(|r| *r != expected) {
                first_model_divergence = Some(format!(
                    "seed {seed}, step {step} ({op}): views of {:?} {} vs model {expected}",
                    QUERIES[query],
                    rendered.join(" ")
                ));
            }
        }
        trace.push_str(&line);
        trace.push('\n');
    }

    DifferentialRun {
        trace,
        first_disagreement,
        first_model_divergence,
        first_unflagged_unbacked_id,
    }
}

const DIFFERENTIAL_STEPS: usize = 40;

/// Seeds that failed one invariant, each with its first failure and its trace.
type Failures = Vec<(String, String)>;

/// Run `seeds` histories; returns the seeds that broke same-query agreement and the seeds that
/// left an unbacked pending id unflagged.
fn differential<S: Store>(
    b_node_tier: Option<DurabilityTier>,
    name: &str,
    seeds: u64,
    cross_writes: bool,
) -> (Failures, Failures) {
    let mut traces = String::new();
    let mut disagreements = Vec::new();
    let mut unflagged = Vec::new();
    let mut divergences = Vec::new();
    for seed in 0..seeds {
        let run = run_differential::<S>(b_node_tier, seed, DIFFERENTIAL_STEPS, cross_writes);
        if let Some(divergence) = run.first_model_divergence {
            divergences.push(divergence);
        }
        traces.push_str(&format!("== seed {seed}\n{}", run.trace));
        if let Some(failure) = run.first_unflagged_unbacked_id {
            unflagged.push((failure, run.trace.clone()));
        }
        if let Some(disagreement) = run.first_disagreement {
            disagreements.push((disagreement, run.trace));
        }
    }
    if let Ok(dir) = std::env::var("CONFIRMATION_DIFFERENTIAL_TRACE_DIR") {
        std::fs::write(format!("{dir}/{name}.trace"), &traces).expect("write trace");
    }
    // The fate model assumes each client writes only its own rows.
    if !cross_writes {
        eprintln!(
            "{name}: {} of {seeds} seeds diverged from the fate model{}",
            divergences.len(),
            divergences
                .first()
                .map(|d| format!("; first: {d}"))
                .unwrap_or_default()
        );
    }
    (disagreements, unflagged)
}

fn assert_none(failures: &Failures, seeds: u64, what: &str) {
    if let Some((first, trace)) = failures.first() {
        panic!(
            "{} of {seeds} seeds {what}; first: {first}\n--- its trace ---\n{trace}",
            failures.len()
        );
    }
}

#[test]
#[ignore = "pre-existing on linsa-v21: same-query subscriptions on a tierless client disagree in 35 of 300 seeds (8 of 60 on SQLite); the confirmation change added none in three runs"]
fn differential_same_query_views_agree_on_a_tierless_client() {
    let (disagreements, _) = differential::<MemoryStorage>(None, "tierless", 300, false);
    assert_none(
        &disagreements,
        300,
        "broke same-query agreement on a tierless B",
    );
}

#[test]
fn differential_pending_row_ids_stay_backed_or_flagged_on_a_tierless_client() {
    let (_, unflagged) = differential::<MemoryStorage>(None, "tierless", 300, false);
    assert_none(
        &unflagged,
        300,
        "left an unbacked pending id unflagged on a tierless B",
    );
}

#[test]
fn differential_same_query_views_agree_on_a_local_tier_client() {
    let (disagreements, unflagged) =
        differential::<MemoryStorage>(Some(DurabilityTier::Local), "local", 300, false);
    assert_none(
        &unflagged,
        300,
        "left an unbacked pending id unflagged on a Local-tier B",
    );
    assert_none(
        &disagreements,
        300,
        "broke same-query agreement on a Local-tier B",
    );
}

#[test]
#[ignore = "pre-existing on linsa-v21: same-query subscriptions on a tierless client disagree in 35 of 300 seeds (8 of 60 on SQLite); the confirmation change added none in three runs"]
fn differential_same_query_views_agree_on_a_tierless_client_sqlite() {
    let (disagreements, _) = differential::<SqliteStorage>(None, "tierless-sqlite", 60, false);
    assert_none(
        &disagreements,
        60,
        "broke same-query agreement on a tierless B",
    );
}

#[test]
fn differential_pending_row_ids_stay_backed_or_flagged_on_a_tierless_client_sqlite() {
    let (_, unflagged) = differential::<SqliteStorage>(None, "tierless-sqlite", 60, false);
    assert_none(
        &unflagged,
        60,
        "left an unbacked pending id unflagged on a tierless B",
    );
}

#[test]
fn differential_same_query_views_agree_on_a_local_tier_client_sqlite() {
    let (disagreements, unflagged) =
        differential::<SqliteStorage>(Some(DurabilityTier::Local), "local-sqlite", 60, false);
    assert_none(
        &unflagged,
        60,
        "left an unbacked pending id unflagged on a Local-tier B",
    );
    assert_none(
        &disagreements,
        60,
        "broke same-query agreement on a Local-tier B",
    );
}

#[test]
#[ignore = "pre-existing on linsa-v21: when the clients also update each other's rows, same-query subscriptions on a tierless client disagree in the same 23 of 300 seeds (2 of 60 on SQLite) on linsa-v21 and with the confirmation change; on linsa-v21 one more seed sometimes fails"]
fn differential_same_query_views_agree_on_a_tierless_client_when_clients_update_each_others_rows() {
    let (disagreements, _) =
        differential::<MemoryStorage>(None, "tierless-cross-agreement", 300, true);
    assert_none(
        &disagreements,
        300,
        "broke same-query agreement on a tierless B whose rows A also updates",
    );
    let (disagreements, _) =
        differential::<SqliteStorage>(None, "tierless-cross-agreement-sqlite", 60, true);
    assert_none(
        &disagreements,
        60,
        "broke same-query agreement on a tierless B whose rows A also updates",
    );
}

#[test]
fn differential_pending_row_ids_stay_backed_or_flagged_when_clients_update_each_others_rows() {
    let (_, unflagged) = differential::<MemoryStorage>(None, "tierless-cross", 300, true);
    assert_none(
        &unflagged,
        300,
        "left an unbacked pending id unflagged on A or on a tierless B",
    );
}

#[test]
fn differential_pending_row_ids_stay_backed_or_flagged_when_clients_update_each_others_rows_sqlite()
{
    let (_, unflagged) = differential::<SqliteStorage>(None, "tierless-cross-sqlite", 60, true);
    assert_none(
        &unflagged,
        60,
        "left an unbacked pending id unflagged on A or on a tierless B",
    );
}

#[test]
fn differential_same_query_views_agree_on_a_local_tier_client_when_clients_update_each_others_rows()
{
    let (disagreements, unflagged) =
        differential::<MemoryStorage>(Some(DurabilityTier::Local), "local-cross", 300, true);
    assert_none(
        &unflagged,
        300,
        "left an unbacked pending id unflagged on A or on a Local-tier B",
    );
    assert_none(
        &disagreements,
        300,
        "broke same-query agreement on a Local-tier B whose rows A also updates",
    );
}

#[test]
fn differential_same_query_views_agree_on_a_local_tier_client_when_clients_update_each_others_rows_sqlite()
 {
    let (disagreements, unflagged) =
        differential::<SqliteStorage>(Some(DurabilityTier::Local), "local-cross-sqlite", 60, true);
    assert_none(
        &unflagged,
        60,
        "left an unbacked pending id unflagged on A or on a Local-tier B",
    );
    assert_none(
        &disagreements,
        60,
        "broke same-query agreement on a Local-tier B whose rows A also updates",
    );
}
