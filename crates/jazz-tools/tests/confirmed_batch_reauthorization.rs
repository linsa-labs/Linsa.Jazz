//! What does one confirmed write cost the subscriptions it never touched?
//!
//! A local write is sent upstream, the server confirms it, and the confirmation comes back as
//! a `BatchFate` carrying a tier. Two sites USED TO set `needs_visibility_recompute` on EVERY
//! subscription whose `durability_tier` was at or below a confirmed tier, with no relation to
//! the rows the confirmed batch contains: `apply_pending_batch_fate_effects`, with the maximum
//! confirmed tier over the new fates, and the runtime core's fate application, with each
//! fate's own tier. `QuerySubscription::needs_settle` includes that flag, so
//! each of those subscriptions settles in the same pass, and a subscription that filters by
//! explicit authorization re-runs `authorized_tuples_from_graph_with_cache` over EVERY output
//! tuple's provenance rows: one storage row load plus one policy evaluation each, since the
//! cross-tick verdict cache is off by default.
//!
//! MEASURED on the Linsa app (simulator, jazz-rn linsa-v21, 2026-09-16): typing a draft into
//! one chat cost 5 067 row-policy evaluations per second against 105/s idle, in passes of
//! ~1 054 rows each — the whole subscribed set of the open screen (messages, attachments,
//! media, members, chats), re-authorized once per keystroke, at ~13 % of a core. The typed
//! row lives in `chat_drafts`; nothing else in that pass had changed.
//!
//! Every subscription the app creates meets both conditions by construction: it carries a
//! session (so explicit authorization filtering is on) and a durability tier (the JS default
//! is `edge` whenever a sync server is configured, `local` otherwise — never none).
//!
//! The write here lands one row in a table only ONE subscription reads, and the other
//! subscription's result set is what grows. Two gates, because there are two ways to be
//! wrong: the first bounds the cost of a single confirmed write, the second says that cost
//! must not follow the size of a subscription the write has nothing to do with. A fix that
//! only shaves a constant passes the first and fails the second.
//!
//! Before the fix the cost obeyed an exact law — `scope_authz_checks = 2·N + 6` for N
//! unrelated rows, one pass per marking site — measured here as 106 at N=50 and 406 at
//! N=200. Both gates bound the cost at N/4, which no per-row law can satisfy and which any
//! implementation whose cost follows the change rather than the subscription passes easily.

#![cfg(feature = "test")]

use std::any::Any;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};

use jazz_tools::query_manager::manager::LocalUpdates;
use jazz_tools::query_manager::policy::PolicyExpr;
use jazz_tools::query_manager::session::{Session, WriteContext};
use jazz_tools::query_manager::settle_cost::SettleCounts;
use jazz_tools::query_manager::types::{
    ColumnType, Schema, SchemaBuilder, TablePolicies, TableSchema, Value,
};
use jazz_tools::runtime_core::{NoopScheduler, ReadDurabilityOptions, RuntimeCore, SyncSender};
use jazz_tools::schema_manager::{AppId, SchemaManager};
use jazz_tools::storage::MemoryStorage;
use jazz_tools::sync_manager::{
    ClientId, Destination, DurabilityTier, InboxEntry, OutboxEntry, QueryPropagation, ServerId,
    Source, SyncManager, SyncPayload,
};
use jazz_tools::{ObjectId, Query};

type Core = RuntimeCore<MemoryStorage, NoopScheduler>;

/// Rows held by the subscription the measured write has nothing to do with.
const UNRELATED_ROWS: usize = 200;

/// The smaller unrelated set the scaling gate compares against.
const FEWER_UNRELATED_ROWS: usize = 50;

/// Upper bound on pump rounds; reaching it means the pair never settled and the test is invalid.
const MAX_EXCHANGE_ROUNDS: usize = 200;

/// Serialises the tests here: the settle counters are process-global, so two scenarios
/// running at once would blend their deltas. Same shape as the lock in
/// `settle_cost_accounting.rs`.
fn measure_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
}

/// What one confirmed write cost, as the settle counters saw it.
///
/// `SettleCounts` are process-global and the server node runs in this same process, so
/// `row_loads` is the pair's total, not the client's. The authorization counters are not
/// blended in practice: the server carries no authorization schema, so it runs no per-row
/// check at all. `full_scans` is the only field read from the client alone.
struct Cost {
    authz_checks: u64,
    authz_evals: u64,
    row_loads: u64,
    /// Full-store history walks the confirmation paid to find its own batch's rows.
    full_scans: u64,
}

/// The node's outbox, kept where the test can drain it — the crate's own `VecSyncSender`
/// accessor is test-internal, so the pump installs this instead.
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

fn structural_schema() -> Schema {
    SchemaBuilder::new()
        .table(TableSchema::builder("notes").column("body", ColumnType::Text))
        .table(TableSchema::builder("drafts").column("body", ColumnType::Text))
        .build()
}

/// The same tables, with a select policy that is a constant. This is the shape Linsa's own
/// permissions compile to — `allowRead.always()` becomes `True` for nearly every table — and
/// it is deliberately the CHEAPEST policy there is: what these gates measure is how often a
/// verdict is recomputed, not how much one costs. It differs from the structural schema,
/// which is what turns explicit authorization filtering on for a session-carrying
/// subscription (`QueryManager`, `uses_explicit_authorization_filtering`).
fn authorization_schema() -> Schema {
    let open = || {
        TablePolicies::new()
            .with_select(PolicyExpr::True)
            .with_insert(PolicyExpr::True)
    };
    SchemaBuilder::new()
        .table(
            TableSchema::builder("notes")
                .column("body", ColumnType::Text)
                .policies(open()),
        )
        .table(
            TableSchema::builder("drafts")
                .column("body", ColumnType::Text)
                .policies(open()),
        )
        .build()
}

/// A node confirms batches only if it has a durability tier of its own: the tier it stamps on
/// the fate is its own, and a tier-less node issues no `DurableDirect` at all.
fn core(app_name: &str, authorization: Option<Schema>, tier: DurabilityTier) -> (Core, Outbox) {
    let schema_manager = SchemaManager::new(
        SyncManager::new().with_durability_tier(tier),
        structural_schema(),
        AppId::from_name(app_name),
        "dev",
        "main",
    )
    .expect("schema manager");
    let mut core = RuntimeCore::new(schema_manager, MemoryStorage::new(), NoopScheduler);
    let outbox = Outbox::default();
    core.set_sync_sender(Box::new(outbox.clone()));
    if let Some(schema) = authorization {
        core.schema_manager_mut()
            .query_manager_mut()
            .set_authorization_schema(schema);
    }
    core.immediate_tick();
    (core, outbox)
}

fn text(column: &str, value: &str) -> HashMap<String, Value> {
    HashMap::from([(column.to_string(), Value::Text(value.to_string()))])
}

/// The app's own shape: local writes visible immediately, delivery gated on a tier.
fn app_durability() -> ReadDurabilityOptions {
    ReadDurabilityOptions {
        tier: Some(DurabilityTier::Local),
        local_updates: LocalUpdates::Immediate,
    }
}

/// Subscribe with a session and a tier, counting the rows the subscription delivers.
///
/// Counted in the callback because `RuntimeCore` drains the query manager's update queue
/// inside its own tick and routes the deltas to subscribers: nothing survives for a caller
/// to collect afterwards.
fn subscribe_counting(core: &mut Core, table: &str, session: &Session) -> Arc<AtomicUsize> {
    subscribe_counting_with(core, table, session, app_durability())
}

fn subscribe_counting_with(
    core: &mut Core,
    table: &str,
    session: &Session,
    durability: ReadDurabilityOptions,
) -> Arc<AtomicUsize> {
    let query = core
        .schema_manager_mut()
        .query_manager_mut()
        .query(table)
        .build();
    subscribe_query_counting(core, query, session, durability)
}

fn subscribe_query_counting(
    core: &mut Core,
    query: Query,
    session: &Session,
    durability: ReadDurabilityOptions,
) -> Arc<AtomicUsize> {
    let delivered = Arc::new(AtomicUsize::new(0));
    let sink = Arc::clone(&delivered);
    core.subscribe_with_durability_and_propagation(
        query,
        move |delta| {
            sink.fetch_add(delta.ordered_delta.added.len(), Ordering::Relaxed);
        },
        Some(session.clone()),
        durability,
        QueryPropagation::Full,
    )
    .expect("subscribe");
    delivered
}

struct Pair {
    client: Core,
    client_outbox: Outbox,
    server: Core,
    server_outbox: Outbox,
    client_id: ClientId,
    server_id: ServerId,
}

impl Pair {
    /// Drive both nodes until the exchange goes quiet, returning everything the server emitted.
    ///
    /// Quiet, not a fixed number of rounds: a seal can reach the server ahead of its rows, and
    /// the server answers that with `BatchFate::Missing` to ask for a resend, so the
    /// confirmation only arrives a couple of round trips after the write.
    fn exchange(&mut self) -> Vec<OutboxEntry> {
        let mut server_outputs = Vec::new();
        for _ in 0..MAX_EXCHANGE_ROUNDS {
            let mut moved = false;

            self.client.batched_tick();
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
                let to_client = entry.destination == Destination::Client(self.client_id);
                server_outputs.push(entry.clone());
                if to_client {
                    self.client.park_sync_message(InboxEntry {
                        source: Source::Server(self.server_id),
                        payload: entry.payload,
                    });
                }
            }
            self.client.batched_tick();
            self.client.immediate_tick();

            if !moved {
                return server_outputs;
            }
        }
        panic!("the client and server never went quiet in {MAX_EXCHANGE_ROUNDS} rounds");
    }
}

/// Seed `unrelated_rows` notes plus one draft, subscribe to both tables, then measure what the
/// client spends on authorization while ONE further `drafts` row is written and confirmed.
fn cost_of_one_confirmed_write(unrelated_rows: usize) -> Cost {
    let (client, client_outbox) = core(
        "confirmed-batch-reauth",
        Some(authorization_schema()),
        DurabilityTier::Local,
    );
    let (server, server_outbox) = core("confirmed-batch-reauth", None, DurabilityTier::EdgeServer);
    let mut pair = Pair {
        client,
        client_outbox,
        server,
        server_outbox,
        client_id: ClientId::new(),
        server_id: ServerId::new(),
    };
    let alice = Session::new("alice");
    pair.server.add_client(pair.client_id, Some(alice.clone()));
    pair.client.add_server(pair.server_id);
    let as_alice = WriteContext::from_session(alice.clone());

    for index in 0..unrelated_rows {
        pair.client
            .insert(
                "notes",
                text("body", &format!("note-{index}")),
                Some(&as_alice),
            )
            .expect("seed an unrelated note");
    }
    pair.client
        .insert("drafts", text("body", "draft"), Some(&as_alice))
        .expect("seed the draft the writes land in");

    let notes_delivered = subscribe_counting(&mut pair.client, "notes", &alice);
    let drafts_delivered = subscribe_counting(&mut pair.client, "drafts", &alice);
    pair.client.immediate_tick();
    let _ = pair.exchange();

    assert_eq!(
        notes_delivered.load(Ordering::Relaxed),
        unrelated_rows,
        "fixture precondition: the unrelated subscription must hold every seeded row, else \
         there is nothing for the confirmation to re-authorize"
    );

    let base = SettleCounts::snapshot();
    let scans_before = pair.client.local_batch_full_scan_count();
    let drafts_before = drafts_delivered.load(Ordering::Relaxed);
    pair.client
        .insert("drafts", text("body", "draft-a"), Some(&as_alice))
        .expect("the measured write — one keystroke into the draft");
    let server_outputs = pair.exchange();
    let counts = SettleCounts::snapshot().since(base);
    let full_scans = pair.client.local_batch_full_scan_count() - scans_before;

    assert!(
        server_outputs.iter().any(|entry| matches!(
            &entry.payload,
            SyncPayload::BatchFate { fate } if fate.confirmed_tier().is_some()
        )),
        "fixture precondition: the server must confirm the write at a tier, else this gates \
         nothing"
    );
    assert_eq!(
        drafts_delivered.load(Ordering::Relaxed),
        drafts_before + 1,
        "fixture precondition: the measured write must actually reach its own subscription. \
         Both gates here bound work DOWNWARDS, so a regression that stopped delivering the \
         write, or stopped settling the writer's own subscription, would make them pass more \
         comfortably instead of failing"
    );

    Cost {
        authz_checks: counts.scope_authz_checks,
        authz_evals: counts.scope_authz_evals,
        row_loads: counts.row_loads,
        full_scans,
    }
}

#[test]
fn a_confirmed_write_does_not_reauthorize_the_subscriptions_it_did_not_touch() {
    let _serialised = measure_lock();

    let cost = cost_of_one_confirmed_write(UNRELATED_ROWS);

    assert!(
        cost.authz_checks < UNRELATED_ROWS as u64 / 4,
        "one confirmed write of a single `drafts` row cost {} authorization checks \
         ({} of them fresh policy evaluations, {} row loads) with {UNRELATED_ROWS} unrelated \
         rows subscribed: the confirmation marks every subscription at or below its tier for \
         a visibility recompute, so the price of a keystroke follows the size of everything \
         subscribed rather than the size of the change",
        cost.authz_checks,
        cost.authz_evals,
        cost.row_loads,
    );

    assert_eq!(
        cost.full_scans, 0,
        "a confirmation must find its own batch's rows through the batchId->rows index, not \
         by walking every table's history: a full-store scan here would trade one unbounded \
         per-write cost for another. This node authored the batch, so the lookup resolves \
         from its own submission or batch record — the assertion guards against a future \
         narrowing of that path, not against anything the current code does"
    );
}

#[test]
fn the_cost_of_a_confirmed_write_does_not_follow_the_size_of_an_unrelated_subscription() {
    let _serialised = measure_lock();

    let small = cost_of_one_confirmed_write(FEWER_UNRELATED_ROWS);
    let large = cost_of_one_confirmed_write(UNRELATED_ROWS);
    let extra_rows = (UNRELATED_ROWS - FEWER_UNRELATED_ROWS) as u64;
    let growth = large.authz_checks.saturating_sub(small.authz_checks);

    assert!(
        growth < extra_rows / 4,
        "the same one-row write cost {} authorization checks with {FEWER_UNRELATED_ROWS} \
         unrelated rows subscribed and {} with {UNRELATED_ROWS}: {growth} more checks for \
         {extra_rows} more rows that the write never touched. A write's authorization work \
         must follow what it changed, not what happens to be subscribed — a fix that only \
         lowers the constant leaves the app paying per row of the open screen on every \
         keystroke",
        small.authz_checks,
        large.authz_checks,
    );
}

/// Rows the client has written before the pending-id cost is measured.
const WRITTEN_ROWS: usize = 120;

/// Subscriptions on the table the written rows land in. None of them is touched while the
/// cost is measured.
const IDLE_SUBSCRIPTIONS: usize = 8;

/// Ticks with nothing to do.
const IDLE_TICKS: usize = 10;

/// The body of the rows a late subscription shows; no other write uses it.
const LATE_BODY: &str = "shown by the late subscription";

/// Notes the client writes before it subscribes to them.
const NOTES_BEFORE_SUBSCRIBING: usize = 120;

/// Written drafts that carry `LATE_BODY`.
const LATE_ROWS: usize = 200;

/// The body of the rows that bring another writer's changes to the client.
const SCOPE_BODY: &str = "in the scope of the client's query";

/// Rows another writer creates, one exchange each.
const INBOUND_ROWS: usize = 20;

fn pending_id_holders(core: &Core, id: ObjectId) -> usize {
    core.schema_manager()
        .query_manager()
        .pending_local_row_id_holders(id)
}

fn pending_ids(core: &Core) -> usize {
    core.schema_manager()
        .query_manager()
        .pending_local_row_id_census()
        .total_ids
}

/// A Local-tier client and an edge server, connected, with `alice` on the client.
fn connected_pair(app_name: &str) -> (Pair, Session, WriteContext) {
    let (client, client_outbox) = core(
        app_name,
        Some(authorization_schema()),
        DurabilityTier::Local,
    );
    let (server, server_outbox) = core(app_name, None, DurabilityTier::EdgeServer);
    let mut pair = Pair {
        client,
        client_outbox,
        server,
        server_outbox,
        client_id: ClientId::new(),
        server_id: ServerId::new(),
    };
    let alice = Session::new("alice");
    pair.server.add_client(pair.client_id, Some(alice.clone()));
    pair.client.add_server(pair.server_id);
    let as_alice = WriteContext::from_session(alice.clone());
    (pair, alice, as_alice)
}

/// Subscribe on the client to the rows of `table` whose body is `body`.
fn subscribe_body(pair: &mut Pair, table: &str, body: &str, session: &Session) -> Arc<AtomicUsize> {
    let query = pair
        .client
        .schema_manager_mut()
        .query_manager_mut()
        .query(table)
        .filter_eq("body", Value::Text(body.to_string()))
        .build();
    subscribe_query_counting(&mut pair.client, query, session, app_durability())
}

/// Have the server rewrite `id`, a row the client wrote, and carry it to the client.
fn server_rewrites(pair: &mut Pair, id: ObjectId) {
    pair.server
        .update(
            id,
            vec![(
                "body".to_string(),
                Value::Text("rewritten by the server".to_string()),
            )],
            None,
        )
        .expect("the server rewrites a row the client wrote");
    let _ = pair.exchange();
    pair.client.batched_tick();
    pair.client.immediate_tick();
}

fn idle(pair: &mut Pair) {
    for _ in 0..IDLE_TICKS {
        pair.client.batched_tick();
        pair.client.immediate_tick();
    }
}

/// A client with `IDLE_SUBSCRIPTIONS` subscriptions on `notes` writes `WRITTEN_ROWS` notes and
/// the server confirms them all. A write keeps its row in `pending_local_row_batches` until it
/// is rejected, a remote version of that row arrives from another batch or confirmed at the
/// global tier on the same branch, or the local overlay is cleared. With an edge server and no
/// other writer none of that happens, so every row this client has written stays there, and in
/// the pending set of every subscription on `notes`, for the life of the process. On the Linsa
/// app that is every message, draft and reaction of the session. A subscription opened later
/// copies every one of those ids, whatever table it reads.
///
/// Dropping the unbacked ones is a retain over such a set. Run on every subscription a settle
/// pass skips, it looked up every one of those ids on every pass. What is counted here is that
/// retain, in the subscriptions a pass skips. The retain that ends every settle still walks the
/// whole set of each subscription the pass settles, as it did on linsa-v21; it is not counted.
///
/// The flag must go up only where an id may have lost its backing, and come down once the
/// retain has run. So the fixture takes the backing of one id away before measuring — the
/// server rewrites one note — and opens a late `drafts` subscription whose filter no write
/// matches. It holds every written id. A keystroke's own write settles it, as a local write
/// settles every subscription on its table, but the keystroke's confirmation does not: a flag
/// raised for a backed write shows there, and a flag never lowered shows on every idle pass.
/// Measured with the flag: the rewrite 121 lookups, ten idle ticks and the keystroke 0. With
/// the retain on every skipped subscription instead: 7 633, 21 460 and 13 848; with every
/// insert raising the flag, 242 for the keystroke.
#[test]
fn idle_ticks_and_keystrokes_do_not_look_up_written_rows_in_subscriptions_a_pass_skips() {
    let _serialised = measure_lock();
    let (mut pair, alice, as_alice) = connected_pair("pending-id-retain-cost");

    let notes_delivered: Vec<_> = (0..IDLE_SUBSCRIPTIONS)
        .map(|_| subscribe_counting(&mut pair.client, "notes", &alice))
        .collect();
    let drafts_delivered = subscribe_counting(&mut pair.client, "drafts", &alice);
    pair.client
        .insert("drafts", text("body", "draft"), Some(&as_alice))
        .expect("seed the draft");
    let mut written: Vec<ObjectId> = Vec::with_capacity(WRITTEN_ROWS);
    for index in 0..WRITTEN_ROWS {
        let ((id, _), _) = pair
            .client
            .insert(
                "notes",
                text("body", &format!("note-{index}")),
                Some(&as_alice),
            )
            .expect("write a note");
        written.push(id);
    }
    let server_outputs = pair.exchange();
    pair.client.batched_tick();
    pair.client.immediate_tick();

    assert!(
        notes_delivered
            .iter()
            .all(|delivered| delivered.load(Ordering::Relaxed) == WRITTEN_ROWS),
        "fixture precondition: every subscription on `notes` holds every written row"
    );
    let confirmed = server_outputs
        .iter()
        .filter(|entry| {
            matches!(
                &entry.payload,
                SyncPayload::BatchFate { fate } if fate.confirmed_tier().is_some()
            )
        })
        .count();
    assert!(
        confirmed > WRITTEN_ROWS,
        "fixture precondition: the server confirms every write at a tier ({confirmed} \
         confirmations for {} writes)",
        WRITTEN_ROWS + 1
    );
    assert!(
        pending_ids(&pair.client) >= WRITTEN_ROWS * IDLE_SUBSCRIPTIONS,
        "fixture precondition: the subscriptions on `notes` still hold the ids of the rows this \
         client wrote ({} pending ids), else there is nothing to scan",
        pending_ids(&pair.client)
    );

    let _late = subscribe_body(&mut pair, "drafts", LATE_BODY, &alice);
    // Let the server answer the new subscription first: its scope snapshot settles it, and
    // that settle would run the retain the rewrite below must leave to the flag.
    let _ = pair.exchange();
    let superseded = written[0];
    assert_eq!(
        pending_id_holders(&pair.client, superseded),
        IDLE_SUBSCRIPTIONS + 1,
        "fixture precondition: a subscription opened after the writes holds their ids too"
    );

    let tracked = pending_ids(&pair.client) as u64;
    let base = SettleCounts::snapshot();
    server_rewrites(&mut pair, superseded);
    let supersede = SettleCounts::snapshot().since(base).pending_id_retain_scans;
    assert_eq!(
        pending_id_holders(&pair.client, superseded),
        0,
        "fixture precondition: the server's version of the note reached the client, ended its \
         tracking of that note and left no subscription holding its id"
    );
    assert!(
        supersede > 0,
        "fixture precondition: the late `drafts` subscription, which the rewrite does not \
         settle, dropped the note's id through the retain this gate counts"
    );

    let base = SettleCounts::snapshot();
    idle(&mut pair);
    let idle = SettleCounts::snapshot().since(base).pending_id_retain_scans;

    let drafts_before = drafts_delivered.load(Ordering::Relaxed);
    let base = SettleCounts::snapshot();
    pair.client
        .insert("drafts", text("body", "draft-a"), Some(&as_alice))
        .expect("the measured write — one keystroke into the draft");
    let _ = pair.exchange();
    let keystroke = SettleCounts::snapshot().since(base).pending_id_retain_scans;
    assert_eq!(
        drafts_delivered.load(Ordering::Relaxed),
        drafts_before + 1,
        "fixture precondition: the measured write reaches its own subscription"
    );

    assert!(
        supersede <= tracked,
        "one note rewritten by the server looked up {supersede} pending local row ids on a \
         client tracking {tracked}: an id losing its backing may cost one retain of each set \
         that holds it, not a retain on every pass"
    );
    assert!(
        idle < WRITTEN_ROWS as u64 / 4,
        "{IDLE_TICKS} ticks with nothing to do looked up {idle} pending local row ids on a \
         client that has written {WRITTEN_ROWS} rows into a table {IDLE_SUBSCRIPTIONS} idle \
         subscriptions read. Dropping the ids no local batch backs any more must follow what \
         stopped backing them, not every pass over everything this client ever wrote"
    );
    assert!(
        keystroke < WRITTEN_ROWS as u64 / 4,
        "one confirmed write to `drafts` looked up {keystroke} pending local row ids in \
         subscriptions its confirmation does not settle, on a client that has written \
         {WRITTEN_ROWS} rows: a keystroke's cost must not grow with the session's history"
    );
}

/// What a pending id no local batch backs costs a subscription that holds it but has nothing
/// to settle.
///
/// The client writes `NOTES_BEFORE_SUBSCRIBING` notes and `LATE_ROWS` drafts with `LATE_BODY`,
/// and only then subscribes, so every subscription holds the id of every one of those rows:
/// - `late_drafts`, `drafts where body = LATE_BODY`, shows the `LATE_ROWS` drafts;
/// - `late_notes`, `notes where body = LATE_BODY`, shows nothing;
/// - `scope`, `notes where body = SCOPE_BODY`, shows the one written note with that body; it is
///   the query that brings the server's rows to the client.
///
/// Two events leave such an id in the late subscriptions without changing what they show. The
/// server rewrites the `scope` note, which ends the client's tracking of its own write, and the
/// fate of the server's version marks the row again. Then the server creates notes with
/// `SCOPE_BODY`: on a node with a durability tier each arrives with a fate, whose marking puts
/// the new row's id, which no local batch ever backed, into every subscription on `notes`.
///
/// Each event may cost a retain of the sets it left an unbacked id in, and nothing after. A
/// settle instead of that retain would re-authorize every row a late subscription shows, once
/// per event — the tier-wide sweep again, one table at a time. `late_drafts` is where that
/// shows: a rewrite on `notes` gives it nothing else to settle. On `notes` it would not show,
/// because an arriving row already re-authorizes every subscription on its table, whatever its
/// filter, as it did on linsa-v21.
#[test]
fn an_unbacked_pending_id_costs_one_retain_of_each_set_that_holds_it() {
    let _serialised = measure_lock();
    let (mut pair, alice, as_alice) = connected_pair("pending-id-lost-backing");

    let mut superseded = None;
    for index in 0..NOTES_BEFORE_SUBSCRIBING {
        let body = if index == 0 {
            SCOPE_BODY.to_string()
        } else {
            format!("note-{index}")
        };
        let ((id, _), _) = pair
            .client
            .insert("notes", text("body", &body), Some(&as_alice))
            .expect("write a note");
        superseded.get_or_insert(id);
    }
    let superseded = superseded.expect("a written note");
    for _ in 0..LATE_ROWS {
        pair.client
            .insert("drafts", text("body", LATE_BODY), Some(&as_alice))
            .expect("write a draft");
    }
    let _ = pair.exchange();
    let late_drafts = subscribe_body(&mut pair, "drafts", LATE_BODY, &alice);
    let _ = pair.exchange();
    let set = pending_ids(&pair.client) as u64;
    let _late_notes = subscribe_body(&mut pair, "notes", LATE_BODY, &alice);
    let scope = subscribe_body(&mut pair, "notes", SCOPE_BODY, &alice);
    let _ = pair.exchange();
    assert_eq!(
        late_drafts.load(Ordering::Relaxed),
        LATE_ROWS,
        "fixture precondition: `late_drafts` shows the drafts with its body"
    );
    assert_eq!(
        scope.load(Ordering::Relaxed),
        1,
        "fixture precondition: `scope` shows the one written note with its body"
    );
    assert!(
        set >= (NOTES_BEFORE_SUBSCRIBING + LATE_ROWS) as u64,
        "fixture precondition: a late subscription holds the id of every written row ({set})"
    );

    let base = SettleCounts::snapshot();
    server_rewrites(&mut pair, superseded);
    idle(&mut pair);
    let rewrite = SettleCounts::snapshot().since(base);
    let rewrite_holders = pending_id_holders(&pair.client, superseded);

    let base = SettleCounts::snapshot();
    for _ in 0..INBOUND_ROWS {
        pair.server
            .insert("notes", text("body", SCOPE_BODY), None)
            .expect("another writer's note");
        let _ = pair.exchange();
    }
    idle(&mut pair);
    let arrivals = SettleCounts::snapshot().since(base);

    assert_eq!(
        rewrite_holders, 0,
        "fixture precondition: the server's version of the note reached the client and no \
         subscription holds its id any more"
    );
    assert!(
        rewrite.pending_id_retain_scans > 0 && arrivals.pending_id_retain_scans > 0,
        "fixture precondition: both events left an unbacked id in a subscription they do not \
         settle ({} and {} lookups); if arriving rows stop doing that, this gate measures \
         nothing for them",
        rewrite.pending_id_retain_scans,
        arrivals.pending_id_retain_scans
    );

    assert!(
        rewrite.scope_authz_checks < LATE_ROWS as u64 / 2,
        "a rewrite on `notes` cost {} authorization checks, and `late_drafts` shows \
         {LATE_ROWS} drafts it has nothing to do with: a lost backing must be dropped by a \
         retain, not by settling the subscriptions that hold it",
        rewrite.scope_authz_checks
    );
    // The retire leaves the id unbacked in `late_drafts`, which has nothing to settle, while the
    // rewrite dirties both subscriptions on `notes` and they settle it away. The fate of the
    // server's version then puts it back into those two, and neither tracks the row any more:
    // three retains, each over `set` ids.
    assert!(
        rewrite.pending_id_retain_scans <= 3 * (set + 1),
        "a rewrite and {IDLE_TICKS} idle ticks looked up {} pending local row ids in sets of \
         {set}: a lost backing costs one retain of each set it touched, not one per pass",
        rewrite.pending_id_retain_scans
    );
    // An arriving row's fate marks its id twice: in the pass the fate arrives in, through the
    // query manager's fate effects, and again after the runtime core applies the fate. The
    // harness delivers the fate with its row, whose arrival settles every subscription on
    // `notes`, so the first marking is settled away and only the second leaves `late_notes`
    // holding the id with nothing to settle. A fate arriving a pass after its row would cost a
    // second retain.
    assert!(
        arrivals.pending_id_retain_scans <= INBOUND_ROWS as u64 * (set + 1),
        "{INBOUND_ROWS} rows from another writer and {IDLE_TICKS} idle ticks looked up {} \
         pending local row ids in sets of {set}: each arriving row costs one retain of \
         `late_notes`, not one per pass",
        arrivals.pending_id_retain_scans
    );
}
