//! What opening one page of a chat costs a client engine, on SQLite.
//!
//! The product path: a room shows the newest 50 messages and opens the 50 before them as
//! the reader scrolls back or jumps to a search hit. Each page is one subscription on the
//! thread query — the messages of one chat in an ordered window, each with its sender, the
//! message it replies to, its attachments and its reactions — and every one of them is
//! opened on the engine's own thread, which on a phone is the thread the UI runs on.
//!
//! Measured on the app (simulator, 100k-message store, 2026-10-01), a page pass cost
//! 22-35 ms and ~340 plan compiles. Four things made up that cost, and each has a count
//! here that is flat in what the page delivers rather than in what it could have scanned:
//!
//! - **plan compiles**: an include compiled one instance per outer row, though most
//!   messages have no attachment, no reaction and no reply. A binding the correlation index
//!   says is empty is now answered from the index (`settle_cost::EMPTY_BINDINGS`);
//! - **storage reads**: every row load probed the exact-locator table and re-read the row
//!   locator; SQLite now remembers, per transaction snapshot, which keys a small prefix
//!   holds and which locators it has read;
//! - **read transactions**: outside a transaction SQLite takes and drops the WAL read lock
//!   around every statement; a tick now reads inside one;
//! - **descriptor hashes**: every clone of a row descriptor hashed its column tree again
//!   before it could find its layout.
//!
//! and a fifth is counted so it cannot come back unseen:
//!
//! - **row codecs**: an include used to decode each outer row and encode it again with
//!   its array appended; it now carries the outer bytes over.
//!
//! The counts are exact where the fixture fixes them and ceilings where it does not; a
//! ratio alone would pass with both sides at zero.
//!
//! After the cost, the same engine — pages open, SQLite read memo warm, ticks reading
//! inside read scopes — takes a randomized run of the writes a chat makes, and after
//! each one every open page is compared with the page evaluated afresh with an instance
//! per binding and nothing remembered. Cheap is only worth having if it is the same
//! answer.
//!
//! One test, because the counters are process-global: nothing else may run beside it.
//!
//! ```text
//! cargo test -p jazz-tools --features test --test page_open_cost -- --nocapture
//! ```

#![cfg(feature = "test")]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use jazz_tools::ObjectId;
use jazz_tools::QueryBuilder;
use jazz_tools::WriteContext;
use jazz_tools::batch_fate::BatchMode;
use jazz_tools::query_manager::graph_nodes::include_routing::{
    check_empty_bindings_against_instances, force_include_empty_probe,
};
use jazz_tools::query_manager::index_declarations::IndexDeclarations;
use jazz_tools::query_manager::query::Query;
use jazz_tools::query_manager::settle_cost::SettleCounts;
use jazz_tools::query_manager::types::{ColumnType, Schema, SchemaBuilder, TableSchema, Value};
use jazz_tools::row_format::decode_row;
use jazz_tools::runtime_core::{NoopScheduler, RuntimeCore, SubscriptionDelta, SubscriptionHandle};
use jazz_tools::schema_manager::{AppId, SchemaManager};
use jazz_tools::storage::{SqliteStorage, force_sqlite_read_path};
use jazz_tools::sync_manager::SyncManager;

type Core = RuntimeCore<SqliteStorage, NoopScheduler>;

/// Messages per page — the app's window.
const PAGE: usize = 50;
/// Pages seeded. The newest ones are opened one after another.
const PAGES: usize = 8;
const MESSAGES: usize = PAGE * PAGES;
/// One message in this many replies to an earlier one, carries an attachment, has a
/// reaction. Coprime, so the three kinds fall on different messages of a page.
const REPLY_EVERY: usize = 7;
/// A reply answers the message this many before it.
const REPLY_REACHES_BACK: usize = 3;
const ATTACHMENT_EVERY: usize = 11;
const REACTION_EVERY: usize = 13;
const BASE_MS: u64 = 1_750_000_000_000;

fn schema() -> Schema {
    SchemaBuilder::new()
        .table(TableSchema::builder("users").column("name", ColumnType::Text))
        .table(
            TableSchema::builder("messages")
                .column("chat", ColumnType::Uuid)
                .fk_column("sender", "users")
                .nullable_fk_column("reply_to", "messages")
                .column("text", ColumnType::Text)
                .column("created_at", ColumnType::Timestamp),
        )
        .table(
            TableSchema::builder("attachments")
                .fk_column("message", "messages")
                .column("kind", ColumnType::Text),
        )
        .table(
            TableSchema::builder("reactions")
                .fk_column("message", "messages")
                .column("emoji", ColumnType::Text),
        )
        .build()
}

fn row(values: &[(&str, Value)]) -> HashMap<String, Value> {
    values
        .iter()
        .map(|(column, value)| (column.to_string(), value.clone()))
        .collect()
}

fn insert(core: &mut Core, table: &str, values: &[(&str, Value)]) -> ObjectId {
    let ((id, _), _) = core
        .insert(table, row(values), None)
        .unwrap_or_else(|error| panic!("insert into {table}: {error:?}"));
    id
}

/// The thread query for the page ending just before `before_ms`.
fn page_query(chat: ObjectId, before_ms: u64) -> Query {
    QueryBuilder::new("messages")
        .filter_eq("chat", Value::Uuid(chat))
        .filter_lt("created_at", Value::Timestamp(before_ms))
        .order_by_desc("created_at")
        .limit(PAGE)
        .with_array("sender_row", |sub| {
            sub.from("users").correlate("id", "messages.sender")
        })
        .with_array("reply", |sub| {
            sub.from("messages").correlate("id", "messages.reply_to")
        })
        .with_array("attachments", |sub| {
            sub.from("attachments").correlate("message", "messages.id")
        })
        .with_array("reactions", |sub| {
            sub.from("reactions").correlate("message", "messages.id")
        })
        .build()
}

/// The same window with the app's nesting: the message replied to carries its own sender
/// and its own attachments.
fn nested_page_query(chat: ObjectId, before_ms: u64) -> Query {
    QueryBuilder::new("messages")
        .filter_eq("chat", Value::Uuid(chat))
        .filter_lt("created_at", Value::Timestamp(before_ms))
        .order_by_desc("created_at")
        .limit(PAGE)
        .with_array("reply", |sub| {
            sub.from("messages")
                .correlate("id", "messages.reply_to")
                .with_array("sender_row", |nested| {
                    nested.from("users").correlate("id", "reply.sender")
                })
                .with_array("attachments", |nested| {
                    nested.from("attachments").correlate("message", "reply.id")
                })
        })
        .build()
}

/// Every delta the engine hands a subscriber, in arrival order.
type Inbox = Arc<Mutex<Vec<SubscriptionDelta>>>;

/// One open page: its subscription and what its deltas have told a subscriber so far.
struct Page {
    handle: SubscriptionHandle,
    query: Query,
    rows: Rows,
}

/// Outer row id to its values, include arrays ordered by row id — an include carries no
/// order of its own, so two evaluations may list the same rows differently.
type Rows = HashMap<ObjectId, Vec<Value>>;

fn normalized(mut values: Vec<Value>) -> Vec<Value> {
    for value in &mut values {
        if let Value::Array(rows) = value {
            // An included row may carry includes of its own.
            for row in rows.iter_mut() {
                if let Value::Row { values, .. } = row {
                    *values = normalized(std::mem::take(values));
                }
            }
            rows.sort_by_key(|row| match row {
                Value::Row { id, .. } => *id,
                _ => None,
            });
        }
    }
    values
}

/// Subscribe the way the app does: through the runtime, with a callback.
fn subscribe(core: &mut Core, inbox: &Inbox, query: Query) -> SubscriptionHandle {
    let inbox = Arc::clone(inbox);
    core.subscribe(
        query,
        move |delta| inbox.lock().expect("inbox").push(delta),
        None,
    )
    .expect("subscribe")
}

/// Route the deltas delivered so far to the pages they belong to; deltas for any other
/// subscription are returned, folded per subscription.
fn absorb(inbox: &Inbox, pages: &mut [Page]) -> Vec<(SubscriptionHandle, Rows)> {
    let mut others: Vec<(SubscriptionHandle, Rows)> = Vec::new();
    for delta in std::mem::take(&mut *inbox.lock().expect("inbox")) {
        let rows = match pages.iter_mut().find(|page| page.handle == delta.handle) {
            Some(page) => &mut page.rows,
            None => {
                let position = others
                    .iter()
                    .position(|(handle, _)| *handle == delta.handle)
                    .unwrap_or_else(|| {
                        others.push((delta.handle, Rows::new()));
                        others.len() - 1
                    });
                &mut others[position].1
            }
        };
        for removed in &delta.ordered_delta.removed {
            rows.remove(&removed.id);
        }
        let updated = delta
            .ordered_delta
            .updated
            .iter()
            .filter_map(|updated| updated.row.as_ref());
        let added = delta.ordered_delta.added.iter().map(|added| &added.row);
        for row in updated.chain(added) {
            let values = decode_row(&delta.descriptor, &row.data).expect("decode a page row");
            rows.insert(row.id, normalized(values));
        }
    }
    others
}

fn page_bound(page: usize) -> u64 {
    BASE_MS + ((MESSAGES - page * PAGE) as u64) * 60_000
}

/// Open page number `page` (0 is the newest) and return it with what settling it cost.
fn open_page(core: &mut Core, inbox: &Inbox, chat: ObjectId, page: usize) -> (Page, SettleCounts) {
    let query = page_query(chat, page_bound(page));
    let before = SettleCounts::snapshot();
    // Subscribing settles the page and delivers it: the runtime ticks on its own.
    let handle = subscribe(core, inbox, query.clone());
    let cost = SettleCounts::snapshot().since(before);
    let mut opened = [Page {
        handle,
        query,
        rows: Rows::new(),
    }];
    let others = absorb(inbox, &mut opened);
    assert!(
        others.is_empty(),
        "page {page}: deltas for a subscription nobody holds"
    );
    let [opened] = opened;
    assert_eq!(
        opened.rows.len(),
        PAGE,
        "page {page} reached its subscriber"
    );
    (opened, cost)
}

/// What a page holds, from the seeding ratios: the includes with something in them.
struct Holds {
    replies: u64,
    attachments: u64,
    reactions: u64,
    /// Replies of the page whose target carries an attachment.
    replies_to_an_attachment: u64,
}

fn page_holds(page: usize) -> Holds {
    let indexes = MESSAGES - (page + 1) * PAGE..MESSAGES - page * PAGE;
    let every = |period: usize| indexes.clone().filter(|i| i % period == period - 1).count() as u64;
    Holds {
        replies: every(REPLY_EVERY),
        attachments: every(ATTACHMENT_EVERY),
        reactions: every(REACTION_EVERY),
        replies_to_an_attachment: indexes
            .clone()
            .filter(|i| {
                i % REPLY_EVERY == REPLY_EVERY - 1
                    && (i - REPLY_REACHES_BACK) % ATTACHMENT_EVERY == ATTACHMENT_EVERY - 1
            })
            .count() as u64,
    }
}

fn seeded_core(dir: &tempfile::TempDir) -> (Core, ObjectId, ObjectId, Vec<ObjectId>) {
    let storage = SqliteStorage::open(dir.path().join("page.sqlite")).expect("open sqlite");
    let schema_manager = SchemaManager::new(
        SyncManager::new(),
        schema(),
        AppId::from_name("page-open-cost"),
        "dev",
        "main",
    )
    .expect("schema manager");
    let mut core = RuntimeCore::new(schema_manager, storage, NoopScheduler);
    // The app's declaration for the thread query, proposed before the first write so
    // every message files its entry as it lands.
    core.schema_manager_mut()
        .query_manager_mut()
        .propose_index_declarations(
            IndexDeclarations::empty()
                .with_composite("messages", "chat", "created_at")
                .expect("declaration"),
        );
    for _ in 0..8 {
        core.immediate_tick();
        core.batched_tick();
    }

    let me = insert(&mut core, "users", &[("name", Value::Text("me".into()))]);
    let peer = insert(&mut core, "users", &[("name", Value::Text("peer".into()))]);
    let chat = ObjectId::new();
    let mut messages: Vec<ObjectId> = Vec::with_capacity(MESSAGES);
    for index in 0..MESSAGES {
        let mut values = vec![
            ("chat", Value::Uuid(chat)),
            (
                "sender",
                Value::Uuid(if index % 2 == 0 { me } else { peer }),
            ),
            ("text", Value::Text(format!("message {index}"))),
            (
                "created_at",
                Value::Timestamp(BASE_MS + index as u64 * 60_000),
            ),
        ];
        if index % REPLY_EVERY == REPLY_EVERY - 1 {
            values.push((
                "reply_to",
                Value::Uuid(messages[index - REPLY_REACHES_BACK]),
            ));
        }
        let message = insert(&mut core, "messages", &values);
        messages.push(message);
        if index % ATTACHMENT_EVERY == ATTACHMENT_EVERY - 1 {
            insert(
                &mut core,
                "attachments",
                &[
                    ("message", Value::Uuid(message)),
                    ("kind", Value::Text("photo".into())),
                ],
            );
        }
        if index % REACTION_EVERY == REACTION_EVERY - 1 {
            insert(
                &mut core,
                "reactions",
                &[
                    ("message", Value::Uuid(message)),
                    ("emoji", Value::Text("+1".into())),
                ],
            );
        }
    }
    for _ in 0..8 {
        core.immediate_tick();
        core.batched_tick();
    }
    core.schema_manager_mut().query_manager_mut().take_updates();
    (core, chat, me, messages)
}

fn report(label: &str, cost: &SettleCounts) {
    eprintln!(
        "{label:<28} rows {:>3} | compiles {:>4} empty {:>4} evals {:>4} | st_ops {:>5} scopes {:>2} | \
         hashes {:>4} layouts {:>4} enc {:>4} dec {:>4} | row_loads {:>4} idx {:>4} authz {:>4}",
        cost.rows_emitted,
        cost.plan_compiles,
        cost.empty_bindings,
        cost.instance_evals,
        cost.storage_read_ops,
        cost.storage_read_scopes,
        cost.descriptor_hashes,
        cost.row_layout_compiles,
        cost.row_encodes,
        cost.row_decodes,
        cost.row_loads,
        cost.index_reads,
        cost.scope_authz_checks,
    );
}

/// Output columns of a page row: the five message columns, then the four includes.
const REPLY_COLUMN: usize = 6;
const ATTACHMENTS_COLUMN: usize = 7;
const REACTIONS_COLUMN: usize = 8;

/// How many rows the subscriber holds in `column` of `message`, or `None` when no open
/// page holds the message.
fn included(pages: &[Page], message: ObjectId, column: usize) -> Option<usize> {
    pages.iter().find_map(|page| {
        page.rows.get(&message).map(|values| match &values[column] {
            Value::Array(rows) => rows.len(),
            other => panic!("column {column} is not an include: {other:?}"),
        })
    })
}

/// One engine tick and its deltas, which must all belong to open pages.
fn tick(core: &mut Core, inbox: &Inbox, pages: &mut [Page]) {
    core.immediate_tick();
    let others = absorb(inbox, pages);
    assert!(others.is_empty(), "deltas for a subscription nobody holds");
}

/// Every open page against the same query evaluated afresh, twice: the way the engine
/// runs by default — empty bindings, the read memo, a read scope when no write is
/// pending — and with an instance per binding and SQLite answering nothing from memory.
fn assert_pages_match_a_fresh_evaluation(
    core: &mut Core,
    inbox: &Inbox,
    pages: &mut [Page],
    context: &str,
) {
    for index in 0..pages.len() {
        let evaluate = |core: &mut Core, pages: &mut [Page]| {
            let handle = subscribe(core, inbox, pages[index].query.clone());
            // Anything an OPEN page receives here arrived a tick after the write that
            // caused it: the step's own tick is over.
            let delivered = inbox.lock().expect("inbox");
            let late = delivered
                .iter()
                .filter(|delta| delta.handle != handle)
                .count();
            assert_eq!(late, 0, "{context}: an open page changed a tick late");
            drop(delivered);
            let mut others = absorb(inbox, pages);
            core.unsubscribe(handle);
            assert_eq!(others.len(), 1, "{context}: the fresh page did not settle");
            others.swap_remove(0).1
        };
        let by_default = evaluate(core, pages);
        let the_long_way = {
            let _instances = force_include_empty_probe(false);
            let _plain = force_sqlite_read_path(false, false);
            evaluate(core, pages)
        };

        let live = &pages[index].rows;
        let mut ids: Vec<&ObjectId> = live.keys().chain(the_long_way.keys()).collect();
        ids.extend(by_default.keys());
        ids.sort();
        ids.dedup();
        for id in ids {
            assert_eq!(
                live.get(id),
                the_long_way.get(id),
                "{context}: page {index}, row {id}: the open page (left) is not what the query \
                 evaluates to (right)"
            );
            assert_eq!(
                by_default.get(id),
                the_long_way.get(id),
                "{context}: page {index}, row {id}: the page opened now (left) is not what the \
                 query evaluates to the long way (right)"
            );
        }
    }
}

struct Lcg(u64);

impl Lcg {
    fn below(&mut self, bound: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) % bound as u64) as usize
    }
}

#[test]
fn a_page_costs_what_it_delivers() {
    let dir = tempfile::tempdir().expect("temp dir");
    let (mut core, chat, me, messages) = seeded_core(&dir);
    let inbox = Inbox::default();

    let mut pages: Vec<Page> = Vec::new();
    let mut costs: Vec<SettleCounts> = Vec::new();
    for page in 0..4 {
        let (opened, cost) = open_page(&mut core, &inbox, chat, page);
        report(&format!("page {page}"), &cost);
        pages.push(opened);
        costs.push(cost);
    }

    // The same page shape with every binding held as a compiled instance — what an
    // engine without empty bindings does.
    let instances_only = {
        let _instances = force_include_empty_probe(false);
        let (opened, cost) = open_page(&mut core, &inbox, chat, 4);
        report("page 4 (instances only)", &cost);
        pages.push(opened);
        cost
    };
    // With SQLite answering nothing from memory and reading outside a transaction, and
    // straight after it the next page read the default way: neighbours in the same
    // store, opened back to back.
    let plain_reads = {
        let _plain = force_sqlite_read_path(false, false);
        let (opened, cost) = open_page(&mut core, &inbox, chat, 5);
        report("page 5 (plain reads)", &cost);
        pages.push(opened);
        cost
    };
    let (opened, after_plain) = open_page(&mut core, &inbox, chat, 6);
    report("page 6", &after_plain);
    pages.push(opened);
    costs.push(after_plain);

    for (page, cost) in [0, 1, 2, 3, 6].into_iter().zip(&costs) {
        let holds = page_holds(page);
        let rows = PAGE as u64;
        assert_eq!(cost.rows_emitted, rows, "page {page} delivered a page");

        // Bindings: one per message for the sender, the attachments and the reactions,
        // and one for each message that replies to another (a NULL reference binds
        // nothing). Each is either compiled or answered from the index — and the page's
        // own plan is one compile more.
        let bindings = 3 * rows + holds.replies;
        assert_eq!(
            cost.plan_compiles + cost.empty_bindings,
            bindings + 1,
            "page {page}: {} compiles and {} empty bindings for {bindings} bindings",
            cost.plan_compiles,
            cost.empty_bindings
        );
        // Compiled: the bindings that hold something, and at most one empty one per
        // include that has any — the instance a node compiles to learn what its
        // instances read. Senders and replies have no empty binding to learn on.
        let held = 1 + rows + holds.replies + holds.attachments + holds.reactions;
        assert!(
            (held..=held + 2).contains(&cost.plan_compiles),
            "page {page} compiled {} plans; {held} of its bindings hold anything",
            cost.plan_compiles
        );

        // The tick that settled the page read inside exactly one transaction.
        assert_eq!(
            cost.storage_read_scopes, 1,
            "page {page} was not read inside one read transaction"
        );
        assert!(
            cost.storage_read_ops <= PAGE_STORAGE_READS,
            "page {page} made {} storage reads",
            cost.storage_read_ops
        );

        // Descriptors are built per compiled query, never per row or per clone, and a
        // row passes through an include as bytes.
        assert!(
            cost.row_layout_compiles <= PAGE_ROW_LAYOUTS && cost.descriptor_hashes == 0,
            "page {page} compiled {} row layouts and hashed {} descriptors",
            cost.row_layout_compiles,
            cost.descriptor_hashes
        );
        // Each included row is decoded once into its array and encoded once into the
        // row that carries it; nothing else is encoded. The other decodes are batch
        // fates, read once per batch: at most one per row the page touches, and the
        // two senders.
        let included = rows + holds.replies + holds.attachments + holds.reactions;
        assert_eq!(
            cost.row_encodes, included,
            "page {page} encoded {} rows for the {included} it includes",
            cost.row_encodes
        );
        assert!(
            (included..=2 * included + 2).contains(&cost.row_decodes),
            "page {page} decoded {} rows for the {included} it includes",
            cost.row_decodes
        );
    }

    let holds = page_holds(4);
    assert_eq!(
        (instances_only.plan_compiles, instances_only.empty_bindings),
        (3 * PAGE as u64 + holds.replies + 1, 0),
        "the instances-only page is no longer an instance per binding: the fixture lost the \
         empty bindings this gate is about"
    );
    assert_eq!(plain_reads.storage_read_scopes, 0);
    // Measured 674 against 416: the exact-locator probe and the repeated row locator of
    // every row load are answered from memory.
    assert!(
        plain_reads.storage_read_ops >= PLAIN_PAGE_STORAGE_READS,
        "the plain page made only {} storage reads: nothing here is being saved any more",
        plain_reads.storage_read_ops
    );

    // A presence beat: the reader's own `users` row, which every open page includes
    // through its senders. It re-settles what holds the row and compiles nothing.
    core.batched_tick();
    let beat = |core: &mut Core, name: &str| {
        let before = SettleCounts::snapshot();
        core.update(
            me,
            vec![("name".to_string(), Value::Text(name.into()))],
            None,
        )
        .expect("beat");
        core.immediate_tick();
        let cost = SettleCounts::snapshot().since(before);
        core.batched_tick();
        cost
    };
    let memoized_beat = beat(&mut core, "me!");
    report("beat", &memoized_beat);
    assert!(absorb(&inbox, &mut pages).is_empty());
    let plain_beat = {
        let _plain = force_sqlite_read_path(false, false);
        let cost = beat(&mut core, "me?");
        report("beat (plain reads)", &cost);
        cost
    };
    assert!(absorb(&inbox, &mut pages).is_empty());
    assert_eq!(memoized_beat.plan_compiles, 0, "a beat compiled plans");
    assert_eq!(
        (
            memoized_beat.row_layout_compiles,
            memoized_beat.descriptor_hashes
        ),
        (0, 0),
        "a beat built descriptors"
    );
    assert_eq!(memoized_beat.rows_emitted, plain_beat.rows_emitted);
    assert!(memoized_beat.rows_emitted > 0, "the beat reached no page");
    assert!(
        memoized_beat.storage_read_ops <= BEAT_STORAGE_READS
            && plain_beat.storage_read_ops >= PLAIN_BEAT_STORAGE_READS,
        "a beat made {} storage reads; with nothing remembered it makes {}",
        memoized_beat.storage_read_ops,
        plain_beat.storage_read_ops
    );
    // The beat settles twice (the write, then its own confirmation), and each settle
    // encodes the changed sender into its array and the few included rows riding the
    // same outer rows: measured 2.7 encodes and 1.1 decodes per row delivered. An
    // include that decoded and re-encoded the outer row it forwards would add two of
    // each per row, per include, per settle.
    assert!(
        memoized_beat.row_encodes <= 3 * memoized_beat.rows_emitted
            && memoized_beat.row_decodes <= 2 * memoized_beat.rows_emitted,
        "a beat that delivered {} rows encoded {} and decoded {}",
        memoized_beat.rows_emitted,
        memoized_beat.row_encodes,
        memoized_beat.row_decodes
    );

    a_nested_include_builds_descriptors_per_instance(&mut core, &inbox, &mut pages, chat);
    assert_pages_match_a_fresh_evaluation(&mut core, &inbox, &mut pages, "after the beats");
    the_open_pages_follow_a_run_of_writes(&mut core, &inbox, &mut pages, chat, me, &messages);
}

/// An include that carries includes of its own — the app's reply, with the sender and
/// the attachments of the message replied to. The nested include nodes live INSIDE each
/// instance of the outer include, so each instance compiled builds them, and their
/// descriptors, again: row layouts here scale with the instances compiled, which the
/// flat page above cannot show. They must not scale with anything else.
///
/// It also means a nested node sees one binding in its life and compiles it to learn
/// what its instances read: nested bindings are never answered from the index.
fn a_nested_include_builds_descriptors_per_instance(
    core: &mut Core,
    inbox: &Inbox,
    pages: &mut [Page],
    chat: ObjectId,
) {
    fn open(
        core: &mut Core,
        inbox: &Inbox,
        pages: &mut [Page],
        query: &Query,
    ) -> (Rows, SettleCounts) {
        let before = SettleCounts::snapshot();
        let handle = subscribe(core, inbox, query.clone());
        let cost = SettleCounts::snapshot().since(before);
        let mut others = absorb(inbox, pages);
        core.unsubscribe(handle);
        assert_eq!(others.len(), 1, "the nested page did not settle alone");
        (others.swap_remove(0).1, cost)
    }

    let query = nested_page_query(chat, page_bound(0));
    let (by_default, cost) = open(core, inbox, pages, &query);
    report("nested page", &cost);
    let (the_long_way, _) = {
        let _instances = force_include_empty_probe(false);
        let _plain = force_sqlite_read_path(false, false);
        open(core, inbox, pages, &query)
    };
    assert_eq!(by_default.len(), PAGE);
    assert!(
        by_default == the_long_way,
        "the nested page is not what its query evaluates to the long way"
    );

    // One instance per reply, and inside each one a sender and an attachments binding.
    let Holds {
        replies,
        replies_to_an_attachment,
        ..
    } = page_holds(0);
    assert!(replies > 0 && replies_to_an_attachment > 0);
    assert_eq!(
        cost.plan_compiles + cost.empty_bindings,
        1 + 3 * replies,
        "the nested page: {} compiles and {} empty bindings for {replies} replies",
        cost.plan_compiles,
        cost.empty_bindings
    );
    // The ceiling is exact — a layout more than these and it fails: the page's own three
    // (the reply's output, the include's output, the sender element), three per reply
    // instance (the nested sender's output and the two nested includes' outputs), one
    // more per reply whose target has an attachment to lay out, and the attachment
    // element once if any has.
    let layouts =
        NESTED_PAGE_ROW_LAYOUTS + NESTED_REPLY_ROW_LAYOUTS * replies + replies_to_an_attachment + 1;
    assert!(
        cost.row_layout_compiles <= layouts && cost.descriptor_hashes == 0,
        "the nested page compiled {} row layouts (at most {layouts}) and hashed {} \
         descriptors for {replies} replies",
        cost.row_layout_compiles,
        cost.descriptor_hashes
    );
}

/// Row layouts of the nested page, measured on four pages of this fixture: 24 for seven
/// replies none of which answers a message with an attachment, 26 with one that does,
/// 29 for eight replies and one.
const NESTED_PAGE_ROW_LAYOUTS: u64 = 3;
const NESTED_REPLY_ROW_LAYOUTS: u64 = 3;

/// Storage reads measured on this fixture, each ceiling a few percent above and each
/// floor a few percent below what it costs today: a page is 412-421 reads (674 with
/// nothing remembered), a beat over the seven open pages 561 (917).
const PAGE_STORAGE_READS: u64 = 440;
const PLAIN_PAGE_STORAGE_READS: u64 = 640;
const BEAT_STORAGE_READS: u64 = 590;
const PLAIN_BEAT_STORAGE_READS: u64 = 870;
/// Row layouts a page compiles: its own descriptors, one set per compiled query.
const PAGE_ROW_LAYOUTS: u64 = 8;

/// Steps of the randomized run.
const STEPS: usize = 80;
/// Empty bindings the open pages must make for themselves during the run: measured
/// 2151, the floor a little below it.
const OWN_EMPTY_BINDINGS: u64 = 2_000;

/// The writes a chat makes, at random, against the pages left open by the cost run:
/// reactions and attachments under messages that had none, their deletes, replies set,
/// cleared and pointed at a message that arrives later, staged writes committed and
/// rolled back, new messages inside an open window, presence beats and flushes.
///
/// After every step each include the step touched is checked against what the step
/// must have done, within the ONE tick that followed it; and every open page is
/// compared with the query evaluated afresh.
fn the_open_pages_follow_a_run_of_writes(
    core: &mut Core,
    inbox: &Inbox,
    pages: &mut [Page],
    chat: ObjectId,
    me: ObjectId,
    messages: &[ObjectId],
) {
    let open_from = MESSAGES - pages.len() * PAGE;
    let mut targets: Vec<ObjectId> = messages[open_from..].to_vec();
    let mut rng = Lcg(0x5eed_cafe_f00d);
    let mut children: Vec<(ObjectId, ObjectId, usize)> = Vec::new();
    let mut awaited: Vec<(ObjectId, ObjectId)> = Vec::new();
    let (mut fills, mut committed, mut rolled_back, mut arrived, mut entered) = (0, 0, 0, 0, 0);
    let elsewhere = ObjectId::new();
    let before = SettleCounts::snapshot();
    // Empty bindings made by the open pages themselves, as the steps' own ticks settle
    // them — not by the pages opened afresh to compare with.
    let mut own_empty_bindings = 0u64;
    // Every binding answered from the index in this run is also settled as an instance,
    // which must yield nothing.
    let _parity = check_empty_bindings_against_instances();

    let child = |table: &str, message: ObjectId| -> Vec<(&'static str, Value)> {
        vec![
            ("message", Value::Uuid(message)),
            if table == "reactions" {
                ("emoji", Value::Text("!".into()))
            } else {
                ("kind", Value::Text("file".into()))
            },
        ]
    };

    for step in 0..STEPS {
        let target = targets[rng.below(targets.len())];
        let (table, column) = if rng.below(2) == 0 {
            ("reactions", REACTIONS_COLUMN)
        } else {
            ("attachments", ATTACHMENTS_COLUMN)
        };
        let context;
        let step_began = SettleCounts::snapshot();
        // Empty bindings made inside the step by pages opened afresh to compare with.
        let mut compared = 0u64;
        match rng.below(14) {
            0..=3 => {
                context = format!("step {step}: a row in {table}");
                let was = included(pages, target, column);
                let id = insert(core, table, &child(table, target));
                tick(core, inbox, pages);
                children.push((id, target, column));
                if let Some(was) = was {
                    assert_eq!(included(pages, target, column), Some(was + 1), "{context}");
                    fills += usize::from(was == 0);
                }
            }
            4 => {
                context = format!("step {step}: a delete");
                if !children.is_empty() {
                    let (id, parent, column) = children.swap_remove(rng.below(children.len()));
                    let was = included(pages, parent, column);
                    core.delete(id, None).expect("delete");
                    tick(core, inbox, pages);
                    if let Some(was) = was {
                        assert_eq!(included(pages, parent, column), Some(was - 1), "{context}");
                    }
                }
            }
            5 => {
                context = format!("step {step}: a reply set or cleared");
                let reply = if rng.below(3) == 0 {
                    Value::Null
                } else {
                    Value::Uuid(messages[rng.below(MESSAGES)])
                };
                let expected = usize::from(reply != Value::Null);
                core.update(target, vec![("reply_to".to_string(), reply)], None)
                    .expect("set the reply");
                tick(core, inbox, pages);
                if included(pages, target, REPLY_COLUMN).is_some() {
                    assert_eq!(
                        included(pages, target, REPLY_COLUMN),
                        Some(expected),
                        "{context}"
                    );
                }
            }
            6 | 7 => {
                context = format!("step {step}: a reply to a message not here yet");
                let ghost = ObjectId::new();
                core.update(
                    target,
                    vec![("reply_to".to_string(), Value::Uuid(ghost))],
                    None,
                )
                .expect("point at a missing message");
                tick(core, inbox, pages);
                if included(pages, target, REPLY_COLUMN).is_some() {
                    assert_eq!(included(pages, target, REPLY_COLUMN), Some(0), "{context}");
                }
                awaited.push((ghost, target));
            }
            8 | 9 => {
                context = format!("step {step}: an awaited message arrives");
                if !awaited.is_empty() {
                    let (ghost, replier) = awaited.swap_remove(rng.below(awaited.len()));
                    let was = included(pages, replier, REPLY_COLUMN);
                    core.insert_with_id(
                        "messages",
                        row(&[
                            ("chat", Value::Uuid(elsewhere)),
                            ("sender", Value::Uuid(me)),
                            ("text", Value::Text("the awaited one".into())),
                            ("created_at", Value::Timestamp(BASE_MS)),
                        ]),
                        Some(ghost),
                        None,
                    )
                    .expect("the awaited message");
                    tick(core, inbox, pages);
                    // The replier may have been pointed elsewhere since: it holds the
                    // arrival only if it still waits for it.
                    if was == Some(0) {
                        arrived += usize::from(included(pages, replier, REPLY_COLUMN) == Some(1));
                    }
                }
            }
            10 => {
                let commit = committed <= rolled_back;
                context = format!(
                    "step {step}: a staged row in {table}, {}",
                    if commit { "committed" } else { "rolled back" }
                );
                let was = included(pages, target, column);
                // An open batch's rows are in no index and no ordinary read: the commit
                // is what files them, on a path of its own.
                let batch = core.begin_batch(BatchMode::Direct);
                let staged = WriteContext::default()
                    .with_batch_mode(BatchMode::Direct)
                    .with_batch_id(batch);
                let ((id, _), _) = core
                    .insert(table, row(&child(table, target)), Some(&staged))
                    .expect("staged insert");
                tick(core, inbox, pages);
                assert_eq!(included(pages, target, column), was, "{context}, staged");
                let comparison_began = SettleCounts::snapshot();
                assert_pages_match_a_fresh_evaluation(
                    core,
                    inbox,
                    pages,
                    &format!("{context}, staged"),
                );
                compared += SettleCounts::snapshot()
                    .since(comparison_began)
                    .empty_bindings;
                if commit {
                    core.commit_batch(batch).expect("commit");
                    committed += 1;
                    children.push((id, target, column));
                } else {
                    core.rollback_batch(batch).expect("roll back");
                    rolled_back += 1;
                }
                tick(core, inbox, pages);
                if let Some(was) = was {
                    assert_eq!(
                        included(pages, target, column),
                        Some(was + usize::from(commit)),
                        "{context}"
                    );
                    fills += usize::from(commit && was == 0);
                }
            }
            11 => {
                context = format!("step {step}: a message inside an open window");
                let index = open_from + rng.below(MESSAGES - open_from);
                let at = BASE_MS + index as u64 * 60_000 + 1 + rng.below(59_000) as u64;
                let message = insert(
                    core,
                    "messages",
                    &[
                        ("chat", Value::Uuid(chat)),
                        ("sender", Value::Uuid(me)),
                        ("text", Value::Text(format!("late {step}"))),
                        ("created_at", Value::Timestamp(at)),
                    ],
                );
                tick(core, inbox, pages);
                // It enters the page whose window it falls in — unless it falls among the
                // rows an earlier arrival pushed out of that page, which no page holds.
                if let Some(held) = included(pages, message, REACTIONS_COLUMN) {
                    assert_eq!(held, 0, "{context}");
                    entered += 1;
                }
                targets.push(message);
            }
            12 => {
                context = format!("step {step}: a beat");
                core.update(
                    me,
                    vec![("name".to_string(), Value::Text(format!("me {step}")))],
                    None,
                )
                .expect("beat");
                tick(core, inbox, pages);
            }
            _ => {
                context = format!("step {step}: an idle tick");
                tick(core, inbox, pages);
            }
        }
        // One time in three the writes are flushed first, so the pages opened for the
        // comparison read in a transaction of their own — a read scope, with whatever
        // SQLite remembered from the write transaction gone; the other two they read
        // inside the transaction the step's writes left open.
        own_empty_bindings += SettleCounts::snapshot().since(step_began).empty_bindings - compared;
        if rng.below(3) == 0 {
            // The flush confirms the step's writes, and a confirmation may be delivered:
            // what it must not do is change what a page holds.
            let held: Vec<Rows> = pages.iter().map(|page| page.rows.clone()).collect();
            core.batched_tick();
            tick(core, inbox, pages);
            for (index, held) in held.iter().enumerate() {
                assert!(
                    *held == pages[index].rows,
                    "{context}: page {index} changed a tick late, when the step was flushed"
                );
            }
        }
        assert_pages_match_a_fresh_evaluation(core, inbox, pages, &context);
    }

    let run = SettleCounts::snapshot().since(before);
    eprintln!(
        "run: {fills} fills of an empty include, {committed} committed, {rolled_back} rolled \
         back, {arrived} awaited arrivals, {entered} messages entered a page, {} empty \
         bindings ({own_empty_bindings} by the open pages), {} read scopes",
        run.empty_bindings, run.storage_read_scopes
    );
    assert!(
        entered >= 2,
        "only {entered} new messages entered an open page"
    );
    assert!(
        fills >= 12,
        "only {fills} writes landed on an empty include"
    );
    assert!(
        committed >= 2 && rolled_back >= 2,
        "{committed} / {rolled_back}"
    );
    assert!(
        arrived >= 2,
        "only {arrived} awaited messages arrived for a waiting reply"
    );
    assert!(
        own_empty_bindings >= OWN_EMPTY_BINDINGS && run.storage_read_scopes > STEPS as u64,
        "the open pages made {own_empty_bindings} empty bindings in {} read scopes: the run \
         did not go down the path it is here to check",
        run.storage_read_scopes
    );
}
