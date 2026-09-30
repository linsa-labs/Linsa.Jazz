//! Declared indexes over rows written before the declaration (`declared_index`).
//!
//! A store gains a declaration after its rows were written, so they have no composite
//! or trigram entries. Until the store has cleared and filled the index, a window or a
//! search must read the first column's own index and still answer exactly; the fill
//! runs a step per `process`, and from the step that completes it the scans read the
//! declared index. A scan trusts only the incarnation it saw complete: an index given
//! up and declared again starts over, and whatever the earlier incarnation — or an
//! engine that did not maintain it — left under its prefix is cleared before the fill.
//! Runs on both backends: the clear and the fill walk raw-table families, and SQLite
//! serves those pages through its own range seek.

use super::*;
use crate::query_manager::declared_index::load_record;
use crate::query_manager::graph_nodes::output::QuerySubscriptionId;
use crate::query_manager::index_declarations::{IndexDeclarations, IndexPhase, IndexState};
use crate::storage::SqliteStorage;

const CHAT_ROWS: u64 = 2_500;
const OTHER_ROWS: u64 = 300;
const PAGE: usize = 20;

fn fill_schema() -> Schema {
    let mut schema = Schema::new();
    schema.insert(
        TableName::new("wmsgs"),
        RowDescriptor::new(vec![
            ColumnDescriptor::new("chat", ColumnType::Uuid),
            ColumnDescriptor::new("at", ColumnType::Timestamp),
            ColumnDescriptor::new("dead", ColumnType::Boolean),
            ColumnDescriptor::new("body", ColumnType::Text),
        ])
        .into(),
    );
    schema
}

struct Row {
    id: ObjectId,
    chat: ObjectId,
    at: u64,
    body: String,
}

fn values(row: &Row) -> [Value; 4] {
    [
        Value::Uuid(row.chat),
        Value::Timestamp(row.at),
        Value::Boolean(false),
        Value::Text(row.body.clone()),
    ]
}

fn insert<H: Storage>(qm: &mut QueryManager, storage: &mut H, row: &Row) {
    let branch = get_branch(qm);
    let schema = qm.schema_context().current_schema.clone();
    qm.insert_on_branch_with_schema_and_write_context_and_id(
        storage,
        "wmsgs",
        &branch,
        &values(row),
        Some(row.id),
        &schema,
        None,
        true,
    )
    .expect("insert wmsgs row");
}

fn page_query(qm: &QueryManager, chat: ObjectId) -> Query {
    qm.query("wmsgs")
        .filter_eq("chat", Value::Uuid(chat))
        .order_by_desc("at")
        .limit(PAGE)
        .build()
}

fn search_query(qm: &QueryManager, chat: ObjectId) -> Query {
    qm.query("wmsgs")
        .filter_eq("chat", Value::Uuid(chat))
        .filter_contains("body", Value::Text("NEEDLE".to_string()))
        .order_by_desc("at")
        .build()
}

fn expected_page(rows: &[Row], chat: ObjectId) -> Vec<ObjectId> {
    let mut chat_rows: Vec<&Row> = rows.iter().filter(|row| row.chat == chat).collect();
    chat_rows.sort_by(|left, right| right.at.cmp(&left.at));
    chat_rows.into_iter().take(PAGE).map(|row| row.id).collect()
}

fn expected_search(rows: &[Row], chat: ObjectId) -> Vec<ObjectId> {
    let mut hits: Vec<&Row> = rows
        .iter()
        .filter(|row| row.chat == chat && row.body.to_lowercase().contains("needle"))
        .collect();
    hits.sort_by(|left, right| right.at.cmp(&left.at));
    hits.into_iter().map(|row| row.id).collect()
}

/// The subscription's rows, and how many rows its scans hold.
fn read(qm: &QueryManager, sub: QuerySubscriptionId) -> (Vec<ObjectId>, usize) {
    let mut scanned = Vec::new();
    qm.subscriptions
        .get(&sub)
        .expect("the subscription")
        .graph
        .collect_scanned_row_ids(&mut scanned);
    let rows = qm
        .get_subscription_results(sub)
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    (rows, scanned.len())
}

/// Rows in `chat` and in `other`, every hundredth holding a needle, written with no
/// declaration.
fn seed<H: Storage>(
    qm: &mut QueryManager,
    storage: &mut H,
    chat: ObjectId,
    other: ObjectId,
) -> Vec<Row> {
    let mut rows = Vec::new();
    for index in 0..CHAT_ROWS + OTHER_ROWS {
        let owner = if index < CHAT_ROWS { chat } else { other };
        let body = if index % 100 == 7 {
            format!("a needle #{index}")
        } else {
            format!("haystack #{index}")
        };
        let row = Row {
            id: ObjectId::new(),
            chat: owner,
            at: index * 10,
            body,
        };
        insert(qm, storage, &row);
        rows.push(row);
    }
    qm.process(storage);
    rows
}

/// Both subscriptions answer exactly; returns how many rows the page's scans hold.
fn assert_exact(
    qm: &QueryManager,
    page: QuerySubscriptionId,
    search: QuerySubscriptionId,
    rows: &[Row],
    chat: ObjectId,
    when: &str,
) -> usize {
    let (got, scanned) = read(qm, page);
    assert_eq!(got, expected_page(rows, chat), "the page {when}");
    let (got, _) = read(qm, search);
    assert_eq!(got, expected_search(rows, chat), "the search {when}");
    scanned
}

fn fills_and_answers_exactly_throughout<H: Storage>(mut qm: QueryManager, mut storage: H) {
    let (chat, other) = (ObjectId::new(), ObjectId::new());
    let mut rows = seed(&mut qm, &mut storage, chat, other);
    let page = qm.subscribe(page_query(&qm, chat)).expect("subscribe page");
    let search = qm
        .subscribe(search_query(&qm, chat))
        .expect("subscribe search");
    qm.process(&mut storage);
    assert_exact(&qm, page, search, &rows, chat, "before the declaration");

    // The declaration: the next pass applies it, recompiles both subscriptions onto the
    // declared indexes, and starts clearing. Nothing is complete yet, so both read the
    // chat through its own index.
    qm.propose_index_declarations(wmsgs_index_declarations());
    qm.process(&mut storage);
    let scanned = assert_exact(&qm, page, search, &rows, chat, "once declared");
    assert!(
        scanned >= CHAT_ROWS as usize,
        "until the fill completes the page reads the chat through its own index, \
         scanned {scanned}"
    );

    // The fill, a step per pass, with a write before every step: each write is filed
    // by the write path, whichever side of the fill's cursor its row lands on.
    let mut steps = 0;
    while qm.has_declared_index_work() {
        steps += 1;
        assert!(steps < 100, "the fill never completed");
        let row = Row {
            id: ObjectId::new(),
            chat,
            at: (CHAT_ROWS + OTHER_ROWS + steps) * 10,
            body: format!("step {steps} Needle"),
        };
        insert(&mut qm, &mut storage, &row);
        rows.push(row);
        qm.process(&mut storage);
        assert_exact(
            &qm,
            page,
            search,
            &rows,
            chat,
            &format!("at fill step {steps}"),
        );
    }
    assert!(
        steps >= 3,
        "the fill ran over several passes, not one: {steps}"
    );
    let record = load_record(&storage).expect("record");
    for index in ["chat+at", "chat>body"] {
        assert!(
            record.complete_incarnation("wmsgs", index).is_some(),
            "{index} is complete: {record:?}"
        );
    }
    // Each index holds its own entries and nothing else: one composite entry per row.
    let composite = storage
        .raw_table_family_keys("idx:wmsgs:chat+at:", None, 1_000_000)
        .expect("composite entries")
        .len();
    assert_eq!(composite, rows.len(), "composite entries after the fill");

    // A write makes both scans read again, now through the declared indexes.
    let row = Row {
        id: ObjectId::new(),
        chat,
        at: (CHAT_ROWS + OTHER_ROWS + 1000) * 10,
        body: "the last Needle".to_string(),
    };
    insert(&mut qm, &mut storage, &row);
    rows.push(row);
    qm.process(&mut storage);
    let scanned = assert_exact(&qm, page, search, &rows, chat, "after the fill");
    // Its first walk (the page and its slack), and every row written in front of its
    // frontier since — at most one per step and the one above.
    assert!(
        scanned <= 2 * PAGE + steps as usize + 1,
        "after the fill the page reads its window, scanned {scanned}"
    );
    let (_, scanned) = read(&qm, search);
    let want = expected_search(&rows, chat);
    assert!(
        scanned <= want.len() + 5,
        "after the fill the search reads its candidates, scanned {scanned}"
    );

    // A subscription opened after the fill reads through the index from the start.
    let fresh = qm
        .subscribe(page_query(&qm, other))
        .expect("subscribe page");
    qm.process(&mut storage);
    let (got, scanned) = read(&qm, fresh);
    assert_eq!(
        got,
        expected_page(&rows, other),
        "a page opened after the fill"
    );
    assert!(
        scanned <= 2 * PAGE,
        "a page opened after the fill scanned {scanned}"
    );
}

fn sqlite_query_manager() -> (QueryManager, SqliteStorage) {
    let mut qm = QueryManager::new(SyncManager::new());
    qm.set_current_schema(fill_schema(), "dev", "main");
    let mut storage = SqliteStorage::open(":memory:").expect("in-memory sqlite storage");
    crate::test_support::persist_test_schema(&mut storage, &qm.schema_context().current_schema);
    (qm, storage)
}

/// The backend a server keeps its store in (`jazz-tools server --data-dir`). The
/// directory lives as long as the process; the OS reclaims it.
fn rocksdb_query_manager() -> (QueryManager, crate::storage::RocksDBStorage) {
    let mut qm = QueryManager::new(SyncManager::new());
    qm.set_current_schema(fill_schema(), "dev", "main");
    let dir = tempfile::TempDir::new().expect("tempdir");
    let mut storage =
        crate::storage::RocksDBStorage::open(dir.path().join("store.rocksdb"), 8 * 1024 * 1024)
            .expect("rocksdb storage");
    std::mem::forget(dir);
    crate::test_support::persist_test_schema(&mut storage, &qm.schema_context().current_schema);
    (qm, storage)
}

#[test]
fn a_store_written_before_the_declaration_answers_exactly_before_during_and_after_the_fill() {
    let (qm, storage) = create_query_manager(SyncManager::new(), fill_schema());
    fills_and_answers_exactly_throughout(qm, storage);
}

#[test]
fn a_store_written_before_the_declaration_answers_exactly_on_sqlite() {
    let (qm, storage) = sqlite_query_manager();
    fills_and_answers_exactly_throughout(qm, storage);
}

#[test]
fn a_store_written_before_the_declaration_answers_exactly_on_rocksdb() {
    let (qm, storage) = rocksdb_query_manager();
    fills_and_answers_exactly_throughout(qm, storage);
}

/// The entry the composite index would hold for `row` filed in `chat` at `at`: what an
/// engine that did not maintain the index leaves behind when it moves the row out.
fn stale_window_entry<H: Storage>(
    qm: &QueryManager,
    storage: &mut H,
    row: ObjectId,
    chat: ObjectId,
    at: u64,
) {
    let value = crate::query_manager::composite_index::composite_value(
        &Value::Uuid(chat),
        &Value::Timestamp(at),
    )
    .expect("composite value");
    storage
        .index_insert("wmsgs", "chat+at", &get_branch(qm), &value, row)
        .expect("stale entry");
}

/// An index declared again starts over: before its fill it clears what is under its
/// prefix. A row an engine without the index moved out of the chat keeps its old entry
/// there, and a page of the chat trusts its window with no filter behind it, so without
/// the clear the moved row would come back in a chat it left.
fn a_re_declared_index_clears_what_it_left_behind<H: Storage>(
    mut qm: QueryManager,
    mut storage: H,
) {
    let (chat, other) = (ObjectId::new(), ObjectId::new());
    let mut rows = seed(&mut qm, &mut storage, chat, other);
    declare_indexes(&mut qm, &mut storage, wmsgs_index_declarations());
    let before = load_record(&storage).expect("record");

    // Given up, as a rollback does; the chat's newest row is moved out with the index
    // unmaintained, and its old entry is left the way such an engine leaves it; the
    // index is declared again.
    qm.propose_index_declarations(IndexDeclarations::empty());
    qm.process(&mut storage);
    let moved = rows
        .iter_mut()
        .filter(|row| row.chat == chat)
        .max_by_key(|row| row.at)
        .expect("a row in the chat");
    let (moved_id, moved_at) = (moved.id, moved.at);
    moved.chat = other;
    let moved_values = values(moved);
    qm.update(&mut storage, moved_id, &moved_values)
        .expect("move the row");
    stale_window_entry(&qm, &mut storage, moved_id, chat, moved_at);
    declare_indexes(&mut qm, &mut storage, wmsgs_index_declarations());
    let after = load_record(&storage).expect("record");
    assert!(
        after.complete_incarnation("wmsgs", "chat+at")
            > before.complete_incarnation("wmsgs", "chat+at"),
        "declared again, the index is a new incarnation: {before:?} then {after:?}"
    );

    let page = qm.subscribe(page_query(&qm, chat)).expect("subscribe page");
    qm.process(&mut storage);
    let (got, scanned) = read(&qm, page);
    assert!(
        !got.contains(&moved_id),
        "the row moved out of the chat came back through its old entry"
    );
    assert_eq!(
        got,
        expected_page(&rows, chat),
        "the page after the re-declaration"
    );
    assert!(
        scanned <= 2 * PAGE,
        "the page reads its window, scanned {scanned}"
    );
}

#[test]
fn a_re_declared_index_clears_what_it_left_behind_in_memory() {
    let (qm, storage) = create_query_manager(SyncManager::new(), fill_schema());
    a_re_declared_index_clears_what_it_left_behind(qm, storage);
}

#[test]
fn a_re_declared_index_clears_what_it_left_behind_on_sqlite() {
    let (qm, storage) = sqlite_query_manager();
    a_re_declared_index_clears_what_it_left_behind(qm, storage);
}

#[test]
fn a_re_declared_index_clears_what_it_left_behind_on_rocksdb() {
    let (qm, storage) = rocksdb_query_manager();
    a_re_declared_index_clears_what_it_left_behind(qm, storage);
}

/// A scan holds to the incarnation it saw complete. When the store starts the index
/// over under it — another runtime over the store gave it up and declared it again, so
/// the declarations this one compiled against are unchanged and nothing recompiles —
/// the scan reads the chat through its own index until the new incarnation completes.
#[test]
fn a_scan_falls_back_when_the_index_it_read_starts_over() {
    let (mut qm, mut storage) = sqlite_query_manager();
    let (chat, other) = (ObjectId::new(), ObjectId::new());
    let mut rows = seed(&mut qm, &mut storage, chat, other);
    declare_indexes(&mut qm, &mut storage, wmsgs_index_declarations());
    let page = qm.subscribe(page_query(&qm, chat)).expect("subscribe page");
    qm.process(&mut storage);
    let (got, scanned) = read(&qm, page);
    assert_eq!(
        got,
        expected_page(&rows, chat),
        "the page through the window"
    );
    assert!(
        scanned <= 2 * PAGE,
        "the page reads its window, scanned {scanned}"
    );

    // The other runtime's restart of the index, written straight to the store.
    let mut record = load_record(&storage).expect("record");
    let incarnation = record.next_incarnation;
    record.next_incarnation += 1;
    record.states.insert(
        ("wmsgs".to_string(), "chat+at".to_string()),
        IndexState {
            incarnation,
            phase: IndexPhase::Clearing { after: None },
        },
    );
    crate::query_manager::declared_index::replace(&mut storage, &record).expect("restart");

    // One pass: the clear empties the index, and a write makes the page read again.
    let row = Row {
        id: ObjectId::new(),
        chat,
        at: (CHAT_ROWS + OTHER_ROWS) * 10,
        body: "after the restart".to_string(),
    };
    insert(&mut qm, &mut storage, &row);
    rows.push(row);
    qm.process(&mut storage);
    let (got, scanned) = read(&qm, page);
    assert_eq!(
        got,
        expected_page(&rows, chat),
        "the page while the index starts over"
    );
    assert!(
        scanned >= CHAT_ROWS as usize,
        "while the index starts over the page reads the chat through its own index, \
         scanned {scanned}"
    );

    while qm.has_declared_index_work() {
        qm.process(&mut storage);
    }
    let row = Row {
        id: ObjectId::new(),
        chat,
        at: (CHAT_ROWS + OTHER_ROWS + 1) * 10,
        body: "after the fill".to_string(),
    };
    insert(&mut qm, &mut storage, &row);
    rows.push(row);
    qm.process(&mut storage);
    let (got, scanned) = read(&qm, page);
    assert_eq!(
        got,
        expected_page(&rows, chat),
        "the page after the new fill"
    );
    assert!(
        scanned <= 2 * PAGE,
        "complete again, the page reads its window, scanned {scanned}"
    );
}

/// `len` letters whose trigrams are nearly all distinct: a long message's worth.
fn long_body(seed: u64, len: usize) -> String {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            char::from(b'a' + ((state >> 33) % 26) as u8)
        })
        .collect()
}

/// A step files the entries of at most `FILL_ENTRIES` worth of rows, not a page of
/// rows: a page of long messages would otherwise hold the runtime for a page times
/// their trigrams.
fn a_fill_step_stays_within_its_entry_budget<H: Storage>(mut qm: QueryManager, mut storage: H) {
    use crate::query_manager::declared_index::FILL_ENTRIES;
    const ROWS: u64 = 40;
    const LETTERS: usize = 400;
    let chat = ObjectId::new();
    for index in 0..ROWS {
        let row = Row {
            id: ObjectId::new(),
            chat,
            at: index * 10,
            body: long_body(index, LETTERS),
        };
        insert(&mut qm, &mut storage, &row);
    }
    qm.process(&mut storage);
    let filed = |storage: &H| {
        storage
            .raw_table_family_keys("idx:wmsgs:chat>body:", None, 1_000_000)
            .expect("trigram entries")
            .len()
    };

    qm.propose_index_declarations(
        IndexDeclarations::empty()
            .with_trigram("wmsgs", "chat", "body")
            .expect("trigram declaration"),
    );
    let mut before = filed(&storage);
    let mut fill_steps = 0;
    while qm.has_declared_index_work() || fill_steps == 0 {
        qm.process(&mut storage);
        let now = filed(&storage);
        let step = now.saturating_sub(before);
        before = now;
        if step > 0 {
            fill_steps += 1;
        }
        assert!(
            step < FILL_ENTRIES + LETTERS,
            "one step filed {step} entries, past its budget of {FILL_ENTRIES} and a row"
        );
        assert!(fill_steps < 100, "the fill never completed");
    }
    let total = filed(&storage);
    assert!(
        total > 4 * FILL_ENTRIES,
        "the rows hold several budgets of entries: {total}"
    );
    assert!(
        fill_steps >= total / (FILL_ENTRIES + LETTERS),
        "{total} entries went in {fill_steps} steps"
    );
    let record = load_record(&storage).expect("record");
    assert!(
        record.complete_incarnation("wmsgs", "chat>body").is_some(),
        "the index is complete: {record:?}"
    );
}

#[test]
fn a_fill_step_stays_within_its_entry_budget_in_memory() {
    let (qm, storage) = create_query_manager(SyncManager::new(), fill_schema());
    a_fill_step_stays_within_its_entry_budget(qm, storage);
}

#[test]
fn a_fill_step_stays_within_its_entry_budget_on_sqlite() {
    let (qm, storage) = sqlite_query_manager();
    a_fill_step_stays_within_its_entry_budget(qm, storage);
}

#[test]
fn a_fill_step_stays_within_its_entry_budget_on_rocksdb() {
    let (qm, storage) = rocksdb_query_manager();
    a_fill_step_stays_within_its_entry_budget(qm, storage);
}

/// A server's store says format 4 while it maintains declared indexes and 3 once a
/// released engine gives them up, as SQLite's does: the manifest is what an engine that
/// does not maintain them refuses.
#[test]
fn a_rocksdb_store_says_format_4_while_it_maintains_declared_indexes() {
    use crate::storage::{STORE_FORMAT_V3, STORE_FORMAT_V4_DECLARED_INDEXES};

    let (mut qm, mut storage) = rocksdb_query_manager();
    let (chat, other) = (ObjectId::new(), ObjectId::new());
    let rows = seed(&mut qm, &mut storage, chat, other);
    assert_eq!(
        storage.store_format_version().expect("manifest"),
        Some(STORE_FORMAT_V3)
    );
    declare_indexes(&mut qm, &mut storage, wmsgs_index_declarations());
    assert_eq!(
        storage.store_format_version().expect("manifest"),
        Some(STORE_FORMAT_V4_DECLARED_INDEXES)
    );

    let mut qm = QueryManager::new(SyncManager::new());
    qm.set_current_schema(fill_schema(), "dev", "main");
    qm.release_declared_indexes();
    qm.open_declared_indexes(&mut storage);
    let mut passes = 0;
    loop {
        qm.process(&mut storage);
        passes += 1;
        if !qm.has_declared_index_work() {
            break;
        }
        assert!(passes < 10_000, "the released indexes were never cleared");
    }
    assert_eq!(
        storage.store_format_version().expect("manifest"),
        Some(STORE_FORMAT_V3)
    );
    let page = qm.subscribe(page_query(&qm, chat)).expect("page");
    let search = qm.subscribe(search_query(&qm, chat)).expect("search");
    qm.process(&mut storage);
    assert_exact(
        &qm,
        page,
        search,
        &rows,
        chat,
        "on a released rocksdb store",
    );
}

/// An engine released from the declared indexes, for a rollback, gives them up when it
/// opens the store — format 3, which an engine that does not maintain them opens — and
/// ignores the app's declarations after; it answers exactly without them.
#[test]
fn a_released_engine_gives_the_indexes_up_and_ignores_the_declarations() {
    use crate::storage::{STORE_FORMAT_V3, STORE_FORMAT_V4_DECLARED_INDEXES};

    let (mut qm, mut storage) = sqlite_query_manager();
    let (chat, other) = (ObjectId::new(), ObjectId::new());
    let rows = seed(&mut qm, &mut storage, chat, other);
    declare_indexes(&mut qm, &mut storage, wmsgs_index_declarations());
    let format = || storage.store_format_version().expect("manifest readable");
    assert_eq!(format(), Some(STORE_FORMAT_V4_DECLARED_INDEXES));

    let mut qm = QueryManager::new(SyncManager::new());
    qm.set_current_schema(fill_schema(), "dev", "main");
    qm.release_declared_indexes();
    qm.open_declared_indexes(&mut storage);
    assert!(
        load_record(&storage)
            .expect("record")
            .declarations
            .is_empty(),
        "a released engine kept the indexes at open"
    );
    assert_eq!(
        storage.store_format_version().expect("manifest"),
        Some(STORE_FORMAT_V3)
    );
    assert!(qm.index_declarations().is_empty());

    // The app still declares them; a released engine does not take them back.
    qm.propose_index_declarations(wmsgs_index_declarations());
    let mut passes = 0;
    loop {
        qm.process(&mut storage);
        passes += 1;
        if !qm.has_declared_index_work() {
            break;
        }
        assert!(passes < 10_000, "the released indexes were never cleared");
    }
    let record = load_record(&storage).expect("record");
    assert!(
        record.declarations.is_empty() && !record.has_work(),
        "a released engine took the app's declarations: {record:?}"
    );
    assert_eq!(
        storage.store_format_version().expect("manifest"),
        Some(STORE_FORMAT_V3)
    );

    let page = qm.subscribe(page_query(&qm, chat)).expect("page");
    let search = qm.subscribe(search_query(&qm, chat)).expect("search");
    qm.process(&mut storage);
    assert_exact(&qm, page, search, &rows, chat, "on a released engine");
}

/// A fresh released engine over `storage`, opened the way a binding opens one: the
/// app's permissions head has already proposed its declarations when the runtime is
/// built and opens the store.
fn open_released_as_a_binding_does(storage: &mut SqliteStorage) -> QueryManager {
    let mut qm = QueryManager::new(SyncManager::new());
    qm.set_current_schema(fill_schema(), "dev", "main");
    qm.release_declared_indexes();
    qm.propose_index_declarations(wmsgs_index_declarations());
    qm.open_declared_indexes(storage);
    let mut passes = 0;
    loop {
        qm.process(storage);
        passes += 1;
        if !qm.has_declared_index_work() {
            break;
        }
        assert!(passes < 10_000, "the released indexes were never cleared");
    }
    qm
}

/// A release holds for every start while the option is set, whatever the app keeps
/// declaring: each restart proposes the declarations before the store opens, and the
/// store stays at format 3 with nothing declared.
#[test]
fn a_released_engine_stays_released_across_restarts_that_declare_first() {
    use crate::storage::{STORE_FORMAT_V3, STORE_FORMAT_V4_DECLARED_INDEXES};

    let (mut qm, mut storage) = sqlite_query_manager();
    let (chat, other) = (ObjectId::new(), ObjectId::new());
    let rows = seed(&mut qm, &mut storage, chat, other);
    declare_indexes(&mut qm, &mut storage, wmsgs_index_declarations());
    assert_eq!(
        storage.store_format_version().expect("manifest"),
        Some(STORE_FORMAT_V4_DECLARED_INDEXES)
    );

    for start in 1..=2 {
        let mut qm = open_released_as_a_binding_does(&mut storage);
        let record = load_record(&storage).expect("record");
        assert!(
            record.declarations.is_empty() && !record.has_work(),
            "start {start}: a released engine took the app's declarations: {record:?}"
        );
        assert_eq!(
            storage.store_format_version().expect("manifest"),
            Some(STORE_FORMAT_V3),
            "start {start}"
        );
        let page = qm.subscribe(page_query(&qm, chat)).expect("page");
        let search = qm.subscribe(search_query(&qm, chat)).expect("search");
        qm.process(&mut storage);
        assert_exact(&qm, page, search, &rows, chat, "on a released engine");
    }
}

/// A release is what a rollback reaches for when the store's record is in doubt, so
/// it gives up a record it cannot decode too: an engine that does not maintain the
/// indexes opens only format 3. What the record named is left under its prefixes,
/// which nothing reads, and an index declared again clears its prefix first.
#[test]
fn a_release_gives_up_a_record_it_cannot_decode() {
    use crate::storage::{STORE_FORMAT_V3, STORE_FORMAT_V4_DECLARED_INDEXES};

    let (mut qm, mut storage) = sqlite_query_manager();
    let (chat, other) = (ObjectId::new(), ObjectId::new());
    let rows = seed(&mut qm, &mut storage, chat, other);
    declare_indexes(&mut qm, &mut storage, wmsgs_index_declarations());
    storage
        .raw_table_put("declared_indexes", "record", &[0xff])
        .expect("garble the record");
    assert!(
        load_record(&storage).is_err(),
        "the record must not decode, or the gate proves nothing"
    );
    assert_eq!(
        storage.store_format_version().expect("manifest"),
        Some(STORE_FORMAT_V4_DECLARED_INDEXES)
    );

    let mut qm = open_released_as_a_binding_does(&mut storage);
    assert_eq!(
        storage.store_format_version().expect("manifest"),
        Some(STORE_FORMAT_V3),
        "a release left an undecodable record's store at format 4"
    );
    let record = load_record(&storage).expect("a release writes a record that decodes");
    assert!(
        record.declarations.is_empty() && !record.has_work(),
        "{record:?}"
    );
    let page = qm.subscribe(page_query(&qm, chat)).expect("page");
    let search = qm.subscribe(search_query(&qm, chat)).expect("search");
    qm.process(&mut storage);
    assert_exact(
        &qm,
        page,
        search,
        &rows,
        chat,
        "after a release over an undecodable record",
    );
}

/// A released engine whose open could not read the record starts unreleased, and is
/// released once the record reads again: its next passes open the store again rather
/// than take the indexes the record still declares, which it would otherwise maintain
/// until a restart, the store kept at format 4.
#[test]
fn a_released_engine_whose_open_failed_is_released_once_the_record_reads() {
    use crate::query_manager::declared_index::tests::FailRecordReadStorage;
    use crate::storage::{STORE_FORMAT_V3, STORE_FORMAT_V4_DECLARED_INDEXES};

    let (mut qm, mut storage) = sqlite_query_manager();
    let (chat, other) = (ObjectId::new(), ObjectId::new());
    let rows = seed(&mut qm, &mut storage, chat, other);
    declare_indexes(&mut qm, &mut storage, wmsgs_index_declarations());
    assert_eq!(
        storage.store_format_version().expect("manifest"),
        Some(STORE_FORMAT_V4_DECLARED_INDEXES)
    );

    let mut qm = QueryManager::new(SyncManager::new());
    qm.set_current_schema(fill_schema(), "dev", "main");
    qm.release_declared_indexes();
    qm.propose_index_declarations(wmsgs_index_declarations());
    let mut failing = FailRecordReadStorage { inner: storage };
    qm.open_declared_indexes(&mut failing);
    let mut storage = failing.inner;
    assert_eq!(
        storage.store_format_version().expect("manifest"),
        Some(STORE_FORMAT_V4_DECLARED_INDEXES),
        "the open must have failed, or the gate proves nothing"
    );

    let mut passes = 0;
    loop {
        qm.process(&mut storage);
        passes += 1;
        if !qm.has_declared_index_work() {
            break;
        }
        assert!(passes < 10_000, "the released indexes were never cleared");
    }
    let record = load_record(&storage).expect("record");
    assert!(
        record.declarations.is_empty() && !record.has_work(),
        "a released engine whose open failed took the indexes back: {record:?}"
    );
    assert_eq!(
        storage.store_format_version().expect("manifest"),
        Some(STORE_FORMAT_V3),
        "a released engine whose open failed left the store at format 4"
    );
    assert!(qm.index_declarations().is_empty());

    let page = qm.subscribe(page_query(&qm, chat)).expect("page");
    let search = qm.subscribe(search_query(&qm, chat)).expect("search");
    qm.process(&mut storage);
    assert_exact(&qm, page, search, &rows, chat, "on a released engine");
}
