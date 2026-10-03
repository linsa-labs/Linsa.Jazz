//! Substring search over a trigram index (`IndexScanNode::new_trigram`).
//!
//! `where chat = x and body contains needle` must load the rows that can hold the
//! needle, not the chat: before the index, the plan scanned the `chat` index, loaded
//! every row of the chat and tested its text. `contains` on text is case-insensitive
//! (both sides folded by `trigram_index::fold`). The first gate pins the cost and the
//! case folding; the second pins the answer against a model, with needles too short
//! for a trigram, text that folds to a different length, scope moves and live
//! writes.
//!
//! A search reads its posting lists in rounds until one ends, and from then on walks
//! the candidates each list still open has not decided, reading the list from one
//! candidate to the next (`IndexScanNode::trigram_scan_ids`). The candidates are the
//! ones reading every list whole would leave; the gates pin what it costs to get them:
//! a needle's common trigrams do not make it cost the chat, no key is read twice, a
//! round reads three times what was read, an open list rules out what its read part
//! does not hold wherever those rows stand in the chat, candidates side by side or a
//! few rows apart in a list are read through and ones far apart cost a read each, a
//! list no row is in ends the search at once. A read the store fails rules nothing
//! out, a walk read answered from before its start stops the walk, and a search none
//! of whose lists can be read falls back on the chat's rows. The model runs again with
//! rounds of a key or two, so that a handful of rows takes a search through every
//! round and every way a list is decided.
//!
//! The rounds lean on what a store's index window read does with a start key and a
//! limit (`Storage::index_window_keys`), which every backend answers its own way: the
//! read-once gate, a walk gate and the model in small rounds run on SQLite and RocksDB
//! too, and through a boxed store, as a server holds its own. A store whose limited read costs
//! the range (the OPFS tree) is read a list at a time, as before the rounds.
//!
//! Each test declares `wmsgs`'s `chat>body` trigram index on its store first
//! (`declare_indexes`).

use super::*;
use crate::query_manager::graph_nodes::index_scan::{
    TRIGRAM_FIRST_ROUND_FOR_TEST, TRIGRAM_KEYS_READ, TRIGRAM_LIST_READ_THAT_FAILS,
    TRIGRAM_LIST_READS, TRIGRAM_WALK_READS, TRIGRAM_WALK_READS_IGNORE_START,
    TRIGRAM_WALK_READS_THAT_FAIL,
};
use crate::query_manager::trigram_index::fold;
#[cfg(feature = "rocksdb")]
use crate::storage::RocksDBStorage;
use crate::storage::SqliteStorage;
use std::collections::HashSet;

/// Runs `body` with a search's first round set to `keys`, on this thread only.
fn with_first_round<T>(keys: usize, body: impl FnOnce() -> T) -> T {
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            TRIGRAM_FIRST_ROUND_FOR_TEST.with(|slot| slot.set(None));
        }
    }
    let _restore = Restore;
    TRIGRAM_FIRST_ROUND_FOR_TEST.with(|slot| slot.set(Some(keys)));
    body()
}

fn search_schema() -> Schema {
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

#[derive(Clone, Debug)]
struct ModelRow {
    id: ObjectId,
    chat: ObjectId,
    at: u64,
    dead: bool,
    body: String,
}

fn memory_search_manager() -> (QueryManager, MemoryStorage) {
    create_query_manager(SyncManager::new(), search_schema())
}

/// The backend a client keeps its store in.
fn sqlite_search_manager() -> (QueryManager, SqliteStorage) {
    let mut qm = QueryManager::new(SyncManager::new());
    qm.set_current_schema(search_schema(), "dev", "main");
    let mut storage = SqliteStorage::open(":memory:").expect("in-memory sqlite storage");
    crate::test_support::persist_test_schema(&mut storage, &qm.schema_context().current_schema);
    (qm, storage)
}

/// The backend a browser keeps its store in: its limited range read costs the range.
fn opfs_search_manager() -> (QueryManager, OpfsBTreeStorage) {
    let mut qm = QueryManager::new(SyncManager::new());
    qm.set_current_schema(search_schema(), "dev", "main");
    let mut storage = OpfsBTreeStorage::memory(4 * 1024 * 1024).expect("open opfs storage");
    crate::test_support::persist_test_schema(&mut storage, &qm.schema_context().current_schema);
    (qm, storage)
}

#[cfg(feature = "rocksdb")]
thread_local! {
    /// The directories of this thread's RocksDB stores, removed when the thread — a
    /// test's own — ends, or when `remove_rocksdb_stores` is called.
    static ROCKSDB_DIRS: std::cell::RefCell<Vec<tempfile::TempDir>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Removes the directories of this thread's RocksDB stores, once the stores are dropped.
fn remove_rocksdb_stores() {
    #[cfg(feature = "rocksdb")]
    ROCKSDB_DIRS.with(|dirs| dirs.borrow_mut().clear());
}

/// The backend a server keeps its store in.
#[cfg(feature = "rocksdb")]
fn rocksdb_search_manager() -> (QueryManager, RocksDBStorage) {
    let mut qm = QueryManager::new(SyncManager::new());
    qm.set_current_schema(search_schema(), "dev", "main");
    let dir = tempfile::TempDir::new().expect("tempdir");
    let mut storage = RocksDBStorage::open(dir.path().join("store.rocksdb"), 8 * 1024 * 1024)
        .expect("rocksdb storage");
    ROCKSDB_DIRS.with(|dirs| dirs.borrow_mut().push(dir));
    crate::test_support::persist_test_schema(&mut storage, &qm.schema_context().current_schema);
    (qm, storage)
}

fn values(row: &ModelRow) -> Vec<Value> {
    vec![
        Value::Uuid(row.chat),
        Value::Timestamp(row.at),
        Value::Boolean(row.dead),
        Value::Text(row.body.clone()),
    ]
}

fn insert_row<H: Storage>(qm: &mut QueryManager, storage: &mut H, row: &ModelRow) {
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

fn search_query(qm: &QueryManager, chat: ObjectId, needle: &str) -> Query {
    qm.query("wmsgs")
        .filter_eq("chat", Value::Uuid(chat))
        .filter_eq("dead", Value::Boolean(false))
        .filter_contains("body", Value::Text(needle.to_string()))
        .order_by_desc("at")
        .build()
}

/// The rows the search must return, newest first (ties by id ascending).
fn expected(model: &[ModelRow], chat: ObjectId, needle: &str) -> Vec<ObjectId> {
    let needle = fold(needle);
    let mut rows: Vec<&ModelRow> = model
        .iter()
        .filter(|row| row.chat == chat && !row.dead && fold(&row.body).contains(&needle))
        .collect();
    rows.sort_by(|left, right| right.at.cmp(&left.at).then(left.id.cmp(&right.id)));
    rows.into_iter().map(|row| row.id).collect()
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

    fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound.max(1)
    }
}

#[test]
fn search_loads_the_candidates_not_the_chat() {
    let (mut qm, mut storage) = create_query_manager(SyncManager::new(), search_schema());
    declare_indexes(&mut qm, &mut storage, wmsgs_index_declarations());
    let chat = ObjectId::new();
    let other = ObjectId::new();
    let mut model = Vec::new();
    for index in 0..300u64 {
        for owner in [chat, other] {
            let body = match index {
                7 => "a Needle here".to_string(),
                70 => "NEEDLES".to_string(),
                170 => "haystack with a needle".to_string(),
                _ => format!("haystack number {index}"),
            };
            let row = ModelRow {
                id: ObjectId::new(),
                chat: owner,
                at: index * 10,
                dead: false,
                body,
            };
            insert_row(&mut qm, &mut storage, &row);
            model.push(row);
        }
    }
    qm.process(&mut storage);
    qm.take_updates();

    let sub = qm
        .subscribe(search_query(&qm, chat, "neeDLE"))
        .expect("subscribe");
    qm.process(&mut storage);
    // The rows the scans hold are the rows the plan loads. Read off the graph, not
    // the process-wide load counter, which tests running alongside also move.
    let mut scanned_ids = Vec::new();
    qm.subscriptions
        .get(&sub)
        .expect("the subscription")
        .graph
        .collect_scanned_row_ids(&mut scanned_ids);
    let scanned = scanned_ids.len();

    let got: Vec<ObjectId> = qm
        .get_subscription_results(sub)
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    let want = expected(&model, chat, "neeDLE");
    assert_eq!(want.len(), 3, "the fixture has three spellings in the chat");
    assert_eq!(got, want, "every spelling of the needle in the chat");
    assert!(
        scanned <= 5,
        "a search with 3 hits in a 300-row chat scanned {scanned} rows"
    );

    // A client names the column with its scope (`wmsgs.body`), as the TS runtime does.
    let scoped = qm
        .query("wmsgs")
        .filter_eq("chat", Value::Uuid(chat))
        .filter_eq("dead", Value::Boolean(false))
        .filter_contains("wmsgs.body", Value::Text("neeDLE".to_string()))
        .order_by_desc("at")
        .build();
    let sub = qm.subscribe(scoped).expect("subscribe");
    qm.process(&mut storage);
    let mut scanned_ids = Vec::new();
    qm.subscriptions
        .get(&sub)
        .expect("the subscription")
        .graph
        .collect_scanned_row_ids(&mut scanned_ids);
    let got: Vec<ObjectId> = qm
        .get_subscription_results(sub)
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    assert_eq!(got, want, "the scoped spelling finds the same rows");
    assert!(
        scanned_ids.len() <= 5,
        "the scoped spelling scanned {} rows",
        scanned_ids.len()
    );
}

/// A list longer than a round is read on from where the round left it. Of "number 1"
/// every list outlasts the first round — five as long as the chat, the sixth 611 rows
/// — so the search reads them all to their ends: each key once, 8 111 of them.
#[test]
fn a_search_reads_no_key_of_a_long_list_twice() {
    let (qm, storage) = memory_search_manager();
    reads_no_key_of_a_long_list_twice(qm, storage);
}

#[test]
fn a_search_reads_no_key_of_a_long_list_twice_on_sqlite() {
    let (qm, storage) = sqlite_search_manager();
    reads_no_key_of_a_long_list_twice(qm, storage);
}

#[cfg(feature = "rocksdb")]
#[test]
fn a_search_reads_no_key_of_a_long_list_twice_on_rocksdb() {
    let (qm, storage) = rocksdb_search_manager();
    reads_no_key_of_a_long_list_twice(qm, storage);
}

/// A server holds its store boxed (`Box<dyn Storage + Send>`): the box answers for the
/// store inside it, and the search reads in rounds there too.
#[cfg(feature = "rocksdb")]
#[test]
fn a_search_reads_no_key_of_a_long_list_twice_through_a_boxed_store() {
    let (qm, storage) = rocksdb_search_manager();
    let storage: Box<dyn Storage + Send> = Box::new(storage);
    reads_no_key_of_a_long_list_twice(qm, storage);
}

fn reads_no_key_of_a_long_list_twice<H: Storage>(mut qm: QueryManager, mut storage: H) {
    declare_indexes(&mut qm, &mut storage, wmsgs_index_declarations());
    let chat = ObjectId::new();
    let mut model = Vec::new();
    for index in 0..1500u64 {
        let row = ModelRow {
            id: ObjectId::new(),
            chat,
            at: index * 10,
            dead: false,
            body: format!("haystack number {index}"),
        };
        insert_row(&mut qm, &mut storage, &row);
        model.push(row);
    }
    qm.process(&mut storage);
    qm.take_updates();

    let needle = "number 1";
    TRIGRAM_KEYS_READ.with(|read| read.set(0));
    TRIGRAM_LIST_READS.with(|reads| reads.set(0));
    let sub = qm
        .subscribe(search_query(&qm, chat, needle))
        .expect("subscribe");
    qm.process(&mut storage);
    let keys_read = TRIGRAM_KEYS_READ.with(|read| read.get());
    let list_reads = TRIGRAM_LIST_READS.with(|reads| reads.get());

    let got: Vec<ObjectId> = qm
        .get_subscription_results(sub)
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    let want = expected(&model, chat, needle);
    assert_eq!(want.len(), 611, "1, 10..=19, 100..=199 and 1000..=1499");
    assert_eq!(got, want, "every row of the chat that holds the needle");
    assert_eq!(
        keys_read,
        5 * 1500 + 611,
        "the keys of the six lists, each read once"
    );
    assert_eq!(list_reads, 12, "in two rounds");
}

/// A haystack row of the cost gates, with an id that orders as `index` does.
fn haystack_row(chat: ObjectId, index: u64) -> ModelRow {
    ModelRow {
        id: ObjectId::from_uuid(uuid::Uuid::from_u128(1_000_000 + index as u128)),
        chat,
        at: index * 10,
        dead: false,
        body: format!("haystack number {index}"),
    }
}

/// The ids a search's scans hold: the rows its plan loads.
fn scanned_rows(
    qm: &QueryManager,
    sub: crate::query_manager::graph_nodes::output::QuerySubscriptionId,
) -> usize {
    let mut scanned_ids = Vec::new();
    qm.subscriptions
        .get(&sub)
        .expect("the subscription")
        .graph
        .collect_scanned_row_ids(&mut scanned_ids);
    scanned_ids.len()
}

/// A list still open has been read from its start, in row id order: a candidate at or
/// before its last key read that is not among those keys is not in the list. Here 300
/// rows hold "137" and none of the needle's common trigrams, and they are the oldest of
/// the chat: the first round of the long lists rules them out for nothing, and what is
/// left to walk is the ten matches past it, side by side in each of the five long
/// lists: one read, of the 40 keys a walk asks for first.
#[test]
fn an_open_list_rules_out_the_candidates_its_read_part_does_not_hold() {
    let (mut qm, mut storage) = memory_search_manager();
    declare_indexes(&mut qm, &mut storage, wmsgs_index_declarations());
    let chat = ObjectId::new();
    let mut model = Vec::new();
    for index in 0..300u64 {
        let row = ModelRow {
            id: ObjectId::from_uuid(uuid::Uuid::from_u128(1 + index as u128)),
            chat,
            at: index,
            dead: false,
            body: "pin 137".to_string(),
        };
        insert_row(&mut qm, &mut storage, &row);
        model.push(row);
    }
    for index in 0..1500u64 {
        let row = haystack_row(chat, index);
        insert_row(&mut qm, &mut storage, &row);
        model.push(row);
    }
    qm.process(&mut storage);
    qm.take_updates();

    let needle = "number 137";
    TRIGRAM_KEYS_READ.with(|read| read.set(0));
    TRIGRAM_WALK_READS.with(|count| count.set(0));
    let sub = qm
        .subscribe(search_query(&qm, chat, needle))
        .expect("subscribe");
    qm.process(&mut storage);
    let keys_read = TRIGRAM_KEYS_READ.with(|read| read.get());
    let walk_reads = TRIGRAM_WALK_READS.with(|count| count.get());

    let got: Vec<ObjectId> = qm
        .get_subscription_results(sub)
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    let want = expected(&model, chat, needle);
    assert_eq!(want.len(), 11, "137 and 1370..=1379");
    assert_eq!(got, want, "every row of the chat that holds the needle");
    assert_eq!(
        scanned_rows(&qm, sub),
        11,
        "the rows that hold \"137\" and none of \"number\" are not loaded"
    );
    assert_eq!(
        keys_read,
        312 + 411 + 6 * 512 + 5 * 40,
        "the two short lists, a round of the six long ones, and the walks"
    );
    assert_eq!(
        walk_reads, 5,
        "1370..=1379 in each list as long as the chat"
    );
}

/// A chat of 2 100 haystack rows and, newer than all of them, 300 rows that hold "137"
/// and none of "number": candidates no list read from its start reaches.
fn chat_with_newest_false_candidates<H: Storage>(
    qm: &mut QueryManager,
    storage: &mut H,
) -> (ObjectId, Vec<ModelRow>) {
    let chat = ObjectId::new();
    let mut model = Vec::new();
    for index in 0..2100u64 {
        let row = haystack_row(chat, index);
        insert_row(qm, storage, &row);
        model.push(row);
    }
    for index in 0..300u64 {
        let row = ModelRow {
            id: ObjectId::from_uuid(uuid::Uuid::from_u128(2_000_000 + index as u128)),
            chat,
            at: 100_000 + index,
            dead: false,
            body: "ser 137".to_string(),
        };
        insert_row(qm, storage, &row);
        model.push(row);
    }
    qm.process(storage);
    qm.take_updates();
    (chat, model)
}

/// The same rows, the newest of the chat: no list read from its start reaches them,
/// and nothing rules them out for nothing. The first long list is walked: a read for
/// 1370..=1379, and a second from the first of the 300 that finds the list at its end,
/// which rules them all out. They are not walked again, and not loaded.
#[test]
fn candidates_past_what_was_read_of_a_list_are_walked_not_loaded() {
    let (qm, storage) = memory_search_manager();
    walks_the_candidates_past_what_was_read_of_a_list(qm, storage);
}

#[test]
fn candidates_past_what_was_read_of_a_list_are_walked_not_loaded_on_sqlite() {
    let (qm, storage) = sqlite_search_manager();
    walks_the_candidates_past_what_was_read_of_a_list(qm, storage);
}

#[cfg(feature = "rocksdb")]
#[test]
fn candidates_past_what_was_read_of_a_list_are_walked_not_loaded_on_rocksdb() {
    let (qm, storage) = rocksdb_search_manager();
    walks_the_candidates_past_what_was_read_of_a_list(qm, storage);
}

fn walks_the_candidates_past_what_was_read_of_a_list<H: Storage>(
    mut qm: QueryManager,
    mut storage: H,
) {
    declare_indexes(&mut qm, &mut storage, wmsgs_index_declarations());
    let (chat, model) = chat_with_newest_false_candidates(&mut qm, &mut storage);

    let needle = "number 137";
    TRIGRAM_KEYS_READ.with(|read| read.set(0));
    TRIGRAM_WALK_READS.with(|count| count.set(0));
    let sub = qm
        .subscribe(search_query(&qm, chat, needle))
        .expect("subscribe");
    qm.process(&mut storage);
    let keys_read = TRIGRAM_KEYS_READ.with(|read| read.get());
    let walk_reads = TRIGRAM_WALK_READS.with(|count| count.get());

    let got: Vec<ObjectId> = qm
        .get_subscription_results(sub)
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    let want = expected(&model, chat, needle);
    assert_eq!(want.len(), 11, "137 and 1370..=1379");
    assert_eq!(got, want, "every row of the chat that holds the needle");
    assert_eq!(
        scanned_rows(&qm, sub),
        11,
        "the rows that hold \"137\" and none of \"number\" are not loaded"
    );
    // Read whole, the eight lists are 12 934 keys.
    assert_eq!(
        keys_read,
        411 + 312 + 6 * 512 + 5 * 40,
        "two lists to their ends, a round of the others, and the walks"
    );
    assert_eq!(
        walk_reads,
        2 + 4,
        "the newest rows in one list, 1370..=1379 in every list as long as the chat"
    );
}

/// Candidates side by side in a list are read in a run. Of "number 2" one list is the
/// 1 111 matches and five are the chat, 6 000 rows; after two rounds 952 matches are
/// past the 2 048 keys read of each long list, one after another in it. A walk reads
/// 40, 80 … 640 keys: five reads, 1 240 keys, and not the 3 000 rows after them.
#[test]
fn candidates_side_by_side_in_a_list_are_read_in_a_run() {
    let (mut qm, mut storage) = memory_search_manager();
    declare_indexes(&mut qm, &mut storage, wmsgs_index_declarations());
    let chat = ObjectId::new();
    let mut model = Vec::new();
    for index in 0..6000u64 {
        let row = haystack_row(chat, index);
        insert_row(&mut qm, &mut storage, &row);
        model.push(row);
    }
    qm.process(&mut storage);
    qm.take_updates();

    let needle = "number 2";
    TRIGRAM_KEYS_READ.with(|read| read.set(0));
    TRIGRAM_WALK_READS.with(|count| count.set(0));
    let sub = qm
        .subscribe(search_query(&qm, chat, needle))
        .expect("subscribe");
    qm.process(&mut storage);
    let keys_read = TRIGRAM_KEYS_READ.with(|read| read.get());
    let walk_reads = TRIGRAM_WALK_READS.with(|count| count.get());

    let got: Vec<ObjectId> = qm
        .get_subscription_results(sub)
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    let want = expected(&model, chat, needle);
    assert_eq!(want.len(), 1111, "2, 20..=29, 200..=299 and 2000..=2999");
    assert_eq!(got, want, "every row of the chat that holds the needle");
    assert_eq!(walk_reads, 5 * 5, "five reads a long list");
    // Read whole, the six lists are 31 111 keys.
    assert_eq!(
        keys_read,
        1111 + 5 * (2048 + 1240),
        "the matches' list, two rounds of the long ones, and the runs"
    );
}

/// Candidates far apart in a list cost a read each. Every fiftieth row starts with
/// "wax": the 60 of them are the candidates of "x haystack", and in each of the six
/// lists as long as the chat they stand 50 rows apart. The first round decides the
/// eleven at or before row 511; the 49 past it are a read each: of eight keys, but for
/// the first and every eighth after it, which ask for the 40 a short read costs and find
/// the next candidate no closer than it was.
#[test]
fn candidates_far_apart_in_a_list_cost_a_read_each() {
    let (mut qm, mut storage) = memory_search_manager();
    declare_indexes(&mut qm, &mut storage, wmsgs_index_declarations());
    let chat = ObjectId::new();
    let mut model = Vec::new();
    for index in 0..3000u64 {
        let mut row = haystack_row(chat, index);
        if index % 50 == 0 {
            row.body = format!("wax {}", row.body);
        }
        insert_row(&mut qm, &mut storage, &row);
        model.push(row);
    }
    qm.process(&mut storage);
    qm.take_updates();

    let needle = "x haystack";
    let trigrams = crate::query_manager::trigram_index::trigrams(&fold(needle)).len();
    assert_eq!(trigrams, 8, "the needle's trigrams");
    TRIGRAM_KEYS_READ.with(|read| read.set(0));
    TRIGRAM_WALK_READS.with(|count| count.set(0));
    let sub = qm
        .subscribe(search_query(&qm, chat, needle))
        .expect("subscribe");
    qm.process(&mut storage);
    let keys_read = TRIGRAM_KEYS_READ.with(|read| read.get());
    let walk_reads = TRIGRAM_WALK_READS.with(|count| count.get());

    let got: Vec<ObjectId> = qm
        .get_subscription_results(sub)
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    let want = expected(&model, chat, needle);
    assert_eq!(want.len(), 60);
    assert_eq!(got, want, "every row of the chat that holds the needle");
    assert_eq!(scanned_rows(&qm, sub), 60);
    assert_eq!(
        walk_reads,
        6 * 49,
        "a read a candidate past the first round"
    );
    // Read whole, the eight lists are 18 120 keys.
    assert_eq!(
        keys_read,
        2 * 60 + 6 * 512 + 6 * (7 * 40 + 42 * 8),
        "the two short lists, a round of the long ones, and the 49 reads of a walk"
    );
}

/// Candidates a few rows apart in a list are read through, not sought one by one, and
/// a round reads three times what was read. Every third row starts with "wax": the two
/// lists of "x haystack" that tell them apart are 3 000 rows and end in the third
/// round — 512, 1 536, then 6 144 keys asked for — which leaves 8 192 keys read of each
/// of the six lists as long as the chat, 9 000 rows. The 269 candidates past them
/// stand three keys apart: a walk reads 40, 80, 160 and 320 keys, each read deciding
/// a third as many candidates, and the 640 it asks for last find the list's end.
#[test]
fn candidates_a_few_rows_apart_in_a_list_are_read_through() {
    let (mut qm, mut storage) = memory_search_manager();
    declare_indexes(&mut qm, &mut storage, wmsgs_index_declarations());
    let chat = ObjectId::new();
    let mut model = Vec::new();
    for index in 0..9000u64 {
        let mut row = haystack_row(chat, index);
        if index % 3 == 0 {
            row.body = format!("wax {}", row.body);
        }
        insert_row(&mut qm, &mut storage, &row);
        model.push(row);
    }
    qm.process(&mut storage);
    qm.take_updates();

    let needle = "x haystack";
    TRIGRAM_KEYS_READ.with(|read| read.set(0));
    TRIGRAM_LIST_READS.with(|reads| reads.set(0));
    TRIGRAM_WALK_READS.with(|count| count.set(0));
    let sub = qm
        .subscribe(search_query(&qm, chat, needle))
        .expect("subscribe");
    qm.process(&mut storage);
    let keys_read = TRIGRAM_KEYS_READ.with(|read| read.get());
    let list_reads = TRIGRAM_LIST_READS.with(|reads| reads.get());
    let walk_reads = TRIGRAM_WALK_READS.with(|count| count.get());

    let got: Vec<ObjectId> = qm
        .get_subscription_results(sub)
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    let want = expected(&model, chat, needle);
    assert_eq!(want.len(), 3000);
    assert_eq!(got, want, "every row of the chat that holds the needle");
    assert_eq!(list_reads, 3 * 8, "three rounds of the eight lists");
    assert_eq!(walk_reads, 6 * 5, "five reads a long list");
    assert_eq!(
        keys_read,
        8 * 512 + 8 * 1536 + 2 * 952 + 6 * 6144 + 6 * 801,
        "three rounds, and the keys from the first candidate past them to the list's end"
    );
}

/// Candidates that draw closer than a short read costs are read through from the read
/// that reaches two of them. Every fiftieth row of the chat's first 2 000 starts with
/// "wax", and every twentieth of its last 2 000: 140 candidates of "x haystack", eleven
/// of them decided by the first round. A walk of a long list reads the 29 that stand 50
/// rows apart one by one — eight keys a read, 40 the first and every eighth — and goes
/// on so into those 20 apart, 2000, 2020 and 2040, until the read that asks for 40
/// again, from 2060, holds 2080 as well: from there the reads double, 80 … 640, and
/// the 1 280 asked for from 3300 find the 700 keys the list has left.
#[test]
fn candidates_closer_than_a_seek_are_read_through_once_a_read_reaches_two() {
    let (mut qm, mut storage) = memory_search_manager();
    declare_indexes(&mut qm, &mut storage, wmsgs_index_declarations());
    let chat = ObjectId::new();
    let mut model = Vec::new();
    for index in 0..4000u64 {
        let mut row = haystack_row(chat, index);
        if index % if index < 2000 { 50 } else { 20 } == 0 {
            row.body = format!("wax {}", row.body);
        }
        insert_row(&mut qm, &mut storage, &row);
        model.push(row);
    }
    qm.process(&mut storage);
    qm.take_updates();

    let needle = "x haystack";
    TRIGRAM_KEYS_READ.with(|read| read.set(0));
    TRIGRAM_WALK_READS.with(|count| count.set(0));
    let sub = qm
        .subscribe(search_query(&qm, chat, needle))
        .expect("subscribe");
    qm.process(&mut storage);
    let keys_read = TRIGRAM_KEYS_READ.with(|read| read.get());
    let walk_reads = TRIGRAM_WALK_READS.with(|count| count.get());

    let got: Vec<ObjectId> = qm
        .get_subscription_results(sub)
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    let want = expected(&model, chat, needle);
    assert_eq!(want.len(), 40 + 100);
    assert_eq!(got, want, "every row of the chat that holds the needle");
    assert_eq!(
        walk_reads,
        6 * (29 + 3 + 6),
        "the candidates far apart, three of those closer, and six reads through the rest"
    );
    // One by one the 100 closer candidates would be 100 reads a list, not nine.
    assert_eq!(
        keys_read,
        2 * 140 + 6 * 512 + 6 * (4 * 40 + 25 * 8 + 3 * 8 + 40 + 80 + 160 + 320 + 640 + 700),
        "the two short lists, a round of the long ones, and the walks"
    );
}

/// Candidates a seek apart are read through, not sought: reading the keys between them
/// costs less than a short read to each. Every 32nd row of 6 000 starts with "wax": 188
/// candidates of "x haystack", 16 of them decided by the first round. A walk of a long
/// list asks for 40 keys from row 512 and finds 544 in them; from there the reads double,
/// 80 … 2 560 keys, and the 5 120 asked for from 5632 find the 368 keys the list has left.
#[test]
fn candidates_a_seek_apart_are_read_through_not_sought() {
    let (mut qm, mut storage) = memory_search_manager();
    declare_indexes(&mut qm, &mut storage, wmsgs_index_declarations());
    let chat = ObjectId::new();
    let mut model = Vec::new();
    for index in 0..6000u64 {
        let mut row = haystack_row(chat, index);
        if index % 32 == 0 {
            row.body = format!("wax {}", row.body);
        }
        insert_row(&mut qm, &mut storage, &row);
        model.push(row);
    }
    qm.process(&mut storage);
    qm.take_updates();

    let needle = "x haystack";
    TRIGRAM_KEYS_READ.with(|read| read.set(0));
    TRIGRAM_WALK_READS.with(|count| count.set(0));
    let sub = qm
        .subscribe(search_query(&qm, chat, needle))
        .expect("subscribe");
    qm.process(&mut storage);
    let keys_read = TRIGRAM_KEYS_READ.with(|read| read.get());
    let walk_reads = TRIGRAM_WALK_READS.with(|count| count.get());

    let got: Vec<ObjectId> = qm
        .get_subscription_results(sub)
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    let want = expected(&model, chat, needle);
    assert_eq!(want.len(), 188);
    assert_eq!(got, want, "every row of the chat that holds the needle");
    // One by one the 172 candidates past the first round would be 172 reads a list.
    assert_eq!(walk_reads, 6 * 8, "eight reads a long list");
    assert_eq!(
        keys_read,
        2 * 188 + 6 * 512 + 6 * (40 + 80 + 160 + 320 + 640 + 1280 + 2560 + 368),
        "the two short lists, a round of the long ones, and the walks"
    );
}

/// Candidates that draw apart are sought one by one again, and the reads that ask
/// for a seek's worth count from there. Of the chat's 9 000 rows every fiftieth of the
/// first 2 000 starts with "wax", every fourth of the next 1 000, and every
/// two-hundredth of the rest: 320 candidates of "x haystack", eleven of them decided
/// by the first round. A walk of a long list reads the 29 that stand 50 rows apart one
/// by one, 40 keys the first and every eighth; the read of eight keys from 2000 holds
/// 2004 as well, and from there the reads double, 16 … 1 024 keys. The last of them,
/// from 3200, decides six candidates — fewer than short reads to them would cost — and
/// the 23 left, from 4400, are a read each again: eight keys, 40 the eighth and the
/// sixteenth.
#[test]
fn candidates_that_draw_apart_are_sought_one_by_one_again() {
    let (mut qm, mut storage) = memory_search_manager();
    declare_indexes(&mut qm, &mut storage, wmsgs_index_declarations());
    let chat = ObjectId::new();
    let mut model = Vec::new();
    for index in 0..9000u64 {
        let mut row = haystack_row(chat, index);
        let every = if index < 2000 {
            50
        } else if index < 3000 {
            4
        } else {
            200
        };
        if index % every == 0 {
            row.body = format!("wax {}", row.body);
        }
        insert_row(&mut qm, &mut storage, &row);
        model.push(row);
    }
    qm.process(&mut storage);
    qm.take_updates();

    let needle = "x haystack";
    TRIGRAM_KEYS_READ.with(|read| read.set(0));
    TRIGRAM_WALK_READS.with(|count| count.set(0));
    let sub = qm
        .subscribe(search_query(&qm, chat, needle))
        .expect("subscribe");
    qm.process(&mut storage);
    let keys_read = TRIGRAM_KEYS_READ.with(|read| read.get());
    let walk_reads = TRIGRAM_WALK_READS.with(|count| count.get());

    let got: Vec<ObjectId> = qm
        .get_subscription_results(sub)
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    let want = expected(&model, chat, needle);
    assert_eq!(want.len(), 40 + 250 + 30);
    assert_eq!(got, want, "every row of the chat that holds the needle");
    assert_eq!(
        walk_reads,
        6 * (29 + 7 + 1 + 23),
        "a read for each candidate far apart, seven through those close together, the \
         read that finds them drawn apart, and a read for each after it"
    );
    // Doubling on, a walk would read the 5 800 keys the list has left after 3200; with
    // the count of short reads not started over at 2000, three of the last 23 would ask
    // for 40 keys, not two.
    assert_eq!(
        keys_read,
        2 * 320
            + 6 * 512
            + 6 * ((4 * 40 + 25 * 8)
                + (8 + 16 + 32 + 64 + 128 + 256 + 512)
                + 1024
                + (2 * 40 + 21 * 8)),
        "the two short lists, a round of the long ones, and the walks"
    );
}

/// A walk's reads that decide one candidate each are counted, and the count starts over
/// only after a read worth its seek. Here every 72 rows hold candidates at 0, 8, 16, 24,
/// 32, 40, 48, 49 and 56: eight-key reads decide one each, but for the one at 48, which
/// holds 49 as well and doubles to sixteen keys, which decide one again. Had that read
/// started the count over, the read that asks for 40 keys would never come, and every
/// 72 rows would cost eight reads where reading through costs 72 keys: forty times
/// over, 321 reads a list. As it is, the read after the eighth that decided one asks for
/// 40 keys, finds five candidates, and the walk reads through the rest, 16 reads in all.
#[test]
fn candidates_that_keep_a_walk_short_cannot_keep_it_from_asking_again() {
    let (mut qm, mut storage) = memory_search_manager();
    declare_indexes(&mut qm, &mut storage, wmsgs_index_declarations());
    let chat = ObjectId::new();
    let mut model = Vec::new();
    for index in 0..4000u64 {
        let mut row = haystack_row(chat, index);
        let in_cycle = (600..600 + 72 * 40).contains(&index)
            && [0, 8, 16, 24, 32, 40, 48, 49, 56].contains(&((index - 600) % 72));
        if index == 520 || in_cycle {
            row.body = format!("wax {}", row.body);
        }
        insert_row(&mut qm, &mut storage, &row);
        model.push(row);
    }
    qm.process(&mut storage);
    qm.take_updates();

    let needle = "x haystack";
    TRIGRAM_KEYS_READ.with(|read| read.set(0));
    TRIGRAM_WALK_READS.with(|count| count.set(0));
    let sub = qm
        .subscribe(search_query(&qm, chat, needle))
        .expect("subscribe");
    qm.process(&mut storage);
    let keys_read = TRIGRAM_KEYS_READ.with(|read| read.get());
    let walk_reads = TRIGRAM_WALK_READS.with(|count| count.get());

    let got: Vec<ObjectId> = qm
        .get_subscription_results(sub)
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    let want = expected(&model, chat, needle);
    assert_eq!(want.len(), 1 + 9 * 40);
    assert_eq!(got, want, "every row of the chat that holds the needle");
    assert_eq!(
        walk_reads,
        6 * 16,
        "the one at 520, eight in the first cycle, and seven through the rest"
    );
    assert_eq!(
        keys_read,
        2 * 361 + 6 * 512 + 6 * (40 + 6 * 8 + 8 + 16 + 40 + 80 + 160 + 320 + 640 + 1280 + 800),
        "the two short lists, a round of the long ones, and the walks"
    );
}

/// Candidates 36 rows apart are read through: a read of twice as many keys as the one
/// before decides about twice as many candidates, and each of them saves a short read,
/// 40 keys' worth. From the read of 40 keys at row 540 the reads double, 80 … 2 560, and
/// the 5 120 asked for from 5760 find the 240 keys the list has left; the read of 320
/// decides nine candidates, worth 360 keys, and had a candidate been reckoned at a seek
/// alone, 288, it would have sent the walk back to short reads.
#[test]
fn candidates_closer_than_a_short_read_costs_are_read_through_however_far() {
    let (mut qm, mut storage) = memory_search_manager();
    declare_indexes(&mut qm, &mut storage, wmsgs_index_declarations());
    let chat = ObjectId::new();
    let mut model = Vec::new();
    for index in 0..6000u64 {
        let mut row = haystack_row(chat, index);
        if index % 36 == 0 {
            row.body = format!("wax {}", row.body);
        }
        insert_row(&mut qm, &mut storage, &row);
        model.push(row);
    }
    qm.process(&mut storage);
    qm.take_updates();

    let needle = "x haystack";
    TRIGRAM_KEYS_READ.with(|read| read.set(0));
    TRIGRAM_WALK_READS.with(|count| count.set(0));
    let sub = qm
        .subscribe(search_query(&qm, chat, needle))
        .expect("subscribe");
    qm.process(&mut storage);
    let keys_read = TRIGRAM_KEYS_READ.with(|read| read.get());
    let walk_reads = TRIGRAM_WALK_READS.with(|count| count.get());

    let got: Vec<ObjectId> = qm
        .get_subscription_results(sub)
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    let want = expected(&model, chat, needle);
    assert_eq!(want.len(), 167);
    assert_eq!(got, want, "every row of the chat that holds the needle");
    assert_eq!(walk_reads, 6 * 8, "eight reads a long list");
    assert_eq!(
        keys_read,
        2 * 167 + 6 * 512 + 6 * (40 + 80 + 160 + 320 + 640 + 1280 + 2560 + 240),
        "the two short lists, a round of the long ones, and the walks"
    );
}

/// Nor does a read of 16 keys start the count over: here every 104 rows hold candidates at
/// 0, 8, 16, 24, 32, 40, 48, 49, 56, 57 and 72. Six eight-key reads decide one each, the
/// one at 48 holds 49 and doubles to 16 keys, the one at 56 holds 57 and doubles to 32,
/// and the read of 32 keys at 72 decides one. Had the read of 16 started the count over,
/// no read of 40 keys would come, and every 104 rows would cost nine reads: 271 a list.
/// As it is, the read of 32 keys is the eighth that decided one, the read after it asks
/// for 40 keys at 704, finds five candidates, and the walk reads through the rest.
#[test]
fn candidates_that_double_a_short_read_twice_cannot_keep_a_walk_short() {
    let (mut qm, mut storage) = memory_search_manager();
    declare_indexes(&mut qm, &mut storage, wmsgs_index_declarations());
    let chat = ObjectId::new();
    let mut model = Vec::new();
    for index in 0..4000u64 {
        let mut row = haystack_row(chat, index);
        let in_cycle = (600..600 + 104 * 30).contains(&index)
            && [0, 8, 16, 24, 32, 40, 48, 49, 56, 57, 72].contains(&((index - 600) % 104));
        if index == 520 || in_cycle {
            row.body = format!("wax {}", row.body);
        }
        insert_row(&mut qm, &mut storage, &row);
        model.push(row);
    }
    qm.process(&mut storage);
    qm.take_updates();

    let needle = "x haystack";
    TRIGRAM_KEYS_READ.with(|read| read.set(0));
    TRIGRAM_WALK_READS.with(|count| count.set(0));
    let sub = qm
        .subscribe(search_query(&qm, chat, needle))
        .expect("subscribe");
    qm.process(&mut storage);
    let keys_read = TRIGRAM_KEYS_READ.with(|read| read.get());
    let walk_reads = TRIGRAM_WALK_READS.with(|count| count.get());

    let got: Vec<ObjectId> = qm
        .get_subscription_results(sub)
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    let want = expected(&model, chat, needle);
    assert_eq!(want.len(), 1 + 11 * 30);
    assert_eq!(got, want, "every row of the chat that holds the needle");
    assert_eq!(
        walk_reads,
        6 * 17,
        "the one at 520, nine in the first cycle, and seven through the rest"
    );
    assert_eq!(
        keys_read,
        2 * 331 + 6 * 512 + 6 * (40 + 6 * 8 + 8 + 16 + 32 + 40 + 80 + 160 + 320 + 640 + 1280 + 752),
        "the two short lists, a round of the long ones, and the walks"
    );
}

/// Candidates 44 rows apart are sought, not read through: a read through them decides a
/// candidate in 44 keys, and each is worth the 40 keys a short read to it costs. Here
/// every 36th row up to row 1800 holds one, then every 44th. From 540 the reads double
/// as at 36 rows apart, up to the read of 1280 keys at 1892, which decides 30 candidates,
/// worth 1 200 keys: fewer than it read, and the walk goes back to short reads, 70 in
/// all. Had a candidate been reckoned at 44 keys or more, the walk would have doubled on
/// to the list's end in 8 reads.
#[test]
fn candidates_further_apart_than_a_short_read_costs_are_sought_however_close_before() {
    let (mut qm, mut storage) = memory_search_manager();
    declare_indexes(&mut qm, &mut storage, wmsgs_index_declarations());
    let chat = ObjectId::new();
    let mut model = Vec::new();
    for index in 0..6000u64 {
        let mut row = haystack_row(chat, index);
        if (index < 1800 && index % 36 == 0) || (index >= 1800 && index % 44 == 0) {
            row.body = format!("wax {}", row.body);
        }
        insert_row(&mut qm, &mut storage, &row);
        model.push(row);
    }
    qm.process(&mut storage);
    qm.take_updates();

    let needle = "x haystack";
    TRIGRAM_KEYS_READ.with(|read| read.set(0));
    TRIGRAM_WALK_READS.with(|count| count.set(0));
    let sub = qm
        .subscribe(search_query(&qm, chat, needle))
        .expect("subscribe");
    qm.process(&mut storage);
    let keys_read = TRIGRAM_KEYS_READ.with(|read| read.get());
    let walk_reads = TRIGRAM_WALK_READS.with(|count| count.get());

    let got: Vec<ObjectId> = qm
        .get_subscription_results(sub)
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    let want = expected(&model, chat, needle);
    assert_eq!(want.len(), 146);
    assert_eq!(got, want, "every row of the chat that holds the needle");
    assert_eq!(walk_reads, 6 * 70, "seventy reads a long list");
    assert_eq!(
        keys_read,
        2 * 146 + 6 * 512 + 6 * 3264,
        "the two short lists, a round of the long ones, and the walks"
    );
}

/// Arms the walk reads `from..to` of this thread to fail until dropped.
struct FailingWalkReads;

impl FailingWalkReads {
    fn arm(from: u64, to: u64) -> Self {
        TRIGRAM_WALK_READS_THAT_FAIL.with(|armed| armed.set(Some((from, to))));
        TRIGRAM_WALK_READS.with(|count| count.set(0));
        Self
    }
}

impl Drop for FailingWalkReads {
    fn drop(&mut self) {
        TRIGRAM_WALK_READS_THAT_FAIL.with(|armed| armed.set(None));
    }
}

/// A walk read the store fails rules nothing out: the candidates not decided yet stay,
/// and the filter decides. With every walk read failing the 300 newest rows are
/// loaded, and not returned.
#[test]
fn a_walk_read_that_fails_rules_nothing_out() {
    let (mut qm, mut storage) = memory_search_manager();
    declare_indexes(&mut qm, &mut storage, wmsgs_index_declarations());
    let (chat, model) = chat_with_newest_false_candidates(&mut qm, &mut storage);

    let needle = "number 137";
    let _failing = FailingWalkReads::arm(0, u64::MAX);
    let sub = qm
        .subscribe(search_query(&qm, chat, needle))
        .expect("subscribe");
    qm.process(&mut storage);
    assert_eq!(
        TRIGRAM_WALK_READS.with(|count| count.get()),
        6,
        "one read of each list that had candidates to decide"
    );
    let got: Vec<ObjectId> = qm
        .get_subscription_results(sub)
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    assert_eq!(got, expected(&model, chat, needle));
    assert_eq!(
        scanned_rows(&qm, sub),
        311,
        "every candidate the walks could not decide is loaded"
    );
}

/// One walk read failing stops that list's walk and no other. The first long list's
/// second read — the one that would find its end and rule the 300 newest rows out —
/// fails: what its first read decided stands, the 300 stay, the next list holds them
/// (four reads), and the one after rules them out.
#[test]
fn a_walk_read_that_fails_leaves_the_rest_of_its_list_to_the_other_lists() {
    let (mut qm, mut storage) = memory_search_manager();
    declare_indexes(&mut qm, &mut storage, wmsgs_index_declarations());
    let (chat, model) = chat_with_newest_false_candidates(&mut qm, &mut storage);

    let needle = "number 137";
    let _failing = FailingWalkReads::arm(1, 2);
    let sub = qm
        .subscribe(search_query(&qm, chat, needle))
        .expect("subscribe");
    qm.process(&mut storage);
    assert_eq!(
        TRIGRAM_WALK_READS.with(|count| count.get()),
        2 + 4 + 2 + 1 + 1,
        "the list cut short, the one that holds the 300, the one that rules them out, \
         and two that have 1370..=1379 left"
    );
    let got: Vec<ObjectId> = qm
        .get_subscription_results(sub)
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    assert_eq!(got, expected(&model, chat, needle));
    assert_eq!(scanned_rows(&qm, sub), 11);
}

/// A walk read answered from before the key it was asked to start from has decided
/// nothing: the walk of that list stops, and its candidates stay. Asked again it would
/// be answered the same, for ever. The search runs on a thread of its own so that a
/// walk that does not stop fails the test instead of hanging it.
#[test]
fn a_walk_read_answered_from_before_its_start_stops_the_walk() {
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let (mut qm, mut storage) = memory_search_manager();
        declare_indexes(&mut qm, &mut storage, wmsgs_index_declarations());
        let chat = ObjectId::new();
        let mut model = Vec::new();
        for index in 0..1500u64 {
            let row = haystack_row(chat, index);
            insert_row(&mut qm, &mut storage, &row);
            model.push(row);
        }
        qm.process(&mut storage);
        qm.take_updates();

        let needle = "number 137";
        TRIGRAM_WALK_READS_IGNORE_START.with(|armed| armed.set(true));
        TRIGRAM_WALK_READS.with(|count| count.set(0));
        let sub = qm
            .subscribe(search_query(&qm, chat, needle))
            .expect("subscribe");
        qm.process(&mut storage);
        let walk_reads = TRIGRAM_WALK_READS.with(|count| count.get());
        let got: Vec<ObjectId> = qm
            .get_subscription_results(sub)
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        let _ = done_tx.send((walk_reads, got, expected(&model, chat, needle)));
    });
    let (walk_reads, got, want) = done_rx
        .recv_timeout(std::time::Duration::from_secs(120))
        .expect("the walk stops");
    assert_eq!(walk_reads, 5, "one read of each list as long as the chat");
    assert_eq!(want.len(), 11, "137 and 1370..=1379");
    assert_eq!(got, want);
}

/// A search none of whose lists can be read knows nothing of the chat's rows: it falls
/// back on all of them, for the filter to narrow. "137" is one trigram, one list; its
/// read failing used to be an empty list — no matches, for a live search until the
/// chat was next written. When it is, the list is read: the rows the fallback loaded
/// and the filter turned away are let go, and the client hears of the new match only.
#[test]
fn a_search_none_of_whose_lists_can_be_read_falls_back_on_the_chat() {
    let (mut qm, mut storage) = memory_search_manager();
    declare_indexes(&mut qm, &mut storage, wmsgs_index_declarations());
    let chat = ObjectId::new();
    let mut model = Vec::new();
    for index in 0..1500u64 {
        let row = haystack_row(chat, index);
        insert_row(&mut qm, &mut storage, &row);
        model.push(row);
    }
    qm.process(&mut storage);
    qm.take_updates();

    let needle = "137";
    let want = expected(&model, chat, needle);
    assert_eq!(want.len(), 12, "137, 1137 and 1370..=1379");
    TRIGRAM_LIST_READ_THAT_FAILS.with(|armed| armed.set(Some(0)));
    let sub = qm
        .subscribe(search_query(&qm, chat, needle))
        .expect("subscribe");
    qm.process(&mut storage);
    assert!(
        TRIGRAM_LIST_READ_THAT_FAILS.with(|armed| armed.get().is_none()),
        "the list read was reached"
    );
    let got: Vec<ObjectId> = qm
        .get_subscription_results(sub)
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    assert_eq!(got, want);
    assert_eq!(scanned_rows(&qm, sub), 1500, "the rows of the chat");
    qm.take_updates();

    let row = haystack_row(chat, 9137);
    insert_row(&mut qm, &mut storage, &row);
    model.push(row.clone());
    qm.process(&mut storage);
    let want = expected(&model, chat, needle);
    assert_eq!(want.len(), 13, "and 9137");
    let got: Vec<ObjectId> = qm
        .get_subscription_results(sub)
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    assert_eq!(got, want);
    assert_eq!(
        scanned_rows(&qm, sub),
        13,
        "the rows that hold the needle's trigram"
    );
    let updates: Vec<_> = qm
        .take_updates()
        .into_iter()
        .filter(|update| update.subscription_id == sub)
        .collect();
    let added: Vec<ObjectId> = updates
        .iter()
        .flat_map(|update| update.delta.added.iter().map(|row| row.id))
        .collect();
    let removed: usize = updates
        .iter()
        .map(|update| update.delta.removed.len())
        .sum();
    assert_eq!(added, vec![row.id], "the new match");
    assert_eq!(removed, 0, "no match was lost");
}

/// A trigram no row of the chat holds ends the search at its list: nothing matches,
/// and the lists after it are not read.
#[test]
fn a_search_for_a_trigram_no_row_holds_stops_at_its_list() {
    let (mut qm, mut storage) = memory_search_manager();
    declare_indexes(&mut qm, &mut storage, wmsgs_index_declarations());
    let chat = ObjectId::new();
    for index in 0..1500u64 {
        insert_row(&mut qm, &mut storage, &haystack_row(chat, index));
    }
    qm.process(&mut storage);
    qm.take_updates();

    // In trigram order: " nu" (every row), then "a n" (none), then six more.
    let needle = "aaa number";
    TRIGRAM_LIST_READS.with(|reads| reads.set(0));
    let sub = qm
        .subscribe(search_query(&qm, chat, needle))
        .expect("subscribe");
    qm.process(&mut storage);
    let list_reads = TRIGRAM_LIST_READS.with(|reads| reads.get());

    assert!(qm.get_subscription_results(sub).is_empty());
    assert_eq!(scanned_rows(&qm, sub), 0);
    assert_eq!(list_reads, 2, "the lists read up to the empty one");
}

/// A list that cannot be read rules nothing out. Of "number 1" every list outlasts the
/// first round; the first read of the second round fails. Taken for the list's end, it
/// left the 512 rows read of it as the whole list, and every match past them was
/// dropped — from a live search, removed.
#[test]
fn a_list_that_cannot_be_read_rules_nothing_out() {
    let (mut qm, mut storage) = memory_search_manager();
    declare_indexes(&mut qm, &mut storage, wmsgs_index_declarations());
    let chat = ObjectId::new();
    let mut model = Vec::new();
    for index in 0..1500u64 {
        let row = haystack_row(chat, index);
        insert_row(&mut qm, &mut storage, &row);
        model.push(row);
    }
    qm.process(&mut storage);
    qm.take_updates();

    let needle = "number 1";
    let want = expected(&model, chat, needle);
    assert_eq!(want.len(), 611, "1, 10..=19, 100..=199 and 1000..=1499");
    // Six lists: reads 0..=5 are the first round, 6 the first of the second, 2 one of
    // the first.
    for failing in [6u64, 2, 11] {
        TRIGRAM_LIST_READ_THAT_FAILS.with(|armed| armed.set(Some(failing)));
        let sub = qm
            .subscribe(search_query(&qm, chat, needle))
            .expect("subscribe");
        qm.process(&mut storage);
        assert!(
            TRIGRAM_LIST_READ_THAT_FAILS.with(|armed| armed.get().is_none()),
            "list read {failing} was reached"
        );
        let got: Vec<ObjectId> = qm
            .get_subscription_results(sub)
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        assert_eq!(got, want, "with list read {failing} failing");
        assert_eq!(
            scanned_rows(&qm, sub),
            611,
            "with list read {failing} failing, the rows the other lists hold"
        );
        qm.unsubscribe_with_sync(sub);
    }
}

/// A store whose limited range read costs the range gains nothing from rounds, and
/// would pay for what is left of a list at every one: there a search reads each list
/// once, whole.
#[test]
fn a_store_without_a_bounded_range_read_is_read_a_list_at_a_time() {
    let (mut qm, mut storage) = opfs_search_manager();
    declare_indexes(&mut qm, &mut storage, wmsgs_index_declarations());
    let chat = ObjectId::new();
    let mut model = Vec::new();
    for index in 0..1500u64 {
        let row = haystack_row(chat, index);
        insert_row(&mut qm, &mut storage, &row);
        model.push(row);
    }
    qm.process(&mut storage);
    qm.take_updates();

    let needle = "number 1";
    TRIGRAM_LIST_READS.with(|reads| reads.set(0));
    TRIGRAM_KEYS_READ.with(|read| read.set(0));
    TRIGRAM_WALK_READS.with(|count| count.set(0));
    let sub = qm
        .subscribe(search_query(&qm, chat, needle))
        .expect("subscribe");
    qm.process(&mut storage);
    let list_reads = TRIGRAM_LIST_READS.with(|reads| reads.get());
    let keys_read = TRIGRAM_KEYS_READ.with(|read| read.get());

    let got: Vec<ObjectId> = qm
        .get_subscription_results(sub)
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    assert_eq!(got, expected(&model, chat, needle));
    assert_eq!(list_reads, 6, "one read of each of the six lists");
    assert_eq!(keys_read, 5 * 1500 + 611, "each whole");
    assert_eq!(TRIGRAM_WALK_READS.with(|count| count.get()), 0);
}

// Few letters, so trigrams collide and posting lists overlap; 'İ' folds to two
// chars, so folded text is longer than the text; a capital sigma lowers by context
// (final or not), and both small sigmas must meet it.
const FOLDING_LETTERS: &[char] = &['a', 'b', 'A', 'B', 'c', ' ', 'İ', 'i', 'Σ', 'σ', 'ς'];
// Three letters: 27 trigrams for the whole chat, so every posting list is tens of
// rows long and a needle's lists differ in length.
const CROWDED_LETTERS: &[char] = &['a', 'b', ' '];

/// A needle's common trigrams must not make it cost the chat. Every row here holds
/// "haystack number", so five of the trigrams of "number 137" have a posting list as
/// long as the chat; the three that tell rows apart are short. Read whole, the lists
/// were 8 233 keys for 11 hits, and as many more for every 1 500 rows the chat grows by.
/// Now a round of each, and the 11 rows walked in the lists that outlast it.
#[test]
fn a_search_reads_no_further_into_a_common_trigram_than_its_rare_ones_are_worth() {
    let (mut qm, mut storage) = create_query_manager(SyncManager::new(), search_schema());
    declare_indexes(&mut qm, &mut storage, wmsgs_index_declarations());
    let chat = ObjectId::new();
    let other = ObjectId::new();
    let mut model = Vec::new();
    for index in 0..1500u64 {
        let owners: &[ObjectId] = if index < 50 { &[chat, other] } else { &[chat] };
        for owner in owners {
            let row = ModelRow {
                id: ObjectId::new(),
                chat: *owner,
                at: index * 10,
                dead: false,
                body: format!("haystack number {index}"),
            };
            insert_row(&mut qm, &mut storage, &row);
            model.push(row);
        }
    }
    qm.process(&mut storage);
    qm.take_updates();

    let needle = "Number 137";
    let trigrams = crate::query_manager::trigram_index::trigrams(&fold(needle)).len();
    assert_eq!(trigrams, 8, "the needle's trigrams");
    TRIGRAM_KEYS_READ.with(|read| read.set(0));
    TRIGRAM_WALK_READS.with(|count| count.set(0));
    let sub = qm
        .subscribe(search_query(&qm, chat, needle))
        .expect("subscribe");
    qm.process(&mut storage);
    let keys_read = TRIGRAM_KEYS_READ.with(|read| read.get());
    let walk_reads = TRIGRAM_WALK_READS.with(|count| count.get());

    let got: Vec<ObjectId> = qm
        .get_subscription_results(sub)
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    let want = expected(&model, chat, needle);
    assert_eq!(want.len(), 11, "137 and 1370..=1379");
    assert_eq!(got, want, "every row of the chat that holds the needle");
    assert_eq!(
        scanned_rows(&qm, sub),
        11,
        "the short lists alone leave the 11 rows that hold \"137\""
    );
    // One round: 512 keys of each of the eight lists at most.
    assert!(
        keys_read <= 8 * 512,
        "a search with 11 hits in a chat of 1 500 rows read {keys_read} index keys"
    );
    assert!(
        walk_reads <= 2 * 6,
        "and walked the six lists that outlast the round in {walk_reads} reads"
    );
}

#[test]
fn search_matches_the_model_through_folding_scope_moves_and_writes() {
    search_matches_the_model(memory_search_manager, FOLDING_LETTERS, 16);
}

#[test]
fn search_matches_the_model_when_every_list_is_long() {
    search_matches_the_model(memory_search_manager, CROWDED_LETTERS, 16);
}

/// A list read or a walk read fails before every settle, at random: what a search returns
/// and what it sends stay the model's, and every candidate is loaded.
#[test]
fn search_matches_the_model_with_reads_that_fail() {
    let failed = with_reads_that_fail(|| {
        with_first_round(1, || {
            search_matches_the_model(memory_search_manager, CROWDED_LETTERS, 16)
        });
        with_first_round(2, || {
            search_matches_the_model(memory_search_manager, FOLDING_LETTERS, 16)
        });
        with_first_round(1, || {
            search_matches_the_model(sqlite_search_manager, CROWDED_LETTERS, 6)
        });
        #[cfg(feature = "rocksdb")]
        with_first_round(1, || {
            search_matches_the_model(rocksdb_search_manager, CROWDED_LETTERS, 6)
        });
    });
    assert!(
        failed.list_reads >= 100 && failed.walk_reads >= 40,
        "list reads failed {} times, walk reads {}",
        failed.list_reads,
        failed.walk_reads
    );
}

/// Rounds of 1, 4, 16 … keys: there are candidates before any list has read far, the
/// lists of a needle end in different rounds, and the ones still open are walked.
#[test]
fn search_matches_the_model_read_a_key_at_a_time() {
    with_first_round(1, || {
        search_matches_the_model(memory_search_manager, FOLDING_LETTERS, 16)
    });
    with_first_round(1, || {
        search_matches_the_model(memory_search_manager, CROWDED_LETTERS, 16)
    });
}

/// Rounds of 2, 8, 32 … and of 3, 12, 48 … keys, each list read on from where the
/// round before left it until one ends.
#[test]
fn search_matches_the_model_read_through_every_round() {
    with_first_round(2, || {
        search_matches_the_model(memory_search_manager, CROWDED_LETTERS, 16)
    });
    with_first_round(3, || {
        search_matches_the_model(memory_search_manager, CROWDED_LETTERS, 16)
    });
}

/// The same rounds on the store a client reads: a round's start key and limit, and a
/// walk's, are answered by SQLite's own range read.
#[test]
fn search_matches_the_model_in_rounds_of_a_few_keys_on_sqlite() {
    with_first_round(1, || {
        search_matches_the_model(sqlite_search_manager, FOLDING_LETTERS, 6)
    });
    for keys in [1, 2, 3] {
        with_first_round(keys, || {
            search_matches_the_model(sqlite_search_manager, CROWDED_LETTERS, 6)
        });
    }
}

/// And on the store a server reads.
#[cfg(feature = "rocksdb")]
#[test]
fn search_matches_the_model_in_rounds_of_a_few_keys_on_rocksdb() {
    with_first_round(1, || {
        search_matches_the_model(rocksdb_search_manager, FOLDING_LETTERS, 6)
    });
    for keys in [1, 2, 3] {
        with_first_round(keys, || {
            search_matches_the_model(rocksdb_search_manager, CROWDED_LETTERS, 6)
        });
    }
}

thread_local! {
    /// Whether the model of this thread arms a read to fail before every settle.
    static MODEL_READS_FAIL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// The armed reads that failed in the model runs of this thread: list reads, walk reads.
    static MODEL_READS_FAILED: std::cell::Cell<(u64, u64)> = const { std::cell::Cell::new((0, 0)) };
}

/// The armed reads that failed.
struct ReadsFailed {
    list_reads: u64,
    walk_reads: u64,
}

/// Runs `body` with a list read or a walk read, picked at random, armed to fail before
/// every settle of the model.
fn with_reads_that_fail(body: impl FnOnce()) -> ReadsFailed {
    MODEL_READS_FAIL.with(|fail| fail.set(true));
    MODEL_READS_FAILED.with(|failed| failed.set((0, 0)));
    body();
    MODEL_READS_FAIL.with(|fail| fail.set(false));
    let (list_reads, walk_reads) = MODEL_READS_FAILED.with(|failed| failed.get());
    ReadsFailed {
        list_reads,
        walk_reads,
    }
}

/// Arms the read a settle of the model fails, if the model's reads fail. A walk is rarer
/// in the model than a list read, and its reads are armed twice as often.
struct ArmedRead {
    walk_read: Option<u64>,
}

impl ArmedRead {
    fn arm(rng: &mut Lcg) -> Self {
        if !MODEL_READS_FAIL.with(|fail| fail.get()) {
            return Self { walk_read: None };
        }
        if rng.below(3) == 0 {
            TRIGRAM_LIST_READ_THAT_FAILS.with(|armed| armed.set(Some(rng.below(12))));
            Self { walk_read: None }
        } else {
            let at = rng.below(3);
            TRIGRAM_WALK_READS.with(|count| count.set(0));
            TRIGRAM_WALK_READS_THAT_FAIL.with(|armed| armed.set(Some((at, at + 1))));
            Self {
                walk_read: Some(at),
            }
        }
    }
}

impl Drop for ArmedRead {
    fn drop(&mut self) {
        let list_read_failed = TRIGRAM_LIST_READ_THAT_FAILS
            .with(|armed| armed.take())
            .is_none();
        TRIGRAM_WALK_READS_THAT_FAIL.with(|armed| armed.set(None));
        let walk_read_failed = self
            .walk_read
            .is_some_and(|at| TRIGRAM_WALK_READS.with(|count| count.get()) > at);
        let armed = MODEL_READS_FAIL.with(|fail| fail.get());
        MODEL_READS_FAILED.with(|count| {
            let (lists, walks) = count.get();
            match self.walk_read {
                Some(_) => count.set((lists, walks + u64::from(walk_read_failed))),
                None => count.set((lists + u64::from(armed && list_read_failed), walks)),
            }
        });
    }
}

fn search_matches_the_model<H: Storage>(
    open: fn() -> (QueryManager, H),
    letters: &[char],
    seeds: u64,
) {
    #[allow(non_snake_case)]
    let LETTERS = letters;
    let word = |rng: &mut Lcg, len: u64| -> String {
        (0..len)
            .map(|_| LETTERS[rng.below(LETTERS.len() as u64) as usize])
            .collect()
    };
    for seed in 1..=seeds {
        let mut rng = Lcg(seed);
        let (mut qm, mut storage) = open();
        declare_indexes(&mut qm, &mut storage, wmsgs_index_declarations());
        let chat = ObjectId::new();
        let other = ObjectId::new();
        let mut model: Vec<ModelRow> = Vec::new();
        for index in 0..(60 + rng.below(60)) {
            let len = rng.below(12);
            let row = ModelRow {
                id: ObjectId::new(),
                chat: if rng.below(3) == 0 { other } else { chat },
                at: index * 10,
                dead: rng.below(5) == 0,
                body: word(&mut rng, len),
            };
            insert_row(&mut qm, &mut storage, &row);
            model.push(row);
        }
        qm.process(&mut storage);
        qm.take_updates();

        // Needles of 1..=5 chars: the short ones have no trigram and take the scan.
        let needles: Vec<String> = (0..5)
            .map(|_| {
                let len = 1 + rng.below(5);
                word(&mut rng, len)
            })
            .collect();
        let subs: Vec<_> = needles
            .iter()
            .map(|needle| {
                qm.subscribe(search_query(&qm, chat, needle))
                    .expect("subscribe")
            })
            .collect();
        let armed = ArmedRead::arm(&mut rng);
        qm.process(&mut storage);
        drop(armed);
        let mut delivered: Vec<HashSet<ObjectId>> = vec![HashSet::new(); subs.len()];
        let absorb = |qm: &mut QueryManager, delivered: &mut Vec<HashSet<ObjectId>>| {
            for update in qm.take_updates() {
                let Some(slot) = subs.iter().position(|sub| *sub == update.subscription_id) else {
                    continue;
                };
                for row in &update.delta.removed {
                    assert!(
                        delivered[slot].remove(&row.id),
                        "seed {seed}: removed a row the client never had"
                    );
                }
                for row in &update.delta.added {
                    assert!(
                        delivered[slot].insert(row.id),
                        "seed {seed}: added a row the client already had"
                    );
                }
            }
        };
        absorb(&mut qm, &mut delivered);

        for step in 0..40 {
            for (slot, (sub, needle)) in subs.iter().zip(needles.iter()).enumerate() {
                let got: Vec<ObjectId> = qm
                    .get_subscription_results(*sub)
                    .into_iter()
                    .map(|(id, _)| id)
                    .collect();
                let want = expected(&model, chat, needle);
                assert_eq!(got, want, "seed {seed}, step {step}, needle {needle:?}");
                // The rows the search loads are the rows of the chat that hold every
                // trigram of the needle: what reading every list whole would leave.
                let needle_trigrams = crate::query_manager::trigram_index::trigrams(&fold(needle));
                if !needle_trigrams.is_empty() {
                    let mut scanned = Vec::new();
                    qm.subscriptions
                        .get(sub)
                        .expect("the subscription")
                        .graph
                        .collect_scanned_row_ids(&mut scanned);
                    let scanned: HashSet<ObjectId> = scanned.into_iter().collect();
                    let candidates: HashSet<ObjectId> = model
                        .iter()
                        .filter(|row| {
                            row.chat == chat
                                && needle_trigrams.is_subset(
                                    &crate::query_manager::trigram_index::trigrams(&fold(
                                        &row.body,
                                    )),
                                )
                        })
                        .map(|row| row.id)
                        .collect();
                    // A read that failed rules nothing out: then the rows loaded may be
                    // more than the candidates, never fewer.
                    if MODEL_READS_FAIL.with(|fail| fail.get()) {
                        assert!(
                            scanned.is_superset(&candidates),
                            "seed {seed}, step {step}, needle {needle:?}: a candidate not loaded"
                        );
                    } else {
                        assert_eq!(
                            scanned, candidates,
                            "seed {seed}, step {step}, needle {needle:?}: the candidates"
                        );
                    }
                }
                // Independent of how text folds: a row holding the needle as typed is found.
                for row in model.iter().filter(|row| row.chat == chat && !row.dead) {
                    assert!(
                        !row.body.contains(needle.as_str()) || got.contains(&row.id),
                        "seed {seed}, step {step}: {:?} holds {needle:?} as typed, and was not found",
                        row.body
                    );
                }
                assert_eq!(
                    delivered[slot],
                    want.iter().copied().collect::<HashSet<_>>(),
                    "seed {seed}, step {step}, needle {needle:?}: the deltas disagree with the result"
                );
            }

            // One write: a new row, a new text, a flag flipped, or a move to the other chat.
            let choice = rng.below(4);
            if model.is_empty() || choice == 0 {
                let len = rng.below(12);
                let row = ModelRow {
                    id: ObjectId::new(),
                    chat: if rng.below(4) == 0 { other } else { chat },
                    at: rng.below(2000),
                    dead: false,
                    body: word(&mut rng, len),
                };
                insert_row(&mut qm, &mut storage, &row);
                model.push(row);
            } else {
                let index = rng.below(model.len() as u64) as usize;
                match choice {
                    1 => {
                        let len = rng.below(12);
                        model[index].body = word(&mut rng, len);
                    }
                    2 => model[index].dead = !model[index].dead,
                    _ => {
                        model[index].chat = if model[index].chat == chat {
                            other
                        } else {
                            chat
                        };
                    }
                }
                let row = model[index].clone();
                qm.update(&mut storage, row.id, &values(&row))
                    .expect("update wmsgs row");
            }
            let armed = ArmedRead::arm(&mut rng);
            qm.process(&mut storage);
            drop(armed);
            absorb(&mut qm, &mut delivered);
        }
    }
}

/// A needle found as typed is found folded. `str::to_lowercase` lowers a capital sigma
/// by its context — `ς` at the end of a word, `σ` elsewhere — so "Σ" alone folded to
/// "σ" while the "Σ" ending "ΟΣ" folded to "ς", and the row went unfound. Both the short
/// needle (a scan) and the long one (the trigram index) are pinned, and a typed final
/// `ς` meets a capital `Σ`.
#[test]
fn search_finds_a_needle_wherever_it_stands_in_a_word() {
    let (mut qm, mut storage) = create_query_manager(SyncManager::new(), search_schema());
    declare_indexes(&mut qm, &mut storage, wmsgs_index_declarations());
    let chat = ObjectId::new();
    let mut model = Vec::new();
    for (index, body) in ["ΟΣ", "ΛΟΓΟΣΑ", "λογος", "ΟΔΟΣ ΚΑΙ", "plain"]
        .iter()
        .enumerate()
    {
        let row = ModelRow {
            id: ObjectId::new(),
            chat,
            at: index as u64 * 10,
            dead: false,
            body: body.to_string(),
        };
        insert_row(&mut qm, &mut storage, &row);
        model.push(row);
    }
    qm.process(&mut storage);
    qm.take_updates();

    let found = |qm: &mut QueryManager, storage: &mut MemoryStorage, needle: &str| {
        let sub = qm
            .subscribe(search_query(qm, chat, needle))
            .expect("subscribe");
        qm.process(storage);
        let bodies: HashSet<String> = qm
            .get_subscription_results(sub)
            .into_iter()
            .map(|(id, _)| {
                model
                    .iter()
                    .find(|row| row.id == id)
                    .map(|row| row.body.clone())
                    .expect("a row of the fixture")
            })
            .collect();
        qm.unsubscribe_with_sync(sub);
        bodies
    };
    let bodies = |list: &[&str]| {
        list.iter()
            .map(|body| body.to_string())
            .collect::<HashSet<_>>()
    };

    assert_eq!(
        found(&mut qm, &mut storage, "Σ"),
        bodies(&["ΟΣ", "ΛΟΓΟΣΑ", "λογος", "ΟΔΟΣ ΚΑΙ"]),
        "a lone capital sigma, as typed in every Greek row"
    );
    assert_eq!(
        found(&mut qm, &mut storage, "ΓΟΣ"),
        bodies(&["ΛΟΓΟΣΑ", "λογος"]),
        "a trigram needle ending in a capital sigma, as typed inside a word"
    );
    assert_eq!(
        found(&mut qm, &mut storage, "ος"),
        bodies(&["ΟΣ", "ΛΟΓΟΣΑ", "λογος", "ΟΔΟΣ ΚΑΙ"]),
        "a typed final sigma meets every capital one"
    );
}

/// Stand, not a gate: what a search costs when its lists are walked against reading
/// every one of them whole, as before the rounds, on the stores a client and a server
/// read. Half of a 40 000-row chat is older than any candidate; in the newer half every
/// `gap`-th row is one. "x haystackzap" holds none of them — four of its long lists
/// hold every candidate and the fifth rules them all out, so the time is index reads
/// alone; "x haystack" holds them all, so the time is index reads and the loads.
///
/// `cargo test --release -p jazz-tools --features test --lib -- --ignored --nocapture
/// stand_walk_against_whole_lists`
#[test]
#[ignore = "stand: prints timings"]
fn stand_walk_against_whole_lists() {
    stand_walk(sqlite_search_manager, "sqlite");
    #[cfg(feature = "rocksdb")]
    stand_walk(rocksdb_search_manager, "rocksdb");
}

fn stand_walk<H: Storage>(open: fn() -> (QueryManager, H), store: &str) {
    const ROWS: u64 = 40_000;
    // "tail": a candidate every `gap` rows of the chat's newer half; "spread": every `gap`
    // rows of the whole chat; "random": one row in `gap`, at random, over the whole chat.
    let shapes: [(&str, &[u64]); 3] = [
        ("tail", &[1, 3, 8, 12, 20, 33, 50, 200]),
        ("spread", &[24, 32, 36, 40, 48]),
        ("random", &[16, 32, 40, 48, 64]),
    ];
    for (shape, gaps) in shapes {
        for &gap in gaps {
            let mut rng = Lcg(gap);
            let (mut qm, mut storage) = open();
            declare_indexes(&mut qm, &mut storage, wmsgs_index_declarations());
            let chat = ObjectId::new();
            let mut candidates = 0usize;
            for index in 0..ROWS {
                let mut row = haystack_row(chat, index);
                let candidate = match shape {
                    "tail" => index >= ROWS / 2 && index % gap == 0,
                    "spread" => index % gap == 0,
                    _ => rng.below(gap) == 0,
                };
                row.body = if candidate {
                    candidates += 1;
                    format!("wax haystackzip {index}")
                } else {
                    format!("haystackzap {index}")
                };
                insert_row(&mut qm, &mut storage, &row);
                if index % 2_000 == 0 {
                    qm.process(&mut storage);
                    qm.take_updates();
                }
            }
            qm.process(&mut storage);
            qm.take_updates();
            storage.flush().expect("flush");

            for (needle, hits) in [("x haystackzap", 0), ("x haystack", candidates)] {
                for (way, first_round) in [("walk ", None), ("whole", Some(usize::MAX))] {
                    let mut best = std::time::Duration::MAX;
                    let mut keys_read = 0;
                    let mut reads = 0;
                    for _ in 0..5 {
                        TRIGRAM_KEYS_READ.with(|read| read.set(0));
                        TRIGRAM_LIST_READS.with(|reads| reads.set(0));
                        TRIGRAM_WALK_READS.with(|count| count.set(0));
                        let started = std::time::Instant::now();
                        let mut search = || {
                            let sub = qm
                                .subscribe(search_query(&qm, chat, needle))
                                .expect("subscribe");
                            qm.process(&mut storage);
                            sub
                        };
                        let sub = match first_round {
                            Some(keys) => with_first_round(keys, search),
                            None => search(),
                        };
                        best = best.min(started.elapsed());
                        assert_eq!(qm.get_subscription_results(sub).len(), hits);
                        assert_eq!(scanned_rows(&qm, sub), hits);
                        keys_read = TRIGRAM_KEYS_READ.with(|read| read.get());
                        reads = TRIGRAM_LIST_READS.with(|reads| reads.get())
                            + TRIGRAM_WALK_READS.with(|count| count.get());
                        qm.unsubscribe_with_sync(sub);
                        qm.process(&mut storage);
                        qm.take_updates();
                    }
                    println!(
                        "[stand] {store:8} {shape:6} gap {gap:>3} candidates {candidates:>5} {needle:14} {way}: \
                     {:>8.2} ms, {keys_read:>6} keys in {reads:>5} reads",
                        best.as_secs_f64() * 1e3
                    );
                }
            }
            drop(storage);
            drop(qm);
            remove_rocksdb_stores();
        }
    }
}
