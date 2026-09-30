//! Differential oracle for declared indexes over a store that already holds rows.
//!
//! Random writes — inserts, text edits, scope moves, time moves, soft and hard deletes —
//! run against a model of the rows, interleaved with declarations that come, go and
//! come back, fill steps granted or withheld, and restarts over the same store. At
//! every pass a window and a search in each chat answer exactly what the model says,
//! the deltas agree with the results, and every index the store's record calls
//! complete holds exactly the entries of the live rows — none missing, none stale.
//! An index the store no longer maintains is gone once its retired clear ends.

use std::collections::HashSet;

use super::*;
use crate::query_manager::composite_index::composite_value;
use crate::query_manager::declared_index::load_record;
use crate::query_manager::graph_nodes::output::QuerySubscriptionId;
use crate::query_manager::index_declarations::{IndexDeclarations, IndexPhase};
use crate::query_manager::trigram_index::{entry_value, fold, trigrams};
use crate::storage::SqliteStorage;

const PAGE: usize = 12;
const COMPOSITE: &str = "chat+at";
const TRIGRAM: &str = "chat>body";
const WORDS: &[&str] = &[
    "hay", "stack", "needle", "NEEDLE", "Needles", "nee", "dle", "neeDLE", "straw", "ΟΣ",
];

fn schema() -> Schema {
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

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

#[derive(Clone)]
struct ModelRow {
    id: ObjectId,
    chat: ObjectId,
    at: u64,
    body: String,
    live: bool,
}

fn values(row: &ModelRow) -> [Value; 4] {
    [
        Value::Uuid(row.chat),
        Value::Timestamp(row.at),
        Value::Boolean(false),
        Value::Text(row.body.clone()),
    ]
}

fn insert<H: Storage>(qm: &mut QueryManager, storage: &mut H, branch: &str, row: &ModelRow) {
    let schema = qm.schema_context().current_schema.clone();
    qm.insert_on_branch_with_schema_and_write_context_and_id(
        storage,
        "wmsgs",
        branch,
        &values(row),
        Some(row.id),
        &schema,
        None,
        true,
    )
    .expect("insert");
}

fn body(rng: &mut Lcg) -> String {
    (0..1 + rng.below(5))
        .map(|_| WORDS[rng.below(WORDS.len() as u64) as usize])
        .collect::<Vec<_>>()
        .join(" ")
}

fn declarations(choice: u64) -> IndexDeclarations {
    let composite = || {
        IndexDeclarations::empty()
            .with_composite("wmsgs", "chat", "at")
            .expect("composite")
    };
    match choice {
        0 => IndexDeclarations::empty(),
        1 => composite(),
        2 => IndexDeclarations::empty()
            .with_trigram("wmsgs", "chat", "body")
            .expect("trigram"),
        _ => composite()
            .with_trigram("wmsgs", "chat", "body")
            .expect("trigram"),
    }
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
        .filter_contains("body", Value::Text("needle".to_string()))
        .order_by_desc("at")
        .build()
}

/// Newest first, ties by id ascending, as the Sort node breaks them.
fn ranked<'a>(rows: impl Iterator<Item = &'a ModelRow>) -> Vec<ObjectId> {
    let mut rows: Vec<&ModelRow> = rows.collect();
    rows.sort_by(|left, right| right.at.cmp(&left.at).then(left.id.cmp(&right.id)));
    rows.into_iter().map(|row| row.id).collect()
}

fn expected_page(model: &[ModelRow], chat: ObjectId) -> Vec<ObjectId> {
    let mut page = ranked(model.iter().filter(|row| row.live && row.chat == chat));
    page.truncate(PAGE);
    page
}

fn expected_search(model: &[ModelRow], chat: ObjectId) -> Vec<ObjectId> {
    let needle = fold("needle");
    ranked(
        model
            .iter()
            .filter(|row| row.live && row.chat == chat && fold(&row.body).contains(&needle)),
    )
}

/// The entry values a live row owns in `index`.
fn row_values(row: &ModelRow, index: &str) -> Vec<Value> {
    let scope = Value::Uuid(row.chat);
    if index == COMPOSITE {
        composite_value(&scope, &Value::Timestamp(row.at))
            .into_iter()
            .collect()
    } else {
        trigrams(&fold(&row.body))
            .iter()
            .filter_map(|trigram| entry_value(&scope, trigram))
            .collect()
    }
}

fn family_len<H: Storage>(storage: &H, index: &str) -> usize {
    storage
        .raw_table_family_keys(&format!("idx:wmsgs:{index}:"), None, 10_000_000)
        .expect("family keys")
        .len()
}

/// Every complete index holds exactly its live rows' entries; an index the store has
/// given up and finished clearing holds nothing. Returns the phases seen.
/// `each_row` also looks every expected entry up; without it only the counts are
/// compared, which a missing entry and a stale one could cancel.
fn check_entries<H: Storage>(
    storage: &H,
    branch: &str,
    model: &[ModelRow],
    each_row: bool,
    when: &str,
) -> Vec<(&'static str, IndexPhase)> {
    let record = load_record(storage).expect("record");
    let mut phases = Vec::new();
    for index in [COMPOSITE, TRIGRAM] {
        let key = ("wmsgs".to_string(), index.to_string());
        match record.state("wmsgs", index) {
            Some(state) => {
                phases.push((index, state.phase.clone()));
                if state.phase != IndexPhase::Complete {
                    continue;
                }
                let mut want = 0;
                for row in model.iter().filter(|row| row.live) {
                    for value in row_values(row, index) {
                        want += 1;
                        if each_row {
                            let ids = storage.index_lookup("wmsgs", index, branch, &value);
                            assert!(
                                ids.contains(&row.id),
                                "{when}: complete {index} misses an entry of a live row"
                            );
                        }
                    }
                }
                assert_eq!(
                    family_len(storage, index),
                    want,
                    "{when}: complete {index} holds entries of no live row"
                );
            }
            None if !record.retired.contains_key(&key) => {
                assert_eq!(
                    family_len(storage, index),
                    0,
                    "{when}: {index} was given up and cleared, yet holds entries"
                );
            }
            None => {}
        }
    }
    phases
}

struct Subscriptions {
    ids: Vec<(QuerySubscriptionId, ObjectId, bool)>,
    delivered: Vec<HashSet<ObjectId>>,
}

fn subscribe_all(qm: &mut QueryManager, chats: &[ObjectId]) -> Subscriptions {
    let mut ids = Vec::new();
    for chat in chats {
        let page = qm.subscribe(page_query(qm, *chat)).expect("page");
        let search = qm.subscribe(search_query(qm, *chat)).expect("search");
        ids.push((page, *chat, false));
        ids.push((search, *chat, true));
    }
    let delivered = vec![HashSet::new(); ids.len()];
    Subscriptions { ids, delivered }
}

fn absorb(qm: &mut QueryManager, subs: &mut Subscriptions, when: &str) {
    for update in qm.take_updates() {
        let Some(slot) = subs
            .ids
            .iter()
            .position(|(sub, _, _)| *sub == update.subscription_id)
        else {
            continue;
        };
        for row in &update.delta.removed {
            assert!(
                subs.delivered[slot].remove(&row.id),
                "{when}: removed a row the client never had"
            );
        }
        for row in &update.delta.added {
            assert!(
                subs.delivered[slot].insert(row.id),
                "{when}: added a row the client already had"
            );
        }
    }
}

fn check_results(qm: &QueryManager, subs: &Subscriptions, model: &[ModelRow], when: &str) {
    for (slot, (sub, chat, search)) in subs.ids.iter().enumerate() {
        let got: Vec<ObjectId> = qm
            .get_subscription_results(*sub)
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        let (kind, want) = if *search {
            ("search", expected_search(model, *chat))
        } else {
            ("page", expected_page(model, *chat))
        };
        assert_eq!(got, want, "{when}: the {kind}");
        assert_eq!(
            subs.delivered[slot],
            want.iter().copied().collect::<HashSet<_>>(),
            "{when}: the {kind}'s deltas disagree with its result"
        );
    }
}

fn open_manager<H: Storage>(storage: &mut H, paced: bool) -> QueryManager {
    let mut qm = QueryManager::new(SyncManager::new());
    qm.set_current_schema(schema(), "dev", "main");
    qm.open_declared_indexes(storage);
    if paced {
        qm.pace_declared_index_steps();
    }
    qm
}

#[derive(Default)]
struct Coverage {
    clearing: usize,
    filling: usize,
    complete: usize,
    retired_cleared: usize,
    restarts: usize,
}

fn run<H: Storage>(seed: u64, mut storage: H, coverage: &mut Coverage) {
    let mut rng = Lcg(seed);
    let paced = seed % 2 == 1;
    let mut qm = open_manager(&mut storage, paced);
    let branch = get_branch(&qm);
    let chats = [ObjectId::new(), ObjectId::new()];
    let elsewhere = ObjectId::new();
    let pick_chat = |rng: &mut Lcg| match rng.below(5) {
        0 => elsewhere,
        1 | 2 => chats[0],
        _ => chats[1],
    };
    let mut model: Vec<ModelRow> = Vec::new();
    let mut next_at = 0u64;

    // History written before any declaration: several fill pages of it.
    for _ in 0..300 + rng.below(200) {
        next_at += 1 + rng.below(3);
        let row = ModelRow {
            id: ObjectId::new(),
            chat: pick_chat(&mut rng),
            at: next_at * 10,
            body: body(&mut rng),
            live: true,
        };
        insert(&mut qm, &mut storage, &branch, &row);
        model.push(row);
    }
    qm.process(&mut storage);
    let mut subs = subscribe_all(&mut qm, &chats);
    qm.process(&mut storage);
    absorb(&mut qm, &mut subs, &format!("seed {seed}, start"));
    qm.propose_index_declarations(declarations(3));

    let mut last_retired = false;
    let completed = |storage: &H| {
        let record = load_record(storage).expect("record");
        [COMPOSITE, TRIGRAM].map(|index| record.complete_incarnation("wmsgs", index))
    };
    let mut last_completed = completed(&storage);
    for step in 0..160 {
        let when = format!("seed {seed}, step {step}");
        let live: Vec<usize> = (0..model.len()).filter(|&i| model[i].live).collect();
        match rng.below(100) {
            0..30 => {
                // A new row, mostly newest, sometimes back in time (ties included).
                let at = if rng.below(4) == 0 {
                    rng.below(next_at + 1) * 10
                } else {
                    next_at += 1;
                    next_at * 10
                };
                let row = ModelRow {
                    id: ObjectId::new(),
                    chat: pick_chat(&mut rng),
                    at,
                    body: body(&mut rng),
                    live: true,
                };
                insert(&mut qm, &mut storage, &branch, &row);
                model.push(row);
            }
            30..62 if !live.is_empty() => {
                let index = live[rng.below(live.len() as u64) as usize];
                match rng.below(3) {
                    0 => model[index].body = body(&mut rng),
                    1 => model[index].chat = pick_chat(&mut rng),
                    _ => model[index].at = rng.below(next_at + 1) * 10,
                }
                let row = model[index].clone();
                qm.update(&mut storage, row.id, &values(&row))
                    .expect("update");
            }
            62..72 if !live.is_empty() => {
                let index = live[rng.below(live.len() as u64) as usize];
                model[index].live = false;
                if rng.below(3) == 0 {
                    qm.hard_delete(&mut storage, model[index].id)
                        .expect("hard delete");
                } else {
                    qm.delete(&mut storage, model[index].id).expect("delete");
                }
            }
            72..80 => {
                // Declarations change: give one or both up, add one, bring one back.
                qm.propose_index_declarations(declarations(rng.below(4)));
            }
            80..83 => {
                // A restart over the same store: the record is all that survives.
                coverage.restarts += 1;
                qm = open_manager(&mut storage, paced);
                subs = subscribe_all(&mut qm, &chats);
            }
            _ => {}
        }
        if paced && rng.below(3) != 0 {
            qm.grant_declared_index_step();
        }
        qm.process(&mut storage);
        absorb(&mut qm, &mut subs, &when);
        check_results(&qm, &subs, &model, &when);
        let each_row = step % 10 == 9 || completed(&storage) != last_completed;
        last_completed = completed(&storage);
        let phases = check_entries(&storage, &branch, &model, each_row, &when);
        for (_, phase) in &phases {
            match phase {
                IndexPhase::Clearing { .. } => coverage.clearing += 1,
                IndexPhase::Filling { .. } => coverage.filling += 1,
                IndexPhase::Complete => coverage.complete += 1,
            }
        }
        let retired = !load_record(&storage).expect("record").retired.is_empty();
        if last_retired && !retired {
            coverage.retired_cleared += 1;
        }
        last_retired = retired;
    }

    // Whatever work the last declarations left runs to the end — a retired clear
    // included — with the answers exact on the way; then every index comes back and
    // fills from scratch.
    for (phase, proposal) in [("settle", None), ("redeclare", Some(declarations(3)))] {
        if let Some(proposal) = proposal {
            qm.propose_index_declarations(proposal);
        }
        let mut steps = 0;
        loop {
            qm.grant_declared_index_step();
            qm.process(&mut storage);
            let when = format!("seed {seed}, {phase} step {steps}");
            absorb(&mut qm, &mut subs, &when);
            check_results(&qm, &subs, &model, &when);
            check_entries(&storage, &branch, &model, true, &when);
            if !qm.has_declared_index_work() {
                break;
            }
            steps += 1;
            assert!(steps < 200, "seed {seed}: the {phase} work never ended");
        }
        let record = load_record(&storage).expect("record");
        assert!(
            record.retired.is_empty(),
            "seed {seed}: {phase}: retired indexes left: {record:?}"
        );
    }
    let record = load_record(&storage).expect("record");
    for index in [COMPOSITE, TRIGRAM] {
        assert!(
            record.complete_incarnation("wmsgs", index).is_some(),
            "seed {seed}: {index} is complete at the end: {record:?}"
        );
    }
}

fn assert_covered(coverage: &Coverage) {
    assert!(
        coverage.clearing > 0
            && coverage.filling > 0
            && coverage.complete > 0
            && coverage.retired_cleared > 0
            && coverage.restarts > 0,
        "the seeds never reached every phase: clearing {}, filling {}, complete {}, \
         retired cleared {}, restarts {}",
        coverage.clearing,
        coverage.filling,
        coverage.complete,
        coverage.retired_cleared,
        coverage.restarts
    );
}

#[test]
fn declared_indexes_match_the_model_through_writes_declarations_and_restarts_in_memory() {
    let mut coverage = Coverage::default();
    for seed in 1..=8u64 {
        let mut qm = QueryManager::new(SyncManager::new());
        qm.set_current_schema(schema(), "dev", "main");
        let storage = seeded_memory_storage(&qm.schema_context().current_schema);
        run(seed, storage, &mut coverage);
    }
    assert_covered(&coverage);
}

#[test]
fn declared_indexes_match_the_model_through_writes_declarations_and_restarts_on_sqlite() {
    let mut coverage = Coverage::default();
    for seed in 101..=104u64 {
        let mut qm = QueryManager::new(SyncManager::new());
        qm.set_current_schema(schema(), "dev", "main");
        let mut storage = SqliteStorage::open(":memory:").expect("in-memory sqlite storage");
        crate::test_support::persist_test_schema(&mut storage, &qm.schema_context().current_schema);
        run(seed, storage, &mut coverage);
    }
    assert_covered(&coverage);
}
