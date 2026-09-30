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
//! Each test declares `wmsgs`'s `chat>body` trigram index on its store first
//! (`declare_indexes`).

use super::*;
use crate::query_manager::trigram_index::fold;
use std::collections::HashSet;

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

fn values(row: &ModelRow) -> Vec<Value> {
    vec![
        Value::Uuid(row.chat),
        Value::Timestamp(row.at),
        Value::Boolean(row.dead),
        Value::Text(row.body.clone()),
    ]
}

fn insert_row(qm: &mut QueryManager, storage: &mut MemoryStorage, row: &ModelRow) {
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

#[test]
fn search_matches_the_model_through_folding_scope_moves_and_writes() {
    // Few letters, so trigrams collide and posting lists overlap; 'İ' folds to two
    // chars, so folded text is longer than the text; a capital sigma lowers by context
    // (final or not), and both small sigmas must meet it.
    const LETTERS: &[char] = &['a', 'b', 'A', 'B', 'c', ' ', 'İ', 'i', 'Σ', 'σ', 'ς'];
    let word = |rng: &mut Lcg, len: u64| -> String {
        (0..len)
            .map(|_| LETTERS[rng.below(LETTERS.len() as u64) as usize])
            .collect()
    };
    for seed in 1..=16u64 {
        let mut rng = Lcg(seed);
        let (mut qm, mut storage) = create_query_manager(SyncManager::new(), search_schema());
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
        qm.process(&mut storage);
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
            qm.process(&mut storage);
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
