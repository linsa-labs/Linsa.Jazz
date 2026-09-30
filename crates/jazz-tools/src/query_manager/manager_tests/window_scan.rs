//! Ordered windows over a composite index (`IndexScanNode::new_window`).
//!
//! A page `where chat = x order by at limit n` must read the page, not the chat:
//! before the window, the plan scanned the `chat` index, materialized every row of
//! the chat and sorted them to keep `n`. The first gate pins the cost; the second
//! pins the answer against a model of the query, through ties at the window's cut,
//! rows the filter drops, cursors on both sides, offsets, and live writes.
//!
//! Each test declares `wmsgs`'s `chat+at` composite index on its store first
//! (`declare_indexes`).

use super::*;
use crate::query_manager::settle_cost::WINDOW_GROWTHS;
use std::collections::HashSet;
use std::sync::atomic::Ordering as AtomicOrdering;

fn window_schema() -> Schema {
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
}

fn values(row: &ModelRow) -> Vec<Value> {
    vec![
        Value::Uuid(row.chat),
        Value::Timestamp(row.at),
        Value::Boolean(row.dead),
        Value::Text(format!("body {}", row.at)),
    ]
}

#[derive(Clone, Copy, Debug)]
enum Cursor {
    None,
    AtMost(u64),
    AtLeast(u64),
}

#[derive(Clone, Copy, Debug)]
struct Page {
    desc: bool,
    cursor: Cursor,
    offset: usize,
    limit: usize,
}

fn page_query(qm: &QueryManager, chat: ObjectId, page: Page) -> Query {
    let mut builder = qm
        .query("wmsgs")
        .filter_eq("chat", Value::Uuid(chat))
        .filter_eq("dead", Value::Boolean(false));
    builder = match page.cursor {
        Cursor::None => builder,
        Cursor::AtMost(at) => builder.filter_le("at", Value::Timestamp(at)),
        Cursor::AtLeast(at) => builder.filter_ge("at", Value::Timestamp(at)),
    };
    builder = if page.desc {
        builder.order_by_desc("at")
    } else {
        builder.order_by("at")
    };
    builder.offset(page.offset).limit(page.limit).build()
}

/// The page the query must return: filter, order by `at` (ties by id ascending, as
/// the Sort node breaks them), then offset and limit.
fn expected(model: &[ModelRow], chat: ObjectId, page: Page) -> Vec<ObjectId> {
    let mut rows: Vec<&ModelRow> = model
        .iter()
        .filter(|row| row.chat == chat && !row.dead)
        .filter(|row| match page.cursor {
            Cursor::None => true,
            Cursor::AtMost(at) => row.at <= at,
            Cursor::AtLeast(at) => row.at >= at,
        })
        .collect();
    rows.sort_by(|left, right| {
        let by_at = if page.desc {
            right.at.cmp(&left.at)
        } else {
            left.at.cmp(&right.at)
        };
        by_at.then(left.id.cmp(&right.id))
    });
    rows.into_iter()
        .skip(page.offset)
        .take(page.limit)
        .map(|row| row.id)
        .collect()
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

/// A deterministic generator, so a failure reproduces from its seed.
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
fn window_page_loads_the_page_not_the_chat() {
    let (mut qm, mut storage) = create_query_manager(SyncManager::new(), window_schema());
    declare_indexes(&mut qm, &mut storage, wmsgs_index_declarations());
    let chat = ObjectId::new();
    let other = ObjectId::new();
    let mut model = Vec::new();
    for index in 0..300u64 {
        for (owner, at) in [(chat, index * 10), (other, index * 10 + 5)] {
            let row = ModelRow {
                id: ObjectId::new(),
                chat: owner,
                at,
                dead: false,
            };
            insert_row(&mut qm, &mut storage, &row);
            model.push(row);
        }
    }
    qm.process(&mut storage);
    qm.take_updates();

    let page = Page {
        desc: true,
        cursor: Cursor::None,
        offset: 0,
        limit: 20,
    };
    let sub = qm
        .subscribe(page_query(&qm, chat, page))
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
    assert_eq!(got, expected(&model, chat, page), "the newest page");
    // The window's first walk is the page plus its slack (16); the chat is 300 rows.
    assert!(
        scanned <= 40,
        "a 20-row page of a 300-row chat scanned {scanned} rows"
    );
}

#[test]
fn window_pages_match_the_model_through_ties_filters_and_writes() {
    for seed in 1..=24u64 {
        let mut rng = Lcg(seed);
        let (mut qm, mut storage) = create_query_manager(SyncManager::new(), window_schema());
        declare_indexes(&mut qm, &mut storage, wmsgs_index_declarations());
        let chat = ObjectId::new();
        let other = ObjectId::new();
        let mut model: Vec<ModelRow> = Vec::new();

        // Ties: several rows share each `at`, so the window's cut lands inside groups.
        // Groups larger than the first walk's slack put the page's last row in the
        // group the walk cut: a descending walk meets a group's ids highest first,
        // while the page ranks them lowest first.
        let rows = 120 + rng.below(120);
        let group = 1 + rng.below(if seed % 2 == 0 { 6 } else { 40 });
        for index in 0..rows {
            let row = ModelRow {
                id: ObjectId::new(),
                chat: if rng.below(4) == 0 { other } else { chat },
                at: (index / group) * 10,
                dead: rng.below(3) == 0,
            };
            insert_row(&mut qm, &mut storage, &row);
            model.push(row);
        }
        qm.process(&mut storage);
        qm.take_updates();

        let span = (rows / group) * 10;
        let pages = [
            Page {
                desc: true,
                cursor: Cursor::None,
                offset: 0,
                limit: 1 + rng.below(25) as usize,
            },
            Page {
                desc: true,
                cursor: Cursor::AtMost(rng.below(span)),
                offset: 0,
                limit: 1 + rng.below(25) as usize,
            },
            Page {
                desc: false,
                cursor: Cursor::AtLeast(rng.below(span)),
                offset: 0,
                limit: 1 + rng.below(25) as usize,
            },
            Page {
                desc: rng.below(2) == 0,
                cursor: Cursor::None,
                offset: rng.below(10) as usize,
                limit: 1 + rng.below(15) as usize,
            },
        ];
        let subs: Vec<_> = pages
            .iter()
            .map(|page| {
                qm.subscribe(page_query(&qm, chat, *page))
                    .expect("subscribe")
            })
            .collect();
        qm.process(&mut storage);
        // What a client learns from the deltas alone.
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
            for (slot, (sub, page)) in subs.iter().zip(pages.iter()).enumerate() {
                let got: Vec<ObjectId> = qm
                    .get_subscription_results(*sub)
                    .into_iter()
                    .map(|(id, _)| id)
                    .collect();
                let want = expected(&model, chat, *page);
                assert_eq!(got, want, "seed {seed}, step {step}, page {page:?}");
                assert_eq!(
                    delivered[slot],
                    want.iter().copied().collect::<HashSet<_>>(),
                    "seed {seed}, step {step}, page {page:?}: the deltas disagree with the result"
                );
            }

            // One write: a new row anywhere in time, or a flag flipped either way.
            if model.is_empty() || rng.below(3) == 0 {
                let row = ModelRow {
                    id: ObjectId::new(),
                    chat: if rng.below(5) == 0 { other } else { chat },
                    at: rng.below(span + 40),
                    dead: false,
                };
                insert_row(&mut qm, &mut storage, &row);
                model.push(row);
            } else {
                let index = rng.below(model.len() as u64) as usize;
                model[index].dead = !model[index].dead;
                let row = model[index].clone();
                qm.update(&mut storage, row.id, &values(&row))
                    .expect("update wmsgs row");
            }
            qm.process(&mut storage);
            absorb(&mut qm, &mut delivered);
        }
    }
    assert!(
        WINDOW_GROWTHS.load(AtomicOrdering::Relaxed) > 0,
        "no window ever grew: the filters never cut a page short, so the refill went untested"
    );
}

/// A contradiction on a column neither index covers (`dead = false and dead = true`)
/// matches no row. The plain plan's scan of that column is empty and it drops the
/// Filter as covered; a window or a search reads the chat instead, so it keeps the
/// Filter — declared and complete, or still filling (the chat's own index then).
#[test]
fn a_contradiction_outside_the_index_columns_matches_nothing() {
    for complete in [true, false] {
        let (mut qm, mut storage) = create_query_manager(SyncManager::new(), window_schema());
        let chat = ObjectId::new();
        for at in 0..40u64 {
            let row = ModelRow {
                id: ObjectId::new(),
                chat,
                at,
                dead: false,
            };
            insert_row(&mut qm, &mut storage, &row);
        }
        qm.process(&mut storage);
        if complete {
            declare_indexes(&mut qm, &mut storage, wmsgs_index_declarations());
        } else {
            qm.propose_index_declarations(wmsgs_index_declarations());
        }
        qm.take_updates();

        let contradiction = || {
            qm.query("wmsgs")
                .filter_eq("chat", Value::Uuid(chat))
                .filter_eq("dead", Value::Boolean(false))
                .filter_eq("dead", Value::Boolean(true))
        };
        let window = contradiction().order_by_desc("at").limit(20).build();
        let search = contradiction()
            .filter_contains("body", Value::Text("body".to_string()))
            .order_by_desc("at")
            .build();
        for (name, query) in [("window", window), ("search", search)] {
            let sub = qm.subscribe(query).expect("subscribe");
            qm.process(&mut storage);
            let got = qm.get_subscription_results(sub);
            assert!(
                got.is_empty(),
                "complete {complete}: the {name} returned {} rows of a query that matches none",
                got.len()
            );
        }
    }
}

/// `wmsgs` as `window_schema`, with a nullable `at`.
fn nullable_window_schema() -> Schema {
    let mut schema = Schema::new();
    schema.insert(
        TableName::new("wmsgs"),
        RowDescriptor::new(vec![
            ColumnDescriptor::new("chat", ColumnType::Uuid),
            ColumnDescriptor::new("at", ColumnType::Timestamp).nullable(),
            ColumnDescriptor::new("dead", ColumnType::Boolean),
            ColumnDescriptor::new("body", ColumnType::Text),
        ])
        .into(),
    );
    schema
}

/// A row of `chat` whose `at` may be null.
fn insert_nullable_row(
    qm: &mut QueryManager,
    storage: &mut MemoryStorage,
    chat: ObjectId,
    at: Option<u64>,
) -> ObjectId {
    let branch = get_branch(qm);
    let schema = qm.schema_context().current_schema.clone();
    let id = ObjectId::new();
    let values = [
        Value::Uuid(chat),
        at.map_or(Value::Null, Value::Timestamp),
        Value::Boolean(false),
        Value::Text("body".to_string()),
    ];
    qm.insert_on_branch_with_schema_and_write_context_and_id(
        storage,
        "wmsgs",
        &branch,
        &values,
        Some(id),
        &schema,
        None,
        true,
    )
    .expect("insert wmsgs row");
    id
}

/// A null `at` sorts below every value, so the query `chat = x and at <= t` returns
/// the chat's rows that have none, while the composite index files no entry for
/// them. A window serves a nullable `at` only under a lower bound, which excludes
/// the nulls as the Filter does.
#[test]
fn a_window_over_a_nullable_column_needs_a_lower_bound() {
    let (mut qm, mut storage) = create_query_manager(SyncManager::new(), nullable_window_schema());
    let chat = ObjectId::new();
    let other = ObjectId::new();
    let mut model: Vec<(ObjectId, Option<u64>)> = Vec::new();
    for at in (1..=10u64)
        .map(|step| Some(step * 10))
        .chain([None, None, None])
    {
        let id = insert_nullable_row(&mut qm, &mut storage, chat, at);
        model.push((id, at));
    }
    for at in [None, Some(25)] {
        insert_nullable_row(&mut qm, &mut storage, other, at);
    }
    qm.process(&mut storage);

    type Bounded = fn(QueryBuilder) -> QueryBuilder;
    type Matches = fn(Option<u64>) -> bool;
    let cases: [(&str, Bounded, Matches, bool); 5] = [
        (
            "at <= 50, newest first",
            |query| query.filter_le("at", Value::Timestamp(50)),
            |at| at.is_none_or(|at| at <= 50),
            true,
        ),
        (
            "at < 50, oldest first",
            |query| query.filter_lt("at", Value::Timestamp(50)),
            |at| at.is_none_or(|at| at < 50),
            false,
        ),
        ("unbounded, newest first", |query| query, |_| true, true),
        (
            "at >= 30, newest first",
            |query| query.filter_ge("at", Value::Timestamp(30)),
            |at| at.is_some_and(|at| at >= 30),
            true,
        ),
        (
            "at > 30, oldest first",
            |query| query.filter_gt("at", Value::Timestamp(30)),
            |at| at.is_some_and(|at| at > 30),
            false,
        ),
    ];
    // Nulls first in ascending order, last in descending; ties by id.
    let expected = |matches: Matches, desc: bool| -> Vec<ObjectId> {
        let mut rows: Vec<_> = model.iter().filter(|(_, at)| matches(*at)).collect();
        rows.sort_by(|left, right| {
            let by_at = if desc {
                right.1.cmp(&left.1)
            } else {
                left.1.cmp(&right.1)
            };
            by_at.then(left.0.cmp(&right.0))
        });
        rows.into_iter().take(20).map(|(id, _)| *id).collect()
    };
    let run = |qm: &mut QueryManager, storage: &mut MemoryStorage, bounded: Bounded, desc: bool| {
        let query = bounded(qm.query("wmsgs").filter_eq("chat", Value::Uuid(chat)));
        let query = if desc {
            query.order_by_desc("at")
        } else {
            query.order_by("at")
        };
        let sub = qm.subscribe(query.limit(20).build()).expect("subscribe");
        qm.process(storage);
        let got: Vec<ObjectId> = qm
            .get_subscription_results(sub)
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        got
    };

    for (name, bounded, matches, desc) in cases {
        assert_eq!(
            run(&mut qm, &mut storage, bounded, desc),
            expected(matches, desc),
            "{name}: the plain plan"
        );
    }
    declare_indexes(&mut qm, &mut storage, wmsgs_index_declarations());
    for (name, bounded, matches, desc) in cases {
        assert_eq!(
            run(&mut qm, &mut storage, bounded, desc),
            expected(matches, desc),
            "{name}: over the declared index"
        );
    }
}

/// A literal is compared as it is written: the plain plan looks the first column up
/// by the literal's own encoding, so an `Integer` names no `BigInt` value. A window
/// that converted it would find rows the plain plan does not, and declaring an index
/// would change the answer.
#[test]
fn a_window_does_not_convert_the_literals_it_is_given() {
    let mut schema = Schema::new();
    schema.insert(
        TableName::new("wseq"),
        RowDescriptor::new(vec![
            ColumnDescriptor::new("owner", ColumnType::BigInt),
            ColumnDescriptor::new("at", ColumnType::Timestamp),
        ])
        .into(),
    );
    let (mut qm, mut storage) = create_query_manager(SyncManager::new(), schema);
    for at in 0..10u64 {
        qm.insert(
            &mut storage,
            "wseq",
            &[Value::BigInt(5), Value::Timestamp(at)],
        )
        .expect("insert wseq row");
    }
    qm.process(&mut storage);

    let results = |qm: &mut QueryManager, storage: &mut MemoryStorage, owner: Value| {
        let query = qm
            .query("wseq")
            .filter_eq("owner", owner)
            .order_by_desc("at")
            .limit(5)
            .build();
        let sub = qm.subscribe(query).expect("subscribe");
        qm.process(storage);
        qm.get_subscription_results(sub).len()
    };
    let plain_integer = results(&mut qm, &mut storage, Value::Integer(5));
    let plain_bigint = results(&mut qm, &mut storage, Value::BigInt(5));
    assert_eq!(
        plain_bigint, 5,
        "the plain plan finds the owner's rows, or this gates nothing"
    );

    declare_indexes(
        &mut qm,
        &mut storage,
        crate::query_manager::index_declarations::IndexDeclarations::empty()
            .with_composite("wseq", "owner", "at")
            .expect("composite declaration"),
    );
    assert_eq!(
        results(&mut qm, &mut storage, Value::Integer(5)),
        plain_integer,
        "an Integer literal over the declared index"
    );
    assert_eq!(
        results(&mut qm, &mut storage, Value::BigInt(5)),
        plain_bigint,
        "a BigInt literal over the declared index"
    );
}
