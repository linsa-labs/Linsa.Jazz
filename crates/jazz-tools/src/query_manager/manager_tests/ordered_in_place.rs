//! A write that changes one row of an ordered list replaces that row where it stands.
//!
//! `Project` and `Output` used to rebuild their whole ordered list from their ordered
//! input on every non-empty delta — clone every tuple out of the input, re-project
//! every row, diff the new list against the old. The app's chat list holds every chat
//! with its members and their users included, so one contact's presence beat, or the
//! author's own `users` row, rebuilt every list that holds it. MEASURED on the client
//! stress bench (75 subscriptions, 4.7k live instances): 13 % of the author's own beat
//! in `ordered_tuples_owned` alone, Output and Project another 11 %.
//!
//! First the cost — one rename must not touch the rest of the list — then a
//! randomized differential: two query managers fed the same writes in lockstep, one
//! updating in place and one rebuilding, must deliver the same updates at every step,
//! over every node shape the in-place path sits behind (a window with and without an
//! offset, a projection with magic columns, two top-level includes) and with each row's
//! provenance compared, not only its data.

use super::*;
use crate::query_manager::graph_nodes::include_routing::{
    IncludeEmptyProbeMode, IncludeRoutingMode, force_include_empty_probe, force_include_routing,
};
use crate::query_manager::graph_nodes::output::QuerySubscriptionId;
use crate::query_manager::manager::QueryUpdate;
use crate::query_manager::ordered_in_place::{
    InPlaceMode, OrderedInPlaceMode, force_ordered_in_place, rebuilt_tuples_on_this_thread,
};
use crate::query_manager::precise_dirty::{PreciseDirtyMode, force_precise_dirty};
use crate::query_manager::session::WriteContext;
use crate::query_manager::settle_skips::{SettleSkipsMode, force_settle_skips};

const CHATS: &str = "chats";
const MEMBERS: &str = "members";
const PEOPLE: &str = "people";

fn schema() -> Schema {
    let mut schema = Schema::new();
    schema.insert(
        TableName::new(CHATS),
        RowDescriptor::new(vec![ColumnDescriptor::new("title", ColumnType::Text)]).into(),
    );
    schema.insert(
        TableName::new(MEMBERS),
        RowDescriptor::new(vec![
            ColumnDescriptor::new("chat_id", ColumnType::Uuid),
            ColumnDescriptor::new("person_id", ColumnType::Uuid),
        ])
        .into(),
    );
    schema.insert(
        TableName::new(PEOPLE),
        RowDescriptor::new(vec![ColumnDescriptor::new("name", ColumnType::Text)]).into(),
    );
    schema
}

/// The chat list's shape: chats by title, each with its members and the person each
/// names. With a page, the includes are built past the LimitOffset, so the list's
/// order comes through the include tail (`deferred_arrays_tail`).
fn subscribe(qm: &mut QueryManager, page: Option<usize>) -> QuerySubscriptionId {
    subscribe_shape(
        qm,
        Shape {
            page,
            ..Shape::PLAIN
        },
    )
}

/// The node shapes the in-place path sits behind.
#[derive(Debug, Clone, Copy)]
struct Shape {
    page: Option<usize>,
    offset: usize,
    /// A projection with magic columns at the top and on the included person: a
    /// `MagicColumns` node in the chain, and `$updatedAt`/`$updatedBy` values that change
    /// on a write that leaves the data as it was.
    magic: bool,
    /// A projection with no magic column: a `Project` whose content stays the same when a
    /// row is rewritten as it was, so only the batch changes.
    projected: bool,
    /// A second top-level include, so the list's order comes through a chain of two.
    second_include: bool,
}

impl Shape {
    const PLAIN: Shape = Shape {
        page: None,
        offset: 0,
        magic: false,
        projected: false,
        second_include: false,
    };
}

const SHAPES: [Shape; 8] = [
    Shape::PLAIN,
    Shape {
        projected: true,
        ..Shape::PLAIN
    },
    Shape {
        page: Some(6),
        ..Shape::PLAIN
    },
    Shape {
        page: Some(6),
        offset: 1,
        ..Shape::PLAIN
    },
    Shape {
        magic: true,
        second_include: true,
        ..Shape::PLAIN
    },
    Shape {
        page: Some(6),
        offset: 1,
        magic: true,
        projected: false,
        second_include: true,
    },
    // A page whose order comes through two includes and no magic column, and a projected
    // page: the ordered walk recurses through both levels with no `MagicColumns` node.
    Shape {
        page: Some(6),
        second_include: true,
        ..Shape::PLAIN
    },
    Shape {
        page: Some(6),
        projected: true,
        ..Shape::PLAIN
    },
];

fn subscribe_shape(qm: &mut QueryManager, shape: Shape) -> QuerySubscriptionId {
    let mut query = qm
        .query(CHATS)
        .with_array("members", |sub| {
            sub.from(MEMBERS)
                .correlate("chat_id", "chats.id")
                .with_array("person", |nested| {
                    let nested = nested.from(PEOPLE).correlate("id", "members.person_id");
                    if shape.magic {
                        nested.select(&["name", "$updatedAt", "$updatedBy"])
                    } else {
                        nested
                    }
                })
        })
        .order_by("title");
    if shape.second_include {
        query = query.with_array("seats", |sub| {
            sub.from(MEMBERS).correlate("chat_id", "chats.id")
        });
    }
    if shape.magic {
        let mut columns = vec!["title", "$updatedAt", "$updatedBy", "members"];
        if shape.second_include {
            columns.push("seats");
        }
        query = query.select(&columns);
    } else if shape.projected {
        query = query.select(&["title", "members"]);
    }
    if let Some(page) = shape.page {
        query = query.limit(page);
    }
    if shape.offset > 0 {
        query = query.offset(shape.offset);
    }
    qm.subscribe(query.build()).expect("subscribe")
}

/// Every process-wide plug pinned, in the shared lock order, the in-place mode last. The
/// others change which deltas reach `Project`/`Output` — a test forcing broadcast routing
/// beside this one made the 64-chat rename rebuild — so a gate that counts work must not
/// read them from whatever another test left forced.
struct Pinned {
    _precise: PreciseDirtyMode,
    _routing: IncludeRoutingMode,
    _empty_probe: IncludeEmptyProbeMode,
    _skips: SettleSkipsMode,
    in_place: OrderedInPlaceMode,
}

fn pin(mode: InPlaceMode) -> Pinned {
    let precise = force_precise_dirty(true);
    let routing = force_include_routing(true);
    let empty_probe = force_include_empty_probe(true);
    let skips = force_settle_skips(true);
    Pinned {
        _precise: precise,
        _routing: routing,
        _empty_probe: empty_probe,
        _skips: skips,
        in_place: force_ordered_in_place(mode),
    }
}

fn insert_with_id(
    qm: &mut QueryManager,
    storage: &mut MemoryStorage,
    table: &str,
    id: ObjectId,
    values: &[Value],
    context: Option<&WriteContext>,
) -> BatchId {
    let schema = qm.schema_context().current_schema.clone();
    let branch = get_branch(qm);
    qm.insert_on_branch_with_schema_and_write_context_and_id(
        storage,
        table,
        &branch,
        values,
        Some(id),
        &schema,
        context,
        true,
    )
    .expect("insert")
    .batch_id
}

/// `chat_count` chats, each with one member naming a person of its own. Returns the
/// people, by chat.
fn seed(qm: &mut QueryManager, storage: &mut MemoryStorage, chat_count: usize) -> Vec<ObjectId> {
    let mut people = Vec::new();
    for chat_index in 0..chat_count {
        let chat = qm
            .insert(
                storage,
                CHATS,
                &[Value::Text(format!("chat-{chat_index:04}"))],
            )
            .expect("insert chat")
            .row_id;
        let person = qm
            .insert(
                storage,
                PEOPLE,
                &[Value::Text(format!("person-{chat_index}"))],
            )
            .expect("insert person")
            .row_id;
        qm.insert(storage, MEMBERS, &[Value::Uuid(chat), Value::Uuid(person)])
            .expect("insert member");
        people.push(person);
    }
    people
}

/// Tuples rebuilt by `Project`/`Output` for one person rename, with `chat_count` chats
/// live — and the number of updated rows the subscriber was sent.
fn rebuilt_for_one_rename(chat_count: usize, page: Option<usize>) -> (u64, usize) {
    let (mut qm, mut storage) = create_query_manager(SyncManager::new(), schema());
    let people = seed(&mut qm, &mut storage, chat_count);
    let sub = subscribe(&mut qm, page);
    qm.process(&mut storage);
    let _ = qm.take_updates();

    let before = rebuilt_tuples_on_this_thread();
    qm.update(&mut storage, people[0], &[Value::Text("renamed".into())])
        .expect("rename the first chat's person");
    qm.process(&mut storage);
    let rebuilt = rebuilt_tuples_on_this_thread() - before;
    let updated = qm
        .take_updates()
        .into_iter()
        .filter(|update| update.subscription_id == sub)
        .map(|update| update.delta.updated.len())
        .sum();
    (rebuilt, updated)
}

/// Under `Verify` every node a skip covers runs and is counted as verified. That count is what a
/// release stand run with `JAZZ_SETTLE_SKIPS=verify` reads to know it checked anything: under
/// `Skip` the skips are counted all the same and the parity count stays zero by construction.
#[test]
fn every_skip_under_verify_runs_and_is_counted_as_verified() {
    use crate::query_manager::settle_cost::SKIPS_ON_THREAD;
    use crate::query_manager::settle_skips::force_settle_skips_verify;

    let _precise = force_precise_dirty(true);
    let _routing = force_include_routing(true);
    let _empty_probe = force_include_empty_probe(true);
    let _skips = force_settle_skips_verify();
    let _in_place = force_ordered_in_place(InPlaceMode::Apply);
    let (skipped_before, verified_before) = SKIPS_ON_THREAD.with(std::cell::Cell::get);
    let (_, updated) = rebuilt_for_one_rename(8, None);
    let (skipped, verified) = SKIPS_ON_THREAD.with(std::cell::Cell::get);
    let (skipped, verified) = (skipped - skipped_before, verified - verified_before);

    assert_eq!(
        updated, 1,
        "fixture precondition: the rename reaches the subscriber"
    );
    assert!(
        skipped > 0,
        "fixture precondition: a skip covered some node"
    );
    assert_eq!(
        verified, skipped,
        "under Verify every node a skip covered must run and be counted as verified"
    );
}

/// THE COST GATE. One rename changes one chat's row; the list of the others must not be
/// rebuilt for it, however many there are.
#[test]
fn renaming_one_person_does_not_rebuild_the_chat_list() {
    let _pinned = pin(InPlaceMode::Apply);
    for page in [None, Some(16)] {
        let (low, low_updated) = rebuilt_for_one_rename(8, page);
        let (high, high_updated) = rebuilt_for_one_rename(64, page);
        assert_eq!(
            (low_updated, high_updated),
            (1, 1),
            "page {page:?}: fixture precondition: the rename reaches the subscriber as one \
             updated chat"
        );
        assert!(
            high <= low,
            "page {page:?}: one rename rebuilt {high} ordered tuples with 64 chats live but \
             {low} with 8 — the whole list is rebuilt for a change to one of its rows"
        );
    }
}

struct Lcg(u64);

impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// What a subscriber is told. Batch ids differ between two nodes that made the same writes,
/// so each is named by the write that made it (`batches`, the node's own ids by write order).
fn canonical(
    updates: Vec<QueryUpdate>,
    sub: QuerySubscriptionId,
    batches: &HashMap<BatchId, usize>,
) -> Vec<String> {
    updates
        .into_iter()
        .filter(|update| update.subscription_id == sub)
        .map(|update| {
            // The provenance is the same on both nodes: they write with the same author and
            // timestamp.
            let row = |row: &crate::query_manager::types::Row| {
                (
                    row.id,
                    row.data.to_vec(),
                    batches.get(&row.batch_id).copied(),
                    row.provenance.updated_by.clone(),
                    row.provenance.updated_at,
                )
            };
            let delta = &update.delta;
            let ordered = &update.ordered_delta;
            format!(
                "added {:?} removed {:?} moved {:?} updated {:?} | ordered added {:?} removed {:?} updated {:?} pending {}",
                delta.added.iter().map(row).collect::<Vec<_>>(),
                delta.removed.iter().map(|row| row.id).collect::<Vec<_>>(),
                delta.moved,
                delta
                    .updated
                    .iter()
                    .map(|(old, new)| (row(old), row(new)))
                    .collect::<Vec<_>>(),
                ordered
                    .added
                    .iter()
                    .map(|added| {
                        (
                            added.id,
                            added.index,
                            added.row.data.to_vec(),
                            batches.get(&added.row.batch_id).copied(),
                        )
                    })
                    .collect::<Vec<_>>(),
                ordered
                    .removed
                    .iter()
                    .map(|removed| (removed.id, removed.index))
                    .collect::<Vec<_>>(),
                ordered
                    .updated
                    .iter()
                    .map(|updated| (
                        updated.id,
                        updated.old_index,
                        updated.new_index,
                        updated
                            .row
                            .as_ref()
                            .map(|row| (row.data.to_vec(), batches.get(&row.batch_id).copied())),
                    ))
                    .collect::<Vec<_>>(),
                ordered.pending,
            )
        })
        .collect()
}

/// Two nodes, the same writes, one updating in place and one rebuilding: every step must
/// deliver the same thing to both. The ops weigh renames — the in-place case — against
/// everything that must take the rebuild: inserts, deletes, a retitle that moves a chat,
/// a member that moves a person between chats. Titles repeat, so equal sort keys are
/// exercised too. A person rewritten with the name it has, by another author, changes
/// only the row's provenance — what the in-place path keeps from the old tuple at every
/// position it does not replace.
#[test]
fn random_writes_deliver_the_same_in_place_as_rebuilt() {
    let pinned = pin(InPlaceMode::Off);
    let mode = &pinned.in_place;
    // Passes in which the subscription's own `Output` (and, on a projected shape, its own
    // `Project`) took the in-place path. Counted on the top-level graph: an include instance's
    // `Output` updating in place says nothing about the walk this test exists for.
    let mut in_place_steps = [(0usize, 0usize); SHAPES.len()];
    for seed_value in [1u64, 2, 3, 0xbeef, 0x51ce] {
        for (shape_index, shape) in SHAPES.into_iter().enumerate() {
            let (mut fast, mut fast_io) = create_query_manager(SyncManager::new(), schema());
            let (mut slow, mut slow_io) = create_query_manager(SyncManager::new(), schema());
            let fast_sub = subscribe_shape(&mut fast, shape);
            let slow_sub = subscribe_shape(&mut slow, shape);
            let mut rng = Lcg(seed_value);
            let mut chats: Vec<ObjectId> = Vec::new();
            let mut people: Vec<ObjectId> = Vec::new();
            let mut names: HashMap<ObjectId, String> = HashMap::new();
            let mut members: Vec<ObjectId> = Vec::new();
            // Each node's batch ids by the write that made them, so the two can be compared.
            let mut fast_batches: HashMap<BatchId, usize> = HashMap::new();
            let mut slow_batches: HashMap<BatchId, usize> = HashMap::new();
            let mut writes = 0usize;

            for step in 0..160u64 {
                let op = rng.below(11);
                // Both nodes write as the same author at the same time, so the rows they
                // serve differ only in batch ids.
                let context = WriteContext {
                    attribution: Some(format!("author-{}", rng.below(3))),
                    updated_at: Some(1_000 + step),
                    ..WriteContext::default()
                };
                let mut write = |table: &str, id: ObjectId, values: Vec<Value>, fresh: bool| {
                    for ((qm, io), batches) in [
                        ((&mut fast, &mut fast_io), &mut fast_batches),
                        ((&mut slow, &mut slow_io), &mut slow_batches),
                    ] {
                        let batch = if fresh {
                            insert_with_id(qm, io, table, id, &values, Some(&context))
                        } else {
                            qm.update_with_write_context(io, id, &values, Some(&context))
                                .expect("update")
                        };
                        batches.insert(batch, writes);
                    }
                    writes += 1;
                };
                match op {
                    0 | 1 if chats.len() < 12 => {
                        let id = ObjectId::new();
                        let title = format!("chat-{}", rng.below(8));
                        write(CHATS, id, vec![Value::Text(title)], true);
                        chats.push(id);
                    }
                    2 if !chats.is_empty() => {
                        let id = chats[rng.below(chats.len())];
                        let title = format!("chat-{}", rng.below(8));
                        write(CHATS, id, vec![Value::Text(title)], false);
                    }
                    3 if chats.len() > 1 => {
                        let id = chats.swap_remove(rng.below(chats.len()));
                        let fast_batch = fast
                            .delete_with_write_context(&mut fast_io, id, Some(&context))
                            .expect("delete chat")
                            .batch_id;
                        let slow_batch = slow
                            .delete_with_write_context(&mut slow_io, id, Some(&context))
                            .expect("delete chat")
                            .batch_id;
                        fast_batches.insert(fast_batch, writes);
                        slow_batches.insert(slow_batch, writes);
                        writes += 1;
                    }
                    4 if !chats.is_empty() => {
                        let person = ObjectId::new();
                        let name = format!("person-{}", rng.below(100));
                        write(PEOPLE, person, vec![Value::Text(name.clone())], true);
                        names.insert(person, name);
                        people.push(person);
                        let member = ObjectId::new();
                        let chat = chats[rng.below(chats.len())];
                        write(
                            MEMBERS,
                            member,
                            vec![Value::Uuid(chat), Value::Uuid(person)],
                            true,
                        );
                        members.push(member);
                    }
                    5 if !members.is_empty() && !chats.is_empty() => {
                        let member = members[rng.below(members.len())];
                        let chat = chats[rng.below(chats.len())];
                        let person = people[rng.below(people.len())];
                        write(
                            MEMBERS,
                            member,
                            vec![Value::Uuid(chat), Value::Uuid(person)],
                            false,
                        );
                    }
                    6 if members.len() > 1 => {
                        let id = members.swap_remove(rng.below(members.len()));
                        let fast_batch = fast
                            .delete_with_write_context(&mut fast_io, id, Some(&context))
                            .expect("delete member")
                            .batch_id;
                        let slow_batch = slow
                            .delete_with_write_context(&mut slow_io, id, Some(&context))
                            .expect("delete member")
                            .batch_id;
                        fast_batches.insert(fast_batch, writes);
                        slow_batches.insert(slow_batch, writes);
                        writes += 1;
                    }
                    7 if !people.is_empty() => {
                        let id = people[rng.below(people.len())];
                        let name = names[&id].clone();
                        write(PEOPLE, id, vec![Value::Text(name)], false);
                    }
                    _ if !people.is_empty() => {
                        let id = people[rng.below(people.len())];
                        let name = format!("person-{}", rng.below(100));
                        write(PEOPLE, id, vec![Value::Text(name.clone())], false);
                        names.insert(id, name);
                    }
                    _ => continue,
                }

                mode.set(InPlaceMode::Apply);
                let (output_before, project_before) =
                    fast.subscriptions[&fast_sub].graph.applied_in_place();
                fast.process(&mut fast_io);
                let (output_after, project_after) =
                    fast.subscriptions[&fast_sub].graph.applied_in_place();
                in_place_steps[shape_index].0 += usize::from(output_after > output_before);
                in_place_steps[shape_index].1 += usize::from(project_after > project_before);
                mode.set(InPlaceMode::Off);
                slow.process(&mut slow_io);

                assert_eq!(
                    canonical(fast.take_updates(), fast_sub, &fast_batches),
                    canonical(slow.take_updates(), slow_sub, &slow_batches),
                    "seed {seed_value:#x} {shape:?} step {step} op {op}: the in-place node \
                     delivered differently from the rebuilding one"
                );
                // The sync scope rides on the same update: `Output` keeps it from its
                // tuples' provenance, and a server serves a client by it.
                assert_eq!(
                    fast.get_subscription_contributing_ids(fast_sub),
                    slow.get_subscription_contributing_ids(slow_sub),
                    "seed {seed_value:#x} {shape:?} step {step} op {op}: the in-place node's \
                     sync scope differs from the rebuilding one's"
                );
                // The batches each tuple carries are read in production (which subscriptions
                // a batch's fate re-checks, whether a transactional batch is visible), and a
                // write that leaves the data as it was changes only them.
                let batch_provenance =
                    |qm: &QueryManager,
                     sub: QuerySubscriptionId,
                     batches: &HashMap<BatchId, usize>| {
                        qm.subscriptions[&sub]
                            .graph
                            .current_output_tuples_ref()
                            .iter()
                            .map(|tuple| {
                                let mut writes: Vec<Option<usize>> = tuple
                                    .batch_provenance()
                                    .iter()
                                    .map(|batch| batches.get(batch).copied())
                                    .collect();
                                writes.sort();
                                // The rows each tuple stands on: a server reads them per
                                // tuple to scope what it sends.
                                let mut rows: Vec<String> = tuple
                                    .provenance()
                                    .iter()
                                    .map(|row| format!("{row:?}"))
                                    .collect();
                                rows.sort();
                                (tuple.first_id(), writes, rows)
                            })
                            .collect::<Vec<_>>()
                    };
                assert_eq!(
                    batch_provenance(&fast, fast_sub, &fast_batches),
                    batch_provenance(&slow, slow_sub, &slow_batches),
                    "seed {seed_value:#x} {shape:?} step {step} op {op}: the in-place node's \
                     tuples carry other batches or rows than the rebuilding one's"
                );
            }
        }
    }
    eprintln!("in-place steps per shape (output, project): {in_place_steps:?}");
    for (shape, (output, project)) in SHAPES.iter().zip(in_place_steps) {
        assert!(
            output > 0,
            "{shape:?}: the subscription's Output never took the in-place path, so the \
             differential compared nothing for it"
        );
        assert!(
            !shape.projected || project > 0,
            "{shape:?}: the subscription's Project never took the in-place path"
        );
    }
}
