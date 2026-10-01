//! Liveness matrix for correlation-routed include dirt (v14 lever L1).
//!
//! Routing decides which cached subgraph instances a changed row can reach.
//! Getting that decision wrong in the permissive direction costs work. Getting
//! it wrong in the RESTRICTIVE direction — a mark that never reaches the
//! instance holding the row — produces no error, no log and no panic: the
//! subscriber simply keeps serving a stale include array forever. Absence of a
//! test for a mutation class IS the risk, so this file enumerates the classes
//! rather than sampling them.
//!
//! Every case: apply ONE mutation, run ONE `process()`, and assert the
//! subscriber's mirror — rebuilt from the emitted deltas, never read out of
//! engine state — equals the expected id tree. A mark that was routed nowhere
//! leaves the mirror on the pre-image and fails here.
//!
//! The matrix is {insert, content update, delete, filter flip, correlate-value
//! change, correlate change while filtered out} × {leaf include, nested
//! include}, plus the depth-2 classes (grandchild insert / update / delete /
//! re-home / re-home across parents). Each case runs under BOTH routing modes:
//! routed (the default) and broadcast (`JAZZ_INCLUDE_ROUTING=0`), so a failure
//! says immediately whether the bug is in routing or older than it.
//!
//! Since v24 a binding whose instance would scan nothing is answered from the
//! correlation index and holds no instance at all (`EmptyBinding`). Three of
//! the four seeded parents are childless, so most slots in this fixture are
//! exactly that — and "a row arrived for an empty binding and nobody probed it
//! again" is the same silent failure as a mark routed nowhere. Every case
//! therefore also runs with empty bindings ON and OFF
//! (`JAZZ_INCLUDE_EMPTY_PROBE=0`), and the classes that only exist for an empty
//! binding have their own section at the end.
//!
//! This complements, and does not duplicate, `subscription_output_oracle`:
//! that proves routed and unrouted output streams are byte-identical over
//! randomized op sequences; this pins each named mutation class to a
//! hand-written expectation, so a change that breaks BOTH modes identically —
//! which a differential cannot see — still fails.

use std::collections::HashMap;

use super::*;
use crate::query_manager::graph_nodes::include_routing::{
    empty_bindings_bound_on_this_thread, force_include_empty_probe, force_include_routing,
};
use crate::query_manager::manager::QueryUpdate;
use crate::query_manager::precise_dirty::force_precise_dirty;

const PARENT_TABLE: &str = "parents";
const CHILD_TABLE: &str = "children";
const GRANDCHILD_TABLE: &str = "grandchildren";

/// Seeded parents. Above `MIN_ROUTABLE_INSTANCES`, deliberately: a node at or
/// below the floor broadcasts, and a "routed" case that silently broadcast
/// would prove nothing.
const PARENT_COUNT: usize = 4;

/// Column index of the `children` array in the parent output row.
const PARENT_INCLUDE_COLUMN: usize = 1;
/// Column index of the `grandchildren` array inside a child row (`children`
/// carries `title`, `is_deleted`, `parent_id`, then the nested array).
const CHILD_INCLUDE_COLUMN: usize = 3;

/// (parent id, [(child id, [grandchild id])]) — the whole observable shape.
type Tree = Vec<(ObjectId, Vec<(ObjectId, Vec<ObjectId>)>)>;

/// Which include depth a case runs against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Depth {
    /// `parents -> children`.
    Leaf,
    /// `parents -> children -> grandchildren`.
    Nested,
}

/// Which routing mode the engine runs in for a case.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    /// Correlation routing on — the default runtime path.
    Routed,
    /// Every mark broadcast to every cached instance — the v13-3 path, kept
    /// reachable by `JAZZ_INCLUDE_ROUTING=0`.
    Broadcast,
}

fn liveness_schema() -> Schema {
    let mut schema = Schema::new();
    schema.insert(
        TableName::new(PARENT_TABLE),
        RowDescriptor::new(vec![ColumnDescriptor::new("title", ColumnType::Text)]).into(),
    );
    schema.insert(
        TableName::new(CHILD_TABLE),
        RowDescriptor::new(vec![
            ColumnDescriptor::new("title", ColumnType::Text),
            ColumnDescriptor::new("is_deleted", ColumnType::Boolean),
            ColumnDescriptor::new("parent_id", ColumnType::Uuid),
        ])
        .into(),
    );
    schema.insert(
        TableName::new(GRANDCHILD_TABLE),
        RowDescriptor::new(vec![
            ColumnDescriptor::new("title", ColumnType::Text),
            ColumnDescriptor::new("child_id", ColumnType::Uuid),
        ])
        .into(),
    );
    schema
}

/// One engine, one subscription, and the client mirror it feeds.
struct Harness {
    qm: QueryManager,
    storage: MemoryStorage,
    sub_id: crate::query_manager::graph_nodes::output::QuerySubscriptionId,
    depth: Depth,
    rows: HashMap<ObjectId, crate::query_manager::types::Row>,
    ordered: Vec<ObjectId>,
    descriptor: Option<RowDescriptor>,
    /// Seeded parents, in insertion order — the fixture's stable handles.
    parents: Vec<ObjectId>,
}

impl Harness {
    /// `PARENT_COUNT` parents, two children under the first, one grandchild
    /// under each child. Small enough to write expectations by hand, wide
    /// enough that a mark routed to the WRONG instance shows up as a missing
    /// update rather than an accidental pass.
    fn seed(depth: Depth) -> (Self, Vec<ObjectId>, Vec<ObjectId>) {
        let (mut qm, mut storage) = create_query_manager(SyncManager::new(), liveness_schema());

        let parents: Vec<ObjectId> = (0..PARENT_COUNT)
            .map(|index| {
                qm.insert(
                    &mut storage,
                    PARENT_TABLE,
                    &[Value::Text(format!("parent-{index}"))],
                )
                .expect("insert parent")
                .row_id
            })
            .collect();

        let children: Vec<ObjectId> = (0..2)
            .map(|index| {
                qm.insert(
                    &mut storage,
                    CHILD_TABLE,
                    &[
                        Value::Text(format!("child-{index}")),
                        Value::Boolean(false),
                        Value::Uuid(parents[0]),
                    ],
                )
                .expect("insert child")
                .row_id
            })
            .collect();

        let grandchildren: Vec<ObjectId> = children
            .iter()
            .enumerate()
            .map(|(index, child)| {
                qm.insert(
                    &mut storage,
                    GRANDCHILD_TABLE,
                    &[
                        Value::Text(format!("grandchild-{index}")),
                        Value::Uuid(*child),
                    ],
                )
                .expect("insert grandchild")
                .row_id
            })
            .collect();

        let query = qm
            .query(PARENT_TABLE)
            .with_array("children", |sub| {
                let sub = sub
                    .from(CHILD_TABLE)
                    .correlate("parent_id", "parents.id")
                    .filter_eq("is_deleted", Value::Boolean(false))
                    .order_by("title");
                match depth {
                    Depth::Leaf => sub,
                    Depth::Nested => sub.with_array("grandchildren", |nested| {
                        nested
                            .from(GRANDCHILD_TABLE)
                            .correlate("child_id", "children.id")
                            .order_by("title")
                    }),
                }
            })
            .order_by("title")
            .build();
        let sub_id = qm.subscribe(query).expect("subscribe");

        let mut harness = Self {
            qm,
            storage,
            sub_id,
            depth,
            rows: HashMap::new(),
            ordered: Vec::new(),
            descriptor: None,
            parents,
        };
        harness.settle_once();
        let seeded = expect(
            &harness,
            &[(
                0,
                vec![
                    (children[0], nested_ids(depth, &grandchildren[..1])),
                    (children[1], nested_ids(depth, &grandchildren[1..])),
                ],
            )],
        );
        assert_eq!(
            harness.tree(),
            seeded,
            "the cold-open snapshot must already carry the whole tree"
        );
        (harness, children, grandchildren)
    }

    /// One settle pass, folding whatever it emits into the mirror.
    ///
    /// ONE pass, deliberately: an include array is allowed to arrive in the
    /// same `process()` as the write that caused it, and the whole point of
    /// this file is that it still does once marks are routed instead of
    /// broadcast.
    fn settle_once(&mut self) {
        self.qm.process(&mut self.storage);
        let sub_id = self.sub_id;
        let updates: Vec<QueryUpdate> = self
            .qm
            .take_updates()
            .into_iter()
            .filter(|update| update.subscription_id == sub_id)
            .collect();
        for update in updates {
            self.absorb(update);
        }
    }

    fn absorb(&mut self, update: QueryUpdate) {
        self.descriptor = Some(update.descriptor.clone());
        for row in &update.delta.removed {
            self.rows.remove(&row.id);
            self.ordered.retain(|id| *id != row.id);
        }
        for (_, new_row) in &update.delta.updated {
            self.rows.insert(new_row.id, new_row.clone());
        }
        for row in &update.delta.added {
            if self.rows.insert(row.id, row.clone()).is_none() {
                self.ordered.push(row.id);
            }
        }
        // The fixture's parents sort by a stable title, so mirror order is the
        // seeded order. This file asserts CONTENT liveness; ordering is covered
        // byte-for-byte by `subscription_output_oracle`.
        let parents = self.parents.clone();
        self.ordered
            .sort_by_key(|id| parents.iter().position(|parent| parent == id));
    }

    /// Decode the mirror into the id tree the subscriber can actually see.
    fn tree(&self) -> Tree {
        let Some(descriptor) = self.descriptor.as_ref() else {
            return Vec::new();
        };
        self.ordered
            .iter()
            .map(|parent_id| {
                let values = decode_row(descriptor, &self.rows[parent_id].data)
                    .expect("decode mirrored parent row");
                let children = include_array(&values, PARENT_INCLUDE_COLUMN)
                    .iter()
                    .map(|child| {
                        let (child_id, child_values) = included_row(child);
                        let grandchildren = match self.depth {
                            Depth::Leaf => Vec::new(),
                            Depth::Nested => include_array(child_values, CHILD_INCLUDE_COLUMN)
                                .iter()
                                .map(|grandchild| included_row(grandchild).0)
                                .collect(),
                        };
                        (child_id, grandchildren)
                    })
                    .collect();
                (*parent_id, children)
            })
            .collect()
    }
}

/// Build an expected tree: the listed parent slots get the given children,
/// every other seeded parent an empty include array.
fn expect(harness: &Harness, assigned: &[(usize, Vec<(ObjectId, Vec<ObjectId>)>)]) -> Tree {
    harness
        .parents
        .iter()
        .enumerate()
        .map(|(slot, parent)| {
            let children = assigned
                .iter()
                .find(|(assigned_slot, _)| *assigned_slot == slot)
                .map(|(_, children)| children.clone())
                .unwrap_or_default();
            (*parent, children)
        })
        .collect()
}

/// Grandchild ids, but only where the subscription actually asks for them.
fn nested_ids(depth: Depth, ids: &[ObjectId]) -> Vec<ObjectId> {
    match depth {
        Depth::Leaf => Vec::new(),
        Depth::Nested => ids.to_vec(),
    }
}

fn include_array(values: &[Value], column: usize) -> &[Value] {
    match values.get(column) {
        Some(Value::Array(items)) => items,
        other => panic!("expected an include array at column {column}, got {other:?}"),
    }
}

fn included_row(value: &Value) -> (ObjectId, &Vec<Value>) {
    match value {
        Value::Row {
            id: Some(id),
            values,
        } => (*id, values),
        other => panic!("expected an included row carrying its id, got {other:?}"),
    }
}

/// Whether a binding with nothing to scan holds an instance or an
/// `EmptyBinding`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Empty {
    /// Answered from the correlation index — the default runtime path.
    Bindings,
    /// A compiled instance per binding — the v23 path, kept reachable by
    /// `JAZZ_INCLUDE_EMPTY_PROBE=0`.
    Instances,
}

/// Run one case under one routing mode and one way of holding empty bindings.
///
/// The mutation runs inside the mode guards, so both the write-time buffering
/// and the settle-time routing decision belong to the mode under test.
fn run_case(
    depth: Depth,
    mode: Mode,
    empty: Empty,
    case: impl Fn(&mut Harness, &[ObjectId], &[ObjectId]) -> Tree,
) {
    let _precise = force_precise_dirty(true);
    let _routing = force_include_routing(matches!(mode, Mode::Routed));
    let _empty = force_include_empty_probe(matches!(empty, Empty::Bindings));

    let bound_before = empty_bindings_bound_on_this_thread();
    let (mut harness, children, grandchildren) = Harness::seed(depth);
    let bound_at_seed = empty_bindings_bound_on_this_thread() - bound_before;
    match empty {
        // The include learns its probe from the first instance it compiles, which
        // may itself be a childless parent's: at least all but one of them.
        Empty::Bindings => assert!(
            bound_at_seed >= (PARENT_COUNT - 2) as u64,
            "{depth:?}/{mode:?}: the fixture's childless parents hold instances \
             ({bound_at_seed} empty bindings), so this run says nothing about empty bindings"
        ),
        Empty::Instances => assert_eq!(
            bound_at_seed, 0,
            "{depth:?}/{mode:?}: the kill switch still made empty bindings"
        ),
    }
    let expected = case(&mut harness, &children, &grandchildren);
    assert_eq!(
        harness.tree(),
        expected,
        "{depth:?}/{mode:?}/{empty:?}: the mutation did not reach the subscriber within one \
         settle — a mark that routes nowhere is silent, so this assertion is the only signal"
    );
}

/// Run one case at both depths, in both routing modes, with and without empty
/// bindings.
fn matrix(case: impl Fn(&mut Harness, &[ObjectId], &[ObjectId]) -> Tree + Copy) {
    for depth in [Depth::Leaf, Depth::Nested] {
        for mode in [Mode::Routed, Mode::Broadcast] {
            for empty in [Empty::Bindings, Empty::Instances] {
                run_case(depth, mode, empty, case);
            }
        }
    }
}

/// Depth-2 classes only exist in the nested shape.
fn nested_matrix(case: impl Fn(&mut Harness, &[ObjectId], &[ObjectId]) -> Tree + Copy) {
    for mode in [Mode::Routed, Mode::Broadcast] {
        for empty in [Empty::Bindings, Empty::Instances] {
            run_case(Depth::Nested, mode, empty, case);
        }
    }
}

/// The children the seed leaves under the first parent, unchanged.
fn seeded_children(
    depth: Depth,
    children: &[ObjectId],
    grandchildren: &[ObjectId],
) -> Vec<(ObjectId, Vec<ObjectId>)> {
    vec![
        (children[0], nested_ids(depth, &grandchildren[..1])),
        (children[1], nested_ids(depth, &grandchildren[1..])),
    ]
}

// ============================================================================
// Leaf and nested classes.
// ============================================================================

/// INSERT into the include's inner table: the new row carries the correlate,
/// so it routes by binding lookup — there is no reverse-index entry for a row
/// that did not exist a moment ago.
#[test]
fn a_new_child_reaches_its_parents_include() {
    matrix(|harness, children, grandchildren| {
        let new_child = harness
            .qm
            .insert(
                &mut harness.storage,
                CHILD_TABLE,
                &[
                    Value::Text("child-9".into()),
                    Value::Boolean(false),
                    Value::Uuid(harness.parents[1]),
                ],
            )
            .expect("insert child")
            .row_id;
        harness.settle_once();
        expect(
            harness,
            &[
                (0, seeded_children(harness.depth, children, grandchildren)),
                (1, vec![(new_child, Vec::new())]),
            ],
        )
    });
}

/// CONTENT UPDATE of a row an instance already holds: the row keeps its
/// correlate, so it routes by binding AND by reverse index. This is the class
/// v13-2 shipped broken (the array stayed byte-stale forever).
#[test]
fn an_edited_child_reaches_its_parents_include() {
    matrix(|harness, children, grandchildren| {
        harness
            .qm
            .update(
                &mut harness.storage,
                children[0],
                &[
                    Value::Text("child-0-edited".into()),
                    Value::Boolean(false),
                    Value::Uuid(harness.parents[0]),
                ],
            )
            .expect("update child");
        harness.settle_once();
        assert_eq!(
            child_title(harness, children[0]),
            "child-0-edited",
            "the edited title must be visible in the mirrored include array"
        );
        expect(
            harness,
            &[(0, seeded_children(harness.depth, children, grandchildren))],
        )
    });
}

/// DELETE of a held row: the loader stops seeing it, so nothing resolves by
/// correlation — it routes purely through the reverse index of instances that
/// held it.
#[test]
fn a_deleted_child_leaves_its_parents_include() {
    matrix(|harness, children, grandchildren| {
        harness
            .qm
            .delete(&mut harness.storage, children[0])
            .expect("delete child");
        harness.settle_once();
        expect(
            harness,
            &[(
                0,
                vec![(children[1], nested_ids(harness.depth, &grandchildren[1..]))],
            )],
        )
    });
}

/// FILTER FLIP: a content update that moves a row across the include's
/// `is_deleted` predicate. The row stays correlated to the same parent and
/// stays inside that instance's SCAN, so only a downstream filter retracts it
/// — the mark still has to arrive.
#[test]
fn a_filter_flipped_child_leaves_and_re_enters_its_parents_include() {
    matrix(|harness, children, grandchildren| {
        let flip = |harness: &mut Harness, deleted: bool| {
            let parent = harness.parents[0];
            harness
                .qm
                .update(
                    &mut harness.storage,
                    children[0],
                    &[
                        Value::Text("child-0".into()),
                        Value::Boolean(deleted),
                        Value::Uuid(parent),
                    ],
                )
                .expect("flip is_deleted");
            harness.settle_once();
        };

        flip(harness, true);
        let retracted = expect(
            harness,
            &[(
                0,
                vec![(children[1], nested_ids(harness.depth, &grandchildren[1..]))],
            )],
        );
        assert_eq!(
            harness.tree(),
            retracted,
            "flipping is_deleted must retract the child within one settle"
        );

        // ...and back. The retracted row is no longer in any array, so only the
        // scan-membership half of the reverse index can name its instance.
        flip(harness, false);
        expect(
            harness,
            &[(0, seeded_children(harness.depth, children, grandchildren))],
        )
    });
}

/// CORRELATION-VALUE CHANGE: one write must retract from the old instance and
/// add to the new one. Both halves are routed, and each comes from a different
/// index — the old home from `held_by`, the new one from `by_correlation`.
#[test]
fn a_re_parented_child_moves_between_includes_in_one_settle() {
    matrix(|harness, children, grandchildren| {
        let target = harness.parents[1];
        harness
            .qm
            .update(
                &mut harness.storage,
                children[0],
                &[
                    Value::Text("child-0".into()),
                    Value::Boolean(false),
                    Value::Uuid(target),
                ],
            )
            .expect("re-parent child");
        harness.settle_once();
        let depth = harness.depth;
        expect(
            harness,
            &[
                (
                    0,
                    vec![(children[1], nested_ids(depth, &grandchildren[1..]))],
                ),
                (
                    1,
                    vec![(children[0], nested_ids(depth, &grandchildren[..1]))],
                ),
            ],
        )
    });
}

/// A re-parented child that is INVISIBLE in its old parent's array (filtered
/// out) still has to leave that instance's scan.
///
/// The array-membership half of the reverse index cannot see this row, so the
/// mark reaches the old instance only through scan membership. Getting it
/// wrong leaves the old instance's incremental index-scan baseline ahead of a
/// full rescan — silent until the row flips back.
#[test]
fn a_filtered_out_child_can_still_be_re_parented() {
    matrix(|harness, children, grandchildren| {
        let write = |harness: &mut Harness, deleted: bool, parent: ObjectId| {
            harness
                .qm
                .update(
                    &mut harness.storage,
                    children[0],
                    &[
                        Value::Text("child-0".into()),
                        Value::Boolean(deleted),
                        Value::Uuid(parent),
                    ],
                )
                .expect("write child");
            harness.settle_once();
        };

        // Hide it from every array, re-home it while hidden, then show it.
        let (first, second) = (harness.parents[0], harness.parents[1]);
        write(harness, true, first);
        write(harness, true, second);
        write(harness, false, second);

        let depth = harness.depth;
        expect(
            harness,
            &[
                (
                    0,
                    vec![(children[1], nested_ids(depth, &grandchildren[1..]))],
                ),
                (
                    1,
                    vec![(children[0], nested_ids(depth, &grandchildren[..1]))],
                ),
            ],
        )
    });
}

// ============================================================================
// Depth-2 classes: a grandchild change routes through the child instance
// holding it, recursively.
// ============================================================================

/// A grandchild INSERT: no reverse-index entry exists for it, so it routes by
/// its own correlate — which names the CHILD row, and the child row is what
/// the outer instance holds.
#[test]
fn a_new_grandchild_reaches_the_nested_include() {
    nested_matrix(|harness, children, grandchildren| {
        let new_grandchild = harness
            .qm
            .insert(
                &mut harness.storage,
                GRANDCHILD_TABLE,
                &[Value::Text("grandchild-9".into()), Value::Uuid(children[1])],
            )
            .expect("insert grandchild")
            .row_id;
        harness.settle_once();
        expect(
            harness,
            &[(
                0,
                vec![
                    (children[0], vec![grandchildren[0]]),
                    (children[1], vec![grandchildren[1], new_grandchild]),
                ],
            )],
        )
    });
}

/// A grandchild CONTENT UPDATE under an unchanged child set — the nested
/// staleness class the oracle found in v13-1.
#[test]
fn an_edited_grandchild_reaches_the_nested_include() {
    nested_matrix(|harness, children, grandchildren| {
        harness
            .qm
            .update(
                &mut harness.storage,
                grandchildren[0],
                &[
                    Value::Text("grandchild-0-edited".into()),
                    Value::Uuid(children[0]),
                ],
            )
            .expect("update grandchild");
        harness.settle_once();
        assert_eq!(
            grandchild_title(harness, children[0], grandchildren[0]),
            "grandchild-0-edited",
            "the edited grandchild title must be visible two levels down"
        );
        expect(
            harness,
            &[(
                0,
                vec![
                    (children[0], vec![grandchildren[0]]),
                    (children[1], vec![grandchildren[1]]),
                ],
            )],
        )
    });
}

/// A grandchild DELETE: unloadable, so it routes only through the instances
/// whose subtree tracked it.
#[test]
fn a_deleted_grandchild_leaves_the_nested_include() {
    nested_matrix(|harness, children, grandchildren| {
        harness
            .qm
            .delete(&mut harness.storage, grandchildren[0])
            .expect("delete grandchild");
        harness.settle_once();
        expect(
            harness,
            &[(
                0,
                vec![
                    (children[0], Vec::new()),
                    (children[1], vec![grandchildren[1]]),
                ],
            )],
        )
    });
}

/// A grandchild MOVED to another child — the depth-2 correlate change. Both
/// nested instances live inside the SAME outer instance here, so the recursion
/// has to route within the child level too.
#[test]
fn a_re_homed_grandchild_moves_between_nested_includes() {
    nested_matrix(|harness, children, grandchildren| {
        harness
            .qm
            .update(
                &mut harness.storage,
                grandchildren[0],
                &[Value::Text("grandchild-0".into()), Value::Uuid(children[1])],
            )
            .expect("re-home grandchild");
        harness.settle_once();
        expect(
            harness,
            &[(
                0,
                vec![
                    (children[0], Vec::new()),
                    (children[1], vec![grandchildren[0], grandchildren[1]]),
                ],
            )],
        )
    });
}

/// A grandchild moved to a child under a DIFFERENT parent: the change has to
/// cross outer instances, which is the case a reverse index keyed only on the
/// changed row's own id cannot resolve.
#[test]
fn a_grandchild_can_move_to_a_child_of_another_parent() {
    nested_matrix(|harness, children, grandchildren| {
        let second_parent = harness.parents[1];
        let other_child = harness
            .qm
            .insert(
                &mut harness.storage,
                CHILD_TABLE,
                &[
                    Value::Text("child-9".into()),
                    Value::Boolean(false),
                    Value::Uuid(second_parent),
                ],
            )
            .expect("insert child under the second parent")
            .row_id;
        harness.settle_once();

        harness
            .qm
            .update(
                &mut harness.storage,
                grandchildren[0],
                &[Value::Text("grandchild-0".into()), Value::Uuid(other_child)],
            )
            .expect("re-home grandchild across parents");
        harness.settle_once();

        expect(
            harness,
            &[
                (
                    0,
                    vec![
                        (children[0], Vec::new()),
                        (children[1], vec![grandchildren[1]]),
                    ],
                ),
                (1, vec![(other_child, vec![grandchildren[0]])]),
            ],
        )
    });
}

// ============================================================================
// Helpers that read a value out of the mirror rather than out of the engine.
// ============================================================================

// ============================================================================
// Classes that only exist for an empty binding.
// ============================================================================

fn insert_child(harness: &mut Harness, title: &str, hidden: bool, parent: ObjectId) -> ObjectId {
    harness
        .qm
        .insert(
            &mut harness.storage,
            CHILD_TABLE,
            &[
                Value::Text(title.into()),
                Value::Boolean(hidden),
                Value::Uuid(parent),
            ],
        )
        .expect("insert child")
        .row_id
}

/// The FIRST row of a binding, hidden by the include's filter: the correlation
/// index is no longer empty, so the slot needs an instance — whose filter then
/// keeps the array empty. Flipping the row is a content update of a row that
/// instance holds, and has to show it.
#[test]
fn a_filtered_out_first_child_enters_once_it_is_flipped() {
    matrix(|harness, children, grandchildren| {
        let target = harness.parents[2];
        let hidden = insert_child(harness, "child-7", true, target);
        harness.settle_once();
        assert_eq!(
            harness.tree(),
            expect(
                harness,
                &[(0, seeded_children(harness.depth, children, grandchildren))]
            ),
            "a filtered-out child must not show in its parent's include"
        );

        harness
            .qm
            .update(
                &mut harness.storage,
                hidden,
                &[
                    Value::Text("child-7".into()),
                    Value::Boolean(false),
                    Value::Uuid(target),
                ],
            )
            .expect("flip child");
        harness.settle_once();
        expect(
            harness,
            &[
                (0, seeded_children(harness.depth, children, grandchildren)),
                (2, vec![(hidden, Vec::new())]),
            ],
        )
    });
}

/// A child hops across childless parents: each hop fills a binding that was
/// proven empty and leaves behind an instance with nothing in it.
#[test]
fn a_child_hops_across_childless_parents() {
    matrix(|harness, children, grandchildren| {
        let child = insert_child(harness, "child-7", false, harness.parents[1]);
        harness.settle_once();
        for slot in [2usize, 3, 1] {
            harness
                .qm
                .update(
                    &mut harness.storage,
                    child,
                    &[
                        Value::Text("child-7".into()),
                        Value::Boolean(false),
                        Value::Uuid(harness.parents[slot]),
                    ],
                )
                .expect("re-parent child");
            harness.settle_once();
            assert_eq!(
                harness.tree(),
                expect(
                    harness,
                    &[
                        (0, seeded_children(harness.depth, children, grandchildren)),
                        (slot, vec![(child, Vec::new())]),
                    ]
                ),
                "the child must be under parent slot {slot} and nowhere else"
            );
        }
        expect(
            harness,
            &[
                (0, seeded_children(harness.depth, children, grandchildren)),
                (1, vec![(child, Vec::new())]),
            ],
        )
    });
}

/// A parent loses every child and gets a new one: the slot goes from a filled
/// instance to an empty one and back, with deletes in between.
#[test]
fn an_emptied_parent_can_be_filled_again() {
    matrix(|harness, children, _grandchildren| {
        for child in children {
            harness
                .qm
                .delete(&mut harness.storage, *child)
                .expect("delete child");
        }
        harness.settle_once();
        assert_eq!(
            harness.tree(),
            expect(harness, &[]),
            "every include must be empty once the only children are deleted"
        );

        let reborn = insert_child(harness, "child-8", false, harness.parents[0]);
        let other = insert_child(harness, "child-9", false, harness.parents[3]);
        harness.settle_once();
        expect(
            harness,
            &[
                (0, vec![(reborn, Vec::new())]),
                (3, vec![(other, Vec::new())]),
            ],
        )
    });
}

/// Depth 2: a child that arrives with no grandchildren is an empty binding of
/// the NESTED include, inside its parent's instance. Its first grandchild has
/// to fill it — the mark travels through the outer instance to a slot that
/// holds no graph.
#[test]
fn a_first_grandchild_reaches_a_childless_child() {
    nested_matrix(|harness, children, grandchildren| {
        let lone_child = insert_child(harness, "child-5", false, harness.parents[0]);
        let far_child = insert_child(harness, "child-6", false, harness.parents[2]);
        harness.settle_once();

        let mut new_grandchildren = Vec::new();
        for child in [lone_child, far_child] {
            new_grandchildren.push(
                harness
                    .qm
                    .insert(
                        &mut harness.storage,
                        GRANDCHILD_TABLE,
                        &[Value::Text("grandchild-9".into()), Value::Uuid(child)],
                    )
                    .expect("insert grandchild")
                    .row_id,
            );
        }
        harness.settle_once();
        expect(
            harness,
            &[
                (
                    0,
                    vec![
                        (children[0], vec![grandchildren[0]]),
                        (children[1], vec![grandchildren[1]]),
                        (lone_child, vec![new_grandchildren[0]]),
                    ],
                ),
                (2, vec![(far_child, vec![new_grandchildren[1]])]),
            ],
        )
    });
}

fn child_title(harness: &Harness, child_id: ObjectId) -> String {
    let descriptor = harness.descriptor.as_ref().expect("mirror descriptor");
    for parent_id in &harness.ordered {
        let values =
            decode_row(descriptor, &harness.rows[parent_id].data).expect("decode parent row");
        for child in include_array(&values, PARENT_INCLUDE_COLUMN) {
            let (id, child_values) = included_row(child);
            if id == child_id {
                return match &child_values[0] {
                    Value::Text(title) => title.clone(),
                    other => panic!("expected a child title, got {other:?}"),
                };
            }
        }
    }
    panic!("child {child_id} is not in the mirror");
}

fn grandchild_title(harness: &Harness, child_id: ObjectId, grandchild_id: ObjectId) -> String {
    let descriptor = harness.descriptor.as_ref().expect("mirror descriptor");
    for parent_id in &harness.ordered {
        let values =
            decode_row(descriptor, &harness.rows[parent_id].data).expect("decode parent row");
        for child in include_array(&values, PARENT_INCLUDE_COLUMN) {
            let (id, child_values) = included_row(child);
            if id != child_id {
                continue;
            }
            for grandchild in include_array(child_values, CHILD_INCLUDE_COLUMN) {
                let (nested_id, nested_values) = included_row(grandchild);
                if nested_id == grandchild_id {
                    return match &nested_values[0] {
                        Value::Text(title) => title.clone(),
                        other => panic!("expected a grandchild title, got {other:?}"),
                    };
                }
            }
        }
    }
    panic!("grandchild {grandchild_id} is not in the mirror");
}
