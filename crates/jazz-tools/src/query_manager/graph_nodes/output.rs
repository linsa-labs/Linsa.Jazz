use ahash::AHashSet;
use std::collections::{HashMap, HashSet};

use crate::object::{BranchName, ObjectId};
use crate::query_manager::encoding::decode_row;
use crate::query_manager::types::{
    Row, RowDelta, RowDescriptor, Tuple, TupleDelta, TupleDescriptor, Value,
};

use super::{
    RowNode,
    tuple_delta::{compute_tuple_delta, has_tuple_content_changed},
};

/// Output mode for query results.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputMode {
    /// Emit deltas as they happen.
    Delta,
    /// Emit full result set on each change.
    Full,
}

/// Query subscription identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QuerySubscriptionId(pub u64);

/// Output node - terminal node that delivers results to subscribers.
///
/// Materializes any remaining unmaterialized elements and flattens
/// multi-element tuples to single Row outputs.
#[derive(Debug)]
pub struct OutputNode {
    descriptor: RowDescriptor,
    /// Output tuple descriptor (always fully materialized).
    output_tuple_descriptor: TupleDescriptor,
    mode: OutputMode,
    /// Current result tuples (for RowNode trait).
    current_tuples: AHashSet<Tuple>,
    /// Ordered tuples for deterministic output (preserves sort order).
    ordered_tuples: Vec<Tuple>,
    /// Current sync scope derived from output tuple provenance.
    sync_scope: HashSet<(ObjectId, BranchName)>,
    /// Reference counts for scope entries, since several output tuples can
    /// depend on the same source object.
    sync_scope_counts: HashMap<(ObjectId, BranchName), usize>,
    /// Pending tuple deltas to deliver.
    pending_tuple_deltas: Vec<TupleDelta>,
    /// True if subscriber has received initial snapshot.
    subscriber_initialized: bool,
    dirty: bool,
    /// Passes this node took the in-place path in, for tests that must tell it from an
    /// include instance's.
    #[cfg(test)]
    pub(crate) applied_in_place: usize,
}

impl OutputNode {
    /// Create an OutputNode with TupleDescriptor.
    /// The output descriptor is always fully materialized.
    pub fn with_tuple_descriptor(tuple_descriptor: TupleDescriptor, mode: OutputMode) -> Self {
        let descriptor = tuple_descriptor.combined_descriptor();
        // Output is always fully materialized
        let output_tuple_descriptor = tuple_descriptor.clone().with_all_materialized();
        Self {
            descriptor,
            output_tuple_descriptor,
            mode,
            current_tuples: AHashSet::new(),
            ordered_tuples: Vec::new(),
            sync_scope: HashSet::new(),
            sync_scope_counts: HashMap::new(),
            pending_tuple_deltas: Vec::new(),
            subscriber_initialized: false,
            dirty: true,
            #[cfg(test)]
            applied_in_place: 0,
        }
    }

    /// Get the output tuple descriptor.
    pub fn output_tuple_descriptor(&self) -> &TupleDescriptor {
        &self.output_tuple_descriptor
    }

    /// Get the output mode.
    pub fn mode(&self) -> OutputMode {
        self.mode
    }

    /// Take pending tuple deltas (for delta mode).
    pub fn take_tuple_deltas(&mut self) -> Vec<TupleDelta> {
        std::mem::take(&mut self.pending_tuple_deltas)
    }

    /// Get current result as rows (extracts from tuples).
    /// For single-table queries, converts length-1 tuples to rows.
    /// Returns rows in insertion order (preserves sort order from upstream nodes).
    pub fn current_rows(&self) -> Vec<Row> {
        self.ordered_tuples
            .iter()
            .filter_map(|t| t.to_single_row())
            .collect()
    }

    /// Decode current rows to Values (for output to user).
    pub fn decode_current(&self) -> Vec<Vec<Value>> {
        self.current_rows()
            .iter()
            .filter_map(|row| decode_row(&self.descriptor, &row.data).ok())
            .collect()
    }

    /// Whether this output has delivered its first list. Until it has, an
    /// empty input delta is no proof the list is current (see
    /// `query_manager::settle_skips`).
    pub(crate) fn has_materialized(&self) -> bool {
        self.subscriber_initialized
    }

    /// Whether this output's list can be updated in place: it has materialized, and it holds
    /// no id twice. A duplicate is reported as moved by every rebuild (`compute_tuple_delta`
    /// keeps the last position of an id), which no in-place update reproduces. The set is keyed
    /// by ids, so it is shorter than the list exactly then.
    pub(crate) fn can_update_in_place(&self) -> bool {
        self.has_materialized() && self.current_tuples.len() == self.ordered_tuples.len()
    }

    /// Ordered tuples as received from upstream nodes.
    pub fn ordered_tuples(&self) -> &[Tuple] {
        &self.ordered_tuples
    }

    /// Current sync scope derived incrementally from output tuple provenance.
    pub fn sync_scope(&self) -> &HashSet<(ObjectId, BranchName)> {
        &self.sync_scope
    }

    fn add_tuple_to_scope(&mut self, tuple: &Tuple) {
        for scoped_object in tuple.provenance().iter().copied() {
            let count = self.sync_scope_counts.entry(scoped_object).or_insert(0);
            *count += 1;
            if *count == 1 {
                self.sync_scope.insert(scoped_object);
            }
        }
    }

    fn remove_tuple_from_scope(&mut self, tuple: &Tuple) {
        for scoped_object in tuple.provenance() {
            if let Some(count) = self.sync_scope_counts.get_mut(scoped_object) {
                *count -= 1;
                if *count == 0 {
                    self.sync_scope_counts.remove(scoped_object);
                    self.sync_scope.remove(scoped_object);
                }
            }
        }
    }

    fn apply_delta_to_scope(&mut self, delta: &TupleDelta) {
        for tuple in &delta.removed {
            self.remove_tuple_from_scope(tuple);
        }
        for (old_tuple, new_tuple) in &delta.updated {
            self.remove_tuple_from_scope(old_tuple);
            self.add_tuple_to_scope(new_tuple);
        }
        for tuple in &delta.added {
            self.add_tuple_to_scope(tuple);
        }
    }

    /// The update-only pass of `process_with_ordered_input` when the ordered input kept
    /// the list's ids: replace the updated tuples where they stand.
    pub(crate) fn apply_in_place(&mut self, updates: Vec<(usize, Tuple)>) -> TupleDelta {
        #[cfg(test)]
        {
            self.applied_in_place += 1;
        }
        let mut delta = TupleDelta::new();
        for (position, new) in updates {
            let old = std::mem::replace(&mut self.ordered_tuples[position], new.clone());
            self.current_tuples.replace(new.clone());
            if has_tuple_content_changed(&old, &new) {
                delta.updated.push((old, new));
            }
        }
        self.apply_delta_to_scope(&delta);
        self.dirty = false;
        self.subscriber_initialized = true;
        if !delta.is_empty() {
            self.pending_tuple_deltas.push(delta.clone());
        }
        delta
    }

    /// Rebuild ordered output from a full ordered upstream input.
    pub fn process_with_ordered_input(&mut self, ordered_tuples: &[Tuple]) -> TupleDelta {
        #[cfg(test)]
        crate::query_manager::ordered_in_place::note_rebuilt_tuples(ordered_tuples.len());
        let delta = if self.ordered_tuples.is_empty() {
            TupleDelta {
                added: ordered_tuples.to_vec(),
                removed: Vec::new(),
                moved: Vec::new(),
                updated: Vec::new(),
            }
        } else {
            compute_tuple_delta(&self.ordered_tuples, ordered_tuples)
        };

        self.apply_delta_to_scope(&delta);
        self.ordered_tuples = ordered_tuples.to_vec();
        self.current_tuples = self.ordered_tuples.iter().cloned().collect();
        self.dirty = false;
        self.subscriber_initialized = true;

        if !delta.is_empty() {
            self.pending_tuple_deltas.push(delta.clone());
        }
        delta
    }

    /// Decode a delta to Values.
    pub fn decode_delta(&self, delta: &RowDelta) -> DecodedDelta {
        DecodedDelta {
            added: delta
                .added
                .iter()
                .filter_map(|row| {
                    decode_row(&self.descriptor, &row.data)
                        .ok()
                        .map(|v| (row.id, v))
                })
                .collect(),
            removed: delta
                .removed
                .iter()
                .filter_map(|row| {
                    decode_row(&self.descriptor, &row.data)
                        .ok()
                        .map(|v| (row.id, v))
                })
                .collect(),
            updated: delta
                .updated
                .iter()
                .filter_map(|(old, new)| {
                    let old_v = decode_row(&self.descriptor, &old.data).ok()?;
                    let new_v = decode_row(&self.descriptor, &new.data).ok()?;
                    Some((old.id, old_v, new_v))
                })
                .collect(),
        }
    }
}

/// Decoded delta with Values instead of binary.
#[derive(Debug, Clone)]
pub struct DecodedDelta {
    pub added: Vec<(crate::object::ObjectId, Vec<Value>)>,
    pub removed: Vec<(crate::object::ObjectId, Vec<Value>)>,
    pub updated: Vec<(crate::object::ObjectId, Vec<Value>, Vec<Value>)>,
}

impl RowNode for OutputNode {
    fn output_descriptor(&self) -> &RowDescriptor {
        &self.descriptor
    }

    fn process(&mut self, input: TupleDelta) -> TupleDelta {
        self.apply_delta_to_scope(&input);

        // Apply changes to current_tuples and ordered_tuples
        for tuple in &input.removed {
            self.current_tuples.remove(tuple);
            self.ordered_tuples.retain(|t| t != tuple);
        }

        for tuple in &input.added {
            self.current_tuples.insert(tuple.clone());
            self.ordered_tuples.push(tuple.clone());
        }

        for tuple in &input.moved {
            if let Some(pos) = self.ordered_tuples.iter().position(|t| t == tuple) {
                let existing = self.ordered_tuples.remove(pos);
                self.ordered_tuples.push(existing);
            }
        }

        for (old_tuple, new_tuple) in &input.updated {
            self.current_tuples.remove(old_tuple);
            self.current_tuples.insert(new_tuple.clone());
            // Update in place in ordered_tuples to preserve position
            if let Some(pos) = self.ordered_tuples.iter().position(|t| t == old_tuple) {
                self.ordered_tuples[pos] = new_tuple.clone();
            }
        }

        self.dirty = false;

        // Deliver immediately
        self.subscriber_initialized = true;
        if !input.is_empty() {
            self.pending_tuple_deltas.push(input.clone());
        }

        input
    }

    fn current_tuples(&self) -> &AHashSet<Tuple> {
        &self.current_tuples
    }

    fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    fn is_dirty(&self) -> bool {
        self.dirty
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::ObjectId;
    use crate::query_manager::encoding::encode_row;
    use crate::query_manager::types::{ColumnDescriptor, ColumnType, TupleElement};

    fn test_descriptor() -> RowDescriptor {
        RowDescriptor::new(vec![
            ColumnDescriptor::new("id", ColumnType::Integer),
            ColumnDescriptor::new("name", ColumnType::Text),
        ])
    }

    fn make_tuple(id: ObjectId, n: i32, name: &str) -> Tuple {
        let descriptor = test_descriptor();
        let data = encode_row(&descriptor, &[Value::Integer(n), Value::Text(name.into())]).unwrap();
        Tuple::new(vec![TupleElement::Row {
            id,
            content: data.into(),
            batch_id: crate::row_histories::BatchId([0; 16]),
            row_provenance: crate::metadata::RowProvenance::for_insert("jazz:test", 0),
        }])
    }

    fn make_output_node(mode: OutputMode) -> OutputNode {
        let descriptor = test_descriptor();
        let tuple_desc = TupleDescriptor::single_with_materialization("", descriptor, true);
        OutputNode::with_tuple_descriptor(tuple_desc, mode)
    }

    #[test]
    fn output_stores_deltas() {
        let mut node = make_output_node(OutputMode::Delta);

        let id1 = ObjectId::new();
        let tuple1 = make_tuple(id1, 1, "Alice");

        let delta = TupleDelta {
            added: vec![tuple1],
            removed: vec![],
            moved: vec![],
            updated: vec![],
        };

        node.process(delta);

        let deltas = node.take_tuple_deltas();
        assert_eq!(deltas.len(), 1);
        assert_eq!(deltas[0].added.len(), 1);
    }

    // Scenario: a list that holds one id twice (a join that emitted the same pair twice) is
    // never updated in place: every rebuild reports the duplicate as moved.
    #[test]
    fn a_list_holding_one_id_twice_is_never_updated_in_place() {
        let mut node = make_output_node(OutputMode::Delta);
        let (a, b) = (ObjectId::new(), ObjectId::new());
        node.process_with_ordered_input(&[make_tuple(a, 1, "Alice"), make_tuple(b, 2, "Bob")]);
        assert!(
            node.can_update_in_place(),
            "fixture: a materialized list of distinct ids can"
        );

        node.process_with_ordered_input(&[
            make_tuple(a, 1, "Alice"),
            make_tuple(a, 1, "Alicia"),
            make_tuple(b, 2, "Bob"),
        ]);

        assert!(!node.can_update_in_place());
    }

    // Scenario: an in-place pass replaces the tuples where they stand, in both of the
    // node's views, and reports only the ones whose content changed.
    //
    // ASCII:
    // ordered:   [id1 -> (1, "Alice"), id2 -> (2, "Bob")]
    // in place:  [0 -> (1, "Alice") unchanged, 1 -> (2, "Robert")]
    // delta:     updated [id2]
    #[test]
    fn in_place_updates_both_views_and_reports_only_changed_tuples() {
        let mut node = make_output_node(OutputMode::Delta);
        let (id1, id2) = (ObjectId::new(), ObjectId::new());
        node.process_with_ordered_input(&[make_tuple(id1, 1, "Alice"), make_tuple(id2, 2, "Bob")]);
        node.take_tuple_deltas();

        let renamed = make_tuple(id2, 2, "Robert");
        let delta =
            node.apply_in_place(vec![(0, make_tuple(id1, 1, "Alice")), (1, renamed.clone())]);

        assert!(delta.added.is_empty() && delta.removed.is_empty() && delta.moved.is_empty());
        let updated: Vec<_> = delta
            .updated
            .iter()
            .map(|(_, new)| new.first_id())
            .collect();
        assert_eq!(
            updated,
            vec![Some(id2)],
            "only the renamed tuple is reported"
        );
        assert_eq!(
            node.ordered_tuples()[1].get(0).and_then(|e| e.content()),
            renamed.get(0).and_then(|e| e.content())
        );
        let held = node
            .current_tuples()
            .get(&renamed)
            .expect("the renamed tuple is held");
        assert_eq!(
            held.get(0).and_then(|e| e.content()),
            renamed.get(0).and_then(|e| e.content()),
            "the unordered view holds the new content"
        );
        assert_eq!(node.take_tuple_deltas().len(), 1);
    }

    // Scenario: an in-place pass over a tuple whose data is as it was but whose provenance
    // moved to another branch (a writer on a second live schema branch, its values lens-equal)
    // reports it, and the sync scope follows it.
    //
    // ASCII:
    // ordered:   [id -> (1, "Alice") from a]
    // in place:  [0 -> (1, "Alice") from b]
    // delta:     updated [id]; scope {(id, a)} -> {(id, b)}
    #[test]
    fn in_place_reports_a_provenance_only_change_and_moves_the_scope() {
        let mut node = make_output_node(OutputMode::Delta);
        let id = ObjectId::new();
        let (a, b) = (BranchName::new("a"), BranchName::new("b"));
        let from = |branch: BranchName| {
            make_tuple(id, 1, "Alice").with_provenance([(id, branch)].into_iter().collect())
        };
        node.process_with_ordered_input(&[from(a)]);
        node.take_tuple_deltas();
        assert!(node.sync_scope().contains(&(id, a)), "fixture");

        let delta = node.apply_in_place(vec![(0, from(b))]);

        assert_eq!(
            delta.updated.len(),
            1,
            "a provenance-only change is an update"
        );
        assert!(node.sync_scope().contains(&(id, b)));
        assert!(!node.sync_scope().contains(&(id, a)));
    }

    // Scenario: an in-place pass over a tuple whose only change is its batch provenance (a
    // nested row rewritten as it was) reports it.
    #[test]
    fn in_place_reports_a_batch_provenance_only_change() {
        let mut node = make_output_node(OutputMode::Delta);
        let id = ObjectId::new();
        let in_batch = |byte: u8| {
            make_tuple(id, 1, "Alice").with_batch_provenance(
                [crate::row_histories::BatchId([byte; 16])]
                    .into_iter()
                    .collect(),
            )
        };
        node.process_with_ordered_input(&[in_batch(1)]);
        node.take_tuple_deltas();

        let delta = node.apply_in_place(vec![(0, in_batch(2))]);

        assert_eq!(delta.updated.len(), 1, "a batch-only change is an update");
        assert_eq!(
            node.ordered_tuples()[0].batch_provenance(),
            in_batch(2).batch_provenance()
        );
    }

    // Scenario: current rows decode to user values.
    //
    // ASCII:
    // tuples:   [id1 -> (1, "Alice")]
    // decoded:  [[1, "Alice"]]
    #[test]
    fn output_decodes_current() {
        let mut node = make_output_node(OutputMode::Full);

        let id1 = ObjectId::new();
        let tuple1 = make_tuple(id1, 1, "Alice");

        node.process(TupleDelta {
            added: vec![tuple1],
            removed: vec![],
            moved: vec![],
            updated: vec![],
        });

        let decoded = node.decode_current();
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0][0], Value::Integer(1));
        assert_eq!(decoded[0][1], Value::Text("Alice".into()));
    }

    // Scenario: decode_delta maps row bytes to typed value vectors.
    //
    // ASCII:
    // row_delta.added: [id1]
    // decoded.added:   [(id1, [1, "Alice"])]
    #[test]
    fn output_decodes_delta() {
        let node = make_output_node(OutputMode::Delta);

        let id1 = ObjectId::new();
        let row1 = Row::new(
            id1,
            encode_row(
                &test_descriptor(),
                &[Value::Integer(1), Value::Text("Alice".into())],
            )
            .unwrap(),
            crate::row_histories::BatchId([0; 16]),
            crate::metadata::RowProvenance::for_insert("jazz:test", 0),
        );

        let delta = RowDelta {
            added: vec![row1],
            removed: vec![],
            moved: vec![],
            updated: vec![],
        };

        let decoded = node.decode_delta(&delta);
        assert_eq!(decoded.added.len(), 1);
        assert_eq!(decoded.added[0].0, id1);
        assert_eq!(decoded.added[0].1[0], Value::Integer(1));
        assert_eq!(decoded.added[0].1[1], Value::Text("Alice".into()));
    }

    // Scenario: empty deltas are not buffered for delivery.
    //
    // ASCII:
    // input_delta:   empty
    // pending_queue: []
    #[test]
    fn empty_delta_not_stored() {
        let mut node = make_output_node(OutputMode::Delta);

        let delta = TupleDelta::new();
        node.process(delta);

        let deltas = node.take_tuple_deltas();
        assert!(deltas.is_empty());
    }

    // Scenario: each non-empty process call is delivered immediately.
    //
    // ASCII:
    // tick1: +A -> deliver [deltaA]
    // tick2: +B -> deliver [deltaB]
    #[test]
    fn output_delivers_immediately() {
        let mut node = make_output_node(OutputMode::Delta);

        let id1 = ObjectId::new();
        let id2 = ObjectId::new();
        let tuple1 = make_tuple(id1, 1, "Alice");
        let tuple2 = make_tuple(id2, 2, "Bob");

        // First delta
        let delta1 = TupleDelta {
            added: vec![tuple1],
            removed: vec![],
            moved: vec![],
            updated: vec![],
        };
        node.process(delta1);

        // Should deliver immediately
        let deltas = node.take_tuple_deltas();
        assert_eq!(deltas.len(), 1);
        assert_eq!(deltas[0].added.len(), 1);

        // Second delta
        let delta2 = TupleDelta {
            added: vec![tuple2],
            removed: vec![],
            moved: vec![],
            updated: vec![],
        };
        node.process(delta2);

        // Should also deliver immediately
        let deltas = node.take_tuple_deltas();
        assert_eq!(deltas.len(), 1);
        assert_eq!(deltas[0].added.len(), 1);
    }

    #[test]
    fn sync_scope_is_maintained_incrementally_with_shared_provenance() {
        let mut node = make_output_node(OutputMode::Delta);
        let branch = BranchName::new("main");
        let shared_id = ObjectId::new();
        let id1 = ObjectId::new();
        let id2 = ObjectId::new();

        let tuple1 = Tuple::new_with_provenance(
            vec![TupleElement::Id(id1)],
            [(shared_id, branch), (id1, branch)].into_iter().collect(),
        );
        let tuple2 = Tuple::new_with_provenance(
            vec![TupleElement::Id(id2)],
            [(shared_id, branch), (id2, branch)].into_iter().collect(),
        );

        node.process(TupleDelta {
            added: vec![tuple1.clone(), tuple2.clone()],
            removed: vec![],
            moved: vec![],
            updated: vec![],
        });
        assert_eq!(node.sync_scope().len(), 3);
        assert!(node.sync_scope().contains(&(shared_id, branch)));
        assert!(node.sync_scope().contains(&(id1, branch)));
        assert!(node.sync_scope().contains(&(id2, branch)));

        node.process(TupleDelta {
            added: vec![],
            removed: vec![tuple1],
            moved: vec![],
            updated: vec![],
        });
        assert_eq!(node.sync_scope().len(), 2);
        assert!(node.sync_scope().contains(&(shared_id, branch)));
        assert!(!node.sync_scope().contains(&(id1, branch)));
        assert!(node.sync_scope().contains(&(id2, branch)));

        let tuple2_without_shared = Tuple::new_with_provenance(
            vec![TupleElement::Id(id2)],
            [(id2, branch)].into_iter().collect(),
        );
        node.process(TupleDelta {
            added: vec![],
            removed: vec![],
            moved: vec![],
            updated: vec![(tuple2, tuple2_without_shared)],
        });
        assert_eq!(node.sync_scope().len(), 1);
        assert!(!node.sync_scope().contains(&(shared_id, branch)));
        assert!(node.sync_scope().contains(&(id2, branch)));
    }

    #[test]
    fn ordered_input_insert_does_not_mark_existing_as_moved() {
        let mut node = make_output_node(OutputMode::Delta);
        let ids: Vec<_> = (0..4).map(|_| ObjectId::new()).collect();
        let base: Vec<_> = ids
            .iter()
            .take(3)
            .enumerate()
            .map(|(i, id)| make_tuple(*id, i as i32, &format!("Row{}", i)))
            .collect();

        node.process_with_ordered_input(&base);

        let inserted = make_tuple(ids[3], 99, "Inserted");
        let delta = node.process_with_ordered_input(&[
            inserted.clone(),
            base[0].clone(),
            base[1].clone(),
            base[2].clone(),
        ]);

        assert_eq!(delta.added.len(), 1);
        assert!(delta.removed.is_empty());
        assert!(delta.moved.is_empty());
    }

    #[test]
    fn ordered_input_remove_does_not_mark_following_as_moved() {
        let mut node = make_output_node(OutputMode::Delta);
        let ids: Vec<_> = (0..4).map(|_| ObjectId::new()).collect();
        let tuples: Vec<_> = ids
            .iter()
            .enumerate()
            .map(|(i, id)| make_tuple(*id, i as i32, &format!("Row{}", i)))
            .collect();

        node.process_with_ordered_input(&tuples);

        let delta = node.process_with_ordered_input(&[
            tuples[0].clone(),
            tuples[2].clone(),
            tuples[3].clone(),
        ]);

        assert_eq!(delta.removed.len(), 1);
        assert_eq!(delta.removed[0].first_id(), tuples[1].first_id());
        assert!(delta.added.is_empty());
        assert!(delta.moved.is_empty());
    }

    #[test]
    fn ordered_input_rotation_marks_only_reordered_tuple_as_moved() {
        let mut node = make_output_node(OutputMode::Delta);
        let ids: Vec<_> = (0..3).map(|_| ObjectId::new()).collect();
        let tuples: Vec<_> = ids
            .iter()
            .enumerate()
            .map(|(i, id)| make_tuple(*id, i as i32, &format!("Row{}", i)))
            .collect();

        node.process_with_ordered_input(&tuples);

        let delta = node.process_with_ordered_input(&[
            tuples[1].clone(),
            tuples[2].clone(),
            tuples[0].clone(),
        ]);

        assert!(delta.added.is_empty());
        assert!(delta.removed.is_empty());
        assert_eq!(delta.moved.len(), 1);
        assert_eq!(delta.moved[0].first_id(), tuples[0].first_id());
    }
}
