use ahash::{AHashMap, AHashSet};
use smallvec::SmallVec;
use std::collections::HashMap;

use crate::object::ObjectId;
use crate::query_manager::graph_nodes::tuple_delta::has_tuple_content_changed;
use crate::query_manager::ordered_in_place::{InPlaceMode, in_place_mode};
use crate::query_manager::settle_cost::{ORDERED_IN_PLACE, ORDERED_IN_PLACE_PARITY_VIOLATIONS};
use crate::query_manager::settle_skips::SkipMode;
use crate::query_manager::settlement_eval_cache::SettlementEvalCache;
use crate::query_manager::types::{
    LoadedRow, RowDelta, TableName, Tuple, TupleDelta, TupleElement,
};
use crate::storage::Storage;
use crate::sync_manager::RowBatchKey;

use super::super::graph_nodes::{NodeId, RowNode, SourceContext, SourceNode, TransformNode};
use super::{GraphNode, QueryGraph};

/// Settle passes one settle may add to grow ordered windows (each doubles the walk).
pub(crate) const MAX_WINDOW_GROWTHS: usize = 24;

impl QueryGraph {
    /// Mark a node as dirty using the bitmap.
    pub fn mark_dirty(&mut self, id: NodeId) {
        let idx = id.0 as usize;
        if idx >= self.dirty_bitmap.len() {
            self.dirty_bitmap.resize(idx + 1, false);
        }
        self.dirty_bitmap.set(idx, true);
    }

    /// Check if a node is dirty.
    pub(super) fn is_dirty(&self, id: NodeId) -> bool {
        let idx = id.0 as usize;
        idx < self.dirty_bitmap.len() && self.dirty_bitmap[idx]
    }

    /// Check if any nodes are dirty.
    pub fn has_dirty_nodes(&self) -> bool {
        self.dirty_bitmap.any()
    }

    /// Clear all dirty flags.
    pub fn clear_dirty(&mut self) {
        self.dirty_bitmap.fill(false);
    }

    /// Force a full graph recompute.
    pub fn mark_all_dirty(&mut self) {
        self.dirty_bitmap.fill(true);
    }

    /// Every row id this graph's scans currently hold, recursing into the
    /// cached instances of any include it contains, plus the ids a forward
    /// reference's instances probe (found or not), so an insert of a missing
    /// target routes to the instance waiting for it.
    ///
    /// This is the source of include routing's reverse index (v14 L1). It is
    /// deliberately PRE-filter and PRE-window: an instance whose scan holds a
    /// row must be re-checked when that row changes even if the row never
    /// reaches its array — otherwise a row moving to a different parent leaves
    /// the old instance's incremental scan baseline permanently ahead of a
    /// full rescan.
    pub(crate) fn collect_scanned_row_ids(&self, out: &mut Vec<ObjectId>) {
        for compact in &self.nodes {
            match &compact.node {
                GraphNode::IndexScan(scan) => out.extend(scan.scanned_row_ids()),
                GraphNode::ArraySubquery(node) => node.extend_routed_row_ids(out),
                _ => {}
            }
        }
    }

    /// Mark index scan nodes dirty for a given table/column.
    /// Also propagates dirty marks to downstream nodes.
    pub fn mark_dirty_for_column(&mut self, table: &str, column: &str) {
        let affected: Vec<NodeId> = self
            .index_scan_nodes
            .iter()
            .filter(|(_, t, c)| {
                t.as_str() == table && (c.as_str() == column || c.as_str() == "_id")
            })
            .map(|(node_id, _, _)| *node_id)
            .collect();
        for node_id in affected {
            if let Some(GraphNode::IndexScan(node)) = self.get_node_mut(node_id) {
                node.note_table_dirty();
            }
            self.mark_dirty(node_id);
            self.mark_downstream_dirty(node_id);
        }
    }

    /// Row-precise variant of [`Self::mark_dirty_for_table`]: scans learn exactly
    /// which rows changed and the next settle re-evaluates only those instead of
    /// rescanning the whole index. Every non-scan dependent (array subqueries, policy
    /// filters, magic columns, recursive relations) is dirtied exactly as the
    /// table-level path does.
    pub fn mark_rows_changed_for_table(&mut self, table: &str, ids: &AHashSet<ObjectId>) {
        let affected_index_scans: Vec<NodeId> = self
            .index_scan_nodes
            .iter()
            .filter_map(|(node_id, t, _)| {
                if t.as_str() == table {
                    Some(*node_id)
                } else {
                    None
                }
            })
            .collect();

        for node_id in affected_index_scans {
            if let Some(GraphNode::IndexScan(node)) = self.get_node_mut(node_id) {
                node.note_rows_changed(ids.iter());
            }
            self.mark_dirty(node_id);
            self.mark_downstream_dirty(node_id);
        }

        self.mark_table_dependents_dirty(table, Some(ids));
    }

    /// Mark all index scan nodes for a table dirty.
    /// Also marks array/recursive subquery nodes dirty if the table is their inner table.
    /// Also marks PolicyFilter nodes dirty if the table is INHERITS-referenced.
    pub fn mark_dirty_for_table(&mut self, table: &str) {
        // Mark index scan nodes and propagate downstream
        let affected_index_scans: Vec<NodeId> = self
            .index_scan_nodes
            .iter()
            .filter_map(|(node_id, t, _)| {
                if t.as_str() == table {
                    Some(*node_id)
                } else {
                    None
                }
            })
            .collect();

        for node_id in affected_index_scans {
            if let Some(GraphNode::IndexScan(node)) = self.get_node_mut(node_id) {
                node.note_table_dirty();
            }
            self.mark_dirty(node_id);
            self.mark_downstream_dirty(node_id);
        }

        self.mark_table_dependents_dirty(table, None);
    }

    /// The non-scan tail shared by [`Self::mark_dirty_for_table`] and
    /// [`Self::mark_rows_changed_for_table`].
    ///
    /// `changed_rows` carries the row-precise changed ids when the caller has
    /// them (F2, include-plan-sharing design §5/§9): array subquery nodes
    /// forward them into their reused subgraph instances so the next settle
    /// re-evaluates only the touched rows instead of `mark_all_dirty`-ing
    /// every instance. `None` (table-level dirt with no row information)
    /// forwards a table-level full-rescan mark instead
    /// (`note_inner_table_dirty`).
    fn mark_table_dependents_dirty(
        &mut self,
        table: &str,
        changed_rows: Option<&AHashSet<ObjectId>>,
    ) {
        // Mark array subquery nodes whose inner table changed
        // Collect node_ids first to avoid borrow conflict
        let affected_array_subqueries: Vec<NodeId> = self
            .array_subquery_tables
            .iter()
            .filter_map(|(node_id, inner_table)| {
                if inner_table.as_str() == table {
                    Some(*node_id)
                } else {
                    None
                }
            })
            .collect();

        for node_id in affected_array_subqueries {
            self.mark_dirty(node_id);
            // Mark the node as needing inner re-evaluation. With row-precise
            // ids the node also threads membership marks into its cached
            // subgraph instances (recursively — nested include tables are
            // registered against the outer node, and the instance graphs
            // route them onward through their own dependents).
            if let Some(GraphNode::ArraySubquery(node)) = self.get_node_mut(node_id) {
                match changed_rows {
                    Some(ids) => node.note_inner_rows_changed(table, ids),
                    None => node.note_inner_table_dirty(table),
                }
            }
            // Propagate dirty marks to downstream nodes (Output, etc.)
            self.mark_downstream_dirty(node_id);
        }

        // Mark PolicyFilter nodes whose policy dependency tables changed.
        //
        // DELIBERATE ASYMMETRY (F2, design §9): policy filters, magic columns
        // and recursive relations keep the coarse table-level channel —
        // `changed_rows` is not threaded into them. Their re-evaluation is a
        // policy/dependency re-check over rows the node already holds (point
        // reads at eval time), not a per-outer-row plan settle, so the F2
        // pathology (O(rows x plan) work per tick) does not apply; making
        // them row-precise is F3 scope (reverse correlation index).
        let affected_policy_filters: Vec<NodeId> = self
            .policy_filter_tables
            .iter()
            .filter_map(|(node_id, inherits_table)| {
                if inherits_table.as_str() == table {
                    Some(*node_id)
                } else {
                    None
                }
            })
            .collect();

        for node_id in affected_policy_filters {
            self.mark_dirty(node_id);
            // Mark the node as needing policy re-evaluation
            if let Some(GraphNode::PolicyFilter(node)) = self.get_node_mut(node_id) {
                node.mark_inherits_dirty();
            }
            // Propagate dirty marks to downstream nodes
            self.mark_downstream_dirty(node_id);
        }

        let affected_magic_columns: Vec<NodeId> = self
            .magic_column_tables
            .iter()
            .filter_map(|(node_id, dependency_table)| {
                if dependency_table.as_str() == table {
                    Some(*node_id)
                } else {
                    None
                }
            })
            .collect();

        for node_id in affected_magic_columns {
            self.mark_dirty(node_id);
            if let Some(GraphNode::MagicColumns(node)) = self.get_node_mut(node_id) {
                node.mark_dependency_dirty();
            }
            self.mark_downstream_dirty(node_id);
        }

        // Mark RecursiveRelation nodes whose step table changed
        let affected_recursive_relations: Vec<NodeId> = self
            .recursive_relation_tables
            .iter()
            .filter_map(|(node_id, step_table)| {
                if step_table.as_str() == table {
                    Some(*node_id)
                } else {
                    None
                }
            })
            .collect();

        for node_id in affected_recursive_relations {
            self.mark_dirty(node_id);
            if let Some(GraphNode::RecursiveRelation(node)) = self.get_node_mut(node_id) {
                node.mark_inner_dirty();
            }
            self.mark_downstream_dirty(node_id);
        }
    }

    /// Check if this graph involves a table (as index scan, array subquery inner table, or INHERITS reference).
    pub fn involves_table(&self, table: &str) -> bool {
        self.index_scan_nodes
            .iter()
            .any(|(_, t, _)| t.as_str() == table)
            || self
                .array_subquery_tables
                .iter()
                .any(|(_, t)| t.as_str() == table)
            || self
                .policy_filter_tables
                .iter()
                .any(|(_, t)| t.as_str() == table)
            || self
                .magic_column_tables
                .iter()
                .any(|(_, t)| t.as_str() == table)
            || self
                .recursive_relation_tables
                .iter()
                .any(|(_, t)| t.as_str() == table)
    }

    /// Check if this graph uses a specific index (table + column combination).
    pub fn uses_index(&self, table: &str, column: &str) -> bool {
        self.index_scan_nodes
            .iter()
            .any(|(_, t, c)| t.as_str() == table && c.as_str() == column)
    }

    /// Array subquery nodes that read `table` — directly or through a nested
    /// include, both registered at compile time (`compile.rs`, the
    /// `nested_stack` walk).
    ///
    /// The content and removal channels below are otherwise TABLE-BLIND, and
    /// blind forwarding is what made a write to any table in the subscription
    /// touch every include instance: a presence heartbeat on `users` walked
    /// every cached subgraph of every include in the graph, none of which can
    /// hold a `users` row. A node not registered for `table` cannot have an
    /// instance holding one of its rows, so skipping it is exact, not
    /// heuristic.
    fn array_subquery_nodes_for_table(&self, table: &str) -> SmallVec<[NodeId; 2]> {
        self.array_subquery_tables
            .iter()
            .filter(|(_, inner_table)| inner_table.as_str() == table)
            .map(|(node_id, _)| *node_id)
            .collect()
    }

    /// Mark a row ID as updated for content checking.
    /// This tells MaterializeNodes to check if the row's content has changed.
    /// With precise dirtiness enabled, array subquery nodes reading `table`
    /// buffer the mark for their cached subgraph instances so include-inner
    /// materializers re-load rows they already hold (the staleness fix —
    /// FINDING manifestation 1 in `manager_tests/subscription_output_oracle.rs`).
    ///
    /// Materializers keep the table-blind contract (they hold rows from every
    /// table the graph reads and filter by id); only the array-subquery
    /// forwarding is table-scoped, see [`Self::array_subquery_nodes_for_table`].
    /// Returns whether any node was marked.
    pub fn mark_rows_updated(&mut self, table: &str, ids: &AHashSet<ObjectId>) -> bool {
        if ids.is_empty() {
            return false;
        }
        let subquery_nodes = self.array_subquery_nodes_for_table(table);

        let marked_node_ids: Vec<NodeId> = self
            .nodes
            .iter_mut()
            .enumerate()
            .filter_map(|(idx, compact)| match &mut compact.node {
                GraphNode::Materialize(mat_node) => {
                    let mut tracked = false;
                    for id in ids {
                        tracked |= mat_node.mark_updated(*id);
                    }
                    tracked.then_some(NodeId(idx as u64))
                }
                GraphNode::ArraySubquery(subquery_node)
                    if subquery_nodes.contains(&NodeId(idx as u64)) =>
                {
                    subquery_node
                        .forward_rows_updated(table, ids)
                        .then_some(NodeId(idx as u64))
                }
                _ => None,
            })
            .collect();

        let any_marked = !marked_node_ids.is_empty();
        for node_id in marked_node_ids {
            self.mark_dirty(node_id);
            self.mark_downstream_dirty(node_id);
        }
        any_marked
    }

    /// Mark row IDs as deleted for removal delta emission.
    /// This tells MaterializeNodes to emit removal deltas for these rows.
    /// Array-subquery forwarding is table-scoped exactly as in
    /// [`Self::mark_rows_updated`] (include-inner rows are held by the instance
    /// materializers, not by this graph's own).
    pub fn mark_rows_deleted(&mut self, table: &str, ids: &AHashSet<ObjectId>) -> bool {
        if ids.is_empty() {
            return false;
        }
        let subquery_nodes = self.array_subquery_nodes_for_table(table);

        let marked_node_ids: Vec<NodeId> = self
            .nodes
            .iter_mut()
            .enumerate()
            .filter_map(|(idx, compact)| match &mut compact.node {
                GraphNode::Materialize(mat_node) => {
                    let mut tracked = false;
                    for id in ids {
                        tracked |= mat_node.mark_deleted(*id);
                    }
                    tracked.then_some(NodeId(idx as u64))
                }
                GraphNode::ArraySubquery(subquery_node)
                    if subquery_nodes.contains(&NodeId(idx as u64)) =>
                {
                    subquery_node
                        .forward_rows_deleted(table, ids)
                        .then_some(NodeId(idx as u64))
                }
                _ => None,
            })
            .collect();

        let any_marked = !marked_node_ids.is_empty();
        for node_id in marked_node_ids {
            self.mark_dirty(node_id);
            self.mark_downstream_dirty(node_id);
        }
        any_marked
    }

    /// Mark all nodes that depend on the given node as dirty (propagate forward).
    fn mark_downstream_dirty(&mut self, node_id: NodeId) {
        if let Some(outputs) = self.get_outputs(node_id) {
            let parents: SmallVec<[NodeId; 2]> = outputs.iter().copied().collect();
            for parent in parents {
                // Only recurse if not already dirty (avoid infinite loops)
                if !self.is_dirty(parent) {
                    self.mark_dirty(parent);
                    // Recursively mark parents of parent
                    self.mark_downstream_dirty(parent);
                }
            }
        }
    }

    /// Topological sort of dirty nodes (dependencies first).
    fn topo_sort_dirty(&self) -> Vec<NodeId> {
        let mut result = Vec::new();
        let mut visited = AHashSet::new();

        fn visit(
            node: NodeId,
            graph: &QueryGraph,
            visited: &mut AHashSet<NodeId>,
            result: &mut Vec<NodeId>,
        ) {
            if visited.contains(&node) {
                return;
            }
            visited.insert(node);

            // Visit dependencies first (inputs)
            if let Some(compact) = graph.nodes.get(node.0 as usize) {
                for dep in &compact.inputs {
                    visit(*dep, graph, visited, result);
                }
            }

            result.push(node);
        }

        // Iterate over dirty nodes using BitVec's iter_ones()
        for idx in self.dirty_bitmap.iter_ones() {
            visit(NodeId(idx as u64), self, &mut visited, &mut result);
        }

        result
    }

    fn process_limit_offset_with_ordered_sort_input(
        &mut self,
        node_id: NodeId,
        input_node: NodeId,
    ) -> Option<TupleDelta> {
        let node_idx = node_id.0 as usize;
        let input_idx = input_node.0 as usize;

        if node_idx == input_idx || node_idx >= self.nodes.len() || input_idx >= self.nodes.len() {
            return None;
        }

        if input_idx < node_idx {
            let (before_node, from_node) = self.nodes.split_at_mut(node_idx);
            let input = &before_node[input_idx].node;
            let node = &mut from_node[0].node;
            return match (input, node) {
                (GraphNode::Sort(sort_node), GraphNode::LimitOffset(limit_offset_node)) => {
                    Some(limit_offset_node.process_with_ordered_input(sort_node.sorted_tuples()))
                }
                _ => None,
            };
        }

        let (before_input, from_input) = self.nodes.split_at_mut(input_idx);
        let node = &mut before_input[node_idx].node;
        let input = &from_input[0].node;
        match (input, node) {
            (GraphNode::Sort(sort_node), GraphNode::LimitOffset(limit_offset_node)) => {
                Some(limit_offset_node.process_with_ordered_input(sort_node.sorted_tuples()))
            }
            _ => None,
        }
    }

    /// Ordered tuples of a node, owned. An include built past a page's LimitOffset
    /// keeps the page's order: its ordered output is its ordered input mapped through
    /// its current tuples (tuple identity is the row ids, which the include keeps).
    fn ordered_tuples_owned(&self, node_id: NodeId) -> Option<Vec<Tuple>> {
        if self.deferred_arrays_tail.is_some()
            && let Some(GraphNode::ArraySubquery(node)) = self.get_node(node_id)
        {
            let input = self.get_inputs(node_id).first().copied()?;
            let ordered_input = self.ordered_tuples_owned(input)?;
            let current = RowNode::current_tuples(node);
            return Some(
                ordered_input
                    .iter()
                    .filter_map(|tuple| current.get(tuple).cloned())
                    .collect(),
            );
        }
        self.ordered_tuples_from_node(node_id)
            .map(<[Tuple]>::to_vec)
    }

    /// Visit a node's ordered tuples without cloning them — the borrowing twin of
    /// `ordered_tuples_owned`. `visit` returns false to stop. The result says whether the
    /// walk reached the end; `None` when the node has no ordered output.
    fn visit_ordered(
        &self,
        node_id: NodeId,
        visit: &mut dyn FnMut(&Tuple) -> bool,
    ) -> Option<bool> {
        if self.deferred_arrays_tail.is_some()
            && let Some(GraphNode::ArraySubquery(node)) = self.get_node(node_id)
        {
            let input = self.get_inputs(node_id).first().copied()?;
            let current = RowNode::current_tuples(node);
            return self.visit_ordered(input, &mut |tuple| match current.get(tuple) {
                Some(mapped) => visit(mapped),
                None => true,
            });
        }
        for tuple in self.ordered_tuples_from_node(node_id)? {
            if !visit(tuple) {
                return Some(false);
            }
        }
        Some(true)
    }

    /// The tuples an update-only `delta` changed in `input`'s ordered output, cloned,
    /// with their positions — when that output still has `len` tuples whose ids
    /// `ids_match` accepts position by position (see `ordered_in_place`). `None`, which
    /// means rebuild, on any difference: a reorder, a length change, an updated tuple
    /// the ordered output does not hold, a delta that is not update-only.
    fn in_place_updates(
        &self,
        input: NodeId,
        delta: &TupleDelta,
        len: usize,
        ids_match: &dyn Fn(usize, &Tuple) -> bool,
    ) -> Option<Vec<(usize, Tuple)>> {
        if !(delta.added.is_empty() && delta.removed.is_empty() && delta.moved.is_empty())
            || delta.updated.is_empty()
        {
            return None;
        }
        let wanted: AHashSet<&Tuple> = delta.updated.iter().map(|(_, new)| new).collect();
        if wanted.len() != delta.updated.len() {
            return None;
        }
        let mut updates = Vec::with_capacity(wanted.len());
        // Each wanted tuple must stand at exactly one position: one met twice would be
        // replaced twice while another, never met, kept its stale content.
        let mut matched: AHashSet<&Tuple> = AHashSet::with_capacity(wanted.len());
        let mut position = 0usize;
        let complete = self.visit_ordered(input, &mut |tuple| {
            if position >= len || !ids_match(position, tuple) {
                return false;
            }
            if let Some(&want) = wanted.get(tuple) {
                if !matched.insert(want) {
                    return false;
                }
                updates.push((position, tuple.clone()));
            }
            position += 1;
            true
        })?;
        (complete && position == len && updates.len() == wanted.len()).then_some(updates)
    }

    /// Whether a node the inputs rule would skip holds work of its own that its dirty bit
    /// does not show. `Project` and `Output` are skipped by another rule and never by being
    /// clean.
    fn clean_with_pending_work(&self, node_id: NodeId) -> bool {
        let node = self.get_node(node_id);
        !matches!(
            node,
            None | Some(GraphNode::Project(_) | GraphNode::Output(_))
        ) && !self.is_dirty(node_id)
            && node.is_some_and(GraphNode::has_pending_work)
    }

    /// Whether this pass can answer `node_id` with an empty delta without
    /// running it (see `query_manager::settle_skips`): its inputs all emitted
    /// nothing, and either it is a clean node that is not a scan, or it is the
    /// `Project`/`Output` tail re-materializing an unchanged list. The tail is
    /// dirtied by every include below it whether or not routing reached an
    /// instance, so it needs the second rule; scans never qualify, because a
    /// clean scan still rescans under a local overlay.
    fn unchanged_by_this_pass(
        &self,
        node_id: NodeId,
        tuple_deltas: &AHashMap<NodeId, TupleDelta>,
    ) -> bool {
        let node = self.get_node(node_id);
        if matches!(node, None | Some(GraphNode::IndexScan(_))) {
            return false;
        }
        let inputs = self.get_inputs(node_id);
        if inputs.is_empty()
            || !inputs
                .iter()
                .all(|dep| tuple_deltas.get(dep).is_none_or(TupleDelta::is_empty))
        {
            return false;
        }
        // The first settle delivers the initial list, empty or not.
        let output_materialized = matches!(
            self.get_node(self.output_node),
            Some(GraphNode::Output(output)) if output.has_materialized()
        );
        if matches!(node, Some(GraphNode::Project(_) | GraphNode::Output(_))) {
            return output_materialized;
        }
        let clean = !self.is_dirty(node_id);
        // A clean node is skipped on its inputs alone, which is sound only while
        // every internal "needs work" state is set together with the node's dirty
        // bit. Nothing enforces that at the mutation sites; this does.
        debug_assert!(
            !clean || node.is_none_or(|node| !node.has_pending_work()),
            "settle skip: node {} is clean but holds pending work of its own",
            node_id.0
        );
        clean
    }

    fn ordered_tuples_from_node(&self, node_id: NodeId) -> Option<&[Tuple]> {
        match self.get_node(node_id) {
            Some(GraphNode::Sort(node)) => Some(node.sorted_tuples()),
            Some(GraphNode::LimitOffset(node)) => Some(node.windowed_tuples()),
            Some(GraphNode::MagicColumns(node)) => Some(node.ordered_tuples()),
            Some(GraphNode::Project(node)) => Some(node.ordered_tuples()),
            _ => None,
        }
    }

    /// Settle the graph - process all dirty nodes in topological order.
    /// Uses tuple-based processing internally, converts to RowDelta for output.
    pub fn settle<F>(&mut self, storage: &dyn Storage, mut row_loader: F) -> RowDelta
    where
        F: FnMut(ObjectId, Option<TableName>) -> Option<LoadedRow>,
    {
        self.settle_with_source_overlay(storage, None, &mut row_loader)
    }

    pub(crate) fn settle_with_source_overlay<F>(
        &mut self,
        storage: &dyn Storage,
        local_overlay_rows: Option<&HashMap<ObjectId, RowBatchKey>>,
        mut row_loader: F,
    ) -> RowDelta
    where
        F: FnMut(ObjectId, Option<TableName>) -> Option<LoadedRow>,
    {
        self.settle_with_context(storage, local_overlay_rows, None, &mut row_loader)
    }

    pub(crate) fn settle_with_settlement_eval_cache<F>(
        &mut self,
        storage: &dyn Storage,
        settlement_eval_cache: Option<&mut SettlementEvalCache>,
        row_loader: F,
    ) -> RowDelta
    where
        F: FnMut(ObjectId, Option<TableName>) -> Option<LoadedRow>,
    {
        self.settle_with_context(storage, None, settlement_eval_cache, row_loader)
    }

    fn settle_with_context<F>(
        &mut self,
        storage: &dyn Storage,
        local_overlay_rows: Option<&HashMap<ObjectId, RowBatchKey>>,
        mut settlement_eval_cache: Option<&mut SettlementEvalCache>,
        mut row_loader: F,
    ) -> RowDelta
    where
        F: FnMut(ObjectId, Option<TableName>) -> Option<LoadedRow>,
    {
        let mut delta = self.settle_pass(
            storage,
            local_overlay_rows,
            settlement_eval_cache.as_deref_mut(),
            &mut row_loader,
        );
        // A window that left the page short walks further, and the graph settles
        // again; each growth doubles the walk, so this bound is never the limit in
        // practice. One settle reports the composed change.
        for _ in 0..MAX_WINDOW_GROWTHS {
            if !self.grow_short_windows() {
                break;
            }
            let next = self.settle_pass(
                storage,
                local_overlay_rows,
                settlement_eval_cache.as_deref_mut(),
                &mut row_loader,
            );
            delta = delta.compose(next);
        }
        delta
    }

    /// Grow every window whose frontier stops short of the page's last row (see
    /// `IndexScanNode::window_needs_growth`) and dirty it for the next pass.
    fn grow_short_windows(&mut self) -> bool {
        let Some((sort, cap)) = self.window_page() else {
            return false;
        };
        let edge = (sort.sorted_tuples().len() >= cap).then(|| cap - 1);
        self.grow_windows_short_of(edge)
    }

    /// The Sort the windows feed and the page's prefix cap, when the graph has windows
    /// and a page.
    fn window_page(&self) -> Option<(&crate::query_manager::graph_nodes::sort::SortNode, usize)> {
        if self.window_scans.is_empty() {
            return None;
        }
        let (Some(sort_id), Some(page_id)) = (self.window_sort, self.pagination_node) else {
            return None;
        };
        let Some(GraphNode::LimitOffset(page)) = self.get_node(page_id) else {
            return None;
        };
        let cap = page.sync_prefix_cap().filter(|cap| *cap > 0)?;
        let Some(GraphNode::Sort(sort)) = self.get_node(sort_id) else {
            return None;
        };
        Some((sort, cap))
    }

    /// Grow every window whose frontier stops short of the Sort's row at `edge` — every
    /// window not yet exhausted when `edge` is `None`, the page having run out of rows —
    /// and dirty it for the next settle. Returns whether any window grew.
    ///
    /// The graph grows its windows against the page's own last row as it settles. A
    /// server graph carries no policy filter, so the rows its scope may use are fewer
    /// than the Sort holds: its scope walk (`filtered_sync_scope_tuples`) passes the
    /// position of the last row the session may read, and grows from there.
    pub(crate) fn grow_windows_short_of(&mut self, edge: Option<usize>) -> bool {
        let Some((sort, _)) = self.window_page() else {
            return false;
        };
        let edge = edge
            .and_then(|position| sort.first_key_value_at(position))
            .and_then(|value| crate::query_manager::composite_index::fixed_width_encoding(&value))
            .map(|bytes| crate::query_manager::composite_index::hex(&bytes));
        let short: Vec<NodeId> = self
            .window_scans
            .iter()
            .copied()
            .filter(|scan_id| {
                matches!(
                    self.get_node(*scan_id),
                    Some(GraphNode::IndexScan(scan)) if scan.window_needs_growth(edge.as_deref())
                )
            })
            .collect();
        let mut grew = false;
        for scan_id in short {
            if let Some(GraphNode::IndexScan(scan)) = self.get_node_mut(scan_id)
                && scan.grow_window()
            {
                grew = true;
                self.mark_dirty(scan_id);
                self.mark_downstream_dirty(scan_id);
            }
        }
        if grew {
            crate::query_manager::settle_cost::bump(
                &crate::query_manager::settle_cost::WINDOW_GROWTHS,
            );
        }
        grew
    }

    fn settle_pass<F>(
        &mut self,
        storage: &dyn Storage,
        local_overlay_rows: Option<&HashMap<ObjectId, RowBatchKey>>,
        mut settlement_eval_cache: Option<&mut SettlementEvalCache>,
        mut row_loader: F,
    ) -> RowDelta
    where
        F: FnMut(ObjectId, Option<TableName>) -> Option<LoadedRow>,
    {
        let order = self.topo_sort_dirty();
        if !order.is_empty() {
            tracing::trace!(dirty_nodes = order.len(), table = %self.table, "settling query graph");
        }
        // Settle-cost accounting: one relaxed add per graph settle rather than
        // one per node, since the node count is known up front.
        crate::query_manager::settle_cost::add(
            &crate::query_manager::settle_cost::GRAPH_NODES_EVALUATED,
            order.len() as u64,
        );
        let mut tuple_deltas: AHashMap<NodeId, TupleDelta> = AHashMap::new();

        let ctx = SourceContext {
            storage,
            local_overlay_rows,
        };
        let skip_mode = crate::query_manager::settle_skips::settle_skip_mode();
        let mut skipped = 0u64;
        let mut verified = 0u64;

        for node_id in order {
            let skippable =
                skip_mode != SkipMode::Off && self.unchanged_by_this_pass(node_id, &tuple_deltas);
            if skippable {
                skipped += 1;
            }
            if skippable && skip_mode == SkipMode::Verify {
                verified += 1;
                if self.clean_with_pending_work(node_id) {
                    crate::query_manager::settle_cost::note_settle_skip_violation(
                        &crate::query_manager::settle_cost::SETTLE_SKIP_PENDING_VIOLATIONS,
                    );
                    tracing::error!(
                        target: "jazz::settle_skips",
                        table = %self.table,
                        node_id = node_id.0,
                        "settle skip: a clean node holds pending work of its own"
                    );
                }
            }
            if skippable && skip_mode == SkipMode::Skip {
                tuple_deltas.insert(node_id, TupleDelta::new());
                continue;
            }
            let node_type = match self.get_node(node_id) {
                Some(GraphNode::IndexScan(_)) => "IndexScan",
                Some(GraphNode::Union(_)) => "Union",
                Some(GraphNode::Alias(_)) => "Alias",
                Some(GraphNode::Join(_)) => "Join",
                Some(GraphNode::MagicColumns(_)) => "MagicColumns",
                Some(GraphNode::Project(_)) => "Project",
                Some(GraphNode::SelectElement(_)) => "SelectElement",
                Some(GraphNode::RecursiveRelation(_)) => "RecursiveRelation",
                Some(GraphNode::Materialize(_)) => "Materialize",
                Some(GraphNode::Filter(_)) => "Filter",
                Some(GraphNode::PolicyFilter(_)) => "PolicyFilter",
                Some(GraphNode::Sort(_)) => "Sort",
                Some(GraphNode::LimitOffset(_)) => "LimitOffset",
                Some(GraphNode::ArraySubquery(_)) => "ArraySubquery",
                Some(GraphNode::Output(_)) => "Output",
                Some(GraphNode::ExistsOutput(_)) => "ExistsOutput",
                None => "Unknown",
            };

            match self.get_node(node_id) {
                Some(GraphNode::IndexScan(_)) => {
                    // `topo_sort_dirty` pulls a dirty node's whole INPUT chain
                    // into the order, so a source node is visited even when
                    // nothing about it changed — which is how one write into
                    // an include's inner table made the settle re-scan the
                    // whole OUTER result set: O(rows) storage reads to
                    // reproduce the membership it already held. A node holding
                    // an exact baseline, with no overlay in play, can only
                    // rescan to that same membership, so the empty delta below
                    // is the one the rescan would have produced (v14 L1's
                    // residual term; `include_instance_flatness` measures it).
                    let holds_baseline = local_overlay_rows.is_none_or(|rows| rows.is_empty())
                        && !self.is_dirty(node_id)
                        && matches!(
                            self.get_node(node_id),
                            Some(GraphNode::IndexScan(scan_node))
                                if scan_node.holds_exact_baseline()
                        );
                    if holds_baseline {
                        tuple_deltas.insert(node_id, TupleDelta::new());
                    } else if let Some(GraphNode::IndexScan(scan_node)) = self.get_node_mut(node_id)
                    {
                        let delta = SourceNode::scan(scan_node, &ctx);
                        tracing::debug!(
                            node_id = node_id.0,
                            node_type,
                            added = delta.added.len(),
                            removed = delta.removed.len(),
                            "graph node evaluated"
                        );
                        tuple_deltas.insert(node_id, delta);
                    }
                }
                Some(GraphNode::Union(_)) => {
                    let inputs = self.collect_tuple_inputs(node_id);
                    if let Some(GraphNode::Union(union_node)) = self.get_node_mut(node_id) {
                        let input_refs: Vec<_> = inputs.iter().collect();
                        let delta = TransformNode::process(union_node, &input_refs);
                        tracing::debug!(
                            node_id = node_id.0,
                            node_type,
                            added = delta.added.len(),
                            removed = delta.removed.len(),
                            "graph node evaluated"
                        );
                        tuple_deltas.insert(node_id, delta);
                    }
                }
                Some(GraphNode::Alias(_)) => {
                    let input_delta = self
                        .get_inputs(node_id)
                        .first()
                        .and_then(|dep| tuple_deltas.get(dep).cloned())
                        .unwrap_or_default();

                    if let Some(GraphNode::Alias(alias_node)) = self.get_node_mut(node_id) {
                        let delta = RowNode::process(alias_node, input_delta);
                        tracing::debug!(
                            node_id = node_id.0,
                            node_type,
                            added = delta.added.len(),
                            removed = delta.removed.len(),
                            "graph node evaluated"
                        );
                        tuple_deltas.insert(node_id, delta);
                    }
                }
                Some(GraphNode::Join(_)) => {
                    // JoinNode has two inputs: left (index 0) and right (index 1)
                    let inputs = self.get_inputs(node_id);
                    let left_delta = inputs
                        .first()
                        .and_then(|dep| tuple_deltas.get(dep).cloned())
                        .unwrap_or_default();
                    let right_delta = inputs
                        .get(1)
                        .and_then(|dep| tuple_deltas.get(dep).cloned())
                        .unwrap_or_default();

                    if let Some(GraphNode::Join(join_node)) = self.get_node_mut(node_id) {
                        // Process left side first, then right side
                        let left_result = join_node.process_left(left_delta);
                        let right_result = join_node.process_right(right_delta);

                        // Merge results
                        let mut merged = TupleDelta::new();
                        merged.added.extend(left_result.added);
                        merged.added.extend(right_result.added);
                        merged.removed.extend(left_result.removed);
                        merged.removed.extend(right_result.removed);

                        tracing::debug!(
                            node_id = node_id.0,
                            node_type,
                            added = merged.added.len(),
                            removed = merged.removed.len(),
                            "graph node evaluated"
                        );
                        tuple_deltas.insert(node_id, merged);
                    }
                }
                Some(GraphNode::Project(_)) => {
                    let input_node = self.get_inputs(node_id).first().copied();
                    let input_delta = input_node
                        .and_then(|dep| tuple_deltas.get(&dep).cloned())
                        .unwrap_or_default();
                    let mode = in_place_mode();
                    let in_place = match (mode, input_node, self.get_node(node_id)) {
                        (
                            InPlaceMode::Apply | InPlaceMode::Verify,
                            Some(dep),
                            Some(GraphNode::Project(project)),
                        ) => project
                            .in_place_len()
                            .and_then(|len| {
                                self.in_place_updates(dep, &input_delta, len, &|position, tuple| {
                                    project.in_place_ids_match(position, tuple)
                                })
                            })
                            .and_then(|updates| project.project_in_place(&updates)),
                        _ => None,
                    };

                    if mode == InPlaceMode::Apply
                        && let Some(projected) = in_place
                    {
                        ORDERED_IN_PLACE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if let Some(GraphNode::Project(project_node)) = self.get_node_mut(node_id) {
                            let delta = project_node.apply_in_place(projected);
                            tracing::debug!(
                                node_id = node_id.0,
                                node_type,
                                updated = delta.updated.len(),
                                "graph node updated in place"
                            );
                            tuple_deltas.insert(node_id, delta);
                        }
                    } else {
                        let before = in_place
                            .as_ref()
                            .and_then(|_| match self.get_node(node_id) {
                                Some(GraphNode::Project(project)) => {
                                    Some(project.ordered_tuples().to_vec())
                                }
                                _ => None,
                            });
                        let ordered_input =
                            input_node.and_then(|dep| self.ordered_tuples_owned(dep));

                        if let Some(GraphNode::Project(project_node)) = self.get_node_mut(node_id) {
                            let delta = if let Some(ordered) = ordered_input {
                                project_node.process_with_ordered_input(input_delta, &ordered)
                            } else {
                                RowNode::process(project_node, input_delta)
                            };
                            if let (Some(before), Some(projected)) = (before, in_place.as_ref()) {
                                check_in_place(
                                    node_id,
                                    node_type,
                                    &before,
                                    projected,
                                    project_node.ordered_tuples(),
                                    &delta,
                                );
                            }
                            tracing::debug!(
                                node_id = node_id.0,
                                node_type,
                                added = delta.added.len(),
                                removed = delta.removed.len(),
                                "graph node evaluated"
                            );
                            tuple_deltas.insert(node_id, delta);
                        }
                    }
                }
                Some(GraphNode::SelectElement(_)) => {
                    let input_delta = self
                        .get_inputs(node_id)
                        .first()
                        .and_then(|dep| tuple_deltas.get(dep).cloned())
                        .unwrap_or_default();

                    if let Some(GraphNode::SelectElement(select_node)) = self.get_node_mut(node_id)
                    {
                        let delta = RowNode::process(select_node, input_delta);
                        tracing::debug!(
                            node_id = node_id.0,
                            node_type,
                            added = delta.added.len(),
                            removed = delta.removed.len(),
                            "graph node evaluated"
                        );
                        tuple_deltas.insert(node_id, delta);
                    }
                }
                Some(GraphNode::RecursiveRelation(_)) => {
                    let input_delta = self
                        .get_inputs(node_id)
                        .first()
                        .and_then(|dep| tuple_deltas.get(dep).cloned())
                        .unwrap_or_default();

                    if let Some(GraphNode::RecursiveRelation(recursive_node)) =
                        self.get_node_mut(node_id)
                    {
                        let delta = recursive_node.process_with_context(
                            input_delta,
                            storage,
                            settlement_eval_cache.as_deref_mut(),
                            &mut |id, hint| row_loader(id, hint),
                        );
                        tracing::debug!(
                            node_id = node_id.0,
                            node_type,
                            added = delta.added.len(),
                            removed = delta.removed.len(),
                            "graph node evaluated"
                        );
                        tuple_deltas.insert(node_id, delta);
                    }
                }
                Some(GraphNode::Materialize(_)) => {
                    let input_delta = self
                        .get_inputs(node_id)
                        .first()
                        .and_then(|dep| tuple_deltas.get(dep).cloned())
                        .unwrap_or_default();

                    if let Some(GraphNode::Materialize(mat_node)) = self.get_node_mut(node_id) {
                        let deleted_delta = mat_node.check_deleted_tuples();
                        let new_delta = mat_node.materialize_tuples(input_delta, &mut row_loader);
                        let update_delta = mat_node.check_updated_tuples(&mut row_loader);

                        let mut merged = TupleDelta::new();
                        merged.added.extend(new_delta.added);
                        merged.added.extend(update_delta.added);
                        merged.removed.extend(deleted_delta.removed);
                        merged.removed.extend(new_delta.removed);
                        merged.removed.extend(update_delta.removed);
                        merged.updated.extend(new_delta.updated);
                        merged.updated.extend(update_delta.updated);

                        tracing::debug!(
                            node_id = node_id.0,
                            node_type,
                            added = merged.added.len(),
                            removed = merged.removed.len(),
                            "graph node evaluated"
                        );
                        tuple_deltas.insert(node_id, merged);
                    }
                }
                Some(GraphNode::MagicColumns(_)) => {
                    let input_node = self.get_inputs(node_id).first().copied();
                    let input_delta = input_node
                        .and_then(|dep| tuple_deltas.get(&dep).cloned())
                        .unwrap_or_default();
                    let ordered_input = input_node.and_then(|dep| self.ordered_tuples_owned(dep));

                    if let Some(GraphNode::MagicColumns(magic_node)) = self.get_node_mut(node_id) {
                        let delta = if let Some(ordered) = ordered_input {
                            magic_node.process_with_ordered_input(
                                input_delta,
                                &ordered,
                                storage,
                                &mut |id, hint| row_loader(id, hint),
                            )
                        } else {
                            magic_node.process_with_context(
                                input_delta,
                                storage,
                                &mut |id, hint| row_loader(id, hint),
                            )
                        };
                        tracing::debug!(
                            node_id = node_id.0,
                            node_type,
                            added = delta.added.len(),
                            removed = delta.removed.len(),
                            "graph node evaluated"
                        );
                        tuple_deltas.insert(node_id, delta);
                    }
                }
                Some(GraphNode::Filter(_)) => {
                    let input_delta = self
                        .get_inputs(node_id)
                        .first()
                        .and_then(|dep| tuple_deltas.get(dep).cloned())
                        .unwrap_or_default();

                    if let Some(GraphNode::Filter(filter_node)) = self.get_node_mut(node_id) {
                        let delta = RowNode::process(filter_node, input_delta);
                        tracing::debug!(
                            node_id = node_id.0,
                            node_type,
                            added = delta.added.len(),
                            removed = delta.removed.len(),
                            "graph node evaluated"
                        );
                        tuple_deltas.insert(node_id, delta);
                    }
                }
                Some(GraphNode::PolicyFilter(_)) => {
                    let input_delta = self
                        .get_inputs(node_id)
                        .first()
                        .and_then(|dep| tuple_deltas.get(dep).cloned())
                        .unwrap_or_default();

                    if let Some(GraphNode::PolicyFilter(policy_node)) = self.get_node_mut(node_id) {
                        // Use process_with_context if the policy has INHERITS clauses
                        let delta = if policy_node.has_inherits() {
                            policy_node.process_with_context(
                                input_delta,
                                storage,
                                &mut |id, hint| row_loader(id, hint),
                            )
                        } else {
                            RowNode::process(policy_node, input_delta)
                        };
                        tracing::debug!(
                            node_id = node_id.0,
                            node_type,
                            added = delta.added.len(),
                            removed = delta.removed.len(),
                            "graph node evaluated"
                        );
                        tuple_deltas.insert(node_id, delta);
                    }
                }
                Some(GraphNode::Sort(_)) => {
                    let input_delta = self
                        .get_inputs(node_id)
                        .first()
                        .and_then(|dep| tuple_deltas.get(dep).cloned())
                        .unwrap_or_default();

                    if let Some(GraphNode::Sort(sort_node)) = self.get_node_mut(node_id) {
                        let delta = RowNode::process(sort_node, input_delta);
                        tracing::debug!(
                            node_id = node_id.0,
                            node_type,
                            added = delta.added.len(),
                            removed = delta.removed.len(),
                            "graph node evaluated"
                        );
                        tuple_deltas.insert(node_id, delta);
                    }
                }
                Some(GraphNode::LimitOffset(_)) => {
                    let input_node = self.get_inputs(node_id).first().copied();
                    let delta = input_node
                        .and_then(|dep| {
                            self.process_limit_offset_with_ordered_sort_input(node_id, dep)
                        })
                        .or_else(|| {
                            let Some(GraphNode::LimitOffset(lo_node)) = self.get_node_mut(node_id)
                            else {
                                return None;
                            };
                            let input_delta = input_node
                                .and_then(|dep| tuple_deltas.get(&dep).cloned())
                                .unwrap_or_default();
                            Some(RowNode::process(lo_node, input_delta))
                        });

                    if let Some(delta) = delta {
                        tracing::debug!(
                            node_id = node_id.0,
                            node_type,
                            added = delta.added.len(),
                            removed = delta.removed.len(),
                            "graph node evaluated"
                        );
                        tuple_deltas.insert(node_id, delta);
                    }
                }
                Some(GraphNode::ArraySubquery(_)) => {
                    let input_delta = self
                        .get_inputs(node_id)
                        .first()
                        .and_then(|dep| tuple_deltas.get(dep).cloned())
                        .unwrap_or_default();

                    if let Some(GraphNode::ArraySubquery(subquery_node)) =
                        self.get_node_mut(node_id)
                    {
                        // Outer input FIRST, inner re-evaluation second. The
                        // reverse order rebuilt instances from a stale
                        // outer-tuple snapshot when one settle carried both an
                        // outer-row update and an inner change: the outer
                        // path's retraction then missed `current_tuples` and
                        // the same parent was served twice (FINDING
                        // manifestation 3 in
                        // `manager_tests/subscription_output_oracle.rs`).
                        // Processing the outer delta first keeps instance
                        // state in step with the input stream; re-evaluation
                        // then works from current outer tuples and emits a
                        // cleanly chained update.
                        let mut delta = subquery_node.process_with_context(
                            input_delta,
                            storage,
                            &mut |id, hint| row_loader(id, hint),
                        );

                        // Re-evaluate existing instances if inner data changed
                        if subquery_node.is_inner_dirty() {
                            let reevaluated = subquery_node
                                .reevaluate_all(storage, &mut |id, hint| row_loader(id, hint));
                            delta.merge(reevaluated);
                        }
                        tracing::debug!(
                            node_id = node_id.0,
                            node_type,
                            added = delta.added.len(),
                            removed = delta.removed.len(),
                            "graph node evaluated"
                        );
                        tuple_deltas.insert(node_id, delta);
                    }
                }
                Some(GraphNode::Output(_)) => {
                    let input_node = self.get_inputs(node_id).first().copied();
                    let mode = in_place_mode();
                    let in_place = match (mode, input_node, self.get_node(node_id)) {
                        (
                            InPlaceMode::Apply | InPlaceMode::Verify,
                            Some(dep),
                            Some(GraphNode::Output(output)),
                        ) if output.can_update_in_place() => {
                            tuple_deltas.get(&dep).and_then(|input_delta| {
                                let list = output.ordered_tuples();
                                self.in_place_updates(
                                    dep,
                                    input_delta,
                                    list.len(),
                                    &|position, tuple| {
                                        list.get(position).is_some_and(|held| {
                                            held.len() == tuple.len()
                                                && held.id_iter().eq(tuple.id_iter())
                                        })
                                    },
                                )
                            })
                        }
                        _ => None,
                    };

                    if mode == InPlaceMode::Apply
                        && let Some(updates) = in_place
                    {
                        ORDERED_IN_PLACE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if let Some(GraphNode::Output(output_node)) = self.get_node_mut(node_id) {
                            let delta = output_node.apply_in_place(updates);
                            tracing::debug!(
                                node_id = node_id.0,
                                node_type,
                                updated = delta.updated.len(),
                                "graph node updated in place"
                            );
                            tuple_deltas.insert(node_id, delta);
                        }
                    } else {
                        let before = in_place
                            .as_ref()
                            .and_then(|_| match self.get_node(node_id) {
                                Some(GraphNode::Output(output)) => {
                                    Some(output.ordered_tuples().to_vec())
                                }
                                _ => None,
                            });
                        let ordered_input =
                            input_node.and_then(|dep| self.ordered_tuples_owned(dep));

                        if let Some(GraphNode::Output(output_node)) = self.get_node_mut(node_id) {
                            let delta = if let Some(ordered) = ordered_input {
                                output_node.process_with_ordered_input(&ordered)
                            } else {
                                let input_delta = input_node
                                    .and_then(|dep| tuple_deltas.get(&dep).cloned())
                                    .unwrap_or_default();
                                RowNode::process(output_node, input_delta)
                            };
                            if let (Some(before), Some(updates)) = (before, in_place.as_ref()) {
                                check_in_place(
                                    node_id,
                                    node_type,
                                    &before,
                                    updates,
                                    output_node.ordered_tuples(),
                                    &delta,
                                );
                            }
                            tracing::debug!(
                                node_id = node_id.0,
                                node_type,
                                added = delta.added.len(),
                                removed = delta.removed.len(),
                                "graph node evaluated"
                            );
                            tuple_deltas.insert(node_id, delta);
                        }
                    }
                }
                Some(GraphNode::ExistsOutput(_)) => {
                    let input_delta = self
                        .get_inputs(node_id)
                        .first()
                        .and_then(|dep| tuple_deltas.get(dep).cloned())
                        .unwrap_or_default();

                    if let Some(GraphNode::ExistsOutput(exists_node)) = self.get_node_mut(node_id) {
                        let delta = RowNode::process(exists_node, input_delta);
                        tracing::debug!(
                            node_id = node_id.0,
                            node_type,
                            added = delta.added.len(),
                            removed = delta.removed.len(),
                            "graph node evaluated"
                        );
                        tuple_deltas.insert(node_id, delta);
                    }
                }
                None => {}
            }
            // `SkipMode::Verify`: the node ran, and must have emitted what the skip
            // would have. A release build counts and logs a violation instead of
            // panicking, so a stand run with `JAZZ_SETTLE_SKIPS=verify` can show
            // zero on real traffic.
            if skippable
                && tuple_deltas
                    .get(&node_id)
                    .is_some_and(|delta| !delta.is_empty())
            {
                crate::query_manager::settle_cost::note_settle_skip_violation(
                    &crate::query_manager::settle_cost::SETTLE_SKIP_PARITY_VIOLATIONS,
                );
                let emitted = &tuple_deltas[&node_id];
                tracing::error!(
                    target: "jazz::settle_skips",
                    table = %self.table,
                    node_type,
                    node_id = node_id.0,
                    added = emitted.added.len(),
                    removed = emitted.removed.len(),
                    updated = emitted.updated.len(),
                    moved = emitted.moved.len(),
                    "settle skip parity: a node emitted a delta from unchanged inputs"
                );
                if cfg!(debug_assertions) {
                    panic!(
                        "settle skip parity: {node_type} node {} emitted a delta from unchanged \
                         inputs",
                        node_id.0
                    );
                }
            }
        }
        crate::query_manager::settle_cost::add(
            &crate::query_manager::settle_cost::GRAPH_NODES_SKIPPED,
            skipped,
        );
        crate::query_manager::settle_cost::add(
            &crate::query_manager::settle_cost::SETTLE_SKIPS_VERIFIED,
            verified,
        );
        #[cfg(test)]
        crate::query_manager::settle_cost::SKIPS_ON_THREAD.with(|counts| {
            let (skipped_before, verified_before) = counts.get();
            counts.set((skipped_before + skipped, verified_before + verified));
        });

        self.dirty_bitmap.fill(false);

        // Convert TupleDelta to RowDelta for output
        // For single-table queries: use simple conversion
        // For join queries: flatten multi-element tuples using table descriptors
        tuple_deltas
            .remove(&self.output_node)
            .and_then(|td| {
                if self.table_descriptors.len() == 1 {
                    // Single-table query - direct conversion
                    td.to_row_delta()
                } else {
                    // Join query - flatten multi-element tuples
                    td.flatten_to_row_delta(&self.table_descriptors, &self.combined_descriptor)
                }
            })
            .unwrap_or_default()
    }

    /// Collect tuple sets from input nodes for a transform node.
    fn collect_tuple_inputs(&self, node_id: NodeId) -> Vec<AHashSet<Tuple>> {
        self.get_inputs(node_id)
            .iter()
            .filter_map(|dep| match &self.nodes[dep.0 as usize].node {
                GraphNode::IndexScan(n) => Some(n.current_tuples().clone()),
                GraphNode::Union(n) => Some(n.current_tuples().clone()),
                _ => None,
            })
            .collect()
    }
}

/// Every field of two tuples is equal: ids, element kind, content, batch and row provenance,
/// and the tuple's provenance sets — everything a reader could tell apart, not only what
/// `has_tuple_content_changed` counts as a change.
fn tuples_identical(a: &Tuple, b: &Tuple) -> bool {
    a.provenance() == b.provenance()
        && a.batch_provenance() == b.batch_provenance()
        && a.len() == b.len()
        && a.iter().zip(b.iter()).all(|pair| match pair {
            (TupleElement::Id(a), TupleElement::Id(b)) => a == b,
            (
                TupleElement::Row {
                    id: a_id,
                    content: a_content,
                    batch_id: a_batch,
                    row_provenance: a_provenance,
                },
                TupleElement::Row {
                    id: b_id,
                    content: b_content,
                    batch_id: b_batch,
                    row_provenance: b_provenance,
                },
            ) => {
                a_id == b_id
                    && a_content == b_content
                    && a_batch == b_batch
                    && a_provenance == b_provenance
            }
            _ => false,
        })
}

/// Under `InPlaceMode::Verify`: check a rebuild against the in-place update it stands in
/// for — `before` with `updates` replaced where they stand, and a delta that updates
/// exactly those of them whose content changed. A difference is a list the in-place path
/// would have served wrong.
fn check_in_place(
    node_id: NodeId,
    node_type: &str,
    before: &[Tuple],
    updates: &[(usize, Tuple)],
    rebuilt: &[Tuple],
    delta: &TupleDelta,
) {
    ORDERED_IN_PLACE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut expected: Vec<&Tuple> = before.iter().collect();
    let mut changed: AHashSet<&Tuple> = AHashSet::new();
    for (position, new) in updates {
        if before
            .get(*position)
            .is_none_or(|old| has_tuple_content_changed(old, new))
        {
            changed.insert(new);
        }
        if let Some(slot) = expected.get_mut(*position) {
            *slot = new;
        }
    }
    let delta_updated: AHashSet<&Tuple> = delta.updated.iter().map(|(_, new)| new).collect();
    let matches = delta.added.is_empty()
        && delta.removed.is_empty()
        && delta.moved.is_empty()
        && expected.len() == rebuilt.len()
        && expected
            .iter()
            .zip(rebuilt)
            .all(|(expected, rebuilt)| tuples_identical(expected, rebuilt))
        && delta_updated.len() == delta.updated.len()
        && delta_updated == changed;
    if !matches {
        ORDERED_IN_PLACE_PARITY_VIOLATIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        tracing::error!(
            node_id = node_id.0,
            node_type,
            "ordered in place: the rebuild differs from the in-place update"
        );
        debug_assert!(
            matches,
            "ordered in place: node {} ({node_type}) rebuilt differently from its in-place update",
            node_id.0
        );
    }
}
