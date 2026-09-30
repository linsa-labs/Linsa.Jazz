//! Query graph: compiled relation IR ready for incremental settlement.
//!
//! The module is split into two phases:
//! - [`compile`] turns relation IR into [`QueryGraph`] nodes (pure transform).
//! - [`execute`] handles dirty tracking, topological settlement, and row I/O via a closure.
//!
//! Shared types live here; both submodules add `impl QueryGraph` blocks against them.

use std::collections::HashSet;
use std::fmt;

use bitvec::prelude::*;
use smallvec::SmallVec;

use crate::object::{BranchName, ObjectId};
use crate::query_manager::query::ArraySubquerySpec;
use crate::query_manager::types::{Row, RowDelta, RowDescriptor, TableName, Tuple, TupleDelta};

use super::graph_nodes::alias::AliasNode;
use super::graph_nodes::array_subquery::ArraySubqueryNode;
use super::graph_nodes::exists_output::ExistsOutputNode;
use super::graph_nodes::filter::FilterNode;
use super::graph_nodes::index_scan::IndexScanNode;
use super::graph_nodes::join::JoinNode;
use super::graph_nodes::limit_offset::LimitOffsetNode;
use super::graph_nodes::magic_columns::MagicColumnsNode;
use super::graph_nodes::materialize::MaterializeNode;
use super::graph_nodes::output::OutputNode;
use super::graph_nodes::policy_filter::PolicyFilterNode;
use super::graph_nodes::project::ProjectNode;
use super::graph_nodes::recursive_relation::RecursiveRelationNode;
use super::graph_nodes::select_element::SelectElementNode;
use super::graph_nodes::sort::SortNode;
use super::graph_nodes::union::UnionNode;
use super::graph_nodes::{NodeId, RowNode};
use super::types::{ColumnName, TupleProvenance};

pub mod compile;
pub mod execute;

#[cfg(test)]
mod tests;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryCompileError {
    UnknownTable(TableName),
    InvalidPlan(String),
}

impl fmt::Display for QueryCompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QueryCompileError::UnknownTable(table) => {
                write!(f, "unknown table referenced in relation_ir: {}", table)
            }
            QueryCompileError::InvalidPlan(reason) => write!(f, "invalid relation plan: {reason}"),
        }
    }
}

impl std::error::Error for QueryCompileError {}

/// A node in the query graph (type-erased).
#[derive(Debug)]
pub enum GraphNode {
    IndexScan(IndexScanNode),
    Union(UnionNode),
    Alias(AliasNode),
    Join(JoinNode),
    Materialize(MaterializeNode),
    MagicColumns(MagicColumnsNode),
    Project(ProjectNode),
    SelectElement(SelectElementNode),
    RecursiveRelation(RecursiveRelationNode),
    Filter(FilterNode),
    PolicyFilter(PolicyFilterNode),
    Sort(SortNode),
    LimitOffset(LimitOffsetNode),
    ArraySubquery(ArraySubqueryNode),
    Output(OutputNode),
    ExistsOutput(ExistsOutputNode),
}

/// Which rows of a paginated query get their includes built.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IncludePlacement {
    /// Only the page's rows, when the plan allows it: the includes are attached past
    /// the LimitOffset.
    PageRows,
    /// Every row the query's filters match: the includes are attached ahead of the
    /// LimitOffset. Needed when the sync scope can take rows past the page — a server
    /// graph compiled without the policy filter walks over rows the authorization
    /// schema denies to fill its prefix, and those rows need their includes too.
    MatchingRows,
}

/// Compact node with inline edge storage.
/// Most nodes have 0-2 inputs/outputs, so inline storage avoids heap allocation.
#[derive(Debug)]
pub struct CompactNode {
    pub node: GraphNode,
    /// Input edges (children/dependencies). Most nodes have 0-2 inputs.
    pub inputs: SmallVec<[NodeId; 2]>,
    /// Output edges (parents/dependents). Most nodes have 0-2 outputs.
    pub outputs: SmallVec<[NodeId; 2]>,
}

/// Compiled query graph for a single query.
#[derive(Debug)]
pub struct QueryGraph {
    /// Dense node storage (NodeId.0 is index).
    pub nodes: Vec<CompactNode>,
    /// Dirty tracking bitmap (1 bit per node, indexed by NodeId.0).
    pub(super) dirty_bitmap: BitVec,
    /// The output node ID.
    pub output_node: NodeId,
    /// The pagination node, when the query applies limit/offset.
    pub(super) pagination_node: Option<NodeId>,
    /// The last include node of a page whose includes are built past the pagination
    /// node (for the page's rows only). Its current tuples carry the includes'
    /// provenance for the page's rows; the rows ahead of the page carry none.
    pub(super) deferred_arrays_tail: Option<NodeId>,
    /// Ordered-window index scans (`IndexScanNode::new_window`), grown after a settle
    /// that leaves the page short of its rows.
    pub(super) window_scans: Vec<NodeId>,
    /// The ordered node the windows feed, whose prefix the page takes.
    pub(super) window_sort: Option<NodeId>,
    /// Table this query operates on.
    pub table: TableName,
    /// Index scan nodes for this query (for marking dirty on updates).
    pub index_scan_nodes: Vec<(NodeId, TableName, ColumnName)>, // (node_id, table, column)
    /// Array subquery nodes and their inner tables (for marking dirty on inner table updates).
    pub array_subquery_tables: Vec<(NodeId, TableName)>, // (node_id, inner_table)
    /// PolicyFilter nodes and their INHERITS-referenced tables (for marking dirty on table updates).
    pub policy_filter_tables: Vec<(NodeId, TableName)>, // (node_id, inherits_table)
    /// MagicColumns nodes and their policy dependency tables (for reactive re-evaluation).
    pub magic_column_tables: Vec<(NodeId, TableName)>, // (node_id, dependency_table)
    /// RecursiveRelation nodes and their step dependency tables (for marking dirty on table updates).
    pub recursive_relation_tables: Vec<(NodeId, TableName)>, // (node_id, step_table)
    /// Per-table descriptors in join order (for flattening multi-element tuples).
    pub table_descriptors: Vec<RowDescriptor>,
    /// Combined descriptor for output (all columns from all tables).
    pub combined_descriptor: RowDescriptor,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct RelationCompileFeatures {
    pub include_deleted: bool,
    pub array_subqueries: Vec<ArraySubquerySpec>,
    pub select_columns: Option<Vec<String>>,
}

impl QueryGraph {
    pub fn new(table: TableName, descriptor: RowDescriptor) -> Self {
        Self {
            nodes: Vec::new(),
            dirty_bitmap: BitVec::new(),
            output_node: NodeId(0),
            pagination_node: None,
            deferred_arrays_tail: None,
            window_scans: Vec::new(),
            window_sort: None,
            table,
            index_scan_nodes: Vec::new(),
            array_subquery_tables: Vec::new(),
            policy_filter_tables: Vec::new(),
            magic_column_tables: Vec::new(),
            recursive_relation_tables: Vec::new(),
            table_descriptors: vec![descriptor.clone()],
            combined_descriptor: descriptor,
        }
    }

    pub(super) fn add_node(&mut self, node: GraphNode) -> NodeId {
        let id = NodeId(self.nodes.len() as u64);
        self.nodes.push(CompactNode {
            node,
            inputs: SmallVec::new(),
            outputs: SmallVec::new(),
        });
        // Grow dirty bitmap to accommodate new node
        self.dirty_bitmap.push(true); // New nodes start dirty
        id
    }

    /// Add an edge from one node to another.
    pub fn add_edge(&mut self, from: NodeId, to: NodeId) {
        self.nodes[from.0 as usize].inputs.push(to);
        self.nodes[to.0 as usize].outputs.push(from);
    }

    /// Add a node and return its ID.
    pub fn add_node_with_id(&mut self, node: GraphNode) -> NodeId {
        self.add_node(node)
    }

    /// Get a reference to a node by ID.
    pub(super) fn get_node(&self, id: NodeId) -> Option<&GraphNode> {
        self.nodes.get(id.0 as usize).map(|c| &c.node)
    }

    /// Get a mutable reference to a node by ID.
    pub(super) fn get_node_mut(&mut self, id: NodeId) -> Option<&mut GraphNode> {
        self.nodes.get_mut(id.0 as usize).map(|c| &mut c.node)
    }

    /// Get input edges for a node.
    pub(super) fn get_inputs(&self, id: NodeId) -> &[NodeId] {
        &self.nodes[id.0 as usize].inputs
    }

    /// Get output edges (reverse edges) for a node.
    pub(super) fn get_outputs(&self, id: NodeId) -> Option<&[NodeId]> {
        self.nodes.get(id.0 as usize).map(|c| c.outputs.as_slice())
    }

    /// Returns ObjectIds contributing to current result set along with their branches.
    ///
    /// These are the objects that, if synced, would affect query results.
    /// Only includes ObjectIds that:
    /// 1. Come from an IndexScanNode (source of all objects)
    /// 2. Survive all filtering/joins to appear in the output
    ///
    /// After calling `settle()`, this method returns the (ObjectId, BranchName) pairs
    /// for all rows currently in the query result.
    pub fn contributing_object_ids(&self) -> HashSet<(ObjectId, BranchName)> {
        self.current_output_scope().cloned().unwrap_or_default()
    }

    /// Returns ObjectIds that must be synced for the client to reproduce the
    /// current query result locally.
    pub fn sync_scope_object_ids(&self) -> HashSet<(ObjectId, BranchName)> {
        if self.pagination_node.is_none() {
            return self.current_output_scope().cloned().unwrap_or_default();
        }

        self.scope_from_tuples(&self.sync_scope_tuples())
    }

    /// Returns a borrowed sync scope when it is maintained directly by the
    /// output node. Paginated queries need the prefix before windowing, so they
    /// still compute scope from the pagination node's sync input.
    pub fn sync_scope_object_ids_ref(&self) -> Option<&HashSet<(ObjectId, BranchName)>> {
        if self.pagination_node.is_some() {
            return None;
        }
        self.current_output_scope()
    }

    /// Returns tuples that must be synced for the client to reproduce the current
    /// query result locally.
    pub fn sync_scope_tuples(&self) -> Vec<Tuple> {
        if let Some(node_id) = self.pagination_node
            && let Some(GraphNode::LimitOffset(limit_offset)) = self.get_node(node_id)
        {
            return limit_offset
                .sync_input_tuples()
                .iter()
                .map(|tuple| self.with_deferred_includes(tuple))
                .collect();
        }

        self.current_output_tuples()
    }

    /// A pagination-prefix tuple as the client must receive it: when the page's
    /// includes are built past the pagination node, a row of the page is taken from the
    /// last include node, carrying the provenance of its included rows; rows ahead of
    /// the page (the offset) have no includes and stay as they are.
    fn deferred_includes_cover(&self, tuple: &Tuple) -> bool {
        self.deferred_arrays_tail
            .and_then(|tail| match self.get_node(tail) {
                Some(GraphNode::ArraySubquery(node)) => Some(node),
                _ => None,
            })
            .is_some_and(|node| RowNode::current_tuples(node).contains(tuple))
    }

    fn with_deferred_includes(&self, tuple: &Tuple) -> Tuple {
        if let Some(tail) = self.deferred_arrays_tail
            && let Some(GraphNode::ArraySubquery(node)) = self.get_node(tail)
            && let Some(full) = RowNode::current_tuples(node).get(tuple)
        {
            return full.clone();
        }
        tuple.clone()
    }

    /// Returns sync-scope tuples after applying a caller-provided visibility
    /// filter. For paginated queries, filtering must happen before selecting the
    /// ordered prefix so denied rows do not count toward offset/limit replay.
    ///
    /// The flag is set when a row of the page lies past the rows whose includes were
    /// built — the page's includes are built past the pagination node and rows ahead
    /// of it were denied, so the page runs beyond the graph's own window. Such a row is
    /// returned without its included rows; the caller must build the includes for
    /// every matching row instead (`IncludePlacement::MatchingRows`).
    ///
    /// The position is that of the prefix's last tuple in the ordered input when the
    /// prefix filled the page, `None` when the input ran out first. A window that stops
    /// short of it (`grow_windows_short_of`) may hold back readable rows the page needs.
    pub fn filtered_sync_scope_tuples(
        &self,
        mut tuple_is_visible: impl FnMut(&Tuple) -> bool,
    ) -> (Vec<Tuple>, bool, Option<usize>) {
        if let Some(node_id) = self.pagination_node
            && let Some(GraphNode::LimitOffset(limit_offset)) = self.get_node(node_id)
        {
            let ordered_input = self
                .get_inputs(node_id)
                .first()
                .and_then(|dep| match self.get_node(*dep) {
                    Some(GraphNode::Sort(sort_node)) => Some(sort_node.sorted_tuples()),
                    _ => None,
                })
                .unwrap_or_else(|| limit_offset.sync_input_tuples());
            // Only the first `offset + limit` visible tuples make the prefix, so the
            // visibility check stops once it has them. Checking the rest of the ordered
            // input — every row the query's filters match, not just the page — costs a
            // row load and a policy evaluation per row on every settle, which made one
            // write into a large table cost O(matching rows) per live page subscription.
            let cap = limit_offset.sync_prefix_cap();
            let mut visible_prefix = Vec::new();
            let mut includes_past_page = false;
            let mut filled_at = None;
            for (position, tuple) in ordered_input.iter().enumerate() {
                if cap.is_some_and(|cap| visible_prefix.len() >= cap) {
                    break;
                }
                let tuple = self.with_deferred_includes(tuple);
                if tuple_is_visible(&tuple) {
                    includes_past_page |= self.deferred_arrays_tail.is_some()
                        && visible_prefix.len() >= limit_offset.offset()
                        && !self.deferred_includes_cover(&tuple);
                    visible_prefix.push(tuple);
                    if cap.is_some_and(|cap| visible_prefix.len() == cap) {
                        filled_at = Some(position);
                    }
                }
            }
            return (visible_prefix, includes_past_page, filled_at);
        }

        let tuples = self
            .current_output_tuples()
            .into_iter()
            .filter(|tuple| tuple_is_visible(tuple))
            .collect();
        (tuples, false, None)
    }

    fn scope_from_tuples(&self, tuples: &[Tuple]) -> HashSet<(ObjectId, BranchName)> {
        tuples
            .iter()
            .flat_map(|tuple| tuple.provenance().iter().copied())
            .collect()
    }

    fn current_output_scope(&self) -> Option<&HashSet<(ObjectId, BranchName)>> {
        match self.get_node(self.output_node) {
            Some(GraphNode::Output(node)) => Some(node.sync_scope()),
            _ => None,
        }
    }

    /// Get current result from output node.
    pub fn current_result(&self) -> Vec<Row> {
        self.current_output_rows_with_provenance()
            .into_iter()
            .map(|(row, _)| row)
            .collect()
    }

    /// Get the current output tuples in output order.
    pub fn current_output_tuples(&self) -> Vec<Tuple> {
        self.current_output_tuples_ref().to_vec()
    }

    /// Borrow the current output tuples in output order.
    pub fn current_output_tuples_ref(&self) -> &[Tuple] {
        match self.get_node(self.output_node) {
            Some(GraphNode::Output(node)) => node.ordered_tuples(),
            _ => &[],
        }
    }

    pub(crate) fn current_output_rows_with_provenance(&self) -> Vec<(Row, TupleProvenance)> {
        self.current_output_tuples()
            .into_iter()
            .filter_map(|tuple| {
                let row = if tuple.len() == 1 {
                    tuple.to_single_row()
                } else {
                    tuple
                        .flatten_with_descriptors(
                            &self.table_descriptors,
                            &self.combined_descriptor,
                        )
                        .and_then(|flattened| flattened.to_single_row())
                }?;
                Some((row, tuple.provenance().clone()))
            })
            .collect()
    }

    /// Returns all current output rows as a RowDelta with everything in `added`.
    /// Used for first delivery after tier-gated settlement.
    pub fn current_result_as_delta(&self) -> RowDelta {
        let output_tuples = self.current_output_tuples();

        if output_tuples.is_empty() {
            return RowDelta::default();
        }

        let td = TupleDelta {
            added: output_tuples,
            removed: vec![],
            moved: vec![],
            updated: vec![],
        };

        if self.table_descriptors.len() == 1 {
            td.to_row_delta().unwrap_or_default()
        } else {
            td.flatten_to_row_delta(&self.table_descriptors, &self.combined_descriptor)
                .unwrap_or_default()
        }
    }

    // ========================================================================
    // Memory profiling
    // ========================================================================

    /// Estimate memory size of this QueryGraph.
    pub fn estimate_memory_size(&self) -> usize {
        let mut size = std::mem::size_of::<Self>();

        // Nodes Vec with CompactNode (node + inline edges)
        for compact in &self.nodes {
            size += std::mem::size_of::<CompactNode>() + 512; // estimate node size
            size += compact.inputs.len() * std::mem::size_of::<NodeId>();
            size += compact.outputs.len() * std::mem::size_of::<NodeId>();
        }

        // Dirty bitmap (1 bit per node)
        size += self.dirty_bitmap.len() / 8 + 1;

        // Table name (interned - shared, but count the string length for this ref)
        size += self.table.as_str().len();

        // Index scan nodes (interned - pointer sized, but count string lengths for reference)
        for (_, table, col) in &self.index_scan_nodes {
            size += std::mem::size_of::<NodeId>() + table.as_str().len() + col.as_str().len();
        }

        // Array subquery tables
        for (_, table) in &self.array_subquery_tables {
            size += std::mem::size_of::<NodeId>() + table.as_str().len();
        }

        // Policy filter tables
        for (_, table) in &self.policy_filter_tables {
            size += std::mem::size_of::<NodeId>() + table.as_str().len();
        }

        // Magic column tables
        for (_, table) in &self.magic_column_tables {
            size += std::mem::size_of::<NodeId>() + table.as_str().len();
        }

        // Recursive relation tables
        for (_, table) in &self.recursive_relation_tables {
            size += std::mem::size_of::<NodeId>() + table.as_str().len();
        }

        // Table descriptors - estimate 200 bytes per descriptor
        size += self.table_descriptors.len() * 200;

        // Combined descriptor
        size += 200;

        size
    }
}
