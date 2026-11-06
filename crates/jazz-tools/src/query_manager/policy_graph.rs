//! PolicyGraph - one-shot graphs for policy evaluation.
//!
//! Creates minimal query graphs to evaluate policy conditions like USING and INHERITS.
//! These graphs are throwaway - created, settled until complete, then discarded.

use crate::object::ObjectId;
use crate::sync_manager::RowBatchKey;
use std::collections::HashMap;
use std::sync::Arc;

use crate::storage::Storage;

use crate::schema_manager::transformer::translate_table_and_column_for_schema;
use crate::schema_manager::{SchemaContext, translate_table_name_to_schema};

use super::graph::{GraphNode, QueryGraph, RelationCompileFeatures};
use super::graph_nodes::NodeId;
use super::graph_nodes::exists_output::ExistsOutputNode;
use super::graph_nodes::index_scan::IndexScanNode;
use super::graph_nodes::materialize::MaterializeNode;
use super::graph_nodes::policy_filter::{PolicyFilterNode, PolicyFilterOptions};
use super::graph_nodes::union::UnionNode;
use super::index::ScanCondition;
use super::policy::{
    CmpOp, Operation, PolicyExpr, PolicyValue, resolve_session_value, uuid_from_text,
};
use super::relation_ir::RelExpr;
use super::relation_ir_query_plan::lower_relation_to_execution_plan;
use super::session::Session;
use super::settlement_eval_cache::SettlementEvalCache;
use super::types::{ColumnName, SchemaHash};
use super::types::{
    ColumnType, LoadedRow, RowPolicyMode, Schema, TableName, TupleDescriptor, Value,
};

/// The generation every branch of a write check's EXISTS universe holds its rows in: the
/// branches as the main context's `branch_schema_map` names them, each generation read from
/// the authorization context. What narrows the scan of each branch to the rows one indexed
/// equality of the condition can still match, and names the table as that generation
/// stores it.
#[derive(Clone, Copy)]
pub(crate) struct GenerationScans<'a> {
    pub(crate) context: &'a SchemaContext,
    pub(crate) branch_schemas: &'a HashMap<String, SchemaHash>,
}

impl GenerationScans<'_> {
    /// The scan of `branch` narrowed to one top-level equality of `condition`, in that
    /// branch's own table and column names: `None` keeps the full scan. The PolicyFilter
    /// above judges every row by the whole condition all the same, so a narrowed scan only
    /// has to keep every row of the branch the equality can match. An index lists every
    /// row under its whole value (`indices.rs`) and the lenses carry values through
    /// unchanged. The policy's equality is strict but for one conversion: a uuid is
    /// compared with the uuid a text value spells (`uuid_from_text`), as a session's
    /// `user_id` names a row's uuid reference. So an index lookup of the value, converted
    /// the same way where the column holds uuids, keeps them all where:
    /// - the value is not a null: an index holds no entry for one, and the policy's
    ///   equality matches a null to a null;
    /// - the lenses carry the column to the branch's generation, under the name it has
    ///   there, and that generation declares and indexes it with the type the policy's
    ///   generation declares; where a lens on the way adds or removes the column, a column
    ///   of the same name on the older generation is another column, whose values the
    ///   lenses do not serve;
    /// - `id` is the row id, a uuid, only where the table declares no `id` column of its
    ///   own;
    /// - the branch holds no rows of the write's own batch in another layout than the
    ///   policy's: the scan matches those by decoding them in the policy's layout.
    fn narrowed_scan(
        &self,
        table: &TableName,
        condition: &PolicyExpr,
        session: &Session,
        branch: &str,
        local_overlay_rows: Option<&HashMap<ObjectId, RowBatchKey>>,
    ) -> Option<(TableName, ColumnName, ScanCondition)> {
        #[cfg(test)]
        if WHOLE_SCANS.with(std::cell::Cell::get) {
            return None;
        }
        let hash = *self.branch_schemas.get(branch)?;
        if hash != self.context.current_hash
            && local_overlay_rows
                .is_some_and(|rows| rows.values().any(|key| key.branch_name.as_str() == branch))
        {
            return None;
        }
        let generation = self.context.get_schema(&hash)?;
        let declared = self.context.current_schema.get(table)?;
        let mut conjuncts = Vec::new();
        top_level_conjuncts(condition, &mut conjuncts);
        conjuncts.into_iter().find_map(|conjunct| {
            let PolicyExpr::Cmp {
                column,
                op: CmpOp::Eq,
                value,
            } = conjunct
            else {
                return None;
            };
            let value = match value {
                PolicyValue::Literal(value) => value.clone(),
                PolicyValue::SessionRef(path) => resolve_session_value(path, session)?,
            };
            if value.is_null() {
                return None;
            }
            let Some(declared_column) = declared.columns.column(column) else {
                if column != "id" {
                    return None;
                }
                let scan_table =
                    translate_table_name_to_schema(self.context, table.as_str(), &hash)?;
                return Some((
                    TableName::new(&scan_table),
                    ColumnName::new("_id"),
                    ScanCondition::Eq(uuid_from_text(value)),
                ));
            };
            // With no lens path at all, a table translates only to itself under identical
            // columns, so the column keeps its name. A path that declines the column is
            // final: the full scan stays.
            let (scan_table, scan_column) = match translate_table_and_column_for_schema(
                self.context,
                table.as_str(),
                column,
                &hash,
            ) {
                Some(translated) => translated,
                None if self.context.lens_path(&hash).is_err() => (
                    translate_table_name_to_schema(self.context, table.as_str(), &hash)?,
                    column.clone(),
                ),
                None => return None,
            };
            let scan_table_schema = generation.get(&TableName::new(&scan_table))?;
            let scan_column_type = &scan_table_schema.columns.column(&scan_column)?.column_type;
            if *scan_column_type != declared_column.column_type
                || !scan_table_schema.is_indexed_column(&scan_column)
            {
                return None;
            }
            let value = match scan_column_type {
                ColumnType::Uuid => uuid_from_text(value),
                _ => value,
            };
            Some((
                TableName::new(&scan_table),
                ColumnName::new(&scan_column),
                ScanCondition::Eq(value),
            ))
        })
    }
}

// Set while a write check's EXISTS is judged again over whole scans, on this thread: the
// verdict a narrowed scan has to match.
#[cfg(test)]
thread_local! {
    pub(crate) static WHOLE_SCANS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whole scans on this thread while it lives, a panic included.
#[cfg(test)]
pub(crate) struct WholeScans;

#[cfg(test)]
impl WholeScans {
    pub(crate) fn begin() -> Self {
        WHOLE_SCANS.with(|whole| whole.set(true));
        Self
    }
}

#[cfg(test)]
impl Drop for WholeScans {
    fn drop(&mut self) {
        WHOLE_SCANS.with(|whole| whole.set(false));
    }
}

impl GenerationScans<'_> {
    /// The table a whole scan of `table` reads on `branch`: the name `branch`'s generation
    /// stores it by, as a renamed table's older rows live under its old name there. `None`
    /// where a lens path to that generation declines the table: it was added on the path,
    /// so the generation holds no rows of it, and its current name there may be another
    /// table's, one a rename on the path freed it from, whose rows the loader serves as
    /// that other table. With no generation known for the branch, or no lens path to it,
    /// the table keeps its name.
    fn whole_scan_table(&self, table: &TableName, branch: &str) -> Option<TableName> {
        let Some(hash) = self.branch_schemas.get(branch) else {
            return Some(*table);
        };
        match translate_table_name_to_schema(self.context, table.as_str(), hash) {
            Some(name) => Some(TableName::new(&name)),
            None if self.context.lens_path(hash).is_err() => Some(*table),
            None => None,
        }
    }
}

fn top_level_conjuncts<'e>(condition: &'e PolicyExpr, conjuncts: &mut Vec<&'e PolicyExpr>) {
    match condition {
        PolicyExpr::And(exprs) => {
            for expr in exprs {
                top_level_conjuncts(expr, conjuncts);
            }
        }
        other => conjuncts.push(other),
    }
}

/// A one-shot graph for evaluating a policy condition.
///
/// Policy graphs are minimal graphs built specifically to evaluate
/// whether a condition is met (EXISTS-style check).
#[derive(Debug)]
pub struct PolicyGraph {
    /// The underlying query graph.
    graph: QueryGraph,
    /// The ExistsOutput node ID.
    exists_node: NodeId,
    /// Table name this graph operates on.
    table: TableName,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PolicyGraphBuildOptions<'a> {
    branch: &'a str,
    initial_depth: usize,
    row_policy_mode: RowPolicyMode,
}

impl<'a> PolicyGraphBuildOptions<'a> {
    pub(crate) fn new(branch: &'a str, row_policy_mode: RowPolicyMode) -> Self {
        Self {
            branch,
            initial_depth: 0,
            row_policy_mode,
        }
    }

    pub(crate) fn with_initial_depth(mut self, initial_depth: usize) -> Self {
        self.initial_depth = initial_depth;
        self
    }
}

impl PolicyGraph {
    fn exists_rel_output_table(
        rel: &RelExpr,
        branches: &[String],
        schema: &Schema,
    ) -> Option<TableName> {
        let plan =
            lower_relation_to_execution_plan(rel, branches, false, Vec::new(), None, schema)?;
        match plan.result_element_index {
            None | Some(0) => Some(plan.table),
            Some(index) => plan.joins.get(index - 1).map(|join| join.table),
        }
    }

    /// Create a graph for USING check: can session see this specific row?
    ///
    /// Graph structure: IndexScan(_id = objectId) → Materialize → PolicyFilter → ExistsOutput
    ///
    /// Returns None if the table is not in the schema.
    pub fn for_using_check(
        table: &TableName,
        object_id: ObjectId,
        policy: &PolicyExpr,
        session: &Session,
        schema: &Schema,
        branch: &str,
        row_policy_mode: RowPolicyMode,
    ) -> Option<Self> {
        Self::for_using_check_with_options(
            table,
            object_id,
            policy,
            session,
            schema,
            PolicyGraphBuildOptions::new(branch, row_policy_mode),
        )
    }

    /// Create a graph for USING check with explicit build options.
    fn for_using_check_with_options(
        table: &TableName,
        object_id: ObjectId,
        policy: &PolicyExpr,
        session: &Session,
        schema: &Schema,
        options: PolicyGraphBuildOptions<'_>,
    ) -> Option<Self> {
        let table_schema = schema.get(table)?;
        let descriptor = table_schema.columns.clone();

        let mut graph = QueryGraph::new(*table, descriptor.clone());

        // IndexScan node: scan _id index for exact match
        let id_column = ColumnName::new("_id");
        let scan_node = IndexScanNode::new_with_branch(
            *table,
            id_column,
            options.branch,
            ScanCondition::Eq(Value::Uuid(object_id)),
            descriptor.clone(),
        );
        let scan_id = graph.add_node_with_id(GraphNode::IndexScan(scan_node));
        graph.index_scan_nodes.push((scan_id, *table, id_column));

        // Materialize node: load row content. Carry the resolved table name so
        // `LensTransformer::new` can translate old-branch rows; an empty hint
        // would degrade to `TableNotFound("")` and silently drop the row.
        let tuple_desc = TupleDescriptor::single(*table, descriptor.clone());
        let mat_node = MaterializeNode::new_all(tuple_desc);
        let mat_id = graph.add_node_with_id(GraphNode::Materialize(mat_node));
        graph.add_edge(mat_id, scan_id);

        // PolicyFilter node: evaluate policy against row
        let policy_node = PolicyFilterNode::new_with_options(
            descriptor.clone(),
            policy.clone(),
            session.clone(),
            Arc::new(schema.clone()),
            table.as_str(),
            PolicyFilterOptions::for_branch(options.branch)
                .with_initial_depth(options.initial_depth)
                .with_row_policy_mode(options.row_policy_mode),
        );
        let policy_id = graph.add_node_with_id(GraphNode::PolicyFilter(policy_node));
        graph.add_edge(policy_id, mat_id);

        // ExistsOutput node: track whether any rows pass
        let exists_node = ExistsOutputNode::new(descriptor);
        let exists_id = graph.add_node_with_id(GraphNode::ExistsOutput(exists_node));
        graph.add_edge(exists_id, policy_id);

        graph.output_node = exists_id;

        Some(Self {
            graph,
            exists_node: exists_id,
            table: *table,
        })
    }

    /// Create a graph for INHERITS: does parent row pass parent's policy?
    ///
    /// Graph structure: IndexScan(parent_table, _id = parent_id) → Materialize → PolicyFilter → ExistsOutput
    ///
    /// Returns None if the parent table is not in the schema.
    pub(crate) fn for_inherits(
        parent_table: &TableName,
        parent_id: ObjectId,
        parent_policy: &PolicyExpr,
        session: &Session,
        schema: &Schema,
        options: PolicyGraphBuildOptions<'_>,
    ) -> Option<Self> {
        // INHERITS is essentially the same as a USING check on the parent table
        Self::for_using_check_with_options(
            parent_table,
            parent_id,
            parent_policy,
            session,
            schema,
            options,
        )
    }

    /// Create a graph for EXISTS: does any row in table match condition?
    ///
    /// Graph structure: IndexScan(All) per branch (+Union) → Materialize →
    /// PolicyFilter → ExistsOutput
    ///
    /// The scan unions over every sanctioned branch, mirroring what the data
    /// path of the enclosing query does: a supporting policy row born under
    /// an old schema generation lives on the old branch, and an exists arm
    /// that scans only the first branch silently denies the rows it guards
    /// (defect 23). EXISTS is disjunctive, so the union cannot deny a row any
    /// single-branch scan would have granted — and a session with no
    /// satisfying row on ANY queried branch still gets nothing.
    ///
    /// Returns None if the table is not in the schema or no branch is given.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn for_exists(
        table: &TableName,
        condition: &PolicyExpr,
        session: &Session,
        schema: &Schema,
        branches: &[String],
        policy_operation: Operation,
        row_policy_mode: RowPolicyMode,
        generations: Option<GenerationScans<'_>>,
        local_overlay_rows: Option<&HashMap<ObjectId, RowBatchKey>>,
    ) -> Option<Self> {
        let table_schema = schema.get(table)?;
        let descriptor = table_schema.columns.clone();

        let mut graph = QueryGraph::new(*table, descriptor.clone());

        // IndexScan nodes, one per branch: what one equality of the condition narrows the
        // branch to, or the whole table.
        let id_column = ColumnName::new("_id");
        let mut scan_ids = Vec::new();
        for branch in branches {
            let (scan_table, scan_column, scan_condition) = generations
                .and_then(|generations| {
                    generations.narrowed_scan(table, condition, session, branch, local_overlay_rows)
                })
                .unwrap_or_else(|| {
                    match generations.map(|generations| generations.whole_scan_table(table, branch))
                    {
                        Some(None) => (*table, id_column, ScanCondition::Empty),
                        Some(Some(scan_table)) => (scan_table, id_column, ScanCondition::All),
                        None => (*table, id_column, ScanCondition::All),
                    }
                });
            let scan_node = IndexScanNode::new_with_branch(
                scan_table,
                scan_column,
                branch,
                scan_condition,
                descriptor.clone(),
            );
            let scan_id = graph.add_node_with_id(GraphNode::IndexScan(scan_node));
            graph
                .index_scan_nodes
                .push((scan_id, scan_table, scan_column));
            scan_ids.push(scan_id);
        }
        let scan_output = if scan_ids.len() > 1 {
            let union_node = UnionNode::new();
            let union_id = graph.add_node_with_id(GraphNode::Union(union_node));
            for scan_id in scan_ids {
                graph.add_edge(union_id, scan_id);
            }
            union_id
        } else {
            *scan_ids.first()?
        };

        // Materialize node: load row content. Carry the resolved table name so
        // `LensTransformer::new` can translate old-branch rows; an empty hint
        // would degrade to `TableNotFound("")` and silently drop the row.
        let tuple_desc = TupleDescriptor::single(*table, descriptor.clone());
        let mat_node = MaterializeNode::new_all(tuple_desc);
        let mat_id = graph.add_node_with_id(GraphNode::Materialize(mat_node));
        graph.add_edge(mat_id, scan_output);

        // PolicyFilter node: evaluate condition against each row
        let policy_node = PolicyFilterNode::new_with_options(
            descriptor.clone(),
            condition.clone(),
            session.clone(),
            Arc::new(schema.clone()),
            table.as_str(),
            PolicyFilterOptions::for_branches(branches.to_vec())
                .with_row_policy_mode(row_policy_mode)
                .with_policy_operation(policy_operation),
        );
        let policy_id = graph.add_node_with_id(GraphNode::PolicyFilter(policy_node));
        graph.add_edge(policy_id, mat_id);

        // ExistsOutput node: track whether any rows pass
        let exists_node = ExistsOutputNode::new(descriptor);
        let exists_id = graph.add_node_with_id(GraphNode::ExistsOutput(exists_node));
        graph.add_edge(exists_id, policy_id);

        graph.output_node = exists_id;

        Some(Self {
            graph,
            exists_node: exists_id,
            table: *table,
        })
    }

    /// Create a graph for declarative EXISTS relation checks.
    ///
    /// Compiles relation IR through the shared query planner, then appends an
    /// ExistsOutput node over the compiled query output.
    pub fn for_exists_rel(
        rel: &RelExpr,
        schema: &Schema,
        branches: &[String],
        session: Option<Session>,
        row_policy_mode: RowPolicyMode,
        current_table: Option<&TableName>,
        structural_scans: bool,
    ) -> Option<Self> {
        let use_structural_rows = structural_scans
            || current_table
                .and_then(|table| {
                    Self::exists_rel_output_table(rel, branches, schema)
                        .map(|output| output == *table)
                })
                .unwrap_or(false);
        let compile_schema: Schema = if use_structural_rows {
            // Same-table relation closures and explicitly structural policy
            // checks must inspect raw relation rows rather than re-running
            // SELECT policies on the scanned tables.
            schema
                .iter()
                .map(|(table_name, table_schema)| {
                    let mut structural = table_schema.clone();
                    structural.policies = crate::query_manager::types::TablePolicies::default();
                    (*table_name, structural)
                })
                .collect()
        } else {
            schema.clone()
        };
        // The relation compiles against the full sanctioned branch set — the
        // shared planner unions its scans per branch, exactly like the
        // enclosing query's data path.
        let schema_context = SchemaContext::with_defaults(compile_schema.clone(), "main");
        let mut graph = QueryGraph::compile_relation_ir_with_schema_context_and_features(
            rel,
            &compile_schema,
            branches,
            session,
            &schema_context,
            // A scan of an older generation lists a row by that copy's value, while the
            // row it loads is the newest version across the universe: an index that covers
            // the condition does not vouch for the row the policy is handed.
            RelationCompileFeatures {
                keep_residual_filter: true,
                ..RelationCompileFeatures::default()
            },
            if use_structural_rows {
                RowPolicyMode::PermissiveLocal
            } else {
                row_policy_mode
            },
            // A policy's relation reads no declared index: authorization answers must
            // not depend on what a store has filled.
            &std::sync::Arc::new(
                crate::query_manager::index_declarations::IndexDeclarations::empty(),
            ),
        )?;
        let output_descriptor = match graph
            .nodes
            .get(graph.output_node.0 as usize)
            .map(|c| &c.node)
        {
            Some(GraphNode::Output(node)) => node.output_tuple_descriptor().combined_descriptor(),
            _ => return None,
        };

        let exists_node = ExistsOutputNode::new(output_descriptor);
        let exists_id = graph.add_node_with_id(GraphNode::ExistsOutput(exists_node));
        graph.add_edge(exists_id, graph.output_node);
        graph.output_node = exists_id;

        Some(Self {
            table: graph.table,
            graph,
            exists_node: exists_id,
        })
    }

    /// Settle the graph. With synchronous Storage, always completes in one pass.
    ///
    /// The row_loader trait object is used to fetch row content by ObjectId and
    /// optional table hint.
    /// Using trait object instead of generic to avoid recursion limit when
    /// INHERITS evaluation calls this method.
    pub fn settle(
        &mut self,
        io: &dyn Storage,
        row_loader: &mut dyn FnMut(ObjectId, Option<TableName>) -> Option<LoadedRow>,
    ) -> bool {
        self.settle_with_settlement_eval_cache(io, None, row_loader)
    }

    pub(crate) fn settle_with_settlement_eval_cache(
        &mut self,
        io: &dyn Storage,
        settlement_eval_cache: Option<&mut SettlementEvalCache>,
        row_loader: &mut dyn FnMut(ObjectId, Option<TableName>) -> Option<LoadedRow>,
    ) -> bool {
        self.settle_with_overlay(io, None, settlement_eval_cache, row_loader)
    }

    /// [`Self::settle_with_settlement_eval_cache`], with the scans also seeing
    /// `local_overlay_rows`. A result read through an overlay holds for that overlay only,
    /// so the two are never combined.
    pub(crate) fn settle_with_overlay(
        &mut self,
        io: &dyn Storage,
        local_overlay_rows: Option<&HashMap<ObjectId, RowBatchKey>>,
        settlement_eval_cache: Option<&mut SettlementEvalCache>,
        row_loader: &mut dyn FnMut(ObjectId, Option<TableName>) -> Option<LoadedRow>,
    ) -> bool {
        assert!(
            local_overlay_rows.is_none() || settlement_eval_cache.is_none(),
            "a policy result read through a batch overlay must not be cached"
        );
        let _delta = self.graph.settle_with_overlay_and_cache(
            io,
            local_overlay_rows,
            settlement_eval_cache,
            &mut |id, hint| row_loader(id, hint),
        );
        true
    }

    /// Get result.
    ///
    /// Returns true if at least one row passed the policy check.
    pub fn result(&self) -> bool {
        match self
            .graph
            .nodes
            .get(self.exists_node.0 as usize)
            .map(|c| &c.node)
        {
            Some(GraphNode::ExistsOutput(node)) => node.exists(),
            _ => false,
        }
    }

    /// Get the table this graph operates on.
    pub fn table(&self) -> &TableName {
        &self.table
    }

    /// Mark all scan nodes dirty (for re-evaluation after data changes).
    pub fn mark_dirty(&mut self) {
        let scan_ids: Vec<NodeId> = self
            .graph
            .index_scan_nodes
            .iter()
            .map(|(node_id, _, _)| *node_id)
            .collect();
        for node_id in scan_ids {
            self.graph.mark_dirty(node_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query_manager::relation_ir::{
        ColumnRef, JoinCondition, JoinKind, KeyRef, PredicateCmpOp, PredicateExpr, ProjectColumn,
        ProjectExpr, RelExpr, RowIdRef, ValueRef,
    };
    use crate::query_manager::types::{
        ColumnDescriptor, ColumnType, RowDescriptor, TablePolicies, TableSchema,
    };

    fn test_schema() -> Schema {
        let mut schema = Schema::new();

        // documents table with owner_id policy
        let docs_descriptor = RowDescriptor::new(vec![
            ColumnDescriptor::new("_id", ColumnType::Uuid),
            ColumnDescriptor::new("owner_id", ColumnType::Text),
            ColumnDescriptor::new("title", ColumnType::Text),
        ]);

        let docs_policies = TablePolicies::new()
            .with_select(PolicyExpr::eq_session("owner_id", vec!["user_id".into()]));

        schema.insert(
            TableName::new("documents"),
            TableSchema::with_policies(docs_descriptor, docs_policies),
        );

        schema
    }

    #[test]
    fn test_for_using_check_creates_graph() {
        let schema = test_schema();
        let session = Session::new("user1");
        let object_id = ObjectId::new();
        let table = TableName::new("documents");

        let policy = PolicyExpr::eq_session("owner_id", vec!["user_id".into()]);

        let policy_graph = PolicyGraph::for_using_check(
            &table,
            object_id,
            &policy,
            &session,
            &schema,
            "main",
            RowPolicyMode::PermissiveLocal,
        );

        assert!(policy_graph.is_some());

        let pg = policy_graph.unwrap();
        // Graph should have 4 nodes: IndexScan, Materialize, PolicyFilter, ExistsOutput
        assert_eq!(pg.graph.nodes.len(), 4);
    }

    #[test]
    fn test_for_using_check_returns_none_for_missing_table() {
        let schema = test_schema();
        let session = Session::new("user1");
        let object_id = ObjectId::new();
        let table = TableName::new("nonexistent");

        let policy = PolicyExpr::True;

        let policy_graph = PolicyGraph::for_using_check(
            &table,
            object_id,
            &policy,
            &session,
            &schema,
            "main",
            RowPolicyMode::PermissiveLocal,
        );

        assert!(policy_graph.is_none());
    }

    #[test]
    fn test_policy_graph_initial_state() {
        let schema = test_schema();
        let session = Session::new("user1");
        let object_id = ObjectId::new();
        let table = TableName::new("documents");

        let policy = PolicyExpr::True;

        let pg = PolicyGraph::for_using_check(
            &table,
            object_id,
            &policy,
            &session,
            &schema,
            "main",
            RowPolicyMode::PermissiveLocal,
        )
        .unwrap();

        // Before settling, result should be false (no rows yet)
        // But it might be pending since we haven't settled
        assert!(!pg.result());
    }

    #[test]
    fn test_policy_graph_with_true_policy() {
        let schema = test_schema();
        let session = Session::new("user1");
        let object_id = ObjectId::new();
        let table = TableName::new("documents");

        // PolicyExpr::True should always pass
        let policy = PolicyExpr::True;

        let mut pg = PolicyGraph::for_using_check(
            &table,
            object_id,
            &policy,
            &session,
            &schema,
            "main",
            RowPolicyMode::PermissiveLocal,
        )
        .unwrap();

        // With no actual data in storage, the scan will return no rows
        let storage = crate::storage::MemoryStorage::new();

        // Row loader returns None for all IDs (no data)
        let mut row_loader =
            |_id: ObjectId, _hint: Option<TableName>| -> Option<LoadedRow> { None };

        // Settle the graph
        pg.settle(&storage, &mut row_loader);

        // No rows found (object doesn't exist in empty OM), so result is false
        assert!(!pg.result());
    }

    #[test]
    fn test_for_exists_rel_creates_graph() {
        let schema = test_schema();
        let rel = RelExpr::Filter {
            input: Box::new(RelExpr::TableScan {
                table: TableName::new("documents"),
            }),
            predicate: PredicateExpr::Cmp {
                left: ColumnRef::unscoped("owner_id"),
                op: PredicateCmpOp::Eq,
                right: ValueRef::Literal(Value::Text("user1".to_string())),
            },
        };

        let graph = PolicyGraph::for_exists_rel(
            &rel,
            &schema,
            &["main".to_string()],
            None,
            RowPolicyMode::PermissiveLocal,
            None,
            false,
        );
        assert!(graph.is_some(), "exists-rel graph should compile");

        let graph = graph.expect("graph");
        assert_eq!(graph.table().as_str(), "documents");
        assert!(matches!(
            graph
                .graph
                .nodes
                .get(graph.exists_node.0 as usize)
                .map(|c| &c.node),
            Some(GraphNode::ExistsOutput(_))
        ));
    }

    #[test]
    fn test_for_exists_rel_ignores_select_policies_on_relation_tables() {
        let schema = test_schema();
        let session = Session::new("user1");
        let current_table = TableName::new("documents");
        let rel = RelExpr::Filter {
            input: Box::new(RelExpr::TableScan {
                table: TableName::new("documents"),
            }),
            predicate: PredicateExpr::Cmp {
                left: ColumnRef::unscoped("owner_id"),
                op: PredicateCmpOp::Eq,
                right: ValueRef::Literal(Value::Text("user1".to_string())),
            },
        };

        let graph = PolicyGraph::for_exists_rel(
            &rel,
            &schema,
            &["main".to_string()],
            Some(session),
            RowPolicyMode::Enforcing,
            Some(&current_table),
            false,
        )
        .expect("exists-rel graph");

        assert!(
            !graph
                .graph
                .nodes
                .iter()
                .any(|ctx| matches!(ctx.node, GraphNode::PolicyFilter(_))),
            "exists-rel evaluation should inspect raw relation rows without injecting select-policy filters"
        );
    }

    #[test]
    fn test_for_exists_rel_keeps_select_policies_for_other_output_tables() {
        let schema = test_schema();
        let current_table = TableName::new("projects");
        let rel = RelExpr::Filter {
            input: Box::new(RelExpr::TableScan {
                table: TableName::new("documents"),
            }),
            predicate: PredicateExpr::Cmp {
                left: ColumnRef::unscoped("owner_id"),
                op: PredicateCmpOp::Eq,
                right: ValueRef::Literal(Value::Text("user1".to_string())),
            },
        };

        let graph = PolicyGraph::for_exists_rel(
            &rel,
            &schema,
            &["main".to_string()],
            Some(Session::new("user1")),
            RowPolicyMode::Enforcing,
            Some(&current_table),
            false,
        )
        .expect("exists-rel graph");

        assert!(
            graph
                .graph
                .nodes
                .iter()
                .any(|ctx| matches!(ctx.node, GraphNode::PolicyFilter(_))),
            "different-table exists-rel checks should still honor scanned-table select policies by default"
        );
    }

    #[test]
    fn test_for_exists_rel_can_force_structural_scans_for_other_output_tables() {
        let schema = test_schema();
        let current_table = TableName::new("projects");
        let rel = RelExpr::Filter {
            input: Box::new(RelExpr::TableScan {
                table: TableName::new("documents"),
            }),
            predicate: PredicateExpr::Cmp {
                left: ColumnRef::unscoped("owner_id"),
                op: PredicateCmpOp::Eq,
                right: ValueRef::Literal(Value::Text("user1".to_string())),
            },
        };

        let graph = PolicyGraph::for_exists_rel(
            &rel,
            &schema,
            &["main".to_string()],
            Some(Session::new("user1")),
            RowPolicyMode::Enforcing,
            Some(&current_table),
            true,
        )
        .expect("exists-rel graph");

        assert!(
            !graph
                .graph
                .nodes
                .iter()
                .any(|ctx| matches!(ctx.node, GraphNode::PolicyFilter(_))),
            "read-side exists-rel policy checks should be able to inspect raw relation rows even when the relation outputs a different table"
        );
    }

    #[test]
    fn test_for_exists_rel_with_gather_post_join_compiles() {
        let mut schema = Schema::new();
        let current_table = TableName::new("resource_access_edges");
        schema.insert(
            TableName::new("teams"),
            TableSchema::with_policies(
                RowDescriptor::new(vec![
                    ColumnDescriptor::new("_id", ColumnType::Uuid),
                    ColumnDescriptor::new("name", ColumnType::Text),
                ]),
                TablePolicies::new().with_select(PolicyExpr::True),
            ),
        );
        schema.insert(
            TableName::new("team_edges"),
            TableSchema::with_policies(
                RowDescriptor::new(vec![
                    ColumnDescriptor::new("_id", ColumnType::Uuid),
                    ColumnDescriptor::new("child_team", ColumnType::Uuid),
                    ColumnDescriptor::new("parent_team", ColumnType::Uuid),
                ]),
                TablePolicies::new().with_select(PolicyExpr::True),
            ),
        );
        schema.insert(
            TableName::new("resource_access_edges"),
            TableSchema::with_policies(
                RowDescriptor::new(vec![
                    ColumnDescriptor::new("_id", ColumnType::Uuid),
                    ColumnDescriptor::new("team", ColumnType::Uuid),
                    ColumnDescriptor::new("resource", ColumnType::Text),
                    ColumnDescriptor::new("grant_role", ColumnType::Text),
                ]),
                TablePolicies::new().with_select(PolicyExpr::True),
            ),
        );

        let rel = RelExpr::Project {
            input: Box::new(RelExpr::Filter {
                input: Box::new(RelExpr::Join {
                    left: Box::new(RelExpr::Gather {
                        seed: Box::new(RelExpr::Filter {
                            input: Box::new(RelExpr::TableScan {
                                table: TableName::new("teams"),
                            }),
                            predicate: PredicateExpr::Cmp {
                                left: ColumnRef::scoped("teams", "name"),
                                op: PredicateCmpOp::Eq,
                                right: ValueRef::Literal(Value::Text("seed".to_string())),
                            },
                        }),
                        step: Box::new(RelExpr::Project {
                            input: Box::new(RelExpr::Join {
                                left: Box::new(RelExpr::Filter {
                                    input: Box::new(RelExpr::TableScan {
                                        table: TableName::new("team_edges"),
                                    }),
                                    predicate: PredicateExpr::Cmp {
                                        left: ColumnRef::scoped("team_edges", "child_team"),
                                        op: PredicateCmpOp::Eq,
                                        right: ValueRef::RowId(RowIdRef::Frontier),
                                    },
                                }),
                                right: Box::new(RelExpr::TableScan {
                                    table: TableName::new("teams"),
                                }),
                                on: vec![JoinCondition {
                                    left: ColumnRef::scoped("team_edges", "parent_team"),
                                    right: ColumnRef::scoped("__recursive_hop_0", "id"),
                                }],
                                join_kind: JoinKind::Inner,
                            }),
                            columns: vec![ProjectColumn {
                                alias: "id".to_string(),
                                expr: ProjectExpr::Column(ColumnRef::scoped(
                                    "__recursive_hop_0",
                                    "id",
                                )),
                            }],
                        }),
                        frontier_key: KeyRef::RowId(RowIdRef::Current),
                        max_depth: 5,
                        dedupe_key: vec![KeyRef::RowId(RowIdRef::Current)],
                    }),
                    right: Box::new(RelExpr::TableScan {
                        table: TableName::new("resource_access_edges"),
                    }),
                    on: vec![JoinCondition {
                        left: ColumnRef::scoped("teams", "id"),
                        right: ColumnRef::scoped("__hop_0", "team"),
                    }],
                    join_kind: JoinKind::Inner,
                }),
                predicate: PredicateExpr::Cmp {
                    left: ColumnRef::scoped("__hop_0", "grant_role"),
                    op: PredicateCmpOp::Eq,
                    right: ValueRef::Literal(Value::Text("viewer".to_string())),
                },
            }),
            columns: vec![ProjectColumn {
                alias: "id".to_string(),
                expr: ProjectExpr::Column(ColumnRef::scoped("__hop_0", "id")),
            }],
        };

        let graph = PolicyGraph::for_exists_rel(
            &rel,
            &schema,
            &["main".to_string()],
            Some(Session::new("user1")),
            RowPolicyMode::Enforcing,
            Some(&current_table),
            false,
        );
        assert!(
            graph.is_some(),
            "gather + post-join exists-rel should compile"
        );

        let graph = graph.expect("graph");
        assert!(
            graph
                .graph
                .nodes
                .iter()
                .any(|ctx| matches!(ctx.node, GraphNode::RecursiveRelation(_)))
        );
        assert!(
            graph
                .graph
                .nodes
                .iter()
                .any(|ctx| matches!(ctx.node, GraphNode::Join(_)))
        );
        assert!(
            !graph
                .graph
                .nodes
                .iter()
                .any(|ctx| matches!(ctx.node, GraphNode::PolicyFilter(_))),
            "recursive exists-rel graphs should not inject select-policy filters for relation tables"
        );
    }
}
