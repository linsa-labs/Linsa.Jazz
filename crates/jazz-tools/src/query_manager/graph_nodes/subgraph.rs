//! Subgraph template and instance management for correlated subqueries.
//!
//! A SubgraphTemplate represents a parameterized query that can be instantiated
//! multiple times with different parameter bindings. Each instantiation creates
//! a SubgraphInstance with its own state.

use crate::query_manager::graph::QueryGraph;
use crate::query_manager::query::{Condition, Query, QueryBuildError, QueryBuilder};
use crate::query_manager::relation_ir_query_plan::ExecutionQueryPlan;
use crate::query_manager::session::Session;
use crate::query_manager::types::{RowDescriptor, RowPolicyMode, Schema, SchemaHash, Value};
use crate::schema_manager::SchemaContext;
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, OnceLock};

/// v18 item 7: a template's inner shape, lowered once.
///
/// `plan` is `lower(instance_query(Null))`: the rebuilt query with `Null` in the correlation
/// slot, lowered to an execution plan with the table and descriptor checks done. Lowering is
/// value-neutral (every step clones the value; the value-dependent work — index-scan choice,
/// scan-value normalisation — is in the compile, which stays per instance), so the plan of
/// any instance is this plan with its value in the slot.
#[derive(Debug, Clone)]
struct CachedShape {
    /// The schema the plan was lowered against; an `instantiate` with another schema lowers
    /// afresh instead of trusting the cache.
    schema: Arc<Schema>,
    plan: ExecutionQueryPlan,
    /// The correlation column as the lowering left it at position 0 of every conjunction
    /// (`_id` for a row-id correlation).
    bound_column: String,
}

/// Template for creating subgraph instances.
///
/// Holds a query definition with correlation parameters that get bound
/// when creating instances. Currently uses the "recompile per binding" approach
/// for simplicity - each instance gets a fresh query graph compiled with the
/// bound parameter values.
#[derive(Debug, Clone)]
pub struct SubgraphTemplate {
    /// Base query for the subgraph (without correlation filters applied).
    base_query: Query,
    /// Column in the inner table to correlate on.
    inner_column: String,
    /// Columns to select from inner query results.
    select_columns: Vec<String>,
    /// Output descriptor for individual result rows.
    output_descriptor: RowDescriptor,
    /// Schema context inherited from the parent graph compile.
    schema_context: Arc<SchemaContext>,
    /// Session inherited from the parent graph compile.
    session: Option<Session>,
    /// Policy mode inherited from the parent graph compile.
    row_policy_mode: RowPolicyMode,
    /// v18 item 7: how many times this template's shape was lowered to an execution plan.
    /// Shared by every clone of the template, so a gate can read it through the node.
    shape_lowerings: Arc<std::sync::atomic::AtomicU64>,
    /// v18 item 7: the shape lowered once. Set only on success: a shape that does not fit the
    /// position-0 invariant is not cached (every instance then lowers uncached, and
    /// `shape_lowerings` shows it). A clone of the template copies the cell, populated or
    /// not; both are correct because the plan is a function of the immutable fields above —
    /// and no clone exists on the hot path (`instantiate` takes `&self`).
    shape: OnceLock<CachedShape>,
    /// v18 item 7: set when the shape could not be lowered or cached (a build or lowering
    /// failure, or a shape outside the position-0 invariant): every later instance goes
    /// straight to the uncached lowering — one lowering per instance, today's cost — instead
    /// of retrying the shape first. The first instance pays the failed attempt plus its own.
    /// Set on every `None`-after-attempt exit of `bound_plan`, under whichever schema (not
    /// keyed by schema). A clone copies the cell, like `shape`: a template retired stays
    /// retired in its clones.
    uncacheable: OnceLock<()>,
    /// v18 item 7: the branch -> schema hash map of `schema_context`, built once per template
    /// and handed to every instance's compile (today's compile rebuilt it per instance).
    branch_schema_map: OnceLock<HashMap<String, SchemaHash>>,
    /// v18 item 7: how many times the branch map was built (a gate reads it: once).
    branch_map_builds: Arc<std::sync::atomic::AtomicU64>,
}

impl SubgraphTemplate {
    /// Create a new subgraph template.
    ///
    /// # Arguments
    /// * `base_query` - The inner query definition
    /// * `inner_column` - Column in the inner table to match against outer value
    /// * `select_columns` - Columns to include in results (empty = all)
    /// * `output_descriptor` - Descriptor for result rows
    pub fn new(
        base_query: Query,
        inner_column: String,
        select_columns: Vec<String>,
        output_descriptor: RowDescriptor,
        schema_context: Arc<SchemaContext>,
        session: Option<Session>,
        row_policy_mode: RowPolicyMode,
    ) -> Self {
        Self {
            base_query,
            inner_column,
            select_columns,
            output_descriptor,
            schema_context,
            session,
            row_policy_mode,
            shape_lowerings: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            shape: OnceLock::new(),
            uncacheable: OnceLock::new(),
            branch_schema_map: OnceLock::new(),
            branch_map_builds: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    /// v18 item 7: lowerings of this template's shape so far (one per instance today).
    #[cfg(any(test, feature = "test"))]
    pub fn shape_lowerings_for_test(&self) -> u64 {
        self.shape_lowerings
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Create a subgraph instance with a bound correlation value.
    ///
    /// The shape — the rebuilt inner query lowered to an execution plan — is lowered once
    /// per template (`lower_shape`); each instance binds its value into a clone of that plan
    /// and compiles it.
    pub fn instantiate(
        &self,
        correlation_value: Value,
        schema: &Arc<Schema>,
    ) -> Option<SubgraphInstance> {
        // Settle-cost accounting: counted on entry, because the binding and compile
        // below run whether or not the compile ultimately succeeds.
        crate::query_manager::settle_cost::bump(
            &crate::query_manager::settle_cost::SUBQUERY_INSTANTIATIONS,
        );
        // Fail closed on a shape miss: today's per-instance lowering, counted as a lowering
        // so the stand shows a template that never caches.
        let plan = match self.bound_plan(&correlation_value, schema) {
            Some(plan) => plan,
            None => self.uncached_plan(&correlation_value, schema)?,
        };
        let branch_schema_map = self.branch_schema_map.get_or_init(|| {
            self.branch_map_builds
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            QueryGraph::branch_schema_map_for_shared_context(&self.schema_context)
        });
        let graph = QueryGraph::compile_plan_with_branch_map(
            &plan,
            schema,
            self.session.clone(),
            &self.schema_context,
            self.row_policy_mode,
            branch_schema_map,
        )
        .ok()?;

        Some(SubgraphInstance {
            graph,
            correlation_value,
            current_results: Vec::new(),
        })
    }

    /// The inner query for one correlation value, rebuilt through the builder exactly as
    /// every instance was built before item 7: correlation `Eq` first, then the base
    /// conditions, order, limit, offset, select and nested subqueries. The cache lowers
    /// this with `Value::Null`; the uncached test path lowers it with the real value.
    fn instance_query(&self, value: &Value) -> Result<Query, QueryBuildError> {
        // Build query with correlation filter
        let mut query_builder = QueryBuilder::new(self.base_query.table);
        if self.base_query.branches.is_empty() {
            query_builder = query_builder.branch("main");
        } else {
            query_builder = query_builder.branches_owned(self.base_query.branches.clone());
        }
        if self.base_query.include_deleted {
            query_builder = query_builder.include_deleted();
        }

        // Add joins from base query
        for join_spec in &self.base_query.joins {
            query_builder = query_builder.join(join_spec.table);
            if let Some(ref alias) = join_spec.alias {
                query_builder = query_builder.alias(alias);
            }
            if let Some((ref left, ref right)) = join_spec.on {
                query_builder = query_builder.on(left, right);
            }
        }

        // Add correlation filter: inner_column = correlation_value
        query_builder = query_builder.filter_eq(&self.inner_column, value.clone());

        // Apply original filters from base query
        for disjunct in &self.base_query.disjuncts {
            for condition in &disjunct.conditions {
                query_builder = match condition {
                    crate::query_manager::query::Condition::Eq { column, value } => {
                        query_builder.filter_eq(column, value.clone())
                    }
                    crate::query_manager::query::Condition::Ne { column, value } => {
                        query_builder.filter_ne(column, value.clone())
                    }
                    crate::query_manager::query::Condition::Lt { column, value } => {
                        query_builder.filter_lt(column, value.clone())
                    }
                    crate::query_manager::query::Condition::Le { column, value } => {
                        query_builder.filter_le(column, value.clone())
                    }
                    crate::query_manager::query::Condition::Gt { column, value } => {
                        query_builder.filter_gt(column, value.clone())
                    }
                    crate::query_manager::query::Condition::Ge { column, value } => {
                        query_builder.filter_ge(column, value.clone())
                    }
                    crate::query_manager::query::Condition::Between { column, min, max } => {
                        query_builder.filter_between(column, min.clone(), max.clone())
                    }
                    crate::query_manager::query::Condition::Contains { column, value } => {
                        query_builder.filter_contains(column, value.clone())
                    }
                    crate::query_manager::query::Condition::IsNull { column } => {
                        query_builder.filter_is_null(column)
                    }
                    crate::query_manager::query::Condition::IsNotNull { column } => {
                        query_builder.filter_is_not_null(column)
                    }
                };
            }
        }

        // Apply order by
        for (col, dir) in &self.base_query.order_by {
            query_builder = match dir {
                crate::query_manager::graph_nodes::sort::SortDirection::Ascending => {
                    query_builder.order_by(col)
                }
                crate::query_manager::graph_nodes::sort::SortDirection::Descending => {
                    query_builder.order_by_desc(col)
                }
            };
        }

        // Apply limit/offset
        if let Some(limit) = self.base_query.limit {
            query_builder = query_builder.limit(limit);
        }
        if self.base_query.offset > 0 {
            query_builder = query_builder.offset(self.base_query.offset);
        }

        // Apply select columns
        if !self.select_columns.is_empty() {
            let cols: Vec<&str> = self.select_columns.iter().map(|s| s.as_str()).collect();
            query_builder = query_builder.select(&cols);
        }

        query_builder =
            query_builder.with_array_subqueries(self.base_query.array_subqueries.clone());
        if let Some(index) = self.base_query.result_element_index {
            query_builder = query_builder.result_element_index(index);
        }

        query_builder.try_build()
    }

    /// A shape miss: the cache is not used for this template, and every instance falls back
    /// to today's per-instance lowering (`uncacheable` remembers it). A `warn!` and, on the
    /// stand, `shape_lowerings` ≈ instances instead of ≈ templates. Not an assert: a miss is
    /// survivable, and one injected condition ahead of the correlation (a future implicit
    /// predicate) must not panic every debug build.
    fn shape_miss(what: &str) -> Option<CachedShape> {
        tracing::warn!(
            what,
            "subgraph shape miss: the include lowers uncached per instance"
        );
        None
    }

    /// Count one lowering of this template's shape (cached or uncached).
    fn note_lowering(&self) {
        self.shape_lowerings
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        crate::query_manager::settle_cost::bump(
            &crate::query_manager::settle_cost::SHAPE_LOWERINGS,
        );
    }

    /// Today's per-instance lowering, `lower(instance_query(value))` — the fallback of a
    /// shape miss. Counted as a lowering.
    fn uncached_plan(&self, value: &Value, schema: &Arc<Schema>) -> Option<ExecutionQueryPlan> {
        self.note_lowering();
        let query = self.instance_query(value).ok()?;
        QueryGraph::lower_query_with_schema_context_shared(&query, schema, &self.schema_context)
            .ok()
    }

    /// Lower the shape once: `lower(instance_query(Null))`. Counted on this template and in
    /// `settle_cost::SHAPE_LOWERINGS`. Fails closed: if the correlation is not the `Eq` at
    /// position 0 of every conjunction the shape is not cacheable — a `Null` left in the slot
    /// would compile as `IS NULL` — and `instantiate` lowers the instance uncached instead.
    fn lower_shape(&self, schema: &Arc<Schema>) -> Option<CachedShape> {
        self.note_lowering();
        let query = match self.instance_query(&Value::Null) {
            Ok(query) => query,
            Err(error) => {
                tracing::warn!(
                    table = self.table(),
                    column = %self.inner_column,
                    %error,
                    "subgraph shape miss: the include's instance query does not build"
                );
                return None;
            }
        };
        let plan = match QueryGraph::lower_query_with_schema_context_shared(
            &query,
            schema,
            &self.schema_context,
        ) {
            Ok(plan) => plan,
            Err(error) => {
                tracing::warn!(
                    table = self.table(),
                    column = %self.inner_column,
                    %error,
                    "subgraph shape miss: the include's shape does not lower under this schema"
                );
                return None;
            }
        };
        let Some(first_conjunction) = plan.disjuncts.first() else {
            return Self::shape_miss("the lowering has no conjunction");
        };
        let bound_column = match first_conjunction.conditions.first() {
            Some(Condition::Eq {
                column,
                value: Value::Null,
            }) => column.clone(),
            other => {
                return Self::shape_miss(&format!(
                    "the correlation must lower to position 0 of the first conjunction, found {other:?}"
                ));
            }
        };
        let every_conjunction_leads_with_it = plan.disjuncts.iter().all(|conjunction| {
            matches!(
                conjunction.conditions.first(),
                Some(Condition::Eq { column, value: Value::Null }) if *column == bound_column
            )
        });
        if !every_conjunction_leads_with_it {
            return Self::shape_miss(&format!(
                "every conjunction must lead with the correlation column {bound_column}"
            ));
        }
        Some(CachedShape {
            schema: Arc::clone(schema),
            plan,
            bound_column,
        })
    }

    /// The cached shape bound to one correlation value: the value replaces the `Null` at
    /// position 0 of every conjunction — by position and column, never by searching for a
    /// `Null` (base conditions may carry `Null` values of their own).
    fn bound_plan(&self, value: &Value, schema: &Arc<Schema>) -> Option<ExecutionQueryPlan> {
        if self.uncacheable.get().is_some() {
            return None;
        }
        let fresh;
        let shape = match self.shape.get() {
            Some(shape) if Arc::ptr_eq(&shape.schema, schema) => shape,
            // Another schema than the cached one: lower afresh for this instance and keep
            // the cache (the node's own schema is the one that recurs). A failed lowering
            // here marks the template too: `uncacheable` is not keyed by schema (a miss
            // under any schema retires the cache for good; this arm is unreachable in
            // production — the node's schema is the one that recurs).
            Some(_) => {
                let Some(lowered) = self.lower_shape(schema) else {
                    let _ = self.uncacheable.set(());
                    return None;
                };
                fresh = lowered;
                &fresh
            }
            // Not cached yet: lower, and cache only a success; a miss is remembered so the
            // next instance does not pay the attempt again.
            None => {
                let Some(lowered) = self.lower_shape(schema) else {
                    let _ = self.uncacheable.set(());
                    return None;
                };
                let _ = self.shape.set(lowered);
                match self.shape.get() {
                    Some(shape) if Arc::ptr_eq(&shape.schema, schema) => shape,
                    // Unreachable by construction (`&mut` reaches `instantiate`, no clone
                    // site between the set and this read); marked all the same so every
                    // `None`-after-attempt exit retires the cache (diff r4 SF1).
                    _ => {
                        let _ = self.uncacheable.set(());
                        return None;
                    }
                }
            }
        };
        let mut plan = shape.plan.clone();
        for conjunction in &mut plan.disjuncts {
            match conjunction.conditions.first_mut() {
                Some(Condition::Eq {
                    column,
                    value: slot,
                }) if *column == shape.bound_column && *slot == Value::Null => {
                    *slot = value.clone();
                }
                other => {
                    Self::shape_miss(&format!(
                        "the cached shape lost its correlation slot: {other:?}"
                    ));
                    let _ = self.uncacheable.set(());
                    return None;
                }
            }
        }
        Some(plan)
    }

    /// v18 item 7, test only: today's per-instance lowering, `lower(instance_query(value))`,
    /// for the plan-equality gates. The same rebuild the cache uses, so it duplicates no
    /// production lowering.
    #[cfg(any(test, feature = "test"))]
    pub(crate) fn instantiate_plan_uncached_for_test(
        &self,
        value: &Value,
        schema: &Arc<Schema>,
    ) -> Option<ExecutionQueryPlan> {
        let query = self.instance_query(value).ok()?;
        QueryGraph::lower_query_with_schema_context_shared(&query, schema, &self.schema_context)
            .ok()
    }

    /// v18 item 7, test only: how many times this template built its branch map.
    #[cfg(any(test, feature = "test"))]
    pub fn branch_map_builds_for_test(&self) -> u64 {
        self.branch_map_builds
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// v18 item 7, test only: the cached shape bound to `value`, the plan `instantiate`
    /// compiles.
    #[cfg(any(test, feature = "test"))]
    pub(crate) fn bind_for_test(
        &self,
        value: &Value,
        schema: &Arc<Schema>,
    ) -> Option<ExecutionQueryPlan> {
        self.bound_plan(value, schema)
    }

    /// Get the inner table name.
    pub fn table(&self) -> &str {
        &self.base_query.table.0
    }

    /// Get the inner correlation column.
    pub fn inner_column(&self) -> &str {
        &self.inner_column
    }

    /// OFFSET the instantiated inner query carries.
    ///
    /// Include specs cannot set one (`ArraySubquerySpec` has no offset field),
    /// and correlation routing depends on that: with an offset, deleting a row
    /// BEFORE an instance's window shifts the window even though the instance
    /// never held the row, so "route a vanished row only to the instances that
    /// held it" would under-route. `DirtRouting` reads this and disables
    /// itself rather than trusting the invariant.
    pub fn inner_offset(&self) -> usize {
        self.base_query.offset
    }

    /// Nested include specs carried inside this template's inner query.
    ///
    /// Their tables are registered against the OUTER node (`graph/compile.rs`,
    /// the `nested_stack` walk), so the outer node needs their correlation
    /// shape to route a grandchild change to the child instance holding it.
    pub fn nested_specs(&self) -> &[crate::query_manager::query::ArraySubquerySpec] {
        &self.base_query.array_subqueries
    }

    /// Get the output descriptor for result rows.
    pub fn output_descriptor(&self) -> &RowDescriptor {
        &self.output_descriptor
    }

    /// Identity of everything that can change what an instantiation computes,
    /// *given the invariant below*. Folded into `RecursiveRelationNode`'s
    /// `SettlementEvalCache` site key.
    ///
    /// `schema_context` is deliberately NOT hashed. It carries the schema hash,
    /// the live older schemas and the lenses between them, and it does change
    /// which index column a step scan reads (a lens can rename `email` to
    /// `email_address` without moving the output descriptor — see
    /// `subgraph_template_inherits_parent_schema_and_branch_context`). Omitting
    /// it is safe only because of the scoping of the one consumer:
    ///
    /// 1. `SettlementEvalCache` is never stored. Its three construction sites
    ///    (`server_queries.rs`, `manager.rs`) are stack locals consumed by a
    ///    single `authorized_*_from_graph*` call, so a cache never outlives one
    ///    authorization pass over one subscription graph within one tick. A
    ///    schema republish cannot reach an entry written before it.
    /// 2. That pass resolves `auth_schema`/`auth_context` once and holds them
    ///    fixed for every row it authorizes.
    /// 3. The cache is attached to exactly one evaluator
    ///    (`server_queries::evaluate_authorization_policy`) and only under
    ///    `Operation::Select`; the nested evaluators in `policy_filter.rs` and
    ///    `magic_columns.rs` run uncached.
    /// 4. The only compile path that can build a cache-consulting
    ///    `RecursiveRelationNode` is `PolicyGraph::for_exists_rel`, which takes
    ///    no `SchemaContext` argument at all — it synthesizes
    ///    `SchemaContext::with_defaults(compile_schema, "main")`: zero lenses,
    ///    zero live schemas, fixed env and user branch. Lens-driven divergence
    ///    is unrepresentable there. Because `structural_scans` is
    ///    `operation == Select`, `compile_schema` is always the same
    ///    policy-stripped copy of the single `auth_schema`.
    ///
    /// Recursive relations compiled on the subscription path *do* inherit a
    /// lens-carrying parent context, but those nodes settle with no cache and
    /// never call this.
    ///
    /// The one axis that genuinely varies inside a cache lifetime is the
    /// branch: a pass authorizes rows across branches, and the branch list is
    /// baked into `base_query.branches` at compile time, so it is discriminated
    /// here. `semantic_fingerprint_discriminates_branch` pins that.
    ///
    /// Give this fingerprint a consumer with a wider scope — a cache that
    /// survives a tick, a republish, or two subscriptions — and the omission
    /// stops being safe: add the schema context discriminator then.
    pub(crate) fn semantic_fingerprint(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        serde_json::to_vec(&self.base_query)
            .unwrap_or_else(|_| format!("{:?}", self.base_query).into_bytes())
            .hash(&mut hasher);
        self.inner_column.hash(&mut hasher);
        self.select_columns.hash(&mut hasher);
        self.output_descriptor.content_hash().hash(&mut hasher);
        serde_json::to_vec(&self.session)
            .unwrap_or_else(|_| format!("{:?}", self.session).into_bytes())
            .hash(&mut hasher);
        format!("{:?}", self.row_policy_mode).hash(&mut hasher);
        hasher.finish()
    }
}

/// A live instance of a subgraph for one outer row.
///
/// Contains the compiled query graph with bound parameters and tracks
/// the current array result.
#[derive(Debug)]
pub struct SubgraphInstance {
    /// The instantiated query graph with bound correlation value.
    pub graph: QueryGraph,
    /// The correlation value this instance is bound to.
    pub correlation_value: Value,
    /// Current array result (values from settling the graph).
    pub current_results: Vec<Value>,
}

impl SubgraphInstance {
    /// Get the current results as an array Value.
    pub fn as_array(&self) -> Value {
        Value::Array(self.current_results.clone())
    }
}

/// Builder for creating SubgraphTemplates.
#[derive(Debug)]
pub struct SubgraphBuilder {
    table: String,
    inner_column: String,
    select_columns: Vec<String>,
    filters: Vec<(String, Value)>, // Simple equality filters for now
    order_by: Vec<(String, bool)>, // (column, is_descending)
    limit: Option<usize>,
    session: Option<Session>,
    row_policy_mode: RowPolicyMode,
}

impl SubgraphBuilder {
    /// Create a new subgraph builder for the given table.
    pub fn new(table: impl Into<String>) -> Self {
        Self {
            table: table.into(),
            inner_column: String::new(),
            select_columns: Vec::new(),
            filters: Vec::new(),
            order_by: Vec::new(),
            limit: None,
            session: None,
            row_policy_mode: RowPolicyMode::PermissiveLocal,
        }
    }

    /// Set the correlation column (inner table column to match against outer value).
    pub fn correlate(mut self, inner_column: impl Into<String>) -> Self {
        self.inner_column = inner_column.into();
        self
    }

    /// Select specific columns.
    pub fn select(mut self, columns: &[&str]) -> Self {
        self.select_columns = columns.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Add an equality filter.
    pub fn filter_eq(mut self, column: impl Into<String>, value: Value) -> Self {
        self.filters.push((column.into(), value));
        self
    }

    /// Add ascending order by.
    pub fn order_by(mut self, column: impl Into<String>) -> Self {
        self.order_by.push((column.into(), false));
        self
    }

    /// Add descending order by.
    pub fn order_by_desc(mut self, column: impl Into<String>) -> Self {
        self.order_by.push((column.into(), true));
        self
    }

    /// Set a limit on results.
    pub fn limit(mut self, n: usize) -> Self {
        self.limit = Some(n);
        self
    }

    /// Set the session used when compiling subgraph instances.
    pub fn with_session(mut self, session: Session) -> Self {
        self.session = Some(session);
        self
    }

    /// Set the row policy mode used when compiling subgraph instances.
    pub fn with_row_policy_mode(mut self, row_policy_mode: RowPolicyMode) -> Self {
        self.row_policy_mode = row_policy_mode;
        self
    }

    /// Build the SubgraphTemplate.
    pub fn build(self, schema: &Schema) -> Option<SubgraphTemplate> {
        let table_name = crate::query_manager::types::TableName::new(&self.table);
        let table_schema = schema.get(&table_name)?;
        let descriptor = table_schema.columns.clone();

        // Build base query
        let mut query_builder = QueryBuilder::new(&self.table);

        for (col, value) in &self.filters {
            query_builder = query_builder.filter_eq(col, value.clone());
        }

        for (col, is_desc) in &self.order_by {
            query_builder = if *is_desc {
                query_builder.order_by_desc(col)
            } else {
                query_builder.order_by(col)
            };
        }

        if let Some(limit) = self.limit {
            query_builder = query_builder.limit(limit);
        }

        if !self.select_columns.is_empty() {
            let cols: Vec<&str> = self.select_columns.iter().map(|s| s.as_str()).collect();
            query_builder = query_builder.select(&cols);
        }

        let base_query = query_builder.build();

        // Build output descriptor (selected columns or all columns)
        let output_descriptor = if self.select_columns.is_empty() {
            descriptor
        } else {
            let columns = self
                .select_columns
                .iter()
                .filter_map(|name| descriptor.columns.iter().find(|c| &c.name == name).cloned())
                .collect::<Vec<_>>();
            RowDescriptor::new(columns)
        };

        Some(SubgraphTemplate::new(
            base_query,
            self.inner_column,
            self.select_columns,
            output_descriptor,
            Arc::new(SchemaContext::with_defaults(schema.clone(), "main")),
            self.session,
            self.row_policy_mode,
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::query_manager::encoding::decode_row;
    use crate::query_manager::graph::GraphNode;
    use crate::query_manager::manager::QueryManager;
    use crate::query_manager::policy::PolicyExpr;
    use crate::query_manager::query::QueryBuilder;
    use crate::query_manager::types::{ColumnDescriptor, ColumnType, TableName, TablePolicies};
    use crate::query_manager::types::{ComposedBranchName, SchemaBuilder, SchemaHash, TableSchema};
    use crate::schema_manager::{Lens, LensOp, LensTransform};
    use crate::sync_manager::SyncManager;
    use crate::test_support::seeded_memory_storage;

    fn test_schema() -> Schema {
        let mut schema = HashMap::new();
        schema.insert(
            TableName::new("posts"),
            RowDescriptor::new(vec![
                ColumnDescriptor::new("id", ColumnType::Integer),
                ColumnDescriptor::new("title", ColumnType::Text),
                ColumnDescriptor::new("author_id", ColumnType::Integer),
            ])
            .into(),
        );
        schema.insert(
            TableName::new("users"),
            RowDescriptor::new(vec![
                ColumnDescriptor::new("id", ColumnType::Integer),
                ColumnDescriptor::new("name", ColumnType::Text),
            ])
            .into(),
        );
        schema
    }

    #[test]
    fn subgraph_builder_creates_template() {
        let schema = test_schema();

        let template = SubgraphBuilder::new("posts")
            .correlate("author_id")
            .select(&["id", "title"])
            .order_by_desc("id")
            .limit(10)
            .build(&schema);

        assert!(template.is_some());
        let template = template.unwrap();
        assert_eq!(template.table(), "posts");
        assert_eq!(template.inner_column(), "author_id");
        assert_eq!(template.output_descriptor().columns.len(), 2);
    }

    #[test]
    fn subgraph_template_instantiates() {
        let schema = test_schema();

        let template = SubgraphBuilder::new("posts")
            .correlate("author_id")
            .build(&schema)
            .unwrap();

        let instance = template.instantiate(Value::Integer(42), &Arc::new(schema.clone()));
        assert!(instance.is_some());

        let instance = instance.unwrap();
        assert_eq!(instance.correlation_value, Value::Integer(42));
    }

    #[test]
    fn subgraph_builder_accepts_session_and_row_policy_mode() {
        let schema = test_schema();

        let template = SubgraphBuilder::new("posts")
            .correlate("author_id")
            .with_session(Session::new("alice"))
            .with_row_policy_mode(RowPolicyMode::Enforcing)
            .build(&schema)
            .unwrap();

        assert_eq!(
            template
                .session
                .as_ref()
                .map(|session| session.user_id.as_str()),
            Some("alice")
        );
        assert_eq!(template.row_policy_mode, RowPolicyMode::Enforcing);
    }

    #[test]
    fn array_subgraph_inherits_parent_session_for_permission_magic_columns() {
        let schema = SchemaBuilder::new()
            .table(
                TableSchema::builder("users")
                    .column("user_id", ColumnType::Text)
                    .column("team_id", ColumnType::Text)
                    .policies(TablePolicies::new().with_select(PolicyExpr::True)),
            )
            .table(
                TableSchema::builder("documents")
                    .column("owner_id", ColumnType::Text)
                    .column("team_id", ColumnType::Text)
                    .column("title", ColumnType::Text)
                    .policies(
                        TablePolicies::new()
                            .with_select(PolicyExpr::True)
                            .with_update(
                                Some(PolicyExpr::eq_session("owner_id", vec!["user_id".into()])),
                                PolicyExpr::True,
                            ),
                    ),
            )
            .build();
        let mut qm = QueryManager::new(SyncManager::new());
        qm.set_current_schema(schema, "dev", "main");
        let mut storage = seeded_memory_storage(&qm.schema_context().current_schema);

        qm.insert(
            &mut storage,
            "users",
            &[Value::Text("alice".into()), Value::Text("eng".into())],
        )
        .expect("insert user");
        qm.insert(
            &mut storage,
            "documents",
            &[
                Value::Text("alice".into()),
                Value::Text("eng".into()),
                Value::Text("Alice Draft".into()),
            ],
        )
        .expect("insert alice document");
        qm.insert(
            &mut storage,
            "documents",
            &[
                Value::Text("bob".into()),
                Value::Text("eng".into()),
                Value::Text("Bob Plan".into()),
            ],
        )
        .expect("insert bob document");

        let query = qm
            .query("users")
            .with_array("documents", |sub| {
                sub.from("documents")
                    .correlate("team_id", "users.team_id")
                    .select(&["title", "$canEdit"])
            })
            .build();
        let sub_id = qm
            .subscribe_with_session(query, Some(Session::new("alice")), None)
            .expect("subscribe with session");

        qm.process(&mut storage);

        let update = qm
            .take_updates()
            .into_iter()
            .find(|update| update.subscription_id == sub_id)
            .expect("subscription should produce initial update");
        let values = decode_row(&update.descriptor, &update.delta.added[0].data)
            .expect("decode subscription row");
        let documents = values[2].as_array().expect("documents should be an array");
        let permissions_by_title: HashMap<String, bool> = documents
            .iter()
            .map(|document| {
                let values = document.as_row().expect("document should be a row");
                let title = match &values[0] {
                    Value::Text(title) => title.clone(),
                    other => panic!("expected document title text, got {other:?}"),
                };
                let can_edit = match &values[1] {
                    Value::Boolean(can_edit) => *can_edit,
                    other => panic!("expected $canEdit boolean, got {other:?}"),
                };
                (title, can_edit)
            })
            .collect();

        assert_eq!(permissions_by_title.get("Alice Draft"), Some(&true));
        assert_eq!(permissions_by_title.get("Bob Plan"), Some(&false));
    }

    #[test]
    fn subgraph_instance_as_array() {
        let schema = test_schema();

        let template = SubgraphBuilder::new("posts")
            .correlate("author_id")
            .build(&schema)
            .unwrap();

        let mut instance = template
            .instantiate(Value::Integer(1), &Arc::new(schema.clone()))
            .unwrap();
        instance.current_results = vec![Value::Integer(10), Value::Integer(20)];

        let array = instance.as_array();
        assert_eq!(
            array,
            Value::Array(vec![Value::Integer(10), Value::Integer(20)])
        );
    }

    #[test]
    fn subgraph_template_inherits_parent_schema_and_branch_context() {
        let v1 = SchemaBuilder::new()
            .table(
                TableSchema::builder("users")
                    .column("id", ColumnType::Integer)
                    .column("email", ColumnType::Text),
            )
            .build();
        let v2 = SchemaBuilder::new()
            .table(
                TableSchema::builder("users")
                    .column("id", ColumnType::Integer)
                    .column("email_address", ColumnType::Text),
            )
            .build();

        let v1_hash = SchemaHash::compute(&v1);
        let v2_hash = SchemaHash::compute(&v2);

        let mut transform = LensTransform::new();
        transform.push(
            LensOp::RenameColumn {
                table: "users".to_string(),
                old_name: "email".to_string(),
                new_name: "email_address".to_string(),
            },
            false,
        );
        let lens = Lens::new(v1_hash, v2_hash, transform);

        let mut schema_context = SchemaContext::new(v2.clone(), "dev", "main");
        schema_context.add_live_schema(v1.clone(), lens);

        let v1_branch = ComposedBranchName::new("dev", v1_hash, "main")
            .to_branch_name()
            .as_str()
            .to_string();

        let base_query = QueryBuilder::new("users")
            .branches(&[v1_branch.as_str()])
            .build();
        let output_descriptor = v2.get(&TableName::new("users")).unwrap().columns.clone();
        let template = SubgraphTemplate::new(
            base_query,
            "email_address".to_string(),
            Vec::new(),
            output_descriptor,
            Arc::new(schema_context),
            None,
            RowPolicyMode::PermissiveLocal,
        );

        let instance = template
            .instantiate(
                Value::Text("alice@example.com".to_string()),
                &Arc::new(v2.clone()),
            )
            .expect("subgraph should compile using inherited schema context");
        assert_eq!(instance.graph.index_scan_nodes.len(), 1);

        let (scan_id, _table, scan_column) = &instance.graph.index_scan_nodes[0];
        assert_eq!(
            scan_column.as_str(),
            "email",
            "index lookup should translate new column name to old-branch index column"
        );

        let scan_branch = instance
            .graph
            .nodes
            .get(scan_id.0 as usize)
            .and_then(|ctx| match &ctx.node {
                GraphNode::IndexScan(scan) => Some(scan.branch.as_str()),
                _ => None,
            })
            .expect("index scan node must exist");
        assert_eq!(
            scan_branch, v1_branch,
            "subgraph should keep the parent branch list when instantiating"
        );
    }

    /// `semantic_fingerprint` omits the schema context on purpose (see the
    /// invariant documented there); the branch is the one axis that genuinely
    /// varies within a `SettlementEvalCache` lifetime, because one
    /// authorization pass authorizes rows across branches. If the branch ever
    /// stopped reaching the fingerprint, a memoized recursive-relation result
    /// from one branch would be served for another.
    #[test]
    fn semantic_fingerprint_discriminates_branch() {
        let schema = test_schema();
        let output_descriptor = schema
            .get(&TableName::new("posts"))
            .unwrap()
            .columns
            .clone();

        let template_for_branch = |branch: &str| {
            SubgraphTemplate::new(
                QueryBuilder::new("posts").branches(&[branch]).build(),
                "author_id".to_string(),
                Vec::new(),
                output_descriptor.clone(),
                Arc::new(SchemaContext::with_defaults(schema.clone(), "main")),
                None,
                RowPolicyMode::PermissiveLocal,
            )
        };

        let main = template_for_branch("main");
        let preview = template_for_branch("preview");

        assert_eq!(
            main.semantic_fingerprint(),
            template_for_branch("main").semantic_fingerprint(),
            "fingerprint must be stable for identical templates"
        );
        assert_ne!(
            main.semantic_fingerprint(),
            preview.semantic_fingerprint(),
            "templates differing only in branch must not share a cache entry"
        );
    }
}
