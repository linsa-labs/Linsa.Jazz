//! Cross-tick cache of row-level select-policy verdicts.
//!
//! Every write-tick re-authorizes the full result scope of each affected subscription,
//! and each check loads the row from storage and evaluates its policy. That made the
//! cost of a single write proportional to the size of every subscribed result set. This
//! cache keeps verdicts between ticks; correctness rests on invalidation:
//!
//! - a changed row drops its own verdicts (all sessions);
//! - a write to any table a policy *reads* (Exists / ExistsRel / Inherits /
//!   InheritsReferencing — extracted statically, transitively, from the policy AST)
//!   drops the verdicts of every row whose table depends on it;
//! - a different auth schema or policy mode drops everything;
//! - a policy whose dependencies cannot be fully analyzed marks its table's verdicts
//!   as dependent on *every* table — cached, but dropped on any write.
//!
//! In debug builds every cache hit is re-verified against a fresh evaluation by the
//! caller (see `provenance_row_matches_current_select_policy`), so the whole test
//! suite continuously exercises the equivalence.

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::object::{BranchName, ObjectId};
use crate::query_manager::policy::{Operation, PolicyExpr};
use crate::query_manager::relation_ir::RelExpr;
use crate::query_manager::session::Session;
use crate::query_manager::types::branch::SchemaHash;
use crate::query_manager::types::{RowPolicyMode, Schema, TableName};

/// Identity of the policy universe a verdict was computed under. A marker change
/// (schema migration, republished permissions, mode flip) clears the cache wholesale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct AuthzMarker {
    pub schema_hash: SchemaHash,
    /// Assignment counter of the authorization schema. Permissions can be republished
    /// without changing the data-schema hash, so the hash alone under-invalidates.
    pub auth_generation: u64,
    pub mode: RowPolicyMode,
    /// Fingerprint of the sanctioned branch universe read-path policy arms
    /// evaluate against (defect 24). Activating another generation of the
    /// family changes what a readReferencing/EXISTS arm can see, so verdicts
    /// computed under a different universe must not be served.
    pub branch_universe: u64,
}

/// Cheap owned identity of a session for keying verdicts. Sessions are value objects
/// (user id + claims); two sessions hashing equal are treated as the same principal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct AuthzSessionKey(u64);

impl AuthzSessionKey {
    pub fn for_session(session: Option<&Session>) -> Self {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        match session {
            None => 0u8.hash(&mut hasher),
            Some(session) => {
                1u8.hash(&mut hasher);
                session.user_id.hash(&mut hasher);
                format!("{:?}", session.auth_mode).hash(&mut hasher);
                // claims is a serde_json::Value (no Hash impl); its canonical string is
                // stable for identical claims.
                session.claims.to_string().hash(&mut hasher);
            }
        }
        Self(hasher.finish())
    }
}

/// The tables a table's select policy reads, or `Unknown` when the analysis cannot
/// prove completeness — `Unknown` verdicts drop on any write.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum TableDeps {
    Known(HashSet<TableName>),
    Unknown,
}

impl TableDeps {
    fn hit_by(&self, changed_tables: &HashSet<&str>) -> bool {
        match self {
            TableDeps::Unknown => true,
            TableDeps::Known(deps) => deps.iter().any(|dep| changed_tables.contains(dep.as_str())),
        }
    }

    /// Whether a write to `table` can change a verdict that depends on these reads.
    pub(super) fn reads(&self, table: &str) -> bool {
        match self {
            TableDeps::Unknown => true,
            TableDeps::Known(deps) => deps.iter().any(|dep| dep.as_str() == table),
        }
    }
}

/// Extract every table `table`'s select policy reads, transitively through Inherits
/// chains. Conservative: any construct whose reads cannot be enumerated yields
/// `Unknown`.
pub(super) fn select_policy_dependency_tables(
    auth_schema: &Schema,
    table: &TableName,
) -> TableDeps {
    let mut deps = HashSet::new();
    let mut visited = HashSet::new();
    if collect_operation_policy_deps(
        auth_schema,
        table,
        Operation::Select,
        &mut deps,
        &mut visited,
    ) {
        TableDeps::Known(deps)
    } else {
        TableDeps::Unknown
    }
}

fn collect_operation_policy_deps(
    auth_schema: &Schema,
    table: &TableName,
    operation: Operation,
    deps: &mut HashSet<TableName>,
    visited: &mut HashSet<(TableName, Operation)>,
) -> bool {
    if !visited.insert((*table, operation)) {
        return true;
    }
    let Some(table_schema) = auth_schema.get(table) else {
        // Policy for an unknown table: nothing to read.
        return true;
    };
    let policy = match operation {
        Operation::Select => table_schema.policies.select.using.as_ref(),
        Operation::Insert => table_schema.policies.insert.using.as_ref(),
        Operation::Update => table_schema.policies.update.using.as_ref(),
        Operation::Delete => table_schema.policies.delete.using.as_ref(),
    };
    match policy {
        None => true,
        Some(policy) => collect_policy_expr_deps(auth_schema, table, policy, deps, visited),
    }
}

fn collect_policy_expr_deps(
    auth_schema: &Schema,
    table: &TableName,
    policy: &PolicyExpr,
    deps: &mut HashSet<TableName>,
    visited: &mut HashSet<(TableName, Operation)>,
) -> bool {
    match policy {
        PolicyExpr::Cmp { .. }
        | PolicyExpr::SessionCmp { .. }
        | PolicyExpr::IsNull { .. }
        | PolicyExpr::SessionIsNull { .. }
        | PolicyExpr::IsNotNull { .. }
        | PolicyExpr::SessionIsNotNull { .. }
        | PolicyExpr::Contains { .. }
        | PolicyExpr::SessionContains { .. }
        | PolicyExpr::In { .. }
        | PolicyExpr::InList { .. }
        | PolicyExpr::SessionInList { .. }
        | PolicyExpr::True
        | PolicyExpr::False => true,
        PolicyExpr::Exists {
            table: exists_table,
            condition,
        } => {
            let exists_table = TableName::new(exists_table);
            deps.insert(exists_table);
            collect_policy_expr_deps(auth_schema, &exists_table, condition, deps, visited)
        }
        PolicyExpr::ExistsRel { rel } => collect_rel_expr_deps(rel, deps),
        PolicyExpr::Inherits {
            operation,
            via_column,
            ..
        } => {
            let Some(target): Option<TableName> = auth_schema
                .get(table)
                .and_then(|schema| schema.columns.column(via_column))
                .and_then(|column| column.references)
            else {
                // FK target unknown — cannot bound what the policy reads.
                return false;
            };
            deps.insert(target);
            collect_operation_policy_deps(auth_schema, &target, *operation, deps, visited)
        }
        PolicyExpr::InheritsReferencing {
            operation,
            source_table,
            ..
        } => {
            let source = TableName::new(source_table);
            deps.insert(source);
            collect_operation_policy_deps(auth_schema, &source, *operation, deps, visited)
        }
        PolicyExpr::And(items) | PolicyExpr::Or(items) => items
            .iter()
            .all(|item| collect_policy_expr_deps(auth_schema, table, item, deps, visited)),
        PolicyExpr::Not(inner) => {
            collect_policy_expr_deps(auth_schema, table, inner, deps, visited)
        }
    }
}

fn collect_rel_expr_deps(rel: &RelExpr, deps: &mut HashSet<TableName>) -> bool {
    match rel {
        RelExpr::TableScan { table } => {
            deps.insert(*table);
            true
        }
        RelExpr::Filter { input, .. }
        | RelExpr::Project { input, .. }
        | RelExpr::Distinct { input, .. }
        | RelExpr::OrderBy { input, .. }
        | RelExpr::Offset { input, .. }
        | RelExpr::Limit { input, .. } => collect_rel_expr_deps(input, deps),
        RelExpr::Union { inputs } => inputs
            .iter()
            .all(|input| collect_rel_expr_deps(input, deps)),
        RelExpr::Join { left, right, .. } => {
            collect_rel_expr_deps(left, deps) && collect_rel_expr_deps(right, deps)
        }
        RelExpr::Gather { seed, step, .. } => {
            collect_rel_expr_deps(seed, deps) && collect_rel_expr_deps(step, deps)
        }
    }
}

#[derive(Debug)]
struct RowVerdicts {
    deps: Arc<TableDeps>,
    by_session: HashMap<AuthzSessionKey, bool>,
}

/// Verdict entries above this count clear the cache wholesale on the next store —
/// simple, safe, and refilled by normal traffic.
const MAX_CACHED_ROWS: usize = 32_768;

#[derive(Debug, Default)]
pub(super) struct AuthzVerdictCache {
    marker: Option<AuthzMarker>,
    deps_by_table: HashMap<TableName, Arc<TableDeps>>,
    rows: HashMap<(ObjectId, BranchName), RowVerdicts>,
    /// Served-from-cache count. Cheap enough to keep unconditionally; tests use it to
    /// prove the cache actually served hits rather than passing vacuously.
    hits: u64,
}

// OFF by default, opt-in via JAZZ_AUTHZ_CACHE_ENABLE. Field evidence (linsa-v5): with
// the cache on, subscription row sets flapped (full → empty → full on every settle
// wave), which kept clients re-rendering and re-syncing until iOS killed the app at its
// per-process memory limit. The embedded mobile runtime cannot set environment
// variables, so the safe state must be the default. Re-enable only after the
// verdict-flap bug is found and covered by a regression test.
//
// 0 = uninitialised, 1 = enabled, 2 = disabled. A relaxed atomic (not OnceLock) so
// tests can force a deterministic state regardless of execution order.
static CACHE_STATE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

fn cache_enabled() -> bool {
    use std::sync::atomic::Ordering;
    match CACHE_STATE.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => {
            let enabled = std::env::var_os("JAZZ_AUTHZ_CACHE_ENABLE").is_some();
            CACHE_STATE.store(if enabled { 1 } else { 2 }, Ordering::Relaxed);
            enabled
        }
    }
}

/// Test hook: force the cache on/off for this process, bypassing the env lookup.
#[cfg(any(test, feature = "test"))]
pub fn set_cache_enabled_for_tests(enabled: bool) {
    CACHE_STATE.store(
        if enabled { 1 } else { 2 },
        std::sync::atomic::Ordering::Relaxed,
    );
}

impl AuthzVerdictCache {
    fn ensure_marker(&mut self, marker: AuthzMarker) {
        if self.marker != Some(marker) {
            self.rows.clear();
            self.deps_by_table.clear();
            self.marker = Some(marker);
        }
    }

    pub fn get(
        &mut self,
        marker: AuthzMarker,
        object_id: ObjectId,
        branch: BranchName,
        session: AuthzSessionKey,
    ) -> Option<bool> {
        self.ensure_marker(marker);
        if !cache_enabled() {
            return None;
        }
        let verdict = self
            .rows
            .get(&(object_id, branch))
            .and_then(|row| row.by_session.get(&session))
            .copied();
        if verdict.is_some() {
            self.hits += 1;
        }
        verdict
    }

    /// How many verdicts were served from the cache since construction.
    pub fn hit_count(&self) -> u64 {
        self.hits
    }

    #[allow(clippy::too_many_arguments)]
    pub fn store(
        &mut self,
        marker: AuthzMarker,
        object_id: ObjectId,
        branch: BranchName,
        table: TableName,
        session: AuthzSessionKey,
        verdict: bool,
        auth_schema: &Schema,
    ) {
        // Off means off: `get` serves nothing then, so rows stored here would only pile up to
        // `MAX_CACHED_ROWS` and make every `invalidate` scan them.
        if !cache_enabled() {
            return;
        }
        self.ensure_marker(marker);
        if self.rows.len() >= MAX_CACHED_ROWS {
            self.rows.clear();
        }
        let deps = self
            .deps_by_table
            .entry(table)
            .or_insert_with(|| Arc::new(select_policy_dependency_tables(auth_schema, &table)))
            .clone();
        self.rows
            .entry((object_id, branch))
            .or_insert_with(|| RowVerdicts {
                deps,
                by_session: HashMap::new(),
            })
            .by_session
            .insert(session, verdict);
    }

    /// Apply one tick's invalidation: drop verdicts of every changed row, and of every
    /// row whose table's policy reads any of the changed tables.
    pub fn invalidate<'a>(
        &mut self,
        changed_tables: impl IntoIterator<Item = &'a str>,
        changed_rows: impl IntoIterator<Item = ObjectId>,
    ) {
        if self.rows.is_empty() {
            return;
        }
        let changed_tables: HashSet<&str> = changed_tables.into_iter().collect();
        let changed_rows: HashSet<ObjectId> = changed_rows.into_iter().collect();
        if changed_tables.is_empty() && changed_rows.is_empty() {
            return;
        }
        self.rows.retain(|(object_id, _), row| {
            !changed_rows.contains(object_id) && !row.deps.hit_by(&changed_tables)
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query_manager::types::{ColumnType, PolicyExpr as Expr, TablePolicies};
    use crate::query_manager::types::{SchemaBuilder, TableSchema};

    fn deps_of(schema: &Schema, table: &str) -> TableDeps {
        select_policy_dependency_tables(schema, &TableName::new(table))
    }

    fn known(tables: &[&str]) -> TableDeps {
        TableDeps::Known(tables.iter().map(|t| TableName::new(*t)).collect())
    }

    #[test]
    fn no_policy_reads_no_tables() {
        let schema = SchemaBuilder::new()
            .table(TableSchema::builder("docs").column("title", ColumnType::Text))
            .build();
        assert_eq!(deps_of(&schema, "docs"), known(&[]));
    }

    #[test]
    fn session_only_policy_reads_no_tables() {
        let schema = SchemaBuilder::new()
            .table(
                TableSchema::builder("docs")
                    .column("owner_id", ColumnType::Text)
                    .policies(
                        TablePolicies::new()
                            .with_select(Expr::eq_session("owner_id", vec!["user_id".into()])),
                    ),
            )
            .build();
        assert_eq!(deps_of(&schema, "docs"), known(&[]));
    }

    #[test]
    fn exists_policy_reads_its_table() {
        let schema = SchemaBuilder::new()
            .table(
                TableSchema::builder("docs").policies(TablePolicies::new().with_select(
                    Expr::Exists {
                        table: "memberships".into(),
                        condition: Box::new(Expr::True),
                    },
                )),
            )
            .build();
        assert_eq!(deps_of(&schema, "docs"), known(&["memberships"]));
    }

    #[test]
    fn inherits_follows_the_fk_and_the_target_policy_transitively() {
        let schema = SchemaBuilder::new()
            .table(
                TableSchema::builder("messages")
                    .fk_column("chat_id", "chats")
                    .policies(TablePolicies::new().with_select(Expr::Inherits {
                        operation: Operation::Select,
                        via_column: "chat_id".into(),
                        max_depth: None,
                    })),
            )
            .table(
                TableSchema::builder("chats").policies(TablePolicies::new().with_select(
                    Expr::Exists {
                        table: "memberships".into(),
                        condition: Box::new(Expr::True),
                    },
                )),
            )
            .table(TableSchema::builder("memberships").column("user_id", ColumnType::Text))
            .build();
        assert_eq!(
            deps_of(&schema, "messages"),
            known(&["chats", "memberships"])
        );
    }

    #[test]
    fn inherits_without_fk_metadata_is_unknown() {
        let schema = SchemaBuilder::new()
            .table(
                TableSchema::builder("messages")
                    .column("chat_id", ColumnType::Uuid)
                    .policies(TablePolicies::new().with_select(Expr::Inherits {
                        operation: Operation::Select,
                        via_column: "chat_id".into(),
                        max_depth: None,
                    })),
            )
            .build();
        assert_eq!(deps_of(&schema, "messages"), TableDeps::Unknown);
    }

    #[test]
    fn recursive_inherits_terminates() {
        let schema = SchemaBuilder::new()
            .table(
                TableSchema::builder("folders")
                    .fk_column("parent_id", "folders")
                    .policies(TablePolicies::new().with_select(Expr::Inherits {
                        operation: Operation::Select,
                        via_column: "parent_id".into(),
                        max_depth: None,
                    })),
            )
            .build();
        assert_eq!(deps_of(&schema, "folders"), known(&["folders"]));
    }

    #[test]
    fn verdicts_drop_on_row_change_and_dep_table_change() {
        let schema = SchemaBuilder::new()
            .table(
                TableSchema::builder("docs").policies(TablePolicies::new().with_select(
                    Expr::Exists {
                        table: "memberships".into(),
                        condition: Box::new(Expr::True),
                    },
                )),
            )
            .build();
        let marker = AuthzMarker {
            schema_hash: SchemaHash::compute(&schema),
            auth_generation: 7,
            mode: RowPolicyMode::Enforcing,
            branch_universe: 0,
        };
        let session = AuthzSessionKey::for_session(None);
        let branch = BranchName::new("main");
        let row_a = ObjectId::new();
        let row_b = ObjectId::new();

        set_cache_enabled_for_tests(true);
        let mut cache = AuthzVerdictCache::default();
        cache.store(
            marker,
            row_a,
            branch,
            TableName::new("docs"),
            session,
            true,
            &schema,
        );
        cache.store(
            marker,
            row_b,
            branch,
            TableName::new("docs"),
            session,
            false,
            &schema,
        );
        assert_eq!(cache.get(marker, row_a, branch, session), Some(true));
        assert_eq!(cache.get(marker, row_b, branch, session), Some(false));

        // Unrelated table: nothing drops.
        cache.invalidate(["weather"], []);
        assert_eq!(cache.get(marker, row_a, branch, session), Some(true));

        // A row change drops only that row.
        cache.invalidate([], [row_a]);
        assert_eq!(cache.get(marker, row_a, branch, session), None);
        assert_eq!(cache.get(marker, row_b, branch, session), Some(false));

        // A dep-table change drops every docs verdict.
        cache.invalidate(["memberships"], []);
        assert_eq!(cache.get(marker, row_b, branch, session), None);
    }

    #[test]
    fn marker_change_clears_everything() {
        let schema = SchemaBuilder::new()
            .table(TableSchema::builder("docs").column("t", ColumnType::Text))
            .build();
        let marker = AuthzMarker {
            schema_hash: SchemaHash::compute(&schema),
            auth_generation: 7,
            mode: RowPolicyMode::Enforcing,
            branch_universe: 0,
        };
        let session = AuthzSessionKey::for_session(None);
        let branch = BranchName::new("main");
        let row = ObjectId::new();

        set_cache_enabled_for_tests(true);
        let mut cache = AuthzVerdictCache::default();
        cache.store(
            marker,
            row,
            branch,
            TableName::new("docs"),
            session,
            true,
            &schema,
        );
        let other_marker = AuthzMarker {
            schema_hash: marker.schema_hash,
            auth_generation: marker.auth_generation,
            mode: RowPolicyMode::PermissiveLocal,
            branch_universe: marker.branch_universe,
        };
        assert_eq!(cache.get(other_marker, row, branch, session), None);
    }
}
