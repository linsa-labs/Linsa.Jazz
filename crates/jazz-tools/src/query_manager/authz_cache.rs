//! Row-level select-policy verdicts that are not computed again while they cannot have
//! changed.
//!
//! Every write-tick re-authorizes the full result scope of each affected subscription,
//! and each check loads the row from storage and evaluates its policy. That made the
//! cost of a single write proportional to the size of every subscribed result set.
//!
//! Three kinds of verdict are held, each for as long as what it depends on stands still:
//!
//! - a verdict reached in a settle pass, for that pass: storage is only read under one;
//! - an ALLOWED verdict of a table whose select policy is the constant `True`, for any
//!   session, until the row changes: such a policy reads neither the session nor
//!   another table, so only the row going away can turn it;
//! - every verdict, across ticks, when the opt-in switch is on (off by default).
//!
//! Correctness rests on invalidation:
//!
//! - a changed row drops its own verdicts (all sessions);
//! - a write to any table a policy *reads* (Exists / ExistsRel / Inherits /
//!   InheritsReferencing — extracted statically, transitively, from the policy AST)
//!   drops the verdicts of every row whose table depends on it;
//! - a different auth schema, policy mode or set of authorization contexts drops
//!   everything;
//! - a policy whose dependencies cannot be fully analyzed marks its table's verdicts
//!   as dependent on *every* table — cached, but dropped on any write;
//! - any change at all drops what the open pass reached.
//!
//! In debug builds every cache hit is re-verified against a fresh evaluation by the
//! caller (see `provenance_row_matches_current_select_policy`), so the whole test
//! suite continuously exercises the equivalence.

use std::collections::{HashMap, HashSet};

use ahash::AHashMap;
use smallvec::SmallVec;

use crate::object::{BranchName, ObjectId};
use crate::query_manager::policy::{Operation, PolicyExpr};
use crate::query_manager::relation_ir::RelExpr;
use crate::query_manager::session::Session;
use crate::query_manager::types::branch::SchemaHash;
use crate::query_manager::types::{RowPolicyMode, Schema, TableName};

/// Identity of the policy universe a verdict was computed under. A marker change
/// (schema migration, republished permissions, mode flip, a lens or a schema generation
/// arriving) clears the cache wholesale. Everything in it is the same for every
/// subscription of a runtime, so subscriptions never clear each other's verdicts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct AuthzMarker {
    pub schema_hash: SchemaHash,
    /// Assignment counter of the authorization schema. Permissions can be republished
    /// without changing the data-schema hash, so the hash alone under-invalidates.
    pub auth_generation: u64,
    pub mode: RowPolicyMode,
    /// How many times the authorization contexts were thrown away to be rebuilt. A
    /// context is what a row is transformed through before its policy reads it, and a
    /// lens or a schema generation can arrive without the authorization schema moving.
    pub context_epoch: u64,
}

/// The key a session's verdicts are stored under, issued by
/// [`AuthzVerdictCache::session_key`]. Never issued twice, and two sessions share one
/// only when they are equal in full.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct AuthzSessionKey(u64);

impl AuthzSessionKey {
    /// A read without a session.
    pub const NO_SESSION: Self = Self(0);
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SessionIdentity {
    user_id: String,
    auth_mode: u8,
    /// `claims` is a `serde_json::Value` (no `Hash`); its string is the same for the
    /// same claims and different for different ones.
    claims: String,
}

/// The key of what a walk reads rows through, issued by [`AuthzVerdictCache::scope`].
/// Subscriptions of clients on different schema generations, or on different user
/// branches, read the same row through different scopes (defect 24: activating another
/// generation of the family changes what a readReferencing/EXISTS arm can see).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct AuthzScope(u64);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ScopeIdentity {
    policy_branches: Vec<String>,
    branch_schemas: Vec<(String, SchemaHash)>,
}

/// The tables a table's select policy reads, or `Unknown` when the analysis cannot
/// prove completeness — `Unknown` verdicts drop on any write.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum TableDeps {
    Known(HashSet<TableName>),
    Unknown,
}

impl TableDeps {
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

/// The verdicts kept for one row, whatever branch it was read on. Almost always one:
/// the constant-policy verdict of the branch the row lives on.
#[derive(Debug)]
struct KeptRow {
    table: TableName,
    verdicts: SmallVec<[(VerdictKey, bool); 1]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VerdictKey {
    branch: BranchName,
    scope: AuthzScope,
    reader: Reader,
}

impl VerdictKey {
    /// Whether a verdict stored under this key answers a read by `session`.
    fn answers(&self, branch: BranchName, scope: AuthzScope, session: AuthzSessionKey) -> bool {
        self.branch == branch
            && self.scope == scope
            && match self.reader {
                // Nobody is not anyone: without a session a row is denied whatever its
                // policy.
                Reader::Anyone => session != AuthzSessionKey::NO_SESSION,
                Reader::One(reader) => reader == session,
            }
    }
}

/// Whom a verdict was reached for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Reader {
    /// Whoever asks with a session: a constant policy allows a row without reading who
    /// reads it.
    Anyone,
    One(AuthzSessionKey),
}

/// What a verdict of the open settle pass is stored under.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PassKey {
    object_id: ObjectId,
    branch: BranchName,
    scope: AuthzScope,
    session: AuthzSessionKey,
}

impl PassKey {
    fn of(
        object_id: ObjectId,
        branch: BranchName,
        scope: AuthzScope,
        session: AuthzSessionKey,
    ) -> Self {
        Self {
            object_id,
            branch,
            scope,
            session,
        }
    }
}

/// How many verdicts are held across ticks at most. They live in two generations of half
/// this size: a verdict that is served moves to the young one, and when the young one is
/// full the old one is dropped — what was not asked for since the last turn. Nothing is
/// ever cleared wholesale for size, so a store at the limit costs the verdicts nobody
/// reads, not a re-evaluation of every row every subscription holds.
///
/// What is read again and again is kept only while it fits ONE generation: of more
/// verdicts than that, all asked for in every pass, each turn drops the ones not yet
/// asked for again, and those are evaluated as they were before verdicts were kept.
const MAX_KEPT_VERDICTS: usize = 131_072;

/// How many verdicts one settle pass remembers at most. Past it the pass evaluates what
/// it asks for, as it did before there was anything to remember.
const MAX_PASS_VERDICTS: usize = 262_144;

/// A pass that remembered more than this gives its table back when it ends, instead of
/// holding the room of its largest pass for as long as the runtime lives.
const PASS_ROOM_HELD: usize = 16_384;

/// Distinct sessions and scopes remembered for keying. Past these the table is dropped:
/// keys are never issued twice, so what was stored under a dropped key is just not found
/// again and leaves with its generation.
const MAX_KNOWN_SESSIONS: usize = 4_096;
const MAX_KNOWN_SCOPES: usize = 64;

#[derive(Debug, Default)]
pub(super) struct AuthzVerdictCache {
    marker: Option<AuthzMarker>,
    /// What the select policy of each table with a kept verdict reads.
    deps_by_table: HashMap<TableName, TableDeps>,
    /// Verdicts kept or served since the generations last turned.
    young: AHashMap<ObjectId, KeptRow>,
    /// Verdicts of the generation before. A verdict is in one of the two, never in both;
    /// a row is in both when only some of its verdicts were asked for since the turn.
    old: AHashMap<ObjectId, KeptRow>,
    young_verdicts: usize,
    /// A settle pass is open: what it reaches and cannot keep is remembered until it
    /// closes.
    in_pass: bool,
    /// What the open pass reached. Apart from the kept verdicts: a pass over many
    /// sessions' rows must not push out what the next tick will ask for again.
    of_the_pass: AHashMap<PassKey, bool>,
    sessions: HashMap<SessionIdentity, AuthzSessionKey>,
    sessions_issued: u64,
    scopes: HashMap<ScopeIdentity, AuthzScope>,
    scopes_issued: u64,
    /// Test hook: this cache's own answer to `cache_enabled`, whatever the process says.
    #[cfg(test)]
    pub(super) keeps_every_verdict: Option<bool>,
    /// Test hook: `MAX_KEPT_VERDICTS` for this cache.
    #[cfg(test)]
    pub(super) max_kept_verdicts: Option<usize>,
    /// Test hook: `MAX_PASS_VERDICTS` for this cache.
    #[cfg(test)]
    pub(super) max_pass_verdicts: Option<usize>,
    /// Test hook: how many passes began while the one before was still open — whatever
    /// ran between the two ran inside a pass that was over.
    #[cfg(test)]
    pub(super) passes_begun_over_an_open_one: u64,
    /// Test hook: how many times a table write walked everything held.
    #[cfg(test)]
    pub(super) table_walks: u64,
    /// Test hook: nothing is served and nothing is stored — every verdict is evaluated,
    /// as if there were no cache. What a comparison against the cache is made with.
    #[cfg(any(test, feature = "test"))]
    pub(super) bypassed: bool,
    /// Served-from-cache count. Cheap enough to keep unconditionally; tests use it to
    /// prove the cache actually served hits rather than passing vacuously.
    hits: u64,
    /// Verdicts asked for and not held: each is a row load and a policy evaluation.
    misses: u64,
}

// OFF by default, opt-in via JAZZ_AUTHZ_CACHE_ENABLE. Field evidence (linsa-v5): with
// the cache on, subscription row sets flapped (full → empty → full on every settle
// wave), which kept clients re-rendering and re-syncing until iOS killed the app at its
// per-process memory limit. The embedded mobile runtime cannot set environment
// variables, so the safe state must be the default. Re-enable only after the
// verdict-flap bug is found and covered by a regression test.
//
// The switch decides whether EVERY verdict is kept across ticks. Two kinds are held
// whatever it says, because nothing they depend on can move while they are held: the
// verdicts of an open settle pass, and the allowed verdicts of a constant policy.
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
    /// Whether every verdict is kept across ticks (the switch that is off by default).
    fn keeps_every_verdict(&self) -> bool {
        #[cfg(test)]
        if let Some(keeps) = self.keeps_every_verdict {
            return keeps;
        }
        cache_enabled()
    }

    fn generation_size(&self) -> usize {
        #[cfg(test)]
        if let Some(max) = self.max_kept_verdicts {
            return (max / 2).max(1);
        }
        MAX_KEPT_VERDICTS / 2
    }

    fn pass_size(&self) -> usize {
        #[cfg(test)]
        if let Some(max) = self.max_pass_verdicts {
            return max;
        }
        MAX_PASS_VERDICTS
    }

    fn bypassed(&self) -> bool {
        #[cfg(any(test, feature = "test"))]
        {
            self.bypassed
        }
        #[cfg(not(any(test, feature = "test")))]
        {
            false
        }
    }

    fn ensure_marker(&mut self, marker: AuthzMarker) {
        if self.marker != Some(marker) {
            self.young.clear();
            self.old.clear();
            self.young_verdicts = 0;
            self.of_the_pass.clear();
            self.deps_by_table.clear();
            // The keys of sessions and scopes stay: they name values, not anything the
            // marker describes, and a walk asks for its keys before its first verdict.
            self.marker = Some(marker);
        }
    }

    /// The key verdicts of `session` are stored under. Two sessions get the same key
    /// only when they are the same principal with the same claims, compared in full:
    /// call it once per walk, not once per row.
    pub fn session_key(&mut self, session: Option<&Session>) -> AuthzSessionKey {
        let Some(session) = session else {
            return AuthzSessionKey::NO_SESSION;
        };
        let identity = SessionIdentity {
            user_id: session.user_id.clone(),
            auth_mode: session.auth_mode as u8,
            claims: session.claims.to_string(),
        };
        if let Some(key) = self.sessions.get(&identity) {
            return *key;
        }
        if self.sessions.len() >= MAX_KNOWN_SESSIONS {
            self.sessions.clear();
        }
        self.sessions_issued += 1;
        let key = AuthzSessionKey(self.sessions_issued);
        self.sessions.insert(identity, key);
        key
    }

    /// The key of what a walk reads rows through: the branches its policies may look at
    /// and the schema each branch's rows were written under. Compared in full, as
    /// sessions are, and whatever order the caller lists them in; once per walk.
    pub fn scope(
        &mut self,
        policy_branches: &[String],
        branch_schemas: &HashMap<String, SchemaHash>,
    ) -> AuthzScope {
        let mut policy_branches = policy_branches.to_vec();
        policy_branches.sort_unstable();
        let mut branch_schemas: Vec<(String, SchemaHash)> = branch_schemas
            .iter()
            .map(|(branch, hash)| (branch.clone(), *hash))
            .collect();
        branch_schemas.sort_by(|left, right| left.0.cmp(&right.0));
        let identity = ScopeIdentity {
            policy_branches,
            branch_schemas,
        };
        if let Some(scope) = self.scopes.get(&identity) {
            return *scope;
        }
        if self.scopes.len() >= MAX_KNOWN_SCOPES {
            self.scopes.clear();
        }
        self.scopes_issued += 1;
        let scope = AuthzScope(self.scopes_issued);
        self.scopes.insert(identity, scope);
        scope
    }

    /// The generations turn: what the old one held and nobody asked for is dropped.
    fn turn_generations(&mut self) {
        self.old = std::mem::take(&mut self.young);
        self.young_verdicts = 0;
    }

    /// Put a verdict in the young generation.
    fn keep(&mut self, object_id: ObjectId, table: TableName, key: VerdictKey, allowed: bool) {
        if let Some(kept) = self
            .young
            .get_mut(&object_id)
            .and_then(|row| row.verdicts.iter_mut().find(|(kept, _)| *kept == key))
        {
            kept.1 = allowed;
            return;
        }
        if self.young_verdicts >= self.generation_size() {
            self.turn_generations();
        }
        self.young
            .entry(object_id)
            .or_insert_with(|| KeptRow {
                table,
                verdicts: SmallVec::new(),
            })
            .verdicts
            .push((key, allowed));
        self.young_verdicts += 1;
    }

    /// The verdict kept across ticks that answers this read. One found in the old
    /// generation moves to the young one — it alone, not what else its row holds: a
    /// verdict stored under a key nobody asks with any more ages out however often the
    /// row is read.
    fn kept(
        &mut self,
        object_id: ObjectId,
        branch: BranchName,
        scope: AuthzScope,
        session: AuthzSessionKey,
    ) -> Option<bool> {
        if let Some(row) = self.young.get(&object_id) {
            let served = row
                .verdicts
                .iter()
                .find(|(key, _)| key.answers(branch, scope, session));
            if let Some((_, allowed)) = served {
                return Some(*allowed);
            }
        }
        let row = self.old.get_mut(&object_id)?;
        let index = row
            .verdicts
            .iter()
            .position(|(key, _)| key.answers(branch, scope, session))?;
        let (key, allowed) = row.verdicts.swap_remove(index);
        let table = row.table;
        if row.verdicts.is_empty() {
            self.old.remove(&object_id);
        }
        self.keep(object_id, table, key, allowed);
        Some(allowed)
    }

    pub fn get(
        &mut self,
        marker: AuthzMarker,
        object_id: ObjectId,
        branch: BranchName,
        scope: AuthzScope,
        session: AuthzSessionKey,
    ) -> Option<bool> {
        if self.bypassed() {
            self.misses += 1;
            return None;
        }
        self.ensure_marker(marker);
        let verdict = match self.kept(object_id, branch, scope, session) {
            Some(allowed) => Some(allowed),
            None if self.of_the_pass.is_empty() => None,
            None => self
                .of_the_pass
                .get(&PassKey::of(object_id, branch, scope, session))
                .copied(),
        };
        if verdict.is_some() {
            self.hits += 1;
        } else {
            self.misses += 1;
        }
        verdict
    }

    /// How many verdicts were asked for and had to be evaluated since construction.
    pub fn miss_count(&self) -> u64 {
        self.misses
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
        scope: AuthzScope,
        session: AuthzSessionKey,
        verdict: bool,
        constant: bool,
        auth_schema: &Schema,
    ) {
        if self.bypassed() {
            return;
        }
        // An allowed verdict of a constant policy depends on nothing but the row being
        // there — not on who reads it either — so it holds for every session until the
        // row changes. Any other verdict is kept across ticks only when the cache is
        // switched on, and otherwise remembered for the open pass.
        let reader = if constant && verdict && session != AuthzSessionKey::NO_SESSION {
            Reader::Anyone
        } else if self.keeps_every_verdict() {
            Reader::One(session)
        } else {
            if self.in_pass {
                self.ensure_marker(marker);
                if self.of_the_pass.len() < self.pass_size() {
                    self.of_the_pass
                        .insert(PassKey::of(object_id, branch, scope, session), verdict);
                }
            }
            return;
        };
        self.ensure_marker(marker);
        self.deps_by_table
            .entry(table)
            .or_insert_with(|| select_policy_dependency_tables(auth_schema, &table));
        let key = VerdictKey {
            branch,
            scope,
            reader,
        };
        self.keep(object_id, table, key, verdict);
    }

    /// A settle pass begins: until `end_pass`, storage is only read.
    pub fn begin_pass(&mut self) {
        #[cfg(test)]
        if self.in_pass {
            self.passes_begun_over_an_open_one += 1;
        }
        self.forget_the_pass();
        self.in_pass = true;
    }

    pub fn end_pass(&mut self) {
        self.forget_the_pass();
        self.in_pass = false;
    }

    #[cfg(test)]
    pub(super) fn pass_is_open(&self) -> bool {
        self.in_pass
    }

    /// How many verdicts are kept across ticks, in both generations.
    #[cfg(test)]
    pub(super) fn kept_verdict_count(&self) -> usize {
        self.young
            .values()
            .chain(self.old.values())
            .map(|row| row.verdicts.len())
            .sum()
    }

    /// How many verdicts the open pass remembers.
    #[cfg(test)]
    pub(super) fn pass_verdict_count(&self) -> usize {
        self.of_the_pass.len()
    }

    /// The bytes the tables hold for what is in them now, by their own sizes: what the
    /// allocator hands out on top is not in it.
    #[cfg(test)]
    pub(super) fn table_bytes(&self) -> usize {
        fn table<K, V>(map: &AHashMap<K, V>) -> usize {
            // A table keeps one control byte per bucket and fills seven buckets of eight.
            map.capacity() * 8 / 7 * (std::mem::size_of::<(K, V)>() + 1)
        }
        let spilled = |rows: &AHashMap<ObjectId, KeptRow>| -> usize {
            rows.values()
                .filter(|row| row.verdicts.spilled())
                .map(|row| row.verdicts.capacity() * std::mem::size_of::<(VerdictKey, bool)>())
                .sum()
        };
        table(&self.young)
            + table(&self.old)
            + table(&self.of_the_pass)
            + spilled(&self.young)
            + spilled(&self.old)
    }

    fn forget_the_pass(&mut self) {
        if self.of_the_pass.capacity() > PASS_ROOM_HELD {
            self.of_the_pass = AHashMap::new();
        } else if !self.of_the_pass.is_empty() {
            self.of_the_pass.clear();
        }
    }

    /// Apply one tick's invalidation: drop verdicts of every changed row, and of every
    /// row whose table's policy reads any of the changed tables.
    ///
    /// A changed row costs its own entry. A changed table costs a pass over what is kept
    /// only when the policy of a table with kept verdicts reads it.
    pub fn invalidate<'a>(
        &mut self,
        changed_tables: impl IntoIterator<Item = &'a str>,
        changed_rows: impl IntoIterator<Item = ObjectId>,
    ) {
        let mut changed = false;
        for object_id in changed_rows {
            changed = true;
            if let Some(row) = self.young.remove(&object_id) {
                self.young_verdicts = self.young_verdicts.saturating_sub(row.verdicts.len());
            }
            self.old.remove(&object_id);
        }
        let mut changed_tables = changed_tables.into_iter().peekable();
        if !changed && changed_tables.peek().is_none() {
            return;
        }
        // Something changed under an open pass: what it reached so far no longer holds.
        self.forget_the_pass();
        let changed_tables: SmallVec<[&str; 4]> = changed_tables.collect();
        let hit: SmallVec<[TableName; 4]> = self
            .deps_by_table
            .iter()
            .filter(|(_, deps)| changed_tables.iter().any(|table| deps.reads(table)))
            .map(|(table, _)| *table)
            .collect();
        if hit.is_empty() {
            return;
        }
        #[cfg(test)]
        {
            self.table_walks += 1;
        }
        self.forget_tables(&hit);
    }

    /// Drop every verdict held, of every table: something went over stored rows that
    /// says neither which rows it touched nor under which table name — the one a row was
    /// written under or the one the authorization schema gives it — they are kept here.
    pub fn forget_everything(&mut self) {
        self.forget_the_pass();
        self.young.clear();
        self.old.clear();
        self.young_verdicts = 0;
        self.deps_by_table.clear();
    }

    fn forget_tables(&mut self, tables: &[TableName]) {
        self.young.retain(|_, row| !tables.contains(&row.table));
        self.old.retain(|_, row| !tables.contains(&row.table));
        self.young_verdicts = self.young.values().map(|row| row.verdicts.len()).sum();
        // Nothing is kept for these tables now: a later write to what their policies
        // read has nothing of theirs to walk for.
        for table in tables {
            self.deps_by_table.remove(table);
        }
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

    fn marker_of(schema: &Schema) -> AuthzMarker {
        AuthzMarker {
            schema_hash: SchemaHash::compute(schema),
            auth_generation: 7,
            mode: RowPolicyMode::Enforcing,
            context_epoch: 0,
        }
    }

    /// A cache with the cross-tick switch forced, whatever the process says, and the
    /// scope and sessions the tests below read through.
    struct Fixture {
        cache: AuthzVerdictCache,
        schema: Schema,
        marker: AuthzMarker,
        scope: AuthzScope,
        alice: AuthzSessionKey,
        bob: AuthzSessionKey,
        branch: BranchName,
        docs: TableName,
    }

    fn fixture(schema: Schema, keeps_every_verdict: bool) -> Fixture {
        let mut cache = AuthzVerdictCache {
            keeps_every_verdict: Some(keeps_every_verdict),
            ..Default::default()
        };
        let marker = marker_of(&schema);
        let scope = cache.scope(&["main".to_string()], &HashMap::new());
        let alice = cache.session_key(Some(&Session::new("alice")));
        let bob = cache.session_key(Some(&Session::new("bob")));
        Fixture {
            cache,
            schema,
            marker,
            scope,
            alice,
            bob,
            branch: BranchName::new("main"),
            docs: TableName::new("docs"),
        }
    }

    impl Fixture {
        fn store(
            &mut self,
            row: ObjectId,
            session: AuthzSessionKey,
            verdict: bool,
            constant: bool,
        ) {
            self.cache.store(
                self.marker,
                row,
                self.branch,
                self.docs,
                self.scope,
                session,
                verdict,
                constant,
                &self.schema,
            );
        }

        fn get(&mut self, row: ObjectId, session: AuthzSessionKey) -> Option<bool> {
            self.cache
                .get(self.marker, row, self.branch, self.scope, session)
        }
    }

    fn docs_reading_memberships() -> Schema {
        SchemaBuilder::new()
            .table(
                TableSchema::builder("docs").policies(TablePolicies::new().with_select(
                    Expr::Exists {
                        table: "memberships".into(),
                        condition: Box::new(Expr::True),
                    },
                )),
            )
            .build()
    }

    fn plain_docs() -> Schema {
        SchemaBuilder::new()
            .table(TableSchema::builder("docs").column("t", ColumnType::Text))
            .build()
    }

    #[test]
    fn verdicts_drop_on_row_change_and_dep_table_change() {
        let mut f = fixture(docs_reading_memberships(), true);
        let (row_a, row_b) = (ObjectId::new(), ObjectId::new());
        let alice = f.alice;

        f.store(row_a, alice, true, false);
        f.store(row_b, alice, false, false);
        assert_eq!(f.get(row_a, alice), Some(true));
        assert_eq!(f.get(row_b, alice), Some(false));

        // Unrelated table: nothing drops.
        f.cache.invalidate(["weather"], []);
        assert_eq!(f.get(row_a, alice), Some(true));

        // A row change drops only that row.
        f.cache.invalidate([], [row_a]);
        assert_eq!(f.get(row_a, alice), None);
        assert_eq!(f.get(row_b, alice), Some(false));

        // A dep-table change drops every docs verdict.
        f.cache.invalidate(["memberships"], []);
        assert_eq!(f.get(row_b, alice), None);
    }

    #[test]
    fn marker_change_clears_everything() {
        let mut f = fixture(plain_docs(), true);
        let row = ObjectId::new();
        let alice = f.alice;
        f.store(row, alice, true, false);

        for other in [
            AuthzMarker {
                mode: RowPolicyMode::PermissiveLocal,
                ..f.marker
            },
            AuthzMarker {
                auth_generation: f.marker.auth_generation + 1,
                ..f.marker
            },
            // A lens or a schema generation arrived: rows are transformed through other
            // contexts from here on.
            AuthzMarker {
                context_epoch: f.marker.context_epoch + 1,
                ..f.marker
            },
        ] {
            let mut f = fixture(plain_docs(), true);
            let alice = f.alice;
            f.store(row, alice, true, false);
            assert_eq!(f.get(row, alice), Some(true), "fixture");
            assert_eq!(
                f.cache.get(other, row, f.branch, f.scope, alice),
                None,
                "{other:?}"
            );
        }
    }

    #[test]
    fn a_verdict_of_a_pass_is_served_in_it_and_forgotten_when_it_ends() {
        let mut f = fixture(plain_docs(), false);
        let (allowed, denied) = (ObjectId::new(), ObjectId::new());
        let alice = f.alice;

        // Outside a pass nothing is kept: storage may change before the next read.
        f.store(allowed, alice, true, false);
        assert_eq!(f.get(allowed, alice), None);

        f.cache.begin_pass();
        f.store(allowed, alice, true, false);
        f.store(denied, alice, false, false);
        assert_eq!(f.get(allowed, alice), Some(true));
        assert_eq!(f.get(denied, alice), Some(false));
        f.cache.end_pass();

        assert_eq!(
            f.get(allowed, alice),
            None,
            "a verdict of a pass was served after it"
        );
        assert_eq!(f.get(denied, alice), None);
        assert_eq!(f.cache.kept_verdict_count(), 0);
        assert_eq!(f.cache.pass_verdict_count(), 0);

        // The pass is over: what is reached now is not kept either.
        f.store(allowed, alice, true, false);
        assert_eq!(
            f.get(allowed, alice),
            None,
            "a verdict reached after the pass ended was kept as one of it"
        );

        // And nothing of it is there in the next pass.
        f.cache.begin_pass();
        assert_eq!(f.get(allowed, alice), None);
        f.cache.end_pass();
    }

    #[test]
    fn a_change_under_an_open_pass_drops_what_the_pass_reached() {
        let mut f = fixture(plain_docs(), false);
        let (one, other) = (ObjectId::new(), ObjectId::new());
        let alice = f.alice;

        f.cache.begin_pass();
        f.store(one, alice, true, false);
        f.store(other, alice, true, false);
        f.cache.invalidate(["weather"], []);
        assert_eq!(
            f.get(one, alice),
            None,
            "a verdict reached before a change was served after it"
        );
        assert_eq!(f.get(other, alice), None);
        f.cache.end_pass();
    }

    #[test]
    fn an_allowed_verdict_of_a_constant_policy_is_kept_until_its_row_changes() {
        let mut f = fixture(plain_docs(), false);
        let (allowed, denied, untouched) = (ObjectId::new(), ObjectId::new(), ObjectId::new());
        let (alice, bob) = (f.alice, f.bob);

        f.cache.begin_pass();
        f.store(allowed, alice, true, true);
        f.store(untouched, alice, true, true);
        f.store(denied, alice, false, true);
        f.cache.end_pass();

        assert_eq!(f.get(allowed, alice), Some(true));
        assert_eq!(
            f.get(denied, alice),
            None,
            "a denied verdict was kept: the row may become readable without a change of its own"
        );
        assert_eq!(
            f.get(allowed, bob),
            Some(true),
            "a constant policy does not read who asks: one check serves every session"
        );
        assert_eq!(
            f.get(allowed, AuthzSessionKey::NO_SESSION),
            None,
            "a read without a session is denied whatever the policy, and was served an allowed verdict"
        );

        // A write to another table, or to another row, leaves it.
        f.cache.invalidate(["docs", "weather"], [denied]);
        assert_eq!(f.get(allowed, alice), Some(true));

        // Its own change drops it, and only it.
        f.cache.invalidate([], [allowed]);
        assert_eq!(
            f.get(allowed, alice),
            None,
            "a row changed and its verdict was still served"
        );
        assert_eq!(f.get(untouched, alice), Some(true));

        // Another policy universe starts from nothing.
        let republished = AuthzMarker {
            auth_generation: f.marker.auth_generation + 1,
            ..f.marker
        };
        assert_eq!(
            f.cache
                .get(republished, untouched, f.branch, f.scope, alice),
            None
        );
    }

    /// The key is the session, compared in full: a hash of it would serve one
    /// principal's verdict to another whenever two of them collide.
    #[test]
    fn sessions_that_differ_in_anything_get_different_keys() {
        use crate::query_manager::session::AuthMode;

        let mut cache = AuthzVerdictCache::default();
        let alice = Session::new("alice");
        let sessions = [
            alice.clone(),
            Session::new("bob"),
            alice
                .clone()
                .with_claims(serde_json::json!({ "teams": ["eng"] })),
            alice
                .clone()
                .with_claims(serde_json::json!({ "teams": ["design"] })),
            alice.clone().with_auth_mode(AuthMode::Anonymous),
        ];
        let keys: Vec<_> = sessions
            .iter()
            .map(|session| cache.session_key(Some(session)))
            .collect();
        for (index, key) in keys.iter().enumerate() {
            assert_ne!(*key, AuthzSessionKey::NO_SESSION, "{:?}", sessions[index]);
            for (other_index, other) in keys.iter().enumerate().skip(index + 1) {
                assert_ne!(
                    key, other,
                    "{:?} and {:?} share a key",
                    sessions[index], sessions[other_index]
                );
            }
        }
        assert_eq!(
            cache.session_key(Some(&Session::new("alice"))),
            keys[0],
            "the same session asked again got another key: nothing kept for it is found"
        );
        assert_eq!(cache.session_key(None), AuthzSessionKey::NO_SESSION);
    }

    /// A verdict is kept for the walk that could have reached it: a row is loaded and
    /// transformed through the branches and the branch schemas of the subscription.
    #[test]
    fn a_verdict_is_not_served_to_a_walk_that_reads_through_other_branches() {
        let mut f = fixture(plain_docs(), false);
        let row = ObjectId::new();
        let alice = f.alice;
        f.store(row, alice, true, true);
        assert_eq!(f.get(row, alice), Some(true), "fixture");

        let hash = SchemaHash::compute(&f.schema);
        let branches = ["main".to_string()];
        let other_schemas = HashMap::from([("main".to_string(), hash)]);
        let other_universe = ["main".to_string(), "dev-0000-main".to_string()];
        for (what, scope) in [
            (
                "other branch schemas",
                f.cache.scope(&branches, &other_schemas),
            ),
            (
                "another policy universe",
                f.cache.scope(&other_universe, &HashMap::new()),
            ),
        ] {
            assert_ne!(scope, f.scope, "{what}");
            assert_eq!(
                f.cache.get(f.marker, row, f.branch, scope, alice),
                None,
                "{what}"
            );
        }
        assert_eq!(
            f.cache.scope(&branches, &HashMap::new()),
            f.scope,
            "the same scope asked again got another key"
        );
        assert_eq!(f.get(row, alice), Some(true));
    }

    /// Both tables key a verdict by the branch the row was read on and by the scope it
    /// was read through: what a pass reaches is what reads the session and other tables,
    /// where another scope is another answer.
    #[test]
    fn a_verdict_is_served_only_on_the_branch_and_through_the_scope_it_was_reached() {
        let mut f = fixture(plain_docs(), false);
        let (kept, of_the_pass) = (ObjectId::new(), ObjectId::new());
        let alice = f.alice;
        let other_scope = f.cache.scope(
            &["main".to_string(), "dev-0000-main".to_string()],
            &HashMap::new(),
        );
        let other_branch = BranchName::new("dev-0000-main");

        f.cache.begin_pass();
        f.store(kept, alice, true, true);
        f.store(of_the_pass, alice, true, false);
        for row in [kept, of_the_pass] {
            assert_eq!(f.get(row, alice), Some(true), "fixture");
            assert_eq!(
                f.cache.get(f.marker, row, f.branch, other_scope, alice),
                None,
                "a verdict was served to a walk that reads through another scope"
            );
            assert_eq!(
                f.cache.get(f.marker, row, other_branch, f.scope, alice),
                None,
                "a verdict was served for the row on another branch"
            );
        }
        f.cache.end_pass();
    }

    /// A patch that says neither which rows it touched nor under which name they are kept
    /// costs everything held: both generations, the open pass, every table.
    #[test]
    fn forgetting_everything_leaves_no_verdict_of_any_table_or_of_the_pass() {
        let mut f = fixture(plain_docs(), false);
        f.cache.max_kept_verdicts = Some(4);
        let alice = f.alice;
        let docs: Vec<ObjectId> = (0..3).map(|_| ObjectId::new()).collect();
        let (other, of_the_pass) = (ObjectId::new(), ObjectId::new());
        // Two fill a generation and the third turns it: the table is in both.
        for row in &docs {
            f.store(*row, alice, true, true);
        }
        f.cache.store(
            f.marker,
            other,
            f.branch,
            TableName::new("plain"),
            f.scope,
            alice,
            true,
            true,
            &f.schema,
        );
        f.cache.begin_pass();
        f.store(of_the_pass, alice, true, false);
        assert!(!f.cache.deps_by_table.is_empty(), "fixture");

        f.cache.forget_everything();

        for row in &docs {
            assert_eq!(f.get(*row, alice), None, "kept in one of the generations");
        }
        assert_eq!(
            f.get(of_the_pass, alice),
            None,
            "reached by the open pass before the rows were gone over"
        );
        assert_eq!(f.get(other, alice), None, "kept under another table's name");
        assert_eq!(f.cache.kept_verdict_count(), 0);
        assert_eq!(
            f.cache.young_verdicts, 0,
            "the count that turns the generations still holds what was dropped"
        );
        assert!(
            f.cache.deps_by_table.is_empty(),
            "a table with nothing kept is still walked for"
        );
        f.store(docs[0], alice, true, true);
        assert_eq!(
            f.get(docs[0], alice),
            Some(true),
            "nothing is kept afterwards"
        );
        f.cache.end_pass();
    }

    /// At the limit the cache drops what nobody asked for, not everything: a wholesale
    /// clear would make the next pass check every row of every subscription again.
    #[test]
    fn a_full_cache_drops_the_verdicts_nobody_asked_for_and_keeps_the_ones_it_served() {
        let mut f = fixture(plain_docs(), false);
        f.cache.max_kept_verdicts = Some(8);
        let alice = f.alice;
        let rows: Vec<ObjectId> = (0..12).map(|_| ObjectId::new()).collect();

        // Four fill a generation; the fifth turns it.
        for row in &rows[..5] {
            f.store(*row, alice, true, true);
        }
        // The first is asked for; the three after it are not.
        assert_eq!(f.get(rows[0], alice), Some(true));
        // Another generation fills and turns.
        for row in &rows[5..9] {
            f.store(*row, alice, true, true);
        }

        assert_eq!(
            f.get(rows[0], alice),
            Some(true),
            "a verdict that was being served was dropped for size"
        );
        for row in &rows[1..4] {
            assert_eq!(
                f.get(*row, alice),
                None,
                "a verdict nobody asked for outlived two generations"
            );
        }

        // However many rows pass through, no more than the limit is held.
        for round in 0..5 {
            for row in &rows {
                f.store(*row, alice, true, true);
                assert!(
                    f.cache.kept_verdict_count() <= 8 + 1,
                    "round {round}: {} verdicts held under a limit of 8",
                    f.cache.kept_verdict_count()
                );
            }
        }
    }

    /// A row that changes is dropped from wherever its verdicts are, the generation
    /// nobody has asked in since the last turn included.
    #[test]
    fn a_changed_row_is_dropped_from_the_generation_that_holds_it() {
        let mut f = fixture(plain_docs(), false);
        f.cache.max_kept_verdicts = Some(4);
        let alice = f.alice;
        let rows: Vec<ObjectId> = (0..3).map(|_| ObjectId::new()).collect();
        for row in &rows {
            f.store(*row, alice, true, true);
        }
        // Two filled a generation and the third turned it: the first two are in the old
        // one, the third in the young one.
        f.cache.invalidate([], [rows[0], rows[2]]);

        assert_eq!(f.get(rows[0], alice), None, "kept in the old generation");
        assert_eq!(f.get(rows[2], alice), None, "kept in the young generation");
        assert_eq!(f.get(rows[1], alice), Some(true));
    }

    /// What a pass reaches is remembered apart from what is kept across ticks: a pass
    /// over many sessions' rows would otherwise turn the generations and push out the
    /// verdicts the next tick asks for again.
    #[test]
    fn a_pass_does_not_push_out_what_is_kept_across_ticks() {
        let mut f = fixture(plain_docs(), false);
        f.cache.max_kept_verdicts = Some(8);
        let (alice, bob) = (f.alice, f.bob);
        let kept: Vec<ObjectId> = (0..3).map(|_| ObjectId::new()).collect();
        let of_a_pass: Vec<ObjectId> = (0..20).map(|_| ObjectId::new()).collect();
        for row in &kept {
            f.store(*row, alice, true, true);
        }

        // Each pass reaches more verdicts than both generations hold together.
        for _ in 0..2 {
            f.cache.begin_pass();
            for row in &of_a_pass {
                f.store(*row, alice, true, false);
                f.store(*row, bob, false, false);
            }
            assert_eq!(f.cache.pass_verdict_count(), 2 * of_a_pass.len());
            assert_eq!(f.get(of_a_pass[0], alice), Some(true));
            assert_eq!(f.get(of_a_pass[0], bob), Some(false));
            f.cache.end_pass();
            assert_eq!(f.cache.pass_verdict_count(), 0);
        }

        assert_eq!(
            f.cache.kept_verdict_count(),
            kept.len(),
            "what a pass reached was put among the verdicts kept across ticks"
        );
        for row in &kept {
            assert_eq!(
                f.get(*row, bob),
                Some(true),
                "a pass pushed out a verdict kept across ticks"
            );
        }
        for row in &of_a_pass {
            assert_eq!(f.get(*row, alice), None);
        }
    }

    /// A pass remembers up to its limit and evaluates the rest, as every read was
    /// evaluated before there was a pass to remember in.
    #[test]
    fn a_pass_remembers_no_more_than_its_limit() {
        let mut f = fixture(plain_docs(), false);
        f.cache.max_pass_verdicts = Some(4);
        let alice = f.alice;
        let rows: Vec<ObjectId> = (0..10).map(|_| ObjectId::new()).collect();

        f.cache.begin_pass();
        for row in &rows {
            f.store(*row, alice, true, false);
        }
        assert_eq!(f.cache.pass_verdict_count(), 4);
        for row in &rows[..4] {
            assert_eq!(f.get(*row, alice), Some(true));
        }
        for row in &rows[4..] {
            assert_eq!(f.get(*row, alice), None);
        }
        f.cache.end_pass();
    }

    /// A pass that remembered a great many verdicts does not hold their room for as long
    /// as the runtime lives.
    #[test]
    fn a_large_pass_gives_its_room_back() {
        let mut f = fixture(plain_docs(), false);
        let alice = f.alice;
        f.cache.begin_pass();
        for _ in 0..2 * PASS_ROOM_HELD {
            f.store(ObjectId::new(), alice, true, false);
        }
        assert_eq!(f.cache.pass_verdict_count(), 2 * PASS_ROOM_HELD);
        f.cache.end_pass();
        assert_eq!(
            f.cache.table_bytes(),
            0,
            "the tables of an empty cache still hold the room of its largest pass"
        );
    }

    /// Being served is what keeps a verdict, one verdict at a time: a row that is read
    /// all the time does not carry along what was stored for someone who stopped asking.
    #[test]
    fn a_verdict_nobody_asks_for_ages_out_of_a_row_that_is_read() {
        let mut f = fixture(plain_docs(), true);
        f.cache.max_kept_verdicts = Some(8);
        let (alice, bob) = (f.alice, f.bob);
        let read = ObjectId::new();
        let others: Vec<ObjectId> = (0..6).map(|_| ObjectId::new()).collect();

        // Four fill a generation; the fifth turns it.
        f.store(read, alice, true, false);
        f.store(read, bob, false, false);
        for row in &others[..3] {
            f.store(*row, alice, true, false);
        }
        // Alice goes on reading the row; bob does not.
        assert_eq!(f.get(read, alice), Some(true));
        // Another generation fills and turns.
        for row in &others[3..] {
            f.store(*row, alice, true, false);
        }

        assert_eq!(
            f.get(read, alice),
            Some(true),
            "a verdict that was being served was dropped for size"
        );
        assert_eq!(
            f.get(read, bob),
            None,
            "a verdict nobody asked for outlived two generations on a row that was read"
        );
    }

    /// The same branches listed in another order are the same scope: the list comes out
    /// of a hash map, and every order of it would otherwise take a key of its own.
    #[test]
    fn branches_listed_in_another_order_are_the_same_scope() {
        let mut cache = AuthzVerdictCache::default();
        let listed = ["main".to_string(), "dev-0000-main".to_string()];
        let reversed = ["dev-0000-main".to_string(), "main".to_string()];
        assert_eq!(
            cache.scope(&listed, &HashMap::new()),
            cache.scope(&reversed, &HashMap::new())
        );
    }

    /// The room the tables take with as many verdicts kept as there can be: one
    /// constant-policy verdict for each of 131 072 rows.
    #[test]
    fn a_full_cache_takes_a_bounded_room() {
        let mut f = fixture(plain_docs(), false);
        let alice = f.alice;
        for _ in 0..MAX_KEPT_VERDICTS + MAX_KEPT_VERDICTS / 2 {
            f.store(ObjectId::new(), alice, true, true);
        }
        let held = f.cache.kept_verdict_count();
        let bytes = f.cache.table_bytes();
        eprintln!(
            "{held} verdicts kept in {} KiB of tables ({} B a verdict)",
            bytes / 1024,
            bytes / held
        );
        assert!(held <= MAX_KEPT_VERDICTS);
        assert!(
            bytes <= 24 << 20,
            "{held} verdicts take {} KiB of tables",
            bytes / 1024
        );
    }

    /// The cost of a write is what it changed: a row drops its own entry, and a table
    /// is walked for only when some kept verdict depends on a table at all.
    #[test]
    fn a_write_walks_the_cache_only_when_a_kept_verdict_depends_on_a_table() {
        let mut f = fixture(plain_docs(), false);
        let alice = f.alice;
        let rows: Vec<ObjectId> = (0..16).map(|_| ObjectId::new()).collect();
        for row in &rows {
            f.store(*row, alice, true, true);
        }

        f.cache.invalidate(["docs", "weather"], [rows[0]]);
        assert_eq!(f.cache.table_walks, 0, "nothing kept depends on a table");
        assert_eq!(f.get(rows[0], alice), None);
        assert_eq!(f.get(rows[1], alice), Some(true));

        // A table whose policy reads another was checked in a pass that is over: what
        // it reads is written every time someone joins a chat, and nothing is held
        // that such a write could change.
        let mut f = fixture(docs_reading_memberships(), false);
        let alice = f.alice;
        f.cache.begin_pass();
        f.store(rows[0], alice, true, false);
        f.cache.end_pass();
        f.cache.store(
            f.marker,
            rows[1],
            f.branch,
            TableName::new("plain"),
            f.scope,
            alice,
            true,
            true,
            &f.schema,
        );
        f.cache.invalidate(["memberships"], []);
        assert_eq!(
            f.cache.table_walks, 0,
            "a pass's verdicts are gone with it: nothing kept depends on a table"
        );
        assert_eq!(f.get(rows[1], alice), Some(true));

        // Switched on, verdicts that read `memberships` are kept: its writes walk.
        let mut f = fixture(docs_reading_memberships(), true);
        let alice = f.alice;
        for row in &rows {
            f.store(*row, alice, true, false);
        }
        f.cache.invalidate(["weather"], []);
        assert_eq!(f.cache.table_walks, 0, "no kept verdict reads `weather`");
        f.cache.invalidate(["memberships"], []);
        assert_eq!(f.cache.table_walks, 1);
        assert_eq!(f.cache.kept_verdict_count(), 0);
        assert_eq!(
            f.cache.young_verdicts, 0,
            "the count that turns the generations still holds what a walk dropped"
        );
        // Nothing of `docs` is kept any more: the next write has nothing to walk for.
        f.cache.invalidate(["memberships"], []);
        assert_eq!(
            f.cache.table_walks, 1,
            "a walk over a cache that reads nothing"
        );
    }

    /// What a comparison is made with: nothing served, nothing stored, and what was
    /// held before is still held after.
    #[test]
    fn a_bypassed_cache_evaluates_everything_and_leaves_what_it_held() {
        let mut f = fixture(plain_docs(), false);
        let (kept, asked) = (ObjectId::new(), ObjectId::new());
        let alice = f.alice;
        f.store(kept, alice, true, true);

        f.cache.bypassed = true;
        assert_eq!(f.get(kept, alice), None);
        f.store(asked, alice, true, true);
        f.cache.bypassed = false;

        assert_eq!(f.get(kept, alice), Some(true));
        assert_eq!(f.get(asked, alice), None);
    }
}
