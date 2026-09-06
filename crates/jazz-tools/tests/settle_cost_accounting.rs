//! Accounting gate for the settle-cost instrumentation.
//!
//! The instrumentation in `query_manager::settle_cost` exists so a production
//! settle can be split into re-evaluated include instances, subquery compiles,
//! per-row policy work and plain storage reads. An instrument that lies is
//! worse than none, so this gate pins the accounting itself against a scenario
//! whose shape is known exactly: N cached include instances, one write into the
//! include's inner table.
//!
//! It also pins the threshold mechanism in both directions — a pass under the
//! threshold must stay silent, a pass over it must emit exactly one line.

#![cfg(feature = "test")]

use std::io;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};

use jazz_tools::query_manager::manager::QueryManager;
use jazz_tools::query_manager::precise_dirty::force_precise_dirty;
use jazz_tools::query_manager::session::Session;
use jazz_tools::query_manager::settle_cost::{
    SettleCounts, force_settle_log_ms, note_subscription_settle,
};
use jazz_tools::query_manager::types::{
    ColumnDescriptor, ColumnType, RowDescriptor, Schema, SchemaBuilder, TableName, TableSchema,
    Value, permissions, policy_expr as pe,
};
use jazz_tools::storage::MemoryStorage;
use jazz_tools::sync_manager::{
    ClientId, InboxEntry, QueryId, QueryPropagation, Source, SyncManager, SyncPayload,
};
use jazz_tools::test_support::seeded_memory_storage;
use tracing_subscriber::fmt::MakeWriter;

/// Serialises every test here: the settle counters are process-global, so two
/// scenarios running at once would blend their deltas. Same shape as the
/// allocator lock in `include_instance_flatness.rs`.
fn measure_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
}

// ============================================================================
// Log capture
// ============================================================================

#[derive(Clone, Default)]
struct CapturedLogs(Arc<Mutex<Vec<u8>>>);

impl CapturedLogs {
    fn take(&self) -> String {
        let mut buffer = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        String::from_utf8_lossy(&std::mem::take(&mut *buffer)).into_owned()
    }
}

impl io::Write for CapturedLogs {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl<'writer> MakeWriter<'writer> for CapturedLogs {
    type Writer = Self;

    fn make_writer(&'writer self) -> Self::Writer {
        self.clone()
    }
}

/// The one global subscriber for this binary, installed on first use.
fn captured_logs() -> &'static CapturedLogs {
    static LOGS: OnceLock<CapturedLogs> = OnceLock::new();
    LOGS.get_or_init(|| {
        let logs = CapturedLogs::default();
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .with_ansi(false)
            .with_writer(logs.clone())
            .finish();
        tracing::subscriber::set_global_default(subscriber)
            .expect("install the capturing subscriber");
        logs
    })
}

// ============================================================================
// Scenario: `users`, each carrying an include of their `posts`
// ============================================================================

/// Outer rows, hence cached include instances. Small enough to read the
/// expected counts off the scenario by hand.
const OUTER_ROWS: usize = 8;

fn users_posts_schema() -> Schema {
    let mut schema = Schema::new();
    schema.insert(
        TableName::new("users"),
        RowDescriptor::new(vec![
            ColumnDescriptor::new("id", ColumnType::Integer),
            ColumnDescriptor::new("name", ColumnType::Text),
        ])
        .into(),
    );
    schema.insert(
        TableName::new("posts"),
        RowDescriptor::new(vec![
            ColumnDescriptor::new("id", ColumnType::Integer),
            ColumnDescriptor::new("title", ColumnType::Text),
            ColumnDescriptor::new("author_id", ColumnType::Integer),
        ])
        .into(),
    );
    schema
}

struct Scenario {
    query_manager: QueryManager,
    storage: MemoryStorage,
    schema: Schema,
    branch: String,
}

impl Scenario {
    /// `OUTER_ROWS` users, one post each, subscribed and fully settled.
    fn settled() -> Self {
        let mut query_manager = QueryManager::new(SyncManager::new());
        query_manager.set_current_schema(users_posts_schema(), "dev", "main");
        let schema = query_manager.schema_context().current_schema.clone();
        let branch = query_manager
            .schema_context()
            .branch_name()
            .as_str()
            .to_string();
        let storage = seeded_memory_storage(&schema);

        let mut scenario = Self {
            query_manager,
            storage,
            schema,
            branch,
        };

        for index in 0..OUTER_ROWS {
            scenario.insert_user(index as i32 + 1, index);
            scenario.insert_post(1_000_000 + index as i32 + 1, index as i32 + 1);
        }

        let query = scenario
            .query_manager
            .query("users")
            .with_array("posts", |sub| {
                sub.from("posts").correlate("author_id", "users.id")
            })
            .build();
        scenario
            .query_manager
            .subscribe(query)
            .expect("subscribe to the include query");
        scenario.query_manager.process(&mut scenario.storage);

        let delivered: usize = scenario
            .query_manager
            .take_updates()
            .iter()
            .map(|update| update.delta.added.len())
            .sum();
        assert_eq!(
            delivered, OUTER_ROWS,
            "every outer row must be live before the measured pass"
        );

        scenario
    }

    fn insert_user(&mut self, user_id: i32, index: usize) {
        self.insert(
            "users",
            &[
                Value::Integer(user_id),
                Value::Text(format!("user-{index}")),
            ],
        );
    }

    fn insert_post(&mut self, post_id: i32, author_id: i32) {
        self.insert(
            "posts",
            &[
                Value::Integer(post_id),
                Value::Text(format!("post-{post_id}")),
                Value::Integer(author_id),
            ],
        );
    }

    fn insert(&mut self, table: &str, values: &[Value]) {
        self.query_manager
            .insert_on_branch_with_schema_and_write_context_and_id(
                &mut self.storage,
                table,
                &self.branch,
                values,
                None,
                &self.schema,
                None,
                true,
            )
            .unwrap_or_else(|error| panic!("insert into {table} failed: {error:?}"));
    }

    /// One `process` pass, reported as the cost the instrumentation recorded.
    fn measured_pass(&mut self) -> SettleCounts {
        let base = SettleCounts::snapshot();
        self.query_manager.process(&mut self.storage);
        SettleCounts::snapshot().since(base)
    }
}

fn report(label: &str, cost: &SettleCounts) {
    eprintln!("{label}: {cost:?}");
}

// ============================================================================
// Gates
// ============================================================================

/// One row into the include's inner table, with `OUTER_ROWS` instances cached.
///
/// The numbers are not "whatever it does today" — each is what the scenario
/// implies:
///
/// * ONE subscription is live, so exactly one settles.
/// * The written row correlates to exactly ONE outer row, so correlation
///   routing (v14 L1) marks exactly one cached instance and exactly one is
///   re-evaluated; the other `OUTER_ROWS - 1` cannot hold it and stay clean.
///   Before routing this read `OUTER_ROWS` — buffered dirt carried a
///   node-global generation, so one mark made every cached instance look stale
///   (the target `include_instance_flatness.rs` pinned). The point here is
///   that the instrument REPORTS the difference exactly, instead of
///   under-counting either state.
/// * Every instance's correlation binding is unchanged, so not one of them may
///   compile. Zero instantiations, zero plan compiles — the split between "(a)
///   re-evaluating instances" and "(b) compiling plans" is the whole question
///   the instrument exists to answer, and a compile leaking into a pure
///   re-evaluation pass would silently answer it wrong.
/// * The schema carries no policies, so no per-row policy work may be reported.
/// * Exactly one outer row gained a post, so exactly one row is emitted.
#[test]
fn inner_row_write_reports_instance_evals_without_compiles() {
    let _serialised = measure_lock();
    let _precise = force_precise_dirty(true);

    let mut scenario = Scenario::settled();
    scenario.insert_post(2_000_000, 1);
    let cost = scenario.measured_pass();
    report("inner-row write", &cost);

    assert_eq!(cost.subscriptions, 1, "one live subscription settles once");
    assert_eq!(
        cost.instance_evals, 1,
        "correlation routing marks only the instance whose binding matches the written row"
    );
    assert_eq!(
        cost.subquery_instantiations, 0,
        "no correlation binding changed, so no instance may be re-instantiated"
    );
    assert_eq!(
        cost.plan_compiles, 0,
        "a pure re-evaluation pass must not compile a plan"
    );
    assert_eq!(
        cost.policy_row_evals, 0,
        "the scenario schema declares no policies"
    );
    assert_eq!(
        cost.scope_authz_checks, 0,
        "a local subscription computes no authorized sync scope"
    );
    assert_eq!(
        cost.rows_emitted, 1,
        "exactly the one outer row whose include gained a post changes"
    );
    assert!(
        cost.row_loads > 0 && cost.graph_nodes > 0 && cost.index_reads > 0,
        "the pass did real graph and storage work: {cost:?}"
    );
    // The gauge is what separates "N instances evaluated once" from "one
    // instance evaluated N times" — two readings that are equal here only
    // because this scenario re-evaluates every instance exactly once. An
    // instance gauge that merely tracked the evaluation counter would report
    // the same number and answer nothing.
    assert_eq!(
        cost.live_instances, OUTER_ROWS as u64,
        "one cached subgraph instance per outer row is live"
    );
    assert_eq!(
        cost.live_instance_nodes, 1,
        "the subscription graph carries exactly one include node"
    );
}

/// A new outer row, which is the shape that MUST compile.
///
/// One user arrives with no cached instance for its correlation value, so
/// exactly one instantiation happens, and it compiles exactly one plan (the
/// include's inner query has no nested include of its own). The inner table did
/// not move, so the other instances stay clean and only the new one is
/// evaluated. This is the counter-test to the one above: the same instrument
/// must separate the two shapes, not report a constant.
#[test]
fn new_outer_row_reports_exactly_one_instantiation() {
    let _serialised = measure_lock();
    let _precise = force_precise_dirty(true);

    let mut scenario = Scenario::settled();
    scenario.insert_user(OUTER_ROWS as i32 + 1, OUTER_ROWS);
    let cost = scenario.measured_pass();
    report("new outer row", &cost);

    assert_eq!(cost.subscriptions, 1, "one live subscription settles once");
    assert_eq!(
        cost.instance_evals, 1,
        "only the new outer row's instance is evaluated"
    );
    assert_eq!(
        cost.subquery_instantiations, 1,
        "the new correlation binding is the only one needing a fresh subgraph"
    );
    assert_eq!(
        cost.plan_compiles, 1,
        "one instantiation compiles exactly one inner plan"
    );
    assert_eq!(cost.rows_emitted, 1, "one outer row is added");
}

/// Per-row policy work is reported, and reported per row.
///
/// `policy_row_evals` is the (c) of the split the instrumentation exists to
/// settle, and it is the one counter that would read plausibly as `0` if it
/// were wired to a site the settle never reaches. Pinning it to the row count
/// of a policy-bearing table's first settle is what makes a silent `0` in
/// production mean "no policy work", not "no instrument".
#[test]
fn policy_bearing_table_reports_one_row_eval_per_row() {
    let _serialised = measure_lock();

    let schema = SchemaBuilder::new()
        .table(
            TableSchema::builder("documents")
                .column("title", ColumnType::Text)
                .nullable_column("deleted_at", ColumnType::Text)
                .policies(permissions(|p| {
                    p.allow_read().where_(pe::is_null("deleted_at"));
                })),
        )
        .build();

    let mut query_manager = QueryManager::new(SyncManager::new());
    query_manager.set_current_schema(schema, "dev", "main");
    let schema = query_manager.schema_context().current_schema.clone();
    let branch = query_manager
        .schema_context()
        .branch_name()
        .as_str()
        .to_string();
    let mut storage = seeded_memory_storage(&schema);

    for index in 0..OUTER_ROWS {
        query_manager
            .insert_on_branch_with_schema_and_write_context_and_id(
                &mut storage,
                "documents",
                &branch,
                &[Value::Text(format!("doc-{index}")), Value::Null],
                None,
                &schema,
                None,
                true,
            )
            .expect("seed document");
    }

    // A session is what makes the compiler insert a `PolicyFilter` node —
    // without one the policy is never evaluated per row.
    let query = query_manager.query("documents").build();
    query_manager
        .subscribe_with_session(query, Some(Session::new("alice")), None)
        .expect("subscribe to the policy-bearing table");

    let base = SettleCounts::snapshot();
    query_manager.process(&mut storage);
    let cost = SettleCounts::snapshot().since(base);
    report("policy-bearing first settle", &cost);

    assert_eq!(
        cost.policy_row_evals, OUTER_ROWS as u64,
        "the select policy is evaluated once per row entering the result set"
    );
    assert_eq!(
        cost.rows_emitted, OUTER_ROWS as u64,
        "every row passes the policy, so every row is emitted"
    );
}

/// Scope authorization for a session-scoped downstream client is reported,
/// and the pass names the client it was spent on.
///
/// This is the fan-out half of the production question: an isolated
/// backend-role repro cost ~70 ms/message against ~1 s live, and the gap was
/// attributed to per-row policy work for session-scoped subscribers without
/// ever being measured. `scope_authz_checks` / `scope_authz_evals` are that
/// measurement, so they must be non-zero exactly when a session-scoped server
/// subscription settles — and the line must say WHICH client.
#[test]
fn session_scoped_server_subscription_reports_scope_authorization() {
    let _serialised = measure_lock();
    let logs = captured_logs();

    let schema = SchemaBuilder::new()
        .table(
            TableSchema::builder("items")
                .column("owner_id", ColumnType::Text)
                .column("name", ColumnType::Text)
                .policies(permissions(|p| {
                    p.allow_read()
                        .where_(pe::eq("owner_id", pe::session("user_id")));
                })),
        )
        .build();

    let mut query_manager = QueryManager::new(SyncManager::new());
    query_manager.set_current_schema(schema, "dev", "main");
    let schema = query_manager.schema_context().current_schema.clone();
    let branch = query_manager
        .schema_context()
        .branch_name()
        .as_str()
        .to_string();
    let mut storage = seeded_memory_storage(&schema);

    for index in 0..OUTER_ROWS {
        query_manager
            .insert_on_branch_with_schema_and_write_context_and_id(
                &mut storage,
                "items",
                &branch,
                &[
                    Value::Text("alice".into()),
                    Value::Text(format!("item-{index}")),
                ],
                None,
                &schema,
                None,
                true,
            )
            .expect("seed item");
    }
    query_manager.process(&mut storage);

    let client_id = ClientId(uuid::Uuid::new_v4());
    query_manager
        .sync_manager_mut()
        .add_client_with_storage(&storage, client_id);
    let query = query_manager.query("items").build();
    query_manager.sync_manager_mut().push_inbox(InboxEntry {
        source: Source::Client(client_id),
        payload: SyncPayload::QuerySubscription {
            query_id: QueryId(1),
            query: Box::new(query),
            session: Some(Session::new("alice")),
            required_tier: None,
            propagation: QueryPropagation::Full,
            policy_context_tables: vec![],
        },
    });

    let _threshold = force_settle_log_ms(0);
    logs.take();
    let base = SettleCounts::snapshot();
    query_manager.process(&mut storage);
    let cost = SettleCounts::snapshot().since(base);
    let line = logs.take();
    report("session-scoped server subscription", &cost);

    assert_eq!(
        cost.scope_authz_checks, OUTER_ROWS as u64,
        "the sync scope authorizes every row it carries"
    );
    assert_eq!(
        cost.scope_authz_evals, OUTER_ROWS as u64,
        "a cold verdict cache means every check is a real evaluation"
    );
    assert!(
        cost.policy_row_evals >= OUTER_ROWS as u64,
        "each scope evaluation runs the select policy against a row: {cost:?}"
    );
    assert!(
        line.contains(&format!("hot_client=\"{client_id}\"")) && line.contains("hot_query=1"),
        "the cost line must name the client and query it was spent on, got: {line}"
    );
}

/// The threshold gate, both directions, plus the shape of the emitted line.
#[test]
fn threshold_decides_whether_the_pass_is_logged() {
    let _serialised = measure_lock();
    let _precise = force_precise_dirty(true);
    let logs = captured_logs();

    let mut scenario = Scenario::settled();

    // Above any real settle duration: silence.
    {
        let _threshold = force_settle_log_ms(60_000);
        logs.take();
        scenario.insert_post(3_000_000, 1);
        scenario.measured_pass();
        assert_eq!(
            logs.take(),
            "",
            "a pass under the threshold must emit nothing at all"
        );
    }

    // Zero: every pass logs.
    let line = {
        let _threshold = force_settle_log_ms(0);
        logs.take();
        scenario.insert_post(3_000_001, 1);
        scenario.measured_pass();
        logs.take()
    };

    eprintln!("sample settle-cost line:\n{line}");
    assert_eq!(
        line.matches("jazz settle pass cost").count(),
        1,
        "exactly one cost line per settle pass, got: {line}"
    );
    let live_instances_field = format!("live_instances={OUTER_ROWS}");
    for field in [
        "micros=",
        "subscriptions=1",
        "graph_nodes=",
        "instance_evals=",
        "subquery_instantiations=0",
        "plan_compiles=0",
        "shape_lowerings=0",
        "units_deferred=0",
        "ticks_rearmed=0",
        "policy_row_evals=0",
        "scope_authz_checks=0",
        "scope_authz_evals=0",
        "row_loads=",
        "index_reads=",
        "rows_emitted=1",
        live_instances_field.as_str(),
        "live_instance_nodes=1",
        "hot_micros=",
        "hot_client=\"local\"",
        "hot_query=",
    ] {
        assert!(line.contains(field), "cost line is missing {field}: {line}");
    }
}

/// The hot-subscription slot keeps the pass maximum, not the last writer.
#[test]
fn hot_subscription_slot_keeps_the_maximum() {
    let _serialised = measure_lock();
    let logs = captured_logs();
    let _threshold = force_settle_log_ms(0);

    let pass = jazz_tools::query_manager::settle_cost::SettlePass::begin();
    note_subscription_settle(None, 7, std::time::Duration::from_millis(40));
    note_subscription_settle(None, 9, std::time::Duration::from_millis(2));
    logs.take();
    drop(pass);

    let line = logs.take();
    assert!(
        line.contains("hot_micros=40000") && line.contains("hot_query=7"),
        "the pass must report its most expensive subscription, got: {line}"
    );
}
