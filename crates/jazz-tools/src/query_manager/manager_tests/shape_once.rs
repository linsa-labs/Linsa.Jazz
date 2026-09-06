//! v18 item 7: the shape of a correlated subquery is lowered once per template.
//!
//! `SubgraphTemplate::instantiate` (`graph_nodes/subgraph.rs`) rebuilds a `Query` through the
//! builder with the correlation value as an equality filter, regenerates its relation IR,
//! lowers that IR to an execution plan and compiles the plan — once per outer row. Only the
//! value differs between instances; the lowering is the same work every time (prod: one new
//! `messages` subscription = 1 182 instantiations = 1 182 lowerings; sampled here: the
//! instantiate is 18 % of the pass, the lowering, IR refresh and branch resolution a quarter
//! of that).
//!
//! Internal on purpose: how many times a shape was lowered is not visible through any client
//! API; the gate reaches the template through the server subscription's graph. The
//! plan-equality gates build templates directly, the way `compile_array_subquery` does, and
//! compare execution plans — a crate-private type.
//!
//! Fixture style: this file uses the module's `qm.insert` / `RowDescriptor` fixtures
//! (`users_posts_schema`, `Schema::new()` + inserts), not `row_input!` / `TableSchema`
//! builders, except where a shape needs an explicit index set (`index_only`).

use super::*;
use crate::query_manager::graph::GraphNode;
use crate::query_manager::graph_nodes::subgraph::SubgraphTemplate;
use crate::query_manager::query::{Condition, Query};
use crate::query_manager::types::{RowPolicyMode, SchemaBuilder, TableSchema};
use crate::schema_manager::SchemaContext;
use crate::sync_manager::{ClientId, QueryId};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::Arc;

const USERS: i32 = 40;

fn seeded_server() -> (QueryManager, MemoryStorage) {
    let (mut qm, mut storage) = create_query_manager(SyncManager::new(), users_posts_schema());
    for user in 1..=USERS {
        qm.insert(
            &mut storage,
            "users",
            &[Value::Integer(user), Value::Text(format!("user-{user}"))],
        )
        .unwrap();
        qm.insert(
            &mut storage,
            "posts",
            &[
                Value::Integer(user * 100),
                Value::Text(format!("post-{user}")),
                Value::Integer(user),
            ],
        )
        .unwrap();
    }
    qm.process(&mut storage);
    let _ = qm.sync_manager_mut().take_outbox();
    (qm, storage)
}

/// G7-1. One include registration over forty outer rows instantiates forty subgraphs and
/// lowers the inner shape once. Red today: forty lowerings.
#[test]
fn an_include_over_forty_rows_lowers_its_shape_once() {
    let (mut qm, mut storage) = seeded_server();
    let client = ClientId::new();
    connect_client(&mut qm, &storage, client);
    let query = qm
        .query("users")
        .with_array("posts", |sub| {
            sub.from("posts").correlate("author_id", "users.id")
        })
        .build();
    push_query_subscription(&mut qm, client, 1, query);
    qm.process(&mut storage);
    let sub = qm
        .server_subscriptions
        .get(&(client, QueryId(1)))
        .expect("the registration settled into a server subscription");
    let node = sub
        .graph
        .nodes
        .iter()
        .find_map(|node| match &node.node {
            GraphNode::ArraySubquery(node) => Some(node),
            _ => None,
        })
        .expect("the include compiles to an array-subquery node");
    assert_eq!(
        node.cached_subgraph_count(),
        USERS as usize,
        "fixture: one instance per outer row"
    );
    assert_eq!(
        node.subgraph_template_for_test().shape_lowerings_for_test(),
        1,
        "the inner shape must be lowered once per template, not once per instance"
    );
    assert_eq!(
        node.subgraph_template_for_test()
            .branch_map_builds_for_test(),
        1,
        "the branch -> schema map must be built once per template, not once per instance"
    );
}

/// G7-7 (diff r3 Blocking 2). A join-shaped include — the plan `compile_execution_plan`
/// hands to `compile_join_plan` — is a template like any other: forty outer rows, one
/// lowering of the inner shape and one build of the branch map. Measures what item 7 buys
/// the family of includes the production symptom is made of.
#[test]
fn a_join_shaped_include_over_forty_rows_lowers_its_shape_once() {
    let (mut qm, mut storage) = seeded_server();
    let client = ClientId::new();
    connect_client(&mut qm, &storage, client);
    let query = qm
        .query("users")
        .with_array("posts_with_authors", |sub| {
            sub.from("posts")
                .join("users")
                .on("posts.author_id", "users.id")
                .correlate("author_id", "users.id")
        })
        .build();
    push_query_subscription(&mut qm, client, 1, query);
    qm.process(&mut storage);
    let sub = qm
        .server_subscriptions
        .get(&(client, QueryId(1)))
        .expect("the registration settled into a server subscription");
    let node = sub
        .graph
        .nodes
        .iter()
        .find_map(|node| match &node.node {
            GraphNode::ArraySubquery(node) => Some(node),
            _ => None,
        })
        .expect("the include compiles to an array-subquery node");
    assert_eq!(
        node.cached_subgraph_count(),
        USERS as usize,
        "fixture: one instance per outer row"
    );
    assert_eq!(
        node.subgraph_template_for_test().shape_lowerings_for_test(),
        1,
        "a join-shaped inner shape must be lowered once per template, not once per instance"
    );
    assert_eq!(
        node.subgraph_template_for_test()
            .branch_map_builds_for_test(),
        1,
        "the branch -> schema map must be built once per template for a join-shaped include"
    );
}

/// `shapes_schema` with `comments` declaring an `id` column, so the same correlation on
/// `comments.id` binds to the declared column here and to the row id there.
fn shapes_schema_with_comment_ids() -> Schema {
    SchemaBuilder::new()
        .table(
            TableSchema::builder("users")
                .column("id", ColumnType::Integer)
                .column("name", ColumnType::Text),
        )
        .table(
            TableSchema::builder("posts")
                .column("id", ColumnType::Integer)
                .column("title", ColumnType::Text)
                .column("author_id", ColumnType::Integer),
        )
        .table(
            TableSchema::builder("comments")
                .column("id", ColumnType::Integer)
                .column("body", ColumnType::Text)
                .column("post_id", ColumnType::Integer),
        )
        .build()
}

/// G7-5. A template bound under another `Arc<Schema>` than the one it cached lowers afresh
/// for that instance and binds equal to the uncached lowering under that schema; the cache
/// is kept for the schema that recurs. The two schemas differ where it shows: `comments`
/// declares `id` in one and not in the other, so the row-id binding flips.
#[test]
fn a_template_bound_under_another_schema_lowers_afresh_and_binds_equal() {
    let without_id = Arc::new(shapes_schema(&[]));
    let with_id = Arc::new(shapes_schema_with_comment_ids());
    let main = |table: &str| QueryBuilder::new(table).branches(&["main"]);
    let template = template(&without_id, main("comments").build(), "id", &[]);
    let value = Value::Integer(3);

    let cached = template.bind_for_test(&value, &without_id);
    assert_eq!(
        cached,
        template.instantiate_plan_uncached_for_test(&value, &without_id),
        "the cached schema binds equal to the uncached lowering"
    );
    assert_eq!(template.shape_lowerings_for_test(), 1);

    let other = template.bind_for_test(&value, &with_id);
    assert_eq!(
        other,
        template.instantiate_plan_uncached_for_test(&value, &with_id),
        "another schema: a fresh lowering, equal to the uncached one under that schema"
    );
    assert_ne!(
        cached, other,
        "fixture: the two schemas lower differently (row id vs declared id)"
    );
    assert_eq!(
        template.shape_lowerings_for_test(),
        2,
        "the other schema lowered afresh once"
    );

    let again = template.bind_for_test(&value, &without_id);
    assert_eq!(
        again, cached,
        "the cached schema still serves from the cache"
    );
    assert_eq!(
        template.shape_lowerings_for_test(),
        2,
        "the cache survived the detour through another schema"
    );
}

/// G7-8 (diff r5 SF3, exit B). A template that cached its shape and then fails to lower under
/// ANOTHER schema retires the cache for good — `uncacheable` is not keyed by schema: the next
/// bind under the schema that recurs goes straight to the uncached lowering and no attempt is
/// paid. Red under a per-schema (or unset) flag: the third bind serves the cached shape.
#[test]
fn a_miss_under_another_schema_retires_the_cache_for_the_recurring_one() {
    let with_comments = Arc::new(shapes_schema(&[]));
    let without_comments = Arc::new(test_schema());
    let main = |table: &str| QueryBuilder::new(table).branches(&["main"]);
    let template = template(&with_comments, main("comments").build(), "id", &[]);
    let value = Value::Integer(3);

    assert!(
        template.bind_for_test(&value, &with_comments).is_some(),
        "fixture: the shape caches under the schema it was built for"
    );
    assert_eq!(template.shape_lowerings_for_test(), 1);
    assert!(
        template.bind_for_test(&value, &without_comments).is_none(),
        "fixture: the inner table is absent from the other schema, the lowering fails (exit B)"
    );
    assert_eq!(
        template.shape_lowerings_for_test(),
        2,
        "the failed lowering is one attempt (attempts are counted, as in G7-6)"
    );
    assert!(
        template.bind_for_test(&value, &with_comments).is_none(),
        "the cache is retired for the recurring schema too: the instance lowers uncached"
    );
    assert_eq!(
        template.shape_lowerings_for_test(),
        2,
        "no attempt is paid once the template is retired"
    );
}

/// G7-6 (diff r2 Blocking 1): a shape that cannot be lowered is remembered on the template —
/// the next instance goes straight to the uncached lowering instead of paying the failed
/// attempt again. Here the schema the template is bound under lacks the inner table, so
/// the lowering fails; the first bind counts one attempt, the second bind counts none.
#[test]
fn a_template_that_missed_once_does_not_retry_the_shape() {
    let with_comments = Arc::new(shapes_schema(&[]));
    let without_comments = Arc::new(test_schema());
    let main = |table: &str| QueryBuilder::new(table).branches(&["main"]);
    let template = template(&with_comments, main("comments").build(), "id", &[]);
    let value = Value::Integer(3);

    assert!(
        template.bind_for_test(&value, &without_comments).is_none(),
        "fixture: the inner table is absent from this schema, the shape cannot be lowered"
    );
    let after_miss = template.shape_lowerings_for_test();
    assert!(
        template.bind_for_test(&value, &without_comments).is_none(),
        "the miss is remembered: no plan"
    );
    assert_eq!(
        template.shape_lowerings_for_test(),
        after_miss,
        "the second instance did not retry the shape"
    );
    assert!(
        template.bind_for_test(&value, &with_comments).is_none(),
        "the template is uncacheable for good, under any schema (instances lower uncached)"
    );
}

/// A template built the way `compile_array_subquery` builds one: base query, correlation
/// column, selected columns, the inner table's descriptor as output.
fn template(schema: &Schema, base: Query, inner_column: &str, select: &[&str]) -> SubgraphTemplate {
    let output = schema
        .get(&base.table)
        .expect("the inner table exists")
        .columns
        .clone();
    SubgraphTemplate::new(
        base,
        inner_column.to_string(),
        select.iter().map(|column| column.to_string()).collect(),
        output,
        Arc::new(SchemaContext::with_defaults(schema.clone(), "main")),
        None,
        RowPolicyMode::PermissiveLocal,
    )
}

/// `posts` with an explicit index set; `comments` has no `id` column (a row-id table).
fn shapes_schema(index_only: &[&str]) -> Schema {
    let mut posts = TableSchema::builder("posts")
        .column("id", ColumnType::Integer)
        .column("title", ColumnType::Text)
        .column("author_id", ColumnType::Integer);
    if !index_only.is_empty() {
        posts = posts.index_only(index_only.iter().copied());
    }
    SchemaBuilder::new()
        .table(
            TableSchema::builder("users")
                .column("id", ColumnType::Integer)
                .column("name", ColumnType::Text),
        )
        .table(posts)
        .table(
            TableSchema::builder("comments")
                .column("body", ColumnType::Text)
                .column("post_id", ColumnType::Integer),
        )
        .build()
}

/// G7-2. The cached shape bound to a value is today's per-instance lowering, plan for plan,
/// across the shapes that could break the position-0 rule; and it is lowered once.
#[test]
fn the_bound_cached_shape_equals_the_uncached_lowering_across_shapes() {
    let plain = shapes_schema(&[]);
    let indexed = shapes_schema(&["author_id"]);
    let two = shapes_schema(&["title", "author_id"]);
    let main = |table: &str| QueryBuilder::new(table).branches(&["main"]);
    let shapes: Vec<(&str, &Schema, SubgraphTemplate)> = vec![
        (
            "declared column",
            &plain,
            template(&plain, main("posts").build(), "author_id", &[]),
        ),
        (
            "id on a row-id table",
            &plain,
            template(&plain, main("comments").build(), "id", &[]),
        ),
        (
            "id on a table declaring id",
            &plain,
            template(&plain, main("posts").build(), "id", &[]),
        ),
        (
            "explicit index",
            &indexed,
            template(&indexed, main("posts").build(), "author_id", &[]),
        ),
        (
            "empty-branch base",
            &plain,
            template(&plain, QueryBuilder::new("posts").build(), "author_id", &[]),
        ),
        (
            "base select (no effect: the template's own select applies, `instance_query`)",
            &plain,
            template(&plain, main("posts").select(&[]).build(), "author_id", &[]),
        ),
        (
            "base Eq on another indexed column",
            &two,
            template(
                &two,
                main("posts")
                    .filter_eq("title", Value::Text("t".to_string()))
                    .build(),
                "author_id",
                &[],
            ),
        ),
        (
            "order, limit, offset, select",
            &plain,
            template(
                &plain,
                main("posts")
                    .order_by_desc("title")
                    .limit(3)
                    .offset(1)
                    .build(),
                "author_id",
                &["title"],
            ),
        ),
    ];
    // Every kind the correlation takes in production (uuid array-FK includes, integer and
    // text keys), plus the kinds a reader will doubt: `Null` as a REAL bound value, bool,
    // double, bigint, bytes. Lowering is value-neutral; binding must be too.
    let values = [
        Value::Integer(1),
        Value::Integer(42),
        Value::Text("q".to_string()),
        Value::Null,
        Value::Boolean(true),
        Value::Double(1.5),
        Value::BigInt(7),
        Value::Uuid(crate::object::ObjectId::new()),
        Value::Bytea(vec![1, 2, 3]),
    ];
    for (name, schema, template) in &shapes {
        let schema = Arc::new((*schema).clone());
        for value in &values {
            let uncached = template.instantiate_plan_uncached_for_test(value, &schema);
            assert!(
                uncached.is_some(),
                "fixture: shape {name} lowers with {value:?}"
            );
            let bound = template.bind_for_test(value, &schema);
            assert_eq!(
                bound, uncached,
                "shape {name}, value {value:?}: the bound cached shape must be today's lowering"
            );
        }
        assert_eq!(
            template.shape_lowerings_for_test(),
            1,
            "shape {name}: lowered once for every value"
        );
    }
}

/// G7-3. The rebuild collapses a base with several conditions to ONE conjunction with the
/// correlation first — today's behaviour, unchanged by the cache — and the cached shape
/// binds it in that position.
#[test]
fn a_base_with_several_conditions_binds_as_one_conjunction_equal_to_todays() {
    let schema = Arc::new(shapes_schema(&[]));
    let template = template(
        &schema,
        QueryBuilder::new("posts")
            .branches(&["main"])
            .filter_eq("title", Value::Text("t".to_string()))
            .filter_gt("id", Value::Integer(10))
            .build(),
        "author_id",
        &[],
    );
    let value = Value::Integer(7);
    let bound = template
        .bind_for_test(&value, &schema)
        .expect("the shape binds");
    assert_eq!(
        bound.disjuncts.len(),
        1,
        "one conjunction: {:?}",
        bound.disjuncts
    );
    assert_eq!(bound.disjuncts[0].conditions.len(), 3);
    assert!(
        matches!(&bound.disjuncts[0].conditions[0], Condition::Eq { value: bound_value, .. } if *bound_value == value),
        "the correlation leads the conjunction: {:?}",
        bound.disjuncts[0].conditions
    );
    assert_eq!(
        Some(bound),
        template.instantiate_plan_uncached_for_test(&value, &schema),
        "equal to today's lowering, plan for plan"
    );
}

/// The differential for item 7: 200 random shapes over G7-2's axes (table, correlation
/// column, base conditions, order, limit, offset, select, index set, branch list), three
/// values each. The bound cached plan must equal the uncached lowering plan for plan, and
/// the shape lowers once. Shapes the builder rejects are skipped (both paths share it).
#[test]
fn random_shapes_bind_equal_to_the_uncached_lowering() {
    let seed = std::env::var("JAZZ_SHAPE_SEED")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or_else(rand::random::<u64>);
    eprintln!("shape differential seed = {seed} (JAZZ_SHAPE_SEED to replay)");
    let mut rng = StdRng::seed_from_u64(seed);
    let schemas = [
        shapes_schema(&[]),
        shapes_schema(&["author_id"]),
        shapes_schema(&["title", "author_id"]),
    ];
    let mut compared = 0;
    for round in 0..200 {
        let schema = &schemas[rng.gen_range(0..schemas.len())];
        let (table, columns, inner) = if rng.gen_bool(0.7) {
            (
                "posts",
                ["id", "title", "author_id"],
                if rng.gen_bool(0.5) { "author_id" } else { "id" },
            )
        } else {
            (
                "comments",
                ["post_id", "body", "post_id"],
                if rng.gen_bool(0.5) { "post_id" } else { "id" },
            )
        };
        let mut query = QueryBuilder::new(table);
        if rng.gen_bool(0.8) {
            query = query.branches(&["main"]);
        }
        for _ in 0..rng.gen_range(0..=2) {
            let column = columns[rng.gen_range(0..columns.len())];
            query = match rng.gen_range(0..5) {
                0 => query.filter_eq(column, Value::Integer(rng.gen_range(0..9))),
                1 => query.filter_ne(column, Value::Text("x".to_string())),
                2 => query.filter_gt(column, Value::Integer(3)),
                3 => query.filter_is_null(column),
                _ => query.filter_lt(column, Value::Integer(5)),
            };
        }
        query = match rng.gen_range(0..3) {
            0 => query,
            1 => query.order_by(columns[1]),
            _ => query.order_by_desc(columns[1]),
        };
        if rng.gen_bool(0.4) {
            query = query.limit(rng.gen_range(1..5));
        }
        if rng.gen_bool(0.3) {
            query = query.offset(rng.gen_range(1..3));
        }
        let select: Vec<&str> = if rng.gen_bool(0.3) {
            vec![columns[1]]
        } else {
            vec![]
        };
        let Ok(base) = query.try_build() else {
            continue;
        };
        let template = template(schema, base, inner, &select);
        let schema = Arc::new(schema.clone());
        for _ in 0..3 {
            let value = match rng.gen_range(0..6) {
                0 => Value::Integer(rng.gen_range(0..100)),
                1 => Value::Text(format!("k{}", rng.gen_range(0..100))),
                2 => Value::Null,
                3 => Value::Boolean(rng.gen_bool(0.5)),
                4 => Value::Uuid(crate::object::ObjectId::new()),
                _ => Value::BigInt(rng.gen_range(0..1_000_000)),
            };
            assert_eq!(
                template.bind_for_test(&value, &schema),
                template.instantiate_plan_uncached_for_test(&value, &schema),
                "seed {seed}, round {round}, value {value:?}"
            );
            compared += 1;
        }
        assert_eq!(
            template.shape_lowerings_for_test(),
            1,
            "seed {seed}, round {round}: the shape lowers once"
        );
    }
    assert!(
        compared >= 300,
        "seed {seed}: the builder rejected too many shapes ({compared} comparisons)"
    );
}

fn users_posts_comments_schema() -> Schema {
    let mut schema = users_posts_schema();
    schema.insert(
        TableName::new("comments"),
        RowDescriptor::new(vec![
            ColumnDescriptor::new("id", ColumnType::Integer),
            ColumnDescriptor::new("body", ColumnType::Text),
            ColumnDescriptor::new("post_id", ColumnType::Integer),
        ])
        .into(),
    );
    schema
}

/// G7-4 (r3 S6). A nested include: the outer template lowers once; a child template is made
/// per parent instance and each lowers once — N parents give 1 + N lowerings, never
/// N + N·M compiles' worth.
#[test]
fn a_nested_include_lowers_once_per_template_at_each_level() {
    const COMMENTS_PER_POST: i32 = 2;
    let (mut qm, mut storage) =
        create_query_manager(SyncManager::new(), users_posts_comments_schema());
    for user in 1..=USERS {
        qm.insert(
            &mut storage,
            "users",
            &[Value::Integer(user), Value::Text(format!("user-{user}"))],
        )
        .unwrap();
        qm.insert(
            &mut storage,
            "posts",
            &[
                Value::Integer(user * 100),
                Value::Text(format!("post-{user}")),
                Value::Integer(user),
            ],
        )
        .unwrap();
        for comment in 1..=COMMENTS_PER_POST {
            qm.insert(
                &mut storage,
                "comments",
                &[
                    Value::Integer(user * 1000 + comment),
                    Value::Text(format!("comment-{user}-{comment}")),
                    Value::Integer(user * 100),
                ],
            )
            .unwrap();
        }
    }
    qm.process(&mut storage);
    let _ = qm.sync_manager_mut().take_outbox();

    let client = ClientId::new();
    connect_client(&mut qm, &storage, client);
    let query = qm
        .query("users")
        .with_array("posts", |sub| {
            sub.from("posts")
                .correlate("author_id", "users.id")
                .with_array("comments", |nested| {
                    nested.from("comments").correlate("post_id", "posts.id")
                })
        })
        .build();
    push_query_subscription(&mut qm, client, 1, query);
    qm.process(&mut storage);
    let sub = qm
        .server_subscriptions
        .get(&(client, QueryId(1)))
        .expect("the registration settled into a server subscription");
    let outer = sub
        .graph
        .nodes
        .iter()
        .find_map(|node| match &node.node {
            GraphNode::ArraySubquery(node) => Some(node),
            _ => None,
        })
        .expect("the include compiles to an array-subquery node");
    assert_eq!(
        outer.cached_subgraph_count(),
        USERS as usize,
        "fixture: one instance per user"
    );
    assert_eq!(
        outer
            .subgraph_template_for_test()
            .shape_lowerings_for_test(),
        1,
        "the outer shape lowers once"
    );
    let mut child_templates = 0;
    for instance in outer.cached_subgraph_instances_for_test() {
        let child = instance
            .graph
            .nodes
            .iter()
            .find_map(|node| match &node.node {
                GraphNode::ArraySubquery(node) => Some(node),
                _ => None,
            })
            .expect("each parent instance carries the nested include node");
        assert_eq!(
            child.cached_subgraph_count(),
            1,
            "fixture: one post per user, one child instance"
        );
        assert_eq!(
            child
                .subgraph_template_for_test()
                .shape_lowerings_for_test(),
            1,
            "a child template lowers its shape once"
        );
        child_templates += 1;
    }
    assert_eq!(
        child_templates, USERS as usize,
        "one child template per parent instance"
    );
}
