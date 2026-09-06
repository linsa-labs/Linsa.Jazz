//! Gates for admission control on the sync server (linsa-v18 item 2, defect #31): an
//! over-cap registration is refused before it costs a compile, every cap is exact, and
//! nothing the app issues today comes near a cap.
//!
//! Black-box: `JazzServer` with caps set through the builder (so the verdicts do not depend
//! on the environment), `JazzClient`s connected as users or as the backend. A refusal is
//! observed through a one-shot read at the edge tier, which fails with the server's
//! rejection; standing subscriptions are counted on the server through the test hook
//! `inspect_core_for_test`, because no wire message reports "admitted".
#![cfg(feature = "test")]

mod support;

use std::time::Duration;

use jazz_tools::query_manager::query::{ArraySubqueryBuilder, Query, QueryBuilder};
use jazz_tools::query_manager::types::{ColumnType, Schema, SchemaBuilder, TableSchema, Value};
use jazz_tools::server::JazzServer;
use jazz_tools::sync_manager::{DurabilityTier, SubscriptionCaps};
use jazz_tools::{AppContext, ClientStorage, JazzClient};
use std::collections::HashMap;

fn nodes_schema() -> Schema {
    SchemaBuilder::new()
        .table(
            TableSchema::builder("nodes")
                .column("label", ColumnType::Text)
                .nullable_fk_column("parent", "nodes"),
        )
        .build()
}

/// `nodes` with a `children` include nested `depth` levels deep (0 = no include).
fn children_chain(depth: usize) -> Query {
    fn nest(builder: ArraySubqueryBuilder, remaining: usize) -> ArraySubqueryBuilder {
        let builder = builder.from("nodes").correlate("parent", "nodes.id");
        if remaining <= 1 {
            builder
        } else {
            builder.with_array("children", |sub| nest(sub, remaining - 1))
        }
    }
    let mut query = QueryBuilder::new("nodes");
    if depth > 0 {
        query = query.with_array("children", |sub| nest(sub, depth));
    }
    query.build()
}

fn caps_with(update: impl FnOnce(&mut SubscriptionCaps)) -> SubscriptionCaps {
    let mut caps = SubscriptionCaps::unlimited();
    update(&mut caps);
    caps
}

async fn server_with(caps: SubscriptionCaps) -> JazzServer {
    JazzServer::builder()
        .with_schema(nodes_schema())
        .with_subscription_caps(caps)
        .start()
        .await
}

async fn user(server: &JazzServer, name: &str) -> JazzClient {
    JazzClient::connect(server.make_client_context_for_user(nodes_schema(), name))
        .await
        .expect("user connects")
}

async fn backend(server: &JazzServer) -> JazzClient {
    let user_context = server.make_client_context_for_user(nodes_schema(), "unused");
    let context = AppContext {
        jwt_token: None,
        backend_secret: Some(server.backend_secret().to_string()),
        storage: ClientStorage::Memory,
        ..user_context
    };
    JazzClient::connect(context)
        .await
        .expect("backend connects")
}

async fn edge_read(client: &JazzClient, query: Query) -> Result<usize, String> {
    client
        .query(query, Some(DurabilityTier::EdgeServer))
        .await
        .map(|rows| rows.len())
        .map_err(|err| err.to_string())
}

fn assert_refused(outcome: &Result<usize, String>, cap: &str) {
    match outcome {
        Err(message) => assert!(
            message.contains("subscription_over_cap") && message.contains(cap),
            "expected a `subscription_over_cap` refusal naming `{cap}`, got: {message}"
        ),
        Ok(rows) => panic!("expected a refusal naming `{cap}`, the read returned {rows} rows"),
    }
}

fn admitted_count(server: &JazzServer) -> usize {
    server.server_state().runtime.inspect_core_for_test(|core| {
        core.schema_manager()
            .query_manager()
            .server_subscription_count()
    })
}

async fn seed_row(client: &JazzClient) {
    let (_, _, batch) = client
        .insert(
            "nodes",
            HashMap::from([("label".to_string(), Value::Text("root".to_string()))]),
        )
        .expect("insert");
    client
        .wait_for_batch(batch, DurabilityTier::EdgeServer)
        .await
        .expect("row reaches the server");
}

/// Gate 1 — include depth. 7 levels are refused before any compile; 6 are admitted (the app's
/// deepest include today is 4).
#[tokio::test]
async fn an_include_chain_deeper_than_the_cap_is_refused_and_one_at_the_cap_is_admitted() {
    let server = server_with(caps_with(|caps| caps.max_include_depth = 6)).await;
    let alice = user(&server, "alice").await;
    seed_row(&alice).await;

    let refused = edge_read(&alice, children_chain(7)).await;
    assert_refused(&refused, "include depth");

    let admitted = edge_read(&alice, children_chain(6)).await;
    assert!(
        admitted.is_ok(),
        "a chain at the cap must be admitted: {admitted:?}"
    );
    server.shutdown().await;
}

/// Gate 2 — explicit limit.
#[tokio::test]
async fn a_limit_above_the_cap_is_refused_and_one_at_the_cap_is_admitted() {
    let server = server_with(caps_with(|caps| caps.max_query_limit = 1_000)).await;
    let alice = user(&server, "alice").await;
    seed_row(&alice).await;

    let refused = edge_read(&alice, QueryBuilder::new("nodes").limit(1_001).build()).await;
    assert_refused(&refused, "limit");
    let admitted = edge_read(&alice, QueryBuilder::new("nodes").limit(1_000).build()).await;
    assert!(admitted.is_ok(), "{admitted:?}");
    server.shutdown().await;
}

/// Gate 3 — standing registrations per user, across that user's client ids. Three standing
/// subscriptions fill the cap; a fourth registration (a one-shot, so the refusal is
/// observable) is refused, from a second client id of the same user too; unsubscribing one
/// admits the next.
#[tokio::test]
async fn standing_subscriptions_per_user_are_capped_exactly_and_released_on_unsubscribe() {
    let server = server_with(caps_with(|caps| caps.max_subscriptions_per_user = 3)).await;
    let alice = user(&server, "alice").await;
    let alice_phone = user(&server, "alice").await;
    seed_row(&alice).await;

    let mut streams = Vec::new();
    for limit in 1..=3 {
        let stream = alice
            .subscribe(QueryBuilder::new("nodes").limit(limit).build())
            .await
            .expect("standing subscription");
        streams.push(stream);
    }
    support::wait_for(
        Duration::from_secs(10),
        "three registrations admitted",
        || async { (admitted_count(&server) == 3).then_some(()) },
    )
    .await;

    let refused = edge_read(&alice, QueryBuilder::new("nodes").limit(4).build()).await;
    assert_refused(&refused, "standing subscriptions");
    let refused_other_client = edge_read(&alice_phone, QueryBuilder::new("nodes").build()).await;
    assert_refused(&refused_other_client, "standing subscriptions");
    assert_eq!(
        admitted_count(&server),
        3,
        "a refusal leaves no server state behind"
    );

    let released = streams.pop().expect("a stream to release");
    alice
        .unsubscribe(released.handle())
        .await
        .expect("unsubscribe");
    support::wait_for(
        Duration::from_secs(10),
        "the release reaches the server",
        || async { (admitted_count(&server) == 2).then_some(()) },
    )
    .await;
    let admitted = edge_read(&alice_phone, QueryBuilder::new("nodes").build()).await;
    assert!(
        admitted.is_ok(),
        "after a release the next registration is admitted: {admitted:?}"
    );
    server.shutdown().await;
}

/// Gate 3b — the global ceiling. Under `--allow-local-first-auth` a self-signed identity is
/// free, so a per-user cap alone is evadable by minting users; the ceiling is what bounds the
/// server. Two users share a ceiling of 3; the fourth standing subscription is refused
/// whoever asks, and a release by either user makes room for the other.
#[tokio::test]
async fn the_global_ceiling_bounds_standing_subscriptions_across_users() {
    let server = server_with(caps_with(|caps| caps.max_total_subscriptions = 3)).await;
    let alice = user(&server, "alice").await;
    let bob = user(&server, "bob").await;
    seed_row(&alice).await;

    let mut alice_streams = Vec::new();
    for limit in 1..=2 {
        alice_streams.push(
            alice
                .subscribe(QueryBuilder::new("nodes").limit(limit).build())
                .await
                .expect("alice's standing subscription"),
        );
    }
    let bob_stream = bob
        .subscribe(QueryBuilder::new("nodes").build())
        .await
        .expect("bob's standing subscription");
    support::wait_for(
        Duration::from_secs(10),
        "three registrations admitted",
        || async { (admitted_count(&server) == 3).then_some(()) },
    )
    .await;

    let refused = edge_read(&bob, QueryBuilder::new("nodes").limit(7).build()).await;
    assert_refused(&refused, "total standing subscriptions");
    let refused_alice = edge_read(&alice, QueryBuilder::new("nodes").limit(7).build()).await;
    assert_refused(&refused_alice, "total standing subscriptions");
    assert_eq!(
        admitted_count(&server),
        3,
        "a refusal leaves no server state behind"
    );
    // The ceiling is what free identities can fill; the backend is the trusted path with its
    // own cap and must keep working when the ceiling is full, or filling the ceiling takes
    // the product down.
    let backend = backend(&server).await;
    let backend_read = edge_read(&backend, QueryBuilder::new("nodes").build()).await;
    assert!(
        backend_read.is_ok(),
        "the backend is admitted above a full ceiling: {backend_read:?}"
    );

    let released = alice_streams.pop().expect("a stream to release");
    alice
        .unsubscribe(released.handle())
        .await
        .expect("unsubscribe");
    support::wait_for(
        Duration::from_secs(10),
        "the release reaches the server",
        || async { (admitted_count(&server) == 2).then_some(()) },
    )
    .await;
    let admitted = edge_read(&bob, QueryBuilder::new("nodes").build()).await;
    assert!(
        admitted.is_ok(),
        "after any user's release the next registration is admitted: {admitted:?}"
    );
    drop(bob_stream);
    server.shutdown().await;
}

/// Gate 4 — churn. Counting must be exact: 300 one-shot reads, each admitted and released,
/// must not eat a cap of 3. (Against counting on `ClientState.queries`, which keeps every
/// unsubscribed query until the client is reaped, the 4th read would be refused.)
#[tokio::test]
async fn released_one_shot_reads_do_not_count_against_the_cap() {
    let server = server_with(caps_with(|caps| caps.max_subscriptions_per_user = 3)).await;
    let alice = user(&server, "alice").await;
    seed_row(&alice).await;

    for round in 0..300 {
        let outcome = edge_read(&alice, QueryBuilder::new("nodes").build()).await;
        assert!(
            outcome.is_ok(),
            "read {round} must be admitted: {outcome:?}"
        );
    }
    let stream = alice
        .subscribe(QueryBuilder::new("nodes").build())
        .await
        .expect("a standing subscription after the churn");
    support::wait_for(
        Duration::from_secs(10),
        "the standing subscription is admitted",
        || async { (admitted_count(&server) == 1).then_some(()) },
    )
    .await;
    drop(stream);
    server.shutdown().await;
}

/// Gate 5 — registration rate per user per window, refused attempts included, shared by the
/// user's client ids; the backend is not rate limited.
#[tokio::test]
async fn registrations_per_user_per_window_are_capped_and_the_backend_is_not() {
    let server = server_with(caps_with(|caps| {
        caps.max_registrations_per_user_per_window = 3;
        caps.registration_window = Duration::from_secs(600);
    }))
    .await;
    let alice = user(&server, "alice").await;
    let alice_phone = user(&server, "alice").await;
    let backend = backend(&server).await;
    seed_row(&backend).await;

    for round in 0..2 {
        let outcome = edge_read(&alice, QueryBuilder::new("nodes").build()).await;
        assert!(
            outcome.is_ok(),
            "registration {round} admitted: {outcome:?}"
        );
    }
    let third = edge_read(&alice_phone, QueryBuilder::new("nodes").build()).await;
    assert!(
        third.is_ok(),
        "the window is per user, and 3 is within it: {third:?}"
    );
    let fourth = edge_read(&alice_phone, QueryBuilder::new("nodes").build()).await;
    assert_refused(&fourth, "registrations");
    let fifth = edge_read(&alice, QueryBuilder::new("nodes").build()).await;
    assert_refused(&fifth, "registrations");

    for round in 0..8 {
        let outcome = edge_read(&backend, QueryBuilder::new("nodes").build()).await;
        assert!(
            outcome.is_ok(),
            "backend registration {round} admitted: {outcome:?}"
        );
    }
    server.shutdown().await;
}

/// Gate 6 — nothing the app issues comes near a cap: every registration in a burst of
/// app-shaped reads (depth 4, limit 500, thirteen standing subscriptions, ten one-shots) is
/// admitted under the DEFAULT caps.
#[tokio::test]
async fn app_shaped_traffic_is_admitted_under_the_default_caps() {
    let server = server_with(SubscriptionCaps::default()).await;
    let alice = user(&server, "alice").await;
    seed_row(&alice).await;

    let mut streams = Vec::new();
    for limit in 1..=13 {
        streams.push(
            alice
                .subscribe(children_chain(4).limit_for_test(limit))
                .await
                .expect("standing app subscription"),
        );
    }
    for _ in 0..10 {
        let outcome = edge_read(&alice, QueryBuilder::new("nodes").limit(500).build()).await;
        assert!(outcome.is_ok(), "{outcome:?}");
    }
    let deep = edge_read(&alice, children_chain(4)).await;
    assert!(deep.is_ok(), "{deep:?}");
    server.shutdown().await;
}

trait LimitForTest {
    fn limit_for_test(self, limit: usize) -> Query;
}

impl LimitForTest for Query {
    fn limit_for_test(mut self, limit: usize) -> Query {
        self.limit = Some(limit);
        self.refresh_relation_ir()
            .expect("a builder-made query stays valid");
        self
    }
}
