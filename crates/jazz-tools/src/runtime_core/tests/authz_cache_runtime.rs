//! Runtime-level differential coverage for the cross-tick authorization verdict cache.
//!
//! A live subscription with a session runs explicit authorization filtering on every
//! settle. In debug builds every cache hit is re-verified against a fresh policy
//! evaluation inside `provenance_row_matches_current_select_policy`, so stale verdicts
//! panic the test rather than silently leak or hide rows. The assertions below
//! additionally pin the visible behavior: revocation and grants through the policy's
//! dependency table must take effect on the very next settle.

use super::*;
use crate::query_manager::relation_ir::{
    ColumnRef, JoinCondition, JoinKind, PredicateCmpOp, PredicateExpr, RelExpr, RowIdRef, ValueRef,
};

/// Structural (runtime) schema: no policies.
fn cache_teams_structural_schema() -> Schema {
    SchemaBuilder::new()
        .table(TableSchema::builder("teams").column("name", ColumnType::Text))
        .table(
            TableSchema::builder("user_team_edges")
                .column("user_id", ColumnType::Text)
                .column("team_id", ColumnType::Uuid),
        )
        .build()
}

/// Authorization schema: a team is visible when an edge for the session's user points
/// at it. Deliberately references ONLY `user_team_edges`, so writes to `teams` leave
/// other teams' verdicts valid — that is what makes cache hits possible at all.
fn cache_teams_auth_schema() -> Schema {
    let team_select_policy = PolicyExpr::ExistsRel {
        rel: RelExpr::Filter {
            input: Box::new(RelExpr::TableScan {
                table: TableName::new("user_team_edges"),
            }),
            predicate: PredicateExpr::And(vec![
                PredicateExpr::Cmp {
                    left: ColumnRef::scoped("user_team_edges", "user_id"),
                    op: PredicateCmpOp::Eq,
                    right: ValueRef::SessionRef(vec!["user_id".into()]),
                },
                PredicateExpr::Cmp {
                    left: ColumnRef::scoped("user_team_edges", "team_id"),
                    op: PredicateCmpOp::Eq,
                    right: ValueRef::RowId(RowIdRef::Outer),
                },
            ]),
        },
    };

    SchemaBuilder::new()
        .table(
            TableSchema::builder("teams")
                .column("name", ColumnType::Text)
                .policies(
                    TablePolicies::new()
                        .with_select(team_select_policy)
                        .with_insert(PolicyExpr::True),
                ),
        )
        .table(
            TableSchema::builder("user_team_edges")
                .column("user_id", ColumnType::Text)
                .column("team_id", ColumnType::Uuid)
                .policies(TablePolicies::new().with_insert(PolicyExpr::True)),
        )
        .build()
}

fn team_ids(core: &mut TestCore, sub_id: QuerySubscriptionId) -> Vec<ObjectId> {
    let mut ids: Vec<ObjectId> = core
        .schema_manager_mut()
        .query_manager_mut()
        .get_subscription_results(sub_id)
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    ids.sort();
    ids
}

fn insert_edge(core: &mut TestCore, user: &str, team_id: ObjectId) -> ObjectId {
    let ((id, _), _) = core
        .insert(
            "user_team_edges",
            HashMap::from([
                ("user_id".to_string(), Value::Text(user.into())),
                ("team_id".to_string(), Value::Uuid(team_id)),
            ]),
            None,
        )
        .unwrap();
    id
}

#[test]
fn authz_cache_serves_hits_and_tracks_policy_dep_writes() {
    let mut core =
        create_runtime_with_schema(cache_teams_structural_schema(), "authz-cache-runtime");
    core.schema_manager_mut()
        .query_manager_mut()
        .set_authorization_schema(cache_teams_auth_schema());

    let sub = core
        .schema_manager_mut()
        .query_manager_mut()
        .subscribe_with_session(Query::new("teams"), Some(Session::new("alice")), None)
        .unwrap();
    core.immediate_tick();

    // Alice can see T1 through her edge.
    let ((team1, _), _) = core
        .insert(
            "teams",
            HashMap::from([("name".to_string(), Value::Text("one".into()))]),
            None,
        )
        .unwrap();
    let edge1 = insert_edge(&mut core, "alice", team1);
    core.immediate_tick();
    assert_eq!(team_ids(&mut core, sub), vec![team1]);

    // A second team without an edge: the settle re-checks T1 (cache hit, parity
    // verified in debug) and evaluates T2 to invisible.
    let hits_before = core
        .schema_manager_mut()
        .query_manager_mut()
        .authz_cache_hit_count();
    let ((team2, _), _) = core
        .insert(
            "teams",
            HashMap::from([("name".to_string(), Value::Text("two".into()))]),
            None,
        )
        .unwrap();
    core.immediate_tick();
    assert_eq!(
        team_ids(&mut core, sub),
        vec![team1].into_iter().chain([]).collect::<Vec<_>>()
    );
    let hits_after = core
        .schema_manager_mut()
        .query_manager_mut()
        .authz_cache_hit_count();
    // The cache is opt-in (off by default since the linsa-v5 verdict-flap incident), so
    // hits accrue only when the test process runs with JAZZ_AUTHZ_CACHE_ENABLE set.
    if std::env::var_os("JAZZ_AUTHZ_CACHE_ENABLE").is_some() {
        assert!(
            hits_after > hits_before,
            "the unchanged team's verdict should have been served from the cache \
             (before {hits_before}, after {hits_after})",
        );
    }

    // Granting through the dependency table must invalidate and grant on the next
    // settle. A cache that missed the user_team_edges dependency would keep T2 hidden
    // — and the debug parity assert would fire before this assertion even runs.
    insert_edge(&mut core, "alice", team2);
    core.immediate_tick();
    let mut expected = vec![team1, team2];
    expected.sort();
    assert_eq!(team_ids(&mut core, sub), expected);

    // Revocation through the dependency table. The subscription's maintained result
    // set only refreshes on a settle, and an edge delete alone does not force one —
    // that is upstream behavior, the same with this cache disabled
    // (cache left at its default: disabled). What the cache must guarantee: at the NEXT settle
    // the dropped edge is reflected — a stale verdict here would both fail this
    // assertion and trip the debug parity assert.
    core.delete(edge1, None).unwrap();
    core.update(
        team2,
        vec![("name".to_string(), Value::Text("two-renamed".into()))],
        None,
    )
    .unwrap();
    core.immediate_tick();
    assert_eq!(team_ids(&mut core, sub), vec![team2]);
}

fn owned_docs_structural_schema() -> Schema {
    SchemaBuilder::new()
        .table(TableSchema::builder("docs").column("owner_id", ColumnType::Text))
        .build()
}

/// Anyone may create a doc; only its owner reads it or edits it, and an edit may hand it to
/// anyone. No policy reads another table.
fn owned_docs_auth_schema() -> Schema {
    let owner = || PolicyExpr::eq_session("owner_id", vec!["user_id".into()]);
    SchemaBuilder::new()
        .table(
            TableSchema::builder("docs")
                .column("owner_id", ColumnType::Text)
                .policies(
                    TablePolicies::new()
                        .with_select(owner())
                        .with_insert(PolicyExpr::True)
                        .with_update(Some(owner()), PolicyExpr::True),
                ),
        )
        .build()
}

fn settle_link(
    client: &mut TestCore,
    server: &mut TestCore,
    client_id: ClientId,
    server_id: ServerId,
) -> Vec<OutboxEntry> {
    let mut server_outputs = Vec::new();
    for _ in 0..10 {
        client.batched_tick();
        pump_client_messages_to_server(client, server, server_id, client_id);
        pump_server_messages_to_clients(
            server,
            &mut [ClientForServer {
                core: &mut *client,
                server_id,
                client_id,
            }],
            &mut server_outputs,
        );
        client.batched_tick();
        client.immediate_tick();
    }
    server_outputs
}

/// The verdict cache drops a changed row's own verdicts only where batched visibility effects
/// are applied. A rejected update is taken back through `clear_local_pending_row_overlay`, which
/// never gets there, so the verdict computed for the pending version outlives that version.
#[test]
fn a_rejected_update_does_not_leave_its_pending_verdict_in_the_cache() {
    crate::query_manager::authz_cache::set_cache_enabled_for_tests(true);
    let runtime = |app_name: &str| {
        let mut core = create_runtime_with_schema(owned_docs_structural_schema(), app_name);
        core.schema_manager_mut()
            .query_manager_mut()
            .set_authorization_schema(owned_docs_auth_schema());
        core
    };
    let mut client = runtime("rejected-update-verdict");
    let mut server = runtime("rejected-update-verdict");
    let client_id = ClientId::new();
    let server_id = ServerId::new();
    // The server knows this connection as mallory: creating a doc needs no owner, handing one
    // over needs to own it. Alice's hand-over is admitted locally and rejected upstream.
    server.add_client(client_id, Some(Session::new("mallory")));
    client.add_server(server_id);
    let alice = Session::new("alice");
    let as_alice = WriteContext::from_session(alice.clone());

    let sub = client
        .schema_manager_mut()
        .query_manager_mut()
        .subscribe_with_session(Query::new("docs"), Some(alice), None)
        .unwrap();
    client.immediate_tick();
    let ((doc, _), _) = client
        .insert(
            "docs",
            HashMap::from([("owner_id".to_string(), Value::Text("alice".into()))]),
            Some(&as_alice),
        )
        .unwrap();
    let _ = settle_link(&mut client, &mut server, client_id, server_id);
    assert_eq!(
        team_ids(&mut client, sub),
        vec![doc],
        "fixture precondition: the doc is accepted and alice sees it"
    );

    client
        .update(
            doc,
            vec![("owner_id".to_string(), Value::Text("bob".into()))],
            Some(&as_alice),
        )
        .unwrap();
    client.immediate_tick();
    assert!(
        team_ids(&mut client, sub).is_empty(),
        "fixture precondition: handed to bob, the pending doc leaves alice's view"
    );

    let server_outputs = settle_link(&mut client, &mut server, client_id, server_id);
    assert!(
        server_outputs.iter().any(|entry| matches!(
            &entry.payload,
            SyncPayload::BatchFate { fate } if fate.is_rejected()
        )),
        "fixture precondition: the server must reject the hand-over, else this gates nothing"
    );

    assert_eq!(
        team_ids(&mut client, sub),
        vec![doc],
        "the hand-over was rejected and the doc is alice's again, but the subscription still \
         hides it: the verdict cached for the pending version outlived that version"
    );
}
