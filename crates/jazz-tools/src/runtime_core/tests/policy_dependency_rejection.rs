//! A write the server rejects is taken back locally (`retract_local_rejected_row`). When
//! the rejected row lived in a table that another table's select policy reads, taking it
//! back withdraws a grant — but the retraction marks only the table it touched, and a
//! subscription filtered by explicit authorization never has the policy's tables in its
//! graph. Nothing re-authorizes it, so the row the rejected grant exposed stays in the
//! subscription's result for as long as nothing else makes it settle.
//!
//! Until 2026-09-15 a leftover `has_pending_local_updates` made most such subscriptions
//! settle on every pass anyway (`query_manager::manager_tests::stuck_local_updates`), which
//! hid this. With that flag cleared, the policy-dependency mark has to reach every path that
//! changes a table — the rejection paths call `mark_subscriptions_dirty_local` directly,
//! not through the batched visibility effects.

use super::*;
use crate::query_manager::relation_ir::{
    ColumnRef, PredicateCmpOp, PredicateExpr, RelExpr, RowIdRef, ValueRef,
};

fn structural_schema() -> Schema {
    SchemaBuilder::new()
        .table(TableSchema::builder("teams").column("name", ColumnType::Text))
        .table(
            TableSchema::builder("user_team_edges")
                .column("user_id", ColumnType::Text)
                .column("team_id", ColumnType::Uuid),
        )
        .build()
}

/// A team is visible when an edge for the session's user points at it; an edge may only be
/// written for the writer's own user.
fn authorization_schema() -> Schema {
    authorization_schema_with_edge_delete(PolicyExpr::eq_session("user_id", vec!["user_id".into()]))
}

fn authorization_schema_with_edge_delete(edge_delete_policy: PolicyExpr) -> Schema {
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
                .policies(
                    TablePolicies::new()
                        .with_select(PolicyExpr::eq_session("user_id", vec!["user_id".into()]))
                        .with_insert(PolicyExpr::eq_session("user_id", vec!["user_id".into()]))
                        .with_delete(edge_delete_policy),
                ),
        )
        .build()
}

fn runtime(app_name: &str) -> TestCore {
    runtime_with_authorization(app_name, authorization_schema())
}

fn runtime_with_authorization(app_name: &str, authorization: Schema) -> TestCore {
    let mut core = create_runtime_with_schema(structural_schema(), app_name);
    core.schema_manager_mut()
        .query_manager_mut()
        .set_authorization_schema(authorization);
    core
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

fn exchange(
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

#[test]
fn a_rejected_grant_through_a_policy_dependency_is_withdrawn_from_the_subscription() {
    let mut client = runtime("rejected-policy-dependency");
    let mut server = runtime("rejected-policy-dependency");
    let client_id = ClientId::new();
    let server_id = ServerId::new();
    // The server knows this connection as mallory while the client writes as alice: alice's
    // own policy admits her edge, mallory's does not. That is a real `Rejected` fate.
    server.add_client(client_id, Some(Session::new("mallory")));
    client.add_server(server_id);
    let alice = Session::new("alice");
    let as_alice = WriteContext::from_session(alice.clone());

    let sub = client
        .schema_manager_mut()
        .query_manager_mut()
        .subscribe_with_session(Query::new("teams"), Some(alice), None)
        .unwrap();
    client.immediate_tick();

    let ((team, _), _) = client
        .insert(
            "teams",
            HashMap::from([("name".to_string(), Value::Text("one".into()))]),
            Some(&as_alice),
        )
        .unwrap();
    // Settle the team write upstream first, so its acceptance is not what re-settles the
    // subscription after the rejection below.
    let _ = exchange(&mut client, &mut server, client_id, server_id);
    assert!(
        team_ids(&mut client, sub).is_empty(),
        "fixture precondition: without an edge alice sees no team"
    );

    client
        .insert(
            "user_team_edges",
            HashMap::from([
                ("user_id".to_string(), Value::Text("alice".into())),
                ("team_id".to_string(), Value::Uuid(team)),
            ]),
            Some(&as_alice),
        )
        .unwrap();
    client.immediate_tick();
    assert_eq!(
        team_ids(&mut client, sub),
        vec![team],
        "fixture precondition: alice's pending edge grants her the team locally"
    );

    let server_outputs = exchange(&mut client, &mut server, client_id, server_id);
    assert!(
        server_outputs.iter().any(|entry| matches!(
            &entry.payload,
            SyncPayload::BatchFate { fate } if fate.is_rejected()
        )),
        "fixture precondition: the server must reject the edge, else this gates nothing"
    );

    assert!(
        team_ids(&mut client, sub).is_empty(),
        "the only edge granting the team was rejected and taken back, but the subscription \
         still shows the team: the retraction marked `user_team_edges` alone, and nothing \
         re-authorized a subscription whose policy reads it"
    );
}

fn edge_values(user: &str, team: ObjectId) -> HashMap<String, Value> {
    HashMap::from([
        ("user_id".to_string(), Value::Text(user.into())),
        ("team_id".to_string(), Value::Uuid(team)),
    ])
}

fn server_subscribed_to(server: &TestCore, table: &str) -> bool {
    server
        .schema_manager()
        .query_manager()
        .server_subscription_telemetry()
        .iter()
        .any(|group| group.table == table && group.count > 0)
}

fn in_server_scope(server: &TestCore, client_id: ClientId, row: ObjectId) -> bool {
    let branch = server.schema_manager().branch_name();
    server
        .schema_manager()
        .query_manager()
        .sync_manager()
        .get_client(client_id)
        .is_some_and(|client| client.is_in_scope(row, &branch))
}

fn edge_ids(core: &mut TestCore) -> Vec<ObjectId> {
    let sub = core
        .schema_manager_mut()
        .query_manager_mut()
        .subscribe_with_session(Query::new("user_team_edges"), None, None)
        .unwrap();
    core.immediate_tick();
    let ids = core
        .schema_manager()
        .query_manager()
        .get_subscription_results(sub)
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    core.schema_manager_mut()
        .query_manager_mut()
        .unsubscribe_with_sync(sub);
    ids
}

/// A team alice wrote while no edge granted it settles denied, and nothing re-settles a
/// subscription whose last pass changed nothing. When the grant then arrives from the server,
/// the policy-dependency mark is the only thing that can re-authorize the subscription.
#[test]
fn a_denied_local_write_is_delivered_when_its_grant_arrives_from_the_server() {
    let mut client = runtime("remote-grant-policy-dependency");
    let mut server = runtime("remote-grant-policy-dependency");
    let client_id = ClientId::new();
    let server_id = ServerId::new();
    server.add_client(client_id, Some(Session::new("alice")));
    client.add_server(server_id);
    let alice = Session::new("alice");

    let sub = client
        .schema_manager_mut()
        .query_manager_mut()
        .subscribe_with_sync(Query::new("teams"), Some(alice.clone()), None)
        .unwrap();
    // Alice also reads her edges, so the server has a reason to send her the one it grants.
    let _edges = client
        .schema_manager_mut()
        .query_manager_mut()
        .subscribe_with_sync(Query::new("user_team_edges"), Some(alice.clone()), None)
        .unwrap();
    client.immediate_tick();
    let ((team, _), _) = client
        .insert(
            "teams",
            HashMap::from([("name".to_string(), Value::Text("one".into()))]),
            Some(&WriteContext::from_session(alice)),
        )
        .unwrap();
    let _ = exchange(&mut client, &mut server, client_id, server_id);
    assert!(
        team_ids(&mut client, sub).is_empty(),
        "fixture precondition: without an edge alice sees no team"
    );
    assert!(
        client
            .schema_manager()
            .query_manager()
            .subscriptions_awaiting_settle()
            .is_empty(),
        "fixture precondition: the subscription is clean, so only a mark can re-settle it"
    );

    let ((edge, _), _) = server
        .insert("user_team_edges", edge_values("alice", team), None)
        .unwrap();
    let _ = exchange(&mut client, &mut server, client_id, server_id);
    let server_has_edges_subscription = server_subscribed_to(&server, "user_team_edges");
    let edge_in_scope = in_server_scope(&server, client_id, edge);
    assert!(
        edge_ids(&mut client).contains(&edge),
        "fixture precondition: alice's edge must reach the client, else this gates nothing \
         (server holds an edges subscription: {server_has_edges_subscription}; edge in alice's \
         server scope: {edge_in_scope})"
    );

    assert_eq!(
        team_ids(&mut client, sub),
        vec![team],
        "alice's edge arrived from the server and grants her the team, but the subscription \
         never re-authorized: the arrival marked `user_team_edges`, which its graph does not read"
    );
}

/// The server keeps a scope per client subscription and re-derives it only when that
/// subscription's graph is dirty. A grant or revoke through a policy dependency dirties nothing
/// the graph reads, so the team never enters alice's scope, and once in, never leaves it.
#[test]
fn a_grant_and_revoke_through_a_policy_dependency_move_the_server_scope() {
    let mut client = runtime("server-scope-policy-dependency");
    let mut server = runtime("server-scope-policy-dependency");
    let client_id = ClientId::new();
    let server_id = ServerId::new();
    server.add_client(client_id, Some(Session::new("alice")));
    client.add_server(server_id);

    client
        .schema_manager_mut()
        .query_manager_mut()
        .subscribe_with_sync(Query::new("teams"), Some(Session::new("alice")), None)
        .unwrap();
    let _ = exchange(&mut client, &mut server, client_id, server_id);
    assert!(
        server_subscribed_to(&server, "teams"),
        "fixture precondition: the server must hold alice's teams subscription, else the scope \
         assertions below gate nothing"
    );

    let ((team, _), _) = server
        .insert(
            "teams",
            HashMap::from([("name".to_string(), Value::Text("one".into()))]),
            None,
        )
        .unwrap();
    let _ = exchange(&mut client, &mut server, client_id, server_id);
    assert!(
        !in_server_scope(&server, client_id, team),
        "fixture precondition: without an edge the team is outside alice's server scope"
    );

    let ((edge, _), _) = server
        .insert("user_team_edges", edge_values("alice", team), None)
        .unwrap();
    let _ = exchange(&mut client, &mut server, client_id, server_id);
    assert!(
        in_server_scope(&server, client_id, team),
        "alice's edge grants her the team, but her server scope never re-authorized: the edge \
         write dirtied only `user_team_edges`, which the teams subscription graph does not read"
    );

    server.delete(edge, None).unwrap();
    let _ = exchange(&mut client, &mut server, client_id, server_id);
    assert!(
        !in_server_scope(&server, client_id, team),
        "the only edge granting the team is gone, but the team stays in alice's server scope"
    );
}

/// A delete the server rejects is put back locally (`restore_local_rejected_delete_row`). When the
/// deleted row is the grant a select policy reads, putting it back grants again, and only the
/// policy-dependency mark can re-authorize a subscription whose graph does not read that table.
#[test]
fn a_rejected_delete_of_a_grant_through_a_policy_dependency_is_restored_to_the_subscription() {
    let mut client = runtime("rejected-delete-policy-dependency");
    // The server's copy of the permissions is stricter than the client's and refuses every edge
    // delete, as it does while a permissions change is still on its way to the client. Alice's
    // own copy admits her delete, so it goes upstream and comes back as a real `Rejected` fate.
    let mut server = runtime_with_authorization(
        "rejected-delete-policy-dependency",
        authorization_schema_with_edge_delete(PolicyExpr::False),
    );
    let client_id = ClientId::new();
    let server_id = ServerId::new();
    server.add_client(client_id, Some(Session::new("alice")));
    client.add_server(server_id);
    let alice = Session::new("alice");

    let sub = client
        .schema_manager_mut()
        .query_manager_mut()
        .subscribe_with_sync(Query::new("teams"), Some(alice.clone()), None)
        .unwrap();
    let _edges = client
        .schema_manager_mut()
        .query_manager_mut()
        .subscribe_with_sync(Query::new("user_team_edges"), Some(alice.clone()), None)
        .unwrap();
    let _ = exchange(&mut client, &mut server, client_id, server_id);

    let ((team, _), _) = server
        .insert(
            "teams",
            HashMap::from([("name".to_string(), Value::Text("one".into()))]),
            None,
        )
        .unwrap();
    let ((edge, _), _) = server
        .insert("user_team_edges", edge_values("alice", team), None)
        .unwrap();
    let _ = exchange(&mut client, &mut server, client_id, server_id);
    assert!(
        edge_ids(&mut client).contains(&edge),
        "fixture precondition: alice's edge must reach the client, else there is nothing to delete"
    );
    assert_eq!(
        team_ids(&mut client, sub),
        vec![team],
        "fixture precondition: the server's edge grants alice the team"
    );

    client
        .delete(edge, Some(&WriteContext::from_session(alice)))
        .unwrap();
    client.immediate_tick();
    assert!(
        team_ids(&mut client, sub).is_empty(),
        "fixture precondition: alice's pending delete withdraws the team locally"
    );

    let server_outputs = exchange(&mut client, &mut server, client_id, server_id);
    assert!(
        server_outputs.iter().any(|entry| matches!(
            &entry.payload,
            SyncPayload::BatchFate { fate } if fate.is_rejected()
        )),
        "fixture precondition: the server must reject the delete, else this gates nothing"
    );
    assert!(
        edge_ids(&mut client).contains(&edge),
        "fixture precondition: the rejected delete puts alice's edge back"
    );

    assert_eq!(
        team_ids(&mut client, sub),
        vec![team],
        "the rejected delete put the granting edge back, but the subscription still hides the \
         team: the restore marked `user_team_edges` alone, and nothing re-authorized a \
         subscription whose policy reads it"
    );
    assert!(
        client
            .schema_manager()
            .query_manager()
            .subscriptions_awaiting_settle()
            .is_empty(),
        "a subscription is still waiting for a settle after the rejection was applied"
    );
}
