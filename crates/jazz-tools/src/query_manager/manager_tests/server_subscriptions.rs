use super::*;

fn owned_items_schema() -> Schema {
    let mut schema = Schema::new();
    let descriptor = RowDescriptor::new(vec![
        ColumnDescriptor::new("owner_id", ColumnType::Text),
        ColumnDescriptor::new("name", ColumnType::Text),
    ]);
    let policies = TablePolicies::new()
        .with_select(PolicyExpr::eq_session("owner_id", vec!["user_id".into()]));
    schema.insert(
        TableName::new("items"),
        TableSchema::with_policies(descriptor, policies),
    );
    schema
}

/// Confirm the given payloads as the receiver would, so the server's delivery bookkeeping
/// advances. Claims are applied on the receiver's confirmation, not at queue time, so a
/// test that drains the outbox has to say the rows arrived — otherwise the peer looks like
/// one still owed rows and is rightly re-offered them.
fn confirm_delivered(
    qm: &mut crate::query_manager::manager::QueryManager,
    entries: &[crate::sync_manager::OutboxEntry],
) {
    use crate::sync_manager::{Destination, SyncPayload};
    let confirmed: Vec<_> = entries
        .iter()
        .filter_map(|entry| match (&entry.destination, &entry.payload) {
            (Destination::Client(client_id), SyncPayload::RowBatchNeeded { row, .. }) => Some((
                *client_id,
                row.row_id,
                crate::object::BranchName::new(row.branch.as_str()),
                row.batch_id,
            )),
            _ => None,
        })
        .collect();
    qm.sync_manager_mut().confirm_client_deliveries(&confirmed);
}

#[test]
fn server_builds_query_graph_on_subscription() {
    use crate::sync_manager::{ClientId, Destination, InboxEntry, QueryId, Source, SyncPayload};
    use uuid::Uuid;

    let sync_manager = SyncManager::new();
    let schema = test_schema();
    let (mut server_qm, mut storage) = create_query_manager(sync_manager, schema);

    // Server has existing data: 3 users, 2 with score > 50
    let handle1 = server_qm
        .insert(
            &mut storage,
            "users",
            &[Value::Text("Alice".into()), Value::Integer(100)],
        )
        .unwrap();
    let _handle2 = server_qm
        .insert(
            &mut storage,
            "users",
            &[Value::Text("Bob".into()), Value::Integer(30)],
        )
        .unwrap();
    let handle3 = server_qm
        .insert(
            &mut storage,
            "users",
            &[Value::Text("Charlie".into()), Value::Integer(75)],
        )
        .unwrap();
    server_qm.process(&mut storage);

    // Add a client
    let client_id = ClientId(Uuid::new_v7(uuid::Timestamp::now(uuid::NoContext)));
    connect_client(&mut server_qm, &storage, client_id);

    // Client sends QuerySubscription for score > 50
    let query = server_qm
        .query("users")
        .filter_gt("score", Value::Integer(50))
        .build();

    server_qm.sync_manager_mut().push_inbox(InboxEntry {
        source: Source::Client(client_id),
        payload: SyncPayload::QuerySubscription {
            query_id: QueryId(1),
            query: Box::new(query),
            session: None,
            required_tier: None,
            propagation: crate::sync_manager::QueryPropagation::Full,
            policy_context_tables: vec![],
        },
    });

    server_qm.process(&mut storage);

    // Server should send RowBatchNeeded for matching users (Alice, Charlie)
    let outbox = server_qm.sync_manager_mut().take_outbox();

    let row_updates: Vec<_> = outbox
        .iter()
        .filter(|e| matches!(e.destination, Destination::Client(id) if id == client_id))
        .filter_map(|e| match &e.payload {
            SyncPayload::RowBatchNeeded { row, .. } => Some(row.row_id),
            _ => None,
        })
        .collect();

    assert_eq!(
        row_updates.len(),
        2,
        "Should send 2 RowBatchNeeded messages for matching users"
    );

    let sent_ids: std::collections::HashSet<_> = row_updates.into_iter().collect();

    assert!(sent_ids.contains(&handle1.row_id), "Alice should be sent");
    assert!(sent_ids.contains(&handle3.row_id), "Charlie should be sent");
}

#[test]
fn initial_query_settled_is_not_queued_behind_later_query_scope_rows() {
    use crate::sync_manager::{ClientId, Destination, InboxEntry, QueryId, Source, SyncPayload};
    use uuid::Uuid;

    let sync_manager = SyncManager::new();
    let schema = test_schema();
    let (mut server_qm, mut storage) = create_query_manager(sync_manager, schema);

    let alice = server_qm
        .insert(
            &mut storage,
            "users",
            &[Value::Text("Alice".into()), Value::Integer(100)],
        )
        .unwrap();
    let bob = server_qm
        .insert(
            &mut storage,
            "users",
            &[Value::Text("Bob".into()), Value::Integer(30)],
        )
        .unwrap();
    let _charlie = server_qm
        .insert(
            &mut storage,
            "users",
            &[Value::Text("Charlie".into()), Value::Integer(75)],
        )
        .unwrap();
    server_qm.process(&mut storage);

    let client_id = ClientId(Uuid::new_v7(uuid::Timestamp::now(uuid::NoContext)));
    connect_client(&mut server_qm, &storage, client_id);
    server_qm.sync_manager_mut().take_outbox();

    let small_query = server_qm
        .query("users")
        .filter_gt("score", Value::Integer(90))
        .build();
    let broad_query = server_qm
        .query("users")
        .filter_gt("score", Value::Integer(0))
        .build();

    for (query_id, query) in [(QueryId(1), small_query), (QueryId(2), broad_query)] {
        server_qm.sync_manager_mut().push_inbox(InboxEntry {
            source: Source::Client(client_id),
            payload: SyncPayload::QuerySubscription {
                query_id,
                query: Box::new(query),
                session: None,
                required_tier: None,
                propagation: crate::sync_manager::QueryPropagation::Full,
                policy_context_tables: vec![],
            },
        });
    }

    server_qm.process(&mut storage);

    let outbox = server_qm.sync_manager_mut().take_outbox();
    let first_settled_idx = outbox
        .iter()
        .position(|entry| {
            matches!(
                entry,
                crate::sync_manager::OutboxEntry {
                    destination: Destination::Client(id),
                    payload: SyncPayload::QuerySettled { query_id: QueryId(1), .. },
                } if *id == client_id
            )
        })
        .expect("first query should settle");
    let second_query_only_row_idx = outbox
        .iter()
        .position(|entry| {
            matches!(
                entry,
                crate::sync_manager::OutboxEntry {
                    destination: Destination::Client(id),
                    payload: SyncPayload::RowBatchNeeded { row, .. },
                } if *id == client_id && row.row_id == bob.row_id
            )
        })
        .expect("second query should queue its additional rows");

    assert!(
        first_settled_idx < second_query_only_row_idx,
        "query 1 settlement should follow query 1 rows ({}) before query 2 rows ({}); outbox={outbox:?}",
        alice.row_id,
        bob.row_id
    );
}

#[test]
fn server_subscription_does_not_repeat_same_scope_same_tier_settlement() {
    use crate::sync_manager::{ClientId, Destination, InboxEntry, QueryId, Source, SyncPayload};

    let sync_manager = SyncManager::new();
    let schema = test_schema();
    let (mut server_qm, mut storage) = create_query_manager(sync_manager, schema);

    server_qm
        .insert(
            &mut storage,
            "users",
            &[Value::Text("Alice".into()), Value::Integer(100)],
        )
        .unwrap();
    server_qm.process(&mut storage);

    let client_id = ClientId::new();
    connect_client(&mut server_qm, &storage, client_id);
    let _ = server_qm.sync_manager_mut().take_outbox();

    let query = server_qm
        .query("users")
        .filter_gt("score", Value::Integer(50))
        .build();

    server_qm.sync_manager_mut().push_inbox(InboxEntry {
        source: Source::Client(client_id),
        payload: SyncPayload::QuerySubscription {
            query_id: QueryId(1),
            query: Box::new(query),
            session: None,
            required_tier: None,
            propagation: crate::sync_manager::QueryPropagation::Full,
            policy_context_tables: vec![],
        },
    });

    server_qm.process(&mut storage);
    let outbox = server_qm.sync_manager_mut().take_outbox();
    assert!(
        outbox.iter().any(|entry| matches!(
            entry,
            crate::sync_manager::OutboxEntry {
                destination: Destination::Client(id),
                payload: SyncPayload::QuerySettled { query_id: QueryId(1), .. },
            } if *id == client_id
        )),
        "initial settlement should still be emitted"
    );

    server_qm
        .insert(
            &mut storage,
            "users",
            &[Value::Text("Bob".into()), Value::Integer(10)],
        )
        .unwrap();
    server_qm.process(&mut storage);

    let outbox = server_qm.sync_manager_mut().take_outbox();
    assert!(
        outbox.iter().all(|entry| !matches!(
            entry,
            crate::sync_manager::OutboxEntry {
                destination: Destination::Client(id),
                payload: SyncPayload::QuerySettled { query_id: QueryId(1), .. },
            } if *id == client_id
        )),
        "dirty graph passes with unchanged scope and unchanged tier should not resend QuerySettled"
    );
}

#[test]
fn duplicate_server_subscription_does_not_replay_same_scope_same_tier_settlement() {
    use crate::sync_manager::{ClientId, Destination, InboxEntry, QueryId, Source, SyncPayload};

    let sync_manager = SyncManager::new();
    let schema = test_schema();
    let (mut server_qm, mut storage) = create_query_manager(sync_manager, schema);

    server_qm
        .insert(
            &mut storage,
            "users",
            &[Value::Text("Alice".into()), Value::Integer(100)],
        )
        .unwrap();
    server_qm.process(&mut storage);

    let client_id = ClientId::new();
    connect_client(&mut server_qm, &storage, client_id);
    let _ = server_qm.sync_manager_mut().take_outbox();

    let query = server_qm
        .query("users")
        .filter_gt("score", Value::Integer(50))
        .build();

    let mut outbox = Vec::new();
    for _ in 0..2 {
        server_qm.sync_manager_mut().push_inbox(InboxEntry {
            source: Source::Client(client_id),
            payload: SyncPayload::QuerySubscription {
                query_id: QueryId(1),
                query: Box::new(query.clone()),
                session: None,
                required_tier: None,
                propagation: crate::sync_manager::QueryPropagation::Full,
                policy_context_tables: vec![],
            },
        });

        server_qm.process(&mut storage);

        // Confirm between the registrations. A peer with rows still outstanding is owed a
        // re-offer, and the second registration would rightly give it one; this test is
        // about the other case — a peer that has everything and merely re-registers.
        let delivered = server_qm.sync_manager_mut().take_outbox();
        confirm_delivered(&mut server_qm, &delivered);
        outbox.extend(delivered);
    }
    let settled_count = outbox
        .iter()
        .filter(|entry| {
            matches!(
                entry,
                crate::sync_manager::OutboxEntry {
                    destination: Destination::Client(id),
                    payload: SyncPayload::QuerySettled { query_id: QueryId(1), .. },
                } if *id == client_id
            )
        })
        .count();

    assert_eq!(
        settled_count, 1,
        "re-registering an equivalent active subscription should not replay an unchanged settlement"
    );
}

/// A server at `server_tier` holding one user, a client that subscribed to it under query
/// id 1 requiring `required_tier`, was answered, and confirmed what it was sent. Returns
/// what the server sent for that first subscription.
fn a_subscription_answered_and_confirmed(
    server_tier: Option<DurabilityTier>,
    required_tier: Option<DurabilityTier>,
) -> (
    QueryManager,
    MemoryStorage,
    crate::sync_manager::ClientId,
    Vec<crate::sync_manager::OutboxEntry>,
) {
    let sync_manager = match server_tier {
        Some(tier) => SyncManager::new().with_durability_tier(tier),
        None => SyncManager::new(),
    };
    let (mut server_qm, mut storage) = create_query_manager(sync_manager, test_schema());
    server_qm
        .insert(
            &mut storage,
            "users",
            &[Value::Text("Alice".into()), Value::Integer(100)],
        )
        .unwrap();
    server_qm.process(&mut storage);

    let client_id = crate::sync_manager::ClientId::new();
    connect_client(&mut server_qm, &storage, client_id);
    let _ = server_qm.sync_manager_mut().take_outbox();

    let first = send_the_subscription(&mut server_qm, &mut storage, client_id, required_tier);
    (server_qm, storage, client_id, first)
}

/// The client sends its subscription under query id 1; the server works through it; what
/// it sent back is confirmed and returned.
fn send_the_subscription(
    server_qm: &mut QueryManager,
    storage: &mut MemoryStorage,
    client_id: crate::sync_manager::ClientId,
    required_tier: Option<DurabilityTier>,
) -> Vec<crate::sync_manager::OutboxEntry> {
    use crate::sync_manager::{InboxEntry, QueryId, Source, SyncPayload};

    let query = server_qm
        .query("users")
        .filter_gt("score", Value::Integer(50))
        .build();
    server_qm.sync_manager_mut().push_inbox(InboxEntry {
        source: Source::Client(client_id),
        payload: SyncPayload::QuerySubscription {
            query_id: QueryId(1),
            query: Box::new(query),
            session: None,
            required_tier,
            propagation: crate::sync_manager::QueryPropagation::Full,
            policy_context_tables: vec![],
        },
    });
    server_qm.process(storage);
    let sent = server_qm.sync_manager_mut().take_outbox();
    confirm_delivered(server_qm, &sent);
    sent
}

fn settled_tiers(
    sent: &[crate::sync_manager::OutboxEntry],
    client_id: crate::sync_manager::ClientId,
) -> Vec<(DurabilityTier, usize)> {
    use crate::sync_manager::{Destination, QueryId, SyncPayload};
    sent.iter()
        .filter_map(|entry| match (&entry.destination, &entry.payload) {
            (
                Destination::Client(id),
                SyncPayload::QuerySettled {
                    query_id: QueryId(1),
                    tier,
                    scope,
                    ..
                },
            ) if *id == client_id => Some((*tier, scope.len())),
            _ => None,
        })
        .collect()
}

fn rows_sent(sent: &[crate::sync_manager::OutboxEntry]) -> usize {
    use crate::sync_manager::SyncPayload;
    sent.iter()
        .filter(|entry| {
            matches!(
                entry.payload,
                SyncPayload::RowBatchNeeded { .. } | SyncPayload::RowBatchCreated { .. }
            )
        })
        .count()
}

/// An app that is closed and opened keeps its client id and numbers its queries from the
/// start, so it sends the subscription the server holds, under the id the server holds it
/// under — and it is a new engine, which was told nothing.
#[test]
fn a_subscription_sent_again_on_a_new_connection_is_answered_from_the_scope_held() {
    let tier = DurabilityTier::EdgeServer;
    let (mut server_qm, mut storage, client_id, first) =
        a_subscription_answered_and_confirmed(Some(tier), Some(tier));
    assert_eq!(
        settled_tiers(&first, client_id),
        vec![(tier, 1)],
        "the first subscription is not answered, so nothing below measures a second"
    );

    server_qm
        .sync_manager_mut()
        .note_client_connected(client_id);
    let again = send_the_subscription(&mut server_qm, &mut storage, client_id, Some(tier));

    assert_eq!(
        settled_tiers(&again, client_id),
        vec![(tier, 1)],
        "a read that waits for this tier is given nothing to stop waiting on"
    );
    assert_eq!(
        rows_sent(&again),
        0,
        "the peer confirmed the row; the answer is the scope, not the rows again"
    );
}

#[test]
fn a_subscription_sent_twice_on_one_connection_is_answered_once() {
    let tier = DurabilityTier::EdgeServer;
    let (mut server_qm, mut storage, client_id, _first) =
        a_subscription_answered_and_confirmed(Some(tier), Some(tier));

    server_qm
        .sync_manager_mut()
        .note_client_connected(client_id);
    let _ = send_the_subscription(&mut server_qm, &mut storage, client_id, Some(tier));
    let third = send_the_subscription(&mut server_qm, &mut storage, client_id, Some(tier));

    assert_eq!(
        settled_tiers(&third, client_id),
        Vec::new(),
        "the engine on this connection was told; telling it again is a scope to decode \
         for nothing"
    );
}

/// A server below the tier a subscription waits for still hands over its scope once, as
/// the first snapshot the reader has. A new connection is owed that snapshot as well.
#[test]
fn a_new_connection_is_given_the_scope_of_a_server_below_the_tier_asked_for() {
    let required = DurabilityTier::GlobalServer;
    let (mut server_qm, mut storage, client_id, first) =
        a_subscription_answered_and_confirmed(Some(DurabilityTier::EdgeServer), Some(required));
    assert_eq!(
        settled_tiers(&first, client_id),
        vec![(DurabilityTier::EdgeServer, 1)]
    );

    server_qm
        .sync_manager_mut()
        .note_client_connected(client_id);
    let again = send_the_subscription(&mut server_qm, &mut storage, client_id, Some(required));

    assert_eq!(
        settled_tiers(&again, client_id),
        vec![(DurabilityTier::EdgeServer, 1)]
    );
}

/// The client's subscription under query id 1, read from the socket that opened
/// `connection`.
fn send_the_subscription_on(
    server_qm: &mut QueryManager,
    storage: &mut MemoryStorage,
    client_id: crate::sync_manager::ClientId,
    required_tier: Option<DurabilityTier>,
    connection: u64,
) -> Vec<crate::sync_manager::OutboxEntry> {
    server_qm.sync_manager_mut().note_subscription_asked_on(
        client_id,
        crate::sync_manager::QueryId(1),
        connection,
    );
    send_the_subscription(server_qm, storage, client_id, required_tier)
}

/// An app is closed with a frame of its still on the way through the server, and opened:
/// the new engine's connection is counted before the old engine's frame is taken up. The
/// frame is the old engine's, which was told; the answer a new connection is owed is owed
/// to what comes on that connection.
#[test]
fn a_subscription_on_its_way_when_a_new_connection_opened_does_not_take_its_answer() {
    let tier = DurabilityTier::EdgeServer;
    let (mut server_qm, mut storage, client_id, _first) =
        a_subscription_answered_and_confirmed(Some(tier), Some(tier));
    let old = server_qm.sync_manager().client_connection(client_id);
    server_qm
        .sync_manager_mut()
        .note_client_connected(client_id);
    let new = server_qm.sync_manager().client_connection(client_id);

    let late = send_the_subscription_on(&mut server_qm, &mut storage, client_id, Some(tier), old);
    assert_eq!(
        settled_tiers(&late, client_id),
        Vec::new(),
        "the engine that sent it was told on its own connection, and is gone"
    );

    let asked = send_the_subscription_on(&mut server_qm, &mut storage, client_id, Some(tier), new);
    assert_eq!(
        settled_tiers(&asked, client_id),
        vec![(tier, 1)],
        "the new engine's subscription found its answer taken by a frame of the old one, \
         and its read waits for ever"
    );
}

/// A client may have more than one connection open. Each is a reader that was told
/// nothing, whichever of them the subscription first came from.
#[test]
fn each_connection_a_client_has_open_is_answered_the_first_time_it_asks() {
    let tier = DurabilityTier::EdgeServer;
    let (mut server_qm, mut storage) =
        create_query_manager(SyncManager::new().with_durability_tier(tier), test_schema());
    server_qm
        .insert(
            &mut storage,
            "users",
            &[Value::Text("Alice".into()), Value::Integer(100)],
        )
        .unwrap();
    server_qm.process(&mut storage);
    let client_id = crate::sync_manager::ClientId::new();
    connect_client(&mut server_qm, &storage, client_id);
    let first = server_qm.sync_manager().client_connection(client_id);
    server_qm
        .sync_manager_mut()
        .note_client_connected(client_id);
    let second = server_qm.sync_manager().client_connection(client_id);
    let _ = server_qm.sync_manager_mut().take_outbox();

    let to_first =
        send_the_subscription_on(&mut server_qm, &mut storage, client_id, Some(tier), first);
    assert_eq!(settled_tiers(&to_first, client_id), vec![(tier, 1)]);

    let to_second =
        send_the_subscription_on(&mut server_qm, &mut storage, client_id, Some(tier), second);
    assert_eq!(
        settled_tiers(&to_second, client_id),
        vec![(tier, 1)],
        "the subscription was made on the client's older connection, and the newer one \
         asking for it was taken to have been told"
    );
}

/// A client that comes back replays every subscription it holds, and each is answered with
/// its scope. That is bounded per pass like any other replay: the answers of one client do
/// not hold the pass for everyone.
#[test]
fn a_client_that_comes_back_with_many_subscriptions_is_answered_a_pass_at_a_time() {
    use crate::sync_manager::{InboxEntry, QueryId, Source, SyncPayload};
    const SUBSCRIPTIONS: u64 = 40;
    const PER_PASS: usize = 32;

    let tier = DurabilityTier::EdgeServer;
    let (mut server_qm, mut storage) =
        create_query_manager(SyncManager::new().with_durability_tier(tier), test_schema());
    server_qm
        .insert(
            &mut storage,
            "users",
            &[Value::Text("Alice".into()), Value::Integer(100)],
        )
        .unwrap();
    server_qm.process(&mut storage);
    let client_id = crate::sync_manager::ClientId::new();
    connect_client(&mut server_qm, &storage, client_id);
    let _ = server_qm.sync_manager_mut().take_outbox();

    // Every subscription the client holds, sent at once; then pass after pass until the
    // server has nothing more to say. Returns how many it answered in each pass.
    let send_them_all = |server_qm: &mut QueryManager, storage: &mut MemoryStorage| {
        let query = server_qm
            .query("users")
            .filter_gt("score", Value::Integer(50))
            .build();
        for id in 1..=SUBSCRIPTIONS {
            server_qm.sync_manager_mut().push_inbox(InboxEntry {
                source: Source::Client(client_id),
                payload: SyncPayload::QuerySubscription {
                    query_id: QueryId(id),
                    query: Box::new(query.clone()),
                    session: None,
                    required_tier: Some(tier),
                    propagation: crate::sync_manager::QueryPropagation::Full,
                    policy_context_tables: vec![],
                },
            });
        }
        let mut answered = Vec::new();
        loop {
            server_qm.process(storage);
            let sent = server_qm.sync_manager_mut().take_outbox();
            if sent.is_empty() {
                return answered;
            }
            confirm_delivered(server_qm, &sent);
            answered.push(
                sent.iter()
                    .filter(|entry| matches!(entry.payload, SyncPayload::QuerySettled { .. }))
                    .count(),
            );
        }
    };

    let first = send_them_all(&mut server_qm, &mut storage);
    assert_eq!(
        first.iter().sum::<usize>(),
        SUBSCRIPTIONS as usize,
        "not every subscription was answered the first time, so nothing below measures a \
         replay"
    );

    server_qm
        .sync_manager_mut()
        .note_client_connected(client_id);
    let replay = send_them_all(&mut server_qm, &mut storage);

    assert_eq!(replay, vec![PER_PASS, SUBSCRIPTIONS as usize - PER_PASS]);
}

#[test]
fn pending_duplicate_server_subscription_is_compiled_once() {
    use crate::sync_manager::{ClientId, Destination, InboxEntry, QueryId, Source, SyncPayload};

    let sync_manager = SyncManager::new();
    let schema = test_schema();
    let (mut server_qm, mut storage) = create_query_manager(sync_manager, schema);

    server_qm
        .insert(
            &mut storage,
            "users",
            &[Value::Text("Alice".into()), Value::Integer(100)],
        )
        .unwrap();
    server_qm.process(&mut storage);

    let client_id = ClientId::new();
    connect_client(&mut server_qm, &storage, client_id);
    let _ = server_qm.sync_manager_mut().take_outbox();

    let query = server_qm
        .query("users")
        .filter_gt("score", Value::Integer(50))
        .build();

    for _ in 0..2 {
        server_qm.sync_manager_mut().push_inbox(InboxEntry {
            source: Source::Client(client_id),
            payload: SyncPayload::QuerySubscription {
                query_id: QueryId(1),
                query: Box::new(query.clone()),
                session: None,
                required_tier: None,
                propagation: crate::sync_manager::QueryPropagation::Full,
                policy_context_tables: vec![],
            },
        });
    }

    server_qm.process(&mut storage);

    let outbox = server_qm.sync_manager_mut().take_outbox();
    let settled_count = outbox
        .iter()
        .filter(|entry| {
            matches!(
                entry,
                crate::sync_manager::OutboxEntry {
                    destination: Destination::Client(id),
                    payload: SyncPayload::QuerySettled { query_id: QueryId(1), .. },
                } if *id == client_id
            )
        })
        .count();

    assert_eq!(
        settled_count, 1,
        "duplicate pending registrations for the same client/query should compile and settle once"
    );
}

#[test]
fn server_subscription_reads_visible_region_after_legacy_commit_history_is_removed() {
    use crate::sync_manager::{ClientId, Destination, InboxEntry, QueryId, Source, SyncPayload};
    use uuid::Uuid;

    let schema = test_schema();
    let (mut writer_qm, mut storage) = create_query_manager(SyncManager::new(), schema.clone());
    let _branch = get_branch(&writer_qm);

    let handle = writer_qm
        .insert(
            &mut storage,
            "users",
            &[Value::Text("Alice".into()), Value::Integer(75)],
        )
        .unwrap();
    writer_qm.process(&mut storage);

    let (mut server_qm, _) = create_query_manager(SyncManager::new(), schema);
    let client_id = ClientId(Uuid::new_v7(uuid::Timestamp::now(uuid::NoContext)));
    connect_client(&mut server_qm, &storage, client_id);

    let query = server_qm
        .query("users")
        .filter_gt("score", Value::Integer(50))
        .build();

    server_qm.sync_manager_mut().push_inbox(InboxEntry {
        source: Source::Client(client_id),
        payload: SyncPayload::QuerySubscription {
            query_id: QueryId(1),
            query: Box::new(query),
            session: None,
            required_tier: None,
            propagation: crate::sync_manager::QueryPropagation::Full,
            policy_context_tables: vec![],
        },
    });

    server_qm.process(&mut storage);

    let outbox = server_qm.sync_manager_mut().take_outbox();
    let row_updates: Vec<_> = outbox
        .iter()
        .filter(|entry| matches!(entry.destination, Destination::Client(id) if id == client_id))
        .filter_map(|entry| match &entry.payload {
            SyncPayload::RowBatchNeeded { row, .. } => Some(row.row_id),
            _ => None,
        })
        .collect();

    assert_eq!(
        row_updates.len(),
        1,
        "server subscription should settle from visible rows without legacy object-backed storage"
    );
    assert_eq!(row_updates[0], handle.row_id);
}

#[test]
fn server_authorizes_subscription_sync_scope_without_rechecking_output_scope() {
    use crate::query_manager::session::Session;
    use crate::sync_manager::{
        ClientId, InboxEntry, QueryId, QueryPropagation, Source, SyncPayload,
    };

    let mut server_qm = QueryManager::new(SyncManager::new());
    server_qm.set_current_schema(owned_items_schema(), "dev", "main");
    let inner = seeded_memory_storage(&server_qm.schema_context().current_schema);
    let mut storage = CountingCatalogueUpsertsStorage::with_inner(inner);

    for index in 0..4 {
        server_qm
            .insert(
                &mut storage,
                "items",
                &[
                    Value::Text("alice".to_string()),
                    Value::Text(format!("Item {index}")),
                ],
            )
            .expect("insert item");
    }
    server_qm.process(&mut storage);

    let client_id = ClientId::new();
    connect_client(&mut server_qm, &storage, client_id);
    let _ = server_qm.sync_manager_mut().take_outbox();
    storage.reset_visible_query_loads();

    let query = server_qm.query("items").limit(4).build();
    server_qm.sync_manager_mut().push_inbox(InboxEntry {
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

    server_qm.process(&mut storage);

    assert!(
        storage.visible_query_loads() <= 12,
        "server subscription should not re-run an extra output-scope authorization pass, got {} visible row loads",
        storage.visible_query_loads()
    );
}

#[test]
fn authorization_schema_context_is_reused_for_matching_env_and_user_branch() {
    let schema = owned_items_schema();
    let schema_hash = crate::query_manager::types::SchemaHash::compute(&schema);
    let mut server_qm = QueryManager::new(SyncManager::new());
    server_qm.set_current_schema(schema.clone(), "dev", "main");
    server_qm.set_authorization_schema(schema.clone());
    server_qm.set_known_schemas(std::sync::Arc::new(HashMap::from([(schema_hash, schema)])));

    let (_, first_context) = server_qm
        .authorization_schema_for_context("dev", "main")
        .expect("authorization context should be available");
    let (_, second_context) = server_qm
        .authorization_schema_for_context("dev", "main")
        .expect("authorization context should be cached");

    assert!(
        std::sync::Arc::ptr_eq(&first_context, &second_context),
        "authorization schema context should be reused for repeated subscription settlement"
    );
    assert_eq!(server_qm.authorization_context_cache.len(), 1);

    server_qm.set_known_schemas(std::sync::Arc::new(HashMap::new()));
    assert!(
        server_qm.authorization_context_cache.is_empty(),
        "known schema changes must invalidate cached authorization contexts"
    );
}

#[test]
fn local_stale_recompile_failure_drops_subscription_and_reports_failure() {
    let sync_manager = SyncManager::new();
    let schema = test_schema();
    let (mut qm, mut storage) = create_query_manager(sync_manager, schema);

    let sub_id = qm.subscribe(qm.query("users").build()).unwrap();
    qm.process(&mut storage);
    let _ = qm.take_updates();

    {
        let sub = qm
            .subscriptions
            .get_mut(&sub_id)
            .expect("subscription should exist");
        sub.query = QueryBuilder::new("no_such_table").build();
        sub.needs_recompile = true;
    }

    qm.process(&mut storage);

    assert!(
        !qm.subscriptions.contains_key(&sub_id),
        "failed stale recompile should drop the local subscription"
    );

    let failures = qm.take_failed_subscriptions();
    assert_eq!(
        failures.len(),
        1,
        "expected exactly one reported local subscription failure"
    );
    assert_eq!(failures[0].subscription_id, sub_id);
    assert!(
        failures[0].reason.contains("no_such_table"),
        "failure reason should include compile context: {}",
        failures[0].reason
    );
}

#[test]
fn server_sends_error_for_uncompilable_query_subscription() {
    use crate::sync_manager::{
        ClientId, Destination, InboxEntry, QueryId, Source, SyncError, SyncPayload,
    };
    use uuid::Uuid;

    let sync_manager = SyncManager::new();
    let schema = test_schema();
    let (mut server_qm, mut storage) = create_query_manager(sync_manager, schema);

    let client_id = ClientId(Uuid::new_v7(uuid::Timestamp::now(uuid::NoContext)));
    connect_client(&mut server_qm, &storage, client_id);

    // Query references a table that does not exist in schema.
    let invalid_query = QueryBuilder::new("no_such_table").build();
    server_qm.sync_manager_mut().push_inbox(InboxEntry {
        source: Source::Client(client_id),
        payload: SyncPayload::QuerySubscription {
            query_id: QueryId(42),
            query: Box::new(invalid_query),
            session: None,
            required_tier: None,
            propagation: crate::sync_manager::QueryPropagation::Full,
            policy_context_tables: vec![],
        },
    });

    server_qm.process(&mut storage);

    let outbox = server_qm.sync_manager_mut().take_outbox();
    let (code, reason) = outbox
        .iter()
        .find_map(|entry| match (&entry.destination, &entry.payload) {
            (
                Destination::Client(id),
                SyncPayload::Error(SyncError::QuerySubscriptionRejected {
                    query_id,
                    code,
                    reason,
                }),
            ) if *id == client_id && *query_id == QueryId(42) => {
                Some((code.clone(), reason.clone()))
            }
            _ => None,
        })
        .expect("Server should send an error payload when query subscription compilation fails");
    assert_eq!(code, "query_compilation_failed");
    assert!(
        reason.contains("query_id 42"),
        "error reason should include query id context: {reason}"
    );
    assert!(
        reason.contains("no_such_table"),
        "error reason should include compile error context: {reason}"
    );
}

#[test]
fn server_stale_recompile_failure_drops_subscription_and_notifies_client() {
    use crate::sync_manager::{
        ClientId, Destination, InboxEntry, QueryId, ServerId, Source, SyncError, SyncPayload,
    };
    use uuid::Uuid;

    let sync_manager = SyncManager::new();
    let schema = test_schema();
    let (mut server_qm, mut storage) = create_query_manager(sync_manager, schema);

    let upstream_id = ServerId(Uuid::new_v7(uuid::Timestamp::now(uuid::NoContext)));
    connect_server(&mut server_qm, &storage, upstream_id);
    let _ = server_qm.sync_manager_mut().take_outbox();

    let client_id = ClientId(Uuid::new_v7(uuid::Timestamp::now(uuid::NoContext)));
    connect_client(&mut server_qm, &storage, client_id);

    let valid_query = server_qm.query("users").build();
    server_qm.sync_manager_mut().push_inbox(InboxEntry {
        source: Source::Client(client_id),
        payload: SyncPayload::QuerySubscription {
            query_id: QueryId(7),
            query: Box::new(valid_query),
            session: None,
            required_tier: None,
            propagation: crate::sync_manager::QueryPropagation::Full,
            policy_context_tables: vec![],
        },
    });
    server_qm.process(&mut storage);
    let _ = server_qm.sync_manager_mut().take_outbox();

    {
        let sub = server_qm
            .server_subscriptions
            .get_mut(&(client_id, QueryId(7)))
            .expect("server subscription should exist");
        sub.query = QueryBuilder::new("no_such_table").build();
        sub.needs_recompile = true;
    }

    server_qm.process(&mut storage);

    assert!(
        !server_qm
            .server_subscriptions
            .contains_key(&(client_id, QueryId(7))),
        "failed stale recompile should drop the server subscription"
    );
    assert!(
        !server_qm
            .sync_manager()
            .get_client(client_id)
            .expect("client should still exist")
            .queries
            .contains_key(&QueryId(7)),
        "client query scope should be cleared after fail-fast drop"
    );

    let outbox = server_qm.sync_manager_mut().take_outbox();
    let (rejection_code, rejection_reason) = outbox
        .iter()
        .find_map(|entry| match (&entry.destination, &entry.payload) {
            (
                Destination::Client(id),
                SyncPayload::Error(SyncError::QuerySubscriptionRejected {
                    query_id,
                    code,
                    reason,
                }),
            ) if *id == client_id && *query_id == QueryId(7) => {
                Some((code.clone(), reason.clone()))
            }
            _ => None,
        })
        .expect("client should receive QuerySubscriptionRejected on stale recompile failure");
    assert_eq!(rejection_code, "query_recompile_failed");
    assert!(
        rejection_reason.contains("query recompilation failed for query_id 7"),
        "rejection should include query id context: {rejection_reason}"
    );
    assert!(
        rejection_reason.contains("no_such_table"),
        "rejection should include compile error context: {rejection_reason}"
    );

    assert!(
        outbox.iter().any(|entry| matches!(
            (&entry.destination, &entry.payload),
            (
                Destination::Server(id),
                SyncPayload::QueryUnsubscription { query_id }
            ) if *id == upstream_id && *query_id == QueryId(7)
        )),
        "stale recompile failure should forward QueryUnsubscription upstream"
    );
}

#[test]
fn server_pushes_new_matches() {
    use crate::sync_manager::{ClientId, Destination, InboxEntry, QueryId, Source, SyncPayload};
    use uuid::Uuid;

    let sync_manager = SyncManager::new();
    let schema = test_schema();
    let (mut server_qm, mut storage) = create_query_manager(sync_manager, schema);

    // Server has 1 user initially
    let _handle1 = server_qm
        .insert(
            &mut storage,
            "users",
            &[Value::Text("Alice".into()), Value::Integer(100)],
        )
        .unwrap();
    server_qm.process(&mut storage);

    // Add client and subscribe
    let client_id = ClientId(Uuid::new_v7(uuid::Timestamp::now(uuid::NoContext)));
    connect_client(&mut server_qm, &storage, client_id);

    let query = server_qm
        .query("users")
        .filter_gt("score", Value::Integer(50))
        .build();

    server_qm.sync_manager_mut().push_inbox(InboxEntry {
        source: Source::Client(client_id),
        payload: SyncPayload::QuerySubscription {
            query_id: QueryId(1),
            query: Box::new(query),
            session: None,
            required_tier: None,
            propagation: crate::sync_manager::QueryPropagation::Full,
            policy_context_tables: vec![],
        },
    });

    server_qm.process(&mut storage);

    // Clear initial outbox, confirming it as the receiver would.
    let initial = server_qm.sync_manager_mut().take_outbox();
    confirm_delivered(&mut server_qm, &initial);

    // Insert new matching user
    let handle2 = server_qm
        .insert(
            &mut storage,
            "users",
            &[Value::Text("Charlie".into()), Value::Integer(75)],
        )
        .unwrap();
    server_qm.process(&mut storage);

    // Should send RowBatchNeeded for new matching user
    let outbox = server_qm.sync_manager_mut().take_outbox();

    let row_updates: Vec<_> = outbox
        .iter()
        .filter(|e| matches!(e.destination, Destination::Client(id) if id == client_id))
        .filter_map(|e| match &e.payload {
            SyncPayload::RowBatchNeeded { row, .. } => Some(row.row_id),
            _ => None,
        })
        .collect();

    assert_eq!(
        row_updates.len(),
        1,
        "Should send 1 RowBatchNeeded for new matching user"
    );

    assert_eq!(
        row_updates[0], handle2.row_id,
        "Should send Charlie's ObjectId"
    );
}

#[test]
fn server_subscription_telemetry_tracks_grouping_and_unsubscribe_lifecycle() {
    use crate::sync_manager::{
        ClientId, InboxEntry, QueryId, QueryPropagation, Source, SyncPayload,
    };

    let sync_manager = SyncManager::new();
    let schema = test_schema();
    let (mut server_qm, mut storage) = create_query_manager(sync_manager, schema);

    let repeated_query = server_qm.query("users").build();
    let repeated_query_json = serde_json::to_string(&repeated_query).unwrap();
    let filtered_query = server_qm
        .query("users")
        .filter_eq("name", Value::Text("Alice".into()))
        .build();

    let client_a = ClientId::new();
    let client_b = ClientId::new();
    let client_c = ClientId::new();
    for client_id in [client_a, client_b, client_c] {
        connect_client(&mut server_qm, &storage, client_id);
    }

    for (client_id, query_id, query, propagation) in [
        (
            client_a,
            QueryId(1),
            repeated_query.clone(),
            QueryPropagation::Full,
        ),
        (
            client_b,
            QueryId(2),
            repeated_query.clone(),
            QueryPropagation::Full,
        ),
        (
            client_c,
            QueryId(3),
            repeated_query.clone(),
            QueryPropagation::LocalOnly,
        ),
        (
            client_c,
            QueryId(4),
            filtered_query.clone(),
            QueryPropagation::Full,
        ),
    ] {
        server_qm.sync_manager_mut().push_inbox(InboxEntry {
            source: Source::Client(client_id),
            payload: SyncPayload::QuerySubscription {
                query_id,
                query: Box::new(query),
                session: None,
                required_tier: None,
                propagation,
                policy_context_tables: vec![],
            },
        });
    }

    server_qm.process(&mut storage);

    let telemetry = server_qm.server_subscription_telemetry();
    assert_eq!(telemetry.len(), 3);
    assert!(telemetry.iter().any(|group| {
        group.count == 2 && group.propagation == QueryPropagation::Full && group.table == "users"
    }));
    assert!(
        telemetry
            .iter()
            .any(|group| { group.count == 1 && group.propagation == QueryPropagation::LocalOnly })
    );
    assert!(
        telemetry
            .iter()
            .any(|group| { group.count == 1 && group.query.contains("\"name\"") })
    );

    server_qm.sync_manager_mut().push_inbox(InboxEntry {
        source: Source::Client(client_b),
        payload: SyncPayload::QueryUnsubscription {
            query_id: QueryId(2),
        },
    });
    server_qm.process(&mut storage);

    let telemetry_after_unsubscribe = server_qm.server_subscription_telemetry();
    assert!(telemetry_after_unsubscribe.iter().any(|group| {
        group.count == 1
            && group.propagation == QueryPropagation::Full
            && group.query == repeated_query_json
    }));
}

#[test]
fn server_does_not_push_non_matching() {
    use crate::sync_manager::{ClientId, Destination, InboxEntry, QueryId, Source, SyncPayload};
    use uuid::Uuid;

    let sync_manager = SyncManager::new();
    let schema = test_schema();
    let (mut server_qm, mut storage) = create_query_manager(sync_manager, schema);

    // Add client and subscribe to score > 50
    let client_id = ClientId(Uuid::new_v7(uuid::Timestamp::now(uuid::NoContext)));
    connect_client(&mut server_qm, &storage, client_id);

    let query = server_qm
        .query("users")
        .filter_gt("score", Value::Integer(50))
        .build();

    server_qm.sync_manager_mut().push_inbox(InboxEntry {
        source: Source::Client(client_id),
        payload: SyncPayload::QuerySubscription {
            query_id: QueryId(1),
            query: Box::new(query),
            session: None,
            required_tier: None,
            propagation: crate::sync_manager::QueryPropagation::Full,
            policy_context_tables: vec![],
        },
    });

    server_qm.process(&mut storage);
    let _ = server_qm.sync_manager_mut().take_outbox();

    // Insert non-matching user (score = 30)
    let _handle = server_qm
        .insert(
            &mut storage,
            "users",
            &[Value::Text("Bob".into()), Value::Integer(30)],
        )
        .unwrap();
    server_qm.process(&mut storage);

    // Should NOT send RowBatchNeeded for non-matching user
    let outbox = server_qm.sync_manager_mut().take_outbox();

    let row_updates: Vec<_> = outbox
        .iter()
        .filter(|e| matches!(e.destination, Destination::Client(id) if id == client_id))
        .filter(|e| matches!(e.payload, SyncPayload::RowBatchNeeded { .. }))
        .collect();

    assert_eq!(
        row_updates.len(),
        0,
        "Should NOT send RowBatchNeeded for non-matching user"
    );
}

/// A peer whose transport dropped and came back must be re-offered what it missed.
///
/// This is the shape a phone produces and the one no other test reaches. A brief network gap
/// does not tear down the client's subscription — the transport reconnects underneath it and
/// replays the SAME query id. The server then finds an equivalent, already-settled
/// subscription and answers from the cached scope without re-deriving anything.
/// Re-derivation is the only place that sets `force_resend`, so a row the peer never
/// confirmed is otherwise never offered again.
///
/// Every existing offline test reconnects by building a fresh client, which mints a NEW
/// query id; the server has no prior state for that key, re-derives, and force-resends
/// everything as a side effect. They are green for a reason that does not apply in the field.
#[test]
fn a_transport_reconnect_re_offers_a_row_the_peer_never_confirmed() {
    use crate::sync_manager::{ClientId, Destination, InboxEntry, QueryId, Source, SyncPayload};

    let sync_manager = SyncManager::new();
    let schema = test_schema();
    let (mut server_qm, mut storage) = create_query_manager(sync_manager, schema);

    server_qm
        .insert(
            &mut storage,
            "users",
            &[Value::Text("Alice".into()), Value::Integer(100)],
        )
        .unwrap();
    server_qm.process(&mut storage);

    let client_id = ClientId::new();
    connect_client(&mut server_qm, &storage, client_id);
    let _ = server_qm.sync_manager_mut().take_outbox();

    let query = server_qm
        .query("users")
        .filter_gt("score", Value::Integer(50))
        .build();
    let subscribe = |qm: &mut crate::query_manager::manager::QueryManager| {
        qm.sync_manager_mut().push_inbox(InboxEntry {
            source: Source::Client(client_id),
            payload: SyncPayload::QuerySubscription {
                query_id: QueryId(1),
                query: Box::new(query.clone()),
                session: None,
                required_tier: None,
                propagation: crate::sync_manager::QueryPropagation::Full,
                policy_context_tables: vec![],
            },
        });
    };

    subscribe(&mut server_qm);
    server_qm.process(&mut storage);
    let initial = server_qm.sync_manager_mut().take_outbox();
    confirm_delivered(&mut server_qm, &initial);

    // The gap: a message arrives while the peer cannot receive it. Draining the outbox
    // without confirming is what a dead socket does to the payload.
    let missed = server_qm
        .insert(
            &mut storage,
            "users",
            &[Value::Text("sent while away".into()), Value::Integer(150)],
        )
        .unwrap();
    server_qm.process(&mut storage);
    let dropped = server_qm.sync_manager_mut().take_outbox();
    assert!(
        dropped.iter().any(|entry| matches!(
            entry,
            crate::sync_manager::OutboxEntry {
                destination: Destination::Client(id),
                payload: SyncPayload::RowBatchNeeded { row, .. },
            } if *id == client_id && row.row_id == missed.row_id
        )),
        "precondition: the row should have been offered to the subscribed peer once"
    );
    drop(dropped);

    // The peer is back. Nothing is signalled by hand: the server still believes the old
    // socket is alive — a three-second drop tells TCP nothing — so no disconnect was ever
    // recorded. What IS true is that the peer never confirmed the row.
    subscribe(&mut server_qm);
    server_qm.process(&mut storage);

    let after = server_qm.sync_manager_mut().take_outbox();
    assert!(
        after.iter().any(|entry| matches!(
            entry,
            crate::sync_manager::OutboxEntry {
                destination: Destination::Client(id),
                payload: SyncPayload::RowBatchNeeded { row, .. },
            } if *id == client_id && row.row_id == missed.row_id
        )),
        "the peer reconnected and the row it missed was never offered again — the server \
         recognised the replayed subscription as equivalent and already settled, so nothing \
         re-derived its scope and nothing set force_resend. This is the message that never \
         arrives after a brief network drop."
    );
}

/// The re-offer stops once the peer is caught up.
///
/// If the trigger stayed true after confirmation, every resubscribe — and a phone
/// resubscribes on every foreground — would re-send. This pins the other half.
#[test]
fn the_re_offer_stops_once_the_peer_is_caught_up() {
    use crate::sync_manager::{ClientId, Destination, InboxEntry, QueryId, Source, SyncPayload};

    let sync_manager = SyncManager::new();
    let schema = test_schema();
    let (mut server_qm, mut storage) = create_query_manager(sync_manager, schema);

    let client_id = ClientId::new();
    connect_client(&mut server_qm, &storage, client_id);
    let _ = server_qm.sync_manager_mut().take_outbox();

    let query = server_qm
        .query("users")
        .filter_gt("score", Value::Integer(50))
        .build();
    let subscribe = |qm: &mut crate::query_manager::manager::QueryManager| {
        qm.sync_manager_mut().push_inbox(InboxEntry {
            source: Source::Client(client_id),
            payload: SyncPayload::QuerySubscription {
                query_id: QueryId(1),
                query: Box::new(query.clone()),
                session: None,
                required_tier: None,
                propagation: crate::sync_manager::QueryPropagation::Full,
                policy_context_tables: vec![],
            },
        });
    };

    subscribe(&mut server_qm);
    server_qm.process(&mut storage);

    server_qm
        .insert(
            &mut storage,
            "users",
            &[Value::Text("delivered".into()), Value::Integer(150)],
        )
        .unwrap();
    server_qm.process(&mut storage);
    let delivered = server_qm.sync_manager_mut().take_outbox();
    confirm_delivered(&mut server_qm, &delivered);

    subscribe(&mut server_qm);
    server_qm.process(&mut storage);

    let after = server_qm.sync_manager_mut().take_outbox();
    let re_offered = after
        .iter()
        .filter(|entry| {
            matches!(
                entry,
                crate::sync_manager::OutboxEntry {
                    destination: Destination::Client(id),
                    payload: SyncPayload::RowBatchNeeded { .. },
                } if *id == client_id
            )
        })
        .count();
    assert_eq!(
        re_offered, 0,
        "a peer that confirmed everything was re-offered {re_offered} rows on a plain \
         resubscribe — the trigger never clears, so every foreground re-sends"
    );
}

/// The re-offer carries what the peer is missing, not everything it can see.
#[test]
fn the_re_offer_carries_only_what_the_peer_is_missing() {
    use crate::sync_manager::{ClientId, Destination, InboxEntry, QueryId, Source, SyncPayload};

    let sync_manager = SyncManager::new();
    let schema = test_schema();
    let (mut server_qm, mut storage) = create_query_manager(sync_manager, schema);

    let client_id = ClientId::new();
    connect_client(&mut server_qm, &storage, client_id);
    let _ = server_qm.sync_manager_mut().take_outbox();

    let query = server_qm
        .query("users")
        .filter_gt("score", Value::Integer(50))
        .build();
    let subscribe = |qm: &mut crate::query_manager::manager::QueryManager| {
        qm.sync_manager_mut().push_inbox(InboxEntry {
            source: Source::Client(client_id),
            payload: SyncPayload::QuerySubscription {
                query_id: QueryId(1),
                query: Box::new(query.clone()),
                session: None,
                required_tier: None,
                propagation: crate::sync_manager::QueryPropagation::Full,
                policy_context_tables: vec![],
            },
        });
    };

    subscribe(&mut server_qm);
    server_qm.process(&mut storage);

    let held = server_qm
        .insert(
            &mut storage,
            "users",
            &[Value::Text("already held".into()), Value::Integer(150)],
        )
        .unwrap();
    server_qm.process(&mut storage);
    let delivered = server_qm.sync_manager_mut().take_outbox();
    confirm_delivered(&mut server_qm, &delivered);

    let missed = server_qm
        .insert(
            &mut storage,
            "users",
            &[Value::Text("sent while away".into()), Value::Integer(160)],
        )
        .unwrap();
    server_qm.process(&mut storage);
    let _dropped = server_qm.sync_manager_mut().take_outbox();

    subscribe(&mut server_qm);
    server_qm.process(&mut storage);

    let after = server_qm.sync_manager_mut().take_outbox();
    let offered: Vec<_> = after
        .iter()
        .filter_map(|entry| match (&entry.destination, &entry.payload) {
            (Destination::Client(id), SyncPayload::RowBatchNeeded { row, .. })
                if *id == client_id =>
            {
                Some(row.row_id)
            }
            _ => None,
        })
        .collect();

    assert!(
        offered.contains(&missed.row_id),
        "the unconfirmed row was not re-offered"
    );
    assert!(
        !offered.contains(&held.row_id),
        "a row the peer already confirmed was re-sent as part of the recovery — the \
         re-offer ships the whole scope instead of what is missing, so one outstanding row \
         costs a full retransmission on every resubscribe"
    );
}

/// A batch a later batch supersedes must stop being owed.
///
/// A re-offer can only ship the CURRENT row, so an unconfirmed batch that a newer one has
/// replaced can never be confirmed — nothing will send it again. If the entry survives that,
/// the peer counts as owed rows forever: every registration re-derives, and the entry leaks
/// until the client is reaped, which never happens while it stays connected.
#[test]
fn a_superseded_unconfirmed_batch_stops_being_owed() {
    use crate::sync_manager::{ClientId, Destination, InboxEntry, QueryId, Source, SyncPayload};

    let sync_manager = SyncManager::new();
    let schema = test_schema();
    let (mut server_qm, mut storage) = create_query_manager(sync_manager, schema);

    let client_id = ClientId::new();
    connect_client(&mut server_qm, &storage, client_id);
    let _ = server_qm.sync_manager_mut().take_outbox();

    let query = server_qm
        .query("users")
        .filter_gt("score", Value::Integer(50))
        .build();
    let subscribe = |qm: &mut crate::query_manager::manager::QueryManager| {
        qm.sync_manager_mut().push_inbox(InboxEntry {
            source: Source::Client(client_id),
            payload: SyncPayload::QuerySubscription {
                query_id: QueryId(1),
                query: Box::new(query.clone()),
                session: None,
                required_tier: None,
                propagation: crate::sync_manager::QueryPropagation::Full,
                policy_context_tables: vec![],
            },
        });
    };

    subscribe(&mut server_qm);
    server_qm.process(&mut storage);

    let row = server_qm
        .insert(
            &mut storage,
            "users",
            &[Value::Text("first".into()), Value::Integer(150)],
        )
        .unwrap();
    server_qm.process(&mut storage);
    let _dropped = server_qm.sync_manager_mut().take_outbox();

    // Edited while the peer is still away: the first batch can never be sent again.
    server_qm
        .update(
            &mut storage,
            row.row_id,
            &[Value::Text("second".into()), Value::Integer(160)],
        )
        .unwrap();
    server_qm.process(&mut storage);
    let _ = server_qm.sync_manager_mut().take_outbox();

    // The peer returns, is handed the current row, and confirms it.
    subscribe(&mut server_qm);
    server_qm.process(&mut storage);
    let delivered = server_qm.sync_manager_mut().take_outbox();
    assert!(
        delivered.iter().any(|entry| matches!(
            entry,
            crate::sync_manager::OutboxEntry {
                destination: Destination::Client(id),
                payload: SyncPayload::RowBatchNeeded { .. },
            } if *id == client_id
        )),
        "precondition: the returning peer should have been re-offered the row"
    );
    confirm_delivered(&mut server_qm, &delivered);

    subscribe(&mut server_qm);
    server_qm.process(&mut storage);

    let after = server_qm.sync_manager_mut().take_outbox();
    let re_offered = after
        .iter()
        .filter(|entry| {
            matches!(
                entry,
                crate::sync_manager::OutboxEntry {
                    destination: Destination::Client(id),
                    payload: SyncPayload::RowBatchNeeded { .. },
                } if *id == client_id
            )
        })
        .count();
    assert_eq!(
        re_offered, 0,
        "the caught-up peer was re-offered {re_offered} rows"
    );
    assert!(
        !server_qm
            .sync_manager()
            .client_has_undelivered_payloads(client_id),
        "the peer is caught up but is still recorded as owed rows — the entry outlives \
         every chance to confirm it, so it leaks until the client is reaped, and a \
         connected client is never reaped"
    );
}

fn push_subscription(
    server_qm: &mut QueryManager,
    client_id: crate::sync_manager::ClientId,
    query_id: crate::sync_manager::QueryId,
    query: &crate::query_manager::query::Query,
) {
    use crate::sync_manager::{InboxEntry, QueryPropagation, Source, SyncPayload};
    server_qm.sync_manager_mut().push_inbox(InboxEntry {
        source: Source::Client(client_id),
        payload: SyncPayload::QuerySubscription {
            query_id,
            query: Box::new(query.clone()),
            session: None,
            required_tier: None,
            propagation: QueryPropagation::Full,
            policy_context_tables: vec![],
        },
    });
}

fn push_unsubscription(
    server_qm: &mut QueryManager,
    client_id: crate::sync_manager::ClientId,
    query_id: crate::sync_manager::QueryId,
) {
    use crate::sync_manager::{InboxEntry, Source, SyncPayload};
    server_qm.sync_manager_mut().push_inbox(InboxEntry {
        source: Source::Client(client_id),
        payload: SyncPayload::QueryUnsubscription { query_id },
    });
}

fn live_server_subscriptions(server_qm: &QueryManager) -> usize {
    server_qm
        .server_subscription_telemetry()
        .iter()
        .map(|group| group.count)
        .sum()
}

/// A one-shot read that the client answers from its own store sends its subscription and
/// its unsubscription in the same flush, so both reach the server inside one pass. The pass
/// drains unsubscriptions before subscriptions, so the unsubscription finds nothing to remove,
/// the queued subscription is registered afterwards, and it lives until the client
/// disconnects: every later settle pass walks it, and every write to a table it reads
/// re-settles it.
#[test]
fn a_subscription_withdrawn_in_the_same_pass_is_never_registered() {
    use crate::sync_manager::{ClientId, QueryId};

    let (mut server_qm, mut storage) = create_query_manager(SyncManager::new(), test_schema());
    let client_id = ClientId::new();
    connect_client(&mut server_qm, &storage, client_id);
    let query = server_qm.query("users").build();

    push_subscription(&mut server_qm, client_id, QueryId(1), &query);
    push_unsubscription(&mut server_qm, client_id, QueryId(1));
    server_qm.process(&mut storage);

    assert_eq!(
        live_server_subscriptions(&server_qm),
        0,
        "the client subscribed and unsubscribed before the server's pass, yet the server \
         registered the subscription and will carry it until the client disconnects"
    );
}

/// The reverse order within one pass must keep the subscription. A single client never sends
/// it (query ids are not reused), but a hub forwards its downstream clients' ids under its own
/// client id, so a later subscription can share the key upstream. Green on the stock core as
/// well: it pins that a withdrawal cancels at arrival, not when the pass runs.
#[test]
fn a_resubscription_after_the_unsubscription_in_the_same_pass_stays_registered() {
    use crate::sync_manager::{ClientId, QueryId};

    let (mut server_qm, mut storage) = create_query_manager(SyncManager::new(), test_schema());
    let client_id = ClientId::new();
    connect_client(&mut server_qm, &storage, client_id);
    let query = server_qm.query("users").build();

    push_subscription(&mut server_qm, client_id, QueryId(1), &query);
    server_qm.process(&mut storage);
    assert_eq!(
        live_server_subscriptions(&server_qm),
        1,
        "fixture: registered"
    );

    push_unsubscription(&mut server_qm, client_id, QueryId(1));
    push_subscription(&mut server_qm, client_id, QueryId(1), &query);
    server_qm.process(&mut storage);

    assert_eq!(
        live_server_subscriptions(&server_qm),
        1,
        "the subscription that arrived after the unsubscription was dropped"
    );
}

/// Subscribe, withdraw and subscribe again inside one pass: the last word is a subscription.
/// Green on the stock core as well; it would go red if the cancel ran when the pass does.
#[test]
fn the_last_of_subscribe_unsubscribe_subscribe_in_one_pass_wins() {
    use crate::sync_manager::{ClientId, QueryId};

    let (mut server_qm, mut storage) = create_query_manager(SyncManager::new(), test_schema());
    let client_id = ClientId::new();
    connect_client(&mut server_qm, &storage, client_id);
    let query = server_qm.query("users").build();

    push_subscription(&mut server_qm, client_id, QueryId(1), &query);
    push_unsubscription(&mut server_qm, client_id, QueryId(1));
    push_subscription(&mut server_qm, client_id, QueryId(1), &query);
    server_qm.process(&mut storage);

    assert_eq!(live_server_subscriptions(&server_qm), 1);
}

/// A withdrawal only cancels its own client's queued subscription, never another client's
/// subscription under the same query id.
#[test]
fn a_withdrawal_does_not_cancel_another_clients_queued_subscription() {
    use crate::sync_manager::{ClientId, QueryId};

    let (mut server_qm, mut storage) = create_query_manager(SyncManager::new(), test_schema());
    let alice = ClientId::new();
    let bob = ClientId::new();
    connect_client(&mut server_qm, &storage, alice);
    connect_client(&mut server_qm, &storage, bob);
    let query = server_qm.query("users").build();

    push_subscription(&mut server_qm, alice, QueryId(1), &query);
    push_subscription(&mut server_qm, bob, QueryId(1), &query);
    push_unsubscription(&mut server_qm, alice, QueryId(1));
    server_qm.process(&mut storage);

    assert_eq!(live_server_subscriptions(&server_qm), 1);
}

/// Every settle pass walks every server subscription. One whose graph is clean and already
/// settled has nothing to do, but the pass used to build its branch-schema map (a String and a
/// HashMap entry per live schema) before finding that out, and threw it away. With hundreds of
/// live subscriptions that idle work was a large share of the sync server's CPU.
#[test]
fn a_settle_pass_over_clean_subscriptions_builds_no_branch_schema_maps() {
    use crate::query_manager::server_queries::BRANCH_SCHEMA_MAPS_BUILT;
    use crate::sync_manager::{ClientId, QueryId};

    let (mut server_qm, mut storage) = create_query_manager(SyncManager::new(), test_schema());
    let client_id = ClientId::new();
    connect_client(&mut server_qm, &storage, client_id);
    let query = server_qm.query("users").build();

    for id in 1..=50u64 {
        push_subscription(&mut server_qm, client_id, QueryId(id), &query);
    }
    // A pass stops admitting registrations once the outbox holds a full initial replay; run
    // passes until all are in.
    for _ in 0..20 {
        server_qm.process(&mut storage);
        if live_server_subscriptions(&server_qm) == 50 {
            break;
        }
    }
    server_qm.process(&mut storage);
    assert_eq!(
        live_server_subscriptions(&server_qm),
        50,
        "fixture: registered"
    );
    assert!(
        server_qm
            .server_subscriptions
            .values()
            .all(|sub| sub.settled_once && !sub.graph.has_dirty_nodes() && !sub.needs_recompile),
        "fixture: every subscription must be clean and settled, else this measures real work"
    );

    let before = BRANCH_SCHEMA_MAPS_BUILT.with(|built| built.get());
    server_qm.process(&mut storage);
    let built = BRANCH_SCHEMA_MAPS_BUILT.with(|built| built.get()) - before;

    assert_eq!(
        built, 0,
        "a settle pass with nothing to settle built {built} branch-schema maps for 50 clean \
         subscriptions"
    );
}

/// A subscription that did not fit into its pass waits in the queue for a later one, and the
/// client can withdraw it meanwhile. The next pass drains unsubscriptions first, so unless the
/// withdrawal reaches the queue the subscription is registered after it and never removed.
#[test]
fn a_subscription_withdrawn_while_it_waits_for_a_later_pass_is_never_registered() {
    use crate::sync_manager::{ClientId, QueryId};

    let (mut server_qm, mut storage) = create_query_manager(SyncManager::new(), test_schema());
    let client_id = ClientId::new();
    connect_client(&mut server_qm, &storage, client_id);
    let query = server_qm.query("users").build();

    for id in 1..=40u64 {
        push_subscription(&mut server_qm, client_id, QueryId(id), &query);
    }
    server_qm.process(&mut storage);
    assert!(
        !server_qm
            .server_subscriptions
            .contains_key(&(client_id, QueryId(40))),
        "fixture: the last subscription must still be waiting after the first pass"
    );

    push_unsubscription(&mut server_qm, client_id, QueryId(40));
    for _ in 0..30 {
        server_qm.process(&mut storage);
    }

    assert!(
        !server_qm
            .server_subscriptions
            .contains_key(&(client_id, QueryId(40))),
        "the client withdrew a subscription while it waited in the queue, and a later pass \
         registered it anyway"
    );
    assert_eq!(
        live_server_subscriptions(&server_qm),
        39,
        "every other waiting subscription must still be registered"
    );
}

/// A client that reconnects replays its live subscriptions. If it withdraws one before the
/// server's pass, the replay must not bring it back: the withdrawal has to take both the
/// registration and the replayed subscription still in the queue.
#[test]
fn a_replayed_subscription_withdrawn_in_the_same_pass_leaves_nothing_registered() {
    use crate::sync_manager::{ClientId, QueryId};

    let (mut server_qm, mut storage) = create_query_manager(SyncManager::new(), test_schema());
    let client_id = ClientId::new();
    connect_client(&mut server_qm, &storage, client_id);
    let query = server_qm.query("users").build();

    push_subscription(&mut server_qm, client_id, QueryId(1), &query);
    server_qm.process(&mut storage);
    assert_eq!(
        live_server_subscriptions(&server_qm),
        1,
        "fixture: registered"
    );

    push_subscription(&mut server_qm, client_id, QueryId(1), &query);
    push_unsubscription(&mut server_qm, client_id, QueryId(1));
    server_qm.process(&mut storage);

    assert_eq!(
        live_server_subscriptions(&server_qm),
        0,
        "the withdrawal removed the registration, and the replay queued before it registered \
         the subscription again"
    );
}

/// A hub forwards a downstream subscription upstream when it registers it. One withdrawn while
/// still queued was never forwarded, so its withdrawal must not be forwarded either: upstream
/// keys are the hub's client id plus the downstream query id, ids repeat across the hub's
/// downstream clients, and a stray withdrawal removes whichever live subscription shares the id.
#[test]
fn a_hub_sends_nothing_upstream_for_a_subscription_withdrawn_before_registration() {
    use crate::sync_manager::{ClientId, Destination, QueryId, ServerId, SyncPayload};

    let (mut hub, mut storage) = create_query_manager(SyncManager::new(), test_schema());
    let upstream_id = ServerId::new();
    let client_id = ClientId::new();
    connect_server(&mut hub, &storage, upstream_id);
    connect_client(&mut hub, &storage, client_id);
    let query = hub.query("users").build();
    hub.process(&mut storage);
    let _ = hub.sync_manager_mut().take_outbox();

    push_subscription(&mut hub, client_id, QueryId(42), &query);
    push_unsubscription(&mut hub, client_id, QueryId(42));
    hub.process(&mut storage);

    let upstream: Vec<_> = hub
        .sync_manager_mut()
        .take_outbox()
        .into_iter()
        .filter(|entry| entry.destination == Destination::Server(upstream_id))
        .filter_map(|entry| match entry.payload {
            SyncPayload::QuerySubscription { query_id, .. } => Some(("subscription", query_id)),
            SyncPayload::QueryUnsubscription { query_id } => Some(("unsubscription", query_id)),
            _ => None,
        })
        .collect();

    assert!(
        upstream.is_empty(),
        "the hub never registered the subscription, yet sent upstream: {upstream:?}"
    );
}

fn subscribe_page_as(
    server_qm: &mut QueryManager,
    storage: &mut CountingCatalogueUpsertsStorage,
    client_id: crate::sync_manager::ClientId,
    user: &str,
    offset: usize,
    page: usize,
) {
    let query = server_qm
        .query("items")
        .order_by("name")
        .offset(offset)
        .limit(page)
        .build();
    subscribe_query_as(server_qm, storage, client_id, user, query);
}

fn subscribe_query_as(
    server_qm: &mut QueryManager,
    storage: &mut CountingCatalogueUpsertsStorage,
    client_id: crate::sync_manager::ClientId,
    user: &str,
    query: Query,
) {
    use crate::query_manager::session::Session;
    use crate::sync_manager::{QueryId, QueryPropagation};

    server_qm.sync_manager_mut().push_inbox(InboxEntry {
        source: Source::Client(client_id),
        payload: SyncPayload::QuerySubscription {
            query_id: QueryId(1),
            query: Box::new(query),
            session: Some(Session::new(user)),
            required_tier: None,
            propagation: QueryPropagation::Full,
            policy_context_tables: vec![],
        },
    });
    server_qm.process(storage);
    let outbox = server_qm.sync_manager_mut().take_outbox();
    confirm_delivered(server_qm, &outbox);
    server_qm.process(storage);
    let _ = server_qm.sync_manager_mut().take_outbox();
}

fn insert_item(
    server_qm: &mut QueryManager,
    storage: &mut CountingCatalogueUpsertsStorage,
    owner: &str,
    name: &str,
) -> ObjectId {
    server_qm
        .insert(
            storage,
            "items",
            &[
                Value::Text(owner.to_string()),
                Value::Text(name.to_string()),
            ],
        )
        .expect("insert item")
        .row_id
}

fn server_page_scope(server_qm: &QueryManager) -> std::collections::HashSet<ObjectId> {
    server_qm
        .server_subscriptions
        .values()
        .flat_map(|sub| sub.last_scope.iter().map(|(id, _)| *id))
        .collect()
}

/// A page subscription (`order_by … limit n`) replays only its ordered prefix, so the server
/// needs the read verdicts of the first `offset + limit` readable rows and no others. A write
/// that lands outside the page must not re-authorize every row the query's filters match:
/// on linsa-v22 that made one message written into a 100k-message chat cost ~0.7 s of server
/// CPU per live page subscription over it (profiled 2026-09-30).
#[test]
fn a_write_outside_a_page_authorizes_only_the_page() {
    use crate::sync_manager::ClientId;

    const ROWS: usize = 300;
    const PAGE: usize = 5;

    let mut server_qm = QueryManager::new(SyncManager::new());
    server_qm.set_current_schema(owned_items_schema(), "dev", "main");
    let inner = seeded_memory_storage(&server_qm.schema_context().current_schema);
    let mut storage = CountingCatalogueUpsertsStorage::with_inner(inner);

    let ids: Vec<ObjectId> = (0..ROWS)
        .map(|index| {
            insert_item(
                &mut server_qm,
                &mut storage,
                "alice",
                &format!("Item {index:03}"),
            )
        })
        .collect();
    server_qm.process(&mut storage);

    let client_id = ClientId::new();
    connect_client(&mut server_qm, &storage, client_id);
    let _ = server_qm.sync_manager_mut().take_outbox();
    subscribe_page_as(&mut server_qm, &mut storage, client_id, "alice", 0, PAGE);

    let page: std::collections::HashSet<ObjectId> = ids[..PAGE].iter().copied().collect();
    assert_eq!(
        server_page_scope(&server_qm),
        page,
        "fixture: the page is the first {PAGE} rows"
    );

    storage.reset_visible_query_loads();
    insert_item(&mut server_qm, &mut storage, "alice", "Item 999");
    server_qm.process(&mut storage);
    let loads = storage.visible_query_loads();

    assert_eq!(
        server_page_scope(&server_qm),
        page,
        "a write outside the page leaves the page as it was"
    );
    assert!(
        loads <= 4 * PAGE,
        "a write outside a {PAGE}-row page loaded {loads} rows (of {ROWS} matching) to re-authorize it"
    );
}

/// Rows the session may not read never count toward a page: when the head of the ordering
/// belongs to someone else, the page is still the first `limit` rows the session CAN read,
/// and a readable row written ahead of them pushes the last one out.
#[test]
fn a_page_prefix_counts_only_rows_the_session_may_read() {
    use crate::sync_manager::ClientId;

    const PAGE: usize = 5;

    let mut server_qm = QueryManager::new(SyncManager::new());
    server_qm.set_current_schema(owned_items_schema(), "dev", "main");
    let inner = seeded_memory_storage(&server_qm.schema_context().current_schema);
    let mut storage = CountingCatalogueUpsertsStorage::with_inner(inner);

    let mut alice_ids = Vec::new();
    for index in 0..40 {
        let owner = if index % 2 == 0 { "bob" } else { "alice" };
        let id = insert_item(
            &mut server_qm,
            &mut storage,
            owner,
            &format!("Item {index:03}"),
        );
        if owner == "alice" {
            alice_ids.push(id);
        }
    }
    server_qm.process(&mut storage);

    let client_id = ClientId::new();
    connect_client(&mut server_qm, &storage, client_id);
    let _ = server_qm.sync_manager_mut().take_outbox();
    subscribe_page_as(&mut server_qm, &mut storage, client_id, "alice", 0, PAGE);

    let page: std::collections::HashSet<ObjectId> = alice_ids[..PAGE].iter().copied().collect();
    assert_eq!(
        server_page_scope(&server_qm),
        page,
        "the page holds the first {PAGE} rows alice may read, skipping bob's"
    );

    insert_item(&mut server_qm, &mut storage, "bob", "Item 000a");
    server_qm.process(&mut storage);
    assert_eq!(
        server_page_scope(&server_qm),
        page,
        "a denied row written at the head of the ordering does not enter or shift the page"
    );

    let head = insert_item(&mut server_qm, &mut storage, "alice", "Item 000b");
    server_qm.process(&mut storage);
    let shifted: std::collections::HashSet<ObjectId> = std::iter::once(head)
        .chain(alice_ids[..PAGE - 1].iter().copied())
        .collect();
    assert_eq!(
        server_page_scope(&server_qm),
        shifted,
        "a readable row written at the head enters the page and pushes its last row out"
    );
}

/// xorshift64* — deterministic, dependency-free.
struct PagePrng(u64);

impl PagePrng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }
}

/// Differential for a page's authorized sync scope. A seeded stream of inserts, renames
/// (moves within the ordering), owner flips (visibility changes) and deletes runs against a
/// model of the intended semantics: the scope of an `order_by(name) offset o limit n` page is
/// the first `o + n` rows the session may read, in (name, id) order — the ordered prefix the
/// client needs to replay the window, with rows it may not read never counted.
#[test]
fn page_scope_matches_the_first_readable_rows_under_random_writes() {
    use crate::sync_manager::ClientId;
    use std::collections::{BTreeMap, HashSet};

    const STEPS: usize = 300;
    const NAMES: usize = 40;

    for seed in [1u64, 7, 42, 1337, 9001, 65_537] {
        let mut rng = PagePrng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let offset = rng.below(4);
        let page = 1 + rng.below(8);

        let mut server_qm = QueryManager::new(SyncManager::new());
        server_qm.set_current_schema(owned_items_schema(), "dev", "main");
        let inner = seeded_memory_storage(&server_qm.schema_context().current_schema);
        let mut storage = CountingCatalogueUpsertsStorage::with_inner(inner);

        let mut model: BTreeMap<ObjectId, (&'static str, String)> = BTreeMap::new();
        let owners = ["alice", "bob"];
        for _ in 0..30 {
            let owner = owners[rng.below(2)];
            let name = format!("Item {:02}", rng.below(NAMES));
            let id = insert_item(&mut server_qm, &mut storage, owner, &name);
            model.insert(id, (owner, name));
        }
        server_qm.process(&mut storage);

        let client_id = ClientId::new();
        connect_client(&mut server_qm, &storage, client_id);
        let _ = server_qm.sync_manager_mut().take_outbox();
        subscribe_page_as(
            &mut server_qm,
            &mut storage,
            client_id,
            "alice",
            offset,
            page,
        );

        let expected = |model: &BTreeMap<ObjectId, (&'static str, String)>| -> HashSet<ObjectId> {
            let mut readable: Vec<(&String, ObjectId)> = model
                .iter()
                .filter(|(_, (owner, _))| *owner == "alice")
                .map(|(id, (_, name))| (name, *id))
                .collect();
            readable.sort();
            readable
                .into_iter()
                .take(offset + page)
                .map(|(_, id)| id)
                .collect()
        };

        for step in 0..STEPS {
            let live: Vec<ObjectId> = model.keys().copied().collect();
            let roll = rng.below(100);
            let op = if roll < 40 || live.is_empty() {
                let owner = owners[rng.below(2)];
                let name = format!("Item {:02}", rng.below(NAMES));
                let id = insert_item(&mut server_qm, &mut storage, owner, &name);
                model.insert(id, (owner, name.clone()));
                format!("insert {owner} {name}")
            } else if roll < 60 {
                let id = live[rng.below(live.len())];
                let owner = model[&id].0;
                let name = format!("Item {:02}", rng.below(NAMES));
                server_qm
                    .update(
                        &mut storage,
                        id,
                        &[Value::Text(owner.to_string()), Value::Text(name.clone())],
                    )
                    .expect("rename");
                model.insert(id, (owner, name.clone()));
                format!("rename {id} -> {name}")
            } else if roll < 75 {
                let id = live[rng.below(live.len())];
                let (owner, name) = model[&id].clone();
                let flipped = if owner == "alice" { "bob" } else { "alice" };
                server_qm
                    .update(
                        &mut storage,
                        id,
                        &[Value::Text(flipped.to_string()), Value::Text(name.clone())],
                    )
                    .expect("flip owner");
                model.insert(id, (flipped, name));
                format!("flip {id} -> {flipped}")
            } else if roll < 90 {
                let id = live[rng.below(live.len())];
                server_qm.delete(&mut storage, id).expect("delete");
                model.remove(&id);
                format!("delete {id}")
            } else {
                "process".to_string()
            };
            server_qm.process(&mut storage);
            let outbox = server_qm.sync_manager_mut().take_outbox();
            confirm_delivered(&mut server_qm, &outbox);

            assert_eq!(
                server_page_scope(&server_qm),
                expected(&model),
                "seed {seed} (offset {offset}, limit {page}), step {step} after `{op}`: \
                 the page scope is not the first {} rows alice may read",
                offset + page,
            );
        }
    }
}

/// `items` pages with their `tags` included. Tags are readable by everyone; `item_select` is
/// the read policy of the items.
fn items_with_tags_schema(item_select: PolicyExpr) -> Schema {
    let mut schema = Schema::new();
    schema.insert(
        TableName::new("items"),
        TableSchema::with_policies(
            RowDescriptor::new(vec![
                ColumnDescriptor::new("owner_id", ColumnType::Text),
                ColumnDescriptor::new("name", ColumnType::Text),
            ]),
            TablePolicies::new().with_select(item_select),
        ),
    );
    schema.insert(
        TableName::new("tags"),
        TableSchema::with_policies(
            RowDescriptor::new(vec![
                ColumnDescriptor::new("item_id", ColumnType::Uuid).references("items"),
                ColumnDescriptor::new("label", ColumnType::Text),
            ]),
            TablePolicies::new().with_select(PolicyExpr::True),
        ),
    );
    schema
}

fn structural(schema: &Schema) -> Schema {
    schema
        .iter()
        .map(|(table_name, table_schema)| {
            let mut structural = table_schema.clone();
            structural.policies = TablePolicies::default();
            (*table_name, structural)
        })
        .collect()
}

/// A server in the shape production runs: it compiles queries against the structural
/// schema and authorizes the sync scope against a separate authorization schema — so its
/// graphs carry no policy filter, and rows the session may not read stay in the ordering.
fn server_with_authorization(
    authorization: Schema,
) -> (QueryManager, CountingCatalogueUpsertsStorage) {
    let mut server_qm = QueryManager::new(SyncManager::new());
    server_qm.set_current_schema(structural(&authorization), "dev", "main");
    server_qm.set_authorization_schema(authorization);
    let inner = seeded_memory_storage(&server_qm.schema_context().current_schema);
    (
        server_qm,
        CountingCatalogueUpsertsStorage::with_inner(inner),
    )
}

fn items_page_with_tags(qm: &QueryManager, offset: usize, page: usize) -> Query {
    qm.query("items")
        .order_by("name")
        .offset(offset)
        .limit(page)
        .with_array("tags", |sub| {
            sub.from("tags").correlate("item_id", "items.id")
        })
        .build()
}

fn insert_tag<H: Storage>(
    qm: &mut QueryManager,
    storage: &mut H,
    item: ObjectId,
    label: &str,
) -> ObjectId {
    qm.insert(
        storage,
        "tags",
        &[Value::Uuid(item), Value::Text(label.to_string())],
    )
    .expect("insert tag")
    .row_id
}

fn server_include_instances(server_qm: &QueryManager) -> usize {
    use crate::query_manager::graph::GraphNode;

    server_qm
        .server_subscriptions
        .values()
        .flat_map(|sub| sub.graph.nodes.iter())
        .map(|node| match &node.node {
            GraphNode::ArraySubquery(array) => array.cached_subgraph_count(),
            _ => 0,
        })
        .sum()
}

/// A page with includes builds them for the page's rows, not for every row the query's
/// filters match: on linsa-v22 a 40-row page with the thread include over a 10k-message
/// chat held 10k include instances, took ~2.3 s of server CPU to open and ~640 MB of RSS,
/// and re-evaluated them on each write (measured 2026-09-30). Production shape: rows every
/// session may read, a separate authorization schema.
#[test]
fn a_page_builds_its_includes_for_its_own_rows_only() {
    use crate::sync_manager::ClientId;

    // Includes must track their rows precisely: a parallel test may otherwise hold the
    // legacy coarse path, which has known staleness of its own (and fails this identically
    // without deferred includes).
    let _precise = crate::query_manager::precise_dirty::force_precise_dirty(true);
    let _routing = crate::query_manager::graph_nodes::include_routing::force_include_routing(true);

    const ROWS: usize = 300;
    const PAGE: usize = 5;

    let (mut server_qm, mut storage) =
        server_with_authorization(items_with_tags_schema(PolicyExpr::True));
    let mut expected = std::collections::HashSet::new();
    for index in 0..ROWS {
        let item = insert_item(
            &mut server_qm,
            &mut storage,
            "alice",
            &format!("Item {index:03}"),
        );
        let tag = insert_tag(
            &mut server_qm,
            &mut storage,
            item,
            &format!("tag {index:03}"),
        );
        if index < PAGE {
            expected.extend([item, tag]);
        }
    }
    server_qm.process(&mut storage);

    let client_id = ClientId::new();
    connect_client(&mut server_qm, &storage, client_id);
    let _ = server_qm.sync_manager_mut().take_outbox();
    let query = items_page_with_tags(&server_qm, 0, PAGE);
    subscribe_query_as(&mut server_qm, &mut storage, client_id, "alice", query);

    assert_eq!(
        server_page_scope(&server_qm),
        expected,
        "the page scope is its {PAGE} rows and their tags"
    );
    let instances = server_include_instances(&server_qm);
    assert!(
        instances <= PAGE,
        "a {PAGE}-row page over {ROWS} matching rows holds {instances} include instances"
    );

    let outside = insert_item(&mut server_qm, &mut storage, "alice", "Item 999");
    insert_tag(&mut server_qm, &mut storage, outside, "tag 999");
    server_qm.process(&mut storage);
    assert_eq!(
        server_page_scope(&server_qm),
        expected,
        "a write outside the page leaves it"
    );
    let instances = server_include_instances(&server_qm);
    assert!(
        instances <= PAGE,
        "a write outside a {PAGE}-row page left {instances} include instances"
    );
}

/// When the authorization schema denies rows ahead of the page, the page is the first rows
/// the session may read — past the graph's own window — and every one of them still carries
/// its includes to the client. Production shape: the graph has no policy filter, so the
/// denied rows sit in its ordering.
#[test]
fn a_page_past_denied_rows_carries_the_includes_of_every_row() {
    use crate::sync_manager::ClientId;

    // Includes must track their rows precisely: a parallel test may otherwise hold the
    // legacy coarse path, which has known staleness of its own (and fails this identically
    // without deferred includes).
    let _precise = crate::query_manager::precise_dirty::force_precise_dirty(true);
    let _routing = crate::query_manager::graph_nodes::include_routing::force_include_routing(true);

    const PAGE: usize = 5;

    let (mut server_qm, mut storage) = server_with_authorization(items_with_tags_schema(
        PolicyExpr::eq_session("owner_id", vec!["user_id".into()]),
    ));
    let mut expected = std::collections::HashSet::new();
    let mut alice_rows = 0;
    for index in 0..40 {
        let owner = if index % 2 == 0 { "bob" } else { "alice" };
        let item = insert_item(
            &mut server_qm,
            &mut storage,
            owner,
            &format!("Item {index:03}"),
        );
        let tag = insert_tag(
            &mut server_qm,
            &mut storage,
            item,
            &format!("tag {index:03}"),
        );
        if owner == "alice" {
            if alice_rows < PAGE {
                expected.extend([item, tag]);
            }
            alice_rows += 1;
        }
    }
    server_qm.process(&mut storage);

    let client_id = ClientId::new();
    connect_client(&mut server_qm, &storage, client_id);
    let _ = server_qm.sync_manager_mut().take_outbox();
    let query = items_page_with_tags(&server_qm, 0, PAGE);
    subscribe_query_as(&mut server_qm, &mut storage, client_id, "alice", query);

    assert_eq!(
        server_page_scope(&server_qm),
        expected,
        "the page is alice's first {PAGE} rows, each with its tag"
    );
}

/// Differential for the sync scope of a page with includes, on a production-shape server,
/// once with rows everyone may read (the includes are built for the page alone) and once
/// with rows owned per session (denied rows sit in the ordering). A seeded stream of item
/// inserts, renames, owner flips and deletes and of tag inserts, moves and deletes runs
/// against a model: the scope holds the first `offset + limit` items alice may read and the
/// tags of the page's items, and may hold the tags of the offset's items — the client needs
/// those rows to place the page, not their includes.
#[test]
fn page_scope_with_includes_matches_the_model_under_random_writes() {
    use crate::sync_manager::ClientId;
    use std::collections::{BTreeMap, HashSet};

    // Includes must track their rows precisely: a parallel test may otherwise hold the
    // legacy coarse path, which has known staleness of its own (and fails this identically
    // without deferred includes).
    let _precise = crate::query_manager::precise_dirty::force_precise_dirty(true);
    let _routing = crate::query_manager::graph_nodes::include_routing::force_include_routing(true);

    const STEPS: usize = 300;
    const NAMES: usize = 40;

    use crate::query_manager::graph::IncludePlacement;

    let owned_policy = PolicyExpr::eq_session("owner_id", vec!["user_id".into()]);
    // The last variant makes the server predict wrongly that no row can be denied, so the
    // scope walk has to catch every overrun and rebuild the graph.
    let policies = [
        ("readable", PolicyExpr::True, None),
        ("owned", owned_policy.clone(), None),
        (
            "owned, mispredicted",
            owned_policy,
            Some(IncludePlacement::PageRows),
        ),
    ];
    for (policy_name, item_select, placement_override) in policies {
        for seed in [1u64, 7, 42, 1337, 9001, 65_537] {
            let mut rng = PagePrng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
            let offset = rng.below(4);
            let page = 1 + rng.below(8);
            let owned = policy_name != "readable";

            let (mut server_qm, mut storage) =
                server_with_authorization(items_with_tags_schema(item_select.clone()));
            server_qm.include_placement_override = placement_override;

            let owners = ["alice", "bob"];
            let mut items: BTreeMap<ObjectId, (&'static str, String)> = BTreeMap::new();
            let mut tags: BTreeMap<ObjectId, ObjectId> = BTreeMap::new();
            for index in 0..30 {
                let owner = owners[rng.below(2)];
                let name = format!("Item {:02}", rng.below(NAMES));
                let item = insert_item(&mut server_qm, &mut storage, owner, &name);
                items.insert(item, (owner, name));
                if rng.below(3) > 0 {
                    let tag = insert_tag(&mut server_qm, &mut storage, item, &format!("t{index}"));
                    tags.insert(tag, item);
                }
            }
            server_qm.process(&mut storage);

            let client_id = ClientId::new();
            connect_client(&mut server_qm, &storage, client_id);
            let _ = server_qm.sync_manager_mut().take_outbox();
            let query = items_page_with_tags(&server_qm, offset, page);
            subscribe_query_as(&mut server_qm, &mut storage, client_id, "alice", query);

            // (required, allowed)
            let model = |items: &BTreeMap<ObjectId, (&'static str, String)>,
                         tags: &BTreeMap<ObjectId, ObjectId>|
             -> (HashSet<ObjectId>, HashSet<ObjectId>) {
                let mut readable: Vec<(&String, ObjectId)> = items
                    .iter()
                    .filter(|(_, (owner, _))| !owned || *owner == "alice")
                    .map(|(id, (_, name))| (name, *id))
                    .collect();
                readable.sort();
                let prefix: Vec<ObjectId> = readable
                    .into_iter()
                    .take(offset + page)
                    .map(|(_, id)| id)
                    .collect();
                let page_items: HashSet<ObjectId> = prefix.iter().skip(offset).copied().collect();
                let offset_items: HashSet<ObjectId> = prefix.iter().take(offset).copied().collect();
                let mut required: HashSet<ObjectId> = prefix.iter().copied().collect();
                required.extend(
                    tags.iter()
                        .filter(|(_, item)| page_items.contains(item))
                        .map(|(tag, _)| *tag),
                );
                let mut allowed = required.clone();
                allowed.extend(
                    tags.iter()
                        .filter(|(_, item)| offset_items.contains(item))
                        .map(|(tag, _)| *tag),
                );
                (required, allowed)
            };

            for step in 0..STEPS {
                let live_items: Vec<ObjectId> = items.keys().copied().collect();
                let live_tags: Vec<ObjectId> = tags.keys().copied().collect();
                let roll = rng.below(100);
                let op = if roll < 20 || live_items.is_empty() {
                    let owner = owners[rng.below(2)];
                    let name = format!("Item {:02}", rng.below(NAMES));
                    let item = insert_item(&mut server_qm, &mut storage, owner, &name);
                    items.insert(item, (owner, name.clone()));
                    format!("insert item {owner} {name}")
                } else if roll < 32 {
                    let item = live_items[rng.below(live_items.len())];
                    let owner = items[&item].0;
                    let name = format!("Item {:02}", rng.below(NAMES));
                    server_qm
                        .update(
                            &mut storage,
                            item,
                            &[Value::Text(owner.to_string()), Value::Text(name.clone())],
                        )
                        .expect("rename");
                    items.insert(item, (owner, name.clone()));
                    format!("rename {item} -> {name}")
                } else if roll < 42 {
                    let item = live_items[rng.below(live_items.len())];
                    let (owner, name) = items[&item].clone();
                    let flipped = if owner == "alice" { "bob" } else { "alice" };
                    server_qm
                        .update(
                            &mut storage,
                            item,
                            &[Value::Text(flipped.to_string()), Value::Text(name.clone())],
                        )
                        .expect("flip owner");
                    items.insert(item, (flipped, name));
                    format!("flip {item} -> {flipped}")
                } else if roll < 50 {
                    let item = live_items[rng.below(live_items.len())];
                    server_qm.delete(&mut storage, item).expect("delete item");
                    items.remove(&item);
                    format!("delete item {item}")
                } else if roll < 72 {
                    let item = live_items[rng.below(live_items.len())];
                    let tag = insert_tag(&mut server_qm, &mut storage, item, &format!("s{step}"));
                    tags.insert(tag, item);
                    format!("tag {item}")
                } else if roll < 84 && !live_tags.is_empty() {
                    let tag = live_tags[rng.below(live_tags.len())];
                    let item = live_items[rng.below(live_items.len())];
                    server_qm
                        .update(
                            &mut storage,
                            tag,
                            &[Value::Uuid(item), Value::Text(format!("m{step}"))],
                        )
                        .expect("move tag");
                    tags.insert(tag, item);
                    format!("move tag {tag} -> {item}")
                } else if roll < 94 && !live_tags.is_empty() {
                    let tag = live_tags[rng.below(live_tags.len())];
                    server_qm.delete(&mut storage, tag).expect("delete tag");
                    tags.remove(&tag);
                    format!("delete tag {tag}")
                } else {
                    "process".to_string()
                };
                server_qm.process(&mut storage);
                let outbox = server_qm.sync_manager_mut().take_outbox();
                confirm_delivered(&mut server_qm, &outbox);

                if placement_override.is_none() {
                    assert!(
                        server_qm
                            .server_subscriptions
                            .values()
                            .all(|sub| !sub.includes_past_page),
                        "{policy_name} rows, seed {seed}, step {step} after `{op}`: the policies \
                         predicted where the includes go, yet the page ran past them"
                    );
                }
                let scope = server_page_scope(&server_qm);
                let (required, allowed) = model(&items, &tags);
                assert!(
                    required.is_subset(&scope) && scope.is_subset(&allowed),
                    "{policy_name} rows, seed {seed} (offset {offset}, limit {page}), step {step} \
                     after `{op}`: missing {:?}, unexpected {:?}",
                    required.difference(&scope).collect::<Vec<_>>(),
                    scope.difference(&allowed).collect::<Vec<_>>(),
                );
            }
        }
    }
}

/// Differential for what a page with includes emits: a local subscription over a seeded
/// stream of writes must show, in order, the page's items each with exactly its tags —
/// both in its current result and in the deltas it sends, applied one after another.
#[test]
fn page_with_includes_emits_the_model_page_under_random_writes() {
    use std::collections::{BTreeMap, HashMap};

    // Includes must track their rows precisely: a parallel test may otherwise hold the
    // legacy coarse path, which has known staleness of its own (and fails this identically
    // without deferred includes).
    let _precise = crate::query_manager::precise_dirty::force_precise_dirty(true);
    let _routing = crate::query_manager::graph_nodes::include_routing::force_include_routing(true);

    const STEPS: usize = 300;
    const NAMES: usize = 40;

    fn decode(values: &[Value]) -> (String, Vec<String>) {
        let name = match &values[1] {
            Value::Text(name) => name.clone(),
            other => panic!("expected a name, got {other:?}"),
        };
        let mut labels: Vec<String> = values[2]
            .as_array()
            .expect("tags is an array")
            .iter()
            .map(|tag| match &tag.as_row().expect("a tag row")[1] {
                Value::Text(label) => label.clone(),
                other => panic!("expected a label, got {other:?}"),
            })
            .collect();
        labels.sort();
        (name, labels)
    }

    for seed in [3u64, 11, 99, 2024, 31_337, 77_777] {
        let mut rng = PagePrng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let offset = rng.below(4);
        let page = 1 + rng.below(8);

        let (mut qm, mut storage) = create_query_manager(
            SyncManager::new(),
            structural(&items_with_tags_schema(PolicyExpr::True)),
        );

        let mut items: BTreeMap<ObjectId, String> = BTreeMap::new();
        let mut tags: BTreeMap<ObjectId, (ObjectId, String)> = BTreeMap::new();
        for index in 0..30 {
            let name = format!("Item {:02}", rng.below(NAMES));
            let item = qm
                .insert(
                    &mut storage,
                    "items",
                    &[Value::Text("alice".into()), Value::Text(name.clone())],
                )
                .expect("insert item")
                .row_id;
            items.insert(item, name);
            if rng.below(3) > 0 {
                let label = format!("t{index}");
                let tag = insert_tag(&mut qm, &mut storage, item, &label);
                tags.insert(tag, (item, label));
            }
        }
        let query = items_page_with_tags(&qm, offset, page);
        let sub_id = qm.subscribe(query).expect("subscribe");
        qm.process(&mut storage);

        let expected = |items: &BTreeMap<ObjectId, String>,
                        tags: &BTreeMap<ObjectId, (ObjectId, String)>|
         -> Vec<(ObjectId, (String, Vec<String>))> {
            let mut ordered: Vec<(&String, ObjectId)> =
                items.iter().map(|(id, name)| (name, *id)).collect();
            ordered.sort();
            ordered
                .into_iter()
                .skip(offset)
                .take(page)
                .map(|(name, id)| {
                    let mut labels: Vec<String> = tags
                        .values()
                        .filter(|(item, _)| *item == id)
                        .map(|(_, label)| label.clone())
                        .collect();
                    labels.sort();
                    (id, (name.clone(), labels))
                })
                .collect()
        };

        let mut emitted: HashMap<ObjectId, (String, Vec<String>)> = HashMap::new();
        for step in 0..=STEPS {
            let op = if step == 0 {
                "subscribe".to_string()
            } else {
                let live_items: Vec<ObjectId> = items.keys().copied().collect();
                let live_tags: Vec<ObjectId> = tags.keys().copied().collect();
                let roll = rng.below(100);
                let op = if roll < 20 || live_items.is_empty() {
                    let name = format!("Item {:02}", rng.below(NAMES));
                    let item = qm
                        .insert(
                            &mut storage,
                            "items",
                            &[Value::Text("alice".into()), Value::Text(name.clone())],
                        )
                        .expect("insert item")
                        .row_id;
                    items.insert(item, name.clone());
                    format!("insert item {name}")
                } else if roll < 35 {
                    let item = live_items[rng.below(live_items.len())];
                    let name = format!("Item {:02}", rng.below(NAMES));
                    qm.update(
                        &mut storage,
                        item,
                        &[Value::Text("alice".into()), Value::Text(name.clone())],
                    )
                    .expect("rename");
                    items.insert(item, name.clone());
                    format!("rename {item} -> {name}")
                } else if roll < 45 {
                    let item = live_items[rng.below(live_items.len())];
                    qm.delete(&mut storage, item).expect("delete item");
                    items.remove(&item);
                    format!("delete item {item}")
                } else if roll < 70 {
                    let item = live_items[rng.below(live_items.len())];
                    let label = format!("s{step}");
                    let tag = insert_tag(&mut qm, &mut storage, item, &label);
                    tags.insert(tag, (item, label));
                    format!("tag {item}")
                } else if roll < 85 && !live_tags.is_empty() {
                    let tag = live_tags[rng.below(live_tags.len())];
                    let item = live_items[rng.below(live_items.len())];
                    let label = format!("m{step}");
                    qm.update(
                        &mut storage,
                        tag,
                        &[Value::Uuid(item), Value::Text(label.clone())],
                    )
                    .expect("move tag");
                    tags.insert(tag, (item, label));
                    format!("move tag {tag} -> {item}")
                } else if roll < 95 && !live_tags.is_empty() {
                    let tag = live_tags[rng.below(live_tags.len())];
                    qm.delete(&mut storage, tag).expect("delete tag");
                    tags.remove(&tag);
                    format!("delete tag {tag}")
                } else {
                    "process".to_string()
                };
                qm.process(&mut storage);
                op
            };

            for update in qm
                .take_updates()
                .into_iter()
                .filter(|u| u.subscription_id == sub_id)
            {
                for row in &update.delta.removed {
                    emitted.remove(&row.id);
                }
                for row in &update.delta.added {
                    let values = decode_row(&update.descriptor, &row.data).expect("decode added");
                    emitted.insert(row.id, decode(&values));
                }
                for (_, row) in &update.delta.updated {
                    let values = decode_row(&update.descriptor, &row.data).expect("decode updated");
                    emitted.insert(row.id, decode(&values));
                }
            }

            let want = expected(&items, &tags);
            let current: Vec<(ObjectId, (String, Vec<String>))> = qm
                .get_subscription_results(sub_id)
                .into_iter()
                .map(|(id, values)| (id, decode(&values)))
                .collect();
            assert_eq!(
                current, want,
                "seed {seed} (offset {offset}, limit {page}), step {step} after `{op}`: \
                 the current page"
            );
            let want_map: HashMap<ObjectId, (String, Vec<String>)> = want.into_iter().collect();
            assert_eq!(
                emitted, want_map,
                "seed {seed} (offset {offset}, limit {page}), step {step} after `{op}`: \
                 the page the deltas add up to"
            );
        }
    }
}

/// The server predicts from the policies alone whether rows ahead of a page can be
/// denied, and a row can be denied for a reason no policy shows (it fails to load, or has
/// no lens into the authorization schema). When the prediction misses, the scope walk
/// finds a page row past the rows whose includes were built, and the subscription is
/// rebuilt with includes for every matching row before its first scope goes out.
#[test]
fn a_mispredicted_page_is_rebuilt_with_includes_before_its_scope_goes_out() {
    use crate::query_manager::graph::IncludePlacement;
    use crate::sync_manager::ClientId;

    const PAGE: usize = 5;

    let _precise = crate::query_manager::precise_dirty::force_precise_dirty(true);
    let _routing = crate::query_manager::graph_nodes::include_routing::force_include_routing(true);

    let (mut server_qm, mut storage) = server_with_authorization(items_with_tags_schema(
        PolicyExpr::eq_session("owner_id", vec!["user_id".into()]),
    ));
    server_qm.include_placement_override = Some(IncludePlacement::PageRows);
    let mut expected = std::collections::HashSet::new();
    let mut alice_rows = 0;
    for index in 0..40 {
        let owner = if index % 2 == 0 { "bob" } else { "alice" };
        let item = insert_item(
            &mut server_qm,
            &mut storage,
            owner,
            &format!("Item {index:03}"),
        );
        let tag = insert_tag(
            &mut server_qm,
            &mut storage,
            item,
            &format!("tag {index:03}"),
        );
        if owner == "alice" {
            if alice_rows < PAGE {
                expected.extend([item, tag]);
            }
            alice_rows += 1;
        }
    }
    server_qm.process(&mut storage);

    let client_id = ClientId::new();
    connect_client(&mut server_qm, &storage, client_id);
    let _ = server_qm.sync_manager_mut().take_outbox();
    let query = items_page_with_tags(&server_qm, 0, PAGE);
    subscribe_query_as(&mut server_qm, &mut storage, client_id, "alice", query);

    assert_eq!(
        server_page_scope(&server_qm),
        expected,
        "the page is alice's first {PAGE} rows, each with its tag"
    );
    assert!(
        server_qm
            .server_subscriptions
            .values()
            .all(|sub| sub.includes_past_page),
        "the subscription now builds includes for every matching row"
    );
}

/// The same miss on a live subscription: a row of the page becomes one the session may not
/// read, the page slides one row past the graph's window, and the settle rebuilds the
/// graph in that pass rather than sending the new row without its includes.
#[test]
fn a_page_that_slides_past_its_includes_is_rebuilt_in_the_same_pass() {
    use crate::query_manager::graph::IncludePlacement;
    use crate::sync_manager::ClientId;

    const PAGE: usize = 5;

    let _precise = crate::query_manager::precise_dirty::force_precise_dirty(true);
    let _routing = crate::query_manager::graph_nodes::include_routing::force_include_routing(true);

    let (mut server_qm, mut storage) = server_with_authorization(items_with_tags_schema(
        PolicyExpr::eq_session("owner_id", vec!["user_id".into()]),
    ));
    server_qm.include_placement_override = Some(IncludePlacement::PageRows);
    let mut rows = Vec::new();
    for index in 0..20 {
        let name = format!("Item {index:03}");
        let item = insert_item(&mut server_qm, &mut storage, "alice", &name);
        let tag = insert_tag(
            &mut server_qm,
            &mut storage,
            item,
            &format!("tag {index:03}"),
        );
        rows.push((item, tag, name));
    }
    server_qm.process(&mut storage);

    let client_id = ClientId::new();
    connect_client(&mut server_qm, &storage, client_id);
    let _ = server_qm.sync_manager_mut().take_outbox();
    let query = items_page_with_tags(&server_qm, 0, PAGE);
    subscribe_query_as(&mut server_qm, &mut storage, client_id, "alice", query);
    assert!(
        server_qm
            .server_subscriptions
            .values()
            .all(|sub| !sub.includes_past_page),
        "fixture: nothing ahead of the page is denied yet, so its includes stay on the page"
    );

    let (head, _, head_name) = &rows[0];
    server_qm
        .update(
            &mut storage,
            *head,
            &[
                Value::Text("bob".to_string()),
                Value::Text(head_name.clone()),
            ],
        )
        .expect("give the head row to bob");
    server_qm.process(&mut storage);

    let expected: std::collections::HashSet<ObjectId> = rows[1..=PAGE]
        .iter()
        .flat_map(|(item, tag, _)| [*item, *tag])
        .collect();
    assert_eq!(
        server_page_scope(&server_qm),
        expected,
        "the page slid one row, and its new last row came with its tag"
    );
    assert!(
        server_qm
            .server_subscriptions
            .values()
            .all(|sub| sub.includes_past_page),
        "the subscription now builds includes for every matching row"
    );

    // A recompile (a schema or permissions change) keeps that placement.
    for sub in server_qm.server_subscriptions.values_mut() {
        sub.needs_recompile = true;
    }
    server_qm.process(&mut storage);
    assert!(
        server_qm
            .server_subscriptions
            .values()
            .all(|sub| sub.includes_past_page && sub.graph.deferred_arrays_tail.is_none()),
        "a recompiled subscription still builds includes for every matching row"
    );
    let (second, _, second_name) = &rows[1];
    server_qm
        .update(
            &mut storage,
            *second,
            &[
                Value::Text("bob".to_string()),
                Value::Text(second_name.clone()),
            ],
        )
        .expect("give the next head row to bob");
    server_qm.process(&mut storage);
    let expected: std::collections::HashSet<ObjectId> = rows[2..=PAGE + 1]
        .iter()
        .flat_map(|(item, tag, _)| [*item, *tag])
        .collect();
    assert_eq!(
        server_page_scope(&server_qm),
        expected,
        "after the recompile the page slides again with its includes"
    );
}

/// `wmsgs` (the tests declare its `chat+at` composite index, `declare_indexes`) with
/// rows readable by the session named in `body`.
fn owned_window_messages_schema() -> Schema {
    let mut schema = Schema::new();
    schema.insert(
        TableName::new("wmsgs"),
        TableSchema::with_policies(
            RowDescriptor::new(vec![
                ColumnDescriptor::new("chat", ColumnType::Uuid),
                ColumnDescriptor::new("at", ColumnType::Timestamp),
                ColumnDescriptor::new("dead", ColumnType::Boolean),
                ColumnDescriptor::new("body", ColumnType::Text),
            ]),
            TablePolicies::new()
                .with_select(PolicyExpr::eq_session("body", vec!["user_id".into()])),
        ),
    );
    schema
}

fn window_message_values(chat: ObjectId, at: u64, dead: bool, owner: &str) -> [Value; 4] {
    [
        Value::Uuid(chat),
        Value::Timestamp(at),
        Value::Boolean(dead),
        Value::Text(owner.to_string()),
    ]
}

fn window_page_query(
    qm: &QueryManager,
    chat: ObjectId,
    desc: bool,
    offset: usize,
    limit: usize,
) -> Query {
    let builder = qm
        .query("wmsgs")
        .filter_eq("chat", Value::Uuid(chat))
        .filter_eq("dead", Value::Boolean(false));
    let builder = if desc {
        builder.order_by_desc("at")
    } else {
        builder.order_by("at")
    };
    builder.offset(offset).limit(limit).build()
}

/// A page read through a composite-index window (`IndexScanNode::new_window`) on a
/// production-shape server. The graph carries no policy filter, so the window holds rows
/// the session may not read, and the page is the first rows the session may read — past
/// the window's first walk when the rows ahead of them are denied. The window has to walk
/// on until it holds them: sized for the page alone, it left the scope empty.
#[test]
fn a_windowed_page_past_denied_rows_is_the_first_readable_rows() {
    use crate::sync_manager::ClientId;

    const ROWS: u64 = 300;
    const DENIED_AHEAD: u64 = 60;
    const PAGE: usize = 10;

    let (mut server_qm, mut storage) = server_with_authorization(owned_window_messages_schema());

    declare_indexes(&mut server_qm, &mut storage, wmsgs_index_declarations());
    let chat = ObjectId::new();
    let mut alice = Vec::new();
    for index in 0..ROWS {
        // The newest rows are bob's, so a newest-first page walks them first.
        let owner = if index >= ROWS - DENIED_AHEAD {
            "bob"
        } else {
            "alice"
        };
        let id = server_qm
            .insert(
                &mut storage,
                "wmsgs",
                &window_message_values(chat, index * 10, false, owner),
            )
            .expect("insert message")
            .row_id;
        if owner == "alice" {
            alice.push(id);
        }
    }
    server_qm.process(&mut storage);

    let client_id = ClientId::new();
    connect_client(&mut server_qm, &storage, client_id);
    let _ = server_qm.sync_manager_mut().take_outbox();
    let query = window_page_query(&server_qm, chat, true, 0, PAGE);
    subscribe_query_as(&mut server_qm, &mut storage, client_id, "alice", query);

    let expected: std::collections::HashSet<ObjectId> =
        alice.iter().rev().take(PAGE).copied().collect();
    assert_eq!(
        server_page_scope(&server_qm),
        expected,
        "the page is alice's newest {PAGE} rows, behind {DENIED_AHEAD} of bob's"
    );
    // The page came through the window, not through a scan of the chat.
    let mut scanned = Vec::new();
    for sub in server_qm.server_subscriptions.values() {
        sub.graph.collect_scanned_row_ids(&mut scanned);
    }
    assert!(
        scanned.len() < ROWS as usize / 2,
        "the windowed page scanned {} of the chat's {ROWS} rows",
        scanned.len()
    );
}

/// Differential for a windowed page's sync scope on a production-shape server: a seeded
/// stream of inserts (ties included), owner flips and `dead` flips against a model — the
/// scope is the first `offset + limit` rows alice may read, in the page's order. Most
/// rows are bob's on half the seeds, so the rows alice may read sit past the window's
/// first walk.
#[test]
fn windowed_page_scope_matches_the_model_under_random_writes() {
    use crate::sync_manager::ClientId;
    use std::collections::{BTreeMap, HashSet};

    for seed in [1u64, 7, 42, 1337, 9001, 65_537, 3, 11, 101, 4096] {
        let mut rng = PagePrng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let desc = rng.below(2) == 0;
        let offset = rng.below(4);
        let limit = 1 + rng.below(10);
        let bob_share = if seed % 2 == 0 { 50 } else { 90 };

        let (mut server_qm, mut storage) =
            server_with_authorization(owned_window_messages_schema());

        declare_indexes(&mut server_qm, &mut storage, wmsgs_index_declarations());
        let chat = ObjectId::new();
        let other = ObjectId::new();
        // id -> (chat, at, dead, owner)
        let mut rows: BTreeMap<ObjectId, (ObjectId, u64, bool, &'static str)> = BTreeMap::new();
        let write = |server_qm: &mut QueryManager,
                     storage: &mut CountingCatalogueUpsertsStorage,
                     rows: &mut BTreeMap<ObjectId, (ObjectId, u64, bool, &'static str)>,
                     id: Option<ObjectId>,
                     row: (ObjectId, u64, bool, &'static str)| {
            let values = window_message_values(row.0, row.1, row.2, row.3);
            let id = match id {
                Some(id) => {
                    server_qm
                        .update(storage, id, &values)
                        .expect("update message");
                    id
                }
                None => {
                    server_qm
                        .insert(storage, "wmsgs", &values)
                        .expect("insert message")
                        .row_id
                }
            };
            rows.insert(id, row);
        };
        for _ in 0..(120 + rng.below(120)) {
            let owner = if rng.below(100) < bob_share {
                "bob"
            } else {
                "alice"
            };
            let owner_chat = if rng.below(5) == 0 { other } else { chat };
            // Ties: groups of rows share an `at`.
            let at = (rng.below(80) * 10) as u64;
            let dead = rng.below(6) == 0;
            write(
                &mut server_qm,
                &mut storage,
                &mut rows,
                None,
                (owner_chat, at, dead, owner),
            );
        }
        server_qm.process(&mut storage);

        let client_id = ClientId::new();
        connect_client(&mut server_qm, &storage, client_id);
        let _ = server_qm.sync_manager_mut().take_outbox();
        let query = window_page_query(&server_qm, chat, desc, offset, limit);
        subscribe_query_as(&mut server_qm, &mut storage, client_id, "alice", query);

        let model = |rows: &BTreeMap<ObjectId, (ObjectId, u64, bool, &'static str)>| {
            let mut readable: Vec<(u64, ObjectId)> = rows
                .iter()
                .filter(|(_, (row_chat, _, dead, owner))| {
                    *row_chat == chat && !dead && *owner == "alice"
                })
                .map(|(id, (_, at, _, _))| (*at, *id))
                .collect();
            readable.sort_by(|left, right| {
                let by_at = if desc {
                    right.0.cmp(&left.0)
                } else {
                    left.0.cmp(&right.0)
                };
                by_at.then(left.1.cmp(&right.1))
            });
            readable
                .into_iter()
                .take(offset + limit)
                .map(|(_, id)| id)
                .collect::<HashSet<ObjectId>>()
        };

        for step in 0..120 {
            let scope = server_page_scope(&server_qm);
            let want = model(&rows);
            assert_eq!(
                scope,
                want,
                "seed {seed} (desc {desc}, offset {offset}, limit {limit}), step {step}: \
                 missing {:?}, unexpected {:?}",
                want.difference(&scope).collect::<Vec<_>>(),
                scope.difference(&want).collect::<Vec<_>>(),
            );

            let ids: Vec<ObjectId> = rows.keys().copied().collect();
            let roll = rng.below(100);
            if roll < 30 || ids.is_empty() {
                let owner = if rng.below(100) < bob_share {
                    "bob"
                } else {
                    "alice"
                };
                let at = (rng.below(90) * 10) as u64;
                write(
                    &mut server_qm,
                    &mut storage,
                    &mut rows,
                    None,
                    (chat, at, false, owner),
                );
            } else {
                let id = ids[rng.below(ids.len())];
                let mut row = rows[&id];
                if roll < 65 {
                    row.3 = if row.3 == "alice" { "bob" } else { "alice" };
                } else {
                    row.2 = !row.2;
                }
                write(&mut server_qm, &mut storage, &mut rows, Some(id), row);
            }
            server_qm.process(&mut storage);
            let outbox = server_qm.sync_manager_mut().take_outbox();
            confirm_delivered(&mut server_qm, &outbox);
        }
    }
}
