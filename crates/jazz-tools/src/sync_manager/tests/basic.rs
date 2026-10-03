use super::*;

#[test]
fn can_create_sync_manager() {
    let sm = SyncManager::new();
    assert!(sm.servers.is_empty());
    assert!(sm.clients.is_empty());
}

#[test]
fn memory_size_separates_sync_state_buckets() {
    let empty = SyncManager::new().memory_size();
    assert_eq!(empty, (0, 0, 0, 0, 0));

    let mut sm = SyncManager::new();
    let io = MemoryStorage::new();
    let client_id = ClientId::new();
    let server_id = ServerId::new();
    let row_id = ObjectId::new();
    let row = visible_row(row_id, "main", Vec::new(), 1_000, b"alice");
    let row_key = RowBatchKey::from_row(&row);
    let query = QueryBuilder::new("users").branch("main").build();
    let session = crate::query_manager::session::Session::new("alice");

    add_client(&mut sm, &io, client_id);
    add_server(&mut sm, &io, server_id);
    set_client_query_scope(
        &mut sm,
        &io,
        client_id,
        QueryId(7),
        HashSet::from([(row_id, BranchName::new("main"))]),
        Some(session.clone()),
    );

    sm.row_batch_interest.insert(row_key, client_id);
    sm.query_origin
        .insert(QueryId(7), HashSet::from([client_id]));
    sm.clients
        .get_mut(&client_id)
        .expect("client should exist")
        .sent_batch_ids
        .insert(
            (row_id, BranchName::new("main")),
            SentBatchIds::from([row.batch_id]),
        );
    sm.servers
        .get_mut(&server_id)
        .expect("server should exist")
        .sent_metadata
        .insert(row_id);

    sm.outbox.push(OutboxEntry {
        destination: Destination::Client(client_id),
        payload: SyncPayload::QuerySettled {
            query_id: QueryId(7),
            tier: DurabilityTier::Local,
            scope: vec![],
            through_seq: 1,
        },
    });
    sm.inbox.push(InboxEntry {
        source: Source::Server(server_id),
        payload: SyncPayload::RowBatchNeeded {
            metadata: Some(RowMetadata {
                id: row_id,
                metadata: row_metadata("users"),
            }),
            row: row.clone(),
        },
    });
    sm.pending_query_subscriptions
        .push(PendingQuerySubscription {
            client_id,
            query_id: QueryId(8),
            query: query.clone(),
            session: Some(session.clone()),
            required_tier: None,
            propagation: QueryPropagation::Full,
            policy_context_tables: vec![],
            connection: 0,
        });
    sm.pending_query_unsubscriptions
        .push(PendingQueryUnsubscription {
            client_id,
            query_id: QueryId(7),
            cancelled_queued_subscription: false,
        });
    sm.pending_query_settled.push(PendingQuerySettled {
        server_id: Some(server_id),
        query_id: QueryId(7),
        tier: DurabilityTier::Local,
        through_seq: 2,
    });
    sm.pending_permission_checks.push(PendingPermissionCheck {
        id: PendingUpdateId(1),
        client_id,
        payload: SyncPayload::RowBatchCreated {
            metadata: Some(RowMetadata {
                id: row_id,
                metadata: row_metadata("users"),
            }),
            row: row.clone(),
        },
        session,
        schema_wait_started_at: None,
        metadata: row_metadata("users"),
        old_content: None,
        old_content_schema_hash: None,
        new_content: Some(b"alice".to_vec()),
        operation: crate::query_manager::policy::Operation::Insert,
    });

    let (catalogue, connections, subscriptions, queues, total) = sm.memory_size();

    assert_eq!(catalogue, 0);
    assert!(connections > 0);
    assert!(subscriptions > 0);
    assert!(queues > 0);
    assert_eq!(total, catalogue + connections + subscriptions + queues);
}

/// Row interest is counted in `memory_size`.
#[test]
fn memory_size_counts_row_interest() {
    let client_id = ClientId::new();
    let row = |batch_id| RowBatchKey::new(ObjectId::new(), BranchName::new("main"), batch_id);
    let mut sm = SyncManager::new();
    let (_, _, before, _, _) = sm.memory_size();
    sm.row_batch_interest.insert(row(BatchId::new()), client_id);
    let one_row = sm.row_batch_interest.memory_size();
    let shared = BatchId::new();
    sm.row_batch_interest.insert(row(shared), client_id);
    sm.row_batch_interest.insert(row(shared), client_id);
    let (_, _, after, _, _) = sm.memory_size();

    assert!(one_row > 0);
    assert_eq!(after - before, sm.row_batch_interest.memory_size());
}

/// A client that goes leaves no row interest behind.
#[test]
fn a_removed_client_leaves_no_row_interest() {
    let mut sm = SyncManager::new();
    let io = MemoryStorage::new();
    let client_id = ClientId::new();
    let other_id = ClientId::new();
    add_client(&mut sm, &io, client_id);
    add_client(&mut sm, &io, other_id);
    let batch_id = BatchId::new();
    let key = RowBatchKey::new(ObjectId::new(), BranchName::new("main"), batch_id);
    sm.row_batch_interest.insert(key, client_id);
    sm.row_batch_interest.insert(
        RowBatchKey::new(ObjectId::new(), BranchName::new("main"), batch_id),
        other_id,
    );

    assert!(
        !sm.row_batch_interest.holds_inline(batch_id),
        "a batch of two rows holds a table of them, else this gates nothing"
    );

    assert!(sm.remove_client(client_id));
    assert!(!sm.row_batch_interest.batch_has_client(batch_id, client_id));
    assert!(sm.row_batch_interest.batch_has_client(batch_id, other_id));
    assert!(
        sm.row_batch_interest.holds_inline(batch_id),
        "the batch's one row left is held inline again"
    );
}

/// A row that leaves a client's scope takes the client's interest in the batch it was sent.
#[test]
fn a_row_leaving_a_clients_scope_takes_its_row_interest() {
    let mut sm = SyncManager::new();
    let mut io = MemoryStorage::new();
    let client_id = ClientId::new();
    let row_id = ObjectId::new();
    let row = visible_row(row_id, "main", Vec::new(), 1_000, b"alice");
    add_client(&mut sm, &io, client_id);
    seed_visible_row(&mut sm, &mut io, "users", row.clone());
    set_client_query_scope(
        &mut sm,
        &io,
        client_id,
        QueryId(1),
        HashSet::from([(row_id, BranchName::new("main"))]),
        None,
    );
    confirm_queued(&mut sm);
    sm.take_outbox();
    assert!(
        sm.row_batch_interest
            .batch_has_client(row.batch_id, client_id),
        "the delivered row must be held, else this gates nothing"
    );

    sm.drop_client_query_subscription(&io, client_id, QueryId(1));

    assert!(
        !sm.row_batch_interest
            .batch_has_client(row.batch_id, client_id)
    );
}

#[test]
fn schema_warning_from_server_relays_to_interested_clients() {
    let mut sm = SyncManager::new();
    let mut io = MemoryStorage::new();
    let client_id = ClientId::new();
    let server_id = ServerId::new();
    let query_id = QueryId(42);

    add_client(&mut sm, &io, client_id);
    sm.take_outbox();
    sm.query_origin
        .entry(query_id)
        .or_default()
        .insert(client_id);

    sm.process_from_server(
        &mut io,
        server_id,
        SyncPayload::SchemaWarning(SchemaWarning {
            query_id,
            table_name: "users".to_string(),
            row_count: 3,
            from_hash: crate::query_manager::types::SchemaHash([0xAA; 32]),
            to_hash: crate::query_manager::types::SchemaHash([0xBB; 32]),
        }),
    );

    let outbox = sm.take_outbox();
    assert_eq!(outbox.len(), 1);
    match &outbox[0] {
        OutboxEntry {
            destination: Destination::Client(id),
            payload: SyncPayload::SchemaWarning(warning),
        } => {
            assert_eq!(*id, client_id);
            assert_eq!(warning.query_id, query_id);
            assert_eq!(warning.table_name, "users");
        }
        other => panic!("expected relayed schema warning, got {other:?}"),
    }
}
