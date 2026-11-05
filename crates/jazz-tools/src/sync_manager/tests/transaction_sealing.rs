use super::*;

#[test]
fn transactional_row_from_client_stays_staged_until_batch_is_sealed() {
    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::Local);
    let mut io = MemoryStorage::new();
    let client_id = ClientId::new();
    let row_id = ObjectId::new();
    let row = row_with_state(
        visible_row(row_id, "main", Vec::new(), 1_000, b"alice"),
        crate::row_histories::RowState::StagingPending,
        None,
    );
    seed_users_schema(&mut io);

    add_client(&mut sm, &io, client_id);
    sm.set_client_role(client_id, ClientRole::Peer);
    sm.take_outbox();

    sm.process_from_client(
        &mut io,
        client_id,
        SyncPayload::RowBatchCreated {
            metadata: Some(RowMetadata {
                id: row_id,
                metadata: row_metadata("users"),
            }),
            row: row.clone(),
        },
    );

    let history_rows = io.scan_history_row_batches("users", row_id).unwrap();
    assert_eq!(history_rows.len(), 1);
    assert_eq!(
        history_rows[0].state,
        crate::row_histories::RowState::StagingPending
    );
    assert_eq!(
        io.load_visible_region_row("users", "main", row_id).unwrap(),
        None,
        "staging rows should not become visible until the batch is sealed"
    );
    assert_eq!(
        io.load_authoritative_batch_fate(row.batch_id).unwrap(),
        None,
        "authority should not decide a transactional batch before it is sealed"
    );

    assert!(sm.take_outbox().into_iter().all(|entry| !matches!(
        entry,
        OutboxEntry {
            destination: Destination::Client(id),
            payload:
                SyncPayload::BatchFate { .. },
        } if id == client_id
    )));
}

#[test]
fn direct_batch_from_client_sends_one_settlement_on_seal() {
    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::Local);
    let mut io = MemoryStorage::new();
    let client_id = ClientId::new();
    let batch_id = crate::row_histories::BatchId::new();
    let alice_id = ObjectId::new();
    let bob_id = ObjectId::new();
    seed_users_schema(&mut io);

    add_client(&mut sm, &io, client_id);
    sm.set_client_role(client_id, ClientRole::Peer);
    sm.take_outbox();

    let alice = row_with_batch_state(
        visible_row(alice_id, "main", Vec::new(), 1_000, b"alice"),
        batch_id,
        crate::row_histories::RowState::VisibleDirect,
        None,
    );
    let bob = row_with_batch_state(
        visible_row(bob_id, "main", Vec::new(), 1_100, b"bob"),
        batch_id,
        crate::row_histories::RowState::VisibleDirect,
        None,
    );

    for row in [alice.clone(), bob.clone()] {
        sm.process_from_client(
            &mut io,
            client_id,
            SyncPayload::RowBatchCreated {
                metadata: Some(RowMetadata {
                    id: row.row_id,
                    metadata: row_metadata("users"),
                }),
                row,
            },
        );
    }

    assert!(sm.take_outbox().into_iter().all(|entry| !matches!(
        entry,
        OutboxEntry {
            destination: Destination::Client(id),
            payload:
                SyncPayload::BatchFate { .. },
        } if id == client_id
    )));

    sm.process_from_client(
        &mut io,
        client_id,
        SyncPayload::SealBatch {
            submission: sealed_submission(
                batch_id,
                "main",
                vec![
                    SealedBatchMember {
                        object_id: alice_id,
                        row_digest: alice.content_digest(),
                    },
                    SealedBatchMember {
                        object_id: bob_id,
                        row_digest: bob.content_digest(),
                    },
                ],
                Vec::new(),
            ),
        },
    );

    let outbox = sm.take_outbox();
    let settlements = outbox
        .iter()
        .filter_map(|entry| match entry {
            OutboxEntry {
                destination: Destination::Client(id),
                payload: SyncPayload::BatchFate { fate },
            } if *id == client_id => Some(fate),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(settlements.len(), 1);
    let BatchFate::DurableDirect {
        batch_id: settled_batch_id,
        confirmed_tier,
    } = settlements[0]
    else {
        panic!(
            "expected durable direct settlement, got {:?}",
            settlements[0]
        );
    };
    assert_eq!(*settled_batch_id, batch_id);
    assert_eq!(*confirmed_tier, DurabilityTier::Local);
}

#[test]
fn direct_batch_seal_promotes_existing_local_settlement_to_authority_tier() {
    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::EdgeServer);
    let mut io = MemoryStorage::new();
    let client_id = ClientId::new();
    let batch_id = crate::row_histories::BatchId::new();
    let row_id = ObjectId::new();
    seed_users_schema(&mut io);

    add_client(&mut sm, &io, client_id);
    sm.set_client_role(client_id, ClientRole::Peer);
    sm.take_outbox();

    let row = row_with_batch_state(
        visible_row(row_id, "main", Vec::new(), 1_000, b"alice"),
        batch_id,
        crate::row_histories::RowState::VisibleDirect,
        None,
    );
    sm.process_from_client(
        &mut io,
        client_id,
        SyncPayload::RowBatchCreated {
            metadata: Some(RowMetadata {
                id: row.row_id,
                metadata: row_metadata("users"),
            }),
            row: row.clone(),
        },
    );
    io.upsert_authoritative_batch_fate(&BatchFate::DurableDirect {
        batch_id,
        confirmed_tier: DurabilityTier::Local,
    })
    .unwrap();

    sm.process_from_client(
        &mut io,
        client_id,
        SyncPayload::SealBatch {
            submission: sealed_submission(
                batch_id,
                "main",
                vec![SealedBatchMember {
                    object_id: row_id,
                    row_digest: row.content_digest(),
                }],
                Vec::new(),
            ),
        },
    );

    assert_eq!(
        io.load_authoritative_batch_fate(batch_id).unwrap(),
        Some(BatchFate::DurableDirect {
            batch_id,
            confirmed_tier: DurabilityTier::EdgeServer,
        }),
        "sealing a direct batch should promote a stale local settlement to the accepting authority tier"
    );
}

#[test]
fn direct_client_settlement_is_not_authoritative() {
    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::Local);
    let mut io = MemoryStorage::new();
    let client_id = ClientId::new();
    let batch_id = crate::row_histories::BatchId::new();
    let row_id = ObjectId::new();
    seed_users_schema(&mut io);

    add_client(&mut sm, &io, client_id);
    sm.set_client_role(client_id, ClientRole::Peer);
    sm.take_outbox();

    let row = row_with_batch_state(
        visible_row(row_id, "main", Vec::new(), 1_000, b"alice"),
        batch_id,
        crate::row_histories::RowState::VisibleDirect,
        None,
    );
    sm.process_from_client(
        &mut io,
        client_id,
        SyncPayload::RowBatchCreated {
            metadata: Some(RowMetadata {
                id: row.row_id,
                metadata: row_metadata("users"),
            }),
            row: row.clone(),
        },
    );
    sm.process_from_client(
        &mut io,
        client_id,
        SyncPayload::BatchFate {
            fate: BatchFate::DurableDirect {
                batch_id,
                confirmed_tier: DurabilityTier::Local,
            },
        },
    );

    assert_eq!(
        io.load_authoritative_batch_fate(batch_id).unwrap(),
        None,
        "direct client fate must not become an authoritative settlement"
    );
    assert_eq!(io.load_local_batch_record(batch_id).unwrap(), None);
    assert_eq!(io.load_sealed_batch_submission(batch_id).unwrap(), None);
}

#[test]
fn direct_client_settlement_before_row_is_not_authoritative_after_row() {
    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::Local);
    let mut io = MemoryStorage::new();
    let client_id = ClientId::new();
    let server_id = ServerId::new();
    let batch_id = crate::row_histories::BatchId::new();
    let row_id = ObjectId::new();
    seed_users_schema(&mut io);

    add_client(&mut sm, &io, client_id);
    sm.set_client_role(client_id, ClientRole::Peer);
    sm.add_pending_server(server_id);
    sm.take_outbox();

    let row = row_with_batch_state(
        visible_row(row_id, "main", Vec::new(), 1_000, b"alice"),
        batch_id,
        crate::row_histories::RowState::VisibleDirect,
        None,
    );
    sm.process_from_client(
        &mut io,
        client_id,
        SyncPayload::BatchFate {
            fate: BatchFate::DurableDirect {
                batch_id,
                confirmed_tier: DurabilityTier::Local,
            },
        },
    );
    sm.process_from_client(
        &mut io,
        client_id,
        SyncPayload::RowBatchCreated {
            metadata: Some(RowMetadata {
                id: row.row_id,
                metadata: row_metadata("users"),
            }),
            row: row.clone(),
        },
    );

    assert_eq!(
        io.load_authoritative_batch_fate(batch_id).unwrap(),
        None,
        "direct client fate must not become authoritative after the row arrives"
    );
    let outbox = sm.take_outbox();
    assert!(
        outbox.is_empty(),
        "a fate received without SealBatch must not synthesize replayable membership; outbox={outbox:?}"
    );
}

#[test]
fn rejected_sealed_batch_records_late_declared_member_as_rejected() {
    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::Local);
    let mut io = MemoryStorage::new();
    let client_id = ClientId::new();
    let batch_id = BatchId::new();
    let row_id = ObjectId::new();
    seed_users_schema(&mut io);

    add_client(&mut sm, &io, client_id);
    sm.set_client_role(client_id, ClientRole::Peer);
    sm.take_outbox();

    let row = row_with_batch_state(
        visible_row(row_id, "main", Vec::new(), 1_000, b"alice"),
        batch_id,
        crate::row_histories::RowState::StagingPending,
        None,
    );
    sm.process_from_client(
        &mut io,
        client_id,
        SyncPayload::SealBatch {
            submission: transactional_sealed_submission(
                batch_id,
                "main",
                vec![SealedBatchMember {
                    object_id: row_id,
                    row_digest: row.content_digest(),
                }],
                Vec::new(),
            ),
        },
    );
    let rejected = BatchFate::Rejected {
        batch_id,
        code: "permission_denied".to_string(),
        reason: "writer lacks publish rights".to_string(),
    };
    io.upsert_authoritative_batch_fate(&rejected).unwrap();

    sm.process_from_client(
        &mut io,
        client_id,
        SyncPayload::RowBatchCreated {
            metadata: Some(RowMetadata {
                id: row_id,
                metadata: row_metadata("users"),
            }),
            row,
        },
    );

    assert_eq!(
        io.load_history_row_batch("users", "main", row_id, batch_id)
            .unwrap()
            .unwrap()
            .state,
        crate::row_histories::RowState::Rejected,
        "a late declared member must be accounted for without becoming visible"
    );
    assert_eq!(
        io.load_authoritative_batch_fate(batch_id).unwrap(),
        Some(rejected)
    );
}

#[test]
fn direct_client_settlement_before_row_keeps_sealed_submission_without_server() {
    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::Local);
    let mut io = MemoryStorage::new();
    let client_id = ClientId::new();
    let batch_id = crate::row_histories::BatchId::new();
    let row_id = ObjectId::new();
    seed_users_schema(&mut io);

    add_client(&mut sm, &io, client_id);
    sm.set_client_role(client_id, ClientRole::Peer);
    sm.take_outbox();

    let row = row_with_batch_state(
        visible_row(row_id, "main", Vec::new(), 1_000, b"alice"),
        batch_id,
        crate::row_histories::RowState::VisibleDirect,
        None,
    );
    sm.process_from_client(
        &mut io,
        client_id,
        SyncPayload::BatchFate {
            fate: BatchFate::DurableDirect {
                batch_id,
                confirmed_tier: DurabilityTier::Local,
            },
        },
    );
    sm.process_from_client(
        &mut io,
        client_id,
        SyncPayload::RowBatchCreated {
            metadata: Some(RowMetadata {
                id: row.row_id,
                metadata: row_metadata("users"),
            }),
            row: row.clone(),
        },
    );

    assert_eq!(
        io.load_authoritative_batch_fate(batch_id).unwrap(),
        None,
        "direct client fate must not become authoritative"
    );
    assert_eq!(
        io.load_sealed_batch_submission(batch_id).unwrap(),
        None,
        "a fate received without SealBatch must not synthesize replayable membership"
    );
}

#[test]
fn seal_batch_to_servers_targets_pending_server_transport() {
    let mut sm = SyncManager::new();
    let server_id = ServerId::new();
    let batch_id = crate::row_histories::BatchId::new();
    let row_id = ObjectId::new();
    let row = visible_row(row_id, "main", Vec::new(), 1_000, b"alice");

    sm.add_pending_server(server_id);
    sm.seal_batch_to_servers(sealed_submission(
        batch_id,
        "main",
        vec![SealedBatchMember {
            object_id: row_id,
            row_digest: row.content_digest(),
        }],
        Vec::new(),
    ));

    assert!(sm.take_outbox().into_iter().any(|entry| matches!(
        entry,
        OutboxEntry {
            destination: Destination::Server(id),
            payload: SyncPayload::SealBatch { submission },
        } if id == server_id && submission.batch_id == batch_id
    )));
}

#[test]
fn direct_batch_from_client_settles_when_rows_arrive_after_seal() {
    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::Local);
    let mut io = MemoryStorage::new();
    let client_id = ClientId::new();
    let batch_id = crate::row_histories::BatchId::new();
    let alice_id = ObjectId::new();
    let bob_id = ObjectId::new();
    seed_users_schema(&mut io);

    add_client(&mut sm, &io, client_id);
    sm.set_client_role(client_id, ClientRole::Peer);
    sm.take_outbox();

    let alice = row_with_batch_state(
        visible_row(alice_id, "main", Vec::new(), 1_000, b"alice"),
        batch_id,
        crate::row_histories::RowState::VisibleDirect,
        None,
    );
    let bob = row_with_batch_state(
        visible_row(bob_id, "main", Vec::new(), 1_100, b"bob"),
        batch_id,
        crate::row_histories::RowState::VisibleDirect,
        None,
    );

    sm.process_from_client(
        &mut io,
        client_id,
        SyncPayload::RowBatchCreated {
            metadata: Some(RowMetadata {
                id: alice.row_id,
                metadata: row_metadata("users"),
            }),
            row: alice.clone(),
        },
    );
    sm.process_from_client(
        &mut io,
        client_id,
        SyncPayload::SealBatch {
            submission: sealed_submission(
                batch_id,
                "main",
                vec![
                    SealedBatchMember {
                        object_id: alice_id,
                        row_digest: alice.content_digest(),
                    },
                    SealedBatchMember {
                        object_id: bob_id,
                        row_digest: bob.content_digest(),
                    },
                ],
                Vec::new(),
            ),
        },
    );
    sm.take_outbox();

    sm.process_from_client(
        &mut io,
        client_id,
        SyncPayload::RowBatchCreated {
            metadata: Some(RowMetadata {
                id: bob.row_id,
                metadata: row_metadata("users"),
            }),
            row: bob,
        },
    );

    let outbox = sm.take_outbox();
    assert!(outbox.iter().any(|entry| matches!(
        entry,
        OutboxEntry {
            destination: Destination::Client(id),
            payload: SyncPayload::BatchFate { fate: BatchFate::DurableDirect {
                    batch_id: settled_batch_id,
                    confirmed_tier: DurabilityTier::Local,},
            },
        } if *id == client_id
            && *settled_batch_id == batch_id    )));
}

#[test]
fn seal_batch_collapses_same_row_to_latest_visible_member() {
    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::Local);
    let mut io = MemoryStorage::new();
    let client_id = ClientId::new();
    let batch_id = crate::row_histories::BatchId::new();
    let row_id = ObjectId::new();
    seed_users_schema(&mut io);

    add_client(&mut sm, &io, client_id);
    sm.set_client_role(client_id, ClientRole::Peer);
    sm.take_outbox();

    // client
    //   first staged version  -> same row, same batch
    //   second staged version -> same row, same batch
    //   seal batch
    //
    // authority
    //   settles one visible member for that row
    //   publishes only the latest staged content
    let first_row = row_with_batch_state(
        visible_row(row_id, "main", Vec::new(), 1_000, b"alice"),
        batch_id,
        crate::row_histories::RowState::StagingPending,
        None,
    );

    let second_row = row_with_batch_state(
        visible_row(row_id, "main", Vec::new(), 1_100, b"alice-updated"),
        batch_id,
        crate::row_histories::RowState::StagingPending,
        None,
    );

    for row in [first_row.clone(), second_row.clone()] {
        sm.process_from_client(
            &mut io,
            client_id,
            SyncPayload::RowBatchCreated {
                metadata: Some(RowMetadata {
                    id: row.row_id,
                    metadata: row_metadata("users"),
                }),
                row,
            },
        );
    }
    sm.take_outbox();

    sm.process_from_client(
        &mut io,
        client_id,
        SyncPayload::SealBatch {
            submission: transactional_sealed_submission(
                batch_id,
                "main",
                vec![SealedBatchMember {
                    object_id: row_id,
                    row_digest: second_row.content_digest(),
                }],
                Vec::new(),
            ),
        },
    );

    let settlement = io
        .load_authoritative_batch_fate(batch_id)
        .unwrap()
        .expect("sealed transactional batch should persist an authoritative settlement");
    let BatchFate::AcceptedTransaction {
        batch_id: settled_batch_id,
        confirmed_tier,
    } = settlement
    else {
        panic!("expected accepted transactional settlement, got {settlement:?}");
    };
    assert_eq!(settled_batch_id, batch_id);
    assert_eq!(confirmed_tier, DurabilityTier::Local);

    let visible = io
        .load_visible_region_row("users", "main", row_id)
        .unwrap()
        .expect("latest row should become visible after seal");
    assert_eq!(visible.batch_id(), batch_id);
    assert!(visible.parents.is_empty());
    assert_eq!(visible.data, second_row.data);
    assert_eq!(visible.batch_id, batch_id);
    assert_eq!(
        visible.state,
        crate::row_histories::RowState::VisibleTransactional
    );

    let history_rows = io.scan_history_row_batches("users", row_id).unwrap();
    assert_eq!(history_rows.len(), 1);
    assert_eq!(history_rows[0].batch_id(), batch_id);
    assert_eq!(
        history_rows[0].state,
        crate::row_histories::RowState::VisibleTransactional
    );
    assert_eq!(history_rows[0].data, second_row.data);

    let outbox = sm.take_outbox();
    assert!(outbox.iter().any(|entry| matches!(
        entry,
        OutboxEntry {
            destination: Destination::Client(id),
            payload: SyncPayload::BatchFate {
                fate: BatchFate::AcceptedTransaction {
                    batch_id: changed_batch_id,
                    confirmed_tier: DurabilityTier::Local,
                },
            },
        } if *id == client_id && *changed_batch_id == batch_id
    )));
    assert!(!outbox.iter().any(|entry| matches!(
        entry,
        OutboxEntry {
            destination: Destination::Client(id),
            payload: SyncPayload::RowBatchNeeded { row, .. },
        } if *id == client_id && row.row_id == row_id
    )));
}

#[test]
fn seal_batch_same_row_preserves_pre_transaction_parent_frontier() {
    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::Local);
    let mut io = MemoryStorage::new();
    let client_id = ClientId::new();
    let batch_id = crate::row_histories::BatchId::new();
    let row_id = ObjectId::new();
    let base_row = visible_row(row_id, "main", Vec::new(), 900, b"base");
    seed_visible_row(&mut sm, &mut io, "users", base_row.clone());

    add_client(&mut sm, &io, client_id);
    sm.set_client_role(client_id, ClientRole::Peer);
    sm.take_outbox();

    let first_row = row_with_batch_state(
        visible_row(row_id, "main", vec![base_row.batch_id()], 1_000, b"alice"),
        batch_id,
        crate::row_histories::RowState::StagingPending,
        None,
    );

    let second_row = row_with_batch_state(
        visible_row(
            row_id,
            "main",
            vec![base_row.batch_id()],
            1_100,
            b"alice-updated",
        ),
        batch_id,
        crate::row_histories::RowState::StagingPending,
        None,
    );

    for row in [first_row.clone(), second_row.clone()] {
        sm.process_from_client(
            &mut io,
            client_id,
            SyncPayload::RowBatchCreated {
                metadata: Some(RowMetadata {
                    id: row.row_id,
                    metadata: row_metadata("users"),
                }),
                row,
            },
        );
    }
    sm.take_outbox();

    sm.process_from_client(
        &mut io,
        client_id,
        SyncPayload::SealBatch {
            submission: transactional_sealed_submission(
                batch_id,
                "main",
                vec![SealedBatchMember {
                    object_id: row_id,
                    row_digest: second_row.content_digest(),
                }],
                vec![CapturedFrontierMember {
                    object_id: row_id,
                    branch_name: BranchName::new("main"),
                    batch_id: base_row.batch_id(),
                }],
            ),
        },
    );

    let visible = io
        .load_visible_region_row("users", "main", row_id)
        .unwrap()
        .expect("sealed batch should publish the accepted row");
    assert_eq!(visible.batch_id(), batch_id);
    assert_eq!(visible.parents.as_slice(), [base_row.batch_id()]);
    assert_eq!(visible.data, second_row.data);
}

#[test]
fn seal_batch_waits_for_all_declared_rows_before_accepting() {
    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::Local);
    let mut io = MemoryStorage::new();
    let client_id = ClientId::new();
    let batch_id = crate::row_histories::BatchId::new();
    let first_row_id = ObjectId::new();
    let second_row_id = ObjectId::new();
    seed_users_schema(&mut io);

    add_client(&mut sm, &io, client_id);
    sm.set_client_role(client_id, ClientRole::Peer);
    sm.take_outbox();

    let first_row = row_with_batch_state(
        visible_row(first_row_id, "main", Vec::new(), 1_000, b"alice"),
        batch_id,
        crate::row_histories::RowState::StagingPending,
        None,
    );
    let second_row = row_with_batch_state(
        visible_row(second_row_id, "main", Vec::new(), 1_100, b"bob"),
        batch_id,
        crate::row_histories::RowState::StagingPending,
        None,
    );
    let first_row_batch_id = first_row.content_digest();
    let second_row_batch_id = second_row.content_digest();

    sm.process_from_client(
        &mut io,
        client_id,
        SyncPayload::RowBatchCreated {
            metadata: Some(RowMetadata {
                id: first_row.row_id,
                metadata: row_metadata("users"),
            }),
            row: first_row,
        },
    );
    sm.take_outbox();

    sm.process_from_client(
        &mut io,
        client_id,
        SyncPayload::SealBatch {
            submission: transactional_sealed_submission(
                batch_id,
                "main",
                vec![
                    SealedBatchMember {
                        object_id: first_row_id,
                        row_digest: first_row_batch_id,
                    },
                    SealedBatchMember {
                        object_id: second_row_id,
                        row_digest: second_row_batch_id,
                    },
                ],
                Vec::new(),
            ),
        },
    );

    assert_eq!(
        io.load_authoritative_batch_fate(batch_id).unwrap(),
        None,
        "authority should wait for all declared rows before settling the batch"
    );
    assert_eq!(
        io.load_visible_region_row("users", "main", first_row_id)
            .unwrap(),
        None,
        "partial sealed batches should not publish visible rows yet"
    );

    sm.process_from_client(
        &mut io,
        client_id,
        SyncPayload::RowBatchCreated {
            metadata: Some(RowMetadata {
                id: second_row.row_id,
                metadata: row_metadata("users"),
            }),
            row: second_row,
        },
    );

    let settlement = io
        .load_authoritative_batch_fate(batch_id)
        .unwrap()
        .expect("authority should settle once all declared rows have arrived");
    let BatchFate::AcceptedTransaction { .. } = settlement else {
        panic!("expected accepted transactional settlement, got {settlement:?}");
    };
}

#[test]
fn seal_batch_waits_for_declared_latest_row_batch_before_accepting() {
    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::Local);
    let mut io = MemoryStorage::new();
    let client_id = ClientId::new();
    let batch_id = crate::row_histories::BatchId::new();
    let row_id = ObjectId::new();
    seed_users_schema(&mut io);

    add_client(&mut sm, &io, client_id);
    sm.set_client_role(client_id, ClientRole::Peer);
    sm.take_outbox();

    let first_row = row_with_batch_state(
        visible_row(row_id, "main", Vec::new(), 1_000, b"alice"),
        batch_id,
        crate::row_histories::RowState::StagingPending,
        None,
    );

    let second_row = row_with_batch_state(
        visible_row(row_id, "main", Vec::new(), 1_100, b"alice-updated"),
        batch_id,
        crate::row_histories::RowState::StagingPending,
        None,
    );

    sm.process_from_client(
        &mut io,
        client_id,
        SyncPayload::RowBatchCreated {
            metadata: Some(RowMetadata {
                id: first_row.row_id,
                metadata: row_metadata("users"),
            }),
            row: first_row.clone(),
        },
    );
    sm.take_outbox();

    sm.process_from_client(
        &mut io,
        client_id,
        SyncPayload::SealBatch {
            submission: transactional_sealed_submission(
                batch_id,
                "main",
                vec![SealedBatchMember {
                    object_id: row_id,
                    row_digest: second_row.content_digest(),
                }],
                Vec::new(),
            ),
        },
    );

    assert_eq!(
        io.load_authoritative_batch_fate(batch_id).unwrap(),
        None,
        "authority should wait for the declared final row batch entry, not just any row for that object"
    );
    assert_eq!(
        io.load_visible_region_row("users", "main", row_id).unwrap(),
        None,
        "earlier staged versions should not become visible just because the object id was declared"
    );

    sm.process_from_client(
        &mut io,
        client_id,
        SyncPayload::RowBatchCreated {
            metadata: Some(RowMetadata {
                id: second_row.row_id,
                metadata: row_metadata("users"),
            }),
            row: second_row.clone(),
        },
    );

    let settlement = io
        .load_authoritative_batch_fate(batch_id)
        .unwrap()
        .expect("authority should settle once the declared final row batch entry arrives");
    let BatchFate::AcceptedTransaction {
        batch_id: settled_batch_id,
        confirmed_tier,
    } = settlement
    else {
        panic!("expected accepted transactional settlement, got {settlement:?}");
    };
    assert_eq!(settled_batch_id, batch_id);
    assert_eq!(confirmed_tier, DurabilityTier::Local);

    let visible = io
        .load_visible_region_row("users", "main", row_id)
        .unwrap()
        .expect("declared final row batch entry should become visible");
    assert_eq!(visible.batch_id(), batch_id);
}

#[test]
fn same_row_staging_in_one_batch_keeps_only_latest_live_pending_member_before_seal() {
    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::Local);
    let mut io = MemoryStorage::new();
    let client_id = ClientId::new();
    let batch_id = crate::row_histories::BatchId::new();
    let row_id = ObjectId::new();
    seed_users_schema(&mut io);

    add_client(&mut sm, &io, client_id);
    sm.set_client_role(client_id, ClientRole::Peer);
    sm.take_outbox();

    let first_row = row_with_batch_state(
        visible_row(row_id, "main", Vec::new(), 1_000, b"alice"),
        batch_id,
        crate::row_histories::RowState::StagingPending,
        None,
    );

    let second_row = row_with_batch_state(
        visible_row(row_id, "main", Vec::new(), 1_100, b"alice-updated"),
        batch_id,
        crate::row_histories::RowState::StagingPending,
        None,
    );

    sm.process_from_client(
        &mut io,
        client_id,
        SyncPayload::RowBatchCreated {
            metadata: Some(RowMetadata {
                id: first_row.row_id,
                metadata: row_metadata("users"),
            }),
            row: first_row.clone(),
        },
    );
    sm.process_from_client(
        &mut io,
        client_id,
        SyncPayload::RowBatchCreated {
            metadata: Some(RowMetadata {
                id: second_row.row_id,
                metadata: row_metadata("users"),
            }),
            row: second_row.clone(),
        },
    );

    let history_rows = io.scan_history_row_batches("users", row_id).unwrap();
    let live_pending_rows: Vec<_> = history_rows
        .iter()
        .filter(|row| matches!(row.state, crate::row_histories::RowState::StagingPending))
        .collect();
    assert_eq!(
        live_pending_rows.len(),
        1,
        "authority staging should keep one live pending member for a same-row batch rewrite"
    );
    assert_eq!(history_rows.len(), 1);
    assert_eq!(live_pending_rows[0].batch_id(), batch_id);
    assert_eq!(live_pending_rows[0].data, second_row.data);
    assert_eq!(
        io.load_visible_region_row("users", "main", row_id).unwrap(),
        None,
        "pre-seal transactional rewrites should remain non-visible"
    );
}

#[test]
fn seal_batch_rejects_members_spanning_multiple_target_branches() {
    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::Local);
    let mut io = MemoryStorage::new();
    let client_id = ClientId::new();
    let batch_id = crate::row_histories::BatchId::new();
    let main_row_id = ObjectId::new();
    let draft_row_id = ObjectId::new();
    seed_users_schema(&mut io);

    add_client(&mut sm, &io, client_id);
    sm.set_client_role(client_id, ClientRole::Peer);
    sm.take_outbox();

    let main_row = row_with_batch_state(
        visible_row(main_row_id, "main", Vec::new(), 1_000, b"alice"),
        batch_id,
        crate::row_histories::RowState::StagingPending,
        None,
    );

    let draft_row = row_with_batch_state(
        visible_row(draft_row_id, "draft", Vec::new(), 1_100, b"bob"),
        batch_id,
        crate::row_histories::RowState::StagingPending,
        None,
    );

    sm.process_from_client(
        &mut io,
        client_id,
        SyncPayload::RowBatchCreated {
            metadata: Some(RowMetadata {
                id: main_row.row_id,
                metadata: row_metadata("users"),
            }),
            row: main_row.clone(),
        },
    );
    sm.process_from_client(
        &mut io,
        client_id,
        SyncPayload::RowBatchCreated {
            metadata: Some(RowMetadata {
                id: draft_row.row_id,
                metadata: row_metadata("users"),
            }),
            row: draft_row.clone(),
        },
    );
    sm.take_outbox();

    sm.process_from_client(
        &mut io,
        client_id,
        SyncPayload::SealBatch {
            submission: sealed_submission(
                batch_id,
                "main",
                vec![
                    SealedBatchMember {
                        object_id: main_row_id,
                        row_digest: main_row.content_digest(),
                    },
                    SealedBatchMember {
                        object_id: draft_row_id,
                        row_digest: draft_row.content_digest(),
                    },
                ],
                Vec::new(),
            ),
        },
    );

    assert_eq!(
        io.load_authoritative_batch_fate(batch_id).unwrap(),
        Some(BatchFate::Rejected {
            batch_id,
            code: "invalid_batch_submission".to_string(),
            reason: "sealed batch rows must belong to the declared target branch".to_string(),
        })
    );
    let withdrawn = sm.take_pending_row_withdrawals();
    assert!(
        withdrawn.contains(&main_row_id) && withdrawn.contains(&draft_row_id),
        "rows of a rejected batch were not reported to whoever keeps something per row: \
         {withdrawn:?}"
    );
    assert_eq!(
        io.load_visible_region_row("users", "main", main_row_id)
            .unwrap(),
        None
    );
    assert_eq!(
        io.load_visible_region_row("users", "draft", draft_row_id)
            .unwrap(),
        None
    );

    let main_history_rows = io.scan_history_row_batches("users", main_row_id).unwrap();
    let draft_history_rows = io.scan_history_row_batches("users", draft_row_id).unwrap();
    assert_eq!(main_history_rows.len(), 1);
    assert_eq!(draft_history_rows.len(), 1);
    assert_eq!(
        main_history_rows[0].state,
        crate::row_histories::RowState::Rejected
    );
    assert_eq!(
        draft_history_rows[0].state,
        crate::row_histories::RowState::Rejected
    );
}

#[test]
fn seal_batch_rejects_when_batch_digest_does_not_match_members() {
    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::Local);
    let mut io = MemoryStorage::new();
    let client_id = ClientId::new();
    let batch_id = crate::row_histories::BatchId::new();
    let row_id = ObjectId::new();
    seed_users_schema(&mut io);

    add_client(&mut sm, &io, client_id);
    sm.set_client_role(client_id, ClientRole::Peer);
    sm.take_outbox();

    let staged_row = row_with_batch_state(
        visible_row(row_id, "main", Vec::new(), 1_000, b"alice"),
        batch_id,
        crate::row_histories::RowState::StagingPending,
        None,
    );

    sm.process_from_client(
        &mut io,
        client_id,
        SyncPayload::RowBatchCreated {
            metadata: Some(RowMetadata {
                id: staged_row.row_id,
                metadata: row_metadata("users"),
            }),
            row: staged_row.clone(),
        },
    );
    sm.take_outbox();

    let mut submission = sealed_submission(
        batch_id,
        "main",
        vec![SealedBatchMember {
            object_id: row_id,
            row_digest: staged_row.content_digest(),
        }],
        Vec::new(),
    );
    submission.batch_digest = crate::digest::Digest32([255; 32]);

    sm.process_from_client(&mut io, client_id, SyncPayload::SealBatch { submission });

    assert_eq!(
        io.load_authoritative_batch_fate(batch_id).unwrap(),
        Some(BatchFate::Rejected {
            batch_id,
            code: "invalid_batch_submission".to_string(),
            reason: "sealed batch digest does not match declared members".to_string(),
        })
    );
    assert_eq!(
        io.load_visible_region_row("users", "main", row_id).unwrap(),
        None,
        "invalid batch digests should be rejected before publication"
    );
}

#[test]
fn seal_batch_acceptance_stops_when_submission_persistence_fails() {
    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::Local);
    let mut io = FailingHistoryPatchStorage::new();
    io.fail_sealed_submission_upsert = true;
    let client_id = ClientId::new();
    let batch_id = crate::row_histories::BatchId::new();
    let row_id = ObjectId::new();
    seed_users_schema(io.inner_mut());

    sm.add_client_with_storage(&io, client_id);
    sm.set_client_role(client_id, ClientRole::Peer);
    sm.take_outbox();

    let staged_row = row_with_batch_state(
        visible_row(row_id, "main", Vec::new(), 1_000, b"alice"),
        batch_id,
        crate::row_histories::RowState::StagingPending,
        None,
    );

    sm.process_from_client(
        &mut io,
        client_id,
        SyncPayload::RowBatchCreated {
            metadata: Some(RowMetadata {
                id: staged_row.row_id,
                metadata: row_metadata("users"),
            }),
            row: staged_row.clone(),
        },
    );
    sm.take_outbox();

    let submission = sealed_submission(
        batch_id,
        "main",
        vec![SealedBatchMember {
            object_id: row_id,
            row_digest: staged_row.content_digest(),
        }],
        Vec::new(),
    );

    sm.process_from_client(&mut io, client_id, SyncPayload::SealBatch { submission });

    assert_eq!(io.load_sealed_batch_submission(batch_id).unwrap(), None);
    assert_eq!(io.load_authoritative_batch_fate(batch_id).unwrap(), None);
    assert_eq!(
        io.load_visible_region_row("users", "main", row_id).unwrap(),
        None,
        "failed sealed-submission persistence should leave the batch unpublished"
    );
    assert!(sm.take_outbox().is_empty());
}

#[test]
fn seal_batch_accepts_when_unrelated_family_visible_frontier_changed() {
    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::Local);
    let mut io = MemoryStorage::new();
    let client_id = ClientId::new();
    let batch_id = crate::row_histories::BatchId::new();
    let existing_row_id = ObjectId::new();
    let conflicting_row_id = ObjectId::new();
    let staged_row_id = ObjectId::new();
    let target_branch = "dev-aaaaaaaaaaaa-main";
    let sibling_branch = "dev-bbbbbbbbbbbb-main";
    seed_users_schema(&mut io);

    add_client(&mut sm, &io, client_id);
    sm.set_client_role(client_id, ClientRole::Peer);
    sm.take_outbox();

    let existing_row = visible_row(existing_row_id, target_branch, Vec::new(), 900, b"seen");
    seed_visible_row(&mut sm, &mut io, "users", existing_row.clone());

    let staged_row = row_with_batch_state(
        visible_row(staged_row_id, target_branch, Vec::new(), 1_000, b"alice"),
        batch_id,
        crate::row_histories::RowState::StagingPending,
        None,
    );

    sm.process_from_client(
        &mut io,
        client_id,
        SyncPayload::RowBatchCreated {
            metadata: Some(RowMetadata {
                id: staged_row.row_id,
                metadata: row_metadata("users"),
            }),
            row: staged_row.clone(),
        },
    );
    sm.take_outbox();

    let conflicting_row = visible_row(
        conflicting_row_id,
        sibling_branch,
        Vec::new(),
        1_050,
        b"bob",
    );
    sm.process_from_client(
        &mut io,
        client_id,
        SyncPayload::RowBatchCreated {
            metadata: Some(RowMetadata {
                id: conflicting_row.row_id,
                metadata: row_metadata("users"),
            }),
            row: conflicting_row,
        },
    );
    sm.take_outbox();

    sm.process_from_client(
        &mut io,
        client_id,
        SyncPayload::SealBatch {
            submission: sealed_submission(
                batch_id,
                target_branch,
                vec![SealedBatchMember {
                    object_id: staged_row_id,
                    row_digest: staged_row.content_digest(),
                }],
                vec![CapturedFrontierMember {
                    object_id: existing_row_id,
                    branch_name: BranchName::new(target_branch),
                    batch_id: existing_row.batch_id(),
                }],
            ),
        },
    );

    assert_eq!(
        io.load_authoritative_batch_fate(batch_id).unwrap(),
        Some(BatchFate::AcceptedTransaction {
            batch_id,
            confirmed_tier: DurabilityTier::Local,
        })
    );
    let visible = io
        .load_visible_region_row("users", target_branch, staged_row_id)
        .unwrap()
        .expect("accepted sealed batch should publish its staged row");
    assert_eq!(
        visible.state,
        crate::row_histories::RowState::VisibleTransactional
    );

    let history_rows = io.scan_history_row_batches("users", staged_row_id).unwrap();
    assert_eq!(history_rows.len(), 1);
    assert_eq!(
        history_rows[0].state,
        crate::row_histories::RowState::VisibleTransactional
    );
}

#[test]
fn seal_batch_accepts_when_family_visible_frontier_matches() {
    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::Local);
    let mut io = MemoryStorage::new();
    let client_id = ClientId::new();
    let batch_id = crate::row_histories::BatchId::new();
    let existing_row_id = ObjectId::new();
    let staged_row_id = ObjectId::new();
    let target_branch = "dev-aaaaaaaaaaaa-main";
    seed_users_schema(&mut io);

    add_client(&mut sm, &io, client_id);
    sm.set_client_role(client_id, ClientRole::Peer);
    sm.take_outbox();

    let existing_row = visible_row(existing_row_id, target_branch, Vec::new(), 900, b"seen");
    seed_visible_row(&mut sm, &mut io, "users", existing_row.clone());

    let staged_row = row_with_batch_state(
        visible_row(staged_row_id, target_branch, Vec::new(), 1_000, b"alice"),
        batch_id,
        crate::row_histories::RowState::StagingPending,
        None,
    );

    sm.process_from_client(
        &mut io,
        client_id,
        SyncPayload::RowBatchCreated {
            metadata: Some(RowMetadata {
                id: staged_row.row_id,
                metadata: row_metadata("users"),
            }),
            row: staged_row.clone(),
        },
    );
    sm.take_outbox();

    sm.process_from_client(
        &mut io,
        client_id,
        SyncPayload::SealBatch {
            submission: sealed_submission(
                batch_id,
                target_branch,
                vec![SealedBatchMember {
                    object_id: staged_row_id,
                    row_digest: staged_row.content_digest(),
                }],
                vec![CapturedFrontierMember {
                    object_id: existing_row_id,
                    branch_name: BranchName::new(target_branch),
                    batch_id: existing_row.batch_id(),
                }],
            ),
        },
    );

    assert!(matches!(
        io.load_authoritative_batch_fate(batch_id).unwrap(),
        Some(BatchFate::AcceptedTransaction {
            batch_id: settled_batch_id,
            confirmed_tier: DurabilityTier::Local,
        }) if settled_batch_id == batch_id
    ));
}

#[test]
fn seal_batch_rejects_when_transactional_row_parent_frontier_changed() {
    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::Local);
    let mut io = MemoryStorage::new();
    let client_id = ClientId::new();
    let batch_id = crate::row_histories::BatchId::new();
    let row_id = ObjectId::new();
    let target_branch = "dev-aaaaaaaaaaaa-main";
    seed_users_schema(&mut io);

    add_client(&mut sm, &io, client_id);
    sm.set_client_role(client_id, ClientRole::Peer);
    sm.take_outbox();

    let base_row = visible_row(row_id, target_branch, Vec::new(), 900, b"base");
    seed_visible_row(&mut sm, &mut io, "users", base_row.clone());

    let staged_row = row_with_batch_state(
        visible_row(
            row_id,
            target_branch,
            vec![base_row.batch_id()],
            1_000,
            b"alice",
        ),
        batch_id,
        crate::row_histories::RowState::StagingPending,
        None,
    );
    sm.process_from_client(
        &mut io,
        client_id,
        SyncPayload::RowBatchCreated {
            metadata: Some(RowMetadata {
                id: staged_row.row_id,
                metadata: row_metadata("users"),
            }),
            row: staged_row.clone(),
        },
    );
    sm.take_outbox();

    let newer_row = visible_row(
        row_id,
        target_branch,
        vec![base_row.batch_id()],
        1_100,
        b"bob",
    );
    io.append_history_region_rows("users", std::slice::from_ref(&newer_row))
        .unwrap();
    io.upsert_visible_region_rows(
        "users",
        std::slice::from_ref(&crate::row_histories::VisibleRowEntry::rebuild(
            newer_row.clone(),
            &[base_row.clone(), newer_row.clone()],
        )),
    )
    .unwrap();

    sm.process_from_client(
        &mut io,
        client_id,
        SyncPayload::SealBatch {
            submission: sealed_submission(
                batch_id,
                target_branch,
                vec![SealedBatchMember {
                    object_id: row_id,
                    row_digest: staged_row.content_digest(),
                }],
                vec![CapturedFrontierMember {
                    object_id: row_id,
                    branch_name: BranchName::new(target_branch),
                    batch_id: newer_row.batch_id(),
                }],
            ),
        },
    );

    assert_eq!(
        io.load_authoritative_batch_fate(batch_id).unwrap(),
        Some(BatchFate::Rejected {
            batch_id,
            code: "transaction_conflict".to_string(),
            reason: "row visible parent changed since transaction write was staged".to_string(),
        })
    );
    assert_eq!(
        io.load_visible_region_row("users", target_branch, row_id)
            .unwrap()
            .expect("newer visible row should remain visible")
            .batch_id(),
        newer_row.batch_id()
    );
}

#[test]
fn seal_batch_accepts_when_transactional_row_parent_frontier_matches() {
    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::Local);
    let mut io = MemoryStorage::new();
    let client_id = ClientId::new();
    let batch_id = crate::row_histories::BatchId::new();
    let row_id = ObjectId::new();
    let target_branch = "dev-aaaaaaaaaaaa-main";
    seed_users_schema(&mut io);

    add_client(&mut sm, &io, client_id);
    sm.set_client_role(client_id, ClientRole::Peer);
    sm.take_outbox();

    let base_row = visible_row(row_id, target_branch, Vec::new(), 900, b"base");
    seed_visible_row(&mut sm, &mut io, "users", base_row.clone());

    let staged_row = row_with_batch_state(
        visible_row(
            row_id,
            target_branch,
            vec![base_row.batch_id()],
            1_000,
            b"alice",
        ),
        batch_id,
        crate::row_histories::RowState::StagingPending,
        None,
    );
    sm.process_from_client(
        &mut io,
        client_id,
        SyncPayload::RowBatchCreated {
            metadata: Some(RowMetadata {
                id: staged_row.row_id,
                metadata: row_metadata("users"),
            }),
            row: staged_row.clone(),
        },
    );
    sm.take_outbox();

    sm.process_from_client(
        &mut io,
        client_id,
        SyncPayload::SealBatch {
            submission: sealed_submission(
                batch_id,
                target_branch,
                vec![SealedBatchMember {
                    object_id: row_id,
                    row_digest: staged_row.content_digest(),
                }],
                vec![CapturedFrontierMember {
                    object_id: row_id,
                    branch_name: BranchName::new(target_branch),
                    batch_id: base_row.batch_id(),
                }],
            ),
        },
    );

    assert!(matches!(
        io.load_authoritative_batch_fate(batch_id).unwrap(),
        Some(BatchFate::AcceptedTransaction {
            batch_id: settled_batch_id,
            confirmed_tier: DurabilityTier::Local,
        }) if settled_batch_id == batch_id
    ));
}

#[test]
fn seal_batch_rejects_transactional_insert_when_row_is_already_visible() {
    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::Local);
    let mut io = MemoryStorage::new();
    let client_id = ClientId::new();
    let batch_id = crate::row_histories::BatchId::new();
    let row_id = ObjectId::new();
    let target_branch = "dev-aaaaaaaaaaaa-main";
    seed_users_schema(&mut io);

    add_client(&mut sm, &io, client_id);
    sm.set_client_role(client_id, ClientRole::Peer);
    sm.take_outbox();

    let existing_row = visible_row(row_id, target_branch, Vec::new(), 900, b"base");
    seed_visible_row(&mut sm, &mut io, "users", existing_row.clone());

    let staged_row = row_with_batch_state(
        visible_row(row_id, target_branch, Vec::new(), 1_000, b"alice"),
        batch_id,
        crate::row_histories::RowState::StagingPending,
        None,
    );
    sm.process_from_client(
        &mut io,
        client_id,
        SyncPayload::RowBatchCreated {
            metadata: Some(RowMetadata {
                id: staged_row.row_id,
                metadata: row_metadata("users"),
            }),
            row: staged_row.clone(),
        },
    );
    sm.take_outbox();

    sm.process_from_client(
        &mut io,
        client_id,
        SyncPayload::SealBatch {
            submission: sealed_submission(
                batch_id,
                target_branch,
                vec![SealedBatchMember {
                    object_id: row_id,
                    row_digest: staged_row.content_digest(),
                }],
                vec![CapturedFrontierMember {
                    object_id: row_id,
                    branch_name: BranchName::new(target_branch),
                    batch_id: existing_row.batch_id(),
                }],
            ),
        },
    );

    assert_eq!(
        io.load_authoritative_batch_fate(batch_id).unwrap(),
        Some(BatchFate::Rejected {
            batch_id,
            code: "transaction_conflict".to_string(),
            reason: "row visible parent changed since transaction write was staged".to_string(),
        })
    );
}

#[test]
fn seal_batch_accepts_transactional_insert_when_row_is_not_visible() {
    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::Local);
    let mut io = MemoryStorage::new();
    let client_id = ClientId::new();
    let batch_id = crate::row_histories::BatchId::new();
    let row_id = ObjectId::new();
    let target_branch = "dev-aaaaaaaaaaaa-main";
    seed_users_schema(&mut io);

    add_client(&mut sm, &io, client_id);
    sm.set_client_role(client_id, ClientRole::Peer);
    sm.take_outbox();

    let staged_row = row_with_batch_state(
        visible_row(row_id, target_branch, Vec::new(), 1_000, b"alice"),
        batch_id,
        crate::row_histories::RowState::StagingPending,
        None,
    );
    sm.process_from_client(
        &mut io,
        client_id,
        SyncPayload::RowBatchCreated {
            metadata: Some(RowMetadata {
                id: staged_row.row_id,
                metadata: row_metadata("users"),
            }),
            row: staged_row.clone(),
        },
    );
    sm.take_outbox();

    sm.process_from_client(
        &mut io,
        client_id,
        SyncPayload::SealBatch {
            submission: transactional_sealed_submission(
                batch_id,
                target_branch,
                vec![SealedBatchMember {
                    object_id: row_id,
                    row_digest: staged_row.content_digest(),
                }],
                Vec::new(),
            ),
        },
    );

    assert!(matches!(
        io.load_authoritative_batch_fate(batch_id).unwrap(),
        Some(BatchFate::AcceptedTransaction {
            batch_id: settled_batch_id,
            confirmed_tier: DurabilityTier::Local,
        }) if settled_batch_id == batch_id
    ));
}

#[test]
fn seal_batch_validates_full_transactional_parent_frontier() {
    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::Local);
    let mut io = MemoryStorage::new();
    let client_id = ClientId::new();
    let batch_id = crate::row_histories::BatchId::new();
    let row_id = ObjectId::new();
    let target_branch = "dev-aaaaaaaaaaaa-main";
    seed_users_schema(&mut io);

    add_client(&mut sm, &io, client_id);
    sm.set_client_role(client_id, ClientRole::Peer);
    sm.take_outbox();

    let base_row = visible_row(row_id, target_branch, Vec::new(), 900, b"base");
    let left_row = visible_row(
        row_id,
        target_branch,
        vec![base_row.batch_id()],
        1_000,
        b"left",
    );
    let right_row = visible_row(
        row_id,
        target_branch,
        vec![base_row.batch_id()],
        1_001,
        b"right",
    );
    create_test_row_with_id(&mut io, row_id, Some(row_metadata("users")));
    io.append_history_region_rows(
        "users",
        &[base_row.clone(), left_row.clone(), right_row.clone()],
    )
    .unwrap();
    io.upsert_visible_region_rows(
        "users",
        std::slice::from_ref(&crate::row_histories::VisibleRowEntry::rebuild(
            right_row.clone(),
            &[base_row.clone(), left_row.clone(), right_row.clone()],
        )),
    )
    .unwrap();
    let current_frontier = io
        .load_visible_region_frontier("users", target_branch, row_id)
        .unwrap()
        .expect("multi-parent visible row should expose a frontier");
    assert_eq!(current_frontier.len(), 2);
    let captured_frontier = io
        .capture_family_visible_frontier(BranchName::new(target_branch))
        .unwrap();

    let staged_row = row_with_batch_state(
        visible_row(
            row_id,
            target_branch,
            current_frontier.iter().rev().copied().collect(),
            1_100,
            b"alice",
        ),
        batch_id,
        crate::row_histories::RowState::StagingPending,
        None,
    );
    sm.process_from_client(
        &mut io,
        client_id,
        SyncPayload::RowBatchCreated {
            metadata: Some(RowMetadata {
                id: staged_row.row_id,
                metadata: row_metadata("users"),
            }),
            row: staged_row.clone(),
        },
    );
    sm.take_outbox();

    sm.process_from_client(
        &mut io,
        client_id,
        SyncPayload::SealBatch {
            submission: sealed_submission(
                batch_id,
                target_branch,
                vec![SealedBatchMember {
                    object_id: row_id,
                    row_digest: staged_row.content_digest(),
                }],
                captured_frontier,
            ),
        },
    );

    assert!(matches!(
        io.load_authoritative_batch_fate(batch_id).unwrap(),
        Some(BatchFate::AcceptedTransaction {
            batch_id: settled_batch_id,
            confirmed_tier: DurabilityTier::Local,
        }) if settled_batch_id == batch_id
    ));
}

#[test]
fn seal_batch_replay_returns_existing_settlement_after_frontier_moves() {
    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::Local);
    let mut io = MemoryStorage::new();
    let client_id = ClientId::new();
    let batch_id = crate::row_histories::BatchId::new();
    let existing_row_id = ObjectId::new();
    let staged_row_id = ObjectId::new();
    let newer_row_id = ObjectId::new();
    let target_branch = "dev-aaaaaaaaaaaa-main";
    let sibling_branch = "dev-bbbbbbbbbbbb-main";
    seed_users_schema(&mut io);

    add_client(&mut sm, &io, client_id);
    sm.set_client_role(client_id, ClientRole::Peer);
    sm.take_outbox();

    let existing_row = visible_row(existing_row_id, target_branch, Vec::new(), 900, b"seen");
    seed_visible_row(&mut sm, &mut io, "users", existing_row.clone());

    let staged_row = row_with_batch_state(
        visible_row(staged_row_id, target_branch, Vec::new(), 1_000, b"alice"),
        batch_id,
        crate::row_histories::RowState::StagingPending,
        None,
    );
    sm.process_from_client(
        &mut io,
        client_id,
        SyncPayload::RowBatchCreated {
            metadata: Some(RowMetadata {
                id: staged_row.row_id,
                metadata: row_metadata("users"),
            }),
            row: staged_row.clone(),
        },
    );
    sm.take_outbox();

    let submission = sealed_submission(
        batch_id,
        target_branch,
        vec![SealedBatchMember {
            object_id: staged_row_id,
            row_digest: staged_row.content_digest(),
        }],
        vec![CapturedFrontierMember {
            object_id: existing_row_id,
            branch_name: BranchName::new(target_branch),
            batch_id: existing_row.batch_id(),
        }],
    );
    sm.process_from_client(
        &mut io,
        client_id,
        SyncPayload::SealBatch {
            submission: submission.clone(),
        },
    );
    sm.take_outbox();

    let accepted_settlement = BatchFate::AcceptedTransaction {
        batch_id,
        confirmed_tier: DurabilityTier::Local,
    };
    assert_eq!(
        io.load_authoritative_batch_fate(batch_id).unwrap(),
        Some(accepted_settlement.clone())
    );

    seed_visible_row(
        &mut sm,
        &mut io,
        "users",
        visible_row(newer_row_id, sibling_branch, Vec::new(), 1_100, b"bob"),
    );

    sm.process_from_client(&mut io, client_id, SyncPayload::SealBatch { submission });

    assert_eq!(
        io.load_authoritative_batch_fate(batch_id).unwrap(),
        Some(accepted_settlement.clone()),
        "replayed seals must be idempotent once the authority has decided the batch"
    );
    let outbox = sm.take_outbox();
    assert!(outbox.iter().any(|entry| matches!(
        entry,
        OutboxEntry {
            destination: Destination::Client(id),
            payload: SyncPayload::BatchFate { fate },
        } if *id == client_id && *fate == accepted_settlement
    )));
}

/// Accepting the parents-blind digest must not weaken the frontier check.
///
/// A delivered row arrives with `parents` stripped, so its digest cannot match
/// the full form this authority stores, and membership matching accepts either
/// (see `delivered_row_reseal.rs`). Membership is only the question of WHICH
/// stored rows a seal is about; whether those rows may commit is a separate
/// question, answered from their real stored parents. This pins that separation:
/// the same conflict must still be rejected when the declaration is blind.
#[test]
fn seal_batch_rejects_a_stale_frontier_declared_with_the_parents_blind_digest() {
    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::Local);
    let mut io = MemoryStorage::new();
    let client_id = ClientId::new();
    let batch_id = crate::row_histories::BatchId::new();
    let row_id = ObjectId::new();
    let target_branch = "dev-aaaaaaaaaaaa-main";
    seed_users_schema(&mut io);

    add_client(&mut sm, &io, client_id);
    sm.set_client_role(client_id, ClientRole::Peer);
    sm.take_outbox();

    let base_row = visible_row(row_id, target_branch, Vec::new(), 900, b"base");
    seed_visible_row(&mut sm, &mut io, "users", base_row.clone());

    let staged_row = row_with_batch_state(
        visible_row(
            row_id,
            target_branch,
            vec![base_row.batch_id()],
            1_000,
            b"alice",
        ),
        batch_id,
        crate::row_histories::RowState::StagingPending,
        None,
    );
    sm.process_from_client(
        &mut io,
        client_id,
        SyncPayload::RowBatchCreated {
            metadata: Some(RowMetadata {
                id: staged_row.row_id,
                metadata: row_metadata("users"),
            }),
            row: staged_row.clone(),
        },
    );
    sm.take_outbox();

    // Someone else moved the row on after the transaction staged its write.
    let newer_row = visible_row(
        row_id,
        target_branch,
        vec![base_row.batch_id()],
        1_100,
        b"bob",
    );
    io.append_history_region_rows("users", std::slice::from_ref(&newer_row))
        .unwrap();
    io.upsert_visible_region_rows(
        "users",
        std::slice::from_ref(&crate::row_histories::VisibleRowEntry::rebuild(
            newer_row.clone(),
            &[base_row.clone(), newer_row.clone()],
        )),
    )
    .unwrap();

    let blind_digest = staged_row.content_digest_ignoring_parents();
    assert_ne!(
        blind_digest,
        staged_row.content_digest(),
        "the staged row has parents, so the two digest forms must differ — otherwise this \
         gates nothing"
    );

    sm.process_from_client(
        &mut io,
        client_id,
        SyncPayload::SealBatch {
            submission: sealed_submission(
                batch_id,
                target_branch,
                vec![SealedBatchMember {
                    object_id: row_id,
                    row_digest: blind_digest,
                }],
                vec![CapturedFrontierMember {
                    object_id: row_id,
                    branch_name: BranchName::new(target_branch),
                    batch_id: newer_row.batch_id(),
                }],
            ),
        },
    );

    assert_eq!(
        io.load_authoritative_batch_fate(batch_id).unwrap(),
        Some(BatchFate::Rejected {
            batch_id,
            code: "transaction_conflict".to_string(),
            reason: "row visible parent changed since transaction write was staged".to_string(),
        }),
        "a blind declaration matched the member and then skipped the frontier check"
    );
    assert_eq!(
        io.load_visible_region_row("users", target_branch, row_id)
            .unwrap()
            .expect("the newer visible row must remain visible")
            .batch_id(),
        newer_row.batch_id()
    );
}

// ---- Who may add rows to a batch and seal it.

fn users_row_of_batch(batch_id: BatchId, author: &str) -> crate::row_histories::StoredRowBatch {
    let mut row = row_with_batch_state(
        visible_row(ObjectId::new(), "main", Vec::new(), 1_000, b"alice"),
        batch_id,
        crate::row_histories::RowState::VisibleDirect,
        None,
    );
    row.created_by = author.into();
    row.updated_by = author.into();
    row
}

fn row_payload(row: &crate::row_histories::StoredRowBatch) -> SyncPayload {
    SyncPayload::RowBatchCreated {
        metadata: Some(RowMetadata {
            id: row.row_id,
            metadata: row_metadata("users"),
        }),
        row: row.clone(),
    }
}

fn user_client(sm: &mut SyncManager, io: &MemoryStorage, user: &str) -> ClientId {
    let client_id = ClientId::new();
    add_client(sm, io, client_id);
    sm.set_client_session(client_id, Session::new(user));
    client_id
}

/// Every queued policy check approved, as a server whose policies allow the rows does.
fn approve_all(sm: &mut SyncManager, io: &mut MemoryStorage) {
    for check in sm.take_pending_permission_checks() {
        sm.approve_permission_check(io, check);
    }
}

/// Alice's row of a direct batch, approved and applied: what the server holds, and
/// serves, before her seal arrives.
fn alices_direct_row(
    sm: &mut SyncManager,
    io: &mut MemoryStorage,
) -> (ClientId, BatchId, crate::row_histories::StoredRowBatch) {
    seed_users_schema(io);
    let alice = user_client(sm, io, "alice");
    let batch_id = BatchId::new();
    let row = users_row_of_batch(batch_id, "alice");
    sm.process_from_client(io, alice, row_payload(&row));
    approve_all(sm, io);
    sm.take_outbox();
    (alice, batch_id, row)
}

fn honest_seal(
    batch_id: BatchId,
    rows: &[&crate::row_histories::StoredRowBatch],
) -> SealedBatchSubmission {
    sealed_submission(
        batch_id,
        "main",
        rows.iter()
            .map(|row| SealedBatchMember {
                object_id: row.row_id,
                row_digest: row.content_digest(),
            })
            .collect(),
        Vec::new(),
    )
}

fn broken(mut submission: SealedBatchSubmission) -> SealedBatchSubmission {
    submission.batch_digest = crate::digest::Digest32([0; 32]);
    submission
}

fn indexed_rows(io: &MemoryStorage, batch_id: BatchId) -> Vec<ObjectId> {
    io.load_local_batch_row_index(batch_id)
        .unwrap()
        .unwrap_or_default()
        .into_iter()
        .map(|member| member.object_id)
        .collect()
}

fn writer_mismatch_for(outbox: &[OutboxEntry], client_id: ClientId, object_id: ObjectId) -> bool {
    outbox.iter().any(|entry| {
        matches!(
            entry,
            OutboxEntry {
                destination: Destination::Client(id),
                payload: SyncPayload::Error(SyncError::PermissionDenied { object_id: denied, code, .. }),
            } if *id == client_id && *denied == object_id && code == "batch_writer_mismatch"
        )
    })
}

/// A row the policies would allow, sent by another user under alice's batch id, is
/// refused alone: it never joins the batch, so alice's seal still covers what she wrote.
#[test]
fn another_users_allowed_row_does_not_join_a_batch() {
    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::GlobalServer);
    let mut io = MemoryStorage::new();
    let (alice, batch_id, row) = alices_direct_row(&mut sm, &mut io);
    let mallory = user_client(&mut sm, &io, "mallory");
    sm.take_outbox();

    let injected = users_row_of_batch(batch_id, "mallory");
    sm.process_from_client(&mut io, mallory, row_payload(&injected));
    assert!(
        sm.take_pending_permission_checks().is_empty(),
        "the stranger's row never reaches a policy check"
    );
    assert!(
        writer_mismatch_for(&sm.take_outbox(), mallory, injected.row_id),
        "the stranger's row is refused alone"
    );
    assert_eq!(
        indexed_rows(&io, batch_id),
        vec![row.row_id],
        "the batch holds alice's row only"
    );

    sm.process_from_client(
        &mut io,
        alice,
        SyncPayload::SealBatch {
            submission: honest_seal(batch_id, &[&row]),
        },
    );
    assert!(
        matches!(
            io.load_authoritative_batch_fate(batch_id).unwrap(),
            Some(BatchFate::DurableDirect { .. })
        ),
        "alice's seal settles her batch: {:?}",
        io.load_authoritative_batch_fate(batch_id)
    );
}

/// On a node that can still promote a lower tier's direct fate: the writer's own broken
/// re-seal leaves the fate as it is, and her next sound seal promotes it.
#[test]
fn the_writers_broken_reseal_leaves_a_promotable_fate() {
    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::EdgeServer);
    let mut io = MemoryStorage::new();
    let (alice, batch_id, row) = alices_direct_row(&mut sm, &mut io);
    let lower = BatchFate::DurableDirect {
        batch_id,
        confirmed_tier: DurabilityTier::Local,
    };
    io.upsert_authoritative_batch_fate(&lower).unwrap();

    sm.process_from_client(
        &mut io,
        alice,
        SyncPayload::SealBatch {
            submission: broken(honest_seal(batch_id, &[&row])),
        },
    );
    assert_eq!(
        io.load_authoritative_batch_fate(batch_id).unwrap(),
        Some(lower)
    );

    sm.process_from_client(
        &mut io,
        alice,
        SyncPayload::SealBatch {
            submission: honest_seal(batch_id, &[&row]),
        },
    );
    assert_eq!(
        io.load_authoritative_batch_fate(batch_id).unwrap(),
        Some(BatchFate::DurableDirect {
            batch_id,
            confirmed_tier: DurabilityTier::EdgeServer,
        }),
        "the writer's sound seal still promotes the batch"
    );
}

/// A batch settled at this node's own tier: the writer's broken re-seal changes nothing and
/// is answered with the fate, so her client stops asking.
#[test]
fn the_writers_broken_reseal_of_a_settled_batch_is_answered_with_its_fate() {
    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::GlobalServer);
    let mut io = MemoryStorage::new();
    let (alice, batch_id, row) = alices_direct_row(&mut sm, &mut io);
    sm.process_from_client(
        &mut io,
        alice,
        SyncPayload::SealBatch {
            submission: honest_seal(batch_id, &[&row]),
        },
    );
    let settled = io.load_authoritative_batch_fate(batch_id).unwrap();
    assert!(
        matches!(settled, Some(BatchFate::DurableDirect { .. })),
        "control: {settled:?}"
    );
    sm.take_outbox();

    sm.process_from_client(
        &mut io,
        alice,
        SyncPayload::SealBatch {
            submission: broken(honest_seal(batch_id, &[&row])),
        },
    );
    assert_eq!(io.load_authoritative_batch_fate(batch_id).unwrap(), settled);
    let settled = settled.unwrap();
    assert!(
        sm.take_outbox().iter().any(|entry| matches!(
            entry,
            OutboxEntry {
                destination: Destination::Client(id),
                payload: SyncPayload::BatchFate { fate },
            } if *id == alice && *fate == settled
        )),
        "the writer is answered with the fate"
    );
}

/// The same node, a stranger's broken seal: the fate stands and the writer's seal promotes.
#[test]
fn another_users_seal_does_not_reject_a_promotable_direct_fate() {
    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::EdgeServer);
    let mut io = MemoryStorage::new();
    let (alice, batch_id, row) = alices_direct_row(&mut sm, &mut io);
    let lower = BatchFate::DurableDirect {
        batch_id,
        confirmed_tier: DurabilityTier::Local,
    };
    io.upsert_authoritative_batch_fate(&lower).unwrap();
    let mallory = user_client(&mut sm, &io, "mallory");
    sm.take_outbox();

    sm.process_from_client(
        &mut io,
        mallory,
        SyncPayload::SealBatch {
            submission: broken(honest_seal(batch_id, &[&row])),
        },
    );
    assert_eq!(
        io.load_authoritative_batch_fate(batch_id).unwrap(),
        Some(lower)
    );
    assert!(sm.take_outbox().is_empty(), "the stranger is not answered");

    sm.process_from_client(
        &mut io,
        alice,
        SyncPayload::SealBatch {
            submission: honest_seal(batch_id, &[&row]),
        },
    );
    assert_eq!(
        io.load_authoritative_batch_fate(batch_id).unwrap(),
        Some(BatchFate::DurableDirect {
            batch_id,
            confirmed_tier: DurabilityTier::EdgeServer,
        })
    );
}

/// A stranger's well-formed seal naming a row that is not in the batch is not stored as
/// the batch's submission, so the next sweep never judges the batch by it.
#[test]
fn another_users_seal_naming_no_row_does_not_replace_the_submission() {
    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::GlobalServer);
    let mut io = MemoryStorage::new();
    let (_alice, batch_id, row) = alices_direct_row(&mut sm, &mut io);
    let mallory = user_client(&mut sm, &io, "mallory");

    let stray = sealed_submission(
        batch_id,
        "main",
        vec![SealedBatchMember {
            object_id: ObjectId::new(),
            row_digest: row.content_digest(),
        }],
        Vec::new(),
    );
    sm.process_from_client(
        &mut io,
        mallory,
        SyncPayload::SealBatch {
            submission: stray.clone(),
        },
    );
    assert_eq!(io.load_sealed_batch_submission(batch_id).unwrap(), None);
    assert_eq!(io.load_authoritative_batch_fate(batch_id).unwrap(), None);
}

/// A batch a peer carried in is the peer's: a user cannot seal it.
#[test]
fn a_user_cannot_seal_a_batch_a_peer_carried() {
    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::GlobalServer);
    let mut io = MemoryStorage::new();
    seed_users_schema(&mut io);
    let peer = ClientId::new();
    add_client(&mut sm, &io, peer);
    sm.set_client_role(peer, ClientRole::Peer);
    let batch_id = BatchId::new();
    let row = users_row_of_batch(batch_id, "alice");
    sm.process_from_client(&mut io, peer, row_payload(&row));
    let mallory = user_client(&mut sm, &io, "mallory");
    sm.take_outbox();

    sm.process_from_client(
        &mut io,
        mallory,
        SyncPayload::SealBatch {
            submission: broken(honest_seal(batch_id, &[&row])),
        },
    );
    assert_eq!(io.load_authoritative_batch_fate(batch_id).unwrap(), None);
    assert_eq!(io.load_sealed_batch_submission(batch_id).unwrap(), None);
    // Nor can alice, whose name the row carries: the peer is who sent it here.
    let alice = user_client(&mut sm, &io, "alice");
    sm.process_from_client(&mut io, alice, row_payload(&row));
    assert!(writer_mismatch_for(&sm.take_outbox(), alice, row.row_id));

    sm.process_from_client(
        &mut io,
        peer,
        SyncPayload::SealBatch {
            submission: honest_seal(batch_id, &[&row]),
        },
    );
    assert!(matches!(
        io.load_authoritative_batch_fate(batch_id).unwrap(),
        Some(BatchFate::DurableDirect { .. })
    ));
}

/// A backend's batch: written and sealed by the backend, out of every user's reach.
#[test]
fn a_backends_batch_is_settled_by_the_backend_alone() {
    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::GlobalServer);
    let mut io = MemoryStorage::new();
    seed_users_schema(&mut io);
    let backend = ClientId::new();
    add_client(&mut sm, &io, backend);
    sm.set_client_role(backend, ClientRole::Backend);
    let batch_id = BatchId::new();
    let row = users_row_of_batch(batch_id, "alice");
    sm.process_from_client(&mut io, backend, row_payload(&row));
    let mallory = user_client(&mut sm, &io, "mallory");
    sm.take_outbox();

    let injected = users_row_of_batch(batch_id, "mallory");
    sm.process_from_client(&mut io, mallory, row_payload(&injected));
    sm.process_from_client(
        &mut io,
        mallory,
        SyncPayload::SealBatch {
            submission: broken(honest_seal(batch_id, &[&row])),
        },
    );
    assert!(sm.take_pending_permission_checks().is_empty());
    assert_eq!(indexed_rows(&io, batch_id), vec![row.row_id]);
    assert_eq!(io.load_authoritative_batch_fate(batch_id).unwrap(), None);

    sm.process_from_client(
        &mut io,
        backend,
        SyncPayload::SealBatch {
            submission: honest_seal(batch_id, &[&row]),
        },
    );
    assert!(matches!(
        io.load_authoritative_batch_fate(batch_id).unwrap(),
        Some(BatchFate::DurableDirect { .. })
    ));
}

/// A refreshed token, and a second connection of the same user: still the writer.
#[test]
fn the_writer_keeps_her_batch_across_a_new_token_and_a_new_connection() {
    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::GlobalServer);
    let mut io = MemoryStorage::new();
    let (alice, batch_id, row) = alices_direct_row(&mut sm, &mut io);
    sm.set_client_session(
        alice,
        Session {
            claims: serde_json::json!({ "exp": 2 }),
            ..Session::new("alice")
        },
    );
    let second = users_row_of_batch(batch_id, "alice");
    sm.process_from_client(&mut io, alice, row_payload(&second));
    assert_eq!(
        sm.take_pending_permission_checks().len(),
        1,
        "her next row is judged"
    );

    let reconnected = user_client(&mut sm, &io, "alice");
    sm.take_outbox();
    sm.process_from_client(&mut io, reconnected, row_payload(&row));
    assert!(
        !writer_mismatch_for(&sm.take_outbox(), reconnected, row.row_id),
        "her replay on a new connection is hers"
    );
    sm.process_from_client(
        &mut io,
        reconnected,
        SyncPayload::SealBatch {
            submission: honest_seal(batch_id, &[&row]),
        },
    );
    assert!(matches!(
        io.load_authoritative_batch_fate(batch_id).unwrap(),
        Some(BatchFate::DurableDirect { .. })
    ));
}

/// A restart keeps the binding: a fresh sync manager over the same store refuses the
/// stranger as the old one did.
#[test]
fn the_binding_outlives_a_restart() {
    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::GlobalServer);
    let mut io = MemoryStorage::new();
    let (_alice, batch_id, _row) = alices_direct_row(&mut sm, &mut io);
    drop(sm);

    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::GlobalServer);
    let mallory = user_client(&mut sm, &io, "mallory");
    sm.take_outbox();
    let injected = users_row_of_batch(batch_id, "mallory");
    sm.process_from_client(&mut io, mallory, row_payload(&injected));
    assert!(writer_mismatch_for(
        &sm.take_outbox(),
        mallory,
        injected.row_id
    ));
}

/// A batch stored before bindings existed is its rows' author's: she can still replay
/// and seal it, a stranger can do neither.
#[test]
fn a_batch_from_before_bindings_belongs_to_its_rows_author() {
    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::GlobalServer);
    let mut io = MemoryStorage::new();
    let batch_id = BatchId::new();
    let row = users_row_of_batch(batch_id, "alice");
    seed_visible_row(&mut sm, &mut io, "users", row.clone());
    assert_eq!(
        io.load_batch_writer(batch_id).unwrap(),
        None,
        "control: no binding"
    );
    assert_eq!(
        indexed_rows(&io, batch_id),
        vec![row.row_id],
        "control: the row is indexed"
    );

    let mallory = user_client(&mut sm, &io, "mallory");
    sm.take_outbox();
    let injected = users_row_of_batch(batch_id, "mallory");
    sm.process_from_client(&mut io, mallory, row_payload(&injected));
    assert!(writer_mismatch_for(
        &sm.take_outbox(),
        mallory,
        injected.row_id
    ));
    sm.process_from_client(
        &mut io,
        mallory,
        SyncPayload::SealBatch {
            submission: broken(honest_seal(batch_id, &[&row])),
        },
    );
    assert_eq!(io.load_authoritative_batch_fate(batch_id).unwrap(), None);
    assert_eq!(
        io.load_batch_writer(batch_id).unwrap(),
        None,
        "the stranger bound nothing"
    );

    let alice = user_client(&mut sm, &io, "alice");
    sm.take_outbox();
    sm.process_from_client(
        &mut io,
        alice,
        SyncPayload::SealBatch {
            submission: honest_seal(batch_id, &[&row]),
        },
    );
    assert_eq!(
        io.load_batch_writer(batch_id).unwrap(),
        Some(crate::batch_fate::BatchWriter::User("alice".to_string()))
    );
    assert!(matches!(
        io.load_authoritative_batch_fate(batch_id).unwrap(),
        Some(BatchFate::DurableDirect { .. })
    ));
}

/// A connection whose session changes to another user mid-batch is that other user from
/// then on: the batch stays the first user's.
#[test]
fn a_new_user_on_the_same_connection_does_not_get_the_batch() {
    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::GlobalServer);
    let mut io = MemoryStorage::new();
    let (connection, batch_id, row) = alices_direct_row(&mut sm, &mut io);
    sm.set_client_session(connection, Session::new("mallory"));

    let next = users_row_of_batch(batch_id, "mallory");
    sm.process_from_client(&mut io, connection, row_payload(&next));
    assert!(sm.take_pending_permission_checks().is_empty());
    assert!(writer_mismatch_for(
        &sm.take_outbox(),
        connection,
        next.row_id
    ));
    sm.process_from_client(
        &mut io,
        connection,
        SyncPayload::SealBatch {
            submission: broken(honest_seal(batch_id, &[&row])),
        },
    );
    assert_eq!(io.load_authoritative_batch_fate(batch_id).unwrap(), None);
    assert_eq!(
        io.load_batch_writer(batch_id).unwrap(),
        Some(crate::batch_fate::BatchWriter::User("alice".to_string()))
    );
}

fn batch_writer_key(batch_id: BatchId) -> String {
    let hex: String = batch_id
        .as_bytes()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    format!("batch:{hex}")
}

/// A binding the store cannot read refuses the user's row with its own code and answers
/// her seal with `Missing`, so she retransmits instead of waiting for a reconnect.
#[test]
fn an_unreadable_binding_refuses_the_user_and_asks_her_again() {
    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::GlobalServer);
    let mut io = MemoryStorage::new();
    let (alice, batch_id, row) = alices_direct_row(&mut sm, &mut io);
    io.raw_table_put("__batch_writer", &batch_writer_key(batch_id), &[9])
        .unwrap();
    assert!(
        io.load_batch_writer(batch_id).is_err(),
        "control: the binding is unreadable"
    );

    let next = users_row_of_batch(batch_id, "alice");
    sm.process_from_client(&mut io, alice, row_payload(&next));
    assert!(sm.take_pending_permission_checks().is_empty());
    assert!(sm.take_outbox().iter().any(|entry| matches!(
        entry,
        OutboxEntry {
            destination: Destination::Client(id),
            payload: SyncPayload::Error(SyncError::PermissionDenied { code, .. }),
        } if *id == alice && code == "batch_writer_unavailable"
    )));

    sm.process_from_client(
        &mut io,
        alice,
        SyncPayload::SealBatch {
            submission: honest_seal(batch_id, &[&row]),
        },
    );
    assert!(sm.take_outbox().iter().any(|entry| matches!(
        entry,
        OutboxEntry {
            destination: Destination::Client(id),
            payload: SyncPayload::BatchFate { fate: BatchFate::Missing { batch_id: missing } },
        } if *id == alice && *missing == batch_id
    )));
    assert_eq!(io.load_authoritative_batch_fate(batch_id).unwrap(), None);
}

/// A batch from before bindings whose index names a row the store no longer holds still
/// belongs to the author of the rows it does hold; with none of them held, nobody is
/// let in.
#[test]
fn a_batch_from_before_bindings_with_a_lost_row_stays_its_authors() {
    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::GlobalServer);
    let mut io = MemoryStorage::new();
    let batch_id = BatchId::new();
    let row = users_row_of_batch(batch_id, "alice");
    seed_visible_row(&mut sm, &mut io, "users", row.clone());
    let mut members = io.load_local_batch_row_index(batch_id).unwrap().unwrap();
    let mut lost = members[0].clone();
    lost.object_id = ObjectId::new();
    members.push(lost.clone());
    io.upsert_local_batch_row_index(batch_id, &members).unwrap();

    let alice = user_client(&mut sm, &io, "alice");
    sm.take_outbox();
    sm.process_from_client(
        &mut io,
        alice,
        SyncPayload::SealBatch {
            submission: honest_seal(batch_id, &[&row]),
        },
    );
    assert_eq!(
        io.load_batch_writer(batch_id).unwrap(),
        Some(crate::batch_fate::BatchWriter::User("alice".to_string())),
        "the lost row does not lock the author out"
    );

    let only_lost = BatchId::new();
    let mut lost_alone = lost;
    lost_alone.object_id = ObjectId::new();
    io.upsert_local_batch_row_index(only_lost, &[lost_alone])
        .unwrap();
    let probe = users_row_of_batch(only_lost, "alice");
    sm.process_from_client(&mut io, alice, row_payload(&probe));
    assert!(sm.take_pending_permission_checks().is_empty());
    assert_eq!(io.load_batch_writer(only_lost).unwrap(), None);
}

/// An edge holding a batch settled at its own tier sends upstream only a re-seal it would
/// itself settle; a well-formed one naming another branch stays here.
#[test]
fn an_edge_forwards_only_a_reseal_it_would_settle() {
    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::EdgeServer);
    let mut io = MemoryStorage::new();
    let upstream = ServerId::new();
    add_server(&mut sm, &io, upstream);
    let (alice, batch_id, row) = alices_direct_row(&mut sm, &mut io);
    sm.process_from_client(
        &mut io,
        alice,
        SyncPayload::SealBatch {
            submission: honest_seal(batch_id, &[&row]),
        },
    );
    assert_eq!(
        io.load_authoritative_batch_fate(batch_id).unwrap(),
        Some(BatchFate::DurableDirect {
            batch_id,
            confirmed_tier: DurabilityTier::EdgeServer,
        }),
        "control"
    );
    sm.take_outbox();
    let forwarded = |outbox: &[OutboxEntry]| {
        outbox.iter().any(|entry| {
            matches!(
                entry,
                OutboxEntry {
                    destination: Destination::Server(id),
                    payload: SyncPayload::SealBatch { submission },
                } if *id == upstream && submission.batch_id == batch_id
            )
        })
    };

    let mut elsewhere = honest_seal(batch_id, &[&row]);
    elsewhere.target_branch_name = BranchName::new("elsewhere");
    sm.process_from_client(
        &mut io,
        alice,
        SyncPayload::SealBatch {
            submission: elsewhere,
        },
    );
    assert!(
        !forwarded(&sm.take_outbox()),
        "a seal naming another branch stays here"
    );

    sm.process_from_client(
        &mut io,
        alice,
        SyncPayload::SealBatch {
            submission: honest_seal(batch_id, &[&row]),
        },
    );
    assert!(
        forwarded(&sm.take_outbox()),
        "a sound re-seal still goes upstream"
    );
}

// ---- Random interleavings of writers and strangers, against a model of the rules.

mod batch_writer_differential {
    use super::*;
    use crate::batch_fate::{BatchMode, BatchWriter};
    use crate::row_histories::{RowState, StoredRowBatch};
    use std::collections::{BTreeMap, HashSet};

    /// SplitMix64, as the other differential gates use.
    struct Rng(u64);

    impl Rng {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        fn below(&mut self, n: usize) -> usize {
            (self.next_u64() % n as u64) as usize
        }
    }

    const USERS: [&str; 3] = ["alice", "bob", "mallory"];

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Role {
        User(&'static str),
        Peer,
        Backend,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Op {
        AllowedRow,
        DeniedRow,
        Replay,
        HonestSeal,
        /// A digest that does not match the members.
        BrokenSeal,
        /// Well formed, for another branch.
        WrongBranch,
        /// Well formed, leaving one of the batch's rows out.
        MemberLeftOut,
        /// Well formed, naming a row the batch does not hold.
        ExtraMember,
    }

    const OPS: [Op; 8] = [
        Op::AllowedRow,
        Op::DeniedRow,
        Op::Replay,
        Op::HonestSeal,
        Op::BrokenSeal,
        Op::WrongBranch,
        Op::MemberLeftOut,
        Op::ExtraMember,
    ];

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Fate {
        None,
        /// A direct fate confirmed below this node's tier, which this node can promote.
        AcceptedBelow,
        Accepted,
        Rejected,
    }

    struct Batch {
        id: BatchId,
        mode: BatchMode,
        writer: Option<BatchWriter>,
        fate: Fate,
    }

    fn claimant(role: Role) -> BatchWriter {
        match role {
            Role::User(user) => BatchWriter::User(user.to_string()),
            Role::Peer | Role::Backend => BatchWriter::Trusted,
        }
    }

    fn connect(sm: &mut SyncManager, io: &MemoryStorage, client_id: ClientId, role: Role) {
        add_client(sm, io, client_id);
        match role {
            Role::User(user) => sm.set_client_session(client_id, Session::new(user)),
            Role::Peer => sm.set_client_role(client_id, ClientRole::Peer),
            Role::Backend => sm.set_client_role(client_id, ClientRole::Backend),
        }
    }

    fn stored_fate(io: &MemoryStorage, batch_id: BatchId, tier: DurabilityTier) -> Fate {
        match io.load_authoritative_batch_fate(batch_id).unwrap() {
            None => Fate::None,
            Some(BatchFate::DurableDirect { confirmed_tier, .. }) if confirmed_tier < tier => {
                Fate::AcceptedBelow
            }
            Some(BatchFate::DurableDirect { .. } | BatchFate::AcceptedTransaction { .. }) => {
                Fate::Accepted
            }
            Some(BatchFate::Rejected { .. }) => Fate::Rejected,
            Some(BatchFate::Missing { .. }) => panic!("a stored Missing"),
        }
    }

    fn stored_rows(io: &MemoryStorage, batch_id: BatchId) -> Vec<StoredRowBatch> {
        io.load_local_batch_row_index(batch_id)
            .unwrap()
            .unwrap_or_default()
            .into_iter()
            .filter_map(|member| {
                io.load_history_row_batch_for_schema_hash(
                    member.table_name.as_str(),
                    member.schema_hash,
                    member.branch_name.as_str(),
                    member.object_id,
                    batch_id,
                )
                .unwrap()
            })
            .collect()
    }

    /// A new row of `batch` by `author`: staged in a transactional batch, visible in a
    /// direct one.
    fn row_of(batch: &Batch, author: &str) -> StoredRowBatch {
        let state = match batch.mode {
            BatchMode::Direct => RowState::VisibleDirect,
            BatchMode::Transactional => RowState::StagingPending,
        };
        let mut row = row_with_batch_state(
            visible_row(ObjectId::new(), "main", Vec::new(), 1_000, b"alice"),
            batch.id,
            state,
            None,
        );
        row.created_by = author.into();
        row.updated_by = author.into();
        row
    }

    fn members_of(rows: &[StoredRowBatch]) -> Vec<SealedBatchMember> {
        rows.iter()
            .map(|row| SealedBatchMember {
                object_id: row.row_id,
                row_digest: row.content_digest(),
            })
            .collect()
    }

    /// The model's admission: a trusted connection always; a user if she holds the batch,
    /// or nobody does and every row it holds is hers. Only a row or a seal that validates
    /// (`binds`) may bind; a seal that does not decides nothing for a batch nobody holds
    /// and that holds nothing.
    fn admit(batch: &mut Batch, role: Role, stored: &[StoredRowBatch], binds: bool) -> bool {
        let claimant = claimant(role);
        match (&batch.writer, role) {
            (Some(_), Role::Peer | Role::Backend) => true,
            (Some(writer), Role::User(_)) => *writer == claimant,
            (None, Role::Peer | Role::Backend) => {
                if binds && stored.is_empty() {
                    batch.writer = Some(claimant);
                }
                true
            }
            (None, Role::User(_)) if stored.is_empty() && !binds => false,
            (None, Role::User(user)) => {
                let admitted = stored.iter().all(|row| row.updated_by.as_str() == user);
                if admitted && binds {
                    batch.writer = Some(claimant);
                }
                admitted
            }
        }
    }

    /// The model's fate after an admitted step. A batch with no fate is decided by its
    /// writer's denied row or by a seal of its rows; a fate below this node is promoted by
    /// a seal that settles, and nothing else; any other fate stands.
    fn next_fate(fate: Fate, op: Op, role: Role, stored: &[StoredRowBatch]) -> Fate {
        let names_rows = !stored.is_empty();
        match (fate, op) {
            (Fate::None, Op::DeniedRow) if matches!(role, Role::User(_)) => Fate::Rejected,
            (Fate::None | Fate::AcceptedBelow, Op::HonestSeal | Op::MemberLeftOut)
                if names_rows =>
            {
                Fate::Accepted
            }
            (Fate::None, Op::BrokenSeal | Op::WrongBranch) if names_rows => Fate::Rejected,
            (fate, _) => fate,
        }
    }

    fn run(seed: u64, steps: usize, tier: DurabilityTier) {
        let mut rng = Rng(seed);
        let mut sm = SyncManager::new().with_durability_tier(tier);
        let mut io = MemoryStorage::new();
        seed_users_schema(&mut io);

        // Two connections for alice, one for everyone else.
        let connections: Vec<(ClientId, Role)> = [
            Role::User("alice"),
            Role::User("bob"),
            Role::User("mallory"),
            Role::User("alice"),
            Role::Peer,
            Role::Backend,
        ]
        .into_iter()
        .map(|role| (ClientId::new(), role))
        .collect();
        for &(client_id, role) in &connections {
            connect(&mut sm, &io, client_id, role);
        }

        let mut batches: Vec<Batch> = [
            BatchMode::Direct,
            BatchMode::Direct,
            BatchMode::Transactional,
            BatchMode::Transactional,
        ]
        .into_iter()
        .map(|mode| Batch {
            id: BatchId::new(),
            mode,
            writer: None,
            fate: Fate::None,
        })
        .collect();
        // A direct batch stored before bindings existed, written by alice.
        let legacy = BatchId::new();
        seed_visible_row(
            &mut sm,
            &mut io,
            "users",
            users_row_of_batch(legacy, "alice"),
        );
        batches.push(Batch {
            id: legacy,
            mode: BatchMode::Direct,
            writer: None,
            fate: Fate::None,
        });
        if tier == DurabilityTier::EdgeServer {
            // Batches settled below this node, which a seal that settles promotes: the one
            // from before bindings, and one alice wrote through this node.
            let relayed = BatchId::new();
            seed_visible_row(
                &mut sm,
                &mut io,
                "users",
                users_row_of_batch(relayed, "alice"),
            );
            let alice = BatchWriter::User("alice".to_string());
            io.put_batch_writer(relayed, &alice).unwrap();
            batches.push(Batch {
                id: relayed,
                mode: BatchMode::Direct,
                writer: Some(alice),
                fate: Fate::None,
            });
            for batch in batches
                .iter_mut()
                .filter(|batch| batch.id == legacy || batch.id == relayed)
            {
                io.upsert_authoritative_batch_fate(&BatchFate::DurableDirect {
                    batch_id: batch.id,
                    confirmed_tier: DurabilityTier::Local,
                })
                .unwrap();
                batch.fate = Fate::AcceptedBelow;
            }
        }

        let mut denied_rows = HashSet::new();
        // Every row an admitted connection sent, by batch: the batch may hold only these
        // (and the seeded rows).
        let mut admitted_rows: BTreeMap<BatchId, HashSet<ObjectId>> = BTreeMap::new();
        for batch in &batches {
            admitted_rows
                .entry(batch.id)
                .or_default()
                .extend(stored_rows(&io, batch.id).iter().map(|row| row.row_id));
        }
        let mut sent: Vec<(ClientId, StoredRowBatch)> = Vec::new();
        sm.take_outbox();

        for step in 0..steps {
            if rng.below(30) == 0 {
                // A restart: a fresh node over the same store, every connection back.
                sm = SyncManager::new().with_durability_tier(tier);
                for &(client_id, role) in &connections {
                    connect(&mut sm, &io, client_id, role);
                }
                sm.take_outbox();
                continue;
            }
            if rng.below(15) == 0 {
                // A new batch, so the time before a batch has a fate keeps coming back.
                batches.push(Batch {
                    id: BatchId::new(),
                    mode: [BatchMode::Direct, BatchMode::Transactional][rng.below(2)],
                    writer: None,
                    fate: Fate::None,
                });
                continue;
            }
            let (client_id, role) = connections[rng.below(connections.len())];
            // Half the steps go to the newest batch, which is the likeliest to have no fate.
            let index = if rng.below(2) == 0 {
                batches.len() - 1
            } else {
                rng.below(batches.len())
            };
            let op = OPS[rng.below(OPS.len())];
            let batch_id = batches[index].id;
            let stored = stored_rows(&io, batch_id);
            let context =
                format!("seed {seed:#x} {tier:?} step {step}: {role:?} {op:?} on batch {index}");

            let payload = match op {
                Op::AllowedRow | Op::DeniedRow => {
                    let author = match role {
                        Role::User(user) => user,
                        Role::Peer | Role::Backend => USERS[rng.below(USERS.len())],
                    };
                    let row = row_of(&batches[index], author);
                    if op == Op::DeniedRow {
                        denied_rows.insert(row.row_id);
                    }
                    sent.push((client_id, row.clone()));
                    row_payload(&row)
                }
                Op::Replay => {
                    let mine: Vec<_> = sent
                        .iter()
                        .filter(|(sender, row)| *sender == client_id && row.batch_id == batch_id)
                        .map(|(_, row)| row.clone())
                        .collect();
                    if mine.is_empty() {
                        continue;
                    }
                    row_payload(&mine[rng.below(mine.len())])
                }
                Op::HonestSeal
                | Op::BrokenSeal
                | Op::WrongBranch
                | Op::MemberLeftOut
                | Op::ExtraMember => {
                    let mut members = members_of(&stored);
                    match op {
                        Op::MemberLeftOut if members.len() < 2 => continue,
                        Op::MemberLeftOut => {
                            members.remove(rng.below(members.len()));
                        }
                        Op::ExtraMember => members.push(SealedBatchMember {
                            object_id: ObjectId::new(),
                            row_digest: crate::digest::Digest32([7; 32]),
                        }),
                        _ => {}
                    }
                    let branch = if op == Op::WrongBranch {
                        "elsewhere"
                    } else {
                        "main"
                    };
                    let submission = sealed_submission_with_mode(
                        batch_id,
                        batches[index].mode,
                        branch,
                        members,
                        Vec::new(),
                    );
                    SyncPayload::SealBatch {
                        submission: if op == Op::BrokenSeal {
                            broken(submission)
                        } else {
                            submission
                        },
                    }
                }
            };
            let is_row = matches!(payload, SyncPayload::RowBatchCreated { .. });
            let object_id = match &payload {
                SyncPayload::RowBatchCreated { row, .. } => Some(row.row_id),
                _ => None,
            };

            // The model, before the step runs.
            let batch = &mut batches[index];
            let binds = match op {
                Op::AllowedRow | Op::DeniedRow | Op::Replay => true,
                Op::HonestSeal | Op::WrongBranch => !stored.is_empty(),
                Op::MemberLeftOut | Op::ExtraMember => true,
                Op::BrokenSeal => false,
            };
            let admitted = admit(batch, role, &stored, binds);
            if admitted && let Some(object_id) = object_id {
                admitted_rows.entry(batch_id).or_default().insert(object_id);
            }
            if admitted {
                batch.fate = next_fate(batch.fate, op, role, &stored);
            }

            // The system.
            sm.process_from_client(&mut io, client_id, payload);
            for check in sm.take_pending_permission_checks() {
                let denied = match &check.payload {
                    SyncPayload::RowBatchCreated { row, .. } => denied_rows.contains(&row.row_id),
                    _ => false,
                };
                if denied {
                    sm.reject_permission_check(&mut io, check, "denied by the model".to_string());
                } else {
                    sm.approve_permission_check(&mut io, check);
                }
            }
            let outbox = sm.take_outbox();

            let batch = &batches[index];
            assert_eq!(
                io.load_batch_writer(batch_id).unwrap(),
                batch.writer,
                "{context}: binding"
            );
            for batch in &batches {
                assert_eq!(
                    stored_fate(&io, batch.id, tier),
                    batch.fate,
                    "{context}: fate of {:?}",
                    batch.id
                );
                let allowed = admitted_rows.get(&batch.id).cloned().unwrap_or_default();
                for row in stored_rows(&io, batch.id) {
                    assert!(
                        allowed.contains(&row.row_id),
                        "{context}: batch {:?} holds a row no admitted connection sent",
                        batch.id
                    );
                }
            }
            if !admitted {
                let to_stranger: Vec<_> = outbox
                    .iter()
                    .filter(|entry| entry.destination == Destination::Client(client_id))
                    .collect();
                if is_row {
                    assert!(
                        writer_mismatch_for(&outbox, client_id, object_id.unwrap()),
                        "{context}: the stranger's row is refused alone: {to_stranger:?}"
                    );
                    assert_eq!(to_stranger.len(), 1, "{context}: {to_stranger:?}");
                } else {
                    assert!(
                        to_stranger.is_empty(),
                        "{context}: the stranger's seal is answered: {to_stranger:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn random_interleavings_of_writers_and_strangers_match_the_model() {
        for seed in 0..100u64 {
            run(
                0xA1A1_0000_0000_0000 ^ seed,
                300,
                DurabilityTier::GlobalServer,
            );
        }
    }

    /// The same on an edge, where batches settled below it are promoted by a seal that
    /// settles and by nothing else.
    #[test]
    fn random_interleavings_on_an_edge_holding_lower_fates_match_the_model() {
        for seed in 0..100u64 {
            run(
                0xA1A2_0000_0000_0000 ^ seed,
                300,
                DurabilityTier::EdgeServer,
            );
        }
    }
}
