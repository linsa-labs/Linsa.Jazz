//! A client that is its own authority at the local tier, as an app's client is, accepts its
//! sealed transaction there and sends the rows again, accepted, before its server has
//! decided the batch. Measured 11-05 on the stand (a jazz-wasm client, a v29.2 server):
//! the server stored those rows as sent and served them, and its sweep then rejected the
//! batch for holding rows neither direct nor staged, with nobody told.

use super::*;
use crate::row_histories::{RowState, StoredRowBatch};

const BRANCH: &str = "main";

fn server_with_alice() -> (SyncManager, MemoryStorage, ClientId) {
    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::GlobalServer);
    let mut io = MemoryStorage::new();
    seed_users_schema(&mut io);
    let alice = user_client(&mut sm, &io, "alice");
    sm.take_outbox();
    (sm, io, alice)
}

fn user_client(sm: &mut SyncManager, io: &MemoryStorage, user: &str) -> ClientId {
    let client_id = ClientId::new();
    add_client(sm, io, client_id);
    sm.set_client_session(client_id, Session::new(user));
    client_id
}

/// Alice's staged row of a transactional batch.
fn staged_row(batch_id: BatchId) -> StoredRowBatch {
    let mut row = row_with_batch_state(
        visible_row(ObjectId::new(), BRANCH, Vec::new(), 1_000, b"alice"),
        batch_id,
        RowState::StagingPending,
        None,
    );
    row.created_by = "alice".into();
    row.updated_by = "alice".into();
    row
}

/// The row as her client sends it again once it accepted the batch at its own tier.
fn accepted_locally(staged: &StoredRowBatch) -> StoredRowBatch {
    staged.accepted_transaction_output(DurabilityTier::Local)
}

fn sent(row: &StoredRowBatch) -> SyncPayload {
    SyncPayload::RowBatchCreated {
        metadata: Some(RowMetadata {
            id: row.row_id,
            metadata: row_metadata("users"),
        }),
        row: row.clone(),
    }
}

fn sent_again(row: &StoredRowBatch) -> SyncPayload {
    SyncPayload::RowBatchNeeded {
        metadata: Some(RowMetadata {
            id: row.row_id,
            metadata: row_metadata("users"),
        }),
        row: row.clone(),
    }
}

fn seal_of(batch_id: BatchId, rows: &[&StoredRowBatch]) -> SealedBatchSubmission {
    transactional_sealed_submission(
        batch_id,
        BRANCH,
        rows.iter()
            .map(|row| SealedBatchMember {
                object_id: row.row_id,
                row_digest: row.content_digest(),
            })
            .collect(),
        Vec::new(),
    )
}

fn approve_all(sm: &mut SyncManager, io: &mut MemoryStorage) {
    for check in sm.take_pending_permission_checks() {
        sm.approve_permission_check(io, check);
    }
}

fn stored(io: &MemoryStorage, row: &StoredRowBatch) -> StoredRowBatch {
    io.load_history_row_batch("users", BRANCH, row.row_id, row.batch_id)
        .unwrap()
        .expect("the row is stored")
}

fn served(io: &MemoryStorage, row: &StoredRowBatch) -> Option<StoredRowBatch> {
    io.load_visible_region_row("users", BRANCH, row.row_id)
        .unwrap()
}

fn told(outbox: &[OutboxEntry], client_id: ClientId, batch_id: BatchId) -> Vec<BatchFate> {
    outbox
        .iter()
        .filter_map(|entry| match entry {
            OutboxEntry {
                destination: Destination::Client(id),
                payload: SyncPayload::BatchFate { fate },
            } if *id == client_id && fate.batch_id() == batch_id => Some(fate.clone()),
            _ => None,
        })
        .collect()
}

/// Taken in before its batch is decided, the row is held as the staged row it was: the
/// client's acceptance is its own, and a row served before its batch is decided breaks the
/// batch's all or nothing.
#[test]
fn a_row_its_client_accepted_is_held_as_staged_until_the_server_decides() {
    let (mut sm, mut io, alice) = server_with_alice();
    let batch_id = BatchId::new();
    let row = accepted_locally(&staged_row(batch_id));

    sm.process_from_client(&mut io, alice, sent_again(&row));
    approve_all(&mut sm, &mut io);

    let held = stored(&io, &row);
    assert_eq!(held.state, RowState::StagingPending);
    assert_eq!(held.confirmed_tier, None);
    assert!(
        served(&io, &row).is_none(),
        "a row of an undecided batch is not served"
    );
}

/// The rows, the seal, and the copies the client sends once it accepted the batch, in one
/// exchange as an app's client sends them: the batch is accepted on its rows, and the
/// copies checked after that leave its rows served as the server accepted them.
#[test]
fn the_copies_of_an_accepted_transaction_leave_its_rows_served() {
    let (mut sm, mut io, alice) = server_with_alice();
    let batch_id = BatchId::new();
    let row = staged_row(batch_id);

    sm.process_from_client(&mut io, alice, sent(&row));
    sm.process_from_client(
        &mut io,
        alice,
        SyncPayload::SealBatch {
            submission: seal_of(batch_id, &[&row]),
        },
    );
    sm.process_from_client(&mut io, alice, sent_again(&accepted_locally(&row)));
    approve_all(&mut sm, &mut io);

    assert_eq!(
        io.load_authoritative_batch_fate(batch_id).unwrap(),
        Some(BatchFate::AcceptedTransaction {
            batch_id,
            confirmed_tier: DurabilityTier::GlobalServer,
        })
    );
    // The history first: a copy applied over the accepted row turns it back to staged
    // there while the visible region still holds the accepted version, until the next
    // rebuild of the row drops it.
    let held = stored(&io, &row);
    assert_eq!(
        (held.state, held.confirmed_tier),
        (
            RowState::VisibleTransactional,
            Some(DurabilityTier::GlobalServer)
        ),
        "the copy does not stand in for the server's acceptance"
    );
    let served = served(&io, &row).expect("the accepted row is served");
    assert_eq!(served.batch_id, batch_id);
    assert_eq!(served.state, RowState::VisibleTransactional);
    assert!(
        told(&sm.take_outbox(), alice, batch_id)
            .iter()
            .all(|fate| matches!(fate, BatchFate::AcceptedTransaction { .. })),
        "alice is told only that her batch is accepted"
    );
}

/// After the server accepted the batch, a copy the client sends again (a reconnect) is the
/// row the server holds: a replay, never checked again.
#[test]
fn a_copy_sent_again_after_the_server_accepted_is_a_replay() {
    let (mut sm, mut io, alice) = server_with_alice();
    let batch_id = BatchId::new();
    let row = staged_row(batch_id);
    sm.process_from_client(&mut io, alice, sent(&row));
    sm.process_from_client(
        &mut io,
        alice,
        SyncPayload::SealBatch {
            submission: seal_of(batch_id, &[&row]),
        },
    );
    approve_all(&mut sm, &mut io);
    assert_eq!(stored(&io, &row).state, RowState::VisibleTransactional);
    sm.take_outbox();

    sm.process_from_client(&mut io, alice, sent_again(&accepted_locally(&row)));
    assert!(
        sm.take_pending_permission_checks().is_empty(),
        "a replay is not checked again"
    );
    assert!(served(&io, &row).is_some());
}

/// While the batch waits for a row, a copy of a staged row it holds is that row: a replay.
#[test]
fn a_copy_of_a_staged_row_of_an_undecided_batch_is_a_replay() {
    let (mut sm, mut io, alice) = server_with_alice();
    let batch_id = BatchId::new();
    let row = staged_row(batch_id);
    let late = staged_row(batch_id);
    sm.process_from_client(&mut io, alice, sent(&row));
    sm.process_from_client(
        &mut io,
        alice,
        SyncPayload::SealBatch {
            submission: seal_of(batch_id, &[&row, &late]),
        },
    );
    approve_all(&mut sm, &mut io);
    assert_eq!(io.load_authoritative_batch_fate(batch_id).unwrap(), None);
    assert_eq!(stored(&io, &row).state, RowState::StagingPending);

    sm.process_from_client(&mut io, alice, sent_again(&accepted_locally(&row)));
    assert!(
        sm.take_pending_permission_checks().is_empty(),
        "a replay is not checked again"
    );

    sm.process_from_client(&mut io, alice, sent_again(&accepted_locally(&late)));
    approve_all(&mut sm, &mut io);
    assert!(matches!(
        io.load_authoritative_batch_fate(batch_id).unwrap(),
        Some(BatchFate::AcceptedTransaction { .. })
    ));
    assert!(served(&io, &row).is_some() && served(&io, &late).is_some());
}

/// A seal stored with its rows but not decided, as a crash between the two leaves it, is
/// decided by the sweep with no client's frame to answer: the writer waiting on the batch
/// is told all the same, whether it is accepted or rejected.
#[test]
fn a_batch_the_sweep_decides_is_told_to_its_writer() {
    for conflict in [false, true] {
        let (mut sm, mut io, alice) = server_with_alice();
        let batch_id = BatchId::new();
        let base = visible_row(ObjectId::new(), BRANCH, Vec::new(), 900, b"base");
        seed_visible_row(&mut sm, &mut io, "users", base.clone());
        let mut row = row_with_batch_state(
            visible_row(base.row_id, BRANCH, vec![base.batch_id()], 1_000, b"alice"),
            batch_id,
            RowState::StagingPending,
            None,
        );
        row.created_by = "alice".into();
        row.updated_by = "alice".into();
        sm.process_from_client(&mut io, alice, sent(&row));
        approve_all(&mut sm, &mut io);
        let mut frontier = base.batch_id();
        if conflict {
            // Another write lands on the row the transaction updates before it is decided.
            let newer = visible_row(base.row_id, BRANCH, vec![base.batch_id()], 1_100, b"bob");
            io.append_history_region_rows("users", std::slice::from_ref(&newer))
                .unwrap();
            io.upsert_visible_region_rows(
                "users",
                std::slice::from_ref(&VisibleRowEntry::rebuild(
                    newer.clone(),
                    &[base.clone(), newer.clone()],
                )),
            )
            .unwrap();
            frontier = newer.batch_id();
        }
        let seal = transactional_sealed_submission(
            batch_id,
            BRANCH,
            vec![SealedBatchMember {
                object_id: row.row_id,
                row_digest: row.content_digest(),
            }],
            vec![CapturedFrontierMember {
                object_id: row.row_id,
                branch_name: BranchName::new(BRANCH),
                batch_id: frontier,
            }],
        );
        io.upsert_sealed_batch_submission(&seal).unwrap();
        sm.take_outbox();

        sm.recover_completed_sealed_batches_with_storage(&mut io);

        let fate = io
            .load_authoritative_batch_fate(batch_id)
            .unwrap()
            .expect("the sweep decides the batch");
        assert_eq!(
            matches!(fate, BatchFate::Rejected { ref code, .. } if code == "transaction_conflict"),
            conflict,
            "{fate:?}"
        );
        assert_eq!(
            told(&sm.take_outbox(), alice, batch_id),
            vec![fate],
            "the writer is told what the sweep decided"
        );
    }
}

/// A node that is its own authority at the local tier with no server above it (a browser's
/// worker, offline) decides its client's transaction there and keeps the seal, for the
/// server it reaches later to decide the batch with. Neither the decision nor a row the
/// client sends again lets the seal go.
#[test]
fn a_transaction_decided_below_a_global_server_keeps_its_seal() {
    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::Local);
    let mut io = MemoryStorage::new();
    seed_users_schema(&mut io);
    let alice = user_client(&mut sm, &io, "alice");
    sm.take_outbox();
    let batch_id = BatchId::new();
    let row = staged_row(batch_id);
    sm.process_from_client(&mut io, alice, sent(&row));
    sm.process_from_client(
        &mut io,
        alice,
        SyncPayload::SealBatch {
            submission: seal_of(batch_id, &[&row]),
        },
    );
    approve_all(&mut sm, &mut io);
    assert_eq!(
        io.load_authoritative_batch_fate(batch_id).unwrap(),
        Some(BatchFate::AcceptedTransaction {
            batch_id,
            confirmed_tier: DurabilityTier::Local,
        })
    );
    assert!(
        io.load_sealed_batch_submission(batch_id).unwrap().is_some(),
        "decided here, the seal stays"
    );

    sm.process_from_client(&mut io, alice, sent(&row));
    approve_all(&mut sm, &mut io);
    assert!(
        io.load_sealed_batch_submission(batch_id).unwrap().is_some(),
        "a row sent again lets the seal go no more than the decision"
    );
}

fn backend_client(sm: &mut SyncManager, io: &MemoryStorage) -> ClientId {
    let client_id = ClientId::new();
    add_client(sm, io, client_id);
    sm.set_client_role(client_id, ClientRole::Backend);
    client_id
}

/// A backend's rows are trusted, but a transaction it accepted at its own tier is still the
/// server's to decide: the rows it sends again, accepted, are held as the staged rows they
/// stand for, and a second copy of one is that staged row, a replay forwarded upstream once.
#[test]
fn a_backend_copy_of_a_row_the_server_holds_staged_is_a_replay() {
    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::EdgeServer);
    let mut io = MemoryStorage::new();
    seed_users_schema(&mut io);
    let upstream = ServerId::new();
    add_server(&mut sm, &io, upstream);
    let backend = backend_client(&mut sm, &io);
    sm.take_outbox();
    let batch_id = BatchId::new();
    let row = accepted_locally(&staged_row(batch_id));

    sm.process_from_client(&mut io, backend, sent_again(&row));
    let held = stored(&io, &row);
    assert_eq!(
        (held.state, held.confirmed_tier),
        (RowState::StagingPending, None)
    );
    sm.process_from_client(&mut io, backend, sent_again(&row));

    let forwarded = sm
        .take_outbox()
        .iter()
        .filter(|entry| match entry {
            OutboxEntry {
                destination: Destination::Server(id),
                payload:
                    SyncPayload::RowBatchCreated { row: sent, .. }
                    | SyncPayload::RowBatchNeeded { row: sent, .. },
            } => *id == upstream && sent.row_id == row.row_id && sent.batch_id == batch_id,
            _ => false,
        })
        .count();
    assert_eq!(
        forwarded, 1,
        "the second copy is the staged row the server holds"
    );
}

/// A backend that settled its transaction at its own tier and missed the server's rejection
/// sends its rows again, accepted. The batch is decided: the copy leaves its row rejected,
/// unserved, and staged no more. The store's own guard for rows of a rejected batch
/// (`try_apply_row_updated`) holds this as well as the trusted arm does.
#[test]
fn a_backend_copy_of_a_rejected_transaction_leaves_it_rejected() {
    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::GlobalServer);
    let mut io = MemoryStorage::new();
    seed_users_schema(&mut io);
    let backend = backend_client(&mut sm, &io);
    sm.take_outbox();
    let batch_id = BatchId::new();
    let base = visible_row(ObjectId::new(), BRANCH, Vec::new(), 900, b"base");
    seed_visible_row(&mut sm, &mut io, "users", base.clone());
    let row = row_with_batch_state(
        visible_row(
            base.row_id,
            BRANCH,
            vec![base.batch_id()],
            1_000,
            b"backend",
        ),
        batch_id,
        RowState::StagingPending,
        None,
    );
    sm.process_from_client(&mut io, backend, sent(&row));
    // Another write lands on the row the transaction updates before it is decided.
    let newer = visible_row(base.row_id, BRANCH, vec![base.batch_id()], 1_100, b"bob");
    io.append_history_region_rows("users", std::slice::from_ref(&newer))
        .unwrap();
    io.upsert_visible_region_rows(
        "users",
        std::slice::from_ref(&VisibleRowEntry::rebuild(
            newer.clone(),
            &[base.clone(), newer.clone()],
        )),
    )
    .unwrap();
    sm.process_from_client(
        &mut io,
        backend,
        SyncPayload::SealBatch {
            submission: seal_of(batch_id, &[&row]),
        },
    );
    assert!(
        matches!(
            io.load_authoritative_batch_fate(batch_id).unwrap(),
            Some(BatchFate::Rejected { ref code, .. }) if code == "transaction_conflict"
        ),
        "{:?}",
        io.load_authoritative_batch_fate(batch_id).unwrap()
    );
    let rejected = stored(&io, &row).state;
    assert_ne!(rejected, RowState::StagingPending);

    sm.process_from_client(&mut io, backend, sent_again(&accepted_locally(&row)));

    assert!(matches!(
        io.load_authoritative_batch_fate(batch_id).unwrap(),
        Some(BatchFate::Rejected { .. })
    ));
    assert_eq!(
        stored(&io, &row).state,
        rejected,
        "a copy of a decided batch's row does not stage it again"
    );
    assert_ne!(
        served(&io, &row).map(|served| served.batch_id),
        Some(batch_id),
        "a rejected row is not served"
    );
}

/// A backend holds its accepted row as the server delivered it back, without its parents
/// (`scope_delivery_row`), and sends that copy again: no replay of the row the server holds,
/// and of a batch the server decided. The row stays accepted and served. A belt: a copy
/// delivered from a batch accepted here carries this server's tier, so the writer settles
/// the batch with it and has nothing left to send. Only a row an older server sent with no
/// tier, or an odd peer, leaves such a copy to arrive.
#[test]
fn a_backend_copy_of_an_accepted_transaction_as_delivered_leaves_it_accepted() {
    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::GlobalServer);
    let mut io = MemoryStorage::new();
    seed_users_schema(&mut io);
    let backend = backend_client(&mut sm, &io);
    sm.take_outbox();
    let batch_id = BatchId::new();
    let base = visible_row(ObjectId::new(), BRANCH, Vec::new(), 900, b"base");
    seed_visible_row(&mut sm, &mut io, "users", base.clone());
    let row = row_with_batch_state(
        visible_row(
            base.row_id,
            BRANCH,
            vec![base.batch_id()],
            1_000,
            b"backend",
        ),
        batch_id,
        RowState::StagingPending,
        None,
    );
    sm.process_from_client(&mut io, backend, sent(&row));
    sm.process_from_client(
        &mut io,
        backend,
        SyncPayload::SealBatch {
            submission: seal_of(batch_id, &[&row]),
        },
    );
    assert_eq!(
        io.load_authoritative_batch_fate(batch_id).unwrap(),
        Some(BatchFate::AcceptedTransaction {
            batch_id,
            confirmed_tier: DurabilityTier::GlobalServer,
        })
    );
    let delivered = SyncManager::scope_delivery_row(accepted_locally(&row));
    assert!(delivered.parents.is_empty() && !row.parents.is_empty());

    sm.process_from_client(&mut io, backend, sent_again(&delivered));

    let held = stored(&io, &row);
    assert_eq!(
        (held.state, held.confirmed_tier),
        (
            RowState::VisibleTransactional,
            Some(DurabilityTier::GlobalServer)
        ),
        "a copy of a decided batch's row does not stage it again"
    );
    let served = served(&io, &row).expect("the accepted row is served");
    assert_eq!(
        (served.batch_id, served.state),
        (batch_id, RowState::VisibleTransactional)
    );
}
