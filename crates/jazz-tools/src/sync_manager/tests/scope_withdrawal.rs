//! What a client's server keeps about a row once the row leaves the client's scope.
//!
//! The client keeps the version it holds; the server stops forwarding the row. Two things must
//! still hold: the client hears the batch's fate if it is not settled yet, and nothing else about
//! the row stays behind for good — no owed entry that forces re-derivations, no row interest, no
//! delivery recorded by a confirmation that comes after the row left.

use super::*;

const MAIN: &str = "main";

/// A client holding one row through one query, the row stored with `confirmed_tier`.
fn client_holding_a_row(
    confirmed_tier: Option<DurabilityTier>,
) -> (SyncManager, MemoryStorage, ClientId, StoredRowBatch) {
    let mut sm = SyncManager::new();
    let mut io = MemoryStorage::new();
    let client_id = ClientId::new();
    let row_id = ObjectId::new();
    let mut row = visible_row(row_id, MAIN, Vec::new(), 1_000, b"alice");
    row.confirmed_tier = confirmed_tier;
    add_client(&mut sm, &io, client_id);
    sm.take_outbox();
    seed_visible_row(&mut sm, &mut io, "users", row.clone());
    persist_visible_row_settlement(&mut io, row_id, &row);
    set_client_query_scope(
        &mut sm,
        &io,
        client_id,
        QueryId(1),
        HashSet::from([(row_id, BranchName::new(MAIN))]),
        None,
    );
    (sm, io, client_id, row)
}

fn direct_fate(row: &StoredRowBatch) -> BatchFate {
    BatchFate::DurableDirect {
        batch_id: row.batch_id,
        confirmed_tier: DurabilityTier::Local,
    }
}

/// A direct row is forwarded as soon as it is applied, before its seal settles the batch, so on
/// a node with no upstream a row can leave a client's scope with no fate stored yet. The seal's
/// fate, when it comes, must still reach the client that holds the row.
#[test]
fn a_row_leaving_scope_before_its_fate_is_stored_keeps_the_client_on_the_fate() {
    let (mut sm, io, client_id, row) = client_holding_a_row(None);
    confirm_queued(&mut sm);
    sm.take_outbox();
    assert!(!sm.has_servers_or_pending_servers(), "fixture: no upstream");

    sm.drop_client_query_subscription(&io, client_id, QueryId(1));

    assert!(
        sm.interested_clients_for_batch_fate(&direct_fate(&row))
            .contains(&client_id),
        "the client holds a version whose batch has no fate yet, and nothing would bring it one"
    );
}

/// A row whose fate is settled leaves nothing behind: the client already has the last word.
#[test]
fn a_row_leaving_scope_with_a_settled_fate_leaves_nothing_behind() {
    let (mut sm, io, client_id, row) = client_holding_a_row(Some(DurabilityTier::Local));
    confirm_queued(&mut sm);
    sm.take_outbox();

    sm.drop_client_query_subscription(&io, client_id, QueryId(1));

    assert!(
        !sm.interested_clients_for_batch_fate(&direct_fate(&row))
            .contains(&client_id)
    );
    assert_eq!(sm.batch_fate_interest_clients(row.batch_id), 0);
    assert!(sm.get_client(client_id).unwrap().sent_batch_ids.is_empty());
}

/// A row offered and not confirmed when it leaves the client's scope is no longer owed: a
/// re-offer only covers rows in scope, so the entry would stay for good and force a
/// re-derivation on every resubscription. Its row interest goes with it, and a confirmation
/// that comes after the row left records nothing.
#[test]
fn a_row_left_unconfirmed_is_no_longer_owed_once_it_leaves_scope() {
    let (mut sm, io, client_id, row) = client_holding_a_row(Some(DurabilityTier::Local));
    let queued = sm.take_outbox();
    assert!(
        sm.client_has_undelivered_payloads(client_id),
        "fixture: the row is owed"
    );

    sm.drop_client_query_subscription(&io, client_id, QueryId(1));

    assert!(
        !sm.client_has_undelivered_payloads(client_id),
        "a row out of the client's scope is never offered again, so it cannot stay owed"
    );
    assert!(
        !sm.interested_clients_for_batch_fate(&direct_fate(&row))
            .contains(&client_id),
        "the row interest of the offered batch outlived the row's scope"
    );

    let confirmed: Vec<_> = queued
        .iter()
        .filter_map(|entry| match (&entry.destination, &entry.payload) {
            (Destination::Client(id), SyncPayload::RowBatchNeeded { row, .. }) => Some((
                *id,
                row.row_id,
                BranchName::new(row.branch.as_str()),
                row.batch_id,
            )),
            _ => None,
        })
        .collect();
    assert_eq!(confirmed.len(), 1, "fixture: the row was offered");
    sm.confirm_client_deliveries(&confirmed);
    assert!(
        sm.get_client(client_id).unwrap().sent_batch_ids.is_empty(),
        "a confirmation after the row left the client's scope recorded a delivery"
    );
}

/// A row another query of the client still holds stays: its delivery and its row interest.
#[test]
fn a_row_another_query_holds_stays_in_scope() {
    let (mut sm, io, client_id, row) = client_holding_a_row(Some(DurabilityTier::Local));
    confirm_queued(&mut sm);
    sm.take_outbox();
    set_client_query_scope(
        &mut sm,
        &io,
        client_id,
        QueryId(2),
        HashSet::from([(row.row_id, BranchName::new(MAIN))]),
        None,
    );
    sm.take_outbox();

    sm.drop_client_query_subscription(&io, client_id, QueryId(1));

    let client = sm.get_client(client_id).unwrap();
    assert!(client.is_in_scope(row.row_id, &BranchName::new(MAIN)));
    assert!(
        client
            .sent_batch_ids
            .get(&(row.row_id, BranchName::new(MAIN)))
            .is_some_and(|sent| sent.contains(&row.batch_id))
    );
    assert!(
        sm.interested_clients_for_batch_fate(&direct_fate(&row))
            .contains(&client_id)
    );
}

/// The rows of withdrawn queries keep a client on at most `MAX_PRUNED_FATE_INTEREST` batches
/// whose fate is not settled. A fate that never comes — a legacy row stored with none, a seal
/// that cannot complete — would otherwise hold its entry for as long as the client stays
/// connected. Past the bound the oldest entries go: the batches that left last are the ones
/// most likely to have a seal still in flight.
#[test]
fn fate_interest_kept_for_rows_out_of_scope_is_bounded_per_client_oldest_first() {
    let mut sm = SyncManager::new();
    let mut io = MemoryStorage::new();
    let client_id = ClientId::new();
    add_client(&mut sm, &io, client_id);
    sm.take_outbox();
    let hold = |sm: &mut SyncManager, io: &mut MemoryStorage, query: u64, count: usize| {
        let rows: Vec<StoredRowBatch> = (0..count)
            .map(|_| visible_row(ObjectId::new(), MAIN, Vec::new(), 1_000, b"alice"))
            .collect();
        for row in &rows {
            seed_visible_row(sm, io, "users", row.clone());
        }
        let scope = rows
            .iter()
            .map(|row| (row.row_id, BranchName::new(MAIN)))
            .collect();
        set_client_query_scope(sm, io, client_id, QueryId(query), scope, None);
        rows
    };
    let first = hold(&mut sm, &mut io, 1, MAX_PRUNED_FATE_INTEREST - 3);
    let second = hold(&mut sm, &mut io, 2, 8);
    confirm_queued(&mut sm);
    sm.take_outbox();

    sm.drop_client_query_subscription(&io, client_id, QueryId(1));
    assert_eq!(
        sm.batch_fate_interest_len(),
        first.len(),
        "fixture: every row of the first query left with no fate stored"
    );
    sm.drop_client_query_subscription(&io, client_id, QueryId(2));

    let kept = |rows: &[StoredRowBatch]| {
        rows.iter()
            .filter(|row| sm.batch_fate_interest_clients(row.batch_id) == 1)
            .count()
    };
    assert_eq!(
        kept(&second),
        second.len(),
        "the batches that left last must all be kept"
    );
    assert_eq!(
        kept(&first),
        first.len() - 5,
        "past the bound the oldest batches must go, and only as many as needed"
    );
    assert_eq!(sm.batch_fate_interest_len(), MAX_PRUNED_FATE_INTEREST);
    assert_eq!(
        sm.batch_fate_interest.len(),
        MAX_PRUNED_FATE_INTEREST,
        "a batch the bound took its last client off must not stay behind with no one on it"
    );
}

/// `count` fresh rows with no fate stored, put in `client_id`'s scope through `query`.
fn hold_rows(
    sm: &mut SyncManager,
    io: &mut MemoryStorage,
    client_id: ClientId,
    query: u64,
    count: usize,
) -> Vec<StoredRowBatch> {
    let rows: Vec<StoredRowBatch> = (0..count)
        .map(|_| visible_row(ObjectId::new(), MAIN, Vec::new(), 1_000, b"alice"))
        .collect();
    for row in &rows {
        seed_visible_row(sm, io, "users", row.clone());
    }
    let scope = rows
        .iter()
        .map(|row| (row.row_id, BranchName::new(MAIN)))
        .collect();
    set_client_query_scope(sm, io, client_id, QueryId(query), scope, None);
    rows
}

/// `rows` back in `client_id`'s scope through `query`, delivered, and out again.
fn pass_through_scope(
    sm: &mut SyncManager,
    io: &MemoryStorage,
    client_id: ClientId,
    query: u64,
    rows: &[StoredRowBatch],
) {
    let scope = rows
        .iter()
        .map(|row| (row.row_id, BranchName::new(MAIN)))
        .collect();
    set_client_query_scope(sm, io, client_id, QueryId(query), scope, None);
    confirm_queued(sm);
    sm.take_outbox();
    sm.drop_client_query_subscription(io, client_id, QueryId(query));
}

/// The bound takes the evicting client off a batch, not the batch: another client the prune
/// keeps on the same fate still hears it.
#[test]
fn the_bound_takes_only_its_own_client_off_a_shared_batch() {
    let mut sm = SyncManager::new();
    let mut io = MemoryStorage::new();
    let (first, second) = (ClientId::new(), ClientId::new());
    add_client(&mut sm, &io, first);
    add_client(&mut sm, &io, second);
    sm.take_outbox();
    let shared = hold_rows(&mut sm, &mut io, first, 1, 1).remove(0);
    pass_through_scope(&mut sm, &io, second, 1, std::slice::from_ref(&shared));
    confirm_queued(&mut sm);
    sm.take_outbox();
    sm.drop_client_query_subscription(&io, first, QueryId(1));
    assert_eq!(
        sm.batch_fate_interest_clients(shared.batch_id),
        2,
        "fixture: the row left both clients with no fate stored"
    );

    hold_rows(&mut sm, &mut io, first, 2, MAX_PRUNED_FATE_INTEREST);
    confirm_queued(&mut sm);
    sm.take_outbox();
    sm.drop_client_query_subscription(&io, first, QueryId(2));

    assert_eq!(
        sm.batch_fate_interest_clients(shared.batch_id),
        1,
        "the bound of one client took the other client off the batch too"
    );
    assert!(
        sm.interested_clients_for_batch_fate(&direct_fate(&shared))
            .contains(&second)
    );
}

/// A client that goes away takes the prune's queue with it, and nothing it kept stays counted.
#[test]
fn a_client_that_goes_away_leaves_no_fate_interest_queue_behind() {
    let mut sm = SyncManager::new();
    let mut io = MemoryStorage::new();
    let baseline = sm.memory_size().2;
    let client_id = ClientId::new();
    add_client(&mut sm, &io, client_id);
    sm.take_outbox();
    hold_rows(&mut sm, &mut io, client_id, 1, 3);
    confirm_queued(&mut sm);
    sm.take_outbox();
    sm.drop_client_query_subscription(&io, client_id, QueryId(1));
    assert_eq!(sm.pruned_fate_interest_len(client_id), 3, "fixture");

    assert!(sm.remove_client(client_id));

    assert_eq!(sm.pruned_fate_interest_len(client_id), 0);
    assert_eq!(
        sm.memory_size().2,
        baseline,
        "the subscription state of a client that went away is still counted"
    );
}

/// A batch the client is already kept on is not queued again when it leaves the scope once
/// more: the queue holds one entry per registration, so the bound counts batches, not exits.
#[test]
fn a_batch_already_kept_is_not_queued_again() {
    let mut sm = SyncManager::new();
    let mut io = MemoryStorage::new();
    let client_id = ClientId::new();
    add_client(&mut sm, &io, client_id);
    sm.take_outbox();
    let rows = hold_rows(&mut sm, &mut io, client_id, 1, 1);
    confirm_queued(&mut sm);
    sm.take_outbox();
    sm.drop_client_query_subscription(&io, client_id, QueryId(1));
    assert_eq!(sm.pruned_fate_interest_len(client_id), 1, "fixture");

    pass_through_scope(&mut sm, &io, client_id, 2, &rows);

    assert_eq!(sm.batch_fate_interest_clients(rows[0].batch_id), 1);
    assert_eq!(
        sm.pruned_fate_interest_len(client_id),
        1,
        "a batch the client was already kept on took a second place in the queue"
    );
}

/// A registration that ended before it reached the front of the queue takes nothing when it
/// gets there, even when the prune has registered the same batch again since: the fresh
/// registration waits its own turn. The fate here is settled where it is relayed and absent
/// from the store, which is what a failed fate read, a rising settlement target or a fate of
/// another kind merged in all leave the prune to see.
#[test]
fn a_stale_queue_entry_does_not_evict_a_fresh_registration() {
    let mut sm = SyncManager::new();
    let mut io = MemoryStorage::new();
    let client_id = ClientId::new();
    add_client(&mut sm, &io, client_id);
    sm.take_outbox();
    let rows = hold_rows(&mut sm, &mut io, client_id, 1, 1);
    confirm_queued(&mut sm);
    sm.take_outbox();
    sm.drop_client_query_subscription(&io, client_id, QueryId(1));
    sm.retire_batch_fate_interest_if_settled(&direct_fate(&rows[0]));
    assert_eq!(
        sm.batch_fate_interest_clients(rows[0].batch_id),
        0,
        "fixture: the registration ended"
    );
    pass_through_scope(&mut sm, &io, client_id, 2, &rows);
    assert_eq!(
        sm.batch_fate_interest_clients(rows[0].batch_id),
        1,
        "fixture: the prune registered the batch again"
    );
    assert_eq!(
        sm.pruned_fate_interest_len(client_id),
        2,
        "fixture: the ended registration is still queued in front of the fresh one"
    );

    hold_rows(&mut sm, &mut io, client_id, 3, MAX_PRUNED_FATE_INTEREST - 1);
    confirm_queued(&mut sm);
    sm.take_outbox();
    sm.drop_client_query_subscription(&io, client_id, QueryId(3));
    assert_eq!(
        sm.batch_fate_interest_clients(rows[0].batch_id),
        1,
        "the ended registration reached the front and took the fresh one with it"
    );

    hold_rows(&mut sm, &mut io, client_id, 4, 1);
    confirm_queued(&mut sm);
    sm.take_outbox();
    sm.drop_client_query_subscription(&io, client_id, QueryId(4));
    assert_eq!(
        sm.batch_fate_interest_clients(rows[0].batch_id),
        0,
        "the fresh registration is the oldest now, and the bound takes it"
    );
}

/// The bound evicts only what the prune registered. A batch the client then asks about itself
/// (`BatchFateNeeded`, sent on reconnect for its own pending batches) stays its interest, however
/// many batches leave its scope after: it would not ask again before it reconnects.
#[test]
fn a_batch_the_client_asked_about_is_not_evicted_by_the_prune_bound() {
    let mut sm = SyncManager::new();
    let mut io = MemoryStorage::new();
    let client_id = ClientId::new();
    add_client(&mut sm, &io, client_id);
    sm.take_outbox();
    let asked = hold_rows(&mut sm, &mut io, client_id, 1, 1);
    confirm_queued(&mut sm);
    sm.take_outbox();
    sm.drop_client_query_subscription(&io, client_id, QueryId(1));
    assert_eq!(
        sm.batch_fate_interest_clients(asked[0].batch_id),
        1,
        "fixture: the row left with no fate stored, so the prune kept the client on it"
    );
    sm.process_from_client(
        &mut io,
        client_id,
        SyncPayload::BatchFateNeeded {
            batch_ids: vec![asked[0].batch_id],
        },
    );
    sm.take_outbox();

    let later = hold_rows(&mut sm, &mut io, client_id, 2, MAX_PRUNED_FATE_INTEREST + 4);
    confirm_queued(&mut sm);
    sm.take_outbox();
    sm.drop_client_query_subscription(&io, client_id, QueryId(2));

    assert_eq!(
        sm.batch_fate_interest_clients(asked[0].batch_id),
        1,
        "the client asked about this batch itself; the prune's bound must not take it off"
    );
    assert_eq!(
        later
            .iter()
            .filter(|row| sm.batch_fate_interest_clients(row.batch_id) == 1)
            .count(),
        MAX_PRUNED_FATE_INTEREST,
        "the prune's own registrations stay bounded"
    );
}

/// A row's batch that entered the scope settled is not read again when the row leaves: the
/// client keeps what it was sent. That fate is final only at the target it was settled at, so
/// once the target rises (an upstream is added) the prune reads the fate again, and keeps the
/// client on a batch the upstream has not confirmed.
#[test]
fn a_fate_settled_when_sent_is_read_again_once_the_settlement_target_rises() {
    let (mut sm, io, client_id, row) = client_holding_a_row(Some(DurabilityTier::Local));
    confirm_queued(&mut sm);
    sm.take_outbox();
    assert!(
        sm.get_client(client_id)
            .unwrap()
            .settled_fates_sent
            .contains_key(&row.batch_id),
        "fixture: the batch entered the scope settled"
    );
    sm.add_pending_server(ServerId::new());
    assert_eq!(
        sm.settlement_target(),
        DurabilityTier::GlobalServer,
        "fixture"
    );

    sm.drop_client_query_subscription(&io, client_id, QueryId(1));

    assert_eq!(
        sm.batch_fate_interest_clients(row.batch_id),
        1,
        "the batch is confirmed below the new target, and the client was not kept on its fate"
    );
    assert!(
        sm.get_client(client_id)
            .unwrap()
            .settled_fates_sent
            .is_empty(),
        "the row left the scope, and what it entered with stayed behind"
    );
}

/// A fate sent to the client after its row entered the scope means the stored fate has moved
/// (`BatchFate::merged_with` takes a fate of another kind as it comes), so the prune reads it
/// again instead of trusting what the row entered with.
#[test]
fn a_later_fate_sent_to_the_client_makes_the_prune_read_the_fate_again() {
    let (mut sm, mut io, client_id, row) = client_holding_a_row(Some(DurabilityTier::GlobalServer));
    confirm_queued(&mut sm);
    sm.take_outbox();
    assert!(
        sm.get_client(client_id)
            .unwrap()
            .settled_fates_sent
            .contains_key(&row.batch_id),
        "fixture: the batch entered the scope settled"
    );
    let lower = BatchFate::AcceptedTransaction {
        batch_id: row.batch_id,
        confirmed_tier: DurabilityTier::Local,
    };
    io.upsert_authoritative_batch_fate(&lower).unwrap();
    sm.queue_batch_fate_to_client_unfiltered(client_id, lower);
    sm.take_outbox();
    sm.add_pending_server(ServerId::new());

    sm.drop_client_query_subscription(&io, client_id, QueryId(1));

    assert_eq!(
        sm.batch_fate_interest_clients(row.batch_id),
        1,
        "the prune trusted the fate the row entered with over the one stored since"
    );
}

/// The same through the path a relayed fate takes. Here the batch was stored accepted as a
/// transaction, settled; an upstream's fate of another kind replaces it unsettled, and only the
/// relay tells the clients that hold the batch's rows (a transactional fate also reaches them
/// through the rows it settles; a direct one does not). The prune must read the fate again for
/// a client told of it that way.
#[test]
fn a_fate_relayed_from_upstream_makes_the_prune_read_the_fate_again() {
    let mut sm = SyncManager::new();
    let mut io = MemoryStorage::new();
    let client_id = ClientId::new();
    let row_id = ObjectId::new();
    let mut row = visible_row(row_id, MAIN, Vec::new(), 1_000, b"alice");
    row.confirmed_tier = Some(DurabilityTier::GlobalServer);
    add_client(&mut sm, &io, client_id);
    sm.take_outbox();
    seed_visible_row(&mut sm, &mut io, "users", row.clone());
    io.upsert_authoritative_batch_fate(&BatchFate::AcceptedTransaction {
        batch_id: row.batch_id,
        confirmed_tier: DurabilityTier::GlobalServer,
    })
    .unwrap();
    set_client_query_scope(
        &mut sm,
        &io,
        client_id,
        QueryId(1),
        HashSet::from([(row_id, BranchName::new(MAIN))]),
        None,
    );
    confirm_queued(&mut sm);
    sm.take_outbox();
    let server_id = ServerId::new();
    add_server(&mut sm, &io, server_id);
    sm.take_outbox();
    assert_eq!(
        sm.get_client(client_id)
            .unwrap()
            .settled_fates_sent
            .get(&row.batch_id),
        Some(&DurabilityTier::GlobalServer),
        "fixture: the batch entered the scope settled"
    );

    sm.process_from_server(
        &mut io,
        server_id,
        SyncPayload::BatchFate {
            fate: BatchFate::DurableDirect {
                batch_id: row.batch_id,
                confirmed_tier: DurabilityTier::Local,
            },
        },
    );
    assert_eq!(
        io.load_authoritative_batch_fate(row.batch_id).unwrap(),
        Some(BatchFate::DurableDirect {
            batch_id: row.batch_id,
            confirmed_tier: DurabilityTier::Local,
        }),
        "fixture: the relayed fate of another kind replaced the stored one"
    );
    assert!(
        sm.take_outbox().iter().any(|entry| {
            entry.destination == Destination::Client(client_id)
                && matches!(&entry.payload, SyncPayload::BatchFate { fate } if fate.batch_id() == row.batch_id)
        }),
        "fixture: the relayed fate reached the client that holds the row"
    );

    sm.drop_client_query_subscription(&io, client_id, QueryId(1));

    assert_eq!(
        sm.batch_fate_interest_clients(row.batch_id),
        1,
        "the prune trusted the fate the row entered with over the one relayed since"
    );
}

/// A new version of `row`, settled at `tier`, stored and forwarded to the clients holding the row.
fn update_and_forward(
    sm: &mut SyncManager,
    io: &mut MemoryStorage,
    row: &StoredRowBatch,
    tier: DurabilityTier,
) -> StoredRowBatch {
    update_and_forward_with_parents(sm, io, row, tier, vec![row.batch_id])
}

/// [`update_and_forward`] with the parents the stored version carries. The durable stores load
/// a visible row without its parents, so `Vec::new()` is what forwarding sees there.
fn update_and_forward_with_parents(
    sm: &mut SyncManager,
    io: &mut MemoryStorage,
    row: &StoredRowBatch,
    tier: DurabilityTier,
    parents: Vec<BatchId>,
) -> StoredRowBatch {
    let mut next = visible_row(row.row_id, MAIN, parents, row.updated_at + 1, b"bob");
    next.confirmed_tier = Some(tier);
    seed_visible_row(sm, io, "users", next.clone());
    persist_visible_row_settlement(io, next.row_id, &next);
    sm.forward_update_to_clients_with_storage(io, next.row_id, BranchName::new(MAIN));
    next
}

/// A row that moves off the batch it entered the scope with takes that batch off the delivered
/// frontier, so the scope prune never walks to it again: what the batch entered settled with
/// must go with it, not wait for the client to go away.
#[test]
fn a_row_moving_off_the_batch_it_entered_with_takes_its_settled_fate_out() {
    let (mut sm, mut io, client_id, row) = client_holding_a_row(Some(DurabilityTier::Local));
    confirm_queued(&mut sm);
    sm.take_outbox();
    assert!(
        sm.get_client(client_id)
            .unwrap()
            .settled_fates_sent
            .contains_key(&row.batch_id),
        "fixture: the batch entered the scope settled"
    );

    update_and_forward(&mut sm, &mut io, &row, DurabilityTier::Local);
    confirm_queued(&mut sm);
    sm.take_outbox();
    assert!(
        !sm.get_client(client_id)
            .unwrap()
            .settled_fates_sent
            .contains_key(&row.batch_id),
        "the row moved off the batch it entered with, and that batch stayed behind"
    );

    sm.drop_client_query_subscription(&io, client_id, QueryId(1));
    assert!(
        sm.get_client(client_id)
            .unwrap()
            .settled_fates_sent
            .is_empty(),
        "the row left the scope and something it entered with stayed behind"
    );
}

/// A merge supersedes every batch it names as a parent, not just the first: the one the row
/// entered the scope with leaves the frontier as the second parent too, and what it entered
/// settled with goes with it.
#[test]
fn a_merge_naming_the_batch_as_its_second_parent_takes_its_settled_fate_out() {
    let (mut sm, mut io, client_id, row) = client_holding_a_row(Some(DurabilityTier::Local));
    confirm_queued(&mut sm);
    sm.take_outbox();
    assert!(
        sm.get_client(client_id)
            .unwrap()
            .settled_fates_sent
            .contains_key(&row.batch_id),
        "fixture: the batch entered the scope settled"
    );

    update_and_forward_with_parents(
        &mut sm,
        &mut io,
        &row,
        DurabilityTier::Local,
        vec![BatchId::new(), row.batch_id],
    );
    assert!(
        !sm.get_client(client_id)
            .unwrap()
            .settled_fates_sent
            .contains_key(&row.batch_id),
        "the merge superseded the batch the row entered with, and that batch stayed behind"
    );
}

/// A client that confirms deliveries is offered a row it has not confirmed yet when a newer
/// version replaces the offer. Its confirmation of the old batch is dropped from then on, so the
/// scope prune walks only the new one: what the old batch entered settled with goes now. The new
/// version is stored without parents, as the durable stores load it, so nothing else takes it.
#[test]
fn an_unconfirmed_offer_a_newer_version_replaces_takes_its_settled_fate_out() {
    let (mut sm, mut io, client_id, row) = client_holding_a_row(Some(DurabilityTier::Local));
    sm.take_outbox();
    assert!(
        sm.get_client(client_id)
            .unwrap()
            .settled_fates_sent
            .contains_key(&row.batch_id),
        "fixture: the batch entered the scope settled"
    );
    assert_eq!(
        sm.pending_client_deliveries[&client_id][&(row.row_id, BranchName::new(MAIN))].batch_id,
        row.batch_id,
        "fixture: the row is offered and not confirmed"
    );

    let next =
        update_and_forward_with_parents(&mut sm, &mut io, &row, DurabilityTier::Local, Vec::new());
    assert_eq!(
        sm.pending_client_deliveries[&client_id][&(row.row_id, BranchName::new(MAIN))].batch_id,
        next.batch_id,
        "fixture: the newer version replaced the unconfirmed offer"
    );
    assert!(
        !sm.get_client(client_id)
            .unwrap()
            .settled_fates_sent
            .contains_key(&row.batch_id),
        "the replaced offer will never be confirmed, and its settled fate stayed behind"
    );

    confirm_queued(&mut sm);
    sm.take_outbox();
    sm.drop_client_query_subscription(&io, client_id, QueryId(1));
    assert!(
        sm.get_client(client_id)
            .unwrap()
            .settled_fates_sent
            .is_empty(),
        "the row left the scope and something it entered with stayed behind"
    );
}

/// Random scope changes, withdrawals and new versions of rows, for one client that confirms what
/// it applies and one that does not. A new version is stored with its parent or, as the durable
/// stores load it, without. After every step each batch the client is recorded as having been
/// sent settled is one the scope prune walks: a batch a row in scope was delivered at or is
/// offered at. Nothing the prune can no longer reach stays behind.
#[test]
fn settled_fates_sent_only_ever_name_batches_of_rows_in_scope() {
    const ROWS: usize = 8;
    const QUERIES: u64 = 3;
    const SEEDS: u64 = 16;
    const STEPS: usize = 200;
    let branch = BranchName::new(MAIN);
    let mut moved_off = 0;
    let mut offers_replaced = 0;
    for seed in 0..SEEDS {
        let mut rng = Rng(seed.wrapping_add(7).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let mut sm = SyncManager::new();
        let mut io = MemoryStorage::new();
        let mut rows: Vec<StoredRowBatch> = (0..ROWS)
            .map(|index| {
                let mut row = visible_row(
                    ObjectId::new(),
                    MAIN,
                    Vec::new(),
                    1_000 + index as u64 * 1_000,
                    b"r",
                );
                row.confirmed_tier = Some(DurabilityTier::Local);
                row
            })
            .collect();
        for row in &rows {
            seed_visible_row(&mut sm, &mut io, "users", row.clone());
            persist_visible_row_settlement(&mut io, row.row_id, row);
        }
        let silent = ClientId::new();
        sm.add_client_with_storage(&io, silent);
        let acking = ClientId::new();
        add_client(&mut sm, &io, acking);
        let clients = [silent, acking];
        sm.take_outbox();

        for step in 0..STEPS {
            let client_id = clients[rng.below(2) as usize];
            match rng.below(4) {
                0 => sm.drop_client_query_subscription(&io, client_id, QueryId(rng.below(QUERIES))),
                1 => {
                    let index = rng.below(ROWS as u64) as usize;
                    let tier = if rng.below(2) == 0 {
                        DurabilityTier::Local
                    } else {
                        DurabilityTier::GlobalServer
                    };
                    let held_before = clients.iter().any(|&id| {
                        sm.get_client(id)
                            .unwrap()
                            .settled_fates_sent
                            .contains_key(&rows[index].batch_id)
                    });
                    moved_off += usize::from(held_before);
                    let key = (rows[index].row_id, branch);
                    offers_replaced += usize::from(
                        sm.pending_client_deliveries
                            .get(&acking)
                            .and_then(|owed| owed.get(&key))
                            .is_some_and(|pending| {
                                sm.get_client(acking)
                                    .unwrap()
                                    .settled_fates_sent
                                    .contains_key(&pending.batch_id)
                            }),
                    );
                    let parents = if rng.below(2) == 0 {
                        vec![rows[index].batch_id]
                    } else {
                        Vec::new()
                    };
                    rows[index] = update_and_forward_with_parents(
                        &mut sm,
                        &mut io,
                        &rows[index],
                        tier,
                        parents,
                    );
                }
                _ => {
                    let scope = (0..ROWS)
                        .filter(|_| rng.below(3) == 0)
                        .map(|index| (rows[index].row_id, branch))
                        .collect();
                    set_client_query_scope(
                        &mut sm,
                        &io,
                        client_id,
                        QueryId(rng.below(QUERIES)),
                        scope,
                        None,
                    );
                }
            }
            if rng.below(2) == 0 {
                confirm_queued(&mut sm);
            }
            sm.take_outbox();
            for &id in &clients {
                let state = sm.get_client(id).unwrap();
                let owed = sm.pending_client_deliveries.get(&id);
                let walked = |batch_id: &BatchId| {
                    rows.iter()
                        .filter(|row| state.is_in_scope(row.row_id, &branch))
                        .any(|row| {
                            let key = (row.row_id, branch);
                            state
                                .sent_batch_ids
                                .get(&key)
                                .is_some_and(|sent| sent.contains(batch_id))
                                || owed
                                    .and_then(|owed| owed.get(&key))
                                    .is_some_and(|pending| pending.batch_id == *batch_id)
                        })
                };
                for batch_id in state.settled_fates_sent.keys() {
                    assert!(
                        walked(batch_id),
                        "seed {seed} step {step}: a settled fate stayed behind for a batch the \
                         scope prune no longer walks"
                    );
                }
            }
        }
    }
    assert!(
        moved_off > 50,
        "only {moved_off} rows moved off a recorded batch"
    );
    assert!(
        offers_replaced > 20,
        "only {offers_replaced} unconfirmed offers of a recorded batch were replaced"
    );
}

/// Rows that entered the scope settled leave it without a single fate load: the measured cost
/// MEDIUM-A removes (about 1 100 loads per dense read). Once the target rises above what they
/// settled at, the prune reads each batch once, however many of its rows leave.
#[test]
fn rows_that_entered_settled_leave_without_reading_their_fates() {
    let mut sm = SyncManager::new();
    let mut io = MemoryStorage::new();
    let client_id = ClientId::new();
    add_client(&mut sm, &io, client_id);
    sm.take_outbox();
    let settled: Vec<StoredRowBatch> = (0..16)
        .map(|index| {
            let mut row = visible_row(ObjectId::new(), MAIN, Vec::new(), 1_000 + index, b"alice");
            row.confirmed_tier = Some(DurabilityTier::Local);
            row
        })
        .collect();
    let first = settled[0].clone();
    let mut sibling = visible_row(ObjectId::new(), MAIN, Vec::new(), 2_000, b"alice");
    sibling.batch_id = first.batch_id;
    sibling.confirmed_tier = Some(DurabilityTier::Local);
    for row in settled.iter().chain(std::iter::once(&sibling)) {
        seed_visible_row(&mut sm, &mut io, "users", row.clone());
        persist_visible_row_settlement(&mut io, row.row_id, row);
    }
    set_client_query_scope(
        &mut sm,
        &io,
        client_id,
        QueryId(1),
        settled[1..]
            .iter()
            .map(|row| (row.row_id, BranchName::new(MAIN)))
            .collect(),
        None,
    );
    set_client_query_scope(
        &mut sm,
        &io,
        client_id,
        QueryId(2),
        [&first, &sibling]
            .iter()
            .map(|row| (row.row_id, BranchName::new(MAIN)))
            .collect(),
        None,
    );
    confirm_queued(&mut sm);
    sm.take_outbox();

    io.reset_authoritative_fate_lookups();
    sm.drop_client_query_subscription(&io, client_id, QueryId(1));
    assert_eq!(
        io.authoritative_fate_lookups(),
        0,
        "rows that entered settled read their fates again on the way out"
    );

    sm.add_pending_server(ServerId::new());
    io.reset_authoritative_fate_lookups();
    sm.drop_client_query_subscription(&io, client_id, QueryId(2));
    assert_eq!(
        io.authoritative_fate_lookups(),
        1,
        "past a risen target the prune reads a batch once, however many of its rows leave"
    );
}

/// The map is a cache: past its bound it is dropped whole, and the scope it served still works.
#[test]
fn settled_fates_sent_stay_bounded() {
    let mut sm = SyncManager::new();
    let mut io = MemoryStorage::new();
    let client_id = ClientId::new();
    add_client(&mut sm, &io, client_id);
    sm.take_outbox();
    let rows: Vec<StoredRowBatch> = (0..MAX_SETTLED_FATES_SENT + 3)
        .map(|index| {
            let mut row = visible_row(
                ObjectId::new(),
                MAIN,
                Vec::new(),
                1_000 + index as u64,
                b"a",
            );
            row.confirmed_tier = Some(DurabilityTier::Local);
            row
        })
        .collect();
    for row in &rows {
        seed_visible_row(&mut sm, &mut io, "users", row.clone());
        persist_visible_row_settlement(&mut io, row.row_id, row);
    }
    let (to_the_bound, past_it) = rows.split_at(MAX_SETTLED_FATES_SENT);
    let held_after = |sm: &mut SyncManager, query: u64, rows: &[StoredRowBatch]| {
        set_client_query_scope(
            sm,
            &io,
            client_id,
            QueryId(query),
            rows.iter()
                .map(|row| (row.row_id, BranchName::new(MAIN)))
                .collect(),
            None,
        );
        sm.get_client(client_id).unwrap().settled_fates_sent.len()
    };
    assert_eq!(
        held_after(&mut sm, 1, to_the_bound),
        MAX_SETTLED_FATES_SENT,
        "the map holds every settled fate up to its bound"
    );
    assert_eq!(
        held_after(&mut sm, 2, past_it),
        3,
        "the map is dropped whole when it reaches its bound, so it holds the 3 recorded after"
    );

    sm.drop_client_query_subscription(&io, client_id, QueryId(1));
    sm.drop_client_query_subscription(&io, client_id, QueryId(2));
    let state = sm.get_client(client_id).unwrap();
    assert!(
        state.settled_fates_sent.is_empty() && state.queries.is_empty(),
        "the scope the dropped map served still leaves cleanly"
    );
    assert!(
        rows.iter()
            .all(|row| sm.batch_fate_interest_clients(row.batch_id) == 0),
        "the prune read the settled fates it no longer had cached and kept no interest"
    );
}

/// The same claim for a request past the tracked-answers cap: the batches registered from the
/// truncated list are the client's own interest, and the prune bound never takes them.
#[test]
fn a_batch_named_in_an_oversized_request_is_not_evicted_by_the_prune_bound() {
    let mut sm = SyncManager::new();
    let mut io = MemoryStorage::new();
    let client_id = ClientId::new();
    add_client(&mut sm, &io, client_id);
    sm.take_outbox();
    let asked = hold_rows(&mut sm, &mut io, client_id, 1, 1).remove(0);
    confirm_queued(&mut sm);
    sm.take_outbox();
    sm.drop_client_query_subscription(&io, client_id, QueryId(1));
    assert_eq!(
        sm.batch_fate_interest_clients(asked.batch_id),
        1,
        "fixture: the prune kept the client on the batch"
    );
    let mut named = vec![asked.batch_id];
    for index in 0..(MAX_TRACKED_MISSING_ANSWERS + 8) as u32 {
        let mut bytes = [0xFF_u8; 16];
        bytes[12..].copy_from_slice(&index.to_be_bytes());
        named.push(BatchId(bytes));
    }
    assert!(
        named[1..].iter().all(|other| *other > asked.batch_id),
        "fixture: the asked batch sorts first, so the truncated list keeps it"
    );
    sm.process_from_client(
        &mut io,
        client_id,
        SyncPayload::BatchFateNeeded { batch_ids: named },
    );
    sm.take_outbox();

    hold_rows(&mut sm, &mut io, client_id, 2, MAX_PRUNED_FATE_INTEREST + 4);
    confirm_queued(&mut sm);
    sm.take_outbox();
    sm.drop_client_query_subscription(&io, client_id, QueryId(2));

    assert_eq!(
        sm.batch_fate_interest_clients(asked.batch_id),
        1,
        "the client named this batch in an oversized request; the prune bound took it off"
    );
}

/// xorshift64*, so a failing seed replays exactly.
struct Rng(u64);

impl Rng {
    fn below(&mut self, bound: u64) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D) % bound
    }
}

/// What one client holds, by row index, as the server did before a scope change cost only what
/// changed: every query's scope, flattened.
#[derive(Default)]
struct ModelClient {
    queries: HashMap<u64, HashSet<usize>>,
    /// Rows recorded as delivered.
    sent: HashSet<usize>,
    /// Rows offered and not confirmed.
    owed: HashSet<usize>,
}

impl ModelClient {
    fn scope(&self) -> HashSet<usize> {
        self.queries.values().flatten().copied().collect()
    }
}

/// Random scope changes and withdrawals over a few queries of two clients — one that confirms
/// what it applies and one that does not — against a model that flattens every query of a
/// client to find what leaves its scope. Checked after every step: the rows queued, the scope,
/// the deliveries recorded, the rows owed and the row interest.
#[test]
fn scope_changes_match_a_flattening_model() {
    const ROWS: usize = 10;
    const QUERIES: u64 = 4;
    const SEEDS: u64 = 24;
    const STEPS: usize = 300;
    let branch = BranchName::new(MAIN);
    let mut withdrawals = 0;
    let mut left_owed = 0;
    for seed in 0..SEEDS {
        let mut rng = Rng(seed.wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let mut sm = SyncManager::new();
        let mut io = MemoryStorage::new();
        let rows: Vec<StoredRowBatch> = (0..ROWS)
            .map(|index| {
                let mut row = visible_row(
                    ObjectId::new(),
                    MAIN,
                    Vec::new(),
                    1_000 + index as u64,
                    b"r",
                );
                row.confirmed_tier = Some(DurabilityTier::Local);
                row
            })
            .collect();
        for row in &rows {
            seed_visible_row(&mut sm, &mut io, "users", row.clone());
            persist_visible_row_settlement(&mut io, row.row_id, row);
        }
        let silent = ClientId::new();
        sm.add_client_with_storage(&io, silent);
        let acking = ClientId::new();
        add_client(&mut sm, &io, acking);
        let clients = [silent, acking];
        let mut models = [ModelClient::default(), ModelClient::default()];
        sm.take_outbox();

        for step in 0..STEPS {
            let which = rng.below(2) as usize;
            let client_id = clients[which];
            let query = rng.below(QUERIES);
            let model = &mut models[which];
            let old_scope = model.scope();
            let mut expected_queued = HashSet::new();
            match rng.below(4) {
                0 => {
                    withdrawals += 1;
                    sm.drop_client_query_subscription(&io, client_id, QueryId(query));
                    model.queries.remove(&query);
                }
                1 if which == 1 => {
                    // The acking client confirms some of what it is owed.
                    let confirmed: Vec<_> = model
                        .owed
                        .iter()
                        .copied()
                        .filter(|_| rng.below(2) == 0)
                        .collect();
                    sm.confirm_client_deliveries(
                        &confirmed
                            .iter()
                            .map(|&index| {
                                (client_id, rows[index].row_id, branch, rows[index].batch_id)
                            })
                            .collect::<Vec<_>>(),
                    );
                    for index in confirmed {
                        model.owed.remove(&index);
                        model.sent.insert(index);
                    }
                }
                _ => {
                    let new_query_scope: HashSet<usize> =
                        (0..ROWS).filter(|_| rng.below(3) == 0).collect();
                    let old_query_scope = model.queries.get(&query).cloned().unwrap_or_default();
                    expected_queued = new_query_scope
                        .iter()
                        .copied()
                        .filter(|index| {
                            !old_query_scope.contains(index) || model.owed.contains(index)
                        })
                        .collect();
                    set_client_query_scope(
                        &mut sm,
                        &io,
                        client_id,
                        QueryId(query),
                        new_query_scope
                            .iter()
                            .map(|&index| (rows[index].row_id, branch))
                            .collect(),
                        None,
                    );
                    model.queries.insert(query, new_query_scope);
                }
            }
            let new_scope = model.scope();
            for index in old_scope.difference(&new_scope) {
                model.sent.remove(index);
                left_owed += usize::from(model.owed.remove(index));
            }
            for &index in &expected_queued {
                if which == 1 {
                    model.owed.insert(index);
                } else {
                    model.sent.insert(index);
                }
            }

            let queued: Vec<usize> = sm
                .take_outbox()
                .into_iter()
                .filter_map(|entry| match (entry.destination, entry.payload) {
                    (Destination::Client(id), SyncPayload::RowBatchNeeded { row, .. })
                        if id == client_id =>
                    {
                        rows.iter().position(|seeded| seeded.row_id == row.row_id)
                    }
                    _ => None,
                })
                .collect();
            let context = format!("seed {seed} step {step} client {which}");
            assert_eq!(
                queued.iter().copied().collect::<HashSet<_>>(),
                expected_queued,
                "{context}: rows queued"
            );
            assert_eq!(
                queued.len(),
                expected_queued.len(),
                "{context}: each row once"
            );
            let state = sm.get_client(client_id).unwrap();
            let model = &models[which];
            for (index, row) in rows.iter().enumerate() {
                assert_eq!(
                    state.is_in_scope(row.row_id, &branch),
                    new_scope.contains(&index),
                    "{context}: row {index} in scope"
                );
                assert_eq!(
                    state
                        .sent_batch_ids
                        .get(&(row.row_id, branch))
                        .is_some_and(|sent| sent.contains(&row.batch_id)),
                    model.sent.contains(&index),
                    "{context}: row {index} recorded as delivered"
                );
                assert_eq!(
                    sm.row_batch_interest
                        .clients_of_batch(row.batch_id)
                        .any(|id| id == client_id),
                    new_scope.contains(&index),
                    "{context}: row interest of row {index}"
                );
            }
            assert_eq!(
                sm.client_has_undelivered_payloads(client_id),
                !model.owed.is_empty(),
                "{context}: rows owed"
            );
        }
    }
    assert!(
        withdrawals > 1000 && left_owed > 100,
        "{withdrawals} withdrawals, {left_owed} owed rows that left scope"
    );
}
