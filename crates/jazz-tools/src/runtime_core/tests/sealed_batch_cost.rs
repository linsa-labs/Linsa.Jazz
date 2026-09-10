//! A transactional write must not cost a walk of the whole store.
//!
//! `sealed_batch_submission` captures a "family visible frontier" for every
//! `Transactional` batch, and `capture_family_visible_frontier` builds it from EVERY
//! visible row in the branch family. The payload is compatibility-only: upstream PR #920
//! removed the validation that read it (conflicts are decided from the staged rows' own
//! parents, `sync_manager::inbox::validate_transactional_parent_frontiers`) and left the
//! capture behind, commented for removal at the next storage-format break. Nothing has
//! read it since.
//!
//! While rows are small this is invisible. Measured on a device store holding 390
//! one-megabyte blob rows, the same code turned every sent message into a 384 MB read and
//! froze the app on every write.
//!
//! The gate asserts the SHAPE, not a duration and not bytes: the captured frontier must be
//! bounded by the batch, never by the store. Shape is what makes it backend-independent —
//! `MemoryStorage` overrides the capture with an in-memory walk that costs nothing to
//! traverse, so a bytes-read assertion would pass here while the real backends bled.

use std::collections::HashMap;

use super::*;

/// Unrelated rows seeded before the measured write. More than a handful, so a frontier
/// that tracks the store is unmistakable next to one that tracks the batch.
const UNRELATED_ROWS: usize = 25;

fn transactional() -> WriteContext {
    WriteContext {
        session: None,
        attribution: None,
        updated_at: None,
        batch_mode: Some(crate::batch_fate::BatchMode::Transactional),
        batch_id: None,
        target_branch_name: None,
    }
}

fn user_values(id: ObjectId, name: &str) -> HashMap<String, Value> {
    HashMap::from([
        ("id".to_string(), Value::Uuid(id)),
        ("name".to_string(), Value::Text(name.to_string())),
    ])
}

#[test]
fn a_sealed_transactional_batch_does_not_carry_the_whole_store() {
    let mut s = create_3tier_rc();

    // Seeded as plain direct writes: they become visible immediately, which is what puts
    // them in the visible region the capture walks. Rows left staging-pending are not
    // visible yet and would not reach the frontier, so the gate would pass while the
    // defect stood.
    for index in 0..UNRELATED_ROWS {
        s.a.insert(
            "users",
            user_values(ObjectId::new(), &format!("seed-{index}")),
            None,
        )
        .expect("seed an unrelated row");
    }

    let ((row_id, _values), _receiver) = insert_and_wait_for_batch(
        &mut s.a,
        "users",
        user_values(ObjectId::new(), "Alice"),
        Some(&transactional()),
        DurabilityTier::Local,
    )
    .expect("write the measured row");

    let history =
        s.a.storage()
            .scan_history_row_batches("users", row_id)
            .expect("read the written row's history");
    let batch_id = history
        .first()
        .expect("the write produced a history entry")
        .batch_id;

    // The submission is only persisted once the batch commits.
    s.a.commit_batch(batch_id).expect("commit the batch");

    let submission =
        s.a.storage()
            .load_sealed_batch_submission(batch_id)
            .expect("read the sealed submission")
            .expect("a transactional write seals a submission");

    assert!(
        submission.captured_frontier.len() <= 1,
        "sealing one row carried a frontier of {} members with {UNRELATED_ROWS} unrelated \
         rows in the store — the capture follows the size of the STORE, not the size of \
         the change. Nothing reads this payload (upstream PR #920); on a store holding \
         blob rows it costs a full read of every one of them per write.",
        submission.captured_frontier.len(),
    );
}

/// A tick must cost what there is to do, not what the store has kept.
///
/// `recover_completed_sealed_batches_with_storage` is the first statement of every
/// `immediate_tick` (`runtime_core/ticks.rs`), and it re-reads the WHOLE retained
/// sealed-submission table each time: a prefix scan, a decode per row (which resolves a
/// branch name per member ord), then a point-get of that batch's authoritative fate — and
/// then `continue`s on everything already fated. The work it can actually do is bounded by
/// the submissions that are still drivable; the price it pays is bounded by the table.
///
/// Retention makes that gap permanent rather than transient. A submission whose stored fate
/// is `Missing` can never be deleted (`fate_settled_at` does not call it terminal) and can
/// never be resolved (`fate_needs_settlement_at` excludes it), so it is re-read forever. A
/// production store measured 1284 of them — every one already fated, none drivable — at
/// 11.06 ms of storage reads per tick. Under a user simply typing into one chat draft that
/// was 70% of the server's CPU, and because ticks are serialized under the runtime mutex
/// and debounced at 1 ms, an 11 ms floor per tick does not merely burn CPU: it sets the
/// tick rate for every client on the process.
///
/// The gate asserts the SHAPE — how many fates the tick looked up and whether it walked the
/// table at all — not a duration and not bytes. A timing assertion would encode this
/// machine, and a bytes assertion would pass on a backend that answers from memory.
const RETAINED_BUT_UNDRIVABLE: usize = 200;

#[test]
fn a_tick_does_not_re_read_every_settled_submission() {
    let sweep = Arc::new(Mutex::new(SweepCallCounts::default()));
    let app_id = AppId::from_name("sealed-batch-sweep-cost");
    // The sweep returns immediately when `my_tiers` is empty, so a runtime without a
    // durability tier would gate nothing at all.
    let schema_manager = SchemaManager::new(
        SyncManager::new().with_durability_tier(DurabilityTier::Local),
        test_schema(),
        app_id,
        "dev",
        "main",
    )
    .unwrap();
    let mut core = new_test_core(
        schema_manager,
        Box::new(RowMutationObservingStorage::observing_sweep(Arc::clone(
            &sweep,
        ))) as Box<dyn Storage>,
        NoopScheduler,
    );

    // Every seeded submission carries a fate the sweep can do nothing with: confirmed above
    // this node's own tier, so `can_promote_direct_fate` is false and the loop `continue`s.
    // This is the population that accumulates in a real store, and it is seeded BEFORE the
    // first tick because that is how a process meets it: already on disk when it opens the
    // store. The first tick is allowed to look at every one of them once; the gate is about
    // every tick after that.
    for index in 0..RETAINED_BUT_UNDRIVABLE {
        let batch_id = BatchId::new();
        let row_id = ObjectId::new();
        core.storage_mut()
            .upsert_sealed_batch_submission(&SealedBatchSubmission::new(
                batch_id,
                crate::batch_fate::BatchMode::Direct,
                crate::object::BranchName::new("main"),
                vec![SealedBatchMember {
                    object_id: row_id,
                    row_digest: crate::digest::Digest32([index as u8; 32]),
                }],
                Vec::new(),
            ))
            .unwrap();
        core.storage_mut()
            .upsert_authoritative_batch_fate(&crate::batch_fate::BatchFate::DurableDirect {
                batch_id,
                confirmed_tier: DurabilityTier::GlobalServer,
            })
            .unwrap();
    }

    core.immediate_tick();

    *sweep.lock().unwrap() = SweepCallCounts::default();
    core.immediate_tick();
    let measured = *sweep.lock().unwrap();

    assert!(
        measured.authoritative_fate_gets <= 4,
        "a tick looked up {} batch fates with nothing to settle. Every retained submission \
         costs one point read per tick, forever: the ones seeded here are confirmed above \
         this node's tier, so the sweep reads each of them only to `continue`. The cost \
         must follow the drivable work ({} submissions are not drivable), not the size of \
         the table. Scans of the submission table this tick: {}",
        measured.authoritative_fate_gets,
        RETAINED_BUT_UNDRIVABLE,
        measured.sealed_submission_scans
    );
}

/// A tick must not read the rows of submissions it cannot drive.
///
/// The sweep's discriminator is cheap — one point read of the batch's authoritative fate —
/// and its payload is expensive: reading a submission row means decoding it, and every
/// decode resolves a branch name by ord, a second random read. Doing the expensive half
/// first meant a store full of already-fated submissions paid for a decode and a
/// branch-name lookup per row, every tick, to learn each time that there was nothing to do.
///
/// Measured on a production store: 1284 retained submissions, none drivable, 11.06 ms per
/// tick — of which 8.75 ms was the value scan and its decodes and only 2.31 ms the fate
/// reads. This gate pins the order, not the timing: fate first, row only for survivors.
#[test]
fn a_tick_does_not_read_the_rows_of_submissions_it_cannot_drive() {
    let sweep = Arc::new(Mutex::new(SweepCallCounts::default()));
    let app_id = AppId::from_name("sealed-batch-sweep-order");
    let schema_manager = SchemaManager::new(
        SyncManager::new().with_durability_tier(DurabilityTier::Local),
        test_schema(),
        app_id,
        "dev",
        "main",
    )
    .unwrap();
    let mut core = new_test_core(
        schema_manager,
        Box::new(RowMutationObservingStorage::observing_sweep(Arc::clone(
            &sweep,
        ))) as Box<dyn Storage>,
        NoopScheduler,
    );

    for index in 0..RETAINED_BUT_UNDRIVABLE {
        let batch_id = BatchId::new();
        core.storage_mut()
            .upsert_sealed_batch_submission(&SealedBatchSubmission::new(
                batch_id,
                crate::batch_fate::BatchMode::Direct,
                crate::object::BranchName::new("main"),
                vec![SealedBatchMember {
                    object_id: ObjectId::new(),
                    row_digest: crate::digest::Digest32([index as u8; 32]),
                }],
                Vec::new(),
            ))
            .unwrap();
        core.storage_mut()
            .upsert_authoritative_batch_fate(&crate::batch_fate::BatchFate::DurableDirect {
                batch_id,
                confirmed_tier: DurabilityTier::GlobalServer,
            })
            .unwrap();
    }

    core.immediate_tick();

    *sweep.lock().unwrap() = SweepCallCounts::default();
    core.immediate_tick();
    let measured = *sweep.lock().unwrap();

    assert_eq!(
        (
            measured.submission_row_reads,
            measured.branch_name_gets,
            measured.sealed_submission_scans
        ),
        (0, 0, 0),
        "a tick read {} submission rows, {} branch names and did {} value scans of the \
         submission table, with {} retained submissions and none of them drivable. The fate \
         of each batch already said so before any row was touched: the row read, its decode, \
         and the branch-name lookup the decode performs are all work spent to reach a \
         `continue`.",
        measured.submission_row_reads,
        measured.branch_name_gets,
        measured.sealed_submission_scans,
        RETAINED_BUT_UNDRIVABLE
    );
}

// ------------------------------------------------------------------------------------------
// The sweep must be driven by what changed, not by what the store has kept.
//
// A production store measured on 2026-09-08 held 2258 sealed submissions with no fate and
// no rows: the seal reached the server, the rows never did (the 2026-08-09 incident, defect
// #11), and nothing ever writes a fate for a submission whose rows are absent, so nothing
// ever deletes it. Every `immediate_tick` re-read every one of them — a fate get, a row read
// with its decode, a row-index lookup — only to reach `continue` on `declared_rows_for_
// submission`. On the 2026-09-08 production profile that was 62.6 % of the server's CPU,
// and on the prod-store stand a read cost 74 ms against 22 ms with the table
// emptied; the slope was 0.021 ms per retained submission on P-cores, 0.116 on E-cores.
//
// The gates below pin the shape the fix has to have: a submission is examined by a tick
// only when something about it changed since the last tick — it was sealed, one of its rows
// arrived, its fate was written — or when the process opened the store and has not looked
// yet. A submission nothing happened to costs a tick nothing. And they pin what the sweep is
// FOR, so the cost cannot be bought by dropping the work: every seal that can be completed
// still is.

/// The population from the production store: sealed, fate-less, row-less, undrivable.
const FATELESS_AND_ROWLESS: usize = 200;

/// A store full of seals nothing can happen to must cost a tick nothing.
///
/// This is the production population exactly: 2258 submissions with no fate and no rows.
/// They are not drivable — no rows, no settlement — and they are not deletable — no fate,
/// no retirement — so they are permanent. The first tick after the store is opened may look
/// at each of them once; that walk is how the process learns what it is holding. Every tick
/// after that has no reason to touch them: nothing about them changed, and the only events
/// that could change them (a row arriving, a fate being written) go through the sync manager,
/// which is the same place the sweep lives.
#[test]
fn a_tick_does_not_re_read_submissions_it_could_not_drive_last_time() {
    let (sweep, mut core) = sweeping_core(DurabilityTier::Local, "sealed-batch-sweep-fateless");

    for index in 0..FATELESS_AND_ROWLESS {
        core.storage_mut()
            .upsert_sealed_batch_submission(&SealedBatchSubmission::new(
                BatchId::new(),
                crate::batch_fate::BatchMode::Direct,
                crate::object::BranchName::new("main"),
                vec![SealedBatchMember {
                    object_id: ObjectId::new(),
                    row_digest: crate::digest::Digest32([index as u8; 32]),
                }],
                Vec::new(),
            ))
            .unwrap();
    }

    // Opening the store: one walk over the table is the price of learning what it holds.
    // (The seeding above walks it too, once, to check the table header; that is not the
    // tick's doing.)
    *sweep.lock().unwrap() = SweepCallCounts::default();
    core.immediate_tick();
    let opening = *sweep.lock().unwrap();
    assert_eq!(
        opening.sealed_submission_key_scans, 1,
        "the first tick after opening the store walks the submission table's ids exactly \
         once; got {} walks",
        opening.sealed_submission_key_scans
    );

    *sweep.lock().unwrap() = SweepCallCounts::default();
    core.immediate_tick();
    let measured = *sweep.lock().unwrap();

    assert_eq!(
        (
            measured.authoritative_fate_gets,
            measured.submission_row_reads,
            measured.sealed_submission_key_scans + measured.sealed_submission_scans
        ),
        (0, 0, 0),
        "a tick with nothing to do looked up {} fates, read {} submission rows and walked \
         the submission table {} times, over {} retained submissions that have no fate and \
         no rows. Nothing happened to any of them since the previous tick, so there was \
         nothing to learn: this is the per-tick floor a production server paid for 2258 \
         permanently uncompletable seals — 62.6 % of its CPU and 70 % of every read's \
         latency. The cost of a tick must follow what changed, not what the store kept.",
        measured.authoritative_fate_gets,
        measured.submission_row_reads,
        measured.sealed_submission_key_scans + measured.sealed_submission_scans,
        FATELESS_AND_ROWLESS
    );
}

// ---- What the sweep is for. These hold before and after the fix; they exist so the cost
// ---- gates above cannot be satisfied by a sweep that stopped doing its job.

/// A seal whose rows are already here, left on disk without a fate, is completed by the
/// first tick after the store is reopened.
///
/// The shape a crash leaves behind: the rows were applied and the submission persisted,
/// and the process died before the completion that normally follows the persist in the
/// same call. Nothing will ever re-send these; the sweep is the only thing that finishes
/// them, and after the fix it is the opening walk that must do so.
#[test]
fn a_completable_seal_left_on_disk_is_settled_by_the_first_tick_after_reopen() {
    let (_, mut core) = sweeping_core(DurabilityTier::Local, "sealed-batch-sweep-reopen");
    let client_id = peer_client(&mut core);
    let batch_id = BatchId::new();
    let row_id = ObjectId::new();
    let row = user_row(
        row_id,
        batch_id,
        crate::row_histories::RowState::VisibleDirect,
        None,
    );

    // The row arrives through the engine, as it would in production, so the batch row
    // index the sweep reads through is written by the same code that writes it live.
    deliver(
        &mut core,
        Source::Client(client_id),
        SyncPayload::RowBatchCreated {
            metadata: Some(users_row_metadata(row_id)),
            row: row.clone(),
        },
    );
    assert_eq!(
        core.storage()
            .load_authoritative_batch_fate(batch_id)
            .unwrap(),
        None,
        "a row without a seal must not acquire a fate on its own"
    );

    // The crash window: the seal reached the disk, the completion that follows it did not.
    let mut storage = core.into_storage();
    storage
        .upsert_sealed_batch_submission(&seal_declaring(
            batch_id,
            crate::batch_fate::BatchMode::Direct,
            &[row],
        ))
        .unwrap();

    let mut core = sweeping_core_over(storage, DurabilityTier::Local, "sealed-batch-sweep-reopen");
    core.immediate_tick();

    assert_eq!(
        core.storage()
            .load_authoritative_batch_fate(batch_id)
            .unwrap(),
        Some(crate::batch_fate::BatchFate::DurableDirect {
            batch_id,
            confirmed_tier: DurabilityTier::Local,
        }),
        "a seal whose declared rows are all on disk is completable, and the first tick after \
         reopening the store must complete it: nothing else ever will"
    );
    assert_eq!(
        core.storage()
            .load_sealed_batch_submission(batch_id)
            .unwrap(),
        None,
        "a completed seal must retire its submission"
    );
}

/// A row that reaches a retained seal by a path that does not complete seals is still
/// completed by the next tick.
///
/// Every client-origin row arm re-checks the seal it belongs to; the server-origin arm
/// applies the row and stops (`sync_manager/tests/server_origin_seal.rs`). A staged
/// transactional row from upstream therefore leaves its seal completable and untouched,
/// and the sweep is what completes it. This is the event the fix must not lose: the seal
/// was examined once, could not be driven, and became drivable later by a row arriving —
/// so the row's arrival is what has to put it back in front of the sweep.
#[test]
fn a_seal_made_completable_by_a_row_the_client_arms_never_saw_is_settled_by_the_next_tick() {
    let (sweep, mut core) = sweeping_core(DurabilityTier::Local, "sealed-batch-sweep-late-row");
    let client_id = peer_client(&mut core);
    let server_id = ServerId::new();
    core.add_server(server_id);
    let batch_id = BatchId::new();
    let row_id = ObjectId::new();
    let row = user_row(
        row_id,
        batch_id,
        crate::row_histories::RowState::StagingPending,
        None,
    );

    // The seal first, from the client: its row is not here yet, so it cannot be completed.
    deliver(
        &mut core,
        Source::Client(client_id),
        SyncPayload::SealBatch {
            submission: seal_declaring(
                batch_id,
                crate::batch_fate::BatchMode::Transactional,
                &[row.clone()],
            ),
        },
    );
    core.immediate_tick();
    assert_eq!(
        core.storage()
            .load_authoritative_batch_fate(batch_id)
            .unwrap(),
        None,
        "a seal with no rows is not completable and must not be fated"
    );
    assert!(
        core.storage()
            .load_sealed_batch_submission(batch_id)
            .unwrap()
            .is_some(),
        "an incomplete seal is retained until its rows arrive"
    );

    // Then the row, by the one path that applies it without re-checking the seal.
    *sweep.lock().unwrap() = SweepCallCounts::default();
    deliver(
        &mut core,
        Source::Server(server_id),
        SyncPayload::RowBatchNeeded {
            metadata: Some(users_row_metadata(row_id)),
            row,
        },
    );
    core.immediate_tick();

    assert_eq!(
        core.storage()
            .load_authoritative_batch_fate(batch_id)
            .unwrap(),
        Some(crate::batch_fate::BatchFate::AcceptedTransaction {
            batch_id,
            confirmed_tier: DurabilityTier::Local,
        }),
        "the last row this seal was waiting for has arrived, by a path that does not \
         complete seals, so the next tick's sweep must: a submission that was undrivable \
         when the sweep last saw it and became drivable since must be seen again"
    );
    // The submission itself is retained here, correctly: with an upstream registered this
    // node's settlement target is `GlobalServer`, and a transaction accepted at `Local` is
    // not terminal there — the seal waits for the upstream's fate. Retirement is that
    // fate's business (`persist_authoritative_batch_fate`'s callers), not the sweep's.
    assert!(
        core.storage()
            .load_sealed_batch_submission(batch_id)
            .unwrap()
            .is_some(),
        "an accepted-at-Local transaction on a node with an upstream is not settled yet; \
         its submission is retained until the upstream's fate arrives"
    );
}

/// A tiered node's own transactional write is settled by the tick that follows its commit.
///
/// `commit_batch` persists the submission and ends in an `immediate_tick`; on a node with a
/// durability tier and no upstream, that tick's sweep is what turns the staged rows into an
/// accepted transaction. The persist happens in `runtime_core`, outside the sync manager —
/// the one seal write the sync manager does not see go by.
#[test]
fn a_tiered_node_settles_its_own_transactional_write_at_commit() {
    let (_, mut core) = sweeping_core(DurabilityTier::Local, "sealed-batch-sweep-local-write");
    core.immediate_tick();

    let (_row, batch_id) = core
        .insert(
            "users",
            user_values(ObjectId::new(), "Alice"),
            Some(&transactional()),
        )
        .expect("stage a transactional row");
    core.commit_batch(batch_id).expect("commit the batch");

    assert_eq!(
        core.storage()
            .load_authoritative_batch_fate(batch_id)
            .unwrap(),
        Some(crate::batch_fate::BatchFate::AcceptedTransaction {
            batch_id,
            confirmed_tier: DurabilityTier::Local,
        }),
        "a committed transactional write on a node that is its own authority must be \
         accepted by the tick that commit ends in"
    );
    assert_eq!(
        core.storage()
            .load_sealed_batch_submission(batch_id)
            .unwrap(),
        None,
        "a settled seal must retire its submission"
    );
}

/// A direct fate confirmed below this node's tier is promoted by the first tick after the
/// store is reopened.
///
/// The one case in which a FATED submission is still drivable: `can_promote_direct_fate`.
/// It must survive the fix's opening walk exactly as it survives today's full scan.
#[test]
fn a_direct_fate_confirmed_below_this_tier_is_promoted_by_the_first_tick_after_reopen() {
    let (_, mut core) = sweeping_core(DurabilityTier::EdgeServer, "sealed-batch-sweep-promote");
    let client_id = peer_client(&mut core);
    let batch_id = BatchId::new();
    let row_id = ObjectId::new();
    let row = user_row(
        row_id,
        batch_id,
        crate::row_histories::RowState::VisibleDirect,
        Some(DurabilityTier::Local),
    );
    deliver(
        &mut core,
        Source::Client(client_id),
        SyncPayload::RowBatchCreated {
            metadata: Some(users_row_metadata(row_id)),
            row: row.clone(),
        },
    );

    let mut storage = core.into_storage();
    storage
        .upsert_sealed_batch_submission(&seal_declaring(
            batch_id,
            crate::batch_fate::BatchMode::Direct,
            &[row],
        ))
        .unwrap();
    storage
        .upsert_authoritative_batch_fate(&crate::batch_fate::BatchFate::DurableDirect {
            batch_id,
            confirmed_tier: DurabilityTier::Local,
        })
        .unwrap();

    let mut core = sweeping_core_over(
        storage,
        DurabilityTier::EdgeServer,
        "sealed-batch-sweep-promote",
    );
    core.immediate_tick();

    assert_eq!(
        core.storage()
            .load_authoritative_batch_fate(batch_id)
            .unwrap(),
        Some(crate::batch_fate::BatchFate::DurableDirect {
            batch_id,
            confirmed_tier: DurabilityTier::EdgeServer,
        }),
        "a direct fate confirmed by a lower tier is drivable on this one: the opening walk \
         must promote it as the full scan did"
    );
    assert_eq!(
        core.storage()
            .load_sealed_batch_submission(batch_id)
            .unwrap(),
        None,
        "a promoted seal is settled at this tier and must retire its submission"
    );
}

// ---- The cost side, positively: what a tick DOES look at is what changed.

/// After the opening walk, a tick reads the fates of the batches touched since the last
/// tick and nothing else.
#[test]
fn a_tick_reads_only_the_batches_touched_since_the_last_one() {
    let (sweep, mut core) = sweeping_core(DurabilityTier::Local, "sealed-batch-sweep-touched");
    let client_id = peer_client(&mut core);

    for index in 0..FATELESS_AND_ROWLESS {
        core.storage_mut()
            .upsert_sealed_batch_submission(&SealedBatchSubmission::new(
                BatchId::new(),
                crate::batch_fate::BatchMode::Direct,
                crate::object::BranchName::new("main"),
                vec![SealedBatchMember {
                    object_id: ObjectId::new(),
                    row_digest: crate::digest::Digest32([index as u8; 32]),
                }],
                Vec::new(),
            ))
            .unwrap();
    }
    core.immediate_tick();

    // One batch happens: its row, then its seal, from a client. The client arm completes
    // it on arrival; the sweep's share is to confirm there is nothing left to do.
    let batch_id = BatchId::new();
    let row_id = ObjectId::new();
    let row = user_row(
        row_id,
        batch_id,
        crate::row_histories::RowState::VisibleDirect,
        None,
    );
    deliver(
        &mut core,
        Source::Client(client_id),
        SyncPayload::RowBatchCreated {
            metadata: Some(users_row_metadata(row_id)),
            row: row.clone(),
        },
    );
    deliver(
        &mut core,
        Source::Client(client_id),
        SyncPayload::SealBatch {
            submission: seal_declaring(batch_id, crate::batch_fate::BatchMode::Direct, &[row]),
        },
    );
    assert!(
        core.storage()
            .load_authoritative_batch_fate(batch_id)
            .unwrap()
            .is_some(),
        "the client arm completes a seal whose rows are present"
    );

    *sweep.lock().unwrap() = SweepCallCounts::default();
    core.immediate_tick();
    let measured = *sweep.lock().unwrap();

    assert!(
        measured.authoritative_fate_gets <= 3 && measured.sealed_submission_key_scans == 0,
        "one batch was touched since the previous tick, and the tick looked up {} fates \
         and walked the table {} times over {} retained submissions nothing happened to. \
         What a tick reads must be what changed.",
        measured.authoritative_fate_gets,
        measured.sealed_submission_key_scans,
        FATELESS_AND_ROWLESS
    );
}

// ---- What must not be lost when the sweep stops re-reading everything.

/// A read that fails is not a submission that cannot be driven: the batch is examined
/// again by the next tick, and completed once the store answers.
///
/// Today's sweep re-reads everything every tick, so a failed read was retried for free.
/// A sweep that examines only what changed has to treat "could not read it" as a change.
#[test]
fn a_seal_whose_fate_read_failed_is_settled_by_the_tick_after_the_store_recovers() {
    let sweep = Arc::new(Mutex::new(SweepCallCounts::default()));
    let fail_fate_gets = Arc::new(Mutex::new(false));
    let storage = Box::new(RowMutationObservingStorage::observing_sweep_with_faults(
        Arc::clone(&sweep),
        Arc::clone(&fail_fate_gets),
        Arc::new(Mutex::new(false)),
    )) as Box<dyn Storage>;
    let mut core = sweeping_core_over(storage, DurabilityTier::Local, "sealed-batch-sweep-fault");
    let client_id = peer_client(&mut core);
    core.immediate_tick();

    let batch_id = BatchId::new();
    let row_id = ObjectId::new();
    let row = user_row(
        row_id,
        batch_id,
        crate::row_histories::RowState::StagingPending,
        None,
    );
    deliver(
        &mut core,
        Source::Client(client_id),
        SyncPayload::SealBatch {
            submission: seal_declaring(
                batch_id,
                crate::batch_fate::BatchMode::Transactional,
                &[row.clone()],
            ),
        },
    );
    core.immediate_tick();
    // The row lands by the path that leaves completion to the sweep, and the store fails
    // the sweep's first look.
    *fail_fate_gets.lock().unwrap() = true;
    deliver(
        &mut core,
        Source::Server(ServerId::new()),
        SyncPayload::RowBatchNeeded {
            metadata: Some(users_row_metadata(row_id)),
            row,
        },
    );
    core.immediate_tick();
    *fail_fate_gets.lock().unwrap() = false;
    assert_eq!(
        core.storage()
            .load_authoritative_batch_fate(batch_id)
            .unwrap(),
        None,
        "while the store fails fate reads nothing can be settled"
    );

    core.immediate_tick();
    assert_eq!(
        core.storage()
            .load_authoritative_batch_fate(batch_id)
            .unwrap(),
        Some(crate::batch_fate::BatchFate::AcceptedTransaction {
            batch_id,
            confirmed_tier: DurabilityTier::Local,
        }),
        "the store answered on the next tick, and the seal was completable all along: a \
         read that failed must put the batch back in front of the sweep, not drop it \
         until the next restart"
    );
}

/// The sweep's third read of a seal — its rows through the local batch row index — can
/// fail like the other two, and a seal whose rows could not be read is not a seal whose
/// rows are absent.
#[test]
fn a_seal_whose_row_index_read_failed_is_settled_by_the_tick_after_the_store_recovers() {
    let sweep = Arc::new(Mutex::new(SweepCallCounts::default()));
    let fail_row_index_gets = Arc::new(Mutex::new(false));
    let storage = Box::new(
        RowMutationObservingStorage::observing_sweep(Arc::clone(&sweep))
            .failing_row_index_gets_when(Arc::clone(&fail_row_index_gets)),
    ) as Box<dyn Storage>;
    let mut core = sweeping_core_over(
        storage,
        DurabilityTier::Local,
        "sealed-batch-sweep-row-index-fault",
    );
    let client_id = peer_client(&mut core);
    core.immediate_tick();

    let batch_id = BatchId::new();
    let row_id = ObjectId::new();
    let row = user_row(
        row_id,
        batch_id,
        crate::row_histories::RowState::StagingPending,
        None,
    );
    deliver(
        &mut core,
        Source::Client(client_id),
        SyncPayload::SealBatch {
            submission: seal_declaring(
                batch_id,
                crate::batch_fate::BatchMode::Transactional,
                &[row.clone()],
            ),
        },
    );
    core.immediate_tick();
    // The row lands by the path that leaves completion to the sweep, and the store fails
    // the sweep's read of the batch's rows.
    deliver(
        &mut core,
        Source::Server(ServerId::new()),
        SyncPayload::RowBatchNeeded {
            metadata: Some(users_row_metadata(row_id)),
            row,
        },
    );
    *fail_row_index_gets.lock().unwrap() = true;
    core.immediate_tick();
    *fail_row_index_gets.lock().unwrap() = false;
    assert_eq!(
        core.storage()
            .load_authoritative_batch_fate(batch_id)
            .unwrap(),
        None,
        "while the store fails row index reads nothing can be settled"
    );

    core.immediate_tick();
    assert_eq!(
        core.storage()
            .load_authoritative_batch_fate(batch_id)
            .unwrap(),
        Some(crate::batch_fate::BatchFate::AcceptedTransaction {
            batch_id,
            confirmed_tier: DurabilityTier::Local,
        }),
        "the rows were there all along: a row read that failed must put the batch back in \
         front of the sweep, not pass for rows that never arrived"
    );
}

/// The settle itself can fail at its write: a fate the store refused to take is not a fate,
/// and the seal it belongs to has to be driven again.
#[test]
fn a_seal_whose_fate_write_failed_is_settled_by_the_tick_after_the_store_recovers() {
    let sweep = Arc::new(Mutex::new(SweepCallCounts::default()));
    let fail_fate_puts = Arc::new(Mutex::new(false));
    let storage = Box::new(
        RowMutationObservingStorage::observing_sweep(Arc::clone(&sweep))
            .failing_fate_puts_when(Arc::clone(&fail_fate_puts)),
    ) as Box<dyn Storage>;
    let mut core = sweeping_core_over(
        storage,
        DurabilityTier::Local,
        "sealed-batch-sweep-fate-write-fault",
    );
    let client_id = peer_client(&mut core);
    core.immediate_tick();

    let batch_id = BatchId::new();
    let row_id = ObjectId::new();
    let row = user_row(
        row_id,
        batch_id,
        crate::row_histories::RowState::StagingPending,
        None,
    );
    deliver(
        &mut core,
        Source::Client(client_id),
        SyncPayload::SealBatch {
            submission: seal_declaring(
                batch_id,
                crate::batch_fate::BatchMode::Transactional,
                &[row.clone()],
            ),
        },
    );
    core.immediate_tick();
    deliver(
        &mut core,
        Source::Server(ServerId::new()),
        SyncPayload::RowBatchNeeded {
            metadata: Some(users_row_metadata(row_id)),
            row,
        },
    );
    // Completable now; the settle's write of the fate is what the store refuses.
    *fail_fate_puts.lock().unwrap() = true;
    core.immediate_tick();
    *fail_fate_puts.lock().unwrap() = false;
    assert_eq!(
        core.storage()
            .load_authoritative_batch_fate(batch_id)
            .unwrap(),
        None,
        "a fate the store refused must not be reported as written"
    );
    assert!(
        core.storage()
            .load_sealed_batch_submission(batch_id)
            .unwrap()
            .is_some(),
        "a seal whose settle failed keeps its submission"
    );

    core.immediate_tick();
    assert_eq!(
        core.storage()
            .load_authoritative_batch_fate(batch_id)
            .unwrap(),
        Some(crate::batch_fate::BatchFate::AcceptedTransaction {
            batch_id,
            confirmed_tier: DurabilityTier::Local,
        }),
        "the store took the write on the next tick: a fate write that failed must put the \
         batch back in front of the sweep"
    );
    assert_eq!(
        core.storage()
            .load_sealed_batch_submission(batch_id)
            .unwrap(),
        None,
        "a completed seal must retire its submission"
    );
}

/// Two transactions on the same row, examined by one sweep, settle in the order they were
/// made: the one that builds on the other after it.
///
/// `validate_transactional_parent_frontiers` compares a staged row's declared parents with
/// the row's CURRENT visible frontier. Examined first, a child transaction sees a frontier
/// its parent has not joined yet and is rejected with `transaction_conflict` — terminal,
/// and sticky. Batch ids are time-ordered, and the sweep must visit them in that order
/// whatever structure holds them.
#[test]
fn a_sweep_settles_dependent_transactions_parent_first() {
    let (_, mut core) = sweeping_core(DurabilityTier::Local, "sealed-batch-sweep-order");
    let client_id = peer_client(&mut core);
    let upstream = ServerId::new();
    core.immediate_tick();

    let row_id = ObjectId::new();
    let parent_batch = BatchId::new();
    let child_batch = BatchId::new();
    assert!(
        parent_batch < child_batch,
        "batch ids are minted in time order"
    );
    let parent_row = user_row(
        row_id,
        parent_batch,
        crate::row_histories::RowState::StagingPending,
        None,
    );
    let child_row = crate::row_histories::StoredRowBatch::new_with_batch_id(
        child_batch,
        row_id,
        "main",
        vec![parent_batch],
        encode_row(
            &test_schema()[&TableName::new("users")].columns,
            &user_row_values(row_id, "alice again"),
        )
        .expect("user test row should encode"),
        crate::metadata::RowProvenance::for_insert(row_id.to_string(), 2_000),
        HashMap::new(),
        crate::row_histories::RowState::StagingPending,
        None,
    );

    // Both seals arrive first, then both rows by the path that leaves completion to the
    // sweep — parked together, so one tick applies both and one sweep examines both.
    for (batch_id, row) in [(parent_batch, &parent_row), (child_batch, &child_row)] {
        deliver(
            &mut core,
            Source::Client(client_id),
            SyncPayload::SealBatch {
                submission: seal_declaring(
                    batch_id,
                    crate::batch_fate::BatchMode::Transactional,
                    std::slice::from_ref(row),
                ),
            },
        );
    }
    core.immediate_tick();
    for row in [parent_row, child_row] {
        core.park_sync_message(InboxEntry {
            source: Source::Server(upstream),
            payload: SyncPayload::RowBatchNeeded {
                metadata: Some(users_row_metadata(row_id)),
                row,
            },
        });
    }
    core.batched_tick();
    core.immediate_tick();

    for batch_id in [parent_batch, child_batch] {
        assert_eq!(
            core.storage()
                .load_authoritative_batch_fate(batch_id)
                .unwrap(),
            Some(crate::batch_fate::BatchFate::AcceptedTransaction {
                batch_id,
                confirmed_tier: DurabilityTier::Local,
            }),
            "both transactions are valid in the order they were made; a sweep that examined \
             the child before its parent would have rejected it for a conflict that does \
             not exist"
        );
    }
}

/// A node with no durability tier never sweeps, so it must not remember what it would
/// have swept: every transactional write a phone commits would otherwise stay in memory
/// for the life of the process.
///
/// (Rows a client receives from its server do not reach the set even without the guard:
/// the apply records a fate at the server's tier, which is final for the sweep. The
/// node's own commits, and staged rows from upstream, do.)
#[test]
fn a_node_without_a_durability_tier_keeps_no_sweep_backlog() {
    let schema_manager = SchemaManager::new(
        SyncManager::new(),
        test_schema(),
        AppId::from_name("sealed-batch-sweep-tierless"),
        "dev",
        "main",
    )
    .unwrap();
    let mut core = new_test_core(
        schema_manager,
        Box::new(MemoryStorage::new()) as Box<dyn Storage>,
        NoopScheduler,
    );
    core.immediate_tick();

    const COMMITS: usize = 50;
    for index in 0..COMMITS {
        let (_row, batch_id) = core
            .insert(
                "users",
                user_values(ObjectId::new(), &format!("phone-{index}")),
                Some(&transactional()),
            )
            .expect("stage a transactional row");
        core.commit_batch(batch_id).expect("commit the batch");
    }
    core.immediate_tick();

    let backlog = core
        .schema_manager_mut()
        .query_manager_mut()
        .sync_manager_mut()
        .sealed_batches_awaiting_sweep();
    assert_eq!(
        backlog, 0,
        "a client engine committed {COMMITS} transactional batches and is holding \
         {backlog} of them for a sweep it will never run; on a phone that is every write \
         it ever makes, kept until the app is killed"
    );
}

/// A seal that reaches the store by way of a hydrated batch record is completed like any
/// other.
///
/// `hydrate_local_batch_record` persists the record's embedded submission through the
/// storage layer, outside the sync manager: the browser worker rebuilds its records this
/// way after boot. A seal it carries, with rows present and no fate, is drivable.
#[test]
fn a_seal_hydrated_from_a_batch_record_is_settled_by_the_next_tick() {
    let (_, mut core) = sweeping_core(DurabilityTier::Local, "sealed-batch-sweep-hydrate");
    let client_id = peer_client(&mut core);
    core.immediate_tick();

    let batch_id = BatchId::new();
    let row_id = ObjectId::new();
    let row = user_row(
        row_id,
        batch_id,
        crate::row_histories::RowState::StagingPending,
        None,
    );
    deliver(
        &mut core,
        Source::Client(client_id),
        SyncPayload::RowBatchCreated {
            metadata: Some(users_row_metadata(row_id)),
            row: row.clone(),
        },
    );
    core.immediate_tick();

    let mut record = crate::batch_fate::LocalBatchRecord::new(
        batch_id,
        crate::batch_fate::BatchMode::Transactional,
        true,
        None,
    );
    record.mark_sealed(seal_declaring(
        batch_id,
        crate::batch_fate::BatchMode::Transactional,
        &[row],
    ));
    core.hydrate_local_batch_record(record)
        .expect("hydrate the batch record");
    core.immediate_tick();

    assert_eq!(
        core.storage()
            .load_authoritative_batch_fate(batch_id)
            .unwrap(),
        Some(crate::batch_fate::BatchFate::AcceptedTransaction {
            batch_id,
            confirmed_tier: DurabilityTier::Local,
        }),
        "the hydrated record carried a completable seal; the tick after hydration must \
         settle it, the way it settles a seal that arrived through the inbox"
    );
}

/// The reopen gate on the backend the server runs: the seal, the row and its batch row
/// index read back from bytes written by a process that is gone.
#[test]
fn a_completable_seal_left_on_disk_is_settled_by_the_first_tick_after_reopen_on_sqlite() {
    let path = std::env::temp_dir().join(format!(
        "jazz-sealed-batch-sweep-reopen-{}.sqlite",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);
    let storage = Box::new(crate::storage::SqliteStorage::open(&path).expect("open sqlite"))
        as Box<dyn Storage>;
    let mut core = sweeping_core_over(storage, DurabilityTier::Local, "sealed-batch-sweep-sqlite");
    let client_id = peer_client(&mut core);
    core.immediate_tick();

    let batch_id = BatchId::new();
    let row_id = ObjectId::new();
    let row = user_row(
        row_id,
        batch_id,
        crate::row_histories::RowState::VisibleDirect,
        None,
    );
    deliver(
        &mut core,
        Source::Client(client_id),
        SyncPayload::RowBatchCreated {
            metadata: Some(users_row_metadata(row_id)),
            row: row.clone(),
        },
    );
    core.batched_tick();
    drop(core);

    let mut storage = crate::storage::SqliteStorage::open(&path).expect("reopen sqlite");
    storage
        .upsert_sealed_batch_submission(&seal_declaring(
            batch_id,
            crate::batch_fate::BatchMode::Direct,
            &[row],
        ))
        .unwrap();
    let mut core = sweeping_core_over(
        Box::new(storage) as Box<dyn Storage>,
        DurabilityTier::Local,
        "sealed-batch-sweep-sqlite",
    );
    core.immediate_tick();

    let fate = core
        .storage()
        .load_authoritative_batch_fate(batch_id)
        .unwrap();
    let submission = core
        .storage()
        .load_sealed_batch_submission(batch_id)
        .unwrap();
    let _ = std::fs::remove_file(&path);
    assert_eq!(
        fate,
        Some(crate::batch_fate::BatchFate::DurableDirect {
            batch_id,
            confirmed_tier: DurabilityTier::Local,
        }),
        "reopened from bytes, the seal and its row are complete and the first tick must \
         settle them"
    );
    assert_eq!(
        submission, None,
        "a completed seal must retire its submission"
    );
}

/// A seal whose completion on arrival failed transiently is settled by the next tick.
///
/// The client arm persists the submission and then completes it in the same call; if the
/// completion's own read of the submission fails, the arm returns and nothing re-sends
/// the seal. Persisting a seal is what puts it in front of the sweep, whatever happens in
/// the rest of the call.
#[test]
fn a_seal_whose_completion_on_arrival_failed_is_settled_by_the_next_tick() {
    let sweep = Arc::new(Mutex::new(SweepCallCounts::default()));
    let fail_submission_gets = Arc::new(Mutex::new(false));
    let storage = Box::new(RowMutationObservingStorage::observing_sweep_with_faults(
        Arc::clone(&sweep),
        Arc::new(Mutex::new(false)),
        Arc::clone(&fail_submission_gets),
    )) as Box<dyn Storage>;
    let mut core = sweeping_core_over(
        storage,
        DurabilityTier::Local,
        "sealed-batch-sweep-arrival-fault",
    );
    let client_id = peer_client(&mut core);
    core.immediate_tick();

    let batch_id = BatchId::new();
    let row_id = ObjectId::new();
    let row = user_row(
        row_id,
        batch_id,
        crate::row_histories::RowState::VisibleDirect,
        None,
    );
    deliver(
        &mut core,
        Source::Client(client_id),
        SyncPayload::RowBatchCreated {
            metadata: Some(users_row_metadata(row_id)),
            row: row.clone(),
        },
    );
    core.immediate_tick();

    // The seal is written; the completion that follows cannot read it back.
    *fail_submission_gets.lock().unwrap() = true;
    core.park_sync_message(InboxEntry {
        source: Source::Client(client_id),
        payload: SyncPayload::SealBatch {
            submission: seal_declaring(batch_id, crate::batch_fate::BatchMode::Direct, &[row]),
        },
    });
    core.batched_tick();
    // The store is still failing when the sweep takes its own look at the seal.
    let reads_before = sweep.lock().unwrap().submission_row_reads;
    core.immediate_tick();
    *fail_submission_gets.lock().unwrap() = false;
    assert!(
        sweep.lock().unwrap().submission_row_reads > reads_before,
        "the sweep must have tried to read the seal it was handed"
    );
    assert!(
        core.storage()
            .load_sealed_batch_submission(batch_id)
            .unwrap()
            .is_some(),
        "the seal was persisted before its completion failed"
    );
    assert_eq!(
        core.storage()
            .load_authoritative_batch_fate(batch_id)
            .unwrap(),
        None,
        "with its read failing the seal could not be completed on arrival or by the sweep"
    );

    core.immediate_tick();
    assert_eq!(
        core.storage()
            .load_authoritative_batch_fate(batch_id)
            .unwrap(),
        Some(crate::batch_fate::BatchFate::DurableDirect {
            batch_id,
            confirmed_tier: DurabilityTier::Local,
        }),
        "the seal and its row are on disk and nothing will re-send them: the tick after \
         the store recovers must settle the batch, however many reads of it failed"
    );
    assert_eq!(
        core.storage()
            .load_sealed_batch_submission(batch_id)
            .unwrap(),
        None,
        "a completed seal must retire its submission"
    );
}

/// A row parked for a parent it has not seen yet, drained when the parent arrives, still
/// completes its seal by the next tick.
///
/// Rows do not always apply on arrival: one whose declared parent is unknown is parked
/// (`park_failed_row_batch`) and applied later by `drain_parked_row_batches`, when the
/// parent lands. That later apply is the event that makes the seal completable, and it
/// runs by the same code as a first-attempt apply — this gate is what says so.
#[test]
fn a_row_parked_for_a_missing_parent_completes_its_seal_when_drained() {
    let (_, mut core) = sweeping_core(DurabilityTier::Local, "sealed-batch-sweep-parked");
    let client_id = peer_client(&mut core);
    let upstream = ServerId::new();
    core.immediate_tick();

    let row_id = ObjectId::new();
    let parent_batch = BatchId::new();
    let child_batch = BatchId::new();
    let parent_row = user_row(
        row_id,
        parent_batch,
        crate::row_histories::RowState::StagingPending,
        None,
    );
    let child_row = crate::row_histories::StoredRowBatch::new_with_batch_id(
        child_batch,
        row_id,
        "main",
        vec![parent_batch],
        encode_row(
            &test_schema()[&TableName::new("users")].columns,
            &user_row_values(row_id, "alice again"),
        )
        .expect("user test row should encode"),
        crate::metadata::RowProvenance::for_insert(row_id.to_string(), 2_000),
        HashMap::new(),
        crate::row_histories::RowState::StagingPending,
        None,
    );

    for (batch_id, row) in [(parent_batch, &parent_row), (child_batch, &child_row)] {
        deliver(
            &mut core,
            Source::Client(client_id),
            SyncPayload::SealBatch {
                submission: seal_declaring(
                    batch_id,
                    crate::batch_fate::BatchMode::Transactional,
                    std::slice::from_ref(row),
                ),
            },
        );
    }
    core.immediate_tick();

    // The child first: its parent is unknown, so it is parked, not applied.
    deliver(
        &mut core,
        Source::Server(upstream),
        SyncPayload::RowBatchNeeded {
            metadata: Some(users_row_metadata(row_id)),
            row: child_row,
        },
    );
    core.immediate_tick();
    assert_eq!(
        core.storage()
            .load_authoritative_batch_fate(child_batch)
            .unwrap(),
        None,
        "a parked row applies nothing, so its seal is not completable yet"
    );

    // The parent lands, applies, and drains the child behind it.
    deliver(
        &mut core,
        Source::Server(upstream),
        SyncPayload::RowBatchNeeded {
            metadata: Some(users_row_metadata(row_id)),
            row: parent_row,
        },
    );
    core.immediate_tick();

    for batch_id in [parent_batch, child_batch] {
        assert_eq!(
            core.storage()
                .load_authoritative_batch_fate(batch_id)
                .unwrap(),
            Some(crate::batch_fate::BatchFate::AcceptedTransaction {
                batch_id,
                confirmed_tier: DurabilityTier::Local,
            }),
            "the parent applied on arrival and the child when drained behind it; both seals \
             are completable and the next tick must complete them"
        );
    }
}
