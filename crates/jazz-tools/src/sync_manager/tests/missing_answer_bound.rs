//! Telling a client a batch is `Missing` must not be something the client can buy forever.
//!
//! `Missing` is an instruction, not advice: on the peer it drives
//! `retransmit_local_batch_to_servers`, and `force_row_batch_to_servers`
//! deliberately clears the dedup bookkeeping so nothing suppresses the resend.
//! When the batch cannot be completed, the answer buys the work that produces
//! the next question, and the cycle's rate is whatever this side can answer at.
//! Production 2026-08-10: one core pinned, sync dead behind it.
//!
//! One cause of an uncompletable batch is fixed at the source (delivery strips
//! `parents` while the digest covered them, see `delivered_row_reseal.rs`), but
//! the shape has other causes — a member this authority never indexed, a branch
//! mismatch, drift in a payload or its authorship. So these gates are about the
//! ANSWER: how fast it may repeat, how many times, and what stops it.
//!
//! Both emitters are covered, because bounding one bounds nothing: the seal that
//! cannot complete, and the fate request synthesised for a batch with no stored
//! fate — which the replay short-circuit queues for every replayed row, so rows
//! alone regenerate the answer without the seal ever coming back.
//!
//! Stopping must retract nothing: the submission stays, no terminal fate is
//! invented, and a reconnect re-arms the answer. The peer keeps its row; it
//! simply stops being told to send it again.

use super::*;

/// Comfortably past the cap, and the shape of a real burst.
const ATTEMPTS: usize = 40;

fn missing_answers(sm: &mut SyncManager, client_id: ClientId) -> usize {
    sm.take_outbox()
        .into_iter()
        .filter(|entry| {
            matches!(
                entry,
                OutboxEntry {
                    destination: Destination::Client(id),
                    payload: SyncPayload::BatchFate {
                        fate: BatchFate::Missing { .. }
                    },
                } if *id == client_id
            )
        })
        .count()
}

/// Rewind the budget beyond the rate limit, as only a test can, so a burst can
/// be driven without the interval standing in for the bound under test.
fn allow_another_answer_now(sm: &mut SyncManager, client_id: ClientId, batch_id: BatchId) {
    if let Some(budget) = sm
        .missing_answers
        .get_mut(&client_id)
        .and_then(|tracked| tracked.budgets.get_mut(&batch_id))
    {
        budget.last_answered_at = budget
            .last_answered_at
            .saturating_sub(crate::sync_manager::MISSING_ANSWER_MIN_INTERVAL_MICROS + 1);
    }
}

fn age_past_the_grace(sm: &mut SyncManager, client_id: ClientId, batch_id: BatchId) {
    let budget = sm
        .missing_answers
        .get_mut(&client_id)
        .and_then(|tracked| tracked.budgets.get_mut(&batch_id))
        .expect("an answered batch must be tracked, or there is nothing to bound");
    budget.first_answered_at = budget
        .first_answered_at
        .saturating_sub(crate::sync_manager::MISSING_ANSWER_GIVE_UP_AFTER_MICROS + 1);
}

// ============================================================================
// The seal emitter
// ============================================================================

struct UnmatchableSeal {
    io: MemoryStorage,
    sm: SyncManager,
    client_id: ClientId,
    batch_id: BatchId,
    row_id: ObjectId,
}

impl UnmatchableSeal {
    /// An authority and a peer whose seal names a row that was never sent, so no
    /// stored row can ever carry the declared digest.
    fn new() -> Self {
        let mut io = MemoryStorage::new();
        seed_users_schema(&mut io);
        let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::EdgeServer);
        let client_id = ClientId::new();
        add_client(&mut sm, &io, client_id);
        sm.set_client_role(client_id, ClientRole::Peer);
        Self {
            io,
            sm,
            client_id,
            batch_id: BatchId(*ObjectId::new().uuid().as_bytes()),
            row_id: ObjectId::new(),
        }
    }

    fn declaration(&self, payload: &[u8]) -> SealedBatchMember {
        SealedBatchMember {
            object_id: self.row_id,
            row_digest: visible_row(self.row_id, "main", Vec::new(), 2_000, payload)
                .content_digest(),
        }
    }

    /// Re-send the seal, and count the answers it drew.
    fn reseal(&mut self, declared: &SealedBatchMember) -> usize {
        self.sm.process_from_client(
            &mut self.io,
            self.client_id,
            SyncPayload::SealBatch {
                submission: sealed_submission(
                    self.batch_id,
                    "main",
                    vec![declared.clone()],
                    Vec::new(),
                ),
            },
        );
        missing_answers(&mut self.sm, self.client_id)
    }

    /// Re-send it as fast as the peer would, with the rate limit stood down so
    /// the give-up policy is what the test is measuring.
    fn reseal_unthrottled(&mut self, declared: &SealedBatchMember) -> usize {
        allow_another_answer_now(&mut self.sm, self.client_id, self.batch_id);
        self.reseal(declared)
    }
}

#[test]
fn a_burst_of_reseals_is_answered_at_the_rate_limit_not_at_its_own_rate() {
    let mut world = UnmatchableSeal::new();
    let declared = world.declaration(b"never sent");

    let answers: usize = (0..ATTEMPTS).map(|_| world.reseal(&declared)).sum();
    assert_eq!(
        answers, 1,
        "{ATTEMPTS} reseals inside one rate-limit window drew {answers} answers; a peer \
         retrying in a tight loop must not be able to set the rate at which this authority \
         hands out retransmission instructions"
    );
}

#[test]
fn an_unanswerable_seal_stops_being_answered() {
    let mut world = UnmatchableSeal::new();
    let declared = world.declaration(b"never sent");

    // Within the grace the answer must keep coming: a peer whose rows are merely
    // slow needs to be told to resend them, and a bare attempt cap would cut it
    // off in the milliseconds a reconnect storm takes.
    let within_grace: usize = (0..ATTEMPTS)
        .map(|_| world.reseal_unthrottled(&declared))
        .sum();
    assert_eq!(
        within_grace, ATTEMPTS,
        "the scenario must produce the answer at all, else it gates nothing — and a burst \
         must not spend the grace"
    );

    age_past_the_grace(&mut world.sm, world.client_id, world.batch_id);
    let past_grace: usize = (0..ATTEMPTS)
        .map(|_| world.reseal_unthrottled(&declared))
        .sum();
    assert_eq!(
        past_grace, 0,
        "past the cap and the grace, {ATTEMPTS} reseals were still answered {past_grace} \
         times; every answer asks the peer to send it all again, so an unbounded answer is \
         an unbounded loop"
    );

    // Giving up must cost the peer nothing: the submission stays, and no
    // terminal fate is invented for a batch we simply could not complete.
    assert!(
        world
            .io
            .load_sealed_batch_submission(world.batch_id)
            .expect("submission lookup")
            .is_some(),
        "the submission must survive: giving up on answering is not giving up on the batch"
    );
    assert!(
        world
            .io
            .load_authoritative_batch_fate(world.batch_id)
            .expect("fate lookup")
            .is_none(),
        "no terminal fate may be invented — on the peer a Rejected retracts the row, and \
         the graft tool is the only way back"
    );
}

#[test]
fn alternating_declarations_do_not_buy_a_fresh_budget() {
    let mut world = UnmatchableSeal::new();
    let first = world.declaration(b"never sent");
    let second = world.declaration(b"also never sent");
    assert_ne!(
        first.row_digest, second.row_digest,
        "the two declarations must differ, or this gates nothing"
    );

    for _ in 0..ATTEMPTS {
        world.reseal_unthrottled(&first);
    }
    age_past_the_grace(&mut world.sm, world.client_id, world.batch_id);
    assert_eq!(
        world.reseal_unthrottled(&first),
        0,
        "the bound must have taken hold"
    );

    // Remembering the declaration so that a *different* one re-arms the budget
    // reads as fairness and is in fact the hole: a peer alternating two of them,
    // or perturbing one member per round, would be answered forever. A
    // declaration that can be matched never reaches this path at all.
    let alternating: usize = (0..ATTEMPTS)
        .map(|attempt| {
            let declared = if attempt % 2 == 0 { &second } else { &first };
            world.reseal_unthrottled(&declared.clone())
        })
        .sum();
    assert_eq!(
        alternating, 0,
        "alternating declarations drew {alternating} answers past the bound; varying the \
         claim must not reset what this authority has already spent on the batch"
    );
}

#[test]
fn a_reconnect_asks_about_a_batch_this_authority_gave_up_on() {
    let mut world = UnmatchableSeal::new();
    let declared = world.declaration(b"never sent");
    for _ in 0..ATTEMPTS {
        world.reseal_unthrottled(&declared);
    }
    age_past_the_grace(&mut world.sm, world.client_id, world.batch_id);
    assert_eq!(
        world.reseal_unthrottled(&declared),
        0,
        "the bound must have taken hold"
    );

    // A reconnect does not always mint a new client — `ensure_client_with_session`
    // updates it in place, and a reconnecting client is pulled back out of the
    // disconnect candidates rather than reaped — and the session does not mark it
    // either, because the same user presents the same session value. Only the
    // handshake knows, and it is the handshake that says so:
    // `ensure_client_with_role_and_catalogue_state_hash` calls this on every
    // connection.
    world.sm.set_client_session(
        world.client_id,
        crate::query_manager::session::Session::new("same user, new socket"),
    );
    assert_eq!(
        world.reseal_unthrottled(&declared),
        0,
        "a session update is not a connection: re-arming on it would make the bound depend \
         on a value that does not change when the socket does"
    );
    world.sm.note_client_connected(world.client_id);

    assert_eq!(
        world.reseal_unthrottled(&declared),
        1,
        "a fresh connection must ask again about the very same declaration: silence defers \
         a batch, it must not abandon it"
    );
}

// ============================================================================
// The fate-request emitter, which rows reach without any seal
// ============================================================================

/// A row this authority already stores, replayed by the client that sent it.
///
/// This is the production shape the seal gates do not reach: `apply_row_updated`
/// short-circuits the replay and queues a fate request for its batch, and
/// `process_inbox` drains that into the synthesising emitter. So every
/// retransmitted row regenerates `Missing` on its own — which is exactly what a
/// `Missing` told the peer to send.
#[test]
fn replayed_rows_alone_stop_drawing_missing() {
    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::Local);
    let mut io = MemoryStorage::new();
    let client_id = ClientId::new();
    let row_id = ObjectId::new();

    let row = row_with_state(
        visible_row(row_id, "main", Vec::new(), 1_000, b"alice"),
        crate::row_histories::RowState::VisibleDirect,
        Some(DurabilityTier::Local),
    );
    let batch_id = row.batch_id;
    // Seeded WITHOUT a settlement: a batch with a stored fate gets that fate as its
    // answer, so the synthesised `Missing` — the amplifier — only exists for a batch
    // whose fate this authority never recorded, which is the loop's own shape.
    seed_visible_row(&mut sm, &mut io, "users", row.clone());
    // The replayer wrote the batch: the binding its first send made.
    io.put_batch_writer(
        batch_id,
        &crate::batch_fate::BatchWriter::User("alice".to_string()),
    )
    .unwrap();

    add_client(&mut sm, &io, client_id);
    sm.set_client_role(client_id, ClientRole::User);
    sm.set_client_session(
        client_id,
        crate::query_manager::session::Session::new("alice"),
    );
    sm.take_outbox();

    let mut replay = |sm: &mut SyncManager, io: &mut MemoryStorage| {
        sm.push_inbox(InboxEntry {
            source: Source::Client(client_id),
            payload: SyncPayload::RowBatchCreated {
                metadata: Some(RowMetadata {
                    id: row_id,
                    metadata: row_metadata("users"),
                }),
                row: row.clone(),
            },
        });
        sm.process_inbox(io);
        missing_answers(sm, client_id)
    };

    let first = replay(&mut sm, &mut io);
    assert_eq!(
        first, 1,
        "a replayed row must reach the synthesising emitter at all, else this gates nothing"
    );

    let burst: usize = (0..ATTEMPTS).map(|_| replay(&mut sm, &mut io)).sum();
    assert_eq!(
        burst, 0,
        "{ATTEMPTS} replayed rows drew {burst} further answers inside one rate-limit \
         window; the rows a Missing asks for must not each buy another Missing"
    );

    for _ in 0..ATTEMPTS {
        allow_another_answer_now(&mut sm, client_id, batch_id);
        replay(&mut sm, &mut io);
    }
    age_past_the_grace(&mut sm, client_id, batch_id);
    let past_grace: usize = (0..ATTEMPTS)
        .map(|_| {
            allow_another_answer_now(&mut sm, client_id, batch_id);
            replay(&mut sm, &mut io)
        })
        .sum();
    assert_eq!(
        past_grace, 0,
        "past the cap and the grace, replayed rows were still answered {past_grace} times"
    );
}

/// A client holding the maximum number of tracked batches must still be able to
/// have a real one repaired.
///
/// The budget needs one entry per batch to count against, so the tracking is
/// itself something a peer can grow. Declining to track anything new past the
/// cap would let one flooding stream permanently deny that client the answer
/// that exists to repair a genuinely interrupted upload.
#[test]
fn a_client_at_the_tracking_cap_can_still_be_told_about_a_new_batch() {
    let mut world = UnmatchableSeal::new();
    let declared = world.declaration(b"never sent");

    let tracked = world.sm.missing_answers.entry(world.client_id).or_default();
    for _ in 0..crate::sync_manager::MAX_TRACKED_MISSING_ANSWERS {
        let silenced = BatchId(*ObjectId::new().uuid().as_bytes());
        tracked.order.push_back(silenced);
        tracked.budgets.insert(
            silenced,
            crate::sync_manager::MissingAnswerBudget {
                answers: crate::sync_manager::MAX_MISSING_ANSWERS,
                first_answered_at: 1,
                last_answered_at: 1,
                silenced: true,
            },
        );
    }

    assert_eq!(
        world.reseal(&declared),
        1,
        "a client at the tracking cap was refused an answer about a batch this authority \
         had never seen; one flooding stream must not deny that client its own recovery"
    );
    let tracked = world
        .sm
        .missing_answers
        .get(&world.client_id)
        .expect("the client must still be tracked");
    assert!(
        tracked.budgets.len() <= crate::sync_manager::MAX_TRACKED_MISSING_ANSWERS,
        "tracking grew past its cap to {}; making room must mean evicting, not appending",
        tracked.budgets.len()
    );
}

// ============================================================================
// Fresh batch ids, which is where a per-batch budget can be walked around
// ============================================================================

/// The first answer for a batch is free, so that a genuinely interrupted upload
/// is repaired without waiting out an interval. That makes a fresh batch id the
/// cheapest thing a peer can buy an answer with, and a peer can mint them
/// endlessly. Only the tracking cap stands in the way — so making room must cost
/// the peer something, and here nothing has gone quiet to evict.
#[test]
fn cycling_fresh_batch_ids_stops_buying_answers() {
    let mut world = UnmatchableSeal::new();
    let cap = crate::sync_manager::MAX_TRACKED_MISSING_ANSWERS;

    let answers: usize = (0..cap * 2)
        .map(|_| {
            world.sm.process_from_client(
                &mut world.io,
                world.client_id,
                SyncPayload::BatchFateNeeded {
                    batch_ids: vec![BatchId(*ObjectId::new().uuid().as_bytes())],
                },
            );
            missing_answers(&mut world.sm, world.client_id)
        })
        .sum();

    assert_eq!(
        answers,
        cap,
        "{} fresh batch ids drew {answers} answers; a budget kept per batch is not a bound \
         at all if a new id always comes with a fresh one",
        cap * 2
    );
}

/// One frame may name any number of batches, and each named batch costs a
/// storage read and a queued answer.
#[test]
fn one_oversized_fate_request_is_answered_only_up_to_the_cap() {
    let mut world = UnmatchableSeal::new();
    let cap = crate::sync_manager::MAX_TRACKED_MISSING_ANSWERS;

    // Reversed, because ids are minted in ascending order: sent in that order the peer's
    // list is already canonical, and a gate over it would pass whatever the code did.
    let mut asked: Vec<BatchId> = (0..cap * 4)
        .map(|_| BatchId(*ObjectId::new().uuid().as_bytes()))
        .collect();
    // Reversed after minting, because ids are minted in ascending order: sent that way the
    // peer's list is already canonical and a gate over its ordering would pass whatever the
    // code did. (Reversing the range instead would not help — the ids are minted in call
    // order either way.)
    asked.reverse();
    world.io.reset_authoritative_fate_lookups();
    world.sm.process_from_client(
        &mut world.io,
        world.client_id,
        SyncPayload::BatchFateNeeded {
            batch_ids: asked.clone(),
        },
    );

    // Counting answers alone would pass without any truncation, because the
    // per-batch budget declines the tail once the tracking is full — the reply
    // would be bounded while the work behind it was not. The reads are the work.
    let reads = world.io.authoritative_fate_lookups();
    assert!(
        reads <= cap,
        "one frame naming {} batches cost {reads} fate lookups; the length of a peer's list \
         must not be the size of this authority's work",
        cap * 4
    );
    let answers = missing_answers(&mut world.sm, world.client_id);
    assert_eq!(
        answers,
        cap,
        "one frame naming {} batches drew {answers} answers",
        cap * 4
    );
    let interest = world.sm.batch_fate_interest.len();
    assert!(
        interest <= cap,
        "one frame naming {} batches left {interest} interest entries, held until the \
         client goes away",
        cap * 4
    );

    // The two caps must land on the same batches. Registering the peer's raw order while
    // answering a sorted one would answer about batches this client is not recorded as
    // interested in, and that interest gates every later fate broadcast to it.
    let mut answered = asked.clone();
    answered.sort();
    answered.dedup();
    answered.truncate(cap);
    let registered_but_unanswered = world
        .sm
        .batch_fate_interest
        .keys()
        .filter(|batch_id| !answered.contains(batch_id))
        .count();
    assert_eq!(
        registered_but_unanswered, 0,
        "{registered_but_unanswered} batches were registered as interesting to this client \
         but are not the ones it was answered about"
    );
}

/// A client that filled the tracking once and then went quiet must not be locked
/// out of its own repairs.
///
/// Silencing a batch takes sustained interest — the cap AND the grace. A peer
/// that asks once about a great many ids leaves budgets nobody will ever silence,
/// so "evict only what we gave up on" would decline every genuinely new batch
/// until the connection dropped. A batch nobody has asked about in a give-up
/// window is not mid-repair either.
#[test]
fn making_room_may_take_a_batch_nobody_has_asked_about_in_a_long_time() {
    let mut world = UnmatchableSeal::new();
    let cap = crate::sync_manager::MAX_TRACKED_MISSING_ANSWERS;

    world.sm.process_from_client(
        &mut world.io,
        world.client_id,
        SyncPayload::BatchFateNeeded {
            batch_ids: (0..cap)
                .map(|_| BatchId(*ObjectId::new().uuid().as_bytes()))
                .collect(),
        },
    );
    world.sm.take_outbox();

    let tracked = world
        .sm
        .missing_answers
        .get_mut(&world.client_id)
        .expect("the client must be tracked");
    assert_eq!(
        tracked.budgets.len(),
        cap,
        "the tracking must be full, or this gates nothing"
    );
    assert!(
        tracked.budgets.values().all(|budget| !budget.silenced),
        "none of these may be silenced: one answer each is not sustained interest"
    );
    // Nobody asked again, and a give-up window went by.
    for budget in tracked.budgets.values_mut() {
        budget.last_answered_at = budget
            .last_answered_at
            .saturating_sub(crate::sync_manager::MISSING_ANSWER_GIVE_UP_AFTER_MICROS + 1);
    }

    assert_eq!(
        world.reseal(&world.declaration(b"a real interrupted upload")),
        1,
        "a client that once named a lot of batches was refused an answer about a new one; \
         filling the tracking must not lock a client out of its own repairs"
    );
}

/// Making room must not take a budget that is still doing its job.
///
/// Evicting a live entry hands the peer back the free first answer for a batch
/// it is already being throttled on — the same hole as re-arming, moved into the
/// id space — and it drops the bookkeeping for a batch that may be genuinely
/// mid-repair.
#[test]
fn making_room_never_evicts_a_budget_that_is_still_live() {
    let mut world = UnmatchableSeal::new();
    let cap = crate::sync_manager::MAX_TRACKED_MISSING_ANSWERS;

    let live: Vec<BatchId> = (0..cap)
        .map(|_| BatchId(*ObjectId::new().uuid().as_bytes()))
        .collect();
    let tracked = world.sm.missing_answers.entry(world.client_id).or_default();
    for batch_id in &live {
        tracked.budgets.insert(
            *batch_id,
            crate::sync_manager::MissingAnswerBudget {
                answers: 1,
                first_answered_at: 1,
                last_answered_at: 1,
                silenced: false,
            },
        );
    }

    world.sm.process_from_client(
        &mut world.io,
        world.client_id,
        SyncPayload::BatchFateNeeded {
            batch_ids: vec![BatchId(*ObjectId::new().uuid().as_bytes())],
        },
    );
    assert_eq!(
        missing_answers(&mut world.sm, world.client_id),
        0,
        "with nothing given up on, there is nothing to evict, and the answer must wait \
         rather than be paid for out of another batch's budget"
    );

    let tracked = world
        .sm
        .missing_answers
        .get(&world.client_id)
        .expect("the client must still be tracked");
    let surviving = live
        .iter()
        .filter(|batch_id| tracked.budgets.contains_key(*batch_id))
        .count();
    assert_eq!(
        surviving,
        live.len(),
        "{} live budgets were dropped to make room",
        live.len() - surviving
    );
}
