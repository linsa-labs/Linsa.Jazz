//! Randomized differential test: batch fate handling and the reconnect offer
//! set, against a model of their intended semantics.
//!
//! `BatchFate::Missing` is an instruction — "I do not have this batch, send it
//! again" — not a fact about a batch. Both writers used to store it like any
//! other fate, and that is what turned the request into the reason it could
//! never be answered: `merged_with` ends in `_ => incoming.clone()`, so a
//! stored `Missing` overwrote the unsettled `DurableDirect{Local}` underneath
//! it, and `Missing` needs no settlement — so the batch left
//! `pending_batch_ids_needing_reconciliation` for good. For a write committed
//! while no server was attached, that fate was the batch's ONLY ticket into the
//! offer set: `retire_settled_batch` had already dropped its submission and its
//! record and kept nothing but the row index. Measured 2026-08-20: 397
//! unanswered asks in thirteen minutes, presence dead through a socket
//! reconnect and an app restart (commit 58de59884).
//!
//! That is state-and-ordering behaviour across three layers — the stored fate,
//! the retained bookkeeping, and the in-memory caches a restart drops — so the
//! hand-written gates only encode the sequences we already thought of. This
//! compares against a model on randomized streams.
//!
//! The model, in full:
//!
//! * `fate` — what storage should hold. Every non-`Missing` fate merges in
//!   (`BatchFate::merged_with`, applied again by the storage upsert itself);
//!   `Missing` NEVER writes, and is always reported as changed so the caller
//!   acts on it.
//! * `sealed` / `rows_present` — whether this node holds a committed batch's
//!   rows. Row history is never deleted here, so `rows_present` stays true past
//!   every retirement; only the bookkeeping around it goes.
//! * `has_submission` / `has_record` / `has_row_index` — the retained
//!   bookkeeping. `retire_settled_batch` drops the record at any tier and the
//!   row index only from `GlobalServer` up, which is exactly the state the
//!   production store was in. The submission follows
//!   `SyncManager::batch_fate_lets_seal_go`: a direct batch's goes once its
//!   tier reaches the settlement target, a transaction's only from
//!   `GlobalServer` up, like the row index.
//! * `connected` — one upstream server or none. It is not decoration: it moves
//!   `settlement_target()` between `GlobalServer` and `Local`, and a
//!   `DurableDirect{Local}` is terminal at one and unsettled at the other.
//!
//! The invariant, over that state:
//!
//! > A batch whose rows this node holds, whose fate is not terminal at the
//! > node's settlement target and is not `Rejected`, MUST be offered on every
//! > reconnect, and MUST be retransmitted on every inbound `Missing`.
//!
//! Both halves are asserted after every op, along with the stored fate itself,
//! so a stream can never drift into agreeing about offers while disagreeing
//! about what is on disk.
//!
//! No proptest/quickcheck in this crate, so the stream comes from the same
//! hand-rolled xorshift over a fixed seed list the storage and catalogue
//! differentials use; every assertion carries the seed and op index so a
//! failure replays exactly.
//!
//! Backends: the randomized stream runs on `MemoryStorage`, and
//! `batch_fate_offer_differential_sqlite_round_trip` reruns a subset on
//! `SqliteStorage`. That second arm is not ceremony. `MemoryStorage` overrides
//! `upsert_/load_authoritative_batch_fate` (`storage/memory.rs:517,549`) and
//! answers from a live `HashMap` without re-decoding, so on it "the stored fate
//! matches the model" never touches `encode_storage_row`/`decode_storage_row`.
//! Only the sqlite arm proves the fate survived as bytes — which is the whole
//! difference between a fate that outlives a restart and one that does not.
//!
//! Generator limitations (conscious):
//! - One node under test, with at most one upstream server, and fates injected
//!   by hand rather than produced by a real peer. What a server would have
//!   answered is a different question, gated elsewhere (`settlements.rs`).
//! - The node registers no durability tier of its own, so it is a client:
//!   `settlement_target()` is `GlobalServer` connected and `Local` offline, and
//!   `recover_completed_sealed_batches_with_storage` is inert on it.
//! - Rows are never deleted, so `rows_present` is monotone. The defect is about
//!   bookkeeping vanishing out from under rows that are still there.
//! - Open (begun, not yet sealed) transactional batches are exempted from the
//!   "offered ⊆ model" direction only. `pending_batch_ids_needing_reconciliation`
//!   reads `local_batch_record_cache`, which holds a record from the moment a
//!   batch is BEGUN — so an unsealed batch is in the offer set while the process
//!   lives and gone after a restart, because the derivation never consults the
//!   persisted record. Whether an uncommitted batch should be offered at all is
//!   a separate question from this defect, and answering it either way in the
//!   model would turn a design opinion into a failing assertion. The
//!   "model ⊆ offered" direction — the one the outage lived in — still covers
//!   every sealed batch.

use std::collections::{BTreeMap, BTreeSet};

use crate::batch_fate::{BatchFate, BatchMode};
use crate::query_manager::session::WriteContext;
use crate::row_histories::BatchId;
use crate::schema_manager::AppId;
use crate::storage::{MemoryStorage, SqliteStorage, Storage};
use crate::sync_manager::{
    DurabilityTier, InboxEntry, OutboxEntry, ServerId, Source, SyncManager, SyncPayload,
};

use super::*;

const OPS_PER_SEED: usize = 300;
const SEEDS: [u64; 24] = [
    0xFA7E_0F5E_0000_0001,
    0xFA7E_0F5E_0000_0002,
    0xFA7E_0F5E_0000_0003,
    0xFA7E_0F5E_0000_0004,
    0xFA7E_0F5E_0000_0005,
    0xFA7E_0F5E_0000_0006,
    0xFA7E_0F5E_0000_0007,
    0xFA7E_0F5E_0000_0008,
    0xFA7E_0F5E_0000_0009,
    0xFA7E_0F5E_0000_000A,
    0xFA7E_0F5E_0000_000B,
    0xFA7E_0F5E_0000_000C,
    0xD1FF_0F5E_0000_000D,
    0xD1FF_0F5E_0000_000E,
    0xD1FF_0F5E_0000_000F,
    0xD1FF_0F5E_0000_0010,
    0xD1FF_0F5E_0000_0011,
    0xD1FF_0F5E_0000_0012,
    0xD1FF_0F5E_0000_0013,
    0xD1FF_0F5E_0000_0014,
    0xD1FF_0F5E_0000_0015,
    0xD1FF_0F5E_0000_0016,
    0xD1FF_0F5E_0000_0017,
    0xD1FF_0F5E_0000_0018,
];

struct Xorshift(u64);

impl Xorshift {
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

/// Which batch the next op acts on. A uniform choice over every batch ever
/// created produces a stream of one-op-per-batch histories, and every shape the
/// fix is about is a MULTI-op history on one batch — `Missing` after a global
/// confirmation, `Missing` twice with nothing in between, a restart wedged
/// between two fates. So three times out of four the pick comes from a recent
/// window, which is also how a real client behaves: the batches a server is
/// still asking about are the recent ones. Measured effect on pair coverage is
/// in the op mix this test prints.
fn pick_batch(rng: &mut Xorshift, order: &[BatchId]) -> BatchId {
    const RECENT_WINDOW: usize = 6;
    if rng.below(4) < 3 {
        let window = RECENT_WINDOW.min(order.len());
        order[order.len() - window + rng.below(window)]
    } else {
        order[rng.below(order.len())]
    }
}

/// What the model says this node knows about one batch.
#[derive(Debug, Clone)]
struct ModelBatch {
    /// The fate storage must hold, or `None` for "no fate on record".
    fate: Option<BatchFate>,
    /// Committed (direct) or sealed (transactional). Only a sealed batch is
    /// something this node owes a peer.
    sealed: bool,
    /// This node holds the batch's rows. Monotone here: nothing in the op
    /// alphabet deletes row history.
    rows_present: bool,
    /// Retained sealed submission. Asserted against storage.
    has_submission: bool,
    /// Written in a transaction. A transaction's seal outlives a settlement
    /// below `GlobalServer`; a direct batch's does not.
    transactional: bool,
    /// The runtime can still find the batch's local record — cache OR storage.
    /// Deliberately a union rather than two fields: the only thing it gates is
    /// whether `commit_batch` finds a record to seal, and that lookup consults
    /// both. It is not asserted against storage for the same reason, because on
    /// an offline direct commit the record lives ONLY in the cache
    /// (`retire_settled_batch` deletes the persisted one and `commit_batch`
    /// re-inserts the cached one immediately after).
    has_record: bool,
    /// Retained batch→rows index. Asserted against storage: this is what
    /// survives a local-tier retirement, and it is how a batch whose
    /// bookkeeping is gone is still findable at all.
    has_row_index: bool,
}

impl ModelBatch {
    /// The invariant's antecedent: rows here, fate not terminal at `target`,
    /// not `Rejected`. `SyncManager::fate_needs_settlement_at` folds the last
    /// two together — `Rejected` counts as settled — except for `Missing`,
    /// which it excludes and which the model never stores.
    fn offerable(&self, target: DurabilityTier) -> bool {
        self.sealed
            && self.rows_present
            && SyncManager::fate_needs_settlement_at(self.fate.as_ref(), target)
    }
}

#[derive(Debug, Default)]
struct OpCounts {
    commit_direct_online: usize,
    commit_direct_offline: usize,
    commit_transactional: usize,
    seal: usize,
    fate_durable_direct: usize,
    fate_accepted_transaction: usize,
    fate_rejected: usize,
    fate_missing: usize,
    fate_missing_while_connected: usize,
    connect: usize,
    disconnect: usize,
    restart: usize,
    tick: usize,
    retire: usize,
    /// Pair coverage, counted where the code was wrong.
    missing_right_after_global_durable: usize,
    repeated_missing_no_change_between: usize,
    missing_after_rejected: usize,
    offline_commit_connect_missing_restart_connect: usize,
    restart_between_fate_deliveries: usize,
    /// How many batch-offers the wire assertion actually demanded. Without this
    /// a green run cannot be told apart from one where `must_offer` was always
    /// empty and the second half of the invariant was never checked.
    wire_offers_demanded: usize,
}

impl OpCounts {
    fn merge(&mut self, other: &OpCounts) {
        self.commit_direct_online += other.commit_direct_online;
        self.commit_direct_offline += other.commit_direct_offline;
        self.commit_transactional += other.commit_transactional;
        self.seal += other.seal;
        self.fate_durable_direct += other.fate_durable_direct;
        self.fate_accepted_transaction += other.fate_accepted_transaction;
        self.fate_rejected += other.fate_rejected;
        self.fate_missing += other.fate_missing;
        self.fate_missing_while_connected += other.fate_missing_while_connected;
        self.connect += other.connect;
        self.disconnect += other.disconnect;
        self.restart += other.restart;
        self.tick += other.tick;
        self.retire += other.retire;
        self.missing_right_after_global_durable += other.missing_right_after_global_durable;
        self.repeated_missing_no_change_between += other.repeated_missing_no_change_between;
        self.missing_after_rejected += other.missing_after_rejected;
        self.offline_commit_connect_missing_restart_connect +=
            other.offline_commit_connect_missing_restart_connect;
        self.restart_between_fate_deliveries += other.restart_between_fate_deliveries;
        self.wire_offers_demanded += other.wire_offers_demanded;
    }
}

/// One step of the per-batch history the pair counters watch. Only the shapes
/// the fix is about; everything else is `Other`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mark {
    OfflineCommit,
    Connect,
    Missing,
    Rejected,
    GlobalDurable,
    Restart,
    Other,
}

/// The node under test, its model, and the plumbing that keeps them in step.
struct Harness<S: Storage> {
    core: Option<RuntimeCore<S, NoopScheduler>>,
    app_name: &'static str,
    server: Option<ServerId>,
    model: BTreeMap<BatchId, ModelBatch>,
    order: Vec<BatchId>,
    open_batches: Vec<BatchId>,
    /// Per-batch history of the marks above, for pair coverage.
    marks: BTreeMap<BatchId, Vec<Mark>>,
    counts: OpCounts,
    seed: u64,
}

impl<S: Storage> Harness<S> {
    fn new(app_name: &'static str, storage: S, seed: u64) -> Self {
        let app_id = AppId::from_name(app_name);
        let schema_manager =
            SchemaManager::new(SyncManager::new(), test_schema(), app_id, "dev", "main").unwrap();
        let mut core = new_test_core(schema_manager, storage, NoopScheduler);
        core.immediate_tick();
        core.batched_tick();
        core.sync_sender().take();
        Self {
            core: Some(core),
            app_name,
            server: None,
            model: BTreeMap::new(),
            order: Vec::new(),
            open_batches: Vec::new(),
            marks: BTreeMap::new(),
            counts: OpCounts::default(),
            seed,
        }
    }

    fn core(&mut self) -> &mut RuntimeCore<S, NoopScheduler> {
        self.core
            .as_mut()
            .expect("the runtime is only ever taken and immediately replaced")
    }

    fn connected(&self) -> bool {
        self.server.is_some()
    }

    /// The node under test registers no tier of its own, so this is exactly
    /// `SyncManager::settlement_target()`: `GlobalServer` with an upstream,
    /// `Local` without one.
    fn target(&self) -> DurabilityTier {
        if self.connected() {
            DurabilityTier::GlobalServer
        } else {
            DurabilityTier::Local
        }
    }

    fn mark(&mut self, batch_id: BatchId, mark: Mark) {
        self.marks.entry(batch_id).or_default().push(mark);
    }

    fn mark_all(&mut self, mark: Mark) {
        for batch_id in &self.order {
            self.marks.entry(*batch_id).or_default().push(mark);
        }
    }

    /// The model's answer to "what must a reconnect offer right now".
    fn model_offer_set(&self) -> BTreeSet<BatchId> {
        let target = self.target();
        self.model
            .iter()
            .filter(|(_, batch)| batch.offerable(target))
            .map(|(batch_id, _)| *batch_id)
            .collect()
    }

    // ---------------------------------------------------------------- model

    /// Apply an arriving fate to the model exactly as the two writers do:
    /// `Missing` never writes and always counts as changed; anything else
    /// merges — first in `persist_authoritative_batch_fate`, then again inside
    /// the storage upsert itself, which is idempotent over the merged value.
    /// The bool is "the runtime acts on this", which is what gates the retire.
    fn model_apply_fate(&mut self, batch_id: BatchId, fate: &BatchFate) -> bool {
        if matches!(fate, BatchFate::Missing { .. }) {
            return true;
        }
        let Some(batch) = self.model.get_mut(&batch_id) else {
            return false;
        };
        let merged = match batch.fate.as_ref() {
            Some(existing) => existing.merged_with(fate),
            None => fate.clone(),
        };
        let changed = batch.fate.as_ref() != Some(&merged);
        batch.fate = Some(merged.clone());
        if !changed {
            return false;
        }
        // `apply_received_batch_fate` retires only on a confirmed tier that
        // reaches the settlement target — `Rejected` keeps its record for the
        // mutation-error replay and `Missing` carries no tier at all.
        if let Some(acked) = merged.confirmed_tier()
            && acked >= self.target()
        {
            self.model_retire(batch_id, acked);
        }
        true
    }

    /// `retire_settled_batch`: the record goes at any tier, the row index only
    /// from `GlobalServer` up, and the submission as `batch_fate_lets_seal_go`
    /// says. Row history is untouched — which is the whole reason a stranded
    /// batch is recoverable at all.
    fn model_retire(&mut self, batch_id: BatchId, confirmed_tier: DurabilityTier) {
        let target = self.target();
        if let Some(batch) = self.model.get_mut(&batch_id) {
            let seal_goes = if batch.transactional {
                confirmed_tier >= DurabilityTier::GlobalServer
            } else {
                confirmed_tier >= target
            };
            if seal_goes {
                batch.has_submission = false;
            }
            batch.has_record = false;
            if confirmed_tier >= DurabilityTier::GlobalServer {
                batch.has_row_index = false;
            }
        }
    }

    // ------------------------------------------------------------------ ops

    fn commit_direct(&mut self) -> BatchId {
        let connected = self.connected();
        let ((_, _), batch_id) = self
            .core()
            .insert("users", user_insert_values(ObjectId::new(), "Alice"), None)
            .expect("a direct insert should commit");
        let settled_at_commit = !connected;
        self.model.insert(
            batch_id,
            ModelBatch {
                fate: Some(BatchFate::DurableDirect {
                    batch_id,
                    confirmed_tier: DurabilityTier::Local,
                }),
                sealed: true,
                rows_present: true,
                // Settled at commit means `retire_settled_batch(Local)` already
                // ran: submission and record gone, row index kept. That is the
                // production shape the defect needed.
                has_submission: !settled_at_commit,
                transactional: false,
                has_record: true,
                has_row_index: true,
            },
        );
        self.order.push(batch_id);
        if connected {
            self.counts.commit_direct_online += 1;
            self.mark(batch_id, Mark::Other);
        } else {
            self.counts.commit_direct_offline += 1;
            self.mark(batch_id, Mark::OfflineCommit);
        }
        batch_id
    }

    fn commit_transactional(&mut self) -> BatchId {
        let batch_id = self.core().begin_batch(BatchMode::Transactional);
        let write_context = WriteContext::default()
            .with_batch_mode(BatchMode::Transactional)
            .with_batch_id(batch_id);
        self.core()
            .insert(
                "users",
                user_insert_values(ObjectId::new(), "Bob"),
                Some(&write_context),
            )
            .expect("a transactional insert should stage");
        self.model.insert(
            batch_id,
            ModelBatch {
                fate: None,
                sealed: false,
                rows_present: true,
                has_submission: false,
                transactional: true,
                has_record: true,
                has_row_index: true,
            },
        );
        self.order.push(batch_id);
        self.open_batches.push(batch_id);
        self.counts.commit_transactional += 1;
        self.mark(batch_id, Mark::Other);
        batch_id
    }

    fn seal(&mut self, batch_id: BatchId) {
        self.core()
            .commit_batch(batch_id)
            .expect("sealing an open transactional batch should succeed");
        if let Some(batch) = self.model.get_mut(&batch_id) {
            // `commit_batch` opens with a record lookup, and a batch whose fate
            // already reached the settlement target has had that record retired
            // out from under it (`retire_settled_batch` clears the cache AND
            // storage). The commit then falls into `complete_empty_batch` and
            // seals nothing: no submission, no seal on the wire. A settled batch
            // has nothing left to seal, so this is a no-op rather than a loss.
            if batch.has_record {
                batch.sealed = true;
                batch.has_submission = true;
            }
        }
        self.open_batches.retain(|open| *open != batch_id);
        self.counts.seal += 1;
        self.mark(batch_id, Mark::Other);
    }

    /// A fate arrives from the upstream. Delivered even while disconnected —
    /// `process_from_server` does not check that the sender is a registered
    /// server, so a fate can land on a node with nothing attached, and the
    /// settlement target it is judged against is `Local` rather than
    /// `GlobalServer` when it does.
    fn deliver_fate(&mut self, fate: BatchFate) {
        let batch_id = fate.batch_id();
        let source = Source::Server(self.server.unwrap_or_else(ServerId::new));
        self.core().park_sync_message(InboxEntry {
            source,
            payload: SyncPayload::BatchFate { fate: fate.clone() },
        });
        self.model_apply_fate(batch_id, &fate);
        match &fate {
            BatchFate::Missing { .. } => {
                self.counts.fate_missing += 1;
                if self.connected() {
                    self.counts.fate_missing_while_connected += 1;
                }
                self.mark(batch_id, Mark::Missing);
            }
            BatchFate::Rejected { .. } => {
                self.counts.fate_rejected += 1;
                self.mark(batch_id, Mark::Rejected);
            }
            BatchFate::DurableDirect { confirmed_tier, .. } => {
                self.counts.fate_durable_direct += 1;
                let mark = if *confirmed_tier == DurabilityTier::GlobalServer {
                    Mark::GlobalDurable
                } else {
                    Mark::Other
                };
                self.mark(batch_id, mark);
            }
            BatchFate::AcceptedTransaction { confirmed_tier, .. } => {
                self.counts.fate_accepted_transaction += 1;
                let mark = if *confirmed_tier == DurabilityTier::GlobalServer {
                    Mark::GlobalDurable
                } else {
                    Mark::Other
                };
                self.mark(batch_id, mark);
            }
        }
    }

    fn connect(&mut self) {
        let server_id = ServerId::new();
        self.core().add_server(server_id);
        self.server = Some(server_id);
        self.counts.connect += 1;
        self.mark_all(Mark::Connect);
    }

    fn disconnect(&mut self) {
        if let Some(server_id) = self.server.take() {
            self.core().remove_server(server_id);
            self.counts.disconnect += 1;
        }
        self.mark_all(Mark::Other);
    }

    /// Drop every in-memory cache — `local_batch_record_cache`,
    /// `known_empty_batch_scans`, the SyncManager's `missing_answers` and its
    /// server set — and rebuild the runtime over the same storage. This is the
    /// half of the defect that outlived a socket reconnect.
    fn restart(&mut self, rebuild: &dyn Fn(S) -> S) {
        let mut core = self.core.take().expect("the runtime is present");
        core.batched_tick();
        core.sync_sender().take();
        let storage = rebuild(core.into_storage());
        let app_id = AppId::from_name(self.app_name);
        let schema_manager =
            SchemaManager::new(SyncManager::new(), test_schema(), app_id, "dev", "main").unwrap();
        let mut core = new_test_core(schema_manager, storage, NoopScheduler);
        core.immediate_tick();
        self.core = Some(core);
        self.server = None;
        // The record CACHE is gone; whatever is on disk stays. An open batch's
        // record survives on disk but no longer reaches the offer derivation —
        // and `ensure_batch_is_open` no longer knows the batch either, so it
        // can never be sealed again. Its staged rows are orphaned, which is a
        // real property of a restart mid-transaction, not a harness artefact.
        self.open_batches.clear();
        self.counts.restart += 1;
        self.mark_all(Mark::Restart);
    }

    // ----------------------------------------------------------- assertions

    /// Everything checked after every op: the stored fate, the retained
    /// bookkeeping, and both directions of the offer set.
    fn assert_agrees(&mut self, op_index: usize, op: &str) {
        let seed = self.seed;
        let target = self.target();
        let expected_offers = self.model_offer_set();
        let model = self.model.clone();
        let core = self.core.as_mut().expect("the runtime is present");

        for (batch_id, batch) in &model {
            let stored = core
                .storage()
                .load_authoritative_batch_fate(*batch_id)
                .expect("loading an authoritative batch fate should succeed");
            assert_eq!(
                stored.as_ref(),
                batch.fate.as_ref(),
                "seed {seed:#x} op {op_index} ({op}): storage must hold the fate the model \
                 recorded for {batch_id:?}. A `Missing` written here is a peer's request to \
                 resend recorded as a fact about the batch — it overwrites the unsettled fate \
                 that carries the batch into the reconnect offer set, and needs no settlement \
                 itself, so the ask is what makes answering impossible."
            );

            assert_eq!(
                core.storage()
                    .load_sealed_batch_submission(*batch_id)
                    .expect("loading a sealed batch submission should succeed")
                    .is_some(),
                batch.has_submission,
                "seed {seed:#x} op {op_index} ({op}): retained sealed submission for \
                 {batch_id:?} disagrees with the model"
            );
            assert_eq!(
                core.storage()
                    .load_local_batch_row_index(*batch_id)
                    .expect("loading a local batch row index should succeed")
                    .is_some(),
                batch.has_row_index,
                "seed {seed:#x} op {op_index} ({op}): retained row index for {batch_id:?} \
                 disagrees with the model. The row index is what survives a local-tier \
                 retirement, and it is how a stranded batch is still findable."
            );
        }

        let offered: BTreeSet<BatchId> = core
            .pending_batch_ids_needing_reconciliation_for_test()
            .into_iter()
            .collect();

        // The direction the outage lived in.
        for batch_id in &expected_offers {
            assert!(
                offered.contains(batch_id),
                "seed {seed:#x} op {op_index} ({op}): {batch_id:?} is unreachable. This node \
                 holds its rows and its fate is not terminal at {target:?}, so every reconnect \
                 must offer it — and it is absent from the offer set. Model fate \
                 {:?}, model bookkeeping submission={} record={} row_index={}. Offered: {:?}",
                model[batch_id].fate,
                model[batch_id].has_submission,
                model[batch_id].has_record,
                model[batch_id].has_row_index,
                offered,
            );
        }

        // And the other way: nothing settled may be offered. Open batches are
        // exempt — see the module doc: they ride the in-memory record cache, so
        // whether they belong in the offer set is a live design question this
        // oracle deliberately does not answer.
        for batch_id in &offered {
            let is_open = model.get(batch_id).is_some_and(|batch| !batch.sealed);
            if expected_offers.contains(batch_id) || is_open {
                continue;
            }
            let batch = model.get(batch_id);
            panic!(
                "seed {seed:#x} op {op_index} ({op}): {batch_id:?} is offered but the model \
                 says it is settled at {target:?} (fate {:?}). Re-offering a settled batch is \
                 work bought from every reconnect for nothing.",
                batch.map(|batch| batch.fate.clone()),
            );
        }
    }

    /// The second half of the invariant, checked at the two moments it is
    /// testable: a reconnect, and an inbound `Missing`.
    fn assert_offers_on_the_wire(
        &self,
        outbox: &[OutboxEntry],
        must_offer: &BTreeSet<BatchId>,
        op_index: usize,
        op: &str,
    ) {
        let seed = self.seed;
        for batch_id in must_offer {
            assert!(
                outbox_offers_batch(outbox, *batch_id),
                "seed {seed:#x} op {op_index} ({op}): nothing went out for {batch_id:?}. The \
                 row, its history and its batch row index are all on this node; the only thing \
                 missing is the answer. Outbox: {outbox:?}"
            );
        }
    }
}

fn outbox_offers_batch(outbox: &[OutboxEntry], batch_id: BatchId) -> bool {
    outbox.iter().any(|entry| match &entry.payload {
        SyncPayload::RowBatchCreated { row, .. } | SyncPayload::RowBatchNeeded { row, .. } => {
            row.batch_id == batch_id
        }
        SyncPayload::SealBatch { submission } => submission.batch_id == batch_id,
        _ => false,
    })
}

fn pooled_fate(rng: &mut Xorshift, batch_id: BatchId) -> BatchFate {
    let tier = match rng.below(3) {
        0 => DurabilityTier::Local,
        1 => DurabilityTier::EdgeServer,
        _ => DurabilityTier::GlobalServer,
    };
    match rng.below(10) {
        // Weighted at `Missing`: it is the instruction the fix is about, and
        // repeats of it are where the deduplication bug lived.
        0..=4 => BatchFate::Missing { batch_id },
        5 | 6 => BatchFate::DurableDirect {
            batch_id,
            confirmed_tier: tier,
        },
        7 | 8 => BatchFate::AcceptedTransaction {
            batch_id,
            confirmed_tier: tier,
        },
        _ => BatchFate::Rejected {
            batch_id,
            code: "policy_denied".into(),
            reason: "the differential oracle rejected this batch".into(),
        },
    }
}

/// Does this batch's mark history contain `pattern` as a contiguous run?
fn saw_run(marks: &[Mark], pattern: &[Mark]) -> bool {
    marks.windows(pattern.len()).any(|window| window == pattern)
}

fn tally_pairs(harness: &mut Harness<impl Storage>) {
    let marks = harness.marks.clone();
    for history in marks.values() {
        if saw_run(history, &[Mark::GlobalDurable, Mark::Missing]) {
            harness.counts.missing_right_after_global_durable += 1;
        }
        if saw_run(history, &[Mark::Missing, Mark::Missing]) {
            harness.counts.repeated_missing_no_change_between += 1;
        }
        if saw_run(history, &[Mark::Rejected, Mark::Missing]) {
            harness.counts.missing_after_rejected += 1;
        }
        if saw_run(
            history,
            &[
                Mark::OfflineCommit,
                Mark::Connect,
                Mark::Missing,
                Mark::Restart,
                Mark::Connect,
            ],
        ) {
            harness
                .counts
                .offline_commit_connect_missing_restart_connect += 1;
        }
        for window in history.windows(3) {
            if window[1] == Mark::Restart
                && matches!(
                    window[0],
                    Mark::Missing | Mark::Rejected | Mark::GlobalDurable
                )
                && matches!(
                    window[2],
                    Mark::Missing | Mark::Rejected | Mark::GlobalDurable
                )
            {
                harness.counts.restart_between_fate_deliveries += 1;
            }
        }
    }
}

// =============================================================================
// The scripted shapes
// =============================================================================

/// The sequences the current code was wrong on, played through the same model
/// and the same assertions as the randomized stream.
///
/// These are not a substitute for the random stream and not a duplicate of the
/// hand-written gates: they exist so that every run is guaranteed to have
/// executed the shapes the fix is about, no matter what the generator happened
/// to draw.
#[test]
fn batch_fate_offer_differential_required_shapes() {
    // `Missing` immediately after a global durable confirmation.
    {
        let mut h = Harness::new("fate-shape-durable-then-missing", MemoryStorage::new(), 1);
        h.connect();
        let batch_id = h.commit_direct();
        h.settle_and_assert(0, "commit_direct");
        h.deliver_fate(BatchFate::DurableDirect {
            batch_id,
            confirmed_tier: DurabilityTier::GlobalServer,
        });
        h.settle_and_assert(1, "durable(GlobalServer)");
        h.deliver_fate(BatchFate::Missing { batch_id });
        h.settle_and_assert(2, "missing after durable(GlobalServer)");
    }

    // The same `Missing` k times with nothing changing in between. The stored
    // fate used to make every repeat look like a duplicate.
    {
        let mut h = Harness::new("fate-shape-repeated-missing", MemoryStorage::new(), 2);
        h.connect();
        let batch_id = h.commit_transactional();
        h.seal(batch_id);
        h.settle_and_assert(0, "seal");
        for repeat in 0..4 {
            h.deliver_fate(BatchFate::Missing { batch_id });
            let outbox = h.settle_and_assert(1 + repeat, "repeated missing");
            let must = BTreeSet::from([batch_id]);
            h.assert_offers_on_the_wire(&outbox, &must, 1 + repeat, "repeated missing");
        }
    }

    // The production shape, end to end.
    {
        let mut h = Harness::new("fate-shape-offline-commit", MemoryStorage::new(), 3);
        let batch_id = h.commit_direct();
        h.settle_and_assert(0, "commit_direct offline");
        h.connect();
        let outbox = h.settle_and_assert(1, "connect");
        h.assert_offers_on_the_wire(&outbox, &BTreeSet::from([batch_id]), 1, "connect");
        h.deliver_fate(BatchFate::Missing { batch_id });
        let outbox = h.settle_and_assert(2, "missing");
        h.assert_offers_on_the_wire(&outbox, &BTreeSet::from([batch_id]), 2, "missing");
        h.restart(&std::convert::identity);
        h.settle_and_assert(3, "restart");
        h.connect();
        let outbox = h.settle_and_assert(4, "reconnect after restart");
        h.assert_offers_on_the_wire(
            &outbox,
            &BTreeSet::from([batch_id]),
            4,
            "reconnect after restart",
        );
    }

    // `Rejected` then `Missing`: the rejection must survive the ask.
    {
        let mut h = Harness::new("fate-shape-rejected-then-missing", MemoryStorage::new(), 4);
        h.connect();
        let batch_id = h.commit_direct();
        h.settle_and_assert(0, "commit_direct");
        h.deliver_fate(BatchFate::Rejected {
            batch_id,
            code: "policy_denied".into(),
            reason: "no".into(),
        });
        h.settle_and_assert(1, "rejected");
        h.deliver_fate(BatchFate::Missing { batch_id });
        h.settle_and_assert(2, "missing after rejected");
        h.restart(&std::convert::identity);
        h.settle_and_assert(3, "restart");
        h.deliver_fate(BatchFate::Missing { batch_id });
        h.settle_and_assert(4, "missing after restart");
    }
}

impl<S: Storage> Harness<S> {
    /// Drain the tick, take the outbox, and check every invariant. Returns the
    /// outbox so a caller can additionally assert what went on the wire.
    fn settle_and_assert(&mut self, op_index: usize, op: &str) -> Vec<OutboxEntry> {
        self.core().batched_tick();
        let outbox = self.core().sync_sender().take();
        self.assert_agrees(op_index, op);
        outbox
    }
}

// =============================================================================
// The randomized stream
// =============================================================================

#[test]
fn batch_fate_offer_differential_random_ops() {
    let mut totals = OpCounts::default();
    for seed in SEEDS {
        let mut harness = Harness::new("batch-fate-offer-differential", MemoryStorage::new(), seed);
        run_seed(&mut harness, seed, &std::convert::identity);
        tally_pairs(&mut harness);
        totals.merge(&harness.counts);
    }

    // Silence is only interpretable if the mix is reported. These bounds are
    // the floor under "the stream actually exercised the defect's shapes",
    // not a coverage target to tune.
    assert!(
        totals.restart > 100 && totals.connect > 100,
        "the op mix stopped exercising restarts and reconnects: {totals:?}"
    );
    assert!(
        totals.fate_missing_while_connected > 100,
        "the op mix stopped delivering `Missing` to a connected node: {totals:?}"
    );
    assert!(
        totals.commit_direct_offline > 20,
        "the op mix stopped committing while offline — the production shape: {totals:?}"
    );
    assert!(
        totals.wire_offers_demanded > 500,
        "the second half of the invariant — what actually goes on the wire — was barely \
         checked, so a green run says little about it: {totals:?}"
    );
    assert!(
        totals.repeated_missing_no_change_between > 0
            && totals.missing_right_after_global_durable > 0
            && totals.missing_after_rejected > 0
            && totals.restart_between_fate_deliveries > 0,
        "the randomized stream no longer covers the pairs the fix is about: {totals:?}"
    );
    println!("batch fate offer differential op mix: {totals:?}");
}

/// The same stream on a real store. `MemoryStorage` answers
/// `load_authoritative_batch_fate` from a live map, so on it "the stored fate
/// matches the model" never exercises `encode_storage_row`/`decode_storage_row`
/// — the fate could be unrepresentable on disk and this oracle would not
/// notice. Sqlite round-trips every read through the bytes.
///
/// Fewer seeds and a shorter stream than the memory arm. The question here is
/// whether a fate survives as bytes, and that is answered by the first restart
/// on each seed; op count buys interleavings, which the memory arm already
/// pays for far more cheaply.
#[test]
fn batch_fate_offer_differential_sqlite_round_trip() {
    for (index, seed) in SEEDS.iter().take(4).enumerate() {
        let path = std::env::temp_dir().join(format!(
            "jazz-fate-offer-differential-{}-{index}.sqlite",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        let storage = SqliteStorage::open(&path).expect("sqlite storage should open");
        let mut harness = Harness::new("batch-fate-offer-differential-sqlite", storage, *seed);
        // A restart reopens the file, so the fate really is read back from
        // bytes written by a process that is gone.
        let path_for_restart = path.clone();
        let rebuild = move |storage: SqliteStorage| {
            drop(storage);
            SqliteStorage::open(&path_for_restart).expect("sqlite storage should reopen")
        };
        run_seed_with_ops(&mut harness, *seed, &rebuild, 60);
        let _ = std::fs::remove_file(&path);
    }
}

fn run_seed<S: Storage>(harness: &mut Harness<S>, seed: u64, rebuild: &dyn Fn(S) -> S) {
    run_seed_with_ops(harness, seed, rebuild, OPS_PER_SEED);
}

fn run_seed_with_ops<S: Storage>(
    harness: &mut Harness<S>,
    seed: u64,
    rebuild: &dyn Fn(S) -> S,
    ops: usize,
) {
    let mut rng = Xorshift(seed);

    for op_index in 0..ops {
        let (label, emitted): (String, Emitted) = match rng.below(20) {
            0..=2 => {
                harness.commit_direct();
                ("commit_direct".into(), Emitted::Nothing)
            }
            3 => {
                harness.commit_transactional();
                ("commit_transactional".into(), Emitted::Nothing)
            }
            4 => {
                if harness.open_batches.is_empty() {
                    harness.commit_transactional();
                    (
                        "commit_transactional (no open batch to seal)".into(),
                        Emitted::Nothing,
                    )
                } else {
                    let index = rng.below(harness.open_batches.len());
                    let batch_id = harness.open_batches[index];
                    harness.seal(batch_id);
                    ("seal".into(), Emitted::Nothing)
                }
            }
            5..=10 => {
                if harness.order.is_empty() {
                    harness.commit_direct();
                    ("commit_direct (no batch to fate)".into(), Emitted::Nothing)
                } else {
                    let batch_id = pick_batch(&mut rng, &harness.order);
                    let fate = pooled_fate(&mut rng, batch_id);
                    let label = format!("deliver_fate {fate:?}");
                    let is_missing = matches!(fate, BatchFate::Missing { .. });
                    harness.deliver_fate(fate);
                    let emitted = if is_missing && harness.connected() {
                        Emitted::MissingFor(batch_id)
                    } else {
                        Emitted::Nothing
                    };
                    (label, emitted)
                }
            }
            11..=13 => {
                if harness.connected() {
                    harness.disconnect();
                    ("disconnect".into(), Emitted::Nothing)
                } else {
                    harness.connect();
                    ("connect".into(), Emitted::Reconnect)
                }
            }
            14..=16 => {
                harness.restart(rebuild);
                ("restart".into(), Emitted::Nothing)
            }
            17 | 18 => {
                harness.counts.tick += 1;
                harness.mark_all(Mark::Other);
                ("tick".into(), Emitted::Nothing)
            }
            _ => {
                // `retire_settled_batch` only ever runs against a batch whose
                // fate reached a confirmed tier — driving it otherwise would
                // manufacture a state the runtime cannot produce and turn a
                // model artefact into a finding.
                let retirable: Vec<(BatchId, DurabilityTier)> = harness
                    .model
                    .iter()
                    .filter_map(|(batch_id, batch)| {
                        batch
                            .fate
                            .as_ref()
                            .and_then(|fate| fate.confirmed_tier())
                            .map(|tier| (*batch_id, tier))
                    })
                    .collect();
                if retirable.is_empty() {
                    harness.counts.tick += 1;
                    harness.mark_all(Mark::Other);
                    ("tick (nothing retirable)".into(), Emitted::Nothing)
                } else {
                    let (batch_id, tier) = retirable[rng.below(retirable.len())];
                    harness.core().retire_settled_batch(batch_id, tier);
                    harness.model_retire(batch_id, tier);
                    harness.counts.retire += 1;
                    harness.mark(batch_id, Mark::Other);
                    (format!("retire {tier:?}"), Emitted::Nothing)
                }
            }
        };

        // What must be on the wire for THIS op, computed before the outbox is
        // drained. A reconnect owes the whole offer set; an inbound `Missing`
        // owes exactly the batch it asked about.
        let target = harness.target();
        let must_offer: BTreeSet<BatchId> = match emitted {
            Emitted::Nothing => BTreeSet::new(),
            Emitted::Reconnect => harness.model_offer_set(),
            Emitted::MissingFor(batch_id) => harness
                .model
                .get(&batch_id)
                .filter(|batch| batch.offerable(target))
                .map(|_| BTreeSet::from([batch_id]))
                .unwrap_or_default(),
        };

        harness.counts.wire_offers_demanded += must_offer.len();
        let outbox = harness.settle_and_assert(op_index, &label);
        harness.assert_offers_on_the_wire(&outbox, &must_offer, op_index, &label);
    }
}

/// What the op just performed owes the wire.
enum Emitted {
    Nothing,
    Reconnect,
    MissingFor(BatchId),
}
