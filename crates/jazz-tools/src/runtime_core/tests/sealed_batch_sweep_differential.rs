//! Differential oracle for the recovery sweep: a random stream of seals, rows, ticks,
//! reopens and storage faults against a model of what the sweep owes and what it may cost.
//!
//! The fix under test replaces "every tick re-reads every retained submission" with "every
//! tick reads the submissions something happened to since the last one, plus one walk of
//! the table when the store is opened". The hand-written gates in `sealed_batch_cost.rs`
//! cover the shapes we thought of; this stream is for the interleavings we did not.
//!
//! Two things are checked after every op:
//!
//! - **Liveness.** Every submission that is completable — sealed, all declared rows on
//!   disk, no fate yet — has been completed: the fate the model expects is stored, and the
//!   submission is retired. The node under test has no upstream, so its settlement target
//!   is `Local` and every fate it writes is terminal. Completion is owed by the end of any
//!   op that ends in a tick whose storage reads were not failing.
//! - **Cost.** On the memory arm, whose storage counts reads: after the opening walk, a tick
//!   reads fates in proportion to the batches touched since the previous tick, never in
//!   proportion to the `RETAINED` permanently uncompletable seals the store is seeded with.
//!
//! Routes into a drivable state that the stream exercises: a seal from a client (the arm
//! completes it on arrival when its rows are present), a row from a client (same), a row
//! from an upstream server (applied without completing the seal: the sweep's job), a seal
//! written to disk by a process that died before completing it (the opening walk's job),
//! and a tick whose fate reads fail (the batch must be looked at again, not dropped).
//!
//! Conscious limits: every batch declares fresh rows, so no transaction ever conflicts with
//! another and no `Rejected` fate is produced; rows from upstream are transactional only,
//! because a visible direct row from upstream records a fate on apply and the retained
//! seal it belongs to is then past the sweep's reach (an open defect gated in
//! `sync_manager/tests/server_origin_seal.rs`, unchanged here); fate promotion reaches a
//! node only through a hydrated batch record or the opening walk, neither of which the
//! stream drives, and is gated on reopen in `sealed_batch_cost.rs`.

use std::collections::{BTreeMap, BTreeSet};

use super::*;
use crate::batch_fate::{BatchFate, BatchMode};
use crate::storage::SqliteStorage;

const OPS_PER_SEED: usize = 200;
const SEEDS: [u64; 16] = [
    0x5EA1_0F5E_0000_0001,
    0x5EA1_0F5E_0000_0002,
    0x5EA1_0F5E_0000_0003,
    0x5EA1_0F5E_0000_0004,
    0x5EA1_0F5E_0000_0005,
    0x5EA1_0F5E_0000_0006,
    0x5EA1_0F5E_0000_0007,
    0x5EA1_0F5E_0000_0008,
    0x5EA1_0F5E_0000_0009,
    0x5EA1_0F5E_0000_000A,
    0x5EA1_0F5E_0000_000B,
    0x5EA1_0F5E_0000_000C,
    0x5EA1_0F5E_0000_000D,
    0x5EA1_0F5E_0000_000E,
    0x5EA1_0F5E_0000_000F,
    0x5EA1_0F5E_0000_0010,
];

/// Permanently uncompletable seals on disk before the first tick: no fate, no rows. The
/// population the production store carries. Large enough that a sweep whose reads track
/// the table is unmistakable next to one whose reads track the stream.
const RETAINED: usize = 300;

/// The most fate point-reads one touched batch was observed to cost an op across the
/// stream, with margin: a client row that completes its seal on arrival costs the arm's
/// own check, the settlement's re-read and the merge-before-write inside the persist,
/// then the sweep's look at the batch the row noted, and the same again for the seal
/// (14 measured). What matters is that this is a constant per touched batch and not a
/// function of `RETAINED`.
const FATE_READS_PER_TOUCHED_BATCH: usize = 16;
/// Reads an op makes with nothing touched: the sweep's look at whatever a completion's
/// row patching put back in front of it.
const FATE_READS_PER_OP: usize = 2;

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

struct ModelBatch {
    mode: BatchMode,
    rows: Vec<crate::row_histories::StoredRowBatch>,
    /// The submission has been written — by the inbox, or straight to disk before a reopen.
    sealed: bool,
    present: BTreeSet<ObjectId>,
    /// Completed: the terminal fate is stored and the submission retired.
    complete: bool,
}

impl ModelBatch {
    fn completable(&self) -> bool {
        self.sealed
            && !self.complete
            && self
                .rows
                .iter()
                .all(|row| self.present.contains(&row.row_id))
    }

    fn expected_fate(&self, batch_id: BatchId) -> BatchFate {
        match self.mode {
            BatchMode::Direct => BatchFate::DurableDirect {
                batch_id,
                confirmed_tier: DurabilityTier::Local,
            },
            BatchMode::Transactional => BatchFate::AcceptedTransaction {
                batch_id,
                confirmed_tier: DurabilityTier::Local,
            },
        }
    }

    fn seal(&self, batch_id: BatchId) -> SealedBatchSubmission {
        seal_declaring(batch_id, self.mode, &self.rows)
    }
}

#[derive(Default, Debug)]
struct OpCounts {
    new_batch: usize,
    seal_from_client: usize,
    row_from_client: usize,
    row_from_upstream: usize,
    tick: usize,
    restart: usize,
    seal_on_disk_then_restart: usize,
    faulted_tick: usize,
    completed_on_arrival: usize,
    completed_by_sweep: usize,
    completed_by_opening_walk: usize,
}

struct Harness {
    core: Option<SweepingCore>,
    app_name: &'static str,
    client_id: ClientId,
    upstream: ServerId,
    model: BTreeMap<BatchId, ModelBatch>,
    order: Vec<BatchId>,
    /// Read counters, on the arm whose storage has them.
    sweep: Option<Arc<Mutex<SweepCallCounts>>>,
    fail_fate_gets: Arc<Mutex<bool>>,
    /// Batches an op touched (sealed, row applied) since the last cost check — carried
    /// across an op whose reads were failing, because the sweep re-queues what it could not
    /// read — and the ones the previous check saw completed, which buys the budget slack for
    /// the runtime's own re-read of a fresh fate (`apply_received_batch_fate`) landing in
    /// the op after the one that measured the completion.
    touched: BTreeSet<BatchId>,
    completed_last: BTreeSet<BatchId>,
    /// The next tick is the opening walk: it may read everything once.
    opening: bool,
    counts: OpCounts,
    seed: u64,
}

impl Harness {
    fn new(
        app_name: &'static str,
        storage: Box<dyn Storage>,
        sweep: Option<Arc<Mutex<SweepCallCounts>>>,
        fail_fate_gets: Arc<Mutex<bool>>,
        seed: u64,
    ) -> Self {
        let mut storage = storage;
        for index in 0..RETAINED {
            storage
                .upsert_sealed_batch_submission(&SealedBatchSubmission::new(
                    BatchId::new(),
                    BatchMode::Direct,
                    crate::object::BranchName::new("main"),
                    vec![SealedBatchMember {
                        object_id: ObjectId::new(),
                        row_digest: crate::digest::Digest32([index as u8; 32]),
                    }],
                    Vec::new(),
                ))
                .expect("seed a retained submission");
        }
        let mut harness = Self {
            core: None,
            app_name,
            client_id: ClientId::new(),
            upstream: ServerId::new(),
            model: BTreeMap::new(),
            order: Vec::new(),
            sweep,
            fail_fate_gets,
            touched: BTreeSet::new(),
            completed_last: BTreeSet::new(),
            opening: true,
            counts: OpCounts::default(),
            seed,
        };
        harness.open(storage);
        harness
    }

    /// Build the runtime over `storage` and let it open the store. `add_client` ticks, so
    /// the opening walk happens in here; it is the one sweep allowed to read the table.
    fn open(&mut self, storage: Box<dyn Storage>) {
        self.reset_reads();
        let mut core = sweeping_core_over(storage, DurabilityTier::Local, self.app_name);
        core.add_client(self.client_id, None);
        core.schema_manager_mut()
            .query_manager_mut()
            .sync_manager_mut()
            .set_client_role(self.client_id, ClientRole::Peer);
        core.immediate_tick();
        if let Some(sweep) = &self.sweep {
            let reads = *sweep.lock().unwrap();
            assert!(
                reads.sealed_submission_key_scans >= 1 && reads.authoritative_fate_gets >= RETAINED,
                "seed {:#x}: opening the store must walk the submission table once and look \
                 at every retained submission; walked {} times, read {} fates",
                self.seed,
                reads.sealed_submission_key_scans,
                reads.authoritative_fate_gets
            );
        }
        self.core = Some(core);
        self.opening = true;
    }

    fn core(&mut self) -> &mut SweepingCore {
        self.core.as_mut().expect("the runtime is present")
    }

    fn reset_reads(&self) {
        if let Some(sweep) = &self.sweep {
            *sweep.lock().unwrap() = SweepCallCounts::default();
        }
    }

    // ------------------------------------------------------------------ ops

    fn new_batch(&mut self, rng: &mut Xorshift) -> BatchId {
        let batch_id = BatchId::new();
        let mode = if rng.below(2) == 0 {
            BatchMode::Direct
        } else {
            BatchMode::Transactional
        };
        let state = match mode {
            BatchMode::Direct => crate::row_histories::RowState::VisibleDirect,
            BatchMode::Transactional => crate::row_histories::RowState::StagingPending,
        };
        let rows = (0..1 + rng.below(2))
            .map(|_| user_row(ObjectId::new(), batch_id, state, None))
            .collect();
        self.model.insert(
            batch_id,
            ModelBatch {
                mode,
                rows,
                sealed: false,
                present: BTreeSet::new(),
                complete: false,
            },
        );
        self.order.push(batch_id);
        self.counts.new_batch += 1;
        batch_id
    }

    fn seal_from_client(&mut self, batch_id: BatchId) {
        let submission = self.model[&batch_id].seal(batch_id);
        let client_id = self.client_id;
        self.deliver(
            Source::Client(client_id),
            SyncPayload::SealBatch { submission },
        );
        let batch = self.model.get_mut(&batch_id).expect("modelled batch");
        batch.sealed = true;
        self.touched.insert(batch_id);
        self.counts.seal_from_client += 1;
    }

    fn row_from_client(&mut self, batch_id: BatchId, index: usize) {
        let row = self.model[&batch_id].rows[index].clone();
        let row_id = row.row_id;
        let client_id = self.client_id;
        self.deliver(
            Source::Client(client_id),
            SyncPayload::RowBatchCreated {
                metadata: Some(users_row_metadata(row_id)),
                row,
            },
        );
        self.model
            .get_mut(&batch_id)
            .expect("modelled batch")
            .present
            .insert(row_id);
        self.touched.insert(batch_id);
        self.counts.row_from_client += 1;
    }

    /// Transactional batches only — see the module comment.
    fn row_from_upstream(&mut self, batch_id: BatchId, index: usize) {
        let row = self.model[&batch_id].rows[index].clone();
        let row_id = row.row_id;
        let upstream = self.upstream;
        self.deliver(
            Source::Server(upstream),
            SyncPayload::RowBatchNeeded {
                metadata: Some(users_row_metadata(row_id)),
                row,
            },
        );
        self.model
            .get_mut(&batch_id)
            .expect("modelled batch")
            .present
            .insert(row_id);
        self.touched.insert(batch_id);
        self.counts.row_from_upstream += 1;
    }

    fn deliver(&mut self, source: Source, payload: SyncPayload) {
        let core = self.core();
        core.park_sync_message(InboxEntry { source, payload });
        core.batched_tick();
        core.sync_sender().take();
    }

    fn tick(&mut self) {
        self.core().immediate_tick();
        self.counts.tick += 1;
    }

    fn restart(&mut self, rebuild: &dyn Fn(Box<dyn Storage>) -> Box<dyn Storage>) {
        let mut core = self.core.take().expect("the runtime is present");
        core.batched_tick();
        core.sync_sender().take();
        // An orderly shutdown: what the runtime wrote is on disk. Whether every inbox
        // write is flushed promptly is a different question from what the sweep does with
        // what it finds, and this oracle asks only the second.
        core.flush_storage()
            .expect("flush storage before the runtime goes away");
        let storage = rebuild(core.into_storage());
        self.open(storage);
        self.counts.restart += 1;
    }

    /// The crash window: a submission reached the disk and the process died before the
    /// completion that follows the write in the same call.
    fn seal_on_disk_then_restart(
        &mut self,
        batch_id: BatchId,
        rebuild: &dyn Fn(Box<dyn Storage>) -> Box<dyn Storage>,
    ) {
        let submission = self.model[&batch_id].seal(batch_id);
        let mut core = self.core.take().expect("the runtime is present");
        core.batched_tick();
        core.sync_sender().take();
        core.flush_storage()
            .expect("flush storage before the runtime goes away");
        // Written by the process that is gone, on the store the next one opens: a write
        // made through the live runtime's handle is not durable until that runtime flushes.
        let mut storage = rebuild(core.into_storage());
        storage
            .upsert_sealed_batch_submission(&submission)
            .expect("write the seal to disk");
        self.model
            .get_mut(&batch_id)
            .expect("modelled batch")
            .sealed = true;
        self.open(storage);
        self.counts.restart += 1;
        self.counts.seal_on_disk_then_restart += 1;
    }

    // ----------------------------------------------------------- assertions

    /// Everything checked after every op. `settles` says whether this op ended in a tick
    /// whose reads were allowed to succeed, and so whether completion is owed now.
    fn assert_agrees(&mut self, op_index: usize, op: &str, settles: bool) {
        let seed = self.seed;
        let opening = self.opening;
        let reads = self.sweep.as_ref().map(|sweep| *sweep.lock().unwrap());
        let mut completed_now = BTreeSet::new();
        let batch_ids: Vec<BatchId> = self.order.clone();
        for batch_id in batch_ids {
            let (expected_fate, completable, complete, sealed) = {
                let batch = &self.model[&batch_id];
                (
                    batch.expected_fate(batch_id),
                    batch.completable(),
                    batch.complete,
                    batch.sealed,
                )
            };
            let stored_fate = self
                .core()
                .storage()
                .load_authoritative_batch_fate(batch_id)
                .expect("loading a batch fate should succeed");
            let stored_seal = self
                .core()
                .storage()
                .load_sealed_batch_submission(batch_id)
                .expect("loading a submission should succeed")
                .is_some();

            if complete || (completable && settles) {
                assert_eq!(
                    stored_fate.as_ref(),
                    Some(&expected_fate),
                    "seed {seed:#x} op {op_index} ({op}): batch {batch_id:?} is completable \
                     — sealed, every declared row on disk — and the op ended in a tick, so \
                     it must be complete. A submission that became drivable and was not \
                     looked at again is exactly what a sweep that reads only what changed \
                     can get wrong. Sealed={sealed} completable={completable} settles={settles}"
                );
                assert!(
                    !stored_seal,
                    "seed {seed:#x} op {op_index} ({op}): batch {batch_id:?} is complete at \
                     this node's settlement target and its submission must be retired"
                );
                if !complete {
                    completed_now.insert(batch_id);
                }
            } else {
                // Not completable, or completable under failing reads: nothing may have
                // been written on its behalf, and its submission is retained iff sealed.
                assert_eq!(
                    stored_fate, None,
                    "seed {seed:#x} op {op_index} ({op}): batch {batch_id:?} has no rows to \
                     complete it (or its reads were failing) and must not have a fate"
                );
                assert_eq!(
                    stored_seal, sealed,
                    "seed {seed:#x} op {op_index} ({op}): batch {batch_id:?} submission \
                     presence must follow whether it was sealed"
                );
            }
        }
        for batch_id in &completed_now {
            self.model
                .get_mut(batch_id)
                .expect("modelled batch")
                .complete = true;
        }

        // Cost: after the opening walk, what a tick read is what changed. The opening walk
        // itself may read the whole table once; nothing else may.
        if let Some(reads) = reads
            && !opening
        {
            let budget = FATE_READS_PER_TOUCHED_BATCH
                * (self.touched.len() + self.completed_last.len())
                + FATE_READS_PER_OP;
            assert!(
                reads.authoritative_fate_gets <= budget && reads.sealed_submission_key_scans == 0,
                "seed {seed:#x} op {op_index} ({op}): the op read {} batch fates and walked \
                 the submission table {} times, with {} batches touched since the last check \
                 and {} completed by the previous one, over {RETAINED} retained submissions \
                 nothing happened to. Reads must follow what changed, not what the store \
                 kept.",
                reads.authoritative_fate_gets,
                reads.sealed_submission_key_scans,
                self.touched.len(),
                self.completed_last.len()
            );
        }
        if opening && matches!(op, "restart" | "seal_on_disk_then_restart") {
            self.counts.completed_by_opening_walk += completed_now.len();
        } else if matches!(op, "seal_from_client" | "row_from_client") {
            self.counts.completed_on_arrival += completed_now.len();
        } else {
            self.counts.completed_by_sweep += completed_now.len();
        }
        self.opening = false;
        self.completed_last = completed_now;
        if settles {
            self.touched.clear();
        }
        self.reset_reads();
    }
}

fn run_seed(
    harness: &mut Harness,
    seed: u64,
    rebuild: &dyn Fn(Box<dyn Storage>) -> Box<dyn Storage>,
    ops: usize,
    with_faults: bool,
) {
    let mut rng = Xorshift(seed);
    // The store was opened and walked in `Harness::new`; the tick after it is an ordinary one.
    harness.tick();
    harness.assert_agrees(0, "open", true);

    for op_index in 1..=ops {
        let (label, settles): (&str, bool) = match rng.below(24) {
            0..=3 => {
                harness.new_batch(&mut rng);
                harness.tick();
                ("new_batch", true)
            }
            4..=7 => {
                let candidates: Vec<BatchId> = harness
                    .model
                    .iter()
                    .filter(|(_, batch)| !batch.sealed)
                    .map(|(batch_id, _)| *batch_id)
                    .collect();
                if candidates.is_empty() {
                    harness.new_batch(&mut rng);
                    harness.tick();
                    ("new_batch (nothing to seal)", true)
                } else {
                    let batch_id = candidates[rng.below(candidates.len())];
                    harness.seal_from_client(batch_id);
                    harness.tick();
                    ("seal_from_client", true)
                }
            }
            8..=13 => {
                let candidates: Vec<(BatchId, usize)> = harness
                    .model
                    .iter()
                    .flat_map(|(batch_id, batch)| {
                        batch
                            .rows
                            .iter()
                            .enumerate()
                            .filter(|(_, row)| !batch.present.contains(&row.row_id))
                            .map(|(index, _)| (*batch_id, index))
                            .collect::<Vec<_>>()
                    })
                    .collect();
                if candidates.is_empty() {
                    harness.new_batch(&mut rng);
                    harness.tick();
                    ("new_batch (no row to send)", true)
                } else {
                    let (batch_id, index) = candidates[rng.below(candidates.len())];
                    harness.row_from_client(batch_id, index);
                    harness.tick();
                    ("row_from_client", true)
                }
            }
            14..=18 => {
                let candidates: Vec<(BatchId, usize)> = harness
                    .model
                    .iter()
                    .filter(|(_, batch)| batch.mode == BatchMode::Transactional)
                    .flat_map(|(batch_id, batch)| {
                        batch
                            .rows
                            .iter()
                            .enumerate()
                            .filter(|(_, row)| !batch.present.contains(&row.row_id))
                            .map(|(index, _)| (*batch_id, index))
                            .collect::<Vec<_>>()
                    })
                    .collect();
                if candidates.is_empty() {
                    harness.tick();
                    ("tick (no upstream row to send)", true)
                } else {
                    let (batch_id, index) = candidates[rng.below(candidates.len())];
                    if with_faults && rng.below(4) == 0 {
                        // The row lands, and the tick that would complete its seal cannot
                        // read a single fate.
                        *harness.fail_fate_gets.lock().unwrap() = true;
                        harness.row_from_upstream(batch_id, index);
                        harness.tick();
                        *harness.fail_fate_gets.lock().unwrap() = false;
                        harness.counts.faulted_tick += 1;
                        ("row_from_upstream under failing fate reads", false)
                    } else {
                        harness.row_from_upstream(batch_id, index);
                        harness.tick();
                        ("row_from_upstream", true)
                    }
                }
            }
            19 | 20 => {
                harness.tick();
                ("tick", true)
            }
            21 => {
                harness.restart(rebuild);
                harness.tick();
                ("restart", true)
            }
            _ => {
                let candidates: Vec<BatchId> = harness
                    .model
                    .iter()
                    .filter(|(_, batch)| !batch.sealed)
                    .map(|(batch_id, _)| *batch_id)
                    .collect();
                if candidates.is_empty() {
                    harness.restart(rebuild);
                    harness.tick();
                    ("restart", true)
                } else {
                    let batch_id = candidates[rng.below(candidates.len())];
                    harness.seal_on_disk_then_restart(batch_id, rebuild);
                    harness.tick();
                    ("seal_on_disk_then_restart", true)
                }
            }
        };
        harness.assert_agrees(op_index, label, settles);
    }

    // A seed may end on an op whose reads were failing; the batch it left completable is
    // owed to the next tick like any other.
    harness.tick();
    harness.assert_agrees(ops + 1, "close", true);
}

/// The randomized stream on the counting memory storage: liveness and cost.
#[test]
fn sealed_batch_sweep_differential() {
    let mut totals = OpCounts::default();
    for seed in SEEDS {
        let sweep = Arc::new(Mutex::new(SweepCallCounts::default()));
        let fail_fate_gets = Arc::new(Mutex::new(false));
        let storage = Box::new(RowMutationObservingStorage::observing_sweep_with_faults(
            Arc::clone(&sweep),
            Arc::clone(&fail_fate_gets),
            Arc::new(Mutex::new(false)),
        )) as Box<dyn Storage>;
        let mut harness = Harness::new(
            "sealed-batch-sweep-differential",
            storage,
            Some(sweep),
            fail_fate_gets,
            seed,
        );
        run_seed(&mut harness, seed, &|storage| storage, OPS_PER_SEED, true);
        let counts = harness.counts;
        totals.new_batch += counts.new_batch;
        totals.seal_from_client += counts.seal_from_client;
        totals.row_from_client += counts.row_from_client;
        totals.row_from_upstream += counts.row_from_upstream;
        totals.tick += counts.tick;
        totals.restart += counts.restart;
        totals.seal_on_disk_then_restart += counts.seal_on_disk_then_restart;
        totals.faulted_tick += counts.faulted_tick;
        totals.completed_on_arrival += counts.completed_on_arrival;
        totals.completed_by_sweep += counts.completed_by_sweep;
        totals.completed_by_opening_walk += counts.completed_by_opening_walk;
    }
    // Coverage: every route into a drivable state and every way out of it was exercised.
    assert!(
        totals.completed_on_arrival > 0
            && totals.completed_by_sweep > 0
            && totals.completed_by_opening_walk > 0
            && totals.faulted_tick > 0
            && totals.seal_on_disk_then_restart > 0,
        "the stream did not cover every completion route: {totals:?}"
    );
}

/// The same stream on the backend the server persists to, reopening the file on every
/// restart: what the opening walk reads is bytes written by a process that is gone.
#[test]
fn sealed_batch_sweep_differential_sqlite_reopen() {
    for (index, seed) in SEEDS.iter().take(4).enumerate() {
        let path = std::env::temp_dir().join(format!(
            "jazz-sealed-batch-sweep-differential-{}-{index}.sqlite",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        let storage = Box::new(SqliteStorage::open(&path).expect("sqlite storage should open"))
            as Box<dyn Storage>;
        let mut harness = Harness::new(
            "sealed-batch-sweep-differential-sqlite",
            storage,
            None,
            Arc::new(Mutex::new(false)),
            *seed,
        );
        let path_for_restart = path.clone();
        let rebuild = move |storage: Box<dyn Storage>| {
            drop(storage);
            Box::new(SqliteStorage::open(&path_for_restart).expect("sqlite storage should reopen"))
                as Box<dyn Storage>
        };
        run_seed(&mut harness, *seed, &rebuild, 60, false);
        let _ = std::fs::remove_file(&path);
    }
}
