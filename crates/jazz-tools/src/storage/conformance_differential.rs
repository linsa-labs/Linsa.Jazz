//! Randomized differential test: stored visible entries vs full-history rebuilds.
//!
//! This is the equivalence backbone for the row-history fast paths
//! (`apply_row_batch` / `patch_row_batch_state` shortcuts that will skip
//! loading full row history): after EVERY mutation the stored visible-region
//! entry must be byte-identical to a [`VisibleRowEntry`] rebuilt from scratch
//! over the row's complete branch history. Today both mutation verbs already
//! recompute from full history, so the invariant holds trivially — the test
//! pins the contract the fast paths must preserve.
//!
//! The crate deliberately has no proptest/quickcheck dependency, so the op
//! stream comes from a hand-rolled xorshift PRNG over a fixed seed list. All
//! row/batch ids and timestamps also derive from the seed, so a failing seed
//! replays identically; every assertion message carries the seed and op index.
//!
//! Invoked per storage backend through `storage_conformance_tests!`.
//!
//! Generator limitations (conscious, documented):
//! - Single branch only. TODO: multi-branch op streams (per-branch frontiers
//!   and `patch_row_batch_state`'s cross-branch early return are untouched).
//! - The empty-payload hard-delete row must never become a concurrent tip of
//!   any (unfiltered or tier-filtered) preview frontier: the current merge
//!   resolution errors on that shape for counter columns ("counter merge
//!   expected INTEGER contender ... got Null"), i.e. the full path itself
//!   rejects such histories today. Hence: hard deletes name every current tip
//!   as parent, carry no confirmed tier (keeps them out of tier-filtered row
//!   sets), are excluded from tier bumps, and once a row hard-deletes, forks
//!   are disabled for it, previously staged batches are never published, and nothing is
//!   rejected any more: rejecting a version the delete descends from through others
//!   hands its parent back as a tip beside the delete, which is that same shape.
//! - `RowVisibilityChange` events are not compared, only the persisted entry.
//! - No storage restarts mid-stream (persistence is covered by the
//!   close/reopen conformance tests).

use std::collections::{HashMap, HashSet};

use uuid::Uuid;

use crate::metadata::{DeleteKind, MetadataKey, RowProvenance};
use crate::object::{BranchName, ObjectId};
use crate::query_manager::types::{
    ColumnDescriptor, ColumnMergeStrategy, ColumnType, RowDescriptor, Schema, TableName,
    TableSchema, Value,
};
use crate::row_format::encode_row;
use crate::row_histories::FORKED_FASTPATH_ARMS_ON_THREAD;
use crate::row_histories::{
    BatchId, HISTORY_FASTPATH_HITS, HistoryScan, PATCH_FASTPATH_HITS, RowState, StoredRowBatch,
    VisibleRowEntry, apply_row_batch, decode_flat_visible_row_entry, encode_flat_visible_row_entry,
    force_history_fastpath, patch_row_batch_state,
};
use crate::storage::{RowLocator, Storage};
use crate::sync_manager::DurabilityTier;
use crate::test_support::persist_test_schema;

const BRANCH: &str = "main";
/// Entries each stream must have built by each arm of the path for forked rows: by a
/// write over every tip of a row with several, and by a write over one of them. The run
/// is the same on every backend and every time (fixed seeds): 73 and 237 for the first
/// tierless stream, 63 and 279 for the one with skewed clocks. The floors sit just under
/// the smaller of each, so that a change which takes fewer writes has to say so here.
const OVER_EVERY_TIP_FLOOR: u64 = 60;
const OVER_ONE_TIP_FLOOR: u64 = 230;
const TABLES: [&str; 2] = ["diff_docs", "diff_notes"];
const ROWS_PER_TABLE: usize = 2;
const OPS_PER_SEED: usize = 200;
const SEEDS: [u64; 8] = [
    0x0BAD_5EED_0000_0001,
    0x0BAD_5EED_0000_0002,
    0x0BAD_5EED_0000_0003,
    0x0BAD_5EED_0000_0004,
    0xD1FF_0000_0000_0005,
    0xD1FF_0000_0000_0006,
    0xD1FF_0000_0000_0007,
    0xD1FF_0000_0000_0008,
];
const AUTHORS: [&str; 3] = ["alice", "bob", "carol"];
const TAG_POOL: [&str; 4] = ["red", "green", "blue", "gold"];
const TIERS: [DurabilityTier; 3] = [
    DurabilityTier::Local,
    DurabilityTier::EdgeServer,
    DurabilityTier::GlobalServer,
];

/// What a seed's op stream is made of, besides the ops every stream has.
///
/// One stream cannot stand for every store. A version taken as a direct write carries no
/// tier — the tier lives with the batch's fate — so a store that only ever takes direct
/// writes holds rows no tier has a view of, and the rules for such rows are their own
/// (`fastpath::try_forked_fastpath_entry` admits nothing else). A stream where every
/// second version has a tier almost never reaches them. And writers' clocks disagree:
/// a version may be older, by its own timestamp, than the version it was written over,
/// and the merge of two tips picks their common ancestor by timestamp.
#[derive(Clone, Copy, Debug)]
struct Stream {
    /// No version carries a tier, and no tier is ever raised.
    tierless: bool,
    /// One write in four is stamped earlier than the write before it.
    skewed: bool,
}

/// Tiers with disagreeing clocks is not a stream yet: it runs into an open defect of the
/// serial path that is older than the streams (seed `0xd1ff000000000008`, op 34, with
/// that stream; `row_histories::tests::
/// a_tier_view_keeps_a_creation_a_delivered_snapshot_superseded`).
const STREAMS: [Stream; 3] = [
    Stream {
        tierless: false,
        skewed: false,
    },
    Stream {
        tierless: true,
        skewed: false,
    },
    Stream {
        tierless: true,
        skewed: true,
    },
];

/// Entry point invoked by `storage_conformance_tests!` for every backend.
///
/// Dual run: the invariant must hold with the serial-write fast path enabled
/// (the default — this is the direct equivalence proof for the fast path) AND
/// with it disabled (pins the full path unchanged). The forced-mode guard
/// also serialises differential runs across backends so their overrides never
/// interleave.
pub fn test_visible_entry_differential_random_ops(factory: &dyn Fn() -> Box<dyn Storage>) {
    use std::sync::atomic::Ordering;

    for fastpath_enabled in [true, false] {
        let _mode = force_history_fastpath(fastpath_enabled);
        let apply_hits_before = HISTORY_FASTPATH_HITS.load(Ordering::Relaxed);
        let patch_hits_before = PATCH_FASTPATH_HITS.load(Ordering::Relaxed);
        for stream in STREAMS {
            let forked_before = FORKED_FASTPATH_ARMS_ON_THREAD.get();
            for seed in SEEDS {
                SeedRun::new(factory(), seed, stream).run();
            }
            let forked = FORKED_FASTPATH_ARMS_ON_THREAD.get();
            let [over_every_tip, over_one_tip] =
                [forked[0] - forked_before[0], forked[1] - forked_before[1]];
            println!(
                "{stream:?}: entries built from a row's tips: {over_every_tip} by a write \
                 over every tip, {over_one_tip} by a write over one of several"
            );
            if fastpath_enabled && stream.tierless {
                assert!(
                    over_every_tip > OVER_EVERY_TIP_FLOOR && over_one_tip > OVER_ONE_TIP_FLOOR,
                    "{stream:?}: the run barely built an entry from a row's tips \
                     ({over_every_tip} by a write over every tip, {over_one_tip} by a write \
                     over one of several) — the equivalence check would prove nothing \
                     about that path"
                );
            }
        }
        if fastpath_enabled {
            assert!(
                HISTORY_FASTPATH_HITS.load(Ordering::Relaxed) > apply_hits_before,
                "differential run with the fast path enabled never took the \
                 apply fast path — the equivalence check would prove nothing"
            );
            assert!(
                PATCH_FASTPATH_HITS.load(Ordering::Relaxed) > patch_hits_before,
                "differential run with the fast path enabled never took the \
                 patch fast path — the equivalence check would prove nothing"
            );
        }
    }
}

/// xorshift64* — deterministic, dependency-free.
struct Prng(u64);

impl Prng {
    fn new(seed: u64) -> Self {
        // xorshift state must be non-zero.
        Self(seed | 1)
    }

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

    fn coin(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }
}

/// Deterministic id mint: unique uuids derived from (seed, counter), so a
/// failing seed replays with identical row and batch identities.
struct IdMint {
    seed: u64,
    counter: u64,
}

impl IdMint {
    fn next_uuid(&mut self) -> Uuid {
        self.counter += 1;
        Uuid::from_u128((u128::from(self.seed) << 64) | u128::from(self.counter))
    }
}

#[derive(Clone, Copy, Debug)]
enum OpKind {
    SerialAppend,
    Fork,
    MergeCommit,
    SupersetMerge,
    StepParentAppend,
    StagingApply,
    StagingPublish,
    Reject,
    TierBumpReapply,
    TierBumpPatch,
    StrippedCopy,
    SoftDelete,
    HardDelete,
}

const OP_WEIGHTS: [(OpKind, u32); 13] = [
    (OpKind::SerialAppend, 30),
    (OpKind::Fork, 12),
    (OpKind::MergeCommit, 8),
    (OpKind::SupersetMerge, 5),
    (OpKind::StepParentAppend, 8),
    (OpKind::StagingApply, 10),
    (OpKind::StagingPublish, 10),
    (OpKind::Reject, 8),
    (OpKind::TierBumpReapply, 10),
    (OpKind::TierBumpPatch, 6),
    (OpKind::StrippedCopy, 8),
    (OpKind::SoftDelete, 4),
    (OpKind::HardDelete, 2),
];

fn pick_op(prng: &mut Prng) -> OpKind {
    let total: u32 = OP_WEIGHTS.iter().map(|(_, weight)| weight).sum();
    let mut roll = prng.below(total as usize) as u32;
    for (op, weight) in OP_WEIGHTS {
        if roll < weight {
            return op;
        }
        roll -= weight;
    }
    unreachable!("op roll is always below the summed weights")
}

/// Plain LWW columns plus one Counter-strategy and one GSet-strategy column,
/// so forks exercise per-column merge, not just last-writer-wins.
fn differential_descriptor() -> RowDescriptor {
    RowDescriptor::new(vec![
        ColumnDescriptor::new("title", ColumnType::Text),
        ColumnDescriptor::new("score", ColumnType::Integer),
        ColumnDescriptor::new("count", ColumnType::Integer)
            .merge_strategy(ColumnMergeStrategy::Counter),
        ColumnDescriptor::new(
            "tags",
            ColumnType::Array {
                element: Box::new(ColumnType::Text),
            },
        )
        .merge_strategy(ColumnMergeStrategy::GSet),
    ])
}

fn differential_schema() -> Schema {
    TABLES
        .iter()
        .map(|table| {
            (
                TableName::new(*table),
                TableSchema::new(differential_descriptor()),
            )
        })
        .collect()
}

fn random_values(prng: &mut Prng) -> Vec<Value> {
    let tags = TAG_POOL
        .iter()
        .filter(|_| prng.coin())
        .map(|tag| Value::Text((*tag).to_string()))
        .collect();
    vec![
        Value::Text(format!("t{}", prng.below(1000))),
        Value::Integer(prng.below(1000) as i32),
        Value::Integer(prng.below(100) as i32),
        Value::Array(tags),
    ]
}

/// Generator-side mirror of one row's history, enough to construct valid ops
/// (parents must reference existing batch ids) without re-reading storage.
struct RowHarness {
    table: &'static str,
    row_id: ObjectId,
    descriptor: RowDescriptor,
    /// Visible branch frontier after the previous op (from the last rebuild).
    frontier: Vec<BatchId>,
    /// Every batch id present in this row's history, in application order.
    all_batches: Vec<BatchId>,
    /// What each batch was written over.
    parents: HashMap<BatchId, Vec<BatchId>>,
    /// The generator's view of each batch's current state.
    batch_states: HashMap<BatchId, RowState>,
    /// StagingPending batches still eligible for publishing.
    staging: Vec<BatchId>,
    /// Hard-delete batch ids — excluded from tier bumps (see module docs).
    hard_delete_batches: HashSet<BatchId>,
    /// Sticky once a hard-delete batch lands (see module docs).
    hard_deleted: bool,
    /// (created_by, created_at) of the row's first batch, reused for updates.
    created: Option<(String, u64)>,
    next_ts: u64,
}

struct SeedRun {
    storage: Box<dyn Storage>,
    branch: BranchName,
    prng: Prng,
    mint: IdMint,
    rows: Vec<RowHarness>,
    seed: u64,
    stream: Stream,
}

impl SeedRun {
    fn new(mut storage: Box<dyn Storage>, seed: u64, stream: Stream) -> Self {
        let schema = differential_schema();
        let schema_hash = persist_test_schema(storage.as_mut(), &schema);
        let mut mint = IdMint { seed, counter: 0 };
        let mut rows = Vec::new();
        for table in TABLES {
            let descriptor = schema[&table.into()].columns.clone();
            for _ in 0..ROWS_PER_TABLE {
                let row_id = ObjectId::from_uuid(mint.next_uuid());
                storage
                    .put_row_locator(
                        row_id,
                        Some(&RowLocator {
                            table: table.into(),
                            origin_schema_hash: Some(schema_hash),
                        }),
                    )
                    .expect("row locator should persist");
                rows.push(RowHarness {
                    table,
                    row_id,
                    descriptor: descriptor.clone(),
                    frontier: Vec::new(),
                    all_batches: Vec::new(),
                    parents: HashMap::new(),
                    batch_states: HashMap::new(),
                    staging: Vec::new(),
                    hard_delete_batches: HashSet::new(),
                    hard_deleted: false,
                    created: None,
                    next_ts: 10,
                });
            }
        }
        Self {
            storage,
            branch: BranchName::new(BRANCH),
            prng: Prng::new(seed),
            mint,
            rows,
            seed,
            stream,
        }
    }

    fn run(mut self) {
        for op_index in 0..OPS_PER_SEED {
            let row_index = self.prng.below(self.rows.len());
            let op = pick_op(&mut self.prng);
            let desc = self.execute(op, row_index);
            let context = format!(
                "seed {seed:#x} {stream:?} op #{op_index} [{desc}] row {row} table {table}",
                seed = self.seed,
                stream = self.stream,
                row = self.rows[row_index].row_id,
                table = self.rows[row_index].table,
            );
            let frontier = self.assert_stored_matches_rebuild(row_index, &context);
            self.rows[row_index].frontier = frontier;
        }
    }

    fn execute(&mut self, op: OpKind, row_index: usize) -> String {
        match op {
            OpKind::SerialAppend => self.serial_append(row_index),
            OpKind::Fork => self.fork(row_index),
            OpKind::MergeCommit => self.merge_commit(row_index),
            OpKind::SupersetMerge => self.superset_merge(row_index),
            OpKind::StepParentAppend => self.step_parent_append(row_index),
            OpKind::StagingApply => self.staging_apply(row_index),
            OpKind::StagingPublish => self.staging_publish(row_index),
            OpKind::Reject => self.reject(row_index),
            OpKind::TierBumpReapply => self.tier_bump_reapply(row_index),
            OpKind::TierBumpPatch => self.tier_bump_patch(row_index),
            OpKind::StrippedCopy => self.stripped_copy(row_index),
            OpKind::SoftDelete => self.delete(row_index, DeleteKind::Soft),
            OpKind::HardDelete => self.delete(row_index, DeleteKind::Hard),
        }
    }

    /// Extend one current tip (the common case; with a multi-tip frontier this
    /// keeps the other branches concurrent instead of merging them).
    fn serial_append(&mut self, row_index: usize) -> String {
        let parents = {
            let frontier = &self.rows[row_index].frontier;
            if frontier.is_empty() {
                Vec::new()
            } else {
                vec![frontier[self.prng.below(frontier.len())]]
            }
        };
        let tier = self.random_tier();
        self.apply_new_batch(row_index, parents, RowState::VisibleDirect, tier, None);
        "serial-append".to_string()
    }

    /// New batch whose parent is some older batch — creates concurrent tips.
    fn fork(&mut self, row_index: usize) -> String {
        let row = &self.rows[row_index];
        if row.hard_deleted || row.all_batches.is_empty() {
            return format!("{} (fork fallback)", self.serial_append(row_index));
        }
        let parent = row.all_batches[self.prng.below(row.all_batches.len())];
        let tier = self.random_tier();
        self.apply_new_batch(row_index, vec![parent], RowState::VisibleDirect, tier, None);
        "fork".to_string()
    }

    /// New batch naming every current tip as parent.
    fn merge_commit(&mut self, row_index: usize) -> String {
        let parents = self.rows[row_index].frontier.clone();
        if parents.len() < 2 {
            return format!("{} (merge fallback)", self.serial_append(row_index));
        }
        let tier = self.random_tier();
        self.apply_new_batch(row_index, parents, RowState::VisibleDirect, tier, None);
        "merge-commit".to_string()
    }

    /// New batch naming every current tip and one more version of the row: a device that
    /// holds, as a tip, a version this store has already seen written over.
    fn superset_merge(&mut self, row_index: usize) -> String {
        let row = &self.rows[row_index];
        let others: Vec<BatchId> = row
            .all_batches
            .iter()
            .copied()
            .filter(|batch_id| !row.frontier.contains(batch_id))
            .collect();
        if row.hard_deleted || row.frontier.is_empty() || others.is_empty() {
            return format!("{} (superset fallback)", self.serial_append(row_index));
        }
        let mut parents = row.frontier.clone();
        let extra = others[self.prng.below(others.len())];
        parents.insert(self.prng.below(parents.len() + 1), extra);
        let tier = self.random_tier();
        self.apply_new_batch(row_index, parents, RowState::VisibleDirect, tier, None);
        "superset-merge".to_string()
    }

    /// New batch over one tip that also names what that tip was written over: the write
    /// of a device that held both as tips.
    fn step_parent_append(&mut self, row_index: usize) -> String {
        let row = &self.rows[row_index];
        if row.hard_deleted || row.frontier.is_empty() {
            return format!("{} (step-parent fallback)", self.serial_append(row_index));
        }
        let tip = row.frontier[self.prng.below(row.frontier.len())];
        let grandparents = row.parents.get(&tip).cloned().unwrap_or_default();
        if grandparents.is_empty() {
            return format!("{} (step-parent fallback)", self.serial_append(row_index));
        }
        let mut parents = vec![tip];
        // One time in three the second parent is any version at all: a version the tip
        // does not descend from brings its own ancestry into the write, and with it,
        // possibly, a later version all the tips have in common.
        if self.prng.below(3) == 0 {
            let any = self.rows[row_index].all_batches
                [self.prng.below(self.rows[row_index].all_batches.len())];
            if any != tip {
                parents.push(any);
            }
            let tier = self.random_tier();
            self.apply_new_batch(row_index, parents, RowState::VisibleDirect, tier, None);
            return "tip-and-any-append".to_string();
        }
        for grandparent in grandparents {
            if parents.len() == 1 || self.prng.coin() {
                parents.insert(self.prng.below(parents.len() + 1), grandparent);
            }
        }
        let tier = self.random_tier();
        self.apply_new_batch(row_index, parents, RowState::VisibleDirect, tier, None);
        "step-parent-append".to_string()
    }

    fn staging_apply(&mut self, row_index: usize) -> String {
        let parents = {
            let frontier = &self.rows[row_index].frontier;
            if frontier.is_empty() {
                Vec::new()
            } else {
                vec![frontier[self.prng.below(frontier.len())]]
            }
        };
        self.apply_new_batch(row_index, parents, RowState::StagingPending, None, None);
        "staging-apply".to_string()
    }

    /// Mirrors `runtime_core/writes.rs`: publish a staged batch to
    /// `VisibleDirect` after local durability, no tier change.
    fn staging_publish(&mut self, row_index: usize) -> String {
        if self.rows[row_index].staging.is_empty() {
            return format!("{} (publish fallback)", self.serial_append(row_index));
        }
        let pick = self.prng.below(self.rows[row_index].staging.len());
        let batch_id = self.rows[row_index].staging.remove(pick);
        self.patch(row_index, batch_id, Some(RowState::VisibleDirect), None);
        self.rows[row_index]
            .batch_states
            .insert(batch_id, RowState::VisibleDirect);
        "staging-publish".to_string()
    }

    fn reject(&mut self, row_index: usize) -> String {
        if self.rows[row_index].hard_deleted || self.rows[row_index].all_batches.is_empty() {
            return format!("{} (reject fallback)", self.serial_append(row_index));
        }
        let pick = self.prng.below(self.rows[row_index].all_batches.len());
        let batch_id = self.rows[row_index].all_batches[pick];
        self.patch(row_index, batch_id, Some(RowState::Rejected), None);
        let row = &mut self.rows[row_index];
        row.batch_states.insert(batch_id, RowState::Rejected);
        row.staging.retain(|staged| *staged != batch_id);
        "reject".to_string()
    }

    /// Mirrors the sync inbox (`BatchFate::AcceptedTransaction`): re-apply an
    /// existing visible row through `apply_row_batch` with a raised confirmed
    /// tier via `accepted_transaction_output`.
    fn tier_bump_reapply(&mut self, row_index: usize) -> String {
        if self.stream.tierless {
            return format!("{} (tierless)", self.serial_append(row_index));
        }
        let candidates: Vec<BatchId> = self.rows[row_index]
            .all_batches
            .iter()
            .copied()
            .filter(|batch_id| {
                let row = &self.rows[row_index];
                !row.hard_delete_batches.contains(batch_id)
                    && row
                        .batch_states
                        .get(batch_id)
                        .is_some_and(|state| state.is_visible())
            })
            .collect();
        if candidates.is_empty() {
            return format!("{} (tier-bump fallback)", self.serial_append(row_index));
        }
        let batch_id = candidates[self.prng.below(candidates.len())];
        let row_id = self.rows[row_index].row_id;
        let table = self.rows[row_index].table;
        let existing = self
            .storage
            .load_history_row_batch(table, BRANCH, row_id, batch_id)
            .expect("history row lookup should succeed")
            .expect("picked batch should exist in history");
        let tier = self.raised_tier(existing.confirmed_tier);
        let reapplied = existing.accepted_transaction_output(tier);
        apply_row_batch(&mut self.storage, row_id, &self.branch, reapplied, &[]).unwrap_or_else(
            |err| panic!("seed {:#x}: tier-bump re-apply failed: {err:?}", self.seed),
        );
        self.rows[row_index]
            .batch_states
            .insert(batch_id, RowState::VisibleTransactional);
        "tier-bump-reapply".to_string()
    }

    /// Mirrors a query-scope delivery of a batch this store already holds: the sender
    /// clears the parents (`sync_logic::scope_delivery_row`), reads the row from a visible
    /// region that stores no metadata, and the inbox clears the tier. Half the time the
    /// content differs as well, as it does when the sender's row has several tips and what
    /// it delivers is their merge under the newest tip's id.
    ///
    /// Beyond the entry matching its rebuild, the model here is that a copy says nothing
    /// about where the batch stands: its parents, its metadata and the row's tips stay.
    fn stripped_copy(&mut self, row_index: usize) -> String {
        let candidates: Vec<BatchId> = self.rows[row_index]
            .all_batches
            .iter()
            .copied()
            .filter(|batch_id| {
                let row = &self.rows[row_index];
                !row.hard_delete_batches.contains(batch_id)
                    && row
                        .batch_states
                        .get(batch_id)
                        .is_some_and(|state| state.is_visible())
            })
            .collect();
        if candidates.is_empty() {
            return format!("{} (stripped-copy fallback)", self.serial_append(row_index));
        }
        let batch_id = candidates[self.prng.below(candidates.len())];
        let row_id = self.rows[row_index].row_id;
        let table = self.rows[row_index].table;
        let existing = self
            .storage
            .load_history_row_batch(table, BRANCH, row_id, batch_id)
            .expect("history row lookup should succeed")
            .expect("picked batch should exist in history");
        let tips_before = self
            .storage
            .load_visible_region_entry(table, BRANCH, row_id)
            .expect("visible entry lookup should succeed")
            .map(|entry| entry.branch_frontier);

        let mut copy = existing.clone();
        copy.parents.clear();
        copy.metadata = Default::default();
        copy.confirmed_tier = None;
        let other_content = existing.delete_kind.is_none() && self.prng.coin();
        if other_content {
            let values = random_values(&mut self.prng);
            copy.data = encode_row(&self.rows[row_index].descriptor, &values)
                .expect("row values should encode")
                .into();
        }
        apply_row_batch(&mut self.storage, row_id, &self.branch, copy.clone(), &[])
            .unwrap_or_else(|err| panic!("seed {:#x}: stripped copy failed: {err:?}", self.seed));

        let after = self
            .storage
            .load_history_row_batch(table, BRANCH, row_id, batch_id)
            .expect("history row lookup should succeed")
            .expect("the batch is still stored");
        assert_eq!(
            after.parents, existing.parents,
            "seed {:#x}: a stripped copy of {batch_id:?} changed what the batch descends from",
            self.seed
        );
        assert_eq!(
            after.metadata, existing.metadata,
            "seed {:#x}: a stripped copy of {batch_id:?} dropped the batch's metadata",
            self.seed
        );
        assert_eq!(
            after.data, copy.data,
            "seed {:#x}: the copy's content was not stored",
            self.seed
        );
        let tips_after = self
            .storage
            .load_visible_region_entry(table, BRANCH, row_id)
            .expect("visible entry lookup should succeed")
            .map(|entry| entry.branch_frontier);
        assert_eq!(
            tips_after, tips_before,
            "seed {:#x}: a stripped copy of {batch_id:?} changed the row's tips",
            self.seed
        );
        if other_content {
            "stripped-copy-other-content".to_string()
        } else {
            "stripped-copy".to_string()
        }
    }

    /// Direct `patch_row_batch_state(state: None, confirmed_tier: Some(..))` —
    /// no production caller today, but the API allows it and Fix B will touch
    /// it. The patch maxes tiers internally, so any tier is a valid input.
    fn tier_bump_patch(&mut self, row_index: usize) -> String {
        if self.stream.tierless {
            return format!("{} (tierless)", self.serial_append(row_index));
        }
        let candidates: Vec<BatchId> = self.rows[row_index]
            .all_batches
            .iter()
            .copied()
            .filter(|batch_id| !self.rows[row_index].hard_delete_batches.contains(batch_id))
            .collect();
        if candidates.is_empty() {
            return format!("{} (tier-patch fallback)", self.serial_append(row_index));
        }
        let batch_id = candidates[self.prng.below(candidates.len())];
        let tier = TIERS[self.prng.below(TIERS.len())];
        self.patch(row_index, batch_id, None, Some(tier));
        "tier-bump-patch".to_string()
    }

    /// Deletes name every current tip so the delete lands as the sole frontier
    /// tip; hard deletes additionally carry no confirmed tier (see the module
    /// docs on hard deletes and concurrent merges).
    fn delete(&mut self, row_index: usize, kind: DeleteKind) -> String {
        let parents = self.rows[row_index].frontier.clone();
        let tier = match kind {
            DeleteKind::Soft => self.random_tier(),
            DeleteKind::Hard => None,
        };
        let batch_id = self.apply_new_batch(
            row_index,
            parents,
            RowState::VisibleDirect,
            tier,
            Some(kind),
        );
        if matches!(kind, DeleteKind::Hard) {
            let row = &mut self.rows[row_index];
            row.hard_delete_batches.insert(batch_id);
            row.hard_deleted = true;
            row.staging.clear();
        }
        match kind {
            DeleteKind::Soft => "soft-delete".to_string(),
            DeleteKind::Hard => "hard-delete".to_string(),
        }
    }

    fn apply_new_batch(
        &mut self,
        row_index: usize,
        parents: Vec<BatchId>,
        state: RowState,
        confirmed_tier: Option<DurabilityTier>,
        delete: Option<DeleteKind>,
    ) -> BatchId {
        let batch_id = BatchId::from_uuid(self.mint.next_uuid());
        let author = AUTHORS[self.prng.below(AUTHORS.len())];
        let values = random_values(&mut self.prng);
        // A writer whose clock is behind stamps its write earlier than the version it
        // writes over. Never the row's first version: its stamp names the lineage.
        let behind = if self.stream.skewed
            && self.rows[row_index].created.is_some()
            && self.prng.below(4) == 0
        {
            5 + 10 * self.prng.below(3) as u64
        } else {
            0
        };
        let row = &mut self.rows[row_index];
        let ts = row.next_ts.saturating_sub(behind);
        row.next_ts += 10;
        let provenance = match &row.created {
            None => RowProvenance::for_insert(author.to_string(), ts),
            Some((created_by, created_at)) => RowProvenance {
                created_by: created_by.clone(),
                created_at: *created_at,
                updated_by: author.to_string(),
                updated_at: ts,
            },
        };
        if row.created.is_none() {
            row.created = Some((author.to_string(), ts));
        }
        let data = match delete {
            Some(DeleteKind::Hard) => Vec::new(),
            _ => encode_row(&row.descriptor, &values).expect("row values should encode"),
        };
        let metadata = match delete {
            None => HashMap::new(),
            Some(DeleteKind::Soft) => {
                HashMap::from([(MetadataKey::Delete.to_string(), "soft".to_string())])
            }
            Some(DeleteKind::Hard) => {
                HashMap::from([(MetadataKey::Delete.to_string(), "hard".to_string())])
            }
        };
        row.parents.insert(batch_id, parents.clone());
        let batch = StoredRowBatch::new_with_batch_id(
            batch_id,
            row.row_id,
            BRANCH,
            parents,
            data,
            provenance,
            metadata,
            state,
            confirmed_tier,
        );
        let row_id = row.row_id;
        row.all_batches.push(batch_id);
        row.batch_states.insert(batch_id, state);
        if matches!(state, RowState::StagingPending) {
            row.staging.push(batch_id);
        }
        apply_row_batch(&mut self.storage, row_id, &self.branch, batch, &[])
            .unwrap_or_else(|err| panic!("seed {:#x}: apply_row_batch failed: {err:?}", self.seed));
        batch_id
    }

    fn patch(
        &mut self,
        row_index: usize,
        batch_id: BatchId,
        state: Option<RowState>,
        confirmed_tier: Option<DurabilityTier>,
    ) {
        let row_id = self.rows[row_index].row_id;
        patch_row_batch_state(
            &mut self.storage,
            row_id,
            &self.branch,
            batch_id,
            state,
            confirmed_tier,
        )
        .unwrap_or_else(|err| {
            panic!(
                "seed {:#x} {:?}: patch_row_batch_state({batch_id}) failed: {err:?}",
                self.seed, self.stream
            )
        });
    }

    fn random_tier(&mut self) -> Option<DurabilityTier> {
        if self.stream.tierless {
            return None;
        }
        match self.prng.below(TIERS.len() + 1) {
            0 => None,
            other => Some(TIERS[other - 1]),
        }
    }

    fn raised_tier(&mut self, current: Option<DurabilityTier>) -> DurabilityTier {
        let eligible: Vec<DurabilityTier> = TIERS
            .iter()
            .copied()
            .filter(|tier| current.is_none_or(|existing| *tier >= existing))
            .collect();
        eligible[self.prng.below(eligible.len())]
    }

    /// The differential oracle: rebuild the visible entry from the full branch
    /// history and require the stored entry (struct and bytes) to match.
    /// Returns the rebuilt frontier for the generator's bookkeeping.
    fn assert_stored_matches_rebuild(&mut self, row_index: usize, context: &str) -> Vec<BatchId> {
        let row = &self.rows[row_index];
        let table = row.table;
        let row_id = row.row_id;
        let descriptor = &row.descriptor;

        let history = self
            .storage
            .scan_history_region(table, BRANCH, HistoryScan::Row { row_id })
            .unwrap_or_else(|err| panic!("{context}: scan history failed: {err:?}"));
        let expected = VisibleRowEntry::rebuild_with_descriptor(descriptor, &history)
            .unwrap_or_else(|err| panic!("{context}: full-history rebuild failed: {err}"));
        let stored = self
            .storage
            .load_visible_region_entry(table, BRANCH, row_id)
            .unwrap_or_else(|err| panic!("{context}: load stored visible entry failed: {err:?}"));
        let stored_bytes = self
            .storage
            .load_visible_region_row_bytes(table, BRANCH, row_id)
            .unwrap_or_else(|err| panic!("{context}: load stored visible bytes failed: {err:?}"));

        match (&expected, &stored, &stored_bytes) {
            (None, None, None) => Vec::new(),
            (Some(expected), Some(stored), Some(stored_bytes)) => {
                let expected_bytes = encode_flat_visible_row_entry(descriptor, expected)
                    .unwrap_or_else(|err| panic!("{context}: encode rebuilt entry failed: {err}"));
                // The flat visible codec intentionally drops
                // `current_row.parents` / `.metadata` (both derivable from the
                // history region), so a backend legitimately serves either the
                // exact entry it was handed (memory keeps the struct) or the
                // codec-normalized decoding of the stored bytes
                // (rocksdb/sqlite). Accept exactly those two shapes — anything
                // else is a divergence.
                let expected_roundtrip =
                    decode_flat_visible_row_entry(descriptor, row_id, BRANCH, &expected_bytes)
                        .unwrap_or_else(|err| {
                            panic!("{context}: decode rebuilt entry failed: {err}")
                        });
                assert!(
                    *stored == *expected || *stored == expected_roundtrip,
                    "{context}: stored visible entry diverges from full-history rebuild\n\
                     stored:  {stored:?}\n\
                     rebuilt: {expected:?}",
                );
                assert_eq!(
                    stored_bytes, &expected_bytes,
                    "{context}: stored visible bytes diverge from re-encoded full-history rebuild",
                );
                expected.branch_frontier.clone()
            }
            (expected, stored, stored_bytes) => panic!(
                "{context}: stored visible entry presence diverges from rebuild: \
                 rebuilt={:?} stored={:?} stored_bytes_present={}",
                expected.as_ref().map(|entry| entry.current_row.batch_id()),
                stored.as_ref().map(|entry| entry.current_row.batch_id()),
                stored_bytes.is_some(),
            ),
        }
    }
}
