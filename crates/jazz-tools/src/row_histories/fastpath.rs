//! History fast paths for [`apply_row_batch`](super::apply_row_batch) and
//! [`patch_row_batch_state`](super::patch_row_batch_state).
//!
//! Three constructions of the next [`VisibleRowEntry`] that bypass
//! `load_branch_history` / full `visible_entry_from_history_rows` rebuilds — two of
//! them O(1), the third reading the row's tips and the one version they have in
//! common:
//!
//! - [`try_serial_fastpath_entry`] — a visible batch whose parent set equals
//!   the entire previous branch frontier dominates every old tip: after
//!   insertion the frontier is exactly `{row}` and the unfiltered visible
//!   preview is the row itself. Serves both fresh serial appends and
//!   invisible→visible flips of an existing batch (staging publishes), which
//!   are the same domination event because the previous entry never saw the
//!   invisible version.
//! - [`try_in_place_tip_update_entry`] — the SAME batch id with only
//!   `state`/`confirmed_tier` changed while it is the sole frontier tip
//!   (tier confirmations): `current_row` swaps in place, the frontier is
//!   unchanged.
//! - [`try_forked_fastpath_entry`] — a write to a row with more than one tip,
//!   which the serial path declines: over every tip, or over one of them. The
//!   entry is merged from the tips, read by their ids, over the version the
//!   previous entry recorded as common to them.
//!
//! Visible→non-visible flips (`→ Rejected`/`→ Superseded`) NEVER get a fast
//! path: removing a batch from the visible set can expose a previously
//! hidden ancestor as the new winner, which no O(1) entry update can
//! compute. That routing decision lives at the call sites in `mutations.rs`.
//!
//! The bar is byte-exactness: the constructed entry must equal the full
//! rebuild bit for bit (enforced after every op by the randomized differential
//! oracle in `storage::conformance_differential`). Every eligibility guard
//! below exists to keep that provable from the entry and the tips; any miss
//! falls back to the full path, which is always correct.
//!
//! What every construction proves is a step: IF the previous entry is the one
//! the history builds, so is the next. None of them looks under the entry, so
//! none repairs an entry that is not. Entries that are not are known:
//!
//! - [`snapshot_dominates_frontier`] asks whether the tip it replaces has
//!   parents, and a store that keeps entries in the flat visible encoding reads
//!   every tip back without them — a delivered snapshot then takes the place of
//!   a tip the history keeps beside it. A rebuild puts that tip back; a write
//!   taken by the serial path leaves it out;
//! - a delete restored after its rejection is written as a one-tip entry
//!   without a look at the history;
//! - the cleanup of a batch this store itself rejected rebuilds the entry from
//!   the versions stored under one schema version;
//! - the offline graft (`storage::graft`) adds versions to a history and leaves
//!   the entry built from the history without them; it takes the recorded merge
//!   base off first.
//!
//! The first two leave an entry with one tip and the other two leave it without
//! a merge base, so [`try_forked_fastpath_entry`] takes none of them. The serial
//! path does, and the in-place path a one-tip entry, as they always have: they
//! trust an entry's tips, not what the tips descend from.
//!
//! Why the per-tier carry-forward is subtle: the rebuild computes each tier
//! preview over the *tier-filtered* row set. Since v13-2 the tier frontier
//! uses causal domination through the full history DAG
//! (`resolution::build_computed_visible_preview`): a tier-satisfying version
//! whose lineage a newer tier-satisfying version builds on is superseded at
//! that tier even when the chain between them is not tier-confirmed. That
//! makes a row which dominates the whole visible set provably dominate every
//! tier set it enters, so tier-ENTERING transitions are O(1)
//! (`carried_tier_pointer` / [`in_place_tier_pointer`], the B3 wall fix).
//! Tier-LEAVING transitions can still expose rows the entry never tracked
//! and always decline to the full rebuild.

use std::sync::OnceLock;
use std::sync::atomic::AtomicU64;

use smallvec::SmallVec;

use crate::sync_manager::DurabilityTier;

use std::collections::HashMap;

use crate::query_manager::types::RowDescriptor;

use super::codecs::tier_satisfies;
use super::resolution::{assign_winner_ordinals, merge_frontier_preview};
use super::types::{BatchId, MergeBase, StoredRowBatch, VisibleRowEntry};

/// Applies that constructed the visible entry without a history rebuild.
/// Population: previous entry present, incoming row visible — covering fresh
/// serial appends, publish-shaped re-applies (invisible stored version
/// flipped visible) and in-place tier confirmations.
pub static HISTORY_FASTPATH_HITS: AtomicU64 = AtomicU64::new(0);
/// Applies in the targeted population (previous entry present, incoming row
/// visible) that still took the full-history path.
pub static HISTORY_FASTPATH_FALLBACKS: AtomicU64 = AtomicU64::new(0);

/// Patches (`patch_row_batch_state`) that constructed the visible entry
/// without a history rebuild. Population: previous entry present and the
/// PATCHED row visible — i.e. staging publishes and tier bumps. Transitions
/// that land outside the visible set (`→ Rejected` / `→ Superseded`) are
/// routed to the full path by design and are NOT part of this population, so
/// fallbacks measure real misses.
pub static PATCH_FASTPATH_HITS: AtomicU64 = AtomicU64::new(0);
/// Patches in the same population that still took the full-history rebuild.
pub static PATCH_FASTPATH_FALLBACKS: AtomicU64 = AtomicU64::new(0);

/// Applies to a row with more than one tip that constructed the visible entry from its
/// tips and their recorded common version ([`try_forked_fastpath_entry`]).
/// Counted in [`HISTORY_FASTPATH_HITS`] as well.
pub static FORKED_FASTPATH_HITS: AtomicU64 = AtomicU64::new(0);

#[cfg(test)]
thread_local! {
    /// [`FORKED_FASTPATH_HITS`] as this thread alone raised it, by what the entry was built
    /// from: `[a write over every tip, a write over one tip of several]`. A test that
    /// needs to know each was exercised counts here — the process-wide counter is raised
    /// by every test running at the same moment and does not tell the two apart.
    pub(crate) static FORKED_FASTPATH_ARMS_ON_THREAD: std::cell::Cell<[u64; 2]> =
        const { std::cell::Cell::new([0, 0]) };
}

/// Hot-path scan tripwire: tier-gated query reads
/// (`load_visible_region_row_for_tier`) that fell into the full
/// `scan_history_region` because the current row's batch-fate tier does not
/// satisfy the query's required tier.
///
/// Design rule (history-fastpaths §6): serving a query must not walk row
/// history. This site could not be rewritten onto the `VisibleRowEntry`
/// sidecar because the sidecar's per-tier pointers are computed from STORED
/// `confirmed_tier` values while this read overlays authoritative batch
/// fates — and direct-write rows are stored with `confirmed_tier: None`
/// everywhere (client publish, server inbox, server fate application), so
/// the two sources systematically disagree. See the divergence fixture
/// `tier_read_batch_fate_overlay_diverges_from_visible_entry_sidecar` in
/// `storage/memory.rs`. Until the sidecar is made fate-aware, this counter
/// measures how often production tier reads still pay O(history).
pub static QUERY_TIER_READ_HISTORY_SCANS: AtomicU64 = AtomicU64::new(0);

/// Hot-path scan tripwire: provenance lookups during query serving
/// (`current_row_provenance`, `query_manager/server_queries.rs`) that missed
/// the visible-entry point read and fell back to a full
/// `scan_history_row_batches`. Expected to fire only for legacy rows written
/// before visible entries existed (the lazy backfill in
/// `load_previous_visible_entry` heals them on the next write).
pub static QUERY_PROVENANCE_HISTORY_SCANS: AtomicU64 = AtomicU64::new(0);

/// Kill switch: `JAZZ_HISTORY_FASTPATH=0` (or `false`) forces the full path.
/// Default is on. The env var is read once; tests use
/// [`force_history_fastpath`] instead to avoid env races.
pub fn history_fastpath_enabled() -> bool {
    #[cfg(any(test, feature = "test"))]
    if let Some(forced) = test_override::forced() {
        return forced;
    }

    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("JAZZ_HISTORY_FASTPATH").as_deref(),
            Ok("0") | Ok("false")
        )
    })
}

#[cfg(any(test, feature = "test"))]
mod test_override {
    use std::sync::atomic::{AtomicU8, Ordering};
    use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};

    const UNSET: u8 = 0;
    const FORCED_ON: u8 = 1;
    const FORCED_OFF: u8 = 2;

    static STATE: AtomicU8 = AtomicU8::new(UNSET);

    fn lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    /// Holds the fast path in a forced mode for the guard's lifetime.
    ///
    /// The embedded mutex guard serialises every test that forces a mode (or
    /// asserts on the global hit counter), so parallel tests cannot observe
    /// each other's override. Dropping restores the unforced default.
    pub struct HistoryFastpathMode {
        _serialised: MutexGuard<'static, ()>,
    }

    impl Drop for HistoryFastpathMode {
        fn drop(&mut self) {
            STATE.store(UNSET, Ordering::SeqCst);
        }
    }

    /// Force the fast path on or off for the returned guard's lifetime.
    pub fn force_history_fastpath(enabled: bool) -> HistoryFastpathMode {
        let guard = lock().lock().unwrap_or_else(PoisonError::into_inner);
        STATE.store(
            if enabled { FORCED_ON } else { FORCED_OFF },
            Ordering::SeqCst,
        );
        HistoryFastpathMode { _serialised: guard }
    }

    pub(super) fn forced() -> Option<bool> {
        match STATE.load(Ordering::SeqCst) {
            FORCED_ON => Some(true),
            FORCED_OFF => Some(false),
            _ => None,
        }
    }
}

#[cfg(any(test, feature = "test"))]
pub use test_override::{HistoryFastpathMode, force_history_fastpath};

/// Attempt the O(1) serial-write construction of the next [`VisibleRowEntry`].
///
/// Returns `None` when any eligibility guard misses; the caller then takes the
/// full-history rebuild.
///
/// Two caller shapes are admitted — they differ only in the caller's
/// obligations, never in the math:
///
/// - a NEW visible row not yet present in history (serial append). Caller
///   obligations (checked upstream in `apply_row_batch_with_context`): the
///   row is not a known-new object, its batch id is absent from history, and
///   parent existence was validated.
/// - an EXISTING history row whose stored version is NOT visible, replaced by
///   a visible version (staging publish via `patch_row_batch_state`, or a
///   `DurableDirect`/`AcceptedTransaction` fate re-apply over a staged row).
///   The previous entry was computed over visible rows only, so it is
///   oblivious to the invisible stored version — flipping it visible is the
///   same domination event as inserting a fresh visible row. A visible
///   descendant already naming this batch as parent cannot slip through: it
///   would have to postdate this batch (parents are existence-validated at
///   apply time), while the frontier-coverage guard forces every frontier tip
///   to be one of this batch's own parents, all of which predate it.
pub(super) fn try_serial_fastpath_entry(
    previous_entry: Option<&VisibleRowEntry>,
    row: &StoredRowBatch,
) -> Option<VisibleRowEntry> {
    if !history_fastpath_enabled() {
        return None;
    }
    // Only VisibleDirect | VisibleTransactional. A StagingPending batch
    // contributes nothing to the frontier and must never become `current_row`
    // through a shortcut — that would publish a staged batch. (Staging rows
    // also keep `supersede_older_staging_rows_for_batch` on the full path by
    // construction.) Deletes interact with the delete-winner overlay of the
    // preview and stay on the full path.
    if !row.state.is_visible() || row.delete_kind.is_some() {
        return None;
    }
    let previous = previous_entry?;
    if previous.current_row.branch != row.branch {
        return None;
    }
    // Domination: the parent SET must equal the previous frontier SET
    // (order-insensitive, duplicates rejected). Not `len() == 1` — set
    // equality also admits an explicit merge-commit naming every current tip,
    // after which the frontier is trivially `{row}`.
    if !parents_cover_frontier_exactly(&row.parents, &previous.branch_frontier)
        && !snapshot_dominates_frontier(row, previous)
    {
        return None;
    }
    // Never-forked precondition: a historically forked-then-linearised row can
    // still carry a live merged tier preview (populated pool/ordinals) whose
    // O(1) maintenance across arbitrary interleavings is not provable.
    if !previous.winner_batch_pool.is_empty()
        || previous.current_winner_ordinals.is_some()
        || previous.worker_winner_ordinals.is_some()
        || previous.edge_winner_ordinals.is_some()
        || previous.global_winner_ordinals.is_some()
        || records_more_than_a_merge_base(previous)
    {
        return None;
    }

    let old_tip = &previous.current_row;
    let worker_batch_id = carried_tier_pointer(
        row,
        old_tip,
        DurabilityTier::Local,
        previous.worker_batch_id,
    )?;
    let edge_batch_id = carried_tier_pointer(
        row,
        old_tip,
        DurabilityTier::EdgeServer,
        previous.edge_batch_id,
    )?;
    let global_batch_id = carried_tier_pointer(
        row,
        old_tip,
        DurabilityTier::GlobalServer,
        previous.global_batch_id,
    )?;

    Some(VisibleRowEntry {
        current_row: row.clone(),
        branch_frontier: vec![row.batch_id()],
        worker_batch_id,
        edge_batch_id,
        global_batch_id,
        winner_batch_pool: Vec::new(),
        current_winner_ordinals: None,
        worker_winner_ordinals: None,
        edge_winner_ordinals: None,
        global_winner_ordinals: None,
        merge_artifacts: None,
    })
}

/// Whether the entry's `merge_artifacts` say anything besides which version its tips
/// descend from.
///
/// A rebuild records that version ([`MergeBase`]) for every entry with more than one tip,
/// and it is no evidence of a live merged preview: an entry whose newest tip won every
/// column has an empty pool and no ordinals, and a write naming exactly its tips has always
/// been taken here. Bytes this build cannot read are evidence of something it does not
/// know, and decline.
fn records_more_than_a_merge_base(previous: &VisibleRowEntry) -> bool {
    previous
        .merge_artifacts
        .as_deref()
        .is_some_and(|bytes| MergeBase::decode(bytes).is_none())
}

/// The second admission arm: a delivered snapshot dominating a frontier that is itself one
/// delivered snapshot.
///
/// `parents_cover_frontier_exactly` refuses empty parents outright, on the reading that a
/// parentless row over an existing entry is a concurrent root insert. That reading is the
/// overloaded meaning `elided_snapshot_dominator` disambiguates: a visible row with
/// `parents.is_empty() && created_at != updated_at` provably is not a creation, it is a batch
/// whose ancestry the sender stripped on delivery.
///
/// Without this arm the frontier rule fixes correctness and leaves the cost defect exactly
/// where it was: every delivered arrival still misses the fast path, still loads the whole
/// branch history, and still pays O(depth). Measured on the depth-flatness harness before this
/// arm existed: 377,750 reads at depth 500 against 4,127,750 at depth 8000, and 6.36GB of
/// allocations — for a row whose frontier had already collapsed to a single tip.
///
/// The admission is deliberately narrow. The incoming row must be an elided snapshot; the
/// previous frontier must be exactly one batch and that batch must be the entry's own current
/// row (so its parents are known here without a history read); that tip must itself be
/// parentless, so the new rule really does supersede it rather than the ancestry doing it; and
/// the incoming row must be strictly newer under `(updated_at, batch_id)`, the same total order
/// the dominator uses. Under those conditions the full rebuild provably yields
/// `frontier == [row]` — a one-element frontier short-circuits the merge and returns the tip
/// itself — which is what this path then writes.
fn snapshot_dominates_frontier(row: &StoredRowBatch, previous: &VisibleRowEntry) -> bool {
    let old_tip = &previous.current_row;
    // The whole visible set must be reachable from the entry alone, or this cannot decide
    // without a history read. A frontier of width one that IS the entry's current row gives
    // exactly that: any visible row carrying parents would force a non-parentless tip through
    // the finite DAG, so a parentless sole tip proves every visible row is parentless.
    if previous.branch_frontier.len() != 1
        || previous.branch_frontier[0] != old_tip.batch_id()
        || !old_tip.parents.is_empty()
    {
        return false;
    }
    // Decided by the SAME helpers the full rebuild uses, rather than a second copy of the rule
    // that could drift away from it.
    let dominator = super::resolution::elided_snapshot_dominator([row, old_tip].into_iter());
    dominator.is_some_and(|winner| winner.batch_id == row.batch_id())
        && super::resolution::superseded_by_snapshot(old_tip, dominator)
}

/// Attempt the construction of the next [`VisibleRowEntry`] of a row that has more than one
/// tip, from its tips and the version they have in common. `load` reads one version of the
/// row by its id.
///
/// A row keeps more than one tip for as long as nobody writes over all of them at once,
/// and nobody may ever do: a device shown the row under its newest tip's id writes over
/// that one, and the others stay. [`try_serial_fastpath_entry`] declines every write to
/// such a row, and the full path reads the row's whole history for it — measured at
/// 215 ms a write on a `users` row with 20 000 versions and one such tip.
///
/// Two shapes are admitted. In both the incoming row is new to the history, visible, not
/// a delete, names at least one parent, and neither it nor the entry carries a tier
/// (see [`carries_no_tier`]) — which is how every directly written row is stored. In both
/// the entry has several tips and records the version they descend from
/// ([`VisibleRowEntry::recorded_merge_base`]).
///
/// - **It names every tip.** Whatever else it names is a version already written over,
///   so afterwards it is the only tip and the row readers see is the incoming row.
/// - **It names one tip, and besides it only that tip's own parents.** The other tips
///   stay. The row readers see is the merge of the new set of tips over their common
///   ancestor, and that ancestor is the one the entry recorded (`MergeBase`): the
///   incoming row descends from what the tip it replaces descended from and from nothing
///   else, so the versions common to all tips are the same set as before.
///
/// Anything else — a write naming two tips of three, a write naming no tip, a delete, a
/// row with tier state, an entry with one tip or one that recorded no base — returns
/// `None`, and the caller reads the history, which is always right.
pub(super) fn try_forked_fastpath_entry(
    user_descriptor: &RowDescriptor,
    previous_entry: Option<&VisibleRowEntry>,
    row: &StoredRowBatch,
    load: &dyn Fn(BatchId) -> Option<StoredRowBatch>,
) -> Option<VisibleRowEntry> {
    if !history_fastpath_enabled() {
        return None;
    }
    if !row.state.is_visible() || row.delete_kind.is_some() || row.parents.is_empty() {
        return None;
    }
    let previous = previous_entry?;
    if previous.current_row.branch != row.branch || !carries_no_tier(previous, row) {
        return None;
    }
    // Only an entry that has several tips and says what they descend from is taken at
    // its word: a rebuild wrote it, or this path did from one a rebuild wrote. An entry
    // with one tip can stand for a history that has more (see the module's note on
    // delivered snapshots), and a write naming that tip and other versions is where a
    // rebuild finds them again.
    let base = previous.recorded_merge_base()?;
    let frontier = &previous.branch_frontier;
    let named: SmallVec<[BatchId; 2]> = frontier
        .iter()
        .copied()
        .filter(|tip| row.parents.contains(tip))
        .collect();

    if named.len() == frontier.len() {
        // Every tip is named: one tip from here on, and nothing to merge.
        return Some(VisibleRowEntry::new(row.clone()));
    }
    let [named] = named.as_slice() else {
        return None;
    };

    // The tip the row replaces is one this store shows, like every other version the
    // entry is trusted about below. What the row names besides it must be something the
    // tip itself descends from, or the row brings ancestors of its own and the common
    // ancestor may move.
    let replaced = load(*named)?;
    if !replaced.state.is_visible()
        || row
            .parents
            .iter()
            .any(|parent| parent != named && !replaced.parents.contains(parent))
    {
        return None;
    }

    let mut tips: Vec<StoredRowBatch> = Vec::with_capacity(frontier.len());
    for tip in frontier {
        if tip == named {
            continue;
        }
        let tip = load(*tip)?;
        if !tip.state.is_visible() {
            return None;
        }
        tips.push(tip);
    }
    tips.push(row.clone());
    let ancestor = match base.ancestor {
        Some(ancestor) => {
            let ancestor = load(ancestor)?;
            if !ancestor.state.is_visible() {
                return None;
            }
            Some(ancestor)
        }
        None => None,
    };

    let mut ordered: Vec<&StoredRowBatch> = tips.iter().collect();
    ordered.sort_by_key(|tip| (tip.updated_at, tip.batch_id()));
    let preview = merge_frontier_preview(user_descriptor, &ordered, ancestor.as_ref()).ok()?;
    // A merged row carries the lowest tier among the rows it took a column from, and
    // none of them has one.
    if preview.row.confirmed_tier.is_some() {
        return None;
    }

    let mut branch_frontier: Vec<BatchId> = tips.iter().map(StoredRowBatch::batch_id).collect();
    branch_frontier.sort();
    let mut winner_batch_pool = Vec::new();
    let current_winner_ordinals = assign_winner_ordinals(
        preview.winner_batch_ids.as_deref(),
        &mut winner_batch_pool,
        &mut HashMap::new(),
    )
    .ok()?;

    Some(VisibleRowEntry {
        current_row: preview.row,
        branch_frontier,
        worker_batch_id: None,
        edge_batch_id: None,
        global_batch_id: None,
        winner_batch_pool,
        current_winner_ordinals,
        worker_winner_ordinals: None,
        edge_winner_ordinals: None,
        global_winner_ordinals: None,
        merge_artifacts: Some(base.encode()),
    })
}

/// No version of the row satisfies any tier, before the incoming row or with it.
///
/// A tier's view of a row is computed over the versions confirmed at that tier, and the
/// entry holds a pointer to it wherever it differs from the row readers see. With every
/// pointer empty and the row readers see itself unconfirmed, no tier has a view at all:
/// a tier whose view exists and equals the current row would have confirmed the current
/// row. An incoming row with no tier adds nothing to any of them.
fn carries_no_tier(previous: &VisibleRowEntry, row: &StoredRowBatch) -> bool {
    row.confirmed_tier.is_none()
        && previous.current_row.confirmed_tier.is_none()
        && previous.worker_batch_id.is_none()
        && previous.edge_batch_id.is_none()
        && previous.global_batch_id.is_none()
        && previous.worker_winner_ordinals.is_none()
        && previous.edge_winner_ordinals.is_none()
        && previous.global_winner_ordinals.is_none()
}

fn parents_cover_frontier_exactly(parents: &[BatchId], frontier: &[BatchId]) -> bool {
    // Empty parents over an existing entry would be a concurrent root insert
    // (and `frontier` of a live entry is never empty) — never a domination.
    if parents.is_empty() || parents.len() != frontier.len() {
        return false;
    }
    let mut sorted_parents: SmallVec<[BatchId; 2]> = SmallVec::from_slice(parents);
    sorted_parents.sort_unstable();
    if sorted_parents.windows(2).any(|pair| pair[0] == pair[1]) {
        return false;
    }
    let mut sorted_frontier: SmallVec<[BatchId; 2]> = SmallVec::from_slice(frontier);
    sorted_frontier.sort_unstable();
    sorted_parents == sorted_frontier
}

/// Carry one tier sidecar pointer across a domination event, or decline.
///
/// NOTE on the premise below: it is written for the frontier-coverage caller, where
/// `C.parents` equals the whole unfiltered visible frontier. The second admission arm
/// (`snapshot_dominates_frontier`) reaches this with `C.parents` EMPTY, so "every visible row
/// is a proper ancestor of C" is false there and the conclusion holds by a different argument:
/// on that arm every visible row is parentless and of one lineage, and C is the lineage's
/// newest, so C dominates the visible set — and therefore every tier subset of it — directly.
///
/// `Some(pointer)` is proven byte-equal to what the full rebuild
/// (`VisibleRowEntry::rebuild_with_descriptor` →
/// `preview_override_sidecar`) stores; `None` means "not provable from the
/// entry alone — take the full path". Let S be the tier-filtered visible row
/// set before this write, B the old current row, C the incoming row. The
/// never-forked guard already established that all sidecar ordinals are
/// `None`, which makes every stored tier preview equal to an actual history
/// row. Case analysis:
///
/// - `C` does not satisfy the tier: S is unchanged by this insert, so the
///   rebuilt candidate preview is the same one the previous rebuild compared,
///   and it can never match the new current row `C` (its metadata row predates
///   `C`). Therefore:
///   - old pointer `None` with `B` satisfying the tier ⇒ the old preview
///     matched `B` exactly ⇒ the new pointer is `Some(B)`;
///   - old pointer `None` with `B` not satisfying ⇒ S is empty ⇒ `None`;
///   - old pointer `Some(x)` ⇒ carried verbatim (the preview still is row
///     `x`). Note this deliberately refines the design sketch's "old tip
///     satisfies ⇒ point at old tip": when a pointer is already present it is
///     the rebuild's answer, the old tip is not.
/// - `C` satisfies the tier ⇒ the new tier preview is `{C}` and the pointer
///   is `None`, unconditionally (v13-2, the B3 tier-pointer wall). Proof
///   under the tier frontier's causal-domination semantics
///   (`build_computed_visible_preview`, `required_tier == Some(..)`): the
///   caller's frontier-coverage guard makes `C.parents` equal the whole
///   unfiltered visible frontier. Every visible row is an ancestor of some
///   frontier tip — follow its visible parent-naming chain upward; the
///   history DAG is finite and acyclic (parents are existence-validated at
///   apply time), so the chain terminates at an un-named row, i.e. a tip —
///   and every tip is a parent of `C`, so EVERY visible row is a proper
///   ancestor of `C`. `S ∪ {C}` therefore has the causal frontier `{C}`:
///   all of S is dominated, and nothing can dominate the brand-new `C`.
///   The rebuilt candidate is the singleton preview of the new current row
///   ⇒ pointer `None`. This holds regardless of tier holes below — the
///   pre-v13-2 one-step frontier resurfaced hole-hidden ancestors and made
///   this case undecidable from the entry, which was the wall.
fn carried_tier_pointer(
    row: &StoredRowBatch,
    old_tip: &StoredRowBatch,
    tier: DurabilityTier,
    previous_pointer: Option<BatchId>,
) -> Option<Option<BatchId>> {
    let old_tip_satisfies = tier_satisfies(old_tip.confirmed_tier, tier);
    if tier_satisfies(row.confirmed_tier, tier) {
        Some(None)
    } else if old_tip_satisfies && previous_pointer.is_none() {
        Some(Some(old_tip.batch_id()))
    } else {
        Some(previous_pointer)
    }
}

/// Attempt the O(1) in-place tip update: the SAME batch id re-applied (via
/// `accepted_transaction_output` tier confirmations) or patched (pure
/// `confirmed_tier` bump) with only `state`/`confirmed_tier` changed, while it
/// is the sole frontier tip of a never-forked entry and stays visible.
///
/// The unfiltered preview of a sole-tip entry IS the tip row, before and
/// after the flip, so `current_row` is replaced in place and the frontier is
/// unchanged (`[batch_id]`). Only the per-tier sidecar needs care — see
/// [`in_place_tier_pointer`] for the case analysis. Returns `None` when any
/// guard misses; the caller then takes the full rebuild.
pub(super) fn try_in_place_tip_update_entry(
    previous_entry: Option<&VisibleRowEntry>,
    existing_row: &StoredRowBatch,
    row: &StoredRowBatch,
) -> Option<VisibleRowEntry> {
    if !history_fastpath_enabled() {
        return None;
    }
    // Both versions must be visible. Invisible → visible is a publish
    // (domination) event served by `try_serial_fastpath_entry`; visible →
    // invisible is a removal event that can expose a previously hidden
    // ancestor as the new winner and always takes the full path.
    if !existing_row.state.is_visible() || !row.state.is_visible() {
        return None;
    }
    // Deletes interact with the delete-winner overlay of every preview.
    if row.delete_kind.is_some() {
        return None;
    }
    // Only `state`/`confirmed_tier` may differ. A content change re-runs
    // every merge the tip participates in (a tier preview that coincided
    // with the tip can stop coinciding), which is not provable from the
    // entry alone.
    if !same_row_except_state_and_tier(existing_row, row) {
        return None;
    }
    let previous = previous_entry?;
    if previous.current_row.branch != row.branch {
        return None;
    }
    let batch_id = row.batch_id();
    // The batch must be the SOLE frontier tip. `current_row` carrying this
    // batch id alone is not enough: a multi-tip frontier can coincidentally
    // normalise its merged preview onto this batch while concurrent state
    // hides behind it.
    if previous.branch_frontier.as_slice() != [batch_id]
        || previous.current_row.batch_id() != batch_id
    {
        return None;
    }
    // Never-forked precondition — same rationale as the serial fast path.
    if !previous.winner_batch_pool.is_empty()
        || previous.current_winner_ordinals.is_some()
        || previous.worker_winner_ordinals.is_some()
        || previous.edge_winner_ordinals.is_some()
        || previous.global_winner_ordinals.is_some()
        || previous.merge_artifacts.is_some()
    {
        return None;
    }

    let worker_batch_id = in_place_tier_pointer(
        existing_row,
        row,
        DurabilityTier::Local,
        previous.worker_batch_id,
    )?;
    let edge_batch_id = in_place_tier_pointer(
        existing_row,
        row,
        DurabilityTier::EdgeServer,
        previous.edge_batch_id,
    )?;
    let global_batch_id = in_place_tier_pointer(
        existing_row,
        row,
        DurabilityTier::GlobalServer,
        previous.global_batch_id,
    )?;

    Some(VisibleRowEntry {
        current_row: row.clone(),
        branch_frontier: vec![batch_id],
        worker_batch_id,
        edge_batch_id,
        global_batch_id,
        winner_batch_pool: Vec::new(),
        current_winner_ordinals: None,
        worker_winner_ordinals: None,
        edge_winner_ordinals: None,
        global_winner_ordinals: None,
        merge_artifacts: None,
    })
}

/// Byte-equality on every field except `state` and `confirmed_tier` — the
/// only fields a tier confirmation or state patch may legally change without
/// disturbing merge inputs (values, parents, provenance ordering keys,
/// delete ranking).
fn same_row_except_state_and_tier(existing: &StoredRowBatch, row: &StoredRowBatch) -> bool {
    existing.row_id == row.row_id
        && existing.batch_id == row.batch_id
        && existing.branch == row.branch
        && existing.parents == row.parents
        && existing.updated_at == row.updated_at
        && existing.created_by == row.created_by
        && existing.created_at == row.created_at
        && existing.updated_by == row.updated_by
        && existing.delete_kind == row.delete_kind
        && existing.is_deleted == row.is_deleted
        && existing.data == row.data
        && existing.metadata == row.metadata
}

/// Carry one tier sidecar pointer across an in-place tip update, or decline.
///
/// NOT the same rule as [`carried_tier_pointer`]: there the tier-filtered set
/// S gains a NEW row C on top of old tip B; here "C" and "B" are positionally
/// the same row — the sole tip T is replaced by T' (identical except
/// state/tier), so S keeps, gains, or loses THE SAME row. The guards in
/// [`try_in_place_tip_update_entry`] already established: T is the sole
/// frontier tip (the unfiltered preview equals T exactly), the entry is
/// never-forked (all sidecar ordinals `None`, so any stored tier preview
/// equals an actual history row byte-for-byte), and T/T' differ only in
/// state/tier (per-column merge inputs, winner ordering keys and delete
/// ranking are untouched). Case analysis against the rebuild
/// (`preview_override_sidecar` over `build_computed_visible_preview`):
///
/// - membership unchanged (`old_sat == new_sat`): the tier set keeps the
///   exact same rows (T ∈ S in neither or both versions, and only its
///   state/tier — no merge input — moved), so the candidate preview is
///   unchanged ⇒ carried verbatim (`None` stays `None`, `Some(x)` stays
///   `Some(x)`).
/// - T' enters the tier (`!old_sat && new_sat` — the tier-confirmation hot
///   case) ⇒ the new tier preview is `{T'}` and the pointer is `None`,
///   unconditionally (v13-2, the B3 tier-pointer wall). Proof under the
///   tier frontier's causal-domination semantics
///   (`build_computed_visible_preview`, `required_tier == Some(..)`): T is
///   the SOLE unfiltered frontier tip, so every other visible row is a
///   proper ancestor of T — follow its visible parent-naming chain upward;
///   the DAG is finite and acyclic, so the chain ends at the only un-named
///   row, T. Hence every member of S (all visible) is dominated by
///   T' ∈ S ∪ {T'}, nothing can dominate the sole tip T' itself, and the
///   causal tier frontier is exactly `{T'}` — whose singleton preview
///   matches the new current row ⇒ `None`. This holds for pointer `None`
///   (empty S) and pointer `Some(x)` alike, tier holes included: this is
///   what makes confirm-over-confirmed-chain — the alternating
///   append+confirm workload — O(1)
///   (`append_confirm_per_write_at_edge_server_is_flat_in_history_depth`).
///   Under the pre-v13-2 one-step tier frontier, `Some(x)` had to decline:
///   a hole-hidden ancestor could resurface as a concurrent tier tip whose
///   merge output depends on deep history.
/// - T' leaves the tier (`old_sat && !new_sat`): a removal event — the new
///   tier frontier may expose rows the entry never tracked ⇒ decline.
fn in_place_tier_pointer(
    existing_row: &StoredRowBatch,
    row: &StoredRowBatch,
    tier: DurabilityTier,
    previous_pointer: Option<BatchId>,
) -> Option<Option<BatchId>> {
    let old_sat = tier_satisfies(existing_row.confirmed_tier, tier);
    let new_sat = tier_satisfies(row.confirmed_tier, tier);
    match (old_sat, new_sat) {
        (true, false) => None,
        (false, true) => Some(None),
        _ => Some(previous_pointer),
    }
}
