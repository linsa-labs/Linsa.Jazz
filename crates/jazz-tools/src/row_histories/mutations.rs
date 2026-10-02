//! Storage-mutating row-history operations.
//!
//! Owns the public verbs that change row-history state and keep the derived
//! visible region consistent afterward:
//! - [`apply_row_batch`] — insert or update a history batch, then recompute and
//!   persist visibility (and supersede stale staging siblings if needed)
//! - [`patch_row_batch_state`] — flip an existing batch's state and
//!   recompute visibility
//!
//! Each mutation captures the visible row before/after the change as an
//! [`AppliedRowBatch`], from which a [`RowVisibilityChange`] is derived for
//! downstream observers (sync, indexers, query subscribers).
//!
//! Pure visibility/merge math lives in [`super::resolution`]; this module
//! only orchestrates: load → mutate → recompute → write.

use std::sync::atomic::Ordering;

use crate::object::{BranchName, ObjectId};
use crate::query_manager::types::{RowDescriptor, SharedString};
use crate::storage::{IndexMutation, PreparedRowWriteContext, RowLocator, Storage, StorageError};
use crate::sync_manager::DurabilityTier;

use super::fastpath::{
    FORKED_FASTPATH_HITS, HISTORY_FASTPATH_FALLBACKS, HISTORY_FASTPATH_HITS,
    PATCH_FASTPATH_FALLBACKS, PATCH_FASTPATH_HITS, try_forked_fastpath_entry,
    try_in_place_tip_update_entry, try_serial_fastpath_entry,
};
use super::resolution::visible_entry_from_history_rows;
use super::types::{
    ApplyRowBatchResult, BatchId, HistoryScan, RowHistoryError, RowState, RowVisibilityChange,
    StoredRowBatch, VisibleRowEntry,
};

#[derive(Debug, Clone)]
pub(super) struct AppliedRowBatch {
    row_locator: RowLocator,
    previous_visible: Option<StoredRowBatch>,
    current_visible: Option<StoredRowBatch>,
    is_new_object: bool,
    visible_changed: bool,
}

pub(crate) struct ApplyRowBatchWithContext<'a> {
    pub(crate) object_id: ObjectId,
    pub(crate) branch_name: &'a BranchName,
    pub(crate) row: StoredRowBatch,
    pub(crate) index_mutations: &'a [IndexMutation<'a>],
    pub(crate) row_locator: RowLocator,
    pub(crate) table: String,
    pub(crate) branch: SharedString,
    pub(crate) context: PreparedRowWriteContext,
    pub(crate) is_known_new_object: bool,
}

pub(super) fn row_locator_from_storage<H: Storage>(
    io: &H,
    object_id: ObjectId,
) -> Result<RowLocator, RowHistoryError> {
    io.load_row_locator(object_id)
        .map_err(RowHistoryError::StorageError)?
        .ok_or(RowHistoryError::ObjectNotFound(object_id))
}

pub(super) fn load_branch_history<H: Storage>(
    io: &H,
    table: &str,
    object_id: ObjectId,
    branch_name: &SharedString,
) -> Result<Vec<StoredRowBatch>, RowHistoryError> {
    io.scan_history_region(
        table,
        branch_name.as_str(),
        HistoryScan::Row { row_id: object_id },
    )
    .map_err(RowHistoryError::StorageError)
}

pub(super) fn rebuild_visible_entry_from_history<H: Storage>(
    io: &H,
    table: &str,
    object_id: ObjectId,
    branch_name: &SharedString,
    user_descriptor: &RowDescriptor,
) -> Result<Option<VisibleRowEntry>, RowHistoryError> {
    let history_rows = load_branch_history(io, table, object_id, branch_name)?;
    visible_entry_from_history_rows(user_descriptor, &history_rows).map_err(|err| {
        RowHistoryError::StorageError(StorageError::IoError(format!(
            "rebuild visible entry: {err}"
        )))
    })
}

pub(super) fn load_previous_visible_entry<H: Storage>(
    io: &H,
    table: &str,
    object_id: ObjectId,
    branch_name: &SharedString,
    user_descriptor: &RowDescriptor,
) -> Result<Option<VisibleRowEntry>, RowHistoryError> {
    match io.load_visible_region_entry(table, branch_name.as_str(), object_id) {
        Ok(Some(entry)) => Ok(Some(entry)),
        Ok(None) => {
            rebuild_visible_entry_from_history(io, table, object_id, branch_name, user_descriptor)
        }
        Err(_) => {
            rebuild_visible_entry_from_history(io, table, object_id, branch_name, user_descriptor)
        }
    }
}

pub(super) fn visibility_change_from_applied(
    object_id: ObjectId,
    applied: AppliedRowBatch,
) -> Option<RowVisibilityChange> {
    if !applied.visible_changed {
        return None;
    }

    let current_visible = applied.current_visible?;
    Some(RowVisibilityChange {
        object_id,
        row_locator: applied.row_locator,
        row: current_visible,
        previous_row: applied.previous_visible,
        is_new_object: applied.is_new_object,
    })
}

pub(super) fn supersede_older_staging_rows_for_batch<H: Storage>(
    io: &mut H,
    table: &str,
    object_id: ObjectId,
    branch_name: &BranchName,
    batch_id: BatchId,
) -> Result<(), RowHistoryError> {
    let branch = SharedString::from(branch_name.as_str().to_string());
    let history_rows = load_branch_history(io, table, object_id, &branch)?;
    let mut pending_rows = history_rows
        .into_iter()
        .filter(|row| row.batch_id == batch_id && matches!(row.state, RowState::StagingPending))
        .collect::<Vec<_>>();

    if pending_rows.len() <= 1 {
        return Ok(());
    }

    pending_rows.sort_by_key(|row| (row.updated_at, row.batch_id()));
    pending_rows.pop();

    for row in pending_rows {
        let _ = patch_row_batch_state(
            io,
            object_id,
            branch_name,
            row.batch_id(),
            Some(RowState::Superseded),
            None,
        )?;
    }

    Ok(())
}

pub fn apply_row_batch<H: Storage>(
    io: &mut H,
    object_id: ObjectId,
    branch_name: &BranchName,
    row: StoredRowBatch,
    index_mutations: &[IndexMutation<'_>],
) -> Result<ApplyRowBatchResult, RowHistoryError> {
    let row_locator = row_locator_from_storage(io, object_id)?;
    let table = row_locator.table.to_string();
    let branch = SharedString::from(branch_name.as_str().to_string());
    let context = crate::storage::resolve_history_row_write_context(io, &table, &row)
        .map_err(RowHistoryError::StorageError)?;
    // The locator must name the raw table the write actually lands in, or
    // every locator-directed read misses the row forever. They diverge when a
    // row arrives before the catalogue knows its origin schema: the resolve
    // ladder falls through to the any-decoding-descriptor fallback (a hash
    // whose descriptor provably decodes these bytes) while the locator still
    // carries the server-stamped origin (defect 20, production 2026-08-15).
    // Aligning the locator to the resolved hash is safe for every consumer —
    // the ladder never picks a descriptor the bytes do not decode under.
    //
    // The aligned locator rides the request, but PERSISTS only after the
    // apply succeeds: a batch that fails validation (a routine ParentNotFound
    // on out-of-order delivery) must not leave the locator pointing at a
    // generation whose write never landed — that strands every history point
    // read for batches stored without exact locators.
    // The aligned locator rides the request so the visibility change this apply
    // reports names the family the bytes went to. PERSISTING it belongs to
    // `storage::enforce_single_visible_family_after_write`, which runs where the
    // bytes land and therefore also covers the inbound sync path — that path
    // builds its locator from the incoming metadata and never comes through here
    // (defect 27).
    let context_hash = context.history_row_raw_table_id().schema_hash;
    let row_locator = if row_locator.origin_schema_hash == Some(context_hash) {
        row_locator
    } else {
        crate::storage::RowLocator {
            table: row_locator.table.clone(),
            origin_schema_hash: Some(context_hash),
        }
    };
    apply_row_batch_with_context(
        io,
        ApplyRowBatchWithContext {
            object_id,
            branch_name,
            row,
            index_mutations,
            row_locator,
            table,
            branch,
            context,
            is_known_new_object: false,
        },
    )
}

/// Apply one row batch under an already-resolved write context.
///
/// No family realignment happens here. Every visible write — this one included —
/// lands through `storage::enforce_single_visible_family_after_write`, which
/// measures which families hold `(branch, row)` at the moment the bytes land,
/// MOVES the head rather than forking it, and points both locators at it. An
/// earlier version of this fix did that work again here, on a predicted head
/// family; it was measured redundant (disarm this pass alone and no gate falls,
/// disarm both and seven do) and removed rather than kept as an unfalsifiable
/// second belt.
pub(crate) fn apply_row_batch_with_context<H: Storage>(
    io: &mut H,
    request: ApplyRowBatchWithContext<'_>,
) -> Result<ApplyRowBatchResult, RowHistoryError> {
    let ApplyRowBatchWithContext {
        object_id,
        branch_name,
        mut row,
        index_mutations,
        row_locator,
        table,
        branch,
        context,
        is_known_new_object,
    } = request;
    let batch_id = row.batch_id();
    debug_assert!(
        !is_known_new_object || row.parents.is_empty(),
        "known-new row apply skips parent lookups"
    );
    let previous_entry = if is_known_new_object {
        None
    } else {
        load_previous_visible_entry(
            io,
            &table,
            object_id,
            &branch,
            context.user_descriptor().as_ref(),
        )?
    };
    let previous_visible = previous_entry
        .as_ref()
        .map(|entry| entry.current_row.clone());

    if !is_known_new_object {
        for parent in &row.parents {
            if io
                .load_history_row_batch(&table, branch_name.as_str(), object_id, *parent)
                .map_err(RowHistoryError::StorageError)?
                .is_none()
            {
                return Err(RowHistoryError::ParentNotFound(*parent));
            }
        }
    }

    let current_entry = if is_known_new_object {
        visible_entry_from_history_rows(
            context.user_descriptor().as_ref(),
            std::slice::from_ref(&row),
        )
        .map_err(|err| {
            RowHistoryError::StorageError(StorageError::IoError(format!(
                "rebuild visible entry after append: {err}"
            )))
        })?
    } else {
        // Point lookup first: an identical already-applied batch is an
        // idempotent no-op.
        let existing_row = io
            .load_history_row_batch(&table, branch_name.as_str(), object_id, batch_id)
            .map_err(RowHistoryError::StorageError)?;
        // What a batch descends from is fixed when it is written, and so is the metadata it
        // was written with. A copy delivered for a query scope carries neither: the sender
        // clears the parents (`sync_logic::scope_delivery_row`) and reads the row from its
        // visible region, which stores no metadata. That says nothing about them, so where
        // this store already holds the batch, what it holds stands. Taking the copy's
        // word cut the batch off its own history — its parent, named by nobody any more,
        // became a tip again, and the next local write merged two states that were never
        // concurrent — and cost a rebuild of the visible entry from the row's whole
        // history each time.
        if let Some(existing) = existing_row.as_ref() {
            if row.parents.is_empty() && !existing.parents.is_empty() {
                row.parents = existing.parents.clone();
            }
            if row.metadata.is_empty() && !existing.metadata.is_empty() {
                row.metadata = existing.metadata.clone();
            }
        }
        if existing_row.as_ref() == Some(&row) {
            return Ok(ApplyRowBatchResult {
                batch_id,
                row_locator,
                visibility_change: None,
            });
        }

        let fast_entry = match existing_row.as_ref() {
            // Brand-new batch: the serial-append fast path, and for a row with more
            // than one tip the construction from its tips.
            None => try_serial_fastpath_entry(previous_entry.as_ref(), &row).or_else(|| {
                let entry = try_forked_fastpath_entry(
                    context.user_descriptor().as_ref(),
                    previous_entry.as_ref(),
                    &row,
                    &|batch_id| {
                        io.load_history_row_batch(&table, branch_name.as_str(), object_id, batch_id)
                            .ok()
                            .flatten()
                    },
                )?;
                FORKED_FASTPATH_HITS.fetch_add(1, Ordering::Relaxed);
                #[cfg(test)]
                super::fastpath::FORKED_FASTPATH_ARMS_ON_THREAD.with(|arms| {
                    let mut taken = arms.get();
                    taken[usize::from(entry.branch_frontier.len() > 1)] += 1;
                    arms.set(taken);
                });
                Some(entry)
            }),
            // Same batch id with different content — an in-place replace.
            // Two provable re-apply shapes get a fast path; everything else
            // takes the full path.
            Some(existing) if !existing.state.is_visible() => {
                // Publish-shaped re-apply (`DurableDirect` /
                // `AcceptedTransaction` fates over a staged row): the stored
                // version is invisible, so the previous entry is oblivious
                // to this batch — replacing it with a visible row is the
                // same domination event as inserting a fresh visible row.
                try_serial_fastpath_entry(previous_entry.as_ref(), &row)
            }
            Some(existing) => {
                // Visible → visible with only `state`/`confirmed_tier`
                // changed (tier confirmations re-applied via
                // `accepted_transaction_output`): in-place tip update.
                try_in_place_tip_update_entry(previous_entry.as_ref(), existing, &row)
            }
        };
        // Telemetry over the population the fast path targets (previous entry
        // present, incoming row visible) so fallbacks measure real misses, not
        // staging or first-write traffic.
        if previous_entry.is_some() && row.state.is_visible() {
            let counter = if fast_entry.is_some() {
                &HISTORY_FASTPATH_HITS
            } else {
                &HISTORY_FASTPATH_FALLBACKS
            };
            counter.fetch_add(1, Ordering::Relaxed);
        }

        if let Some(entry) = fast_entry {
            Some(entry)
        } else {
            let mut patched_history = load_branch_history(io, &table, object_id, &branch)?;
            if let Some(existing) = patched_history
                .iter_mut()
                .find(|candidate| candidate.batch_id() == batch_id)
            {
                *existing = row.clone();
            } else {
                patched_history.push(row.clone());
            }
            visible_entry_from_history_rows(context.user_descriptor().as_ref(), &patched_history)
                .map_err(|err| {
                    RowHistoryError::StorageError(StorageError::IoError(format!(
                        "rebuild visible entry after append: {err}"
                    )))
                })?
        }
    };
    let current_visible = current_entry
        .as_ref()
        .map(|entry| entry.current_row.clone());
    let visible_entry_changed = current_entry.as_ref() != previous_entry.as_ref();
    let visible_entries: &[VisibleRowEntry] = match (visible_entry_changed, current_entry.as_ref())
    {
        (true, Some(entry)) => std::slice::from_ref(entry),
        _ => &[],
    };
    let visible_changed = previous_visible != current_visible;
    let can_encode_visible_with_row_context = visible_entries.len() == 1
        && visible_entries[0].current_row.row_id == row.row_id
        && visible_entries[0].current_row.branch == row.branch
        && visible_entries[0].current_row.batch_id() == row.batch_id();

    if visible_entries.is_empty() || can_encode_visible_with_row_context {
        let encoded_history = crate::storage::encode_history_row_bytes_with_context(&context, &row)
            .map_err(RowHistoryError::StorageError)?;
        let encoded_visible = if let Some(entry) = visible_entries.first() {
            vec![
                crate::storage::encode_visible_row_bytes_with_context(&context, entry)
                    .map_err(RowHistoryError::StorageError)?,
            ]
        } else {
            Vec::new()
        };
        <H as Storage>::apply_prepared_row_mutation(
            io,
            &table,
            std::slice::from_ref(&row),
            visible_entries,
            std::slice::from_ref(&encoded_history),
            &encoded_visible,
            index_mutations,
        )
        .map_err(RowHistoryError::StorageError)?;
    } else {
        <H as Storage>::apply_row_mutation(
            io,
            &table,
            std::slice::from_ref(&row),
            visible_entries,
            index_mutations,
        )
        .map_err(RowHistoryError::StorageError)?;
    }

    if matches!(row.state, RowState::StagingPending) {
        supersede_older_staging_rows_for_batch(io, &table, object_id, branch_name, row.batch_id)?;
    }

    let applied = AppliedRowBatch {
        row_locator: row_locator.clone(),
        previous_visible: previous_visible.clone(),
        current_visible,
        is_new_object: previous_visible.is_none(),
        visible_changed,
    };

    Ok(ApplyRowBatchResult {
        batch_id,
        row_locator,
        visibility_change: visibility_change_from_applied(object_id, applied),
    })
}

pub fn patch_row_batch_state<H: Storage>(
    io: &mut H,
    object_id: ObjectId,
    branch_name: &BranchName,
    batch_id: BatchId,
    state: Option<RowState>,
    confirmed_tier: Option<DurabilityTier>,
) -> Result<Option<RowVisibilityChange>, RowHistoryError> {
    let row_locator = row_locator_from_storage(io, object_id)?;
    let table = row_locator.table.to_string();
    let branch = SharedString::from(branch_name.as_str().to_string());
    let original_row = io
        .load_history_row_batch(&table, branch_name.as_str(), object_id, batch_id)
        .map_err(RowHistoryError::StorageError)?
        .ok_or(RowHistoryError::ObjectNotFound(object_id))?;
    if original_row.branch.as_str() != branch_name.as_str() {
        return Ok(None);
    }
    // The family this batch ALREADY lives in, not the one the locator names: a
    // patch must rewrite the batch where it is, or it leaves the original copy
    // behind in another generation with its old state and creates a second one
    // here. `scan_history_row_batches` is sibling-complete, so the row then has
    // two versions of one batch, and the visible resolution takes whichever is
    // visible — a batch patched to `Rejected` goes on being served.
    let context = crate::storage::existing_history_row_write_context(
        io,
        &table,
        branch_name.as_str(),
        object_id,
        batch_id,
        &original_row,
    )
    .map_err(RowHistoryError::StorageError)?;
    let previous_entry = load_previous_visible_entry(
        io,
        &table,
        object_id,
        &branch,
        context.user_descriptor().as_ref(),
    )?;
    let previous_visible = previous_entry
        .as_ref()
        .map(|entry| entry.current_row.clone());

    let mut patched_row = original_row.clone();
    if let Some(state) = state {
        patched_row.state = state;
    }
    patched_row.confirmed_tier = match (patched_row.confirmed_tier, confirmed_tier) {
        (Some(existing), Some(incoming)) => Some(existing.max(incoming)),
        (Some(existing), None) => Some(existing),
        (None, incoming) => incoming,
    };

    // Routing decision, in order:
    // - Patched state NOT visible (`→ Rejected`, `→ Superseded` — including
    //   any visible→non-visible flip): ALWAYS the full path. Removing a batch
    //   from the visible set can expose a previously hidden ancestor as the
    //   new winner, which no O(1) entry update can compute. First-class
    //   invariant of the design, not an optimisation miss.
    // - Previously invisible → now visible (staging publish after local
    //   durability, `runtime_core/writes.rs`): a domination event over the
    //   pre-flip frontier — identical math to inserting a fresh visible row,
    //   shared with the serial-append fast path.
    // - Visible → visible (pure `confirmed_tier` bump; no production caller
    //   today): in-place tip update, provable only when the batch is the sole
    //   frontier tip of a never-forked entry. A bump on a non-tip batch or on
    //   a pooled merge contributor takes the full path (recomputing a merged
    //   tier preview over the bounded `winner_batch_pool` is a possible
    //   future refinement — design doc §5).
    let fast_entry = if !patched_row.state.is_visible() {
        None
    } else if !original_row.state.is_visible() {
        try_serial_fastpath_entry(previous_entry.as_ref(), &patched_row)
    } else {
        try_in_place_tip_update_entry(previous_entry.as_ref(), &original_row, &patched_row)
    };
    // Telemetry over the population the patch fast paths target (previous
    // entry present, PATCHED row visible). Transitions out of the visible set
    // are full-path by design and deliberately not counted as fallbacks.
    if previous_entry.is_some() && patched_row.state.is_visible() {
        let counter = if fast_entry.is_some() {
            &PATCH_FASTPATH_HITS
        } else {
            &PATCH_FASTPATH_FALLBACKS
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    let patched_entry = if fast_entry.is_some() {
        fast_entry
    } else {
        let mut history_rows = load_branch_history(io, &table, object_id, &branch)?;
        let Some(existing) = history_rows
            .iter_mut()
            .find(|candidate| candidate.batch_id() == batch_id)
        else {
            return Err(RowHistoryError::ObjectNotFound(object_id));
        };
        *existing = patched_row.clone();
        visible_entry_from_history_rows(context.user_descriptor().as_ref(), &history_rows).map_err(
            |err| {
                RowHistoryError::StorageError(StorageError::IoError(format!(
                    "rebuild visible entry after patch: {err}"
                )))
            },
        )?
    };
    let visible_entries: Vec<_> = patched_entry.iter().cloned().collect();
    // The head this patch lands belongs to the WINNER, never to the batch being
    // patched: rejecting a generation-B tip can hand the row back to a
    // generation-A version, and the head then belongs in generation A. Nothing is
    // predicted here any more — the write below encodes the entry, and
    // `storage::enforce_single_visible_family_after_write` then measures which
    // families actually hold `(branch, row)` and moves the head out of the one
    // the winner left. Measuring after the fact is strictly better than
    // predicting before it: this path re-enters itself
    // (`supersede_older_staging_rows_for_batch`), so a prediction taken here can
    // be stale by the time the bytes land.
    if patched_entry.is_some() {
        // Encoded with the EXISTING batch's context, so the patched bytes
        // replace the batch in place instead of forking it into the family the
        // locator happens to name. The visible entries keep their own
        // per-winner resolution.
        let encoded_history =
            crate::storage::encode_history_row_bytes_with_context(&context, &patched_row)
                .map_err(RowHistoryError::StorageError)?;
        let encoded_visible =
            crate::storage::encode_visible_row_bytes_for_storage(io, &table, &visible_entries)
                .map_err(RowHistoryError::StorageError)?;
        <H as Storage>::apply_prepared_row_mutation(
            io,
            &table,
            std::slice::from_ref(&patched_row),
            &visible_entries,
            std::slice::from_ref(&encoded_history),
            &encoded_visible,
            &[],
        )
        .map_err(RowHistoryError::StorageError)?;
    } else {
        io.append_history_region_rows(&table, std::slice::from_ref(&patched_row))
            .map_err(RowHistoryError::StorageError)?;
        io.delete_visible_region_row(&table, branch_name.as_str(), object_id)
            .map_err(RowHistoryError::StorageError)?;
    }

    let current_visible = patched_entry
        .as_ref()
        .map(|entry| entry.current_row.clone());
    if previous_visible == current_visible {
        return Ok(None);
    }

    let Some(current_visible) = current_visible else {
        return Ok(None);
    };

    Ok(Some(RowVisibilityChange {
        object_id,
        row_locator,
        row: current_visible,
        previous_row: previous_visible.clone(),
        is_new_object: previous_visible.is_none(),
    }))
}
