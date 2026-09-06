use super::*;
use crate::batch_fate::{LocalBatchMember, SealedBatchMember, SealedBatchSubmission};
use crate::row_histories::{BatchId, RowState, StoredRowBatch, patch_row_batch_state};
use crate::storage::metadata_from_row_locator;
use crate::sync_manager::{RowMetadata, SyncPayload};

type LocalBatchRow = (LocalBatchMember, crate::storage::RowLocator, StoredRowBatch);

/// Fires when every batchId->rows member source misses and `local_batch_rows`
/// degrades to scanning the history of every row in every table. That walk is
/// O(total rows x history depth) and must stay cold: production telemetry and
/// the linsa_schema_profile harness both watch this counter.
/// What a full-store batch scan cost, for the warn that reports it.
#[derive(Default)]
struct ScanCost {
    objects: usize,
    history_entries: usize,
}

/// Who asked `local_batch_rows` for a batch's rows. Named in the fallback warn
/// so a production scan storm identifies its own driver instead of requiring a
/// stack dump on a live server.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LocalBatchLookup {
    /// A confirmed fate refreshing the subscriptions that hold its rows.
    ConfirmedFate,
    /// Retransmitting a local batch upstream (a `Missing` fate, a reconnect).
    Retransmit,
    /// Deriving the set of batches still needing settlement (reconnect, boot).
    PendingReconciliation,
    /// Rebuilding local batch records for worker sync.
    WorkerSync,
}

impl LocalBatchLookup {
    fn as_str(self) -> &'static str {
        match self {
            Self::ConfirmedFate => "confirmed_fate",
            Self::Retransmit => "retransmit",
            Self::PendingReconciliation => "pending_reconciliation",
            Self::WorkerSync => "worker_sync",
        }
    }

    /// One line on what the caller was trying to do and what a miss means for
    /// it — the context a responder needs at 3am.
    fn detail(self) -> &'static str {
        match self {
            Self::ConfirmedFate => {
                "a batch was confirmed durable and its rows are being marked for \
                 subscription recompute; a miss means the confirmation cannot refresh \
                 any subscriber"
            }
            Self::Retransmit => {
                "re-offering a local batch to a server; a miss means there is nothing \
                 left to send and the peer will keep asking"
            }
            Self::PendingReconciliation => {
                "deriving which local batches still need settlement, on reconnect or \
                 boot; a miss here is one scan per unsettled batch on record"
            }
            Self::WorkerSync => {
                "rebuilding local batch records for a worker runtime; a miss drops that \
                 batch from the replay"
            }
        }
    }
}

pub static LOCAL_BATCH_FULL_SCANS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

impl<S: Storage, Sch: Scheduler> RuntimeCore<S, Sch> {
    fn local_batch_row_from_member(
        &self,
        batch_id: BatchId,
        member: &LocalBatchMember,
    ) -> Option<(crate::storage::RowLocator, StoredRowBatch)> {
        let row_locator = self
            .storage
            .load_row_locator(member.object_id)
            .ok()
            .flatten()
            .unwrap_or_else(|| crate::storage::RowLocator {
                table: member.table_name.clone().into(),
                origin_schema_hash: None,
            });
        let row = self
            .storage
            .load_history_row_batch_for_schema_hash(
                member.table_name.as_str(),
                member.schema_hash,
                member.branch_name.as_str(),
                member.object_id,
                batch_id,
            )
            .ok()
            .flatten()?;
        // Either rule, as at the other check sites: the mint is parent-blind, and members
        // minted with parents included are already on disk in every installed store.
        (row.content_digest_ignoring_parents() == member.row_digest
            || row.content_digest() == member.row_digest)
            .then_some((row_locator, row))
    }

    fn sealed_submission_batch_members(&self, batch_id: BatchId) -> Vec<LocalBatchMember> {
        let Some(submission) = self
            .storage
            .load_sealed_batch_submission(batch_id)
            .ok()
            .flatten()
        else {
            return Vec::new();
        };

        submission
            .members
            .into_iter()
            .filter_map(|sealed_member| {
                let row_locator = self
                    .storage
                    .load_row_locator(sealed_member.object_id)
                    .ok()
                    .flatten()?;
                let schema_hash = self
                    .local_batch_member_schema_hash(
                        submission.target_branch_name,
                        sealed_member.object_id,
                        batch_id,
                    )
                    .ok()?;
                Some(LocalBatchMember {
                    object_id: sealed_member.object_id,
                    table_name: row_locator.table.to_string(),
                    branch_name: submission.target_branch_name,
                    schema_hash,
                    row_digest: sealed_member.row_digest,
                })
            })
            .collect()
    }

    fn cached_local_batch_members(&self, batch_id: BatchId) -> Vec<LocalBatchMember> {
        self.local_batch_record_cache
            .get(&batch_id)
            .map(|record| record.members.clone())
            .unwrap_or_default()
    }

    fn persisted_local_batch_members(&self, batch_id: BatchId) -> Vec<LocalBatchMember> {
        self.storage
            .load_local_batch_record(batch_id)
            .ok()
            .flatten()
            .map(|record| record.members)
            .unwrap_or_default()
    }

    fn indexed_local_batch_members(&self, batch_id: BatchId) -> Vec<LocalBatchMember> {
        self.storage
            .load_local_batch_row_index(batch_id)
            .ok()
            .flatten()
            .unwrap_or_default()
    }

    fn local_batch_rows_from_members(
        &self,
        batch_id: BatchId,
        members: Vec<LocalBatchMember>,
    ) -> Vec<LocalBatchRow> {
        members
            .into_iter()
            .filter_map(|member| {
                self.local_batch_row_from_member(batch_id, &member)
                    .map(|(row_locator, row)| (member, row_locator, row))
            })
            .collect()
    }

    /// The full-store scan, plus what it cost. The counts are what makes a
    /// production warn actionable: they say how much of the store this one
    /// lookup walked.
    fn scan_local_batch_rows_measured(&self, batch_id: BatchId) -> (Vec<LocalBatchRow>, ScanCost) {
        let mut cost = ScanCost::default();
        let Ok(row_locators) = self.storage.scan_row_locators() else {
            return (Vec::new(), cost);
        };

        let mut rows = Vec::new();
        for (object_id, row_locator) in row_locators {
            cost.objects += 1;
            let Ok(history_rows) = self
                .storage
                .scan_history_row_batches(row_locator.table.as_str(), object_id)
            else {
                continue;
            };
            cost.history_entries += history_rows.len();
            for row in history_rows
                .into_iter()
                .filter(|row| row.batch_id == batch_id)
            {
                let branch_name = BranchName::new(row.branch.as_str());
                let Ok(schema_hash) =
                    self.local_batch_member_schema_hash(branch_name, object_id, batch_id)
                else {
                    continue;
                };
                let member = LocalBatchMember {
                    object_id,
                    table_name: row_locator.table.to_string(),
                    branch_name,
                    schema_hash,
                    // Parent-blind on purpose: a member identifies a payload, and the two
                    // copies of a row that meet here do not agree about parents once
                    // delivery stamps them. See `content_digest_ignoring_parents`.
                    row_digest: row.content_digest_ignoring_parents(),
                };
                rows.push((member, row_locator.clone(), row));
            }
        }
        (rows, cost)
    }

    fn sort_local_batch_rows(rows: &mut [LocalBatchRow]) {
        rows.sort_by(
            |(left_member, left_locator, left_row), (right_member, right_locator, right_row)| {
                left_member
                    .object_id
                    .uuid()
                    .as_bytes()
                    .cmp(right_member.object_id.uuid().as_bytes())
                    .then_with(|| {
                        left_locator
                            .table
                            .as_str()
                            .cmp(right_locator.table.as_str())
                    })
                    .then_with(|| left_row.branch.as_str().cmp(right_row.branch.as_str()))
                    .then_with(|| {
                        left_member
                            .schema_hash
                            .as_bytes()
                            .cmp(right_member.schema_hash.as_bytes())
                    })
                    .then_with(|| left_row.batch_id.0.cmp(&right_row.batch_id.0))
            },
        );
    }

    fn local_batch_row_was_insert(
        &self,
        table: &str,
        row: &crate::row_histories::StoredRowBatch,
    ) -> bool {
        if !row.parents.is_empty() {
            return false;
        }

        let Ok(history_rows) = self.storage.scan_history_row_batches(table, row.row_id) else {
            return true;
        };
        !history_rows.iter().any(|candidate| {
            candidate.branch == row.branch
                && candidate.batch_id != row.batch_id
                && !matches!(candidate.state, RowState::Rejected)
        })
    }

    /// Full-store batch scans this runtime has paid.
    pub fn local_batch_full_scan_count(&self) -> u64 {
        self.local_batch_full_scans.get()
    }

    /// The four point-lookup member sources — everything this node already
    /// tracks about a batch. No scan, no fallback.
    fn local_batch_rows_from_tracked_sources(&self, batch_id: BatchId) -> Vec<LocalBatchRow> {
        let member_sources: [fn(&Self, BatchId) -> Vec<LocalBatchMember>; 4] = [
            Self::sealed_submission_batch_members,
            Self::cached_local_batch_members,
            Self::persisted_local_batch_members,
            Self::indexed_local_batch_members,
        ];

        let mut rows = Vec::new();
        for load_members in member_sources {
            rows = self.local_batch_rows_from_members(batch_id, load_members(self, batch_id));
            if !rows.is_empty() {
                break;
            }
        }
        rows
    }

    /// Rows of a batch that this node already tracks, never paying the
    /// full-store fallback. For callers whose work is bookkeeping over rows we
    /// hold: with no bookkeeping there is nothing for them to do, and
    /// rediscovering that by walking every table's history is unbounded work.
    pub(crate) fn local_batch_rows_tracked_only(&self, batch_id: BatchId) -> Vec<LocalBatchRow> {
        let mut rows = self.local_batch_rows_from_tracked_sources(batch_id);
        Self::sort_local_batch_rows(&mut rows);
        rows
    }

    pub(crate) fn local_batch_rows(
        &self,
        batch_id: BatchId,
        caller: LocalBatchLookup,
    ) -> Vec<LocalBatchRow> {
        let mut rows = self.local_batch_rows_from_tracked_sources(batch_id);
        if rows.is_empty() {
            // A scan that already answered "no rows" stays valid until a row
            // batch with this id arrives (see `known_empty_batch_scans`) —
            // answer repeat questions from the cache instead of re-scanning.
            if self.known_empty_batch_scans.borrow().contains(&batch_id) {
                return Vec::new();
            }
            // Last-resort fallback if the batchId->rows index is not found: a
            // walk of every table's history. The fields below exist because a
            // bare "index missed" warn cost this team hours in the 2026-08-09
            // incident — it named neither who asked nor what the answer cost.
            LOCAL_BATCH_FULL_SCANS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.local_batch_full_scans
                .set(self.local_batch_full_scans.get() + 1);
            let started = web_time::Instant::now();
            let (scanned, cost) = self.scan_local_batch_rows_measured(batch_id);
            rows = scanned;
            tracing::warn!(
                ?batch_id,
                caller = caller.as_str(),
                caller_detail = caller.detail(),
                rows_found = rows.len(),
                objects_scanned = cost.objects,
                history_entries_scanned = cost.history_entries,
                scan_millis = started.elapsed().as_millis() as u64,
                has_sealed_submission = self
                    .storage
                    .load_sealed_batch_submission(batch_id)
                    .ok()
                    .flatten()
                    .is_some(),
                has_local_record = self
                    .storage
                    .load_local_batch_record(batch_id)
                    .ok()
                    .flatten()
                    .is_some(),
                has_row_index = self
                    .storage
                    .load_local_batch_row_index(batch_id)
                    .ok()
                    .flatten()
                    .is_some(),
                cached_record = self.local_batch_record_cache.contains_key(&batch_id),
                "batchId->rows index missed; falling back to full-store history scan"
            );
            if rows.is_empty() {
                self.known_empty_batch_scans.borrow_mut().insert(batch_id);
            }
        }

        Self::sort_local_batch_rows(&mut rows);
        rows
    }

    pub(crate) fn direct_sealed_submission_from_local_batch_rows(
        batch_id: crate::row_histories::BatchId,
        rows: &[(
            LocalBatchMember,
            crate::storage::RowLocator,
            crate::row_histories::StoredRowBatch,
        )],
    ) -> Option<SealedBatchSubmission> {
        let first_branch = rows.first()?.0.branch_name;
        if rows
            .iter()
            .any(|(member, _, _)| member.branch_name != first_branch)
        {
            return None;
        }
        if rows
            .iter()
            .any(|(_, _, row)| !matches!(row.state, crate::row_histories::RowState::VisibleDirect))
        {
            return None;
        }
        Some(SealedBatchSubmission::new(
            batch_id,
            crate::batch_fate::BatchMode::Direct,
            first_branch,
            rows.iter()
                .map(|(member, _, _)| SealedBatchMember {
                    object_id: member.object_id,
                    row_digest: member.row_digest,
                })
                .collect(),
            Vec::new(),
        ))
    }

    fn apply_received_batch_fate(&mut self, fate: crate::batch_fate::BatchFate) {
        let batch_id = fate.batch_id();
        // Same reason as `SyncManager::persist_authoritative_batch_fate`, and this is the
        // write that actually reached the disk in production: it is unconditional and it
        // bypasses `merged_with` entirely, so it also clobbers a stored `Rejected` that the
        // merge is written to keep. `Missing` is an instruction to resend, not a state to
        // remember.
        if !matches!(fate, crate::batch_fate::BatchFate::Missing { .. })
            && let Err(error) = self.storage.upsert_authoritative_batch_fate(&fate)
        {
            tracing::warn!(
                ?batch_id,
                %error,
                "failed to persist batch fate"
            );
        }

        if let crate::batch_fate::BatchFate::Rejected { code, reason, .. } = &fate {
            self.mark_local_batch_rows_rejected(batch_id);
            let acknowledged = self
                .is_rejected_batch_acknowledged(batch_id)
                .unwrap_or(false);
            if !acknowledged {
                let handled_by_waiter = self.durability.record_rejection(batch_id, code, reason);
                if !handled_by_waiter {
                    let batch = self
                        .local_batch_record(batch_id)
                        .ok()
                        .flatten()
                        .unwrap_or_else(|| {
                            crate::batch_fate::LocalBatchRecord::new(
                                batch_id,
                                crate::batch_fate::BatchMode::Direct,
                                true,
                                Some(fate.clone()),
                            )
                        });
                    self.queue_mutation_error_event(crate::runtime_core::MutationErrorEvent {
                        code: code.clone(),
                        reason: reason.clone(),
                        batch,
                    });
                }
            }
        } else if matches!(fate, crate::batch_fate::BatchFate::Missing { .. }) {
            self.retransmit_local_batch_to_servers(batch_id);
        } else if matches!(
            fate,
            crate::batch_fate::BatchFate::AcceptedTransaction { .. }
        ) {
            self.schema_manager
                .query_manager_mut()
                .mark_subscriptions_visibility_recompute_for_batch(batch_id);
        }

        if let Some(acked_tier) = fate.confirmed_tier() {
            self.schema_manager
                .query_manager_mut()
                .mark_subscriptions_visibility_recompute_for_tier(acked_tier);
            for (member, row_locator, _) in
                self.local_batch_rows(batch_id, LocalBatchLookup::ConfirmedFate)
            {
                self.schema_manager
                    .query_manager_mut()
                    .mark_local_row_updated_in_subscriptions(
                        row_locator.table.as_str(),
                        member.object_id,
                    );
            }
            self.durability.record_batch_ack(batch_id, acked_tier);
            // Only confirmed fates retire bookkeeping here: `Rejected` keeps
            // its record for the mutation-error replay path and `Missing`
            // pends retransmission, and neither carries a confirmed tier.
            if acked_tier >= self.settlement_target() {
                self.retire_settled_batch(batch_id, acked_tier);
            }
        }
    }

    pub(crate) fn mark_local_batch_rows_rejected(
        &mut self,
        batch_id: crate::row_histories::BatchId,
    ) {
        let mut cleared_rows = Vec::new();
        let mut batch_patch_succeeded_by_table = std::collections::HashMap::new();

        // Tracked rows only, never the full-store fallback. Marking a batch's
        // rows rejected is bookkeeping over rows this node holds; when it holds
        // none there is nothing to mark. Rejections arrive with peer input
        // (a policy denial per write), so paying a walk of every table's
        // history per rejection is unbounded work bought by a peer — measured
        // in production 2026-08-10: 264 denied writes, 264 full scans, one core
        // pinned. A rejected write also never lands, so the scan it used to pay
        // for could not have found anything: zero of the incident store's 1049
        // rejected batches existed anywhere in its history.
        for (member, row_locator, row) in self.local_batch_rows_tracked_only(batch_id) {
            let was_visible = matches!(row.state, RowState::VisibleDirect)
                || (matches!(row.state, RowState::Rejected)
                    && self
                        .storage
                        .load_visible_region_row(
                            row_locator.table.as_str(),
                            row.branch.as_str(),
                            member.object_id,
                        )
                        .ok()
                        .flatten()
                        .is_some_and(|visible_row| visible_row.batch_id() == row.batch_id()));

            if !was_visible && !matches!(row.state, RowState::StagingPending | RowState::Superseded)
            {
                continue;
            }

            cleared_rows.push((
                row_locator.table.to_string(),
                member.schema_hash,
                row.branch.to_string(),
                member.object_id,
                row.batch_id(),
                row.data.to_vec(),
                was_visible,
                row.delete_kind.is_some(),
                self.local_batch_row_was_insert(row_locator.table.as_str(), &row),
            ));
        }

        for (table, _, _, _, _, _, was_visible, _, _) in &cleared_rows {
            if *was_visible {
                continue;
            }
            batch_patch_succeeded_by_table
                .entry(table.clone())
                .or_insert_with(|| {
                    self.storage
                        .patch_row_region_rows_by_batch(
                            table,
                            batch_id,
                            Some(RowState::Rejected),
                            None,
                        )
                        .is_ok()
                });
        }

        let query_manager = self.schema_manager.query_manager_mut();
        for (
            table,
            schema_hash,
            branch,
            row_id,
            member_batch_id,
            row_data,
            was_visible,
            was_delete,
            was_insert,
        ) in cleared_rows
        {
            if was_visible {
                let branch_name = crate::object::BranchName::new(&branch);
                let _ = self.storage.patch_row_region_rows_by_batch(
                    &table,
                    member_batch_id,
                    Some(RowState::Rejected),
                    None,
                );
                let _ = patch_row_batch_state(
                    &mut self.storage,
                    row_id,
                    &branch_name,
                    member_batch_id,
                    Some(RowState::Rejected),
                    None,
                );
                let _ = self.storage.patch_exact_row_batch_for_schema_hash(
                    &table,
                    schema_hash,
                    &branch,
                    row_id,
                    member_batch_id,
                    Some(RowState::Rejected),
                    None,
                );
            } else if !batch_patch_succeeded_by_table
                .get(&table)
                .copied()
                .unwrap_or(false)
            {
                let _ = self.storage.patch_exact_row_batch_for_schema_hash(
                    &table,
                    schema_hash,
                    &branch,
                    row_id,
                    member_batch_id,
                    Some(RowState::Rejected),
                    None,
                );
            }
            if was_visible {
                if was_delete {
                    query_manager.restore_local_rejected_delete_row(
                        &mut self.storage,
                        &table,
                        &branch,
                        row_id,
                        &row_data,
                    );
                } else if !was_insert {
                    query_manager.clear_local_pending_row_overlay(&table, row_id);
                } else {
                    let _ = self
                        .storage
                        .delete_visible_region_row(&table, &branch, row_id);
                    query_manager.retract_local_rejected_row(
                        &mut self.storage,
                        &table,
                        &branch,
                        row_id,
                        &row_data,
                        true,
                    );
                }
            } else {
                query_manager.clear_local_pending_row_overlay(&table, row_id);
            }
        }
    }

    pub fn retransmit_local_batch_to_servers(&mut self, batch_id: crate::row_histories::BatchId) {
        let sealed_submission = self
            .storage
            .load_sealed_batch_submission(batch_id)
            .ok()
            .flatten();

        let local_rows = self.local_batch_rows(batch_id, LocalBatchLookup::Retransmit);
        let rows_to_retransmit = local_rows
            .iter()
            .map(|(member, row_locator, row)| {
                (
                    member.object_id,
                    metadata_from_row_locator(row_locator),
                    row.clone(),
                )
            })
            .collect::<Vec<_>>();
        let sealed_submission = sealed_submission.or_else(|| {
            Self::direct_sealed_submission_from_local_batch_rows(batch_id, &local_rows)
        });

        let sync_manager = self.schema_manager.query_manager_mut().sync_manager_mut();
        for (row_id, metadata, row) in rows_to_retransmit {
            sync_manager.force_row_batch_to_servers(row_id, metadata, row);
        }
        if let Some(submission) = sealed_submission {
            sync_manager.seal_batch_to_servers(submission);
        }
    }

    // =========================================================================
    // Tick Methods
    // =========================================================================

    /// Synchronous tick - processes managers, fulfills completed queries.
    ///
    /// Schedules batched_tick if there are outbound messages or storage writes
    /// waiting on the WAL flush barrier.
    ///
    /// Call this after any mutation operation (insert, update, delete, etc.)
    /// to process the change and schedule any required I/O.
    pub fn immediate_tick(&mut self) -> TickOutput {
        // v18 item 6: one settle clock per lock hold — the outermost tick owns it, nested
        // ticks inherit (design v3 § B2, v4 SF2). The re-arm belongs to the clock's owner:
        // when its hold ends it takes the flag a pass set by deferring non-stalled work for
        // budget and re-arms exactly once. A nested tick leaves the flag for its owner —
        // a batched tick's continuation (3b) reads it first, and re-arming here as well
        // counted one deferral twice (design v8, gate G6-7(f)).
        // The clock is released even if the tick panics (diff r2 S7): a clock left set would
        // make every later `begin_tick` look nested, and under a budget no unit would run
        // again.
        //
        // CORRECTED (diff r30). This used to end "...on a host that survives a caught panic
        // (RN's panic boundary)", and that host does not exist in this tree: every host holds
        // the core behind a `Mutex` and none calls `into_inner()`, so a panic that unwinds
        // through a tick poisons it and the runtime is finished — `runtime_tokio.rs:166-173`
        // says so in as many words, and `jazz-rn:773-779` returns `lock poisoned`. RN's own
        // boundary is one layer further out: it wraps the JS subscription callback
        // (`jazz-rn/rust/src/lib.rs:399-410`) so that panic never reaches a tick at all. The
        // release stays — it costs a landing pad this function already pays for and it keeps
        // the invariant true — but it is hygiene, not a live outage being prevented, and
        // `sqlite.rs:737` ("no host runs another pass over a poisoned core") is the statement
        // of this that the sources actually support.
        let owns_clock = self.schema_manager.query_manager_mut().begin_tick();
        // Only the owner catches (diff r3 S2): a nested tick has nothing to release, so it
        // pays no landing pad. The clock is released BEFORE the re-arm (diff r3 S1): the
        // re-arm ends in the host's `schedule_batched_tick`, which may panic, and it reads
        // only the flag, never the clock.
        let output = if owns_clock {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.immediate_tick_inner()))
        } else {
            Ok(self.immediate_tick_inner())
        };
        if owns_clock {
            self.schema_manager.query_manager_mut().end_tick();
            if output.is_ok() {
                self.rearm_for_deferred_settle_work();
            }
        }
        match output {
            Ok(output) => output,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }
    fn immediate_tick_inner(&mut self) -> TickOutput {
        let _span = trace_span!("immediate_tick", tier = self.tier_label).entered();
        // v18 item 4 (C1): one pass transaction for every read below (nested ticks inherit).
        self.begin_read_pass_in_tick();
        let recovered_sealed_batches = self
            .schema_manager
            .query_manager_mut()
            .sync_manager_mut()
            .recover_completed_sealed_batches_with_storage(&mut self.storage);
        if recovered_sealed_batches {
            self.mark_storage_write_pending_flush();
        }

        // 1. Process logical updates (sync, subscriptions)
        self.schema_manager.process(&mut self.storage);

        // 2. Second process() handles deferred query subscriptions that couldn't
        //    compile on first pass (schema wasn't available yet, e.g. catalogue
        //    was just processed and made the schema available).
        self.schema_manager.process(&mut self.storage);

        // 2b. Release QuerySettled notifications whose upstream stream watermark
        // has definitely been applied.
        let ready_query_settled = {
            let pending = self
                .schema_manager
                .query_manager_mut()
                .sync_manager_mut()
                .take_pending_query_settled();
            let mut ready = Vec::new();
            let mut blocked = Vec::new();

            for pending_settled in pending {
                let is_ready = pending_settled.server_id.is_none_or(|server_id| {
                    self.last_applied_server_seq
                        .get(&server_id)
                        .copied()
                        .unwrap_or(0)
                        >= pending_settled.through_seq
                });
                if is_ready {
                    tracing::trace!(
                        server_id = ?pending_settled.server_id,
                        query_id = pending_settled.query_id.0,
                        tier = ?pending_settled.tier,
                        through_seq = pending_settled.through_seq,
                        "jazz trace query settled ready for query manager"
                    );
                    ready.push(pending_settled);
                } else {
                    tracing::trace!(
                        server_id = ?pending_settled.server_id,
                        query_id = pending_settled.query_id.0,
                        tier = ?pending_settled.tier,
                        through_seq = pending_settled.through_seq,
                        last_applied = pending_settled.server_id.and_then(|server_id| {
                            self.last_applied_server_seq.get(&server_id).copied()
                        }),
                        "jazz trace query settled blocked on stream sequence"
                    );
                    blocked.push(pending_settled);
                }
            }

            if !blocked.is_empty() {
                self.schema_manager
                    .query_manager_mut()
                    .sync_manager_mut()
                    .requeue_pending_query_settled(blocked);
            }

            ready
        };

        if !ready_query_settled.is_empty() {
            {
                let query_manager = self.schema_manager.query_manager_mut();
                for pending_settled in ready_query_settled {
                    if let Some(server_id) = pending_settled.server_id {
                        query_manager
                            .sync_manager_mut()
                            .relay_query_settled_to_origins(
                                server_id,
                                pending_settled.query_id,
                                pending_settled.tier,
                            );
                    }
                    query_manager.apply_query_settled(
                        pending_settled.query_id,
                        pending_settled.tier,
                        pending_settled.server_id.is_some(),
                    );
                }
            }
            self.schema_manager.process(&mut self.storage);
        }

        // 2c. Apply replayable batch fates before collecting subscription
        // updates so fate-driven visibility changes land in the same tick.
        let received_batch_fates = self
            .schema_manager
            .query_manager_mut()
            .sync_manager_mut()
            .take_pending_batch_fates();
        if !received_batch_fates.is_empty() {
            for fate in received_batch_fates {
                self.apply_received_batch_fate(fate);
            }
            self.schema_manager.process(&mut self.storage);
        }

        if self.transport_catalogue_state_hash_dirty {
            self.refresh_transport_catalogue_state_hash();
        }

        // 3. Collect subscription updates
        let subscription_updates = self.schema_manager.query_manager_mut().take_updates();
        let subscription_failures = self
            .schema_manager
            .query_manager_mut()
            .take_failed_subscriptions();

        // Track one-shot queries that completed this tick
        let mut completed_one_shots: Vec<SubscriptionHandle> = Vec::new();
        let mut failed_one_shots: Vec<SubscriptionHandle> = Vec::new();
        let mut callbacks_fired: u64 = 0;

        // 3. Call subscription callbacks AND handle one-shot queries
        for update in &subscription_updates {
            if let Some(&handle) = self.subscription_reverse.get(&update.subscription_id) {
                // Check if this is a one-shot query
                if let Some(pending) = self.pending_one_shot_queries.get_mut(&handle) {
                    // First callback = graph settled, fulfill the future
                    if let Some(sender) = pending.sender.take() {
                        // Decode rows using the query's output descriptor
                        let results: Vec<(ObjectId, Vec<Value>)> = update
                            .ordered_delta
                            .added
                            .iter()
                            .filter_map(|row| {
                                decode_row(&update.descriptor, &row.row.data)
                                    .ok()
                                    .map(|values| (row.row.id, values))
                            })
                            .collect();
                        let _ = sender.send(Ok(results));
                    }
                    // Mark for cleanup (unsubscribe happens after loop)
                    completed_one_shots.push(handle);
                } else if let Some(state) = self.subscriptions.get(&handle) {
                    // Regular subscription - call callback
                    let delta = SubscriptionDelta {
                        handle,
                        ordered_delta: update.ordered_delta.clone(),
                        descriptor: update.descriptor.clone(),
                    };
                    (state.callback)(delta);
                    callbacks_fired += 1;
                }
            }
        }
        tracing::debug!(callbacks_fired, "subscription callbacks fired this tick");

        for failure in &subscription_failures {
            if let Some(&handle) = self.subscription_reverse.get(&failure.subscription_id) {
                if let Some(pending) = self.pending_one_shot_queries.get_mut(&handle) {
                    if let Some(sender) = pending.sender.take() {
                        // The server's rejection code travels in the message: the facade
                        // on the other side of the binding can only see this string.
                        let _ = sender.send(Err(RuntimeError::QueryError(format!(
                            "query subscription {} failed [{}]: {}",
                            failure.subscription_id.0, failure.code, failure.reason
                        ))));
                    }
                    failed_one_shots.push(handle);
                } else if self.subscriptions.remove(&handle).is_some() {
                    self.subscription_reverse.remove(&failure.subscription_id);
                    tracing::error!(
                        handle = handle.0,
                        sub_id = failure.subscription_id.0,
                        error = %failure.reason,
                        "subscription failed during schema recompile and was dropped"
                    );
                }
            } else {
                tracing::error!(
                    sub_id = failure.subscription_id.0,
                    error = %failure.reason,
                    "subscription failed during schema recompile and was dropped"
                );
            }
        }

        // 2b. Cleanup completed one-shot queries
        for handle in completed_one_shots {
            if let Some(pending) = self.pending_one_shot_queries.remove(&handle) {
                // Unsubscribe from the underlying subscription
                self.schema_manager
                    .query_manager_mut()
                    .unsubscribe_with_sync(pending.subscription_id);
                self.subscription_reverse.remove(&pending.subscription_id);
            }
        }

        // 2c. Cleanup failed one-shot queries.
        // The underlying subscriptions were already removed by QueryManager.
        for handle in failed_one_shots {
            if let Some(pending) = self.pending_one_shot_queries.remove(&handle) {
                self.subscription_reverse.remove(&pending.subscription_id);
            }
        }

        // v18 item 4: the pass ends before the tail test reads the flush flag — a dirty
        // pass transaction is what the barrier is scheduled for.
        self.end_read_pass_in_tick();
        // 4. Schedule batched_tick if outbound messages exist or a WAL flush
        // barrier is pending. After a reported loss the storage arm is off (flag 3): the
        // barrier can never succeed until the store is reopened, and a tick that
        // re-scheduled itself for it would spin.
        if self.has_outbound()
            || (self.storage_write_pending_flush && !self.lost_writes_barrier_reported)
        {
            self.scheduler.schedule_batched_tick();
        }

        TickOutput {
            subscription_updates,
        }
    }

    /// Batched tick - handles all I/O, then processes parked messages.
    ///
    /// Called by the platform when the scheduled tick fires. This:
    /// 1. Sends all outgoing sync messages via SyncSender
    /// 2. Processes parked sync messages
    ///
    /// Each step is followed by an immediate_tick to process results.
    pub fn batched_tick(&mut self) {
        // v18 item 6: the batched tick owns the settle clock of its lock hold (nested
        // `immediate_tick`s inherit it) unless it runs inside one.
        // Same panic discipline as `immediate_tick` (diff r2 S7).
        let owns_clock = self.schema_manager.query_manager_mut().begin_tick();
        let outcome = if owns_clock {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.batched_tick_inner()))
        } else {
            self.batched_tick_inner();
            Ok(())
        };
        if owns_clock {
            self.schema_manager.query_manager_mut().end_tick();
            if outcome.is_ok() {
                self.rearm_for_deferred_settle_work();
            }
        }
        if let Err(payload) = outcome {
            std::panic::resume_unwind(payload);
        }
    }

    /// v18 item 6 (design v8): the clock owner's one re-arm. A pass that deferred
    /// non-stalled work for budget left the flag set; at the end of its lock hold the owner
    /// peeks at it and schedules the batched tick that continues the work — once per hold,
    /// whichever passes ran inside it. The flag stays set: it is the continuation's (3b)
    /// to take, in the batched tick this re-arm asked for. Invariant at the end of a hold:
    /// the flag is set iff a re-arm was scheduled by this hold.
    fn rearm_for_deferred_settle_work(&mut self) {
        if self.schema_manager.query_manager().settle_work_remains() {
            self.scheduler.schedule_batched_tick();
            // Counted after the call (diff r4 S3): a scheduler that panics scheduled nothing.
            crate::query_manager::settle_cost::bump(
                &crate::query_manager::settle_cost::SETTLE_TICKS_REARMED,
            );
            #[cfg(any(test, feature = "test"))]
            {
                self.settle_rearms += 1;
            }
        }
    }
    fn batched_tick_inner(&mut self) {
        let _span = debug_span!("batched_tick", tier = self.tier_label).entered();
        // v18 item 4 (C1): the batched tick's own pass transaction; the nested immediate
        // ticks below inherit it. Ended before the early return and before the barrier.
        self.begin_read_pass_in_tick();
        self.handle_transport_messages();

        if !self.has_outbound()
            && self
                .schema_manager
                .query_manager()
                .sync_manager()
                .has_pending_query_subscriptions()
        {
            self.immediate_tick();
        }

        // 1. Send all outgoing sync messages
        self.flush_runtime_outbox("flushing outbox");

        // 2. Drain parked sync messages. This must run regardless of whether
        //    a query subscription is currently deferred — the parked queue
        //    may carry the CatalogueEntryUpdated that satisfies the deferred
        //    sub, and `handle_sync_messages` is the only path that drains
        //    `parked_sync_messages`. It also calls `immediate_tick`
        //    internally when anything was applied, so deferred subs that can
        //    now compile do so before we return.
        let drained_any = self.handle_sync_messages();

        // 3. Flush any new outbox entries generated by processing.
        // The scheduler's debounce prevents immediate_tick() from scheduling
        // another batched_tick while we're inside one, so we must flush here.
        self.flush_runtime_outbox("flushing post-process outbox");

        // 3b. v18 item 6: the continuation. The last pass deferred non-stalled work for
        // budget, or tripped the outbox limiter with registrations left behind. Flush first
        // (the limiter reads the outbox); then, only if this lock hold's clock still has
        // room, take the flag and run one nested immediate tick that serves what fits (and
        // sets the flag again if it defers). Under a spent clock a nested tick is a no-op
        // pass — it would refuse every unit — so nothing runs and the flag stays set for
        // the owner's one re-arm at the end of the hold (`rearm_for_deferred_settle_work`;
        // design v8, diff r1 S6).
        if self.schema_manager.query_manager().settle_work_remains() {
            self.flush_runtime_outbox("flushing continuation outbox");
            if self.schema_manager.query_manager().settle_clock_has_room() {
                let _ = self
                    .schema_manager
                    .query_manager_mut()
                    .take_settle_work_remains();
                self.immediate_tick();
            }
        }
        // 4. If subscriptions are still pending, reschedule — but only if we
        //    actually drained something this tick. Otherwise no new work is
        //    waiting; the next inbound message will schedule us via
        //    `park_sync_message` itself. Rescheduling on a progressless tick
        //    hot-loops the JS scheduler.
        if drained_any
            && self
                .schema_manager
                .query_manager()
                .sync_manager()
                .has_pending_query_subscriptions()
        {
            self.end_read_pass_in_tick();
            self.scheduler.schedule_batched_tick();
            return;
        }
        // v18 item 4: the pass ends here, so the barrier below commits what it wrote.
        self.end_read_pass_in_tick();
        // Flush the storage durability barrier so writes survive a hard kill (tab close, crash).
        let mut barrier_failed = false;
        if self.storage_write_pending_flush {
            let _span = tracing::debug_span!("flush_wal").entered();
            // v18 item 8: the checkpoint policy lives under this barrier, and the settle line
            // cannot see it — `SettlePass::begin()` closes back in `QueryManager::process`,
            // above. Measured on the stand: `checkpoints=0` on all 175 settle lines while the
            // policy underneath was doing its job. So the barrier reports its own.
            let checkpoints_before = crate::query_manager::settle_cost::CheckpointCounts::read();
            let barrier_started = std::time::Instant::now();
            if let Err(error) = self.flush_wal_barrier() {
                barrier_failed = true;
                // A `LostWrites` was logged once by `record_storage_flush_error`; every
                // barrier after it would repeat the same line, so only other errors log here,
                // and no retry is scheduled for a loss (flag 3, set by the barrier just now).
                if !matches!(error, StorageError::LostWrites { .. }) {
                    tracing::error!(%error, "storage WAL flush failed");
                }
                // diff r23 SF5: the guard vetoes FIRST. `should_schedule_storage_flush_retry`
                // latches `storage_flush_retry_scheduled` as a side effect, so evaluating it on
                // the left set the latch on a lost store although no retry was scheduled. No
                // live reader can see that today, and a predicate that lies is still a trap.
                if !self.lost_writes_barrier_reported && self.should_schedule_storage_flush_retry()
                {
                    self.scheduler.schedule_batched_tick();
                }
            }
            // Only when something actually happened. A barrier that checkpointed nothing is the
            // common case once item 8 is on — that is the whole point of it — and logging a line
            // per tick to say so would reintroduce, in the log, the per-tick cost the item
            // removed from the disk.
            let checkpoints = crate::query_manager::settle_cost::CheckpointCounts::read()
                .since(checkpoints_before);
            if !checkpoints.is_empty() {
                tracing::info!(
                    checkpoints = checkpoints.checkpoints,
                    checkpoints_blocked = checkpoints.blocked,
                    checkpoint_failures = checkpoints.failures,
                    micros = barrier_started.elapsed().as_micros() as u64,
                    "jazz wal barrier"
                );
            }
        }

        // Only now may this node report what it applied. Confirming before the barrier
        // would let the sender clear its claim for a row a crash would take with it — the
        // same loss the confirmation exists to prevent, from the other side.
        if !barrier_failed {
            self.confirm_applied_rows_upstream();
        }
    }

    /// Report rows applied this tick to the server that sent them.
    fn confirm_applied_rows_upstream(&mut self) {
        // Prefer the transport's server, but fall back to a registered upstream: tying the
        // confirmation to a transport handle would make it silent in every setup that syncs
        // without one, and untestable below the transport layer.
        let server_id = match self.transport.as_ref().map(|handle| handle.server_id) {
            Some(server_id) => server_id,
            None => {
                let servers: Vec<_> = self
                    .schema_manager
                    .query_manager()
                    .sync_manager()
                    .server_ids()
                    .collect();
                match servers.as_slice() {
                    [server_id] => *server_id,
                    _ => return,
                }
            }
        };
        self.schema_manager
            .query_manager_mut()
            .sync_manager_mut()
            .queue_applied_row_confirmations(server_id);
        self.flush_runtime_outbox("flushing delivery confirmations");
    }
    pub(super) fn flush_runtime_outbox(&mut self, log_message: &str) {
        self.schema_manager
            .query_manager()
            .sync_manager()
            .publish_undelivered_gauges();

        let outbox = self
            .schema_manager
            .query_manager_mut()
            .sync_manager_mut()
            .take_outbox();
        if !outbox.is_empty() {
            debug!(count = outbox.len(), "{log_message}");
        }

        let mut unsent = Vec::new();
        for msg in outbox {
            let peer_kind = msg.destination.peer_kind();
            let peer_id = msg.destination.peer_uuid();
            let payload = msg.payload.variant_name();
            let _send_span = debug_span!(
                "sync.send",
                peer_kind = peer_kind,
                peer_id = %peer_id,
                payload = payload,
                payload_json = %serde_json::to_string(&msg.payload).unwrap_or_default(),
                tier = self.tier_label,
            )
            .entered();

            let handled_by_transport = self
                .transport
                .as_ref()
                .is_some_and(|handle| matches!(msg.destination, crate::sync_manager::Destination::Server(server_id) if server_id == handle.server_id));
            if handled_by_transport
                && matches!(msg.payload, SyncPayload::CatalogueEntryUpdated { .. })
            {
                if let Some(handle) = self.transport.as_ref() {
                    tracing::debug!(
                        server_id = %handle.server_id,
                        "dropping catalogue publish for transport; catalogue publication uses HTTP admin forwarding"
                    );
                }
                continue;
            }

            if let Some((ref tracer, ref name)) = self.sync_tracer {
                tracer.record_outgoing(name, &msg.destination, &msg.payload);
            }

            if handled_by_transport {
                if let Some(handle) = self.transport.as_ref() {
                    handle.send_outbox(msg);
                }
            } else if let Some(sync_sender) = self.sync_sender.as_ref() {
                sync_sender.send_sync_message(msg);
            } else if self.buffer_outbox_without_sync_sender {
                unsent.push(msg);
            }
        }

        if !unsent.is_empty() {
            self.schema_manager
                .query_manager_mut()
                .sync_manager_mut()
                .prepend_outbox(unsent);
        }
    }

    fn handle_transport_messages(&mut self) {
        let Some(server_id) = self.transport.as_ref().map(|handle| handle.server_id) else {
            return;
        };

        let mut inbound = Vec::new();
        if let Some(handle) = self.transport.as_mut() {
            while let Some(message) = handle.try_recv_inbound() {
                inbound.push(message);
            }
        }

        for message in inbound {
            match message {
                crate::transport_manager::TransportInbound::Connected {
                    catalogue_state_hash,
                    next_sync_seq,
                    supports_delivery_acks,
                } => {
                    // Learned per connection: a reconnect may land on a different server.
                    self.schema_manager
                        .query_manager_mut()
                        .sync_manager_mut()
                        .set_upstream_supports_delivery_acks(supports_delivery_acks);
                    if let Some(next_sync_seq) = next_sync_seq {
                        self.set_next_expected_server_sequence(server_id, next_sync_seq);
                    }
                    self.add_server_with_catalogue_state_hash_and_permission(
                        server_id,
                        catalogue_state_hash.as_deref(),
                        false,
                    );
                }
                crate::transport_manager::TransportInbound::Sync { entry, sequence } => {
                    if let Some(sequence) = sequence {
                        self.park_sync_message_with_sequence(*entry, sequence);
                    } else {
                        self.park_sync_message(*entry);
                    }
                }
                crate::transport_manager::TransportInbound::SyncBatch { entries } => {
                    self.park_sync_message_batch(entries);
                }
                crate::transport_manager::TransportInbound::Disconnected => {
                    self.remove_server(server_id);
                    self.schema_manager
                        .query_manager_mut()
                        .sync_manager_mut()
                        .add_pending_server(server_id);
                }
                crate::transport_manager::TransportInbound::ConnectFailed { reason } => {
                    tracing::warn!(%server_id, %reason, "transport connect failed");
                    self.schema_manager
                        .query_manager_mut()
                        .sync_manager_mut()
                        .remove_pending_server(server_id);
                }
                crate::transport_manager::TransportInbound::AuthFailure { reason } => {
                    tracing::warn!(%server_id, %reason, "transport auth failure");
                    self.remove_server(server_id);
                    if let Some(callback) = self.auth_failure_callback.as_ref() {
                        callback(reason);
                    }
                }
            }
        }
    }

    /// Apply parked sync messages and tick.
    ///
    /// Returns `true` if at least one parked message was pushed to the inbox
    /// this call. Callers use this to decide whether `batched_tick` made
    /// forward progress and therefore whether to reschedule itself.
    fn handle_sync_messages(&mut self) -> bool {
        let messages = std::mem::take(&mut self.parked_sync_messages);
        let mut applied_messages = 0usize;

        if !messages.is_empty() {
            debug!(
                count = messages.len(),
                "processing parked unsequenced sync messages"
            );
        }
        for msg in messages {
            if msg.payload.writes_storage() {
                self.mark_storage_write_pending_flush();
            }
            self.push_sync_inbox(msg);
            applied_messages += 1;
        }

        let server_ids: Vec<ServerId> = self
            .parked_sync_messages_by_server_seq
            .keys()
            .copied()
            .collect();
        let mut applied_since_last_tick = applied_messages > 0;
        for server_id in server_ids {
            let mut next_expected = *self.next_expected_server_seq.get(&server_id).unwrap_or(&1);
            let mut ready_messages = Vec::new();
            let mut remove_buffer = false;
            if let Some(buffered) = self.parked_sync_messages_by_server_seq.get_mut(&server_id) {
                while let Some(msg) = buffered.remove(&next_expected) {
                    ready_messages.push((next_expected, msg));
                    next_expected += 1;
                }

                if buffered.is_empty() {
                    remove_buffer = true;
                }
            }
            let mut last_applied = self
                .last_applied_server_seq
                .get(&server_id)
                .copied()
                .unwrap_or(next_expected.saturating_sub(1));
            for (sequence, msg) in ready_messages {
                if msg.payload.writes_storage() {
                    self.mark_storage_write_pending_flush();
                }
                self.push_sync_inbox(msg);
                applied_messages += 1;
                applied_since_last_tick = true;
                last_applied = sequence;
            }
            self.next_expected_server_seq
                .insert(server_id, next_expected);
            self.last_applied_server_seq.insert(server_id, last_applied);
            if remove_buffer {
                self.parked_sync_messages_by_server_seq.remove(&server_id);
            }
        }

        if applied_messages > 0 {
            debug!(count = applied_messages, "applied parked sync messages");
            if applied_since_last_tick {
                self.immediate_tick();
            }
        }
        applied_messages > 0
    }

    /// Check if there are outbound messages requiring a batched_tick.
    pub fn has_outbound(&self) -> bool {
        !self
            .schema_manager
            .query_manager()
            .sync_manager()
            .outbox()
            .is_empty()
    }

    /// Park a sync message for processing in next batched_tick.
    pub fn park_sync_message(&mut self, message: InboxEntry) {
        let _recv_span = debug_span!(
            "sync.recv",
            peer_kind = message.source.peer_kind(),
            peer_id = %message.source.peer_uuid(),
            payload = message.payload.variant_name(),
            payload_json = %serde_json::to_string(&message.payload).unwrap_or_default(),
            tier = self.tier_label,
        )
        .entered();
        if let Some((ref tracer, ref name)) = self.sync_tracer {
            tracer.record_incoming(&message.source, name, &message.payload);
        }
        self.parked_sync_messages.push(message);
        self.scheduler.schedule_batched_tick();
    }

    /// Park an entry the transport staging already admitted. The caller is the tick that is
    /// about to run (or `flush()`), so no tick is scheduled.
    pub fn park_staged_sync_message(&mut self, message: InboxEntry) {
        if let Some((ref tracer, ref name)) = self.sync_tracer {
            tracer.record_incoming(&message.source, name, &message.payload);
        }
        self.parked_sync_messages.push(message);
    }

    /// Park a sequenced sync message for in-order processing in next batched_tick.
    pub fn park_sync_message_with_sequence(&mut self, message: InboxEntry, sequence: u64) {
        match message.source {
            crate::sync_manager::Source::Server(server_id) => {
                let _recv_span = debug_span!(
                    "sync.recv",
                    peer_kind = "server",
                    peer_id = %server_id,
                    payload = message.payload.variant_name(),
                    payload_json = %serde_json::to_string(&message.payload).unwrap_or_default(),
                    sequence = sequence,
                    tier = self.tier_label,
                )
                .entered();
                let next_expected = self
                    .next_expected_server_seq
                    .entry(server_id)
                    .or_insert(sequence);
                if sequence < *next_expected {
                    trace!(
                        ?server_id,
                        sequence,
                        next_expected = *next_expected,
                        "dropping already-applied sequenced sync message"
                    );
                    return;
                }

                if let Some((ref tracer, ref name)) = self.sync_tracer {
                    tracer.record_incoming(&message.source, name, &message.payload);
                }

                self.parked_sync_messages_by_server_seq
                    .entry(server_id)
                    .or_default()
                    .insert(sequence, message);
                self.scheduler.schedule_batched_tick();
            }
            _ => self.park_sync_message(message),
        }
    }

    fn park_sync_message_batch(
        &mut self,
        entries: Vec<crate::transport_manager::SequencedInboxEntry>,
    ) {
        let mut parked_any = false;
        for crate::transport_manager::SequencedInboxEntry { entry, sequence } in entries {
            if let Some(sequence) = sequence {
                match entry.source {
                    crate::sync_manager::Source::Server(server_id) => {
                        let next_expected = self
                            .next_expected_server_seq
                            .entry(server_id)
                            .or_insert(sequence);
                        if sequence < *next_expected {
                            trace!(
                                ?server_id,
                                sequence,
                                next_expected = *next_expected,
                                "dropping already-applied sequenced sync message"
                            );
                            continue;
                        }

                        if let Some((ref tracer, ref name)) = self.sync_tracer {
                            tracer.record_incoming(&entry.source, name, &entry.payload);
                        }

                        self.parked_sync_messages_by_server_seq
                            .entry(server_id)
                            .or_default()
                            .insert(sequence, entry);
                        parked_any = true;
                    }
                    _ => {
                        self.parked_sync_messages.push(entry);
                        parked_any = true;
                    }
                }
            } else {
                if let Some((ref tracer, ref name)) = self.sync_tracer {
                    tracer.record_incoming(&entry.source, name, &entry.payload);
                }
                self.parked_sync_messages.push(entry);
                parked_any = true;
            }
        }

        if parked_any {
            self.scheduler.schedule_batched_tick();
        }
    }

    /// Set the next expected sequenced message for a server stream.
    pub fn set_next_expected_server_sequence(&mut self, server_id: ServerId, next_sequence: u64) {
        let next_sequence = next_sequence.max(1);
        self.next_expected_server_seq
            .insert(server_id, next_sequence);
        self.last_applied_server_seq
            .insert(server_id, next_sequence.saturating_sub(1));
        if let Some(buffered) = self.parked_sync_messages_by_server_seq.get_mut(&server_id) {
            buffered.retain(|seq, _| *seq >= next_sequence);
        }
    }

    /// Test seam: directly dispatch a `TransportInbound` event as if it arrived
    /// from `server_id`, exercising the same match arm as `batched_tick`.
    #[cfg(test)]
    #[cfg(feature = "transport-websocket")]
    pub(crate) fn handle_transport_inbound_for_test(
        &mut self,
        server_id: ServerId,
        event: crate::transport_manager::TransportInbound,
    ) {
        let mut released_server_hold = false;
        match event {
            crate::transport_manager::TransportInbound::Connected {
                catalogue_state_hash,
                next_sync_seq,
                supports_delivery_acks,
            } => {
                self.schema_manager
                    .query_manager_mut()
                    .sync_manager_mut()
                    .set_upstream_supports_delivery_acks(supports_delivery_acks);
                self.remove_server(server_id);
                self.add_server_with_catalogue_state_hash_and_permission(
                    server_id,
                    catalogue_state_hash.as_deref(),
                    false,
                );
                if let Some(seq) = next_sync_seq {
                    self.set_next_expected_server_sequence(server_id, seq);
                }
            }
            crate::transport_manager::TransportInbound::Sync { entry, sequence } => {
                if let Some(seq) = sequence {
                    self.park_sync_message_with_sequence(*entry, seq);
                } else {
                    self.park_sync_message(*entry);
                }
            }
            crate::transport_manager::TransportInbound::SyncBatch { entries } => {
                self.park_sync_message_batch(entries);
            }
            crate::transport_manager::TransportInbound::Disconnected => {
                self.remove_server(server_id);
                self.schema_manager
                    .query_manager_mut()
                    .sync_manager_mut()
                    .add_pending_server(server_id);
            }
            crate::transport_manager::TransportInbound::ConnectFailed { reason } => {
                debug!(%reason, "transport connect failed; releasing pending-server hold");
                self.schema_manager
                    .query_manager_mut()
                    .sync_manager_mut()
                    .remove_pending_server(server_id);
                released_server_hold = true;
            }
            crate::transport_manager::TransportInbound::AuthFailure { reason } => {
                self.remove_server(server_id);
                released_server_hold = true;
                if let Some(ref cb) = self.auth_failure_callback {
                    cb(reason);
                }
            }
        }

        if released_server_hold {
            self.immediate_tick();
        }
    }

    pub fn local_batch_replay_payloads(
        &self,
        batch_id: crate::row_histories::BatchId,
    ) -> Vec<SyncPayload> {
        let local_rows = self.local_batch_rows(batch_id, LocalBatchLookup::Retransmit);
        let mut payloads = local_rows
            .iter()
            .map(|(member, row_locator, row)| SyncPayload::RowBatchCreated {
                metadata: Some(RowMetadata {
                    id: member.object_id,
                    metadata: metadata_from_row_locator(row_locator),
                }),
                row: row.clone(),
            })
            .collect::<Vec<_>>();

        if let Some(submission) =
            Self::direct_sealed_submission_from_local_batch_rows(batch_id, &local_rows)
        {
            payloads.push(SyncPayload::SealBatch { submission });
        }

        payloads
    }
}
