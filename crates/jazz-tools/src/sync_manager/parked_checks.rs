//! Write checks denied only for want of a row of their own batch.
//!
//! The rows of an explicit batch stay staged until the batch settles, so a policy that
//! leans on a sibling (an attachment on the message written with it) finds nothing and
//! denies, and the denial rejects the whole batch. Such a denial is parked here instead.
//! The batch is decided once its seal and every member it declares are in: the parked
//! checks are evaluated again against the batch's own approved rows
//! (`QueryManager::decide_parked_batches`), and whatever still fails rejects the batch as
//! the first denial would have.
//!
//! Parked checks live in memory only. A restart or a reap loses them, and the seal they
//! leave incomplete asks the client for its rows again.
//!
//! A parked update keeps the old row it was queued with: the row it changes is decided as
//! of the write's arrival, as every check was before batches waited, and only the rows of
//! its own batch are read as of the decision.

use super::*;
use crate::row_histories::{BatchId, StoredRowBatch};
use std::collections::BTreeMap;

/// How much may wait, and for how long.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ParkingBounds {
    /// Past any of these three a denial is rejected at once, as it was before batches
    /// waited.
    pub(crate) max_checks_per_batch: usize,
    pub(crate) max_batches_per_writer: usize,
    pub(crate) max_bytes_per_writer: usize,
    /// How long a sealed batch holds its writer's later writes while a member has not
    /// arrived. Counted from the first pass that saw its seal, and never restarted by a
    /// retransmit. Past it the writes go on, and the batch keeps what it has parked.
    pub(crate) sealed_deadline_micros: u64,
    /// How long a parked batch is kept after the last of its rows arrived. A sealed one
    /// is then asked for again; an unsealed one is dropped silently, since a rolled-back
    /// batch tells the server nothing.
    pub(crate) idle_ttl_micros: u64,
}

impl Default for ParkingBounds {
    fn default() -> Self {
        Self {
            max_checks_per_batch: 256,
            max_batches_per_writer: 64,
            max_bytes_per_writer: 16 << 20,
            sealed_deadline_micros: 30_000_000,
            idle_ttl_micros: 120_000_000,
        }
    }
}

/// Who a batch's waiting is charged to: one session on one connection. A client id can
/// change its session in place, and one user's waiting batch must not hold another's
/// writes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct ParkingWriter {
    pub(crate) client_id: ClientId,
    pub(crate) user_id: String,
}

impl ParkingWriter {
    pub(crate) fn of(check: &PendingPermissionCheck) -> Self {
        Self {
            client_id: check.client_id,
            user_id: check.session.user_id.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ParkedCheck {
    pub(crate) check: PendingPermissionCheck,
    pub(crate) code: String,
    pub(crate) reason: String,
    bytes: usize,
}

impl ParkedCheck {
    pub(crate) fn row(&self) -> Option<&StoredRowBatch> {
        check_row(&self.check)
    }

    pub(crate) fn table(&self) -> Option<&str> {
        check_table(&self.check)
    }
}

fn check_table(check: &PendingPermissionCheck) -> Option<&str> {
    check
        .metadata
        .get(crate::metadata::MetadataKey::Table.as_str())
        .map(String::as_str)
}

fn check_row(check: &PendingPermissionCheck) -> Option<&StoredRowBatch> {
    match &check.payload {
        SyncPayload::RowBatchCreated { row, .. } | SyncPayload::RowBatchNeeded { row, .. } => {
            Some(row)
        }
        _ => None,
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ParkedBatch {
    pub(crate) batch_id: BatchId,
    first_parked_at: u64,
    /// The pass that first found the batch's seal stored. A sealed batch holds its
    /// writer's later writes, and sealed batches are decided in this order.
    seal_seen_at: Option<u64>,
    /// When the last row of the batch arrived, parked or approved.
    last_arrival_at: u64,
    /// Something that can change the decision happened since the batch was last decided:
    /// a row of it arrived, or its seal did.
    pub(crate) dirty: bool,
    pub(crate) checks: BTreeMap<ObjectId, ParkedCheck>,
}

impl ParkedBatch {
    pub(crate) fn is_sealed(&self) -> bool {
        self.seal_seen_at.is_some()
    }

    /// Whether the batch still holds its writer's later writes: sealed, and inside its
    /// deadline.
    pub(crate) fn holds_writer(&self, now: u64, bounds: &ParkingBounds) -> bool {
        self.seal_seen_at
            .is_some_and(|seen| now.saturating_sub(seen) <= bounds.sealed_deadline_micros)
    }

    pub(crate) fn idle_past_ttl(&self, now: u64, bounds: &ParkingBounds) -> bool {
        now.saturating_sub(self.last_arrival_at) > bounds.idle_ttl_micros
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ParkedWriter {
    batches: Vec<ParkedBatch>,
    bytes: usize,
}

impl SyncManager {
    pub(crate) fn now_micros_for_parking(&mut self) -> u64 {
        self.clock.reserve_timestamp()
    }

    pub(crate) fn parking_bounds(&self) -> ParkingBounds {
        self.parking_bounds
    }

    #[cfg(test)]
    pub(crate) fn set_parking_bounds(&mut self, bounds: ParkingBounds) {
        self.parking_bounds = bounds;
    }

    /// Takes a trusted writer's rows in as sent, as an engine from before
    /// `trusted_client_row_payload` did, or as this one does once upgraded.
    #[cfg(test)]
    pub(crate) fn take_trusted_rows_as_sent(&mut self, as_sent: bool) {
        self.trusted_rows_as_sent = as_sent;
    }

    /// As if every `Missing` this node sent was sent longer ago than its rate limit.
    #[cfg(test)]
    pub(crate) fn forget_missing_answers(&mut self) {
        self.missing_answers.clear();
    }

    /// Park a check its policy denied for want of a sibling. `sealed` says whether the
    /// batch's seal is stored. Hands the check back when a bound is reached: the caller
    /// rejects it as before.
    pub(crate) fn park_permission_check(
        &mut self,
        check: PendingPermissionCheck,
        batch_id: BatchId,
        sealed: bool,
        code: String,
        reason: String,
    ) -> Result<(), Box<PendingPermissionCheck>> {
        let Some(object_id) = check.payload.object_id() else {
            return Err(Box::new(check));
        };
        let writer_key = ParkingWriter::of(&check);
        let now = self.clock.reserve_timestamp();
        let bounds = self.parking_bounds;
        let bytes = check.new_content.as_ref().map_or(0, Vec::len)
            + check.old_content.as_ref().map_or(0, Vec::len)
            + match &check.payload {
                SyncPayload::RowBatchCreated { row, .. }
                | SyncPayload::RowBatchNeeded { row, .. } => row.data.len(),
                _ => 0,
            };
        if let Some(owner) = self.parked_batch_owners.get(&batch_id)
            && *owner != writer_key
        {
            return Err(Box::new(check));
        }
        // A row is in one table: the same id parked under another one refuses the batch.
        if self
            .parked_batch(batch_id)
            .and_then(|batch| batch.checks.get(&object_id))
            .is_some_and(|parked| parked.table() != check_table(&check))
        {
            return Err(Box::new(check));
        }
        let writer = self.parked_checks.entry(writer_key.clone()).or_default();
        let replaced_bytes = writer
            .batches
            .iter()
            .find(|batch| batch.batch_id == batch_id)
            .and_then(|batch| batch.checks.get(&object_id))
            .map_or(0, |parked| parked.bytes);
        let batch_exists = writer
            .batches
            .iter()
            .any(|batch| batch.batch_id == batch_id);
        let over_bytes = writer.bytes - replaced_bytes + bytes > bounds.max_bytes_per_writer;
        let over_batches = !batch_exists && writer.batches.len() >= bounds.max_batches_per_writer;
        let over_checks = writer
            .batches
            .iter()
            .find(|batch| batch.batch_id == batch_id)
            .is_some_and(|batch| {
                replaced_bytes == 0
                    && !batch.checks.contains_key(&object_id)
                    && batch.checks.len() >= bounds.max_checks_per_batch
            });
        if over_bytes || over_batches || over_checks {
            if writer.batches.is_empty() {
                self.parked_checks.remove(&writer_key);
            }
            return Err(Box::new(check));
        }
        if !batch_exists {
            writer.batches.push(ParkedBatch {
                batch_id,
                first_parked_at: now,
                seal_seen_at: None,
                last_arrival_at: now,
                dirty: true,
                checks: BTreeMap::new(),
            });
        }
        let batch = writer
            .batches
            .iter_mut()
            .find(|batch| batch.batch_id == batch_id)
            .expect("the batch was found or pushed above");
        if sealed && batch.seal_seen_at.is_none() {
            batch.seal_seen_at = Some(now);
        }
        batch.last_arrival_at = now;
        batch.dirty = true;
        batch.checks.insert(
            object_id,
            ParkedCheck {
                check,
                code,
                reason,
                bytes,
            },
        );
        writer.bytes = writer.bytes - replaced_bytes + bytes;
        self.parked_batch_owners.insert(batch_id, writer_key);
        Ok(())
    }

    /// Whether `batch_id` has a parked batch at all (it may hold no check while one of
    /// them is being evaluated again).
    pub(crate) fn has_parked_batch(&self, batch_id: BatchId) -> bool {
        self.parked_batch_owners.contains_key(&batch_id)
    }

    pub(crate) fn is_parked(&self, batch_id: BatchId, object_id: ObjectId) -> bool {
        self.parked_batch(batch_id)
            .is_some_and(|batch| batch.checks.contains_key(&object_id))
    }

    pub(crate) fn parked_batch(&self, batch_id: BatchId) -> Option<&ParkedBatch> {
        let owner = self.parked_batch_owners.get(&batch_id)?;
        self.parked_checks
            .get(owner)?
            .batches
            .iter()
            .find(|batch| batch.batch_id == batch_id)
    }

    fn parked_batch_mut(&mut self, batch_id: BatchId) -> Option<&mut ParkedBatch> {
        let owner = self.parked_batch_owners.get(&batch_id)?;
        self.parked_checks
            .get_mut(owner)?
            .batches
            .iter_mut()
            .find(|batch| batch.batch_id == batch_id)
    }

    /// Take one parked check out to evaluate it again. The batch stays parked, so a
    /// check denied again goes back into it.
    pub(crate) fn take_parked_check(
        &mut self,
        batch_id: BatchId,
        object_id: ObjectId,
    ) -> Option<ParkedCheck> {
        let owner = self.parked_batch_owners.get(&batch_id)?.clone();
        let writer = self.parked_checks.get_mut(&owner)?;
        let batch = writer
            .batches
            .iter_mut()
            .find(|batch| batch.batch_id == batch_id)?;
        let parked = batch.checks.remove(&object_id)?;
        writer.bytes -= parked.bytes;
        Some(parked)
    }

    pub(crate) fn mark_parked_batch_sealed(&mut self, batch_id: BatchId, now: u64) {
        if let Some(batch) = self.parked_batch_mut(batch_id)
            && batch.seal_seen_at.is_none()
        {
            batch.seal_seen_at = Some(now);
            batch.last_arrival_at = now;
            batch.dirty = true;
        }
    }

    /// A seal of a parked batch was stored, the first or a later one.
    pub(super) fn note_parked_batch_seal_stored(&mut self, batch_id: BatchId) {
        if !self.has_parked_batch(batch_id) {
            return;
        }
        let now = self.clock.reserve_timestamp();
        if let Some(batch) = self.parked_batch_mut(batch_id) {
            batch.seal_seen_at.get_or_insert(now);
            batch.last_arrival_at = now;
            batch.dirty = true;
        }
    }

    pub(crate) fn mark_parked_batch_decided(&mut self, batch_id: BatchId) {
        if let Some(batch) = self.parked_batch_mut(batch_id) {
            batch.dirty = false;
        }
    }

    /// A row of a parked batch was approved: the batch may be complete now. An earlier
    /// version of the same row parked for it is superseded, since the seal declares one
    /// version of each row; a parked row of that id in another table stays, and refuses
    /// the batch when it is decided.
    pub(super) fn note_parked_batch_row_approved(&mut self, check: &PendingPermissionCheck) {
        let Some(row) = check_row(check) else {
            return;
        };
        let (batch_id, object_id) = (row.batch_id, row.row_id);
        if !self.has_parked_batch(batch_id) {
            return;
        }
        let now = self.clock.reserve_timestamp();
        let superseded = self
            .parked_batch(batch_id)
            .and_then(|batch| batch.checks.get(&object_id))
            .is_some_and(|parked| parked.table() == check_table(check));
        if superseded {
            self.take_parked_check(batch_id, object_id);
        }
        if let Some(batch) = self.parked_batch_mut(batch_id) {
            batch.last_arrival_at = now;
            batch.dirty = true;
        }
    }

    /// Let go of a parked batch whose rows stopped arriving. If its seal is stored, the
    /// client is asked for the batch again, as for any seal left incomplete; its resend
    /// parks the rows anew.
    pub(crate) fn let_go_idle_parked_batch(&mut self, batch_id: BatchId) {
        let Some(batch) = self.parked_batch(batch_id) else {
            return;
        };
        let sealed = batch.is_sealed();
        let client_id = self
            .parked_batch_owners
            .get(&batch_id)
            .map(|owner| owner.client_id);
        self.drop_parked_batch(batch_id);
        if sealed
            && let Some(client_id) = client_id
            && self.may_tell_client_a_batch_is_missing(client_id, batch_id)
        {
            self.queue_batch_fate_to_client_unfiltered(
                client_id,
                crate::batch_fate::BatchFate::Missing { batch_id },
            );
        }
    }

    /// A parked batch whose every row was approved: it is no longer parked, and its seal
    /// settles it now, as it would have without the wait.
    pub(crate) fn settle_decided_parked_batch<H: Storage>(
        &mut self,
        storage: &mut H,
        batch_id: BatchId,
    ) {
        let client_id = self
            .parked_batch_owners
            .get(&batch_id)
            .map(|owner| owner.client_id);
        self.drop_parked_batch(batch_id);
        if let Some(client_id) = client_id {
            self.try_accept_completed_sealed_batch_from_client(storage, client_id, batch_id);
        }
    }

    /// Forget a parked batch and every check it still holds.
    pub(crate) fn drop_parked_batch(&mut self, batch_id: BatchId) {
        let Some(owner) = self.parked_batch_owners.remove(&batch_id) else {
            return;
        };
        let Some(writer) = self.parked_checks.get_mut(&owner) else {
            return;
        };
        if let Some(index) = writer
            .batches
            .iter()
            .position(|batch| batch.batch_id == batch_id)
        {
            let batch = writer.batches.remove(index);
            writer.bytes -= batch
                .checks
                .values()
                .map(|parked| parked.bytes)
                .sum::<usize>();
        }
        if writer.batches.is_empty() {
            self.parked_checks.remove(&owner);
        }
    }

    pub(super) fn drop_parked_checks_of_client(&mut self, client_id: ClientId) {
        let writers = self
            .parked_checks
            .keys()
            .filter(|writer| writer.client_id == client_id)
            .cloned()
            .collect::<Vec<_>>();
        for writer in writers {
            if let Some(parked) = self.parked_checks.remove(&writer) {
                for batch in parked.batches {
                    self.parked_batch_owners.remove(&batch.batch_id);
                }
            }
        }
    }

    pub(crate) fn has_parked_checks(&self) -> bool {
        !self.parked_checks.is_empty()
    }

    /// Every writer with a parked batch, the one that parked first first, and its parked
    /// batches in the order they were first parked.
    pub(crate) fn parked_writers(&self) -> Vec<(ParkingWriter, Vec<BatchId>)> {
        let mut writers = self
            .parked_checks
            .iter()
            .map(|(writer, parked)| {
                let first = parked
                    .batches
                    .iter()
                    .map(|batch| (batch.first_parked_at, batch.batch_id))
                    .min();
                (
                    first,
                    writer.clone(),
                    parked
                        .batches
                        .iter()
                        .map(|batch| batch.batch_id)
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();
        writers.sort_by(|left, right| left.0.cmp(&right.0));
        writers
            .into_iter()
            .map(|(_, writer, batches)| (writer, batches))
            .collect()
    }

    /// A writer's sealed parked batches, in the order their seals were seen.
    pub(crate) fn sealed_parked_batches(&self, writer: &ParkingWriter) -> Vec<BatchId> {
        let Some(parked) = self.parked_checks.get(writer) else {
            return Vec::new();
        };
        let mut sealed = parked
            .batches
            .iter()
            .filter_map(|batch| batch.seal_seen_at.map(|seen| (seen, batch.batch_id)))
            .collect::<Vec<_>>();
        sealed.sort();
        sealed.into_iter().map(|(_, batch_id)| batch_id).collect()
    }

    /// Whether a check of `batch_id` from `writer` has to wait: its writer has a sealed
    /// batch still being decided inside its deadline, and this check is not of a sealed
    /// parked batch. Anything written after that batch was committed may lean on it.
    pub(crate) fn parked_batch_holds_check(
        &self,
        writer: &ParkingWriter,
        batch_id: Option<BatchId>,
        now: u64,
    ) -> bool {
        let Some(parked) = self.parked_checks.get(writer) else {
            return false;
        };
        let own_batch_is_sealed = batch_id.is_some_and(|batch_id| {
            parked
                .batches
                .iter()
                .any(|batch| batch.batch_id == batch_id && batch.is_sealed())
        });
        !own_batch_is_sealed
            && parked
                .batches
                .iter()
                .any(|batch| batch.holds_writer(now, &self.parking_bounds))
    }

    /// The checks a pass took from the queue and has not evaluated yet. They are on their
    /// way as much as the queued ones: the approval of the first row of a drain must not
    /// ask for the second. Counted, because each version of a row its batch wrote more than
    /// once is a check of its own: the first one to leave must not take the row's place
    /// from the version still on its way, which is the one the seal declares.
    pub(crate) fn note_checks_in_flight(&mut self, checks: &[PendingPermissionCheck]) {
        self.checks_in_flight.clear();
        for check in checks {
            if let SyncPayload::RowBatchCreated { row, .. }
            | SyncPayload::RowBatchNeeded { row, .. } = &check.payload
            {
                *self
                    .checks_in_flight
                    .entry((check.client_id, row.batch_id, row.row_id))
                    .or_insert(0) += 1;
            }
        }
    }

    pub(crate) fn check_leaves_flight(&mut self, check: &PendingPermissionCheck) {
        if let SyncPayload::RowBatchCreated { row, .. } | SyncPayload::RowBatchNeeded { row, .. } =
            &check.payload
        {
            let key = (check.client_id, row.batch_id, row.row_id);
            if let Some(count) = self.checks_in_flight.get_mut(&key) {
                *count -= 1;
                if *count == 0 {
                    self.checks_in_flight.remove(&key);
                }
            }
        }
    }

    /// Whether every check a pass noted has left it.
    pub(crate) fn no_checks_in_flight(&self) -> bool {
        self.checks_in_flight.is_empty()
    }

    /// Whether rows of `batch_id` from this client are here and not decided yet: queued,
    /// in the pass under way, or parked. Its fate is on its way with them.
    pub(super) fn batch_rows_are_waiting(&self, client_id: ClientId, batch_id: BatchId) -> bool {
        self.has_parked_batch(batch_id)
            || self
                .checks_in_flight
                .keys()
                .any(|(client, batch, _)| *client == client_id && *batch == batch_id)
            || self.pending_permission_checks.iter().any(|pending| {
                pending.client_id == client_id
                    && matches!(
                        &pending.payload,
                        SyncPayload::RowBatchCreated { row, .. }
                            | SyncPayload::RowBatchNeeded { row, .. }
                            if row.batch_id == batch_id
                    )
            })
    }

    /// Whether every member a seal declares but the store does not hold yet is on its
    /// way: parked for the batch, or queued for its check. Such a seal is decided when
    /// they are, so it is not answered `Missing`.
    pub(super) fn absent_members_are_waiting(
        &self,
        client_id: ClientId,
        batch_id: BatchId,
        absent: &[ObjectId],
    ) -> bool {
        !absent.is_empty()
            && absent.iter().all(|object_id| {
                self.is_parked(batch_id, *object_id)
                    || self
                        .checks_in_flight
                        .contains_key(&(client_id, batch_id, *object_id))
                    || self.pending_permission_checks.iter().any(|pending| {
                        pending.client_id == client_id
                            && matches!(
                                &pending.payload,
                                SyncPayload::RowBatchCreated { row, .. }
                                    | SyncPayload::RowBatchNeeded { row, .. }
                                    if row.batch_id == batch_id && row.row_id == *object_id
                            )
                    })
            })
    }
}
