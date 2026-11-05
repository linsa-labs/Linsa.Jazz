//! A batch whose rows lean on each other.
//!
//! The rows of an explicit batch stay staged until the batch settles, and a write check
//! reads only visible rows. So a child written in the same batch as its parent (an
//! attachment and its message, a membership and the chat it joins) is denied, and with
//! it the whole batch, while the same two rows written one batch each are accepted.
//!
//! Such a denial is parked instead of rejected (`deny_write_by_policy`). Once the
//! batch's seal and every row it declares are in, the parked checks are evaluated again
//! with the batch's approved rows visible to the policy (`decide_parked_batches`). Each
//! approval joins what the next check sees, so every row accepted this way was allowed
//! at its turn by the rows accepted before it — never by a loop of rows vouching for
//! each other. Whatever is still denied rejects the batch as the first denial would
//! have, so a batch accepted before is decided the same way.

use std::collections::{HashMap, HashSet};

use crate::batch_fate::{BatchWriter, SealedBatchSubmission};
use crate::object::{BranchName, ObjectId};
use crate::query_manager::policy::OUTER_ROW_SESSION_PREFIX;
use crate::row_histories::{BatchId, RowState, StoredRowBatch};
use crate::storage::Storage;
use crate::sync_manager::{PendingPermissionCheck, RowBatchKey, SyncManager, SyncPayload};

use super::manager::{BatchedSubscriptionVisibilityEffects, QueryManager};
use super::policy::{CmpOp, Operation, PolicyExpr, PolicyValue};
use super::server_queries::WriteCheckOutcome;
use super::types::{Schema, SchemaHash, TableName};

/// How many times the parked checks of one batch are evaluated again. A batch whose
/// rows lean on each other in a chain needs one pass per link; Linsa's deepest is four.
const MAX_PARKED_PASSES: usize = 8;

/// How deep an inherited policy is followed when deciding whether a denial may wait.
const MAX_INHERITS_DEPTH: usize = 8;

/// A row of the batch, as its policy reads it when another row of the batch is evaluated
/// again.
#[derive(Debug, Clone)]
pub(crate) struct OverlayMember {
    pub(crate) table_name: String,
    pub(crate) schema_hash: SchemaHash,
    pub(crate) branch_name: BranchName,
}

/// The approved, declared rows of one batch, for the re-evaluation of one of its parked
/// checks: never the evaluated row itself.
///
/// The evaluated row is a member only when the same content is both stored and parked: it
/// was approved in a fixed point that ended waiting on a check deferred for its schema,
/// and was sent again in another state. Shown to its own check, an `idIsFree` would find
/// the row itself and refuse an honest batch; leaving it out reads its id as it was before
/// the batch.
#[derive(Debug, Clone)]
pub(crate) struct BatchOverlay {
    pub(crate) batch_id: BatchId,
    pub(crate) members: HashMap<ObjectId, OverlayMember>,
    pub(crate) evaluated: ObjectId,
}

impl BatchOverlay {
    /// The rows a scan on `branch` sees, as the query graph's overlay wants them.
    pub(crate) fn row_keys(&self, branch: &str) -> HashMap<ObjectId, RowBatchKey> {
        self.members
            .iter()
            .filter(|(object_id, member)| {
                **object_id != self.evaluated && member.branch_name.as_str() == branch
            })
            .map(|(object_id, member)| {
                (
                    *object_id,
                    RowBatchKey {
                        row_id: *object_id,
                        branch_name: member.branch_name,
                        batch_id: self.batch_id,
                    },
                )
            })
            .collect()
    }

    pub(crate) fn member(&self, object_id: ObjectId) -> Option<&OverlayMember> {
        (object_id != self.evaluated)
            .then(|| self.members.get(&object_id))
            .flatten()
    }
}

/// What became of a parked batch in one decision pass.
enum ParkedBatchDecision {
    Decided,
    /// Not complete yet, or a check of it waits for its schema: the writer's later
    /// sealed batches wait behind it while it holds her writes.
    Waiting,
}

/// Whether every row `policy` reads goes through the evaluator that sees the batch's
/// rows. A policy that reads anywhere else would see two different stores, and under a
/// negation that can admit a row no order of writes admits; its denial is not parked.
///
/// Covered: comparisons; `Exists` whose condition is comparisons only (a nested one is
/// evaluated by a fresh evaluator that does not see the batch); a negation of
/// comparisons, or of an `Exists` on the row's own table by the row's own id (the batch
/// never shows the evaluated row, so that negation reads what it read before; under an
/// `Inherits` the own row is the parent, and the negation reads the very version the
/// loader returned for it); and `Inherits`, whose parent policy runs on the same
/// evaluator. Not covered: `ExistsRel`
/// (its subqueries settle on their own) and `InheritsReferencing` (it reads the index
/// directly).
pub(crate) fn policy_reads_through_batch_overlay(
    policy: &PolicyExpr,
    own_table: TableName,
    schema: &Schema,
) -> bool {
    covered(policy, own_table, schema, 0, &mut HashSet::new())
}

/// Whether the denying policy and every write policy of its table read through the batch.
/// A re-evaluation runs all the policies of the write, not only the one that denied: an
/// update denied by its USING is judged by its WITH CHECK too, and a check queued as an
/// insert can be judged as an update once the old row is found.
fn write_policies_read_through_batch_overlay(
    denying: &PolicyExpr,
    own_table: TableName,
    schema: &Schema,
) -> bool {
    let Some(table) = schema.get(&own_table) else {
        return false;
    };
    let policies = &table.policies;
    [
        Some(denying),
        policies.insert_policy(),
        policies.update_using_policy(),
        policies.update_check_policy(),
        policies.effective_delete_using(),
    ]
    .into_iter()
    .flatten()
    .all(|policy| policy_reads_through_batch_overlay(policy, own_table, schema))
}

fn covered(
    expr: &PolicyExpr,
    own_table: TableName,
    schema: &Schema,
    depth: usize,
    inherited: &mut HashSet<(TableName, Operation)>,
) -> bool {
    match expr {
        PolicyExpr::And(exprs) | PolicyExpr::Or(exprs) => exprs
            .iter()
            .all(|expr| covered(expr, own_table, schema, depth, inherited)),
        PolicyExpr::Not(inner) => is_flat(inner) || is_own_id_lookup(inner, own_table),
        PolicyExpr::Exists { condition, .. } => is_flat(condition),
        PolicyExpr::ExistsRel { .. } | PolicyExpr::InheritsReferencing { .. } => false,
        PolicyExpr::Inherits {
            operation,
            via_column,
            ..
        } => {
            if depth >= MAX_INHERITS_DEPTH {
                return false;
            }
            let Some(parent_table) = schema
                .get(&own_table)
                .and_then(|table| table.columns.column(via_column))
                .and_then(|column| column.references)
            else {
                return true;
            };
            let Some(parent) = schema.get(&parent_table) else {
                return true;
            };
            // A policy already being followed: the evaluator stops there and denies.
            if !inherited.insert((parent_table, *operation)) {
                return true;
            }
            let parent_policy = match operation {
                Operation::Select => parent.policies.select_policy(),
                Operation::Insert => parent.policies.insert_policy(),
                Operation::Update => parent.policies.update_using_policy(),
                Operation::Delete => parent.policies.effective_delete_using(),
            };
            parent_policy
                .is_none_or(|policy| covered(policy, parent_table, schema, depth + 1, inherited))
        }
        leaf => is_flat(leaf),
    }
}

/// No row read at all: comparisons on the row and the session only.
fn is_flat(expr: &PolicyExpr) -> bool {
    match expr {
        PolicyExpr::Exists { .. }
        | PolicyExpr::ExistsRel { .. }
        | PolicyExpr::Inherits { .. }
        | PolicyExpr::InheritsReferencing { .. } => false,
        PolicyExpr::And(exprs) | PolicyExpr::Or(exprs) => exprs.iter().all(is_flat),
        PolicyExpr::Not(inner) => is_flat(inner),
        _ => true,
    }
}

/// `Exists(own table where id = the row's id AND …)`: `idIsFree`, `idIsFreeOrOwn`.
fn is_own_id_lookup(expr: &PolicyExpr, own_table: TableName) -> bool {
    let PolicyExpr::Exists { table, condition } = expr else {
        return false;
    };
    if table.as_str() != own_table.as_str() || !is_flat(condition) {
        return false;
    }
    let is_own_id = |expr: &PolicyExpr| {
        matches!(
            expr,
            PolicyExpr::Cmp { column, op: CmpOp::Eq, value: PolicyValue::SessionRef(path) }
                if column == "id"
                    && path.len() == 2
                    && path[0] == OUTER_ROW_SESSION_PREFIX
                    && path[1] == "id"
        )
    };
    match condition.as_ref() {
        PolicyExpr::And(exprs) => exprs.iter().any(is_own_id),
        other => is_own_id(other),
    }
}

fn payload_row(check: &PendingPermissionCheck) -> Option<&StoredRowBatch> {
    match &check.payload {
        SyncPayload::RowBatchCreated { row, .. } | SyncPayload::RowBatchNeeded { row, .. } => {
            Some(row)
        }
        _ => None,
    }
}

fn declares(submission: &SealedBatchSubmission, row: &StoredRowBatch) -> bool {
    row.branch.as_str() == submission.target_branch_name.as_str()
        && submission.members.iter().any(|member| {
            member.object_id == row.row_id
                && (row.content_digest() == member.row_digest
                    || row.content_digest_ignoring_parents() == member.row_digest)
        })
}

impl QueryManager {
    /// A write check its policy denied. Parked when the denial may only be for want of a
    /// row of its own batch; rejected as before otherwise.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn deny_write_by_policy<H: Storage>(
        &mut self,
        storage: &mut H,
        check: PendingPermissionCheck,
        reason: String,
        policy: &PolicyExpr,
        auth_schema: &Schema,
        auth_table: TableName,
    ) {
        let parked = match self.parking_place(storage, &check, policy, auth_schema, auth_table) {
            Some((batch_id, sealed)) => self.sync_manager.park_permission_check(
                check,
                batch_id,
                sealed,
                "permission_denied".to_string(),
                reason.clone(),
            ),
            None => Err(Box::new(check)),
        };
        if let Err(check) = parked {
            self.sync_manager
                .reject_permission_check(storage, *check, reason);
        }
    }

    /// The batch a denied check may wait in, and whether its seal is stored.
    fn parking_place<H: Storage>(
        &self,
        storage: &H,
        check: &PendingPermissionCheck,
        policy: &PolicyExpr,
        auth_schema: &Schema,
        auth_table: TableName,
    ) -> Option<(BatchId, bool)> {
        let row = payload_row(check)?;
        let batch_id = row.batch_id;
        if !matches!(storage.load_authoritative_batch_fate(batch_id), Ok(None)) {
            return None;
        }
        let seal = storage
            .load_sealed_batch_submission(batch_id)
            .ok()
            .flatten();
        let eligible = match row.state {
            RowState::StagingPending => true,
            // A direct batch's rows are published at commit, so a resend after it (offline,
            // `Missing`, a restart) carries them visible. Its stored seal tells them from a
            // write made outside any batch.
            RowState::VisibleDirect => seal.as_ref().is_some_and(|seal| declares(seal, row)),
            _ => false,
        };
        if !eligible
            || !SyncManager::holds_batch(
                storage,
                batch_id,
                &BatchWriter::User(check.session.user_id.clone()),
            )
            || !write_policies_read_through_batch_overlay(policy, auth_table, auth_schema)
        {
            return None;
        }
        Some((batch_id, seal.is_some()))
    }

    /// Forget batches that no longer need deciding, let go of the ones whose rows stopped
    /// arriving, and note a seal stored without passing the inbox. Runs before a pass, and
    /// returns the time it judged by, if anything is parked.
    pub(super) fn refresh_parked_batches<H: Storage>(&mut self, storage: &H) -> Option<u64> {
        if !self.sync_manager.has_parked_checks() {
            return None;
        }
        let now = self.sync_manager.now_micros_for_parking();
        let bounds = self.sync_manager.parking_bounds();
        for (_, batch_ids) in self.sync_manager.parked_writers() {
            for batch_id in batch_ids {
                if !matches!(storage.load_authoritative_batch_fate(batch_id), Ok(None)) {
                    self.sync_manager.drop_parked_batch(batch_id);
                    continue;
                }
                let Some(batch) = self.sync_manager.parked_batch(batch_id) else {
                    continue;
                };
                if batch.idle_past_ttl(now, &bounds) {
                    self.sync_manager.let_go_idle_parked_batch(batch_id);
                } else if !batch.is_sealed()
                    && matches!(storage.load_sealed_batch_submission(batch_id), Ok(Some(_)))
                {
                    self.sync_manager.mark_parked_batch_sealed(batch_id, now);
                }
            }
        }
        Some(now)
    }

    /// Decide the sealed parked batches something happened to. A writer's sealed batches
    /// are decided in the order their seals came, each only after the one before it, as
    /// long as that one still holds her writes.
    pub(super) fn decide_parked_batches<H: Storage>(
        &mut self,
        storage: &mut H,
        now: Option<u64>,
        effects: &mut BatchedSubscriptionVisibilityEffects,
    ) {
        if !self.sync_manager.has_parked_checks() {
            return;
        }
        let now = match now {
            Some(now) => now,
            None => self.sync_manager.now_micros_for_parking(),
        };
        let bounds = self.sync_manager.parking_bounds();
        for (writer, _) in self.sync_manager.parked_writers() {
            for batch_id in self.sync_manager.sealed_parked_batches(&writer) {
                if self
                    .sync_manager
                    .parked_batch(batch_id)
                    .is_some_and(|batch| batch.dirty)
                {
                    self.sync_manager.mark_parked_batch_decided(batch_id);
                    if let ParkedBatchDecision::Decided =
                        self.decide_parked_batch(storage, batch_id, effects)
                    {
                        continue;
                    }
                }
                if self
                    .sync_manager
                    .parked_batch(batch_id)
                    .is_some_and(|batch| batch.holds_writer(now, &bounds))
                {
                    break;
                }
            }
        }
    }

    fn decide_parked_batch<H: Storage>(
        &mut self,
        storage: &mut H,
        batch_id: BatchId,
        effects: &mut BatchedSubscriptionVisibilityEffects,
    ) -> ParkedBatchDecision {
        if !matches!(storage.load_authoritative_batch_fate(batch_id), Ok(None)) {
            self.sync_manager.drop_parked_batch(batch_id);
            return ParkedBatchDecision::Decided;
        }
        let Some(seal) = storage
            .load_sealed_batch_submission(batch_id)
            .ok()
            .flatten()
        else {
            return ParkedBatchDecision::Waiting;
        };
        let Some(parked) = self.sync_manager.parked_batch(batch_id).cloned() else {
            return ParkedBatchDecision::Decided;
        };

        // A parked row the seal does not declare is never served; its denial stands for
        // the batch, as it did before batches waited.
        if let Some((object_id, _)) = parked
            .checks
            .iter()
            .find(|(_, check)| check.row().is_none_or(|row| !declares(&seal, row)))
        {
            self.reject_parked_batch(storage, batch_id, Some(*object_id));
            self.index_new_row_visibility(storage, effects);
            return ParkedBatchDecision::Decided;
        }

        let parked_tables = parked
            .checks
            .iter()
            .map(|(object_id, check)| (*object_id, check.table().unwrap_or_default().to_string()));
        let Some(mut members) =
            Self::stored_declared_members(storage, &seal, parked_tables.collect())
        else {
            self.reject_parked_batch(storage, batch_id, None);
            self.index_new_row_visibility(storage, effects);
            return ParkedBatchDecision::Decided;
        };
        let complete = seal.members.iter().all(|member| {
            members.contains_key(&member.object_id) || parked.checks.contains_key(&member.object_id)
        });
        if !complete {
            // A member still on its way is no denial: after a restart or a `Missing` its
            // resend can take longer than any wait. The batch keeps what it parked, holds
            // its writer only until its deadline, and is let go once its rows stop coming.
            return ParkedBatchDecision::Waiting;
        }

        for _ in 0..MAX_PARKED_PASSES {
            let waiting = self
                .sync_manager
                .parked_batch(batch_id)
                .map(|batch| batch.checks.keys().copied().collect::<Vec<_>>())
                .unwrap_or_default();
            if waiting.is_empty() {
                break;
            }
            let mut approved_any = false;
            for object_id in waiting {
                let Some(parked_check) = self.sync_manager.take_parked_check(batch_id, object_id)
                else {
                    continue;
                };
                self.write_check_overlay = Some(BatchOverlay {
                    batch_id,
                    members: members.clone(),
                    evaluated: object_id,
                });
                let outcome = self.evaluate_write_permission(storage, parked_check.check);
                self.write_check_overlay = None;
                self.index_new_row_visibility(storage, effects);

                if !self.sync_manager.has_parked_batch(batch_id) {
                    // Rejected as a whole.
                    return ParkedBatchDecision::Decided;
                }
                if self.sync_manager.is_parked(batch_id, object_id) {
                    continue;
                }
                if outcome == WriteCheckOutcome::Deferred {
                    return ParkedBatchDecision::Waiting;
                }
                // Approved: what the next check reads, if it is the row the seal declares.
                if let Some(member) = Self::stored_declared_member(storage, &seal, object_id) {
                    members.insert(object_id, member);
                    approved_any = true;
                }
            }
            if !approved_any {
                break;
            }
        }

        if self
            .sync_manager
            .parked_batch(batch_id)
            .is_some_and(|batch| !batch.checks.is_empty())
        {
            self.reject_parked_batch(storage, batch_id, None);
        } else {
            self.sync_manager
                .settle_decided_parked_batch(storage, batch_id);
        }
        // Settled rows reach readers in this pass, as an approval's do.
        self.index_new_row_visibility(storage, effects);
        ParkedBatchDecision::Decided
    }

    /// The members the seal declares and the store holds as declared, in a state that
    /// can still settle. `None` when two rows of the batch, stored or parked, claim one
    /// id: a row is in one table, and the batch is refused.
    fn stored_declared_members<H: Storage>(
        storage: &H,
        seal: &SealedBatchSubmission,
        parked_tables: Vec<(ObjectId, String)>,
    ) -> Option<HashMap<ObjectId, OverlayMember>> {
        let mut members = HashMap::new();
        let mut tables_by_id: HashMap<ObjectId, HashSet<String>> = HashMap::new();
        for (object_id, table) in parked_tables {
            tables_by_id.entry(object_id).or_default().insert(table);
        }
        let index = storage
            .load_local_batch_row_index(seal.batch_id)
            .ok()
            .flatten()
            .unwrap_or_default();
        for member in index {
            if member.branch_name.as_str() != seal.target_branch_name.as_str() {
                continue;
            }
            let Some(row) = storage
                .load_history_row_batch_for_schema_hash(
                    member.table_name.as_str(),
                    member.schema_hash,
                    member.branch_name.as_str(),
                    member.object_id,
                    seal.batch_id,
                )
                .ok()
                .flatten()
            else {
                continue;
            };
            tables_by_id
                .entry(member.object_id)
                .or_default()
                .insert(member.table_name.clone());
            if !matches!(
                row.state,
                RowState::StagingPending | RowState::VisibleDirect
            ) || !declares(seal, &row)
            {
                continue;
            }
            members.insert(
                member.object_id,
                OverlayMember {
                    table_name: member.table_name,
                    schema_hash: member.schema_hash,
                    branch_name: member.branch_name,
                },
            );
        }
        if tables_by_id.values().any(|tables| tables.len() > 1) {
            return None;
        }
        Some(members)
    }

    /// One row of the batch as `stored_declared_members` would give it: the row just
    /// approved, without reading the batch's other rows again.
    fn stored_declared_member<H: Storage>(
        storage: &H,
        seal: &SealedBatchSubmission,
        object_id: ObjectId,
    ) -> Option<OverlayMember> {
        let index = storage
            .load_local_batch_row_index(seal.batch_id)
            .ok()
            .flatten()?;
        index.into_iter().find_map(|member| {
            if member.object_id != object_id
                || member.branch_name.as_str() != seal.target_branch_name.as_str()
            {
                return None;
            }
            let row = storage
                .load_history_row_batch_for_schema_hash(
                    member.table_name.as_str(),
                    member.schema_hash,
                    member.branch_name.as_str(),
                    member.object_id,
                    seal.batch_id,
                )
                .ok()
                .flatten()?;
            (matches!(
                row.state,
                RowState::StagingPending | RowState::VisibleDirect
            ) && declares(seal, &row))
            .then_some(OverlayMember {
                table_name: member.table_name,
                schema_hash: member.schema_hash,
                branch_name: member.branch_name,
            })
        })
    }

    /// Reject a parked batch through the path its first denial would have taken: the
    /// batch-level answer for its writer. `first` names the check whose reason it carries;
    /// otherwise the first parked one.
    fn reject_parked_batch<H: Storage>(
        &mut self,
        storage: &mut H,
        batch_id: BatchId,
        first: Option<ObjectId>,
    ) {
        let first = first.or_else(|| {
            self.sync_manager
                .parked_batch(batch_id)
                .and_then(|batch| batch.checks.keys().next().copied())
        });
        if let Some(object_id) = first
            && let Some(parked) = self.sync_manager.take_parked_check(batch_id, object_id)
        {
            self.sync_manager.reject_permission_check_with_code(
                storage,
                parked.check,
                parked.code,
                parked.reason,
            );
        }
        // Whatever the answer above did not cover is denied on its own row.
        while let Some(object_id) = self
            .sync_manager
            .parked_batch(batch_id)
            .and_then(|batch| batch.checks.keys().next().copied())
        {
            let Some(parked) = self.sync_manager.take_parked_check(batch_id, object_id) else {
                break;
            };
            self.sync_manager.reject_permission_check_with_code(
                storage,
                parked.check,
                parked.code,
                parked.reason,
            );
        }
        self.sync_manager.drop_parked_batch(batch_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_overlay_never_shows_the_evaluated_row() {
        let (evaluated, sibling) = (ObjectId::new(), ObjectId::new());
        let branch = BranchName::new("main");
        let member = OverlayMember {
            table_name: "messages".to_string(),
            schema_hash: SchemaHash::from_bytes([0; 32]),
            branch_name: branch,
        };
        let overlay = BatchOverlay {
            batch_id: BatchId([1; 16]),
            members: HashMap::from([(evaluated, member.clone()), (sibling, member)]),
            evaluated,
        };
        let keys = overlay.row_keys("main");
        assert!(!keys.contains_key(&evaluated));
        assert!(keys.contains_key(&sibling));
        assert!(overlay.member(evaluated).is_none());
        assert!(overlay.member(sibling).is_some());
    }
}
