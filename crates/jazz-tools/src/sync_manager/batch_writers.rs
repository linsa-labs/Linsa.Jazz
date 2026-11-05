//! Who a batch belongs to ([`BatchWriter`]), and the checks that keep everyone else from
//! adding rows to it or sealing it.
//!
//! A batch's fate covers every row it holds, and a direct batch's rows are served as each
//! is approved, before the batch has a fate. So only the batch's writer may add rows to it
//! or seal it. The batch is bound to the first connection a row of it (or, with no row
//! yet, a sound seal) arrives on, and the binding lives as long as the fate: dropped at
//! settle, the next connection to send a row of the batch could claim it and get the
//! writer's own replays refused.

use super::*;
use crate::batch_fate::BatchWriter;
use crate::row_histories::{BatchId, StoredRowBatch};
use crate::storage::{Storage, StorageError};

/// The answer to "may this connection add to the batch, or seal it?".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Admission {
    Writer,
    /// Someone else holds the batch.
    Stranger,
    /// The store could not say. Nothing was bound, so the claimant's retransmit asks again.
    Unavailable,
}

/// Whether an admission may bind a batch nobody holds yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Binding {
    /// A row, or a seal that validates: either can start a batch.
    Bind,
    /// A seal that does not validate decides nothing for a batch nobody holds.
    CheckOnly,
}

impl SyncManager {
    /// Whether `claimant` may add a row to `batch_id` or seal it, binding a batch nobody
    /// holds yet when `binding` allows.
    ///
    /// A trusted connection is always let through: it carries other writers' rows by
    /// design. It binds only a batch that has no row yet, so it never takes over a batch
    /// a user already wrote.
    ///
    /// A batch with rows and no binding was written before bindings existed. It is the
    /// claimant's if every row of it the store still holds says the claimant wrote it.
    pub(super) fn admit_batch_writer<H: Storage>(
        storage: &mut H,
        batch_id: BatchId,
        claimant: &BatchWriter,
        binding: Binding,
    ) -> Admission {
        let BatchWriter::User(user_id) = claimant else {
            if binding == Binding::Bind {
                Self::bind_trusted_batch(storage, batch_id);
            }
            return Admission::Writer;
        };
        match storage.load_batch_writer(batch_id) {
            Ok(Some(writer)) if writer == *claimant => return Admission::Writer,
            Ok(Some(_)) => return Admission::Stranger,
            Ok(None) => {}
            Err(error) => {
                tracing::warn!(?batch_id, %error, "failed to load a batch's writer");
                return Admission::Unavailable;
            }
        }
        let admission = match Self::rows_written_before_bindings(storage, batch_id) {
            Ok(None) if binding == Binding::Bind => Admission::Writer,
            // Nobody holds it and it holds nothing: there is no fate for a bad seal to decide.
            Ok(None) => return Admission::Stranger,
            // Permanent, not transient: with none of its rows left there is nothing that
            // could ever show who wrote the batch, so its writer stays refused. Liveness
            // only, and only for a batch from before bindings.
            Ok(Some(rows)) if rows.is_empty() => {
                tracing::warn!(?batch_id, "none of a batch's indexed rows is stored");
                return Admission::Unavailable;
            }
            Ok(Some(rows)) if rows.iter().all(|row| row.updated_by.as_str() == user_id) => {
                Admission::Writer
            }
            Ok(Some(_)) => return Admission::Stranger,
            Err(error) => {
                tracing::warn!(?batch_id, %error, "failed to read a batch's rows to find its writer");
                return Admission::Unavailable;
            }
        };
        if binding == Binding::Bind
            && let Err(error) = storage.put_batch_writer(batch_id, claimant)
        {
            tracing::warn!(?batch_id, %error, "failed to bind a batch to its writer");
            return Admission::Unavailable;
        }
        admission
    }

    /// Bind a batch that has no row yet to the trusted connection starting it. Best
    /// effort: a trusted connection is let through either way, and a batch left unbound
    /// is still judged by its rows' authors.
    pub(super) fn bind_trusted_batch<H: Storage>(storage: &mut H, batch_id: BatchId) {
        let result = storage.load_batch_writer(batch_id).and_then(|writer| {
            let has_rows = storage
                .load_local_batch_row_index(batch_id)?
                .is_some_and(|members| !members.is_empty());
            if writer.is_some() || has_rows {
                return Ok(());
            }
            storage.put_batch_writer(batch_id, &BatchWriter::Trusted)
        });
        if let Err(error) = result {
            tracing::warn!(?batch_id, %error, "failed to bind a batch to the trusted connection starting it");
        }
    }

    /// Whether `claimant` is who `batch_id` is bound to. Read at decision time, after the
    /// row that is being decided has bound the batch, so it never binds. A store that
    /// cannot say answers no, so the denial stays on its own row; a transactional batch
    /// left that way waits for its seal to be judged again rather than being rejected.
    pub(super) fn holds_batch<H: Storage>(
        storage: &H,
        batch_id: BatchId,
        claimant: &BatchWriter,
    ) -> bool {
        matches!(storage.load_batch_writer(batch_id), Ok(Some(writer)) if writer == *claimant)
    }

    /// The batch's stored rows, or `None` when the batch has none indexed. An indexed row
    /// the store no longer holds is skipped, as `local_batch_rows` does.
    fn rows_written_before_bindings<H: Storage>(
        storage: &H,
        batch_id: BatchId,
    ) -> Result<Option<Vec<StoredRowBatch>>, StorageError> {
        let Some(members) = storage.load_local_batch_row_index(batch_id)? else {
            return Ok(None);
        };
        if members.is_empty() {
            return Ok(None);
        }
        let mut rows = Vec::with_capacity(members.len());
        for member in members {
            if let Some(row) = storage.load_history_row_batch_for_schema_hash(
                member.table_name.as_str(),
                member.schema_hash,
                member.branch_name.as_str(),
                member.object_id,
                batch_id,
            )? {
                rows.push(row);
            }
        }
        Ok(Some(rows))
    }
}
