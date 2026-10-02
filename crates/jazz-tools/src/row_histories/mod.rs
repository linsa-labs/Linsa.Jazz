//! Row-history data types, codecs, and the algorithms that turn them into the
//! row a reader sees and the writes a sync target receives.
//!
//! Split into five submodules so each concern is independently navigable:
//! - [`types`]: data types (BatchId, RowState, QueryRowBatch, StoredRowBatch,
//!   RowMetadata, VisibleRowEntry, error types)
//! - [`codecs`]: descriptor builders and flat-row encode/decode
//! - [`resolution`]: pure visibility/merge math — frontier walk, latest common
//!   ancestor, per-column merge, delete-winner, computed visible preview. No
//!   storage access; called by both `mutations` and `types` (via
//!   `VisibleRowEntry::rebuild_*`).
//! - [`fastpath`]: pure O(1) serial-write construction of the next
//!   `VisibleRowEntry` (guards + tier-pointer carry-forward), its kill switch
//!   and telemetry counters. No storage access.
//! - [`mutations`]: the storage-mutating verbs (`apply_row_batch`,
//!   `patch_row_batch_state`) and their direct support — load history,
//!   recompute visibility via `resolution` (or `fastpath` when eligible),
//!   write through `Storage`, emit a `RowVisibilityChange`.

mod codecs;
mod fastpath;
mod mutations;
mod resolution;
mod types;

pub(crate) use codecs::{
    FlatRowCodecs, decode_flat_history_row_with_codecs, decode_flat_visible_row_entry_with_codecs,
    flat_row_codecs,
};
pub use codecs::{
    compute_row_digest, decode_flat_history_row, decode_flat_visible_row_entry,
    encode_flat_history_row, encode_flat_visible_row_entry, history_row_physical_descriptor,
    visible_row_physical_descriptor,
};
#[cfg(test)]
pub(crate) use fastpath::FORKED_FASTPATH_ARMS_ON_THREAD;
pub use fastpath::{
    FORKED_FASTPATH_HITS, HISTORY_FASTPATH_FALLBACKS, HISTORY_FASTPATH_HITS,
    PATCH_FASTPATH_FALLBACKS, PATCH_FASTPATH_HITS, QUERY_PROVENANCE_HISTORY_SCANS,
    QUERY_TIER_READ_HISTORY_SCANS, history_fastpath_enabled,
};
#[cfg(any(test, feature = "test"))]
pub use fastpath::{HistoryFastpathMode, force_history_fastpath};
pub(crate) use mutations::{ApplyRowBatchWithContext, apply_row_batch_with_context};
pub use mutations::{apply_row_batch, patch_row_batch_state};
pub(crate) use resolution::{
    elided_snapshot_dominator, superseded_by_snapshot, visible_row_preview_from_history_rows,
};
pub use types::{
    ApplyRowBatchResult, BatchId, HistoryScan, QueryRowBatch, RowHistoryError, RowMetadata,
    RowState, RowVisibilityChange, StoredRowBatch, VisibleRowEntry,
};

#[cfg(test)]
mod elided_snapshot_tests;

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use uuid::Uuid;

    use super::fastpath::{
        try_forked_fastpath_entry, try_in_place_tip_update_entry, try_serial_fastpath_entry,
    };
    use super::*;
    use crate::metadata::{DeleteKind, MetadataKey, RowProvenance};
    use crate::object::{BranchName, ObjectId};
    use crate::query_manager::types::{
        ColumnDescriptor, ColumnMergeStrategy, ColumnType, RowDescriptor, Schema, TableName,
        TableSchema, Value,
    };
    use crate::row_format::{decode_row, encode_row};
    use crate::storage::{MemoryStorage, RowLocator, Storage};
    use crate::sync_manager::DurabilityTier;

    fn visible_row(updated_at: u64, confirmed_tier: Option<DurabilityTier>) -> StoredRowBatch {
        StoredRowBatch::new(
            ObjectId::new(),
            "main",
            Vec::new(),
            vec![updated_at as u8],
            RowProvenance::for_insert("alice".to_string(), updated_at),
            HashMap::new(),
            RowState::VisibleDirect,
            confirmed_tier,
        )
    }

    #[test]
    fn flat_visible_row_binary_roundtrips_retained_visible_columns() {
        let user_descriptor = RowDescriptor::new(vec![
            ColumnDescriptor::new("title", ColumnType::Text),
            ColumnDescriptor::new("done", ColumnType::Boolean).nullable(),
        ]);
        let global = StoredRowBatch::new(
            ObjectId::from_uuid(Uuid::from_u128(21)),
            "main",
            Vec::new(),
            encode_row(
                &user_descriptor,
                &[Value::Text("ship it".into()), Value::Boolean(true)],
            )
            .expect("encode global row"),
            RowProvenance::for_insert("alice".to_string(), 10),
            HashMap::from([("source".to_string(), "global".to_string())]),
            RowState::VisibleDirect,
            Some(DurabilityTier::GlobalServer),
        );
        let current = StoredRowBatch::new(
            global.row_id,
            "main",
            vec![global.batch_id()],
            encode_row(
                &user_descriptor,
                &[Value::Text("ship it".into()), Value::Boolean(false)],
            )
            .expect("encode current row"),
            RowProvenance::for_update(&global.row_provenance(), "bob".to_string(), 30),
            HashMap::from([("source".to_string(), "local".to_string())]),
            RowState::VisibleDirect,
            Some(DurabilityTier::Local),
        );
        let entry = VisibleRowEntry {
            current_row: current,
            branch_frontier: vec![global.batch_id()],
            worker_batch_id: None,
            edge_batch_id: Some(global.batch_id()),
            global_batch_id: Some(global.batch_id()),
            winner_batch_pool: Vec::new(),
            current_winner_ordinals: None,
            worker_winner_ordinals: None,
            edge_winner_ordinals: None,
            global_winner_ordinals: None,
            merge_artifacts: Some(vec![1, 2, 3, 4]),
        };

        let encoded =
            encode_flat_visible_row_entry(&user_descriptor, &entry).expect("encode flat visible");
        let decoded = decode_flat_visible_row_entry(
            &user_descriptor,
            entry.current_row.row_id,
            entry.current_row.branch.as_str(),
            &encoded,
        )
        .expect("decode flat visible");

        assert_eq!(decoded.current_row.row_id, entry.current_row.row_id);
        assert_eq!(decoded.current_row.batch_id(), entry.current_row.batch_id());
        assert_eq!(decoded.current_row.branch, entry.current_row.branch);
        assert!(decoded.current_row.parents.is_empty());
        assert_eq!(decoded.current_row.updated_at, entry.current_row.updated_at);
        assert_eq!(decoded.current_row.created_by, entry.current_row.created_by);
        assert_eq!(decoded.current_row.created_at, entry.current_row.created_at);
        assert_eq!(decoded.current_row.updated_by, entry.current_row.updated_by);
        assert_eq!(decoded.current_row.state, entry.current_row.state);
        assert_eq!(
            decoded.current_row.confirmed_tier,
            entry.current_row.confirmed_tier
        );
        assert_eq!(
            decoded.current_row.delete_kind,
            entry.current_row.delete_kind
        );
        assert!(decoded.current_row.metadata.is_empty());
        assert_eq!(decoded.current_row.data, entry.current_row.data);
        assert_eq!(decoded.branch_frontier, entry.branch_frontier);
        assert_eq!(decoded.worker_batch_id, entry.worker_batch_id);
        assert_eq!(decoded.edge_batch_id, entry.edge_batch_id);
        assert_eq!(decoded.global_batch_id, entry.global_batch_id);
        assert_eq!(decoded.merge_artifacts, entry.merge_artifacts);
    }

    #[test]
    fn visible_row_entry_omits_tier_pointers_when_current_is_globally_confirmed() {
        let current = visible_row(30, Some(DurabilityTier::GlobalServer));
        let entry = VisibleRowEntry::rebuild(current.clone(), std::slice::from_ref(&current));

        assert_eq!(entry.branch_frontier, vec![current.batch_id()]);
        assert_eq!(entry.worker_batch_id, None);
        assert_eq!(entry.edge_batch_id, None);
        assert_eq!(entry.global_batch_id, None);
        assert_eq!(entry.merge_artifacts, None);
    }

    #[test]
    fn visible_row_entry_resolves_tier_fallback_chain() {
        let global = visible_row(10, Some(DurabilityTier::GlobalServer));
        let edge = StoredRowBatch::new(
            global.row_id,
            "main",
            vec![global.batch_id()],
            vec![2],
            RowProvenance::for_update(&global.row_provenance(), "alice".to_string(), 20),
            HashMap::new(),
            RowState::VisibleDirect,
            Some(DurabilityTier::EdgeServer),
        );
        let current = StoredRowBatch::new(
            global.row_id,
            "main",
            vec![edge.batch_id()],
            vec![3],
            RowProvenance::for_update(&edge.row_provenance(), "alice".to_string(), 30),
            HashMap::new(),
            RowState::VisibleDirect,
            Some(DurabilityTier::Local),
        );
        let history = vec![global.clone(), edge.clone(), current.clone()];

        let entry = VisibleRowEntry::rebuild(current.clone(), &history);

        assert_eq!(entry.branch_frontier, vec![current.batch_id()]);
        assert_eq!(entry.worker_batch_id, None);
        assert_eq!(entry.edge_batch_id, Some(edge.batch_id()));
        assert_eq!(entry.global_batch_id, Some(global.batch_id()));
        assert_eq!(
            entry.batch_id_for_tier(DurabilityTier::Local),
            Some(current.batch_id())
        );
        assert_eq!(
            entry.batch_id_for_tier(DurabilityTier::EdgeServer),
            Some(edge.batch_id())
        );
        assert_eq!(
            entry.batch_id_for_tier(DurabilityTier::GlobalServer),
            Some(global.batch_id())
        );
    }

    #[test]
    fn visible_row_entry_returns_none_when_no_version_meets_required_tier() {
        let current = visible_row(30, Some(DurabilityTier::Local));
        let entry = VisibleRowEntry::rebuild(current.clone(), std::slice::from_ref(&current));

        assert_eq!(entry.branch_frontier, vec![current.batch_id()]);
        assert_eq!(entry.batch_id_for_tier(DurabilityTier::EdgeServer), None);
        assert_eq!(entry.batch_id_for_tier(DurabilityTier::GlobalServer), None);
    }

    #[test]
    fn visible_row_entry_preserves_multiple_branch_tips() {
        let base = visible_row(10, Some(DurabilityTier::Local));
        let left = StoredRowBatch::new(
            base.row_id,
            "main",
            vec![base.batch_id()],
            vec![1],
            RowProvenance::for_update(&base.row_provenance(), "alice".to_string(), 20),
            HashMap::new(),
            RowState::VisibleDirect,
            Some(DurabilityTier::Local),
        );
        let right = StoredRowBatch::new(
            base.row_id,
            "main",
            vec![base.batch_id()],
            vec![2],
            RowProvenance::for_update(&base.row_provenance(), "bob".to_string(), 21),
            HashMap::new(),
            RowState::VisibleDirect,
            Some(DurabilityTier::Local),
        );

        let entry = VisibleRowEntry::rebuild(right.clone(), &[base, left.clone(), right.clone()]);

        assert_eq!(
            entry.branch_frontier,
            vec![left.batch_id(), right.batch_id()]
        );
    }

    #[test]
    fn visible_row_entry_merges_conflicting_field_updates() {
        let descriptor = user_descriptor();
        let base = StoredRowBatch::new(
            ObjectId::new(),
            "main",
            Vec::new(),
            encode_row(
                &descriptor,
                &[Value::Text("task".into()), Value::Boolean(false)],
            )
            .unwrap(),
            RowProvenance::for_insert("alice".to_string(), 10),
            HashMap::new(),
            RowState::VisibleDirect,
            Some(DurabilityTier::Local),
        );
        let left = StoredRowBatch::new(
            base.row_id,
            "main",
            vec![base.batch_id()],
            encode_row(
                &descriptor,
                &[Value::Text("alice-title".into()), Value::Boolean(false)],
            )
            .unwrap(),
            RowProvenance::for_update(&base.row_provenance(), "alice".to_string(), 20),
            HashMap::new(),
            RowState::VisibleDirect,
            Some(DurabilityTier::Local),
        );
        let right = StoredRowBatch::new(
            base.row_id,
            "main",
            vec![base.batch_id()],
            encode_row(
                &descriptor,
                &[Value::Text("task".into()), Value::Boolean(true)],
            )
            .unwrap(),
            RowProvenance::for_update(&base.row_provenance(), "bob".to_string(), 21),
            HashMap::new(),
            RowState::VisibleDirect,
            Some(DurabilityTier::Local),
        );

        let entry = VisibleRowEntry::rebuild_with_descriptor(
            &descriptor,
            &[base, left.clone(), right.clone()],
        )
        .unwrap()
        .expect("merged visible entry");

        assert_eq!(
            decode_row(&descriptor, &entry.current_row.data).unwrap(),
            vec![Value::Text("alice-title".into()), Value::Boolean(true)]
        );
        assert_eq!(entry.current_row.batch_id(), right.batch_id());
        assert_eq!(entry.current_row.updated_by.as_str(), "bob");
        assert_eq!(
            entry.branch_frontier,
            vec![left.batch_id(), right.batch_id()]
        );
    }

    #[test]
    fn visible_row_entry_applies_counter_merge_strategy_per_column() {
        let descriptor = counter_descriptor();
        let base = StoredRowBatch::new(
            ObjectId::new(),
            "main",
            Vec::new(),
            encode_row(
                &descriptor,
                &[Value::Text("task".into()), Value::Integer(5)],
            )
            .unwrap(),
            RowProvenance::for_insert("alice".to_string(), 10),
            HashMap::new(),
            RowState::VisibleDirect,
            Some(DurabilityTier::Local),
        );
        let left = StoredRowBatch::new(
            base.row_id,
            "main",
            vec![base.batch_id()],
            encode_row(
                &descriptor,
                &[Value::Text("alice-title".into()), Value::Integer(7)],
            )
            .unwrap(),
            RowProvenance::for_update(&base.row_provenance(), "alice".to_string(), 20),
            HashMap::new(),
            RowState::VisibleDirect,
            Some(DurabilityTier::Local),
        );
        let right = StoredRowBatch::new(
            base.row_id,
            "main",
            vec![base.batch_id()],
            encode_row(
                &descriptor,
                &[Value::Text("task".into()), Value::Integer(4)],
            )
            .unwrap(),
            RowProvenance::for_update(&base.row_provenance(), "bob".to_string(), 21),
            HashMap::new(),
            RowState::VisibleDirect,
            Some(DurabilityTier::Local),
        );

        let entry = VisibleRowEntry::rebuild_with_descriptor(
            &descriptor,
            &[base, left.clone(), right.clone()],
        )
        .unwrap()
        .expect("merged visible entry");

        assert_eq!(
            decode_row(&descriptor, &entry.current_row.data).unwrap(),
            vec![Value::Text("alice-title".into()), Value::Integer(6)]
        );
        assert_eq!(entry.current_row.batch_id(), right.batch_id());
        assert_eq!(entry.current_row.updated_by.as_str(), "bob");
        assert_eq!(
            entry.branch_frontier,
            vec![left.batch_id(), right.batch_id()]
        );
        assert_eq!(
            entry.winner_batch_pool,
            vec![left.batch_id(), right.batch_id()]
        );
        assert_eq!(entry.current_winner_ordinals, Some(vec![0, 1]));
    }

    #[test]
    fn visible_row_entry_uses_consumer_schema_merge_strategy() {
        let counter_descriptor = counter_descriptor();
        let lww_descriptor = lww_integer_descriptor();
        let base = StoredRowBatch::new(
            ObjectId::new(),
            "main",
            Vec::new(),
            encode_row(
                &counter_descriptor,
                &[Value::Text("task".into()), Value::Integer(5)],
            )
            .unwrap(),
            RowProvenance::for_insert("alice".to_string(), 10),
            HashMap::new(),
            RowState::VisibleDirect,
            Some(DurabilityTier::Local),
        );
        let left = StoredRowBatch::new(
            base.row_id,
            "main",
            vec![base.batch_id()],
            encode_row(
                &counter_descriptor,
                &[Value::Text("alice-title".into()), Value::Integer(7)],
            )
            .unwrap(),
            RowProvenance::for_update(&base.row_provenance(), "alice".to_string(), 20),
            HashMap::new(),
            RowState::VisibleDirect,
            Some(DurabilityTier::Local),
        );
        let right = StoredRowBatch::new(
            base.row_id,
            "main",
            vec![base.batch_id()],
            encode_row(
                &counter_descriptor,
                &[Value::Text("task".into()), Value::Integer(4)],
            )
            .unwrap(),
            RowProvenance::for_update(&base.row_provenance(), "bob".to_string(), 21),
            HashMap::new(),
            RowState::VisibleDirect,
            Some(DurabilityTier::Local),
        );
        let history = vec![base, left, right];

        let counter_entry = VisibleRowEntry::rebuild_with_descriptor(&counter_descriptor, &history)
            .unwrap()
            .expect("counter merged visible entry");
        let lww_entry = VisibleRowEntry::rebuild_with_descriptor(&lww_descriptor, &history)
            .unwrap()
            .expect("lww merged visible entry");

        assert_eq!(
            decode_row(&counter_descriptor, &counter_entry.current_row.data).unwrap(),
            vec![Value::Text("alice-title".into()), Value::Integer(6)]
        );
        assert_eq!(
            decode_row(&lww_descriptor, &lww_entry.current_row.data).unwrap(),
            vec![Value::Text("alice-title".into()), Value::Integer(4)]
        );
    }

    #[test]
    fn visible_row_entry_errors_when_counter_merge_overflows() {
        let descriptor = counter_only_descriptor();
        let base = StoredRowBatch::new(
            ObjectId::new(),
            "main",
            Vec::new(),
            encode_row(&descriptor, &[Value::Integer(i32::MAX - 1)]).unwrap(),
            RowProvenance::for_insert("alice".to_string(), 10),
            HashMap::new(),
            RowState::VisibleDirect,
            Some(DurabilityTier::Local),
        );
        let left = StoredRowBatch::new(
            base.row_id,
            "main",
            vec![base.batch_id()],
            encode_row(&descriptor, &[Value::Integer(i32::MAX)]).unwrap(),
            RowProvenance::for_update(&base.row_provenance(), "alice".to_string(), 20),
            HashMap::new(),
            RowState::VisibleDirect,
            Some(DurabilityTier::Local),
        );
        let right = StoredRowBatch::new(
            base.row_id,
            "main",
            vec![base.batch_id()],
            encode_row(&descriptor, &[Value::Integer(i32::MAX)]).unwrap(),
            RowProvenance::for_update(&base.row_provenance(), "bob".to_string(), 21),
            HashMap::new(),
            RowState::VisibleDirect,
            Some(DurabilityTier::Local),
        );

        let error = VisibleRowEntry::rebuild_with_descriptor(&descriptor, &[base, left, right])
            .expect_err("counter overflow should fail");

        assert!(
            error.to_string().contains("overflow"),
            "expected overflow error, got {error}"
        );
    }

    #[test]
    fn visible_row_entry_merges_accepted_transactional_rows_but_ignores_staging_and_rejected_rows()
    {
        let descriptor = user_descriptor();
        let base = StoredRowBatch::new(
            ObjectId::new(),
            "main",
            Vec::new(),
            encode_row(
                &descriptor,
                &[Value::Text("task".into()), Value::Boolean(false)],
            )
            .unwrap(),
            RowProvenance::for_insert("alice".to_string(), 10),
            HashMap::new(),
            RowState::VisibleDirect,
            Some(DurabilityTier::Local),
        );
        let accepted_transaction = StoredRowBatch::new(
            base.row_id,
            "main",
            vec![base.batch_id()],
            encode_row(
                &descriptor,
                &[Value::Text("txn-title".into()), Value::Boolean(false)],
            )
            .unwrap(),
            RowProvenance::for_update(&base.row_provenance(), "alice".to_string(), 20),
            HashMap::new(),
            RowState::VisibleTransactional,
            Some(DurabilityTier::Local),
        );
        let direct = StoredRowBatch::new(
            base.row_id,
            "main",
            vec![base.batch_id()],
            encode_row(
                &descriptor,
                &[Value::Text("task".into()), Value::Boolean(true)],
            )
            .unwrap(),
            RowProvenance::for_update(&base.row_provenance(), "bob".to_string(), 21),
            HashMap::new(),
            RowState::VisibleDirect,
            Some(DurabilityTier::Local),
        );
        let staging = StoredRowBatch::new(
            base.row_id,
            "main",
            vec![base.batch_id()],
            encode_row(
                &descriptor,
                &[
                    Value::Text("staging-should-not-win".into()),
                    Value::Boolean(false),
                ],
            )
            .unwrap(),
            RowProvenance::for_update(&base.row_provenance(), "mallory".to_string(), 30),
            HashMap::new(),
            RowState::StagingPending,
            None,
        );
        let rejected = StoredRowBatch::new(
            base.row_id,
            "main",
            vec![base.batch_id()],
            encode_row(
                &descriptor,
                &[
                    Value::Text("rejected-should-not-win".into()),
                    Value::Boolean(false),
                ],
            )
            .unwrap(),
            RowProvenance::for_update(&base.row_provenance(), "mallory".to_string(), 31),
            HashMap::new(),
            RowState::Rejected,
            None,
        );

        let entry = VisibleRowEntry::rebuild_with_descriptor(
            &descriptor,
            &[
                base,
                accepted_transaction,
                direct.clone(),
                staging,
                rejected,
            ],
        )
        .unwrap()
        .expect("merged visible entry");

        assert_eq!(
            decode_row(&descriptor, &entry.current_row.data).unwrap(),
            vec![Value::Text("txn-title".into()), Value::Boolean(true)]
        );
        assert_eq!(entry.current_row.batch_id(), direct.batch_id());
        assert_eq!(entry.current_row.updated_by.as_str(), "bob");
    }

    #[test]
    fn visible_row_entry_roundtrips_current_winner_ordinals() {
        let descriptor = user_descriptor();
        let base = StoredRowBatch::new(
            ObjectId::new(),
            "main",
            Vec::new(),
            encode_row(
                &descriptor,
                &[Value::Text("task".into()), Value::Boolean(false)],
            )
            .unwrap(),
            RowProvenance::for_insert("alice".to_string(), 10),
            HashMap::new(),
            RowState::VisibleDirect,
            Some(DurabilityTier::Local),
        );
        let left = StoredRowBatch::new(
            base.row_id,
            "main",
            vec![base.batch_id()],
            encode_row(
                &descriptor,
                &[Value::Text("alice-title".into()), Value::Boolean(false)],
            )
            .unwrap(),
            RowProvenance::for_update(&base.row_provenance(), "alice".to_string(), 20),
            HashMap::new(),
            RowState::VisibleDirect,
            Some(DurabilityTier::Local),
        );
        let right = StoredRowBatch::new(
            base.row_id,
            "main",
            vec![base.batch_id()],
            encode_row(
                &descriptor,
                &[Value::Text("task".into()), Value::Boolean(true)],
            )
            .unwrap(),
            RowProvenance::for_update(&base.row_provenance(), "bob".to_string(), 21),
            HashMap::new(),
            RowState::VisibleDirect,
            Some(DurabilityTier::Local),
        );

        let entry = VisibleRowEntry::rebuild_with_descriptor(
            &descriptor,
            &[base, left.clone(), right.clone()],
        )
        .unwrap()
        .expect("merged visible entry");
        assert_eq!(
            entry.winner_batch_pool,
            vec![left.batch_id(), right.batch_id()]
        );
        assert_eq!(entry.current_winner_ordinals, Some(vec![0, 1]));

        let encoded =
            encode_flat_visible_row_entry(&descriptor, &entry).expect("encode merged visible row");
        let decoded = decode_flat_visible_row_entry(
            &descriptor,
            entry.current_row.row_id,
            entry.current_row.branch.as_str(),
            &encoded,
        )
        .expect("decode merged visible row");

        assert_eq!(decoded.winner_batch_pool, entry.winner_batch_pool);
        assert_eq!(
            decoded.current_winner_ordinals,
            entry.current_winner_ordinals
        );
        assert_eq!(decoded.edge_winner_ordinals, None);
        assert_eq!(decoded.global_winner_ordinals, None);
    }

    #[test]
    fn visible_row_entry_materializes_tier_preview_when_batch_id_matches_current() {
        let descriptor = user_descriptor();
        let base = StoredRowBatch::new(
            ObjectId::new(),
            "main",
            Vec::new(),
            encode_row(
                &descriptor,
                &[Value::Text("task".into()), Value::Boolean(false)],
            )
            .unwrap(),
            RowProvenance::for_insert("alice".to_string(), 10),
            HashMap::new(),
            RowState::VisibleDirect,
            Some(DurabilityTier::GlobalServer),
        );
        let worker_done = StoredRowBatch::new(
            base.row_id,
            "main",
            vec![base.batch_id()],
            encode_row(
                &descriptor,
                &[Value::Text("task".into()), Value::Boolean(true)],
            )
            .unwrap(),
            RowProvenance::for_update(&base.row_provenance(), "bob".to_string(), 20),
            HashMap::new(),
            RowState::VisibleDirect,
            Some(DurabilityTier::Local),
        );
        let edge_title = StoredRowBatch::new(
            base.row_id,
            "main",
            vec![base.batch_id()],
            encode_row(
                &descriptor,
                &[Value::Text("edge-title".into()), Value::Boolean(false)],
            )
            .unwrap(),
            RowProvenance::for_update(&base.row_provenance(), "alice".to_string(), 30),
            HashMap::new(),
            RowState::VisibleDirect,
            Some(DurabilityTier::EdgeServer),
        );
        let history_rows = vec![base.clone(), worker_done.clone(), edge_title.clone()];
        let row_by_batch_id = history_rows
            .iter()
            .cloned()
            .map(|row| (row.batch_id(), row))
            .collect::<HashMap<_, _>>();

        let entry = VisibleRowEntry::rebuild_with_descriptor(&descriptor, &history_rows)
            .unwrap()
            .expect("visible entry");

        assert_eq!(entry.current_row.batch_id(), edge_title.batch_id());
        assert_eq!(entry.edge_batch_id, Some(edge_title.batch_id()));
        assert_eq!(entry.edge_winner_ordinals, None);

        let edge_preview = entry
            .materialize_preview_for_tier_from_loaded_rows(
                &descriptor,
                DurabilityTier::EdgeServer,
                &row_by_batch_id,
            )
            .unwrap()
            .expect("edge preview");
        assert_eq!(
            decode_row(&descriptor, &edge_preview.data).unwrap(),
            vec![Value::Text("edge-title".into()), Value::Boolean(false)]
        );
    }

    #[test]
    fn visible_row_entry_persists_merged_tier_override_ordinals() {
        let descriptor = user_descriptor();
        let base = StoredRowBatch::new(
            ObjectId::new(),
            "main",
            Vec::new(),
            encode_row(
                &descriptor,
                &[Value::Text("task".into()), Value::Boolean(false)],
            )
            .unwrap(),
            RowProvenance::for_insert("alice".to_string(), 10),
            HashMap::new(),
            RowState::VisibleDirect,
            Some(DurabilityTier::GlobalServer),
        );
        let edge_title = StoredRowBatch::new(
            base.row_id,
            "main",
            vec![base.batch_id()],
            encode_row(
                &descriptor,
                &[Value::Text("edge-title".into()), Value::Boolean(false)],
            )
            .unwrap(),
            RowProvenance::for_update(&base.row_provenance(), "alice".to_string(), 20),
            HashMap::new(),
            RowState::VisibleDirect,
            Some(DurabilityTier::EdgeServer),
        );
        let edge_done = StoredRowBatch::new(
            base.row_id,
            "main",
            vec![base.batch_id()],
            encode_row(
                &descriptor,
                &[Value::Text("task".into()), Value::Boolean(true)],
            )
            .unwrap(),
            RowProvenance::for_update(&base.row_provenance(), "bob".to_string(), 21),
            HashMap::new(),
            RowState::VisibleDirect,
            Some(DurabilityTier::EdgeServer),
        );
        let worker_current = StoredRowBatch::new(
            base.row_id,
            "main",
            vec![edge_title.batch_id(), edge_done.batch_id()],
            encode_row(
                &descriptor,
                &[Value::Text("edge-title".into()), Value::Boolean(true)],
            )
            .unwrap(),
            RowProvenance::for_update(&edge_done.row_provenance(), "charlie".to_string(), 30),
            HashMap::new(),
            RowState::VisibleDirect,
            Some(DurabilityTier::Local),
        );
        let history_rows = vec![
            base.clone(),
            edge_title.clone(),
            edge_done.clone(),
            worker_current.clone(),
        ];
        let row_by_batch_id = history_rows
            .iter()
            .cloned()
            .map(|row| (row.batch_id(), row))
            .collect::<HashMap<_, _>>();

        let entry = VisibleRowEntry::rebuild_with_descriptor(&descriptor, &history_rows)
            .unwrap()
            .expect("visible entry");

        assert_eq!(entry.current_row.batch_id(), worker_current.batch_id());
        assert_eq!(entry.current_winner_ordinals, None);
        assert_eq!(entry.edge_batch_id, Some(edge_done.batch_id()));
        assert_eq!(
            entry.winner_batch_pool,
            vec![edge_title.batch_id(), edge_done.batch_id()]
        );
        assert_eq!(entry.edge_winner_ordinals, Some(vec![0, 1]));

        let edge_preview = entry
            .materialize_preview_for_tier_from_loaded_rows(
                &descriptor,
                DurabilityTier::EdgeServer,
                &row_by_batch_id,
            )
            .unwrap()
            .expect("edge preview");
        assert_eq!(edge_preview.batch_id(), edge_done.batch_id());
        assert_eq!(
            decode_row(&descriptor, &edge_preview.data).unwrap(),
            vec![Value::Text("edge-title".into()), Value::Boolean(true)]
        );
    }

    fn user_descriptor() -> RowDescriptor {
        RowDescriptor::new(vec![
            ColumnDescriptor::new("title", ColumnType::Text),
            ColumnDescriptor::new("done", ColumnType::Boolean),
        ])
    }

    fn lww_integer_descriptor() -> RowDescriptor {
        RowDescriptor::new(vec![
            ColumnDescriptor::new("title", ColumnType::Text),
            ColumnDescriptor::new("count", ColumnType::Integer),
        ])
    }

    fn counter_descriptor() -> RowDescriptor {
        RowDescriptor::new(vec![
            ColumnDescriptor::new("title", ColumnType::Text),
            ColumnDescriptor::new("count", ColumnType::Integer)
                .merge_strategy(ColumnMergeStrategy::Counter),
        ])
    }

    fn counter_only_descriptor() -> RowDescriptor {
        RowDescriptor::new(vec![
            ColumnDescriptor::new("count", ColumnType::Integer)
                .merge_strategy(ColumnMergeStrategy::Counter),
        ])
    }

    #[test]
    fn history_row_physical_descriptor_appends_nullable_user_columns() {
        let descriptor = history_row_physical_descriptor(&user_descriptor());

        let title = descriptor
            .column("title")
            .expect("physical descriptor should contain title");
        assert!(title.nullable, "physical user columns should be nullable");

        let done = descriptor
            .column("done")
            .expect("physical descriptor should contain done");
        assert!(done.nullable, "physical user columns should be nullable");
    }

    #[test]
    fn history_row_physical_descriptor_omits_key_derived_and_marker_columns() {
        let descriptor = history_row_physical_descriptor(&user_descriptor());

        assert_eq!(
            descriptor
                .columns
                .iter()
                .filter(|column| column.name == "_jazz_batch_id")
                .count(),
            0,
            "flat history rows should not store batch identity from the key in the payload"
        );
        assert!(
            descriptor.column("_jazz_format_id").is_none(),
            "flat history rows should not need an in-payload format marker once decoding is key-aware"
        );
        assert!(
            descriptor.column("_jazz_row_id").is_none(),
            "flat history rows should not store row id from the key in the payload"
        );
        assert!(
            descriptor.column("_jazz_branch").is_none(),
            "flat history rows should not store branch from the key in the payload"
        );
    }

    #[test]
    fn visible_row_physical_descriptor_keeps_current_batch_id_but_omits_marker() {
        let descriptor = visible_row_physical_descriptor(&user_descriptor());

        assert!(
            descriptor.column("_jazz_format_id").is_none(),
            "visible rows should not need an in-payload format marker once keyed decoding is available"
        );
        assert_eq!(
            descriptor
                .columns
                .iter()
                .filter(|column| column.name == "_jazz_batch_id")
                .count(),
            1,
            "visible rows should keep the current visible batch id in the flat payload"
        );
        assert!(
            descriptor.column("_jazz_row_id").is_none(),
            "visible rows should derive row id from the storage key"
        );
        assert!(
            descriptor.column("_jazz_branch").is_none(),
            "visible rows should derive branch from the storage key"
        );
        assert!(
            descriptor.column("_jazz_parents").is_none(),
            "visible rows should not duplicate history parents in the hot visible payload"
        );
        assert!(
            descriptor.column("_jazz_metadata").is_none(),
            "visible rows should not duplicate history metadata in the hot visible payload"
        );
        assert!(
            descriptor.column("_jazz_is_deleted").is_none(),
            "visible rows should derive deletion state from delete_kind in the hot payload"
        );
    }

    #[test]
    fn flat_visible_row_common_case_omits_empty_arrays_and_metadata() {
        let descriptor = user_descriptor();
        let current = visible_row(10, Some(DurabilityTier::Local));
        let entry = VisibleRowEntry::rebuild(current.clone(), std::slice::from_ref(&current));

        let encoded =
            encode_flat_visible_row_entry(&descriptor, &entry).expect("encode visible row");
        let values = decode_row(&visible_row_physical_descriptor(&descriptor), &encoded)
            .expect("decode visible row");

        assert_eq!(
            values[8],
            Value::Null,
            "singleton frontier matching current batch should be implicit"
        );
    }

    #[test]
    fn flat_history_row_binary_roundtrips_user_and_system_columns() {
        let user_descriptor = user_descriptor();
        let user_values = vec![Value::Text("Write docs".into()), Value::Boolean(false)];
        let user_data = crate::row_format::encode_row(&user_descriptor, &user_values).unwrap();
        let row = StoredRowBatch::new(
            ObjectId::from_uuid(Uuid::from_u128(42)),
            "main",
            vec![BatchId([9; 16])],
            user_data.clone(),
            RowProvenance {
                created_by: "alice".to_string(),
                created_at: 100,
                updated_by: "bob".to_string(),
                updated_at: 123,
            },
            HashMap::from([("source".to_string(), "test".to_string())]),
            RowState::VisibleTransactional,
            Some(DurabilityTier::EdgeServer),
        );

        let encoded =
            encode_flat_history_row(&user_descriptor, &row).expect("encode flat history row");
        let decoded = decode_flat_history_row(
            &user_descriptor,
            row.row_id,
            row.branch.as_str(),
            row.batch_id(),
            &encoded,
        )
        .expect("decode flat history row");

        assert_eq!(decoded, row);

        let physical_descriptor = history_row_physical_descriptor(&user_descriptor);
        let physical_values = decode_row(&physical_descriptor, &encoded).expect("decode values");
        assert_eq!(
            physical_values[physical_descriptor.column_index("title").unwrap()],
            Value::Text("Write docs".into())
        );
        assert_eq!(
            physical_values[physical_descriptor.column_index("done").unwrap()],
            Value::Boolean(false)
        );
    }

    #[test]
    fn flat_history_row_binary_roundtrips_nonempty_metadata() {
        let user_descriptor = user_descriptor();
        let row = StoredRowBatch::new(
            ObjectId::from_uuid(Uuid::from_u128(44)),
            "main",
            vec![BatchId([3; 16])],
            encode_row(
                &user_descriptor,
                &[Value::Text("Ship".into()), Value::Boolean(true)],
            )
            .expect("encode user row"),
            RowProvenance::for_insert("alice".to_string(), 100),
            HashMap::from([
                ("source".to_string(), "local".to_string()),
                ("kind".to_string(), "task".to_string()),
            ]),
            RowState::VisibleDirect,
            Some(DurabilityTier::Local),
        );

        let encoded =
            encode_flat_history_row(&user_descriptor, &row).expect("encode flat history row");
        let decoded = decode_flat_history_row(
            &user_descriptor,
            row.row_id,
            row.branch.as_str(),
            row.batch_id(),
            &encoded,
        )
        .expect("decode flat history row");

        assert_eq!(decoded.metadata, row.metadata);
    }

    #[test]
    fn flat_history_row_hard_delete_uses_null_user_columns() {
        let user_descriptor = user_descriptor();
        let deleted = StoredRowBatch::new(
            ObjectId::from_uuid(Uuid::from_u128(43)),
            "main",
            vec![BatchId([7; 16])],
            vec![],
            RowProvenance::for_insert("alice".to_string(), 100),
            HashMap::from([(
                crate::metadata::MetadataKey::Delete.to_string(),
                "hard".to_string(),
            )]),
            RowState::VisibleDirect,
            None,
        );

        let encoded =
            encode_flat_history_row(&user_descriptor, &deleted).expect("encode hard delete");
        let physical_descriptor = history_row_physical_descriptor(&user_descriptor);
        let physical_values = decode_row(&physical_descriptor, &encoded).expect("decode values");

        assert_eq!(
            physical_values[physical_descriptor.column_index("title").unwrap()],
            Value::Null
        );
        assert_eq!(
            physical_values[physical_descriptor.column_index("done").unwrap()],
            Value::Null
        );

        let decoded = decode_flat_history_row(
            &user_descriptor,
            deleted.row_id,
            deleted.branch.as_str(),
            deleted.batch_id(),
            &encoded,
        )
        .expect("decode hard delete");
        assert_eq!(decoded.data.as_ref(), &[] as &[u8]);
        assert!(decoded.is_hard_deleted());
    }

    #[test]
    fn flat_history_row_binary_compacts_hot_enums_to_single_bytes() {
        let user_descriptor = user_descriptor();
        let mut row = StoredRowBatch::new(
            ObjectId::from_uuid(Uuid::from_u128(45)),
            "main",
            vec![BatchId([4; 16])],
            encode_row(
                &user_descriptor,
                &[Value::Text("Compact".into()), Value::Boolean(false)],
            )
            .expect("encode user row"),
            RowProvenance::for_insert("alice".to_string(), 100),
            HashMap::new(),
            RowState::VisibleTransactional,
            Some(DurabilityTier::EdgeServer),
        );
        row.delete_kind = Some(DeleteKind::Hard);

        let encoded =
            encode_flat_history_row(&user_descriptor, &row).expect("encode flat history row");
        let descriptor = history_row_physical_descriptor(&user_descriptor);
        let layout = crate::row_format::compiled_row_layout(&descriptor);

        let state = crate::row_format::column_bytes_with_layout(
            &descriptor,
            layout.as_ref(),
            &encoded,
            descriptor.column_index("_jazz_state").unwrap(),
        )
        .expect("read state bytes")
        .expect("state should be present");
        let tier = crate::row_format::column_bytes_with_layout(
            &descriptor,
            layout.as_ref(),
            &encoded,
            descriptor.column_index("_jazz_confirmed_tier").unwrap(),
        )
        .expect("read tier bytes")
        .expect("tier should be present");
        let delete_kind = crate::row_format::column_bytes_with_layout(
            &descriptor,
            layout.as_ref(),
            &encoded,
            descriptor.column_index("_jazz_delete_kind").unwrap(),
        )
        .expect("read delete kind bytes")
        .expect("delete kind should be present");

        assert_eq!(state.len(), 1);
        assert_eq!(tier.len(), 1);
        assert_eq!(delete_kind.len(), 1);
    }

    // ─── serial-write fast path ─────────────────────────────────────────────
    //
    // Fixtures for `fastpath::try_serial_fastpath_entry`. The bar throughout:
    // a constructed fast entry must equal `rebuild_with_descriptor` over the
    // full history byte for byte, and every unprovable shape must decline.
    // (The randomized cross-check lives in `storage::conformance_differential`.)

    fn root_batch(
        descriptor: &RowDescriptor,
        values: &[Value],
        updated_at: u64,
        confirmed_tier: Option<DurabilityTier>,
    ) -> StoredRowBatch {
        StoredRowBatch::new(
            ObjectId::new(),
            "main",
            Vec::new(),
            encode_row(descriptor, values).expect("encode root row"),
            RowProvenance::for_insert("alice".to_string(), updated_at),
            HashMap::new(),
            RowState::VisibleDirect,
            confirmed_tier,
        )
    }

    fn serial_batch(
        prev: &StoredRowBatch,
        descriptor: &RowDescriptor,
        values: &[Value],
        updated_at: u64,
        confirmed_tier: Option<DurabilityTier>,
    ) -> StoredRowBatch {
        StoredRowBatch::new(
            prev.row_id,
            "main",
            vec![prev.batch_id()],
            encode_row(descriptor, values).expect("encode serial row"),
            RowProvenance::for_update(&prev.row_provenance(), "bob".to_string(), updated_at),
            HashMap::new(),
            RowState::VisibleDirect,
            confirmed_tier,
        )
    }

    fn rebuilt_entry(descriptor: &RowDescriptor, history: &[StoredRowBatch]) -> VisibleRowEntry {
        VisibleRowEntry::rebuild_with_descriptor(descriptor, history)
            .expect("rebuild visible entry")
            .expect("history has a visible row")
    }

    #[test]
    fn history_fastpath_matches_full_rebuild_on_tier_sparse_serial_chain() {
        let _mode = force_history_fastpath(true);
        let descriptor = user_descriptor();
        let a = root_batch(
            &descriptor,
            &[Value::Text("v1".into()), Value::Boolean(false)],
            10,
            Some(DurabilityTier::GlobalServer),
        );
        let b = serial_batch(
            &a,
            &descriptor,
            &[Value::Text("v2".into()), Value::Boolean(true)],
            20,
            Some(DurabilityTier::EdgeServer),
        );
        let previous = rebuilt_entry(&descriptor, &[a.clone(), b.clone()]);
        assert_eq!(previous.worker_batch_id, None);
        assert_eq!(previous.edge_batch_id, None);
        assert_eq!(previous.global_batch_id, Some(a.batch_id()));

        // Unconfirmed serial append: old tip becomes the worker/edge pointer,
        // the deeper global pointer is carried verbatim.
        let c = serial_batch(
            &b,
            &descriptor,
            &[Value::Text("v3".into()), Value::Boolean(false)],
            30,
            None,
        );
        let fast = try_serial_fastpath_entry(Some(&previous), &c)
            .expect("unconfirmed serial append takes the fast path");
        let mut history = vec![a.clone(), b.clone(), c.clone()];
        assert_eq!(fast, rebuilt_entry(&descriptor, &history));
        assert_eq!(fast.branch_frontier, vec![c.batch_id()]);
        assert_eq!(fast.worker_batch_id, Some(b.batch_id()));
        assert_eq!(fast.edge_batch_id, Some(b.batch_id()));
        assert_eq!(fast.global_batch_id, Some(a.batch_id()));

        // Extending past an unconfirmed tip carries every pointer verbatim.
        let d = serial_batch(
            &c,
            &descriptor,
            &[Value::Text("v4".into()), Value::Boolean(true)],
            40,
            None,
        );
        let fast = try_serial_fastpath_entry(Some(&fast), &d)
            .expect("second unconfirmed serial append takes the fast path");
        history.push(d.clone());
        assert_eq!(fast, rebuilt_entry(&descriptor, &history));
        assert_eq!(fast.worker_batch_id, Some(b.batch_id()));
        assert_eq!(fast.edge_batch_id, Some(b.batch_id()));
        assert_eq!(fast.global_batch_id, Some(a.batch_id()));
    }

    #[test]
    fn history_fastpath_clears_pointers_when_confirmed_row_tops_unconfirmed_chain() {
        let _mode = force_history_fastpath(true);
        let descriptor = user_descriptor();
        let a = root_batch(
            &descriptor,
            &[Value::Text("v1".into()), Value::Boolean(false)],
            10,
            None,
        );
        let b = serial_batch(
            &a,
            &descriptor,
            &[Value::Text("v2".into()), Value::Boolean(true)],
            20,
            None,
        );
        let previous = rebuilt_entry(&descriptor, &[a.clone(), b.clone()]);

        // No older batch is confirmed anywhere (provably empty tier sets), so
        // a globally confirmed new row satisfies every tier itself: all three
        // pointers must be None, matching the full rebuild.
        let c = serial_batch(
            &b,
            &descriptor,
            &[Value::Text("v3".into()), Value::Boolean(false)],
            30,
            Some(DurabilityTier::GlobalServer),
        );
        let fast = try_serial_fastpath_entry(Some(&previous), &c)
            .expect("confirmed row over an unconfirmed chain takes the fast path");
        assert_eq!(fast, rebuilt_entry(&descriptor, &[a, b, c]));
        assert_eq!(fast.worker_batch_id, None);
        assert_eq!(fast.edge_batch_id, None);
        assert_eq!(fast.global_batch_id, None);
    }

    #[test]
    fn history_fastpath_tier_confirmed_row_dominates_tier_hole() {
        let _mode = force_history_fastpath(true);
        let descriptor = RowDescriptor::new(vec![
            ColumnDescriptor::new("title", ColumnType::Text),
            ColumnDescriptor::new("count", ColumnType::Integer)
                .merge_strategy(ColumnMergeStrategy::Counter),
            ColumnDescriptor::new(
                "tags",
                ColumnType::Array {
                    element: Box::new(ColumnType::Text),
                },
            )
            .merge_strategy(ColumnMergeStrategy::GSet),
        ]);
        // r2 is a tier hole: it never got confirmed. Pre-v13-2 the one-step
        // tier frontier resurfaced r1 as a concurrent tier tip beside every
        // newer confirmed row, so a tier-satisfying append had to decline —
        // no entry-local rule could know what hid behind the hole. Since
        // v13-2 the tier frontier uses causal domination through the full
        // history DAG: r1 is a proper ancestor of r3 (via the unconfirmed
        // r2), so it is superseded at every tier r3 satisfies, holes
        // included — and a new row that dominates the whole visible set
        // provably dominates every tier set it enters.
        let r1 = root_batch(
            &descriptor,
            &[
                Value::Text("r1".into()),
                Value::Integer(0),
                Value::Array(vec![Value::Text("red".into())]),
            ],
            10,
            Some(DurabilityTier::GlobalServer),
        );
        let r2 = serial_batch(
            &r1,
            &descriptor,
            &[
                Value::Text("r2".into()),
                Value::Integer(0),
                Value::Array(vec![Value::Text("red".into())]),
            ],
            20,
            None,
        );
        let r3 = serial_batch(
            &r2,
            &descriptor,
            &[
                Value::Text("r3".into()),
                Value::Integer(5),
                Value::Array(vec![Value::Text("blue".into()), Value::Text("red".into())]),
            ],
            30,
            Some(DurabilityTier::GlobalServer),
        );
        let previous = rebuilt_entry(&descriptor, &[r1.clone(), r2.clone(), r3.clone()]);
        // Causal domination makes the holed chain look linear at every tier:
        // the global set {r1, r3} has the frontier {r3}, whose singleton
        // preview matches the current row — the entry stays clean.
        assert_eq!(previous.worker_batch_id, None);
        assert_eq!(previous.global_batch_id, None);
        assert!(previous.winner_batch_pool.is_empty());
        assert_eq!(previous.global_winner_ordinals, None);

        // A new globally confirmed row over the holed chain dominates every
        // visible row (its parents cover the whole frontier), so the tier
        // frontier collapses to the new row and the fast path proves the
        // clean entry in O(1) — byte-equal to the full rebuild. Pre-v13-2
        // this had to decline and pay an O(depth) rebuild (r1 resurfaced as
        // a concurrent tier tip and its counter delta was even
        // double-counted into the merged preview).
        let c = serial_batch(
            &r3,
            &descriptor,
            &[
                Value::Text("c".into()),
                Value::Integer(7),
                Value::Array(vec![Value::Text("green".into())]),
            ],
            40,
            Some(DurabilityTier::GlobalServer),
        );
        let fast = try_serial_fastpath_entry(Some(&previous), &c)
            .expect("tier-satisfying append over a holed chain takes the fast path");

        let rebuilt = rebuilt_entry(&descriptor, &[r1, r2, r3, c.clone()]);
        assert_eq!(fast, rebuilt);
        assert_eq!(rebuilt.current_row, c);
        assert_eq!(rebuilt.global_batch_id, None);
        assert_eq!(rebuilt.global_winner_ordinals, None);
        assert!(rebuilt.winner_batch_pool.is_empty());
    }

    #[test]
    fn history_fastpath_requires_exact_frontier_coverage() {
        let _mode = force_history_fastpath(true);
        let descriptor = user_descriptor();
        let base = root_batch(
            &descriptor,
            &[Value::Text("task".into()), Value::Boolean(false)],
            10,
            Some(DurabilityTier::Local),
        );
        let left = serial_batch(
            &base,
            &descriptor,
            &[Value::Text("left-title".into()), Value::Boolean(false)],
            20,
            Some(DurabilityTier::Local),
        );
        let right = serial_batch(
            &base,
            &descriptor,
            &[Value::Text("task".into()), Value::Boolean(true)],
            21,
            Some(DurabilityTier::Local),
        );
        let previous = rebuilt_entry(&descriptor, &[base.clone(), left.clone(), right.clone()]);
        assert_eq!(
            previous.branch_frontier,
            vec![left.batch_id(), right.batch_id()]
        );
        assert!(previous.current_winner_ordinals.is_some());

        let extend = |parents: Vec<BatchId>| {
            StoredRowBatch::new(
                base.row_id,
                "main",
                parents,
                encode_row(
                    &descriptor,
                    &[Value::Text("next".into()), Value::Boolean(true)],
                )
                .expect("encode extension row"),
                RowProvenance::for_update(&base.row_provenance(), "carol".to_string(), 30),
                HashMap::new(),
                RowState::VisibleDirect,
                None,
            )
        };

        // Single-parent append on a two-tip row: parents ⊂ frontier, decline.
        assert_eq!(
            try_serial_fastpath_entry(Some(&previous), &extend(vec![right.batch_id()])),
            None
        );
        // Duplicate parents never count as covering the frontier.
        assert_eq!(
            try_serial_fastpath_entry(
                Some(&previous),
                &extend(vec![right.batch_id(), right.batch_id()])
            ),
            None
        );
        // An explicit merge-commit naming every tip passes the set-equality
        // guard, but a live two-tip row carries merged-preview pool/ordinals
        // from its concurrent state, so the never-forked guard declines — in
        // practice merge-commits take the full path.
        assert_eq!(
            try_serial_fastpath_entry(
                Some(&previous),
                &extend(vec![left.batch_id(), right.batch_id()])
            ),
            None
        );
    }

    #[test]
    fn history_fastpath_declines_staging_delete_and_cross_branch_writes() {
        let _mode = force_history_fastpath(true);
        let descriptor = user_descriptor();
        let a = root_batch(
            &descriptor,
            &[Value::Text("v1".into()), Value::Boolean(false)],
            10,
            None,
        );
        let b = serial_batch(
            &a,
            &descriptor,
            &[Value::Text("v2".into()), Value::Boolean(true)],
            20,
            None,
        );
        let previous = rebuilt_entry(&descriptor, &[a.clone(), b.clone()]);

        // StagingPending must never become `current_row` through a shortcut:
        // it contributes nothing to the frontier, and shortcutting it would
        // make a staged batch publicly visible (supersede logic also stays on
        // the full path by construction).
        let mut staged = serial_batch(
            &b,
            &descriptor,
            &[Value::Text("staged".into()), Value::Boolean(false)],
            30,
            None,
        );
        staged.state = RowState::StagingPending;
        assert_eq!(try_serial_fastpath_entry(Some(&previous), &staged), None);

        let mut rejected = staged.clone();
        rejected.state = RowState::Rejected;
        assert_eq!(try_serial_fastpath_entry(Some(&previous), &rejected), None);

        // Deletes interact with the delete-winner preview overlay: full path.
        let soft_delete = StoredRowBatch::new(
            a.row_id,
            "main",
            vec![b.batch_id()],
            encode_row(
                &descriptor,
                &[Value::Text("gone".into()), Value::Boolean(false)],
            )
            .expect("encode delete row"),
            RowProvenance::for_update(&a.row_provenance(), "carol".to_string(), 31),
            HashMap::from([(MetadataKey::Delete.to_string(), "soft".to_string())]),
            RowState::VisibleDirect,
            None,
        );
        assert_eq!(
            try_serial_fastpath_entry(Some(&previous), &soft_delete),
            None
        );

        // Writes on another branch never consult this branch's entry.
        let mut cross_branch = serial_batch(
            &b,
            &descriptor,
            &[Value::Text("branched".into()), Value::Boolean(true)],
            32,
            None,
        );
        cross_branch.branch =
            crate::query_manager::types::SharedString::from("feature".to_string());
        assert_eq!(
            try_serial_fastpath_entry(Some(&previous), &cross_branch),
            None
        );

        // Without a previous entry there is nothing to carry forward.
        let fresh = serial_batch(
            &b,
            &descriptor,
            &[Value::Text("fresh".into()), Value::Boolean(true)],
            33,
            None,
        );
        assert_eq!(try_serial_fastpath_entry(None, &fresh), None);
    }

    /// A row two writers wrote over the same version, with a counter column so that a
    /// merge is more than picking the newer tip: `base`, then `left` and `right` over it.
    fn forked_history(
        descriptor: &RowDescriptor,
        tier: Option<DurabilityTier>,
    ) -> (StoredRowBatch, StoredRowBatch, StoredRowBatch) {
        let base = root_batch(
            descriptor,
            &[Value::Text("task".into()), Value::Integer(5)],
            10,
            tier,
        );
        let left = serial_batch(
            &base,
            descriptor,
            &[Value::Text("left".into()), Value::Integer(7)],
            20,
            tier,
        );
        let right = serial_batch(
            &base,
            descriptor,
            &[Value::Text("task".into()), Value::Integer(4)],
            21,
            tier,
        );
        (base, left, right)
    }

    fn write_over(
        row_like: &StoredRowBatch,
        descriptor: &RowDescriptor,
        parents: Vec<BatchId>,
        values: &[Value],
        updated_at: u64,
    ) -> StoredRowBatch {
        StoredRowBatch::new(
            row_like.row_id,
            "main",
            parents,
            encode_row(descriptor, values).expect("encode row"),
            RowProvenance::for_update(&row_like.row_provenance(), "carol".to_string(), updated_at),
            HashMap::new(),
            RowState::VisibleDirect,
            None,
        )
    }

    fn forked_entry(
        descriptor: &RowDescriptor,
        previous: &VisibleRowEntry,
        history: &[StoredRowBatch],
        row: &StoredRowBatch,
    ) -> Option<VisibleRowEntry> {
        try_forked_fastpath_entry(descriptor, Some(previous), row, &|batch_id| {
            history
                .iter()
                .find(|stored| stored.batch_id() == batch_id)
                .cloned()
        })
    }

    #[test]
    fn a_write_over_one_of_two_tips_is_built_from_the_tips_as_the_history_builds_it() {
        let _mode = force_history_fastpath(true);
        let descriptor = counter_descriptor();
        let (base, left, right) = forked_history(&descriptor, None);
        let mut history = vec![base.clone(), left.clone(), right.clone()];
        let mut previous = rebuilt_entry(&descriptor, &history);
        assert_eq!(previous.branch_frontier.len(), 2);
        assert!(
            previous.merge_artifacts.is_some(),
            "a row with two tips records the version they both descend from"
        );

        // Ten writes over the right tip, the left one never named: the shape a restored
        // device leaves a server in. The second also names what its tip was written
        // over, as a device that held both as tips does.
        let mut tip = right.clone();
        for step in 0..10u64 {
            let mut parents = vec![tip.batch_id()];
            if step == 1 {
                parents.extend(tip.parents.iter().copied());
            }
            let next = write_over(
                &base,
                &descriptor,
                parents,
                &[
                    Value::Text(format!("t{step}")),
                    Value::Integer(4 + step as i32),
                ],
                30 + step,
            );
            let built = forked_entry(&descriptor, &previous, &history, &next)
                .unwrap_or_else(|| panic!("write {step} over one of two tips was declined"));
            history.push(next.clone());
            assert_eq!(
                built,
                rebuilt_entry(&descriptor, &history),
                "write {step}: the entry built from the tips is not the entry the history builds"
            );
            assert_eq!(built.branch_frontier.len(), 2);
            previous = built;
            tip = next;
        }

        // And one write over both tips and a version they descend from: one tip again.
        let merge = write_over(
            &base,
            &descriptor,
            vec![left.batch_id(), base.batch_id(), tip.batch_id()],
            &[Value::Text("merged".into()), Value::Integer(20)],
            90,
        );
        let built = forked_entry(&descriptor, &previous, &history, &merge)
            .expect("a write over every tip was declined");
        history.push(merge.clone());
        assert_eq!(built, rebuilt_entry(&descriptor, &history));
        assert_eq!(built.branch_frontier, vec![merge.batch_id()]);
        assert_eq!(built.merge_artifacts, None);
        // From here the row has one tip and no trace of the fork: the serial path takes it.
        let after = write_over(
            &base,
            &descriptor,
            vec![merge.batch_id()],
            &[Value::Text("after".into()), Value::Integer(21)],
            100,
        );
        let serial = try_serial_fastpath_entry(Some(&built), &after)
            .expect("a healed row is a serial row again");
        history.push(after);
        assert_eq!(serial, rebuilt_entry(&descriptor, &history));
    }

    /// The order the tips are merged in is part of the answer. A counter column no tip
    /// has a value for, in a row whose tips share no ancestor, is credited to the newest
    /// tip — and the newest is by the versions' own clocks, not by which was written to
    /// this store last. The write below is older than the tip it leaves standing; the one
    /// after it is newer.
    #[test]
    fn the_tips_are_merged_in_the_order_of_their_clocks() {
        let _mode = force_history_fastpath(true);
        let descriptor = RowDescriptor::new(vec![
            ColumnDescriptor::new("title", ColumnType::Text),
            ColumnDescriptor::new("count", ColumnType::Integer)
                .nullable()
                .merge_strategy(ColumnMergeStrategy::Counter),
        ]);
        // Two creations of one row that never heard of each other: no common ancestor.
        let first = root_batch(
            &descriptor,
            &[Value::Text("first".into()), Value::Null],
            10,
            None,
        );
        let mut second = root_batch(
            &descriptor,
            &[Value::Text("second".into()), Value::Null],
            40,
            None,
        );
        second.row_id = first.row_id;
        let mut history = vec![first.clone(), second.clone()];
        let mut previous = rebuilt_entry(&descriptor, &history);
        assert_eq!(previous.branch_frontier.len(), 2);
        assert_eq!(
            types::MergeBase::decode(
                previous
                    .merge_artifacts
                    .as_deref()
                    .expect("a recorded base")
            ),
            Some(types::MergeBase { ancestor: None })
        );

        let mut tip = first.clone();
        for (step, at) in [(0, 30u64), (1, 50)] {
            let next = write_over(
                &first,
                &descriptor,
                vec![tip.batch_id()],
                &[Value::Text(format!("t{step}")), Value::Null],
                at,
            );
            let built = forked_entry(&descriptor, &previous, &history, &next)
                .unwrap_or_else(|| panic!("write {step} over one of two tips was declined"));
            history.push(next.clone());
            assert_eq!(
                built,
                rebuilt_entry(&descriptor, &history),
                "write {step} at {at}: the tips were merged in another order than the \
                 history merges them in"
            );
            previous = built;
            tip = next;
        }
    }

    #[test]
    fn a_write_the_tips_alone_cannot_resolve_is_left_to_the_history() {
        let _mode = force_history_fastpath(true);
        let descriptor = counter_descriptor();
        let (base, left, right) = forked_history(&descriptor, None);
        let third = write_over(
            &base,
            &descriptor,
            vec![base.batch_id()],
            &[Value::Text("third".into()), Value::Integer(9)],
            22,
        );
        let history = vec![base.clone(), left.clone(), right.clone(), third.clone()];
        let previous = rebuilt_entry(&descriptor, &history);
        assert_eq!(previous.branch_frontier.len(), 3);
        let values = [Value::Text("next".into()), Value::Integer(1)];
        let over = |parents: Vec<BatchId>| write_over(&base, &descriptor, parents, &values, 40);
        let declined = |row: &StoredRowBatch, why: &str| {
            assert_eq!(
                forked_entry(&descriptor, &previous, &history, row),
                None,
                "{why}"
            );
        };

        declined(
            &over(vec![left.batch_id(), right.batch_id()]),
            "two tips of three: the version the remaining tips have in common may move",
        );
        declined(
            &over(vec![base.batch_id()]),
            "no tip named: a new tip, and a common version the entry does not know",
        );
        // `other` is a version `right` does not descend from: it brings its own ancestry.
        let other = write_over(
            &base,
            &descriptor,
            vec![left.batch_id()],
            &[Value::Text("other".into()), Value::Integer(3)],
            23,
        );
        let mut wider = history.clone();
        wider.push(other.clone());
        let wider_previous = rebuilt_entry(&descriptor, &wider);
        assert_eq!(
            forked_entry(
                &descriptor,
                &wider_previous,
                &wider,
                &over(vec![right.batch_id(), left.batch_id()])
            ),
            None,
            "a second parent the named tip does not descend from"
        );

        let mut delete = over(vec![right.batch_id()]);
        delete.delete_kind = Some(DeleteKind::Soft);
        delete.is_deleted = true;
        declined(&delete, "a delete");
        let mut confirmed = over(vec![right.batch_id()]);
        confirmed.confirmed_tier = Some(DurabilityTier::Local);
        declined(&confirmed, "a version that carries a tier");
        let mut staged = over(vec![right.batch_id()]);
        staged.state = RowState::StagingPending;
        declined(&staged, "a staged version");

        // An entry stored before the common version was recorded.
        let mut unrecorded = previous.clone();
        unrecorded.merge_artifacts = None;
        assert_eq!(
            forked_entry(
                &descriptor,
                &unrecorded,
                &history,
                &over(vec![right.batch_id()])
            ),
            None,
            "an entry that does not say which version its tips have in common"
        );

        // An entry that points at a tier's own version of the row, though neither the row
        // readers see nor the incoming one carries a tier.
        for point in [
            |entry: &mut VisibleRowEntry, at| entry.worker_batch_id = Some(at),
            |entry: &mut VisibleRowEntry, at| entry.edge_batch_id = Some(at),
            |entry: &mut VisibleRowEntry, at| entry.global_batch_id = Some(at),
        ] {
            let mut pointed = previous.clone();
            point(&mut pointed, left.batch_id());
            assert_eq!(
                forked_entry(
                    &descriptor,
                    &pointed,
                    &history,
                    &over(vec![right.batch_id()])
                ),
                None,
                "an entry that points at a tier's version"
            );
        }

        // What the path reads by id must be there and be visible: a tip that is staged
        // or rejected by now, a recorded common version that is, or either one gone.
        let with = |change: &dyn Fn(&mut Vec<StoredRowBatch>)| {
            let mut changed = history.clone();
            change(&mut changed);
            forked_entry(
                &descriptor,
                &previous,
                &changed,
                &over(vec![right.batch_id()]),
            )
        };
        assert!(
            with(&|_| ()).is_some(),
            "the write these cases vary is itself taken"
        );
        assert_eq!(
            with(&|history| history[1].state = RowState::Rejected),
            None,
            "another tip that is no longer visible"
        );
        assert_eq!(
            with(&|history| {
                history.remove(1);
            }),
            None,
            "another tip the store does not hold"
        );
        assert_eq!(
            with(&|history| history[2].state = RowState::Rejected),
            None,
            "the tip the write replaces is no longer visible"
        );
        assert_eq!(
            with(&|history| {
                history.remove(2);
            }),
            None,
            "the tip the write replaces is one the store does not hold"
        );
        assert_eq!(
            with(&|history| history[0].state = RowState::Rejected),
            None,
            "a recorded common version that is no longer visible"
        );
        assert_eq!(
            with(&|history| {
                history.remove(0);
            }),
            None,
            "a recorded common version the store does not hold"
        );

        // Bytes this build cannot read where the common version is recorded: whatever
        // they say, neither a write over one tip nor a write over every tip is built
        // without the history — and a row with one tip is no different.
        let mut unknown = previous.clone();
        unknown.merge_artifacts = Some(vec![9, 9, 9]);
        for parents in [
            vec![right.batch_id()],
            vec![left.batch_id(), right.batch_id(), third.batch_id()],
        ] {
            let named = parents.len();
            assert_eq!(
                forked_entry(&descriptor, &unknown, &history, &over(parents)),
                None,
                "a write over {named} of three tips of an entry with bytes nobody can read"
            );
        }
        // A row some tier has a view of: that view is not carried by this path.
        let (tier_base, tier_left, tier_right) =
            forked_history(&descriptor, Some(DurabilityTier::EdgeServer));
        let tiered = vec![tier_base.clone(), tier_left, tier_right.clone()];
        let tiered_previous = rebuilt_entry(&descriptor, &tiered);
        assert_eq!(
            forked_entry(
                &descriptor,
                &tiered_previous,
                &tiered,
                &write_over(
                    &tier_base,
                    &descriptor,
                    vec![tier_right.batch_id()],
                    &values,
                    40
                )
            ),
            None,
            "a row a tier has its own view of"
        );
    }

    /// The path for forked rows takes only an entry with several tips that records what
    /// they descend from, and a row with one tip is never built by it.
    #[test]
    fn a_row_with_one_tip_is_never_built_from_its_tips() {
        let _mode = force_history_fastpath(true);
        let descriptor = counter_descriptor();
        let (base, left, right) = forked_history(&descriptor, None);
        let history = vec![base.clone(), left.clone(), right.clone()];
        let previous = rebuilt_entry(&descriptor, &history);
        assert_eq!(previous.branch_frontier.len(), 2);
        let values = [Value::Text("next".into()), Value::Integer(1)];
        let over = |parents: Vec<BatchId>| write_over(&base, &descriptor, parents, &values, 40);
        assert!(
            forked_entry(
                &descriptor,
                &previous,
                &history,
                &over(vec![left.batch_id(), right.batch_id(), base.batch_id()])
            )
            .is_some(),
            "a write over every tip and more, of an entry that records its ancestor, is taken"
        );

        // A row with one tip is never built here, whatever the write names besides the
        // tip. Its entry can stand for a history that has more tips — a delivered snapshot
        // taken over a tip that had parents leaves exactly this entry — and a write naming
        // the tip and other versions is where reading the history finds them again.
        let line = vec![base.clone(), left.clone()];
        let one_tip = rebuilt_entry(&descriptor, &line);
        assert_eq!(one_tip.branch_frontier, vec![left.batch_id()]);
        assert_eq!(
            forked_entry(
                &descriptor,
                &one_tip,
                &line,
                &over(vec![left.batch_id(), base.batch_id()])
            ),
            None,
            "a write over the only tip and a version under it"
        );
        // And a write over exactly that tip, the serial path's own, is not built by either
        // path from an entry with bytes nobody can read.
        let serial_write = over(vec![left.batch_id()]);
        assert!(
            try_serial_fastpath_entry(Some(&one_tip), &serial_write).is_some(),
            "the write this case varies is itself taken"
        );
        let mut unknown_line = one_tip.clone();
        unknown_line.merge_artifacts = Some(vec![9, 9, 9]);
        assert_eq!(
            try_serial_fastpath_entry(Some(&unknown_line), &serial_write)
                .or_else(|| { forked_entry(&descriptor, &unknown_line, &line, &serial_write) }),
            None,
            "a write over a row with one tip and bytes nobody can read"
        );
    }

    /// An entry with no tip names nothing a write could be over — and "every tip of
    /// none" is not a reason to take it.
    #[test]
    fn an_entry_with_no_tip_is_left_to_the_history() {
        let _mode = force_history_fastpath(true);
        let descriptor = counter_descriptor();
        let (base, left, right) = forked_history(&descriptor, None);
        let history = vec![base.clone(), left, right.clone()];
        let mut tipless = rebuilt_entry(&descriptor, &history);
        assert!(tipless.records_merge_base());
        tipless.branch_frontier.clear();
        let write = write_over(
            &base,
            &descriptor,
            vec![right.batch_id()],
            &[Value::Text("next".into()), Value::Integer(1)],
            40,
        );
        assert_eq!(
            forked_entry(&descriptor, &tipless, &history, &write),
            None,
            "an entry with no tip"
        );
    }

    /// A row with two tips whose newest won every column keeps no merged preview: its
    /// entry says only which version the tips descend from. A write naming exactly those
    /// tips has always been a serial write, tier state and all, and recording that version
    /// must not turn it into a read of the row's history.
    #[test]
    fn a_write_over_exactly_the_tips_of_a_tiered_row_is_still_a_serial_write() {
        let _mode = force_history_fastpath(true);
        let descriptor = user_descriptor();
        let tier = Some(DurabilityTier::EdgeServer);
        let base = root_batch(
            &descriptor,
            &[Value::Text("task".into()), Value::Boolean(false)],
            10,
            tier,
        );
        let left = serial_batch(
            &base,
            &descriptor,
            &[Value::Text("left".into()), Value::Boolean(false)],
            20,
            tier,
        );
        let right = serial_batch(
            &base,
            &descriptor,
            &[Value::Text("right".into()), Value::Boolean(true)],
            21,
            tier,
        );
        let mut history = vec![base.clone(), left.clone(), right.clone()];
        let previous = rebuilt_entry(&descriptor, &history);
        assert_eq!(previous.branch_frontier.len(), 2);
        assert!(
            previous.winner_batch_pool.is_empty() && previous.current_winner_ordinals.is_none(),
            "the newest tip won every column: this case needs an entry with nothing merged"
        );
        assert!(previous.merge_artifacts.is_some());

        let merge = StoredRowBatch::new(
            base.row_id,
            "main",
            vec![left.batch_id(), right.batch_id()],
            encode_row(
                &descriptor,
                &[Value::Text("merged".into()), Value::Boolean(true)],
            )
            .expect("encode row"),
            RowProvenance::for_update(&base.row_provenance(), "carol".to_string(), 30),
            HashMap::new(),
            RowState::VisibleDirect,
            tier,
        );
        assert_eq!(
            forked_entry(&descriptor, &previous, &history, &merge),
            None,
            "the path for forked rows carries no tier: only the serial path can take this"
        );
        let built = try_serial_fastpath_entry(Some(&previous), &merge)
            .expect("a write over exactly the tips of a tiered row was declined");
        history.push(merge);
        assert_eq!(built, rebuilt_entry(&descriptor, &history));

        // Bytes this build cannot read are not a merge base: the history decides.
        let mut unknown = previous.clone();
        unknown.merge_artifacts = Some(vec![9, 9, 9]);
        assert_eq!(try_serial_fastpath_entry(Some(&unknown), &history[3]), None);
    }

    /// OPEN DEFECT, older than the forked path and not touched by it (found 2026-10-02 by
    /// the storage differential once its streams were widened).
    ///
    /// A delivered snapshot supersedes the parentless versions of its lineage
    /// (`resolution::superseded_by_snapshot`), and each tier's view applies that rule
    /// over the versions the tier can see. A snapshot the tier cannot see supersedes
    /// nothing there, so a creation it hid from readers is still a tip of the tier's
    /// view. `fastpath::carried_tier_pointer` assumes the opposite — that a write over
    /// the whole frontier descends from every visible version — and carries no pointer.
    /// The rebuild merges the creation with the write in that tier's view, and with a
    /// counter column the merge is not the write: the creation's count is added again.
    ///
    /// Which of the two is the intended row is itself undecided: the tier view resurrects
    /// a version every reader stopped seeing.
    #[test]
    #[ignore = "open defect: a tier's view keeps a creation that a delivered snapshot superseded"]
    fn a_tier_view_keeps_a_creation_a_delivered_snapshot_superseded() {
        let _mode = force_history_fastpath(true);
        let descriptor = counter_descriptor();
        let creation = root_batch(
            &descriptor,
            &[Value::Text("created".into()), Value::Integer(5)],
            10,
            Some(DurabilityTier::GlobalServer),
        );
        let snapshot = StoredRowBatch::new(
            creation.row_id,
            "main",
            Vec::new(),
            encode_row(
                &descriptor,
                &[Value::Text("delivered".into()), Value::Integer(5)],
            )
            .expect("encode snapshot"),
            RowProvenance::for_update(&creation.row_provenance(), "bob".to_string(), 20),
            HashMap::new(),
            RowState::VisibleDirect,
            None,
        );
        let history = vec![creation.clone(), snapshot.clone()];
        let previous = rebuilt_entry(&descriptor, &history);
        assert_eq!(previous.branch_frontier, vec![snapshot.batch_id()]);

        let write = serial_batch(
            &snapshot,
            &descriptor,
            &[Value::Text("written".into()), Value::Integer(7)],
            30,
            Some(DurabilityTier::GlobalServer),
        );
        let carried = try_serial_fastpath_entry(Some(&previous), &write)
            .expect("a write over the only tip is a serial write");
        let mut extended = history;
        extended.push(write);
        assert_eq!(carried, rebuilt_entry(&descriptor, &extended));
    }

    #[test]
    fn history_fastpath_declines_linear_extension_after_resolved_fork_with_live_preview() {
        let _mode = force_history_fastpath(true);
        let descriptor = counter_descriptor();
        let base = root_batch(
            &descriptor,
            &[Value::Text("task".into()), Value::Integer(5)],
            10,
            Some(DurabilityTier::EdgeServer),
        );
        let left = serial_batch(
            &base,
            &descriptor,
            &[Value::Text("task".into()), Value::Integer(7)],
            20,
            Some(DurabilityTier::EdgeServer),
        );
        let right = serial_batch(
            &base,
            &descriptor,
            &[Value::Text("task".into()), Value::Integer(4)],
            21,
            Some(DurabilityTier::EdgeServer),
        );
        // The merge-commit resolves the fork on the unfiltered view, but the
        // edge-tier preview still merges {left, right} concurrently (the merge
        // commit is only Local-confirmed), keeping pool/ordinals live.
        let merge = StoredRowBatch::new(
            base.row_id,
            "main",
            vec![left.batch_id(), right.batch_id()],
            encode_row(
                &descriptor,
                &[Value::Text("task".into()), Value::Integer(6)],
            )
            .expect("encode merge row"),
            RowProvenance::for_update(&base.row_provenance(), "carol".to_string(), 30),
            HashMap::new(),
            RowState::VisibleDirect,
            Some(DurabilityTier::Local),
        );
        let history = vec![base.clone(), left.clone(), right.clone(), merge.clone()];
        let previous = rebuilt_entry(&descriptor, &history);
        assert_eq!(previous.current_row.batch_id(), merge.batch_id());
        assert_eq!(previous.branch_frontier, vec![merge.batch_id()]);
        assert!(previous.edge_winner_ordinals.is_some());
        assert!(!previous.winner_batch_pool.is_empty());

        // Linear extension over the resolved fork: guards 1–5 pass, the
        // never-forked guard declines, and the full path keeps the live
        // merged tier preview intact.
        let c = serial_batch(
            &merge,
            &descriptor,
            &[Value::Text("task".into()), Value::Integer(6)],
            40,
            None,
        );
        assert_eq!(try_serial_fastpath_entry(Some(&previous), &c), None);

        let mut extended = history;
        extended.push(c.clone());
        let rebuilt = rebuilt_entry(&descriptor, &extended);
        assert_eq!(rebuilt.current_row.batch_id(), c.batch_id());
        assert_eq!(rebuilt.worker_batch_id, Some(merge.batch_id()));
        assert!(rebuilt.edge_winner_ordinals.is_some());
        assert_eq!(rebuilt.winner_batch_pool, previous.winner_batch_pool);
    }

    #[test]
    fn history_fastpath_serial_appends_through_storage_hit_and_match_rebuild() {
        let table = "fastpath_docs";
        let descriptor = user_descriptor();
        let schema: Schema = [(TableName::new(table), TableSchema::new(descriptor.clone()))]
            .into_iter()
            .collect();
        let mut storage = MemoryStorage::new();
        let schema_hash = crate::test_support::persist_test_schema(&mut storage, &schema);
        let branch = BranchName::new("main");

        let root = root_batch(
            &descriptor,
            &[Value::Text("v0".into()), Value::Boolean(false)],
            10,
            None,
        );
        let row_id = root.row_id;
        storage
            .put_row_locator(
                row_id,
                Some(&RowLocator {
                    table: table.into(),
                    origin_schema_hash: Some(schema_hash),
                }),
            )
            .expect("row locator should persist");

        let assert_stored_matches_rebuild = |storage: &MemoryStorage| {
            let history = storage
                .scan_history_region(table, "main", HistoryScan::Row { row_id })
                .expect("scan history");
            let expected = VisibleRowEntry::rebuild_with_descriptor(&descriptor, &history)
                .expect("rebuild entry")
                .expect("visible entry present");
            let stored = storage
                .load_visible_region_entry(table, "main", row_id)
                .expect("load stored entry")
                .expect("stored entry present");
            assert_eq!(stored, expected, "stored entry diverges from rebuild");
        };

        // Fast path forced ON: every serial append after the first must hit,
        // and the stored entry must stay byte-equal to the full rebuild.
        let mut prev = root.clone();
        {
            let _mode = force_history_fastpath(true);
            let hits_before = HISTORY_FASTPATH_HITS.load(std::sync::atomic::Ordering::Relaxed);
            apply_row_batch(&mut storage, row_id, &branch, root, &[]).expect("apply root");
            for i in 0..6u64 {
                let next = serial_batch(
                    &prev,
                    &descriptor,
                    &[
                        Value::Text(format!("v{}", i + 1)),
                        Value::Boolean(i % 2 == 0),
                    ],
                    20 + 10 * i,
                    None,
                );
                apply_row_batch(&mut storage, row_id, &branch, next.clone(), &[])
                    .expect("apply serial append");
                assert_stored_matches_rebuild(&storage);
                prev = next;
            }
            let hits_after = HISTORY_FASTPATH_HITS.load(std::sync::atomic::Ordering::Relaxed);
            assert!(
                hits_after >= hits_before + 6,
                "expected all 6 serial appends to take the fast path \
                 (hits before {hits_before}, after {hits_after})"
            );
        }

        // Kill switch (forced OFF): same writes, same stored bytes.
        {
            let _mode = force_history_fastpath(false);
            for i in 0..3u64 {
                let next = serial_batch(
                    &prev,
                    &descriptor,
                    &[Value::Text(format!("off{i}")), Value::Boolean(i % 2 == 1)],
                    100 + 10 * i,
                    None,
                );
                apply_row_batch(&mut storage, row_id, &branch, next.clone(), &[])
                    .expect("apply serial append with fast path off");
                assert_stored_matches_rebuild(&storage);
                prev = next;
            }
        }
    }

    /// Legacy-row backfill regression (history-fastpaths design §6).
    ///
    /// Rows written by pre-fork-era code (or raw storage writes) have history
    /// rows but NO visible-region entry. The frontier-first read paths must
    /// keep answering correctly for them via their scan fallbacks, and the
    /// next `apply_row_batch` must lazily backfill the visible entry (the
    /// `rebuild_visible_entry_from_history` fallback inside
    /// `load_previous_visible_entry`), after which the O(1) serial fast path
    /// becomes eligible.
    #[test]
    fn legacy_history_without_visible_entry_backfills_on_next_apply() {
        // Held for the whole test: forces the fast path on AND serialises
        // access to the global hit counter asserted below.
        let _mode = force_history_fastpath(true);

        let table = "legacy_docs";
        let descriptor = user_descriptor();
        let schema: Schema = [(TableName::new(table), TableSchema::new(descriptor.clone()))]
            .into_iter()
            .collect();
        let mut storage = MemoryStorage::new();
        let schema_hash = crate::test_support::persist_test_schema(&mut storage, &schema);
        let branch = BranchName::new("main");

        let a = root_batch(
            &descriptor,
            &[Value::Text("v1".into()), Value::Boolean(false)],
            10,
            None,
        );
        let row_id = a.row_id;
        storage
            .put_row_locator(
                row_id,
                Some(&RowLocator {
                    table: table.into(),
                    origin_schema_hash: Some(schema_hash),
                }),
            )
            .expect("row locator should persist");
        let b = serial_batch(
            &a,
            &descriptor,
            &[Value::Text("v2".into()), Value::Boolean(true)],
            20,
            None,
        );
        // History only — deliberately NO visible-region entry.
        storage
            .append_history_region_rows(table, &[a.clone(), b.clone()])
            .expect("append legacy history rows");

        // Point reads see no visible entry (this is exactly the miss that
        // routes the provenance path onto its scan fallback) …
        assert_eq!(
            storage
                .load_visible_region_entry(table, "main", row_id)
                .expect("load entry"),
            None
        );
        assert_eq!(
            storage
                .load_visible_region_frontier(table, "main", row_id)
                .expect("load frontier"),
            None
        );
        // … but the scan fallbacks still answer correctly: the tip-ids read
        // (frontier-first, scan as safety net) and the provenance-shaped
        // "latest visible non-deleted row" scan both find b.
        assert_eq!(
            storage
                .scan_row_branch_tip_ids(table, "main", row_id)
                .expect("tip ids"),
            vec![b.batch_id()]
        );
        let provenance_row = storage
            .scan_history_row_batches(table, row_id)
            .expect("scan history")
            .into_iter()
            .filter(|row| row.state.is_visible() && row.delete_kind.is_none())
            .max_by_key(|row| (row.updated_at, row.batch_id()))
            .expect("legacy row has a visible version");
        assert_eq!(provenance_row.batch_id(), b.batch_id());

        // Next apply: `load_previous_visible_entry` rebuilds the entry from
        // history and the write persists it — the lazy backfill.
        let c = serial_batch(
            &b,
            &descriptor,
            &[Value::Text("v3".into()), Value::Boolean(false)],
            30,
            None,
        );
        apply_row_batch(&mut storage, row_id, &branch, c.clone(), &[])
            .expect("apply over legacy history");

        let history = storage
            .scan_history_region(table, "main", HistoryScan::Row { row_id })
            .expect("scan history after apply");
        let expected = VisibleRowEntry::rebuild_with_descriptor(&descriptor, &history)
            .expect("rebuild entry")
            .expect("visible entry present");
        let stored = storage
            .load_visible_region_entry(table, "main", row_id)
            .expect("load stored entry")
            .expect("backfilled entry present");
        assert_eq!(stored, expected, "backfilled entry diverges from rebuild");
        assert_eq!(
            storage
                .load_visible_region_frontier(table, "main", row_id)
                .expect("load frontier"),
            Some(vec![c.batch_id()])
        );
        assert_eq!(
            storage
                .scan_row_branch_tip_ids(table, "main", row_id)
                .expect("tip ids"),
            vec![c.batch_id()]
        );

        // With the entry persisted, the next serial append takes the O(1)
        // fast path.
        let hits_before = HISTORY_FASTPATH_HITS.load(std::sync::atomic::Ordering::Relaxed);
        let d = serial_batch(
            &c,
            &descriptor,
            &[Value::Text("v4".into()), Value::Boolean(true)],
            40,
            None,
        );
        apply_row_batch(&mut storage, row_id, &branch, d.clone(), &[])
            .expect("apply post-backfill serial append");
        let hits_after = HISTORY_FASTPATH_HITS.load(std::sync::atomic::Ordering::Relaxed);
        assert!(
            hits_after > hits_before,
            "post-backfill serial append should take the fast path \
             (hits before {hits_before}, after {hits_after})"
        );
        let stored = storage
            .load_visible_region_entry(table, "main", row_id)
            .expect("load stored entry")
            .expect("entry present after fast-path append");
        assert_eq!(stored.current_row.batch_id(), d.batch_id());
        assert_eq!(stored.branch_frontier, vec![d.batch_id()]);
    }

    // ─── in-place / patch fast paths ────────────────────────────────────────
    //
    // Fixtures for `fastpath::try_in_place_tip_update_entry` and the routing
    // in `patch_row_batch_state`. Same bar as the serial fixtures: a fast
    // entry must equal the full rebuild byte for byte, every unprovable shape
    // must decline, and `→ Rejected` must always take the full path.

    #[test]
    fn in_place_fastpath_tier_confirmation_matches_full_rebuild() {
        let _mode = force_history_fastpath(true);
        let descriptor = user_descriptor();
        let a = root_batch(
            &descriptor,
            &[Value::Text("v1".into()), Value::Boolean(false)],
            10,
            None,
        );
        let b = serial_batch(
            &a,
            &descriptor,
            &[Value::Text("v2".into()), Value::Boolean(true)],
            20,
            None,
        );
        let previous = rebuilt_entry(&descriptor, &[a.clone(), b.clone()]);
        assert_eq!(previous.worker_batch_id, None);
        assert_eq!(previous.edge_batch_id, None);
        assert_eq!(previous.global_batch_id, None);

        // Tier confirmation of the tip over an all-unconfirmed chain: every
        // tier set was provably empty (pointer None, old version
        // unsatisfying), so the confirmed tip satisfies each tier itself.
        let confirmed = b.accepted_transaction_output(DurabilityTier::GlobalServer);
        let fast = try_in_place_tip_update_entry(Some(&previous), &b, &confirmed)
            .expect("tier confirmation of the sole tip takes the in-place fast path");
        assert_eq!(
            fast,
            rebuilt_entry(&descriptor, &[a.clone(), confirmed.clone()])
        );
        assert_eq!(fast.branch_frontier, vec![b.batch_id()]);
        assert_eq!(fast.worker_batch_id, None);
        assert_eq!(fast.edge_batch_id, None);
        assert_eq!(fast.global_batch_id, None);

        // Next serial append re-arms the pointers (Fix A carry). The
        // following confirmation is the B3 confirm-over-confirmed-chain case
        // (v13-2): the confirmed sole tip causally dominates every visible
        // row, so each tier set it enters collapses to it and the pointers
        // flip to None in O(1), byte-equal to the full rebuild. This exact
        // shape used to decline on the `Some` pointer and pay an O(depth)
        // rebuild per confirm (the flatness-gate wall).
        let c = serial_batch(
            &confirmed,
            &descriptor,
            &[Value::Text("v3".into()), Value::Boolean(false)],
            30,
            None,
        );
        let previous = rebuilt_entry(&descriptor, &[a.clone(), confirmed.clone(), c.clone()]);
        assert_eq!(previous.worker_batch_id, Some(b.batch_id()));
        assert_eq!(previous.global_batch_id, Some(b.batch_id()));

        let confirmed_c = c.accepted_transaction_output(DurabilityTier::GlobalServer);
        let fast = try_in_place_tip_update_entry(Some(&previous), &c, &confirmed_c)
            .expect("confirm over a confirmed linear chain takes the in-place fast path");
        let full = rebuilt_entry(&descriptor, &[a, confirmed, confirmed_c.clone()]);
        assert_eq!(fast, full);
        assert_eq!(full.current_row, confirmed_c);
        assert_eq!(fast.worker_batch_id, None);
        assert_eq!(fast.edge_batch_id, None);
        assert_eq!(fast.global_batch_id, None);
    }

    #[test]
    fn in_place_fastpath_carries_pointers_when_tier_membership_unchanged() {
        let _mode = force_history_fastpath(true);
        let descriptor = user_descriptor();
        let a = root_batch(
            &descriptor,
            &[Value::Text("v1".into()), Value::Boolean(false)],
            10,
            Some(DurabilityTier::GlobalServer),
        );
        let b = serial_batch(
            &a,
            &descriptor,
            &[Value::Text("v2".into()), Value::Boolean(true)],
            20,
            Some(DurabilityTier::Local),
        );
        let previous = rebuilt_entry(&descriptor, &[a.clone(), b.clone()]);
        assert_eq!(previous.worker_batch_id, None);
        assert_eq!(previous.edge_batch_id, Some(a.batch_id()));
        assert_eq!(previous.global_batch_id, Some(a.batch_id()));

        // Same-tier re-confirmation (state VisibleDirect → VisibleTransactional,
        // tier unchanged): membership is unchanged for every tier, so all
        // pointers carry verbatim — including the deep `Some(a)` ones.
        let flipped = b.accepted_transaction_output(DurabilityTier::Local);
        assert!(flipped.state != b.state, "state must actually change");
        let fast = try_in_place_tip_update_entry(Some(&previous), &b, &flipped)
            .expect("membership-preserving state flip takes the in-place fast path");
        assert_eq!(fast, rebuilt_entry(&descriptor, &[a.clone(), flipped]));
        assert_eq!(fast.worker_batch_id, None);
        assert_eq!(fast.edge_batch_id, Some(a.batch_id()));
        assert_eq!(fast.global_batch_id, Some(a.batch_id()));
    }

    #[test]
    fn in_place_fastpath_tier_confirmation_dominates_tier_hole() {
        let _mode = force_history_fastpath(true);
        let descriptor = counter_descriptor();
        // r2 is a tier hole. Pre-v13-2's one-step tier frontier resurfaced
        // r1 as a concurrent global tip when r3 got confirmed ({r1, r3}, r2
        // not in the tier set) and merged it — double-counting r1's counter
        // delta into the preview — so the in-place path had to decline on
        // the carried `Some(r1)` pointer. Under causal domination (v13-2)
        // r1 is a proper ancestor of r3 through the full DAG, the confirmed
        // tip dominates the whole tier set, and the O(1) claim is provable
        // and byte-equal to the rebuild.
        let r1 = root_batch(
            &descriptor,
            &[Value::Text("task".into()), Value::Integer(2)],
            10,
            Some(DurabilityTier::GlobalServer),
        );
        let r2 = serial_batch(
            &r1,
            &descriptor,
            &[Value::Text("task".into()), Value::Integer(5)],
            20,
            None,
        );
        let r3 = serial_batch(
            &r2,
            &descriptor,
            &[Value::Text("task".into()), Value::Integer(7)],
            30,
            None,
        );
        let previous = rebuilt_entry(&descriptor, &[r1.clone(), r2.clone(), r3.clone()]);
        assert_eq!(previous.global_batch_id, Some(r1.batch_id()));
        assert_eq!(previous.global_winner_ordinals, None);
        assert!(previous.winner_batch_pool.is_empty());

        let confirmed = r3.accepted_transaction_output(DurabilityTier::GlobalServer);
        let fast = try_in_place_tip_update_entry(Some(&previous), &r3, &confirmed)
            .expect("tier confirmation over a holed chain takes the in-place fast path");

        let full = rebuilt_entry(&descriptor, &[r1, r2, confirmed.clone()]);
        assert_eq!(fast, full);
        assert_eq!(full.current_row, confirmed);
        assert_eq!(full.global_batch_id, None);
        assert_eq!(full.global_winner_ordinals, None);
        assert!(full.winner_batch_pool.is_empty());
    }

    #[test]
    fn in_place_fastpath_declines_non_tip_forked_lowered_and_content_shapes() {
        let _mode = force_history_fastpath(true);
        let descriptor = user_descriptor();
        let a = root_batch(
            &descriptor,
            &[Value::Text("v1".into()), Value::Boolean(false)],
            10,
            None,
        );
        let b = serial_batch(
            &a,
            &descriptor,
            &[Value::Text("v2".into()), Value::Boolean(true)],
            20,
            None,
        );
        let previous = rebuilt_entry(&descriptor, &[a.clone(), b.clone()]);

        // Confirming a NON-tip batch: the frontier guard declines.
        let confirmed_a = a.accepted_transaction_output(DurabilityTier::GlobalServer);
        assert_eq!(
            try_in_place_tip_update_entry(Some(&previous), &a, &confirmed_a),
            None
        );

        // Content changed beyond state/tier: declines even on the tip.
        let mut retitled = b.accepted_transaction_output(DurabilityTier::GlobalServer);
        retitled.data = encode_row(
            &descriptor,
            &[Value::Text("rewritten".into()), Value::Boolean(true)],
        )
        .expect("encode replacement row")
        .into();
        assert_eq!(
            try_in_place_tip_update_entry(Some(&previous), &b, &retitled),
            None
        );

        // Visible → invisible and invisible → visible are other paths'
        // business (full path and publish fast path respectively).
        let mut rejected = b.clone();
        rejected.state = RowState::Rejected;
        assert_eq!(
            try_in_place_tip_update_entry(Some(&previous), &b, &rejected),
            None
        );
        let mut staged_existing = b.clone();
        staged_existing.state = RowState::StagingPending;
        assert_eq!(
            try_in_place_tip_update_entry(Some(&previous), &staged_existing, &b),
            None
        );

        // Tier LOWERING (leaves the tier set): a removal event — declines.
        let mut b_global = b.clone();
        b_global.confirmed_tier = Some(DurabilityTier::GlobalServer);
        let mut b_edge = b.clone();
        b_edge.confirmed_tier = Some(DurabilityTier::EdgeServer);
        let previous_global = rebuilt_entry(&descriptor, &[a.clone(), b_global.clone()]);
        assert_eq!(
            try_in_place_tip_update_entry(Some(&previous_global), &b_global, &b_edge),
            None
        );

        // Without a previous entry there is nothing to update in place.
        let confirmed_b = b.accepted_transaction_output(DurabilityTier::GlobalServer);
        assert_eq!(try_in_place_tip_update_entry(None, &b, &confirmed_b), None);

        // A forked (two-tip) entry declines via the frontier guard.
        let left = serial_batch(
            &a,
            &descriptor,
            &[Value::Text("left".into()), Value::Boolean(false)],
            21,
            None,
        );
        let forked = rebuilt_entry(&descriptor, &[a, b.clone(), left]);
        assert_eq!(forked.branch_frontier.len(), 2);
        assert_eq!(
            try_in_place_tip_update_entry(Some(&forked), &b, &confirmed_b),
            None
        );
    }

    #[test]
    fn patch_fastpath_staging_publish_and_tier_bump_hit_through_storage() {
        let _mode = force_history_fastpath(true);
        let table = "patch_fastpath_docs";
        let descriptor = user_descriptor();
        let schema: Schema = [(TableName::new(table), TableSchema::new(descriptor.clone()))]
            .into_iter()
            .collect();
        let mut storage = MemoryStorage::new();
        let schema_hash = crate::test_support::persist_test_schema(&mut storage, &schema);
        let branch = BranchName::new("main");

        let root = root_batch(
            &descriptor,
            &[Value::Text("v0".into()), Value::Boolean(false)],
            10,
            None,
        );
        let row_id = root.row_id;
        storage
            .put_row_locator(
                row_id,
                Some(&RowLocator {
                    table: table.into(),
                    origin_schema_hash: Some(schema_hash),
                }),
            )
            .expect("row locator should persist");
        apply_row_batch(&mut storage, row_id, &branch, root.clone(), &[]).expect("apply root");

        let assert_stored_matches_rebuild = |storage: &MemoryStorage| {
            let history = storage
                .scan_history_region(table, "main", HistoryScan::Row { row_id })
                .expect("scan history");
            let expected = VisibleRowEntry::rebuild_with_descriptor(&descriptor, &history)
                .expect("rebuild entry")
                .expect("visible entry present");
            let stored = storage
                .load_visible_region_entry(table, "main", row_id)
                .expect("load stored entry")
                .expect("stored entry present");
            assert_eq!(stored, expected, "stored entry diverges from rebuild");
            expected
        };

        let mut staged = serial_batch(
            &root,
            &descriptor,
            &[Value::Text("staged".into()), Value::Boolean(true)],
            20,
            None,
        );
        staged.state = RowState::StagingPending;
        apply_row_batch(&mut storage, row_id, &branch, staged.clone(), &[]).expect("apply staged");

        use std::sync::atomic::Ordering;
        let hits_before = PATCH_FASTPATH_HITS.load(Ordering::Relaxed);
        let fallbacks_before = PATCH_FASTPATH_FALLBACKS.load(Ordering::Relaxed);

        // Staging publish (`runtime_core/writes.rs` shape): parents cover the
        // frontier, never-forked ⇒ the publish rides the domination fast path.
        let change = patch_row_batch_state(
            &mut storage,
            row_id,
            &branch,
            staged.batch_id(),
            Some(RowState::VisibleDirect),
            None,
        )
        .expect("publish staged batch")
        .expect("publish changes visibility");
        assert_eq!(change.row.batch_id(), staged.batch_id());
        let entry = assert_stored_matches_rebuild(&storage);
        assert_eq!(entry.current_row.batch_id(), staged.batch_id());
        assert_eq!(entry.branch_frontier, vec![staged.batch_id()]);

        // Pure tier bump on the (now published) tip: the in-place fast path.
        patch_row_batch_state(
            &mut storage,
            row_id,
            &branch,
            staged.batch_id(),
            None,
            Some(DurabilityTier::GlobalServer),
        )
        .expect("tier bump on tip");
        let entry = assert_stored_matches_rebuild(&storage);
        assert_eq!(
            entry.current_row.confirmed_tier,
            Some(DurabilityTier::GlobalServer)
        );
        assert_eq!(entry.worker_batch_id, None);
        assert_eq!(entry.global_batch_id, None);

        assert_eq!(
            PATCH_FASTPATH_HITS.load(Ordering::Relaxed),
            hits_before + 2,
            "publish and tier bump should both hit the patch fast path"
        );
        assert_eq!(
            PATCH_FASTPATH_FALLBACKS.load(Ordering::Relaxed),
            fallbacks_before
        );

        // Publish with a concurrent visible sibling: the flip IS a fork
        // event — the fast path declines and the full path merges both tips.
        let mut staged_two = serial_batch(
            &staged,
            &descriptor,
            &[Value::Text("staged-two".into()), Value::Boolean(false)],
            30,
            None,
        );
        staged_two.state = RowState::StagingPending;
        apply_row_batch(&mut storage, row_id, &branch, staged_two.clone(), &[])
            .expect("apply second staged batch");
        let sibling = serial_batch(
            &staged,
            &descriptor,
            &[Value::Text("sibling".into()), Value::Boolean(true)],
            31,
            None,
        );
        apply_row_batch(&mut storage, row_id, &branch, sibling.clone(), &[])
            .expect("apply visible sibling");

        patch_row_batch_state(
            &mut storage,
            row_id,
            &branch,
            staged_two.batch_id(),
            Some(RowState::VisibleDirect),
            None,
        )
        .expect("publish staged batch with concurrent sibling");
        let entry = assert_stored_matches_rebuild(&storage);
        assert_eq!(entry.branch_frontier.len(), 2, "both tips must survive");
        assert_eq!(
            PATCH_FASTPATH_FALLBACKS.load(Ordering::Relaxed),
            fallbacks_before + 1,
            "the forked publish must fall back to the full path"
        );
    }

    #[test]
    fn patch_fastpath_rejection_of_sole_tip_reexposes_ancestor() {
        let _mode = force_history_fastpath(true);
        let table = "patch_reject_docs";
        let descriptor = user_descriptor();
        let schema: Schema = [(TableName::new(table), TableSchema::new(descriptor.clone()))]
            .into_iter()
            .collect();
        let mut storage = MemoryStorage::new();
        let schema_hash = crate::test_support::persist_test_schema(&mut storage, &schema);
        let branch = BranchName::new("main");

        let a = root_batch(
            &descriptor,
            &[Value::Text("v1".into()), Value::Boolean(false)],
            10,
            Some(DurabilityTier::GlobalServer),
        );
        let row_id = a.row_id;
        storage
            .put_row_locator(
                row_id,
                Some(&RowLocator {
                    table: table.into(),
                    origin_schema_hash: Some(schema_hash),
                }),
            )
            .expect("row locator should persist");
        apply_row_batch(&mut storage, row_id, &branch, a.clone(), &[]).expect("apply a");
        let b = serial_batch(
            &a,
            &descriptor,
            &[Value::Text("v2".into()), Value::Boolean(true)],
            20,
            None,
        );
        apply_row_batch(&mut storage, row_id, &branch, b.clone(), &[]).expect("apply b");

        use std::sync::atomic::Ordering;
        let hits_before = PATCH_FASTPATH_HITS.load(Ordering::Relaxed);
        let fallbacks_before = PATCH_FASTPATH_FALLBACKS.load(Ordering::Relaxed);

        // Rejecting the sole frontier tip must take the full path and
        // re-expose the superseded ancestor as the visible row.
        let change = patch_row_batch_state(
            &mut storage,
            row_id,
            &branch,
            b.batch_id(),
            Some(RowState::Rejected),
            None,
        )
        .expect("reject tip")
        .expect("rejection changes visibility");
        assert_eq!(change.row.batch_id(), a.batch_id());
        assert_eq!(
            change.previous_row.as_ref().map(|row| row.batch_id()),
            Some(b.batch_id())
        );

        let history = storage
            .scan_history_region(table, "main", HistoryScan::Row { row_id })
            .expect("scan history");
        let expected = VisibleRowEntry::rebuild_with_descriptor(&descriptor, &history)
            .expect("rebuild entry")
            .expect("visible entry present");
        let stored = storage
            .load_visible_region_entry(table, "main", row_id)
            .expect("load stored entry")
            .expect("stored entry present");
        assert_eq!(stored, expected);
        assert_eq!(stored.current_row.batch_id(), a.batch_id());
        assert_eq!(stored.branch_frontier, vec![a.batch_id()]);

        // Rejections are full-path by design: not part of the fast-path
        // population, so neither counter moves.
        assert_eq!(PATCH_FASTPATH_HITS.load(Ordering::Relaxed), hits_before);
        assert_eq!(
            PATCH_FASTPATH_FALLBACKS.load(Ordering::Relaxed),
            fallbacks_before
        );

        // Rejecting the re-exposed root empties the visible set entirely.
        let change = patch_row_batch_state(
            &mut storage,
            row_id,
            &branch,
            a.batch_id(),
            Some(RowState::Rejected),
            None,
        )
        .expect("reject root");
        assert_eq!(change, None);
        assert_eq!(
            storage
                .load_visible_region_entry(table, "main", row_id)
                .expect("load stored entry"),
            None
        );
    }

    #[test]
    fn direct_row_writes_use_batch_identity() {
        let provenance = RowProvenance::for_insert("alice".to_string(), 100);
        let first = StoredRowBatch::new(
            ObjectId::from_uuid(Uuid::from_u128(101)),
            "main",
            Vec::new(),
            vec![1, 2, 3],
            provenance.clone(),
            HashMap::new(),
            RowState::VisibleDirect,
            Some(DurabilityTier::Local),
        );

        assert_eq!(
            first.batch_id(),
            first.batch_id,
            "direct visible rows should publish under their batch identity"
        );
    }
}
