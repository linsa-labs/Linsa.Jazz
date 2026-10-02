//! Graft missing history links for one row from source stores into a target store.
//!
//! The repair the engine otherwise lacks: a receiver missing one link of a row's
//! parent-linked chain rejects every later batch with `ParentNotFound`, forever. When the
//! missing links still exist somewhere — an old snapshot, another peer's store — copying
//! them back makes the chain contiguous and the next incoming batch applies normally.
//!
//! What a graft writes, per missing batch, all through the sanctioned mutation verb so the
//! pieces the APPLY PATH actually reads come along — the raw-table header, the per-batch
//! exact table locator (without which the parent point-lookup returns `None` even when the
//! bytes exist), and the local-batch index:
//!
//! - the history entry, re-encoded under the SOURCE's schema hash, never re-derived by the
//!   decode heuristic — an old-schema batch that happens to decode under the current
//!   descriptor would otherwise be silently re-homed under the wrong layout;
//! - its authoritative batch fate, only where the target has none;
//! - the row locator, only if the target lacks it.
//!
//! What it never writes: which version a row shows and which are its tips (they move
//! through the normal apply path when a batch lands that does not name every tip — one
//! that does is taken over the entry as it stands), secondary indexes (derived from
//! visibility), sealed submissions (deleted after settlement; recovery would re-validate
//! ancient ones against today's frontier), branch ords (store-local), local batch records
//! (authoring-side).
//!
//! One thing it takes away, before it adds anything: what the visible entry records
//! about the version its tips descend from (`VisibleRowEntry::recorded_merge_base`), on
//! every branch the source has a version of that the target cannot reach — the branches
//! a version may be added to, whether or not this run gets as far as adding it. That
//! record was made from a history the graft is about to change, and a write over one of
//! the tips would be merged on it without a look at the history; without it that write
//! reads the history. First, so that a graft that stops part-way leaves none behind: a
//! second run finds the versions the first one added and, with nothing left to add,
//! leaves the entry alone. The entry is stored again in the raw table it was read from,
//! under the descriptor it was read with, and is not placed anew by the row locator. A
//! read that found it by neither locator, in the table of some schema version, leaves
//! that table named by the entry's own locator on the way back: the same place, now
//! found directly.
//!
//! Conflicts are judged by AUTHORSHIP — branch, data, timestamp, author — not by bytes
//! (encoders drift), not by full struct equality (state and tier legitimately differ), and
//! not by content digest either: the digest covers parents, and a client's copy of a
//! delivered batch has its parents STRIPPED by the sender — normal, not divergence.
//! Parentless copies of non-root batches are skipped, never grafted: writing one would
//! hollow out the very chain being repaired.

use std::collections::{BTreeMap, BTreeSet};

use super::{
    OwnedVisibleRowBytes, Storage, StorageError, encode_history_row_bytes_with_context,
    load_visible_region_row_bytes_with_storage, prepared_row_write_context_for_schema_hash,
};
use crate::object::ObjectId;
use crate::row_histories::{RowState, StoredRowBatch};

/// What one graft run did. Run again, it grafts nothing and copies no fate, and counts
/// as present what the first run added.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct GraftReport {
    pub batches_grafted: usize,
    pub batches_already_present: usize,
    pub fates_copied: usize,
    /// Parent-stripped delivery duplicates in a source that were not usable as truth.
    pub stripped_copies_skipped: usize,
}

/// Copy the missing history links for `row_id` from `source` into `target`.
///
/// Both stores must be offline copies — the RocksDB lock enforces it for that backend.
/// Additive and idempotent: no version, fate or locator that exists is overwritten, and
/// a second run grafts nothing, and writes nothing on a branch where every version of
/// the source is within its reach. The one thing rewritten is the visible entry of a
/// branch a version is missing from, less its recorded merge base (see the module doc);
/// that, and a row locator the target lacked, are written before a refusal that comes
/// later in the run.
pub fn graft_row_history<T, S>(
    target: &mut T,
    source: &S,
    table: &str,
    row_id: ObjectId,
) -> Result<GraftReport, StorageError>
where
    T: Storage,
    S: Storage + ?Sized,
{
    let mut report = GraftReport::default();

    // The locator first: encode-time schema resolution hangs off it.
    let source_locator = source.load_row_locator(row_id)?;
    match (target.load_row_locator(row_id)?, source_locator.as_ref()) {
        (None, Some(locator)) => target.put_row_locator(row_id, Some(locator))?,
        (Some(existing), Some(incoming)) if existing.table != incoming.table => {
            return Err(StorageError::IoError(format!(
                "graft refused: target locates row {row_id} in {:?}, source in {:?} — this \
                 is divergence, not a missing link",
                existing.table, incoming.table
            )));
        }
        _ => {}
    }

    // Decoded batches for identity and state.
    let source_batches = source.scan_history_row_batches(table, row_id)?;

    // The target's existing batches, decoded: the conflict rule needs authorship fields.
    // A client-store copy of a delivered batch differs LEGITIMATELY in parents — the
    // sender strips them on delivery — so digest inequality alone is not divergence.
    let target_batches: BTreeMap<_, _> = target
        .scan_history_row_batches(table, row_id)?
        .into_iter()
        .map(|row| ((row.branch.to_string(), row.batch_id), row))
        .collect();

    // Before anything is added (see the module doc): the branches the source has a
    // version of that the target cannot reach. More than the branches this run adds to
    // — it may skip such a version, or refuse one — and never fewer.
    let mut branches: BTreeSet<&str> = BTreeSet::new();
    for row in &source_batches {
        let branch = row.branch.as_str();
        if !branches.contains(branch)
            && target
                .load_history_row_batch(table, branch, row_id, row.batch_id)?
                .is_none()
        {
            branches.insert(branch);
        }
    }
    for branch in branches {
        forget_recorded_merge_base(target, table, branch, row_id)?;
    }

    let same_authorship = |a: &StoredRowBatch, b: &StoredRowBatch| {
        a.branch == b.branch
            && a.data == b.data
            && a.updated_at == b.updated_at
            && a.updated_by == b.updated_by
    };

    for row in &source_batches {
        if matches!(row.state, RowState::StagingPending) {
            return Err(StorageError::IoError(format!(
                "graft refused: batch {:?} of row {row_id} is staging-pending — grafting \
                 an unsettled write is not a repair",
                row.batch_id
            )));
        }

        let key = (row.branch.to_string(), row.batch_id);

        // Authorship first, before any skip: a batch the target holds under the same id
        // with different content is divergence whether or not it is reachable. Parents are
        // deliberately not compared — a delivery-stripped source copy differs there and
        // that is normal.
        if let Some(existing) = target_batches.get(&key)
            && !same_authorship(existing, row)
        {
            return Err(StorageError::IoError(format!(
                "graft refused: batch {:?} of row {row_id} exists in the target with \
                 different content — this is divergence, not a missing link",
                row.batch_id
            )));
        }

        // Reachable in the target: done. Bytes present but unreachable fall through — the
        // rewrite through the sanctioned verb heals the missing locator.
        if target
            .load_history_row_batch(table, row.branch.as_str(), row_id, row.batch_id)?
            .is_some()
        {
            report.batches_already_present += 1;
            continue;
        }

        // A parentless copy of a non-root batch is a delivery-stripped duplicate, never a
        // source of truth: grafting it would hollow out the chain being repaired. The
        // genuine root also has no parents, but a root is only graftable into a row the
        // target does not know at all.
        if row.parents.is_empty() && !target_batches.is_empty() {
            report.stripped_copies_skipped += 1;
            continue;
        }

        // The schema hash this batch was actually encoded under, resolved the way the
        // READ path resolves it: the per-batch exact locator when one exists, the row
        // locator's origin hash otherwise. Never the decode heuristic.
        let schema_hash = source
            .load_history_row_batch_table_locator(row.branch.as_str(), row_id, row.batch_id)?
            .map(|locator| locator.schema_hash)
            .or_else(|| {
                source_locator
                    .as_ref()
                    .and_then(|locator| locator.origin_schema_hash)
            })
            .ok_or_else(|| {
                StorageError::IoError(format!(
                    "graft refused: no schema placement for batch {:?} — the source has                      neither an exact locator nor a row-locator origin hash",
                    row.batch_id
                ))
            })?;
        let context =
            prepared_row_write_context_for_schema_hash(target, table, schema_hash, row_id)?;
        let encoded = encode_history_row_bytes_with_context(&context, row)?;
        let rows = std::slice::from_ref(row);
        let encoded_rows = std::slice::from_ref(&encoded);
        target.apply_encoded_row_mutation(table, encoded_rows, &[], &[])?;
        target.index_local_batch_history_rows(table, rows, encoded_rows)?;
        report.batches_grafted += 1;

        // The fate, only where the target has none. A blind upsert could clobber a live
        // fate with a snapshot's rejected one and retro-reject a visible batch.
        if target
            .load_authoritative_batch_fate(row.batch_id)?
            .is_none()
            && let Some(fate) = source.load_authoritative_batch_fate(row.batch_id)?
        {
            target.upsert_authoritative_batch_fate(&fate)?;
            report.fates_copied += 1;
        }
    }

    Ok(report)
}

/// Stores the visible entry of `(branch, row)` again without the version it records its
/// tips as descending from, if it records one. Bytes this build cannot read stay: they
/// keep the entry away from every path that does not read the history.
fn forget_recorded_merge_base<T: Storage>(
    target: &mut T,
    table: &str,
    branch: &str,
    row_id: ObjectId,
) -> Result<(), StorageError> {
    let Some(entry) = target.load_visible_region_entry(table, branch, row_id)? else {
        return Ok(());
    };
    if !entry.records_merge_base() {
        return Ok(());
    }
    let entry = entry.without_merge_base();
    let Some(stored) = load_visible_region_row_bytes_with_storage(target, table, branch, row_id)?
    else {
        // A backend that keeps its entries decoded (`MemoryStorage`): there is no raw
        // table to put this one back into, and its reads follow no locator to one.
        return target.upsert_visible_region_rows(table, &[entry]);
    };
    // Back where it was read from, under the descriptor it was read with. Placing it
    // anew would ask the row locator, which may name another schema version than the
    // one the entry lives under, and move it there.
    let bytes = crate::row_histories::encode_flat_visible_row_entry(
        stored.user_descriptor.as_ref(),
        &entry,
    )
    .map_err(|err| StorageError::IoError(format!("encode flat visible row: {err}")))?;
    target.apply_encoded_row_mutation(table, &[], &[OwnedVisibleRowBytes { bytes, ..stored }], &[])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::MemoryStorage;
    use std::collections::HashMap;

    use crate::catalogue::CatalogueEntry;
    use crate::metadata::MetadataKey;
    use crate::metadata::ObjectType;
    use crate::metadata::RowProvenance;
    use crate::query_manager::types::{
        ColumnType, Schema, SchemaBuilder, SchemaHash, TableSchema, Value,
    };
    use crate::row_format::encode_row;
    use crate::row_histories::BatchId;
    use crate::storage::RowLocator;

    fn users_schema_v1() -> Schema {
        SchemaBuilder::new()
            .table(
                TableSchema::builder("users")
                    .column("id", ColumnType::Uuid)
                    .column("name", ColumnType::Text),
            )
            .build()
    }

    fn users_schema_v2() -> Schema {
        SchemaBuilder::new()
            .table(
                TableSchema::builder("users")
                    .column("id", ColumnType::Uuid)
                    .column("name", ColumnType::Text)
                    .column("email", ColumnType::Text),
            )
            .build()
    }

    fn persist_schema<H: Storage>(storage: &mut H, schema: &Schema) -> SchemaHash {
        let schema_hash = SchemaHash::compute(schema);
        storage
            .upsert_catalogue_entry(&CatalogueEntry {
                object_id: schema_hash.to_object_id(),
                metadata: HashMap::from([(
                    MetadataKey::Type.to_string(),
                    ObjectType::CatalogueSchema.to_string(),
                )]),
                content: crate::schema_manager::encode_schema(schema),
            })
            .expect("persist schema");
        schema_hash
    }

    fn chain(row_id: ObjectId, schema: &Schema, len: usize) -> Vec<StoredRowBatch> {
        let mut out = Vec::new();
        let mut parent: Option<BatchId> = None;
        for level in 0..len {
            let data = encode_row(
                &schema[&crate::query_manager::types::TableName::new("users")].columns,
                &[Value::Uuid(row_id), Value::Text(format!("v{level}"))],
            )
            .expect("encode row");
            let row = StoredRowBatch::new(
                row_id,
                "main",
                parent.into_iter().collect::<Vec<_>>(),
                data,
                RowProvenance::for_insert("author", 1_000 + level as u64),
                HashMap::new(),
                RowState::VisibleDirect,
                None,
            );
            parent = Some(row.batch_id());
            out.push(row);
        }
        out
    }

    /// The review's decisive scenario: a batch authored under one schema grafted into a
    /// target whose row locator points at another. A byte-copy closes the SCAN and still
    /// fails the POINT LOOKUP — the check the apply path actually runs — because the
    /// per-batch exact locator is missing. The sanctioned verb writes it.
    #[test]
    fn scan_green_is_not_enough_the_point_lookup_must_close_too() {
        let row_id = ObjectId::new();

        let mut source = MemoryStorage::new();
        let v1 = persist_schema(&mut source, &users_schema_v1());
        source
            .put_row_locator(
                row_id,
                Some(&RowLocator {
                    table: "users".into(),
                    origin_schema_hash: Some(v1),
                }),
            )
            .unwrap();
        let rows = chain(row_id, &users_schema_v1(), 3);
        source.append_history_region_rows("users", &rows).unwrap();

        let mut target = MemoryStorage::new();
        persist_schema(&mut target, &users_schema_v1());
        let v2 = persist_schema(&mut target, &users_schema_v2());
        // The target believes the row originates under v2: grafted v1 batches land in a
        // raw table the origin-hash fast path will not find.
        target
            .put_row_locator(
                row_id,
                Some(&RowLocator {
                    table: "users".into(),
                    origin_schema_hash: Some(v2),
                }),
            )
            .unwrap();

        let report = graft_row_history(&mut target, &source, "users", row_id).unwrap();
        assert_eq!(report.batches_grafted, 3);

        // Scan closure — chain_doctor's oracle.
        assert_eq!(
            target
                .scan_history_row_batches("users", row_id)
                .unwrap()
                .len(),
            3
        );
        // Point-lookup closure — the APPLY path's oracle, for every batch.
        for row in &rows {
            assert!(
                target
                    .load_history_row_batch("users", "main", row_id, row.batch_id())
                    .unwrap()
                    .is_some(),
                "batch {:?} is scan-visible but unreachable by the point lookup the apply \
                 path uses — the graft omitted the exact table locator, and every \
                 descendant of this batch still dies with ParentNotFound",
                row.batch_id()
            );
        }

        // Idempotence.
        let again = graft_row_history(&mut target, &source, "users", row_id).unwrap();
        assert_eq!(again.batches_grafted, 0, "a second run must write nothing");
        assert_eq!(again.batches_already_present, 3);
    }

    /// A row the target holds without one link of its chain — c0, c2 and two versions over
    /// c2, but not c1 — and a source that has all of it. The target's entry was built
    /// from what the target holds, so it records an ancestor found without c1.
    #[cfg(feature = "sqlite")]
    struct MissingLink {
        _dir: tempfile::TempDir,
        source: crate::storage::SqliteStorage,
        target: crate::storage::SqliteStorage,
        row_id: ObjectId,
        /// c0, c1, c2, left, right.
        rows: Vec<StoredRowBatch>,
        /// The schema version the target's entry was stored under.
        entry_stored_under: SchemaHash,
        /// The schema version the target's row locator names.
        row_located_under: SchemaHash,
    }

    #[cfg(feature = "sqlite")]
    impl MissingLink {
        /// `entry_elsewhere`: the target's entry lives under another schema version than
        /// its row locator names, with the same `users` — as a row does that was first
        /// shown before a change of schema that left its table alone.
        fn new(entry_elsewhere: bool) -> Self {
            use crate::row_histories::VisibleRowEntry;
            use crate::storage::SqliteStorage;

            let dir = tempfile::tempdir().expect("tempdir");
            let row_id = ObjectId::new();
            let schema = users_schema_v1();
            // The same `users` beside a table the other version does not have.
            let schema_beside = SchemaBuilder::new()
                .table(
                    TableSchema::builder("users")
                        .column("id", ColumnType::Uuid)
                        .column("name", ColumnType::Text),
                )
                .table(TableSchema::builder("notes").column("body", ColumnType::Text))
                .build();
            let locator = |hash| RowLocator {
                table: "users".into(),
                origin_schema_hash: Some(hash),
            };
            let users = &schema[&crate::query_manager::types::TableName::new("users")].columns;

            // c0 <- c1 <- c2, and two versions over c2.
            let mut rows = chain(row_id, &schema, 3);
            for (name, at) in [("left", 2_000), ("right", 2_001)] {
                let data = encode_row(users, &[Value::Uuid(row_id), Value::Text(name.to_string())])
                    .expect("encode row");
                rows.push(StoredRowBatch::new(
                    row_id,
                    "main",
                    vec![rows[2].batch_id()],
                    data,
                    RowProvenance::for_insert("author", at),
                    HashMap::new(),
                    RowState::VisibleDirect,
                    None,
                ));
            }

            let mut source = SqliteStorage::open(dir.path().join("source.sqlite")).unwrap();
            let v1 = persist_schema(&mut source, &schema);
            source.put_row_locator(row_id, Some(&locator(v1))).unwrap();
            source.append_history_region_rows("users", &rows).unwrap();

            let mut target = SqliteStorage::open(dir.path().join("target.sqlite")).unwrap();
            persist_schema(&mut target, &schema);
            let beside = persist_schema(&mut target, &schema_beside);
            let row_located_under = if entry_elsewhere { beside } else { v1 };
            target
                .put_row_locator(row_id, Some(&locator(row_located_under)))
                .unwrap();
            let held: Vec<StoredRowBatch> = rows
                .iter()
                .filter(|row| row.batch_id() != rows[1].batch_id())
                .cloned()
                .collect();
            target.append_history_region_rows("users", &held).unwrap();
            let built = VisibleRowEntry::rebuild_with_descriptor(users, &held)
                .unwrap()
                .expect("the row has a visible version");
            let context =
                prepared_row_write_context_for_schema_hash(&target, "users", v1, row_id).unwrap();
            let encoded =
                super::super::encode_visible_row_bytes_with_context(&context, &built).unwrap();
            target
                .apply_encoded_row_mutation("users", &[], &[encoded], &[])
                .unwrap();

            let fixture = Self {
                _dir: dir,
                source,
                target,
                row_id,
                rows,
                entry_stored_under: v1,
                row_located_under,
            };
            let entry = fixture.entry();
            // The two versions over c2, and c0: without c1 nothing the target holds stands
            // on it.
            assert_eq!(entry.branch_frontier.len(), 3);
            assert!(
                entry.records_merge_base(),
                "the fixture's entry records no ancestor: the gate would prove nothing"
            );
            assert_eq!(
                fixture.entry_lives_in(),
                vec![fixture.visible_table(v1)],
                "control: the entry is stored once, under the version it was written under"
            );
            fixture
        }

        fn entry(&self) -> crate::row_histories::VisibleRowEntry {
            self.target
                .load_visible_region_entry("users", "main", self.row_id)
                .unwrap()
                .expect("the row is visible")
        }

        fn visible_table(&self, hash: SchemaHash) -> String {
            crate::storage::RowRawTableId::new(
                crate::storage::RowRawTableKind::Visible,
                "users",
                hash,
            )
            .raw_table_name()
            .to_string()
        }

        /// Every raw table that holds a visible entry of the row.
        fn entry_lives_in(&self) -> Vec<String> {
            super::super::visible_row_raw_tables_holding(&self.target, "users", "main", self.row_id)
                .unwrap()
        }

        fn graft(&mut self) -> Result<GraftReport, StorageError> {
            graft_row_history(&mut self.target, &self.source, "users", self.row_id)
        }
    }

    /// A graft changes the history an entry with several tips was built from, and such an
    /// entry records the version its tips descend from: after the graft the record is
    /// gone and nothing else of the entry has changed, so the next write over one tip is
    /// merged on the history — which now says the creation is no tip.
    #[cfg(feature = "sqlite")]
    #[test]
    fn a_graft_takes_the_recorded_ancestor_off_the_entry_it_grafted_under() {
        let mut row = MissingLink::new(false);
        let before = row.entry();

        let report = row.graft().unwrap();
        assert_eq!(report.batches_grafted, 1);
        let after = row.entry();
        assert!(
            !after.records_merge_base(),
            "the entry still records an ancestor found in the history as it was before the graft"
        );
        assert_eq!(
            after,
            before.clone().without_merge_base(),
            "the graft changed more of the entry than what it records of its tips' ancestor"
        );
        assert_eq!(
            row.entry_lives_in(),
            vec![row.visible_table(row.entry_stored_under)],
            "the entry is not where it was"
        );

        // A write over one of the tips: merged on the history as it is now.
        let left = row.rows[3].batch_id();
        let right = row.rows[4].batch_id();
        let users =
            &users_schema_v1()[&crate::query_manager::types::TableName::new("users")].columns;
        let over_left = StoredRowBatch::new(
            row.row_id,
            "main",
            vec![left],
            encode_row(
                users,
                &[
                    Value::Uuid(row.row_id),
                    Value::Text("over left".to_string()),
                ],
            )
            .expect("encode row"),
            RowProvenance::for_insert("author", 3_000),
            HashMap::new(),
            RowState::VisibleDirect,
            None,
        );
        crate::row_histories::apply_row_batch(
            &mut row.target,
            row.row_id,
            &crate::object::BranchName::new("main"),
            over_left.clone(),
            &[],
        )
        .expect("the write applies");
        let written = row.entry();
        let mut tips = written.branch_frontier.clone();
        tips.sort();
        let mut expected = vec![right, over_left.batch_id()];
        expected.sort();
        assert_eq!(
            tips, expected,
            "the creation is still a tip: the write was merged on what the entry said before the graft"
        );
        assert!(written.records_merge_base());
    }

    /// The entry goes back where it was read from. A row whose entry lives under another
    /// schema version than its row locator names — the read finds it by its own locator —
    /// is not moved to the version the row locator names, and both locators read as they
    /// did. (An entry found by neither locator is not built here: its way back leaves
    /// its own locator naming the table it was found in.)
    #[cfg(feature = "sqlite")]
    #[test]
    fn a_graft_leaves_the_entry_under_the_schema_version_it_lives_under() {
        let mut row = MissingLink::new(true);
        assert_ne!(row.entry_stored_under, row.row_located_under);
        let before = row.entry();
        let located = |row: &MissingLink| {
            (
                row.target.load_row_locator(row.row_id).unwrap(),
                row.target
                    .load_visible_row_table_locator("main", row.row_id)
                    .unwrap()
                    .map(|locator| locator.schema_hash),
            )
        };
        let located_before = located(&row);
        assert_eq!(
            located_before.1,
            Some(row.entry_stored_under),
            "control: the entry is found by a locator of its own"
        );

        let report = row.graft().unwrap();
        assert_eq!(report.batches_grafted, 1);
        assert_eq!(
            row.entry_lives_in(),
            vec![row.visible_table(row.entry_stored_under)],
            "the graft moved the entry to another schema version"
        );
        assert_eq!(located(&row), located_before, "the graft changed a locator");
        assert_eq!(row.entry(), before.without_merge_base());
    }

    /// The recorded ancestor is taken off before anything is added: a graft that stops —
    /// here refused, at a version of the source that has not settled — may have added
    /// versions already, and a second run that finds them present adds nothing and would
    /// have no reason to come back for the entry.
    #[cfg(feature = "sqlite")]
    #[test]
    fn a_graft_that_stops_has_taken_the_recorded_ancestor_off_first() {
        let mut row = MissingLink::new(false);
        let users =
            &users_schema_v1()[&crate::query_manager::types::TableName::new("users")].columns;
        let unsettled = StoredRowBatch::new(
            row.row_id,
            "main",
            vec![row.rows[2].batch_id()],
            encode_row(
                users,
                &[
                    Value::Uuid(row.row_id),
                    Value::Text("unsettled".to_string()),
                ],
            )
            .expect("encode row"),
            RowProvenance::for_insert("author", 2_500),
            HashMap::new(),
            RowState::StagingPending,
            None,
        );
        row.source
            .append_history_region_rows("users", &[unsettled])
            .unwrap();

        row.graft()
            .expect_err("a source with an unsettled version is refused");
        assert!(
            !row.entry().records_merge_base(),
            "the graft stopped and left the entry recording an ancestor"
        );
    }

    /// A run that finds every version of the source within reach on a branch leaves that
    /// branch's entry alone, whatever it records: it changes no history there.
    #[cfg(feature = "sqlite")]
    #[test]
    fn a_graft_with_nothing_to_add_leaves_the_entry_as_it_is() {
        let mut row = MissingLink::new(false);
        let recording = row.entry();
        assert_eq!(row.graft().unwrap().batches_grafted, 1);
        // An entry that records an ancestor — the one from before the first run, put
        // back; which ancestor it records is beside the point here.
        row.target
            .upsert_visible_region_rows("users", std::slice::from_ref(&recording))
            .unwrap();
        assert!(row.entry().records_merge_base());

        let again = row.graft().unwrap();
        assert_eq!(again.batches_grafted, 0);
        assert_eq!(again.batches_already_present, 5);
        assert_eq!(
            row.entry(),
            recording,
            "a graft that added nothing rewrote the entry"
        );
    }

    /// The strip is per branch. A row with versions on three branches, two of which the
    /// target misses a link of and one it holds whole: after the graft the two have lost
    /// what their entries recorded of an ancestor, and the third has kept it. In memory:
    /// the branches are chosen by the same code on every backend, from what its point
    /// lookup reaches; how an entry goes back into a raw table, and that it goes before
    /// anything is added, is held by the gates above.
    #[test]
    fn a_graft_takes_the_recorded_ancestor_off_each_branch_missing_a_version_and_off_no_whole_one()
    {
        use crate::row_histories::VisibleRowEntry;

        let row_id = ObjectId::new();
        let schema = users_schema_v1();
        let users = &schema[&crate::query_manager::types::TableName::new("users")].columns;
        let locator = |hash| RowLocator {
            table: "users".into(),
            origin_schema_hash: Some(hash),
        };
        // c0 <- c1 <- c2 and two versions over c2, all on one branch.
        let forked = |branch: &str, at: u64| {
            let version = |name: String, parents: Vec<BatchId>, at: u64| {
                StoredRowBatch::new(
                    row_id,
                    branch,
                    parents,
                    encode_row(users, &[Value::Uuid(row_id), Value::Text(name)])
                        .expect("encode row"),
                    RowProvenance::for_insert("author", at),
                    HashMap::new(),
                    RowState::VisibleDirect,
                    None,
                )
            };
            let mut rows: Vec<StoredRowBatch> = Vec::new();
            for level in 0..3 {
                let parents = rows.last().map(|row| row.batch_id()).into_iter().collect();
                rows.push(version(format!("{branch} v{level}"), parents, at + level));
            }
            for (name, later) in [("left", 10), ("right", 11)] {
                rows.push(version(
                    format!("{branch} {name}"),
                    vec![rows[2].batch_id()],
                    at + later,
                ));
            }
            rows
        };

        let mut source = MemoryStorage::new();
        let v1 = persist_schema(&mut source, &schema);
        source.put_row_locator(row_id, Some(&locator(v1))).unwrap();
        let mut target = MemoryStorage::new();
        persist_schema(&mut target, &schema);
        target.put_row_locator(row_id, Some(&locator(v1))).unwrap();
        let mut entries = Vec::new();
        for (branch, at, misses_a_link) in [
            ("main", 1_000, true),
            ("draft", 2_000, true),
            ("whole", 3_000, false),
        ] {
            let rows = forked(branch, at);
            source.append_history_region_rows("users", &rows).unwrap();
            let held: Vec<StoredRowBatch> = rows
                .iter()
                .filter(|row| !misses_a_link || row.batch_id() != rows[1].batch_id())
                .cloned()
                .collect();
            target.append_history_region_rows("users", &held).unwrap();
            let built = VisibleRowEntry::rebuild_with_descriptor(users, &held)
                .unwrap()
                .expect("the row has a visible version");
            assert!(
                built.records_merge_base(),
                "branch {branch}: the fixture's entry records no ancestor"
            );
            target
                .upsert_visible_region_rows("users", std::slice::from_ref(&built))
                .unwrap();
            entries.push((branch, misses_a_link, built));
        }

        let report = graft_row_history(&mut target, &source, "users", row_id).unwrap();
        assert_eq!(report.batches_grafted, 2);
        for (branch, missed_a_link, built) in entries {
            let after = target
                .load_visible_region_entry("users", branch, row_id)
                .unwrap()
                .expect("the row is visible");
            if missed_a_link {
                assert_eq!(
                    after,
                    built.without_merge_base(),
                    "branch {branch}: a version was added under an entry that still records an ancestor"
                );
            } else {
                assert_eq!(
                    after, built,
                    "branch {branch}: nothing was added to it, and its entry was rewritten"
                );
            }
        }
    }

    /// Only what this build wrote is taken off. Bytes it cannot read keep the entry away
    /// from every path that does not read the history, and they stay.
    #[cfg(feature = "sqlite")]
    #[test]
    fn a_graft_leaves_what_it_cannot_read_on_the_entry() {
        let mut row = MissingLink::new(false);
        let mut entry = row.entry();
        entry.merge_artifacts = Some(vec![0xff, 0xff, 0xff]);
        assert!(!entry.records_merge_base());
        row.target
            .upsert_visible_region_rows("users", &[entry.clone()])
            .unwrap();

        row.graft().unwrap();
        assert_eq!(
            row.entry().merge_artifacts,
            entry.merge_artifacts,
            "the graft took bytes it cannot read off the entry"
        );
    }

    /// The same on a backend that keeps its visible entries decoded, where there is no raw
    /// table to put the entry back into.
    #[test]
    fn a_graft_takes_the_recorded_ancestor_off_an_entry_kept_decoded() {
        use crate::row_histories::VisibleRowEntry;

        let row_id = ObjectId::new();
        let schema = users_schema_v1();
        let users = &schema[&crate::query_manager::types::TableName::new("users")].columns;
        let locator = |hash| RowLocator {
            table: "users".into(),
            origin_schema_hash: Some(hash),
        };
        let mut rows = chain(row_id, &schema, 3);
        for (name, at) in [("left", 2_000), ("right", 2_001)] {
            let data = encode_row(users, &[Value::Uuid(row_id), Value::Text(name.to_string())])
                .expect("encode row");
            rows.push(StoredRowBatch::new(
                row_id,
                "main",
                vec![rows[2].batch_id()],
                data,
                RowProvenance::for_insert("author", at),
                HashMap::new(),
                RowState::VisibleDirect,
                None,
            ));
        }
        let mut source = MemoryStorage::new();
        let v1 = persist_schema(&mut source, &schema);
        source.put_row_locator(row_id, Some(&locator(v1))).unwrap();
        source.append_history_region_rows("users", &rows).unwrap();

        let mut target = MemoryStorage::new();
        persist_schema(&mut target, &schema);
        target.put_row_locator(row_id, Some(&locator(v1))).unwrap();
        let held: Vec<StoredRowBatch> = rows
            .iter()
            .filter(|row| row.batch_id() != rows[1].batch_id())
            .cloned()
            .collect();
        target.append_history_region_rows("users", &held).unwrap();
        let built = VisibleRowEntry::rebuild_with_descriptor(users, &held)
            .unwrap()
            .expect("the row has a visible version");
        assert!(built.records_merge_base());
        target
            .upsert_visible_region_rows("users", std::slice::from_ref(&built))
            .unwrap();

        let report = graft_row_history(&mut target, &source, "users", row_id).unwrap();
        assert_eq!(report.batches_grafted, 1);
        let after = target
            .load_visible_region_entry("users", "main", row_id)
            .unwrap()
            .expect("the row is visible");
        assert_eq!(after, built.without_merge_base());
    }

    /// The decisive backend for the decisive scenario: on the raw-table backends the point
    /// lookup resolves through the row locator's origin hash FIRST and the per-batch exact
    /// locator second. MemoryStorage overrides the lookup with a plain map read, so only
    /// this variant has teeth — a graft that omitted exact locators passes the memory
    /// variant and fails here.
    #[cfg(feature = "sqlite")]
    #[test]
    fn the_point_lookup_closes_on_a_raw_table_backend_too() {
        use crate::storage::SqliteStorage;

        let dir = tempfile::tempdir().expect("tempdir");
        let row_id = ObjectId::new();

        let mut source = SqliteStorage::open(dir.path().join("source.sqlite")).unwrap();
        let v1 = persist_schema(&mut source, &users_schema_v1());
        source
            .put_row_locator(
                row_id,
                Some(&RowLocator {
                    table: "users".into(),
                    origin_schema_hash: Some(v1),
                }),
            )
            .unwrap();
        let rows = chain(row_id, &users_schema_v1(), 3);
        source.append_history_region_rows("users", &rows).unwrap();

        let mut target = SqliteStorage::open(dir.path().join("target.sqlite")).unwrap();
        persist_schema(&mut target, &users_schema_v1());
        let v2 = persist_schema(&mut target, &users_schema_v2());
        target
            .put_row_locator(
                row_id,
                Some(&RowLocator {
                    table: "users".into(),
                    origin_schema_hash: Some(v2),
                }),
            )
            .unwrap();

        let report = graft_row_history(&mut target, &source, "users", row_id).unwrap();
        assert_eq!(report.batches_grafted, 3);

        for row in &rows {
            assert!(
                target
                    .load_history_row_batch("users", "main", row_id, row.batch_id())
                    .unwrap()
                    .is_some(),
                "batch {:?} is scan-visible but unreachable by the point lookup the apply \
                 path uses — the graft omitted the exact table locator, and every \
                 descendant of this batch still dies with ParentNotFound",
                row.batch_id()
            );
        }

        let again = graft_row_history(&mut target, &source, "users", row_id).unwrap();
        assert_eq!(again.batches_grafted, 0, "a second run must write nothing");
    }

    /// Same batch id with different content is divergence, and the graft must refuse it.
    #[test]
    fn a_content_conflict_aborts_the_graft() {
        let row_id = ObjectId::new();
        let schema = users_schema_v1();

        let mut source = MemoryStorage::new();
        let v1 = persist_schema(&mut source, &schema);
        source
            .put_row_locator(
                row_id,
                Some(&RowLocator {
                    table: "users".into(),
                    origin_schema_hash: Some(v1),
                }),
            )
            .unwrap();
        let rows = chain(row_id, &schema, 2);
        source.append_history_region_rows("users", &rows).unwrap();

        let mut target = MemoryStorage::new();
        persist_schema(&mut target, &schema);
        target
            .put_row_locator(
                row_id,
                Some(&RowLocator {
                    table: "users".into(),
                    origin_schema_hash: Some(v1),
                }),
            )
            .unwrap();
        // The same batch id holds DIFFERENT content in the target.
        let mut forged = rows[0].clone();
        forged.data = encode_row(
            &schema[&crate::query_manager::types::TableName::new("users")].columns,
            &[Value::Uuid(row_id), Value::Text("forged".into())],
        )
        .unwrap()
        .into();
        target
            .append_history_region_rows("users", std::slice::from_ref(&forged))
            .unwrap();

        let error = graft_row_history(&mut target, &source, "users", row_id).unwrap_err();
        assert!(
            error.to_string().contains("divergence"),
            "expected a divergence refusal, got: {error}"
        );
    }
}
