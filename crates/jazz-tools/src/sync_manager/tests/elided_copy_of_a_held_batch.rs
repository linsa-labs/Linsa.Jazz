//! A row delivered for a query scope goes out with its parents cleared
//! (`sync_logic::scope_delivery_row`). When the receiver already holds that batch WITH its
//! parents — its own write coming back, most often — the copy used to replace the stored
//! row, and the batch forgot what it descends from. Its parent, named by nobody any more,
//! became a tip again; the next local write merged the two and named a parent that the
//! sender's delivered frontier had already let go of.
//!
//! Measured on a client store: `old_parents=1 new_parents=0 same_data=true`, then a write
//! with two parents, then 17 189 batches of one `users` row sent to the server.
//!
//! What a batch descends from is fixed when it is written. A copy that does not say is not
//! a copy that says "nothing".

use super::*;
use crate::row_histories::{RowState, apply_row_batch};
use crate::storage::SqliteStorage;

const CREATED_AT: u64 = 1_000;

fn store() -> SqliteStorage {
    let mut io = SqliteStorage::open(":memory:").expect("in-memory sqlite storage should open");
    persist_test_schema(&mut io, &users_test_schema());
    io
}

fn version(row_id: ObjectId, parents: Vec<BatchId>, at: u64, label: &str) -> StoredRowBatch {
    let provenance = if at == CREATED_AT {
        RowProvenance::for_insert("alice".to_string(), CREATED_AT)
    } else {
        RowProvenance::for_update(
            &RowProvenance::for_insert("alice".to_string(), CREATED_AT),
            "alice".to_string(),
            at,
        )
    };
    StoredRowBatch::new(
        row_id,
        "main",
        parents,
        encode_row(
            &users_test_schema()[&"users".into()].columns,
            &[Value::Text(label.to_string())],
        )
        .expect("test row should encode"),
        provenance.clone(),
        crate::metadata::row_provenance_metadata(&provenance, None)
            .into_iter()
            .collect(),
        RowState::VisibleDirect,
        None,
    )
}

fn apply(
    io: &mut SqliteStorage,
    row: &StoredRowBatch,
) -> crate::row_histories::ApplyRowBatchResult {
    apply_row_batch(io, row.row_id, &BranchName::new("main"), row.clone(), &[])
        .expect("row batch should apply")
}

/// Three versions of one row, written here: a creation and two updates.
fn written_here(io: &mut SqliteStorage) -> (ObjectId, [StoredRowBatch; 3]) {
    let row_id = ObjectId::new();
    create_test_row_with_id(io, row_id, Some(row_metadata("users")));
    let first = version(row_id, Vec::new(), CREATED_AT, "first");
    let second = version(row_id, vec![first.batch_id()], CREATED_AT + 1, "second");
    let third = version(row_id, vec![second.batch_id()], CREATED_AT + 2, "third");
    for row in [&first, &second, &third] {
        apply(io, row);
    }
    (row_id, [first, second, third])
}

/// The row as a query scope delivers it: the sender clears the parents, and the visible
/// region it reads the row from stores no metadata.
fn as_delivered(row: &StoredRowBatch) -> StoredRowBatch {
    let mut delivered = row.clone();
    delivered.parents.clear();
    delivered.metadata = Default::default();
    delivered
}

fn stored(io: &SqliteStorage, row: &StoredRowBatch) -> StoredRowBatch {
    io.load_history_row_batch("users", "main", row.row_id, row.batch_id())
        .expect("history read")
        .expect("the batch is stored")
}

fn tips(io: &SqliteStorage, row_id: ObjectId) -> Vec<BatchId> {
    io.load_visible_region_entry("users", "main", row_id)
        .expect("visible read")
        .expect("the row is visible")
        .branch_frontier
}

#[test]
fn a_delivered_copy_of_a_held_batch_leaves_its_parents_alone() {
    let mut io = store();
    let (row_id, [_, second, third]) = written_here(&mut io);
    assert_eq!(tips(&io, row_id), vec![third.batch_id()]);

    let result = apply(&mut io, &as_delivered(&third));

    assert_eq!(
        stored(&io, &third).parents.as_slice(),
        &[second.batch_id()],
        "the stored batch knows its parent; a copy that carries no parents does not know better"
    );
    assert_eq!(
        stored(&io, &third),
        third,
        "everything this store held for the batch is still held, metadata included"
    );
    assert_eq!(
        tips(&io, row_id),
        vec![third.batch_id()],
        "nothing was written concurrently: the row still has one tip, so the next local \
         write names one parent"
    );
    assert!(
        result.visibility_change.is_none(),
        "the copy brought nothing this store did not hold"
    );
}

#[test]
fn a_delivered_copy_still_brings_its_confirmation() {
    let mut io = store();
    let (row_id, [_, second, third]) = written_here(&mut io);

    let mut confirmed = as_delivered(&third);
    confirmed.confirmed_tier = Some(DurabilityTier::EdgeServer);
    apply(&mut io, &confirmed);

    let after = stored(&io, &third);
    assert_eq!(after.parents.as_slice(), &[second.batch_id()]);
    assert_eq!(after.metadata, third.metadata);
    assert_eq!(after.confirmed_tier, Some(DurabilityTier::EdgeServer));
    assert_eq!(tips(&io, row_id), vec![third.batch_id()]);
}

#[test]
fn a_delivered_copy_of_an_older_held_batch_does_not_reopen_its_parent() {
    // The copy is of the middle version: the tip above it must stay the only tip.
    let mut io = store();
    let (row_id, [first, second, third]) = written_here(&mut io);

    apply(&mut io, &as_delivered(&second));

    assert_eq!(stored(&io, &second).parents.as_slice(), &[first.batch_id()]);
    assert_eq!(tips(&io, row_id), vec![third.batch_id()]);
}

#[test]
fn a_batch_first_seen_as_a_delivered_copy_is_stored_as_it_came() {
    // Control: nothing is held for this batch, so there is no ancestry to keep.
    let mut io = store();
    let (row_id, [_, _, third]) = written_here(&mut io);

    let elsewhere = version(row_id, vec![third.batch_id()], CREATED_AT + 3, "elsewhere");
    apply(&mut io, &as_delivered(&elsewhere));

    let after = stored(&io, &elsewhere);
    assert!(after.parents.is_empty());
    assert!(after.metadata.is_empty());
}

#[test]
fn a_held_copy_takes_the_parents_when_they_arrive() {
    // Control, the other direction: the copy came first and the full batch later. Parents
    // that arrive are information; only their absence is not.
    let mut io = store();
    let (row_id, [_, _, third]) = written_here(&mut io);

    let elsewhere = version(row_id, vec![third.batch_id()], CREATED_AT + 3, "elsewhere");
    apply(&mut io, &as_delivered(&elsewhere));
    apply(&mut io, &elsewhere);

    assert_eq!(stored(&io, &elsewhere), elsewhere);
    assert_eq!(tips(&io, row_id), vec![elsewhere.batch_id()]);
}

/// On a server where the row has more than one tip, what a scope delivers is the merged
/// row under the id of the newest tip: a held batch's id with content the batch was not
/// written with. The content is the server's and is taken; where the batch stands in the
/// row's history is not something the copy speaks about, and stays.
#[test]
fn a_delivered_copy_with_other_content_keeps_the_batch_where_it_was() {
    let mut io = store();
    let (row_id, [_, second, third]) = written_here(&mut io);

    let merged = {
        let mut merged = as_delivered(&third);
        merged.data = version(row_id, Vec::new(), CREATED_AT + 2, "merged elsewhere")
            .data
            .clone();
        merged
    };
    apply(&mut io, &merged);

    let after = stored(&io, &third);
    assert_eq!(
        after.data, merged.data,
        "the server's content was not taken"
    );
    assert_eq!(
        after.parents.as_slice(),
        &[second.batch_id()],
        "the batch was cut off its parent"
    );
    assert_eq!(after.metadata, third.metadata);
    assert_eq!(
        tips(&io, row_id),
        vec![third.batch_id()],
        "the parent is a tip again, and the next local write would merge it"
    );
    let visible = io
        .load_visible_region_row("users", "main", row_id)
        .expect("visible read")
        .expect("the row is visible");
    assert_eq!(
        visible.data, merged.data,
        "the row does not show what was delivered"
    );
}
