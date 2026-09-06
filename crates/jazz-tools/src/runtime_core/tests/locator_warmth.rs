//! v18 item 5 (D): a recovered row-table locator is kept — warm in the process and
//! persisted in the store — so the ladder walks a split row once, not on every read.
//!
//! Companion of `locator_ladder_heal.rs` (the open-defect gate this item arms). The fixture
//! is the same crossing: a row inserted under generation A, a write under generation B that
//! moves `__row_locator` to B's family while branch A keeps its own head, so a read on
//! branch A asks a locator that names the wrong family and falls to the ladder.
//!
//! Internal on purpose: the ladder is a storage-internal fallback with no public observer
//! but its counter (`LOCATOR_LADDER_RECOVERIES`), and whether a pointer was persisted is a
//! question to the store. `SqliteStorage`, because `MemoryStorage` never executes the ladder.

use super::support::{docs_v2, read_on, runtime_over, split_store};
use super::*;
use crate::storage::SqliteStorage;

/// Ladder walks this store served. Per store since v18 item 5: the process-global
/// `LOCATOR_LADDER_RECOVERIES` is bumped by every store in the binary, so a parallel test
/// pollutes the delta, and a dropped `Box` forward leaves it moving while the store's own
/// counter does not (design v17 § retraction of v16).
fn recoveries(core: &RuntimeCore<SqliteStorage, NoopScheduler>) -> u64 {
    core.storage().visible_ladder_recoveries_for_test()
}

/// G-D2. A read OUTSIDE a pass: the first read ladders (that is what the ladder is for),
/// the store keeps what it found, and after a restart it answers from the persisted pointer
/// without walking again. Red on the base tree at the last assertion (the ladder never
/// writes the pointer back).
#[test]
fn a_recovered_locator_survives_a_restart() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let path = dir.path().join("warm.sqlite");
    let (core, row_id, branch_a) = split_store(&path, "locator-restart");

    let before = recoveries(&core);
    let first = core
        .storage()
        .load_visible_region_row("docs", branch_a.as_str(), row_id)
        .expect("visible row readable")
        .expect("fixture: branch A must still serve a head for the row");
    assert!(
        recoveries(&core) > before,
        "fixture precondition: the first read must walk the ladder, else the split was not \
         reproduced"
    );
    drop(core);
    let storage = SqliteStorage::open(&path).expect("reopen");
    let core = runtime_over(docs_v2(), "locator-restart", storage);
    let after_restart = core
        .storage()
        .load_visible_region_row("docs", branch_a.as_str(), row_id)
        .expect("visible row readable")
        .expect("the row is still visible on branch A after the restart");
    assert_eq!(
        after_restart.batch_id(),
        first.batch_id(),
        "the same head must be served"
    );
    assert_eq!(
        recoveries(&core),
        0,
        "a locator the ladder recovered must be persisted: after a restart the read must \
         answer from the pointer, not walk the families again"
    );
}

/// G-D2 + G-C4. The same recovery INSIDE a settle pass (a one-shot read on branch A): the
/// pass that laddered must hand its pointer to the durability barrier — flagged as pending,
/// committed by the next `batched_tick` — and the restart must not walk again. Red on the
/// base tree at the pending-flush assertion.
///
/// Internal on purpose: the observable is "the pass FLAGGED a write for the barrier rather
/// than committing it itself", which is `has_storage_write_pending_flush` — a core-private
/// latch no client API exposes. From outside, a locator committed by the pass and one
/// committed by the next barrier are indistinguishable until a crash lands between them.
#[test]
fn a_recovery_inside_a_pass_is_handed_to_the_durability_barrier() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let path = dir.path().join("warm-pass.sqlite");
    let (mut core, row_id, branch_a) = split_store(&path, "locator-pass");
    let before = recoveries(&core);
    read_on(&mut core, &branch_a);
    assert!(
        recoveries(&core) > before,
        "fixture precondition: the read on branch A must walk the ladder inside the pass, \
         else the pass did not point-load the split row"
    );
    assert!(
        core.has_storage_write_pending_flush(),
        "a pass whose ladder recovered a locator wrote that pointer and must be flagged for \
         the durability barrier"
    );
    core.batched_tick();
    assert!(
        core.storage().is_autocommit_for_test(),
        "the barrier must commit the pass's transaction"
    );
    drop(core);
    let storage = SqliteStorage::open(&path).expect("reopen");
    let core = runtime_over(docs_v2(), "locator-pass", storage);
    core.storage()
        .load_visible_region_row("docs", branch_a.as_str(), row_id)
        .expect("visible row readable")
        .expect("the row is still visible on branch A after the restart");
    assert_eq!(
        recoveries(&core),
        0,
        "the pointer the pass recovered must have reached the store"
    );
}

/// What a store answers for the row on `branch`: the head's batch and whether it is a
/// tombstone (deletes are tombstones here — `StoredRowBatch::is_deleted` — not absences).
fn answer(
    storage: &SqliteStorage,
    branch: &BranchName,
    row_id: ObjectId,
) -> Option<(BatchId, bool)> {
    storage
        .load_visible_region_row("docs", branch.as_str(), row_id)
        .expect("visible row readable")
        .map(|row| (row.batch_id(), row.is_deleted))
}

/// G-D3. A warm pointer must never change an answer. Heads are per branch, and a write on
/// branch A applied by the generation-B engine moves that head out of family A into the
/// live family ("moved the head instead of forking it", defect 27) and realigns the
/// pointer — exactly the write that would leave a warm `(A, row)` entry naming the family
/// the head just left. After the ladder warmed the entry, an update and then a delete on
/// branch A must be read the same way by the warm process and by a cold connection with no
/// map at all, and both must serve the new head. Green on the base tree (there is no warm
/// map yet); red when a read-through on the map is added without the write path keeping it
/// current — the eviction in `put_visible_row_table_locator` and `delete_visible_region_row`
/// and the insert in `apply_encoded_row_mutation` are what make the read-through safe.
#[test]
fn a_warm_locator_never_changes_an_answer() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let path = dir.path().join("evict.sqlite");
    let (mut core, row_id, branch_a) = split_store(&path, "locator-evict");
    let on_branch_a = WriteContext {
        target_branch_name: Some(branch_a.as_str().to_string()),
        ..WriteContext::from_session(Session::new("alice"))
    };
    // Warm the pointer through the ladder.
    let before = recoveries(&core);
    core.storage()
        .load_visible_region_row("docs", branch_a.as_str(), row_id)
        .expect("visible row readable")
        .expect("fixture: branch A serves a head");
    assert!(
        recoveries(&core) > before,
        "fixture precondition: the read must walk the ladder"
    );
    let updated = core
        .update(
            row_id,
            vec![("body".to_string(), Value::Text("moved".to_string()))],
            Some(&on_branch_a),
        )
        .expect("the update on branch A applies");
    core.batched_tick();
    core.immediate_tick();
    core.batched_tick();
    let warm = answer(core.storage(), &branch_a, row_id);
    let cold = answer(
        &SqliteStorage::open(&path).expect("cold reopen"),
        &branch_a,
        row_id,
    );
    assert_eq!(
        warm, cold,
        "after an update the warm process and a cold connection disagree"
    );
    assert_eq!(
        warm.map(|(batch, _)| batch),
        Some(updated),
        "the update's head must be the one served on branch A"
    );
    core.delete(row_id, Some(&on_branch_a))
        .expect("the row deletes on branch A");
    core.batched_tick();
    core.immediate_tick();
    core.batched_tick();
    let warm = answer(core.storage(), &branch_a, row_id);
    let cold = answer(
        &SqliteStorage::open(&path).expect("cold reopen"),
        &branch_a,
        row_id,
    );
    assert_eq!(
        warm, cold,
        "after a delete the warm process and a cold connection disagree"
    );
    assert!(
        warm.is_some_and(|(_, deleted)| deleted),
        "the delete's tombstone must be the head served on branch A"
    );
}
