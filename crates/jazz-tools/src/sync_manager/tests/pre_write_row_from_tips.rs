//! The row a client's write is judged against is read from what the write names, not from
//! everything those versions descend from.
//!
//! `pre_batch_visible_row` gives the permission check its old content. A write naming one
//! parent reads that parent. A write naming two walked every visible version either
//! descends from and merged that set's tips — and a device that holds two tips of a row
//! names both in its next write. On a `users` row that presence beats have grown to
//! 20 000 versions that is 20 000 point reads to learn what two of the versions, or the
//! row the store already shows, say.
//!
//! The read from the tips answers three shapes and leaves the rest to the walk. It may
//! never answer differently from the walk: the last test holds the two together over random
//! histories.
//!
//! The other tests here are about the write itself rather than the row it is judged
//! against: what applying a write to a row with many tips reads (it needs the same long
//! row on the same store); the open defect that makes an entry with one tip something
//! neither read may trust, and what a write over such an entry and more does about it;
//! and the entry the cleanup of a rejected batch leaves, on a row stored under one schema
//! version and under two.

use super::*;
use crate::query_manager::settle_cost::HISTORY_ENTRIES_ON_THREAD;
use crate::row_histories::{
    RowState, apply_row_batch, force_history_fastpath, patch_row_batch_state,
};
use crate::storage::SqliteStorage;

const CREATED_AT: u64 = 1_000;

/// Versions the row has before anything is counted: far above every count asserted.
const DEPTH: usize = 200;

/// Versions a write may have read to be judged: the ones it names, each by its id, and
/// each of them looked up once more to be sure it is held. Measured at 4 and 6.
const BY_ID: u64 = 8;

/// Tips the row has in the test of what applying a write reads.
const TIPS: usize = 12;

fn store() -> SqliteStorage {
    let mut io = SqliteStorage::open(":memory:").expect("in-memory sqlite storage should open");
    persist_test_schema(&mut io, &users_test_schema());
    io
}

fn version(row_id: ObjectId, parents: Vec<BatchId>, at: u64, label: &str) -> StoredRowBatch {
    let creation = RowProvenance::for_insert("alice".to_string(), CREATED_AT);
    let provenance = if parents.is_empty() && at == CREATED_AT {
        creation
    } else {
        RowProvenance::for_update(&creation, "alice".to_string(), at)
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

fn apply(io: &mut SqliteStorage, row: &StoredRowBatch) {
    apply_row_batch(io, row.row_id, &BranchName::new("main"), row.clone(), &[])
        .expect("row batch should apply");
}

/// A row written `DEPTH` times in a line; the last version is returned.
fn long_row(io: &mut SqliteStorage) -> (ObjectId, StoredRowBatch) {
    let row_id = ObjectId::new();
    create_test_row_with_id(io, row_id, Some(row_metadata("users")));
    let mut tip = version(row_id, Vec::new(), CREATED_AT, "v0");
    apply(io, &tip);
    for step in 1..DEPTH {
        let next = version(
            row_id,
            vec![tip.batch_id()],
            CREATED_AT + step as u64,
            &format!("v{step}"),
        );
        apply(io, &next);
        tip = next;
    }
    (row_id, tip)
}

/// What the read cost in versions of the row, and what it answered.
fn read(io: &SqliteStorage, row: &StoredRowBatch, from_tips: bool) -> (u64, Option<Vec<u8>>) {
    let before = HISTORY_ENTRIES_ON_THREAD.get();
    let answer = SyncManager::new()
        .pre_batch_visible_row_read(io, "users", row, from_tips)
        .map(|row| row.data.to_vec());
    (HISTORY_ENTRIES_ON_THREAD.get() - before, answer)
}

#[test]
fn a_write_over_both_tips_of_a_row_is_judged_against_the_row_the_store_shows() {
    let mode = force_history_fastpath(true);
    let mut io = store();
    let (row_id, tip) = long_row(&mut io);
    let at = CREATED_AT + DEPTH as u64;
    let left = version(row_id, vec![tip.batch_id()], at, "left");
    let right = version(row_id, vec![tip.batch_id()], at + 1, "right");
    apply(&mut io, &left);
    apply(&mut io, &right);
    let shown = io
        .load_visible_region_entry("users", "main", row_id)
        .expect("visible read")
        .expect("the row is visible");
    assert_eq!(shown.branch_frontier.len(), 2);

    let write = version(
        row_id,
        vec![left.batch_id(), right.batch_id()],
        at + 2,
        "over both",
    );
    let (walked, by_walk) = read(&io, &write, false);
    assert!(
        walked >= DEPTH as u64,
        "the walk no longer reads the history ({walked} versions): this gate compares \
         against nothing"
    );
    let (versions, by_tips) = read(&io, &write, true);
    assert_eq!(
        by_tips, by_walk,
        "the write is judged against another row than the one its parents resolve to"
    );
    assert_eq!(by_tips.as_deref(), Some(&shown.current_row.data[..]));
    assert!(
        versions <= BY_ID,
        "a write over two tips read {versions} versions of a row to be judged"
    );

    // `JAZZ_HISTORY_FASTPATH=0` takes this read away with the other shortcuts.
    drop(mode);
    let _mode = force_history_fastpath(false);
    let (versions, switched_off) = read(&io, &write, true);
    assert_eq!(switched_off, by_walk);
    assert!(
        versions >= DEPTH as u64,
        "with the fast paths switched off the write was still judged from the tips \
         ({versions} versions read)"
    );
}

#[test]
fn a_write_over_a_tip_and_the_version_under_it_is_judged_against_the_tip() {
    let _mode = force_history_fastpath(true);
    let mut io = store();
    let (row_id, tip) = long_row(&mut io);
    let at = CREATED_AT + DEPTH as u64;
    let over = version(row_id, vec![tip.batch_id()], at, "over");
    apply(&mut io, &over);
    // Another writer's tip, so that the write below does not name every tip of the row.
    let aside = version(row_id, vec![tip.batch_id()], at + 1, "aside");
    apply(&mut io, &aside);

    let write = version(
        row_id,
        vec![tip.batch_id(), over.batch_id()],
        at + 2,
        "over a tip and its parent",
    );
    let (walked, by_walk) = read(&io, &write, false);
    assert!(walked >= DEPTH as u64);
    let (versions, by_tips) = read(&io, &write, true);
    assert_eq!(by_tips, by_walk);
    assert_eq!(by_tips.as_deref(), Some(&over.data[..]));
    assert!(
        versions <= BY_ID,
        "a write over a tip and its parent read {versions} versions of a row to be judged"
    );
}

/// A write naming a version this store does not hold is refused for it, and there is no
/// row to judge it against. The walk says so only when it comes to that parent, and it
/// takes the parents from the end of the list: named first, the absent one is reached
/// after every version the tip descends from.
#[test]
fn a_write_naming_a_version_that_is_not_held_reads_no_history_to_be_refused() {
    let _mode = force_history_fastpath(true);
    let mut io = store();
    let (row_id, tip) = long_row(&mut io);
    let at = CREATED_AT + DEPTH as u64;
    let absent = BatchId::new();

    for (order, parents) in [
        ("first", vec![absent, tip.batch_id()]),
        ("last", vec![tip.batch_id(), absent]),
    ] {
        let write = version(row_id, parents, at, "over a version nobody holds");
        let (walked, by_walk) = read(&io, &write, false);
        assert_eq!(by_walk, None);
        if order == "first" {
            assert!(
                walked >= DEPTH as u64,
                "the walk no longer reads the history ({walked} versions): this gate \
                 compares against nothing"
            );
        }
        let (versions, by_tips) = read(&io, &write, true);
        assert_eq!(by_tips, None);
        assert!(
            versions <= BY_ID,
            "a write naming a version that is not held {order} read {versions} versions \
             of a row to be refused"
        );
    }
}

/// Applying a write to a row with many tips reads the tips and the one version they have
/// in common, each by its id — not the versions between, however many they are.
#[test]
fn a_write_to_a_row_with_many_tips_reads_the_tips_and_not_their_history() {
    let mut io = store();
    let (row_id, tip) = long_row(&mut io);
    let at = CREATED_AT + DEPTH as u64;
    // Writers that all wrote over the same version and never over each other.
    let mut newest = tip.clone();
    for writer in 0..TIPS {
        newest = version(
            row_id,
            vec![tip.batch_id()],
            at + writer as u64,
            &format!("w{writer}"),
        );
        apply(&mut io, &newest);
    }
    let tips = |io: &SqliteStorage| {
        io.load_visible_region_entry("users", "main", row_id)
            .expect("visible read")
            .expect("the row is visible")
            .branch_frontier
            .len()
    };
    assert_eq!(tips(&io), TIPS);
    let cost = |io: &mut SqliteStorage, row: &StoredRowBatch| {
        let before = HISTORY_ENTRIES_ON_THREAD.get();
        apply(io, row);
        HISTORY_ENTRIES_ON_THREAD.get() - before
    };

    // What the same write costs read from the history: the number the rest is held against.
    let from_history = {
        let _mode = force_history_fastpath(false);
        let next = version(
            row_id,
            vec![newest.batch_id()],
            at + 100,
            "from the history",
        );
        let versions = cost(&mut io, &next);
        newest = next;
        versions
    };
    assert!(
        from_history >= DEPTH as u64,
        "a write built from the history read {from_history} versions: this gate compares \
         against nothing"
    );

    let _mode = force_history_fastpath(true);
    for step in 0..5u64 {
        let next = version(
            row_id,
            vec![newest.batch_id()],
            at + 101 + step,
            &format!("over the newest tip {step}"),
        );
        let versions = cost(&mut io, &next);
        assert!(
            versions <= TIPS as u64 + 4,
            "write {step} over one of {TIPS} tips read {versions} versions of a row with \
             {DEPTH} under them"
        );
        assert_eq!(tips(&io), TIPS);
        newest = next;
    }
}

/// OPEN DEFECT, older than the read from the tips, and the reason neither it nor the path
/// that builds an entry from the tips trusts an entry with one tip.
///
/// A copy of a version delivered for a query comes without its parents. Over a tip that
/// has parents the history keeps both as tips: nothing says the copy descends from the
/// tip. `fastpath::snapshot_dominates_frontier` is to decline exactly there, and asks the
/// entry whether its tip has parents — but a store that keeps entries in the flat visible
/// encoding reads every tip back without them, takes the write, and leaves one tip where
/// the history has two. `MemoryStorage` keeps the entry as it was built and declines.
#[test]
#[ignore = "open defect: on a store that encodes its entries, a delivered copy hides a tip \
            that has parents"]
fn a_delivered_copy_leaves_a_tip_that_has_parents_standing_on_every_store() {
    fn tips_after<H: Storage>(io: &mut H) -> usize {
        persist_test_schema(io, &users_test_schema());
        let row_id = ObjectId::new();
        create_test_row_with_id(io, row_id, Some(row_metadata("users")));
        let creation = version(row_id, Vec::new(), CREATED_AT, "created");
        let written = version(
            row_id,
            vec![creation.batch_id()],
            CREATED_AT + 10,
            "written",
        );
        // Another device's later version, as a query delivers it: no parents.
        let delivered = version(row_id, Vec::new(), CREATED_AT + 20, "delivered");
        for row in [creation, written, delivered] {
            apply_row_batch(io, row_id, &BranchName::new("main"), row, &[])
                .expect("row batch should apply");
        }
        io.load_visible_region_entry("users", "main", row_id)
            .expect("visible read")
            .expect("the row is visible")
            .branch_frontier
            .len()
    }
    let _mode = force_history_fastpath(true);
    let memory = tips_after(&mut MemoryStorage::new());
    let sqlite = tips_after(
        &mut SqliteStorage::open(":memory:").expect("in-memory sqlite storage should open"),
    );
    assert_eq!(
        sqlite, memory,
        "the same three writes leave the row {sqlite} tips on SQLite and {memory} in memory"
    );
}

/// A row with one tip is not built from its tips, because of the defect above: after a
/// delivered copy hid a tip, a write over the copy and any other version is where reading
/// the history finds the hidden tip again. Built from the entry instead, the write would
/// be the row's only tip and the hidden one would stay hidden through every later write.
#[test]
fn a_write_over_a_delivered_copy_and_more_puts_back_the_tip_the_copy_hid() {
    let _mode = force_history_fastpath(true);
    let mut io = store();
    let row_id = ObjectId::new();
    create_test_row_with_id(&mut io, row_id, Some(row_metadata("users")));
    let creation = version(row_id, Vec::new(), CREATED_AT, "created");
    let written = version(
        row_id,
        vec![creation.batch_id()],
        CREATED_AT + 10,
        "written",
    );
    let delivered = version(row_id, Vec::new(), CREATED_AT + 20, "delivered");
    for row in [&creation, &written, &delivered] {
        apply(&mut io, row);
    }
    let tips = |io: &SqliteStorage| {
        let mut tips = io
            .load_visible_region_entry("users", "main", row_id)
            .expect("visible read")
            .expect("the row is visible")
            .branch_frontier;
        tips.sort();
        tips
    };
    assert_eq!(
        tips(&io),
        vec![delivered.batch_id()],
        "the delivered copy no longer hides the tip it was applied over: the open defect \
         this gate stands on is cured, and the gate needs another way to an entry that \
         has fewer tips than its history"
    );

    let over = version(
        row_id,
        vec![delivered.batch_id(), creation.batch_id()],
        CREATED_AT + 30,
        "over the copy and the creation",
    );
    apply(&mut io, &over);
    let mut both = vec![written.batch_id(), over.batch_id()];
    both.sort();
    assert_eq!(
        tips(&io),
        both,
        "the write was built from an entry that hid a tip, and the tip stays hidden"
    );
}

/// The cleanup of a batch this store itself rejected rebuilds the row's entry from the
/// versions stored under one schema version — not the branch's history. Its entry
/// therefore says nothing of what its tips descend from: neither the read from the tips
/// nor the path that builds an entry from them takes it at its word, and the next write
/// over one of its tips reads the history. (A write naming exactly its tips needs no
/// ancestor and is taken as before: the tips themselves are still what the entry says.)
#[test]
fn the_entry_a_local_rejection_leaves_records_no_ancestor() {
    /// Three writers over one version, the third rejected by the cleanup: whether the
    /// entry it leaves records what its two tips descend from.
    fn records_after_rejection<H: Storage>(io: &mut H) -> bool {
        persist_test_schema(io, &users_test_schema());
        let row_id = ObjectId::new();
        create_test_row_with_id(io, row_id, Some(row_metadata("users")));
        let creation = version(row_id, Vec::new(), CREATED_AT, "created");
        let over = |at: u64, label: &str| version(row_id, vec![creation.batch_id()], at, label);
        let local = over(CREATED_AT + 3, "local");
        for row in [
            creation.clone(),
            over(CREATED_AT + 1, "left"),
            over(CREATED_AT + 2, "right"),
            local.clone(),
        ] {
            apply_row_batch(io, row_id, &BranchName::new("main"), row, &[])
                .expect("row batch should apply");
        }
        let entry = |io: &H| {
            io.load_visible_region_entry("users", "main", row_id)
                .expect("visible read")
                .expect("the row is visible")
        };
        assert!(
            entry(io).records_merge_base(),
            "three tips built from the history record what they descend from"
        );
        assert!(
            io.patch_exact_row_batch_for_schema_hash(
                "users",
                users_schema_hash(),
                "main",
                row_id,
                local.batch_id(),
                Some(RowState::Rejected),
                None,
            )
            .expect("the rejected batch is patched")
        );
        let left = entry(io);
        assert_eq!(left.branch_frontier.len(), 2);
        left.records_merge_base()
    }
    let _mode = force_history_fastpath(true);
    assert!(
        !records_after_rejection(&mut MemoryStorage::new()),
        "in memory, the entry the cleanup left says what its tips descend from"
    );
    assert!(
        !records_after_rejection(
            &mut SqliteStorage::open(":memory:").expect("in-memory sqlite storage should open")
        ),
        "on SQLite, the entry the cleanup left says what its tips descend from"
    );

    // And on a long row: the write after the cleanup reads the history, once.
    let mut io = store();
    let (row_id, tip) = long_row(&mut io);
    let at = CREATED_AT + DEPTH as u64;
    let left = version(row_id, vec![tip.batch_id()], at, "left");
    let right = version(row_id, vec![tip.batch_id()], at + 1, "right");
    let local = version(row_id, vec![tip.batch_id()], at + 2, "local");
    for row in [&left, &right, &local] {
        apply(&mut io, row);
    }
    io.patch_exact_row_batch_for_schema_hash(
        "users",
        users_schema_hash(),
        "main",
        row_id,
        local.batch_id(),
        Some(RowState::Rejected),
        None,
    )
    .expect("the rejected batch is patched");
    let over_both = version(
        row_id,
        vec![left.batch_id(), right.batch_id()],
        at + 3,
        "over both",
    );
    assert_eq!(
        SyncManager::pre_batch_visible_row_from_tips(&io, "users", &over_both),
        None,
        "the row a write over both tips is judged against was read from that entry"
    );
    let cost = |io: &mut SqliteStorage, row: &StoredRowBatch| {
        let before = HISTORY_ENTRIES_ON_THREAD.get();
        apply(io, row);
        HISTORY_ENTRIES_ON_THREAD.get() - before
    };
    let first = version(row_id, vec![right.batch_id()], at + 4, "first after");
    let versions = cost(&mut io, &first);
    assert!(
        versions >= DEPTH as u64,
        "the first write after the cleanup read {versions} versions: it was built from an \
         entry that had not read the branch's history"
    );
    let second = version(row_id, vec![first.batch_id()], at + 5, "second after");
    let versions = cost(&mut io, &second);
    assert!(
        versions <= BY_ID,
        "the second write after the cleanup read {versions} versions of the row"
    );

    // And a write naming exactly the tips the cleanup left, on a row of its own: the
    // apply takes it over the entry as it stands, with no look at the history, as it did
    // before entries recorded anything. (What an inbound write is judged against is
    // another read, and for this entry it walks: asserted above.)
    let mut io = store();
    let (row_id, tip) = long_row(&mut io);
    let left = version(row_id, vec![tip.batch_id()], at, "left");
    let right = version(row_id, vec![tip.batch_id()], at + 1, "right");
    let local = version(row_id, vec![tip.batch_id()], at + 2, "local");
    for row in [&left, &right, &local] {
        apply(&mut io, row);
    }
    io.patch_exact_row_batch_for_schema_hash(
        "users",
        users_schema_hash(),
        "main",
        row_id,
        local.batch_id(),
        Some(RowState::Rejected),
        None,
    )
    .expect("the rejected batch is patched");
    let over_both = version(
        row_id,
        vec![left.batch_id(), right.batch_id()],
        at + 3,
        "over both",
    );
    let versions = cost(&mut io, &over_both);
    assert!(
        versions <= BY_ID,
        "a write naming exactly the tips the cleanup left read {versions} versions"
    );
    let entry = io
        .load_visible_region_entry("users", "main", row_id)
        .unwrap()
        .expect("the row is visible");
    assert_eq!(entry.branch_frontier, vec![over_both.batch_id()]);
    assert_eq!(&entry.current_row.data[..], &over_both.data[..]);
}

/// Where what the cleanup reads is NOT the row's history: a row whose versions are stored
/// under two schema versions, as a row is when a change of schema that leaves its table
/// alone falls between two of its writes (how many rows of a real store that is has not
/// been counted). The cleanup of a batch rejected under one of them reads that one's rows
/// only, so the version every tip descends from is not among them — and a merge that takes
/// "no common ancestor" at its word lets the newest tip win every column, including one
/// only the older tip wrote.
#[test]
fn a_local_rejection_on_a_row_in_two_schema_versions_leaves_the_merge_to_the_history() {
    // The same `users` beside a table the other version does not have: another schema
    // version, and the same bytes for a `users` row.
    let schema_before = SchemaBuilder::new()
        .table(TableSchema::builder("users").column("value", ColumnType::Text))
        .table(TableSchema::builder("notes").column("body", ColumnType::Text))
        .build();
    let hash_before = SchemaHash::compute(&schema_before);
    assert_ne!(hash_before, users_schema_hash());

    let _mode = force_history_fastpath(true);
    let mut io = store();
    persist_test_schema(&mut io, &schema_before);
    let row_id = ObjectId::new();
    create_test_row_with_id(
        &mut io,
        row_id,
        Some(HashMap::from([
            (MetadataKey::Table.to_string(), "users".to_string()),
            (
                MetadataKey::OriginSchemaHash.to_string(),
                hash_before.to_string(),
            ),
        ])),
    );
    let created = version(row_id, Vec::new(), CREATED_AT, "kept");
    apply(&mut io, &created);

    // From here on the row is written under the other schema version, as its locator says.
    crate::test_support::put_test_row_metadata(&mut io, row_id, row_metadata("users"));
    let over_creation = |at: u64, label: &str| version(row_id, vec![created.batch_id()], at, label);
    let left = over_creation(CREATED_AT + 1, "kept");
    let right = over_creation(CREATED_AT + 2, "renamed");
    let local = over_creation(CREATED_AT + 3, "local");
    for row in [&left, &right, &local] {
        apply(&mut io, row);
    }
    // The store holds this one row: what a schema version's history table lists is its.
    let stored_under = |io: &SqliteStorage, hash: SchemaHash| {
        let versions = crate::storage::RowRawTableId::new(
            crate::storage::RowRawTableKind::History,
            "users",
            hash,
        );
        io.raw_table_scan_prefix_keys(versions.raw_table_name(), "")
            .expect("history of one schema version")
            .len()
    };
    assert_eq!(
        (
            stored_under(&io, hash_before),
            stored_under(&io, users_schema_hash())
        ),
        (1, 3),
        "control: the row's history is in two schema versions"
    );

    assert!(
        io.patch_exact_row_batch_for_schema_hash(
            "users",
            users_schema_hash(),
            "main",
            row_id,
            local.batch_id(),
            Some(RowState::Rejected),
            None,
        )
        .expect("the rejected batch is patched")
    );
    let entry = |io: &SqliteStorage| {
        io.load_visible_region_entry("users", "main", row_id)
            .expect("visible read")
            .expect("the row is visible")
    };
    // What the entry says of its tips' ancestor is the other test's; this one is about
    // what a wrong answer does to the row.
    assert_eq!(entry(&io).branch_frontier.len(), 2);

    // One tip is written over. The other is the only one that renamed the row, and the
    // new one says what the creation said: merged over the creation, the rename stands.
    let over_left = version(row_id, vec![left.batch_id()], CREATED_AT + 4, "kept");
    apply(&mut io, &over_left);
    let after_write = entry(&io);
    let descriptor = &users_test_schema()[&"users".into()].columns;
    let history = io
        .scan_history_region(
            "users",
            "main",
            crate::row_histories::HistoryScan::Row { row_id },
        )
        .expect("the branch's history");
    assert_eq!(
        history.len(),
        5,
        "control: every version of the row is read"
    );
    let rebuilt = VisibleRowEntry::rebuild_with_descriptor(descriptor, &history)
        .expect("the history builds an entry")
        .expect("the row is visible");
    assert_eq!(
        after_write.current_row.data.to_vec(),
        encode_row(descriptor, &[Value::Text("renamed".to_string())])
            .expect("the merged row encodes"),
        "the rename only the other tip made was merged away"
    );
    // What a stored entry keeps of the row it shows: a store reads it back without its
    // parents.
    let shown = |entry: &VisibleRowEntry| {
        (
            entry.branch_frontier.clone(),
            entry.current_row.batch_id(),
            entry.current_row.data.to_vec(),
            entry.merge_artifacts.clone(),
        )
    };
    assert_eq!(
        shown(&after_write),
        shown(&rebuilt),
        "the write after the cleanup left another entry than the branch's history builds"
    );

    // A write over both tips needs no ancestor: it reads the two versions it names.
    let over_both = version(
        row_id,
        vec![right.batch_id(), over_left.batch_id()],
        CREATED_AT + 5,
        "both",
    );
    let before = HISTORY_ENTRIES_ON_THREAD.get();
    apply(&mut io, &over_both);
    assert_eq!(
        HISTORY_ENTRIES_ON_THREAD.get() - before,
        2,
        "a write over every tip read more of the row than the tips"
    );
    assert_eq!(
        entry(&io).branch_frontier.as_slice(),
        [over_both.batch_id()]
    );
}

/// xorshift64*: the crate carries no property-testing dependency.
struct Prng(u64);

impl Prng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next() % bound as u64) as usize
    }
}

/// A schema where the row a write is judged against depends on more than its newest
/// parent: two columns the newest writer of each wins, and a counter that adds up what
/// every tip did since the version they share.
fn merged_schema() -> crate::query_manager::types::Schema {
    use crate::query_manager::types::{
        ColumnDescriptor, ColumnMergeStrategy, RowDescriptor, TableName,
    };
    [(
        TableName::new("merged"),
        TableSchema::new(RowDescriptor::new(vec![
            ColumnDescriptor::new("value", ColumnType::Text),
            ColumnDescriptor::new("other", ColumnType::Text),
            ColumnDescriptor::new("count", ColumnType::Integer)
                .merge_strategy(ColumnMergeStrategy::Counter),
        ])),
    )]
    .into_iter()
    .collect()
}

fn merged_version(
    row_id: ObjectId,
    parents: Vec<BatchId>,
    at: u64,
    values: &[Value],
    delete: Option<crate::metadata::DeleteKind>,
) -> StoredRowBatch {
    let creation = RowProvenance::for_insert("alice".to_string(), CREATED_AT);
    let provenance = if parents.is_empty() && at == CREATED_AT {
        creation
    } else {
        RowProvenance::for_update(&creation, "alice".to_string(), at)
    };
    StoredRowBatch::new(
        row_id,
        "main",
        parents,
        encode_row(&merged_schema()[&"merged".into()].columns, values)
            .expect("test row should encode"),
        provenance.clone(),
        crate::metadata::row_provenance_metadata(&provenance, delete)
            .into_iter()
            .collect(),
        RowState::VisibleDirect,
        None,
    )
}

/// Random histories — forks, merges, writes naming old versions, rejected versions,
/// deletes, copies delivered without their parents, clocks that disagree — over a row
/// whose columns merge, and random writes over them. Whatever the read from the tips
/// answers, the walk answers the same; and each of its shapes answers often.
#[test]
fn the_read_from_the_tips_never_answers_differently_from_the_walk() {
    use super::super::inbox::PRE_WRITE_ROW_FROM_TIPS_ON_THREAD;
    use crate::metadata::DeleteKind;

    // The histories below are applied through the fast paths: an entry they wrote that
    // differs from the history is what the read from the tips must not trust, and another
    // test switching them off for the whole process would take those entries away.
    let _mode = force_history_fastpath(true);
    // Counted over the direct calls only: the read through `pre_batch_visible_row_read`
    // further down asks the same question again.
    let mut answered = [0u64; 3];
    let mut named_itself = 0u32;
    let mut asked = 0u32;
    let mut merged_answers = 0u32;
    for seed in 1..=40u64 {
        let mut prng = Prng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let mut io = SqliteStorage::open(":memory:").expect("in-memory sqlite storage should open");
        let schema_hash = persist_test_schema(&mut io, &merged_schema());
        let row_id = ObjectId::new();
        create_test_row_with_id(
            &mut io,
            row_id,
            Some(HashMap::from([
                (MetadataKey::Table.to_string(), "merged".to_string()),
                (
                    MetadataKey::OriginSchemaHash.to_string(),
                    schema_hash.to_string(),
                ),
            ])),
        );
        let values = |prng: &mut Prng, from: Option<&Vec<Value>>| {
            let mut next = from.cloned().unwrap_or_else(|| {
                vec![
                    Value::Text("a".into()),
                    Value::Text("a".into()),
                    Value::Integer(0),
                ]
            });
            // One column moves, sometimes two: a version that changes every column
            // would win every column, and the merge would be the newest tip again.
            for _ in 0..1 + prng.below(2) {
                match prng.below(3) {
                    0 => next[0] = Value::Text(format!("v{}", prng.below(5))),
                    1 => next[1] = Value::Text(format!("o{}", prng.below(5))),
                    _ => next[2] = Value::Integer(prng.below(7) as i32),
                }
            }
            next
        };
        let first_values = values(&mut prng, None);
        let first = merged_version(row_id, Vec::new(), CREATED_AT, &first_values, None);
        apply(&mut io, &first);
        let mut all = vec![(first, first_values)];
        let tips = |io: &SqliteStorage| {
            io.load_visible_region_entry("merged", "main", row_id)
                .expect("visible read")
                .map(|entry| entry.branch_frontier)
                .unwrap_or_default()
        };

        for step in 1..60usize {
            // What the new version is written over: every tip, a tip and older versions,
            // older versions only, one tip — or nothing, as a copy delivered without its
            // parents is stored.
            let frontier = tips(&io);
            let mut parents: Vec<BatchId> = Vec::new();
            match prng.below(12) {
                0 | 1 if !frontier.is_empty() => parents = frontier.clone(),
                2..=5 if !frontier.is_empty() => {
                    parents.push(frontier[prng.below(frontier.len())]);
                    for _ in 0..prng.below(3) {
                        parents.push(all[prng.below(all.len())].0.batch_id());
                    }
                }
                6 | 7 => {
                    for _ in 0..1 + prng.below(2) {
                        parents.push(all[prng.below(all.len())].0.batch_id());
                    }
                }
                8 => {}
                _ if !frontier.is_empty() => {
                    parents.push(frontier[prng.below(frontier.len())]);
                }
                _ => parents.push(all[prng.below(all.len())].0.batch_id()),
            }
            parents.sort();
            parents.dedup();
            let from = parents
                .first()
                .and_then(|parent| all.iter().find(|(row, _)| row.batch_id() == *parent))
                .map(|(_, values)| values.clone());
            let next_values = values(&mut prng, from.as_ref());
            // One clock in four is behind.
            let at = CREATED_AT + 10 * step as u64 - if prng.below(4) == 0 { 25 } else { 0 };
            let delete = (!parents.is_empty() && prng.below(15) == 0).then_some(DeleteKind::Soft);
            let next = merged_version(
                row_id,
                parents,
                at.max(CREATED_AT + 1),
                &next_values,
                delete,
            );
            apply(&mut io, &next);
            all.push((next, next_values));
            if prng.below(8) == 0 {
                let rejected = all[prng.below(all.len())].0.batch_id();
                patch_row_batch_state(
                    &mut io,
                    row_id,
                    &BranchName::new("main"),
                    rejected,
                    Some(RowState::Rejected),
                    None,
                )
                .expect("reject a version");
            }

            // A write some device might send now, named over two to four versions.
            for _ in 0..4 {
                let frontier = tips(&io);
                let mut named: Vec<BatchId> = Vec::new();
                if prng.below(2) == 0 {
                    named.extend(frontier.iter().copied());
                }
                for _ in 0..1 + prng.below(3) {
                    let pick = &all[prng.below(all.len())].0;
                    named.push(pick.batch_id());
                    if prng.below(2) == 0 {
                        named.extend(pick.parents.iter().copied());
                    }
                }
                if prng.below(12) == 0 {
                    named.push(BatchId::new());
                }
                named.sort();
                named.dedup();
                if named.len() < 2 {
                    continue;
                }
                let incoming = values(&mut prng, None);
                let delete = (prng.below(10) == 0).then_some(DeleteKind::Soft);
                let mut write = merged_version(row_id, named, at + 1, &incoming, delete);
                // One write in ten takes the id of one of the versions it names — a
                // version sent again, naming itself. What it names may be held, rejected
                // or not held at all: a held one is no parent of its own, and the read
                // from the tips leaves such a write to the walk — unless another version
                // it names is not held, which answers it first, as it answers the walk.
                if prng.below(10) == 0 {
                    let own = write.parents[prng.below(write.parents.len())];
                    write.batch_id = own;
                    named_itself += 1;
                }
                asked += 1;
                let answers_before = PRE_WRITE_ROW_FROM_TIPS_ON_THREAD.get();
                let from_tips = SyncManager::pre_batch_visible_row_from_tips(&io, "merged", &write);
                let answers = PRE_WRITE_ROW_FROM_TIPS_ON_THREAD.get();
                for shape in 0..answered.len() {
                    answered[shape] += answers[shape] - answers_before[shape];
                }
                let by_walk = SyncManager::new()
                    .pre_batch_visible_row_read(&io, "merged", &write, false)
                    .map(|row| row.data.to_vec());
                let context = format!(
                    "seed {seed} step {step}: a write over {} versions of a row with {} tips",
                    write.parents.len(),
                    frontier.len()
                );
                if let Some(answer) = from_tips {
                    let answer = answer.map(|row| row.data.to_vec());
                    assert_eq!(
                        answer, by_walk,
                        "{context} is judged against another row than the one its parents \
                         resolve to"
                    );
                    // How often the answer is a row no single version holds: the cases
                    // where a wrong set of tips or a wrong common version would show.
                    if answer
                        .is_some_and(|data| all.iter().all(|(row, _)| row.data[..] != data[..]))
                    {
                        merged_answers += 1;
                    }
                }
                assert_eq!(
                    SyncManager::new()
                        .pre_batch_visible_row_read(&io, "merged", &write, true)
                        .map(|row| row.data.to_vec()),
                    by_walk,
                    "{context}"
                );
            }
        }
    }
    let [one_above, every_tip, not_held] = answered;
    println!(
        "asked {asked} ({named_itself} naming themselves): one parent above the rest \
         {one_above}, every tip named {every_tip}, a parent not held {not_held}, answered \
         with a merged row {merged_answers}"
    );
    // The run repeats itself: the same counts three times over (982, 1 564, 713, 1 287).
    // The floors sit some 5 % under them: a change that takes a kind of answer away
    // shows here, one that takes a few answers of a kind does not.
    assert!(named_itself > 0, "no write of the run named itself");
    assert!(
        one_above > 930 && every_tip > 1_480 && not_held > 670 && merged_answers > 1_220,
        "of {asked} writes the read from the tips answered {one_above} by one parent above \
         the rest, {every_tip} by naming every tip and {not_held} by a parent that is not \
         held, {merged_answers} of them with a merged row: the comparison proves little"
    );
}
