//! Declared indexes (`composite_index`, `trigram_index`): which ones a store maintains,
//! the entries a row owns in them, and the work that makes a newly declared one whole.
//!
//! The store owns the set it maintains: a record (`MaintainedIndexes`) in the raw table
//! `RECORD_TABLE`. Every writer files the entries of every index in it once the index
//! has cleared its prefix (`MaintainedIndexes::written`); the app's declarations (the
//! permissions head) only propose the next set.
//!
//! Every path that files or retires a row's entries goes through `row_entries`: the
//! write path (`indices.rs`, via `push_mutations`), the repair of a row split across
//! schema generations (`storage::column_index_entries_for_row_bytes`) and the fill
//! below. An entry one of them forgets outlives its row, and a window or a search
//! returns the row for values it no longer holds.
//!
//! Rows written before an index was added have no entries. Each added index — re-added
//! ones included — first clears whatever is left under its prefix, then files the
//! entries of every row live when the clear ended, walking the `_id` index a page at a
//! time (`advance`), and only then is it `Complete`. A runtime takes one step per
//! batched tick and none in the ticks of its writes (`pace_declared_index_steps`), so a
//! fill holds it a bounded page at a time; a bare `QueryManager` steps in every
//! `process`. Neither walk chases writes: nothing is filed under the prefix while it
//! clears, and rows written after it are filed by their writes. A scan uses the
//! index only in the incarnation it saw complete (`IndexScanNode`); until then it reads
//! the first column's own index, exact and only slower. So correctness never depends on
//! the fill having run — only speed does.
//!
//! An engine that does not maintain the entries must not write to a store whose indexes
//! are complete: its rows would be missing from them. The store's manifest says format 4
//! while anything is maintained (`STORE_FORMAT_V4_DECLARED_INDEXES`), which such engines
//! refuse to open.
//!
//! For a rollback, an engine released from them (`QueryManager::release_declared_indexes`)
//! gives every declared index up at open, writes format 3 back, and ignores the app's
//! declarations from then on. A binding releases its runtime when the app sets
//! `releaseDeclaredIndexes`; a server process also when started with
//! `JAZZ_RELEASE_DECLARED_INDEXES=1`. Short of that, a permissions head published
//! without declarations gives them up on every store that applies it. A release belongs
//! to the runtime, not the store: a runtime without it that opens the store declares
//! them again. So a rollback ships the release to every runtime that can open a store
//! (every tab that can lead, in a browser), and ships the older engine only once the
//! stores read format 3. The entries of the given-up indexes are deleted over later
//! ticks; an older engine started before that leaves them, and never reads them. A
//! record the released engine cannot decode is given up whole, and nothing records its
//! prefixes after that: their entries stay until an index of the same name is declared
//! on the same table again, whose clear deletes them. A record it fails to read is left
//! as it is, and so is the store's format: the runtime starts without the indexes and
//! opens the store again at each later pass, which gives them up once the record reads.
//!
//! Not closed: a runtime whose open failed, released or not, writes rows without their
//! entries while the record may still call an index complete. A later start that reads
//! the record trusts the index, and its windows and searches miss those rows. It takes a
//! read of the record that fails for a runtime's whole life and succeeds at the next.
//!
//! The fill counts on one runtime per store. Rows written after the clear ends are
//! filed by their writes, and a runtime reads the phase it writes under at the start
//! of each pass: a second runtime over the store that has not passed since another one
//! ended a clear files nothing for the rows it writes, and the index completes without
//! them.
//!
//! The record's encoding belongs to format 4, and a tag it does not define fails the
//! decode. An engine that cannot decode a store's record trusts none of its indexes,
//! and its repairs of split rows retire column entries only (`repair_declarations`); so
//! a new encoding of the record, a new phase included, comes with a new store format,
//! which engines that cannot read it refuse to open.

use std::collections::BTreeSet;

use crate::object::ObjectId;
use crate::query_manager::encoding::decode_column;
use crate::query_manager::index_declarations::{
    DeclarationError, IndexDeclarations, IndexPhase, MaintainedIndexes,
};
use crate::query_manager::types::{RowDescriptor, Value};
use crate::storage::{
    IndexMutation, RawTableMutation, STORE_FORMAT_V3, STORE_FORMAT_V4_DECLARED_INDEXES, Storage,
    StorageError,
};

/// The raw table holding the record of the maintained indexes, under `RECORD_KEY`.
pub(crate) const RECORD_TABLE: &str = "declared_indexes";
const RECORD_KEY: &str = "record";

/// Where the env-declared prototype kept its completeness marks. They meant "filled" to
/// an engine that trusted them; nothing reads them now, and `open` deletes them.
const LEGACY_MARK_TABLE: &str = "declared_index_complete";

/// Index entries one clearing step deletes.
const CLEAR_BATCH: usize = 4096;
/// Rows one filling step reads at most. A step holds the runtime, and everything
/// waiting on it waits out the whole step: a page of 1024 rows made writes and reads
/// wait hundreds of milliseconds behind a fill on a 100k-row table.
pub(crate) const FILL_PAGE: usize = 128;
/// Entries one filling step files at most, finishing the row that crosses it. A row's
/// trigrams are as many as its text's, so a page of rows alone does not bound a step.
pub(crate) const FILL_ENTRIES: usize = 2048;

/// The env switch that releases a server process's engine from its declared indexes
/// (see the module doc).
pub(crate) const RELEASE_SWITCH: &str = "JAZZ_RELEASE_DECLARED_INDEXES";

/// Whether this process was started to give the declared indexes up: where an engine
/// has no environment to read (a browser, a phone), this is always false and the
/// binding's option is the only switch.
pub(crate) fn release_requested() -> bool {
    std::env::var(RELEASE_SWITCH).is_ok_and(|value| value == "1")
}

fn record_error(error: DeclarationError) -> StorageError {
    StorageError::IoError(format!("declared index record: {error}"))
}

/// The record of the indexes `storage` maintains; empty when it has none.
pub(crate) fn load_record<H: Storage + ?Sized>(
    storage: &H,
) -> Result<MaintainedIndexes, StorageError> {
    match storage.raw_table_get(RECORD_TABLE, RECORD_KEY)? {
        Some(bytes) => MaintainedIndexes::decode(&bytes).map_err(record_error),
        None => Ok(MaintainedIndexes::default()),
    }
}

/// The declared indexes whose entries a repair of a split row retires. A record that
/// does not decode gives none: nothing trusts its entries (a scan reads the first
/// column's own index while the record is unreadable) and only a release rewrites it,
/// empty, after which an index declared again clears its prefix before it is trusted.
/// So the repair retires the column entries alone rather than failing the write it
/// runs in.
/// A failed read is not that: the record may read again and be trusted, so it fails.
pub(crate) fn repair_declarations<H: Storage + ?Sized>(
    storage: &H,
) -> Result<IndexDeclarations, StorageError> {
    static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    let Some(bytes) = storage.raw_table_get(RECORD_TABLE, RECORD_KEY)? else {
        return Ok(IndexDeclarations::empty());
    };
    match MaintainedIndexes::decode(&bytes) {
        Ok(record) => Ok(record.declarations),
        Err(error) => {
            // The startup sweep repairs every split row: once per process is enough.
            if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                tracing::warn!(
                    %error,
                    "split-row repairs retire column entries only: the declared index record does not decode"
                );
            }
            Ok(IndexDeclarations::empty())
        }
    }
}

fn store_record<H: Storage + ?Sized>(
    storage: &mut H,
    record: &MaintainedIndexes,
) -> Result<(), StorageError> {
    storage.raw_table_put(RECORD_TABLE, RECORD_KEY, &record.encode())
}

fn format_for(record: &MaintainedIndexes) -> i32 {
    if record.declarations.is_empty() {
        STORE_FORMAT_V3
    } else {
        STORE_FORMAT_V4_DECLARED_INDEXES
    }
}

fn ensure_format<H: Storage + ?Sized>(storage: &mut H, format: i32) -> Result<(), StorageError> {
    match storage.store_format_version()? {
        Some(current) if current != format => storage.set_store_format_version(format),
        _ => Ok(()),
    }
}

/// Make `next` the record. The manifest says format 4 before a record that maintains
/// anything is written, and says 3 only after one that maintains nothing is: a crash
/// between the two writes leaves a store an older engine still refuses, never one it
/// opens while entries it would not maintain are trusted. `open` finishes the flip.
pub(crate) fn replace<H: Storage + ?Sized>(
    storage: &mut H,
    next: &MaintainedIndexes,
) -> Result<(), StorageError> {
    let format = format_for(next);
    if format == STORE_FORMAT_V4_DECLARED_INDEXES {
        ensure_format(storage, format)?;
        store_record(storage, next)
    } else {
        store_record(storage, next)?;
        ensure_format(storage, format)
    }
}

/// The record at open, after the repairs a start owes the store: the prototype's marks
/// deleted, the indexes given up when the engine is released from them (`release`), and
/// the manifest's format brought in line with the record, which a crash inside `replace`
/// can leave behind. Returns the record, and whether anything was written.
pub(crate) fn open<H: Storage + ?Sized>(
    storage: &mut H,
    release: bool,
) -> Result<(MaintainedIndexes, bool), StorageError> {
    let format_before = storage.store_format_version()?;
    let marks = storage.raw_table_scan_prefix_keys(LEGACY_MARK_TABLE, "")?;
    let mut wrote = !marks.is_empty();
    if !marks.is_empty() {
        let deletes: Vec<RawTableMutation<'_>> = marks
            .iter()
            .map(|key| RawTableMutation::Delete {
                table: LEGACY_MARK_TABLE,
                key,
            })
            .collect();
        storage.apply_raw_table_mutations(&deletes)?;
    }
    // A read that fails fails the open, released or not, and the runtime opens again
    // the next pass: the record may read by then, and still name indexes whose entries
    // writers keep.
    let mut record = match storage.raw_table_get(RECORD_TABLE, RECORD_KEY)? {
        None => MaintainedIndexes::default(),
        Some(bytes) => match MaintainedIndexes::decode(&bytes) {
            Ok(record) => record,
            // Kept, a record this engine cannot decode would keep the store at format 4,
            // which the engine rolled back to refuses. Nothing is trusted under the
            // prefixes it named, and an index declared again clears its prefix before it
            // is.
            Err(error) if release => {
                let next = MaintainedIndexes::default();
                replace(storage, &next)?;
                tracing::warn!(
                    %error,
                    "gave an undecodable declared index record up: this engine is released from them"
                );
                wrote = true;
                next
            }
            Err(error) => return Err(record_error(error)),
        },
    };
    if release && let Some(next) = record.propose(&IndexDeclarations::empty()) {
        replace(storage, &next)?;
        tracing::warn!(
            released = ?next.retired.keys().collect::<Vec<_>>(),
            "gave the declared indexes up: this engine is released from them"
        );
        record = next;
        wrote = true;
    }
    ensure_format(storage, format_for(&record))?;
    wrote |= storage.store_format_version()? != format_before;
    Ok((record, wrote))
}

/// Apply `declarations` as the next set, if it differs. Returns whether it did.
pub(crate) fn propose<H: Storage + ?Sized>(
    storage: &mut H,
    declarations: &IndexDeclarations,
) -> Result<bool, StorageError> {
    match load_record(storage)?.propose(declarations) {
        Some(next) => replace(storage, &next).map(|()| true),
        None => Ok(false),
    }
}

/// One step of the work the record still owes, maintained indexes first: they are what
/// scans wait for. Returns the record after it.
///
/// A step re-reads the record before recording its progress and records it only if the
/// index is still in the incarnation and phase the step worked for; otherwise the newer
/// plan stands and the step's writes are the kind that plan redoes anyway.
pub(crate) fn advance<H: Storage + ?Sized>(
    storage: &mut H,
) -> Result<MaintainedIndexes, StorageError> {
    let record = load_record(storage)?;
    if let Some((key, state)) = record
        .states
        .iter()
        .find(|(_, state)| state.phase != IndexPhase::Complete)
    {
        let (table, index) = (key.0.as_str(), key.1.as_str());
        let next_phase = match &state.phase {
            IndexPhase::Clearing { after } => {
                match clear_batch(storage, table, index, after.as_deref())? {
                    Some(last) => IndexPhase::Clearing { after: Some(last) },
                    // The rows the fill owes are the ones live now; writes file the rest.
                    None => {
                        match storage.raw_table_family_last_key(&family_prefix(table, "_id"))? {
                            Some(until) => IndexPhase::Filling { after: None, until },
                            None => IndexPhase::Complete,
                        }
                    }
                }
            }
            IndexPhase::Filling { after, until } => {
                match fill_page(
                    storage,
                    &record.declarations,
                    table,
                    index,
                    after.as_deref(),
                    until,
                )? {
                    Some(last) => IndexPhase::Filling {
                        after: Some(last),
                        until: until.clone(),
                    },
                    None => IndexPhase::Complete,
                }
            }
            IndexPhase::Complete => unreachable!("found by its phase"),
        };
        let mut current = load_record(storage)?;
        match current.states.get_mut(key) {
            Some(now) if *now == *state => {
                if next_phase == IndexPhase::Complete {
                    tracing::info!(
                        table,
                        index,
                        incarnation = now.incarnation,
                        "declared index complete"
                    );
                }
                now.phase = next_phase;
            }
            _ => return Ok(current),
        }
        store_record(storage, &current)?;
        return Ok(current);
    }
    if let Some((key, after)) = record.retired.iter().next() {
        let next = clear_batch(storage, &key.0, &key.1, after.as_deref())?;
        let mut current = load_record(storage)?;
        if current.retired.get(key) != Some(after) {
            return Ok(current);
        }
        match next {
            Some(last) => {
                current.retired.insert(key.clone(), Some(last));
            }
            None => {
                current.retired.remove(key);
            }
        }
        store_record(storage, &current)?;
        return Ok(current);
    }
    Ok(record)
}

/// The per-branch raw tables of one index share the name prefix
/// `idx:{table}:{column}:`; a key of the family is `{branch}:{segment}:{uuid}`, and
/// neither the segment nor the uuid holds a `:`. Returns the branch, the entry key
/// within its raw table, and the row.
fn split_family_key(rest: &str) -> Option<(&str, &str, ObjectId)> {
    let (branch, entry_key) = rest
        .rsplitn(3, ':')
        .nth(2)
        .map(|branch| (branch, &rest[branch.len() + 1..]))?;
    let row_id = crate::query_manager::composite_index::entry_row_id(entry_key)?;
    Some((branch, entry_key, row_id))
}

fn family_prefix(table: &str, column: &str) -> String {
    format!("idx:{table}:{column}:")
}

/// Delete the next batch of entries under `index`'s prefix, on every branch, after
/// `after`. Returns the last one deleted, or `None` when none was left.
fn clear_batch<H: Storage + ?Sized>(
    storage: &mut H,
    table: &str,
    index: &str,
    after: Option<&str>,
) -> Result<Option<String>, StorageError> {
    let prefix = family_prefix(table, index);
    let keys = storage.raw_table_family_keys(&prefix, after, CLEAR_BATCH)?;
    let Some(last) = keys.last().cloned() else {
        return Ok(None);
    };
    let targets: Vec<(String, &str)> = keys
        .iter()
        .filter_map(|rest| {
            let (branch, entry_key, _) = split_family_key(rest)?;
            Some((format!("{prefix}{branch}"), entry_key))
        })
        .collect();
    let deletes: Vec<RawTableMutation<'_>> = targets
        .iter()
        .map(|(raw_table, key)| RawTableMutation::Delete {
            table: raw_table,
            key,
        })
        .collect();
    storage.apply_raw_table_mutations(&deletes)?;
    Ok(Some(last))
}

/// File `index`'s entries for the next page of live rows of `table`, every branch,
/// walking the `_id` index after `after` up to `until`: at most `FILL_PAGE` rows, and
/// no row past the one that brings the page to `FILL_ENTRIES` entries. Returns the last
/// `_id` entry handled, or `None` once `until` is reached.
///
/// Each row is its visible winner, read the way a point read serves it; soft-deleted
/// rows are not in `_id`, and a winner that is deleted files nothing.
fn fill_page<H: Storage + ?Sized>(
    storage: &mut H,
    declarations: &IndexDeclarations,
    table: &str,
    index: &str,
    after: Option<&str>,
    until: &str,
) -> Result<Option<String>, StorageError> {
    let keys = storage.raw_table_family_keys(&family_prefix(table, "_id"), after, FILL_PAGE)?;
    // The entries of the one index this step fills, not of every index on the table.
    let mut others = declarations.index_names();
    others.remove(&(table.to_string(), index.to_string()));
    let declarations = declarations.without(&others);
    let mut rows = Vec::new();
    let mut filed = 0;
    let mut next = None;
    for (position, rest) in keys.iter().enumerate() {
        if rest.as_str() > until {
            break;
        }
        if let Some((branch, _, row_id)) = split_family_key(rest)
            && let Some(row) = storage.load_visible_region_row(table, branch, row_id)?
            && !row.is_deleted
        {
            let descriptor = crate::storage::row_user_descriptor(storage, table, &row)?;
            let entries = row_entries(&declarations, table, &descriptor, row.data.as_ref());
            filed += entries.len();
            rows.push((branch, row_id, entries));
        }
        if rest.as_str() == until {
            break;
        }
        let more = position + 1 < keys.len() || keys.len() == FILL_PAGE;
        if more && (filed >= FILL_ENTRIES || position + 1 == keys.len()) {
            next = Some(rest.clone());
            break;
        }
    }
    let mutations: Vec<_> = rows
        .iter()
        .flat_map(|(branch, row_id, entries)| {
            entries.iter().map(|(_, bytes)| IndexMutation::Insert {
                table,
                column: index,
                branch,
                value: Value::Bytea(bytes.clone()),
                row_id: *row_id,
            })
        })
        .collect();
    storage.apply_index_mutations(&mutations)?;
    Ok(next)
}

/// The declared-index entries of a row of `table` whose bytes are `data`: the index
/// name and the entry value's bytes (`Value::Bytea`). A column the layout lacks, a null
/// or a value without a fixed-width encoding files nothing, as a null files nothing in a
/// single-column index.
pub(crate) fn row_entries<'a>(
    declarations: &'a IndexDeclarations,
    table: &str,
    descriptor: &RowDescriptor,
    data: &[u8],
) -> BTreeSet<(&'a str, Vec<u8>)> {
    let column = |name: &str| -> Option<Value> {
        decode_column(descriptor, data, descriptor.column_index(name)?).ok()
    };
    let mut entries = BTreeSet::new();
    for index in declarations.composites(table) {
        if let (Some(first), Some(second)) = (column(&index.first), column(&index.second))
            && let Some(Value::Bytea(bytes)) =
                crate::query_manager::composite_index::composite_value(&first, &second)
        {
            entries.insert((index.name.as_str(), bytes));
        }
    }
    for index in declarations.trigrams(table) {
        use crate::query_manager::trigram_index::{entry_value, fold, trigrams};
        let (Some(scope), Some(Value::Text(text))) = (column(&index.scope), column(&index.text))
        else {
            continue;
        };
        for trigram in trigrams(&fold(&text)) {
            if let Some(Value::Bytea(bytes)) = entry_value(&scope, &trigram) {
                entries.insert((index.name.as_str(), bytes));
            }
        }
    }
    entries
}

/// The declared-index mutations of a row of `table` going from `old_data` to
/// `new_data` (either side absent): only the entries that change are written, so an
/// update that leaves the indexed columns alone writes nothing.
#[allow(clippy::too_many_arguments)]
pub(crate) fn push_mutations<'a>(
    mutations: &mut Vec<IndexMutation<'a>>,
    declarations: &'a IndexDeclarations,
    table: &'a str,
    branch: &'a str,
    row_id: ObjectId,
    descriptor: &RowDescriptor,
    old_data: Option<&[u8]>,
    new_data: Option<&[u8]>,
) {
    if declarations.composites(table).is_empty() && declarations.trigrams(table).is_empty() {
        return;
    }
    let entries = |data: Option<&[u8]>| {
        data.map(|data| row_entries(declarations, table, descriptor, data))
            .unwrap_or_default()
    };
    let (old_entries, new_entries) = (entries(old_data), entries(new_data));
    for (column, bytes) in old_entries.difference(&new_entries) {
        mutations.push(IndexMutation::Remove {
            table,
            column,
            branch,
            value: Value::Bytea(bytes.clone()),
            row_id,
        });
    }
    for (column, bytes) in new_entries.difference(&old_entries) {
        mutations.push(IndexMutation::Insert {
            table,
            column,
            branch,
            value: Value::Bytea(bytes.clone()),
            row_id,
        });
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::row_histories::{StoredRowBatch, VisibleRowEntry};
    use crate::storage::{
        HistoryRowBytes, MemoryStorage, OwnedHistoryRowBytes, OwnedVisibleRowBytes, RawTableKeys,
        RawTableRows, STORE_FORMAT_V4_DECLARED_INDEXES, SqliteStorage, VisibleRowBytes,
    };

    fn declarations() -> IndexDeclarations {
        IndexDeclarations::empty()
            .with_composite("wmsgs", "chat", "at")
            .expect("composite")
    }

    fn sqlite_file() -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("store.sqlite");
        (dir, path)
    }

    fn format(storage: &SqliteStorage) -> Option<i32> {
        storage.store_format_version().expect("manifest readable")
    }

    /// A store says format 4 exactly while it maintains a declared index, across a
    /// reopen, and goes back to 3 when it maintains none.
    #[test]
    fn the_store_format_follows_what_the_store_maintains() {
        let (_dir, path) = sqlite_file();
        let mut storage = SqliteStorage::open(&path).expect("open");
        assert_eq!(format(&storage), Some(STORE_FORMAT_V3), "a new store");

        assert!(propose(&mut storage, &declarations()).expect("declare"));
        assert_eq!(format(&storage), Some(STORE_FORMAT_V4_DECLARED_INDEXES));
        assert!(
            !propose(&mut storage, &declarations()).expect("declare again"),
            "the same declarations change nothing"
        );
        storage.flush().expect("flush");
        drop(storage);

        let mut storage = SqliteStorage::open(&path).expect("a format-4 store opens");
        assert_eq!(format(&storage), Some(STORE_FORMAT_V4_DECLARED_INDEXES));
        let (record, wrote) = open(&mut storage, false).expect("open the indexes");
        assert!(!wrote, "a consistent store needs no repair");
        assert_eq!(record.declarations, declarations());

        assert!(propose(&mut storage, &IndexDeclarations::empty()).expect("give up"));
        assert_eq!(format(&storage), Some(STORE_FORMAT_V3));
        let record = load_record(&storage).expect("record");
        assert!(record.declarations.is_empty());
        assert!(
            !record.retired.is_empty(),
            "the entries are still to be cleared"
        );
    }

    /// `replace` writes the record and the manifest one after the other; a crash between
    /// them leaves either order behind, and the next open finishes the flip.
    #[test]
    fn open_repairs_a_format_a_crash_left_behind() {
        let mut storage = SqliteStorage::open(":memory:").expect("open");
        let declared = MaintainedIndexes::default()
            .propose(&declarations())
            .expect("a change");
        store_record(&mut storage, &declared).expect("record without its manifest");
        assert_eq!(format(&storage), Some(STORE_FORMAT_V3));
        let (_, wrote) = open(&mut storage, false).expect("open");
        assert!(wrote);
        assert_eq!(format(&storage), Some(STORE_FORMAT_V4_DECLARED_INDEXES));

        let released = declared
            .propose(&IndexDeclarations::empty())
            .expect("a change");
        store_record(&mut storage, &released).expect("record without its manifest");
        assert_eq!(format(&storage), Some(STORE_FORMAT_V4_DECLARED_INDEXES));
        let (_, wrote) = open(&mut storage, false).expect("open");
        assert!(wrote);
        assert_eq!(format(&storage), Some(STORE_FORMAT_V3));
    }

    /// The release switch gives the indexes up at open and writes format 3 back, so an
    /// engine that does not maintain them opens the store; the prototype's marks go.
    #[test]
    fn the_release_switch_gives_the_indexes_up_and_old_marks_go() {
        let mut storage = SqliteStorage::open(":memory:").expect("open");
        propose(&mut storage, &declarations()).expect("declare");
        storage
            .raw_table_put(LEGACY_MARK_TABLE, "wmsgs:chat+at", b"1")
            .expect("prototype mark");

        let (record, wrote) = open(&mut storage, true).expect("open");
        assert!(wrote);
        assert!(record.declarations.is_empty());
        assert_eq!(format(&storage), Some(STORE_FORMAT_V3));
        assert!(
            storage
                .raw_table_scan_prefix_keys(LEGACY_MARK_TABLE, "")
                .expect("marks readable")
                .is_empty(),
            "the prototype's marks are deleted"
        );
    }

    fn put_entries(storage: &mut MemoryStorage, column: &str, branch: &str, count: usize) {
        for _ in 0..count {
            let row = ObjectId::new();
            storage
                .index_insert("wmsgs", column, branch, &Value::Uuid(row), row)
                .expect("entry");
        }
    }

    fn family_len(storage: &MemoryStorage, column: &str) -> usize {
        storage
            .raw_table_family_keys(&family_prefix("wmsgs", column), None, usize::MAX)
            .expect("family")
            .len()
    }

    /// The phases in order, each step bounded: a new index clears its prefix on every
    /// branch, then files every live row, then is complete; a removed one is cleared and
    /// forgotten. Nothing else under the table is touched.
    #[test]
    fn work_clears_then_fills_then_completes_and_clears_what_is_retired() {
        let mut storage = MemoryStorage::new();
        put_entries(&mut storage, "chat+at", "main", CLEAR_BATCH + 10);
        put_entries(&mut storage, "chat+at", "other-branch", 5);
        put_entries(&mut storage, "chat+atx", "main", 3);
        put_entries(&mut storage, "chat", "main", 7);
        propose(&mut storage, &declarations()).expect("declare");

        let phase = |storage: &MemoryStorage| {
            load_record(storage)
                .expect("record")
                .state("wmsgs", "chat+at")
                .expect("maintained")
                .phase
                .clone()
        };
        assert!(advance(&mut storage).expect("step").has_work());
        assert!(matches!(
            phase(&storage),
            IndexPhase::Clearing { after: Some(_) }
        ));
        assert!(advance(&mut storage).expect("step").has_work());
        assert!(matches!(
            phase(&storage),
            IndexPhase::Clearing { after: Some(_) }
        ));
        assert_eq!(family_len(&storage, "chat+at"), 0, "every branch cleared");
        assert_eq!(
            family_len(&storage, "chat+atx"),
            3,
            "a longer name is not the prefix"
        );
        assert_eq!(family_len(&storage, "chat"), 7, "other indexes untouched");

        // No live rows (`_id` is empty): the step that ends the clear completes it.
        assert!(!advance(&mut storage).expect("step").has_work());
        assert_eq!(phase(&storage), IndexPhase::Complete);
        assert!(!advance(&mut storage).expect("nothing to do").has_work());

        put_entries(&mut storage, "chat+at", "main", 4);
        propose(&mut storage, &IndexDeclarations::empty()).expect("give up");
        while advance(&mut storage).expect("step").has_work() {}
        let record = load_record(&storage).expect("record");
        assert!(record.retired.is_empty() && record.states.is_empty());
        assert_eq!(
            family_len(&storage, "chat+at"),
            0,
            "the retired index is cleared"
        );
    }

    /// The fill owes the rows live when its index's clear ended; rows written while it
    /// runs are their writes' to file. Their ids are UUIDv7, so they sort after the
    /// fill's cursor: a walk that took them too would not end while a page of them
    /// arrives per step.
    #[test]
    fn the_fill_ends_at_the_rows_live_when_it_began() {
        let mut storage = MemoryStorage::new();
        put_entries(&mut storage, "_id", "main", 2 * FILL_PAGE);
        propose(&mut storage, &declarations()).expect("declare");
        let mut steps = 0;
        while advance(&mut storage).expect("step").has_work() {
            steps += 1;
            assert!(steps < 20, "the fill never ended");
            put_entries(&mut storage, "_id", "main", FILL_PAGE);
        }
        assert_eq!(
            load_record(&storage)
                .expect("record")
                .complete_incarnation("wmsgs", "chat+at"),
            Some(0)
        );
    }

    /// The bound is a key, and its row can leave `_id` before the fill gets there: the
    /// walk passes where it was and still stops.
    #[test]
    fn the_fill_ends_at_its_bound_after_the_bounding_row_is_gone() {
        let mut storage = MemoryStorage::new();
        put_entries(&mut storage, "_id", "main", 2 * FILL_PAGE);
        propose(&mut storage, &declarations()).expect("declare");
        let record = advance(&mut storage).expect("the clear");
        let Some(IndexPhase::Filling { until, .. }) = record
            .state("wmsgs", "chat+at")
            .map(|state| state.phase.clone())
        else {
            panic!("the clear ended in a fill: {record:?}");
        };
        let (branch, _, row) = split_family_key(&until).expect("a family key");
        storage
            .index_remove("wmsgs", "_id", branch, &Value::Uuid(row), row)
            .expect("remove the bounding entry");
        let mut steps = 0;
        while advance(&mut storage).expect("step").has_work() {
            steps += 1;
            assert!(steps < 20, "the fill never ended");
            put_entries(&mut storage, "_id", "main", FILL_PAGE);
        }
        assert_eq!(
            load_record(&storage)
                .expect("record")
                .complete_incarnation("wmsgs", "chat+at"),
            Some(0)
        );
    }

    #[test]
    fn a_family_key_splits_from_the_right() {
        let id = ObjectId::new();
        let rest = format!("dev-abc-feature:x:0a0b:{}", id.uuid().simple());
        let (branch, key, row) = split_family_key(&rest).expect("splits");
        assert_eq!(branch, "dev-abc-feature:x");
        assert_eq!(key, format!("0a0b:{}", id.uuid().simple()));
        assert_eq!(row, id);
    }

    /// A store whose record cannot be read, while it may still read later: `inner`
    /// reads it.
    pub(crate) struct FailRecordReadStorage<S = MemoryStorage> {
        pub(crate) inner: S,
    }

    pub(crate) const SIMULATED_READ_FAILURE: &str = "simulated read failure";

    impl<S: Storage> Storage for FailRecordReadStorage<S> {
        fn raw_table_put(
            &mut self,
            table: &str,
            key: &str,
            value: &[u8],
        ) -> Result<(), StorageError> {
            self.inner.raw_table_put(table, key, value)
        }

        fn raw_table_delete(&mut self, table: &str, key: &str) -> Result<(), StorageError> {
            self.inner.raw_table_delete(table, key)
        }

        fn apply_raw_table_mutations(
            &mut self,
            mutations: &[RawTableMutation<'_>],
        ) -> Result<(), StorageError> {
            self.inner.apply_raw_table_mutations(mutations)
        }

        fn raw_table_get(&self, table: &str, key: &str) -> Result<Option<Vec<u8>>, StorageError> {
            if table == RECORD_TABLE {
                return Err(StorageError::IoError(SIMULATED_READ_FAILURE.into()));
            }
            self.inner.raw_table_get(table, key)
        }

        fn raw_table_scan_prefix_keys(
            &self,
            table: &str,
            prefix: &str,
        ) -> Result<RawTableKeys, StorageError> {
            self.inner.raw_table_scan_prefix_keys(table, prefix)
        }

        fn store_format_version(&self) -> Result<Option<i32>, StorageError> {
            self.inner.store_format_version()
        }

        fn set_store_format_version(&mut self, version: i32) -> Result<(), StorageError> {
            self.inner.set_store_format_version(version)
        }

        fn raw_table_scan_prefix(
            &self,
            table: &str,
            prefix: &str,
        ) -> Result<RawTableRows, StorageError> {
            self.inner.raw_table_scan_prefix(table, prefix)
        }

        fn raw_table_scan_range(
            &self,
            table: &str,
            start: Option<&str>,
            end: Option<&str>,
        ) -> Result<RawTableRows, StorageError> {
            self.inner.raw_table_scan_range(table, start, end)
        }

        fn append_history_region_row_bytes(
            &mut self,
            table: &str,
            rows: &[HistoryRowBytes<'_>],
        ) -> Result<(), StorageError> {
            self.inner.append_history_region_row_bytes(table, rows)
        }

        fn upsert_visible_region_row_bytes(
            &mut self,
            table: &str,
            rows: &[VisibleRowBytes<'_>],
        ) -> Result<(), StorageError> {
            self.inner.upsert_visible_region_row_bytes(table, rows)
        }

        fn apply_encoded_row_mutation(
            &mut self,
            table: &str,
            history_rows: &[OwnedHistoryRowBytes],
            visible_rows: &[OwnedVisibleRowBytes],
            index_mutations: &[IndexMutation<'_>],
        ) -> Result<(), StorageError> {
            self.inner.apply_encoded_row_mutation(
                table,
                history_rows,
                visible_rows,
                index_mutations,
            )
        }

        fn apply_prepared_row_mutation(
            &mut self,
            table: &str,
            history_rows: &[StoredRowBatch],
            visible_entries: &[VisibleRowEntry],
            encoded_history_rows: &[OwnedHistoryRowBytes],
            encoded_visible_rows: &[OwnedVisibleRowBytes],
            index_mutations: &[IndexMutation<'_>],
        ) -> Result<(), StorageError> {
            self.inner.apply_prepared_row_mutation(
                table,
                history_rows,
                visible_entries,
                encoded_history_rows,
                encoded_visible_rows,
                index_mutations,
            )
        }

        fn apply_index_mutations(
            &mut self,
            index_mutations: &[IndexMutation<'_>],
        ) -> Result<(), StorageError> {
            self.inner.apply_index_mutations(index_mutations)
        }

        fn upsert_catalogue_entry(
            &mut self,
            entry: &crate::catalogue::CatalogueEntry,
        ) -> Result<(), StorageError> {
            self.inner.upsert_catalogue_entry(entry)
        }

        fn load_catalogue_entry(
            &self,
            object_id: crate::object::ObjectId,
        ) -> Result<Option<crate::catalogue::CatalogueEntry>, StorageError> {
            self.inner.load_catalogue_entry(object_id)
        }
    }

    /// A split row's repair retires the entries of the indexes the record declares; of
    /// none when the record does not decode, which nothing trusts or rewrites. A read
    /// that fails is not that: the record may read again, and be trusted with the
    /// entries the repair left behind, so the repair fails with it.
    #[test]
    fn a_repair_skips_an_undecodable_record_but_not_a_failed_read() {
        let mut storage = MemoryStorage::new();
        assert_eq!(
            repair_declarations(&storage).expect("no record"),
            IndexDeclarations::empty()
        );
        propose(&mut storage, &declarations()).expect("declare");
        assert_eq!(
            repair_declarations(&storage).expect("a record"),
            declarations()
        );
        storage
            .raw_table_put(RECORD_TABLE, RECORD_KEY, &[0xff])
            .expect("garble the record");
        assert_eq!(
            repair_declarations(&storage).expect("an undecodable record"),
            IndexDeclarations::empty()
        );

        let failing = FailRecordReadStorage {
            inner: MemoryStorage::new(),
        };
        assert!(
            repair_declarations(&failing).is_err(),
            "a failed read of the record must fail the repair"
        );
    }

    /// A release gives up a record it cannot decode, and only that: a read that fails
    /// may succeed later, and the record it would have read still names the indexes
    /// whose entries writers keep. So `open` fails, released or not, and writes nothing;
    /// the runtime retries it (`QueryManager::refresh_declared_indexes`).
    #[test]
    fn a_release_resets_only_a_record_it_cannot_decode() {
        for release in [true, false] {
            let (_dir, path) = sqlite_file();
            let mut inner = SqliteStorage::open(&path).expect("open");
            propose(&mut inner, &declarations()).expect("declare");
            let before = inner
                .raw_table_get(RECORD_TABLE, RECORD_KEY)
                .expect("read")
                .expect("a record");
            assert_eq!(
                inner.store_format_version().expect("format"),
                Some(STORE_FORMAT_V4_DECLARED_INDEXES),
                "the store must be at format 4, or the gate proves nothing"
            );
            let mut failing = FailRecordReadStorage { inner };

            let error = open(&mut failing, release)
                .expect_err("an open over a record it failed to read must fail");
            assert!(
                error.to_string().contains(SIMULATED_READ_FAILURE),
                "release {release}: the open failed for another reason: {error}"
            );
            assert_eq!(
                failing
                    .inner
                    .raw_table_get(RECORD_TABLE, RECORD_KEY)
                    .expect("read"),
                Some(before),
                "release {release}: an open over a record it failed to read rewrote it"
            );
            assert_eq!(
                failing.inner.store_format_version().expect("format"),
                Some(STORE_FORMAT_V4_DECLARED_INDEXES),
                "release {release}: an open over a record it failed to read moved the format"
            );
        }
    }
}
