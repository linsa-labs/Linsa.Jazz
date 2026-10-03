use ahash::AHashSet;
use std::ops::Bound;

use crate::object::{BranchName, ObjectId};
use crate::query_manager::index::ScanCondition;
use crate::query_manager::types::{
    ColumnName, RowDescriptor, TableName, Tuple, TupleDelta, TupleDescriptor, Value,
};
use crate::row_format::decode_row;

use super::{SourceContext, SourceNode};

/// Source node that scans an index via Storage.
/// Emits TupleDelta with length-1 tuples based on the scan condition.
#[derive(Debug)]
pub struct IndexScanNode {
    pub table: TableName,
    pub column: ColumnName,
    pub branch: String,
    pub condition: ScanCondition,

    /// Output tuple descriptor (single element, unmaterialized).
    output_descriptor: TupleDescriptor,
    row_descriptor: RowDescriptor,

    /// Current set of tuples (length-1) matching the condition.
    current_tuples: AHashSet<Tuple>,
    /// Last scanned IDs (for computing deltas).
    last_scanned_ids: AHashSet<ObjectId>,
    /// Whether this node needs reprocessing.
    dirty: bool,
    /// Rows known to have changed since the last scan, delivered by
    /// [`Self::note_rows_changed`]. When the next settle finds ONLY these (no
    /// `needs_full`, empty overlay, baseline present), `scan` re-evaluates just these
    /// rows instead of rescanning the whole index — the full rescan made every
    /// write-tick cost O(result set) rather than O(delta).
    pending_changed_rows: AHashSet<ObjectId>,
    /// Forces the next scan to be a full rescan. Set initially, by any table-level
    /// dirty, and on pending-set overflow. Deliberately one-directional per cycle:
    /// row notes never clear it.
    ///
    /// Dirty sources that bypass this node entirely (graph-level `mark_all_dirty`,
    /// direct `mark_dirty(node_id)`, policy graphs) leave `pending_changed_rows`
    /// empty — and an empty pending set also falls through to the full rescan, so an
    /// unknown dirty source can never be served a stale incremental no-op.
    needs_full: bool,
    /// Whether at least one full scan has completed. Incremental deltas can only be
    /// applied on top of an exact baseline.
    has_scanned: bool,
    /// Ordered-window mode over a composite index (see [`WindowScan`]). When set,
    /// `condition` is unused.
    window: Option<WindowScan>,
    /// Substring-search mode over a trigram index (see [`TrigramScan`]). When set,
    /// `condition` is unused.
    trigram: Option<TrigramScan>,
    /// The incarnation of the declared index a window or a search reads that this scan
    /// last found complete in the store's record (`declared_index`). While there is
    /// none the scan reads the first column's own index (`undeclared_scan_ids`).
    ready_incarnation: Option<u64>,
}

/// Keys of a posting list a trigram search reads in its first round.
const TRIGRAM_FIRST_ROUND_KEYS: usize = 512;
/// Each round takes what is read of a list to this many times what it was.
const TRIGRAM_ROUND_GROWTH: usize = 4;
/// What the seek that starts a read of a walk costs, in keys read in sequence: 4–5 µs
/// against 0.15 µs a key on SQLite and on RocksDB (`stand_walk_against_whole_lists`).
const TRIGRAM_WALK_SEEK_KEYS: usize = 32;
/// The fewest keys a read of a walk asks for: among candidates far apart in a list every
/// read decides one, and what it reads past it is wasted.
const TRIGRAM_WALK_REACH: usize = 8;
/// What a read of the fewest keys costs, in keys: its seek and its keys. Candidates
/// closer together than this are cheaper read through than sought one by one, and a
/// read is worth its keys when it decided candidates such reads would have cost as much.
const TRIGRAM_WALK_SHORT_READ_COST: usize = TRIGRAM_WALK_SEEK_KEYS + TRIGRAM_WALK_REACH;
/// Of the reads in a row that decide a single candidate, one in this many — and the
/// first of a walk — asks for as many keys as a short read costs: a read of fewer cannot
/// tell candidates past its reach but cheaper to read through from candidates far apart.
const TRIGRAM_WALK_PROBE_EVERY: usize = 8;

#[cfg(test)]
thread_local! {
    /// Index keys the trigram searches of this thread have read.
    pub(crate) static TRIGRAM_KEYS_READ: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// Reads of a posting list the trigram searches of this thread have made.
    pub(crate) static TRIGRAM_LIST_READS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// Reads the trigram searches of this thread have made walking a list's candidates.
    pub(crate) static TRIGRAM_WALK_READS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// The walk reads of this thread that fail: from the first of the pair up to, not
    /// including, the second, counted as `TRIGRAM_WALK_READS` counts them.
    pub(crate) static TRIGRAM_WALK_READS_THAT_FAIL: std::cell::Cell<Option<(u64, u64)>> =
        const { std::cell::Cell::new(None) };
    /// Whether the walk reads of this thread are answered from the start of the list,
    /// as by a store that ignores where a range read starts.
    pub(crate) static TRIGRAM_WALK_READS_IGNORE_START: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
    /// The list read of this thread that fails, counted from when it was armed.
    pub(crate) static TRIGRAM_LIST_READ_THAT_FAILS: std::cell::Cell<Option<u64>> =
        const { std::cell::Cell::new(None) };
    /// A test's own first-round bound — a key at the least, nought is taken for one — so
    /// that a handful of rows takes a search through every round and every way a list is
    /// decided.
    pub(crate) static TRIGRAM_FIRST_ROUND_FOR_TEST: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
}

/// Counts a list read of this thread, and says whether it is the one armed to fail.
#[cfg(test)]
fn trigram_list_read_fails() -> bool {
    TRIGRAM_LIST_READS.with(|reads| reads.set(reads.get() + 1));
    TRIGRAM_LIST_READ_THAT_FAILS.with(|armed| match armed.get() {
        Some(0) => {
            armed.set(None);
            true
        }
        Some(left) => {
            armed.set(Some(left - 1));
            false
        }
        None => false,
    })
}

/// Counts a read of a walk of this thread, and says whether it is armed to fail.
#[cfg(test)]
fn trigram_walk_read_fails() -> bool {
    let read = TRIGRAM_WALK_READS.with(|count| {
        count.set(count.get() + 1);
        count.get() - 1
    });
    TRIGRAM_WALK_READS_THAT_FAIL.with(|armed| {
        armed
            .get()
            .is_some_and(|(from, to)| (from..to).contains(&read))
    })
}

/// The first round's bound, in keys.
fn trigram_first_round() -> usize {
    #[cfg(test)]
    if let Some(keys) = TRIGRAM_FIRST_ROUND_FOR_TEST.with(|keys| keys.get()) {
        return keys.max(1);
    }
    TRIGRAM_FIRST_ROUND_KEYS
}

/// The rows of one scope value whose case-folded text may hold a needle: the
/// intersection of the needle's trigram posting lists in a trigram index
/// (`query_manager::trigram_index`). A superset of the matches — the filter after
/// the load keeps only real ones — read as index keys, without loading a row.
#[derive(Debug, Clone)]
pub struct TrigramScan {
    /// The value segment of each of the needle's trigrams.
    segments: Vec<String>,
    /// For deciding overlay rows without the index.
    scope_column: usize,
    text_column: usize,
    scope_value: Value,
    folded_needle: String,
}

impl TrigramScan {
    /// A search for `needle` among the rows whose `scope` column is `scope_value`.
    /// Returns `None` when the folded needle has no trigram, or the scope value has no
    /// fixed-width encoding.
    pub fn new(
        scope_column: usize,
        text_column: usize,
        scope_value: Value,
        needle: &str,
    ) -> Option<Self> {
        use crate::query_manager::trigram_index::{entry_segment, fold, trigrams};
        let folded_needle = fold(needle);
        let segments = trigrams(&folded_needle)
            .iter()
            .map(|trigram| entry_segment(&scope_value, trigram))
            .collect::<Option<Vec<_>>>()?;
        if segments.is_empty() {
            return None;
        }
        Some(Self {
            segments,
            scope_column,
            text_column,
            scope_value,
            folded_needle,
        })
    }
}

/// An ordered window over a composite `(first, second)` index: the entries whose
/// first component equals one value, walked in second-component order from one
/// end of a range, and cut once enough of them are held.
///
/// Members are every entry between the walk's origin and the FRONTIER, which is the
/// second-component value of the last group fetched. A cut never splits a group of
/// equal second-component values, because the Sort downstream orders ties by id and
/// a split group could leave out the row it ranks first.
///
/// A settle with changes rescans only up to the frontier, which is O(window). When
/// the graph finds the window short after its filters (deleted or denied rows, rows
/// that left the range), it calls [`IndexScanNode::grow_window`], and the next scan
/// walks further.
#[derive(Debug, Clone)]
pub struct WindowScan {
    /// `"09" ++ hex(encode(first))`: the value-segment prefix every member shares.
    prefix_hex: String,
    /// The range's entry-key bounds, `[start, end)`.
    start: String,
    end: String,
    /// Walk from `end` towards `start` (descending second component).
    reverse: bool,
    /// How many entries the next extending walk must reach before it may stop.
    want: usize,
    /// Hex of the second component of the last group the walk fetched. `None`
    /// until the first walk, and whenever the walk ran to the end of its range.
    frontier: Option<String>,
    /// The last walk reached the end of its range: every entry in `[start, end)`
    /// is a member.
    exhausted: bool,
    /// Set by [`IndexScanNode::grow_window`]: the next scan walks past the
    /// frontier instead of rescanning inside it.
    extend: bool,
    /// For deciding overlay rows without the index: the composite columns in the
    /// row descriptor, the first component's value and the second component's
    /// encoded bounds.
    first_column: usize,
    second_column: usize,
    first_value: Value,
    lower_hex: Bound<String>,
    upper_hex: Bound<String>,
}

impl WindowScan {
    /// A window over `first = first_value` with the second component in
    /// `lower..upper`. Returns `None` when a value has no fixed-width encoding.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        first_column: usize,
        second_column: usize,
        first_value: Value,
        lower: Bound<Value>,
        upper: Bound<Value>,
        reverse: bool,
        want: usize,
    ) -> Option<Self> {
        use crate::query_manager::composite_index::{fixed_width_encoding, hex};
        let prefix_hex = format!("09{}", hex(&fixed_width_encoding(&first_value)?));
        let encode_bound = |bound: Bound<Value>| -> Option<Bound<String>> {
            Some(match bound {
                Bound::Unbounded => Bound::Unbounded,
                Bound::Included(value) => Bound::Included(hex(&fixed_width_encoding(&value)?)),
                Bound::Excluded(value) => Bound::Excluded(hex(&fixed_width_encoding(&value)?)),
            })
        };
        let lower_hex = encode_bound(lower)?;
        let upper_hex = encode_bound(upper)?;
        // Entry keys are `{prefix}{second}:{uuid}` with a fixed-width `second`, so
        // `{prefix}{second};` sorts after every key of that second value and before
        // any larger one (';' is the byte after ':'), and `{prefix}g` after every key
        // of the prefix ('g' follows the hex digits).
        let start = match &lower_hex {
            Bound::Unbounded => prefix_hex.clone(),
            Bound::Included(second) => format!("{prefix_hex}{second}"),
            Bound::Excluded(second) => format!("{prefix_hex}{second};"),
        };
        let end = match &upper_hex {
            Bound::Unbounded => format!("{prefix_hex}g"),
            Bound::Included(second) => format!("{prefix_hex}{second};"),
            Bound::Excluded(second) => format!("{prefix_hex}{second}"),
        };
        Some(Self {
            prefix_hex,
            start,
            end,
            reverse,
            want: want.max(1),
            frontier: None,
            exhausted: false,
            extend: true,
            first_column,
            second_column,
            first_value,
            lower_hex,
            upper_hex,
        })
    }

    /// The second component of an entry key, as hex.
    fn second_of<'k>(&self, key: &'k str) -> Option<&'k str> {
        let (segment, _) = key.rsplit_once(':')?;
        segment.strip_prefix(self.prefix_hex.as_str())
    }

    /// The member bounds: the whole range when exhausted, else the range cut at the
    /// frontier group.
    fn member_bounds(&self) -> (String, String) {
        match (&self.frontier, self.exhausted) {
            (Some(frontier), false) if self.reverse => {
                (format!("{}{frontier}", self.prefix_hex), self.end.clone())
            }
            (Some(frontier), false) => (
                self.start.clone(),
                format!("{}{frontier};", self.prefix_hex),
            ),
            _ => (self.start.clone(), self.end.clone()),
        }
    }

    /// Whether a second-component value (hex) is inside the member bounds.
    fn holds_second(&self, second_hex: &str) -> bool {
        let lower_ok = match &self.lower_hex {
            Bound::Unbounded => true,
            Bound::Included(lower) => second_hex >= lower.as_str(),
            Bound::Excluded(lower) => second_hex > lower.as_str(),
        };
        let upper_ok = match &self.upper_hex {
            Bound::Unbounded => true,
            Bound::Included(upper) => second_hex <= upper.as_str(),
            Bound::Excluded(upper) => second_hex < upper.as_str(),
        };
        let frontier_ok = match (&self.frontier, self.exhausted) {
            (Some(frontier), false) if self.reverse => second_hex >= frontier.as_str(),
            (Some(frontier), false) => second_hex <= frontier.as_str(),
            _ => true,
        };
        lower_ok && upper_ok && frontier_ok
    }
}

/// Above this many accumulated changed rows a full rescan is cheaper and the set is
/// dropped. Also bounds the memory a long-unsettled graph can pin.
const MAX_PENDING_CHANGED_ROWS: usize = 4096;

impl IndexScanNode {
    /// Create a new index scan node.
    pub fn new_with_branch(
        table: impl Into<TableName>,
        column: impl Into<ColumnName>,
        branch: impl Into<String>,
        condition: ScanCondition,
        row_descriptor: RowDescriptor,
    ) -> Self {
        let table = table.into();
        let output_descriptor = TupleDescriptor::single(table.as_str(), row_descriptor.clone());
        Self {
            table,
            column: column.into(),
            branch: branch.into(),
            condition,
            output_descriptor,
            row_descriptor,
            current_tuples: AHashSet::new(),
            last_scanned_ids: AHashSet::new(),
            dirty: true,
            pending_changed_rows: AHashSet::new(),
            needs_full: true,
            has_scanned: false,
            window: None,
            trigram: None,
            ready_incarnation: None,
        }
    }

    /// A substring-search scan over the trigram index `column` (see [`TrigramScan`]).
    pub fn new_trigram(
        table: impl Into<TableName>,
        column: impl Into<ColumnName>,
        branch: impl Into<String>,
        trigram: TrigramScan,
        row_descriptor: RowDescriptor,
    ) -> Self {
        let mut node =
            Self::new_with_branch(table, column, branch, ScanCondition::All, row_descriptor);
        node.trigram = Some(trigram);
        node
    }

    /// An ordered-window scan over the composite index `column` (see [`WindowScan`]).
    pub fn new_window(
        table: impl Into<TableName>,
        column: impl Into<ColumnName>,
        branch: impl Into<String>,
        window: WindowScan,
        row_descriptor: RowDescriptor,
    ) -> Self {
        let mut node =
            Self::new_with_branch(table, column, branch, ScanCondition::All, row_descriptor);
        node.window = Some(window);
        node
    }

    /// Whether the window must walk further for the page to be exact. `edge` is the
    /// encoded (hex) second-component value of the page's last row, or `None` when the
    /// ordered input holds fewer rows than the page. Rows past the frontier order after
    /// it, so the page is exact once the frontier reaches the edge.
    pub(crate) fn window_needs_growth(&self, edge: Option<&str>) -> bool {
        let Some(window) = self.window.as_ref() else {
            return false;
        };
        if window.exhausted {
            return false;
        }
        match (edge, window.frontier.as_deref()) {
            (Some(edge), Some(frontier)) if window.reverse => frontier > edge,
            (Some(edge), Some(frontier)) => frontier < edge,
            _ => true,
        }
    }

    /// Whether this scan reads an ordered window (`new_window`).
    #[cfg(test)]
    pub(crate) fn is_window(&self) -> bool {
        self.window.is_some()
    }

    /// Whether this is a window whose walk may stop before the end of its range, so that
    /// its graph has to grow it (`QueryGraph::grow_windows_short_of`).
    #[cfg(test)]
    pub(crate) fn window_may_stop_short(&self) -> bool {
        self.window
            .as_ref()
            .is_some_and(|window| window.want != usize::MAX && !window.exhausted)
    }

    /// Walk the window's whole range from now on, for a graph that cannot grow it.
    pub(crate) fn walk_whole_window(&mut self) {
        if let Some(window) = self.window.as_mut() {
            window.want = usize::MAX;
        }
    }

    /// Ask the next scan to walk past the frontier, twice as far as the last walk.
    /// Returns false when the window already holds its whole range.
    pub(crate) fn grow_window(&mut self) -> bool {
        let Some(window) = self.window.as_mut() else {
            return false;
        };
        if window.exhausted {
            return false;
        }
        window.want = window.want.saturating_mul(2);
        window.extend = true;
        self.dirty = true;
        self.needs_full = true;
        true
    }

    /// Record that exactly `ids` changed in this node's table since the last scan.
    ///
    /// Never downgrades a required full rescan: once row information was lost for this
    /// cycle, only a full rescan restores an exact baseline.
    pub fn note_rows_changed<'a>(&mut self, ids: impl IntoIterator<Item = &'a ObjectId>) {
        self.dirty = true;
        if self.needs_full {
            return;
        }
        self.pending_changed_rows.extend(ids.into_iter().copied());
        if self.pending_changed_rows.len() > MAX_PENDING_CHANGED_ROWS {
            self.pending_changed_rows.clear();
            self.needs_full = true;
        }
    }

    /// Record a table-level change with no row information: the next scan is full.
    pub fn note_table_dirty(&mut self) {
        self.dirty = true;
        self.pending_changed_rows.clear();
        self.needs_full = true;
    }

    /// Create a new index scan node on the default "main" branch.
    pub fn new(
        table: impl Into<TableName>,
        column: impl Into<ColumnName>,
        condition: ScanCondition,
        row_descriptor: RowDescriptor,
    ) -> Self {
        Self::new_with_branch(table, column, "main", condition, row_descriptor)
    }

    /// Get the output tuple descriptor.
    pub fn output_tuple_descriptor(&self) -> &TupleDescriptor {
        &self.output_descriptor
    }

    fn overlay_value_matches_condition(&self, row_id: ObjectId, data: &[u8]) -> bool {
        let Some(value) = self.overlay_index_value(row_id, data) else {
            return false;
        };
        match &self.condition {
            ScanCondition::All => true,
            ScanCondition::Empty => false,
            ScanCondition::Eq(expected) => value == *expected || array_contains(&value, expected),
            ScanCondition::Range { min, max } => {
                bound_matches(min, &value, true) && bound_matches(max, &value, false)
            }
        }
    }

    fn overlay_index_value(&self, row_id: ObjectId, data: &[u8]) -> Option<Value> {
        if self.column.as_str() == "_id" {
            return Some(Value::Uuid(row_id));
        }
        if self.column.as_str() == "_id_deleted" {
            return None;
        }

        let column_index = self.row_descriptor.column_index(self.column.as_str())?;
        let values = decode_row(&self.row_descriptor, data).ok()?;
        values.get(column_index).cloned()
    }

    /// The value of a plain `column = value` lookup, or `None` for any other
    /// scan: a range, a full or empty scan, an ordered window, a search.
    pub(crate) fn plain_eq_value(&self) -> Option<&Value> {
        match &self.condition {
            ScanCondition::Eq(value) if self.window.is_none() && self.trigram.is_none() => {
                Some(value)
            }
            _ => None,
        }
    }

    /// Row ids this scan currently considers members.
    ///
    /// This is PRE-filter membership: everything the correlate predicate
    /// matches, including rows a downstream filter, policy or window later
    /// drops. Include routing needs exactly that set — a row an instance
    /// scans but does not output still has to be re-checked when it changes,
    /// or the scan's incremental baseline drifts from a full rescan (the
    /// parity harness in [`Self::scan`] is what catches the drift).
    pub(crate) fn scanned_row_ids(&self) -> impl Iterator<Item = ObjectId> + '_ {
        self.last_scanned_ids.iter().copied()
    }

    /// Whether this node holds an exact baseline that nothing has invalidated:
    /// it has scanned at least once, no table-level mark demanded a rescan, no
    /// row was reported changed, and it is not itself dirty.
    ///
    /// A node in this state can only rescan to the membership it already has,
    /// so the settle may skip it — see the call site in `graph/execute.rs`,
    /// which is the only place allowed to make that judgement because it is
    /// the only one that can also see the graph-level dirty bit
    /// (`mark_all_dirty` sets the bitmap without touching this node).
    pub(crate) fn holds_exact_baseline(&self) -> bool {
        self.has_scanned && !self.needs_full && !self.dirty && self.pending_changed_rows.is_empty()
    }

    /// Whether this node's condition admits per-row membership checks against the
    /// index itself. Only shapes answerable by an exact index-key point read qualify —
    /// reading the same raw table the full scan traverses is what makes the
    /// incremental path immune to row-encoding and schema-variant concerns.
    fn supports_incremental_membership(&self) -> bool {
        if self.window.is_some() || self.trigram.is_some() {
            return false;
        }
        match &self.condition {
            ScanCondition::Empty | ScanCondition::Eq(_) => true,
            // An `All` scan over the `_id` index: a row's entry value is its own id.
            ScanCondition::All => self.column.as_str() == "_id",
            ScanCondition::Range { .. } => false,
        }
    }

    /// Per-row membership straight from the index — the exact raw table the full scan
    /// reads, via a point read. Only called for conditions where
    /// [`Self::supports_incremental_membership`] holds.
    fn index_row_membership(&self, ctx: &SourceContext, row_id: ObjectId) -> bool {
        // Settle-cost accounting: one index point read.
        crate::query_manager::settle_cost::bump(&crate::query_manager::settle_cost::INDEX_READS);
        match &self.condition {
            ScanCondition::Empty => false,
            ScanCondition::Eq(value) => ctx
                .storage
                .index_contains(
                    self.table.as_str(),
                    self.column.as_str(),
                    &self.branch,
                    value,
                    row_id,
                )
                .unwrap_or(false),
            ScanCondition::All => ctx
                .storage
                .index_contains(
                    self.table.as_str(),
                    self.column.as_str(),
                    &self.branch,
                    &Value::Uuid(row_id),
                    row_id,
                )
                .unwrap_or(false),
            ScanCondition::Range { .. } => false,
        }
    }

    /// The window's members, walking past the frontier when asked to grow. Moves the
    /// frontier, so it runs once per scan (the parity harness never reaches it: a
    /// window has no incremental path).
    fn window_scan_ids(&mut self, ctx: &SourceContext) -> AHashSet<ObjectId> {
        let Some(window) = self.window.as_mut() else {
            return AHashSet::new();
        };
        let reverse = window.reverse;
        let read = |start: &str, end: &str, limit: Option<usize>| {
            crate::query_manager::settle_cost::bump(
                &crate::query_manager::settle_cost::INDEX_READS,
            );
            ctx.storage
                .index_window_keys(
                    self.table.as_str(),
                    self.column.as_str(),
                    &self.branch,
                    start,
                    end,
                    reverse,
                    limit,
                )
                .unwrap_or_default()
        };
        let keys: Vec<String> = if window.extend || window.frontier.is_none() && !window.exhausted {
            window.extend = false;
            let limit = (window.want != usize::MAX).then_some(window.want);
            let mut keys = read(&window.start, &window.end, limit);
            if keys.len() < window.want {
                window.exhausted = true;
                window.frontier = None;
            } else {
                window.exhausted = false;
                let last_second = keys
                    .last()
                    .and_then(|key| window.second_of(key))
                    .map(str::to_string);
                if let Some(second) = last_second {
                    // Finish the last group so a tie is never split.
                    let group_start = format!("{}{second}", window.prefix_hex);
                    let group_end = format!("{}{second};", window.prefix_hex);
                    for key in read(&group_start, &group_end, None) {
                        if !keys.contains(&key) {
                            keys.push(key);
                        }
                    }
                    window.frontier = Some(second);
                }
            }
            keys
        } else {
            let (start, end) = window.member_bounds();
            read(&start, &end, None)
        };
        keys.iter()
            .filter_map(|key| crate::query_manager::composite_index::entry_row_id(key))
            .collect()
    }

    /// The candidates of a trigram search: the needle's posting lists intersected.
    ///
    /// A posting list is as long as its trigram is common in the scope, and a needle's
    /// common trigrams rule out next to nothing its rare ones have not: reading every
    /// list whole made a search cost the scope, whatever it found. So the lists are read
    /// in rounds, each list from where the round before left it and up to a bound that
    /// grows from round to round, until one ends: the lists that end are intersected,
    /// and from then on a list still open is no longer read on.
    ///
    /// A list is in row id order. What an open one has read, from its start to its last
    /// key, has decided every candidate at or before that key. The candidates past it
    /// are walked in id order: the list is read from the first of them, which decides
    /// every candidate up to the last key that read returns, and then from the next one
    /// still undecided. A read asks for twice as many keys as the read before while
    /// the keys read stay cheaper than seeks to the candidates they decide, and for a
    /// few keys otherwise: far apart in the list, candidates cost a seek each; close
    /// together, the keys between them; and the list never its length. Either way that
    /// is within a small factor of the cheaper of seeking to every candidate and reading
    /// the list through them.
    ///
    /// What comes back is the intersection reading every list whole would give — a
    /// superset of the matches, which the filter after the load narrows, as it did. A
    /// read the store fails rules nothing out, and `None` comes back when no list could
    /// be read: nothing is known of the scope's rows then.
    fn trigram_scan_ids(&self, ctx: &SourceContext) -> Option<AHashSet<ObjectId>> {
        let trigram = self.trigram.as_ref()?;
        /// A posting list not read to its end yet.
        struct Open<'a> {
            segment: &'a str,
            ids: AHashSet<ObjectId>,
            /// The last key read, once a round has read any.
            last: Option<String>,
            /// Keys read of it so far.
            read: usize,
        }
        let first_round = trigram_first_round();
        // A store that reads a whole range to hand back its first keys reads every list
        // whole in one round: a round there costs the list, however few keys it asks for.
        let first_round = if ctx.storage.limited_range_scans_are_bounded() {
            first_round
        } else {
            usize::MAX
        };

        let mut candidates: Option<AHashSet<ObjectId>> = None;
        let mut open: Vec<Open<'_>> = trigram
            .segments
            .iter()
            .map(|segment| Open {
                segment,
                ids: AHashSet::new(),
                last: None,
                read: 0,
            })
            .collect();
        while !open.is_empty() {
            let mut ended: Vec<AHashSet<ObjectId>> = Vec::new();
            let mut still_open = Vec::with_capacity(open.len());
            for mut list in open {
                let round_keys = if list.read == 0 {
                    first_round
                } else {
                    list.read.saturating_mul(TRIGRAM_ROUND_GROWTH - 1)
                };
                if let Some(candidates) = candidates.as_mut() {
                    // What the list has read is decided (below); the candidates past
                    // its last key are not, and the list is read from each of them in
                    // turn rather than on from where it stands.
                    let last = list
                        .last
                        .as_deref()
                        .and_then(crate::query_manager::composite_index::entry_row_id);
                    let mut undecided: Vec<ObjectId> = candidates
                        .iter()
                        .copied()
                        .filter(|id| last.is_none_or(|last| *id > last))
                        .collect();
                    undecided.sort_unstable();
                    let end = format!("{};", list.segment);
                    let mut reach = TRIGRAM_WALK_SHORT_READ_COST;
                    // Reads in a row that were not worth their keys.
                    let mut alone = 0usize;
                    let mut at = 0;
                    while at < undecided.len() {
                        let start = format!(
                            "{}:{}",
                            list.segment,
                            crate::query_manager::composite_index::hex(
                                undecided[at].uuid().as_bytes()
                            )
                        );
                        #[cfg(test)]
                        let start = if TRIGRAM_WALK_READS_IGNORE_START.with(|armed| armed.get()) {
                            format!("{}:", list.segment)
                        } else {
                            start
                        };
                        crate::query_manager::settle_cost::bump(
                            &crate::query_manager::settle_cost::INDEX_READS,
                        );
                        let keys = ctx.storage.index_window_keys(
                            self.table.as_str(),
                            self.column.as_str(),
                            &self.branch,
                            &start,
                            &end,
                            false,
                            Some(reach),
                        );
                        #[cfg(test)]
                        let keys = if trigram_walk_read_fails() {
                            Err(crate::storage::StorageError::IoError(
                                "a walk read armed to fail".to_string(),
                            ))
                        } else {
                            keys
                        };
                        // A read that fails rules nothing out: the candidates from here
                        // on stay.
                        let Ok(keys) = keys else {
                            break;
                        };
                        #[cfg(test)]
                        TRIGRAM_KEYS_READ.with(|read| read.set(read.get() + keys.len() as u64));
                        let held: Vec<ObjectId> = keys
                            .iter()
                            .filter_map(|key| {
                                crate::query_manager::composite_index::entry_row_id(key)
                            })
                            .collect();
                        // Fewer keys than asked for is the end of the list: it has
                        // decided every candidate left. Otherwise the read has decided
                        // the candidates up to its last key — and nothing, when that
                        // key is before the one it was asked to start from: the walk
                        // stops rather than ask again.
                        let through = if keys.len() < reach {
                            None
                        } else {
                            match held.last() {
                                Some(through) if *through >= undecided[at] => Some(*through),
                                _ => break,
                            }
                        };
                        let from = at;
                        let mut held_at = 0;
                        while at < undecided.len()
                            && through.is_none_or(|through| undecided[at] <= through)
                        {
                            while held_at < held.len() && held[held_at] < undecided[at] {
                                held_at += 1;
                            }
                            if held.get(held_at) != Some(&undecided[at]) {
                                candidates.remove(&undecided[at]);
                            }
                            at += 1;
                        }
                        // A read whose keys cost no more than short reads to the
                        // candidates it decided would have is the cheaper way through
                        // this part of the list: the next read reaches twice as far.
                        // After one that was not, the candidates are taken to be far
                        // apart, and the reads short — but for one in a few, which asks
                        // again whether they are.
                        let decided = at - from;
                        reach = if decided >= 2
                            && decided.saturating_mul(TRIGRAM_WALK_SHORT_READ_COST) >= reach
                        {
                            // Two candidates in a read of fewer keys than a short read
                            // costs are no sign yet that the ones after them stand
                            // close: only a read worth its seek starts the count over.
                            if reach >= TRIGRAM_WALK_SHORT_READ_COST {
                                alone = 0;
                            }
                            reach.saturating_mul(2)
                        } else {
                            alone += 1;
                            if alone.is_multiple_of(TRIGRAM_WALK_PROBE_EVERY) {
                                TRIGRAM_WALK_SHORT_READ_COST
                            } else {
                                TRIGRAM_WALK_REACH
                            }
                        };
                    }
                    continue;
                }
                // A key followed by a NUL byte is the least key greater than it.
                let start = match list.last.as_deref() {
                    Some(last) => format!("{last}\0"),
                    None => format!("{}:", list.segment),
                };
                crate::query_manager::settle_cost::bump(
                    &crate::query_manager::settle_cost::INDEX_READS,
                );
                let keys = ctx.storage.index_window_keys(
                    self.table.as_str(),
                    self.column.as_str(),
                    &self.branch,
                    &start,
                    &format!("{};", list.segment),
                    false,
                    Some(round_keys),
                );
                #[cfg(test)]
                let keys = if trigram_list_read_fails() {
                    Err(crate::storage::StorageError::IoError(
                        "a list read armed to fail".to_string(),
                    ))
                } else {
                    keys
                };
                // A list that cannot be read rules nothing out, and what was read of it
                // is not the list: it is left out.
                let Ok(mut keys) = keys else {
                    continue;
                };
                #[cfg(test)]
                TRIGRAM_KEYS_READ.with(|read| read.set(read.get() + keys.len() as u64));
                list.ids
                    .extend(keys.iter().filter_map(|key| {
                        crate::query_manager::composite_index::entry_row_id(key)
                    }));
                if keys.len() >= round_keys {
                    // As many as were asked for: there may be more.
                    list.read = list.read.saturating_add(keys.len());
                    list.last = keys.pop();
                    still_open.push(list);
                } else if list.ids.is_empty() {
                    // No row of the scope holds this trigram.
                    return Some(AHashSet::new());
                } else {
                    ended.push(list.ids);
                }
            }
            open = still_open;
            ended.sort_by_key(|list| list.len());
            for list in ended {
                match candidates.as_mut() {
                    Some(candidates) => candidates.retain(|id| list.contains(id)),
                    None => candidates = Some(list),
                }
            }
            // A list is in row id order and an open one was read from its start to its
            // last key: a candidate at or before that key and not among what was read is
            // not in the list, whatever the rest of it holds.
            if let Some(candidates) = candidates.as_mut() {
                for list in &open {
                    let Some(last) = list
                        .last
                        .as_deref()
                        .and_then(crate::query_manager::composite_index::entry_row_id)
                    else {
                        continue;
                    };
                    candidates.retain(|id| *id > last || list.ids.contains(id));
                }
            }
        }
        candidates
    }

    /// Whether the declared index this window or search reads is complete. Asked of the
    /// store's record on every scan and held to the incarnation seen complete: an index
    /// removed, or re-added and so filled afresh, sends the scan back to the first
    /// column's own index. A window starts its walk over whenever its source changes.
    fn declared_index_ready(&mut self, ctx: &SourceContext) -> bool {
        let complete = crate::query_manager::declared_index::load_record(ctx.storage)
            .ok()
            .and_then(|record| {
                record.complete_incarnation(self.table.as_str(), self.column.as_str())
            });
        if complete != self.ready_incarnation {
            self.ready_incarnation = complete;
            if let Some(window) = self.window.as_mut() {
                window.exhausted = false;
                window.frontier = None;
                window.extend = false;
            }
        }
        self.ready_incarnation.is_some()
    }

    /// The rows of the first (scope) column's value, read through that column's own
    /// index, as the plan read them before the declaration. A superset of the window's
    /// or the search's rows, which the Filter, the Sort and the page after the scan
    /// narrow exactly: the Filter is elided only when that one column covers the whole
    /// predicate, and then these are exactly the rows.
    fn undeclared_scan_ids(&mut self, ctx: &SourceContext) -> AHashSet<ObjectId> {
        let (column, value) = if let Some(window) = self.window.as_mut() {
            // It holds its whole range, so nothing grows it.
            window.exhausted = true;
            window.frontier = None;
            (window.first_column, window.first_value.clone())
        } else if let Some(trigram) = self.trigram.as_ref() {
            (trigram.scope_column, trigram.scope_value.clone())
        } else {
            return AHashSet::new();
        };
        crate::query_manager::settle_cost::bump(&crate::query_manager::settle_cost::INDEX_READS);
        let name = self.row_descriptor.columns[column].name;
        ctx.storage
            .index_lookup(self.table.as_str(), name.as_str(), &self.branch, &value)
            .into_iter()
            .collect()
    }

    /// Overlay membership for a trigram search: decided from the row itself.
    fn trigram_holds_row(&self, data: &[u8]) -> bool {
        let Some(trigram) = self.trigram.as_ref() else {
            return false;
        };
        let Ok(values) = decode_row(&self.row_descriptor, data) else {
            return false;
        };
        match (
            values.get(trigram.scope_column),
            values.get(trigram.text_column),
        ) {
            (Some(scope), Some(Value::Text(text))) => {
                *scope == trigram.scope_value
                    && crate::query_manager::trigram_index::fold(text)
                        .contains(trigram.folded_needle.as_str())
            }
            _ => false,
        }
    }

    /// Overlay membership for a window: decided from the row itself, since the
    /// overlay row's index entry may not exist yet.
    fn window_holds_row(&self, data: &[u8]) -> bool {
        let Some(window) = self.window.as_ref() else {
            return false;
        };
        let Ok(values) = decode_row(&self.row_descriptor, data) else {
            return false;
        };
        let (Some(first), Some(second)) = (
            values.get(window.first_column),
            values.get(window.second_column),
        ) else {
            return false;
        };
        if *first != window.first_value {
            return false;
        }
        let Some(second) = crate::query_manager::composite_index::fixed_width_encoding(second)
        else {
            return false;
        };
        window.holds_second(&crate::query_manager::composite_index::hex(&second))
    }

    /// The full-scan membership computation, factored out so the incremental path can
    /// assert parity against it in debug builds.
    fn full_scan_ids(&self, ctx: &SourceContext) -> AHashSet<ObjectId> {
        let mut new_ids: AHashSet<ObjectId> = match &self.condition {
            ScanCondition::Empty => AHashSet::new(),
            ScanCondition::All => ctx
                .storage
                .index_scan_all(self.table.as_str(), self.column.as_str(), &self.branch)
                .into_iter()
                .collect(),
            ScanCondition::Eq(value) => ctx
                .storage
                .index_lookup(
                    self.table.as_str(),
                    self.column.as_str(),
                    &self.branch,
                    value,
                )
                .into_iter()
                .collect(),
            ScanCondition::Range { min, max } => {
                let start = min.as_ref();
                let end = max.as_ref();
                ctx.storage
                    .index_range(
                        self.table.as_str(),
                        self.column.as_str(),
                        &self.branch,
                        start,
                        end,
                    )
                    .into_iter()
                    .collect()
            }
        };
        self.apply_local_overlay_rows(ctx, &mut new_ids);
        new_ids
    }

    fn apply_local_overlay_rows(&self, ctx: &SourceContext, new_ids: &mut AHashSet<ObjectId>) {
        let Some(local_overlay_rows) = ctx.local_overlay_rows else {
            return;
        };

        for (&row_id, row_batch_key) in local_overlay_rows {
            if row_batch_key.branch_name.as_str() != self.branch {
                continue;
            }
            // The overlay is `QueryManager::pending_local_row_batches`, which is global: it
            // holds every locally written row of every table until a non-local update
            // confirms it. The load below resolves by row id and ignores the table name it
            // is handed, so without this check a scan over one table reads the full bytes
            // of rows belonging to another — and, where the condition happens to match,
            // admits them into this index's id set. Measured on a device store, an upload
            // left ~390 unconfirmed one-megabyte `file_parts` rows in the overlay and every
            // settle paid 384 MB walking them.
            //
            // A row with no locator falls through, preserving the resolver's own fallback.
            if let Ok(Some(locator)) = ctx.storage.load_row_locator(row_id)
                && locator.table.as_str() != self.table.as_str()
            {
                continue;
            }
            let Ok(Some(row)) = ctx.storage.load_history_query_row_batch(
                self.table.as_str(),
                self.branch.as_str(),
                row_id,
                row_batch_key.batch_id,
            ) else {
                continue;
            };
            if self.column.as_str() == "_id_deleted" && row.is_soft_deleted() {
                new_ids.insert(row_id);
            } else if row.is_soft_deleted() || row.is_hard_deleted() {
                new_ids.remove(&row_id);
            } else if if self.window.is_some() {
                self.window_holds_row(&row.data)
            } else if self.trigram.is_some() {
                self.trigram_holds_row(&row.data)
            } else {
                self.overlay_value_matches_condition(row_id, &row.data)
            } {
                new_ids.insert(row_id);
            } else {
                new_ids.remove(&row_id);
            }
        }
    }
}

fn array_contains(value: &Value, expected: &Value) -> bool {
    matches!(value, Value::Array(values) if values.iter().any(|value| value == expected))
}

fn compare_values_for_ordering(left: &Value, right: &Value) -> Option<std::cmp::Ordering> {
    match (left, right) {
        (Value::Integer(a), Value::Integer(b)) => Some(a.cmp(b)),
        (Value::BigInt(a), Value::BigInt(b)) => Some(a.cmp(b)),
        (Value::Double(a), Value::Double(b)) => Some(a.total_cmp(b)),
        (Value::Boolean(a), Value::Boolean(b)) => Some(a.cmp(b)),
        (Value::Text(a), Value::Text(b)) => Some(a.cmp(b)),
        (Value::Timestamp(a), Value::Timestamp(b)) => Some(a.cmp(b)),
        (Value::Uuid(a), Value::Uuid(b)) => Some(a.cmp(b)),
        (Value::Null, Value::Null) => Some(std::cmp::Ordering::Equal),
        (Value::Null, _) => Some(std::cmp::Ordering::Less),
        (_, Value::Null) => Some(std::cmp::Ordering::Greater),
        _ => None,
    }
}

fn bound_matches(bound: &Bound<Value>, value: &Value, is_lower: bool) -> bool {
    match bound {
        Bound::Unbounded => true,
        Bound::Included(bound) => compare_values_for_ordering(value, bound)
            .map(|ordering| {
                if is_lower {
                    matches!(
                        ordering,
                        std::cmp::Ordering::Greater | std::cmp::Ordering::Equal
                    )
                } else {
                    matches!(
                        ordering,
                        std::cmp::Ordering::Less | std::cmp::Ordering::Equal
                    )
                }
            })
            .unwrap_or(false),
        Bound::Excluded(bound) => compare_values_for_ordering(value, bound)
            .map(|ordering| {
                if is_lower {
                    ordering == std::cmp::Ordering::Greater
                } else {
                    ordering == std::cmp::Ordering::Less
                }
            })
            .unwrap_or(false),
    }
}

impl SourceNode for IndexScanNode {
    fn scan(&mut self, ctx: &SourceContext) -> TupleDelta {
        // Incremental path: an exact baseline exists, nothing demanded a full rescan,
        // and this cycle's changes are known row by row. Only while the local overlay
        // is empty — overlay rows override storage state with an independent
        // lifecycle, so any overlay presence falls back to the full rescan that
        // handles it today.
        let changed = std::mem::take(&mut self.pending_changed_rows);
        let overlay_is_empty = ctx
            .local_overlay_rows
            .map(|rows| rows.is_empty())
            .unwrap_or(true);
        if !self.needs_full
            && self.has_scanned
            && overlay_is_empty
            && !changed.is_empty()
            && self.supports_incremental_membership()
        {
            let branch = BranchName::new(&self.branch);
            let mut added: Vec<ObjectId> = Vec::new();
            let mut removed: Vec<ObjectId> = Vec::new();
            for row_id in changed {
                let is_member = self.index_row_membership(ctx, row_id);
                let was_member = self.last_scanned_ids.contains(&row_id);
                match (was_member, is_member) {
                    (false, true) => {
                        self.last_scanned_ids.insert(row_id);
                        self.current_tuples
                            .insert(Tuple::from_scoped_id(row_id, branch));
                        added.push(row_id);
                    }
                    (true, false) => {
                        self.last_scanned_ids.remove(&row_id);
                        self.current_tuples
                            .remove(&Tuple::from_scoped_id(row_id, branch));
                        removed.push(row_id);
                    }
                    _ => {}
                }
            }

            // Parity harness: in debug builds every incremental result is checked
            // against the full rescan, so the entire test suite exercises the
            // equivalence continuously.
            #[cfg(debug_assertions)]
            {
                let full = self.full_scan_ids(ctx);
                debug_assert_eq!(
                    full, self.last_scanned_ids,
                    "incremental index scan diverged from full rescan (table {}, column {}, branch {})",
                    self.table, self.column, self.branch,
                );
            }

            tracing::trace!(
                table = %self.table,
                branch = %self.branch,
                added = added.len(),
                removed = removed.len(),
                "IndexScan incremental results"
            );

            self.dirty = false;
            return TupleDelta {
                added: added
                    .into_iter()
                    .map(|id| Tuple::from_scoped_id(id, branch))
                    .collect(),
                removed: removed
                    .into_iter()
                    .map(|id| Tuple::from_scoped_id(id, branch))
                    .collect(),
                moved: vec![],
                updated: vec![],
            };
        }

        // Settle-cost accounting: one full index scan. Counted here rather than
        // inside `full_scan_ids` so the debug-only parity harness above, which
        // calls the same function purely to check the incremental path, does
        // not make debug builds report reads a release build never does.
        let new_ids = if self.window.is_some() || self.trigram.is_some() {
            let mut ids = if !self.declared_index_ready(ctx) {
                self.undeclared_scan_ids(ctx)
            } else if self.window.is_some() {
                self.window_scan_ids(ctx)
            } else {
                match self.trigram_scan_ids(ctx) {
                    Some(ids) => ids,
                    // No list could be read: the scope's rows, for the filter to narrow.
                    None => self.undeclared_scan_ids(ctx),
                }
            };
            self.apply_local_overlay_rows(ctx, &mut ids);
            ids
        } else {
            crate::query_manager::settle_cost::bump(
                &crate::query_manager::settle_cost::INDEX_READS,
            );
            self.full_scan_ids(ctx)
        };

        // Diff against last scan
        let added: Vec<ObjectId> = new_ids
            .difference(&self.last_scanned_ids)
            .copied()
            .collect();
        let removed: Vec<ObjectId> = self
            .last_scanned_ids
            .difference(&new_ids)
            .copied()
            .collect();

        tracing::trace!(
            table = %self.table,
            column = %self.column,
            branch = %self.branch,
            scanned = new_ids.len(),
            added = added.len(),
            removed = removed.len(),
            "IndexScan results"
        );

        self.last_scanned_ids = new_ids;
        let branch = BranchName::new(&self.branch);
        self.current_tuples = self
            .last_scanned_ids
            .iter()
            .map(|&id| Tuple::from_scoped_id(id, branch))
            .collect();
        self.dirty = false;
        self.has_scanned = true;
        self.needs_full = false;

        TupleDelta {
            added: added
                .into_iter()
                .map(|id| Tuple::from_scoped_id(id, branch))
                .collect(),
            removed: removed
                .into_iter()
                .map(|id| Tuple::from_scoped_id(id, branch))
                .collect(),
            moved: vec![],
            updated: vec![],
        }
    }

    fn current_tuples(&self) -> &AHashSet<Tuple> {
        &self.current_tuples
    }

    fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    fn is_dirty(&self) -> bool {
        self.dirty
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query_manager::types::{ColumnDescriptor, ColumnType, Value};
    use crate::storage::{MemoryStorage, Storage};
    use std::ops::Bound;

    fn make_ctx(storage: &dyn crate::storage::Storage) -> SourceContext<'_> {
        SourceContext {
            storage,
            local_overlay_rows: None,
        }
    }

    fn test_descriptor() -> RowDescriptor {
        RowDescriptor::new(vec![
            ColumnDescriptor::new("_id", ColumnType::Uuid),
            ColumnDescriptor::new("name", ColumnType::Text),
        ])
    }

    /// Helper to check if delta contains a tuple with given ID.
    fn contains_id(tuples: &[Tuple], id: ObjectId) -> bool {
        tuples.iter().any(|t| t.ids().contains(&id))
    }

    #[test]
    fn scan_all_returns_all_rows() {
        let mut storage = MemoryStorage::new();
        let row1 = ObjectId::new();
        let row2 = ObjectId::new();
        let row3 = ObjectId::new();

        storage
            .index_insert("users", "_id", "main", &Value::Uuid(row1), row1)
            .unwrap();
        storage
            .index_insert("users", "_id", "main", &Value::Uuid(row2), row2)
            .unwrap();
        storage
            .index_insert("users", "_id", "main", &Value::Uuid(row3), row3)
            .unwrap();

        let mut node = IndexScanNode::new("users", "_id", ScanCondition::All, test_descriptor());
        let ctx = make_ctx(&storage);
        let delta = node.scan(&ctx);

        assert_eq!(delta.added.len(), 3);
        assert!(contains_id(&delta.added, row1));
        assert!(contains_id(&delta.added, row2));
        assert!(contains_id(&delta.added, row3));
        assert!(delta.removed.is_empty());
    }

    #[test]
    fn scan_eq_returns_matching_rows() {
        let mut storage = MemoryStorage::new();
        let row1 = ObjectId::new();
        let row2 = ObjectId::new();

        storage
            .index_insert(
                "users",
                "email",
                "main",
                &Value::Text("alice@example.com".into()),
                row1,
            )
            .unwrap();
        storage
            .index_insert(
                "users",
                "email",
                "main",
                &Value::Text("bob@example.com".into()),
                row2,
            )
            .unwrap();

        let mut node = IndexScanNode::new(
            "users",
            "email",
            ScanCondition::Eq(Value::Text("alice@example.com".into())),
            test_descriptor(),
        );
        let ctx = make_ctx(&storage);
        let delta = node.scan(&ctx);

        assert_eq!(delta.added.len(), 1);
        assert!(contains_id(&delta.added, row1));
    }

    #[test]
    fn scan_range_returns_rows_in_range() {
        let mut storage = MemoryStorage::new();
        let row1 = ObjectId::new();
        let row2 = ObjectId::new();
        let row3 = ObjectId::new();

        storage
            .index_insert("users", "score", "main", &Value::Integer(10), row1)
            .unwrap();
        storage
            .index_insert("users", "score", "main", &Value::Integer(20), row2)
            .unwrap();
        storage
            .index_insert("users", "score", "main", &Value::Integer(30), row3)
            .unwrap();

        let mut node = IndexScanNode::new(
            "users",
            "score",
            ScanCondition::Range {
                min: Bound::Included(Value::Integer(15)),
                max: Bound::Included(Value::Integer(25)),
            },
            test_descriptor(),
        );
        let ctx = make_ctx(&storage);
        let delta = node.scan(&ctx);

        assert_eq!(delta.added.len(), 1);
        assert!(contains_id(&delta.added, row2));
    }

    #[test]
    fn rescan_detects_changes() {
        let mut storage = MemoryStorage::new();
        let row1 = ObjectId::new();
        let row2 = ObjectId::new();

        storage
            .index_insert("users", "_id", "main", &Value::Uuid(row1), row1)
            .unwrap();

        let mut node = IndexScanNode::new("users", "_id", ScanCondition::All, test_descriptor());
        let ctx = make_ctx(&storage);
        let delta1 = node.scan(&ctx);
        assert_eq!(delta1.added.len(), 1);
        assert!(contains_id(&delta1.added, row1));

        // Add another row
        storage
            .index_insert("users", "_id", "main", &Value::Uuid(row2), row2)
            .unwrap();

        let ctx = make_ctx(&storage);
        let delta2 = node.scan(&ctx);
        assert_eq!(delta2.added.len(), 1);
        assert!(contains_id(&delta2.added, row2));
        assert!(delta2.removed.is_empty());

        // Remove first row
        storage
            .index_remove("users", "_id", "main", &Value::Uuid(row1), row1)
            .unwrap();

        let ctx = make_ctx(&storage);
        let delta3 = node.scan(&ctx);
        assert!(delta3.added.is_empty());
        assert_eq!(delta3.removed.len(), 1);
        assert!(contains_id(&delta3.removed, row1));
    }

    #[test]
    fn output_descriptor_has_unmaterialized_state() {
        let desc = test_descriptor();
        let node = IndexScanNode::new("users", "_id", ScanCondition::All, desc);
        let output = node.output_tuple_descriptor();

        assert_eq!(output.element_count(), 1);
        assert!(!output.materialization().is_materialized(0));
    }
}
