//! A declared index's fill is background work. The runtime takes one step of it per
//! batched tick and none in the ticks its writes run, so a fill over a large table
//! holds the runtime a page at a time and puts no page in front of a write.
//!
//! Measured on a 100k-message store with a writer inserting every 50 ms: a fill that
//! stepped in every `process` (several per tick) of 1024 rows each, WAL-flushing every
//! step, held the writer's acks for the whole fill (up to 8 s); stepping once per tick
//! kept them at the idle level (p99 15 ms).

use super::*;
use crate::query_manager::declared_index::{FILL_PAGE, load_record};
use crate::query_manager::index_declarations::{IndexDeclarations, IndexPhase};

fn notes_schema() -> Schema {
    SchemaBuilder::new()
        .table(
            TableSchema::builder("notes")
                .column("owner", ColumnType::Uuid)
                .column("at", ColumnType::Timestamp)
                .column("body", ColumnType::Text),
        )
        .build()
}

fn notes_indexes() -> IndexDeclarations {
    IndexDeclarations::empty()
        .with_composite("notes", "owner", "at")
        .expect("composite declaration")
}

fn insert_note(core: &mut TestCore, owner: ObjectId, at: u64) {
    core.insert(
        "notes",
        HashMap::from([
            ("owner".to_string(), Value::Uuid(owner)),
            ("at".to_string(), Value::Timestamp(at)),
            ("body".to_string(), Value::Text(format!("note {at}"))),
        ]),
        None,
    )
    .expect("insert note");
}

fn phase(core: &TestCore) -> IndexPhase {
    load_record(&core.storage)
        .expect("record")
        .state("notes", "owner+at")
        .expect("declared")
        .phase
        .clone()
}

#[test]
fn a_runtime_takes_one_fill_step_per_batched_tick_and_none_in_a_write() {
    let mut core = create_runtime_with_schema(notes_schema(), "declared-index-pacing");
    let owner = ObjectId::new();
    let rows = 3 * FILL_PAGE as u64 + 5;
    for at in 0..rows {
        insert_note(&mut core, owner, at);
    }
    core.batched_tick();

    // A write's tick applies the declaration and takes no step of its work.
    core.schema_manager
        .query_manager_mut()
        .propose_index_declarations(notes_indexes());
    insert_note(&mut core, owner, rows);
    core.immediate_tick();
    assert_eq!(
        phase(&core),
        IndexPhase::Clearing { after: None },
        "a write's tick took a step of the fill"
    );

    // Each granted step moves the work on by one: the clear, then a page at a time.
    // Steps before the last need no WAL barrier of their own; the last one does.
    core.flush_wal().expect("flush the write's barrier");
    let mut ticks = 0;
    let mut cursors = Vec::new();
    loop {
        core.schema_manager
            .query_manager_mut()
            .grant_declared_index_step();
        core.immediate_tick();
        ticks += 1;
        let now = phase(&core);
        let done = now == IndexPhase::Complete;
        assert_eq!(
            core.has_storage_write_pending_flush(),
            done,
            "step {ticks} ({now:?}): only the step that ends the work asks for a flush"
        );
        core.flush_wal().expect("flush");
        if done {
            break;
        }
        match now {
            IndexPhase::Filling { after, .. } => cursors.push(after),
            other => panic!("step {ticks} left {other:?}"),
        }
        assert!(ticks < 20, "the fill never completed");
    }
    // One clear, then ceil(rows / FILL_PAGE) pages, the last of which completes it.
    let pages = (rows as usize + 1).div_ceil(FILL_PAGE);
    assert_eq!(ticks, 1 + pages, "cursors {cursors:?}");
    assert!(
        cursors.windows(2).all(|pair| pair[0] < pair[1]),
        "each step moved the cursor one page on: {cursors:?}"
    );
}

/// A pass that cannot read the record takes no step, keeps no grant for the next pass (a
/// write's) and asks for no tick of its own: it would be rescheduled for as long as the
/// record stays unreadable.
#[test]
fn a_pass_that_cannot_read_the_record_spends_its_grant_and_asks_for_no_tick() {
    let mut core = create_runtime_with_schema(notes_schema(), "declared-index-pacing");
    let owner = ObjectId::new();
    for at in 0..3 * FILL_PAGE as u64 {
        insert_note(&mut core, owner, at);
    }
    core.batched_tick();
    core.schema_manager
        .query_manager_mut()
        .propose_index_declarations(notes_indexes());
    core.immediate_tick();
    assert_eq!(phase(&core), IndexPhase::Clearing { after: None });

    let record = core
        .storage
        .raw_table_get("declared_indexes", "record")
        .expect("read the record")
        .expect("a record");
    core.storage
        .raw_table_put("declared_indexes", "record", &[0xff])
        .expect("garble the record");
    core.schema_manager
        .query_manager_mut()
        .grant_declared_index_step();
    core.immediate_tick();
    assert!(
        !core
            .schema_manager
            .query_manager()
            .has_declared_index_work(),
        "a pass that could not read the record still asks for ticks"
    );

    core.storage
        .raw_table_put("declared_indexes", "record", &record)
        .expect("restore the record");
    insert_note(&mut core, owner, 3 * FILL_PAGE as u64);
    core.immediate_tick();
    assert_eq!(
        phase(&core),
        IndexPhase::Clearing { after: None },
        "a write's tick took the step granted to the pass that could not read the record"
    );
    assert!(
        core.schema_manager
            .query_manager()
            .has_declared_index_work(),
        "the work resumes once the record reads again"
    );
}

#[test]
fn a_batched_tick_takes_one_fill_step() {
    let mut core = create_runtime_with_schema(notes_schema(), "declared-index-pacing");
    let owner = ObjectId::new();
    for at in 0..3 * FILL_PAGE as u64 {
        insert_note(&mut core, owner, at);
    }
    core.batched_tick();
    core.schema_manager
        .query_manager_mut()
        .propose_index_declarations(notes_indexes());
    core.immediate_tick();

    core.batched_tick();
    assert!(
        matches!(phase(&core), IndexPhase::Filling { after: None, .. }),
        "one batched tick took more than the clear: {:?}",
        phase(&core)
    );
    core.batched_tick();
    assert!(
        matches!(phase(&core), IndexPhase::Filling { after: Some(_), .. }),
        "the next batched tick took more than one page: {:?}",
        phase(&core)
    );
}
