//! Defect-27 gate: a row whose family spans two schema generations must READ as the
//! generation the writes are landing in.
//!
//! Measured live (production 2026-08-16, sync server, engine v16.15). A policied `users`
//! presence row exists in two generations. The server's raw tables and both pointers,
//! read directly off a copy of the running store:
//!
//! ```text
//! rowtable:visible:users:53710882d8e0…  this_row=[dev-53710882-main, dev-b32dae47-main]
//! rowtable:visible:users:b32dae47bbd9…  this_row=[dev-b32dae47-main]
//! __row_locator                         -> 53710882…  (the FOSSIL generation)
//! __visible_row_table_locator
//!   branch dev-53710882-main            -> None
//!   branch dev-b32dae47-main            -> b32dae47…  (PRESENT, and CORRECT)
//! ```
//!
//! Every 10s heartbeat IS applied — into the CURRENT generation's raw table — while every
//! reader (query layer, inspector) serves the OLD generation's copy, frozen at the last
//! server restart. `load_visible_region_row_bytes_with_storage` consults the locator
//! DERIVED from `__row_locator.origin_schema_hash` first and returns on a hit; the exact
//! per-(branch,row) visible locator the writer keeps current, and the defect-20 sibling
//! scan, are only reached on a MISS. The stale sibling hits, so neither ever runs.
//!
//! Two details from the measurement worth carrying: the authoritative locator SURVIVED the
//! damage and names the live family, so inverting the ladder's precedence is what makes
//! this row read correctly again — the sweep then removes the duplicate head. And the
//! fossil family holds heads for TWO branches while the live family holds only the current
//! one, so the split is per `(row, branch)`: `dev-b32dae47-main` has two heads and
//! `dev-53710882-main` has one, which is why every repair here reasons per branch key
//! rather than per row.

use super::*;
use crate::storage::SqliteStorage;

/// A backend that actually models the `rowtable:*:<table>:<schema-hash>` raw-table
/// families. `MemoryStorage` keeps visible entries as structs and overrides
/// `load_visible_region_row`/`_entry` (`storage/memory.rs:899`, `:929`), so it never runs
/// the locator ladder in `load_visible_region_row_bytes_with_storage` — the code under
/// test. Every real deployment (sqlite on the client, rocksdb on the server) does.
fn split_capable_storage() -> SqliteStorage {
    SqliteStorage::open(":memory:").expect("in-memory sqlite storage should open")
}

/// Generation B of `users`: the same logical table after a column was added, so it hashes
/// to a different schema and lands in a different `rowtable:*:users:<hash>` family.
pub(super) fn users_next_generation_schema() -> crate::query_manager::types::Schema {
    SchemaBuilder::new()
        .table(
            TableSchema::builder("users")
                .column("value", ColumnType::Text)
                .column("presence", ColumnType::Text),
        )
        .build()
}

pub(super) fn next_generation_schema_hash() -> SchemaHash {
    SchemaHash::compute(&users_next_generation_schema())
}

pub(super) fn next_generation_metadata() -> HashMap<String, String> {
    HashMap::from([
        (MetadataKey::Table.to_string(), "users".to_string()),
        (
            MetadataKey::OriginSchemaHash.to_string(),
            next_generation_schema_hash().to_string(),
        ),
    ])
}

/// A row encoded under generation B's descriptor — what the client sends after the
/// schema deployment.
pub(super) fn next_generation_row(
    row_id: ObjectId,
    parents: Vec<BatchId>,
    updated_at: u64,
    value: &str,
    presence: &str,
) -> StoredRowBatch {
    StoredRowBatch::new(
        row_id,
        "main",
        parents,
        encode_row(
            &users_next_generation_schema()[&"users".into()].columns,
            &[
                Value::Text(value.to_string()),
                Value::Text(presence.to_string()),
            ],
        )
        .expect("generation-B row should encode"),
        RowProvenance::for_insert(row_id.to_string(), updated_at),
        HashMap::new(),
        crate::row_histories::RowState::VisibleDirect,
        None,
    )
}

pub(super) fn send_and_approve<H: Storage>(
    sm: &mut SyncManager,
    io: &mut H,
    client_id: ClientId,
    metadata: HashMap<String, String>,
    row: &StoredRowBatch,
) {
    sm.push_inbox(InboxEntry {
        source: Source::Client(client_id),
        payload: SyncPayload::RowBatchCreated {
            metadata: Some(RowMetadata {
                id: row.row_id,
                metadata,
            }),
            row: row.clone(),
        },
    });
    sm.process_inbox(io);
    for check in sm.take_pending_permission_checks() {
        sm.approve_permission_check(io, check);
    }
}

fn visible_raw_tables_for_users<H: Storage>(io: &H) -> Vec<String> {
    io.scan_raw_table_headers()
        .expect("raw table header scan should succeed")
        .into_iter()
        .map(|(name, _)| name)
        .filter(|name| name.starts_with("rowtable:visible:users:"))
        .collect()
}

/// The gate. A policied row seeded under generation A receives an update authored under
/// generation B. The visible read must serve generation B's value.
#[test]
fn a_cross_generation_update_is_what_the_visible_read_serves() {
    let mut io = split_capable_storage();
    crate::test_support::persist_test_schema(&mut io, &users_test_schema());
    crate::test_support::persist_test_schema(&mut io, &users_next_generation_schema());

    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::GlobalServer);
    let client_id = ClientId::new();
    sm.add_client_with_storage(&io, client_id);
    sm.set_client_acks_deliveries(client_id, true);
    sm.set_client_role(client_id, ClientRole::User);
    sm.set_client_session(
        client_id,
        crate::query_manager::session::Session::new("alice"),
    );
    sm.take_outbox();

    let row_id = ObjectId::new();

    // Generation A: the row as it existed before the schema deployment.
    let old_generation = visible_row(row_id, "main", Vec::new(), 1_000, b"before");
    send_and_approve(
        &mut sm,
        &mut io,
        client_id,
        row_metadata("users"),
        &old_generation,
    );

    let seeded = io
        .load_visible_region_row("users", "main", row_id)
        .expect("visible read should succeed")
        .expect("the seeded row must be visible");
    assert_eq!(
        seeded.batch_id(),
        old_generation.batch_id,
        "control: the generation-A seed must be what the read serves before the deployment"
    );

    // Generation B: the heartbeat after the deployment, parented on the old row.
    let new_generation = next_generation_row(
        row_id,
        vec![old_generation.batch_id],
        2_000,
        "after",
        "online",
    );
    send_and_approve(
        &mut sm,
        &mut io,
        client_id,
        next_generation_metadata(),
        &new_generation,
    );

    let served = io
        .load_visible_region_row("users", "main", row_id)
        .expect("visible read should succeed")
        .expect("the row must still be visible after the cross-generation update");

    assert_eq!(
        served.batch_id(),
        new_generation.batch_id,
        "the visible read must serve the cross-generation update, not the older \
         generation's sibling copy; visible raw tables present: {:?}",
        visible_raw_tables_for_users(&io)
    );
}

/// The same split seen from the other side: no `(row, branch)` may hold a visible entry in
/// more than one generation's raw table. Two heads for one row is the state the reader
/// then has to guess between — and it guesses by a locator nothing keeps current.
#[test]
fn one_row_and_branch_has_exactly_one_visible_head_across_generations() {
    let mut io = split_capable_storage();
    crate::test_support::persist_test_schema(&mut io, &users_test_schema());
    crate::test_support::persist_test_schema(&mut io, &users_next_generation_schema());

    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::GlobalServer);
    let client_id = ClientId::new();
    sm.add_client_with_storage(&io, client_id);
    sm.set_client_acks_deliveries(client_id, true);
    sm.set_client_role(client_id, ClientRole::User);
    sm.set_client_session(
        client_id,
        crate::query_manager::session::Session::new("alice"),
    );
    sm.take_outbox();

    let row_id = ObjectId::new();
    let old_generation = visible_row(row_id, "main", Vec::new(), 1_000, b"before");
    send_and_approve(
        &mut sm,
        &mut io,
        client_id,
        row_metadata("users"),
        &old_generation,
    );
    let new_generation = next_generation_row(
        row_id,
        vec![old_generation.batch_id],
        2_000,
        "after",
        "online",
    );
    send_and_approve(
        &mut sm,
        &mut io,
        client_id,
        next_generation_metadata(),
        &new_generation,
    );

    // `<branch>:<row-uuid-hex>` is the visible raw-table key layout
    // (`storage::key_codec::visible_row_raw_table_key`).
    let key = format!("main:{}", row_id.uuid().simple());
    let heads: Vec<String> = visible_raw_tables_for_users(&io)
        .into_iter()
        .filter(|raw_table| {
            io.raw_table_get(raw_table, &key)
                .expect("raw table probe should succeed")
                .is_some()
        })
        .collect();

    assert_eq!(
        heads.len(),
        1,
        "one (row, branch) must have exactly one visible head; it has {}: {heads:?}",
        heads.len()
    );
}

/// Which visible families physically hold `(branch, row)`.
fn visible_heads_for<H: Storage>(io: &H, branch: &str, row_id: ObjectId) -> Vec<String> {
    let key = format!("{branch}:{}", row_id.uuid().simple());
    visible_raw_tables_for_users(io)
        .into_iter()
        .filter(|raw_table| {
            io.raw_table_get(raw_table, &key)
                .expect("raw table probe should succeed")
                .is_some()
        })
        .collect()
}

/// Re-create the production damage as MEASURED, not as assumed.
///
/// Probed from a copy of the live sync server (2026-08-16, `users` row
/// `c24432b4…`, via `live_incident_store_pointers`):
///
/// ```text
/// rowtable:visible:users:53710882…  this_row=[dev-53710882-main:…, dev-b32dae47-main:…]
/// rowtable:visible:users:b32dae47…  this_row=[dev-b32dae47-main:…]
/// __row_locator                     -> 53710882…   (the FOSSIL generation)
/// __visible_row_table_locator
///   branch dev-53710882-main        -> None
///   branch dev-b32dae47-main        -> b32dae47…   (PRESENT, and CORRECT)
/// ```
///
/// The authoritative locator is present and names the live family: the inbound
/// cross-generation write stamps it (`storage::apply_encoded_row_mutation`
/// whenever `needs_exact_locator`) and nothing ever un-stamps it. An earlier
/// version of this helper CLEARED it, which modelled a harsher store than the
/// one we have and quietly overstated what the sweep is for. It matters which:
/// with the authoritative pointer intact, the read-ladder inversion alone makes
/// this row read correctly at the first restart, and the sweep's job is to
/// remove the duplicate head rather than to fix the read.
///
/// (The state with NO authoritative locator is still covered — the adversarial
/// gates in `adversarial_cross_generation.rs` build it deliberately — it is just
/// not what production holds.)
fn reseed_the_measured_production_split<H: Storage>(
    io: &mut H,
    branch: &str,
    row_id: ObjectId,
    stale_family: &str,
    stale_bytes: &[u8],
    stale_schema_hash: SchemaHash,
    live_schema_hash: SchemaHash,
) {
    let key = format!("{branch}:{}", row_id.uuid().simple());
    io.raw_table_put(stale_family, &key, stale_bytes)
        .expect("re-seeding the stale head should succeed");
    io.put_row_locator(
        row_id,
        Some(&crate::storage::RowLocator {
            table: "users".into(),
            origin_schema_hash: Some(stale_schema_hash),
        }),
    )
    .expect("rewinding the row locator should succeed");
    io.put_visible_row_table_locator(
        branch,
        row_id,
        Some(&crate::storage::visible_row_table_locator_for(
            "users",
            live_schema_hash,
        )),
    )
    .expect("stamping the authoritative locator at the live family should succeed");
}

/// The repair gate. A store that is ALREADY split — the measured shape of the
/// incident server — must converge to one head, with the newest content winning,
/// without a wipe and without a manual step.
///
/// Note what this gate does NOT claim any more. On the measured store the
/// authoritative locator is present and correct, so the read-ladder inversion
/// already serves the right version before the sweep runs; the assertion below
/// pins exactly that, because it is the thing that makes the incident heal at
/// the first restart. What the sweep adds is STRUCTURAL: the duplicate head is
/// what makes the scan surface return the row twice, what a delete has to reach,
/// and what keeps the fossil's index entries alive. Removing it is the fix's
/// cleanup half, not its read half.
#[test]
fn the_sweep_heals_a_store_that_is_already_split() {
    let mut io = split_capable_storage();
    crate::test_support::persist_test_schema(&mut io, &users_test_schema());
    crate::test_support::persist_test_schema(&mut io, &users_next_generation_schema());

    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::GlobalServer);
    let client_id = ClientId::new();
    sm.add_client_with_storage(&io, client_id);
    sm.set_client_acks_deliveries(client_id, true);
    sm.set_client_role(client_id, ClientRole::User);
    sm.set_client_session(
        client_id,
        crate::query_manager::session::Session::new("alice"),
    );
    sm.take_outbox();

    let row_id = ObjectId::new();
    let old_generation = visible_row(row_id, "main", Vec::new(), 1_000, b"before");
    send_and_approve(
        &mut sm,
        &mut io,
        client_id,
        row_metadata("users"),
        &old_generation,
    );

    // Snapshot generation A's head before the crossing moves it.
    let key = format!("main:{}", row_id.uuid().simple());
    let stale_family = visible_heads_for(&io, "main", row_id)
        .pop()
        .expect("the generation-A head must exist before the crossing");
    let stale_bytes = io
        .raw_table_get(&stale_family, &key)
        .expect("raw table probe should succeed")
        .expect("the generation-A head must exist before the crossing");

    let new_generation = next_generation_row(
        row_id,
        vec![old_generation.batch_id],
        2_000,
        "after",
        "online",
    );
    send_and_approve(
        &mut sm,
        &mut io,
        client_id,
        next_generation_metadata(),
        &new_generation,
    );

    reseed_the_measured_production_split(
        &mut io,
        "main",
        row_id,
        &stale_family,
        &stale_bytes,
        users_schema_hash(),
        next_generation_schema_hash(),
    );

    // Precondition, so the gate cannot pass vacuously: the store really is split
    // and really is serving the fossil.
    assert_eq!(
        visible_heads_for(&io, "main", row_id).len(),
        2,
        "the re-seeded store must actually be split before the sweep runs"
    );
    // The measured store's reads are ALREADY correct at this point: the
    // authoritative locator survived the damage, and the ladder consults it
    // first. This is the whole reason the incident heals on restart rather than
    // needing the sweep to finish.
    assert_eq!(
        io.load_visible_region_row("users", "main", row_id)
            .expect("visible read should succeed")
            .expect("the row must be visible")
            .batch_id(),
        new_generation.batch_id,
        "on the measured production shape the authoritative locator is intact, so the \
         read must already serve the live generation before the sweep — if this ever \
         reads as the fossil, the ladder inversion has regressed and the sweep is \
         carrying the incident on its own"
    );

    let report = crate::storage::repair_split_visible_row_families(&mut io, "users")
        .expect("the sweep should succeed");
    assert_eq!(
        (
            report.split_rows,
            report.dropped_heads,
            report.rebuilt_rows,
            report.unresolved_rows
        ),
        (1, 1, 1, 0),
        "the sweep must report exactly what it touched: {report:?}"
    );

    assert_eq!(
        visible_heads_for(&io, "main", row_id).len(),
        1,
        "the sweep must leave exactly one visible head"
    );
    assert_eq!(
        io.load_visible_region_row("users", "main", row_id)
            .expect("visible read should succeed")
            .expect("the row must still be visible after the sweep")
            .batch_id(),
        new_generation.batch_id,
        "the sweep must leave the NEWEST version as the surviving head"
    );
    assert_eq!(
        io.load_row_locator(row_id)
            .expect("row locator read should succeed")
            .and_then(|locator| locator.origin_schema_hash),
        Some(next_generation_schema_hash()),
        "the sweep must stamp the row locator at the surviving head's generation \
         — this is what `old_content_schema_hash` is read from \
         (sync_manager::inbox, query_manager::server_queries)"
    );

    // Idempotent: a second pass over the now-healthy store must do nothing.
    let second = crate::storage::repair_split_visible_row_families(&mut io, "users")
        .expect("the second sweep should succeed");
    assert!(
        second.is_noop(),
        "the sweep must be a no-op on a healthy store: {second:?}"
    );
}

/// The other half of the repair gate: a store that was never split must not be
/// touched, and the sweep must not need a second generation to exist to be safe.
#[test]
fn the_sweep_is_a_no_op_on_a_healthy_store() {
    let mut io = split_capable_storage();
    crate::test_support::persist_test_schema(&mut io, &users_test_schema());
    crate::test_support::persist_test_schema(&mut io, &users_next_generation_schema());

    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::GlobalServer);
    let client_id = ClientId::new();
    sm.add_client_with_storage(&io, client_id);
    sm.set_client_acks_deliveries(client_id, true);
    sm.set_client_role(client_id, ClientRole::User);
    sm.set_client_session(
        client_id,
        crate::query_manager::session::Session::new("alice"),
    );
    sm.take_outbox();

    let row_id = ObjectId::new();
    let old_generation = visible_row(row_id, "main", Vec::new(), 1_000, b"before");
    send_and_approve(
        &mut sm,
        &mut io,
        client_id,
        row_metadata("users"),
        &old_generation,
    );
    let new_generation = next_generation_row(
        row_id,
        vec![old_generation.batch_id],
        2_000,
        "after",
        "online",
    );
    send_and_approve(
        &mut sm,
        &mut io,
        client_id,
        next_generation_metadata(),
        &new_generation,
    );

    let served_before = io
        .load_visible_region_row("users", "main", row_id)
        .expect("visible read should succeed");
    let report = crate::storage::repair_split_visible_row_families(&mut io, "users")
        .expect("the sweep should succeed");
    assert!(
        report.is_noop(),
        "a store the write path kept healthy must give the sweep nothing to do: {report:?}"
    );
    assert_eq!(
        io.load_visible_region_row("users", "main", row_id)
            .expect("visible read should succeed")
            .map(|row| row.batch_id()),
        served_before.map(|row| row.batch_id()),
        "the sweep must not move a head it did not need to move"
    );
}

/// The live store's exact pointer state, which neither of the gates above
/// reproduces.
///
/// On the incident server the inbound cross-generation write DID stamp the
/// authoritative `__visible_row_table_locator` at the family it wrote
/// (`storage::apply_encoded_row_mutation` does that whenever
/// `needs_exact_locator`), while `__row_locator` stayed on the generation the
/// row was born in. So the store held a CORRECT pointer and a STALE one, and the
/// read ladder consulted the stale one first and returned on its hit — which is
/// why the server's logs showed zero locator alignments and zero defect-20
/// sibling-scan hits: step one always hit.
///
/// This is the gate for the ladder's precedence on its own: no sweep, no
/// re-write, just the two pointers disagreeing the way production had them.
#[test]
fn a_stale_derived_locator_does_not_outrank_the_locator_writes_keep_current() {
    let mut io = split_capable_storage();
    crate::test_support::persist_test_schema(&mut io, &users_test_schema());
    crate::test_support::persist_test_schema(&mut io, &users_next_generation_schema());

    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::GlobalServer);
    let client_id = ClientId::new();
    sm.add_client_with_storage(&io, client_id);
    sm.set_client_acks_deliveries(client_id, true);
    sm.set_client_role(client_id, ClientRole::User);
    sm.set_client_session(
        client_id,
        crate::query_manager::session::Session::new("alice"),
    );
    sm.take_outbox();

    let row_id = ObjectId::new();
    let old_generation = visible_row(row_id, "main", Vec::new(), 1_000, b"before");
    send_and_approve(
        &mut sm,
        &mut io,
        client_id,
        row_metadata("users"),
        &old_generation,
    );

    let key = format!("main:{}", row_id.uuid().simple());
    let stale_family = visible_heads_for(&io, "main", row_id)
        .pop()
        .expect("the generation-A head must exist before the crossing");
    let stale_bytes = io
        .raw_table_get(&stale_family, &key)
        .expect("raw table probe should succeed")
        .expect("the generation-A head must exist before the crossing");

    let new_generation = next_generation_row(
        row_id,
        vec![old_generation.batch_id],
        2_000,
        "after",
        "online",
    );
    send_and_approve(
        &mut sm,
        &mut io,
        client_id,
        next_generation_metadata(),
        &new_generation,
    );

    // Put the store back into the production shape: the fossil head restored,
    // `__row_locator` rewound to generation A — and the authoritative locator
    // LEFT ALONE, still naming generation B, because that is what the inbound
    // write stamped and nothing ever un-stamped.
    io.raw_table_put(&stale_family, &key, &stale_bytes)
        .expect("re-seeding the stale head should succeed");
    io.put_row_locator(
        row_id,
        Some(&crate::storage::RowLocator {
            table: "users".into(),
            origin_schema_hash: Some(users_schema_hash()),
        }),
    )
    .expect("rewinding the row locator should succeed");

    // Precondition: the two pointers really do disagree, and the authoritative
    // one really is the correct one.
    assert_eq!(
        io.load_visible_row_table_locator("main", row_id)
            .expect("exact visible locator read should succeed")
            .map(|locator| locator.schema_hash),
        Some(next_generation_schema_hash()),
        "the inbound cross-generation write must have stamped the authoritative locator \
         at generation B, or this gate is not reproducing the live store"
    );
    assert_eq!(
        io.load_row_locator(row_id)
            .expect("row locator read should succeed")
            .and_then(|locator| locator.origin_schema_hash),
        Some(users_schema_hash()),
        "the derived locator must be the stale one, or this gate is not reproducing \
         the live store"
    );

    assert_eq!(
        io.load_visible_region_row("users", "main", row_id)
            .expect("visible read should succeed")
            .expect("the row must be visible")
            .batch_id(),
        new_generation.batch_id,
        "with the two locators disagreeing, the read must follow the one the write path \
         keeps current, not the one nothing updates"
    );
}

/// What the defect-27 write chokepoint costs on the COMMON path: a
/// single-generation store, where it can only ever measure the families once
/// and find nothing to move. Run:
///   cargo test -p jazz-tools --features "sqlite test-utils" --release \
///     --lib visible_write_chokepoint_cost -- --ignored --nocapture
#[test]
#[ignore = "measurement, not a gate"]
fn visible_write_chokepoint_cost() {
    const WRITES: usize = 2_000;

    let mut io = split_capable_storage();
    crate::test_support::persist_test_schema(&mut io, &users_test_schema());

    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::GlobalServer);
    let client_id = ClientId::new();
    sm.add_client_with_storage(&io, client_id);
    sm.set_client_acks_deliveries(client_id, true);
    sm.set_client_role(client_id, ClientRole::User);
    sm.set_client_session(
        client_id,
        crate::query_manager::session::Session::new("alice"),
    );
    sm.take_outbox();

    let whole_write = std::time::Instant::now();
    for index in 0..WRITES {
        let row = visible_row(
            ObjectId::new(),
            "main",
            Vec::new(),
            1_000 + index as u64,
            format!("row-{index}").as_bytes(),
        );
        send_and_approve(&mut sm, &mut io, client_id, row_metadata("users"), &row);
    }
    let whole_write = whole_write.elapsed();

    // The chokepoint's own probe: the header prefix scan that decides whether
    // more than one family exists for this table.
    let probe = std::time::Instant::now();
    for _ in 0..WRITES {
        let _ = std::hint::black_box(crate::storage::visible_row_families_holding(
            &io,
            "users",
            "main",
            ObjectId::new(),
        ));
    }
    let probe = probe.elapsed();

    let whole_ns = whole_write.as_nanos() as f64 / WRITES as f64;
    let probe_ns = probe.as_nanos() as f64 / WRITES as f64;
    println!(
        "visible write path: whole write {whole_ns:.0} ns/op, chokepoint family probe \
         {probe_ns:.0} ns/op ({:.1}% of the write it was added to)",
        100.0 * probe_ns / whole_ns
    );
}

/// The cost of the read-ladder inversion, measured rather than argued.
///
/// Making the authoritative `__visible_row_table_locator` the FIRST step means
/// the common case — a row that never crossed a generation, so no exact locator
/// was ever recorded — pays one extra point read that misses before the derived
/// locator hits. This measures that extra read against the whole visible read it
/// was added to.
///
/// `cargo test -p jazz-tools --features "rocksdb sqlite test-utils" --lib \
///  visible_read_ladder_cost -- --ignored --nocapture`
#[test]
#[ignore = "measurement, not a gate"]
fn visible_read_ladder_cost() {
    const ROWS: usize = 2_000;
    const READS: usize = 20_000;

    let mut io = split_capable_storage();
    crate::test_support::persist_test_schema(&mut io, &users_test_schema());

    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::GlobalServer);
    let client_id = ClientId::new();
    sm.add_client_with_storage(&io, client_id);
    sm.set_client_acks_deliveries(client_id, true);
    sm.set_client_role(client_id, ClientRole::User);
    sm.set_client_session(
        client_id,
        crate::query_manager::session::Session::new("alice"),
    );
    sm.take_outbox();

    let row_ids: Vec<ObjectId> = (0..ROWS)
        .map(|index| {
            let row_id = ObjectId::new();
            let row = visible_row(
                row_id,
                "main",
                Vec::new(),
                1_000 + index as u64,
                format!("row-{index}").as_bytes(),
            );
            send_and_approve(&mut sm, &mut io, client_id, row_metadata("users"), &row);
            row_id
        })
        .collect();

    // Precondition: this is the COMMON case the reorder taxes — none of these
    // rows ever crossed a generation, so the added first step always misses.
    assert!(
        row_ids.iter().all(|row_id| io
            .load_visible_row_table_locator("main", *row_id)
            .expect("exact visible locator read should succeed")
            .is_none()),
        "the measurement must be over rows with NO exact visible locator, or it \
         measures the wrong case"
    );

    let whole_read = std::time::Instant::now();
    for index in 0..READS {
        let row_id = row_ids[index % ROWS];
        std::hint::black_box(
            io.load_visible_region_row("users", "main", row_id)
                .expect("visible read should succeed"),
        );
    }
    let whole_read = whole_read.elapsed();

    let added_step = std::time::Instant::now();
    for index in 0..READS {
        let row_id = row_ids[index % ROWS];
        std::hint::black_box(
            io.load_visible_row_table_locator("main", row_id)
                .expect("exact visible locator read should succeed"),
        );
    }
    let added_step = added_step.elapsed();

    let whole_ns = whole_read.as_nanos() as f64 / READS as f64;
    let added_ns = added_step.as_nanos() as f64 / READS as f64;
    println!(
        "visible read ladder: whole read {whole_ns:.0} ns/op, added authoritative-locator \
         step {added_ns:.0} ns/op ({:.1}% of the read it was added to)",
        100.0 * added_ns / whole_ns
    );
}

// ---------------------------------------------------------------------------
// The index surface. Index raw tables are `idx:<table>:<column>:<branch>` with
// NO schema hash in the name (`key_codec::index_raw_table`), so both generations
// write into the SAME index. Moving a head between families therefore leaves the
// dropped head's values indexed with nothing behind them.
//
// Three consumers trust the index without re-reading the row, so a stale entry
// is not a slow path, it is a wrong answer:
//   * a fully-covered indexed predicate drops its residual filter entirely
//     (`graph/compile.rs::build_remaining_predicate_from_disjuncts` returns
//     `Predicate::True`) — a phantom query RESULT;
//   * `row_is_indexed_on_branch` / `row_is_deleted_on_branch` are pure index
//     reads (`query_manager/writes.rs`);
//   * REBAC edge traversal takes `index_lookup` directly when the referencing
//     column is an indexed scalar `Uuid`, and only the NON-indexed fallback
//     re-verifies with `declared_edge_references_target` — so a stale entry
//     GRANTS ACCESS across an edge that no longer exists.
// ---------------------------------------------------------------------------

/// `docs` generation A: an owner edge (indexed scalar `Uuid` — the REBAC shape),
/// an array-of-uuid reference (the one column that owns N+1 index entries), a
/// double (the signed-zero split `index_remove` does not cover) and a `Bytea`
/// (never indexed).
fn docs_edge_schema() -> crate::query_manager::types::Schema {
    SchemaBuilder::new()
        .table(
            TableSchema::builder("docs")
                .column("title", ColumnType::Text)
                .fk_column("owner", "users")
                .array_fk_column("editors", "users")
                .column("score", ColumnType::Double)
                .column("blob", ColumnType::Bytea),
        )
        .build()
}

/// Generation B: identical `docs`, one more table, so the schema hash — and the
/// family — differ while the row's own bytes stay comparable.
fn docs_edge_next_generation_schema() -> crate::query_manager::types::Schema {
    SchemaBuilder::new()
        .table(
            TableSchema::builder("docs")
                .column("title", ColumnType::Text)
                .fk_column("owner", "users")
                .array_fk_column("editors", "users")
                .column("score", ColumnType::Double)
                .column("blob", ColumnType::Bytea),
        )
        .table(TableSchema::builder("tags").column("label", ColumnType::Text))
        .build()
}

struct EdgeRow {
    owner: ObjectId,
    editors: Vec<ObjectId>,
    score: f64,
    bytes: Vec<u8>,
}

fn encode_edge_row(
    schema: &crate::query_manager::types::Schema,
    title: &str,
    owner: ObjectId,
    editors: &[ObjectId],
    score: f64,
) -> EdgeRow {
    let columns = &schema[&"docs".into()].columns;
    let values = vec![
        Value::Text(title.to_string()),
        Value::Uuid(owner),
        Value::Array(editors.iter().copied().map(Value::Uuid).collect()),
        Value::Double(score),
        Value::Bytea(vec![7, 7, 7]),
    ];
    EdgeRow {
        owner,
        editors: editors.to_vec(),
        score,
        bytes: crate::query_manager::encoding::encode_row(columns, &values)
            .expect("docs row should encode"),
    }
}

/// Write the index entries a row owns, the way the write path does.
fn index_the_row<H: Storage>(io: &mut H, branch: &str, row_id: ObjectId, row: &EdgeRow) {
    io.index_insert("docs", "_id", branch, &Value::Uuid(row_id), row_id)
        .expect("_id index insert");
    io.index_insert("docs", "owner", branch, &Value::Uuid(row.owner), row_id)
        .expect("owner index insert");
    io.index_insert(
        "docs",
        "editors",
        branch,
        &Value::Array(row.editors.iter().copied().map(Value::Uuid).collect()),
        row_id,
    )
    .expect("editors array index insert");
    for editor in &row.editors {
        io.index_insert("docs", "editors", branch, &Value::Uuid(*editor), row_id)
            .expect("editors element index insert");
    }
    io.index_insert("docs", "score", branch, &Value::Double(row.score), row_id)
        .expect("score index insert");
}

/// A store where `(docs, main, row)` has a head in BOTH generations, and both
/// heads' index entries are present — the state a cross-generation write leaves
/// before the head move runs.
fn split_docs_store() -> (SqliteStorage, ObjectId, ObjectId, SchemaHash, SchemaHash) {
    let mut io = split_capable_storage();
    let schema_a = docs_edge_schema();
    let schema_b = docs_edge_next_generation_schema();
    crate::test_support::persist_test_schema(&mut io, &schema_a);
    crate::test_support::persist_test_schema(&mut io, &schema_b);
    let hash_a = SchemaHash::compute(&schema_a);
    let hash_b = SchemaHash::compute(&schema_b);

    let row_id = ObjectId::new();
    let old_owner = ObjectId::new();
    let new_owner = ObjectId::new();
    let old_editor = ObjectId::new();
    let shared_editor = ObjectId::new();

    let stale = encode_edge_row(
        &schema_a,
        "before",
        old_owner,
        &[old_editor, shared_editor],
        0.0,
    );
    let live = encode_edge_row(&schema_b, "after", new_owner, &[shared_editor], 1.5);

    let key = format!("main:{}", row_id.uuid().simple());
    io.raw_table_put(
        &format!("rowtable:visible:docs:{hash_a}"),
        &key,
        &stale.bytes,
    )
    .expect("stale head write");
    io.raw_table_put(
        &format!("rowtable:visible:docs:{hash_b}"),
        &key,
        &live.bytes,
    )
    .expect("live head write");

    index_the_row(&mut io, "main", row_id, &stale);
    index_the_row(&mut io, "main", row_id, &live);

    (io, row_id, old_owner, hash_a, hash_b)
}

/// The REBAC form, which is the one that matters: an edge that no longer exists
/// must not be traversable. `evaluate_referencing_inherited_access_recursive`
/// takes `index_lookup` straight for an indexed scalar `Uuid` column and does
/// NOT re-read the row, so an index entry left pointing at a dropped head is an
/// authorization grant.
#[test]
fn a_moved_head_does_not_leave_its_edges_traversable_in_the_index() {
    let (mut io, row_id, old_owner, hash_a, hash_b) = split_docs_store();

    // Precondition: the stale edge really is traversable before the move, or the
    // gate proves nothing.
    assert!(
        io.index_lookup("docs", "owner", "main", &Value::Uuid(old_owner))
            .contains(&row_id),
        "the stale owner edge must be indexed before the head move"
    );

    let moved = crate::storage::drop_stale_visible_row_family_entry(
        &mut io,
        "docs",
        "main",
        row_id,
        hash_a,
        Some(hash_b),
    )
    .expect("the head move should succeed");
    assert!(moved, "the stale head must actually have been dropped");

    assert!(
        !io.index_lookup("docs", "owner", "main", &Value::Uuid(old_owner))
            .contains(&row_id),
        "the dropped head's owner edge is still in the index; REBAC's indexed \
         branch calls `index_lookup` and never re-reads the row, so this grants \
         access across an edge that no longer exists"
    );
}

/// The rest of the retirement contract in one pass: what must go, what must
/// STAY, and the three encoding rules the storage-layer helper has to mirror
/// from `query_manager/indices.rs` (array-of-uuid references own one entry per
/// element as well as the whole-array entry; `Bytea` is never indexed; a signed
/// zero encodes to two different segments).
#[test]
fn a_moved_head_retires_its_own_index_entries_and_only_its_own() {
    let (mut io, row_id, old_owner, hash_a, hash_b) = split_docs_store();

    // Re-derive the values the two heads were built from.
    let key = format!("main:{}", row_id.uuid().simple());
    let live_bytes = io
        .raw_table_get(&format!("rowtable:visible:docs:{hash_b}"), &key)
        .expect("probe")
        .expect("live head");
    let schema_b = docs_edge_next_generation_schema();
    let live_values =
        crate::query_manager::encoding::decode_row(&schema_b[&"docs".into()].columns, &live_bytes)
            .expect("live row decodes");
    let Value::Uuid(live_owner) = live_values[1] else {
        panic!("owner column must decode as a uuid");
    };
    let Value::Array(live_editors) = live_values[2].clone() else {
        panic!("editors column must decode as an array");
    };

    crate::storage::drop_stale_visible_row_family_entry(
        &mut io,
        "docs",
        "main",
        row_id,
        hash_a,
        Some(hash_b),
    )
    .expect("the head move should succeed");

    // GONE: values only the dropped head had.
    assert!(
        !io.index_lookup("docs", "owner", "main", &Value::Uuid(old_owner))
            .contains(&row_id),
        "the dropped head's owner value must be retired"
    );
    assert!(
        !io.index_lookup("docs", "score", "main", &Value::Double(0.0))
            .contains(&row_id),
        "the dropped head's score must be retired"
    );
    assert!(
        !io.index_lookup("docs", "score", "main", &Value::Double(-0.0))
            .contains(&row_id),
        "`Value::Double(0.0)` and `-0.0` encode to different index segments and \
         the lookup path probes both, so the retirement must remove both"
    );

    // KEPT: the surviving head's own values, and the implicit `_id` entry — the
    // row still exists, and dropping `_id` would hide it from every `_id` scan,
    // including REBAC's non-indexed fallback.
    assert!(
        io.index_lookup("docs", "owner", "main", &Value::Uuid(live_owner))
            .contains(&row_id),
        "the surviving head's owner edge must remain traversable"
    );
    assert!(
        io.index_lookup("docs", "_id", "main", &Value::Uuid(row_id))
            .contains(&row_id),
        "`_id` must survive while a head survives, or `row_is_indexed_on_branch` \
         reports a live row as absent"
    );
    // The editor both heads shared is still justified by the surviving head, so
    // the difference-based retirement must not have taken it.
    for editor in &live_editors {
        assert!(
            io.index_lookup("docs", "editors", "main", editor)
                .contains(&row_id),
            "an array element the SURVIVING head still references was retired: {editor:?}"
        );
    }
    assert!(
        io.index_lookup("docs", "editors", "main", &Value::Array(live_editors))
            .contains(&row_id),
        "the surviving head's whole-array entry must remain"
    );
}

/// The sweep's WIRING, which the two sweep gates above do not touch: they call
/// `repair_all_split_visible_row_families` directly, so deleting the call in
/// `RuntimeCore::new` leaves them both green. What production actually relies on
/// is that constructing a runtime over a damaged store heals it before anything
/// reads — the sync server, the node binding, the web binding and jazz-rn all
/// reach `RuntimeCore::new` (`runtime_tokio::TokioRuntime`, `client.rs`).
///
/// So: build a runtime over an already-split store and ask it, through the
/// runtime's own storage, what the row is. No direct sweep call anywhere.
#[test]
fn constructing_a_runtime_over_a_split_store_heals_it_before_the_first_read() {
    use crate::runtime_core::{NoopScheduler, RuntimeCore};

    let mut io = split_capable_storage();
    crate::test_support::persist_test_schema(&mut io, &users_test_schema());
    crate::test_support::persist_test_schema(&mut io, &users_next_generation_schema());

    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::GlobalServer);
    let client_id = ClientId::new();
    sm.add_client_with_storage(&io, client_id);
    sm.set_client_acks_deliveries(client_id, true);
    sm.set_client_role(client_id, ClientRole::User);
    sm.set_client_session(
        client_id,
        crate::query_manager::session::Session::new("alice"),
    );
    sm.take_outbox();

    let row_id = ObjectId::new();
    let generation_a = visible_row(row_id, "main", Vec::new(), 1_000, b"before");
    send_and_approve(
        &mut sm,
        &mut io,
        client_id,
        row_metadata("users"),
        &generation_a,
    );
    let key = format!("main:{}", row_id.uuid().simple());
    let stale_family = visible_heads_for(&io, "main", row_id)
        .pop()
        .expect("generation A must hold the head");
    let stale_bytes = io
        .raw_table_get(&stale_family, &key)
        .expect("probe")
        .expect("generation A head bytes");

    let generation_b = next_generation_row(
        row_id,
        vec![generation_a.batch_id],
        2_000,
        "after",
        "online",
    );
    send_and_approve(
        &mut sm,
        &mut io,
        client_id,
        next_generation_metadata(),
        &generation_b,
    );
    reseed_the_measured_production_split(
        &mut io,
        "main",
        row_id,
        &stale_family,
        &stale_bytes,
        users_schema_hash(),
        next_generation_schema_hash(),
    );
    assert_eq!(
        visible_heads_for(&io, "main", row_id).len(),
        2,
        "the store handed to the runtime must actually be split"
    );

    // The wiring under test: nothing here calls the sweep.
    let app_id = crate::schema_manager::AppId::from_name("defect27-wiring");
    let schema_manager = crate::schema_manager::SchemaManager::new(
        SyncManager::new(),
        users_next_generation_schema(),
        app_id,
        "dev",
        "main",
    )
    .expect("schema manager");
    let core = RuntimeCore::new(schema_manager, io, NoopScheduler);

    let heads = visible_heads_for(core.storage(), "main", row_id);
    assert_eq!(
        heads.len(),
        1,
        "constructing the runtime must have healed the split before any read; \
         heads: {heads:?}"
    );
    assert_eq!(
        core.storage()
            .load_visible_region_row("users", "main", row_id)
            .expect("visible read should succeed")
            .expect("the row must be visible")
            .batch_id(),
        generation_b.batch_id,
        "the first read through a freshly constructed runtime must serve the \
         generation the writes landed in"
    );
}

/// PROBE, not a gate: read the two POINTERS out of a copy of the live incident
/// store and print them.
///
/// SETTLED (2026-08-16) — the answer is recorded in
/// `reseed_the_measured_production_split`: two heads on the current branch, the
/// derived `__row_locator` naming the FOSSIL, and the authoritative locator
/// PRESENT and naming the live family. The fossil family also holds a head for a
/// second branch, which is not itself split. Re-run this probe after any change
/// to how either pointer is written.
///
/// It was run to settle two gates in this file that encoded mutually exclusive
/// claims about what production looks like:
///
///   * `a_stale_derived_locator_does_not_outrank_the_locator_writes_keep_current`
///     assumes the authoritative `__visible_row_table_locator` is PRESENT and
///     names the live generation (the inbound write stamps it), while
///     `__row_locator` is stale. If that is the real shape, the read-ladder
///     inversion alone fixes production reads.
///   * `the_sweep_heals_a_store_that_is_already_split` assumed the authoritative
///     locator was ABSENT — if that had been the real shape, the ladder inversion
///     could not help and the sweep would be the only thing healing the incident.
///     It was not the real shape; that gate has been corrected to the measurement.
///
/// Run against a COPY, never the live volume:
/// ```text
/// docker run --rm -v linsa_jazz-sync-data:/data:ro -v /tmp/srv-copy:/out \
///   node:22-bookworm-slim sh -c 'cp -r /data/jazz.rocksdb /out/'
/// JAZZ_PROBE_PATH=/tmp/srv-copy/jazz.rocksdb \
/// JAZZ_PROBE_ROW_ID=c24432b4-c5d0-5d58-a636-de2d99b6d932 \
/// cargo test -p jazz-tools --features "rocksdb sqlite test-utils" --lib \
///   live_incident_store_pointers -- --ignored --nocapture
/// ```
/// Delete the copy afterwards.
#[cfg(feature = "rocksdb")]
#[test]
#[ignore = "probe against a copy of the live store; needs JAZZ_PROBE_PATH"]
fn live_incident_store_pointers() {
    use crate::storage::RocksDBStorage;

    let Ok(path) = std::env::var("JAZZ_PROBE_PATH") else {
        println!("JAZZ_PROBE_PATH unset; nothing to probe");
        return;
    };
    let row_id = ObjectId::from_uuid(
        std::env::var("JAZZ_PROBE_ROW_ID")
            .expect("JAZZ_PROBE_ROW_ID must be set")
            .parse()
            .expect("JAZZ_PROBE_ROW_ID must parse as a uuid"),
    );

    let io = RocksDBStorage::open(&path, 64 * 1024 * 1024).expect("open the store COPY");

    // Which families physically hold this row, per branch — the ground truth
    // both pointers are supposed to agree with.
    let mut families: Vec<(String, usize, Vec<String>)> = Vec::new();
    for (raw_table, _) in io
        .scan_raw_table_headers()
        .expect("raw table header scan")
        .into_iter()
        .filter(|(name, _)| name.starts_with("rowtable:visible:users:"))
    {
        let keys = io
            .raw_table_scan_prefix_keys(&raw_table, "")
            .expect("family key scan");
        let hits: Vec<String> = keys
            .iter()
            .filter(|key| key.ends_with(&row_id.uuid().simple().to_string()))
            .cloned()
            .collect();
        families.push((raw_table, keys.len(), hits));
    }
    println!("=== visible families for the probed row ===");
    for (raw_table, total_keys, hits) in &families {
        println!("{raw_table}  total_keys={total_keys}  this_row={hits:?}");
    }

    println!("=== __row_locator (derived) ===");
    println!(
        "{:?}",
        io.load_row_locator(row_id)
            .expect("row locator read")
            .map(|locator| (
                locator.table.to_string(),
                locator
                    .origin_schema_hash
                    .map(|schema_hash| schema_hash.to_string())
            ))
    );

    println!("=== __visible_row_table_locator (authoritative), per branch holding the row ===");
    let mut branches: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for (_, _, hits) in &families {
        for key in hits {
            if let Some((branch, _)) = key.rsplit_once(':') {
                branches.insert(branch.to_string());
            }
        }
    }
    if branches.is_empty() {
        println!("(the row has no visible head in any family)");
    }
    for branch in &branches {
        println!(
            "branch {branch}: {:?}",
            io.load_visible_row_table_locator(branch, row_id)
                .expect("exact visible locator read")
                .map(|locator| locator.schema_hash.to_string())
        );
    }
}

/// The sweep marker must bound the cost WITHOUT ever skipping a sweep the store
/// needs. The risk it introduces is a false "already clean": if the marker
/// survived a deployment, the sweep would never run again on the exact stores it
/// exists for.
///
/// So: sweep a two-generation store clean, confirm a second pass is skipped, then
/// register a THIRD generation and confirm the sweep re-arms and repairs a split
/// created under it.
#[test]
fn the_sweep_marker_re_arms_when_a_new_generation_appears() {
    let mut io = split_capable_storage();
    crate::test_support::persist_test_schema(&mut io, &users_test_schema());
    crate::test_support::persist_test_schema(&mut io, &users_next_generation_schema());

    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::GlobalServer);
    let client_id = ClientId::new();
    sm.add_client_with_storage(&io, client_id);
    sm.set_client_acks_deliveries(client_id, true);
    sm.set_client_role(client_id, ClientRole::User);
    sm.set_client_session(
        client_id,
        crate::query_manager::session::Session::new("alice"),
    );
    sm.take_outbox();

    let row_id = ObjectId::new();
    let generation_a = visible_row(row_id, "main", Vec::new(), 1_000, b"before");
    send_and_approve(
        &mut sm,
        &mut io,
        client_id,
        row_metadata("users"),
        &generation_a,
    );
    let key = format!("main:{}", row_id.uuid().simple());
    let stale_family = visible_heads_for(&io, "main", row_id)
        .pop()
        .expect("generation A head");
    let stale_bytes = io
        .raw_table_get(&stale_family, &key)
        .expect("probe")
        .expect("generation A bytes");
    let generation_b = next_generation_row(
        row_id,
        vec![generation_a.batch_id],
        2_000,
        "after",
        "online",
    );
    send_and_approve(
        &mut sm,
        &mut io,
        client_id,
        next_generation_metadata(),
        &generation_b,
    );

    // Split it, sweep it clean, and confirm the marker now short-circuits.
    reseed_the_measured_production_split(
        &mut io,
        "main",
        row_id,
        &stale_family,
        &stale_bytes,
        users_schema_hash(),
        next_generation_schema_hash(),
    );
    let first = crate::storage::repair_all_split_visible_row_families(&mut io).expect("sweep");
    assert_eq!(
        first.split_rows, 1,
        "the first pass must actually repair the split: {first:?}"
    );

    reseed_the_measured_production_split(
        &mut io,
        "main",
        row_id,
        &stale_family,
        &stale_bytes,
        users_schema_hash(),
        next_generation_schema_hash(),
    );
    let second = crate::storage::repair_all_split_visible_row_families(&mut io).expect("sweep");
    assert!(
        second.is_noop(),
        "with no new generation registered the marker must short-circuit the pass, \
         even though the store was damaged again behind its back: {second:?}"
    );

    // A deployment: a third generation registers, which must re-arm the sweep.
    let third_generation_schema = SchemaBuilder::new()
        .table(
            TableSchema::builder("users")
                .column("value", ColumnType::Text)
                .column("presence", ColumnType::Text)
                .column("device", ColumnType::Text),
        )
        .build();
    crate::test_support::persist_test_schema(&mut io, &third_generation_schema);
    let third_hash = SchemaHash::compute(&third_generation_schema);
    // Register the family the way a deployment does — through a real inbound
    // write — so it appears in the header scan the generation-set is built from.
    let other_row_id = ObjectId::new();
    let third_generation_row = StoredRowBatch::new(
        other_row_id,
        "main",
        Vec::new(),
        encode_row(
            &third_generation_schema[&"users".into()].columns,
            &[
                Value::Text("c".to_string()),
                Value::Text("online".to_string()),
                Value::Text("phone".to_string()),
            ],
        )
        .expect("generation-C row should encode"),
        RowProvenance::for_insert(other_row_id.to_string(), 5_000),
        HashMap::new(),
        crate::row_histories::RowState::VisibleDirect,
        None,
    );
    send_and_approve(
        &mut sm,
        &mut io,
        client_id,
        HashMap::from([
            (MetadataKey::Table.to_string(), "users".to_string()),
            (
                MetadataKey::OriginSchemaHash.to_string(),
                third_hash.to_string(),
            ),
        ]),
        &third_generation_row,
    );
    // And damage the original row again, now that a new generation exists.
    reseed_the_measured_production_split(
        &mut io,
        "main",
        row_id,
        &stale_family,
        &stale_bytes,
        users_schema_hash(),
        next_generation_schema_hash(),
    );

    let third = crate::storage::repair_all_split_visible_row_families(&mut io).expect("sweep");
    assert_eq!(
        third.split_rows, 1,
        "a newly registered generation must re-arm the sweep — this is the only thing \
         standing between the marker and a store that is never swept again: {third:?}"
    );
    assert_eq!(
        visible_heads_for(&io, "main", row_id).len(),
        1,
        "and the re-armed pass must actually heal the split"
    );
}

/// The four OTHER writers that put visible bytes into a caller-chosen family —
/// batch-rejection patching (`patch_exact_row_batch_for_schema_hash`,
/// `patch_row_region_rows_by_batch_with_storage`) and the two rejected-delete
/// restores — pick that family from a locator rather than measuring where the
/// head is. `__row_locator` is keyed by ROW ID alone while heads are keyed by
/// `(branch, row)`, and `scan_history_row_batches` carries no branch filter, so
/// each of them can write a head into a family that is not the one the row is
/// already in. Visible keys carry no batch id, so that ADDS a head.
///
/// Rather than patch four call sites and hope there is no fifth, the invariant
/// is enforced where the bytes land. This gate drives the shared chokepoint with
/// the pointer state those writers produce: a head in generation B, and
/// `__row_locator` still naming generation A.
#[test]
fn a_visible_write_aimed_by_a_stale_locator_moves_the_head_instead_of_forking_it() {
    let mut io = split_capable_storage();
    crate::test_support::persist_test_schema(&mut io, &users_test_schema());
    crate::test_support::persist_test_schema(&mut io, &users_next_generation_schema());

    let mut sm = SyncManager::new().with_durability_tier(DurabilityTier::GlobalServer);
    let client_id = ClientId::new();
    sm.add_client_with_storage(&io, client_id);
    sm.set_client_acks_deliveries(client_id, true);
    sm.set_client_role(client_id, ClientRole::User);
    sm.set_client_session(
        client_id,
        crate::query_manager::session::Session::new("alice"),
    );
    sm.take_outbox();

    let row_id = ObjectId::new();
    let generation_a = visible_row(row_id, "main", Vec::new(), 1_000, b"before");
    send_and_approve(
        &mut sm,
        &mut io,
        client_id,
        row_metadata("users"),
        &generation_a,
    );
    let generation_b = next_generation_row(
        row_id,
        vec![generation_a.batch_id],
        2_000,
        "after",
        "online",
    );
    send_and_approve(
        &mut sm,
        &mut io,
        client_id,
        next_generation_metadata(),
        &generation_b,
    );
    assert_eq!(
        visible_heads_for(&io, "main", row_id).len(),
        1,
        "control: the store must be healthy before the stale-aimed write"
    );

    // The pointer state those four writers act on: the head is in generation B,
    // `__row_locator` says generation A. Any of them would now resolve the write
    // family as A and add a second head there.
    io.put_row_locator(
        row_id,
        Some(&crate::storage::RowLocator {
            table: "users".into(),
            origin_schema_hash: Some(users_schema_hash()),
        }),
    )
    .expect("rewinding the row locator should succeed");

    let restored = visible_row(
        row_id,
        "main",
        vec![generation_b.batch_id],
        3_000,
        b"restored",
    );
    io.upsert_visible_region_rows(
        "users",
        &[crate::row_histories::VisibleRowEntry::new(restored.clone())],
    )
    .expect("the restore-shaped write should succeed");

    let heads = visible_heads_for(&io, "main", row_id);
    assert_eq!(
        heads.len(),
        1,
        "a visible write aimed by a stale locator must MOVE the head, not add one; \
         heads: {heads:?}"
    );
    assert_eq!(
        io.load_visible_region_row("users", "main", row_id)
            .expect("visible read should succeed")
            .expect("the row must be visible")
            .batch_id(),
        restored.batch_id,
        "and the read must serve what that write wrote"
    );
    // Both pointers follow the head, or the next read aims away from it again.
    let head_family = heads[0]
        .strip_prefix("rowtable:visible:users:")
        .expect("visible family name");
    assert_eq!(
        io.load_row_locator(row_id)
            .expect("row locator read")
            .and_then(|locator| locator.origin_schema_hash)
            .map(|schema_hash| schema_hash.to_string())
            .as_deref(),
        Some(head_family),
        "__row_locator must name the family the head ended up in"
    );
    assert_eq!(
        io.load_visible_row_table_locator("main", row_id)
            .expect("exact visible locator read")
            .map(|locator| locator.schema_hash.to_string())
            .as_deref(),
        Some(head_family),
        "the authoritative locator must name it too"
    );
}

/// `wmsgs` in two generations (B adds a table, so the family differs while the row's
/// bytes stay comparable). The store declares its `chat+at` composite and `chat>body`
/// trigram indexes (`split_window_messages_store`).
fn window_messages_generations() -> (
    crate::query_manager::types::Schema,
    crate::query_manager::types::Schema,
) {
    let wmsgs = || {
        TableSchema::builder("wmsgs")
            .column("chat", ColumnType::Uuid)
            .column("at", ColumnType::Timestamp)
            .column("dead", ColumnType::Boolean)
            .column("body", ColumnType::Text)
    };
    (
        SchemaBuilder::new().table(wmsgs()).build(),
        SchemaBuilder::new()
            .table(wmsgs())
            .table(TableSchema::builder("tags").column("label", ColumnType::Text))
            .build(),
    )
}

/// A store where one `wmsgs` row has a head in both generations — the stale one at
/// `at = 10` saying "alpha", the live one at `at = 20` saying "alpine", in the same
/// chat — with both heads' declared-index entries filed the way the write path files
/// them.
fn split_window_messages_store() -> (SqliteStorage, ObjectId, ObjectId, SchemaHash, SchemaHash) {
    let mut io = split_capable_storage();
    let (schema_a, schema_b) = window_messages_generations();
    crate::test_support::persist_test_schema(&mut io, &schema_a);
    crate::test_support::persist_test_schema(&mut io, &schema_b);
    let (hash_a, hash_b) = (
        SchemaHash::compute(&schema_a),
        SchemaHash::compute(&schema_b),
    );

    // The store maintains them, so the repair paths retire their entries.
    let declarations = crate::query_manager::index_declarations::IndexDeclarations::empty()
        .with_composite("wmsgs", "chat", "at")
        .and_then(|declarations| declarations.with_trigram("wmsgs", "chat", "body"))
        .expect("declarations");
    crate::query_manager::declared_index::propose(&mut io, &declarations)
        .expect("declare the indexes");

    let row_id = ObjectId::new();
    let chat = ObjectId::new();
    let key = format!("main:{}", row_id.uuid().simple());
    for (schema, hash, at, body) in [
        (&schema_a, hash_a, 10, "alpha"),
        (&schema_b, hash_b, 20, "alpine"),
    ] {
        let columns = &schema[&"wmsgs".into()].columns;
        let bytes = crate::query_manager::encoding::encode_row(
            columns,
            &[
                Value::Uuid(chat),
                Value::Timestamp(at),
                Value::Boolean(false),
                Value::Text(body.to_string()),
            ],
        )
        .expect("wmsgs row should encode");
        // The family's header, so the store enumerates it as a family holding the row.
        let family = crate::storage::RowRawTableId::new(
            crate::storage::RowRawTableKind::Visible,
            "wmsgs",
            hash,
        );
        io.upsert_raw_table_header(
            family.raw_table_name(),
            &crate::storage::RawTableHeader::row_raw_table(
                crate::storage::RowRawTableKind::Visible,
                "wmsgs",
                hash,
                columns,
            ),
        )
        .expect("family header write");
        io.raw_table_put(family.raw_table_name(), &key, &bytes)
            .expect("head write");
        let mut mutations = Vec::new();
        crate::query_manager::declared_index::push_mutations(
            &mut mutations,
            &declarations,
            "wmsgs",
            "main",
            row_id,
            columns,
            None,
            Some(&bytes),
        );
        io.apply_index_mutations(&mutations)
            .expect("declared entries");
    }
    (io, row_id, chat, hash_a, hash_b)
}

fn window_entry_holds(io: &SqliteStorage, row_id: ObjectId, chat: ObjectId, at: u64) -> bool {
    let value = crate::query_manager::composite_index::composite_value(
        &Value::Uuid(chat),
        &Value::Timestamp(at),
    )
    .expect("a composite value");
    io.index_lookup("wmsgs", "chat+at", "main", &value)
        .contains(&row_id)
}

fn trigram_entry_holds(
    io: &SqliteStorage,
    row_id: ObjectId,
    chat: ObjectId,
    trigram: &str,
) -> bool {
    let value = crate::query_manager::trigram_index::entry_value(&Value::Uuid(chat), trigram)
        .expect("a trigram entry value");
    io.index_lookup("wmsgs", "chat>body", "main", &value)
        .contains(&row_id)
}

/// Declared-index entries follow a dropped head out, as column entries do: a window
/// over the stale `at` or a search for the stale text would otherwise return the row
/// for values it no longer holds — and when the window covers the whole predicate, no
/// filter re-reads the row to catch it. What both heads justify stays.
#[test]
fn a_moved_head_retires_its_declared_index_entries_and_only_its_own() {
    let (mut io, row_id, chat, hash_a, hash_b) = split_window_messages_store();
    assert!(
        window_entry_holds(&io, row_id, chat, 10) && trigram_entry_holds(&io, row_id, chat, "pha"),
        "the stale head's entries must be filed before the head move, or the gate proves nothing"
    );

    crate::storage::drop_stale_visible_row_family_entry(
        &mut io,
        "wmsgs",
        "main",
        row_id,
        hash_a,
        Some(hash_b),
    )
    .expect("the head move should succeed");

    assert!(
        !window_entry_holds(&io, row_id, chat, 10),
        "the dropped head's window entry (at = 10) must be retired"
    );
    assert!(
        !trigram_entry_holds(&io, row_id, chat, "pha"),
        "the dropped head's trigram \"pha\" must be retired"
    );
    assert!(
        window_entry_holds(&io, row_id, chat, 20),
        "the surviving head's window entry must remain"
    );
    for trigram in ["alp", "ine"] {
        assert!(
            trigram_entry_holds(&io, row_id, chat, trigram),
            "the surviving head's trigram {trigram:?} must remain"
        );
    }
}

/// Deleting a row that is still split retires every head's declared-index entries,
/// not only those of the version the delete read.
#[test]
fn deleting_a_split_row_retires_every_heads_declared_index_entries() {
    let (mut io, row_id, chat, _, _) = split_window_messages_store();

    crate::storage::retire_index_entries_for_extra_visible_heads(&mut io, "wmsgs", "main", row_id)
        .expect("the retirement should succeed");

    for at in [10, 20] {
        assert!(
            !window_entry_holds(&io, row_id, chat, at),
            "the window entry at = {at} outlived the delete"
        );
    }
    for trigram in ["alp", "pha", "ine"] {
        assert!(
            !trigram_entry_holds(&io, row_id, chat, trigram),
            "the trigram {trigram:?} outlived the delete"
        );
    }
}

/// A store whose declared-index record cannot be decoded trusts none of its declared
/// indexes, and nothing rewrites the record. A repair of a split row then retires the
/// dropped head's column entries alone, rather than failing, and with it the write
/// that moved the head or the delete.
#[test]
fn an_unreadable_index_record_does_not_block_a_split_rows_repair() {
    // The split store, its heads' `at` column entries filed, its record garbled.
    let unreadable_split_store = || {
        let (mut io, row_id, _, hash_a, hash_b) = split_window_messages_store();
        for at in [10, 20] {
            io.index_insert("wmsgs", "at", "main", &Value::Timestamp(at), row_id)
                .expect("column entry");
        }
        io.raw_table_put("declared_indexes", "record", &[0xff])
            .expect("garble the record");
        assert!(
            crate::query_manager::declared_index::load_record(&io).is_err(),
            "the record must be unreadable, or the gate proves nothing"
        );
        (io, row_id, hash_a, hash_b)
    };
    let at_holds = |io: &SqliteStorage, row_id: ObjectId, at: u64| {
        io.index_lookup("wmsgs", "at", "main", &Value::Timestamp(at))
            .contains(&row_id)
    };

    let (mut io, row_id, hash_a, hash_b) = unreadable_split_store();

    crate::storage::drop_stale_visible_row_family_entry(
        &mut io,
        "wmsgs",
        "main",
        row_id,
        hash_a,
        Some(hash_b),
    )
    .expect("the head move should succeed");
    assert!(
        !at_holds(&io, row_id, 10),
        "the dropped head's column entry (at = 10) must be retired"
    );
    assert!(
        at_holds(&io, row_id, 20),
        "the surviving head's column entry must remain"
    );

    let (mut io, row_id, _, _) = unreadable_split_store();
    crate::storage::retire_index_entries_for_extra_visible_heads(&mut io, "wmsgs", "main", row_id)
        .expect("the retirement should succeed");
    for at in [10, 20] {
        assert!(
            !at_holds(&io, row_id, at),
            "the column entry at = {at} outlived the delete"
        );
    }
}
