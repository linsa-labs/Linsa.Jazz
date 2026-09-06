//! A split row must cost the ladder walk once, not on every read.
//!
//! When a row's locator names a raw-table family that does not hold it, the store falls
//! back to probing every visible family for `<branch>:<row>`. That fallback is a real
//! correctness net — a row delivered before the catalogue knew its origin schema is placed
//! in the CURRENT family while its locator keeps the server-stamped origin hash — and it
//! already returns `needs_exact_locator`, the flag whose whole purpose is "correct me".
//!
//! Every consumer of that flag is on a WRITE path. A row that is only ever read therefore
//! never heals: it pays the walk on every read for the lifetime of the store.
//!
//! MEASURED in production 2026-08-18, on an otherwise idle server with nothing happening in
//! any chat: 1,200 recoveries in five minutes — four per second — all for ONE `users` row.
//! Settle passes accounted for 1.3% of a core while the process burned 7.4%, so the walk
//! was most of what the server was doing.
//!
//! `SqliteStorage`, because `MemoryStorage` keeps visible entries as structs and never
//! executes the ladder at all.

use super::support::{docs_v1, docs_v2, runtime_over};
use super::*;
use crate::storage::SqliteStorage;
/// Ladder walks this store served (per store since v18 item 5; see `locator_warmth.rs`).
fn recoveries(core: &RuntimeCore<SqliteStorage, NoopScheduler>) -> u64 {
    core.storage().visible_ladder_recoveries_for_test()
}

#[test]
fn a_split_row_is_walked_once_and_then_healed() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let path = dir.path().join("heal.sqlite");
    let storage = SqliteStorage::open(&path).expect("sqlite storage should open");

    let mut core = runtime_over(docs_v1(), "ladder-heal", storage);
    let alice = WriteContext::from_session(Session::new("alice"));

    let ((row_id, _), _) = insert_and_wait_for_batch(
        &mut core,
        "docs",
        HashMap::from([
            ("owner".to_string(), Value::Text("alice".to_string())),
            ("body".to_string(), Value::Text("draft".to_string())),
        ]),
        Some(&alice),
        DurabilityTier::Local,
    )
    .expect("the row inserts under generation A");
    core.batched_tick();
    core.immediate_tick();

    let branch_a = crate::storage::sole_branch_name(core.storage())
        .expect("branch registry readable")
        .expect("the seeded row registered a branch");

    // The crossing, then a write on the new generation. The write moves `__row_locator` to
    // the new family while generation A keeps its own head — so a read addressed to branch
    // A now asks a locator that names the wrong family, which is the production shape.
    let storage = core.into_storage();
    let mut core = runtime_over(docs_v2(), "ladder-heal", storage);
    core.update(
        row_id,
        vec![("body".to_string(), Value::Text("edited".to_string()))],
        Some(&alice),
    )
    .expect("the update across the crossing applies");
    core.batched_tick();
    core.immediate_tick();

    // First read: the ladder is allowed to fire — that is what it is for.
    let before = recoveries(&core);
    let first = core
        .storage()
        .load_visible_region_row("docs", branch_a.as_str(), row_id)
        .expect("visible row readable");
    let after_first = recoveries(&core);
    assert!(
        first.is_some(),
        "fixture precondition: branch A must still serve a head for the row, else there is \
         nothing for the ladder to find and this gate proves nothing"
    );
    assert!(
        after_first > before,
        "fixture precondition: the first read must actually walk the ladder — it did not, \
         so the locator already names the right family and the split was not reproduced"
    );

    // Second read of the SAME row. The store has already been told, by its own return
    // value, which family holds it.
    let _second = core
        .storage()
        .load_visible_region_row("docs", branch_a.as_str(), row_id)
        .expect("visible row readable");
    let after_second = recoveries(&core);
    eprintln!(
        "ladder walks: first read {}, second read {}",
        after_first - before,
        after_second - after_first
    );
    assert_eq!(
        after_second, after_first,
        "a split row must be walked once and then healed. The fallback already returns \
         `needs_exact_locator`, and every consumer of that flag is on a write path — so a \
         row that is only read pays the walk forever. Production measured 1,200 walks in \
         five minutes against one row on an idle server."
    );
}
