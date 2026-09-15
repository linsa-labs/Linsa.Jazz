//! A local write marks every subscription whose graph involves the written table — by
//! table, not by row (`mark_subscriptions_rows_changed`, `mark_subscriptions_dirty_with_origin`
//! both set `has_pending_local_updates`). That is fine as long as the next `process()`
//! leaves each marked subscription clean again. It did not: the flag was cleared only when
//! a non-empty delta was delivered, so a subscription whose visible result the write did
//! not change kept it, `should_process_subscription` stayed true, and it was re-settled —
//! and, with a session, re-authorized row by row — on every later pass, for good.
//!
//! MEASURED 2026-09-15 on the iOS simulator while typing into a chat draft (one upsert per
//! keystroke): ~25 ms of engine work per letter, `authorized_tuples_from_graph_with_cache`
//! 59 % of the JS thread, split almost evenly between the two back-to-back `process()`
//! calls at the top of `immediate_tick`, and `build_ordered_delta_with_post_ids` at zero —
//! the processed subscriptions paid for authorization and delivered nothing. The app's
//! presence heartbeat updates the own `users` row every 10 s, which poisons every
//! subscription with `users` anywhere in its graph that does not show that row.
//! Reproduced natively against a copy of the simulator store: one heartbeat took a letter
//! from 9 subscriptions settled and 12 authorization checks to 17 and 102.

use super::*;
use crate::query_manager::QuerySubscriptionId;

fn users_named(qm: &QueryManager, name: &str) -> Query {
    qm.query("users")
        .filter_eq("name", Value::Text(name.into()))
        .build()
}

fn assert_clean(qm: &QueryManager, sub: QuerySubscriptionId, context: &str) {
    let subscription = &qm.subscriptions[&sub];
    assert!(subscription.settled_once, "{context}: settled once");
    assert!(!subscription.needs_recompile, "{context}: needs_recompile");
    assert!(
        !subscription.needs_visibility_recompute,
        "{context}: needs_visibility_recompute"
    );
    assert!(
        !subscription.graph.has_dirty_nodes(),
        "{context}: graph has dirty nodes"
    );
    assert!(
        !subscription.has_pending_local_updates,
        "{context}: a local write that changed nothing this subscription shows left it \
         flagged, so every later process() re-settles and re-authorizes it"
    );
}

#[test]
fn a_local_write_that_changes_nothing_visible_leaves_the_subscription_clean() {
    let (mut qm, mut storage) = create_query_manager(SyncManager::new(), test_schema());
    let alice_query = users_named(&qm, "Alice");
    let alice = qm.subscribe(alice_query).unwrap();
    let bob_query = users_named(&qm, "Bob");
    let bob = qm.subscribe(bob_query).unwrap();

    qm.insert(
        &mut storage,
        "users",
        &[Value::Text("Bob".into()), Value::Integer(1)],
    )
    .unwrap();
    qm.process(&mut storage);
    let _ = qm.take_updates();

    // Lands in `users`, so both subscriptions are marked; only Bob's result changes.
    let written = qm
        .insert(
            &mut storage,
            "users",
            &[Value::Text("Bob".into()), Value::Integer(2)],
        )
        .unwrap();
    qm.process(&mut storage);
    let updates = qm.take_updates();
    assert!(
        updates.iter().any(|update| update.subscription_id == bob
            && update
                .delta
                .added
                .iter()
                .any(|row| row.id == written.row_id)),
        "the subscription the write belongs to still gets it"
    );
    assert!(
        updates.iter().all(|update| update.subscription_id != alice),
        "nothing visible changed for Alice's subscription"
    );

    assert_clean(&qm, alice, "after one unrelated local write");
}

#[test]
fn a_cleared_subscription_still_delivers_the_next_write_that_concerns_it() {
    let (mut qm, mut storage) = create_query_manager(SyncManager::new(), test_schema());
    let alice_query = users_named(&qm, "Alice");
    let alice = qm.subscribe(alice_query).unwrap();

    qm.insert(
        &mut storage,
        "users",
        &[Value::Text("Bob".into()), Value::Integer(1)],
    )
    .unwrap();
    qm.process(&mut storage);
    qm.insert(
        &mut storage,
        "users",
        &[Value::Text("Bob".into()), Value::Integer(2)],
    )
    .unwrap();
    qm.process(&mut storage);
    let _ = qm.take_updates();

    let written = qm
        .insert(
            &mut storage,
            "users",
            &[Value::Text("Alice".into()), Value::Integer(3)],
        )
        .unwrap();
    qm.process(&mut storage);
    let delivered: Vec<_> = qm
        .take_updates()
        .into_iter()
        .filter(|update| update.subscription_id == alice)
        .flat_map(|update| update.delta.added)
        .map(|row| row.id)
        .collect();
    assert_eq!(delivered, vec![written.row_id]);
}

/// The app's shape: sync-backed at the local tier with no server, so every local write
/// stays in `pending_local_row_batches` and is fed to the settle as a source overlay.
#[test]
fn a_sync_backed_local_tier_subscription_comes_out_clean_after_an_unrelated_write() {
    let (mut qm, mut storage) = create_query_manager(SyncManager::new(), test_schema());
    let alice_query = users_named(&qm, "Alice");
    let alice = qm
        .subscribe_with_sync(alice_query, None, Some(DurabilityTier::Local))
        .unwrap();
    let bob_query = users_named(&qm, "Bob");
    let bob = qm
        .subscribe_with_sync(bob_query, None, Some(DurabilityTier::Local))
        .unwrap();

    qm.insert(
        &mut storage,
        "users",
        &[Value::Text("Bob".into()), Value::Integer(1)],
    )
    .unwrap();
    qm.process(&mut storage);
    let _ = qm.take_updates();
    assert_clean(&qm, bob, "Bob after his own first write");

    let written = qm
        .insert(
            &mut storage,
            "users",
            &[Value::Text("Bob".into()), Value::Integer(2)],
        )
        .unwrap();
    qm.process(&mut storage);
    let updates = qm.take_updates();
    assert!(updates.iter().any(|update| {
        update.subscription_id == bob
            && update
                .delta
                .added
                .iter()
                .any(|row| row.id == written.row_id)
    }));
    assert_clean(&qm, alice, "sync-backed, after an unrelated local write");

    // A second back-to-back pass, as `immediate_tick` runs, must find nothing to do.
    qm.process(&mut storage);
    assert!(qm.take_updates().is_empty());
    assert_clean(&qm, alice, "after the second pass");
}
