//! Edge cases around delivering what was written while a peer was away.
//!
//! The reproduction and the reap gate live in `offline_reap_delivery.rs`. This file pins
//! the behaviour a fix must NOT break, and the shapes the single reproduction does not
//! reach: more than one row in the gap, more than one subscription on the returning peer,
//! exactly-once delivery, and an untouched fast path for a peer that never left.
//!
//! Each test says what a wrong fix would do to it, because a green test whose failure mode
//! is unstated is not a gate.

#![cfg(feature = "test")]

mod support;

use std::collections::{BTreeSet, HashMap};
use std::time::Duration;

use futures::StreamExt as _;
use jazz_tools::object::ObjectId;

use jazz_tools::server::JazzServer;
use jazz_tools::{
    ColumnType, DurabilityTier, JazzClient, QueryBuilder, SchemaBuilder, TableSchema, Value,
};
use support::{TestingClient, wait_for_query};

const READY_TIMEOUT: Duration = Duration::from_secs(30);
const QUERY_TIMEOUT: Duration = Duration::from_secs(25);
/// Rows written into a single offline window. More than one, because a frontier cursor that
/// prunes too eagerly can deliver the newest and swallow everything before it — which one
/// row can never show.
const GAP_ROWS: usize = 25;

fn test_schema() -> jazz_tools::Schema {
    SchemaBuilder::new()
        .table(
            TableSchema::builder("todos")
                .column("title", ColumnType::Text)
                .column("completed", ColumnType::Boolean),
        )
        .table(TableSchema::builder("notes").column("body", ColumnType::Text))
        .build()
}

fn todo(title: &str) -> HashMap<String, Value> {
    HashMap::from([
        ("title".to_string(), Value::Text(title.to_string())),
        ("completed".to_string(), Value::Boolean(false)),
    ])
}

fn note(body: &str) -> HashMap<String, Value> {
    HashMap::from([("body".to_string(), Value::Text(body.to_string()))])
}

async fn expect_delivered(
    stream: &mut jazz_tools::SubscriptionStream,
    expected: &BTreeSet<ObjectId>,
    what: &str,
) {
    let mut seen: BTreeSet<ObjectId> = BTreeSet::new();
    let deadline = tokio::time::Instant::now() + QUERY_TIMEOUT;
    while !expected.is_subset(&seen) {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            let missing: Vec<_> = expected.difference(&seen).collect();
            panic!(
                "{what}: subscription never delivered {missing:?} (saw {} rows)",
                seen.len()
            );
        }
        let delta = tokio::time::timeout(remaining, stream.next())
            .await
            .unwrap_or_else(|_| {
                let missing: Vec<_> = expected.difference(&seen).collect();
                panic!(
                    "{what}: timed out; {} of {} still missing, first few {:?}",
                    missing.len(),
                    expected.len(),
                    missing.iter().take(5).collect::<Vec<_>>(),
                )
            })
            .unwrap_or_else(|| panic!("{what}: subscription stream closed early"));
        for added in &delta.added {
            seen.insert(added.id);
        }
        for updated in &delta.updated {
            seen.insert(updated.id);
        }
    }
}

/// Park the peer offline without reaping it, and assert it really is parked.
///
/// Asserting matters: reaping is what heals the defect, so a test that silently got a reap
/// would pass for the wrong reason and would keep passing if the fix were reverted.
async fn park_offline_unreaped(server: &JazzServer) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    while server.disconnect_candidate_count().await == 0 {
        assert!(
            tokio::time::Instant::now() < deadline,
            "timed out waiting for the disconnect candidate to register",
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// An EDIT to a row the peer already had arrives today — and must keep arriving.
///
/// This was written expecting a second, separate failure. A review argued that resettle
/// resends only `scope.difference(&old_query_scope)` (`sync_manager/mod.rs:795-812`), that
/// an edit never changes membership, and that an unreaped peer therefore never hears about
/// it. That reading of the code is right and the conclusion is wrong: measured here, the
/// edit does arrive. Some other path repairs already-visible rows on reconnect.
///
/// So the defect is narrower than it looked — it is about rows that become NEWLY VISIBLE
/// during the gap. This test stays as a guard, because a fix that touches per-client
/// delivery state could easily break the half that works.
///
/// Asserted on the SUBSCRIPTION, not on `query`: the row id is already in the peer's local
/// store, so an id-only assertion is vacuous, and a one-shot `query` is worse — the first
/// version of this test read the edited value back in 0.49 s with the peer's store
/// untouched, which is `query` fetching from the server, not the peer being up to date.
/// What cannot be faked is an `updated` delta: the peer's local state only changes when the
/// edited batch is actually applied to it. Verified by a control run with the edit
/// suppressed — the assertion then times out, so it is capable of failing.
#[tokio::test]
async fn an_edit_made_during_the_gap_reaches_a_returning_peer() {
    let schema = test_schema();
    let server = JazzServer::start_with_schema(schema.clone()).await;

    let alice = TestingClient::builder()
        .with_server(&server)
        .with_schema(schema.clone())
        .with_user_id("alice-edit-gap")
        .ready_on("todos", READY_TIMEOUT)
        .connect()
        .await;

    let (bob_ctx, bob) = TestingClient::builder()
        .with_server(&server)
        .with_schema(schema.clone())
        .with_user_id("bob-edit-gap")
        .with_persistent_storage()
        .ready_on("todos", READY_TIMEOUT)
        .connect_with_context()
        .await;
    let pinned_client_id = bob.client_id();

    let query = QueryBuilder::new("todos").build();
    let (row_id, _, _) = alice.insert("todos", todo("original")).expect("insert");
    let mut before = bob.subscribe(query.clone()).await.expect("bob subscribes");
    expect_delivered(
        &mut before,
        &[row_id].into_iter().collect(),
        "bob while online",
    )
    .await;

    bob.shutdown().await.expect("bob goes offline");
    park_offline_unreaped(&server).await;

    // Nothing is inserted here on purpose: an insert would land in the newly-visible diff
    // and could carry the test to green while the edit path stayed broken.
    alice
        .update(
            row_id,
            vec![(
                "title".to_string(),
                Value::Text("edited-while-away".to_string()),
            )],
        )
        .expect("alice edits while bob is away");
    wait_for_query(
        &alice,
        query.clone(),
        Some(DurabilityTier::EdgeServer),
        QUERY_TIMEOUT,
        "alice's edit reached the server",
        |rows| {
            matches!(rows.first(), Some((_, values)) if matches!(values.first(), Some(Value::Text(t)) if t == "edited-while-away"))
                .then_some(())
        },
    )
    .await;

    let mut reconnect_ctx = bob_ctx.clone();
    reconnect_ctx.client_id = pinned_client_id;
    let bob_back = JazzClient::connect(reconnect_ctx)
        .await
        .expect("bob reconnects with the same identity");
    let mut sub = bob_back
        .subscribe(query.clone())
        .await
        .expect("bob resubscribes");

    // CHANGED 2026-09-06 (diff r27), and the old assertion is quoted here because it was not
    // wrong so much as accidentally specific. It was:
    //
    //     the initial snapshot replays what bob already had, so `added` proves nothing here.
    //     Only an `updated` for this row means the edited batch was applied to his store.
    //     ...
    //     applied = delta.updated.iter().any(|row| row.id == row_id);
    //
    // That reasoning assumed the snapshot would carry the STALE value, which is true only of
    // the unbounded settle path. Measured 8/8 deterministically at each budget: under a settle
    // budget the row bob is owed reaches him BEFORE his resubscribe, so his first delta carries
    // the edit in `added` and no `updated` ever follows. His store is correct either way — the
    // read-back below returns `edited-while-away` at every budget — and it is not a race the
    // unbounded path happens to win: with 0/5/20/100/500 ms inserted between connect and
    // subscribe, `None` still waits for the resubscribe to push the row.
    //
    // So the assertion now says what the comment above always meant: the edited batch is in
    // bob's store. A fix that leaves him stale still fails, and the incidental dependence on
    // delta kind is gone.
    //
    // AMENDED 2026-09-06 (diff r28), because the first version of this change was VACUOUS and
    // the file's own header at the top warns against exactly the two fakes it fell into:
    //
    //   * it accepted the row by ID alone, and the resubscribe snapshot always replays the row
    //     bob already had — so the loop exited on the first delta whatever it carried;
    //   * the read-back below was described as local. It is not. `wait_for_query` goes through
    //     `TokioRuntime::query`, which hardcodes `QueryPropagation::Full`
    //     (`runtime_tokio.rs:614-629`), so a `None` tier still lets the server answer.
    //
    // What cannot be faked is the CONTENT of the delta. The row is inserted as `"original"`
    // and edited to `"edited-while-away"`; a stale replay carries the first string and the
    // applied edit carries the second. The bytes are searched rather than decoded because
    // `SubscriptionStream` yields `OrderedRowDelta` with no `RowDescriptor`
    // (`lib.rs:180-204`) — the descriptor rides on `SubscriptionDelta`, one layer down — and
    // reconstructing an output descriptor here would test my reconstruction, not the delivery.
    /// The edited title as it appears inside an encoded row. `Value::Text` is written into the
    /// row payload verbatim, so a substring search over the bytes distinguishes the applied
    /// edit from a stale replay without needing the query's output descriptor.
    const EDITED: &[u8] = b"edited-while-away";
    fn carries_edit(row: &jazz_tools::query_manager::types::Row, id: ObjectId) -> bool {
        row.id == id && row.data.windows(EDITED.len()).any(|w| w == EDITED)
    }

    let deadline = tokio::time::Instant::now() + QUERY_TIMEOUT;
    let mut applied = false;
    let mut seen_stale = false;
    while !applied {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        let complaint = if seen_stale {
            "bob's subscription replayed the row but never carried the edited value — he is \
             back online with a stale copy, which is the defect this test exists for"
        } else {
            "bob's subscription never mentioned the row at all — not as an update and not in \
             its first snapshot"
        };
        assert!(!remaining.is_zero(), "{complaint}");
        let delta = tokio::time::timeout(remaining, sub.next())
            .await
            .unwrap_or_else(|_| panic!("{complaint}"))
            .expect("subscription stream closed early");
        seen_stale |= delta.added.iter().any(|a| a.id == row_id)
            || delta.updated.iter().any(|u| u.id == row_id);
        applied = delta.added.iter().any(|a| carries_edit(&a.row, row_id))
            || delta
                .updated
                .iter()
                .any(|u| u.row.as_ref().is_some_and(|row| carries_edit(row, row_id)));
    }

    // Belt and braces, and labelled honestly: this is a `QueryPropagation::Full` read, so on
    // its own it would prove nothing about bob's store. It stays because the assertion above
    // already established the delivery, and this catches a store that took the delta on the
    // wire and failed to persist it.
    wait_for_query(
        &bob_back,
        query.clone(),
        None,
        QUERY_TIMEOUT,
        "bob's store never took the edit made while he was away",
        |rows| {
            rows.iter()
                .any(|(id, values)| {
                    *id == row_id
                        && matches!(values.first(), Some(Value::Text(t)) if t == "edited-while-away")
                })
                .then_some(())
        },
    )
    .await;

    bob_back.shutdown().await.ok();
    alice.shutdown().await.ok();
    server.shutdown().await;
}

/// Every row written during the gap arrives — not just the newest.
///
/// A fix that re-derives the subscription but leaves the delivery cursor claiming the whole
/// gap would deliver nothing here. A fix that advances the cursor to the newest batch and
/// prunes its ancestors too eagerly would deliver the last row and swallow the other 24.
#[tokio::test]
async fn every_row_written_during_a_long_gap_arrives() {
    let schema = test_schema();
    let server = JazzServer::start_with_schema(schema.clone()).await;

    let alice = TestingClient::builder()
        .with_server(&server)
        .with_schema(schema.clone())
        .with_user_id("alice-gap-many")
        .ready_on("todos", READY_TIMEOUT)
        .connect()
        .await;

    let (bob_ctx, bob) = TestingClient::builder()
        .with_server(&server)
        .with_schema(schema.clone())
        .with_user_id("bob-gap-many")
        .with_persistent_storage()
        .ready_on("todos", READY_TIMEOUT)
        .connect_with_context()
        .await;
    let pinned_client_id = bob.client_id();

    let (anchor_id, _, _) = alice.insert("todos", todo("anchor")).expect("insert");
    let query = QueryBuilder::new("todos").build();
    let mut before = bob.subscribe(query.clone()).await.expect("bob subscribes");
    expect_delivered(
        &mut before,
        &[anchor_id].into_iter().collect(),
        "bob while online",
    )
    .await;

    bob.shutdown().await.expect("bob goes offline");
    park_offline_unreaped(&server).await;

    let mut written: BTreeSet<ObjectId> = BTreeSet::new();
    for i in 0..GAP_ROWS {
        let (id, _, _) = alice
            .insert("todos", todo(&format!("gap-{i}")))
            .expect("alice writes while bob is away");
        written.insert(id);
    }
    wait_for_query(
        &alice,
        query.clone(),
        Some(DurabilityTier::EdgeServer),
        QUERY_TIMEOUT,
        "alice's writes reached the server",
        |rows| (rows.len() == GAP_ROWS + 1).then_some(()),
    )
    .await;

    let mut reconnect_ctx = bob_ctx.clone();
    reconnect_ctx.client_id = pinned_client_id;
    let bob_back = JazzClient::connect(reconnect_ctx)
        .await
        .expect("bob reconnects with the same identity");
    let mut sub = bob_back
        .subscribe(query.clone())
        .await
        .expect("bob resubscribes");

    written.insert(anchor_id);
    expect_delivered(&mut sub, &written, "bob after a long gap").await;

    bob_back.shutdown().await.ok();
    alice.shutdown().await.ok();
    server.shutdown().await;
}

/// A returning peer with two subscriptions gets the gap rows on BOTH of them.
///
/// The fast path is keyed per subscription, so a fix that invalidates only the first
/// subscription it happens to process leaves the second one serving a cached scope. The
/// second table is the whole point: one subscription cannot show this.
#[tokio::test]
async fn each_subscription_of_a_returning_peer_re_derives() {
    let schema = test_schema();
    let server = JazzServer::start_with_schema(schema.clone()).await;

    let alice = TestingClient::builder()
        .with_server(&server)
        .with_schema(schema.clone())
        .with_user_id("alice-two-subs")
        .ready_on("todos", READY_TIMEOUT)
        .connect()
        .await;

    let (bob_ctx, bob) = TestingClient::builder()
        .with_server(&server)
        .with_schema(schema.clone())
        .with_user_id("bob-two-subs")
        .with_persistent_storage()
        .ready_on("todos", READY_TIMEOUT)
        .connect_with_context()
        .await;
    let pinned_client_id = bob.client_id();

    let todos_query = QueryBuilder::new("todos").build();
    let notes_query = QueryBuilder::new("notes").build();

    let (todo_anchor, _, _) = alice.insert("todos", todo("anchor")).expect("insert");
    let (note_anchor, _, _) = alice.insert("notes", note("anchor")).expect("insert");
    let mut todos_before = bob
        .subscribe(todos_query.clone())
        .await
        .expect("bob subscribes to todos");
    let mut notes_before = bob
        .subscribe(notes_query.clone())
        .await
        .expect("bob subscribes to notes");
    expect_delivered(
        &mut todos_before,
        &[todo_anchor].into_iter().collect(),
        "bob's todos while online",
    )
    .await;
    expect_delivered(
        &mut notes_before,
        &[note_anchor].into_iter().collect(),
        "bob's notes while online",
    )
    .await;

    bob.shutdown().await.expect("bob goes offline");
    park_offline_unreaped(&server).await;

    let (gap_todo, _, _) = alice.insert("todos", todo("gap")).expect("insert");
    let (gap_note, _, _) = alice.insert("notes", note("gap")).expect("insert");
    wait_for_query(
        &alice,
        notes_query.clone(),
        Some(DurabilityTier::EdgeServer),
        QUERY_TIMEOUT,
        "alice's note reached the server",
        |rows| (rows.len() == 2).then_some(()),
    )
    .await;

    let mut reconnect_ctx = bob_ctx.clone();
    reconnect_ctx.client_id = pinned_client_id;
    let bob_back = JazzClient::connect(reconnect_ctx)
        .await
        .expect("bob reconnects");
    let mut todos_sub = bob_back
        .subscribe(todos_query)
        .await
        .expect("bob resubscribes to todos");
    let mut notes_sub = bob_back
        .subscribe(notes_query)
        .await
        .expect("bob resubscribes to notes");

    expect_delivered(
        &mut todos_sub,
        &[todo_anchor, gap_todo].into_iter().collect(),
        "bob's todos after returning",
    )
    .await;
    expect_delivered(
        &mut notes_sub,
        &[note_anchor, gap_note].into_iter().collect(),
        "bob's notes after returning",
    )
    .await;

    bob_back.shutdown().await.ok();
    alice.shutdown().await.ok();
    server.shutdown().await;
}

/// A row written during the gap is delivered once, not repeatedly.
///
/// Deferring the delivery record opens a window where the same batch can be queued twice
/// before the first one is confirmed. This is the test that would catch that: it counts
/// how many times the id shows up after it first arrives.
#[tokio::test]
async fn a_returning_peer_receives_a_gap_row_exactly_once() {
    let schema = test_schema();
    let server = JazzServer::start_with_schema(schema.clone()).await;

    let alice = TestingClient::builder()
        .with_server(&server)
        .with_schema(schema.clone())
        .with_user_id("alice-once")
        .ready_on("todos", READY_TIMEOUT)
        .connect()
        .await;

    let (bob_ctx, bob) = TestingClient::builder()
        .with_server(&server)
        .with_schema(schema.clone())
        .with_user_id("bob-once")
        .with_persistent_storage()
        .ready_on("todos", READY_TIMEOUT)
        .connect_with_context()
        .await;
    let pinned_client_id = bob.client_id();

    let query = QueryBuilder::new("todos").build();
    let (anchor_id, _, _) = alice.insert("todos", todo("anchor")).expect("insert");
    let mut before = bob.subscribe(query.clone()).await.expect("bob subscribes");
    expect_delivered(
        &mut before,
        &[anchor_id].into_iter().collect(),
        "bob while online",
    )
    .await;

    bob.shutdown().await.expect("bob goes offline");
    park_offline_unreaped(&server).await;

    // Inserted and never touched again, so any second appearance is a re-send and not an
    // update the peer was legitimately told about.
    let (gap_id, _, _) = alice.insert("todos", todo("gap-once")).expect("insert");
    wait_for_query(
        &alice,
        query.clone(),
        Some(DurabilityTier::EdgeServer),
        QUERY_TIMEOUT,
        "alice's write reached the server",
        |rows| (rows.len() == 2).then_some(()),
    )
    .await;

    let mut reconnect_ctx = bob_ctx.clone();
    reconnect_ctx.client_id = pinned_client_id;
    let bob_back = JazzClient::connect(reconnect_ctx)
        .await
        .expect("bob reconnects");
    let mut sub = bob_back.subscribe(query).await.expect("bob resubscribes");

    expect_delivered(
        &mut sub,
        &[anchor_id, gap_id].into_iter().collect(),
        "bob after returning",
    )
    .await;

    // Drain whatever else the server has to say for a while. The row must not come back.
    let mut extra = 0;
    let quiet_until = tokio::time::Instant::now() + Duration::from_secs(3);
    while let Ok(Some(delta)) = tokio::time::timeout(
        quiet_until.saturating_duration_since(tokio::time::Instant::now()),
        sub.next(),
    )
    .await
    {
        extra += delta.added.iter().filter(|row| row.id == gap_id).count();
        extra += delta.updated.iter().filter(|row| row.id == gap_id).count();
    }
    assert_eq!(
        extra, 0,
        "the gap row was delivered again after it had already arrived",
    );

    bob_back.shutdown().await.ok();
    alice.shutdown().await.ok();
    server.shutdown().await;
}

/// A peer that never disconnects keeps the fast path.
///
/// The fix invalidates the resubscribe fast path after a lost stream. Scoped wrongly — per
/// client instead of per subscription, or never cleared — it would also force a full
/// re-derive for every duplicate subscribe inside a live session, which is the cold-start
/// cost the fast path exists to avoid. This test does not measure that cost; it pins the
/// observable half: a second subscribe on a live connection still works and does not
/// disturb the first.
#[tokio::test]
async fn a_peer_that_never_left_can_resubscribe() {
    let schema = test_schema();
    let server = JazzServer::start_with_schema(schema.clone()).await;

    let alice = TestingClient::builder()
        .with_server(&server)
        .with_schema(schema.clone())
        .with_user_id("alice-live")
        .ready_on("todos", READY_TIMEOUT)
        .connect()
        .await;

    let bob = TestingClient::builder()
        .with_server(&server)
        .with_schema(schema.clone())
        .with_user_id("bob-live")
        .ready_on("todos", READY_TIMEOUT)
        .connect()
        .await;

    let query = QueryBuilder::new("todos").build();
    let (first_id, _, _) = alice.insert("todos", todo("first")).expect("insert");
    let mut first_sub = bob.subscribe(query.clone()).await.expect("first subscribe");
    expect_delivered(
        &mut first_sub,
        &[first_id].into_iter().collect(),
        "bob's first subscription",
    )
    .await;

    let mut second_sub = bob
        .subscribe(query.clone())
        .await
        .expect("second subscribe");
    expect_delivered(
        &mut second_sub,
        &[first_id].into_iter().collect(),
        "bob's second subscription on the same live connection",
    )
    .await;

    // The first subscription is still live and still fed.
    let (later_id, _, _) = alice.insert("todos", todo("later")).expect("insert");
    expect_delivered(
        &mut first_sub,
        &[later_id].into_iter().collect(),
        "bob's first subscription after a duplicate subscribe",
    )
    .await;

    bob.shutdown().await.ok();
    alice.shutdown().await.ok();
    server.shutdown().await;
}

/// A peer arriving for the first time gets everything, including rows written before it
/// existed.
///
/// This is the healthy path, and the fix touches per-client delivery state — breaking
/// initial catch-up while fixing the reconnect case would be a silent regression.
///
/// Note on a scenario that does NOT exist: "same store, fresh wire identity" is
/// unreachable. The client id is persisted with the peer's store, so clearing
/// `ctx.client_id` changes nothing — measured, after an assertion caught this test
/// reconnecting under the old id twice. That is also why a reinstall heals the field
/// defect: it drops the store and the identity together.
#[tokio::test]
async fn a_brand_new_peer_receives_everything_written_before_it_arrived() {
    let schema = test_schema();
    let server = JazzServer::start_with_schema(schema.clone()).await;

    let alice = TestingClient::builder()
        .with_server(&server)
        .with_schema(schema.clone())
        .with_user_id("alice-fresh-id")
        .ready_on("todos", READY_TIMEOUT)
        .connect()
        .await;

    // Two rows exist before bob has ever connected, one of them written while another peer
    // was parked offline — so the server's per-client bookkeeping is already in play when
    // the newcomer arrives.
    let query = QueryBuilder::new("todos").build();
    let (anchor_id, _, _) = alice.insert("todos", todo("anchor")).expect("insert");

    let (_, parked) = TestingClient::builder()
        .with_server(&server)
        .with_schema(schema.clone())
        .with_user_id("parked-peer")
        .with_persistent_storage()
        .ready_on("todos", READY_TIMEOUT)
        .connect_with_context()
        .await;
    let mut parked_sub = parked.subscribe(query.clone()).await.expect("subscribe");
    expect_delivered(
        &mut parked_sub,
        &[anchor_id].into_iter().collect(),
        "the parked peer while online",
    )
    .await;
    parked.shutdown().await.expect("the peer goes offline");
    park_offline_unreaped(&server).await;

    let (gap_id, _, _) = alice.insert("todos", todo("gap")).expect("insert");
    wait_for_query(
        &alice,
        query.clone(),
        Some(DurabilityTier::EdgeServer),
        QUERY_TIMEOUT,
        "alice's write reached the server",
        |rows| (rows.len() == 2).then_some(()),
    )
    .await;

    let newcomer = TestingClient::builder()
        .with_server(&server)
        .with_schema(schema.clone())
        .with_user_id("newcomer")
        .ready_on("todos", READY_TIMEOUT)
        .connect()
        .await;
    let mut sub = newcomer
        .subscribe(query)
        .await
        .expect("newcomer subscribes");
    expect_delivered(
        &mut sub,
        &[anchor_id, gap_id].into_iter().collect(),
        "a peer connecting for the first time",
    )
    .await;

    newcomer.shutdown().await.ok();
    alice.shutdown().await.ok();
    server.shutdown().await;
}
