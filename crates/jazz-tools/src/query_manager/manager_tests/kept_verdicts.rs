//! A write re-authorizes what it can have changed, not everything its subscriptions hold.
//!
//! A subscription with a session is filtered by the select policies after every settle:
//! each row its tuples are made of was loaded from storage and checked against its policy,
//! for every subscription the write dirtied, on every pass. On the phone that was half of
//! what the presence write cost every ten seconds (profiled 2026-10-02: 128 of 310 engine
//! samples per beat, 82 of them the row loads), and on the server it is paid for every
//! write into a table a live subscription reads.
//!
//! Two things make a verdict safe to reuse, and the gates below pin each:
//!
//! * storage does not change under a settle pass, so a verdict reached for one
//!   subscription holds for the next one in the same pass;
//! * a table whose select policy is constant allows a row for what the row is — being
//!   there — so an allowed verdict holds until the row itself changes.
//!
//! The cross-tick cache of every verdict stays switched off (`JAZZ_AUTHZ_CACHE_ENABLE`);
//! the stands here say so explicitly, because another test in the process may have
//! switched it on.

use super::*;
use crate::query_manager::graph_nodes::output::QuerySubscriptionId;
use crate::query_manager::types::SchemaHash;
use crate::storage::SqliteStorage;

const NOTES: usize = 40;

fn notes() -> RowDescriptor {
    RowDescriptor::new(vec![
        ColumnDescriptor::new("owner_id", ColumnType::Text),
        ColumnDescriptor::new("text", ColumnType::Text),
    ])
}

/// What the client compiles queries against: no policies, as on the phone.
fn structure() -> Schema {
    let mut schema = Schema::new();
    schema.insert(TableName::new("notes"), notes().into());
    schema
}

fn permissions(select: PolicyExpr) -> Schema {
    let mut schema = Schema::new();
    schema.insert(
        TableName::new("notes"),
        TableSchema::with_policies(notes(), TablePolicies::new().with_select(select)),
    );
    schema
}

fn own_notes() -> PolicyExpr {
    PolicyExpr::eq_session("owner_id", vec!["user_id".into()])
}

#[derive(Debug)]
struct Verdicts {
    evaluated: u64,
    kept: u64,
}

struct Stand<S: Storage> {
    qm: QueryManager,
    storage: S,
    ids: Vec<ObjectId>,
}

/// A runtime over `select`, with the cross-tick switch off whatever the process says.
fn manager(select: PolicyExpr) -> QueryManager {
    let mut qm = QueryManager::new(SyncManager::new());
    qm.set_current_schema(structure(), "dev", "main");
    qm.set_authorization_schema(permissions(select));
    qm.authz_verdicts.keeps_every_verdict = Some(false);
    qm
}

fn seeded<S: Storage>(mut qm: QueryManager, mut storage: S) -> Stand<S> {
    let ids = (0..NOTES)
        .map(|index| {
            qm.insert(
                &mut storage,
                "notes",
                &[
                    Value::Text("alice".into()),
                    Value::Text(format!("note {index:02}")),
                ],
            )
            .expect("insert note")
            .row_id
        })
        .collect();
    qm.process(&mut storage);
    Stand { qm, storage, ids }
}

/// `NOTES` notes of alice's under `select`, settled.
fn stand(select: PolicyExpr) -> Stand<CountingCatalogueUpsertsStorage> {
    let qm = manager(select);
    let inner = seeded_memory_storage(&qm.schema_context().current_schema);
    seeded(qm, CountingCatalogueUpsertsStorage::with_inner(inner))
}

/// The same on the backend the phone and the server keep their stores in.
fn sqlite_stand(select: PolicyExpr) -> Stand<SqliteStorage> {
    let qm = manager(select);
    let mut storage = SqliteStorage::open(":memory:").expect("in-memory sqlite storage");
    crate::test_support::persist_test_schema(&mut storage, &qm.schema_context().current_schema);
    seeded(qm, storage)
}

impl<S: Storage> Stand<S> {
    fn subscribe(&mut self, user: &str) -> QuerySubscriptionId {
        let query = self.qm.query("notes").build();
        let sub = self
            .qm
            .subscribe_with_session(query, Some(PolicySession::new(user)), None)
            .expect("subscribe");
        self.qm.process(&mut self.storage);
        sub
    }

    fn texts(&self, sub: QuerySubscriptionId) -> Vec<String> {
        let mut texts: Vec<String> = self
            .qm
            .get_subscription_results(sub)
            .into_iter()
            .map(|(_, values)| match &values[1] {
                Value::Text(text) => text.clone(),
                other => panic!("a note's text is {other:?}"),
            })
            .collect();
        texts.sort();
        texts
    }

    /// The verdicts the engine had to evaluate while `act` ran — a row load and a
    /// policy evaluation each — and the ones it had kept.
    ///
    /// Storage loads are not the measure here: a debug build re-evaluates every kept
    /// verdict it serves, to compare the two.
    fn verdicts_of<T>(&mut self, act: impl FnOnce(&mut Self) -> T) -> (T, Verdicts) {
        let (evaluated, kept) = (
            self.qm.authz_cache_miss_count(),
            self.qm.authz_cache_hit_count(),
        );
        let result = act(self);
        let verdicts = Verdicts {
            evaluated: self.qm.authz_cache_miss_count() - evaluated,
            kept: self.qm.authz_cache_hit_count() - kept,
        };
        (result, verdicts)
    }

    /// Rewrites one note and settles.
    fn rewrite(&mut self, index: usize, text: &str) -> Verdicts {
        self.verdicts_of(|stand| {
            stand
                .qm
                .update(
                    &mut stand.storage,
                    stand.ids[index],
                    &[Value::Text("alice".into()), Value::Text(text.into())],
                )
                .expect("update note");
            stand.qm.process(&mut stand.storage);
        })
        .1
    }

    /// The runtime is started again over the store: nothing it kept in memory is there.
    fn restarted(self, select: PolicyExpr) -> Self {
        Stand {
            qm: manager(select),
            storage: self.storage,
            ids: self.ids,
        }
    }
}

fn all_notes_with(index: usize, text: &str) -> Vec<String> {
    let mut texts: Vec<String> = (0..NOTES)
        .map(|note| {
            if note == index {
                text.to_string()
            } else {
                format!("note {note:02}")
            }
        })
        .collect();
    texts.sort();
    texts
}

fn all_notes_but(index: usize) -> Vec<String> {
    let mut texts = all_notes_with(index, "\u{0}");
    texts.retain(|text| text != "\u{0}");
    texts
}

fn a_write_to_one_row_checks_that_row<S: Storage>(mut stand: Stand<S>) {
    let sub = stand.subscribe("alice");
    assert_eq!(stand.texts(sub), all_notes_with(NOTES, ""), "fixture");

    let verdicts = stand.rewrite(7, "rewritten");

    assert_eq!(stand.texts(sub), all_notes_with(7, "rewritten"));
    assert_eq!(
        verdicts.evaluated, 1,
        "a write to one of {NOTES} rows the subscription holds: {verdicts:?}"
    );
    assert_eq!(verdicts.kept, (NOTES - 1) as u64, "{verdicts:?}");
}

/// Under a constant policy a row is allowed for being there, and the write changed one
/// row: that one is checked again, the other thirty-nine are not loaded.
#[test]
fn a_write_to_one_row_of_a_constant_policy_table_does_not_load_the_others_to_authorize_them() {
    a_write_to_one_row_checks_that_row(stand(PolicyExpr::True));
}

#[test]
fn a_write_to_one_row_of_a_constant_policy_table_checks_that_row_on_sqlite() {
    a_write_to_one_row_checks_that_row(sqlite_stand(PolicyExpr::True));
}

/// The same write again: the verdicts that were kept are still the right ones, and the
/// row that changed twice was checked twice.
#[test]
fn a_row_under_a_constant_policy_is_checked_again_each_time_it_changes() {
    let mut stand = stand(PolicyExpr::True);
    let sub = stand.subscribe("alice");

    stand.rewrite(7, "once");
    let verdicts = stand.rewrite(7, "twice");

    assert_eq!(stand.texts(sub), all_notes_with(7, "twice"));
    assert_eq!(
        verdicts.evaluated, 1,
        "the row that changed is the one checked again: {verdicts:?}"
    );
    assert_eq!(verdicts.kept, (NOTES - 1) as u64, "{verdicts:?}");
}

/// A constant policy does not read who asks. On a server every client that holds a row
/// is another session over the same row: the first one's check serves them all.
#[test]
fn one_check_of_a_constant_policy_row_serves_every_session() {
    let mut stand = stand(PolicyExpr::True);
    let alice = stand.subscribe("alice");

    let (bob, verdicts) = stand.verdicts_of(|stand| stand.subscribe("bob"));

    assert_eq!(stand.texts(bob), stand.texts(alice));
    assert_eq!(
        verdicts.evaluated, 0,
        "bob's first settle over rows alice's had checked: {verdicts:?}"
    );
    assert_eq!(verdicts.kept, NOTES as u64, "{verdicts:?}");
}

/// A policy that reads the row keeps nothing across ticks, but one pass settles every
/// subscription over the same storage: four subscriptions cost the checks of one.
#[test]
fn one_pass_checks_a_row_once_for_every_subscription_that_holds_it() {
    const SUBSCRIPTIONS: usize = 4;

    let mut stand = stand(own_notes());
    let subs: Vec<_> = (0..SUBSCRIPTIONS)
        .map(|_| stand.subscribe("alice"))
        .collect();

    let verdicts = stand.rewrite(7, "rewritten");

    for sub in subs {
        assert_eq!(stand.texts(sub), all_notes_with(7, "rewritten"));
    }
    assert_eq!(
        verdicts.evaluated, NOTES as u64,
        "{SUBSCRIPTIONS} subscriptions over {NOTES} rows, one pass: {verdicts:?}"
    );
    assert_eq!(
        verdicts.kept,
        ((SUBSCRIPTIONS - 1) * NOTES) as u64,
        "{verdicts:?}"
    );
    assert!(
        !stand.qm.authz_verdicts.pass_is_open(),
        "the pass was left open: what is reached before the next one would be kept as its own"
    );
    assert_eq!(
        stand.qm.authz_verdicts.kept_verdict_count() + stand.qm.authz_verdicts.pass_verdict_count(),
        0,
        "verdicts of a policy that reads the row outlived the pass"
    );
}

/// What a pass reached for one session is not what another session is told: bob's
/// subscription settles in the same pass as alice's, over the same rows, and stays empty —
/// whichever of the two settles first.
#[test]
fn a_pass_does_not_serve_one_sessions_verdict_to_another() {
    for (first, second) in [("alice", "bob"), ("bob", "alice")] {
        let mut stand = stand(own_notes());
        let first_sub = stand.subscribe(first);
        let second_sub = stand.subscribe(second);

        stand.rewrite(7, "rewritten");

        for (user, sub) in [(first, first_sub), (second, second_sub)] {
            let expected = if user == "alice" {
                all_notes_with(7, "rewritten")
            } else {
                Vec::new()
            };
            assert_eq!(
                stand.texts(sub),
                expected,
                "{user}'s notes, subscribed {first} then {second}"
            );
        }
    }
}

/// A verdict of a pass does not outlive it: the note changes hands in one tick, and the
/// next settle checks it for what it holds now.
#[test]
fn a_row_that_changes_hands_leaves_the_subscription_of_the_one_who_lost_it() {
    let mut stand = stand(own_notes());
    let alice = stand.subscribe("alice");
    let bob = stand.subscribe("bob");
    stand.rewrite(7, "rewritten");

    stand
        .qm
        .update(
            &mut stand.storage,
            stand.ids[7],
            &[Value::Text("bob".into()), Value::Text("bob's now".into())],
        )
        .expect("hand the note over");
    stand.qm.process(&mut stand.storage);

    assert_eq!(stand.texts(alice), all_notes_but(7));
    assert_eq!(stand.texts(bob), vec!["bob's now".to_string()]);
}

/// Kept verdicts belong to the permissions they were reached under: when a constant policy
/// is replaced by one that reads the row, every row is checked against the new one.
#[test]
fn republished_permissions_check_every_row_again() {
    let mut stand = stand(PolicyExpr::True);
    let bob = stand.subscribe("bob");
    stand.rewrite(7, "rewritten");
    assert_eq!(
        stand.texts(bob),
        all_notes_with(7, "rewritten"),
        "fixture: everyone reads every note"
    );

    stand.qm.set_authorization_schema(permissions(own_notes()));
    stand.qm.authz_verdicts.keeps_every_verdict = Some(false);
    stand.qm.process(&mut stand.storage);

    assert_eq!(
        stand.texts(bob),
        Vec::<String>::new(),
        "bob still reads notes the new permissions give to their owner alone"
    );
}

/// A row is transformed through an authorization context before its policy reads it, and
/// a lens or a schema generation rebuilds the contexts without the permissions moving:
/// what was reached through the old ones is checked again.
#[test]
fn rebuilt_authorization_contexts_check_every_row_again() {
    let mut stand = stand(PolicyExpr::True);
    let sub = stand.subscribe("alice");
    let settled = stand.rewrite(7, "once");
    assert_eq!(settled.evaluated, 1, "fixture: {settled:?}");

    stand.qm.forget_authorization_contexts();
    let verdicts = stand.rewrite(7, "twice");

    assert_eq!(stand.texts(sub), all_notes_with(7, "twice"));
    assert_eq!(
        verdicts.evaluated, NOTES as u64,
        "verdicts reached through contexts that were thrown away were served: {verdicts:?}"
    );
    let again = stand.rewrite(7, "thrice");
    assert_eq!(again.evaluated, 1, "{again:?}");
}

/// Another generation of the family the stands run on: `notes` with one more column.
fn next_generation() -> Schema {
    let mut schema = Schema::new();
    schema.insert(
        TableName::new("notes"),
        RowDescriptor::new(vec![
            ColumnDescriptor::new("owner_id", ColumnType::Text),
            ColumnDescriptor::new("text", ColumnType::Text),
            ColumnDescriptor::new("pinned", ColumnType::Boolean),
        ])
        .into(),
    );
    schema
}

fn lens_to_the_next_generation() -> crate::schema_manager::lens::Lens {
    use crate::schema_manager::lens::{Lens, LensOp, LensTransform};

    let mut transform = LensTransform::new();
    transform.push(
        LensOp::AddColumn {
            table: "notes".to_string(),
            column: "pinned".to_string(),
            column_type: ColumnType::Boolean,
            default: Value::Boolean(false),
        },
        false,
    );
    Lens::new(
        SchemaHash::compute(&structure()),
        SchemaHash::compute(&next_generation()),
        transform,
    )
}

/// Each way the runtime has of throwing its authorization contexts away, by the entry
/// point that does it: after any of them, what was kept is checked again. The gate above
/// pins what forgetting does; this one, that nobody who must forget has stopped.
#[test]
fn every_way_of_rebuilding_authorization_contexts_checks_every_row_again() {
    // The sixth, setting the current schema, happens once, before a row is read.
    let ways: [(&str, fn(&mut QueryManager)); 5] = [
        ("the permissions are published again", |qm| {
            qm.set_authorization_schema(permissions(PolicyExpr::True));
        }),
        ("permissions become required", |qm| {
            qm.require_authorization_schema();
        }),
        ("a schema generation goes live", |qm| {
            qm.add_live_schema(next_generation());
        }),
        ("a lens is registered", |qm| {
            qm.register_lens(lens_to_the_next_generation());
        }),
        ("the known schemas are replaced", |qm| {
            qm.set_known_schemas(std::sync::Arc::new(HashMap::from([(
                SchemaHash::compute(&next_generation()),
                next_generation(),
            )])));
        }),
    ];
    for (way, rebuild) in ways {
        let mut stand = stand(PolicyExpr::True);
        let sub = stand.subscribe("alice");
        let settled = stand.rewrite(7, "once");
        assert_eq!(settled.evaluated, 1, "fixture, {way}: {settled:?}");

        rebuild(&mut stand.qm);
        stand.qm.authz_verdicts.keeps_every_verdict = Some(false);
        let verdicts = stand.rewrite(7, "twice");

        assert_eq!(stand.texts(sub), all_notes_with(7, "twice"), "{way}");
        assert!(
            verdicts.evaluated >= NOTES as u64 && verdicts.kept == 0,
            "{way}: verdicts reached through contexts that were thrown away were served: \
             {verdicts:?}"
        );
        let again = stand.rewrite(7, "thrice");
        assert_eq!(again.evaluated, 1, "{way}: {again:?}");
    }
}

/// A runtime nobody reads a verdict in — no subscription with a session — is still told
/// of every row a rejection withdraws. What it is told does not wait for a read that
/// never comes.
#[test]
fn rows_withdrawn_from_a_runtime_that_reads_no_verdict_do_not_pile_up() {
    let mut stand = stand(PolicyExpr::True);
    for round in 0..3 {
        for id in stand.ids.clone() {
            stand.qm.sync_manager_mut().row_withdrawn(id);
        }
        stand.qm.process(&mut stand.storage);
        let waiting = stand.qm.sync_manager_mut().take_pending_row_withdrawals();
        assert!(
            waiting.is_empty(),
            "round {round}: {} withdrawn rows still wait to be forgotten",
            waiting.len()
        );
    }
}

fn a_deleted_row_leaves<S: Storage>(
    mut stand: Stand<S>,
    delete: impl FnOnce(&mut QueryManager, &mut S, ObjectId),
) {
    let alice = stand.subscribe("alice");
    let bob = stand.subscribe("bob");
    stand.rewrite(7, "rewritten");

    delete(&mut stand.qm, &mut stand.storage, stand.ids[7]);
    stand.qm.process(&mut stand.storage);

    assert_eq!(stand.texts(alice), all_notes_but(7));
    assert_eq!(stand.texts(bob), all_notes_but(7));
}

/// A deleted row is gone for everyone, kept verdict or not.
#[test]
fn a_deleted_row_under_a_constant_policy_leaves_the_subscription() {
    a_deleted_row_leaves(stand(PolicyExpr::True), |qm, storage, id| {
        qm.delete(storage, id).expect("delete note");
    });
    a_deleted_row_leaves(sqlite_stand(PolicyExpr::True), |qm, storage, id| {
        qm.delete(storage, id).expect("delete note");
    });
}

#[test]
fn a_hard_deleted_row_under_a_constant_policy_leaves_the_subscription() {
    a_deleted_row_leaves(stand(PolicyExpr::True), |qm, storage, id| {
        qm.hard_delete(storage, id).expect("hard delete note");
    });
    a_deleted_row_leaves(sqlite_stand(PolicyExpr::True), |qm, storage, id| {
        qm.hard_delete(storage, id).expect("hard delete note");
    });
}

#[test]
fn a_truncated_row_under_a_constant_policy_leaves_the_subscription() {
    a_deleted_row_leaves(sqlite_stand(PolicyExpr::True), |qm, storage, id| {
        qm.delete(storage, id).expect("delete note");
        qm.process(storage);
        qm.truncate(storage, id).expect("truncate note");
    });
}

/// Verdicts are kept in memory: a runtime started again over its store checks what it
/// serves, and keeps from there.
#[test]
fn a_runtime_started_again_over_its_store_checks_every_row_it_serves() {
    let mut stand = sqlite_stand(PolicyExpr::True);
    stand.subscribe("alice");
    stand.rewrite(7, "rewritten");

    let mut stand = stand.restarted(PolicyExpr::True);
    let (sub, first) = stand.verdicts_of(|stand| stand.subscribe("alice"));
    assert_eq!(stand.texts(sub), all_notes_with(7, "rewritten"));
    assert_eq!(first.evaluated, NOTES as u64, "{first:?}");
    assert_eq!(first.kept, 0, "{first:?}");

    let verdicts = stand.rewrite(9, "after the restart");
    assert_eq!(verdicts.evaluated, 1, "{verdicts:?}");
}

/// A rejected batch can take a row out of sight and leave nothing in its place. No
/// visibility change is published for it — there is no row to publish — and a runtime
/// started again since the row arrived holds none of the bookkeeping that would name it.
/// Before verdicts were kept, loading the row to authorize it was what found it gone.
fn the_only_version_of_a_row_is_rejected(bypassed: bool) -> (Vec<String>, Vec<String>) {
    use crate::row_histories::{RowState, patch_row_batch_state};

    let stand = sqlite_stand(PolicyExpr::True);
    let mut stand = stand.restarted(PolicyExpr::True);
    stand.qm.bypass_authz_verdicts_for_tests(bypassed);
    let sub = stand.subscribe("alice");
    let before = stand.texts(sub);

    let gone = stand.ids[7];
    let batch_id = stand
        .storage
        .scan_history_row_batches("notes", gone)
        .expect("history of the note")[0]
        .batch_id;
    let branch = stand.qm.schema_context().branch_name();
    let change = patch_row_batch_state(
        &mut stand.storage,
        gone,
        &branch,
        batch_id,
        Some(RowState::Rejected),
        None,
    )
    .expect("reject the note's only version");
    assert!(
        change.is_none(),
        "fixture: a row that is gone has no visibility change to publish"
    );
    // What the sync manager does where it rejects a batch.
    stand.qm.sync_manager_mut().row_withdrawn(gone);
    stand
        .qm
        .sync_manager_mut()
        .push_pending_batch_fate(crate::batch_fate::BatchFate::Rejected {
            batch_id,
            code: "permission_denied".to_string(),
            reason: "writer lacks publish rights".to_string(),
        });
    stand.qm.process(&mut stand.storage);

    (before, stand.texts(sub))
}

#[test]
fn a_row_whose_only_version_is_rejected_leaves_the_subscription() {
    // Without kept verdicts the row leaves: that is what must not change.
    let (before, after) = the_only_version_of_a_row_is_rejected(true);
    assert_eq!(before, all_notes_with(NOTES, ""), "fixture");
    assert_eq!(after, all_notes_but(7), "fixture: every verdict evaluated");

    let (before, after) = the_only_version_of_a_row_is_rejected(false);
    assert_eq!(before, all_notes_with(NOTES, ""), "fixture");
    assert_eq!(
        after,
        all_notes_but(7),
        "a row that is gone from the store is still served: its verdict was kept"
    );
}

/// A binding catches a panic of the engine and calls it again. The pass the panic cut
/// short must be over before that call does anything: what it reaches ahead of its own
/// pass is reached over storage it is about to write.
#[test]
fn a_pass_cut_short_by_a_panic_is_over_when_the_runtime_is_called_again() {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    let mut stand = stand(own_notes());
    stand.subscribe("alice");

    stand.storage.panic_at_visible_query_load(NOTES / 2);
    let cut_short = catch_unwind(AssertUnwindSafe(|| stand.rewrite(7, "rewritten")));
    assert!(cut_short.is_err(), "fixture: the settle pass panicked");
    assert!(
        stand.qm.authz_verdicts.pass_is_open(),
        "fixture: the panic left the pass open"
    );

    stand.qm.process(&mut stand.storage);

    assert_eq!(
        stand.qm.authz_verdicts.passes_begun_over_an_open_one, 0,
        "the runtime ran up to its next pass inside the one the panic cut short"
    );
    assert!(!stand.qm.authz_verdicts.pass_is_open());
    assert_eq!(
        stand.qm.authz_verdicts.kept_verdict_count() + stand.qm.authz_verdicts.pass_verdict_count(),
        0,
        "verdicts of the pass the panic cut short are still held"
    );
    // The subscription that was settling when the engine panicked is the binding's to
    // recover; one made now reads every row.
    let (again, verdicts) = stand.verdicts_of(|stand| stand.subscribe("alice"));
    assert_eq!(stand.texts(again).len(), NOTES);
    assert_eq!(verdicts.evaluated, NOTES as u64, "{verdicts:?}");
}

/// The limit is reached in the middle of a pass and in the middle of a walk: rows stay
/// right, the pass's verdicts are forgotten, and no more than the limit is held.
#[test]
fn a_cache_at_its_limit_serves_the_same_rows() {
    const LIMIT: usize = 16;

    for select in [PolicyExpr::True, own_notes()] {
        let mut stand = stand(select.clone());
        stand.qm.authz_verdicts.max_kept_verdicts = Some(LIMIT);
        let alice = stand.subscribe("alice");
        let second = stand.subscribe("alice");
        let bob = stand.subscribe("bob");

        for round in 0..3 {
            let text = format!("round {round}");
            stand.rewrite(7, &text);
            assert_eq!(stand.texts(alice), all_notes_with(7, &text), "{select:?}");
            assert_eq!(stand.texts(second), all_notes_with(7, &text), "{select:?}");
            let bobs = if matches!(select, PolicyExpr::True) {
                all_notes_with(7, &text)
            } else {
                Vec::new()
            };
            assert_eq!(stand.texts(bob), bobs, "{select:?}");
            let held = stand.qm.authz_verdicts.kept_verdict_count();
            assert!(
                held <= LIMIT,
                "{held} verdicts held under a limit of {LIMIT}"
            );
            if !matches!(select, PolicyExpr::True) {
                assert_eq!(held, 0, "verdicts of a pass outlived it at the limit");
            }
        }
    }
}

/// A stand, not a gate: what a write costs a runtime that serves many sessions the same
/// rows, with every verdict evaluated (as before verdicts were kept) and with them kept.
/// `cargo test --release ... -- --ignored many_sessions --nocapture`; a debug build
/// evaluates every verdict it serves and so measures nothing.
#[test]
#[ignore = "stand: prints timings"]
fn stand_many_sessions_over_one_window() {
    const ROWS: usize = 2_000;
    const SESSIONS: usize = 50;

    for bypassed in [true, false] {
        let mut qm = manager(PolicyExpr::True);
        let mut storage = SqliteStorage::open(":memory:").expect("in-memory sqlite storage");
        crate::test_support::persist_test_schema(&mut storage, &qm.schema_context().current_schema);
        let ids: Vec<ObjectId> = (0..ROWS)
            .map(|index| {
                qm.insert(
                    &mut storage,
                    "notes",
                    &[
                        Value::Text("alice".into()),
                        Value::Text(format!("note {index:04}")),
                    ],
                )
                .expect("insert note")
                .row_id
            })
            .collect();
        qm.process(&mut storage);
        qm.bypass_authz_verdicts_for_tests(bypassed);
        for session in 0..SESSIONS {
            let query = qm.query("notes").build();
            qm.subscribe_with_session(
                query,
                Some(PolicySession::new(format!("user-{session}"))),
                None,
            )
            .expect("subscribe");
        }
        let started = web_time::Instant::now();
        qm.process(&mut storage);
        let first_settle = started.elapsed();

        let mut one_row = Vec::new();
        for round in 0..20 {
            qm.update(
                &mut storage,
                ids[round * 7],
                &[
                    Value::Text("alice".into()),
                    Value::Text(format!("round {round}")),
                ],
            )
            .expect("update note");
            let started = web_time::Instant::now();
            qm.process(&mut storage);
            one_row.push(started.elapsed());
        }
        one_row.sort();

        for (index, id) in ids.iter().enumerate() {
            qm.update(
                &mut storage,
                *id,
                &[
                    Value::Text("alice".into()),
                    Value::Text(format!("all {index:04}")),
                ],
            )
            .expect("update note");
        }
        let started = web_time::Instant::now();
        qm.process(&mut storage);
        let every_row = started.elapsed();

        println!(
            "{SESSIONS} sessions x {ROWS} rows, verdicts {}: first settle {first_settle:?}, \
             one row written p50 {:?} max {:?}, every row written {every_row:?}, \
             evaluated {} kept {}",
            if bypassed { "evaluated" } else { "kept" },
            one_row[one_row.len() / 2],
            one_row[one_row.len() - 1],
            qm.authz_cache_miss_count(),
            qm.authz_cache_hit_count(),
        );
    }
}
