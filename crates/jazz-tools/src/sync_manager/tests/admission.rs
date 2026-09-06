//! Admission control at registration (v18 item 2), at the level the check lives on.
//!
//! Internal on purpose: the caps over the relation IR guard shapes the public query builders
//! cannot produce (a `Gather` with a ten-million-step depth, a union of seventeen scans, a
//! nine-way join), but the wire accepts them from any client — `translateQuery` ships the IR
//! verbatim and the server compiles what it receives. These gates hand the sync manager the
//! same frames a hostile client would send and look at what it leaves behind: nothing, and a
//! rejection on the outbox. The black-box half (caps the builders can reach, per-principal
//! counts, the rate window, the global ceiling) lives in `tests/admission_control.rs`.

use super::*;
use crate::query_manager::relation_ir::{
    ColumnRef, JoinCondition, JoinKind, KeyRef, PredicateExpr, RelExpr, RowIdRef, ValueRef,
};
use crate::sync_manager::admission::{Admission, Principal, Registration, SubscriptionCaps};
use crate::sync_manager::wire_depth::WIRE_MAX_NESTING;
use crate::sync_manager::{InboxEntry, Source, SyncError};
use std::time::Duration;

fn scan(table: &str) -> RelExpr {
    RelExpr::TableScan {
        table: table.into(),
    }
}

fn wire_query(relation_ir: RelExpr) -> Query {
    let mut query = QueryBuilder::new("users").build();
    query.relation_ir = relation_ir;
    query
}

fn subscription(client_id: ClientId, query_id: QueryId, query: Query) -> InboxEntry {
    InboxEntry {
        source: Source::Client(client_id),
        payload: SyncPayload::QuerySubscription {
            query_id,
            query: Box::new(query),
            session: Some(Session::new("alice")),
            required_tier: None,
            propagation: QueryPropagation::Full,
            policy_context_tables: vec![],
        },
    }
}

/// A registration as the admission sees it, with nothing but the query set.
fn reg(query: &Query) -> Registration<'_> {
    Registration {
        query,
        session: None,
        required_tier: None,
        propagation: QueryPropagation::Full,
        policy_context_tables: &[],
    }
}

/// What the server holds for `(client, query)` after the frame was processed, and the
/// rejection it sent, if any.
struct Aftermath {
    rejected_with: Option<(String, String)>,
    parked: bool,
    origin_recorded: bool,
    admitted: bool,
}

fn register(sm: &mut SyncManager, io: &mut MemoryStorage, query: Query) -> Aftermath {
    let client_id = ClientId::new();
    add_client(sm, io, client_id);
    let query_id = QueryId(41);
    sm.push_inbox(subscription(client_id, query_id, query));
    sm.process_inbox(io);
    let rejected_with = sm.outbox.iter().find_map(|entry| match &entry.payload {
        SyncPayload::Error(SyncError::QuerySubscriptionRejected {
            query_id: rejected,
            code,
            reason,
        }) if *rejected == query_id && entry.destination == Destination::Client(client_id) => {
            Some((code.clone(), reason.clone()))
        }
        _ => None,
    });
    Aftermath {
        rejected_with,
        parked: sm
            .pending_query_subscriptions
            .iter()
            .any(|pending| pending.client_id == client_id && pending.query_id == query_id),
        origin_recorded: sm
            .query_origin
            .get(&query_id)
            .is_some_and(|clients| clients.contains(&client_id)),
        admitted: sm.admission.holds(client_id, query_id),
    }
}

fn assert_refused(aftermath: &Aftermath, cap_text: &str) {
    let (code, reason) = aftermath
        .rejected_with
        .as_ref()
        .expect("the registration must be answered with a rejection frame");
    assert_eq!(code, "subscription_over_cap");
    assert!(
        reason.contains(cap_text),
        "the reason must name the cap: got {reason:?}, wanted {cap_text:?}"
    );
    assert!(
        !aftermath.parked,
        "a refused registration must not be parked for the pass"
    );
    assert!(
        !aftermath.origin_recorded,
        "a refused registration must leave no query_origin entry"
    );
    assert!(
        !aftermath.admitted,
        "a refused registration must not be counted"
    );
}

fn assert_admitted(aftermath: &Aftermath) {
    assert!(
        aftermath.rejected_with.is_none(),
        "unexpected rejection: {:?}",
        aftermath.rejected_with
    );
    assert!(
        aftermath.parked,
        "an admitted registration is parked for the pass"
    );
    assert!(aftermath.admitted, "an admitted registration is counted");
}

fn row_id_key() -> KeyRef {
    KeyRef::RowId(RowIdRef::Current)
}

fn gather(max_depth: usize) -> RelExpr {
    RelExpr::Gather {
        seed: Box::new(scan("users")),
        step: Box::new(scan("users")),
        frontier_key: row_id_key(),
        max_depth,
        dedupe_key: vec![row_id_key()],
    }
}

#[test]
fn a_gather_deeper_than_the_cap_is_refused_before_any_work() {
    let mut sm = SyncManager::new().with_subscription_caps(SubscriptionCaps::default());
    let mut io = MemoryStorage::new();
    seed_users_schema(&mut io);

    // `normalize_gather_depth` has no caller on the wire path; the planner would take this
    // depth as written.
    let refused = register(&mut sm, &mut io, wire_query(gather(10_000_000)));
    assert_refused(&refused, "gather depth");

    let at_cap = sm.subscription_caps().max_gather_depth;
    let admitted = register(&mut sm, &mut io, wire_query(gather(at_cap)));
    assert_admitted(&admitted);
}

#[test]
fn a_union_wider_than_the_cap_is_refused() {
    let mut sm = SyncManager::new().with_subscription_caps(SubscriptionCaps::default());
    let mut io = MemoryStorage::new();
    seed_users_schema(&mut io);
    let cap = sm.subscription_caps().max_union_arity;

    let refused = register(
        &mut sm,
        &mut io,
        wire_query(RelExpr::Union {
            inputs: (0..cap + 1).map(|_| scan("users")).collect(),
        }),
    );
    assert_refused(&refused, "union arity");

    let admitted = register(
        &mut sm,
        &mut io,
        wire_query(RelExpr::Union {
            inputs: (0..cap).map(|_| scan("users")).collect(),
        }),
    );
    assert_admitted(&admitted);
}

fn join_chain(joins: usize) -> RelExpr {
    let mut expr = scan("users");
    for _ in 0..joins {
        expr = RelExpr::Join {
            left: Box::new(expr),
            right: Box::new(scan("users")),
            on: vec![JoinCondition {
                left: ColumnRef {
                    scope: None,
                    column: "id".into(),
                },
                right: ColumnRef {
                    scope: None,
                    column: "id".into(),
                },
            }],
            join_kind: JoinKind::Inner,
        };
    }
    expr
}

#[test]
fn more_joins_than_the_cap_are_refused() {
    let mut sm = SyncManager::new().with_subscription_caps(SubscriptionCaps::default());
    let mut io = MemoryStorage::new();
    seed_users_schema(&mut io);
    let cap = sm.subscription_caps().max_ir_joins;

    let refused = register(&mut sm, &mut io, wire_query(join_chain(cap + 1)));
    assert_refused(&refused, "joins");

    let admitted = register(&mut sm, &mut io, wire_query(join_chain(cap)));
    assert_admitted(&admitted);
}

#[test]
fn a_relation_tree_with_more_nodes_than_the_cap_is_refused() {
    let mut sm = SyncManager::new().with_subscription_caps(SubscriptionCaps::default());
    let mut io = MemoryStorage::new();
    seed_users_schema(&mut io);
    let cap = sm.subscription_caps().max_ir_nodes;

    // Wide and shallow, so only the node count is over: a union at the arity cap whose
    // inputs are short `Distinct` chains (one node each, no joins), well inside the nesting
    // bound — a single chain of `cap` nodes would trip that bound first.
    let arity = sm.subscription_caps().max_union_arity;
    let chain = |nodes: usize| {
        let mut expr = scan("users");
        for _ in 1..nodes {
            expr = RelExpr::Distinct {
                input: Box::new(expr),
                key: vec![row_id_key()],
            };
        }
        expr
    };
    let per_input = cap / arity + 1;
    let over = RelExpr::Union {
        inputs: (0..arity).map(|_| chain(per_input)).collect(),
    };
    assert!(
        1 + arity * per_input > cap,
        "fixture: the union must be over the node cap"
    );
    let refused = register(&mut sm, &mut io, wire_query(over));
    assert_refused(&refused, "relation nodes");

    // Exactly `cap` nodes: the root plus inputs sized to land on the cap.
    let base = (cap - 1) / arity;
    let mut sizes = vec![base; arity];
    let mut remainder = (cap - 1) - arity * base;
    for size in &mut sizes {
        if remainder == 0 {
            break;
        }
        *size += 1;
        remainder -= 1;
    }
    assert_eq!(
        1 + sizes.iter().sum::<usize>(),
        cap,
        "fixture: at the cap exactly"
    );
    let at_cap = RelExpr::Union {
        inputs: sizes.into_iter().map(chain).collect(),
    };
    let admitted = register(&mut sm, &mut io, wire_query(at_cap));
    assert_admitted(&admitted);
}

#[test]
fn a_replayed_registration_is_not_charged_to_the_rate_window() {
    // Every successful handshake replays every standing subscription; a link that flaps
    // would otherwise spend the window on its own replays and lose them all.
    let mut caps = SubscriptionCaps::unlimited();
    caps.max_registrations_per_user_per_window = 2;
    caps.registration_window = Duration::from_secs(600);
    let mut admission = Admission::new(caps);
    let alice = Principal::User("alice".into());
    let client = ClientId::new();
    let query = QueryBuilder::new("users").build();
    let now = web_time::Instant::now();

    admission
        .admit(alice.clone(), client, QueryId(1), reg(&query), now)
        .expect("first registration");
    for _ in 0..10 {
        admission
            .admit(alice.clone(), client, QueryId(1), reg(&query), now)
            .expect("a replay of a held registration is free");
    }
    admission
        .admit(alice.clone(), client, QueryId(2), reg(&query), now)
        .expect("second new registration fills the window");
    let refused = admission
        .admit(alice.clone(), client, QueryId(3), reg(&query), now)
        .expect_err("the third new registration in the window is refused");
    assert_eq!(refused.cap_name(), "registration_rate");
    assert_eq!(
        admission.admitted_count(&alice),
        2,
        "a refused registration is not counted as standing"
    );

    // The window rolls over.
    admission
        .admit(
            alice,
            client,
            QueryId(3),
            reg(&query),
            now + Duration::from_secs(600),
        )
        .expect("a new window admits again");
}

#[test]
fn a_refused_registration_still_counts_toward_the_window() {
    // A client refused a hundred times a second is exactly the client the window is for:
    // refusals are cheap, but not free, and they must not be a way to probe the cap for free.
    let mut caps = SubscriptionCaps::unlimited();
    caps.max_registrations_per_user_per_window = 3;
    caps.max_query_limit = 10;
    let mut admission = Admission::new(caps);
    let alice = Principal::User("alice".into());
    let client = ClientId::new();
    let now = web_time::Instant::now();
    let mut oversized = QueryBuilder::new("users").build();
    oversized.limit = Some(11);

    for id in 0..3 {
        let refused = admission
            .admit(alice.clone(), client, QueryId(id), reg(&oversized), now)
            .expect_err("over the limit cap");
        assert_eq!(refused.cap_name(), "query_limit");
    }
    let fine = QueryBuilder::new("users").build();
    let refused = admission
        .admit(alice.clone(), client, QueryId(9), reg(&fine), now)
        .expect_err("the window is spent");
    assert_eq!(refused.cap_name(), "registration_rate");
}

#[test]
fn the_backend_and_anonymous_principals_are_never_rate_limited() {
    let mut caps = SubscriptionCaps::unlimited();
    caps.max_registrations_per_user_per_window = 1;
    let mut admission = Admission::new(caps);
    let query = QueryBuilder::new("users").build();
    let now = web_time::Instant::now();
    let backend = ClientId::new();
    let anonymous = ClientId::new();
    for id in 0..5 {
        admission
            .admit(
                Principal::Backend(backend),
                backend,
                QueryId(id),
                reg(&query),
                now,
            )
            .expect("the backend is trusted");
        admission
            .admit(
                Principal::Anonymous(anonymous),
                anonymous,
                QueryId(id),
                reg(&query),
                now,
            )
            .expect("an anonymous connection is bounded by its per-principal count instead");
    }
}

#[test]
fn a_re_registration_under_a_new_session_moves_the_charge() {
    let mut caps = SubscriptionCaps::unlimited();
    caps.max_subscriptions_per_user = 1;
    let mut admission = Admission::new(caps);
    let query = QueryBuilder::new("users").build();
    let now = web_time::Instant::now();
    let client = ClientId::new();
    let alice = Principal::User("alice".into());
    let bob = Principal::User("bob".into());

    admission
        .admit(alice.clone(), client, QueryId(1), reg(&query), now)
        .expect("alice's one");
    admission
        .admit(bob.clone(), client, QueryId(1), reg(&query), now)
        .expect("the same id re-registered by bob is bob's now");
    assert_eq!(admission.admitted_count(&alice), 0);
    assert_eq!(admission.admitted_count(&bob), 1);
    assert_eq!(admission.total_admitted(), 1);
}

#[test]
fn a_disconnect_releases_every_registration_the_client_held() {
    let mut sm = SyncManager::new().with_subscription_caps(SubscriptionCaps::default());
    let mut io = MemoryStorage::new();
    seed_users_schema(&mut io);
    let client_id = ClientId::new();
    add_client(&mut sm, &mut io, client_id);
    for id in 0..4 {
        sm.push_inbox(subscription(
            client_id,
            QueryId(id),
            QueryBuilder::new("users").build(),
        ));
    }
    sm.process_inbox(&mut io);
    assert_eq!(sm.total_admitted_subscriptions(), 4);

    sm.remove_client(client_id);
    assert_eq!(
        sm.total_admitted_subscriptions(),
        0,
        "a client that is gone holds nothing"
    );
}

/// Round-6 complement: the four non-query fields reach the admission through the inbox
/// arm (`inbox.rs`, the `QuerySubscription` frame), not only through `Admission::admit`
/// called directly. A byte-identical replay of a held id is free; the same id re-registered
/// with one changed claim is what the server compares as a different subscription, and is
/// charged to the window like any new registration.
#[test]
fn the_inbox_charges_a_re_registration_whose_claims_changed() {
    use crate::query_manager::session::AuthMode;
    let mut sm = SyncManager::new().with_subscription_caps(SubscriptionCaps {
        max_registrations_per_user_per_window: 1,
        registration_window: Duration::from_secs(600),
        ..SubscriptionCaps::default()
    });
    let mut io = MemoryStorage::new();
    seed_users_schema(&mut io);
    let client_id = ClientId::new();
    add_client(&mut sm, &mut io, client_id);
    let query_id = QueryId(7);
    let registration = |claims: serde_json::Value| InboxEntry {
        source: Source::Client(client_id),
        payload: SyncPayload::QuerySubscription {
            query_id,
            query: Box::new(QueryBuilder::new("users").build()),
            session: Some(Session {
                user_id: "alice".into(),
                claims,
                auth_mode: AuthMode::default(),
            }),
            required_tier: None,
            propagation: QueryPropagation::Full,
            policy_context_tables: vec![],
        },
    };
    let rejection = |sm: &SyncManager| {
        sm.outbox.iter().find_map(|entry| match &entry.payload {
            SyncPayload::Error(SyncError::QuerySubscriptionRejected {
                query_id: rejected,
                code,
                reason,
            }) if *rejected == query_id && entry.destination == Destination::Client(client_id) => {
                Some((code.clone(), reason.clone()))
            }
            _ => None,
        })
    };

    sm.push_inbox(registration(serde_json::json!({ "join_code": "a" })));
    sm.process_inbox(&mut io);
    assert_eq!(
        rejection(&sm),
        None,
        "the first registration is within the window"
    );

    sm.push_inbox(registration(serde_json::json!({ "join_code": "a" })));
    sm.process_inbox(&mut io);
    assert_eq!(
        rejection(&sm),
        None,
        "a byte-identical replay of a held id is not a new registration"
    );

    sm.push_inbox(registration(serde_json::json!({ "join_code": "b" })));
    sm.process_inbox(&mut io);
    let (code, reason) = rejection(&sm).expect(
        "a re-registration with a changed claim is a new registration and must be refused \
         by the window of one",
    );
    assert!(
        reason.contains("registration_rate") || reason.contains("registrations"),
        "the refusal must name the window: {code}: {reason}"
    );
}

fn in_list(values: usize) -> RelExpr {
    RelExpr::Filter {
        input: Box::new(scan("users")),
        predicate: PredicateExpr::In {
            left: ColumnRef {
                scope: None,
                column: "name".into(),
            },
            values: (0..values)
                .map(|i| ValueRef::Literal(Value::Text(format!("v{i}"))))
                .collect(),
        },
    }
}

#[test]
fn a_predicate_with_more_leaves_than_the_cap_is_refused() {
    // Three IR nodes, no joins, depth 0 — every structural cap calls this small; the
    // planner evaluates every term per row on every settle.
    let mut sm = SyncManager::new().with_subscription_caps(SubscriptionCaps::default());
    let mut io = MemoryStorage::new();
    seed_users_schema(&mut io);
    let cap = sm.subscription_caps().max_predicate_leaves;
    let refused = register(&mut sm, &mut io, wire_query(in_list(100_000)));
    assert_refused(&refused, "predicate leaves");
    let or_of_cmps = RelExpr::Filter {
        input: Box::new(scan("users")),
        predicate: PredicateExpr::Or(
            (0..cap + 1)
                .map(|i| PredicateExpr::Cmp {
                    left: ColumnRef {
                        scope: None,
                        column: "name".into(),
                    },
                    op: crate::query_manager::relation_ir::PredicateCmpOp::Eq,
                    right: ValueRef::Literal(Value::Text(format!("v{i}"))),
                })
                .collect(),
        ),
    };
    let refused = register(&mut sm, &mut io, wire_query(or_of_cmps));
    assert_refused(&refused, "predicate leaves");
    let admitted = register(&mut sm, &mut io, wire_query(in_list(cap)));
    assert_admitted(&admitted);
}

fn nested_not(depth: usize) -> PredicateExpr {
    let mut predicate = PredicateExpr::True;
    for _ in 0..depth {
        predicate = PredicateExpr::Not(Box::new(predicate));
    }
    predicate
}

fn nested_distinct(depth: usize) -> RelExpr {
    let mut expr = scan("users");
    for _ in 0..depth {
        expr = RelExpr::Distinct {
            input: Box::new(expr),
            key: vec![row_id_key()],
        };
    }
    expr
}

fn filtered(predicate: PredicateExpr) -> RelExpr {
    RelExpr::Filter {
        input: Box::new(scan("users")),
        predicate,
    }
}

/// Encoded on a thread with room for the recursion (encoding a deep value recurses too, and
/// the value is dropped there); the DECODE is the side that must be bounded.
fn encoded_subscription(relation_ir: RelExpr) -> Vec<u8> {
    std::thread::Builder::new()
        .stack_size(512 << 20)
        .spawn(move || {
            subscription(ClientId::new(), QueryId(1), wire_query(relation_ir))
                .payload
                .to_bytes()
                .expect("the payload encodes")
        })
        .expect("encode thread")
        .join()
        .expect("encode thread returns")
}

/// A tokio worker's stack: 2 MiB (`tokio::runtime::Builder`'s default, never raised here).
fn decode_on_a_worker_stack(bytes: Vec<u8>) -> Result<(), String> {
    std::thread::Builder::new()
        .stack_size(2 << 20)
        .spawn(move || {
            SyncPayload::from_bytes(&bytes)
                .map(|_| ())
                .map_err(|error| error.to_string())
        })
        .expect("decode thread")
        .join()
        .expect("the decode thread must return, not abort")
}

#[test]
fn a_frame_nested_deeper_than_the_wire_bound_is_refused_at_decode() {
    // Postcard has no recursion limit and every recursive wire type (`PredicateExpr::Not`,
    // the boxed `RelExpr` inputs, nested includes, `Value::Array`) is a derived, recursive
    // deserializer, so a ~12 KB frame of nested `Not` overflows a 2 MiB worker stack at
    // decode — a `SIGABRT`, not a panic, before admission sees the frame. On a tree without
    // the bound this gate does not fail: the test process aborts.
    let deep_predicate = encoded_subscription(filtered(nested_not(20_000)));
    assert!(
        decode_on_a_worker_stack(deep_predicate).is_err(),
        "a frame nested 20 000 levels deep must be refused at decode"
    );
    let deep_relation = encoded_subscription(nested_distinct(20_000));
    assert!(
        decode_on_a_worker_stack(deep_relation).is_err(),
        "a relation tree nested 20 000 levels deep must be refused at decode"
    );
    let shallow = encoded_subscription(filtered(nested_not(32)));
    decode_on_a_worker_stack(shallow).expect("a query nested 32 levels deep still decodes");
}

#[test]
fn a_query_nested_deeper_than_the_wire_bound_is_refused_by_the_walk() {
    // Defence in depth for the decode bound: the shape walk recurses too, and it must stop
    // counting past the bound rather than trust that every caller decoded through it. 1 000
    // levels is far past any bound a real query needs and shallow enough that neither the
    // walk on a tree without the guard nor the derived `Clone`/`Drop` of the value overflows
    // a 2 MiB test thread (5 000 did) — the missing refusal, not an abort, is what this
    // gate is red on.
    let mut sm = SyncManager::new().with_subscription_caps(SubscriptionCaps::default());
    let mut io = MemoryStorage::new();
    seed_users_schema(&mut io);
    let refused = register(&mut sm, &mut io, wire_query(filtered(nested_not(1_000))));
    assert_refused(&refused, "nesting");
    let refused = register(&mut sm, &mut io, wire_query(nested_distinct(1_000)));
    assert_refused(&refused, "nesting");
    let admitted = register(&mut sm, &mut io, wire_query(filtered(nested_not(32))));
    assert_admitted(&admitted);
}

#[test]
fn a_re_registration_with_a_changed_query_is_charged_to_the_rate_window() {
    // The replay exemption is for byte-identical replays, which take the equivalent fast
    // path anyway. A held id re-registered with a DIFFERENT query is a fresh derivation
    // (compile + first settle, the 0.46 s of defect #31) and is charged like one — otherwise
    // one id flipped between two filters at line speed re-derives forever, free, with the
    // standing count never moving.
    let mut caps = SubscriptionCaps::unlimited();
    caps.max_registrations_per_user_per_window = 2;
    caps.registration_window = Duration::from_secs(600);
    let mut admission = Admission::new(caps);
    let alice = Principal::User("alice".into());
    let client = ClientId::new();
    let now = web_time::Instant::now();
    let with_limit = |limit: usize| {
        let mut query = QueryBuilder::new("users").build();
        query.limit = Some(limit);
        query
    };
    admission
        .admit(alice.clone(), client, QueryId(1), reg(&with_limit(10)), now)
        .expect("first registration");
    admission
        .admit(alice.clone(), client, QueryId(1), reg(&with_limit(20)), now)
        .expect("a changed query is admitted while the window has room");
    let refused = admission
        .admit(alice.clone(), client, QueryId(1), reg(&with_limit(30)), now)
        .expect_err("the third distinct derivation in the window is refused");
    assert_eq!(refused.cap_name(), "registration_rate");
    assert_eq!(
        admission.admitted_count(&alice),
        1,
        "one id is one standing subscription however often its query changes"
    );
    for _ in 0..10 {
        admission
            .admit(alice.clone(), client, QueryId(1), reg(&with_limit(20)), now)
            .expect("a byte-identical replay of the held query stays free");
    }
}

#[test]
fn a_changed_re_registration_at_the_standing_cap_replaces_and_does_not_add() {
    // Guard for the fingerprint check above: a changed query on a held id is charged to the
    // window but is not a second standing subscription — at the standing cap it must be
    // admitted as a replacement, not refused as growth.
    let mut caps = SubscriptionCaps::unlimited();
    caps.max_subscriptions_per_user = 1;
    let mut admission = Admission::new(caps);
    let alice = Principal::User("alice".into());
    let client = ClientId::new();
    let now = web_time::Instant::now();
    let with_limit = |limit: usize| {
        let mut query = QueryBuilder::new("users").build();
        query.limit = Some(limit);
        query
    };
    admission
        .admit(alice.clone(), client, QueryId(1), reg(&with_limit(10)), now)
        .expect("the one standing subscription");
    admission
        .admit(alice.clone(), client, QueryId(1), reg(&with_limit(20)), now)
        .expect("a changed query on the held id replaces it, at the cap");
    assert_eq!(admission.admitted_count(&alice), 1);
    let refused = admission
        .admit(alice.clone(), client, QueryId(2), reg(&with_limit(10)), now)
        .expect_err("a second id is growth");
    assert_eq!(refused.cap_name(), "subscriptions");
}

#[test]
fn a_re_registration_that_changes_what_the_server_compares_is_charged() {
    // The server takes the equivalent path only when query, session, required tier,
    // propagation AND policy context tables all match (`existing_subscription_state` in
    // `server_queries.rs`); a held id re-registered with any of them changed re-derives
    // (compile + first settle). The replay exemption must be keyed on the same five, or one
    // id flipped between two claim sets, two tiers or two context-table lists re-derives
    // at line speed for free — round 4 closed this hole for the query and left it open for
    // the other four.
    let query = QueryBuilder::new("users").build();
    let alice_session = Session::new("alice");
    let mut with_claim = Session::new("alice");
    with_claim.claims = serde_json::json!({ "join_code": "x" });
    let tables = vec!["chat_members".to_string()];
    let base = Registration {
        query: &query,
        session: Some(&alice_session),
        required_tier: None,
        propagation: QueryPropagation::Full,
        policy_context_tables: &[],
    };
    let changed = [
        (
            "required_tier",
            Registration {
                required_tier: Some(crate::sync_manager::DurabilityTier::Local),
                ..base
            },
        ),
        (
            "propagation",
            Registration {
                propagation: QueryPropagation::LocalOnly,
                ..base
            },
        ),
        (
            "policy_context_tables",
            Registration {
                policy_context_tables: &tables,
                ..base
            },
        ),
        (
            "session claims",
            Registration {
                session: Some(&with_claim),
                ..base
            },
        ),
    ];
    for (field, changed) in changed {
        let mut caps = SubscriptionCaps::unlimited();
        caps.max_registrations_per_user_per_window = 1;
        caps.registration_window = Duration::from_secs(600);
        let mut admission = Admission::new(caps);
        let alice = Principal::User("alice".into());
        let client = ClientId::new();
        let now = web_time::Instant::now();
        admission
            .admit(alice.clone(), client, QueryId(1), base, now)
            .expect("the first registration is admitted");
        admission
            .admit(alice.clone(), client, QueryId(1), base, now)
            .expect("a byte-identical replay is free");
        let refused = admission
            .admit(alice.clone(), client, QueryId(1), changed, now)
            .expect_err(&format!(
                "a re-registration with a changed {field} is a new derivation at the server \
                 and must be charged to the rate window"
            ));
        assert_eq!(
            refused.cap_name(),
            "registration_rate",
            "{field}: {refused}"
        );
    }
}

#[test]
fn a_registration_over_the_byte_cap_is_refused() {
    // A 60 MiB text literal in one comparison is one predicate leaf and three IR nodes: it
    // passes every structural cap, and without a byte cap it is serialized under the engine
    // lock on every registration and compared against every row at settle. The hash that
    // computes the identity stops at the cap instead of measuring the frame, so the refusal
    // costs the lock nothing beyond the walk.
    let mut caps = SubscriptionCaps::unlimited();
    caps.max_query_bytes = 4096;
    let mut admission = Admission::new(caps);
    let alice = Principal::User("alice".into());
    let client = ClientId::new();
    let now = web_time::Instant::now();
    let literal = |bytes: usize| {
        QueryBuilder::new("users")
            .filter_eq("name", Value::Text("x".repeat(bytes)))
            .build()
    };
    let fine = literal(1024);
    admission
        .admit(alice.clone(), client, QueryId(1), reg(&fine), now)
        .expect("a registration under the byte cap is admitted");
    let oversized = literal(8192);
    let refused = admission
        .admit(alice.clone(), client, QueryId(2), reg(&oversized), now)
        .expect_err("a registration over the byte cap must be refused");
    assert_eq!(refused.cap_name(), "query_bytes", "{refused}");
    assert!(
        !admission.holds(client, QueryId(2)),
        "a refused registration leaves nothing behind"
    );
}

#[test]
fn a_frame_at_the_wire_bound_decodes_on_a_worker_stack() {
    // The bound's value is chosen so that a frame AT it decodes on a tokio worker's 2 MiB
    // stack in the debug profile, where the adapter's frames are largest. Measured, not
    // assumed: the deepest `Not` chain the bound admits (the payload's own wrappers take
    // the remaining levels) must decode on such a stack, and the wrappers must not eat
    // more than a handful of levels.
    let deepest = (0..=WIRE_MAX_NESTING)
        .rev()
        .find(|&depth| {
            decode_on_a_worker_stack(encoded_subscription(filtered(nested_not(depth)))).is_ok()
        })
        .expect("some nesting depth decodes");
    assert!(
        deepest + 16 >= WIRE_MAX_NESTING,
        "the payload's wrappers take {} of the {WIRE_MAX_NESTING} levels",
        WIRE_MAX_NESTING - deepest
    );
}
