//! Nested includes that name their target by the target's own id.
//!
//! The app's chat list nests `chat_members -> user` under every chat, and its
//! history pages nest `message_system -> actorUser` and the like under every
//! message: a nested include whose inner column is the target row's id, read
//! off a foreign key of the row above it. A write to one such target (a
//! contact's presence beat on `users`) has to reach the enclosing instances
//! whose subtree names that id, and no other.
//!
//! Liveness first, under routed and broadcast modes alike: a target that
//! changes, arrives after the row naming it, leaves, or is named anew must
//! show up in the subscriber's mirror after ONE settle. Then the cost: a write
//! to a target named under one chat must not re-evaluate the instances of
//! every other chat.

use std::collections::{BTreeMap, HashMap};

use super::*;
use crate::query_manager::graph_nodes::include_routing::{
    IncludeEmptyProbeMode, IncludeRoutingMode, force_include_empty_probe, force_include_routing,
    instance_evals_on_this_thread,
};
use crate::query_manager::manager::QueryUpdate;
use crate::query_manager::precise_dirty::{PreciseDirtyMode, force_precise_dirty};
use crate::query_manager::settle_skips::{SettleSkipsMode, force_settle_skips};

const CHATS: &str = "chats";
const MEMBERS: &str = "members";
const PEOPLE: &str = "people";

/// Column of the `members` array in a chat row (`title`, then the array).
const CHAT_MEMBERS_COLUMN: usize = 1;
/// Column of the `person` array in a member row (`chat_id`, `person_id`, then
/// the array).
const MEMBER_PERSON_COLUMN: usize = 2;

/// (chat, [(member, [(person, name)])]) — what the subscriber can see.
type Tree = Vec<(ObjectId, Vec<(ObjectId, Vec<(ObjectId, String)>)>)>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Routed,
    Broadcast,
}

fn schema() -> Schema {
    let mut schema = Schema::new();
    schema.insert(
        TableName::new(CHATS),
        RowDescriptor::new(vec![ColumnDescriptor::new("title", ColumnType::Text)]).into(),
    );
    schema.insert(
        TableName::new(MEMBERS),
        RowDescriptor::new(vec![
            ColumnDescriptor::new("chat_id", ColumnType::Uuid),
            ColumnDescriptor::new("person_id", ColumnType::Uuid),
        ])
        .into(),
    );
    schema.insert(
        TableName::new(PEOPLE),
        RowDescriptor::new(vec![ColumnDescriptor::new("name", ColumnType::Text)]).into(),
    );
    schema
}

struct Fixture {
    qm: QueryManager,
    storage: MemoryStorage,
    sub_id: crate::query_manager::graph_nodes::output::QuerySubscriptionId,
    chats: Vec<ObjectId>,
    /// Per chat, its two members.
    members: Vec<Vec<ObjectId>>,
    /// Per chat, the people its two members name.
    people: Vec<Vec<ObjectId>>,
    rows: HashMap<ObjectId, crate::query_manager::types::Row>,
    descriptor: Option<RowDescriptor>,
}

impl Fixture {
    /// `chat_count` chats, each with two members naming two people of their own.
    fn seed(chat_count: usize) -> Self {
        let (mut qm, mut storage) = create_query_manager(SyncManager::new(), schema());
        let mut chats = Vec::new();
        let mut members = Vec::new();
        let mut people = Vec::new();
        for chat_index in 0..chat_count {
            let chat = qm
                .insert(
                    &mut storage,
                    CHATS,
                    &[Value::Text(format!("chat-{chat_index:04}"))],
                )
                .expect("insert chat")
                .row_id;
            let mut chat_people = Vec::new();
            let mut chat_members = Vec::new();
            for person_index in 0..2 {
                let person = qm
                    .insert(
                        &mut storage,
                        PEOPLE,
                        &[Value::Text(format!("person-{chat_index}-{person_index}"))],
                    )
                    .expect("insert person")
                    .row_id;
                let member = qm
                    .insert(
                        &mut storage,
                        MEMBERS,
                        &[Value::Uuid(chat), Value::Uuid(person)],
                    )
                    .expect("insert member")
                    .row_id;
                chat_people.push(person);
                chat_members.push(member);
            }
            chats.push(chat);
            members.push(chat_members);
            people.push(chat_people);
        }

        let query = qm
            .query(CHATS)
            .with_array("members", |sub| {
                sub.from(MEMBERS)
                    .correlate("chat_id", "chats.id")
                    .with_array("person", |nested| {
                        nested.from(PEOPLE).correlate("id", "members.person_id")
                    })
            })
            .order_by("title")
            .build();
        let sub_id = qm.subscribe(query).expect("subscribe");

        let mut fixture = Self {
            qm,
            storage,
            sub_id,
            chats,
            members,
            people,
            rows: HashMap::new(),
            descriptor: None,
        };
        fixture.settle_once();
        assert_eq!(
            fixture.tree().len(),
            chat_count,
            "the cold-open snapshot must carry every chat"
        );
        fixture
    }

    fn settle_once(&mut self) {
        self.qm.process(&mut self.storage);
        let sub_id = self.sub_id;
        let updates: Vec<QueryUpdate> = self
            .qm
            .take_updates()
            .into_iter()
            .filter(|update| update.subscription_id == sub_id)
            .collect();
        for update in updates {
            self.descriptor = Some(update.descriptor.clone());
            for row in &update.delta.removed {
                self.rows.remove(&row.id);
            }
            for (_, new_row) in &update.delta.updated {
                self.rows.insert(new_row.id, new_row.clone());
            }
            for row in &update.delta.added {
                self.rows.insert(row.id, row.clone());
            }
        }
    }

    fn tree(&self) -> Tree {
        let Some(descriptor) = self.descriptor.as_ref() else {
            return Vec::new();
        };
        self.chats
            .iter()
            .filter(|chat| self.rows.contains_key(chat))
            .map(|chat| {
                let values =
                    decode_row(descriptor, &self.rows[chat].data).expect("decode chat row");
                let mut members: Vec<(ObjectId, Vec<(ObjectId, String)>)> =
                    array_at(&values, CHAT_MEMBERS_COLUMN)
                        .iter()
                        .map(|member| {
                            let (member_id, member_values) = row_of(member);
                            let person = array_at(member_values, MEMBER_PERSON_COLUMN)
                                .iter()
                                .map(|person| {
                                    let (person_id, person_values) = row_of(person);
                                    match person_values.first() {
                                        Some(Value::Text(name)) => (person_id, name.clone()),
                                        other => panic!("expected a person name, got {other:?}"),
                                    }
                                })
                                .collect();
                            (member_id, person)
                        })
                        .collect();
                members.sort_by_key(|(member, _)| *member);
                (*chat, members)
            })
            .collect()
    }

    /// The `person` array the mirror shows for one member.
    fn person_of(&self, chat_index: usize, member: ObjectId) -> Vec<(ObjectId, String)> {
        let chat = self.chats[chat_index];
        self.tree()
            .into_iter()
            .find(|(id, _)| *id == chat)
            .and_then(|(_, members)| members.into_iter().find(|(id, _)| *id == member))
            .map(|(_, person)| person)
            .unwrap_or_else(|| panic!("member missing from chat {chat_index} in the mirror"))
    }

    fn insert_person_with_id(&mut self, id: ObjectId, name: &str) {
        let schema = self.qm.schema_context().current_schema.clone();
        let branch = get_branch(&self.qm);
        self.qm
            .insert_on_branch_with_schema_and_write_context_and_id(
                &mut self.storage,
                PEOPLE,
                &branch,
                &[Value::Text(name.to_string())],
                Some(id),
                &schema,
                None,
                true,
            )
            .expect("insert person with id");
    }
}

fn array_at(values: &[Value], column: usize) -> &[Value] {
    match values.get(column) {
        Some(Value::Array(items)) => items,
        other => panic!("expected an include array at column {column}, got {other:?}"),
    }
}

fn row_of(value: &Value) -> (ObjectId, &Vec<Value>) {
    match value {
        Value::Row {
            id: Some(id),
            values,
        } => (*id, values),
        other => panic!("expected an included row carrying its id, got {other:?}"),
    }
}

/// The engine paths these tests run on, pinned for the guard's lifetime: precise
/// dirtiness and empty-binding probes at their defaults, and either the routed engine
/// with real settle skips or the reference one — broadcast, nothing skipped. Other
/// tests in the process force these process-wide plugs (`subscription_output_oracle`
/// runs the legacy path), so an unpinned case can settle on whatever they last set.
/// Taken in the oracle's order — precise, routing, empty probe — and the skips last,
/// so none of them deadlock.
struct Engaged {
    _precise: PreciseDirtyMode,
    _routing: IncludeRoutingMode,
    _empty_probe: IncludeEmptyProbeMode,
    _skips: SettleSkipsMode,
}

fn engage(routed: bool) -> Engaged {
    let precise = force_precise_dirty(true);
    let routing = force_include_routing(routed);
    let empty_probe = force_include_empty_probe(true);
    Engaged {
        _precise: precise,
        _routing: routing,
        _empty_probe: empty_probe,
        _skips: force_settle_skips(routed),
    }
}

fn each_mode(case: impl Fn(Mode)) {
    for mode in [Mode::Routed, Mode::Broadcast] {
        let _engaged = engage(mode == Mode::Routed);
        case(mode);
    }
}

const CHAT_COUNT: usize = 6;

#[test]
fn a_named_target_that_changes_reaches_the_member() {
    each_mode(|mode| {
        let mut fixture = Fixture::seed(CHAT_COUNT);
        let person = fixture.people[3][1];
        let member = fixture.members[3][1];
        fixture
            .qm
            .update(
                &mut fixture.storage,
                person,
                &[Value::Text("renamed".into())],
            )
            .expect("update person");
        fixture.settle_once();
        assert_eq!(
            fixture.person_of(3, member),
            vec![(person, "renamed".to_string())],
            "{mode:?}: the renamed person must reach the member naming it in one settle"
        );
    });
}

#[test]
fn a_target_that_arrives_after_the_member_naming_it_fills_it() {
    each_mode(|mode| {
        let mut fixture = Fixture::seed(CHAT_COUNT);
        let late = ObjectId::new();
        let member = fixture
            .qm
            .insert(
                &mut fixture.storage,
                MEMBERS,
                &[Value::Uuid(fixture.chats[2]), Value::Uuid(late)],
            )
            .expect("insert member naming a person not yet there")
            .row_id;
        fixture.settle_once();
        assert_eq!(
            fixture.person_of(2, member),
            vec![],
            "{mode:?}: a member naming a missing person carries an empty array"
        );

        fixture.insert_person_with_id(late, "late");
        fixture.settle_once();
        assert_eq!(
            fixture.person_of(2, member),
            vec![(late, "late".to_string())],
            "{mode:?}: the person arriving after its member must fill it in one settle"
        );
    });
}

#[test]
fn a_target_that_leaves_empties_the_member() {
    each_mode(|mode| {
        let mut fixture = Fixture::seed(CHAT_COUNT);
        let person = fixture.people[4][0];
        let member = fixture.members[4][0];
        fixture
            .qm
            .delete(&mut fixture.storage, person)
            .expect("delete person");
        fixture.settle_once();
        assert_eq!(
            fixture.person_of(4, member),
            vec![],
            "{mode:?}: a deleted person must leave the member naming it in one settle"
        );
    });
}

#[test]
fn a_member_renamed_to_a_missing_target_fills_when_it_arrives() {
    each_mode(|mode| {
        let mut fixture = Fixture::seed(CHAT_COUNT);
        let member = fixture.members[5][0];
        let fresh = ObjectId::new();
        fixture
            .qm
            .update(
                &mut fixture.storage,
                member,
                &[Value::Uuid(fixture.chats[5]), Value::Uuid(fresh)],
            )
            .expect("point the member at a person not yet there");
        fixture.settle_once();
        assert_eq!(fixture.person_of(5, member), vec![], "{mode:?}");

        fixture.insert_person_with_id(fresh, "fresh");
        fixture.settle_once();
        assert_eq!(
            fixture.person_of(5, member),
            vec![(fresh, "fresh".to_string())],
            "{mode:?}: the newly named person must fill the member in one settle"
        );

        // The person it named before must no longer reach it.
        let old = fixture.people[5][0];
        fixture
            .qm
            .update(&mut fixture.storage, old, &[Value::Text("stale".into())])
            .expect("update the person no longer named");
        fixture.settle_once();
        assert_eq!(
            fixture.person_of(5, member),
            vec![(fresh, "fresh".to_string())],
            "{mode:?}"
        );
    });
}

#[test]
fn a_target_named_in_two_chats_reaches_both() {
    each_mode(|mode| {
        let mut fixture = Fixture::seed(CHAT_COUNT);
        let shared = fixture.people[0][0];
        let member = fixture
            .qm
            .insert(
                &mut fixture.storage,
                MEMBERS,
                &[Value::Uuid(fixture.chats[1]), Value::Uuid(shared)],
            )
            .expect("insert a second member naming the same person")
            .row_id;
        fixture.settle_once();
        fixture
            .qm
            .update(
                &mut fixture.storage,
                shared,
                &[Value::Text("shared".into())],
            )
            .expect("update shared person");
        fixture.settle_once();
        assert_eq!(
            fixture.person_of(0, fixture.members[0][0]),
            vec![(shared, "shared".to_string())],
            "{mode:?}"
        );
        assert_eq!(
            fixture.person_of(1, member),
            vec![(shared, "shared".to_string())],
            "{mode:?}"
        );
    });
}

/// Instance evaluations one person write and its settle cost, with
/// `chat_count` chats live.
fn instance_evals_for_one_target_write(chat_count: usize) -> u64 {
    let _engaged = engage(true);
    let mut fixture = Fixture::seed(chat_count);
    let person = fixture.people[0][0];
    let member = fixture.members[0][0];
    let before = instance_evals_on_this_thread();
    fixture
        .qm
        .update(&mut fixture.storage, person, &[Value::Text("beat".into())])
        .expect("update person");
    fixture.settle_once();
    let evals = instance_evals_on_this_thread() - before;
    assert_eq!(
        fixture.person_of(0, member),
        vec![(person, "beat".to_string())],
        "the write must reach the one member naming the person"
    );
    evals
}

/// A presence beat on one contact re-evaluated every chat's instance: the
/// nested `person` link was unroutable, so its table's marks broadcast to all
/// of them. The instances that can see the change are the ones naming that
/// person, whatever the number of chats.
#[test]
fn a_target_write_evaluates_only_the_instances_naming_it() {
    let low = instance_evals_for_one_target_write(8);
    let high = instance_evals_for_one_target_write(64);
    assert!(
        high <= low,
        "one person write re-evaluated {high} include instances with 64 chats live but {low} \
         with 8 — the write reaches instances that do not name the person"
    );
}

// ============================================================================
// Randomized differential: routed against broadcast, both against a model.
// ============================================================================

/// One step's worth of state the generator holds: the reference model.
#[derive(Default, Clone)]
struct Model {
    chats: HashMap<ObjectId, String>,
    members: HashMap<ObjectId, (ObjectId, ObjectId)>,
    people: HashMap<ObjectId, String>,
    /// Ids members name that no live person carries.
    dangling: Vec<ObjectId>,
}

/// (chat, title, [(member, [(person, name)])]) — chats in title order, members by id.
type ChatTree = Vec<(ObjectId, String, Vec<(ObjectId, Vec<(ObjectId, String)>)>)>;
/// (member, [(person, name)]) — members by id.
type MemberTree = Vec<(ObjectId, Vec<(ObjectId, String)>)>;

/// Outer window of the windowed subscription.
const WINDOW: usize = 3;

impl Model {
    fn person(&self, id: ObjectId) -> Vec<(ObjectId, String)> {
        self.people
            .get(&id)
            .map(|name| vec![(id, name.clone())])
            .unwrap_or_default()
    }

    fn chat_tree(&self, window: Option<usize>) -> ChatTree {
        let mut chats: Vec<(ObjectId, String)> = self
            .chats
            .iter()
            .map(|(id, title)| (*id, title.clone()))
            .collect();
        chats.sort_by(|a, b| a.1.cmp(&b.1));
        if let Some(window) = window {
            chats.truncate(window);
        }
        chats
            .into_iter()
            .map(|(chat, title)| {
                let mut members: Vec<(ObjectId, Vec<(ObjectId, String)>)> = self
                    .members
                    .iter()
                    .filter(|(_, (member_chat, _))| *member_chat == chat)
                    .map(|(member, (_, person))| (*member, self.person(*person)))
                    .collect();
                members.sort_by_key(|(member, _)| *member);
                (chat, title, members)
            })
            .collect()
    }

    fn member_tree(&self) -> MemberTree {
        let mut members: MemberTree = self
            .members
            .iter()
            .map(|(member, (_, person))| (*member, self.person(*person)))
            .collect();
        members.sort_by_key(|(member, _)| *member);
        members
    }
}

/// A row mirror rebuilt from one subscription's deltas.
#[derive(Default)]
struct Mirror {
    rows: HashMap<ObjectId, crate::query_manager::types::Row>,
    descriptor: Option<RowDescriptor>,
}

impl Mirror {
    fn absorb(&mut self, update: &QueryUpdate) {
        self.descriptor = Some(update.descriptor.clone());
        for row in &update.delta.removed {
            assert!(
                self.rows.remove(&row.id).is_some(),
                "a removed row the subscriber never held"
            );
        }
        for (old, new) in &update.delta.updated {
            assert_eq!(old.id, new.id);
            assert!(
                self.rows.insert(new.id, new.clone()).is_some(),
                "an updated row the subscriber never held"
            );
        }
        for row in &update.delta.added {
            self.rows.insert(row.id, row.clone());
        }
    }

    fn decoded(&self) -> Vec<(ObjectId, Vec<Value>)> {
        let Some(descriptor) = self.descriptor.as_ref() else {
            return Vec::new();
        };
        self.rows
            .iter()
            .map(|(id, row)| (*id, decode_row(descriptor, &row.data).expect("decode row")))
            .collect()
    }
}

fn person_array(values: &[Value]) -> Vec<(ObjectId, String)> {
    array_at(values, MEMBER_PERSON_COLUMN)
        .iter()
        .map(|person| {
            let (id, values) = row_of(person);
            match values.first() {
                Some(Value::Text(name)) => (id, name.clone()),
                other => panic!("expected a person name, got {other:?}"),
            }
        })
        .collect()
}

fn chat_tree_of(mirror: &Mirror) -> ChatTree {
    let mut chats: ChatTree = mirror
        .decoded()
        .into_iter()
        .map(|(chat, values)| {
            let title = match values.first() {
                Some(Value::Text(title)) => title.clone(),
                other => panic!("expected a chat title, got {other:?}"),
            };
            let mut members: Vec<(ObjectId, Vec<(ObjectId, String)>)> =
                array_at(&values, CHAT_MEMBERS_COLUMN)
                    .iter()
                    .map(|member| {
                        let (id, values) = row_of(member);
                        (id, person_array(values))
                    })
                    .collect();
            members.sort_by_key(|(member, _)| *member);
            (chat, title, members)
        })
        .collect();
    chats.sort_by(|a, b| a.1.cmp(&b.1));
    chats
}

fn member_tree_of(mirror: &Mirror) -> MemberTree {
    let mut members: MemberTree = mirror
        .decoded()
        .into_iter()
        .map(|(member, values)| (member, person_array(&values)))
        .collect();
    members.sort_by_key(|(member, _)| *member);
    members
}

/// One engine of the dual run and its three subscriptions' mirrors.
struct Side {
    routed: bool,
    qm: QueryManager,
    storage: MemoryStorage,
    subs: [crate::query_manager::graph_nodes::output::QuerySubscriptionId; 3],
    mirrors: [Mirror; 3],
}

impl Side {
    fn new(routed: bool) -> Self {
        let _engaged = engage(routed);
        let (mut qm, storage) = create_query_manager(SyncManager::new(), schema());
        let chats_with_people = |qm: &QueryManager, window: Option<usize>| {
            let builder = qm
                .query(CHATS)
                .with_array("members", |sub| {
                    sub.from(MEMBERS)
                        .correlate("chat_id", "chats.id")
                        .with_array("person", |nested| {
                            nested.from(PEOPLE).correlate("id", "members.person_id")
                        })
                })
                .order_by("title");
            match window {
                Some(window) => builder.limit(window).build(),
                None => builder.build(),
            }
        };
        let all = chats_with_people(&qm, None);
        let windowed = chats_with_people(&qm, Some(WINDOW));
        let members = qm
            .query(MEMBERS)
            .with_array("person", |sub| {
                sub.from(PEOPLE).correlate("id", "members.person_id")
            })
            .build();
        let subs = [
            qm.subscribe(all).expect("subscribe chats"),
            qm.subscribe(windowed).expect("subscribe window"),
            qm.subscribe(members).expect("subscribe members"),
        ];
        Self {
            routed,
            qm,
            storage,
            subs,
            mirrors: Default::default(),
        }
    }

    fn insert(&mut self, table: &str, id: ObjectId, values: &[Value]) {
        let _engaged = engage(self.routed);
        let schema = self.qm.schema_context().current_schema.clone();
        let branch = get_branch(&self.qm);
        self.qm
            .insert_on_branch_with_schema_and_write_context_and_id(
                &mut self.storage,
                table,
                &branch,
                values,
                Some(id),
                &schema,
                None,
                true,
            )
            .unwrap_or_else(|error| panic!("insert into {table}: {error:?}"));
    }

    fn update(&mut self, id: ObjectId, values: &[Value]) {
        let _engaged = engage(self.routed);
        self.qm
            .update(&mut self.storage, id, values)
            .unwrap_or_else(|error| panic!("update: {error:?}"));
    }

    fn delete(&mut self, id: ObjectId) {
        let _engaged = engage(self.routed);
        self.qm
            .delete(&mut self.storage, id)
            .unwrap_or_else(|error| panic!("delete: {error:?}"));
    }

    /// One `process()`; returns whether anything was emitted.
    fn pass(&mut self) -> bool {
        let _engaged = engage(self.routed);
        self.qm.process(&mut self.storage);
        let updates = self.qm.take_updates();
        for update in &updates {
            if let Some(slot) = self
                .subs
                .iter()
                .position(|sub| *sub == update.subscription_id)
            {
                self.mirrors[slot].absorb(update);
            }
        }
        !updates.is_empty()
    }

    fn trees(&self) -> (ChatTree, ChatTree, MemberTree) {
        (
            chat_tree_of(&self.mirrors[0]),
            chat_tree_of(&self.mirrors[1]),
            member_tree_of(&self.mirrors[2]),
        )
    }
}

struct Prng(u64);

impl Prng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }

    fn pick<T: Copy>(&mut self, items: &[T]) -> Option<T> {
        (!items.is_empty()).then(|| items[self.below(items.len())])
    }
}

/// Deterministic ids: both engines see identical row ids.
struct Mint {
    seed: u64,
    counter: u64,
}

impl Mint {
    fn next(&mut self) -> ObjectId {
        self.counter += 1;
        ObjectId::from_uuid(uuid::Uuid::from_u128(
            (u128::from(self.seed) << 64) | u128::from(self.counter),
        ))
    }
}

const DIFFERENTIAL_SEEDS: [u64; 6] = [
    0xF0A1_0000_0000_0001,
    0xF0A1_0000_0000_0002,
    0xF0A1_0000_0000_0003,
    0xF0A1_0000_0000_0004,
    0xF0A1_0000_0000_0005,
    0xF0A1_0000_0000_0006,
];
const DIFFERENTIAL_STEPS: usize = 300;
/// One `process()` serves a write; a second pass may only confirm quiescence.
const PASS_BOUND: usize = 2;

fn run_differential(seed: u64) {
    let mut rng = Prng(seed | 1);
    let mut mint = Mint { seed, counter: 0 };
    let mut model = Model::default();
    let mut sides = [Side::new(true), Side::new(false)];

    for step in 0..DIFFERENTIAL_STEPS {
        // Draw one operation against the model, then replay it on both sides.
        let chat_ids: Vec<ObjectId> = model.chats.keys().copied().collect();
        let member_ids: Vec<ObjectId> = model.members.keys().copied().collect();
        let people_ids: Vec<ObjectId> = model.people.keys().copied().collect();
        let roll = rng.below(100);
        let target_of_member = |rng: &mut Prng, model: &mut Model, mint: &mut Mint| -> ObjectId {
            match rng.below(3) {
                0 => rng.pick(&people_ids).unwrap_or_else(|| mint.next()),
                1 => rng.pick(&model.dangling).unwrap_or_else(|| mint.next()),
                _ => {
                    let fresh = mint.next();
                    model.dangling.push(fresh);
                    fresh
                }
            }
        };
        let label = match roll {
            0..=9 => {
                let id = mint.next();
                let title = format!("{:05}-{step}", rng.below(100_000));
                for side in &mut sides {
                    side.insert(CHATS, id, &[Value::Text(title.clone())]);
                }
                model.chats.insert(id, title);
                "insert chat"
            }
            10..=17 if !chat_ids.is_empty() => {
                let id = rng.pick(&chat_ids).expect("a chat");
                let title = format!("{:05}-{step}", rng.below(100_000));
                for side in &mut sides {
                    side.update(id, &[Value::Text(title.clone())]);
                }
                model.chats.insert(id, title);
                "retitle chat"
            }
            18..=21 if !chat_ids.is_empty() => {
                let id = rng.pick(&chat_ids).expect("a chat");
                for side in &mut sides {
                    side.delete(id);
                }
                model.chats.remove(&id);
                "delete chat"
            }
            22..=39 => {
                // Fill a dangling id half the time: the late-arrival class.
                let id = if !model.dangling.is_empty() && rng.below(2) == 0 {
                    let index = rng.below(model.dangling.len());
                    model.dangling.swap_remove(index)
                } else {
                    mint.next()
                };
                let name = format!("p-{step}");
                for side in &mut sides {
                    side.insert(PEOPLE, id, &[Value::Text(name.clone())]);
                }
                model.people.insert(id, name);
                "insert person"
            }
            40..=59 if !people_ids.is_empty() => {
                let id = rng.pick(&people_ids).expect("a person");
                let name = format!("r-{step}");
                for side in &mut sides {
                    side.update(id, &[Value::Text(name.clone())]);
                }
                model.people.insert(id, name);
                "rename person"
            }
            60..=67 if !people_ids.is_empty() => {
                let id = rng.pick(&people_ids).expect("a person");
                for side in &mut sides {
                    side.delete(id);
                }
                model.people.remove(&id);
                "delete person"
            }
            68..=81 if !chat_ids.is_empty() => {
                let id = mint.next();
                let chat = rng.pick(&chat_ids).expect("a chat");
                let person = target_of_member(&mut rng, &mut model, &mut mint);
                for side in &mut sides {
                    side.insert(MEMBERS, id, &[Value::Uuid(chat), Value::Uuid(person)]);
                }
                model.members.insert(id, (chat, person));
                "insert member"
            }
            82..=93 if !member_ids.is_empty() => {
                let id = rng.pick(&member_ids).expect("a member");
                let chat = model.members[&id].0;
                let person = target_of_member(&mut rng, &mut model, &mut mint);
                for side in &mut sides {
                    side.update(id, &[Value::Uuid(chat), Value::Uuid(person)]);
                }
                model.members.insert(id, (chat, person));
                "re-point member"
            }
            94..=99 if !member_ids.is_empty() => {
                let id = rng.pick(&member_ids).expect("a member");
                for side in &mut sides {
                    side.delete(id);
                }
                model.members.remove(&id);
                "delete member"
            }
            _ => continue,
        };

        // Lockstep passes: routed and broadcast must agree after EVERY pass, and
        // the first pass must already serve the model.
        for pass in 0..PASS_BOUND {
            let emitted: Vec<bool> = sides.iter_mut().map(Side::pass).collect();
            let (routed, broadcast) = (sides[0].trees(), sides[1].trees());
            assert!(
                routed == broadcast,
                "seed {seed:#x} step {step} ({label}) pass {pass}: routed and broadcast \
                 engines serve different results"
            );
            if pass == 0 {
                let expected = (
                    model.chat_tree(None),
                    model.chat_tree(Some(WINDOW)),
                    model.member_tree(),
                );
                assert!(
                    routed.0 == expected.0,
                    "seed {seed:#x} step {step} ({label}): chats subscription differs from the model"
                );
                assert!(
                    routed.1 == expected.1,
                    "seed {seed:#x} step {step} ({label}): windowed subscription differs from the model"
                );
                assert!(
                    routed.2 == expected.2,
                    "seed {seed:#x} step {step} ({label}): members subscription differs from the model"
                );
            } else {
                assert!(
                    emitted.iter().all(|emitted| !emitted),
                    "seed {seed:#x} step {step} ({label}): a second pass still emitted"
                );
            }
        }
    }
}

/// Inserts, renames, deletes and late arrivals of include targets, members
/// re-pointed to present, absent and fresh ids, chats re-titled across an
/// outer window: the routed engine must serve exactly what the broadcast one
/// does, and both what the model says, after one settle.
#[test]
fn forward_reference_differential_routed_vs_broadcast() {
    for seed in DIFFERENTIAL_SEEDS {
        run_differential(seed);
    }
}

/// A forward reference with another forward reference to the SAME table inside
/// it: `members -> person (people by id) -> manager (people by id)`. The person
/// node's own link is a row id, but the manager include over the same table
/// makes `people` unroutable for that node's own routing. It must still export
/// the ids it probes: the enclosing members node routes a `people` insert by
/// them alone, and an instance waiting for a person who does not exist yet is
/// found only through that export.
#[test]
fn a_target_arriving_under_a_self_referencing_include_fills_it() {
    const ORGS: &str = "orgs";
    let schema = {
        let mut schema = Schema::new();
        schema.insert(
            TableName::new(ORGS),
            RowDescriptor::new(vec![ColumnDescriptor::new("title", ColumnType::Text)]).into(),
        );
        schema.insert(
            TableName::new(MEMBERS),
            RowDescriptor::new(vec![
                ColumnDescriptor::new("org_id", ColumnType::Uuid),
                ColumnDescriptor::new("person_id", ColumnType::Uuid),
            ])
            .into(),
        );
        schema.insert(
            TableName::new(PEOPLE),
            RowDescriptor::new(vec![
                ColumnDescriptor::new("name", ColumnType::Text),
                ColumnDescriptor::new("manager_id", ColumnType::Uuid),
            ])
            .into(),
        );
        schema
    };
    each_mode(|mode| {
        let (mut qm, mut storage) = create_query_manager(SyncManager::new(), schema.clone());
        let boss = qm
            .insert(
                &mut storage,
                PEOPLE,
                &[Value::Text("boss".into()), Value::Uuid(ObjectId::new())],
            )
            .expect("insert boss")
            .row_id;
        let mut orgs = Vec::new();
        for org_index in 0..CHAT_COUNT {
            let org = qm
                .insert(
                    &mut storage,
                    ORGS,
                    &[Value::Text(format!("org-{org_index:04}"))],
                )
                .expect("insert org")
                .row_id;
            let person = qm
                .insert(
                    &mut storage,
                    PEOPLE,
                    &[Value::Text(format!("p-{org_index}")), Value::Uuid(boss)],
                )
                .expect("insert person")
                .row_id;
            qm.insert(
                &mut storage,
                MEMBERS,
                &[Value::Uuid(org), Value::Uuid(person)],
            )
            .expect("insert member");
            orgs.push(org);
        }
        let query = qm
            .query(ORGS)
            .with_array("members", |sub| {
                sub.from(MEMBERS)
                    .correlate("org_id", "orgs.id")
                    .with_array("person", |person| {
                        person
                            .from(PEOPLE)
                            .correlate("id", "members.person_id")
                            .with_array("manager", |manager| {
                                manager.from(PEOPLE).correlate("id", "people.manager_id")
                            })
                    })
            })
            .order_by("title")
            .build();
        let sub_id = qm.subscribe(query).expect("subscribe");
        let mut rows: HashMap<ObjectId, crate::query_manager::types::Row> = HashMap::new();
        let mut descriptor = None;
        let mut settle = |qm: &mut QueryManager, storage: &mut MemoryStorage| {
            qm.process(storage);
            for update in qm.take_updates() {
                if update.subscription_id != sub_id {
                    continue;
                }
                descriptor = Some(update.descriptor.clone());
                for row in &update.delta.removed {
                    rows.remove(&row.id);
                }
                for (_, new_row) in &update.delta.updated {
                    rows.insert(new_row.id, new_row.clone());
                }
                for row in &update.delta.added {
                    rows.insert(row.id, row.clone());
                }
            }
            (rows.clone(), descriptor.clone())
        };
        let person_of = |rows: &HashMap<ObjectId, crate::query_manager::types::Row>,
                         descriptor: &Option<RowDescriptor>,
                         member: ObjectId|
         -> Vec<ObjectId> {
            let descriptor = descriptor
                .as_ref()
                .expect("a delivery carried the descriptor");
            let values = decode_row(descriptor, &rows[&orgs[1]].data).expect("decode org row");
            array_at(&values, CHAT_MEMBERS_COLUMN)
                .iter()
                .map(row_of)
                .find(|(id, _)| *id == member)
                .map(|(_, member_values)| {
                    array_at(member_values, MEMBER_PERSON_COLUMN)
                        .iter()
                        .map(|person| row_of(person).0)
                        .collect()
                })
                .unwrap_or_else(|| panic!("{mode:?}: member missing from org 1"))
        };
        settle(&mut qm, &mut storage);

        let late = ObjectId::new();
        let member = qm
            .insert(
                &mut storage,
                MEMBERS,
                &[Value::Uuid(orgs[1]), Value::Uuid(late)],
            )
            .expect("insert member naming a person not yet there")
            .row_id;
        let (now, desc) = settle(&mut qm, &mut storage);
        assert_eq!(
            person_of(&now, &desc, member),
            Vec::<ObjectId>::new(),
            "{mode:?}: a member naming a missing person carries an empty array"
        );

        let schema_now = qm.schema_context().current_schema.clone();
        let branch = get_branch(&qm);
        qm.insert_on_branch_with_schema_and_write_context_and_id(
            &mut storage,
            PEOPLE,
            &branch,
            &[Value::Text("late".into()), Value::Uuid(boss)],
            Some(late),
            &schema_now,
            None,
            true,
        )
        .expect("insert the late person");
        let (now, desc) = settle(&mut qm, &mut storage);
        assert_eq!(
            person_of(&now, &desc, member),
            vec![late],
            "{mode:?}: the person arriving after its member must fill it in one settle, \
             though a manager include over the same table sits inside the person include"
        );
    });
}

// ---------------------------------------------------------------------------------------------
// Three levels: `orgs -> members -> person (people by id) -> manager (people by id)`.
//
// The app's chat list routes a `users` write through two exports: the nested node exports the
// ids it probes, they land in the enclosing members node's `held_by`, and that node passes them
// up to the org instance (`extend_tracked_row_ids`). A grandchild that arrives late, changes, or
// is deleted must reach the org instance through that chain alone — a deleted grandchild cannot
// be loaded, so `held_by` is the only thing that can name it.
// ---------------------------------------------------------------------------------------------

const ORGS: &str = "orgs";
/// Column of the `manager` array in a person row (`name`, `manager_id`, then the array).
const PERSON_MANAGER_COLUMN: usize = 2;

fn three_level_schema() -> Schema {
    let mut schema = Schema::new();
    schema.insert(
        TableName::new(ORGS),
        RowDescriptor::new(vec![ColumnDescriptor::new("title", ColumnType::Text)]).into(),
    );
    schema.insert(
        TableName::new(MEMBERS),
        RowDescriptor::new(vec![
            ColumnDescriptor::new("org_id", ColumnType::Uuid),
            ColumnDescriptor::new("person_id", ColumnType::Uuid),
        ])
        .into(),
    );
    schema.insert(
        TableName::new(PEOPLE),
        RowDescriptor::new(vec![
            ColumnDescriptor::new("name", ColumnType::Text),
            ColumnDescriptor::new("manager_id", ColumnType::Uuid),
        ])
        .into(),
    );
    schema
}

fn three_level_query(qm: &QueryManager) -> Query {
    qm.query(ORGS)
        .with_array("members", |sub| {
            sub.from(MEMBERS)
                .correlate("org_id", "orgs.id")
                .with_array("person", |person| {
                    person
                        .from(PEOPLE)
                        .correlate("id", "members.person_id")
                        .with_array("manager", |manager| {
                            manager.from(PEOPLE).correlate("id", "people.manager_id")
                        })
                })
        })
        .order_by("title")
        .build()
}

/// (person, name, [(manager, name)])
type PersonTree = Vec<(ObjectId, String, Vec<(ObjectId, String)>)>;
/// org -> [(member, person tree)], members by id.
type OrgTree = BTreeMap<ObjectId, Vec<(ObjectId, PersonTree)>>;

fn name_of(values: &[Value]) -> String {
    match values.first() {
        Some(Value::Text(name)) => name.clone(),
        other => panic!("expected a name, got {other:?}"),
    }
}

fn org_tree_of(mirror: &Mirror) -> OrgTree {
    let Some(descriptor) = mirror.descriptor.as_ref() else {
        return OrgTree::new();
    };
    mirror
        .rows
        .iter()
        .map(|(org, row)| {
            let values = decode_row(descriptor, &row.data).expect("decode org row");
            let mut members: Vec<(ObjectId, PersonTree)> = array_at(&values, CHAT_MEMBERS_COLUMN)
                .iter()
                .map(|member| {
                    let (member_id, member_values) = row_of(member);
                    let person = array_at(member_values, MEMBER_PERSON_COLUMN)
                        .iter()
                        .map(|person| {
                            let (person_id, person_values) = row_of(person);
                            let manager = array_at(person_values, PERSON_MANAGER_COLUMN)
                                .iter()
                                .map(|manager| {
                                    let (manager_id, manager_values) = row_of(manager);
                                    (manager_id, name_of(manager_values))
                                })
                                .collect();
                            (person_id, name_of(person_values), manager)
                        })
                        .collect();
                    (member_id, person)
                })
                .collect();
            members.sort_by_key(|(member, _)| *member);
            (*org, members)
        })
        .collect()
}

/// One engine of the three-level dual run.
struct ThreeLevelSide {
    routed: bool,
    qm: QueryManager,
    storage: MemoryStorage,
    sub: crate::query_manager::graph_nodes::output::QuerySubscriptionId,
    mirror: Mirror,
}

impl ThreeLevelSide {
    fn new(routed: bool) -> Self {
        let _engaged = engage(routed);
        let (mut qm, storage) = create_query_manager(SyncManager::new(), three_level_schema());
        let query = three_level_query(&qm);
        let sub = qm.subscribe(query).expect("subscribe orgs");
        Self {
            routed,
            qm,
            storage,
            sub,
            mirror: Mirror::default(),
        }
    }

    fn insert(&mut self, table: &str, id: ObjectId, values: &[Value]) {
        let _engaged = engage(self.routed);
        let schema = self.qm.schema_context().current_schema.clone();
        let branch = get_branch(&self.qm);
        self.qm
            .insert_on_branch_with_schema_and_write_context_and_id(
                &mut self.storage,
                table,
                &branch,
                values,
                Some(id),
                &schema,
                None,
                true,
            )
            .unwrap_or_else(|error| panic!("insert into {table}: {error:?}"));
    }

    fn update(&mut self, id: ObjectId, values: &[Value]) {
        let _engaged = engage(self.routed);
        self.qm
            .update(&mut self.storage, id, values)
            .unwrap_or_else(|error| panic!("update: {error:?}"));
    }

    fn delete(&mut self, id: ObjectId) {
        let _engaged = engage(self.routed);
        self.qm
            .delete(&mut self.storage, id)
            .unwrap_or_else(|error| panic!("delete: {error:?}"));
    }

    /// One `process()`; returns whether anything was emitted.
    fn pass(&mut self) -> bool {
        let _engaged = engage(self.routed);
        self.qm.process(&mut self.storage);
        let updates = self.qm.take_updates();
        for update in &updates {
            if update.subscription_id == self.sub {
                self.mirror.absorb(update);
            }
        }
        !updates.is_empty()
    }

    fn tree(&self) -> OrgTree {
        org_tree_of(&self.mirror)
    }
}

/// A grandchild under a forward reference arrives after the person naming it, changes, and is
/// deleted: each must reach the org's array in ONE settle, in both modes. The delete is the
/// case only the export chain can serve — the deleted manager cannot be loaded, so its write
/// routes by `held_by` alone, two levels up.
#[test]
fn a_grandchild_under_a_forward_reference_arrives_changes_and_leaves_in_one_settle() {
    // Not `each_mode`: the side pins the engine itself, call by call.
    for mode in [Mode::Routed, Mode::Broadcast] {
        let mut side = ThreeLevelSide::new(mode == Mode::Routed);
        let mut orgs = Vec::new();
        for org_index in 0..CHAT_COUNT {
            let org = ObjectId::new();
            side.insert(ORGS, org, &[Value::Text(format!("org-{org_index:04}"))]);
            let person = ObjectId::new();
            let boss = ObjectId::new();
            side.insert(
                PEOPLE,
                boss,
                &[
                    Value::Text(format!("boss-{org_index}")),
                    Value::Uuid(ObjectId::new()),
                ],
            );
            side.insert(
                PEOPLE,
                person,
                &[Value::Text(format!("p-{org_index}")), Value::Uuid(boss)],
            );
            side.insert(
                MEMBERS,
                ObjectId::new(),
                &[Value::Uuid(org), Value::Uuid(person)],
            );
            orgs.push(org);
        }
        side.pass();

        let late_manager = ObjectId::new();
        let person = ObjectId::new();
        side.insert(
            PEOPLE,
            person,
            &[Value::Text("waiting".into()), Value::Uuid(late_manager)],
        );
        let member = ObjectId::new();
        side.insert(
            MEMBERS,
            member,
            &[Value::Uuid(orgs[1]), Value::Uuid(person)],
        );
        side.pass();
        let managers_of = |side: &ThreeLevelSide| -> Vec<(ObjectId, String)> {
            side.tree()[&orgs[1]]
                .iter()
                .find(|(id, _)| *id == member)
                .and_then(|(_, person_tree)| person_tree.first())
                .map(|(_, _, managers)| managers.clone())
                .unwrap_or_else(|| panic!("{mode:?}: the waiting person is missing from org 1"))
        };
        assert_eq!(
            managers_of(&side),
            Vec::new(),
            "{mode:?}: a person naming a missing manager carries an empty manager array"
        );

        side.insert(
            PEOPLE,
            late_manager,
            &[Value::Text("late".into()), Value::Uuid(ObjectId::new())],
        );
        side.pass();
        assert_eq!(
            managers_of(&side),
            vec![(late_manager, "late".to_string())],
            "{mode:?}: a manager arriving after the person naming it must fill it in one settle"
        );

        side.update(
            late_manager,
            &[Value::Text("renamed".into()), Value::Uuid(ObjectId::new())],
        );
        side.pass();
        assert_eq!(
            managers_of(&side),
            vec![(late_manager, "renamed".to_string())],
            "{mode:?}: a manager's change must reach the org two levels up in one settle"
        );

        side.delete(late_manager);
        side.pass();
        assert_eq!(
            managers_of(&side),
            Vec::new(),
            "{mode:?}: a deleted manager — a row no loader can read any more — must leave the \
             org's array in one settle; only the ids the person node exported can route it"
        );
    }
}

/// The reference model of the three-level shape.
#[derive(Default)]
struct ThreeLevelModel {
    orgs: Vec<ObjectId>,
    /// member -> (org, person)
    members: BTreeMap<ObjectId, (ObjectId, ObjectId)>,
    /// person -> (name, manager)
    people: BTreeMap<ObjectId, (String, ObjectId)>,
    /// Ids named but never inserted: a later insert of one is a late arrival.
    dangling: Vec<ObjectId>,
}

impl ThreeLevelModel {
    fn tree(&self) -> OrgTree {
        let mut tree: OrgTree = self.orgs.iter().map(|org| (*org, Vec::new())).collect();
        for (member, (org, person)) in &self.members {
            let person_tree = self
                .people
                .get(person)
                .map(|(name, manager)| {
                    let managers = self
                        .people
                        .get(manager)
                        .map(|(manager_name, _)| vec![(*manager, manager_name.clone())])
                        .unwrap_or_default();
                    vec![(*person, name.clone(), managers)]
                })
                .unwrap_or_default();
            tree.get_mut(org)
                .expect("a member's org exists")
                .push((*member, person_tree));
        }
        tree
    }
}

fn run_three_level_differential(seed: u64) {
    let mut rng = Prng(seed | 1);
    let mut mint = Mint { seed, counter: 0 };
    let mut model = ThreeLevelModel::default();
    let mut sides = [ThreeLevelSide::new(true), ThreeLevelSide::new(false)];

    for step in 0..DIFFERENTIAL_STEPS {
        let people: Vec<ObjectId> = model.people.keys().copied().collect();
        let members: Vec<ObjectId> = model.members.keys().copied().collect();
        // A target: present, named-but-missing, or fresh (and so missing).
        let target = |rng: &mut Prng, model: &mut ThreeLevelModel, mint: &mut Mint| -> ObjectId {
            match rng.below(3) {
                0 => rng.pick(&people).unwrap_or_else(|| mint.next()),
                1 => rng.pick(&model.dangling).unwrap_or_else(|| mint.next()),
                _ => {
                    let fresh = mint.next();
                    model.dangling.push(fresh);
                    fresh
                }
            }
        };
        let roll = rng.below(100);
        let op: Box<dyn Fn(&mut ThreeLevelSide)> = match roll {
            0..=5 => {
                let org = mint.next();
                model.orgs.push(org);
                let title = format!("org-{step:04}");
                Box::new(move |side| side.insert(ORGS, org, &[Value::Text(title.clone())]))
            }
            6..=25 if !model.orgs.is_empty() => {
                let member = mint.next();
                let org = rng.pick(&model.orgs).expect("an org");
                let person = target(&mut rng, &mut model, &mut mint);
                model.members.insert(member, (org, person));
                Box::new(move |side| {
                    side.insert(MEMBERS, member, &[Value::Uuid(org), Value::Uuid(person)])
                })
            }
            26..=50 => {
                // A person: a fresh id, or one a member or a person is waiting for.
                let id = if !model.dangling.is_empty() && rng.below(2) == 0 {
                    model.dangling.swap_remove(rng.below(model.dangling.len()))
                } else {
                    mint.next()
                };
                let manager = target(&mut rng, &mut model, &mut mint);
                let name = format!("person-{step}");
                model.people.insert(id, (name.clone(), manager));
                Box::new(move |side| {
                    side.insert(
                        PEOPLE,
                        id,
                        &[Value::Text(name.clone()), Value::Uuid(manager)],
                    )
                })
            }
            51..=70 if !people.is_empty() => {
                let id = rng.pick(&people).expect("a person");
                let manager = if rng.below(2) == 0 {
                    model.people[&id].1
                } else {
                    target(&mut rng, &mut model, &mut mint)
                };
                let name = format!("renamed-{step}");
                model.people.insert(id, (name.clone(), manager));
                Box::new(move |side| {
                    side.update(id, &[Value::Text(name.clone()), Value::Uuid(manager)])
                })
            }
            71..=80 if !people.is_empty() => {
                let id = rng.pick(&people).expect("a person");
                model.people.remove(&id);
                Box::new(move |side| side.delete(id))
            }
            81..=90 if !members.is_empty() => {
                let id = rng.pick(&members).expect("a member");
                let person = target(&mut rng, &mut model, &mut mint);
                let org = model.members[&id].0;
                model.members.insert(id, (org, person));
                Box::new(move |side| side.update(id, &[Value::Uuid(org), Value::Uuid(person)]))
            }
            91..=99 if !members.is_empty() => {
                let id = rng.pick(&members).expect("a member");
                model.members.remove(&id);
                Box::new(move |side| side.delete(id))
            }
            _ => Box::new(|_| {}),
        };
        for side in &mut sides {
            op(side);
        }
        // Lockstep passes: routed and broadcast must agree after EVERY pass, and the first
        // pass must already serve the model.
        for pass in 0..PASS_BOUND {
            for side in &mut sides {
                side.pass();
            }
            let (routed, broadcast) = (sides[0].tree(), sides[1].tree());
            assert!(
                routed == broadcast,
                "seed {seed:#x} step {step} roll {roll} pass {pass}: routed and broadcast \
                 engines serve different three-level trees"
            );
            if pass == 0 {
                assert_eq!(
                    routed,
                    model.tree(),
                    "seed {seed:#x} step {step} roll {roll}: the first settle must serve the \
                     model's three-level tree"
                );
            }
        }
    }
}

/// Late arrivals, changes and deletes at every level of a two-hop forward-reference chain,
/// routed vs broadcast in lockstep and both against the model.
#[test]
fn a_three_level_forward_reference_differential_routed_vs_broadcast() {
    for seed in DIFFERENTIAL_SEEDS {
        run_three_level_differential(seed);
    }
}

/// A forward reference through a `UUID[]` foreign key (`files.parts -> file_parts` in the app's
/// schema): a nested include whose instances probe EVERY id of the parent's array. A late id of
/// the array, a change of one, and its delete must reach the outer instance through the ids the
/// nested node exports — element by element.
#[test]
fn a_uuid_array_forward_reference_routes_each_element() {
    let schema = {
        let mut schema = Schema::new();
        schema.insert(
            TableName::new(ORGS),
            RowDescriptor::new(vec![ColumnDescriptor::new("title", ColumnType::Text)]).into(),
        );
        schema.insert(
            TableName::new(MEMBERS),
            RowDescriptor::new(vec![
                ColumnDescriptor::new("org_id", ColumnType::Uuid),
                ColumnDescriptor::new(
                    "person_ids",
                    ColumnType::Array {
                        element: Box::new(ColumnType::Uuid),
                    },
                ),
            ])
            .into(),
        );
        schema.insert(
            TableName::new(PEOPLE),
            RowDescriptor::new(vec![ColumnDescriptor::new("name", ColumnType::Text)]).into(),
        );
        schema
    };
    for mode in [Mode::Routed, Mode::Broadcast] {
        let _engaged = engage(mode == Mode::Routed);
        let (mut qm, mut storage) = create_query_manager(SyncManager::new(), schema.clone());
        let mut orgs = Vec::new();
        for org_index in 0..CHAT_COUNT {
            let org = qm
                .insert(
                    &mut storage,
                    ORGS,
                    &[Value::Text(format!("org-{org_index:04}"))],
                )
                .expect("insert org")
                .row_id;
            let person = qm
                .insert(
                    &mut storage,
                    PEOPLE,
                    &[Value::Text(format!("p-{org_index}"))],
                )
                .expect("insert person")
                .row_id;
            qm.insert(
                &mut storage,
                MEMBERS,
                &[Value::Uuid(org), Value::Array(vec![Value::Uuid(person)])],
            )
            .expect("insert member");
            orgs.push(org);
        }
        let query = qm
            .query(ORGS)
            .with_array("members", |sub| {
                sub.from(MEMBERS)
                    .correlate("org_id", "orgs.id")
                    .with_array("people", |people| {
                        people.from(PEOPLE).correlate("id", "members.person_ids")
                    })
            })
            .order_by("title")
            .build();
        let sub_id = qm.subscribe(query).expect("subscribe");
        let mut mirror = Mirror::default();
        let mut settle =
            |qm: &mut QueryManager, storage: &mut MemoryStorage, mirror: &mut Mirror| {
                qm.process(storage);
                for update in qm.take_updates() {
                    if update.subscription_id == sub_id {
                        mirror.absorb(&update);
                    }
                }
            };
        settle(&mut qm, &mut storage, &mut mirror);

        let present = qm
            .insert(&mut storage, PEOPLE, &[Value::Text("present".into())])
            .expect("insert present person")
            .row_id;
        let late = ObjectId::new();
        let member = qm
            .insert(
                &mut storage,
                MEMBERS,
                &[
                    Value::Uuid(orgs[1]),
                    Value::Array(vec![Value::Uuid(present), Value::Uuid(late)]),
                ],
            )
            .expect("insert a member naming a present and a missing person")
            .row_id;
        settle(&mut qm, &mut storage, &mut mirror);
        let people_of = |mirror: &Mirror| -> Vec<String> {
            let (_, values) = mirror
                .decoded()
                .into_iter()
                .find(|(id, _)| *id == orgs[1])
                .unwrap_or_else(|| panic!("{mode:?}: org 1 missing from the mirror"));
            let mut names: Vec<String> = array_at(&values, CHAT_MEMBERS_COLUMN)
                .iter()
                .map(row_of)
                .find(|(id, _)| *id == member)
                .map(|(_, member_values)| {
                    array_at(member_values, MEMBER_PERSON_COLUMN)
                        .iter()
                        .map(|person| name_of(row_of(person).1))
                        .collect()
                })
                .unwrap_or_else(|| panic!("{mode:?}: the member is missing from org 1"));
            names.sort();
            names
        };
        assert_eq!(
            people_of(&mirror),
            vec!["present".to_string()],
            "{mode:?}: only the present id resolves"
        );

        let schema_now = qm.schema_context().current_schema.clone();
        let branch = get_branch(&qm);
        qm.insert_on_branch_with_schema_and_write_context_and_id(
            &mut storage,
            PEOPLE,
            &branch,
            &[Value::Text("late".into())],
            Some(late),
            &schema_now,
            None,
            true,
        )
        .expect("insert the late person");
        settle(&mut qm, &mut storage, &mut mirror);
        assert_eq!(
            people_of(&mirror),
            vec!["late".to_string(), "present".to_string()],
            "{mode:?}: the second element of the array arriving late must fill it in one settle"
        );

        qm.update(&mut storage, late, &[Value::Text("later".into())])
            .expect("rename the late person");
        settle(&mut qm, &mut storage, &mut mirror);
        assert_eq!(
            people_of(&mirror),
            vec!["later".to_string(), "present".to_string()],
            "{mode:?}: a change of one element must reach the org in one settle"
        );

        qm.delete(&mut storage, late)
            .expect("delete the late person");
        settle(&mut qm, &mut storage, &mut mirror);
        assert_eq!(
            people_of(&mirror),
            vec!["present".to_string()],
            "{mode:?}: a deleted element must leave the org's array in one settle"
        );
    }
}
