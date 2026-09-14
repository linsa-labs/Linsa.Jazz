//! A server subscription follows the last word its client sent for it, in arrival order, never
//! how those words were grouped into passes or whether the subscription had to wait in the queue
//! for a later pass.
//!
//! The gates in `server_subscriptions.rs` pin single orderings. This drives random sequences of
//! subscriptions (re-subscriptions with another shape included), withdrawals, passes and outbox
//! drains through a hub with one upstream server and several downstream clients. A pass whose
//! outbox already holds a full initial replay admits one more registration and defers the rest,
//! and an undrained outbox stays full, so subscriptions really do wait across passes. The hub is
//! checked against a model:
//!
//! - after every pass, a key whose last word is a withdrawal is not registered;
//! - on every drain, the hub never withdraws a query id upstream that it has not subscribed there;
//! - at every checkpoint, once the outbox is drained, nothing is left in the queue, the registered
//!   keys are exactly those whose last word is a subscription, each with that word's shape, and a
//!   query id is subscribed upstream exactly when its key is registered.
//!
//! One client subscribes local-only under the same query ids as another, so a withdrawal that
//! matched a queued subscription by query id alone would take the other client's. Its
//! subscriptions never go upstream, so only its registrations are compared.
//!
//! Clients withdraw only what they last subscribed, as a runtime does, and every subscription
//! here is one the hub accepts: withdrawing a subscription the hub rejected is forwarded upstream
//! as if it had been registered with full propagation, which this model does not cover. Among the
//! full-propagation clients query ids are disjoint: a hub forwards downstream ids under its own
//! client id, so two of them using one id would share an upstream key, which is a separate defect
//! this file does not model. Client removal is left out for the same reason: it drops a client's
//! registrations without withdrawing them upstream.

use super::*;

use std::collections::HashMap;

use crate::query_manager::query::Query;
use crate::sync_manager::{
    ClientId, Destination, InboxEntry, QueryId, QueryPropagation, ServerId, Source, SyncPayload,
};

/// Clients below this subscribe with full propagation, under disjoint query ids.
const FULL_CLIENTS: usize = 3;
/// Subscribes local-only, under client 0's query ids.
const LOCAL_ONLY_CLIENT: usize = FULL_CLIENTS;
const CLIENTS: usize = FULL_CLIENTS + 1;
const IDS_PER_CLIENT: u64 = 4;
const SEEDS: u64 = 48;
const STEPS: usize = 300;

/// xorshift64*, so a failing seed replays exactly.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn below(&mut self, n: u64) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D) % n
    }
}

/// The last thing a client said about one of its query ids.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Word {
    Subscribe { shape: usize },
    Withdraw,
}

struct Run {
    seed: u64,
    hub: QueryManager,
    storage: MemoryStorage,
    upstream_id: ServerId,
    clients: Vec<ClientId>,
    shapes: Vec<Query>,
    last_word: HashMap<(usize, u64), Word>,
    /// Passes run so far, and how many had run when each key was last subscribed.
    passes: usize,
    subscribed_at: HashMap<(usize, u64), usize>,
    /// Whether each query id is subscribed upstream, replayed from the drained outbox.
    upstream: HashMap<QueryId, bool>,
    /// Random-phase passes after which a key whose last word is a subscription was unregistered.
    waited: usize,
    /// Withdrawals of a subscription that a pass had already deferred to the queue.
    withdrawn_while_waiting: usize,
}

impl Run {
    fn new(seed: u64) -> Self {
        let (mut hub, storage) = create_query_manager(SyncManager::new(), test_schema());
        let upstream_id = ServerId::new();
        connect_server(&mut hub, &storage, upstream_id);
        let clients: Vec<ClientId> = (0..CLIENTS).map(|_| ClientId::new()).collect();
        for &client_id in &clients {
            connect_client(&mut hub, &storage, client_id);
        }
        let shapes = vec![
            hub.query("users").build(),
            hub.query("users")
                .filter_gt("score", Value::Integer(50))
                .build(),
        ];
        let mut run = Self {
            seed,
            hub,
            storage,
            upstream_id,
            clients,
            shapes,
            last_word: HashMap::new(),
            passes: 0,
            subscribed_at: HashMap::new(),
            upstream: HashMap::new(),
            waited: 0,
            withdrawn_while_waiting: 0,
        };
        run.pass(0, false);
        run.drain(0);
        run
    }

    /// Disjoint across the full-propagation clients; the local-only client reuses client 0's.
    fn query_id(client: usize, id: u64) -> QueryId {
        let owner = if client == LOCAL_ONLY_CLIENT {
            0
        } else {
            client
        };
        QueryId(owner as u64 * 100 + id)
    }

    fn propagation(client: usize) -> QueryPropagation {
        if client == LOCAL_ONLY_CLIENT {
            QueryPropagation::LocalOnly
        } else {
            QueryPropagation::Full
        }
    }

    fn subscribe(&mut self, client: usize, id: u64, shape: usize) {
        self.hub.sync_manager_mut().push_inbox(InboxEntry {
            source: Source::Client(self.clients[client]),
            payload: SyncPayload::QuerySubscription {
                query_id: Self::query_id(client, id),
                query: Box::new(self.shapes[shape].clone()),
                session: None,
                required_tier: None,
                propagation: Self::propagation(client),
                policy_context_tables: vec![],
            },
        });
        self.last_word
            .insert((client, id), Word::Subscribe { shape });
        self.subscribed_at.insert((client, id), self.passes);
    }

    /// A pass processes its whole inbox, so a subscription that is still not registered with its
    /// shape after a pass has run was deferred to the queue rather than left unread.
    fn is_waiting(&self, client: usize, id: u64) -> bool {
        let Some(Word::Subscribe { shape }) = self.last_word.get(&(client, id)) else {
            return false;
        };
        let key = (self.clients[client], Self::query_id(client, id));
        let passed = self
            .subscribed_at
            .get(&(client, id))
            .is_some_and(|&at| self.passes > at);
        passed
            && self
                .hub
                .server_subscriptions
                .get(&key)
                .map(|sub| &sub.query)
                != Some(&self.shapes[*shape])
    }

    /// Every waiting key, in a stable order so a seed replays exactly.
    fn waiting_keys(&self) -> Vec<(usize, u64)> {
        let mut keys: Vec<(usize, u64)> = self
            .last_word
            .keys()
            .copied()
            .filter(|&(client, id)| self.is_waiting(client, id))
            .collect();
        keys.sort_unstable();
        keys
    }

    fn withdraw(&mut self, client: usize, id: u64) {
        if self.is_waiting(client, id) {
            self.withdrawn_while_waiting += 1;
        }
        self.hub.sync_manager_mut().push_inbox(InboxEntry {
            source: Source::Client(self.clients[client]),
            payload: SyncPayload::QueryUnsubscription {
                query_id: Self::query_id(client, id),
            },
        });
        self.last_word.insert((client, id), Word::Withdraw);
    }

    fn pass(&mut self, step: usize, count_waits: bool) {
        self.hub.process(&mut self.storage);
        self.passes += 1;
        for (&(client, id), &word) in &self.last_word {
            let key = (self.clients[client], Self::query_id(client, id));
            let registered = self.hub.server_subscriptions.contains_key(&key);
            match word {
                Word::Withdraw => assert!(
                    !registered,
                    "seed {} step {step}: client {client} withdrew query {id} last, yet it is \
                     registered",
                    self.seed
                ),
                Word::Subscribe { .. } => {
                    if !registered && count_waits {
                        self.waited += 1;
                    }
                }
            }
        }
    }

    fn drain(&mut self, step: usize) {
        for entry in self.hub.sync_manager_mut().take_outbox() {
            if entry.destination != Destination::Server(self.upstream_id) {
                continue;
            }
            match entry.payload {
                SyncPayload::QuerySubscription { query_id, .. } => {
                    self.upstream.insert(query_id, true);
                }
                SyncPayload::QueryUnsubscription { query_id } => {
                    let subscribed = self.upstream.insert(query_id, false).unwrap_or(false);
                    assert!(
                        subscribed,
                        "seed {} step {step}: the hub withdrew query {} upstream without having \
                         subscribed it there",
                        self.seed, query_id.0
                    );
                }
                _ => {}
            }
        }
    }

    /// Drain and pass until the queue has emptied, then compare the whole hub with the model.
    fn settle(&mut self, step: usize) {
        for _ in 0..4 {
            self.drain(step);
            self.pass(step, false);
        }
        self.drain(step);
        assert!(
            !self.hub.sync_manager().has_pending_query_subscriptions(),
            "seed {} step {step}: subscriptions are still queued after the checkpoint passes, so \
             the hub is not at rest for the comparison",
            self.seed
        );
        for client in 0..CLIENTS {
            for id in 1..=IDS_PER_CLIENT {
                let key = (self.clients[client], Self::query_id(client, id));
                let registered = self
                    .hub
                    .server_subscriptions
                    .get(&key)
                    .map(|sub| &sub.query);
                let expected = match self.last_word.get(&(client, id)) {
                    Some(Word::Subscribe { shape }) => Some(&self.shapes[*shape]),
                    _ => None,
                };
                assert_eq!(
                    registered, expected,
                    "seed {} step {step}: client {client} query {id} once the queue emptied",
                    self.seed
                );
                if client == LOCAL_ONLY_CLIENT {
                    continue;
                }
                assert_eq!(
                    self.upstream.get(&key.1).copied().unwrap_or(false),
                    expected.is_some(),
                    "seed {} step {step}: upstream state of client {client} query {id}",
                    self.seed
                );
            }
        }
    }
}

#[test]
fn a_server_subscription_follows_the_last_word_its_client_sent() {
    let mut waited = 0;
    let mut withdrawn_while_waiting = 0;
    for seed in 0..SEEDS {
        let mut run = Run::new(seed);
        let mut rng = Rng::new(seed);
        // Odd seeds never drain between checkpoints and reach them less often, so the outbox
        // fills past the initial-replay cap and registrations wait in the queue across passes.
        let (drain_per_mille, settle_every) = if seed % 2 == 0 { (250, 50) } else { (0, 150) };
        for step in 1..=STEPS {
            let client = rng.below(CLIENTS as u64) as usize;
            let id = 1 + rng.below(IDS_PER_CLIENT);
            let roll = rng.below(1000);
            let subscribed = matches!(
                run.last_word.get(&(client, id)),
                Some(Word::Subscribe { .. })
            );
            // Some withdrawals aim at a subscription a pass has deferred, which a uniform pick
            // reaches too rarely to rely on.
            let waiting = if (350..425).contains(&roll) {
                run.waiting_keys()
            } else {
                Vec::new()
            };
            if roll < 350 {
                let shape = rng.below(2) as usize;
                run.subscribe(client, id, shape);
            } else if !waiting.is_empty() {
                let (client, id) = waiting[rng.below(waiting.len() as u64) as usize];
                run.withdraw(client, id);
            } else if roll < 600 && subscribed {
                run.withdraw(client, id);
            } else if (600..600 + drain_per_mille).contains(&roll) {
                run.drain(step);
            } else {
                run.pass(step, true);
            }
            if step % settle_every == 0 {
                run.settle(step);
            }
        }
        waited += run.waited;
        withdrawn_while_waiting += run.withdrawn_while_waiting;
    }
    assert!(
        waited > 0,
        "fixture: no subscription ever waited in the queue, so deferral was never exercised"
    );
    assert!(
        withdrawn_while_waiting > 0,
        "fixture: no subscription was withdrawn while deferred, so the cancel never had to find \
         one in the queue"
    );
}
