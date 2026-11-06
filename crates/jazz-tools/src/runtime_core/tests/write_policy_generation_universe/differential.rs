//! Randomised membership histories across deployments, judged against a model of the
//! intended semantics.
//!
//! The gates next door encode the cases we thought of. This one lets a generator choose
//! the interleavings: memberships joined, handed to someone else, moved to another chat
//! and left — in a batch or not, from the device on the current schema or from one that
//! stayed on the first — with up to two deployments in between, the second adding a
//! `status` column the rules then require. After every step alice tries to post in a
//! chat, and the verdict is compared with a model stated without the engine:
//!
//! > Alice may post in a chat exactly when a membership of hers in that chat exists:
//! > the newest version of each membership is the one that counts, and a deleted one is
//! > gone for good. A deployment changes where memberships are stored and nothing about
//! > which of them exist.
//!
//! `SqliteStorage` throughout, as the gates: `MemoryStorage` never runs the locator ladder.

use super::*;

const SEEDS: [u64; 6] = [
    0x5750_4755_0000_0001,
    0x5750_4755_0000_0002,
    0x5750_4755_0000_0003,
    0x5750_4755_0000_0004,
    0x5750_4755_0000_0005,
    0x5750_4755_0000_0006,
];
const OPS_PER_SEED: usize = 120;
const CHATS_IN_PLAY: [&str; 3] = ["chat-a", "chat-b", "chat-c"];
const PEOPLE: [&str; 2] = [ME, "bob"];

struct Xorshift(u64);

impl Xorshift {
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

    fn pick<'a>(&mut self, items: &[&'a str]) -> &'a str {
        items[self.below(items.len())]
    }
}

/// The generations a history walks through: the first, one that adds another table, one
/// that adds `status` (default 'active') to memberships.
fn generation(index: usize) -> Schema {
    match index {
        0 => schema_with(false, false),
        1 => schema_with(false, true),
        _ => schema_with(true, true),
    }
}

fn lens_into(index: usize) -> Lens {
    let (old, new) = (generation(index - 1), generation(index));
    match index {
        1 => crate::schema_manager::auto_lens::generate_lens(&old, &new),
        _ => Lens::new(
            SchemaHash::compute(&old),
            SchemaHash::compute(&new),
            LensTransform::with_ops(vec![LensOp::AddColumn {
                table: MEMBERS.to_string(),
                column: "status".to_string(),
                column_type: ColumnType::Text,
                default: text("active"),
            }]),
        ),
    }
}

/// One membership as the model holds it: its newest version, and the device that wrote it.
#[derive(Debug, Clone)]
struct Membership {
    chat: &'static str,
    user: &'static str,
    alive: bool,
    written_by: From,
}

/// Which device a write comes from.
#[derive(Debug, Clone, Copy, PartialEq)]
enum From {
    Current,
    Old,
}

struct History {
    device: Device,
    server: Core,
    current: Core,
    old: Core,
    server_id: ServerId,
    current_id: ClientId,
    old_id: ClientId,
    generation: usize,
    model: Vec<(ObjectId, Membership)>,
    log: Vec<String>,
}

impl History {
    fn new(device: Device) -> Self {
        let mut server = server_runtime(generation(0), sqlite());
        let current = client_runtime(generation(0), sqlite(), device);
        let old = device_runtime(generation(0), sqlite());
        let (server_id, current_id, old_id) = (ServerId::new(), ClientId::new(), ClientId::new());
        server.add_client(current_id, Some(Session::new(ME)));
        server.add_client(old_id, Some(Session::new(ME)));
        let mut history = Self {
            device,
            server,
            current,
            old,
            server_id,
            current_id,
            old_id,
            generation: 0,
            model: Vec::new(),
            log: Vec::new(),
        };
        history.current.add_server(server_id);
        history.old.add_server(server_id);
        follow_memberships(&mut history.current);
        follow_memberships(&mut history.old);
        history.sync();
        history
    }

    /// Every frame delivered to its destination until all three are quiet; the server's
    /// refusals on the way. `pump` serves one client and drops what the server sent the
    /// other, which leaves a device without the memberships the other one wrote.
    fn exchange(&mut self) -> Vec<String> {
        let mut refusals = Vec::new();
        for _ in 0..16 {
            let mut any = false;
            for (device, device_id) in [
                (&mut self.current, self.current_id),
                (&mut self.old, self.old_id),
            ] {
                device.batched_tick();
                for entry in device.sync_sender().take() {
                    if entry.destination == Destination::Server(self.server_id) {
                        any = true;
                        self.server.park_sync_message(InboxEntry {
                            source: Source::Client(device_id),
                            payload: entry.payload,
                        });
                    }
                }
            }
            self.server.batched_tick();
            self.server.immediate_tick();
            self.server.batched_tick();
            for entry in self.server.sync_sender().take() {
                if let SyncPayload::BatchFate {
                    fate: BatchFate::Rejected { reason, .. },
                } = &entry.payload
                {
                    refusals.push(reason.clone());
                }
                let device = match entry.destination {
                    Destination::Client(id) if id == self.current_id => &mut self.current,
                    Destination::Client(id) if id == self.old_id => &mut self.old,
                    _ => continue,
                };
                any = true;
                device.park_sync_message(InboxEntry {
                    source: Source::Server(self.server_id),
                    payload: entry.payload,
                });
            }
            for device in [&mut self.current, &mut self.old] {
                device.batched_tick();
                device.immediate_tick();
            }
            if !any {
                break;
            }
        }
        refusals
    }

    /// Membership writes are open: a refusal of one is the harness, not the verdict.
    fn sync(&mut self) {
        let refusals = self.exchange();
        assert!(
            refusals.is_empty(),
            "control: a membership write was refused ({refusals:?})\nhistory:\n{}",
            self.log.join("\n")
        );
    }

    fn device(&mut self, from: From) -> (&mut Core, ClientId) {
        match from {
            From::Current => (&mut self.current, self.current_id),
            From::Old => (&mut self.old, self.old_id),
        }
    }

    /// A membership's columns as `from` writes them: the old device's layout has no status.
    fn membership_values(&self, from: From, chat: &str, user: &str) -> HashMap<String, Value> {
        let mut values = HashMap::from([
            ("chatId".to_string(), text(chat)),
            ("userId".to_string(), text(user)),
        ]);
        if from == From::Current && self.generation >= 2 {
            values.insert("status".to_string(), text("active"));
        }
        values
    }

    fn deploy(&mut self) {
        self.generation += 1;
        let next = generation(self.generation);
        let server_storage =
            std::mem::replace(&mut self.server, device_runtime(generation(0), sqlite()))
                .into_storage();
        let current_storage =
            std::mem::replace(&mut self.current, device_runtime(generation(0), sqlite()))
                .into_storage();
        self.server = server_runtime(next.clone(), server_storage);
        self.server
            .publish_lens(&lens_into(self.generation))
            .expect("lens publishes");
        self.current = client_runtime(next, current_storage, self.device);
        self.server
            .add_client(self.current_id, Some(Session::new(ME)));
        self.server.add_client(self.old_id, Some(Session::new(ME)));
        self.current.add_server(self.server_id);
        follow_memberships(&mut self.current);
        self.old.remove_server(self.server_id);
        self.old.add_server(self.server_id);
        self.server.immediate_tick();
        self.current.immediate_tick();
        self.old.immediate_tick();
        self.sync();
    }

    fn join(&mut self, from: From, chat: &'static str, user: &'static str) {
        let values = self.membership_values(from, chat, user);
        let (device, _) = self.device(from);
        let ((id, _), _) = device
            .insert(MEMBERS, values, Some(&World::alice()))
            .expect("memberships are open");
        self.sync();
        self.model.push((
            id,
            Membership {
                chat,
                user,
                alive: true,
                written_by: from,
            },
        ));
    }

    /// `column` of membership `index` set to `value`; false when `from` could not write it.
    fn rewrite(&mut self, from: From, index: usize, column: &str, value: &'static str) -> bool {
        let id = self.model[index].0;
        let (device, _) = self.device(from);
        let written = device
            .update(
                id,
                vec![(column.to_string(), text(value))],
                Some(&World::alice()),
            )
            .is_ok();
        self.sync();
        if written {
            let membership = &mut self.model[index].1;
            membership.written_by = from;
            match column {
                "chatId" => membership.chat = value,
                _ => membership.user = value,
            }
        }
        written
    }

    fn leave(&mut self, from: From, index: usize, in_batch: bool) -> bool {
        let id = self.model[index].0;
        let (device, _) = self.device(from);
        let written = if in_batch {
            let batch = device.begin_batch(BatchMode::Direct);
            let context = World::alice()
                .with_batch_mode(BatchMode::Direct)
                .with_batch_id(batch);
            let deleted = device.delete(id, Some(&context)).is_ok();
            device.commit_batch(batch).expect("the batch commits");
            deleted
        } else {
            device.delete(id, Some(&World::alice())).is_ok()
        };
        self.sync();
        if written {
            self.model[index].1.alive = false;
        }
        written
    }

    /// What the model says of alice posting in `chat`.
    fn may_post(&self, chat: &str) -> bool {
        self.model.iter().any(|(_, membership)| {
            membership.alive && membership.chat == chat && membership.user == ME
        })
    }

    fn post(&mut self, from: From, chat: &str) -> Verdict {
        let (device, _) = self.device(from);
        let written = device
            .insert(MESSAGES, message(chat), Some(&World::alice()))
            .map(|_| ())
            .map_err(|error| format!("{error:?}"));
        verdict_of(written, || self.exchange())
    }

    /// The memberships of `chat` an end serves alice, by model index where the model knows
    /// the id: what a failure names first.
    fn served(&mut self, end: &str, chat: &str) -> Vec<String> {
        let core = match end {
            "server" => &mut self.server,
            "current" => &mut self.current,
            _ => &mut self.old,
        };
        let manager = core.schema_manager_mut().query_manager_mut();
        let query = manager
            .query(MEMBERS)
            .filter_eq("chatId", text(chat))
            .build();
        let sub = manager
            .subscribe_with_session(query, Some(Session::new(ME)), None)
            .expect("the read subscribes");
        core.immediate_tick();
        let rows = core
            .schema_manager()
            .query_manager()
            .get_subscription_results(sub);
        core.schema_manager_mut()
            .query_manager_mut()
            .unsubscribe_with_sync(sub);
        core.sync_sender().take();
        rows.into_iter()
            .map(|(id, values)| {
                let index = self.model.iter().position(|(known, _)| *known == id);
                format!("#{index:?} {values:?}")
            })
            .collect()
    }

    fn alive_indices(&self) -> Vec<usize> {
        self.model
            .iter()
            .enumerate()
            .filter(|(_, (_, membership))| membership.alive)
            .map(|(index, _)| index)
            .collect()
    }
}

/// A standing read of every membership, as the app's chat list keeps one: without it a
/// device holds only the rows it wrote, and a membership written on the other device never
/// reaches the rules that judge this one's posts. A reconnect replays it upstream.
fn follow_memberships(device: &mut Core) {
    device
        .subscribe(Query::new(MEMBERS), |_| {}, Some(Session::new(ME)))
        .expect("memberships subscribe");
    device.immediate_tick();
}

fn run(seed: u64, device: Device) {
    let mut rng = Xorshift(seed);
    let mut history = History::new(device);
    let mut skipped = 0usize;
    let mut carved_out = 0usize;
    for step in 0..OPS_PER_SEED {
        let from = if history.generation > 0 && rng.below(3) == 0 {
            From::Old
        } else {
            From::Current
        };
        let alive = history.alive_indices();
        let roll = rng.below(100);
        let op = if roll < 4 && history.generation < 2 && step > 10 {
            history.deploy();
            format!("deploy → generation {}", history.generation)
        } else if roll < 30 || alive.is_empty() {
            let chat = rng.pick(&CHATS_IN_PLAY);
            let user = rng.pick(&PEOPLE);
            history.join(from, chat, user);
            format!("{from:?} join {chat} as {user}")
        } else {
            let index = alive[rng.below(alive.len())];
            let (done, what) = match rng.below(4) {
                0 => {
                    let user = rng.pick(&PEOPLE);
                    (
                        history.rewrite(from, index, "userId", user),
                        format!("hand to {user}"),
                    )
                }
                1 => {
                    let chat = rng.pick(&CHATS_IN_PLAY);
                    (
                        history.rewrite(from, index, "chatId", chat),
                        format!("move to {chat}"),
                    )
                }
                2 => (history.leave(from, index, false), "leave".to_string()),
                // Not generated: a batch leaving a membership the other device wrote last,
                // after a deployment (`a_membership_left_in_a_batch_on_one_device_is_left_on_
                // the_other`, a device convergence defect apart from these write checks).
                _ if history.generation > 0 && history.model[index].1.written_by != from => {
                    carved_out += 1;
                    (
                        history.leave(from, index, false),
                        "leave (not in a batch: the other device wrote it last)".to_string(),
                    )
                }
                _ => (
                    history.leave(from, index, true),
                    "leave in a batch".to_string(),
                ),
            };
            if !done {
                skipped += 1;
            }
            format!(
                "{from:?} membership #{index}: {what}{}",
                if done { "" } else { " (not written)" }
            )
        };
        history.log.push(op);

        let chat = rng.pick(&CHATS_IN_PLAY);
        let poster = if history.generation > 0 && rng.below(2) == 0 {
            From::Old
        } else {
            From::Current
        };
        let expected = history.may_post(chat);
        let verdict = history.post(poster, chat);
        let sent = verdict == Verdict::Sent;
        // A device holding the rules refuses before it sends what the server would reject.
        let refused_where_judged = match (poster, device) {
            (From::Current, Device::ExplicitHead) => {
                matches!(verdict, Verdict::RefusedLocally(_))
            }
            _ => matches!(verdict, Verdict::RejectedByServer(_)),
        };
        if !(if expected { sent } else { refused_where_judged }) {
            let views: Vec<String> = ["server", "current", "old"]
                .into_iter()
                .map(|end| format!("{end} serves {:?}", history.served(end, chat)))
                .collect();
            panic!(
                "seed {seed:#x} ({device:?}), step {step}: {poster:?} posting in {chat} came \
                 out {verdict:?}, the model says {}.\n{}\nmemberships: {:?}\nhistory:\n{}",
                if expected { "allowed" } else { "refused" },
                views.join("\n"),
                history.model,
                history.log.join("\n")
            );
        }
        history
            .log
            .push(format!("  {poster:?} posts in {chat}: {verdict:?}"));
    }
    eprintln!(
        "write-universe differential seed {seed:#x} ({device:?}): {OPS_PER_SEED} steps, \
         generation {}, {} memberships, {skipped} writes not made, {carved_out} batch \
         leaves made plain leaves",
        history.generation,
        history.model.len()
    );
}

#[test]
fn memberships_across_deployments_decide_posts_as_the_model_does() {
    for seed in SEEDS {
        run(seed, Device::ServerJudges);
    }
}

#[test]
fn memberships_across_deployments_decide_posts_as_the_model_does_on_the_phone() {
    for seed in SEEDS {
        run(seed, Device::ExplicitHead);
    }
}

/// A membership left in a batch on one device after a deployment, while the other device
/// wrote its newest version: the server serves it as left, and so must the other device,
/// whose own rules judge alice's posts by it. Measured identical on 6776c0e473: the batch's
/// tombstone lands on the leaver's generation and the other device keeps serving its own
/// generation's copy; with no deployment, or with a plain leave, every end converges.
#[test]
#[ignore = "pre-existing device convergence defect, tracked apart from the write universe"]
fn a_membership_left_in_a_batch_on_one_device_is_left_on_the_other() {
    for (writer, leaver) in [(From::Current, From::Old), (From::Old, From::Current)] {
        let mut history = History::new(Device::ExplicitHead);
        history.join(From::Current, "chat-c", ME);
        history.deploy();
        assert!(
            history.rewrite(writer, 0, "userId", ME),
            "control: the rewrite"
        );
        assert!(history.leave(leaver, 0, true), "control: the leave");
        assert!(
            history.served("server", "chat-c").is_empty(),
            "control: the server serves the membership as left"
        );
        for end in ["current", "old"] {
            assert!(
                history.served(end, "chat-c").is_empty(),
                "{leaver:?} left the membership {writer:?} wrote last, in a batch, yet the \
                 {end} device still serves it"
            );
        }
    }
}
