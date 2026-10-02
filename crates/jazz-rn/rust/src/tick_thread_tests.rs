//! Where the engine's scheduled tick runs, and what the JS thread is promised about it.
//!
//! The test's own thread plays the JS thread: it runs a loop that services what other
//! threads post to it, the way the generated glue does. A callback method called on
//! another thread is posted to that loop and its caller waits until it has run, so a
//! design that calls JS while holding something the JS thread needs is a test that
//! times out here rather than an app that hangs.
//!
//! A server core in the same process feeds the runtime the way the transport does: what
//! it sends is parked in the client's inbox, which schedules the client's tick.

use std::any::Any;
use std::collections::BTreeMap;
use std::sync::atomic::AtomicUsize;
use std::sync::mpsc;
use std::thread::ThreadId;

use jazz_tools::query_manager::types::{ColumnType, SchemaBuilder, TableSchema};
use jazz_tools::runtime_core::{NoopScheduler, SyncSender};
use jazz_tools::storage::MemoryStorage;
use jazz_tools::sync_manager::{ClientId, Destination, InboxEntry, OutboxEntry, ServerId, Source};

use super::*;

const PATIENCE: Duration = Duration::from_secs(10);

fn schema() -> Schema {
    SchemaBuilder::new()
        .table(TableSchema::builder("notes").column("text", ColumnType::Text))
        .build()
}

fn note(text: &str) -> String {
    serde_json::json!({ "text": { "type": "Text", "value": text } }).to_string()
}

type Hop = Box<dyn FnOnce() + Send>;

/// One thing JS was told, in the order it was told.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Told {
    /// A subscription's delta: its label, the ids added, updated and removed, and the
    /// text of every row that came with it.
    Delta {
        subscription: &'static str,
        added: Vec<String>,
        updated: Vec<String>,
        removed: Vec<String>,
        rows: Vec<(String, String)>,
    },
    AuthFailure(String),
    /// A write the server refused: the batch.
    MutationError(String),
    /// The test's own mark: what JS itself was doing at that point.
    Mark(&'static str),
}

/// What the mocks share with the thread that plays JS.
struct Js {
    thread: ThreadId,
    hops: Mutex<mpsc::Sender<(Hop, mpsc::Sender<()>)>>,
    /// The adapter's `requestBatchedTick`: a microtask that calls `batchedTick()`.
    tick_requested: AtomicBool,
    tick_requests: AtomicUsize,
    /// Calls from other threads that are waiting for the JS thread to run them.
    waiting: AtomicUsize,
    told: Mutex<Vec<Told>>,
    /// What must never happen. Every test ends by asserting this is empty.
    violations: Mutex<Vec<String>>,
}

impl Js {
    fn on_js_thread(&self) -> bool {
        std::thread::current().id() == self.thread
    }

    fn violation(&self, what: String) {
        self.violations.lock().unwrap().push(what);
    }

    fn tell(&self, told: Told) {
        self.told.lock().unwrap().push(told);
    }

    /// What is true of every callback method, whichever it is: the caller does not hold
    /// the core lock, because off the JS thread the call waits for that thread.
    fn called(&self, what: &str) {
        if CORE_LOCKS_HELD.with(std::cell::Cell::get) > 0 {
            self.violation(format!("{what}: called with the core lock held"));
        }
    }

    /// A callback method, as the glue invokes it: inline on the JS thread, otherwise
    /// posted to it while the caller waits.
    fn invoke_blocking(&self, what: &'static str, call: impl FnOnce() + Send + 'static) {
        self.called(what);
        if self.on_js_thread() {
            call();
            return;
        }
        let (done, ran) = mpsc::channel();
        self.waiting.fetch_add(1, Ordering::SeqCst);
        let posted = self.hops.lock().unwrap().send((Box::new(call), done));
        if posted.is_ok() && ran.recv_timeout(PATIENCE).is_err() {
            self.violation(format!("{what}: the JS thread never ran it"));
        }
        self.waiting.fetch_sub(1, Ordering::SeqCst);
    }
}

struct TickRequests(Arc<Js>);

impl BatchedTickCallback for TickRequests {
    fn request_batched_tick(&self) {
        let js = Arc::clone(&self.0);
        self.0.invoke_blocking("request_batched_tick", move || {
            js.tick_requests.fetch_add(1, Ordering::SeqCst);
            js.tick_requested.store(true, Ordering::SeqCst);
        });
    }
}

struct AuthFailures(Arc<Js>);

impl AuthFailureCallback for AuthFailures {
    fn on_failure(&self, reason: String) {
        self.0.called("on_failure");
        if !self.0.on_js_thread() {
            self.0
                .violation("an auth failure was reported off the JS thread".to_string());
        }
        self.0.tell(Told::AuthFailure(reason));
    }
}

struct MutationErrors(Arc<Js>);

impl MutationErrorCallback for MutationErrors {
    fn on_error(&self, event_json: String) {
        let js = Arc::clone(&self.0);
        self.0.invoke_blocking("on_error", move || {
            let event: serde_json::Value = serde_json::from_str(&event_json).unwrap();
            js.tell(Told::MutationError(
                event["batch"]["batchId"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string(),
            ));
        });
    }
}

type Handler = Box<dyn FnMut(&Told) + Send>;

struct Subscriber {
    js: Arc<Js>,
    label: &'static str,
    /// Set once `unsubscribe` has returned.
    unsubscribed: Arc<AtomicBool>,
    /// What the app's own handler does with a delta.
    handler: Mutex<Option<Handler>>,
}

impl SubscriptionCallback for Subscriber {
    fn on_update(&self, delta_json: String, _blobs: Vec<Vec<u8>>) {
        self.js.called("on_update");
        if !self.js.on_js_thread() {
            self.js.violation(format!(
                "{}: a delta was delivered off the JS thread",
                self.label
            ));
        }
        if self.unsubscribed.load(Ordering::SeqCst) {
            self.js.violation(format!(
                "{}: a delta was delivered after unsubscribe",
                self.label
            ));
        }
        let changes: Vec<serde_json::Value> = serde_json::from_str(&delta_json).unwrap();
        let ids = |kind: u64| {
            changes
                .iter()
                .filter(|change| change["kind"] == kind)
                .map(|change| change["id"].as_str().unwrap().to_string())
                .collect::<Vec<_>>()
        };
        let rows = changes
            .iter()
            .filter(|change| change["row"].is_object())
            .map(|change| {
                (
                    change["id"].as_str().unwrap().to_string(),
                    change["row"]["values"][0]["value"]
                        .as_str()
                        .unwrap()
                        .to_string(),
                )
            })
            .collect();
        let told = Told::Delta {
            subscription: self.label,
            added: ids(0),
            updated: ids(2),
            removed: ids(1),
            rows,
        };
        self.js.tell(told.clone());
        // Taken out while it runs: what it calls may come back here.
        let handler = self.handler.lock().unwrap().take();
        if let Some(mut handler) = handler {
            handler(&told);
            *self.handler.lock().unwrap() = Some(handler);
        }
    }
}

/// Where a core's sync messages go, and which threads sent them: the engine only sends
/// from its batched tick, so these are the threads that ticked.
#[derive(Clone, Default)]
struct Outbox {
    entries: Arc<Mutex<Vec<OutboxEntry>>>,
    senders: Arc<Mutex<Vec<ThreadId>>>,
}

impl Outbox {
    fn take(&self) -> Vec<OutboxEntry> {
        std::mem::take(&mut self.entries.lock().unwrap())
    }

    fn senders(&self) -> Vec<ThreadId> {
        self.senders.lock().unwrap().clone()
    }
}

impl SyncSender for Outbox {
    fn send_sync_message(&self, message: OutboxEntry) {
        self.senders
            .lock()
            .unwrap()
            .push(std::thread::current().id());
        self.entries.lock().unwrap().push(message);
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

struct Subscription {
    handle: u64,
    unsubscribed: Arc<AtomicBool>,
}

/// A future of the runtime's, polled by hand the way the glue polls it: on the JS
/// thread, again whenever it was woken.
struct Asked<'a, T> {
    future: Pin<Box<dyn Future<Output = T> + 'a>>,
    wakes: Arc<Wakes>,
}

#[derive(Default)]
struct Wakes(AtomicUsize);

impl std::task::Wake for Wakes {
    fn wake(self: Arc<Self>) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

impl<'a, T> Asked<'a, T> {
    fn new(future: impl Future<Output = T> + 'a) -> Self {
        Self {
            future: Box::pin(future),
            wakes: Arc::new(Wakes::default()),
        }
    }

    fn poll(&mut self) -> Poll<T> {
        let waker = Waker::from(Arc::clone(&self.wakes));
        self.future.as_mut().poll(&mut Context::from_waker(&waker))
    }

    fn wakes(&self) -> usize {
        self.wakes.0.load(Ordering::SeqCst)
    }
}

/// A runtime, the thread that plays JS for it, and a server it syncs with.
struct Stand {
    js: Arc<Js>,
    posted: mpsc::Receiver<(Hop, mpsc::Sender<()>)>,
    runtime: Arc<RnRuntime>,
    client_out: Outbox,
    server: RuntimeCore<MemoryStorage, NoopScheduler>,
    server_out: Outbox,
    client_id: ClientId,
    server_id: ServerId,
    session_json: String,
    query_json: String,
    /// While set, what the server sends stays on the wire.
    withhold: bool,
    withheld: Vec<OutboxEntry>,
    /// Holds the store's directory for as long as the stand lives.
    _dir: tempfile::TempDir,
}

impl Stand {
    fn new(name: &str) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let (hops, posted) = mpsc::channel();
        let js = Arc::new(Js {
            thread: std::thread::current().id(),
            hops: Mutex::new(hops),
            tick_requested: AtomicBool::new(false),
            tick_requests: AtomicUsize::new(0),
            waiting: AtomicUsize::new(0),
            told: Mutex::new(Vec::new()),
            violations: Mutex::new(Vec::new()),
        });

        let runtime = RnRuntime::new(
            serde_json::to_string(&schema()).unwrap(),
            name.to_string(),
            "dev".to_string(),
            "main".to_string(),
            Some("local".to_string()),
            Some(
                dir.path()
                    .join("client.sqlite")
                    .to_string_lossy()
                    .into_owned(),
            ),
            None,
        )
        .unwrap();
        runtime
            .on_batched_tick_needed(Some(Box::new(TickRequests(Arc::clone(&js)))))
            .unwrap();

        let server_schema = SchemaManager::new(
            SyncManager::new().with_durability_tier(DurabilityTier::EdgeServer),
            schema(),
            AppId::from_string(name).unwrap_or_else(|_| AppId::from_name(name)),
            "dev",
            "main",
        )
        .unwrap();
        let mut server = RuntimeCore::new(server_schema, MemoryStorage::new(), NoopScheduler);
        let server_out = Outbox::default();
        server.set_sync_sender(Box::new(server_out.clone()));
        for _ in 0..4 {
            server.immediate_tick();
            server.batched_tick();
        }

        let client_id = ClientId::new();
        let server_id = ServerId::new();
        let session = Session::new("alice");
        server.add_client(client_id, Some(session.clone()));
        let client_out = Outbox::default();
        let query_json = {
            let mut core = runtime.core.lock().unwrap();
            core.set_sync_sender(Box::new(client_out.clone()));
            core.add_server(server_id);
            let query = core
                .schema_manager_mut()
                .query_manager_mut()
                .query("notes")
                .build();
            serde_json::to_string(&query).unwrap()
        };

        Self {
            js,
            posted,
            runtime,
            client_out,
            server,
            server_out,
            client_id,
            server_id,
            session_json: serde_json::to_string(&session).unwrap(),
            query_json,
            withhold: false,
            withheld: Vec::new(),
            _dir: dir,
        }
    }

    fn probe(&self) -> &TickProbe {
        &self.runtime.scheduler.probe
    }

    /// Subscribes to every note, as the client does: in two phases.
    fn subscribe(&self, label: &'static str) -> Subscription {
        self.subscribe_with(label, None)
    }

    fn subscribe_with(&self, label: &'static str, handler: Option<Handler>) -> Subscription {
        let handle = self
            .runtime
            .create_subscription(
                self.query_json.clone(),
                Some(self.session_json.clone()),
                Some("local".to_string()),
            )
            .unwrap();
        let unsubscribed = Arc::new(AtomicBool::new(false));
        self.runtime
            .execute_subscription(
                handle,
                Box::new(Subscriber {
                    js: Arc::clone(&self.js),
                    label,
                    unsubscribed: Arc::clone(&unsubscribed),
                    handler: Mutex::new(handler),
                }),
            )
            .unwrap();
        Subscription {
            handle,
            unsubscribed,
        }
    }

    fn unsubscribe(&self, subscription: &Subscription) {
        self.runtime.unsubscribe(subscription.handle).unwrap();
        subscription.unsubscribed.store(true, Ordering::SeqCst);
    }

    fn server_writes(&mut self, text: &str) -> String {
        let ((id, _), _) = self
            .server
            .insert(
                "notes",
                [("text".to_string(), Value::Text(text.to_string()))].into(),
                None,
            )
            .unwrap();
        id.uuid().to_string()
    }

    /// A write made by the app: the row's id and its batch.
    fn js_writes(&self, text: &str) -> (String, String) {
        written(
            &self
                .runtime
                .insert("notes".to_string(), note(text), None, None)
                .unwrap(),
        )
    }

    /// The microtask the adapter queues when it is asked for a tick.
    fn run_microtasks(&self) {
        if self.js.tick_requested.swap(false, Ordering::SeqCst) {
            let _ = self.runtime.batched_tick();
        }
    }

    /// One turn of the JS loop: what was posted to it, each followed by its microtasks.
    fn turn(&self) {
        self.run_microtasks();
        while let Ok((hop, done)) = self.posted.try_recv() {
            hop();
            let _ = done.send(());
            self.run_microtasks();
        }
    }

    /// What was posted to the JS loop is run, and the microtasks it queued are not —
    /// the instant between the two, which a call made on the JS thread never sees but
    /// a tick on the worker does.
    fn answer(&self) {
        while let Ok((hop, done)) = self.posted.try_recv() {
            hop();
            let _ = done.send(());
        }
    }

    /// The wire: what each side sent reaches the other. The client's inbox is filled the
    /// way the transport fills it, which schedules the client's tick. Returns whether
    /// anything moved.
    fn carry(&mut self) -> bool {
        let mut moved = false;
        for entry in self.client_out.take() {
            moved = true;
            if entry.destination == Destination::Server(self.server_id) {
                self.server.park_sync_message(InboxEntry {
                    source: Source::Client(self.client_id),
                    payload: entry.payload,
                });
            }
        }
        self.server.batched_tick();
        self.server.immediate_tick();
        for entry in self.server_out.take() {
            moved = true;
            if entry.destination == Destination::Client(self.client_id) {
                self.withheld.push(entry);
            }
        }
        if !self.withhold {
            self.hand_over();
        }
        moved
    }

    /// Everything on the wire reaches the client at once: one tick finds all of it.
    fn hand_over(&mut self) {
        if self.withheld.is_empty() {
            return;
        }
        let mut core = self.runtime.core.lock().unwrap();
        for entry in self.withheld.drain(..) {
            core.park_sync_message(InboxEntry {
                source: Source::Server(self.server_id),
                payload: entry.payload,
            });
        }
    }

    /// The server refuses a batch of the client's.
    fn server_rejects(&self, batch: &str) {
        let batch_id = parse_batch_id_input(batch).unwrap();
        self.runtime
            .core
            .lock()
            .unwrap()
            .park_sync_message(InboxEntry {
                source: Source::Server(self.server_id),
                payload: jazz_tools::sync_manager::SyncPayload::BatchFate {
                    fate: jazz_tools::batch_fate::BatchFate::Rejected {
                        batch_id,
                        code: "denied".to_string(),
                        reason: "the test said so".to_string(),
                    },
                },
            });
    }

    fn patience(&self, started: Instant, what: &str) {
        assert!(
            started.elapsed() < PATIENCE,
            "never happened: {what}; told {:?}; violations {:?}",
            self.told(),
            self.violations()
        );
        std::thread::sleep(Duration::from_millis(1));
    }

    /// Runs the JS loop and the wire until `done`.
    fn run_until(&mut self, what: &str, mut done: impl FnMut(&Self) -> bool) {
        let started = Instant::now();
        loop {
            self.turn();
            self.carry();
            if done(self) {
                return;
            }
            self.patience(started, what);
        }
    }

    /// Runs the wire alone until `done`: JS is busy with something else, and what is
    /// posted to it waits.
    fn wire_until(&mut self, what: &str, mut done: impl FnMut(&Self) -> bool) {
        let started = Instant::now();
        loop {
            self.carry();
            if done(self) {
                return;
            }
            self.patience(started, what);
        }
    }

    /// Waits for `done` with JS busy and the wire still.
    fn wait_until(&self, what: &str, mut done: impl FnMut(&Self) -> bool) {
        let started = Instant::now();
        while !done(self) {
            self.patience(started, what);
        }
    }

    /// The worker has something to do or is doing it.
    fn worker_busy(&self) -> bool {
        self.runtime.scheduler.scheduled.load(Ordering::SeqCst)
            || self.probe().working.load(Ordering::SeqCst)
            || self.js.waiting.load(Ordering::SeqCst) > 0
    }

    fn quiet(&mut self, what: &str, js: impl Fn(&Self)) {
        let started = Instant::now();
        let mut idle = 0;
        while idle < 4 {
            js(self);
            let moved = self.carry();
            idle = if moved || self.worker_busy() {
                0
            } else {
                idle + 1
            };
            self.patience(started, what);
        }
    }

    /// Until neither side has anything left to say and the client has nothing to do.
    fn run_until_quiet(&mut self) {
        loop {
            self.quiet("the stand goes quiet", Self::turn);
            if !self.js.tick_requested.load(Ordering::SeqCst) && !self.runtime.outbox.has_ready() {
                return;
            }
        }
    }

    /// The same with JS busy: what the worker asks of it waits, and so does the worker.
    fn wire_until_quiet(&mut self) {
        let started = Instant::now();
        let mut idle = 0;
        while idle < 4 {
            let moved = self.carry();
            let worker = self.runtime.scheduler.scheduled.load(Ordering::SeqCst)
                || (self.probe().working.load(Ordering::SeqCst)
                    && self.js.waiting.load(Ordering::SeqCst) == 0);
            idle = if moved || worker { 0 } else { idle + 1 };
            self.patience(started, "the wire goes quiet");
        }
    }

    /// The same with JS answering what is posted to it and running no microtask.
    fn answer_until_quiet(&mut self) {
        self.quiet("the worker goes quiet", Self::answer);
    }

    fn told(&self) -> Vec<Told> {
        self.js.told.lock().unwrap().clone()
    }

    fn violations(&self) -> Vec<String> {
        self.js.violations.lock().unwrap().clone()
    }

    /// The ids `subscription` has been told were added, in order.
    fn added(&self, subscription: &'static str) -> Vec<String> {
        added(&self.told(), subscription)
    }
}

fn added(told: &[Told], subscription: &'static str) -> Vec<String> {
    told.iter()
        .filter_map(|told| match told {
            Told::Delta {
                subscription: label,
                added,
                ..
            } if *label == subscription => Some(added.clone()),
            _ => None,
        })
        .flatten()
        .collect()
}

/// Lets a tick that is held before it releases the core go on once somebody is waiting
/// for the core behind it. Returns whether anybody was.
fn release_once_awaited(runtime: &Arc<RnRuntime>) -> std::thread::JoinHandle<bool> {
    let runtime = Arc::clone(runtime);
    std::thread::spawn(move || {
        let probe = &runtime.scheduler.probe;
        let started = Instant::now();
        while probe.at_the_lock.load(Ordering::SeqCst) == 0 && started.elapsed() < PATIENCE {
            std::thread::sleep(Duration::from_millis(1));
        }
        let awaited = probe.at_the_lock.load(Ordering::SeqCst) > 0;
        probe.held_before_release.store(false, Ordering::SeqCst);
        awaited
    })
}

/// The id and the batch of a row a write call returned.
fn written(returned: &str) -> (String, String) {
    let row: serde_json::Value = serde_json::from_str(returned).unwrap();
    (
        row["id"].as_str().unwrap().to_string(),
        row["batchId"].as_str().unwrap().to_string(),
    )
}

/// The symptom: rows a server delivers are stored by the engine's scheduled tick, and on
/// a phone that tick ran on the thread the app renders on — 2.4 s of it for one search.
/// The tick runs on a thread of the engine's own; the JS thread is only told the result.
#[test]
fn a_scheduled_tick_does_not_run_on_the_js_thread() {
    let mut stand = Stand::new("tick-thread");
    stand.subscribe("notes");
    let written: Vec<String> = (0..5)
        .map(|index| stand.server_writes(&format!("note {index}")))
        .collect();
    stand.run_until("the notes reach the subscription", |stand| {
        stand.added("notes").len() == written.len()
    });

    let mut delivered = stand.added("notes");
    delivered.sort();
    let mut expected = written.clone();
    expected.sort();
    assert_eq!(delivered, expected);
    assert_eq!(stand.violations(), Vec::<String>::new());

    let ticked = stand.client_out.senders();
    assert!(!ticked.is_empty(), "the client never ticked");
    assert!(
        ticked.iter().all(|thread| *thread != stand.js.thread),
        "the engine's scheduled tick ran on the JS thread ({} of {} sends)",
        ticked
            .iter()
            .filter(|thread| **thread == stand.js.thread)
            .count(),
        ticked.len()
    );
}

/// What a write changes, its subscriptions have been told by the time the write returns:
/// the app reads its own write back at once. What ticks recorded before it comes first.
#[test]
fn a_write_has_told_js_its_own_delta_when_it_returns() {
    let mut stand = Stand::new("own-delta");
    stand.subscribe("notes");
    stand.run_until_quiet();

    // A tick stores a row from the server while JS is busy: recorded, not told yet.
    let theirs = stand.server_writes("theirs");
    stand.wire_until("the tick has recorded the server's row", |stand| {
        stand.runtime.outbox.has_ready()
    });
    assert_eq!(stand.added("notes"), Vec::<String>::new());

    let (mine, _) = stand.js_writes("mine");
    assert_eq!(
        stand.added("notes"),
        vec![theirs, mine],
        "by the time the write returned"
    );
    assert_eq!(stand.violations(), Vec::<String>::new());
}

/// Nothing is told about a subscription once `unsubscribe` has returned — not what a
/// tick recorded before the call, and not what a tick was recording during it. And
/// nothing is told inside the call: it is made from places that are not ready for it.
#[test]
fn nothing_is_told_about_a_subscription_once_it_is_unsubscribed() {
    let mut stand = Stand::new("unsubscribed");
    let recorded = stand.subscribe("gone before it was told");
    let recording = stand.subscribe("gone while it was recorded");
    stand.subscribe("stays");
    stand.run_until_quiet();
    let before = stand.told();

    // Recorded by a tick that has ended.
    let first = stand.server_writes("first");
    stand.wire_until("the tick has recorded the row", |stand| {
        stand.runtime.outbox.has_ready()
    });
    stand.unsubscribe(&recorded);
    assert_eq!(stand.told(), before, "told inside unsubscribe");
    stand.run_until("the subscription that stays is told", |stand| {
        stand.added("stays") == vec![first.clone()]
    });

    // Recorded by a tick that still holds the core when the call is made.
    stand
        .probe()
        .held_before_release
        .store(true, Ordering::SeqCst);
    let second = stand.server_writes("second");
    stand.wire_until("the tick holds what it recorded", |stand| {
        stand.probe().holding.load(Ordering::SeqCst)
    });
    let told = stand.told();
    let released = release_once_awaited(&stand.runtime);
    // Waits for the tick; what the tick recorded for the handle goes with it.
    stand.unsubscribe(&recording);
    assert!(
        released.join().unwrap(),
        "the call did not wait for the tick"
    );
    assert_eq!(stand.told(), told, "told inside unsubscribe");
    stand.run_until("the subscription that stays is told again", |stand| {
        stand.added("stays") == vec![first.clone(), second.clone()]
    });
    stand.run_until_quiet();

    assert_eq!(stand.added("gone before it was told"), Vec::<String>::new());
    assert_eq!(
        stand.added("gone while it was recorded"),
        vec![first.clone()]
    );
    assert_eq!(stand.violations(), Vec::<String>::new());
}

/// What a tick records becomes JS's to take when the tick has released the core, all of
/// it together: a drain that runs in the middle of a tick finds none of it.
#[test]
fn js_is_told_nothing_of_a_tick_that_still_holds_the_core() {
    let mut stand = Stand::new("whole-ticks");
    stand.subscribe("one");
    stand.subscribe("other");
    stand.run_until_quiet();
    let before = stand.told();

    stand
        .probe()
        .held_before_release
        .store(true, Ordering::SeqCst);
    let row = stand.server_writes("row");
    stand.wire_until("the tick holds what it recorded", |stand| {
        stand.probe().holding.load(Ordering::SeqCst)
    });
    stand.runtime.batched_tick().unwrap();
    stand.turn();
    assert_eq!(stand.told(), before, "told of a tick that had not ended");

    stand
        .probe()
        .held_before_release
        .store(false, Ordering::SeqCst);
    stand.wait_until("the tick ends", |stand| {
        !stand.probe().holding.load(Ordering::SeqCst) && stand.runtime.outbox.has_ready()
    });
    // One drain tells all of it.
    stand.runtime.batched_tick().unwrap();
    assert_eq!(stand.added("one"), vec![row.clone()]);
    assert_eq!(stand.added("other"), vec![row]);
    assert_eq!(stand.violations(), Vec::<String>::new());
}

/// A closed runtime does not tick and tells JS nothing: not the tick that was in flight
/// when it was closed, not one that was waiting for the core behind the close.
#[test]
fn a_closed_runtime_does_not_tick_and_tells_js_nothing() {
    // A tick in flight: close waits for it, and what it recorded is not told.
    let mut stand = Stand::new("closed-in-flight");
    stand.subscribe("notes");
    stand.run_until_quiet();
    stand
        .probe()
        .held_before_release
        .store(true, Ordering::SeqCst);
    stand.server_writes("row");
    stand.wire_until("the tick holds what it recorded", |stand| {
        stand.probe().holding.load(Ordering::SeqCst)
    });
    let released = release_once_awaited(&stand.runtime);
    stand.runtime.close().unwrap();
    assert!(
        released.join().unwrap(),
        "the call did not wait for the tick"
    );
    let told = stand.told();
    let ticked = stand.probe().engine_ticks.load(Ordering::SeqCst);
    for _ in 0..30 {
        stand.turn();
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(stand.told(), told, "told after close");
    assert_eq!(stand.probe().engine_ticks.load(Ordering::SeqCst), ticked);
    assert_eq!(stand.violations(), Vec::<String>::new());

    // A tick waiting for the core when the runtime is closed: whichever of the two gets
    // the core first, the engine does not tick again.
    let mut stand = Stand::new("closed-behind");
    stand.subscribe("notes");
    stand.run_until_quiet();
    let ticked = stand.probe().engine_ticks.load(Ordering::SeqCst);
    let held = stand.runtime.core.lock().unwrap();
    stand.runtime.scheduler.schedule_batched_tick();
    stand.wait_until("the tick waits for the core", |stand| {
        stand.probe().waiting_for_core.load(Ordering::SeqCst)
    });
    let runtime = Arc::clone(&stand.runtime);
    let closing = std::thread::spawn(move || runtime.close());
    stand.wait_until("the close has shut the scheduler down", |stand| {
        stand.runtime.scheduler.shutdown.load(Ordering::SeqCst)
    });
    std::thread::sleep(Duration::from_millis(5));
    drop(held);
    closing.join().unwrap().unwrap();
    for _ in 0..30 {
        stand.turn();
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(
        stand.probe().engine_ticks.load(Ordering::SeqCst),
        ticked,
        "the engine ticked on a runtime that was being closed"
    );
    assert_eq!(stand.violations(), Vec::<String>::new());
}

/// A query resolves after JS has been told what the tick that answered it recorded. The
/// tick resolves it in its middle, under the core lock, and the app would otherwise read
/// the answer ahead of the deltas that came with it.
#[test]
fn a_query_resolves_after_js_was_told_what_its_tick_recorded() {
    let mut stand = Stand::new("query-told-first");
    stand.subscribe("notes");
    stand.server_writes("before");
    stand.run_until_quiet();

    // The server's new row and its answer to the query reach the client in one tick.
    stand.withhold = true;
    let row = stand.server_writes("with the answer");
    let runtime = Arc::clone(&stand.runtime);
    let mut asked = Asked::new(runtime.query(
        stand.query_json.clone(),
        Some(stand.session_json.clone()),
        Some("edge".to_string()),
        None,
    ));
    assert!(asked.poll().is_pending());
    stand.wire_until_quiet();
    let told = stand.told();
    assert_eq!(asked.wakes(), 0);

    stand
        .probe()
        .held_before_release
        .store(true, Ordering::SeqCst);
    stand.hand_over();
    stand.wait_until("the tick holds what it recorded", |stand| {
        stand.probe().holding.load(Ordering::SeqCst)
    });
    assert!(asked.wakes() > 0, "the tick has not answered the query");
    assert!(
        asked.poll().is_pending(),
        "the query resolved ahead of what its tick recorded for JS"
    );
    assert_eq!(stand.told(), told);

    let wakes = asked.wakes();
    stand
        .probe()
        .held_before_release
        .store(false, Ordering::SeqCst);
    stand.wait_until("the poll is woken by the end of the tick", |_| {
        asked.wakes() > wakes
    });
    let Poll::Ready(answer) = asked.poll() else {
        panic!("the query did not resolve once its tick had ended");
    };
    assert!(answer.unwrap().contains(&row));
    assert!(
        stand.added("notes").contains(&row),
        "the query resolved before JS was told what its tick recorded"
    );
    assert_eq!(stand.violations(), Vec::<String>::new());
}

/// The same for a wait on a batch: the tick that learns its fate resolves the wait.
#[test]
fn a_batch_wait_resolves_after_js_was_told_what_its_tick_recorded() {
    let mut stand = Stand::new("batch-told-first");
    stand.subscribe("notes");
    stand.run_until_quiet();

    stand.withhold = true;
    let row = stand.server_writes("with the verdict");
    let (_, batch) = stand.js_writes("mine");
    let runtime = Arc::clone(&stand.runtime);
    let mut waited = Asked::new(runtime.wait_for_batch(batch, "edge".to_string()));
    assert!(waited.poll().is_pending());
    stand.wire_until_quiet();
    let told = stand.told();
    assert_eq!(waited.wakes(), 0);

    stand
        .probe()
        .held_before_release
        .store(true, Ordering::SeqCst);
    stand.hand_over();
    stand.wait_until("the tick holds what it recorded", |stand| {
        stand.probe().holding.load(Ordering::SeqCst)
    });
    assert!(waited.wakes() > 0, "the tick has not settled the batch");
    assert!(
        waited.poll().is_pending(),
        "the wait resolved ahead of what its tick recorded for JS"
    );
    assert_eq!(
        stand.told(),
        told,
        "told inside a poll that resolved nothing"
    );

    let wakes = waited.wakes();
    stand
        .probe()
        .held_before_release
        .store(false, Ordering::SeqCst);
    stand.wait_until("the poll is woken by the end of the tick", |_| {
        waited.wakes() > wakes
    });
    let Poll::Ready(settled) = waited.poll() else {
        panic!("the wait did not resolve once its tick had ended");
    };
    settled.unwrap();
    assert!(
        stand.added("notes").contains(&row),
        "the wait resolved before JS was told what its tick recorded"
    );
    assert_eq!(stand.violations(), Vec::<String>::new());
}

/// A handler that calls back into the runtime while it is being told: it is not told the
/// next thing inside its own call, and nothing deadlocks.
#[test]
fn a_handler_that_calls_back_into_the_runtime_is_not_told_the_next_thing_inside_itself() {
    // It writes.
    let mut stand = Stand::new("handler-writes");
    stand.subscribe("quiet");
    stand.run_until_quiet();
    let runtime = Arc::clone(&stand.runtime);
    let js = Arc::clone(&stand.js);
    let wrote = Arc::new(Mutex::new(None));
    let writes = Arc::clone(&wrote);
    stand.subscribe_with(
        "writes",
        Some(Box::new(move |told| {
            let Told::Delta { added, .. } = told else {
                return;
            };
            if added.is_empty() || writes.lock().unwrap().is_some() {
                return;
            }
            js.tell(Told::Mark("the handler writes"));
            let (id, _) = written(
                &runtime
                    .insert("notes".to_string(), note("from the handler"), None, None)
                    .unwrap(),
            );
            *writes.lock().unwrap() = Some(id);
            js.tell(Told::Mark("the handler returns"));
        })),
    );
    stand.run_until_quiet();
    let row = stand.server_writes("row");
    stand.run_until("the handler's own write is told", |stand| {
        wrote
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(|id| stand.added("writes").contains(id))
    });
    stand.run_until_quiet();
    let told = stand.told();
    let from = told
        .iter()
        .position(|told| *told == Told::Mark("the handler writes"))
        .unwrap();
    let to = told
        .iter()
        .position(|told| *told == Told::Mark("the handler returns"))
        .unwrap();
    assert_eq!(
        to,
        from + 1,
        "told inside a handler: {:?}",
        &told[from..=to]
    );
    let own = wrote.lock().unwrap().clone().unwrap();
    assert_eq!(stand.added("writes"), vec![row.clone(), own.clone()]);
    assert_eq!(stand.added("quiet"), vec![row, own]);
    assert_eq!(stand.violations(), Vec::<String>::new());

    // It unsubscribes itself and another, both of which have more to be told.
    let mut stand = Stand::new("handler-unsubscribes");
    let other = stand.subscribe("other");
    let runtime = Arc::clone(&stand.runtime);
    let own_handle = Arc::new(Mutex::new(None::<Subscription>));
    let handle = Arc::clone(&own_handle);
    let other_handle = other.handle;
    let other_gone = Arc::clone(&other.unsubscribed);
    let itself = stand.subscribe_with(
        "itself",
        Some(Box::new(move |told| {
            let Told::Delta { added, .. } = told else {
                return;
            };
            if added.is_empty() {
                return;
            }
            if let Some(own) = handle.lock().unwrap().as_ref() {
                runtime.unsubscribe(own.handle).unwrap();
                own.unsubscribed.store(true, Ordering::SeqCst);
            }
            runtime.unsubscribe(other_handle).unwrap();
            other_gone.store(true, Ordering::SeqCst);
        })),
    );
    *own_handle.lock().unwrap() = Some(itself);
    stand.subscribe("stays");
    stand.run_until_quiet();
    // Two ticks' worth, both recorded before JS is told the first.
    let first = stand.server_writes("first");
    stand.wire_until("the first is recorded", |stand| {
        stand.runtime.outbox.has_ready()
    });
    let second = stand.server_writes("second");
    stand.answer_until_quiet();
    stand.run_until("the subscription that stays is told both", |stand| {
        stand.added("stays") == vec![first.clone(), second.clone()]
    });
    stand.run_until_quiet();
    assert_eq!(stand.added("itself"), vec![first.clone()]);
    assert!(!stand.added("other").contains(&second));
    assert_eq!(stand.violations(), Vec::<String>::new());

    // It closes the runtime, with more waiting to be told: the rest of the tick it was
    // told of, in whatever order the tick recorded it, and all of the next.
    let mut stand = Stand::new("handler-closes");
    stand.subscribe("one");
    let runtime = Arc::clone(&stand.runtime);
    stand.subscribe_with(
        "closes",
        Some(Box::new(move |told| {
            if matches!(told, Told::Delta { added, .. } if !added.is_empty()) {
                runtime.close().unwrap();
            }
        })),
    );
    stand.subscribe("other");
    stand.run_until_quiet();
    let first = stand.server_writes("first");
    stand.wire_until("the first is recorded", |stand| {
        stand.runtime.outbox.has_ready()
    });
    stand.answer_until_quiet();
    let second = stand.server_writes("second");
    stand.answer_until_quiet();
    for _ in 0..30 {
        stand.turn();
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(stand.added("closes"), vec![first]);
    let told = stand.told();
    assert!(
        matches!(
            told.last(),
            Some(Told::Delta {
                subscription: "closes",
                ..
            })
        ),
        "told after close: {told:?}"
    );
    assert!(!stand.added("one").contains(&second));
    assert!(!stand.added("other").contains(&second));
    assert_eq!(stand.violations(), Vec::<String>::new());
}

/// A tick that panics leaves the core poisoned, as it did on the JS thread: every later
/// call says so. What it had recorded for JS is given up, and a poll that was waiting
/// for the tick to end is woken, not left waiting.
#[test]
fn a_tick_that_panics_poisons_the_core_and_tells_js_nothing_of_itself() {
    let mut stand = Stand::new("tick-panics");
    stand.subscribe("notes");
    stand.server_writes("before");
    stand.run_until_quiet();
    let told = stand.told();

    stand.withhold = true;
    stand.server_writes("with the answer");
    let runtime = Arc::clone(&stand.runtime);
    let mut asked = Asked::new(runtime.query(
        stand.query_json.clone(),
        Some(stand.session_json.clone()),
        Some("edge".to_string()),
        None,
    ));
    assert!(asked.poll().is_pending());
    stand.wire_until_quiet();

    stand
        .probe()
        .held_before_release
        .store(true, Ordering::SeqCst);
    stand
        .probe()
        .panic_before_release
        .store(true, Ordering::SeqCst);
    stand.hand_over();
    stand.wait_until("the tick holds what it recorded", |stand| {
        stand.probe().holding.load(Ordering::SeqCst)
    });
    assert!(asked.poll().is_pending());
    let wakes = asked.wakes();
    stand
        .probe()
        .held_before_release
        .store(false, Ordering::SeqCst);
    stand.wait_until("the poll is woken by the tick's end", |_| {
        asked.wakes() > wakes
    });
    assert!(asked.poll().is_ready(), "left waiting for a tick that died");

    for _ in 0..30 {
        stand.turn();
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(stand.told(), told, "told of a tick that panicked");
    assert!(matches!(
        stand.runtime.get_schema_hash(),
        Err(JazzRnError::Internal { .. })
    ));
    assert!(matches!(
        stand
            .runtime
            .insert("notes".to_string(), note("after"), None, None),
        Err(JazzRnError::Internal { .. })
    ));
    // Closing it says so as well, and still leaves a runtime that tells JS nothing.
    assert!(matches!(
        stand.runtime.close(),
        Err(JazzRnError::Internal { .. })
    ));
    stand.turn();
    assert_eq!(stand.told(), told);
    assert_eq!(stand.violations(), Vec::<String>::new());
}

/// A tick that panicked leaves the core poisoned, and `unsubscribe` says so. The app has
/// let go of the handle all the same: what an earlier tick recorded for it is not told.
#[test]
fn nothing_is_told_about_a_subscription_unsubscribed_on_a_poisoned_core() {
    let mut stand = Stand::new("unsubscribed-poisoned");
    let notes = stand.subscribe("notes");
    stand.run_until_quiet();

    // Recorded by a tick that has ended; JS is asked and does not come yet.
    stand.server_writes("first");
    stand.wire_until("the tick has recorded the row", |stand| {
        stand.runtime.outbox.has_ready()
    });

    // The next tick that has something for JS panics. JS answers what the worker asks
    // of it meanwhile, and gets to nothing else.
    stand.withhold = true;
    stand.server_writes("second");
    stand.wire_until("the row is on the wire", |stand| !stand.withheld.is_empty());
    stand
        .probe()
        .panic_before_release
        .store(true, Ordering::SeqCst);
    stand.hand_over();
    let started = Instant::now();
    while stand.probe().panic_before_release.load(Ordering::SeqCst)
        || stand.runtime.get_schema_hash().is_ok()
    {
        stand.answer();
        stand.patience(started, "the tick panics and leaves the core poisoned");
    }

    let told = stand.told();
    assert!(matches!(
        stand.runtime.unsubscribe(notes.handle),
        Err(JazzRnError::Internal { .. })
    ));
    notes.unsubscribed.store(true, Ordering::SeqCst);
    stand.run_microtasks();
    let _ = stand.runtime.batched_tick();
    assert_eq!(stand.told(), told, "told after unsubscribe");
    assert_eq!(stand.violations(), Vec::<String>::new());
}

/// The core lock is not fair: a worker that took it again the moment it had released it
/// would keep a call from JS waiting for as long as there are ticks to run, and under a
/// busy connection there always are. A call from JS gets the core when the tick it found
/// running ends.
#[test]
fn a_call_from_js_waits_for_one_tick_at_most() {
    let stand = Stand::new("fair");
    stand.probe().tick_takes_millis.store(5, Ordering::SeqCst);
    // Ticks without end, asked for the way the transport asks.
    let stop = Arc::new(AtomicBool::new(false));
    let flood = {
        let stop = Arc::clone(&stop);
        let scheduler = stand.runtime.scheduler.clone();
        std::thread::spawn(move || {
            while !stop.load(Ordering::SeqCst) {
                scheduler.schedule_batched_tick();
                std::thread::sleep(Duration::from_micros(200));
            }
        })
    };
    stand.wait_until("the worker ticks", |stand| {
        stand.probe().engine_ticks.load(Ordering::SeqCst) >= 3
    });

    let mut most = 0;
    for call in 0..40u64 {
        let before = stand.probe().engine_ticks.load(Ordering::SeqCst);
        stand.runtime.get_schema_hash().unwrap();
        let began = stand.probe().engine_ticks.load(Ordering::SeqCst) - before;
        most = most.max(began);
        // Not in step with the ticks.
        std::thread::sleep(Duration::from_micros(700 + 370 * (call % 7)));
    }
    stop.store(true, Ordering::SeqCst);
    flood.join().unwrap();
    stand.probe().tick_takes_millis.store(0, Ordering::SeqCst);
    assert!(
        most <= 1,
        "a call from JS waited while {most} ticks began: the worker kept the core to itself"
    );
    assert_eq!(stand.violations(), Vec::<String>::new());
}

/// Calls that did not tick when the tick ran on the JS thread still tell JS nothing —
/// they are made from places that are not ready to be told, a render among them — and
/// what waits to be told is asked for by somebody, whatever recorded it.
#[test]
fn calls_that_do_not_tick_tell_js_nothing_and_what_waits_is_asked_for() {
    let mut stand = Stand::new("placement");
    stand.subscribe("notes");
    stand.run_until_quiet();

    let first = stand.server_writes("first");
    stand.wire_until("the tick has recorded the row", |stand| {
        stand.runtime.outbox.has_ready()
    });
    let told = stand.told();
    let created = stand
        .runtime
        .create_subscription(
            stand.query_json.clone(),
            Some(stand.session_json.clone()),
            Some("local".to_string()),
        )
        .unwrap();
    assert_eq!(stand.told(), told, "told inside create_subscription");
    stand.runtime.get_schema_hash().unwrap();
    assert_eq!(stand.told(), told, "told inside get_schema_hash");
    stand.runtime.unsubscribe(created).unwrap();
    assert_eq!(stand.told(), told, "told inside unsubscribe");
    let batch = stand.runtime.begin_batch("direct".to_string());
    assert_eq!(stand.told(), told, "told inside begin_batch: {batch:?}");
    stand.run_until("the row is told", |stand| {
        stand.added("notes") == vec![first.clone()]
    });
    stand.run_until_quiet();

    // Recorded while there was nobody to ask: asked for when there is.
    stand.runtime.on_batched_tick_needed(None).unwrap();
    let second = stand.server_writes("second");
    stand.wire_until("the tick has recorded the row", |stand| {
        stand.runtime.outbox.has_ready()
    });
    let asked = stand.js.tick_requests.load(Ordering::SeqCst);
    for _ in 0..20 {
        stand.turn();
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(stand.js.tick_requests.load(Ordering::SeqCst), asked);
    assert_eq!(stand.added("notes"), vec![first.clone()]);
    let told = stand.told();
    stand
        .runtime
        .on_batched_tick_needed(Some(Box::new(TickRequests(Arc::clone(&stand.js)))))
        .unwrap();
    assert_eq!(stand.told(), told, "told inside on_batched_tick_needed");
    stand.run_until("the row is told once JS can be asked", |stand| {
        stand.added("notes") == vec![first.clone(), second.clone()]
    });
    stand.run_until_quiet();

    // Recorded under a hold of the core that does not deliver, with no tick to follow
    // it: the hold itself has JS asked.
    stand
        .runtime
        .scheduler
        .scheduled
        .store(true, Ordering::SeqCst);
    let third = {
        let mut core = stand.runtime.core.lock().unwrap();
        let ((id, _), _) = core
            .insert(
                "notes",
                [("text".to_string(), Value::Text("third".to_string()))].into(),
                None,
            )
            .unwrap();
        id.uuid().to_string()
    };
    assert_eq!(stand.added("notes"), vec![first.clone(), second.clone()]);
    let started = Instant::now();
    while stand.added("notes").len() < 3 {
        stand.turn();
        stand.patience(started, "the row a plain hold recorded is told");
    }
    assert_eq!(stand.added("notes"), vec![first, second, third]);
    stand
        .runtime
        .scheduler
        .scheduled
        .store(false, Ordering::SeqCst);
    assert_eq!(stand.violations(), Vec::<String>::new());
}

/// A rejected handshake is reported from inside the tick that reads the transport, under
/// the core lock. JS is told on its own thread, after the tick.
#[test]
fn an_auth_failure_is_told_on_the_js_thread_after_the_tick() {
    use jazz_tools::transport_manager::{AuthConfig, TransportInbound};

    let stand = Stand::new("auth-failure");
    stand
        .runtime
        .on_auth_failure(Box::new(AuthFailures(Arc::clone(&stand.js))))
        .unwrap();
    let (mut handle, _never_run) = jazz_tools::transport_manager::create::<
        jazz_tools::ws_stream::NativeWsStream,
        RnTickNotifier,
    >(
        "ws://127.0.0.1:9".to_string(),
        AuthConfig::default(),
        RnTickNotifier {
            scheduler: stand.runtime.scheduler.clone(),
        },
    );
    let (wire, inbound) = futures::channel::mpsc::unbounded();
    handle.inbound_rx = inbound;
    stand.runtime.core.lock().unwrap().set_transport(handle);

    wire.unbounded_send(TransportInbound::AuthFailure {
        reason: "expired".to_string(),
    })
    .unwrap();
    stand.runtime.scheduler.schedule_batched_tick();
    let started = Instant::now();
    while !stand
        .told()
        .contains(&Told::AuthFailure("expired".to_string()))
    {
        stand.turn();
        stand.patience(started, "the auth failure is told");
    }
    assert_eq!(stand.violations(), Vec::<String>::new());
}

/// A runtime JS lets go of without closing it, with a tick in flight: the tick ends
/// first, and the engine goes on the thread that let go of it. Not on the worker at the
/// end of its tick: the thread that frees a runtime nobody closed is tearing down the JS
/// that runtime was told, and the engine's own teardown must not run beside that.
#[test]
fn a_runtime_dropped_in_the_middle_of_a_tick_goes_on_the_thread_that_dropped_it() {
    let mut stand = Stand::new("dropped");
    stand.subscribe("notes");
    stand.run_until_quiet();
    stand
        .probe()
        .held_before_release
        .store(true, Ordering::SeqCst);
    stand.server_writes("row");
    stand.wire_until("the tick holds what it recorded", |stand| {
        stand.probe().holding.load(Ordering::SeqCst)
    });

    let Stand {
        runtime,
        js,
        posted,
        ..
    } = stand;
    let core = Arc::downgrade(&runtime.core);
    // Not the scheduler itself: a handle on it keeps the worker's queue open.
    let probe = Arc::clone(&runtime.scheduler.probe);
    let dropping = std::thread::spawn(move || {
        drop(runtime);
        std::thread::current().id()
    });
    std::thread::sleep(Duration::from_millis(50));
    assert!(
        !dropping.is_finished(),
        "the runtime was let go of while a tick held its engine"
    );
    assert!(core.strong_count() > 0, "the tick lost its engine");
    probe.held_before_release.store(false, Ordering::SeqCst);
    let dropped_on = dropping.join().unwrap();
    assert_eq!(core.strong_count(), 0, "the engine was left behind");
    assert_eq!(
        *probe.engine_dropped_on.lock().unwrap(),
        Some(dropped_on),
        "the engine went on another thread than the one that let go of the runtime"
    );
    let started = Instant::now();
    while !probe.worker_exited.load(Ordering::SeqCst) {
        while let Ok((hop, done)) = posted.try_recv() {
            hop();
            let _ = done.send(());
        }
        assert!(started.elapsed() < PATIENCE, "the worker was left behind");
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(js.violations.lock().unwrap().clone(), Vec::<String>::new());
}

/// A refusal the worker is taking to JS when the runtime is let go of. The call waits
/// for the JS thread, for as long as that takes; the worker must not be holding the
/// engine meanwhile, or it is the one left with it when JS frees the runtime — and
/// tears it down at the end of a call into a JS that is going away.
#[test]
fn a_runtime_dropped_while_a_refusal_is_on_its_way_to_js_goes_on_the_thread_that_dropped_it() {
    /// Reports nothing: the call stays where the glue would leave it, waiting for JS.
    struct Waiting {
        entered: Arc<AtomicBool>,
        released: Arc<AtomicBool>,
    }
    impl MutationErrorCallback for Waiting {
        fn on_error(&self, _event_json: String) {
            self.entered.store(true, Ordering::SeqCst);
            let started = Instant::now();
            while !self.released.load(Ordering::SeqCst) && started.elapsed() < PATIENCE {
                std::thread::sleep(Duration::from_millis(1));
            }
        }
    }

    let mut stand = Stand::new("dropped-reporting");
    let entered = Arc::new(AtomicBool::new(false));
    let released = Arc::new(AtomicBool::new(false));
    stand
        .runtime
        .on_mutation_error(Box::new(Waiting {
            entered: Arc::clone(&entered),
            released: Arc::clone(&released),
        }))
        .unwrap();
    stand.subscribe("notes");
    stand.run_until_quiet();
    stand.withhold = true;
    let (_, batch) = stand.js_writes("mine");
    stand.server_rejects(&batch);
    stand.run_until("the refusal is on its way to JS", |_| {
        entered.load(Ordering::SeqCst)
    });

    let Stand {
        runtime,
        js,
        posted,
        ..
    } = stand;
    let core = Arc::downgrade(&runtime.core);
    let probe = Arc::clone(&runtime.scheduler.probe);
    let dropped_on = std::thread::spawn(move || {
        drop(runtime);
        std::thread::current().id()
    })
    .join()
    .unwrap();
    assert_eq!(
        core.strong_count(),
        0,
        "the worker held the engine while it waited for JS"
    );
    assert_eq!(
        *probe.engine_dropped_on.lock().unwrap(),
        Some(dropped_on),
        "the engine went on another thread than the one that let go of the runtime"
    );
    released.store(true, Ordering::SeqCst);
    let started = Instant::now();
    while !probe.worker_exited.load(Ordering::SeqCst) {
        while let Ok((hop, done)) = posted.try_recv() {
            hop();
            let _ = done.send(());
        }
        assert!(started.elapsed() < PATIENCE, "the worker was left behind");
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(js.violations.lock().unwrap().clone(), Vec::<String>::new());
}

/// The same refusal, the runtime let go of a moment earlier: the worker has reached the
/// engine to take what it has to report and has not given it back. That is a hold like
/// a tick's, and the thread that lets go of the runtime waits for it. What the worker
/// took under the hold it still reports, after the runtime is gone.
#[test]
fn a_runtime_dropped_while_the_worker_takes_what_it_has_to_report_waits_for_it() {
    struct Told(Arc<AtomicBool>);
    impl MutationErrorCallback for Told {
        fn on_error(&self, _event_json: String) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    let mut stand = Stand::new("dropped-taking-a-report");
    let told = Arc::new(AtomicBool::new(false));
    stand
        .runtime
        .on_mutation_error(Box::new(Told(Arc::clone(&told))))
        .unwrap();
    stand.subscribe("notes");
    stand.run_until_quiet();
    stand.probe().held_at_report.store(true, Ordering::SeqCst);
    stand.withhold = true;
    let (_, batch) = stand.js_writes("mine");
    stand.server_rejects(&batch);
    stand.run_until(
        "the worker has reached the engine for its report",
        |stand| stand.probe().holding_at_report.load(Ordering::SeqCst),
    );

    let Stand {
        runtime,
        js,
        posted,
        ..
    } = stand;
    let core = Arc::downgrade(&runtime.core);
    let probe = Arc::clone(&runtime.scheduler.probe);
    let dropping = std::thread::spawn(move || {
        drop(runtime);
        std::thread::current().id()
    });
    std::thread::sleep(Duration::from_millis(50));
    assert!(
        !dropping.is_finished(),
        "the runtime was let go of while the worker held its engine for a report"
    );
    probe.held_at_report.store(false, Ordering::SeqCst);
    let dropped_on = dropping.join().unwrap();
    assert_eq!(core.strong_count(), 0, "the engine was left behind");
    assert_eq!(
        *probe.engine_dropped_on.lock().unwrap(),
        Some(dropped_on),
        "the engine went on another thread than the one that let go of the runtime"
    );
    let started = Instant::now();
    while !probe.worker_exited.load(Ordering::SeqCst) {
        while let Ok((hop, done)) = posted.try_recv() {
            hop();
            let _ = done.send(());
        }
        assert!(started.elapsed() < PATIENCE, "the worker was left behind");
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(
        told.load(Ordering::SeqCst),
        "the refusal the worker had taken from the engine was not reported"
    );
    assert_eq!(js.violations.lock().unwrap().clone(), Vec::<String>::new());
}

/// A poll of a query that resolves nothing still tells JS what is deliverable: the
/// query's own first tick recorded it, and nobody else has been asked to come for it.
#[test]
fn a_query_poll_that_resolves_nothing_tells_js_what_is_deliverable() {
    let mut stand = Stand::new("pending-polls-tell");
    stand.subscribe("notes");
    stand.run_until_quiet();

    // Still waiting for the server.
    let earlier = stand.server_writes("earlier");
    stand.wire_until("the tick has recorded the row", |stand| {
        stand.runtime.outbox.has_ready()
    });
    stand.withhold = true;
    let row = stand.server_writes("with the answer");
    let runtime = Arc::clone(&stand.runtime);
    let mut asked = Asked::new(runtime.query(
        stand.query_json.clone(),
        Some(stand.session_json.clone()),
        Some("edge".to_string()),
        None,
    ));
    assert!(asked.poll().is_pending());
    assert_eq!(
        stand.added("notes"),
        vec![earlier.clone()],
        "a poll that found no answer told JS nothing"
    );
    stand.answer_until_quiet();

    // Answered by a tick that still holds the core. Something else is deliverable
    // by then, and JS has not come for it.
    let local = {
        let mut core = stand.runtime.core.lock().unwrap();
        let ((id, _), _) = core
            .insert(
                "notes",
                [("text".to_string(), Value::Text("local".to_string()))].into(),
                None,
            )
            .unwrap();
        id.uuid().to_string()
    };
    stand.answer_until_quiet();
    assert!(stand.runtime.outbox.has_ready());
    assert_eq!(stand.added("notes"), vec![earlier.clone()]);

    stand
        .probe()
        .held_before_release
        .store(true, Ordering::SeqCst);
    stand.hand_over();
    stand.wait_until("the tick holds what it recorded", |stand| {
        stand.probe().holding.load(Ordering::SeqCst)
    });
    assert!(asked.wakes() > 0, "the tick has not answered the query");
    assert!(asked.poll().is_pending());
    assert_eq!(
        stand.added("notes"),
        vec![earlier.clone(), local.clone()],
        "a poll that waits for the tick to end told JS nothing"
    );

    let wakes = asked.wakes();
    stand
        .probe()
        .held_before_release
        .store(false, Ordering::SeqCst);
    stand.wait_until("the poll is woken by the end of the tick", |_| {
        asked.wakes() > wakes
    });
    assert!(asked.poll().is_ready());
    assert_eq!(stand.added("notes"), vec![earlier, local, row]);
    assert_eq!(stand.violations(), Vec::<String>::new());
}

/// A wait for a batch that has settled already resolves at once — after JS has been
/// told what the ticks before it recorded, the tick that settled it among them. This
/// is the one call that tells JS something where it told nothing before the tick left
/// the JS thread: there the tick had told it by the time the call could be made.
#[test]
fn a_wait_for_a_batch_that_has_settled_tells_js_what_was_recorded_first() {
    let mut stand = Stand::new("settled-batch");
    stand.subscribe("notes");
    let (mine, batch) = stand.js_writes("mine");
    stand.run_until_quiet();

    let later = stand.server_writes("later");
    stand.wire_until("the tick has recorded the row", |stand| {
        stand.runtime.outbox.has_ready()
    });
    assert_eq!(stand.added("notes"), vec![mine.clone()]);
    let runtime = Arc::clone(&stand.runtime);
    let mut waited = Asked::new(runtime.wait_for_batch(batch, "edge".to_string()));
    let Poll::Ready(settled) = waited.poll() else {
        panic!("a wait for a settled batch did not resolve at once");
    };
    settled.unwrap();
    assert_eq!(
        stand.added("notes"),
        vec![mine, later],
        "the wait resolved ahead of what had been recorded for JS"
    );
    stand.run_until_quiet();
    assert_eq!(stand.violations(), Vec::<String>::new());
}

/// A handler that asks the runtime for more: a query, a subscription of its own.
#[test]
fn a_handler_that_queries_and_subscribes_is_not_told_inside_itself() {
    let mut stand = Stand::new("handler-asks");
    let runtime = Arc::clone(&stand.runtime);
    let js = Arc::clone(&stand.js);
    let query_json = stand.query_json.clone();
    let session_json = stand.session_json.clone();
    let answers = Arc::new(Mutex::new(Vec::new()));
    let answered = Arc::clone(&answers);
    let asked_once = AtomicBool::new(false);
    stand.subscribe_with(
        "asks",
        Some(Box::new(move |told| {
            let Told::Delta { added, .. } = told else {
                return;
            };
            if added.is_empty() || asked_once.swap(true, Ordering::SeqCst) {
                return;
            }
            js.tell(Told::Mark("the handler asks"));
            let mut asked = Asked::new(runtime.query(
                query_json.clone(),
                Some(session_json.clone()),
                Some("local".to_string()),
                None,
            ));
            if let Poll::Ready(answer) = asked.poll() {
                answered.lock().unwrap().push(answer.unwrap());
            }
            drop(asked);
            let handle = runtime
                .create_subscription(
                    query_json.clone(),
                    Some(session_json.clone()),
                    Some("local".to_string()),
                )
                .unwrap();
            runtime
                .execute_subscription(
                    handle,
                    Box::new(Subscriber {
                        js: Arc::clone(&js),
                        label: "its own",
                        unsubscribed: Arc::new(AtomicBool::new(false)),
                        handler: Mutex::new(None),
                    }),
                )
                .unwrap();
            js.tell(Told::Mark("the handler returns"));
        })),
    );
    stand.run_until_quiet();
    let row = stand.server_writes("row");
    stand.run_until("the handler's subscription is told", |stand| {
        stand.added("its own") == vec![row.clone()]
    });
    stand.run_until_quiet();

    let told = stand.told();
    let from = told
        .iter()
        .position(|told| *told == Told::Mark("the handler asks"))
        .unwrap();
    assert_eq!(
        told[from + 1],
        Told::Mark("the handler returns"),
        "told inside a handler"
    );
    let answers = answers.lock().unwrap();
    assert_eq!(answers.len(), 1, "the handler's query did not resolve");
    assert!(answers[0].contains(&row));
    assert_eq!(stand.violations(), Vec::<String>::new());
}

/// A query answered by a tick that is still running when the runtime is closed: the
/// promise resolves, and JS is told nothing of that tick.
#[test]
fn a_promise_waiting_for_its_tick_resolves_when_the_runtime_is_closed() {
    let mut stand = Stand::new("closed-while-waiting");
    stand.subscribe("notes");
    stand.server_writes("before");
    stand.run_until_quiet();

    stand.withhold = true;
    let row = stand.server_writes("with the answer");
    let runtime = Arc::clone(&stand.runtime);
    let mut asked = Asked::new(runtime.query(
        stand.query_json.clone(),
        Some(stand.session_json.clone()),
        Some("edge".to_string()),
        None,
    ));
    assert!(asked.poll().is_pending());
    stand.wire_until_quiet();
    stand
        .probe()
        .held_before_release
        .store(true, Ordering::SeqCst);
    stand.hand_over();
    stand.wait_until("the tick holds what it recorded", |stand| {
        stand.probe().holding.load(Ordering::SeqCst)
    });
    assert!(asked.poll().is_pending());
    let told = stand.told();

    let released = release_once_awaited(&stand.runtime);
    stand.runtime.close().unwrap();
    assert!(
        released.join().unwrap(),
        "the call did not wait for the tick"
    );
    let Poll::Ready(answer) = asked.poll() else {
        panic!("left waiting on a closed runtime");
    };
    assert!(answer.unwrap().contains(&row));
    stand.turn();
    assert_eq!(stand.told(), told, "told after close");
    assert_eq!(stand.violations(), Vec::<String>::new());
}

/// The worker asks JS for a drain by a call that waits for the JS thread. A call made
/// on the JS thread meanwhile must not wait for the worker: taking the tick callback
/// away and closing the runtime both return, with the worker still waiting.
#[test]
fn js_can_take_the_callback_away_and_close_while_the_worker_waits_for_it() {
    let mut stand = Stand::new("closed-during-the-hop");
    stand.subscribe("notes");
    stand.run_until_quiet();

    // In the middle of a tick: the callback goes without waiting for the tick.
    stand
        .probe()
        .held_before_release
        .store(true, Ordering::SeqCst);
    let first = stand.server_writes("first");
    stand.wire_until("the tick holds what it recorded", |stand| {
        stand.probe().holding.load(Ordering::SeqCst)
    });
    stand.runtime.on_batched_tick_needed(None).unwrap();
    assert!(
        stand.probe().holding.load(Ordering::SeqCst),
        "taking the callback away waited for the tick"
    );
    stand
        .runtime
        .on_batched_tick_needed(Some(Box::new(TickRequests(Arc::clone(&stand.js)))))
        .unwrap();
    stand
        .probe()
        .held_before_release
        .store(false, Ordering::SeqCst);
    stand.run_until("the row is told", |stand| {
        stand.added("notes") == vec![first.clone()]
    });
    stand.run_until_quiet();

    // In the middle of the worker's call into JS.
    stand.server_writes("second");
    stand.wait_until("nothing: the wire is still", |_| true);
    stand.wire_until("the worker waits for the JS thread", |stand| {
        stand.js.waiting.load(Ordering::SeqCst) > 0
    });
    let told = stand.told();
    let started = Instant::now();
    stand.runtime.on_batched_tick_needed(None).unwrap();
    stand.runtime.close().unwrap();
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "closing waited for the worker, which waits for the JS thread"
    );
    assert!(stand.js.waiting.load(Ordering::SeqCst) > 0);
    // The call it was waiting with is run at last, and its microtask after it.
    let started = Instant::now();
    while stand.js.waiting.load(Ordering::SeqCst) > 0 {
        stand.turn();
        stand.patience(started, "the worker's call into JS returns");
    }
    stand.turn();
    assert_eq!(stand.told(), told, "told after close");
    assert_eq!(stand.violations(), Vec::<String>::new());
}

/// A write the server refuses is reported from the worker, by a call that waits for
/// the JS thread: made with the core lock held it would never return.
#[test]
fn a_refused_write_is_reported_without_the_core_lock() {
    let mut stand = Stand::new("refused");
    stand
        .runtime
        .on_mutation_error(Box::new(MutationErrors(Arc::clone(&stand.js))))
        .unwrap();
    stand.subscribe("notes");
    stand.run_until_quiet();

    stand.withhold = true;
    let (_, batch) = stand.js_writes("mine");
    stand.server_rejects(&batch);
    stand.run_until("the refusal is reported", |stand| {
        stand.told().contains(&Told::MutationError(batch.clone()))
    });
    stand.run_until_quiet();
    assert_eq!(stand.violations(), Vec::<String>::new());
}

/// What JS is told, against what the engine holds.
///
/// Rows are written, changed and deleted by the app and by a server, subscriptions come
/// and go, queries are asked, handlers ask for more from inside a delta, and the JS loop
/// turns when it turns — with the engine's ticks on their own thread racing all of it.
///
/// - Every delta told to a subscription applies to what the subscription was told
///   before: nothing added twice, nothing changed or removed that it never had.
/// - Nothing is told to a subscription that was unsubscribed.
/// - A query that resolves finds every subscription at least as far along as its answer,
///   whether it resolves on the spot or steps later.
/// - Whenever JS looks, its live subscriptions have been told the same: what one hold
///   of the core recorded is told together, however many holds JS is told of at once.
/// - When everything has gone quiet, what each live subscription has been told adds up
///   to the rows a query returns.
/// - The same script with everything allowed to settle between two steps ends with the
///   same rows.
#[test]
fn what_js_is_told_adds_up_to_what_the_engine_holds() {
    struct Rng(u64);
    impl Rng {
        fn below(&mut self, bound: usize) -> usize {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 % bound as u64) as usize
        }
    }

    const LABELS: [&str; 8] = ["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7"];

    /// What a subscription has been told, replayed: the rows it has, and those it had.
    fn replay(told: &[Told], label: &'static str) -> (BTreeMap<String, String>, Vec<String>) {
        let mut rows = BTreeMap::new();
        let mut gone = Vec::new();
        for told in told {
            let Told::Delta {
                subscription,
                added,
                updated,
                removed,
                rows: texts,
            } = told
            else {
                continue;
            };
            if *subscription != label {
                continue;
            }
            let text = |id: &String| texts.iter().find(|(of, _)| of == id).map(|(_, text)| text);
            for id in removed {
                assert!(
                    rows.remove(id).is_some(),
                    "{label}: removed a row it was never told of"
                );
                gone.push(id.clone());
            }
            for id in updated {
                let row = rows
                    .get_mut(id)
                    .unwrap_or_else(|| panic!("{label}: updated a row it was never told of"));
                if let Some(text) = text(id) {
                    *row = text.clone();
                }
            }
            for id in added {
                let text = text(id)
                    .expect("an added row comes with its values")
                    .clone();
                assert!(
                    rows.insert(id.clone(), text).is_none(),
                    "{label}: added a row it had been told of"
                );
            }
        }
        (rows, gone)
    }

    fn rows_of(answer: &str) -> BTreeMap<String, String> {
        serde_json::from_str::<Vec<serde_json::Value>>(answer)
            .unwrap()
            .iter()
            .map(|row| {
                (
                    row["id"].as_str().unwrap().to_string(),
                    row["values"][0]["value"].as_str().unwrap().to_string(),
                )
            })
            .collect()
    }

    /// Every text is written once, numbered in the order of the script.
    fn written_at(text: &str) -> u64 {
        text[1..].parse().unwrap()
    }

    type Pending = Asked<'static, Result<String, JazzRnError>>;

    /// A query, as the app asks it; it owns what it needs, so it can wait across steps.
    fn ask(stand: &Stand, tier: &'static str) -> Pending {
        let runtime = Arc::clone(&stand.runtime);
        let query = stand.query_json.clone();
        let session = stand.session_json.clone();
        Asked::new(async move {
            runtime
                .query(query, Some(session), Some(tier.to_string()), None)
                .await
        })
    }

    /// A query has resolved: every live subscription has been told at least that much.
    /// The answer may be steps old by the time it is taken: a row in it that has been
    /// deleted since is one a subscription made after that was never told of.
    fn not_behind(
        stand: &Stand,
        live: &[(&'static str, Subscription)],
        answer: &str,
        exists: &dyn Fn(&str) -> bool,
        seed: u64,
    ) {
        let answer = rows_of(answer);
        let told = stand.told();
        for (label, _) in live {
            let (rows, gone) = replay(&told, label);
            for (id, text) in &answer {
                match rows.get(id) {
                    Some(has) => assert!(
                        written_at(has) >= written_at(text),
                        "seed {seed}: {label} is behind a query's answer"
                    ),
                    None => assert!(
                        gone.contains(id) || !exists(id),
                        "seed {seed}: {label} was not told of a row a query returned"
                    ),
                }
            }
        }
    }

    /// What one hold of the core recorded, JS is told together: whenever JS looks,
    /// outside a handler, its live subscriptions to the same query have the same rows.
    fn agree(stand: &Stand, live: &[(&'static str, Subscription)], seed: u64, step: usize) {
        let told = stand.told();
        let mut first: Option<(&'static str, BTreeMap<String, String>)> = None;
        for (label, _) in live {
            let (rows, _) = replay(&told, label);
            match &first {
                None => first = Some((label, rows)),
                Some((other, has)) => assert_eq!(
                    &rows, has,
                    "seed {seed}, step {step}: {label} and {other} were told different things"
                ),
            }
        }
    }

    /// Runs the script for `seed`; returns the texts of the rows it ends with.
    fn run(seed: u64, racing: bool) -> Vec<String> {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15 ^ (seed + 1).wrapping_mul(0x2545_f491_4f6c_dd1d));
        let mut stand = Stand::new(&format!("differential-{seed}-{racing}"));
        let mut live: Vec<(&'static str, Subscription)> = Vec::new();
        let mut labels = LABELS.iter();
        let mut mine: Vec<String> = Vec::new();
        let mut theirs: Vec<ObjectId> = Vec::new();
        let mut pending: Vec<Pending> = Vec::new();
        let mut written = 0;

        for step in 0..120 {
            let still: Vec<String> = mine
                .iter()
                .cloned()
                .chain(theirs.iter().map(|id| id.uuid().to_string()))
                .collect();
            let exists = |id: &str| still.iter().any(|has| has == id);
            match rng.below(21) {
                0 | 1 => {
                    written += 1;
                    let (id, _) = stand.js_writes(&format!("j{written}"));
                    mine.push(id);
                }
                2 if !mine.is_empty() => {
                    written += 1;
                    let id = mine[rng.below(mine.len())].clone();
                    stand
                        .runtime
                        .update(id, note(&format!("j{written}")), None)
                        .unwrap();
                }
                3 if !mine.is_empty() => {
                    let id = mine.swap_remove(rng.below(mine.len()));
                    stand.runtime.delete_row(id, None).unwrap();
                }
                4 | 5 => {
                    written += 1;
                    let ((id, _), _) = stand
                        .server
                        .insert(
                            "notes",
                            [("text".to_string(), Value::Text(format!("s{written}")))].into(),
                            None,
                        )
                        .unwrap();
                    theirs.push(id);
                }
                6 if !theirs.is_empty() => {
                    written += 1;
                    let id = theirs[rng.below(theirs.len())];
                    stand
                        .server
                        .update(
                            id,
                            vec![("text".to_string(), Value::Text(format!("s{written}")))],
                            None,
                        )
                        .unwrap();
                }
                7 if !theirs.is_empty() => {
                    let id = theirs.swap_remove(rng.below(theirs.len()));
                    stand.server.delete(id, None).unwrap();
                }
                8 if live.len() < 3 => {
                    let Some(label) = labels.next() else {
                        continue;
                    };
                    // Every other one asks for more from inside its handler: a query,
                    // and a subscription it drops again before it can be told anything.
                    let handler: Option<Handler> = (rng.below(2) == 0).then(|| {
                        let runtime = Arc::clone(&stand.runtime);
                        let js = Arc::clone(&stand.js);
                        let query_json = stand.query_json.clone();
                        let session_json = stand.session_json.clone();
                        Box::new(move |_: &Told| {
                            let mut asked = Asked::new(runtime.query(
                                query_json.clone(),
                                Some(session_json.clone()),
                                Some("local".to_string()),
                                None,
                            ));
                            let _ = asked.poll();
                            drop(asked);
                            let handle = runtime
                                .create_subscription(
                                    query_json.clone(),
                                    Some(session_json.clone()),
                                    Some("local".to_string()),
                                )
                                .unwrap();
                            let unsubscribed = Arc::new(AtomicBool::new(false));
                            runtime
                                .execute_subscription(
                                    handle,
                                    Box::new(Subscriber {
                                        js: Arc::clone(&js),
                                        label: "dropped by a handler",
                                        unsubscribed: Arc::clone(&unsubscribed),
                                        handler: Mutex::new(None),
                                    }),
                                )
                                .unwrap();
                            runtime.unsubscribe(handle).unwrap();
                            unsubscribed.store(true, Ordering::SeqCst);
                        }) as Handler
                    });
                    live.push((label, stand.subscribe_with(label, handler)));
                }
                9 if !live.is_empty() => {
                    let (_, gone) = live.swap_remove(rng.below(live.len()));
                    stand.unsubscribe(&gone);
                }
                10 => stand.turn(),
                11 => {
                    stand.carry();
                }
                12 | 13 => {
                    // A query, polled as JS polls it; one that has no answer on the
                    // spot is polled again later.
                    let mut asked = ask(&stand, if rng.below(2) == 0 { "local" } else { "edge" });
                    match asked.poll() {
                        Poll::Ready(answer) => {
                            not_behind(&stand, &live, &answer.unwrap(), &exists, seed)
                        }
                        Poll::Pending => pending.push(asked),
                    }
                }
                14 => {
                    let mut waiting = Vec::new();
                    for mut asked in pending.drain(..) {
                        match asked.poll() {
                            Poll::Ready(answer) => {
                                not_behind(&stand, &live, &answer.unwrap(), &exists, seed);
                            }
                            Poll::Pending => waiting.push(asked),
                        }
                    }
                    pending = waiting;
                }
                // JS answers what the worker asks of it and gets to nothing else: the
                // worker goes on, and what its next tick records waits with the last.
                15 => stand.answer(),
                16 if !theirs.is_empty() => {
                    // Two ticks' worth for JS to be told at once: what became of a
                    // row, and then what became of it after that.
                    for _ in 0..2 {
                        written += 1;
                        let id = theirs[rng.below(theirs.len())];
                        stand
                            .server
                            .update(
                                id,
                                vec![("text".to_string(), Value::Text(format!("s{written}")))],
                                None,
                            )
                            .unwrap();
                        stand.answer_until_quiet();
                    }
                }
                17 => {
                    // Nobody to ask for a while: what is recorded meanwhile is asked
                    // for when there is somebody again.
                    stand.runtime.on_batched_tick_needed(None).unwrap();
                    stand.carry();
                    std::thread::sleep(Duration::from_millis(rng.below(4) as u64));
                    stand
                        .runtime
                        .on_batched_tick_needed(Some(Box::new(TickRequests(Arc::clone(&stand.js)))))
                        .unwrap();
                }
                _ => std::thread::sleep(Duration::from_millis(rng.below(3) as u64)),
            }
            agree(&stand, &live, seed, step);
            if !racing {
                stand.run_until_quiet();
                agree(&stand, &live, seed, step);
            }
        }
        stand.run_until_quiet();
        agree(&stand, &live, seed, usize::MAX);

        // Every query asked along the way resolves, and finds the same.
        let still: Vec<String> = mine
            .iter()
            .cloned()
            .chain(theirs.iter().map(|id| id.uuid().to_string()))
            .collect();
        let exists = |id: &str| still.iter().any(|has| has == id);
        let started = Instant::now();
        while !pending.is_empty() {
            let mut waiting = Vec::new();
            for mut asked in pending.drain(..) {
                match asked.poll() {
                    Poll::Ready(answer) => {
                        not_behind(&stand, &live, &answer.unwrap(), &exists, seed)
                    }
                    Poll::Pending => waiting.push(asked),
                }
            }
            pending = waiting;
            stand.turn();
            stand.carry();
            stand.patience(started, "the queries asked along the way resolve");
        }

        // What the server holds, asked for the way the app asks.
        let answer = {
            let mut asked = ask(&stand, "edge");
            let started = Instant::now();
            loop {
                if let Poll::Ready(answer) = asked.poll() {
                    break answer.unwrap();
                }
                stand.turn();
                stand.carry();
                stand.patience(started, "the final query resolves");
            }
        };
        stand.run_until_quiet();
        let holds = rows_of(&answer);
        assert_eq!(holds.len(), mine.len() + theirs.len(), "seed {seed}");

        let told = stand.told();
        for (label, _) in &live {
            assert_eq!(
                replay(&told, label).0,
                holds,
                "seed {seed}, subscription {label}"
            );
        }
        for label in LABELS {
            // Applies cleanly as far as it went, for those that are gone as well.
            replay(&told, label);
        }
        assert_eq!(
            added(&told, "dropped by a handler"),
            Vec::<String>::new(),
            "seed {seed}"
        );
        assert_eq!(stand.violations(), Vec::<String>::new(), "seed {seed}");
        stand.runtime.close().unwrap();

        let mut texts: Vec<String> = holds.into_values().collect();
        texts.sort();
        texts
    }

    for seed in 0..12 {
        assert_eq!(run(seed, true), run(seed, false), "seed {seed}");
    }
}
