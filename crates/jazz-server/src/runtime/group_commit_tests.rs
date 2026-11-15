//! Group commit ([`group_commit`]) on a file store: the runtime commits at
//! `synchronous = NORMAL` and syncs the WAL once per tick that committed
//! anything, before any of that tick's output leaves, and before a runtime
//! job replies; host writes and runtime jobs that reply commit at `FULL`; a
//! failed sync fail-stops.
//! The owner records each step in order ([`Trace`]), and every output that
//! left while its connection had row changes no sync covered, which these
//! gates read back.

use super::*;
use jazz::tools::public_schema::{ColumnType, SchemaBuilder, TableSchema};

const FULL: i64 = 2;
const NORMAL: i64 = 1;

fn todos_source() -> jazz::tools::public_schema::Schema {
    SchemaBuilder::new()
        .table(TableSchema::builder("todos").column("title", ColumnType::Text))
        .build()
}

fn todos() -> JazzSchema {
    JazzSchema::new(&todos_source()).unwrap()
}

/// A file store in `dir`.
fn grouped(dir: &std::path::Path) -> AppStore {
    AppStore::open(&StorageConfig::data_dir(dir)).unwrap()
}

/// `PRAGMA synchronous` on the connection `tables` writes through.
fn synchronous(tables: &dyn HostTables) -> i64 {
    let mut level = -1;
    tables
        .host_sql(&mut |conn| {
            level = conn
                .query_row("PRAGMA synchronous", [], |row| row.get(0))
                .map_err(|error| {
                    jazz::db::Error::new(jazz::db::ErrorCode::Storage, error.to_string())
                })?;
            Ok(())
        })
        .unwrap();
    level
}

/// `PRAGMA synchronous` on the runtime's connection, read by a job that
/// does not reply from inside (as the runtime's own commits run).
fn runtime_synchronous(runtime: &ServerRuntimeHandle) -> i64 {
    let (tx, rx) = std_mpsc::channel();
    runtime
        .send(Box::new(move |owner| {
            let _ = tx.send(synchronous(&owner.db));
        }))
        .unwrap();
    rx.recv().unwrap()
}

/// `PRAGMA synchronous` inside a host job.
fn host_synchronous(store: &AppStore) -> i64 {
    let (tx, rx) = std_mpsc::channel();
    HostSender {
        host: store.inner.host.clone(),
        wake: store.sender().unwrap(),
    }
    .send(Box::new(move |tables: &dyn HostTables| {
        let _ = tx.send(synchronous(tables));
    }))
    .unwrap();
    rx.recv().unwrap()
}

fn trace(runtime: &ServerRuntimeHandle) -> Vec<Trace> {
    block_on(runtime.run(|owner| Ok(owner.trace.clone()))).unwrap()
}

/// Every send happened inside a tick, after that tick's WAL sync (or its
/// finding that nothing was committed since the last); every reply right
/// after its job's (or its fail-stop, when the reply is that error).
#[track_caller]
fn assert_sent_after_the_ticks_sync(trace: &[Trace]) {
    let (mut in_tick, mut synced) = (false, false);
    for (at, step) in trace.iter().enumerate() {
        let around = || &trace[at.saturating_sub(6)..(at + 2).min(trace.len())];
        match step {
            Trace::Tick => (in_tick, synced) = (true, false),
            Trace::Synced(_) | Trace::Clean => synced = true,
            Trace::FailStop | Trace::TickEnd => (in_tick, synced) = (false, false),
            Trace::Sent(_) | Trace::ThinSent(_) | Trace::ChunkIo => assert!(
                in_tick && synced,
                "{step:?} left before its tick's WAL sync (step {at}): {:?}",
                around()
            ),
            Trace::Replied => assert!(
                at > 0
                    && matches!(
                        trace[at - 1],
                        Trace::Synced(_) | Trace::Clean | Trace::FailStop
                    ),
                "a reply left without its job's WAL sync (step {at}): {:?}",
                around()
            ),
        }
    }
}

/// No output left while the runtime's connection had row changes that no
/// WAL sync covered.
#[track_caller]
fn assert_no_unsynced_output(runtime: &ServerRuntimeHandle) {
    let unsynced = block_on(runtime.run(|owner| Ok(owner.unsynced_output.clone()))).unwrap();
    assert!(
        unsynced.is_empty(),
        "output left before a WAL sync covered its changes: {unsynced:?}"
    );
}

/// Every traced `Synced` step was a real fdatasync (a sync that returns Ok
/// without writing would leave the count short), the open's first.
#[track_caller]
fn assert_every_sync_reached_the_disk(runtime: &ServerRuntimeHandle) {
    let (trace, syncs) = block_on(runtime.run(|owner| {
        Ok((
            owner.trace.clone(),
            owner.wal.as_ref().map(|wal| wal.total_syncs),
        ))
    }))
    .unwrap();
    // The k-th traced sync is the WAL's k-th fdatasync: a sync moved after
    // the sends, its trace left in place, would carry the count from
    // before it.
    let stamps: Vec<u64> = trace
        .iter()
        .filter_map(|step| match step {
            Trace::Synced(count) => Some(*count),
            _ => None,
        })
        .collect();
    let expected: Vec<u64> = (1..stamps.len() as u64 + 1).collect();
    assert_eq!(
        stamps, expected,
        "each traced sync is one fdatasync more: {trace:?}"
    );
    assert_eq!(
        syncs,
        Some(stamps.len() as u64),
        "one fdatasync per traced sync: {trace:?}"
    );
}

/// The client side of a sync session: a replica whose frames the test
/// carries to and from the runtime.
#[derive(Clone, Default)]
struct Wire(Rc<RefCell<(VecDeque<Vec<u8>>, VecDeque<Vec<u8>>)>>);

impl jazz::db::WireTransport for Wire {
    fn send_frame(&mut self, frame: Vec<u8>) -> Result<(), TransportError> {
        self.0.borrow_mut().1.push_back(frame);
        Ok(())
    }

    fn try_recv_frame(&mut self) -> Option<Vec<u8>> {
        self.0.borrow_mut().0.pop_front()
    }
}

struct SyncClient {
    db: Db,
    wire: Wire,
    _upstream: jazz::db::Upstream,
    session: ServerSession,
    stream: ServerRuntimeFrameStream,
}

impl SyncClient {
    fn open(runtime: &ServerRuntimeHandle, node: u8) -> Self {
        let db = block_on(Db::open(DbConfig::new(
            todos(),
            DbIdentity {
                node: NodeUuid::from_bytes([node; 16]),
                author: AuthorSubject::SYSTEM,
            },
        )))
        .unwrap();
        let wire = Wire::default();
        let upstream = block_on(db.connect_upstream(Box::new(
            jazz::db::WireTransportAdapter::current(wire.clone()),
        )));
        let session = block_on(runtime.open_with_session_context(
            AuthorSubject::SYSTEM,
            BTreeMap::new(),
            None,
            CommitUnitTrust::TrustedBackend,
            jazz::wire::current_wire_features(),
            None,
            ServerLinkAdmission::OrdinarySession,
        ))
        .unwrap();
        let stream = runtime.open_wire_stream(session).unwrap();
        Self {
            db,
            wire,
            _upstream: upstream,
            session,
            stream,
        }
    }

    /// What the client sends goes to the runtime; what the runtime sent
    /// comes back.
    fn exchange(&mut self, runtime: &ServerRuntimeHandle) {
        block_on(self.db.tick()).unwrap();
        let frames: Vec<Vec<u8>> = self.wire.0.borrow_mut().1.drain(..).collect();
        if !frames.is_empty() {
            let bytes = frames.iter().map(Vec::len).sum();
            let room = block_on(runtime.admit_wire_frames(self.session, bytes)).unwrap();
            runtime
                .receive_wire_frames(self.session, frames, room)
                .unwrap();
            runtime.request_wire_tick(self.session).unwrap();
        }
        while let Ok((frames, bytes)) = self.stream.receiver.try_recv() {
            release_egress(&self.stream.egress, bytes, &self.stream.runtime);
            self.wire.0.borrow_mut().0.extend(frames.unwrap());
        }
        block_on(self.db.tick()).unwrap();
    }

    fn titles(&self) -> Vec<String> {
        let rows = block_on(
            self.db.all(
                &self
                    .db
                    .prepare_query(&jazz::query::Query::from("todos"))
                    .unwrap(),
                jazz::db::ReadOpts {
                    tier: jazz::db::ReadTier::LocalOnly,
                    ..Default::default()
                },
            ),
        )
        .unwrap();
        rows.iter()
            .filter_map(|row| match row.cell("title") {
                Some(Value::String(title)) => Some(title),
                _ => None,
            })
            .collect()
    }
}

/// Carry frames both ways until `done`, ticking the runtime between.
fn pump(
    runtime: &ServerRuntimeHandle,
    clients: &mut [SyncClient],
    done: impl Fn(&[SyncClient]) -> bool,
) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !done(clients) {
        assert!(Instant::now() < deadline, "the clients never got there");
        for client in clients.iter_mut() {
            client.exchange(runtime);
        }
        // A job is followed by a tick.
        block_on(runtime.run(|_| Ok(()))).unwrap();
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// Gate 1: a sync session's commit is answered (its fate, and the row's
/// fan-out to another session) only after the tick that committed it has
/// synced the WAL.
#[test]
fn a_session_commits_output_leaves_after_the_ticks_wal_sync() {
    let dir = tempfile::tempdir().unwrap();
    let store = grouped(dir.path());
    let runtime = store.start_runtime(todos(), NodeRole::Core, false).unwrap();
    let mut clients = vec![SyncClient::open(&runtime, 1), SyncClient::open(&runtime, 2)];
    let reader = clients[1]
        .db
        .prepare_query(&jazz::query::Query::from("todos"))
        .unwrap();
    let _live = block_on(clients[1].db.subscribe(
        &reader,
        jazz::db::ReadOpts::for_read_tier(jazz::db::ReadTier::Remote),
    ))
    .unwrap();
    pump(&runtime, &mut clients, |_| true);
    block_on(clients[0].db.insert(
        "todos",
        BTreeMap::from([("title".to_owned(), Value::String("t".to_owned()))]),
        Default::default(),
    ))
    .unwrap();
    pump(&runtime, &mut clients, |clients| {
        clients[1].titles() == ["t"]
    });
    // The writer's fate, if it was not in the same tick.
    let rounds = std::cell::Cell::new(0);
    pump(&runtime, &mut clients, |_| {
        rounds.set(rounds.get() + 1);
        rounds.get() > 5
    });
    let trace = trace(&runtime);
    let (writer, reader) = (clients[0].session, clients[1].session);
    assert!(
        trace.contains(&Trace::Sent(writer)) && trace.contains(&Trace::Sent(reader)),
        "both sessions were sent frames: {trace:?}"
    );
    assert_sent_after_the_ticks_sync(&trace);
    assert_every_sync_reached_the_disk(&runtime);
    assert_no_unsynced_output(&runtime);
    store.close().unwrap();
}

/// Gate 2: a thin write's answer leaves only after a tick's WAL sync.
#[test]
fn a_thin_writes_answer_leaves_after_the_ticks_wal_sync() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let store = grouped(dir.path());
    let runtime = store.start_runtime(docs(), NodeRole::Core, false).unwrap();
    rt.block_on(async {
        let (session, mut stream) = thin_docs_session(&runtime).await;
        send_thin(&runtime, session, 1, insert_doc()).await;
        assert!(
            matches!(
                next_thin(&mut stream, Duration::from_secs(5)).await,
                Some((1, jazz::thin::ThinBody::Written { .. }))
            ),
            "the insert is answered"
        );
        let trace = trace(&runtime);
        assert!(
            trace.contains(&Trace::ThinSent(session)),
            "the answer was sent: {trace:?}"
        );
        assert_sent_after_the_ticks_sync(&trace);
        assert_every_sync_reached_the_disk(&runtime);
        assert_no_unsynced_output(&runtime);
    });
    store.close().unwrap();
}

/// Gate 3: a runtime job that replies from inside commits at FULL, and
/// the runtime's own commits run at NORMAL around it.
#[test]
fn a_runtime_job_that_replies_commits_at_full() {
    let dir = tempfile::tempdir().unwrap();
    let store = grouped(dir.path());
    let runtime = store.start_runtime(todos(), NodeRole::Core, false).unwrap();
    assert_eq!(
        runtime_synchronous(&runtime),
        NORMAL,
        "the runtime commits at NORMAL"
    );
    let inside = block_on(runtime.run(|owner| {
        block_on(owner.db.insert(
            "todos",
            BTreeMap::from([("title".to_owned(), Value::String("t".to_owned()))]),
            Default::default(),
        ))
        .map_err(|error| error.to_string())?;
        Ok(synchronous(&owner.db))
    }))
    .unwrap();
    assert_eq!(
        inside, FULL,
        "a job that replies from inside commits at FULL"
    );
    assert_eq!(runtime_synchronous(&runtime), NORMAL, "and NORMAL after it");
    store.close().unwrap();
}

/// Gate 4: a host job runs at FULL, with no runtime and with one.
#[test]
fn host_jobs_commit_at_full_with_or_without_a_runtime() {
    let dir = tempfile::tempdir().unwrap();
    let store = grouped(dir.path());
    assert_eq!(host_synchronous(&store), FULL, "a host job with no runtime");
    let runtime = store.start_runtime(todos(), NodeRole::Core, false).unwrap();
    assert_eq!(
        host_synchronous(&store),
        FULL,
        "a host job beside the runtime"
    );
    assert_eq!(runtime_synchronous(&runtime), NORMAL, "and NORMAL after it");
    store.close().unwrap();
}

/// Gate 5: a failed WAL sync fail-stops the store, and nothing the tick
/// produced is sent. (In a process the fail-stop aborts; here a counter
/// takes its place and the answer goes out on the next tick.)
#[test]
fn a_failed_wal_sync_fail_stops_before_anything_is_sent() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let store = grouped(dir.path());
    let runtime = store.start_runtime(docs(), NodeRole::Core, false).unwrap();
    let stops = Arc::new(AtomicU64::new(0));
    rt.block_on(async {
        let (session, mut stream) = thin_docs_session(&runtime).await;
        // The failing sync and the write land in one tick.
        let release = hold(&runtime);
        let counted = Arc::clone(&stops);
        runtime
            .send(Box::new(move |owner| {
                let wal = owner.wal.as_mut().expect("group commit is on");
                wal.stops = Some(counted);
                wal.fail_next = true;
            }))
            .unwrap();
        send_thin(&runtime, session, 1, insert_doc()).await;
        release.send(()).unwrap();
        assert!(
            matches!(
                next_thin(&mut stream, Duration::from_secs(5)).await,
                Some((1, jazz::thin::ThinBody::Written { .. }))
            ),
            "the insert is answered by a later tick"
        );
        let trace = trace(&runtime);
        assert_eq!(
            stops.load(Ordering::SeqCst),
            1,
            "the failed sync fail-stopped"
        );
        let failed = trace
            .iter()
            .position(|step| *step == Trace::FailStop)
            .unwrap_or_else(|| panic!("no fail-stop in {trace:?}"));
        // The insert's answer is the session's last send (the schema's came
        // before).
        let answered = trace
            .iter()
            .rposition(|step| *step == Trace::ThinSent(session))
            .expect("the answer was sent");
        assert!(
            answered > failed,
            "the answer was held over the failed tick: {trace:?}"
        );
        assert_sent_after_the_ticks_sync(&trace);
        assert_every_sync_reached_the_disk(&runtime);
        assert_no_unsynced_output(&runtime);
    });
    store.close().unwrap();
}

/// Gate 6: a file store runs group commit with no option to ask for it
/// (the file opens at NORMAL, host jobs switch to FULL, a runtime gets a
/// WAL to sync). An in-memory store has no WAL. A store whose large values
/// live in a directory commits at FULL everywhere and syncs nothing itself:
/// the engine deletes reclaimed chunk files inside its own tick.
#[test]
fn which_stores_group_their_commits() {
    let dir = tempfile::tempdir().unwrap();
    let store = AppStore::open(&StorageConfig::data_dir(dir.path())).unwrap();
    assert_eq!(host_synchronous(&store), FULL, "a host job commits at FULL");
    let runtime = store.start_runtime(todos(), NodeRole::Core, false).unwrap();
    assert_eq!(
        runtime_synchronous(&runtime),
        NORMAL,
        "the runtime commits at NORMAL"
    );
    assert!(
        block_on(runtime.run(|owner| Ok(owner.wal.is_some()))).unwrap(),
        "and syncs its WAL"
    );
    store.close().unwrap();

    let memory = AppStore::open(&StorageConfig::InMemory).unwrap();
    let runtime = memory
        .start_runtime(todos(), NodeRole::Core, false)
        .unwrap();
    assert!(
        block_on(runtime.run(|owner| Ok(owner.wal.is_none()))).unwrap(),
        "an in-memory store has no WAL to sync"
    );
    memory.close().unwrap();

    let (dir, blobs) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let store =
        AppStore::open(&StorageConfig::data_dir(dir.path()).with_blob_dir(blobs.path())).unwrap();
    assert_eq!(host_synchronous(&store), FULL);
    let runtime = store.start_runtime(todos(), NodeRole::Core, false).unwrap();
    assert_eq!(
        runtime_synchronous(&runtime),
        FULL,
        "a store with a chunk directory commits at FULL"
    );
    block_on(runtime.run(|_| Ok(()))).unwrap();
    let trace = trace(&runtime);
    assert!(
        block_on(runtime.run(|owner| Ok(owner.wal.is_none()))).unwrap()
            && trace.contains(&Trace::Tick)
            && !trace
                .iter()
                .any(|step| matches!(step, Trace::Synced(_) | Trace::Clean)),
        "and syncs no WAL itself: {trace:?}"
    );
    store.close().unwrap();
}

/// Gate 7: what opening the runtime wrote (migrations, admissions, the
/// authority mint) is synced before `start_runtime` returns: the open's
/// fdatasync is the first step traced, ahead of any tick.
#[test]
fn opening_the_runtime_syncs_the_wal_before_it_serves() {
    let dir = tempfile::tempdir().unwrap();
    let store = grouped(dir.path());
    let runtime = store.start_runtime(todos(), NodeRole::Core, false).unwrap();
    let trace = trace(&runtime);
    assert_eq!(
        trace.first(),
        Some(&Trace::Synced(1)),
        "the open synced before anything ran: {trace:?}"
    );
    assert_every_sync_reached_the_disk(&runtime);
    store.close().unwrap();
}

/// Gate 8: outside the tests' counting hook a failed WAL sync aborts the
/// process (a logged failure would let the next sync succeed and the held
/// answers go out). Run in a child copy of this test binary.
#[cfg(unix)]
#[test]
fn a_failed_wal_sync_aborts_the_process() {
    const CHILD: &str = "JAZZ_GROUP_COMMIT_ABORT_CHILD";
    if let Some(dir) = std::env::var_os(CHILD) {
        let store = grouped(std::path::Path::new(&dir));
        let runtime = store.start_runtime(todos(), NodeRole::Core, false).unwrap();
        runtime
            .send(Box::new(|owner| {
                owner.wal.as_mut().expect("group commit is on").fail_next = true;
            }))
            .unwrap();
        // The sync before the reply of a job that wrote fails.
        block_on(runtime.run(|owner| {
            block_on(owner.db.insert(
                "todos",
                BTreeMap::from([("title".to_owned(), Value::String("t".to_owned()))]),
                Default::default(),
            ))
            .map(|_| ())
            .map_err(|error| error.to_string())
        }))
        .unwrap();
        std::thread::sleep(Duration::from_secs(10));
        eprintln!("child: still alive after the failed sync");
        std::process::exit(3);
    }
    let name = format!(
        "{}::a_failed_wal_sync_aborts_the_process",
        module_path!().split_once("::").unwrap().1
    );
    // The child aborts without cleaning up: its store lives in ours.
    let dir = tempfile::tempdir().unwrap();
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", &name, "--nocapture", "--test-threads=1"])
        .env(CHILD, dir.path())
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    use std::os::unix::process::ExitStatusExt;
    assert_eq!(
        out.status.signal(),
        Some(6),
        "the child dies of SIGABRT: {:?}\n{stderr}",
        out.status
    );
    assert!(
        stderr.contains("WAL fdatasync failed, aborting the process"),
        "and says why: {stderr}"
    );
}

/// Gate 9: a failed sync at open fail-stops and the runtime does not start
/// (in a process the fail-stop aborts; here a counter takes its place); the
/// store stays usable and a later start succeeds.
#[test]
fn a_failed_sync_at_open_fail_stops_and_does_not_start() {
    let dir = tempfile::tempdir().unwrap();
    let store = grouped(dir.path());
    let stops = group_commit::tests::fail_open_sync(dir.path());
    assert!(
        store.start_runtime(todos(), NodeRole::Core, false).is_err(),
        "the runtime does not start"
    );
    assert_eq!(
        stops.load(Ordering::SeqCst),
        1,
        "the failed sync fail-stopped"
    );
    assert_eq!(
        host_synchronous(&store),
        FULL,
        "the store serves host jobs again"
    );
    let runtime = store.start_runtime(todos(), NodeRole::Core, false).unwrap();
    assert_every_sync_reached_the_disk(&runtime);
    store.close().unwrap();
}

/// Gate 10: a runtime job's reply waits for a WAL sync, even when the job
/// itself wrote nothing: a job without a reply committed at NORMAL just
/// before it, in the same batch, with no tick between.
#[test]
fn a_reply_after_a_normal_commit_waits_for_the_wal_sync() {
    let dir = tempfile::tempdir().unwrap();
    let store = grouped(dir.path());
    let runtime = store.start_runtime(todos(), NodeRole::Core, false).unwrap();
    // A batch ends once an item overruns the tick slice, so a slow insert
    // lets a tick in between: try until the pair shares one batch.
    let mut shared = None;
    for _ in 0..50 {
        let release = hold(&runtime);
        let (committed_at, committed) = std_mpsc::channel();
        runtime
            .send(Box::new(move |owner| {
                block_on(owner.db.insert(
                    "todos",
                    BTreeMap::from([("title".to_owned(), Value::String("t".to_owned()))]),
                    Default::default(),
                ))
                .unwrap();
                committed_at.send(owner.trace.len()).unwrap();
            }))
            .unwrap();
        let replied = {
            let runtime = runtime.clone();
            std::thread::spawn(move || block_on(runtime.run(|owner| Ok(owner.trace.len()))))
        };
        // Both jobs queue behind the hold.
        std::thread::sleep(Duration::from_millis(20));
        release.send(()).unwrap();
        let at = replied.join().unwrap().unwrap();
        let from = committed.recv().unwrap();
        let trace = trace(&runtime);
        if !trace[from..at].contains(&Trace::Tick) {
            shared = Some(trace);
            break;
        }
    }
    let trace = shared.expect("the commit and the reply never shared a batch");
    assert_no_unsynced_output(&runtime);
    assert_sent_after_the_ticks_sync(&trace);
    assert_every_sync_reached_the_disk(&runtime);
    store.close().unwrap();
}

/// The WAL's fdatasyncs so far.
fn syncs(runtime: &ServerRuntimeHandle) -> u64 {
    block_on(runtime.run(|owner| Ok(owner.wal.as_ref().expect("group commit is on").total_syncs)))
        .unwrap()
}

/// Gate 12: only a tick (or reply) with something committed since the last
/// sync pays an fdatasync: idle ticks and read-only jobs pay none, a thin
/// write pays one.
#[test]
fn only_what_committed_is_synced() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let store = grouped(dir.path());
    let runtime = store.start_runtime(docs(), NodeRole::Core, false).unwrap();
    rt.block_on(async {
        let (session, mut stream) = thin_docs_session(&runtime).await;
        let before = syncs(&runtime);
        let from = trace(&runtime).len();
        // Read-only jobs, each followed by a tick; then an idle tick.
        for _ in 0..5 {
            block_on(runtime.run(|owner| owner.db.catalogue().map_err(|e| e.to_string()))).unwrap();
        }
        send_thin(&runtime, session, 1, thin_query(None)).await;
        assert!(
            matches!(
                next_thin(&mut stream, Duration::from_secs(5)).await,
                Some((1, jazz::thin::ThinBody::Rows { .. }))
            ),
            "the query is answered"
        );
        tokio::time::sleep(IDLE_TICK + Duration::from_millis(300)).await;
        assert_eq!(syncs(&runtime), before, "nothing committed, nothing synced");
        let quiet = trace(&runtime);
        assert!(
            quiet[from..]
                .iter()
                .filter(|step| **step == Trace::Tick)
                .count()
                >= 6
                && quiet[from..].contains(&Trace::ThinSent(session)),
            "ticks ran and the answer left meanwhile: {:?}",
            &quiet[from..]
        );
        send_thin(&runtime, session, 2, insert_doc()).await;
        assert!(
            matches!(
                next_thin(&mut stream, Duration::from_secs(5)).await,
                Some((2, jazz::thin::ThinBody::Written { .. }))
            ),
            "the insert is answered"
        );
        assert_eq!(syncs(&runtime), before + 1, "one commit, one sync");
        assert_no_unsynced_output(&runtime);
        assert_sent_after_the_ticks_sync(&trace(&runtime));
    });
    store.close().unwrap();
}

/// Gate 13: a thin write that committed is answered even when the runtime
/// stops before the next tick: stopping syncs and sends what was held.
#[test]
fn a_stop_sends_the_answers_held_for_the_sync() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let store = grouped(dir.path());
    // A batch ends once an item overruns the tick slice, so a slow insert
    // lets a tick in before the stop: try until the pair shares a batch.
    let mut shared = false;
    for _ in 0..50 {
        let runtime = store.start_runtime(docs(), NodeRole::Core, false).unwrap();
        shared = rt.block_on(async {
            let (session, mut stream) = thin_docs_session(&runtime).await;
            // The statements are prepared: the next insert is quick.
            send_thin(&runtime, session, 1, insert_doc()).await;
            assert!(
                next_thin(&mut stream, Duration::from_secs(5))
                    .await
                    .is_some(),
                "the first insert is answered"
            );
            let release = hold(&runtime);
            let ticks = Arc::clone(&store.inner.ticks);
            let (ticks_before, before) = std_mpsc::channel();
            runtime
                .send(Box::new(move |_| {
                    let _ = ticks_before.send(ticks.load(Ordering::Relaxed));
                }))
                .unwrap();
            let room = runtime
                .admit_thin_request(session, 1024, false)
                .await
                .unwrap();
            let request = jazz::thin::ThinRequest {
                id: 2,
                op: insert_doc(),
            };
            runtime
                .receive_thin_request(session, request, room)
                .unwrap();
            let stopping = {
                let store = store.clone();
                std::thread::spawn(move || store.stop_runtime())
            };
            // All three queue behind the hold.
            tokio::time::sleep(Duration::from_millis(20)).await;
            release.send(()).unwrap();
            stopping.join().unwrap().unwrap();
            assert!(
                matches!(
                    next_thin(&mut stream, Duration::from_secs(5)).await,
                    Some((2, jazz::thin::ThinBody::Written { .. }))
                ),
                "the committed insert is answered"
            );
            before.recv().unwrap() == store.inner.ticks.load(Ordering::Relaxed)
        });
        if shared {
            break;
        }
    }
    assert!(shared, "the write and the stop never shared a batch");
    store.close().unwrap();
}

/// Gate 14: with a chunk store on the network, its runner is offered work
/// (leaves to put, reclaimed leaves to delete from the bucket) only after
/// the tick's WAL sync: what it deletes, a durable commit reclaimed.
#[test]
fn chunk_store_work_is_handed_out_after_the_ticks_wal_sync() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let bucket = crate::s3::mock::MockS3::start();
    let dir = tempfile::tempdir().unwrap();
    let store = AppStore::open(
        &StorageConfig::data_dir(dir.path()).with_blobs(bucket_blobs(&bucket, "app/", "me")),
    )
    .unwrap();
    let runtime = store.start_runtime(docs(), NodeRole::Core, false).unwrap();
    rt.block_on(async {
        let (session, mut stream) = thin_docs_session(&runtime).await;
        for id in 1..=3 {
            send_thin(&runtime, session, id, insert_doc()).await;
            assert!(
                matches!(
                    next_thin(&mut stream, Duration::from_secs(5)).await,
                    Some((answered, jazz::thin::ThinBody::Written { .. })) if answered == id
                ),
                "insert {id} is answered"
            );
        }
        let trace = trace(&runtime);
        assert!(
            trace.contains(&Trace::ChunkIo),
            "the runner was offered work: {trace:?}"
        );
        assert_no_unsynced_output(&runtime);
        assert_sent_after_the_ticks_sync(&trace);
        assert_every_sync_reached_the_disk(&runtime);
    });
    store.close().unwrap();
}

/// A seeded xorshift: the differential's ops, reproducible by seed.
struct Rng(u64);

impl Rng {
    fn below(&mut self, n: u64) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 % n
    }
}

/// Every body a thin query's rows hold.
fn bodies(body: jazz::thin::ThinBody) -> BTreeSet<Vec<u8>> {
    let jazz::thin::ThinBody::Rows { rows, .. } = body else {
        panic!("expected rows, got {body:?}");
    };
    jazz::binding_codec::decode_rows(&rows)
        .unwrap()
        .iter()
        .flat_map(|chunk| chunk.rows.iter())
        .map(|row| match row.cells.get("body") {
            Some(Value::Bytes(bytes)) => bytes.clone(),
            other => panic!("expected a body, got {other:?}"),
        })
        .collect()
}

/// No cells, encoded: an upload's row has only the uploaded column.
fn no_cells() -> Vec<u8> {
    let descriptor =
        jazz::records::RecordDescriptor::new([] as [(&str, jazz::records::ValueType); 0]);
    let record = jazz::records::OwnedRecord::new(descriptor.create(&[]).unwrap(), descriptor);
    jazz::binding_codec::encode_named_cells(&record).unwrap()
}

fn insert_body(body: &[u8], transaction: Option<&str>) -> jazz::thin::ThinOp {
    let descriptor =
        jazz::records::RecordDescriptor::new([("body", jazz::records::ValueType::Bytes)]);
    let record = jazz::records::OwnedRecord::new(
        descriptor.create(&[Value::Bytes(body.to_vec())]).unwrap(),
        descriptor,
    );
    jazz::thin::ThinOp::Insert {
        table: "docs".to_owned(),
        cells: jazz::binding_codec::encode_named_cells(&record).unwrap(),
        options: Vec::new(),
        transaction: transaction.map(str::to_owned),
    }
}

/// What a thin request expects back.
#[derive(Debug)]
enum Expect {
    /// A transaction's begin or an upload's begin or chunk (`Ok`), or an
    /// insert staged in a transaction (`Staged`).
    Step,
    /// A write of these bodies (an insert, or a transaction's commit):
    /// `Written`, after which they are acked.
    Written(Vec<Vec<u8>>),
    /// A query: rows holding at least every body acked before it was sent.
    Rows(BTreeSet<Vec<u8>>),
}

/// The model's state: what each thin session waits for, in order, and
/// which bodies are acked (their writer was told they committed) or of
/// unknown fate (a reply that failed with its sync).
#[derive(Default)]
struct Model {
    waiting: Vec<std::collections::VecDeque<(u64, Expect)>>,
    next_id: Vec<u64>,
    /// Each session's open transaction and the bodies staged in it.
    open: Vec<Option<(String, Vec<Vec<u8>>)>>,
    /// Each session's open large-value upload: its id, its body, and
    /// whether the body was pushed.
    streams: Vec<Option<(u64, Vec<u8>, bool)>>,
    acked: BTreeSet<Vec<u8>>,
    unknown: BTreeSet<Vec<u8>>,
    host_acked: BTreeSet<Vec<u8>>,
    injected: u64,
}

impl Model {
    /// Check one thin answer against the head of `session`'s queue: the
    /// answers of a session arrive once each, in request order.
    fn answered(&mut self, session: usize, id: u64, body: jazz::thin::ThinBody) {
        let (want, expect) = self.waiting[session]
            .pop_front()
            .unwrap_or_else(|| panic!("session {session}: unexpected answer {id}: {body:?}"));
        assert_eq!(
            id, want,
            "session {session}: answers arrive in request order"
        );
        match expect {
            Expect::Step => assert!(
                matches!(
                    body,
                    jazz::thin::ThinBody::Ok | jazz::thin::ThinBody::Staged { .. }
                ),
                "session {session}: transaction step {id} answered {body:?}"
            ),
            Expect::Written(written) => {
                assert!(
                    matches!(body, jazz::thin::ThinBody::Written { .. }),
                    "session {session}: write {id} answered {body:?}"
                );
                self.acked.extend(written);
            }
            Expect::Rows(before) => {
                let seen = bodies(body);
                assert!(
                    before.is_subset(&seen),
                    "session {session}: query {id} misses acked bodies {:?}",
                    before.difference(&seen).collect::<Vec<_>>()
                );
            }
        }
    }
}

/// Gate 11: a randomized differential. Three thin sessions (plain writes,
/// queries, mergeable and serializable transactions and large-value
/// uploads left open across the other work), runtime jobs
/// that reply (writing or not), host writes, batches held into one tick,
/// and injected sync failures interleave by seed; the model says what the
/// writers must see: each session's answers once, in order; a query sees
/// every body acked before it; no output (frame or reply) leaves while a
/// row change is unsynced; every traced sync is a real fdatasync; each
/// injected failure fail-stops once; and after a reopen the store holds
/// every acked body and nothing but acked or unknown ones.
#[test]
fn group_commit_differential() {
    let seeds: Vec<u64> = match std::env::var("GROUP_COMMIT_SEEDS") {
        Ok(seeds) => seeds
            .split(',')
            .filter(|seed| !seed.is_empty())
            .map(|seed| seed.parse().unwrap())
            .collect(),
        Err(_) => (1..=6).collect(),
    };
    for seed in seeds {
        differential_run(seed, 60);
    }
}

fn differential_run(seed: u64, steps: usize) {
    const SESSIONS: usize = 3;
    let rt = tokio::runtime::Runtime::new().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let store = grouped(dir.path());
    let runtime = store.start_runtime(docs(), NodeRole::Core, false).unwrap();
    let host = HostSender {
        host: store.inner.host.clone(),
        wake: store.sender().unwrap(),
    };
    const KV: HostTable = HostTable::new("gc_differential");
    let stops = Arc::new(AtomicU64::new(0));
    {
        let counted = Arc::clone(&stops);
        block_on(runtime.run(move |owner| {
            owner.wal.as_mut().expect("group commit is on").stops = Some(counted);
            Ok(())
        }))
        .unwrap();
    }
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let mut model = Model {
        waiting: (0..SESSIONS).map(|_| Default::default()).collect(),
        next_id: vec![1; SESSIONS],
        open: vec![None; SESSIONS],
        streams: vec![None; SESSIONS],
        ..Default::default()
    };
    let mut transactions = 0u32;
    let mut uploads = 0u64;
    let mut fresh = 0u64;
    let mut body = || {
        fresh += 1;
        format!("{seed}-{fresh}").into_bytes()
    };
    rt.block_on(async {
        let mut sessions = Vec::new();
        for _ in 0..SESSIONS {
            sessions.push(thin_docs_session(&runtime).await);
        }
        // A thin request on session `k`.
        async fn thin(
            runtime: &ServerRuntimeHandle,
            sessions: &[(ServerSession, ServerRuntimeFrameStream)],
            model: &mut Model,
            k: usize,
            op: jazz::thin::ThinOp,
            expect: Expect,
        ) {
            let id = model.next_id[k];
            model.next_id[k] += 1;
            model.waiting[k].push_back((id, expect));
            send_thin(runtime, sessions[k].0, id, op).await;
        }
        // A runtime job that replies, writing `written` if given.
        fn reply_job(
            runtime: &ServerRuntimeHandle,
            written: Option<Vec<u8>>,
        ) -> tokio::task::JoinHandle<(Option<Vec<u8>>, Result<(), String>)> {
            let runtime = runtime.clone();
            tokio::task::spawn_blocking(move || {
                let row = written.clone();
                let result = block_on(runtime.run(move |owner| {
                    if let Some(row) = row {
                        block_on(owner.db.insert(
                            "docs",
                            BTreeMap::from([("body".to_owned(), Value::Bytes(row))]),
                            Default::default(),
                        ))
                        .map_err(|error| error.to_string())?;
                    }
                    Ok(())
                }));
                (written, result)
            })
        }
        for _ in 0..steps {
            let mut jobs = Vec::new();
            let mut hosts = Vec::new();
            let mut injections = Vec::new();
            let held = rng.below(4) == 0;
            let release = held.then(|| hold(&runtime));
            let ops = if held { 2 + rng.below(4) } else { 1 };
            for _ in 0..ops {
                match rng.below(14) {
                    0..=3 => {
                        let k = rng.below(SESSIONS as u64) as usize;
                        let row = body();
                        thin(
                            &runtime,
                            &sessions,
                            &mut model,
                            k,
                            insert_body(&row, None),
                            Expect::Written(vec![row]),
                        )
                        .await;
                    }
                    10..=11 => {
                        // A transaction's next step: begin, stage, commit.
                        let k = rng.below(SESSIONS as u64) as usize;
                        let (op, expect) = match model.open[k].take() {
                            None => {
                                transactions += 1;
                                let id = format!("0190f3e40000700080000000{transactions:08x}");
                                model.open[k] = Some((id.clone(), Vec::new()));
                                let kind = ["mergeable", "serializable"][transactions as usize % 2];
                                let op = jazz::thin::ThinOp::Begin {
                                    transaction: id,
                                    kind: kind.to_owned(),
                                };
                                (op, Expect::Step)
                            }
                            Some((id, mut staged)) if staged.len() < 2 && rng.below(2) == 0 => {
                                let row = body();
                                let op = insert_body(&row, Some(&id));
                                staged.push(row);
                                model.open[k] = Some((id, staged));
                                (op, Expect::Step)
                            }
                            Some((id, staged)) => (
                                jazz::thin::ThinOp::Commit { transaction: id },
                                Expect::Written(staged),
                            ),
                        };
                        thin(&runtime, &sessions, &mut model, k, op, expect).await;
                    }
                    4..=5 => {
                        let k = rng.below(SESSIONS as u64) as usize;
                        let before = model.acked.clone();
                        thin(
                            &runtime,
                            &sessions,
                            &mut model,
                            k,
                            thin_query(None),
                            Expect::Rows(before),
                        )
                        .await;
                    }
                    12..=13 => {
                        // An upload's next step: begin, push, finish.
                        let k = rng.below(SESSIONS as u64) as usize;
                        let (op, expect) = match model.streams[k].take() {
                            None => {
                                model.streams[k] = Some((model.next_id[k], body(), false));
                                uploads += 1;
                                let mut row = [0; 16];
                                row[..8].copy_from_slice(&seed.to_be_bytes());
                                row[8..].copy_from_slice(&uploads.to_be_bytes());
                                let op = jazz::thin::ThinOp::BeginStream {
                                    table: "docs".to_owned(),
                                    row,
                                    cells: no_cells(),
                                    column: "body".to_owned(),
                                    mutation: "insert".to_owned(),
                                    options: Vec::new(),
                                    transaction: None,
                                };
                                (op, Expect::Step)
                            }
                            Some((stream, row, false)) => {
                                let op = jazz::thin::ThinOp::StreamChunk {
                                    stream,
                                    chunk: row.clone(),
                                };
                                model.streams[k] = Some((stream, row, true));
                                (op, Expect::Step)
                            }
                            Some((stream, row, true)) => (
                                jazz::thin::ThinOp::FinishStream { stream },
                                Expect::Written(vec![row]),
                            ),
                        };
                        thin(&runtime, &sessions, &mut model, k, op, expect).await;
                    }
                    6 => jobs.push(reply_job(&runtime, Some(body()))),
                    7 => jobs.push(reply_job(&runtime, None)),
                    8 => {
                        let key = body();
                        let (tx, rx) = std_mpsc::channel();
                        let pair = (key.clone(), b"v".to_vec());
                        host.send(Box::new(move |tables: &dyn HostTables| {
                            let _ = tx.send(tables.host_put_many(KV, &[pair]).is_ok());
                        }))
                        .unwrap();
                        hosts.push((key, rx));
                    }
                    _ => {
                        // A failure still pending from before is the same one.
                        let (tx, rx) = std_mpsc::channel();
                        runtime
                            .send(Box::new(move |owner| {
                                let wal = owner.wal.as_mut().expect("group commit is on");
                                let _ = tx.send(!std::mem::replace(&mut wal.fail_next, true));
                            }))
                            .unwrap();
                        injections.push(rx);
                    }
                }
            }
            // Jobs queue behind the hold before it lifts.
            if let Some(release) = release {
                tokio::time::sleep(Duration::from_millis(5)).await;
                release.send(()).unwrap();
            }
            for rx in injections {
                model.injected += u64::from(rx.recv_timeout(Duration::from_secs(10)).unwrap());
            }
            // Host work runs between runtime items, so after the hold.
            for (key, rx) in hosts {
                assert!(rx.recv_timeout(Duration::from_secs(10)).unwrap());
                model.host_acked.insert(key);
            }
            for job in jobs {
                let (written, result) = job.await.unwrap();
                match (written, result) {
                    (Some(row), Ok(())) => {
                        model.acked.insert(row);
                    }
                    // Its sync failed: committed at FULL, but never acked.
                    (Some(row), Err(_)) => {
                        model.unknown.insert(row);
                    }
                    (None, _) => {}
                }
            }
            // Every answer due arrives; a tick after a failed one sends the
            // held ones.
            let deadline = Instant::now() + Duration::from_secs(10);
            for (k, (session, stream)) in sessions.iter_mut().enumerate() {
                while !model.waiting[k].is_empty() {
                    assert!(
                        Instant::now() < deadline,
                        "seed {seed}: session {k} never answered"
                    );
                    runtime.request_wire_tick(*session).unwrap();
                    if let Some(batch) = next_thin_batch(stream, Duration::from_millis(200)).await {
                        for (id, answer) in batch {
                            model.answered(k, id, answer);
                        }
                    }
                }
            }
        }
        // An injection no sync met (nothing was committed after it) did
        // not fail one.
        let pending = block_on(runtime.run(|owner| {
            let wal = owner.wal.as_mut().expect("group commit is on");
            Ok(std::mem::take(&mut wal.fail_next))
        }))
        .unwrap();
        model.injected -= u64::from(pending);
        assert_no_unsynced_output(&runtime);
        let trace = trace(&runtime);
        assert_sent_after_the_ticks_sync(&trace);
        assert_every_sync_reached_the_disk(&runtime);
        assert_eq!(
            stops.load(Ordering::SeqCst),
            model.injected,
            "seed {seed}: each injected failure fail-stopped once"
        );
    });
    store.close().unwrap();
    // What the writers were told is in the store, and nothing else.
    let store = grouped(dir.path());
    let runtime = store.start_runtime(docs(), NodeRole::Core, false).unwrap();
    let stored: BTreeSet<Vec<u8>> = block_on(runtime.run(|owner| {
        let rows = block_on(
            owner.db.all(
                &owner
                    .db
                    .prepare_query(&jazz::query::Query::from("docs"))
                    .unwrap(),
                jazz::db::ReadOpts {
                    tier: jazz::db::ReadTier::LocalOnly,
                    ..Default::default()
                },
            ),
        )
        .map_err(|error| error.to_string())?;
        Ok(rows
            .iter()
            .filter_map(|row| match row.cells().find(|(name, _)| *name == "body") {
                Some((_, Some(Value::Bytes(bytes)))) => Some(bytes.clone()),
                _ => None,
            })
            .collect())
    }))
    .unwrap();
    assert!(
        model.acked.is_subset(&stored),
        "seed {seed}: acked bodies lost: {:?}",
        model.acked.difference(&stored).collect::<Vec<_>>()
    );
    let known: BTreeSet<_> = model.acked.union(&model.unknown).cloned().collect();
    assert!(
        stored.is_subset(&known),
        "seed {seed}: bodies nobody wrote: {:?}",
        stored.difference(&known).collect::<Vec<_>>()
    );
    let (tx, rx) = std_mpsc::channel();
    HostSender {
        host: store.inner.host.clone(),
        wake: store.sender().unwrap(),
    }
    .send(Box::new(move |tables: &dyn HostTables| {
        let _ = tx.send(tables.host_prefix(KV, b"").unwrap());
    }))
    .unwrap();
    let keys: BTreeSet<Vec<u8>> = rx.recv().unwrap().into_iter().map(|(key, _)| key).collect();
    assert_eq!(
        keys, model.host_acked,
        "seed {seed}: every acked host write"
    );
    store.close().unwrap();
}

fn docs() -> JazzSchema {
    super::tests::docs()
}

fn insert_doc() -> jazz::thin::ThinOp {
    jazz::thin::ThinOp::Insert {
        table: "docs".to_owned(),
        cells: thin_cells(),
        options: Vec::new(),
        transaction: None,
    }
}

use super::tests::{
    bucket_blobs, hold, next_thin, next_thin_batch, send_thin, thin_docs_session, thin_query,
};
use std::collections::BTreeSet;

fn thin_cells() -> Vec<u8> {
    let descriptor =
        jazz::records::RecordDescriptor::new([("body", jazz::records::ValueType::Bytes)]);
    let record = jazz::records::OwnedRecord::new(
        descriptor.create(&[Value::Bytes(b"x".to_vec())]).unwrap(),
        descriptor,
    );
    jazz::binding_codec::encode_named_cells(&record).unwrap()
}
