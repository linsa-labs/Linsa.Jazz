//! Group commit ([`StoreOptions::group_commit`]) on a file store: the
//! runtime commits at `synchronous = NORMAL` and syncs the WAL once per
//! tick, before any of that tick's output leaves; host writes and runtime
//! jobs that reply from inside commit at `FULL`; a failed sync fail-stops.
//! The owner records each tick's steps in order ([`Trace`]), which these
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

const GROUP: StoreOptions = StoreOptions { group_commit: true };

/// A file store in `dir`, group commit on.
fn grouped(dir: &std::path::Path) -> AppStore {
    AppStore::open_with(&StorageConfig::data_dir(dir), GROUP).unwrap()
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

/// Every send happened inside a tick, after that tick's WAL sync.
#[track_caller]
fn assert_sent_after_the_ticks_sync(trace: &[Trace]) {
    let (mut in_tick, mut synced) = (false, false);
    for (at, step) in trace.iter().enumerate() {
        match step {
            Trace::Tick => (in_tick, synced) = (true, false),
            Trace::Synced(_) => synced = true,
            Trace::FailStop | Trace::TickEnd => (in_tick, synced) = (false, false),
            Trace::Sent(_) | Trace::ThinSent(_) => assert!(
                in_tick && synced,
                "{step:?} left before its tick's WAL sync (step {at}): {:?}",
                &trace[at.saturating_sub(6)..(at + 2).min(trace.len())]
            ),
        }
    }
}

/// Every traced `Synced` step was a real fdatasync (a sync that returns Ok
/// without writing would leave the count short), and opening the runtime
/// synced once more.
#[track_caller]
fn assert_every_sync_reached_the_disk(runtime: &ServerRuntimeHandle) {
    let (trace, syncs) = block_on(runtime.run(|owner| {
        Ok((
            owner.trace.clone(),
            owner.wal.as_ref().map(|wal| wal.total_syncs),
        ))
    }))
    .unwrap();
    // The k-th traced sync is the WAL's (k+1)-th fdatasync (the first is
    // the open's): a sync moved after the sends, its trace left in place,
    // would carry the count from before it.
    let stamps: Vec<u64> = trace
        .iter()
        .filter_map(|step| match step {
            Trace::Synced(count) => Some(*count),
            _ => None,
        })
        .collect();
    let expected: Vec<u64> = (2..stamps.len() as u64 + 2).collect();
    assert_eq!(
        stamps, expected,
        "each traced sync is one fdatasync more: {trace:?}"
    );
    assert_eq!(
        syncs,
        Some(stamps.len() as u64 + 1),
        "one fdatasync at open plus one per traced sync: {trace:?}"
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
    });
    store.close().unwrap();
}

/// Gate 6: with the option off the store commits at FULL everywhere and
/// syncs nothing per tick; an in-memory store ignores the option. The env
/// turns it on unless `JAZZ_GROUP_COMMIT=0` or `JAZZ_X_ALL=0`.
#[test]
fn group_commit_off_commits_at_full() {
    let off = |var: &str| std::env::var(var).as_deref() == Ok("0");
    assert_eq!(
        StoreOptions::from_env().group_commit,
        !(off("JAZZ_GROUP_COMMIT") || off("JAZZ_X_ALL")),
        "group commit is on unless the env turns it off"
    );
    let dir = tempfile::tempdir().unwrap();
    let store = AppStore::open_with(
        &StorageConfig::data_dir(dir.path()),
        StoreOptions { group_commit: false },
    )
    .unwrap();
    assert_eq!(host_synchronous(&store), FULL);
    let runtime = store.start_runtime(todos(), NodeRole::Core, false).unwrap();
    assert_eq!(
        runtime_synchronous(&runtime),
        FULL,
        "the runtime commits at FULL"
    );
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
    block_on(runtime.run(|_| Ok(()))).unwrap();
    let trace = trace(&runtime);
    assert!(
        block_on(runtime.run(|owner| Ok(owner.wal.is_none()))).unwrap()
            && trace.contains(&Trace::Tick)
            && !trace.iter().any(|step| matches!(step, Trace::Synced(_))),
        "no WAL sync per tick: {trace:?}"
    );
    store.close().unwrap();
    let memory = AppStore::open_with(&StorageConfig::InMemory, GROUP).unwrap();
    let runtime = memory
        .start_runtime(todos(), NodeRole::Core, false)
        .unwrap();
    assert!(
        block_on(runtime.run(|owner| Ok(owner.wal.is_none()))).unwrap(),
        "an in-memory store has no WAL to sync"
    );
    memory.close().unwrap();
}

/// Gate 7: what opening the runtime wrote (migrations, admissions, the
/// authority mint) is synced before `start_runtime` returns: one fdatasync
/// more than the ticks traced.
#[test]
fn opening_the_runtime_syncs_the_wal_before_it_serves() {
    let dir = tempfile::tempdir().unwrap();
    let store = grouped(dir.path());
    let runtime = store.start_runtime(todos(), NodeRole::Core, false).unwrap();
    assert_every_sync_reached_the_disk(&runtime);
    store.close().unwrap();
}

/// Gate 8: outside the tests' counting hook a failed WAL sync aborts the
/// process (a logged failure would let the next sync succeed and the held
/// answers go out). Run in a child copy of this test binary.
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
        // The job is followed by a tick, whose sync fails.
        block_on(runtime.run(|_| Ok(()))).unwrap();
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

use super::tests::{hold, next_thin, send_thin, thin_docs_session};

fn thin_cells() -> Vec<u8> {
    let descriptor =
        jazz::records::RecordDescriptor::new([("body", jazz::records::ValueType::Bytes)]);
    let record = jazz::records::OwnedRecord::new(
        descriptor.create(&[Value::Bytes(b"x".to_vec())]).unwrap(),
        descriptor,
    );
    jazz::binding_codec::encode_named_cells(&record).unwrap()
}
