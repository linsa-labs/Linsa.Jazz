//! The write path of the SQLite store: what a flush leaves where and on which thread,
//! what bounds the log when the checkpointer cannot keep up, how the store closes, and —
//! with the checkpointer racing the writer throughout — that the store holds what a
//! model of it holds, in the process and in the files a dead process leaves behind.

use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant};

use super::*;
use crate::query_manager::types::Value;

const TABLE: &str = "family";

fn probe(storage: &SqliteStorage) -> Arc<CheckpointerProbe> {
    storage
        .with_inner(|inner| {
            Ok(Arc::clone(
                &inner
                    .checkpointer
                    .as_ref()
                    .expect("a store on a file has a checkpointer")
                    .probe,
            ))
        })
        .unwrap()
}

/// Checkpoints the store has run on its caller's thread.
fn moved_by_the_writer(storage: &SqliteStorage) -> u64 {
    storage
        .with_inner(|inner| Ok(inner.checkpoints.get()))
        .unwrap()
}

fn pragma(storage: &SqliteStorage, name: &str) -> i64 {
    storage
        .with_inner(|inner| {
            Ok(inner
                .conn
                .query_row(&format!("PRAGMA {name}"), [], |row| row.get::<_, i64>(0))
                .unwrap())
        })
        .unwrap()
}

fn log_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push("-wal");
    PathBuf::from(name)
}

/// The files a process that died now would leave: the database file and, if asked, the
/// log beside it. Never the wal-index, which does not survive a process.
fn files_left_behind(path: &Path, with_log: bool) -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::TempDir::new().unwrap();
    let copy = dir.path().join("copy.sqlite");
    std::fs::copy(path, &copy).unwrap();
    if with_log && log_path(path).exists() {
        std::fs::copy(log_path(path), log_path(&copy)).unwrap();
    }
    (dir, copy)
}

/// Whether `key` is in the files a process that died now would leave. `None` if the
/// database file was copied while the checkpointer was writing it and does not read.
fn left_behind(path: &Path, with_log: bool, key: &str) -> Option<bool> {
    let (_dir, copy) = files_left_behind(path, with_log);
    let conn = rusqlite::Connection::open(&copy).ok()?;
    let storage_key = key_codec::raw_table_entry_key(TABLE, key);
    conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM kv WHERE key = ?1)",
        rusqlite::params![storage_key.as_bytes()],
        |row| row.get::<_, i64>(0),
    )
    .ok()
    .map(|found| found != 0)
}

/// Waits for a checkpoint that moved `key` into the database file to have finished: the
/// pages show in the file before the pass has synced it, recorded how far it got and
/// been counted, and the log is only rewound once it has.
fn wait_until_moved(probe: &CheckpointerProbe, path: &Path, key: &str) {
    let started = Instant::now();
    loop {
        let passes = probe.passes.load(Ordering::Relaxed);
        if left_behind(path, false, key) == Some(true) {
            // The pass that moved it, or one that began after it was moved, has ended.
            while probe.passes.load(Ordering::Relaxed) == passes
                && probe.busy.load(Ordering::Relaxed)
            {
                std::thread::sleep(Duration::from_millis(1));
            }
            return;
        }
        assert!(
            started.elapsed() < Duration::from_secs(20),
            "the checkpointer never moved {key} into the database file"
        );
        std::thread::sleep(Duration::from_millis(2));
    }
}

/// Waits until the checkpointer has ended its `passes`-th pass.
fn wait_for_passes(probe: &CheckpointerProbe, passes: u64) {
    let started = Instant::now();
    while probe.passes.load(Ordering::Relaxed) < passes || probe.busy.load(Ordering::Relaxed) {
        assert!(
            started.elapsed() < Duration::from_secs(20),
            "the checkpointer made {} of {passes} passes",
            probe.passes.load(Ordering::Relaxed)
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// A closed store of a few megabytes at `path`, and the reads a read-through of it takes.
fn a_store_worth_reading_through(path: &Path) -> u64 {
    let mut storage = SqliteStorage::open(path).unwrap();
    for index in 0..40 {
        storage
            .raw_table_put(TABLE, &format!("main:{index:03}"), &[index as u8; 100_000])
            .unwrap();
    }
    storage.flush().unwrap();
    storage.close().unwrap();
    std::fs::metadata(path).unwrap().len() / READ_THROUGH_CHUNK_BYTES as u64 + 1
}

/// Opens the store at `path` to be read through, with the read held up after its first
/// chunk until the probe lets it go.
fn open_with_the_read_through_held(path: &Path) -> (SqliteStorage, Arc<CheckpointerProbe>) {
    let held = Arc::new(CheckpointerProbe::default());
    held.reading_held.store(true, Ordering::Relaxed);
    let storage = SqliteStorage::open_checkpointed(path, WAL_VALVE_FRAMES, |conn| {
        Checkpointer::start_probed(conn, path, true, Arc::clone(&held))
    })
    .unwrap();
    (storage, held)
}

fn open(dir: &tempfile::TempDir) -> (SqliteStorage, PathBuf) {
    let path = dir.path().join("test.sqlite");
    (SqliteStorage::open(&path).unwrap(), path)
}

/// The settings the write path rests on. Each is a measured cost when it is missing, and
/// one of them is what a flush's promise rests on.
#[test]
fn a_store_on_a_file_opens_with_its_write_path_settings() {
    let dir = tempfile::TempDir::new().unwrap();
    let (storage, _path) = open(&dir);

    assert_eq!(
        pragma(&storage, "temp_store"),
        2,
        "statement journals in memory"
    );
    assert!(
        storage
            .with_inner(|inner| Ok(inner.checkpointer.is_some()))
            .unwrap()
    );
    assert_eq!(
        pragma(&storage, "synchronous"),
        2,
        "with the checkpoint off the flush, the commit is what syncs the log"
    );
    assert_eq!(
        storage
            .with_inner(|inner| Ok(inner.log_valve_frames))
            .unwrap(),
        WAL_VALVE_FRAMES
    );
    assert_eq!(
        pragma(&storage, "journal_size_limit"),
        i64::from(WAL_VALVE_FRAMES) * WAL_FRAME_BYTES
    );
    assert_eq!(
        pragma(&storage, "wal_autocheckpoint"),
        0,
        "SQLite's own checkpoint-every-1000-frames is still on the writer's thread"
    );
}

/// A store that opens reads its database file through once, off the opener's thread and
/// through the connection the checkpointer already has: the whole file when it is under
/// the bound, 64 KiB a read, and the read that finds the end. A store opened plainly is
/// not read through at all.
#[test]
fn a_store_that_opens_reads_its_database_file_through() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("test.sqlite");
    assert!(a_store_worth_reading_through(&path) > 3);

    // Only a store opened to be read through is.
    let mut storage = SqliteStorage::open(&path).unwrap();
    let unread = probe(&storage);
    storage
        .raw_table_put(TABLE, "main:after", &[1; 64])
        .unwrap();
    storage.flush_wal().unwrap();
    wait_until_moved(&unread, &path, "main:after");
    assert_eq!(unread.read_through.load(Ordering::Relaxed), 0);
    storage.close().unwrap();
    let bytes = std::fs::metadata(&path).unwrap().len();

    let storage = SqliteStorage::open_read_through(&path).unwrap();
    let read = probe(&storage);
    let started = Instant::now();
    while read.read_through.load(Ordering::Relaxed) == 0 {
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "never read through"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(
        read.read_through.load(Ordering::Relaxed),
        bytes / READ_THROUGH_CHUNK_BYTES as u64 + 1,
        "a {bytes}-byte file"
    );
    // The store is whole and the checkpointer serves it afterwards.
    assert_eq!(
        storage.raw_table_get(TABLE, "main:039").unwrap(),
        Some(vec![39; 100_000])
    );
}

/// A store closed while its file is still being read through does not wait for the
/// read: the read ends at its next chunk.
#[test]
fn closing_the_store_ends_the_read_through() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("test.sqlite");
    let chunks = a_store_worth_reading_through(&path);

    // Asked to stop before it starts, the read issues nothing.
    let conn = rusqlite::Connection::open(&path).unwrap();
    assert_eq!(read_through(&conn, &AtomicBool::new(true), || {}), 0);
    let mut between = 0;
    assert_eq!(
        read_through(&conn, &AtomicBool::new(false), || between += 1),
        chunks
    );
    // The last read is the short one that found the end of the file.
    assert_eq!(between, chunks - 1);
    // A connection with no file has nothing to read.
    let memory = rusqlite::Connection::open_in_memory().unwrap();
    assert!(read_through(&memory, &AtomicBool::new(false), || {}) <= 1);
}

/// A store that writes as it opens flushes while its file is still being read. The log
/// that flush leaves is moved between two reads, not after the last one: the read is a
/// quarter of a gigabyte long on a large store, and a log nobody moves until it ends is
/// a log past its bound, moved by the writer on its own thread.
#[test]
fn a_flush_during_the_read_through_is_served_between_two_reads() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("test.sqlite");
    a_store_worth_reading_through(&path);
    let (mut storage, held) = open_with_the_read_through_held(&path);

    storage
        .raw_table_put(TABLE, "main:while-reading", &[7; 2048])
        .unwrap();
    storage.flush_wal().unwrap();
    // One more read, and the thread is held again: what moves the log is whatever it
    // does between the two.
    held.reading_step.store(true, Ordering::Relaxed);
    wait_until_moved(&held, &path, "main:while-reading");
    assert_eq!(
        held.read_through.load(Ordering::Relaxed),
        0,
        "the read had ended by the time the log was moved"
    );
    assert!(held.passes.load(Ordering::Relaxed) >= 1);

    held.reading_held.store(false, Ordering::Relaxed);
    let started = Instant::now();
    while held.read_through.load(Ordering::Relaxed) == 0 {
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "the read never ended"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    storage.close().unwrap();
}

/// The store's own close, with the read under way: it returns, the thread is gone, and
/// the read stopped where it was.
#[test]
fn closing_the_store_in_the_middle_of_the_read_through_does_not_wait_for_its_end() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("test.sqlite");
    let whole = a_store_worth_reading_through(&path);
    let (storage, held) = open_with_the_read_through_held(&path);

    // Held between two reads: the first is behind it.
    let started = Instant::now();
    while !held.reading_parked.load(Ordering::Relaxed) {
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "the read-through never began"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    // A checkpoint asked for and not yet made is the closing connection's to make.
    storage.flush_wal().unwrap();
    // A close that waited for the hold to be lifted would never return, so it runs where
    // it can be given up on.
    let (closed, close_returned) = std::sync::mpsc::channel();
    std::thread::spawn(move || closed.send(storage.close()));
    close_returned
        .recv_timeout(Duration::from_secs(10))
        .expect("close() waited for the read-through")
        .unwrap();
    assert!(held.exited.load(Ordering::Relaxed));
    let read = held.read_through.load(Ordering::Relaxed);
    assert!(
        read >= 1 && read < whole,
        "{read} of {whole} reads were issued"
    );
    assert_eq!(
        held.passes.load(Ordering::Relaxed),
        0,
        "the thread checkpointed after it was told to stop"
    );
}

/// A thread that ended without being stopped is found out by the flush that tries to
/// wake it: that flush moves the log itself, and so does every one after it.
#[test]
fn a_flush_that_finds_the_checkpointer_gone_checkpoints_itself() {
    let dir = tempfile::TempDir::new().unwrap();
    let (mut storage, path) = open(&dir);
    let probe = probe(&storage);

    // The thread leaves on its next wake, as it would after a panic: nobody stopped it,
    // and the store still holds the sending end.
    storage
        .with_inner(|inner| {
            let checkpointer = inner.checkpointer.as_ref().unwrap();
            checkpointer.stopping.store(true, Ordering::Relaxed);
            checkpointer.wake.as_ref().unwrap().send(()).unwrap();
            Ok(())
        })
        .unwrap();
    let started = Instant::now();
    while !probe.exited.load(Ordering::Relaxed) {
        assert!(started.elapsed() < Duration::from_secs(10));
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(
        storage
            .with_inner(|inner| Ok(inner.checkpointer.as_ref().unwrap().wake.is_some()))
            .unwrap()
    );

    for key in ["main:found-gone", "main:known-gone"] {
        storage.raw_table_put(TABLE, key, &[7; 2048]).unwrap();
        storage.flush_wal().unwrap();
        assert_eq!(left_behind(&path, false, key), Some(true));
        assert!(
            storage
                .with_inner(|inner| Ok(inner.checkpointer.as_ref().unwrap().wake.is_none()))
                .unwrap()
        );
    }
    storage.close().unwrap();
}

/// The log file is cut back to its bound by the first commit after it is rewound: one
/// large delivery does not leave a file of its size on the device for good.
#[test]
fn a_log_that_grew_past_its_bound_is_cut_back_once_it_is_rewound() {
    const VALVE: u32 = 64;
    let bound = u64::from(VALVE) * WAL_FRAME_BYTES as u64;

    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("test.sqlite");
    let mut storage = SqliteStorage::open_with_wal_valve(&path, VALVE).unwrap();
    let probe = probe(&storage);
    let log_bytes = || std::fs::metadata(log_path(&path)).unwrap().len();

    probe.stalled.store(true, Ordering::Relaxed);
    for key in 0..300 {
        storage
            .raw_table_put(TABLE, &format!("main:large-{key}"), &[7; 3000])
            .unwrap();
    }
    storage.flush_wal().unwrap();
    assert!(log_bytes() > 4 * bound, "{} bytes", log_bytes());
    probe.stalled.store(false, Ordering::Relaxed);
    wait_until_moved(&probe, &path, "main:large-299");
    assert!(
        log_bytes() > 4 * bound,
        "moving the log does not shorten it"
    );

    probe.stalled.store(true, Ordering::Relaxed);
    storage
        .raw_table_put(TABLE, "main:small", &[7; 64])
        .unwrap();
    storage.flush_wal().unwrap();
    assert!(log_bytes() <= bound, "{} bytes", log_bytes());

    probe.stalled.store(false, Ordering::Relaxed);
    storage.close().unwrap();
}

/// No file, no log, nothing for a checkpointer to do: the store is as it always was.
#[test]
fn a_store_in_memory_has_no_checkpointer() {
    let mut storage = SqliteStorage::open(":memory:").unwrap();
    assert!(
        storage
            .with_inner(|inner| Ok(inner.checkpointer.is_none()))
            .unwrap()
    );
    assert_eq!(pragma(&storage, "synchronous"), 1);
    assert_eq!(pragma(&storage, "wal_autocheckpoint"), 1000);

    storage.raw_table_put(TABLE, "main:row", b"v").unwrap();
    storage.flush_wal().unwrap();
    storage.flush().unwrap();
    assert_eq!(
        storage.raw_table_get(TABLE, "main:row").unwrap(),
        Some(b"v".to_vec())
    );
    storage.close().unwrap();
}

/// `flush_wal()` leaves the writes in the log, where a dead process leaves them behind,
/// and does not move the log itself; the checkpointer does, afterwards.
#[test]
fn a_flush_of_the_log_commits_and_leaves_the_checkpoint_to_the_checkpointer() {
    let dir = tempfile::TempDir::new().unwrap();
    let (mut storage, path) = open(&dir);
    let probe = probe(&storage);

    probe.stalled.store(true, Ordering::Relaxed);
    storage
        .raw_table_put(TABLE, "main:row", &[7; 2048])
        .unwrap();
    assert_eq!(
        left_behind(&path, true, "main:row"),
        Some(false),
        "an unflushed write is not in the files"
    );
    storage.flush_wal().unwrap();

    assert_eq!(
        left_behind(&path, true, "main:row"),
        Some(true),
        "a flushed write is in the files the process would leave behind"
    );
    // Not even the table is there yet: nothing has moved any of the log.
    assert_ne!(
        left_behind(&path, false, "main:row"),
        Some(true),
        "the flush moved the log into the database file on the writer's thread"
    );

    probe.stalled.store(false, Ordering::Relaxed);
    wait_until_moved(&probe, &path, "main:row");
    assert!(probe.passes.load(Ordering::Relaxed) >= 1);
    storage.close().unwrap();
}

/// `flush()` is the one that does not return before the log is moved.
#[test]
fn a_flush_moves_the_log_before_it_returns() {
    let dir = tempfile::TempDir::new().unwrap();
    let (mut storage, path) = open(&dir);
    let probe = probe(&storage);

    // The checkpointer never gets to run: whatever moves the log is the flush.
    probe.stalled.store(true, Ordering::Relaxed);
    storage
        .raw_table_put(TABLE, "main:row", &[7; 2048])
        .unwrap();
    storage.flush().unwrap();
    assert_eq!(left_behind(&path, false, "main:row"), Some(true));

    probe.stalled.store(false, Ordering::Relaxed);
    storage.close().unwrap();
}

/// A checkpointer that is gone, or whose last checkpoint failed, hands the checkpoint
/// back to the flush — where a failure is the flush's failure.
#[test]
fn a_flush_of_the_log_checkpoints_itself_when_the_checkpointer_cannot() {
    let dir = tempfile::TempDir::new().unwrap();
    let (mut storage, path) = open(&dir);
    let probe = probe(&storage);

    // Its last checkpoint failed: this flush checkpoints, the next one asks it again.
    probe.stalled.store(true, Ordering::Relaxed);
    storage
        .with_inner(|inner| {
            inner
                .checkpointer
                .as_ref()
                .unwrap()
                .failed
                .store(true, Ordering::Relaxed);
            Ok(())
        })
        .unwrap();
    storage
        .raw_table_put(TABLE, "main:failed", &[7; 2048])
        .unwrap();
    storage.flush_wal().unwrap();
    assert_eq!(left_behind(&path, false, "main:failed"), Some(true));

    storage
        .raw_table_put(TABLE, "main:asked", &[7; 2048])
        .unwrap();
    storage.flush_wal().unwrap();
    assert_ne!(
        left_behind(&path, false, "main:asked"),
        Some(true),
        "the failure was reported once; the checkpointer is asked again"
    );
    probe.stalled.store(false, Ordering::Relaxed);
    wait_until_moved(&probe, &path, "main:asked");

    // Gone for good.
    storage
        .with_inner_mut(|inner| {
            inner.checkpointer.as_mut().unwrap().stop();
            Ok(())
        })
        .unwrap();
    assert!(probe.exited.load(Ordering::Relaxed));
    for key in ["main:gone-1", "main:gone-2"] {
        storage.raw_table_put(TABLE, key, &[7; 2048]).unwrap();
        storage.flush_wal().unwrap();
        assert_eq!(left_behind(&path, false, key), Some(true));
    }
    storage.close().unwrap();
}

/// A full flush moved the log itself, whatever its commit found: the flush of the log
/// that follows it has nothing of its own to move and asks the checkpointer, as any
/// other does.
#[test]
fn a_flush_of_the_log_after_a_full_flush_is_left_to_the_checkpointer() {
    const VALVE: u32 = 64;

    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("test.sqlite");
    let mut storage = SqliteStorage::open_with_wal_valve(&path, VALVE).unwrap();
    let probe = probe(&storage);

    // A commit that finds the log past its bound, flushed in full.
    probe.stalled.store(true, Ordering::Relaxed);
    for key in 0..120 {
        storage
            .raw_table_put(TABLE, &format!("main:first-{key}"), &[7; 3000])
            .unwrap();
    }
    storage.flush_wal().unwrap();
    storage
        .raw_table_put(TABLE, "main:second", &[7; 64])
        .unwrap();
    storage.flush().unwrap();
    assert_eq!(left_behind(&path, false, "main:second"), Some(true));
    // The pass the first flush asked for, held up until now, ends: from here on a pass
    // is one somebody asked for after this.
    probe.stalled.store(false, Ordering::Relaxed);
    wait_for_passes(&probe, 1);

    // Nothing written since: the commit reports nothing, and what the last one found
    // must not stand.
    let passes = probe.passes.load(Ordering::Relaxed);
    let by_the_writer = moved_by_the_writer(&storage);
    storage.flush_wal().unwrap();
    assert_eq!(
        moved_by_the_writer(&storage),
        by_the_writer,
        "the flush moved the log itself: what the commit before the full flush found stood"
    );
    let started = Instant::now();
    while probe.passes.load(Ordering::Relaxed) == passes {
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "the flush never asked the checkpointer"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    storage.close().unwrap();
}

/// A flush that finds the log past its bound moves it on its own thread — as far as it
/// is let: with the checkpointer in the middle of a pass, or a reader in the way, that
/// is nowhere, and nothing says so. So such a flush asks the checkpointer as well, or
/// what it could not move stays in the log until somebody writes again.
#[test]
fn a_flush_that_moves_the_log_itself_asks_the_checkpointer_as_well() {
    const VALVE: u32 = 64;

    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("test.sqlite");
    let mut storage = SqliteStorage::open_with_wal_valve(&path, VALVE).unwrap();
    let probe = probe(&storage);
    storage
        .raw_table_put(TABLE, "main:first", &[1; 64])
        .unwrap();
    storage.flush().unwrap();

    // A reader that began before the writing: nobody's checkpoint gets past it.
    let reader = rusqlite::Connection::open(&path).unwrap();
    reader.execute_batch("BEGIN").unwrap();
    reader
        .query_row("SELECT count(*) FROM kv", [], |row| row.get::<_, i64>(0))
        .unwrap();

    // Larger than the bound by itself, into a rewound log: the checkpointer's.
    let before = probe.passes.load(Ordering::Relaxed);
    for key in 0..120 {
        storage
            .raw_table_put(TABLE, &format!("main:large-{key}"), &[7; 3000])
            .unwrap();
    }
    storage.flush_wal().unwrap();
    wait_for_passes(&probe, before + 1);
    assert_eq!(left_behind(&path, false, "main:large-119"), Some(false));

    // Added to a log past its bound: the writer's, and it moves nothing either.
    let before = probe.passes.load(Ordering::Relaxed);
    let by_the_writer = moved_by_the_writer(&storage);
    storage.raw_table_put(TABLE, "main:last", &[2; 64]).unwrap();
    storage.flush_wal().unwrap();
    assert_eq!(
        moved_by_the_writer(&storage),
        by_the_writer + 1,
        "the log was past its bound and the flush did not move it itself"
    );
    let started = Instant::now();
    while probe.passes.load(Ordering::Relaxed) == before {
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "the flush that moved the log itself did not ask the checkpointer"
        );
        std::thread::sleep(Duration::from_millis(1));
    }

    // However the checkpointer's last pass ended: this flush has run its own, so there
    // is no failure left for it to take over.
    wait_for_passes(&probe, before + 1);
    storage
        .with_inner(|inner| {
            inner
                .checkpointer
                .as_ref()
                .unwrap()
                .failed
                .store(true, Ordering::Relaxed);
            Ok(())
        })
        .unwrap();
    let before = probe.passes.load(Ordering::Relaxed);
    let by_the_writer = moved_by_the_writer(&storage);
    storage
        .raw_table_put(TABLE, "main:after-a-failed-pass", &[3; 64])
        .unwrap();
    storage.flush_wal().unwrap();
    assert_eq!(moved_by_the_writer(&storage), by_the_writer + 1);
    let started = Instant::now();
    while probe.passes.load(Ordering::Relaxed) == before {
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "a pass that failed kept the flush from asking the checkpointer"
        );
        std::thread::sleep(Duration::from_millis(1));
    }

    reader.execute_batch("COMMIT").unwrap();
    drop(reader);
    storage.close().unwrap();
}

/// A process that dies leaves its log behind, and SQLite takes every frame of a log it
/// finds for unmoved. A log past its bound would send the first flush of the next
/// process to checkpoint all of it on the writer's thread — the first write after the
/// app is launched again. It is the checkpointer's: asked for as the store opens, and
/// not the writer's until that pass has ended.
#[test]
fn a_log_left_by_a_process_that_died_is_moved_by_the_checkpointer() {
    const VALVE: u32 = 64;
    const ROUNDS: usize = 150;

    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("test.sqlite");
    let mut storage = SqliteStorage::open_with_wal_valve(&path, VALVE).unwrap();
    let dying = probe(&storage);
    storage
        .raw_table_put(TABLE, "main:first", &[1; 64])
        .unwrap();
    storage.flush().unwrap();
    dying.stalled.store(true, Ordering::Relaxed);
    for key in 0..120 {
        storage
            .raw_table_put(TABLE, &format!("main:old-{key}"), &[7; 3000])
            .unwrap();
    }
    storage.flush_wal().unwrap();
    let (_first_dir, held_copy) = files_left_behind(&path, true);
    let (_second_dir, free_copy) = files_left_behind(&path, true);
    dying.stalled.store(false, Ordering::Relaxed);
    storage.close().unwrap();
    assert!(
        std::fs::metadata(log_path(&held_copy)).unwrap().len() / WAL_FRAME_BYTES as u64
            >= u64::from(VALVE)
    );

    // Opened, and nothing written: the log is moved all the same.
    let free = SqliteStorage::open_with_wal_valve(&free_copy, VALVE).unwrap();
    let asked = probe(&free);
    let started = Instant::now();
    while asked.passes.load(Ordering::Relaxed) == 0 || asked.busy.load(Ordering::Relaxed) {
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "nobody asked the checkpointer to move the log a dead process left"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(left_behind(&free_copy, false, "main:old-119"), Some(true));
    free.close().unwrap();

    // Opened with the checkpointer held up, and written to at once.
    let held = Arc::new(CheckpointerProbe::default());
    held.stalled.store(true, Ordering::Relaxed);
    let mut reopened = SqliteStorage::open_checkpointed(&held_copy, VALVE, |conn| {
        Checkpointer::start_probed(conn, &held_copy, false, Arc::clone(&held))
    })
    .unwrap();
    assert_eq!(
        reopened.raw_table_get(TABLE, "main:old-119").unwrap(),
        Some(vec![7; 3000])
    );
    reopened.raw_table_put(TABLE, "main:new", &[2; 64]).unwrap();
    reopened.flush_wal().unwrap();
    assert_eq!(
        left_behind(&held_copy, false, "main:old-119"),
        Some(false),
        "the first flush moved the log a dead process left on the writer's thread"
    );
    assert_eq!(held.passes.load(Ordering::Relaxed), 0);
    held.stalled.store(false, Ordering::Relaxed);
    wait_until_moved(&held, &held_copy, "main:new");

    // Once that pass has ended the log is bounded as any other is: by the writer, when
    // the checkpointer does not get to it.
    held.stalled.store(true, Ordering::Relaxed);
    let mut largest_log = 0;
    for round in 0..ROUNDS {
        for key in 0..8 {
            reopened
                .raw_table_put(TABLE, &format!("main:{round}-{key}"), &[round as u8; 3000])
                .unwrap();
        }
        reopened.flush_wal().unwrap();
        largest_log = largest_log.max(std::fs::metadata(log_path(&held_copy)).unwrap().len());
    }
    let frames = largest_log / WAL_FRAME_BYTES as u64;
    assert!(
        frames < 4 * u64::from(VALVE),
        "the log reached {frames} frames after the one a dead process left was moved"
    );
    held.stalled.store(false, Ordering::Relaxed);
    reopened.close().unwrap();
}

/// The log lies beside the file SQLite opened, which is not the path it was given when
/// that path is a link: a log left behind is found all the same.
#[cfg(unix)]
#[test]
fn a_log_left_behind_is_found_when_the_store_is_opened_through_a_link() {
    const VALVE: u32 = 64;

    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("test.sqlite");
    let mut storage = SqliteStorage::open_with_wal_valve(&path, VALVE).unwrap();
    let dying = probe(&storage);
    storage
        .raw_table_put(TABLE, "main:first", &[1; 64])
        .unwrap();
    storage.flush().unwrap();
    dying.stalled.store(true, Ordering::Relaxed);
    for key in 0..120 {
        storage
            .raw_table_put(TABLE, &format!("main:old-{key}"), &[7; 3000])
            .unwrap();
    }
    storage.flush_wal().unwrap();
    let (copy_dir, copy) = files_left_behind(&path, true);
    dying.stalled.store(false, Ordering::Relaxed);
    storage.close().unwrap();

    let link = copy_dir.path().join("link.sqlite");
    std::os::unix::fs::symlink(&copy, &link).unwrap();
    let held = Arc::new(CheckpointerProbe::default());
    held.stalled.store(true, Ordering::Relaxed);
    let mut reopened = SqliteStorage::open_checkpointed(&link, VALVE, |conn| {
        Checkpointer::start_probed(conn, &link, false, Arc::clone(&held))
    })
    .unwrap();
    let by_the_writer = moved_by_the_writer(&reopened);
    reopened.raw_table_put(TABLE, "main:new", &[2; 64]).unwrap();
    reopened.flush_wal().unwrap();
    assert_eq!(
        moved_by_the_writer(&reopened),
        by_the_writer,
        "the first flush moved the log a dead process left on the writer's thread"
    );
    held.stalled.store(false, Ordering::Relaxed);
    wait_until_moved(&held, &copy, "main:new");
    reopened.close().unwrap();
}

/// What a commit wrote is counted in pages written to the log, and a transaction larger
/// than the page cache writes a page it changes twice, twice: such a commit counts for
/// more than the frames it added. The bound holds all the same, one such commit later.
#[test]
fn a_commit_that_writes_its_pages_more_than_once_still_bounds_the_log() {
    const VALVE: u32 = 64;
    const ROUNDS: usize = 150;

    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("test.sqlite");
    let mut storage = SqliteStorage::open_with_wal_valve(&path, VALVE).unwrap();
    let probe = probe(&storage);
    probe.stalled.store(true, Ordering::Relaxed);
    // Eight pages of cache: a round below changes some thirty, each of them twice.
    storage
        .with_inner(|inner| {
            inner
                .conn
                .execute_batch("PRAGMA cache_size = 8")
                .map_err(|error| StorageError::IoError(error.to_string()))
        })
        .unwrap();
    let state = |storage: &SqliteStorage| {
        storage
            .with_inner(|inner| Ok((inner.log_frames, log_pages_written(&inner.conn))))
            .unwrap()
    };

    let mut largest_log = 0;
    let mut counted_twice = false;
    for round in 0..ROUNDS {
        let (frames_before, written_before) = state(&storage);
        for pass in 0..2 {
            for key in 0..24 {
                storage
                    .raw_table_put(
                        TABLE,
                        &format!("main:{key}"),
                        &[(round * 2 + pass) as u8; 3000],
                    )
                    .unwrap();
            }
        }
        storage.flush_wal().unwrap();
        let (frames, written) = state(&storage);
        counted_twice |=
            frames > frames_before && written.wrapping_sub(written_before) > frames - frames_before;
        largest_log = largest_log.max(std::fs::metadata(log_path(&path)).unwrap().len());
    }
    assert!(
        counted_twice,
        "no commit wrote more pages than it added frames: nothing was written twice"
    );
    let frames = largest_log / WAL_FRAME_BYTES as u64;
    assert!(
        frames < 6 * u64::from(VALVE),
        "the log reached {frames} frames with the checkpointer stalled"
    );
    assert_eq!(
        storage.raw_table_get(TABLE, "main:23").unwrap(),
        Some(vec![(ROUNDS * 2 - 1) as u8; 3000])
    );
    probe.stalled.store(false, Ordering::Relaxed);
    storage.close().unwrap();
}

/// A passive checkpoint moves what is committed when it starts, and the log is only
/// rewound by a write that begins with all of it moved. A writer the checkpointer cannot
/// keep up with — here, one it never gets a turn against — therefore has to move the log
/// itself now and then, or the log grows for as long as the writing lasts.
#[test]
fn a_log_the_checkpointer_cannot_keep_up_with_is_moved_by_the_writer() {
    const VALVE: u32 = 64;
    const ROUNDS: usize = 150;

    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("test.sqlite");
    let mut storage = SqliteStorage::open_with_wal_valve(&path, VALVE).unwrap();
    let probe = probe(&storage);
    probe.stalled.store(true, Ordering::Relaxed);

    let mut largest_log = 0;
    for round in 0..ROUNDS {
        for key in 0..8 {
            storage
                .raw_table_put(TABLE, &format!("main:{round}-{key}"), &[round as u8; 3000])
                .unwrap();
        }
        storage.flush_wal().unwrap();
        largest_log = largest_log.max(std::fs::metadata(log_path(&path)).unwrap().len());
    }

    // 150 commits of some ten pages each: 1 500 frames if nothing rewinds the log. The
    // first commit to find the log at 64 frames or more and leave it larger moves it, so
    // it holds at most the bound and the two commits that crossed it.
    let frames = largest_log / WAL_FRAME_BYTES as u64;
    assert!(
        frames < 2 * u64::from(VALVE),
        "the log reached {frames} frames with the checkpointer stalled"
    );
    assert_eq!(probe.passes.load(Ordering::Relaxed), 0);
    for round in [0, ROUNDS / 2, ROUNDS - 1] {
        assert_eq!(
            storage
                .raw_table_get(TABLE, &format!("main:{round}-7"))
                .unwrap(),
            Some(vec![round as u8; 3000])
        );
    }

    probe.stalled.store(false, Ordering::Relaxed);
    storage.close().unwrap();
}

/// The bound is on a log that keeps growing, not on one commit: a commit larger than the
/// bound by itself — a search's worth of rows — is exactly the one whose checkpoint must
/// stay off the writer's thread.
#[test]
fn a_commit_larger_than_the_bound_is_left_to_the_checkpointer() {
    const VALVE: u32 = 64;

    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("test.sqlite");
    let mut storage = SqliteStorage::open_with_wal_valve(&path, VALVE).unwrap();
    let probe = probe(&storage);
    let log_frames =
        |storage: &SqliteStorage| storage.with_inner(|inner| Ok(inner.log_frames)).unwrap();
    let commit = |storage: &mut SqliteStorage, name: &str, values: usize| {
        for key in 0..values {
            storage
                .raw_table_put(TABLE, &format!("main:{name}-{key}"), &[7; 3000])
                .unwrap();
        }
        storage.flush_wal().unwrap();
    };

    // Larger than the bound, into a log that was not: left to the checkpointer.
    probe.stalled.store(true, Ordering::Relaxed);
    commit(&mut storage, "first", 120);
    let first = log_frames(&storage);
    assert!(first >= 2 * VALVE, "the commit was {first} frames");
    assert_ne!(left_behind(&path, false, "main:first-0"), Some(true));
    probe.stalled.store(false, Ordering::Relaxed);
    wait_until_moved(&probe, &path, "main:first-119");

    // The next one rewinds the log, and is left to the checkpointer as well.
    probe.stalled.store(true, Ordering::Relaxed);
    commit(&mut storage, "second", 100);
    let second = log_frames(&storage);
    assert!(
        second >= VALVE && second < first,
        "the log was not rewound: {first} then {second} frames"
    );
    assert_ne!(left_behind(&path, false, "main:second-0"), Some(true));

    // With that one still not moved, a further commit finds the log past its bound and
    // leaves it larger: that is the log outgrowing the checkpointer, and the writer
    // moves it.
    commit(&mut storage, "third", 1);
    assert!(log_frames(&storage) > second);
    assert_eq!(left_behind(&path, false, "main:third-0"), Some(true));
    assert_eq!(left_behind(&path, false, "main:second-0"), Some(true));

    probe.stalled.store(false, Ordering::Relaxed);
    storage.close().unwrap();
}

/// A log that was past its bound and has since been moved in full is rewound by the next
/// write: what that commit leaves in it is the commit alone, however it compares with what
/// the log held before. Two searches in a row are this — the second brings a few more rows
/// than the first — and the second one's checkpoint belongs to the checkpointer as much
/// as the first one's did.
#[test]
fn a_commit_into_a_rewound_log_is_left_to_the_checkpointer_however_large() {
    const VALVE: u32 = 64;

    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("test.sqlite");
    let mut storage = SqliteStorage::open_with_wal_valve(&path, VALVE).unwrap();
    let probe = probe(&storage);
    let log_frames =
        |storage: &SqliteStorage| storage.with_inner(|inner| Ok(inner.log_frames)).unwrap();
    let commit = |storage: &mut SqliteStorage, name: &str, values: usize| {
        for key in 0..values {
            storage
                .raw_table_put(TABLE, &format!("main:{name}-{key}"), &[7; 3000])
                .unwrap();
        }
        storage.flush_wal().unwrap();
    };

    probe.stalled.store(true, Ordering::Relaxed);
    commit(&mut storage, "first", 120);
    let first = log_frames(&storage);
    assert!(first >= 2 * VALVE, "the commit was {first} frames");
    probe.stalled.store(false, Ordering::Relaxed);
    wait_until_moved(&probe, &path, "main:first-119");

    // Larger than the one before it, into a log that one no longer occupies.
    probe.stalled.store(true, Ordering::Relaxed);
    commit(&mut storage, "second", 130);
    let second = log_frames(&storage);
    assert!(
        second > first && second < first + VALVE,
        "the log was not rewound, or the commit was not the larger: {first} then {second} frames"
    );
    assert_ne!(
        left_behind(&path, false, "main:second-0"),
        Some(true),
        "the writer moved a log that had been rewound under it"
    );

    probe.stalled.store(false, Ordering::Relaxed);
    storage.close().unwrap();
}

/// What the writer remembers about its keys is dropped when another connection commits
/// (`note_transaction_began`). The checkpointer is another connection and runs after
/// every flush; its checkpoint is not a commit and must not look like one.
#[test]
fn a_checkpoint_on_the_other_thread_does_not_cost_the_writer_what_it_remembers() {
    let dir = tempfile::TempDir::new().unwrap();
    let (mut storage, path) = open(&dir);
    let probe = probe(&storage);

    let mut seen_before = None;
    for round in 0..5 {
        // A write transaction, begun after the checkpoint of the one before it — and,
        // the log being moved in full by then, the one that rewinds the log.
        let key = format!("main:{round}");
        storage.raw_table_put(TABLE, &key, &[7; 2048]).unwrap();
        let seen = storage
            .with_inner(|inner| Ok(inner.seen_data_version))
            .unwrap();
        assert!(seen.is_some());
        if round > 0 {
            assert_eq!(
                seen, seen_before,
                "the checkpoint looked like another connection's commit"
            );
        }
        seen_before = seen;
        storage.flush_wal().unwrap();
        wait_until_moved(&probe, &path, &key);
    }
    assert!(probe.passes.load(Ordering::Relaxed) >= 5);
    storage.close().unwrap();
}

/// The connection that closes last folds the log away and unlinks it by name. That has
/// to be the engine's, before `close()` returns or the store is dropped: a caller told
/// the store is closed may delete it and open a new one at the same path, and a
/// checkpointer still closing would unlink the new store's log.
#[test]
fn closing_the_store_closes_its_checkpointer_first() {
    let dir = tempfile::TempDir::new().unwrap();
    let (mut storage, path) = open(&dir);
    let closed = probe(&storage);
    storage
        .raw_table_put(TABLE, "main:row", &[7; 2048])
        .unwrap();
    storage.flush_wal().unwrap();
    assert!(log_path(&path).exists());

    storage.close().unwrap();
    assert!(
        closed.exited.load(Ordering::Relaxed),
        "close() returned with the checkpointer's connection still open"
    );
    assert!(
        !log_path(&path).exists(),
        "the last connection to close folds the log away"
    );

    // Dropped without a close: the same, and what was not flushed is gone with it.
    let mut storage = SqliteStorage::open(&path).unwrap();
    let dropped = probe(&storage);
    storage
        .raw_table_put(TABLE, "main:unflushed", b"v")
        .unwrap();
    drop(storage);
    assert!(
        dropped.exited.load(Ordering::Relaxed),
        "the store was dropped with the checkpointer's connection still open"
    );
    assert!(!log_path(&path).exists());

    // A close that cannot commit still closes everything.
    let mut storage = SqliteStorage::open(&path).unwrap();
    let failed = probe(&storage);
    storage.raw_table_put(TABLE, "main:lost", b"v").unwrap();
    storage
        .with_inner(|inner| {
            inner.conn.execute_batch("ROLLBACK").unwrap();
            Ok(())
        })
        .unwrap();
    assert!(storage.close().is_err());
    assert!(failed.exited.load(Ordering::Relaxed));
    assert!(!log_path(&path).exists());

    // Deleted and opened again at the same path, as a sign-out does: the new store keeps
    // its log.
    std::fs::remove_file(&path).unwrap();
    let mut storage = SqliteStorage::open(&path).unwrap();
    storage
        .raw_table_put(TABLE, "main:new", &[7; 2048])
        .unwrap();
    storage.flush_wal().unwrap();
    assert_eq!(left_behind(&path, true, "main:new"), Some(true));
    assert_eq!(left_behind(&path, true, "main:row"), Some(false));
    storage.close().unwrap();

    let storage = SqliteStorage::open(&path).unwrap();
    assert_eq!(
        storage.raw_table_get(TABLE, "main:new").unwrap(),
        Some(vec![7; 2048])
    );
}

/// A batch of one index entry is written without a savepoint. It still writes nothing
/// when it fails, and what the writer remembers about its keys does not survive the
/// failure.
#[test]
fn a_batch_of_one_index_mutation_that_fails_writes_nothing() {
    let dir = tempfile::TempDir::new().unwrap();
    let (mut storage, _path) = open(&dir);
    let row_id = ObjectId::new();
    let unkeyable_column = "c".repeat(6 * 1024);

    let written = [IndexMutation::Insert {
        table: "t",
        column: "c",
        branch: "main",
        value: Value::Integer(1),
        row_id,
    }];
    storage.apply_index_mutations(&written).unwrap();
    assert!(
        storage
            .index_contains("t", "c", "main", &Value::Integer(1), row_id)
            .unwrap()
    );

    let unkeyable = [IndexMutation::Insert {
        table: "t",
        column: &unkeyable_column,
        branch: "main",
        value: Value::Integer(2),
        row_id,
    }];
    storage
        .with_inner(|inner| {
            inner
                .memo
                .borrow_mut()
                .memoize_prefix("remembered:".to_string(), PrefixKeys::Small(HashSet::new()));
            Ok(())
        })
        .unwrap();
    assert!(matches!(
        storage.apply_index_mutations(&unkeyable),
        Err(StorageError::IndexKeyTooLarge { .. })
    ));
    assert!(
        storage
            .with_inner(|inner| Ok(inner.memo.borrow().prefix_keys.is_empty()))
            .unwrap(),
        "a failed write left the read memo standing"
    );
    assert!(
        storage
            .index_contains("t", "c", "main", &Value::Integer(1), row_id)
            .unwrap(),
        "the entry written before the failed one is still there"
    );
    storage.flush_wal().unwrap();
    storage.close().unwrap();
}

/// One statement that SQLite itself refuses, after it has written: the statement is
/// undone by SQLite, the transaction it was part of goes on, and the memo is dropped.
#[test]
fn a_single_statement_that_sqlite_fails_is_undone_and_the_transaction_goes_on() {
    let dir = tempfile::TempDir::new().unwrap();
    let (mut storage, path) = open(&dir);
    storage
        .with_inner(|inner| {
            inner
                .conn
                .execute_batch(
                    "CREATE TEMP TRIGGER refused AFTER INSERT ON main.kv
                     WHEN length(new.value) = 13
                     BEGIN SELECT RAISE(ABORT, 'refused'); END;",
                )
                .unwrap();
            Ok(())
        })
        .unwrap();

    storage
        .raw_table_put(TABLE, "main:kept", b"before")
        .unwrap();
    storage
        .with_inner(|inner| {
            inner
                .memo
                .borrow_mut()
                .memoize_prefix("remembered:".to_string(), PrefixKeys::Small(HashSet::new()));
            Ok(())
        })
        .unwrap();
    // A replacement and a first write, both refused once the row is already in.
    for key in ["main:kept", "main:never"] {
        assert!(matches!(
            storage.raw_table_put(TABLE, key, &[7; 13]),
            Err(StorageError::IoError(_))
        ));
    }
    assert!(
        storage
            .with_inner(|inner| Ok(inner.memo.borrow().prefix_keys.is_empty()))
            .unwrap(),
        "a failed write left the read memo standing"
    );
    assert_eq!(
        storage.raw_table_get(TABLE, "main:kept").unwrap(),
        Some(b"before".to_vec())
    );
    assert_eq!(storage.raw_table_get(TABLE, "main:never").unwrap(), None);

    // The transaction the failed statements were in is still the one being written.
    storage
        .raw_table_put(TABLE, "main:after", b"after")
        .unwrap();
    storage.flush_wal().unwrap();
    assert_eq!(left_behind(&path, true, "main:kept"), Some(true));
    assert_eq!(left_behind(&path, true, "main:after"), Some(true));
    assert_eq!(left_behind(&path, true, "main:never"), Some(false));
    storage.close().unwrap();
}

/// The store against a model of it, with the checkpointer racing the writer.
///
/// Writes of one statement and of several, batches that fail, flushes of both kinds,
/// reads inside and outside read scopes — and, at any step, the three ways a store stops
/// being this process's: it is closed, it is dropped, or the process dies and leaves its
/// files. Whatever was flushed is in what comes back, nothing that was not flushed is,
/// and while the store is open it reads exactly what was written to it.
///
/// Half the runs use a log bound of a few dozen frames, so the writer's own checkpoint
/// and the rewinding of the log happen every few steps and collide with the
/// checkpointer's. In all of them the checkpointer is stalled and released at random,
/// which is the writer outrunning it.
#[test]
fn write_path_differential_against_a_model_with_the_checkpointer_racing() {
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, bound: usize) -> usize {
            (self.next() % bound as u64) as usize
        }
        fn chance(&mut self, one_in: usize) -> bool {
            self.below(one_in) == 0
        }
    }

    type Rows = BTreeMap<String, Vec<u8>>;
    type Entries = BTreeSet<(i32, ObjectId)>;

    const KEYS: usize = 40;
    const STEPS: usize = 1_200;

    fn key(rng: &mut Rng) -> String {
        format!("main:{:03}", rng.below(KEYS))
    }
    fn value(rng: &mut Rng) -> Vec<u8> {
        // From a few bytes to several pages: the larger ones are what fills the log.
        let len = match rng.below(4) {
            0 => rng.below(16),
            1 => 200 + rng.below(800),
            _ => 2_000 + rng.below(9_000),
        };
        vec![rng.below(251) as u8; len]
    }
    fn entry(rng: &mut Rng, row_ids: &[ObjectId]) -> (i32, ObjectId) {
        (rng.below(6) as i32, row_ids[rng.below(row_ids.len())])
    }
    fn index_mutation((value, row_id): (i32, ObjectId), insert: bool) -> IndexMutation<'static> {
        if insert {
            IndexMutation::Insert {
                table: "t",
                column: "c",
                branch: "main",
                value: Value::Integer(value),
                row_id,
            }
        } else {
            IndexMutation::Remove {
                table: "t",
                column: "c",
                branch: "main",
                value: Value::Integer(value),
                row_id,
            }
        }
    }
    fn assert_holds(storage: &SqliteStorage, rows: &Rows, entries: &Entries, what: &str) {
        let stored: Rows = storage
            .raw_table_scan_prefix(TABLE, "")
            .unwrap()
            .into_iter()
            .collect();
        assert_eq!(&stored, rows, "{what}: rows");
        // One id per entry, whatever its value: the count is the entries the index holds.
        let mut stored = storage.index_scan_all("t", "c", "main");
        stored.sort();
        let mut expected: Vec<ObjectId> = entries.iter().map(|(_, row_id)| *row_id).collect();
        expected.sort();
        assert_eq!(stored, expected, "{what}: index entries");
        for (value, row_id) in entries {
            assert!(
                storage
                    .index_contains("t", "c", "main", &Value::Integer(*value), *row_id)
                    .unwrap(),
                "{what}: index entry {value} of {row_id:?}"
            );
        }
    }

    let unkeyable_column = "c".repeat(6 * 1024);

    for (run, seed) in [
        0x9E37_79B9_7F4A_7C15u64,
        0xD1B5_4A32_D192_ED03,
        0x2545_F491_4F6C_DD1D,
        0x94D0_49BB_1331_11EB,
    ]
    .into_iter()
    .enumerate()
    {
        let mut rng = Rng(seed);
        let small_valve = run % 2 == 0;
        let reopen = |path: &Path| {
            if small_valve {
                SqliteStorage::open_with_wal_valve(path, 48).unwrap()
            } else {
                SqliteStorage::open(path).unwrap()
            }
        };
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("test.sqlite");
        let mut storage = reopen(&path);
        let row_ids: Vec<ObjectId> = (0..6).map(|_| ObjectId::new()).collect();

        // What the open store reads, and what its files hold: the former as of the last
        // flush.
        let mut live_rows = Rows::new();
        let mut live_entries = Entries::new();
        let mut flushed_rows = Rows::new();
        let mut flushed_entries = Entries::new();

        let mut deaths = 0usize;
        let mut drops = 0usize;
        let mut closes = 0usize;
        let mut failed_batches = 0usize;
        let mut stalls = 0usize;
        let mut passes = 0u64;

        for step in 0..STEPS {
            match rng.below(100) {
                // One statement: no savepoint.
                0..=24 => {
                    let (key, value) = (key(&mut rng), value(&mut rng));
                    storage.raw_table_put(TABLE, &key, &value).unwrap();
                    live_rows.insert(key, value);
                }
                25..=32 => {
                    let key = key(&mut rng);
                    storage.raw_table_delete(TABLE, &key).unwrap();
                    live_rows.remove(&key);
                }
                // Several statements in one savepoint.
                33..=42 => {
                    let puts: Vec<(String, Option<Vec<u8>>)> = (0..2 + rng.below(5))
                        .map(|_| {
                            let key = key(&mut rng);
                            (key, (!rng.chance(4)).then(|| value(&mut rng)))
                        })
                        .collect();
                    let mutations: Vec<RawTableMutation<'_>> = puts
                        .iter()
                        .map(|(key, value)| match value {
                            Some(value) => RawTableMutation::Put {
                                table: TABLE,
                                key,
                                value,
                            },
                            None => RawTableMutation::Delete { table: TABLE, key },
                        })
                        .collect();
                    storage.apply_raw_table_mutations(&mutations).unwrap();
                    for (key, value) in puts {
                        match value {
                            Some(value) => live_rows.insert(key, value),
                            None => live_rows.remove(&key),
                        };
                    }
                }
                // Index entries, one to a batch as a row from sync writes them, or several.
                43..=57 => {
                    let batch: Vec<((i32, ObjectId), bool)> = (0..1 + rng.below(3) * rng.below(3))
                        .map(|_| (entry(&mut rng, &row_ids), !rng.chance(3)))
                        .collect();
                    let mutations: Vec<IndexMutation<'_>> = batch
                        .iter()
                        .map(|(entry, insert)| index_mutation(*entry, *insert))
                        .collect();
                    storage.apply_index_mutations(&mutations).unwrap();
                    for (entry, insert) in batch {
                        if insert {
                            live_entries.insert(entry);
                        } else {
                            live_entries.remove(&entry);
                        }
                    }
                }
                // A batch that fails after its first entry, or as its only one: nothing of
                // it is written.
                58..=60 => {
                    let mut mutations = Vec::new();
                    if rng.chance(2) {
                        let fresh = (100 + step as i32, row_ids[0]);
                        mutations.push(index_mutation(fresh, true));
                    }
                    mutations.push(IndexMutation::Insert {
                        table: "t",
                        column: &unkeyable_column,
                        branch: "main",
                        value: Value::Integer(0),
                        row_id: row_ids[0],
                    });
                    assert!(storage.apply_index_mutations(&mutations).is_err());
                    failed_batches += 1;
                }
                61..=72 => {
                    storage.flush_wal().unwrap();
                    flushed_rows = live_rows.clone();
                    flushed_entries = live_entries.clone();
                }
                73..=76 => {
                    storage.flush().unwrap();
                    flushed_rows = live_rows.clone();
                    flushed_entries = live_entries.clone();
                }
                // The process dies here: what its files hold is what was flushed.
                77..=82 => {
                    let (_copy_dir, copy) = files_left_behind(&path, true);
                    let after = SqliteStorage::open(&copy).unwrap();
                    assert_holds(&after, &flushed_rows, &flushed_entries, "after a death");
                    after.close().unwrap();
                    deaths += 1;
                }
                // Closed and opened again: a close commits.
                83..=85 => {
                    probe(&storage).stalled.store(false, Ordering::Relaxed);
                    passes += probe(&storage).passes.load(Ordering::Relaxed);
                    storage.close().unwrap();
                    flushed_rows = live_rows.clone();
                    flushed_entries = live_entries.clone();
                    storage = reopen(&path);
                    closes += 1;
                }
                // Dropped and opened again: what was not flushed is gone.
                86..=88 => {
                    probe(&storage).stalled.store(false, Ordering::Relaxed);
                    passes += probe(&storage).passes.load(Ordering::Relaxed);
                    drop(storage);
                    live_rows = flushed_rows.clone();
                    live_entries = flushed_entries.clone();
                    storage = reopen(&path);
                    drops += 1;
                }
                // The checkpointer falls behind, or catches up.
                89..=93 => {
                    let stalled = &probe(&storage).stalled;
                    let was = stalled.load(Ordering::Relaxed);
                    stalled.store(!was, Ordering::Relaxed);
                    stalls += usize::from(!was);
                }
                // Reads, in a scope or outside one.
                _ => {
                    let scoped = rng.chance(2);
                    if scoped {
                        storage.begin_read_scope();
                    }
                    for _ in 0..3 {
                        let key = key(&mut rng);
                        assert_eq!(
                            storage.raw_table_get(TABLE, &key).unwrap(),
                            live_rows.get(&key).cloned(),
                            "seed {seed:#x} step {step}: {key}"
                        );
                        let entry = entry(&mut rng, &row_ids);
                        assert_eq!(
                            storage
                                .index_contains("t", "c", "main", &Value::Integer(entry.0), entry.1)
                                .unwrap(),
                            live_entries.contains(&entry),
                            "seed {seed:#x} step {step}: {entry:?}"
                        );
                    }
                    if scoped {
                        storage.end_read_scope();
                    }
                }
            }

            if step % 25 == 0 {
                assert_holds(
                    &storage,
                    &live_rows,
                    &live_entries,
                    &format!("seed {seed:#x} step {step}"),
                );
            }
        }

        probe(&storage).stalled.store(false, Ordering::Relaxed);
        passes += probe(&storage).passes.load(Ordering::Relaxed);
        storage.close().unwrap();
        let storage = reopen(&path);
        assert_holds(&storage, &live_rows, &live_entries, "after the last close");
        storage.close().unwrap();

        // The run did what it is for.
        assert!(
            deaths >= 20 && drops >= 10 && closes >= 10 && failed_batches >= 10 && stalls >= 10,
            "seed {seed:#x}: {deaths} deaths, {drops} drops, {closes} closes, \
             {failed_batches} failed batches, {stalls} stalls"
        );
        assert!(passes > 0, "seed {seed:#x}: the checkpointer never ran");
    }
}
