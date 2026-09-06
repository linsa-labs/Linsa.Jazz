use super::*;

#[test]
fn rc_row_writes_do_not_touch_legacy_commit_storage() {
    let calls = Arc::new(Mutex::new(LegacyStorageCallCounts::default()));
    let mut core = create_runtime_with_boxed_storage(
        test_schema(),
        "row-no-legacy-commit-storage",
        Box::new(LegacyPersistenceObservingStorage::new(Arc::clone(&calls))),
    );

    let ((row_id, _row_values), _) = core
        .insert("users", user_insert_values(ObjectId::new(), "Alice"), None)
        .unwrap();

    core.update(
        row_id,
        vec![("name".into(), Value::Text("Bob".into()))],
        None,
    )
    .unwrap();
    core.delete(row_id, None).unwrap();

    assert_eq!(
        *calls.lock().unwrap(),
        LegacyStorageCallCounts::default(),
        "row writes should persist only via row histories, not legacy branch commit storage"
    );
}

#[test]
fn rc_local_row_writes_batch_row_and_index_mutations() {
    let calls = Arc::new(Mutex::new(RowMutationCallCounts::default()));
    let mut core = create_runtime_with_boxed_storage(
        test_schema(),
        "row-batched-storage-mutation",
        Box::new(RowMutationObservingStorage::new(Arc::clone(&calls))),
    );

    let ((row_id, _row_values), _) = core
        .insert("users", user_insert_values(ObjectId::new(), "Alice"), None)
        .unwrap();
    core.update(
        row_id,
        vec![("name".into(), Value::Text("Bob".into()))],
        None,
    )
    .unwrap();
    core.delete(row_id, None).unwrap();

    assert_eq!(
        *calls.lock().unwrap(),
        RowMutationCallCounts {
            row_mutation_calls: 3,
            separate_index_mutation_calls: 0,
            flush_wal_calls: 0,
            local_batch_record_get_calls: 3,
        },
        "local row writes should persist row history, visible heads, and index changes in one storage mutation"
    );
}

#[test]
fn rc_batched_tick_skips_flush_wal_without_storage_writes() {
    let calls = Arc::new(Mutex::new(RowMutationCallCounts::default()));
    let mut core = create_runtime_with_boxed_storage(
        test_schema(),
        "row-batched-no-flush",
        Box::new(RowMutationObservingStorage::new(Arc::clone(&calls))),
    );

    core.batched_tick();

    assert_eq!(
        calls.lock().unwrap().flush_wal_calls,
        0,
        "read-only batched ticks should not flush the WAL"
    );
}

#[test]
fn rc_local_write_without_outbox_still_schedules_batched_tick_for_flush() {
    let scheduler = CountingScheduler::default();
    let app_id = AppId::from_name("row-schedule-flush-without-outbox");
    let schema_manager =
        SchemaManager::new(SyncManager::new(), test_schema(), app_id, "dev", "main").unwrap();
    let mut core = new_test_core(schema_manager, MemoryStorage::new(), scheduler.clone());
    core.immediate_tick();
    let scheduled_before = scheduler.schedule_count();

    core.insert("users", user_insert_values(ObjectId::new(), "Alice"), None)
        .unwrap();

    assert!(
        scheduler.schedule_count() > scheduled_before,
        "local writes without peers should still schedule batched_tick so the WAL flush barrier runs"
    );
}

#[test]
fn rc_batched_tick_flushes_wal_after_local_write() {
    let calls = Arc::new(Mutex::new(RowMutationCallCounts::default()));
    let mut core = create_runtime_with_boxed_storage(
        test_schema(),
        "row-batched-flush-after-write",
        Box::new(RowMutationObservingStorage::new(Arc::clone(&calls))),
    );

    core.insert("users", user_insert_values(ObjectId::new(), "Alice"), None)
        .unwrap();
    core.batched_tick();

    assert_eq!(
        calls.lock().unwrap().flush_wal_calls,
        1,
        "a batched tick after a local write should flush the WAL once"
    );
}

#[test]
fn rc_flush_storage_records_error_and_clears_after_transient_success() {
    let flush_error = StorageError::IoError("transient full flush failure".to_string());
    let app_id = AppId::from_name("row-full-flush-transient-failure");
    let schema_manager =
        SchemaManager::new(SyncManager::new(), test_schema(), app_id, "dev", "main").unwrap();
    let storage = MemoryStorage::new().with_transient_flush_failures(flush_error.clone(), 1);
    let mut core = new_test_core(schema_manager, storage, NoopScheduler);

    core.insert("users", user_insert_values(ObjectId::new(), "Alice"), None)
        .unwrap();

    let error = core.flush_storage().expect_err("first flush should fail");
    assert_eq!(error, flush_error);
    assert!(core.has_storage_write_pending_flush());
    assert_eq!(core.take_storage_flush_error(), Some(flush_error));

    core.flush_storage()
        .expect("second flush should clear transient failure");
    assert!(!core.has_storage_write_pending_flush());
    assert_eq!(core.take_storage_flush_error(), None);
}

#[test]
fn rc_batched_tick_reschedules_after_transient_wal_flush_failure() {
    let flush_error = StorageError::IoError("transient WAL flush failure".to_string());
    let scheduler = CountingScheduler::default();
    let app_id = AppId::from_name("row-reschedule-wal-flush-failure");
    let schema_manager =
        SchemaManager::new(SyncManager::new(), test_schema(), app_id, "dev", "main").unwrap();
    let storage = MemoryStorage::new().with_transient_flush_wal_failures(flush_error.clone(), 1);
    let mut core = new_test_core(schema_manager, storage, scheduler.clone());

    core.insert("users", user_insert_values(ObjectId::new(), "Alice"), None)
        .unwrap();
    let scheduled_after_write = scheduler.schedule_count();

    core.batched_tick();

    assert!(core.has_storage_write_pending_flush());
    assert_eq!(core.take_storage_flush_error(), Some(flush_error));
    assert!(
        scheduler.schedule_count() > scheduled_after_write,
        "a transient WAL flush failure should schedule a retry tick"
    );

    core.batched_tick();
    assert!(!core.has_storage_write_pending_flush());
    assert_eq!(core.take_storage_flush_error(), None);
}

#[test]
fn rc_batched_tick_skips_flush_wal_for_query_settled_only_message() {
    let calls = Arc::new(Mutex::new(RowMutationCallCounts::default()));
    let mut core = create_runtime_with_boxed_storage(
        test_schema(),
        "row-batched-query-settled",
        Box::new(RowMutationObservingStorage::new(Arc::clone(&calls))),
    );

    core.push_sync_inbox(InboxEntry {
        source: Source::Server(ServerId::new()),
        payload: SyncPayload::QuerySettled {
            query_id: crate::sync_manager::QueryId(1),
            tier: DurabilityTier::Local,
            scope: vec![],
            through_seq: 1,
        },
    });
    core.batched_tick();

    assert_eq!(
        calls.lock().unwrap().flush_wal_calls,
        0,
        "query-settled notifications alone should not flush the WAL"
    );
}

/// G-C14. A `LostWrites` is reported to the host EXACTLY ONCE, and it does not re-arm the
/// barrier. Every other storage error keeps today's behaviour: overwrite the carrier on each
/// occurrence and schedule a retry. The distinction is the point — a transient failure is
/// worth retrying, a store that cannot persist is not, and a host that is told about it on
/// every tick forever cannot tell the two apart.
///
/// Three writes and three barriers, each of which fails: the host sees the loss on the first
/// and nothing after it, while the core's own latch stays set.
///
/// Internal on purpose: the carrier, the latch and the scheduler count are internal, and the
/// injected failure has no public route — `MemoryStorage`'s failure hooks are `#[cfg(test)]`.
#[test]
fn rc_a_lost_writes_barrier_failure_is_reported_once_and_does_not_rearm() {
    let detail = "the write transaction was ended behind the store's back".to_string();
    let scheduler = CountingScheduler::default();
    let app_id = AppId::from_name("row-lost-writes-once");
    let schema_manager =
        SchemaManager::new(SyncManager::new(), test_schema(), app_id, "dev", "main").unwrap();
    let storage = MemoryStorage::new().with_transient_flush_wal_failures(
        StorageError::LostWrites {
            detail: detail.clone(),
        },
        3,
    );
    let mut core = new_test_core(schema_manager, storage, scheduler.clone());

    for tick in 1..=3 {
        core.insert("users", user_insert_values(ObjectId::new(), "Alice"), None)
            .unwrap();
        let scheduled_after_write = scheduler.schedule_count();

        core.batched_tick();

        let taken = core.take_storage_flush_error();
        if tick == 1 {
            assert!(
                matches!(taken, Some(StorageError::LostWrites { .. })),
                "the first lost-writes barrier must hand the host the loss, got {taken:?}"
            );
        } else {
            assert!(
                taken.is_none(),
                "tick {tick}: a store that already reported its loss must not report it \
                 again; the host cannot distinguish that from a fresh failure"
            );
        }
        assert!(
            core.lost_writes_barrier_reported_for_test(),
            "tick {tick}: the core's own latch must stay set once the barrier reported"
        );
        assert_eq!(
            scheduler.schedule_count(),
            scheduled_after_write,
            "tick {tick}: a lost-writes barrier must not schedule a retry — the retry cannot \
             succeed, and the tick it costs is paid under the core lock"
        );
    }
}
