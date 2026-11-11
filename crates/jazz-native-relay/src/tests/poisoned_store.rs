//! A relay store a build without the app's migrations opened cannot serve
//! foregrounds that open with them. The host learns why from `TickEvents`,
//! as non-retrying upstream state, while ticks keep succeeding: the channel
//! the TS runtime turns into failed remote waiters, not thrown ticks.
use super::*;
use crate::relay::migrations_tests::{author, identity, migrations, relay, v2};

#[test]
fn a_poisoned_store_reports_its_refusal_in_tick_events_and_keeps_ticking() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("alice.sqlite");
    drop(relay(path.clone(), None));

    let mut host = NativeRelayHost::default();
    let admitted = host
        .admit_scope(RelayScopeAdmissionRequest {
            scope: RelayScopeRequest {
                app_namespace: "app".to_owned(),
                storage_namespace: "default".to_owned(),
                auth_scope: Some("alice".to_owned()),
            },
            sqlite_path: path.display().to_string(),
            schema_json: serde_json::to_string(v2().public_schema()).unwrap(),
            migrations_json: Some(String::from_utf8(migrations()).unwrap()),
            identity: DbIdentity {
                author: author(),
                ..identity(0x01)
            },
            claims: BTreeMap::new(),
        })
        .unwrap();
    let foreground = host
        .open_foreground(admitted, DIRECT_FOREGROUND_RUNTIME_TOKEN)
        .unwrap();
    let drain = |host: &mut NativeRelayHost| {
        assert_eq!(
            host.tick_foreground(foreground),
            Ok(()),
            "ticks keep running"
        );
        match host
            .execute_foreground_command(foreground, ForegroundDbCommandRequest::DrainTickEvents)
            .unwrap()
        {
            ForegroundDbCommandResponse::TickEvents {
                upstream_failure, ..
            } => upstream_failure,
            other => panic!("unexpected tick events: {other:?}"),
        }
    };
    let failure = drain(&mut host).expect("the refusal reaches TickEvents");
    assert!(!failure.retrying);
    assert!(
        failure.reason.contains("refused its relay store")
            && failure.reason.contains("stores otherwise"),
        "{}",
        failure.reason
    );
    assert_eq!(
        drain(&mut host),
        Some(failure),
        "reported once: one sequence"
    );
}
