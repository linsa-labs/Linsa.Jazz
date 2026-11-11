//! A relay opens a schema the app reached through migrations. Its upstream
//! stores that version under the names its lenses carried forward from the
//! versions before it, so a relay that admits the version as its first one
//! names its columns otherwise and its upstream's catalogue does not mirror:
//! the relay must open with the app's migrations, as napi and wasm do.
use super::*;
use crate::{NATIVE_RELAY_ABI_V1, NativeRelayAbiRange};
use jazz::db::{FileOptions, InsertOptions, ReadOpts};
use jazz::ids::NodeUuid;
use jazz::query::Query;
use jazz::tools::deployment::published::{
    ClientMigrations, PublishLensOp, PublishMigrationRequest, PublishTableLens, PublishedSchema,
    client_migrations,
};
use jazz::tools::{
    ColumnType, PolicyExpr, Schema, SchemaBuilder, SchemaHash, TablePolicies, TableSchemaBuilder,
    Value as PublicValue,
};

fn allow_all() -> TablePolicies {
    let allow = PolicyExpr::True;
    TablePolicies::new()
        .with_select(allow.clone())
        .with_insert(allow.clone())
        .with_update(Some(allow.clone()), allow.clone())
        .with_delete(allow)
}

fn parents(extra: impl FnOnce(TableSchemaBuilder) -> TableSchemaBuilder) -> TableSchemaBuilder {
    extra(TableSchemaBuilder::new("parents").column("title", ColumnType::Text))
        .policies(allow_all())
}

/// `parents(title)`: the genesis version.
fn v1_source() -> Schema {
    SchemaBuilder::new().table(parents(|t| t)).build()
}

/// v1 plus `parents.rank: bigint?`.
pub(crate) fn v2_source() -> Schema {
    SchemaBuilder::new()
        .table(parents(|t| t.nullable_column("rank", ColumnType::BigInt)))
        .build()
}

fn v1() -> JazzSchema {
    JazzSchema::new(&v1_source()).unwrap()
}

fn v2() -> JazzSchema {
    JazzSchema::new(&v2_source()).unwrap()
}

/// The migrations the app ships, as `defineApp` records them: both versions
/// and the generated migration that adds `parents.rank`.
pub(crate) fn migrations() -> Vec<u8> {
    let schema = |schema: Schema| PublishedSchema {
        hash: SchemaHash::compute(&schema).to_string(),
        schema,
    };
    serde_json::to_vec(&ClientMigrations {
        schemas: vec![schema(v1_source()), schema(v2_source())],
        migrations: vec![PublishMigrationRequest {
            from_hash: SchemaHash::compute(&v1_source()).to_string(),
            to_hash: SchemaHash::compute(&v2_source()).to_string(),
            forward: vec![PublishTableLens {
                table: "parents".to_owned(),
                added: false,
                removed: false,
                renamed_from: None,
                operations: vec![PublishLensOp::Introduce {
                    column: "rank".to_owned(),
                    column_type: ColumnType::BigInt,
                    value: PublicValue::Null,
                }],
            }],
        }],
    })
    .unwrap()
}

fn author() -> AuthorSubject {
    AuthorSubject::for_test_bytes([0x5a; 16])
}

fn identity(node: u8) -> DbIdentity {
    DbIdentity {
        node: NodeUuid::from_bytes([node; 16]),
        author: author(),
    }
}

/// The authority as a server core runs it: opened under its genesis v1,
/// with v2 admitted through the app's lens.
fn authority(path: &std::path::Path) -> Binding {
    let identity = DbIdentity {
        node: NodeUuid::from_bytes([0; 16]),
        author: AuthorSubject::SYSTEM,
    };
    let db = block_on(Db::open_at(
        DbConfig::new(v1(), identity),
        path,
        FileOptions::default(),
    ))
    .unwrap();
    db.set_authority();
    let (_, lenses) = client_migrations(&migrations()).unwrap();
    block_on(db.admit_schema_with_lenses(v2(), &lenses)).unwrap();
    Binding::new(db, identity, true)
}

fn relay(path: std::path::PathBuf, migrations: Option<Vec<u8>>) -> NativeRelay {
    NativeRelay::spawn(RelayOpenConfig {
        supported_abi: NativeRelayAbiRange {
            minimum: NATIVE_RELAY_ABI_V1,
            maximum: NATIVE_RELAY_ABI_V1,
        },
        scope: RelayScope {
            app_namespace: "app".to_owned(),
            storage_namespace: "default".to_owned(),
            auth_scope: Some("alice".to_owned()),
        },
        sqlite_path: path,
        schema: v2(),
        identity: identity(0x01),
        migrations,
    })
    .unwrap()
}

/// Carry frames between the relay's host-carried upstream and the authority.
fn pump(relay: &NativeRelay, wire: &NativeRelayWire, server: &Binding, link: &Link) {
    for _ in 0..16 {
        relay.pump().unwrap();
        link.receive(wire.take_outbound_encoded().unwrap());
        block_on(server.tick()).unwrap();
        for frame in link.take_outbound() {
            wire.push_inbound_encoded(&frame).unwrap();
        }
    }
    relay.pump().unwrap();
}

fn refusal(relay: &NativeRelay) -> Option<String> {
    relay
        .run(|worker| {
            Ok(worker
                .upstream
                .as_ref()
                .and_then(|upstream| upstream.link.upstream_refusal())
                .map(|error| error.to_string()))
        })
        .unwrap()
}

fn titles(db: &Db) -> Vec<(String, Option<Value>)> {
    let prepared = db.prepare_query(&Query::from("parents")).unwrap();
    let mut rows: Vec<_> = block_on(db.all(&prepared, ReadOpts::default()))
        .unwrap()
        .iter()
        .map(|row| {
            let cell = |name: &str| {
                row.cells()
                    .find(|(column, _)| *column == name)
                    .and_then(|(_, value)| value.cloned())
            };
            let title = match cell("title") {
                Some(Value::String(title)) => title,
                other => panic!("title cell: {other:?}"),
            };
            (title, cell("rank"))
        })
        .collect();
    rows.sort_by(|a, b| a.0.cmp(&b.0));
    rows
}

/// ```
/// authority(v1) ── admit v2 via the app's lens ──┐
/// relay(v2 + the app's migrations) ──────────────┴── catalogue ──► mirrors
///                                     (without them: refused, nothing syncs)
/// relay client ──insert parents{title, rank}──► authority stores it
/// ```
#[test]
fn a_relay_on_a_migrated_version_mirrors_its_upstream_and_syncs() {
    let directory = tempfile::tempdir().unwrap();
    let server = authority(&directory.path().join("core.sqlite"));
    let relay = relay(directory.path().join("alice.sqlite"), Some(migrations()));
    let wire = relay.wire().unwrap();
    let link = server.accept_subscriber(author(), BTreeMap::new()).unwrap();
    pump(&relay, &wire, &server, &link);
    assert_eq!(refusal(&relay), None, "the relay refused its upstream");
    assert_eq!(wire.refusal(), None);

    let client = relay
        .attach_client(identity(0x02), BTreeMap::new())
        .unwrap();
    client
        .with_db(|db| {
            block_on(db.insert(
                "parents",
                BTreeMap::from([
                    (
                        "title".to_owned(),
                        Value::String("from the relay".to_owned()),
                    ),
                    (
                        "rank".to_owned(),
                        Value::Nullable(Some(Box::new(Value::I64(7)))),
                    ),
                ]),
                InsertOptions {
                    row_id: Some(RowUuid::from_bytes([0x07; 16])),
                    ..Default::default()
                },
            ))
            .map_err(|error| RelayError::Drive(error.to_string()))?;
            // Written past the binding, so tick it here: the drive moves
            // what the tick queued.
            block_on(db.tick()).map_err(|error| RelayError::Drive(error.to_string()))
        })
        .unwrap();
    pump(&relay, &wire, &server, &link);
    assert_eq!(refusal(&relay), None, "the relay refused its upstream");
    // The authority runs under its genesis v1; read the row as v2 stores it.
    let v2_view = block_on(server.db.register_schema_view(v2())).unwrap();
    assert_eq!(
        titles(&v2_view),
        vec![(
            "from the relay".to_owned(),
            Some(Value::Nullable(Some(Box::new(Value::I64(7)))))
        )]
    );
}

/// A relay that cannot mirror its upstream says so on the connection it
/// was refused over, so its carrier stops the socket and reports why
/// instead of keeping a link open that syncs nothing.
#[test]
fn a_relay_that_refuses_its_upstream_reports_it_on_the_wire() {
    let directory = tempfile::tempdir().unwrap();
    let server = authority(&directory.path().join("core.sqlite"));
    let relay = relay(directory.path().join("alice.sqlite"), None);
    let wire = relay.wire().unwrap();
    let link = server.accept_subscriber(author(), BTreeMap::new()).unwrap();
    pump(&relay, &wire, &server, &link);
    let reason = wire.refusal().expect("the refusal reaches the wire");
    assert!(
        reason.contains("stores otherwise than its upstream"),
        "{reason}"
    );
    assert_eq!(refusal(&relay).as_deref(), Some(reason.as_str()));
}
