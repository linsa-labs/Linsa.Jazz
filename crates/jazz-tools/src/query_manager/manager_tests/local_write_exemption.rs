//! Read-your-own-writes must expire. A row this node authored keeps its exemption
//! from server-scope filtering forever, which turns every read into a read of this
//! node's own possibly-stale replica.
//!
//! MEASURED end to end on the live stack (2026-08-17), `unique_names` row
//! `01a0116b-4641-76d1-874b-87a72296fa22` on branch `dev-b32dae47bbd9-main`:
//!
//! ```text
//!   rpc-server store   1 version, VisibleDirect, idx:…:userId = 1
//!   jazz-sync store    2 versions, winner deleted=true kind=Soft,
//!                      idx:…:_id_deleted = 1, _id/userId/uniqueName = 0
//!   handler reading at tier 'edge'   claimants=1 owned=1 outcome="alreadyOwned"
//!   after restarting the rpc process claimants=0 owned=0 outcome="taken"
//! ```
//!
//! Nothing but the process restart changed — the store still held the row. What the
//! restart cleared is `QueryManager::pending_local_row_batches`, an in-memory map filled
//! by every local write (`writes.rs` → `manager.rs`, `local_update = true`) and consulted
//! by `filter_synced_query_scope_tuples`, which keeps a tuple unconditionally when its id
//! is in that map — before it ever looks at the remote scope.
//!
//! The map was cleared only by an INBOUND update for the same object and branch, and none
//! reached the author: `inbox.rs` forwards to everyone EXCEPT the originating client,
//! `forwarding.rs` additionally requires scope membership, which a node holding no live
//! subscription never has, and a copy of its own batch sent back to it is a no-op. So for a
//! node whose reads are one-shot —
//! the rpc-server's whole backend facade — the exemption never expires, and every row it
//! has ever written is frozen in its own reads for the process lifetime.
//!
//! Product consequence, measured: every uniqueness check reads the node's own replica
//! while believing it asked the server. That is `uniqueName.take`, email binding and the
//! apple-identity lookup.
//!
//! What must NOT break while fixing it is pinned by the second test here: a row that is
//! not yet durable must stay visible to its author even when the server's scope is empty.
//! That is the contract the exemption exists for, and six tests elsewhere in this crate
//! already depend on it.

use super::*;

/// The author's node: no durability identity of its own, one upstream server.
///
/// `server_tier` is the tier that server declares, and it is a parameter because the
/// topology is load-bearing. A jazz server declares `GlobalServer` only when it has NO
/// upstream (`server/builder.rs`); put an edge in front of it and it declares
/// `EdgeServer`, and every `QuerySettled` it emits carries that instead — the tier in a
/// settle is the emitting server's own `max_local_durability_tier()`, never the reader's
/// requested one. A fixture that hardcodes `GlobalServer` therefore tests exactly one
/// deployment shape, and a retirement rule written against the wrong quantity looks
/// correct throughout all of it.
fn author_and_server_at(
    server_tier: DurabilityTier,
) -> (
    QueryManager,
    MemoryStorage,
    QueryManager,
    MemoryStorage,
    crate::sync_manager::ClientId,
    crate::sync_manager::ServerId,
) {
    let (mut client, client_io) = create_query_manager(SyncManager::new(), test_schema());
    let (mut server, server_io) = create_query_manager(
        SyncManager::new().with_durability_tier(server_tier),
        test_schema(),
    );

    let client_id = crate::sync_manager::ClientId::new();
    let server_id = crate::sync_manager::ServerId::new();
    connect_server(&mut client, &client_io, server_id);
    connect_client(&mut server, &server_io, client_id);

    (client, client_io, server, server_io, client_id, server_id)
}

fn author_and_server() -> (
    QueryManager,
    MemoryStorage,
    QueryManager,
    MemoryStorage,
    crate::sync_manager::ClientId,
    crate::sync_manager::ServerId,
) {
    author_and_server_at(DurabilityTier::GlobalServer)
}

/// Carry ONLY `QuerySettled` from server to client — the scope refresh without the row
/// frame. This models the measured divergence: the author's store still holds its own
/// copy while the server has already removed it.
fn pump_scope_only(
    client: &mut QueryManager,
    server: &mut QueryManager,
    client_io: &mut MemoryStorage,
    server_io: &mut MemoryStorage,
    client_id: crate::sync_manager::ClientId,
    server_id: crate::sync_manager::ServerId,
) -> Vec<(u64, DurabilityTier)> {
    pump(
        client, server, client_io, server_io, client_id, server_id, false,
    )
}

/// The same exchange with the author's rows allowed through, so the server learns them and
/// its scope grows to contain them. Without this the two reasons a tuple can survive the
/// filter — the server's scope covers it, or the author is exempt for it — are never in
/// tension, and a rule that exempts far too much is indistinguishable from a correct one.
fn pump_publishing_rows(
    client: &mut QueryManager,
    server: &mut QueryManager,
    client_io: &mut MemoryStorage,
    server_io: &mut MemoryStorage,
    client_id: crate::sync_manager::ClientId,
    server_id: crate::sync_manager::ServerId,
) -> Vec<(u64, DurabilityTier)> {
    pump(
        client, server, client_io, server_io, client_id, server_id, true,
    )
}

fn pump(
    client: &mut QueryManager,
    server: &mut QueryManager,
    client_io: &mut MemoryStorage,
    server_io: &mut MemoryStorage,
    client_id: crate::sync_manager::ClientId,
    server_id: crate::sync_manager::ServerId,
    publish_rows: bool,
) -> Vec<(u64, DurabilityTier)> {
    use crate::sync_manager::{Destination, InboxEntry, Source, SyncPayload};

    let mut settles_delivered: Vec<(u64, DurabilityTier)> = Vec::new();
    for _ in 0..10 {
        let client_outbox = client.sync_manager_mut().take_outbox();
        for entry in client_outbox.into_iter().filter(|e| {
            matches!(e.destination, Destination::Server(id) if id == server_id)
                // The author's ROW is withheld: the server must answer the subscription
                // without knowing it. That is the measured divergence — on the live stack
                // the server had it and somebody else deleted it; either way its scope
                // does not contain the row while the author's store still does.
                && (publish_rows
                    || !matches!(
                        e.payload,
                        SyncPayload::RowBatchCreated { .. } | SyncPayload::RowBatchNeeded { .. }
                    ))
        }) {
            server.sync_manager_mut().push_inbox(InboxEntry {
                source: Source::Client(client_id),
                payload: entry.payload,
            });
        }
        server.process(server_io);

        let server_outbox = server.sync_manager_mut().take_outbox();
        for entry in server_outbox.into_iter().filter(|e| {
            matches!(e.destination, Destination::Client(id) if id == client_id)
                && matches!(e.payload, SyncPayload::QuerySettled { .. })
        }) {
            if let SyncPayload::QuerySettled { query_id, tier, .. } = &entry.payload {
                settles_delivered.push((query_id.0, *tier));
            }
            client.sync_manager_mut().push_inbox(InboxEntry {
                source: Source::Server(server_id),
                payload: entry.payload,
            });
        }
        client.process(client_io);
    }
    settles_delivered
}

/// THE GATE.
///
/// The author writes a row, the write becomes durable at the settlement target, and the
/// server's scope does not contain it. A read at a tier the server answers must reflect
/// the server's answer: once the write is durable, the author's own copy is no longer
/// evidence that the row exists.
///
/// The fixture deliberately differs from the guard below by ONE statement — the injected
/// batch fate. That is the whole invariant: durability, and nothing else, is what must
/// end the exemption.
///
/// Two fixture notes, so nothing here is mistaken for more than it is. The fate is pushed
/// directly, the way `batch_fate_processing_does_not_scan_visible_regions_to_find_members`
/// does, standing in for the confirmation the live stack produces — there every backend
/// write goes through `.wait({ tier: 'global' })` before its handler returns. And the
/// server's scope is empty because it never learned the row; on the live stack it was
/// empty because somebody else had deleted it. The filter cannot tell those apart: it
/// tests membership, nothing else.
#[test]
fn a_durable_local_write_loses_its_read_your_own_writes_exemption() {
    let (mut client, mut client_io, mut server, mut server_io, client_id, server_id) =
        author_and_server();

    let query = client.query("users").build();
    let sub = client
        .subscribe_with_sync(query, None, Some(DurabilityTier::GlobalServer))
        .expect("the author reads at a tier the server answers");

    let handle = client
        .insert(
            &mut client_io,
            "users",
            &[Value::Text("Alice".into()), Value::Integer(100)],
        )
        .expect("the author writes the row");
    client.process(&mut client_io);

    // THE ONE DIFFERENCE from the guard: the write reaches the settlement target.
    client.sync_manager_mut().push_pending_batch_fate(
        crate::batch_fate::BatchFate::DurableDirect {
            batch_id: handle.batch_id,
            confirmed_tier: DurabilityTier::GlobalServer,
        },
    );
    client.process(&mut client_io);

    let _ = pump_scope_only(
        &mut client,
        &mut server,
        &mut client_io,
        &mut server_io,
        client_id,
        server_id,
    );

    let delivered: usize = client
        .take_updates()
        .into_iter()
        .filter(|update| update.subscription_id == sub)
        .map(|update| update.delta.added.len())
        .sum();
    assert_eq!(
        delivered, 0,
        "a locally authored row must lose its read-your-own-writes exemption once it is \
         durable at the settlement target: from then on the server's scope is the authority \
         on whether the row still exists, and the author's own copy is not. Measured live \
         as `claimants=1 outcome=alreadyOwned` from a handler reading at tier 'edge', which \
         flipped to `claimants=0 outcome=taken` on nothing but a process restart — the \
         restart cleared exactly this exemption."
    );
}

/// THE GUARD — what the fix must not break.
///
/// The same shape with the write NOT durable. Here the exemption is load-bearing: the
/// server has no reason to know the row yet, its scope is legitimately empty, and the
/// author must still see what it just wrote. Six tests elsewhere in this crate depend on
/// this contract; this one states it next to the gate so a fix cannot satisfy one by
/// sacrificing the other.
#[test]
fn a_local_write_that_is_not_yet_durable_stays_visible_to_its_author() {
    let (mut client, mut client_io, mut server, mut server_io, client_id, server_id) =
        author_and_server();

    let query = client.query("users").build();
    let sub = client
        .subscribe_with_sync(query, None, Some(DurabilityTier::GlobalServer))
        .expect("subscribe before writing");

    client
        .insert(
            &mut client_io,
            "users",
            &[Value::Text("Bob".into()), Value::Integer(200)],
        )
        .expect("the author writes the row");
    client.process(&mut client_io);

    // Only the scope travels: the server never learns of the row, so its scope stays
    // empty — exactly the state the exemption exists to survive.
    let _ = pump_scope_only(
        &mut client,
        &mut server,
        &mut client_io,
        &mut server_io,
        client_id,
        server_id,
    );

    let delivered: usize = client
        .take_updates()
        .into_iter()
        .filter(|update| update.subscription_id == sub)
        .map(|update| update.delta.added.len())
        .sum();
    assert_eq!(
        delivered, 1,
        "an author must keep seeing its own not-yet-durable write even when the server's \
         scope is empty — this is the contract the exemption exists for, and bounding it \
         must not cost this"
    );
}

/// THE DIFFERENTIAL ORACLE.
///
/// The two tests above are hand-written cases: they encode what we already thought of.
/// This one runs randomised operation sequences against a *model* of the intended
/// semantics, stated independently of the implementation:
///
/// > A locally authored row is delivered to its author if the server's scope contains it,
/// > or if it has not yet BOTH become durable at the settlement target AND been followed
/// > by a scope refresh. Once both have happened, the server's answer accounts for the
/// > write and the author's own copy stops being evidence.
///
/// The fixture withholds the author's rows from the server, so the remote scope never
/// contains them and the model reduces to the second clause. What varies is the
/// interleaving: writes, confirmations at the target tier, confirmations BELOW it (which
/// must not end the exemption), and scope refreshes — in every order the generator finds.
///
/// The model is driven by what the transport actually delivered, not by what the operation
/// intended: `pump_scope_only` returns how many `QuerySettled` frames really reached the
/// author, and a refresh that did not land does not advance the model. An oracle that
/// assumed its own ops took effect would report failures the implementation never had.
struct Lcg(u64);

impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// One row as the model sees it. Durability is a property of the WRITE and so lives here;
/// whether the exemption has been spent is a property of each READER and lives on `Reader`.
struct ModelRow {
    row_id: ObjectId,
    batch_id: BatchId,
    /// Confirmed at or above `settlement_target()`.
    durable: bool,
}

/// One subscription and everything the model knows about what it has been told.
struct Reader {
    sub: crate::query_manager::QuerySubscriptionId,
    tier: DurabilityTier,
    /// Rows delivered to this reader so far, accumulated from its deltas.
    observed: std::collections::HashSet<ObjectId>,
    /// A sync-backed subscription emits nothing before its first authoritative settle —
    /// the engine refuses to answer with a snapshot it knows is only local.
    ever_settled: bool,
    /// Rows whose exemption THIS reader has spent: durable when an answer this reader can
    /// trust arrived.
    spent: std::collections::HashSet<ObjectId>,
}

#[test]
fn randomised_write_confirm_refresh_sequences_match_the_intended_exemption_semantics() {
    const SEEDS: u64 = 60;
    const OPS_PER_SEED: usize = 40;

    // The server's declared tier and the readers' required tiers vary INDEPENDENTLY,
    // because that is where the semantics live and a single pair hides them. A settle
    // carries the emitting server's tier, so `EdgeServer` is the deployment behind an
    // edge, and a reader asking for `GlobalServer` there is asking for more authority than
    // anything attached can attest — its answer must never arrive, and its own copy must
    // never quietly stand in for one.
    //
    // TWO readers at DIFFERENT tiers share every process, which is the part one reader
    // cannot express: the exemption is consumed per subscription, so any rule that picks a
    // single bar for the whole process — the write-side settlement target, the maximum
    // tier across readers, anything global — pins one of these two readers wrongly. With
    // one reader every such rule looks correct.
    for server_tier in [DurabilityTier::GlobalServer, DurabilityTier::EdgeServer] {
        for seed in 1..=SEEDS {
            let mut rng = Lcg(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let (mut client, mut client_io, mut server, mut server_io, client_id, server_id) =
                author_and_server_at(server_tier);

            let mut readers: Vec<Reader> =
                [DurabilityTier::EdgeServer, DurabilityTier::GlobalServer]
                    .into_iter()
                    .map(|tier| {
                        let query = client.query("users").build();
                        Reader {
                            sub: client
                                .subscribe_with_sync(query, None, Some(tier))
                                .expect("the author subscribes at the tier under test"),
                            tier,
                            observed: std::collections::HashSet::new(),
                            ever_settled: false,
                            spent: std::collections::HashSet::new(),
                        }
                    })
                    .collect();

            let mut model: Vec<ModelRow> = Vec::new();
            let mut history: Vec<String> = Vec::new();

            for step in 0..OPS_PER_SEED {
                // Every op returns the settles it actually caused, per query and with the
                // tier that arrived. The model is driven by those rather than by what the
                // op intended: a frame that did not arrive, or arrived below the tier a
                // reader requires, is not an answer to that reader.
                let settles: Vec<(u64, DurabilityTier)> = match rng.below(10) {
                    // Write a row. Withheld from the server unless a later publish op lets
                    // it through, so the exemption is the only reason it can be visible.
                    0..=3 => {
                        let inserted = client
                            .insert(
                                &mut client_io,
                                "users",
                                &[
                                    Value::Text(format!("row-{seed}-{step}")),
                                    Value::Integer(step as i32),
                                ],
                            )
                            .expect("the author writes the row");
                        client.process(&mut client_io);
                        history.push(format!("insert {}", inserted.row_id));
                        model.push(ModelRow {
                            row_id: inserted.row_id,
                            batch_id: inserted.batch_id,
                            durable: false,
                        });
                        Vec::new()
                    }
                    // Confirm at the settlement target: what must eventually end the
                    // exemption, once an answer follows it.
                    4..=5 if !model.is_empty() => {
                        let index = rng.below(model.len());
                        client.sync_manager_mut().push_pending_batch_fate(
                            crate::batch_fate::BatchFate::DurableDirect {
                                batch_id: model[index].batch_id,
                                confirmed_tier: DurabilityTier::GlobalServer,
                            },
                        );
                        client.process(&mut client_io);
                        history.push(format!("confirm@global {}", model[index].row_id));
                        // A ratchet, deliberately: `spent` is never cleared. Re-confirming
                        // a batch that is already durable must not resurrect an exemption
                        // the reader has already spent — the first oracle run asserted it
                        // did, and the engine was right to disagree.
                        model[index].durable = true;
                        Vec::new()
                    }
                    // Confirm BELOW the settlement target: the write has not reached the
                    // tier its author reconciles against, so nothing may change.
                    6 if !model.is_empty() => {
                        let index = rng.below(model.len());
                        client.sync_manager_mut().push_pending_batch_fate(
                            crate::batch_fate::BatchFate::DurableDirect {
                                batch_id: model[index].batch_id,
                                confirmed_tier: DurabilityTier::EdgeServer,
                            },
                        );
                        client.process(&mut client_io);
                        history.push(format!("confirm@edge {}", model[index].row_id));
                        Vec::new()
                    }
                    // A `Missing` fate: the batch is not confirmed and not rejected, it
                    // pends retransmission. `confirmed_tier()` is `None`, so nothing about
                    // the exemption may move — this op exists so that stops being true
                    // silently. (`Rejected` is not modelled here: the retraction it drives
                    // lives in `RuntimeCore`, and is gated by
                    // `runtime_core::tests::rejected_write_retires_tracking`.)
                    7 if !model.is_empty() => {
                        let index = rng.below(model.len());
                        client.sync_manager_mut().push_pending_batch_fate(
                            crate::batch_fate::BatchFate::Missing {
                                batch_id: model[index].batch_id,
                            },
                        );
                        client.process(&mut client_io);
                        history.push(format!("missing {}", model[index].row_id));
                        Vec::new()
                    }
                    // Publish: let the rows written so far reach the server, so its scope
                    // grows to contain them. From then on those rows are visible for a
                    // reason unrelated to the exemption, and a rule that exempts far too
                    // much stops being indistinguishable from a correct one.
                    8 => {
                        let settles = pump_publishing_rows(
                            &mut client,
                            &mut server,
                            &mut client_io,
                            &mut server_io,
                            client_id,
                            server_id,
                        );
                        history.push(format!("publish (settles {})", settles.len()));
                        settles
                    }
                    _ => {
                        let settles = pump_scope_only(
                            &mut client,
                            &mut server,
                            &mut client_io,
                            &mut server_io,
                            client_id,
                            server_id,
                        );
                        history.push(format!("refresh (settles {})", settles.len()));
                        settles
                    }
                };

                for reader in &mut readers {
                    let answered = settles
                        .iter()
                        .any(|(query_id, tier)| *query_id == reader.sub.0 && *tier >= reader.tier);
                    if answered {
                        reader.ever_settled = true;
                        for row in model.iter().filter(|row| row.durable) {
                            reader.spent.insert(row.row_id);
                        }
                    }
                }

                for update in client.take_updates() {
                    let Some(reader) = readers
                        .iter_mut()
                        .find(|reader| reader.sub == update.subscription_id)
                    else {
                        continue;
                    };
                    for row in &update.delta.added {
                        reader.observed.insert(row.id);
                    }
                    for row in &update.delta.removed {
                        reader.observed.remove(&row.id);
                    }
                }

                for reader in &readers {
                    // The server's scope is the filter's INPUT, so the model reads it
                    // rather than predicting it: what is modelled here is the decision the
                    // filter makes given a scope, not the sync protocol that produces one.
                    let remote_scope = client.sync_manager().remote_query_scope_at_least(
                        crate::sync_manager::QueryId(reader.sub.0),
                        reader.tier,
                    );
                    let expected: std::collections::HashSet<ObjectId> = if reader.ever_settled {
                        model
                            .iter()
                            .filter(|row| {
                                remote_scope
                                    .iter()
                                    .any(|(scoped_id, _)| *scoped_id == row.row_id)
                                    || !reader.spent.contains(&row.row_id)
                            })
                            .map(|row| row.row_id)
                            .collect()
                    } else {
                        std::collections::HashSet::new()
                    };

                    assert_eq!(
                        reader.observed,
                        expected,
                        "server={server_tier:?} reader={:?} seed {seed} step {step}: this \
                         reader's delivered set diverged from the intended exemption \
                         semantics. Each reader's exemption is spent by an answer IT can \
                         trust; a rule that picks one bar for the whole process gets one of \
                         these two wrong.\nhistory:\n  {}",
                        reader.tier,
                        history.join("\n  ")
                    );
                }
            }
        }
    }
}

/// Losing a server must not retire an exemption.
///
/// The two-phase release exists so a write stops being exempt at the moment the server's
/// answer accounts for it. `SyncManager::remove_server` raises `remote_query_scope_dirty`
/// for every query the departing server was answering — which is the opposite event: the
/// node has just learned LESS, not more. Keying the release on that flag would drop the
/// exemption exactly when the node has least reason to trust anyone else's scope, and the
/// author would go blind to its own durable write.
///
/// Two servers are load-bearing. With one, dropping it also drops the last remote scope
/// snapshot and `filter_synced_query_scope_tuples` stops filtering at all — the bug is
/// masked, not absent. With two, the survivor's scope keeps driving the filter, and a
/// release triggered by the departure is directly observable as the row disappearing.
#[test]
fn losing_a_server_does_not_retire_a_parked_exemption() {
    let (mut client, client_io) = create_query_manager(SyncManager::new(), test_schema());
    let (mut leaving, leaving_io, mut staying, staying_io) = {
        let (leaving, leaving_io) = create_query_manager(
            SyncManager::new().with_durability_tier(DurabilityTier::GlobalServer),
            test_schema(),
        );
        let (staying, staying_io) = create_query_manager(
            SyncManager::new().with_durability_tier(DurabilityTier::GlobalServer),
            test_schema(),
        );
        (leaving, leaving_io, staying, staying_io)
    };
    let (mut client_io, mut leaving_io, mut staying_io) = (client_io, leaving_io, staying_io);

    let client_id = crate::sync_manager::ClientId::new();
    let leaving_id = crate::sync_manager::ServerId::new();
    let staying_id = crate::sync_manager::ServerId::new();
    connect_server(&mut client, &client_io, leaving_id);
    connect_server(&mut client, &client_io, staying_id);
    connect_client(&mut leaving, &leaving_io, client_id);
    connect_client(&mut staying, &staying_io, client_id);

    let query = client.query("users").build();
    let sub = client
        .subscribe_with_sync(query, None, Some(DurabilityTier::GlobalServer))
        .expect("the author reads at a tier both servers answer");

    let handle = client
        .insert(
            &mut client_io,
            "users",
            &[Value::Text("Carol".into()), Value::Integer(300)],
        )
        .expect("the author writes the row");
    client.process(&mut client_io);

    // Both servers answer the subscription, so a remote scope snapshot exists and the
    // filter is live. Neither learns the row.
    for _ in 0..3 {
        let _ = pump_scope_only(
            &mut client,
            &mut leaving,
            &mut client_io,
            &mut leaving_io,
            client_id,
            leaving_id,
        );
        let _ = pump_scope_only(
            &mut client,
            &mut staying,
            &mut client_io,
            &mut staying_io,
            client_id,
            staying_id,
        );
    }

    // The write becomes durable: the exemption is now parked, waiting for a scope
    // snapshot that accounts for it.
    client.sync_manager_mut().push_pending_batch_fate(
        crate::batch_fate::BatchFate::DurableDirect {
            batch_id: handle.batch_id,
            confirmed_tier: DurabilityTier::GlobalServer,
        },
    );
    client.process(&mut client_io);
    let _ = client.take_updates();

    // The ONLY event from here is the departure. No scope snapshot follows it.
    client.sync_manager_mut().remove_server(leaving_id);
    client.process(&mut client_io);
    client.process(&mut client_io);

    assert!(
        client
            .confirmed_local_rows_awaiting_scope
            .contains_key(&handle.row_id),
        "losing a server must not retire the author's parked exemption: a departure is the \
         node learning less, not the server's answer arriving. Measured both ways — with \
         the release keyed on an applied settle the row stays parked; keyed on \
         `remote_query_scope_dirty` it is retired by the departure alone, because \
         `remove_server` raises that flag for every query the departing server answered. \
         The assertion is on the exemption rather than on a delta because the loss is \
         latent: no delta is emitted in this pass either way, and the author only goes \
         blind at the next recompute, with the surviving server's scope still driving the \
         filter."
    );
}

/// THE GATE, behind an edge.
///
/// Same invariant as `a_durable_local_write_loses_its_read_your_own_writes_exemption`, in
/// the deployment shape that has an edge server between the author and the global one: the
/// server declares `EdgeServer`, the author reads at `EdgeServer`, and every `QuerySettled`
/// the author ever receives carries `EdgeServer`.
///
/// This is a topology, not a corner case — `server/builder.rs` declares `EdgeServer` for
/// any jazz server started with an upstream — and it is the shape in which a retirement
/// rule keyed on `SyncManager::settlement_target()` silently does nothing. That target is
/// `GlobalServer` for any node with a server attached, whatever that server can attest, so
/// the comparison `EdgeServer < GlobalServer` holds forever and the exemption stays exactly
/// as unbounded as it was before it was split. Every other test in this file hands the
/// server `GlobalServer`, so none of them can see it.
#[test]
fn a_durable_local_write_loses_its_exemption_behind_an_edge_server_too() {
    let (mut client, mut client_io, mut server, mut server_io, client_id, server_id) =
        author_and_server_at(DurabilityTier::EdgeServer);

    let query = client.query("users").build();
    let sub = client
        .subscribe_with_sync(query, None, Some(DurabilityTier::EdgeServer))
        .expect("the author reads at the tier this server answers");

    let handle = client
        .insert(
            &mut client_io,
            "users",
            &[Value::Text("Dave".into()), Value::Integer(400)],
        )
        .expect("the author writes the row");
    client.process(&mut client_io);

    // The ASYMMETRY this gate exists for. The write still reaches `GlobalServer` — an edge
    // forwards it upstream and the global confirmation comes back — so phase one, which is
    // about the write, is right to measure it against `settlement_target()`. What never
    // reaches `GlobalServer` is the SETTLE: it carries the emitting server's own tier, and
    // the server the author is connected to is the edge. Phase two must therefore be keyed
    // on what the readers require, not on the write-side target.
    client.sync_manager_mut().push_pending_batch_fate(
        crate::batch_fate::BatchFate::DurableDirect {
            batch_id: handle.batch_id,
            confirmed_tier: DurabilityTier::GlobalServer,
        },
    );
    client.process(&mut client_io);

    let settles = pump_scope_only(
        &mut client,
        &mut server,
        &mut client_io,
        &mut server_io,
        client_id,
        server_id,
    );
    assert!(
        !settles.is_empty(),
        "fixture precondition: the edge server must answer the subscription, else this \
         gates nothing"
    );

    let delivered: usize = client
        .take_updates()
        .into_iter()
        .filter(|update| update.subscription_id == sub)
        .map(|update| update.delta.added.len())
        .sum();
    assert_eq!(
        delivered, 0,
        "an author reading at `EdgeServer` from a server that declares `EdgeServer` must \
         lose its exemption on that server's answer. The settle carries the emitting \
         server's own tier, so behind an edge there is no `GlobalServer` settle to wait \
         for — a rule that requires one waits forever and the read stays stale for the \
         process lifetime, which is the defect this file exists for."
    );
}

/// Only a `DurableDirect` fate at `GlobalServer` takes a direct write out of the source
/// overlay. The overlay is shared by readers of every tier, so a lower tier must leave the
/// row in it; and a transactional fate leaves it by its own path, the visibility change of
/// the rows its sealed submission names, so its kind must not release a direct batch here.
///
/// Checked on the map itself, with the fate pushed straight into the sync manager, so each
/// rule is held by a case of its own rather than by an end-to-end exchange that happens not
/// to produce the fate. Two of the cases need no server at all: with none attached the
/// write-side target is the node's own tier, and a release keyed on that target instead of
/// on the top tier would let an `EdgeServer` fate through.
#[test]
fn only_a_direct_fate_at_the_top_tier_releases_a_direct_write_from_the_overlay() {
    let cases: [(fn(BatchId) -> crate::batch_fate::BatchFate, bool, &str); 3] = [
        (
            |batch_id| crate::batch_fate::BatchFate::AcceptedTransaction {
                batch_id,
                confirmed_tier: DurabilityTier::GlobalServer,
            },
            true,
            "a transactional fate at the top tier, for a direct batch",
        ),
        (
            |batch_id| crate::batch_fate::BatchFate::DurableDirect {
                batch_id,
                confirmed_tier: DurabilityTier::EdgeServer,
            },
            true,
            "a direct fate below the top tier, with a server attached",
        ),
        (
            |batch_id| crate::batch_fate::BatchFate::DurableDirect {
                batch_id,
                confirmed_tier: DurabilityTier::EdgeServer,
            },
            false,
            "a direct fate below the top tier, with no server attached",
        ),
    ];

    for (fate_for, with_server, case) in cases {
        let (mut client, mut client_io) = if with_server {
            let (client, client_io, _, _, _, _) = author_and_server();
            (client, client_io)
        } else {
            create_query_manager(SyncManager::new(), test_schema())
        };

        let handle = client
            .insert(
                &mut client_io,
                "users",
                &[Value::Text("Erin".into()), Value::Integer(500)],
            )
            .expect("the author writes the row");
        client.process(&mut client_io);
        assert!(
            client.local_row_tracking(handle.row_id).pending_batch,
            "{case}: fixture precondition: a local write enters the overlay"
        );

        client
            .sync_manager_mut()
            .push_pending_batch_fate(fate_for(handle.batch_id));
        client.process(&mut client_io);
        assert!(
            client.local_row_tracking(handle.row_id).pending_batch,
            "{case}: the row must stay in the source overlay. A reader at the top tier \
             cannot see a version confirmed below it, and a transactional fate does not \
             patch a direct batch's rows, so the overlay is the only thing serving the \
             author's write to such a reader."
        );

        // The positive control: the same row, the fate that does release it.
        client.sync_manager_mut().push_pending_batch_fate(
            crate::batch_fate::BatchFate::DurableDirect {
                batch_id: handle.batch_id,
                confirmed_tier: DurabilityTier::GlobalServer,
            },
        );
        client.process(&mut client_io);
        assert!(
            !client.local_row_tracking(handle.row_id).pending_batch,
            "{case}: positive control: a direct fate at the top tier releases the row"
        );
    }
}

/// A released row's parked exemption retires on a version of the row from upstream — but
/// only a version on the branch the exemption was parked for. The same id on another branch
/// says nothing about whether the server's scope holds the author's row: rows are per
/// branch, and so is the scope.
///
/// The client reads both branches, so the server sends it the other branch's version. No
/// `QuerySettled` is carried, which is the other way a parked exemption retires; the only
/// event after the release is the foreign version.
#[test]
fn a_version_on_another_branch_leaves_a_released_rows_parked_exemption() {
    use crate::sync_manager::{Destination, InboxEntry, Source, SyncPayload};

    let (mut client, mut client_io, mut server, mut server_io, client_id, server_id) =
        author_and_server();
    let main = get_branch(&client);
    let other = get_branch_for_user_branch(&client, "other");

    let query = client
        .query("users")
        .branches(&[main.as_str(), other.as_str()])
        .build();
    let _sub = client
        .subscribe_with_sync(query, None, Some(DurabilityTier::GlobalServer))
        .expect("the author reads both branches at a tier the server answers");
    let _ = pump_scope_only(
        &mut client,
        &mut server,
        &mut client_io,
        &mut server_io,
        client_id,
        server_id,
    );

    let handle = client
        .insert(
            &mut client_io,
            "users",
            &[Value::Text("Faye".into()), Value::Integer(600)],
        )
        .expect("the author writes the row on the main branch");
    client.process(&mut client_io);
    client.sync_manager_mut().push_pending_batch_fate(
        crate::batch_fate::BatchFate::DurableDirect {
            batch_id: handle.batch_id,
            confirmed_tier: DurabilityTier::GlobalServer,
        },
    );
    client.process(&mut client_io);
    let tracking = client.local_row_tracking(handle.row_id);
    assert!(
        !tracking.pending_batch && tracking.awaiting_scope,
        "fixture precondition: the row is released from the overlay and its exemption \
         parked ({tracking:?})"
    );

    let schema = server.schema_context().current_schema.clone();
    server
        .insert_on_branch_with_schema_and_write_context_and_id(
            &mut server_io,
            "users",
            &other,
            &[Value::Text("Gus".into()), Value::Integer(700)],
            Some(handle.row_id),
            &schema,
            None,
            true,
        )
        .expect("the server writes the same id on the other branch");
    for _ in 0..10 {
        server.process(&mut server_io);
        for entry in server
            .sync_manager_mut()
            .take_outbox()
            .into_iter()
            .filter(|e| {
                matches!(e.destination, Destination::Client(id) if id == client_id)
                    && !matches!(e.payload, SyncPayload::QuerySettled { .. })
            })
        {
            client.sync_manager_mut().push_inbox(InboxEntry {
                source: Source::Server(server_id),
                payload: entry.payload,
            });
        }
        client.process(&mut client_io);
        for entry in client
            .sync_manager_mut()
            .take_outbox()
            .into_iter()
            .filter(|e| {
                matches!(e.destination, Destination::Server(id) if id == server_id)
                    && !matches!(
                        e.payload,
                        SyncPayload::RowBatchCreated { .. } | SyncPayload::RowBatchNeeded { .. }
                    )
            })
        {
            server.sync_manager_mut().push_inbox(InboxEntry {
                source: Source::Client(client_id),
                payload: entry.payload,
            });
        }
    }
    assert_eq!(
        client_io.index_lookup("users", "_id", &other, &Value::Uuid(handle.row_id)),
        vec![handle.row_id],
        "fixture precondition: the other branch's version reached the author"
    );

    assert!(
        client.local_row_tracking(handle.row_id).awaiting_scope,
        "a version of the row on another branch must not retire the exemption parked for \
         the main branch: the server holding the id there is no evidence that its scope \
         holds the author's row here, and with the exemption gone a reader holding a scope \
         older than the write loses the row until the next answer arrives"
    );
}
