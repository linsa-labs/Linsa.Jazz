use super::*;
use crate::catalogue::CatalogueEntry;
use crate::object::{BranchName, ObjectId};
use crate::query_manager::types::SharedString;
use crate::row_histories::StoredRowBatch;
use crate::storage::{RowLocator, metadata_from_row_locator};
use std::collections::HashMap;

type RowSyncData = (SharedString, HashMap<String, String>, StoredRowBatch);

/// The two independent facts about taking in a catalogue entry.
///
/// They used to be one `bool`. That is how a schema generation could sit on
/// disk and stay unknown to the process running over it: "storage already holds
/// these bytes" was read as "the schema layer has already been told about
/// them", and after the first restart following a migration it never was.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CatalogueEntryIntake {
    /// Storage now holds bytes it did not hold before. Gates forwarding onward:
    /// a restart must not re-broadcast a catalogue nobody changed.
    pub(super) storage_changed: bool,
    /// The schema layer has not been handed this exact entry yet. Gates the push
    /// into `pending_catalogue_updates`.
    pub(super) needs_schema_layer: bool,
}

/// Stable digest over everything `CatalogueEntry`'s own equality compares.
/// Metadata is a `HashMap`, so its keys are sorted before hashing — two nodes
/// must agree on the digest of identical entries.
fn catalogue_digest(entry: &CatalogueEntry) -> [u8; 32] {
    let mut keys: Vec<&String> = entry.metadata.keys().collect();
    keys.sort();

    let mut hasher = blake3::Hasher::new();
    hasher.update(entry.object_id.to_string().as_bytes());
    for key in keys {
        hasher.update(key.as_bytes());
        hasher.update(b"\0");
        hasher.update(entry.metadata[key].as_bytes());
        hasher.update(b"\0");
    }
    hasher.update(&entry.content);
    *hasher.finalize().as_bytes()
}

/// Decide whether a stored row is still eligible for upstream row replay.
///
/// A rejected fate is terminal for the row submission, while a missing fate
/// remains replayable. Rows already confirmed above this node's tier are also
/// skipped because an upstream server already has them.
fn should_replay_row_to_server(
    row: &StoredRowBatch,
    authoritative_fate: Option<&BatchFate>,
    my_max_tier: Option<DurabilityTier>,
) -> bool {
    if authoritative_fate.is_some_and(BatchFate::is_rejected) {
        return false;
    }

    let effective_confirmed_tier = row
        .confirmed_tier
        .or_else(|| authoritative_fate.and_then(BatchFate::confirmed_tier));
    match (effective_confirmed_tier, my_max_tier) {
        (None, _) => true,
        (Some(_), None) => false,
        (Some(row_tier), Some(local_tier)) => row_tier <= local_tier,
    }
}

impl SyncManager {
    pub(super) fn scope_delivery_row(mut row: StoredRowBatch) -> StoredRowBatch {
        if row.state.is_visible() {
            row.parents.clear();
        }
        row
    }

    pub(super) fn queue_catalogue_sync_to_server_from_storage<H: Storage>(
        &mut self,
        server_id: ServerId,
        storage: &H,
    ) {
        let Ok(entries) = storage.scan_catalogue_entries() else {
            return;
        };
        for entry in entries {
            self.catalogue_entries
                .insert(entry.object_id, entry.clone());
            self.queue_catalogue_entry_to_server(server_id, entry);
        }
    }

    /// Queue all existing objects to sync to a new server using storage as the
    /// source of truth for row history and current state.
    #[allow(dead_code)]
    pub(super) fn queue_full_sync_to_server_from_storage<H: Storage>(
        &mut self,
        server_id: ServerId,
        storage: &H,
    ) {
        let _span =
            tracing::debug_span!("queue_full_sync_to_server_from_storage", %server_id).entered();
        let Ok(row_locators) = storage.scan_row_locators() else {
            return;
        };
        let authoritative_fates = match storage.scan_authoritative_batch_fates() {
            Ok(fates) => fates,
            Err(error) => {
                tracing::error!(
                    %server_id,
                    %error,
                    "failed to load authoritative batch fates; aborting full server replay"
                );
                return;
            }
        };
        let authoritative_fates: HashMap<BatchId, BatchFate> = authoritative_fates
            .into_iter()
            .map(|fate| (fate.batch_id(), fate))
            .collect();

        let mut row_sync: Vec<RowSyncData> = Vec::new();

        for (object_id, row_locator) in row_locators {
            self.collect_row_sync_versions(
                storage,
                object_id,
                &row_locator,
                &authoritative_fates,
                &mut row_sync,
            );
        }

        for (table, metadata, row) in row_sync {
            self.queue_row_to_server_with_missing_parents(
                storage,
                table.as_str(),
                server_id,
                &metadata,
                row,
                Some(&authoritative_fates),
            );
        }
    }

    fn collect_row_sync_versions<H: Storage>(
        &self,
        storage: &H,
        object_id: ObjectId,
        row_locator: &RowLocator,
        authoritative_fates: &HashMap<BatchId, BatchFate>,
        row_sync: &mut Vec<RowSyncData>,
    ) {
        let metadata = metadata_from_row_locator(row_locator);
        let Ok(rows) = storage.scan_history_row_batches(row_locator.table.as_str(), object_id)
        else {
            return;
        };
        let my_max_tier = self.max_local_durability_tier();

        for row in rows.into_iter() {
            if !should_replay_row_to_server(
                &row,
                authoritative_fates.get(&row.batch_id),
                my_max_tier,
            ) {
                continue;
            }
            row_sync.push((row_locator.table.clone(), metadata.clone(), row));
        }
    }

    pub(super) fn queue_catalogue_sync_to_client_from_storage<H: Storage>(
        &mut self,
        client_id: ClientId,
        storage: &H,
    ) {
        let Ok(entries) = storage.scan_catalogue_entries() else {
            return;
        };

        for entry in entries {
            self.catalogue_entries
                .insert(entry.object_id, entry.clone());
            self.queue_catalogue_entry_to_client(client_id, entry);
        }
    }

    pub fn upsert_catalogue_entry<H: Storage>(&mut self, storage: &mut H, entry: CatalogueEntry) {
        let intake = self.persist_catalogue_entry(storage, entry.clone());
        if !intake.storage_changed {
            return;
        }

        self.forward_catalogue_entry_to_servers(entry.clone());
        self.forward_catalogue_entry_to_clients(entry, None);
    }

    pub(super) fn persist_catalogue_entry<H: Storage>(
        &mut self,
        storage: &mut H,
        entry: CatalogueEntry,
    ) -> CatalogueEntryIntake {
        let needs_schema_layer =
            self.handed_to_schema_layer.get(&entry.object_id) != Some(&catalogue_digest(&entry));

        let existing = self
            .catalogue_entries
            .get(&entry.object_id)
            .cloned()
            .or_else(|| storage.load_catalogue_entry(entry.object_id).ok().flatten());

        if existing.as_ref() == Some(&entry) {
            self.catalogue_entries.insert(entry.object_id, entry);
            return CatalogueEntryIntake {
                storage_changed: false,
                needs_schema_layer,
            };
        }

        if let Err(error) = storage.upsert_catalogue_entry(&entry) {
            tracing::warn!(
                object_id = %entry.object_id,
                %error,
                "failed to persist catalogue entry"
            );
        }

        self.catalogue_entries.insert(entry.object_id, entry);
        CatalogueEntryIntake {
            storage_changed: true,
            needs_schema_layer,
        }
    }

    /// Record that this exact entry has been handed to the schema layer, so a
    /// peer re-offering it on every connect does not re-enter
    /// `process_catalogue_update` forever.
    pub(super) fn mark_handed_to_schema_layer(&mut self, entry: &CatalogueEntry) {
        self.handed_to_schema_layer
            .insert(entry.object_id, catalogue_digest(entry));
    }

    fn queue_catalogue_entry_to_server(&mut self, server_id: ServerId, entry: CatalogueEntry) {
        self.outbox.push(OutboxEntry {
            destination: Destination::Server(server_id),
            payload: SyncPayload::CatalogueEntryUpdated { entry },
        });
    }

    fn queue_catalogue_entry_to_client(&mut self, client_id: ClientId, entry: CatalogueEntry) {
        self.outbox.push(OutboxEntry {
            destination: Destination::Client(client_id),
            payload: SyncPayload::CatalogueEntryUpdated { entry },
        });
    }

    pub(super) fn forward_catalogue_entry_to_servers(&mut self, entry: CatalogueEntry) {
        let server_ids: Vec<_> = self.servers.keys().copied().collect();
        for server_id in server_ids {
            self.queue_catalogue_entry_to_server(server_id, entry.clone());
        }
    }

    pub(super) fn forward_catalogue_entry_to_clients(
        &mut self,
        entry: CatalogueEntry,
        except: Option<ClientId>,
    ) {
        let client_ids: Vec<_> = self
            .clients
            .keys()
            .copied()
            .filter(|client_id| except != Some(*client_id))
            .collect();
        for client_id in client_ids {
            self.queue_catalogue_entry_to_client(client_id, entry.clone());
        }
    }

    pub(super) fn queue_row_to_server_with_metadata(
        &mut self,
        server_id: ServerId,
        object_id: ObjectId,
        metadata: &HashMap<String, String>,
        row: StoredRowBatch,
        include_metadata: bool,
    ) {
        if metadata
            .get(crate::metadata::MetadataKey::NoSync.as_str())
            .map(|v| v == "true")
            .unwrap_or(false)
        {
            return;
        }

        let branch_name = BranchName::new(&row.branch);
        let batch_id = row.batch_id;
        let batch_already_sent = {
            let Some(server) = self.servers.get(&server_id) else {
                return;
            };
            server
                .sent_batch_ids
                .get(&(object_id, branch_name))
                .is_some_and(|sent| sent.contains(&batch_id))
        };

        if batch_already_sent && !include_metadata {
            return;
        }

        let Some(server) = self.servers.get_mut(&server_id) else {
            return;
        };
        if include_metadata {
            server.sent_metadata.insert(object_id);
        }
        server
            .sent_batch_ids
            .entry((object_id, branch_name))
            .or_default()
            .record_delivery(batch_id, &row.parents);

        self.outbox.push(OutboxEntry {
            destination: Destination::Server(server_id),
            payload: SyncPayload::RowBatchCreated {
                metadata: include_metadata.then(|| RowMetadata {
                    id: object_id,
                    metadata: metadata.clone(),
                }),
                row,
            },
        });
    }

    pub(super) fn queue_initial_row_to_client_with_storage<H: Storage + ?Sized>(
        &mut self,
        storage: &H,
        client_id: ClientId,
        object_id: ObjectId,
        branch_name: BranchName,
        force_resend: bool,
    ) -> Option<BatchId> {
        let row_locator = storage.load_row_locator(object_id).ok().flatten()?;
        let metadata = metadata_from_row_locator(&row_locator);
        if let Some(row) =
            self.load_current_row_from_storage(storage, object_id, &branch_name, &row_locator)
        {
            let batch_id = row.batch_id;
            self.queue_row_to_client(client_id, object_id, metadata, row, force_resend);
            return Some(batch_id);
        }

        None
    }

    pub(super) fn queue_row_to_client(
        &mut self,
        client_id: ClientId,
        object_id: ObjectId,
        metadata: HashMap<String, String>,
        row: StoredRowBatch,
        force_resend: bool,
    ) {
        // Capture parent ids before scope stripping: the delivery copy drops
        // them for visible rows, but the frontier cursor prunes by them.
        let parent_ids = row.parents.clone();
        let row = Self::scope_delivery_row(row);
        if metadata
            .get(crate::metadata::MetadataKey::NoSync.as_str())
            .map(|v| v == "true")
            .unwrap_or(false)
        {
            return;
        }

        let branch_name = BranchName::new(&row.branch);
        let batch_id = row.batch_id;

        let (in_scope, include_metadata, batch_already_sent) = {
            let Some(client) = self.clients.get(&client_id) else {
                return;
            };
            let in_scope = client.is_in_scope(object_id, &branch_name);
            // A forced resend must carry metadata: `sent_metadata` can claim
            // delivery for payloads that were dropped at the stream layer while
            // the client had no live connection, and a row without metadata is
            // unresolvable (and silently discarded) on a client that never got
            // the locator.
            let include_metadata = force_resend || !client.sent_metadata.contains(&object_id);
            let batch_already_sent = client
                .sent_batch_ids
                .get(&(object_id, branch_name))
                .is_some_and(|sent| sent.contains(&batch_id));
            (in_scope, include_metadata, batch_already_sent)
        };

        if !in_scope {
            return;
        }

        if !force_resend && batch_already_sent && !include_metadata {
            return;
        }

        let acks_deliveries = self
            .clients
            .get(&client_id)
            .map(|client| client.acks_deliveries)
            .unwrap_or(false);
        if !self.clients.contains_key(&client_id) {
            return;
        }
        if !acks_deliveries {
            // Old client: it will never confirm, so waiting for it would leave every row
            // outstanding and re-offer the scope on every subscription. Record the claim as
            // before — this is exactly today's behaviour, defect included, and no worse.
            if let Some(client) = self.clients.get_mut(&client_id) {
                if include_metadata {
                    client.sent_metadata.insert(object_id);
                }
                client
                    .sent_batch_ids
                    .entry((object_id, branch_name))
                    .or_default()
                    .record_delivery(batch_id, &parent_ids);
            }
        } else {
            // The claim waits for the receiver. Recording it here — as this did — makes a
            // dropped payload indistinguishable from a delivered one, and the row is then never
            // offered again. A newer batch for the same row replaces this entry, because a
            // re-offer can only ever carry the row's current state.
            tracing::debug!(
                target: "jazz::delivery",
                %client_id, %object_id, ?batch_id,
                "queued, awaiting the receiver's confirmation"
            );
            let now = self.clock.reserve_timestamp();
            let owed = self.pending_client_deliveries.entry(client_id).or_default();
            let (attempts, first_offered_at) = owed
                .get(&(object_id, branch_name))
                .filter(|pending| pending.batch_id == batch_id)
                .map(|pending| (pending.attempts + 1, pending.first_offered_at))
                .unwrap_or((1, now));
            let demoted = attempts > crate::sync_manager::MAX_REDELIVERY_ATTEMPTS
                && now.saturating_sub(first_offered_at)
                    > crate::sync_manager::REDELIVERY_GIVE_UP_AFTER_MICROS;
            if demoted
                && !owed
                    .get(&(object_id, branch_name))
                    .is_some_and(|pending| pending.demoted)
            {
                tracing::warn!(
                    %client_id, %object_id, ?batch_id, attempts,
                    "row keeps going unconfirmed; it no longer forces re-derivations but \
                     stays owed"
                );
            }
            {
                owed.insert(
                    (object_id, branch_name),
                    crate::sync_manager::types::PendingDelivery {
                        attempts,
                        first_offered_at,
                        demoted,
                        batch_id,
                        branch_name,
                        metadata: metadata.clone(),
                        parent_ids: parent_ids.to_vec(),
                        include_metadata,
                    },
                );
            }
        }
        self.row_batch_interest.insert(
            RowBatchKey::new(object_id, branch_name, batch_id),
            client_id,
        );

        self.outbox.push(OutboxEntry {
            destination: Destination::Client(client_id),
            payload: SyncPayload::RowBatchNeeded {
                metadata: include_metadata.then_some(RowMetadata {
                    id: object_id,
                    metadata,
                }),
                row,
            },
        });
    }
}
