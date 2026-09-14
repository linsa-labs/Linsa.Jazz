use super::*;
use crate::batch_fate::BatchFate;
use crate::object::{BranchName, ObjectId};
use crate::row_histories::{BatchId, HistoryScan, StoredRowBatch};
use crate::storage::{RowLocator, Storage, metadata_from_row_locator};
use std::collections::HashSet;
use uuid::Uuid;

/// Debug-only sentinel for the structural guarantee of
/// [`SyncManager::queue_row_to_server_with_missing_parents`]: the row it was
/// called with is enqueued on EVERY exit path.
///
/// A post-loop `debug_assert!` cannot express this, because the exit that caused
/// defect 28 was a `return` from inside the walk, which jumps straight over any
/// code placed after it. Checking on drop is what makes the guarantee hold for
/// exits that have not been written yet.
#[cfg(debug_assertions)]
struct TipMustBeQueued<'a> {
    queued: &'a std::cell::Cell<bool>,
    object_id: ObjectId,
    branch_name: BranchName,
    tip_batch_id: BatchId,
}

#[cfg(debug_assertions)]
impl Drop for TipMustBeQueued<'_> {
    fn drop(&mut self) {
        // Never convert an unrelated panic into a double panic, which aborts.
        if std::thread::panicking() || self.queued.get() {
            return;
        }
        panic!(
            "queue_row_to_server_with_missing_parents left without queueing its own row \
             (row {}, branch {}, batch {:?}) — an ancestor-level decision may prune the \
             descent, never skip the tip (defect 28)",
            self.object_id, self.branch_name, self.tip_batch_id
        );
    }
}

/// The `(code, reason)` of a terminal rejection, or `None` for any other fate.
///
/// Owned rather than borrowed so the caller can probe a fate it loaded into a
/// temporary as cheaply as one it holds in a map, and pay the two clones only
/// on the rare rejected branch.
fn rejection_detail(fate: &BatchFate) -> Option<(String, String)> {
    match fate {
        BatchFate::Rejected { code, reason, .. } => Some((code.clone(), reason.clone())),
        BatchFate::DurableDirect { .. }
        | BatchFate::AcceptedTransaction { .. }
        | BatchFate::Missing { .. } => None,
    }
}

impl SyncManager {
    fn object_has_upstream_confirmation<H: Storage>(
        &self,
        storage: &H,
        table: &str,
        branch_name: &BranchName,
        object_id: ObjectId,
    ) -> bool {
        let Ok(Some(visible_entry)) =
            storage.load_visible_region_entry(table, branch_name.as_str(), object_id)
        else {
            return false;
        };
        let local_tier = self.max_local_durability_tier();
        match local_tier {
            None => {
                visible_entry.current_row.confirmed_tier.is_some()
                    || visible_entry.worker_batch_id.is_some()
                    || visible_entry.edge_batch_id.is_some()
                    || visible_entry.global_batch_id.is_some()
            }
            Some(my_tier) => {
                visible_entry
                    .current_row
                    .confirmed_tier
                    .is_some_and(|row_tier| row_tier > my_tier)
                    || matches!(my_tier, DurabilityTier::Local)
                        && (visible_entry.edge_batch_id.is_some()
                            || visible_entry.global_batch_id.is_some())
                    || matches!(my_tier, DurabilityTier::EdgeServer)
                        && visible_entry.global_batch_id.is_some()
            }
        }
    }

    fn queue_row_to_server_with_storage<H: Storage>(
        &mut self,
        storage: &H,
        table: &str,
        server_id: ServerId,
        object_id: ObjectId,
        metadata: &HashMap<String, String>,
        row: StoredRowBatch,
    ) {
        let branch_name = BranchName::new(&row.branch);
        let include_metadata = {
            let Some(server) = self.servers.get(&server_id) else {
                return;
            };
            let metadata_already_sent = server.sent_metadata.contains(&object_id);
            if !metadata_already_sent {
                true
            } else {
                !self.object_has_upstream_confirmation(storage, table, &branch_name, object_id)
            }
        };
        self.queue_row_to_server_with_metadata(
            server_id,
            object_id,
            metadata,
            row,
            include_metadata,
        );
    }

    pub(super) fn load_current_row_from_storage<H: crate::storage::Storage + ?Sized>(
        &self,
        storage: &H,
        object_id: ObjectId,
        branch_name: &BranchName,
        row_locator: &RowLocator,
    ) -> Option<StoredRowBatch> {
        let table = row_locator.table.as_str();

        if let Ok(Some(row)) =
            storage.load_visible_region_row(table, branch_name.as_str(), object_id)
        {
            return Some(row);
        }

        storage
            .scan_history_region(
                table,
                branch_name.as_str(),
                HistoryScan::Row { row_id: object_id },
            )
            .ok()?
            .into_iter()
            .filter(|row| row.state.is_visible())
            .max_by_key(|row| (row.updated_at, row.batch_id()))
    }

    pub(super) fn load_current_batch_fate_from_storage<H: crate::storage::Storage + ?Sized>(
        &self,
        storage: &H,
        object_id: ObjectId,
        branch_name: &BranchName,
        row_locator: &RowLocator,
    ) -> Option<BatchFate> {
        let row =
            self.load_current_row_from_storage(storage, object_id, branch_name, row_locator)?;
        if row.branch != branch_name.as_str() {
            return None;
        }
        match row.state {
            crate::row_histories::RowState::VisibleDirect
            | crate::row_histories::RowState::VisibleTransactional => {
                self.load_batch_fate_by_batch_id_from_storage(storage, row.batch_id)
            }
            crate::row_histories::RowState::StagingPending
            | crate::row_histories::RowState::Superseded
            | crate::row_histories::RowState::Rejected => None,
        }
    }

    pub(super) fn load_batch_fate_by_batch_id_from_storage<H: crate::storage::Storage + ?Sized>(
        &self,
        storage: &H,
        batch_id: BatchId,
    ) -> Option<BatchFate> {
        storage
            .load_authoritative_batch_fate(batch_id)
            .ok()
            .flatten()
    }

    pub(super) fn queue_batch_fate_to_client(&mut self, client_id: ClientId, fate: BatchFate) {
        let Some(fate) = self.batch_fate_for_client(client_id, &fate) else {
            return;
        };
        self.queue_batch_fate_to_client_unfiltered(client_id, fate);
    }

    pub(super) fn queue_batch_fate_to_client_unfiltered(
        &mut self,
        client_id: ClientId,
        fate: BatchFate,
    ) {
        self.outbox.push(OutboxEntry {
            destination: Destination::Client(client_id),
            payload: SyncPayload::BatchFate { fate },
        });
    }

    #[cfg(test)]
    pub fn forward_update_to_servers_with_storage<H: crate::storage::Storage>(
        &mut self,
        storage: &H,
        object_id: ObjectId,
        branch_name: BranchName,
    ) {
        let server_ids: Vec<ServerId> = self.servers.keys().copied().collect();
        if !server_ids.is_empty() {
            tracing::trace!(%object_id, %branch_name, servers = server_ids.len(), "forwarding to servers");
        }

        let Some(row_locator) = storage.load_row_locator(object_id).ok().flatten() else {
            return;
        };
        if let Some(row) =
            self.load_current_row_from_storage(storage, object_id, &branch_name, &row_locator)
        {
            let metadata = metadata_from_row_locator(&row_locator);
            self.forward_row_batch_to_servers_with_storage(
                storage,
                row_locator.table.as_str(),
                object_id,
                metadata,
                row,
            );
        }
    }

    pub(crate) fn forward_row_batch_to_servers_with_storage<H: Storage>(
        &mut self,
        storage: &H,
        table: &str,
        object_id: ObjectId,
        metadata: HashMap<String, String>,
        row: StoredRowBatch,
    ) {
        let server_ids: Vec<ServerId> = self.servers.keys().copied().collect();
        if !server_ids.is_empty() {
            tracing::trace!(
                %object_id,
                branch = row.branch.as_str(),
                servers = server_ids.len(),
                "forwarding row batch entry with parent closure to servers"
            );
        }

        for server_id in server_ids {
            self.queue_row_to_server_with_missing_parents(
                storage,
                table,
                server_id,
                &metadata,
                row.clone(),
                None,
            );
        }
    }

    /// Queue `row` to a server, preceded by whatever ancestors that server is
    /// not known to hold.
    ///
    /// Structural guarantee (hard): there is no exit path that skips enqueueing
    /// `row` itself. An ancestor-level decision — the delivered frontier, an
    /// unreadable parent, a terminal rejection — may prune the DESCENT only;
    /// none of them may abandon the walk. A `return` placed inside the loop
    /// re-introduces defect 28, in which one terminal `Rejected` anywhere in a
    /// row's ancestry silently severed that row's outbound sync forever.
    /// [`TipMustBeQueued`] is that guarantee, armed on every exit path.
    pub(super) fn queue_row_to_server_with_missing_parents<H: Storage>(
        &mut self,
        storage: &H,
        table: &str,
        server_id: ServerId,
        metadata: &HashMap<String, String>,
        row: StoredRowBatch,
        authoritative_fates: Option<&HashMap<BatchId, BatchFate>>,
    ) {
        let object_id = row.row_id;
        let branch_name = BranchName::new(&row.branch);
        let tip_batch_id = row.batch_id;
        let tip_was_queued = std::cell::Cell::new(false);
        #[cfg(debug_assertions)]
        let _tip_guard = TipMustBeQueued {
            queued: &tip_was_queued,
            object_id,
            branch_name,
            tip_batch_id,
        };
        let mut visited = HashSet::new();

        // Iterative post-order walk, not recursion: a deep history chain would
        // otherwise put one frame per ancestor on the stack and overflow it.
        let mut stack: Vec<(StoredRowBatch, usize)> = vec![(row, 0)];
        while !stack.is_empty() {
            let descend = {
                let (current, parent_index) = stack.last_mut().expect("stack is non-empty");
                let mut next_parent = None;
                while *parent_index < current.parents.len() {
                    let parent_batch_id = current.parents[*parent_index];
                    *parent_index += 1;
                    // Borrow rather than clone the sent set. Since fix D1 the
                    // set is the delivered frontier, not the full delivered
                    // history (`SentBatchIds::record_delivery`): for a serial
                    // history the direct parent is always retained, so this
                    // probe terminates the walk in O(1) without touching
                    // storage. A pruned *inner* ancestor (possible when a new
                    // fork branch joins the delivered set below the frontier)
                    // misses here and the walk descends and re-queues batches
                    // the peer already has — the safe, idempotent direction —
                    // once per new branch tip, after which that tip is in the
                    // frontier and the branch is O(1) again.
                    if self
                        .servers
                        .get(&server_id)
                        .and_then(|server| server.sent_batch_ids.get(&(object_id, branch_name)))
                        .is_some_and(|sent| sent.contains(&parent_batch_id))
                    {
                        continue;
                    }
                    let parent_rejection = match authoritative_fates {
                        Some(fates) => fates.get(&parent_batch_id).and_then(rejection_detail),
                        None => storage
                            .load_authoritative_batch_fate(parent_batch_id)
                            .ok()
                            .flatten()
                            .as_ref()
                            .and_then(rejection_detail),
                    };
                    // A terminal rejection prunes the DESCENT ONLY. Withholding the
                    // rejected ancestor itself stays right — the authority denied it, so
                    // resending it only earns the same denial — but its descendants are
                    // fresh writes the authority has never judged, and they must still go.
                    //
                    // Upstream f6d4412b9a ("withhold children of rejected parents") turned
                    // this `continue` into a `return`, which abandoned the whole walk
                    // including the tip: one rejection anywhere in a row's ancestry
                    // severed that row's outbound sync permanently and silently (defect
                    // 28). That was defensible when upstream wrote it, because an
                    // authority then DROPPED a child whose parent it did not know.
                    //
                    // DEPENDENCY — this fix is only safe against an authority that PARKS
                    // such a child instead: `sync_manager/inbox.rs`
                    // `park_failed_row_batch` keeps a batch that failed with
                    // `ParentNotFound`, and `request_missing_ancestor` asks the sender for
                    // the ancestor (v16.15, `da3a7541e`). Against a pre-v16.15 authority
                    // the descendant we send here is dropped with no repair protocol.
                    if let Some((code, reason)) = parent_rejection {
                        // Never silent: the gap between "written locally" and "missing
                        // upstream" is what made defect 28 invisible for days.
                        tracing::warn!(
                            target: "jazz::sync",
                            %server_id,
                            row_id = %object_id,
                            branch_name = %branch_name,
                            batch_id = ?current.batch_id,
                            rejected_ancestor = ?parent_batch_id,
                            rejection_code = %code,
                            rejection_reason = %reason,
                            "withholding a rejected ancestor from a server; its descendants \
                             are still queued and the authority parks them until the gap heals"
                        );
                        continue;
                    }
                    if !visited.insert(parent_batch_id) {
                        continue;
                    }

                    match storage.load_history_row_batch(
                        table,
                        current.branch.as_str(),
                        object_id,
                        parent_batch_id,
                    ) {
                        Ok(Some(parent_row)) => {
                            next_parent = Some(parent_row);
                            break;
                        }
                        Ok(None) => tracing::warn!(
                            %server_id,
                            %object_id,
                            %branch_name,
                            ?parent_batch_id,
                            "missing parent row batch in local storage while queueing row batch to server"
                        ),
                        Err(error) => tracing::warn!(
                            %server_id,
                            %object_id,
                            %branch_name,
                            ?parent_batch_id,
                            %error,
                            "failed to load parent row batch while queueing row batch to server"
                        ),
                    }
                }
                next_parent
            };

            if let Some(parent_row) = descend {
                stack.push((parent_row, 0));
                continue;
            }

            let (current, _) = stack.pop().expect("stack is non-empty");
            if current.batch_id == tip_batch_id {
                tip_was_queued.set(true);
            }
            self.queue_row_to_server_with_storage(
                storage, table, server_id, object_id, metadata, current,
            );
        }
    }

    pub(crate) fn forward_row_batch_to_servers<H: Storage>(
        &mut self,
        storage: &H,
        object_id: ObjectId,
        metadata: HashMap<String, String>,
        row: StoredRowBatch,
    ) {
        let table = metadata
            .get(crate::metadata::MetadataKey::Table.as_str())
            .cloned()
            .or_else(|| {
                storage
                    .load_row_locator(object_id)
                    .ok()
                    .flatten()
                    .map(|locator| locator.table.to_string())
            });

        if let Some(table) = table {
            self.forward_row_batch_to_servers_with_storage(
                storage,
                table.as_str(),
                object_id,
                metadata,
                row,
            );
            return;
        }

        let server_ids: Vec<ServerId> = self.servers.keys().copied().collect();
        if !server_ids.is_empty() {
            tracing::trace!(
                %object_id,
                branch = row.branch.as_str(),
                servers = server_ids.len(),
                "forwarding row batch entry to servers"
            );
        }

        for server_id in server_ids {
            let include_metadata = self
                .servers
                .get(&server_id)
                .is_some_and(|server| !server.sent_metadata.contains(&object_id));
            self.queue_row_to_server_with_metadata(
                server_id,
                object_id,
                &metadata,
                row.clone(),
                include_metadata,
            );
        }
    }

    pub(crate) fn force_row_batch_to_servers(
        &mut self,
        object_id: ObjectId,
        metadata: HashMap<String, String>,
        row: StoredRowBatch,
    ) {
        let branch_name = BranchName::new(&row.branch);
        let batch_id = row.batch_id;
        let server_ids: Vec<ServerId> = self.servers.keys().copied().collect();

        for server_id in server_ids {
            if let Some(server) = self.servers.get_mut(&server_id) {
                server.sent_metadata.remove(&object_id);
                if let Some(sent_batches) = server.sent_batch_ids.get_mut(&(object_id, branch_name))
                {
                    sent_batches.remove(&batch_id);
                    if sent_batches.is_empty() {
                        server.sent_batch_ids.remove(&(object_id, branch_name));
                    }
                }
            }

            self.queue_row_to_server_with_metadata(
                server_id,
                object_id,
                &metadata,
                row.clone(),
                true,
            );
        }
    }

    pub(crate) fn forward_update_to_clients_with_storage(
        &mut self,
        storage: &impl crate::storage::Storage,
        object_id: ObjectId,
        branch_name: BranchName,
    ) {
        self.forward_update_to_clients_except_with_storage(
            storage,
            object_id,
            branch_name,
            ClientId(Uuid::nil()),
        );
    }

    pub(super) fn forward_update_to_clients_except_with_storage<H: crate::storage::Storage>(
        &mut self,
        storage: &H,
        object_id: ObjectId,
        branch_name: BranchName,
        except: ClientId,
    ) {
        let client_ids: Vec<ClientId> = self
            .clients
            .iter()
            .filter(|(id, client)| **id != except && client.is_in_scope(object_id, &branch_name))
            .map(|(id, _)| *id)
            .collect();
        // Every local write comes through here, one per keystroke on a client runtime, so a node
        // with no client holding the row stops before reading storage.
        if client_ids.is_empty() {
            return;
        }

        let _span = tracing::debug_span!("forward_update_to_clients", %object_id, %branch_name, client_count = client_ids.len()).entered();

        let Some(row_locator) = storage.load_row_locator(object_id).ok().flatten() else {
            return;
        };
        if let Some(row) =
            self.load_current_row_from_storage(storage, object_id, &branch_name, &row_locator)
        {
            let metadata = metadata_from_row_locator(&row_locator);
            for client_id in &client_ids {
                tracing::trace!(%client_id, "queuing row update to client");
                self.queue_row_to_client(
                    *client_id,
                    object_id,
                    metadata.clone(),
                    row.clone(),
                    false,
                );
                if let Some(settlement) = self.load_current_batch_fate_from_storage(
                    storage,
                    object_id,
                    &branch_name,
                    &row_locator,
                ) {
                    self.queue_batch_fate_to_client(*client_id, settlement);
                }
            }
        }
    }
}
