use super::*;
use crate::query_manager::manager::LocalUpdates;
use crate::sync_manager::QueryPropagation;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadDurabilityOptions {
    pub tier: Option<DurabilityTier>,
    pub local_updates: LocalUpdates,
}

impl Default for ReadDurabilityOptions {
    fn default() -> Self {
        Self {
            tier: None,
            local_updates: LocalUpdates::Immediate,
        }
    }
}

/// Subscription created but not yet executed (2-phase subscribe).
pub(super) struct PendingSubscription {
    pub query: Query,
    pub session: Option<Session>,
    pub durability: ReadDurabilityOptions,
    pub propagation: QueryPropagation,
}

impl<S: Storage, Sch: Scheduler> RuntimeCore<S, Sch> {
    fn allocate_subscription_handle(&mut self) -> SubscriptionHandle {
        let handle = SubscriptionHandle(self.next_subscription_handle);
        self.next_subscription_handle += 1;
        handle
    }

    fn subscribe_query(
        &mut self,
        query: Query,
        session: Option<Session>,
        durability: ReadDurabilityOptions,
        propagation: QueryPropagation,
    ) -> Result<QuerySubscriptionId, RuntimeError> {
        self.schema_manager
            .query_manager_mut()
            .subscribe_with_sync_and_propagation_with_local_updates(
                query,
                session,
                durability.tier,
                durability.local_updates,
                propagation,
            )
            .map_err(|e| RuntimeError::QueryError(e.to_string()))
    }

    fn activate_subscription(
        &mut self,
        handle: SubscriptionHandle,
        query_sub_id: QuerySubscriptionId,
        callback: SubscriptionCallback,
    ) {
        self.subscriptions.insert(
            handle,
            SubscriptionState {
                query_sub_id,
                callback,
            },
        );
        self.subscription_reverse.insert(query_sub_id, handle);
        // v18 item 3 (half B): the registration leaves in this call, not on the next batched
        // tick, which would have to win the engine lock behind whatever pass is running.
        self.flush_runtime_outbox("flushing the registration from the subscribe call");
        self.immediate_tick();
    }

    // =========================================================================
    // Subscriptions
    // =========================================================================

    /// Subscribe to a query with a callback.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn subscribe<F>(
        &mut self,
        query: Query,
        callback: F,
        session: Option<Session>,
    ) -> Result<SubscriptionHandle, RuntimeError>
    where
        F: Fn(SubscriptionDelta) + Send + 'static,
    {
        self.subscribe_with_durability_and_propagation(
            query,
            callback,
            session,
            ReadDurabilityOptions::default(),
            QueryPropagation::Full,
        )
    }

    /// Subscribe to a query with a callback (WASM version - no Send required).
    #[cfg(target_arch = "wasm32")]
    pub fn subscribe<F>(
        &mut self,
        query: Query,
        callback: F,
        session: Option<Session>,
    ) -> Result<SubscriptionHandle, RuntimeError>
    where
        F: Fn(SubscriptionDelta) + 'static,
    {
        self.subscribe_with_durability_and_propagation(
            query,
            callback,
            session,
            ReadDurabilityOptions::default(),
            QueryPropagation::Full,
        )
    }

    /// Subscribe with explicit durability and propagation options.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn subscribe_with_durability_and_propagation<F>(
        &mut self,
        query: Query,
        callback: F,
        session: Option<Session>,
        durability: ReadDurabilityOptions,
        propagation: QueryPropagation,
    ) -> Result<SubscriptionHandle, RuntimeError>
    where
        F: Fn(SubscriptionDelta) + Send + 'static,
    {
        self.subscribe_impl(query, Box::new(callback), session, durability, propagation)
    }

    /// Subscribe with explicit durability and propagation options (WASM version).
    #[cfg(target_arch = "wasm32")]
    pub fn subscribe_with_durability_and_propagation<F>(
        &mut self,
        query: Query,
        callback: F,
        session: Option<Session>,
        durability: ReadDurabilityOptions,
        propagation: QueryPropagation,
    ) -> Result<SubscriptionHandle, RuntimeError>
    where
        F: Fn(SubscriptionDelta) + 'static,
    {
        self.subscribe_impl(query, Box::new(callback), session, durability, propagation)
    }

    /// Internal subscribe implementation.
    fn subscribe_impl(
        &mut self,
        query: Query,
        callback: SubscriptionCallback,
        session: Option<Session>,
        durability: ReadDurabilityOptions,
        propagation: QueryPropagation,
    ) -> Result<SubscriptionHandle, RuntimeError> {
        let _span = debug_span!(
            "subscribe",
            table = query.table.as_str(),
            ?durability.tier,
            local_updates = ?durability.local_updates
        )
        .entered();
        let query_sub_id = self.subscribe_query(query, session, durability, propagation)?;
        let handle = self.allocate_subscription_handle();
        debug!(handle = handle.0, sub_id = query_sub_id.0, "subscribed");
        self.activate_subscription(handle, query_sub_id, callback);
        Ok(handle)
    }

    // =========================================================================
    // Two-phase subscribe: create + execute
    // =========================================================================

    /// Phase 1: allocate a handle and store query params. No compilation, no
    /// sync, no tick — just bookkeeping.
    pub fn create_subscription(
        &mut self,
        query: Query,
        session: Option<Session>,
        durability: ReadDurabilityOptions,
        propagation: QueryPropagation,
    ) -> SubscriptionHandle {
        let handle = self.allocate_subscription_handle();
        debug!(
            handle = handle.0,
            table = query.table.as_str(),
            "subscription created (pending)"
        );
        self.pending_subscriptions.insert(
            handle,
            PendingSubscription {
                query,
                session,
                durability,
                propagation,
            },
        );
        handle
    }

    /// Phase 2: compile graph, register with QueryManager, sync to servers,
    /// attach callback, and run `immediate_tick` to deliver the first delta.
    ///
    /// No-ops silently if the handle was already unsubscribed between create
    /// and execute.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn execute_subscription<F>(
        &mut self,
        handle: SubscriptionHandle,
        callback: F,
    ) -> Result<(), RuntimeError>
    where
        F: Fn(SubscriptionDelta) + Send + 'static,
    {
        self.execute_subscription_impl(handle, Box::new(callback))
    }

    /// Phase 2 (WASM version — no Send required).
    #[cfg(target_arch = "wasm32")]
    pub fn execute_subscription<F>(
        &mut self,
        handle: SubscriptionHandle,
        callback: F,
    ) -> Result<(), RuntimeError>
    where
        F: Fn(SubscriptionDelta) + 'static,
    {
        self.execute_subscription_impl(handle, Box::new(callback))
    }

    fn execute_subscription_impl(
        &mut self,
        handle: SubscriptionHandle,
        callback: SubscriptionCallback,
    ) -> Result<(), RuntimeError> {
        let Some(pending) = self.pending_subscriptions.remove(&handle) else {
            return Ok(());
        };

        let _span = debug_span!(
            "execute_subscription",
            handle = handle.0,
            table = pending.query.table.as_str(),
            ?pending.durability.tier,
            local_updates = ?pending.durability.local_updates
        )
        .entered();

        let query_sub_id = self.subscribe_query(
            pending.query,
            pending.session,
            pending.durability,
            pending.propagation,
        )?;

        debug!(
            handle = handle.0,
            sub_id = query_sub_id.0,
            "subscription executed"
        );
        self.activate_subscription(handle, query_sub_id, callback);
        Ok(())
    }

    /// Unsubscribe from a query. Works for both pending (created but not
    /// executed) and active subscriptions.
    pub fn unsubscribe(&mut self, handle: SubscriptionHandle) {
        if self.pending_subscriptions.remove(&handle).is_some() {
            debug!(handle = handle.0, "unsubscribed pending subscription");
            return;
        }
        if let Some(state) = self.subscriptions.remove(&handle) {
            self.subscription_reverse.remove(&state.query_sub_id);
            self.schema_manager
                .query_manager_mut()
                .unsubscribe_with_sync(state.query_sub_id);
            // The unsubscription is in the outbox; nothing else is obliged to tick, and a
            // server holds the registration (and counts it against the client) until the
            // frame lands.
            if self.has_outbound() {
                self.scheduler.schedule_batched_tick();
            }
        }
    }

    // =========================================================================
    // Queries
    // =========================================================================

    /// Execute a one-shot query.
    pub fn query(&mut self, query: Query, session: Option<Session>) -> QueryFuture {
        self.query_with_propagation(
            query,
            session,
            ReadDurabilityOptions::default(),
            QueryPropagation::Full,
        )
    }

    pub fn query_with_propagation(
        &mut self,
        query: Query,
        session: Option<Session>,
        durability: ReadDurabilityOptions,
        propagation: QueryPropagation,
    ) -> QueryFuture {
        self.query_with_overlay_rows(query, session, durability, propagation, HashMap::new())
            .1
    }

    /// Like [`Self::query_with_local_batch`], but also hands back the handle of the
    /// one-shot entry so the caller can cancel it with
    /// [`Self::cancel_one_shot_query`] when its own deadline fires.
    pub fn query_with_local_batch_tracked(
        &mut self,
        query: Query,
        session: Option<Session>,
        durability: ReadDurabilityOptions,
        propagation: QueryPropagation,
        batch_id: Option<BatchId>,
    ) -> Result<(SubscriptionHandle, QueryFuture), RuntimeError> {
        let Some(batch_id) = batch_id else {
            return Ok(self.query_with_overlay_rows(
                query,
                session,
                durability,
                propagation,
                HashMap::new(),
            ));
        };

        self.ensure_batch_is_open(batch_id)?;

        match self.batch_query_overlay(batch_id)? {
            Some(overlay) => Ok(self.query_with_local_overlay_tracked(
                query,
                session,
                durability,
                propagation,
                overlay,
            )),
            None => Ok(self.query_with_overlay_rows(
                query,
                session,
                durability,
                propagation,
                HashMap::new(),
            )),
        }
    }

    /// Cancel a one-shot query that has not settled yet.
    ///
    /// Returns `true` when an entry was still pending and has now been released;
    /// `false` when the query already completed or failed (nothing to release).
    pub fn cancel_one_shot_query(&mut self, handle: SubscriptionHandle) -> bool {
        let Some(mut pending) = self.pending_one_shot_queries.remove(&handle) else {
            return false;
        };
        if let Some(sender) = pending.sender.take() {
            let _ = sender.send(Err(RuntimeError::QueryCancelled));
        }
        self.subscription_reverse.remove(&pending.subscription_id);
        // Tolerates a subscription the query manager already dropped (rejection or
        // recompile failure between its `process` and the tick that would have cleaned
        // this entry): `unsubscribe_with_sync` only tells the servers when it removed
        // something.
        self.schema_manager
            .query_manager_mut()
            .unsubscribe_with_sync(pending.subscription_id);
        debug!(handle = handle.0, "cancelled pending one-shot query");
        if self.has_outbound() {
            self.scheduler.schedule_batched_tick();
        }
        true
    }

    pub fn query_with_local_batch(
        &mut self,
        query: Query,
        session: Option<Session>,
        durability: ReadDurabilityOptions,
        propagation: QueryPropagation,
        batch_id: Option<BatchId>,
    ) -> Result<QueryFuture, RuntimeError> {
        self.query_with_local_batch_tracked(query, session, durability, propagation, batch_id)
            .map(|(_, future)| future)
    }

    /// The untracked form, used only by `runtime_core/tests.rs`, which wants the future and not
    /// the handle. `#[cfg(test)]` rather than a bare `pub(crate)`: a plain `cargo build` warns it
    /// is never used — which is true of a shipping build and was very nearly why I deleted it.
    #[cfg(test)]
    pub(crate) fn query_with_local_overlay(
        &mut self,
        query: Query,
        session: Option<Session>,
        durability: ReadDurabilityOptions,
        propagation: QueryPropagation,
        overlay: QueryLocalOverlay,
    ) -> QueryFuture {
        self.query_with_local_overlay_tracked(query, session, durability, propagation, overlay)
            .1
    }

    fn query_with_local_overlay_tracked(
        &mut self,
        query: Query,
        session: Option<Session>,
        durability: ReadDurabilityOptions,
        propagation: QueryPropagation,
        overlay: QueryLocalOverlay,
    ) -> (SubscriptionHandle, QueryFuture) {
        let local_overlay_rows = if overlay.row_ids.is_empty() {
            HashMap::new()
        } else {
            overlay
                .row_ids
                .into_iter()
                .map(|row_id| {
                    (
                        row_id,
                        crate::sync_manager::RowBatchKey::new(
                            row_id,
                            overlay.branch_name,
                            overlay.batch_id,
                        ),
                    )
                })
                .collect()
        };
        self.query_with_overlay_rows(query, session, durability, propagation, local_overlay_rows)
    }

    fn query_with_overlay_rows(
        &mut self,
        query: Query,
        session: Option<Session>,
        durability: ReadDurabilityOptions,
        propagation: QueryPropagation,
        local_overlay_rows: HashMap<ObjectId, crate::sync_manager::RowBatchKey>,
    ) -> (SubscriptionHandle, QueryFuture) {
        let _span = debug_span!(
            "query",
            table = query.table.as_str(),
            ?durability.tier,
            local_updates = ?durability.local_updates
        )
        .entered();
        let (sender, receiver) = oneshot::channel();

        let sub_id = match self
            .schema_manager
            .query_manager_mut()
            .subscribe_with_sync_and_propagation_with_local_overlay(
                query,
                session,
                durability.tier,
                crate::query_manager::subscriptions::SubscriptionExecutionOptions {
                    local_updates: durability.local_updates,
                    propagation,
                    local_overlay_rows,
                },
            ) {
            Ok(id) => id,
            Err(e) => {
                let _ = sender.send(Err(RuntimeError::QueryError(e.to_string())));
                let handle = SubscriptionHandle(self.next_subscription_handle);
                self.next_subscription_handle += 1;
                return (handle, QueryFuture::new(receiver));
            }
        };

        let handle = SubscriptionHandle(self.next_subscription_handle);
        self.next_subscription_handle += 1;

        self.pending_one_shot_queries.insert(
            handle,
            PendingOneShotQuery {
                subscription_id: sub_id,
                sender: Some(sender),
            },
        );
        self.subscription_reverse.insert(sub_id, handle);

        // v18 item 3 (half B): see `activate_subscription`.
        self.flush_runtime_outbox("flushing the registration from the query call");
        self.immediate_tick();
        (handle, QueryFuture::new(receiver))
    }
}
