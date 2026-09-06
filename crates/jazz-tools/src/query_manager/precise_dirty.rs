//! Kill switch for F2 row-precise include dirtiness (include-plan-sharing
//! design §5/§9, v13-2).
//!
//! Default ON: reused array-subquery instances receive row-precise changed-id
//! marks (membership via `mark_rows_changed_for_table`, content via
//! `mark_rows_updated`, removals via `mark_rows_deleted`) instead of a
//! `mark_all_dirty` full re-scan per settle, and `reevaluate_all` skips
//! instances whose subgraphs carry no dirt.
//!
//! `JAZZ_PRECISE_DIRTY=0` (or `false`) reverts to the legacy mark-all path:
//! coarse node-level inner dirt, `mark_all_dirty` on every reused instance,
//! and unconditional re-evaluation of every instance. The legacy path keeps its
//! known staleness bugs (inner content updates and nested-include membership
//! changes never reach subscription outputs — see the FINDING in
//! `manager_tests/subscription_output_oracle.rs`); the switch exists so a
//! precise-dirtiness regression can be reverted independently of everything
//! else, not because legacy is correct.
//!
//! The env var is read once; tests use [`force_precise_dirty`] instead to
//! avoid env races — the same plug shape as
//! `row_histories::force_history_fastpath`.

use std::sync::OnceLock;

/// Whether row-precise include dirtiness is enabled.
pub fn precise_dirty_enabled() -> bool {
    #[cfg(any(test, feature = "test"))]
    if let Some(forced) = test_override::forced() {
        return forced;
    }

    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("JAZZ_PRECISE_DIRTY").as_deref(),
            Ok("0") | Ok("false")
        )
    })
}

#[cfg(any(test, feature = "test"))]
mod test_override {
    use std::sync::atomic::{AtomicU8, Ordering};
    use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};

    const UNSET: u8 = 0;
    const FORCED_ON: u8 = 1;
    const FORCED_OFF: u8 = 2;

    static STATE: AtomicU8 = AtomicU8::new(UNSET);

    fn lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    /// Holds precise dirtiness in a forced mode for the guard's lifetime.
    ///
    /// The embedded mutex guard serialises every test that forces a mode, so
    /// force-users cannot observe each other's override. Tests that do NOT
    /// force a mode may still observe a forced-OFF window from a parallel
    /// force-user; tests that DEPEND on precise behavior must hold a
    /// `force_precise_dirty(true)` guard. Dropping restores the unforced default.
    ///
    /// The reassurance this doc used to offer — "that window reverts to the legacy path, which
    /// every pre-existing suite is green under" — is FALSE for any test the window only partly
    /// covers, and that is the dangerous case. A write inside the window marks a graph node
    /// dirty without buffering instance dirt (`array_subquery::note_inner_rows_changed` returns
    /// early when precise is off); a settle after the window consults the instance dirt and
    /// finds none. Neither half is a legacy run — the signal is simply lost between two
    /// mechanisms. `manager_tests/settle_budget.rs` lost twenty subscriptions this way and read
    /// as an intermittent liveness bug in an unrelated feature for two oracle rounds.
    pub struct PreciseDirtyMode {
        _serialised: MutexGuard<'static, ()>,
    }

    impl Drop for PreciseDirtyMode {
        fn drop(&mut self) {
            STATE.store(UNSET, Ordering::SeqCst);
        }
    }

    /// Force precise dirtiness on or off for the returned guard's lifetime.
    pub fn force_precise_dirty(enabled: bool) -> PreciseDirtyMode {
        let guard = lock().lock().unwrap_or_else(PoisonError::into_inner);
        STATE.store(
            if enabled { FORCED_ON } else { FORCED_OFF },
            Ordering::SeqCst,
        );
        PreciseDirtyMode { _serialised: guard }
    }

    pub(super) fn forced() -> Option<bool> {
        match STATE.load(Ordering::SeqCst) {
            FORCED_ON => Some(true),
            FORCED_OFF => Some(false),
            _ => None,
        }
    }
}

#[cfg(any(test, feature = "test"))]
pub use test_override::{PreciseDirtyMode, force_precise_dirty};
