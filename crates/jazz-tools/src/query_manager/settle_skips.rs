//! Kill switch and parity mode for the settle skips.
//!
//! A write into one row of a widely included table (a contact's presence beat
//! on `users`) dirties every include registered for that table and everything
//! below it, and `topo_sort_dirty` pulls each dirty node's whole input chain
//! into the order. Two skips keep that from costing the whole graph when the
//! write reached no instance:
//!
//! * a node that is clean, is not a scan, and whose inputs all emitted an
//!   empty delta this pass emits an empty delta without running — re-running
//!   it re-built whole windows (`recompute_tuple_window`) upstream of a dirty
//!   include that changed nothing;
//! * a `Project` or `Output` whose input delta is empty, once the output has
//!   materialized, emits an empty delta without rebuilding the ordered list
//!   (`ordered_tuples_owned` clones every tuple through the include tail).
//!
//! Both rest on one property: an empty input delta means the node's ordered
//! output is unchanged. Scans are excluded because a clean scan still rescans
//! under a local overlay, which is how overlay rows reach it.
//!
//! [`SkipMode::Verify`], the default in debug builds, runs the node anyway and
//! checks it emitted nothing, so the whole test suite checks the property at
//! every point a skip would have fired: a debug build panics on a violation, a
//! release build counts it (`settle_cost::SETTLE_SKIP_PARITY_VIOLATIONS`) and
//! logs it. A test that needs the skip itself (a cost gate, or a differential
//! against [`SkipMode::Off`]) forces it with [`force_settle_skips`].
//!
//! `JAZZ_SETTLE_SKIPS`: `0` (or `false`) turns both skips off; `verify` runs
//! `Verify` in any build, so a release stand can count violations on real
//! traffic before the skips ship; unset, a release build skips.

use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SkipMode {
    /// Skip, emitting an empty delta.
    Skip,
    /// Run the node and check it emitted an empty delta.
    Verify,
    /// Run the node.
    Off,
}

pub(crate) fn settle_skip_mode() -> SkipMode {
    #[cfg(any(test, feature = "test"))]
    if let Some(forced) = test_override::forced() {
        return forced;
    }

    static MODE: OnceLock<SkipMode> = OnceLock::new();
    *MODE.get_or_init(|| {
        let mode = mode_from_env(std::env::var("JAZZ_SETTLE_SKIPS").ok().as_deref());
        // Once per process: a verify run on a stand has to be able to show it verified.
        tracing::info!(target: "jazz::settle_skips", ?mode, "settle skips mode");
        mode
    })
}

/// The first 8 bytes of the SHA-256 of `query`'s JSON, in hex: a name for a query that is the
/// same on every runtime, where a subscription id is local to the one that made it.
pub(crate) fn query_digest(query: &crate::query_manager::query::Query) -> String {
    serde_json::to_string(query)
        .map(|json| {
            use sha2::Digest as _;
            hex::encode(&sha2::Sha256::digest(json.as_bytes())[..8])
        })
        .unwrap_or_default()
}

fn mode_from_env(value: Option<&str>) -> SkipMode {
    match value {
        Some("0") | Some("false") => SkipMode::Off,
        Some("verify") => SkipMode::Verify,
        _ if cfg!(debug_assertions) => SkipMode::Verify,
        _ => SkipMode::Skip,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_environment_names_each_mode() {
        assert_eq!(mode_from_env(Some("verify")), SkipMode::Verify);
        assert_eq!(mode_from_env(Some("0")), SkipMode::Off);
        assert_eq!(mode_from_env(Some("false")), SkipMode::Off);
        let default = if cfg!(debug_assertions) {
            SkipMode::Verify
        } else {
            SkipMode::Skip
        };
        assert_eq!(mode_from_env(None), default);
        assert_eq!(mode_from_env(Some("1")), default);
    }
}

#[cfg(any(test, feature = "test"))]
mod test_override {
    use std::sync::atomic::{AtomicU8, Ordering};
    use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};

    const UNSET: u8 = 0;
    const FORCED_ON: u8 = 1;
    const FORCED_OFF: u8 = 2;
    const FORCED_VERIFY: u8 = 3;

    static STATE: AtomicU8 = AtomicU8::new(UNSET);

    fn lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    /// Holds the settle skips forced on (real skips) or off for the guard's
    /// lifetime, serialised against every other forcing test. Take it after
    /// the precise-dirtiness, routing and empty-probe plugs, in that order.
    pub struct SettleSkipsMode {
        _serialised: MutexGuard<'static, ()>,
    }

    impl Drop for SettleSkipsMode {
        fn drop(&mut self) {
            STATE.store(UNSET, Ordering::SeqCst);
        }
    }

    pub fn force_settle_skips(enabled: bool) -> SettleSkipsMode {
        force(if enabled { FORCED_ON } else { FORCED_OFF })
    }

    /// Holds the skips in `Verify` — what a debug build does anyway — so a test can count
    /// what that mode counts without another test forcing a different mode meanwhile.
    pub fn force_settle_skips_verify() -> SettleSkipsMode {
        force(FORCED_VERIFY)
    }

    fn force(state: u8) -> SettleSkipsMode {
        let guard = lock().lock().unwrap_or_else(PoisonError::into_inner);
        STATE.store(state, Ordering::SeqCst);
        SettleSkipsMode { _serialised: guard }
    }

    pub(super) fn forced() -> Option<super::SkipMode> {
        match STATE.load(Ordering::SeqCst) {
            FORCED_ON => Some(super::SkipMode::Skip),
            FORCED_OFF => Some(super::SkipMode::Off),
            FORCED_VERIFY => Some(super::SkipMode::Verify),
            _ => None,
        }
    }
}

#[cfg(any(test, feature = "test"))]
pub use test_override::{SettleSkipsMode, force_settle_skips, force_settle_skips_verify};
