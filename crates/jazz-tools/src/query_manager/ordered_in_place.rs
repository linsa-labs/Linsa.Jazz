//! Kill switch and parity mode for updating an ordered list in place.
//!
//! `Project` and `Output` rebuild their whole ordered list from their ordered input
//! on every non-empty input delta: every tuple is cloned out of the input
//! (`ordered_tuples_owned`, recursively through an include tail), `Project`
//! re-projects every row, and both diff the new list against the old one. A write
//! that changes ONE row of a list — the author's own `users` row inside every
//! chat's members include, a draft upsert — pays for the whole list in every
//! subscription that holds it.
//!
//! When the input delta only updates tuples and the ordered input still has the
//! ids the list was built from, in the same order, the rebuild can only replace
//! the updated tuples where they stand. That is what the in-place path does; the
//! id sequence is compared in full every time, so a reorder the delta does not
//! report still takes the rebuild.
//!
//! [`InPlaceMode::Verify`], the default in debug builds, runs the rebuild and
//! checks it against what the in-place path would have produced: a debug build
//! panics on a difference, a release build counts it
//! (`settle_cost::ORDERED_IN_PLACE_PARITY_VIOLATIONS`).
//!
//! `JAZZ_ORDERED_IN_PLACE`: `0` (or `false`) turns it off; `verify` runs `Verify`
//! in any build; unset, a release build updates in place.

use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InPlaceMode {
    /// Update in place.
    Apply,
    /// Rebuild, and check the rebuild against the in-place result.
    Verify,
    /// Rebuild.
    Off,
}

pub(crate) fn in_place_mode() -> InPlaceMode {
    #[cfg(test)]
    if let Some(forced) = test_override::forced() {
        return forced;
    }

    static MODE: OnceLock<InPlaceMode> = OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("JAZZ_ORDERED_IN_PLACE").as_deref() {
        Ok("0") | Ok("false") => InPlaceMode::Off,
        Ok("verify") => InPlaceMode::Verify,
        _ if cfg!(debug_assertions) => InPlaceMode::Verify,
        _ => InPlaceMode::Apply,
    })
}

#[cfg(test)]
mod test_override {
    use super::InPlaceMode;
    use std::sync::atomic::{AtomicU8, Ordering};
    use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};

    const UNSET: u8 = 0;

    static STATE: AtomicU8 = AtomicU8::new(UNSET);

    fn encode(mode: InPlaceMode) -> u8 {
        match mode {
            InPlaceMode::Apply => 1,
            InPlaceMode::Verify => 2,
            InPlaceMode::Off => 3,
        }
    }

    fn lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    /// Holds the in-place mode forced for the guard's lifetime, serialised against every
    /// other forcing test. `set` switches it while held, so one test can run two query
    /// managers in lockstep under two modes. Take it after the settle-skips plug.
    pub struct OrderedInPlaceMode {
        _serialised: MutexGuard<'static, ()>,
    }

    impl OrderedInPlaceMode {
        pub fn set(&self, mode: InPlaceMode) {
            STATE.store(encode(mode), Ordering::SeqCst);
        }
    }

    impl Drop for OrderedInPlaceMode {
        fn drop(&mut self) {
            STATE.store(UNSET, Ordering::SeqCst);
        }
    }

    pub fn force_ordered_in_place(mode: InPlaceMode) -> OrderedInPlaceMode {
        let guard = lock().lock().unwrap_or_else(PoisonError::into_inner);
        STATE.store(encode(mode), Ordering::SeqCst);
        OrderedInPlaceMode { _serialised: guard }
    }

    pub(super) fn forced() -> Option<InPlaceMode> {
        match STATE.load(Ordering::SeqCst) {
            1 => Some(InPlaceMode::Apply),
            2 => Some(InPlaceMode::Verify),
            3 => Some(InPlaceMode::Off),
            _ => None,
        }
    }
}

#[cfg(test)]
pub(crate) use test_override::{OrderedInPlaceMode, force_ordered_in_place};

#[cfg(test)]
thread_local! {
    /// Tuples this thread's `Project`/`Output` nodes rebuilt from their ordered input —
    /// the work an in-place update replaces. Per thread, so a unit test that asserts on it
    /// is not raised by the tests running beside it.
    static REBUILT_TUPLES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn note_rebuilt_tuples(count: usize) {
    REBUILT_TUPLES.with(|rebuilt| rebuilt.set(rebuilt.get() + count as u64));
}

#[cfg(test)]
pub(crate) fn rebuilt_tuples_on_this_thread() -> u64 {
    REBUILT_TUPLES.with(std::cell::Cell::get)
}
