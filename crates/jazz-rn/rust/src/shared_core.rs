//! The engine behind its lock, and the one way to it.

#[cfg(test)]
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::{JazzRnError, JsOutbox, RnCoreType, RnScheduler, SchedulerJob};

#[cfg(test)]
thread_local! {
    /// Holds of a core lock by this thread.
    pub(super) static CORE_LOCKS_HELD: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// The engine, behind its lock. The lock is only ever held through a [`CoreGuard`],
/// which is what makes whatever was recorded for JS under it deliverable: the fields
/// are private to this module so that nothing else can take it.
pub(super) struct SharedCore {
    core: Mutex<RnCoreType>,
    outbox: Arc<JsOutbox>,
    scheduler: RnScheduler,
}

impl SharedCore {
    pub(super) fn new(core: RnCoreType, outbox: Arc<JsOutbox>, scheduler: RnScheduler) -> Self {
        Self {
            core: Mutex::new(core),
            outbox,
            scheduler,
        }
    }

    /// For a caller that does not tell JS what its hold recorded: if anything was,
    /// JS is asked to come and take it.
    pub(super) fn lock(&self) -> Result<CoreGuard<'_>, JazzRnError> {
        self.enter(false)
    }

    /// For a caller that tells JS itself once the guard is gone: a call from JS that
    /// drains, the worker's tick.
    pub(super) fn lock_delivering(&self) -> Result<CoreGuard<'_>, JazzRnError> {
        self.enter(true)
    }

    fn enter(&self, delivers: bool) -> Result<CoreGuard<'_>, JazzRnError> {
        let asked = Instant::now();
        #[cfg(test)]
        self.scheduler
            .probe
            .at_the_lock
            .fetch_add(1, Ordering::SeqCst);
        let core = self.core.lock();
        #[cfg(test)]
        self.scheduler
            .probe
            .at_the_lock
            .fetch_sub(1, Ordering::SeqCst);
        let core = core.map_err(|_| JazzRnError::Internal {
            message: "lock poisoned".into(),
        })?;
        // The JS thread and the worker wait for each other here: a call from JS that
        // arrives in the middle of a tick waits for the rest of it. What that costs
        // the JS thread is read off this line.
        let waited = asked.elapsed();
        if waited >= Duration::from_millis(1) {
            tracing::debug!(
                target: "jazz_rn::core_lock",
                waited_micros = u64::try_from(waited.as_micros()).unwrap_or(u64::MAX),
                thread = std::thread::current().name().unwrap_or("js"),
                "waited for the core lock"
            );
        }
        self.outbox.open_section();
        #[cfg(test)]
        CORE_LOCKS_HELD.with(|held| held.set(held.get() + 1));
        Ok(CoreGuard {
            core: Some(core),
            shared: self,
            delivers,
        })
    }
}

#[cfg(test)]
impl Drop for SharedCore {
    fn drop(&mut self) {
        *self
            .scheduler
            .probe
            .engine_dropped_on
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(std::thread::current().id());
    }
}

pub(super) struct CoreGuard<'a> {
    /// `None` only inside `drop`.
    core: Option<std::sync::MutexGuard<'a, RnCoreType>>,
    shared: &'a SharedCore,
    delivers: bool,
}

impl std::ops::Deref for CoreGuard<'_> {
    type Target = RnCoreType;

    fn deref(&self) -> &RnCoreType {
        self.core.as_ref().expect("the guard holds the lock")
    }
}

impl std::ops::DerefMut for CoreGuard<'_> {
    fn deref_mut(&mut self) -> &mut RnCoreType {
        self.core.as_mut().expect("the guard holds the lock")
    }
}

impl Drop for CoreGuard<'_> {
    /// Must not panic: it runs while a panicking tick unwinds.
    fn drop(&mut self) {
        let (recorded, waiting, given_up) =
            self.shared.outbox.close_section(!std::thread::panicking());
        // Released before anybody is woken or asked: neither may find the lock held.
        // Unwinding, this poisons it.
        drop(self.core.take());
        #[cfg(test)]
        CORE_LOCKS_HELD.with(|held| held.set(held.get().saturating_sub(1)));
        drop(given_up);
        for waker in waiting {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| waker.wake()));
        }
        if recorded && !self.delivers {
            self.shared.scheduler.send_job(SchedulerJob::Notify);
        }
    }
}
