//! DRAFT switches for the store-thread CPU optimizations.
//!
//! Every optimization is on unless `JAZZ_X_ALL=0` or its own variable is
//! `0`; a shadow check (the old computation run beside the new one) is off
//! unless its variable is `1`. Read once per process by each caller.

/// Whether optimization `name` is on.
pub fn on(name: &str) -> bool {
    let off = |var: &str| std::env::var(var).as_deref() == Ok("0");
    !(off("JAZZ_X_ALL") || off(name))
}

/// Whether shadow check `name` is on.
pub fn shadow(name: &str) -> bool {
    std::env::var(name).as_deref() == Ok("1")
}
