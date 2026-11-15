//! A/B switches for this fork's stand measurements (fork only, not part of
//! any upstream change).
//!
//! A switch is on unless its own variable or `JAZZ_X_ALL` is `0`. Each
//! caller reads its switch once per process.

/// Whether switch `name` is on.
pub fn on(name: &str) -> bool {
    let off = |var: &str| std::env::var(var).as_deref() == Ok("0");
    !(off("JAZZ_X_ALL") || off(name))
}
