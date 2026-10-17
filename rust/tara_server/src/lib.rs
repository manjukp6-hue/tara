//! TARA Server library crate.
//!
//! Exposes all server subsystems as public modules so they can be used
//! from the binary entry point and from integration tests.

// Central coordinators and engines
pub mod brain;
pub mod server;
pub mod subsystems;

// Systematic Subsystem Categories
pub mod cognitive;
pub mod control_plane;
pub mod engines;
pub mod knowledge;
pub mod language;
pub mod learning;
pub mod memory;
pub mod reward;
pub mod routes;
pub mod rules;
pub mod runtime;
pub mod skills;
pub mod voice;
pub mod workload;

/// Return current UTC epoch timestamp in milliseconds.
#[inline]
pub fn now_millis() -> u64 {
    tara_engine::now_millis()
}

/// Return current UTC epoch timestamp in seconds.
#[inline]
pub fn now_secs() -> u64 {
    tara_engine::now_secs()
}

/// Return current UTC epoch timestamp in fractional seconds (f64).
#[inline]
pub fn now_secs_f64() -> f64 {
    tara_engine::now_secs_f64()
}

/// Return current UTC time as an exact ISO 8601 string.
/// Delegates to tara_engine to avoid duplicating the implementation.
#[inline]
pub fn now_iso() -> String {
    tara_engine::now_iso()
}
