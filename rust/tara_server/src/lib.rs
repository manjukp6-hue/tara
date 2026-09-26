//! TARA Server library crate.
//!
//! Exposes all server subsystems as public modules so they can be used
//! from the binary entry point and from integration tests.

pub mod server;
pub mod routes;
pub mod brain;
pub mod nlu;
pub mod context;
pub mod planner;
pub mod evaluator;
pub mod registry;
pub mod tools_registry;
pub mod memory_hub;
pub mod memory;
pub mod knowledge;
pub mod rules;
pub mod access;
pub mod skills;
pub mod learning;
pub mod auto_connect;
pub mod engine_system;
pub mod cognitive;
pub mod workload;
pub mod model_registry;
pub mod subsystems;
pub mod bridge;
pub mod contract;
pub mod control_plane;
pub mod runtime;

/// Return current UTC time as an ISO 8601-like string (approximate).
pub fn now_iso() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
    let s = secs % 60;
    let m = (secs / 60) % 60;
    let h = (secs / 3600) % 24;
    let days = secs / 86400;
    let year = 2024_u64 + days / 365;
    format!("{:04}-01-01T{:02}:{:02}:{:02}Z", year, h, m, s)
}
