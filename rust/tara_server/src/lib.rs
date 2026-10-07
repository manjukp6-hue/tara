//! TARA Server library crate.
//!
//! Exposes all server subsystems as public modules so they can be used
//! from the binary entry point and from integration tests.

pub mod ability;
pub mod access;
pub mod auto_connect;
pub mod brain;

pub use ability::AbilityEngine;
pub mod bridge;
pub mod code_evolution;
pub mod cognitive;
pub mod context;
pub mod contract;
pub mod control_plane;
pub mod engine_system;
pub mod engines;
pub mod evaluator;
pub mod knowledge;
pub mod language;
pub mod learning;
pub mod memory;
pub mod memory_hub;
pub mod model_registry;
pub mod nlu;
pub mod planner;
pub mod registry;
pub mod reward;
pub mod routes;
pub mod rules;
pub mod runtime;
pub mod security;
pub mod server;
pub mod skills;
pub mod storage_retention;
pub mod subsystems;
pub mod tools_registry;
pub mod voice;
pub mod workload;
/// Return current UTC time as an exact ISO 8601 string.
/// Delegates to tara_engine to avoid duplicating the implementation.
#[inline]
pub fn now_iso() -> String {
    tara_engine::now_iso()
}
