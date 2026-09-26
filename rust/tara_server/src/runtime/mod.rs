//! TARA Runtime Management & Universal Promotion Gate.
//!
//! Language-neutral dynamic runtime registration, canonical contract validation,
//! and all-runtime promotion gating.

pub mod registry;
pub mod gate;

pub use registry::{DynamicRuntimeRegistry, RuntimeRecord, RuntimeState};
pub use gate::{UniversalRuntimeGate, GateVerdict, RuntimeEvaluationResult, GateEvaluationResponse};
