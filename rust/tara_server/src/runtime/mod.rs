//! TARA Runtime Management & Universal Promotion Gate.
//!
//! Language-neutral dynamic runtime registration, canonical contract validation,
//! and all-runtime promotion gating.

pub mod architecture_sync;
pub mod gate;
pub mod registry;

pub use architecture_sync::{
    ArchitectureState, ArchitectureSyncConfig, ArchitectureSyncEngine, FileRecord, FolderRecord,
    ReconciliationReport, TreeNode,
};
pub use gate::{
    GateEvaluationResponse, GateVerdict, RuntimeEvaluationResult, UniversalRuntimeGate,
};
pub use registry::{DynamicRuntimeRegistry, RuntimeRecord, RuntimeState};
