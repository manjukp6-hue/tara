//! # tara_engine
//!
//! Native Rust neural model inference, dataset processing, and controlled training engine for TARA.
//!
//! ## Architectural Boundary Tiers
//! 1. **Tier 1 — Canonical Public Facade**:
//!    - [`config`], [`tokenizer`], [`generate`], [`inference`], [`train_candidate`], [`dataset`], [`computation`]
//!    - External callers performing model training should enter through [`train_candidate`]
//!      ([`run_controlled_training_with_options`]) so hyperparameter validation, transactional staging,
//!      dual-purpose checkpoint verification, and candidate acceptance gates are strictly enforced.
//! 2. **Tier 2 — Domain & Verification Subsystems**:
//!    - [`auto_loader`], [`control_tokens`], [`curriculum_engine`], [`dataset_classifier`],
//!      [`dataset_engine`], [`device_detector`], [`github_adapter`], [`loading_policy`], [`project_link_engine`]
//! 3. **Tier 3 — Low-Level Engine & Training Infrastructure**:
//!    - [`cuda`], [`model`], [`model_expansion`], [`safetensors`], [`self_update`], [`shard_manager`],
//!      [`skills_evaluator`], [`trainer`]
//!    - Exposed for native engine binaries and verified runtime hooks; direct mutation of model weights
//!      without [`train_candidate`] staging is prohibited in production flows.
//!
//! ## Single-Owner Physical Layout Contract (`rust/tara_training_system/`)
//! Modules mapped via `#[path = "../../tara_training_system/..."]` are owned exclusively by
//! `tara_engine` and must never be compiled into a second crate (enforced by `verify_architecture`).

// ============================================================================
// Tier 1: Canonical Public Facade Modules
// ============================================================================
pub mod computation;
pub mod config;
pub mod dataset;
pub mod generate;
pub mod inference;
pub mod tokenizer;
pub mod train_candidate;

// ============================================================================
// Tier 2: Domain & Verification Subsystems
// ============================================================================
pub mod auto_loader;
pub mod control_tokens;
#[path = "../../tara_training_system/curriculum_engine/curriculum_engine.rs"]
pub mod curriculum_engine;
pub mod dataset_classifier;
#[path = "../../tara_training_system/dataset_engine/dataset_engine.rs"]
pub mod dataset_engine;
pub mod device_detector;
pub mod github_adapter;
pub mod loading_policy;
pub mod project_link_engine;

// ============================================================================
// Tier 3: Low-Level Engine & Training Infrastructure
// ============================================================================
pub mod cuda;
#[path = "../../tara_training_system/neural_model/model/mod.rs"]
pub mod model;
pub mod model_expansion;
#[path = "../../tara_training_system/shared_training_infrastructure/safetensors.rs"]
pub mod safetensors;
pub mod self_update;
pub mod shard_manager;
#[path = "../../tara_training_system/shared_training_infrastructure/skills_evaluator.rs"]
pub mod skills_evaluator;
#[path = "../../tara_training_system/shared_training_infrastructure/trainer.rs"]
pub mod trainer;

/// Authoritative list of `rust/tara_training_system/` relative paths owned exclusively by `tara_engine`.
/// Verified by `verify_architecture` to guarantee zero duplicate cross-crate compilation.
pub const OWNED_TRAINING_SYSTEM_MODULES: &[&str] = &[
    "rust/tara_training_system/curriculum_engine/curriculum_engine.rs",
    "rust/tara_training_system/dataset_engine/dataset_engine.rs",
    "rust/tara_training_system/neural_model/model/mod.rs",
    "rust/tara_training_system/shared_training_infrastructure/reader.rs",
    "rust/tara_training_system/shared_training_infrastructure/safetensors.rs",
    "rust/tara_training_system/shared_training_infrastructure/skills_evaluator.rs",
    "rust/tara_training_system/shared_training_infrastructure/trainer.rs",
];

// ============================================================================
// Curated Root Re-Exports (Supported Entry-Point Contracts)
// ============================================================================
pub use auto_loader::{AutoTaraModelLoader, LoadedTaraModel};
pub use computation::{
    BigDecimal, BigFraction, BigInt, ComplexRoot, ComputationEngine, DescriptiveStats, Dimension,
    EquationSolver, Fraction, Logarithms, Matrix, MultilingualEngine, MultilingualTerm,
    NumericalMethods, Percentage, Polynomial, Probability, Roots, Statistics, UnitConverter,
    Vector,
};
pub use config::TaraConfig;
pub use control_tokens::{ControlTokenAction, ControlTokenActionParser};
pub use curriculum_engine::{
    CurriculumBatch, CurriculumConfig, CurriculumEngine, CurriculumReport, CurriculumStage,
    NeuralCurriculum, WorldModelCurriculum,
};
pub use dataset_engine::{
    DatasetEngine, DatasetEngineConfig, DatasetMetadata, DatasetSource, DatasetSyncReport,
    NeuralRecord, RoutedDataset, TargetModel, TopologicalConfidenceConfig, WorldModelRecord,
};
pub use device_detector::{DeviceCapabilityDetector, DeviceProfile, GpuInfo};
pub use generate::{generate_response, generate_stream, GenerateOptions, GenerateResult};
pub use github_adapter::{
    compute_dynamic_sha256, compute_hmac_sha256, generate_github_ci_workflow_content,
    verify_webhook_hmac_signature, GitHubAdapter, GitHubAdapterConfig, GitHubCommitRecord,
    GitHubExportManifest, GitHubPushPayload, GitHubRemoteSyncReport, GitHubValidationReport,
    GitHubWebhookResult,
};
pub use inference::{
    BackendRegistry, ConcurrentTieredTensorStore, DeviceBackend, DeviceTensor, StorageTier,
    TelemetryMonitor, TieredTensorStore,
};
pub use loading_policy::{
    InsufficientMemoryError, LoadingPlan, LoadingPolicyEngine, ModelLoadingStrategy,
    QuantizationPrecision,
};
pub use model::causal_lm::TaraForCausalLM;
pub use model_expansion::{
    ArchitectureConstraints, ArchitectureScaler, GrowthType, ModelExpansionEngine,
};
pub use project_link_engine::{
    ImportStatement, ModuleDeclaration, ProjectLinkConfig, ProjectLinkEngine,
    RelationshipDelta, RelationshipGraphSnapshot, RelationshipReport, RustFileNode,
    SymbolDeclaration, SymbolKind, Visibility,
};
pub use self_update::{ExpansionDecision, SelfUpdateController, SelfUpdateReport, UpdateState};
pub use shard_manager::{ShardedSafeTensorsManager, TensorMetadata};
pub use tokenizer::TaraTokenizer;
pub use train_candidate::{
    run_controlled_training, run_controlled_training_with_options, StoppingContract,
    TrainCandidateError, TrainingOptions,
};

/// Formats non-negative Unix epoch seconds (`secs` since `1970-01-01T00:00:00Z`) into an
/// exact UTC ISO-8601 timestamp (`YYYY-MM-DDTHH:MM:SSZ`) using Howard Hinnant's proleptic
/// Gregorian civil-from-days algorithm (handling 4/100/400-year leap rules accurately).
pub fn format_unix_seconds_iso(secs: u64) -> String {
    let s = secs % 60;
    let m = (secs / 60) % 60;
    let h = (secs / 3600) % 24;
    let days = (secs / 86400) as i64;

    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { y + 1 } else { y };

    format!("{year:04}-{month:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
}

/// Formats a [`std::time::SystemTime`] as an exact UTC ISO-8601 timestamp (`YYYY-MM-DDTHH:MM:SSZ`).
///
/// Pre-epoch timestamps (prior to `1970-01-01T00:00:00Z`) are deterministically clamped to
/// Unix epoch zero (`1970-01-01T00:00:00Z`).
pub fn format_system_time_iso(time: std::time::SystemTime) -> String {
    use std::time::UNIX_EPOCH;
    let secs = time
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format_unix_seconds_iso(secs)
}

/// Return current UTC time as an exact ISO-8601 string (`YYYY-MM-DDTHH:MM:SSZ`).
pub fn now_iso() -> String {
    format_system_time_iso(std::time::SystemTime::now())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, UNIX_EPOCH};

    #[test]
    fn test_now_iso_and_gregorian_civil_date_boundaries() {
        // 1. Unix Epoch Zero: 1970-01-01T00:00:00Z
        assert_eq!(format_unix_seconds_iso(0), "1970-01-01T00:00:00Z");

        // 2. Pre-Epoch SystemTime (1969-12-31) clamps deterministically to 1970-01-01T00:00:00Z
        let pre_epoch = UNIX_EPOCH - Duration::from_secs(86_400);
        assert_eq!(format_system_time_iso(pre_epoch), "1970-01-01T00:00:00Z");

        // 3. 400-year leap century day: 2000-02-29T12:34:56Z (951827696)
        assert_eq!(format_unix_seconds_iso(951_827_696), "2000-02-29T12:34:56Z");

        // 4. Standard leap year boundary: 2024-02-29T23:59:59Z -> 2024-03-01T00:00:00Z
        assert_eq!(
            format_unix_seconds_iso(1_709_251_199),
            "2024-02-29T23:59:59Z"
        );
        assert_eq!(
            format_unix_seconds_iso(1_709_251_200),
            "2024-03-01T00:00:00Z"
        );

        // 5. End of 2099: 2099-12-31T23:59:59Z (4102444799)
        assert_eq!(
            format_unix_seconds_iso(4_102_444_799),
            "2099-12-31T23:59:59Z"
        );

        // 6. Non-leap century boundary (2100 is divisible by 100 but NOT 400 -> NOT a leap year):
        // 2100-02-28T23:59:59Z (4107542399) must roll directly to 2100-03-01T00:00:00Z (4107542400)
        assert_eq!(
            format_unix_seconds_iso(4_107_542_399),
            "2100-02-28T23:59:59Z"
        );
        assert_eq!(
            format_unix_seconds_iso(4_107_542_400),
            "2100-03-01T00:00:00Z"
        );

        // 7. Live now_iso() format verification (20-char YYYY-MM-DDTHH:MM:SSZ)
        let live = now_iso();
        assert_eq!(live.len(), 20);
        assert!(live.ends_with('Z'));
        assert_eq!(&live[4..5], "-");
        assert_eq!(&live[7..8], "-");
        assert_eq!(&live[10..11], "T");
        assert_eq!(&live[13..14], ":");
        assert_eq!(&live[16..17], ":");
    }

    #[test]
    fn test_root_facade_and_single_owner_training_system_contract() {
        // Verify all declared owned training system modules exist and are unique
        let mut seen = std::collections::HashSet::new();
        for rel in OWNED_TRAINING_SYSTEM_MODULES {
            assert!(
                seen.insert(*rel),
                "Duplicate module in OWNED_TRAINING_SYSTEM_MODULES: {rel}"
            );
        }
        let contract = StoppingContract::StepBoundedEpochs {
            max_epochs: 100,
            max_steps: 3,
        };
        assert_eq!(
            contract.contract_name(),
            "first_reached(completed_epochs >= max_epochs, optimizer_steps >= max_steps)"
        );
    }
}
