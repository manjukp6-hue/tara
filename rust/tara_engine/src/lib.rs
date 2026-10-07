//! # tara_engine
//!
//! Neural model inference engine for the TARA AI system.
//!
//! Provides a pure-Rust implementation of the TaraForCausalLM transformer,
//! SafeTensors weight loading, tokenization, sampling, and training utilities.

pub mod auto_loader;
pub mod computation;
pub mod config;
pub mod control_tokens;
pub mod cuda;
pub mod dataset;
pub mod dataset_classifier;
#[path = "../../tara_training_system/dataset_engine/dataset_engine.rs"]
pub mod dataset_engine;
#[path = "../../tara_training_system/curriculum_engine/curriculum_engine.rs"]
pub mod curriculum_engine;
pub mod device_detector;
pub mod generate;
pub mod github_adapter;
pub mod inference;
pub mod loading_policy;
#[path = "../../tara_training_system/neural_model/model/mod.rs"]
pub mod model;
pub mod model_expansion;
pub mod project_link_engine;
#[path = "../../tara_training_system/shared_training_infrastructure/safetensors.rs"]
pub mod safetensors;
pub mod self_update;
pub mod shard_manager;
#[path = "../../tara_training_system/shared_training_infrastructure/skills_evaluator.rs"]
pub mod skills_evaluator;
pub mod tokenizer;
pub mod train_candidate;
#[path = "../../tara_training_system/shared_training_infrastructure/trainer.rs"]
pub mod trainer;

pub use project_link_engine::{
    ImportStatement, ModuleDeclaration, ProjectLinkConfig, ProjectLinkEngine,
    RelationshipDelta, RelationshipGraphSnapshot, RelationshipReport, RustFileNode,
    SymbolDeclaration, SymbolKind, Visibility,
};
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
pub use loading_policy::{
    InsufficientMemoryError, LoadingPlan, LoadingPolicyEngine, ModelLoadingStrategy,
    QuantizationPrecision,
};
pub use model::causal_lm::TaraForCausalLM;
pub use model_expansion::{
    ArchitectureConstraints, ArchitectureScaler, GrowthType, ModelExpansionEngine,
};
pub use self_update::{ExpansionDecision, SelfUpdateController, SelfUpdateReport, UpdateState};
pub use shard_manager::{ShardedSafeTensorsManager, TensorMetadata};
pub use tokenizer::TaraTokenizer;

/// Return current UTC time as an exact ISO 8601 string.
pub fn now_iso() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
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
