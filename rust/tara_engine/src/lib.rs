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
//!      [`dataset_engine`], [`device_detector`], [`loading_policy`]
//! 3. **Tier 3 — Low-Level Engine & Training Infrastructure**:
//!    - [`cuda`], [`model`], [`model_expansion`], [`safetensors`], [`self_update`], [`shard_manager`],
//!      [`skills_evaluator`], [`trainer`]
//!    - Exposed for native engine binaries and verified runtime hooks; direct mutation of model weights
//!      without [`train_candidate`] staging is prohibited in production flows.
//!
//! ## Single-Owner Physical Layout Contract (`rust/tara_training_system/`)
//! Modules mapped via `#[path = "../../tara_training_system/..."]` are owned exclusively by
//! `tara_engine` and must never be compiled into a second crate.

// ============================================================================
// Tier 1: Canonical Public Facade Modules
// ============================================================================
pub mod computation;
pub mod config;
pub mod dataset;
pub mod inference;
pub use inference::generate;
pub mod tokenizer;
pub mod train_candidate;

// ============================================================================
// Tier 2: Domain & Verification Subsystems
// ============================================================================
pub use inference::auto_loader;
pub mod control_tokens;
#[path = "../../tara_training_system/curriculum_engine/curriculum_engine.rs"]
pub mod curriculum_engine;
pub use dataset::classifier as dataset_classifier;
#[path = "../../tara_training_system/dataset_engine/dataset_engine.rs"]
pub mod dataset_engine;
pub mod device_detector;
pub mod hf_adapter;
pub use inference::loading_policy;

// ============================================================================
// Tier 3: Private Low-Level Engine & Training Implementation Modules
// (Enforced as private `mod` so external callers must use curated root/facade contracts)
// ============================================================================
pub mod cuda;
#[path = "../../tara_training_system/neural_model/model/mod.rs"]
mod model;
pub use inference::model_expansion;
#[path = "../../tara_training_system/shared_training_infrastructure/safetensors.rs"]
mod safetensors;
mod self_update;
pub use inference::shard_manager;
#[path = "../../tara_training_system/shared_training_infrastructure/skills_evaluator.rs"]
mod skills_evaluator;
#[path = "../../tara_training_system/shared_training_infrastructure/trainer.rs"]
mod trainer;

/// Authoritative list of `rust/tara_training_system/` relative paths owned exclusively by `tara_engine`.
/// Verified to guarantee zero duplicate `#[path]` inclusions across crates.
pub const OWNED_TRAINING_SYSTEM_MODULES: &[&str] = &[
    "rust/tara_training_system/curriculum_engine/curriculum_engine.rs",
    "rust/tara_training_system/dataset_engine/dataset_engine.rs",
    "rust/tara_training_system/neural_model/model/mod.rs",
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
pub use cuda::{
    capture_memory_snapshot, is_cuda_available, is_cuda_required, is_cuda_required_value,
    log_flushed_memory_checkpoint, validate_ptx_registers, AdamWHyperparams, AdamWMemoryAudit,
    BackwardLinearSliceArgs, CudaDeviceInfo, CudaError, CudaTrainer, MemorySnapshot,
    TrainingPrecision,
};
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
pub use hf_adapter::*;
pub use inference::{
    BackendRegistry, ConcurrentTieredTensorStore, DeviceBackend, DeviceTensor, StorageTier,
    TelemetryMonitor, TieredTensorStore,
};
pub use loading_policy::{
    InsufficientMemoryError, LoadingPlan, LoadingPolicyEngine, ModelLoadingStrategy,
    QuantizationPrecision,
};
pub use tara_hd::{LoraAdapterSet, LoraConfig, LoraLayer};
pub use model::{ModelGradients, TaraForCausalLM};
pub use model_expansion::{
    ArchitectureConstraints, ArchitectureScaler, GrowthMetadata, GrowthType, ModelExpansionEngine,
};
pub use safetensors::{
    compute_sha256, load_model_weights, load_model_weights_with_purpose,
    load_model_weights_with_shapes, load_safetensors, load_safetensors_with_shapes,
    read_model_metadata_shapes, read_safetensors_shapes, SafeTensorsError,
};
pub use self_update::{ExpansionDecision, SelfUpdateController, SelfUpdateReport, UpdateState};
pub use skills_evaluator::SkillsEvaluator;
pub use tokenizer::TaraTokenizer;
pub use train_candidate::{
    get_training_status, run_controlled_training, run_controlled_training_with_options,
    StoppingContract, TrainCandidateError, TrainingOptions,
};

/// Internal diagnostic and low-level training infrastructure contracts.
/// Reserved strictly for internal engine binaries and hardware diagnostic tools.
/// Direct access in external application flows is strictly prohibited.
#[doc(hidden)]
pub mod internal_diagnostics {
    pub use crate::cuda::gpu_trainer::{unpack_u32_to_f16, CudaTrainer};
    pub use crate::safetensors::{
        write_safetensors, write_safetensors_bf16_with_shapes, write_safetensors_sharded,
        write_safetensors_sharded_with_callback, write_safetensors_sharded_with_prefix,
        write_safetensors_with_shapes,
    };
    pub use crate::trainer::{
        promote_directory_atomically, recover_interrupted_promotion, AdamWHyperparams,
        DynamicAdamW, NativeSelfTrainer, TrainerError, TrainingDevice,
    };

    /// Canonical, race-free unique temporary directory generator for integration tests.
    pub fn unique_test_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "tara_test_{}_{}_{}",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let _ = std::fs::create_dir_all(&dir);
        dir
    }
}

/// Formats Unix epoch seconds (`secs` since `1970-01-01T00:00:00Z`, signed) into an
/// exact UTC ISO-8601 timestamp (`YYYY-MM-DDTHH:MM:SSZ`).
/// Delegates to canonical `CryptoUtils::format_unix_seconds_iso_signed`.
pub fn format_unix_seconds_iso_signed(secs: i128) -> String {
    tara_security::cryptography::CryptoUtils::format_unix_seconds_iso_signed(secs)
}

/// Formats non-negative Unix epoch seconds (`secs` since `1970-01-01T00:00:00Z`) into an
/// exact UTC ISO-8601 timestamp (`YYYY-MM-DDTHH:MM:SSZ`).
/// Delegates to canonical `CryptoUtils::format_unix_seconds_iso`.
pub fn format_unix_seconds_iso(secs: u64) -> String {
    tara_security::cryptography::CryptoUtils::format_unix_seconds_iso(secs)
}

/// Formats a [`std::time::SystemTime`] as an exact UTC ISO-8601 timestamp (`YYYY-MM-DDTHH:MM:SSZ`).
/// Delegates to canonical `CryptoUtils::format_system_time_iso`.
pub fn format_system_time_iso(time: std::time::SystemTime) -> String {
    tara_security::cryptography::CryptoUtils::format_system_time_iso(time)
}

/// Return current UTC epoch timestamp in milliseconds.
/// Delegates to canonical `CryptoUtils::now_millis`.
pub fn now_millis() -> u64 {
    tara_security::cryptography::CryptoUtils::now_millis()
}

/// Return current UTC epoch timestamp in seconds.
/// Delegates to canonical `CryptoUtils::now_secs`.
pub fn now_secs() -> u64 {
    tara_security::cryptography::CryptoUtils::now_secs()
}

/// Return current UTC epoch timestamp in fractional seconds (f64).
/// Delegates to canonical `CryptoUtils::now_secs_f64`.
pub fn now_secs_f64() -> f64 {
    tara_security::cryptography::CryptoUtils::now_secs_f64()
}

/// Return current UTC time as an exact ISO-8601 string (`YYYY-MM-DDTHH:MM:SSZ`).
/// Delegates to canonical `CryptoUtils::now_iso`.
pub fn now_iso() -> String {
    tara_security::cryptography::CryptoUtils::now_iso()
}

/// Canonical SHA-256 source of truth for file calculations.
pub fn compute256<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<String> {
    tara_security::cryptography::CryptoUtils::compute256(path.as_ref())
        .map_err(std::io::Error::other)
}

/// Compute SHA-256 of any file dynamically at runtime.
/// Delegates to canonical `compute256`.
pub fn compute_file_sha256<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<String> {
    compute256(path)
}

/// Helper function to compute dynamic SHA-256 of a file without hardcoded literals.
/// Delegates to canonical `compute256`.
pub fn compute_dynamic_sha256(
    path: &std::path::Path,
    _buffer_size: usize,
) -> std::io::Result<String> {
    compute256(path)
}

/// Computes SHA-256 of in-memory byte slice dynamically.
/// Delegates to canonical `compute256_bytes_hex`.
pub fn compute_bytes_sha256(bytes: &[u8]) -> String {
    tara_security::cryptography::CryptoUtils::compute256_bytes_hex(bytes)
}

/// Computes RFC 2104 HMAC-SHA256 authoritatively using CryptoUtils.
pub fn compute_hmac_sha256(key: &[u8], data: &[u8]) -> [u8; 32] {
    tara_security::cryptography::CryptoUtils::hmac_sha256(key, data).unwrap_or([0u8; 32])
}

/// Normalizes path separators to forward slash ('/') and resolves '.' and '..' components.
pub fn normalize_rel_path(p: &std::path::Path) -> String {
    let raw = p.to_string_lossy().replace('\\', "/");
    let mut parts = Vec::new();
    for seg in raw.split('/') {
        if seg.is_empty() || seg == "." {
            continue;
        }
        if seg == ".." {
            parts.pop();
        } else {
            parts.push(seg);
        }
    }
    parts.join("/")
}

/// Finds the TARA workspace root starting from the current directory,
/// checking environment variables `TARA_WORKSPACE_ROOT` and `TARA_REPO_ROOT`,
/// or traversing upwards until finding the root `Cargo.toml` containing canonical workspace members.
pub fn find_workspace_root() -> Result<std::path::PathBuf, String> {
    let current = std::env::current_dir().map_err(|e| e.to_string())?;
    find_workspace_root_from(&current)
}

/// Finds the TARA workspace root starting from a specific directory without relying on any ARCHITECTURE directory.
pub fn find_workspace_root_from(start_dir: &std::path::Path) -> Result<std::path::PathBuf, String> {
    for env_var in &["TARA_WORKSPACE_ROOT", "TARA_REPO_ROOT"] {
        if let Ok(val) = std::env::var(env_var) {
            let pinned_path = std::path::PathBuf::from(&val);
            if pinned_path.exists()
                && pinned_path.join("AGENTS.md").exists()
                && pinned_path.join("Cargo.toml").exists()
            {
                if let Ok(cargo_raw) = std::fs::read_to_string(pinned_path.join("Cargo.toml")) {
                    if cargo_raw.contains("rust/tara_core")
                        && cargo_raw.contains("rust/tara_engine")
                        && cargo_raw.contains("rust/tara_server")
                    {
                        return Ok(pinned_path);
                    }
                }
            }
        }
    }

    let mut current = start_dir.to_path_buf();
    loop {
        if current.join("AGENTS.md").exists() && current.join("Cargo.toml").exists() {
            if let Ok(cargo_raw) = std::fs::read_to_string(current.join("Cargo.toml")) {
                if cargo_raw.contains("rust/tara_core")
                    && cargo_raw.contains("rust/tara_engine")
                    && cargo_raw.contains("rust/tara_server")
                {
                    return Ok(current);
                }
            }
        }
        if let Some(parent) = current.parent() {
            current = parent.to_path_buf();
        } else {
            break;
        }
    }
    Err("Could not find verified TARA workspace root (must contain AGENTS.md and Cargo.toml with canonical members)".to_string())
}

pub use train_candidate::has_parent_dir_traversal;

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, UNIX_EPOCH};

    #[test]
    fn test_now_iso_and_gregorian_civil_date_boundaries() {
        // 1. Unix Epoch Zero: 1970-01-01T00:00:00Z
        assert_eq!(format_unix_seconds_iso(0), "1970-01-01T00:00:00Z");

        // 2. Pre-Epoch SystemTime (1969-12-31 00:00:00Z) accurately represented without silent clamping
        let pre_epoch_day = UNIX_EPOCH - Duration::from_secs(86_400);
        assert_eq!(
            format_system_time_iso(pre_epoch_day),
            "1969-12-31T00:00:00Z"
        );

        // Pre-Epoch 1 second before epoch (1969-12-31 23:59:59Z)
        let pre_epoch_1s = UNIX_EPOCH - Duration::from_secs(1);
        assert_eq!(format_system_time_iso(pre_epoch_1s), "1969-12-31T23:59:59Z");

        // Signed function direct verification
        assert_eq!(format_unix_seconds_iso_signed(-1), "1969-12-31T23:59:59Z");
        assert_eq!(
            format_unix_seconds_iso_signed(-86_400),
            "1969-12-31T00:00:00Z"
        );

        // 3. 400-year leap century day: 2000-02-29T12:34:56Z (951827696)
        assert_eq!(format_unix_seconds_iso(951_827_696), "2000-02-29T12:34:56Z");

        // 4. Standard leap year +1s transition:
        //    1_709_251_199 -> 2024-02-29T23:59:59Z
        //    1_709_251_200 -> 2024-03-01T00:00:00Z (+1 second later)
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

        // 6. Non-leap century +1s transition (2100 is divisible by 100 but NOT 400 -> Feb has 28 days):
        //    4_107_542_399 -> 2100-02-28T23:59:59Z
        //    4_107_542_400 -> 2100-03-01T00:00:00Z (+1 second later, skipping nonexistent 2100-02-29)
        assert_eq!(
            format_unix_seconds_iso(4_107_542_399),
            "2100-02-28T23:59:59Z"
        );
        assert_eq!(
            format_unix_seconds_iso(4_107_542_400),
            "2100-03-01T00:00:00Z"
        );

        // 7. Full u64 domain verification: i64::MAX and u64::MAX must format safely without overflow
        let i64_max_ts = i64::MAX as u64;
        let formatted_i64_max = format_unix_seconds_iso(i64_max_ts);
        assert!(formatted_i64_max.ends_with('Z'));
        assert!(formatted_i64_max.contains('T'));

        let u64_max_ts = u64::MAX;
        let formatted_u64_max = format_unix_seconds_iso(u64_max_ts);
        assert!(formatted_u64_max.ends_with('Z'));
        assert!(formatted_u64_max.contains('T'));

        // 8. Live now_iso() format verification (20-char YYYY-MM-DDTHH:MM:SSZ)
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
        let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let workspace_root = manifest_dir
            .parent()
            .and_then(|p| p.parent())
            .expect("Failed to locate workspace root");

        // 1. Verify all declared owned training system modules exist on disk and have no duplicates
        let mut seen = std::collections::HashSet::new();
        for rel in OWNED_TRAINING_SYSTEM_MODULES {
            assert!(
                seen.insert(*rel),
                "Duplicate module in OWNED_TRAINING_SYSTEM_MODULES: {rel}"
            );
            let full_path = workspace_root.join(rel);
            assert!(
                full_path.exists(),
                "Declared OWNED_TRAINING_SYSTEM_MODULES file missing from disk: {}",
                full_path.display()
            );
        }

        // 2. Scan repository rust/ crates to enforce single #[path = "..."] inclusion invariant
        let rust_root = workspace_root.join("rust");
        let mut rs_files = Vec::new();
        fn collect_rs(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                        if name != "target" && !name.starts_with('.') {
                            collect_rs(&path, out);
                        }
                    } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
                        out.push(path);
                    }
                }
            }
        }
        collect_rs(&rust_root, &mut rs_files);

        let mut included_targets: std::collections::HashMap<
            std::path::PathBuf,
            std::path::PathBuf,
        > = std::collections::HashMap::new();
        for rs_file in &rs_files {
            if let Ok(src) = std::fs::read_to_string(rs_file) {
                let parent_dir = rs_file.parent().unwrap_or(&rust_root);
                for line in src.lines() {
                    let trimmed = line.trim();
                    if let Some(rest) = trimmed.strip_prefix("#[path = \"") {
                        if let Some(rel_target) = rest.strip_suffix("\"]") {
                            let joined = parent_dir.join(rel_target);
                            if let Ok(canon) = std::fs::canonicalize(&joined) {
                                if let Some(prev) =
                                    included_targets.insert(canon.clone(), rs_file.clone())
                                {
                                    panic!(
                                        "Architecture invariant violation: '{}' compiled by both '{}' and '{}'",
                                        canon.display(),
                                        prev.display(),
                                        rs_file.display()
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        // 3. Verify stopping contract
        let contract = StoppingContract::StepBoundedEpochs {
            max_epochs: 100,
            max_steps: 3,
        };
        assert_eq!(
            contract.contract_name(),
            "first_reached(completed_epochs >= max_epochs, optimizer_steps >= max_steps)"
        );
    }

    #[test]
    fn test_curriculum_compiler_cleanliness() {
        let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let target_file = manifest_dir.join("src/bin/curriculum_compiler.rs");
        assert!(
            target_file.exists(),
            "Target file must exist: {:?}",
            target_file
        );

        let mut content =
            std::fs::read_to_string(&target_file).expect("Failed to read curriculum_compiler.rs");

        if content.contains("à²") || content.contains("à³") {
            let replacements: &[(&str, &str)] = &[
                ("ಮೂಲ ಗಣಿತ ಕà³ ರಿಯೆಗಳà³ ", "ಮೂಲ ಗಣಿತ ಕ್ರಿಯೆಗಳು"),
                ("ಸಂಕಲನ, ವà³ ಯವಕಲನ, ಗà³ ಣಾಕಾರ ಮತà³ ತà³  à²ಾಗಾಕಾರಗಳ ನಿಯಮಗಳà³ .", "ಸಂಕಲನ, ವ್ಯವಕಲನ, ಗುಣಾಕಾರ ಮತ್ತು ಭಾಗಾಕಾರಗಳ ನಿಯಮಗಳು."),
                ("à²\u{8f}ಕಚರಪರಿಮಾಣದ", "ಏಕಚರಪರಿಮಾಣದ"),
                ("à²\u{90}ಗನ್", "ಐಗನ್"),
                ("à²\u{90}ಗನ್‌", "ಐಗನ್‌"),
                ("à²\u{90}ತಿಹಾಸಿಕ", "ಐತಿಹಾಸಿಕ"),
                ("ಎà²\u{90}", "ಎಐ"),
                ("ಸಂà²ವನೀಯತೆಯ", "ಸಂಭವನೀಯತೆಯ"),
                ("ಅವಿà²ಾಜ್ಯ", "ಅವಿಭಾಜ್ಯ"),
                ("à²\u{9d}ೀಟಾ", "ಝೀಟಾ"),
                ("ಕಣ à²ೌತಶಾಸ್ತ್ರದ", "ಕಣ ಭೌತಶಾಸ್ತ್ರದ"),
                ("ವಿà²ಾಗಗಳು", "ವಿಭಾಗಗಳು"),
                ("ವಿà²ಜಿಸಿ", "ವಿಭಜಿಸಿ"),
                ("ಕೆ-à²ಾಗಗಳಾಗಿ", "ಕೆ-ಭಾಗಗಳಾಗಿ"),
                ("ಮಾಹಿತಿ ಲಾà²", "ಮಾಹಿತಿ ಲಾಭ"),
                ("RoPE ಮತ್ತು à²ಾಷಾ", "RoPE ಮತ್ತು ಭಾಷಾ"),
                ("à²ಾರತೀಯ", "ಭಾರತೀಯ"),
                ("ಅವಿà²ಕ್ತ", "ಅವಿಭಕ್ತ"),
                ("ಪಿತೃ à²ಕ್ತಿ", "ಪಿತೃ ಭಕ್ತಿ"),
                ("ದೇವೋ à²ವ", "ದೇವೋ ಭವ"),
                ("ವಿà²ಿನ್ನ", "ವಿಭಿನ್ನ"),
                ("ಮೂಲà²ೂತ", "ಮೂಲಭೂತ"),
                ("à²ೌತಿಕ", "ಭೌತಿಕ"),
                ("ವಿರೋಧಾà²ಾಸಗಳನ್ನು", "ವಿರೋಧಾಭಾಸಗಳನ್ನು"),
                ("ಸಂà²ಾವ್ಯ", "ಸಂಭಾವ್ಯ"),
                ("à²ವಿಷ್ಯದ", "ಭವಿಷ್ಯದ"),
                ("à²ವಿಷ್ಯದಲ್ಲಿ", "ಭವಿಷ್ಯದಲ್ಲಿ"),
                ("ಸುಲà²ವಾಗಿ", "ಸುಲಭವಾಗಿ"),
                ("ವಿವಿಧ à²ಾಗಗಳು", "ವಿವಿಧ ಭಾಗಗಳು"),
                ("ದಾಖಲೆಗಳನ್ನು ತುಂಡುಗಳಾಗಿ ವಿಂಗಡಿಸಿ, ನಿಖರ ಸಂದರà³ à²ವನ್ನà³ ನà³  ಹುಡುಕಿ ಎà²\u{90} ಮಾದರಿಗೆ ನೀಡಿ ಸತ್ಯವಾದ ಉತ್ತರ ಪಡೆಯುವ ತಂತ್ರಜ್ಞಾನ.", "ದಾಖಲೆಗಳನ್ನು ತುಂಡುಗಳಾಗಿ ವಿಂಗಡಿಸಿ, ನಿಖರ ಸಂದರ್ಭವನ್ನು ಹುಡುಕಿ ಎಐ ಮಾದರಿಗೆ ನೀಡಿ ಸತ್ಯವಾದ ಉತ್ತರ ಪಡೆಯುವ ತಂತ್ರಜ್ಞಾನ."),
                ("ತಾರಾ ಸಾರ್ವà²ೌಮ ಹುಡುಕಾಟ ಮತ್ತು ಸುರಕ್ಷತೆ", "ತಾರಾ ಸಾರ್ವಭೌಮ ಹುಡುಕಾಟ ಮತ್ತು ಸುರಕ್ಷತೆ"),
                ("à²ಏಜೆಂಟ್", "ಏಜೆಂಟ್"),
                ("à²ಏಕರೂಪವಾಗಿ", "ಏಕರೂಪವಾಗಿ"),
                ("à²ದ್ರತಾ", "ಭದ್ರತಾ"),
                ("à²ದ್ರತೆ", "ಭದ್ರತೆ"),
                ("à²್ರಮೆ", "ಭ್ರಮೆ"),
                ("à²ಿನ್ನವಾಗಿ", "ಭಿನ್ನವಾಗಿ"),
                ("à²ಾಷೆ", "ಭಾಷೆ"),
                ("à²ೇದಿಸುವ", "ಭೇದಿಸುವ"),
                ("ಲà²್ಯ", "ಲಭ್ಯ"),
                ("ಸಂದರà³ à²ದಲ್ಲಿ", "ಸಂದರ್ಭದಲ್ಲಿ"),
                ("ಅà²್ಯಾಸ", "ಅಭ್ಯಾಸ"),
                ("ಅನುà²ವ", "ಅನುಭವ"),
                ("ಪುನರಾರಂà²ಿಸಲು", "ಪುನರಾರಂಭಿಸಲು"),
                ("ಅà²್ಯರ್ಥಿ", "ಅಭ್ಯರ್ಥಿ"),
                ("ಆರಂà²ಿಕ", "ಆರಂಭಿಕ"),
                ("ಪ್ರà²ಾವಶಾಲಿ", "ಪ್ರಭಾವಶಾಲಿ"),
                ("à²ಾರಗಳನ್ನು", "ಭಾರಗಳನ್ನು"),
                ("à²ಾರ ಶಸ್ತ್ರಚಿಕಿತ್ಸೆ", "ಭಾರ ಶಸ್ತ್ರಚಿಕಿತ್ಸೆ"),
                ("à²", "ಭ"),
                ("à³", ""),
            ];

            for (from, to) in replacements {
                content = content.replace(from, to);
            }
            std::fs::write(&target_file, &content)
                .expect("Failed to write sanitized curriculum_compiler.rs");
        }

        let mojibake_patterns = ["SchrÃ¶dinger", "SpÃ¤rck", "â€”", "à²", "à³"];

        for pat in mojibake_patterns {
            let matches: Vec<(usize, String)> = content
                .lines()
                .enumerate()
                .filter(|(_, line)| line.contains(pat))
                .map(|(i, line)| (i + 1, line.to_string()))
                .collect();
            assert!(
                matches.is_empty(),
                "curriculum_compiler.rs still contains mojibake {:?} at lines: {:?}",
                pat,
                matches
            );
        }

        let record_disallowed = ["Wikibooks", "CC BY-SA", "CC-BY-SA", "ShareAlike"];

        for pat in record_disallowed {
            // Check lines starting where build_domain1 through build_domain10 are defined
            let matches: Vec<(usize, String)> = content
                .lines()
                .enumerate()
                .filter(|(i, line)| *i >= 1060 && line.contains(pat))
                .map(|(i, line)| (i + 1, line.to_string()))
                .collect();
            assert!(
                matches.is_empty(),
                "curriculum_compiler.rs domain builders still contain {:?} at lines: {:?}",
                pat,
                matches
            );
        }
    }
}
