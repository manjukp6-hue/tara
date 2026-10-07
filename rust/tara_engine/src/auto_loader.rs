//! Automatic Hardware-Aware Model Loader for Permanent TARA Architecture.
//!
//! Executes:
//! 1. Host capability probe via `DeviceCapabilityDetector`.
//! 2. Optimal loading plan computation via `LoadingPolicyEngine`.
//! 3. SafeTensors shard auto-discovery and lazy/full paging via `ShardedSafeTensorsManager`.
//! 4. Preserves unified TARA model identity.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::device_detector::{DeviceCapabilityDetector, DeviceProfile};
use crate::loading_policy::{LoadingPlan, LoadingPolicyEngine, ModelLoadingStrategy};
use crate::safetensors::SafeTensorsError;
use crate::shard_manager::ShardedSafeTensorsManager;

pub struct LoadedTaraModel {
    pub model_dir: PathBuf,
    pub profile: DeviceProfile,
    pub plan: LoadingPlan,
    pub shard_manager: ShardedSafeTensorsManager,
    pub full_weights: Option<HashMap<String, Vec<f32>>>,
    pub is_fallback_active: bool,
    pub active_fallback_reason: String,
}

impl LoadedTaraModel {
    pub fn model_identity(&self) -> &'static str {
        "TARA"
    }

    pub fn get_tensor(&mut self, tensor_name: &str) -> Result<Vec<f32>, SafeTensorsError> {
        if let Some(ref weights) = self.full_weights {
            if let Some(t) = weights.get(tensor_name) {
                return Ok(t.clone());
            }
        }
        self.shard_manager.load_tensor(tensor_name)
    }
}

pub const DEFAULT_MAX_CACHED_SHARDS: usize = 4;

pub fn resolve_max_cached_shards() -> usize {
    std::env::var("TARA_MAX_CACHED_SHARDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_MAX_CACHED_SHARDS)
}

pub struct AutoTaraModelLoader;

impl AutoTaraModelLoader {
    /// Resolves canonical model directory.
    pub fn resolve_canonical_model_dir(custom_path: Option<&str>) -> PathBuf {
        if let Some(p) = custom_path {
            let path = PathBuf::from(p);
            if path.exists() {
                return path;
            }
        }

        let candidates = [
            "storage/models/tara",
            "../storage/models/tara",
            "../../storage/models/tara",
            "storage/models/tara_candidate_v1",
            "../storage/models/tara_candidate_v1",
            "../../storage/models/tara_candidate_v1",
        ];

        for c in &candidates {
            let p = Path::new(c);
            if p.exists()
                && (p.join("model.safetensors").exists()
                    || p.join("model.safetensors.index.json").exists())
            {
                return p.to_path_buf();
            }
        }

        PathBuf::from("storage/models/tara")
    }

    /// Automatically probe hardware, compute policy plan, and load TARA model.
    pub fn auto_load(
        model_dir: Option<&str>,
        requested_strategy: Option<ModelLoadingStrategy>,
    ) -> Result<LoadedTaraModel, SafeTensorsError> {
        let canonical_dir = Self::resolve_canonical_model_dir(model_dir);
        let profile = DeviceCapabilityDetector::detect();

        let max_cached_shards = resolve_max_cached_shards();
        let mut shard_manager = ShardedSafeTensorsManager::new(&canonical_dir, max_cached_shards)?;
        let total_weights_bytes = shard_manager.total_weights_bytes;
        // Estimate parameter count (assume fp32 or fp16 average ~2 bytes per param)
        let estimated_params = (total_weights_bytes / 2).max(1) as usize;

        let plan = LoadingPolicyEngine::create_loading_plan(
            &profile,
            estimated_params,
            total_weights_bytes,
            requested_strategy,
        )
        .map_err(|_| {
            SafeTensorsError::Io(std::io::Error::other(
                "Insufficient system memory to load TARA model safely",
            ))
        })?;

        let mut full_weights = None;
        if plan.strategy == ModelLoadingStrategy::FullLoad {
            full_weights = Some(shard_manager.load_all_tensors()?);
        }

        Ok(LoadedTaraModel {
            model_dir: canonical_dir,
            profile,
            plan,
            shard_manager,
            full_weights,
            is_fallback_active: false,
            active_fallback_reason: String::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_canonical_model_dir_default() {
        let path = AutoTaraModelLoader::resolve_canonical_model_dir(None);
        assert!(!path.as_os_str().is_empty());
    }

    #[test]
    fn test_resolve_max_cached_shards_default() {
        assert_eq!(resolve_max_cached_shards(), DEFAULT_MAX_CACHED_SHARDS);
    }
}

