//! Model parameter expansion (Net2Net weight transfer).
//!
//! Enables architecturally growing the model (deeper/wider) while preserving
//! learned behaviour via Net2Net identity-preserving weight transfer.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

use crate::config::TaraConfig;
use crate::safetensors::{load_model_weights, write_safetensors_with_shapes, SafeTensorsError};

/// How the model's capacity grows.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GrowthType {
    /// Add more transformer layers (depth)
    LayerDepth,
    /// Increase hidden / intermediate dimensions (width)
    HiddenWidth,
    /// Expand the vocabulary
    VocabExpansion,
    /// Expand ALL dimensions simultaneously — depth + width + heads + vocab.
    ///
    /// This is the generic parameterised scaling path used by `ArchitectureScaler`.
    /// No dimension may shrink relative to the source model. All newly created
    /// parameters are zero-initialised (output projections) or small-random-noise
    /// (input projections), following Net2Net identity-preservation conventions.
    FullCapacity,
}

/// Metadata captured during a growth cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrowthMetadata {
    pub growth_type: String,
    pub parent_param_count: u64,
    pub new_param_count: u64,
    pub layers_added: usize,
    pub hidden_delta: i64,
    pub vocab_delta: i64,
}

/// Errors that can occur during model expansion.
#[derive(Debug, Error)]
pub enum ExpansionError {
    #[error("safetensors error: {0}")]
    SafeTensors(#[from] SafeTensorsError),
    #[error("missing source weight: {0}")]
    MissingWeight(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("unsupported model growth: {0}")]
    Unsupported(String),
    #[error("invalid model configuration: {0}")]
    Config(#[from] crate::config::ConfigError),
}

/// Manages model parameter expansion cycles.
pub struct ModelExpansionEngine {
    pub model_dir: String,
}

impl ModelExpansionEngine {
    /// Create a new expansion engine for `model_dir`.
    pub fn new(model_dir: &str) -> Self {
        Self {
            model_dir: model_dir.to_string(),
        }
    }

    /// Expand the model weights according to `growth_type` and `target_config`.
    ///
    /// Layer-depth expansion preserves the existing model exactly by adding
    /// residual layers whose attention and MLP output projections are zero.
    ///
    /// Returns the expanded weight map ready to be written as SafeTensors.
    pub fn expand_model(
        &self,
        target_config: &TaraConfig,
        growth_type: &GrowthType,
    ) -> Result<(HashMap<String, Vec<f32>>, GrowthMetadata), ExpansionError> {
        target_config.validate()?;
        let source = load_model_weights(&self.model_dir)?;
        let source_config = TaraConfig::from_json_file(&format!("{}/config.json", self.model_dir))
            .map_err(|error| {
                ExpansionError::Unsupported(format!("cannot read source model config: {error}"))
            })?;

        let source_layer_count = count_layers(&source);
        let dimensions_match = source_config.effective_head_dim()
            == target_config.effective_head_dim()
            && source_config.max_position_embeddings == target_config.max_position_embeddings
            && source_config.rope_theta == target_config.rope_theta
            && source_config.rms_norm_eps == target_config.rms_norm_eps;
        let growth_valid = match growth_type {
            GrowthType::LayerDepth => {
                dimensions_match
                    && source_config.vocab_size == target_config.vocab_size
                    && source_config.hidden_size == target_config.hidden_size
                    && source_config.intermediate_size == target_config.intermediate_size
                    && source_config.num_attention_heads == target_config.num_attention_heads
                    && source_config.num_key_value_heads == target_config.num_key_value_heads
                    && target_config.num_hidden_layers > source_layer_count
            }
            GrowthType::HiddenWidth => {
                dimensions_match
                    && source_config.vocab_size == target_config.vocab_size
                    && source_layer_count == target_config.num_hidden_layers
                    && target_config.hidden_size >= source_config.hidden_size
                    && target_config.intermediate_size >= source_config.intermediate_size
                    && (target_config.hidden_size > source_config.hidden_size
                        || target_config.intermediate_size > source_config.intermediate_size)
            }
            GrowthType::VocabExpansion => {
                dimensions_match
                    && source_config.vocab_size < target_config.vocab_size
                    && source_config.hidden_size == target_config.hidden_size
                    && source_config.intermediate_size == target_config.intermediate_size
                    && source_layer_count == target_config.num_hidden_layers
                    && source_config.num_attention_heads == target_config.num_attention_heads
                    && source_config.num_key_value_heads == target_config.num_key_value_heads
            }
            // FullCapacity allows ALL dimensions to change simultaneously.
            // The only invariants are: no dimension may shrink, and at least one must grow.
            // `dimensions_match` is deliberately NOT required here — head_dim and rope
            // parameters may all change in a full capacity scaling step.
            GrowthType::FullCapacity => {
                let no_shrink = target_config.hidden_size >= source_config.hidden_size
                    && target_config.intermediate_size >= source_config.intermediate_size
                    && target_config.vocab_size >= source_config.vocab_size
                    && target_config.num_hidden_layers >= source_layer_count
                    && target_config.num_attention_heads >= source_config.num_attention_heads
                    && target_config.num_key_value_heads >= source_config.num_key_value_heads;
                let at_least_one_grows = target_config.hidden_size > source_config.hidden_size
                    || target_config.intermediate_size > source_config.intermediate_size
                    || target_config.vocab_size > source_config.vocab_size
                    || target_config.num_hidden_layers > source_layer_count
                    || target_config.num_attention_heads > source_config.num_attention_heads;
                no_shrink && at_least_one_grows
            }
        };
        if !growth_valid {
            return Err(ExpansionError::Unsupported(format!(
                "target configuration is not a valid {:?} expansion from the source model",
                growth_type
            )));
        }
        let embedding = source
            .get("model.embed_tokens.weight")
            .ok_or_else(|| ExpansionError::MissingWeight("model.embed_tokens.weight".into()))?;
        if embedding.len()
            != source_config
                .vocab_size
                .saturating_mul(source_config.hidden_size)
        {
            return Err(ExpansionError::Unsupported(
                "target vocabulary/hidden dimensions do not match the source embedding tensor"
                    .into(),
            ));
        }
        let mut expanded: HashMap<String, Vec<f32>> = HashMap::new();
        for (name, values) in &source {
            expanded.insert(
                name.clone(),
                resize_tensor(name, values, &source_config, target_config)?,
            );
        }
        match growth_type {
            GrowthType::LayerDepth => {
                // Append an identity-initialised new layer
                let n_layers_src = count_layers(&source);
                let n_layers_tgt = target_config.num_hidden_layers;
                for layer_idx in n_layers_src..n_layers_tgt {
                    let prefix = format!("model.layers.{}", layer_idx);
                    let hs = target_config.hidden_size;
                    let inter = target_config.intermediate_size;
                    let hd = target_config.effective_head_dim();

                    // Zero output projection keeps the newly added residual block
                    // exactly identity-preserving before fine-tuning.
                    expanded.insert(
                        format!("{}.self_attn.q_proj.weight", prefix),
                        random_noise(target_config.num_attention_heads * hd * hs, 0.01),
                    );
                    expanded.insert(
                        format!("{}.self_attn.k_proj.weight", prefix),
                        random_noise(target_config.num_key_value_heads * hd * hs, 0.01),
                    );
                    expanded.insert(
                        format!("{}.self_attn.v_proj.weight", prefix),
                        random_noise(target_config.num_key_value_heads * hd * hs, 0.01),
                    );
                    expanded.insert(
                        format!("{}.self_attn.o_proj.weight", prefix),
                        vec![0.0; hs * target_config.num_attention_heads * hd],
                    );

                    // Norm: ones
                    expanded.insert(
                        format!("{}.input_layernorm.weight", prefix),
                        vec![1.0f32; hs],
                    );
                    expanded.insert(
                        format!("{}.post_attention_layernorm.weight", prefix),
                        vec![1.0f32; hs],
                    );

                    // MLP
                    expanded.insert(
                        format!("{}.mlp.gate_proj.weight", prefix),
                        random_noise(inter * hs, 0.01),
                    );
                    expanded.insert(
                        format!("{}.mlp.up_proj.weight", prefix),
                        random_noise(inter * hs, 0.01),
                    );
                    expanded.insert(
                        format!("{}.mlp.down_proj.weight", prefix),
                        vec![0.0; hs * inter],
                    );
                }
            }
            GrowthType::HiddenWidth | GrowthType::VocabExpansion => {}
            // FullCapacity: existing layers are already resized by resize_tensor above.
            // Newly added layers (if target is deeper) are zero-initialised (identity residuals).
            GrowthType::FullCapacity => {
                let n_layers_src = count_layers(&source);
                let n_layers_tgt = target_config.num_hidden_layers;
                let hs = target_config.hidden_size;
                let inter = target_config.intermediate_size;
                let hd = target_config.effective_head_dim();
                for layer_idx in n_layers_src..n_layers_tgt {
                    let prefix = format!("model.layers.{}", layer_idx);
                    // Small random noise on input projections, zero on output projections.
                    // This keeps newly added residual blocks exactly identity-preserving.
                    expanded.insert(
                        format!("{}.self_attn.q_proj.weight", prefix),
                        random_noise(target_config.num_attention_heads * hd * hs, 0.01),
                    );
                    expanded.insert(
                        format!("{}.self_attn.k_proj.weight", prefix),
                        random_noise(target_config.num_key_value_heads * hd * hs, 0.01),
                    );
                    expanded.insert(
                        format!("{}.self_attn.v_proj.weight", prefix),
                        random_noise(target_config.num_key_value_heads * hd * hs, 0.01),
                    );
                    expanded.insert(
                        format!("{}.self_attn.o_proj.weight", prefix),
                        vec![0.0f32; hs * target_config.num_attention_heads * hd],
                    );
                    expanded.insert(
                        format!("{}.input_layernorm.weight", prefix),
                        vec![1.0f32; hs],
                    );
                    expanded.insert(
                        format!("{}.post_attention_layernorm.weight", prefix),
                        vec![1.0f32; hs],
                    );
                    expanded.insert(
                        format!("{}.mlp.gate_proj.weight", prefix),
                        random_noise(inter * hs, 0.01),
                    );
                    expanded.insert(
                        format!("{}.mlp.up_proj.weight", prefix),
                        random_noise(inter * hs, 0.01),
                    );
                    expanded.insert(
                        format!("{}.mlp.down_proj.weight", prefix),
                        vec![0.0f32; hs * inter],
                    );
                }
            }
        }
        let growth_name = match growth_type {
            GrowthType::LayerDepth => "LayerDepth",
            GrowthType::HiddenWidth => "HiddenWidth",
            GrowthType::VocabExpansion => "VocabExpansion",
            GrowthType::FullCapacity => "FullCapacity",
        };
        let metadata = GrowthMetadata {
            growth_type: growth_name.into(),
            parent_param_count: count_params(&source),
            new_param_count: count_params(&expanded),
            layers_added: target_config
                .num_hidden_layers
                .saturating_sub(source_layer_count),
            hidden_delta: target_config.hidden_size as i64 - source_config.hidden_size as i64,
            vocab_delta: target_config.vocab_size as i64 - source_config.vocab_size as i64,
        };
        Ok((expanded, metadata))
    }

    /// Write an expanded weight map to `output_path` as SafeTensors F32.
    pub fn write_expanded_weights(
        &self,
        weights: &HashMap<String, Vec<f32>>,
        output_path: &str,
    ) -> Result<(), ExpansionError> {
        let config_path = format!("{}/config.json", self.model_dir);
        let config = TaraConfig::from_json_file(&config_path).map_err(|error| {
            ExpansionError::Unsupported(format!("cannot read source model config: {error}"))
        })?;
        self.write_expanded_weights_for_config(weights, &config, output_path)
    }

    /// Write weights using the target architecture's tensor dimensions.
    pub fn write_expanded_weights_for_config(
        &self,
        weights: &HashMap<String, Vec<f32>>,
        target_config: &TaraConfig,
        output_path: &str,
    ) -> Result<(), ExpansionError> {
        target_config.validate()?;
        let shapes = weights
            .iter()
            .map(|(name, values)| {
                (
                    name.clone(),
                    tensor_shape(name, values.len(), target_config),
                )
            })
            .collect();
        write_safetensors_with_shapes(weights, &shapes, output_path)?;
        Ok(())
    }

    /// Write an inference-loadable expanded model directory without modifying
    /// the source model. The destination must not already exist.
    pub fn write_expanded_model(
        &self,
        weights: &HashMap<String, Vec<f32>>,
        target_config: &TaraConfig,
        output_dir: &str,
    ) -> Result<(), ExpansionError> {
        self.write_expanded_model_with_tokens(weights, target_config, output_dir, &[])
    }

    /// Write an expanded model and append caller-supplied tokens when the
    /// vocabulary grows. Every new model row must have exactly one new token.
    pub fn write_expanded_model_with_tokens(
        &self,
        weights: &HashMap<String, Vec<f32>>,
        target_config: &TaraConfig,
        output_dir: &str,
        added_tokens: &[String],
    ) -> Result<(), ExpansionError> {
        let destination = Path::new(output_dir);
        if destination.exists() {
            return Err(ExpansionError::Unsupported(
                "expanded model destination already exists".into(),
            ));
        }
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let temp_dir = destination.with_extension(format!("{stamp}.tmp"));
        fs::create_dir(&temp_dir)?;
        let result = (|| {
            let tokenizer_source = Path::new(&self.model_dir).join("tokenizer.json");
            let tokenizer_raw = fs::read_to_string(tokenizer_source)?;
            let mut tokenizer: serde_json::Value = serde_json::from_str(&tokenizer_raw)
                .map_err(|error| ExpansionError::Unsupported(error.to_string()))?;
            let source_config = TaraConfig::from_json_file(
                &Path::new(&self.model_dir)
                    .join("config.json")
                    .to_string_lossy(),
            )
            .map_err(|error| ExpansionError::Unsupported(error.to_string()))?;
            let vocab = tokenizer
                .get_mut("vocab")
                .and_then(serde_json::Value::as_object_mut)
                .ok_or_else(|| {
                    ExpansionError::Unsupported("tokenizer has no vocabulary object".into())
                })?;
            let source_vocab_size = vocab.len();
            if source_vocab_size != source_config.vocab_size {
                return Err(ExpansionError::Unsupported(
                    "source tokenizer vocabulary does not match the model config".into(),
                ));
            }
            let required_tokens = target_config.vocab_size.saturating_sub(source_vocab_size);
            if added_tokens.len() != required_tokens {
                return Err(ExpansionError::Unsupported(format!(
                    "vocabulary expansion requires exactly {required_tokens} unique added tokens"
                )));
            }
            for (offset, token) in added_tokens.iter().enumerate() {
                if token.trim().is_empty() || vocab.contains_key(token) {
                    return Err(ExpansionError::Unsupported(
                        "added tokens must be non-empty and unique".into(),
                    ));
                }
                vocab.insert(token.clone(), serde_json::json!(source_vocab_size + offset));
            }
            fs::write(
                temp_dir.join("tokenizer.json"),
                serde_json::to_vec_pretty(&tokenizer)
                    .map_err(|error| ExpansionError::Unsupported(error.to_string()))?,
            )?;
            fs::write(
                temp_dir.join("config.json"),
                serde_json::to_vec_pretty(target_config)
                    .map_err(|error| ExpansionError::Unsupported(error.to_string()))?,
            )?;
            let shapes = weights
                .iter()
                .map(|(name, values)| {
                    (
                        name.clone(),
                        tensor_shape(name, values.len(), target_config),
                    )
                })
                .collect();
            write_safetensors_with_shapes(
                weights,
                &shapes,
                &temp_dir.join("model.safetensors").to_string_lossy(),
            )?;
            fs::rename(&temp_dir, destination)?;
            Ok::<(), ExpansionError>(())
        })();
        if result.is_err() {
            let _ = fs::remove_dir_all(&temp_dir);
        }
        result
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Architecture Scaler — parameterised target-size solver
// ─────────────────────────────────────────────────────────────────────────────

/// Explicit architecture constraints and preferences for model expansion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureConstraints {
    /// Minimum allowed transformer decoder layers.
    pub min_layers: usize,
    /// Maximum allowed transformer decoder layers.
    pub max_layers: usize,
    /// Preferred layer count if reachable.
    pub preferred_layers: Option<usize>,
    /// Minimum allowed hidden dimension.
    pub min_hidden: usize,
    /// Maximum allowed hidden dimension.
    pub max_hidden: usize,
    /// Preferred hidden dimension.
    pub preferred_hidden: Option<usize>,
    /// Explicit override for number of attention heads.
    pub preferred_heads: Option<usize>,
    /// Explicit override for number of key/value heads.
    pub preferred_kv_heads: Option<usize>,
    /// Explicit override for intermediate (SwiGLU) size.
    pub preferred_intermediate: Option<usize>,
    /// Head dimension (default: 64).
    pub head_dim: usize,
    /// Target GQA grouping ratio num_heads / num_kv_heads (default: 4).
    pub gqa_ratio: usize,
}

impl ArchitectureConstraints {
    /// Construct default architectural family constraints tailored to `target_params`.
    /// Fully parameterised across model sizes: 10M, 50M, 100M, 300M, 500M, 1B, 2B, 3B, 7B+.
    pub fn for_target(target_params: u64) -> Self {
        if target_params >= 5_000_000_000 {
            // ~7B architecture family (standard 7B LLaMA-class transformer)
            // hidden: 4096, layers: 32-38, heads: 32, kv: 8, head_dim: 128, inter: 11008, GQA 4:1
            Self {
                min_layers: 28,
                max_layers: 40,
                preferred_layers: Some(38),
                min_hidden: 3584,
                max_hidden: 4608,
                preferred_hidden: Some(4096),
                preferred_heads: Some(32),
                preferred_kv_heads: Some(8),
                preferred_intermediate: Some(11008),
                head_dim: 128,
                gqa_ratio: 4,
            }
        } else if target_params >= 2_500_000_000 {
            // ~3B architecture family
            // hidden: 3072, layers: 26-34, heads: 48, kv: 12, head_dim: 64, inter: 8192, GQA 4:1
            Self {
                min_layers: 24,
                max_layers: 34,
                preferred_layers: Some(30),
                min_hidden: 2816,
                max_hidden: 3328,
                preferred_hidden: Some(3072),
                preferred_heads: Some(48),
                preferred_kv_heads: Some(12),
                preferred_intermediate: Some(8192),
                head_dim: 64,
                gqa_ratio: 4,
            }
        } else if target_params >= 1_500_000_000 {
            // ~2B architecture family
            // hidden: 2560, layers: 24-32, heads: 40, kv: 10, head_dim: 64, inter: 6912, GQA 4:1
            Self {
                min_layers: 22,
                max_layers: 32,
                preferred_layers: Some(28),
                min_hidden: 2304,
                max_hidden: 2816,
                preferred_hidden: Some(2560),
                preferred_heads: Some(40),
                preferred_kv_heads: Some(10),
                preferred_intermediate: Some(6912),
                head_dim: 64,
                gqa_ratio: 4,
            }
        } else if target_params >= 750_000_000 {
            // ~1B architecture family
            // hidden: 2048, layers: 18-28, heads: 32, kv: 8, head_dim: 64, inter: 5504, GQA 4:1
            Self {
                min_layers: 18,
                max_layers: 28,
                preferred_layers: Some(22),
                min_hidden: 1792,
                max_hidden: 2304,
                preferred_hidden: Some(2048),
                preferred_heads: Some(32),
                preferred_kv_heads: Some(8),
                preferred_intermediate: Some(5504),
                head_dim: 64,
                gqa_ratio: 4,
            }
        } else if target_params >= 350_000_000 {
            // ~500M architecture family
            // hidden: 1536, layers: 16-24, heads: 24, kv: 6, head_dim: 64, inter: 4096, GQA 4:1
            Self {
                min_layers: 16,
                max_layers: 24,
                preferred_layers: Some(19),
                min_hidden: 1280,
                max_hidden: 1792,
                preferred_hidden: Some(1536),
                preferred_heads: Some(24),
                preferred_kv_heads: Some(6),
                preferred_intermediate: Some(4096),
                head_dim: 64,
                gqa_ratio: 4,
            }
        } else if target_params >= 150_000_000 {
            // ~300M architecture family
            // hidden: 1280, layers: 14-20, heads: 20, kv: 5, head_dim: 64, inter: 3456, GQA 4:1
            Self {
                min_layers: 14,
                max_layers: 20,
                preferred_layers: Some(16),
                min_hidden: 1024,
                max_hidden: 1536,
                preferred_hidden: Some(1280),
                preferred_heads: Some(20),
                preferred_kv_heads: Some(5),
                preferred_intermediate: Some(3456),
                head_dim: 64,
                gqa_ratio: 4,
            }
        } else if target_params >= 60_000_000 {
            // ~100M architecture family
            Self {
                min_layers: 12,
                max_layers: 18,
                preferred_layers: Some(14),
                min_hidden: 768,
                max_hidden: 1024,
                preferred_hidden: Some(768),
                preferred_heads: Some(12),
                preferred_kv_heads: Some(4),
                preferred_intermediate: Some(2048),
                head_dim: 64,
                gqa_ratio: 3,
            }
        } else if target_params >= 25_000_000 {
            // ~50M architecture family
            Self {
                min_layers: 8,
                max_layers: 14,
                preferred_layers: Some(10),
                min_hidden: 512,
                max_hidden: 768,
                preferred_hidden: Some(512),
                preferred_heads: Some(8),
                preferred_kv_heads: Some(2),
                preferred_intermediate: Some(1344),
                head_dim: 64,
                gqa_ratio: 4,
            }
        } else {
            // ~10M architecture family
            Self {
                min_layers: 4,
                max_layers: 8,
                preferred_layers: Some(6),
                min_hidden: 256,
                max_hidden: 384,
                preferred_hidden: Some(256),
                preferred_heads: Some(4),
                preferred_kv_heads: Some(2),
                preferred_intermediate: Some(640),
                head_dim: 64,
                gqa_ratio: 2,
            }
        }
    }
}

/// Solves for a `TaraConfig` that approximately reaches a requested parameter
/// count using a balanced, standard transformer aspect ratio and GQA grouping.
pub struct ArchitectureScaler;

impl ArchitectureScaler {
    /// Return the `TaraConfig` and the exact parameter count that best satisfies
    /// `target_params` while adhering strictly to transformer architectural constraints.
    pub fn solve(target_params: u64, vocab_size: usize) -> (TaraConfig, u64) {
        let constraints = ArchitectureConstraints::for_target(target_params);
        Self::solve_with_constraints(target_params, vocab_size, &constraints)
    }

    /// Solve with explicit user-provided constraints and overrides.
    pub fn solve_with_constraints(
        target_params: u64,
        vocab_size: usize,
        constraints: &ArchitectureConstraints,
    ) -> (TaraConfig, u64) {
        let head_dim = constraints.head_dim;
        let gqa_ratio = constraints.gqa_ratio.max(1);

        // Step size for hidden dimension search: must ensure num_heads % gqa_ratio == 0
        let hs_step = head_dim * gqa_ratio;

        let mut candidate_hidden: Vec<usize> = Vec::new();
        if let Some(pref_hs) = constraints.preferred_hidden {
            candidate_hidden.push(pref_hs);
        }
        let mut h = constraints.min_hidden;
        if !h.is_multiple_of(hs_step) {
            h = h.div_ceil(hs_step) * hs_step;
        }
        while h <= constraints.max_hidden {
            if !candidate_hidden.contains(&h) {
                candidate_hidden.push(h);
            }
            h += hs_step;
        }

        let mut best_config: Option<TaraConfig> = None;
        let mut best_score = f64::MAX;
        let mut best_actual = 0u64;

        for &hs in &candidate_hidden {
            let num_heads = match constraints.preferred_heads {
                Some(nh) => nh,
                None => hs / head_dim,
            };
            if num_heads == 0 || num_heads * head_dim != hs {
                continue;
            }
            let num_kv_heads = match constraints.preferred_kv_heads {
                Some(nkv) => nkv,
                None => (num_heads / gqa_ratio).max(1),
            };
            if !num_heads.is_multiple_of(num_kv_heads) {
                continue;
            }
            let intermediate_size = match constraints.preferred_intermediate {
                Some(inter) => inter,
                None => {
                    let raw = (hs * 8) / 3;
                    raw.div_ceil(64) * 64
                }
            };

            for nl in constraints.min_layers..=constraints.max_layers {
                let actual = Self::count_params(
                    vocab_size,
                    hs,
                    intermediate_size,
                    nl,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                );

                // Relative parameter error
                let param_delta_pct =
                    (actual as f64 - target_params as f64).abs() / target_params as f64;

                // Layer preference penalty (0.5% effective delta per layer away from preferred)
                let layer_penalty = if let Some(pref_l) = constraints.preferred_layers {
                    (nl as f64 - pref_l as f64).abs() * 0.005
                } else {
                    0.0
                };

                // Hidden preference penalty (5% effective delta per 100% hidden shift)
                let hidden_penalty = if let Some(pref_h) = constraints.preferred_hidden {
                    ((hs as f64 - pref_h as f64).abs() / pref_h as f64) * 0.05
                } else {
                    0.0
                };

                // Aspect ratio sanity: hs / nl should normally be between 40 and 130
                let aspect = hs as f64 / nl as f64;
                let aspect_penalty = if aspect > 130.0 {
                    (aspect - 130.0) * 0.01
                } else if aspect < 40.0 {
                    (40.0 - aspect) * 0.01
                } else {
                    0.0
                };

                let score = param_delta_pct + layer_penalty + hidden_penalty + aspect_penalty;

                if score < best_score {
                    best_score = score;
                    best_actual = actual;
                    best_config = Some(TaraConfig {
                        vocab_size,
                        hidden_size: hs,
                        intermediate_size,
                        num_hidden_layers: nl,
                        num_attention_heads: num_heads,
                        num_key_value_heads: num_kv_heads,
                        head_dim,
                        max_position_embeddings: 4096,
                        rope_theta: 500_000.0,
                        rms_norm_eps: 1e-5,
                        initializer_range: 0.02,
                        version: "2.0.0".to_string(),
                        model_type: "tara".to_string(),
                    });
                }
            }
        }

        (
            best_config.expect("ArchitectureScaler must find a valid configuration"),
            best_actual,
        )
    }

    /// Compute the exact total parameter count for the given architecture
    /// dimensions. This mirrors the tensor layout used by `TaraForCausalLM`:
    ///
    /// * `embed_tokens`: `vocab × hidden`
    /// * `model.norm`:   `hidden`
    /// * Per decoder layer (×num_layers):
    ///   * q\_proj: `num_heads × head_dim × hidden`
    ///   * k\_proj / v\_proj: `num_kv_heads × head_dim × hidden` each
    ///   * o\_proj: `hidden × num_heads × head_dim`
    ///   * input\_layernorm / post\_attention\_layernorm: `hidden` each
    ///   * gate\_proj / up\_proj: `intermediate × hidden` each
    ///   * down\_proj: `hidden × intermediate`
    /// * `lm_head`: `vocab × hidden`
    pub fn count_params(
        vocab_size: usize,
        hidden_size: usize,
        intermediate_size: usize,
        num_layers: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> u64 {
        let embed = (vocab_size * hidden_size) as u64;
        let lm_head = (vocab_size * hidden_size) as u64;
        let norm = hidden_size as u64;
        let per_layer = (num_heads * head_dim * hidden_size         // q_proj
            + num_kv_heads * head_dim * hidden_size                  // k_proj
            + num_kv_heads * head_dim * hidden_size                  // v_proj
            + hidden_size * num_heads * head_dim                     // o_proj
            + hidden_size                                            // input_layernorm
            + hidden_size                                            // post_attn_layernorm
            + intermediate_size * hidden_size                        // gate_proj
            + intermediate_size * hidden_size                        // up_proj
            + hidden_size * intermediate_size) as u64; // down_proj
        embed + norm + (num_layers as u64) * per_layer + lm_head
    }
}

fn resize_tensor(
    name: &str,
    source: &[f32],
    source_config: &TaraConfig,
    target_config: &TaraConfig,
) -> Result<Vec<f32>, ExpansionError> {
    let old_h = source_config.hidden_size;
    let new_h = target_config.hidden_size;
    let old_i = source_config.intermediate_size;
    let new_i = target_config.intermediate_size;
    let old_hd = source_config.effective_head_dim();
    let new_hd = target_config.effective_head_dim();
    let old_shape = if name == "model.embed_tokens.weight" || name == "lm_head.weight" {
        Some((source_config.vocab_size, old_h))
    } else if name.ends_with("self_attn.q_proj.weight") {
        Some((source_config.num_attention_heads * old_hd, old_h))
    } else if name.ends_with("self_attn.k_proj.weight") || name.ends_with("self_attn.v_proj.weight")
    {
        Some((source_config.num_key_value_heads * old_hd, old_h))
    } else if name.ends_with("self_attn.o_proj.weight") {
        Some((old_h, source_config.num_attention_heads * old_hd))
    } else if name.ends_with("mlp.gate_proj.weight") || name.ends_with("mlp.up_proj.weight") {
        Some((old_i, old_h))
    } else if name.ends_with("mlp.down_proj.weight") {
        Some((old_h, old_i))
    } else {
        None
    };
    if let Some((old_rows, old_cols)) = old_shape {
        if old_rows.checked_mul(old_cols) != Some(source.len()) {
            return Err(ExpansionError::Unsupported(format!(
                "source tensor '{name}' has an invalid shape"
            )));
        }
        let (new_rows, new_cols) = if name == "model.embed_tokens.weight"
            || name == "lm_head.weight"
        {
            (target_config.vocab_size, new_h)
        } else if name.ends_with("self_attn.q_proj.weight") {
            (target_config.num_attention_heads * new_hd, new_h)
        } else if name.ends_with("self_attn.k_proj.weight")
            || name.ends_with("self_attn.v_proj.weight")
        {
            (target_config.num_key_value_heads * new_hd, new_h)
        } else if name.ends_with("self_attn.o_proj.weight") {
            (new_h, target_config.num_attention_heads * new_hd)
        } else if name.ends_with("mlp.gate_proj.weight") || name.ends_with("mlp.up_proj.weight") {
            (new_i, new_h)
        } else {
            (new_h, new_i)
        };
        let mut resized = vec![0.0; new_rows.saturating_mul(new_cols)];
        let copied_rows = old_rows.min(new_rows);
        let copied_cols = old_cols.min(new_cols);
        for row in 0..copied_rows {
            let src = row * old_cols;
            let dst = row * new_cols;
            resized[dst..dst + copied_cols].copy_from_slice(&source[src..src + copied_cols]);
        }
        if matches!(name, "model.embed_tokens.weight" | "lm_head.weight") && new_rows > old_rows {
            for row in old_rows..new_rows {
                let start = row * new_cols;
                resized[start..start + new_cols].copy_from_slice(&random_noise(new_cols, 0.02));
            }
        }
        return Ok(resized);
    }
    if name.ends_with("norm.weight") || name.ends_with("layernorm.weight") {
        if source.len() != old_h {
            return Err(ExpansionError::Unsupported(format!(
                "source tensor '{name}' has an invalid normalization width"
            )));
        }
        let mut resized = vec![1.0; new_h];
        resized[..old_h.min(new_h)].copy_from_slice(&source[..old_h.min(new_h)]);
        return Ok(resized);
    }
    Ok(source.to_vec())
}

pub(crate) fn tensor_shape(name: &str, value_count: usize, config: &TaraConfig) -> Vec<usize> {
    let hs = config.hidden_size;
    let hd = config.effective_head_dim();
    let shape = if name == "model.embed_tokens.weight" || name == "lm_head.weight" {
        vec![config.vocab_size, hs]
    } else if name.ends_with("self_attn.q_proj.weight") {
        vec![config.num_attention_heads * hd, hs]
    } else if name.ends_with("self_attn.k_proj.weight") || name.ends_with("self_attn.v_proj.weight")
    {
        vec![config.num_key_value_heads * hd, hs]
    } else if name.ends_with("self_attn.o_proj.weight") {
        vec![hs, config.num_attention_heads * hd]
    } else if name.ends_with("mlp.gate_proj.weight") || name.ends_with("mlp.up_proj.weight") {
        vec![config.intermediate_size, hs]
    } else if name.ends_with("mlp.down_proj.weight") {
        vec![hs, config.intermediate_size]
    } else if name.ends_with("norm.weight") || name.ends_with("layernorm.weight") {
        vec![hs]
    } else {
        vec![value_count]
    };
    if shape.iter().product::<usize>() == value_count {
        shape
    } else {
        vec![value_count]
    }
}

fn random_noise(count: usize, standard_deviation: f32) -> Vec<f32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..count)
        .map(|_| (rng.gen::<f32>() * 2.0 - 1.0) * standard_deviation)
        .collect()
}

fn count_layers(weights: &HashMap<String, Vec<f32>>) -> usize {
    let mut max = 0usize;
    for k in weights.keys() {
        if let Some(rest) = k.strip_prefix("model.layers.") {
            if let Some(idx_str) = rest.split('.').next() {
                if let Ok(idx) = idx_str.parse::<usize>() {
                    if idx + 1 > max {
                        max = idx + 1;
                    }
                }
            }
        }
    }
    max
}

fn count_params(weights: &HashMap<String, Vec<f32>>) -> u64 {
    weights.values().map(|v| v.len() as u64).sum()
}

#[cfg(test)]
mod tests {
    use super::resize_tensor;
    use crate::config::TaraConfig;

    fn config() -> TaraConfig {
        TaraConfig {
            vocab_size: 2,
            hidden_size: 2,
            intermediate_size: 3,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 2,
            max_position_embeddings: 8,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            initializer_range: 0.02,
            version: "test".into(),
            model_type: "tara".into(),
        }
    }

    #[test]
    fn vocab_growth_preserves_existing_rows_and_initializes_new_tokens() {
        let source_cfg = config();
        let mut target_cfg = source_cfg.clone();
        target_cfg.vocab_size = 3;
        let expanded = resize_tensor(
            "model.embed_tokens.weight",
            &[1.0, 2.0, 3.0, 4.0],
            &source_cfg,
            &target_cfg,
        )
        .unwrap();
        assert_eq!(&expanded[..4], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(expanded.len(), 6);
        assert!(expanded[4..].iter().all(|value| value.is_finite()));
        assert!(expanded[4..].iter().any(|value| *value != 0.0));
    }

    #[test]
    fn width_growth_remaps_projection_and_normalization_shapes() {
        let source_cfg = config();
        let mut target_cfg = source_cfg.clone();
        target_cfg.hidden_size = 4;
        target_cfg.intermediate_size = 5;
        target_cfg.num_attention_heads = 2;
        target_cfg.num_key_value_heads = 2;
        let projection = resize_tensor(
            "model.layers.0.self_attn.q_proj.weight",
            &[1.0, 2.0, 3.0, 4.0],
            &source_cfg,
            &target_cfg,
        )
        .unwrap();
        assert_eq!(projection.len(), 16);
        assert_eq!(&projection[..4], &[1.0, 2.0, 0.0, 0.0]);
        assert_eq!(&projection[4..8], &[3.0, 4.0, 0.0, 0.0]);
        let norm =
            resize_tensor("model.norm.weight", &[0.5, 0.75], &source_cfg, &target_cfg).unwrap();
        assert_eq!(norm, vec![0.5, 0.75, 1.0, 1.0]);
    }

    #[test]
    fn test_architecture_scaler_solves_1b_correctly() {
        use super::ArchitectureScaler;
        let (config, exact_params) = ArchitectureScaler::solve(1_000_000_000, 8192);

        assert_eq!(config.vocab_size, 8192);
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.intermediate_size, 5504);
        assert_eq!(config.num_hidden_layers, 22);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_key_value_heads, 8);
        assert_eq!(config.head_dim, 64);
        assert_eq!(config.num_attention_heads / config.num_key_value_heads, 4); // GQA 4:1
        assert_eq!(
            config.head_dim * config.num_attention_heads,
            config.hidden_size
        );

        assert_eq!(exact_params, 1_008_297_984);
        let delta_pct = (exact_params as f64 - 1_000_000_000.0).abs() / 1_000_000_000.0 * 100.0;
        assert!(
            delta_pct < 1.0,
            "Delta must be under 1% of 1B (was {:.3}%)",
            delta_pct
        );
    }

    #[test]
    fn test_architecture_scaler_respects_explicit_overrides() {
        use super::{ArchitectureConstraints, ArchitectureScaler};
        let mut constraints = ArchitectureConstraints::for_target(1_000_000_000);
        constraints.preferred_layers = Some(24);
        constraints.min_layers = 24;
        constraints.max_layers = 24;

        let (config, exact_params) =
            ArchitectureScaler::solve_with_constraints(1_000_000_000, 8192, &constraints);
        assert_eq!(config.num_hidden_layers, 24);
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(exact_params, 1_096_910_848);
    }

    #[test]
    fn test_architecture_scaler_solves_multi_capacity_targets_automatically() {
        use super::ArchitectureScaler;

        // 500M target
        let (cfg_500m, p_500m) = ArchitectureScaler::solve(500_000_000, 8192);
        assert_eq!(cfg_500m.hidden_size, 1536);
        assert_eq!(cfg_500m.num_attention_heads, 24);
        assert_eq!(cfg_500m.num_key_value_heads, 6);
        assert_eq!(cfg_500m.num_hidden_layers, 19);
        assert!((p_500m as f64 - 500_000_000.0).abs() / 500_000_000.0 < 0.02);

        // 1B target
        let (cfg_1b, p_1b) = ArchitectureScaler::solve(1_000_000_000, 8192);
        assert_eq!(cfg_1b.hidden_size, 2048);
        assert_eq!(cfg_1b.num_hidden_layers, 22);
        assert_eq!(p_1b, 1_008_297_984);

        // 2B target
        let (cfg_2b, p_2b) = ArchitectureScaler::solve(2_000_000_000, 8192);
        assert_eq!(cfg_2b.hidden_size, 2560);
        assert_eq!(cfg_2b.num_attention_heads, 40);
        assert_eq!(cfg_2b.num_key_value_heads, 10);
        assert_eq!(cfg_2b.num_hidden_layers, 28);
        assert!((p_2b as f64 - 2_000_000_000.0).abs() / 2_000_000_000.0 < 0.02);

        // 3B target
        let (cfg_3b, p_3b) = ArchitectureScaler::solve(3_000_000_000, 8192);
        assert_eq!(cfg_3b.hidden_size, 3072);
        assert_eq!(cfg_3b.num_attention_heads, 48);
        assert_eq!(cfg_3b.num_key_value_heads, 12);
        assert!((p_3b as f64 - 3_000_000_000.0).abs() / 3_000_000_000.0 < 0.05);

        // 7B target
        let (cfg_7b, p_7b) = ArchitectureScaler::solve(7_000_000_000, 8192);
        assert_eq!(cfg_7b.hidden_size, 4096);
        assert_eq!(cfg_7b.head_dim, 128);
        assert_eq!(cfg_7b.num_attention_heads, 32);
        assert_eq!(cfg_7b.num_key_value_heads, 8);
        assert!((p_7b as f64 - 7_000_000_000.0).abs() / 7_000_000_000.0 < 0.05);
    }
}
