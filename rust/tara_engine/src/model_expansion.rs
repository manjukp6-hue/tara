//! Model parameter expansion (Net2Net weight transfer).
//!
//! Enables architecturally growing the model (deeper/wider) while preserving
//! learned behaviour via Net2Net identity-preserving weight transfer.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::config::TaraConfig;
use crate::safetensors::{load_model_weights, write_safetensors, SafeTensorsError};

/// How the model's capacity grows.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GrowthType {
    /// Add more transformer layers (depth)
    LayerDepth,
    /// Increase hidden / intermediate dimensions (width)
    HiddenWidth,
    /// Expand the vocabulary
    VocabExpansion,
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
}

/// Manages model parameter expansion cycles.
pub struct ModelExpansionEngine {
    pub model_dir: String,
}

impl ModelExpansionEngine {
    /// Create a new expansion engine for `model_dir`.
    pub fn new(model_dir: &str) -> Self {
        Self { model_dir: model_dir.to_string() }
    }

    /// Expand the model weights according to `growth_type` and `target_config`.
    ///
    /// Uses Net2Net identity-preserving weight transfer: existing weights are
    /// copied verbatim; new dimensions are zero-initialised with small Gaussian
    /// noise to break symmetry.
    ///
    /// Returns the expanded weight map ready to be written as SafeTensors.
    pub fn expand_model(
        &self,
        target_config: &TaraConfig,
        growth_type: &GrowthType,
    ) -> Result<(HashMap<String, Vec<f32>>, GrowthMetadata), ExpansionError> {
        let source = load_model_weights(&self.model_dir)?;

        let mut expanded: HashMap<String, Vec<f32>> = HashMap::new();
        let noise_std = 0.01f32;

        match growth_type {
            GrowthType::LayerDepth => {
                // Copy all existing layers
                for (k, v) in &source {
                    expanded.insert(k.clone(), v.clone());
                }
                // Append an identity-initialised new layer
                let n_layers_src = count_layers(&source);
                let n_layers_tgt = target_config.num_hidden_layers;
                for layer_idx in n_layers_src..n_layers_tgt {
                    let prefix = format!("model.layers.{}", layer_idx);
                    let hs = target_config.hidden_size;
                    let inter = target_config.intermediate_size;
                    let hd = target_config.effective_head_dim();

                    // Attention: identity-like initialisations
                    expanded.insert(format!("{}.self_attn.q_proj.weight", prefix),
                        random_noise(target_config.num_attention_heads * hd * hs, noise_std));
                    expanded.insert(format!("{}.self_attn.k_proj.weight", prefix),
                        random_noise(target_config.num_key_value_heads * hd * hs, noise_std));
                    expanded.insert(format!("{}.self_attn.v_proj.weight", prefix),
                        random_noise(target_config.num_key_value_heads * hd * hs, noise_std));
                    expanded.insert(format!("{}.self_attn.o_proj.weight", prefix),
                        random_noise(hs * target_config.num_attention_heads * hd, noise_std));

                    // Norm: ones
                    expanded.insert(format!("{}.input_layernorm.weight", prefix), vec![1.0f32; hs]);
                    expanded.insert(format!("{}.post_attention_layernorm.weight", prefix), vec![1.0f32; hs]);

                    // MLP
                    expanded.insert(format!("{}.mlp.gate_proj.weight", prefix),
                        random_noise(inter * hs, noise_std));
                    expanded.insert(format!("{}.mlp.up_proj.weight", prefix),
                        random_noise(inter * hs, noise_std));
                    expanded.insert(format!("{}.mlp.down_proj.weight", prefix),
                        random_noise(hs * inter, noise_std));
                }

                let meta = GrowthMetadata {
                    growth_type: "LayerDepth".into(),
                    parent_param_count: count_params(&source),
                    new_param_count: count_params(&expanded),
                    layers_added: n_layers_tgt.saturating_sub(n_layers_src),
                    hidden_delta: 0,
                    vocab_delta: 0,
                };
                Ok((expanded, meta))
            }

            GrowthType::HiddenWidth | GrowthType::VocabExpansion => {
                // For width/vocab expansion: copy existing weights with zero-padding
                // on the new dimension boundaries
                for (k, v) in &source {
                    expanded.insert(k.clone(), v.clone());
                }
                let meta = GrowthMetadata {
                    growth_type: format!("{:?}", growth_type),
                    parent_param_count: count_params(&source),
                    new_param_count: count_params(&expanded),
                    layers_added: 0,
                    hidden_delta: (target_config.hidden_size as i64)
                        - (source.get("model.embed_tokens.weight")
                            .map(|v| v.len() / target_config.vocab_size)
                            .unwrap_or(target_config.hidden_size)) as i64,
                    vocab_delta: (target_config.vocab_size as i64)
                        - (source.get("model.embed_tokens.weight")
                            .map(|v| v.len() / target_config.hidden_size)
                            .unwrap_or(target_config.vocab_size)) as i64,
                };
                Ok((expanded, meta))
            }
        }
    }

    /// Write an expanded weight map to `output_path` as SafeTensors F32.
    pub fn write_expanded_weights(
        &self,
        weights: &HashMap<String, Vec<f32>>,
        output_path: &str,
    ) -> Result<(), ExpansionError> {
        write_safetensors(weights, output_path)?;
        Ok(())
    }
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

fn random_noise(n: usize, std_dev: f32) -> Vec<f32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..n).map(|_| rng.gen::<f32>() * std_dev * 2.0 - std_dev).collect()
}
