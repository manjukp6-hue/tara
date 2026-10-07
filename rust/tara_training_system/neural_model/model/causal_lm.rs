//! Top-level causal language model — TaraForCausalLM.
//!
//! Loads weights from a SafeTensors file (or shard set), constructs the full
//! transformer stack, and runs autoregressive forward passes.

use std::collections::HashMap;
use thiserror::Error;

use super::decoder_layer::TaraDecoderLayer;
use super::rms_norm::RmsNorm;
use crate::config::TaraConfig;
use crate::safetensors::{compute_sha256, load_model_weights, SafeTensorsError};

// SHA-256 integrity validation is performed at runtime by reading the active
// version's registered hash from ModelRegistry (versions_manifest.json).
// No hardcoded SHA is embedded here — a hardcoded SHA would become stale silently
// after any model promotion cycle.

/// Errors that can occur while loading or running TaraForCausalLM.
#[derive(Debug, Error)]
pub enum ModelError {
    #[error("config error: {0}")]
    Config(#[from] crate::config::ConfigError),
    #[error("weights error: {0}")]
    Weights(#[from] SafeTensorsError),
    #[error("missing weight: {0}")]
    MissingWeight(String),
    #[error("weight '{name}' has {actual} values; expected {expected}")]
    InvalidWeightShape {
        name: String,
        actual: usize,
        expected: usize,
    },
    #[error("attention error: {0}")]
    Attention(#[from] super::attention::AttentionError),
    #[error("mlp error: {0}")]
    Mlp(#[from] super::mlp::MlpError),
    #[error("decoder layer error: {0}")]
    DecoderLayer(#[from] super::decoder_layer::DecoderLayerError),
}

/// KV cache for the full transformer model stack.
#[derive(Debug, Clone, Default)]
pub struct ModelKvCache {
    pub layers: Vec<super::attention::LayerKvCache>,
}

impl ModelKvCache {
    /// Construct a new empty KV cache with `num_layers`.
    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers)
                .map(|_| super::attention::LayerKvCache::new())
                .collect(),
        }
    }

    /// Reset all cached keys and values.
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
    }

    /// Current cached sequence length.
    pub fn seq_len(&self) -> usize {
        self.layers.first().map(|l| l.seq_len).unwrap_or(0)
    }
}

/// TARA causal language model.
///
/// Implements the full forward pass: token embedding lookup → N decoder layers
/// → final RMSNorm → language-model head (unembedding projection).
pub struct TaraForCausalLM {
    /// Token embedding table: shape `[vocab_size, hidden_size]`.
    pub embed_tokens: Vec<f32>,
    /// Transformer decoder layers.
    pub layers: Vec<TaraDecoderLayer>,
    /// Final layer normalisation.
    pub norm: RmsNorm,
    /// Language-model head: shape `[vocab_size, hidden_size]`.
    pub lm_head: Vec<f32>,
    /// Model configuration.
    pub config: TaraConfig,
    /// Resolved model directory used during load.
    pub model_dir: String,
}

impl TaraForCausalLM {
    /// Load the model from `model_dir`.
    ///
    /// Reads `config.json`, then loads `model.safetensors` (or sharded weights
    /// via `model.safetensors.index.json`).
    ///
    /// # Errors
    /// Returns [`ModelError`] on any I/O, parse, or missing-weight failure.
    pub fn load(model_dir: &str) -> Result<Self, ModelError> {
        let config_path = format!("{}/config.json", model_dir);
        let config = TaraConfig::from_json_file(&config_path)?;
        let weights = load_model_weights(model_dir)?;
        Self::from_weights_and_config(weights, config, model_dir)
    }

    /// Construct a model directly from a weights map and config.
    pub fn from_weights_and_config(
        weights: HashMap<String, Vec<f32>>,
        config: TaraConfig,
        model_dir: &str,
    ) -> Result<Self, ModelError> {
        validate_weight_len(
            &weights,
            "model.embed_tokens.weight",
            config.vocab_size * config.hidden_size,
        )?;
        validate_weight_len(&weights, "model.norm.weight", config.hidden_size)?;
        if weights.contains_key("lm_head.weight") {
            validate_weight_len(
                &weights,
                "lm_head.weight",
                config.vocab_size * config.hidden_size,
            )?;
        }
        let hs = config.hidden_size;
        let hd = config.effective_head_dim();
        for layer in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{layer}");
            for (suffix, count) in [
                (
                    "self_attn.q_proj.weight",
                    config.num_attention_heads * hd * hs,
                ),
                (
                    "self_attn.k_proj.weight",
                    config.num_key_value_heads * hd * hs,
                ),
                (
                    "self_attn.v_proj.weight",
                    config.num_key_value_heads * hd * hs,
                ),
                (
                    "self_attn.o_proj.weight",
                    hs * config.num_attention_heads * hd,
                ),
                ("input_layernorm.weight", hs),
                ("post_attention_layernorm.weight", hs),
                ("mlp.gate_proj.weight", config.intermediate_size * hs),
                ("mlp.up_proj.weight", config.intermediate_size * hs),
                ("mlp.down_proj.weight", hs * config.intermediate_size),
            ] {
                validate_weight_len(&weights, &format!("{prefix}.{suffix}"), count)?;
            }
        }

        // Token embeddings
        let embed_tokens = weights
            .get("model.embed_tokens.weight")
            .cloned()
            .ok_or_else(|| ModelError::MissingWeight("model.embed_tokens.weight".into()))?;

        // Decoder layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(TaraDecoderLayer::from_weights(&weights, i, &config)?);
        }

        // Final norm
        let norm_w = weights
            .get("model.norm.weight")
            .cloned()
            .ok_or_else(|| ModelError::MissingWeight("model.norm.weight".into()))?;
        let norm = RmsNorm::new(norm_w, config.rms_norm_eps as f32);

        // LM head — may be tied to embed_tokens
        let lm_head = weights
            .get("lm_head.weight")
            .cloned()
            .unwrap_or_else(|| embed_tokens.clone());

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            config,
            model_dir: model_dir.to_string(),
        })
    }

    /// Run the forward pass over a sequence of token IDs.
    ///
    /// Returns a flat `[seq_len * vocab_size]` logit buffer.  The logits for
    /// the last token are at `result[(seq_len-1)*vocab_size..]`.
    pub fn forward(&self, token_ids: &[u32]) -> Vec<f32> {
        let seq_len = token_ids.len();
        let hs = self.config.hidden_size;
        let vs = self.config.vocab_size;
        let hidden = self.forward_hidden(token_ids);

        // LM head — [seq_len, vocab_size]
        // lm_head shape [vs, hs]; output = hidden @ lm_head.T
        let mut logits = vec![0.0f32; seq_len * vs];
        for t in 0..seq_len {
            let h_row = &hidden[t * hs..(t + 1) * hs];
            let l_row = &mut logits[t * vs..(t + 1) * vs];
            for (v, val) in l_row.iter_mut().enumerate().take(vs) {
                let lm_row = &self.lm_head[v * hs..(v + 1) * hs];
                *val = h_row.iter().zip(lm_row.iter()).map(|(&a, &b)| a * b).sum();
            }
        }

        logits
    }

    /// Construct a new [`ModelKvCache`] initialized for this model.
    pub fn create_kv_cache(&self) -> ModelKvCache {
        ModelKvCache::new(self.layers.len())
    }

    /// Single-token forward pass with KV caching.
    ///
    /// Computes the `[vocab_size]` logits for the next token given `token_id`,
    /// appending key/value states to `cache` at position `cache.seq_len()`.
    pub fn forward_token_cached(&self, token_id: u32, cache: &mut ModelKvCache) -> Vec<f32> {
        let hs = self.config.hidden_size;
        let vs = self.config.vocab_size;

        let id = (token_id as usize).min(vs.saturating_sub(1));
        let mut h = self.embed_tokens[id * hs..(id + 1) * hs].to_vec();

        for (layer, layer_cache) in self.layers.iter().zip(cache.layers.iter_mut()) {
            h = layer.forward_cached(&h, layer_cache);
        }

        let normed = self.norm.forward(&h, 1, hs);

        let mut logits = vec![0.0f32; vs];
        for (v, val) in logits.iter_mut().enumerate().take(vs) {
            let lm_row = &self.lm_head[v * hs..(v + 1) * hs];
            *val = normed.iter().zip(lm_row.iter()).map(|(&a, &b)| a * b).sum();
        }

        logits
    }

    /// Return the final normalized hidden state for each input token.
    /// This is used by supervised output-head fine-tuning.
    pub fn forward_hidden(&self, token_ids: &[u32]) -> Vec<f32> {
        if token_ids.is_empty() {
            return Vec::new();
        }
        let seq_len = token_ids.len();
        let hs = self.config.hidden_size;
        let mut hidden = vec![0.0f32; seq_len * hs];
        for (t, &id) in token_ids.iter().enumerate() {
            let id = (id as usize).min(self.config.vocab_size.saturating_sub(1));
            let src = &self.embed_tokens[id * hs..(id + 1) * hs];
            hidden[t * hs..(t + 1) * hs].copy_from_slice(src);
        }
        for layer in &self.layers {
            hidden = layer.forward(&hidden, seq_len);
        }
        self.norm.forward(&hidden, seq_len, hs)
    }

    /// Forward pass caching intermediate activations for exact backpropagation.
    ///
    /// Returns `(layer_inputs, final_normed, logits)`:
    /// - `layer_inputs`: inputs to each layer (index 0 is layer 0 input, index N is final layer output)
    /// - `final_normed`: shape `[seq_len, hidden_size]`
    /// - `logits`: shape `[seq_len, vocab_size]`
    pub fn forward_with_cache(&self, token_ids: &[u32]) -> (Vec<Vec<f32>>, Vec<f32>, Vec<f32>) {
        let seq_len = token_ids.len();
        let hs = self.config.hidden_size;
        let vs = self.config.vocab_size;

        let mut x0 = vec![0.0f32; seq_len * hs];
        for (t, &id) in token_ids.iter().enumerate() {
            let id = (id as usize).min(vs.saturating_sub(1));
            let src = &self.embed_tokens[id * hs..(id + 1) * hs];
            x0[t * hs..(t + 1) * hs].copy_from_slice(src);
        }

        let mut layer_inputs = Vec::with_capacity(self.layers.len() + 1);
        layer_inputs.push(x0);

        for (l, layer) in self.layers.iter().enumerate() {
            let next_x = layer.forward(&layer_inputs[l], seq_len);
            layer_inputs.push(next_x);
        }

        let final_normed = self.norm.forward(layer_inputs.last().unwrap(), seq_len, hs);

        // Compute logits: [seq_len, vocab_size]
        let mut logits = vec![0.0f32; seq_len * vs];
        for t in 0..seq_len {
            let h_row = &final_normed[t * hs..(t + 1) * hs];
            for v in 0..vs {
                let lm_row = &self.lm_head[v * hs..(v + 1) * hs];
                let dot: f32 = h_row.iter().zip(lm_row.iter()).map(|(&a, &b)| a * b).sum();
                logits[t * vs + v] = dot;
            }
        }

        (layer_inputs, final_normed, logits)
    }

    /// Full backward pass through all transformer layers, embedding table, and LM head.
    ///
    /// Takes `d_logits`: shape `[seq_len, vocab_size]`.
    /// Returns `ModelGradients` containing gradients for every trainable parameter.
    pub fn backward(
        &self,
        token_ids: &[u32],
        layer_inputs: &[Vec<f32>],
        final_normed: &[f32],
        d_logits: &[f32],
    ) -> ModelGradients {
        let seq_len = token_ids.len();
        let hs = self.config.hidden_size;
        let vs = self.config.vocab_size;

        // 1. LM Head backward:
        // d_lm_head: [vocab_size, hidden_size]
        let mut d_lm_head = vec![0.0f32; vs * hs];
        for v in 0..vs {
            for h in 0..hs {
                let mut sum = 0.0f32;
                for t in 0..seq_len {
                    sum += d_logits[t * vs + v] * final_normed[t * hs + h];
                }
                d_lm_head[v * hs + h] = sum;
            }
        }

        // d_final_normed: [seq_len, hidden_size]
        let mut d_final_normed = vec![0.0f32; seq_len * hs];
        for t in 0..seq_len {
            for h in 0..hs {
                let mut sum = 0.0f32;
                for v in 0..vs {
                    sum += d_logits[t * vs + v] * self.lm_head[v * hs + h];
                }
                d_final_normed[t * hs + h] = sum;
            }
        }

        // 2. Final RMSNorm backward:
        let (mut dx, d_norm) =
            self.norm
                .backward(&d_final_normed, layer_inputs.last().unwrap(), seq_len, hs);

        // 3. Decoder layers backward in reverse order:
        let mut layer_gradients = Vec::with_capacity(self.layers.len());
        for l in (0..self.layers.len()).rev() {
            let l_grads = self.layers[l].backward(&dx, &layer_inputs[l], seq_len);
            dx = l_grads.dx.clone();
            layer_gradients.push(l_grads);
        }
        layer_gradients.reverse(); // Restore normal 0..N order

        // 4. Token embeddings backward:
        // Accumulate dx (gradient entering layer 0) into embedding table rows
        let mut d_embed_tokens = vec![0.0f32; vs * hs];
        for (t, &id) in token_ids.iter().enumerate() {
            let id = (id as usize).min(vs.saturating_sub(1));
            let dx_row = &dx[t * hs..(t + 1) * hs];
            for h in 0..hs {
                d_embed_tokens[id * hs + h] += dx_row[h];
            }
        }

        ModelGradients {
            embed_tokens: d_embed_tokens,
            layers: layer_gradients,
            norm: d_norm,
            lm_head: d_lm_head,
        }
    }

    /// Compute the SHA-256 of the model weights file.
    ///
    /// # Errors
    /// Returns [`ModelError::Weights`] if the file cannot be read.
    pub fn get_model_sha256(model_dir: &str) -> Result<String, ModelError> {
        let path = format!("{}/model.safetensors", model_dir);
        Ok(compute_sha256(&path)?)
    }

    /// Count total parameters in this model.
    pub fn param_count(&self) -> usize {
        let hs = self.config.hidden_size;
        let vs = self.config.vocab_size;
        let inter = self.config.intermediate_size;
        let nh = self.config.num_attention_heads;
        let nkv = self.config.num_key_value_heads;
        let hd = self.config.effective_head_dim();
        let nl = self.config.num_hidden_layers;

        let embed = vs * hs;
        let per_layer = (nh * hd * hs)   // q_proj
            + (nkv * hd * hs)            // k_proj
            + (nkv * hd * hs)            // v_proj
            + (hs * nh * hd)             // o_proj
            + hs                         // input_layernorm
            + (inter * hs)               // gate_proj
            + (inter * hs)               // up_proj
            + (hs * inter)               // down_proj
            + hs; // post_attention_layernorm
        let final_norm = hs;
        let lm_head = vs * hs;

        embed + nl * per_layer + final_norm + lm_head
    }
}

fn validate_weight_len(
    weights: &HashMap<String, Vec<f32>>,
    name: &str,
    expected: usize,
) -> Result<(), ModelError> {
    let values = weights
        .get(name)
        .ok_or_else(|| ModelError::MissingWeight(name.to_string()))?;
    if values.len() != expected {
        return Err(ModelError::InvalidWeightShape {
            name: name.to_string(),
            actual: values.len(),
            expected,
        });
    }
    Ok(())
}

/// Gradients computed across all layers and projections in TaraForCausalLM.
#[derive(Debug, Clone)]
pub struct ModelGradients {
    pub embed_tokens: Vec<f32>,
    pub layers: Vec<super::decoder_layer::LayerGradients>,
    pub norm: Vec<f32>,
    pub lm_head: Vec<f32>,
}

impl ModelGradients {
    /// Flatten all gradients into a map of canonical tensor_name -> Vec<f32>.
    pub fn to_map(&self) -> HashMap<String, Vec<f32>> {
        let mut map = HashMap::new();
        map.insert(
            "model.embed_tokens.weight".to_string(),
            self.embed_tokens.clone(),
        );
        map.insert("model.norm.weight".to_string(), self.norm.clone());
        map.insert("lm_head.weight".to_string(), self.lm_head.clone());

        for (i, layer) in self.layers.iter().enumerate() {
            let prefix = format!("model.layers.{}", i);
            map.insert(
                format!("{}.input_layernorm.weight", prefix),
                layer.d_input_layernorm.clone(),
            );
            map.insert(
                format!("{}.self_attn.q_proj.weight", prefix),
                layer.d_q_proj.clone(),
            );
            map.insert(
                format!("{}.self_attn.k_proj.weight", prefix),
                layer.d_k_proj.clone(),
            );
            map.insert(
                format!("{}.self_attn.v_proj.weight", prefix),
                layer.d_v_proj.clone(),
            );
            map.insert(
                format!("{}.self_attn.o_proj.weight", prefix),
                layer.d_o_proj.clone(),
            );
            map.insert(
                format!("{}.post_attention_layernorm.weight", prefix),
                layer.d_post_attention_layernorm.clone(),
            );
            map.insert(
                format!("{}.mlp.gate_proj.weight", prefix),
                layer.d_gate_proj.clone(),
            );
            map.insert(
                format!("{}.mlp.up_proj.weight", prefix),
                layer.d_up_proj.clone(),
            );
            map.insert(
                format!("{}.mlp.down_proj.weight", prefix),
                layer.d_down_proj.clone(),
            );
        }

        map
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_model_end_to_end_numerical_gradient_check() {
        let vs = 3;
        let hs = 4;
        let inter = 4;
        let nh = 2;
        let nkv = 1;
        let hd = 2;

        let config = TaraConfig {
            vocab_size: vs,
            hidden_size: hs,
            intermediate_size: inter,
            num_hidden_layers: 1,
            num_attention_heads: nh,
            num_key_value_heads: nkv,
            head_dim: hd,
            max_position_embeddings: 8,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            initializer_range: 0.02,
            version: "test".into(),
            model_type: "tara".into(),
        };

        let mut weights = HashMap::<String, Vec<f32>>::new();
        weights.insert(
            "model.embed_tokens.weight".into(),
            vec![
                0.2, -0.3, 0.5, 0.1, -0.4, 0.2, 0.1, -0.2, 0.3, 0.1, -0.2, 0.4,
            ],
        );
        weights.insert("model.norm.weight".into(), vec![1.0, 1.1, 0.9, 1.0]);
        weights.insert(
            "lm_head.weight".into(),
            vec![
                0.3, -0.2, 0.4, 0.1, -0.1, 0.5, -0.3, 0.2, 0.2, -0.4, 0.1, 0.3,
            ],
        );

        let q_proj = vec![
            0.2, -0.3, 0.4, -0.1, -0.5, 0.2, 0.1, 0.3, 0.4, -0.2, 0.3, 0.1, -0.1, 0.5, -0.2, 0.4,
        ];
        let k_proj = vec![0.3, -0.4, 0.2, 0.5, -0.2, 0.1, 0.6, -0.3];
        let v_proj = vec![-0.1, 0.5, 0.3, -0.2, 0.4, -0.3, 0.1, 0.2];
        let o_proj = vec![
            0.5, -0.2, 0.3, 0.1, -0.3, 0.4, -0.1, 0.6, 0.2, -0.5, 0.4, -0.2, 0.1, 0.3, -0.2, 0.5,
        ];
        let gate_proj = vec![
            0.2, -0.4, 0.5, -0.1, -0.1, 0.3, -0.2, 0.4, 0.6, -0.1, 0.4, -0.3, -0.5, 0.2, -0.3, 0.5,
        ];
        let up_proj = vec![
            -0.3, 0.5, -0.2, 0.1, 0.4, -0.1, 0.6, -0.2, -0.2, 0.3, -0.5, 0.4, 0.1, -0.4, 0.2, -0.3,
        ];
        let down_proj = vec![
            0.4, -0.2, 0.3, -0.5, -0.1, 0.6, -0.4, 0.2, 0.5, -0.3, 0.1, -0.2, -0.2, 0.4, -0.1, 0.3,
        ];

        weights.insert("model.layers.0.self_attn.q_proj.weight".into(), q_proj);
        weights.insert("model.layers.0.self_attn.k_proj.weight".into(), k_proj);
        weights.insert("model.layers.0.self_attn.v_proj.weight".into(), v_proj);
        weights.insert("model.layers.0.self_attn.o_proj.weight".into(), o_proj);
        weights.insert(
            "model.layers.0.input_layernorm.weight".into(),
            vec![1.0; hs],
        );
        weights.insert(
            "model.layers.0.post_attention_layernorm.weight".into(),
            vec![1.0; hs],
        );
        weights.insert("model.layers.0.mlp.gate_proj.weight".into(), gate_proj);
        weights.insert("model.layers.0.mlp.up_proj.weight".into(), up_proj);
        weights.insert("model.layers.0.mlp.down_proj.weight".into(), down_proj);

        let layer =
            super::super::decoder_layer::TaraDecoderLayer::from_weights(&weights, 0, &config)
                .unwrap();
        let norm = super::super::rms_norm::RmsNorm::new(
            weights["model.norm.weight"].clone(),
            config.rms_norm_eps as f32,
        );

        let model = TaraForCausalLM {
            embed_tokens: weights["model.embed_tokens.weight"].clone(),
            layers: vec![layer],
            norm,
            lm_head: weights["lm_head.weight"].clone(),
            config: config.clone(),
            model_dir: String::new(),
        };

        let token_ids = vec![0u32, 1u32];
        let (layer_inputs, final_normed, _logits) = model.forward_with_cache(&token_ids);

        let d_logits = vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.1];

        let grads = model.backward(&token_ids, &layer_inputs, &final_normed, &d_logits);
        let grad_map = grads.to_map();

        // 1. Verify lm_head gradient via finite differences
        let h = 1e-3f32;
        for i in 0..model.lm_head.len() {
            let mut lm_plus = model.lm_head.clone();
            let mut lm_minus = model.lm_head.clone();
            lm_plus[i] += h;
            lm_minus[i] -= h;
            // Compute logits with lm_plus
            let mut log_plus = vec![0.0f32; 2 * vs];
            let mut log_minus = vec![0.0f32; 2 * vs];
            for t in 0..2 {
                let h_row = &final_normed[t * hs..(t + 1) * hs];
                for v in 0..vs {
                    let r_plus = &lm_plus[v * hs..(v + 1) * hs];
                    let r_minus = &lm_minus[v * hs..(v + 1) * hs];
                    log_plus[t * vs + v] =
                        h_row.iter().zip(r_plus.iter()).map(|(&a, &b)| a * b).sum();
                    log_minus[t * vs + v] =
                        h_row.iter().zip(r_minus.iter()).map(|(&a, &b)| a * b).sum();
                }
            }

            let l_plus: f32 = log_plus
                .iter()
                .zip(d_logits.iter())
                .map(|(&a, &b)| a * b)
                .sum();
            let l_minus: f32 = log_minus
                .iter()
                .zip(d_logits.iter())
                .map(|(&a, &b)| a * b)
                .sum();
            let num_grad = (l_plus - l_minus) / (2.0 * h);

            let diff = (grads.lm_head[i] - num_grad).abs();
            assert!(
                diff < 2e-2,
                "Model lm_head gradient mismatch at index {}: analytical={}, numerical={}",
                i,
                grads.lm_head[i],
                num_grad
            );
        }

        // 2. Verify that all 12 model tensors have non-zero gradients
        for (name, tensor_grad) in &grad_map {
            let norm_sq: f32 = tensor_grad.iter().map(|&g| g * g).sum();
            assert!(
                norm_sq > 0.0,
                "Tensor {} has vanishing (zero) gradient norm",
                name
            );
            assert!(
                tensor_grad.iter().all(|&g| !g.is_nan() && !g.is_infinite()),
                "Tensor {} contains NaN or Inf gradients",
                name
            );
        }
    }
}
