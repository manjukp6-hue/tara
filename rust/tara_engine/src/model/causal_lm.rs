//! Top-level causal language model — TaraForCausalLM.
//!
//! Loads weights from a SafeTensors file (or shard set), constructs the full
//! transformer stack, and runs autoregressive forward passes.

use std::collections::HashMap;
use thiserror::Error;

use crate::config::TaraConfig;
use crate::safetensors::{load_model_weights, compute_sha256, SafeTensorsError};
use super::rms_norm::RmsNorm;
use super::decoder_layer::TaraDecoderLayer;

/// Expected SHA-256 of the production model file.
pub const EXPECTED_MODEL_SHA256: &str =
    "7a50308b799f2654baeafbd64dec31f088c3b07b446e198cd0f9ec2b7c0af309";

/// Errors that can occur while loading or running TaraForCausalLM.
#[derive(Debug, Error)]
pub enum ModelError {
    #[error("config error: {0}")]
    Config(#[from] crate::config::ConfigError),
    #[error("weights error: {0}")]
    Weights(#[from] SafeTensorsError),
    #[error("missing weight: {0}")]
    MissingWeight(String),
    #[error("attention error: {0}")]
    Attention(#[from] super::attention::AttentionError),
    #[error("mlp error: {0}")]
    Mlp(#[from] super::mlp::MlpError),
    #[error("decoder layer error: {0}")]
    DecoderLayer(#[from] super::decoder_layer::DecoderLayerError),
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

        // Embedding lookup — shape [seq_len, hidden_size]
        let mut hidden = vec![0.0f32; seq_len * hs];
        for (t, &id) in token_ids.iter().enumerate() {
            let id = id as usize;
            let src = &self.embed_tokens[id * hs..(id + 1) * hs];
            hidden[t * hs..(t + 1) * hs].copy_from_slice(src);
        }

        // Decoder layers
        for layer in &self.layers {
            hidden = layer.forward(&hidden, seq_len);
        }

        // Final norm
        hidden = self.norm.forward(&hidden, seq_len, hs);

        // LM head — [seq_len, vocab_size]
        // lm_head shape [vs, hs]; output = hidden @ lm_head.T
        let mut logits = vec![0.0f32; seq_len * vs];
        for t in 0..seq_len {
            let h_row = &hidden[t * hs..(t + 1) * hs];
            let l_row = &mut logits[t * vs..(t + 1) * vs];
            for v in 0..vs {
                let lm_row = &self.lm_head[v * hs..(v + 1) * hs];
                l_row[v] = h_row.iter().zip(lm_row.iter()).map(|(&a, &b)| a * b).sum();
            }
        }

        logits
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
            + hs;                        // post_attention_layernorm
        let final_norm = hs;
        let lm_head = vs * hs;

        embed + nl * per_layer + final_norm + lm_head
    }
}
