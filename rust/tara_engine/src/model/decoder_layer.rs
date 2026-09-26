//! Transformer decoder layer.
//!
//! Each TaraDecoderLayer applies pre-norm residual attention followed by
//! pre-norm residual MLP (standard "Pre-LN" transformer).

use std::collections::HashMap;
use thiserror::Error;

use crate::config::TaraConfig;
use super::attention::{TaraAttention, AttentionError};
use super::mlp::{TaraMlp, MlpError};
use super::rms_norm::RmsNorm;

/// Errors building a decoder layer.
#[derive(Debug, Error)]
pub enum DecoderLayerError {
    #[error("attention error: {0}")]
    Attention(#[from] AttentionError),
    #[error("mlp error: {0}")]
    Mlp(#[from] MlpError),
    #[error("missing weight: {0}")]
    MissingWeight(String),
}

/// A single transformer decoder layer.
///
/// Computes:
/// ```text
/// h = x + self_attn( input_layernorm(x) )
/// y = h + mlp( post_attention_layernorm(h) )
/// ```
pub struct TaraDecoderLayer {
    pub self_attn: TaraAttention,
    pub mlp: TaraMlp,
    pub input_layernorm: RmsNorm,
    pub post_attention_layernorm: RmsNorm,
    pub hidden_size: usize,
    pub intermediate_size: usize,
}

impl TaraDecoderLayer {
    /// Load a decoder layer from the global weight map for `layer_idx`.
    pub fn from_weights(
        weights: &HashMap<String, Vec<f32>>,
        layer_idx: usize,
        config: &TaraConfig,
    ) -> Result<Self, DecoderLayerError> {
        let prefix = format!("model.layers.{}", layer_idx);
        let eps = config.rms_norm_eps as f32;

        let input_ln_w = weights
            .get(&format!("{}.input_layernorm.weight", prefix))
            .cloned()
            .ok_or_else(|| DecoderLayerError::MissingWeight(
                format!("{}.input_layernorm.weight", prefix),
            ))?;
        let post_ln_w = weights
            .get(&format!("{}.post_attention_layernorm.weight", prefix))
            .cloned()
            .ok_or_else(|| DecoderLayerError::MissingWeight(
                format!("{}.post_attention_layernorm.weight", prefix),
            ))?;

        let self_attn = TaraAttention::from_weights(weights, layer_idx, config)?;
        let mlp = TaraMlp::from_weights(weights, layer_idx)?;
        let input_layernorm = RmsNorm::new(input_ln_w, eps);
        let post_attention_layernorm = RmsNorm::new(post_ln_w, eps);

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
        })
    }

    /// Run the decoder layer forward pass.
    ///
    /// `x` is a flat `[seq_len, hidden_size]` buffer.
    /// Returns a buffer of the same shape.
    pub fn forward(&self, x: &[f32], seq_len: usize) -> Vec<f32> {
        let hs = self.hidden_size;

        let normed_x = self.input_layernorm.forward(x, seq_len, hs);
        let attn_out = self.self_attn.forward(&normed_x, seq_len);
        let h: Vec<f32> = x.iter().zip(attn_out.iter()).map(|(&a, &b)| a + b).collect();

        let normed_h = self.post_attention_layernorm.forward(&h, seq_len, hs);
        let mlp_out = self.mlp.forward(&normed_h, seq_len, hs, self.intermediate_size);

        h.iter().zip(mlp_out.iter()).map(|(&a, &b)| a + b).collect()
    }
}
