//! Transformer decoder layer.
//!
//! Each TaraDecoderLayer applies pre-norm residual attention followed by
//! pre-norm residual MLP (standard "Pre-LN" transformer).

use std::collections::HashMap;
use thiserror::Error;

use super::attention::{AttentionError, TaraAttention};
use super::mlp::{MlpError, TaraMlp};
use super::rms_norm::RmsNorm;
use crate::config::TaraConfig;

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
            .ok_or_else(|| {
                DecoderLayerError::MissingWeight(format!("{}.input_layernorm.weight", prefix))
            })?;
        let post_ln_w = weights
            .get(&format!("{}.post_attention_layernorm.weight", prefix))
            .cloned()
            .ok_or_else(|| {
                DecoderLayerError::MissingWeight(format!(
                    "{}.post_attention_layernorm.weight",
                    prefix
                ))
            })?;

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
        let h: Vec<f32> = x
            .iter()
            .zip(attn_out.iter())
            .map(|(&a, &b)| a + b)
            .collect();

        let normed_h = self.post_attention_layernorm.forward(&h, seq_len, hs);
        let mlp_out = self
            .mlp
            .forward(&normed_h, seq_len, hs, self.intermediate_size);

        h.iter().zip(mlp_out.iter()).map(|(&a, &b)| a + b).collect()
    }

    /// Single-token forward pass through the decoder layer using KV caching.
    ///
    /// `x`: `[1, hidden_size]` buffer.
    pub fn forward_cached(
        &self,
        x: &[f32],
        cache: &mut super::attention::LayerKvCache,
    ) -> Vec<f32> {
        let hs = self.hidden_size;

        let normed_x = self.input_layernorm.forward(x, 1, hs);
        let attn_out = self.self_attn.forward_cached(&normed_x, cache);
        let h: Vec<f32> = x
            .iter()
            .zip(attn_out.iter())
            .map(|(&a, &b)| a + b)
            .collect();

        let normed_h = self.post_attention_layernorm.forward(&h, 1, hs);
        let mlp_out = self.mlp.forward(&normed_h, 1, hs, self.intermediate_size);

        h.iter().zip(mlp_out.iter()).map(|(&a, &b)| a + b).collect()
    }

    /// Run full backward pass for this decoder layer.
    pub fn backward(&self, dy: &[f32], x: &[f32], seq_len: usize) -> LayerGradients {
        let hs = self.hidden_size;

        // Recompute forward activations needed for backprop
        let normed_x = self.input_layernorm.forward(x, seq_len, hs);
        let attn_out = self.self_attn.forward(&normed_x, seq_len);
        let h: Vec<f32> = x
            .iter()
            .zip(attn_out.iter())
            .map(|(&a, &b)| a + b)
            .collect();

        let normed_h = self.post_attention_layernorm.forward(&h, seq_len, hs);

        // 1. Backward through MLP
        let (d_normed_h, d_gate_proj, d_up_proj, d_down_proj) =
            self.mlp
                .backward(dy, &normed_h, seq_len, hs, self.intermediate_size);

        // 2. Backward through Post-Attention LayerNorm
        let (dh_from_mlp, d_post_attention_layernorm) =
            self.post_attention_layernorm
                .backward(&d_normed_h, &h, seq_len, hs);

        // 3. Accumulate gradient into h (Residual 2 branch)
        let mut dh = vec![0.0f32; seq_len * hs];
        for i in 0..seq_len * hs {
            dh[i] = dh_from_mlp[i] + dy[i];
        }

        // 4. Backward through Attention
        let (d_normed_x, d_q_proj, d_k_proj, d_v_proj, d_o_proj) =
            self.self_attn.backward(&dh, &normed_x, seq_len);

        // 5. Backward through Input LayerNorm
        let (dx_from_attn, d_input_layernorm) =
            self.input_layernorm.backward(&d_normed_x, x, seq_len, hs);

        // 6. Accumulate gradient into x (Residual 1 branch)
        let mut dx = vec![0.0f32; seq_len * hs];
        for i in 0..seq_len * hs {
            dx[i] = dx_from_attn[i] + dh[i];
        }

        LayerGradients {
            dx,
            d_input_layernorm,
            d_q_proj,
            d_k_proj,
            d_v_proj,
            d_o_proj,
            d_post_attention_layernorm,
            d_gate_proj,
            d_up_proj,
            d_down_proj,
        }
    }
}

/// Gradients computed for a single decoder layer.
#[derive(Debug, Clone)]
pub struct LayerGradients {
    pub dx: Vec<f32>,
    pub d_input_layernorm: Vec<f32>,
    pub d_q_proj: Vec<f32>,
    pub d_k_proj: Vec<f32>,
    pub d_v_proj: Vec<f32>,
    pub d_o_proj: Vec<f32>,
    pub d_post_attention_layernorm: Vec<f32>,
    pub d_gate_proj: Vec<f32>,
    pub d_up_proj: Vec<f32>,
    pub d_down_proj: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::rope::RotaryEmbedding;

    #[test]
    fn test_decoder_layer_numerical_gradient_checks() {
        let hs = 4;
        let inter = 4;
        let nh = 2;
        let nkv = 1;
        let hd = 2;
        let seq_len = 2;

        let q_proj = vec![
            0.2, -0.3, 0.4, -0.1, -0.5, 0.2, 0.1, 0.3, 0.4, -0.2, 0.3, 0.1, -0.1, 0.5, -0.2, 0.4,
        ];
        let k_proj = vec![0.3, -0.4, 0.2, 0.5, -0.2, 0.1, 0.6, -0.3];
        let v_proj = vec![-0.1, 0.5, 0.3, -0.2, 0.4, -0.3, 0.1, 0.2];
        let o_proj = vec![
            0.5, -0.2, 0.3, 0.1, -0.3, 0.4, -0.1, 0.6, 0.2, -0.5, 0.4, -0.2, 0.1, 0.3, -0.2, 0.5,
        ];

        let rotary_emb = RotaryEmbedding::new(hd, 16, 10000.0);
        let self_attn = TaraAttention {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: nh,
            num_kv_heads: nkv,
            head_dim: hd,
            hidden_size: hs,
            rotary_emb,
        };

        let gate_proj = vec![
            0.2, -0.4, 0.5, -0.1, -0.1, 0.3, -0.2, 0.4, 0.6, -0.1, 0.4, -0.3, -0.5, 0.2, -0.3, 0.5,
        ];
        let up_proj = vec![
            -0.3, 0.5, -0.2, 0.1, 0.4, -0.1, 0.6, -0.2, -0.2, 0.3, -0.5, 0.4, 0.1, -0.4, 0.2, -0.3,
        ];
        let down_proj = vec![
            0.4, -0.2, 0.3, -0.5, -0.1, 0.6, -0.4, 0.2, 0.5, -0.3, 0.1, -0.2, -0.2, 0.4, -0.1, 0.3,
        ];
        let mlp = TaraMlp::new(gate_proj, up_proj, down_proj);

        let input_layernorm = RmsNorm::new(vec![1.1, 0.9, 1.0, 1.2], 1e-5);
        let post_attention_layernorm = RmsNorm::new(vec![0.9, 1.2, 1.1, 0.8], 1e-5);

        let layer = TaraDecoderLayer {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            hidden_size: hs,
            intermediate_size: inter,
        };

        let x = vec![0.5, -0.3, 0.8, -0.2, 1.1, 0.4, -0.9, 0.7];
        let dy = vec![0.2, -0.1, 0.4, 0.3, -0.3, 0.5, 0.2, -0.1];

        let grads = layer.backward(&dy, &x, seq_len);

        // Verify dx via finite differences
        let h = 1e-3f32;
        for i in 0..x.len() {
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();
            x_plus[i] += h;
            x_minus[i] -= h;

            let y_plus = layer.forward(&x_plus, seq_len);
            let y_minus = layer.forward(&x_minus, seq_len);

            let l_plus: f32 = y_plus.iter().zip(dy.iter()).map(|(&a, &b)| a * b).sum();
            let l_minus: f32 = y_minus.iter().zip(dy.iter()).map(|(&a, &b)| a * b).sum();
            let num_grad = (l_plus - l_minus) / (2.0 * h);

            let diff = (grads.dx[i] - num_grad).abs();
            assert!(
                diff < 3e-2,
                "DecoderLayer dx gradient mismatch at index {}: analytical={}, numerical={}",
                i,
                grads.dx[i],
                num_grad
            );
        }
    }
}
