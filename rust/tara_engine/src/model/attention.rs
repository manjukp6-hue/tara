//! Grouped Query Attention layer (GQA).
//!
//! Implements multi-head attention with optional key-value head grouping,
//! rotary position embeddings, and a causal (autoregressive) mask.

use std::collections::HashMap;
use thiserror::Error;

use crate::config::TaraConfig;
use super::mlp::mat_mul;
use super::rope::RotaryEmbedding;

/// Errors that can occur while constructing an attention layer.
#[derive(Debug, Error)]
pub enum AttentionError {
    /// A required weight tensor was not found.
    #[error("missing weight tensor: {0}")]
    MissingWeight(String),
}

/// Grouped Query Attention layer.
///
/// Supports full multi-head attention (`num_kv_heads == num_heads`) and
/// grouped query attention (`num_kv_heads < num_heads`).
pub struct TaraAttention {
    /// Query projection weight: shape `[num_heads * head_dim, hidden_size]`.
    pub q_proj: Vec<f32>,
    /// Key projection weight: shape `[num_kv_heads * head_dim, hidden_size]`.
    pub k_proj: Vec<f32>,
    /// Value projection weight: shape `[num_kv_heads * head_dim, hidden_size]`.
    pub v_proj: Vec<f32>,
    /// Output projection weight: shape `[hidden_size, num_heads * head_dim]`.
    pub o_proj: Vec<f32>,

    /// Number of query attention heads.
    pub num_heads: usize,
    /// Number of key/value attention heads.
    pub num_kv_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Total hidden state dimension.
    pub hidden_size: usize,

    /// Rotary embedding helper (dim = head_dim).
    pub rotary_emb: RotaryEmbedding,
}

impl TaraAttention {
    /// Build a [`TaraAttention`] from the global weight map for `layer_idx`.
    ///
    /// Looks for keys like `model.layers.{layer_idx}.self_attn.q_proj.weight`.
    ///
    /// # Errors
    /// Returns [`AttentionError::MissingWeight`] if any required tensor is absent.
    pub fn from_weights(
        weights: &HashMap<String, Vec<f32>>,
        layer_idx: usize,
        config: &TaraConfig,
    ) -> Result<Self, AttentionError> {
        let prefix = format!("model.layers.{}.self_attn", layer_idx);

        let q_proj = weights
            .get(&format!("{}.q_proj.weight", prefix))
            .cloned()
            .ok_or_else(|| AttentionError::MissingWeight(format!("{}.q_proj.weight", prefix)))?;
        let k_proj = weights
            .get(&format!("{}.k_proj.weight", prefix))
            .cloned()
            .ok_or_else(|| AttentionError::MissingWeight(format!("{}.k_proj.weight", prefix)))?;
        let v_proj = weights
            .get(&format!("{}.v_proj.weight", prefix))
            .cloned()
            .ok_or_else(|| AttentionError::MissingWeight(format!("{}.v_proj.weight", prefix)))?;
        let o_proj = weights
            .get(&format!("{}.o_proj.weight", prefix))
            .cloned()
            .ok_or_else(|| AttentionError::MissingWeight(format!("{}.o_proj.weight", prefix)))?;

        let rotary_emb = RotaryEmbedding::new(
            config.effective_head_dim(),
            config.max_position_embeddings,
            config.rope_theta as f32,
        );

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.effective_head_dim(),
            hidden_size: config.hidden_size,
            rotary_emb,
        })
    }

    /// Run the attention forward pass.
    ///
    /// `hidden_states`: flat `[seq_len, hidden_size]` buffer.
    /// Returns a flat `[seq_len, hidden_size]` output.
    pub fn forward(&self, hidden_states: &[f32], seq_len: usize) -> Vec<f32> {
        let nh  = self.num_heads;
        let nkv = self.num_kv_heads;
        let hd  = self.head_dim;
        let hs  = self.hidden_size;

        // ── Projections ──────────────────────────────────────────
        // q: [seq, nh*hd]
        let q = mat_mul(hidden_states, &self.q_proj, seq_len, hs, nh * hd);
        // k: [seq, nkv*hd]
        let k = mat_mul(hidden_states, &self.k_proj, seq_len, hs, nkv * hd);
        // v: [seq, nkv*hd]
        let v = mat_mul(hidden_states, &self.v_proj, seq_len, hs, nkv * hd);

        // ── Reshape to [seq, heads, head_dim] ────────────────────
        // (already row-major; we just index carefully)

        // ── Rotary embeddings ─────────────────────────────────────
        let (cos, sin) = self.rotary_emb.get_cos_sin(seq_len);
        let (q_rot, k_rot) = RotaryEmbedding::apply_rotary_emb(
            &q, &k, &cos, &sin, seq_len, nh, nkv, hd,
        );

        // ── GQA: expand K/V by num_kv_groups ─────────────────────
        let kv_groups = nh / nkv;
        // k_expanded: [seq, nh, hd]
        let mut k_exp = vec![0.0f32; seq_len * nh * hd];
        let mut v_exp = vec![0.0f32; seq_len * nh * hd];
        for pos in 0..seq_len {
            for h in 0..nh {
                let kv_head = h / kv_groups;
                let src_k = pos * nkv * hd + kv_head * hd;
                let src_v = pos * nkv * hd + kv_head * hd;
                let dst   = pos * nh  * hd + h * hd;
                k_exp[dst..dst + hd].copy_from_slice(&k_rot[src_k..src_k + hd]);
                v_exp[dst..dst + hd].copy_from_slice(&v[src_v..src_v + hd]);
            }
        }

        // ── Scaled dot-product with causal mask ───────────────────
        let scale = (hd as f32).sqrt().recip();
        // attn_weights: [nh, seq, seq]
        let mut attn_weights = vec![0.0f32; nh * seq_len * seq_len];

        for h in 0..nh {
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let qi = q_rot[i * nh * hd + h * hd..i * nh * hd + h * hd + hd].to_vec();
                    let kj = k_exp[j * nh * hd + h * hd..j * nh * hd + h * hd + hd].to_vec();
                    let dot: f32 = qi.iter().zip(kj.iter()).map(|(&a, &b)| a * b).sum();
                    let masked = if j > i { f32::NEG_INFINITY } else { dot * scale };
                    attn_weights[h * seq_len * seq_len + i * seq_len + j] = masked;
                }
            }
        }

        // Softmax over last dim (per query position, per head)
        softmax_inplace(&mut attn_weights, nh, seq_len);

        // ── Weighted sum over values ──────────────────────────────
        // context: [seq, nh, hd]
        let mut context = vec![0.0f32; seq_len * nh * hd];
        for h in 0..nh {
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let w = attn_weights[h * seq_len * seq_len + i * seq_len + j];
                    let vj = &v_exp[j * nh * hd + h * hd..j * nh * hd + h * hd + hd];
                    let out_slice = &mut context[i * nh * hd + h * hd..i * nh * hd + h * hd + hd];
                    for (o, &vi) in out_slice.iter_mut().zip(vj.iter()) {
                        *o += w * vi;
                    }
                }
            }
        }

        // ── Output projection ─────────────────────────────────────
        // context is [seq, nh*hd]; o_proj is [hs, nh*hd]
        mat_mul(&context, &self.o_proj, seq_len, nh * hd, hs)
    }
}

/// In-place causal softmax over `[nh, seq, seq]` attention weights.
fn softmax_inplace(w: &mut [f32], nh: usize, seq_len: usize) {
    for h in 0..nh {
        for i in 0..seq_len {
            let row_start = h * seq_len * seq_len + i * seq_len;
            let row = &mut w[row_start..row_start + seq_len];

            // Numerically stable softmax
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for v in row.iter_mut() {
                *v = (*v - max).exp();
                sum += *v;
            }
            if sum > 0.0 {
                for v in row.iter_mut() {
                    *v /= sum;
                }
            }
        }
    }
}
