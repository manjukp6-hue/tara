//! Grouped Query Attention layer (GQA).
//!
//! Implements multi-head attention with optional key-value head grouping,
//! rotary position embeddings, and a causal (autoregressive) mask.

use std::collections::HashMap;
use thiserror::Error;

use super::mlp::mat_mul;
use super::rope::{RotaryDimensions, RotaryEmbedding};
use crate::config::TaraConfig;

/// Errors that can occur while constructing an attention layer.
#[derive(Debug, Error)]
pub enum AttentionError {
    /// A required weight tensor was not found.
    #[error("missing weight tensor: {0}")]
    MissingWeight(String),
}

/// KV cache for a single attention layer.
#[derive(Debug, Clone, Default)]
pub struct LayerKvCache {
    /// Cached keys: [cached_tokens, num_kv_heads * head_dim]
    pub k: Vec<f32>,
    /// Cached values: [cached_tokens, num_kv_heads * head_dim]
    pub v: Vec<f32>,
    /// Current number of cached tokens.
    pub seq_len: usize,
}

impl LayerKvCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        self.k.clear();
        self.v.clear();
        self.seq_len = 0;
    }
}

/// Gradients produced by the attention backward pass: `(dx, d_q_proj, d_k_proj, d_v_proj, d_o_proj)`.
pub type AttentionGrads = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

/// Grouped Query Attention layer.
///
/// Supports full multi-head attention (`num_kv_heads == num_heads`) and
/// grouped query attention (`num_kv_heads < num_heads`).
#[derive(Clone)]
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
        let nh = self.num_heads;
        let nkv = self.num_kv_heads;
        let hd = self.head_dim;
        let hs = self.hidden_size;

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
        let dims = RotaryDimensions {
            seq_len,
            num_heads_q: nh,
            num_heads_k: nkv,
            head_dim: hd,
        };
        let (q_rot, k_rot) = RotaryEmbedding::apply_rotary_emb(&q, &k, &cos, &sin, dims);

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
                let dst = pos * nh * hd + h * hd;
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
                    let masked = if j > i {
                        f32::NEG_INFINITY
                    } else {
                        dot * scale
                    };
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

    /// Single-token forward pass using cached keys and values.
    ///
    /// `hidden_state`: flat `[1, hidden_size]` buffer for current token.
    /// Updates `cache` in place with the newly computed key/value pair.
    pub fn forward_cached(&self, hidden_state: &[f32], cache: &mut LayerKvCache) -> Vec<f32> {
        let nh = self.num_heads;
        let nkv = self.num_kv_heads;
        let hd = self.head_dim;
        let hs = self.hidden_size;
        let pos = cache.seq_len;

        // 1. Projections for current token
        let q = mat_mul(hidden_state, &self.q_proj, 1, hs, nh * hd);
        let k = mat_mul(hidden_state, &self.k_proj, 1, hs, nkv * hd);
        let v = mat_mul(hidden_state, &self.v_proj, 1, hs, nkv * hd);

        // 2. Rotary embedding at position `pos`
        let (cos, sin) = self.rotary_emb.get_cos_sin_at(pos);
        let (q_rot, k_rot) =
            RotaryEmbedding::apply_rotary_emb_single(&q, &k, &cos, &sin, nh, nkv, hd);

        // 3. Append to cache
        cache.k.extend_from_slice(&k_rot);
        cache.v.extend_from_slice(&v);
        cache.seq_len += 1;
        let total_seq = cache.seq_len;

        // 4. Scaled dot-product attention against all cached tokens
        let kv_groups = nh / nkv;
        let scale = (hd as f32).sqrt().recip();
        let mut attn_weights = vec![0.0f32; nh * total_seq];

        for h in 0..nh {
            let kv_h = h / kv_groups;
            let qi = &q_rot[h * hd..(h + 1) * hd];

            for j in 0..total_seq {
                let kj_start = j * (nkv * hd) + kv_h * hd;
                let kj = &cache.k[kj_start..kj_start + hd];
                let dot: f32 = qi.iter().zip(kj.iter()).map(|(&a, &b)| a * b).sum();
                attn_weights[h * total_seq + j] = dot * scale;
            }

            // Softmax over cached tokens
            let row = &mut attn_weights[h * total_seq..(h + 1) * total_seq];
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for val in row.iter_mut() {
                *val = (*val - max).exp();
                sum += *val;
            }
            if sum > 0.0 {
                for val in row.iter_mut() {
                    *val /= sum;
                }
            }
        }

        // 5. Weighted sum over values
        let mut context = vec![0.0f32; nh * hd];
        for h in 0..nh {
            let kv_h = h / kv_groups;
            let out_slice = &mut context[h * hd..(h + 1) * hd];

            for j in 0..total_seq {
                let w = attn_weights[h * total_seq + j];
                let vj_start = j * (nkv * hd) + kv_h * hd;
                let vj = &cache.v[vj_start..vj_start + hd];
                for (o, &vi) in out_slice.iter_mut().zip(vj.iter()) {
                    *o += w * vi;
                }
            }
        }

        // 6. Output projection: context [1, nh*hd] -> [1, hs]
        mat_mul(&context, &self.o_proj, 1, nh * hd, hs)
    }

    /// Compute analytical gradients for inputs and attention projection weights:
    /// `(dx, d_q_proj, d_k_proj, d_v_proj, d_o_proj)`.
    pub fn backward(&self, dy: &[f32], hidden_states: &[f32], seq_len: usize) -> AttentionGrads {
        let nh = self.num_heads;
        let nkv = self.num_kv_heads;
        let hd = self.head_dim;
        let hs = self.hidden_size;
        let kv_groups = nh / nkv;

        // Forward activations
        let q = mat_mul(hidden_states, &self.q_proj, seq_len, hs, nh * hd);
        let k = mat_mul(hidden_states, &self.k_proj, seq_len, hs, nkv * hd);
        let v = mat_mul(hidden_states, &self.v_proj, seq_len, hs, nkv * hd);

        let (cos, sin) = self.rotary_emb.get_cos_sin(seq_len);
        let dims = RotaryDimensions {
            seq_len,
            num_heads_q: nh,
            num_heads_k: nkv,
            head_dim: hd,
        };
        let (q_rot, k_rot) = RotaryEmbedding::apply_rotary_emb(&q, &k, &cos, &sin, dims);

        let mut k_exp = vec![0.0f32; seq_len * nh * hd];
        let mut v_exp = vec![0.0f32; seq_len * nh * hd];
        for pos in 0..seq_len {
            for h in 0..nh {
                let kv_head = h / kv_groups;
                let src_k = pos * nkv * hd + kv_head * hd;
                let src_v = pos * nkv * hd + kv_head * hd;
                let dst = pos * nh * hd + h * hd;
                k_exp[dst..dst + hd].copy_from_slice(&k_rot[src_k..src_k + hd]);
                v_exp[dst..dst + hd].copy_from_slice(&v[src_v..src_v + hd]);
            }
        }

        let scale = (hd as f32).sqrt().recip();
        let mut attn_weights = vec![0.0f32; nh * seq_len * seq_len];
        for h in 0..nh {
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let qi = &q_rot[i * nh * hd + h * hd..i * nh * hd + h * hd + hd];
                    let kj = &k_exp[j * nh * hd + h * hd..j * nh * hd + h * hd + hd];
                    let dot: f32 = qi.iter().zip(kj.iter()).map(|(&a, &b)| a * b).sum();
                    let masked = if j > i {
                        f32::NEG_INFINITY
                    } else {
                        dot * scale
                    };
                    attn_weights[h * seq_len * seq_len + i * seq_len + j] = masked;
                }
            }
        }
        softmax_inplace(&mut attn_weights, nh, seq_len);

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

        // 1. Output projection backward:
        // dy: [seq_len, hs], context: [seq_len, nh * hd]
        // o_proj: [hs, nh * hd]
        let mut d_o_proj = vec![0.0f32; hs * nh * hd];
        for h_dim in 0..hs {
            for c_dim in 0..nh * hd {
                let mut sum = 0.0f32;
                for t in 0..seq_len {
                    sum += dy[t * hs + h_dim] * context[t * nh * hd + c_dim];
                }
                d_o_proj[h_dim * nh * hd + c_dim] = sum;
            }
        }

        let mut d_context = vec![0.0f32; seq_len * nh * hd];
        for t in 0..seq_len {
            for c_dim in 0..nh * hd {
                let mut sum = 0.0f32;
                for h_dim in 0..hs {
                    sum += dy[t * hs + h_dim] * self.o_proj[h_dim * nh * hd + c_dim];
                }
                d_context[t * nh * hd + c_dim] = sum;
            }
        }

        // 2. Attention weights and Value gradients:
        let mut d_v_exp = vec![0.0f32; seq_len * nh * hd];
        let mut d_attn_weights = vec![0.0f32; nh * seq_len * seq_len];

        for h in 0..nh {
            for i in 0..seq_len {
                let d_c = &d_context[i * nh * hd + h * hd..i * nh * hd + h * hd + hd];
                for j in 0..seq_len {
                    let w = attn_weights[h * seq_len * seq_len + i * seq_len + j];
                    let vj = &v_exp[j * nh * hd + h * hd..j * nh * hd + h * hd + hd];

                    let mut d_w = 0.0f32;
                    for d in 0..hd {
                        d_w += d_c[d] * vj[d];
                        d_v_exp[j * nh * hd + h * hd + d] += d_c[d] * w;
                    }
                    d_attn_weights[h * seq_len * seq_len + i * seq_len + j] = d_w;
                }
            }
        }

        // 3. Softmax backward:
        let mut d_scores = vec![0.0f32; nh * seq_len * seq_len];
        for h in 0..nh {
            for i in 0..seq_len {
                let w_row = &attn_weights[h * seq_len * seq_len + i * seq_len
                    ..(h * seq_len * seq_len + i * seq_len + seq_len)];
                let dw_row = &d_attn_weights[h * seq_len * seq_len + i * seq_len
                    ..(h * seq_len * seq_len + i * seq_len + seq_len)];

                let sum_w_dw: f32 = w_row
                    .iter()
                    .zip(dw_row.iter())
                    .map(|(&w, &dw)| w * dw)
                    .sum();
                for j in 0..seq_len {
                    if j <= i {
                        let w = w_row[j];
                        let dw = dw_row[j];
                        d_scores[h * seq_len * seq_len + i * seq_len + j] =
                            w * (dw - sum_w_dw) * scale;
                    }
                }
            }
        }

        // 4. Dot product backward (Q_rot and K_exp gradients):
        let mut d_q_rot = vec![0.0f32; seq_len * nh * hd];
        let mut d_k_exp = vec![0.0f32; seq_len * nh * hd];
        for h in 0..nh {
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let d_score = d_scores[h * seq_len * seq_len + i * seq_len + j];
                    if d_score != 0.0 {
                        let qi = &q_rot[i * nh * hd + h * hd..i * nh * hd + h * hd + hd];
                        let kj = &k_exp[j * nh * hd + h * hd..j * nh * hd + h * hd + hd];
                        for d in 0..hd {
                            d_q_rot[i * nh * hd + h * hd + d] += d_score * kj[d];
                            d_k_exp[j * nh * hd + h * hd + d] += d_score * qi[d];
                        }
                    }
                }
            }
        }

        // 5. GQA un-expansion for K and V:
        let mut d_k_rot = vec![0.0f32; seq_len * nkv * hd];
        let mut d_v = vec![0.0f32; seq_len * nkv * hd];
        for pos in 0..seq_len {
            for h in 0..nh {
                let kv_head = h / kv_groups;
                let src = pos * nh * hd + h * hd;
                let dst = pos * nkv * hd + kv_head * hd;
                for d in 0..hd {
                    d_k_rot[dst + d] += d_k_exp[src + d];
                    d_v[dst + d] += d_v_exp[src + d];
                }
            }
        }

        // 6. RoPE backward:
        let dims = RotaryDimensions {
            seq_len,
            num_heads_q: nh,
            num_heads_k: nkv,
            head_dim: hd,
        };
        let (d_q, d_k) =
            RotaryEmbedding::apply_rotary_emb_backward(&d_q_rot, &d_k_rot, &cos, &sin, dims);

        // 7. Input projections backward (q_proj, k_proj, v_proj):
        let mut d_q_proj = vec![0.0f32; (nh * hd) * hs];
        for q_dim in 0..nh * hd {
            for h_dim in 0..hs {
                let mut sum = 0.0f32;
                for t in 0..seq_len {
                    sum += d_q[t * nh * hd + q_dim] * hidden_states[t * hs + h_dim];
                }
                d_q_proj[q_dim * hs + h_dim] = sum;
            }
        }

        let mut d_k_proj = vec![0.0f32; (nkv * hd) * hs];
        for k_dim in 0..nkv * hd {
            for h_dim in 0..hs {
                let mut sum = 0.0f32;
                for t in 0..seq_len {
                    sum += d_k[t * nkv * hd + k_dim] * hidden_states[t * hs + h_dim];
                }
                d_k_proj[k_dim * hs + h_dim] = sum;
            }
        }

        let mut d_v_proj = vec![0.0f32; (nkv * hd) * hs];
        for v_dim in 0..nkv * hd {
            for h_dim in 0..hs {
                let mut sum = 0.0f32;
                for t in 0..seq_len {
                    sum += d_v[t * nkv * hd + v_dim] * hidden_states[t * hs + h_dim];
                }
                d_v_proj[v_dim * hs + h_dim] = sum;
            }
        }

        // dx = d_q @ q_proj + d_k @ k_proj + d_v @ v_proj:
        let mut dx = vec![0.0f32; seq_len * hs];
        for t in 0..seq_len {
            for h_dim in 0..hs {
                let mut sum = 0.0f32;
                for q_dim in 0..nh * hd {
                    sum += d_q[t * nh * hd + q_dim] * self.q_proj[q_dim * hs + h_dim];
                }
                for k_dim in 0..nkv * hd {
                    sum += d_k[t * nkv * hd + k_dim] * self.k_proj[k_dim * hs + h_dim];
                }
                for v_dim in 0..nkv * hd {
                    sum += d_v[t * nkv * hd + v_dim] * self.v_proj[v_dim * hs + h_dim];
                }
                dx[t * hs + h_dim] = sum;
            }
        }

        (dx, d_q_proj, d_k_proj, d_v_proj, d_o_proj)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_numerical_gradient_checks() {
        let hs = 4;
        let nh = 2;
        let nkv = 1;
        let hd = 2; // hd * nh = 4 = hs
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
        let attn = TaraAttention {
            q_proj: q_proj.clone(),
            k_proj: k_proj.clone(),
            v_proj: v_proj.clone(),
            o_proj: o_proj.clone(),
            num_heads: nh,
            num_kv_heads: nkv,
            head_dim: hd,
            hidden_size: hs,
            rotary_emb,
        };

        let x = vec![0.4, -0.2, 0.5, 0.1, -0.3, 0.6, -0.1, 0.7];
        let dy = vec![0.1, -0.4, 0.3, -0.2, 0.5, 0.2, -0.1, 0.4];

        let (dx_ana, d_q_ana, d_k_ana, d_v_ana, d_o_ana) = attn.backward(&dy, &x, seq_len);

        let h = 1e-3f32;

        // 1. Verify dx
        for i in 0..x.len() {
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();
            x_plus[i] += h;
            x_minus[i] -= h;

            let y_plus = attn.forward(&x_plus, seq_len);
            let y_minus = attn.forward(&x_minus, seq_len);

            let l_plus: f32 = y_plus.iter().zip(dy.iter()).map(|(&a, &b)| a * b).sum();
            let l_minus: f32 = y_minus.iter().zip(dy.iter()).map(|(&a, &b)| a * b).sum();
            let num_grad = (l_plus - l_minus) / (2.0 * h);

            let diff = (dx_ana[i] - num_grad).abs();
            assert!(
                diff < 2e-2,
                "Attention dx gradient mismatch at index {}: analytical={}, numerical={}",
                i,
                dx_ana[i],
                num_grad
            );
        }

        // 2. Verify d_o_proj
        for i in 0..o_proj.len() {
            let mut o_plus = o_proj.clone();
            let mut o_minus = o_proj.clone();
            o_plus[i] += h;
            o_minus[i] -= h;

            let mut attn_plus = attn.clone();
            attn_plus.o_proj = o_plus;
            let mut attn_minus = attn.clone();
            attn_minus.o_proj = o_minus;

            let y_plus = attn_plus.forward(&x, seq_len);
            let y_minus = attn_minus.forward(&x, seq_len);

            let l_plus: f32 = y_plus.iter().zip(dy.iter()).map(|(&a, &b)| a * b).sum();
            let l_minus: f32 = y_minus.iter().zip(dy.iter()).map(|(&a, &b)| a * b).sum();
            let num_grad = (l_plus - l_minus) / (2.0 * h);

            let diff = (d_o_ana[i] - num_grad).abs();
            assert!(
                diff < 2e-2,
                "Attention o_proj gradient mismatch at index {}: analytical={}, numerical={}",
                i,
                d_o_ana[i],
                num_grad
            );
        }

        // 3. Verify d_v_proj
        for i in 0..v_proj.len() {
            let mut v_plus = v_proj.clone();
            let mut v_minus = v_proj.clone();
            v_plus[i] += h;
            v_minus[i] -= h;

            let mut attn_plus = attn.clone();
            attn_plus.v_proj = v_plus;
            let mut attn_minus = attn.clone();
            attn_minus.v_proj = v_minus;

            let y_plus = attn_plus.forward(&x, seq_len);
            let y_minus = attn_minus.forward(&x, seq_len);

            let l_plus: f32 = y_plus.iter().zip(dy.iter()).map(|(&a, &b)| a * b).sum();
            let l_minus: f32 = y_minus.iter().zip(dy.iter()).map(|(&a, &b)| a * b).sum();
            let num_grad = (l_plus - l_minus) / (2.0 * h);

            let diff = (d_v_ana[i] - num_grad).abs();
            assert!(
                diff < 2e-2,
                "Attention v_proj gradient mismatch at index {}: analytical={}, numerical={}",
                i,
                d_v_ana[i],
                num_grad
            );
        }

        // 4. Verify d_k_proj
        for i in 0..k_proj.len() {
            let mut k_plus = k_proj.clone();
            let mut k_minus = k_proj.clone();
            k_plus[i] += h;
            k_minus[i] -= h;

            let mut attn_plus = attn.clone();
            attn_plus.k_proj = k_plus;
            let mut attn_minus = attn.clone();
            attn_minus.k_proj = k_minus;

            let y_plus = attn_plus.forward(&x, seq_len);
            let y_minus = attn_minus.forward(&x, seq_len);

            let l_plus: f32 = y_plus.iter().zip(dy.iter()).map(|(&a, &b)| a * b).sum();
            let l_minus: f32 = y_minus.iter().zip(dy.iter()).map(|(&a, &b)| a * b).sum();
            let num_grad = (l_plus - l_minus) / (2.0 * h);

            let diff = (d_k_ana[i] - num_grad).abs();
            assert!(
                diff < 2e-2,
                "Attention k_proj gradient mismatch at index {}: analytical={}, numerical={}",
                i,
                d_k_ana[i],
                num_grad
            );
        }

        // 5. Verify d_q_proj
        for i in 0..q_proj.len() {
            let mut q_plus = q_proj.clone();
            let mut q_minus = q_proj.clone();
            q_plus[i] += h;
            q_minus[i] -= h;

            let mut attn_plus = attn.clone();
            attn_plus.q_proj = q_plus;
            let mut attn_minus = attn.clone();
            attn_minus.q_proj = q_minus;

            let y_plus = attn_plus.forward(&x, seq_len);
            let y_minus = attn_minus.forward(&x, seq_len);

            let l_plus: f32 = y_plus.iter().zip(dy.iter()).map(|(&a, &b)| a * b).sum();
            let l_minus: f32 = y_minus.iter().zip(dy.iter()).map(|(&a, &b)| a * b).sum();
            let num_grad = (l_plus - l_minus) / (2.0 * h);

            let diff = (d_q_ana[i] - num_grad).abs();
            assert!(
                diff < 2e-2,
                "Attention q_proj gradient mismatch at index {}: analytical={}, numerical={}",
                i,
                d_q_ana[i],
                num_grad
            );
        }
    }
}
