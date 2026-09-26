//! Rotary Position Embeddings (RoPE).
//!
//! Implements the rotate-half formulation used in TARA's attention layers.
//! See "RoFormer: Enhanced Transformer with Rotary Position Embedding"
//! (Su et al., 2022).

use std::f32::consts::PI;

/// Rotary position embedding generator.
///
/// Precomputes sin/cos tables on demand; tables are cheap to recompute for
/// every forward pass given TARA's small sequence lengths.
pub struct RotaryEmbedding {
    /// Half the head dimension (number of frequency pairs).
    pub dim: usize,
    /// Maximum number of positions the table covers.
    pub max_positions: usize,
    /// Base frequency θ.
    pub theta: f32,
}

impl RotaryEmbedding {
    /// Construct a new [`RotaryEmbedding`].
    ///
    /// `dim` should be equal to `head_dim`.
    pub fn new(dim: usize, max_positions: usize, theta: f32) -> Self {
        Self { dim, max_positions, theta }
    }

    /// Compute cosine and sine tables for `seq_len` positions.
    ///
    /// Returns two flat vectors of shape `[seq_len, dim]` (row-major).
    /// Each position `p` has `dim/2` cos/sin pairs interleaved as:
    /// `[cos(p*θ_0), cos(p*θ_1), …, cos(p*θ_{d/2-1})]`
    pub fn get_cos_sin(&self, seq_len: usize) -> (Vec<f32>, Vec<f32>) {
        let half_dim = self.dim / 2;
        let mut cos_table = vec![0.0f32; seq_len * self.dim];
        let mut sin_table = vec![0.0f32; seq_len * self.dim];

        for pos in 0..seq_len {
            for i in 0..half_dim {
                let freq = 1.0 / self.theta.powf(2.0 * i as f32 / self.dim as f32);
                let angle = pos as f32 * freq;
                let c = angle.cos();
                let s = angle.sin();
                // Store cos/sin for both halves (rotate-half layout)
                cos_table[pos * self.dim + i] = c;
                cos_table[pos * self.dim + half_dim + i] = c;
                sin_table[pos * self.dim + i] = s;
                sin_table[pos * self.dim + half_dim + i] = s;
            }
        }

        (cos_table, sin_table)
    }

    /// Apply rotary embeddings to query and key tensors.
    ///
    /// Inputs `q` and `k` are flat buffers:
    /// - `q`: shape `[seq_len, num_heads_q, head_dim]`
    /// - `k`: shape `[seq_len, num_heads_k, head_dim]`
    ///
    /// `cos` and `sin` are the tables returned by [`get_cos_sin`], shaped
    /// `[seq_len, head_dim]`.
    ///
    /// Returns `(rotated_q, rotated_k)` with the same shapes.
    pub fn apply_rotary_emb(
        q: &[f32],
        k: &[f32],
        cos: &[f32],
        sin: &[f32],
        seq_len: usize,
        num_heads_q: usize,
        num_heads_k: usize,
        head_dim: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut out_q = vec![0.0f32; seq_len * num_heads_q * head_dim];
        let mut out_k = vec![0.0f32; seq_len * num_heads_k * head_dim];

        rotate_tensor(q, cos, sin, seq_len, num_heads_q, head_dim, &mut out_q);
        rotate_tensor(k, cos, sin, seq_len, num_heads_k, head_dim, &mut out_k);

        (out_q, out_k)
    }
}

/// Apply rotate-half RoPE to a `[seq_len, num_heads, head_dim]` tensor.
fn rotate_tensor(
    src: &[f32],
    cos: &[f32],
    sin: &[f32],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    dst: &mut [f32],
) {
    let half = head_dim / 2;

    for pos in 0..seq_len {
        let cos_row = &cos[pos * head_dim..(pos + 1) * head_dim];
        let sin_row = &sin[pos * head_dim..(pos + 1) * head_dim];

        for h in 0..num_heads {
            let base = pos * num_heads * head_dim + h * head_dim;
            let x = &src[base..base + head_dim];

            // rotate_half: [-x2, x1] interleaved over the two halves
            for i in 0..half {
                let x1 = x[i];
                let x2 = x[half + i];
                dst[base + i]        = x1 * cos_row[i]       - x2 * sin_row[i];
                dst[base + half + i] = x2 * cos_row[half + i] + x1 * sin_row[half + i];
            }
        }
    }
}
