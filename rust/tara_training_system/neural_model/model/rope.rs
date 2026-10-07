//! Rotary Position Embeddings (RoPE).
//!
//! Implements the rotate-half formulation used in TARA's attention layers.
//! See "RoFormer: Enhanced Transformer with Rotary Position Embedding"
//! (Su et al., 2022).

/// Dimensions for applying multi-head rotary embeddings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RotaryDimensions {
    pub seq_len: usize,
    pub num_heads_q: usize,
    pub num_heads_k: usize,
    pub head_dim: usize,
}

/// Rotary position embedding generator.
///
/// Precomputes sin/cos tables on demand; tables are cheap to recompute for
/// every forward pass given TARA's small sequence lengths.
#[derive(Clone, Debug)]
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
        Self {
            dim,
            max_positions,
            theta,
        }
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

    /// Compute cosine and sine tables for a single position `pos`.
    pub fn get_cos_sin_at(&self, pos: usize) -> (Vec<f32>, Vec<f32>) {
        let half_dim = self.dim / 2;
        let mut cos_table = vec![0.0f32; self.dim];
        let mut sin_table = vec![0.0f32; self.dim];

        for i in 0..half_dim {
            let freq = 1.0 / self.theta.powf(2.0 * i as f32 / self.dim as f32);
            let angle = pos as f32 * freq;
            let c = angle.cos();
            let s = angle.sin();
            cos_table[i] = c;
            cos_table[half_dim + i] = c;
            sin_table[i] = s;
            sin_table[half_dim + i] = s;
        }

        (cos_table, sin_table)
    }

    /// Apply rotary embeddings to a single-token query and key slice.
    pub fn apply_rotary_emb_single(
        q: &[f32],
        k: &[f32],
        cos: &[f32],
        sin: &[f32],
        nh: usize,
        nkv: usize,
        hd: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let half_dim = hd / 2;
        let mut q_out = q.to_vec();
        let mut k_out = k.to_vec();

        for h in 0..nh {
            let off = h * hd;
            for i in 0..half_dim {
                let q1 = q[off + i];
                let q2 = q[off + half_dim + i];
                let c = cos[i];
                let s = sin[i];
                q_out[off + i] = q1 * c - q2 * s;
                q_out[off + half_dim + i] = q2 * c + q1 * s;
            }
        }

        for h in 0..nkv {
            let off = h * hd;
            for i in 0..half_dim {
                let k1 = k[off + i];
                let k2 = k[off + half_dim + i];
                let c = cos[i];
                let s = sin[i];
                k_out[off + i] = k1 * c - k2 * s;
                k_out[off + half_dim + i] = k2 * c + k1 * s;
            }
        }

        (q_out, k_out)
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
        dims: RotaryDimensions,
    ) -> (Vec<f32>, Vec<f32>) {
        let seq_len = dims.seq_len;
        let num_heads_q = dims.num_heads_q;
        let num_heads_k = dims.num_heads_k;
        let head_dim = dims.head_dim;
        let mut out_q = vec![0.0f32; seq_len * num_heads_q * head_dim];
        let mut out_k = vec![0.0f32; seq_len * num_heads_k * head_dim];

        rotate_tensor(q, cos, sin, seq_len, num_heads_q, head_dim, &mut out_q);
        rotate_tensor(k, cos, sin, seq_len, num_heads_k, head_dim, &mut out_k);

        (out_q, out_k)
    }

    /// Apply inverse/adjoint rotary embeddings to gradients `dq` and `dk`.
    ///
    /// Since the RoPE transformation is an orthogonal rotation per frequency pair,
    /// its backward pass is its exact transpose (adjoint):
    /// `dx_1 = dy_1 * cos + dy_2 * sin`
    /// `dx_2 = dy_2 * cos - dy_1 * sin`
    pub fn apply_rotary_emb_backward(
        dq: &[f32],
        dk: &[f32],
        cos: &[f32],
        sin: &[f32],
        dims: RotaryDimensions,
    ) -> (Vec<f32>, Vec<f32>) {
        let seq_len = dims.seq_len;
        let num_heads_q = dims.num_heads_q;
        let num_heads_k = dims.num_heads_k;
        let head_dim = dims.head_dim;
        let mut out_dq = vec![0.0f32; seq_len * num_heads_q * head_dim];
        let mut out_dk = vec![0.0f32; seq_len * num_heads_k * head_dim];

        rotate_tensor_backward(dq, cos, sin, seq_len, num_heads_q, head_dim, &mut out_dq);
        rotate_tensor_backward(dk, cos, sin, seq_len, num_heads_k, head_dim, &mut out_dk);

        (out_dq, out_dk)
    }
}

/// Apply adjoint rotate-half RoPE to incoming gradient tensor.
fn rotate_tensor_backward(
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
            let dy = &src[base..base + head_dim];

            for i in 0..half {
                let dy1 = dy[i];
                let dy2 = dy[half + i];
                let c = cos_row[i];
                let s = sin_row[i];
                dst[base + i] = dy1 * c + dy2 * s;
                dst[base + half + i] = dy2 * c - dy1 * s;
            }
        }
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
                dst[base + i] = x1 * cos_row[i] - x2 * sin_row[i];
                dst[base + half + i] = x2 * cos_row[half + i] + x1 * sin_row[half + i];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_numerical_adjoint_gradient_check() {
        let head_dim = 4;
        let seq_len = 3;
        let num_heads_q = 2;
        let num_heads_k = 1;
        let rope = RotaryEmbedding::new(head_dim, 16, 10000.0);
        let (cos, sin) = rope.get_cos_sin(seq_len);

        let q = vec![
            0.5, -0.3, 0.8, -0.2, 1.1, 0.4, -0.9, 0.7, -0.1, 0.6, 0.3, -0.5, 0.8, -0.4, 0.2, 0.9,
            0.4, -0.2, 0.7, 0.1, -0.6, 0.5, -0.3, 0.8,
        ];
        let k = vec![
            0.2, 0.9, -0.4, 0.6, -0.7, 0.3, 0.5, -0.1, 0.6, -0.8, 0.1, 0.4,
        ];

        let dims = RotaryDimensions {
            seq_len,
            num_heads_q,
            num_heads_k,
            head_dim,
        };

        let (rot_q, rot_k) = RotaryEmbedding::apply_rotary_emb(&q, &k, &cos, &sin, dims);

        let dq_in = vec![
            0.1, 0.2, -0.1, 0.3, -0.4, 0.1, 0.2, -0.3, 0.5, -0.2, 0.1, 0.4, -0.1, 0.3, -0.2, 0.1,
            -0.3, 0.4, 0.2, -0.1, 0.1, -0.5, 0.3, 0.2,
        ];
        let dk_in = vec![
            0.3, -0.1, 0.4, -0.2, 0.1, 0.5, -0.3, 0.2, -0.4, 0.2, 0.1, -0.3,
        ];

        let (dq_out, dk_out) =
            RotaryEmbedding::apply_rotary_emb_backward(&dq_in, &dk_in, &cos, &sin, dims);

        // Adjoint Identity: <rot_q, dq_in> must equal <q, dq_out>
        let dot_forward_q: f32 = rot_q.iter().zip(dq_in.iter()).map(|(&a, &b)| a * b).sum();
        let dot_backward_q: f32 = q.iter().zip(dq_out.iter()).map(|(&a, &b)| a * b).sum();
        assert!(
            (dot_forward_q - dot_backward_q).abs() < 1e-4,
            "RoPE Q adjoint check failed: forward_dot={}, backward_dot={}",
            dot_forward_q,
            dot_backward_q
        );

        // Adjoint Identity: <rot_k, dk_in> must equal <k, dk_out>
        let dot_forward_k: f32 = rot_k.iter().zip(dk_in.iter()).map(|(&a, &b)| a * b).sum();
        let dot_backward_k: f32 = k.iter().zip(dk_out.iter()).map(|(&a, &b)| a * b).sum();
        assert!(
            (dot_forward_k - dot_backward_k).abs() < 1e-4,
            "RoPE K adjoint check failed: forward_dot={}, backward_dot={}",
            dot_forward_k,
            dot_backward_k
        );
    }
}
