//! SwiGLU MLP layer.
//!
//! Implements the gated feed-forward network used in each TaraDecoderLayer.
//! Formula: `down_proj( silu(gate_proj(x)) * up_proj(x) )`.

use std::collections::HashMap;
use thiserror::Error;

/// Errors building the MLP layer.
#[derive(Debug, Error)]
pub enum MlpError {
    #[error("missing weight: {0}")]
    MissingWeight(String),
}

/// Compute SiLU (Sigmoid Linear Unit): `x * sigmoid(x)`.
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Dense matrix multiply: `C = A * B^T` where
/// - `A` has shape `[m, k]` (row-major)
/// - `B` has shape `[n, k]` (row-major, i.e., stored as k-width rows)
/// - `C` has shape `[m, n]`
///
/// This is the standard linear projection `y = x @ W^T` convention used
/// by PyTorch `nn.Linear`.
pub fn mat_mul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for l in 0..k {
                sum += a[i * k + l] * b[j * k + l];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

/// SwiGLU MLP layer with gate/up/down projections.
pub struct TaraMlp {
    /// Gate projection weight: shape `[intermediate_size, hidden_size]`.
    pub gate_proj: Vec<f32>,
    /// Up projection weight: shape `[intermediate_size, hidden_size]`.
    pub up_proj: Vec<f32>,
    /// Down projection weight: shape `[hidden_size, intermediate_size]`.
    pub down_proj: Vec<f32>,
}

impl TaraMlp {
    /// Construct from pre-loaded weight vectors.
    pub fn new(gate_proj: Vec<f32>, up_proj: Vec<f32>, down_proj: Vec<f32>) -> Self {
        Self { gate_proj, up_proj, down_proj }
    }

    /// Load from the global weight map for `layer_idx`.
    pub fn from_weights(
        weights: &HashMap<String, Vec<f32>>,
        layer_idx: usize,
    ) -> Result<Self, MlpError> {
        let prefix = format!("model.layers.{}.mlp", layer_idx);
        let gate_proj = weights
            .get(&format!("{}.gate_proj.weight", prefix))
            .cloned()
            .ok_or_else(|| MlpError::MissingWeight(format!("{}.gate_proj.weight", prefix)))?;
        let up_proj = weights
            .get(&format!("{}.up_proj.weight", prefix))
            .cloned()
            .ok_or_else(|| MlpError::MissingWeight(format!("{}.up_proj.weight", prefix)))?;
        let down_proj = weights
            .get(&format!("{}.down_proj.weight", prefix))
            .cloned()
            .ok_or_else(|| MlpError::MissingWeight(format!("{}.down_proj.weight", prefix)))?;
        Ok(Self::new(gate_proj, up_proj, down_proj))
    }

    /// Apply the SwiGLU MLP to `x`.
    ///
    /// `x` is a flat buffer of shape `[seq_len, hidden_size]`.
    /// Returns a buffer of the same shape.
    pub fn forward(
        &self,
        x: &[f32],
        seq_len: usize,
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Vec<f32> {
        // gate = silu(x @ gate_proj^T)   [seq, inter]
        let gate_raw = mat_mul(x, &self.gate_proj, seq_len, hidden_size, intermediate_size);
        let gate: Vec<f32> = gate_raw.iter().map(|&v| silu(v)).collect();

        // up = x @ up_proj^T             [seq, inter]
        let up = mat_mul(x, &self.up_proj, seq_len, hidden_size, intermediate_size);

        // hidden = gate * up             [seq, inter]
        let hidden: Vec<f32> = gate.iter().zip(up.iter()).map(|(&g, &u)| g * u).collect();

        // out = hidden @ down_proj^T     [seq, hidden]
        mat_mul(&hidden, &self.down_proj, seq_len, intermediate_size, hidden_size)
    }
}
