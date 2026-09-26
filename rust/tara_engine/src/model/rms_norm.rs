//! RMSNorm layer used in TaraForCausalLM.
//!
//! Implements Root Mean Square Layer Normalization as described in
//! "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019).

/// Root Mean Square normalisation layer.
///
/// Normalises the last dimension of an activation tensor and applies a learned
/// per-channel scale (`weight`).
pub struct RmsNorm {
    /// Per-channel scale parameters (length = `hidden_size`).
    pub weight: Vec<f32>,
    /// Numerical stability epsilon added inside the square root.
    pub eps: f32,
}

impl RmsNorm {
    /// Construct from pre-loaded weight vector and epsilon.
    pub fn new(weight: Vec<f32>, eps: f32) -> Self {
        Self { weight, eps }
    }

    /// Apply RMSNorm to `x`.
    ///
    /// `x` is a flat buffer of shape `[seq_len, hidden_size]` in row-major
    /// order.  Returns a new buffer of the same shape.
    ///
    /// Formula per token `t`:
    /// ```text
    /// rms   = sqrt( mean(x[t,:]^2) + eps )
    /// y[t,i] = x[t,i] / rms * weight[i]
    /// ```
    pub fn forward(&self, x: &[f32], seq_len: usize, hidden_size: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; seq_len * hidden_size];

        for t in 0..seq_len {
            let row = &x[t * hidden_size..(t + 1) * hidden_size];

            // Compute RMS
            let mean_sq: f32 = row.iter().map(|&v| v * v).sum::<f32>() / hidden_size as f32;
            let rms = (mean_sq + self.eps).sqrt();
            let inv_rms = 1.0 / rms;

            let out_row = &mut out[t * hidden_size..(t + 1) * hidden_size];
            for (i, (&xi, &wi)) in row.iter().zip(self.weight.iter()).enumerate() {
                out_row[i] = xi * inv_rms * wi;
            }
        }

        out
    }
}
