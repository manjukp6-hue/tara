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

    /// Compute gradients for inputs `x` and weights `weight` given incoming gradient `dy`.
    ///
    /// Returns `(dx, dweight)`:
    /// - `dx`: shape `[seq_len, hidden_size]`
    /// - `dweight`: shape `[hidden_size]`
    pub fn backward(
        &self,
        dy: &[f32],
        x: &[f32],
        seq_len: usize,
        hidden_size: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut dx = vec![0.0f32; seq_len * hidden_size];
        let mut dweight = vec![0.0f32; hidden_size];

        for t in 0..seq_len {
            let x_row = &x[t * hidden_size..(t + 1) * hidden_size];
            let dy_row = &dy[t * hidden_size..(t + 1) * hidden_size];

            let mean_sq: f32 = x_row.iter().map(|&v| v * v).sum::<f32>() / hidden_size as f32;
            let rms_sq = mean_sq + self.eps;
            let rms = rms_sq.sqrt();
            let inv_rms = 1.0 / rms;

            let mut sum_dy_w_x = 0.0f32;
            for i in 0..hidden_size {
                let xi = x_row[i];
                let dyi = dy_row[i];
                let wi = self.weight[i];
                dweight[i] += dyi * xi * inv_rms;
                sum_dy_w_x += dyi * wi * xi;
            }

            let dx_row = &mut dx[t * hidden_size..(t + 1) * hidden_size];
            let scale_factor = sum_dy_w_x / (hidden_size as f32 * rms_sq);
            for i in 0..hidden_size {
                let dyi = dy_row[i];
                let wi = self.weight[i];
                let xi = x_row[i];
                dx_row[i] = inv_rms * (dyi * wi - xi * scale_factor);
            }
        }

        (dx, dweight)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_norm_numerical_gradient_check() {
        let hidden_size = 4;
        let seq_len = 2;
        let eps = 1e-5;
        let norm = RmsNorm::new(vec![1.2, 0.9, 1.1, 0.8], eps);
        let x = vec![0.5, -0.3, 0.8, -0.2, 1.1, 0.4, -0.9, 0.7];
        let dy = vec![0.1, -0.2, 0.4, 0.3, -0.3, 0.5, 0.2, -0.1];

        let (dx_analytical, dw_analytical) = norm.backward(&dy, &x, seq_len, hidden_size);

        // Check dx with centered finite difference
        let h = 1e-3f32;
        for i in 0..x.len() {
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();
            x_plus[i] += h;
            x_minus[i] -= h;

            let y_plus = norm.forward(&x_plus, seq_len, hidden_size);
            let y_minus = norm.forward(&x_minus, seq_len, hidden_size);

            let l_plus: f32 = y_plus.iter().zip(dy.iter()).map(|(&a, &b)| a * b).sum();
            let l_minus: f32 = y_minus.iter().zip(dy.iter()).map(|(&a, &b)| a * b).sum();
            let num_grad = (l_plus - l_minus) / (2.0 * h);

            let diff = (dx_analytical[i] - num_grad).abs();
            assert!(
                diff < 1e-2,
                "RMSNorm dx gradient mismatch at index {}: analytical={}, numerical={}",
                i,
                dx_analytical[i],
                num_grad
            );
        }

        // Check dw with centered finite difference
        for i in 0..norm.weight.len() {
            let mut w_plus = norm.weight.clone();
            let mut w_minus = norm.weight.clone();
            w_plus[i] += h;
            w_minus[i] -= h;

            let norm_plus = RmsNorm::new(w_plus, eps);
            let norm_minus = RmsNorm::new(w_minus, eps);

            let y_plus = norm_plus.forward(&x, seq_len, hidden_size);
            let y_minus = norm_minus.forward(&x, seq_len, hidden_size);

            let l_plus: f32 = y_plus.iter().zip(dy.iter()).map(|(&a, &b)| a * b).sum();
            let l_minus: f32 = y_minus.iter().zip(dy.iter()).map(|(&a, &b)| a * b).sum();
            let num_grad = (l_plus - l_minus) / (2.0 * h);

            let diff = (dw_analytical[i] - num_grad).abs();
            assert!(
                diff < 1e-2,
                "RMSNorm dw gradient mismatch at index {}: analytical={}, numerical={}",
                i,
                dw_analytical[i],
                num_grad
            );
        }
    }
}
