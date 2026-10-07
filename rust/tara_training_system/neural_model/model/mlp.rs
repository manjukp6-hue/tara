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
        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
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
        mat_mul(
            &hidden,
            &self.down_proj,
            seq_len,
            intermediate_size,
            hidden_size,
        )
    }

    /// Compute gradients for inputs `x` and weights (`gate_proj`, `up_proj`, `down_proj`)
    /// given incoming gradient `dy`.
    ///
    /// Returns `(dx, d_gate_proj, d_up_proj, d_down_proj)`.
    pub fn backward(
        &self,
        dy: &[f32],
        x: &[f32],
        seq_len: usize,
        hidden_size: usize,
        intermediate_size: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        // Forward activations needed for backprop
        let gate_raw = mat_mul(x, &self.gate_proj, seq_len, hidden_size, intermediate_size);
        let gate: Vec<f32> = gate_raw.iter().map(|&v| silu(v)).collect();
        let up = mat_mul(x, &self.up_proj, seq_len, hidden_size, intermediate_size);
        let hidden: Vec<f32> = gate.iter().zip(up.iter()).map(|(&g, &u)| g * u).collect();

        // 1. d_down_proj = dy^T @ hidden: shape [hidden_size, intermediate_size]
        let mut d_down_proj = vec![0.0f32; hidden_size * intermediate_size];
        for h in 0..hidden_size {
            for i in 0..intermediate_size {
                let mut sum = 0.0f32;
                for t in 0..seq_len {
                    sum += dy[t * hidden_size + h] * hidden[t * intermediate_size + i];
                }
                d_down_proj[h * intermediate_size + i] = sum;
            }
        }

        // 2. d_hidden = dy @ down_proj: shape [seq_len, intermediate_size]
        let mut d_hidden = vec![0.0f32; seq_len * intermediate_size];
        for t in 0..seq_len {
            for i in 0..intermediate_size {
                let mut sum = 0.0f32;
                for h in 0..hidden_size {
                    sum += dy[t * hidden_size + h] * self.down_proj[h * intermediate_size + i];
                }
                d_hidden[t * intermediate_size + i] = sum;
            }
        }

        // 3. d_up = d_hidden * gate: shape [seq_len, intermediate_size]
        // 4. d_gate_raw = d_hidden * up * silu'(gate_raw): shape [seq_len, intermediate_size]
        let mut d_up = vec![0.0f32; seq_len * intermediate_size];
        let mut d_gate_raw = vec![0.0f32; seq_len * intermediate_size];
        for idx in 0..seq_len * intermediate_size {
            let dh = d_hidden[idx];
            d_up[idx] = dh * gate[idx];

            let z = gate_raw[idx];
            let sig = 1.0 / (1.0 + (-z).exp());
            let d_silu = sig * (1.0 + z * (1.0 - sig));
            d_gate_raw[idx] = dh * up[idx] * d_silu;
        }

        // 5. d_up_proj = d_up^T @ x: shape [intermediate_size, hidden_size]
        let mut d_up_proj = vec![0.0f32; intermediate_size * hidden_size];
        for i in 0..intermediate_size {
            for h in 0..hidden_size {
                let mut sum = 0.0f32;
                for t in 0..seq_len {
                    sum += d_up[t * intermediate_size + i] * x[t * hidden_size + h];
                }
                d_up_proj[i * hidden_size + h] = sum;
            }
        }

        // 6. d_gate_proj = d_gate_raw^T @ x: shape [intermediate_size, hidden_size]
        let mut d_gate_proj = vec![0.0f32; intermediate_size * hidden_size];
        for i in 0..intermediate_size {
            for h in 0..hidden_size {
                let mut sum = 0.0f32;
                for t in 0..seq_len {
                    sum += d_gate_raw[t * intermediate_size + i] * x[t * hidden_size + h];
                }
                d_gate_proj[i * hidden_size + h] = sum;
            }
        }

        // 7. dx = d_up @ up_proj + d_gate_raw @ gate_proj: shape [seq_len, hidden_size]
        let mut dx = vec![0.0f32; seq_len * hidden_size];
        for t in 0..seq_len {
            for h in 0..hidden_size {
                let mut sum = 0.0f32;
                for i in 0..intermediate_size {
                    sum += d_up[t * intermediate_size + i] * self.up_proj[i * hidden_size + h]
                        + d_gate_raw[t * intermediate_size + i]
                            * self.gate_proj[i * hidden_size + h];
                }
                dx[t * hidden_size + h] = sum;
            }
        }

        (dx, d_gate_proj, d_up_proj, d_down_proj)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mlp_numerical_gradient_checks() {
        let hidden_size = 3;
        let intermediate_size = 4;
        let seq_len = 2;

        let gate_proj = vec![
            0.2, -0.4, 0.5, -0.1, 0.3, -0.2, 0.6, -0.1, 0.4, -0.5, 0.2, -0.3,
        ];
        let up_proj = vec![
            -0.3, 0.5, -0.2, 0.4, -0.1, 0.6, -0.2, 0.3, -0.5, 0.1, -0.4, 0.2,
        ];
        let down_proj = vec![
            0.4, -0.2, 0.3, -0.5, -0.1, 0.6, -0.4, 0.2, 0.5, -0.3, 0.1, -0.2,
        ];

        let mlp = TaraMlp::new(gate_proj.clone(), up_proj.clone(), down_proj.clone());

        let x = vec![0.3, -0.5, 0.8, -0.2, 0.6, -0.4];
        let dy = vec![0.2, -0.1, 0.4, -0.3, 0.5, -0.2];

        let (dx_ana, d_gate_ana, d_up_ana, d_down_ana) =
            mlp.backward(&dy, &x, seq_len, hidden_size, intermediate_size);

        let h = 1e-3f32;

        // 1. Verify dx
        for i in 0..x.len() {
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();
            x_plus[i] += h;
            x_minus[i] -= h;

            let y_plus = mlp.forward(&x_plus, seq_len, hidden_size, intermediate_size);
            let y_minus = mlp.forward(&x_minus, seq_len, hidden_size, intermediate_size);

            let l_plus: f32 = y_plus.iter().zip(dy.iter()).map(|(&a, &b)| a * b).sum();
            let l_minus: f32 = y_minus.iter().zip(dy.iter()).map(|(&a, &b)| a * b).sum();
            let num_grad = (l_plus - l_minus) / (2.0 * h);

            let diff = (dx_ana[i] - num_grad).abs();
            assert!(
                diff < 2e-2,
                "MLP dx gradient mismatch at index {}: analytical={}, numerical={}",
                i,
                dx_ana[i],
                num_grad
            );
        }

        // 2. Verify d_down_proj
        for i in 0..down_proj.len() {
            let mut down_plus = down_proj.clone();
            let mut down_minus = down_proj.clone();
            down_plus[i] += h;
            down_minus[i] -= h;

            let mlp_plus = TaraMlp::new(gate_proj.clone(), up_proj.clone(), down_plus);
            let mlp_minus = TaraMlp::new(gate_proj.clone(), up_proj.clone(), down_minus);

            let y_plus = mlp_plus.forward(&x, seq_len, hidden_size, intermediate_size);
            let y_minus = mlp_minus.forward(&x, seq_len, hidden_size, intermediate_size);

            let l_plus: f32 = y_plus.iter().zip(dy.iter()).map(|(&a, &b)| a * b).sum();
            let l_minus: f32 = y_minus.iter().zip(dy.iter()).map(|(&a, &b)| a * b).sum();
            let num_grad = (l_plus - l_minus) / (2.0 * h);

            let diff = (d_down_ana[i] - num_grad).abs();
            assert!(
                diff < 2e-2,
                "MLP down_proj gradient mismatch at index {}: analytical={}, numerical={}",
                i,
                d_down_ana[i],
                num_grad
            );
        }

        // 3. Verify d_up_proj
        for i in 0..up_proj.len() {
            let mut up_plus = up_proj.clone();
            let mut up_minus = up_proj.clone();
            up_plus[i] += h;
            up_minus[i] -= h;

            let mlp_plus = TaraMlp::new(gate_proj.clone(), up_plus, down_proj.clone());
            let mlp_minus = TaraMlp::new(gate_proj.clone(), up_minus, down_proj.clone());

            let y_plus = mlp_plus.forward(&x, seq_len, hidden_size, intermediate_size);
            let y_minus = mlp_minus.forward(&x, seq_len, hidden_size, intermediate_size);

            let l_plus: f32 = y_plus.iter().zip(dy.iter()).map(|(&a, &b)| a * b).sum();
            let l_minus: f32 = y_minus.iter().zip(dy.iter()).map(|(&a, &b)| a * b).sum();
            let num_grad = (l_plus - l_minus) / (2.0 * h);

            let diff = (d_up_ana[i] - num_grad).abs();
            assert!(
                diff < 2e-2,
                "MLP up_proj gradient mismatch at index {}: analytical={}, numerical={}",
                i,
                d_up_ana[i],
                num_grad
            );
        }

        // 4. Verify d_gate_proj
        for i in 0..gate_proj.len() {
            let mut gate_plus = gate_proj.clone();
            let mut gate_minus = gate_proj.clone();
            gate_plus[i] += h;
            gate_minus[i] -= h;

            let mlp_plus = TaraMlp::new(gate_plus, up_proj.clone(), down_proj.clone());
            let mlp_minus = TaraMlp::new(gate_minus, up_proj.clone(), down_proj.clone());

            let y_plus = mlp_plus.forward(&x, seq_len, hidden_size, intermediate_size);
            let y_minus = mlp_minus.forward(&x, seq_len, hidden_size, intermediate_size);

            let l_plus: f32 = y_plus.iter().zip(dy.iter()).map(|(&a, &b)| a * b).sum();
            let l_minus: f32 = y_minus.iter().zip(dy.iter()).map(|(&a, &b)| a * b).sum();
            let num_grad = (l_plus - l_minus) / (2.0 * h);

            let diff = (d_gate_ana[i] - num_grad).abs();
            assert!(
                diff < 2e-2,
                "MLP gate_proj gradient mismatch at index {}: analytical={}, numerical={}",
                i,
                d_gate_ana[i],
                num_grad
            );
        }
    }
}
