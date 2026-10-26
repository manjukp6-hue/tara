//! Cross-modal feature projector.
//!
//! Provides linear and multi-layer perceptron (MLP) projection layers mapping
//! modality-specific feature representations (e.g., raw visual patches, audio Mel frames)
//! into the target transformer embedding space (`hidden_size`).

use serde::{Deserialize, Serialize};

/// Type of cross-modal projection layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProjectorType {
    /// Single linear projection: W * x + b
    Linear,
    /// 2-layer MLP with GELU non-linearity: W2 * GELU(W1 * x + b1) + b2
    Mlp,
}

/// Dynamic cross-modal projection module.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossModalProjector {
    pub projector_type: ProjectorType,
    pub input_dim: usize,
    pub output_dim: usize,
    pub intermediate_dim: usize,
    /// Weight matrix 1: shape [intermediate_dim, input_dim] for MLP, or [output_dim, input_dim] for Linear
    pub w1: Vec<f32>,
    /// Optional bias 1: shape [intermediate_dim] or [output_dim]
    pub b1: Vec<f32>,
    /// Weight matrix 2 for MLP: shape [output_dim, intermediate_dim]
    pub w2: Option<Vec<f32>>,
    /// Optional bias 2 for MLP: shape [output_dim]
    pub b2: Option<Vec<f32>>,
}

impl CrossModalProjector {
    /// Construct a new projector with dynamic dimensions and deterministic orthogonal/Xavier initialization.
    pub fn new(
        projector_type: ProjectorType,
        input_dim: usize,
        output_dim: usize,
        intermediate_dim: usize,
    ) -> Self {
        assert!(input_dim > 0, "input_dim must be positive");
        assert!(output_dim > 0, "output_dim must be positive");

        match projector_type {
            ProjectorType::Linear => {
                let w1_size = output_dim * input_dim;
                let scale = (2.0 / (input_dim + output_dim) as f32).sqrt();
                let mut w1 = Vec::with_capacity(w1_size);
                for i in 0..w1_size {
                    let phase = (i as f32 * 0.173 + 0.5).sin();
                    w1.push(phase * scale);
                }
                let b1 = vec![0.0f32; output_dim];
                Self {
                    projector_type,
                    input_dim,
                    output_dim,
                    intermediate_dim: output_dim,
                    w1,
                    b1,
                    w2: None,
                    b2: None,
                }
            }
            ProjectorType::Mlp => {
                let inter = if intermediate_dim == 0 {
                    output_dim * 2
                } else {
                    intermediate_dim
                };
                let w1_size = inter * input_dim;
                let scale1 = (2.0 / (input_dim + inter) as f32).sqrt();
                let mut w1 = Vec::with_capacity(w1_size);
                for i in 0..w1_size {
                    let phase = (i as f32 * 0.231 + 0.3).sin();
                    w1.push(phase * scale1);
                }
                let b1 = vec![0.0f32; inter];

                let w2_size = output_dim * inter;
                let scale2 = (2.0 / (inter + output_dim) as f32).sqrt();
                let mut w2 = Vec::with_capacity(w2_size);
                for i in 0..w2_size {
                    let phase = (i as f32 * 0.317 + 0.7).cos();
                    w2.push(phase * scale2);
                }
                let b2 = vec![0.0f32; output_dim];

                Self {
                    projector_type,
                    input_dim,
                    output_dim,
                    intermediate_dim: inter,
                    w1,
                    b1,
                    w2: Some(w2),
                    b2: Some(b2),
                }
            }
        }
    }

    /// Project a batch of input feature vectors [num_items, input_dim] into [num_items, output_dim].
    pub fn project(&self, inputs: &[f32], num_items: usize) -> Vec<f32> {
        if num_items == 0 || inputs.is_empty() {
            return Vec::new();
        }
        assert!(
            inputs.len() >= num_items * self.input_dim,
            "inputs length {} is smaller than required {}",
            inputs.len(),
            num_items * self.input_dim
        );

        match self.projector_type {
            ProjectorType::Linear => {
                let mut output = vec![0.0f32; num_items * self.output_dim];
                for item in 0..num_items {
                    let in_row = &inputs[item * self.input_dim..(item + 1) * self.input_dim];
                    let out_row = &mut output[item * self.output_dim..(item + 1) * self.output_dim];

                    for (out_idx, out_val) in out_row.iter_mut().enumerate() {
                        let w_row = &self.w1[out_idx * self.input_dim..(out_idx + 1) * self.input_dim];
                        let dot: f32 = in_row.iter().zip(w_row.iter()).map(|(&x, &w)| x * w).sum();
                        *out_val = dot + self.b1[out_idx];
                    }
                }
                output
            }
            ProjectorType::Mlp => {
                let inter = self.intermediate_dim;
                let w2 = self.w2.as_ref().expect("w2 initialized for MLP");
                let b2 = self.b2.as_ref().expect("b2 initialized for MLP");

                let mut output = vec![0.0f32; num_items * self.output_dim];
                let mut hidden = vec![0.0f32; inter];

                for item in 0..num_items {
                    let in_row = &inputs[item * self.input_dim..(item + 1) * self.input_dim];

                    // First layer: Linear + GELU
                    for (h_idx, h_val) in hidden.iter_mut().enumerate() {
                        let w_row = &self.w1[h_idx * self.input_dim..(h_idx + 1) * self.input_dim];
                        let dot: f32 = in_row.iter().zip(w_row.iter()).map(|(&x, &w)| x * w).sum();
                        let linear_val = dot + self.b1[h_idx];
                        *h_val = gelu_approx(linear_val);
                    }

                    // Second layer: Linear
                    let out_row = &mut output[item * self.output_dim..(item + 1) * self.output_dim];
                    for out_idx in 0..self.output_dim {
                        let w_row = &w2[out_idx * inter..(out_idx + 1) * inter];
                        let dot: f32 = hidden.iter().zip(w_row.iter()).map(|(&h, &w)| h * w).sum();
                        out_row[out_idx] = dot + b2[out_idx];
                    }
                }
                output
            }
        }
    }
}

/// Gaussian Error Linear Unit (GELU) approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
#[inline]
pub fn gelu_approx(x: f32) -> f32 {
    let sqrt_2_over_pi = 0.797_884_6_f32;
    let coeff = 0.044715f32;
    let inner = sqrt_2_over_pi * (x + coeff * x.powi(3));
    0.5 * x * (1.0 + inner.tanh())
}
