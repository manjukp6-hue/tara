//! Fast-Weight Synaptic Plasticity Engine.
//!
//! Implements biologically inspired fast associative memory:
//! - Fast weight matrix A of dimension [dim, dim].
//! - Auto-associative and hetero-associative Hebbian updates with decay:
//!   A_{t+1} = lambda * A_t + eta * (x_t * y_t^T)
//! - Associative retrieval: y_pred = A_t * query

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FastWeightPlasticityMatrix {
    pub dim: usize,
    pub weights: Vec<f32>, // Flat [dim, dim]
    pub lambda: f32,       // Decay rate [0.0, 1.0] (e.g. 0.95)
    pub eta: f32,          // Plasticity learning rate (e.g. 0.5)
}

/// Default dimensionality of the fast-weights associative memory matrix.
pub const DEFAULT_DIM: usize = 64;
/// Default per-step decay factor.
pub const DEFAULT_LAMBDA: f32 = 0.95;
/// Default plasticity learning rate.
pub const DEFAULT_ETA: f32 = 0.5;

impl Default for FastWeightPlasticityMatrix {
    fn default() -> Self {
        Self::new(DEFAULT_DIM, DEFAULT_LAMBDA, DEFAULT_ETA)
    }
}

impl FastWeightPlasticityMatrix {
    pub fn new(dim: usize, lambda: f32, eta: f32) -> Self {
        Self {
            dim,
            weights: vec![0.0f32; dim * dim],
            lambda: lambda.clamp(0.0, 1.0),
            eta,
        }
    }

    /// Reset matrix to zero.
    pub fn reset(&mut self) {
        self.weights.fill(0.0);
    }

    /// Associative Hebbian write step:
    /// A = lambda * A + eta * (key * value^T)
    pub fn write_association(&mut self, key: &[f32], value: &[f32]) {
        let d = self.dim;
        assert_eq!(key.len(), d);
        assert_eq!(value.len(), d);

        // Normalize inputs
        let k_norm = norm(key).max(1e-8);
        let v_norm = norm(value).max(1e-8);

        for (i, &k_elem) in key.iter().enumerate() {
            let ki = k_elem / k_norm;
            for (j, &v_elem) in value.iter().enumerate() {
                let vj = v_elem / v_norm;
                let idx = i * d + j;
                self.weights[idx] = self.lambda * self.weights[idx] + self.eta * (ki * vj);
            }
        }
    }

    /// Associative read step:
    /// output = query^T * A = sum_i query_i * A_{i, j}
    pub fn read_association(&self, query: &[f32]) -> Vec<f32> {
        let d = self.dim;
        assert_eq!(query.len(), d);

        let q_norm = norm(query).max(1e-8);
        let mut out = vec![0.0f32; d];

        for (i, &q_elem) in query.iter().enumerate() {
            let qi = q_elem / q_norm;
            for (j, out_elem) in out.iter_mut().enumerate() {
                *out_elem += qi * self.weights[i * d + j];
            }
        }

        out
    }

    /// Compute cosine similarity between two vectors.
    pub fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        let d = self.dim;
        if a.len() != d || b.len() != d {
            return 0.0;
        }

        let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        let na = norm(a);
        let nb = norm(b);

        if na <= 1e-8 || nb <= 1e-8 {
            0.0
        } else {
            dot / (na * nb)
        }
    }
}

fn norm(v: &[f32]) -> f32 {
    v.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_weights_associative_recall() {
        let dim = 8;
        let mut fwm = FastWeightPlasticityMatrix::new(dim, 0.95, 1.0);

        let mut key1 = vec![0.0f32; dim];
        key1[0] = 1.0;
        key1[1] = 1.0;

        let mut val1 = vec![0.0f32; dim];
        val1[4] = 1.0;
        val1[5] = 1.0;

        fwm.write_association(&key1, &val1);

        // Query with key1
        let recalled = fwm.read_association(&key1);
        let sim = fwm.cosine_similarity(&recalled, &val1);
        assert!(
            sim > 0.99,
            "Recalled vector should closely match stored value vector: got {sim}"
        );

        // Query with orthogonal key
        let mut key_ortho = vec![0.0f32; dim];
        key_ortho[2] = 1.0;
        let recalled_ortho = fwm.read_association(&key_ortho);
        let sim_ortho = fwm.cosine_similarity(&recalled_ortho, &val1);
        assert!(
            sim_ortho < 0.01,
            "Orthogonal query should yield zero recall: got {sim_ortho}"
        );
    }
}
