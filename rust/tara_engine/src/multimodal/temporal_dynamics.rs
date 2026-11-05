//! Native Spatio-Temporal World Model Latent Dynamics Predictor.
//!
//! Predicts next-state physical and visual latent transitions S_{t+1} from historical
//! spatiotemporal latent sequence S_{1..t} and optional physical action vectors A_t.
//! Operates in compressed latent token space to prevent CUDA OOM and enable stable
//! multi-step future rollouts without hardcoded capacity limits.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use thiserror::Error;

use super::vision::{RawVideo, VisionEncoder, VisionError};

/// Errors encountered in temporal dynamics prediction.
#[derive(Debug, Error)]
pub enum DynamicsError {
    #[error("empty state input: state vector length cannot be zero")]
    EmptyState,
    #[error("state dimension mismatch: expected {expected}, actual {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("action dimension mismatch: expected {expected}, actual {actual}")]
    ActionDimensionMismatch { expected: usize, actual: usize },
    #[error("zero rollout steps requested")]
    ZeroRolloutSteps,
    #[error("vision encoding error in dynamics: {0}")]
    Vision(#[from] VisionError),
    #[error("safetensors serialization error: {0}")]
    SafeTensors(String),
    #[error("I/O error in dynamics: {0}")]
    Io(#[from] std::io::Error),
}

/// Dynamically resolves learning rate for physical dynamics transitions.
fn default_dynamics_lr() -> f32 {
    std::env::var("TARA_DYNAMICS_LR")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|&v| v > 0.0 && v.is_finite())
        .unwrap_or(0.001)
}

/// Dynamic configuration for the Temporal World Model predictor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalDynamicsConfig {
    pub latent_dim: usize,
    pub action_dim: usize,
    pub intermediate_dim: usize,
    pub temporal_heads: usize,
    #[serde(default = "default_dynamics_lr")]
    pub learning_rate: f32,
}

impl TemporalDynamicsConfig {
    /// Construct dynamic configuration without hardcoded constants.
    pub fn new(
        latent_dim: usize,
        action_dim: usize,
        intermediate_dim: usize,
        temporal_heads: usize,
    ) -> Self {
        Self {
            latent_dim,
            action_dim,
            intermediate_dim,
            temporal_heads,
            learning_rate: default_dynamics_lr(),
        }
    }
}

/// Individual transition step with predicted latent dynamics, expected reward, and terminal status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicTransitionStep {
    pub next_latent: Vec<f32>,
    pub predicted_reward: f32,
    pub predicted_done: bool,
}

/// Native Temporal World Model Dynamics Predictor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalDynamicsPredictor {
    pub config: TemporalDynamicsConfig,
    /// Linear projection matrix: [intermediate_dim, latent_dim + action_dim]
    pub w_proj: Vec<f32>,
    pub b_proj: Vec<f32>,
    /// Update gate weights: [latent_dim, intermediate_dim]
    pub w_gate: Vec<f32>,
    pub b_gate: Vec<f32>,
    /// Candidate transformation weights: [latent_dim, intermediate_dim]
    pub w_cand: Vec<f32>,
    pub b_cand: Vec<f32>,
}

impl TemporalDynamicsPredictor {
    /// Construct a new predictor with deterministic variance-scaled initialization.
    pub fn new(config: TemporalDynamicsConfig) -> Self {
        assert!(config.latent_dim > 0, "latent_dim must be positive");
        assert!(config.intermediate_dim > 0, "intermediate_dim must be positive");

        let in_dim = config.latent_dim + config.action_dim;
        let p_scale = (2.0f32 / (in_dim + config.intermediate_dim) as f32).sqrt();
        let total_proj = config.intermediate_dim * in_dim;
        let mut w_proj = Vec::with_capacity(total_proj);
        let golden_ratio = 0.6180339887f32;

        for i in 0..total_proj {
            let val = (((i as f32 + 1.0) * golden_ratio).fract() * 2.0 - 1.0) * p_scale;
            w_proj.push(val);
        }
        let b_proj = vec![0.0f32; config.intermediate_dim];

        let g_scale = (2.0f32 / (config.intermediate_dim + config.latent_dim) as f32).sqrt();
        let total_gate = config.latent_dim * config.intermediate_dim;
        let mut w_gate = Vec::with_capacity(total_gate);
        let mut w_cand = Vec::with_capacity(total_gate);
        let sqrt2_fraction = 0.4142135623f32;
        let sqrt3_fraction = 0.7320508075f32;

        for i in 0..total_gate {
            let val_g = (((i as f32 + 1.0) * sqrt2_fraction).fract() * 2.0 - 1.0) * g_scale;
            let val_c = (((i as f32 + 1.0) * sqrt3_fraction).fract() * 2.0 - 1.0) * g_scale;
            w_gate.push(val_g);
            w_cand.push(val_c);
        }
        let b_gate = vec![0.0f32; config.latent_dim];
        let b_cand = vec![0.0f32; config.latent_dim];

        Self {
            config,
            w_proj,
            b_proj,
            w_gate,
            b_gate,
            w_cand,
            b_cand,
        }
    }

    /// Predict the next latent state S_{t+1} from current state S_t and optional action A_t.
    ///
    /// Applies a gated physical residual transition:
    /// 1. Combined feature: x = [S_t, A_t]
    /// 2. Intermediate state: h = GELU(W_proj * x + b_proj)
    /// 3. Update gate: z = sigmoid(W_gate * h + b_gate)
    /// 4. Candidate delta: \Delta S = tanh(W_cand * h + b_cand)
    /// 5. State update: S_{t+1} = S_t + z \odot \Delta S
    pub fn predict_next_step(
        &self,
        current_state: &[f32],
        action: Option<&[f32]>,
    ) -> Result<Vec<f32>, DynamicsError> {
        if current_state.is_empty() {
            return Err(DynamicsError::EmptyState);
        }
        if current_state.len() != self.config.latent_dim {
            return Err(DynamicsError::DimensionMismatch {
                expected: self.config.latent_dim,
                actual: current_state.len(),
            });
        }

        let mut in_buf = Vec::with_capacity(self.config.latent_dim + self.config.action_dim);
        in_buf.extend_from_slice(current_state);

        if self.config.action_dim > 0 {
            if let Some(act) = action {
                if act.len() != self.config.action_dim {
                    return Err(DynamicsError::ActionDimensionMismatch {
                        expected: self.config.action_dim,
                        actual: act.len(),
                    });
                }
                in_buf.extend_from_slice(act);
            } else {
                in_buf.extend(vec![0.0f32; self.config.action_dim]);
            }
        }

        // 1. Projection to intermediate feature space: [intermediate_dim]
        let in_dim = in_buf.len();
        let mut h = vec![0.0f32; self.config.intermediate_dim];
        let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
        let gelu_coeff = 0.044715f32;

        for i in 0..self.config.intermediate_dim {
            let mut sum = self.b_proj[i];
            let row_offset = i * in_dim;
            for j in 0..in_dim {
                sum += self.w_proj[row_offset + j] * in_buf[j];
            }
            // GELU non-linearity
            let gelu = 0.5 * sum * (1.0 + (sqrt_2_over_pi * (sum + gelu_coeff * sum.powi(3))).tanh());
            h[i] = gelu;
        }

        // 2. Gated transition: z = sigmoid(W_gate * h + b_gate), cand = tanh(W_cand * h + b_cand)
        let mut next_state = Vec::with_capacity(self.config.latent_dim);
        for i in 0..self.config.latent_dim {
            let row_offset = i * self.config.intermediate_dim;
            let mut sum_g = self.b_gate[i];
            let mut sum_c = self.b_cand[i];

            for j in 0..self.config.intermediate_dim {
                sum_g += self.w_gate[row_offset + j] * h[j];
                sum_c += self.w_cand[row_offset + j] * h[j];
            }

            // Sigmoid activation for gate
            let gate = 1.0 / (1.0 + (-sum_g).exp());
            // Tanh activation for candidate change
            let delta = sum_c.tanh();

            // Residual physical update: persistence + gated delta
            let s_next = current_state[i] + gate * delta;
            next_state.push(s_next);
        }

        Ok(next_state)
    }

    /// Rollout future latent states across N future time steps.
    /// Returns vector of states: [S_{t+1}, S_{t+2}, ..., S_{t+N}].
    pub fn rollout_horizon(
        &self,
        initial_state: &[f32],
        actions: Option<&[Vec<f32>]>,
        steps: usize,
    ) -> Result<Vec<Vec<f32>>, DynamicsError> {
        if steps == 0 {
            return Err(DynamicsError::ZeroRolloutSteps);
        }

        let mut trajectory = Vec::with_capacity(steps);
        let mut curr = initial_state.to_vec();

        for s in 0..steps {
            let act = actions.and_then(|a| a.get(s).map(|v| v.as_slice()));
            let next = self.predict_next_step(&curr, act)?;
            curr = next.clone();
            trajectory.push(next);
        }

        Ok(trajectory)
    }

    /// Predicts next transition step with latent state, reward, and done flags.
    pub fn predict_transition_step(
        &self,
        current_state: &[f32],
        action: Option<&[f32]>,
    ) -> Result<DynamicTransitionStep, DynamicsError> {
        let next_latent = self.predict_next_step(current_state, action)?;

        // Estimate transition reward from latent energy delta
        let curr_norm: f32 = current_state.iter().map(|v| v * v).sum::<f32>().sqrt();
        let next_norm: f32 = next_latent.iter().map(|v| v * v).sum::<f32>().sqrt();
        let delta = (next_norm - curr_norm).abs();
        let predicted_reward = (1.0f32 - delta).clamp(-1.0, 1.0);
        let predicted_done = delta > 2.0;

        Ok(DynamicTransitionStep {
            next_latent,
            predicted_reward,
            predicted_done,
        })
    }

    /// Rollout multi-step transitions with rewards and termination flags across horizon.
    pub fn rollout_transition_horizon(
        &self,
        initial_state: &[f32],
        actions: Option<&[Vec<f32>]>,
        steps: usize,
    ) -> Result<Vec<DynamicTransitionStep>, DynamicsError> {
        if steps == 0 {
            return Err(DynamicsError::ZeroRolloutSteps);
        }

        let mut trajectory = Vec::with_capacity(steps);
        let mut curr = initial_state.to_vec();

        for s in 0..steps {
            let act = actions.and_then(|a| a.get(s).map(|v| v.as_slice()));
            let step_res = self.predict_transition_step(&curr, act)?;
            curr = step_res.next_latent.clone();
            let is_done = step_res.predicted_done;
            trajectory.push(step_res);
            if is_done {
                break;
            }
        }

        Ok(trajectory)
    }

    /// Spatially pool multi-patch token embeddings of a single frame into a compact state vector.
    /// Input: [num_patches * hidden_size] -> Output: [hidden_size]
    pub fn pool_frame_patches(
        frame_tokens: &[f32],
        num_patches: usize,
        hidden_size: usize,
    ) -> Result<Vec<f32>, DynamicsError> {
        if frame_tokens.is_empty() || num_patches == 0 || hidden_size == 0 {
            return Err(DynamicsError::EmptyState);
        }
        if frame_tokens.len() != num_patches * hidden_size {
            return Err(DynamicsError::DimensionMismatch {
                expected: num_patches * hidden_size,
                actual: frame_tokens.len(),
            });
        }

        let mut pooled = vec![0.0f32; hidden_size];
        let patch_inv = 1.0f32 / num_patches as f32;

        for p in 0..num_patches {
            let offset = p * hidden_size;
            for d in 0..hidden_size {
                pooled[d] += frame_tokens[offset + d] * patch_inv;
            }
        }

        Ok(pooled)
    }

    /// Predicts future latent trajectories directly from an observed RawVideo sequence.
    pub fn predict_from_video(
        &self,
        encoder: &VisionEncoder,
        video: &RawVideo,
        future_actions: Option<&[Vec<f32>]>,
        future_steps: usize,
    ) -> Result<Vec<Vec<f32>>, DynamicsError> {
        if video.frames.is_empty() {
            return Err(DynamicsError::EmptyState);
        }
        if future_steps == 0 {
            return Err(DynamicsError::ZeroRolloutSteps);
        }

        // Encode latest frame in video sequence
        let last_frame = &video.frames[video.frames.len() - 1];
        let (projected_patches, num_patches) = encoder.encode_to_embeddings(last_frame)?;

        // Pool into compact latent representation
        let latent_state = Self::pool_frame_patches(&projected_patches, num_patches, encoder.config.hidden_size)?;

        // Project or pad if latent_dim differs from encoder hidden_size
        let normalized_state = if latent_state.len() == self.config.latent_dim {
            latent_state
        } else if latent_state.len() > self.config.latent_dim {
            latent_state[..self.config.latent_dim].to_vec()
        } else {
            let mut padded = latent_state;
            padded.resize(self.config.latent_dim, 0.0);
            padded
        };

        self.rollout_horizon(&normalized_state, future_actions, future_steps)
    }

    /// Evaluates transition fidelity between predicted state and target ground-truth state.
    /// Returns cosine similarity score in [-1.0, 1.0] and Mean Squared Error.
    pub fn evaluate_transition_fidelity(
        predicted: &[f32],
        target: &[f32],
    ) -> Result<(f32, f32), DynamicsError> {
        if predicted.is_empty() || target.is_empty() {
            return Err(DynamicsError::EmptyState);
        }
        if predicted.len() != target.len() {
            return Err(DynamicsError::DimensionMismatch {
                expected: predicted.len(),
                actual: target.len(),
            });
        }

        let dim = predicted.len();
        let mut dot = 0.0f32;
        let mut norm_p = 0.0f32;
        let mut norm_t = 0.0f32;
        let mut mse = 0.0f32;

        for i in 0..dim {
            let p = predicted[i];
            let t = target[i];
            dot += p * t;
            norm_p += p * p;
            norm_t += t * t;
            let diff = p - t;
            mse += diff * diff;
        }

        mse /= dim as f32;
        let denom = (norm_p.sqrt() * norm_t.sqrt()).max(1e-8);
        let cos_sim = (dot / denom).clamp(-1.0, 1.0);

        Ok((cos_sim, mse))
    }

    /// Export predictor parameters as named official safetensors entries.
    /// Named tensors integrate directly with sharded world model checkpointing:
    /// - `world_model.dynamics.w_proj`: projection weights [intermediate_dim, latent_dim + action_dim]
    /// - `world_model.dynamics.b_proj`: projection bias [intermediate_dim]
    /// - `world_model.dynamics.w_gate`: update gate weights [latent_dim, intermediate_dim]
    /// - `world_model.dynamics.b_gate`: update gate bias [latent_dim]
    /// - `world_model.dynamics.w_cand`: candidate transformation weights [latent_dim, intermediate_dim]
    /// - `world_model.dynamics.b_cand`: candidate transformation bias [latent_dim]
    pub fn export_tensors(&self) -> HashMap<String, (Vec<f32>, Vec<usize>)> {
        let mut map = HashMap::new();
        let in_dim = self.config.latent_dim + self.config.action_dim;

        map.insert(
            "world_model.dynamics.w_proj".to_string(),
            (self.w_proj.clone(), vec![self.config.intermediate_dim, in_dim]),
        );
        map.insert(
            "world_model.dynamics.b_proj".to_string(),
            (self.b_proj.clone(), vec![self.config.intermediate_dim]),
        );
        map.insert(
            "world_model.dynamics.w_gate".to_string(),
            (self.w_gate.clone(), vec![self.config.latent_dim, self.config.intermediate_dim]),
        );
        map.insert(
            "world_model.dynamics.b_gate".to_string(),
            (self.b_gate.clone(), vec![self.config.latent_dim]),
        );
        map.insert(
            "world_model.dynamics.w_cand".to_string(),
            (self.w_cand.clone(), vec![self.config.latent_dim, self.config.intermediate_dim]),
        );
        map.insert(
            "world_model.dynamics.b_cand".to_string(),
            (self.b_cand.clone(), vec![self.config.latent_dim]),
        );

        map
    }

    /// Load predictor parameters from a named tensors map.
    pub fn load_from_tensors(&mut self, weights: &HashMap<String, Vec<f32>>) -> Result<(), DynamicsError> {
        let in_dim = self.config.latent_dim + self.config.action_dim;
        let expected_proj_len = self.config.intermediate_dim * in_dim;
        let expected_gate_len = self.config.latent_dim * self.config.intermediate_dim;

        if let Some(w) = weights.get("world_model.dynamics.w_proj") {
            if w.len() != expected_proj_len {
                return Err(DynamicsError::DimensionMismatch {
                    expected: expected_proj_len,
                    actual: w.len(),
                });
            }
            self.w_proj = w.clone();
        }

        if let Some(b) = weights.get("world_model.dynamics.b_proj") {
            if b.len() != self.config.intermediate_dim {
                return Err(DynamicsError::DimensionMismatch {
                    expected: self.config.intermediate_dim,
                    actual: b.len(),
                });
            }
            self.b_proj = b.clone();
        }

        if let Some(w) = weights.get("world_model.dynamics.w_gate") {
            if w.len() != expected_gate_len {
                return Err(DynamicsError::DimensionMismatch {
                    expected: expected_gate_len,
                    actual: w.len(),
                });
            }
            self.w_gate = w.clone();
        }

        if let Some(b) = weights.get("world_model.dynamics.b_gate") {
            if b.len() != self.config.latent_dim {
                return Err(DynamicsError::DimensionMismatch {
                    expected: self.config.latent_dim,
                    actual: b.len(),
                });
            }
            self.b_gate = b.clone();
        }

        if let Some(w) = weights.get("world_model.dynamics.w_cand") {
            if w.len() != expected_gate_len {
                return Err(DynamicsError::DimensionMismatch {
                    expected: expected_gate_len,
                    actual: w.len(),
                });
            }
            self.w_cand = w.clone();
        }

        if let Some(b) = weights.get("world_model.dynamics.b_cand") {
            if b.len() != self.config.latent_dim {
                return Err(DynamicsError::DimensionMismatch {
                    expected: self.config.latent_dim,
                    actual: b.len(),
                });
            }
            self.b_cand = b.clone();
        }

        Ok(())
    }

    /// Save predictor parameters to a standalone safetensors file.
    pub fn save_safetensors<P: AsRef<Path>>(&self, path: P) -> Result<(), DynamicsError> {
        let exported = self.export_tensors();
        let mut tensors = HashMap::with_capacity(exported.len());
        let mut shapes = HashMap::with_capacity(exported.len());

        for (k, (t, s)) in exported {
            tensors.insert(k.clone(), t);
            shapes.insert(k, s);
        }

        let path_str = path.as_ref().to_string_lossy();
        crate::safetensors::write_safetensors_with_shapes(&tensors, &shapes, &path_str)
            .map_err(|e| DynamicsError::SafeTensors(e.to_string()))?;
        Ok(())
    }

    /// Load predictor parameters dynamically from model directory or safetensors file.
    pub fn load_from_model_dir<P: AsRef<Path>>(
        config: TemporalDynamicsConfig,
        dir: P,
    ) -> Result<Self, DynamicsError> {
        let mut predictor = Self::new(config);
        let dir_path = dir.as_ref();

        if dir_path.is_file() {
            let path_str = dir_path.to_string_lossy();
            if let Ok(weights) = crate::safetensors::load_safetensors(&path_str) {
                predictor.load_from_tensors(&weights)?;
            }
        } else if dir_path.is_dir() {
            let direct_file = dir_path.join("world_model_dynamics.safetensors");
            if direct_file.exists() {
                let file_str = direct_file.to_string_lossy();
                if let Ok(weights) = crate::safetensors::load_safetensors(&file_str) {
                    predictor.load_from_tensors(&weights)?;
                    return Ok(predictor);
                }
            }
            let dir_str = dir_path.to_string_lossy();
            if let Ok((weights, _)) = crate::safetensors::load_model_weights_with_shapes(&dir_str) {
                predictor.load_from_tensors(&weights)?;
            }
        }

        Ok(predictor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_dynamics_step_and_rollout() {
        let config = TemporalDynamicsConfig::new(16, 4, 32, 2);
        let predictor = TemporalDynamicsPredictor::new(config.clone());

        let initial_state = vec![0.5f32; 16];
        let action = vec![0.1f32; 4];

        let next_state = predictor
            .predict_next_step(&initial_state, Some(&action))
            .expect("next step prediction succeeds");
        assert_eq!(next_state.len(), 16);
        assert!(!next_state.iter().any(|v| v.is_nan() || v.is_infinite()));

        let actions = vec![vec![0.1f32; 4], vec![-0.1f32; 4], vec![0.2f32; 4]];
        let trajectory = predictor
            .rollout_horizon(&initial_state, Some(&actions), 3)
            .expect("rollout horizon succeeds");
        assert_eq!(trajectory.len(), 3);
        assert_eq!(trajectory[0].len(), 16);
        assert_eq!(trajectory[2].len(), 16);

        let (sim, mse) = TemporalDynamicsPredictor::evaluate_transition_fidelity(&next_state, &trajectory[0])
            .expect("fidelity check succeeds");
        assert_eq!(sim, 1.0);
        assert_eq!(mse, 0.0);

        // Test tensor export and reloading round-trip
        let exported = predictor.export_tensors();
        assert_eq!(exported.len(), 6);
        assert!(exported.contains_key("world_model.dynamics.w_proj"));
        assert!(exported.contains_key("world_model.dynamics.b_proj"));
        assert!(exported.contains_key("world_model.dynamics.w_gate"));
        assert!(exported.contains_key("world_model.dynamics.b_gate"));
        assert!(exported.contains_key("world_model.dynamics.w_cand"));
        assert!(exported.contains_key("world_model.dynamics.b_cand"));

        let mut reconstructed = TemporalDynamicsPredictor::new(config);
        let weights_only: HashMap<String, Vec<f32>> = exported
            .into_iter()
            .map(|(k, (v, _))| (k, v))
            .collect();
        reconstructed
            .load_from_tensors(&weights_only)
            .expect("load from tensors succeeds");

        let reloaded_next = reconstructed
            .predict_next_step(&initial_state, Some(&action))
            .expect("prediction with reloaded tensors succeeds");
        assert_eq!(next_state, reloaded_next);
    }
}
