//! Native Neural World Model with Transformer Latent Dynamics, Value & Viability Prediction,
//! Counterfactual Rollouts, and Online Surprise Learning.
//!
//! Replaces toy MLP text approximations with continuous latent state-space dynamics:
//! 1. State & Action Embedding Fusion with RMSNorm (aligned with 1B model hidden_size=2048)
//! 2. Transformer Latent Dynamics Blocks (GQA Pre-LN Self-Attention & SwiGLU/GELU Feedforward)
//! 3. Continuous Residual Latent Transition: \hat{s}_{t+1} = s_t + \tanh(W_{next} h_L)
//! 4. Neural Reward Head: \hat{r}_t \in [-1.0, 1.0]
//! 5. Neural Viability & Failure Probability Head: p_{fail} \in [0.0, 1.0]
//! 6. Expected Value Function Head: V(s_t)
//! 7. Multi-step Counterfactual Rollouts in Latent Space
//! 8. Online Surprise Learning via Analytical Gradient Backpropagation

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use thiserror::Error;

use crate::safetensors::{
    load_model_weights, load_safetensors, resolve_max_shard_size_bytes,
    write_safetensors_sharded_with_prefix,
};
use crate::tokenizer::TaraTokenizer;

/// Errors encountered in World Model execution.
#[derive(Debug, Error)]
pub enum WorldModelError {
    #[error("empty latent state input")]
    EmptyState,
    #[error("state dimension mismatch: expected {expected}, got {actual}")]
    StateDimensionMismatch { expected: usize, actual: usize },
    #[error("action dimension mismatch: expected {expected}, got {actual}")]
    ActionDimensionMismatch { expected: usize, actual: usize },
    #[error("zero rollout steps requested")]
    ZeroRolloutSteps,
    #[error("world model checkpoint weights are missing")]
    MissingWeights,
    #[error("geometric invariant violation: head_dim ({head_dim}) * num_heads ({num_heads}) != hidden_size ({hidden_size})")]
    GeometryInvariantViolation { head_dim: usize, num_heads: usize, hidden_size: usize },
    #[error("safetensors error: {0}")]
    SafeTensors(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// Dynamic hyperparameter configuration for the Neural World Model.
/// Perfectly suited for the TARA 1B model family (hidden_size=2048, GQA=32/8, RoPE=500000.0).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldModelConfig {
    #[serde(alias = "state_dim")]
    pub latent_dim: usize,
    pub action_dim: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub initializer_range: f32,
    pub learning_rate: f32,
    pub discount_factor: f32,
    pub horizon_limit: usize,
}

impl Default for WorldModelConfig {
    /// Canonical default configuration matching the TARA 1B architecture.
    fn default() -> Self {
        Self {
            latent_dim: 2048,       // 1:1 match with 1B model residual stream
            action_dim: 512,        // Multi-modal/action latent projection dimension
            hidden_size: 2048,      // 1B backbone hidden dimension
            intermediate_size: 5632,// 1B SwiGLU expansion dimension
            num_hidden_layers: 8,   // Dedicated Transformer dynamics depth
            num_attention_heads: 32,// 1B attention heads (head_dim = 64)
            num_key_value_heads: 8, // 1B GQA KV heads (4:1 ratio)
            head_dim: 64,           // 1B canonical head dimension
            rms_norm_eps: 1e-5,     // 1B RMSNorm epsilon
            rope_theta: 500000.0,   // 1B RoPE base frequency
            initializer_range: 0.02,
            learning_rate: 0.0003,
            discount_factor: 0.99,
            horizon_limit: 64,
        }
    }
}

impl WorldModelConfig {
    /// Construct dynamic configuration matching specified state, action, and hidden dimensions,
    /// enforcing strict Attention Tensor Geometry invariants (`head_dim * num_attention_heads == hidden_size`).
    pub fn new(latent_dim: usize, action_dim: usize, hidden_size: usize) -> Self {
        let head_dim_candidate = 64.min(hidden_size).max(1);
        let num_heads = (hidden_size / head_dim_candidate).max(1);
        let head_dim = hidden_size / num_heads;
        let num_kv_heads = (num_heads / 4).max(1);
        let inter_size = (((hidden_size * 8 / 3) + 255) / 256) * 256;

        Self {
            latent_dim,
            action_dim,
            hidden_size,
            intermediate_size: inter_size,
            num_hidden_layers: 8,
            num_attention_heads: num_heads,
            num_key_value_heads: num_kv_heads,
            head_dim,
            rms_norm_eps: 1e-5,
            rope_theta: 500000.0,
            initializer_range: 0.02,
            learning_rate: 0.0003,
            discount_factor: 0.99,
            horizon_limit: 64,
        }
    }

    /// Dynamically load and construct configuration from a base TARA model `config.json`.
    pub fn from_base_config_file<P: AsRef<Path>>(config_path: P) -> Result<Self, WorldModelError> {
        let content = fs::read_to_string(config_path.as_ref())?;
        let v: serde_json::Value = serde_json::from_str(&content)?;

        let hidden_size = v.get("hidden_size").and_then(|x| x.as_u64()).ok_or_else(|| {
            WorldModelError::Serialization(serde::de::Error::custom("missing required config field: hidden_size"))
        })? as usize;

        let num_attention_heads = v.get("num_attention_heads").and_then(|x| x.as_u64()).unwrap_or_else(|| {
            (hidden_size / 64).max(1) as u64
        }) as usize;

        let head_dim = v.get("head_dim").and_then(|x| x.as_u64()).unwrap_or_else(|| {
            (hidden_size / num_attention_heads).max(1) as u64
        }) as usize;

        let num_key_value_heads = v.get("num_key_value_heads").and_then(|x| x.as_u64()).unwrap_or_else(|| {
            (num_attention_heads / 4).max(1) as u64
        }) as usize;

        let intermediate_size = v.get("intermediate_size").and_then(|x| x.as_u64()).unwrap_or_else(|| {
            let raw = (hidden_size * 8) / 3;
            (((raw + 255) / 256) * 256) as u64
        }) as usize;

        let num_hidden_layers = v.get("world_model_layers")
            .or_else(|| v.get("num_hidden_layers"))
            .and_then(|x| x.as_u64())
            .map(|l| (l / 2).max(1))
            .unwrap_or(8) as usize;

        let latent_dim = v.get("latent_dim").and_then(|x| x.as_u64()).unwrap_or(hidden_size as u64) as usize;
        let action_dim = v.get("action_dim").and_then(|x| x.as_u64()).unwrap_or(((hidden_size / 4).max(64)) as u64) as usize;

        let rms_norm_eps = v.get("rms_norm_eps").and_then(|x| x.as_f64()).unwrap_or(1e-5) as f32;
        let rope_theta = v.get("rope_theta").and_then(|x| x.as_f64()).unwrap_or(500000.0) as f32;
        let initializer_range = v.get("initializer_range").and_then(|x| x.as_f64()).unwrap_or(0.02) as f32;
        let learning_rate = v.get("learning_rate").and_then(|x| x.as_f64()).unwrap_or(0.0003) as f32;
        let discount_factor = v.get("discount_factor").and_then(|x| x.as_f64()).unwrap_or(0.99) as f32;
        let horizon_limit = v.get("horizon_limit").and_then(|x| x.as_u64()).unwrap_or(64) as usize;

        let config = Self {
            latent_dim,
            action_dim,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rms_norm_eps,
            rope_theta,
            initializer_range,
            learning_rate,
            discount_factor,
            horizon_limit,
        };

        config.validate()?;
        Ok(config)
    }

    /// Strict geometric invariant verification according to project rules.
    pub fn validate(&self) -> Result<(), WorldModelError> {
        if self.head_dim * self.num_attention_heads != self.hidden_size {
            return Err(WorldModelError::GeometryInvariantViolation {
                head_dim: self.head_dim,
                num_heads: self.num_attention_heads,
                hidden_size: self.hidden_size,
            });
        }
        Ok(())
    }

    pub fn state_dim(&self) -> usize {
        self.latent_dim
    }
}

/// Individual transition step with predicted latent dynamics, reward, and viability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicTransition {
    pub next_latent: Vec<f32>,
    pub predicted_reward: f32,
    pub failure_probability: f32,
    pub expected_value: f32,
    pub viable: bool,
    pub confidence: f32,
}

/// Multi-step counterfactual mental rollout trajectory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatentTrajectory {
    pub transitions: Vec<DynamicTransition>,
    pub cumulative_reward: f32,
    pub overall_viable: bool,
    pub min_confidence: f32,
}

fn default_lora_rank() -> usize {
    8
}

fn default_lora_scale() -> f32 {
    2.0
}

/// Individual Transformer layer weights for latent state-space transition dynamics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicsLayer {
    pub input_layernorm: Vec<f32>,
    pub q_proj: Vec<f32>,
    pub k_proj: Vec<f32>,
    pub v_proj: Vec<f32>,
    pub o_proj: Vec<f32>,
    pub post_attention_layernorm: Vec<f32>,
    pub gate_proj: Vec<f32>,
    pub up_proj: Vec<f32>,
    pub down_proj: Vec<f32>,
    // Native LoRA for autonomous continuous backbone adaptation (Rule 15):
    #[serde(default = "default_lora_rank")]
    pub lora_rank: usize,
    #[serde(default = "default_lora_scale")]
    pub lora_scale: f32,
    #[serde(default)]
    pub lora_q_a: Vec<f32>,
    #[serde(default)]
    pub lora_q_b: Vec<f32>,
    #[serde(default)]
    pub lora_v_a: Vec<f32>,
    #[serde(default)]
    pub lora_v_b: Vec<f32>,
}

/// Native Neural World Model with Transformer-based Latent Dynamics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldModel {
    pub config: WorldModelConfig,

    // 1. Input state-action projection and norm: [hidden_size, latent_dim] & [hidden_size, action_dim]
    pub w_state: Vec<f32>,
    pub w_action: Vec<f32>,
    pub input_norm: Vec<f32>,

    // 2. Transformer Latent Dynamics Backbone
    pub layers: Vec<DynamicsLayer>,

    // 3. Final backbone normalization
    pub final_norm: Vec<f32>,

    // 4. Residual continuous latent transition head: [latent_dim, hidden_size]
    pub w_next_state: Vec<f32>,

    // 5. Neural reward head: [1, hidden_size]
    pub w_reward: Vec<f32>,
    pub b_reward: f32,

    // 6. Viability / Failure probability head: [1, hidden_size]
    pub w_fail: Vec<f32>,
    pub b_fail: f32,

    // 7. Value function head: [1, hidden_size]
    pub w_value: Vec<f32>,
    pub b_value: f32,
}

/// Generates deterministic variance-scaled Gaussian/Xavier initial weights.
fn variance_scaled_gaussian(out_dim: usize, in_dim: usize, seed_offset: usize) -> Vec<f32> {
    let scale = (2.0f32 / (in_dim + out_dim) as f32).sqrt();
    let total = out_dim * in_dim;
    let mut vec = Vec::with_capacity(total);
    let phi = 0.6180339887f32;

    for i in 0..total {
        let u1 = (((i + seed_offset + 1) as f32 * phi).fract()).clamp(1e-6, 1.0 - 1e-6);
        let u2 = (((i + seed_offset + 1) as f32 * 0.41421356f32).fract()).clamp(1e-6, 1.0 - 1e-6);
        let normal = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        vec.push(normal * scale);
    }
    vec
}

impl WorldModel {
    /// Construct a new World Model with variance-scaled Gaussian / Xavier initialization.
    pub fn new(config: WorldModelConfig) -> Self {
        assert!(config.latent_dim > 0, "latent_dim must be positive");
        assert!(config.action_dim > 0, "action_dim must be positive");
        assert!(config.hidden_size > 0, "hidden_size must be positive");
        config.validate().expect("WorldModelConfig must satisfy geometry invariants");

        let h = config.hidden_size;
        let s = config.latent_dim;
        let a = config.action_dim;
        let inter = config.intermediate_size;
        let q_dim = config.num_attention_heads * config.head_dim;
        let kv_dim = config.num_key_value_heads * config.head_dim;

        let w_state = variance_scaled_gaussian(h, s, 101);
        let w_action = variance_scaled_gaussian(h, a, 203);
        let input_norm = vec![1.0f32; h];

        let r = 8;
        let lora_scale = 2.0f32; // alpha 16.0 / r 8.0
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for l in 0..config.num_hidden_layers {
            let offset = (l + 1) * 1000;
            let layer = DynamicsLayer {
                input_layernorm: vec![1.0f32; h],
                q_proj: variance_scaled_gaussian(q_dim, h, offset + 1),
                k_proj: variance_scaled_gaussian(kv_dim, h, offset + 2),
                v_proj: variance_scaled_gaussian(kv_dim, h, offset + 3),
                o_proj: variance_scaled_gaussian(h, q_dim, offset + 4),
                post_attention_layernorm: vec![1.0f32; h],
                gate_proj: variance_scaled_gaussian(inter, h, offset + 5),
                up_proj: variance_scaled_gaussian(inter, h, offset + 6),
                down_proj: variance_scaled_gaussian(h, inter, offset + 7),
                lora_rank: r,
                lora_scale,
                lora_q_a: variance_scaled_gaussian(r, h, offset + 11),
                lora_q_b: vec![0.0f32; q_dim * r],
                lora_v_a: variance_scaled_gaussian(r, h, offset + 12),
                lora_v_b: vec![0.0f32; kv_dim * r],
            };
            layers.push(layer);
        }

        let final_norm = vec![1.0f32; h];
        let w_next_state = variance_scaled_gaussian(s, h, 8001);
        let w_reward = variance_scaled_gaussian(1, h, 8101);
        let w_fail = variance_scaled_gaussian(1, h, 8201);
        let w_value = variance_scaled_gaussian(1, h, 8301);

        Self {
            config,
            w_state,
            w_action,
            input_norm,
            layers,
            final_norm,
            w_next_state,
            w_reward,
            b_reward: 0.0,
            w_fail,
            b_fail: -0.5, // Prior bias towards viability
            w_value,
            b_value: 0.0,
        }
    }

    /// Forward RMSNorm on vector x with weights gamma.
    fn rms_norm(x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
        let n = x.len();
        let sum_sq: f32 = x.iter().map(|&v| v * v).sum();
        let rms = (sum_sq / (n as f32) + eps).sqrt();
        let inv_rms = 1.0 / rms;
        let mut out = vec![0.0f32; n];
        for i in 0..n {
            out[i] = x[i] * inv_rms * gamma[i];
        }
        out
    }

    /// GELU activation function.
    fn gelu(x: f32) -> f32 {
        let sqrt_2_over_pi = 0.79788456f32;
        let gelu_coeff = 0.044715f32;
        0.5 * x * (1.0 + (sqrt_2_over_pi * (x + gelu_coeff * x.powi(3))).tanh())
    }

    /// Matrix-vector multiplication: y = W * x, where W is row-major [out_dim, in_dim].
    fn matvec(w: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; out_dim];
        for i in 0..out_dim {
            let row_offset = i * in_dim;
            let mut sum = 0.0f32;
            for j in 0..in_dim {
                sum += w[row_offset + j] * x[j];
            }
            out[i] = sum;
        }
        out
    }

    /// Apply LoRA low-rank delta adaptation: y += scale * (x * A^T) * B^T
    fn apply_lora(
        a: &[f32],
        b: &[f32],
        scale: f32,
        r: usize,
        in_dim: usize,
        out_dim: usize,
        x: &[f32],
        out: &mut [f32],
    ) {
        if r == 0 || a.len() != r * in_dim || b.len() != out_dim * r {
            return;
        }
        let mut za = vec![0.0f32; r];
        for k in 0..r {
            let a_row = &a[k * in_dim..(k + 1) * in_dim];
            let mut dot = 0.0f32;
            for d in 0..in_dim {
                dot += x[d] * a_row[d];
            }
            za[k] = dot;
        }
        for o in 0..out_dim {
            let b_row = &b[o * r..(o + 1) * r];
            let mut dot = 0.0f32;
            for k in 0..r {
                dot += za[k] * b_row[k];
            }
            out[o] += scale * dot;
        }
    }

    /// Predict the next latent transition, expected reward, and viability given (s_t, a_t).
    pub fn predict_step(
        &self,
        current_state: &[f32],
        action: &[f32],
    ) -> Result<DynamicTransition, WorldModelError> {
        self.predict_step_with_hidden(current_state, action)
            .map(|(t, _, _)| t)
    }

    /// Predict the next latent transition along with the final contextualized state and action hidden activations.
    pub fn predict_step_with_hidden(
        &self,
        current_state: &[f32],
        action: &[f32],
    ) -> Result<(DynamicTransition, Vec<f32>, Vec<f32>), WorldModelError> {
        if current_state.is_empty() {
            return Err(WorldModelError::EmptyState);
        }
        if current_state.len() != self.config.latent_dim {
            return Err(WorldModelError::StateDimensionMismatch {
                expected: self.config.latent_dim,
                actual: current_state.len(),
            });
        }
        if action.len() != self.config.action_dim {
            return Err(WorldModelError::ActionDimensionMismatch {
                expected: self.config.action_dim,
                actual: action.len(),
            });
        }

        let h_dim = self.config.hidden_size;
        let eps = self.config.rms_norm_eps;

        // 1. Project state and action into dual latent sequence tokens [z_state, u_action] (T=2)
        let s_proj = Self::matvec(&self.w_state, current_state, h_dim, self.config.latent_dim);
        let a_proj = Self::matvec(&self.w_action, action, h_dim, self.config.action_dim);

        let tok_state = Self::rms_norm(&s_proj, &self.input_norm, eps);
        let tok_action = Self::rms_norm(&a_proj, &self.input_norm, eps);
        let mut tokens = [tok_state, tok_action];

        // 2. Transformer Latent Dynamics Blocks (Bipartite 2x2 Self-Attention & SwiGLU FFN)
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let inter_dim = self.config.intermediate_size;
        let gqa_ratio = (num_heads / num_kv_heads).max(1);
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        for layer in &self.layers {
            // Pre-norm 1 for both tokens
            let norm_0 = Self::rms_norm(&tokens[0], &layer.input_layernorm, eps);
            let norm_1 = Self::rms_norm(&tokens[1], &layer.input_layernorm, eps);

            // Q, K, V projections for both tokens
            let mut q0 = Self::matvec(&layer.q_proj, &norm_0, q_dim, h_dim);
            let mut q1 = Self::matvec(&layer.q_proj, &norm_1, q_dim, h_dim);
            let k0 = Self::matvec(&layer.k_proj, &norm_0, kv_dim, h_dim);
            let k1 = Self::matvec(&layer.k_proj, &norm_1, kv_dim, h_dim);
            let mut v0 = Self::matvec(&layer.v_proj, &norm_0, kv_dim, h_dim);
            let mut v1 = Self::matvec(&layer.v_proj, &norm_1, kv_dim, h_dim);

            // Apply LoRA low-rank delta adaptation: ΔW = (scale) * B * A
            Self::apply_lora(&layer.lora_q_a, &layer.lora_q_b, layer.lora_scale, layer.lora_rank, h_dim, q_dim, &norm_0, &mut q0);
            Self::apply_lora(&layer.lora_q_a, &layer.lora_q_b, layer.lora_scale, layer.lora_rank, h_dim, q_dim, &norm_1, &mut q1);
            Self::apply_lora(&layer.lora_v_a, &layer.lora_v_b, layer.lora_scale, layer.lora_rank, h_dim, kv_dim, &norm_0, &mut v0);
            Self::apply_lora(&layer.lora_v_a, &layer.lora_v_b, layer.lora_scale, layer.lora_rank, h_dim, kv_dim, &norm_1, &mut v1);

            let mut attn_out0 = vec![0.0f32; q_dim];
            let mut attn_out1 = vec![0.0f32; q_dim];

            // Genuine Multi-Head 2x2 Cross/Self-Attention over [State, Action] tokens
            for head_idx in 0..num_heads {
                let kv_idx = head_idx / gqa_ratio;
                let q_start = head_idx * head_dim;
                let kv_start = kv_idx * head_dim;

                // Scaled Dot-Product attention scores across tokens 0 and 1
                let mut dot_00 = 0.0f32;
                let mut dot_01 = 0.0f32;
                let mut dot_10 = 0.0f32;
                let mut dot_11 = 0.0f32;

                for d in 0..head_dim {
                    let q0_d = q0[q_start + d];
                    let q1_d = q1[q_start + d];
                    let k0_d = k0[kv_start + d];
                    let k1_d = k1[kv_start + d];

                    dot_00 += q0_d * k0_d;
                    dot_01 += q0_d * k1_d;
                    dot_10 += q1_d * k0_d;
                    dot_11 += q1_d * k1_d;
                }

                let s00 = dot_00 * scale;
                let s01 = dot_01 * scale;
                let s10 = dot_10 * scale;
                let s11 = dot_11 * scale;

                // Numerically stable Softmax for token 0 queries
                let max0 = s00.max(s01);
                let exp00 = (s00 - max0).exp();
                let exp01 = (s01 - max0).exp();
                let sum0 = exp00 + exp01;
                let a00 = exp00 / sum0;
                let a01 = exp01 / sum0;

                // Numerically stable Softmax for token 1 queries
                let max1 = s10.max(s11);
                let exp10 = (s10 - max1).exp();
                let exp11 = (s11 - max1).exp();
                let sum1 = exp10 + exp11;
                let a10 = exp10 / sum1;
                let a11 = exp11 / sum1;

                // Weighted value accumulation
                for d in 0..head_dim {
                    let v0_d = v0[kv_start + d];
                    let v1_d = v1[kv_start + d];

                    attn_out0[q_start + d] = a00 * v0_d + a01 * v1_d;
                    attn_out1[q_start + d] = a10 * v0_d + a11 * v1_d;
                }
            }

            // Output projection and residual connection for both tokens
            let o0 = Self::matvec(&layer.o_proj, &attn_out0, h_dim, q_dim);
            let o1 = Self::matvec(&layer.o_proj, &attn_out1, h_dim, q_dim);
            for i in 0..h_dim {
                tokens[0][i] += o0[i];
                tokens[1][i] += o1[i];
            }

            // Pre-norm 2 and SwiGLU / GELU Feed-Forward Network for both tokens
            for tok_idx in 0..2 {
                let norm_h2 = Self::rms_norm(&tokens[tok_idx], &layer.post_attention_layernorm, eps);
                let gate = Self::matvec(&layer.gate_proj, &norm_h2, inter_dim, h_dim);
                let up = Self::matvec(&layer.up_proj, &norm_h2, inter_dim, h_dim);

                let mut ffn_act = vec![0.0f32; inter_dim];
                for i in 0..inter_dim {
                    ffn_act[i] = Self::gelu(gate[i]) * up[i];
                }

                let down = Self::matvec(&layer.down_proj, &ffn_act, h_dim, inter_dim);
                for i in 0..h_dim {
                    tokens[tok_idx][i] += down[i];
                }
            }
        }

        // 3. Final RMSNorm for both State and Action tokens
        let h_state = Self::rms_norm(&tokens[0], &self.final_norm, eps);
        let h_action = Self::rms_norm(&tokens[1], &self.final_norm, eps);

        // 4. Continuous Residual Latent Transition from contextualized State Token:
        // \hat{s}_{t+1} = s_t + tanh(W_next * h_state)
        let s_dim = self.config.latent_dim;
        let delta_s = Self::matvec(&self.w_next_state, &h_state, s_dim, h_dim);
        let mut next_latent = Vec::with_capacity(s_dim);
        for i in 0..s_dim {
            next_latent.push(current_state[i] + delta_s[i].tanh());
        }

        // 5. Reward Prediction Head from contextualized Action Token: r = tanh(W_r * h_action + b_r)
        let mut r_sum = self.b_reward;
        for i in 0..h_dim {
            r_sum += self.w_reward[i] * h_action[i];
        }
        let predicted_reward = r_sum.tanh();

        // 6. Failure Probability Head from contextualized Action Token: p_fail = sigmoid(W_fail * h_action + b_fail)
        let mut f_sum = self.b_fail;
        for i in 0..h_dim {
            f_sum += self.w_fail[i] * h_action[i];
        }
        let failure_probability = 1.0 / (1.0 + (-f_sum).exp());

        // 7. Expected Value Head from State Token: V = W_v * h_state + b_v
        let mut v_sum = self.b_value;
        for i in 0..h_dim {
            v_sum += self.w_value[i] * h_state[i];
        }
        let expected_value = v_sum;

        // 8. Viability & Confidence Score
        let viable = failure_probability < 0.5;
        let target_norm = (s_dim as f32).sqrt();
        let cur_norm = next_latent.iter().map(|&v| v * v).sum::<f32>().sqrt();
        let confidence = (1.0 - (cur_norm - target_norm).abs() / target_norm).clamp(0.1, 0.99);

        Ok((
            DynamicTransition {
                next_latent,
                predicted_reward,
                failure_probability,
                expected_value,
                viable,
                confidence,
            },
            h_state,
            h_action,
        ))
    }

    /// Rollout multi-step counterfactual trajectory in latent space without text decoding.
    pub fn rollout(
        &self,
        initial_state: &[f32],
        actions: &[Vec<f32>],
    ) -> Result<LatentTrajectory, WorldModelError> {
        if actions.is_empty() {
            return Err(WorldModelError::ZeroRolloutSteps);
        }

        let mut transitions = Vec::with_capacity(actions.len());
        let mut curr = initial_state.to_vec();
        let mut cumulative_reward = 0.0f32;
        let mut overall_viable = true;
        let mut min_confidence = 1.0f32;
        let gamma = self.config.discount_factor;

        for (step_idx, action) in actions.iter().enumerate() {
            let step_res = self.predict_step(&curr, action)?;
            let discount = gamma.powi(step_idx as i32);
            cumulative_reward += discount * step_res.predicted_reward;
            if !step_res.viable {
                overall_viable = false;
            }
            if step_res.confidence < min_confidence {
                min_confidence = step_res.confidence;
            }

            curr = step_res.next_latent.clone();
            transitions.push(step_res);
        }

        Ok(LatentTrajectory {
            transitions,
            cumulative_reward,
            overall_viable,
            min_confidence,
        })
    }

    /// Bridges the base neural model's continuous hidden activations directly into the World Model latent space.
    /// Resamples or maps the input hidden vector to the World Model's configured `latent_dim` and normalizes.
    pub fn encode_state_from_hidden(&self, hidden_state: &[f32]) -> Vec<f32> {
        let target_dim = self.config.latent_dim;
        if hidden_state.is_empty() {
            return vec![0.0f32; target_dim];
        }

        let mut out = vec![0.0f32; target_dim];
        let in_len = hidden_state.len();

        if in_len == target_dim {
            out.copy_from_slice(hidden_state);
        } else {
            for i in 0..target_dim {
                let pos = (i as f32 / target_dim as f32) * (in_len as f32);
                let idx0 = (pos.floor() as usize).min(in_len - 1);
                let idx1 = (idx0 + 1).min(in_len - 1);
                let frac = pos - pos.floor();
                out[i] = hidden_state[idx0] * (1.0 - frac) + hidden_state[idx1] * frac;
            }
        }

        let norm: f32 = out.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 1e-6 {
            let inv = 1.0 / norm;
            for v in out.iter_mut() {
                *v *= inv;
            }
        }
        out
    }

    /// Encodes a structured action (tool call, command name, and argument parameters) into a continuous action vector.
    pub fn encode_structured_action(
        &self,
        action_name: &str,
        params: &serde_json::Value,
        tokenizer: Option<&TaraTokenizer>,
    ) -> Vec<f32> {
        let serialized_params = params.to_string();
        let composite = format!("{}({})", action_name.trim(), serialized_params);
        self.encode_action_str(&composite, tokenizer)
    }

    /// Encodes a textual state observation into a continuous latent vector z_t.
    pub fn encode_state_str(&self, state_text: &str, tokenizer: Option<&TaraTokenizer>) -> Vec<f32> {
        let tokens = match tokenizer {
            Some(tok) => tok.encode(state_text),
            None => state_text.bytes().map(|b| b as u32).collect(),
        };
        self.project_tokens_to_latent(&tokens, self.config.latent_dim)
    }

    /// Encodes a candidate action string into an action latent vector a_t.
    pub fn encode_action_str(&self, action_text: &str, tokenizer: Option<&TaraTokenizer>) -> Vec<f32> {
        let tokens = match tokenizer {
            Some(tok) => tok.encode(action_text),
            None => action_text.bytes().map(|b| b as u32).collect(),
        };
        self.project_tokens_to_latent(&tokens, self.config.action_dim)
    }

    /// Projects discrete token sequence into continuous normalized latent representation.
    fn project_tokens_to_latent(&self, tokens: &[u32], target_dim: usize) -> Vec<f32> {
        let mut vec = vec![0.0f32; target_dim];
        if tokens.is_empty() {
            return vec;
        }

        let theta_base = self.config.rope_theta;
        for (pos, &tok) in tokens.iter().enumerate() {
            let p = pos as f32;
            let val = tok as f32;
            for i in 0..(target_dim / 2) {
                let freq = 1.0f32 / theta_base.powf((2 * i) as f32 / target_dim as f32);
                let angle = (val * 0.01 + p) * freq;
                vec[2 * i] += angle.sin();
                vec[2 * i + 1] += angle.cos();
            }
        }

        let norm: f32 = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 1e-6 {
            let inv = 1.0 / norm;
            for v in vec.iter_mut() {
                *v *= inv;
            }
        }
        vec
    }

    /// Online Surprise Learning: Minimizes Prediction Error when real outcome is observed.
    pub fn update_from_experience(
        &mut self,
        current_state: &[f32],
        action: &[f32],
        real_next_state: &[f32],
        real_reward: f32,
        real_failed: bool,
    ) -> Result<f32, WorldModelError> {
        let (transition, h_state, h_action) = self.predict_step_with_hidden(current_state, action)?;

        let d = self.config.latent_dim as f32;
        let mut state_loss = 0.0f32;
        for i in 0..self.config.latent_dim {
            let diff = transition.next_latent[i] - real_next_state[i];
            state_loss += diff * diff;
        }
        state_loss /= d;

        let reward_diff = transition.predicted_reward - real_reward;
        let reward_loss = reward_diff * reward_diff;
        let d_reward = 2.0 * reward_diff;

        let target_fail = if real_failed { 1.0f32 } else { 0.0f32 };
        let fail_diff = transition.failure_probability - target_fail;
        let fail_loss = fail_diff * fail_diff;
        let d_fail = 2.0 * fail_diff;

        let total_loss = state_loss + reward_loss + fail_loss;
        let lr = self.config.learning_rate;

        // Analytical gradient update for Reward Head using true contextualized action activation:
        // r = tanh(W_r * h_action + b_r)
        let dr_act = (1.0 - transition.predicted_reward.powi(2)) * d_reward;
        let h_dim = self.config.hidden_size;
        for i in 0..h_dim {
            let grad = (dr_act * h_action[i]).clamp(-1.0, 1.0);
            self.w_reward[i] -= lr * grad;
        }
        self.b_reward -= lr * dr_act.clamp(-1.0, 1.0);

        // Analytical gradient update for Failure Head using true contextualized action activation:
        // p = sigmoid(W_fail * h_action + b_fail)
        let p = transition.failure_probability;
        let df_act = p * (1.0 - p) * d_fail;
        for i in 0..h_dim {
            let grad = (df_act * h_action[i]).clamp(-1.0, 1.0);
            self.w_fail[i] -= lr * grad;
        }
        self.b_fail -= lr * df_act.clamp(-1.0, 1.0);

        // Analytical gradient update for Latent Transition Head using true contextualized state activation:
        // \hat{s}_{t+1} = s_t + tanh(W_next * h_state)
        let s_dim = self.config.latent_dim;
        let mut d_h_state = vec![0.0f32; h_dim];
        for i in 0..s_dim {
            let diff_s = transition.next_latent[i] - real_next_state[i];
            let delta_tanh = transition.next_latent[i] - current_state[i];
            let dt = (1.0 - delta_tanh * delta_tanh) * diff_s * 2.0 / d;
            let row_offset = i * h_dim;
            for j in 0..h_dim {
                let grad = (dt * h_state[j]).clamp(-1.0, 1.0);
                self.w_next_state[row_offset + j] -= lr * grad;
                d_h_state[j] += dt * self.w_next_state[row_offset + j];
            }
        }

        // Action representation gradient combining reward error and viability failure error:
        let mut d_h_action = vec![0.0f32; h_dim];
        for j in 0..h_dim {
            d_h_action[j] = (dr_act * self.w_reward[j] + df_act * self.w_fail[j]).clamp(-1.0, 1.0);
        }

        // Backpropagate error into the Transformer backbone layers via Native LoRA adapters (Rule 15)
        let q_dim = self.config.num_attention_heads * self.config.head_dim;
        let kv_dim = self.config.num_key_value_heads * self.config.head_dim;
        for layer in self.layers.iter_mut().rev() {
            if layer.lora_rank == 0 {
                continue;
            }
            let r = layer.lora_rank;
            let lora_lr = lr * layer.lora_scale * 0.05;

            // Gradient w.r.t B_q: dL/dB_q = d_h_state * z_a^T where z_a = h_state * A_q^T
            let mut za_q = vec![0.0f32; r];
            for k in 0..r {
                let a_row = &layer.lora_q_a[k * h_dim..(k + 1) * h_dim];
                let mut sum = 0.0f32;
                for d_i in 0..h_dim {
                    sum += h_state[d_i] * a_row[d_i];
                }
                za_q[k] = sum;
            }

            for o in 0..q_dim {
                let grad_out = d_h_state[o % h_dim];
                for k in 0..r {
                    let grad_b = (grad_out * za_q[k]).clamp(-1.0, 1.0);
                    layer.lora_q_b[o * r + k] -= lora_lr * grad_b;
                }
            }

            for k in 0..r {
                let mut d_za = 0.0f32;
                for o in 0..q_dim {
                    d_za += d_h_state[o % h_dim] * layer.lora_q_b[o * r + k];
                }
                for d_i in 0..h_dim {
                    let grad_a = (d_za * h_state[d_i]).clamp(-1.0, 1.0);
                    layer.lora_q_a[k * h_dim + d_i] -= lora_lr * grad_a;
                }
            }

            // Gradient w.r.t B_v: dL/dB_v = d_h_action * z_v^T where z_v = h_action * A_v^T
            let mut za_v = vec![0.0f32; r];
            for k in 0..r {
                let a_row = &layer.lora_v_a[k * h_dim..(k + 1) * h_dim];
                let mut sum = 0.0f32;
                for d_i in 0..h_dim {
                    sum += h_action[d_i] * a_row[d_i];
                }
                za_v[k] = sum;
            }

            for o in 0..kv_dim {
                let grad_out = d_h_action[o % h_dim];
                for k in 0..r {
                    let grad_b = (grad_out * za_v[k]).clamp(-1.0, 1.0);
                    layer.lora_v_b[o * r + k] -= lora_lr * grad_b;
                }
            }

            for k in 0..r {
                let mut d_za = 0.0f32;
                for o in 0..kv_dim {
                    d_za += d_h_action[o % h_dim] * layer.lora_v_b[o * r + k];
                }
                for d_i in 0..h_dim {
                    let grad_a = (d_za * h_action[d_i]).clamp(-1.0, 1.0);
                    layer.lora_v_a[k * h_dim + d_i] -= lora_lr * grad_a;
                }
            }
        }

        Ok(total_loss)
    }

    /// Load Neural World Model from directory containing `world_model.safetensors`
    /// or `world_model_dynamics.safetensors`.
    pub fn load_from_dir<P: AsRef<Path>>(dir: P) -> Result<Self, WorldModelError> {
        let p = dir.as_ref();
        let cfg_path = p.join("world_model_config.json");
        let alt_cfg_path = p.join("config.json");

        let config: WorldModelConfig = if cfg_path.exists() {
            let s = fs::read_to_string(&cfg_path)?;
            serde_json::from_str(&s)?
        } else if alt_cfg_path.exists() {
            let s = fs::read_to_string(&alt_cfg_path)?;
            serde_json::from_str(&s)?
        } else {
            WorldModelConfig::default()
        };

        config.validate()?;
        let mut wm = Self::new(config);

        let index_path = p.join("model.safetensors.index.json");
        let dyn_path = p.join("world_model.safetensors");
        let alt_dyn_path = p.join("world_model_dynamics.safetensors");

        let tensors = if index_path.exists() {
            load_model_weights(&p.to_string_lossy())
                .map_err(|e| WorldModelError::SafeTensors(e.to_string()))?
        } else if dyn_path.exists() {
            load_safetensors(&dyn_path.to_string_lossy())
                .map_err(|e| WorldModelError::SafeTensors(e.to_string()))?
        } else if alt_dyn_path.exists() {
            load_safetensors(&alt_dyn_path.to_string_lossy())
                .map_err(|e| WorldModelError::SafeTensors(e.to_string()))?
        } else {
            return Err(WorldModelError::MissingWeights);
        };

        let mut accepted_tensor_names = HashSet::new();
        let mut require_tensor = |name: &str, expected_len: usize| -> Result<(), WorldModelError> {
            let values = tensors.get(name).ok_or_else(|| {
                WorldModelError::SafeTensors(format!("required tensor '{name}' is missing"))
            })?;
            if values.len() != expected_len {
                return Err(WorldModelError::SafeTensors(format!(
                    "tensor '{name}' has {} values, expected {expected_len}",
                    values.len()
                )));
            }
            if values.iter().any(|value| !value.is_finite()) {
                return Err(WorldModelError::SafeTensors(format!(
                    "tensor '{name}' contains non-finite values"
                )));
            }
            accepted_tensor_names.insert(name.to_string());
            Ok(())
        };

        require_tensor("input_proj.state_proj.weight", wm.w_state.len())?;
        require_tensor("input_proj.action_proj.weight", wm.w_action.len())?;
        require_tensor("input_proj.norm.weight", wm.input_norm.len())?;
        require_tensor("norm.weight", wm.final_norm.len())?;
        require_tensor("heads.next_state_proj.weight", wm.w_next_state.len())?;
        require_tensor("heads.reward.weight", wm.w_reward.len())?;
        require_tensor("heads.reward.bias", 1)?;
        require_tensor("heads.viability_fail.weight", wm.w_fail.len())?;
        require_tensor("heads.viability_fail.bias", 1)?;
        require_tensor("heads.value.weight", wm.w_value.len())?;
        require_tensor("heads.value.bias", 1)?;

        for (layer_idx, layer) in wm.layers.iter().enumerate() {
            let prefix = format!("layers.{layer_idx}");
            require_tensor(
                &format!("{prefix}.input_layernorm.weight"),
                layer.input_layernorm.len(),
            )?;
            require_tensor(
                &format!("{prefix}.self_attn.q_proj.weight"),
                layer.q_proj.len(),
            )?;
            require_tensor(
                &format!("{prefix}.self_attn.k_proj.weight"),
                layer.k_proj.len(),
            )?;
            require_tensor(
                &format!("{prefix}.self_attn.v_proj.weight"),
                layer.v_proj.len(),
            )?;
            require_tensor(
                &format!("{prefix}.self_attn.o_proj.weight"),
                layer.o_proj.len(),
            )?;
            require_tensor(
                &format!("{prefix}.post_attention_layernorm.weight"),
                layer.post_attention_layernorm.len(),
            )?;
            require_tensor(
                &format!("{prefix}.mlp.gate_proj.weight"),
                layer.gate_proj.len(),
            )?;
            require_tensor(
                &format!("{prefix}.mlp.up_proj.weight"),
                layer.up_proj.len(),
            )?;
            require_tensor(
                &format!("{prefix}.mlp.down_proj.weight"),
                layer.down_proj.len(),
            )?;
        }

        let has_lora_tensors = tensors.keys().any(|name| name.contains(".lora_"));
        if has_lora_tensors {
            for (layer_idx, layer) in wm.layers.iter().enumerate() {
                let prefix = format!("layers.{layer_idx}.self_attn");
                require_tensor(
                    &format!("{prefix}.lora_q_a.weight"),
                    layer.lora_q_a.len(),
                )?;
                require_tensor(
                    &format!("{prefix}.lora_q_b.weight"),
                    layer.lora_q_b.len(),
                )?;
                require_tensor(
                    &format!("{prefix}.lora_v_a.weight"),
                    layer.lora_v_a.len(),
                )?;
                require_tensor(
                    &format!("{prefix}.lora_v_b.weight"),
                    layer.lora_v_b.len(),
                )?;
            }
        }

        if let Some(name) = tensors
            .keys()
            .find(|name| !accepted_tensor_names.contains(name.as_str()))
        {
            return Err(WorldModelError::SafeTensors(format!(
                "unexpected tensor '{name}' is incompatible with this world model architecture"
            )));
        }

            if let Some(t) = tensors.get("input_proj.state_proj.weight") {
                if t.len() == wm.w_state.len() {
                    wm.w_state.clone_from(t);
                }
            }
            if let Some(t) = tensors.get("input_proj.action_proj.weight") {
                if t.len() == wm.w_action.len() {
                    wm.w_action.clone_from(t);
                }
            }
            if let Some(t) = tensors.get("input_proj.norm.weight") {
                if t.len() == wm.input_norm.len() {
                    wm.input_norm.clone_from(t);
                }
            }

            for (l, layer) in wm.layers.iter_mut().enumerate() {
                let prefix = format!("layers.{}", l);
                if let Some(t) = tensors.get(&format!("{}.input_layernorm.weight", prefix)) {
                    if t.len() == layer.input_layernorm.len() {
                        layer.input_layernorm.clone_from(t);
                    }
                }
                if let Some(t) = tensors.get(&format!("{}.self_attn.q_proj.weight", prefix)) {
                    if t.len() == layer.q_proj.len() {
                        layer.q_proj.clone_from(t);
                    }
                }
                if let Some(t) = tensors.get(&format!("{}.self_attn.k_proj.weight", prefix)) {
                    if t.len() == layer.k_proj.len() {
                        layer.k_proj.clone_from(t);
                    }
                }
                if let Some(t) = tensors.get(&format!("{}.self_attn.v_proj.weight", prefix)) {
                    if t.len() == layer.v_proj.len() {
                        layer.v_proj.clone_from(t);
                    }
                }
                if let Some(t) = tensors.get(&format!("{}.self_attn.o_proj.weight", prefix)) {
                    if t.len() == layer.o_proj.len() {
                        layer.o_proj.clone_from(t);
                    }
                }
                if let Some(t) = tensors.get(&format!("{}.post_attention_layernorm.weight", prefix)) {
                    if t.len() == layer.post_attention_layernorm.len() {
                        layer.post_attention_layernorm.clone_from(t);
                    }
                }
                if let Some(t) = tensors.get(&format!("{}.mlp.gate_proj.weight", prefix)) {
                    if t.len() == layer.gate_proj.len() {
                        layer.gate_proj.clone_from(t);
                    }
                }
                if let Some(t) = tensors.get(&format!("{}.mlp.up_proj.weight", prefix)) {
                    if t.len() == layer.up_proj.len() {
                        layer.up_proj.clone_from(t);
                    }
                }
                if let Some(t) = tensors.get(&format!("{}.mlp.down_proj.weight", prefix)) {
                    if t.len() == layer.down_proj.len() {
                        layer.down_proj.clone_from(t);
                    }
                }
                if let Some(t) = tensors.get(&format!("{}.self_attn.lora_q_a.weight", prefix)) {
                    if t.len() == layer.lora_q_a.len() {
                        layer.lora_q_a.clone_from(t);
                    }
                }
                if let Some(t) = tensors.get(&format!("{}.self_attn.lora_q_b.weight", prefix)) {
                    if t.len() == layer.lora_q_b.len() {
                        layer.lora_q_b.clone_from(t);
                    }
                }
                if let Some(t) = tensors.get(&format!("{}.self_attn.lora_v_a.weight", prefix)) {
                    if t.len() == layer.lora_v_a.len() {
                        layer.lora_v_a.clone_from(t);
                    }
                }
                if let Some(t) = tensors.get(&format!("{}.self_attn.lora_v_b.weight", prefix)) {
                    if t.len() == layer.lora_v_b.len() {
                        layer.lora_v_b.clone_from(t);
                    }
                }
            }

            if let Some(t) = tensors.get("norm.weight") {
                if t.len() == wm.final_norm.len() {
                    wm.final_norm.clone_from(t);
                }
            }
            if let Some(t) = tensors.get("heads.next_state_proj.weight") {
                if t.len() == wm.w_next_state.len() {
                    wm.w_next_state.clone_from(t);
                }
            }
            if let Some(t) = tensors.get("heads.reward.weight") {
                if t.len() == wm.w_reward.len() {
                    wm.w_reward.clone_from(t);
                }
            }
            if let Some(t) = tensors.get("heads.reward.bias") {
                if !t.is_empty() {
                    wm.b_reward = t[0];
                }
            }
            if let Some(t) = tensors.get("heads.viability_fail.weight") {
                if t.len() == wm.w_fail.len() {
                    wm.w_fail.clone_from(t);
                }
            }
            if let Some(t) = tensors.get("heads.viability_fail.bias") {
                if !t.is_empty() {
                    wm.b_fail = t[0];
                }
            }
            if let Some(t) = tensors.get("heads.value.weight") {
                if t.len() == wm.w_value.len() {
                    wm.w_value.clone_from(t);
                }
            }
            if let Some(t) = tensors.get("heads.value.bias") {
                if !t.is_empty() {
                    wm.b_value = t[0];
                }
            }

        Ok(wm)
    }

    /// Load a checkpoint only when it carries evidence that optimization used real samples.
    pub fn load_trained_from_dir<P: AsRef<Path>>(dir: P) -> Result<Self, WorldModelError> {
        let p = dir.as_ref();
        let state_path = p.join("world_model_training_state.json");
        if !state_path.is_file() {
            return Err(WorldModelError::SafeTensors(
                "trained checkpoint state is missing; initialized or unverified weights cannot be used for trained inference".to_string(),
            ));
        }

        let state: serde_json::Value = serde_json::from_slice(&fs::read(state_path)?)?;
        let status = state.get("status").and_then(serde_json::Value::as_str).unwrap_or("");
        let training_steps = state
            .get("training_steps")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        let training_samples = state
            .get("training_samples")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        let baseline_improved = state
            .get("baseline_improved")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        let empty_validation = serde_json::Value::Null;
        let validation = state.get("last_validation").unwrap_or(&empty_validation);
        let validation_samples = validation
            .get("samples")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        let next_state_mse = validation
            .get("next_state_mse")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(f64::INFINITY);
        let persistence_baseline_mse = validation
            .get("next_state_persistence_baseline_mse")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(f64::INFINITY);
        if status != "trained"
            || training_steps == 0
            || training_samples == 0
            || !baseline_improved
            || validation_samples == 0
            || !next_state_mse.is_finite()
            || !persistence_baseline_mse.is_finite()
            || next_state_mse >= persistence_baseline_mse
        {
            return Err(WorldModelError::SafeTensors(format!(
                "checkpoint lacks successful held-out training evidence (status='{status}', training_steps={training_steps}, training_samples={training_samples}, validation_samples={validation_samples}, baseline_improved={baseline_improved})"
            )));
        }

        Self::load_from_dir(p)
    }

    /// Persist Neural World Model dynamics and config to target directory in part-wise sharded Safetensors format.
    pub fn save_to_dir_sharded<P: AsRef<Path>>(
        &self,
        dir: P,
        max_shard_bytes: usize,
    ) -> Result<Vec<String>, WorldModelError> {
        let p = dir.as_ref();
        fs::create_dir_all(p)?;

        // Save config in both canonical names for universal tooling and engine compatibility
        let cfg_path = p.join("world_model_config.json");
        let alt_cfg_path = p.join("config.json");
        let cfg_str = serde_json::to_string_pretty(&self.config)?;
        fs::write(&cfg_path, &cfg_str)?;
        fs::write(&alt_cfg_path, &cfg_str)?;

        // Save tensors
        let mut tensors = HashMap::new();
        let mut shapes = HashMap::new();

        let h = self.config.hidden_size;
        let s = self.config.latent_dim;
        let a = self.config.action_dim;
        let inter = self.config.intermediate_size;
        let q_dim = self.config.num_attention_heads * self.config.head_dim;
        let kv_dim = self.config.num_key_value_heads * self.config.head_dim;

        tensors.insert("input_proj.state_proj.weight".to_string(), self.w_state.clone());
        shapes.insert("input_proj.state_proj.weight".to_string(), vec![h, s]);

        tensors.insert("input_proj.action_proj.weight".to_string(), self.w_action.clone());
        shapes.insert("input_proj.action_proj.weight".to_string(), vec![h, a]);

        tensors.insert("input_proj.norm.weight".to_string(), self.input_norm.clone());
        shapes.insert("input_proj.norm.weight".to_string(), vec![h]);

        for (l, layer) in self.layers.iter().enumerate() {
            let prefix = format!("layers.{}", l);
            tensors.insert(format!("{}.input_layernorm.weight", prefix), layer.input_layernorm.clone());
            shapes.insert(format!("{}.input_layernorm.weight", prefix), vec![h]);

            tensors.insert(format!("{}.self_attn.q_proj.weight", prefix), layer.q_proj.clone());
            shapes.insert(format!("{}.self_attn.q_proj.weight", prefix), vec![q_dim, h]);

            tensors.insert(format!("{}.self_attn.k_proj.weight", prefix), layer.k_proj.clone());
            shapes.insert(format!("{}.self_attn.k_proj.weight", prefix), vec![kv_dim, h]);

            tensors.insert(format!("{}.self_attn.v_proj.weight", prefix), layer.v_proj.clone());
            shapes.insert(format!("{}.self_attn.v_proj.weight", prefix), vec![kv_dim, h]);

            tensors.insert(format!("{}.self_attn.o_proj.weight", prefix), layer.o_proj.clone());
            shapes.insert(format!("{}.self_attn.o_proj.weight", prefix), vec![h, q_dim]);

            if layer.lora_rank > 0 && !layer.lora_q_a.is_empty() {
                tensors.insert(format!("{}.self_attn.lora_q_a.weight", prefix), layer.lora_q_a.clone());
                shapes.insert(format!("{}.self_attn.lora_q_a.weight", prefix), vec![layer.lora_rank, h]);

                tensors.insert(format!("{}.self_attn.lora_q_b.weight", prefix), layer.lora_q_b.clone());
                shapes.insert(format!("{}.self_attn.lora_q_b.weight", prefix), vec![q_dim, layer.lora_rank]);

                tensors.insert(format!("{}.self_attn.lora_v_a.weight", prefix), layer.lora_v_a.clone());
                shapes.insert(format!("{}.self_attn.lora_v_a.weight", prefix), vec![layer.lora_rank, h]);

                tensors.insert(format!("{}.self_attn.lora_v_b.weight", prefix), layer.lora_v_b.clone());
                shapes.insert(format!("{}.self_attn.lora_v_b.weight", prefix), vec![kv_dim, layer.lora_rank]);
            }

            tensors.insert(format!("{}.post_attention_layernorm.weight", prefix), layer.post_attention_layernorm.clone());
            shapes.insert(format!("{}.post_attention_layernorm.weight", prefix), vec![h]);

            tensors.insert(format!("{}.mlp.gate_proj.weight", prefix), layer.gate_proj.clone());
            shapes.insert(format!("{}.mlp.gate_proj.weight", prefix), vec![inter, h]);

            tensors.insert(format!("{}.mlp.up_proj.weight", prefix), layer.up_proj.clone());
            shapes.insert(format!("{}.mlp.up_proj.weight", prefix), vec![inter, h]);

            tensors.insert(format!("{}.mlp.down_proj.weight", prefix), layer.down_proj.clone());
            shapes.insert(format!("{}.mlp.down_proj.weight", prefix), vec![h, inter]);
        }

        tensors.insert("norm.weight".to_string(), self.final_norm.clone());
        shapes.insert("norm.weight".to_string(), vec![h]);

        tensors.insert("heads.next_state_proj.weight".to_string(), self.w_next_state.clone());
        shapes.insert("heads.next_state_proj.weight".to_string(), vec![s, h]);

        tensors.insert("heads.reward.weight".to_string(), self.w_reward.clone());
        shapes.insert("heads.reward.weight".to_string(), vec![1, h]);

        tensors.insert("heads.reward.bias".to_string(), vec![self.b_reward]);
        shapes.insert("heads.reward.bias".to_string(), vec![1]);

        tensors.insert("heads.viability_fail.weight".to_string(), self.w_fail.clone());
        shapes.insert("heads.viability_fail.weight".to_string(), vec![1, h]);

        tensors.insert("heads.viability_fail.bias".to_string(), vec![self.b_fail]);
        shapes.insert("heads.viability_fail.bias".to_string(), vec![1]);

        tensors.insert("heads.value.weight".to_string(), self.w_value.clone());
        shapes.insert("heads.value.weight".to_string(), vec![1, h]);

        tensors.insert("heads.value.bias".to_string(), vec![self.b_value]);
        shapes.insert("heads.value.bias".to_string(), vec![1]);

        let shard_files = write_safetensors_sharded_with_prefix(
            &tensors,
            &shapes,
            &p.to_string_lossy(),
            max_shard_bytes,
            "world_model",
            false,
        )
        .map_err(|e| WorldModelError::SafeTensors(e.to_string()))?;

        Ok(shard_files)
    }

    /// Save Neural World Model to directory in standard part-wise sharded format (same as neural model).
    pub fn save_to_dir<P: AsRef<Path>>(&self, dir: P) -> Result<Vec<String>, WorldModelError> {
        let max_bytes = resolve_max_shard_size_bytes();
        self.save_to_dir_sharded(dir, max_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transformer_world_model_initialization_and_predict_step() {
        let config = WorldModelConfig::new(64, 32, 64);
        let wm = WorldModel::new(config);

        let state = vec![0.5f32; 64];
        let action = vec![0.1f32; 32];

        let transition = wm.predict_step(&state, &action).expect("predict step succeeds");
        assert_eq!(transition.next_latent.len(), 64);
        assert!(transition.predicted_reward >= -1.0 && transition.predicted_reward <= 1.0);
        assert!(transition.failure_probability >= 0.0 && transition.failure_probability <= 1.0);
        assert!(transition.confidence > 0.0 && transition.confidence <= 1.0);
    }

    #[test]
    fn test_transformer_world_model_1b_geometry_and_invariants() {
        let config = WorldModelConfig::default();
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_key_value_heads, 8);
        assert_eq!(config.head_dim, 64);
        assert_eq!(config.intermediate_size, 5632);
        assert_eq!(config.head_dim * config.num_attention_heads, config.hidden_size);
        assert!(config.validate().is_ok());

        let wm = WorldModel::new(config);
        assert_eq!(wm.w_state.len(), 2048 * 2048);
        assert_eq!(wm.w_action.len(), 2048 * 512);
        assert_eq!(wm.layers.len(), 8);
    }

    #[test]
    fn test_transformer_world_model_counterfactual_rollout() {
        let config = WorldModelConfig::new(64, 32, 64);
        let wm = WorldModel::new(config);

        let state = vec![0.2f32; 64];
        let actions = vec![vec![0.1f32; 32], vec![0.2f32; 32], vec![-0.1f32; 32]];

        let trajectory = wm.rollout(&state, &actions).expect("rollout succeeds");
        assert_eq!(trajectory.transitions.len(), 3);
        assert!(trajectory.min_confidence > 0.0);
    }

    #[test]
    fn test_transformer_world_model_online_surprise_update() {
        let config = WorldModelConfig::new(64, 32, 64);
        let mut wm = WorldModel::new(config);

        let s_t = vec![0.1f32; 64];
        let a_t = vec![0.05f32; 32];
        let real_s_next = vec![0.3f32; 64];

        let loss = wm
            .update_from_experience(&s_t, &a_t, &real_s_next, 0.8, false)
            .expect("update succeeds");
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_transformer_world_model_safetensors_roundtrip() {
        let unique_id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let temp_dir = std::env::temp_dir().join(format!("tara_trans_wm_roundtrip_{}_{}", std::process::id(), unique_id));
        let config = WorldModelConfig::new(64, 32, 64);
        let wm1 = WorldModel::new(config);

        wm1.save_to_dir(&temp_dir).expect("save_to_dir succeeds");
        let wm2 = WorldModel::load_from_dir(&temp_dir).expect("load_from_dir succeeds");

        assert_eq!(wm1.w_state, wm2.w_state);
        assert_eq!(wm1.w_action, wm2.w_action);
        assert_eq!(wm1.layers.len(), wm2.layers.len());
        assert_eq!(wm1.layers[0].q_proj, wm2.layers[0].q_proj);
        assert_eq!(wm1.layers[0].lora_q_a, wm2.layers[0].lora_q_a);
        assert_eq!(wm1.layers[0].lora_q_b, wm2.layers[0].lora_q_b);
        assert_eq!(wm1.layers[0].lora_v_a, wm2.layers[0].lora_v_a);
        assert_eq!(wm1.layers[0].lora_v_b, wm2.layers[0].lora_v_b);
        assert_eq!(wm1.w_reward, wm2.w_reward);
        assert!((wm1.b_reward - wm2.b_reward).abs() < 1e-6);

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_transformer_world_model_hidden_and_action_bridges() {
        let config = WorldModelConfig::new(64, 32, 64);
        let wm = WorldModel::new(config);

        // Test 1: Direct hidden state encoding with dimension match
        let hidden_exact = vec![0.25f32; 64];
        let z1 = wm.encode_state_from_hidden(&hidden_exact);
        assert_eq!(z1.len(), 64);
        let norm1: f32 = z1.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm1 - 1.0).abs() < 1e-4);

        // Test 2: Resampled hidden state encoding from different dimension (e.g., 2048 -> 64)
        let hidden_large = vec![0.1f32; 2048];
        let z2 = wm.encode_state_from_hidden(&hidden_large);
        assert_eq!(z2.len(), 64);
        let norm2: f32 = z2.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm2 - 1.0).abs() < 1e-4);

        // Test 3: Structured action encoding
        let params = serde_json::json!({
            "command": "cargo check",
            "timeout_sec": 30
        });
        let a = wm.encode_structured_action("shell_exec", &params, None);
        assert_eq!(a.len(), 32);
        let norm_a: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm_a - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_transformer_world_model_deep_convergence_and_closed_loop_learning() {
        let config = WorldModelConfig::new(64, 32, 64);
        let mut wm = WorldModel::new(config);

        let s_t = vec![0.3f32; 64];
        let a_t = vec![0.15f32; 32];
        let real_s_next = vec![0.6f32; 64];
        let real_r = 0.9f32;
        let real_fail = false;

        let initial_loss = wm
            .update_from_experience(&s_t, &a_t, &real_s_next, real_r, real_fail)
            .expect("first update succeeds");

        let mut final_loss = initial_loss;
        for _ in 0..15 {
            final_loss = wm
                .update_from_experience(&s_t, &a_t, &real_s_next, real_r, real_fail)
                .expect("subsequent update succeeds");
        }

        assert!(
            final_loss < initial_loss,
            "World Model learning must reduce prediction loss: final {} vs initial {}",
            final_loss,
            initial_loss
        );

        let actions = vec![
            vec![0.1f32; 32],
            vec![0.2f32; 32],
            vec![-0.1f32; 32],
            vec![0.05f32; 32],
        ];
        let traj = wm.rollout(&s_t, &actions).expect("multi-step rollout succeeds");
        assert_eq!(traj.transitions.len(), 4);
        for t in &traj.transitions {
            assert_eq!(t.next_latent.len(), 64);
            assert!(t.predicted_reward >= -1.0 && t.predicted_reward <= 1.0);
            assert!(t.failure_probability >= 0.0 && t.failure_probability <= 1.0);
            assert!(t.confidence >= 0.1 && t.confidence <= 0.99);
            assert!(t.expected_value.is_finite());
        }
    }

    #[test]
    fn test_transformer_world_model_on_disk_shards_integrity() {
        let p = std::path::Path::new("storage/models/tara_world_model");
        if !p.exists() || !p.join("model.safetensors.index.json").exists() {
            return;
        }

        let wm = WorldModel::load_from_dir(p).expect("loading on-disk sharded World Model succeeds");
        assert_eq!(wm.config.latent_dim, 2048);
        assert_eq!(wm.config.hidden_size, 2048);
        assert_eq!(wm.config.action_dim, 512);
        assert_eq!(wm.layers.len(), 8);
        assert_eq!(wm.w_state.len(), 2048 * 2048);
        assert_eq!(wm.w_action.len(), 2048 * 512);

        let test_state = vec![0.01f32; 2048];
        let test_action = vec![0.02f32; 512];
        let step = wm.predict_step(&test_state, &test_action).expect("predict_step on on-disk model succeeds");
        assert_eq!(step.next_latent.len(), 2048);
        assert!(step.predicted_reward.is_finite());
        assert!(step.failure_probability >= 0.0 && step.failure_probability <= 1.0);
        assert!(step.expected_value.is_finite());
        assert!(step.confidence > 0.0);
    }

    #[test]
    fn test_transformer_world_model_load_python_checkpoint() {
        let p = std::path::Path::new("target/test_wm_run");
        if !p.exists() || !p.join("model.safetensors.index.json").exists() {
            return;
        }

        let wm = WorldModel::load_from_dir(p).expect("loading python-exported sharded World Model succeeds");
        assert_eq!(wm.config.latent_dim, 64);
        assert_eq!(wm.config.hidden_size, 64);
        assert_eq!(wm.config.action_dim, 32);
        assert_eq!(wm.layers.len(), 2);

        let test_state = vec![0.05f32; 64];
        let test_action = vec![0.02f32; 32];
        let step = wm.predict_step(&test_state, &test_action).expect("predict_step on python-trained model succeeds");
        assert_eq!(step.next_latent.len(), 64);
        assert!(step.predicted_reward.is_finite());
        assert!(step.failure_probability >= 0.0 && step.failure_probability <= 1.0);
        assert!(step.expected_value.is_finite());
        assert!(step.confidence > 0.0);
    }
}
