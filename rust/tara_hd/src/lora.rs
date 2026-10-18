//! Pure Native Rust Low-Rank Adaptation (LoRA: ΔW = (alpha/r) * B * A) Engine.
//!
//! Enforces:
//! 1. Mathematically pure LoRA forward, backward, and AdamW optimization in 100% native Rust.
//! 2. Zero catastrophic forgetting by freezing base model weights W_0.
//! 3. Bounded memory footprint (30-50 MB ceiling) by avoiding full-model AdamW moment allocation.
//! 4. Seamless adapter hot-swapping, merge_and_unload, and unmerge operations.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for Low-Rank Adaptation (LoRA).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoraConfig {
    /// Rank of the low-rank decomposition (r << min(d_in, d_out)).
    pub r: usize,
    /// Scaling hyperparameter alpha.
    pub alpha: f32,
    /// Target module name prefixes to apply LoRA (e.g. "q_proj", "v_proj").
    pub target_modules: Vec<String>,
    /// Dropout probability for adapter activations (0.0 to disable).
    pub dropout: f32,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            r: 8,
            alpha: 16.0,
            target_modules: vec![
                "q_proj".to_string(),
                "k_proj".to_string(),
                "v_proj".to_string(),
                "o_proj".to_string(),
            ],
            dropout: 0.0,
        }
    }
}

impl LoraConfig {
    /// Returns the effective LoRA scaling coefficient: alpha / r.
    #[inline]
    pub fn scale(&self) -> f32 {
        if self.r == 0 {
            1.0
        } else {
            self.alpha / self.r as f32
        }
    }
}

/// Individual LoRA adapter layer attached to a linear projection.
/// Computes ΔW = scale * (B * A) where:
/// - A is [r, in_dim] (initialized to Gaussian noise / Kaiming scaled)
/// - B is [out_dim, r] (initialized to 0.0 so ΔW = 0 initially)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraLayer {
    pub name: String,
    pub in_dim: usize,
    pub out_dim: usize,
    pub r: usize,
    pub scale: f32,
    /// Matrix A: shape [r, in_dim], stored row-major.
    pub a: Vec<f32>,
    /// Matrix B: shape [out_dim, r], stored row-major.
    pub b: Vec<f32>,
    /// Gradient accumulator for A: shape [r, in_dim].
    #[serde(skip)]
    pub grad_a: Vec<f32>,
    /// Gradient accumulator for B: shape [out_dim, r].
    #[serde(skip)]
    pub grad_b: Vec<f32>,
    /// AdamW 1st moment vector for A.
    #[serde(skip)]
    pub m_a: Vec<f32>,
    /// AdamW 2nd moment vector for A.
    #[serde(skip)]
    pub v_a: Vec<f32>,
    /// AdamW 1st moment vector for B.
    #[serde(skip)]
    pub m_b: Vec<f32>,
    /// AdamW 2nd moment vector for B.
    #[serde(skip)]
    pub v_b: Vec<f32>,
    /// Optimizer step counter.
    #[serde(skip)]
    pub step_count: u64,
    /// Cached activation z_A = x * A^T for backward pass: shape [seq_len, r].
    #[serde(skip)]
    pub last_za: Vec<f32>,
    /// Cached input x for backward pass: shape [seq_len, in_dim].
    #[serde(skip)]
    pub last_x: Vec<f32>,
    /// Cached sequence length of last forward pass.
    #[serde(skip)]
    pub last_seq_len: usize,
    /// Tracks if forward pass has been executed.
    #[serde(skip)]
    pub has_forwarded: bool,
}

impl LoraLayer {
    /// Create a new LoRA layer for linear projection with dimensions (in_dim, out_dim).
    pub fn new(name: &str, in_dim: usize, out_dim: usize, r: usize, scale: f32) -> Self {
        let a_len = r * in_dim;
        let b_len = out_dim * r;

        // Initialize A with deterministic pseudo-random Gaussian distribution (mean 0, std = 1 / sqrt(r))
        let std_dev = if r > 0 { 1.0 / (r as f32).sqrt() } else { 0.02 };
        let mut a = Vec::with_capacity(a_len);
        // Robust byte-wise FNV-1a hash of the layer name to ensure symmetry breaking across all heads and layers
        let mut seed: u64 = 0xcbf29ce484222325;
        for &byte in name.as_bytes() {
            seed = (seed ^ (byte as u64)).wrapping_mul(0x100000001b3);
        }
        for _ in 0..a_len {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u1 = ((seed >> 32) as u32 as f32 + 1.0) / 4294967297.0;
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u2 = ((seed >> 32) as u32 as f32 + 1.0) / 4294967297.0;
            let normal = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            a.push(normal * std_dev);
        }

        // Initialize B to strictly 0.0 to ensure initial ΔW = 0 bit-for-bit
        let b = vec![0.0f32; b_len];

        Self {
            name: name.to_string(),
            in_dim,
            out_dim,
            r,
            scale,
            a,
            b,
            grad_a: vec![0.0f32; a_len],
            grad_b: vec![0.0f32; b_len],
            m_a: vec![0.0f32; a_len],
            v_a: vec![0.0f32; a_len],
            m_b: vec![0.0f32; b_len],
            v_b: vec![0.0f32; b_len],
            step_count: 0,
            last_za: Vec::new(),
            last_x: Vec::new(),
            last_seq_len: 0,
            has_forwarded: false,
        }
    }

    /// Compute total memory consumption of this layer in bytes (weights + gradients + optimizer moments).
    pub fn memory_footprint_bytes(&self) -> usize {
        let param_bytes = (self.a.len() + self.b.len()) * std::mem::size_of::<f32>();
        let grad_bytes = (self.grad_a.len() + self.grad_b.len()) * std::mem::size_of::<f32>();
        let moment_bytes = (self.m_a.len() + self.v_a.len() + self.m_b.len() + self.v_b.len())
            * std::mem::size_of::<f32>();
        let cache_bytes = (self.last_za.len() + self.last_x.len()) * std::mem::size_of::<f32>();
        param_bytes + grad_bytes + moment_bytes + cache_bytes
    }

    /// Forward pass:
    /// Given base projection output `base_out` of shape [seq_len, out_dim] and input `x` of shape [seq_len, in_dim],
    /// adds the low-rank delta: out = base_out + scale * (x * A^T) * B^T.
    pub fn forward(
        &mut self,
        x: &[f32],
        base_out: &[f32],
        seq_len: usize,
    ) -> Result<Vec<f32>, String> {
        if x.len() != seq_len * self.in_dim {
            return Err(format!(
                "LoraLayer '{}' input shape mismatch: expected {}, got {}",
                self.name,
                seq_len * self.in_dim,
                x.len()
            ));
        }
        if base_out.len() != seq_len * self.out_dim {
            return Err(format!(
                "LoraLayer '{}' base_out shape mismatch: expected {}, got {}",
                self.name,
                seq_len * self.out_dim,
                base_out.len()
            ));
        }

        // Cache for backward pass
        self.last_seq_len = seq_len;
        self.last_x = x.to_vec();
        self.has_forwarded = true;

        // 1. Compute z_A = x * A^T: shape [seq_len, r]
        // x is [seq_len, in_dim], A is [r, in_dim], so z_A[s, k] = sum_d (x[s, d] * A[k, d])
        let mut za = vec![0.0f32; seq_len * self.r];
        for s in 0..seq_len {
            let x_s = &x[s * self.in_dim..(s + 1) * self.in_dim];
            for k in 0..self.r {
                let a_k = &self.a[k * self.in_dim..(k + 1) * self.in_dim];
                let mut dot = 0.0f32;
                for d in 0..self.in_dim {
                    dot += x_s[d] * a_k[d];
                }
                za[s * self.r + k] = dot;
            }
        }
        self.last_za = za.clone();

        // 2. Compute delta_h = scale * (z_A * B^T): shape [seq_len, out_dim]
        // B is [out_dim, r], so delta_h[s, o] = scale * sum_k (z_A[s, k] * B[o, k])
        let mut out = base_out.to_vec();
        for s in 0..seq_len {
            let za_s = &za[s * self.r..(s + 1) * self.r];
            for o in 0..self.out_dim {
                let b_o = &self.b[o * self.r..(o + 1) * self.r];
                let mut dot = 0.0f32;
                for k in 0..self.r {
                    dot += za_s[k] * b_o[k];
                }
                out[s * self.out_dim + o] += self.scale * dot;
            }
        }

        Ok(out)
    }

    /// Backward pass:
    /// Given incoming gradient `d_out` of shape [seq_len, out_dim],
    /// computes gradients for A and B, accumulates them, and returns input gradient `d_x_adapter` of shape [seq_len, in_dim].
    pub fn backward(
        &mut self,
        d_out: &[f32],
    ) -> Result<Vec<f32>, String> {
        let seq_len = self.last_seq_len;
        if !self.has_forwarded || seq_len == 0 || self.last_x.is_empty() {
            return Err(format!(
                "LoraLayer '{}' backward called before forward pass",
                self.name
            ));
        }
        if d_out.len() != seq_len * self.out_dim {
            return Err(format!(
                "LoraLayer '{}' d_out shape mismatch: expected {}, got {}",
                self.name,
                seq_len * self.out_dim,
                d_out.len()
            ));
        }

        // Rank 0 adapter yields zero input gradient and has zero parameters
        if self.r == 0 {
            return Ok(vec![0.0f32; seq_len * self.in_dim]);
        }

        let scale = self.scale;

        // 1. Accumulate grad_B: d_B = scale * d_out^T * z_A
        // d_out is [seq_len, out_dim], z_A is [seq_len, r]
        // d_B[o, k] = scale * sum_s (d_out[s, o] * z_A[s, k])
        for s in 0..seq_len {
            let d_out_s = &d_out[s * self.out_dim..(s + 1) * self.out_dim];
            let za_s = &self.last_za[s * self.r..(s + 1) * self.r];
            for (o, &d_out_val) in d_out_s.iter().enumerate().take(self.out_dim) {
                let d_val = scale * d_out_val;
                for (k, &za_val) in za_s.iter().enumerate().take(self.r) {
                    self.grad_b[o * self.r + k] += d_val * za_val;
                }
            }
        }

        // 2. Compute d_zA = scale * d_out * B: shape [seq_len, r]
        // B is [out_dim, r], so d_zA[s, k] = scale * sum_o (d_out[s, o] * B[o, k])
        let mut d_za = vec![0.0f32; seq_len * self.r];
        for s in 0..seq_len {
            let d_out_s = &d_out[s * self.out_dim..(s + 1) * self.out_dim];
            for k in 0..self.r {
                let mut sum_k = 0.0f32;
                for (o, &d_out_val) in d_out_s.iter().enumerate().take(self.out_dim) {
                    sum_k += d_out_val * self.b[o * self.r + k];
                }
                d_za[s * self.r + k] = scale * sum_k;
            }
        }

        // 3. Accumulate grad_A: d_A = d_zA^T * x: shape [r, in_dim]
        // x is [seq_len, in_dim], so d_A[k, d] = sum_s (d_zA[s, k] * x[s, d])
        for s in 0..seq_len {
            let d_za_s = &d_za[s * self.r..(s + 1) * self.r];
            let x_s = &self.last_x[s * self.in_dim..(s + 1) * self.in_dim];
            for (k, &d_za_val) in d_za_s.iter().enumerate().take(self.r) {
                for (d, &x_val) in x_s.iter().enumerate().take(self.in_dim) {
                    self.grad_a[k * self.in_dim + d] += d_za_val * x_val;
                }
            }
        }

        // 4. Compute adapter input gradient d_x_adapter = d_zA * A: shape [seq_len, in_dim]
        // d_x[s, d] = sum_k (d_za[s, k] * A[k, d])
        let mut d_x = vec![0.0f32; seq_len * self.in_dim];
        for s in 0..seq_len {
            let d_za_s = &d_za[s * self.r..(s + 1) * self.r];
            for d in 0..self.in_dim {
                let mut sum_d = 0.0f32;
                for (k, &d_za_val) in d_za_s.iter().enumerate().take(self.r) {
                    sum_d += d_za_val * self.a[k * self.in_dim + d];
                }
                d_x[s * self.in_dim + d] = sum_d;
            }
        }

        Ok(d_x)
    }

    /// Execute localized AdamW optimizer step on this adapter's weights A and B.
    pub fn step_adamw(
        &mut self,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        max_grad_norm: f32,
    ) {
        self.step_count += 1;
        let bc1 = 1.0 - beta1.powi(self.step_count as i32);
        let bc2 = 1.0 - beta2.powi(self.step_count as i32);

        // Compute local gradient norm across A and B
        let mut norm_sq = 0.0f32;
        for &g in self.grad_a.iter().chain(self.grad_b.iter()) {
            norm_sq += g * g;
        }
        let total_norm = norm_sq.sqrt();
        let clip_scale = if total_norm > max_grad_norm && total_norm > 0.0 {
            max_grad_norm / total_norm
        } else {
            1.0f32
        };

        // Update A
        for i in 0..self.a.len() {
            let g = self.grad_a[i] * clip_scale;
            self.m_a[i] = beta1 * self.m_a[i] + (1.0 - beta1) * g;
            self.v_a[i] = beta2 * self.v_a[i] + (1.0 - beta2) * g * g;
            let m_hat = self.m_a[i] / bc1;
            let v_hat = self.v_a[i] / bc2;
            let update = m_hat / (v_hat.sqrt() + eps) + weight_decay * self.a[i];
            self.a[i] -= lr * update;
            self.grad_a[i] = 0.0;
        }

        // Update B
        for i in 0..self.b.len() {
            let g = self.grad_b[i] * clip_scale;
            self.m_b[i] = beta1 * self.m_b[i] + (1.0 - beta1) * g;
            self.v_b[i] = beta2 * self.v_b[i] + (1.0 - beta2) * g * g;
            let m_hat = self.m_b[i] / bc1;
            let v_hat = self.v_b[i] / bc2;
            let update = m_hat / (v_hat.sqrt() + eps) + weight_decay * self.b[i];
            self.b[i] -= lr * update;
            self.grad_b[i] = 0.0;
        }
    }

    /// Compute full ΔW = scale * (B * A) matrix of shape [out_dim, in_dim].
    pub fn delta_weight(&self) -> Vec<f32> {
        let mut delta = vec![0.0f32; self.out_dim * self.in_dim];
        for o in 0..self.out_dim {
            let b_o = &self.b[o * self.r..(o + 1) * self.r];
            for d in 0..self.in_dim {
                let mut sum_k = 0.0f32;
                for (k, &b_val) in b_o.iter().enumerate().take(self.r) {
                    sum_k += b_val * self.a[k * self.in_dim + d];
                }
                delta[o * self.in_dim + d] = self.scale * sum_k;
            }
        }
        delta
    }

    /// Merge adapter weights directly into base weight matrix: W_merged = W_0 + ΔW.
    pub fn merge_into_slice(&self, base_weights: &mut [f32]) -> Result<(), String> {
        if base_weights.len() != self.out_dim * self.in_dim {
            return Err(format!(
                "Base weight slice length {} does not match adapter shape {}x{}",
                base_weights.len(),
                self.out_dim,
                self.in_dim
            ));
        }
        let delta = self.delta_weight();
        for (w, d) in base_weights.iter_mut().zip(delta.iter()) {
            *w += d;
        }
        Ok(())
    }

    /// Unmerge adapter weights from merged weight matrix: W_0 = W_merged - ΔW.
    pub fn unmerge_from_slice(&self, merged_weights: &mut [f32]) -> Result<(), String> {
        if merged_weights.len() != self.out_dim * self.in_dim {
            return Err(format!(
                "Merged weight slice length {} does not match adapter shape {}x{}",
                merged_weights.len(),
                self.out_dim,
                self.in_dim
            ));
        }
        let delta = self.delta_weight();
        for (w, d) in merged_weights.iter_mut().zip(delta.iter()) {
            *w -= d;
        }
        Ok(())
    }
}

/// Collection of LoRA layers representing a complete ability adapter set (e.g. Stage 2 ability or Stage 3 continual adaptation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraAdapterSet {
    /// Unique name of this ability adapter (e.g. "quantum_reasoning", "stage2_ability").
    pub ability_name: String,
    /// LoRA configuration used by this adapter set.
    pub config: LoraConfig,
    /// Map of target module name -> LoraLayer.
    pub layers: HashMap<String, LoraLayer>,
}

impl LoraAdapterSet {
    /// Create a new adapter set for a specific ability and configuration.
    pub fn new(ability_name: &str, config: LoraConfig) -> Self {
        Self {
            ability_name: ability_name.to_string(),
            config,
            layers: HashMap::new(),
        }
    }

    /// Add a target linear layer to this adapter set.
    pub fn add_layer(&mut self, name: &str, in_dim: usize, out_dim: usize) {
        let layer = LoraLayer::new(name, in_dim, out_dim, self.config.r, self.config.scale());
        self.layers.insert(name.to_string(), layer);
    }

    /// Calculate total memory footprint in bytes across all layers in this adapter set.
    pub fn total_memory_footprint_bytes(&self) -> usize {
        self.layers.values().map(|l| l.memory_footprint_bytes()).sum()
    }

    /// Total number of trainable parameters in this adapter set.
    pub fn total_trainable_parameters(&self) -> usize {
        self.layers.values().map(|l| l.a.len() + l.b.len()).sum()
    }

    /// Check if the total memory footprint conforms to the 30-50 MB operational budget ceiling.
    pub fn is_within_memory_budget(&self, max_mb: usize) -> bool {
        let max_bytes = max_mb * 1024 * 1024;
        self.total_memory_footprint_bytes() <= max_bytes
    }

    /// Execute forward pass for a named layer through its LoRA adapter.
    pub fn forward_named(
        &mut self,
        name: &str,
        x: &[f32],
        base_out: &[f32],
        seq_len: usize,
    ) -> Result<Vec<f32>, String> {
        if let Some(layer) = self.layers.get_mut(name) {
            layer.forward(x, base_out, seq_len)
        } else {
            Ok(base_out.to_vec())
        }
    }

    /// Execute backward pass for a named layer through its LoRA adapter.
    pub fn backward_named(
        &mut self,
        name: &str,
        d_out: &[f32],
    ) -> Result<Option<Vec<f32>>, String> {
        if let Some(layer) = self.layers.get_mut(name) {
            let d_x = layer.backward(d_out)?;
            Ok(Some(d_x))
        } else {
            Ok(None)
        }
    }

    /// Execute AdamW optimization step across all adapter layers in this set.
    pub fn step_adamw(
        &mut self,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        max_grad_norm: f32,
    ) {
        for layer in self.layers.values_mut() {
            layer.step_adamw(lr, beta1, beta2, eps, weight_decay, max_grad_norm);
        }
    }

    /// Merge all adapters in this set into the provided model weight dictionary.
    pub fn merge_into_weights(&self, weights: &mut HashMap<String, Vec<f32>>) -> Result<usize, String> {
        let mut merged_count = 0;
        for (name, layer) in &self.layers {
            if let Some(w_slice) = weights.get_mut(name) {
                layer.merge_into_slice(w_slice)?;
                merged_count += 1;
            }
        }
        Ok(merged_count)
    }

    /// Unmerge all adapters in this set from the provided model weight dictionary.
    pub fn unmerge_from_weights(&self, weights: &mut HashMap<String, Vec<f32>>) -> Result<usize, String> {
        let mut unmerged_count = 0;
        for (name, layer) in &self.layers {
            if let Some(w_slice) = weights.get_mut(name) {
                layer.unmerge_from_slice(w_slice)?;
                unmerged_count += 1;
            }
        }
        Ok(unmerged_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_initial_delta_is_zero() {
        let mut layer = LoraLayer::new("test_proj", 64, 128, 8, 2.0);
        assert_eq!(layer.b.iter().filter(|&&x| x != 0.0).count(), 0);

        let delta = layer.delta_weight();
        for &val in &delta {
            assert_eq!(val, 0.0, "Initial LoRA delta must be strictly zero");
        }

        let seq_len = 4;
        let x = vec![1.0f32; seq_len * 64];
        let base_out = vec![2.0f32; seq_len * 128];
        let out = layer.forward(&x, &base_out, seq_len).expect("forward must succeed");

        assert_eq!(out, base_out, "Initial forward pass must equal base_out exactly");
    }

    #[test]
    fn test_lora_gradient_and_adamw_step() {
        let mut layer = LoraLayer::new("q_proj", 32, 64, 4, 1.0);
        let seq_len = 2;
        let x = vec![0.5f32; seq_len * 32];
        let base_out = vec![1.0f32; seq_len * 64];

        let out1 = layer.forward(&x, &base_out, seq_len).unwrap();
        assert_eq!(out1, base_out);

        // Simulated incoming gradient
        let d_out = vec![0.1f32; seq_len * 64];
        let d_x = layer.backward(&d_out).expect("backward must succeed");
        assert_eq!(d_x.len(), seq_len * 32);

        // Verify gradients were accumulated
        let grad_b_sum: f32 = layer.grad_b.iter().copied().sum();
        assert!(grad_b_sum.abs() > 0.0, "grad_b must be non-zero after backward");

        // Execute AdamW step
        layer.step_adamw(0.01, 0.9, 0.999, 1e-8, 0.01, 1.0);
        assert_eq!(layer.step_count, 1);

        // Forward pass after step must now reflect updated B weights!
        let out2 = layer.forward(&x, &base_out, seq_len).unwrap();
        let diff: f32 = out2.iter().zip(base_out.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 0.0, "LoRA adapter must adapt output after AdamW step");
    }

    #[test]
    fn test_lora_merge_and_unmerge_invariance() {
        let mut layer = LoraLayer::new("v_proj", 16, 16, 4, 2.0);
        // Force non-zero weights in B
        layer.b.fill(0.05);

        let original_base = vec![1.5f32; 16 * 16];
        let mut weights = original_base.clone();

        // Merge
        layer.merge_into_slice(&mut weights).unwrap();
        assert_ne!(weights, original_base);

        // Unmerge
        layer.unmerge_from_slice(&mut weights).unwrap();
        for (w, orig) in weights.iter().zip(original_base.iter()) {
            assert!((w - orig).abs() < 1e-6, "Unmerged weights must bit-for-bit restore original base");
        }
    }

    #[test]
    fn test_lora_adapter_set_memory_ceiling() {
        let config = LoraConfig {
            r: 8,
            alpha: 16.0,
            target_modules: vec!["q_proj".into(), "v_proj".into()],
            dropout: 0.0,
        };
        let mut set = LoraAdapterSet::new("test_ability", config);

        // Add 16 layers typical for a 1B model (d_model = 2048)
        for i in 0..16 {
            set.add_layer(&format!("model.layers.{i}.self_attn.q_proj"), 2048, 2048);
            set.add_layer(&format!("model.layers.{i}.self_attn.v_proj"), 2048, 2048);
        }

        let memory_bytes = set.total_memory_footprint_bytes();
        let memory_mb = memory_bytes as f32 / (1024.0 * 1024.0);

        println!("LoRA Adapter Set Memory Footprint: {memory_mb:.2} MB");
        assert!(
            set.is_within_memory_budget(50),
            "LoRA memory footprint must remain strictly within 50 MB budget ceiling (got {memory_mb:.2} MB)"
        );
    }

    #[test]
    fn test_lora_initialization_symmetry_breaking() {
        // Two projections with identical name lengths must NOT produce identical initialization weights
        let l1 = LoraLayer::new("model.layers.0.self_attn.q_proj", 64, 64, 8, 1.0);
        let l2 = LoraLayer::new("model.layers.0.self_attn.k_proj", 64, 64, 8, 1.0);
        assert_eq!(l1.name.len(), l2.name.len());
        assert_ne!(
            l1.a, l2.a,
            "Layers with same name length must have distinct initialization weights to break symmetry"
        );
    }

    #[test]
    fn test_lora_arbitrary_non_divisible_dimensions() {
        // Prime dimensions that are not multiples of rank r=7
        let in_dim = 97;
        let out_dim = 131;
        let r = 7;
        let seq_len = 3;

        let mut layer = LoraLayer::new("arbitrary_proj", in_dim, out_dim, r, 2.0);
        let x = vec![0.3f32; seq_len * in_dim];
        let base_out = vec![1.0f32; seq_len * out_dim];

        let out = layer.forward(&x, &base_out, seq_len).expect("forward must support arbitrary dimensions");
        assert_eq!(out.len(), seq_len * out_dim);

        let d_out = vec![0.05f32; seq_len * out_dim];
        let d_x = layer.backward(&d_out).expect("backward must support arbitrary dimensions");
        assert_eq!(d_x.len(), seq_len * in_dim);

        layer.step_adamw(0.01, 0.9, 0.999, 1e-8, 0.01, 1.0);
        assert_eq!(layer.step_count, 1);

        let mut base_weights = vec![0.5f32; out_dim * in_dim];
        let orig = base_weights.clone();
        layer.merge_into_slice(&mut base_weights).expect("merge must succeed");
        layer.unmerge_from_slice(&mut base_weights).expect("unmerge must succeed");
        for (w, o) in base_weights.iter().zip(orig.iter()) {
            assert!((w - o).abs() < 1e-6);
        }
    }

    #[test]
    fn test_lora_rank_zero_graceful() {
        let mut layer = LoraLayer::new("zero_rank_proj", 32, 64, 0, 1.0);
        assert_eq!(layer.a.len(), 0);
        assert_eq!(layer.b.len(), 0);

        let seq_len = 2;
        let x = vec![0.5f32; seq_len * 32];
        let base_out = vec![1.0f32; seq_len * 64];

        let out = layer.forward(&x, &base_out, seq_len).expect("rank 0 forward must succeed");
        assert_eq!(out, base_out);

        let d_out = vec![0.1f32; seq_len * 64];
        let d_x = layer.backward(&d_out).expect("rank 0 backward must succeed");
        assert_eq!(d_x.len(), seq_len * 32);
        assert!(d_x.iter().all(|&v| v == 0.0));
    }
}
