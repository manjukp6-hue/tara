//! Unified device trait and concrete device implementations for TARA-HD.

use std::any::Any;
use std::fmt::Debug;

use crate::streaming::{CpuStreamHelper, StreamConfig};
use crate::types::{DeviceCapability, DeviceKind, LinearBackwardArgs};

/// Unified interface representing an abstract compute device.
pub trait ComputeDevice: Send + Sync + Debug {
    /// Return static and dynamic capabilities of this device.
    fn info(&self) -> &DeviceCapability;

    /// Returns whether the device is currently healthy and operational.
    fn is_available(&self) -> bool;

    /// Wait for all pending asynchronous compute operations on this device to complete.
    fn synchronize(&self) -> Result<(), String>;

    /// Check if the device natively supports a specific named compute operation.
    fn supports_operation(&self, op_name: &str) -> bool;

    /// Downcast support for concrete device inspection.
    fn as_any(&self) -> &dyn Any;

    /// Optional reference to the CPU on-demand streaming helper (returns `Some` on `CpuDevice`).
    fn stream_helper(&self) -> Option<&CpuStreamHelper> {
        None
    }

    /// Returns human-readable name.
    fn name(&self) -> &str {
        &self.info().name
    }

    /// Returns high-level device category.
    fn kind(&self) -> &DeviceKind {
        &self.info().kind
    }

    /// Linear forward projection: y = x @ weight^T.
    /// x: [seq_len, in_dim], weight: [out_dim, in_dim], out: [seq_len, out_dim].
    fn forward_linear(
        &self,
        weight: &[f32],
        x: &[f32],
        seq_len: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<Vec<f32>, String> {
        if seq_len == 0 || in_dim == 0 || out_dim == 0 {
            return Ok(Vec::new());
        }
        if x.len() < seq_len * in_dim || weight.len() < out_dim * in_dim {
            return Err(format!(
                "forward_linear dimension mismatch: x len={}, weight len={}, expected {} and {}",
                x.len(),
                weight.len(),
                seq_len * in_dim,
                out_dim * in_dim
            ));
        }
        use rayon::prelude::*;
        let mut out = vec![0.0f32; seq_len * out_dim];
        out.par_chunks_exact_mut(out_dim)
            .enumerate()
            .for_each(|(i, out_row)| {
                let x_row = &x[i * in_dim..(i + 1) * in_dim];
                for j in 0..out_dim {
                    let w_row = &weight[j * in_dim..(j + 1) * in_dim];
                    out_row[j] = x_row.iter().zip(w_row.iter()).map(|(&a, &b)| a * b).sum();
                }
            });
        Ok(out)
    }

    /// RMSNorm forward pass.
    fn forward_rmsnorm(
        &self,
        weight: &[f32],
        x: &[f32],
        seq_len: usize,
        hidden_dim: usize,
        eps: f32,
    ) -> Result<Vec<f32>, String> {
        if seq_len == 0 || hidden_dim == 0 {
            return Ok(Vec::new());
        }
        if x.len() < seq_len * hidden_dim || weight.len() < hidden_dim {
            return Err(format!(
                "forward_rmsnorm dimension mismatch: x len={}, weight len={}, expected {} and {}",
                x.len(),
                weight.len(),
                seq_len * hidden_dim,
                hidden_dim
            ));
        }
        use rayon::prelude::*;
        let safe_eps = if eps.is_finite() && eps > 0.0 {
            eps
        } else {
            1e-5
        };
        let mut out = vec![0.0f32; seq_len * hidden_dim];
        out.par_chunks_exact_mut(hidden_dim)
            .enumerate()
            .for_each(|(i, out_row)| {
                let x_row = &x[i * hidden_dim..(i + 1) * hidden_dim];
                let mean_sq: f32 =
                    x_row.iter().map(|&v| v * v).sum::<f32>() / (hidden_dim as f32).max(1.0);
                let inv_rms = 1.0 / (mean_sq + safe_eps).sqrt();
                for (j, (&xv, &wv)) in x_row.iter().zip(weight.iter()).enumerate() {
                    out_row[j] = xv * inv_rms * wv;
                }
            });
        Ok(out)
    }

    /// SwiGLU activation: out = (gate * sigmoid(gate)) * up.
    fn forward_swiglu(&self, gate: &[f32], up: &[f32]) -> Result<Vec<f32>, String> {
        let len = gate.len().min(up.len());
        if len == 0 {
            return Ok(Vec::new());
        }
        use rayon::prelude::*;
        let mut out = vec![0.0f32; len];
        out.par_iter_mut().enumerate().for_each(|(i, val)| {
            let g = gate[i];
            let sig = 1.0 / (1.0 + (-g).exp());
            *val = (g * sig) * up[i];
        });
        Ok(out)
    }

    /// Element-wise residual addition: out = a + b.
    fn add_residual(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, String> {
        let len = a.len().min(b.len());
        if len == 0 {
            return Ok(Vec::new());
        }
        use rayon::prelude::*;
        let mut out = vec![0.0f32; len];
        out.par_iter_mut().enumerate().for_each(|(i, val)| {
            *val = a[i] + b[i];
        });
        Ok(out)
    }

    /// Linear backward projection: computes (d_weight, d_in).
    fn backward_linear(
        &self,
        weight: &[f32],
        d_out: &[f32],
        x: &[f32],
        seq_len: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        if seq_len == 0 || in_dim == 0 || out_dim == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        if d_out.len() < seq_len * out_dim
            || x.len() < seq_len * in_dim
            || weight.len() < out_dim * in_dim
        {
            return Err("backward_linear buffer length mismatch".to_string());
        }
        use rayon::prelude::*;
        let mut d_weight = vec![0.0f32; out_dim * in_dim];
        d_weight
            .par_chunks_exact_mut(in_dim)
            .enumerate()
            .for_each(|(o, w_row)| {
                for s in 0..seq_len {
                    let d_val = d_out[s * out_dim + o];
                    let x_row = &x[s * in_dim..(s + 1) * in_dim];
                    for (i, &xv) in x_row.iter().enumerate() {
                        w_row[i] += d_val * xv;
                    }
                }
            });

        let mut d_in = vec![0.0f32; seq_len * in_dim];
        d_in.par_chunks_exact_mut(in_dim)
            .enumerate()
            .for_each(|(s, in_row)| {
                let d_row = &d_out[s * out_dim..(s + 1) * out_dim];
                for (o, &dv) in d_row.iter().enumerate() {
                    let w_row = &weight[o * in_dim..(o + 1) * in_dim];
                    for (i, &wv) in w_row.iter().enumerate() {
                        in_row[i] += dv * wv;
                    }
                }
            });

        Ok((d_weight, d_in))
    }

    /// Linear forward projection by parameter name (GPU VRAM resident when available) or slice fallback.
    fn forward_linear_named(
        &self,
        _weight_name: &str,
        weight_slice: &[f32],
        x: &[f32],
        seq_len: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<Vec<f32>, String> {
        self.forward_linear(weight_slice, x, seq_len, in_dim, out_dim)
    }

    /// RMSNorm forward pass by parameter name or slice fallback.
    fn forward_rmsnorm_named(
        &self,
        _weight_name: &str,
        weight_slice: &[f32],
        x: &[f32],
        seq_len: usize,
        hidden_dim: usize,
        eps: f32,
    ) -> Result<Vec<f32>, String> {
        self.forward_rmsnorm(weight_slice, x, seq_len, hidden_dim, eps)
    }

    /// Linear backward projection by parameter name or slice fallback.
    fn backward_linear_named(
        &self,
        args: LinearBackwardArgs<'_>,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        self.backward_linear(
            args.weight_slice,
            args.d_out,
            args.x,
            args.seq_len,
            args.in_dim,
            args.out_dim,
        )
    }

    /// Embedding forward lookup: token_id -> embedding vector.
    fn forward_embedding(
        &self,
        token_ids: &[u32],
        embed_table: &[f32],
        hidden_size: usize,
    ) -> Result<Vec<f32>, String> {
        if token_ids.is_empty() || hidden_size == 0 {
            return Ok(Vec::new());
        }
        let vs = embed_table.len() / hidden_size.max(1);
        let mut out = vec![0.0f32; token_ids.len() * hidden_size];
        for (t, &id) in token_ids.iter().enumerate() {
            let id = (id as usize).min(vs.saturating_sub(1));
            let src = &embed_table[id * hidden_size..(id + 1) * hidden_size];
            out[t * hidden_size..(t + 1) * hidden_size].copy_from_slice(src);
        }
        Ok(out)
    }

    /// Accumulate gradient into device parameter buffer.
    fn accumulate_gradient(&self, _param_name: &str, _gradient: &[f32]) -> Result<(), String> {
        Ok(())
    }

    /// Accumulate embedding gradients sparsely for active tokens only, avoiding dense matrix allocation.
    fn accumulate_sparse_embedding_gradient(
        &self,
        _tokens: &[u32],
        _dx: &[f32],
        _hs: usize,
    ) -> Result<f64, String> {
        Ok(0.0)
    }

    /// Backward LM Head projection and chunked/GPU accumulation directly into parameter storage.
    /// Returns (d_lm_head_l2_norm, d_final_normed).
    fn backward_lm_head_accumulate(
        &self,
        d_logits: &[f32],
        final_normed: &[f32],
        seq_len: usize,
        hs: usize,
        vs: usize,
    ) -> Result<(f64, Vec<f32>), String> {
        let (d_lm_head, d_final_normed) = self.backward_linear(
            &[],
            d_logits,
            final_normed,
            seq_len,
            hs,
            vs,
        )?;
        let norm: f64 = d_lm_head.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
        self.accumulate_gradient("lm_head.weight", &d_lm_head)?;
        Ok((norm, d_final_normed))
    }
}

/// Accelerated NVIDIA CUDA GPU device implementation.
#[derive(Debug)]
pub struct CudaDevice {
    info: DeviceCapability,
}

impl CudaDevice {
    pub fn new(info: DeviceCapability) -> Self {
        Self { info }
    }
}

impl ComputeDevice for CudaDevice {
    fn info(&self) -> &DeviceCapability {
        &self.info
    }

    fn is_available(&self) -> bool {
        true
    }

    fn synchronize(&self) -> Result<(), String> {
        Ok(())
    }

    fn supports_operation(&self, op_name: &str) -> bool {
        matches!(
            op_name.trim().to_ascii_lowercase().as_str(),
            "matmul" | "attention" | "rope" | "rmsnorm" | "cross_entropy" | "adamw" | "gemm"
        )
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Google Cloud TPU accelerator device implementation.
#[derive(Debug)]
pub struct TpuDevice {
    info: DeviceCapability,
}

impl TpuDevice {
    pub fn new(info: DeviceCapability) -> Self {
        Self { info }
    }
}

impl ComputeDevice for TpuDevice {
    fn info(&self) -> &DeviceCapability {
        &self.info
    }

    fn is_available(&self) -> bool {
        true
    }

    fn synchronize(&self) -> Result<(), String> {
        Ok(())
    }

    fn supports_operation(&self, op_name: &str) -> bool {
        matches!(
            op_name.trim().to_ascii_lowercase().as_str(),
            "matmul"
                | "attention"
                | "rope"
                | "rmsnorm"
                | "cross_entropy"
                | "adamw"
                | "gemm"
                | "xla_hlo"
        )
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Host CPU compute device equipped with on-demand chunked streaming.
#[derive(Debug)]
pub struct CpuDevice {
    info: DeviceCapability,
    stream_helper: CpuStreamHelper,
}

impl CpuDevice {
    pub fn new(info: DeviceCapability) -> Self {
        // CPU streaming memory budget is computed from available RAM with overridable bounds.
        // Set TARA_CPU_STREAM_MIN_MB / TARA_CPU_STREAM_MAX_MB to clamp to different ranges.
        // Set TARA_CPU_CHUNK_TOKENS to control the token window size per streaming chunk.
        // Set TARA_CPU_RESIDENT_LAYERS to control how many layer activation buffers stay resident.
        let usable_bytes = info
            .available_memory_bytes
            .map(|avail| avail / 2)
            .unwrap_or(info.total_memory_bytes / 4);

        let min_mb = std::env::var("TARA_CPU_STREAM_MIN_MB")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(512);
        let max_mb = std::env::var("TARA_CPU_STREAM_MAX_MB")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(16384);
        let effective_min = min_mb.max(64); // absolute floor: 64 MB for correctness
        let effective_max = max_mb.max(effective_min);
        let budget_mb =
            ((usable_bytes / (1024 * 1024)) as usize).clamp(effective_min, effective_max);

        let chunk_size_tokens = std::env::var("TARA_CPU_CHUNK_TOKENS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(256)
            .max(1);
        let max_resident_layers = std::env::var("TARA_CPU_RESIDENT_LAYERS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(2)
            .max(1);

        let config = StreamConfig {
            chunk_size_tokens,
            max_resident_layers,
            memory_budget_mb: budget_mb,
        };

        Self {
            info,
            stream_helper: CpuStreamHelper::with_config(config),
        }
    }

    /// Create a CPU device with an explicit streaming configuration.
    pub fn with_stream_config(info: DeviceCapability, config: StreamConfig) -> Self {
        Self {
            info,
            stream_helper: CpuStreamHelper::with_config(config),
        }
    }

    /// Reference to the on-demand CPU streaming helper.
    pub fn stream_helper(&self) -> &CpuStreamHelper {
        &self.stream_helper
    }
}

impl ComputeDevice for CpuDevice {
    fn info(&self) -> &DeviceCapability {
        &self.info
    }

    fn is_available(&self) -> bool {
        true
    }

    fn synchronize(&self) -> Result<(), String> {
        Ok(())
    }

    fn supports_operation(&self, _op_name: &str) -> bool {
        true
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn stream_helper(&self) -> Option<&CpuStreamHelper> {
        Some(&self.stream_helper)
    }
}

/// Pluggable adapter for future accelerators (Metal, ROCm, custom NPUs).
#[derive(Debug)]
pub struct CustomDevice {
    info: DeviceCapability,
    supported_ops: Vec<String>,
}

impl CustomDevice {
    pub fn new(info: DeviceCapability, supported_ops: Vec<String>) -> Self {
        Self {
            info,
            supported_ops,
        }
    }

    pub fn supported_operations(&self) -> &[String] {
        &self.supported_ops
    }
}

impl ComputeDevice for CustomDevice {
    fn info(&self) -> &DeviceCapability {
        &self.info
    }

    fn is_available(&self) -> bool {
        true
    }

    fn synchronize(&self) -> Result<(), String> {
        Ok(())
    }

    fn supports_operation(&self, op_name: &str) -> bool {
        let trimmed = op_name.trim();
        self.supported_ops
            .iter()
            .any(|op| op.eq_ignore_ascii_case(trimmed))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
