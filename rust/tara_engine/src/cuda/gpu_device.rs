//! CUDA-accelerated implementation of TARA-HD ComputeDevice.
//!
//! Exposes genuine GPU hardware execution through the unified TARA-HD interface.

use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tara_hd::{ComputeDevice, DeviceCapability};
use crate::cuda::gpu_trainer::CudaTrainer;

/// CUDA GPU Compute Device bridging CudaTrainer with the TARA-HD ComputeDevice trait.
#[derive(Clone)]
pub struct CudaComputeDevice {
    info: DeviceCapability,
    trainer: Arc<Mutex<CudaTrainer>>,
    slice_to_name: Arc<Mutex<HashMap<usize, String>>>,
}

impl std::fmt::Debug for CudaComputeDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let count = self.slice_to_name.lock().map(|m| m.len()).unwrap_or(0);
        f.debug_struct("CudaComputeDevice")
            .field("info", &self.info)
            .field("registered_parameters", &count)
            .finish()
    }
}

impl CudaComputeDevice {
    pub fn new(
        info: DeviceCapability,
        trainer: Arc<Mutex<CudaTrainer>>,
        slice_to_name: HashMap<usize, String>,
    ) -> Self {
        Self {
            info,
            trainer,
            slice_to_name: Arc::new(Mutex::new(slice_to_name)),
        }
    }

    pub fn trainer(&self) -> &Arc<Mutex<CudaTrainer>> {
        &self.trainer
    }

    pub fn update_slice_map(&self, map: HashMap<usize, String>) {
        if let Ok(mut lock) = self.slice_to_name.lock() {
            *lock = map;
        }
    }
}

impl ComputeDevice for CudaComputeDevice {
    fn info(&self) -> &DeviceCapability {
        &self.info
    }

    fn is_available(&self) -> bool {
        self.trainer.lock().is_ok()
    }

    fn synchronize(&self) -> Result<(), String> {
        let t = self.trainer.lock().map_err(|e| e.to_string())?;
        t.synchronize().map_err(|e| e.to_string())
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
                | "linear"
                | "swiglu"
                | "residual"
        )
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn forward_linear(
        &self,
        weight: &[f32],
        x: &[f32],
        seq_len: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<Vec<f32>, String> {
        let ptr = weight.as_ptr() as usize;
        let name_opt = self.slice_to_name.lock().ok().and_then(|m| m.get(&ptr).cloned());
        if let Some(name) = name_opt {
            let t = self.trainer.lock().map_err(|e| e.to_string())?;
            if let Ok(res) = t.forward_linear(&name, x, seq_len, in_dim, out_dim) {
                return Ok(res);
            }
        }

        if seq_len == 0 || in_dim == 0 || out_dim == 0 {
            return Ok(Vec::new());
        }
        if x.len() < seq_len * in_dim || weight.len() < out_dim * in_dim {
            return Err("forward_linear dimension mismatch".to_string());
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

    fn forward_linear_named(
        &self,
        weight_name: &str,
        weight_slice: &[f32],
        x: &[f32],
        seq_len: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<Vec<f32>, String> {
        let t = self.trainer.lock().map_err(|e| e.to_string())?;
        match t.forward_linear(weight_name, x, seq_len, in_dim, out_dim) {
            Ok(v) => Ok(v),
            Err(e) => {
                log::debug!("GPU forward_linear for '{weight_name}' falling back to slice: {e}");
                drop(t);
                self.forward_linear(weight_slice, x, seq_len, in_dim, out_dim)
            }
        }
    }

    fn forward_rmsnorm(
        &self,
        weight: &[f32],
        x: &[f32],
        seq_len: usize,
        hidden_dim: usize,
        eps: f32,
    ) -> Result<Vec<f32>, String> {
        let ptr = weight.as_ptr() as usize;
        let name_opt = self.slice_to_name.lock().ok().and_then(|m| m.get(&ptr).cloned());
        if let Some(name) = name_opt {
            let t = self.trainer.lock().map_err(|e| e.to_string())?;
            if let Ok(res) = t.forward_rmsnorm(&name, x, seq_len, hidden_dim, eps) {
                return Ok(res);
            }
        }

        if seq_len == 0 || hidden_dim == 0 {
            return Ok(Vec::new());
        }
        let safe_eps = if eps.is_finite() && eps > 0.0 { eps } else { 1e-5 };
        let mut out = vec![0.0f32; seq_len * hidden_dim];
        use rayon::prelude::*;
        out.par_chunks_exact_mut(hidden_dim)
            .enumerate()
            .for_each(|(i, out_row)| {
                let x_row = &x[i * hidden_dim..(i + 1) * hidden_dim];
                let mean_sq: f32 = x_row.iter().map(|&v| v * v).sum::<f32>() / (hidden_dim as f32).max(1.0);
                let inv_rms = 1.0 / (mean_sq + safe_eps).sqrt();
                for (j, (&xv, &wv)) in x_row.iter().zip(weight.iter()).enumerate() {
                    out_row[j] = xv * inv_rms * wv;
                }
            });
        Ok(out)
    }

    fn forward_rmsnorm_named(
        &self,
        weight_name: &str,
        weight_slice: &[f32],
        x: &[f32],
        seq_len: usize,
        hidden_dim: usize,
        eps: f32,
    ) -> Result<Vec<f32>, String> {
        let t = self.trainer.lock().map_err(|e| e.to_string())?;
        match t.forward_rmsnorm(weight_name, x, seq_len, hidden_dim, eps) {
            Ok(v) => Ok(v),
            Err(e) => {
                log::debug!("GPU forward_rmsnorm for '{weight_name}' falling back to slice: {e}");
                drop(t);
                self.forward_rmsnorm(weight_slice, x, seq_len, hidden_dim, eps)
            }
        }
    }

    fn forward_swiglu(&self, gate: &[f32], up: &[f32]) -> Result<Vec<f32>, String> {
        let t = self.trainer.lock().map_err(|e| e.to_string())?;
        match t.forward_swiglu(gate, up) {
            Ok(v) => Ok(v),
            Err(_) => {
                drop(t);
                let len = gate.len().min(up.len());
                let mut out = vec![0.0f32; len];
                use rayon::prelude::*;
                out.par_iter_mut().enumerate().for_each(|(i, val)| {
                    let g = gate[i];
                    let sig = 1.0 / (1.0 + (-g).exp());
                    *val = (g * sig) * up[i];
                });
                Ok(out)
            }
        }
    }

    fn add_residual(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, String> {
        let t = self.trainer.lock().map_err(|e| e.to_string())?;
        match t.add_residual(a, b) {
            Ok(v) => Ok(v),
            Err(_) => {
                drop(t);
                let len = a.len().min(b.len());
                let mut out = vec![0.0f32; len];
                use rayon::prelude::*;
                out.par_iter_mut().enumerate().for_each(|(i, val)| {
                    *val = a[i] + b[i];
                });
                Ok(out)
            }
        }
    }

    fn backward_linear(
        &self,
        weight: &[f32],
        d_out: &[f32],
        x: &[f32],
        seq_len: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let ptr = weight.as_ptr() as usize;
        let name_opt = self.slice_to_name.lock().ok().and_then(|m| m.get(&ptr).cloned());
        if let Some(name) = name_opt {
            let t = self.trainer.lock().map_err(|e| e.to_string())?;
            if let Ok(res) = t.backward_linear(&name, d_out, x, seq_len, in_dim, out_dim) {
                return Ok(res);
            }
        }

        use rayon::prelude::*;
        let mut d_weight = vec![0.0f32; out_dim * in_dim];
        d_weight.par_chunks_exact_mut(in_dim).enumerate().for_each(|(o, w_row)| {
            for s in 0..seq_len {
                let d_val = d_out[s * out_dim + o];
                let x_row = &x[s * in_dim..(s + 1) * in_dim];
                for (i, &xv) in x_row.iter().enumerate() {
                    w_row[i] += d_val * xv;
                }
            }
        });
        let mut d_in = vec![0.0f32; seq_len * in_dim];
        d_in.par_chunks_exact_mut(in_dim).enumerate().for_each(|(s, in_row)| {
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

    fn backward_linear_named(
        &self,
        weight_name: &str,
        weight_slice: &[f32],
        d_out: &[f32],
        x: &[f32],
        seq_len: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let t = self.trainer.lock().map_err(|e| e.to_string())?;
        match t.backward_linear(weight_name, d_out, x, seq_len, in_dim, out_dim) {
            Ok(v) => Ok(v),
            Err(e) => {
                log::debug!("GPU backward_linear for '{weight_name}' falling back to slice: {e}");
                drop(t);
                self.backward_linear(weight_slice, d_out, x, seq_len, in_dim, out_dim)
            }
        }
    }

    fn forward_embedding(
        &self,
        token_ids: &[u32],
        embed_table: &[f32],
        hidden_size: usize,
    ) -> Result<Vec<f32>, String> {
        let t = self.trainer.lock().map_err(|e| e.to_string())?;
        match t.forward_embedding(token_ids, hidden_size) {
            Ok(v) => Ok(v),
            Err(_) => {
                drop(t);
                let vs = embed_table.len() / hidden_size.max(1);
                let mut out = vec![0.0f32; token_ids.len() * hidden_size];
                for (t, &id) in token_ids.iter().enumerate() {
                    let id = (id as usize).min(vs.saturating_sub(1));
                    let src = &embed_table[id * hidden_size..(id + 1) * hidden_size];
                    out[t * hidden_size..(t + 1) * hidden_size].copy_from_slice(src);
                }
                Ok(out)
            }
        }
    }

    fn accumulate_gradient(&self, param_name: &str, gradient: &[f32]) -> Result<(), String> {
        let t = self.trainer.lock().map_err(|e| e.to_string())?;
        t.accumulate_gradient(param_name, gradient).map_err(|e| e.to_string())
    }
}
