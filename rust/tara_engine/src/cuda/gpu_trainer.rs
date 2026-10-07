//! Native CUDA Trainer for TARA (FP32 & Mixed-Precision FP16).
//!
//! Executes real GPU acceleration on NVIDIA hardware:
//! - Pure FP32 and genuine FP16 CUDA kernels
//! - FP32 master weights & FP16 model weights / activations / gradients
//! - Real-time VRAM tracking and hardware telemetry

use crate::cuda::driver::{
    CudaBuffer, CudaDeviceInfo, CudaError, CudaKernel, CudaModuleHandle, CudaSession,
};
use crate::cuda::kernels::PTX_TARA_KERNELS;
use half::f16;
use std::collections::HashMap;
use std::ffi::c_void;

/// Numerical training precision backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum TrainingPrecision {
    Fp32,
    Fp16,
    #[default]
    Auto,
}

impl std::str::FromStr for TrainingPrecision {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_lowercase().as_str() {
            "fp32" => Ok(TrainingPrecision::Fp32),
            "fp16" => Ok(TrainingPrecision::Fp16),
            "auto" => Ok(TrainingPrecision::Auto),
            other => Err(format!(
                "Unknown precision '{other}'. Valid options: fp32, fp16, auto"
            )),
        }
    }
}

/// Safely packs f16 values into u32 words without out-of-bounds pointer reads or UB.
pub fn pack_f16_to_u32(values: &[f16]) -> Vec<u32> {
    let num_words = values.len().div_ceil(2);
    let mut out = vec![0u32; num_words];
    for (i, &v) in values.iter().enumerate() {
        let bits = v.to_bits() as u32;
        if i % 2 == 0 {
            out[i / 2] |= bits;
        } else {
            out[i / 2] |= bits << 16;
        }
    }
    out
}

/// Safely unpacks u32 words into f16 values.
pub fn unpack_u32_to_f16(packed: &[u32], len: usize) -> Vec<f16> {
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let word = packed[i / 2];
        let bits = if i % 2 == 0 {
            (word & 0xFFFF) as u16
        } else {
            ((word >> 16) & 0xFFFF) as u16
        };
        out.push(f16::from_bits(bits));
    }
    out
}

pub type OptimizerStateMap = HashMap<String, (Vec<f32>, Vec<f32>)>;

pub struct GpuParamState {
    pub d_weight: Option<CudaBuffer>, // FP32 master weights (None when streamed from host)
    pub d_weight_f16: Option<CudaBuffer>, // FP16 model weights (in FP16 mode)
    pub d_grad: CudaBuffer,           // FP32 or FP16 gradient buffer
    pub d_m: Option<CudaBuffer>,      // FP32 momentum (None when streamed from host)
    pub d_v: Option<CudaBuffer>,      // FP32 variance (None when streamed from host)
    pub len: usize,
}

pub struct CudaTrainer {
    // Parameter states resident on GPU (dropped first to free device buffers)
    params: HashMap<String, GpuParamState>,
    pub precision: TrainingPrecision,
    // Streamed optimizer states on host (used for 1B model / VRAM-constrained environments)
    h_weights: HashMap<String, Vec<f32>>,
    h_m: HashMap<String, Vec<f32>>,
    h_v: HashMap<String, Vec<f32>>,
    d_chunk_w: Option<CudaBuffer>,
    d_chunk_m: Option<CudaBuffer>,
    d_chunk_v: Option<CudaBuffer>,
    stream_optimizer: bool,
    // FP32 Kernels
    k_accum_grads: CudaKernel,
    k_scale_grads: CudaKernel,
    k_adamw_step: CudaKernel,
    k_lm_fwd: CudaKernel,
    k_lm_bwd_weight: CudaKernel,
    k_lm_bwd_input: CudaKernel,
    // FP16 Kernels
    k_adamw_step_f16: CudaKernel,
    k_lm_fwd_f16: CudaKernel,
    k_lm_bwd_weight_f16: CudaKernel,
    k_lm_bwd_input_f16: CudaKernel,
    // Transformer Layer Kernels
    k_emb_fwd: CudaKernel,
    k_emb_fwd_f16: CudaKernel,
    k_emb_bwd: CudaKernel,
    k_rms_fwd: CudaKernel,
    k_swiglu_fwd: CudaKernel,
    k_swiglu_bwd: CudaKernel,
    k_res_add: CudaKernel,
    k_norm_sq: CudaKernel,
    module: CudaModuleHandle,
    step_count: u64,
    d_norm_sq: CudaBuffer,
    // Context session (must drop LAST after all device resources are freed)
    session: CudaSession,
}

impl CudaTrainer {
    pub fn new(device_ordinal: i32) -> Result<Self, CudaError> {
        Self::new_with_precision(device_ordinal, TrainingPrecision::Auto)
    }

    pub fn new_with_precision(
        device_ordinal: i32,
        precision: TrainingPrecision,
    ) -> Result<Self, CudaError> {
        let session = CudaSession::init(device_ordinal)?;
        let module = session.load_ptx_module(PTX_TARA_KERNELS)?;

        // Dynamically resolve Auto precision based on hardware capability
        let resolved_precision = match precision {
            TrainingPrecision::Auto => {
                let info = session.device();
                let (major, minor) = info.compute_capability;
                // Tesla T4 is compute capability 7.5 (Turing). GPUs with CC >= 5.3 have native FP16 support
                if major > 5 || (major == 5 && minor >= 3) {
                    TrainingPrecision::Fp16
                } else {
                    TrainingPrecision::Fp32
                }
            }
            other => other,
        };

        // Load FP32 kernels
        let k_accum_grads = module.get_kernel("accum_grads_kernel")?;
        let k_scale_grads = module.get_kernel("scale_grads_kernel")?;
        let k_adamw_step = module.get_kernel("adamw_step_kernel")?;
        let k_lm_fwd = module.get_kernel("lm_head_fwd_kernel")?;
        let k_lm_bwd_weight = module.get_kernel("lm_head_bwd_weight_kernel")?;
        let k_lm_bwd_input = module.get_kernel("lm_head_bwd_input_kernel")?;

        // Load FP16 kernels
        let k_adamw_step_f16 = module.get_kernel("adamw_step_mixed_f16_kernel")?;
        let k_lm_fwd_f16 = module.get_kernel("lm_head_fwd_f16_kernel")?;
        let k_lm_bwd_weight_f16 = module.get_kernel("lm_head_bwd_weight_f16_kernel")?;
        let k_lm_bwd_input_f16 = module.get_kernel("lm_head_bwd_input_f16_kernel")?;

        // Load Transformer Layer kernels
        let k_emb_fwd = module.get_kernel("embedding_fwd_kernel")?;
        let k_emb_fwd_f16 = module.get_kernel("embedding_fwd_f16_kernel")?;
        let k_emb_bwd = module.get_kernel("embedding_bwd_kernel")?;
        let k_rms_fwd = module.get_kernel("rmsnorm_fwd_kernel")?;
        let k_swiglu_fwd = module.get_kernel("swiglu_fwd_kernel")?;
        let k_swiglu_bwd = module.get_kernel("swiglu_bwd_kernel")?;
        let k_res_add = module.get_kernel("residual_add_kernel")?;
        let k_norm_sq = module.get_kernel("grad_norm_sq_kernel")?;
        let d_norm_sq = session.allocate_f32(1)?;

        Ok(Self {
            params: HashMap::new(),
            precision: resolved_precision,
            h_weights: HashMap::new(),
            h_m: HashMap::new(),
            h_v: HashMap::new(),
            d_chunk_w: None,
            d_chunk_m: None,
            d_chunk_v: None,
            stream_optimizer: false,
            k_accum_grads,
            k_scale_grads,
            k_adamw_step,
            k_lm_fwd,
            k_lm_bwd_weight,
            k_lm_bwd_input,
            k_adamw_step_f16,
            k_lm_fwd_f16,
            k_lm_bwd_weight_f16,
            k_lm_bwd_input_f16,
            k_emb_fwd,
            k_emb_fwd_f16,
            k_emb_bwd,
            k_rms_fwd,
            k_swiglu_fwd,
            k_swiglu_bwd,
            k_res_add,
            k_norm_sq,
            module,
            step_count: 0,
            d_norm_sq,
            session,
        })
    }

    pub fn device(&self) -> &CudaDeviceInfo {
        self.session.device()
    }

    /// Returns the active training precision for this trainer.
    pub fn precision(&self) -> TrainingPrecision {
        self.precision
    }

    pub fn get_vram_info(&self) -> Result<(usize, usize), CudaError> {
        self.session.get_memory_info()
    }

    /// Register and upload parameter weights to GPU VRAM with zero-initialized optimizer states.
    ///
    /// When total weights and optimizer states exceed 50% of free VRAM (e.g. for a 1B model on 16GB GPU),
    /// automatically switches to Streamed Optimizer Mode: FP16 model weights and FP16 gradients stay on GPU,
    /// while FP32 master weights and AdamW states are kept in host memory and streamed through a single
    /// reusable GPU chunk buffer during optimizer updates.
    pub fn register_weights(
        &mut self,
        weights: &HashMap<String, Vec<f32>>,
    ) -> Result<(), CudaError> {
        let mut new_params = HashMap::new();
        let mut new_h_weights = HashMap::new();
        let mut new_h_m = HashMap::new();
        let mut new_h_v = HashMap::new();

        let is_f16 = self.precision == TrainingPrecision::Fp16;
        let total_elements: usize = weights.values().map(|v| v.len()).sum();
        let (free_vram, _) = self.session.get_memory_info()?;

        let naive_bytes_needed = total_elements * 16; // 4B master + 4B m + 4B v + 4B grad / f16 weights

        // Stream optimizer states when VRAM would be overcommitted
        let stream_opt = naive_bytes_needed > (free_vram * 5) / 10;

        let (chunk_w, chunk_m, chunk_v) = if stream_opt {
            let max_tensor_len = weights.values().map(|v| v.len()).max().unwrap_or(0);
            (
                Some(self.session.allocate_f32(max_tensor_len)?),
                Some(self.session.allocate_f32(max_tensor_len)?),
                Some(self.session.allocate_f32(max_tensor_len)?),
            )
        } else {
            (None, None, None)
        };

        for (name, vals) in weights {
            let len = vals.len();

            let (d_weight, d_m, d_v) = if is_f16 {
                // In FP16 mode: master weights and moments can be streamed from host when memory-constrained
                if stream_opt {
                    new_h_weights.insert(name.clone(), vals.clone());
                    new_h_m.insert(name.clone(), vec![0.0f32; len]);
                    new_h_v.insert(name.clone(), vec![0.0f32; len]);
                    (None, None, None)
                } else {
                    let dw = self.session.allocate_f32(len)?;
                    self.session.upload_f32(&dw, vals)?;
                    let dm = self.session.allocate_f32(len)?;
                    let dv = self.session.allocate_f32(len)?;
                    let zeros = vec![0.0f32; len];
                    self.session.upload_f32(&dm, &zeros)?;
                    self.session.upload_f32(&dv, &zeros)?;
                    (Some(dw), Some(dm), Some(dv))
                }
            } else {
                // In FP32 mode: model weights MUST always reside on GPU for forward/backward computation
                let dw = self.session.allocate_f32(len)?;
                self.session.upload_f32(&dw, vals)?;

                if stream_opt {
                    new_h_m.insert(name.clone(), vec![0.0f32; len]);
                    new_h_v.insert(name.clone(), vec![0.0f32; len]);
                    (Some(dw), None, None)
                } else {
                    let dm = self.session.allocate_f32(len)?;
                    let dv = self.session.allocate_f32(len)?;
                    let zeros = vec![0.0f32; len];
                    self.session.upload_f32(&dm, &zeros)?;
                    self.session.upload_f32(&dv, &zeros)?;
                    (Some(dw), Some(dm), Some(dv))
                }
            };

            let (d_weight_f16, d_grad) = if is_f16 {
                // FP16 model weight buffer: packed u32 words = len.div_ceil(2)
                let f16_elements = len.div_ceil(2);
                let d_wf16 = self.session.allocate_f32(f16_elements)?;
                // Gradient buffer is kept strictly in FP32 to prevent underflow and preserve precision
                let d_g = self.session.allocate_f32(len)?;

                // Safe conversion and packing without undefined behavior
                let h_f16: Vec<f16> = vals.iter().map(|&x| f16::from_f32(x)).collect();
                let packed = pack_f16_to_u32(&h_f16);
                self.session.upload_u32(&d_wf16, &packed)?;

                let zeros = vec![0.0f32; len];
                self.session.upload_f32(&d_g, &zeros)?;

                (Some(d_wf16), d_g)
            } else {
                let d_g = self.session.allocate_f32(len)?;
                let zeros = vec![0.0f32; len];
                self.session.upload_f32(&d_g, &zeros)?;
                (None, d_g)
            };

            new_params.insert(
                name.clone(),
                GpuParamState {
                    d_weight,
                    d_weight_f16,
                    d_grad,
                    d_m,
                    d_v,
                    len,
                },
            );
        }

        self.session.synchronize()?;

        // Atomic swap: only update trainer state once all GPU allocations succeed
        self.params = new_params;
        self.h_weights = new_h_weights;
        self.h_m = new_h_m;
        self.h_v = new_h_v;
        self.d_chunk_w = chunk_w;
        self.d_chunk_m = chunk_m;
        self.d_chunk_v = chunk_v;
        self.stream_optimizer = stream_opt;

        Ok(())
    }

    /// Forward pass through LM Head on GPU.
    pub fn forward_lm_head(
        &self,
        final_normed: &[f32],
        seq_len: usize,
        hs: usize,
        vs: usize,
    ) -> Result<Vec<f32>, CudaError> {
        if self.precision == TrainingPrecision::Fp16 {
            self.forward_lm_head_f16(final_normed, seq_len, hs, vs)
        } else {
            self.forward_lm_head_fp32(final_normed, seq_len, hs, vs)
        }
    }

    fn forward_lm_head_fp32(
        &self,
        final_normed: &[f32],
        seq_len: usize,
        hs: usize,
        vs: usize,
    ) -> Result<Vec<f32>, CudaError> {
        let lm_param = self.params.get("lm_head.weight").ok_or_else(|| {
            CudaError::KernelError("lm_head.weight not found in GPU parameters".into())
        })?;

        let d_normed = self.session.allocate_f32(seq_len * hs)?;
        let d_logits = self.session.allocate_f32(seq_len * vs)?;

        self.session.upload_f32(&d_normed, final_normed)?;

        let total = (seq_len * vs) as u32;
        let block_dim = 128u32;
        let grid_dim = total.div_ceil(block_dim);

        let mut arg_normed = d_normed.dptr;
        let mut arg_lm = lm_param.d_weight.as_ref().map(|b| b.dptr).unwrap_or(0);
        let mut arg_logits = d_logits.dptr;
        let mut arg_seq = seq_len as u32;
        let mut arg_hs = hs as u32;
        let mut arg_vs = vs as u32;

        let params: [*mut c_void; 16] = [
            &mut arg_normed as *mut u64 as *mut c_void,
            &mut arg_lm as *mut u64 as *mut c_void,
            &mut arg_logits as *mut u64 as *mut c_void,
            &mut arg_seq as *mut u32 as *mut c_void,
            &mut arg_hs as *mut u32 as *mut c_void,
            &mut arg_vs as *mut u32 as *mut c_void,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        ];

        unsafe {
            self.k_lm_fwd
                .launch((grid_dim, 1, 1), (block_dim, 1, 1), 0, params, 6)?;
        }
        self.session.synchronize()?;

        let mut host_logits = vec![0.0f32; seq_len * vs];
        self.session.download_f32(&d_logits, &mut host_logits)?;
        Ok(host_logits)
    }

    fn forward_lm_head_f16(
        &self,
        final_normed: &[f32],
        seq_len: usize,
        hs: usize,
        vs: usize,
    ) -> Result<Vec<f32>, CudaError> {
        let lm_param = self.params.get("lm_head.weight").ok_or_else(|| {
            CudaError::KernelError("lm_head.weight not found in GPU parameters".into())
        })?;
        let d_lm_f16 = lm_param.d_weight_f16.as_ref().ok_or_else(|| {
            CudaError::KernelError("FP16 model weight buffer not initialized".into())
        })?;

        // Allocate FP16 input buffer: (seq_len * hs + 1) / 2 floats = packed u32 words
        let normed_f16_elems = (seq_len * hs).div_ceil(2);
        let d_normed = self.session.allocate_f32(normed_f16_elems)?;
        let d_logits = self.session.allocate_f32(seq_len * vs)?; // FP32 logits output

        // Convert input to FP16 and safely upload without UB
        let h_normed_f16: Vec<f16> = final_normed.iter().map(|&x| f16::from_f32(x)).collect();
        let packed = pack_f16_to_u32(&h_normed_f16);
        self.session.upload_u32(&d_normed, &packed)?;

        let total = (seq_len * vs) as u32;
        let block_dim = 128u32;
        let grid_dim = total.div_ceil(block_dim);

        let mut arg_normed = d_normed.dptr;
        let mut arg_lm = d_lm_f16.dptr;
        let mut arg_logits = d_logits.dptr;
        let mut arg_seq = seq_len as u32;
        let mut arg_hs = hs as u32;
        let mut arg_vs = vs as u32;

        let params: [*mut c_void; 16] = [
            &mut arg_normed as *mut u64 as *mut c_void,
            &mut arg_lm as *mut u64 as *mut c_void,
            &mut arg_logits as *mut u64 as *mut c_void,
            &mut arg_seq as *mut u32 as *mut c_void,
            &mut arg_hs as *mut u32 as *mut c_void,
            &mut arg_vs as *mut u32 as *mut c_void,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        ];

        unsafe {
            self.k_lm_fwd_f16
                .launch((grid_dim, 1, 1), (block_dim, 1, 1), 0, params, 6)?;
        }
        self.session.synchronize()?;

        let mut host_logits = vec![0.0f32; seq_len * vs];
        self.session.download_f32(&d_logits, &mut host_logits)?;
        Ok(host_logits)
    }

    /// Backward pass through LM Head on GPU.
    pub fn backward_lm_head(
        &self,
        d_logits: &[f32],
        final_normed: &[f32],
        seq_len: usize,
        hs: usize,
        vs: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), CudaError> {
        if self.precision == TrainingPrecision::Fp16 {
            self.backward_lm_head_f16(d_logits, final_normed, seq_len, hs, vs)
        } else {
            self.backward_lm_head_fp32(d_logits, final_normed, seq_len, hs, vs)
        }
    }

    fn backward_lm_head_fp32(
        &self,
        d_logits: &[f32],
        final_normed: &[f32],
        seq_len: usize,
        hs: usize,
        vs: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), CudaError> {
        let lm_param = self.params.get("lm_head.weight").ok_or_else(|| {
            CudaError::KernelError("lm_head.weight not found in GPU parameters".into())
        })?;

        let d_dlogits = self.session.allocate_f32(seq_len * vs)?;
        let d_normed = self.session.allocate_f32(seq_len * hs)?;
        let d_dlm = self.session.allocate_f32(vs * hs)?;
        let d_dnormed = self.session.allocate_f32(seq_len * hs)?;

        self.session.upload_f32(&d_dlogits, d_logits)?;
        self.session.upload_f32(&d_normed, final_normed)?;

        let block_dim = 128u32;

        // Compute d_lm_head on GPU
        let total_w = (vs * hs) as u32;
        let grid_w = total_w.div_ceil(block_dim);
        let mut arg_dlog = d_dlogits.dptr;
        let mut arg_norm = d_normed.dptr;
        let mut arg_out_dlm = d_dlm.dptr;
        let mut arg_seq = seq_len as u32;
        let mut arg_hs = hs as u32;
        let mut arg_vs = vs as u32;

        let params_w: [*mut c_void; 16] = [
            &mut arg_dlog as *mut u64 as *mut c_void,
            &mut arg_norm as *mut u64 as *mut c_void,
            &mut arg_out_dlm as *mut u64 as *mut c_void,
            &mut arg_seq as *mut u32 as *mut c_void,
            &mut arg_hs as *mut u32 as *mut c_void,
            &mut arg_vs as *mut u32 as *mut c_void,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        ];
        unsafe {
            self.k_lm_bwd_weight
                .launch((grid_w, 1, 1), (block_dim, 1, 1), 0, params_w, 6)?;
        }

        // Compute d_final_normed on GPU
        let total_in = (seq_len * hs) as u32;
        let grid_in = total_in.div_ceil(block_dim);
        let mut arg_lm = lm_param.d_weight.as_ref().map(|b| b.dptr).unwrap_or(0);
        let mut arg_out_dnorm = d_dnormed.dptr;

        let params_in: [*mut c_void; 16] = [
            &mut arg_dlog as *mut u64 as *mut c_void,
            &mut arg_lm as *mut u64 as *mut c_void,
            &mut arg_out_dnorm as *mut u64 as *mut c_void,
            &mut arg_seq as *mut u32 as *mut c_void,
            &mut arg_hs as *mut u32 as *mut c_void,
            &mut arg_vs as *mut u32 as *mut c_void,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        ];
        unsafe {
            self.k_lm_bwd_input
                .launch((grid_in, 1, 1), (block_dim, 1, 1), 0, params_in, 6)?;
        }

        self.session.synchronize()?;

        let mut host_dlm = vec![0.0f32; vs * hs];
        let mut host_dnorm = vec![0.0f32; seq_len * hs];
        self.session.download_f32(&d_dlm, &mut host_dlm)?;
        self.session.download_f32(&d_dnormed, &mut host_dnorm)?;

        Ok((host_dlm, host_dnorm))
    }

    fn backward_lm_head_f16(
        &self,
        d_logits: &[f32],
        final_normed: &[f32],
        seq_len: usize,
        hs: usize,
        vs: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), CudaError> {
        let lm_param = self.params.get("lm_head.weight").ok_or_else(|| {
            CudaError::KernelError("lm_head.weight not found in GPU parameters".into())
        })?;
        let d_lm_f16 = lm_param.d_weight_f16.as_ref().ok_or_else(|| {
            CudaError::KernelError("FP16 model weight buffer not initialized".into())
        })?;

        let dlog_elems = (seq_len * vs).div_ceil(2);
        let norm_elems = (seq_len * hs).div_ceil(2);
        let total_w = vs * hs;
        let total_in = seq_len * hs;

        let d_dlogits = self.session.allocate_f32(dlog_elems)?;
        let d_normed = self.session.allocate_f32(norm_elems)?;
        let d_dlm = self.session.allocate_f32(total_w)?;
        let d_dnormed = self.session.allocate_f32(total_in)?;

        // Upload FP16 inputs safely without UB
        let h_dlog_f16: Vec<f16> = d_logits.iter().map(|&x| f16::from_f32(x)).collect();
        let packed_dlog = pack_f16_to_u32(&h_dlog_f16);
        self.session.upload_u32(&d_dlogits, &packed_dlog)?;

        let h_norm_f16: Vec<f16> = final_normed.iter().map(|&x| f16::from_f32(x)).collect();
        let packed_norm = pack_f16_to_u32(&h_norm_f16);
        self.session.upload_u32(&d_normed, &packed_norm)?;

        let block_dim = 128u32;

        // 1. Compute d_lm_head directly into FP32 on GPU
        let grid_w = (total_w as u32).div_ceil(block_dim);
        let mut arg_dlog = d_dlogits.dptr;
        let mut arg_norm = d_normed.dptr;
        let mut arg_out_dlm = d_dlm.dptr;
        let mut arg_seq = seq_len as u32;
        let mut arg_hs = hs as u32;
        let mut arg_vs = vs as u32;

        let params_w: [*mut c_void; 16] = [
            &mut arg_dlog as *mut u64 as *mut c_void,
            &mut arg_norm as *mut u64 as *mut c_void,
            &mut arg_out_dlm as *mut u64 as *mut c_void,
            &mut arg_seq as *mut u32 as *mut c_void,
            &mut arg_hs as *mut u32 as *mut c_void,
            &mut arg_vs as *mut u32 as *mut c_void,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        ];
        unsafe {
            self.k_lm_bwd_weight_f16
                .launch((grid_w, 1, 1), (block_dim, 1, 1), 0, params_w, 6)?;
        }

        // 2. Compute d_final_normed directly into FP32 on GPU
        let grid_in = (total_in as u32).div_ceil(block_dim);
        let mut arg_lm = d_lm_f16.dptr;
        let mut arg_out_dnorm = d_dnormed.dptr;

        let params_in: [*mut c_void; 16] = [
            &mut arg_dlog as *mut u64 as *mut c_void,
            &mut arg_lm as *mut u64 as *mut c_void,
            &mut arg_out_dnorm as *mut u64 as *mut c_void,
            &mut arg_seq as *mut u32 as *mut c_void,
            &mut arg_hs as *mut u32 as *mut c_void,
            &mut arg_vs as *mut u32 as *mut c_void,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        ];
        unsafe {
            self.k_lm_bwd_input_f16
                .launch((grid_in, 1, 1), (block_dim, 1, 1), 0, params_in, 6)?;
        }

        self.session.synchronize()?;

        // Download full-precision FP32 gradients directly
        let mut host_dlm = vec![0.0f32; total_w];
        let mut host_dnorm = vec![0.0f32; total_in];
        self.session.download_f32(&d_dlm, &mut host_dlm)?;
        self.session.download_f32(&d_dnormed, &mut host_dnorm)?;

        Ok((host_dlm, host_dnorm))
    }

    /// Forward embedding lookup on GPU.
    pub fn forward_embedding(&self, tokens: &[u32], hs: usize) -> Result<Vec<f32>, CudaError> {
        let seq_len = tokens.len();
        let embed_param = self
            .params
            .get("model.embed_tokens.weight")
            .ok_or_else(|| {
                CudaError::KernelError(
                    "model.embed_tokens.weight not found in GPU parameters".into(),
                )
            })?;

        let is_f16 = self.precision == TrainingPrecision::Fp16;
        let dptr_embed = if is_f16 {
            embed_param
                .d_weight_f16
                .as_ref()
                .map(|b| b.dptr)
                .ok_or_else(|| CudaError::KernelError("FP16 embedding table not initialized".into()))?
        } else {
            embed_param
                .d_weight
                .as_ref()
                .map(|b| b.dptr)
                .ok_or_else(|| CudaError::KernelError("FP32 embedding table not initialized".into()))?
        };

        let d_tokens = self.session.allocate_f32(seq_len)?;
        let d_out = self.session.allocate_f32(seq_len * hs)?;

        self.session.upload_u32(&d_tokens, tokens)?;

        let total = (seq_len * hs) as u32;
        let block_dim = 128u32;
        let grid_dim = total.div_ceil(block_dim);

        let mut arg_tokens = d_tokens.dptr;
        let mut arg_embed = dptr_embed;
        let mut arg_out = d_out.dptr;
        let mut arg_seq = seq_len as u32;
        let mut arg_hs = hs as u32;

        let params: [*mut c_void; 16] = [
            &mut arg_tokens as *mut u64 as *mut c_void,
            &mut arg_embed as *mut u64 as *mut c_void,
            &mut arg_out as *mut u64 as *mut c_void,
            &mut arg_seq as *mut u32 as *mut c_void,
            &mut arg_hs as *mut u32 as *mut c_void,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        ];

        unsafe {
            if is_f16 {
                self.k_emb_fwd_f16
                    .launch((grid_dim, 1, 1), (block_dim, 1, 1), 0, params, 5)?;
            } else {
                self.k_emb_fwd
                    .launch((grid_dim, 1, 1), (block_dim, 1, 1), 0, params, 5)?;
            }
        }
        self.session.synchronize()?;

        let mut host_out = vec![0.0f32; seq_len * hs];
        self.session.download_f32(&d_out, &mut host_out)?;
        Ok(host_out)
    }

    /// Forward RMSNorm on GPU.
    pub fn forward_rmsnorm(
        &self,
        weight_name: &str,
        input: &[f32],
        seq_len: usize,
        hs: usize,
        eps: f32,
    ) -> Result<Vec<f32>, CudaError> {
        let weight_param = self.params.get(weight_name).ok_or_else(|| {
            CudaError::KernelError(format!(
                "Weight '{}' not found in GPU parameters",
                weight_name
            ))
        })?;
        let dptr_w = if self.precision == TrainingPrecision::Fp16 {
            weight_param
                .d_weight_f16
                .as_ref()
                .map(|b| b.dptr)
                .or_else(|| weight_param.d_weight.as_ref().map(|b| b.dptr))
                .ok_or_else(|| CudaError::KernelError(format!("Weight buffer for '{weight_name}' not found")))?
        } else {
            weight_param
                .d_weight
                .as_ref()
                .map(|b| b.dptr)
                .ok_or_else(|| CudaError::KernelError(format!("Weight buffer for '{weight_name}' not found")))?
        };

        let d_in = self.session.allocate_f32(seq_len * hs)?;
        let d_out = self.session.allocate_f32(seq_len * hs)?;

        self.session.upload_f32(&d_in, input)?;

        let block_dim = 128u32;
        let grid_dim = (seq_len as u32).div_ceil(block_dim);

        let mut arg_in = d_in.dptr;
        let mut arg_w = dptr_w;
        let mut arg_out = d_out.dptr;
        let mut arg_eps = eps;
        let mut arg_seq = seq_len as u32;
        let mut arg_hs = hs as u32;

        let params: [*mut c_void; 16] = [
            &mut arg_in as *mut u64 as *mut c_void,
            &mut arg_w as *mut u64 as *mut c_void,
            &mut arg_out as *mut u64 as *mut c_void,
            &mut arg_eps as *mut f32 as *mut c_void,
            &mut arg_seq as *mut u32 as *mut c_void,
            &mut arg_hs as *mut u32 as *mut c_void,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        ];

        unsafe {
            self.k_rms_fwd
                .launch((grid_dim, 1, 1), (block_dim, 1, 1), 0, params, 6)?;
        }
        self.session.synchronize()?;

        let mut host_out = vec![0.0f32; seq_len * hs];
        self.session.download_f32(&d_out, &mut host_out)?;
        Ok(host_out)
    }

    /// Forward Linear Projection on GPU: out = input @ weight^T.
    pub fn forward_linear(
        &self,
        weight_name: &str,
        input: &[f32],
        seq_len: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<Vec<f32>, CudaError> {
        let weight_param = self.params.get(weight_name).ok_or_else(|| {
            CudaError::KernelError(format!(
                "Weight '{}' not found in GPU parameters",
                weight_name
            ))
        })?;

        let is_f16 = self.precision == TrainingPrecision::Fp16;
        let (dptr_w, is_f16_buf) = if is_f16 {
            if let Some(ref bw16) = weight_param.d_weight_f16 {
                (bw16.dptr, true)
            } else if let Some(ref bw) = weight_param.d_weight {
                (bw.dptr, false)
            } else {
                return Err(CudaError::KernelError(format!("Weight '{weight_name}' has no GPU buffer")));
            }
        } else {
            let dw = weight_param.d_weight.as_ref().ok_or_else(|| {
                CudaError::KernelError(format!("Weight '{weight_name}' has no GPU buffer"))
            })?;
            (dw.dptr, false)
        };

        let d_in = self.session.allocate_f32(seq_len * in_dim)?;
        let d_out = self.session.allocate_f32(seq_len * out_dim)?;

        self.session.upload_f32(&d_in, input)?;

        let total = (seq_len * out_dim) as u32;
        let block_dim = 128u32;
        let grid_dim = total.div_ceil(block_dim);

        let mut arg_in = d_in.dptr;
        let mut arg_w = dptr_w;
        let mut arg_out = d_out.dptr;
        let mut arg_seq = seq_len as u32;
        let mut arg_in_dim = in_dim as u32;
        let mut arg_out_dim = out_dim as u32;

        let params: [*mut c_void; 16] = [
            &mut arg_in as *mut u64 as *mut c_void,
            &mut arg_w as *mut u64 as *mut c_void,
            &mut arg_out as *mut u64 as *mut c_void,
            &mut arg_seq as *mut u32 as *mut c_void,
            &mut arg_in_dim as *mut u32 as *mut c_void,
            &mut arg_out_dim as *mut u32 as *mut c_void,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        ];

        unsafe {
            if is_f16_buf {
                self.k_lm_fwd_f16
                    .launch((grid_dim, 1, 1), (block_dim, 1, 1), 0, params, 6)?;
            } else {
                self.k_lm_fwd
                    .launch((grid_dim, 1, 1), (block_dim, 1, 1), 0, params, 6)?;
            }
        }
        self.session.synchronize()?;

        let mut host_out = vec![0.0f32; seq_len * out_dim];
        self.session.download_f32(&d_out, &mut host_out)?;
        Ok(host_out)
    }

    /// Backward Linear Projection on GPU: computes d_weight and d_input.
    pub fn backward_linear(
        &self,
        weight_name: &str,
        d_out: &[f32],
        input: &[f32],
        seq_len: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), CudaError> {
        let weight_param = self.params.get(weight_name).ok_or_else(|| {
            CudaError::KernelError(format!(
                "Weight '{}' not found in GPU parameters",
                weight_name
            ))
        })?;

        let is_f16 = self.precision == TrainingPrecision::Fp16;
        let (dptr_w, is_f16_buf) = if is_f16 {
            if let Some(ref bw16) = weight_param.d_weight_f16 {
                (bw16.dptr, true)
            } else if let Some(ref bw) = weight_param.d_weight {
                (bw.dptr, false)
            } else {
                return Err(CudaError::KernelError(format!("Weight '{weight_name}' has no GPU buffer")));
            }
        } else {
            let dw = weight_param.d_weight.as_ref().ok_or_else(|| {
                CudaError::KernelError(format!("Weight '{weight_name}' has no GPU buffer"))
            })?;
            (dw.dptr, false)
        };

        let d_dout = self.session.allocate_f32(seq_len * out_dim)?;
        let d_in = self.session.allocate_f32(seq_len * in_dim)?;
        let d_dw = self.session.allocate_f32(out_dim * in_dim)?;
        let d_din = self.session.allocate_f32(seq_len * in_dim)?;

        self.session.upload_f32(&d_dout, d_out)?;
        self.session.upload_f32(&d_in, input)?;

        let block_dim = 128u32;

        // 1. d_weight: [out_dim, in_dim]
        let total_w = (out_dim * in_dim) as u32;
        let grid_w = total_w.div_ceil(block_dim);
        let mut arg_dout = d_dout.dptr;
        let mut arg_in = d_in.dptr;
        let mut arg_dw = d_dw.dptr;
        let mut arg_seq = seq_len as u32;
        let mut arg_in_dim = in_dim as u32;
        let mut arg_out_dim = out_dim as u32;

        let params_w: [*mut c_void; 16] = [
            &mut arg_dout as *mut u64 as *mut c_void,
            &mut arg_in as *mut u64 as *mut c_void,
            &mut arg_dw as *mut u64 as *mut c_void,
            &mut arg_seq as *mut u32 as *mut c_void,
            &mut arg_in_dim as *mut u32 as *mut c_void,
            &mut arg_out_dim as *mut u32 as *mut c_void,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        ];
        unsafe {
            if is_f16_buf {
                self.k_lm_bwd_weight_f16
                    .launch((grid_w, 1, 1), (block_dim, 1, 1), 0, params_w, 6)?;
            } else {
                self.k_lm_bwd_weight
                    .launch((grid_w, 1, 1), (block_dim, 1, 1), 0, params_w, 6)?;
            }
        }

        // 2. d_input: [seq_len, in_dim]
        let total_in = (seq_len * in_dim) as u32;
        let grid_in = total_in.div_ceil(block_dim);
        let mut arg_w = dptr_w;
        let mut arg_din = d_din.dptr;

        let params_in: [*mut c_void; 16] = [
            &mut arg_dout as *mut u64 as *mut c_void,
            &mut arg_w as *mut u64 as *mut c_void,
            &mut arg_din as *mut u64 as *mut c_void,
            &mut arg_seq as *mut u32 as *mut c_void,
            &mut arg_in_dim as *mut u32 as *mut c_void,
            &mut arg_out_dim as *mut u32 as *mut c_void,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        ];
        unsafe {
            if is_f16_buf {
                self.k_lm_bwd_input_f16
                    .launch((grid_in, 1, 1), (block_dim, 1, 1), 0, params_in, 6)?;
            } else {
                self.k_lm_bwd_input
                    .launch((grid_in, 1, 1), (block_dim, 1, 1), 0, params_in, 6)?;
            }
        }

        self.session.synchronize()?;

        let mut host_dw = vec![0.0f32; out_dim * in_dim];
        let mut host_din = vec![0.0f32; seq_len * in_dim];
        self.session.download_f32(&d_dw, &mut host_dw)?;
        self.session.download_f32(&d_din, &mut host_din)?;

        Ok((host_dw, host_din))
    }

    /// Forward SwiGLU on GPU: out = silu(gate) * up.
    pub fn forward_swiglu(&self, gate: &[f32], up: &[f32]) -> Result<Vec<f32>, CudaError> {
        let n = gate.len();
        let d_gate = self.session.allocate_f32(n)?;
        let d_up = self.session.allocate_f32(n)?;
        let d_out = self.session.allocate_f32(n)?;

        self.session.upload_f32(&d_gate, gate)?;
        self.session.upload_f32(&d_up, up)?;

        let block_dim = 128u32;
        let grid_dim = (n as u32).div_ceil(block_dim);

        let mut arg_gate = d_gate.dptr;
        let mut arg_up = d_up.dptr;
        let mut arg_out = d_out.dptr;
        let mut arg_n = n as u32;

        let params: [*mut c_void; 16] = [
            &mut arg_gate as *mut u64 as *mut c_void,
            &mut arg_up as *mut u64 as *mut c_void,
            &mut arg_out as *mut u64 as *mut c_void,
            &mut arg_n as *mut u32 as *mut c_void,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        ];
        unsafe {
            self.k_swiglu_fwd
                .launch((grid_dim, 1, 1), (block_dim, 1, 1), 0, params, 4)?;
        }
        self.session.synchronize()?;

        let mut host_out = vec![0.0f32; n];
        self.session.download_f32(&d_out, &mut host_out)?;
        Ok(host_out)
    }

    /// Residual Add on GPU: out = a + b.
    pub fn add_residual(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, CudaError> {
        let n = a.len();
        let d_a = self.session.allocate_f32(n)?;
        let d_b = self.session.allocate_f32(n)?;
        let d_out = self.session.allocate_f32(n)?;

        self.session.upload_f32(&d_a, a)?;
        self.session.upload_f32(&d_b, b)?;

        let block_dim = 128u32;
        let grid_dim = (n as u32).div_ceil(block_dim);

        let mut arg_a = d_a.dptr;
        let mut arg_b = d_b.dptr;
        let mut arg_out = d_out.dptr;
        let mut arg_n = n as u32;

        let params: [*mut c_void; 16] = [
            &mut arg_a as *mut u64 as *mut c_void,
            &mut arg_b as *mut u64 as *mut c_void,
            &mut arg_out as *mut u64 as *mut c_void,
            &mut arg_n as *mut u32 as *mut c_void,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        ];
        unsafe {
            self.k_res_add
                .launch((grid_dim, 1, 1), (block_dim, 1, 1), 0, params, 4)?;
        }
        self.session.synchronize()?;

        let mut host_out = vec![0.0f32; n];
        self.session.download_f32(&d_out, &mut host_out)?;
        Ok(host_out)
    }

    /// Accumulate a batch sample gradient into the GPU gradient buffer.
    pub fn accumulate_gradient(&self, name: &str, grad: &[f32]) -> Result<(), CudaError> {
        let param = self.params.get(name).ok_or_else(|| {
            CudaError::KernelError(format!("Parameter '{name}' not found on GPU"))
        })?;
        if grad.len() != param.len {
            return Err(CudaError::KernelError(format!(
                "Gradient length mismatch for '{name}': expected {}, got {}",
                param.len,
                grad.len()
            )));
        }

        let block_dim = 128u32;
        let grid_dim = (param.len as u32).div_ceil(block_dim);

        // Gradients are always accumulated in FP32 to prevent underflow and preserve precision
        let d_sample = self.session.allocate_f32(grad.len())?;
        self.session.upload_f32(&d_sample, grad)?;

        let mut arg_accum = param.d_grad.dptr;
        let mut arg_sample = d_sample.dptr;
        let mut arg_n = param.len as u32;

        let mut params = [std::ptr::null_mut(); 16];
        params[0] = &mut arg_accum as *mut u64 as *mut c_void;
        params[1] = &mut arg_sample as *mut u64 as *mut c_void;
        params[2] = &mut arg_n as *mut u32 as *mut c_void;
        unsafe {
            self.k_accum_grads
                .launch((grid_dim, 1, 1), (block_dim, 1, 1), 0, params, 3)?;
        }
        self.session.synchronize()?;
        Ok(())
    }

    /// Scale accumulated gradients by 1.0 / count across all parameters on GPU.
    pub fn scale_accumulated_gradients(&self, count: usize) -> Result<(), CudaError> {
        let scale = 1.0f32 / (count.max(1) as f32);
        let block_dim = 128u32;

        for param in self.params.values() {
            let grid_dim = (param.len as u32).div_ceil(block_dim);
            let mut arg_grad = param.d_grad.dptr;
            let mut arg_scale = scale;
            let mut arg_n = param.len as u32;

            let mut params = [std::ptr::null_mut(); 16];
            params[0] = &mut arg_grad as *mut u64 as *mut c_void;
            params[1] = &mut arg_scale as *mut f32 as *mut c_void;
            params[2] = &mut arg_n as *mut u32 as *mut c_void;

            unsafe {
                self.k_scale_grads
                    .launch((grid_dim, 1, 1), (block_dim, 1, 1), 0, params, 3)?;
            }
        }
        self.session.synchronize()?;
        Ok(())
    }

    /// Execute AdamW optimization step across all parameters directly on GPU.
    /// Returns Ok(true) if step was executed, Ok(false) if skipped due to non-finite gradients.
    pub fn step_adamw(
        &mut self,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        max_grad_norm: f32,
    ) -> Result<bool, CudaError> {
        // High-performance GPU-side global norm reduction: sum of squares across all parameters
        // Explicitly zero accumulator on GPU before launching reduction kernels
        let zero_sq = [0.0f32];
        self.session.upload_f32(&self.d_norm_sq, &zero_sq)?;

        let block_dim = 128u32;
        for param in self.params.values() {
            let grid_dim = (param.len as u32).div_ceil(block_dim);
            let mut arg_grad = param.d_grad.dptr;
            let mut arg_out_sq = self.d_norm_sq.dptr;
            let mut arg_n = param.len as u32;

            let mut params = [std::ptr::null_mut(); 16];
            params[0] = &mut arg_grad as *mut u64 as *mut c_void;
            params[1] = &mut arg_out_sq as *mut u64 as *mut c_void;
            params[2] = &mut arg_n as *mut u32 as *mut c_void;

            unsafe {
                self.k_norm_sq
                    .launch((grid_dim, 1, 1), (block_dim, 1, 1), 0, params, 3)?;
            }
        }
        self.session.synchronize()?;

        let mut host_sq = [0.0f32];
        self.session.download_f32(&self.d_norm_sq, &mut host_sq)?;
        let total_norm_sq = host_sq[0];
        let total_norm = total_norm_sq.sqrt();

        // Non-finite gradient safeguard: if NaN or Inf, zero gradients, do NOT update weights, do NOT increment step_count!
        if !total_norm.is_finite() {
            eprintln!("[CUDA AdamW] Non-finite global gradient norm ({total_norm}) detected. Skipping optimizer step.");
            for param in self.params.values() {
                let zeros = vec![0.0f32; param.len];
                self.session.upload_f32(&param.d_grad, &zeros)?;
            }
            return Ok(false);
        }

        // Only increment step_count once gradients are confirmed finite!
        self.step_count += 1;
        let bc1 = 1.0 - beta1.powi(self.step_count as i32);
        let bc2 = 1.0 - beta2.powi(self.step_count as i32);
        let is_f16 = self.precision == TrainingPrecision::Fp16;

        let clip_scale = if total_norm > max_grad_norm && total_norm > 0.0 {
            max_grad_norm / total_norm
        } else {
            1.0f32
        };

        let block_dim = 128u32;

        for (name, param) in &self.params {
            let grid_dim = (param.len as u32).div_ceil(block_dim);

            if is_f16 {
                let (arg_w, arg_m, arg_v) = if self.stream_optimizer {
                    let d_cw = self.d_chunk_w.as_ref().unwrap();
                    let d_cm = self.d_chunk_m.as_ref().unwrap();
                    let d_cv = self.d_chunk_v.as_ref().unwrap();

                    let hw = self.h_weights.get(name).unwrap();
                    let hm = self.h_m.get(name).unwrap();
                    let hv = self.h_v.get(name).unwrap();

                    self.session.upload_f32_slice(d_cw, hw)?;
                    self.session.upload_f32_slice(d_cm, hm)?;
                    self.session.upload_f32_slice(d_cv, hv)?;

                    (d_cw.dptr, d_cm.dptr, d_cv.dptr)
                } else {
                    (
                        param.d_weight.as_ref().unwrap().dptr,
                        param.d_m.as_ref().unwrap().dptr,
                        param.d_v.as_ref().unwrap().dptr,
                    )
                };

                let mut arg_w = arg_w;
                let mut arg_wf16 = param.d_weight_f16.as_ref().unwrap().dptr;
                let mut arg_g = param.d_grad.dptr;
                let mut arg_m = arg_m;
                let mut arg_v = arg_v;
                let mut arg_lr = lr;
                let mut arg_b1 = beta1;
                let mut arg_b2 = beta2;
                let mut arg_eps = eps;
                let mut arg_wd = weight_decay;
                let mut arg_clip = clip_scale;
                let mut arg_bc1 = bc1;
                let mut arg_bc2 = bc2;
                let mut arg_n = param.len as u32;

                let params: [*mut c_void; 16] = [
                    &mut arg_w as *mut u64 as *mut c_void,
                    &mut arg_wf16 as *mut u64 as *mut c_void,
                    &mut arg_g as *mut u64 as *mut c_void,
                    &mut arg_m as *mut u64 as *mut c_void,
                    &mut arg_v as *mut u64 as *mut c_void,
                    &mut arg_lr as *mut f32 as *mut c_void,
                    &mut arg_b1 as *mut f32 as *mut c_void,
                    &mut arg_b2 as *mut f32 as *mut c_void,
                    &mut arg_eps as *mut f32 as *mut c_void,
                    &mut arg_wd as *mut f32 as *mut c_void,
                    &mut arg_clip as *mut f32 as *mut c_void,
                    &mut arg_bc1 as *mut f32 as *mut c_void,
                    &mut arg_bc2 as *mut f32 as *mut c_void,
                    &mut arg_n as *mut u32 as *mut c_void,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                ];

                unsafe {
                    self.k_adamw_step_f16.launch(
                        (grid_dim, 1, 1),
                        (block_dim, 1, 1),
                        0,
                        params,
                        14,
                    )?;
                }

                if self.stream_optimizer {
                    let d_cw = self.d_chunk_w.as_ref().unwrap();
                    let d_cm = self.d_chunk_m.as_ref().unwrap();
                    let d_cv = self.d_chunk_v.as_ref().unwrap();

                    let hw = self.h_weights.get_mut(name).unwrap();
                    let hm = self.h_m.get_mut(name).unwrap();
                    let hv = self.h_v.get_mut(name).unwrap();

                    self.session.download_f32_slice(d_cw, hw)?;
                    self.session.download_f32_slice(d_cm, hm)?;
                    self.session.download_f32_slice(d_cv, hv)?;
                }

                // Reset gradient buffer
                let zeros = vec![0.0f32; param.len];
                self.session.upload_f32(&param.d_grad, &zeros)?;
            } else {
                let (arg_w, arg_m, arg_v) = if self.stream_optimizer {
                    let d_cw = self.d_chunk_w.as_ref().unwrap();
                    let d_cm = self.d_chunk_m.as_ref().unwrap();
                    let d_cv = self.d_chunk_v.as_ref().unwrap();

                    let hw = self.h_weights.get(name).unwrap();
                    let hm = self.h_m.get(name).unwrap();
                    let hv = self.h_v.get(name).unwrap();

                    self.session.upload_f32_slice(d_cw, hw)?;
                    self.session.upload_f32_slice(d_cm, hm)?;
                    self.session.upload_f32_slice(d_cv, hv)?;

                    (d_cw.dptr, d_cm.dptr, d_cv.dptr)
                } else {
                    (
                        param.d_weight.as_ref().unwrap().dptr,
                        param.d_m.as_ref().unwrap().dptr,
                        param.d_v.as_ref().unwrap().dptr,
                    )
                };

                let mut arg_w = arg_w;
                let mut arg_g = param.d_grad.dptr;
                let mut arg_m = arg_m;
                let mut arg_v = arg_v;
                let mut arg_lr = lr;
                let mut arg_b1 = beta1;
                let mut arg_b2 = beta2;
                let mut arg_eps = eps;
                let mut arg_wd = weight_decay;
                let mut arg_clip = clip_scale;
                let mut arg_bc1 = bc1;
                let mut arg_bc2 = bc2;
                let mut arg_n = param.len as u32;

                let mut params = [std::ptr::null_mut(); 16];
                params[0] = &mut arg_w as *mut u64 as *mut c_void;
                params[1] = &mut arg_g as *mut u64 as *mut c_void;
                params[2] = &mut arg_m as *mut u64 as *mut c_void;
                params[3] = &mut arg_v as *mut u64 as *mut c_void;
                params[4] = &mut arg_lr as *mut f32 as *mut c_void;
                params[5] = &mut arg_b1 as *mut f32 as *mut c_void;
                params[6] = &mut arg_b2 as *mut f32 as *mut c_void;
                params[7] = &mut arg_eps as *mut f32 as *mut c_void;
                params[8] = &mut arg_wd as *mut f32 as *mut c_void;
                params[9] = &mut arg_clip as *mut f32 as *mut c_void;
                params[10] = &mut arg_bc1 as *mut f32 as *mut c_void;
                params[11] = &mut arg_bc2 as *mut f32 as *mut c_void;
                params[12] = &mut arg_n as *mut u32 as *mut c_void;

                unsafe {
                    self.k_adamw_step
                        .launch((grid_dim, 1, 1), (block_dim, 1, 1), 0, params, 13)?;
                }

                if self.stream_optimizer {
                    let d_cw = self.d_chunk_w.as_ref().unwrap();
                    let d_cm = self.d_chunk_m.as_ref().unwrap();
                    let d_cv = self.d_chunk_v.as_ref().unwrap();

                    let hw = self.h_weights.get_mut(name).unwrap();
                    let hm = self.h_m.get_mut(name).unwrap();
                    let hv = self.h_v.get_mut(name).unwrap();

                    self.session.download_f32_slice(d_cw, hw)?;
                    self.session.download_f32_slice(d_cm, hm)?;
                    self.session.download_f32_slice(d_cv, hv)?;
                }

                let zeros = vec![0.0f32; param.len];
                self.session.upload_f32(&param.d_grad, &zeros)?;
            }
        }

        self.session.synchronize()?;
        Ok(true)
    }

    /// Retrieve all updated weights from GPU back to host.
    pub fn download_weights(&self) -> Result<HashMap<String, Vec<f32>>, CudaError> {
        if self.stream_optimizer {
            return Ok(self.h_weights.clone());
        }
        let mut out = HashMap::new();
        for (name, param) in &self.params {
            let mut vals = vec![0.0f32; param.len];
            if let Some(ref dw) = param.d_weight {
                self.session.download_f32(dw, &mut vals)?;
            }
            out.insert(name.clone(), vals);
        }
        Ok(out)
    }

    /// Export optimizer states (m, v) for persistent checkpointing.
    pub fn export_optimizer_state(&self) -> Result<OptimizerStateMap, CudaError> {
        let mut out = HashMap::new();
        if self.stream_optimizer {
            for (name, m) in &self.h_m {
                let v = self
                    .h_v
                    .get(name)
                    .cloned()
                    .unwrap_or_else(|| vec![0.0f32; m.len()]);
                out.insert(name.clone(), (m.clone(), v));
            }
        } else {
            for (name, param) in &self.params {
                let mut m = vec![0.0f32; param.len];
                let mut v = vec![0.0f32; param.len];
                if let (Some(ref dm), Some(ref dv)) = (&param.d_m, &param.d_v) {
                    self.session.download_f32(dm, &mut m)?;
                    self.session.download_f32(dv, &mut v)?;
                }
                out.insert(name.clone(), (m, v));
            }
        }
        Ok(out)
    }

    /// Restore optimizer states (m, v) when resuming from a checkpoint.
    pub fn load_optimizer_state(&mut self, state: &OptimizerStateMap) -> Result<(), CudaError> {
        for (name, (m, v)) in state {
            let param = self.params.get(name).ok_or_else(|| {
                CudaError::KernelError(format!("Optimizer state parameter '{name}' not found on GPU"))
            })?;
            if m.len() != param.len || v.len() != param.len {
                return Err(CudaError::KernelError(format!(
                    "Optimizer state length mismatch for '{name}': param.len={}, m.len={}, v.len={}",
                    param.len,
                    m.len(),
                    v.len()
                )));
            }

            if self.stream_optimizer {
                if let Some(hm) = self.h_m.get_mut(name) {
                    *hm = m.clone();
                }
                if let Some(hv) = self.h_v.get_mut(name) {
                    *hv = v.clone();
                }
            } else if let (Some(ref dm), Some(ref dv)) = (&param.d_m, &param.d_v) {
                self.session.upload_f32(dm, m)?;
                self.session.upload_f32(dv, v)?;
            }
        }
        Ok(())
    }

    pub fn get_step_count(&self) -> u64 {
        self.step_count
    }

    pub fn set_step_count(&mut self, step: u64) {
        self.step_count = step;
    }

    pub fn embedding_bwd_kernel(&self) -> &CudaKernel {
        &self.k_emb_bwd
    }

    pub fn swiglu_bwd_kernel(&self) -> &CudaKernel {
        &self.k_swiglu_bwd
    }

    pub fn module_handle(&self) -> &CudaModuleHandle {
        &self.module
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trainer::{AdamWHyperparams, DynamicAdamW};

    #[test]
    fn test_gpu_cpu_adamw_numerical_equivalence() {
        let mut gpu_trainer = match CudaTrainer::new_with_precision(0, TrainingPrecision::Fp32) {
            Ok(t) => t,
            Err(_) => {
                eprintln!("[SKIP] CUDA device not available on this machine.");
                return;
            }
        };

        let n = 256;
        let mut initial_weights = HashMap::new();
        let mut w1 = vec![0.0f32; n];
        let mut w2 = vec![0.0f32; n];
        for i in 0..n {
            w1[i] = (i as f32 * 0.05).sin() * 0.1;
            w2[i] = (i as f32 * 0.03).cos() * 0.1;
        }
        initial_weights.insert("layer1.weight".to_string(), w1.clone());
        initial_weights.insert("layer2.weight".to_string(), w2.clone());

        gpu_trainer.register_weights(&initial_weights).unwrap();

        let mut cpu_weights = initial_weights.clone();
        let mut cpu_optimizer = DynamicAdamW::new();

        let hp = AdamWHyperparams {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            max_grad_norm: 1.0,
        };

        // Run 5 steps with deterministic synthetic gradients
        for step in 1..=5 {
            let mut g1 = vec![0.0f32; n];
            let mut g2 = vec![0.0f32; n];
            for i in 0..n {
                g1[i] = ((step * 100 + i) as f32 * 0.01).sin() * 0.05;
                g2[i] = ((step * 100 + i) as f32 * 0.02).cos() * 0.05;
            }
            let mut grads = HashMap::new();
            grads.insert("layer1.weight".to_string(), g1.clone());
            grads.insert("layer2.weight".to_string(), g2.clone());

            // CPU step
            let cpu_stepped = cpu_optimizer.step(&mut cpu_weights, &grads, &hp);
            assert!(cpu_stepped);

            // GPU step
            gpu_trainer.accumulate_gradient("layer1.weight", &g1).unwrap();
            gpu_trainer.accumulate_gradient("layer2.weight", &g2).unwrap();
            let gpu_stepped = gpu_trainer
                .step_adamw(hp.lr, hp.beta1, hp.beta2, hp.eps, hp.weight_decay, hp.max_grad_norm)
                .unwrap();
            assert!(gpu_stepped);
        }

        let gpu_weights = gpu_trainer.download_weights().unwrap();

        for name in &["layer1.weight", "layer2.weight"] {
            let cpu_w = &cpu_weights[*name];
            let gpu_w = &gpu_weights[*name];
            assert_eq!(cpu_w.len(), gpu_w.len());
            for i in 0..cpu_w.len() {
                let diff = (cpu_w[i] - gpu_w[i]).abs();
                assert!(
                    diff < 1e-5,
                    "Numerical divergence in {} at index {}: CPU={}, GPU={}, diff={}",
                    name,
                    i,
                    cpu_w[i],
                    gpu_w[i],
                    diff
                );
            }
        }
    }

    #[test]
    fn test_gpu_adamw_non_finite_gradient_skips_step() {
        let mut gpu_trainer = match CudaTrainer::new_with_precision(0, TrainingPrecision::Fp32) {
            Ok(t) => t,
            Err(_) => return,
        };

        let n = 128;
        let mut initial_weights = HashMap::new();
        let w = vec![0.5f32; n];
        initial_weights.insert("test.weight".to_string(), w.clone());
        gpu_trainer.register_weights(&initial_weights).unwrap();

        // Accumulate NaN gradient
        let mut bad_g = vec![0.01f32; n];
        bad_g[10] = f32::NAN;
        gpu_trainer.accumulate_gradient("test.weight", &bad_g).unwrap();

        let stepped = gpu_trainer
            .step_adamw(1e-3, 0.9, 0.999, 1e-8, 0.01, 1.0)
            .unwrap();
        assert!(!stepped, "Step with NaN gradient must be skipped");
        assert_eq!(gpu_trainer.get_step_count(), 0, "Step count must not advance");

        let downloaded = gpu_trainer.download_weights().unwrap();
        assert_eq!(downloaded["test.weight"], w, "Weights must be unchanged after skipped step");
    }

    #[test]
    fn test_gpu_checkpoint_resume_equivalence() {
        let n = 128;
        let mut initial_weights = HashMap::new();
        let mut w = vec![0.0f32; n];
        for i in 0..n {
            w[i] = (i as f32 * 0.1).sin();
        }
        initial_weights.insert("dense.weight".to_string(), w.clone());

        // Run A: 10 steps continuously
        let mut trainer_a = match CudaTrainer::new_with_precision(0, TrainingPrecision::Fp32) {
            Ok(t) => t,
            Err(_) => return,
        };
        trainer_a.register_weights(&initial_weights).unwrap();

        let mut grads_history = Vec::new();
        for step in 1..=10 {
            let mut g = vec![0.0f32; n];
            for i in 0..n {
                g[i] = ((step * 10 + i) as f32 * 0.05).cos() * 0.02;
            }
            grads_history.push(g.clone());
            trainer_a.accumulate_gradient("dense.weight", &g).unwrap();
            trainer_a.step_adamw(1e-3, 0.9, 0.999, 1e-8, 0.01, 1.0).unwrap();
        }
        let weights_a = trainer_a.download_weights().unwrap();

        // Run B: 5 steps -> checkpoint -> resume -> 5 steps
        let mut trainer_b = CudaTrainer::new_with_precision(0, TrainingPrecision::Fp32).unwrap();
        trainer_b.register_weights(&initial_weights).unwrap();

        for g in &grads_history[0..5] {
            trainer_b.accumulate_gradient("dense.weight", g).unwrap();
            trainer_b.step_adamw(1e-3, 0.9, 0.999, 1e-8, 0.01, 1.0).unwrap();
        }

        // Save checkpoint
        let checkpoint_weights = trainer_b.download_weights().unwrap();
        let checkpoint_opt_state = trainer_b.export_optimizer_state().unwrap();
        let checkpoint_step = trainer_b.get_step_count();
        drop(trainer_b);

        // Resume in new trainer
        let mut trainer_c = CudaTrainer::new_with_precision(0, TrainingPrecision::Fp32).unwrap();
        trainer_c.register_weights(&checkpoint_weights).unwrap();
        trainer_c.load_optimizer_state(&checkpoint_opt_state).unwrap();
        trainer_c.set_step_count(checkpoint_step);

        for g in &grads_history[5..10] {
            trainer_c.accumulate_gradient("dense.weight", g).unwrap();
            trainer_c.step_adamw(1e-3, 0.9, 0.999, 1e-8, 0.01, 1.0).unwrap();
        }
        let weights_c = trainer_c.download_weights().unwrap();

        let wa = &weights_a["dense.weight"];
        let wc = &weights_c["dense.weight"];
        for i in 0..n {
            let diff = (wa[i] - wc[i]).abs();
            assert!(
                diff < 1e-6,
                "Checkpoint resume mismatch at index {}: continuous={}, resumed={}, diff={}",
                i,
                wa[i],
                wc[i],
                diff
            );
        }
    }
}
