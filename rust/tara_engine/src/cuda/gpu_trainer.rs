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
    k_accum_grads_f16: CudaKernel,
    k_scale_grads_f16: CudaKernel,
    k_adamw_step_f16: CudaKernel,
    k_lm_fwd_f16: CudaKernel,
    k_lm_bwd_weight_f16: CudaKernel,
    k_lm_bwd_input_f16: CudaKernel,
    // Transformer Layer Kernels
    k_emb_fwd: CudaKernel,
    k_emb_bwd: CudaKernel,
    k_rms_fwd: CudaKernel,
    k_swiglu_fwd: CudaKernel,
    k_swiglu_bwd: CudaKernel,
    k_res_add: CudaKernel,
    module: CudaModuleHandle,
    step_count: u64,
    // Context session (must drop LAST after all device resources are freed)
    session: CudaSession,
}

impl CudaTrainer {
    pub fn new(device_ordinal: i32) -> Result<Self, CudaError> {
        Self::new_with_precision(device_ordinal, TrainingPrecision::Fp32)
    }

    pub fn new_with_precision(
        device_ordinal: i32,
        precision: TrainingPrecision,
    ) -> Result<Self, CudaError> {
        let session = CudaSession::init(device_ordinal)?;
        let module = session.load_ptx_module(PTX_TARA_KERNELS)?;

        // Load FP32 kernels
        let k_accum_grads = module.get_kernel("accum_grads_kernel")?;
        let k_scale_grads = module.get_kernel("scale_grads_kernel")?;
        let k_adamw_step = module.get_kernel("adamw_step_kernel")?;
        let k_lm_fwd = module.get_kernel("lm_head_fwd_kernel")?;
        let k_lm_bwd_weight = module.get_kernel("lm_head_bwd_weight_kernel")?;
        let k_lm_bwd_input = module.get_kernel("lm_head_bwd_input_kernel")?;

        // Load FP16 kernels
        let k_accum_grads_f16 = module.get_kernel("accum_grads_f16_kernel")?;
        let k_scale_grads_f16 = module.get_kernel("scale_grads_f16_kernel")?;
        let k_adamw_step_f16 = module.get_kernel("adamw_step_mixed_f16_kernel")?;
        let k_lm_fwd_f16 = module.get_kernel("lm_head_fwd_f16_kernel")?;
        let k_lm_bwd_weight_f16 = module.get_kernel("lm_head_bwd_weight_f16_kernel")?;
        let k_lm_bwd_input_f16 = module.get_kernel("lm_head_bwd_input_f16_kernel")?;

        // Load Transformer Layer kernels
        let k_emb_fwd = module.get_kernel("embedding_fwd_kernel")?;
        let k_emb_bwd = module.get_kernel("embedding_bwd_kernel")?;
        let k_rms_fwd = module.get_kernel("rmsnorm_fwd_kernel")?;
        let k_swiglu_fwd = module.get_kernel("swiglu_fwd_kernel")?;
        let k_swiglu_bwd = module.get_kernel("swiglu_bwd_kernel")?;
        let k_res_add = module.get_kernel("residual_add_kernel")?;

        Ok(Self {
            params: HashMap::new(),
            precision,
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
            k_accum_grads_f16,
            k_scale_grads_f16,
            k_adamw_step_f16,
            k_lm_fwd_f16,
            k_lm_bwd_weight_f16,
            k_lm_bwd_input_f16,
            k_emb_fwd,
            k_emb_bwd,
            k_rms_fwd,
            k_swiglu_fwd,
            k_swiglu_bwd,
            k_res_add,
            module,
            step_count: 0,
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
        self.params.clear();
        self.h_weights.clear();
        self.h_m.clear();
        self.h_v.clear();
        self.d_chunk_w = None;
        self.d_chunk_m = None;
        self.d_chunk_v = None;

        let is_f16 = self.precision == TrainingPrecision::Fp16;
        let total_elements: usize = weights.values().map(|v| v.len()).sum();
        let (free_vram, _) = self.session.get_memory_info()?;

        let naive_bytes_needed = total_elements * 16; // 4B master + 4B m + 4B v + 4B grad / f16 weights

        // Stream optimizer states when VRAM would be overcommitted
        self.stream_optimizer = naive_bytes_needed > (free_vram * 5) / 10;

        if self.stream_optimizer {
            let max_tensor_len = weights.values().map(|v| v.len()).max().unwrap_or(0);
            self.d_chunk_w = Some(self.session.allocate_f32(max_tensor_len)?);
            self.d_chunk_m = Some(self.session.allocate_f32(max_tensor_len)?);
            self.d_chunk_v = Some(self.session.allocate_f32(max_tensor_len)?);
        }

        for (name, vals) in weights {
            let len = vals.len();

            let (d_weight, d_m, d_v) = if self.stream_optimizer {
                self.h_weights.insert(name.clone(), vals.clone());
                self.h_m.insert(name.clone(), vec![0.0f32; len]);
                self.h_v.insert(name.clone(), vec![0.0f32; len]);
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
            };

            let (d_weight_f16, d_grad) = if is_f16 {
                // FP16 model weight buffer: (len + 1) / 2 floats = len * 2 bytes
                let f16_elements = len.div_ceil(2);
                let d_wf16 = self.session.allocate_f32(f16_elements)?;
                let d_gf16 = self.session.allocate_f32(f16_elements)?;

                // Convert FP32 values to FP16 bytes and upload
                let h_f16: Vec<f16> = vals.iter().map(|&x| f16::from_f32(x)).collect();
                unsafe {
                    let ptr = h_f16.as_ptr() as *const f32;
                    let slice = std::slice::from_raw_parts(ptr, f16_elements);
                    self.session.upload_f32(&d_wf16, slice)?;
                }
                let zeros_f16 = vec![0.0f32; f16_elements];
                self.session.upload_f32(&d_gf16, &zeros_f16)?;

                (Some(d_wf16), d_gf16)
            } else {
                let d_g = self.session.allocate_f32(len)?;
                let zeros = vec![0.0f32; len];
                self.session.upload_f32(&d_g, &zeros)?;
                (None, d_g)
            };

            self.params.insert(
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

        // Allocate FP16 input buffer: (seq_len * hs + 1) / 2 floats
        let normed_f16_elems = (seq_len * hs).div_ceil(2);
        let d_normed = self.session.allocate_f32(normed_f16_elems)?;
        let d_logits = self.session.allocate_f32(seq_len * vs)?; // FP32 logits output

        // Convert input to FP16 and upload
        let h_normed_f16: Vec<f16> = final_normed.iter().map(|&x| f16::from_f32(x)).collect();
        unsafe {
            let ptr = h_normed_f16.as_ptr() as *const f32;
            let slice = std::slice::from_raw_parts(ptr, normed_f16_elems);
            self.session.upload_f32(&d_normed, slice)?;
        }

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
        let dlm_elems = (vs * hs).div_ceil(2);

        let d_dlogits = self.session.allocate_f32(dlog_elems)?;
        let d_normed = self.session.allocate_f32(norm_elems)?;
        let d_dlm = self.session.allocate_f32(dlm_elems)?;
        let d_dnormed = self.session.allocate_f32(norm_elems)?;

        // Upload FP16 inputs
        let h_dlog_f16: Vec<f16> = d_logits.iter().map(|&x| f16::from_f32(x)).collect();
        let h_norm_f16: Vec<f16> = final_normed.iter().map(|&x| f16::from_f32(x)).collect();
        unsafe {
            let p1 = h_dlog_f16.as_ptr() as *const f32;
            let s1 = std::slice::from_raw_parts(p1, dlog_elems);
            self.session.upload_f32(&d_dlogits, s1)?;

            let p2 = h_norm_f16.as_ptr() as *const f32;
            let s2 = std::slice::from_raw_parts(p2, norm_elems);
            self.session.upload_f32(&d_normed, s2)?;
        }

        let block_dim = 128u32;

        // 1. Compute d_lm_head in FP16 on GPU
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
            self.k_lm_bwd_weight_f16
                .launch((grid_w, 1, 1), (block_dim, 1, 1), 0, params_w, 6)?;
        }

        // 2. Compute d_final_normed in FP16 on GPU
        let total_in = (seq_len * hs) as u32;
        let grid_in = total_in.div_ceil(block_dim);
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

        // Download and convert results to FP32
        let mut host_dlm_f16 = vec![f16::ZERO; vs * hs];
        let mut host_dnorm_f16 = vec![f16::ZERO; seq_len * hs];
        unsafe {
            let p1 = host_dlm_f16.as_mut_ptr() as *mut f32;
            let s1 = std::slice::from_raw_parts_mut(p1, dlm_elems);
            self.session.download_f32(&d_dlm, s1)?;

            let p2 = host_dnorm_f16.as_mut_ptr() as *mut f32;
            let s2 = std::slice::from_raw_parts_mut(p2, norm_elems);
            self.session.download_f32(&d_dnormed, s2)?;
        }

        let host_dlm: Vec<f32> = host_dlm_f16.iter().map(|&x| x.to_f32()).collect();
        let host_dnorm: Vec<f32> = host_dnorm_f16.iter().map(|&x| x.to_f32()).collect();

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
        let dptr_embed = if self.precision == TrainingPrecision::Fp16 {
            embed_param
                .d_weight_f16
                .as_ref()
                .map(|b| b.dptr)
                .or_else(|| embed_param.d_weight.as_ref().map(|b| b.dptr))
                .unwrap_or(0)
        } else {
            embed_param.d_weight.as_ref().map(|b| b.dptr).unwrap_or(0)
        };

        let d_tokens = self.session.allocate_f32(seq_len)?;
        let d_out = self.session.allocate_f32(seq_len * hs)?;

        unsafe {
            let slice = std::slice::from_raw_parts(tokens.as_ptr() as *const f32, seq_len);
            self.session.upload_f32(&d_tokens, slice)?;
        }

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
            self.k_emb_fwd
                .launch((grid_dim, 1, 1), (block_dim, 1, 1), 0, params, 5)?;
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
        let dptr_w = weight_param.d_weight.as_ref().map(|b| b.dptr).unwrap_or(0);

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
        let dptr_w = weight_param.d_weight.as_ref().map(|b| b.dptr).unwrap_or(0);

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
            self.k_lm_fwd
                .launch((grid_dim, 1, 1), (block_dim, 1, 1), 0, params, 6)?;
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
        let dptr_w = weight_param.d_weight.as_ref().map(|b| b.dptr).unwrap_or(0);

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
            self.k_lm_bwd_weight
                .launch((grid_w, 1, 1), (block_dim, 1, 1), 0, params_w, 6)?;
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
            self.k_lm_bwd_input
                .launch((grid_in, 1, 1), (block_dim, 1, 1), 0, params_in, 6)?;
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
        if let Some(param) = self.params.get(name) {
            let block_dim = 128u32;
            let grid_dim = (param.len as u32).div_ceil(block_dim);

            if self.precision == TrainingPrecision::Fp16 {
                let f16_elems = grad.len().div_ceil(2);
                let d_sample = self.session.allocate_f32(f16_elems)?;
                let h_f16: Vec<f16> = grad.iter().map(|&x| f16::from_f32(x)).collect();
                unsafe {
                    let ptr = h_f16.as_ptr() as *const f32;
                    let slice = std::slice::from_raw_parts(ptr, f16_elems);
                    self.session.upload_f32(&d_sample, slice)?;
                }

                let mut arg_accum = param.d_grad.dptr;
                let mut arg_sample = d_sample.dptr;
                let mut arg_n = param.len as u32;

                let mut params = [std::ptr::null_mut(); 16];
                params[0] = &mut arg_accum as *mut u64 as *mut c_void;
                params[1] = &mut arg_sample as *mut u64 as *mut c_void;
                params[2] = &mut arg_n as *mut u32 as *mut c_void;
                unsafe {
                    self.k_accum_grads_f16.launch(
                        (grid_dim, 1, 1),
                        (block_dim, 1, 1),
                        0,
                        params,
                        3,
                    )?;
                }
            } else {
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
            }
            self.session.synchronize()?;
        }
        Ok(())
    }

    /// Scale accumulated gradients by 1.0 / count across all parameters on GPU.
    pub fn scale_accumulated_gradients(&self, count: usize) -> Result<(), CudaError> {
        let scale = 1.0f32 / (count.max(1) as f32);
        let block_dim = 128u32;
        let is_f16 = self.precision == TrainingPrecision::Fp16;

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
                if is_f16 {
                    self.k_scale_grads_f16.launch(
                        (grid_dim, 1, 1),
                        (block_dim, 1, 1),
                        0,
                        params,
                        3,
                    )?;
                } else {
                    self.k_scale_grads
                        .launch((grid_dim, 1, 1), (block_dim, 1, 1), 0, params, 3)?;
                }
            }
        }
        self.session.synchronize()?;
        Ok(())
    }

    /// Execute AdamW optimization step across all parameters directly on GPU.
    pub fn step_adamw(
        &mut self,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        max_grad_norm: f32,
    ) -> Result<(), CudaError> {
        self.step_count += 1;
        let bc1 = 1.0 - beta1.powi(self.step_count as i32);
        let bc2 = 1.0 - beta2.powi(self.step_count as i32);
        let is_f16 = self.precision == TrainingPrecision::Fp16;

        // Compute global gradient norm
        let mut total_norm_sq = 0.0f32;
        for param in self.params.values() {
            if is_f16 {
                let f16_elems = param.len.div_ceil(2);
                let mut h_f16 = vec![f16::ZERO; param.len];
                unsafe {
                    let ptr = h_f16.as_mut_ptr() as *mut f32;
                    let slice = std::slice::from_raw_parts_mut(ptr, f16_elems);
                    self.session.download_f32(&param.d_grad, slice)?;
                }
                for &g in &h_f16 {
                    let val = g.to_f32();
                    total_norm_sq += val * val;
                }
            } else {
                let mut host_grads = vec![0.0f32; param.len];
                self.session.download_f32(&param.d_grad, &mut host_grads)?;
                for &g in &host_grads {
                    total_norm_sq += g * g;
                }
            }
        }
        let total_norm = total_norm_sq.sqrt();
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

                // Reset FP16 gradients
                let f16_elems = param.len.div_ceil(2);
                let zeros_f16 = vec![0.0f32; f16_elems];
                self.session.upload_f32(&param.d_grad, &zeros_f16)?;
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
        Ok(())
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
        if self.stream_optimizer {
            for (name, (m, v)) in state {
                if let Some(hm) = self.h_m.get_mut(name) {
                    *hm = m.clone();
                }
                if let Some(hv) = self.h_v.get_mut(name) {
                    *hv = v.clone();
                }
            }
        } else {
            for (name, (m, v)) in state {
                if let Some(param) = self.params.get(name) {
                    if let (Some(ref dm), Some(ref dv)) = (&param.d_m, &param.d_v) {
                        self.session.upload_f32(dm, m)?;
                        self.session.upload_f32(dv, v)?;
                    }
                }
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
