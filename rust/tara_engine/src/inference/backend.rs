//! Pluggable backend dispatch for CPU and GPU execution.
//!
//! Provides genuine device abstractions for CPU host RAM and NVIDIA GPU VRAM:
//! - `DeviceTensor`: Cloneable, reference-counted enum encapsulating CPU `Arc<Vec<f32>>` and CUDA `Arc<CudaBuffer>`
//! - `DeviceBackend`: Pluggable trait supporting genuine device allocation and host <-> device transfers
//! - `CUDABackend`: Authentic CUDA device backend with process-wide cached `Arc<CudaSession>` per ordinal
//! - `CPUBackend`: Pure host RAM implementation
//! - `BackendRegistry`: Policy-driven backend discovery and strict/fallback selection

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use thiserror::Error;

use crate::cuda::driver::{CudaBuffer, CudaError, CudaSession};

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("CUDA error: {0}")]
    Cuda(#[from] CudaError),
    #[error("Device memory allocation failed: {0}")]
    AllocationFailed(String),
    #[error("Device transfer failed: {0}")]
    TransferFailed(String),
    #[error("Backend unavailable: {0}")]
    Unavailable(String),
    #[error("Invalid operation on backend: {0}")]
    InvalidOperation(String),
}

/// Represents a reference-counted tensor handle resident either in CPU host memory or GPU VRAM.
/// Cloning a `DeviceTensor` is O(1) and never copies underlying host vectors or GPU allocations.
#[derive(Debug, Clone)]
pub enum DeviceTensor {
    Cpu(Arc<Vec<f32>>),
    Cuda(Arc<CudaBuffer>),
}

impl DeviceTensor {
    pub fn from_cpu_vec(data: Vec<f32>) -> Self {
        Self::Cpu(Arc::new(data))
    }

    pub fn from_cpu_arc(data: Arc<Vec<f32>>) -> Self {
        Self::Cpu(data)
    }

    pub fn from_cuda_buffer(buf: CudaBuffer) -> Self {
        Self::Cuda(Arc::new(buf))
    }

    pub fn from_cuda_arc(buf: Arc<CudaBuffer>) -> Self {
        Self::Cuda(buf)
    }

    pub fn kind(&self) -> DeviceKind {
        match self {
            Self::Cpu(_) => DeviceKind::Cpu,
            Self::Cuda(_) => DeviceKind::Cuda,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Cpu(v) => v.len(),
            Self::Cuda(b) => b.len_elements,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn bytes(&self) -> usize {
        match self {
            Self::Cpu(v) => v.len() * std::mem::size_of::<f32>(),
            Self::Cuda(b) => b.bytes,
        }
    }

    pub fn is_cuda(&self) -> bool {
        matches!(self, Self::Cuda(_))
    }

    pub fn is_cpu(&self) -> bool {
        matches!(self, Self::Cpu(_))
    }

    pub fn as_cpu_slice(&self) -> Option<&[f32]> {
        match self {
            Self::Cpu(v) => Some(v.as_slice()),
            Self::Cuda(_) => None,
        }
    }

    pub fn as_cpu_arc(&self) -> Option<&Arc<Vec<f32>>> {
        match self {
            Self::Cpu(v) => Some(v),
            Self::Cuda(_) => None,
        }
    }

    pub fn as_cuda_buffer(&self) -> Option<&CudaBuffer> {
        match self {
            Self::Cpu(_) => None,
            Self::Cuda(b) => Some(b.as_ref()),
        }
    }

    pub fn as_cuda_arc(&self) -> Option<&Arc<CudaBuffer>> {
        match self {
            Self::Cpu(_) => None,
            Self::Cuda(b) => Some(b),
        }
    }

    /// Returns true if both `DeviceTensor` handles point to the exact same underlying allocation.
    pub fn ptr_eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Cpu(a), Self::Cpu(b)) => Arc::ptr_eq(a, b),
            (Self::Cuda(a), Self::Cuda(b)) => {
                Arc::ptr_eq(a, b) || (a.dptr != 0 && a.dptr == b.dptr)
            }
            _ => false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    Cpu,
    Cuda,
}

pub trait DeviceBackend: Send + Sync {
    fn name(&self) -> &'static str;
    fn kind(&self) -> DeviceKind;
    fn is_available(&self) -> bool;
    fn allocate_tensor(&self, size: usize) -> Result<DeviceTensor, BackendError>;
    fn transfer_to_device(&self, host_tensor: &[f32]) -> Result<DeviceTensor, BackendError>;
    fn transfer_to_host(&self, device_tensor: &DeviceTensor) -> Result<Vec<f32>, BackendError>;
}

pub struct CPUBackend;

impl DeviceBackend for CPUBackend {
    fn name(&self) -> &'static str {
        "CPU"
    }

    fn kind(&self) -> DeviceKind {
        DeviceKind::Cpu
    }

    fn is_available(&self) -> bool {
        true
    }

    fn allocate_tensor(&self, size: usize) -> Result<DeviceTensor, BackendError> {
        Ok(DeviceTensor::from_cpu_vec(vec![0.0f32; size]))
    }

    fn transfer_to_device(&self, host_tensor: &[f32]) -> Result<DeviceTensor, BackendError> {
        Ok(DeviceTensor::from_cpu_vec(host_tensor.to_vec()))
    }

    fn transfer_to_host(&self, device_tensor: &DeviceTensor) -> Result<Vec<f32>, BackendError> {
        match device_tensor {
            DeviceTensor::Cpu(v) => Ok(v.as_ref().clone()),
            DeviceTensor::Cuda(_) => Err(BackendError::InvalidOperation(
                "Cannot transfer CUDA device tensor using CPUBackend".into(),
            )),
        }
    }
}

/// Process-wide cache of verified `Arc<CudaSession>` instances indexed by CUDA device ordinal.
/// Avoids repeated `cuCtxCreate` and probe roundtrips across multiple `CUDABackend` instances.
static CUDA_SESSION_CACHE: OnceLock<Mutex<HashMap<i32, Arc<CudaSession>>>> = OnceLock::new();

fn get_or_init_cuda_session(ordinal: i32) -> Result<Arc<CudaSession>, BackendError> {
    if ordinal < 0 {
        return Err(BackendError::Unavailable(format!(
            "Invalid negative CUDA device ordinal: {}",
            ordinal
        )));
    }

    let cache_mutex = CUDA_SESSION_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Ok(guard) = cache_mutex.lock() {
        if let Some(existing) = guard.get(&ordinal) {
            return Ok(Arc::clone(existing));
        }
    }

    let session = CudaSession::init(ordinal).map_err(|e| {
        BackendError::Unavailable(format!(
            "Failed to initialize CUDA session for device ordinal {}: {}",
            ordinal, e
        ))
    })?;

    // Verify readiness once per ordinal: allocate 1-float probe buffer and verify HtoD + DtoH roundtrip
    let probe = session.allocate_f32(1).map_err(|e| {
        BackendError::Unavailable(format!(
            "CUDA readiness probe allocation failed on ordinal {}: {}",
            ordinal, e
        ))
    })?;
    session.upload_f32(&probe, &[42.0]).map_err(|e| {
        BackendError::Unavailable(format!(
            "CUDA readiness probe HtoD upload failed on ordinal {}: {}",
            ordinal, e
        ))
    })?;
    let mut out = [0.0f32];
    session.download_f32(&probe, &mut out).map_err(|e| {
        BackendError::Unavailable(format!(
            "CUDA readiness probe DtoH download failed on ordinal {}: {}",
            ordinal, e
        ))
    })?;
    if (out[0] - 42.0).abs() > 1e-4 {
        return Err(BackendError::Unavailable(format!(
            "CUDA readiness probe roundtrip value mismatch on ordinal {}: expected 42.0, got {}",
            ordinal, out[0]
        )));
    }

    let arc_session = Arc::new(session);
    if let Ok(mut guard) = cache_mutex.lock() {
        let entry = guard
            .entry(ordinal)
            .or_insert_with(|| Arc::clone(&arc_session));
        return Ok(Arc::clone(entry));
    }

    Ok(arc_session)
}

pub struct CUDABackend {
    pub device_id: i32,
    session: Option<Arc<CudaSession>>,
    available: bool,
}

impl CUDABackend {
    /// Strict constructor that returns an explicit `BackendError` if the requested CUDA device
    /// ordinal overflows or is unavailable. Reuses a process-wide cached `Arc<CudaSession>` per ordinal.
    pub fn try_new(device_id: u32) -> Result<Self, BackendError> {
        let ordinal = i32::try_from(device_id).map_err(|_| {
            BackendError::Unavailable(format!(
                "CUDA device_id {} exceeds maximum valid i32 ordinal",
                device_id
            ))
        })?;

        let session = get_or_init_cuda_session(ordinal)?;
        Ok(Self {
            device_id: ordinal,
            session: Some(session),
            available: true,
        })
    }

    /// Non-panicking constructor that checks availability via `try_new` and sets `available = false` on error.
    pub fn new(device_id: u32) -> Self {
        match Self::try_new(device_id) {
            Ok(backend) => backend,
            Err(_) => Self {
                device_id: i32::try_from(device_id).unwrap_or(-1),
                session: None,
                available: false,
            },
        }
    }

    pub fn session(&self) -> Option<&Arc<CudaSession>> {
        self.session.as_ref()
    }
}

impl DeviceBackend for CUDABackend {
    fn name(&self) -> &'static str {
        "CUDA"
    }

    fn kind(&self) -> DeviceKind {
        DeviceKind::Cuda
    }

    fn is_available(&self) -> bool {
        self.available
    }

    fn allocate_tensor(&self, size: usize) -> Result<DeviceTensor, BackendError> {
        let session = self
            .session
            .as_ref()
            .ok_or_else(|| BackendError::Unavailable("CUDA session is not initialized".into()))?;
        let buf = session.allocate_f32(size)?;
        Ok(DeviceTensor::from_cuda_buffer(buf))
    }

    fn transfer_to_device(&self, host_tensor: &[f32]) -> Result<DeviceTensor, BackendError> {
        let session = self
            .session
            .as_ref()
            .ok_or_else(|| BackendError::Unavailable("CUDA session is not initialized".into()))?;
        let buf = session.allocate_f32(host_tensor.len())?;
        session.upload_f32(&buf, host_tensor)?;
        Ok(DeviceTensor::from_cuda_buffer(buf))
    }

    fn transfer_to_host(&self, device_tensor: &DeviceTensor) -> Result<Vec<f32>, BackendError> {
        match device_tensor {
            DeviceTensor::Cuda(buf) => {
                let session = self.session.as_ref().ok_or_else(|| {
                    BackendError::Unavailable("CUDA session is not initialized".into())
                })?;
                let mut host = vec![0.0f32; buf.len_elements];
                session.download_f32(buf, &mut host)?;
                Ok(host)
            }
            DeviceTensor::Cpu(v) => Ok(v.as_ref().clone()),
        }
    }
}

/// Default CUDA device ordinal (primary GPU index 0).
pub const PRIMARY_CUDA_DEVICE_ORDINAL: u32 = 0;

impl Default for CUDABackend {
    fn default() -> Self {
        Self::new(PRIMARY_CUDA_DEVICE_ORDINAL)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendPolicy {
    Auto,
    CpuOnly,
    Cuda(u32),
}

pub struct BackendRegistry;

impl BackendRegistry {
    /// Strictly enforces the requested `BackendPolicy`:
    /// - `BackendPolicy::CpuOnly` -> always returns `Ok(CPUBackend)`
    /// - `BackendPolicy::Cuda(id)` -> returns `Ok(CUDABackend)` if ordinal `id` is available,
    ///   or `Err(BackendError::Unavailable)` if unavailable (never silently falls back to CPU).
    /// - `BackendPolicy::Auto` -> returns `CUDABackend` if primary GPU is available, otherwise falls back to `CPUBackend`.
    pub fn try_select_backend(
        policy: BackendPolicy,
    ) -> Result<Box<dyn DeviceBackend>, BackendError> {
        match policy {
            BackendPolicy::CpuOnly => Ok(Box::new(CPUBackend)),
            BackendPolicy::Cuda(id) => {
                let cuda = CUDABackend::try_new(id)?;
                Ok(Box::new(cuda))
            }
            BackendPolicy::Auto => match CUDABackend::try_new(PRIMARY_CUDA_DEVICE_ORDINAL) {
                Ok(cuda) => Ok(Box::new(cuda)),
                Err(_) => Ok(Box::new(CPUBackend)),
            },
        }
    }

    /// Selects a backend according to `policy`. For `BackendPolicy::Auto` or `BackendPolicy::CpuOnly`,
    /// always succeeds. For `BackendPolicy::Cuda(id)`, use `try_select_backend` when explicit failure
    /// on missing GPU is required; `select_backend` falls back to `CPUBackend` only when `try_select_backend` fails.
    pub fn select_backend(policy: BackendPolicy) -> Box<dyn DeviceBackend> {
        Self::try_select_backend(policy).unwrap_or_else(|_| Box::new(CPUBackend))
    }

    pub fn select_best_backend() -> Box<dyn DeviceBackend> {
        Self::select_backend(BackendPolicy::Auto)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_backend_allocation_and_transfers() {
        let backend = CPUBackend;
        assert_eq!(backend.name(), "CPU");
        assert!(backend.is_available());

        let tensor = backend.allocate_tensor(8).expect("allocate cpu tensor");
        assert_eq!(tensor.len(), 8);
        assert_eq!(tensor.bytes(), 32);
        assert!(tensor.is_cpu());
        assert!(!tensor.is_cuda());
        assert_eq!(tensor.kind(), DeviceKind::Cpu);

        let data = vec![1.0, 2.0, 3.0, 4.0];
        let transferred = backend
            .transfer_to_device(&data)
            .expect("transfer to device");
        assert_eq!(transferred.as_cpu_slice(), Some(data.as_slice()));

        // O(1) Arc clone must preserve pointer identity
        let cloned = transferred.clone();
        assert!(transferred.ptr_eq(&cloned));

        let retrieved = backend
            .transfer_to_host(&transferred)
            .expect("transfer to host");
        assert_eq!(retrieved, data);
    }

    #[test]
    fn test_cuda_backend_ordinal_overflow_and_strict_policy_error() {
        let overflow_backend = CUDABackend::new(u32::MAX);
        assert!(!overflow_backend.is_available());
        assert_eq!(overflow_backend.device_id, -1);

        // Explicit Cuda(u32::MAX) policy MUST fail in try_select_backend instead of silently falling back to CPU
        let strict_res = BackendRegistry::try_select_backend(BackendPolicy::Cuda(u32::MAX));
        assert!(strict_res.is_err(), "Explicit invalid Cuda(id) policy must return Err");

        // Non-existent high ordinal (e.g. 9999) must also fail explicitly
        let high_ordinal_res = BackendRegistry::try_select_backend(BackendPolicy::Cuda(9999));
        assert!(
            high_ordinal_res.is_err(),
            "Unavailable CUDA ordinal must return Err under BackendPolicy::Cuda(id)"
        );
    }

    #[test]
    fn test_backend_registry_policy_selection_and_session_caching() {
        let cpu = BackendRegistry::try_select_backend(BackendPolicy::CpuOnly).unwrap();
        assert_eq!(cpu.name(), "CPU");

        let auto = BackendRegistry::select_best_backend();
        assert!(auto.name() == "CPU" || auto.name() == "CUDA");

        // Verify that repeated CUDABackend::new(0) calls reuse the exact same cached Arc<CudaSession>
        let b1 = CUDABackend::new(0);
        let b2 = CUDABackend::new(0);
        if b1.is_available() && b2.is_available() {
            assert!(Arc::ptr_eq(b1.session().unwrap(), b2.session().unwrap()));
        }
    }

    #[test]
    fn test_device_tensor_properties() {
        let empty = DeviceTensor::from_cpu_vec(Vec::new());
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
        assert_eq!(empty.bytes(), 0);
        assert!(empty.is_cpu());
        assert!(!empty.is_cuda());
        assert!(format!("{:?}", empty).contains("Cpu"));
    }
}
