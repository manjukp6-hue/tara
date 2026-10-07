//! Pluggable backend dispatch for CPU and GPU execution.
//!
//! Provides genuine device abstractions for CPU host RAM and NVIDIA GPU VRAM:
//! - `DeviceTensor`: Enum encapsulating CPU `Vec<f32>` and CUDA `CudaBuffer`
//! - `DeviceBackend`: Pluggable trait supporting genuine device allocation and host <-> device transfers
//! - `CUDABackend`: Authentic CUDA device backend holding a live `CudaSession` and physical GPU memory
//! - `CPUBackend`: Pure host RAM implementation
//! - `BackendRegistry`: Policy-driven backend discovery and selection

use std::sync::Arc;
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

/// Represents a tensor resident either in CPU host memory or GPU VRAM.
pub enum DeviceTensor {
    Cpu(Vec<f32>),
    Cuda(CudaBuffer),
}

impl DeviceTensor {
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

    pub fn as_cuda_buffer(&self) -> Option<&CudaBuffer> {
        match self {
            Self::Cpu(_) => None,
            Self::Cuda(b) => Some(b),
        }
    }
}

pub trait DeviceBackend: Send + Sync {
    fn name(&self) -> &'static str;
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

    fn is_available(&self) -> bool {
        true
    }

    fn allocate_tensor(&self, size: usize) -> Result<DeviceTensor, BackendError> {
        Ok(DeviceTensor::Cpu(vec![0.0f32; size]))
    }

    fn transfer_to_device(&self, host_tensor: &[f32]) -> Result<DeviceTensor, BackendError> {
        Ok(DeviceTensor::Cpu(host_tensor.to_vec()))
    }

    fn transfer_to_host(&self, device_tensor: &DeviceTensor) -> Result<Vec<f32>, BackendError> {
        match device_tensor {
            DeviceTensor::Cpu(v) => Ok(v.clone()),
            DeviceTensor::Cuda(_) => Err(BackendError::InvalidOperation(
                "Cannot transfer CUDA device tensor using CPUBackend".into(),
            )),
        }
    }
}

pub struct CUDABackend {
    pub device_id: i32,
    session: Option<Arc<CudaSession>>,
    available: bool,
}

impl CUDABackend {
    pub fn new(device_id: u32) -> Self {
        let ordinal = match i32::try_from(device_id) {
            Ok(ord) => ord,
            Err(_) => {
                return Self {
                    device_id: -1,
                    session: None,
                    available: false,
                };
            }
        };

        match CudaSession::init(ordinal) {
            Ok(session) => {
                // Verify readiness: allocate probe buffer and perform bidirectional roundtrip
                let ready = (|| -> Result<(), CudaError> {
                    let probe = session.allocate_f32(1)?;
                    session.upload_f32(&probe, &[42.0])?;
                    let mut out = [0.0f32];
                    session.download_f32(&probe, &mut out)?;
                    if (out[0] - 42.0).abs() > 1e-4 {
                        return Err(CudaError::KernelError(
                            "Probe memory roundtrip value mismatch".into(),
                        ));
                    }
                    Ok(())
                })()
                .is_ok();

                if ready {
                    Self {
                        device_id: ordinal,
                        session: Some(Arc::new(session)),
                        available: true,
                    }
                } else {
                    Self {
                        device_id: ordinal,
                        session: None,
                        available: false,
                    }
                }
            }
            Err(_) => Self {
                device_id: ordinal,
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

    fn is_available(&self) -> bool {
        self.available
    }

    fn allocate_tensor(&self, size: usize) -> Result<DeviceTensor, BackendError> {
        let session = self
            .session
            .as_ref()
            .ok_or_else(|| BackendError::Unavailable("CUDA session is not initialized".into()))?;
        let buf = session.allocate_f32(size)?;
        Ok(DeviceTensor::Cuda(buf))
    }

    fn transfer_to_device(&self, host_tensor: &[f32]) -> Result<DeviceTensor, BackendError> {
        let session = self
            .session
            .as_ref()
            .ok_or_else(|| BackendError::Unavailable("CUDA session is not initialized".into()))?;
        let buf = session.allocate_f32(host_tensor.len())?;
        session.upload_f32(&buf, host_tensor)?;
        Ok(DeviceTensor::Cuda(buf))
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
            DeviceTensor::Cpu(v) => Ok(v.clone()),
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
    pub fn select_backend(policy: BackendPolicy) -> Box<dyn DeviceBackend> {
        match policy {
            BackendPolicy::CpuOnly => Box::new(CPUBackend),
            BackendPolicy::Cuda(id) => {
                let cuda = CUDABackend::new(id);
                if cuda.is_available() {
                    Box::new(cuda)
                } else {
                    Box::new(CPUBackend)
                }
            }
            BackendPolicy::Auto => {
                let cuda = CUDABackend::default();
                if cuda.is_available() {
                    Box::new(cuda)
                } else {
                    Box::new(CPUBackend)
                }
            }
        }
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

        let data = vec![1.0, 2.0, 3.0, 4.0];
        let transferred = backend
            .transfer_to_device(&data)
            .expect("transfer to device");
        assert_eq!(transferred.as_cpu_slice(), Some(data.as_slice()));

        let retrieved = backend
            .transfer_to_host(&transferred)
            .expect("transfer to host");
        assert_eq!(retrieved, data);
    }

    #[test]
    fn test_cuda_backend_ordinal_overflow_handling() {
        let overflow_backend = CUDABackend::new(u32::MAX);
        assert!(!overflow_backend.is_available());
        assert_eq!(overflow_backend.device_id, -1);
    }

    #[test]
    fn test_backend_registry_policy_selection() {
        let cpu = BackendRegistry::select_backend(BackendPolicy::CpuOnly);
        assert_eq!(cpu.name(), "CPU");

        let auto = BackendRegistry::select_best_backend();
        assert!(auto.name() == "CPU" || auto.name() == "CUDA");
    }

    #[test]
    fn test_device_tensor_properties() {
        let empty = DeviceTensor::Cpu(Vec::new());
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
        assert_eq!(empty.bytes(), 0);
        assert!(empty.is_cpu());
        assert!(!empty.is_cuda());
    }
}
