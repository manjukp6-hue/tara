//! CUDA Acceleration Module for TARA Neural Engine.
//!
//! Provides direct, runtime-linked NVIDIA CUDA Driver API integration,
//! custom PTX kernels, and GPU training acceleration.

pub mod driver;
pub mod gpu_device;
pub mod gpu_trainer;
pub mod kernels;

pub use driver::{CudaDeviceInfo, CudaError};
pub use gpu_device::CudaComputeDevice;
pub use gpu_trainer::{
    capture_memory_snapshot, is_cuda_available, is_cuda_required, is_cuda_required_value,
    log_flushed_memory_checkpoint, AdamWMemoryAudit, CudaTrainer, MemorySnapshot,
    TrainingPrecision,
};
pub use kernels::validate_ptx_registers;
