//! CUDA Acceleration Module for TARA Neural Engine.
//!
//! Provides direct, runtime-linked NVIDIA CUDA Driver API integration,
//! custom PTX kernels, and GPU training acceleration.

pub mod driver;
pub mod gpu_trainer;
pub mod kernels;

pub use driver::{CudaDeviceInfo, CudaDriver, CudaError, CudaSession};
pub use gpu_trainer::{
    is_cuda_required, is_cuda_required_value, unpack_u32_to_f16, CudaTrainer, TrainingPrecision,
};
pub use kernels::validate_ptx_registers;
