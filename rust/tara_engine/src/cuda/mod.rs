//! CUDA Acceleration Module for TARA Neural Engine.
//!
//! Provides direct, runtime-linked NVIDIA CUDA Driver API integration,
//! custom PTX kernels, and GPU training acceleration.

pub(crate) mod driver;
pub(crate) mod gpu_trainer;
pub(crate) mod kernels;

pub(crate) use gpu_trainer::CudaTrainer;

pub use driver::{CudaDeviceInfo, CudaError};
pub use gpu_trainer::{is_cuda_available, is_cuda_required, is_cuda_required_value, TrainingPrecision};
pub use kernels::validate_ptx_registers;
