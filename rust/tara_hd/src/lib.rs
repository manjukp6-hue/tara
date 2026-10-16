//! TARA-HD: Dynamic Hardware Discovery, Universal Compute Orchestration & On-Demand Chunked CPU Streaming.

pub mod channel;
pub mod device;
pub mod discovery;
pub mod streaming;
pub mod types;

pub use channel::ComputeChannel;
pub use device::{ComputeDevice, CpuDevice, CudaDevice, CustomDevice, TpuDevice};
pub use discovery::DiscoveredInventory;
pub use streaming::{CpuStreamHelper, SequenceChunkIterator, StreamConfig, StreamWindow};
pub use types::{DeviceCapability, DeviceKind, PrecisionMode, SelectionPolicy};
