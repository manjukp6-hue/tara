//! TARA-HD: Dynamic Hardware Discovery, Universal Compute Orchestration & On-Demand Chunked CPU Streaming.

pub mod auto_config;
pub mod channel;
pub mod device;
pub mod discovery;
pub mod event;
pub mod lora;
pub mod memory_governor;
pub mod streaming;
pub mod telemetry;
pub mod types;

pub use auto_config::AutoTrainingConfig;
pub use channel::ComputeChannel;
pub use device::{ComputeDevice, CpuDevice, CudaDevice, CustomDevice, TpuDevice};
pub use discovery::DiscoveredInventory;
pub use event::{CancellationToken, MemoryFlowGovernor, StreamEvent};
pub use lora::{DynamicAdapterHub, LoraAdapterSet, LoraConfig, LoraLayer, LoraMode};
pub use memory_governor::{
    MemoryExecutionMode, MemoryGovernor, ModelMemoryBudget, PlacementDecision,
};
pub use streaming::{
    CpuStreamHelper, SequenceChunkIterator, StreamBufferPool, StreamConfig, StreamWindow,
    UniversalStreamEngine, UniversalStreamReceiver, UniversalStreamSender,
};
pub use telemetry::{LiveTelemetryProvider, RuntimeTelemetrySnapshot};
pub use types::{
    DeviceCapability, DeviceKind, LinearBackwardArgs, PrecisionMode, SelectionPolicy, TaraError,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_selection_policy_parsing() {
        assert_eq!(SelectionPolicy::parse("auto"), SelectionPolicy::Auto);
        assert_eq!(
            SelectionPolicy::parse("gpu:1"),
            SelectionPolicy::ExplicitWithFallback(DeviceKind::Gpu, Some(1))
        );
        assert_eq!(
            SelectionPolicy::parse("strict:gpu:1"),
            SelectionPolicy::Explicit(DeviceKind::Gpu, Some(1))
        );
        assert_eq!(
            SelectionPolicy::parse("tpu:0!strict"),
            SelectionPolicy::Explicit(DeviceKind::Tpu, Some(0))
        );
        assert_eq!(
            SelectionPolicy::parse("cpu"),
            SelectionPolicy::Explicit(DeviceKind::Cpu, None)
        );
    }

    #[test]
    fn test_cpu_channel_and_streaming_helper() {
        let channel =
            ComputeChannel::with_options(&SelectionPolicy::parse("cpu:0"), PrecisionMode::Fp32)
                .expect("CPU device 0 must resolve");
        assert!(channel.is_cpu());
        assert_eq!(channel.precision(), PrecisionMode::Fp32);

        let helper = channel
            .stream_helper()
            .expect("CPU channel must expose CpuStreamHelper");
        let tokens: Vec<u32> = (0..600).collect();
        let chunks: Vec<StreamWindow> = helper.stream_sequence_chunks(&tokens).collect();

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].start_idx, 0);
        assert_eq!(chunks[0].end_idx, 256);
        assert!(!chunks[0].is_last_chunk);
        assert_eq!(chunks[2].start_idx, 512);
        assert_eq!(chunks[2].end_idx, 600);
        assert!(chunks[2].is_last_chunk);
    }

    #[test]
    fn test_buffer_pool_reuse() {
        let helper = CpuStreamHelper::new();
        let mut pool = helper.create_buffer_pool(128);

        let mut buf = pool.acquire();
        assert_eq!(buf.len(), 128);
        buf[0] = 42.0;

        pool.release(buf);
        assert_eq!(pool.available_buffers(), 1);

        let buf_reused = pool.acquire();
        assert_eq!(buf_reused.len(), 128);
        assert_eq!(buf_reused[0], 0.0);
    }
}
