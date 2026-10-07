//! Inference runtime components for TARA.
//!
//! Subsystems:
//! - `backend`: Pluggable device execution backend (CPU, CUDA)
//! - `cache_policy`: Adaptive LFRU, LRU, LFU cache eviction policies with hysteresis
//! - `tiered_store`: Three-tier memory hierarchy (VRAM -> RAM -> Disk)
//! - `telemetry`: Inference latency and cache efficiency telemetry
//! - `lookahead`: Speculative prefetching across neural layers
//! - `router`: Routing transition tracker and heat decay

pub mod backend;
pub mod cache_policy;
pub mod lookahead;
pub mod router;
pub mod telemetry;
pub mod tiered_store;

pub use backend::{
    BackendError, BackendPolicy, BackendRegistry, CPUBackend, CUDABackend, DeviceBackend,
    DeviceTensor,
};
pub use cache_policy::{
    tier_decay_value, tier_should_promote, CachePolicy, LFRUCachePolicy, LFUCachePolicy,
    LRUCachePolicy,
};
pub use lookahead::LookaheadPrefetcher;
pub use router::RoutingTracker;
pub use telemetry::TelemetryMonitor;
pub use tiered_store::TieredTensorStore;
