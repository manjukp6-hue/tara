//! Inference runtime components for TARA.
//!
//! Subsystems & 4-Layer Physical Hierarchy Contract:
//! - `backend`: Pluggable device execution backend (`CPUBackend`, `CUDABackend`) with process-wide
//!   cached `Arc<CudaSession>` per ordinal and reference-counted `DeviceTensor` (`Cpu(Arc<Vec<f32>>)`
//!   and `Cuda(Arc<CudaBuffer>)`).
//! - `tiered_store`: Physical three-tier memory hierarchy (`VRAM -> RAM -> Disk`) storing
//!   `DeviceTensor::Cuda` in `vram_pool` and `DeviceTensor::Cpu` in `ram_pool` with single-residency,
//!   zero-DtoH VRAM hits, and transactional promotion/eviction.
//! - `cache_policy`: Adaptive `LFRUCachePolicy`, `LRUCachePolicy`, and `LFUCachePolicy` with
//!   hysteresis margin to prevent ping-pong thrashing.
//! - `telemetry`: Inference latency, transfer volume, and tier hit/miss telemetry.
//! - `lookahead`: Speculative prefetching across neural layers.
//! - `router`: Routing transition tracker and periodic heat decay.

pub mod backend;
pub mod cache_policy;
pub mod lookahead;
pub mod router;
pub mod telemetry;
pub mod tiered_store;

pub use backend::{
    BackendError, BackendPolicy, BackendRegistry, CPUBackend, CUDABackend, DeviceBackend,
    DeviceKind, DeviceTensor, PRIMARY_CUDA_DEVICE_ORDINAL,
};
pub use cache_policy::{
    tier_decay_value, tier_lfru_score, tier_should_promote, CachePolicy, LFRUCachePolicy,
    LFUCachePolicy, LRUCachePolicy, DEFAULT_LFRU_FIXED_MARGIN, DEFAULT_LFRU_MARGIN_PCT,
};
pub use lookahead::LookaheadPrefetcher;
pub use router::RoutingTracker;
pub use telemetry::{
    EvictionRoute, LookupOutcome, RouteTransferSnapshot, StorageTier, TelemetryError,
    TelemetryMonitor, TelemetrySnapshot, TransferRoute, DEFAULT_LATENCY_RESERVOIR_CAPACITY,
};
pub use tiered_store::{TierLocation, TieredTensorStore};

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;
    use crate::safetensors::write_safetensors_sharded;
    use crate::shard_manager::ShardedSafeTensorsManager;

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn test_inference_module_send_sync_and_four_layer_contract() {
        // 1. Verify Send + Sync compatibility across the entire 4-layer stack:
        //    CudaSession / CudaBuffer -> DeviceTensor -> DeviceBackend -> TieredTensorStore / TelemetryMonitor
        assert_send_sync::<DeviceTensor>();
        assert_send_sync::<CPUBackend>();
        assert_send_sync::<CUDABackend>();
        assert_send_sync::<Box<dyn DeviceBackend>>();
        assert_send_sync::<Box<dyn CachePolicy>>();
        assert_send_sync::<TelemetryMonitor>();
        assert_send_sync::<TieredTensorStore>();

        // 2. End-to-end integration of BackendRegistry + TieredTensorStore + LookaheadPrefetcher + RoutingTracker
        let test_dir = std::env::temp_dir().join(format!(
            "tara_inference_mod_test_{:x}",
            rand::random::<u64>()
        ));
        let _ = std::fs::create_dir_all(&test_dir);

        let mut tensors = HashMap::new();
        let mut shapes = HashMap::new();
        tensors.insert("layer0.weight".to_string(), vec![1.0, 2.0, 3.0, 4.0]);
        shapes.insert("layer0.weight".to_string(), vec![2, 2]);
        tensors.insert("layer1.weight".to_string(), vec![5.0, 6.0, 7.0, 8.0]);
        shapes.insert("layer1.weight".to_string(), vec![2, 2]);

        write_safetensors_sharded(&tensors, &shapes, &test_dir.to_string_lossy(), 1024 * 1024)
            .unwrap();
        let mut shard_mgr = ShardedSafeTensorsManager::new(&test_dir, 2).unwrap();

        let mut store = TieredTensorStore::try_new_with_policy(
            64,
            64,
            HashSet::new(),
            BackendPolicy::CpuOnly,
        )
        .unwrap();
        store.sync_disk_catalog(&shard_mgr);

        // Verify initial disk catalog residency
        assert_eq!(store.locate_tier("layer0.weight"), TierLocation::Disk);
        assert_eq!(store.locate_tier("layer1.weight"), TierLocation::Disk);

        // Lookup layer0.weight -> admitted into ram_pool as DeviceTensor::Cpu
        let t0 = store.lookup("layer0.weight", &mut shard_mgr).unwrap();
        assert!(t0.is_cpu());
        assert_eq!(store.locate_tier("layer0.weight"), TierLocation::Ram);
        assert!(store.ram_pool.get("layer0.weight").unwrap().is_cpu());

        // Use LookaheadPrefetcher to speculatively prefetch layer 1 while executing layer 0
        let mut layer_map = HashMap::new();
        layer_map.insert(0, vec!["layer0.weight".to_string()]);
        layer_map.insert(1, vec!["layer1.weight".to_string()]);

        let mut prefetcher = LookaheadPrefetcher::new(1, true);
        prefetcher.on_layer_begin(0, 2, &layer_map, |keys| {
            store.prefetch(keys, &mut shard_mgr)
        });
        prefetcher.on_layer_complete(0, &layer_map);

        assert_eq!(store.locate_tier("layer1.weight"), TierLocation::Ram);
        assert!(store.ram_pool.get("layer1.weight").unwrap().is_cpu());
        assert_eq!(store.telemetry.prefetched(), 1);
        assert_eq!(store.telemetry.prefetch_hits(), 0);

        // Consume prefetched layer1.weight and verify prefetch_hits increments to 1
        let t1 = store.lookup("layer1.weight", &mut shard_mgr).unwrap();
        assert!(t1.is_cpu());
        assert_eq!(store.telemetry.prefetch_hits(), 1);

        // Record routing transitions
        let mut router = RoutingTracker::default();
        router.record_access("layer0.weight");
        router.record_access("layer1.weight");
        assert_eq!(
            router.predict_next("layer0.weight", 1),
            vec!["layer1.weight".to_string()]
        );

        // Verify store and telemetry invariants across all tiers
        assert!(store.verify_invariants().is_ok());

        // Verify multi-thread Send + Sync sharing of Arc<TieredTensorStore>
        let shared_store = Arc::new(store);
        let shared_clone = Arc::clone(&shared_store);
        let handle = std::thread::spawn(move || {
            assert_eq!(shared_clone.locate_tier("layer0.weight"), TierLocation::Ram);
            assert_eq!(shared_clone.locate_tier("layer1.weight"), TierLocation::Ram);
            assert!(shared_clone.verify_invariants().is_ok());
        });
        handle.join().unwrap();

        let _ = std::fs::remove_dir_all(&test_dir);
    }
}
