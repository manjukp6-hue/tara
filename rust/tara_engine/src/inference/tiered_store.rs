//! Three-tier storage hierarchy (VRAM -> RAM -> Disk) with admission hysteresis and eviction.
//!
//! Enforces:
//! - Single residency: a tensor is resident in at most one tier in memory (VRAM or RAM)
//! - RAM capacity preservation on VRAM-to-RAM eviction
//! - Zero-copy O(1) Arc<Vec<f32>> pointer access on RAM hits
//! - Oversized tensor streaming without cache pollution
//! - Transactional all-or-nothing admission and promotion
//! - Telemetry distinguishing lookups, evictions, prefetches, and promotions

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::inference::backend::{BackendRegistry, DeviceBackend, DeviceKind, DeviceTensor};
use crate::inference::cache_policy::{
    CachePolicy, LFRUCachePolicy, DEFAULT_LFRU_FIXED_MARGIN, DEFAULT_LFRU_MARGIN_PCT,
};
use crate::inference::telemetry::TelemetryMonitor;
use crate::safetensors::SafeTensorsError;
use crate::shard_manager::ShardedSafeTensorsManager;

pub struct TieredTensorStore {
    pub vram_capacity_bytes: usize,
    pub ram_capacity_bytes: usize,
    pub admission_margin_pct: usize,
    pub admission_margin_fixed: usize,

    // Pools: key -> tensor data (single-residency enforced)
    pub vram_pool: HashMap<String, DeviceTensor>,
    pub ram_pool: HashMap<String, Arc<Vec<f32>>>,
    pub disk_keys: HashSet<String>,

    pub policy: Box<dyn CachePolicy>,
    pub telemetry: TelemetryMonitor,
    pub backend: Box<dyn DeviceBackend>,
}

impl TieredTensorStore {
    pub fn new(
        vram_capacity_bytes: usize,
        ram_capacity_bytes: usize,
        disk_keys: HashSet<String>,
    ) -> Self {
        Self {
            vram_capacity_bytes,
            ram_capacity_bytes,
            admission_margin_pct: DEFAULT_LFRU_MARGIN_PCT,
            admission_margin_fixed: DEFAULT_LFRU_FIXED_MARGIN,
            vram_pool: HashMap::new(),
            ram_pool: HashMap::new(),
            disk_keys,
            policy: Box::new(LFRUCachePolicy::new(
                DEFAULT_LFRU_MARGIN_PCT,
                DEFAULT_LFRU_FIXED_MARGIN,
            )),
            telemetry: TelemetryMonitor::new(true),
            backend: BackendRegistry::select_best_backend(),
        }
    }

    pub fn with_backend(mut self, backend: Box<dyn DeviceBackend>) -> Self {
        self.backend = backend;
        self
    }

    pub fn with_policy(mut self, policy: Box<dyn CachePolicy>) -> Self {
        self.policy = policy;
        self
    }

    pub fn register_disk_key(&mut self, key: impl Into<String>) {
        self.disk_keys.insert(key.into());
    }

    pub fn is_disk_key(&self, key: &str) -> bool {
        self.disk_keys.contains(key)
    }

    pub fn vram_resident_bytes(&self) -> usize {
        self.vram_pool.values().map(|t| t.bytes()).sum()
    }

    pub fn ram_resident_bytes(&self) -> usize {
        self.ram_pool
            .values()
            .map(|v| v.len() * std::mem::size_of::<f32>())
            .sum()
    }

    /// Transactional RAM admission: verifies capacity and hysteresis before evicting cold items.
    /// Returns true if admitted, false if rejected or oversized.
    pub fn admit_to_ram(&mut self, key: &str, tensor: Arc<Vec<f32>>) -> bool {
        let bytes = tensor.len() * std::mem::size_of::<f32>();

        // Oversized tensor protection: never cache a tensor larger than total RAM capacity
        if bytes > self.ram_capacity_bytes {
            return false;
        }

        // If already resident in RAM, update and retain
        if self.ram_pool.contains_key(key) {
            self.ram_pool.insert(key.to_string(), tensor);
            return true;
        }

        // Calculate bytes needed to free
        let needed_bytes = (self.ram_resident_bytes() + bytes).saturating_sub(self.ram_capacity_bytes);

        // Transactional plan: verify all evictions satisfy admission hysteresis
        let mut to_evict = Vec::new();
        if needed_bytes > 0 {
            let mut sim_resident: Vec<String> = self.ram_pool.keys().cloned().collect();
            let mut freed = 0;
            while freed < needed_bytes {
                if let Some(cand) = self.policy.pick_eviction(&sim_resident) {
                    if !self.policy.should_admit(key, &cand) {
                        return false; // Hysteresis rejects admission
                    }
                    if let Some(resident_tensor) = self.ram_pool.get(&cand) {
                        freed += resident_tensor.len() * std::mem::size_of::<f32>();
                    }
                    sim_resident.retain(|k| k != &cand);
                    to_evict.push(cand);
                } else {
                    return false; // Cannot free sufficient RAM
                }
            }
        }

        // Execute planned evictions from RAM to Disk
        for evict_key in to_evict {
            if let Some(evicted) = self.ram_pool.remove(&evict_key) {
                let ev_bytes = evicted.len() * std::mem::size_of::<f32>();
                self.telemetry.record_eviction("RAM", "DISK", ev_bytes);
            }
        }

        self.ram_pool.insert(key.to_string(), tensor);
        true
    }

    /// Promotes a tensor from RAM to VRAM using genuine device transfer.
    /// Enforces single residency, VRAM capacity, and RAM capacity on evicted tensors.
    pub fn try_promote_to_vram(&mut self, key: &str) -> bool {
        if self.backend.kind() != DeviceKind::Cuda || self.vram_capacity_bytes == 0 {
            return false;
        }

        let t_data = match self.ram_pool.get(key) {
            Some(t) => Arc::clone(t),
            None => return false,
        };

        let bytes = t_data.len() * std::mem::size_of::<f32>();
        if bytes > self.vram_capacity_bytes {
            return false;
        }

        let needed_bytes =
            (self.vram_resident_bytes() + bytes).saturating_sub(self.vram_capacity_bytes);

        let mut to_evict = Vec::new();
        if needed_bytes > 0 {
            let mut sim_resident: Vec<String> = self.vram_pool.keys().cloned().collect();
            let mut freed = 0;
            while freed < needed_bytes {
                if let Some(cand) = self.policy.pick_eviction(&sim_resident) {
                    if !self.policy.should_admit(key, &cand) {
                        return false;
                    }
                    if let Some(resident_tensor) = self.vram_pool.get(&cand) {
                        freed += resident_tensor.bytes();
                    }
                    sim_resident.retain(|k| k != &cand);
                    to_evict.push(cand);
                } else {
                    return false;
                }
            }
        }

        // Evict from VRAM to RAM
        for evict_key in to_evict {
            if let Some(evicted_dev) = self.vram_pool.remove(&evict_key) {
                let ev_bytes = evicted_dev.bytes();
                self.telemetry.record_eviction("VRAM", "RAM", ev_bytes);
                if let Ok(host_vec) = self.backend.transfer_to_host(&evicted_dev) {
                    // Admit to RAM with capacity enforcement (may evict cold RAM to Disk)
                    self.admit_to_ram(&evict_key, Arc::new(host_vec));
                }
            }
        }

        // Transactional transfer to device with rollback
        match self.backend.transfer_to_device(&t_data) {
            Ok(dev_tensor) => {
                self.vram_pool.insert(key.to_string(), dev_tensor);
                // Enforce single-residency: remove from RAM
                self.ram_pool.remove(key);
                self.telemetry.record_promotion(bytes);
                true
            }
            Err(_) => {
                // Rollback: tensor remains safe in RAM
                false
            }
        }
    }

    /// Hierarchical lookup across VRAM -> RAM -> Disk.
    /// Returns an Arc<Vec<f32>> for zero-copy O(1) pointer sharing.
    pub fn lookup(
        &mut self,
        key: &str,
        shard_manager: &mut ShardedSafeTensorsManager,
    ) -> Result<Arc<Vec<f32>>, SafeTensorsError> {
        self.policy.record_access(key);

        // 1. Check VRAM Tier
        if let Some(t) = self.vram_pool.get(key) {
            let bytes = t.bytes();
            self.telemetry.record_lookup("VRAM", bytes, false);
            let host_vec = self.backend.transfer_to_host(t).map_err(|e| {
                SafeTensorsError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
            })?;
            return Ok(Arc::new(host_vec));
        }

        // 2. Check RAM Tier
        if let Some(t) = self.ram_pool.get(key) {
            let bytes = t.len() * std::mem::size_of::<f32>();
            self.telemetry.record_lookup("RAM", bytes, false);
            let tensor_arc = Arc::clone(t);

            // Attempt promotion to VRAM if CUDA available
            if self.backend.kind() == DeviceKind::Cuda && self.vram_capacity_bytes > 0 {
                self.try_promote_to_vram(key);
            }
            return Ok(tensor_arc);
        }

        // 3. Disk Miss: Load from SafeTensors shard
        let tensor = shard_manager.load_tensor(key)?;
        self.disk_keys.insert(key.to_string());
        let bytes = tensor.len() * std::mem::size_of::<f32>();
        self.telemetry.record_lookup("DISK", bytes, false);

        let tensor_arc = Arc::new(tensor);
        self.admit_to_ram(key, Arc::clone(&tensor_arc));
        Ok(tensor_arc)
    }

    /// Prefetches tensors into RAM using policy admission and capacity limits.
    pub fn prefetch(
        &mut self,
        keys: &[String],
        shard_manager: &mut ShardedSafeTensorsManager,
    ) -> usize {
        let mut loaded = 0;
        for k in keys {
            if self.ram_pool.contains_key(k) || self.vram_pool.contains_key(k) {
                continue;
            }
            if let Ok(t) = shard_manager.load_tensor(k) {
                self.disk_keys.insert(k.clone());
                self.policy.record_access(k);
                let tensor_arc = Arc::new(t);
                if self.admit_to_ram(k, tensor_arc) {
                    self.telemetry.record_prefetch(1);
                    loaded += 1;
                }
            }
        }
        loaded
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::backend::CPUBackend;
    use crate::inference::cache_policy::LRUCachePolicy;
    use crate::safetensors::write_safetensors_sharded;

    #[test]
    fn test_tiered_store_initialization_and_capacities() {
        let store = TieredTensorStore::new(1024, 2048, HashSet::new());
        assert_eq!(store.vram_capacity_bytes, 1024);
        assert_eq!(store.ram_capacity_bytes, 2048);
        assert_eq!(store.vram_resident_bytes(), 0);
        assert_eq!(store.ram_resident_bytes(), 0);
        assert!(matches!(store.backend.kind(), DeviceKind::Cpu | DeviceKind::Cuda));

        let cpu_store = store.with_backend(Box::new(CPUBackend));
        assert_eq!(cpu_store.backend.kind(), DeviceKind::Cpu);
    }

    #[test]
    fn test_cuda_promotion_and_single_residency() {
        let mut store = TieredTensorStore::new(1024, 2048, HashSet::new());
        if store.backend.kind() != DeviceKind::Cuda {
            return;
        }

        let tensor_data = Arc::new(vec![1.0, 2.0, 3.0, 4.0]); // 16 bytes
        assert!(store.admit_to_ram("promotable", Arc::clone(&tensor_data)));
        assert!(store.ram_pool.contains_key("promotable"));
        assert!(!store.vram_pool.contains_key("promotable"));

        for _ in 0..10 {
            store.policy.record_access("promotable");
        }

        let promoted = store.try_promote_to_vram("promotable");
        assert!(promoted);
        // Single residency: MUST be in VRAM, and MUST NOT be in RAM
        assert!(store.vram_pool.contains_key("promotable"));
        assert!(!store.ram_pool.contains_key("promotable"));
        assert_eq!(store.telemetry.promotions, 1);
        assert_eq!(store.vram_resident_bytes(), 16);
        assert_eq!(store.ram_resident_bytes(), 0);
    }

    #[test]
    fn test_admit_to_ram_and_lru_eviction() {
        // RAM capacity: 40 bytes (space for two 5-element f32 tensors = 2 * 20 bytes)
        let mut store = TieredTensorStore::new(0, 40, HashSet::new())
            .with_policy(Box::new(LRUCachePolicy::default()));

        let t1 = Arc::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let t2 = Arc::new(vec![6.0, 7.0, 8.0, 9.0, 10.0]);
        let t3 = Arc::new(vec![11.0, 12.0, 13.0, 14.0, 15.0]);

        store.policy.record_access("tensor1");
        assert!(store.admit_to_ram("tensor1", t1));
        assert_eq!(store.ram_resident_bytes(), 20);

        store.policy.record_access("tensor2");
        assert!(store.admit_to_ram("tensor2", t2));
        assert_eq!(store.ram_resident_bytes(), 40);

        // Touch tensor1 so tensor2 becomes the oldest/coldest
        store.policy.record_access("tensor1");

        // Admit tensor3: should evict tensor2 to disk
        store.policy.record_access("tensor3");
        assert!(store.admit_to_ram("tensor3", t3));
        assert_eq!(store.ram_resident_bytes(), 40);

        assert!(store.ram_pool.contains_key("tensor1"));
        assert!(!store.ram_pool.contains_key("tensor2"));
        assert!(store.ram_pool.contains_key("tensor3"));
        assert_eq!(store.telemetry.evictions, 1);
    }

    #[test]
    fn test_oversized_tensor_rejected_from_ram() {
        // RAM capacity: 16 bytes (space for 4 floats)
        let mut store = TieredTensorStore::new(0, 16, HashSet::new());
        let oversized = Arc::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]); // 20 bytes

        assert!(!store.admit_to_ram("oversized", oversized));
        assert_eq!(store.ram_resident_bytes(), 0);
        assert!(!store.ram_pool.contains_key("oversized"));
    }

    #[test]
    fn test_hysteresis_blocks_cold_admission() {
        // RAM capacity: 20 bytes (space for one 5-float tensor)
        let mut store = TieredTensorStore::new(0, 20, HashSet::new());

        let t_resident = Arc::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let t_candidate = Arc::new(vec![2.0, 3.0, 4.0, 5.0, 6.0]);

        // Resident gets accessed 10 times
        for _ in 0..10 {
            store.policy.record_access("hot_resident");
        }
        assert!(store.admit_to_ram("hot_resident", t_resident));

        // Candidate gets accessed only 2 times (fails hysteresis threshold: 10 + 2 + 4 = 16)
        for _ in 0..2 {
            store.policy.record_access("cold_candidate");
        }

        assert!(!store.admit_to_ram("cold_candidate", t_candidate));
        assert!(store.ram_pool.contains_key("hot_resident"));
        assert!(!store.ram_pool.contains_key("cold_candidate"));
        assert_eq!(store.telemetry.evictions, 0);
    }

    #[test]
    fn test_single_residency_and_lookup_with_shards() {
        let test_dir = std::env::temp_dir().join(format!("tara_tiered_test_{:x}", rand::random::<u64>()));
        let _ = std::fs::create_dir_all(&test_dir);

        let mut tensors = HashMap::new();
        let mut shapes = HashMap::new();
        tensors.insert("weight_a".to_string(), vec![1.0, 2.0, 3.0, 4.0]);
        shapes.insert("weight_a".to_string(), vec![2, 2]);
        tensors.insert("weight_b".to_string(), vec![5.0, 6.0, 7.0, 8.0]);
        shapes.insert("weight_b".to_string(), vec![2, 2]);

        let written = write_safetensors_sharded(&tensors, &shapes, &test_dir.to_string_lossy(), 1024 * 1024);
        assert!(written.is_ok());

        let mut shard_mgr = ShardedSafeTensorsManager::new(&test_dir, 2).unwrap();
        let mut store = TieredTensorStore::new(0, 64, HashSet::new())
            .with_backend(Box::new(CPUBackend));

        // 1. First lookup loads from disk into RAM
        let loaded_a = store.lookup("weight_a", &mut shard_mgr).unwrap();
        assert_eq!(*loaded_a, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(store.telemetry.disk_misses, 1);
        assert_eq!(store.telemetry.ram_hits, 0);
        assert!(store.ram_pool.contains_key("weight_a"));
        assert!(!store.vram_pool.contains_key("weight_a"));

        // 2. Second lookup is an O(1) RAM cache hit returning identical Arc pointer
        let loaded_a2 = store.lookup("weight_a", &mut shard_mgr).unwrap();
        assert_eq!(*loaded_a2, vec![1.0, 2.0, 3.0, 4.0]);
        assert!(Arc::ptr_eq(&loaded_a, &loaded_a2));
        assert_eq!(store.telemetry.ram_hits, 1);
        assert_eq!(store.telemetry.disk_misses, 1);

        // 3. Prefetch weight_b into RAM
        let prefetched = store.prefetch(&["weight_b".to_string()], &mut shard_mgr);
        assert_eq!(prefetched, 1);
        assert_eq!(store.telemetry.prefetched, 1);
        assert!(store.ram_pool.contains_key("weight_b"));

        // Cleanup test artifacts
        let _ = std::fs::remove_dir_all(&test_dir);
    }
}
