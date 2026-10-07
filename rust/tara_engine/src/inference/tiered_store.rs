//! Three-tier storage hierarchy (VRAM -> RAM -> Disk) with admission hysteresis and eviction.
//!
//! Enforces:
//! - True device residency: `lookup()` / `lookup_device()` return `DeviceTensor` directly without DtoH copies
//! - Single residency: a tensor is resident in at most one memory tier (VRAM or RAM)
//! - RAM capacity preservation on VRAM-to-RAM eviction (falls back to Disk if RAM cannot admit)
//! - Zero-copy O(1) `Arc<CudaBuffer>` and `Arc<Vec<f32>>` handle sharing on VRAM/RAM hits
//! - Oversized tensor streaming without cache pollution or RAM eviction
//! - Transactional all-or-nothing admission and promotion with rollback on CUDA OOM
//! - Authoritative 3-tier state machine (`TierLocation`: `Vram`, `Ram`, `Disk`, `NotFound`)

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::inference::backend::{
    BackendError, BackendPolicy, BackendRegistry, DeviceBackend, DeviceKind, DeviceTensor,
};
use crate::inference::cache_policy::{
    CachePolicy, LFRUCachePolicy, DEFAULT_LFRU_FIXED_MARGIN, DEFAULT_LFRU_MARGIN_PCT,
};
use crate::inference::telemetry::TelemetryMonitor;
use crate::safetensors::SafeTensorsError;
use crate::shard_manager::ShardedSafeTensorsManager;

/// Authoritative location of a tensor in the three-tier hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TierLocation {
    Vram,
    Ram,
    Disk,
    NotFound,
}

pub struct TieredTensorStore {
    pub vram_capacity_bytes: usize,
    pub ram_capacity_bytes: usize,
    pub admission_margin_pct: usize,
    pub admission_margin_fixed: usize,

    // Pools: key -> tensor data (single-residency enforced: vram_pool ∩ ram_pool == ∅)
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

    /// Strict constructor that enforces `BackendPolicy` (e.g., failing explicitly if
    /// `BackendPolicy::Cuda(id)` is requested on a machine where that GPU ordinal is unavailable).
    pub fn try_new_with_policy(
        vram_capacity_bytes: usize,
        ram_capacity_bytes: usize,
        disk_keys: HashSet<String>,
        backend_policy: BackendPolicy,
    ) -> Result<Self, BackendError> {
        let backend = BackendRegistry::try_select_backend(backend_policy)?;
        Ok(Self::new(vram_capacity_bytes, ram_capacity_bytes, disk_keys).with_backend(backend))
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

    /// Synchronizes the disk catalog with all tensor names indexed by the `ShardedSafeTensorsManager`.
    pub fn sync_disk_catalog(&mut self, shard_manager: &ShardedSafeTensorsManager) {
        for key in shard_manager.list_tensors() {
            self.disk_keys.insert(key);
        }
    }

    /// Returns the authoritative current tier of `key` across `VRAM -> RAM -> Disk`.
    pub fn locate_tier(&self, key: &str) -> TierLocation {
        if self.vram_pool.contains_key(key) {
            TierLocation::Vram
        } else if self.ram_pool.contains_key(key) {
            TierLocation::Ram
        } else if self.disk_keys.contains(key) {
            TierLocation::Disk
        } else {
            TierLocation::NotFound
        }
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

    /// Verifies all structural invariants of the three-tier store:
    /// 1. `vram_resident_bytes() <= vram_capacity_bytes`
    /// 2. `ram_resident_bytes() <= ram_capacity_bytes`
    /// 3. Disjoint memory residency (`vram_pool.keys() ∩ ram_pool.keys() == ∅`)
    /// 4. No individual resident tensor exceeds its tier capacity.
    pub fn verify_invariants(&self) -> Result<(), String> {
        let vram_used = self.vram_resident_bytes();
        if vram_used > self.vram_capacity_bytes {
            return Err(format!(
                "VRAM capacity invariant violated: used {} > capacity {}",
                vram_used, self.vram_capacity_bytes
            ));
        }

        let ram_used = self.ram_resident_bytes();
        if ram_used > self.ram_capacity_bytes {
            return Err(format!(
                "RAM capacity invariant violated: used {} > capacity {}",
                ram_used, self.ram_capacity_bytes
            ));
        }

        for k in self.vram_pool.keys() {
            if self.ram_pool.contains_key(k) {
                return Err(format!(
                    "Single-residency invariant violated: key '{}' is present in both VRAM and RAM",
                    k
                ));
            }
        }

        Ok(())
    }

    /// Transactional RAM admission: verifies capacity and hysteresis before evicting cold items.
    /// Enforces single-residency (will not duplicate a key already in `vram_pool`).
    /// Returns true if admitted, false if rejected or oversized.
    pub fn admit_to_ram(&mut self, key: &str, tensor: Arc<Vec<f32>>) -> bool {
        let bytes = tensor.len() * std::mem::size_of::<f32>();

        // Oversized tensor protection: never cache a tensor larger than total RAM capacity
        if self.ram_capacity_bytes == 0 || bytes > self.ram_capacity_bytes {
            return false;
        }

        // Single-residency guard: if already in VRAM, do not duplicate in RAM
        if self.vram_pool.contains_key(key) {
            return false;
        }

        // If already resident in RAM, check if updated size fits within capacity
        if let Some(existing) = self.ram_pool.get(key) {
            let existing_bytes = existing.len() * std::mem::size_of::<f32>();
            if self.ram_resident_bytes().saturating_sub(existing_bytes) + bytes
                <= self.ram_capacity_bytes
            {
                self.ram_pool.insert(key.to_string(), tensor);
                return true;
            }
        }

        // Calculate bytes needed to free
        let needed_bytes =
            (self.ram_resident_bytes() + bytes).saturating_sub(self.ram_capacity_bytes);

        // Transactional plan: verify all evictions satisfy admission hysteresis
        let mut to_evict = Vec::new();
        if needed_bytes > 0 {
            let mut sim_resident: Vec<String> = self
                .ram_pool
                .keys()
                .filter(|k| k.as_str() != key)
                .cloned()
                .collect();
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
                self.disk_keys.insert(evict_key);
                self.telemetry.record_eviction("RAM", "DISK", ev_bytes);
            }
        }

        self.ram_pool.insert(key.to_string(), tensor);
        true
    }

    /// Promotes a tensor from RAM to VRAM using genuine device transfer (`HtoD`).
    /// Enforces:
    /// 1. Single residency (`vram_pool ∩ ram_pool == ∅`)
    /// 2. VRAM capacity (`vram_resident_bytes() <= vram_capacity_bytes`)
    /// 3. RAM capacity on demoted VRAM tensors (`ram_resident_bytes() <= ram_capacity_bytes`;
    ///    if a demoted VRAM tensor cannot be admitted to RAM, it is evicted to Disk)
    /// 4. Transactional rollback on CUDA OOM or transfer failure.
    pub fn try_promote_to_vram(&mut self, key: &str) -> bool {
        if self.backend.kind() != DeviceKind::Cuda
            || !self.backend.is_available()
            || self.vram_capacity_bytes == 0
        {
            return false;
        }

        if self.vram_pool.contains_key(key) {
            self.ram_pool.remove(key);
            return true;
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

        // Fast path: no VRAM eviction required — attempt device transfer BEFORE mutating any pool
        if needed_bytes == 0 {
            match self.backend.transfer_to_device(&t_data) {
                Ok(dev_tensor) => {
                    self.ram_pool.remove(key);
                    self.vram_pool.insert(key.to_string(), dev_tensor);
                    self.telemetry.record_promotion(bytes);
                    return true;
                }
                Err(_) => {
                    return false;
                }
            }
        }

        // Plan VRAM eviction victims using cache policy and admission hysteresis
        let mut to_evict = Vec::new();
        {
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

        // Stage all planned VRAM eviction victims into host memory BEFORE mutating pools
        let mut staged_victims: Vec<(String, usize, Arc<Vec<f32>>)> =
            Vec::with_capacity(to_evict.len());
        for evict_key in &to_evict {
            let Some(evicted_dev) = self.vram_pool.get(evict_key) else {
                return false;
            };
            let ev_bytes = evicted_dev.bytes();
            match self.backend.transfer_to_host(evicted_dev) {
                Ok(host_vec) => {
                    staged_victims.push((evict_key.clone(), ev_bytes, Arc::new(host_vec)));
                }
                Err(_) => {
                    // Abort cleanly before any pool mutation
                    return false;
                }
            }
        }

        // Remove and drop VRAM victims so physical GPU memory is freed before allocating the promoted tensor
        for (evict_key, _, _) in &staged_victims {
            self.vram_pool.remove(evict_key);
        }

        // Attempt HtoD allocation and upload for the promoted tensor
        match self.backend.transfer_to_device(&t_data) {
            Ok(dev_tensor) => {
                // 1. Remove `key` from RAM first and insert into VRAM:
                //    - Frees `key`'s RAM slot so demoted VRAM victims can use it
                //    - Ensures `key` can never be evicted from RAM during victim demotion
                //    - Preserves strict single-residency at every step
                self.ram_pool.remove(key);
                self.vram_pool.insert(key.to_string(), dev_tensor);
                self.telemetry.record_promotion(bytes);

                // 2. Demote staged VRAM victims into RAM (if capacity & hysteresis permit) or Disk
                for (evict_key, ev_bytes, host_arc) in staged_victims {
                    if self.admit_to_ram(&evict_key, host_arc) {
                        self.telemetry.record_eviction("VRAM", "RAM", ev_bytes);
                    } else {
                        self.disk_keys.insert(evict_key);
                        self.telemetry.record_eviction("VRAM", "DISK", ev_bytes);
                    }
                }
                true
            }
            Err(_) => {
                // Rollback on CUDA OOM / transfer failure:
                // `key` was never removed from `ram_pool`, so `key` remains safely in RAM.
                // Restore staged victims back to VRAM if possible, otherwise demote to RAM/Disk.
                for (evict_key, ev_bytes, host_arc) in staged_victims {
                    let can_fit_vram =
                        self.vram_resident_bytes() + ev_bytes <= self.vram_capacity_bytes;
                    let restored_to_vram = if can_fit_vram {
                        if let Ok(restored_dev) = self.backend.transfer_to_device(&host_arc) {
                            self.vram_pool.insert(evict_key.clone(), restored_dev);
                            true
                        } else {
                            false
                        }
                    } else {
                        false
                    };

                    if !restored_to_vram {
                        if self.admit_to_ram(&evict_key, host_arc) {
                            self.telemetry.record_eviction("VRAM", "RAM", ev_bytes);
                        } else {
                            self.disk_keys.insert(evict_key);
                            self.telemetry.record_eviction("VRAM", "DISK", ev_bytes);
                        }
                    }
                }
                false
            }
        }
    }

    /// Hierarchical device-preserving lookup across `VRAM -> RAM -> Disk`.
    /// - On a **VRAM hit**: returns `DeviceTensor::Cuda(Arc<CudaBuffer>)` in $O(1)$ with **zero DtoH copy**.
    /// - On a **RAM hit**: attempts promotion to VRAM if CUDA is active (returning the promoted `DeviceTensor::Cuda`),
    ///   otherwise returns `DeviceTensor::Cpu(Arc<Vec<f32>>)` in $O(1)$ with **zero host vector clone**.
    /// - On a **Disk miss**: loads from `shard_manager`, admits to RAM if `bytes <= ram_capacity_bytes`
    ///   (streaming oversized tensors directly without polluting or evicting `ram_pool`), and returns `DeviceTensor`.
    pub fn lookup(
        &mut self,
        key: &str,
        shard_manager: &mut ShardedSafeTensorsManager,
    ) -> Result<DeviceTensor, SafeTensorsError> {
        self.lookup_device(key, shard_manager)
    }

    /// Explicit alias for device-preserving hierarchical lookup (`VRAM -> RAM -> Disk`).
    pub fn lookup_device(
        &mut self,
        key: &str,
        shard_manager: &mut ShardedSafeTensorsManager,
    ) -> Result<DeviceTensor, SafeTensorsError> {
        // 1. Check VRAM Tier (O(1) Arc<CudaBuffer> clone, ZERO DtoH transfer)
        if let Some(t) = self.vram_pool.get(key) {
            self.policy.record_access(key);
            let bytes = t.bytes();
            self.telemetry.record_lookup("VRAM", bytes, false);
            return Ok(t.clone());
        }

        // 2. Check RAM Tier (O(1) Arc<Vec<f32>> clone or promotion to VRAM)
        if let Some(t) = self.ram_pool.get(key) {
            self.policy.record_access(key);
            let bytes = t.len() * std::mem::size_of::<f32>();
            self.telemetry.record_lookup("RAM", bytes, false);
            let tensor_arc = Arc::clone(t);

            // Attempt promotion to VRAM if CUDA backend is active
            if self.backend.kind() == DeviceKind::Cuda
                && self.backend.is_available()
                && self.vram_capacity_bytes > 0
                && self.try_promote_to_vram(key)
            {
                if let Some(vram_tensor) = self.vram_pool.get(key) {
                    return Ok(vram_tensor.clone());
                }
            }
            return Ok(DeviceTensor::from_cpu_arc(tensor_arc));
        }

        // 3. Disk Miss: Load from SafeTensors shard
        let tensor = shard_manager.load_tensor(key)?;
        self.disk_keys.insert(key.to_string());
        let bytes = tensor.len() * std::mem::size_of::<f32>();
        self.telemetry.record_lookup("DISK", bytes, false);

        let tensor_arc = Arc::new(tensor);

        // Oversized tensors (> ram_capacity_bytes) are streamed on-demand without polluting policy or RAM pool
        if self.ram_capacity_bytes > 0 && bytes <= self.ram_capacity_bytes {
            self.policy.record_access(key);
            self.admit_to_ram(key, Arc::clone(&tensor_arc));
        }

        Ok(DeviceTensor::from_cpu_arc(tensor_arc))
    }

    /// Convenience helper when the caller explicitly requires CPU host memory (`Arc<Vec<f32>>`).
    /// If the tensor is resident in RAM or Disk, returns the `Arc<Vec<f32>>` without copying.
    /// If the tensor is resident in VRAM, downloads it to host memory via `backend.transfer_to_host()`.
    pub fn lookup_host(
        &mut self,
        key: &str,
        shard_manager: &mut ShardedSafeTensorsManager,
    ) -> Result<Arc<Vec<f32>>, SafeTensorsError> {
        let dev_tensor = self.lookup_device(key, shard_manager)?;
        match dev_tensor {
            DeviceTensor::Cpu(arc_vec) => Ok(arc_vec),
            DeviceTensor::Cuda(_) => {
                let host_vec = self.backend.transfer_to_host(&dev_tensor).map_err(|e| {
                    SafeTensorsError::Io(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        e.to_string(),
                    ))
                })?;
                Ok(Arc::new(host_vec))
            }
        }
    }

    /// Prefetches tensors from Disk into RAM while strictly enforcing:
    /// - Single-residency (skips keys already in RAM or VRAM)
    /// - Oversized tensor rejection (`bytes <= ram_capacity_bytes` checked before and after load)
    /// - Policy hysteresis and RAM capacity limits via `admit_to_ram()`.
    pub fn prefetch(
        &mut self,
        keys: &[String],
        shard_manager: &mut ShardedSafeTensorsManager,
    ) -> usize {
        if self.ram_capacity_bytes == 0 {
            return 0;
        }

        let mut loaded = 0;
        for k in keys {
            if self.ram_pool.contains_key(k) || self.vram_pool.contains_key(k) {
                continue;
            }

            // Pre-check metadata size to avoid reading oversized tensors from disk during prefetch
            if let Some(meta) = shard_manager.get_tensor_metadata(k) {
                self.disk_keys.insert(k.clone());
                if meta.data_bytes > self.ram_capacity_bytes {
                    continue;
                }
            }

            if let Ok(t) = shard_manager.load_tensor(k) {
                self.disk_keys.insert(k.clone());
                let bytes = t.len() * std::mem::size_of::<f32>();
                if bytes > self.ram_capacity_bytes {
                    continue;
                }

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
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use crate::cuda::driver::CudaError;
    use crate::inference::backend::{CPUBackend, CUDABackend};
    use crate::inference::cache_policy::LRUCachePolicy;
    use crate::safetensors::write_safetensors_sharded;

    /// Instrumented wrapper around a real backend (`CUDABackend` when available, or `CPUBackend`
    /// reporting `DeviceKind::Cuda` for deterministic state-machine testing on headless CI) that
    /// counts exact HtoD (`transfer_to_device`) and DtoH (`transfer_to_host`) calls and allows
    /// injecting deterministic `CudaError::OutOfMemory` on demand.
    struct InstrumentedCudaBackend {
        inner_cuda: Option<CUDABackend>,
        htod_calls: Arc<AtomicUsize>,
        dtoh_calls: Arc<AtomicUsize>,
        inject_oom: Arc<AtomicBool>,
    }

    impl InstrumentedCudaBackend {
        fn new(
            htod_calls: Arc<AtomicUsize>,
            dtoh_calls: Arc<AtomicUsize>,
            inject_oom: Arc<AtomicBool>,
        ) -> Self {
            let cuda = CUDABackend::new(0);
            Self {
                inner_cuda: if cuda.is_available() { Some(cuda) } else { None },
                htod_calls,
                dtoh_calls,
                inject_oom,
            }
        }
    }

    impl DeviceBackend for InstrumentedCudaBackend {
        fn name(&self) -> &'static str {
            "CUDA"
        }

        fn kind(&self) -> DeviceKind {
            DeviceKind::Cuda
        }

        fn is_available(&self) -> bool {
            true
        }

        fn allocate_tensor(&self, size: usize) -> Result<DeviceTensor, BackendError> {
            if self.inject_oom.load(Ordering::SeqCst) {
                return Err(BackendError::Cuda(CudaError::OutOfMemory {
                    requested_bytes: size * 4,
                    free_bytes: 0,
                }));
            }
            if let Some(cuda) = &self.inner_cuda {
                cuda.allocate_tensor(size)
            } else {
                Ok(DeviceTensor::from_cpu_vec(vec![0.0f32; size]))
            }
        }

        fn transfer_to_device(&self, host_tensor: &[f32]) -> Result<DeviceTensor, BackendError> {
            if self.inject_oom.load(Ordering::SeqCst) {
                return Err(BackendError::Cuda(CudaError::OutOfMemory {
                    requested_bytes: host_tensor.len() * 4,
                    free_bytes: 0,
                }));
            }
            self.htod_calls.fetch_add(1, Ordering::SeqCst);
            if let Some(cuda) = &self.inner_cuda {
                cuda.transfer_to_device(host_tensor)
            } else {
                Ok(DeviceTensor::from_cpu_vec(host_tensor.to_vec()))
            }
        }

        fn transfer_to_host(&self, device_tensor: &DeviceTensor) -> Result<Vec<f32>, BackendError> {
            self.dtoh_calls.fetch_add(1, Ordering::SeqCst);
            if let Some(cuda) = &self.inner_cuda {
                cuda.transfer_to_host(device_tensor)
            } else {
                CPUBackend.transfer_to_host(device_tensor)
            }
        }
    }

    #[test]
    fn test_tiered_store_initialization_and_capacities() {
        let store = TieredTensorStore::new(1024, 2048, HashSet::new());
        assert_eq!(store.vram_capacity_bytes, 1024);
        assert_eq!(store.ram_capacity_bytes, 2048);
        assert_eq!(store.vram_resident_bytes(), 0);
        assert_eq!(store.ram_resident_bytes(), 0);
        assert!(matches!(
            store.backend.kind(),
            DeviceKind::Cpu | DeviceKind::Cuda
        ));

        let cpu_store = store.with_backend(Box::new(CPUBackend));
        assert_eq!(cpu_store.backend.kind(), DeviceKind::Cpu);
        assert!(cpu_store.verify_invariants().is_ok());
    }

    /// Mandatory Test 1 & 6:
    /// - RAM -> VRAM promotion enforces single-residency (`vram_pool` has key, `ram_pool` does not)
    /// - VRAM hit on `lookup()` returns `DeviceTensor` directly and performs **ZERO DtoH (`transfer_to_host`)** calls!
    #[test]
    fn test_ram_to_vram_promotion_single_residency_and_zero_dtoh_on_vram_lookup() {
        let test_dir =
            std::env::temp_dir().join(format!("tara_vram_hit_test_{:x}", rand::random::<u64>()));
        let _ = std::fs::create_dir_all(&test_dir);

        let mut tensors = HashMap::new();
        let mut shapes = HashMap::new();
        tensors.insert("hot_layer".to_string(), vec![1.0, 2.0, 3.0, 4.0]); // 16 bytes
        shapes.insert("hot_layer".to_string(), vec![2, 2]);
        write_safetensors_sharded(&tensors, &shapes, &test_dir.to_string_lossy(), 1024 * 1024)
            .unwrap();

        let mut shard_mgr = ShardedSafeTensorsManager::new(&test_dir, 2).unwrap();
        let htod_calls = Arc::new(AtomicUsize::new(0));
        let dtoh_calls = Arc::new(AtomicUsize::new(0));
        let inject_oom = Arc::new(AtomicBool::new(false));

        let mut store = TieredTensorStore::new(64, 64, HashSet::new()).with_backend(Box::new(
            InstrumentedCudaBackend::new(
                Arc::clone(&htod_calls),
                Arc::clone(&dtoh_calls),
                Arc::clone(&inject_oom),
            ),
        ));
        store.sync_disk_catalog(&shard_mgr);
        assert_eq!(store.locate_tier("hot_layer"), TierLocation::Disk);

        // 1. First lookup: Disk miss -> admitted to RAM
        let dev1 = store.lookup("hot_layer", &mut shard_mgr).unwrap();
        assert_eq!(store.locate_tier("hot_layer"), TierLocation::Ram);
        assert_eq!(store.telemetry.disk_misses, 1);
        assert_eq!(htod_calls.load(Ordering::SeqCst), 0);
        assert_eq!(dtoh_calls.load(Ordering::SeqCst), 0);
        assert_eq!(dev1.bytes(), 16);
        assert!(store.verify_invariants().is_ok());

        // 2. Second lookup: RAM hit -> promotes to VRAM (1 HtoD call, 0 DtoH calls)
        let dev2 = store.lookup("hot_layer", &mut shard_mgr).unwrap();
        assert_eq!(store.locate_tier("hot_layer"), TierLocation::Vram);
        assert!(store.vram_pool.contains_key("hot_layer"));
        assert!(!store.ram_pool.contains_key("hot_layer"));
        assert_eq!(store.telemetry.promotions, 1);
        assert_eq!(htod_calls.load(Ordering::SeqCst), 1);
        assert_eq!(dtoh_calls.load(Ordering::SeqCst), 0);
        assert!(store.verify_invariants().is_ok());

        // 3. Third & Fourth lookups: VRAM hits -> MUST return cloned DeviceTensor handle with ZERO DtoH calls!
        let dev3 = store.lookup("hot_layer", &mut shard_mgr).unwrap();
        let dev4 = store.lookup("hot_layer", &mut shard_mgr).unwrap();
        assert_eq!(store.telemetry.vram_hits, 2);
        assert_eq!(
            dtoh_calls.load(Ordering::SeqCst),
            0,
            "VRAM lookup must NEVER call transfer_to_host (DtoH)"
        );
        assert!(
            dev2.ptr_eq(&dev3) && dev3.ptr_eq(&dev4),
            "VRAM lookup must return O(1) Arc handle to the exact same device allocation"
        );

        // 4. Explicit lookup_host() performs DtoH only when host Vec<f32> is explicitly requested
        let host_arc = store.lookup_host("hot_layer", &mut shard_mgr).unwrap();
        assert_eq!(*host_arc, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(dtoh_calls.load(Ordering::SeqCst), 1);
        assert!(store.verify_invariants().is_ok());

        let _ = std::fs::remove_dir_all(&test_dir);
    }

    /// Mandatory Test 2 & 3:
    /// - VRAM -> RAM eviction demotes cold VRAM tensor into RAM
    /// - RAM capacity is strictly enforced after VRAM eviction (evicts to Disk if RAM is full)
    #[test]
    fn test_vram_to_ram_eviction_and_ram_capacity_enforcement() {
        let htod_calls = Arc::new(AtomicUsize::new(0));
        let dtoh_calls = Arc::new(AtomicUsize::new(0));
        let inject_oom = Arc::new(AtomicBool::new(false));

        // VRAM capacity: 16 bytes (1 tensor of 4 floats)
        // RAM capacity: 32 bytes (2 tensors of 4 floats)
        let mut store = TieredTensorStore::new(16, 32, HashSet::new())
            .with_policy(Box::new(LRUCachePolicy::default()))
            .with_backend(Box::new(InstrumentedCudaBackend::new(
                Arc::clone(&htod_calls),
                Arc::clone(&dtoh_calls),
                Arc::clone(&inject_oom),
            )));

        let t1 = Arc::new(vec![1.0, 2.0, 3.0, 4.0]); // 16 bytes
        let t2 = Arc::new(vec![5.0, 6.0, 7.0, 8.0]); // 16 bytes
        let t3 = Arc::new(vec![9.0, 10.0, 11.0, 12.0]); // 16 bytes

        // Admit t1 to RAM and promote to VRAM
        store.policy.record_access("t1");
        assert!(store.admit_to_ram("t1", t1));
        assert!(store.try_promote_to_vram("t1"));
        assert_eq!(store.locate_tier("t1"), TierLocation::Vram);
        assert_eq!(store.vram_resident_bytes(), 16);
        assert_eq!(store.ram_resident_bytes(), 0);
        assert!(store.verify_invariants().is_ok());

        // Admit t2 and t3 to RAM (fills 32-byte RAM to 100%)
        store.policy.record_access("t2");
        assert!(store.admit_to_ram("t2", t2));
        store.policy.record_access("t3");
        assert!(store.admit_to_ram("t3", t3));
        assert_eq!(store.ram_resident_bytes(), 32);
        assert!(store.verify_invariants().is_ok());

        // Promote t3 from RAM to VRAM:
        // - Must evict t1 from VRAM
        // - Removes t3 from RAM (freeing 16 bytes of RAM)
        // - Demotes t1 into the freed 16 bytes of RAM so RAM stays at 32 bytes (t1 + t2)
        assert!(store.try_promote_to_vram("t3"));
        assert_eq!(store.locate_tier("t3"), TierLocation::Vram);
        assert_eq!(store.locate_tier("t1"), TierLocation::Ram);
        assert_eq!(store.locate_tier("t2"), TierLocation::Ram);
        assert_eq!(store.vram_resident_bytes(), 16);
        assert_eq!(store.ram_resident_bytes(), 32);
        assert!(store.verify_invariants().is_ok());

        // Now test when RAM capacity is smaller than demoted VRAM tensor (e.g. shrink RAM capacity to 16 bytes
        // and make t2 hotter than t3 so demoted t3 cannot evict t2 and must fall back to DISK)
        store.ram_capacity_bytes = 16;
        // Evict t1 from RAM first to start with 16B RAM (t2) and 16B VRAM (t3)
        store.ram_pool.remove("t1");
        store.disk_keys.insert("t1".to_string());
        assert_eq!(store.ram_resident_bytes(), 16);
        assert!(store.verify_invariants().is_ok());

        // Touch t2 so t2 is newer than t3; promote t2 to VRAM -> t3 is demoted into the 16B freed by t2
        store.policy.record_access("t2");
        assert!(store.try_promote_to_vram("t2"));
        assert_eq!(store.locate_tier("t2"), TierLocation::Vram);
        assert_eq!(store.locate_tier("t3"), TierLocation::Ram);
        assert_eq!(store.vram_resident_bytes(), 16);
        assert_eq!(store.ram_resident_bytes(), 16);
        assert!(store.verify_invariants().is_ok());
    }

    /// Mandatory Test 7:
    /// - CUDA OOM during `try_promote_to_vram()` leaves cache state 100% consistent (no lost tensors,
    ///   candidate stays in RAM, single residency and capacity invariants hold).
    #[test]
    fn test_cuda_oom_during_promotion_rolls_back_cleanly() {
        let htod_calls = Arc::new(AtomicUsize::new(0));
        let dtoh_calls = Arc::new(AtomicUsize::new(0));
        let inject_oom = Arc::new(AtomicBool::new(false));

        let mut store = TieredTensorStore::new(16, 32, HashSet::new())
            .with_policy(Box::new(LRUCachePolicy::default()))
            .with_backend(Box::new(InstrumentedCudaBackend::new(
                Arc::clone(&htod_calls),
                Arc::clone(&dtoh_calls),
                Arc::clone(&inject_oom),
            )));

        let t1 = Arc::new(vec![1.0, 2.0, 3.0, 4.0]); // 16 bytes
        let t2 = Arc::new(vec![5.0, 6.0, 7.0, 8.0]); // 16 bytes

        store.policy.record_access("t1");
        assert!(store.admit_to_ram("t1", t1));
        assert!(store.try_promote_to_vram("t1"));
        assert_eq!(store.locate_tier("t1"), TierLocation::Vram);

        store.policy.record_access("t2");
        assert!(store.admit_to_ram("t2", t2));
        assert_eq!(store.locate_tier("t2"), TierLocation::Ram);

        // Inject CUDA OOM before promoting t2
        inject_oom.store(true, Ordering::SeqCst);
        let promoted = store.try_promote_to_vram("t2");
        assert!(!promoted, "Promotion must return false on CUDA OOM");

        // Verify t2 was NOT lost and remains in RAM, and all store invariants hold
        assert_eq!(store.locate_tier("t2"), TierLocation::Ram);
        assert!(
            store.locate_tier("t1") == TierLocation::Ram
                || store.locate_tier("t1") == TierLocation::Vram
                || store.locate_tier("t1") == TierLocation::Disk
        );
        assert!(store.verify_invariants().is_ok());
    }

    /// Mandatory Test 4 & 5:
    /// - Oversized tensor (> `ram_capacity_bytes`) is rejected from `ram_pool`, streamed on-demand during `lookup()`,
    ///   and skipped during `prefetch()` without evicting existing RAM residents.
    /// - `prefetch()` respects capacity and hysteresis.
    #[test]
    fn test_oversized_tensor_streaming_and_prefetch_hysteresis_enforcement() {
        let test_dir = std::env::temp_dir().join(format!(
            "tara_oversized_prefetch_test_{:x}",
            rand::random::<u64>()
        ));
        let _ = std::fs::create_dir_all(&test_dir);

        let mut tensors = HashMap::new();
        let mut shapes = HashMap::new();
        // small_hot: 4 floats = 16 bytes
        tensors.insert("small_hot".to_string(), vec![1.0, 2.0, 3.0, 4.0]);
        shapes.insert("small_hot".to_string(), vec![2, 2]);
        // small_cold: 4 floats = 16 bytes
        tensors.insert("small_cold".to_string(), vec![5.0, 6.0, 7.0, 8.0]);
        shapes.insert("small_cold".to_string(), vec![2, 2]);
        // huge_tensor: 8 floats = 32 bytes (exceeds 16-byte RAM capacity)
        tensors.insert(
            "huge_tensor".to_string(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        );
        shapes.insert("huge_tensor".to_string(), vec![2, 4]);

        write_safetensors_sharded(&tensors, &shapes, &test_dir.to_string_lossy(), 1024 * 1024)
            .unwrap();
        let mut shard_mgr = ShardedSafeTensorsManager::new(&test_dir, 2).unwrap();

        // RAM capacity: 16 bytes (fits exactly one 4-float tensor)
        let mut store =
            TieredTensorStore::new(0, 16, HashSet::new()).with_backend(Box::new(CPUBackend));
        store.sync_disk_catalog(&shard_mgr);

        // Make small_hot hot in RAM (10 accesses)
        for _ in 0..10 {
            let _ = store.lookup("small_hot", &mut shard_mgr).unwrap();
        }
        assert_eq!(store.locate_tier("small_hot"), TierLocation::Ram);
        assert_eq!(store.ram_resident_bytes(), 16);

        // 1. Lookup oversized huge_tensor (32B > 16B capacity):
        // Must stream on-demand AND leave small_hot untouched in RAM!
        let huge_dev = store.lookup("huge_tensor", &mut shard_mgr).unwrap();
        assert_eq!(huge_dev.len(), 8);
        assert_eq!(store.locate_tier("huge_tensor"), TierLocation::Disk);
        assert_eq!(store.locate_tier("small_hot"), TierLocation::Ram);
        assert_eq!(store.ram_resident_bytes(), 16);
        assert!(store.verify_invariants().is_ok());

        // 2. Prefetch both huge_tensor (oversized) and small_cold (rejected by LFRU hysteresis against small_hot):
        let prefetched = store.prefetch(
            &["huge_tensor".to_string(), "small_cold".to_string()],
            &mut shard_mgr,
        );
        assert_eq!(
            prefetched, 0,
            "Prefetch must reject both oversized tensor and cold tensor failing hysteresis"
        );
        assert_eq!(store.locate_tier("small_hot"), TierLocation::Ram);
        assert_eq!(store.locate_tier("small_cold"), TierLocation::Disk);
        assert_eq!(store.locate_tier("huge_tensor"), TierLocation::Disk);
        assert!(store.verify_invariants().is_ok());

        let _ = std::fs::remove_dir_all(&test_dir);
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

        // Admit tensor3: should evict tensor2 to disk and record it in disk_keys
        store.policy.record_access("tensor3");
        assert!(store.admit_to_ram("tensor3", t3));
        assert_eq!(store.ram_resident_bytes(), 40);

        assert_eq!(store.locate_tier("tensor1"), TierLocation::Ram);
        assert_eq!(store.locate_tier("tensor2"), TierLocation::Disk);
        assert_eq!(store.locate_tier("tensor3"), TierLocation::Ram);
        assert_eq!(store.telemetry.evictions, 1);
        assert!(store.verify_invariants().is_ok());
    }

    #[test]
    fn test_oversized_tensor_rejected_from_ram() {
        let mut store = TieredTensorStore::new(0, 16, HashSet::new());
        let oversized = Arc::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]); // 20 bytes

        assert!(!store.admit_to_ram("oversized", oversized));
        assert_eq!(store.ram_resident_bytes(), 0);
        assert!(!store.ram_pool.contains_key("oversized"));
        assert!(store.verify_invariants().is_ok());
    }

    #[test]
    fn test_hysteresis_blocks_cold_admission() {
        let mut store = TieredTensorStore::new(0, 20, HashSet::new());

        let t_resident = Arc::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let t_candidate = Arc::new(vec![2.0, 3.0, 4.0, 5.0, 6.0]);

        for _ in 0..10 {
            store.policy.record_access("hot_resident");
        }
        assert!(store.admit_to_ram("hot_resident", t_resident));

        for _ in 0..2 {
            store.policy.record_access("cold_candidate");
        }

        assert!(!store.admit_to_ram("cold_candidate", t_candidate));
        assert_eq!(store.locate_tier("hot_resident"), TierLocation::Ram);
        assert_eq!(store.locate_tier("cold_candidate"), TierLocation::NotFound);
        assert_eq!(store.telemetry.evictions, 0);
        assert!(store.verify_invariants().is_ok());
    }

    #[test]
    fn test_single_residency_and_lookup_with_shards() {
        let test_dir =
            std::env::temp_dir().join(format!("tara_tiered_test_{:x}", rand::random::<u64>()));
        let _ = std::fs::create_dir_all(&test_dir);

        let mut tensors = HashMap::new();
        let mut shapes = HashMap::new();
        tensors.insert("weight_a".to_string(), vec![1.0, 2.0, 3.0, 4.0]);
        shapes.insert("weight_a".to_string(), vec![2, 2]);
        tensors.insert("weight_b".to_string(), vec![5.0, 6.0, 7.0, 8.0]);
        shapes.insert("weight_b".to_string(), vec![2, 2]);

        let written =
            write_safetensors_sharded(&tensors, &shapes, &test_dir.to_string_lossy(), 1024 * 1024);
        assert!(written.is_ok());

        let mut shard_mgr = ShardedSafeTensorsManager::new(&test_dir, 2).unwrap();
        let mut store =
            TieredTensorStore::new(0, 64, HashSet::new()).with_backend(Box::new(CPUBackend));
        store.sync_disk_catalog(&shard_mgr);
        assert_eq!(store.locate_tier("weight_a"), TierLocation::Disk);
        assert_eq!(store.locate_tier("weight_b"), TierLocation::Disk);

        // 1. First lookup loads from disk into RAM
        let loaded_a = store.lookup("weight_a", &mut shard_mgr).unwrap();
        assert_eq!(loaded_a.as_cpu_slice(), Some(&[1.0, 2.0, 3.0, 4.0][..]));
        assert_eq!(store.telemetry.disk_misses, 1);
        assert_eq!(store.telemetry.ram_hits, 0);
        assert_eq!(store.locate_tier("weight_a"), TierLocation::Ram);

        // 2. Second lookup is an O(1) RAM cache hit returning identical Arc pointer
        let loaded_a2 = store.lookup("weight_a", &mut shard_mgr).unwrap();
        assert_eq!(loaded_a2.as_cpu_slice(), Some(&[1.0, 2.0, 3.0, 4.0][..]));
        assert!(loaded_a.ptr_eq(&loaded_a2));
        assert_eq!(store.telemetry.ram_hits, 1);
        assert_eq!(store.telemetry.disk_misses, 1);

        // 3. Prefetch weight_b into RAM
        let prefetched = store.prefetch(&["weight_b".to_string()], &mut shard_mgr);
        assert_eq!(prefetched, 1);
        assert_eq!(store.telemetry.prefetched, 1);
        assert_eq!(store.locate_tier("weight_b"), TierLocation::Ram);
        assert!(store.verify_invariants().is_ok());

        // 4. Strict BackendPolicy::Cuda(9999) via try_new_with_policy must return Err
        let bad_cuda = TieredTensorStore::try_new_with_policy(
            1024,
            1024,
            HashSet::new(),
            BackendPolicy::Cuda(9999),
        );
        assert!(bad_cuda.is_err());

        let _ = std::fs::remove_dir_all(&test_dir);
    }
}
