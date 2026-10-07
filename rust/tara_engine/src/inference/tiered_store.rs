//! Three-tier storage hierarchy (VRAM -> RAM -> Disk) with admission hysteresis and eviction.

use std::collections::{HashMap, HashSet};

use crate::inference::backend::{BackendRegistry, DeviceBackend};
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

    // Pools: key -> tensor data
    pub vram_pool: HashMap<String, Vec<f32>>,
    pub ram_pool: HashMap<String, Vec<f32>>,
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
            policy: Box::new(LFRUCachePolicy::default()),
            telemetry: TelemetryMonitor::new(true),
            backend: BackendRegistry::select_best_backend(),
        }
    }

    pub fn vram_resident_bytes(&self) -> usize {
        self.vram_pool
            .values()
            .map(|v| v.len() * std::mem::size_of::<f32>())
            .sum()
    }

    pub fn ram_resident_bytes(&self) -> usize {
        self.ram_pool
            .values()
            .map(|v| v.len() * std::mem::size_of::<f32>())
            .sum()
    }

    pub fn lookup(
        &mut self,
        key: &str,
        shard_manager: &mut ShardedSafeTensorsManager,
    ) -> Result<Vec<f32>, SafeTensorsError> {
        self.policy.record_access(key);

        // 1. Check VRAM Tier
        if let Some(t) = self.vram_pool.get(key) {
            let bytes = t.len() * std::mem::size_of::<f32>();
            self.telemetry.record_lookup("VRAM", bytes, false);
            return Ok(t.clone());
        }

        // 2. Check RAM Tier
        if let Some(t) = self.ram_pool.get(key) {
            let bytes = t.len() * std::mem::size_of::<f32>();
            self.telemetry.record_lookup("RAM", bytes, false);
            let tensor_clone = t.clone();

            // Check promotion to VRAM if GPU available
            if self.backend.name() != "CPU" && self.vram_capacity_bytes > 0 {
                self.try_promote_to_vram(key);
            }
            return Ok(tensor_clone);
        }

        // 3. Disk Miss: Load from SafeTensors shard
        let tensor = shard_manager.load_tensor(key)?;
        let bytes = tensor.len() * std::mem::size_of::<f32>();
        self.telemetry.record_lookup("DISK", bytes, false);

        // Evict from RAM if needed
        while self.ram_resident_bytes() + bytes > self.ram_capacity_bytes
            && !self.ram_pool.is_empty()
        {
            let resident: Vec<String> = self.ram_pool.keys().cloned().collect();
            if let Some(evict_key) = self.policy.pick_eviction(&resident) {
                if let Some(evicted) = self.ram_pool.remove(&evict_key) {
                    let ev_bytes = evicted.len() * std::mem::size_of::<f32>();
                    self.telemetry.record_eviction("RAM", "DISK", ev_bytes);
                }
            } else {
                break;
            }
        }

        self.ram_pool.insert(key.to_string(), tensor.clone());
        Ok(tensor)
    }

    pub fn try_promote_to_vram(&mut self, key: &str) {
        let (bytes, t_data) = match self.ram_pool.get(key) {
            Some(t) => (t.len() * std::mem::size_of::<f32>(), t.clone()),
            None => return,
        };

        while self.vram_resident_bytes() + bytes > self.vram_capacity_bytes
            && !self.vram_pool.is_empty()
        {
            let resident: Vec<String> = self.vram_pool.keys().cloned().collect();
            if let Some(evict_key) = self.policy.pick_eviction(&resident) {
                if self.policy.should_admit(key, &evict_key) {
                    if let Some(evicted) = self.vram_pool.remove(&evict_key) {
                        let ev_bytes = evicted.len() * std::mem::size_of::<f32>();
                        self.telemetry.record_eviction("VRAM", "RAM", ev_bytes);
                        self.ram_pool.insert(evict_key, evicted);
                    }
                } else {
                    return; // Admission rejected due to hysteresis
                }
            } else {
                break;
            }
        }

        if self.vram_resident_bytes() + bytes <= self.vram_capacity_bytes {
            let dev_tensor = self.backend.transfer_to_device(&t_data);
            self.vram_pool.insert(key.to_string(), dev_tensor);
        }
    }

    pub fn prefetch(
        &mut self,
        keys: &[String],
        shard_manager: &mut ShardedSafeTensorsManager,
    ) -> usize {
        let mut loaded = 0;
        for k in keys {
            if !self.ram_pool.contains_key(k) && !self.vram_pool.contains_key(k) {
                if let Ok(t) = shard_manager.load_tensor(k) {
                    self.ram_pool.insert(k.clone(), t);
                    self.telemetry.record_prefetch(1);
                    loaded += 1;
                }
            }
        }
        loaded
    }
}
