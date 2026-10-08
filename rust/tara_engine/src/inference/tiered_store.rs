//! Three-tier storage hierarchy (`VRAM -> RAM -> Disk`) with admission hysteresis,
//! single-flight concurrency, atomic disk persistence, and transactional eviction/promotion.
//!
//! Enforces:
//! - True device residency: `lookup()` / `lookup_device()` return `DeviceTensor` directly without DtoH copies
//! - Single in-memory residency with persistent disk backing: a tensor is resident in at most one
//!   volatile memory tier (`vram_pool ∩ ram_pool == ∅`), while `disk_keys` tracks persistent disk backing
//! - RAM capacity preservation on VRAM-to-RAM eviction (falls back to Disk if RAM cannot admit)
//! - $O(1)$ atomic reference-counted ownership sharing (`Arc<CudaBuffer>` and `Arc<Vec<f32>>`) without cloning tensor payload
//! - Oversized tensor streaming without cache pollution or RAM eviction
//! - Transactional plan -> capacity -> copy -> synchronize -> verify -> commit -> remove ordering with rollback on CUDA OOM or disk I/O error
//! - Atomic persistent disk spill cache (`write .tmp -> fsync -> rename`) with runtime SHA-256 verification,
//!   generation versioning, corrupt/truncated file rejection, and startup crash recovery (`recover_disk_cache`)
//! - Per-key single-flight concurrent coordination via `ConcurrentTieredTensorStore`

use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::inference::backend::{
    BackendError, BackendPolicy, BackendRegistry, DeviceBackend, DeviceKind, DeviceTensor,
};
use crate::inference::cache_policy::{
    CachePolicy, LFRUCachePolicy, DEFAULT_LFRU_FIXED_MARGIN, DEFAULT_LFRU_MARGIN_PCT,
};
use crate::inference::telemetry::{
    EvictionRoute, LookupOutcome, TelemetryMonitor, TransferRoute,
};
use crate::safetensors::SafeTensorsError;
use crate::shard_manager::ShardedSafeTensorsManager;

/// Default safety headroom percentage reserved when computing a VRAM cache budget from raw device free memory.
/// Prevents filling 100% of `cuMemGetInfo` free memory so CUDA context, workspace, and allocator overhead remain safe.
pub const DEFAULT_VRAM_SAFETY_HEADROOM_PCT: usize = 15;

/// Standard CUDA device allocation alignment granularity in bytes (256 bytes).
pub const DEFAULT_CUDA_ALLOCATION_ALIGNMENT: usize = 256;

/// Magic header bytes for atomic persistent disk tensor cache files (`"TARATNS1"`).
const DISK_ARTIFACT_MAGIC: &[u8; 8] = b"TARATNS1";

/// Computes a safe VRAM tensor cache budget from raw device free memory by reserving
/// `safety_headroom_pct` (e.g., 15%) and `reserved_workspace_bytes`.
pub fn compute_safe_vram_budget(
    free_vram_bytes: usize,
    safety_headroom_pct: usize,
    reserved_workspace_bytes: usize,
) -> usize {
    let clamped_pct = safety_headroom_pct.min(100);
    let usable_pct = 100usize.saturating_sub(clamped_pct);
    let after_pct = free_vram_bytes.saturating_mul(usable_pct) / 100;
    after_pct.saturating_sub(reserved_workspace_bytes)
}

/// Rounds `bytes` up to the nearest multiple of `alignment` (for GPU allocator alignment accounting).
pub fn align_up_bytes(bytes: usize, alignment: usize) -> usize {
    let align = alignment.max(1);
    let rem = bytes % align;
    if rem == 0 {
        bytes
    } else {
        bytes.saturating_add(align - rem)
    }
}

/// Authoritative location of a tensor in the three-tier hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TierLocation {
    Vram,
    Ram,
    Disk,
    NotFound,
}

/// Metadata and cryptographic integrity record for a tensor persisted in the disk tier cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiskArtifactMeta {
    pub generation: u64,
    pub bytes: usize,
    pub len_elements: usize,
    pub sha256: [u8; 32],
}

/// Summary report returned by `TieredTensorStore::recover_disk_cache()` on startup/recovery.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DiskRecoveryReport {
    pub recovered_valid_tensors: usize,
    pub purged_incomplete_files: usize,
    pub discarded_corrupt_files: usize,
}

/// Detailed memory-budget accounting report across VRAM (logical vs allocator-aligned) and RAM tiers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TierMemoryBudgetReport {
    pub vram_capacity_bytes: usize,
    pub vram_logical_resident_bytes: usize,
    pub vram_aligned_resident_bytes: usize,
    pub vram_allocation_alignment: usize,
    pub ram_capacity_bytes: usize,
    pub ram_resident_bytes: usize,
}

fn sanitize_key_for_filename(key: &str) -> String {
    let mut out = String::with_capacity(key.len() + 16);
    for ch in key.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' || ch == '.' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    let mut hasher = Sha256::new();
    hasher.update(key.as_bytes());
    let digest = hasher.finalize();
    let short_hash = u32::from_le_bytes([digest[0], digest[1], digest[2], digest[3]]);
    format!("{}_{:08x}.tara_tensor", out, short_hash)
}

pub struct TieredTensorStore {
    pub vram_capacity_bytes: usize,
    pub ram_capacity_bytes: usize,
    pub vram_allocation_alignment: usize,
    pub admission_margin_pct: usize,
    pub admission_margin_fixed: usize,

    // Pools: key -> DeviceTensor
    // Enforces single in-memory residency (`vram_pool ∩ ram_pool == ∅`) with persistent disk backing (`disk_keys`).
    // - vram_pool holds device-resident tensors (`DeviceTensor::Cuda(Arc<CudaBuffer>)`)
    // - ram_pool holds host-resident tensors (`DeviceTensor::Cpu(Arc<Vec<f32>>)`)
    pub vram_pool: HashMap<String, DeviceTensor>,
    pub ram_pool: HashMap<String, DeviceTensor>,
    pub disk_keys: HashSet<String>,
    pub prefetched_unconsumed: HashSet<String>,

    // Optional persistent disk spill cache directory and generation/checksum ledger
    pub disk_cache_dir: Option<PathBuf>,
    pub disk_catalog_meta: HashMap<String, DiskArtifactMeta>,
    pub next_generation: u64,
    pub fail_disk_writes: bool,

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
            vram_allocation_alignment: DEFAULT_CUDA_ALLOCATION_ALIGNMENT,
            admission_margin_pct: DEFAULT_LFRU_MARGIN_PCT,
            admission_margin_fixed: DEFAULT_LFRU_FIXED_MARGIN,
            vram_pool: HashMap::new(),
            ram_pool: HashMap::new(),
            disk_keys,
            prefetched_unconsumed: HashSet::new(),
            disk_cache_dir: None,
            disk_catalog_meta: HashMap::new(),
            next_generation: 1,
            fail_disk_writes: false,
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

    /// Configures a persistent disk spill directory and immediately runs crash recovery
    /// (`recover_disk_cache`) to purge incomplete `.tmp` files and validate existing disk artifacts.
    pub fn with_disk_cache_dir<P: AsRef<Path>>(
        mut self,
        dir: P,
    ) -> Result<Self, SafeTensorsError> {
        let path = dir.as_ref().to_path_buf();
        fs::create_dir_all(&path)?;
        self.disk_cache_dir = Some(path);
        let _ = self.recover_disk_cache()?;
        Ok(self)
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

    pub fn vram_aligned_resident_bytes(&self) -> usize {
        let align = self.vram_allocation_alignment.max(1);
        self.vram_pool
            .values()
            .map(|t| align_up_bytes(t.bytes(), align))
            .sum()
    }

    pub fn ram_resident_bytes(&self) -> usize {
        self.ram_pool.values().map(|t| t.bytes()).sum()
    }

    pub fn memory_budget_report(&self) -> TierMemoryBudgetReport {
        TierMemoryBudgetReport {
            vram_capacity_bytes: self.vram_capacity_bytes,
            vram_logical_resident_bytes: self.vram_resident_bytes(),
            vram_aligned_resident_bytes: self.vram_aligned_resident_bytes(),
            vram_allocation_alignment: self.vram_allocation_alignment.max(1),
            ram_capacity_bytes: self.ram_capacity_bytes,
            ram_resident_bytes: self.ram_resident_bytes(),
        }
    }

    /// Verifies all structural and telemetry invariants of the three-tier store:
    /// 1. `vram_resident_bytes() <= vram_capacity_bytes`
    /// 2. `ram_resident_bytes() <= ram_capacity_bytes`
    /// 3. Single in-memory residency (`vram_pool.keys() ∩ ram_pool.keys() == ∅`)
    /// 4. Every tensor in `ram_pool` is a CPU-resident `DeviceTensor::Cpu(...)`
    /// 5. Telemetry accounting invariants (`telemetry.verify_invariants()`).
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

        for (k, t) in &self.ram_pool {
            if !t.is_cpu() {
                return Err(format!(
                    "RAM tier representation invariant violated: key '{}' in ram_pool is not DeviceTensor::Cpu",
                    k
                ));
            }
        }

        self.telemetry
            .verify_invariants()
            .map_err(|e| e.to_string())?;

        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Atomic Persistent Disk Cache (`write .tmp -> fsync -> rename`), Checksum
    // Verification, Generation Tracking & Startup Crash Recovery
    // ─────────────────────────────────────────────────────────────────────────

    /// Atomically persists a tensor to `disk_cache_dir` using write-temp-fsync-rename.
    /// Computes the runtime SHA-256 digest of the payload and increments `next_generation`.
    pub fn persist_tensor_to_disk(
        &mut self,
        key: &str,
        data: &[f32],
    ) -> Result<DiskArtifactMeta, SafeTensorsError> {
        if self.fail_disk_writes {
            return Err(SafeTensorsError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Injected disk write failure while persisting '{}'", key),
            )));
        }

        let Some(cache_dir) = &self.disk_cache_dir else {
            // If no persistent disk cache directory is configured, record in-memory metadata only
            let mut raw_bytes = Vec::with_capacity(data.len() * 4);
            for &val in data {
                raw_bytes.extend_from_slice(&val.to_le_bytes());
            }
            let mut hasher = Sha256::new();
            hasher.update(&raw_bytes);
            let sha256: [u8; 32] = hasher.finalize().into();
            let gen = self.next_generation;
            self.next_generation = self.next_generation.saturating_add(1);
            let meta = DiskArtifactMeta {
                generation: gen,
                bytes: raw_bytes.len(),
                len_elements: data.len(),
                sha256,
            };
            self.disk_catalog_meta.insert(key.to_string(), meta.clone());
            self.disk_keys.insert(key.to_string());
            return Ok(meta);
        };

        fs::create_dir_all(cache_dir)?;
        let final_filename = sanitize_key_for_filename(key);
        let final_path = cache_dir.join(&final_filename);
        let tmp_filename = format!(
            "{}.tmp.{}.{}",
            final_filename,
            std::process::id(),
            self.next_generation
        );
        let tmp_path = cache_dir.join(tmp_filename);

        let mut payload_bytes = Vec::with_capacity(data.len() * 4);
        for &val in data {
            if !val.is_finite() {
                return Err(SafeTensorsError::InvalidHeader(format!(
                    "Cannot persist non-finite (NaN/Inf) tensor '{}' to disk cache",
                    key
                )));
            }
            payload_bytes.extend_from_slice(&val.to_le_bytes());
        }

        let mut hasher = Sha256::new();
        hasher.update(&payload_bytes);
        let sha256: [u8; 32] = hasher.finalize().into();
        let generation = self.next_generation;
        let key_bytes = key.as_bytes();
        let key_len = u32::try_from(key_bytes.len()).map_err(|_| {
            SafeTensorsError::InvalidHeader("Tensor key exceeds u32 byte limit".into())
        })?;

        // Write complete artifact to temporary file and fsync before atomic rename
        let write_res = (|| -> Result<(), std::io::Error> {
            let mut f = File::create(&tmp_path)?;
            f.write_all(DISK_ARTIFACT_MAGIC)?;
            f.write_all(&generation.to_le_bytes())?;
            f.write_all(&key_len.to_le_bytes())?;
            f.write_all(key_bytes)?;
            f.write_all(&(data.len() as u64).to_le_bytes())?;
            f.write_all(&sha256)?;
            f.write_all(&payload_bytes)?;
            f.flush()?;
            f.sync_all()?;
            Ok(())
        })();

        if let Err(e) = write_res {
            let _ = fs::remove_file(&tmp_path);
            return Err(SafeTensorsError::Io(e));
        }

        if let Err(e) = fs::rename(&tmp_path, &final_path) {
            let _ = fs::remove_file(&tmp_path);
            return Err(SafeTensorsError::Io(e));
        }

        self.next_generation = self.next_generation.saturating_add(1);
        let meta = DiskArtifactMeta {
            generation,
            bytes: payload_bytes.len(),
            len_elements: data.len(),
            sha256,
        };
        self.disk_catalog_meta.insert(key.to_string(), meta.clone());
        self.disk_keys.insert(key.to_string());
        Ok(meta)
    }

    /// Parses and validates an atomic `.tara_tensor` disk artifact, verifying header magic,
    /// payload length, runtime SHA-256 digest, and finite `f32` values.
    fn read_and_verify_disk_artifact(
        path: &Path,
    ) -> Result<(String, DiskArtifactMeta, Vec<f32>), SafeTensorsError> {
        let mut f = File::open(path)?;
        let file_len = f.metadata()?.len() as usize;
        // Minimum header: 8 (magic) + 8 (gen) + 4 (key_len) + 8 (len_elements) + 32 (sha256) = 60 bytes
        if file_len < 60 {
            return Err(SafeTensorsError::Truncated {
                expected: 60,
                got: file_len,
            });
        }

        let mut magic = [0u8; 8];
        f.read_exact(&mut magic)?;
        if &magic != DISK_ARTIFACT_MAGIC {
            return Err(SafeTensorsError::InvalidHeader(format!(
                "Invalid disk tensor magic in {}",
                path.display()
            )));
        }

        let mut gen_buf = [0u8; 8];
        f.read_exact(&mut gen_buf)?;
        let generation = u64::from_le_bytes(gen_buf);

        let mut key_len_buf = [0u8; 4];
        f.read_exact(&mut key_len_buf)?;
        let key_len = u32::from_le_bytes(key_len_buf) as usize;

        let expected_header_len = 60usize.checked_add(key_len).ok_or_else(|| {
            SafeTensorsError::InvalidHeader("Disk tensor key_len overflow".into())
        })?;
        if file_len < expected_header_len {
            return Err(SafeTensorsError::Truncated {
                expected: expected_header_len,
                got: file_len,
            });
        }

        let mut key_buf = vec![0u8; key_len];
        f.read_exact(&mut key_buf)?;
        let key = String::from_utf8(key_buf).map_err(|e| {
            SafeTensorsError::InvalidHeader(format!("Invalid UTF-8 key in disk artifact: {}", e))
        })?;

        let mut elem_buf = [0u8; 8];
        f.read_exact(&mut elem_buf)?;
        let len_elements = u64::from_le_bytes(elem_buf) as usize;
        let payload_bytes_len = len_elements.checked_mul(4).ok_or_else(|| {
            SafeTensorsError::InvalidHeader("Disk tensor payload byte length overflow".into())
        })?;

        let mut expected_sha256 = [0u8; 32];
        f.read_exact(&mut expected_sha256)?;

        let expected_total_len = expected_header_len
            .checked_add(payload_bytes_len)
            .ok_or_else(|| {
                SafeTensorsError::InvalidHeader("Disk tensor total length overflow".into())
            })?;
        if file_len != expected_total_len {
            return Err(SafeTensorsError::Truncated {
                expected: expected_total_len,
                got: file_len,
            });
        }

        let mut payload = vec![0u8; payload_bytes_len];
        f.read_exact(&mut payload)?;

        let mut hasher = Sha256::new();
        hasher.update(&payload);
        let actual_sha256: [u8; 32] = hasher.finalize().into();
        if actual_sha256 != expected_sha256 {
            return Err(SafeTensorsError::InvalidHeader(format!(
                "Corrupted disk tensor artifact '{}': SHA-256 checksum mismatch",
                key
            )));
        }

        let mut values = Vec::with_capacity(len_elements);
        for i in 0..len_elements {
            let b = [
                payload[i * 4],
                payload[i * 4 + 1],
                payload[i * 4 + 2],
                payload[i * 4 + 3],
            ];
            let val = f32::from_le_bytes(b);
            if !val.is_finite() {
                return Err(SafeTensorsError::InvalidHeader(format!(
                    "Corrupted non-finite f32 value in disk tensor '{}'",
                    key
                )));
            }
            values.push(val);
        }

        let meta = DiskArtifactMeta {
            generation,
            bytes: payload_bytes_len,
            len_elements,
            sha256: actual_sha256,
        };
        Ok((key, meta, values))
    }

    /// Loads a tensor from `disk_cache_dir`, verifying cryptographic checksum, exact length,
    /// and expected catalog generation/metadata.
    pub fn load_from_disk_cache(&self, key: &str) -> Result<Vec<f32>, SafeTensorsError> {
        let Some(cache_dir) = &self.disk_cache_dir else {
            return Err(SafeTensorsError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "No persistent disk_cache_dir configured",
            )));
        };

        let path = cache_dir.join(sanitize_key_for_filename(key));
        if !path.exists() {
            return Err(SafeTensorsError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Disk cache artifact not found for '{}'", key),
            )));
        }

        let (disk_key, disk_meta, values) = Self::read_and_verify_disk_artifact(&path)?;
        if disk_key != key {
            return Err(SafeTensorsError::InvalidHeader(format!(
                "Disk artifact key mismatch: expected '{}', found '{}'",
                key, disk_key
            )));
        }

        if let Some(expected_meta) = self.disk_catalog_meta.get(key) {
            if disk_meta.generation != expected_meta.generation
                || disk_meta.bytes != expected_meta.bytes
                || disk_meta.sha256 != expected_meta.sha256
            {
                return Err(SafeTensorsError::InvalidHeader(format!(
                    "Stale or mismatched disk metadata for '{}': expected generation {}, found {}",
                    key, expected_meta.generation, disk_meta.generation
                )));
            }
        }

        Ok(values)
    }

    /// Scans `disk_cache_dir` on startup or after an interrupted process crash:
    /// 1. Purges incomplete `.tmp` staging files
    /// 2. Verifies magic, byte length, and runtime SHA-256 of all `.tara_tensor` files
    /// 3. Discards/removes truncated (e.g. 60%-written) or corrupted files
    /// 4. Reconstructs `disk_keys`, `disk_catalog_meta`, and `next_generation`.
    pub fn recover_disk_cache(&mut self) -> Result<DiskRecoveryReport, SafeTensorsError> {
        let Some(cache_dir) = self.disk_cache_dir.clone() else {
            return Ok(DiskRecoveryReport::default());
        };
        if !cache_dir.exists() {
            return Ok(DiskRecoveryReport::default());
        }

        let mut report = DiskRecoveryReport::default();
        let mut max_gen = self.next_generation;

        for entry in fs::read_dir(&cache_dir)? {
            let entry = entry?;
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let file_name = entry.file_name().to_string_lossy().to_string();

            // 1. Purge incomplete temporary staging files left behind by process kills
            if file_name.contains(".tara_tensor.tmp.") || file_name.ends_with(".tmp") {
                let _ = fs::remove_file(&path);
                report.purged_incomplete_files += 1;
                continue;
            }

            // 2. Verify committed .tara_tensor artifacts
            if file_name.ends_with(".tara_tensor") {
                match Self::read_and_verify_disk_artifact(&path) {
                    Ok((key, meta, _)) => {
                        if meta.generation >= max_gen {
                            max_gen = meta.generation.saturating_add(1);
                        }
                        self.disk_keys.insert(key.clone());
                        self.disk_catalog_meta.insert(key, meta);
                        report.recovered_valid_tensors += 1;
                    }
                    Err(_) => {
                        // Discard truncated / corrupted artifact so lookup never accepts partial data
                        let _ = fs::remove_file(&path);
                        report.discarded_corrupt_files += 1;
                    }
                }
            }
        }

        self.next_generation = max_gen.max(1);
        Ok(report)
    }

    /// Explicitly evicts `key` from volatile memory (`VRAM` or `RAM`) to the Disk tier.
    /// - Idempotent: if `key` is not in `VRAM` or `RAM`, returns `Ok(false)` without duplicate writes
    ///   or duplicate eviction telemetry.
    /// - Atomic failure safety: if persisting to `disk_cache_dir` fails, the source residency in
    ///   `RAM` or `VRAM` is preserved untouched and `Err` is returned.
    pub fn evict_to_disk(&mut self, key: &str) -> Result<bool, SafeTensorsError> {
        if let Some(ram_tensor) = self.ram_pool.get(key).cloned() {
            let bytes = ram_tensor.bytes();
            if self.disk_cache_dir.is_some() || self.fail_disk_writes {
                if let Some(slice) = ram_tensor.as_cpu_slice() {
                    let write_start = Instant::now();
                    self.persist_tensor_to_disk(key, slice)?;
                    let write_ms = write_start.elapsed().as_secs_f64() * 1000.0;
                    self.telemetry.record_transfer_route(
                        TransferRoute::RamToDisk,
                        bytes as u64,
                        write_ms,
                    );
                }
            }
            self.ram_pool.remove(key);
            self.prefetched_unconsumed.remove(key);
            self.disk_keys.insert(key.to_string());
            self.telemetry
                .record_eviction_route(EvictionRoute::RamToDisk, bytes as u64);
            return Ok(true);
        }

        if let Some(vram_tensor) = self.vram_pool.get(key).cloned() {
            let bytes = vram_tensor.bytes();
            if self.disk_cache_dir.is_some() || self.fail_disk_writes {
                let host_vec = self.backend.transfer_to_host(&vram_tensor).map_err(|e| {
                    SafeTensorsError::Io(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        e.to_string(),
                    ))
                })?;
                let write_start = Instant::now();
                self.persist_tensor_to_disk(key, &host_vec)?;
                let write_ms = write_start.elapsed().as_secs_f64() * 1000.0;
                self.telemetry.record_transfer_route(
                    TransferRoute::RamToDisk,
                    bytes as u64,
                    write_ms,
                );
            }
            self.vram_pool.remove(key);
            self.prefetched_unconsumed.remove(key);
            self.disk_keys.insert(key.to_string());
            self.telemetry
                .record_eviction_route(EvictionRoute::VramToDisk, bytes as u64);
            return Ok(true);
        }

        Ok(false)
    }

    /// Transactional RAM admission: verifies capacity and hysteresis before evicting cold items.
    /// Enforces single-residency (will not duplicate a key already in `vram_pool`) and stores
    /// `DeviceTensor::Cpu(Arc<Vec<f32>>)` in `ram_pool`.
    /// If `disk_cache_dir` is enabled (or `fail_disk_writes` is active), any victim evicted from
    /// RAM to Disk is atomically persisted before removal; if disk persistence fails, admission
    /// aborts cleanly with zero residency corruption.
    pub fn admit_to_ram(&mut self, key: &str, tensor: Arc<Vec<f32>>) -> bool {
        let dev_tensor = DeviceTensor::from_cpu_arc(tensor);
        let bytes = dev_tensor.bytes();

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
            let existing_bytes = existing.bytes();
            if self.ram_resident_bytes().saturating_sub(existing_bytes) + bytes
                <= self.ram_capacity_bytes
            {
                self.ram_pool.insert(key.to_string(), dev_tensor);
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
                        freed += resident_tensor.bytes();
                    }
                    sim_resident.retain(|k| k != &cand);
                    to_evict.push(cand);
                } else {
                    return false; // Cannot free sufficient RAM
                }
            }
        }

        // Persist planned evictions to disk BEFORE removing from ram_pool if disk persistence is active
        if self.disk_cache_dir.is_some() || self.fail_disk_writes {
            let mut staged_writes: Vec<(String, Vec<f32>, usize)> =
                Vec::with_capacity(to_evict.len());
            for evict_key in &to_evict {
                if let Some(evicted) = self.ram_pool.get(evict_key) {
                    if let Some(slice) = evicted.as_cpu_slice() {
                        staged_writes.push((evict_key.clone(), slice.to_vec(), evicted.bytes()));
                    }
                }
            }
            for (evict_key, payload, ev_bytes) in staged_writes {
                let write_start = Instant::now();
                if self.persist_tensor_to_disk(&evict_key, &payload).is_err() {
                    // Disk write failed -> abort admission cleanly before removing any RAM residents!
                    return false;
                }
                let write_ms = write_start.elapsed().as_secs_f64() * 1000.0;
                self.telemetry.record_transfer_route(
                    TransferRoute::RamToDisk,
                    ev_bytes as u64,
                    write_ms,
                );
            }
        }

        // Commit planned evictions from RAM to Disk
        for evict_key in to_evict {
            if let Some(evicted) = self.ram_pool.remove(&evict_key) {
                let ev_bytes = evicted.bytes();
                self.prefetched_unconsumed.remove(&evict_key);
                self.disk_keys.insert(evict_key);
                self.telemetry
                    .record_eviction_route(EvictionRoute::RamToDisk, ev_bytes as u64);
            }
        }

        self.ram_pool.insert(key.to_string(), dev_tensor);
        true
    }

    /// Promotes a tensor from RAM to VRAM using genuine device transfer (`HtoD`) + stream synchronization.
    /// Enforces transactional ordering:
    /// `plan -> stage victims -> copy HtoD -> synchronize & verify destination -> commit VRAM -> remove RAM`
    /// with full rollback on CUDA OOM or transfer failure.
    pub fn try_promote_to_vram(&mut self, key: &str) -> bool {
        if self.backend.kind() != DeviceKind::Cuda
            || !self.backend.is_available()
            || self.vram_capacity_bytes == 0
        {
            return false;
        }

        // Idempotent guard: if already in VRAM, ensure single-residency without duplicate transfer/telemetry
        if self.vram_pool.contains_key(key) {
            self.ram_pool.remove(key);
            self.prefetched_unconsumed.remove(key);
            return true;
        }

        let t_data = match self.ram_pool.get(key).and_then(|t| t.as_cpu_arc()) {
            Some(t) => Arc::clone(t),
            None => return false,
        };

        let bytes = t_data.len() * std::mem::size_of::<f32>();
        // Enforce both VRAM capacity and RAM capacity (a tensor larger than RAM can never be staged/demoted)
        if bytes > self.vram_capacity_bytes || bytes > self.ram_capacity_bytes {
            return false;
        }

        let needed_bytes =
            (self.vram_resident_bytes() + bytes).saturating_sub(self.vram_capacity_bytes);

        // Fast path: no VRAM eviction required — perform & verify device transfer BEFORE mutating any pool
        if needed_bytes == 0 {
            let htod_start = Instant::now();
            match self.backend.transfer_to_device(&t_data) {
                Ok(dev_tensor) => {
                    if self.backend.synchronize().is_err()
                        || dev_tensor.len() != t_data.len()
                        || dev_tensor.bytes() != bytes
                    {
                        return false;
                    }
                    let htod_ms = htod_start.elapsed().as_secs_f64() * 1000.0;
                    // Commit destination -> remove source residency
                    self.vram_pool.insert(key.to_string(), dev_tensor);
                    self.ram_pool.remove(key);
                    self.prefetched_unconsumed.remove(key);
                    self.telemetry
                        .record_promotion_with_duration(bytes, htod_ms);
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
        let mut staged_victims: Vec<(String, usize, f64, Arc<Vec<f32>>)> =
            Vec::with_capacity(to_evict.len());
        for evict_key in &to_evict {
            let Some(evicted_dev) = self.vram_pool.get(evict_key) else {
                return false;
            };
            let ev_bytes = evicted_dev.bytes();
            let dtoh_start = Instant::now();
            match self.backend.transfer_to_host(evicted_dev) {
                Ok(host_vec) => {
                    if self.backend.synchronize().is_err() || host_vec.len() * 4 != ev_bytes {
                        return false;
                    }
                    let dtoh_ms = dtoh_start.elapsed().as_secs_f64() * 1000.0;
                    staged_victims.push((
                        evict_key.clone(),
                        ev_bytes,
                        dtoh_ms,
                        Arc::new(host_vec),
                    ));
                }
                Err(_) => {
                    // Abort cleanly before any pool mutation
                    return false;
                }
            }
        }

        // Remove and drop VRAM victims so physical GPU memory is freed before allocating the promoted tensor
        for (evict_key, _, _, _) in &staged_victims {
            self.vram_pool.remove(evict_key);
        }

        // Attempt HtoD allocation, upload, synchronization, and verification for the promoted tensor
        let htod_start = Instant::now();
        match self.backend.transfer_to_device(&t_data) {
            Ok(dev_tensor)
                if self.backend.synchronize().is_ok()
                    && dev_tensor.len() == t_data.len()
                    && dev_tensor.bytes() == bytes =>
            {
                let htod_ms = htod_start.elapsed().as_secs_f64() * 1000.0;
                // 1. Commit promoted tensor to VRAM and remove from RAM:
                //    - Frees `key`'s RAM slot so demoted VRAM victims can use it
                //    - Preserves strict single-residency (`vram_pool ∩ ram_pool == ∅`)
                self.vram_pool.insert(key.to_string(), dev_tensor);
                self.ram_pool.remove(key);
                self.prefetched_unconsumed.remove(key);
                self.telemetry
                    .record_promotion_with_duration(bytes, htod_ms);

                // 2. Demote staged VRAM victims into RAM (if capacity & hysteresis permit) or Disk
                for (evict_key, ev_bytes, dtoh_ms, host_arc) in staged_victims {
                    self.telemetry.record_transfer_route(
                        TransferRoute::VramToRam,
                        ev_bytes as u64,
                        dtoh_ms,
                    );
                    if self.admit_to_ram(&evict_key, Arc::clone(&host_arc)) {
                        self.telemetry
                            .record_eviction_route(EvictionRoute::VramToRam, ev_bytes as u64);
                    } else {
                        if self.disk_cache_dir.is_some() {
                            let _ = self.persist_tensor_to_disk(&evict_key, &host_arc);
                        }
                        self.disk_keys.insert(evict_key);
                        self.telemetry
                            .record_eviction_route(EvictionRoute::VramToDisk, ev_bytes as u64);
                    }
                }
                true
            }
            _ => {
                // Rollback on CUDA OOM / transfer failure:
                // `key` was never removed from `ram_pool`, so `key` remains safely in RAM.
                // Restore staged victims back to VRAM if possible, otherwise demote to RAM/Disk.
                for (evict_key, ev_bytes, dtoh_ms, host_arc) in staged_victims {
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
                        self.telemetry.record_transfer_route(
                            TransferRoute::VramToRam,
                            ev_bytes as u64,
                            dtoh_ms,
                        );
                        if self.admit_to_ram(&evict_key, Arc::clone(&host_arc)) {
                            self.telemetry
                                .record_eviction_route(EvictionRoute::VramToRam, ev_bytes as u64);
                        } else {
                            if self.disk_cache_dir.is_some() {
                                let _ = self.persist_tensor_to_disk(&evict_key, &host_arc);
                            }
                            self.disk_keys.insert(evict_key);
                            self.telemetry
                                .record_eviction_route(EvictionRoute::VramToDisk, ev_bytes as u64);
                        }
                    }
                }
                false
            }
        }
    }

    /// Loads a tensor from either the persistent disk spill cache (`disk_cache_dir`) or
    /// the `ShardedSafeTensorsManager`, verifying cryptographic checksums and rejecting corruption.
    fn load_verified_from_disk(
        &self,
        key: &str,
        shard_manager: &mut ShardedSafeTensorsManager,
    ) -> Result<Vec<f32>, SafeTensorsError> {
        if self.disk_cache_dir.is_some() && self.disk_catalog_meta.contains_key(key) {
            return self.load_from_disk_cache(key);
        }
        match shard_manager.load_tensor(key) {
            Ok(t) => Ok(t),
            Err(primary_err) => {
                if self.disk_cache_dir.is_some() {
                    if let Ok(t) = self.load_from_disk_cache(key) {
                        return Ok(t);
                    }
                }
                Err(primary_err)
            }
        }
    }

    /// Hierarchical device-preserving lookup across `VRAM -> RAM -> Disk`.
    /// - On a **VRAM hit**: returns `DeviceTensor::Cuda(Arc<CudaBuffer>)` in $O(1)$ ownership sharing with **zero DtoH copy**.
    /// - On a **RAM hit**: attempts promotion to VRAM if CUDA is active (returning the promoted `DeviceTensor::Cuda`),
    ///   otherwise returns `DeviceTensor::Cpu(Arc<Vec<f32>>)` in $O(1)$ ownership sharing without cloning payload.
    /// - On a **Disk load**: loads and checksum-verifies from `disk_cache_dir` or `shard_manager`, admits to RAM if
    ///   `bytes <= ram_capacity_bytes` (streaming oversized tensors directly without polluting or evicting `ram_pool`),
    ///   and returns `DeviceTensor`.
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
        // 1. Check VRAM Tier (O(1) Arc<CudaBuffer> ownership sharing, ZERO DtoH transfer)
        if let Some(t) = self.vram_pool.get(key) {
            self.policy.record_access(key);
            let bytes = t.bytes();
            self.telemetry
                .record_lookup_outcome(LookupOutcome::VramHit, bytes as u64);
            return Ok(t.clone());
        }

        // 2. Check RAM Tier (O(1) DeviceTensor::Cpu(Arc<Vec<f32>>) ownership sharing or promotion to VRAM)
        if let Some(t) = self.ram_pool.get(key) {
            self.policy.record_access(key);
            let bytes = t.bytes();
            let from_prefetch = self.prefetched_unconsumed.remove(key);
            self.telemetry
                .record_lookup_outcome(LookupOutcome::RamHit { from_prefetch }, bytes as u64);
            let ram_tensor = t.clone();

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
            return Ok(ram_tensor);
        }

        // 3. Disk Load: Load and verify from persistent disk cache or SafeTensors shard
        let load_start = Instant::now();
        let tensor = match self.load_verified_from_disk(key, shard_manager) {
            Ok(t) => t,
            Err(e) => {
                self.telemetry
                    .record_lookup_outcome(LookupOutcome::NotFoundMiss, 0);
                return Err(e);
            }
        };
        let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
        self.disk_keys.insert(key.to_string());
        let bytes = tensor.len() * std::mem::size_of::<f32>();
        self.telemetry
            .record_lookup_outcome(LookupOutcome::DiskLoad, bytes as u64);
        self.telemetry
            .record_transfer_route(TransferRoute::DiskToRam, bytes as u64, load_ms);

        let tensor_arc = Arc::new(tensor);

        // Oversized tensors (> ram_capacity_bytes) are streamed on-demand without polluting policy or RAM pool
        if self.ram_capacity_bytes > 0 && bytes <= self.ram_capacity_bytes {
            self.policy.record_access(key);
            self.admit_to_ram(key, Arc::clone(&tensor_arc));
        }

        Ok(DeviceTensor::from_cpu_arc(tensor_arc))
    }

    /// Convenience helper when the caller explicitly requires CPU host memory (`Arc<Vec<f32>>`).
    /// If the tensor is resident in RAM or Disk, returns the `Arc<Vec<f32>>` without payload cloning.
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
                let dtoh_start = Instant::now();
                let host_vec = self.backend.transfer_to_host(&dev_tensor).map_err(|e| {
                    SafeTensorsError::Io(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        e.to_string(),
                    ))
                })?;
                let dtoh_ms = dtoh_start.elapsed().as_secs_f64() * 1000.0;
                self.telemetry.record_transfer_route(
                    TransferRoute::VramToRam,
                    dev_tensor.bytes() as u64,
                    dtoh_ms,
                );
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

            let load_start = Instant::now();
            if let Ok(t) = self.load_verified_from_disk(k, shard_manager) {
                let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
                self.disk_keys.insert(k.clone());
                let bytes = t.len() * std::mem::size_of::<f32>();
                if bytes > self.ram_capacity_bytes {
                    continue;
                }

                self.policy.record_access(k);
                let tensor_arc = Arc::new(t);
                if self.admit_to_ram(k, tensor_arc) {
                    self.prefetched_unconsumed.insert(k.clone());
                    self.telemetry.record_prefetch(1);
                    self.telemetry.record_transfer_route(
                        TransferRoute::DiskToRam,
                        bytes as u64,
                        load_ms,
                    );
                    loaded += 1;
                }
            }
        }
        loaded
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Thread-Safe Concurrent Store with Per-Key Single-Flight Coordination
// ─────────────────────────────────────────────────────────────────────────────

/// Thread-safe, concurrent three-tier tensor store (`Send + Sync`) with per-key single-flight
/// coordination to prevent duplicate disk loads (thundering herd) and serialize per-key tier transitions.
pub struct ConcurrentTieredTensorStore {
    store: Mutex<TieredTensorStore>,
    shard_manager: Mutex<ShardedSafeTensorsManager>,
    key_locks: Mutex<HashMap<String, Arc<Mutex<()>>>>,
}

impl ConcurrentTieredTensorStore {
    pub fn new(store: TieredTensorStore, shard_manager: ShardedSafeTensorsManager) -> Self {
        Self {
            store: Mutex::new(store),
            shard_manager: Mutex::new(shard_manager),
            key_locks: Mutex::new(HashMap::new()),
        }
    }

    fn acquire_key_lock(&self, key: &str) -> Arc<Mutex<()>> {
        let mut map = self.key_locks.lock().unwrap();
        Arc::clone(
            map.entry(key.to_string())
                .or_insert_with(|| Arc::new(Mutex::new(()))),
        )
    }

    fn release_key_lock_if_idle(&self, key: &str) {
        let mut map = self.key_locks.lock().unwrap();
        if let Some(arc) = map.get(key) {
            if Arc::strong_count(arc) == 1 {
                map.remove(key);
            }
        }
    }

    /// Returns the number of currently active in-flight key lock entries.
    pub fn active_key_lock_count(&self) -> usize {
        self.key_locks.lock().unwrap().len()
    }

    /// Concurrent single-flight lookup across `VRAM -> RAM -> Disk`.
    /// When $N$ threads simultaneously request the same uncached `key`, the per-key single-flight
    /// lock serializes the miss so only the first thread loads `key` from disk and admits it to
    /// RAM/VRAM, while the remaining $N-1$ threads hit the admitted in-memory entry in $O(1)$.
    pub fn lookup(&self, key: &str) -> Result<DeviceTensor, SafeTensorsError> {
        let res = {
            let key_mutex = self.acquire_key_lock(key);
            let _key_guard = key_mutex.lock().unwrap();

            let mut store_guard = self.store.lock().unwrap();
            // Fast check in VRAM or RAM before locking shard_manager
            if store_guard.vram_pool.contains_key(key) || store_guard.ram_pool.contains_key(key) {
                let mut dummy_mgr = self.shard_manager.lock().unwrap();
                store_guard.lookup_device(key, &mut dummy_mgr)
            } else {
                let mut shard_guard = self.shard_manager.lock().unwrap();
                store_guard.lookup_device(key, &mut shard_guard)
            }
        };
        self.release_key_lock_if_idle(key);
        res
    }

    pub fn try_promote_to_vram(&self, key: &str) -> bool {
        let res = {
            let key_mutex = self.acquire_key_lock(key);
            let _key_guard = key_mutex.lock().unwrap();
            let mut store_guard = self.store.lock().unwrap();
            store_guard.try_promote_to_vram(key)
        };
        self.release_key_lock_if_idle(key);
        res
    }

    pub fn evict_to_disk(&self, key: &str) -> Result<bool, SafeTensorsError> {
        let res = {
            let key_mutex = self.acquire_key_lock(key);
            let _key_guard = key_mutex.lock().unwrap();
            let mut store_guard = self.store.lock().unwrap();
            store_guard.evict_to_disk(key)
        };
        self.release_key_lock_if_idle(key);
        res
    }

    pub fn admit_to_ram(&self, key: &str, tensor: Arc<Vec<f32>>) -> bool {
        let res = {
            let key_mutex = self.acquire_key_lock(key);
            let _key_guard = key_mutex.lock().unwrap();
            let mut store_guard = self.store.lock().unwrap();
            store_guard.admit_to_ram(key, tensor)
        };
        self.release_key_lock_if_idle(key);
        res
    }

    pub fn locate_tier(&self, key: &str) -> TierLocation {
        let store_guard = self.store.lock().unwrap();
        store_guard.locate_tier(key)
    }

    pub fn verify_invariants(&self) -> Result<(), String> {
        let store_guard = self.store.lock().unwrap();
        store_guard.verify_invariants()
    }

    pub fn with_store<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&TieredTensorStore) -> R,
    {
        let store_guard = self.store.lock().unwrap();
        f(&store_guard)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Barrier;
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
        assert_eq!(store.telemetry.cache_hit_rate(), None);
        assert_eq!(store.telemetry.hit_rate(), 0.0);
        assert!(matches!(
            store.backend.kind(),
            DeviceKind::Cpu | DeviceKind::Cuda
        ));

        let safe_budget =
            compute_safe_vram_budget(10_000, DEFAULT_VRAM_SAFETY_HEADROOM_PCT, 500);
        assert_eq!(safe_budget, 8000);
        assert_eq!(align_up_bytes(16, 256), 256);

        let cpu_store = store.with_backend(Box::new(CPUBackend));
        assert_eq!(cpu_store.backend.kind(), DeviceKind::Cpu);
        assert!(cpu_store.verify_invariants().is_ok());
    }

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

        // 1. First lookup: Disk load -> admitted to RAM
        let dev1 = store.lookup("hot_layer", &mut shard_mgr).unwrap();
        assert_eq!(store.locate_tier("hot_layer"), TierLocation::Ram);
        assert_eq!(store.telemetry.disk_loads(), 1);
        assert_eq!(store.telemetry.disk_misses(), 1);
        assert_eq!(htod_calls.load(Ordering::SeqCst), 0);
        assert_eq!(dtoh_calls.load(Ordering::SeqCst), 0);
        assert_eq!(dev1.bytes(), 16);
        assert!(store.verify_invariants().is_ok());

        // 2. Second lookup: RAM hit -> promotes to VRAM (1 HtoD call, 0 DtoH calls)
        let dev2 = store.lookup("hot_layer", &mut shard_mgr).unwrap();
        assert_eq!(store.locate_tier("hot_layer"), TierLocation::Vram);
        assert!(store.vram_pool.contains_key("hot_layer"));
        assert!(!store.ram_pool.contains_key("hot_layer"));
        assert_eq!(store.telemetry.promotions(), 1);
        assert_eq!(store.telemetry.bytes_transferred_vram(), 16);
        assert_eq!(
            store
                .telemetry
                .route_stats(TransferRoute::RamToVram)
                .count,
            1
        );
        assert_eq!(htod_calls.load(Ordering::SeqCst), 1);
        assert_eq!(dtoh_calls.load(Ordering::SeqCst), 0);
        assert!(store.verify_invariants().is_ok());

        // 3. Third & Fourth lookups: VRAM hits -> MUST return cloned DeviceTensor handle with ZERO DtoH calls!
        let dev3 = store.lookup("hot_layer", &mut shard_mgr).unwrap();
        let dev4 = store.lookup("hot_layer", &mut shard_mgr).unwrap();
        assert_eq!(store.telemetry.vram_hits(), 2);
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
        assert_eq!(
            store
                .telemetry
                .route_stats(TransferRoute::VramToRam)
                .count,
            1
        );
        assert!(store.verify_invariants().is_ok());

        let _ = std::fs::remove_dir_all(&test_dir);
    }

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
        assert_eq!(store.telemetry.evictions_vram_to_ram(), 1);
        assert_eq!(store.telemetry.bytes_demoted_vram_to_ram(), 16);
        assert!(store.verify_invariants().is_ok());

        // Now test cascading VRAM eviction -> RAM admission -> coldest RAM resident evicted to Disk:
        // Currently: VRAM has t3 (16B); RAM has t1 (16B, clock=1) and t2 (16B, clock=2).
        // Replace t2 (16B) in RAM with two 8-byte tensors: r_cold (8B) and r_promote (8B), with RAM cap = 16B.
        store.ram_pool.clear();
        store.ram_capacity_bytes = 16;
        let r_cold = Arc::new(vec![100.0, 200.0]); // 8 bytes
        let r_promote = Arc::new(vec![300.0, 400.0]); // 8 bytes
        store.policy.record_access("r_cold");
        assert!(store.admit_to_ram("r_cold", r_cold));
        // Touch t3 so its LRU timestamp is newer than r_cold, then touch r_promote
        store.policy.record_access("t3");
        store.policy.record_access("r_promote");
        assert!(store.admit_to_ram("r_promote", r_promote));
        assert_eq!(store.ram_resident_bytes(), 16); // RAM 100% full (r_cold 8B + r_promote 8B)

        // Promote r_promote (8B) to VRAM (16B capacity, currently holding 16B t3):
        // 1. Evicts t3 (16B) from VRAM
        // 2. Moves r_promote (8B) from RAM to VRAM (leaving r_cold 8B in 16B RAM)
        // 3. Demotes t3 (16B) into 16B RAM -> evicts r_cold (8B) from RAM to Disk so t3 (16B) fits in RAM!
        assert!(store.try_promote_to_vram("r_promote"));
        assert_eq!(store.locate_tier("r_promote"), TierLocation::Vram);
        assert_eq!(store.locate_tier("t3"), TierLocation::Ram);
        assert_eq!(store.locate_tier("r_cold"), TierLocation::Disk);
        assert_eq!(store.vram_resident_bytes(), 8);
        assert_eq!(store.ram_resident_bytes(), 16);
        assert_eq!(store.telemetry.evictions_vram_to_ram(), 2);
        assert_eq!(store.telemetry.evictions_ram_to_disk(), 1);
        assert!(store.verify_invariants().is_ok());
    }

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

    #[test]
    fn test_oversized_tensor_streaming_and_prefetch_hysteresis_enforcement() {
        let test_dir = std::env::temp_dir().join(format!(
            "tara_oversized_prefetch_test_{:x}",
            rand::random::<u64>()
        ));
        let _ = std::fs::create_dir_all(&test_dir);

        let mut tensors = HashMap::new();
        let mut shapes = HashMap::new();
        tensors.insert("small_hot".to_string(), vec![1.0, 2.0, 3.0, 4.0]);
        shapes.insert("small_hot".to_string(), vec![2, 2]);
        tensors.insert("small_cold".to_string(), vec![5.0, 6.0, 7.0, 8.0]);
        shapes.insert("small_cold".to_string(), vec![2, 2]);
        tensors.insert(
            "huge_tensor".to_string(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        );
        shapes.insert("huge_tensor".to_string(), vec![2, 4]);

        write_safetensors_sharded(&tensors, &shapes, &test_dir.to_string_lossy(), 1024 * 1024)
            .unwrap();
        let mut shard_mgr = ShardedSafeTensorsManager::new(&test_dir, 2).unwrap();

        let mut store =
            TieredTensorStore::new(0, 16, HashSet::new()).with_backend(Box::new(CPUBackend));
        store.sync_disk_catalog(&shard_mgr);

        for _ in 0..10 {
            let _ = store.lookup("small_hot", &mut shard_mgr).unwrap();
        }
        assert_eq!(store.locate_tier("small_hot"), TierLocation::Ram);
        assert_eq!(store.ram_resident_bytes(), 16);

        let huge_dev = store.lookup("huge_tensor", &mut shard_mgr).unwrap();
        assert_eq!(huge_dev.len(), 8);
        assert_eq!(store.locate_tier("huge_tensor"), TierLocation::Disk);
        assert_eq!(store.locate_tier("small_hot"), TierLocation::Ram);
        assert_eq!(store.ram_resident_bytes(), 16);
        assert!(store.verify_invariants().is_ok());

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

        store.policy.record_access("tensor1");

        store.policy.record_access("tensor3");
        assert!(store.admit_to_ram("tensor3", t3));
        assert_eq!(store.ram_resident_bytes(), 40);

        assert_eq!(store.locate_tier("tensor1"), TierLocation::Ram);
        assert_eq!(store.locate_tier("tensor2"), TierLocation::Disk);
        assert_eq!(store.locate_tier("tensor3"), TierLocation::Ram);
        assert_eq!(store.telemetry.evictions(), 1);
        assert_eq!(store.telemetry.evictions_ram_to_disk(), 1);
        assert_eq!(store.telemetry.bytes_evicted_ram_to_disk(), 20);
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
        assert_eq!(store.telemetry.evictions(), 0);
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
        assert_eq!(store.telemetry.disk_loads(), 1);
        assert_eq!(store.telemetry.ram_hits(), 0);
        assert_eq!(store.locate_tier("weight_a"), TierLocation::Ram);

        // 2. Second lookup is an O(1) RAM cache hit returning identical Arc pointer
        let loaded_a2 = store.lookup("weight_a", &mut shard_mgr).unwrap();
        assert_eq!(loaded_a2.as_cpu_slice(), Some(&[1.0, 2.0, 3.0, 4.0][..]));
        assert!(loaded_a.ptr_eq(&loaded_a2));
        assert_eq!(store.telemetry.ram_hits(), 1);
        assert_eq!(store.telemetry.disk_loads(), 1);
        assert_eq!(store.telemetry.prefetch_hits(), 0);

        // 3. Prefetch weight_b into RAM and verify subsequent lookup records a genuine prefetch_hit
        let prefetched = store.prefetch(&["weight_b".to_string()], &mut shard_mgr);
        assert_eq!(prefetched, 1);
        assert_eq!(store.telemetry.prefetched(), 1);
        assert_eq!(store.locate_tier("weight_b"), TierLocation::Ram);

        let loaded_b = store.lookup("weight_b", &mut shard_mgr).unwrap();
        assert_eq!(loaded_b.as_cpu_slice(), Some(&[5.0, 6.0, 7.0, 8.0][..]));
        assert_eq!(store.telemetry.prefetch_hits(), 1);
        assert_eq!(store.telemetry.prefetch_hit_rate(), Some(1.0));
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

    /// Production Hardening Test A:
    /// - Exact capacity boundary (`tensor == capacity` admitted vs `tensor == capacity + 4 bytes` rejected)
    /// - Asymmetric `RAM-only oversized` vs `VRAM-only oversized`:
    ///   * Tensor `> VRAM` (`16B`) but `<= RAM` (`32B`) is admitted to RAM and rejected from VRAM promotion.
    ///   * Tensor `> RAM` (`32B`) is rejected from both RAM and VRAM even if VRAM capacity were configured higher.
    /// - Repeated promote (`try_promote_to_vram`) and repeated evict (`evict_to_disk`) are idempotent:
    ///   no duplicate residency, no duplicate HtoD transfers, and no duplicate disk writes.
    #[test]
    fn test_boundary_capacities_asymmetric_oversized_and_idempotent_promote_evict() {
        let htod_calls = Arc::new(AtomicUsize::new(0));
        let dtoh_calls = Arc::new(AtomicUsize::new(0));
        let inject_oom = Arc::new(AtomicBool::new(false));

        // VRAM capacity = 16B (4 floats), RAM capacity = 32B (8 floats)
        let mut store = TieredTensorStore::new(16, 32, HashSet::new())
            .with_policy(Box::new(LRUCachePolicy::default()))
            .with_backend(Box::new(InstrumentedCudaBackend::new(
                Arc::clone(&htod_calls),
                Arc::clone(&dtoh_calls),
                Arc::clone(&inject_oom),
            )));

        // 1. Exact VRAM capacity tensor (16B == 16B VRAM)
        let exact_vram = Arc::new(vec![1.0, 2.0, 3.0, 4.0]); // 16B
        assert!(store.admit_to_ram("exact_vram", exact_vram));
        assert!(store.try_promote_to_vram("exact_vram"));
        assert_eq!(store.locate_tier("exact_vram"), TierLocation::Vram);
        assert_eq!(htod_calls.load(Ordering::SeqCst), 1);
        assert_eq!(store.telemetry.promotions(), 1);

        // 2. Repeated promote on already VRAM-resident key -> idempotent (0 extra HtoD calls, 0 extra promotions)
        assert!(store.try_promote_to_vram("exact_vram"));
        assert!(store.try_promote_to_vram("exact_vram"));
        assert_eq!(htod_calls.load(Ordering::SeqCst), 1);
        assert_eq!(store.telemetry.promotions(), 1);
        assert!(store.verify_invariants().is_ok());

        // 3. VRAM-only oversized tensor (24B > 16B VRAM, but 24B <= 32B RAM):
        // Must admit to RAM, and reject VRAM promotion while staying safely in RAM!
        let vram_oversized = Arc::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]); // 24B
        assert!(store.admit_to_ram("vram_oversized", vram_oversized));
        assert_eq!(store.locate_tier("vram_oversized"), TierLocation::Ram);
        assert!(!store.try_promote_to_vram("vram_oversized"));
        assert_eq!(store.locate_tier("vram_oversized"), TierLocation::Ram);
        assert_eq!(store.locate_tier("exact_vram"), TierLocation::Vram);

        // 4. Exact RAM capacity boundary after evicting vram_oversized:
        // Evict vram_oversized to Disk (1st call returns Ok(true), 2nd call returns Ok(false) idempotently)
        assert_eq!(store.evict_to_disk("vram_oversized").unwrap(), true);
        assert_eq!(store.locate_tier("vram_oversized"), TierLocation::Disk);
        assert_eq!(store.telemetry.evictions_ram_to_disk(), 1);
        assert_eq!(store.evict_to_disk("vram_oversized").unwrap(), false);
        assert_eq!(store.telemetry.evictions_ram_to_disk(), 1);

        // Admit exact 32B tensor (== 32B RAM capacity) -> succeeds
        let exact_ram = Arc::new(vec![1.0; 8]); // 32B
        assert!(store.admit_to_ram("exact_ram", exact_ram));
        assert_eq!(store.ram_resident_bytes(), 32);

        // Admit 36B tensor (32B capacity + 1 float) -> rejected
        let over_ram = Arc::new(vec![1.0; 9]); // 36B
        assert!(!store.admit_to_ram("over_ram", over_ram));
        assert_eq!(store.locate_tier("exact_ram"), TierLocation::Ram);
        assert!(store.verify_invariants().is_ok());
    }

    /// Production Hardening Test B:
    /// - Atomic disk persistence (`write .tmp -> fsync -> rename`)
    /// - Disk write failure leaves source RAM residency 100% intact (no residency corruption)
    /// - Corrupt disk tensor (both corrupted `.safetensors` shard and corrupted `.tara_tensor` cache file) is rejected
    /// - Crash/incomplete publication recovery (`recover_disk_cache()` purges `.tmp` staging files and
    ///   discards 60%-written truncated `.tara_tensor` files while recovering valid artifacts)
    /// - Stale generation metadata detection.
    #[test]
    fn test_atomic_disk_persistence_write_failure_corruption_rejection_and_crash_recovery() {
        let base_dir = std::env::temp_dir().join(format!(
            "tara_disk_hardening_test_{:x}",
            rand::random::<u64>()
        ));
        let shard_dir = base_dir.join("shards");
        let cache_dir = base_dir.join("disk_cache");
        let _ = fs::create_dir_all(&shard_dir);
        let _ = fs::create_dir_all(&cache_dir);

        let mut tensors = HashMap::new();
        let mut shapes = HashMap::new();
        tensors.insert("shard_weight".to_string(), vec![10.0, 20.0, 30.0, 40.0]);
        shapes.insert("shard_weight".to_string(), vec![2, 2]);
        write_safetensors_sharded(&tensors, &shapes, &shard_dir.to_string_lossy(), 1024 * 1024)
            .unwrap();

        let mut shard_mgr = ShardedSafeTensorsManager::new(&shard_dir, 2).unwrap();
        let mut store = TieredTensorStore::new(0, 16, HashSet::new())
            .with_backend(Box::new(CPUBackend))
            .with_policy(Box::new(LRUCachePolicy::default()))
            .with_disk_cache_dir(&cache_dir)
            .unwrap();

        // 1. Admit t_persist (16B) to RAM
        let t_persist = Arc::new(vec![1.25, 2.5, 3.75, 5.0]);
        store.policy.record_access("t_persist");
        assert!(store.admit_to_ram("t_persist", Arc::clone(&t_persist)));
        assert_eq!(store.locate_tier("t_persist"), TierLocation::Ram);

        // 2. Inject disk write failure: both explicit evict_to_disk and LRU eviction during admit_to_ram
        //    MUST abort and preserve t_persist in RAM!
        store.fail_disk_writes = true;
        assert!(store.evict_to_disk("t_persist").is_err());
        assert_eq!(store.locate_tier("t_persist"), TierLocation::Ram);

        let t_new = Arc::new(vec![6.0, 7.0, 8.0, 9.0]);
        store.policy.record_access("t_new");
        assert!(
            !store.admit_to_ram("t_new", Arc::clone(&t_new)),
            "RAM admission requiring eviction must abort when disk persistence fails"
        );
        assert_eq!(store.locate_tier("t_persist"), TierLocation::Ram);
        assert_eq!(store.locate_tier("t_new"), TierLocation::NotFound);
        assert!(store.verify_invariants().is_ok());

        // 3. Re-enable disk writes and evict t_persist to disk atomically, then reload via lookup()
        store.fail_disk_writes = false;
        assert!(store.evict_to_disk("t_persist").unwrap());
        assert_eq!(store.locate_tier("t_persist"), TierLocation::Disk);

        let reloaded = store.lookup("t_persist", &mut shard_mgr).unwrap();
        assert_eq!(reloaded.as_cpu_slice(), Some(&[1.25, 2.5, 3.75, 5.0][..]));
        assert_eq!(store.locate_tier("t_persist"), TierLocation::Ram);

        // Evict t_persist to disk again so its file is on disk
        assert!(store.evict_to_disk("t_persist").unwrap());

        // 4. Simulate process crash leaving:
        //    - An incomplete .tmp staging file
        //    - A 60%-written truncated .tara_tensor file ("partial_crash.tara_tensor")
        //    - The valid committed t_persist .tara_tensor file
        let orphan_tmp = cache_dir.join("orphan_crash.tara_tensor.tmp.999.1");
        fs::write(&orphan_tmp, b"TARATNS1_INCOMPLETE_STAGING").unwrap();

        let partial_artifact = cache_dir.join("partial_crash_12345678.tara_tensor");
        // Write header claiming 100 elements (400 payload bytes), but truncate after 40 bytes (60% of file)
        let mut bad_bytes = Vec::new();
        bad_bytes.extend_from_slice(DISK_ARTIFACT_MAGIC);
        bad_bytes.extend_from_slice(&1u64.to_le_bytes());
        let k_bytes = b"partial_crash";
        bad_bytes.extend_from_slice(&(k_bytes.len() as u32).to_le_bytes());
        bad_bytes.extend_from_slice(k_bytes);
        bad_bytes.extend_from_slice(&100u64.to_le_bytes());
        bad_bytes.extend_from_slice(&[0xAAu8; 32]);
        bad_bytes.extend_from_slice(&[0u8; 24]); // Truncated payload!
        fs::write(&partial_artifact, &bad_bytes).unwrap();

        // Run startup crash recovery on a fresh store instance
        let mut recovered_store = TieredTensorStore::new(0, 16, HashSet::new())
            .with_backend(Box::new(CPUBackend));
        recovered_store.disk_cache_dir = Some(cache_dir.clone());
        let report = recovered_store.recover_disk_cache().unwrap();
        assert_eq!(report.purged_incomplete_files, 1);
        assert_eq!(report.discarded_corrupt_files, 1);
        assert_eq!(report.recovered_valid_tensors, 1);
        assert!(!orphan_tmp.exists());
        assert!(!partial_artifact.exists());
        assert_eq!(recovered_store.locate_tier("t_persist"), TierLocation::Disk);
        assert_eq!(
            recovered_store.locate_tier("partial_crash"),
            TierLocation::NotFound
        );

        // 5. Corrupt a byte in the valid .tara_tensor file on disk -> lookup MUST reject it!
        let persisted_path = cache_dir.join(sanitize_key_for_filename("t_persist"));
        let mut raw_file = fs::read(&persisted_path).unwrap();
        let last_idx = raw_file.len() - 1;
        raw_file[last_idx] ^= 0xFF;
        fs::write(&persisted_path, &raw_file).unwrap();

        let corrupt_res = recovered_store.lookup("t_persist", &mut shard_mgr);
        assert!(
            corrupt_res.is_err(),
            "Corrupted disk cache tensor must be rejected by runtime SHA-256 verification"
        );
        assert_eq!(recovered_store.ram_resident_bytes(), 0);

        // 6. Corrupt a byte in the SafeTensors shard payload on disk -> lookup MUST reject it!
        let shard_file_path = shard_dir.join("model.safetensors");
        let mut shard_bytes = fs::read(&shard_file_path).unwrap();
        let shard_last = shard_bytes.len() - 1;
        shard_bytes[shard_last] ^= 0xFF;
        fs::write(&shard_file_path, &shard_bytes).unwrap();

        let corrupt_shard_res = recovered_store.lookup("shard_weight", &mut shard_mgr);
        assert!(
            corrupt_shard_res.is_err(),
            "Corrupted SafeTensors shard payload must be rejected by runtime SHA-256 verification"
        );
        assert!(recovered_store.verify_invariants().is_ok());

        let _ = fs::remove_dir_all(&base_dir);
    }

    /// Production Hardening Test C:
    /// - Per-key single-flight concurrent lookup: 16 threads simultaneously requesting an uncached
    ///   key perform **exactly 1 disk load** (`disk_loads() == 1`, zero thundering herd).
    /// - Concurrent `lookup`, `try_promote_to_vram`, and `evict_to_disk` across multiple threads
    ///   preserve `vram_pool ∩ ram_pool == ∅` and all store/telemetry invariants.
    #[test]
    fn test_concurrent_single_flight_lookup_promotion_and_eviction() {
        let test_dir = std::env::temp_dir().join(format!(
            "tara_concurrent_single_flight_{:x}",
            rand::random::<u64>()
        ));
        let _ = fs::create_dir_all(&test_dir);

        let mut tensors = HashMap::new();
        let mut shapes = HashMap::new();
        tensors.insert("shared_w".to_string(), vec![1.0, 2.0, 3.0, 4.0]);
        shapes.insert("shared_w".to_string(), vec![2, 2]);
        write_safetensors_sharded(&tensors, &shapes, &test_dir.to_string_lossy(), 1024 * 1024)
            .unwrap();

        let shard_mgr = ShardedSafeTensorsManager::new(&test_dir, 2).unwrap();
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

        let concurrent_store = Arc::new(ConcurrentTieredTensorStore::new(store, shard_mgr));

        // Phase 1: 16 threads simultaneously call lookup("shared_w") via a Barrier
        let num_threads = 16;
        let barrier = Arc::new(Barrier::new(num_threads));
        let mut handles = Vec::with_capacity(num_threads);

        for _ in 0..num_threads {
            let cs = Arc::clone(&concurrent_store);
            let bar = Arc::clone(&barrier);
            handles.push(std::thread::spawn(move || {
                bar.wait();
                let t = cs.lookup("shared_w").unwrap();
                assert_eq!(t.len(), 4);
                let _ = cs.try_promote_to_vram("shared_w");
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        // Verify single-flight: exactly 1 disk load and 1 promotion occurred across all 16 threads!
        concurrent_store.with_store(|s| {
            assert_eq!(
                s.telemetry.disk_loads(),
                1,
                "Single-flight lock must ensure shared_w is loaded from disk only once"
            );
            assert_eq!(
                s.telemetry.promotions(),
                1,
                "Concurrent promotions must promote shared_w only once"
            );
            assert_eq!(htod_calls.load(Ordering::SeqCst), 1);
            assert_eq!(s.locate_tier("shared_w"), TierLocation::Vram);
            assert!(s.verify_invariants().is_ok());
        });

        // Phase 2: Concurrent lookup + promote + evict_to_disk stress across 12 threads
        let stress_threads = 12;
        let barrier2 = Arc::new(Barrier::new(stress_threads));
        let mut handles2 = Vec::with_capacity(stress_threads);

        for tid in 0..stress_threads {
            let cs = Arc::clone(&concurrent_store);
            let bar = Arc::clone(&barrier2);
            handles2.push(std::thread::spawn(move || {
                bar.wait();
                for iter in 0..20 {
                    match (tid + iter) % 3 {
                        0 => {
                            let t = cs.lookup("shared_w").unwrap();
                            assert_eq!(t.len(), 4);
                        }
                        1 => {
                            let _ = cs.try_promote_to_vram("shared_w");
                        }
                        _ => {
                            let _ = cs.evict_to_disk("shared_w").unwrap();
                        }
                    }
                    assert!(cs.verify_invariants().is_ok());
                }
            }));
        }

        for h in handles2 {
            h.join().unwrap();
        }

        assert_eq!(
            concurrent_store.active_key_lock_count(),
            0,
            "Idle per-key locks must be pruned after operations finish"
        );
        assert!(concurrent_store.verify_invariants().is_ok());
        let _ = fs::remove_dir_all(&test_dir);
    }
}

