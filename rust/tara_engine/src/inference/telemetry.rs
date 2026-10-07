//! Production-grade inference telemetry, tier allocation monitoring, route transfer profiling,
//! and latency distribution tracking.
//!
//! Guarantees:
//! - Strongly-typed `StorageTier`, `LookupOutcome`, `EvictionRoute`, and `TransferRoute` (zero string allocation on hot path)
//! - Explicit rejection of unknown tier strings via `TelemetryError::UnknownTier` (never silently counted as disk misses)
//! - Strict separation of `vram_hits`, `ram_hits`, `disk_loads`, and `not_found_misses` with invariant:
//!   `requests == vram_hits + ram_hits + disk_loads + not_found_misses`
//! - `cache_hit_rate()` returns `None` on empty workload (`0` requests), and scalar `hit_rate()` returns `0.0` (never `1.0`)
//! - Lock-free `AtomicU64` monotonic counters safe for concurrent multi-threaded inference (`Send + Sync`)
//! - Directional eviction counters and byte volumes (`VRAM->RAM`, `VRAM->DISK`, `RAM->DISK`)
//! - Per-route transfer statistics (`DiskToRam`, `DiskToVram`, `RamToVram`, `VramToRam`, `RamToDisk`) and
//!   bounded latency reservoir for `p50`, `p95`, `p99` tail-latency percentiles
//! - Explicit prefetch-hit accounting (only RAM/VRAM cache hits on prefetched tensors count as `prefetch_hits`)
//! - Elapsed time, throughput (`requests_per_sec`, `disk_mib_per_sec`, `vram_transfer_mib_per_sec`), and `reset()` lifecycle

use std::collections::HashMap;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;
use thiserror::Error;

/// Default maximum number of recent transfer latency samples retained for percentile calculation.
pub const DEFAULT_LATENCY_RESERVOIR_CAPACITY: usize = 4096;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum TelemetryError {
    #[error("Unknown storage tier '{0}' (expected VRAM, RAM, or DISK)")]
    UnknownTier(String),
    #[error("Invalid eviction route from {from:?} to {to:?}")]
    InvalidEvictionRoute { from: StorageTier, to: StorageTier },
    #[error("Invalid prefetch hit on non-cache outcome {0:?}")]
    InvalidPrefetchHitOutcome(LookupOutcome),
    #[error("Telemetry invariant violation: {0}")]
    InvariantViolation(String),
}

/// Physical storage tier in the three-tier hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StorageTier {
    Vram,
    Ram,
    Disk,
}

impl FromStr for StorageTier {
    type Err = TelemetryError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let trimmed = s.trim();
        if trimmed.eq_ignore_ascii_case("VRAM") {
            Ok(Self::Vram)
        } else if trimmed.eq_ignore_ascii_case("RAM") {
            Ok(Self::Ram)
        } else if trimmed.eq_ignore_ascii_case("DISK") {
            Ok(Self::Disk)
        } else {
            Err(TelemetryError::UnknownTier(s.to_string()))
        }
    }
}

/// Mutually exclusive outcome of a single tensor lookup request.
/// Enforces `requests == vram_hits + ram_hits + disk_loads + not_found_misses`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LookupOutcome {
    VramHit,
    RamHit { from_prefetch: bool },
    DiskLoad,
    NotFoundMiss,
}

/// Directional eviction route across the three-tier hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EvictionRoute {
    VramToRam,
    VramToDisk,
    RamToDisk,
}

impl EvictionRoute {
    pub fn from_tiers(from: StorageTier, to: StorageTier) -> Result<Self, TelemetryError> {
        match (from, to) {
            (StorageTier::Vram, StorageTier::Ram) => Ok(Self::VramToRam),
            (StorageTier::Vram, StorageTier::Disk) => Ok(Self::VramToDisk),
            (StorageTier::Ram, StorageTier::Disk) => Ok(Self::RamToDisk),
            _ => Err(TelemetryError::InvalidEvictionRoute { from, to }),
        }
    }
}

/// Directional data transfer route across tiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransferRoute {
    DiskToRam,
    DiskToVram,
    RamToVram,
    VramToRam,
    RamToDisk,
}

/// Snapshot of transfer volume and latency for a specific `TransferRoute`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RouteTransferSnapshot {
    pub count: u64,
    pub bytes: u64,
    pub total_duration_ms: f64,
    pub max_duration_ms: f64,
    pub avg_duration_ms: Option<f64>,
    pub p50_duration_ms: Option<f64>,
    pub p95_duration_ms: Option<f64>,
    pub p99_duration_ms: Option<f64>,
}

#[derive(Debug, Default)]
struct RouteAccumulator {
    count: u64,
    bytes: u64,
    total_duration_ms: f64,
    max_duration_ms: f64,
    samples_ms: Vec<f64>,
    write_cursor: usize,
}

impl RouteAccumulator {
    fn record(&mut self, bytes: u64, duration_ms: f64, reservoir_cap: usize) {
        let sanitized_ms = if duration_ms.is_finite() && duration_ms >= 0.0 {
            duration_ms
        } else {
            0.0
        };
        self.count = self.count.saturating_add(1);
        self.bytes = self.bytes.saturating_add(bytes);
        self.total_duration_ms += sanitized_ms;
        if sanitized_ms > self.max_duration_ms {
            self.max_duration_ms = sanitized_ms;
        }
        let cap = reservoir_cap.max(1);
        if self.samples_ms.len() < cap {
            self.samples_ms.push(sanitized_ms);
        } else {
            let idx = self.write_cursor % cap;
            self.samples_ms[idx] = sanitized_ms;
            self.write_cursor = self.write_cursor.wrapping_add(1);
        }
    }

    fn snapshot(&self) -> RouteTransferSnapshot {
        let avg = if self.count > 0 {
            Some(self.total_duration_ms / self.count as f64)
        } else {
            None
        };
        RouteTransferSnapshot {
            count: self.count,
            bytes: self.bytes,
            total_duration_ms: self.total_duration_ms,
            max_duration_ms: self.max_duration_ms,
            avg_duration_ms: avg,
            p50_duration_ms: compute_percentile(&self.samples_ms, 50.0),
            p95_duration_ms: compute_percentile(&self.samples_ms, 95.0),
            p99_duration_ms: compute_percentile(&self.samples_ms, 99.0),
        }
    }
}

fn compute_percentile(samples: &[f64], pct: f64) -> Option<f64> {
    if samples.is_empty() {
        return None;
    }
    let mut sorted: Vec<f64> = samples
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    if sorted.is_empty() {
        return None;
    }
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let clamped = pct.clamp(0.0, 100.0) / 100.0;
    let idx = ((sorted.len() as f64 * clamped).floor() as usize).min(sorted.len() - 1);
    sorted.get(idx).copied()
}

#[derive(Debug)]
struct LatencyAndRouteState {
    start_time: Instant,
    reservoir_capacity: usize,
    all_transfer_samples_ms: Vec<f64>,
    write_cursor: usize,
    total_transfer_time_ms: f64,
    routes: HashMap<TransferRoute, RouteAccumulator>,
}

impl LatencyAndRouteState {
    fn new(reservoir_capacity: usize) -> Self {
        Self {
            start_time: Instant::now(),
            reservoir_capacity: reservoir_capacity.max(1),
            all_transfer_samples_ms: Vec::new(),
            write_cursor: 0,
            total_transfer_time_ms: 0.0,
            routes: HashMap::new(),
        }
    }

    fn record_transfer(&mut self, route: TransferRoute, bytes: u64, duration_ms: f64) {
        let sanitized_ms = if duration_ms.is_finite() && duration_ms >= 0.0 {
            duration_ms
        } else {
            0.0
        };
        self.total_transfer_time_ms += sanitized_ms;
        let cap = self.reservoir_capacity.max(1);
        if self.all_transfer_samples_ms.len() < cap {
            self.all_transfer_samples_ms.push(sanitized_ms);
        } else {
            let idx = self.write_cursor % cap;
            self.all_transfer_samples_ms[idx] = sanitized_ms;
            self.write_cursor = self.write_cursor.wrapping_add(1);
        }
        self.routes
            .entry(route)
            .or_default()
            .record(bytes, sanitized_ms, cap);
    }

    fn reset(&mut self) {
        self.start_time = Instant::now();
        self.all_transfer_samples_ms.clear();
        self.write_cursor = 0;
        self.total_transfer_time_ms = 0.0;
        self.routes.clear();
    }
}

/// Immutable point-in-time snapshot of all telemetry counters and derived rates.
#[derive(Debug, Clone, PartialEq)]
pub struct TelemetrySnapshot {
    pub enabled: bool,
    pub is_partial_capture: bool,
    pub disabled_transitions: u64,
    pub requests: u64,
    pub vram_hits: u64,
    pub ram_hits: u64,
    pub disk_loads: u64,
    pub not_found_misses: u64,
    pub prefetched: u64,
    pub prefetch_hits: u64,
    pub promotions: u64,
    pub evictions: u64,
    pub evictions_vram_to_ram: u64,
    pub evictions_vram_to_disk: u64,
    pub evictions_ram_to_disk: u64,
    pub bytes_read_disk: u64,
    pub bytes_transferred_vram: u64,
    pub bytes_demoted_vram_to_ram: u64,
    pub bytes_evicted_vram_to_disk: u64,
    pub bytes_evicted_ram_to_disk: u64,
    pub total_transfer_time_ms: f64,
    pub elapsed_secs: f64,
    pub cache_hit_rate: Option<f64>,
    pub p50_transfer_latency_ms: Option<f64>,
    pub p95_transfer_latency_ms: Option<f64>,
    pub p99_transfer_latency_ms: Option<f64>,
}

/// Thread-safe, atomic telemetry monitor for three-tier inference and transfer profiling.
pub struct TelemetryMonitor {
    enabled: AtomicBool,
    disabled_transitions: AtomicU64,

    requests: AtomicU64,
    vram_hits: AtomicU64,
    ram_hits: AtomicU64,
    disk_loads: AtomicU64,
    not_found_misses: AtomicU64,

    prefetched: AtomicU64,
    prefetch_hits: AtomicU64,

    promotions: AtomicU64,
    evictions: AtomicU64,
    evictions_vram_to_ram: AtomicU64,
    evictions_vram_to_disk: AtomicU64,
    evictions_ram_to_disk: AtomicU64,

    bytes_read_disk: AtomicU64,
    bytes_transferred_vram: AtomicU64,
    bytes_demoted_vram_to_ram: AtomicU64,
    bytes_evicted_vram_to_disk: AtomicU64,
    bytes_evicted_ram_to_disk: AtomicU64,

    state: Mutex<LatencyAndRouteState>,
}

impl Default for TelemetryMonitor {
    fn default() -> Self {
        Self::new(true)
    }
}

impl TelemetryMonitor {
    pub fn new(enabled: bool) -> Self {
        Self::with_reservoir_capacity(enabled, DEFAULT_LATENCY_RESERVOIR_CAPACITY)
    }

    pub fn with_reservoir_capacity(enabled: bool, reservoir_capacity: usize) -> Self {
        Self {
            enabled: AtomicBool::new(enabled),
            disabled_transitions: AtomicU64::new(if enabled { 0 } else { 1 }),
            requests: AtomicU64::new(0),
            vram_hits: AtomicU64::new(0),
            ram_hits: AtomicU64::new(0),
            disk_loads: AtomicU64::new(0),
            not_found_misses: AtomicU64::new(0),
            prefetched: AtomicU64::new(0),
            prefetch_hits: AtomicU64::new(0),
            promotions: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            evictions_vram_to_ram: AtomicU64::new(0),
            evictions_vram_to_disk: AtomicU64::new(0),
            evictions_ram_to_disk: AtomicU64::new(0),
            bytes_read_disk: AtomicU64::new(0),
            bytes_transferred_vram: AtomicU64::new(0),
            bytes_demoted_vram_to_ram: AtomicU64::new(0),
            bytes_evicted_vram_to_disk: AtomicU64::new(0),
            bytes_evicted_ram_to_disk: AtomicU64::new(0),
            state: Mutex::new(LatencyAndRouteState::new(reservoir_capacity)),
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Lifecycle & State Control
    // ─────────────────────────────────────────────────────────────────────────

    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::SeqCst)
    }

    /// Enables or disables telemetry collection. Tracks `disabled_transitions` so downstream
    /// consumers know if counters represent a partial interval (`is_partial_capture() == true`).
    pub fn set_enabled(&self, enable: bool) {
        let prev = self.enabled.swap(enable, Ordering::SeqCst);
        if prev && !enable {
            self.disabled_transitions.fetch_add(1, Ordering::SeqCst);
        }
    }

    pub fn is_partial_capture(&self) -> bool {
        self.disabled_transitions.load(Ordering::SeqCst) > 0
    }

    pub fn disabled_transitions(&self) -> u64 {
        self.disabled_transitions.load(Ordering::SeqCst)
    }

    /// Resets all counters, route accumulators, latency reservoirs, and restarts `start_time`.
    pub fn reset(&self) {
        let currently_enabled = self.is_enabled();
        self.disabled_transitions.store(
            if currently_enabled { 0 } else { 1 },
            Ordering::SeqCst,
        );
        self.requests.store(0, Ordering::SeqCst);
        self.vram_hits.store(0, Ordering::SeqCst);
        self.ram_hits.store(0, Ordering::SeqCst);
        self.disk_loads.store(0, Ordering::SeqCst);
        self.not_found_misses.store(0, Ordering::SeqCst);
        self.prefetched.store(0, Ordering::SeqCst);
        self.prefetch_hits.store(0, Ordering::SeqCst);
        self.promotions.store(0, Ordering::SeqCst);
        self.evictions.store(0, Ordering::SeqCst);
        self.evictions_vram_to_ram.store(0, Ordering::SeqCst);
        self.evictions_vram_to_disk.store(0, Ordering::SeqCst);
        self.evictions_ram_to_disk.store(0, Ordering::SeqCst);
        self.bytes_read_disk.store(0, Ordering::SeqCst);
        self.bytes_transferred_vram.store(0, Ordering::SeqCst);
        self.bytes_demoted_vram_to_ram.store(0, Ordering::SeqCst);
        self.bytes_evicted_vram_to_disk.store(0, Ordering::SeqCst);
        self.bytes_evicted_ram_to_disk.store(0, Ordering::SeqCst);
        if let Ok(mut guard) = self.state.lock() {
            guard.reset();
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Strongly-Typed Event Recording APIs (Lock-Free Hot Path)
    // ─────────────────────────────────────────────────────────────────────────

    /// Primary strongly-typed lookup recorder with zero string allocation.
    /// Strictly enforces:
    /// - `requests == vram_hits + ram_hits + disk_loads + not_found_misses`
    /// - `prefetch_hits` only increments on `RamHit { from_prefetch: true }` when `prefetch_hits < prefetched`.
    pub fn record_lookup_outcome(&self, outcome: LookupOutcome, size_bytes: u64) {
        if !self.is_enabled() {
            return;
        }
        self.requests.fetch_add(1, Ordering::SeqCst);
        match outcome {
            LookupOutcome::VramHit => {
                self.vram_hits.fetch_add(1, Ordering::SeqCst);
            }
            LookupOutcome::RamHit { from_prefetch } => {
                self.ram_hits.fetch_add(1, Ordering::SeqCst);
                if from_prefetch {
                    // Ensure prefetch_hits never exceeds total prefetched tensors
                    let total_pref = self.prefetched.load(Ordering::SeqCst);
                    let _ = self.prefetch_hits.fetch_update(
                        Ordering::SeqCst,
                        Ordering::SeqCst,
                        |curr| {
                            if curr < total_pref {
                                Some(curr + 1)
                            } else {
                                None
                            }
                        },
                    );
                }
            }
            LookupOutcome::DiskLoad => {
                self.disk_loads.fetch_add(1, Ordering::SeqCst);
                self.bytes_read_disk.fetch_add(size_bytes, Ordering::SeqCst);
            }
            LookupOutcome::NotFoundMiss => {
                self.not_found_misses.fetch_add(1, Ordering::SeqCst);
            }
        }
    }

    /// Validated string-tier lookup recorder.
    /// Rejects unknown tier strings with `Err(TelemetryError::UnknownTier)` and rejects
    /// `is_prefetch_hit == true` on `DISK` lookups with `Err(TelemetryError::InvalidPrefetchHitOutcome)`.
    pub fn try_record_lookup(
        &self,
        tier: &str,
        size_bytes: usize,
        is_prefetch_hit: bool,
    ) -> Result<(), TelemetryError> {
        let parsed_tier = StorageTier::from_str(tier)?;
        let outcome = match parsed_tier {
            StorageTier::Vram => {
                if is_prefetch_hit {
                    return Err(TelemetryError::InvalidPrefetchHitOutcome(
                        LookupOutcome::VramHit,
                    ));
                }
                LookupOutcome::VramHit
            }
            StorageTier::Ram => LookupOutcome::RamHit {
                from_prefetch: is_prefetch_hit,
            },
            StorageTier::Disk => {
                if is_prefetch_hit {
                    return Err(TelemetryError::InvalidPrefetchHitOutcome(
                        LookupOutcome::DiskLoad,
                    ));
                }
                LookupOutcome::DiskLoad
            }
        };
        self.record_lookup_outcome(outcome, size_bytes as u64);
        Ok(())
    }

    /// Legacy string-based lookup helper that delegates to `try_record_lookup`.
    /// Unknown tiers or invalid prefetch flags on Disk are ignored rather than corrupting `disk_loads`.
    pub fn record_lookup(&self, tier: &str, size_bytes: usize, is_prefetch_hit: bool) {
        let _ = self.try_record_lookup(tier, size_bytes, is_prefetch_hit);
    }

    /// Records a directional eviction event with full route and byte volume accounting.
    pub fn record_eviction_route(&self, route: EvictionRoute, size_bytes: u64) {
        if !self.is_enabled() {
            return;
        }
        self.evictions.fetch_add(1, Ordering::SeqCst);
        match route {
            EvictionRoute::VramToRam => {
                self.evictions_vram_to_ram.fetch_add(1, Ordering::SeqCst);
                self.bytes_demoted_vram_to_ram
                    .fetch_add(size_bytes, Ordering::SeqCst);
            }
            EvictionRoute::VramToDisk => {
                self.evictions_vram_to_disk.fetch_add(1, Ordering::SeqCst);
                self.bytes_evicted_vram_to_disk
                    .fetch_add(size_bytes, Ordering::SeqCst);
            }
            EvictionRoute::RamToDisk => {
                self.evictions_ram_to_disk.fetch_add(1, Ordering::SeqCst);
                self.bytes_evicted_ram_to_disk
                    .fetch_add(size_bytes, Ordering::SeqCst);
            }
        }
    }

    /// Validated string-based eviction recorder.
    pub fn try_record_eviction(
        &self,
        from_tier: &str,
        to_tier: &str,
        size_bytes: usize,
    ) -> Result<(), TelemetryError> {
        let from = StorageTier::from_str(from_tier)?;
        let to = StorageTier::from_str(to_tier)?;
        let route = EvictionRoute::from_tiers(from, to)?;
        self.record_eviction_route(route, size_bytes as u64);
        Ok(())
    }

    pub fn record_eviction(&self, from_tier: &str, to_tier: &str, size_bytes: usize) {
        let _ = self.try_record_eviction(from_tier, to_tier, size_bytes);
    }

    /// Records a RAM -> VRAM promotion along with its transfer duration in milliseconds.
    pub fn record_promotion_with_duration(&self, size_bytes: usize, duration_ms: f64) {
        if !self.is_enabled() {
            return;
        }
        let bytes = size_bytes as u64;
        self.promotions.fetch_add(1, Ordering::SeqCst);
        self.bytes_transferred_vram
            .fetch_add(bytes, Ordering::SeqCst);
        if let Ok(mut guard) = self.state.lock() {
            guard.record_transfer(TransferRoute::RamToVram, bytes, duration_ms);
        }
    }

    pub fn record_promotion(&self, size_bytes: usize) {
        self.record_promotion_with_duration(size_bytes, 0.0);
    }

    pub fn record_prefetch(&self, count: usize) {
        if !self.is_enabled() {
            return;
        }
        self.prefetched.fetch_add(count as u64, Ordering::SeqCst);
    }

    /// Records a directional transfer across tiers with byte volume and latency in milliseconds.
    pub fn record_transfer_route(&self, route: TransferRoute, bytes_count: u64, duration_ms: f64) {
        if !self.is_enabled() {
            return;
        }
        match route {
            TransferRoute::DiskToVram | TransferRoute::RamToVram => {
                self.bytes_transferred_vram
                    .fetch_add(bytes_count, Ordering::SeqCst);
            }
            TransferRoute::DiskToRam => {
                // Disk read bytes are already tracked in `record_lookup_outcome(DiskLoad)` or prefetch;
                // route stats track the transfer count, bytes, and latency distribution.
            }
            TransferRoute::VramToRam | TransferRoute::RamToDisk => {}
        }
        if let Ok(mut guard) = self.state.lock() {
            guard.record_transfer(route, bytes_count, duration_ms);
        }
    }

    pub fn record_transfer(&self, duration_ms: f64, bytes_count: usize, to_vram: bool) {
        let route = if to_vram {
            TransferRoute::RamToVram
        } else {
            TransferRoute::VramToRam
        };
        self.record_transfer_route(route, bytes_count as u64, duration_ms);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Counter Accessors & Derived Metrics
    // ─────────────────────────────────────────────────────────────────────────

    pub fn requests(&self) -> u64 {
        self.requests.load(Ordering::SeqCst)
    }

    pub fn vram_hits(&self) -> u64 {
        self.vram_hits.load(Ordering::SeqCst)
    }

    pub fn ram_hits(&self) -> u64 {
        self.ram_hits.load(Ordering::SeqCst)
    }

    pub fn disk_loads(&self) -> u64 {
        self.disk_loads.load(Ordering::SeqCst)
    }

    /// Alias for `disk_loads()` (fast-tier cache misses served from Disk).
    pub fn disk_misses(&self) -> u64 {
        self.disk_loads.load(Ordering::SeqCst)
    }

    pub fn not_found_misses(&self) -> u64 {
        self.not_found_misses.load(Ordering::SeqCst)
    }

    pub fn prefetched(&self) -> u64 {
        self.prefetched.load(Ordering::SeqCst)
    }

    pub fn prefetch_hits(&self) -> u64 {
        self.prefetch_hits.load(Ordering::SeqCst)
    }

    pub fn promotions(&self) -> u64 {
        self.promotions.load(Ordering::SeqCst)
    }

    pub fn evictions(&self) -> u64 {
        self.evictions.load(Ordering::SeqCst)
    }

    pub fn evictions_vram_to_ram(&self) -> u64 {
        self.evictions_vram_to_ram.load(Ordering::SeqCst)
    }

    pub fn evictions_vram_to_disk(&self) -> u64 {
        self.evictions_vram_to_disk.load(Ordering::SeqCst)
    }

    pub fn evictions_ram_to_disk(&self) -> u64 {
        self.evictions_ram_to_disk.load(Ordering::SeqCst)
    }

    pub fn bytes_read_disk(&self) -> u64 {
        self.bytes_read_disk.load(Ordering::SeqCst)
    }

    pub fn bytes_transferred_vram(&self) -> u64 {
        self.bytes_transferred_vram.load(Ordering::SeqCst)
    }

    pub fn bytes_demoted_vram_to_ram(&self) -> u64 {
        self.bytes_demoted_vram_to_ram.load(Ordering::SeqCst)
    }

    pub fn bytes_evicted_vram_to_disk(&self) -> u64 {
        self.bytes_evicted_vram_to_disk.load(Ordering::SeqCst)
    }

    pub fn bytes_evicted_ram_to_disk(&self) -> u64 {
        self.bytes_evicted_ram_to_disk.load(Ordering::SeqCst)
    }

    pub fn total_hits(&self) -> u64 {
        self.vram_hits() + self.ram_hits()
    }

    /// Returns the fast-memory (VRAM + RAM) cache hit rate in `[0.0, 1.0]`,
    /// or `None` if `requests == 0` (never fabricates `1.0` on an empty workload).
    pub fn cache_hit_rate(&self) -> Option<f64> {
        let reqs = self.requests();
        if reqs == 0 {
            None
        } else {
            Some(self.total_hits() as f64 / reqs as f64)
        }
    }

    /// Scalar cache hit rate in `[0.0, 1.0]`. Returns `0.0` (never `1.0`) when `requests == 0`.
    pub fn hit_rate(&self) -> f64 {
        self.cache_hit_rate().unwrap_or(0.0)
    }

    pub fn vram_hit_rate(&self) -> Option<f64> {
        let reqs = self.requests();
        if reqs == 0 {
            None
        } else {
            Some(self.vram_hits() as f64 / reqs as f64)
        }
    }

    pub fn ram_hit_rate(&self) -> Option<f64> {
        let reqs = self.requests();
        if reqs == 0 {
            None
        } else {
            Some(self.ram_hits() as f64 / reqs as f64)
        }
    }

    pub fn disk_load_rate(&self) -> Option<f64> {
        let reqs = self.requests();
        if reqs == 0 {
            None
        } else {
            Some(self.disk_loads() as f64 / reqs as f64)
        }
    }

    /// Fraction of prefetched tensors that resulted in an actual cache hit (`prefetch_hits / prefetched`).
    pub fn prefetch_hit_rate(&self) -> Option<f64> {
        let pref = self.prefetched();
        if pref == 0 {
            None
        } else {
            Some(self.prefetch_hits() as f64 / pref as f64)
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Elapsed Time, Throughput, Route Profiles & Latency Percentiles
    // ─────────────────────────────────────────────────────────────────────────

    pub fn elapsed_secs(&self) -> f64 {
        self.state
            .lock()
            .map(|g| g.start_time.elapsed().as_secs_f64())
            .unwrap_or(0.0)
    }

    pub fn requests_per_sec(&self) -> f64 {
        let secs = self.elapsed_secs();
        if secs <= 1e-9 {
            0.0
        } else {
            self.requests() as f64 / secs
        }
    }

    pub fn disk_mib_per_sec(&self) -> f64 {
        let secs = self.elapsed_secs();
        if secs <= 1e-9 {
            0.0
        } else {
            (self.bytes_read_disk() as f64 / (1024.0 * 1024.0)) / secs
        }
    }

    pub fn vram_transfer_mib_per_sec(&self) -> f64 {
        let secs = self.elapsed_secs();
        if secs <= 1e-9 {
            0.0
        } else {
            (self.bytes_transferred_vram() as f64 / (1024.0 * 1024.0)) / secs
        }
    }

    pub fn total_transfer_time_ms(&self) -> f64 {
        self.state
            .lock()
            .map(|g| g.total_transfer_time_ms)
            .unwrap_or(0.0)
    }

    pub fn average_transfer_latency_ms(&self) -> Option<f64> {
        let guard = self.state.lock().ok()?;
        let total_count: u64 = guard.routes.values().map(|r| r.count).sum();
        if total_count == 0 {
            None
        } else {
            Some(guard.total_transfer_time_ms / total_count as f64)
        }
    }

    pub fn transfer_latency_percentile_ms(&self, percentile: f64) -> Option<f64> {
        let guard = self.state.lock().ok()?;
        compute_percentile(&guard.all_transfer_samples_ms, percentile)
    }

    pub fn p50_transfer_latency_ms(&self) -> Option<f64> {
        self.transfer_latency_percentile_ms(50.0)
    }

    pub fn p95_transfer_latency_ms(&self) -> Option<f64> {
        self.transfer_latency_percentile_ms(95.0)
    }

    pub fn p99_transfer_latency_ms(&self) -> Option<f64> {
        self.transfer_latency_percentile_ms(99.0)
    }

    pub fn route_stats(&self, route: TransferRoute) -> RouteTransferSnapshot {
        self.state
            .lock()
            .ok()
            .and_then(|g| g.routes.get(&route).map(|r| r.snapshot()))
            .unwrap_or_default()
    }

    /// Verifies mathematical accounting invariants across all counters:
    /// 1. `requests == vram_hits + ram_hits + disk_loads + not_found_misses`
    /// 2. `prefetch_hits <= ram_hits + vram_hits` and `prefetch_hits <= prefetched`
    /// 3. `evictions == evictions_vram_to_ram + evictions_vram_to_disk + evictions_ram_to_disk`
    pub fn verify_invariants(&self) -> Result<(), TelemetryError> {
        let reqs = self.requests();
        let sum_outcomes =
            self.vram_hits() + self.ram_hits() + self.disk_loads() + self.not_found_misses();
        if reqs != sum_outcomes {
            return Err(TelemetryError::InvariantViolation(format!(
                "requests ({reqs}) != vram_hits + ram_hits + disk_loads + not_found_misses ({sum_outcomes})"
            )));
        }

        let p_hits = self.prefetch_hits();
        let pref = self.prefetched();
        if p_hits > pref {
            return Err(TelemetryError::InvariantViolation(format!(
                "prefetch_hits ({p_hits}) > prefetched ({pref})"
            )));
        }
        if p_hits > self.total_hits() {
            return Err(TelemetryError::InvariantViolation(format!(
                "prefetch_hits ({p_hits}) > total_hits ({})",
                self.total_hits()
            )));
        }

        let ev = self.evictions();
        let sum_ev = self.evictions_vram_to_ram()
            + self.evictions_vram_to_disk()
            + self.evictions_ram_to_disk();
        if ev != sum_ev {
            return Err(TelemetryError::InvariantViolation(format!(
                "evictions ({ev}) != directional eviction sum ({sum_ev})"
            )));
        }

        Ok(())
    }

    pub fn snapshot(&self) -> TelemetrySnapshot {
        TelemetrySnapshot {
            enabled: self.is_enabled(),
            is_partial_capture: self.is_partial_capture(),
            disabled_transitions: self.disabled_transitions(),
            requests: self.requests(),
            vram_hits: self.vram_hits(),
            ram_hits: self.ram_hits(),
            disk_loads: self.disk_loads(),
            not_found_misses: self.not_found_misses(),
            prefetched: self.prefetched(),
            prefetch_hits: self.prefetch_hits(),
            promotions: self.promotions(),
            evictions: self.evictions(),
            evictions_vram_to_ram: self.evictions_vram_to_ram(),
            evictions_vram_to_disk: self.evictions_vram_to_disk(),
            evictions_ram_to_disk: self.evictions_ram_to_disk(),
            bytes_read_disk: self.bytes_read_disk(),
            bytes_transferred_vram: self.bytes_transferred_vram(),
            bytes_demoted_vram_to_ram: self.bytes_demoted_vram_to_ram(),
            bytes_evicted_vram_to_disk: self.bytes_evicted_vram_to_disk(),
            bytes_evicted_ram_to_disk: self.bytes_evicted_ram_to_disk(),
            total_transfer_time_ms: self.total_transfer_time_ms(),
            elapsed_secs: self.elapsed_secs(),
            cache_hit_rate: self.cache_hit_rate(),
            p50_transfer_latency_ms: self.p50_transfer_latency_ms(),
            p95_transfer_latency_ms: self.p95_transfer_latency_ms(),
            p99_transfer_latency_ms: self.p99_transfer_latency_ms(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_unknown_tier_rejected_and_never_counted_as_disk_miss() {
        let tm = TelemetryMonitor::new(true);

        // Empty workload hit rate MUST be None / 0.0 (never 1.0)
        assert_eq!(tm.cache_hit_rate(), None);
        assert_eq!(tm.hit_rate(), 0.0);

        for bad_tier in ["GPU", "CPU_CACHE", "INVALID", ""] {
            let res = tm.try_record_lookup(bad_tier, 1024, false);
            assert!(
                matches!(res, Err(TelemetryError::UnknownTier(_))),
                "Invalid tier '{bad_tier}' must return UnknownTier error"
            );
        }

        // Counters must remain 0 after invalid tier strings
        assert_eq!(tm.requests(), 0);
        assert_eq!(tm.disk_loads(), 0);
        assert_eq!(tm.bytes_read_disk(), 0);
        assert!(tm.verify_invariants().is_ok());
    }

    #[test]
    fn test_lookup_outcomes_prefetch_semantics_and_invariants() {
        let tm = TelemetryMonitor::new(true);

        // Disk load cannot be marked as a prefetch hit
        let bad_pref = tm.try_record_lookup("DISK", 256, true);
        assert!(matches!(
            bad_pref,
            Err(TelemetryError::InvalidPrefetchHitOutcome(
                LookupOutcome::DiskLoad
            ))
        ));
        assert_eq!(tm.requests(), 0);

        // Record 1 prefetch, 1 RAM prefetch hit, 1 VRAM hit, 1 DiskLoad, 1 NotFoundMiss
        tm.record_prefetch(1);
        tm.record_lookup_outcome(LookupOutcome::RamHit { from_prefetch: true }, 64);
        tm.record_lookup_outcome(LookupOutcome::VramHit, 64);
        tm.record_lookup_outcome(LookupOutcome::DiskLoad, 128);
        tm.record_lookup_outcome(LookupOutcome::NotFoundMiss, 0);

        assert_eq!(tm.requests(), 4);
        assert_eq!(tm.vram_hits(), 1);
        assert_eq!(tm.ram_hits(), 1);
        assert_eq!(tm.disk_loads(), 1);
        assert_eq!(tm.not_found_misses(), 1);
        assert_eq!(tm.prefetch_hits(), 1);
        assert_eq!(tm.cache_hit_rate(), Some(0.5));
        assert_eq!(tm.prefetch_hit_rate(), Some(1.0));
        assert!(tm.verify_invariants().is_ok());
    }

    #[test]
    fn test_directional_evictions_promotions_routes_and_percentiles() {
        let tm = TelemetryMonitor::new(true);

        tm.record_eviction_route(EvictionRoute::VramToRam, 100);
        tm.record_eviction_route(EvictionRoute::VramToDisk, 200);
        tm.record_eviction_route(EvictionRoute::RamToDisk, 300);

        assert_eq!(tm.evictions(), 3);
        assert_eq!(tm.evictions_vram_to_ram(), 1);
        assert_eq!(tm.evictions_vram_to_disk(), 1);
        assert_eq!(tm.evictions_ram_to_disk(), 1);
        assert_eq!(tm.bytes_demoted_vram_to_ram(), 100);
        assert_eq!(tm.bytes_evicted_vram_to_disk(), 200);
        assert_eq!(tm.bytes_evicted_ram_to_disk(), 300);

        // Record 99 fast transfers (1.0 ms) and 1 slow tail transfer (5000.0 ms)
        for _ in 0..99 {
            tm.record_transfer_route(TransferRoute::DiskToRam, 1024, 1.0);
        }
        tm.record_transfer_route(TransferRoute::RamToVram, 2048, 5000.0);

        assert_eq!(tm.p50_transfer_latency_ms(), Some(1.0));
        assert_eq!(tm.p95_transfer_latency_ms(), Some(1.0));
        assert_eq!(tm.p99_transfer_latency_ms(), Some(5000.0));

        let disk_route = tm.route_stats(TransferRoute::DiskToRam);
        assert_eq!(disk_route.count, 99);
        assert_eq!(disk_route.bytes, 99 * 1024);

        let vram_route = tm.route_stats(TransferRoute::RamToVram);
        assert_eq!(vram_route.count, 1);
        assert_eq!(vram_route.bytes, 2048);
        assert_eq!(vram_route.max_duration_ms, 5000.0);
        assert!(tm.verify_invariants().is_ok());
    }

    #[test]
    fn test_concurrent_multithreaded_telemetry_and_enable_reset_lifecycle() {
        let tm = Arc::new(TelemetryMonitor::new(true));
        let mut handles = Vec::new();

        for _ in 0..4 {
            let tm_clone = Arc::clone(&tm);
            handles.push(std::thread::spawn(move || {
                for _ in 0..250 {
                    tm_clone.record_lookup_outcome(LookupOutcome::VramHit, 16);
                    tm_clone.record_eviction_route(EvictionRoute::RamToDisk, 16);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(tm.requests(), 1000);
        assert_eq!(tm.vram_hits(), 1000);
        assert_eq!(tm.evictions(), 1000);
        assert_eq!(tm.evictions_ram_to_disk(), 1000);
        assert!(!tm.is_partial_capture());
        assert!(tm.verify_invariants().is_ok());

        // Disable -> verify partial capture flag -> re-enable -> reset
        tm.set_enabled(false);
        assert!(tm.is_partial_capture());
        tm.record_lookup_outcome(LookupOutcome::VramHit, 16);
        assert_eq!(tm.requests(), 1000);

        tm.set_enabled(true);
        assert!(tm.is_partial_capture());

        tm.reset();
        assert_eq!(tm.requests(), 0);
        assert!(!tm.is_partial_capture());
        assert_eq!(tm.cache_hit_rate(), None);
        assert!(tm.verify_invariants().is_ok());
    }
}
