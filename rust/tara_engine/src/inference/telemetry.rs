//! Inference telemetry, tier allocation monitoring, and hit/miss profiling.

use std::time::Instant;

pub struct TelemetryMonitor {
    pub enabled: bool,
    pub requests: usize,
    pub vram_hits: usize,
    pub ram_hits: usize,
    pub disk_misses: usize,
    pub prefetched: usize,
    pub prefetch_hits: usize,
    pub evictions: usize,
    pub promotions: usize,
    pub bytes_read_disk: usize,
    pub bytes_transferred_vram: usize,
    pub total_transfer_time_ms: f64,
    pub start_time: Instant,
}

impl Default for TelemetryMonitor {
    fn default() -> Self {
        Self::new(true)
    }
}

impl TelemetryMonitor {
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            requests: 0,
            vram_hits: 0,
            ram_hits: 0,
            disk_misses: 0,
            prefetched: 0,
            prefetch_hits: 0,
            evictions: 0,
            promotions: 0,
            bytes_read_disk: 0,
            bytes_transferred_vram: 0,
            total_transfer_time_ms: 0.0,
            start_time: Instant::now(),
        }
    }

    pub fn record_lookup(&mut self, tier: &str, size_bytes: usize, is_prefetch_hit: bool) {
        if !self.enabled {
            return;
        }
        self.requests += 1;
        match tier.to_uppercase().as_str() {
            "VRAM" => self.vram_hits += 1,
            "RAM" => self.ram_hits += 1,
            _ => {
                self.disk_misses += 1;
                self.bytes_read_disk += size_bytes;
            }
        }
        if is_prefetch_hit {
            self.prefetch_hits += 1;
        }
    }

    pub fn record_eviction(&mut self, _from_tier: &str, _to_tier: &str, _size_bytes: usize) {
        if !self.enabled {
            return;
        }
        self.evictions += 1;
    }

    pub fn record_promotion(&mut self, size_bytes: usize) {
        if !self.enabled {
            return;
        }
        self.promotions += 1;
        self.bytes_transferred_vram += size_bytes;
    }

    pub fn record_prefetch(&mut self, count: usize) {
        if !self.enabled {
            return;
        }
        self.prefetched += count;
    }

    pub fn record_transfer(&mut self, duration_ms: f64, bytes_count: usize, to_vram: bool) {
        if !self.enabled {
            return;
        }
        self.total_transfer_time_ms += duration_ms;
        if to_vram {
            self.bytes_transferred_vram += bytes_count;
        }
    }

    pub fn total_hits(&self) -> usize {
        self.vram_hits + self.ram_hits
    }

    pub fn hit_rate(&self) -> f64 {
        if self.requests == 0 {
            1.0
        } else {
            self.total_hits() as f64 / self.requests as f64
        }
    }
}
