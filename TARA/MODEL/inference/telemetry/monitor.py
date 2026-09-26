"""
TARA/MODEL/inference/telemetry/monitor.py

Inference telemetry, tier allocation monitoring, and hit/miss profiling.
Adapted from Colibrì telemetry protocol.

Attribution:
Adapted from Colibrì (https://github.com/JustVugg/colibri, c/telemetry.h, c/expert_store.h)
Copyright (c) 2025 JustVugg / Colibrì contributors
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
"""

import time
from typing import Dict, List, Optional, Any


class TelemetryMonitor:
    """
    Tracks inference cache statistics, tier byte distributions,
    and hit/miss rates.
    Adapted from Colibrì c/expert_store.h (ColiExpertStoreStats) and c/telemetry.h.
    """
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.requests: int = 0
        self.vram_hits: int = 0
        self.ram_hits: int = 0
        self.disk_misses: int = 0
        self.prefetched: int = 0
        self.prefetch_hits: int = 0
        self.evictions: int = 0
        self.bytes_read_disk: int = 0
        self.bytes_transferred_vram: int = 0
        self.total_transfer_time_ms: float = 0.0
        self.start_time: float = time.time()

    def record_lookup(self, tier: str, size_bytes: int, is_prefetch_hit: bool = False) -> None:
        if not self.enabled:
            return
        self.requests += 1
        tier_upper = tier.upper()
        if tier_upper == "VRAM":
            self.vram_hits += 1
        elif tier_upper == "RAM":
            self.ram_hits += 1
        else:
            self.disk_misses += 1
            self.bytes_read_disk += size_bytes

        if is_prefetch_hit:
            self.prefetch_hits += 1

    def record_eviction(self, from_tier: str, to_tier: str, size_bytes: int) -> None:
        if not self.enabled:
            return
        self.evictions += 1

    def record_prefetch(self, count: int = 1) -> None:
        if not self.enabled:
            return
        self.prefetched += count

    def record_transfer(self, duration_ms: float, bytes_count: int, to_vram: bool = False) -> None:
        if not self.enabled:
            return
        self.total_transfer_time_ms += duration_ms
        if to_vram:
            self.bytes_transferred_vram += bytes_count

    @property
    def total_hits(self) -> int:
        return self.vram_hits + self.ram_hits

    @property
    def hit_rate(self) -> float:
        if self.requests == 0:
            return 1.0
        return self.total_hits / self.requests

    def get_stats(self) -> Dict[str, Any]:
        return {
            "requests": self.requests,
            "vram_hits": self.vram_hits,
            "ram_hits": self.ram_hits,
            "total_hits": self.total_hits,
            "disk_misses": self.disk_misses,
            "hit_rate": round(self.hit_rate, 4),
            "prefetched": self.prefetched,
            "prefetch_hits": self.prefetch_hits,
            "evictions": self.evictions,
            "bytes_read_disk": self.bytes_read_disk,
            "bytes_transferred_vram": self.bytes_transferred_vram,
            "total_transfer_time_ms": round(self.total_transfer_time_ms, 2),
            "uptime_seconds": round(time.time() - self.start_time, 2)
        }

    def format_tiers_summary(
        self,
        vram_count: int,
        ram_count: int,
        disk_count: int,
        vram_bytes: int,
        ram_bytes: int,
        disk_bytes: int
    ) -> str:
        """
        Emits TIERS line matching Colibrì telemetry protocol:
        TIERS <vram_slots> <ram_slots> <disk_slots> <vram_gb> <ram_gb>
        """
        vram_gb = vram_bytes / 1e9
        ram_gb = ram_bytes / 1e9
        disk_gb = disk_bytes / 1e9
        return (
            f"TIERS: VRAM={vram_count} ({vram_gb:.3f} GB) | "
            f"RAM={ram_count} ({ram_gb:.3f} GB) | "
            f"DISK={disk_count} ({disk_gb:.3f} GB) | "
            f"HitRate={self.hit_rate * 100:.1f}%"
        )
