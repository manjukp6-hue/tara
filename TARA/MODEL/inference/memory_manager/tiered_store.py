"""
TARA/MODEL/inference/memory_manager/tiered_store.py

Three-tier storage hierarchy (VRAM -> RAM -> Disk) with lease mechanics,
admission hysteresis, and eviction.
Adapted from Colibrì tier management & expert store.

Attribution:
Adapted from Colibrì (https://github.com/JustVugg/colibri, c/tier.h, c/expert_store.h)
Copyright (c) 2025 JustVugg / Colibrì contributors
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
"""

import time
import threading
from typing import Dict, List, Optional, Any, Set, Tuple

from ..cache.cache_policy import CachePolicy, LFRUCachePolicy, tier_should_promote
from ..tensor_stream.stream_reader import SafeTensorsStreamReader, TensorView
from ..backends.backend_registry import DeviceBackend, CPUBackend, CUDABackend, BackendRegistry
from ..telemetry.monitor import TelemetryMonitor
from ..resource_planner.planner import ResourcePlanner


class TieredTensorStore:
    """
    Manages three storage tiers for weight tensors:
      - VRAM Tier: Fastest, on-device GPU memory
      - RAM Tier: Fast, host system memory
      - Disk Tier: Large, cold storage backed by SafeTensors files
    """
    def __init__(
        self,
        stream_reader: SafeTensorsStreamReader,
        cache_policy: Optional[CachePolicy] = None,
        telemetry: Optional[TelemetryMonitor] = None,
        backend: Optional[DeviceBackend] = None,
        vram_capacity_bytes: int = 0,
        ram_capacity_bytes: int = 0,
        admission_margin_pct: int = 25,
        admission_margin_fixed: int = 4
    ):
        self.stream_reader = stream_reader
        self.policy = cache_policy or LFRUCachePolicy()
        self.telemetry = telemetry or TelemetryMonitor()
        self.backend = backend or BackendRegistry.select_best_backend()

        self.vram_capacity_bytes = vram_capacity_bytes
        self.ram_capacity_bytes = ram_capacity_bytes
        self.admission_margin_pct = admission_margin_pct
        self.admission_margin_fixed = admission_margin_fixed

        # Storage pools
        self.vram_pool: Dict[str, TensorView] = {}
        self.ram_pool: Dict[str, TensorView] = {}
        self.disk_keys: Set[str] = set(stream_reader.list_tensors())

        # Active leases (tensors currently executing in a forward pass)
        self.active_leases: Dict[str, int] = {}
        self._lock = threading.Lock()

    @property
    def vram_resident_bytes(self) -> int:
        return sum(tv.data_bytes for tv in self.vram_pool.values())

    @property
    def ram_resident_bytes(self) -> int:
        return sum(tv.data_bytes for tv in self.ram_pool.values())

    def lookup(self, key: str) -> TensorView:
        """
        Acquire a lease on a tensor view.
        Moves data from Disk -> RAM -> VRAM as budget and heat dictate.
        """
        with self._lock:
            self.policy.record_access(key)
            t_start = time.perf_counter()

            # 1. Check VRAM Tier (Hit)
            if key in self.vram_pool:
                tv = self.vram_pool[key]
                self.telemetry.record_lookup("VRAM", tv.data_bytes)
                self._acquire_lease(key)
                return tv

            # 2. Check RAM Tier (Hit)
            if key in self.ram_pool:
                tv = self.ram_pool[key]
                self.telemetry.record_lookup("RAM", tv.data_bytes)
                
                # Check if eligible for promotion to VRAM
                if self.backend.name() != "CPU" and self.vram_capacity_bytes > 0:
                    self._try_promote_to_vram(key)
                    if key in self.vram_pool:
                        tv = self.vram_pool[key]

                self._acquire_lease(key)
                return tv

            # 3. Disk Miss - Load from SafeTensors
            if key not in self.disk_keys:
                raise KeyError(f"Tensor {key} not found in model storage")

            # Load tensor from disk
            tv = self.stream_reader.load_tensor(key)
            duration_ms = (time.perf_counter() - t_start) * 1000.0
            self.telemetry.record_lookup("DISK", tv.data_bytes)
            self.telemetry.record_transfer(duration_ms, tv.data_bytes, to_vram=False)

            # Admit into RAM tier (evicting if necessary)
            self._admit_to_ram(tv)

            # Check if immediately promoted to VRAM
            if self.backend.name() != "CPU" and self.vram_capacity_bytes > 0:
                self._try_promote_to_vram(key)
                if key in self.vram_pool:
                    tv = self.vram_pool[key]

            self._acquire_lease(key)
            return tv

    def release(self, key: str) -> None:
        """Release lease on tensor view."""
        with self._lock:
            if key in self.active_leases:
                self.active_leases[key] -= 1
                if self.active_leases[key] <= 0:
                    del self.active_leases[key]

    def _acquire_lease(self, key: str) -> None:
        self.active_leases[key] = self.active_leases.get(key, 0) + 1

    def _try_promote_to_vram(self, key: str) -> None:
        """Promotes a tensor from RAM to VRAM if capacity allows or admission hysteresis is met."""
        if key not in self.ram_pool:
            return
        ram_tv = self.ram_pool[key]
        needed_bytes = ram_tv.data_bytes

        # Evict unleased tensors from VRAM if over capacity
        while (self.vram_resident_bytes + needed_bytes > self.vram_capacity_bytes) and self.vram_pool:
            unleased_vram = [k for k in self.vram_pool if k not in self.active_leases]
            if not unleased_vram:
                break  # Cannot evict active tensors
            evict_cand = self.policy.pick_eviction(unleased_vram)
            if not evict_cand:
                break

            # Check Colibrì admission hysteresis
            if not self.policy.should_admit(key, evict_cand):
                return  # Promotion denied by admission hysteresis

            self._demote_from_vram(evict_cand)

        if self.vram_resident_bytes + needed_bytes <= self.vram_capacity_bytes:
            # Transfer tensor to device
            t0 = time.perf_counter()
            dev_data = self.backend.transfer_to_device(ram_tv.data)
            duration_ms = (time.perf_counter() - t0) * 1000.0
            self.telemetry.record_transfer(duration_ms, needed_bytes, to_vram=True)

            vram_tv = TensorView(
                name=ram_tv.name,
                shape=ram_tv.shape,
                dtype=ram_tv.dtype,
                data=dev_data,
                data_bytes=ram_tv.data_bytes,
                tier="VRAM"
            )
            self.vram_pool[key] = vram_tv
            # Remove from RAM pool to preserve RAM tier budget
            del self.ram_pool[key]

    def _demote_from_vram(self, key: str) -> None:
        """Demotes tensor from VRAM back to RAM."""
        if key not in self.vram_pool:
            return
        vram_tv = self.vram_pool.pop(key)
        host_data = self.backend.transfer_to_host(vram_tv.data)
        ram_tv = TensorView(
            name=vram_tv.name,
            shape=vram_tv.shape,
            dtype=vram_tv.dtype,
            data=host_data,
            data_bytes=vram_tv.data_bytes,
            tier="RAM"
        )
        self.telemetry.record_eviction("VRAM", "RAM", ram_tv.data_bytes)
        self._admit_to_ram(ram_tv)

    def _admit_to_ram(self, tv: TensorView) -> None:
        """Admits tensor to RAM, evicting cold tensors to Disk if RAM capacity is reached."""
        needed_bytes = tv.data_bytes
        
        # Evict unleased tensors from RAM to Disk if needed
        if self.ram_capacity_bytes > 0:
            while (self.ram_resident_bytes + needed_bytes > self.ram_capacity_bytes) and self.ram_pool:
                unleased_ram = [k for k in self.ram_pool if k not in self.active_leases]
                if not unleased_ram:
                    break
                evict_cand = self.policy.pick_eviction(unleased_ram)
                if not evict_cand:
                    break
                # Evict to disk
                evicted_tv = self.ram_pool.pop(evict_cand)
                self.telemetry.record_eviction("RAM", "DISK", evicted_tv.data_bytes)

        tv.tier = "RAM"
        self.ram_pool[tv.name] = tv

    def prefetch(self, keys: List[str]) -> int:
        """
        Advisory prefetch of keys into RAM.
        Adapted from Colibrì c/expert_store.h (prefetch).
        """
        count = 0
        for k in keys:
            with self._lock:
                if k in self.vram_pool or k in self.ram_pool:
                    continue
            # Read asynchronously or synchronously
            if self.stream_reader.enable_async:
                self.stream_reader.submit_async_read(k, callback=self._on_async_prefetch_loaded)
                count += 1
            else:
                try:
                    tv = self.stream_reader.load_tensor(k)
                    with self._lock:
                        self._admit_to_ram(tv)
                    count += 1
                except Exception:
                    pass
        self.telemetry.record_prefetch(count)
        return count

    def _on_async_prefetch_loaded(self, tv: TensorView) -> None:
        with self._lock:
            self._admit_to_ram(tv)

    def get_tier_counts(self) -> Tuple[int, int, int, int, int, int]:
        """Returns (vram_count, ram_count, disk_count, vram_bytes, ram_bytes, disk_bytes)."""
        vram_c = len(self.vram_pool)
        ram_c = len(self.ram_pool)
        disk_c = len(self.disk_keys) - vram_c - ram_c
        disk_c = max(0, disk_c)
        vram_b = self.vram_resident_bytes
        ram_b = self.ram_resident_bytes
        # Estimated cold disk bytes
        disk_b = max(0, self.stream_reader.file_size - vram_b - ram_b)
        return (vram_c, ram_c, disk_c, vram_b, ram_b, disk_b)
