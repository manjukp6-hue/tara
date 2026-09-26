"""
TARA/MODEL/inference/resource_planner/planner.py

System hardware probing and tier budget planning.
Adapted from Colibrì resource planning algorithms.

Attribution:
Adapted from Colibrì (https://github.com/JustVugg/colibri, c/resource_plan.py)
Copyright (c) 2025 JustVugg / Colibrì contributors
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
"""

import os
import sys
import shutil
import re
import subprocess
from typing import Dict, List, Optional, Any, Tuple

GB = 1_000_000_000
MB = 1_000_000


def get_available_ram_bytes() -> int:
    """
    Get system available RAM in bytes.
    Adapted from Colibrì c/resource_plan.py (memory_available).
    """
    # Windows native
    if sys.platform == "win32":
        try:
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX(dwLength=ctypes.sizeof(MEMORYSTATUSEX))
            kernel32 = ctypes.windll.kernel32
            kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.c_void_p]
            kernel32.GlobalMemoryStatusEx.restype = ctypes.c_int
            if kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)) and stat.ullAvailPhys:
                return stat.ullAvailPhys
        except Exception:
            pass

    # Linux /proc/meminfo
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    return int(parts[1]) * 1024
    except Exception:
        pass

    # Fallback estimate (4GB default safe assumption if probing fails)
    return 4 * GB


def get_available_disk_bytes(path: str = ".") -> int:
    """Get available disk storage in bytes."""
    try:
        usage = shutil.disk_usage(path)
        return usage.free
    except Exception:
        return 100 * GB


def discover_gpus() -> List[Dict[str, Any]]:
    """
    Discover available GPUs and free VRAM.
    Adapted from Colibrì c/resource_plan.py (discover_gpus).
    """
    devices = []
    # Try nvidia-smi
    try:
        cmd = ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader,nounits"]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=3)
        if res.returncode == 0:
            for line in res.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    devices.append({
                        "index": int(parts[0]),
                        "name": parts[1],
                        "total_bytes": int(parts[2]) * MB,
                        "free_bytes": int(parts[3]) * MB,
                        "type": "cuda"
                    })
    except Exception:
        pass

    # Try PyTorch torch.cuda if available in environment
    if not devices:
        try:
            import torch
            if torch.cuda.is_available():
                for idx in range(torch.cuda.device_count()):
                    name = torch.cuda.get_device_name(idx)
                    total = torch.cuda.get_device_properties(idx).total_memory
                    free = total - torch.cuda.memory_allocated(idx)
                    devices.append({
                        "index": idx,
                        "name": name,
                        "total_bytes": total,
                        "free_bytes": free,
                        "type": "cuda"
                    })
        except Exception:
            pass

    return devices


class ResourcePlanner:
    """
    Plans memory tier allocations for TARA based on hardware limits and config.
    """
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        self.config = config_dict or {}

    def plan_tiers(
        self,
        model_size_bytes: int,
        context_len: int = 2048,
        forced_vram_gb: Optional[float] = None,
        forced_ram_gb: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculates optimal byte budgets for VRAM, RAM, and Disk tiers.
        """
        avail_ram = get_available_ram_bytes()
        avail_disk = get_available_disk_bytes()
        gpus = discover_gpus()

        total_vram_free = sum(g["free_bytes"] for g in gpus)
        
        # Determine VRAM Budget
        cfg_vram = self.config.get("vram_budget_gb", "auto")
        if forced_vram_gb is not None:
            vram_budget = int(forced_vram_gb * GB)
        elif isinstance(cfg_vram, (int, float)):
            vram_budget = int(cfg_vram * GB)
        elif total_vram_free > 0:
            # Leave 500MB safety margin for driver
            vram_budget = max(0, total_vram_free - int(0.5 * GB))
        else:
            vram_budget = 0

        # Determine RAM Budget
        cfg_ram = self.config.get("ram_budget_gb", "auto")
        if forced_ram_gb is not None:
            ram_budget = int(forced_ram_gb * GB)
        elif isinstance(cfg_ram, (int, float)):
            ram_budget = int(cfg_ram * GB)
        else:
            # Reserve 20% of free system RAM for OS / other apps
            ram_budget = max(int(0.5 * GB), int(avail_ram * 0.80))

        # Disk cache
        cfg_disk = self.config.get("disk_cache_gb", "auto")
        if isinstance(cfg_disk, (int, float)):
            disk_budget = int(cfg_disk * GB)
        else:
            disk_budget = max(int(10 * GB), int(avail_disk * 0.5))

        return {
            "vram_budget_bytes": vram_budget,
            "ram_budget_bytes": ram_budget,
            "disk_budget_bytes": disk_budget,
            "available_ram_bytes": avail_ram,
            "available_disk_bytes": avail_disk,
            "gpu_count": len(gpus),
            "gpus": gpus,
            "model_size_bytes": model_size_bytes,
            "can_fit_in_vram": vram_budget >= model_size_bytes,
            "can_fit_in_ram": (vram_budget + ram_budget) >= model_size_bytes,
            "requires_disk_streaming": (vram_budget + ram_budget) < model_size_bytes
        }
