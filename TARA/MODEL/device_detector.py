"""
TARA/MODEL/device_detector.py
================================================================================
Automatic Device Capability & Hardware Environment Detection for TARA
================================================================================

Detects at runtime with zero manual configuration:
- OS / Platform
- CPU architecture and core counts
- Total and available system RAM
- GPU availability, vendor, architecture, and free VRAM
- Available disk storage
- Environment classification (MOBILE, DESKTOP, SERVER)
"""

import os
import sys
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

GB = 1024 * 1024 * 1024
MB = 1024 * 1024


@dataclass
class GpuInfo:
    available: bool = False
    vendor: str = "NONE"          # "NVIDIA", "APPLE", "AMD", "INTEL", "NONE"
    device_name: str = "None"
    total_vram_bytes: int = 0
    free_vram_bytes: int = 0
    type: str = "NONE"            # "CUDA", "MPS", "ROCM", "DIRECTML", "NONE"
    device_index: int = 0

    @property
    def total_vram_gb(self) -> float:
        return round(self.total_vram_bytes / GB, 2)

    @property
    def free_vram_gb(self) -> float:
        return round(self.free_vram_bytes / GB, 2)


@dataclass
class DeviceProfile:
    os_name: str
    os_version: str
    cpu_arch: str
    cpu_cores_logical: int
    cpu_cores_physical: int
    total_ram_bytes: int
    available_ram_bytes: int
    available_storage_bytes: int
    gpu: GpuInfo
    environment_type: str        # "MOBILE", "DESKTOP", "SERVER"
    is_mobile: bool = False
    is_server: bool = False
    is_desktop: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_ram_gb(self) -> float:
        return round(self.total_ram_bytes / GB, 2)

    @property
    def available_ram_gb(self) -> float:
        return round(self.available_ram_bytes / GB, 2)

    @property
    def available_storage_gb(self) -> float:
        return round(self.available_storage_bytes / GB, 2)

    def summary(self) -> str:
        gpu_str = f"{self.gpu.vendor} {self.gpu.device_name} ({self.gpu.free_vram_gb:.1f} GB free)" if self.gpu.available else "None (CPU only)"
        return (
            f"DeviceProfile[{self.environment_type} | {self.os_name} {self.cpu_arch} | "
            f"CPU: {self.cpu_cores_logical} cores | RAM: {self.available_ram_gb:.1f}/{self.total_ram_gb:.1f} GB | "
            f"GPU: {gpu_str} | Storage: {self.available_storage_gb:.1f} GB free]"
        )

    @classmethod
    def mock(
        cls,
        total_ram_gb: float = 16.0,
        available_ram_gb: float = 12.0,
        gpu_vram_gb: float = 0.0,
        gpu_vendor: str = "NONE",
        gpu_name: str = "None",
        is_mobile: bool = False,
        is_server: bool = False,
        cpu_cores: int = 8,
        storage_gb: float = 100.0,
        os_name: str = "Windows",
        cpu_arch: str = "AMD64"
    ) -> "DeviceProfile":
        """Creates a mock hardware profile for testing all capability regimes."""
        env_type = "MOBILE" if is_mobile else ("SERVER" if is_server else "DESKTOP")
        gpu_avail = gpu_vram_gb > 0.0 and gpu_vendor != "NONE"
        gpu_info = GpuInfo(
            available=gpu_avail,
            vendor=gpu_vendor if gpu_avail else "NONE",
            device_name=gpu_name if gpu_avail else "None",
            total_vram_bytes=int(gpu_vram_gb * GB),
            free_vram_bytes=int(gpu_vram_gb * GB),
            type="CUDA" if gpu_vendor == "NVIDIA" else ("MPS" if gpu_vendor == "APPLE" else "NONE"),
            device_index=0
        )
        return cls(
            os_name=os_name,
            os_version="Mock-1.0",
            cpu_arch=cpu_arch,
            cpu_cores_logical=cpu_cores,
            cpu_cores_physical=max(1, cpu_cores // 2),
            total_ram_bytes=int(total_ram_gb * GB),
            available_ram_bytes=int(available_ram_gb * GB),
            available_storage_bytes=int(storage_gb * GB),
            gpu=gpu_info,
            environment_type=env_type,
            is_mobile=is_mobile,
            is_server=is_server,
            is_desktop=not is_mobile and not is_server
        )


class DeviceCapabilityDetector:
    """Probes host hardware and environment characteristics accurately."""

    @staticmethod
    def get_ram_info() -> Tuple[int, int]:
        """Returns (total_ram_bytes, available_ram_bytes)."""
        # Try psutil first if installed
        try:
            import psutil
            vm = psutil.virtual_memory()
            return int(vm.total), int(vm.available)
        except Exception:
            pass

        # Windows GlobalMemoryStatusEx
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
                if kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                    return int(stat.ullTotalPhys), int(stat.ullAvailPhys)
            except Exception:
                pass

        # Linux /proc/meminfo
        if sys.platform.startswith("linux"):
            try:
                mem_total = 0
                mem_avail = 0
                with open("/proc/meminfo", "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            mem_total = int(line.split()[1]) * 1024
                        elif line.startswith("MemAvailable:"):
                            mem_avail = int(line.split()[1]) * 1024
                if mem_total > 0:
                    if mem_avail == 0:
                        mem_avail = int(mem_total * 0.5)
                    return mem_total, mem_avail
            except Exception:
                pass

        # macOS sysctl
        if sys.platform == "darwin":
            try:
                out = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip()
                total = int(out)
                return total, int(total * 0.6)  # safe estimate
            except Exception:
                pass

        # Fallback conservative default
        return 8 * GB, 4 * GB

    @staticmethod
    def get_gpu_info() -> GpuInfo:
        """Detects GPU availability, vendor, and free VRAM."""
        # 1. PyTorch CUDA / MPS detection
        try:
            import torch
            if torch.cuda.is_available():
                idx = torch.cuda.current_device()
                name = torch.cuda.get_device_name(idx)
                total = torch.cuda.get_device_properties(idx).total_memory
                allocated = torch.cuda.memory_allocated(idx)
                free = total - allocated
                vendor = "NVIDIA"
                if "amd" in name.lower() or "radeon" in name.lower():
                    vendor = "AMD"
                elif "intel" in name.lower() or "arc" in name.lower():
                    vendor = "INTEL"

                return GpuInfo(
                    available=True,
                    vendor=vendor,
                    device_name=name,
                    total_vram_bytes=int(total),
                    free_vram_bytes=int(free),
                    type="CUDA",
                    device_index=idx
                )
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                total_ram, avail_ram = DeviceCapabilityDetector.get_ram_info()
                return GpuInfo(
                    available=True,
                    vendor="APPLE",
                    device_name="Apple Silicon (Unified Memory MPS)",
                    total_vram_bytes=int(total_ram * 0.7),
                    free_vram_bytes=int(avail_ram * 0.7),
                    type="MPS",
                    device_index=0
                )
        except Exception:
            pass

        # 2. nvidia-smi subprocess probe
        try:
            cmd = ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=2)
            if res.returncode == 0 and res.stdout.strip():
                parts = [p.strip() for p in res.stdout.strip().splitlines()[0].split(",")]
                if len(parts) >= 3:
                    return GpuInfo(
                        available=True,
                        vendor="NVIDIA",
                        device_name=parts[0],
                        total_vram_bytes=int(parts[1]) * MB,
                        free_vram_bytes=int(parts[2]) * MB,
                        type="CUDA",
                        device_index=0
                    )
        except Exception:
            pass

        return GpuInfo(available=False, vendor="NONE", device_name="None", total_vram_bytes=0, free_vram_bytes=0, type="NONE")

    @classmethod
    def detect(cls, root_storage_path: str = ".") -> DeviceProfile:
        """Runs full hardware detection and returns a DeviceProfile."""
        os_name = platform.system()
        os_ver = platform.release()
        cpu_arch = platform.machine()
        logical_cores = os.cpu_count() or 4
        physical_cores = max(1, logical_cores // 2)

        total_ram, avail_ram = cls.get_ram_info()
        gpu_info = cls.get_gpu_info()

        # Disk storage
        try:
            usage = shutil.disk_usage(root_storage_path)
            avail_storage = usage.free
        except Exception:
            avail_storage = 10 * GB

        # Mobile detection heuristic
        is_mobile = False
        if any(k in os.environ for k in ["ANDROID_ROOT", "ANDROID_DATA", "TERMUX_VERSION", "TERMUX_APP_PID"]):
            is_mobile = True
        elif os_name.lower() in ["ios"]:
            is_mobile = True
        elif ("arm" in cpu_arch.lower() or "aarch64" in cpu_arch.lower()) and total_ram <= 6 * GB and not gpu_info.available:
            is_mobile = True

        # Server detection heuristic
        is_server = False
        if not is_mobile:
            enterprise_gpus = ["a100", "h100", "v100", "t4", "l4", "a10g", "a40", "rtx 6000", "tesla"]
            gpu_name_l = gpu_info.device_name.lower()
            if any(eg in gpu_name_l for eg in enterprise_gpus):
                is_server = True
            elif logical_cores >= 32 or total_ram >= 64 * GB:
                is_server = True

        env_type = "MOBILE" if is_mobile else ("SERVER" if is_server else "DESKTOP")

        return DeviceProfile(
            os_name=os_name,
            os_version=os_ver,
            cpu_arch=cpu_arch,
            cpu_cores_logical=logical_cores,
            cpu_cores_physical=physical_cores,
            total_ram_bytes=total_ram,
            available_ram_bytes=avail_ram,
            available_storage_bytes=avail_storage,
            gpu=gpu_info,
            environment_type=env_type,
            is_mobile=is_mobile,
            is_server=is_server,
            is_desktop=not is_mobile and not is_server
        )
