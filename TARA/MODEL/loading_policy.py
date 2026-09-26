"""
TARA/MODEL/loading_policy.py
================================================================================
Deterministic Hardware-Capability Policy Engine & Loading Strategy Selector
================================================================================

Selects optimal execution mode for the permanent TARA model:
- Strategy: FULL_LOAD, LAZY_LOAD, PARTIAL_LOAD, QUANTIZED_LOAD
- Target Device: GPU, CPU
- Precision: NONE_FP32, FP16, BF16, INT8, INT4
- Enforces strict resource safety limits and automatic fallback chains.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from .device_detector import DeviceProfile, GB, MB


class ModelLoadingStrategy(str, Enum):
    FULL_LOAD = "FULL_LOAD"
    LAZY_LOAD = "LAZY_LOAD"
    PARTIAL_LOAD = "PARTIAL_LOAD"
    QUANTIZED_LOAD = "QUANTIZED_LOAD"


class DeviceTarget(str, Enum):
    GPU = "GPU"
    CPU = "CPU"


class QuantizationPrecision(str, Enum):
    NONE_FP32 = "NONE_FP32"
    FP16 = "FP16"
    BF16 = "BF16"
    INT8 = "INT8"
    INT4 = "INT4"


@dataclass
class LoadingPlan:
    strategy: ModelLoadingStrategy
    device_target: DeviceTarget
    precision: QuantizationPrecision
    estimated_memory_bytes: int
    vram_budget_bytes: int
    ram_budget_bytes: int
    max_shard_cache_shards: int
    enable_mmap: bool
    fallback_chain: List["LoadingPlan"] = field(default_factory=list)
    reason: str = ""

    @property
    def estimated_memory_mb(self) -> float:
        return round(self.estimated_memory_bytes / MB, 2)

    def summary(self) -> str:
        return (
            f"LoadingPlan[Strategy={self.strategy.value} | Device={self.device_target.value} | "
            f"Precision={self.precision.value} | EstMemory={self.estimated_memory_mb:.1f} MB | "
            f"CacheShards={self.max_shard_cache_shards} | Reason='{self.reason}']"
        )


class InsufficientMemoryError(RuntimeError):
    """Raised when available system resources cannot safely support even the most constrained fallback model profile."""
    pass


class LoadingPolicyEngine:
    """
    Deterministic hardware-capability policy engine.
    Calculates safety margins, selects optimal loading strategy, and configures fallback chains.
    """

    BYTES_PER_PARAM = {
        QuantizationPrecision.NONE_FP32: 4.0,
        QuantizationPrecision.FP16: 2.0,
        QuantizationPrecision.BF16: 2.0,
        QuantizationPrecision.INT8: 1.0,
        QuantizationPrecision.INT4: 0.5,
    }

    @classmethod
    def estimate_model_memory(
        cls,
        param_count: int,
        precision: QuantizationPrecision,
        overhead_multiplier: float = 1.25
    ) -> int:
        """Estimates required memory in bytes for weights + minimal runtime buffers."""
        bpp = cls.BYTES_PER_PARAM.get(precision, 4.0)
        raw_bytes = int(param_count * bpp)
        return int(raw_bytes * overhead_multiplier)

    @classmethod
    def create_loading_plan(
        cls,
        profile: DeviceProfile,
        model_size_bytes: int,
        param_count: int = 118080,
        single_shard_size_bytes: int = 100 * MB
    ) -> LoadingPlan:
        """
        Determines the optimal loading plan and populates fallback options.
        
        Rules:
        1. Mobile:
           - Automatically prefer mobile-compatible quantized representation (INT4 / INT8)
           - Use LAZY_LOAD / PARTIAL_LOAD
           - CPU execution unless GPU with >= 2GB unified free is detected
        2. Low-RAM PC (avail_ram < 3.0 GB or total_ram <= 4.0 GB):
           - Use QUANTIZED_LOAD (INT8/INT4) with LAZY_LOAD or PARTIAL_LOAD
           - Restrict shard cache to 1-2 shards
        3. Medium PC (avail_ram >= 3.0 GB, < 8.0 GB or GPU VRAM < model):
           - Use LAZY_LOAD with 100 MB SafeTensors shards
           - Use GPU if VRAM fits working set, else CPU
        4. High-End PC / Sufficient VRAM:
           - If GPU has enough free VRAM for full model + 500MB safety buffer:
             FULL_LOAD on GPU
           - Else if available RAM >= 8.0 GB:
             FULL_LOAD on CPU (or LAZY_LOAD with fast NVMe mmap)
        5. Fallback Chain:
           - If preferred fails, step down through fallback chain:
             FULL_LOAD (GPU) -> FULL_LOAD (CPU) -> LAZY_LOAD (100MB shards) -> QUANTIZED_LOAD (INT8) -> QUANTIZED_LOAD (INT4 / PARTIAL)
        6. Insufficient Memory Protection:
           - If avail_ram < 80 MB and cannot fit minimum layer (approx 10 MB), reject with InsufficientMemoryError.
        """
        avail_ram = profile.available_ram_bytes
        avail_vram = profile.gpu.free_vram_bytes if profile.gpu.available else 0

        # Safety minimum check
        min_layer_budget = int(10 * MB)
        if avail_ram < min_layer_budget:
            raise InsufficientMemoryError(
                f"FATAL: Insufficient system memory ({profile.available_ram_gb:.2f} GB free). "
                f"TARA requires at least {min_layer_budget / MB:.0f} MB available RAM to execute safely."
            )

        vram_safety_margin = int(500 * MB)
        usable_vram = max(0, avail_vram - vram_safety_margin)
        usable_ram = int(avail_ram * 0.75)  # 25% safety reserve for host OS/applications

        # Estimate memory for different precisions
        fp32_mem = cls.estimate_model_memory(param_count, QuantizationPrecision.NONE_FP32)
        fp16_mem = cls.estimate_model_memory(param_count, QuantizationPrecision.FP16)
        int8_mem = cls.estimate_model_memory(param_count, QuantizationPrecision.INT8)
        int4_mem = cls.estimate_model_memory(param_count, QuantizationPrecision.INT4)

        plans: List[LoadingPlan] = []

        # --- Evaluate Scenarios ---

        # 1. MOBILE ENVIRONMENT
        if profile.is_mobile:
            chosen_prec = QuantizationPrecision.INT4 if profile.available_ram_gb < 2.0 else QuantizationPrecision.INT8
            est_mem = int4_mem if chosen_prec == QuantizationPrecision.INT4 else int8_mem
            plan = LoadingPlan(
                strategy=ModelLoadingStrategy.QUANTIZED_LOAD,
                device_target=DeviceTarget.CPU,
                precision=chosen_prec,
                estimated_memory_bytes=est_mem,
                vram_budget_bytes=0,
                ram_budget_bytes=usable_ram,
                max_shard_cache_shards=1,
                enable_mmap=True,
                reason="Mobile environment: Prioritizing low-memory quantized footprint with single-shard cache."
            )
            # Mobile fallbacks
            plan.fallback_chain = [
                LoadingPlan(
                    strategy=ModelLoadingStrategy.PARTIAL_LOAD,
                    device_target=DeviceTarget.CPU,
                    precision=QuantizationPrecision.INT4,
                    estimated_memory_bytes=int4_mem,
                    vram_budget_bytes=0,
                    ram_budget_bytes=usable_ram,
                    max_shard_cache_shards=1,
                    enable_mmap=False,
                    reason="Fallback: Partial layer-by-layer streaming on mobile CPU."
                )
            ]
            return plan

        # 2. HIGH-END PC WITH SUFFICIENT VRAM
        if profile.gpu.available and usable_vram >= fp16_mem:
            gpu_prec = QuantizationPrecision.FP16 if usable_vram < fp32_mem else QuantizationPrecision.NONE_FP32
            gpu_mem = fp16_mem if gpu_prec == QuantizationPrecision.FP16 else fp32_mem
            plan = LoadingPlan(
                strategy=ModelLoadingStrategy.FULL_LOAD,
                device_target=DeviceTarget.GPU,
                precision=gpu_prec,
                estimated_memory_bytes=gpu_mem,
                vram_budget_bytes=usable_vram,
                ram_budget_bytes=usable_ram,
                max_shard_cache_shards=100,
                enable_mmap=False,
                reason=f"High-end hardware: Sufficient VRAM ({profile.gpu.free_vram_gb:.1f} GB) for full GPU acceleration."
            )
            # Add fallbacks
            plan.fallback_chain = [
                LoadingPlan(
                    strategy=ModelLoadingStrategy.FULL_LOAD,
                    device_target=DeviceTarget.CPU,
                    precision=QuantizationPrecision.NONE_FP32,
                    estimated_memory_bytes=fp32_mem,
                    vram_budget_bytes=0,
                    ram_budget_bytes=usable_ram,
                    max_shard_cache_shards=100,
                    enable_mmap=True,
                    reason="Fallback 1: Full load on host CPU memory."
                ),
                LoadingPlan(
                    strategy=ModelLoadingStrategy.LAZY_LOAD,
                    device_target=DeviceTarget.CPU,
                    precision=QuantizationPrecision.NONE_FP32,
                    estimated_memory_bytes=single_shard_size_bytes * 2,
                    vram_budget_bytes=0,
                    ram_budget_bytes=usable_ram,
                    max_shard_cache_shards=4,
                    enable_mmap=True,
                    reason="Fallback 2: Lazy loading with 100MB shards."
                )
            ]
            return plan

        # 3. HIGH-RAM PC WITHOUT GPU (OR INSUFFICIENT VRAM)
        if usable_ram >= fp32_mem * 2 and profile.available_ram_gb >= 6.0:
            plan = LoadingPlan(
                strategy=ModelLoadingStrategy.FULL_LOAD,
                device_target=DeviceTarget.CPU,
                precision=QuantizationPrecision.NONE_FP32,
                estimated_memory_bytes=fp32_mem,
                vram_budget_bytes=0,
                ram_budget_bytes=usable_ram,
                max_shard_cache_shards=50,
                enable_mmap=True,
                reason=f"High-RAM PC ({profile.available_ram_gb:.1f} GB RAM): Full model fit in host memory."
            )
            plan.fallback_chain = [
                LoadingPlan(
                    strategy=ModelLoadingStrategy.LAZY_LOAD,
                    device_target=DeviceTarget.CPU,
                    precision=QuantizationPrecision.NONE_FP32,
                    estimated_memory_bytes=single_shard_size_bytes * 2,
                    vram_budget_bytes=0,
                    ram_budget_bytes=usable_ram,
                    max_shard_cache_shards=4,
                    enable_mmap=True,
                    reason="Fallback: Lazy 100MB shard streaming."
                )
            ]
            return plan

        # 4. MEDIUM HARDWARE (Avail RAM 3.0GB - 6.0GB)
        if usable_ram >= fp16_mem and profile.available_ram_gb >= 2.5:
            plan = LoadingPlan(
                strategy=ModelLoadingStrategy.LAZY_LOAD,
                device_target=DeviceTarget.CPU,
                precision=QuantizationPrecision.NONE_FP32,
                estimated_memory_bytes=single_shard_size_bytes * 2,
                vram_budget_bytes=0,
                ram_budget_bytes=usable_ram,
                max_shard_cache_shards=4,
                enable_mmap=True,
                reason="Medium hardware: 100 MB SafeTensors shards with lazy on-demand paging."
            )
            plan.fallback_chain = [
                LoadingPlan(
                    strategy=ModelLoadingStrategy.QUANTIZED_LOAD,
                    device_target=DeviceTarget.CPU,
                    precision=QuantizationPrecision.INT8,
                    estimated_memory_bytes=int8_mem,
                    vram_budget_bytes=0,
                    ram_budget_bytes=usable_ram,
                    max_shard_cache_shards=2,
                    enable_mmap=True,
                    reason="Fallback: Quantized INT8 lazy loading."
                )
            ]
            return plan

        # 5. LOW-RAM PC (< 2.5GB RAM)
        target_prec = QuantizationPrecision.INT4 if profile.available_ram_gb < 1.5 else QuantizationPrecision.INT8
        est_mem = int4_mem if target_prec == QuantizationPrecision.INT4 else int8_mem
        plan = LoadingPlan(
            strategy=ModelLoadingStrategy.QUANTIZED_LOAD,
            device_target=DeviceTarget.CPU,
            precision=target_prec,
            estimated_memory_bytes=est_mem,
            vram_budget_bytes=0,
            ram_budget_bytes=usable_ram,
            max_shard_cache_shards=2,
            enable_mmap=True,
            reason=f"Low-RAM PC ({profile.available_ram_gb:.1f} GB RAM): Automatic {target_prec.value} quantization with lazy loading."
        )
        plan.fallback_chain = [
            LoadingPlan(
                strategy=ModelLoadingStrategy.PARTIAL_LOAD,
                device_target=DeviceTarget.CPU,
                precision=QuantizationPrecision.INT4,
                estimated_memory_bytes=int4_mem,
                vram_budget_bytes=0,
                ram_budget_bytes=usable_ram,
                max_shard_cache_shards=1,
                enable_mmap=False,
                reason="Emergency Fallback: Layer-by-layer minimal streaming."
            )
        ]
        return plan
