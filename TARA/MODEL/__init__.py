"""
TARA/MODEL/__init__.py
================================================================================
TARA AI Neural Model Subsystem: Loading, Tiering & Shard Management
================================================================================
"""

from .core.tiered_model import TieredTaraModel, run_verification_inference
from .device_detector import DeviceCapabilityDetector, DeviceProfile, GpuInfo
from .loading_policy import (
    LoadingPolicyEngine,
    LoadingPlan,
    ModelLoadingStrategy,
    DeviceTarget,
    QuantizationPrecision,
    InsufficientMemoryError
)
from .shard_manager import ShardedSafeTensorsManager
from .quantization import QuantizationEngine, QuantizedTensor
from .auto_loader import AutoTaraModelLoader, LoadedTaraModel

__all__ = [
    "TieredTaraModel",
    "run_verification_inference",
    "DeviceCapabilityDetector",
    "DeviceProfile",
    "GpuInfo",
    "LoadingPolicyEngine",
    "LoadingPlan",
    "ModelLoadingStrategy",
    "DeviceTarget",
    "QuantizationPrecision",
    "InsufficientMemoryError",
    "ShardedSafeTensorsManager",
    "QuantizationEngine",
    "QuantizedTensor",
    "AutoTaraModelLoader",
    "LoadedTaraModel"
]
