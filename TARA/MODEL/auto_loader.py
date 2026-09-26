"""
TARA/MODEL/auto_loader.py
================================================================================
Automatic Hardware-Aware Model Loader for Permanent TARA Architecture
================================================================================

Seamlessly executes:
1. Device capability auto-detection (CPU, RAM, GPU, VRAM, OS, mobile/desktop).
2. Deterministic strategy selection (FULL_LOAD, LAZY_LOAD, PARTIAL_LOAD, QUANTIZED_LOAD).
3. ~100 MB SafeTensors shard auto-discovery and on-demand paging.
4. Dynamic INT8/INT4/FP16 quantization infrastructure.
5. Resource safety verification with automatic fallback chains.
6. Unified permanent TARA model identity preservation.
"""

import os
import sys
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union

from .device_detector import DeviceCapabilityDetector, DeviceProfile
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

logger = logging.getLogger("TARA.AutoLoader")


@dataclass
class LoadedTaraModel:
    """
    Handle for a running TARA model instance loaded via AutoTaraModelLoader.
    Treats all shards and quantized variants as the same logical TARA model.
    """
    model_dir: str
    config_dict: Dict[str, Any]
    profile: DeviceProfile
    plan: LoadingPlan
    shard_manager: ShardedSafeTensorsManager
    quantized_cache: Dict[str, QuantizedTensor] = field(default_factory=dict)
    full_weights: Optional[Dict[str, Any]] = None
    is_fallback_active: bool = False
    active_fallback_reason: str = ""

    @property
    def model_identity(self) -> str:
        return "TARA"

    @property
    def strategy(self) -> str:
        return self.plan.strategy.value

    @property
    def target_device(self) -> str:
        return self.plan.device_target.value

    @property
    def precision(self) -> str:
        return self.plan.precision.value

    def get_tensor(self, tensor_name: str) -> Any:
        """
        Retrieves tensor seamlessly under the active strategy:
        - FULL_LOAD: from memory cache
        - QUANTIZED_LOAD: retrieves quantized representation (or dequantizes if needed)
        - LAZY_LOAD: paged from 100MB shard via LRU
        - PARTIAL_LOAD: on-demand single read
        """
        if self.full_weights is not None and tensor_name in self.full_weights:
            return self.full_weights[tensor_name]

        if self.plan.strategy == ModelLoadingStrategy.QUANTIZED_LOAD:
            if tensor_name in self.quantized_cache:
                return self.quantized_cache[tensor_name]
            # Quantize on the fly from shard
            t_info = self.shard_manager.load_tensor(tensor_name)
            q_tensor = QuantizationEngine.quantize_to_precision(
                raw_bytes=t_info["bytes"],
                shape=t_info["shape"],
                name=tensor_name,
                target_precision=self.plan.precision,
                source_dtype=t_info["dtype"]
            )
            if isinstance(q_tensor, QuantizedTensor):
                self.quantized_cache[tensor_name] = q_tensor
            return q_tensor

        # LAZY_LOAD / PARTIAL_LOAD
        return self.shard_manager.load_tensor(tensor_name)

    def close(self):
        """Releases all open resources and file handles."""
        self.shard_manager.close()
        self.quantized_cache.clear()
        self.full_weights = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class AutoTaraModelLoader:
    """
    High-level entrypoint for automatic capability-driven model loading.
    Zero manual configuration required.
    """

    @staticmethod
    def resolve_canonical_model_dir(model_dir: Optional[str] = None) -> str:
        """Finds active canonical TARA model directory."""
        if model_dir and os.path.exists(os.path.join(model_dir, "config.json")):
            return os.path.abspath(model_dir)

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

        # Check current_model.json manifest
        manifest_p = os.path.join(repo_root, "TARA", "MODEL", "current_model.json")
        if os.path.exists(manifest_p):
            try:
                with open(manifest_p, "r", encoding="utf-8") as f:
                    mf = json.load(f)
                loc = mf.get("current_artifact_location", "")
                full_loc = os.path.join(repo_root, loc) if not os.path.isabs(loc) else loc
                if os.path.exists(os.path.join(full_loc, "config.json")):
                    return os.path.abspath(full_loc)
            except Exception:
                pass

        candidates = [
            os.path.join(repo_root, "storage", "models", "tara"),
            os.path.join(repo_root, "storage", "models", "TARA-0.1-tokenizer-aligned"),
            os.path.join(repo_root, "storage", "models", "tara-0.1"),
            os.path.join(repo_root, "storage", "models", "tara-0.2")
        ]
        for c in candidates:
            if os.path.exists(os.path.join(c, "config.json")):
                return os.path.abspath(c)

        raise FileNotFoundError(f"No valid TARA model directory found in workspace: {repo_root}")

    @classmethod
    def auto_load(
        cls,
        model_dir: Optional[str] = None,
        forced_profile: Optional[DeviceProfile] = None,
        forced_strategy: Optional[ModelLoadingStrategy] = None,
        verbose: bool = True
    ) -> LoadedTaraModel:
        """
        Main entrypoint:
        1. Detects hardware capabilities automatically (or uses forced_profile).
        2. Resolves canonical TARA model directory.
        3. Determines optimal loading strategy.
        4. Instantiates ShardedSafeTensorsManager and loads weights safely.
        5. Automatically executes fallback chain if resource limits trigger.
        """
        active_dir = cls.resolve_canonical_model_dir(model_dir)

        # 1. Hardware Detection
        profile = forced_profile or DeviceCapabilityDetector.detect(active_dir)

        if verbose:
            print("\n" + "=" * 80)
            print("         TARA RUNTIME: AUTOMATIC HARDWARE CAPABILITY DETECTOR")
            print("=" * 80)
            print(f"Platform / OS:        {profile.os_name} {profile.os_version} ({profile.cpu_arch})")
            print(f"CPU Cores:            {profile.cpu_cores_logical} Logical ({profile.cpu_cores_physical} Physical)")
            print(f"Total / Avail RAM:    {profile.total_ram_gb:.2f} GB / {profile.available_ram_gb:.2f} GB Free")
            if profile.gpu.available:
                print(f"GPU Accelerator:      {profile.gpu.vendor} {profile.gpu.device_name} ({profile.gpu.free_vram_gb:.2f} GB Free VRAM)")
            else:
                print("GPU Accelerator:      None (Optimized CPU Inference Active)")
            print(f"Environment Type:     {profile.environment_type}")
            print(f"Available Storage:    {profile.available_storage_gb:.2f} GB")
            print("=" * 80)

        # 2. Inspect Model Config
        cfg_file = os.path.join(active_dir, "config.json")
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)

        hidden_size = cfg_dict.get("hidden_size", 64)
        num_layers = cfg_dict.get("num_hidden_layers", 2)
        vocab_size = cfg_dict.get("vocab_size", 344)
        # Approximate parameter count
        param_count = cfg_dict.get("param_count") or (vocab_size * hidden_size * 2 + num_layers * hidden_size * hidden_size * 8)

        # 3. Formulate Deterministic Loading Plan
        initial_plan = LoadingPolicyEngine.create_loading_plan(
            profile=profile,
            model_size_bytes=int(param_count * 4),
            param_count=param_count
        )

        if forced_strategy:
            initial_plan.strategy = forced_strategy

        plans_to_try = [initial_plan] + initial_plan.fallback_chain

        last_error = None
        for plan_idx, plan in enumerate(plans_to_try):
            is_fallback = plan_idx > 0
            if verbose:
                prefix = "[Automatic Strategy Selection]" if not is_fallback else f"[Strategy Fallback #{plan_idx}]"
                print(f"{prefix} Strategy: {plan.strategy.value} | Device: {plan.device_target.value} | Precision: {plan.precision.value}")
                print(f"                           Est Memory: {plan.estimated_memory_mb:.1f} MB | Reason: {plan.reason}")

            try:
                # Initialize Shard Manager
                shard_mgr = ShardedSafeTensorsManager(
                    model_dir=active_dir,
                    max_cached_shards=plan.max_shard_cache_shards,
                    enable_mmap=plan.enable_mmap
                )

                loaded_model = LoadedTaraModel(
                    model_dir=active_dir,
                    config_dict=cfg_dict,
                    profile=profile,
                    plan=plan,
                    shard_manager=shard_mgr,
                    is_fallback_active=is_fallback,
                    active_fallback_reason=plan.reason if is_fallback else ""
                )

                # Execute Strategy
                if plan.strategy == ModelLoadingStrategy.FULL_LOAD:
                    loaded_model.full_weights = shard_mgr.load_all_tensors()
                    if verbose:
                        print(f"      -> FULL_LOAD Success: {len(loaded_model.full_weights)} tensors memory-mapped and verified.")

                elif plan.strategy == ModelLoadingStrategy.LAZY_LOAD:
                    if verbose:
                        print(f"      -> LAZY_LOAD Success: {shard_mgr.total_shards} shards discovered; on-demand paging ready.")

                elif plan.strategy == ModelLoadingStrategy.QUANTIZED_LOAD:
                    if verbose:
                        print(f"      -> QUANTIZED_LOAD Success: Dynamic {plan.precision.value} quantization active.")

                elif plan.strategy == ModelLoadingStrategy.PARTIAL_LOAD:
                    if verbose:
                        print(f"      -> PARTIAL_LOAD Success: Layer streaming minimal footprint active.")

                if verbose:
                    print("=" * 80 + "\n")

                return loaded_model

            except Exception as e:
                last_error = e
                if verbose:
                    print(f"      [!] Attempt with plan {plan.strategy.value} failed ({e}). Proceeding to next fallback plan...")
                continue

        # If all plans failed
        raise InsufficientMemoryError(f"FATAL: All model loading strategies and fallbacks failed. Last error: {last_error}")
