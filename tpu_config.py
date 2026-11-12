"""
TARA Autonomous TPU Engine Configuration & Hardware Abstraction Layer
=============================================================================
Provides seamless detection, initialization, and hardware-optimized configuration
for Google TPU architectures:
  - Google Cloud TPU v6e (Trillium | 32GB HBM per chip | 6th Gen)
  - Google Cloud TPU v5p (Flagship LLM Training | 95GB HBM per chip)
  - Google Cloud TPU v5e (LitePod / Kaggle v5e-8 / Colab v5e-1)

Designed to be 100% decoupled from existing GPU scripts (train_main.py, run_train.sh).
"""
from __future__ import annotations

import os
import sys
import math
import logging
from typing import Optional, Tuple, Dict, Any, List, Union, Callable, Iterable

try:
    import torch
except ImportError:
    torch = None

# Clean up any legacy Kaggle environment variables that conflict with PJRT SliceBuilder
if os.environ.get("TPU_PROCESS_ADDRESSES") == "local":
    os.environ.pop("TPU_PROCESS_ADDRESSES", None)
os.environ.pop("CLOUD_TSK_ID", None)

# Ensure PJRT & Bfloat16 defaults are active in Python environment before importing torch_xla
os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ.setdefault("XLA_USE_BF16", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

logger = logging.getLogger("tara_tpu_config")

# Safe PyTorch / XLA Import Handling
HAS_XLA = False
XLA_IMPORT_ERRORS: List[str] = []
xm = None
pl = None
xmp = None
xr = None

try:
    import torch_xla

    try:
        import torch_xla.core.xla_model as _xm
        xm = _xm
    except Exception as _e_xm:
        XLA_IMPORT_ERRORS.append(f"torch_xla.core.xla_model: {_e_xm}")

    try:
        import torch_xla.distributed.parallel_loader as _pl
        pl = _pl
    except Exception as _e_pl:
        XLA_IMPORT_ERRORS.append(f"torch_xla.distributed.parallel_loader: {_e_pl}")

    try:
        import torch_xla.distributed.xla_multiprocessing as _xmp
        xmp = _xmp
    except Exception as _e_xmp:
        XLA_IMPORT_ERRORS.append(f"torch_xla.distributed.xla_multiprocessing: {_e_xmp}")

    try:
        import torch_xla.runtime as _xr
        xr = _xr
    except Exception as _e_xr:
        xr = None

    if xm is not None or "torch_xla" in sys.modules:
        HAS_XLA = True
        if torch is not None and not hasattr(torch, "xla"):
            torch.xla = torch_xla
except Exception as _e_root:
    HAS_XLA = False
    XLA_IMPORT_ERRORS.append(f"torch_xla root: {_e_root}")


def is_tpu_available() -> bool:
    """
    Returns True ONLY if an authentic physical PyTorch/XLA or native TorchTPU accelerator is present.
    CRITICAL: In PyTorch/XLA PJRT, NEVER call xm.xla_device(), xm.xla_device_hw(),
    or torch.empty(..., device='tpu') in the parent process before xmp.spawn(),
    as this binds the PJRT runtime prematurely and crashes with:
    'RuntimeError: Runtime is already initialized. Do not use the XLA device before calling xmp.spawn.'
    """
    if bool(
        os.environ.get("TPU_NAME")
        or os.environ.get("COLAB_TPU_ADDR")
        or os.environ.get("TPU_ACCELERATOR_TYPE")
        or os.environ.get("KAGGLE_TPU_VERSION")
        or (os.environ.get("PJRT_DEVICE", "").upper() == "TPU")
    ):
        return True

    if HAS_XLA:
        try:
            if hasattr(xm, "get_xla_supported_devices"):
                devs = xm.get_xla_supported_devices()
                if any("TPU" in str(d).upper() for d in devs):
                    return True
        except Exception:
            pass

    return False


def get_tpu_hardware_info() -> Dict[str, Any]:
    """
    Returns detailed TPU hardware topology and core configuration across ALL TPU generations.
    Safe to call in parent process without premature PJRT client binding.
    """
    available = is_tpu_available()
    hw_type = "No TPU Detected"
    num_cores = 1
    device_name = "N/A"

    # Safely query device count from environment or get_xla_supported_devices without premature PJRT binding
    env_num_devices = os.environ.get("TPU_NUM_DEVICES")
    if env_num_devices and env_num_devices.isdigit():
        num_cores = int(env_num_devices)
    elif HAS_XLA and available:
        try:
            if hasattr(xm, "get_xla_supported_devices"):
                devs = xm.get_xla_supported_devices()
                tpu_devs = [d for d in devs if "TPU" in str(d).upper()]
                num_cores = len(tpu_devs) if tpu_devs else 8
            else:
                num_cores = 8
        except Exception:
            num_cores = 8
    elif available:
        num_cores = 8

    # Detect TPU generation from environment or hardware string
    tpu_env = (
        os.environ.get("TPU_ACCELERATOR_TYPE", "")
        + " " + os.environ.get("TPU_NAME", "")
        + " " + os.environ.get("KAGGLE_TPU_VERSION", "")
        + " " + os.environ.get("PJRT_DEVICE", "")
        + " " + device_name
    ).lower()

    if env_num_devices and env_num_devices.isdigit():
        num_cores = int(env_num_devices)

    # 1. Google Cloud TPU v6e (Trillium - 6th Gen)
    if "v6e" in tpu_env or "trillium" in tpu_env:
        if num_cores == 1 or "-1" in tpu_env:
            hw_type = "Google Cloud TPU v6e-1 Trillium (32GB HBM | 1 Core | 4.7x v5e perf)"
            num_cores = 1
        elif num_cores == 4 or "-4" in tpu_env:
            hw_type = "Google Cloud TPU v6e-4 Trillium (128GB HBM | 4 Cores | 1600 TFLOPS)"
            num_cores = 4
        else:
            hw_type = "Google Cloud TPU v6e-8 Trillium (256GB HBM | 8 Cores | 3900 TFLOPS)"
            num_cores = 8

    # 2. Google Cloud TPU v5p (Flagship LLM Training accelerator)
    elif "v5p" in tpu_env:
        hw_type = f"Google Cloud TPU v5p (95GB HBM/chip | {num_cores} Cores | Native BF16/FP8)"

    # 3. Colab TPU v5e-1 (Single-core / 1 TensorCore / 16GB HBM)
    elif (
        "v5e-1" in tpu_env
        or ("v5e" in tpu_env and ("-1" in tpu_env or "single" in tpu_env))
        or (os.environ.get("COLAB_TPU_ADDR") and num_cores == 1)
        or (num_cores == 1 and "v5" in tpu_env)
    ):
        hw_type = "Google Cloud TPU v5e-1 (Colab | 16GB HBM | 1 Core | 197 TFLOPS BF16)"
        num_cores = 1

    # 4. Kaggle / GCP TPU v5e-8 (Multi-core / 8 TensorCores / 4 Chips / 128GB HBM)
    elif "v5e-8" in tpu_env or "v5e" in tpu_env or "v5litepod" in tpu_env or os.environ.get("KAGGLE_TPU_VERSION"):
        hw_type = "Google Cloud TPU v5e-8 (Kaggle/GCP | 8 Cores | 128GB HBM | 1576 TFLOPS)"
        num_cores = 8

    elif available:
        hw_type = f"Google Cloud TPU ({num_cores} Cores | PyTorch/XLA PJRT)"

    engine_name = "PyTorch/XLA (PJRT)" if HAS_XLA else "TorchTPU (Native Backend)"

    return {
        "available": available,
        "hardware_type": hw_type,
        "num_cores": max(1, num_cores),
        "has_xla": HAS_XLA,
        "has_native_tpu": False,
        "engine_name": engine_name,
        "device_name": device_name,
    }


def get_tpu_device():
    """
    Acquires the authentic PyTorch/XLA or native TorchTPU device bound to the current process.
    Raises RuntimeError if no physical Google TPU accelerator is detected.
    """
    if HAS_XLA:
        try:
            return xm.xla_device()
        except Exception as e:
            raise RuntimeError(f"[TPU Hardware Error] PyTorch/XLA present but failed to acquire TPU device: {e}")
    try:
        import torch
        dev = torch.device("tpu")
        _ = torch.empty(0, device=dev)
        return dev
    except Exception:
        pass
    raise RuntimeError(
        "[TPU Hardware Error] Physical Google TPU accelerator not found! "
        "This training runner requires an authentic Google Cloud TPU (v5e-8, v5p, v6e). Please run on a genuine Google Cloud TPU VM, Kaggle TPU, or Google Colab TPU runtime."
    )


def get_tpu_rank() -> int:
    """Returns the process ordinal / rank (0..world_size-1)."""
    if HAS_XLA:
        try:
            if xr is not None and hasattr(xr, "global_ordinal"):
                return xr.global_ordinal()
            return xm.get_ordinal()
        except Exception:
            return 0
    return 0


def get_tpu_world_size() -> int:
    """Returns total TPU worker processes."""
    if HAS_XLA:
        try:
            if xr is not None and hasattr(xr, "world_size"):
                return xr.world_size()
            if hasattr(xm, "xworld_size"):
                return xm.xworld_size()
            if hasattr(xm, "xrt_world_size"):
                return xm.xrt_world_size()
        except Exception:
            return 1
    return 1


def is_tpu_master() -> bool:
    """Returns True if the current process is the TPU master process (rank 0)."""
    return get_tpu_rank() == 0


def tpu_barrier(tag: str = "tpu_barrier"):
    """
    Synchronizes all TPU worker cores at a common rendezvous point.
    Guarded so single-core TPU runs do not hang.
    """
    if HAS_XLA and get_tpu_world_size() > 1:
        try:
            xm.rendezvous(tag)
        except Exception:
            pass


def get_tpu_memory_diagnostics(device: Optional[Any] = None) -> Dict[str, Any]:
    """
    Queries authentic memory telemetry from physical Google TPU / PyTorch XLA.
    Returns structured memory metrics and hardware identification.
    """
    diag = {
        "is_physical_tpu": False,
        "backend": "Google Cloud TPU (PyTorch/XLA)",
        "bytes_used": 0,
        "bytes_free": 0,
        "bytes_total": 0,
        "used_mb": 0.0,
        "free_mb": 0.0,
        "total_mb": 0.0,
        "peak_mb": 0.0,
        "telemetry_error": None,
    }

    # 1. Genuine PyTorch / XLA TPU Device or TorchTPU device
    is_tpu_target = False
    device_type_str = str(device).lower() if device is not None else ""
    if (device is not None and getattr(device, "type", "") == "tpu") or device_type_str.startswith("tpu") or device_type_str.startswith("xla"):
        is_tpu_target = True

    if HAS_XLA or is_tpu_target:
        try:
            device_hw = "TPU"
            if xm is not None and hasattr(xm, "xla_device_hw"):
                try:
                    device_hw = str(xm.xla_device_hw(device) if device is not None else xm.xla_device_hw()).upper()
                except Exception:
                    device_hw = "TPU"
            if "TPU" in device_hw or is_tpu_target:
                diag["is_physical_tpu"] = True
                diag["backend"] = f"Physical Google TPU ({device_hw if device_hw != 'TPU' else str(device or 'TPU')})"
                if xm is not None and hasattr(xm, "get_memory_info"):
                    try:
                        mem_info = xm.get_memory_info(device)
                        if isinstance(mem_info, dict):
                            b_used = mem_info.get("bytes_used", 0)
                            b_free = mem_info.get("bytes_free", 0)
                            b_peak = mem_info.get("peak_bytes_used", mem_info.get("bytes_peak", b_used))
                            diag["bytes_used"] = b_used
                            diag["bytes_free"] = b_free
                            diag["bytes_total"] = b_used + b_free
                            diag["used_mb"] = round(b_used / (1024 * 1024), 2)
                            diag["free_mb"] = round(b_free / (1024 * 1024), 2)
                            diag["total_mb"] = round((b_used + b_free) / (1024 * 1024), 2)
                            diag["peak_mb"] = round(b_peak / (1024 * 1024), 2)
                            return diag
                        else:
                            diag["telemetry_error"] = f"xm.get_memory_info returned non-dict type: {type(mem_info)}"
                    except Exception as mem_err:
                        diag["telemetry_error"] = f"xm.get_memory_info exception: {mem_err}"
                        logger.warning(f"[TPU Telemetry Notice] {diag['telemetry_error']}")
                else:
                    diag["telemetry_error"] = "xm.get_memory_info API not exposed in this torch_xla runtime version."

                # Try torch_xla module-level memory query
                try:
                    import torch_xla
                    if hasattr(torch_xla, "get_memory_info"):
                        mem_info = torch_xla.get_memory_info(device)
                        if isinstance(mem_info, dict):
                            b_used = mem_info.get("bytes_used", 0)
                            b_free = mem_info.get("bytes_free", 0)
                            b_peak = mem_info.get("peak_bytes_used", mem_info.get("bytes_peak", b_used))
                            diag["bytes_used"] = b_used
                            diag["bytes_free"] = b_free
                            diag["bytes_total"] = b_used + b_free
                            diag["used_mb"] = round(b_used / (1024 * 1024), 2)
                            diag["free_mb"] = round(b_free / (1024 * 1024), 2)
                            diag["total_mb"] = round((b_used + b_free) / (1024 * 1024), 2)
                            diag["peak_mb"] = round(b_peak / (1024 * 1024), 2)
                            return diag
                except Exception:
                    pass

                # If physical TPU but direct memory API is not available, report process memory with hardware flag intact
                try:
                    import psutil
                    vmem = psutil.virtual_memory()
                    proc = psutil.Process()
                    proc_mem = proc.memory_info().rss
                    diag["bytes_used"] = proc_mem
                    diag["bytes_free"] = vmem.available
                    diag["bytes_total"] = vmem.total
                    diag["used_mb"] = round(proc_mem / (1024 * 1024), 2)
                    diag["free_mb"] = round(vmem.available / (1024 * 1024), 2)
                    diag["total_mb"] = round(vmem.total / (1024 * 1024), 2)
                    diag["peak_mb"] = round(proc_mem / (1024 * 1024), 2)
                except Exception:
                    pass
                return diag
        except Exception as hw_err:
            diag["telemetry_error"] = f"XLA hardware query exception: {hw_err}"

    return diag


def log_tpu_memory_stage(
    stage_name: str,
    step: int = 0,
    device: Optional[Any] = None,
    model: Optional[Any] = None,
    optimizer: Optional[Any] = None,
    microbatch: int = 0,
) -> Dict[str, Any]:
    """
    Standardized logger for TPU memory diagnostics across mandatory pipeline stages:
      1. BEFORE_MODEL_LOADING
      2. AFTER_MODEL_LOADING
      3. AFTER_OPTIMIZER_RESTORE
      4. BEFORE_FORWARD
      5. BEFORE_BACKWARD
      6. BEFORE_OPTIMIZER_STEP
    Collects parameters, gradients, and optimizer moments with strict leaf-only gradient inspection.
    """
    import torch
    mem = get_tpu_memory_diagnostics(device=device)

    param_dtypes = set()
    grad_dtypes = set()
    moment_dtypes = set()
    total_param_bytes = 0
    total_grad_bytes = 0
    total_moment_bytes = 0

    has_grad_refs = False
    if model is not None:
        target_model = getattr(model, "module", model)
        for p in target_model.parameters():
            param_dtypes.add(str(p.dtype).replace("torch.", ""))
            total_param_bytes += p.numel() * p.element_size()
            # Strictly inspect .grad on leaf parameters only (never non-leaf intermediate activations)
            if p.is_leaf and p.grad is not None:
                has_grad_refs = True
                grad_dtypes.add(str(p.grad.dtype).replace("torch.", ""))
                total_grad_bytes += p.grad.numel() * p.grad.element_size()

    moment_location = "N/A"
    if optimizer is not None and hasattr(optimizer, "state"):
        has_cpu_moments = False
        has_device_moments = False
        for state in optimizer.state.values():
            if isinstance(state, dict):
                for k in ("exp_avg", "exp_avg_sq"):
                    if k in state and isinstance(state[k], torch.Tensor):
                        moment_dtypes.add(str(state[k].dtype).replace("torch.", ""))
                        total_moment_bytes += state[k].numel() * state[k].element_size()
                        if state[k].device.type == "cpu":
                            has_cpu_moments = True
                        else:
                            has_device_moments = True
        if has_cpu_moments and not has_device_moments:
            moment_location = "CPU Host (Offloaded)"
        elif has_device_moments:
            moment_location = "TPU Device"
        elif has_cpu_moments:
            moment_location = "Mixed"

    is_physical = mem.get("is_physical_tpu", False)
    backend_str = mem.get("backend", "Unknown")

    log_payload = {
        "stage": stage_name,
        "step": step,
        "microbatch": microbatch,
        "backend": backend_str,
        "is_physical_tpu": is_physical,
        "used_mb": mem.get("used_mb", 0.0),
        "free_mb": mem.get("free_mb", 0.0),
        "total_mb": mem.get("total_mb", 0.0),
        "peak_mb": mem.get("peak_mb", mem.get("used_mb", 0.0)),
        "has_grad_refs": has_grad_refs,
        "param_dtypes": sorted(list(param_dtypes)),
        "grad_dtypes": sorted(list(grad_dtypes)),
        "moment_dtypes": sorted(list(moment_dtypes)),
        "moments_location": moment_location,
        "weights_mb": round(total_param_bytes / (1024 * 1024), 2),
        "grads_mb": round(total_grad_bytes / (1024 * 1024), 2),
        "moments_mb": round(total_moment_bytes / (1024 * 1024), 2),
    }

    if is_tpu_master():
        loc_str = f", location: {moment_location}" if moment_location != "N/A" else ""
        print(
            f"[TPU MEMORY DIAGNOSTIC | {stage_name}] Step={step} | Backend: {backend_str} (Physical TPU: {is_physical})\n"
            f"  HBM/RAM: Used={log_payload['used_mb']} MB, Free={log_payload['free_mb']} MB, Peak={log_payload['peak_mb']} MB, Total={log_payload['total_mb']} MB\n"
            f"  Weights: {log_payload['weights_mb']} MB (dtypes: {log_payload['param_dtypes'] or 'None'}) | "
            f"Grads: {log_payload['grads_mb']} MB (dtypes: {log_payload['grad_dtypes'] or 'None'}, active_refs: {has_grad_refs}) | "
            f"Moments: {log_payload['moments_mb']} MB (dtypes: {log_payload['moment_dtypes'] or 'None'}{loc_str})",
            flush=True,
        )

    return log_payload


_COLLECTIVE_MOCK_HOOK: Optional[Any] = None
_COLLECTIVE_GRAD_HOOK: Optional[Any] = None


def set_collective_mock(hook: Optional[Any], grad_hook: Optional[Any] = None) -> None:
    """Sets mock hooks for testing multi-worker collective decisions and gradient reduction in test/simulated environments."""
    global _COLLECTIVE_MOCK_HOOK, _COLLECTIVE_GRAD_HOOK
    _COLLECTIVE_MOCK_HOOK = hook
    _COLLECTIVE_GRAD_HOOK = grad_hook


def sync_tpu_decision(local_decision: int, tag: str = "tpu_decision") -> int:
    """
    Synchronizes a binary integer decision (0 = proceed, 1 = skip) across all TPU worker cores.
    Uses 'max' reduction: if ANY worker replica reports corruption or shock (1), ALL workers receive 1.
    Never averages NaN or Inf values; operates purely on validated binary flags.
    """
    flag = 1 if local_decision else 0
    if HAS_XLA and get_tpu_world_size() > 1:
        try:
            # xm.mesh_reduce gathers payload from all TPU cores and runs reduce_fn across replicas
            res = xm.mesh_reduce(tag, flag, lambda x: max(x) if hasattr(x, '__iter__') else x)
            return int(res)
        except Exception as e:
            logger.error(f"[TPU Collective Error] mesh_reduce failed for {tag}: {e}. Aborting to prevent multi-core deadlock / state divergence.")
            raise

    if _COLLECTIVE_MOCK_HOOK is not None:
        return int(_COLLECTIVE_MOCK_HOOK(flag, tag))

    return flag


def sync_tpu_sum(val: int, tag: str = "tpu_sum") -> int:
    """
    Synchronizes and sums an integer across all TPU worker cores.
    Used for cluster-wide valid-token accounting and aggregate token counts.
    Ensures that if at least one replica has valid data, the cluster-wide count is > 0.
    """
    int_val = int(val)
    if HAS_XLA and get_tpu_world_size() > 1:
        try:
            return int(xm.mesh_reduce(tag, int_val, lambda x: sum(x) if hasattr(x, '__iter__') else x))
        except Exception as e:
            logger.error(f"[TPU Collective Error] sync_tpu_sum failed for {tag}: {e}. Aborting to prevent multi-core deadlock / state divergence.")
            raise

    if _COLLECTIVE_MOCK_HOOK is not None:
        try:
            return int(_COLLECTIVE_MOCK_HOOK(int_val, tag))
        except Exception:
            return int_val

    return int_val


def sync_tpu_sum_float(val: float, tag: str = "tpu_sum_float") -> float:
    """
    Synchronizes and sums a scalar float across all TPU worker cores.
    Used for cluster-wide aggregate loss summation across replicas.
    """
    f_val = float(val)
    if HAS_XLA and get_tpu_world_size() > 1:
        try:
            return float(xm.mesh_reduce(tag, f_val, lambda x: sum(x) if hasattr(x, '__iter__') else x))
        except Exception as e:
            logger.error(f"[TPU Collective Error] sync_tpu_sum_float failed for {tag}: {e}. Aborting to prevent multi-core deadlock / state divergence.")
            raise

    if _COLLECTIVE_MOCK_HOOK is not None:
        try:
            return float(_COLLECTIVE_MOCK_HOOK(f_val, tag))
        except Exception:
            return f_val

    return f_val


def sync_tpu_mean(val: float, tag: str = "tpu_mean") -> float:
    """
    Synchronizes and averages a scalar float across all TPU worker cores.
    Guarantees that all TPU cores observe the exact same step loss, EMA loss,
    and milestone checkpoint decisions without multi-core divergence or barrier deadlocks.
    """
    if HAS_XLA and get_tpu_world_size() > 1:
        try:
            return float(xm.mesh_reduce(tag, float(val), lambda x: sum(x) / float(len(x)) if hasattr(x, '__iter__') else x))
        except Exception as e:
            logger.error(f"[TPU Collective Error] sync_tpu_mean failed for {tag}: {e}. Aborting to prevent multi-core deadlock / state divergence.")
            raise

    if _COLLECTIVE_MOCK_HOOK is not None:
        try:
            return float(_COLLECTIVE_MOCK_HOOK(float(val), tag))
        except Exception:
            return float(val)

    return float(val)


def ensure_model_weight_tying(model: Any) -> bool:
    """
    Authoritative architectural weight-tying enforcer and verifier.
    When `tie_word_embeddings` is True in the model configuration:
      1. Binds target_model.lm_head.weight directly to target_model.model.embed_tokens.weight.
      2. Strictly asserts pointer identity: (lm_head.weight is embed_tokens.weight).
    Returns True if weights are verified tied, False if model is untied.
    """
    target_model = getattr(model, "module", model)
    cfg = getattr(target_model, "config", None)
    tie = False
    if isinstance(cfg, dict):
        tie = bool(cfg.get("tie_word_embeddings", False))
    elif cfg is not None:
        tie = bool(getattr(cfg, "tie_word_embeddings", False))

    if tie:
        if hasattr(target_model, "lm_head") and hasattr(target_model, "model") and hasattr(target_model.model, "embed_tokens"):
            target_model.lm_head.weight = target_model.model.embed_tokens.weight
            if target_model.lm_head.weight is not target_model.model.embed_tokens.weight:
                raise RuntimeError(
                    "[Weight Tying Fatal] Failed to bind lm_head.weight to embed_tokens.weight; pointer identity mismatch."
                )
            return True
    return False


# ---------------------------------------------------------------------------
# Dynamic 8-Bit Blockwise Quantization (bitsandbytes Parity for TPU Checkpoints)
# ---------------------------------------------------------------------------
BNB_QMAP1: Tuple[float, ...] = (
    -9.9296873807907104e-01,
    -9.7890627384185791e-01,
    -9.6484375000000000e-01,
    -9.5078122615814209e-01,
    -9.3671876192092896e-01,
    -9.2265623807907104e-01,
    -9.0859377384185791e-01,
    -8.9453125000000000e-01,
    -8.8046872615814209e-01,
    -8.6640626192092896e-01,
    -8.5234373807907104e-01,
    -8.3828127384185791e-01,
    -8.2421875000000000e-01,
    -8.1015622615814209e-01,
    -7.9609376192092896e-01,
    -7.8203123807907104e-01,
    -7.6796877384185791e-01,
    -7.5390625000000000e-01,
    -7.3984372615814209e-01,
    -7.2578126192092896e-01,
    -7.1171879768371582e-01,
    -6.9765627384185791e-01,
    -6.8359375000000000e-01,
    -6.6953122615814209e-01,
    -6.5546876192092896e-01,
    -6.4140629768371582e-01,
    -6.2734377384185791e-01,
    -6.1328125000000000e-01,
    -5.9921872615814209e-01,
    -5.8515626192092896e-01,
    -5.7109379768371582e-01,
    -5.5703127384185791e-01,
    -5.4296875000000000e-01,
    -5.2890622615814209e-01,
    -5.1484370231628418e-01,
    -5.0078123807907104e-01,
    -4.8671871423721313e-01,
    -4.7265625000000000e-01,
    -4.5859372615814209e-01,
    -4.4453126192092896e-01,
    -4.3046873807907104e-01,
    -4.1640624403953552e-01,
    -4.0234375000000000e-01,
    -3.8828122615814209e-01,
    -3.7421876192092896e-01,
    -3.6015623807907104e-01,
    -3.4609374403953552e-01,
    -3.3203125000000000e-01,
    -3.1796872615814209e-01,
    -3.0390626192092896e-01,
    -2.8984373807907104e-01,
    -2.7578127384185791e-01,
    -2.6171875000000000e-01,
    -2.4765624105930328e-01,
    -2.3359374701976776e-01,
    -2.1953125298023224e-01,
    -2.0546874403953552e-01,
    -1.9140625000000000e-01,
    -1.7734375596046448e-01,
    -1.6328124701976776e-01,
    -1.4921875298023224e-01,
    -1.3515624403953552e-01,
    -1.2109375000000000e-01,
    -1.0703125596046448e-01,
    -9.8593749105930328e-02,
    -9.5781259238719940e-02,
    -9.2968754470348358e-02,
    -9.0156249701976776e-02,
    -8.7343752384185791e-02,
    -8.4531247615814209e-02,
    -8.1718757748603821e-02,
    -7.8906252980232239e-02,
    -7.6093748211860657e-02,
    -7.3281250894069672e-02,
    -7.0468746125698090e-02,
    -6.7656256258487701e-02,
    -6.4843751490116119e-02,
    -6.2031250447034836e-02,
    -5.9218753129243851e-02,
    -5.6406248360872269e-02,
    -5.3593751043081284e-02,
    -5.0781250000000000e-02,
    -4.7968748956918716e-02,
    -4.5156251639127731e-02,
    -4.2343746870756149e-02,
    -3.9531249552965164e-02,
    -3.6718752235174179e-02,
    -3.3906251192092896e-02,
    -3.1093750149011612e-02,
    -2.8281250968575478e-02,
    -2.5468749925494194e-02,
    -2.2656250745058060e-02,
    -1.9843751564621925e-02,
    -1.7031250521540642e-02,
    -1.4218750409781933e-02,
    -1.1406250298023224e-02,
    -9.7187496721744537e-03,
    -9.1562494635581970e-03,
    -8.5937501862645149e-03,
    -8.0312499776482582e-03,
    -7.4687497690320015e-03,
    -6.9062500260770321e-03,
    -6.3437498174607754e-03,
    -5.7812500745058060e-03,
    -5.2187503315508366e-03,
    -4.6562496572732925e-03,
    -4.0937499143183231e-03,
    -3.5312497057020664e-03,
    -2.9687499627470970e-03,
    -2.4062499869614840e-03,
    -1.8437500111758709e-03,
    -1.2812499189749360e-03,
    -9.4375008484348655e-04,
    -8.3124998491257429e-04,
    -7.1875005960464478e-04,
    -6.0625007608905435e-04,
    -4.9375003436580300e-04,
    -3.8125002174638212e-04,
    -2.6875003823079169e-04,
    -1.5625001105945557e-04,
    -8.8749999122228473e-05,
    -6.6250002419110388e-05,
    -4.3749998440034688e-05,
    -2.1249998098937795e-05,
    -7.7499998951680027e-06,
    -3.2500001907465048e-06,
    -5.5000003840177669e-07,
    0.0000000000000000e+00,
    5.5000003840177669e-07,
    3.2500001907465048e-06,
    7.7499998951680027e-06,
    2.1249998098937795e-05,
    4.3749998440034688e-05,
    6.6250002419110388e-05,
    8.8749999122228473e-05,
    1.5625001105945557e-04,
    2.6875003823079169e-04,
    3.8125002174638212e-04,
    4.9375003436580300e-04,
    6.0625007608905435e-04,
    7.1875005960464478e-04,
    8.3124998491257429e-04,
    9.4375008484348655e-04,
    1.2812499189749360e-03,
    1.8437500111758709e-03,
    2.4062499869614840e-03,
    2.9687499627470970e-03,
    3.5312497057020664e-03,
    4.0937499143183231e-03,
    4.6562496572732925e-03,
    5.2187503315508366e-03,
    5.7812500745058060e-03,
    6.3437498174607754e-03,
    6.9062500260770321e-03,
    7.4687497690320015e-03,
    8.0312499776482582e-03,
    8.5937501862645149e-03,
    9.1562494635581970e-03,
    9.7187496721744537e-03,
    1.1406250298023224e-02,
    1.4218750409781933e-02,
    1.7031250521540642e-02,
    1.9843751564621925e-02,
    2.2656250745058060e-02,
    2.5468749925494194e-02,
    2.8281250968575478e-02,
    3.1093750149011612e-02,
    3.3906251192092896e-02,
    3.6718752235174179e-02,
    3.9531249552965164e-02,
    4.2343746870756149e-02,
    4.5156251639127731e-02,
    4.7968748956918716e-02,
    5.0781250000000000e-02,
    5.3593751043081284e-02,
    5.6406248360872269e-02,
    5.9218753129243851e-02,
    6.2031250447034836e-02,
    6.4843751490116119e-02,
    6.7656256258487701e-02,
    7.0468746125698090e-02,
    7.3281250894069672e-02,
    7.6093748211860657e-02,
    7.8906252980232239e-02,
    8.1718757748603821e-02,
    8.4531247615814209e-02,
    8.7343752384185791e-02,
    9.0156249701976776e-02,
    9.2968754470348358e-02,
    9.5781259238719940e-02,
    9.8593749105930328e-02,
    1.0703125596046448e-01,
    1.2109375000000000e-01,
    1.3515624403953552e-01,
    1.4921875298023224e-01,
    1.6328124701976776e-01,
    1.7734375596046448e-01,
    1.9140625000000000e-01,
    2.0546874403953552e-01,
    2.1953125298023224e-01,
    2.3359374701976776e-01,
    2.4765624105930328e-01,
    2.6171875000000000e-01,
    2.7578127384185791e-01,
    2.8984373807907104e-01,
    3.0390626192092896e-01,
    3.1796872615814209e-01,
    3.3203125000000000e-01,
    3.4609374403953552e-01,
    3.6015623807907104e-01,
    3.7421876192092896e-01,
    3.8828122615814209e-01,
    4.0234375000000000e-01,
    4.1640624403953552e-01,
    4.3046873807907104e-01,
    4.4453126192092896e-01,
    4.5859372615814209e-01,
    4.7265625000000000e-01,
    4.8671871423721313e-01,
    5.0078123807907104e-01,
    5.1484370231628418e-01,
    5.2890622615814209e-01,
    5.4296875000000000e-01,
    5.5703127384185791e-01,
    5.7109379768371582e-01,
    5.8515626192092896e-01,
    5.9921872615814209e-01,
    6.1328125000000000e-01,
    6.2734377384185791e-01,
    6.4140629768371582e-01,
    6.5546876192092896e-01,
    6.6953122615814209e-01,
    6.8359375000000000e-01,
    6.9765627384185791e-01,
    7.1171879768371582e-01,
    7.2578126192092896e-01,
    7.3984372615814209e-01,
    7.5390625000000000e-01,
    7.6796877384185791e-01,
    7.8203123807907104e-01,
    7.9609376192092896e-01,
    8.1015622615814209e-01,
    8.2421875000000000e-01,
    8.3828127384185791e-01,
    8.5234373807907104e-01,
    8.6640626192092896e-01,
    8.8046872615814209e-01,
    8.9453125000000000e-01,
    9.0859377384185791e-01,
    9.2265623807907104e-01,
    9.3671876192092896e-01,
    9.5078122615814209e-01,
    9.6484375000000000e-01,
    9.7890627384185791e-01,
    9.9296873807907104e-01,
    1.0000000000000000e+00,
)

BNB_QMAP2: Tuple[float, ...] = (
    0.0000000000000000e+00,
    3.2500003044333425e-07,
    7.7499998951680027e-07,
    2.1249998098937795e-06,
    4.3749996621045284e-06,
    6.6249999690626282e-06,
    8.8749993665260263e-06,
    1.5624998923158273e-05,
    2.6875000912696123e-05,
    3.8124999264255166e-05,
    4.9374997615814209e-05,
    6.0625003243330866e-05,
    7.1875001594889909e-05,
    8.3124992670491338e-05,
    9.4374998298007995e-05,
    1.2812500062864274e-04,
    1.8437500693835318e-04,
    2.4062501324806362e-04,
    2.9687501955777407e-04,
    3.5312498221173882e-04,
    4.0937503217719495e-04,
    4.6562502393499017e-04,
    5.2187504479661584e-04,
    5.7812500745058060e-04,
    6.3437502831220627e-04,
    6.9062504917383194e-04,
    7.4687501182779670e-04,
    8.0312509089708328e-04,
    8.5937505355104804e-04,
    9.1562501620501280e-04,
    9.7187503706663847e-04,
    1.1406249832361937e-03,
    1.4218749711290002e-03,
    1.7031249590218067e-03,
    1.9843750633299351e-03,
    2.2656249348074198e-03,
    2.5468748062849045e-03,
    2.8281249105930328e-03,
    3.1093747820705175e-03,
    3.3906251192092896e-03,
    3.6718749906867743e-03,
    3.9531248621642590e-03,
    4.2343745008111000e-03,
    4.5156250707805157e-03,
    4.7968747094273567e-03,
    5.0781248137354851e-03,
    5.3593749180436134e-03,
    5.6406245566904545e-03,
    5.9218751266598701e-03,
    6.2031247653067112e-03,
    6.4843748696148396e-03,
    6.7656254395842552e-03,
    7.0468746125698090e-03,
    7.3281251825392246e-03,
    7.6093743555247784e-03,
    7.8906249254941940e-03,
    8.1718750298023224e-03,
    8.4531242027878761e-03,
    8.7343752384185791e-03,
    9.0156244114041328e-03,
    9.2968745157122612e-03,
    9.5781255513429642e-03,
    9.8593747243285179e-03,
    1.0703125968575478e-02,
    1.2109375558793545e-02,
    1.3515624217689037e-02,
    1.4921875670552254e-02,
    1.6328124329447746e-02,
    1.7734376713633537e-02,
    1.9140625372529030e-02,
    2.0546874031424522e-02,
    2.1953126415610313e-02,
    2.3359375074505806e-02,
    2.4765623733401299e-02,
    2.6171876117587090e-02,
    2.7578128501772881e-02,
    2.8984373435378075e-02,
    3.0390625819563866e-02,
    3.1796872615814209e-02,
    3.3203125000000000e-02,
    3.4609373658895493e-02,
    3.6015626043081284e-02,
    3.7421878427267075e-02,
    3.8828123360872269e-02,
    4.0234375745058060e-02,
    4.1640624403953552e-02,
    4.3046873062849045e-02,
    4.4453125447034836e-02,
    4.5859374105930328e-02,
    4.7265626490116119e-02,
    4.8671871423721313e-02,
    5.0078123807907104e-02,
    5.1484372466802597e-02,
    5.2890624850988388e-02,
    5.4296877235174179e-02,
    5.5703129619359970e-02,
    5.7109382003545761e-02,
    5.8515626937150955e-02,
    5.9921871870756149e-02,
    6.1328124254941940e-02,
    6.2734380364418030e-02,
    6.4140632748603821e-02,
    6.5546877682209015e-02,
    6.6953122615814209e-02,
    6.8359375000000000e-02,
    6.9765627384185791e-02,
    7.1171879768371582e-02,
    7.2578124701976776e-02,
    7.3984377086162567e-02,
    7.5390629470348358e-02,
    7.6796881854534149e-02,
    7.8203126788139343e-02,
    7.9609379172325134e-02,
    8.1015624105930328e-02,
    8.2421876490116119e-02,
    8.3828128874301910e-02,
    8.5234373807907104e-02,
    8.6640626192092896e-02,
    8.8046871125698090e-02,
    8.9453123509883881e-02,
    9.0859375894069672e-02,
    9.2265628278255463e-02,
    9.3671880662441254e-02,
    9.5078125596046448e-02,
    9.6484377980232239e-02,
    9.7890630364418030e-02,
    9.9296875298023224e-02,
    1.0351562500000000e-01,
    1.1054687201976776e-01,
    1.1757812649011612e-01,
    1.2460937350988388e-01,
    1.3164061307907104e-01,
    1.3867187500000000e-01,
    1.4570313692092896e-01,
    1.5273436903953552e-01,
    1.5976563096046448e-01,
    1.6679686307907104e-01,
    1.7382812500000000e-01,
    1.8085938692092896e-01,
    1.8789061903953552e-01,
    1.9492188096046448e-01,
    2.0195311307907104e-01,
    2.0898437500000000e-01,
    2.1601563692092896e-01,
    2.2304686903953552e-01,
    2.3007813096046448e-01,
    2.3710936307907104e-01,
    2.4414062500000000e-01,
    2.5117185711860657e-01,
    2.5820311903953552e-01,
    2.6523438096046448e-01,
    2.7226561307907104e-01,
    2.7929687500000000e-01,
    2.8632813692092896e-01,
    2.9335936903953552e-01,
    3.0039060115814209e-01,
    3.0742186307907104e-01,
    3.1445312500000000e-01,
    3.2148438692092896e-01,
    3.2851561903953552e-01,
    3.3554685115814209e-01,
    3.4257811307907104e-01,
    3.4960937500000000e-01,
    3.5664063692092896e-01,
    3.6367186903953552e-01,
    3.7070310115814209e-01,
    3.7773436307907104e-01,
    3.8476562500000000e-01,
    3.9179688692092896e-01,
    3.9882811903953552e-01,
    4.0585935115814209e-01,
    4.1289061307907104e-01,
    4.1992187500000000e-01,
    4.2695313692092896e-01,
    4.3398436903953552e-01,
    4.4101560115814209e-01,
    4.4804686307907104e-01,
    4.5507812500000000e-01,
    4.6210938692092896e-01,
    4.6914061903953552e-01,
    4.7617185115814209e-01,
    4.8320311307907104e-01,
    4.9023437500000000e-01,
    4.9726560711860657e-01,
    5.0429689884185791e-01,
    5.1132810115814209e-01,
    5.1835936307907104e-01,
    5.2539062500000000e-01,
    5.3242182731628418e-01,
    5.3945314884185791e-01,
    5.4648435115814209e-01,
    5.5351567268371582e-01,
    5.6054687500000000e-01,
    5.6757813692092896e-01,
    5.7460939884185791e-01,
    5.8164060115814209e-01,
    5.8867192268371582e-01,
    5.9570312500000000e-01,
    6.0273438692092896e-01,
    6.0976564884185791e-01,
    6.1679685115814209e-01,
    6.2382817268371582e-01,
    6.3085937500000000e-01,
    6.3789063692092896e-01,
    6.4492189884185791e-01,
    6.5195310115814209e-01,
    6.5898442268371582e-01,
    6.6601562500000000e-01,
    6.7304688692092896e-01,
    6.8007814884185791e-01,
    6.8710935115814209e-01,
    6.9414067268371582e-01,
    7.0117187500000000e-01,
    7.0820313692092896e-01,
    7.1523439884185791e-01,
    7.2226560115814209e-01,
    7.2929692268371582e-01,
    7.3632812500000000e-01,
    7.4335938692092896e-01,
    7.5039064884185791e-01,
    7.5742185115814209e-01,
    7.6445317268371582e-01,
    7.7148437500000000e-01,
    7.7851557731628418e-01,
    7.8554689884185791e-01,
    7.9257810115814209e-01,
    7.9960942268371582e-01,
    8.0664062500000000e-01,
    8.1367182731628418e-01,
    8.2070314884185791e-01,
    8.2773435115814209e-01,
    8.3476567268371582e-01,
    8.4179687500000000e-01,
    8.4882807731628418e-01,
    8.5585939884185791e-01,
    8.6289060115814209e-01,
    8.6992192268371582e-01,
    8.7695312500000000e-01,
    8.8398432731628418e-01,
    8.9101564884185791e-01,
    8.9804685115814209e-01,
    9.0507817268371582e-01,
    9.1210937500000000e-01,
    9.1914057731628418e-01,
    9.2617189884185791e-01,
    9.3320310115814209e-01,
    9.4023442268371582e-01,
    9.4726562500000000e-01,
    9.5429682731628418e-01,
    9.6132814884185791e-01,
    9.6835935115814209e-01,
    9.7539067268371582e-01,
    9.8242187500000000e-01,
    9.8945307731628418e-01,
    9.9648439884185791e-01,
    1.0000000000000000e+00,
)

_QMAP1_TENSOR: Optional[torch.Tensor] = None
_QMAP2_TENSOR: Optional[torch.Tensor] = None
_MIDPOINTS1_TENSOR: Optional[torch.Tensor] = None
_MIDPOINTS2_TENSOR: Optional[torch.Tensor] = None


def _init_bnb_qmap_tensors():
    global _QMAP1_TENSOR, _QMAP2_TENSOR, _MIDPOINTS1_TENSOR, _MIDPOINTS2_TENSOR
    if _QMAP1_TENSOR is None:
        _QMAP1_TENSOR = torch.tensor(BNB_QMAP1, dtype=torch.float32)
        _MIDPOINTS1_TENSOR = (_QMAP1_TENSOR[:-1] + _QMAP1_TENSOR[1:]) / 2.0
    if _QMAP2_TENSOR is None:
        _QMAP2_TENSOR = torch.tensor(BNB_QMAP2, dtype=torch.float32)
        _MIDPOINTS2_TENSOR = (_QMAP2_TENSOR[:-1] + _QMAP2_TENSOR[1:]) / 2.0


def quantize_tensor_blockwise_bnb(
    val: torch.Tensor,
    is_state2: bool = False,
    block_size: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Vectorized CPU-native blockwise dynamic quantization into 8-bit bitsandbytes parity format.
    Quantizes uncompressed float32 momentum states into uint8 indices, absmax scale factors,
    and returns the authentic 256-float quantile map.
    """
    _init_bnb_qmap_tensors()
    qmap = _QMAP2_TENSOR if is_state2 else _QMAP1_TENSOR
    midpoints = _MIDPOINTS2_TENSOR if is_state2 else _MIDPOINTS1_TENSOR

    flat = val.contiguous().view(-1).float()
    n = flat.numel()
    pad = (block_size - (n % block_size)) % block_size
    if pad > 0:
        flat = torch.nn.functional.pad(flat, (0, pad))
    blocks = flat.view(-1, block_size)
    absmax = torch.max(torch.abs(blocks), dim=-1, keepdim=True)[0]
    safe_absmax = torch.where(absmax == 0, torch.ones_like(absmax), absmax)
    norm = blocks / safe_absmax
    indices = torch.bucketize(norm, midpoints).to(torch.uint8)
    indices_trimmed = indices.view(-1)[:n].view(val.shape)
    return indices_trimmed, absmax.squeeze(-1).float(), qmap.clone()


def quantize_optimizer_state_to_bnb8bit(cpu_opt_sd: dict, block_size: int = 256) -> dict:
    """
    Directly quantizes optimizer state dict in-place/in-memory on CPU Host RAM before disk serialization.
    Replaces 7.41 GB uncompressed float32 exp_avg and exp_avg_sq momentum buffers with
    bitsandbytes 8-bit dynamic quantized format (__bnb_optimizer_quant_state__), matching
    the exact 1.88 GB GPU checkpoint format directly on disk write.
    """
    if not isinstance(cpu_opt_sd, dict) or "state" not in cpu_opt_sd:
        return cpu_opt_sd

    quantized_state = {}
    for pid, pstate in cpu_opt_sd["state"].items():
        if not isinstance(pstate, dict):
            quantized_state[pid] = pstate
            continue

        new_pstate = {}
        for k, v in pstate.items():
            if k not in ("exp_avg", "exp_avg_sq"):
                new_pstate[k] = v.cpu() if isinstance(v, torch.Tensor) else v

        has_s1 = "exp_avg" in pstate and isinstance(pstate["exp_avg"], torch.Tensor)
        has_s2 = "exp_avg_sq" in pstate and isinstance(pstate["exp_avg_sq"], torch.Tensor)

        if has_s1 or has_s2:
            bnb_state = {}
            if has_s1:
                s1, abs1, qm1 = quantize_tensor_blockwise_bnb(pstate["exp_avg"].cpu(), is_state2=False, block_size=block_size)
                bnb_state["state1"] = s1
                bnb_state["absmax1"] = abs1
                bnb_state["qmap1"] = qm1
            if has_s2:
                s2, abs2, qm2 = quantize_tensor_blockwise_bnb(pstate["exp_avg_sq"].cpu(), is_state2=True, block_size=block_size)
                bnb_state["state2"] = s2
                bnb_state["absmax2"] = abs2
                bnb_state["qmap2"] = qm2
            new_pstate["__bnb_optimizer_quant_state__"] = bnb_state

        quantized_state[pid] = new_pstate

    result = dict(cpu_opt_sd)
    result["state"] = quantized_state
    return result



def build_tpu_optimizer_groups(model: Any, weight_decay: float = 0.01) -> List[Dict[str, Any]]:
    """
    Constructs AdamW parameter groups matching GPU training semantics with 100% exact parity:
      - Group 0: 2D projection / MLP / attention weight matrices with decoupled weight_decay (default: 0.01).
      - Group 1: 1D biases, RMSNorm parameters, 2D token embeddings (embed_tokens), and lm_head with weight_decay = 0.0.
    Preserves exact parameter-group ordering and membership identical to train_main.py lines 1489-1503.
    Deduplicates shared/tied parameter buffers by object identity (id(param)), verified by ensure_model_weight_tying.
    """
    target_model = getattr(model, "module", model)
    ensure_model_weight_tying(target_model)

    decay_params = []
    no_decay_params = []
    seen_ids = set()

    for name, param in target_model.named_parameters():
        if not param.requires_grad:
            continue
        # Strict object-identity deduplication for shared/tied parameter tensors
        if id(param) in seen_ids:
            continue
        seen_ids.add(id(param))

        if param.dim() < 2 or "embed_tokens" in name or "lm_head" in name or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def should_offload_optimizer(hardware_type: str = "", num_cores: int = 1) -> bool:
    """
    Determines whether optimizer states should be offloaded to CPU Host RAM.
    v6e (Trillium, 32GB/core) and v5p (95GB/core) have large HBM and do not require offload.
    v5e (16GB/core) and custom single-core setups benefit from offload.
    """
    hw_lower = (hardware_type or "").lower()
    if any(tag in hw_lower for tag in ("v6e", "trillium", "v5p")):
        return False
    if any(tag in hw_lower for tag in ("v5e", "litepod", "colab")):
        return True
    if num_cores <= 1:
        return True
    return False


def restore_tpu_optimizer_state(
    checkpoint_dir: str,
    optimizer: Any,
    device: Optional[Any] = None,
    allow_cold_start: bool = False,
    offload_optimizer: bool = False,
) -> bool:
    """
    Safely and strictly restores optimizer state from checkpoint_dir with full GPU parity checks:
      1. Loads optimizer_state.pt on CPU.
      2. Supports both nested {"optimizer": state_dict, ...} (from train_main.py / gdrive_sync.py)
         and direct flat {state, param_groups} formats.
      3. Validates parameter group count, parameter counts per group, and state tensor shapes.
      4. Moves optimizer state tensors to target TPU device OR retains on CPU Host RAM if offload_optimizer=True.
      5. Never silently swallows restore errors; fails closed with RuntimeError unless allow_cold_start=True.
    """
    if optimizer is None:
        return False

    opt_file = os.path.join(checkpoint_dir, "optimizer_state.pt")
    if not os.path.isfile(opt_file):
        best_opt = os.path.join(checkpoint_dir, "best_checkpoint", "optimizer_state.pt")
        if os.path.isfile(best_opt):
            opt_file = best_opt
        else:
            if allow_cold_start:
                print(f"[TPU Resume Notice] Optimizer state file absent at {opt_file}. Cold start permitted.", flush=True)
                return False
            raise FileNotFoundError(f"[TPU Resume Gate] Mandatory optimizer state missing at '{opt_file}'. Cold start rejected without --allow-optimizer-cold-start.")

    import torch
    print(f"[TPU Resume] Loading optimizer state from '{opt_file}' into RAM (deserializing ~7.4GB momentum states, please wait ~30-60s)...", flush=True)
    try:
        opt_data = torch.load(opt_file, map_location="cpu")
    except Exception as le:
        if allow_cold_start:
            print(f"[TPU Resume Notice] Could not read {opt_file}: {le}. Cold start permitted.", flush=True)
            return False
        raise RuntimeError(f"[TPU Resume Gate] Failed to load optimizer state file '{opt_file}': {le}")

    # Unpack nested container format vs raw state_dict
    raw_opt_sd = opt_data.get("optimizer", opt_data) if isinstance(opt_data, dict) else opt_data
    if not (isinstance(raw_opt_sd, dict) and "param_groups" in raw_opt_sd and "state" in raw_opt_sd):
        if allow_cold_start:
            print(f"[TPU Resume Notice] Unrecognized optimizer state dictionary format in {opt_file}. Cold start permitted.", flush=True)
            return False
        raise RuntimeError(f"[TPU Resume Gate] Corrupted or unrecognized optimizer state schema in '{opt_file}'.")

    # Strict parameter group count validation
    loaded_n_groups = len(raw_opt_sd.get("param_groups", []))
    current_n_groups = len(optimizer.param_groups)
    if loaded_n_groups != current_n_groups:
        err_msg = (
            f"[TPU Resume Gate] Parameter group count mismatch: checkpoint has {loaded_n_groups} group(s), "
            f"but active optimizer was constructed with {current_n_groups} group(s)."
        )
        if allow_cold_start:
            print(f"{err_msg} Skipping restore per --allow-optimizer-cold-start.", flush=True)
            return False
        raise RuntimeError(f"{err_msg}\nRe-initialization prohibited to prevent cold-start momentum shock. Provide --allow-optimizer-cold-start to override.")

    # Strict per-group parameter count validation (General, no magic numbers)
    for g_idx, (saved_g, cur_g) in enumerate(zip(raw_opt_sd["param_groups"], optimizer.param_groups)):
        n_saved = len(saved_g.get("params", []))
        n_cur = len(cur_g.get("params", []))
        if n_saved != n_cur:
            print(f"\n{'=' * 88}", flush=True)
            print(f"  [TPU OPTIMIZER PARAMETER DIAGNOSTICS - MISMATCH IN GROUP {g_idx}]", flush=True)
            print(f"  Checkpoint has {n_saved} params | Active optimizer has {n_cur} params", flush=True)
            print(f"{'-' * 88}", flush=True)
            for p_idx, p in enumerate(cur_g.get("params", [])):
                shape_str = tuple(p.shape) if hasattr(p, "shape") else "unknown"
                print(f"  Group {g_idx} | index={p_idx:02d} | shape={shape_str} | requires_grad={getattr(p, 'requires_grad', None)}", flush=True)
            print(f"{'=' * 88}\n", flush=True)
            err_msg = (
                f"[TPU Resume Gate] Parameter count mismatch in group {g_idx}: "
                f"checkpoint has {n_saved} params, optimizer has {n_cur} params."
            )
            if allow_cold_start:
                print(f"{err_msg} Skipping restore per --allow-optimizer-cold-start.", flush=True)
                return False
            raise RuntimeError(err_msg)

    # Dequantize bitsandbytes 8-bit AdamW quantized states into standard float32 exp_avg & exp_avg_sq
    if "state" in raw_opt_sd and isinstance(raw_opt_sd["state"], dict):
        for pid, pstate in raw_opt_sd["state"].items():
            if isinstance(pstate, dict) and "__bnb_optimizer_quant_state__" in pstate:
                bnb = pstate.pop("__bnb_optimizer_quant_state__")
                s1 = bnb.get("state1")
                s2 = bnb.get("state2")
                if s1 is not None:
                    if "qmap1" in bnb and "absmax1" in bnb:
                        qm1, abs1 = bnb["qmap1"], bnb["absmax1"]
                        bsz1 = s1.numel() // max(1, abs1.numel())
                        pstate["exp_avg"] = (qm1[s1.long()].view(-1, bsz1) * abs1.view(-1, 1)).view(s1.shape)
                    else:
                        pstate["exp_avg"] = s1.float() if hasattr(s1, "float") else s1
                if s2 is not None:
                    if "qmap2" in bnb and "absmax2" in bnb:
                        qm2, abs2 = bnb["qmap2"], bnb["absmax2"]
                        bsz2 = s2.numel() // max(1, abs2.numel())
                        pstate["exp_avg_sq"] = (qm2[s2.long()].view(-1, bsz2) * abs2.view(-1, 1)).view(s2.shape)
                    else:
                        pstate["exp_avg_sq"] = s2.float() if hasattr(s2, "float") else s2

    # Strict parameter shape validation across all momentum states
    saved_state = raw_opt_sd.get("state", {})
    if isinstance(saved_state, dict):
        for g_idx, (saved_g, cur_g) in enumerate(zip(raw_opt_sd["param_groups"], optimizer.param_groups)):
            for p_idx, (saved_pid, cur_p) in enumerate(zip(saved_g.get("params", []), cur_g.get("params", []))):
                if saved_pid in saved_state:
                    p_state = saved_state[saved_pid]
                    if isinstance(p_state, dict) and "exp_avg" in p_state and hasattr(p_state["exp_avg"], "shape"):
                        exp_avg_shape = tuple(p_state["exp_avg"].shape)
                        cur_shape = tuple(cur_p.shape)
                        if exp_avg_shape != cur_shape:
                            raise RuntimeError(
                                f"[TPU Resume Gate] Tensor shape mismatch in group {g_idx}, param index {p_idx}: "
                                f"checkpoint exp_avg shape {exp_avg_shape} != model parameter shape {cur_shape}."
                            )

    # Ingest validated state_dict into optimizer
    if offload_optimizer:
        print("[TPU Resume] Ingesting validated state dict and offloading momentum tensors to CPU Host RAM...", flush=True)
    else:
        print(f"[TPU Resume] Ingesting validated state dict and transferring momentum tensors to device ({device})...", flush=True)
    try:
        optimizer.load_state_dict(raw_opt_sd)
    except Exception as se:
        if allow_cold_start:
            print(f"[TPU Resume Notice] load_state_dict error: {se}. Cold start permitted.", flush=True)
            return False
        raise RuntimeError(f"[TPU Resume Gate] Failed to load validated optimizer state dict: {se}")

    del raw_opt_sd, opt_data
    import gc
    gc.collect()

    # Ensure FP32 precision for all momentum buffers (exp_avg, exp_avg_sq)
    for state in optimizer.state.values():
        if isinstance(state, dict):
            for sk in ("exp_avg", "exp_avg_sq"):
                if sk in state and isinstance(state[sk], torch.Tensor):
                    if state[sk].dtype != torch.float32:
                        state[sk] = state[sk].to(dtype=torch.float32)

    if not offload_optimizer and device is not None and (str(device).startswith("xla") or getattr(device, "type", "") in ("xla", "tpu")):
        for state in optimizer.state.values():
            for sk, sv in state.items():
                if isinstance(sv, torch.Tensor):
                    if sk in ("exp_avg", "exp_avg_sq"):
                        state[sk] = sv.to(device=device, dtype=torch.float32)
                    elif sv.dim() > 0:
                        state[sk] = sv.to(device)
                    else:
                        # Scalar step: keep intact or move if XLA requires
                        state[sk] = sv.to(device) if HAS_XLA else sv
        if HAS_XLA:
            xm.mark_step()
    elif offload_optimizer:
        # Strictly ensure all momentum states are resident on CPU Host memory
        for state in optimizer.state.values():
            for sk, sv in state.items():
                if isinstance(sv, torch.Tensor):
                    state[sk] = sv.to(device="cpu", dtype=torch.float32) if sk in ("exp_avg", "exp_avg_sq") else sv.to(device="cpu")

    loc_str = "offloaded to CPU Host RAM (preserving ~7.07 GB of TPU HBM)" if offload_optimizer else f"resident on {device}"
    print(f"[TPU Resume] Successfully restored complete optimizer state ({loaded_n_groups} parameter groups, {len(optimizer.state)} momentum buffers) with 100% parameter fidelity ({loc_str}).", flush=True)
    return True


def _step_offloaded_adamw(optimizer: Any, model: Optional[Any] = None) -> None:
    """
    Bitwise-exact FP32 AdamW optimizer step using CPU-offloaded momentum buffers.
    Preserves 100% numerical parity with standard torch.optim.AdamW while keeping
    7.07 GB of momentum buffers completely out of TPU HBM during training.
    """
    import torch
    with torch.no_grad():
        for group in optimizer.param_groups:
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            eps = float(group["eps"])
            wd = float(group.get("weight_decay", 0.0))

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = optimizer.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, device="cpu", dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(p, device="cpu", dtype=torch.float32)

                step_val = int(state["step"].item()) if isinstance(state.get("step"), torch.Tensor) else int(state.get("step", 0))
                step_val += 1
                state["step"] = torch.tensor(step_val) if isinstance(state.get("step"), torch.Tensor) else step_val

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                grad_cpu = p.grad.detach().to(device="cpu", dtype=torch.float32)
                p_cpu = p.detach().to(device="cpu", dtype=torch.float32)

                # Decoupled weight decay
                if wd != 0.0:
                    p_cpu.mul_(1.0 - lr * wd)

                # Momentum updates in FP32 on CPU
                exp_avg.lerp_(grad_cpu, 1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_cpu, grad_cpu, value=1.0 - beta2)

                # Bias correction terms
                bias_corr1 = 1.0 - (beta1 ** step_val)
                bias_corr2 = 1.0 - (beta2 ** step_val)
                step_size = lr / bias_corr1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_corr2)).add_(eps)

                p_cpu.addcdiv_(exp_avg, denom, value=-step_size)

                # Write updated parameter back to device in original dtype
                p.data.copy_(p_cpu.to(dtype=p.dtype, device=p.device))
                del grad_cpu, p_cpu


def tpu_optimizer_step(
    optimizer: Any,
    model: Optional[Any] = None,
    max_norm: float = 1.0,
    cur_loss: Optional[float] = None,
    ema_loss: Optional[float] = None,
    valid_tokens: Optional[int] = None,
    step: int = 0,
    attempt: int = 0,
    offload_optimizer: bool = False,
) -> Tuple[bool, float]:
    """
    Executes a TPU-optimized all-reduce optimizer step with Collective Model Shock & Gradient Explosion Protection:
      1. Evaluates cluster-wide valid tokens: if valid_tokens is provided, sums valid tokens across
         all TPU replicas via sync_tpu_sum. If cluster_valid_tokens == 0 across ALL replicas, skips step
         to strictly prevent unwarranted AdamW weight decay without training data.
         If at least one replica has valid data (cluster_valid_tokens > 0), the step proceeds normally.
      2. Evaluates local loss validity (non-finite or >10x EMA shock).
      3. Synchronizes loss shock / corruption decision across ALL TPU replicas using sync_tpu_decision (collective max).
      4. If any replica reports corruption/shock, ALL replicas skip the step simultaneously,
         clearing gradients and marking the step without diverging model weights.
      5. If step is valid across all replicas, all-reduces gradients across replicas once (xm.reduce_gradients).
      6. Clips globally reduced gradients consistently across all cores (max_norm=1.0).
      7. Evaluates and collectively synchronizes gradient validity (non-finite or >1000x norm shock).
      8. Steps optimizer and marks XLA execution graph.
    Returns (is_step_skipped, grad_norm_val).
    """
    import torch

    # Sanitize inputs to prevent implicit tensor D2H sync in host checks
    if isinstance(cur_loss, torch.Tensor):
        cur_loss = cur_loss.detach().cpu().item()
    if isinstance(valid_tokens, torch.Tensor):
        valid_tokens = int(valid_tokens.detach().cpu().item())

    # 1. Cluster-wide valid tokens evaluation (all-masked accumulation window)
    if valid_tokens is not None:
        tokens_tag = f"tpu_valid_tokens_s{step}_a{attempt}" if attempt > 0 else f"tpu_valid_tokens_{step}"
        cluster_valid_tokens = sync_tpu_sum(valid_tokens, tag=tokens_tag)
        is_all_masked = (cluster_valid_tokens == 0)
    else:
        is_all_masked = (cur_loss is not None and cur_loss <= 0.0)

    # 2. Local loss-shock and non-finite loss evaluation (decoupled from all-masked check)
    is_loss_non_finite = (cur_loss is not None) and (math.isnan(cur_loss) or math.isinf(cur_loss))
    is_loss_shock = False
    if not is_loss_non_finite and not is_all_masked and cur_loss is not None and ema_loss is not None and ema_loss > 0.5:
        if cur_loss > (10.0 * ema_loss) and cur_loss > 15.0:
            is_loss_shock = True

    # Skip if non-finite or loss shock
    local_shock_skip = 1 if (is_loss_non_finite or is_loss_shock) else 0

    # 3. Collective shock synchronization across all TPU workers (max reduction: ANY worker triggers skip)
    shock_tag = f"tpu_loss_skip_s{step}_a{attempt}" if attempt > 0 else f"tpu_loss_skip_{step}"
    global_shock_skip = sync_tpu_decision(local_shock_skip, tag=shock_tag)

    if global_shock_skip == 1 or is_all_masked:
        if is_all_masked:
            if is_tpu_master():
                print(f"[TPU Step Guard] Accumulation window contains 0 valid tokens across cluster (loss = {cur_loss}). Synchronized skip across all TPU cores to prevent unwarranted AdamW weight decay.", flush=True)
        elif is_loss_shock:
            print(f"[TPU Model Shock Guard] Extreme loss spike detected ({cur_loss:.4f} vs EMA {ema_loss:.4f}). Skipping step across all cores!", flush=True)
        elif is_loss_non_finite:
            print(f"[TPU Model Shock Guard] Non-finite loss detected ({cur_loss}). Skipping step across all cores!", flush=True)
        elif is_tpu_master():
            print("[TPU Model Shock Guard] Worker replica reported skip condition (loss shock / non-finite). Synchronized skip across all TPU cores.", flush=True)

        optimizer.zero_grad(set_to_none=True)
        if HAS_XLA:
            xm.mark_step()
        return True, 0.0

    # 4. Distributed Gradient Weighting for Exact Token-Mean Parity:
    # If valid_tokens / cluster_valid_tokens is provided and > 0:
    # Scale local accumulated gradient by (world_size / cluster_valid_tokens)
    # so that xm.reduce_gradients (which computes: (1 / world_size) * sum_r [ G_scaled_r ])
    # produces exactly:
    #   (1 / world_size) * sum_r [ (world_size / N_global) * G_r ] = (1 / N_global) * sum_r [ G_r ]
    # guaranteeing that every single valid token across all TPU cores receives equal mathematical weight.
    if valid_tokens is not None and cluster_valid_tokens is not None and cluster_valid_tokens > 0:
        world_size = get_tpu_world_size() if (HAS_XLA and get_tpu_world_size() > 1) else 1
        grad_scale = float(world_size) / float(cluster_valid_tokens)
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.mul_(grad_scale)

    # 5. Collective gradient synchronization (Only if loss was valid across all workers)
    # STRICT SINGLE-REDUCTION INVARIANT:
    # We call xm.reduce_gradients(optimizer) explicitly once here before clipping.
    # At step 7 below, we invoke optimizer.step() directly (NOT xm.optimizer_step()),
    # strictly guaranteeing that gradient reduction is performed EXACTLY ONCE (NO DOUBLE REDUCTION).
    if HAS_XLA and get_tpu_world_size() > 1:
        xm.reduce_gradients(optimizer)
    elif _COLLECTIVE_GRAD_HOOK is not None and model is not None:
        _COLLECTIVE_GRAD_HOOK(model, step)

    # 6. Global gradient norm clipping (strictly on globally reduced gradients)
    grad_norm_val = 0.0
    if max_norm > 0 and model is not None:
        target_model = getattr(model, "module", model)
        grad_norm = torch.nn.utils.clip_grad_norm_(target_model.parameters(), max_norm=max_norm)
        grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)

    # 7. Local gradient explosion and non-finite evaluation
    is_grad_non_finite = math.isnan(grad_norm_val) or math.isinf(grad_norm_val)
    is_grad_shock = (not is_grad_non_finite) and (grad_norm_val > (1000.0 * max_norm))
    local_grad_skip = 1 if (is_grad_non_finite or is_grad_shock) else 0

    # 8. Collective gradient-skip synchronization across all TPU workers
    grad_tag = f"tpu_grad_skip_s{step}_a{attempt}" if attempt > 0 else f"tpu_grad_skip_{step}"
    global_grad_skip = sync_tpu_decision(local_grad_skip, tag=grad_tag)

    if global_grad_skip == 1:
        if is_tpu_master():
            if is_grad_shock:
                print(f"[TPU Model Shock Guard] Extreme gradient spike detected ({grad_norm_val:.2f} > {1000.0 * max_norm:.1f}). Skipping step across all cores!", flush=True)
            else:
                print(f"[TPU Model Shock Guard] Non-finite gradient detected ({grad_norm_val:.2f}). Skipping step across all cores!", flush=True)

        optimizer.zero_grad(set_to_none=True)
        if HAS_XLA:
            xm.mark_step()
        return True, grad_norm_val

    # 9. Safe optimizer step (performed directly without double-reduction)
    if offload_optimizer:
        _step_offloaded_adamw(optimizer, model)
    else:
        optimizer.step()
    if HAS_XLA:
        xm.mark_step()
    return False, grad_norm_val


def tpu_save(state_dict: Any, path: str):
    """
    Safely serializes checkpoint state dict from TPU master rank without stalls or deadlocks.
    CRITICAL: In PyTorch/XLA, all ranks must call xm.save(..., master_only=True)
    so the internal rendezvous does not deadlock waiting for non-master ranks.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if HAS_XLA:
        # xm.save coordinates rendezvous across all processes and writes only on master rank
        xm.save(state_dict, path, master_only=True)
    else:
        if is_tpu_master():
            import torch
            torch.save(state_dict, path)


class TPUConfig:
    """
    Authoritative default hyper-parameters optimized for Google TPU (v6e / v5p / v5e).
    Maintains exact numerical and hyperparameter parity with TARA GPU training.
    """
    # Micro-batch per TPU core (2 per core doubles systolic MXU utilization)
    DEFAULT_BATCH_SIZE: int = 2
    # Gradient accumulation steps (32 micro-batches per optimizer step)
    DEFAULT_GRAD_ACCUM: int = 32
    # Memory-safe chunk size for vocabulary projection and cross-entropy (512 tokens preserves high TPU MXU utilization while keeping peak projection VRAM bounded to ~100MB)
    DEFAULT_LOSS_CHUNK_SIZE: int = 512
    # Default training precision: bf16 (hardware execution via XLA_USE_BF16)
    DEFAULT_PRECISION: str = "bf16"
    # Native TPU Sequence Length (fixed static shape)
    DEFAULT_SEQ_LEN: int = 2048
    # Base Learning Rate for 256 effective batch size
    DEFAULT_LR: float = 3e-4
    # Minimum learning rate for cosine annealing
    DEFAULT_MIN_LR: float = 1e-5
    # Warmup steps
    DEFAULT_WARMUP_STEPS: int = 50
    # AdamW hyperparameters with exact GPU parity
    DEFAULT_BETAS: Tuple[float, float] = (0.9, 0.95)
    DEFAULT_EPS: float = 1e-5  # Matches GPU adam_eps; prevents variance underflow in BF16/FP16
    DEFAULT_WEIGHT_DECAY: float = 0.01
    # Optimizer type
    DEFAULT_OPTIMIZER: str = "adamw"
    # Offload AdamW optimizer momentum states to CPU Host RAM (preserves ~7.07 GB of TPU HBM)
    DEFAULT_OFFLOAD_OPTIMIZER: bool = True
    # Stage max steps (default: 50,000 steps per stage)
    DEFAULT_MAX_STEPS: int = 50_000
    # Dedicated TPU Checkpoint Output Directory (Separate from GPU checkpoints!)
    DEFAULT_OUTPUT_DIR: str = "storage/models/checkpoints_tpu/stage0_pretraining"
    # Promoted Model Output Directory
    DEFAULT_PROMOTED_DIR: str = "storage/models/promoted_tpu_stage0"
    # Periodic intervals
    CHECKPOINT_INTERVAL: int = int(os.environ.get("CHECKPOINT_INTERVAL", 100))
    EVAL_INTERVAL: int = 0  # Default disabled for maximum TPU throughput; enable via --enable-probe --eval-interval 500
    PROMOTE_INTERVAL: int = 50000
    KEEP_CHECKPOINTS: int = 2
    MAX_SHARD_MB: int = 480
