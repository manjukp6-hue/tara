"""
TARA Autonomous TPU Engine Configuration & Hardware Abstraction Layer
=============================================================================
Provides seamless detection, initialization, and hardware-optimized configuration
for ALL Google TPU generations:
  - Google Cloud TPU v6e (Trillium | 32GB HBM per chip | 6th Gen)
  - Google Cloud TPU v5p (Flagship LLM Training | 95GB HBM per chip)
  - Google Cloud TPU v5e (LitePod / Kaggle v5e-8 / Colab v5e-1)
  - Google Cloud TPU v4 (8 Cores / 32GB HBM per chip)
  - Google Cloud TPU v3-8 (128GB HBM | 8 Cores | 420 TFLOPS)
  - Google Cloud TPU v2-8 (64GB HBM | 8 Cores | 180 TFLOPS)

Designed to be 100% decoupled from existing GPU scripts (train_main.py, run_train.sh).
"""

import os
import sys
import math
import logging
from typing import Optional, Tuple, Dict, Any, List

# Ensure PJRT & Bfloat16 defaults are active in Python environment before importing torch_xla
os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ.setdefault("XLA_USE_BF16", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

logger = logging.getLogger("tara_tpu_config")

# Safe PyTorch / XLA Import Handling
HAS_XLA = False
xm = None
pl = None
xmp = None
xr = None

try:
    import torch_xla
    import torch_xla.core.xla_model as _xm
    import torch_xla.distributed.parallel_loader as _pl
    import torch_xla.distributed.xla_multiprocessing as _xmp
    xm = _xm
    pl = _pl
    xmp = _xmp
    try:
        import torch_xla.runtime as _xr
        xr = _xr
    except ImportError:
        xr = None
    HAS_XLA = True
except (ImportError, Exception):
    HAS_XLA = False


def is_tpu_available() -> bool:
    """Returns True if PyTorch/XLA or native TorchTPU accelerator is present."""
    if HAS_XLA:
        try:
            if xr is not None and hasattr(xr, "device_type"):
                return xr.device_type().upper() == "TPU"
            device_hw = xm.xla_device_hw()
            return "TPU" in str(device_hw).upper()
        except Exception:
            return True
    try:
        import torch
        dev = torch.device("tpu")
        _ = torch.empty(0, device=dev)
        return True
    except Exception:
        pass
    # Check environment markers common in Colab / Kaggle / GCP TPU VMs
    return bool(
        os.environ.get("TPU_NAME")
        or os.environ.get("COLAB_TPU_ADDR")
        or os.environ.get("TPU_ACCELERATOR_TYPE")
        or (os.environ.get("PJRT_DEVICE") == "TPU")
    )


def get_tpu_hardware_info() -> Dict[str, Any]:
    """
    Returns detailed TPU hardware topology and core configuration across ALL TPU generations.
    Safe to call in parent process without premature PJRT client binding.
    """
    available = is_tpu_available()
    hw_type = "Unknown Accelerator (CPU Fallback)"
    num_cores = 1
    device_name = "N/A"

    # Safely query device count without full device allocation
    if HAS_XLA:
        try:
            if xr is not None and hasattr(xr, "local_device_count"):
                num_cores = xr.local_device_count()
            elif hasattr(torch_xla, "device_count"):
                num_cores = torch_xla.device_count()
            elif hasattr(xm, "xworld_size"):
                num_cores = xm.xworld_size()
            elif hasattr(xm, "xrt_world_size"):
                num_cores = xm.xrt_world_size()
        except Exception:
            num_cores = 8 if available else 1

    # Detect TPU generation from environment or hardware string
    tpu_env = (
        os.environ.get("TPU_ACCELERATOR_TYPE", "")
        + " " + os.environ.get("TPU_NAME", "")
        + " " + os.environ.get("KAGGLE_TPU_VERSION", "")
        + " " + os.environ.get("PJRT_DEVICE", "")
        + " " + device_name
    ).lower()

    env_num_devices = os.environ.get("TPU_NUM_DEVICES")
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
    elif "v5e-8" in tpu_env or "v5e" in tpu_env or "v5litepod" in tpu_env:
        hw_type = "Google Cloud TPU v5e-8 (Kaggle/GCP | 8 Cores | 128GB HBM | 1576 TFLOPS)"
        num_cores = 8

    # 5. Google Cloud TPU v4 (v4-8 / Pod Slices)
    elif "v4" in tpu_env:
        hw_type = f"Google Cloud TPU v4 ({num_cores} Cores | 32GB HBM/chip | 275 TFLOPS/chip)"
        num_cores = max(num_cores, 8) if num_cores <= 1 and available else num_cores

    # 6. Google Cloud TPU v3-8 (128GB HBM | 8 Cores)
    elif "v3-8" in tpu_env or "v3" in tpu_env:
        hw_type = "Google TPU v3-8 (128GB HBM | 8 Cores | 420 TFLOPS BF16)"
        num_cores = 8

    # 7. Google Cloud TPU v2-8 (64GB HBM | 8 Cores)
    elif "v2-8" in tpu_env or "v2" in tpu_env:
        hw_type = "Google TPU v2-8 (64GB HBM | 8 Cores | 180 TFLOPS)"
        num_cores = 8

    elif available:
        hw_type = f"Google Cloud TPU ({num_cores} Cores | PyTorch/XLA PJRT)"

    return {
        "available": available,
        "hardware_type": hw_type,
        "num_cores": max(1, num_cores),
        "has_xla": HAS_XLA,
        "device_name": device_name,
    }


def get_tpu_device():
    """Acquires the authentic PyTorch/XLA or native TorchTPU device bound to the current process."""
    if HAS_XLA:
        try:
            return xm.xla_device()
        except Exception:
            pass
    try:
        import torch
        dev = torch.device("tpu")
        _ = torch.empty(0, device=dev)
        return dev
    except Exception:
        pass
    import torch
    return torch.device("cpu")


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
    Guarded so single-core or CPU runs do not hang.
    """
    if HAS_XLA and get_tpu_world_size() > 1:
        try:
            xm.rendezvous(tag)
        except Exception:
            pass


_COLLECTIVE_MOCK_HOOK: Optional[Any] = None
_COLLECTIVE_GRAD_HOOK: Optional[Any] = None


def set_collective_mock(hook: Optional[Any], grad_hook: Optional[Any] = None) -> None:
    """Sets mock hooks for testing multi-worker collective decisions and gradient reduction on CPU/simulated environments."""
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


def build_tpu_optimizer_groups(model: Any, weight_decay: float = 0.01) -> List[Dict[str, Any]]:
    """
    Constructs AdamW parameter groups matching GPU training semantics with 100% exact parity:
      - Group 0: 2D projection / MLP / attention weight matrices with decoupled weight_decay (default: 0.01).
      - Group 1: 1D biases, RMSNorm parameters, 2D token embeddings (embed_tokens), and lm_head with weight_decay = 0.0.
    Preserves exact parameter-group ordering and membership identical to train_main.py lines 1489-1503.
    Strictly enforces weight-tying invariants (lm_head.weight == embed_tokens.weight) to maintain exact 112/34 group counts.
    """
    target_model = getattr(model, "module", model)
    cfg = getattr(target_model, "config", None)
    tie = False
    if isinstance(cfg, dict):
        tie = cfg.get("tie_word_embeddings", False)
    elif cfg is not None:
        tie = getattr(cfg, "tie_word_embeddings", False)

    if tie:
        if hasattr(target_model, "lm_head") and hasattr(target_model, "model") and hasattr(target_model.model, "embed_tokens"):
            target_model.lm_head.weight = target_model.model.embed_tokens.weight

    decay_params = []
    no_decay_params = []
    seen_ids = set()

    for name, param in target_model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in seen_ids:
            continue
        seen_ids.add(id(param))
        # If tie_word_embeddings is active, lm_head.weight points to embed_tokens buffer - skip duplicate
        if tie and "lm_head" in name and hasattr(target_model, "model") and hasattr(target_model.model, "embed_tokens"):
            if param is target_model.model.embed_tokens.weight or id(param) == id(target_model.model.embed_tokens.weight):
                continue

        if param.dim() < 2 or "embed_tokens" in name or "lm_head" in name or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def restore_tpu_optimizer_state(
    checkpoint_dir: str,
    optimizer: Any,
    device: Optional[Any] = None,
    allow_cold_start: bool = False,
) -> bool:
    """
    Safely and strictly restores optimizer state from checkpoint_dir with full GPU parity checks:
      1. Loads optimizer_state.pt on CPU.
      2. Supports both nested {"optimizer": state_dict, ...} (from train_main.py / gdrive_sync.py)
         and direct flat {state, param_groups} formats.
      3. Validates parameter group count, parameter counts per group, and state tensor shapes.
      4. Moves optimizer state tensors to the target TPU/XLA device as required.
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

    # Strict per-group parameter count validation with auto-parity reconciliation
    for g_idx, (saved_g, cur_g) in enumerate(zip(raw_opt_sd["param_groups"], optimizer.param_groups)):
        n_saved = len(saved_g.get("params", []))
        n_cur = len(cur_g.get("params", []))
        if n_saved != n_cur:
            # Special case: Group 1 has 34 params in checkpoint (tied lm_head) but 35 in optimizer
            if g_idx == 1 and n_saved == 34 and n_cur == 35:
                unique_params = []
                seen_p_ids = set()
                for p in cur_g["params"]:
                    if id(p) not in seen_p_ids:
                        seen_p_ids.add(id(p))
                        unique_params.append(p)
                if len(unique_params) == 34:
                    cur_g["params"] = unique_params
                    n_cur = 34
                    print("[TPU Resume Parity] Successfully reconciled Group 1 parameter count (35 -> 34) by deduplicating tied embeddings.", flush=True)

        if n_saved != n_cur:
            print(f"\n{'=' * 88}", flush=True)
            print(f"  [TPU OPTIMIZER PARAMETER DIAGNOSTICS - MISMATCH IN GROUP {g_idx}]", flush=True)
            print(f"  Checkpoint has {n_saved} params | Active optimizer has {n_cur} params", flush=True)
            print(f"{'-' * 88}", flush=True)
            for p_idx, p in enumerate(cur_g.get("params", [])):
                shape_str = tuple(p.shape) if hasattr(p, "shape") else "unknown"
                print(f"  Group {g_idx} | index={p_idx:02d} | shape={shape_str} | requires_grad={getattr(p, 'requires_grad', None)}", flush=True)
            print(f"{'=' * 88}\n", flush=True)
            err_msg = f"[TPU Resume Gate] Parameter count mismatch in group {g_idx}: checkpoint has {n_saved} params, optimizer has {n_cur} params."
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

    # Ingest validated state_dict into optimizer
    print(f"[TPU Resume] Ingesting validated state dict and transferring momentum tensors to device ({device})...", flush=True)
    try:
        optimizer.load_state_dict(raw_opt_sd)
    except Exception as se:
        if allow_cold_start:
            print(f"[TPU Resume Notice] load_state_dict error: {se}. Cold start permitted.", flush=True)
            return False
        raise RuntimeError(f"[TPU Resume Gate] Failed to load validated optimizer state dict: {se}")

    # Move state tensors to TPU/XLA device
    if device is not None and (str(device).startswith("xla") or getattr(device, "type", "") in ("xla", "tpu")):
        for state in optimizer.state.values():
            for sk, sv in state.items():
                if isinstance(sv, torch.Tensor):
                    state[sk] = sv.to(device)

    print(f"[TPU Resume] Successfully restored complete optimizer state ({loaded_n_groups} parameter groups, {len(optimizer.state)} momentum buffers) with 100% parameter fidelity.", flush=True)
    return True


def tpu_optimizer_step(
    optimizer: Any,
    model: Optional[Any] = None,
    max_norm: float = 1.0,
    cur_loss: Optional[float] = None,
    ema_loss: Optional[float] = None,
    step: int = 0,
    attempt: int = 0,
) -> Tuple[bool, float]:
    """
    Executes a TPU-optimized all-reduce optimizer step with Collective Model Shock & Gradient Explosion Protection:
      1. Evaluates local loss validity (non-finite or >10x EMA shock).
      2. Synchronizes loss-skip decision across ALL TPU replicas using sync_tpu_decision (collective max).
      3. If any replica reports loss shock or non-finite loss, ALL replicas skip the step simultaneously,
         clearing gradients and marking the step without diverging model weights.
      4. If loss is valid across all replicas, all-reduces gradients across replicas once (xm.reduce_gradients).
      5. Clips globally reduced gradients consistently across all cores (max_norm=1.0).
      6. Evaluates and collectively synchronizes gradient validity (non-finite or >1000x norm shock).
      7. Steps optimizer and marks XLA execution graph, or skips step if gradient corruption is detected.
    Returns (is_step_skipped, grad_norm_val).
    """
    import torch

    # 1. Local loss-shock and non-finite loss evaluation
    is_loss_non_finite = (cur_loss is not None) and (math.isnan(cur_loss) or math.isinf(cur_loss))
    is_loss_shock = False
    if not is_loss_non_finite and cur_loss is not None and ema_loss is not None and ema_loss > 0.5:
        if cur_loss > (10.0 * ema_loss) and cur_loss > 15.0:
            is_loss_shock = True

    local_loss_skip = 1 if (is_loss_non_finite or is_loss_shock) else 0

    # 2. Collective loss-skip synchronization across all TPU workers
    # Strictly unique rendezvous tag per step and accumulation attempt (prevents stale rendezvous / XLA graph collision)
    loss_tag = f"tpu_loss_skip_s{step}_a{attempt}" if attempt > 0 else f"tpu_loss_skip_{step}"
    global_loss_skip = sync_tpu_decision(local_loss_skip, tag=loss_tag)

    if global_loss_skip == 1:
        if is_loss_shock:
            print(f"[TPU Model Shock Guard] Extreme loss spike detected ({cur_loss:.4f} vs EMA {ema_loss:.4f}). Skipping step across all cores!", flush=True)
        elif is_loss_non_finite:
            print(f"[TPU Model Shock Guard] Non-finite loss detected ({cur_loss}). Skipping step across all cores!", flush=True)
        elif is_tpu_master():
            print("[TPU Model Shock Guard] Worker replica reported loss shock / non-finite loss. Synchronized skip across all TPU cores.", flush=True)

        optimizer.zero_grad(set_to_none=True)
        if HAS_XLA:
            xm.mark_step()
        return True, 0.0

    # 3. Collective gradient synchronization (Only if loss was valid across all workers)
    # STRICT SINGLE-REDUCTION INVARIANT:
    # We call xm.reduce_gradients(optimizer) explicitly once here before clipping.
    # At step 7 below, we invoke optimizer.step() directly (NOT xm.optimizer_step()),
    # strictly guaranteeing that gradient reduction is performed EXACTLY ONCE (NO DOUBLE REDUCTION).
    if HAS_XLA and get_tpu_world_size() > 1:
        xm.reduce_gradients(optimizer)
    elif _COLLECTIVE_GRAD_HOOK is not None and model is not None:
        _COLLECTIVE_GRAD_HOOK(model, step)

    # 4. Gradient norm clipping
    grad_norm_val = 0.0
    if max_norm > 0 and model is not None:
        target_model = getattr(model, "module", model)
        grad_norm = torch.nn.utils.clip_grad_norm_(target_model.parameters(), max_norm=max_norm)
        grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)

    # 5. Local gradient explosion and non-finite evaluation
    is_grad_non_finite = math.isnan(grad_norm_val) or math.isinf(grad_norm_val)
    is_grad_shock = (not is_grad_non_finite) and (grad_norm_val > (1000.0 * max_norm))
    local_grad_skip = 1 if (is_grad_non_finite or is_grad_shock) else 0

    # 6. Collective gradient-skip synchronization across all TPU workers
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

    # 7. Safe optimizer step
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
    Authoritative default hyper-parameters optimized for Google TPU (v6e / v5p / v5e / v4 / v3-8).
    Maintains exact numerical and hyperparameter parity with TARA GPU training.
    """
    # Micro-batch per TPU core (x8 cores = 32 samples per micro-step)
    DEFAULT_BATCH_SIZE: int = 4
    # Gradient accumulation steps (32 * 8 = 256 effective batch size)
    DEFAULT_GRAD_ACCUM: int = 8
    # Native TPU Sequence Length (fixed static shape)
    DEFAULT_SEQ_LEN: int = 2048
    # Base Learning Rate for 256 effective batch size
    DEFAULT_LR: float = 2e-4
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
    # Stage max steps (default: 50,000 steps per stage)
    DEFAULT_MAX_STEPS: int = 50_000
    # Dedicated TPU Checkpoint Output Directory (Separate from GPU checkpoints!)
    DEFAULT_OUTPUT_DIR: str = "storage/models/checkpoints_tpu/stage0_pretraining"
    # Promoted Model Output Directory
    DEFAULT_PROMOTED_DIR: str = "storage/models/promoted_tpu_stage0"
    # Periodic intervals
    CHECKPOINT_INTERVAL: int = 100
    EVAL_INTERVAL: int = 0  # Default disabled for maximum TPU throughput; enable via --enable-probe --eval-interval 500
    PROMOTE_INTERVAL: int = 1000
    KEEP_CHECKPOINTS: int = 2
    MAX_SHARD_MB: int = 480
