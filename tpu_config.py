"""
TARA Autonomous TPU Engine Configuration & Hardware Abstraction Layer
=============================================================================
Provides seamless detection, initialization, and hardware-optimized configuration
for Google TPU v3-8 (and TPU v2-8 / Cloud TPU VM / Kaggle / Colab environments).

Designed to be 100% decoupled from existing GPU scripts (train_main.py, run_train.sh).
"""

import os
import sys
import logging
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger("tara_tpu_config")

# Safe PyTorch / XLA Import Handling
HAS_XLA = False
xm = None
pl = None
xmp = None

try:
    import torch_xla
    import torch_xla.core.xla_model as _xm
    import torch_xla.distributed.parallel_loader as _pl
    import torch_xla.distributed.xla_multiprocessing as _xmp
    xm = _xm
    pl = _pl
    xmp = _xmp
    HAS_XLA = True
except (ImportError, Exception):
    HAS_XLA = False


def is_tpu_available() -> bool:
    """Returns True if PyTorch/XLA is installed and a TPU accelerator is present."""
    if not HAS_XLA:
        return False
    try:
        device_hw = xm.xla_device_hw()
        return "TPU" in str(device_hw).upper()
    except Exception:
        # Check environment markers common in Colab / Kaggle / GCP TPU VMs
        return bool(
            os.environ.get("TPU_NAME")
            or os.environ.get("COLAB_TPU_ADDR")
            or os.environ.get("TPU_ACCELERATOR_TYPE")
            or (os.environ.get("PJRT_DEVICE") == "TPU")
        )


def get_tpu_hardware_info() -> Dict[str, Any]:
    """Returns detailed TPU hardware topology and core configuration."""
    available = is_tpu_available()
    hw_type = "Unknown"
    num_cores = 1
    device_name = "N/A"

    if HAS_XLA:
        try:
            device_name = str(xm.xla_device_hw())
            num_cores = xm.xrt_world_size() if hasattr(xm, "xrt_world_size") else 8
        except Exception:
            pass

    # Detect TPU generation from environment or hardware string
    tpu_env = (
        os.environ.get("TPU_ACCELERATOR_TYPE")
        or os.environ.get("TPU_NAME", "")
        or device_name
    ).lower()

    if "v3-8" in tpu_env or "v3" in tpu_env:
        hw_type = "Google TPU v3-8 (128GB HBM | 8 Cores | 420 TFLOPS)"
        num_cores = 8
    elif "v2-8" in tpu_env or "v2" in tpu_env:
        hw_type = "Google TPU v2-8 (64GB HBM | 8 Cores | 180 TFLOPS)"
        num_cores = 8
    elif "v4" in tpu_env:
        hw_type = "Google TPU v4 (8 Cores | 275 TFLOPS/chip)"
        num_cores = 8
    elif available:
        hw_type = f"Google TPU ({device_name})"
        num_cores = 8

    return {
        "available": available,
        "hardware_type": hw_type,
        "num_cores": num_cores,
        "has_xla": HAS_XLA,
        "device_name": device_name,
    }


def get_tpu_device():
    """Acquires the authentic PyTorch/XLA device bound to the current process."""
    if HAS_XLA:
        return xm.xla_device()
    import torch
    return torch.device("cpu")


def get_tpu_rank() -> int:
    """Returns the process ordinal / rank (0..7 on TPU v3-8)."""
    if HAS_XLA:
        try:
            return xm.get_ordinal()
        except Exception:
            return 0
    return 0


def get_tpu_world_size() -> int:
    """Returns total TPU workers (typically 8 on TPU v3-8)."""
    if HAS_XLA:
        try:
            return xm.xrt_world_size()
        except Exception:
            return 1
    return 1


def is_tpu_master() -> bool:
    """Returns True if the current process is the TPU master process (rank 0)."""
    return get_tpu_rank() == 0


def tpu_barrier(tag: str = "tpu_barrier"):
    """Synchronizes all TPU worker cores at a common rendezvous point."""
    if HAS_XLA:
        try:
            xm.rendezvous(tag)
        except Exception:
            pass


def tpu_optimizer_step(optimizer, barrier: bool = True):
    """Executes a TPU-optimized all-reduce optimizer step."""
    if HAS_XLA:
        xm.optimizer_step(optimizer, barrier=barrier)
        xm.mark_step()
    else:
        optimizer.step()


def tpu_save(state_dict: Any, path: str):
    """Safely serializes checkpoint state dict from TPU master rank without stalls."""
    if not is_tpu_master():
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if HAS_XLA:
        xm.save(state_dict, path)
    else:
        import torch
        torch.save(state_dict, path)


class TPUConfig:
    """
    Authoritative default hyper-parameters optimized for Google TPU v3-8.
    TPU v3-8 has 16GB HBM per core (128GB total), supporting bfloat16 natively.
    """
    # Micro-batch per TPU core (x8 cores = 32 samples per micro-step)
    DEFAULT_BATCH_SIZE: int = 4
    # Gradient accumulation steps (32 * 8 = 256 effective batch size)
    DEFAULT_GRAD_ACCUM: int = 8
    # Native TPU Sequence Length
    DEFAULT_SEQ_LEN: int = 2048
    # Base Learning Rate for 256 effective batch size (Linear Scaling Rule)
    DEFAULT_LR: float = 3e-4
    # Minimum learning rate for cosine annealing
    DEFAULT_MIN_LR: float = 3e-5
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
    EVAL_INTERVAL: int = 100
    PROMOTE_INTERVAL: int = 1000
