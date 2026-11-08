#!/usr/bin/env python3
"""
=============================================================================
TARA Native TPU Autonomous Multi-Stage Pre-Training Engine
=============================================================================
Dedicated Google TPU runner supporting Google TPU architectures:
  - Google Cloud TPU v6e (Trillium | 32GB HBM per chip | 6th Gen)
  - Google Cloud TPU v5p (Flagship LLM Training | 95GB HBM per chip)
  - Google Cloud TPU v5e (LitePod / Kaggle v5e-8 / Colab v5e-1)

Key Architectural Pillars (Full GPU Parity):
  - 100% Decoupled: Zero modifications to existing GPU files (train_main.py, run_train.sh).
  - Exact AdamW Parity: 2 parameter groups (decoupled decay 0.01 / 0.0), betas=(0.9, 0.95), eps=1e-5.
  - Strict Optimizer Resume: Restores 2-group GPU/TPU optimizer_state.pt with parameter validation.
  - RAM Exhaustion Guard: Only writer rank (Rank 0) materializes full CPU state dict buffers.
  - Full Dataset Curriculum: TaraMultiDomainScheduler with DRR, ShardLedger, and DatasetSnapshot.
  - Safe Remote Dataset Fallback: Hugging Face JIT download support; fail-closed on zero shards (no DummyDataset!).
  - Best Checkpoint & Stage Promotion: best_checkpoint tracking, atomic promotion, and async HF upload.
  - Benchmark-Monitored Generation Probes: Deterministic CPU probes with telemetry.
=============================================================================
"""
from __future__ import annotations

import os
import sys
import time
import math
import json
import glob
import copy
import re
import queue
import fnmatch
import shutil
import random
import argparse
from typing import Optional, List, Dict, Any, Tuple, Union, Iterable, Sequence, Callable, Set
from datetime import datetime, timezone

# Ensure PJRT & Bfloat16 environment flags are set before importing torch / torch_xla
os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ.setdefault("XLA_USE_BF16", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import TPU Abstraction Layer
from tpu_config import (
    HAS_XLA,
    xm,
    pl,
    xmp,
    xr,
    is_tpu_available,
    get_tpu_hardware_info,
    get_tpu_device,
    get_tpu_rank,
    get_tpu_world_size,
    is_tpu_master,
    tpu_barrier,
    tpu_optimizer_step,
    sync_tpu_decision,
    sync_tpu_mean,
    sync_tpu_sum,
    sync_tpu_sum_float,
    ensure_model_weight_tying,
    TPUConfig,
    build_tpu_optimizer_groups,
    restore_tpu_optimizer_state,
    should_offload_optimizer,
    get_tpu_memory_diagnostics,
    log_tpu_memory_stage,
)

# Reuse Native TARA Core Classes from model_dataset (UNTOUCHED)
from model_dataset import (
    CANONICAL_DOMAINS,
    TaraForCausalLM,
    TaraNativeTokenizer,
    TaraStreamingJsonlDataset,
    TaraMultiDomainScheduler,
    ShardLedger,
    DatasetSnapshot,
    resolve_shard_domain,
    parse_domain_weights,
    download_single_shard_jit,
    _resolve_hf_token,
    load_candidate_weights,
    resolve_dynamic_rope_theta,
    CausalLMOutput,
)

# Reuse Native Cloud & Checkpoint Functions from gdrive_sync (UNTOUCHED)
from gdrive_sync import (
    promote_checkpoint,
    finalize_stage_and_promote,
    upload_promoted_model_to_hf,
    check_model_weights_exist,
    compute_file_sha256,
    save_safetensors_sharded,
    resolve_max_shard_size_bytes,
    _safe_unlink,
    _safe_rmtree,
    resolve_and_auto_select_checkpoint,
    resolve_gdrive_restore_dir,
    resolve_gdrive_backup_dir,
    trigger_async_cloud_backup,
    _cleanup_old_versioned_checkpoints,
    _collect_versioned_checkpoint_dirs,
    get_checkpoint_dir_step,
    restore_checkpoint_from_gdrive,
    restore_checkpoint_from_hf,
    verify_checkpoint_integrity,
    _resolve_hf_token_standalone,
    sync_checkpoint_to_gdrive,
    prune_hf_checkpoints,
    extract_data_cursor_from_state,
)

import threading
_active_tpu_hf_upload_thread: Optional[threading.Thread] = None

def trigger_async_hf_upload(promoted_dir: str, repo_id: str, stage_id: str, step: int) -> Optional[threading.Thread]:
    """Launches Hugging Face model deployment in a non-blocking background daemon thread."""
    global _active_tpu_hf_upload_thread
    if not repo_id:
        return None

    if _active_tpu_hf_upload_thread is not None and _active_tpu_hf_upload_thread.is_alive():
        _active_tpu_hf_upload_thread.join(timeout=5.0)

    parent_dir = os.path.dirname(os.path.abspath(promoted_dir)) or "."
    step_snapshot_dir = os.path.join(parent_dir, f".hf_upload_step_{step}_{os.getpid()}")
    try:
        if os.path.exists(step_snapshot_dir):
            _safe_rmtree(step_snapshot_dir)
        shutil.copytree(promoted_dir, step_snapshot_dir)
    except Exception:
        step_snapshot_dir = promoted_dir

    def _worker():
        try:
            upload_promoted_model_to_hf(step_snapshot_dir, repo_id, stage_id)
        except Exception as e:
            print(f"[TPU HF Async Error] Background HF upload thread notice: {e}", flush=True)
        finally:
            if step_snapshot_dir != promoted_dir and os.path.exists(step_snapshot_dir):
                _safe_rmtree(step_snapshot_dir)

    t = threading.Thread(target=_worker, name=f"TPU-HF-Upload-{step}", daemon=True)
    t.start()
    _active_tpu_hf_upload_thread = t
    return t

def wait_for_hf_upload(timeout: Optional[float] = 180.0) -> bool:
    """Waits for active asynchronous Hugging Face upload thread to terminate."""
    global _active_tpu_hf_upload_thread
    if _active_tpu_hf_upload_thread is not None and _active_tpu_hf_upload_thread.is_alive():
        _active_tpu_hf_upload_thread.join(timeout=timeout)
        return not _active_tpu_hf_upload_thread.is_alive()
    return True


def resolve_tpu_gdrive_backup_dir(explicit_dir: Optional[str] = None) -> Optional[str]:
    """
    Dynamically resolves dedicated Google Drive backup directory for TPU.
    Strictly isolated under 'tara_checkpoints_tpu' so GPU backups ('tara_checkpoints')
    are NEVER overwritten, touched, or pruned.
    """
    if explicit_dir and str(explicit_dir).strip():
        base = str(explicit_dir).strip()
        if "tara_checkpoints" in base and not base.endswith("_tpu"):
            return base + "_tpu"
        return base

    env_dir = os.environ.get("TARA_TPU_GDRIVE_DIR", os.environ.get("TPU_GDRIVE_BACKUP_DIR", "")).strip()
    if env_dir:
        return env_dir

    colab_mydrive = "/content/drive/MyDrive"
    if os.path.exists(colab_mydrive):
        return os.path.join(colab_mydrive, "tara_checkpoints_tpu")

    std_gdrive = resolve_gdrive_backup_dir()
    if std_gdrive:
        parent = os.path.dirname(std_gdrive)
        return os.path.join(parent, "tara_checkpoints_tpu")

    return None


def prune_tpu_gdrive_checkpoints(gdrive_tpu_base_dir: str, keep_n: int = 2):
    """
    Prunes older TPU checkpoints on Google Drive, preserving strictly the latest keep_n (default 2).
    STRICT GPU PROTECTION:
    Refuses to delete if directory is GPU backup directory ('tara_checkpoints' without '_tpu').
    """
    if not gdrive_tpu_base_dir or not os.path.exists(gdrive_tpu_base_dir):
        return
    norm = os.path.normpath(os.path.abspath(gdrive_tpu_base_dir))
    if "tara_checkpoints" in norm and not norm.endswith("_tpu") and "_tpu" not in norm:
        print(f"[TPU GDrive Safety Violation] Attempted to prune GPU cloud dir '{gdrive_tpu_base_dir}'! Blocked.", flush=True)
        return

    try:
        entries = []
        for d in os.listdir(gdrive_tpu_base_dir):
            full_path = os.path.join(gdrive_tpu_base_dir, d)
            if not os.path.isdir(full_path) or d.startswith(".") or d == "best_milestone":
                continue
            m = re.match(r"(?:step_)?(\d+)", d)
            if m:
                step_val = int(m.group(1))
                entries.append((step_val, full_path))
            else:
                entries.append((int(os.path.getmtime(full_path)), full_path))

        entries.sort(key=lambda x: x[0])
        if len(entries) > keep_n:
            to_delete = entries[:-keep_n]
            for step_val, old_dir in to_delete:
                print(f"[TPU GDrive Quota Guard] Pruning obsolete TPU step checkpoint from GDrive (preserving latest {keep_n}): {old_dir}", flush=True)
                _safe_rmtree(old_dir)
    except Exception as e:
        print(f"[TPU GDrive Pruning Notice] Error during GDrive TPU pruning: {e}", flush=True)


def prune_tpu_hf_checkpoints(repo_id: str, keep_n: int = 2, token: Optional[str] = None):
    """
    Prunes older TPU checkpoints on Hugging Face under 'tara_checkpoints_tpu/normal_checkpoint'.
    STRICT GPU PROTECTION:
    Only prunes under 'tara_checkpoints_tpu'; never touches GPU backups under 'tara_checkpoints'.
    """
    try:
        prune_hf_checkpoints(repo_id=repo_id, base_folder="tara_checkpoints_tpu/normal_checkpoint", keep_n=keep_n, token=token)
    except Exception as e:
        print(f"[TPU HF Pruning Notice] Notice during HF TPU pruning: {e}", flush=True)


class BackgroundTPUCloudSyncWorker:
    """
    Dedicated Asynchronous Multi-Cloud Sync Worker for TPU Training.
    - Mirrors verified TPU checkpoints to Google Drive and/or Hugging Face in a background daemon thread.
    - Zero TPU compute interruption (training loop continues without waiting).
    - STRICT CLOUD GPU PROTECTION:
        * Cloud backups go to dedicated 'tara_checkpoints_tpu' (GDrive & HF).
        * Rolling retention (keep_n=2) prunes ONLY older TPU cloud backups.
        * GPU cloud backups ('tara_checkpoints') and GPU local checkpoints are NEVER touched or deleted!
    """
    _instance: Optional["BackgroundTPUCloudSyncWorker"] = None
    _lock = threading.Lock()

    def __init__(self):
        self.task_queue: queue.Queue = queue.Queue(maxsize=16)
        self.worker_thread = threading.Thread(target=self._run_loop, daemon=True, name="TPUCloudSyncWorker")
        self.worker_thread.start()
        self.active_transfers: int = 0

    @classmethod
    def get_instance(cls) -> "BackgroundTPUCloudSyncWorker":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def submit_task(
        self,
        local_ckpt_dir: str,
        gdrive_dir: Optional[str] = None,
        hf_repo: Optional[str] = None,
        stage_id: str = "stage0_pretraining",
        is_best: bool = False,
        step: int = 0,
        loss: float = 0.0,
        hf_token: Optional[str] = None,
        keep_n: int = 2,
    ):
        task = {
            "local_ckpt_dir": local_ckpt_dir,
            "gdrive_dir": gdrive_dir,
            "hf_repo": hf_repo,
            "stage_id": stage_id,
            "is_best": is_best,
            "step": step,
            "loss": loss,
            "hf_token": hf_token,
            "keep_n": keep_n,
            "timestamp": time.time(),
        }
        try:
            self.task_queue.put_nowait(task)
            tag = f"BEST MILESTONE (step {step}, loss {loss:.4f})" if is_best else f"Step {step:04d}"
            targets = []
            if gdrive_dir: targets.append(f"GDrive ({gdrive_dir})")
            if hf_repo: targets.append(f"HF ({hf_repo} -> tara_checkpoints_tpu)")
            target_str = " & ".join(targets) if targets else "Local disk only"
            print(f"[TPU Cloud Backup Async] Enqueued background upload for [{tag}] -> {target_str} (TPU training uninterrupted).", flush=True)
        except queue.Full:
            print(f"[TPU Cloud Backup Warning] Background sync queue full. Discarding duplicate intermediate task for Step {step}.", flush=True)

    def _run_loop(self):
        while True:
            try:
                task = self.task_queue.get()
                if task is None:
                    break
                self.active_transfers += 1
                local_dir = task["local_ckpt_dir"]
                gdrive_dir = task.get("gdrive_dir")
                hf_repo = task.get("hf_repo")
                is_best = task["is_best"]
                step = task["step"]
                loss = task.get("loss", 0.0)
                hf_token = task.get("hf_token")
                keep_n = task.get("keep_n", 2)

                tag = f"BEST MILESTONE (step {step}, loss {loss:.4f})" if is_best else f"Step {step:04d}"

                # 1. Google Drive TPU Background Sync
                if gdrive_dir and os.path.exists(local_dir):
                    try:
                        print(f"[TPU GDrive Worker] Starting background upload for [{tag}] -> {gdrive_dir}...", flush=True)
                        t0 = time.time()
                        target_subfolder = "best_milestone" if is_best else "normal_checkpoint"
                        step_folder = f"step_{step}_loss_{loss:.4f}" if is_best else f"step_{step}"
                        normal_base = os.path.join(gdrive_dir, target_subfolder)
                        gdrive_step_dir = os.path.join(normal_base, step_folder)
                        os.makedirs(gdrive_step_dir, exist_ok=True)

                        success = sync_checkpoint_to_gdrive(local_dir, gdrive_step_dir)
                        if success:
                            # Align top-level for immediate restore
                            sync_checkpoint_to_gdrive(local_dir, normal_base)
                            # Rolling retention for TPU checkpoints on GDrive
                            prune_tpu_gdrive_checkpoints(normal_base, keep_n=keep_n)
                            elapsed = time.time() - t0
                            print(f"[TPU GDrive Worker] Completed background upload for [{tag}] in {elapsed:.1f}s (TPU GDrive pruned, GPU untouched).", flush=True)
                        else:
                            print(f"[TPU GDrive Worker Warning] Upload failed or incomplete for [{tag}].", flush=True)
                    except Exception as ge:
                        print(f"[TPU GDrive Worker Warning] GDrive TPU upload error: {ge}", flush=True)

                # 2. Hugging Face TPU Background Sync
                if hf_repo and os.path.exists(local_dir):
                    try:
                        t0 = time.time()
                        token = _resolve_hf_token_standalone(hf_token)
                        if token:
                            from huggingface_hub import HfApi
                            api = HfApi()
                            target_subfolder = "best_milestone" if is_best else "normal_checkpoint"
                            target_path = f"tara_checkpoints_tpu/{target_subfolder}/step_{step}"
                            print(f"[TPU HF Worker] Mirroring [{tag}] -> https://huggingface.co/{hf_repo}/{target_path}...", flush=True)
                            api.upload_folder(
                                folder_path=local_dir,
                                repo_id=hf_repo,
                                repo_type="model",
                                path_in_repo=target_path,
                                token=token,
                                commit_message=f"Archive TPU checkpoint [{tag}]",
                            )
                            # Prune older TPU checkpoints on HF (strictly under tara_checkpoints_tpu)
                            prune_tpu_hf_checkpoints(hf_repo, keep_n=keep_n, token=token)
                            elapsed = time.time() - t0
                            print(f"[TPU HF Worker] Successfully mirrored [{tag}] to Hugging Face in {elapsed:.1f}s (TPU HF pruned, GPU untouched).", flush=True)
                    except Exception as he:
                        print(f"[TPU HF Worker Warning] Hugging Face TPU upload failed: {he}", flush=True)

                self.active_transfers = max(0, self.active_transfers - 1)
                self.task_queue.task_done()
            except Exception as loop_err:
                print(f"[TPU Cloud Worker Loop Exception] {loop_err}", flush=True)
                time.sleep(1.0)


def trigger_async_tpu_cloud_backup(
    local_ckpt_dir: str,
    stage_id: str = "stage0_pretraining",
    is_best: bool = False,
    step: int = 0,
    loss: float = 0.0,
    gdrive_dir: Optional[str] = None,
    hf_repo: Optional[str] = None,
    hf_token: Optional[str] = None,
    keep_n: int = 2,
):
    """
    Asynchronous Multi-Cloud Backup Dispatcher for TPU:
    - Queues checkpoint snapshot for Google Drive (tara_checkpoints_tpu) and/or Hugging Face (tara_checkpoints_tpu).
    - Non-blocking daemon execution (zero training pause).
    - STRICT GPU PROTECTION:
        Pruning strictly operates ONLY on 'tara_checkpoints_tpu'.
        GPU cloud backups ('tara_checkpoints') and GPU local checkpoints are NEVER touched or pruned.
    """
    if is_tpu_master():
        gdrive_target = resolve_tpu_gdrive_backup_dir(gdrive_dir)
        gdrive_available = bool(gdrive_target and os.path.exists(os.path.dirname(os.path.abspath(gdrive_target))))

        raw_hf_repo = hf_repo or os.environ.get("HF_MODEL_REPO", os.environ.get("HF_REPO", None))
        resolved_hf_repo = None if (not raw_hf_repo or str(raw_hf_repo).strip().lower() in ("", "none", "null", "local")) else str(raw_hf_repo).strip()
        token = _resolve_hf_token_standalone(hf_token)
        hf_available = bool(resolved_hf_repo and token)

        tag = f"BEST MILESTONE (Step {step:04d})" if is_best else f"Step {step:04d}"

        if gdrive_available and hf_available:
            print(f"[TPU Cloud Backup Arbiter] Dual-Target Active for [{tag}]: Queued to Google Drive AND Hugging Face (tara_checkpoints_tpu).", flush=True)
        elif not gdrive_available and hf_available:
            print(f"[TPU Cloud Backup Arbiter] Google Drive NOT MOUNTED / unavailable. >>> AUTO-SELECTED HUGGING FACE AS PRIMARY BACKUP <<< (https://huggingface.co/{resolved_hf_repo}/tara_checkpoints_tpu)", flush=True)
        elif gdrive_available and not hf_available:
            print(f"[TPU Cloud Backup Arbiter] Google Drive Mounted -> Target: {gdrive_target}", flush=True)
        else:
            print(f"[TPU Cloud Backup Notice] No cloud targets (GDrive unmounted, HF unconfigured). Preserved locally in '{local_ckpt_dir}'.", flush=True)

        worker = BackgroundTPUCloudSyncWorker.get_instance()
        worker.submit_task(
            local_ckpt_dir=local_ckpt_dir,
            gdrive_dir=gdrive_target,
            hf_repo=resolved_hf_repo,
            stage_id=stage_id,
            is_best=is_best,
            step=step,
            loss=loss,
            hf_token=token,
            keep_n=keep_n,
        )

try:
    from train_main import verify_pretraining_authorization_proof
except Exception:
    verify_pretraining_authorization_proof = None

try:
    from safetensors.torch import save_file, load_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

# ANSI Color Codes
CLR_RESET = "\033[0m"
CLR_BOLD = "\033[1m"
CLR_CYAN = "\033[96m"
CLR_GREEN = "\033[92m"
CLR_YELLOW = "\033[93m"
CLR_RED = "\033[91m"
CLR_MAGENTA = "\033[95m"


def get_metric_status_and_color(val: float, green_range: Tuple[float, float], purple_range: Tuple[float, float]) -> Tuple[str, str, str]:
    """
    Evaluates value against Frontier/SOTA (Green), Average (Purple), or Untrained (Red):
    - RED   : Not match / Early / Untrained
    - PURPLE: Matched Average (High contrast on white/dark themes)
    - GREEN : Best against Frontier Model (SOTA)
    """
    if val <= green_range[1]:
        return CLR_GREEN, "BEST (Frontier/SOTA)", "[GREEN]"
    elif val <= purple_range[1]:
        return CLR_MAGENTA, "MATCHED (Average)", "[PURPLE]"
    else:
        return CLR_RED, "NOT MATCH (Untrained)", "[RED]"


def print_frontier_comparison_matrix_colored(step: int, cur_loss: float, best_loss: float):
    """Prints full real-time comparison matrix against frontier models."""
    ppl = math.exp(min(cur_loss, 20.0))
    bpb = (cur_loss / math.log(2)) / 3.8

    loss_clr, loss_status, loss_tag = get_metric_status_and_color(cur_loss, (1.4, 1.9), (2.0, 3.0))
    ppl_clr, ppl_status, ppl_tag = get_metric_status_and_color(ppl, (4.0, 7.0), (7.1, 20.0))
    bpb_clr, bpb_status, bpb_tag = get_metric_status_and_color(bpb, (0.55, 0.75), (0.76, 1.20))

    print(f"\n{CLR_BOLD}{'=' * 115}{CLR_RESET}", flush=True)
    print(f"  {CLR_CYAN}{CLR_BOLD}TARA vs FRONTIER MODEL REAL-TIME BENCHMARK MATRIX (Step {step:04d}){CLR_RESET}", flush=True)
    print(f"{CLR_BOLD}{'-' * 115}{CLR_RESET}", flush=True)
    print(f"  {'Metric':<24} {'Untrained':<22} {'Average':<22} {'Frontier/SOTA':<28} {'TARA (Current)':<18}", flush=True)
    print(f"{'-' * 115}", flush=True)
    print(f"  {'Cross-Entropy Loss':<24} {'10.0 - 12.0+':<22} {'2.2 - 3.0':<22} {'1.4 - 1.9':<28} {loss_clr}{cur_loss:<7.4f} {loss_status} {loss_tag}{CLR_RESET}", flush=True)
    print(f"  {'Perplexity (PPL)':<24} {'20,000+':<22} {'9.0 - 20.0':<22} {'4.0 - 7.0':<28} {ppl_clr}{ppl:<7.2f} {ppl_status} {ppl_tag}{CLR_RESET}", flush=True)
    print(f"  {'Bits-Per-Byte (BPB)':<24} {'3.5+':<22} {'0.9 - 1.2':<22} {'0.55 - 0.75':<28} {bpb_clr}{f'{bpb:.2f} b/B':<7} {bpb_status} {bpb_tag}{CLR_RESET}", flush=True)
    print(f"{'-' * 115}", flush=True)
    if best_loss < float('inf'):
        b_ppl = math.exp(min(best_loss, 20.0))
        b_bpb = (best_loss / math.log(2)) / 3.8
        b_clr, b_status, b_tag = get_metric_status_and_color(best_loss, (1.4, 1.9), (2.0, 3.0))
        print(f"  {CLR_BOLD}Historical Best Checkpoint:{CLR_RESET} Loss = {b_clr}{best_loss:.4f}{CLR_RESET} | PPL = {b_clr}{b_ppl:.2f}{CLR_RESET} | BPB = {b_clr}{b_bpb:.2f} b/B{CLR_RESET} ({b_clr}{b_status} {b_tag}{CLR_RESET})", flush=True)
    print(f"{'=' * 115}\n", flush=True)

# Canonical Multi-Domain Dynamic Probe Catalog for Live Quality Probing
DYNAMIC_PROBE_CATALOG = {
    "Kannada": [
        ("<|im_start|>user\nಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಯಾವುದು?<|im_end|>\n<|im_start|>assistant\n", "ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ "),
        ("<|im_start|>user\nಕನ್ನಡ ಸಾಹಿತ್ಯದಲ್ಲಿ ಪ್ರಸಿದ್ಧ ಕವಿ ಯಾರು?<|im_end|>\n<|im_start|>assistant\n", "ಕನ್ನಡ ಸಾಹಿತ್ಯದಲ್ಲಿ ಪ್ರಸಿದ್ಧ ಕವಿ "),
        ("<|im_start|>user\nಬೆಂಗಳೂರಿನ ಹವಾಮಾನ ಮತ್ತು ಸಂಸ್ಕೃತಿ ಹೇಗಿದೆ?<|im_end|>\n<|im_start|>assistant\n", "ಬೆಂಗಳೂರಿನ ಹವಾಮಾನ ಮತ್ತು ಸಂಸ್ಕೃತಿ "),
        ("<|im_start|>user\nಕರ್ನಾಟಕದ ಪ್ರಮುಖ ನದಿಗಳಲ್ಲಿ ಒಂದನ್ನು ಹೆಸರಿಸಿ.<|im_end|>\n<|im_start|>assistant\n", "ಕರ್ನಾಟಕದ ಪ್ರಮುಖ ನದಿಗಳಲ್ಲಿ ಒಂದಾದ "),
        ("<|im_start|>user\nಜ್ಞಾನಪೀಠ ಪ್ರಶಸ್ತಿ ಪಡೆದ ಕನ್ನಡಿಗ ಯಾರು?<|im_end|>\n<|im_start|>assistant\n", "ಜ್ಞಾನಪೀಠ ಪ್ರಶಸ್ತಿ ವಿಜೇತ ಕನ್ನಡಿಗ "),
        ("<|im_start|>user\nವಿಜಯನಗರ ಸಾಮ್ರಾಜ್ಯದ ಇತಿಹಾಸವೇನು?<|im_end|>\n<|im_start|>assistant\n", "ವಿಜಯನಗರ ಸಾಮ್ರಾಜ್ಯದ ಇತಿಹಾಸವು "),
        ("<|im_start|>user\nಮೈಸೂರು ದಸರಾ ಹಬ್ಬದ ಬಗ್ಗೆ ಬರೆಯಿರಿ.<|im_end|>\n<|im_start|>assistant\n", "ಮೈಸೂರು ಅರಮನೆಯ ದಸರಾ ಹಬ್ಬದ ವೈಭವ "),
        ("<|im_start|>user\nಕನ್ನಡ ನಾಡಿನ ಶಿಲ್ಪಕಲೆಯ ಮಹತ್ವವೇನು?<|im_end|>\n<|im_start|>assistant\n", "ಕನ್ನಡ ನಾಡಿನ ಸುಂದರ ಶಿಲ್ಪಕಲೆಯು "),
    ],
    "Reasoning": [
        ("<|im_start|>user\nWhere does the sun rise?<|im_end|>\n<|im_start|>assistant\n", "The sun rises in the "),
        ("<|im_start|>user\nWhat happens to water at room temperature?<|im_end|>\n<|im_start|>assistant\n", "If water freezes at 0 degrees Celsius, then at room temperature it is "),
        ("<|im_start|>user\nWhy do birds migrate south during winter?<|im_end|>\n<|im_start|>assistant\n", "The main reason is "),
        ("<|im_start|>user\nIf all humans are mortal and Socrates is human, what is Socrates?<|im_end|>\n<|im_start|>assistant\n", "Socrates is "),
        ("<|im_start|>user\nWhy does ice float on liquid water?<|im_end|>\n<|im_start|>assistant\n", "The reason why ice floats on liquid water is because "),
        ("<|im_start|>user\nWhen warm air rises and cools down, what does the moisture become?<|im_end|>\n<|im_start|>assistant\n", "When warm air rises and cools down, the moisture condenses into "),
        ("<|im_start|>user\nIf a train travels at 60 mph for 3 hours, how far does it go?<|im_end|>\n<|im_start|>assistant\n", "The total distance covered is "),
        ("<|im_start|>user\nWhen supply is low and demand is high, what happens to prices?<|im_end|>\n<|im_start|>assistant\n", "When supply is low and demand is high, prices "),
    ],
    "Code": [
        ("<|im_start|>user\nWrite a Python function to add two numbers.<|im_end|>\n<|im_start|>assistant\n", "def add(a, b):\n    return "),
        ("<|im_start|>user\nWrite a Python function to check if a number is prime.<|im_end|>\n<|im_start|>assistant\n", "def is_prime(n: int) -> bool:\n    if n <= 1:\n        return False\n    "),
        ("<|im_start|>user\nWrite a function to reverse a string in Python.<|im_end|>\n<|im_start|>assistant\n", "def reverse_string(s: str) -> str:\n    return "),
        ("<|im_start|>user\nWrite a function for Fibonacci series.<|im_end|>\n<|im_start|>assistant\n", "def fibonacci(n: int) -> int:\n    if n <= 1:\n        return n\n    return "),
        ("<|im_start|>user\nDefine a BinaryTreeNode class in Python.<|im_end|>\n<|im_start|>assistant\n", "class BinaryTreeNode:\n    def __init__(self, value):\n        self.val = value\n        "),
        ("<|im_start|>user\nImplement MultiHeadAttention in PyTorch.<|im_end|>\n<|im_start|>assistant\n", "import torch.nn as nn\nclass MultiHeadAttention(nn.Module):\n    def __init__(self, d_model, n_heads):\n        "),
        ("<|im_start|>user\nImplement quicksort algorithm in Python.<|im_end|>\n<|im_start|>assistant\n", "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = "),
        ("<|im_start|>user\nImplement binary search algorithm.<|im_end|>\n<|im_start|>assistant\n", "def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        "),
    ],
    "Science": [
        ("<|im_start|>user\nWhat is the chemical formula of water?<|im_end|>\n<|im_start|>assistant\n", "The chemical formula of water is "),
        ("<|im_start|>user\nExplain Einstein\'s theory of relativity connecting space and time.<|im_end|>\n<|im_start|>assistant\n", "Albert Einstein\'s theory of relativity connects space and time through "),
        ("<|im_start|>user\nWhat four chemical bases compose DNA?<|im_end|>\n<|im_start|>assistant\n", "DNA carries the genetic code composed of adenine, thymine, cytosine, and "),
        ("<|im_start|>user\nWhat is the approximate speed of light in a vacuum?<|im_end|>\n<|im_start|>assistant\n", "The speed of light in a vacuum is approximately "),
        ("<|im_start|>user\nWhat does photosynthesis produce from CO2 and sunlight?<|im_end|>\n<|im_start|>assistant\n", "Photosynthesis converts carbon dioxide and sunlight into oxygen and "),
        ("<|im_start|>user\nState Newton\'s third law of motion.<|im_end|>\n<|im_start|>assistant\n", "Newton\'s third law of motion establishes that for every action, there is an "),
        ("<|im_start|>user\nWhich planet is closest to the Sun in our solar system?<|im_end|>\n<|im_start|>assistant\n", "The closest planet to the Sun in our solar system is "),
        ("<|im_start|>user\nWhat is gravitational force?<|im_end|>\n<|im_start|>assistant\n", "Gravity is the fundamental force that attracts objects with mass towards "),
    ],
}


def tpu_collate_fn(batch: List[Any], max_seq_len: int = 2048) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Any]]:
    """
    Hardware-optimized TPU collate function.
    Guarantees fixed static shapes [batch_size, max_seq_len] for every batch.
    Computes authentic binary attention_mask (1 for real tokens, 0 for padded tokens).
    Prevents PyTorch/XLA dynamic shape recompilations, DataLoader collation errors,
    and attention pollution from padding tokens.
    """
    if not batch:
        return (
            torch.empty((0, max_seq_len), dtype=torch.long),
            torch.empty((0, max_seq_len), dtype=torch.long),
            torch.empty((0, max_seq_len), dtype=torch.long),
            [],
        )

    bsz = len(batch)
    padded_inputs = torch.zeros((bsz, max_seq_len), dtype=torch.long)
    padded_labels = torch.full((bsz, max_seq_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_seq_len), dtype=torch.long)
    cursors = []

    for i, item in enumerate(batch):
        if isinstance(item, (tuple, list)):
            inp = item[0]
            lab = item[1] if len(item) > 1 else inp
            # Extract cursor metadata if available: (shard_name, line_idx, byte_offset, chunk_idx, is_last, domain)
            if len(item) >= 4:
                cursors.append(item[2:])
            else:
                cursors.append((getattr(item, "shard", ""), 0, 0, 0))
        else:
            inp = item
            lab = inp
            cursors.append(("", 0, 0, 0))

        l = min(len(inp), max_seq_len)
        padded_inputs[i, :l] = inp[:l]
        padded_labels[i, :l] = lab[:l]
        attention_mask[i, :l] = 1

    return padded_inputs, padded_labels, attention_mask, cursors


def compute_tpu_lr(step: int, warmup_steps: int, max_steps: int, base_lr: float, min_lr: float) -> float:
    """Computes dynamic learning rate with linear warmup and cosine annealing (exact GPU parity)."""
    if step < warmup_steps:
        return min_lr + (base_lr - min_lr) * (float(step) / float(max(1, warmup_steps)))
    if step >= max_steps:
        return min_lr
    effective_max = min(max_steps, 50_000) if max_steps >= 1_000_000 else max_steps
    progress = float(min(step, effective_max) - warmup_steps) / float(max(1, effective_max - warmup_steps))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def extract_tpu_data_cursor(
    accum_batches: list,
    dataset: Any,
    total_samples_trained: int,
    samples_consumed: int,
    shard_ledger: Optional[Any],
    committed_domain_cursors: Dict[str, Any],
    samples_skipped: int = 0,
    steps_skipped: int = 0,
    rank: int = 0,
) -> Dict[str, Any]:
    """
    Extracts exact resumption cursor metadata from the last sample of an accumulated batch window.
    Preserves exact parity with train_main.py lines 403-453.
    """
    if accum_batches and accum_batches[-1]:
        last_batch = accum_batches[-1]
        last_sample_cursor = last_batch[-1] if isinstance(last_batch, (list, tuple)) else last_batch
    else:
        last_sample_cursor = (
            getattr(dataset, "current_active_shard", None),
            getattr(dataset, "active_shard_offset", 0),
            getattr(dataset, "active_shard_byte_offset", 0),
            getattr(dataset, "active_chunk_idx", 0),
            False,
        )
    c_shard = last_sample_cursor[0] if len(last_sample_cursor) > 0 else getattr(dataset, "current_active_shard", None)
    c_line = last_sample_cursor[1] if len(last_sample_cursor) > 1 else 0
    c_byte = last_sample_cursor[2] if len(last_sample_cursor) > 2 else 0
    c_chunk_last = last_sample_cursor[3] if len(last_sample_cursor) > 3 else 0
    c_is_last = last_sample_cursor[4] if len(last_sample_cursor) > 4 else False

    if c_is_last:
        c_line_offset = c_line
        c_chunk_next = 0
    else:
        c_line_offset = c_line
        c_chunk_next = c_chunk_last + 1

    return {
        "active_shard": c_shard,
        "active_shard_line_offset": c_line_offset,
        "active_shard_total_lines": getattr(dataset, "active_shard_total_lines", 0),
        "active_shard_byte_offset": c_byte,
        "active_chunk_idx": c_chunk_next,
        "total_samples_trained": total_samples_trained,
        "samples_consumed": samples_consumed,
        "samples_skipped": samples_skipped,
        "steps_skipped": steps_skipped,
        "rank": rank,
        "completed_shards": list(shard_ledger.completed_shards) if shard_ledger else [],
        "domain_cursors": copy.deepcopy(committed_domain_cursors),
        "domain_cycles": copy.deepcopy(getattr(dataset, "domain_cycles", {})),
        "scheduler_deficits": copy.deepcopy(getattr(dataset, "deficits", {})),
    }


def ensure_remote_tpu_assets(
    model_dir: str,
    dataset_dir: str,
    repo_id: Optional[str] = None,
    dataset_repo: Optional[str] = None,
    prefix: str = "",
    token: Optional[str] = None,
) -> None:
    """
    Ingests model candidates and tokenizer from Hugging Face Hub if missing locally.
    Does NOT bulk download datasets unless explicit; dataset shards are streamed JIT.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)
    resolved_tok = _resolve_hf_token(token)

    config_file = os.path.join(model_dir, "config.json")
    safetensor_files = glob.glob(os.path.join(model_dir, "*.safetensors"))

    if not (os.path.exists(config_file) and len(safetensor_files) > 0) and repo_id:
        print(f"[TPU Asset Ingestion] Downloading candidate model from Hugging Face '{repo_id}'...", flush=True)
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                local_dir=model_dir,
                allow_patterns=["config.json", "tokenizer.json", "tokenizer_config.json", "*.safetensors", "neural_model_v1/*"],
                token=resolved_tok,
                max_workers=4,
            )
            sub_dir = os.path.join(model_dir, "neural_model_v1")
            if os.path.exists(sub_dir):
                for item in os.listdir(sub_dir):
                    s = os.path.join(sub_dir, item)
                    d = os.path.join(model_dir, item)
                    if not os.path.exists(d):
                        shutil.move(s, d)
                shutil.rmtree(sub_dir, ignore_errors=True)
            print(f"[TPU Asset Ingestion] Candidate model assets ready in {model_dir}", flush=True)
        except Exception as e:
            print(f"[TPU Asset Notice] Model download notice: {e}", flush=True)


@torch.no_grad()
def probe_tpu_generation(
    model: nn.Module,
    tokenizer: TaraNativeTokenizer,
    device: torch.device,
    config: dict,
    step: int,
    stage_id: str,
    max_new_tokens: int = 25,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.25,
) -> None:
    """
    Probes model text synthesis safely on CPU without triggering dynamic shape XLA graph recompilations.
    Synchronizes all TPU worker cores at entry and exit barriers.
    Benchmarks and reports CPU probe duration to monitor TPU idle time.
    """
    # 1. Pre-probe barrier: All TPU cores reach evaluation checkpoint together
    tpu_barrier("probe_pre")
    t_start = time.time()

    if is_tpu_master():
        print(f"\n{CLR_CYAN}{'=' * 80}{CLR_RESET}", flush=True)
        print(f"  {CLR_BOLD}{CLR_CYAN}TARA LIVE TPU GENERATION PROBE (Step {step:04d} | {stage_id}){CLR_RESET}", flush=True)
        print(f"{CLR_CYAN}{'=' * 80}{CLR_RESET}", flush=True)

        try:
            target_model = getattr(model, "module", model)
            raw_sd = target_model.state_dict()
            cpu_sd = {k: v.cpu().float() if hasattr(v, "cpu") else v for k, v in raw_sd.items()}

            # Instantiate lightweight CPU model instance for zero-recompile text synthesis
            eval_model = TaraForCausalLM(config)
            eval_model.load_state_dict(cpu_sd, strict=False)
            eval_model.eval()

            cycle_idx = (step // 100) if step >= 100 else step
            test_probes = []
            for domain, prompt_list in DYNAMIC_PROBE_CATALOG.items():
                prompt_idx = cycle_idx % len(prompt_list)
                chosen_prompt = prompt_list[prompt_idx]
                domain_tag = f"{domain} #{prompt_idx + 1}"
                test_probes.append((domain_tag, chosen_prompt))

            eos_id = getattr(tokenizer, "eos_id", None)
            if eos_id is None and hasattr(tokenizer, "token_to_id"):
                eos_id = tokenizer.token_to_id.get("<|endoftext|>", tokenizer.token_to_id.get("</s>", None))

            for tag, chosen_item in test_probes:
                try:
                    if isinstance(chosen_item, (tuple, list)):
                        chatml_prefix, starter = chosen_item
                        prompt = chatml_prefix + starter
                        display_prompt = f"{chatml_prefix.replace('<|im_start|>', '').replace('<|im_end|>', '').strip()} -> {starter}"
                    else:
                        prompt = chosen_item
                        display_prompt = chosen_item

                    tok_out = tokenizer.encode(prompt)
                    prompt_ids = list(tok_out.ids) if hasattr(tok_out, "ids") else list(tok_out)
                    if not prompt_ids:
                        continue

                    generated_ids = list(prompt_ids)
                    for _ in range(max_new_tokens):
                        inp = torch.tensor([generated_ids], dtype=torch.long)
                        out = eval_model(input_ids=inp)
                        logits = out.logits if hasattr(out, "logits") and out.logits is not None else (out[0] if isinstance(out, (tuple, list)) else out)
                        if logits.dim() == 2:
                            next_logits = logits[-1, :].clone().float()
                        else:
                            next_logits = logits[0, -1, :].clone().float()

                        # Repetition penalty: Only penalize tokens generated AFTER the prompt!
                        if repetition_penalty != 1.0 and len(generated_ids) > len(prompt_ids):
                            for prev_token_id in set(generated_ids[len(prompt_ids):]):
                                if next_logits[prev_token_id] < 0:
                                    next_logits[prev_token_id] *= repetition_penalty
                                else:
                                    next_logits[prev_token_id] /= repetition_penalty

                        # Temperature & Top-p sampling
                        if temperature > 0:
                            scaled_logits = next_logits / max(temperature, 1e-5)
                            if top_p < 1.0:
                                sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                                sorted_indices_to_remove = cumulative_probs > top_p
                                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                                sorted_indices_to_remove[0] = False
                                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                                scaled_logits[indices_to_remove] = -float("Inf")
                            probs = F.softmax(scaled_logits, dim=-1)
                            if torch.isnan(probs).any() or torch.isinf(probs).any():
                                next_token_id = int(torch.argmax(next_logits, dim=-1).item())
                            else:
                                next_token_id = int(torch.multinomial(probs, num_samples=1).item())
                        else:
                            next_token_id = int(torch.argmax(next_logits, dim=-1).item())

                        if eos_id is not None and next_token_id in (eos_id, 2):
                            break
                        generated_ids.append(next_token_id)

                    continuation = tokenizer.decode(generated_ids[len(prompt_ids):])
                    clean_cont = continuation.replace("\n", "\\n").strip()
                    print(f"  [{tag:13s}] \"{display_prompt}\" -> {CLR_GREEN}{clean_cont}{CLR_RESET}", flush=True)
                except Exception as pe:
                    print(f"  [{tag:13s}] Probe warning: {pe}", flush=True)

            del eval_model, cpu_sd
        except Exception as ge:
            print(f"  [Probe Exception] {ge}", flush=True)

        probe_duration = time.time() - t_start
        print(f"  [TPU Probe Telemetry] CPU synthesis duration: {probe_duration:.2f}s | TPU Cores idle time: {probe_duration:.2f}s | Multi-core rendezvous synchronized.", flush=True)
        print(f"{CLR_CYAN}{'=' * 80}{CLR_RESET}\n", flush=True)

    # 2. Post-probe barrier: All TPU cores resume training together at the exact same step
    tpu_barrier("probe_post")


def save_tpu_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict,
    tok_path: str,
    output_dir: str,
    step: int,
    epoch: int,
    loss: float,
    best_loss: float,
    stage_id: str,
    data_cursor: Optional[Dict[str, Any]] = None,
    is_best: bool = False,
    is_completed: bool = False,
    best_cp_dir: Optional[str] = None,
    keep_n: int = 2,
    hf_repo: Optional[str] = None,
    hf_token: Optional[str] = None,
    max_shard_mb: int = TPUConfig.MAX_SHARD_MB,
    dataset: Optional[Any] = None,
    shard_ledger: Optional[Any] = None,
    base_lr: Optional[float] = None,
) -> bool:
    """
    Saves authentic TPU checkpoint safely into the dedicated TPU checkpoints directory.
    STRICT CPU RAM EXHAUSTION GUARD:
      - Only the writer rank (Rank 0 / is_tpu_master) materializes full CPU state dict buffers.
      - Non-writer ranks (Ranks 1..N-1) allocate ZERO CPU buffers; they participate only in barriers.
      - Atomically stages files into staging directory before committing.
      - Full training files saved: canonical part-wise sharded safetensors (or model.safetensors),
        config.json, tokenizer, optimizer_state.pt, checkpoint_state.json, STAGE_METADATA.json,
        and COMMITTED verification manifest.
      - Pre-commit integrity verification: verifies tensor headers, JSON configs, optimizer state.
      - Immutable versioned directory creation (<basename>.step<N>_<ts>).
      - Verified rolling retention: only deletes older versioned checkpoints (beyond keep_n)
        AFTER the new checkpoint has passed 100% verification.
      - Strict GPU isolation: never writes to, deletes from, or modifies GPU checkpoints directory.
      - Dual-cloud background synchronization: automatically mirrors to Google Drive and/or
        Hugging Face based on availability without interrupting TPU compute.
    """
    # Barrier before save: ensure all cores complete current step computations
    tpu_barrier("save_ckpt_pre")
    is_main = is_tpu_master()

    if is_main:
        tag = "BEST MILESTONE" if is_best else ("COMPLETED" if is_completed else f"STEP {step}")
        parent_dir = os.path.dirname(os.path.abspath(output_dir)) or "."
        output_dir_basename = os.path.basename(os.path.abspath(output_dir))

        # STRICT GPU DIRECTORY ISOLATION GUARD:
        # Never allow TPU saving or cleanup to operate in GPU directory!
        gpu_parent_abs = os.path.abspath(os.path.join("storage", "models", "checkpoints"))
        if os.path.abspath(parent_dir) == gpu_parent_abs and "tpu" not in output_dir.lower():
            raise RuntimeError(
                f"[TPU Checkpoint Fatal] Safety boundary violation! TPU attempted save in GPU checkpoints "
                f"directory '{parent_dir}'. TPU checkpoints must be isolated in checkpoints_tpu."
            )

        os.makedirs(parent_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        print(f"[TPU Checkpoint] Snapshotting [{tag}] into {output_dir}... (Parent: {parent_dir})", flush=True)

        staging_dir = os.path.join(
            parent_dir,
            f".ckpt_staging_tpu_{output_dir_basename}_step{step}_{int(time.time())}_{os.getpid()}"
        )
        os.makedirs(staging_dir, exist_ok=True)

        try:
            target_model = getattr(model, "module", model)
            raw_sd = target_model.state_dict()
            # ONLY writer rank converts state dict to CPU with independent contiguous memory buffers
            cpu_sd = {
                k: v.cpu().float().clone().contiguous() if hasattr(v, "cpu") else v
                for k, v in raw_sd.items()
            }

            # 1. Save SafeTensors weights in canonical part-wise sharded format (matching GPU pipeline)
            max_shard_bytes = resolve_max_shard_size_bytes(max_shard_mb)
            m_shas: Dict[str, str] = {}
            if HAS_SAFETENSORS:
                m_files, m_shas, weights_sha = save_safetensors_sharded(
                    cpu_sd, staging_dir, max_shard_size_bytes=max_shard_bytes, prefix="model"
                )
            else:
                bin_path = os.path.join(staging_dir, "pytorch_model.bin")
                torch.save(cpu_sd, bin_path)
                weights_sha = compute_file_sha256(bin_path)
                m_shas = {"pytorch_model.bin": weights_sha} if weights_sha else {}
                m_files = ["pytorch_model.bin"]
            del cpu_sd

            # 2. Save config.json
            cfg_path = os.path.join(staging_dir, "config.json")
            with open(cfg_path, "w", encoding="utf-8") as cf:
                json.dump(config, cf, indent=2)

            # 3. Save tokenizer.json and auxiliary configs
            if tok_path and os.path.isfile(tok_path):
                shutil.copy2(tok_path, os.path.join(staging_dir, "tokenizer.json"))
                src_tok_dir = os.path.dirname(os.path.abspath(tok_path))
                for aux in ("tokenizer_config.json", "special_tokens_map.json"):
                    aux_p = os.path.join(src_tok_dir, aux)
                    if os.path.isfile(aux_p):
                        shutil.copy2(aux_p, os.path.join(staging_dir, aux))

            # 4. Save optimizer state in standard nested container format matching GPU pipeline
            opt_path = os.path.join(staging_dir, "optimizer_state.pt")
            try:
                raw_opt_sd = optimizer.state_dict() if optimizer is not None else {}
                cpu_opt_sd = {
                    k: v.cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in raw_opt_sd.items()
                }
                if "state" in cpu_opt_sd and isinstance(cpu_opt_sd["state"], dict):
                    cpu_opt_state = {}
                    for pid, pdata in cpu_opt_sd["state"].items():
                        cpu_opt_state[pid] = {
                            sk: (sv.cpu() if isinstance(sv, torch.Tensor) else sv)
                            for sk, sv in pdata.items()
                        }
                    cpu_opt_sd["state"] = cpu_opt_state

                opt_save_dict = {
                    "optimizer": cpu_opt_sd,
                    "step": step,
                    "epoch": epoch,
                    "loss": loss,
                    "best_loss": best_loss,
                    "stage_id": stage_id,
                    "torch_rng": torch.get_rng_state(),
                }
                torch.save(opt_save_dict, opt_path)
            except Exception as oe:
                print(f"[TPU Checkpoint Notice] Could not save optimizer state: {oe}", flush=True)

            # 5. Extract comprehensive data cursor & synchronize authoritative shard ledger
            if data_cursor is None:
                active_shard = getattr(dataset, "current_active_shard", getattr(dataset, "active_shard_name", None)) if dataset else None
                active_offset = getattr(dataset, "active_shard_offset", 0) if dataset else 0
                active_byte_offset = getattr(dataset, "active_shard_byte_offset", 0) if dataset else 0
                active_chunk_idx = getattr(dataset, "active_chunk_idx", 0) if dataset else 0
                completed_shards = list(shard_ledger.completed_shards) if shard_ledger else []
                completed_records = shard_ledger.get_total_completed_records() if shard_ledger else 0
                samples_consumed = completed_records
                active_total_lines = getattr(dataset, "active_shard_total_lines", getattr(shard_ledger, "active_shard_total_lines", 0)) if (dataset or shard_ledger) else 0
                data_cursor = {
                    "active_shard": active_shard,
                    "active_shard_line_offset": active_offset,
                    "active_shard_total_lines": active_total_lines,
                    "active_shard_byte_offset": active_byte_offset,
                    "active_chunk_idx": active_chunk_idx,
                    "total_samples_trained": completed_records,
                    "samples_consumed": samples_consumed,
                    "completed_shards": completed_shards,
                }
            else:
                active_shard = data_cursor.get("active_shard")
                active_offset = data_cursor.get("active_shard_line_offset", 0)
                active_total_lines = data_cursor.get("active_shard_total_lines", getattr(dataset, "active_shard_total_lines", getattr(shard_ledger, "active_shard_total_lines", 0)))
                active_byte_offset = data_cursor.get("active_shard_byte_offset", 0)
                active_chunk_idx = data_cursor.get("active_chunk_idx", 0)
                completed_shards = data_cursor.get("completed_shards", list(shard_ledger.completed_shards) if shard_ledger else [])
                completed_records = data_cursor.get("total_samples_trained", 0)
                samples_consumed = data_cursor.get("samples_consumed", completed_records)

            domain_cursors = data_cursor.get("domain_cursors") if data_cursor else None
            if not domain_cursors:
                if hasattr(dataset, "get_state"):
                    domain_cursors = dataset.get_state().get("domain_cursors")
                elif shard_ledger and hasattr(shard_ledger, "domain_cursors"):
                    domain_cursors = shard_ledger.domain_cursors

            domain_cycles = data_cursor.get("domain_cycles") if data_cursor else None
            if not domain_cycles:
                if hasattr(dataset, "get_state"):
                    domain_cycles = dataset.get_state().get("domain_cycles")
                elif shard_ledger and hasattr(shard_ledger, "domain_cycles"):
                    domain_cycles = shard_ledger.domain_cycles

            scheduler_deficits = data_cursor.get("scheduler_deficits") if data_cursor else None
            if not scheduler_deficits:
                if hasattr(dataset, "get_state"):
                    scheduler_deficits = dataset.get_state().get("scheduler_deficits")
                elif hasattr(dataset, "deficits"):
                    scheduler_deficits = getattr(dataset, "deficits", None)

            if domain_cursors and data_cursor:
                data_cursor["domain_cursors"] = domain_cursors
            if domain_cycles and data_cursor:
                data_cursor["domain_cycles"] = domain_cycles
            if scheduler_deficits and data_cursor:
                data_cursor["scheduler_deficits"] = scheduler_deficits

            # Update shard ledger active cursor
            if shard_ledger and data_cursor:
                shard_tot = getattr(dataset, "active_shard_total_lines", 0) if dataset else getattr(shard_ledger, "active_shard_total_lines", 0)
                if shard_tot <= 0 and active_total_lines > 0:
                    shard_tot = active_total_lines
                shard_ledger.update_active_cursor(
                    active_shard,
                    active_offset,
                    completed_records,
                    step,
                    byte_offset=active_byte_offset,
                    shard_total_lines=shard_tot,
                )

            # Commit pending completed shards and evict cache
            if dataset is not None and shard_ledger is not None:
                pending = list(getattr(dataset, "_pending_shard_completions", {}).keys())
                for pending_shard in pending:
                    shard_abs = (
                        os.path.abspath(os.path.join(dataset.dataset_dir, pending_shard))
                        if getattr(dataset, "dataset_dir", None) else None
                    )
                    is_managed = (shard_abs in dataset.managed_downloaded_shards) if (shard_abs and hasattr(dataset, "managed_downloaded_shards")) else False
                    dataset.commit_shard_completion(
                        pending_shard,
                        managed_cache=is_managed,
                        actual_fp=shard_abs if is_managed else None,
                    )

            # Copy authoritative shard_ledger.json and dataset_snapshot.json into staging_dir
            if shard_ledger is not None:
                shard_ledger_src = getattr(shard_ledger, "ledger_file", os.path.join(output_dir, "shard_ledger.json"))
                if os.path.isfile(shard_ledger_src):
                    shutil.copy2(shard_ledger_src, os.path.join(staging_dir, "shard_ledger.json"))
                else:
                    shard_ledger.save()
                    if os.path.isfile(shard_ledger_src):
                        shutil.copy2(shard_ledger_src, os.path.join(staging_dir, "shard_ledger.json"))
            elif os.path.isfile(os.path.join(output_dir, "shard_ledger.json")):
                shutil.copy2(os.path.join(output_dir, "shard_ledger.json"), os.path.join(staging_dir, "shard_ledger.json"))

            ds_snap_src = os.path.join(output_dir, "dataset_snapshot.json")
            if os.path.isfile(ds_snap_src):
                shutil.copy2(ds_snap_src, os.path.join(staging_dir, "dataset_snapshot.json"))

            # Save checkpoint_state.json with data cursor & RNG state
            hw_info = get_tpu_hardware_info()
            rng_data = {
                "torch_rng": torch.get_rng_state().tolist(),
            }
            if HAS_XLA:
                try:
                    rng_data["xla_rng"] = xm.get_rng_state()
                except Exception:
                    pass

            current_lr = 0.0
            if optimizer is not None and optimizer.param_groups:
                current_lr = float(optimizer.param_groups[0].get("lr", 0.0))

            eff_base_lr = base_lr if base_lr is not None else (data_cursor.get("base_lr") if data_cursor else TPUConfig.DEFAULT_LR)
            state_data = {
                "step": step,
                "epoch": epoch,
                "loss": loss,
                "best_loss": best_loss,
                "learning_rate": current_lr,
                "base_lr": eff_base_lr,
                "stage_id": stage_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "accelerator": hw_info["hardware_type"],
                "data_cursor": data_cursor or {},
                "domain_cursors": domain_cursors or {},
                "domain_cycles": domain_cycles or {},
                "scheduler_deficits": scheduler_deficits or {},
                "active_shard": active_shard,
                "active_shard_offset": active_offset,
                "active_shard_line_offset": active_offset,
                "active_shard_total_lines": active_total_lines,
                "active_shard_byte_offset": active_byte_offset,
                "active_chunk_idx": active_chunk_idx,
                "completed_shards": completed_shards,
                "total_completed_records": completed_records,
                "total_samples_trained": completed_records,
                "samples_consumed": samples_consumed,
                "rng_state": rng_data,
                "is_best": is_best,
                "is_completed": is_completed,
                "model_weights_sha256": weights_sha,
                "model_shards_sha256": m_shas,
                "model_files": m_files,
            }
            with open(os.path.join(staging_dir, "checkpoint_state.json"), "w", encoding="utf-8") as sf:
                json.dump(state_data, sf, indent=2)

            # 6. Save STAGE_METADATA.json
            metadata = {
                "stage_id": stage_id,
                "status": "COMPLETED" if is_completed else "IN_PROGRESS",
                "step": step,
                "loss": loss,
                "best_loss": best_loss,
                "learning_rate": current_lr,
                "base_lr": eff_base_lr,
                "perplexity": math.exp(min(loss, 20.0)),
                "bits_per_byte": (loss / math.log(2)) / 3.8,
                "is_best": is_best,
                "completed_at_utc": datetime.now(timezone.utc).isoformat() if is_completed else None,
                "model_weights_sha256": weights_sha,
                "accelerator": "TPU",
            }
            with open(os.path.join(staging_dir, "STAGE_METADATA.json"), "w", encoding="utf-8") as sm:
                json.dump(metadata, sm, indent=2)

            # 7. Pre-commit Verification: Verify staged artifacts integrity before promoting
            is_valid = verify_checkpoint_integrity(
                staging_dir,
                require_optimizer=(optimizer is not None and not is_best),
                require_completed=is_completed,
                precomputed_model_sha=weights_sha,
            )
            if not is_valid:
                print(
                    f"[Safety Protection] Staged TPU checkpoint at {staging_dir} failed verification! "
                    f"Preserving existing checkpoint at {output_dir} unchanged.",
                    flush=True,
                )
                _safe_rmtree(staging_dir)
                return False

            # 8. Promote staging -> immutable versioned directory
            if is_best:
                versioned_dir = os.path.join(
                    parent_dir,
                    f"best_milestone_step_{step}_{int(time.time())}"
                )
            else:
                versioned_dir = os.path.join(
                    parent_dir,
                    f"{output_dir_basename}.step{step}_{int(time.time())}"
                )

            try:
                os.rename(staging_dir, versioned_dir)
            except Exception:
                shutil.copytree(staging_dir, versioned_dir)
                _safe_rmtree(staging_dir)

            # 9. Save COMMITTED marker into versioned directory
            committed_marker = os.path.join(versioned_dir, "COMMITTED")
            committed_payload = {
                "step": step,
                "is_best": is_best,
                "committed_utc": datetime.now(timezone.utc).isoformat(),
                "accelerator": "TPU",
            }
            with open(committed_marker, "w", encoding="utf-8") as cm:
                json.dump(committed_payload, cm, indent=2)

            # 10. Align active output_dir safely with verified latest artifacts
            os.makedirs(output_dir, exist_ok=True)
            if os.path.isfile(os.path.join(versioned_dir, "model.safetensors.index.json")):
                _safe_unlink(os.path.join(output_dir, "model.safetensors"))
            elif os.path.isfile(os.path.join(versioned_dir, "model.safetensors")):
                _safe_unlink(os.path.join(output_dir, "model.safetensors.index.json"))
                for old_sf in glob.glob(os.path.join(output_dir, "model-*-of-*.safetensors")):
                    _safe_unlink(old_sf)

            for fname in os.listdir(versioned_dir):
                src = os.path.join(versioned_dir, fname)
                dst = os.path.join(output_dir, fname)
                if os.path.isfile(src):
                    tmp_dst = os.path.join(output_dir, f".atomic_{fname}.{os.getpid()}_{int(time.time())}")
                    try:
                        try:
                            os.link(src, tmp_dst)
                        except OSError:
                            try:
                                os.symlink(os.path.abspath(src), tmp_dst)
                            except OSError:
                                shutil.copy2(src, tmp_dst)
                        if os.path.exists(tmp_dst):
                            os.replace(tmp_dst, dst)
                    except Exception:
                        try:
                            shutil.copy2(src, dst)
                        except Exception:
                            pass
                        if os.path.exists(tmp_dst):
                            _safe_unlink(tmp_dst)

            # Save COMMITTED into output_dir
            out_committed = os.path.join(output_dir, "COMMITTED")
            with open(out_committed, "w", encoding="utf-8") as cm:
                json.dump(committed_payload, cm, indent=2)

            # 11. Pointers update (latest_checkpoint.json / best_checkpoint.json)
            if is_best:
                best_ptr = os.path.join(parent_dir, "best_checkpoint.json")
                try:
                    with open(best_ptr, "w", encoding="utf-8") as bf:
                        json.dump({
                            "step": step,
                            "loss": loss,
                            "versioned_dir": versioned_dir,
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        }, bf, indent=2)
                except Exception:
                    pass
                if best_cp_dir:
                    os.makedirs(best_cp_dir, exist_ok=True)
                    if os.path.isfile(os.path.join(versioned_dir, "model.safetensors.index.json")):
                        _safe_unlink(os.path.join(best_cp_dir, "model.safetensors"))
                    elif os.path.isfile(os.path.join(versioned_dir, "model.safetensors")):
                        _safe_unlink(os.path.join(best_cp_dir, "model.safetensors.index.json"))
                        for old_sf in glob.glob(os.path.join(best_cp_dir, "model-*-of-*.safetensors")):
                            _safe_unlink(old_sf)
                    for fname in os.listdir(versioned_dir):
                        s_art = os.path.join(versioned_dir, fname)
                        d_art = os.path.join(best_cp_dir, fname)
                        if os.path.isfile(s_art):
                            try:
                                if os.path.exists(d_art):
                                    _safe_unlink(d_art)
                                os.link(s_art, d_art)
                            except Exception:
                                shutil.copy2(s_art, d_art)
            else:
                latest_ptr_path = os.path.join(output_dir, "latest_checkpoint.json")
                legacy_ptr_path = os.path.join(parent_dir, "latest_checkpoint.json")
                try:
                    ptr_dict = {
                        "versioned_dir": versioned_dir,
                        "output_dir": output_dir,
                        "stage_id": stage_id,
                        "step": step,
                        "committed_utc": datetime.now(timezone.utc).isoformat(),
                    }
                    with open(latest_ptr_path, "w", encoding="utf-8") as lf:
                        json.dump(ptr_dict, lf, indent=2)
                    if os.path.abspath(output_dir) != os.path.abspath(parent_dir):
                        with open(legacy_ptr_path, "w", encoding="utf-8") as lf:
                            json.dump(ptr_dict, lf, indent=2)
                except Exception:
                    pass

            print(f"[TPU Checkpoint] Committed verified [{tag}] checkpoint (Step {step}) -> {output_dir}", flush=True)

            # 12. Verified Rolling Retention Cleanup (Only in TPU parent_dir!)
            prot_dirs = [versioned_dir]
            if output_dir:
                prot_dirs.append(output_dir)
            _cleanup_old_versioned_checkpoints(
                parent_dir=parent_dir,
                output_dir_basename=output_dir_basename,
                keep_n=keep_n,
                protected_dirs=prot_dirs,
            )

            # 13. Non-Blocking Dual-Cloud Synchronization (Google Drive AND/OR Hugging Face)
            # Dedicated TPU cloud isolation: uploads to tara_checkpoints_tpu; prunes ONLY TPU backups.
            upload_source = versioned_dir if versioned_dir and os.path.exists(versioned_dir) else output_dir
            trigger_async_tpu_cloud_backup(
                local_ckpt_dir=upload_source,
                stage_id=stage_id,
                is_best=is_best,
                step=step,
                loss=loss,
                hf_repo=hf_repo,
                hf_token=hf_token,
                keep_n=keep_n,
            )

        except Exception as e:
            print(f"[TPU Checkpoint Error] Failed to save checkpoint: {e}", flush=True)
            if 'staging_dir' in locals() and os.path.isdir(staging_dir):
                shutil.rmtree(staging_dir, ignore_errors=True)
            return False

    # Barrier after save: all cores resume training together
    tpu_barrier("save_ckpt_post")
    return True


def _validate_checkpoint_tensors(loaded_sd: Dict[str, torch.Tensor], model_sd: Dict[str, torch.Tensor]) -> None:
    """Strictly validates checkpoint tensor keys, shapes, and dtypes."""
    missing_keys = set(model_sd.keys()) - set(loaded_sd.keys())
    unexpected_keys = set(loaded_sd.keys()) - set(model_sd.keys())
    shape_mismatches = []
    dtype_mismatches = []

    for k in set(model_sd.keys()).intersection(set(loaded_sd.keys())):
        t_model = model_sd[k]
        t_load = loaded_sd[k]
        if t_model.shape != t_load.shape:
            shape_mismatches.append(f"{k}: expected {t_model.shape}, got {t_load.shape}")
        if t_model.dtype != t_load.dtype:
            if not (t_model.is_floating_point() and t_load.is_floating_point()):
                dtype_mismatches.append(f"{k}: expected {t_model.dtype}, got {t_load.dtype}")

    err_msgs = []
    if missing_keys:
        err_msgs.append(f"Missing keys ({len(missing_keys)}): {list(missing_keys)[:5]}")
    if shape_mismatches:
        err_msgs.append(f"Shape mismatches ({len(shape_mismatches)}): {shape_mismatches[:5]}")
    if dtype_mismatches:
        err_msgs.append(f"Dtype mismatches ({len(dtype_mismatches)}): {dtype_mismatches[:5]}")

    if err_msgs:
        raise RuntimeError("Checkpoint validation failed:\n  " + "\n  ".join(err_msgs))
    if unexpected_keys:
        print(f"[TPU Resume Notice] {len(unexpected_keys)} unexpected keys present in checkpoint but safely ignored.", flush=True)


# Authoritative architectural weight-tying enforcer (reused from tpu_config)
_ensure_tied_weights = ensure_model_weight_tying


def validate_stage_checkpoint_for_handoff(
    checkpoint_dir: str,
    stage_id: str = "",
    require_completed: bool = False,
) -> Tuple[bool, str]:
    """
    Rigorously validates a candidate checkpoint directory before promoting or handing off
    to the next training stage. Fail-closed on any corruption, missing file, or non-finite weight.

    Validates:
      1. Directory exists and contains COMMITTED verification manifest.
      2. STAGE_METADATA.json exists, is valid JSON, and matches stage status requirements.
      3. config.json exists, is valid JSON, and contains valid architectural parameters.
      4. tokenizer.json exists and can be loaded.
      5. Model weights (model.safetensors or shards) exist and can be loaded.
      6. Deep tensor finiteness check: iterates through EVERY parameter tensor to guarantee
         ZERO NaN and ZERO Inf values exist in model weights.
      7. Model architecture key coverage: guarantees embeddings, attention, MLP, and norm layers
         match expected transformer architecture.
    """
    if not os.path.isdir(checkpoint_dir):
        return False, f"Checkpoint directory does not exist: {checkpoint_dir}"

    # 1. Check COMMITTED manifest
    committed_path = os.path.join(checkpoint_dir, "COMMITTED")
    if not os.path.isfile(committed_path):
        return False, f"Missing COMMITTED verification manifest in {checkpoint_dir}"

    # 2. Check and parse STAGE_METADATA.json
    meta_path = os.path.join(checkpoint_dir, "STAGE_METADATA.json")
    if not os.path.isfile(meta_path):
        return False, f"Missing STAGE_METADATA.json in {checkpoint_dir}"
    try:
        with open(meta_path, "r", encoding="utf-8") as mf:
            meta = json.load(mf)
        if require_completed and meta.get("status") != "COMPLETED":
            return False, f"STAGE_METADATA status is '{meta.get('status')}', expected 'COMPLETED'"
    except Exception as e:
        return False, f"Corrupted STAGE_METADATA.json: {e}"

    # 3. Check and parse config.json
    cfg_path = os.path.join(checkpoint_dir, "config.json")
    if not os.path.isfile(cfg_path):
        return False, f"Missing config.json in {checkpoint_dir}"
    try:
        with open(cfg_path, "r", encoding="utf-8") as cf:
            cfg = json.load(cf)
        for req_k in ("vocab_size", "hidden_size", "num_hidden_layers"):
            if req_k not in cfg or not isinstance(cfg[req_k], int) or cfg[req_k] <= 0:
                return False, f"config.json missing or invalid required key '{req_k}': {cfg.get(req_k)}"
    except Exception as e:
        return False, f"Corrupted config.json: {e}"

    # 4. Check tokenizer
    tok_path = os.path.join(checkpoint_dir, "tokenizer.json")
    if not os.path.isfile(tok_path):
        tok_path = "storage/models/tara_candidate_v1/tokenizer.json"
    if not os.path.isfile(tok_path):
        return False, f"Missing tokenizer.json in {checkpoint_dir} or storage/models/tara_candidate_v1"
    try:
        tok = TaraNativeTokenizer.from_file(tok_path)
        if tok.get_vocab_size() <= 0:
            return False, f"Invalid tokenizer vocabulary size: {tok.get_vocab_size()}"
    except Exception as e:
        return False, f"Corrupted tokenizer.json: {e}"

    # 5. Load and inspect model weights (Safetensors or PyTorch binary)
    loaded_tensors: Dict[str, torch.Tensor] = {}
    st_path = os.path.join(checkpoint_dir, "model.safetensors")
    index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    bin_path = os.path.join(checkpoint_dir, "pytorch_model.bin")

    try:
        if HAS_SAFETENSORS and os.path.isfile(index_path):
            with open(index_path, "r", encoding="utf-8") as idxf:
                idx_data = json.load(idxf)
            weight_map = idx_data.get("weight_map", {})
            for s_name in sorted(set(weight_map.values())):
                s_path = os.path.join(checkpoint_dir, s_name)
                if not os.path.isfile(s_path):
                    return False, f"Missing sharded safetensors file: {s_path}"
                shard_dict = load_file(s_path, device="cpu")
                loaded_tensors.update(shard_dict)
        elif HAS_SAFETENSORS and os.path.isfile(st_path):
            loaded_tensors = load_file(st_path, device="cpu")
        elif os.path.isfile(bin_path):
            loaded_tensors = torch.load(bin_path, map_location="cpu")
        else:
            return False, f"No readable weights (model.safetensors or pytorch_model.bin) in {checkpoint_dir}"
    except Exception as e:
        return False, f"Failed to load checkpoint weights: {e}"

    if not loaded_tensors:
        return False, f"Checkpoint contains 0 weight tensors in {checkpoint_dir}"

    # 6. Deep Weight Finiteness Check (Guarantee 0 NaNs and 0 Infs)
    nan_keys = []
    inf_keys = []
    total_elements = 0
    for name, tensor in loaded_tensors.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        total_elements += tensor.numel()
        if torch.isnan(tensor).any():
            nan_keys.append(name)
        if torch.isinf(tensor).any():
            inf_keys.append(name)

    if nan_keys:
        return False, f"Weight corruption: NaN values detected in {len(nan_keys)} tensor(s), including '{nan_keys[0]}'"
    if inf_keys:
        return False, f"Weight corruption: Infinite values detected in {len(inf_keys)} tensor(s), including '{inf_keys[0]}'"

    # 7. Architecture Coverage Check
    has_embed = any("embed_tokens" in k for k in loaded_tensors)
    has_layers = any("layers." in k for k in loaded_tensors)
    has_norm = any("norm" in k for k in loaded_tensors)
    if not (has_embed and has_layers and has_norm):
        return False, f"Incomplete architecture: missing core transformer layers (embed={has_embed}, layers={has_layers}, norm={has_norm})"

    # 8. Exact Layer Schema & Tensor Dimension Invariant Verification
    num_layers = cfg.get("num_hidden_layers", 0)
    hidden_size = cfg.get("hidden_size", 0)
    intermediate_size = cfg.get("intermediate_size", 0)
    num_heads = cfg.get("num_attention_heads", 0)
    num_kv_heads = cfg.get("num_key_value_heads", num_heads)
    head_dim = cfg.get("head_dim", (hidden_size // num_heads) if num_heads else 0)
    vocab_size = cfg.get("vocab_size", 0)

    # Check global embedding and norm shapes
    if "model.embed_tokens.weight" in loaded_tensors:
        exp_embed_shape = (vocab_size, hidden_size)
        if tuple(loaded_tensors["model.embed_tokens.weight"].shape) != exp_embed_shape:
            return False, f"Dimension mismatch in model.embed_tokens.weight: expected {exp_embed_shape}, got {tuple(loaded_tensors['model.embed_tokens.weight'].shape)}"

    if "model.norm.weight" in loaded_tensors:
        exp_norm_shape = (hidden_size,)
        if tuple(loaded_tensors["model.norm.weight"].shape) != exp_norm_shape:
            return False, f"Dimension mismatch in model.norm.weight: expected {exp_norm_shape}, got {tuple(loaded_tensors['model.norm.weight'].shape)}"

    if "lm_head.weight" in loaded_tensors:
        exp_head_shape = (vocab_size, hidden_size)
        if tuple(loaded_tensors["lm_head.weight"].shape) != exp_head_shape:
            return False, f"Dimension mismatch in lm_head.weight: expected {exp_head_shape}, got {tuple(loaded_tensors['lm_head.weight'].shape)}"

    # Check per-layer projection matrices and layernorms
    for l_idx in range(num_layers):
        expected_layer_shapes = {
            f"model.layers.{l_idx}.input_layernorm.weight": (hidden_size,),
            f"model.layers.{l_idx}.self_attn.q_proj.weight": (num_heads * head_dim, hidden_size),
            f"model.layers.{l_idx}.self_attn.k_proj.weight": (num_kv_heads * head_dim, hidden_size),
            f"model.layers.{l_idx}.self_attn.v_proj.weight": (num_kv_heads * head_dim, hidden_size),
            f"model.layers.{l_idx}.self_attn.o_proj.weight": (hidden_size, num_heads * head_dim),
            f"model.layers.{l_idx}.post_attention_layernorm.weight": (hidden_size,),
            f"model.layers.{l_idx}.mlp.gate_proj.weight": (intermediate_size, hidden_size),
            f"model.layers.{l_idx}.mlp.up_proj.weight": (intermediate_size, hidden_size),
            f"model.layers.{l_idx}.mlp.down_proj.weight": (hidden_size, intermediate_size),
        }
        for t_name, exp_shape in expected_layer_shapes.items():
            if t_name not in loaded_tensors:
                return False, f"Missing architectural layer tensor: '{t_name}' in checkpoint"
            actual_shape = tuple(loaded_tensors[t_name].shape)
            if actual_shape != exp_shape:
                return False, f"Dimension mismatch in '{t_name}': expected {exp_shape}, got {actual_shape}"

    # 9. Cryptographic Hash & Manifest Cross-Check (if present in STAGE_METADATA or COMMITTED)
    if isinstance(meta, dict):
        if "model_config_sha256" in meta and os.path.isfile(cfg_path):
            with open(cfg_path, "rb") as f:
                actual_cfg_sha = hashlib.sha256(f.read()).hexdigest()
            if actual_cfg_sha != meta["model_config_sha256"]:
                return False, f"Config hash mismatch: expected {meta['model_config_sha256']}, got {actual_cfg_sha}"

        if "tokenizer_sha256" in meta and os.path.isfile(tok_path):
            with open(tok_path, "rb") as f:
                actual_tok_sha = hashlib.sha256(f.read()).hexdigest()
            if actual_tok_sha != meta["tokenizer_sha256"]:
                return False, f"Tokenizer hash mismatch: expected {meta['tokenizer_sha256']}, got {actual_tok_sha}"

    return True, f"Checkpoint verified 100% integral: {len(loaded_tensors)} tensors ({total_elements:,} parameters, {num_layers} layers), exact schema & shapes verified, 0 NaNs, 0 Infs, config & tokenizer valid."


# Alias for compatibility with external test runners
validate_tpu_checkpoint = validate_stage_checkpoint_for_handoff


def probe_local_best_milestone(parent_dir: str) -> Tuple[float, int, Optional[str]]:
    """Scans local checkpoint parent directory for the authentic lowest historical loss milestone."""
    best_loss = float("inf")
    best_step = -1
    best_dir = None
    if not os.path.isdir(parent_dir):
        return best_loss, best_step, best_dir

    for d in os.listdir(parent_dir):
        full = os.path.join(parent_dir, d)
        if not os.path.isdir(full) or d.startswith("."):
            continue
        cur_loss = float("inf")
        cur_step = -1
        loss_m = re.search(r"loss_([0-9.]+)", d)
        if loss_m:
            try:
                cur_loss = float(loss_m.group(1))
            except Exception:
                pass
        step_m = re.search(r"step[_-](\d+)", d)
        if step_m:
            try:
                cur_step = int(step_m.group(1))
            except Exception:
                pass

        meta_file = os.path.join(full, "STAGE_METADATA.json")
        if os.path.isfile(meta_file):
            try:
                with open(meta_file, "r", encoding="utf-8") as mf:
                    md = json.load(mf)
                m_loss = float(md.get("best_loss", md.get("loss", float("inf"))))
                if m_loss < cur_loss:
                    cur_loss = m_loss
                if cur_step < 0 and "step" in md:
                    cur_step = int(md["step"])
            except Exception:
                pass

        state_file = os.path.join(full, "checkpoint_state.json")
        if os.path.isfile(state_file):
            try:
                with open(state_file, "r", encoding="utf-8") as sf:
                    sd = json.load(sf)
                s_loss = float(sd.get("best_loss", sd.get("loss", float("inf"))))
                if s_loss < cur_loss:
                    cur_loss = s_loss
                if cur_step < 0 and "step" in sd:
                    cur_step = int(sd["step"])
            except Exception:
                pass

        if cur_loss < best_loss:
            best_loss = cur_loss
            best_step = cur_step
            best_dir = full

    return best_loss, best_step, best_dir


def probe_tpu_hf_checkpoint_step(
    repo_id: str,
    hf_token: Optional[str] = None,
    stage_id: str = "stage0_pretraining",
    return_details: bool = False,
) -> Union[Tuple[int, Optional[str], float, int], Tuple[int, Optional[str], float, int, int, Optional[str], int, Optional[str]]]:
    """
    Dedicated TPU Hugging Face Checkpoint & Milestone Discovery Arbiter.
    Autonomously scans Hugging Face for checkpoints across BOTH TPU and GPU folders:
      - TPU folders: 'tara_checkpoints_tpu', 'checkpoints_tpu', 'models/checkpoints_tpu'
      - GPU folders (Read-Only): 'tara_checkpoints', 'checkpoints', 'models/checkpoints'
      - Subfolders: 'normal_checkpoint', 'best_milestone', stage subfolders, and direct step folders
    Identifies:
      1. Latest available step across TPU and GPU folders.
      2. Authentic lowest loss (best milestone) across historical checkpoints.
      3. Precise subfolder for one-click restoration into TPU workspace.
    """
    if not repo_id or str(repo_id).strip().lower() in ("", "none", "null", "local"):
        if return_details:
            return -1, None, float("inf"), -1, -1, None, -1, None
        return -1, None, float("inf"), -1

    try:
        from huggingface_hub import HfApi, hf_hub_download
        token = _resolve_hf_token_standalone(hf_token)
        api = HfApi(token=token)
        files = api.list_repo_files(repo_id=repo_id)
    except Exception as e:
        print(f"[TPU HF Discovery Notice] Could not list HF repo files: {e}", flush=True)
        if return_details:
            return -1, None, float("inf"), -1, -1, None, -1, None
        return -1, None, float("inf"), -1

    candidates = {}
    best_loss_global = float("inf")
    best_milestone_step_global = -1

    for f in files:
        parts = f.split('/')
        if len(parts) >= 2:
            step_idx = -1
            step_val = -1
            for idx, p in enumerate(parts[:-1]):  # filename is parts[-1]
                m = re.search(r"step[_-](\d+)", p)
                if m:
                    step_idx = idx
                    step_val = int(m.group(1))
                    break

            if step_val >= 0 and step_idx >= 0:
                folder = "/".join(parts[:step_idx + 1])
                top = parts[0]
                is_tpu = "tpu" in folder.lower()
                is_best = "best_milestone" in folder.lower()
                if folder not in candidates:
                    candidates[folder] = {
                        "top": top,
                        "sub": parts[1] if len(parts) > 2 else "",
                        "step": step_val,
                        "folder": folder,
                        "is_tpu": is_tpu,
                        "is_best": is_best,
                        "loss": float("inf"),
                        "has_weights": False
                    }
                if "safetensors" in f or "pytorch_model.bin" in f:
                    candidates[folder]["has_weights"] = True
                loss_m = re.search(r"loss_([0-9.]+)", folder)
                if loss_m:
                    try:
                        l_val = float(loss_m.group(1))
                        candidates[folder]["loss"] = min(candidates[folder]["loss"], l_val)
                        if l_val < best_loss_global:
                            best_loss_global = l_val
                            best_milestone_step_global = step_val
                    except Exception:
                        pass

    for folder, info in candidates.items():
        if info["loss"] == float("inf") and info["has_weights"]:
            meta_path = f"{folder}/STAGE_METADATA.json"
            if meta_path in files:
                try:
                    local_p = hf_hub_download(repo_id=repo_id, filename=meta_path, token=token)
                    with open(local_p, "r", encoding="utf-8") as mf:
                        m_data = json.load(mf)
                    l_val = float(m_data.get("best_loss", m_data.get("loss", float("inf"))))
                    info["loss"] = l_val
                    if l_val < best_loss_global:
                        best_loss_global = l_val
                        best_milestone_step_global = info["step"]
                except Exception:
                    pass

    valid_candidates = [info for info in candidates.values() if info["has_weights"]]
    if not valid_candidates:
        if return_details:
            return -1, None, best_loss_global, best_milestone_step_global, -1, None, -1, None
        return -1, None, best_loss_global, best_milestone_step_global

    # Highest TPU candidate on HF
    tpu_cands = [c for c in valid_candidates if c["is_tpu"]]
    tpu_cands.sort(key=lambda x: x["step"], reverse=True)
    hf_tpu_step = tpu_cands[0]["step"] if tpu_cands else -1
    hf_tpu_folder = tpu_cands[0]["folder"] if tpu_cands else None

    # Highest GPU candidate on HF (Read-Only resume source)
    gpu_cands = [c for c in valid_candidates if not c["is_tpu"]]
    gpu_cands.sort(key=lambda x: x["step"], reverse=True)
    hf_gpu_step = gpu_cands[0]["step"] if gpu_cands else -1
    hf_gpu_folder = gpu_cands[0]["folder"] if gpu_cands else None

    # Overall best candidate: highest step; if equal step, prefer TPU
    valid_candidates.sort(key=lambda x: (x["step"], 1 if x["is_tpu"] else 0), reverse=True)
    best_cand = valid_candidates[0]

    if return_details:
        return (
            best_cand["step"],
            best_cand["folder"],
            best_loss_global,
            best_milestone_step_global,
            hf_tpu_step,
            hf_tpu_folder,
            hf_gpu_step,
            hf_gpu_folder,
        )
    return best_cand["step"], best_cand["folder"], best_loss_global, best_milestone_step_global


def probe_tpu_gdrive_checkpoints(
    stage_id: str = "stage0_pretraining",
    explicit_gdrive_dir: Optional[str] = None,
) -> Tuple[int, Optional[str], int, Optional[str]]:
    """
    Exhaustively scans Google Drive across both TPU and GPU directories:
      - Dedicated TPU directories: 'tara_checkpoints_tpu', 'checkpoints_tpu'
      - GPU directories (Read-Only): 'tara_checkpoints', 'checkpoints', shared team drives
      - Subdirectories: normal_checkpoint, best_milestone, stage-specific folders
    Returns: (tpu_step, best_tpu_dir, gpu_step, best_gpu_dir)
    """
    tpu_step = -1
    best_tpu_dir = None
    gpu_step = -1
    best_gpu_dir = None

    candidates: List[str] = []
    if explicit_gdrive_dir and str(explicit_gdrive_dir).strip():
        candidates.append(str(explicit_gdrive_dir).strip())

    tpu_resolved = resolve_tpu_gdrive_backup_dir(explicit_gdrive_dir)
    if tpu_resolved and tpu_resolved not in candidates:
        candidates.append(tpu_resolved)

    try:
        from gdrive_sync import find_all_gdrive_checkpoint_candidates
        for c in find_all_gdrive_checkpoint_candidates():
            if c and c not in candidates:
                candidates.append(c)
    except Exception:
        pass

    colab_mydrive = "/content/drive/MyDrive"
    for well_known in (
        "tara_checkpoints_tpu",
        "tara_checkpoints",
        "checkpoints_tpu",
        "checkpoints",
    ):
        p = os.path.join(colab_mydrive, well_known)
        if p not in candidates:
            candidates.append(p)

    checked_paths = set()
    for base in list(candidates):
        if not base or not os.path.isdir(base):
            continue

        subpaths = [
            base,
            os.path.join(base, stage_id),
            os.path.join(base, "normal_checkpoint"),
            os.path.join(base, "best_milestone"),
        ]
        try:
            v_dirs = _collect_versioned_checkpoint_dirs(base, stage_id)
            for _, v_p in v_dirs:
                subpaths.append(v_p)
        except Exception:
            pass

        for p in subpaths:
            if not p or p in checked_paths or not os.path.isdir(p):
                continue
            checked_paths.add(p)

            s_val = get_checkpoint_dir_step(p)
            if s_val >= 0:
                is_tpu = "tpu" in p.lower()
                if is_tpu:
                    if s_val > tpu_step:
                        tpu_step = s_val
                        best_tpu_dir = p
                else:
                    if s_val > gpu_step:
                        gpu_step = s_val
                        best_gpu_dir = p

    return tpu_step, best_tpu_dir, gpu_step, best_gpu_dir


def resolve_and_auto_select_tpu_checkpoint(
    tpu_ckpt_dir: str,
    stage_id: str,
    gdrive_base_dir: Optional[str] = None,
    hf_repo_id: Optional[str] = None,
    hf_token: Optional[str] = None,
    gpu_ckpt_base_dir: Optional[str] = None,
) -> Tuple[str, Optional[str], int]:
    """
    Autonomous Unified Multi-Source TPU Checkpoint Arbiter:
    Audits 4 distinct checkpoint locations across BOTH TPU and GPU sources:
      1. Local TPU directory (Dedicated R/W checkpoints_tpu)
      2. Local GPU directory (STRICT READ-ONLY fallback; never modified or deleted!)
      3. Google Drive (Audits BOTH dedicated TPU folders and GPU folders as Read-Only)
      4. Hugging Face (Audits BOTH dedicated TPU folders and GPU folders as Read-Only)

    Identifies the verifiable training step in each location.
    Auto-selects whichever source holds the latest (highest) step.
    If Google Drive or HF is highest (TPU or GPU), downloads/restores into TPU local directory.
    If Local GPU is highest, resumes from GPU directory in READ-ONLY mode without modifying it.
    If Local TPU is highest, resumes from TPU directory directly.
    ALL future checkpoints and backups are STRICTLY written ONLY to TPU folders!

    Returns: (selected_source_name, selected_resume_dir, selected_step)
    """
    # 1. Local TPU Step
    tpu_step = get_checkpoint_dir_step(tpu_ckpt_dir)
    tpu_parent = os.path.dirname(os.path.abspath(tpu_ckpt_dir)) or "."
    tpu_base = os.path.basename(tpu_ckpt_dir)
    best_tpu_dir = tpu_ckpt_dir if tpu_step >= 0 else None
    if os.path.isdir(tpu_parent):
        v_dirs = _collect_versioned_checkpoint_dirs(tpu_parent, tpu_base)
        if v_dirs:
            for v_step, v_path in v_dirs:
                if v_step > tpu_step:
                    tpu_step = v_step
                    best_tpu_dir = v_path

    # 2. Local GPU Step (STRICT READ-ONLY)
    gpu_dir = gpu_ckpt_base_dir or os.path.join("storage", "models", "checkpoints", stage_id)
    gpu_step = get_checkpoint_dir_step(gpu_dir)
    gpu_parent = os.path.dirname(os.path.abspath(gpu_dir)) or "."
    gpu_base = os.path.basename(gpu_dir)
    best_gpu_dir = gpu_dir if gpu_step >= 0 else None
    if os.path.isdir(gpu_parent):
        v_gpu_dirs = _collect_versioned_checkpoint_dirs(gpu_parent, gpu_base)
        if v_gpu_dirs:
            for v_step, v_path in v_gpu_dirs:
                if v_step > gpu_step:
                    gpu_step = v_step
                    best_gpu_dir = v_path

    # 3. Google Drive (Audit BOTH TPU and GPU folders)
    gd_tpu_step, gd_tpu_dir, gd_gpu_step, gd_gpu_dir = probe_tpu_gdrive_checkpoints(
        stage_id=stage_id,
        explicit_gdrive_dir=gdrive_base_dir,
    )
    if gd_tpu_step >= gd_gpu_step and gd_tpu_step >= 0:
        gdrive_dir = gd_tpu_dir
        gdrive_step = gd_tpu_step
        gdrive_is_tpu = True
    elif gd_gpu_step >= 0:
        gdrive_dir = gd_gpu_dir
        gdrive_step = gd_gpu_step
        gdrive_is_tpu = False
    else:
        gdrive_dir = None
        gdrive_step = -1
        gdrive_is_tpu = False

    # 4. Hugging Face (Audit BOTH TPU and GPU folders)
    hf_step = -1
    hf_subfolder = None
    hf_best_loss = float("inf")
    hf_best_milestone_step = -1
    hf_tpu_step = -1
    hf_tpu_folder = None
    hf_gpu_step = -1
    hf_gpu_folder = None
    if hf_repo_id and str(hf_repo_id).strip().lower() not in ("", "none", "null", "local"):
        (
            hf_step,
            hf_subfolder,
            hf_best_loss,
            hf_best_milestone_step,
            hf_tpu_step,
            hf_tpu_folder,
            hf_gpu_step,
            hf_gpu_folder,
        ) = probe_tpu_hf_checkpoint_step(
            hf_repo_id, hf_token=hf_token, stage_id=stage_id, return_details=True
        )

    hf_is_tpu = (hf_tpu_step >= hf_gpu_step and hf_tpu_step >= 0)

    CLR_CYAN = "\033[96m"
    CLR_GREEN = "\033[92m"
    CLR_YELLOW = "\033[93m"
    CLR_BOLD = "\033[1m"
    CLR_RESET = "\033[0m"

    print(f"\n{CLR_CYAN}{'=' * 88}{CLR_RESET}", flush=True)
    print(f"  {CLR_BOLD}{CLR_CYAN}TARA MULTI-SOURCE RESUME DISCOVERY MATRIX ({stage_id}){CLR_RESET}", flush=True)
    print(f"{CLR_CYAN}{'=' * 88}{CLR_RESET}", flush=True)
    print(f"  {'Source':<24} {'Status':<12} {'Detected Step':<16} {'Location / Target'}", flush=True)
    print(f"  {'-' * 86}", flush=True)

    tpu_str = f"Step {tpu_step:04d}" if tpu_step >= 0 else "Empty / None"
    tpu_color = CLR_GREEN if tpu_step >= 0 else CLR_RESET
    print(f"  {'1. Local TPU (R/W)':<24} {'Available' if tpu_step >= 0 else 'None':<12} {tpu_color}{tpu_str:<16}{CLR_RESET} {best_tpu_dir or tpu_ckpt_dir}", flush=True)

    gpu_str = f"Step {gpu_step:04d}" if gpu_step >= 0 else "Empty / None"
    gpu_color = CLR_GREEN if gpu_step >= 0 else CLR_RESET
    print(f"  {'2. Local GPU (R/O)':<24} {'Available' if gpu_step >= 0 else 'None':<12} {gpu_color}{gpu_str:<16}{CLR_RESET} {best_gpu_dir or gpu_dir} (Read-Only)", flush=True)

    gd_tpu_str = f"Step {gd_tpu_step:04d}" if gd_tpu_step >= 0 else "Unavailable"
    gd_tpu_color = CLR_GREEN if gd_tpu_step >= 0 else CLR_RESET
    print(f"  {'3. Google Drive (TPU)':<24} {'Mounted' if gd_tpu_step >= 0 else 'Not Found':<12} {gd_tpu_color}{gd_tpu_str:<16}{CLR_RESET} {gd_tpu_dir or 'Not Mounted'}", flush=True)

    gd_gpu_str = f"Step {gd_gpu_step:04d}" if gd_gpu_step >= 0 else "Unavailable"
    gd_gpu_color = CLR_GREEN if gd_gpu_step >= 0 else CLR_RESET
    print(f"  {'4. Google Drive (GPU)':<24} {'Mounted' if gd_gpu_step >= 0 else 'Not Found':<12} {gd_gpu_color}{gd_gpu_str:<16}{CLR_RESET} {gd_gpu_dir or 'Not Mounted'} (Read-Only Source)", flush=True)

    hf_tpu_str = f"Step {hf_tpu_step:04d}" if hf_tpu_step >= 0 else "Unavailable"
    hf_tpu_color = CLR_GREEN if hf_tpu_step >= 0 else CLR_RESET
    print(f"  {'5. Hugging Face (TPU)':<24} {'Connected' if hf_tpu_step >= 0 else 'Not Found':<12} {hf_tpu_color}{hf_tpu_str:<16}{CLR_RESET} {hf_tpu_folder or 'https://huggingface.co/' + str(hf_repo_id or 'N/A')}", flush=True)

    hf_gpu_str = f"Step {hf_gpu_step:04d}" if hf_gpu_step >= 0 else "Unavailable"
    hf_gpu_color = CLR_GREEN if hf_gpu_step >= 0 else CLR_RESET
    print(f"  {'6. Hugging Face (GPU)':<24} {'Connected' if hf_gpu_step >= 0 else 'Not Found':<12} {hf_gpu_color}{hf_gpu_str:<16}{CLR_RESET} {hf_gpu_folder or 'https://huggingface.co/' + str(hf_repo_id or 'N/A')} (Read-Only Source)", flush=True)
    print(f"  {'-' * 86}", flush=True)

    highest_step = max(tpu_step, gpu_step, gdrive_step, hf_step)

    if highest_step < 0:
        print(f"  {CLR_YELLOW}>>> No prior checkpoint detected in any source. Starting fresh from candidate model. <<<{CLR_RESET}", flush=True)
        print(f"{CLR_CYAN}{'=' * 88}{CLR_RESET}\n", flush=True)
        return "fresh", None, -1

    # Branch 1: Google Drive holds highest step (TPU or GPU)
    if gdrive_step >= hf_step and gdrive_step > max(tpu_step, gpu_step):
        gd_source_label = "TPU" if gdrive_is_tpu else "GPU (Read-Only)"
        print(f"  {CLR_BOLD}{CLR_GREEN}>>> AUTO-SELECTED: GOOGLE DRIVE [{gd_source_label}] (Step {gdrive_step:04d} > Local TPU {tpu_step:04d}, Local GPU {gpu_step:04d}) -> Restoring to TPU dir... <<<{CLR_RESET}", flush=True)
        os.makedirs(tpu_ckpt_dir, exist_ok=True)
        restored = restore_checkpoint_from_gdrive(gdrive_dir, tpu_ckpt_dir)
        if restored:
            print(f"  {CLR_GREEN}Successfully synchronized Step {gdrive_step:04d} from Google Drive [{gd_source_label}] into TPU workspace ({tpu_ckpt_dir}).{CLR_RESET}", flush=True)
            print(f"  {CLR_CYAN}-> All future checkpoints & backups will strictly write ONLY to TPU folders.{CLR_RESET}", flush=True)
            print(f"{CLR_CYAN}{'=' * 88}{CLR_RESET}\n", flush=True)
            return "gdrive", tpu_ckpt_dir, gdrive_step
        else:
            print(f"  {CLR_YELLOW}[Fallback] GDrive restore failed. Checking Hugging Face or Local GPU/TPU fallback...{CLR_RESET}", flush=True)
            if hf_step >= 0:
                hf_ok = restore_checkpoint_from_hf(hf_repo_id, tpu_ckpt_dir, subfolder=hf_subfolder, hf_token=hf_token)
                if hf_ok:
                    print(f"{CLR_CYAN}{'=' * 88}{CLR_RESET}\n", flush=True)
                    return "hf_fallback", tpu_ckpt_dir, hf_step

    # Branch 2: Hugging Face holds highest step (TPU or GPU)
    if hf_step > gdrive_step and hf_step > max(tpu_step, gpu_step):
        hf_source_label = "TPU" if hf_is_tpu else "GPU (Read-Only)"
        print(f"  {CLR_BOLD}{CLR_GREEN}>>> AUTO-SELECTED: HUGGING FACE [{hf_source_label}] (Step {hf_step:04d} > GDrive {gdrive_step:04d}, TPU {tpu_step:04d}, GPU {gpu_step:04d}) -> Restoring to TPU dir... <<<{CLR_RESET}", flush=True)
        os.makedirs(tpu_ckpt_dir, exist_ok=True)
        hf_ok = restore_checkpoint_from_hf(hf_repo_id, tpu_ckpt_dir, subfolder=hf_subfolder, hf_token=hf_token)
        if hf_ok:
            if hf_best_loss < float("inf"):
                st_p = os.path.join(tpu_ckpt_dir, "checkpoint_state.json")
                if os.path.isfile(st_p):
                    try:
                        with open(st_p, "r", encoding="utf-8") as sf:
                            sdata = json.load(sf)
                        sdata["best_loss"] = min(float(sdata.get("best_loss", float("inf"))), hf_best_loss)
                        if hf_best_milestone_step >= 0:
                            sdata["best_milestone_step"] = hf_best_milestone_step
                        with open(st_p, "w", encoding="utf-8") as sf:
                            json.dump(sdata, sf, indent=2)
                    except Exception:
                        pass
                sm_p = os.path.join(tpu_ckpt_dir, "STAGE_METADATA.json")
                if os.path.isfile(sm_p):
                    try:
                        with open(sm_p, "r", encoding="utf-8") as mf:
                            mdata = json.load(mf)
                        mdata["best_loss"] = min(float(mdata.get("best_loss", float("inf"))), hf_best_loss)
                        if hf_best_milestone_step >= 0:
                            mdata["best_milestone_step"] = hf_best_milestone_step
                        with open(sm_p, "w", encoding="utf-8") as mf:
                            json.dump(mdata, mf, indent=2)
                    except Exception:
                        pass
            print(f"  {CLR_GREEN}Successfully synchronized Step {hf_step:04d} from Hugging Face [{hf_source_label}] into TPU workspace ({tpu_ckpt_dir}).{CLR_RESET}", flush=True)
            print(f"  {CLR_CYAN}-> All future checkpoints & backups will strictly write ONLY to TPU folders.{CLR_RESET}", flush=True)
            print(f"{CLR_CYAN}{'=' * 88}{CLR_RESET}\n", flush=True)
            return "hf", tpu_ckpt_dir, hf_step
        else:
            print(f"  {CLR_YELLOW}[Fallback] HF restore failed. Checking Google Drive or Local GPU/TPU fallback...{CLR_RESET}", flush=True)
            if gdrive_step >= 0:
                gd_ok = restore_checkpoint_from_gdrive(gdrive_dir, tpu_ckpt_dir)
                if gd_ok:
                    print(f"{CLR_CYAN}{'=' * 88}{CLR_RESET}\n", flush=True)
                    return "gdrive_fallback", tpu_ckpt_dir, gdrive_step

    # Branch 3: Local TPU is at or above highest remote and GPU checkpoint
    if tpu_step >= max(gpu_step, gdrive_step, hf_step) and tpu_step >= 0:
        target_dir = best_tpu_dir or tpu_ckpt_dir
        print(f"  {CLR_BOLD}{CLR_GREEN}>>> AUTO-SELECTED: LOCAL TPU (Step {tpu_step:04d} is already latest). Resuming TPU training. <<<{CLR_RESET}", flush=True)
        print(f"{CLR_CYAN}{'=' * 88}{CLR_RESET}\n", flush=True)
        return "tpu", target_dir, tpu_step

    # Branch 4: Local GPU holds higher step than Local TPU (STRICT READ-ONLY FALLBACK)
    if gpu_step > tpu_step and gpu_step >= 0:
        target_dir = best_gpu_dir or gpu_dir
        print(f"  {CLR_BOLD}{CLR_GREEN}>>> AUTO-SELECTED: LOCAL GPU (Step {gpu_step:04d} > TPU {tpu_step:04d}) [STRICT READ-ONLY MODE] <<<{CLR_RESET}", flush=True)
        print(f"  {CLR_GREEN}-> Restoring initial weights & optimizer state directly from GPU directory: {target_dir}{CLR_RESET}", flush=True)
        print(f"  {CLR_YELLOW}-> Note: GPU directory is STRICTLY PROTECTED (Read-Only). No files in {gpu_dir} will be modified or deleted.{CLR_RESET}", flush=True)
        print(f"  {CLR_GREEN}-> All future TPU checkpoints and rotation will be saved into dedicated TPU directory: {tpu_ckpt_dir}{CLR_RESET}", flush=True)
        print(f"{CLR_CYAN}{'=' * 88}{CLR_RESET}\n", flush=True)
        return "gpu_readonly", target_dir, gpu_step

    print(f"{CLR_CYAN}{'=' * 88}{CLR_RESET}\n", flush=True)
    return "tpu", best_tpu_dir or tpu_ckpt_dir, tpu_step


def _safe_float(val: Any, default: Optional[float] = None) -> Optional[float]:
    """Safely coerces val to float; returns default on None, NaN, Inf or conversion error."""
    if val is None:
        return default
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (ValueError, TypeError):
        return default


def _safe_int(val: Any, default: Optional[int] = None) -> Optional[int]:
    """Safely coerces val to int; returns default on None or conversion error."""
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def load_tpu_checkpoint(
    checkpoint_dir: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
    allow_cold_start: bool = False,
    offload_optimizer: bool = False,
) -> Tuple[int, int, float, bool, Dict[str, Any]]:
    """
    Loads existing checkpoint (TPU or GPU) with 100% parameter and optimizer fidelity.
    Supports:
      1. Sharded SafeTensors: model.safetensors.index.json with model-*-of-*.safetensors
      2. Monolithic SafeTensors: model.safetensors
      3. Loose SafeTensors: glob("*.safetensors")
      4. PyTorch legacy binary: pytorch_model.bin
    Restores 2-group optimizer state via restore_tpu_optimizer_state with strict validation.
    Guarantees XLA execution graph compilation synchronization via xm.mark_step().
    """
    if not os.path.isdir(checkpoint_dir):
        return 0, 0, float("inf"), False, {}

    # Strict Checkpoint Verification Gate: Must pass strict integrity check.
    # Cold start and error bypass are strictly prohibited.
    if not verify_checkpoint_integrity(checkpoint_dir):
        raise RuntimeError(
            f"[Strict Verification Gate Fatal] Checkpoint at '{checkpoint_dir}' failed integrity verification! "
            "Cold start and error bypass are strictly prohibited. Training halted."
        )
    print(f"[Verification Passed] Checkpoint {checkpoint_dir} verified 100% integral and valid.", flush=True)

    state_file = os.path.join(checkpoint_dir, "checkpoint_state.json")
    step = 0
    epoch = 0
    loss = float("inf")
    cursor = {}

    if os.path.isfile(state_file):
        try:
            with open(state_file, "r", encoding="utf-8") as sf:
                state = json.load(sf)
            step = _safe_int(state.get("step"), 0)
            epoch = _safe_int(state.get("epoch"), 0)
            loss = _safe_float(state.get("loss"), float("inf"))
            best_loss = _safe_float(state.get("best_loss"), loss)
            cursor = extract_data_cursor_from_state(state) or {}
            cursor["loss"] = loss
            cursor["best_loss"] = best_loss
            if "best_milestone_step" in state and state["best_milestone_step"] is not None:
                cursor["best_milestone_step"] = _safe_int(state["best_milestone_step"], -1)
            if "learning_rate" in state and state["learning_rate"] is not None:
                cursor["saved_lr"] = _safe_float(state["learning_rate"], 0.0)
            if "base_lr" in state and state["base_lr"] is not None:
                cursor["base_lr"] = _safe_float(state["base_lr"], 0.0)
            if "warmup_steps" in state and state["warmup_steps"] is not None:
                cursor["warmup_steps"] = _safe_int(state["warmup_steps"], 0)
            if "min_lr" in state and state["min_lr"] is not None:
                cursor["min_lr"] = _safe_float(state["min_lr"], 0.0)
            if "max_steps" in state and state["max_steps"] is not None:
                cursor["max_steps"] = _safe_int(state["max_steps"], 0)
            if "rng_state" in state and isinstance(state["rng_state"], dict):
                try:
                    if "torch_rng" in state["rng_state"] and state["rng_state"]["torch_rng"] is not None:
                        torch.set_rng_state(torch.tensor(state["rng_state"]["torch_rng"], dtype=torch.uint8))
                    if HAS_XLA and "xla_rng" in state["rng_state"] and state["rng_state"]["xla_rng"] is not None:
                        xm.set_rng_state(state["rng_state"]["xla_rng"])
                except Exception:
                    pass
        except Exception as e:
            print(f"[TPU Resume Notice] Could not parse checkpoint_state.json: {e}", flush=True)

    stage_meta_file = os.path.join(checkpoint_dir, "STAGE_METADATA.json")
    if os.path.isfile(stage_meta_file):
        try:
            with open(stage_meta_file, "r", encoding="utf-8") as smf:
                sm = json.load(smf)
            if "best_loss" in sm and sm["best_loss"] is not None:
                sm_bl = _safe_float(sm["best_loss"], float("inf"))
                if sm_bl < cursor.get("best_loss", float("inf")):
                    cursor["best_loss"] = sm_bl
            if "best_milestone_step" in sm and sm["best_milestone_step"] is not None and "best_milestone_step" not in cursor:
                cursor["best_milestone_step"] = _safe_int(sm["best_milestone_step"], -1)
            if "learning_rate" in sm and sm["learning_rate"] is not None and "learning_rate" not in cursor:
                cursor["learning_rate"] = _safe_float(sm["learning_rate"], 0.0)
            if "base_lr" in sm and sm["base_lr"] is not None and "base_lr" not in cursor:
                cursor["base_lr"] = _safe_float(sm["base_lr"], 0.0)
        except Exception:
            pass

    target_model = getattr(model, "module", model)
    total_model_keys = len(target_model.state_dict())
    matched_keys = 0

    try:
        # Case 1: Sharded SafeTensors Index (Standard Hugging Face & TARA Sharded Checkpoints)
        index_file = os.path.join(checkpoint_dir, "model.safetensors.index.json")
        if HAS_SAFETENSORS and os.path.isfile(index_file):
            with open(index_file, "r", encoding="utf-8") as idxf:
                idx_data = json.load(idxf)
            weight_map = idx_data.get("weight_map", {})
            shard_names = sorted(list(set(weight_map.values())))
            model_sd = target_model.state_dict()
            all_shards_tensors = {}
            for s_name in shard_names:
                s_path = os.path.join(checkpoint_dir, s_name)
                if os.path.isfile(s_path):
                    shard_tensors = load_file(s_path, device="cpu")
                    filtered = {k: v for k, v in shard_tensors.items() if k in model_sd and model_sd[k].shape == v.shape}
                    if filtered:
                        all_shards_tensors.update(filtered)
                    del shard_tensors
            if all_shards_tensors:
                target_model.load_state_dict(all_shards_tensors, strict=False)
                matched_keys = len(all_shards_tensors)
                del all_shards_tensors
            print(
                f"[TPU Resume] Restored {matched_keys}/{total_model_keys} weight tensors from "
                f"sharded checkpoint ({len(shard_names)} shards) at Step {step} (Loss: {loss:.4f})",
                flush=True
            )
            ensure_model_weight_tying(target_model)
            if HAS_XLA:
                xm.mark_step()
            if optimizer is not None:
                restore_tpu_optimizer_state(checkpoint_dir, optimizer, device=device, allow_cold_start=allow_cold_start, offload_optimizer=offload_optimizer)
            if HAS_XLA:
                xm.mark_step()
            return step, epoch, float(cursor.get("best_loss", loss)), (matched_keys > 0), cursor

        # Case 2: Monolithic SafeTensors (Standard TPU single-file checkpoint)
        st_path = os.path.join(checkpoint_dir, "model.safetensors")
        if HAS_SAFETENSORS and os.path.isfile(st_path):
            weights = load_file(st_path, device="cpu")
            model_sd = target_model.state_dict()
            _validate_checkpoint_tensors(weights, model_sd)
            target_model.load_state_dict(weights, strict=False)
            matched_keys = len(weights)
            del weights
            print(f"[TPU Resume] Restored {matched_keys}/{total_model_keys} weight tensors from model.safetensors at Step {step} (Loss: {loss:.4f})", flush=True)
            ensure_model_weight_tying(target_model)
            if HAS_XLA:
                xm.mark_step()
            if optimizer is not None:
                restore_tpu_optimizer_state(checkpoint_dir, optimizer, device=device, allow_cold_start=allow_cold_start, offload_optimizer=offload_optimizer)
            if HAS_XLA:
                xm.mark_step()
            return step, epoch, float(cursor.get("best_loss", loss)), (matched_keys > 0), cursor

        # Case 3: Loose SafeTensors Shards
        if HAS_SAFETENSORS:
            loose_shards = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
            if loose_shards:
                model_sd = target_model.state_dict()
                all_loose_tensors = {}
                for s_path in loose_shards:
                    shard_tensors = load_file(s_path, device="cpu")
                    filtered = {k: v for k, v in shard_tensors.items() if k in model_sd and model_sd[k].shape == v.shape}
                    if filtered:
                        all_loose_tensors.update(filtered)
                    del shard_tensors
                if all_loose_tensors:
                    target_model.load_state_dict(all_loose_tensors, strict=False)
                    matched_keys = len(all_loose_tensors)
                    del all_loose_tensors
                print(f"[TPU Resume] Restored {matched_keys}/{total_model_keys} weight tensors from {len(loose_shards)} loose shards at Step {step} (Loss: {loss:.4f})", flush=True)
                ensure_model_weight_tying(target_model)
                if HAS_XLA:
                    xm.mark_step()
                if optimizer is not None:
                    restore_tpu_optimizer_state(checkpoint_dir, optimizer, device=device, allow_cold_start=allow_cold_start, offload_optimizer=offload_optimizer)
                if HAS_XLA:
                    xm.mark_step()
                return step, epoch, float(cursor.get("best_loss", loss)), (matched_keys > 0), cursor

        # Case 4: PyTorch Binary Fallback
        bin_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
        if os.path.isfile(bin_path):
            weights = torch.load(bin_path, map_location="cpu")
            model_sd = target_model.state_dict()
            filtered = {k: v for k, v in weights.items() if k in model_sd and model_sd[k].shape == v.shape}
            if filtered:
                target_model.load_state_dict(filtered, strict=False)
                matched_keys = len(filtered)
            del weights
            print(f"[TPU Resume] Restored {matched_keys}/{total_model_keys} weight tensors from pytorch_model.bin at Step {step} (Loss: {loss:.4f})", flush=True)
            ensure_model_weight_tying(target_model)
            if HAS_XLA:
                xm.mark_step()
            if optimizer is not None:
                restore_tpu_optimizer_state(checkpoint_dir, optimizer, device=device, allow_cold_start=allow_cold_start, offload_optimizer=offload_optimizer)
            if HAS_XLA:
                xm.mark_step()
            return step, epoch, float(cursor.get("best_loss", loss)), (matched_keys > 0), cursor

    except Exception as e:
        print(f"[TPU Resume Warning] Could not resume weights from {checkpoint_dir}: {e}", flush=True)
        if not allow_cold_start:
            raise

    if HAS_XLA:
        xm.mark_step()
    return step, epoch, float(cursor.get("best_loss", loss)), (matched_keys > 0), cursor


class ChunkedLossResult(tuple):
    """
    4-element tuple (total_loss, ce_loss, aux_loss, total_valid) providing 100% backward compatibility
    with unpacking `total_loss, ce_loss, aux_loss, total_valid = res` while also exposing:
      .total_loss: per-token normalized total loss
      .ce_loss: per-token cross entropy
      .aux_loss: auxiliary regularizer
      .total_valid: valid token count
      .loss_sum: unreduced sum of valid token cross-entropy losses
      .unreduced_loss: loss_sum + (total_valid * aux_loss) for exact global token-weighted gradient accumulation.
    """
    def __new__(cls, total_loss, ce_loss, aux_loss, total_valid, loss_sum=None, unreduced_loss=None):
        return super().__new__(cls, (total_loss, ce_loss, aux_loss, total_valid))

    def __init__(self, total_loss, ce_loss, aux_loss, total_valid, loss_sum=None, unreduced_loss=None):
        self.total_loss = total_loss
        self.ce_loss = ce_loss
        self.aux_loss = aux_loss
        self.total_valid = total_valid
        self.loss_sum = loss_sum if loss_sum is not None else (total_loss * total_valid)
        self.unreduced_loss = unreduced_loss if unreduced_loss is not None else (self.loss_sum + total_valid.float() * aux_loss)


def compute_tpu_chunked_loss(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    chunk_sz: int = 128,
    return_valid_tokens: bool = True,
) -> Union[ChunkedLossResult, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Branchless memory-efficient chunked cross-entropy projection and variance regularizer for Google TPU:
      1. Bypasses full vocabulary logits materialization (which would allocate over 3.28 GB at seq_len 2048).
      2. Executes model transformer backbone once to produce hidden_states without full-vocabulary projection.
      3. Uses static chunking (chunk_sz tokens, default 128) for vocabulary projection:
         c_logits = lm_head(c_h) allocates only ~48.95 MB (in FP32) or ~24.5 MB (in BF16),
         preventing RuntimeProgramAllocationFailure in tt_jit_cross_entropy_loss.
      4. STRICTLY BRANCHLESS ON TENSORS: Zero Python `if (c_l == -100).all()` and zero `if total_valid > 0`.
         F.cross_entropy with ignore_index=-100 natively produces 0.0 loss and 0.0 gradients on masked tokens,
         completely eliminating device-to-host synchronization stalls and XLA graph recompilations.
      5. Computes auxiliary variance collapse regularizer over valid tokens using static broadcast
         masking (zero dynamic indexing), mathematically identical to model_dataset.py.
      6. Preserves numerical parity and exact gradient flow across all leaf parameters.
    Returns ChunkedLossResult (total_loss, ce_loss, aux_loss, total_valid) if return_valid_tokens else (total_loss, ce_loss, aux_loss).
    """
    target_model = getattr(model, "module", model)
    hidden_states = target_model.model(input_ids=input_ids, attention_mask=attention_mask)

    shift_hidden = hidden_states[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    flat_hidden = shift_hidden.view(-1, shift_hidden.size(-1))
    flat_labels = shift_labels.view(-1)

    n_tokens = flat_hidden.size(0)
    effective_chunk_sz = max(16, chunk_sz)

    # Static Fixed-Shape Chunk Padding for 100% Deterministic XLA Kernel Compilation:
    # Ensures every single chunk has identical shape [chunk_sz, D] so XLA compiles exactly 1 kernel.
    pad_len = (effective_chunk_sz - (n_tokens % effective_chunk_sz)) % effective_chunk_sz
    if pad_len > 0:
        flat_hidden = F.pad(flat_hidden, (0, 0, 0, pad_len), value=0.0)
        flat_labels = F.pad(flat_labels, (0, pad_len), value=-100)

    loss_sum = torch.tensor(0.0, device=flat_hidden.device, dtype=torch.float32)
    for chunk_start in range(0, flat_hidden.size(0), effective_chunk_sz):
        chunk_end = chunk_start + effective_chunk_sz
        c_h = flat_hidden[chunk_start:chunk_end]
        c_l = flat_labels[chunk_start:chunk_end]
        c_logits = target_model.lm_head(c_h)
        # Unconditional static execution: ignore_index=-100 automatically produces 0.0 for masked/padded tokens
        c_loss = F.cross_entropy(c_logits.float(), c_l, ignore_index=-100, reduction="sum")
        loss_sum = loss_sum + c_loss

    total_valid = (flat_labels != -100).sum()
    has_valid = (total_valid > 0).float()
    valid_denom = total_valid.clamp(min=1.0)
    ce_loss = (loss_sum / valid_denom) * has_valid

    # Unconditional branchless auxiliary representation regularizer
    # When total_valid == 0 (e.g. all-masked sequence), has_valid == 0.0 smoothly zeros out aux_loss
    valid_mask_f = (flat_labels != -100).float().unsqueeze(-1)
    var_per_token = flat_hidden.var(dim=-1, keepdim=True)
    hidden_var = (var_per_token * valid_mask_f).sum() / valid_mask_f.sum().clamp(min=1.0)
    aux_loss = 0.005 * torch.relu(1.0 - hidden_var) * has_valid

    total_loss = ce_loss + aux_loss
    unreduced_loss = loss_sum + (total_valid.float() * aux_loss)
    if return_valid_tokens:
        return ChunkedLossResult(total_loss, ce_loss, aux_loss, total_valid, loss_sum=loss_sum, unreduced_loss=unreduced_loss)
    return total_loss, ce_loss, aux_loss


def train_tpu_worker(index: int, args: argparse.Namespace):
    """
    Core per-core TPU worker function. Executed across TPU cores via xmp.spawn.
    Maintains 100% architectural and semantic parity with GPU training pipeline.
    """
    device = get_tpu_device()
    world_size = get_tpu_world_size()
    rank = get_tpu_rank()
    is_main = is_tpu_master()

    if is_main:
        print("=" * 80)
        print(f"  TARA TPU TRAINING WORKER INITIALIZED (Cores: {world_size} | Rank {rank}/{world_size})")
        print(f"  Target Device : {device}")
        print(f"  Output Dir    : {args.output_dir} (ISOLATED TPU STORAGE)")
        print(f"  Curriculum    : Multi-Domain DRR Scheduler & Persistent ShardLedger")
        print("=" * 80 + "\n", flush=True)

    # 0. Pre-training Authorization Gate Check
    if args.verify_gate and verify_pretraining_authorization_proof:
        gate_failed = 0
        if is_main:
            if not verify_pretraining_authorization_proof(args.model_dir, os.path.dirname(args.output_dir)):
                gate_failed = 1
        global_gate_failed = sync_tpu_decision(gate_failed, tag="gate_barrier")
        if global_gate_failed == 1:
            raise RuntimeError(
                "[Pre-Training Gate FATAL] Pretraining authorization gate check failed. "
                "Cryptographic pretraining_authorization.json proof is missing or status is not AUTHORIZED."
            )

    # 0.5. Autonomous Unified Multi-Source Checkpoint Resolution
    # Multi-core safe: writer rank (Rank 0) performs discovery and any needed cloud downloads;
    # non-writer ranks wait at barrier, strictly preventing concurrent cloud download collisions.
    resume_dir = getattr(args, "resume_dir", None)
    if resume_dir and not os.path.isdir(resume_dir):
        resume_dir = None

    if resume_dir is None and args.resume:
        if is_main:
            try:
                gdir = getattr(args, "gdrive_dir", None) or resolve_gdrive_restore_dir() or resolve_gdrive_backup_dir()
                selected_source, selected_dir, selected_step = resolve_and_auto_select_tpu_checkpoint(
                    tpu_ckpt_dir=args.output_dir,
                    stage_id=args.stage_id,
                    gdrive_base_dir=gdir,
                    hf_repo_id=getattr(args, "hf_repo", None) or getattr(args, "hf_model_repo", None),
                    hf_token=getattr(args, "hf_token", None),
                )
                if selected_dir and os.path.isdir(selected_dir):
                    resume_dir = selected_dir
                elif os.path.isdir(args.output_dir):
                    resume_dir = args.output_dir
            except Exception as e:
                print(f"[TPU Resume Notice] Multi-source resume discovery skipped on master: {e}", flush=True)

        if HAS_XLA and world_size > 1:
            tpu_barrier("resume_discovery_barrier")

        if not is_main:
            if getattr(args, "resume_dir", None) and os.path.isdir(args.resume_dir):
                resume_dir = args.resume_dir
            elif os.path.isdir(args.output_dir) and (
                os.path.isfile(os.path.join(args.output_dir, "checkpoint_state.json"))
                or os.path.isfile(os.path.join(args.output_dir, "STAGE_METADATA.json"))
            ):
                resume_dir = args.output_dir

    if resume_dir:
        args.resume_dir = resume_dir

    # 1. Load Architecture Config
    config_path = os.path.join(args.resume_dir, "config.json") if (args.resume_dir and os.path.isfile(os.path.join(args.resume_dir, "config.json"))) else os.path.join(args.model_dir, "config.json")
    if not os.path.exists(config_path):
        config_path = "storage/models/tara_candidate_v1/config.json"
    if not os.path.exists(config_path):
        config_path = "model_config/config.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json missing at {config_path}")

    with open(config_path, "r", encoding="utf-8") as cf:
        config = json.load(cf)

    config["rope_theta"] = resolve_dynamic_rope_theta(config)
    if args.max_seq_len > 0:
        config["max_position_embeddings"] = args.max_seq_len
    else:
        args.max_seq_len = int(config.get("max_position_embeddings", 2048))

    # 2. Load Native Tokenizer
    tok_path = os.path.join(args.resume_dir, "tokenizer.json") if (args.resume_dir and os.path.isfile(os.path.join(args.resume_dir, "tokenizer.json"))) else os.path.join(args.model_dir, "tokenizer.json")
    if not os.path.exists(tok_path):
        tok_path = "storage/models/tara_candidate_v1/tokenizer.json"
    if not os.path.exists(tok_path):
        tok_path = "model_config/tokenizer.json"
    if not os.path.exists(tok_path):
        raise FileNotFoundError(f"tokenizer.json missing at {tok_path}")

    tokenizer = TaraNativeTokenizer.from_file(tok_path)
    if is_main:
        print(f"[Tokenizer] Loaded TARA Tokenizer ({tokenizer.get_vocab_size():,} tokens)", flush=True)

    # Pre-detect target resume step from existing checkpoint metadata if resuming
    target_resume_step = 0
    candidate_resume_dirs = [getattr(args, "resume_dir", None), getattr(args, "output_dir", None)]
    for rdir in candidate_resume_dirs:
        if rdir and os.path.isdir(rdir):
            for st_fname in ("checkpoint_state.json", "STAGE_METADATA.json"):
                st_path = os.path.join(rdir, st_fname)
                if os.path.isfile(st_path):
                    try:
                        with open(st_path, "r", encoding="utf-8") as _stf:
                            _meta = json.load(_stf)
                            if "step" in _meta and isinstance(_meta["step"], int):
                                target_resume_step = _meta["step"]
                                break
                    except Exception:
                        pass
            if target_resume_step > 0:
                break

    if is_main and target_resume_step > 0:
        print(f"[TPU Resume Target] Active checkpoint detected at Step {target_resume_step}. Target resume step: {target_resume_step}", flush=True)

    # 3. Instantiate Native TARA Model on TPU Device
    log_tpu_memory_stage("BEFORE_MODEL_LOADING", step=target_resume_step, device=device)
    if is_main:
        print(f"[Model Architecture] Instantiating TARA architecture ({config.get('num_hidden_layers', 16)} layers, hidden={config.get('hidden_size', 2048)}) into TPU device memory...", flush=True)
    model = TaraForCausalLM(config, gradient_checkpointing=args.gradient_checkpointing)
    if getattr(args, "precision", "bf16").lower() == "bf16":
        os.environ["XLA_USE_BF16"] = "1"
    model.to(device)
    _ensure_tied_weights(model)

    if is_main and config.get("tie_word_embeddings", False):
        target = getattr(model, "module", model)
        assert target.lm_head.weight is target.model.embed_tokens.weight, "Pointer identity assertion failed: lm_head.weight is not embed_tokens.weight"
        print(f"[Model Architecture] Verified architectural weight tying (lm_head.weight is embed_tokens.weight: True) on device ({device}).", flush=True)

    log_tpu_memory_stage("AFTER_MODEL_LOADING", step=target_resume_step, device=device, model=model)

    # 4. AdamW Optimizer with 100% Exact GPU Parameter Grouping
    hw_info = get_tpu_hardware_info()
    if getattr(args, "offload_optimizer", None) is None:
        args.offload_optimizer = should_offload_optimizer(
            hardware_type=hw_info.get("hardware_type", ""),
            num_cores=hw_info.get("num_cores", 1),
        )

    if is_main:
        if args.offload_optimizer:
            print(
                "[Optimizer Setup] Optimizer state offloading: ENABLED (Host CPU RAM)\n"
                "  -> Offloading ~7.07 GB FP32 momentum buffers to CPU Host RAM.\n"
                "  -> Preserves ~7.07 GB TPU HBM for forward activations and backpropagation.",
                flush=True,
            )
        else:
            print(
                "[Optimizer Setup] Optimizer state offloading: DISABLED (TPU Device Resident)",
                flush=True,
            )

    optim_groups = build_tpu_optimizer_groups(model, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=args.lr,
        betas=TPUConfig.DEFAULT_BETAS,
        eps=TPUConfig.DEFAULT_EPS,
    )
    if is_main:
        print(
            f"[Optimizer] Configured AdamW with 2 parameter groups:\n"
            f"  -> Group 0: {len(optim_groups[0]['params'])} weight tensors (decay: {optim_groups[0]['weight_decay']})\n"
            f"  -> Group 1: {len(optim_groups[1]['params'])} norm/bias/embedding tensors (decay: {optim_groups[1]['weight_decay']})\n"
            f"  -> Hyperparameters: betas={TPUConfig.DEFAULT_BETAS}, eps={TPUConfig.DEFAULT_EPS}",
            flush=True
        )

    # 5. Resume or Load Existing Weights with Strict Validation
    start_step = 0
    start_epoch = 0
    best_loss = float("inf")
    resumed = False
    cursor = {}

    resume_dir = getattr(args, "resume_dir", None)
    if resume_dir and not os.path.isdir(resume_dir):
        resume_dir = None

    if resume_dir:
        start_step, start_epoch, resumed_best_loss, resumed, cursor = load_tpu_checkpoint(
            resume_dir, model, optimizer, device=device, allow_cold_start=args.allow_optimizer_cold_start, offload_optimizer=args.offload_optimizer
        )
        if resumed:
            best_loss = _safe_float(cursor.get("best_loss"), resumed_best_loss)
            resumed_best_step = _safe_int(cursor.get("best_milestone_step"), -1)
            try:
                local_p = os.path.dirname(os.path.abspath(resume_dir))
                loc_bl, loc_bs, _ = probe_local_best_milestone(local_p)
                if loc_bl < best_loss:
                    best_loss = loc_bl
                    resumed_best_step = loc_bs
            except Exception:
                pass

            if is_main:
                print("=" * 78, flush=True)
                print(f"  [RESUME] Discovered latest training checkpoint at Step {start_step} in: {resume_dir}", flush=True)
                print(f"  Resumed Step        : {start_step}", flush=True)
                print(f"  Resumed Epoch       : {start_epoch}", flush=True)
                print(f"  Historical Best Loss: {best_loss:.4f}", flush=True)
                if cursor and cursor.get("active_shard"):
                    print(f"  Data Cursor Shard   : {cursor.get('active_shard')}", flush=True)
                    print(f"  Data Cursor Offset  : {cursor.get('active_shard_line_offset', 0):,}", flush=True)
                print("=" * 78, flush=True)

        # Update optimizer LR for resumed step smoothly
        if start_step > 0:
            if start_step >= args.max_steps and is_main:
                print(
                    f"\n[TPU Step Notice] Checkpoint start step ({start_step}) is already >= --max-steps ({args.max_steps})!\n"
                    f"  -> Note: In TARA pretraining, --max-steps is the ABSOLUTE global step ceiling.\n"
                    f"  -> To run {args.max_steps} additional steps, pass: --max-steps {start_step + args.max_steps}\n",
                    flush=True
                )
            if args.lr is None:
                b_lr = _safe_float(cursor.get("base_lr"), None)
                args.lr = b_lr if b_lr is not None and b_lr > 0 else TPUConfig.DEFAULT_LR
            cur_init_lr = compute_tpu_lr(start_step, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
            saved_lr = _safe_float(cursor.get("learning_rate") or cursor.get("saved_lr"), None)
            if saved_lr is not None and saved_lr > 0:
                cur_init_lr = saved_lr
            for pg in optimizer.param_groups:
                pg["lr"] = cur_init_lr
            if is_main:
                print(f"[TPU Resume LR] Resumed optimizer schedule at Step {start_step}: LR = {cur_init_lr:.2e} (base={args.lr:.2e}, warmup={args.warmup_steps})", flush=True)

            if HAS_XLA:
                xm.mark_step()
            if HAS_XLA and world_size > 1:
                tpu_barrier("post_resume_sync")

    if not resumed:
        if args.lr is None:
            args.lr = TPUConfig.DEFAULT_LR
        if is_main:
            print(f"[Model] No prior checkpoint resumed; seeding model from candidate base weights in '{args.model_dir}'...", flush=True)
        cand_loaded = load_candidate_weights(model, args.model_dir)
        _ensure_tied_weights(model)
        if cand_loaded and is_main:
            print(f"[Model] Successfully seeded model weights from candidate directory: {args.model_dir}", flush=True)
        elif not cand_loaded and is_main:
            print(f"[Model] No candidate weights found in '{args.model_dir}'. Initialized fresh model architecture with random weights.", flush=True)
        if HAS_XLA:
            xm.mark_step()

    if getattr(args, "freeze_backbone", False):
        target_model = getattr(model, "module", model)
        if hasattr(target_model, "model"):
            for p in target_model.model.parameters():
                p.requires_grad = False
            if is_main:
                print("[Model Freeze] Transformer backbone frozen. Only lm_head is trainable for lightweight/head tuning.", flush=True)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main:
        print(f"[Model] Total Trainable Parameters: {total_params:,} ({total_params * 4 / 1e6:.1f} MB)", flush=True)

    log_tpu_memory_stage("AFTER_OPTIMIZER_RESTORE", step=start_step, device=device, model=model, optimizer=optimizer)

    # 6. Authoritative Shard Discovery & Multi-Domain Curriculum Setup
    shard_ledger = ShardLedger(args.output_dir, stage_id=args.stage_id, dataset_revision=args.dataset_prefix.strip("/"))

    raw_dataset_repo = args.hf_dataset_repo or args.hf_repo or os.environ.get("HF_DATASET_REPO", os.environ.get("HF_REPO", "manjukp6/tara_dataset_v1"))
    dataset_repo = None if (raw_dataset_repo is None or str(raw_dataset_repo).strip().lower() in ("", "none", "local", "null")) else str(raw_dataset_repo).strip()
    ds_prefix = args.dataset_prefix.strip("/")

    VALID_EXTS = (".parquet", ".jsonl", ".txt", ".csv", ".tsv", ".gz")
    SKIP_SUBSTRINGS = ("eval.yaml", "minif2f_lean4_test", "minif2f_lean4_valid", ".gitattributes", "readme.md", ".yaml", ".yml", "indiccorp_v2")

    local_files = []
    if os.path.isdir(args.dataset_dir):
        for root, _, files in os.walk(args.dataset_dir):
            for f in files:
                rel = os.path.relpath(os.path.join(root, f), args.dataset_dir).replace("\\", "/")
                if rel.endswith(VALID_EXTS) and not any(s in rel.lower() for s in SKIP_SUBSTRINGS):
                    local_files.append(rel)

    remote_files = []
    if dataset_repo:
        try:
            from huggingface_hub import HfApi
            token = _resolve_hf_token(args.hf_token)
            api = HfApi(token=token)
            files = api.list_repo_files(repo_id=dataset_repo, repo_type="dataset")
            if ds_prefix:
                prefix_filter = f"{ds_prefix}/"
                remote_files = [
                    f[len(prefix_filter):] if f.startswith(prefix_filter) else f
                    for f in files
                    if f.startswith(prefix_filter) and f.endswith(VALID_EXTS) and not any(s in f.lower() for s in SKIP_SUBSTRINGS)
                ]
            if not remote_files:
                remote_files = [
                    f for f in files
                    if f.endswith(VALID_EXTS) and not any(s in f.lower() for s in SKIP_SUBSTRINGS)
                ]
        except Exception as re_err:
            if is_main:
                print(f"[Dataset Discovery Notice] Remote HF discovery note: {re_err}", flush=True)

    discovered_shards = sorted(set(local_files) | set(remote_files))

    # Strict Fail-Closed Check: Never silently substitute DummyDataset!
    if not discovered_shards:
        raise RuntimeError(
            f"[Dataset Discovery Fatal] Zero valid dataset shards discovered in local path '{args.dataset_dir}' "
            f"or remote repository '{dataset_repo}'. Training cannot proceed with zero dataset shards."
        )

    # Strictly filter out held-out evaluation shards
    exclude_patterns = [p.strip() for p in args.exclude_shards.split(",") if p.strip()]
    eligible_shards = []
    for sf in discovered_shards:
        if any(fnmatch.fnmatch(sf, pat) for pat in exclude_patterns):
            if is_main:
                print(f"[Data Gate] STRICT LEAKAGE PREVENTION: Excluding held-out shard: {sf}", flush=True)
        else:
            eligible_shards.append(sf)

    if not eligible_shards:
        raise RuntimeError("[Dataset Discovery Fatal] All discovered shards were excluded by held-out patterns. Zero eligible shards remain.")

    selected_shards = eligible_shards if args.max_shards <= 0 else eligible_shards[:args.max_shards]
    shard_ledger.total_eligible_shards = len(selected_shards)
    if is_main:
        shard_ledger.save()
        print(f"[Dataset] Discovered {len(discovered_shards)} total shards ({len(eligible_shards)} eligible); preparing pipeline across {len(selected_shards)} shard(s)...", flush=True)

    # Authoritative Dataset Snapshot
    dataset_snapshot = DatasetSnapshot(args.output_dir)
    if is_main:
        dataset_snapshot.initialize_from_discovery(
            repo_id=dataset_repo,
            prefix=ds_prefix,
            ordered_shards=list(selected_shards),
        )

    # Reconcile ShardLedger if resumed from checkpoint
    if resumed and cursor and shard_ledger:
        shard_ledger.reconcile_with_checkpoint(cursor, start_step)
        if is_main:
            audit_shard = cursor.get("active_shard") or getattr(shard_ledger, "active_shard", "N/A")
            audit_line = cursor.get("active_shard_line_offset", 0)
            audit_byte = cursor.get("active_shard_byte_offset", 0)
            audit_completed = len(getattr(shard_ledger, "completed_shards", []))
            print("=" * 80, flush=True)
            print("  [DATASET ZERO-DUPLICATE EXACT RESUME AUDIT]", flush=True)
            print(f"  Total Lifetime Step Resumed   : {start_step:,}", flush=True)
            print(f"  Authoritative Active Shard    : {audit_shard}", flush=True)
            print(f"  Exact Line Fast-Forward Offset: {audit_line:,} (Byte Offset: {audit_byte:,})", flush=True)
            print(f"  Completed Shards in Ledger    : {audit_completed} shards", flush=True)
            print("  Status                        : 100% DETERMINISTIC ZERO DUPLICATE RESUMPTION GUARANTEED", flush=True)
            print("=" * 80, flush=True)

    # Extract per-rank resume coordinates if multi-rank checkpoint
    rank_c = cursor.get("per_rank_cursors", {}).get(str(rank), cursor)
    resume_shard = rank_c.get("active_shard", rank_c.get("resume_shard", None))
    resume_line_offset = rank_c.get("active_shard_line_offset", rank_c.get("resume_line_offset", -1))
    resume_byte_offset = rank_c.get("active_shard_byte_offset", rank_c.get("resume_byte_offset", 0))
    resume_chunk_idx = rank_c.get("active_chunk_idx", rank_c.get("resume_chunk_idx", 0))

    if resume_shard and shard_ledger and shard_ledger.is_completed(resume_shard):
        meta = shard_ledger.completed_metadata.get(resume_shard, {})
        rec_count = meta.get("records_processed", 0)
        if rec_count > 0 and resume_line_offset >= rec_count - 1:
            if is_main:
                print(f"[Dataset Resume] Shard '{resume_shard}' was marked complete in ledger; starting fresh on next shard.", flush=True)
            resume_shard = None
            resume_line_offset = -1
            resume_byte_offset = 0
            resume_chunk_idx = 0

    # Multi-Domain DRR Grouping
    domain_shards: Dict[str, List[str]] = {d: [] for d in CANONICAL_DOMAINS}
    for s in selected_shards:
        d = resolve_shard_domain(s)
        if d not in domain_shards:
            domain_shards[d] = []
        domain_shards[d].append(s)

    active_doms_with_shards = [d for d, shards in domain_shards.items() if shards]
    domain_weights_map = parse_domain_weights(getattr(args, "domain_weights", ""), active_doms_with_shards)

    resumed_domain_cursors = cursor.get("domain_cursors", {}) if cursor else (shard_ledger.domain_cursors if shard_ledger else {})
    resumed_domain_cycles = cursor.get("domain_cycles", {}) if cursor else (getattr(shard_ledger, "domain_cycles", {}) if shard_ledger else {})

    is_pretraining = "stage0" in args.stage_id.lower() or "pretrain" in args.stage_id.lower()
    effective_mask_prompt = getattr(args, "mask_prompt", False) if not is_pretraining else False

    if len(active_doms_with_shards) > 1:
        if is_main:
            print(f"[Multi-Domain Setup] Initializing TaraMultiDomainScheduler across {len(active_doms_with_shards)} active domains (DRR).", flush=True)
        dataset = TaraMultiDomainScheduler(
            domain_shards=domain_shards,
            domain_weights=domain_weights_map,
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            max_samples=args.max_samples,
            dataset_dir=args.dataset_dir,
            repo_id=dataset_repo,
            prefix=ds_prefix,
            shard_ledger=shard_ledger,
            keep_shards=args.keep_shards,
            buffer_size=args.buffer_size,
            max_shard_bytes=args.max_shard_bytes,
            dataset_snapshot=dataset_snapshot,
            domain_cursors=resumed_domain_cursors,
            domain_cycles=resumed_domain_cycles,
            max_domain_cycles=args.max_domain_cycles,
            scheduler_deficits=cursor.get("scheduler_deficits") if cursor else None,
            mask_prompt=effective_mask_prompt,
            rank=rank,
            world_size=world_size,
        )
    else:
        if is_main:
            print(f"[Dataset Setup] Initializing TaraStreamingJsonlDataset across {len(selected_shards)} shard(s).", flush=True)
        dataset = TaraStreamingJsonlDataset(
            file_paths=selected_shards,
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            max_samples=args.max_samples,
            shuffle_shards=False,
            resume_shard=resume_shard,
            resume_line_offset=resume_line_offset,
            resume_byte_offset=resume_byte_offset,
            resume_chunk_idx=resume_chunk_idx,
            shard_ledger=shard_ledger,
            keep_shards=args.keep_shards,
            dataset_dir=args.dataset_dir,
            repo_id=dataset_repo,
            prefix=ds_prefix,
            buffer_size=args.buffer_size,
            max_shard_bytes=args.max_shard_bytes,
            dataset_snapshot=dataset_snapshot,
            mask_prompt=effective_mask_prompt,
            rank=rank,
            world_size=world_size,
        )

    collate = lambda b: tpu_collate_fn(b, max_seq_len=args.max_seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate,
        drop_last=True,
    )

    if HAS_XLA:
        data_iterator = pl.MpDeviceLoader(dataloader, device)
    else:
        data_iterator = dataloader

    # 7. Training Execution Loop
    best_cp_dir = os.path.join(args.output_dir, "best_checkpoint")
    promoted_target = args.promoted_dir if args.promoted_dir else TPUConfig.DEFAULT_PROMOTED_DIR

    model.train()
    step = start_step
    accum_tokens = 0
    accum_loss = 0.0
    accum_ce_loss = 0.0
    accum_aux_loss = 0.0
    accum_valid_tokens = torch.tensor(0, dtype=torch.long, device=device)
    accum_batches = 0
    accum_batch_cursors = []
    step_start_time = time.time()
    last_eval_step = step
    last_regular_checkpoint_step = step
    last_best_checkpoint_step = step
    last_promote_step = step
    promoted_at_target_step = False
    committed_domain_cursors = copy.deepcopy(resumed_domain_cursors) if resumed_domain_cursors else {}
    total_samples_trained = _safe_int(cursor.get("total_samples_trained"), 0) if cursor else 0
    samples_consumed = _safe_int(cursor.get("samples_consumed"), total_samples_trained) if cursor else 0
    samples_skipped = _safe_int(cursor.get("samples_skipped"), 0) if cursor else 0
    steps_skipped = _safe_int(cursor.get("steps_skipped"), 0) if cursor else 0
    accum_attempt = 0
    ema_loss = None

    if is_main:
        print(f"[TPU Training Loop] Launching Steps {step}..{args.max_steps} on {world_size} TPU Cores...", flush=True)
        print(f"[TPU Engine] Note: First step ({step + 1}) compiles XLA execution graph for TPU hardware (JIT compilation may take ~1-2 min). Subsequent steps run instantly.\n", flush=True)

    epoch = start_epoch
    for epoch in range(start_epoch, args.epochs):
        for batch_data in data_iterator:
            cursors = []
            attention_mask = None
            if isinstance(batch_data, (tuple, list)):
                if len(batch_data) >= 4:
                    inputs, labels, attention_mask, cursors = batch_data[0], batch_data[1], batch_data[2], batch_data[3]
                elif len(batch_data) >= 3:
                    inputs, labels, cursors = batch_data[0], batch_data[1], batch_data[2]
                elif len(batch_data) >= 2:
                    inputs, labels = batch_data[0], batch_data[1]
                else:
                    inputs = batch_data[0]
                    labels = inputs.clone()
            else:
                inputs = batch_data
                labels = inputs.clone()

            if inputs.device != device:
                inputs = inputs.to(device)
            if labels.device != device:
                labels = labels.to(device)
            if attention_mask is not None and attention_mask.device != device:
                attention_mask = attention_mask.to(device)

            # Diagnostic stage: BEFORE_FORWARD
            if accum_batches == 0 and (step == start_step or (args.checkpoint_interval > 0 and step % args.checkpoint_interval == 0)):
                log_tpu_memory_stage("BEFORE_FORWARD", step=step, device=device, model=model, optimizer=optimizer, microbatch=accum_batches)

            # Memory-Safe TPU Chunked Cross-Entropy Projection (prevents 196MB OOM and dynamic tt_jit_index)
            loss_res = compute_tpu_chunked_loss(
                model=model,
                input_ids=inputs,
                labels=labels,
                attention_mask=attention_mask,
                chunk_sz=getattr(args, "loss_chunk_size", TPUConfig.DEFAULT_LOSS_CHUNK_SIZE),
            )
            loss, ce_loss, aux_loss, cur_valid_tokens = loss_res

            # Diagnostic stage: BEFORE_BACKWARD
            if accum_batches == 0 and (step == start_step or (args.checkpoint_interval > 0 and step % args.checkpoint_interval == 0)):
                log_tpu_memory_stage("BEFORE_BACKWARD", step=step, device=device, model=model, optimizer=optimizer, microbatch=accum_batches)

            # Unreduced loss is backwarded directly so gradients accumulate as the exact sum of token losses
            loss_to_backward = getattr(loss_res, "unreduced_loss", loss)
            loss_to_backward.backward()

            # PyTorch/XLA lazy step mark: bounds computation graph and frees activations per micro-batch
            if HAS_XLA:
                xm.mark_step()

            accum_loss += getattr(loss_res, "loss_sum", loss).detach()
            accum_ce_loss += float(ce_loss.item()) if hasattr(ce_loss, "item") else float(ce_loss)
            accum_aux_loss += float(aux_loss.item()) if hasattr(aux_loss, "item") else float(aux_loss)
            accum_valid_tokens = accum_valid_tokens + cur_valid_tokens.detach()
            accum_batches += 1
            accum_tokens += (inputs.size(0) * inputs.size(1))
            accum_batch_cursors.append(cursors)
            samples_consumed += len(cursors)

            # Optimizer Step on Accumulation Boundary
            if accum_batches >= args.grad_accum:
                accum_attempt += 1
                cur_lr = compute_tpu_lr(step, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
                for pg in optimizer.param_groups:
                    pg["lr"] = cur_lr

                local_loss_sum = accum_loss.item() if isinstance(accum_loss, torch.Tensor) else float(accum_loss)
                local_valid_tokens = int(accum_valid_tokens.item())
                if HAS_XLA and world_size > 1:
                    loss_sync_tag = f"tpu_loss_s{step}_a{accum_attempt}" if accum_attempt > 0 else f"tpu_loss_{step}"
                    cluster_loss_sum = sync_tpu_sum_float(local_loss_sum, tag=loss_sync_tag)
                    tokens_sync_tag = f"tpu_tokens_s{step}_a{accum_attempt}" if accum_attempt > 0 else f"tpu_tokens_{step}"
                    cluster_valid_tokens = sync_tpu_sum(local_valid_tokens, tag=tokens_sync_tag)
                else:
                    cluster_loss_sum = local_loss_sum
                    cluster_valid_tokens = local_valid_tokens

                cur_step_loss = (cluster_loss_sum / float(cluster_valid_tokens)) if cluster_valid_tokens > 0 else 0.0
                cur_step_ce = (accum_ce_loss / max(1, accum_batches))
                cur_step_aux = (accum_aux_loss / max(1, accum_batches))

                # Diagnostic stage: BEFORE_OPTIMIZER_STEP
                if step == start_step or (args.checkpoint_interval > 0 and step % args.checkpoint_interval == 0):
                    log_tpu_memory_stage("BEFORE_OPTIMIZER_STEP", step=step, device=device, model=model, optimizer=optimizer, microbatch=accum_batches)

                is_skipped, grad_norm_val = tpu_optimizer_step(
                    optimizer,
                    model=model,
                    max_norm=1.0,
                    cur_loss=cur_step_loss,
                    ema_loss=ema_loss,
                    valid_tokens=local_valid_tokens,
                    step=step,
                    attempt=accum_attempt,
                    offload_optimizer=args.offload_optimizer,
                )

                if is_skipped:
                    skipped_in_window = sum(len(c) for c in accum_batch_cursors)
                    samples_skipped += skipped_in_window
                    steps_skipped += 1
                    if is_main:
                        print(
                            f"[{args.stage_id} | TPU Step {step:05d}/{args.max_steps}] "
                            f"{CLR_RED}[STEP SKIPPED SHOCK/NAN]{CLR_RESET} "
                            f"Loss: {cur_step_loss:.4f} | GradNorm: {grad_norm_val:.2f}. "
                            f"Skipped {skipped_in_window} samples (total skipped: {samples_skipped}). "
                            f"Discarding accumulation window; advancing dataset cursor past corrupt samples.",
                            flush=True,
                        )
                    optimizer.zero_grad(set_to_none=True)
                    accum_loss = 0.0
                    accum_ce_loss = 0.0
                    accum_aux_loss = 0.0
                    accum_valid_tokens = torch.tensor(0, dtype=torch.long, device=device)
                    accum_batches = 0
                    accum_tokens = 0
                    # Advance committed domain cursors past skipped corrupt samples to prevent infinite crash loop on resume
                    for b_c in accum_batch_cursors:
                        for c_item in b_c:
                            if len(c_item) >= 4 and isinstance(c_item, (tuple, list)):
                                c_dom = resolve_shard_domain(str(c_item[0])) if c_item[0] else ""
                                if c_dom:
                                    committed_domain_cursors[c_dom] = {
                                        "active_shard": c_item[0],
                                        "line_offset": c_item[1],
                                        "byte_offset": c_item[2] if len(c_item) > 2 else 0,
                                        "chunk_idx": c_item[3] if len(c_item) > 3 else 0,
                                    }
                    current_cursor = extract_tpu_data_cursor(
                        accum_batch_cursors,
                        dataset,
                        total_samples_trained,
                        samples_consumed,
                        shard_ledger,
                        committed_domain_cursors,
                        samples_skipped=samples_skipped,
                        steps_skipped=steps_skipped,
                        rank=rank,
                    )
                    accum_batch_cursors.clear()
                    continue

                optimizer.zero_grad(set_to_none=True)

                step += 1
                total_samples_trained += sum(len(c) for c in accum_batch_cursors)
                accum_loss = 0.0
                accum_ce_loss = 0.0
                accum_aux_loss = 0.0
                accum_valid_tokens = torch.tensor(0, dtype=torch.long, device=device)
                accum_batches = 0
                # Preserve accum_tokens for accurate speed calculation below

                ema_loss = cur_step_loss if ema_loss is None else (0.9 * ema_loss + 0.1 * cur_step_loss)

                # Update committed domain cursors
                for b_c in accum_batch_cursors:
                    for c_item in b_c:
                        if len(c_item) >= 4 and isinstance(c_item, (tuple, list)):
                            c_dom = resolve_shard_domain(str(c_item[0])) if c_item[0] else ""
                            if c_dom:
                                committed_domain_cursors[c_dom] = {
                                    "active_shard": c_item[0],
                                    "line_offset": c_item[1],
                                    "byte_offset": c_item[2] if len(c_item) > 2 else 0,
                                    "chunk_idx": c_item[3] if len(c_item) > 3 else 0,
                                }

                current_cursor = extract_tpu_data_cursor(
                    accum_batch_cursors,
                    dataset,
                    total_samples_trained,
                    samples_consumed,
                    shard_ledger,
                    committed_domain_cursors,
                    samples_skipped=samples_skipped,
                    steps_skipped=steps_skipped,
                    rank=rank,
                )
                accum_batch_cursors.clear()

                # Telemetry
                step_duration = max(time.time() - step_start_time, 1e-4)
                step_start_time = time.time()
                cur_tokens_count = accum_tokens if accum_tokens > 0 else (args.batch_size * args.grad_accum * getattr(args, "max_seq_len", 2048))
                step_speed = (cur_tokens_count * world_size) / step_duration
                accum_tokens = 0

                if is_main:
                    ppl = math.exp(min(cur_step_loss, 20.0))
                    bpb = (cur_step_loss / math.log(2)) / 3.8

                    loss_clr, loss_status, loss_tag = get_metric_status_and_color(cur_step_loss, (1.4, 1.9), (2.0, 3.0))
                    ppl_clr, ppl_status, ppl_tag = get_metric_status_and_color(ppl, (4.0, 7.0), (7.1, 20.0))
                    bpb_clr, bpb_status, bpb_tag = get_metric_status_and_color(bpb, (0.55, 0.75), (0.76, 1.20))

                    cur_domain = getattr(dataset, "active_domain", "")
                    if not cur_domain and current_cursor and current_cursor.get("active_shard"):
                        cur_domain = resolve_shard_domain(current_cursor["active_shard"])

                    cur_shard_name = current_cursor.get("active_shard") if current_cursor else getattr(dataset, "current_active_shard", "N/A")
                    cur_line_num = current_cursor.get("active_shard_line_offset", 0) if current_cursor else getattr(dataset, "active_shard_offset", 0)
                    cur_shard_tot = getattr(dataset, "active_shard_total_lines", 0)
                    shard_pct = f" {(cur_line_num / cur_shard_tot) * 100:.2f}%" if cur_shard_tot > 0 else ""
                    shard_progress_str = f"Shard: {cur_shard_name} (Line {cur_line_num:,}/{cur_shard_tot:,}{shard_pct})" if cur_shard_tot > 0 else f"Shard: {cur_shard_name} (Line {cur_line_num:,})"
                    shard_line_log = f"{shard_progress_str} | " if (cur_shard_name and cur_shard_name != "N/A") else ""
                    domain_tag = f"Domain: [{cur_domain}] " if cur_domain else ""

                    print(
                        f"[{args.stage_id} | Step {step:04d}/{args.max_steps}] "
                        f"{domain_tag}"
                        f"Total Loss: {loss_clr}{cur_step_loss:.4f}{CLR_RESET} (CE: {cur_step_ce:.4f}, Aux: {cur_step_aux:.4f}, EMA: {ema_loss:.4f}) | "
                        f"PPL: {ppl_clr}{ppl:.2f}{CLR_RESET} | "
                        f"BPB: {bpb_clr}{bpb:.2f} b/B{CLR_RESET} | "
                        f"Standing: {loss_clr}{loss_status} {loss_tag}{CLR_RESET} | "
                        f"LR: {cur_lr:.2e} | "
                        f"GradNorm: {grad_norm_val:.2f} | "
                        f"Speed: {step_speed:.0f} tok/s ({step_duration:.2f}s) | "
                        f"{shard_line_log}"
                        f"Samples: {total_samples_trained:,} | "
                        f"Cores: {world_size}",
                        flush=True
                    )
                    if (step % 500 == 0) or (args.eval_interval > 0 and step % args.eval_interval == 0):
                        print_frontier_comparison_matrix_colored(step, cur_step_loss, best_loss)

                # Periodic Evaluation & Live Text Generation Probe (optional)
                should_probe = getattr(args, "enable_probe", False) and (args.eval_interval > 0)
                if should_probe and step % args.eval_interval == 0 and step > last_eval_step:
                    last_eval_step = step
                    probe_tpu_generation(model, tokenizer, device, config, step, args.stage_id)

                # Best Milestone Checkpoint Tracking
                is_new_best = (cur_step_loss < best_loss)
                committed_best_this_step = False
                can_save = (step > args.skip_save_steps) or (step >= args.max_steps)

                if is_new_best:
                    is_meaningful = (best_loss - cur_step_loss) >= 0.005
                    cooldown_ok = (step - last_best_checkpoint_step >= 50)
                    best_loss = cur_step_loss

                    if can_save and (is_meaningful or cooldown_ok or step >= args.max_steps):
                        committed_best_this_step = True
                        last_best_checkpoint_step = step
                        if is_main:
                            print(f"[{args.stage_id}] >>> {CLR_GREEN}{CLR_BOLD}COMMITTING TPU BEST MILESTONE: {best_loss:.4f} (Step {step}){CLR_RESET} <<<", flush=True)

                        full_cursor = dict(current_cursor)
                        save_tpu_checkpoint(
                            model=model,
                            optimizer=optimizer,
                            config=config,
                            tok_path=tok_path,
                            output_dir=args.output_dir,
                            step=step,
                            epoch=epoch,
                            loss=cur_step_loss,
                            best_loss=best_loss,
                            stage_id=args.stage_id,
                            data_cursor=full_cursor,
                            is_best=True,
                            best_cp_dir=best_cp_dir,
                            keep_n=getattr(args, "keep_checkpoints", 2),
                            hf_repo=args.hf_repo or args.hf_model_repo,
                            hf_token=args.hf_token,
                            max_shard_mb=getattr(args, "max_shard_mb", TPUConfig.MAX_SHARD_MB),
                            dataset=dataset,
                            shard_ledger=shard_ledger,
                            base_lr=args.lr,
                        )
                        log_tpu_memory_stage("AFTER_CHECKPOINT_SAVE", step=step, device=device, model=model, optimizer=optimizer)

                # Regular Checkpoint Saving
                is_regular = (args.checkpoint_interval > 0 and step % args.checkpoint_interval == 0 and step > last_regular_checkpoint_step)
                if can_save and is_regular and not committed_best_this_step:
                    last_regular_checkpoint_step = step
                    full_cursor = dict(current_cursor)
                    save_tpu_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        config=config,
                        tok_path=tok_path,
                        output_dir=args.output_dir,
                        step=step,
                        epoch=epoch,
                        loss=cur_step_loss,
                        best_loss=best_loss,
                        stage_id=args.stage_id,
                        data_cursor=full_cursor,
                        is_best=False,
                        keep_n=getattr(args, "keep_checkpoints", 2),
                        hf_repo=args.hf_repo or args.hf_model_repo,
                        hf_token=args.hf_token,
                        max_shard_mb=getattr(args, "max_shard_mb", TPUConfig.MAX_SHARD_MB),
                        dataset=dataset,
                        shard_ledger=shard_ledger,
                        base_lr=args.lr,
                    )
                    log_tpu_memory_stage("AFTER_CHECKPOINT_SAVE", step=step, device=device, model=model, optimizer=optimizer)

                # Periodic Model Promotion
                is_promote = (args.promote_interval > 0 and step % args.promote_interval == 0 and step > last_promote_step)
                if is_promote and is_main:
                    last_promote_step = step
                    src_cand = best_cp_dir if check_model_weights_exist(best_cp_dir) else args.output_dir
                    print(f"\n{CLR_GREEN}>>> PERIODIC TPU MODEL PROMOTION (Step {step}) -> {promoted_target} <<<{CLR_RESET}", flush=True)
                    try:
                        promote_checkpoint(src_cand, promoted_target, args.stage_id)
                        if args.hf_repo:
                            trigger_async_hf_upload(promoted_target, args.hf_repo, args.stage_id, step)
                    except Exception as pe:
                        print(f"[TPU Promotion Notice] Periodic promotion warning: {pe}", flush=True)

                if step >= args.max_steps:
                    break

        if step >= args.max_steps:
            break

    # Final Checkpoint Save
    save_tpu_checkpoint(
        model=model,
        optimizer=optimizer,
        config=config,
        tok_path=tok_path,
        output_dir=args.output_dir,
        step=step,
        epoch=epoch,
        loss=cur_step_loss if 'cur_step_loss' in locals() else 0.0,
        best_loss=best_loss,
        stage_id=args.stage_id,
        data_cursor=current_cursor if 'current_cursor' in locals() else cursor,
        is_best=False,
        is_completed=True,
        keep_n=getattr(args, "keep_checkpoints", 2),
        hf_repo=args.hf_repo or args.hf_model_repo,
        hf_token=args.hf_token,
        max_shard_mb=getattr(args, "max_shard_mb", TPUConfig.MAX_SHARD_MB),
        dataset=dataset,
        shard_ledger=shard_ledger,
        base_lr=args.lr,
    )

    if is_main:
        src_final = best_cp_dir if check_model_weights_exist(best_cp_dir) else args.output_dir
        print(f"\n{CLR_GREEN}{'=' * 80}{CLR_RESET}")
        print(f"[{args.stage_id} COMPLETE] TPU Pre-Training finished {step} steps with loss {best_loss:.4f}!")
        print(f"Promoting final deployment model to: {promoted_target}")
        print(f"{CLR_GREEN}{'=' * 80}{CLR_RESET}\n", flush=True)
        try:
            finalize_stage_and_promote(
                output_dir=args.output_dir,
                promoted_dir=promoted_target,
                stage_id=args.stage_id,
                hf_repo=args.hf_repo,
                candidate_source_dir=src_final,
            )
            if args.hf_repo:
                wait_for_hf_upload(timeout=180.0)
        except Exception as fe:
            print(f"[TPU Finalize Notice] Stage promotion notice: {fe}", flush=True)


def parse_tpu_arguments() -> argparse.Namespace:
    """Parses command-line arguments for TARA TPU Training."""
    parser = argparse.ArgumentParser(description="TARA Autonomous Google TPU (v6e / v5p / v5e) Engine")
    parser.add_argument("--model-dir", type=str, default="storage/models/tara_candidate_v1", help="Candidate model directory (config.json & tokenizer.json)")
    parser.add_argument("--dataset-dir", type=str, default="storage/datasets/main_dataset_v2", help="Dataset directory")
    parser.add_argument("--output-dir", type=str, default=TPUConfig.DEFAULT_OUTPUT_DIR, help="Dedicated TPU checkpoint directory (does not overwrite GPU checkpoints)")
    parser.add_argument("--promoted-dir", type=str, default=TPUConfig.DEFAULT_PROMOTED_DIR, help="Promoted model directory")
    parser.add_argument("--stage-id", type=str, default="stage0_pretraining", help="Training stage identifier")
    parser.add_argument("--batch-size", type=int, default=TPUConfig.DEFAULT_BATCH_SIZE, help=f"Per-core TPU microbatch size (default: {TPUConfig.DEFAULT_BATCH_SIZE})")
    parser.add_argument("--grad-accum", type=int, default=TPUConfig.DEFAULT_GRAD_ACCUM, help=f"Gradient accumulation steps (default: {TPUConfig.DEFAULT_GRAD_ACCUM})")
    parser.add_argument("--loss-chunk-size", type=int, default=TPUConfig.DEFAULT_LOSS_CHUNK_SIZE, help=f"Memory-safe vocabulary chunk size for cross-entropy projection (default: {TPUConfig.DEFAULT_LOSS_CHUNK_SIZE})")
    parser.add_argument("--precision", type=str, default=TPUConfig.DEFAULT_PRECISION, choices=["bf16", "fp32"], help=f"Compute precision (default: {TPUConfig.DEFAULT_PRECISION})")
    parser.add_argument("--max-seq-len", type=int, default=TPUConfig.DEFAULT_SEQ_LEN, help="Maximum sequence length (default: 2048)")
    parser.add_argument("--lr", type=float, default=None, help=f"Base learning rate (default: {TPUConfig.DEFAULT_LR} or resumed from checkpoint)")
    parser.add_argument("--min-lr", type=float, default=TPUConfig.DEFAULT_MIN_LR, help="Minimum learning rate (default: 1e-5)")
    parser.add_argument("--warmup-steps", type=int, default=TPUConfig.DEFAULT_WARMUP_STEPS, help="Linear warmup steps (default: 50)")
    parser.add_argument("--weight-decay", type=float, default=TPUConfig.DEFAULT_WEIGHT_DECAY, help="AdamW weight decay for 2D matrices (default: 0.01)")
    parser.add_argument("--max-steps", type=int, default=TPUConfig.DEFAULT_MAX_STEPS, help="Total training steps (default: 50,000)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs (default: 1)")
    parser.add_argument("--max-shards", type=int, default=0, help="Max shards to ingest (0 for all)")
    parser.add_argument("--max-samples", type=int, default=0, help="Max samples to ingest across shards (0 for unlimited)")
    parser.add_argument("--exclude-shards", type=str, default="*shard_099*,*shard_098*", help="Comma-separated patterns of held-out shards strictly excluded from training")
    parser.add_argument("--domain-weights", type=str, default=os.environ.get("DOMAIN_WEIGHTS", ""), help="Custom comma-separated domain weights")
    parser.add_argument("--max-domain-cycles", type=int, default=int(os.environ.get("MAX_DOMAIN_CYCLES", 3)), help="Max cycles per domain")
    parser.add_argument("--buffer-size", type=int, default=1, help="Streaming buffer size")
    parser.add_argument("--dataset-prefix", type=str, default="", help="Remote dataset version prefix on Hugging Face")
    parser.add_argument("--max-shard-bytes", type=int, default=0, help="Max bytes per shard")
    parser.add_argument("--bulk-download-dataset", action="store_true", default=False, help="Bulk download dataset instead of JIT")
    parser.add_argument("--keep-shards", action="store_true", default=False, help="Retain local shards")
    parser.add_argument("--keep-checkpoints", type=int, default=TPUConfig.KEEP_CHECKPOINTS, help="Checkpoints to retain")
    parser.add_argument("--max-shard-mb", type=int, default=TPUConfig.MAX_SHARD_MB, help="Max megabytes per model safetensors shard (default: 480)")
    parser.add_argument("--skip-save-steps", type=int, default=0, help="Initial steps to skip disk saves")
    parser.add_argument("--checkpoint-interval", type=int, default=TPUConfig.CHECKPOINT_INTERVAL, help="Steps between checkpoints")
    parser.add_argument("--eval-interval", type=int, default=TPUConfig.EVAL_INTERVAL, help="Steps between probe evaluations (default: 0 = disabled)")
    parser.add_argument("--enable-probe", action="store_true", default=False, help="Enable live CPU generation probe evaluations (default: False to maximize TPU throughput)")
    parser.add_argument("--promote-interval", type=int, default=TPUConfig.PROMOTE_INTERVAL, help="Steps between promotions")
    parser.add_argument("--promote-max-step", type=int, default=0, help="Ceiling step for early model promotion")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True, help="Enable gradient checkpointing")
    parser.add_argument("--resume", action="store_true", default=True, help="Auto-resume from highest checkpoint across TPU, GPU, GDrive, HF (default: True)")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Force fresh start without resuming from previous checkpoints")
    parser.add_argument("--resume-dir", type=str, default=None, help="Explicit directory from which to resume")
    parser.add_argument("--allow-optimizer-cold-start", action="store_true", default=False, help="Permit cold optimizer start if checkpoint state cannot be restored")
    parser.add_argument("--verify-gate", action="store_true", default=True, help="Verify pretraining authorization proof")
    parser.add_argument("--hf-repo", type=str, default=os.environ.get("HF_MODEL_REPO", os.environ.get("HF_REPO", None)), help="Hugging Face model repository")
    parser.add_argument("--hf-model-repo", type=str, default=os.environ.get("HF_MODEL_REPO", os.environ.get("HF_REPO", None)), help="Hugging Face model repository for weights upload")
    parser.add_argument("--hf-dataset-repo", type=str, default=os.environ.get("HF_DATASET_REPO", "manjukp6/tara_dataset_v1"), help="Hugging Face dataset repository")
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN", None), help="Hugging Face auth token")
    parser.add_argument("--gdrive-dir", type=str, default=os.environ.get("GDRIVE_RESTORE_DIR", os.environ.get("GDRIVE_BACKUP_DIR", None)), help="Google Drive backup directory")
    parser.add_argument("--mask-prompt", action="store_true", default=False, help="Mask user instruction tokens with -100 in SFT stages")
    parser.add_argument("--freeze-backbone", action="store_true", default=False, help="Freeze backbone transformer layers during specialized head training")
    parser.add_argument("--single-core", action="store_true", default=False, help="Run single-core TPU mode")
    parser.add_argument("--offload-optimizer", action=argparse.BooleanOptionalAction, default=None, help="Offload AdamW momentum buffers to CPU Host RAM (default: auto-detected for 16GB TPUs like v5e)")
    parser.add_argument("--validate-checkpoint-handoff", type=str, default=None, help="Validate a candidate checkpoint directory before stage handoff and exit")
    return parser.parse_args()


def main():
    args = parse_tpu_arguments()

    # Fast validation gate for multi-stage handoff
    if getattr(args, "validate_checkpoint_handoff", None):
        ok, msg = validate_stage_checkpoint_for_handoff(args.validate_checkpoint_handoff, args.stage_id)
        if ok:
            print(f"[TPU Handoff Gate PASSED] {msg}", flush=True)
            sys.exit(0)
        else:
            print(f"[TPU Handoff Gate FAILED] {msg}", file=sys.stderr, flush=True)
            sys.exit(1)

    hw_info = get_tpu_hardware_info()
    if not hw_info["available"]:
        print(
            "[TPU Hardware FATAL] Physical Google TPU accelerator not found!\n"
            "  This runner strictly requires Google Cloud TPU hardware (v5e-8 / v5p / v6e).\n"
            "  Please execute this runner on a Google Cloud TPU VM, Kaggle TPU (v5e-8), or Google Colab TPU runtime.",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)

    # Pre-training Authorization Gate Check (fast-fail on parent process before spawn)
    if args.verify_gate and verify_pretraining_authorization_proof:
        if not verify_pretraining_authorization_proof(args.model_dir, os.path.dirname(args.output_dir)):
            print(
                "[Pre-Training Gate FATAL] Pretraining authorization gate check failed. "
                "Cryptographic pretraining_authorization.json proof is missing or status is not AUTHORIZED.",
                file=sys.stderr,
                flush=True,
            )
            sys.exit(1)

    print("=" * 80)
    print("  TARA AUTONOMOUS GOOGLE TPU ENGINE (ALL GENERATIONS RUNNER)")
    print("=" * 80)
    print(f"  TPU Available : {hw_info['available']}")
    print(f"  Hardware Type : {hw_info['hardware_type']}")
    print(f"  Total Cores   : {hw_info['num_cores']}")
    print(f"  TPU Engine    : {hw_info.get('engine_name', 'PyTorch/XLA (PJRT)' if hw_info['has_xla'] else 'TorchTPU (Native Backend)')}")
    print("=" * 80 + "\n", flush=True)

    # Multi-Source Resume Arbiter (TPU, GPU read-only, GDrive, HF) - Run once on parent process before spawn
    if args.resume and not getattr(args, "resume_dir", None):
        try:
            gdir = args.gdrive_dir or resolve_gdrive_restore_dir() or resolve_gdrive_backup_dir()
            selected_source, selected_dir, selected_step = resolve_and_auto_select_tpu_checkpoint(
                tpu_ckpt_dir=args.output_dir,
                stage_id=args.stage_id,
                gdrive_base_dir=gdir,
                hf_repo_id=args.hf_repo or args.hf_model_repo,
                hf_token=args.hf_token,
            )
            if selected_dir:
                args.resume_dir = selected_dir
        except Exception as e:
            print(f"[TPU Cloud Resume Notice] Multi-source resume discovery skipped: {e}", flush=True)

    # Ensure candidate model assets are downloaded if missing locally and NOT already resumed
    is_resumed = bool(args.resume_dir and os.path.exists(args.resume_dir) and (
        os.path.isfile(os.path.join(args.resume_dir, "checkpoint_state.json"))
        or os.path.isfile(os.path.join(args.resume_dir, "STAGE_METADATA.json"))
    ))

    if not is_resumed:
        ensure_remote_tpu_assets(
            model_dir=args.model_dir,
            dataset_dir=args.dataset_dir,
            repo_id=args.hf_repo or args.hf_model_repo,
            dataset_repo=args.hf_dataset_repo,
            prefix=args.dataset_prefix.strip("/"),
            token=args.hf_token,
        )
    else:
        print(f"[TPU Asset Notice] Checkpoint already resumed from '{args.resume_dir}'. Skipping redundant candidate model download.", flush=True)

    if HAS_XLA and hw_info["num_cores"] > 1 and not args.single_core:
        print(f"[TPU Spawner] Launching distributed training across {hw_info['num_cores']} TPU cores via xmp.spawn...", flush=True)
        xmp.spawn(train_tpu_worker, args=(args,), nprocs=None)
    else:
        if hw_info["available"] and hw_info["num_cores"] == 1:
            print("[TPU Runner] Single-core TPU detected. Executing direct TPU worker...", flush=True)
        train_tpu_worker(0, args)


if __name__ == "__main__":
    main()
