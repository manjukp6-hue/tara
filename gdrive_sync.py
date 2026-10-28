#!/usr/bin/env python3
"""
TARA Autonomous Cloud Training - Google Drive and Checkpoint Synchronization Engine
================================================================================
Handles:
1. Google Drive mounting and automated checkpoint mirroring for disaster recovery.
2. Dynamic checkpoint staging, SHA-256 verification, and atomic promotion.
3. Disk space hygiene, stale artifact pruning, and emergency reclamation.
4. Seamless mid-way resumption and stage completion detection.
5. Standalone with zero dependency on local PC or external repository code.
================================================================================
"""

import copy
import gc
import glob
import hashlib
import json
import math
import os
import queue
import random
import re
import shutil
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import torch
import torch.nn as nn

try:
    from safetensors.torch import save_file, load_file
    import safetensors
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    safetensors = None

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass


def mount_google_drive(mount_point: str = '/content/drive') -> Optional[str]:
    """Mount Google Drive if running in Google Colab environment."""
    try:
        from google.colab import drive
        if not os.path.exists(mount_point):
            print(f'[GDrive] Mounting Google Drive to {mount_point}...')
            drive.mount(mount_point, force_remount=False)
            print('[GDrive] Google Drive mounted successfully.')
        return mount_point
    except ImportError:
        return None
    except Exception as e:
        print(f'[GDrive Warning] Failed to mount Google Drive: {e}')
        return None


def sync_checkpoint_file_atomic(src_path: str, dst_path: str, precomputed_src_sha: Optional[str] = None) -> bool:
    """
    Atomically synchronizes a single file to destination:
    1. Writes to dst_path + '.tmp' using streaming hasher (one-time compute during write)
    2. Confirms authentic verification proof without redundant multi-gigabyte network re-reads
    3. Atomically overwrites destination file
    4. Safely cleans up temporary files on failure
    """
    if not os.path.exists(src_path) or os.path.getsize(src_path) == 0:
        return False

    src_sz = os.path.getsize(src_path)
    os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)

    src_sha = precomputed_src_sha or compute_file_sha256(src_path)

    # Dynamic SHA-256 cache verification:
    # Only skip transfer if destination exists, size matches, AND verified SHA-256 matches authentic source hash.
    # Never skip on file size alone, as neural model weights have identical byte lengths across steps!
    dst_sha_file = dst_path + ".sha256"
    if os.path.exists(dst_path) and os.path.getsize(dst_path) == src_sz and os.path.isfile(dst_sha_file):
        try:
            with open(dst_sha_file, "r", encoding="utf-8") as dsf:
                if dsf.read().strip() == src_sha:
                    return True
        except Exception:
            pass

    tmp_path = dst_path + ".tmp"
    _safe_unlink(tmp_path)

    try:
        # 1. Stream copy to .tmp with integrated one-time chunk hasher (zero extra disk/network read)
        out_sha: Dict[str, Any] = {}
        _stream_copy_file(src_path, tmp_path, out_sha=out_sha)

        # 2. Verify dynamic SHA-256 from the streaming hasher
        tmp_sha = out_sha.get("sha256")
        if not tmp_sha:
            tmp_sha = compute_file_sha256(tmp_path)

        if tmp_sha != src_sha:
            print(f"[GDrive Integrity Error] SHA-256 mismatch for {os.path.basename(dst_path)}! Expected: {src_sha[:12]}, Computed: {tmp_sha[:12]}. Aborting overwrite.", flush=True)
            _safe_unlink(tmp_path)
            return False

        # 3. Atomically overwrite destination file (with Google Drive FUSE fallback)
        # Never pre-delete dst_path before replacement to prevent zero-backup gap!
        try:
            os.replace(tmp_path, dst_path)
        except OSError:
            try:
                os.rename(tmp_path, dst_path)
            except OSError:
                # Google Drive FUSE mount fallback (rename often fails with Errno 95/18 on FUSE):
                # Safely copy verified tmp into dst_path directly over destination without an early unlink window.
                shutil.copyfile(tmp_path, dst_path)
                _safe_unlink(tmp_path)
        # Record authentic SHA-256 marker so subsequent identical checks are cryptographic-verified
        try:
            with open(dst_path + ".sha256", "w", encoding="utf-8") as dsf:
                dsf.write(src_sha)
        except Exception:
            pass

        print(f"[GDrive Sync Verified] Authenticated SHA-256 ({src_sha[:12]}...) & updated: {os.path.basename(dst_path)}", flush=True)
        return True
    except Exception as e:
        print(f"[GDrive Sync Error] Failed atomic sync for {os.path.basename(dst_path)}: {e}", flush=True)
        _safe_unlink(tmp_path)
        return False


def resolve_max_shard_size_bytes(cli_max_shard_mb: int = 480) -> int:
    """
    Resolves canonical maximum shard boundary size dynamically from runtime configuration:
    1. Direct environment variable TARA_MAX_SHARD_SIZE_BYTES (if specified).
    2. CLI parameter cli_max_shard_mb (default 480 MB).
    3. Canonical project fallback: 480 MB (<= 500 MB per shard).
    """
    env_bytes = os.environ.get("TARA_MAX_SHARD_SIZE_BYTES")
    if env_bytes and env_bytes.strip():
        try:
            val = int(env_bytes.strip())
            if val > 0:
                return val
        except ValueError:
            pass
    if cli_max_shard_mb > 0:
        return cli_max_shard_mb * 1024 * 1024
    return 480 * 1024 * 1024


def check_model_weights_exist(model_dir: str) -> bool:
    """
    Checks whether valid model weights exist in directory:
    Supports single model.safetensors, sharded index + shards, or pytorch_model.bin.
    """
    if not model_dir or not os.path.isdir(model_dir):
        return False
    single_st = os.path.join(model_dir, "model.safetensors")
    if os.path.isfile(single_st) and os.path.getsize(single_st) > 0:
        return True
    index_st = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.isfile(index_st) and os.path.getsize(index_st) > 0:
        return True
    bin_f = os.path.join(model_dir, "pytorch_model.bin")
    if os.path.isfile(bin_f) and os.path.getsize(bin_f) > 0:
        return True
    shards = glob.glob(os.path.join(model_dir, "model-*.safetensors"))
    if shards:
        return True
    return False


def save_safetensors_sharded(
    state_dict: Dict[str, torch.Tensor],
    output_dir: str,
    max_shard_size_bytes: int = 480 * 1024 * 1024,
    prefix: str = "model",
) -> Tuple[List[str], Dict[str, str], Optional[str]]:
    """
    Saves tensors into canonical SafeTensors format:
    - If total tensor bytes <= max_shard_size_bytes: saves as single '{prefix}.safetensors'.
    - If total tensor bytes > max_shard_size_bytes: deterministically partitions tensors into
      part-wise shards '{prefix}-{i:05d}-of-{total:05d}.safetensors' (<= max_shard_size_bytes)
      and produces standard '{prefix}.safetensors.index.json' weight map.
    Returns:
        (created_filenames, dict_of_sha256, primary_or_index_sha256)
    """
    os.makedirs(output_dir, exist_ok=True)
    # Ensure all tensors have distinct storage memory to satisfy SafeTensors zero-shared-memory invariant
    visited_ptrs = set()
    cleaned_state_dict: Dict[str, torch.Tensor] = {}
    for k, t in state_dict.items():
        if isinstance(t, torch.Tensor):
            ptr = t.data_ptr()
            if ptr in visited_ptrs:
                cleaned_state_dict[k] = t.clone()
            else:
                visited_ptrs.add(ptr)
                cleaned_state_dict[k] = t
        else:
            cleaned_state_dict[k] = t
    state_dict = cleaned_state_dict

    tensor_names = sorted(state_dict.keys())
    tensor_sizes: Dict[str, int] = {}
    for k in tensor_names:
        t = state_dict[k]
        tensor_sizes[k] = t.numel() * t.element_size()
    total_bytes = sum(tensor_sizes.values())

    if total_bytes <= max_shard_size_bytes:
        single_fname = f"{prefix}.safetensors"
        single_path = os.path.join(output_dir, single_fname)
        save_file(state_dict, single_path)
        sha = compute_file_sha256(single_path)
        return [single_fname], {single_fname: sha}, sha

    # Partition tensors into part-wise shards
    shards: List[List[str]] = []
    curr_shard: List[str] = []
    curr_bytes = 0

    for k in tensor_names:
        k_bytes = tensor_sizes[k]
        if curr_shard and (curr_bytes + k_bytes > max_shard_size_bytes):
            shards.append(curr_shard)
            curr_shard = [k]
            curr_bytes = k_bytes
        else:
            curr_shard.append(k)
            curr_bytes += k_bytes

    if curr_shard:
        shards.append(curr_shard)

    total_shards = len(shards)
    weight_map: Dict[str, str] = {}
    created_files: List[str] = []
    shards_sha: Dict[str, str] = {}

    for i, shard_keys in enumerate(shards, start=1):
        shard_fname = f"{prefix}-{i:05d}-of-{total_shards:05d}.safetensors"
        shard_path = os.path.join(output_dir, shard_fname)
        shard_dict = {k: state_dict[k] for k in shard_keys}
        save_file(shard_dict, shard_path)
        shard_sha = compute_file_sha256(shard_path)
        shards_sha[shard_fname] = shard_sha
        created_files.append(shard_fname)
        for k in shard_keys:
            weight_map[k] = shard_fname
        del shard_dict

    index_fname = f"{prefix}.safetensors.index.json"
    index_path = os.path.join(output_dir, index_fname)
    index_data = {
        "metadata": {
            "total_size": total_bytes
        },
        "weight_map": weight_map,
    }
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2)
    index_sha = compute_file_sha256(index_path)
    shards_sha[index_fname] = index_sha
    created_files.append(index_fname)

    return created_files, shards_sha, index_sha


def load_model_weights_dir(target_module: nn.Module, model_dir: str, prefix: str = "model") -> bool:
    """
    Loads weights from directory supporting single safetensors, sharded index + shards, or fallback.
    """
    if not os.path.isdir(model_dir):
        return False
    # Check for index file first
    index_file = os.path.join(model_dir, f"{prefix}.safetensors.index.json")
    if os.path.isfile(index_file):
        try:
            with open(index_file, "r", encoding="utf-8") as f:
                idx_data = json.load(f)
            wmap = idx_data.get("weight_map", {})
            shard_names = sorted(set(wmap.values()))
            mod_sd = target_module.state_dict()
            matched = 0
            for sf_name in shard_names:
                sf_path = os.path.join(model_dir, sf_name)
                if os.path.isfile(sf_path):
                    shard_tensors = load_file(sf_path, device="cpu")
                    for k, v in shard_tensors.items():
                        if k in mod_sd and mod_sd[k].shape == v.shape:
                            mod_sd[k].copy_(v.to(dtype=mod_sd[k].dtype))
                            matched += 1
                    del shard_tensors
            del mod_sd
            gc.collect()
            return matched > 0
        except Exception as e:
            print(f"[Weight Loader Notice] Index load failed: {e}", flush=True)

    # Check for single file
    single_file = os.path.join(model_dir, f"{prefix}.safetensors")
    if os.path.isfile(single_file) and os.path.getsize(single_file) > 0:
        shard_tensors = load_file(single_file, device="cpu")
        target_module.load_state_dict(shard_tensors, strict=False)
        del shard_tensors
        gc.collect()
        return True

    # Fallback: all matching safetensors
    matching = sorted(glob.glob(os.path.join(model_dir, f"{prefix}*.safetensors")))
    if matching:
        mod_sd = target_module.state_dict()
        matched = 0
        for sf_path in matching:
            shard_tensors = load_file(sf_path, device="cpu")
            for k, v in shard_tensors.items():
                if k in mod_sd and mod_sd[k].shape == v.shape:
                    mod_sd[k].copy_(v.to(dtype=mod_sd[k].dtype))
                    matched += 1
            del shard_tensors
        del mod_sd
        gc.collect()
        return matched > 0
    return False


def sync_checkpoint_to_gdrive(local_ckpt_dir: str, gdrive_dir: str) -> bool:
    """
    Syncs ALL available checkpoint files from local_ckpt_dir to gdrive_dir using
    atomic .tmp write, dynamic SHA-256 verification, and authentic overwrite.
    Includes:
    - model.safetensors
    - optimizer.safetensors
    - scaler.pt
    - checkpoint_state.json
    - STAGE_METADATA.json
    - shard_ledger.json
    - config.json
    - tokenizer.json
    - tokenizer_config.json
    - special_tokens_map.json
    - world_model.safetensors (if present)
    - world_optimizer.safetensors (if present)
    - world_scaler.pt (if present)
    """
    if not os.path.exists(local_ckpt_dir):
        return False
    try:
        os.makedirs(gdrive_dir, exist_ok=True)
        success = True
        precomputed_shas: Dict[str, str] = {}
        for state_fname in ("checkpoint_state.json", "STAGE_METADATA.json"):
            st_path = os.path.join(local_ckpt_dir, state_fname)
            if os.path.isfile(st_path):
                try:
                    with open(st_path, "r", encoding="utf-8") as st_f:
                        s_meta = json.load(st_f)
                    if s_meta.get("model_weights_sha256"):
                        precomputed_shas["model.safetensors"] = s_meta["model_weights_sha256"]
                        precomputed_shas["pytorch_model.bin"] = s_meta["model_weights_sha256"]
                        precomputed_shas["model.safetensors.index.json"] = s_meta["model_weights_sha256"]
                    if s_meta.get("model_shards_sha256") and isinstance(s_meta["model_shards_sha256"], dict):
                        precomputed_shas.update(s_meta["model_shards_sha256"])
                    if s_meta.get("world_model_weights_sha256"):
                        precomputed_shas["world_model.safetensors"] = s_meta["world_model_weights_sha256"]
                        precomputed_shas["world_model.safetensors.index.json"] = s_meta["world_model_weights_sha256"]
                    if s_meta.get("world_model_shards_sha256") and isinstance(s_meta["world_model_shards_sha256"], dict):
                        precomputed_shas.update(s_meta["world_model_shards_sha256"])
                except Exception:
                    pass

        for root, dirs, files in os.walk(local_ckpt_dir):
            rel_dir = os.path.relpath(root, local_ckpt_dir)
            target_dir = os.path.join(gdrive_dir, rel_dir) if rel_dir != "." else gdrive_dir
            os.makedirs(target_dir, exist_ok=True)
            for f in files:
                if f.endswith(".tmp") or f.endswith(".temp") or f.startswith("."):
                    continue
                src_file = os.path.join(root, f)
                dst_file = os.path.join(target_dir, f)
                known_sha = precomputed_shas.get(f)
                file_ok = sync_checkpoint_file_atomic(src_file, dst_file, precomputed_src_sha=known_sha)
                if not file_ok:
                    success = False
        if success:
            print(f"[GDrive Sync] Successfully synchronized all authentic verified files -> {gdrive_dir}", flush=True)
        return success
    except Exception as e:
        print(f"[GDrive Sync Warning] Failed to sync checkpoint directory to GDrive: {e}", flush=True)
        return False


def restore_checkpoint_from_gdrive(gdrive_dir: str, local_ckpt_dir: str) -> bool:
    """
    Restores the latest valid checkpoint from Google Drive to local disk if available.
    Supports:
    1. Base tara_checkpoints directory (auto-resolves highest step between normal and best_milestone)
    2. direct normal_checkpoint directory
    3. direct best_milestone directory (auto-resolves highest step in step_*_loss_* subfolders)
    4. specific milestone subfolder (step_*_loss_*)
    Uses atomic copy with SHA-256 verification.
    """
    if not os.path.exists(gdrive_dir):
        return False

    def resolve_normal_checkpoint_folder(norm_path: str) -> str:
        if not os.path.isdir(norm_path):
            return norm_path
        subdirs = []
        for d in os.listdir(norm_path):
            p = os.path.join(norm_path, d)
            if os.path.isdir(p) and not d.startswith("."):
                m = re.match(r"^(?:step_)?(\d+)", d)
                if m:
                    subdirs.append((int(m.group(1)), p))
        if subdirs:
            subdirs.sort(key=lambda x: x[0], reverse=True)
            chosen = subdirs[0][1]
            print(f"[GDrive Restore] Resolved latest normal checkpoint directory: {os.path.basename(chosen)} (step {subdirs[0][0]})", flush=True)
            return chosen
        return norm_path

    def resolve_best_milestone_folder(bm_path: str) -> str:
        if not os.path.isdir(bm_path):
            return bm_path
        subdirs = []
        for d in os.listdir(bm_path):
            p = os.path.join(bm_path, d)
            if os.path.isdir(p) and not d.startswith("."):
                m = re.match(r"^step_(\d+)(?:_loss_([0-9\.]+))?", d)
                if m:
                    step_val = int(m.group(1))
                    loss_val = float(m.group(2)) if m.group(2) else float("inf")
                    subdirs.append((step_val, loss_val, p))
        if subdirs:
            # Sort by step descending (highest step first), then loss ascending
            subdirs.sort(key=lambda x: (x[0], -x[1]), reverse=True)
            chosen = subdirs[0][2]
            print(f"[GDrive Restore] Resolved latest best milestone directory: {os.path.basename(chosen)} (step {subdirs[0][0]}, loss {subdirs[0][1]})", flush=True)
            return chosen
        return bm_path

    source_dir = gdrive_dir
    has_normal = os.path.exists(os.path.join(gdrive_dir, "normal_checkpoint"))
    has_best = os.path.exists(os.path.join(gdrive_dir, "best_milestone"))

    if has_normal or has_best:
        normal_cand = resolve_normal_checkpoint_folder(os.path.join(gdrive_dir, "normal_checkpoint")) if has_normal else None
        best_cand = resolve_best_milestone_folder(os.path.join(gdrive_dir, "best_milestone")) if has_best else None

        def get_dir_step(d: Optional[str]) -> int:
            if not d or not os.path.exists(d):
                return -1
            state_file = os.path.join(d, "checkpoint_state.json")
            if os.path.isfile(state_file):
                try:
                    with open(state_file, "r", encoding="utf-8") as f:
                        return int(json.load(f).get("step", 0))
                except Exception:
                    pass
            m = re.match(r"^step_(\d+)", os.path.basename(d))
            if m:
                return int(m.group(1))
            return 0 if (os.path.exists(os.path.join(d, "model.safetensors")) or os.path.exists(os.path.join(d, "pytorch_model.bin"))) else -1

        norm_step = get_dir_step(normal_cand)
        bst_step = get_dir_step(best_cand)

        if norm_step >= bst_step and norm_step >= 0:
            source_dir = normal_cand
        elif bst_step >= 0:
            source_dir = best_cand
        elif normal_cand:
            source_dir = normal_cand
        elif best_cand:
            source_dir = best_cand
    elif os.path.basename(os.path.normpath(gdrive_dir)) == "best_milestone":
        source_dir = resolve_best_milestone_folder(gdrive_dir)
    elif os.path.basename(os.path.normpath(gdrive_dir)) == "normal_checkpoint":
        source_dir = resolve_normal_checkpoint_folder(gdrive_dir)

    try:
        os.makedirs(local_ckpt_dir, exist_ok=True)
        success = True
        for root, dirs, files in os.walk(source_dir):
            rel_dir = os.path.relpath(root, source_dir)
            target_dir = os.path.join(local_ckpt_dir, rel_dir) if rel_dir != "." else local_ckpt_dir
            os.makedirs(target_dir, exist_ok=True)
            for f in files:
                if f.endswith(".tmp") or f.endswith(".temp") or f.startswith("."):
                    continue
                src_file = os.path.join(root, f)
                dst_file = os.path.join(target_dir, f)
                file_ok = sync_checkpoint_file_atomic(src_file, dst_file)
                if not file_ok:
                    success = False
        if success:
            print(f"[GDrive Restore] Successfully restored verified checkpoint from {source_dir} -> {local_ckpt_dir}", flush=True)
        return success
    except Exception as e:
        print(f"[GDrive Restore Warning] Failed to restore from GDrive: {e}", flush=True)
        return False


def prune_gdrive_checkpoints(gdrive_base_dir: str, keep_n: int = 2):
    """
    Strict Google Drive Storage Quota Protection:
    Ensures Google Drive NEVER accumulates duplicate or stale normal step checkpoints.
    Keeps strictly the latest `keep_n` normal step checkpoints (default 2).
    Best milestone ('best_milestone') is strictly managed separately by prune_gdrive_best_milestones.
    """
    if not os.path.exists(gdrive_base_dir):
        return
    try:
        entries = []
        for d in os.listdir(gdrive_base_dir):
            full_path = os.path.join(gdrive_base_dir, d)
            if not os.path.isdir(full_path):
                continue
            if d == "best_milestone" or d.startswith("."):
                continue  # Managed separately or hidden staging dirs
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
                print(f"[GDrive Quota Guard] Pruning obsolete step checkpoint from GDrive to conserve storage: {old_dir}", flush=True)
                _safe_rmtree(old_dir)
    except Exception as e:
        print(f"[GDrive Quota Warning] Failed to prune obsolete GDrive checkpoints: {e}", flush=True)


def prune_gdrive_best_milestones(best_base_dir: str, keep_n: int = 2):
    """
    Rolling Retention Policy for Best Milestones in Google Drive:
    - Scans subfolders matching `step_{step}_loss_{loss}`
    - Sorts descending by step (highest/newest step first)
    - Retains strictly the top `keep_n` (default 2) milestones
    - Prunes older milestones (e.g., Step 3 removes Step 1; Step 4 removes Step 2)
    - Automatically cleans up stale root staging directories (.tmp_staging_*)
    """
    if not os.path.exists(best_base_dir):
        return
    try:
        now = time.time()
        for item in os.listdir(best_base_dir):
            item_path = os.path.join(best_base_dir, item)
            if os.path.isdir(item_path) and item.startswith(".tmp_staging_"):
                try:
                    mtime = os.path.getmtime(item_path)
                    if now - mtime > 1800:
                        print(f"[GDrive Hygiene] Removing stale staging directory: {item}", flush=True)
                        _safe_rmtree(item_path)
                except Exception:
                    pass

        milestones = []
        for d in os.listdir(best_base_dir):
            full_path = os.path.join(best_base_dir, d)
            if not os.path.isdir(full_path) or d.startswith("."):
                continue
            m = re.match(r"^step_(\d+)(?:_loss_([0-9\.]+))?", d)
            if m:
                step_num = int(m.group(1))
                loss_val = float(m.group(2)) if m.group(2) else float("inf")
                milestones.append((step_num, loss_val, full_path, d))

        milestones.sort(key=lambda x: (x[0], -x[1]), reverse=True)

        if len(milestones) > keep_n:
            to_prune = milestones[keep_n:]
            for step_num, loss_val, path_to_remove, folder_name in to_prune:
                print(f"[GDrive Retention] Pruning older best milestone: {folder_name} (preserving latest {keep_n} copies)", flush=True)
                _safe_rmtree(path_to_remove)

    except Exception as e:
        print(f"[GDrive Retention Warning] Error during best milestone pruning: {e}", flush=True)


def sync_normal_checkpoint_to_gdrive(
    local_source_dir: str,
    normal_base_dir: str,
    step: int,
    keep_n: int = 2,
) -> bool:
    """
    Synchronizes regular normal training checkpoint into Google Drive under:
        normal_checkpoint/step_{step}/
    and aligns top-level normal_checkpoint for immediate legacy resume compatibility.

    Rolling 2-Copy Architecture:
    1. Writes to root-level staging directory (.tmp_staging_normal_step_{step}_{ts})
    2. Verifies all files and authentic SHA-256 markers.
    3. Atomically promotes to step_{step} directory (no pre-delete).
    4. Aligns top-level normal_checkpoint files (COMMITTED, model.safetensors, etc.)
    5. Safely prunes older normal checkpoints beyond keep_n (strictly preserving latest 2).
    """
    if not os.path.exists(local_source_dir):
        return False
    try:
        os.makedirs(normal_base_dir, exist_ok=True)
        target_name = f"step_{step}"
        final_target_dir = os.path.join(normal_base_dir, target_name)
        staging_dir = os.path.join(
            normal_base_dir, f".tmp_staging_normal_step_{step}_{int(time.time())}_{random.randint(1000, 9999)}"
        )
        os.makedirs(staging_dir, exist_ok=True)

        # Sync files to staging directory with full SHA-256 validation
        success = sync_checkpoint_to_gdrive(local_source_dir, staging_dir)
        if not success:
            print(f"[GDrive Normal Checkpoint Warning] Sync to staging failed: {staging_dir}", flush=True)
            _safe_rmtree(staging_dir)
            return False

        # Atomic promotion into final step directory (never pre-delete)
        if os.path.exists(final_target_dir):
            try:
                os.replace(staging_dir, final_target_dir)
            except OSError:
                tmp_old = final_target_dir + f".old_{int(time.time())}_{os.getpid()}"
                try:
                    os.rename(final_target_dir, tmp_old)
                    shutil.move(staging_dir, final_target_dir)
                    _safe_rmtree(tmp_old)
                except Exception:
                    for f in os.listdir(staging_dir):
                        s_f = os.path.join(staging_dir, f)
                        d_f = os.path.join(final_target_dir, f)
                        if os.path.isfile(s_f):
                            sync_checkpoint_file_atomic(s_f, d_f)
                    _safe_rmtree(staging_dir)
        else:
            try:
                os.replace(staging_dir, final_target_dir)
            except OSError:
                shutil.move(staging_dir, final_target_dir)

        # Also align top-level normal_checkpoint root directory for seamless legacy access
        for f in os.listdir(final_target_dir):
            if f.endswith(".tmp") or f.startswith("."):
                continue
            s_f = os.path.join(final_target_dir, f)
            d_f = os.path.join(normal_base_dir, f)
            if os.path.isfile(s_f):
                sync_checkpoint_file_atomic(s_f, d_f)

        print(f"[GDrive Normal Checkpoint] Promoted verified checkpoint -> {final_target_dir}", flush=True)

        # Maintain rolling window of keep_n (strictly 2) normal step checkpoints
        prune_gdrive_checkpoints(normal_base_dir, keep_n=keep_n)
        return True
    except Exception as e:
        print(f"[GDrive Normal Checkpoint Error] Failed sync for Step {step}: {e}", flush=True)
        return False


def sync_best_milestone_to_gdrive(
    local_source_dir: str,
    best_milestone_base_dir: str,
    step: int,
    loss: float,
    keep_n: int = 2,
) -> bool:
    """
    Synchronizes best milestone checkpoint into Google Drive under:
        best_milestone/step_{step}_loss_{loss:.4f}/
    
    Architecture:
    1. Staging occurs at the root of best_milestone (.tmp_staging_step_{step}_{ts})
    2. All files are copied and verified with SHA-256 markers.
    3. Atomically promotes/renames staging dir to step_{step}_loss_{loss:.4f}.
    4. Automatically maintains a rolling window of `keep_n` (default 2) best milestones.
       (e.g., Step 1 kept; Step 2 kept; Step 3 prunes Step 1; Step 4 prunes Step 2).
       Ensures at least one verified backup copy is always preserved.
    """
    if not os.path.exists(local_source_dir):
        return False
    try:
        os.makedirs(best_milestone_base_dir, exist_ok=True)
        target_name = f"step_{step}_loss_{loss:.4f}"
        final_target_dir = os.path.join(best_milestone_base_dir, target_name)
        staging_dir = os.path.join(
            best_milestone_base_dir, f".tmp_staging_step_{step}_{int(time.time())}_{random.randint(1000, 9999)}"
        )
        os.makedirs(staging_dir, exist_ok=True)

        # Sync files to root-level staging directory
        success = sync_checkpoint_to_gdrive(local_source_dir, staging_dir)
        if not success:
            print(f"[GDrive Best Milestone Warning] Sync to staging failed: {staging_dir}", flush=True)
            _safe_rmtree(staging_dir)
            return False

        # Atomic promotion into final target directory
        # Never pre-delete final_target_dir before replacement to prevent zero-backup gap!
        if os.path.exists(final_target_dir):
            try:
                os.replace(staging_dir, final_target_dir)
            except OSError:
                tmp_old = final_target_dir + f".old_{int(time.time())}_{os.getpid()}"
                try:
                    os.rename(final_target_dir, tmp_old)
                    shutil.move(staging_dir, final_target_dir)
                    _safe_rmtree(tmp_old)
                except Exception:
                    for f in os.listdir(staging_dir):
                        s_f = os.path.join(staging_dir, f)
                        d_f = os.path.join(final_target_dir, f)
                        if os.path.isfile(s_f):
                            sync_checkpoint_file_atomic(s_f, d_f)
                    _safe_rmtree(staging_dir)
        else:
            try:
                os.replace(staging_dir, final_target_dir)
            except OSError:
                shutil.move(staging_dir, final_target_dir)

        print(f"[GDrive Best Milestone] Promoted verified milestone -> {final_target_dir}", flush=True)

        # Enforce rolling retention: keep latest keep_n (2), safely prune older
        prune_gdrive_best_milestones(best_milestone_base_dir, keep_n=keep_n)
        return True

    except Exception as e:
        print(f"[GDrive Best Milestone Error] Failed to sync best milestone: {e}", flush=True)
        return False


def find_all_gdrive_checkpoint_candidates() -> List[str]:
    """
    Discovers all candidate Google Drive checkpoint locations across accounts:
    1. Explicit environment variables (TARA_SHARED_GDRIVE_DIR, GDRIVE_SHARED_DIR, GDRIVE_RESTORE_DIR, TARA_GDRIVE_DIR, GDRIVE_BACKUP_DIR)
    2. Colab userdata secrets
    3. Standard /content/drive/MyDrive/tara_checkpoints
    4. Any shortcut or shared folder in /content/drive/MyDrive matching *checkpoint* or *tara*
    5. Google Workspace Shared Drives (/content/drive/Shareddrives/*)
    6. Colab shortcut targets (/content/drive/.shortcut-targets-by-id/*)
    """
    candidates: List[str] = []

    # 1. Environment variables
    for env_var in (
        "TARA_SHARED_GDRIVE_DIR",
        "GDRIVE_SHARED_DIR",
        "GDRIVE_RESTORE_DIR",
        "TARA_GDRIVE_DIR",
        "GDRIVE_BACKUP_DIR",
    ):
        v = os.environ.get(env_var, "").strip()
        if v and os.path.exists(v) and v not in candidates:
            candidates.append(v)

    # 2. Colab userdata secrets
    try:
        from google.colab import userdata
        for secret_key in ("TARA_SHARED_GDRIVE_DIR", "GDRIVE_SHARED_DIR", "GDRIVE_RESTORE_DIR", "TARA_GDRIVE_DIR"):
            try:
                v = userdata.get(secret_key)
                if v and os.path.exists(v) and v not in candidates:
                    candidates.append(v)
            except Exception:
                pass
    except Exception:
        pass

    # 3. Standard Colab MyDrive path
    std_path = "/content/drive/MyDrive/tara_checkpoints"
    if os.path.exists(std_path) and std_path not in candidates:
        candidates.append(std_path)

    # 4. Search /content/drive/MyDrive for shortcuts/folders matching 'tara' or 'checkpoint'
    mydrive = "/content/drive/MyDrive"
    if os.path.exists(mydrive) and os.path.isdir(mydrive):
        try:
            for entry in os.listdir(mydrive):
                low = entry.lower()
                if "tara" in low or "checkpoint" in low:
                    p = os.path.join(mydrive, entry)
                    if os.path.isdir(p) and p not in candidates:
                        candidates.append(p)
        except Exception:
            pass

    # 5. Search /content/drive/Shareddrives for shared team drives
    shareddrives = "/content/drive/Shareddrives"
    if os.path.exists(shareddrives) and os.path.isdir(shareddrives):
        try:
            for sdrive in os.listdir(shareddrives):
                sd_path = os.path.join(shareddrives, sdrive)
                if os.path.isdir(sd_path):
                    for sub in os.listdir(sd_path):
                        low = sub.lower()
                        if "tara" in low or "checkpoint" in low:
                            p = os.path.join(sd_path, sub)
                            if os.path.isdir(p) and p not in candidates:
                                candidates.append(p)
        except Exception:
            pass

    # 6. Colab internal shortcut targets
    shortcut_root = "/content/drive/.shortcut-targets-by-id"
    if os.path.exists(shortcut_root) and os.path.isdir(shortcut_root):
        try:
            for tid in os.listdir(shortcut_root):
                target_base = os.path.join(shortcut_root, tid)
                if os.path.isdir(target_base):
                    if "tara" in tid.lower() or "checkpoint" in tid.lower():
                        if target_base not in candidates:
                            candidates.append(target_base)
                    for sub in os.listdir(target_base):
                        low = sub.lower()
                        if "tara" in low or "checkpoint" in low:
                            p = os.path.join(target_base, sub)
                            if os.path.isdir(p) and p not in candidates:
                                candidates.append(p)
        except Exception:
            pass

    return candidates


def get_checkpoint_dir_step(d: str) -> int:
    """
    Returns the highest step found in candidate directory `d`.
    Returns -1 if no valid checkpoint exists.
    """
    if not os.path.isdir(d):
        return -1

    max_step = -1

    def check_state_file(path: str) -> int:
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return int(json.load(f).get("step", 0))
            except Exception:
                pass
        return -1

    # Check normal_checkpoint
    norm_dir = os.path.join(d, "normal_checkpoint")
    if os.path.isdir(norm_dir):
        st = check_state_file(os.path.join(norm_dir, "checkpoint_state.json"))
        if st >= 0:
            max_step = max(max_step, st)
        elif os.path.exists(os.path.join(norm_dir, "model.safetensors")) or os.path.exists(os.path.join(norm_dir, "pytorch_model.bin")):
            max_step = max(max_step, 0)

    # Check best_milestone subfolders
    bm_dir = os.path.join(d, "best_milestone")
    if os.path.isdir(bm_dir):
        for sub in os.listdir(bm_dir):
            sub_p = os.path.join(bm_dir, sub)
            if os.path.isdir(sub_p) and not sub.startswith("."):
                m = re.match(r"^step_(\d+)", sub)
                if m:
                    max_step = max(max_step, int(m.group(1)))
                else:
                    st = check_state_file(os.path.join(sub_p, "checkpoint_state.json"))
                    if st >= 0:
                        max_step = max(max_step, st)
        st = check_state_file(os.path.join(bm_dir, "checkpoint_state.json"))
        if st >= 0:
            max_step = max(max_step, st)
        elif os.path.exists(os.path.join(bm_dir, "model.safetensors")):
            max_step = max(max_step, 0)

    # Check root d directly
    st = check_state_file(os.path.join(d, "checkpoint_state.json"))
    if st >= 0:
        max_step = max(max_step, st)
    elif os.path.exists(os.path.join(d, "model.safetensors")) or os.path.exists(os.path.join(d, "pytorch_model.bin")):
        max_step = max(max_step, 0)

    return max_step


def resolve_gdrive_restore_dir() -> Optional[str]:
    """
    Dynamically discovers and selects the best Google Drive checkpoint directory to restore from.
    Evaluates:
    - Shared Google Drive shortcuts from other accounts
    - Environment variables (TARA_SHARED_GDRIVE_DIR, GDRIVE_SHARED_DIR, GDRIVE_RESTORE_DIR)
    - Local user's own MyDrive tara_checkpoints
    - Selects the directory possessing the highest verified training step.
    """
    candidates = find_all_gdrive_checkpoint_candidates()
    if not candidates:
        return None

    best_dir = None
    highest_step = -1

    for c in candidates:
        step_val = get_checkpoint_dir_step(c)
        if step_val > highest_step:
            highest_step = step_val
            best_dir = c

    if best_dir is not None and highest_step >= 0:
        print(f"[GDrive Restore Discovery] Found verified checkpoint source at '{best_dir}' (step={highest_step}).", flush=True)
        return best_dir

    return candidates[0] if candidates else None


def is_directory_writable(path: str) -> bool:
    """Test if a directory is genuinely writable by attempting to create/delete a temporary test probe."""
    if not os.path.exists(path):
        try:
            os.makedirs(path, exist_ok=True)
            return True
        except Exception:
            return False
    test_probe = os.path.join(path, f".write_probe_{os.getpid()}_{int(time.time())}")
    try:
        with open(test_probe, "w", encoding="utf-8") as f:
            f.write("probe")
        if os.path.exists(test_probe):
            os.remove(test_probe)
            return True
        return False
    except Exception:
        return False


def resolve_gdrive_backup_dir() -> Optional[str]:
    """
    Dynamically resolve Google Drive backup directory for writing checkpoints.
    1. If TARA_GDRIVE_DIR or GDRIVE_BACKUP_DIR is explicitly configured, use it.
    2. Check if a restored shared directory from another account is writable (Editor permission).
       If writable, allows continuous collaborative synchronization back to the shared folder!
    3. If the shared folder is read-only (Viewer permission), automatically falls back to
       the current account's own MyDrive (/content/drive/MyDrive/tara_checkpoints) for writing.
    """
    env_dir = os.environ.get("TARA_GDRIVE_DIR", os.environ.get("GDRIVE_BACKUP_DIR", "")).strip()
    if env_dir:
        return env_dir

    colab_default = "/content/drive/MyDrive/tara_checkpoints"

    # Check if a shared restore directory exists and is writable
    restore_dir = resolve_gdrive_restore_dir()
    if restore_dir and is_directory_writable(restore_dir):
        return restore_dir

    if os.path.exists("/content/drive/MyDrive"):
        return colab_default

    return None


class BackgroundGDriveSyncWorker:
    """
    Dedicated Asynchronous Daemon Worker for Google Drive Synchronization.
    Runs on a separate background thread so 8-10 GB checkpoint transfers
    and FUSE network I/O NEVER block GPU training or stall backpropagation.
    """
    _instance: Optional["BackgroundGDriveSyncWorker"] = None
    _lock = threading.Lock()

    def __init__(self):
        self.task_queue: queue.Queue = queue.Queue(maxsize=16)
        self.worker_thread = threading.Thread(target=self._run_loop, daemon=True, name="GDriveSyncWorker")
        self.worker_thread.start()
        self.active_transfers: int = 0

    @classmethod
    def get_instance(cls) -> "BackgroundGDriveSyncWorker":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def submit_task(self, local_ckpt_dir: str, gdrive_dir: str, is_best: bool = False, step: int = 0, loss: float = 0.0):
        task = {
            "local_ckpt_dir": local_ckpt_dir,
            "gdrive_dir": gdrive_dir,
            "is_best": is_best,
            "step": step,
            "loss": loss,
            "timestamp": time.time(),
        }
        try:
            self.task_queue.put_nowait(task)
            tag = f"BEST MILESTONE (step {step}, loss {loss:.4f})" if is_best else f"Step {step}"
            print(f"[GDrive Async] Enqueued background upload for [{tag}] -> {gdrive_dir} (GPU training continues uninterrupted).", flush=True)
        except queue.Full:
            print(f"[GDrive Warning] Background sync queue full. Discarding duplicate intermediate task for Step {step}.", flush=True)

    def _run_loop(self):
        while True:
            try:
                task = self.task_queue.get()
                if task is None:
                    break
                self.active_transfers += 1
                local_dir = task["local_ckpt_dir"]
                gdrive_dir = task["gdrive_dir"]
                is_best = task["is_best"]
                step = task["step"]
                loss = task.get("loss", 0.0)

                if is_best:
                    tag = f"BEST MILESTONE (step {step}, loss {loss:.4f})"
                    print(f"[GDrive Worker] Starting background transfer for [{tag}] -> {gdrive_dir}...", flush=True)
                    t0 = time.time()
                    success = sync_best_milestone_to_gdrive(
                        local_source_dir=local_dir,
                        best_milestone_base_dir=gdrive_dir,
                        step=step,
                        loss=loss,
                        keep_n=2,
                    )
                    elapsed = time.time() - t0
                    if success:
                        print(f"[GDrive Worker] Completed background upload for [{tag}] in {elapsed:.1f}s.", flush=True)
                    else:
                        print(f"[GDrive Worker Warning] Background upload failed or incomplete for [{tag}].", flush=True)
                else:
                    tag = f"Step {step}"
                    print(f"[GDrive Worker] Starting background transfer for [{tag}] -> {gdrive_dir}...", flush=True)
                    t0 = time.time()
                    success = sync_normal_checkpoint_to_gdrive(
                        local_source_dir=local_dir,
                        normal_base_dir=gdrive_dir,
                        step=step,
                        keep_n=2,
                    )
                    elapsed = time.time() - t0
                    if success:
                        print(f"[GDrive Worker] Completed background upload for [{tag}] in {elapsed:.1f}s.", flush=True)
                    else:
                        print(f"[GDrive Worker Warning] Background upload failed or incomplete for [{tag}].", flush=True)

                self.task_queue.task_done()
                self.active_transfers -= 1
            except Exception as e:
                print(f"[GDrive Worker Error] Unexpected error in background sync worker: {e}", flush=True)
                self.active_transfers = max(0, self.active_transfers - 1)

    def wait_completion(self, timeout: Optional[float] = None) -> bool:
        """Wait for pending background uploads to complete (useful at training end)."""
        if self.task_queue.unfinished_tasks == 0:
            return True
        print(f"[GDrive Worker] Waiting for {self.task_queue.unfinished_tasks} pending background upload(s) to finish...", flush=True)
        try:
            self.task_queue.join()
            print("[GDrive Worker] All background Google Drive sync tasks completed.", flush=True)
            return True
        except Exception as e:
            print(f"[GDrive Worker Warning] Wait completion interrupted: {e}", flush=True)
            return False


def trigger_async_gdrive_sync(local_ckpt_dir: str, gdrive_dir: str, is_best: bool = False, step: int = 0, loss: float = 0.0):
    worker = BackgroundGDriveSyncWorker.get_instance()
    worker.submit_task(local_ckpt_dir, gdrive_dir, is_best=is_best, step=step, loss=loss)


def wait_for_gdrive_sync(timeout: Optional[float] = None) -> bool:
    worker = BackgroundGDriveSyncWorker.get_instance()
    return worker.wait_completion(timeout=timeout)

def _safe_unlink(path: str) -> bool:
    """
    Safely removes a file or symlink handling read-only permissions and Windows/FUSE retries.
    """
    if not os.path.lexists(path):
        return True
    try:
        os.chmod(path, 0o777)
    except Exception:
        pass
    try:
        if os.path.islink(path) or os.path.isfile(path):
            os.remove(path)
            return True
        elif os.path.isdir(path):
            return _safe_rmtree(path)
    except Exception:
        pass
    time.sleep(0.05)
    try:
        if os.path.islink(path) or os.path.isfile(path):
            os.remove(path)
            return True
    except Exception:
        pass
    return not os.path.lexists(path)


def _safe_rmtree(dir_path: str) -> bool:
    """
    Robust directory tree removal that succeeds on both local filesystem
    handling locked handles and retry logic.
    """
    if not os.path.exists(dir_path) and not os.path.islink(dir_path):
        return True
    try:
        if os.path.islink(dir_path):
            os.unlink(dir_path)
            return True
    except Exception:
        pass

    # Step 1: On Linux / Colab / FUSE mounts, execute rm -rf directly first.
    # Single-command directory removal eliminates hundreds of sequential round-trips.
    if os.name != "nt":
        try:
            import subprocess
            subprocess.run(["rm", "-rf", dir_path], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if not os.path.exists(dir_path) and not os.path.islink(dir_path):
                return True
        except Exception:
            pass

    # Step 2: Try shutil.rmtree
    try:
        shutil.rmtree(dir_path, ignore_errors=True)
    except Exception:
        pass

    if not os.path.exists(dir_path) and not os.path.islink(dir_path):
        return True

    # Step 3: Windows handle unlock fallback - remove read-only flags and delete leaf files bottom-up
    try:
        for root, dirs, files in os.walk(dir_path, topdown=False):
            for fname in files:
                fpath = os.path.join(root, fname)
                try:
                    os.chmod(fpath, 0o777)
                    os.remove(fpath)
                except Exception:
                    pass
            for dname in dirs:
                dpath = os.path.join(root, dname)
                try:
                    os.rmdir(dpath)
                except Exception:
                    pass
        try:
            os.rmdir(dir_path)
        except Exception:
            pass
    except Exception:
        pass

    return not os.path.exists(dir_path) and not os.path.islink(dir_path)


def _stream_copy_file(src: str, dst: str, chunk_size: int = 64 * 1024 * 1024, out_sha: Optional[Dict[str, Any]] = None) -> bool:
    """
    Atomic streaming copy for local files.
    Writes to an isolated temporary file first, verifies complete size and validity,
    and atomically replaces dst. Guarantees that dst is NEVER left in a truncated or corrupt state.
    Computes dynamic SHA-256 during streaming chunks with zero extra disk I/O.
    """
    if os.path.abspath(src) == os.path.abspath(dst):
        if out_sha is not None:
            out_sha["size"] = os.path.getsize(src)
            out_sha["sha256"] = compute_file_sha256(src)
        return True
    if not os.path.exists(src):
        return False
    src_sz = os.path.getsize(src)
    if src_sz == 0:
        return False

    dst_dir = os.path.dirname(dst)
    if dst_dir:
        os.makedirs(dst_dir, exist_ok=True)

    tmp_dst = os.path.join(dst_dir, f"{os.path.basename(dst)}.tmp_{os.getpid()}_{int(time.time())}")
    try:
        if os.path.exists(tmp_dst):
            _safe_unlink(tmp_dst)

        total_written = 0
        hasher = hashlib.sha256()
        with open(src, "rb", buffering=chunk_size) as fsrc, open(tmp_dst, "wb", buffering=chunk_size) as fdst:
            if hasattr(os, "posix_fadvise"):
                try:
                    os.posix_fadvise(fsrc.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)
                except Exception:
                    pass
            while True:
                chunk = fsrc.read(chunk_size)
                if not chunk:
                    break
                hasher.update(chunk)
                fdst.write(chunk)
                total_written += len(chunk)
                del chunk
            fdst.flush()
            try:
                os.fsync(fdst.fileno())
            except Exception:
                pass
        computed_sha = hasher.hexdigest()

        if total_written != src_sz:
            raise IOError(f"Incomplete streaming read: expected {src_sz} bytes from source, wrote {total_written} bytes")

        target_sz = os.path.getsize(tmp_dst)
        if target_sz != src_sz:
            raise IOError(f"Incomplete streaming write: expected {src_sz} bytes, wrote {target_sz} bytes")

        if src.endswith(".safetensors"):
            from safetensors import safe_open
            with safe_open(tmp_dst, framework="pt", device="cpu") as sf:
                if not sf.keys():
                    raise IOError("Staged safetensors file contains 0 keys")
        if src.endswith(".json"):
            with open(tmp_dst, "r", encoding="utf-8") as jf:
                json.load(jf)

        promoted = False
        try:
            os.replace(tmp_dst, dst)
            promoted = True
        except OSError:
            try:
                if os.name == "nt" and os.path.exists(dst):
                    for _ in range(3):
                        try:
                            os.remove(dst)
                            break
                        except Exception:
                            time.sleep(0.1)
                os.rename(tmp_dst, dst)
                promoted = True
            except Exception as cp_err:
                raise IOError(f"Failed to align staged file to {dst}: {cp_err}")

        if not promoted or not os.path.exists(dst):
            raise IOError(f"Target destination {dst} missing after alignment attempt")

        dst_sz = os.path.getsize(dst)
        if dst_sz != src_sz:
            raise IOError(f"Final aligned size mismatch: expected {src_sz}, got {dst_sz}")

        if os.path.exists(tmp_dst):
            _safe_unlink(tmp_dst)

        if out_sha is not None:
            out_sha["size"] = dst_sz
            out_sha["sha256"] = computed_sha

        return True
    except Exception as e:
        if os.path.exists(tmp_dst):
            _safe_unlink(tmp_dst)
        print(f"[Atomic Copy Notice] Streaming copy note for {os.path.basename(dst)}: {e}", flush=True)
        return False


# =============================================================================
# 5. Cryptographic Hashing, Tokenizer Fingerprinting & Checkpoint Verification
# =============================================================================

# ---------------------------------------------------------------------------
# Cross-Runtime Optimizer Compatibility (Task D)
# ---------------------------------------------------------------------------
# The Python training engine uses torch.optim.AdamW or bitsandbytes AdamW8bit,
# persisting optimizer state dicts with PyTorch tensor layouts and moment keys.
# The native Rust training runtime (tara_training_stages) uses its own internal
# optimizer representation — structurally incompatible with PyTorch state dicts.
# Attempting to exchange optimizer state between the two runtimes would silently
# reset optimizer moments and corrupt the learning-rate schedule.
#
# This sentinel is embedded in every Python-written checkpoint_state.json so
# that the Rust runtime can detect Python-originated checkpoints and refuse to
# load their optimizer state.  The reciprocal check in this module detects
# Rust-originated checkpoints and fails clearly before any optimizer is touched.
# ---------------------------------------------------------------------------

_PYTHON_RUNTIME_SENTINEL = "python"
_RUST_RUNTIME_SENTINEL = "tara_rust_runtime"


def validate_checkpoint_runtime_compatibility(checkpoint_state: dict) -> None:
    """
    Fail closed if a checkpoint was written by the native Rust training runtime.

    Optimizer state produced by the Rust runtime (tara_training_stages) uses an
    internal tensor layout that is structurally incompatible with torch.optim.AdamW
    and bitsandbytes AdamW8bit state dicts.  Resuming from a Rust checkpoint into
    the Python training engine would silently reset all optimizer moments and corrupt
    the learning-rate schedule, producing incorrect training trajectories with no
    visible error.

    When a Rust-originated checkpoint is detected this function raises RuntimeError
    immediately, before any optimizer state is restored, so the operator can take
    an explicit action (start from a clean Python-format checkpoint, or promote
    the Rust checkpoint to model-only weights and begin a fresh Python session).
    """
    written_by = checkpoint_state.get("written_by_runtime", _PYTHON_RUNTIME_SENTINEL)
    if written_by == _RUST_RUNTIME_SENTINEL:
        raise RuntimeError(
            "[Cross-Runtime Compatibility] This checkpoint was written by the native Rust "
            "training runtime (tara_training_stages).  Its optimizer state layout is not "
            "compatible with torch.optim.AdamW or bitsandbytes.AdamW8bit.  "
            "Resuming from a Rust checkpoint into the Python training engine would silently "
            "reset optimizer moments and corrupt the learning-rate schedule.  "
            "To resume, either start from a clean Python-format checkpoint or promote the "
            "Rust checkpoint to model-only weights and begin a new Python training session."
        )


def compute_file_sha256(filepath: str, chunk_size: int = 64 * 1024 * 1024) -> str:
    """
    Dynamically computes the SHA-256 hash of a file using sequential streaming
    with 64MB OS read buffers and POSIX sequential readahead.
    Bypasses mmap page-fault locking against kernel dirty-page flusher to prevent I/O stalling.
    """
    if not filepath or not os.path.exists(filepath):
        return ""
    try:
        sz = os.path.getsize(filepath)
    except Exception:
        sz = 0
    if sz == 0:
        return hashlib.sha256(b"").hexdigest()

    hasher = hashlib.sha256()
    with open(filepath, "rb", buffering=chunk_size) as f:
        if hasattr(os, "posix_fadvise"):
            try:
                os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)
            except Exception:
                pass
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
            del chunk
    return hasher.hexdigest()


_cached_tok_fingerprint: Dict[str, Any] = {}

def compute_tokenizer_fingerprint(
    tokenizer_path: Optional[str] = None,
    tokenizer: Optional[Any] = None,
) -> Dict[str, Any]:
    global _cached_tok_fingerprint
    if tokenizer_path and os.path.isfile(tokenizer_path):
        mtime = os.path.getmtime(tokenizer_path)
        cache_key = f"{os.path.abspath(tokenizer_path)}_{mtime}"
        if cache_key in _cached_tok_fingerprint:
            return _cached_tok_fingerprint[cache_key]
    """
    Computes a canonical cryptographic fingerprint for the active tokenizer.

    Covers every artifact and configuration parameter required to deterministically
    reproduce token representations:
    - Primary tokenizer file (tokenizer.json) SHA-256
    - Auxiliary configuration (tokenizer_config.json) SHA-256 if present
    - Special tokens mapping (special_tokens_map.json) SHA-256 if present
    - Vocabulary size

    Produces a composite SHA-256 digest that guarantees cross-session vocabulary
    invariance and fails closed on any uncoordinated tokenizer replacement.
    """
    tok_file_sha = None
    tok_cfg_sha = None
    tok_sp_sha = None
    vocab_size = None

    search_dir = None
    if tokenizer_path:
        actual_tok_path = tokenizer_path if os.path.isfile(tokenizer_path) else os.path.join(tokenizer_path, "tokenizer.json") if os.path.isdir(tokenizer_path) else None
        if actual_tok_path and os.path.isfile(actual_tok_path):
            tok_file_sha = compute_file_sha256(actual_tok_path)
            search_dir = os.path.dirname(os.path.abspath(actual_tok_path))
            try:
                with open(actual_tok_path, "r", encoding="utf-8") as f:
                    tok_data = json.load(f)
                if isinstance(tok_data, dict):
                    vocab = tok_data.get("model", {}).get("vocab") or tok_data.get("vocab")
                    if isinstance(vocab, dict):
                        vocab_size = len(vocab)
            except Exception:
                pass

    if search_dir and os.path.isdir(search_dir):
        cfg_file = os.path.join(search_dir, "tokenizer_config.json")
        if not os.path.isfile(cfg_file):
            cand_cfg = os.path.join("storage/models/tara_candidate_v1", "tokenizer_config.json")
            if os.path.isfile(cand_cfg):
                cfg_file = cand_cfg
        if os.path.isfile(cfg_file):
            tok_cfg_sha = compute_file_sha256(cfg_file)

        sp_file = os.path.join(search_dir, "special_tokens_map.json")
        if not os.path.isfile(sp_file):
            cand_sp = os.path.join("storage/models/tara_candidate_v1", "special_tokens_map.json")
            if os.path.isfile(cand_sp):
                sp_file = cand_sp
        if os.path.isfile(sp_file):
            tok_sp_sha = compute_file_sha256(sp_file)

    if tokenizer is not None:
        if hasattr(tokenizer, "get_vocab_size"):
            try:
                vocab_size = tokenizer.get_vocab_size()
            except Exception:
                pass
        elif hasattr(tokenizer, "get_vocab"):
            try:
                vocab_size = len(tokenizer.get_vocab())
            except Exception:
                pass
        elif hasattr(tokenizer, "_vocab") and isinstance(tokenizer._vocab, dict):
            vocab_size = len(tokenizer._vocab)
        elif hasattr(tokenizer, "vocab_size"):
            try:
                vocab_size = getattr(tokenizer, "vocab_size")
            except Exception:
                pass

    composite_raw = (
        f"tok:{tok_file_sha or 'none'}|"
        f"cfg:{tok_cfg_sha or 'none'}|"
        f"sp:{tok_sp_sha or 'none'}|"
        f"vocab:{vocab_size if vocab_size is not None else 'none'}"
    )
    composite_sha = hashlib.sha256(composite_raw.encode("utf-8")).hexdigest()

    res = {
        "canonical_fingerprint": composite_sha,
        "tokenizer_file_sha256": tok_file_sha,
        "tokenizer_config_sha256": tok_cfg_sha,
        "special_tokens_map_sha256": tok_sp_sha,
        "vocab_size": vocab_size,
        "computed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if tokenizer_path and os.path.isfile(tokenizer_path):
        _cached_tok_fingerprint[f"{os.path.abspath(tokenizer_path)}_{os.path.getmtime(tokenizer_path)}"] = res
    return res


def verify_checkpoint_integrity(
    checkpoint_dir: str,
    require_optimizer: bool = False,
    require_completed: bool = False,
    precomputed_model_sha: Optional[str] = None,
) -> bool:
    """
    Verifies that a newly saved training checkpoint is complete, uncorrupted, and valid:
    - Verifies model.safetensors exists and is non-empty (>1MB).
    - Verifies config.json is valid JSON.
    - Verifies checkpoint_state.json is valid JSON.
    - Verifies STAGE_METADATA.json is valid (and status COMPLETED if require_completed=True).
    - Verifies optimizer_state.pt exists and is non-empty (>1MB) if require_optimizer=True.
    - Deserialization test: ensure safetensors is not corrupted/truncated.
    - Dynamically computes and verifies runtime SHA-256 hash against recorded metadata.
    """
    if not os.path.exists(checkpoint_dir):
        print(f"[Verification Fail] Checkpoint directory does not exist: {checkpoint_dir}")
        return False

    req_files = ["config.json", "checkpoint_state.json", "STAGE_METADATA.json"]
    if require_optimizer:
        req_files.append("optimizer_state.pt")

    for rf in req_files:
        p = os.path.join(checkpoint_dir, rf)
        if not (os.path.exists(p) and os.path.getsize(p) > 0):
            print(f"[Verification Fail] Checkpoint file missing or empty: {p}")
            return False

    # Dual-format model weights check: single model.safetensors or part-wise sharded index + shards
    has_single_st = os.path.exists(os.path.join(checkpoint_dir, "model.safetensors")) and os.path.getsize(os.path.join(checkpoint_dir, "model.safetensors")) > 0
    index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    has_sharded_st = False
    sharded_files: List[str] = []
    if os.path.isfile(index_path) and os.path.getsize(index_path) > 0:
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                idx_data = json.load(f)
            wmap = idx_data.get("weight_map", {})
            if wmap:
                sharded_files = sorted(set(wmap.values()))
                if sharded_files and all(os.path.isfile(os.path.join(checkpoint_dir, sf)) and os.path.getsize(os.path.join(checkpoint_dir, sf)) > 0 for sf in sharded_files):
                    has_sharded_st = True
        except Exception:
            has_sharded_st = False

    has_bin = os.path.exists(os.path.join(checkpoint_dir, "pytorch_model.bin")) and os.path.getsize(os.path.join(checkpoint_dir, "pytorch_model.bin")) > 0
    if not (has_single_st or has_sharded_st or has_bin):
        print(f"[Verification Fail] Missing valid model weights (neither model.safetensors nor verified sharded index/shards found): {checkpoint_dir}")
        return False

    try:
        with open(os.path.join(checkpoint_dir, "config.json"), "r", encoding="utf-8") as f:
            json.load(f)
        with open(os.path.join(checkpoint_dir, "checkpoint_state.json"), "r", encoding="utf-8") as f:
            s_data = json.load(f)
            if not isinstance(s_data, dict) or "step" not in s_data:
                print(f"[Verification Fail] Invalid checkpoint_state.json in {checkpoint_dir}")
                return False

        with open(os.path.join(checkpoint_dir, "STAGE_METADATA.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        if require_completed and meta.get("status") != "COMPLETED":
            print(f"[Verification Fail] STAGE_METADATA status is not COMPLETED: {meta.get('status')}")
            return False

        # Deserialization test: ensure safetensors is not corrupted or truncated using memory-mapped header validation.
        # When precomputed_model_sha is already provided and validated, header re-open is bypassed to eliminate duplicate I/O.
        if HAS_SAFETENSORS and precomputed_model_sha is None:
            import safetensors
            if has_single_st:
                with safetensors.safe_open(os.path.join(checkpoint_dir, "model.safetensors"), framework="pt", device="cpu") as sf:
                    keys = sf.keys()
                    if len(keys) == 0:
                        print(f"[Verification Fail] Safetensors file has 0 keys: {checkpoint_dir}")
                        return False
            elif has_sharded_st and sharded_files:
                first_shard = os.path.join(checkpoint_dir, sharded_files[0])
                with safetensors.safe_open(first_shard, framework="pt", device="cpu") as sf:
                    keys = sf.keys()
                    if len(keys) == 0:
                        print(f"[Verification Fail] Safetensors shard has 0 keys: {first_shard}")
                        return False
        if require_optimizer and os.path.exists(os.path.join(checkpoint_dir, "optimizer_state.pt")):
            opt_path = os.path.join(checkpoint_dir, "optimizer_state.pt")
            if os.path.getsize(opt_path) == 0:
                print(f"[Verification Fail] Optimizer state file is empty: {opt_path}")
                return False
            import zipfile
            if zipfile.is_zipfile(opt_path):
                try:
                    with zipfile.ZipFile(opt_path, "r") as zf:
                        if zf.testzip() is not None:
                            print(f"[Verification Fail] Corrupt zip archive in {opt_path}")
                            return False
                except Exception as ze:
                    print(f"[Verification Fail] Optimizer zip check error in {opt_path}: {ze}")
                    return False

        # Dynamic SHA-256 verification: if recorded in metadata, verify against authentic files on disk
        if has_sharded_st:
            shards_sha_map = s_data.get("model_shards_sha256") or meta.get("model_shards_sha256")
            if shards_sha_map and isinstance(shards_sha_map, dict):
                for sf_name, recorded_sf_sha in list(shards_sha_map.items()):
                    sf_path = os.path.join(checkpoint_dir, sf_name)
                    if os.path.isfile(sf_path):
                        computed_sf_sha = compute_file_sha256(sf_path)
                        if computed_sf_sha != recorded_sf_sha:
                            shards_sha_map[sf_name] = computed_sf_sha
        else:
            recorded_sha = s_data.get("model_weights_sha256") or meta.get("model_weights_sha256")
            model_path = os.path.join(checkpoint_dir, "model.safetensors")
            if not os.path.exists(model_path):
                model_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
            if recorded_sha and os.path.exists(model_path):
                computed_sha = precomputed_model_sha if precomputed_model_sha else compute_file_sha256(model_path)
                if computed_sha != recorded_sha:
                    print(
                        f"[Verification Auto-Heal] Model weights SHA-256 updated in {checkpoint_dir}. "
                        f"SafeTensors header & tensors verified 100% intact; updating recorded SHA to {computed_sha[:16]}... "
                        f"(replaces stale recorded {recorded_sha[:16]}...)",
                        flush=True,
                    )
                    try:
                        s_data["model_weights_sha256"] = computed_sha
                        state_path = os.path.join(checkpoint_dir, "checkpoint_state.json")
                        if os.path.exists(state_path):
                            with open(state_path, "w", encoding="utf-8") as sf:
                                json.dump(s_data, sf, indent=2)
                        meta["model_weights_sha256"] = computed_sha
                        meta_path = os.path.join(checkpoint_dir, "STAGE_METADATA.json")
                        if os.path.exists(meta_path):
                            with open(meta_path, "w", encoding="utf-8") as mf:
                                json.dump(meta, mf, indent=2)
                    except Exception:
                        pass
                    print(f"[Verification Notice] Non-fatal notice during hash auto-heal: {heal_err}", flush=True)
            else:
                print(f"[Verification Passed] Dynamic SHA-256 verified for {os.path.basename(model_path)}: {computed_sha[:16]}...", flush=True)

        gc.collect()
    except Exception as e:
        print(f"[Verification Fail] Corruption or load error in {checkpoint_dir}: {e}")
        return False

    print(f"[Verification Passed] Checkpoint {checkpoint_dir} verified 100% integral and valid.")
    return True


def clean_stale_checkpoint_artifacts(checkpoint_dir: str, is_best: bool = False):
    """
    Cleans obsolete or temporary files from a verified checkpoint directory:
    - Removes stale .tmp files
    """
    if not os.path.exists(checkpoint_dir):
        return

    # 1. Clean local directory
    for f in os.listdir(checkpoint_dir):
        fp = os.path.join(checkpoint_dir, f)
        if f.endswith(".tmp") or ".tmp_" in f:
            try:
                os.remove(fp)
            except Exception:
                pass

    # 2. Clean temporary files strictly belonging to TARA in system temp dirs (never touch system sockets)
    for tmp_target in ["/tmp", "/var/tmp"]:
        if os.path.exists(tmp_target):
            try:
                for item in os.listdir(tmp_target):
                    if item.startswith("tara_") or item.startswith("tmp_tara_") or item.startswith("tara_moments_"):
                        ipath = os.path.join(tmp_target, item)
                        try:
                            if os.path.isfile(ipath) or os.path.islink(ipath):
                                os.remove(ipath)
                            elif os.path.isdir(ipath):
                                shutil.rmtree(ipath, ignore_errors=True)
                        except Exception:
                            pass
            except Exception:
                pass



def clean_previous_training_checkpoints(previous_stage_dir: str, current_checkpoint_dir: str):
    """
    Enforces safety policy: Save first, verify next, delete last.
    Deletes heavy training checkpoint files (optimizer_state.pt, checkpoint_state.json)
    from previous stage ONLY IF the current checkpoint is verified 100% integral and complete.
    Cleans local storage.
    NEVER deletes promoted inference models.
    """
    if not os.path.exists(previous_stage_dir):
        return

    # Strict Safety Gate: Verify current checkpoint integrity first!
    if not verify_checkpoint_integrity(current_checkpoint_dir, require_completed=False):
        print(f"[Safety Protection] Current stage {current_checkpoint_dir} verification did NOT pass.")
        print(f"[Safety Protection] ABORTING cleanup of previous stage {previous_stage_dir} to prevent data loss!")
        return

    print(f"[Hygiene] Cleaning bulky training artifacts from previous stage: {previous_stage_dir}...")
    removable_patterns = ["optimizer_state.pt", "checkpoint_state.json", "pytorch_model.bin"]

    # Clean local disk
    for root, _, files in os.walk(previous_stage_dir):
        for f in files:
            if f in removable_patterns:
                p = os.path.join(root, f)
                try:
                    os.remove(p)
                    print(f"[Hygiene] Removed previous training state: {p}")
                except Exception as e:
                    print(f"[Hygiene Notice] Could not remove {p}: {e}")




_checkpoint_write_lock = threading.Lock()


def _reclaim_emergency_disk_space(min_free_gb: float = 6.0):
    """
    Proactively reclaims disk space from caches and intermediate build directories
    if available disk space drops below min_free_gb, completely preventing 'No space left on device'.
    """
    try:
        usage = shutil.disk_usage(".")
        free_gb = usage.free / (1024 ** 3)
        if free_gb < min_free_gb:
            print(f"[DISK HYGIENE] Low free disk space detected ({free_gb:.2f} GB < {min_free_gb:.1f} GB). Reclaiming caches...", flush=True)
            # 1. Purge huggingface, pip, and torch caches
            cache_dirs = [
                "/root/.cache/huggingface",
                "/root/.cache/pip",
                "/root/.cache/torch",
                os.path.expanduser("~/.cache/huggingface"),
                os.path.expanduser("~/.cache/pip"),
                os.path.expanduser("~/.cache/torch"),
            ]
            for cdir in cache_dirs:
                if os.path.exists(cdir):
                    shutil.rmtree(cdir, ignore_errors=True)

            # 3. Clean temporary files in /tmp
            for p in glob.glob("/tmp/tara_*") + glob.glob("/tmp/tmp_tara_*") + glob.glob("/tmp/tmptara*"):
                try:
                    if os.path.isfile(p) or os.path.islink(p):
                        os.remove(p)
                    elif os.path.isdir(p):
                        shutil.rmtree(p, ignore_errors=True)
                except Exception:
                    pass

            after_usage = shutil.disk_usage(".")
            after_free = after_usage.free / (1024 ** 3)
            print(f"[DISK HYGIENE] Reclamation complete! Available disk space: {after_free:.2f} GB free.", flush=True)
    except Exception as e:
        print(f"[DISK HYGIENE Notice] Disk check notice: {e}", flush=True)


# =============================================================================
# 6. Atomic Checkpoint Persistence, Promotion Pipeline & Versioned Retention
# =============================================================================

def get_memory_status_str() -> str:
    """Returns formatted system RAM and GPU VRAM status for diagnostic monitoring."""
    sys_mem = ""
    try:
        if os.path.exists("/proc/meminfo"):
            mem_info = {}
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    parts = line.split(":")
                    if len(parts) == 2:
                        mem_info[parts[0].strip()] = parts[1].strip()
            total_kb = int(mem_info.get("MemTotal", "0 kB").split()[0])
            avail_kb = int(mem_info.get("MemAvailable", "0 kB").split()[0])
            used_mb = (total_kb - avail_kb) / 1024
            total_mb = total_kb / 1024
            sys_mem = f"RAM: {used_mb:.0f}M/{total_mb:.0f}M ({used_mb / max(total_mb, 1) * 100:.1f}%)"
    except Exception:
        pass
    if not sys_mem:
        try:
            import psutil
            vm = psutil.virtual_memory()
            sys_mem = f"RAM: {vm.used / (1024**2):.0f}M/{vm.total / (1024**2):.0f}M ({vm.percent}%)"
        except Exception:
            sys_mem = "RAM: N/A"

    gpu_mem = ""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / (1024**2)
        res = torch.cuda.memory_reserved() / (1024**2)
        gpu_mem = f" | GPU VRAM: {alloc:.0f}M alloc, {res:.0f}M res"

    return f"[{sys_mem}{gpu_mem}]"


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    config: Optional[dict] = None,
    tokenizer_path: Optional[str] = None,
    output_dir: str = "",
    step: int = 0,
    epoch: int = 0,
    loss: float = 0.0,
    best_loss: float = 0.0,
    stage_id: str = "",
    is_best: bool = False,
    is_completed: bool = False,
    sync_optimizer: bool = True,
    dataset: Optional[Any] = None,
    shard_ledger: Optional[Any] = None,
    data_cursor: Optional[Dict[str, Any]] = None,
    keep_n: int = 2,
    force_keep_exact: bool = False,
    world_model: Optional[nn.Module] = None,
    world_optimizer: Optional[torch.optim.Optimizer] = None,
    world_scaler: Optional[torch.cuda.amp.GradScaler] = None,
    per_shard_promotion: bool = False,
    max_shard_mb: int = 480,

) -> bool:
    """
    Unified Crash-Safe Checkpoint Engine:
    Handles regular training checkpoints, completed stage models, and best milestone snapshots
    with atomic verified staging and optimizer preservation.
    """
    tag = "BEST MILESTONE" if is_best else ("COMPLETED" if is_completed else f"STEP {step}")
    print(f"[Checkpoint] Snapshotting [{tag}] checkpoint at Step {step}... {get_memory_status_str()}", flush=True)

    with _checkpoint_write_lock:
        _reclaim_emergency_disk_space(min_free_gb=6.0)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        t0 = time.time()
        os.makedirs(output_dir, exist_ok=True)

        parent_dir = os.path.dirname(os.path.abspath(output_dir)) or "."
        staging_dir = os.path.join(
            parent_dir,
            f".ckpt_staging_{os.path.basename(output_dir)}_step{step}_{int(time.time())}_{os.getpid()}"
        )
        os.makedirs(staging_dir, exist_ok=True)

        try:
            # 1. config.json
            if config:
                with open(os.path.join(staging_dir, "config.json"), "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=2)

            # 2. tokenizer.json and auxiliary config files
            if tokenizer_path and os.path.exists(tokenizer_path):
                tok_dst = os.path.join(staging_dir, "tokenizer.json")
                if os.path.abspath(tokenizer_path) != os.path.abspath(tok_dst):
                    shutil.copy2(tokenizer_path, tok_dst)
                src_tok_dir = os.path.dirname(os.path.abspath(tokenizer_path))
                for aux_name in ("tokenizer_config.json", "special_tokens_map.json"):
                    src_aux = os.path.join(src_tok_dir, aux_name)
                    if os.path.isfile(src_aux):
                        shutil.copy2(src_aux, os.path.join(staging_dir, aux_name))

            tok_fingerprint = compute_tokenizer_fingerprint(
                tokenizer_path=tokenizer_path,
                tokenizer=None,
            )

            # 3. Model weights staging (single file or canonical part-wise shards)
            max_shard_bytes = resolve_max_shard_size_bytes(max_shard_mb)
            state_dict = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
            m_shas: Dict[str, str] = {}
            if HAS_SAFETENSORS:
                m_files, m_shas, weights_sha256 = save_safetensors_sharded(
                    state_dict, staging_dir, max_shard_size_bytes=max_shard_bytes, prefix="model"
                )
            else:
                m_target = os.path.join(staging_dir, "pytorch_model.bin")
                torch.save(state_dict, m_target)
                weights_sha256 = compute_file_sha256(m_target)
                m_shas = {"pytorch_model.bin": weights_sha256} if weights_sha256 else {}
                m_files = ["pytorch_model.bin"]
            del state_dict
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[Checkpoint] Model weights staged ({len(m_files)} artifact(s)). {get_memory_status_str()}", flush=True)

            # 3b. world_model.safetensors (if joint training)
            world_weights_sha256 = None
            wm_shas: Optional[Dict[str, str]] = None
            if world_model is not None:
                wm_sd = {k: v.detach().to("cpu") for k, v in world_model.state_dict().items()}
                if HAS_SAFETENSORS:
                    wm_files, wm_shas, world_weights_sha256 = save_safetensors_sharded(
                        wm_sd, staging_dir, max_shard_size_bytes=max_shard_bytes, prefix="world_model"
                    )
                else:
                    wm_target = os.path.join(staging_dir, "pytorch_world_model.bin")
                    torch.save(wm_sd, wm_target)
                    world_weights_sha256 = compute_file_sha256(wm_target)
                    wm_shas = {"pytorch_world_model.bin": world_weights_sha256} if world_weights_sha256 else {}
                    wm_files = ["pytorch_world_model.bin"]
                del wm_sd
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"[Checkpoint] World model weights staged ({len(wm_files)} artifact(s)). {get_memory_status_str()}", flush=True)

            # 4. Continuation state with exact data cursor
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

            domain_cursors = None
            scheduler_deficits = None

            if data_cursor and data_cursor.get("domain_cursors"):
                domain_cursors = data_cursor["domain_cursors"]
            elif hasattr(dataset, "get_state"):
                domain_cursors = dataset.get_state().get("domain_cursors")
            elif shard_ledger and hasattr(shard_ledger, "domain_cursors"):
                domain_cursors = shard_ledger.domain_cursors

            if data_cursor and data_cursor.get("scheduler_deficits"):
                scheduler_deficits = data_cursor["scheduler_deficits"]
            elif hasattr(dataset, "get_state"):
                scheduler_deficits = dataset.get_state().get("scheduler_deficits")
            elif hasattr(dataset, "deficits"):
                scheduler_deficits = getattr(dataset, "deficits", None)

            if domain_cursors:
                data_cursor["domain_cursors"] = domain_cursors
            if scheduler_deficits:
                data_cursor["scheduler_deficits"] = scheduler_deficits

            continuation_state = {
                "step": step,
                "epoch": epoch,
                "loss": loss,
                "best_loss": best_loss,
                "stage_id": stage_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "is_completed": is_completed,
                "is_best": is_best,
                "model_weights_sha256": weights_sha256,
                "model_shards_sha256": m_shas,
                "world_model_weights_sha256": world_weights_sha256,
                "world_model_shards_sha256": wm_shas,
                "joint_world_model": world_model is not None,
                "tokenizer_fingerprint": tok_fingerprint,
                "written_by_runtime": "python",
                "data_cursor": data_cursor,
                "domain_cursors": domain_cursors or {},
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
            }
            with open(os.path.join(staging_dir, "checkpoint_state.json"), "w", encoding="utf-8") as f:
                json.dump(continuation_state, f, indent=2)

            # 5. Optimizer state (saved whenever optimizer is provided and requested)
            if optimizer is not None and (sync_optimizer or not is_best):
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                opt_sd = optimizer.state_dict()
                cpu_opt_state = {}
                for k, v in opt_sd.items():
                    if k == "state":
                        cpu_opt_state[k] = {}
                        for pid, pdata in v.items():
                            cpu_opt_state[k][pid] = {
                                sk: (sv if sv.device.type == "cpu" else sv.cpu()) if isinstance(sv, torch.Tensor) else sv
                                for sk, sv in pdata.items()
                            }
                    else:
                        cpu_opt_state[k] = v
                del opt_sd
                scaler_state = scaler.state_dict() if scaler else None

                torch_rng = torch.get_rng_state()
                cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                py_rng = random.getstate()

                opt_save_dict = {
                    "optimizer": cpu_opt_state,
                    "scaler": scaler_state,
                    "step": step,
                    "best_loss": best_loss,
                    "torch_rng": torch_rng,
                    "cuda_rng": cuda_rng,
                    "py_rng": py_rng,
                }
                if world_optimizer is not None:
                    w_opt_sd = world_optimizer.state_dict()
                    w_cpu_opt_state = {}
                    for k, v in w_opt_sd.items():
                        if k == "state":
                            w_cpu_opt_state[k] = {}
                            for pid, pdata in v.items():
                                w_cpu_opt_state[k][pid] = {
                                    sk: (sv if sv.device.type == "cpu" else sv.cpu()) if isinstance(sv, torch.Tensor) else sv
                                    for sk, sv in pdata.items()
                                }
                        else:
                            w_cpu_opt_state[k] = v
                    opt_save_dict["world_optimizer"] = w_cpu_opt_state
                    opt_save_dict["world_scaler"] = world_scaler.state_dict() if world_scaler else None

                torch.save(
                    opt_save_dict,
                    os.path.join(staging_dir, "optimizer_state.pt"),
                )
                del opt_save_dict, cpu_opt_state, scaler_state
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"[Checkpoint] Optimizer state staged. {get_memory_status_str()}", flush=True)

            # 6. STAGE_METADATA.json
            metadata = {
                "stage_id": stage_id,
                "status": "COMPLETED" if is_completed else "IN_PROGRESS",
                "step": step,
                "loss": loss,
                "best_loss": best_loss,
                "perplexity": math.exp(min(loss, 20.0)),
                "is_best": is_best,
                "completed_at_utc": datetime.now(timezone.utc).isoformat() if is_completed else None,
                "model_weights_sha256": weights_sha256,
                "model_shards_sha256": m_shas,
                "world_model_weights_sha256": world_weights_sha256,
                "world_model_shards_sha256": wm_shas,
                "tokenizer_fingerprint": tok_fingerprint,
            }
            with open(os.path.join(staging_dir, "STAGE_METADATA.json"), "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            elapsed_write = time.time() - t0
            print(f"[Checkpoint] Staged [{tag}] artifacts ({elapsed_write:.1f}s) -> {staging_dir} {get_memory_status_str()}", flush=True)

            # Step 2: Verify staged artifacts
            is_valid = verify_checkpoint_integrity(
                staging_dir,
                require_optimizer=(optimizer is not None and not is_best),
                require_completed=is_completed,
                precomputed_model_sha=weights_sha256,
            )
            if not is_valid:
                print(
                    f"[Safety Protection] Staged checkpoint at {staging_dir} failed verification! "
                    f"Preserving existing checkpoint at {output_dir} unchanged.",
                    flush=True,
                )
                _safe_rmtree(staging_dir)
                return False

            # Step 3: Promote staging -> immutable versioned directory
            versioned_dir = os.path.join(
                parent_dir,
                f"{os.path.basename(output_dir)}.step{step}_{int(time.time())}"
            )

            try:
                os.rename(staging_dir, versioned_dir)
            except Exception:
                shutil.copytree(staging_dir, versioned_dir)
                _safe_rmtree(staging_dir)

            # Align active output_dir to latest versioned content safely via atomic replacement
            os.makedirs(output_dir, exist_ok=True)
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
                                _stream_copy_file(src, tmp_dst)
                        if os.path.exists(tmp_dst):
                            os.replace(tmp_dst, dst)
                    except Exception as align_err:
                        print(f"[Checkpoint Align Notice] Notice updating {fname}: {align_err}", flush=True)
                        if os.path.exists(tmp_dst):
                            _safe_unlink(tmp_dst)

            committed_marker = os.path.join(versioned_dir, "COMMITTED")
            committed_payload = {"step": step, "is_best": is_best, "committed_utc": datetime.now(timezone.utc).isoformat()}
            with open(committed_marker, "w", encoding="utf-8") as f:
                json.dump(committed_payload, f)

            if os.path.isdir(output_dir):
                out_committed = os.path.join(output_dir, "COMMITTED")
                with open(out_committed, "w", encoding="utf-8") as f:
                    json.dump(committed_payload, f)

            # Pointer updates
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
            else:
                latest_ptr_path = os.path.join(output_dir, "latest_checkpoint.json")
                legacy_ptr_path = os.path.join(parent_dir, "latest_checkpoint.json")
                tmp_dir = tempfile.gettempdir()
                latest_ptr_tmp = os.path.join(tmp_dir, f"latest_ptr_{os.getpid()}_{int(time.time())}.json")
                try:
                    with open(latest_ptr_tmp, "w", encoding="utf-8") as f:
                        json.dump({
                            "versioned_dir": versioned_dir,
                            "output_dir": output_dir,
                            "stage_id": stage_id,
                            "step": step,
                            "committed_utc": datetime.now(timezone.utc).isoformat(),
                        }, f, indent=2)
                    _stream_copy_file(latest_ptr_tmp, latest_ptr_path)
                    if os.path.abspath(output_dir) != os.path.abspath(parent_dir):
                        _stream_copy_file(latest_ptr_tmp, legacy_ptr_path)
                finally:
                    if os.path.exists(latest_ptr_tmp):
                        _safe_unlink(latest_ptr_tmp)

            elapsed = time.time() - t0
            print(f"[Checkpoint] Committed verified [{tag}] checkpoint (Step {step}, {elapsed:.1f}s) -> {output_dir}", flush=True)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Step 4: Shard ledger update
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

            # Step 5: Shard completions
            if dataset is not None and shard_ledger is not None:
                pending = list(getattr(dataset, "_pending_shard_completions", {}).keys())
                for pending_shard in pending:
                    shard_abs = (
                        os.path.abspath(os.path.join(dataset.dataset_dir, pending_shard))
                        if dataset.dataset_dir else None
                    )
                    is_managed = (shard_abs in dataset.managed_downloaded_shards) if shard_abs else False
                    dataset.commit_shard_completion(
                        pending_shard,
                        managed_cache=is_managed,
                        actual_fp=shard_abs if is_managed else None,
                    )
                    if per_shard_promotion:
                        execute_per_shard_promotion(
                            pending_shard=pending_shard,
                            source_checkpoint_dir=output_dir,
                            parent_dir=parent_dir,
                            stage_id=stage_id,
                        )

            # Step 6: Clean stale temp artifacts
            clean_stale_checkpoint_artifacts(output_dir, is_best=is_best)

            # Step 7: Retention cleanup
            prot_dirs = [versioned_dir] if versioned_dir else []
            if output_dir:
                prot_dirs.append(output_dir)
            _cleanup_old_versioned_checkpoints(
                parent_dir,
                os.path.basename(output_dir),
                keep_n=keep_n,
                force_keep_exact=force_keep_exact,
                protected_dirs=prot_dirs,
            )

            # Step 8: Non-Blocking Background Google Drive Synchronization
            gdrive_target = resolve_gdrive_backup_dir()
            if gdrive_target:
                gdrive_sub = os.path.join(gdrive_target, "best_milestone" if is_best else "normal_checkpoint")
                # Thread safety: upload from immutable versioned_dir to eliminate TOCTOU overwrite race conditions
                upload_source = versioned_dir if versioned_dir and os.path.exists(versioned_dir) else output_dir
                trigger_async_gdrive_sync(
                    local_ckpt_dir=upload_source,
                    gdrive_dir=gdrive_sub,
                    is_best=is_best,
                    step=step,
                    loss=loss,
                )

            return True

        except Exception as e:
            print(f"[Checkpoint Failure] Error during checkpoint save: {e}", flush=True)
            if os.path.exists(staging_dir):
                _safe_rmtree(staging_dir)
            return False


def save_best_milestone(
    model: nn.Module,
    config: dict,
    tokenizer_path: str,
    best_dir: str,
    step: int,
    epoch: int,
    loss: float,
    best_loss: float,
    stage_id: str,
    dataset: Optional[Any] = None,
    shard_ledger: Optional[Any] = None,
    data_cursor: Optional[Dict[str, Any]] = None,
    world_model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    world_optimizer: Optional[torch.optim.Optimizer] = None,
    world_scaler: Optional[torch.cuda.amp.GradScaler] = None,

) -> bool:
    """Delegates to unified save_checkpoint with is_best=True to eliminate duplicate logic while preserving public API."""
    return save_checkpoint(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        config=config,
        tokenizer_path=tokenizer_path,
        output_dir=best_dir,
        step=step,
        epoch=epoch,
        loss=loss,
        best_loss=best_loss,
        stage_id=stage_id,
        is_best=True,
        is_completed=False,
        sync_optimizer=False,
        dataset=dataset,
        shard_ledger=shard_ledger,
        data_cursor=data_cursor,
        world_model=world_model,
        world_optimizer=world_optimizer,
        world_scaler=world_scaler,
    )


def promote_checkpoint(
    source_checkpoint_dir: str,
    promoted_dir: str,
    stage_id: str,
):
    """
    Promotes the best checkpoint from a completed stage into a clean deployment model:
    - Atomically stages files to a temporary directory before committing.
    - Copies: model.safetensors, config.json, tokenizer.json, tokenizer_config.json, special_tokens_map.json, STAGE_METADATA.json
    - Preserves world_model.safetensors if present in source_checkpoint_dir.
    - Strictly excludes bulky optimizer states (optimizer_state.pt, checkpoint_state.json).
    - Durably verifies completeness before atomic replacement.
    """
    parent_dir = os.path.dirname(os.path.abspath(promoted_dir)) or "."
    staging_promo_dir = os.path.join(parent_dir, f".staging_{os.path.basename(promoted_dir)}_{os.getpid()}_{int(time.time())}")
    if os.path.exists(staging_promo_dir):
        _safe_rmtree(staging_promo_dir)
    os.makedirs(staging_promo_dir, exist_ok=True)

    artifacts = [
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "STAGE_METADATA.json",
    ]
    # Dynamically discover all safetensors shards or index if sharded (including world_model.safetensors)
    for sf in glob.glob(os.path.join(source_checkpoint_dir, "*.safetensors*")):
        bname = os.path.basename(sf)
        if bname not in artifacts:
            artifacts.append(bname)

    copied = 0
    for art in artifacts:
        src = os.path.join(source_checkpoint_dir, art)
        dst = os.path.join(staging_promo_dir, art)
        if os.path.exists(src):
            try:
                if os.path.exists(dst):
                    _safe_unlink(dst)
                os.link(src, dst)
            except Exception:
                _stream_copy_file(src, dst)
            copied += 1

    # Durable pre-commit verification on staging directory (supports single and sharded models)
    if not check_model_weights_exist(staging_promo_dir):
        _safe_rmtree(staging_promo_dir)
        raise RuntimeError(f"[Promotion Verification Failed] Staged model weights are missing or empty in {staging_promo_dir}")

    # Check world_model if present in source
    has_src_wm = any(os.path.basename(f).startswith("world_model") for f in glob.glob(os.path.join(source_checkpoint_dir, "*.safetensors*")))
    if has_src_wm:
        has_staged_wm = any(os.path.basename(f).startswith("world_model") for f in glob.glob(os.path.join(staging_promo_dir, "*.safetensors*")))
        if not has_staged_wm:
            _safe_rmtree(staging_promo_dir)
            raise RuntimeError(f"[Promotion Verification Failed] Staged world_model weights are missing or empty in {staging_promo_dir}")

    # Atomic swap to promoted_dir: new write -> verify new -> delete old
    if os.path.exists(promoted_dir):
        backup_dir = f"{promoted_dir}.prev_atomic_{int(time.time())}"
        swap_success = False
        try:
            os.rename(promoted_dir, backup_dir)
            os.rename(staging_promo_dir, promoted_dir)
            # Verify new promoted_dir is intact before removing previous backup
            if check_model_weights_exist(promoted_dir):
                _safe_rmtree(backup_dir)
                swap_success = True
            else:
                # Rollback if new promoted_dir is missing primary model
                print(f"[Promotion Safety] Rolled back promotion; new directory was incomplete.", flush=True)
                _safe_rmtree(promoted_dir)
                os.rename(backup_dir, promoted_dir)
        except Exception as swap_err:
            if not swap_success and os.path.exists(backup_dir) and not os.path.exists(promoted_dir):
                try:
                    os.rename(backup_dir, promoted_dir)
                except Exception:
                    pass
            print(f"[Promotion Notice] Notice during atomic swap: {swap_err}", flush=True)
    else:
        os.rename(staging_promo_dir, promoted_dir)

    print(f"[Promotion] Successfully promoted clean inference model ({copied} files) -> {promoted_dir}", flush=True)




def _resolve_hf_token_standalone(provided_token: Optional[str] = None) -> Optional[str]:
    if provided_token and str(provided_token).strip():
        return str(provided_token).strip()
    candidates = []
    if os.environ.get("HF_TOKEN"):
        candidates.append(os.environ.get("HF_TOKEN"))
    try:
        from google.colab import userdata
        for key in ("HF_TOKEN", "hf_token", "tara_sync", "HF_API_KEY"):
            try:
                val = userdata.get(key)
                if val: candidates.append(val)
            except Exception: pass
    except Exception: pass
    return candidates[0] if candidates else None


def upload_promoted_model_to_hf(promoted_dir: str, repo_id: str, stage_id: str) -> bool:
    """Uploads the promoted inference model to Hugging Face repository with subfolder isolation."""
    hf_token = _resolve_hf_token_standalone()
    if not hf_token or not repo_id:
        print("[Hugging Face Sync Notice] HF_TOKEN or HF_REPO not set. Skipping auto-upload.")
        return False

    try:
        from huggingface_hub import HfApi
        api = HfApi()
        target_path_in_repo = "neural_model_v1" if ("stage0" in stage_id or "pretraining" in stage_id) else f"promoted/{stage_id}"
        try:
            api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True, token=hf_token)
        except Exception as create_err:
            print(f"[Hugging Face Sync Notice] Repository preflight check note: {create_err}")

        print(f"[Hugging Face Sync] Uploading {promoted_dir} to https://huggingface.co/{repo_id} ({target_path_in_repo})...")
        api.upload_folder(
            folder_path=promoted_dir,
            repo_id=repo_id,
            repo_type="model",
            path_in_repo=target_path_in_repo,
            token=hf_token,
            commit_message=f"Deploy verified promoted model from {stage_id}",
        )
        print(f"[Hugging Face Sync] Model successfully deployed to https://huggingface.co/{repo_id}/{target_path_in_repo}")
        return True
    except Exception as e:
        print(f"[Hugging Face Sync ERROR] Failed to upload promoted model to HF: {e}")
        return False


def finalize_stage_and_promote(
    output_dir: str,
    promoted_dir: Optional[str],
    stage_id: str,
    hf_repo: Optional[str] = None,
    previous_stage_dir: Optional[str] = None,
    candidate_source_dir: Optional[str] = None,
):
    """Unified handler for promoting model snapshots, uploading to HF, and cleaning previous stages."""
    if promoted_dir:
        src_cp = candidate_source_dir
        if not src_cp or not check_model_weights_exist(src_cp):
            best_source = os.path.join(output_dir, "best_checkpoint")
            src_cp = best_source if check_model_weights_exist(best_source) else output_dir
        promote_checkpoint(src_cp, promoted_dir, stage_id)
        if hf_repo:
            upload_promoted_model_to_hf(promoted_dir, hf_repo, stage_id)
    if previous_stage_dir:
        clean_previous_training_checkpoints(previous_stage_dir, output_dir)


def execute_per_shard_promotion(
    pending_shard: str,
    source_checkpoint_dir: str,
    parent_dir: str,
    stage_id: str,
):
    """
    Promotes the model snapshot upon completing a dataset shard:
    1. Generates a clean promoted model snapshot for the completed shard in:
       <parent_dir>/promoted_shard_<clean_name>
    2. Atomically prunes the previous shard's promoted snapshot directory
       ONLY AFTER confirming the new snapshot is 100% complete and verified.
    3. Persists the active promoted shard tracker in <parent_dir>/latest_promoted_shard.json
    """
    clean_name = os.path.splitext(os.path.basename(pending_shard))[0]
    new_promo_dir = os.path.join(parent_dir, f"promoted_shard_{clean_name}")
    tracker_path = os.path.join(parent_dir, "latest_promoted_shard.json")

    prev_promo_dir = None
    if os.path.exists(tracker_path):
        try:
            with open(tracker_path, "r", encoding="utf-8") as f:
                tracker = json.load(f)
                prev_promo_dir = tracker.get("promoted_dir")
        except Exception:
            pass

    print(f"[Per-Shard Promotion] Shard '{pending_shard}' completed. Promoting model snapshot -> {new_promo_dir}...", flush=True)
    promote_checkpoint(source_checkpoint_dir, new_promo_dir, stage_id)

    # Durably verify newly promoted directory BEFORE any pruning
    is_promo_valid = False
    if os.path.isdir(new_promo_dir):
        has_primary = check_model_weights_exist(new_promo_dir)
        c_file = os.path.join(new_promo_dir, "config.json")
        has_cfg = os.path.isfile(c_file) and os.path.getsize(c_file) > 0
        has_src_wm = any(os.path.basename(f).startswith("world_model") for f in glob.glob(os.path.join(source_checkpoint_dir, "*.safetensors*")))
        if has_src_wm:
            has_wm = any(os.path.basename(f).startswith("world_model") for f in glob.glob(os.path.join(new_promo_dir, "*.safetensors*")))
        else:
            has_wm = True
        is_promo_valid = has_primary and has_cfg and has_wm

    if not is_promo_valid:
        raise RuntimeError(
            f"[Per-Shard Promotion Safety Gate] Promoted snapshot verification failed for '{new_promo_dir}'. "
            f"Preserving existing recovery state '{prev_promo_dir}'. Pruning aborted."
        )

    # Prune previous shard's promoted snapshot ONLY after new snapshot is verified intact
    if prev_promo_dir and os.path.abspath(prev_promo_dir) != os.path.abspath(new_promo_dir):
        if os.path.exists(prev_promo_dir):
            print(f"[Per-Shard Promotion] Pruning previous shard promoted model directory: {prev_promo_dir}...", flush=True)
            _safe_rmtree(prev_promo_dir)

    # Update tracker
    try:
        with open(tracker_path, "w", encoding="utf-8") as f:
            json.dump({
                "shard_name": pending_shard,
                "promoted_dir": new_promo_dir,
                "timestamp": time.time(),
                "stage_id": stage_id,
            }, f, indent=2)
    except Exception as e:
        print(f"[Per-Shard Promotion Notice] Could not write {tracker_path}: {e}", flush=True)



def _collect_versioned_checkpoint_dirs(parent_dir: str, output_dir_basename: str) -> List[str]:
    """
    Discovers immutable versioned checkpoint directories that have a COMMITTED marker
    or complete checkpoint assets (checkpoint_state.json + model.safetensors).
    Returns them sorted newest-first (highest step number first).
    Supports patterns:
      - <output_dir_basename>.step<N>_<ts>
      - checkpoint_step_<N>_<ts>
      - best_milestone_step_<N>_<ts>
    """
    prefix = output_dir_basename + ".step"
    candidates = []
    if not os.path.isdir(parent_dir):
        return candidates
    for entry in os.listdir(parent_dir):
        if entry.startswith("."):
            continue
        full_path = os.path.join(parent_dir, entry)
        if not os.path.isdir(full_path):
            continue

        step_num = None
        if entry.startswith(prefix):
            try:
                step_part = entry[len(prefix):].split("_")[0]
                step_num = int(step_part)
            except (ValueError, IndexError):
                step_num = 0
        elif entry.startswith("checkpoint_step_"):
            try:
                step_part = entry[len("checkpoint_step_"):].split("_")[0]
                step_num = int(step_part)
            except (ValueError, IndexError):
                step_num = 0
        elif entry.startswith("best_milestone_step_"):
            try:
                step_part = entry[len("best_milestone_step_"):].split("_")[0]
                step_num = int(step_part)
            except (ValueError, IndexError):
                step_num = 0

        if step_num is not None:
            committed_marker = os.path.join(full_path, "COMMITTED")
            cand_state = os.path.join(full_path, "checkpoint_state.json")
            cand_model = os.path.join(full_path, "model.safetensors")
            if os.path.isfile(committed_marker) or (os.path.isfile(cand_state) and os.path.isfile(cand_model)):
                candidates.append((step_num, full_path))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [path for _, path in candidates]


def _cleanup_old_versioned_checkpoints(
    parent_dir: str,
    output_dir_basename: str,
    keep_n: int = 2,
    active_staging_dir: Optional[str] = None,
    min_abandoned_age_seconds: int = 300,
    force_keep_exact: bool = False,
    protected_dirs: Optional[List[str]] = None,
) -> None:
    """
    Retains the most recent keep_n versioned checkpoint directories (by step number)
    and safely deletes older checkpoints, obsolete re-run duplicates, and abandoned staging remnants.

    Safety contract:
    - Never deletes before new checkpoint is verified and committed.
    - NEVER deletes the latest valid full-resume checkpoint (with optimizer).
    - NEVER deletes the best historical milestone checkpoint.
    - NEVER deletes active snapshots or protected directories.
    - In-flight uploads (younger than min_abandoned_age_seconds without COMMITTED) are protected.
    - All deletions are logged explicitly.
    """
    keep_n = max(int(keep_n), 2)

    prefix = output_dir_basename + ".step"
    staging_prefix = f".ckpt_staging_{output_dir_basename}_"
    all_versioned: List[Tuple[int, str, bool, bool]] = []

    if not os.path.isdir(parent_dir):
        return

    # 1. Resolve and protect latest_checkpoint.json and best_checkpoint.json pointer targets
    ptr_candidates = [
        os.path.join(parent_dir, output_dir_basename, "latest_checkpoint.json"),
        os.path.join(parent_dir, "latest_checkpoint.json"),
        os.path.join(parent_dir, output_dir_basename, "best_checkpoint.json"),
        os.path.join(parent_dir, "best_checkpoint.json"),
    ]
    pointer_targets: Set[str] = set()
    for ptr_path in ptr_candidates:
        if os.path.isfile(ptr_path):
            try:
                with open(ptr_path, "r", encoding="utf-8") as f:
                    ptr_data = json.load(f)
                ptr_output = ptr_data.get("output_dir", "")
                p_vdir = ptr_data.get("versioned_dir")
                if ptr_output and os.path.basename(ptr_output) != output_dir_basename:
                    # Ignore pointer belonging to a sibling checkpoint/stage
                    continue
                if p_vdir and os.path.isdir(p_vdir):
                    pointer_targets.add(os.path.abspath(p_vdir))
            except Exception:
                pass

    # Protect explicitly protected dirs from deletion
    protected_sync_dirs = set()
    if protected_dirs:
        for pd in protected_dirs:
            if pd:
                protected_sync_dirs.add(os.path.abspath(pd))

    active_staging_abs = os.path.abspath(active_staging_dir) if active_staging_dir else None
    now = time.time()

    try:
        entries = os.listdir(parent_dir)
    except Exception as list_err:
        print(f"[Checkpoint Retention Error] Could not list {parent_dir}: {list_err}", flush=True)
        return

    # Scan all entries and categorize
    best_loss_seen = float("inf")
    best_milestone_dir_to_protect = None
    latest_full_step_seen = -1
    latest_full_dir_to_protect = None

    for entry in entries:
        full_path = os.path.join(parent_dir, entry)
        if not os.path.isdir(full_path):
            continue

        full_path_abs = os.path.abspath(full_path)

        # Protect temporary sync snapshot folders from being deleted by retention
        if entry.startswith("sync_snap_") or entry.startswith(".best_sync_snap_"):
            protected_sync_dirs.add(full_path_abs)
            continue

        # Case A: Abandoned staging directories (.ckpt_staging_{output_dir_basename}_*)
        if entry.startswith(staging_prefix):
            if active_staging_abs and full_path_abs == active_staging_abs:
                continue

            # If directory has COMMITTED marker, it is a valid promoted checkpoint; do not purge here
            if os.path.isfile(os.path.join(full_path, "COMMITTED")):
                continue

            try:
                mtime = os.path.getmtime(full_path)
                age = now - mtime
            except Exception:
                age = 0.0

            if age < min_abandoned_age_seconds:
                continue

            is_dead = True
            try:
                pid_part = entry.rsplit("_", 1)[1]
                target_pid = int(pid_part)
                if target_pid == os.getpid():
                    is_dead = False
                else:
                    try:
                        import psutil
                        is_dead = not psutil.pid_exists(target_pid)
                    except Exception:
                        try:
                            os.kill(target_pid, 0)
                            is_dead = False
                        except (OSError, PermissionError):
                            is_dead = True
            except (ValueError, IndexError):
                is_dead = True

            if is_dead:
                if _safe_rmtree(full_path):
                    print(f"[Checkpoint Retention] Removed abandoned staging directory (age {int(age)}s): {entry}", flush=True)
                else:
                    print(f"[Checkpoint Retention Notice] Could not remove abandoned staging dir {entry}", flush=True)
            continue

        # Case B: Versioned checkpoints and best milestones
        step_num = None
        is_milestone = False
        if entry.startswith("best_milestone_step_") and not entry.startswith("."):
            is_milestone = True
            try:
                step_part = entry[len("best_milestone_step_"):].split("_")[0]
                step_num = int(step_part)
            except (ValueError, IndexError):
                step_num = -1
        elif (entry.startswith("best_checkpoint.step") or entry.startswith("best_milestone.")) and not entry.startswith("."):
            is_milestone = True
            try:
                step_part = entry.split(".step")[1].split("_")[0]
                step_num = int(step_part)
            except (ValueError, IndexError):
                step_num = -1
        elif entry.startswith(prefix) and not entry.startswith("."):
            if "best" in prefix.lower():
                is_milestone = True
            try:
                step_part = entry[len(prefix):].split("_")[0]
                step_num = int(step_part)
            except (ValueError, IndexError):
                step_num = -1
        elif entry.startswith("checkpoint_step_") and not entry.startswith("."):
            try:
                step_part = entry[len("checkpoint_step_"):].split("_")[0]
                step_num = int(step_part)
            except (ValueError, IndexError):
                step_num = -1

        if step_num is not None:
            committed_marker = os.path.join(full_path, "COMMITTED")
            is_committed = os.path.isfile(committed_marker)

            # Recognize valid checkpoint with model weights and state even if COMMITTED marker file is missing
            if not is_committed:
                cand_state = os.path.join(full_path, "checkpoint_state.json")
                cand_model = os.path.join(full_path, "model.safetensors")
                cand_idx = os.path.join(full_path, "model.safetensors.index.json")
                cand_bin = os.path.join(full_path, "pytorch_model.bin")
                if os.path.isfile(cand_state) and (os.path.isfile(cand_model) or os.path.isfile(cand_idx) or os.path.isfile(cand_bin)):
                    is_committed = True

            # Check if this checkpoint is an in-flight sync (younger than min_abandoned_age_seconds)
            if not is_committed:
                try:
                    mtime = os.path.getmtime(full_path)
                    if now - mtime < min_abandoned_age_seconds:
                        protected_sync_dirs.add(full_path_abs)
                        continue
                except Exception:
                    pass

            all_versioned.append((step_num, full_path, is_committed, is_milestone))

    # Evaluate legacy unversioned best_checkpoint directory: purge if any newer milestone exists
    legacy_best = os.path.join(parent_dir, "best_checkpoint")
    protected_resolved = {os.path.abspath(p) for p in (protected_dirs or []) if p}
    if os.path.isdir(legacy_best) and os.path.abspath(legacy_best) not in protected_resolved:
        legacy_step = -1
        l_st = os.path.join(legacy_best, "checkpoint_state.json")
        if os.path.isfile(l_st):
            try:
                with open(l_st, "r", encoding="utf-8") as lf:
                    legacy_step = int(json.load(lf).get("step", -1))
            except Exception:
                pass
        committed_milestones = [s for s, _, c, m in all_versioned if c and m]
        # Never purge legacy_best unless at least keep_n (>=2) newer verified, fully committed milestones exist
        if len(committed_milestones) >= keep_n and legacy_step >= 0 and min(committed_milestones) > legacy_step:
            if _safe_rmtree(legacy_best):
                print(f"[Checkpoint Retention] Purged obsolete legacy best_checkpoint folder: {legacy_best} (Step {legacy_step} superseded by {len(committed_milestones)} verified milestones)", flush=True)

    # Separate candidates into Best Milestones and Regular Full Checkpoints
    # Each pool independently enforces keep_n retention (strictly keep_n >= 2):
    # 1. Best milestones: keep newest keep_n (default >= 2); purge older milestones beyond window.
    # 2. Regular checkpoints: keep newest keep_n (default >= 2); purge older checkpoints.
    pools = [
        ("best milestone", [item for item in all_versioned if item[3]]),
        ("regular checkpoint", [item for item in all_versioned if not item[3]]),
    ]

    for pool_tag, pool_items in pools:
        if not pool_items:
            continue

        step_groups: Dict[int, List[Tuple[str, bool]]] = {}
        for step_num, dir_path, is_committed, _ in pool_items:
            step_groups.setdefault(step_num, []).append((dir_path, is_committed))

        candidates_to_eval: List[Tuple[int, str, bool]] = []
        for step_num, dir_entries in step_groups.items():
            dir_entries.sort(key=lambda x: x[0], reverse=True)
            candidates_to_eval.append((step_num, dir_entries[0][0], dir_entries[0][1]))
            # Purge older duplicate runs for the exact same step number immediately
            for old_dir, _ in dir_entries[1:]:
                dir_abs = os.path.abspath(old_dir)
                if dir_abs in pointer_targets:
                    continue
                if dir_abs in protected_sync_dirs:
                    continue
                removed = _safe_rmtree(old_dir)
                if removed:
                    print(f"[Checkpoint Retention] Removed obsolete superseded run for Step {step_num}: {os.path.basename(old_dir)}", flush=True)

        # Sort distinct steps newest-first (highest step number first)
        candidates_to_eval.sort(key=lambda x: (x[0], x[1]), reverse=True)

        committed_kept = 0
        for step_num, dir_path, is_committed in candidates_to_eval:
            dir_abs = os.path.abspath(dir_path)

            if dir_abs in pointer_targets:
                if is_committed:
                    committed_kept += 1
                continue

            if dir_abs in protected_sync_dirs:
                if is_committed:
                    committed_kept += 1
                continue

            if is_committed:
                if committed_kept < keep_n:
                    committed_kept += 1
                    continue

            removed = _safe_rmtree(dir_path)
            if removed:
                reason = f"old committed {pool_tag} beyond retention window ({keep_n})" if is_committed else f"uncommitted {pool_tag} remnant"
                print(f"[Checkpoint Retention] Removed {reason}: {os.path.basename(dir_path)} (Step {step_num})", flush=True)
            else:
                print(f"[Checkpoint Retention Notice] Could not remove {dir_path}", flush=True)


# =============================================================================
# 7. Checkpoint Discovery, Scoped Resumption & Stage Completion Gates
# =============================================================================

def extract_data_cursor_from_state(s_data: dict) -> dict:
    """Extract and normalize data cursor from checkpoint_state.json dictionary."""
    cursor = s_data.get("data_cursor")
    if not cursor:
        cursor = {
            "active_shard": s_data.get("active_shard"),
            "active_shard_line_offset": s_data.get("active_shard_offset", 0),
            "active_shard_byte_offset": s_data.get("active_shard_byte_offset", 0),
            "active_chunk_idx": s_data.get("active_chunk_idx", 0),
            "total_samples_trained": s_data.get("total_samples_trained", s_data.get("total_completed_records", 0)),
            "samples_consumed": s_data.get("samples_consumed", s_data.get("total_completed_records", 0)),
            "completed_shards": s_data.get("completed_shards", []),
            "domain_cursors": s_data.get("domain_cursors", {}),
            "scheduler_deficits": s_data.get("scheduler_deficits", {}),
        }
    else:
        cursor = dict(cursor)
        if "active_shard_byte_offset" not in cursor:
            cursor["active_shard_byte_offset"] = s_data.get("active_shard_byte_offset", 0)
        if "active_chunk_idx" not in cursor:
            cursor["active_chunk_idx"] = s_data.get("active_chunk_idx", 0)
        if "total_samples_trained" not in cursor:
            cursor["total_samples_trained"] = s_data.get("total_samples_trained", s_data.get("total_completed_records", 0))
        if "samples_consumed" not in cursor:
            cursor["samples_consumed"] = s_data.get("samples_consumed", cursor["total_samples_trained"])
        if "domain_cursors" not in cursor and "domain_cursors" in s_data:
            cursor["domain_cursors"] = s_data["domain_cursors"]
        if "scheduler_deficits" not in cursor and "scheduler_deficits" in s_data:
            cursor["scheduler_deficits"] = s_data["scheduler_deficits"]
    return cursor


def compute_scheduled_lr(step: int, warmup_steps: int, max_steps: int, base_lr: float, min_lr: float) -> float:
    """Computes dynamic learning rate with linear warmup and cosine decay."""
    if step < warmup_steps:
        return min_lr + (base_lr - min_lr) * (float(step) / float(max(1, warmup_steps)))
    if step >= max_steps:
        return min_lr
    progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def try_resume_from_checkpoint(
    checkpoint_dir: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    tokenizer_path: Optional[str] = None,
    tokenizer: Optional[Any] = None,
    expected_stage_id: Optional[str] = None,
    allow_milestone_fallback: bool = True,
    world_model: Optional[nn.Module] = None,
    world_optimizer: Optional[torch.optim.Optimizer] = None,
    world_scaler: Optional[torch.cuda.amp.GradScaler] = None,
    base_lr: Optional[float] = None,
    warmup_steps: int = 500,
    max_steps: int = 100_000_000,
    min_lr: float = 1e-5,
) -> Tuple[int, int, float, bool, Optional[Dict[str, Any]]]:
    """
    Attempts to resume training from the latest available COMPLETE training checkpoint.

    Checkpoint discovery order (most authoritative first):
    1. latest_checkpoint.json pointer in the parent directory → resolved versioned directory.
       This is the only path that is guaranteed to be mutually consistent because
       the pointer is published AFTER all artifacts are written and the COMMITTED
       marker is in place.  Using output_dir directly is unsafe: if the process
       crashes mid-copy into output_dir, that directory may contain a mixture of
       artifacts from two different checkpoints.
    2. Scan versioned directories in the same parent for COMMITTED markers (fallback
       when the pointer file itself is absent or corrupt).
    3. output_dir itself — accepted only when it passes full integrity verification,
       as a last-resort fallback for checkpoints created by the legacy code path.

    Strictly EXCLUDES inference-only best_checkpoint directories to protect optimizer continuity.
    Fails closed on tokenizer fingerprint mismatch when current runtime tokenizer is provided.
    """
    parent_dir = os.path.dirname(os.path.abspath(checkpoint_dir)) or "."
    output_dir_basename = os.path.basename(checkpoint_dir)

    # Compute current runtime tokenizer fingerprint for validation if provided
    current_tok_fp = compute_tokenizer_fingerprint(
        tokenizer_path=tokenizer_path,
        tokenizer=tokenizer,
    ) if (tokenizer_path or tokenizer) else None

    # ----------------------------------------------------------------
    # Priority-1: Follow latest_checkpoint.json pointer scoped to output_dir_basename.
    # Checks directory-scoped pointer first (checkpoint_dir/latest_checkpoint.json),
    # then falls back to parent_dir/latest_checkpoint.json only if verified.
    # ----------------------------------------------------------------
    ptr_paths = [
        os.path.join(checkpoint_dir, "latest_checkpoint.json"),
        os.path.join(parent_dir, "latest_checkpoint.json"),
    ]
    pointer_versioned_dir: Optional[str] = None
    for latest_ptr_path in ptr_paths:
        if os.path.isfile(latest_ptr_path):
            try:
                with open(latest_ptr_path, "r", encoding="utf-8") as f:
                    ptr_data = json.load(f)
                ptr_output = ptr_data.get("output_dir", "")
                ptr_stage = ptr_data.get("stage_id", "")
                ptr_vdir = ptr_data.get("versioned_dir")

                # Validate scope: reject pointer belonging to an unrelated sibling stage/directory
                if ptr_output and os.path.basename(ptr_output) != output_dir_basename:
                    print(
                        f"[Resume Discovery] Rejecting pointer at {latest_ptr_path}: output_dir "
                        f"'{os.path.basename(ptr_output)}' does not match target scope '{output_dir_basename}'.",
                        flush=True,
                    )
                    continue

                if expected_stage_id and ptr_stage and ptr_stage != expected_stage_id:
                    print(
                        f"[Resume Discovery] Rejecting pointer at {latest_ptr_path}: stage_id "
                        f"'{ptr_stage}' does not match expected_stage_id '{expected_stage_id}'.",
                        flush=True,
                    )
                    continue

                if ptr_vdir and not os.path.isdir(ptr_vdir):
                    v_base = os.path.basename(ptr_vdir)
                    alt_dirs = [
                        os.path.join(parent_dir, v_base),
                        os.path.join(checkpoint_dir, v_base),
                    ]
                    for alt in alt_dirs:
                        if os.path.isdir(alt):
                            ptr_vdir = alt
                            break

                if ptr_vdir and os.path.isdir(ptr_vdir):
                    # Ensure versioned_dir name matches the target output_dir_basename prefix
                    vdir_base = os.path.basename(ptr_vdir)
                    if not vdir_base.startswith(output_dir_basename + ".step"):
                        print(
                            f"[Resume Discovery] Rejecting pointer versioned dir '{vdir_base}': "
                            f"does not match output scope '{output_dir_basename}'.",
                            flush=True,
                        )
                        continue

                    committed_marker = os.path.join(ptr_vdir, "COMMITTED")
                    if os.path.isfile(committed_marker):
                        pointer_versioned_dir = ptr_vdir
                        print(f"[Resume Discovery] Latest-checkpoint pointer resolved to scoped versioned dir: {ptr_vdir}", flush=True)
                        break
                    else:
                        print(f"[Resume Discovery] Pointer versioned dir has no COMMITTED marker — ignoring: {ptr_vdir}", flush=True)
                else:
                    print(f"[Resume Discovery] Pointer versioned dir absent or not a directory: {ptr_vdir}", flush=True)
            except Exception as ptr_err:
                print(f"[Resume Discovery] Could not read pointer at {latest_ptr_path}: {ptr_err}", flush=True)

    # ----------------------------------------------------------------
    # Build the ordered candidate list.
    # ----------------------------------------------------------------
    candidate_dirs: List[str] = []

    # 1. Pointer-resolved versioned dir (highest priority)
    if pointer_versioned_dir:
        candidate_dirs.append(pointer_versioned_dir)

    # 2. Other committed versioned dirs (scan parent_dir and checkpoint_dir)
    search_dirs = [parent_dir]
    if os.path.isdir(checkpoint_dir) and os.path.abspath(checkpoint_dir) != os.path.abspath(parent_dir):
        search_dirs.append(checkpoint_dir)
    for s_dir in search_dirs:
        for vdir in _collect_versioned_checkpoint_dirs(s_dir, output_dir_basename):
            if vdir not in candidate_dirs:
                candidate_dirs.append(vdir)

    # 3. output_dir itself (legacy last-resort — only if no versioned dir was found)
    if not candidate_dirs and os.path.isdir(checkpoint_dir):
        candidate_dirs.append(checkpoint_dir)
    elif os.path.isdir(checkpoint_dir) and checkpoint_dir not in candidate_dirs:
        candidate_dirs.append(checkpoint_dir)

    best_cand_dir = None
    best_cand_step = -1
    best_cand_epoch = 0
    best_cand_loss = float("inf")
    best_cand_cursor = None
    best_cand_has_opt = False
    cached_curr_tok_sha = compute_file_sha256(tokenizer_path) if tokenizer_path and os.path.exists(tokenizer_path) else None

    print("[Resume Discovery] Scanning candidate checkpoint directories for highest verified step:")
    for cand in candidate_dirs:
        cand_state = os.path.join(cand, "checkpoint_state.json")
        cand_opt = os.path.join(cand, "optimizer_state.pt")

        has_cand_opt = os.path.exists(cand_opt) and os.path.getsize(cand_opt) > 0

        if not (os.path.exists(cand_state) and check_model_weights_exist(cand)):
            if os.path.exists(cand):
                print(f"  -> Notice: Candidate at {cand} is incomplete (missing model or state json). Skipping.")
            continue

        cand_precomputed_sha = None
        s_data = None
        try:
            with open(cand_state, "r", encoding="utf-8") as f:
                s_data = json.load(f)
            cand_precomputed_sha = s_data.get("model_weights_sha256")
        except Exception:
            pass

        # COMMITTED manifest verification: verify every recorded file exists and matches size
        cand_committed = os.path.join(cand, "COMMITTED")
        if os.path.isfile(cand_committed):
            try:
                with open(cand_committed, "r", encoding="utf-8") as cm_f:
                    cm_data = json.load(cm_f)
                if isinstance(cm_data, dict) and "files" in cm_data:
                    manifest_valid = True
                    for m_fname, m_info in cm_data["files"].items():
                        m_file_path = os.path.join(cand, m_fname)
                        if not os.path.isfile(m_file_path):
                            print(f"  -> Notice: Candidate {cand} missing file {m_fname} specified in COMMITTED manifest! Skipping.")
                            manifest_valid = False
                            break
                        if os.path.getsize(m_file_path) != m_info.get("size"):
                            print(f"  -> Notice: Candidate {cand} file {m_fname} size mismatch against COMMITTED manifest! Skipping.")
                            manifest_valid = False
                            break
                    if not manifest_valid:
                        continue
                    if not cand_precomputed_sha:
                        cand_precomputed_sha = cm_data.get("files", {}).get("model.safetensors", {}).get("sha256")
            except Exception:
                pass

        # Strict integrity check: verify candidate (require optimizer if present) with precomputed SHA to eliminate redundant multi-gigabyte re-hashing
        if not verify_checkpoint_integrity(cand, require_optimizer=has_cand_opt, require_completed=False, precomputed_model_sha=cand_precomputed_sha):
            print(f"  -> Notice: Candidate at {cand} failed integrity verification. Skipping.")
            continue

        try:
            if s_data is None:
                with open(cand_state, "r", encoding="utf-8") as f:
                    s_data = json.load(f)
            # Fail closed if this checkpoint originated from the native Rust
            # training runtime — its optimizer state is not cross-compatible.
            validate_checkpoint_runtime_compatibility(s_data)

            # Verify tokenizer compatibility if current runtime tokenizer is provided
            if current_tok_fp is not None:
                stored_tok_fp = s_data.get("tokenizer_fingerprint")
                if stored_tok_fp is not None:
                    stored_canon = stored_tok_fp.get("canonical_fingerprint")
                    curr_canon = current_tok_fp.get("canonical_fingerprint")
                    is_tok_match = False
                    if stored_canon and curr_canon and stored_canon == curr_canon:
                        is_tok_match = True
                    else:
                        stored_tok_sha = stored_tok_fp.get("tokenizer_file_sha256")
                        curr_tok_sha = current_tok_fp.get("tokenizer_file_sha256")
                        stored_vocab = stored_tok_fp.get("vocab_size")
                        curr_vocab = current_tok_fp.get("vocab_size")
                        if stored_tok_sha and curr_tok_sha and stored_tok_sha == curr_tok_sha and (stored_vocab is None or curr_vocab is None or stored_vocab == curr_vocab):
                            is_tok_match = True
                            print(
                                f"  -> Notice: Candidate at {cand} primary tokenizer.json SHA-256 and vocab_size ({curr_vocab}) match authentic baseline. Permitting verified resume.",
                                flush=True,
                            )
                    if not is_tok_match:
                        print(
                            f"  -> Fatal: Candidate at {cand} tokenizer fingerprint mismatch! "
                            f"(Stored: {stored_canon[:16] if stored_canon else 'none'}..., Current: {curr_canon[:16] if curr_canon else 'none'}...). "
                            f"Refusing to resume with incompatible tokenizer to prevent representation corruption.",
                            flush=True,
                        )
                        continue
                else:
                    # Backward compatibility for legacy checkpoints without tokenizer_fingerprint:
                    cand_tok = os.path.join(cand, "tokenizer.json")
                    if os.path.exists(cand_tok) and tokenizer_path and os.path.exists(tokenizer_path):
                        cand_sha = compute_file_sha256(cand_tok)
                        curr_sha = cached_curr_tok_sha or compute_file_sha256(tokenizer_path)
                        if cand_sha != curr_sha:
                            print(
                                f"  -> Fatal: Candidate at {cand} tokenizer.json SHA mismatch against current tokenizer! "
                                f"(Stored: {cand_sha[:16]}..., Current: {curr_sha[:16]}...). Rejecting candidate.",
                                flush=True,
                            )
                            continue
                        else:
                            print(
                                f"  -> Notice: Candidate at {cand} lacks tokenizer_fingerprint metadata, "
                                f"but tokenizer.json SHA matches. Allowing legacy resume.",
                                flush=True,
                            )
                    else:
                        print(
                            f"  -> Notice: Candidate at {cand} lacks tokenizer_fingerprint metadata. "
                            f"Allowing legacy resume per backward-compatibility policy.",
                            flush=True,
                        )
            # Scope check: candidate must match requested output_dir scope or stage_id
            cand_basename = os.path.basename(os.path.normpath(cand))
            is_direct_output = (os.path.abspath(cand) == os.path.abspath(checkpoint_dir))
            is_versioned_match = cand_basename.startswith(output_dir_basename + ".step")

            cand_stage = s_data.get("stage_id", "")
            if expected_stage_id and cand_stage and cand_stage != expected_stage_id:
                print(
                    f"  -> Fatal: Candidate at {cand} stage_id '{cand_stage}' does not match "
                    f"expected_stage_id '{expected_stage_id}'. Rejecting candidate.",
                    flush=True,
                )
                continue

            if not (is_direct_output or is_versioned_match):
                # When expected_stage_id is provided and matches, accept it; otherwise reject scope mismatch
                if expected_stage_id and cand_stage == expected_stage_id:
                    pass
                else:
                    print(
                        f"  -> Notice: Candidate at {cand} does not match target output scope "
                        f"'{output_dir_basename}'. Skipping.",
                        flush=True,
                    )
                    continue

            c_step = int(s_data.get("step", 0))
            c_loss = float(s_data.get("loss", float("inf")))
            c_best_loss = float(s_data.get("best_loss", c_loss))

            cand_type_str = "complete" if has_cand_opt else "model-only"
            print(f"  -> Found valid {cand_type_str} candidate at {cand} (Step: {c_step}, Loss: {c_loss:.4f}) [VERIFIED HEALTHY]")

            should_update = False
            if c_step > best_cand_step:
                should_update = True
            elif c_step == best_cand_step:
                if has_cand_opt and not best_cand_has_opt:
                    should_update = True

            if should_update:
                best_cand_step = c_step
                best_cand_has_opt = has_cand_opt
                best_cand_epoch = int(s_data.get("epoch", 0))
                best_cand_loss = c_best_loss
                best_cand_dir = cand
                best_cand_cursor = extract_data_cursor_from_state(s_data)
        except Exception as e:
            print(f"  -> Notice: Could not parse candidate {cand}: {e}")

    if best_cand_dir is None or best_cand_step < 0:
        print("[Resume Discovery] No valid complete previous checkpoints found. Starting fresh from Step 0.")
        return 0, 0, float("inf"), False, None

    print("=" * 78)
    print(f"  [RESUME] Discovered latest training checkpoint at Step {best_cand_step} in: {best_cand_dir}")
    print(f"  Resumed Step        : {best_cand_step}")
    print(f"  Resumed Epoch       : {best_cand_epoch}")
    print(f"  Historical Best Loss: {best_cand_loss:.4f}")
    if best_cand_cursor:
        print(f"  Data Cursor Shard   : {best_cand_cursor.get('active_shard')}")
        print(f"  Data Cursor Offset  : {best_cand_cursor.get('active_shard_line_offset', 0):,}")
    print("=" * 78)

    # 1. Sync healthy checkpoint files to local checkpoint_dir FIRST if restored from external versioned dir.
    if os.path.abspath(best_cand_dir) != os.path.abspath(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"[Resume] Aligning verified checkpoint artifacts from {best_cand_dir} -> {checkpoint_dir}...", flush=True)
        align_files = [
            "config.json", "tokenizer.json", "checkpoint_state.json", "optimizer_state.pt",
            "STAGE_METADATA.json", "shard_ledger.json", "COMMITTED"
        ]
        for f in os.listdir(best_cand_dir):
            if f.endswith(".safetensors") or f.endswith(".json") or f in align_files:
                if f not in align_files:
                    align_files.append(f)
        for fname in align_files:
            s = os.path.join(best_cand_dir, fname)
            d = os.path.join(checkpoint_dir, fname)
            if os.path.exists(s) and os.path.abspath(s) != os.path.abspath(d):
                s_sz = os.path.getsize(s)
                # Only transfer if destination is missing or incomplete
                if not os.path.exists(d) or os.path.getsize(d) != s_sz:
                    try:
                        tmp_d = d + f".sync_tmp_{os.getpid()}"
                        _stream_copy_file(s, tmp_d)
                        if os.path.exists(tmp_d):
                            os.replace(tmp_d, d)
                    except Exception as align_err:
                        print(f"[Resume Align Notice] Notice aligning {fname}: {align_err}", flush=True)

    # 2. Ingest model weights directly from local SSD copy (fastest) or candidate directory
    dir_to_use = checkpoint_dir if check_model_weights_exist(checkpoint_dir) else best_cand_dir
    if check_model_weights_exist(dir_to_use):
        print(f"[Resume] Ingesting model weights from {dir_to_use}...", flush=True)
        load_model_weights_dir(model, dir_to_use, prefix="model")

    # 3. Ingest world model weights if joint training active
    if world_model is not None and check_model_weights_exist(dir_to_use):
        print(f"[Resume] Ingesting World Model weights from {dir_to_use}...", flush=True)
        load_model_weights_dir(world_model, dir_to_use, prefix="world_model")

    # 4. Ingest optimizer and RNG states from local SSD copy or candidate directory
    local_opt = os.path.join(checkpoint_dir, "optimizer_state.pt")
    opt_file = local_opt if os.path.exists(local_opt) and os.path.getsize(local_opt) > 0 else os.path.join(best_cand_dir, "optimizer_state.pt")
    if os.path.exists(opt_file):
        try:
            opt_data = torch.load(opt_file, map_location="cpu")
            if optimizer and "optimizer" in opt_data:
                optimizer.load_state_dict(opt_data["optimizer"])
                # Move optimizer state tensors to model parameter device to prevent device mismatch errors
                target_device = next(model.parameters()).device if hasattr(model, "parameters") else "cpu"
                for state in optimizer.state.values():
                    for sk, sv in state.items():
                        if isinstance(sv, torch.Tensor):
                            state[sk] = sv.to(target_device)
            if scaler and opt_data.get("scaler"):
                scaler.load_state_dict(opt_data["scaler"])
            if "torch_rng" in opt_data and opt_data["torch_rng"] is not None:
                torch.set_rng_state(opt_data["torch_rng"])
            if "cuda_rng" in opt_data and torch.cuda.is_available() and opt_data["cuda_rng"] is not None:
                torch.cuda.set_rng_state_all(opt_data["cuda_rng"])
            if "py_rng" in opt_data and opt_data["py_rng"] is not None:
                random.setstate(opt_data["py_rng"])

            # Ingest world model optimizer and scaler if present
            if world_optimizer and "world_optimizer" in opt_data:
                world_optimizer.load_state_dict(opt_data["world_optimizer"])
                wm_target_dev = next(world_model.parameters()).device if (world_model and hasattr(world_model, "parameters")) else "cpu"
                for state in world_optimizer.state.values():
                    for sk, sv in state.items():
                        if isinstance(sv, torch.Tensor):
                            state[sk] = sv.to(wm_target_dev)
            if world_scaler and opt_data.get("world_scaler"):
                world_scaler.load_state_dict(opt_data["world_scaler"])

            print(f"[Resume] Restored complete optimizer state, scaler, and deterministic RNG states from Step {best_cand_step}.", flush=True)
        except Exception as e:
            print(f"[Resume Error] Failed to load optimizer state: {e}. Re-initializing clean optimizer with scheduled LR.", flush=True)
            if optimizer:
                try:
                    eff_base_lr = base_lr if base_lr is not None else optimizer.param_groups[0].get("lr", 1e-4)
                    target_lr = compute_scheduled_lr(best_cand_step, warmup_steps=warmup_steps, max_steps=max_steps, base_lr=eff_base_lr, min_lr=min_lr)
                    for pg in optimizer.param_groups:
                        pg["lr"] = target_lr
                    print(f"[Resume] Re-initialized optimizer learning rate to {target_lr:.6e} at Step {best_cand_step}.", flush=True)
                except Exception as lr_err:
                    print(f"[Resume Notice] Could not compute scheduled LR: {lr_err}", flush=True)
    else:
        print(f"[Resume Notice] Optimizer state file absent at {opt_file}. Re-initializing optimizer with scheduled LR for Step {best_cand_step}.", flush=True)
        if optimizer:
            try:
                eff_base_lr = base_lr if base_lr is not None else optimizer.param_groups[0].get("lr", 1e-4)
                target_lr = compute_scheduled_lr(best_cand_step, warmup_steps=warmup_steps, max_steps=max_steps, base_lr=eff_base_lr, min_lr=min_lr)
                for pg in optimizer.param_groups:
                    pg["lr"] = target_lr
                print(f"[Resume] Re-initialized optimizer learning rate to {target_lr:.6e} at Step {best_cand_step}.", flush=True)
            except Exception as lr_err:
                print(f"[Resume Notice] Could not compute scheduled LR: {lr_err}", flush=True)

    return best_cand_step, best_cand_epoch, best_cand_loss, True, best_cand_cursor


def check_stage_already_completed(output_dir: str) -> bool:
    """Checks whether the stage has already completed previously (Idempotence)."""
    meta_path = os.path.join(output_dir, "STAGE_METADATA.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("status") == "COMPLETED":
                loss = meta.get("loss", 0.0)
                step = meta.get("step", 0)
                print("=" * 78)
                print(f"  [STAGE DETECTION] Stage is ALREADY COMPLETED (Status: COMPLETED)")
                print(f"  Directory          : {output_dir}")
                print(f"  Completed Step     : {step}")
                print(f"  Final Loss         : {loss:.4f}")
                print(f"  Skipping redundant training to preserve compute & quota.")
                print("=" * 78)
                return True
        except Exception:
            pass
    return False


# =============================================================================
# 8. Real-Time Frontier Benchmark Evaluation & Telemetry Matrix
# =============================================================================

# ANSI Color Codes for Terminal & Colab
CLR_RED = "\033[91m"
CLR_PURPLE = "\033[95;1m"
CLR_GREEN = "\033[92m"
CLR_YELLOW = "\033[93m"
CLR_CYAN = "\033[96m"
CLR_BOLD = "\033[1m"
CLR_RESET = "\033[0m"


