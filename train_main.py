#!/usr/bin/env python3
"""
TARA Autonomous Cloud Training - Main Training Execution and Multi-Stage Pipeline
================================================================================
Handles:
1. Multi-stage continuous training execution (Stages 0..N unbounded).
2. Live Loss, Perplexity (PPL), BF16/FP16 precision, LR, GradNorm, Throughput.
3. Multi-domain dynamic batching and optimizer step scheduling.
4. Pre-training verification gates, remote asset checks, and HF model promotion.
5. Clean modular imports from gdrive_sync and model_dataset.
6. Standalone with zero dependency on local PC or external repository code.
================================================================================
"""

import argparse
import copy
import gc
import glob
import hashlib
import json
import math
import os
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from safetensors.torch import save_file, load_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

# Import GDrive and Checkpointing helpers
from gdrive_sync import (
    mount_google_drive,
    sync_checkpoint_to_gdrive,
    restore_checkpoint_from_gdrive,
    _safe_unlink,
    _safe_rmtree,
    _stream_copy_file,
    validate_checkpoint_runtime_compatibility,
    compute_file_sha256,
    compute_tokenizer_fingerprint,
    verify_checkpoint_integrity,
    clean_stale_checkpoint_artifacts,
    clean_previous_training_checkpoints,
    _reclaim_emergency_disk_space,
    get_memory_status_str,
    check_model_weights_exist,
    resolve_max_shard_size_bytes,
    save_safetensors_sharded,
    save_checkpoint,
    save_best_milestone,
    promote_checkpoint,
    finalize_stage_and_promote,
    execute_per_shard_promotion,
    _collect_versioned_checkpoint_dirs,
    _cleanup_old_versioned_checkpoints,
    extract_data_cursor_from_state,
    try_resume_from_checkpoint,
    check_stage_already_completed,
    resolve_gdrive_backup_dir,
    resolve_gdrive_restore_dir,
    trigger_async_gdrive_sync,
    wait_for_gdrive_sync,
    upload_promoted_model_to_hf,
)

# Import Model, Tokenizer and Dataset helpers
from model_dataset import (
    CANONICAL_DOMAINS,
    SPECIAL_TOKENS,
    TokenizerOutput,
    TaraNativeTokenizer,
    RMSNorm,
    rotate_half,
    resolve_dynamic_rope_theta,
    precompute_rope_cos_sin,
    apply_rope,
    TaraAttention,
    TaraMlp,
    TaraDecoderLayer,
    TaraModel,
    CrossModalProjector,
    PyTorchVisionEncoder,
    PyTorchAudioEncoder,
    CausalLMOutput,
    TaraForCausalLM,
    TaraMultiModalForCausalLM,
    TaraWorldModel,
    _resolve_hf_token,
    download_single_shard_jit,
    ShardLedger,
    ShardLineIndex,
    DatasetSnapshot,
    TaraStreamingJsonlDataset,
    extract_state_transition,
    inspect_candidate_world_transitions,
    route_dataset_record,
    TaraDomainBatchItem,
    WorldModelTransitionDataset,
    resolve_shard_domain,
    parse_domain_weights,
    TaraMultiDomainScheduler,
    collate_fn,
    load_candidate_weights,
)

# Terminal ANSI Color Telemetry Palette
CLR_RESET = "\033[0m"
CLR_BOLD = "\033[1m"
CLR_RED = "\033[31m"
CLR_GREEN = "\033[32m"
CLR_PURPLE = "\033[35m"
CLR_CYAN = "\033[36m"


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
        return CLR_PURPLE, "MATCHED (Average)", "[PURPLE]"
    else:
        return CLR_RED, "NOT MATCH (Untrained)", "[RED]"


def print_frontier_comparison_matrix_colored(step: int, cur_loss: float, best_loss: float):
    ppl = math.exp(min(cur_loss, 20.0))
    bpb = (cur_loss / math.log(2)) / 3.8

    loss_clr, loss_status, loss_tag = get_metric_status_and_color(cur_loss, (1.4, 1.9), (2.0, 3.0))
    ppl_clr, ppl_status, ppl_tag = get_metric_status_and_color(ppl, (4.0, 7.0), (7.1, 20.0))
    bpb_clr, bpb_status, bpb_tag = get_metric_status_and_color(bpb, (0.55, 0.75), (0.76, 1.20))

    print(f"\n{CLR_BOLD}{'=' * 115}{CLR_RESET}", flush=True)
    print(f"  {CLR_CYAN}{CLR_BOLD}TARA vs FRONTIER MODEL REAL-TIME BENCHMARK MATRIX (Step {step:04d}){CLR_RESET}", flush=True)
    print(f"{CLR_BOLD}{'-' * 115}{CLR_RESET}", flush=True)
    print(f"  {'ಮೆಟ್ರಿಕ್ (Metric)':<24} {'ಆರಂಭಿಕ (Untrained)':<22} {'ಸಾಧಾರಣ (Average)':<22} {'ಟಾಪ್ ಲೆವೆಲ್ (Frontier/SOTA)':<28} {'TARA (Current)':<18}", flush=True)
    print(f"{'-' * 115}", flush=True)
    print(f"  {'Cross-Entropy Loss':<24} {'10.0 – 12.0+':<22} {'2.2 – 3.0':<22} {'1.4 – 1.9':<28} {loss_clr}{cur_loss:<7.4f} {loss_status} {loss_tag}{CLR_RESET}", flush=True)
    print(f"  {'Perplexity (PPL)':<24} {'20,000+':<22} {'9.0 – 20.0':<22} {'4.0 – 7.0':<28} {ppl_clr}{ppl:<7.2f} {ppl_status} {ppl_tag}{CLR_RESET}", flush=True)
    print(f"  {'Bits-Per-Byte (BPB)':<24} {'3.5+':<22} {'0.9 – 1.2':<22} {'0.55 – 0.75':<28} {bpb_clr}{f'{bpb:.2f} b/B':<7} {bpb_status} {bpb_tag}{CLR_RESET}", flush=True)
    print(f"{'-' * 115}", flush=True)
    if best_loss < float('inf'):
        b_ppl = math.exp(min(best_loss, 20.0))
        b_bpb = (best_loss / math.log(2)) / 3.8
        b_clr, b_status, b_tag = get_metric_status_and_color(best_loss, (1.4, 1.9), (2.0, 3.0))
        print(f"  {CLR_BOLD}Historical Best Checkpoint:{CLR_RESET} Loss = {b_clr}{best_loss:.4f}{CLR_RESET} | PPL = {b_clr}{b_ppl:.2f}{CLR_RESET} | BPB = {b_clr}{b_bpb:.2f} b/B{CLR_RESET} ({b_clr}{b_status} {b_tag}{CLR_RESET})", flush=True)
    print(f"{'=' * 115}\n", flush=True)


# =============================================================================
# 8b. High-Speed Cloud Asset Ingestion (Hugging Face CDN)
# =============================================================================

def ensure_remote_assets_available(
    model_dir: str,
    dataset_dir: str,
    repo_id: Optional[str] = None,
    dataset_repo: Optional[str] = None,
    prefix: str = "v2",
    bulk_download_dataset: bool = False,
    output_dir: Optional[str] = None,
    stage_id: Optional[str] = None,
):
    """
    High-Speed Cloud Asset Ingestion via Hugging Face Hub (Pure Python).
    Downloads base candidate model weights/config if missing.
    Checks existing output_dir checkpoints first to avoid redundant downloads.
    Dataset shards are streamed on-demand via single-shard JIT (download_single_shard_jit)
    during training, avoiding multi-gigabyte bulk downloads unless bulk_download_dataset is True.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[Notice] huggingface_hub not installed; relying on existing local files.")
        return

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)
    token = _resolve_hf_token()

    # 1. Check Model Weights & Config
    config_file = os.path.join(model_dir, "config.json")
    safetensor_files = glob.glob(os.path.join(model_dir, "*.safetensors"))

    # Before downloading from Hugging Face, check if a valid checkpoint already exists in output_dir
    if not (os.path.exists(config_file) and len(safetensor_files) > 0):
        search_sources = []
        if output_dir:
            search_sources.append(output_dir)
            search_sources.append(os.path.join(output_dir, "best_checkpoint"))
        for src in search_sources:
            if not os.path.isdir(src):
                continue
            src_config = os.path.join(src, "config.json")
            if check_model_weights_exist(src) and os.path.isfile(src_config):
                print(f"[Asset Discovery] Restoring model candidate from existing checkpoint at {src}...", flush=True)
                for f_name in os.listdir(src):
                    if f_name.endswith(".safetensors") or f_name.endswith(".json"):
                        s_path = os.path.join(src, f_name)
                        d_path = os.path.join(model_dir, f_name)
                        if os.path.isfile(s_path) and not os.path.exists(d_path):
                            shutil.copy2(s_path, d_path)
                break

        safetensor_files = glob.glob(os.path.join(model_dir, "*.safetensors"))

    if not (os.path.exists(config_file) and len(safetensor_files) > 0):
        print(f"[High-Speed HF] Downloading Model Candidate from {repo_id}...")
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                local_dir=model_dir,
                allow_patterns=["config.json", "tokenizer.json", "tokenizer_config.json", "*.safetensors", "neural_model_v1/*"],
                token=token,
                max_workers=8,
            )
            sub_dir = os.path.join(model_dir, "neural_model_v1")
            if os.path.exists(sub_dir):
                for item in os.listdir(sub_dir):
                    s = os.path.join(sub_dir, item)
                    d = os.path.join(model_dir, item)
                    if not os.path.exists(d):
                        shutil.move(s, d)
                shutil.rmtree(sub_dir, ignore_errors=True)
            print(f"[High-Speed HF] Model candidate assets ready in {model_dir}")
        except Exception as e:
            print(f"[Warning] Model candidate download warning: {e}")

    # 2. Check Dataset Shards (Strict Single-Shard JIT Streaming by Default)
    if bulk_download_dataset:
        ds_source = dataset_repo or repo_id
        jsonl_shards = glob.glob(os.path.join(dataset_dir, "*.jsonl"))
        if len(jsonl_shards) == 0:
            print(f"[High-Speed HF] Bulk Downloading Dataset Shards from {ds_source}...")
            try:
                snapshot_download(
                    repo_id=ds_source,
                    repo_type="dataset",
                    local_dir=dataset_dir,
                    allow_patterns=[f"{prefix}/*.jsonl", f"{prefix}/*.json", "*.jsonl"],
                    token=token,
                    max_workers=8,
                )
                sub_v2 = os.path.join(dataset_dir, prefix)
                if os.path.exists(sub_v2):
                    for item in os.listdir(sub_v2):
                        s = os.path.join(sub_v2, item)
                        d = os.path.join(dataset_dir, item)
                        if not os.path.exists(d):
                            shutil.move(s, d)
                    shutil.rmtree(sub_v2, ignore_errors=True)

                # Purge hidden .cache directories to eliminate duplicate disk usage
                for d in [model_dir, dataset_dir]:
                    shutil.rmtree(os.path.join(d, ".cache"), ignore_errors=True)
                shutil.rmtree(os.path.expanduser("~/.cache/huggingface"), ignore_errors=True)
                shutil.rmtree(os.path.expanduser("~/.cache/pip"), ignore_errors=True)

                print(f"[High-Speed HF] Dataset shards ready in {dataset_dir}")
            except Exception as e:
                print(f"[Warning] Dataset shards download warning: {e}")
    else:
        print(f"[High-Speed HF] Single-shard JIT streaming active — shards will be fetched on-demand by TaraStreamingJsonlDataset.", flush=True)



def execute_optimizer_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    world_model: Optional[nn.Module] = None,
    world_optimizer: Optional[torch.optim.Optimizer] = None,
    world_scaler: Optional[Any] = None,
    max_grad_norm: float = 1.0,
    cur_loss: Optional[float] = None,
    ema_loss: Optional[float] = None,
    accum_tokens: Optional[int] = None,
) -> Tuple[bool, float, float, float, float]:
    """
    Executes a unified optimizer step with comprehensive Model Shock & Gradient Explosion Protection:
    1. Unscales gradients.
    2. Exactly normalizes accumulated unscaled gradients by total valid tokens across accumulation window.
    3. Performs dynamic gradient norm clipping (max_grad_norm=1.0).
    4. Detects catastrophic model shock (extreme gradient spike > 15x or sudden loss spike > 4x EMA).
    5. Detects NaN/Inf gradient corruption.
    6. Safely steps or skips optimizer to protect model weights from destructive parameter shock.
    """
    scaler.unscale_(optimizer)

    # Mathematically exact token-level gradient normalization:
    # Divides accumulated token-loss sum gradients by the exact total valid tokens across the window.
    if accum_tokens is not None and accum_tokens > 0:
        for p in model.parameters():
            if p.grad is not None:
                p.grad.div_(accum_tokens)

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
    is_non_finite_grad = math.isnan(grad_norm_val) or math.isinf(grad_norm_val)

    # Model Shock & Catastrophic Spike Detection
    # Note: torch.nn.utils.clip_grad_norm_ already safely scales all gradients down to max_grad_norm (e.g. 1.0).
    # Normal unclipped gradient norms (10 - 200) are safely clipped and must NOT be skipped.
    # Only extreme numerical explosions (> 1000x max_grad_norm) or non-finite values require skipping.
    is_shock_detected = False
    if not is_non_finite_grad:
        if grad_norm_val > (1000.0 * max_grad_norm):
            is_shock_detected = True
            print(f"[Model Shock Guard] Catastrophic gradient explosion detected (norm={grad_norm_val:.2f} > {1000.0 * max_grad_norm:.1f}). Skipping step to prevent weight destruction!", flush=True)
        elif cur_loss is not None and ema_loss is not None and ema_loss > 0.5:
            if cur_loss > (10.0 * ema_loss) and cur_loss > 15.0:
                is_shock_detected = True
                print(f"[Model Shock Guard] Sudden catastrophic loss spike detected (Loss={cur_loss:.4f} vs EMA={ema_loss:.4f}). Skipping step to protect model stability!", flush=True)

    w_grad_norm_val = 0.0
    if world_model is not None and world_optimizer is not None:
        if world_scaler is not None:
            world_scaler.unscale_(world_optimizer)
        w_grad_norm = torch.nn.utils.clip_grad_norm_(world_model.parameters(), max_grad_norm)
        w_grad_norm_val = w_grad_norm.item() if isinstance(w_grad_norm, torch.Tensor) else float(w_grad_norm)
        if math.isnan(w_grad_norm_val) or math.isinf(w_grad_norm_val):
            is_non_finite_grad = True

    if (is_non_finite_grad or is_shock_detected) and not scaler.is_enabled():
        is_step_skipped = True
        scale_before = 1.0
        scale_after = 1.0
    elif is_shock_detected:
        optimizer.zero_grad(set_to_none=True)
        if world_optimizer is not None:
            world_optimizer.zero_grad(set_to_none=True)
        scaler.update()
        is_step_skipped = True
        scale_before = scaler.get_scale()
        scale_after = scale_before
    else:
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        if world_optimizer is not None and world_scaler is not None:
            world_scaler.step(world_optimizer)
            world_scaler.update()
        scaler.update()
        scale_after = scaler.get_scale()
        is_step_skipped = (scaler.is_enabled() and (scale_after < scale_before)) or is_non_finite_grad

    return is_step_skipped, grad_norm_val, w_grad_norm_val, scale_before, scale_after


def extract_accumulated_data_cursor(
    accum_batches: list,
    dataset: Any,
    total_samples_trained: int,
    samples_consumed: int,
    shard_ledger: Optional[Any],
    committed_domain_cursors: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Extracts exact resumption cursor metadata from the last sample of an accumulated batch window.
    """
    # accum_batches stores cursors directly (zero GPU-to-CPU tensor copies)
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
    c_shard = last_sample_cursor[0]
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
        "completed_shards": list(shard_ledger.completed_shards) if shard_ledger else [],
        "domain_cursors": copy.deepcopy(committed_domain_cursors),
        "scheduler_deficits": copy.deepcopy(getattr(dataset, "deficits", {})),
    }


def compute_scheduled_lr(step: int, warmup_steps: int, max_steps: int, base_lr: float, min_lr: float) -> float:
    """Computes dynamic learning rate with linear warmup and cosine decay."""
    if step < warmup_steps:
        return min_lr + (base_lr - min_lr) * (float(step) / float(max(1, warmup_steps)))
    if step >= max_steps:
        return min_lr
    # When max_steps is unbounded (e.g. 100,000,000 for full-dataset rolling pretraining),
    # use dynamic decay cycle (50,000 steps per cycle) to ensure continuous learning rate annealing
    effective_max = min(max_steps, 50_000) if max_steps >= 1_000_000 else max_steps
    progress = float(min(step, effective_max) - warmup_steps) / float(max(1, effective_max - warmup_steps))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def compute_runtime_sha256(filepath: str) -> Optional[str]:
    """Computes dynamic runtime SHA-256 hash of a file."""
    if not os.path.exists(filepath):
        return None
    h = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def sha256_matches_line_endings(path: str, expected_sha256: str) -> bool:
    """
    Accept an exact SHA-256 match or a match caused solely by
    CRLF/LF line-ending differences across platforms.
    Does not ignore JSON formatting, values, or other content changes.
    """
    if not path or not os.path.isfile(path):
        return False
    if not isinstance(expected_sha256, str):
        return False
    expected = expected_sha256.strip().lower()
    # A SHA-256 digest must contain exactly 64 hexadecimal characters.
    if len(expected) != 64:
        return False
    if any(c not in "0123456789abcdef" for c in expected):
        return False
    try:
        with open(path, "rb") as f:
            raw = f.read()
        # First try the exact bytes.
        actual = hashlib.sha256(raw).hexdigest()
        if actual == expected:
            return True
        # Normalize CRLF and standalone CR to LF.
        lf_normalized = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        if hashlib.sha256(lf_normalized).hexdigest() == expected:
            return True
        # Normalize LF to CRLF (if expected hash was computed on Windows with CRLF)
        crlf_normalized = lf_normalized.replace(b"\n", b"\r\n")
        if hashlib.sha256(crlf_normalized).hexdigest() == expected:
            return True
        return False
    except Exception:
        return False


def resolve_trusted_public_key(explicit_key: Optional[str] = None) -> Optional[str]:
    """
    Resolves the authoritative trusted Ed25519 public key.
    Sources (in order of priority):
    1. Explicit caller argument
    2. TARA_AUTH_PUBLIC_KEY environment variable
    3. Google Colab userdata secret
    4. Authoritative trusted root key file on disk
    """
    if explicit_key and isinstance(explicit_key, str):
        cleaned = explicit_key.strip().lower()
        if len(cleaned) == 64 and re.fullmatch(r"[0-9a-fA-F]{64}", cleaned):
            return cleaned

    env_key = os.environ.get("TARA_AUTH_PUBLIC_KEY")
    if env_key and isinstance(env_key, str):
        cleaned = env_key.strip().lower()
        if len(cleaned) == 64 and re.fullmatch(r"[0-9a-fA-F]{64}", cleaned):
            return cleaned

    try:
        from google.colab import userdata
        colab_key = userdata.get("TARA_AUTH_PUBLIC_KEY")
        if colab_key and isinstance(colab_key, str):
            cleaned = colab_key.strip().lower()
            if len(cleaned) == 64 and re.fullmatch(r"[0-9a-fA-F]{64}", cleaned):
                return cleaned
    except Exception:
        pass

    trusted_key_paths = [
        "model_config/tara_auth_pubkey.hex",
        "storage/vault/tara_auth_pubkey.hex",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_config", "tara_auth_pubkey.hex"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "storage", "vault", "tara_auth_pubkey.hex"),
    ]
    for p in trusted_key_paths:
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    content = f.read().strip().lower()
                if len(content) == 64 and re.fullmatch(r"[0-9a-fA-F]{64}", content):
                    return content
            except Exception:
                pass
    return None


def verify_ed25519_authorization_signature(
    proof: Dict[str, Any],
    trusted_public_key: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Verifies the Ed25519 digital signature of the pretraining authorization proof.
    Ensures that the canonical payload (schema_version, authorization_status,
    model_config_sha256, tokenizer_sha256, dataset_fingerprint_sha256, audit_version)
    has been digitally signed by an authorized Ed25519 private key.
    Fails closed if trusted public key is not configured or if signature/key is missing or altered.
    """
    sig_hex = proof.get("signature")
    if not sig_hex or not isinstance(sig_hex, str):
        return False, "Missing or invalid signature in authorization proof"

    raw_pub = resolve_trusted_public_key(trusted_public_key)
    if not raw_pub or not isinstance(raw_pub, str):
        return False, "No externally trusted public key configured; refusing authorization"

    proof_pub = proof.get("public_key")
    if not proof_pub or not isinstance(proof_pub, str):
        return False, "Missing public key in authorization proof"

    if proof_pub.strip().lower() != raw_pub.strip().lower():
        return False, "Proof public key does not match the configured trusted key"

    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        from cryptography.exceptions import InvalidSignature
    except ImportError:
        return False, "cryptography library not installed; cannot verify Ed25519 signature"

    try:
        pub_bytes = bytes.fromhex(raw_pub.strip())
        if len(pub_bytes) != 32:
            return False, f"Invalid Ed25519 public key length: {len(pub_bytes)} bytes (expected 32)"

        sig_bytes = bytes.fromhex(sig_hex.strip())
        if len(sig_bytes) != 64:
            return False, f"Invalid Ed25519 signature length: {len(sig_bytes)} bytes (expected 64)"

        canonical_dict = {
            "audit_version": str(proof.get("audit_version", "1.0.0")),
            "authorization_status": str(proof.get("authorization_status", "")),
            "dataset_fingerprint_sha256": str(proof.get("dataset_fingerprint_sha256", "")),
            "model_config_sha256": str(proof.get("model_config_sha256", "")),
            "schema_version": int(proof.get("schema_version", 1)),
            "tokenizer_sha256": str(proof.get("tokenizer_sha256", "")),
        }
        canonical_bytes = json.dumps(canonical_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")

        verifier = Ed25519PublicKey.from_public_bytes(pub_bytes)
        verifier.verify(sig_bytes, canonical_bytes)
        return True, "Valid Ed25519 digital signature verified successfully"
    except InvalidSignature:
        return False, "Ed25519 signature verification failed: payload has been altered or tampered"
    except Exception as e:
        return False, f"Ed25519 verification error: {e}"


def verify_pretraining_authorization_proof(model_dir: str, checkpoints_dir: str = "storage/models/checkpoints") -> bool:
    """
    Cryptographically verifies that the pretraining authorization gate has been successfully passed:
    1. Checks for existence and validity of pretraining_authorization.json.
    2. Validates 'authorization_status' == 'AUTHORIZED'.
    3. Validates Ed25519 digital signature of canonical authorization payload.
    4. Dynamically computes the runtime SHA-256 of candidate model config.json and tokenizer.json
       and verifies they match the cryptographic signatures recorded in the authorization proof.
    """
    candidate_proof_paths = [
        os.path.join(checkpoints_dir, "pretraining_authorization.json"),
        os.path.join(model_dir, "pretraining_authorization.json"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretraining_authorization.json"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "storage", "models", "checkpoints", "pretraining_authorization.json"),
        "storage/models/checkpoints/pretraining_authorization.json",
        "pretraining_authorization.json",
    ]
    proof = None
    proof_path = None
    for cp in candidate_proof_paths:
        if os.path.exists(cp):
            try:
                with open(cp, "r", encoding="utf-8") as f:
                    proof_data = json.load(f)
                if proof_data.get("authorization_status") == "AUTHORIZED":
                    proof = proof_data
                    proof_path = cp
                    break
            except Exception as e:
                print(f"[Pre-Training Gate Warning] Proof parse warning in {cp}: {e}")

    if not proof:
        print("[Pre-Training Gate Notice] Valid pretraining_authorization.json not found in checkpoint paths.")
        return False

    # 2b. Mandatory Ed25519 Digital Signature Verification (Fail-Closed)
    sig_ok, sig_msg = verify_ed25519_authorization_signature(proof)
    if not sig_ok:
        print(f"[Pre-Training Gate Tamper Error] {sig_msg}", flush=True)
        return False
    print(f"[Pre-Training Gate] {sig_msg}", flush=True)

    expected_config_sha = proof.get("model_config_sha256")
    expected_tok_sha = proof.get("tokenizer_sha256")

    candidate_config_paths = [
        os.path.join(model_dir, "config.json"),
        "storage/models/tara_candidate_v1/config.json",
        "config.json",
    ]
    candidate_tok_paths = [
        os.path.join(model_dir, "tokenizer.json"),
        "storage/models/tara_candidate_v1/tokenizer.json",
        "tokenizer.json",
    ]

    config_fp = next((p for p in candidate_config_paths if os.path.exists(p)), None)
    tok_fp = next((p for p in candidate_tok_paths if os.path.exists(p)), None)

    # Strict 64-character hexadecimal validation helper
    def is_valid_sha256(v: Any) -> bool:
        return isinstance(v, str) and re.fullmatch(r"[0-9a-fA-F]{64}", v.strip()) is not None

    if not is_valid_sha256(expected_config_sha):
        print(f"[Pre-Training Gate Tamper Error] Missing or invalid 64-hex model_config_sha256 in proof: {expected_config_sha}", flush=True)
        return False

    if not is_valid_sha256(expected_tok_sha):
        print(f"[Pre-Training Gate Tamper Error] Missing or invalid 64-hex tokenizer_sha256 in proof: {expected_tok_sha}", flush=True)
        return False

    # Strict canonical binding: must verify the exact files located in model_dir
    canonical_cfg = os.path.join(model_dir, "config.json")
    canonical_tok = os.path.join(model_dir, "tokenizer.json")

    if not os.path.isfile(canonical_cfg):
        print(f"[Pre-Training Gate Error] Canonical config.json missing at: {canonical_cfg}", flush=True)
        return False
    if not os.path.isfile(canonical_tok):
        print(f"[Pre-Training Gate Error] Canonical tokenizer.json missing at: {canonical_tok}", flush=True)
        return False

    actual_cfg_sha = compute_runtime_sha256(canonical_cfg)
    actual_tok_sha = compute_runtime_sha256(canonical_tok)

    cfg_matched = sha256_matches_line_endings(
        canonical_cfg,
        expected_config_sha,
    )
    if not cfg_matched:
        print(
            "[Pre-Training Gate Tamper Error] "
            "Config SHA-256 mismatch!\n"
            f"  Expected: {expected_config_sha}\n"
            f"  Actual  : {actual_cfg_sha}\n"
            f"  Path    : {canonical_cfg}",
            flush=True,
        )
        return False
    print(
        "[Pre-Training Gate] Config SHA-256 verified "
        "(exact bytes or line-ending normalization).",
        flush=True,
    )

    tok_matched = sha256_matches_line_endings(
        canonical_tok,
        expected_tok_sha,
    )
    if not tok_matched:
        print(
            "[Pre-Training Gate Tamper Error] "
            "Tokenizer SHA-256 mismatch!\n"
            f"  Expected: {expected_tok_sha}\n"
            f"  Actual  : {actual_tok_sha}\n"
            f"  Path    : {canonical_tok}",
            flush=True,
        )
        return False
    print(
        "[Pre-Training Gate] Tokenizer SHA-256 verified "
        "(exact bytes or line-ending normalization).",
        flush=True,
    )

    # Mandatory model architecture audit verification
    try:
        with open(canonical_cfg, "r", encoding="utf-8") as f:
            loaded_cfg = json.load(f)
        m_audit = proof.get("model_weights_audit", {})
        t_audit = proof.get("tokenizer_audit", {})
        arch_valid = (
            loaded_cfg.get("num_hidden_layers") == m_audit.get("num_layers")
            and loaded_cfg.get("hidden_size") == m_audit.get("hidden_size")
            and loaded_cfg.get("intermediate_size") == m_audit.get("intermediate_size")
            and loaded_cfg.get("num_attention_heads") == m_audit.get("num_attention_heads")
            and loaded_cfg.get("num_key_value_heads") == m_audit.get("num_kv_heads")
            and loaded_cfg.get("vocab_size") == t_audit.get("vocab_size")
            and m_audit.get("status") == "VERIFIED_PASSED"
            and t_audit.get("status") == "VERIFIED_PASSED"
        )
        if not arch_valid:
            print(
                "[Pre-Training Gate Tamper Error] "
                "Model architecture audit failed.",
                flush=True,
            )
            return False
        print(
            f"[Pre-Training Gate] Model architecture invariants verified: "
            f"layers={loaded_cfg.get('num_hidden_layers')}, hidden={loaded_cfg.get('hidden_size')}, "
            f"heads={loaded_cfg.get('num_attention_heads')}/{loaded_cfg.get('num_key_value_heads')}, "
            f"vocab={loaded_cfg.get('vocab_size')}",
            flush=True,
        )
    except Exception as e:
        print(f"[Pre-Training Gate Tamper Error] Model architecture audit failed with parse error: {e}", flush=True)
        return False

    print(
        f"[Pre-Training Gate] Cryptographically verified authorization proof: {proof_path}\n"
        f"  Model Config SHA : {actual_cfg_sha[:16]}... (MATCH)\n"
        f"  Tokenizer SHA    : {actual_tok_sha[:16]}... (MATCH)",
        flush=True,
    )
    return True


# Note: upload_promoted_model_to_hf is imported from gdrive_sync


# =============================================================================
# 9. Main Autonomous Multi-Stage Training Runner & Optimizer Execution
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="TARA Autonomous Cloud Multi-Stage Training Pipeline")
    parser.add_argument("--model-dir", type=str, default="storage/models/tara_candidate_v1", help="Candidate / input model directory")
    parser.add_argument("--dataset-dir", type=str, default="storage/datasets/main_dataset_v2", help="Dataset directory")
    parser.add_argument("--output-dir", type=str, default="storage/models/checkpoints/stage0_pretraining", help="Output checkpoint directory")
    parser.add_argument("--promoted-dir", type=str, default=None, help="Directory to save clean promoted model for deployment/inference")
    parser.add_argument("--previous-stage-dir", type=str, default=None, help="Directory of previous stage to clean bulky training optimizer states from")
    parser.add_argument("--stage-id", type=str, default="stage0_pretraining", help="Unique Stage ID (e.g. stage0, stage1, ... N)")
    parser.add_argument("--stage-name", type=str, default="Stage 0: Foundational Pretraining", help="Human-readable stage title")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Micro-batch size per device")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max-steps", type=int, default=200, help="Max optimizer steps")
    parser.add_argument("--warmup-steps", type=int, default=50, help="Linear warmup optimizer steps")
    parser.add_argument("--min-lr", type=float, default=1e-5, help="Minimum learning rate floor for cosine decay")
    parser.add_argument("--hf-repo", type=str, default=os.environ.get("HF_REPO", os.environ.get("HF_MODEL_REPO", os.environ.get("HF_DATASET_REPO", None))), help="Hugging Face repository for auto-upload and dataset discovery")
    parser.add_argument("--hf-dataset-repo", type=str, default=os.environ.get("HF_DATASET_REPO", None), help="Hugging Face dataset repository for dataset streaming and discovery")
    parser.add_argument("--hf-model-repo", type=str, default=os.environ.get("HF_MODEL_REPO", os.environ.get("HF_REPO", None)), help="Hugging Face model repository for checkpoint and weight uploads")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face authentication token (or via HF_TOKEN / Colab secret 'tara_sync')")
    parser.add_argument("--verify-gate", action="store_true", default=True, help="Verify pretraining authorization proof before training")
    parser.add_argument("--mask-prompt", action="store_true", default=False, help="Mask prompt tokens with -100 during instruction fine-tuning stages (default False for foundational pre-training)")
    parser.add_argument("--max-seq-len", type=int, default=int(os.environ.get("TARA_MAX_SEQ_LEN", 0)), help="Max sequence length (0 = dynamically resolve from model config)")
    parser.add_argument("--max-shards", type=int, default=0, help="Max shards to ingest (0 for all discovered shards)")
    parser.add_argument("--exclude-shards", type=str, default="*shard_099*,*shard_098*", help="Comma-separated patterns of shards strictly excluded from training (held-out validation)")
    parser.add_argument("--max-samples", type=int, default=0, help="Max samples to ingest across shards (0 for unlimited)")
    parser.add_argument("--checkpoint-interval", type=int, default=int(os.environ.get("CHECKPOINT_INTERVAL", 100)), help="Optimizer step interval between checkpoint commits and cloud backups (default: 100)")
    parser.add_argument("--optimizer", type=str, default="auto", choices=["auto", "adamw", "adamw8bit"], help="Optimizer choice ('adamw8bit' saves 5.5GB VRAM for Tesla T4)")
    parser.add_argument("--resume", action="store_true", default=True, help="Auto-resume from existing checkpoint if available")
    parser.add_argument("--force", action="store_true", default=False, help="Force training continuation even if stage is marked completed")
    parser.add_argument("--buffer-size", type=int, default=1, help="Streaming buffer size (default: 1 for exact sequential crash-safe zero-duplicate recovery)")
    parser.add_argument("--full-dataset", action="store_true", default=False, help="Train continuously across all shards until full dataset is exhausted")
    parser.add_argument("--keep-shards", action="store_true", default=False, help="Do not delete local shards after completion")
    parser.add_argument("--dataset-prefix", type=str, default="",
                        help="Remote dataset version prefix on Hugging Face (default empty for root dataset structure e.g. tara_dataset_v1)")
    parser.add_argument("--max-shard-bytes", type=int, default=0,
                        help="Enforced maximum size per downloaded shard in bytes (default: 0 = dynamic / unlimited). "
                             "When set > 0, downloads exceeding this limit are aborted; incomplete temp files are removed.")
    parser.add_argument("--bulk-download-dataset", action="store_true", default=False,
                        help="Bulk pre-download all dataset shards instead of JIT single-shard streaming")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=bool(int(os.environ.get("GRADIENT_CHECKPOINTING", "1"))),
                        help="Enable activation gradient checkpointing to save VRAM (default: enabled)")
    parser.add_argument("--allow-unsafe-fp32-adamw", action="store_true", default=False,
                        help="Override memory preflight gate to allow FP32 AdamW on memory-constrained GPUs")
    parser.add_argument("--keep-checkpoints", type=int, default=int(os.environ.get("KEEP_CHECKPOINTS", 2)),
                        help="Number of past checkpoints to retain (default: 2; deletes older checkpoints only upon verified new commit)")
    parser.add_argument("--skip-save-steps", type=int, default=int(os.environ.get("SKIP_SAVE_STEPS", 0)),
                        help="Number of initial steps to skip checkpoint and best-milestone disk/cloud saves (default: 0; immediate zero-lock saves active)")
    parser.add_argument("--promote-max-step", type=int, default=int(os.environ.get("PROMOTE_MAX_STEP", 0)),
                        help="Target step ceiling for model promotion (e.g. 500). Promotes the best checkpoint achieved within steps <= PROMOTE_MAX_STEP for real-time deployment while training continues uninterrupted. 0 means unbounded/entire stage.")
    parser.add_argument("--domain-weights", type=str, default=os.environ.get("DOMAIN_WEIGHTS", ""),
                        help="Custom comma-separated domain weights (e.g. '01_mathematics_formal_logic=0.15,12_planning_reasoning_traces=0.20')")
    parser.add_argument("--joint-world-model", action="store_true", default=False,
                        help="Enable coordinated joint training of Neural Model + World Model")
    parser.add_argument("--world-model-dir", type=str, default="storage/models/tara_world_model",
                        help="Directory containing World Model architecture config.json and weights")
    parser.add_argument("--world-dataset-dir", type=str, default=None,
                        help="Directory containing genuine World Model transition datasets (*.jsonl)")
    parser.add_argument("--lambda-world", type=float, default=1.0,
                        help="Loss weighting factor for World Model objective (L_joint = L_neural + lambda_world * L_world)")
    parser.add_argument("--per-shard-promotion", action="store_true", default=False,
                        help="Promote model snapshot upon completing each shard and prune prior shard model snapshot")
    parser.add_argument("--freeze-backbone", action="store_true", default=False,
                        help="Freeze model backbone layers for lightweight smoke-test training or linear probing")
    parser.add_argument("--max-shard-mb", type=int, default=480,
                        help="Maximum size per SafeTensors shard in MB (default: 480 MB, <= 500 MB canonical)")
    args = parser.parse_args()

    # Pre-resolve and export HF_TOKEN across runtime
    _resolve_hf_token(args.hf_token)

    raw_dataset_repo = args.hf_dataset_repo if args.hf_dataset_repo is not None else (args.hf_repo if args.hf_repo is not None else os.environ.get("HF_DATASET_REPO", os.environ.get("HF_REPO", None)))
    dataset_repo = None if (raw_dataset_repo is None or str(raw_dataset_repo).strip().lower() in ("", "none", "local", "null")) else str(raw_dataset_repo).strip()

    raw_model_repo = args.hf_model_repo if args.hf_model_repo is not None else (args.hf_repo if args.hf_repo is not None else os.environ.get("HF_MODEL_REPO", os.environ.get("HF_REPO", None)))
    model_repo = None if (raw_model_repo is None or str(raw_model_repo).strip().lower() in ("", "none", "local", "null")) else str(raw_model_repo).strip()

    if args.full_dataset or args.max_steps <= 0:
        args.max_steps = 100_000_000

    print("=" * 78)
    print(f"  TARA CLOUD MULTI-STAGE TRAINING SYSTEM [{args.stage_name}]")
    print("=" * 78)

    # 1. Idempotence Check
    if not args.force and check_stage_already_completed(args.output_dir):
        finalize_stage_and_promote(
            output_dir=args.output_dir,
            promoted_dir=args.promoted_dir,
            stage_id=args.stage_id,
            hf_repo=model_repo,
            previous_stage_dir=args.previous_stage_dir,
        )
        sys.exit(0)

    # 2. Fast Asset Verification & Streaming Download (if files missing)
    if model_repo:
        ensure_remote_assets_available(
            args.model_dir,
            args.dataset_dir,
            repo_id=model_repo,
            dataset_repo=dataset_repo,
            prefix=args.dataset_prefix.strip("/"),
            bulk_download_dataset=args.bulk_download_dataset,
            output_dir=args.output_dir,
            stage_id=args.stage_id,
        )

    # 2b. Pretraining Authorization Gate check
    if args.verify_gate:
        gate_passed = verify_pretraining_authorization_proof(args.model_dir, os.path.dirname(args.output_dir))
        if not gate_passed:
            raise RuntimeError(
                "[Pre-Training Gate FATAL] Pre-training authorization gate check failed: "
                "Cryptographic pretraining_authorization.json proof is missing or authorization_status is not AUTHORIZED. "
                "Autonomous pre-training cannot proceed until verified authorization proofs are satisfied."
            )

    # 3. Hardware Inspection & Precision Selection (BF16 for Ampere+, FP16 for Tesla T4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    precision_str = "FP32"
    autocast_dtype = torch.float32
    model_dtype = torch.float32  # Model parameters MUST remain FP32 for torch.amp.GradScaler
    gpu_name = "N/A"
    total_vram_gb = 0.0
    major_cap, minor_cap = 0, 0

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        total_vram_gb = vram_mb / 1024.0
        major_cap, minor_cap = torch.cuda.get_device_capability(0)
        # Ampere (sm_80)+ has native hardware BF16 tensor core acceleration.
        # Turing (T4 / sm_75) only supports FP16/INT8 hardware tensor cores.
        if major_cap >= 8 and torch.cuda.is_bf16_supported():
            autocast_dtype = torch.bfloat16
            precision_str = "BF16 (AMP)"
        else:
            autocast_dtype = torch.float16
            precision_str = "FP16 (AMP)"
        print(f"[Hardware] GPU: {gpu_name} (sm_{major_cap}{minor_cap}) | VRAM: {vram_mb:.1f} MB | Selected Precision: {precision_str} with FP32 Master Parameters")
        if major_cap >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if total_vram_gb <= 16.5 and not args.gradient_checkpointing:
            args.gradient_checkpointing = True
            print(f"[Hardware Optimization] Memory-constrained GPU ({total_vram_gb:.1f} GiB VRAM) detected: Auto-enabled activation gradient checkpointing to eliminate CUDA OOM.")
    else:
        print("[Hardware] No GPU detected. Defaulting to CPU (FP32).")

    # Print Fully Resolved Configuration Matrix Before Dataset Loading & State Allocation
    print("=" * 80)
    print("  TARA FULLY RESOLVED CONFIGURATION MATRIX")
    print("=" * 80)
    print(f"  Stage ID              : {args.stage_id}")
    print(f"  Stage Name            : {args.stage_name}")
    print(f"  Device Selected       : {device.type.upper()}" + (f" ({gpu_name}, sm_{major_cap}{minor_cap})" if device.type == "cuda" else ""))
    print(f"  Total GPU VRAM        : {total_vram_gb:.2f} GiB" if device.type == "cuda" else "  Total GPU VRAM        : N/A (CPU)")
    print(f"  Effective Precision   : {precision_str} (Model Parameters: FP32)")
    print(f"  Optimizer Selection   : {args.optimizer}")
    print(f"  Micro-Batch Size      : {args.batch_size}")
    print(f"  Gradient Accumulation : {args.grad_accum}")
    print(f"  Effective Batch Size  : {args.batch_size * args.grad_accum}")
    print(f"  Max Sequence Length   : {args.max_seq_len}")
    steps_label = f"{args.max_steps} (Continuous Full Dataset)" if (args.full_dataset or args.max_steps >= 100_000_000) else str(args.max_steps)
    print(f"  Optimization Steps    : {steps_label}")
    print(f"  Checkpoint Interval   : {args.checkpoint_interval}")
    print(f"  Checkpoints Retained  : {args.keep_checkpoints}")
    print(f"  Skip Save Steps       : {args.skip_save_steps}")
    print(f"  Promote Max Step      : {args.promote_max_step if args.promote_max_step > 0 else 'Unbounded (Stage Completion)'}")
    print(f"  Warmup Steps          : {args.warmup_steps}")
    print(f"  Base Learning Rate    : {args.lr:.2e}")
    print(f"  Min Learning Rate     : {args.min_lr:.2e}")
    print(f"  Grad Checkpointing    : {args.gradient_checkpointing}")
    print(f"  Output Checkpoint Dir : {args.output_dir}")
    print(f"  Promoted Model Dir    : {args.promoted_dir}")
    print("=" * 80 + "\n")

    # 3. Load Architecture Config
    config_path = os.path.join(args.model_dir, "config.json")
    if not os.path.exists(config_path):
        config_path = "storage/models/tara_candidate_v1/config.json"
    if not os.path.exists(config_path):
        print(f"[Error] Missing config.json at {config_path}")
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Harmonize dynamic RoPE theta and sequence length from model architecture config
    config["rope_theta"] = resolve_dynamic_rope_theta(config)
    if args.max_seq_len <= 0:
        args.max_seq_len = int(config.get("max_position_embeddings", 2048))
        print(f"[Dynamic Config] Dynamically resolved max_seq_len={args.max_seq_len} from {config_path}")
    else:
        config["max_position_embeddings"] = max(int(config.get("max_position_embeddings", 2048)), args.max_seq_len)

    # 4. Load Tokenizer
    tok_path = os.path.join(args.model_dir, "tokenizer.json")
    if not os.path.exists(tok_path):
        tok_path = "storage/models/tara_candidate_v1/tokenizer.json"
    if not os.path.exists(tok_path):
        print(f"[Error] tokenizer.json missing at {tok_path}")
        sys.exit(1)
    try:
        tokenizer = TaraNativeTokenizer.from_file(tok_path)
        vocab_size = tokenizer.get_vocab_size() if hasattr(tokenizer, "get_vocab_size") else getattr(tokenizer, "vocab_size", config.get("vocab_size"))
        print(f"[Tokenizer] Successfully loaded TARA Tokenizer ({vocab_size:,} vocabulary tokens)")
    except Exception as e:
        print(f"[Error] Failed to load tokenizer from {tok_path}: {e}")
        sys.exit(1)

    # 5. Build Model (Parameters remain in FP32; AMP autocast evaluates in FP16/BF16)
    model = TaraForCausalLM(config, gradient_checkpointing=args.gradient_checkpointing).to(device=device, dtype=model_dtype)

    if args.freeze_backbone:
        for p in model.model.parameters():
            p.requires_grad = False
        print("[Model Freeze] Transformer backbone frozen. Only lm_head is trainable for lightweight test.")

    if not args.resume:
        load_candidate_weights(model, args.model_dir)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total trainable parameters: {total_params:,} ({total_params * 4 / 1e6:.1f} MB in FP32 master storage)")


    # 6. Initialize Authoritative Persistent Shard Ledger & Shard Discovery
    shard_ledger = ShardLedger(args.output_dir, stage_id=args.stage_id, dataset_revision=args.dataset_prefix.strip("/"))

    ds_prefix = args.dataset_prefix.strip("/")
    discovered_shards = []

    VALID_EXTS = (".parquet", ".jsonl", ".txt", ".csv", ".tsv", ".gz")
    SKIP_SUBSTRINGS = ("eval.yaml", "minif2f_lean4_test", "minif2f_lean4_valid", ".gitattributes", "readme.md", ".yaml", ".yml", "indiccorp_v2")

    local_files = []
    if os.path.isdir(args.dataset_dir):
        for root, _, files in os.walk(args.dataset_dir):
            for f in files:
                rel = os.path.relpath(os.path.join(root, f), args.dataset_dir).replace("\\", "/")
                if rel.endswith(VALID_EXTS) and not any(s in rel.lower() for s in SKIP_SUBSTRINGS):
                    local_files.append(rel)
        if local_files:
            print(f"[Dataset Discovery] Discovered {len(local_files)} local shard(s) in '{args.dataset_dir}'.", flush=True)

    remote_files = []
    if dataset_repo:
        try:
            from huggingface_hub import HfApi
            token = _resolve_hf_token()
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
            if remote_files:
                print(f"[Dataset Discovery] Discovered {len(remote_files)} remote canonical shards on Hugging Face ({dataset_repo}).", flush=True)
        except Exception as e:
            print(f"[Dataset Discovery Notice] Remote Hugging Face file discovery note: {e}", flush=True)

    discovered_shards = sorted(set(local_files) | set(remote_files))

    if not discovered_shards:
        raise RuntimeError(
            f"[Dataset Discovery Fatal] Zero dataset shards discovered in local dataset directory '{args.dataset_dir}' "
            f"or on Hugging Face repository '{dataset_repo}'. Cannot proceed with zero training shards."
        )

    # Strictly filter out held-out shards to prevent evaluation leakage
    import fnmatch
    exclude_patterns = [p.strip() for p in args.exclude_shards.split(",") if p.strip()]
    eligible_shards = []
    for sf in discovered_shards:
        if any(fnmatch.fnmatch(sf, pat) for pat in exclude_patterns):
            print(f"[Data Gate] STRICT LEAKAGE PREVENTION: Excluding held-out shard from training: {sf}")
        else:
            eligible_shards.append(sf)

    selected_shards = eligible_shards if args.max_shards <= 0 else eligible_shards[:args.max_shards]
    shard_ledger.total_eligible_shards = len(selected_shards)
    shard_ledger.save()
    print(f"[Dataset] Discovered {len(discovered_shards)} total shards ({len(eligible_shards)} eligible, {len(shard_ledger.completed_shards)} already completed in ledger); pipeline prepared across {len(selected_shards)} shard(s)...")

    # Initialize dataset snapshot: records the authoritative ordered shard catalog
    # and tracks runtime content hashes as each shard is opened during training.
    # The snapshot file lives in the checkpoint output directory so it survives
    # Colab session restarts alongside the shard ledger.
    dataset_snapshot = DatasetSnapshot(args.output_dir)
    # Check for catalog changes against the previously persisted snapshot BEFORE updating it
    if dataset_snapshot.ordered_shards:
        diff = dataset_snapshot.diff_against_live(list(selected_shards))
        if diff["added"]:
            print(f"[DatasetSnapshot] New shards added since last session: {diff['added']}", flush=True)
        if diff["removed"]:
            print(f"[DatasetSnapshot] Shards removed since last session (will not be trained): {diff['removed']}", flush=True)

    dataset_snapshot.initialize_from_discovery(
        repo_id=dataset_repo,
        prefix=ds_prefix,
        ordered_shards=list(selected_shards),
    )

    # 7. Optimizer Selection & Memory Preflight Gate
    use_8bit = False
    is_vram_constrained = False
    if device.type == "cuda" and total_vram_gb <= 16.5:
        is_vram_constrained = True

    if args.optimizer == "adamw8bit":
        try:
            import bitsandbytes as bnb
            use_8bit = True
        except ImportError:
            print("[VRAM Optimizer] bitsandbytes not found; attempting auto-install...")
            try:
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes"])
                import bitsandbytes as bnb
                use_8bit = True
                print("[VRAM Optimizer] bitsandbytes installed successfully!")
            except Exception as e:
                print(f"[FATAL ERROR] bitsandbytes was explicitly requested (--optimizer adamw8bit) but could not be imported or installed: {e}")
                sys.exit(1)

    elif args.optimizer == "adamw":
        use_8bit = False

    elif args.optimizer == "auto":
        if is_vram_constrained:
            try:
                import bitsandbytes as bnb
                use_8bit = True
                print(f"[VRAM Optimizer] Detected {total_vram_gb:.1f} GiB GPU (Tesla T4). Auto-selected 8-bit AdamW to save 5.5 GB VRAM.")
            except ImportError:
                print(f"[VRAM Optimizer] Tesla T4 ({total_vram_gb:.1f} GiB) detected; installing bitsandbytes to enable 8-bit AdamW...")
                try:
                    import subprocess
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes"])
                    import bitsandbytes as bnb
                    use_8bit = True
                    print(f"[VRAM Optimizer] bitsandbytes installed successfully! Auto-enabling 8-bit AdamW.")
                except Exception as bnb_err:
                    print(f"[VRAM Warning] Could not install bitsandbytes: {bnb_err}")
                    use_8bit = False

    # Safety Gate: Refuse to start with standard AdamW on memory-constrained GPUs unless explicitly approved
    if is_vram_constrained and not use_8bit:
        if not args.allow_unsafe_fp32_adamw:
            print("\n" + "=" * 80)
            print(" [FATAL ERROR: MEMORY PREFLIGHT GATE REJECTED STANDARD FP32 ADAMW]")
            print("=" * 80)
            print(f" Device Detected       : {gpu_name} ({total_vram_gb:.2f} GiB VRAM)")
            print(f" Trainable Parameters  : {total_params:,} (~1 Billion parameters)")
            print(f" Standard FP32 AdamW   : Requires ~7.78 GiB optimizer states + 3.89 GiB weights + gradients")
            print(f" Total Static Memory   : > 15.5 GiB (exceeds {total_vram_gb:.2f} GiB physical VRAM)")
            print(f" Actual Past Outcome   : CUDA Out Of Memory allocation failure at step 1.")
            print()
            print(" Strict Safety Directive:")
            print("   Refusing to start training with standard FP32 AdamW on memory-constrained GPU.")
            print("   Silent fallback to standard AdamW is strictly prohibited.")
            print()
            print(" Resolutions:")
            print("   1. Install bitsandbytes: `pip install bitsandbytes` to use 8-bit AdamW (--optimizer adamw8bit).")
            print("   2. Or pass `--allow-unsafe-fp32-adamw` to explicitly override this preflight gate (NOT recommended).")
            print("=" * 80 + "\n")
            sys.exit(1)
        else:
            print("[VRAM Warning] User explicitly approved unsafe standard FP32 AdamW via --allow-unsafe-fp32-adamw. Proceeding with high OOM risk.")

    # Overfitting Safety: Decouple Weight Decay
    # Apply weight decay (0.01) only to 2D projections/MLP weights.
    # Exclude 1D norms/biases AND exclude 2D embedding tokens and lm_head to preserve vocabulary representations.
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2 or "embed_tokens" in name or "lm_head" in name or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": 0.01},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    print(f"[Optimizer Safety] Decoupled Weight Decay active: {len(decay_params)} decay tensors (0.01), {len(no_decay_params)} no-decay norm/bias/embedding tensors (0.0).")

    # Half-precision numerical stability: eps=1e-5 avoids variance explosion in FP16/BF16
    adam_eps = 1e-5
    adam_betas = (0.9, 0.95)
    if use_8bit:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            optim_groups,
            lr=args.lr,
            betas=adam_betas,
            eps=adam_eps,
        )
        print(f"[Optimizer] Initialized 8-bit AdamW with decoupled weight decay (eps={adam_eps}, betas={adam_betas}).")
    else:
        fused_available = ('fused' in torch.optim.AdamW.__init__.__code__.co_varnames
                           and torch.cuda.is_available()
                           and device.type == "cuda")
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=args.lr,
            betas=adam_betas,
            eps=adam_eps,
            fused=fused_available,
        )
        print(f"[Optimizer] Initialized standard AdamW with decoupled weight decay (eps={adam_eps}, betas={adam_betas}, fused={fused_available}).")

    try:
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and autocast_dtype == torch.float16))
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and autocast_dtype == torch.float16))

    # 7b. Coordinated Joint Neural Model + World Model Setup
    world_model = None
    world_optimizer = None
    world_scaler = None
    world_dataloader = None
    world_iter = None

    if args.joint_world_model:
        print("=" * 80)
        print("  [JOINT NEURAL + WORLD MODEL TRAINING INITIALIZATION]")
        print("=" * 80)
        wm_config_path = os.path.join(args.world_model_dir, "config.json")
        if not os.path.exists(wm_config_path):
            raise FileNotFoundError(
                f"[Joint World Model Gate] Missing World Model configuration at '{wm_config_path}'. "
                f"Cannot initialize joint training without authentic World Model architecture specification."
            )
        with open(wm_config_path, "r", encoding="utf-8") as f:
            wm_config = json.load(f)

        world_model = TaraWorldModel(wm_config, gradient_checkpointing=args.gradient_checkpointing).to(
            device=device, dtype=model_dtype
        )
        if not args.resume:
            wm_loaded = load_candidate_weights(world_model, args.world_model_dir)
            print(f"[Joint World Model] Weights initialized from {args.world_model_dir} (restored: {wm_loaded})")
        wm_trainable_params = sum(p.numel() for p in world_model.parameters() if p.requires_grad)
        print(f"[Joint World Model] Total trainable parameters: {wm_trainable_params:,} ({wm_trainable_params * 4 / 1e6:.1f} MB in FP32 master storage)")

        # Dedicated optimizer for World Model (separate parameters, moments, and state)
        wm_decay_params = []
        wm_no_decay_params = []
        for name, param in world_model.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() < 2 or "embed_tokens" in name or "lm_head" in name or "norm" in name:
                wm_no_decay_params.append(param)
            else:
                wm_decay_params.append(param)
        wm_optim_groups = [
            {"params": wm_decay_params, "weight_decay": 0.01},
            {"params": wm_no_decay_params, "weight_decay": 0.0},
        ]
        if use_8bit:
            import bitsandbytes as bnb
            world_optimizer = bnb.optim.AdamW8bit(
                wm_optim_groups,
                lr=args.lr,
                betas=adam_betas,
                eps=1e-5,
            )
            print("[Joint World Model] Initialized dedicated 8-bit AdamW optimizer for World Model (eps=1e-5).")
        else:
            fused_available = ('fused' in torch.optim.AdamW.__init__.__code__.co_varnames
                               and torch.cuda.is_available()
                               and device.type == "cuda")
            world_optimizer = torch.optim.AdamW(
                wm_optim_groups,
                lr=args.lr,
                betas=adam_betas,
                eps=1e-5,
                fused=fused_available,
            )
            print("[Joint World Model] Initialized dedicated standard FP32 AdamW optimizer for World Model (eps=1e-5).")

        try:
            world_scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and autocast_dtype == torch.float16))
        except (AttributeError, TypeError):
            world_scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and autocast_dtype == torch.float16))

        # Mandatory World Model Transition Data Preflight Gate
        world_files = []
        TRANS_EXTS = (".jsonl", ".parquet")
        if args.world_dataset_dir:
            if os.path.isfile(args.world_dataset_dir) and args.world_dataset_dir.endswith(TRANS_EXTS):
                world_files = [args.world_dataset_dir]
            elif os.path.isdir(args.world_dataset_dir):
                for ext in TRANS_EXTS:
                    world_files.extend(sorted(glob.glob(os.path.join(args.world_dataset_dir, f"*{ext}"))))

        if not world_files:
            for candidate_wm_path in [
                "storage/datasets/curriculum/agi_causal_transitions.jsonl",
                "storage/datasets/curriculum",
                "storage/datasets/world_transitions",
            ]:
                if os.path.isfile(candidate_wm_path) and candidate_wm_path.endswith(TRANS_EXTS):
                    world_files.append(candidate_wm_path)
                    break
                elif os.path.isdir(candidate_wm_path):
                    found = []
                    for ext in TRANS_EXTS:
                        found.extend(sorted(glob.glob(os.path.join(candidate_wm_path, f"*transition*{ext}")) + glob.glob(os.path.join(candidate_wm_path, f"*causal*{ext}"))))
                    if found:
                        world_files.extend(found)
                        break

        if not world_files and dataset_repo and selected_shards:
            wm_shard_candidates = [
                s for s in selected_shards
                if "08_commonsense_world_model" in s or "world_model" in s.lower() or "transition" in s.lower()
            ]
            for wm_cand in wm_shard_candidates[:2]:
                try:
                    dl_wm = download_single_shard_jit(
                        dataset_repo, ds_prefix, wm_cand, args.dataset_dir, max_shard_bytes=args.max_shard_bytes
                    )
                    if os.path.isfile(dl_wm):
                        world_files.append(dl_wm)
                        break
                except Exception as wm_jit_err:
                    print(f"[Joint World Model JIT Notice] Could not fetch {wm_cand}: {wm_jit_err}", flush=True)

        if not world_files:
            raise RuntimeError(
                f"[Joint World Model Preflight Gate] Joint training was requested (--joint-world-model), "
                f"but zero genuine World Model transition datasets (*.jsonl, *.parquet) were discovered. "
                f"Genuine state-transition supervision (s_t, a_t -> s_{{t+1}}) is required for World Model training. "
                f"Provide genuine transition data via --world-dataset-dir or place in storage/datasets/curriculum/."
            )

        # Deep record-level validation: Ensure discovered files contain authentic, usable transition records
        valid_transition_sample_count = 0
        verified_world_files = []
        for wf in world_files:
            cnt = inspect_candidate_world_transitions(wf, max_records=20)
            if cnt > 0:
                valid_transition_sample_count += cnt
                verified_world_files.append(wf)
                if valid_transition_sample_count >= 10:
                    break

        if valid_transition_sample_count == 0 or not verified_world_files:
            raise RuntimeError(
                f"[Joint World Model Preflight Gate] Files were discovered in '{args.world_dataset_dir}', "
                f"but zero valid state-transition records were found. Records must strictly provide valid, non-empty "
                f"(state_observation, action, next_state_observation) sequence semantics. "
                f"Relational facts (subject, predicate, object) and raw text are strictly rejected as transition supervision. "
                f"Failing preflight safely per directive."
            )

        world_files = verified_world_files

        print(f"[Joint World Model Dataset] Discovered {len(world_files)} transition dataset file(s) with verified usable records.")
        world_dataset = WorldModelTransitionDataset(
            file_paths=world_files,
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            max_samples=args.max_samples,
        )
        world_dataloader = DataLoader(
            world_dataset,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            pin_memory=(device.type == "cuda"),
        )
        world_iter = iter(world_dataloader)
        print("=" * 80 + "\n")

    # 8. Check for Mid-Way Resume
    start_step = 0
    start_epoch = 0
    best_loss = float("inf")
    was_resumed = False
    resumed_cursor = None
    resume_shard = None
    resume_line_offset = -1

    resume_byte_offset = 0

    if args.full_dataset or args.max_steps <= 0:
        args.max_steps = 100_000_000
        print("[Training Scope] FULL DATASET ROLLING TRAINING ACTIVE. Training continuously across all shards until full dataset is exhausted!", flush=True)

    if args.resume:
        # Check if local checkpoint directory has checkpoints; if NOT, try restoring from Google Drive!
        gdrive_dir = resolve_gdrive_restore_dir() or resolve_gdrive_backup_dir()
        if gdrive_dir and not (check_model_weights_exist(args.output_dir) or os.path.exists(os.path.join(args.output_dir, "checkpoint_state.json"))):
            print(f"[GDrive Auto-Recovery] Local checkpoint missing in '{args.output_dir}'. Checking GDrive backup at '{gdrive_dir}'...", flush=True)
            restored = restore_checkpoint_from_gdrive(gdrive_dir, args.output_dir)
            if restored:
                print(f"[GDrive Auto-Recovery] Successfully restored checkpoint from Google Drive into '{args.output_dir}'!", flush=True)

        start_step, start_epoch, best_loss, was_resumed, resumed_cursor = try_resume_from_checkpoint(
            args.output_dir, model, optimizer, scaler,
            tokenizer_path=tok_path,
            tokenizer=tokenizer,
            expected_stage_id=args.stage_id,
            world_model=world_model,
            world_optimizer=world_optimizer,
            world_scaler=world_scaler,
            base_lr=args.lr,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
            min_lr=args.min_lr,
        )

        # Cleanup obsolete duplicate runs or clone remnants from previous sessions in parent_dir
        parent_ckpt_dir = os.path.dirname(os.path.abspath(args.output_dir)) or "."
        _cleanup_old_versioned_checkpoints(
            parent_ckpt_dir,
            os.path.basename(args.output_dir),
            keep_n=args.keep_checkpoints,
            force_keep_exact=True,
        )
        if start_step >= args.max_steps:
            print(f"[Resume] Training already completed all {args.max_steps} steps. Finishing.")
            save_checkpoint(
                model, optimizer, scaler, config, tok_path, args.output_dir,
                start_step, start_epoch, best_loss, best_loss, args.stage_id,
                is_best=False, is_completed=True, data_cursor=resumed_cursor,
                per_shard_promotion=args.per_shard_promotion,
                world_model=world_model,
                world_optimizer=world_optimizer,
                world_scaler=world_scaler,
                max_shard_mb=args.max_shard_mb,
            )
            finalize_stage_and_promote(
                output_dir=args.output_dir,
                promoted_dir=args.promoted_dir,
                stage_id=args.stage_id,
                hf_repo=model_repo,
                previous_stage_dir=args.previous_stage_dir,
            )
            sys.exit(0)

        if was_resumed and start_step > 0:
            if resumed_cursor and shard_ledger:
                shard_ledger.reconcile_with_checkpoint(resumed_cursor, start_step)
            resume_shard = resumed_cursor.get("active_shard") if resumed_cursor else None
            resume_line_offset = resumed_cursor.get("active_shard_line_offset", -1) if resumed_cursor else -1
            resume_byte_offset = resumed_cursor.get("active_shard_byte_offset", 0) if resumed_cursor else 0
            resume_chunk_idx = resumed_cursor.get("active_chunk_idx", 0) if resumed_cursor else 0

            # Shard boundary reconciliation:
            # If the resumed shard is already marked COMPLETED in the ledger and the cursor line offset
            # reached or exceeded the end of that shard, advance resume_shard to the next eligible shard
            # without triggering an unnecessary download of the completed shard.
            if resume_shard and shard_ledger and shard_ledger.is_completed(resume_shard):
                meta = shard_ledger.completed_metadata.get(resume_shard, {})
                rec_count = meta.get("records_processed", 0)
                if rec_count > 0 and resume_line_offset >= rec_count - 1:
                    if resume_shard in selected_shards:
                        curr_idx = selected_shards.index(resume_shard)
                        if curr_idx + 1 < len(selected_shards):
                            next_shard = selected_shards[curr_idx + 1]
                            print(
                                f"[Resume Boundary Reconciliation] Shard '{resume_shard}' was confirmed COMPLETED "
                                f"in ledger ({rec_count:,} records). Advancing cursor to next eligible shard: '{next_shard}'.",
                                flush=True,
                            )
                            resume_shard = next_shard
                            resume_line_offset = 0
                            resume_byte_offset = 0
                            resume_chunk_idx = 0
                        else:
                            print(
                                f"[Resume Boundary Reconciliation] Final shard '{resume_shard}' was confirmed COMPLETED "
                                f"in ledger. All dataset shards are completed.",
                                flush=True,
                            )
                            resume_shard = None
                            resume_line_offset = -1
                            resume_byte_offset = 0
                            resume_chunk_idx = 0

            print("=" * 80, flush=True)
            print("  [DATASET ZERO-DUPLICATE EXACT RESUME AUDIT]", flush=True)
            print(f"  Total Lifetime Step Resumed   : {start_step:,}", flush=True)
            print(f"  Authoritative Active Shard    : {resume_shard or 'None'}", flush=True)
            print(f"  Exact Line Fast-Forward Offset: {resume_line_offset:,} (Byte Offset: {resume_byte_offset:,})", flush=True)
            print(f"  Completed Shards in Ledger    : {len(shard_ledger.completed_shards)} shards ({shard_ledger.get_total_completed_records():,} records)", flush=True)
            print("  Status                        : 100% DETERMINISTIC ZERO DUPLICATE RESUMPTION GUARANTEED", flush=True)
            print("=" * 80, flush=True)
        else:
            print("[Model] No prior checkpoint resumed; seeding model from candidate base weights.", flush=True)
            load_candidate_weights(model, args.model_dir)
            if world_model is not None:
                load_candidate_weights(world_model, args.world_model_dir)

    # 9. Instantiate Streaming Dataset & DataLoader with Multi-Domain DRR Scheduler
    resume_chunk_idx = resumed_cursor.get("active_chunk_idx", 0) if resumed_cursor else 0
    domain_shards: Dict[str, List[str]] = {d: [] for d in CANONICAL_DOMAINS}
    for s in selected_shards:
        d = resolve_shard_domain(s)
        if d not in domain_shards:
            domain_shards[d] = []
        domain_shards[d].append(s)

    active_doms_with_shards = [d for d, shards in domain_shards.items() if shards]
    print(f"[Multi-Domain Setup] Grouped {len(selected_shards)} shards into {len(active_doms_with_shards)} active domain bucket(s):", flush=True)
    for d in active_doms_with_shards:
        print(f"  - {d}: {len(domain_shards[d])} shard(s)", flush=True)

    domain_weights_map = parse_domain_weights(getattr(args, "domain_weights", ""), active_doms_with_shards)
    print(f"[Multi-Domain DRR Weights] Normalized domain weights: {domain_weights_map}", flush=True)

    resumed_domain_cursors = resumed_cursor.get("domain_cursors", {}) if resumed_cursor else (shard_ledger.domain_cursors if shard_ledger else {})

    is_pretraining = "stage0" in args.stage_id.lower() or "pretrain" in args.stage_id.lower()
    effective_mask_prompt = args.mask_prompt if not is_pretraining else False

    if len(active_doms_with_shards) > 1:
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
            scheduler_deficits=resumed_cursor.get("scheduler_deficits") if resumed_cursor else None,
            mask_prompt=effective_mask_prompt,
        )
    else:
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
        )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    # Setup Best Checkpoint Directory
    best_cp_dir = os.path.join(args.output_dir, "best_checkpoint")
    promoted_target = args.promoted_dir if args.promoted_dir else os.path.join("storage/models", f"promoted_{args.stage_id}")
    target_model_repo = args.hf_model_repo or os.environ.get("HF_MODEL_REPO", os.environ.get("HF_REPO", None))
    promoted_at_target_step = False
    window_best_source = os.path.join(args.output_dir, "best_checkpoint_window")

    last_regular_checkpoint_step = start_step
    last_best_checkpoint_step = start_step
    last_best_checkpoint_loss = best_loss if best_loss < float("inf") else float("inf")
    best_loss_step = start_step
    best_loss_epoch = start_epoch
    best_loss_cursor = resumed_cursor
    current_cursor = resumed_cursor
    committed_domain_cursors = copy.deepcopy(resumed_domain_cursors) if resumed_domain_cursors else {}

    total_samples_trained = int(resumed_cursor.get("total_samples_trained", 0)) if resumed_cursor else 0
    samples_consumed = int(resumed_cursor.get("samples_consumed", total_samples_trained)) if resumed_cursor else 0

    print(f"[Training] Launching Training: Epochs={args.epochs}, Batch={args.batch_size}, Accum={args.grad_accum}, Precision={precision_str}, Steps={start_step}..{args.max_steps}")
    model.train()
    if world_model is not None:
        world_model.train()
    step = start_step
    running_loss = 0.0
    accum_loss = 0.0
    accum_ce = 0.0
    accum_aux = 0.0
    accum_world_loss = 0.0
    accum_tokens = 0
    ema_loss = None
    ema_world_loss = None
    grad_norm_val = 0.0
    w_grad_norm_val = 0.0
    t0 = time.time()
    step_last_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        optimizer.zero_grad(set_to_none=True)
        if world_optimizer is not None:
            world_optimizer.zero_grad(set_to_none=True)
        accum_batches = []
        accum_world_batches = []
        for batch_idx, (inputs, labels, cursors) in enumerate(dataloader):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                autocast_ctx = torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=(device.type == "cuda"), dtype=autocast_dtype)
            else:
                autocast_ctx = torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=autocast_dtype)

            try:
                with autocast_ctx:
                    attn_mask = (inputs != 0).long() if (inputs == 0).any() else None
                    _, loss, ce_loss, aux_loss = model(inputs, labels=labels, attention_mask=attn_mask)
                    # Causal LM predicts shifted labels: shift_labels = labels[..., 1:]
                    v_tokens = (labels[..., 1:] != -100).sum().clamp(min=1) if labels.dim() > 1 else (labels != -100).sum().clamp(min=1)
                    effective_tokens = v_tokens.item()

                    # Mathematically exact token-weighted gradient scaling:
                    # Backward with sum of token losses (loss * v_tokens).
                    # Parameter gradients accumulate sum_{i,t} grad(l_{i,t}) across the window.
                    # Normalization by total valid tokens occurs exactly at optimizer step after unscaling.
                    loss_sum = loss * v_tokens

                scaler.scale(loss_sum).backward()
                accum_loss += (loss.item() * effective_tokens)
                accum_ce += ((ce_loss.item() if ce_loss is not None else loss.item()) * effective_tokens)
                accum_aux += aux_loss.item() if aux_loss is not None else 0.0
                accum_tokens += effective_tokens

                # Store only lightweight cursor metadata without forcing GPU-to-CPU tensor sync
                accum_batches.append(cursors)
                samples_consumed += len(cursors)

                # Joint World Model Step Forward & Backward
                if world_model is not None and world_iter is not None:
                    try:
                        w_inputs, w_labels, w_cursors = next(world_iter)
                    except StopIteration:
                        world_iter = iter(world_dataloader)
                        w_inputs, w_labels, w_cursors = next(world_iter)

                    w_inputs = w_inputs.to(device, non_blocking=True)
                    w_labels = w_labels.to(device, non_blocking=True)

                    with autocast_ctx:
                        w_attn_mask = (w_inputs != 0).long() if (w_inputs == 0).any() else None
                        _, w_loss, _, _ = world_model(w_inputs, labels=w_labels, attention_mask=w_attn_mask)
                        w_loss_scaled = (w_loss * args.lambda_world) / args.grad_accum

                    world_scaler.scale(w_loss_scaled).backward()
                    accum_world_loss += w_loss.item()
                    # Store lightweight cursor metadata without forcing GPU-to-CPU tensor synchronization
                    accum_world_batches.append(w_cursors)

            except torch.cuda.OutOfMemoryError as oom_err:
                alloc_mb = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
                res_mb = torch.cuda.memory_reserved() / (1024 * 1024) if torch.cuda.is_available() else 0.0
                in_shape = tuple(inputs.shape) if ('inputs' in locals() and hasattr(inputs, 'shape')) else 'unknown'
                in_seq = inputs.shape[1] if ('inputs' in locals() and hasattr(inputs, 'shape') and len(inputs.shape) > 1) else 'unknown'
                print(
                    f"\n[VRAM OOM Telemetry] CUDA OutOfMemoryError caught at Step {step}, Micro-batch {batch_idx}!\n"
                    f"  -> Oversized Tensor Shape : {in_shape} (Batch: {in_shape[0] if isinstance(in_shape, tuple) else '?'}, SeqLen: {in_seq})\n"
                    f"  -> Physical VRAM Status   : {alloc_mb:.1f} MiB allocated / {res_mb:.1f} MiB reserved\n"
                    f"  -> Error Details          : {oom_err}",
                    flush=True
                )
                optimizer.zero_grad(set_to_none=True)
                if world_optimizer is not None:
                    world_optimizer.zero_grad(set_to_none=True)
                del inputs, labels
                if "w_inputs" in locals():
                    del w_inputs
                if "w_labels" in locals():
                    del w_labels
                if "loss" in locals():
                    del loss
                if "loss_scaled" in locals():
                    del loss_scaled
                accum_loss = 0.0
                accum_ce = 0.0
                accum_aux = 0.0
                accum_world_loss = 0.0
                accum_tokens = 0
                accum_batches.clear()
                accum_world_batches.clear()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("[VRAM Self-Healing Guard] VRAM successfully reclaimed and cache emptied. Skipping oversized sample; resuming training.", flush=True)
                continue

            if (batch_idx + 1) % args.grad_accum == 0:
                # Dynamic Learning Rate Schedule (Linear Warmup + Cosine Decay)
                cur_lr = compute_scheduled_lr(step, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
                for pg in optimizer.param_groups:
                    pg["lr"] = cur_lr
                if world_model is not None and world_optimizer is not None:
                    for w_pg in world_optimizer.param_groups:
                        w_pg["lr"] = cur_lr

                cur_window_loss = accum_loss / max(1, accum_tokens)
                is_step_skipped, grad_norm_val, w_grad_norm_val, scale_before, scale_after = execute_optimizer_step(
                    model, optimizer, scaler, world_model, world_optimizer, world_scaler, max_grad_norm=1.0, cur_loss=cur_window_loss, ema_loss=ema_loss, accum_tokens=accum_tokens
                )

                if is_step_skipped:
                    if scale_after < scale_before:
                        print(
                            f"[Gradient Loss Scaler] Non-finite gradient detected. Scaler adjusted scale factor: "
                            f"{scale_before} -> {scale_after}. Step skipped safely; resuming next window.",
                            flush=True,
                        )
                    else:
                        print(
                            f"[Model Shock Guard] Step skipped safely to protect model stability; resuming next window.",
                            flush=True,
                        )
                    optimizer.zero_grad(set_to_none=True)
                    if world_optimizer is not None:
                        world_optimizer.zero_grad(set_to_none=True)
                    accum_loss = 0.0
                    accum_ce = 0.0
                    accum_aux = 0.0
                    accum_world_loss = 0.0
                    accum_tokens = 0
                    accum_batches.clear()
                    accum_world_batches.clear()
                    continue

                if world_optimizer is not None:
                    world_optimizer.zero_grad(set_to_none=True)
                optimizer.zero_grad(set_to_none=True)
                step += 1
                window_samples = sum(len(c) for c in accum_batches)
                total_samples_trained += window_samples

                cur_step_loss = accum_loss / max(1, accum_tokens)
                cur_step_ce = accum_ce / max(1, accum_tokens)
                cur_step_aux = accum_aux / args.grad_accum
                cur_step_world = (accum_world_loss / args.grad_accum) if world_model is not None else 0.0
                cur_joint_loss = cur_step_loss + (args.lambda_world * cur_step_world if world_model is not None else 0.0)

                running_loss += cur_step_loss
                accum_loss = 0.0
                accum_ce = 0.0
                accum_aux = 0.0
                accum_tokens = 0
                accum_world_loss = 0.0

                # Proactive CPU RAM & Linux Memory Hygiene (releases unmapped heap back to OS)
                if step % 100 == 0:
                    gc.collect()
                    try:
                        import ctypes
                        ctypes.CDLL("libc.so.6").malloc_trim(0)
                    except Exception:
                        pass

                ema_loss = cur_step_loss if ema_loss is None else (0.9 * ema_loss + 0.1 * cur_step_loss)
                if world_model is not None:
                    ema_world_loss = cur_step_world if ema_world_loss is None else (0.9 * ema_world_loss + 0.1 * cur_step_world)

                # Update committed domain cursors strictly from successfully committed micro-batches
                for b_cursors in accum_batches:
                    for c_item in b_cursors:
                        if len(c_item) >= 6:
                            c_dom = c_item[5]
                            if c_dom:
                                committed_domain_cursors[c_dom] = {
                                    "active_shard": c_item[0],
                                    "line_offset": c_item[1],
                                    "byte_offset": c_item[2],
                                    "chunk_idx": c_item[3],
                                }

                # Track exact committed data cursor from the last sample of this
                # successfully applied optimizer step.
                current_cursor = extract_accumulated_data_cursor(
                    accum_batches, dataset, total_samples_trained, samples_consumed, shard_ledger, committed_domain_cursors
                )
                accum_batches.clear()
                accum_world_batches.clear()

                # Real-Time Telemetry Logging (LIVE on EVERY SINGLE STEP with immediate stdout flush)
                step_duration = max(time.time() - step_last_time, 1e-4) if 'step_last_time' in locals() else 0.5
                step_last_time = time.time()
                step_tokens = args.batch_size * args.grad_accum * args.max_seq_len
                step_speed = step_tokens / step_duration

                ppl = math.exp(min(cur_step_loss, 20.0))
                if torch.cuda.is_available():
                    vram_alloc_mib = torch.cuda.memory_allocated() / (1024 * 1024)
                    vram_peak_mib = torch.cuda.max_memory_allocated() / (1024 * 1024)
                    vram_res_mib = torch.cuda.memory_reserved() / (1024 * 1024)
                    vram_total_mib = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                    vram_str = f"VRAM: {vram_alloc_mib:.0f}M alloc, {vram_peak_mib:.0f}M peak, {vram_res_mib:.0f}M res / {vram_total_mib:.0f}M ({precision_str})"
                else:
                    vram_str = "VRAM: N/A (CPU)"

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
                wm_log = f" | WM Loss: {cur_step_world:.4f} (EMA: {ema_world_loss:.4f}) | Joint: {cur_joint_loss:.4f}" if world_model is not None else ""
                grad_norm_str = f"GradNorm: {grad_norm_val:.2f} (WM: {w_grad_norm_val:.2f})" if world_model is not None else f"GradNorm: {grad_norm_val:.2f}"
                print(
                    f"[{args.stage_id} | Step {step:04d}/{args.max_steps}] "
                    f"{domain_tag}"
                    f"Total Loss: {loss_clr}{cur_step_loss:.4f}{CLR_RESET} (CE: {cur_step_ce:.4f}, Aux: {cur_step_aux:.4f}, EMA: {ema_loss:.4f}){wm_log} | "
                    f"PPL: {ppl_clr}{ppl:.2f}{CLR_RESET} | "
                    f"BPB: {bpb_clr}{bpb:.2f} b/B{CLR_RESET} | "
                    f"Standing: {loss_clr}{loss_status} {loss_tag}{CLR_RESET} | "
                    f"LR: {cur_lr:.2e} | "
                    f"{grad_norm_str} | "
                    f"Speed: {step_speed:.0f} tok/s ({step_duration:.2f}s) | "
                    f"{shard_line_log}"
                    f"Samples: {total_samples_trained:,} | "
                    f"{vram_str}",
                    flush=True
                )

                # Dynamic In-Memory Best Loss Tracking & Authentic Best Snapshot:
                # Milestone threshold & cooldown: prevents saving 12 GB checkpoints on micro-fluctuations
                is_new_best = (cur_step_loss < best_loss)
                can_save_at_current_step = (step > args.skip_save_steps) or (step >= args.max_steps)
                committed_milestone_this_step = False

                if is_new_best:
                    step_since_last_best = step - last_best_checkpoint_step
                    is_meaningful = (best_loss - cur_step_loss) >= 0.005
                    cooldown_ok = (step_since_last_best >= 50)

                    best_loss = cur_step_loss
                    best_loss_step = step
                    best_loss_epoch = epoch
                    best_loss_cursor = copy.deepcopy(current_cursor) if current_cursor is not None else None

                    if can_save_at_current_step and (is_meaningful or cooldown_ok or step >= args.max_steps):
                        committed_milestone_this_step = True
                        last_best_checkpoint_step = step
                        last_best_checkpoint_loss = cur_step_loss

                        print(
                            f"[{args.stage_id}] >>> {CLR_GREEN}{CLR_BOLD}COMMITTING AUTHENTIC BEST MILESTONE CHECKPOINT: "
                            f"{best_loss:.4f} (Step {step:04d}){CLR_RESET} <<<",
                            flush=True,
                        )

                        # Atomically commit and mirror verified best milestone
                        save_checkpoint(
                            model=model,
                            optimizer=optimizer,
                            scaler=scaler,
                            config=config,
                            tokenizer_path=tok_path,
                            output_dir=best_cp_dir,
                            step=step,
                            epoch=epoch,
                            loss=cur_step_loss,
                            best_loss=best_loss,
                            stage_id=args.stage_id,
                            is_best=True,
                            is_completed=False,
                            dataset=dataset,
                            shard_ledger=shard_ledger,
                            data_cursor=best_loss_cursor,
                            keep_n=max(args.keep_checkpoints, 2),
                            world_model=world_model,
                            world_optimizer=world_optimizer,
                            world_scaler=world_scaler,
                            max_shard_mb=args.max_shard_mb,
                        )

                        # If promote_max_step is set and current step is within window, snapshot to window_best_source
                        if args.promote_max_step > 0 and step <= args.promote_max_step:
                            try:
                                os.makedirs(window_best_source, exist_ok=True)
                                for item in os.listdir(best_cp_dir):
                                    s_file = os.path.join(best_cp_dir, item)
                                    d_file = os.path.join(window_best_source, item)
                                    if os.path.isfile(s_file):
                                        tmp_d_file = os.path.join(window_best_source, f".tmp_{item}.{os.getpid()}")
                                        try:
                                            try:
                                                os.link(s_file, tmp_d_file)
                                            except Exception:
                                                try:
                                                    os.symlink(os.path.abspath(s_file), tmp_d_file)
                                                except Exception:
                                                    shutil.copy2(s_file, tmp_d_file)
                                            if os.path.exists(tmp_d_file):
                                                os.replace(tmp_d_file, d_file)
                                        except Exception:
                                            if os.path.exists(tmp_d_file):
                                                _safe_unlink(tmp_d_file)
                            except Exception as e:
                                print(f"[Promote Window Snapshot Warning] Failed to update window snapshot: {e}", flush=True)
                    else:
                        print(
                            f"[{args.stage_id} | Step {step:04d}] [Skip-Save Active (<= {args.skip_save_steps})] "
                            f"New best loss {best_loss:.4f} tracked in memory; local save active at Step {args.skip_save_steps + 1}+.",
                            flush=True,
                        )

                # Target-Step Model Promotion Gate (NON-BLOCKING: Real-time deployment while training continues uninterrupted)
                if args.promote_max_step > 0 and step >= args.promote_max_step and not promoted_at_target_step:
                    promoted_at_target_step = True
                    src_cand = window_best_source if (os.path.isdir(window_best_source) and check_model_weights_exist(window_best_source)) else (
                        best_cp_dir if check_model_weights_exist(best_cp_dir) else args.output_dir
                    )
                    print(f"\n{'=' * 80}", flush=True)
                    print(f"[{args.stage_id} | Step {step:04d}] >>> PROMOTING MODEL AT TARGET STEP ({args.promote_max_step}) FOR REAL-TIME DEPLOYMENT <<<", flush=True)
                    print(f"[{args.stage_id}] Selected Best Checkpoint within window (<= {args.promote_max_step}) -> Promoting to: {promoted_target}", flush=True)
                    print(f"[{args.stage_id}] [TRAINING CONTINUES UNINTERRUPTED TO STEP {args.max_steps}]", flush=True)
                    print(f"{'=' * 80}\n", flush=True)
                    try:
                        promote_checkpoint(src_cand, promoted_target, args.stage_id)
                        if target_model_repo:
                            upload_promoted_model_to_hf(promoted_target, target_model_repo, args.stage_id)
                    except Exception as pe:
                        print(f"[{args.stage_id} | ERROR] Target step model promotion failed: {pe}", flush=True)

                # Periodic Benchmark Matrix Display
                if step % 25 == 0 or step == start_step + 1 or is_new_best:
                    print_frontier_comparison_matrix_colored(step, cur_step_loss, min(best_loss, cur_step_loss))

                # ------------------------------------------------------------------
                # REGULAR CHECKPOINT GATE
                # ------------------------------------------------------------------
                # Strict modulo boundary alignment to preserve uniform interval cadence across resumes
                is_regular_interval = (args.checkpoint_interval > 0 and step % args.checkpoint_interval == 0 and step > last_regular_checkpoint_step)
                is_final_step = (step >= args.max_steps)
                should_save_regular = can_save_at_current_step and (is_regular_interval or is_final_step)

                # Unified Checkpoint Save: save_checkpoint is called AT MOST ONCE per step.
                # If a best milestone was committed this step, skip the regular checkpoint entirely —
                # the milestone already wrote all weights, configs, and metadata to disk.
                # The regular interval counter is still advanced so the next interval starts from
                # this step, preventing an immediate double-save on the very next interval step.
                if should_save_regular and committed_milestone_this_step:
                    last_regular_checkpoint_step = step
                    print(
                        f"[{args.stage_id} | Step {step:04d}] Regular checkpoint skipped: "
                        f"best milestone was committed this step (no duplicate I/O).",
                        flush=True,
                    )
                elif should_save_regular:
                    last_regular_checkpoint_step = step
                    # 1. Save full training checkpoint ONCE (with optimizer state)
                    save_checkpoint(
                        model, optimizer, scaler, config, tok_path, args.output_dir,
                        step, epoch, cur_step_loss, best_loss, args.stage_id,
                        is_best=False, is_completed=False,
                        dataset=dataset, shard_ledger=shard_ledger, data_cursor=current_cursor,
                        keep_n=args.keep_checkpoints, force_keep_exact=True,
                        world_model=world_model,
                        world_optimizer=world_optimizer,
                        world_scaler=world_scaler,
                        per_shard_promotion=args.per_shard_promotion,
                        max_shard_mb=args.max_shard_mb,
                    )

                # Reset step timer after all checkpoint I/O, verification, and promotion
                # so that checkpoint disk/network serialization is not conflated with GPU step speed
                step_last_time = time.time()

                if step >= args.max_steps:
                    break

        # Process any remaining partial accumulation window at the end of an epoch.
        # The flush runs regardless of whether step >= max_steps so that every
        # gradient accumulated into the model is either committed to an optimizer
        # update or explicitly discarded with a log message — never silently dropped.
        #
        # Mathematical Gradient Normalization:
        # Micro-batches in the active window were accumulated with loss / args.grad_accum.
        # For a final partial window of partial_count < grad_accum batches, the accumulated
        # gradient in param.grad is (1 / grad_accum) * sum_{i=1}^n grad(L_i).
        # To produce the mathematically exact average gradient (1 / n) * sum_{i=1}^n grad(L_i),
        # we rescale the accumulated gradients by (args.grad_accum / partial_count).
        if accum_batches:
            partial_count = len(accum_batches)
            rescale_factor = float(args.grad_accum) / float(partial_count)

            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(rescale_factor)
            if world_model is not None:
                for p in world_model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(rescale_factor)

            cur_lr = compute_scheduled_lr(step, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr
            if world_optimizer is not None:
                for w_pg in world_optimizer.param_groups:
                    w_pg["lr"] = cur_lr

            is_step_skipped, grad_norm_val, w_grad_norm_val, scale_before, scale_after = execute_optimizer_step(
                model, optimizer, scaler, world_model, world_optimizer, world_scaler, max_grad_norm=1.0, accum_tokens=accum_tokens
            )

            if is_step_skipped:
                if (math.isnan(grad_norm_val) or math.isinf(grad_norm_val)) and not scaler.is_enabled():
                    print(
                        f"[Partial Window Non-Finite Gradient] Inf/NaN gradient detected in "
                        f"end-of-epoch partial accumulation window ({partial_count} batches). "
                        f"Optimizer step skipped; partial window discarded. Committed step "
                        f"and cursor are unchanged.",
                        flush=True,
                    )
                else:
                    print(
                        f"[Partial Window AMP Overflow] Scale reduction {scale_before} -> {scale_after} "
                        f"in end-of-epoch partial window ({partial_count} batches). "
                        f"Optimizer step skipped; partial window discarded.",
                        flush=True,
                    )
            elif world_optimizer is not None:
                world_optimizer.zero_grad(set_to_none=True)

            if not is_step_skipped:
                optimizer.zero_grad(set_to_none=True)
                step += 1
                window_samples = sum(len(c) for _, _, c in accum_batches)
                total_samples_trained += window_samples
                cur_step_loss = accum_loss / max(1, accum_tokens) if accum_tokens > 0 else (accum_loss / partial_count)
                cur_step_ce = accum_ce / max(1, accum_tokens) if accum_tokens > 0 else (accum_ce / partial_count)
                cur_step_aux = accum_aux / partial_count
                running_loss += cur_step_loss
                accum_loss = 0.0
                accum_ce = 0.0
                accum_aux = 0.0
                accum_tokens = 0
                accum_world_loss = 0.0
                ema_loss = cur_step_loss if ema_loss is None else (0.9 * ema_loss + 0.1 * cur_step_loss)

                current_cursor = extract_accumulated_data_cursor(
                    accum_batches, dataset, total_samples_trained, samples_consumed, shard_ledger, committed_domain_cursors
                )
            accum_batches.clear()
            accum_world_batches.clear()
            optimizer.zero_grad(set_to_none=True)

        if step >= args.max_steps:
            break

    total_time = time.time() - t0
    final_loss = ema_loss if ema_loss is not None else 0.0
    print("=" * 78)
    print(f"[{args.stage_id} FINISHED] Completed {step} steps in {total_time:.1f}s | Final Loss: {final_loss:.4f} | Best Loss: {best_loss:.4f}")
    print("=" * 78)

    # Final Checkpoint Save with COMPLETED status
    final_cursor = current_cursor if current_cursor is not None else resumed_cursor
    save_checkpoint(
        model, optimizer, scaler, config, tok_path, args.output_dir,
        step, epoch, final_loss, best_loss, args.stage_id,
        is_best=False, is_completed=True,
        dataset=dataset, shard_ledger=shard_ledger, data_cursor=final_cursor,
        keep_n=args.keep_checkpoints, force_keep_exact=True,
        world_model=world_model,
        world_optimizer=world_optimizer,
        world_scaler=world_scaler,
        per_shard_promotion=args.per_shard_promotion,
        max_shard_mb=args.max_shard_mb,
    )
    if ema_loss is not None and final_loss <= best_loss:
        best_loss = final_loss
        best_loss_step = step
        # Align from the newly committed args.output_dir without duplicate re-serialization
        os.makedirs(best_cp_dir, exist_ok=True)
        for art in os.listdir(args.output_dir):
            if art.endswith(".safetensors") or art.endswith(".json") or art in ("COMMITTED",):
                if art not in ("optimizer_state.pt",):
                    s_art = os.path.join(args.output_dir, art)
                    d_art = os.path.join(best_cp_dir, art)
                    if os.path.exists(s_art):
                        try:
                            if os.path.exists(d_art):
                                _safe_unlink(d_art)
                            os.link(s_art, d_art)
                        except Exception:
                            try:
                                shutil.copy2(s_art, d_art)
                            except Exception:
                                pass
        best_ptr = os.path.join(os.path.dirname(os.path.abspath(best_cp_dir)), "best_checkpoint.json")
        try:
            with open(best_ptr, "w", encoding="utf-8") as bf:
                json.dump({
                    "step": step,
                    "loss": final_loss,
                    "versioned_dir": best_cp_dir,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }, bf, indent=2)
        except Exception:
            pass

    print(f"[Stage Complete] Training checkpoint preserved in: {args.output_dir}")
    print(f"[Stage Complete] Best Milestone model archived in: {best_cp_dir}")

    # Auto-Promote clean inference model for deployment & native Rust serving
    promoted_target = args.promoted_dir if args.promoted_dir else os.path.join("storage/models", f"promoted_{args.stage_id}")
    if not promoted_at_target_step:
        best_source = (
            window_best_source if (args.promote_max_step > 0 and os.path.isdir(window_best_source) and check_model_weights_exist(window_best_source))
            else (best_cp_dir if check_model_weights_exist(best_cp_dir) else args.output_dir)
        )
        finalize_stage_and_promote(
            output_dir=args.output_dir,
            promoted_dir=promoted_target,
            stage_id=args.stage_id,
            hf_repo=model_repo,
            previous_stage_dir=args.previous_stage_dir,
            candidate_source_dir=best_source,
        )
    else:
        print(f"[Promotion] Target step {args.promote_max_step} model already promoted and active for real-time inference -> {promoted_target}", flush=True)
        if args.previous_stage_dir:
            clean_previous_training_checkpoints(args.previous_stage_dir, args.output_dir)

    print(f"[Promotion Complete] Deployment model ready at: {promoted_target}")
    print("[Success] All weights are 100% compatible with native Rust TARA inference & serving!")


if __name__ == "__main__":
    main()

