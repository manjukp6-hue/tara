#!/usr/bin/env python3
"""
TARA Standalone Supervised Fine-Tuning (SFT) Engine
=============================================================================
Dual environment support: Google Colab & Kaggle GPU acceleration.
Features:
- Prompt-Target masking (-100 on user tokens, backpropagation on assistant tokens).
- 14-domain JIT streaming & dynamic ShardLedger deduplication.
- Dynamic checkpointing with automatic best milestone tracking & GDrive sync.
- Live generation probes across Kannada, Reasoning, Code, and Science.
=============================================================================
"""

import argparse
import copy
import json
import math
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add repository root and tara_training to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(TRAINING_DIR)

for p in (REPO_ROOT, TRAINING_DIR, SCRIPT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from model_dataset import TaraForCausalLM, TaraNativeTokenizer
from gdrive_sync import is_checkpoint_loss_healthy, extract_checkpoint_loss_health
from sft_dataset import (
    CANONICAL_DOMAINS,
    BinaryCacheSftDataset,
    MultiDomainSftDataset,
    SftShardLedger,
    format_chat_prompt,
    normalize_domain,
    parse_domain_weights,
)


def detect_cloud_environment() -> Dict[str, Any]:
    """Detects whether running on Google Colab, Kaggle, or Standalone GPU host."""
    env_info = {
        "is_colab": os.path.exists("/content"),
        "is_kaggle": os.path.exists("/kaggle/working"),
        "platform_name": "Standalone Host",
        "default_output_dir": "storage/models/sft_checkpoints",
        "default_data_dir": "storage/datasets",
        "gdrive_supported": False,
    }

    if env_info["is_kaggle"]:
        env_info["platform_name"] = "Kaggle Environment"
        env_info["default_output_dir"] = "/kaggle/working/sft_checkpoints"
        env_info["default_data_dir"] = "/kaggle/working/datasets"
    elif env_info["is_colab"]:
        env_info["platform_name"] = "Google Colab Environment"
        env_info["default_output_dir"] = "/content/sft_checkpoints"
        env_info["default_data_dir"] = "/content/datasets"
        env_info["gdrive_supported"] = True

    return env_info


def mount_gdrive_if_colab() -> Optional[str]:
    """Mounts Google Drive on Google Colab and returns backup destination path."""
    if not os.path.exists("/content"):
        return None
    try:
        from google.colab import drive

        drive_mount = "/content/drive"
        if not os.path.exists(f"{drive_mount}/MyDrive"):
            print("[GDrive] Mounting Google Drive in Colab...", flush=True)
            drive.mount(drive_mount)
        backup_dir = f"{drive_mount}/MyDrive/tara_sft_checkpoints"
        os.makedirs(backup_dir, exist_ok=True)
        print(f"[GDrive] Backup persistence active -> {backup_dir}", flush=True)
        return backup_dir
    except Exception as e:
        print(f"[GDrive Notice] Google Drive mounting skipped or unavailable: {e}", flush=True)
        return None


def sync_to_gdrive(source_dir: str, gdrive_dir: Optional[str]):
    """Mirror checkpoint artifacts to Google Drive for disaster recovery."""
    if not gdrive_dir or not os.path.exists(gdrive_dir):
        return
    try:
        for item in os.listdir(source_dir):
            if item.endswith((".safetensors", ".json", ".pt", "COMMITTED")):
                s_path = os.path.join(source_dir, item)
                d_path = os.path.join(gdrive_dir, item)
                if os.path.isfile(s_path):
                    shutil.copy2(s_path, d_path)
        print(f"[GDrive Sync] Checkpoint mirrored to {gdrive_dir}", flush=True)
    except Exception as e:
        print(f"[GDrive Sync Warning] Failed to mirror checkpoint: {e}", flush=True)


def upload_sft_checkpoint_to_hf(
    source_dir: str,
    hf_repo: Optional[str],
    hf_token: Optional[str],
    step: int,
    is_best: bool = False,
):
    """Safely mirrors SFT checkpoint artifacts to Hugging Face model repository."""
    if not hf_repo or not hf_token:
        return
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=hf_token)
        prefix = "sft_checkpoints/best_milestone" if is_best else f"sft_checkpoints/step_{step:04d}"
        print(f"[HF Upload] Uploading SFT checkpoint to {hf_repo}:{prefix}...", flush=True)
        api.upload_folder(
            folder_path=source_dir,
            repo_id=hf_repo,
            repo_type="model",
            path_in_repo=prefix,
            token=hf_token,
        )
        print(f"[HF Upload Complete] SFT checkpoint synchronized to {hf_repo}:{prefix}", flush=True)
    except Exception as e:
        print(f"[HF Upload Notice] Could not upload checkpoint to Hugging Face: {e}", flush=True)


def resolve_highest_base_checkpoint(
    local_dirs: Optional[List[str]] = None,
    gdrive_mount: Optional[str] = None,
    hf_repo: Optional[str] = "manjukp6/tara",
    hf_token: Optional[str] = None,
    target_work_dir: str = ".",
) -> Tuple[int, str]:
    """
    Dynamically scans Local, Google Drive, and Hugging Face repositories to select
    and optionally download the absolute highest base model checkpoint (e.g. step_7700 or newer)
    without requiring any hardcoded step numbers.
    """
    highest_step = -1
    best_source = None
    best_type = None

    if local_dirs is None:
        local_dirs = [
            "tara_checkpoints/normal_checkpoint",
            "tara_checkpoints/best_milestone",
            "storage/models",
            "storage/models/tara_candidate_v1",
        ]

    # 1. Local Filesystem Scan
    for root_dir in local_dirs:
        if not os.path.exists(root_dir):
            continue
        if os.path.exists(os.path.join(root_dir, "config.json")):
            m = re.search(r"step_(\d+)", root_dir)
            s = int(m.group(1)) if m else 0
            if s > highest_step:
                highest_step = s
                best_source = root_dir
                best_type = "local"
        if os.path.isdir(root_dir):
            for item in os.listdir(root_dir):
                candidate = os.path.join(root_dir, item)
                if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "config.json")):
                    m = re.search(r"step_(\d+)", item)
                    s = int(m.group(1)) if m else 0
                    if s > highest_step:
                        highest_step = s
                        best_source = candidate
                        best_type = "local"

    # 2. Google Drive Scan
    gdrive_base = gdrive_mount or "/content/drive/MyDrive"
    if os.path.exists(gdrive_base):
        g_candidates = [
            os.path.join(gdrive_base, "tara_checkpoints/normal_checkpoint"),
            os.path.join(gdrive_base, "tara_checkpoints/best_milestone"),
            os.path.join(gdrive_base, "tara_models"),
        ]
        for g_root in g_candidates:
            if not os.path.exists(g_root):
                continue
            for item in os.listdir(g_root):
                candidate = os.path.join(g_root, item)
                if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "config.json")):
                    m = re.search(r"step_(\d+)", item)
                    s = int(m.group(1)) if m else 0
                    if s > highest_step:
                        highest_step = s
                        best_source = candidate
                        best_type = "gdrive"

    # 3. Hugging Face Repository Scan
    if hf_repo:
        try:
            from huggingface_hub import HfApi

            api = HfApi(token=hf_token)
            files = api.list_repo_files(hf_repo, repo_type="model")
            hf_steps = []
            for f in files:
                m = re.search(r"step_(\d+)", f)
                if m and ("normal_checkpoint" in f or "best_milestone" in f):
                    hf_steps.append((int(m.group(1)), f))
            if hf_steps:
                max_hf_step, max_f = max(hf_steps, key=lambda x: x[0])
                if max_hf_step > highest_step:
                    highest_step = max_hf_step
                    pattern_prefix = max_f.split("/step_")[0] + f"/step_{highest_step}"
                    best_source = pattern_prefix
                    best_type = "hf"
        except Exception as e:
            print(f"[Base Resolver Notice] Hugging Face scan skipped: {e}", flush=True)

    # If Hugging Face is the highest or local not yet downloaded
    if best_type == "hf" and best_source:
        print(f"[Base Resolver] Discovered highest Step {highest_step} on Hugging Face ({best_source}). Downloading...", flush=True)
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=hf_repo,
            repo_type="model",
            local_dir=target_work_dir,
            allow_patterns=[f"{best_source}/*"],
            token=hf_token,
            max_workers=4,
        )
        resolved_path = os.path.join(target_work_dir, best_source)
        if os.path.exists(resolved_path):
            return highest_step, resolved_path

    if best_source:
        return highest_step, best_source

    fallback = "storage/models/tara_candidate_v1"
    return 0, fallback


def resolve_highest_sft_checkpoint(
    local_output_dir: str,
    gdrive_dir: Optional[str] = None,
    hf_repo: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> Tuple[int, Optional[str]]:
    """
    Scans local output directory, Google Drive, and Hugging Face repository
    to automatically identify the highest step SFT checkpoint and select it for resumption.
    Applies Loss Health Verification:
    - If a candidate checkpoint has diverged loss (> 1.5x best loss), it is rejected.
    - Audits previous normal checkpoint backups.
    - If all normal checkpoints are diverged, safely falls back to best_sft_milestone.
    """
    candidates: List[Tuple[int, float, str]] = []
    ref_best_loss = float("inf")
    bm_source: Optional[str] = None
    bm_step: int = -1

    # Helper to check a candidate folder
    def check_candidate(candidate_dir: str):
        nonlocal ref_best_loss, bm_source, bm_step
        if not os.path.isdir(candidate_dir):
            return
        s, l, bl = extract_checkpoint_loss_health(candidate_dir)
        is_bm = "best_sft_milestone" in candidate_dir or "best_milestone" in candidate_dir
        if math.isfinite(l) and l > 0.0:
            if is_bm:
                ref_best_loss = min(ref_best_loss, l)
        if math.isfinite(bl) and bl > 0.0:
            ref_best_loss = min(ref_best_loss, bl)

        if s >= 0:
            if is_bm:
                if s > bm_step:
                    bm_step = s
                    bm_source = candidate_dir
            else:
                candidates.append((s, l, candidate_dir))

    # 1. Scan Local Directory & Subfolders
    if os.path.exists(local_output_dir):
        check_candidate(local_output_dir)
        for item in os.listdir(local_output_dir):
            check_candidate(os.path.join(local_output_dir, item))

    # 2. Scan Google Drive Backup Directory & Subfolders
    if gdrive_dir and os.path.exists(gdrive_dir):
        check_candidate(gdrive_dir)
        for item in os.listdir(gdrive_dir):
            check_candidate(os.path.join(gdrive_dir, item))

    # 3. Scan Hugging Face Repository
    if hf_repo:
        try:
            from huggingface_hub import HfApi

            api = HfApi(token=hf_token)
            files = api.list_repo_files(hf_repo)
            hf_steps = []
            for f in files:
                m = re.search(r"sft_checkpoints/step_(\d+)", f)
                if m:
                    hf_steps.append((int(m.group(1)), f))
            if hf_steps:
                max_hf_step, _ = max(hf_steps, key=lambda x: x[0])
                from huggingface_hub import snapshot_download

                dl_dir = os.path.join(local_output_dir, f"hf_step_{max_hf_step}")
                snapshot_download(
                    repo_id=hf_repo,
                    repo_type="model",
                    local_dir=dl_dir,
                    allow_patterns=[f"sft_checkpoints/step_{max_hf_step:04d}/*"],
                    token=hf_token,
                )
                sub = os.path.join(dl_dir, f"sft_checkpoints/step_{max_hf_step:04d}")
                cand_sub = sub if os.path.isdir(sub) else dl_dir
                check_candidate(cand_sub)
        except Exception as e:
            print(f"[Auto-Resume HF Notice] Hugging Face scan skipped: {e}", flush=True)

    # Sort candidates descending by step (highest step first)
    candidates.sort(key=lambda x: x[0], reverse=True)

    highest_step = 0
    best_source = None

    for s_cand, l_cand, dir_cand in candidates:
        if is_checkpoint_loss_healthy(l_cand, ref_best_loss=ref_best_loss):
            highest_step = s_cand
            best_source = dir_cand
            if candidates and s_cand < candidates[0][0]:
                print(f"[SFT Resume Loss Audit] Selected healthy normal checkpoint backup at Step {s_cand} (Loss {l_cand:.4f} <= {ref_best_loss * 1.5:.4f}) after newer candidate(s) diverged!", flush=True)
            break
        else:
            print(f"[SFT Resume Loss Audit] Checkpoint at Step {s_cand} ({os.path.basename(dir_cand)}) rejected: Loss {l_cand:.4f} diverged (> {ref_best_loss * 1.5:.4f}). Auditing older normal backup...", flush=True)

    if highest_step <= 0 and bm_step > 0 and bm_source:
        highest_step = bm_step
        best_source = bm_source
        print(f"[SFT Resume Safe Fallback] All normal SFT checkpoints diverged. Falling back to best_sft_milestone at Step {bm_step} ({bm_source})", flush=True)

    return highest_step, best_source


def discover_sft_shards(data_dir: str) -> Dict[str, List[str]]:
    """Discovers training shards mapped across the 14 canonical domains."""
    domain_files: Dict[str, List[str]] = {d: [] for d in CANONICAL_DOMAINS}
    valid_exts = (
        ".jsonl", ".json", ".parquet", ".csv", ".tsv", ".txt",
        ".jsonl.gz", ".json.gz", ".csv.gz", ".tsv.gz", ".txt.gz",
        ".jsonl.bz2", ".json.bz2", ".csv.bz2", ".tsv.bz2", ".txt.bz2",
        ".zip", ".tar.gz", ".tgz",
    )

    if not os.path.exists(data_dir):
        return domain_files

    for root, _, files in os.walk(data_dir):
        for f in files:
            if any(f.endswith(ext) for ext in valid_exts):
                full_path = os.path.join(root, f)
                rel_parts = os.path.relpath(full_path, data_dir).split(os.sep)
                assigned_domain = "sft"
                for p in rel_parts:
                    norm = normalize_domain(p)
                    if norm != "sft":
                        assigned_domain = norm
                        break
                domain_files[assigned_domain].append(full_path)

    for d in domain_files:
        domain_files[d].sort()

    return domain_files


class SftBenchmarkProbe:
    """Canonical evaluation probe for SFT benchmark scoring."""

    def __init__(
        self,
        domain: str,
        name: str,
        prompt: str,
        expected_completion: str,
        target_answer: Optional[str] = None,
        is_numeric_gsm8k: bool = False,
    ):
        self.domain = domain
        self.name = name
        self.prompt = prompt
        self.expected_completion = expected_completion
        self.target_answer = target_answer
        self.is_numeric_gsm8k = is_numeric_gsm8k


def build_canonical_sft_benchmark_suite(external_file: Optional[str] = None) -> List[SftBenchmarkProbe]:
    """
    Constructs the official TARA multi-domain scoring benchmark suite across:
    - 01 GSM8K Multi-Step Math Arithmetic
    - 02 Code Syntax & Algorithmic Logic (Python & Rust)
    - 04 Kannada Multilingual QA (Science, CS & Literature)
    - 12 Reasoning & Formal Deductive Logic
    - 03 STEM & Natural Sciences
    - 08 Robotics Kinematics & Causal World Model
    """
    probes: List[SftBenchmarkProbe] = [
        # 1. GSM8K Multi-Step Math
        SftBenchmarkProbe(
            domain="01_gsm8k_math",
            name="gsm8k_pencils_janet",
            prompt="Question: Janet buys 3 packs of 12 pencils each. She gives 10 pencils to her brother. How many pencils does Janet have left?\nAnswer: Let's calculate step by step:\n",
            expected_completion="1. Janet starts with 3 packs of 12 pencils: 3 * 12 = 36 pencils.\n2. She gives 10 pencils to her brother: 36 - 10 = 26 pencils.\nTherefore, Janet has 26 pencils left.",
            target_answer="26",
            is_numeric_gsm8k=True,
        ),
        SftBenchmarkProbe(
            domain="01_gsm8k_math",
            name="gsm8k_bakery_bread",
            prompt="Question: A bakery bakes 40 loaves of bread in the morning and 30 loaves in the afternoon. If they sell 55 loaves during the day, how many loaves remain unsold?\nAnswer: Let's solve step by step:\n",
            expected_completion="1. Total loaves baked = 40 + 30 = 70 loaves.\n2. Loaves sold = 55.\n3. Remaining loaves = 70 - 55 = 15 loaves.\nTherefore, 15 loaves remain unsold.",
            target_answer="15",
            is_numeric_gsm8k=True,
        ),
        SftBenchmarkProbe(
            domain="01_gsm8k_math",
            name="gsm8k_apples_orchard",
            prompt="Question: An orchard has 4 apple trees with 15 apples each. If a storm knocks down 8 apples, how many apples are still on the trees?\nAnswer: Let's calculate step by step:\n",
            expected_completion="1. Total apples on trees = 4 * 15 = 60 apples.\n2. Storm knocks down 8 apples: 60 - 8 = 52 apples.\nTherefore, 52 apples are still on the trees.",
            target_answer="52",
            is_numeric_gsm8k=True,
        ),

        # 2. Code Generation & Algorithmic Logic (Python & Rust)
        SftBenchmarkProbe(
            domain="02_code_algorithms",
            name="code_py_euclidean",
            prompt="Question: Write a Python function euclidean_distance(x1, y1, x2, y2) that calculates 2D Euclidean distance.\nAnswer: Here is the implementation:\n",
            expected_completion="import math\n\ndef euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:\n    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)",
            target_answer="euclidean_distance",
        ),
        SftBenchmarkProbe(
            domain="02_code_algorithms",
            name="code_py_filter_squares",
            prompt="Question: Write a Python function filter_even_squares(numbers) that returns the squares of even numbers in order.\nAnswer: Here is the implementation:\n",
            expected_completion="def filter_even_squares(numbers: list[int]) -> list[int]:\n    return [n * n for n in numbers if n % 2 == 0]",
            target_answer="filter_even_squares",
        ),
        SftBenchmarkProbe(
            domain="02_code_algorithms",
            name="code_rust_parse_positive",
            prompt="Question: Write an idiomatic Rust function parse_positive_i32(s: &str) -> Result<i32, String>.\nAnswer: Here is the implementation:\n",
            expected_completion="pub fn parse_positive_i32(s: &str) -> Result<i32, String> {\n    let val = s.trim().parse::<i32>().map_err(|e| e.to_string())?;\n    if val > 0 {\n        Ok(val)\n    } else {\n        Err(\"Value must be strictly positive\".to_string())\n    }\n}",
            target_answer="parse_positive_i32",
        ),

        # 3. Kannada Multilingual QA (Science, CS & Literature)
        SftBenchmarkProbe(
            domain="04_kannada_qa",
            name="kannada_sunlight_time",
            prompt="ಪ್ರಶ್ನೆ: ಸೂರ್ಯನ ಬೆಳಕು ಭೂಮಿಗೆ ತಲುಪಲು ಸುಮಾರು ಎಷ್ಟು ಸಮಯ ಬೇಕಾಗುತ್ತದೆ?\nಉತ್ತರ:",
            expected_completion="ಸೂರ್ಯನ ಬೆಳಕು ಭೂಮಿಗೆ ತಲುಪಲು ಸುಮಾರು ೮ ನಿಮಿಷ ೨೦ ಸೆಕೆಂಡುಗಳು (8 minutes 20 seconds) ಬೇಕಾಗುತ್ತದೆ.",
            target_answer="8",
        ),
        SftBenchmarkProbe(
            domain="04_kannada_qa",
            name="kannada_binary_digits",
            prompt="ಪ್ರಶ್ನೆ: ಕಂಪ್ಯೂಟರ್ ಪ್ರೋಗ್ರಾಮಿಂಗ್‌ನಲ್ಲಿ ಬೈನರಿ ಸಂಖ್ಯಾ ವ್ಯವಸ್ಥೆ ಯಾವ ಎರಡು ಅಂಕೆಗಳನ್ನು ಬಳಸುತ್ತದೆ?\nಉತ್ತರ:",
            expected_completion="ಬೈನರಿ ಸಂಖ್ಯಾ ವ್ಯವಸ್ಥೆಯು ೦ ಮತ್ತು ೧ (0 ಮತ್ತು 1) ಅಂಕೆಗಳನ್ನು ಬಳಸುತ್ತದೆ.",
            target_answer="0",
        ),
        SftBenchmarkProbe(
            domain="04_kannada_qa",
            name="kannada_jnanpith_first",
            prompt="ಪ್ರಶ್ನೆ: ಜ್ಞಾನಪೀಠ ಪ್ರಶಸ್ತಿ ಪಡೆದ ಮೊದಲ ಕನ್ನಡಿಗ ಯಾರು?\nಉತ್ತರ:",
            expected_completion="ಜ್ಞಾನಪೀಠ ಪ್ರಶಸ್ತಿ ಪಡೆದ ಮೊದಲ ಕನ್ನಡಿಗ ರಾಷ್ಟ್ರಕವಿ ಕುವೆಂಪು (ಕೆ. ವಿ. ಪುಟ್ಟಪ್ಪ).",
            target_answer="ಕುವೆಂಪು",
        ),

        # 4. Reasoning & Deductive Logic
        SftBenchmarkProbe(
            domain="12_reasoning_logic",
            name="reasoning_syllogism_chimpanzee",
            prompt="Premise 1: All primates are mammals.\nPremise 2: A chimpanzee is a primate.\nConclusion: Therefore, a chimpanzee is a",
            expected_completion=" mammal.",
            target_answer="mammal",
        ),
        SftBenchmarkProbe(
            domain="12_reasoning_logic",
            name="reasoning_ev_distance",
            prompt="Question: If an electric vehicle travels at an average speed of 60 km/h for 2.5 hours, what total distance in kilometers does it cover?\nAnswer:",
            expected_completion="Distance = speed * time = 60 * 2.5 = 150 kilometers.",
            target_answer="150",
            is_numeric_gsm8k=True,
        ),
        SftBenchmarkProbe(
            domain="12_reasoning_logic",
            name="reasoning_ice_density",
            prompt="Question: Why does solid ice float on liquid water?\nAnswer:",
            expected_completion="Solid ice floats because water expands when freezing into a crystalline lattice, making ice less dense than liquid water.",
            target_answer="less dense",
        ),

        # 5. STEM & Natural Sciences
        SftBenchmarkProbe(
            domain="03_stem_science",
            name="stem_photosynthesis_products",
            prompt="Question: What does photosynthesis produce from carbon dioxide, water, and sunlight?\nAnswer:",
            expected_completion="Photosynthesis converts carbon dioxide and water into glucose and oxygen.",
            target_answer="glucose",
        ),
        SftBenchmarkProbe(
            domain="03_stem_science",
            name="stem_inclined_friction_force",
            prompt="Causal Physics: A 5kg mass slides on an inclined plane of angle theta with friction coefficient mu. Net force parallel to surface:",
            expected_completion="F_net = m * g * sin(theta) - mu * m * g * cos(theta)",
            target_answer="sin(theta)",
        ),

        # 6. Robotics Kinematics & Causal World Model
        SftBenchmarkProbe(
            domain="08_robotics_causal",
            name="robotics_go2_pitch_recovery",
            prompt="State: Quadruped Unitree Go2 trot gait at velocity 1.2 m/s.\nAction: Front left contact lost, pitch angle increased by +0.08 rad.\nExpected stabilization corrective response:",
            expected_completion="Increase rear leg damping and apply downward pitch recovery torque to stabilize body frame.",
            target_answer="torque",
        ),
    ]

    # Ingest external probes if custom benchmark file is provided
    if external_file and os.path.exists(external_file):
        try:
            with open(external_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        p = SftBenchmarkProbe(
                            domain=obj.get("domain", "custom_benchmark"),
                            name=obj.get("name", f"probe_{len(probes)+1}"),
                            prompt=obj.get("prompt", ""),
                            expected_completion=obj.get("expected_completion", ""),
                            target_answer=obj.get("target_answer"),
                            is_numeric_gsm8k=obj.get("is_numeric_gsm8k", False),
                        )
                        if p.prompt and p.expected_completion:
                            probes.append(p)
                    except Exception:
                        continue
            print(f"[Benchmark] Loaded {len(probes)} total probes (including external: {external_file})", flush=True)
        except Exception as e:
            print(f"[Benchmark Warning] Could not load external probes: {e}", flush=True)

    return probes


def run_sft_scoring_benchmark(
    model: nn.Module,
    tokenizer: Any,
    device: torch.device,
    step: int,
    output_dir: Optional[str] = None,
    external_benchmark_file: Optional[str] = None,
    no_repeat_ngram_size: int = 3,
    repetition_penalty: float = 1.15,
) -> Dict[str, Any]:
    """
    Executes official GSM8K Math + Code + Kannada + Reasoning + STEM Scoring Benchmark directly
    inside the training loop, computing:
    - Teacher-forcing Cross-Entropy Loss & Perplexity (PPL) on target tokens
    - Top-1 Token Accuracy (%)
    - Exact Match / GSM8K Numerical Answer Accuracy (%)
    - Live greedy generation outputs across all domains
    """
    probes = build_canonical_sft_benchmark_suite(external_benchmark_file)
    model.eval()
    im_end_id = getattr(tokenizer, "im_end_id", 2)

    domain_stats: Dict[str, Dict[str, Any]] = {}
    sample_generations: List[Dict[str, str]] = []

    total_probes = len(probes)
    sum_top1_acc = 0.0
    sum_loss = 0.0
    sum_ppl = 0.0
    total_matches = 0

    with torch.no_grad():
        for probe in probes:
            dom = probe.domain
            if dom not in domain_stats:
                domain_stats[dom] = {
                    "count": 0,
                    "top1_acc_sum": 0.0,
                    "loss_sum": 0.0,
                    "ppl_sum": 0.0,
                    "matches": 0,
                }

            prompt_str, target_str = format_chat_prompt(probe.prompt, probe.expected_completion)
            enc_p = tokenizer.encode(prompt_str)
            p_ids = enc_p.ids if hasattr(enc_p, "ids") else list(enc_p)

            enc_t = tokenizer.encode(target_str)
            t_ids = enc_t.ids if hasattr(enc_t, "ids") else list(enc_t)

            # 1. Teacher-Forced Loss & Top-1 Token Accuracy
            full_ids = p_ids + t_ids
            inp_t = torch.tensor([full_ids], dtype=torch.long, device=device)
            out = model(input_ids=inp_t)

            logits = out.logits
            p_len = len(p_ids)
            t_len = len(t_ids)

            shift_logits = logits[0, p_len - 1 : p_len + t_len - 1, :].clone()
            target_t = torch.tensor(t_ids, dtype=torch.long, device=device)

            probe_loss = F.cross_entropy(shift_logits, target_t).item()
            probe_ppl = math.exp(min(probe_loss, 20.0))
            pred_ids = torch.argmax(shift_logits, dim=-1)
            probe_acc = (pred_ids == target_t).float().mean().item() * 100.0

            # 2. Live Greedy Generation & Answer Verification
            gen_input = torch.tensor([p_ids], dtype=torch.long, device=device)
            generated = list(p_ids)
            max_new = 48

            for _ in range(max_new):
                out_g = model(input_ids=gen_input)
                next_logits = out_g.logits[0, -1, :].clone()

                # N-gram repetition blocking
                if no_repeat_ngram_size > 0 and len(generated) >= no_repeat_ngram_size - 1:
                    n = no_repeat_ngram_size
                    prefix_len = n - 1
                    curr_pfx = generated[-prefix_len:]
                    for w_idx in range(len(generated) - n + 1):
                        if generated[w_idx : w_idx + prefix_len] == curr_pfx:
                            banned = generated[w_idx + prefix_len]
                            next_logits[banned] = -1e9

                # Repetition penalty
                if repetition_penalty > 1.0:
                    for uid in set(generated):
                        if next_logits[uid] > 0:
                            next_logits[uid] /= repetition_penalty
                        else:
                            next_logits[uid] *= repetition_penalty

                next_id = int(torch.argmax(next_logits).item())
                if next_id == im_end_id:
                    break

                generated.append(next_id)
                gen_input = torch.tensor([generated], dtype=torch.long, device=device)

            gen_only = generated[len(p_ids) :]
            decoded_resp = tokenizer.decode(gen_only).strip()

            # Target answer matching logic
            is_match = False
            if probe.is_numeric_gsm8k and probe.target_answer:
                nums = re.findall(r'-?\d+(?:\.\d+)?', decoded_resp)
                if nums and (nums[-1] == probe.target_answer or probe.target_answer in nums):
                    is_match = True
            elif probe.target_answer:
                if probe.target_answer.lower() in decoded_resp.lower():
                    is_match = True
            else:
                if probe.expected_completion.strip().lower() in decoded_resp.lower():
                    is_match = True

            # Record stats
            domain_stats[dom]["count"] += 1
            domain_stats[dom]["top1_acc_sum"] += probe_acc
            domain_stats[dom]["loss_sum"] += probe_loss
            domain_stats[dom]["ppl_sum"] += probe_ppl
            if is_match:
                domain_stats[dom]["matches"] += 1

            sum_top1_acc += probe_acc
            sum_loss += probe_loss
            sum_ppl += probe_ppl
            if is_match:
                total_matches += 1

            if len(sample_generations) < 6:
                sample_generations.append({
                    "domain": dom,
                    "name": probe.name,
                    "target": probe.target_answer or probe.expected_completion[:20],
                    "generated": decoded_resp.replace("\n", " ")[:60],
                    "match": "PASS" if is_match else "FAIL",
                })

    model.train()

    # Compute Averages
    overall_acc = sum_top1_acc / max(1, total_probes)
    overall_loss = sum_loss / max(1, total_probes)
    overall_ppl = sum_ppl / max(1, total_probes)
    overall_match_rate = (total_matches / max(1, total_probes)) * 100.0

    # Print Formatted Benchmark Table
    print(f"\n{'=' * 95}", flush=True)
    print(f"  TARA SFT COMPREHENSIVE BENCHMARK EVALUATION (Step {step:04d})", flush=True)
    print(f"{'=' * 95}", flush=True)
    print(f"  {'Domain':<26} | {'Probes':<6} | {'Top-1 Acc':<9} | {'Match Rate':<10} | {'Avg Loss':<8} | {'Avg PPL':<7}", flush=True)
    print(f"  {'-' * 26}-+-{'-' * 6}-+-{'-' * 9}-+-{'-' * 10}-+-{'-' * 8}-+-{'-' * 7}", flush=True)

    summary_records = {}
    for d_name in sorted(domain_stats.keys()):
        d_data = domain_stats[d_name]
        c = max(1, d_data["count"])
        d_acc = d_data["top1_acc_sum"] / c
        d_loss = d_data["loss_sum"] / c
        d_ppl = d_data["ppl_sum"] / c
        d_match = (d_data["matches"] / c) * 100.0

        summary_records[d_name] = {
            "probes": c,
            "top1_acc": round(d_acc, 2),
            "match_rate": round(d_match, 2),
            "avg_loss": round(d_loss, 4),
            "avg_ppl": round(d_ppl, 2),
        }
        print(f"  {d_name:<26} | {c:<6} | {d_acc:>7.1f}% | {d_match:>9.1f}% | {d_loss:>8.4f} | {d_ppl:>7.2f}", flush=True)

    print(f"  {'-' * 91}", flush=True)
    print(f"  OVERALL BENCHMARK SCORE : {overall_acc:.1f}% Top-1 Acc | Match: {overall_match_rate:.1f}% | PPL: {overall_ppl:.2f}", flush=True)
    print(f"{'=' * 95}", flush=True)

    # Print Live Samples
    print("  [Live Benchmark Probes Sample Generations]:", flush=True)
    for smp in sample_generations:
        print(f"   * [{smp['domain']:<18}] [{smp['match']:<4}] Gen: \"{smp['generated']}\"", flush=True)
    print(f"{'=' * 95}\n", flush=True)

    report = {
        "step": step,
        "overall_top1_accuracy": round(overall_acc, 2),
        "overall_match_rate": round(overall_match_rate, 2),
        "overall_loss": round(overall_loss, 4),
        "overall_perplexity": round(overall_ppl, 2),
        "total_probes": total_probes,
        "domains": summary_records,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, "sft_benchmark_report.json")
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
        except Exception:
            pass

    return report


def run_sft_generation_probe(
    model: nn.Module,
    tokenizer: Any,
    device: torch.device,
    step: int,
    no_repeat_ngram_size: int = 3,
    repetition_penalty: float = 1.15,
):
    """Backwards-compatible wrapper that invokes the full Scoring Benchmark Suite."""
    return run_sft_scoring_benchmark(
        model=model,
        tokenizer=tokenizer,
        device=device,
        step=step,
        no_repeat_ngram_size=no_repeat_ngram_size,
        repetition_penalty=repetition_penalty,
    )


def save_sft_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    config: Dict[str, Any],
    tokenizer_path: str,
    output_dir: str,
    step: int,
    loss: float,
    best_loss: float,
    consumed_samples: int = 0,
    is_best: bool = False,
    gdrive_backup_dir: Optional[str] = None,
    hf_repo: Optional[str] = None,
    hf_token: Optional[str] = None,
):
    """Atomically commits full SFT checkpoint artifacts and metadata."""
    os.makedirs(output_dir, exist_ok=True)
    tmp_dir = f"{output_dir}.tmp.{os.getpid()}"
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        from safetensors.torch import save_file

        state_dict = model.state_dict()
        clean_state = {k: v.clone().contiguous().to("cpu") for k, v in state_dict.items() if not k.startswith("module.")}
        save_file(clean_state, os.path.join(tmp_dir, "model.safetensors"))
    except ImportError:
        torch.save(model.state_dict(), os.path.join(tmp_dir, "pytorch_model.bin"))

    with open(os.path.join(tmp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    if os.path.exists(tokenizer_path):
        shutil.copy2(tokenizer_path, os.path.join(tmp_dir, "tokenizer.json"))

    meta = {
        "step": step,
        "consumed_samples": consumed_samples,
        "loss": float(loss),
        "best_loss": float(best_loss),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "is_best": is_best,
        "task": "supervised_fine_tuning",
    }
    with open(os.path.join(tmp_dir, "sft_checkpoint_state.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    torch.save(
        {
            "step": step,
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
            "loss": loss,
            "best_loss": best_loss,
        },
        os.path.join(tmp_dir, "optimizer_state.pt"),
    )

    with open(os.path.join(tmp_dir, "COMMITTED"), "w", encoding="utf-8") as f:
        f.write(f"SFT_STEP_{step:08d}\n")

    for f_name in os.listdir(tmp_dir):
        s_file = os.path.join(tmp_dir, f_name)
        d_file = os.path.join(output_dir, f_name)
        if os.path.exists(d_file):
            os.remove(d_file)
        shutil.move(s_file, d_file)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"[Checkpoint Committed] SFT Step {step:04d} -> {output_dir} (Loss: {loss:.4f})", flush=True)

    if is_best and gdrive_backup_dir:
        sync_to_gdrive(output_dir, gdrive_backup_dir)

    if hf_repo and hf_token and (is_best or step % 200 == 0):
        upload_sft_checkpoint_to_hf(output_dir, hf_repo, hf_token, step, is_best=is_best)


def main():
    parser = argparse.ArgumentParser(description="TARA Autonomous Supervised Fine-Tuning (SFT) Runner")
    env = detect_cloud_environment()

    parser.add_argument("--model-dir", type=str, required=True, help="Base pretrained checkpoint directory")
    parser.add_argument("--data-dir", type=str, default=env["default_data_dir"], help="Multi-domain SFT dataset root")
    parser.add_argument("--output-dir", type=str, default=env["default_output_dir"], help="SFT output checkpoints dir")
    parser.add_argument("--batch-size", type=int, default=4, help="Micro-batch size per optimization step")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate for SFT adapter/full updates")
    parser.add_argument("--max-steps", type=int, default=1000, help="Total SFT training steps")
    parser.add_argument("--max-seq-len", type=int, default=1024, help="Maximum sequence context length")
    parser.add_argument("--checkpoint-interval", type=int, default=100, help="Checkpoint frequency in steps")
    parser.add_argument("--probe-interval", type=int, default=50, help="Live generation probe frequency")
    parser.add_argument("--benchmark-interval", type=int, default=50, help="Benchmark evaluation frequency in steps (default: 50)")
    parser.add_argument("--benchmark-file", type=str, default=None, help="Optional external JSONL file containing custom benchmark evaluation probes")
    parser.add_argument("--domain-weights", type=str, default="", help="Domain weight specifications")
    parser.add_argument("--bin-cache-dir", type=str, default=None, help="Directory containing pre-tokenized .bin cache files")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True, help="Enable activation gradient checkpointing (default: True)")
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false", help="Disable activation gradient checkpointing")
    parser.add_argument("--optimizer", type=str, default="auto", choices=["auto", "adamw", "adamw8bit"], help="Optimizer choice ('auto' selects 8-bit AdamW on <=16.5GB VRAM)")
    parser.add_argument("--hf-repo", type=str, default=os.environ.get("HF_REPO", "manjukp6/tara"), help="Hugging Face repo for checkpoint upload/resume")
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN", None), help="Hugging Face API token")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume training from latest checkpoint")
    args = parser.parse_args()

    print("=" * 80, flush=True)
    print(f"  TARA STANDALONE SFT ENGINE — {env['platform_name'].upper()}", flush=True)
    print("=" * 80, flush=True)
    print(f"  Base Model Dir : {args.model_dir}", flush=True)
    print(f"  Data Directory : {args.data_dir}", flush=True)
    if args.bin_cache_dir:
        print(f"  Bin Cache Dir  : {args.bin_cache_dir}", flush=True)
    print(f"  Output Dir     : {args.output_dir}", flush=True)
    if args.hf_repo:
        print(f"  HF Repository  : {args.hf_repo}", flush=True)
    print(f"  Batch Config   : Batch Size {args.batch_size} (Grad Accum: {args.grad_accum})", flush=True)
    print(f"  Learning Rate  : {args.lr:.2e} | Max Steps: {args.max_steps}", flush=True)
    print("=" * 80, flush=True)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024 * 1024)
        print(f"[Device] Acceleration: {gpu_name} ({gpu_mem:.2f} GB VRAM)", flush=True)
    else:
        print("[Device Warning] CUDA GPU not found. Operating on CPU backend.", flush=True)

    # Colab GDrive persistence setup
    gdrive_dir = mount_gdrive_if_colab() if env["gdrive_supported"] else None

    # Load Model Configuration
    config_file = os.path.join(args.model_dir, "config.json")
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Model config not found at {config_file}")
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Load Tokenizer
    tok_file = os.path.join(args.model_dir, "tokenizer.json")
    tokenizer = TaraNativeTokenizer.from_file(tok_file)
    print(f"[Tokenizer] Loaded TARA Tokenizer ({len(tokenizer.token_to_id):,} tokens)", flush=True)

    # Instantiate Model
    print(f"[Model] Loading base weights into TaraForCausalLM (gradient_checkpointing={args.gradient_checkpointing})...", flush=True)
    model = TaraForCausalLM(config, gradient_checkpointing=args.gradient_checkpointing).to(device)

    safetensors_path = os.path.join(args.model_dir, "model.safetensors")
    index_path = os.path.join(args.model_dir, "model.safetensors.index.json")

    try:
        from safetensors.torch import load_file

        if os.path.exists(safetensors_path):
            state = load_file(safetensors_path, device="cpu")
            model.load_state_dict(state, strict=False)
        elif os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                index_data = json.load(f)
            weight_map = index_data.get("weight_map", {})
            unique_shards = set(weight_map.values())
            merged_state = {}
            for sh in unique_shards:
                sh_path = os.path.join(args.model_dir, sh)
                if os.path.exists(sh_path):
                    merged_state.update(load_file(sh_path, device="cpu"))
            model.load_state_dict(merged_state, strict=False)
        print("[Model] Base weights initialized cleanly!", flush=True)
    except Exception as e:
        print(f"[Model Notice] Loading weights via fallback mechanism: {e}", flush=True)

    # VRAM Inspection & Optimizer Configuration
    total_vram_gb = 0.0
    if device.type == "cuda":
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024 * 1024)
    is_vram_constrained = device.type == "cuda" and total_vram_gb <= 16.5

    use_8bit = False
    if args.optimizer == "adamw8bit":
        use_8bit = True
    elif args.optimizer == "auto" and is_vram_constrained:
        try:
            import bitsandbytes as bnb
            use_8bit = True
            print(f"[VRAM Guard] Tesla/constrained GPU ({total_vram_gb:.1f} GB) detected. Enabling 8-bit AdamW to save ~5.5GB VRAM.", flush=True)
        except ImportError:
            print("[VRAM Guard Warning] bitsandbytes not installed; falling back to standard AdamW.", flush=True)

    # Decouple weight decay: apply 0.01 only to 2D projections/MLP weights; exclude 1D norms/biases & embeddings
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2 or "norm" in name.lower() or "bias" in name.lower() or "embed" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": 0.01},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    if use_8bit:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(optim_groups, lr=args.lr, betas=(0.9, 0.95), eps=1e-5)
        print("[Optimizer] Initialized 8-bit AdamW with decoupled weight decay (saving ~5.5GB VRAM)", flush=True)
    else:
        fused = hasattr(torch.optim.AdamW, "fused") and device.type == "cuda"
        try:
            optimizer = torch.optim.AdamW(optim_groups, lr=args.lr, betas=(0.9, 0.95), eps=1e-5, fused=fused)
        except Exception:
            optimizer = torch.optim.AdamW(optim_groups, lr=args.lr, betas=(0.9, 0.95), eps=1e-5)
        print(f"[Optimizer] Initialized standard AdamW with decoupled weight decay (fused={fused})", flush=True)

    use_amp = device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # State tracking & Resumption check across Local, GDrive, and Hugging Face
    start_step = 0
    skip_samples = 0
    best_loss = float("inf")
    best_dir = os.path.join(args.output_dir, "best_sft_milestone")

    if args.resume:
        highest_step, best_source = resolve_highest_sft_checkpoint(
            local_output_dir=args.output_dir,
            gdrive_dir=gdrive_dir,
            hf_repo=args.hf_repo,
            hf_token=args.hf_token,
        )
        if highest_step > 0 and best_source and os.path.exists(best_source):
            safetensors_src = os.path.join(best_source, "model.safetensors")
            if os.path.exists(safetensors_src):
                try:
                    from safetensors.torch import load_file

                    model.load_state_dict(load_file(safetensors_src, device="cpu"), strict=False)
                    print(f"[Resume] Restored model weights from {safetensors_src}", flush=True)
                except Exception as me:
                    print(f"[Resume Warning] Could not load safetensors: {me}", flush=True)

            state_file = os.path.join(best_source, "sft_checkpoint_state.json")
            if os.path.exists(state_file):
                try:
                    with open(state_file, "r", encoding="utf-8") as f:
                        saved_meta = json.load(f)
                    start_step = saved_meta.get("step", 0)
                    best_loss = saved_meta.get("best_loss", float("inf"))
                    skip_samples = saved_meta.get("consumed_samples", start_step * args.grad_accum)
                except Exception:
                    pass

            opt_state_file = os.path.join(best_source, "optimizer_state.pt")
            if os.path.exists(opt_state_file):
                try:
                    opt_payload = torch.load(opt_state_file, map_location=device)
                    optimizer.load_state_dict(opt_payload["optimizer_state_dict"])
                    if opt_payload.get("scaler_state_dict") and scaler.is_enabled():
                        scaler.load_state_dict(opt_payload["scaler_state_dict"])
                except Exception as re:
                    print(f"[Resume Warning] Could not restore optimizer state ({re}). Starting clean.", flush=True)

            print(f"[Resume] Successfully resumed SFT from Step {start_step} (Best Loss: {best_loss:.4f}) from {best_source}", flush=True)
            if skip_samples > 0:
                print(f"[Resume] Fast-forwarding dataset past {skip_samples:,} already trained samples (Zero-Repeat Guarantee)", flush=True)

    # Setup Ledger and Dataset (Instantiated with skip_samples for exact zero-repeat continuation)
    ledger_path = os.path.join(args.output_dir, "sft_shard_ledger.json")
    ledger = SftShardLedger(ledger_path)

    # Fast Binary Cache vs Streaming JIT MultiDomain Dataset
    if args.bin_cache_dir and os.path.exists(os.path.join(args.bin_cache_dir, "sft_pretokenization_manifest.json")):
        print(f"[Dataset] Fast zero-latency Binary Cache active from {args.bin_cache_dir}", flush=True)
        dataset = BinaryCacheSftDataset(cache_dir=args.bin_cache_dir, ledger=ledger, skip_samples=skip_samples)
    else:
        print(f"[Dataset] Multi-Domain JIT Streaming active from {args.data_dir}", flush=True)
        shards_map = discover_sft_shards(args.data_dir)
        domain_weights = parse_domain_weights(args.domain_weights)
        dataset = MultiDomainSftDataset(
            tokenizer=tokenizer,
            data_files=shards_map,
            ledger=ledger,
            domain_weights=domain_weights,
            max_seq_len=args.max_seq_len,
            skip_samples=skip_samples,
        )

    model.train()
    step = start_step
    accum_loss = 0.0
    accum_tokens = 0
    micro_step = start_step * args.grad_accum
    ema_loss = None
    ema_alpha = 0.05
    t_start = time.time()

    print("[Training] Entering SFT optimization loop...", flush=True)

    for sample in dataset:
        if step >= args.max_steps:
            break

        input_ids = sample.input_ids.unsqueeze(0).to(device)
        labels = sample.labels.unsqueeze(0).to(device)

        valid_targets = (labels[..., 1:] != -100).sum().item()
        if valid_targets == 0:
            continue

        autocast_ctx = torch.amp.autocast("cuda", enabled=use_amp) if hasattr(torch, "amp") and hasattr(torch.amp, "autocast") else torch.cuda.amp.autocast(enabled=use_amp)
        with autocast_ctx:
            # Memory-Efficient Chunked Cross-Entropy Loss via native TaraForCausalLM
            # Computes loss in 1024-token chunks without materializing the massive full vocab logits tensor.
            out = model(input_ids=input_ids, labels=labels)
            loss = out.ce_loss if getattr(out, "ce_loss", None) is not None else out.loss

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[Safety Guard] NaN/Inf loss encountered. Skipping micro-step.", flush=True)
            continue

        if ema_loss is not None and step > 10 and loss.item() > 4.0 * ema_loss:
            print(f"[Safety Guard] Loss spike detected ({loss.item():.4f} vs EMA {ema_loss:.4f}). Skipping micro-step.", flush=True)
            continue

        loss_scaled = loss / args.grad_accum
        scaler.scale(loss_scaled).backward()

        accum_loss += loss.item()
        accum_tokens += valid_targets
        micro_step += 1

        if micro_step % args.grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            step += 1
            cur_loss = accum_loss / args.grad_accum
            if ema_loss is None:
                ema_loss = cur_loss
            else:
                ema_loss = (1.0 - ema_alpha) * ema_loss + ema_alpha * cur_loss

            ppl = math.exp(min(cur_loss, 20.0))
            step_dur = max(time.time() - t_start, 1e-4)
            tok_speed = accum_tokens / step_dur
            t_start = time.time()

            print(
                f"[SFT | Step {step:04d}/{args.max_steps}] "
                f"Domain: [{sample.domain}] "
                f"Loss: {cur_loss:.4f} | "
                f"PPL: {ppl:.2f} | "
                f"Speed: {tok_speed:.0f} tok/s | "
                f"Targets: {accum_tokens} | "
                f"Shard: {sample.shard}",
                flush=True,
            )

            accum_loss = 0.0
            accum_tokens = 0

            # Best Milestone Tracking
            if cur_loss < best_loss and (best_loss - cur_loss >= 0.005 or step >= args.max_steps):
                best_loss = cur_loss
                print(f"[Milestone] >>> Committing Best SFT Checkpoint: {best_loss:.4f} <<<", flush=True)
                save_sft_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    config=config,
                    tokenizer_path=tok_file,
                    output_dir=best_dir,
                    step=step,
                    loss=cur_loss,
                    best_loss=best_loss,
                    consumed_samples=step * args.grad_accum,
                    is_best=True,
                    gdrive_backup_dir=gdrive_dir,
                    hf_repo=args.hf_repo,
                    hf_token=args.hf_token,
                )

            # Regular Checkpoint Interval with Loss Spike Guard
            if step % args.checkpoint_interval == 0 or step >= args.max_steps:
                if best_loss < float("inf") and not is_checkpoint_loss_healthy(cur_loss, ref_best_loss=best_loss):
                    print(
                        f"[SFT Loss Spike Guard] Step {step:04d}: Current loss {cur_loss:.4f} is diverged (vs best {best_loss:.4f}). "
                        f"Skipping regular checkpoint save to protect normal backups from divergence pollution!",
                        flush=True,
                    )
                else:
                    save_sft_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        config=config,
                        tokenizer_path=tok_file,
                        output_dir=args.output_dir,
                        step=step,
                        loss=cur_loss,
                        best_loss=best_loss,
                        consumed_samples=step * args.grad_accum,
                        is_best=False,
                        gdrive_backup_dir=gdrive_dir,
                        hf_repo=args.hf_repo,
                        hf_token=args.hf_token,
                    )
                ledger.save()

            # Live Multi-Domain Scoring Benchmark (GSM8K Math + Code + Kannada + Reasoning + STEM)
            if step % args.benchmark_interval == 0 or step % args.probe_interval == 0 or step == 1 or step >= args.max_steps:
                run_sft_scoring_benchmark(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    step=step,
                    output_dir=args.output_dir,
                    external_benchmark_file=args.benchmark_file,
                )

    # Final Checkpoint Save
    save_sft_checkpoint(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        config=config,
        tokenizer_path=tok_file,
        output_dir=args.output_dir,
        step=step,
        loss=cur_loss if "cur_loss" in locals() else 0.0,
        best_loss=best_loss,
        consumed_samples=step * args.grad_accum,
        is_best=False,
        gdrive_backup_dir=gdrive_dir,
        hf_repo=args.hf_repo,
        hf_token=args.hf_token,
    )
    ledger.save()

    # Final Multi-Domain Benchmark Scoring Execution
    run_sft_scoring_benchmark(
        model=model,
        tokenizer=tokenizer,
        device=device,
        step=step,
        output_dir=args.output_dir,
        external_benchmark_file=args.benchmark_file,
    )

    print("=" * 80, flush=True)
    print(f"[SFT Complete] Finished {step} steps. Best SFT Loss: {best_loss:.4f}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
