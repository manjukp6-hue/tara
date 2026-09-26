#!/usr/bin/env python3
"""
scripts/generate_pre_push_manifest.py

Generates the complete pre-push manifest of:
- every file
- every folder
- file size
- release status (PUBLISH / EXCLUDE)
- Model checksum verification
- Targets and timestamp
"""

import os
import sys
import json
import hashlib
import subprocess
from datetime import datetime, timezone

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXPECTED_MODEL_SHA256 = "7a50308b799f2654baeafbd64dec31f088c3b07b446e198cd0f9ec2b7c0af309"

def compute_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"

def batch_get_git_ignored(all_rel_paths):
    """Checks all paths against git ignore in a single batch process."""
    try:
        proc = subprocess.Popen(
            ["git", "check-ignore", "--stdin"],
            cwd=REPO_ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        out, _ = proc.communicate(input="\n".join(all_rel_paths))
        return set(line.strip().replace("\\", "/") for line in out.splitlines() if line.strip())
    except Exception:
        return set()

def generate_manifest():
    print("=" * 80)
    print("      TARA CORE PRE-PUSH RELEASE AUDIT & MANIFEST GENERATOR")
    print("=" * 80)

    # 1. Model Verification
    model_path = os.path.join(REPO_ROOT, "storage", "models", "tara", "model.safetensors")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file missing: {model_path}")
    
    actual_model_sha256 = compute_sha256(model_path)
    print(f"\n[Model Verification]")
    print(f"  Path:            storage/models/tara/model.safetensors")
    print(f"  Expected SHA256: {EXPECTED_MODEL_SHA256}")
    print(f"  Actual SHA256:   {actual_model_sha256}")
    if actual_model_sha256.lower() != EXPECTED_MODEL_SHA256.lower():
        raise ValueError("MODEL CHECKSUM MISMATCH! Refusing to proceed.")
    print("  Status:          [MATCH VERIFIED - 100% INTACT]")

    # 2. Targets & Timestamps
    current_time_utc = datetime.now(timezone.utc).isoformat()
    github_target = "https://github.com/manjukp6-hue/tara.git (branch: main)"
    hf_model_target = "https://huggingface.co/tara-project/tara (Model Hub: SafeTensors, configs, tokenizer)"
    hf_space_target = "https://huggingface.co/spaces/tara-project/tara (Static Web Replica: frontend)"

    print(f"\n[Release Targets & Timestamp]")
    print(f"  Release Timestamp:  {current_time_utc}")
    print(f"  GitHub Target:      {github_target}")
    print(f"  Hugging Face Model: {hf_model_target}")
    print(f"  Hugging Face Space: {hf_space_target}")

    # 3. Discover Filesystem
    EXCLUDE_DIR_NAMES = {".git", ".pytest_cache", "target", "node_modules", ".gemini"}
    raw_files = []
    folders_manifest = set()

    for root, dirs, files in os.walk(REPO_ROOT):
        rel_root = os.path.relpath(root, REPO_ROOT).replace("\\", "/")
        if rel_root == ".":
            rel_root = ""
        
        parts = rel_root.split("/") if rel_root else []
        if any(p in EXCLUDE_DIR_NAMES for p in parts):
            continue

        if rel_root:
            folders_manifest.add(rel_root)

        for f in files:
            rel_file = (f"{rel_root}/{f}" if rel_root else f).replace("\\", "/")
            full_path = os.path.join(root, f)
            try:
                f_size = os.path.getsize(full_path)
            except OSError:
                f_size = 0
            raw_files.append((rel_file, f, f_size))

    # Fast batch check ignore
    all_rel_paths = [rf[0] for rf in raw_files]
    ignored_paths = batch_get_git_ignored(all_rel_paths)

    files_manifest = []
    total_publish_bytes = 0
    total_exclude_bytes = 0
    publish_count = 0
    exclude_count = 0

    for rel_file, f, f_size in raw_files:
        is_excluded = False
        exclude_reason = None

        if rel_file.startswith(".git/"):
            is_excluded = True
            exclude_reason = "VCS Internal (.git)"
        elif rel_file in ignored_paths:
            is_excluded = True
            exclude_reason = "Ignored (.gitignore / Security Policy)"
        elif f.endswith(".pyc") or "__pycache__" in rel_file:
            is_excluded = True
            exclude_reason = "Python bytecode cache"
        elif f.endswith(".keystore") or f.endswith(".secret") or f.endswith(".key") or f.endswith(".sealed"):
            is_excluded = True
            exclude_reason = "Security: Cryptographic credential / keystore"
        elif f == ".env" or f.startswith(".env."):
            is_excluded = True
            exclude_reason = "Security: Local environment credentials"
        elif "storage/vault" in rel_file or "TARA/ACCESS/vault" in rel_file:
            is_excluded = True
            exclude_reason = "Security: Local protected storage"
        elif "storage/audit" in rel_file:
            is_excluded = True
            exclude_reason = "Test artifact: Audit run results"
        elif "storage/memory" in rel_file:
            is_excluded = True
            exclude_reason = "Local machine data: Episodic memory cache"

        status = "EXCLUDE" if is_excluded else "PUBLISH"
        if is_excluded:
            exclude_count += 1
            total_exclude_bytes += f_size
        else:
            publish_count += 1
            total_publish_bytes += f_size

        files_manifest.append({
            "path": rel_file,
            "size_bytes": f_size,
            "size_formatted": format_size(f_size),
            "status": status,
            "exclude_reason": exclude_reason
        })

    files_manifest.sort(key=lambda x: x["path"])
    sorted_folders = sorted(list(folders_manifest))

    manifest_output = {
        "timestamp_utc": current_time_utc,
        "github_target": github_target,
        "huggingface_model_target": hf_model_target,
        "huggingface_space_target": hf_space_target,
        "model_sha256": actual_model_sha256,
        "summary": {
            "total_folders": len(sorted_folders),
            "total_files": len(files_manifest),
            "publish_files": publish_count,
            "publish_bytes": total_publish_bytes,
            "publish_formatted": format_size(total_publish_bytes),
            "exclude_files": exclude_count,
            "exclude_bytes": total_exclude_bytes,
            "exclude_formatted": format_size(total_exclude_bytes)
        },
        "folders": sorted_folders,
        "files": files_manifest
    }

    output_path = os.path.join(REPO_ROOT, "storage", "pre_push_manifest.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest_output, f, indent=2)

    print(f"\n[Summary Statistics]")
    print(f"  Total Folders:       {len(sorted_folders)}")
    print(f"  Total Files Scanned: {len(files_manifest)}")
    print(f"  PUBLISH Files:       {publish_count} ({format_size(total_publish_bytes)})")
    print(f"  EXCLUDE Files:       {exclude_count} ({format_size(total_exclude_bytes)})")
    print(f"  Manifest JSON Saved: {output_path}")

    # Top PUBLISH folders breakdown
    folder_publish_counts = {}
    for item in files_manifest:
        if item["status"] == "PUBLISH":
            top_dir = item["path"].split("/")[0] if "/" in item["path"] else "(root)"
            folder_publish_counts[top_dir] = folder_publish_counts.get(top_dir, 0) + 1

    print("\n[PUBLISH Files by Top-Level Directory]")
    for fld, cnt in sorted(folder_publish_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {fld:30s}: {cnt:4d} files")

    return manifest_output

if __name__ == "__main__":
    generate_manifest()
