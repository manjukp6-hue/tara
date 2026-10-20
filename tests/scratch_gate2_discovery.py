#!/usr/bin/env python3
"""
Phase 3.2 Gate 2 — Read-only production dataset discovery dry-run.
No training, no download, no state modification.
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

dataset_repo = os.environ.get("HF_DATASET_REPO", os.environ.get("HF_REPO", "manjukp6/tara"))
token = os.environ.get("HF_TOKEN") or None
ds_prefix = "v2"

print(f"[Discovery] Repository: {dataset_repo}")
print(f"[Discovery] Prefix:     {ds_prefix}/")
print(f"[Discovery] Token:      {'SET' if token else 'NOT SET (anonymous)'}")
print()

try:
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    print(f"[Discovery] Calling HfApi.list_repo_files(repo_id={dataset_repo!r}, repo_type='dataset')...")
    files = list(api.list_repo_files(repo_id=dataset_repo, repo_type="dataset"))
    print(f"[Discovery] Total files in repo: {len(files)}")
    prefix_filter = f"{ds_prefix}/"
    jsonl_files = sorted([f for f in files if f.startswith(prefix_filter) and f.endswith(".jsonl")])
    basenames = [os.path.basename(f) for f in jsonl_files]
    print(f"[Discovery] JSONL shards matching prefix {prefix_filter!r}: {len(basenames)}")
    for i, name in enumerate(basenames):
        print(f"  [{i+1:03d}] {name}")
    if not basenames:
        print("[Discovery] ZERO shards discovered — check prefix and repo access.")
    else:
        print(f"[Discovery] First shard: {basenames[0]}")
        print(f"[Discovery] Last shard:  {basenames[-1]}")
    print()
    # Check for any non-v2 prefixes that might exist (other versions)
    all_prefixes = sorted(set(f.split("/")[0] for f in files if "/" in f))
    print(f"[Discovery] All top-level directories in repo: {all_prefixes}")
    print()
    print("[Discovery] Completeness basis: HfApi.list_repo_files() traverses all API pages automatically.")
    print("[Discovery] No hard-coded shard count in discovery logic (see cloud_train.py L3420).")
    print("[Discovery] Zero downloads, zero training, zero state modification performed.")
    sys.exit(0)
except ImportError:
    print("[Discovery] huggingface_hub not installed — cannot run remote discovery.")
    sys.exit(2)
except Exception as e:
    print(f"[Discovery] ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
