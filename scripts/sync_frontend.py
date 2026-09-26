#!/usr/bin/env python3
"""
scripts/sync_frontend.py

CLI entrypoint to execute automated synchronization of the canonical TARA frontend
across Cloudflare, ModelScope, Hugging Face, Render, Python Core, and Rust servers.
"""

import os
import sys
import json

# Ensure stdout handles UTF-8 gracefully on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON_DIR = os.path.join(REPO_ROOT, "python")
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)

from tara_core.frontend_sync import sync_all_targets, verify_sync_integrity

def main():
    print("[TARA] Executing Canonical Frontend Synchronization across all provider replicas...")
    results = sync_all_targets()
    print(f"[TARA] Canonical SHA256: {results['canonical_sha256']}")
    print(f"[TARA] Total Size: {results['byte_size']} bytes")
    
    print("\n[TARA] Running Integrity Audit across target replicas...")
    audit = verify_sync_integrity()
    for target, stat in audit["target_status"].items():
        print(f"  - {target:15}: {stat['status']} ({stat['path']})")
    
    if audit["all_in_sync"]:
        print("\n[OK] All provider replicas are 100% SYNCHRONIZED with 0 drift.")
        sys.exit(0)
    else:
        print("\n[FAIL] Replication MISMATCH detected!")
        sys.exit(1)

if __name__ == "__main__":
    main()
