"""
TARA/TOOLS/file_inspector.py

Standard inspection utility for files within authorized TARA project boundaries.
Provides size, format, line count, and hash checks without leaking sensitive secrets.
"""

import os
import hashlib
from typing import Dict, Any, Optional

DENIED_SUBSTRINGS = [
    os.path.join("ACCESS", "vault"),
    ".git",
    "private_key",
    "credentials",
    ".env"
]

def inspect_file(file_path: str, repo_root: Optional[str] = None) -> Dict[str, Any]:
    """
    Safely inspects a file's metadata and SHA-256 checksum.
    Refuses to inspect files containing private keys, credentials, or secrets.
    """
    abs_path = os.path.abspath(file_path)

    # Boundary & Secret check
    for denied in DENIED_SUBSTRINGS:
        if denied in abs_path:
            raise PermissionError(f"Access Denied: Cannot inspect sensitive path '{denied}'.")

    if not os.path.exists(abs_path):
        return {
            "status": "ERROR",
            "error": f"File '{file_path}' does not exist",
            "exists": False
        }

    if os.path.isdir(abs_path):
        return {
            "status": "SUCCESS",
            "exists": True,
            "is_dir": True,
            "path": abs_path,
            "child_count": len(os.listdir(abs_path))
        }

    size_bytes = os.path.getsize(abs_path)
    hasher = hashlib.sha256()
    line_count = 0

    with open(abs_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)

    try:
        with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
            line_count = sum(1 for _ in f)
    except Exception:
        line_count = -1

    return {
        "status": "SUCCESS",
        "exists": True,
        "is_dir": False,
        "path": abs_path,
        "size_bytes": size_bytes,
        "sha256": hasher.hexdigest(),
        "line_count": line_count,
        "extension": os.path.splitext(abs_path)[1]
    }
