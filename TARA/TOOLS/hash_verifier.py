"""
TARA/TOOLS/hash_verifier.py

Cryptographic hash verification tool for TARA integrity checks.
Supports SHA-256, SHA-512, and string or stream verification.
"""

import hashlib
import os
from typing import Dict, Any, Optional

def compute_hash(data: bytes, algorithm: str = "sha256") -> str:
    """Computes hexadecimal digest for bytes."""
    algo = algorithm.lower().strip()
    if algo == "sha256":
        return hashlib.sha256(data).hexdigest()
    elif algo == "sha512":
        return hashlib.sha512(data).hexdigest()
    elif algo == "sha1":
        return hashlib.sha1(data).hexdigest()
    elif algo == "md5":
        return hashlib.md5(data).hexdigest()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")

def verify_file_hash(file_path: str, expected_hash: str, algorithm: str = "sha256") -> Dict[str, Any]:
    """Computes file hash and compares against expected hash."""
    if not os.path.exists(file_path):
        return {
            "verified": False,
            "error": f"File '{file_path}' does not exist"
        }

    algo = algorithm.lower().strip()
    hasher = getattr(hashlib, algo, hashlib.sha256)()

    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)

    computed = hasher.hexdigest()
    is_match = (computed.lower() == expected_hash.lower().strip())

    return {
        "verified": is_match,
        "computed_hash": computed,
        "expected_hash": expected_hash,
        "algorithm": algo,
        "file_path": os.path.abspath(file_path)
    }
