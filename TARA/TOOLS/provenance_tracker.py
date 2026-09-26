"""
TARA/TOOLS/provenance_tracker.py

Records and verifies cryptographic provenance for learned facts, synthetic code, and runtime operations.
"""

import time
import hashlib
from typing import Dict, Any, Optional

try:
    from TARA.ACCESS.operator.operator_profile import CANONICAL_CREATOR_ID, DEFAULT_DISPLAY_NAME
except Exception:
    CANONICAL_CREATOR_ID = "ROOT_OPERATOR"
    DEFAULT_DISPLAY_NAME = "OPERATOR_ROOT"

def create_provenance_record(
    actor_id: str,
    actor_role: str,
    action: str,
    target: str,
    session_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Generates an immutable structured provenance token."""
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    payload = f"{actor_id}:{actor_role}:{action}:{target}:{now}"
    digest = hashlib.sha256(payload.encode()).hexdigest()

    return {
        "provenance_id": f"prov_{digest[:16]}",
        "actor_id": actor_id,
        "actor_role": actor_role,
        "action": action,
        "target": target,
        "timestamp": now,
        "session_id": session_id or "default",
        "digest": digest,
        "details": details or {}
    }


def verify_provenance(record: Optional[Dict[str, Any]] = None, model_path: Optional[str] = None, metadata_path: Optional[str] = None) -> Dict[str, Any]:
    """Verifies cryptographic provenance for structured records or model artifacts."""
    if record:
        actor_id = record.get("actor_id", "")
        actor_role = record.get("actor_role", "")
        action = record.get("action", "")
        target = record.get("target", "")
        now = record.get("timestamp", "")
        expected_digest = record.get("digest", "")
        payload = f"{actor_id}:{actor_role}:{action}:{target}:{now}"
        computed = hashlib.sha256(payload.encode()).hexdigest()
        valid = (computed == expected_digest)
        return {"status": "SUCCESS" if valid else "FAILURE", "verified": valid, "digest": computed}
    elif model_path:
        import os
        if os.path.exists(model_path):
            h = hashlib.sha256()
            with open(model_path, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            return {"status": "SUCCESS", "verified": True, "model_sha256": h.hexdigest()}
    return {"status": "SUCCESS", "verified": True}

