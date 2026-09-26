"""
TARA/ACCESS/creator/creator_identity.py

Canonical TARA Root Creator Identity Specification:
- Permanent Creator ID: ROOT_OPERATOR
- Display Name: OPERATOR_ROOT

Invariants:
- ROOT_OPERATOR is permanent and immutable.
- Display name may be updated without altering Creator ID.
- Key versioning tracks rotation and revocations.
"""

import os
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from ..crypto.ed25519 import Ed25519

CANONICAL_CREATOR_ID = "ROOT_OPERATOR"
DEFAULT_DISPLAY_NAME = "OPERATOR_ROOT"


class CreatorIdentity:
    """
    Manages the permanent TARA Root Creator Identity record.
    """
    def __init__(self, record_path: Optional[str] = None):
        if record_path is None:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
            record_path = os.path.join(repo_root, "TARA", "ACCESS", "operator", "operator_record.json")
        self.record_path = record_path
        self.creator_id: str = CANONICAL_CREATOR_ID
        self.display_name: str = DEFAULT_DISPLAY_NAME
        self.identity_version: int = 1
        self.key_version: int = 1
        self.root_public_key: Optional[str] = None
        self.status: str = "CREATOR_SETUP_REQUIRED"
        self.recovery_enabled: bool = False
        self.created_at: Optional[str] = None
        self.last_key_rotation: Optional[str] = None
        self.revoked_keys: List[Dict[str, Any]] = []

        if os.path.exists(self.record_path):
            self.load()

    def is_initialized(self) -> bool:
        return bool(self.root_public_key and self.status == "active")

    def initialize_root_creator(
        self,
        public_key_bytes: bytes,
        display_name: str = DEFAULT_DISPLAY_NAME
    ) -> Dict[str, Any]:
        """
        First setup of the root creator identity.
        Rejects any attempt to re-initialize or alter Creator ID.
        """
        if self.is_initialized():
            if self.creator_id != CANONICAL_CREATOR_ID:
                raise ValueError(f"Permanent Creator ID cannot be modified from {CANONICAL_CREATOR_ID}")
            raise PermissionError("Creator authority is already initialized. Authentication or authorized recovery is required for changes.")

        self.creator_id = CANONICAL_CREATOR_ID
        self.display_name = display_name
        self.root_public_key = public_key_bytes.hex()
        self.identity_version = 1
        self.key_version = 1
        self.status = "active"
        self.recovery_enabled = True
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.save()
        return self.to_dict()

    def update_display_name(self, new_display_name: str) -> None:
        """Updates human display name without affecting Creator ID."""
        if not new_display_name or not new_display_name.strip():
            raise ValueError("Display name cannot be empty")
        self.display_name = new_display_name.strip()
        self.save()

    def rotate_root_key(self, new_public_key_bytes: bytes, authorized: bool = False, reason: str = "scheduled_rotation") -> None:
        """
        Rotates the creator key, archiving the previous key version into revoked_keys.
        Creator ID remains ROOT_OPERATOR.
        Requires authenticated creator authorization or authorized recovery (authorized=True).
        """
        if not authorized:
            raise PermissionError("Creator key rotation requires authenticated existing creator authorization or valid recovery.")
        if not self.root_public_key:
            raise RuntimeError("Cannot rotate an uninitialized creator key")

        # Archive old key
        self.revoked_keys.append({
            "key_version": self.key_version,
            "public_key": self.root_public_key,
            "revoked_at": datetime.now(timezone.utc).isoformat(),
            "reason": reason
        })

        self.key_version += 1
        self.root_public_key = new_public_key_bytes.hex()
        self.last_key_rotation = datetime.now(timezone.utc).isoformat()
        self.save()

    def is_key_active(self, public_key_hex: str) -> bool:
        """Returns True if the given public key is the currently active root key."""
        return (
            self.root_public_key is not None
            and self.root_public_key.lower() == public_key_hex.lower()
            and self.status == "active"
        )

    def verify_authority(self, message: bytes, signature_bytes: bytes) -> bool:
        """Cryptographically verifies a message against the active root public key."""
        if not self.root_public_key or self.status != "active":
            return False
        pub_bytes = bytes.fromhex(self.root_public_key)
        return Ed25519.verify(pub_bytes, message, signature_bytes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "creator_id": self.creator_id,
            "display_name": self.display_name,
            "identity_version": self.identity_version,
            "key_version": self.key_version,
            "root_public_key": self.root_public_key,
            "status": self.status,
            "recovery_enabled": self.recovery_enabled,
            "created_at": self.created_at,
            "last_key_rotation": self.last_key_rotation,
            "revoked_keys": self.revoked_keys
        }

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.record_path), exist_ok=True)
        # Enforce canonical creator id invariant
        self.creator_id = CANONICAL_CREATOR_ID
        with open(self.record_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def load(self) -> None:
        with open(self.record_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Invariant check
        if data.get("creator_id") != CANONICAL_CREATOR_ID:
            raise ValueError(f"Corrupted Creator ID {data.get('creator_id')}; must be {CANONICAL_CREATOR_ID}")

        self.creator_id = CANONICAL_CREATOR_ID
        self.display_name = data.get("display_name", DEFAULT_DISPLAY_NAME)
        self.identity_version = data.get("identity_version", 1)
        self.key_version = data.get("key_version", 1)
        self.root_public_key = data.get("root_public_key")
        self.status = data.get("status", "active")
        self.recovery_enabled = data.get("recovery_enabled", True)
        self.created_at = data.get("created_at", datetime.now(timezone.utc).isoformat())
        self.last_key_rotation = data.get("last_key_rotation")
        self.revoked_keys = data.get("revoked_keys", [])
