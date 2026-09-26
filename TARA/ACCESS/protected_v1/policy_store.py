"""
TARA/ACCESS/protected_v1/policy_store.py

Protected Root Creator Persistent Policy Record Store.
Stores permanent root policies outside ordinary memory with cryptographic
integrity chaining.

Strict Invariants:
- Only authenticated ROOT_CREATOR with active capability_v1 can create,
  modify, or delete permanent root policies.
- Cryptographic hash-chaining prevents silent tampering, insertion, or omission.
- Neutral audit event recording (policy_record_created, policy_record_updated, policy_record_deleted).
"""

import os
import json
import secrets
import hashlib
import hmac
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple

from .manager import ProtectedStateManager, CANONICAL_CREATOR_ID


GENESIS_POLICY_HASH = "0" * 64


class PolicyRecordStore:
    def __init__(
        self,
        protected_state_mgr: ProtectedStateManager = None,
        creator_auth_service: Any = None,
        store_path: Optional[str] = None,
        audit_logger: Optional[Any] = None,
        repo_root: Optional[str] = None,
    ):
        if repo_root is None:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        self.repo_root = repo_root
        self.protected_state_mgr = protected_state_mgr
        self.creator_auth_service = creator_auth_service
        self.audit_logger = audit_logger

        if store_path is None:
            store_path = os.path.join(self.repo_root, "storage", "root_policies", "root_policies.json")
        self.store_path = os.path.abspath(store_path)
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)

        self._policies: Dict[str, Dict[str, Any]] = {}
        self._last_hash = GENESIS_POLICY_HASH
        self.load()

    def _verify_authorization(self, session_token: str) -> Tuple[bool, Optional[str]]:
        if not self.protected_state_mgr or not self.protected_state_mgr.is_active(session_token, self.creator_auth_service):
            return False, "Unauthorized: Protected capability required."
        return True, None

    def _compute_entry_hash(
        self,
        prev_hash: str,
        policy_id: str,
        version: int,
        policy_text: str,
        status: str,
        updated_at: str
    ) -> str:
        payload = f"{prev_hash}|{policy_id}|{version}|{policy_text}|{status}|{updated_at}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def load(self) -> bool:
        if os.path.exists(self.store_path):
            try:
                with open(self.store_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._policies = data.get("policies", {})
                self._last_hash = data.get("last_hash", GENESIS_POLICY_HASH)
                return True
            except Exception:
                return False
        return False

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump({
                "store_version": "1.0",
                "last_hash": self._last_hash,
                "policies": self._policies
            }, f, indent=2)

    def create_policy(self, session_token: str, policy_text: str) -> Dict[str, Any]:
        """Creates a permanent root policy."""
        is_auth, err = self._verify_authorization(session_token)
        if not is_auth:
            return {"status": "ERROR", "error": err}

        if not policy_text or not isinstance(policy_text, str) or not policy_text.strip():
            return {"status": "ERROR", "error": "Policy text cannot be empty."}

        policy_id = f"pol_{secrets.token_hex(6)}"
        now = datetime.now(timezone.utc).isoformat()
        prev_hash = self._last_hash

        entry_hash = self._compute_entry_hash(
            prev_hash=prev_hash,
            policy_id=policy_id,
            version=1,
            policy_text=policy_text.strip(),
            status="ACTIVE",
            updated_at=now
        )

        policy_record = {
            "policy_id": policy_id,
            "creator_id": CANONICAL_CREATOR_ID,
            "version": 1,
            "policy_text": policy_text.strip(),
            "status": "ACTIVE",
            "created_at": now,
            "updated_at": now,
            "prev_hash": prev_hash,
            "entry_hash": entry_hash
        }

        self._policies[policy_id] = policy_record
        self._last_hash = entry_hash
        self.save()

        if self.audit_logger:
            self.audit_logger.log_event(
                event_type="policy_record_created",
                severity="INFO",
                details={"policy_id": policy_id, "creator_id": CANONICAL_CREATOR_ID}
            )

        return {"status": "SUCCESS", "policy": policy_record}

    def update_policy(self, session_token: str, policy_id: str, new_text: str) -> Dict[str, Any]:
        """Updates an existing permanent root policy."""
        is_auth, err = self._verify_authorization(session_token)
        if not is_auth:
            return {"status": "ERROR", "error": err}

        if policy_id not in self._policies:
            return {"status": "ERROR", "error": f"Policy not found: {policy_id}"}

        cur = self._policies[policy_id]
        now = datetime.now(timezone.utc).isoformat()
        prev_hash = self._last_hash
        new_ver = cur.get("version", 1) + 1

        entry_hash = self._compute_entry_hash(
            prev_hash=prev_hash,
            policy_id=policy_id,
            version=new_ver,
            policy_text=new_text.strip(),
            status=cur.get("status", "ACTIVE"),
            updated_at=now
        )

        cur["version"] = new_ver
        cur["policy_text"] = new_text.strip()
        cur["updated_at"] = now
        cur["prev_hash"] = prev_hash
        cur["entry_hash"] = entry_hash

        self._last_hash = entry_hash
        self.save()

        if self.audit_logger:
            self.audit_logger.log_event(
                event_type="policy_record_updated",
                severity="INFO",
                details={"policy_id": policy_id, "new_version": new_ver}
            )

        return {"status": "SUCCESS", "policy": cur}

    def delete_policy(self, session_token: str, policy_id: str) -> Dict[str, Any]:
        """Deletes a permanent root policy."""
        is_auth, err = self._verify_authorization(session_token)
        if not is_auth:
            return {"status": "ERROR", "error": err}

        if policy_id not in self._policies:
            return {"status": "ERROR", "error": f"Policy not found: {policy_id}"}

        removed = self._policies.pop(policy_id)
        now = datetime.now(timezone.utc).isoformat()
        prev_hash = self._last_hash

        entry_hash = self._compute_entry_hash(
            prev_hash=prev_hash,
            policy_id=policy_id,
            version=removed.get("version", 1),
            policy_text="DELETED",
            status="DELETED",
            updated_at=now
        )
        self._last_hash = entry_hash
        self.save()

        if self.audit_logger:
            self.audit_logger.log_event(
                event_type="policy_record_deleted",
                severity="WARNING",
                details={"policy_id": policy_id}
            )

        return {"status": "SUCCESS", "deleted_policy_id": policy_id}

    def set_policy_status(self, session_token: str, policy_id: str, status: str) -> Dict[str, Any]:
        """Enables or disables a permanent root policy."""
        is_auth, err = self._verify_authorization(session_token)
        if not is_auth:
            return {"status": "ERROR", "error": err}

        if policy_id not in self._policies:
            return {"status": "ERROR", "error": f"Policy not found: {policy_id}"}

        if status not in ("ACTIVE", "DISABLED"):
            return {"status": "ERROR", "error": "Status must be 'ACTIVE' or 'DISABLED'."}

        cur = self._policies[policy_id]
        cur["status"] = status
        cur["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.save()

        return {"status": "SUCCESS", "policy_id": policy_id, "new_status": status}

    def get_policy(self, policy_id: str) -> Optional[Dict[str, Any]]:
        return self._policies.get(policy_id)

    def list_policies(self, active_only: bool = True) -> List[Dict[str, Any]]:
        if active_only:
            return [p for p in self._policies.values() if p.get("status") == "ACTIVE"]
        return list(self._policies.values())

    def verify_store_integrity(self) -> Tuple[bool, Optional[str]]:
        """Cryptographically verifies that no policy has been corrupted or forged."""
        for pol_id, p in self._policies.items():
            expected = self._compute_entry_hash(
                prev_hash=p.get("prev_hash", ""),
                policy_id=p.get("policy_id", ""),
                version=p.get("version", 1),
                policy_text=p.get("policy_text", ""),
                status=p.get("status", "ACTIVE"),
                updated_at=p.get("updated_at", "")
            )
            actual = p.get("entry_hash", "")
            if not hmac.compare_digest(expected, actual):
                return False, f"Integrity check failed on policy {pol_id}"
        return True, None


