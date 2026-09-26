"""
TARA/ACCESS/creator/multi_creator_registry.py

Multi-Creator Identity and Authority Registry for TARA.
Supports:
- Creator 1 / ROOT_OPERATOR (ROOT_CREATOR)
- Creator 2 (CREATOR)
- Creator 3 (CREATOR)
- Future authorized creators

Explicit Mapping Schema Per Creator:
- creator_id: Canonical uppercase identifier (e.g. 'ROOT_OPERATOR')
- display_name: Human-friendly name (e.g. 'OPERATOR_ROOT')
- role: 'ROOT_CREATOR' (strictly ROOT_OPERATOR only) or 'CREATOR'
- authorized_google_email: Bound Google/Gmail address (None if unconfigured)
- google_subject_id: Google OAuth 'sub' claim (None if unconfigured)
- public_key: Ed25519 public key hex (64 chars)
- key_id: Public key identifier / SHA256 fingerprint
- status: 'active', 'suspended', or 'unconfigured'

Security Invariants:
- Role Enforcement: ROOT_OPERATOR is permanently ROOT_CREATOR.
- No Self-Escalation: Normal CREATORs (Creator 2, Creator 3) cannot self-escalate to ROOT_CREATOR.
- Zero Invented Identities: Unconfigured creators have authorized_google_email = None.
- Status Enforcement: Only 'active' status accounts can authenticate.
"""

import os
import json
import hashlib
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from .operator_profile import CANONICAL_CREATOR_ID, DEFAULT_DISPLAY_NAME


def compute_key_id(public_key_hex: Optional[str]) -> Optional[str]:
    """Computes a SHA256 key_id fingerprint for an Ed25519 public key."""
    if not public_key_hex:
        return None
    clean = public_key_hex.strip().lower()
    return hashlib.sha256(clean.encode("utf-8")).hexdigest()[:16]


class MultiCreatorRegistry:
    """
    Registry managing identities, authorized public keys, roles, and statuses
    for all authorized TARA creators.
    """

    def __init__(self, registry_path: Optional[str] = None):
        if registry_path is None:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
            registry_path = os.path.join(repo_root, "TARA", "ACCESS", "operator", "operators_registry.json")
        self.registry_path = registry_path
        self._lock = threading.RLock()
        self._creators: Dict[str, Dict[str, Any]] = {}
        self._init_defaults()
        if os.path.exists(self.registry_path):
            self.load()

    def _init_defaults(self):
        """Initializes canonical creators with explicit schema."""
        self._creators = {
            CANONICAL_CREATOR_ID: {
                "creator_id": CANONICAL_CREATOR_ID,
                "display_name": DEFAULT_DISPLAY_NAME,
                "role": "ROOT_CREATOR",
                "authorized_google_email": os.environ.get("TARA_CREATOR_EMAIL", "operator@internal.local"),
                "google_subject_id": None,
                "public_key": None,
                "key_id": None,
                "status": "active",
                "created_at": datetime.now(timezone.utc).isoformat()
            },
            "CREATOR_2": {
                "creator_id": "CREATOR_2",
                "display_name": "Creator Two",
                "role": "CREATOR",
                "authorized_google_email": None,  # unconfigured until legitimately enrolled
                "google_subject_id": None,
                "public_key": None,
                "key_id": None,
                "status": "active",
                "created_at": datetime.now(timezone.utc).isoformat()
            },
            "CREATOR_3": {
                "creator_id": "CREATOR_3",
                "display_name": "Creator Three",
                "role": "CREATOR",
                "authorized_google_email": None,  # unconfigured until legitimately enrolled
                "google_subject_id": None,
                "public_key": None,
                "key_id": None,
                "status": "active",
                "created_at": datetime.now(timezone.utc).isoformat()
            }
        }

    def register_creator(
        self,
        creator_id: str,
        display_name: str,
        role: str = "CREATOR",
        authorized_google_email: Optional[str] = None,
        google_subject_id: Optional[str] = None,
        public_key: Optional[str] = None,
        status: str = "active"
    ) -> Dict[str, Any]:
        """
        Registers or updates an authorized creator.
        Enforces that ROOT_CREATOR is strictly reserved for ROOT_OPERATOR.
        """
        clean_id = creator_id.strip().upper()
        assigned_role = "ROOT_CREATOR" if clean_id == CANONICAL_CREATOR_ID else "CREATOR"
        clean_pub = public_key.lower().strip() if public_key else None
        key_id = compute_key_id(clean_pub)

        record = {
            "creator_id": clean_id,
            "display_name": display_name.strip(),
            "role": assigned_role,
            "authorized_google_email": authorized_google_email.lower().strip() if authorized_google_email else None,
            "google_subject_id": google_subject_id.strip() if google_subject_id else None,
            "public_key": clean_pub,
            "key_id": key_id,
            "status": status.lower().strip(),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        with self._lock:
            self._creators[clean_id] = record
            self.save()
        return record

    def enroll_google_identity(
        self,
        creator_id: str,
        google_email: str,
        google_subject_id: Optional[str] = None,
        authorized_by: str = CANONICAL_CREATOR_ID
    ) -> Dict[str, Any]:
        """
        Securely enrolls or updates the verified Google identity for a creator.
        Requires ROOT_CREATOR authority (authorized_by == ROOT_OPERATOR).
        Enforces uniqueness: no two creators can bind the same Google identity.
        """
        auth_actor = authorized_by.strip().upper()
        if auth_actor != CANONICAL_CREATOR_ID:
            raise PermissionError(f"Unauthorized: Only ROOT_CREATOR ({CANONICAL_CREATOR_ID}) can enroll creator identities.")

        clean_id = creator_id.strip().upper()
        clean_email = google_email.lower().strip()
        clean_sub = google_subject_id.strip() if google_subject_id else None

        if not clean_email or "@" not in clean_email:
            raise ValueError("Invalid Google email address format.")

        with self._lock:
            if clean_id not in self._creators:
                raise KeyError(f"Creator '{clean_id}' not found in registry.")

            for other_id, other_rec in self._creators.items():
                if other_id != clean_id:
                    if other_rec.get("authorized_google_email") == clean_email:
                        raise ValueError(f"Email '{clean_email}' is already bound to creator '{other_id}'.")
                    if clean_sub and other_rec.get("google_subject_id") == clean_sub:
                        raise ValueError(f"Google subject ID is already bound to creator '{other_id}'.")

            creator_rec = self._creators[clean_id]
            creator_rec["authorized_google_email"] = clean_email
            if clean_sub:
                creator_rec["google_subject_id"] = clean_sub
            creator_rec["updated_at"] = datetime.now(timezone.utc).isoformat()
            self.save()
            return dict(creator_rec)

    def bind_google_subject_id(self, creator_id: str, google_subject_id: str) -> bool:
        """Binds the verified Google OAuth subject ID ('sub') on first verified authentication."""
        clean_id = creator_id.strip().upper()
        clean_sub = google_subject_id.strip()
        if not clean_sub:
            return False

        with self._lock:
            rec = self._creators.get(clean_id)
            if not rec:
                return False
            current_sub = rec.get("google_subject_id")
            if current_sub and current_sub != clean_sub:
                return False
            rec["google_subject_id"] = clean_sub
            self.save()
            return True

    def get_creator(self, creator_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            rec = self._creators.get(creator_id.strip().upper())
            return dict(rec) if rec else None

    def get_creator_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        clean_email = email.lower().strip()
        with self._lock:
            for c in self._creators.values():
                email_val = c.get("authorized_google_email") or c.get("email")
                if email_val and email_val.lower() == clean_email:
                    return dict(c)
            return None

    def get_creator_by_google_identity(
        self,
        email: Optional[str] = None,
        subject_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Looks up a creator by verified Google email or Google subject ID.
        Returns None if identity is unconfigured, mismatched, or not found.
        """
        clean_email = email.lower().strip() if email else None
        clean_sub = subject_id.strip() if subject_id else None

        with self._lock:
            for c in self._creators.values():
                c_email = (c.get("authorized_google_email") or "").lower().strip()
                c_sub = (c.get("google_subject_id") or "").strip()

                if clean_email and c_email:
                    if c_email == clean_email:
                        # If subject_id is already bound, it must match
                        if clean_sub and c_sub and c_sub != clean_sub:
                            continue
                        return dict(c)
                elif clean_sub and c_sub and not clean_email:
                    if c_sub == clean_sub:
                        return dict(c)
            return None

    def get_creator_by_public_key(self, public_key_hex: str) -> Optional[Dict[str, Any]]:
        clean_key = public_key_hex.lower().strip()
        with self._lock:
            for c in self._creators.values():
                if c.get("public_key") and c["public_key"].lower() == clean_key:
                    return dict(c)
            return None

    def set_public_key(self, creator_id: str, public_key_hex: str, key_id: Optional[str] = None) -> bool:
        clean_id = creator_id.strip().upper()
        clean_pub = public_key_hex.lower().strip()
        computed_kid = key_id or compute_key_id(clean_pub)
        with self._lock:
            if clean_id in self._creators:
                self._creators[clean_id]["public_key"] = clean_pub
                self._creators[clean_id]["key_id"] = computed_kid
                self.save()
                return True
            return False

    def set_authorized_google_email(self, creator_id: str, email: Optional[str]) -> bool:
        clean_id = creator_id.strip().upper()
        clean_email = email.lower().strip() if email else None
        with self._lock:
            if clean_id in self._creators:
                self._creators[clean_id]["authorized_google_email"] = clean_email
                self.save()
                return True
            return False

    def update_status(self, creator_id: str, status: str) -> bool:
        clean_id = creator_id.strip().upper()
        with self._lock:
            if clean_id in self._creators:
                self._creators[clean_id]["status"] = status.lower().strip()
                self.save()
                return True
            return False

    def is_active(self, creator_id: str) -> bool:
        c = self.get_creator(creator_id)
        return c is not None and c.get("status") == "active"

    def resolve_role(self, creator_id: str, claimed_role: Optional[str] = None) -> str:
        clean_id = creator_id.strip().upper()
        c = self.get_creator(clean_id)
        if not c:
            return "USER"
        if clean_id == CANONICAL_CREATOR_ID:
            return "ROOT_CREATOR"
        return "CREATOR"

    def list_creators(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(c) for c in self._creators.values()]

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        with self._lock:
            with open(self.registry_path, "w", encoding="utf-8") as f:
                json.dump(self._creators, f, indent=2)

    def load(self) -> None:
        try:
            with open(self.registry_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            with self._lock:
                for k, v in data.items():
                    cid = k.upper()
                    if cid == CANONICAL_CREATOR_ID:
                        v["role"] = "ROOT_CREATOR"
                    elif v.get("role") == "ROOT_CREATOR":
                        v["role"] = "CREATOR"

                    if "authorized_google_email" not in v:
                        legacy_email = v.get("email")
                        if legacy_email and not legacy_email.endswith("@example.com"):
                            v["authorized_google_email"] = legacy_email
                        else:
                            v["authorized_google_email"] = None

                    if "google_subject_id" not in v:
                        v["google_subject_id"] = None

                    if "key_id" not in v:
                        v["key_id"] = compute_key_id(v.get("public_key"))

                    self._creators[cid] = v
        except Exception:
            pass
