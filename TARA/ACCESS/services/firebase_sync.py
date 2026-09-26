"""
TARA/ACCESS/services/firebase_sync.py

Firebase/Firestore supporting synchronization service.

Security Invariants:
- Firebase does NOT own the permanent Creator identity.
- Never store creator private key, device private key, biometric data, or raw recovery secrets.
- If Firebase becomes unavailable, existing authorized devices continue to operate locally.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone


class FirebaseSyncService:
    """
    Manages non-secret metadata synchronization with Firebase / Firestore.
    Maintains an offline cache ensuring TARA never depends on Firebase availability.
    """
    def __init__(self, is_online: bool = True):
        self.is_online = is_online
        # Simulated Firestore document collections
        self.firestore_users: Dict[str, Dict[str, Any]] = {}
        self.firestore_creators: Dict[str, Dict[str, Any]] = {}
        self.firestore_devices: Dict[str, Dict[str, Any]] = {}
        self.offline_cache: Dict[str, Any] = {}

    def set_online_status(self, online: bool) -> None:
        """Simulates network connectivity changes."""
        self.is_online = online

    def sync_creator_metadata(
        self,
        creator_id: str,
        display_name: str,
        public_key: str,
        key_version: int,
        status: str
    ) -> bool:
        """
        Synchronizes public, non-secret creator record to Firestore:
        creators/{creator_id}
        """
        # Security sanitization check: ensure no private keys or secrets are synced
        record = {
            "creator_id": creator_id,
            "display_name": display_name,
            "public_key": public_key,
            "key_version": key_version,
            "status": status,
            "synced_at": datetime.now(timezone.utc).isoformat()
        }

        # Always update local cache
        self.offline_cache[f"creators/{creator_id}"] = record

        if not self.is_online:
            # Queued offline
            return False

        self.firestore_creators[creator_id] = record
        return True

    def sync_device_metadata(
        self,
        creator_id: str,
        device_id: str,
        device_public_key: str,
        status: str,
        created_at: str,
        last_verified: str
    ) -> bool:
        """
        Synchronizes public device metadata to Firestore:
        creators/{creator_id}/devices/{device_id}
        """
        record = {
            "creator_id": creator_id,
            "device_id": device_id,
            "device_public_key": device_public_key,
            "status": status,
            "created_at": created_at,
            "last_verified": last_verified,
            "synced_at": datetime.now(timezone.utc).isoformat()
        }
        self.offline_cache[f"devices/{device_id}"] = record

        if not self.is_online:
            return False

        self.firestore_devices[f"{creator_id}/{device_id}"] = record
        return True

    def get_creator_metadata(self, creator_id: str) -> Optional[Dict[str, Any]]:
        """Reads metadata from Firestore or falls back to offline cache."""
        if self.is_online and creator_id in self.firestore_creators:
            return self.firestore_creators[creator_id]
        return self.offline_cache.get(f"creators/{creator_id}")

    def authenticate_user(self, uid: str, email: str, role: str = "USER", requested_role: Optional[str] = None) -> Dict[str, Any]:
        """
        Authenticates a normal user via Firebase Auth.
        Rule: Normal users cannot assign themselves 'creator' or 'root_creator'.
        """
        desired_role = requested_role or role
        # Security rule: client-side role requests are clamped to USER
        safe_role = "USER"
        if desired_role.upper() in ("CREATOR", "ROOT_CREATOR"):
            # Block privilege escalation attempt
            safe_role = "USER"

        user_record = {
            "uid": uid,
            "email": email,
            "role": safe_role,
            "creator_id": None,
            "account_status": "active"
        }
        self.firestore_users[uid] = user_record
        return user_record

    def update_user_profile(self, uid: str, updates: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Enforces Firestore rules: Users cannot modify 'role' to 'creator'
        or attach 'creator_id = ROOT_OPERATOR'.
        """
        if uid not in self.firestore_users:
            return False, "USER_NOT_FOUND"

        if "role" in updates and updates["role"].upper() in ("CREATOR", "ROOT_CREATOR"):
            return False, "PERMISSION_DENIED: Cannot escalate role to creator via client update"

        if "creator_id" in updates and updates["creator_id"] is not None:
            return False, "PERMISSION_DENIED: Cannot bind creator_id via client update"

        self.firestore_users[uid].update(updates)
        return True, "UPDATED"
