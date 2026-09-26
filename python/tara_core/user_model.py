"""
python/tara_core/user_model.py

Dynamic, expandable User and Profile Management for TARA.
Supports multi-user architectures with granular Role-Based Access Control (RBAC),
per-user memory scopes, preference tracking, and device bindings.
Enforces strict cryptographic distinction for the permanent Creator authority (ROOT_OPERATOR).
"""

import os
import json
import logging
import threading
from enum import Enum
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone

from TARA.ACCESS.operator.operator_profile import CANONICAL_CREATOR_ID, CreatorIdentity

logger = logging.getLogger("TARA.UserModel")


class UserRole(str, Enum):
    CREATOR = "CREATOR"
    ADMIN = "ADMIN"
    OPERATOR = "OPERATOR"
    USER = "USER"
    GUEST = "GUEST"


@dataclass
class UserProfile:
    user_id: str
    display_name: str
    role: UserRole = UserRole.USER
    permissions: List[str] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    memory_scope: str = ""
    device_bindings: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = "active"  # active, suspended, revoked

    def __post_init__(self):
        if not self.memory_scope:
            self.memory_scope = f"user_{self.user_id}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "display_name": self.display_name,
            "role": self.role.value if isinstance(self.role, UserRole) else str(self.role),
            "permissions": self.permissions,
            "preferences": self.preferences,
            "memory_scope": self.memory_scope,
            "device_bindings": self.device_bindings,
            "created_at": self.created_at,
            "status": self.status
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProfile":
        role_val = data.get("role", "USER")
        try:
            role = UserRole(role_val)
        except ValueError:
            role = UserRole.USER

        return cls(
            user_id=data["user_id"],
            display_name=data.get("display_name", data["user_id"]),
            role=role,
            permissions=data.get("permissions", []),
            preferences=data.get("preferences", {}),
            memory_scope=data.get("memory_scope", f"user_{data['user_id']}"),
            device_bindings=data.get("device_bindings", []),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            status=data.get("status", "active")
        )


class UserManager:
    """
    Manages open-ended users and profiles in TARA.
    Enforces that Creator authority cannot be forged or claimed by standard users.
    """
    _instance: Optional["UserManager"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self, storage_file: Optional[str] = None):
        if storage_file is None:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
            storage_file = os.path.join(repo_root, "TARA", "ACCESS", "users.json")
        self.storage_file = os.path.abspath(storage_file)
        os.makedirs(os.path.dirname(self.storage_file), exist_ok=True)
        self._users: Dict[str, UserProfile] = {}
        self._user_lock = threading.RLock()
        self._creator_identity = CreatorIdentity()
        self._load_users()

    @classmethod
    def get_default(cls, storage_file: Optional[str] = None) -> "UserManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(storage_file=storage_file)
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        with cls._lock:
            cls._instance = None

    def _load_users(self) -> None:
        with self._user_lock:
            if os.path.exists(self.storage_file):
                try:
                    with open(self.storage_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    for uid, udata in data.items():
                        self._users[uid] = UserProfile.from_dict(udata)
                except Exception as e:
                    logger.warning(f"Could not load users file: {e}")

            # Only populate canonical Creator profile if creator authority is verified initialized
            if self._creator_identity.is_initialized():
                if CANONICAL_CREATOR_ID not in self._users:
                    creator_profile = UserProfile(
                        user_id=CANONICAL_CREATOR_ID,
                        display_name=self._creator_identity.display_name or "OPERATOR_ROOT",
                        role=UserRole.CREATOR,
                        permissions=["*"],
                        preferences={"local_mode": True, "offline_first": True},
                        memory_scope="creator_scope",
                        status="active"
                    )
                    self._users[CANONICAL_CREATOR_ID] = creator_profile
                    self._save_users()
            else:
                if CANONICAL_CREATOR_ID in self._users:
                    del self._users[CANONICAL_CREATOR_ID]
                    self._save_users()

    def _save_users(self) -> None:
        with self._user_lock:
            data = {uid: p.to_dict() for uid, p in self._users.items()}
            with open(self.storage_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

    def register_user(self, profile: UserProfile, actor_id: str = CANONICAL_CREATOR_ID) -> UserProfile:
        """
        Registers a new user.
        Only CREATOR or ADMIN can create accounts with ADMIN/CREATOR privileges.
        Standard users cannot register themselves as CREATOR.
        """
        if profile.user_id == CANONICAL_CREATOR_ID and profile.role != UserRole.CREATOR:
            raise ValueError(f"User ID '{CANONICAL_CREATOR_ID}' is reserved for the Root Creator.")

        if profile.role == UserRole.CREATOR and profile.user_id != CANONICAL_CREATOR_ID:
            raise PermissionError("Only CANONICAL_CREATOR_ID can possess UserRole.CREATOR.")

        with self._user_lock:
            # Check calling actor permissions
            actor = self.get_user(actor_id)
            if profile.role in (UserRole.CREATOR, UserRole.ADMIN):
                if not actor or actor.role not in (UserRole.CREATOR, UserRole.ADMIN):
                    raise PermissionError("Privileged role creation requires Creator or Admin authorization.")

            self._users[profile.user_id] = profile
            self._save_users()
            logger.info(f"Registered user '{profile.user_id}' with role '{profile.role.value}'")
            return profile

    def get_user(self, user_id: str) -> Optional[UserProfile]:
        with self._user_lock:
            return self._users.get(user_id)

    def list_users(self) -> List[UserProfile]:
        with self._user_lock:
            return list(self._users.values())

    def update_preferences(self, user_id: str, preferences: Dict[str, Any]) -> bool:
        with self._user_lock:
            user = self.get_user(user_id)
            if user:
                user.preferences.update(preferences)
                self._save_users()
                return True
            return False

    def check_permission(self, user_id: str, required_permission: str) -> bool:
        """
        Evaluates permissions fail-closed.
        Creator has implicit wildcard '*'.
        """
        with self._user_lock:
            user = self.get_user(user_id)
            if not user or user.status != "active":
                return False

            if user.role == UserRole.CREATOR or "*" in user.permissions:
                return True

            if required_permission in user.permissions:
                return True

            # Domain check (e.g., 'tools:*' matches 'tools:execute')
            req_parts = required_permission.split(":")
            for p in user.permissions:
                p_parts = p.split(":")
                if len(p_parts) == 2 and p_parts[1] == "*" and p_parts[0] == req_parts[0]:
                    return True

            return False

    def is_creator(self, user_id: str) -> bool:
        return user_id == CANONICAL_CREATOR_ID

    def set_user_status(self, user_id: str, status: str, actor_id: str = CANONICAL_CREATOR_ID) -> bool:
        """Updates account status: ACTIVE, SUSPENDED, BLOCKED, DELETED."""
        valid_statuses = ("active", "suspended", "blocked", "deleted")
        if status.lower() not in valid_statuses:
            raise ValueError(f"Invalid status '{status}'. Must be one of {valid_statuses}")

        with self._user_lock:
            user = self.get_user(user_id)
            if not user:
                return False
            if user_id == CANONICAL_CREATOR_ID and status.lower() != "active":
                raise PermissionError("Root Creator account cannot be suspended or deleted.")

            if actor_id != CANONICAL_CREATOR_ID:
                actor = self.get_user(actor_id)
                if not actor or actor.role not in (UserRole.CREATOR, UserRole.ADMIN):
                    raise PermissionError("Only Creator or Admin can change account status.")

            user.status = status.lower()
            self._save_users()
            return True

    def delete_user(self, user_id: str, actor_id: str = CANONICAL_CREATOR_ID) -> bool:
        """Soft-deletes or marks user account as deleted."""
        return self.set_user_status(user_id, "deleted", actor_id=actor_id)

    def export_user_data(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Privacy export: returns user profile and metadata."""
        with self._user_lock:
            user = self.get_user(user_id)
            if not user:
                return None
            return {
                "profile": user.to_dict(),
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "data_controller": "TARA Core"
            }

