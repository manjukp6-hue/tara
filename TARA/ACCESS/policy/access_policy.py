"""
TARA/ACCESS/policy/identity_policy.py

TARA Role Hierarchy & Authorization Policy.

Role Hierarchy:
ROOT CREATOR (Rank 4) -> CREATOR (Rank 3) -> ADMIN (Rank 2) -> STAFF (Rank 1) -> USER (Rank 0)

Security Rules:
- Only verified ROOT_OPERATOR root identity receives ROOT CREATOR / CREATOR authority.
- Normal Firebase user != Creator.
- Google account alone != Creator.
- Creator ID string alone != Creator.
- Cryptographic proof or authorized device token required for Creator authority.
- Offline creator operations permitted for authorized devices according to local policy.
"""

from enum import IntEnum
from typing import Dict, Optional, Any, List

from ..operator.operator_profile import CANONICAL_CREATOR_ID


class TaraRole(IntEnum):
    USER = 0
    STAFF = 1
    ADMIN = 2
    CREATOR = 3
    ROOT_CREATOR = 4


class IdentityPolicy:
    """
    Evaluates permissions and enforces strict creator verification rules.
    """
    @staticmethod
    def is_root_creator_identity(creator_id: str) -> bool:
        """Verifies if candidate ID matches the canonical creator identity."""
        return creator_id == CANONICAL_CREATOR_ID

    @staticmethod
    def evaluate_creator_permission(
        claimed_creator_id: str,
        is_device_authorized: bool,
        cryptographic_proof_valid: bool,
        biometric_valid: Optional[bool] = None,
        is_offline: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluates whether an entity possesses Creator / Root Creator authority.
        """
        # Rule 1: Creator ID must match ROOT_OPERATOR
        if not IdentityPolicy.is_root_creator_identity(claimed_creator_id):
            return {
                "authorized": False,
                "role": TaraRole.USER,
                "reason": "INVALID_CREATOR_ID: Candidate ID does not match canonical root creator"
            }

        # Rule 2: Device must be authorized
        if not is_device_authorized:
            return {
                "authorized": False,
                "role": TaraRole.USER,
                "reason": "UNAUTHORIZED_DEVICE: Device is not registered or has been revoked"
            }

        # Rule 3: Must provide valid cryptographic proof
        if not cryptographic_proof_valid:
            return {
                "authorized": False,
                "role": TaraRole.USER,
                "reason": "INVALID_CRYPTOGRAPHIC_PROOF: Signature verification failed"
            }

        # Rule 4: If biometric is required and failed, deny or fallback
        if biometric_valid is False:
            return {
                "authorized": False,
                "role": TaraRole.USER,
                "reason": "BIOMETRIC_CHECK_FAILED"
            }

        # Valid creator authority granted
        return {
            "authorized": True,
            "role": TaraRole.ROOT_CREATOR,
            "offline_execution": is_offline,
            "reason": "AUTHORIZED_ROOT_CREATOR"
        }

    @staticmethod
    def can_modify_system_model(role: TaraRole) -> bool:
        return role >= TaraRole.CREATOR

    @staticmethod
    def can_manage_devices(role: TaraRole) -> bool:
        return role >= TaraRole.CREATOR

    @staticmethod
    def can_perform_key_rotation(role: TaraRole) -> bool:
        return role >= TaraRole.ROOT_CREATOR
