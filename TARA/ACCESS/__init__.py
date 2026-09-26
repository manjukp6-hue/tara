"""
TARA/ACCESS/__init__.py

TARA Root Operator Access & Security Subsystem.
Canonical Root Operator: ROOT_OPERATOR
Display Name: OPERATOR_ROOT
"""

from .access_manager import IdentityManager, IdentityManager as AccessManager
from .operator.operator_profile import CreatorIdentity, CreatorIdentity as OperatorProfile, CANONICAL_CREATOR_ID, DEFAULT_DISPLAY_NAME
from .devices.device_registry import DeviceRegistry
from .crypto.ed25519 import Ed25519
from .crypto.secure_storage import SecureKeyStorage
from .policy.access_policy import IdentityPolicy, TaraRole

__all__ = [
    "IdentityManager",
    "AccessManager",
    "CreatorIdentity",
    "OperatorProfile",
    "CANONICAL_CREATOR_ID",
    "DEFAULT_DISPLAY_NAME",
    "DeviceRegistry",
    "Ed25519",
    "SecureKeyStorage",
    "IdentityPolicy",
    "TaraRole"
]
