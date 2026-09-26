"""
TARA/ACCESS/protected_v1/__init__.py

Protected Creator Capability Subsystem for TARA.
Opaque Identifier: capability_v1 / protected_state_v1.
Provides additive privileged capabilities strictly for authenticated ROOT_CREATOR (ROOT_OPERATOR):
- Tamper-evident private verifier validation
- Action broker for authorized self-maintenance
- Persistent policy record store
- Existing lifecycle integration
- Command dispatcher
"""

from .manager import ProtectedStateManager, CAPABILITY_V1
from .broker import ActionBroker
from .policy_store import PolicyRecordStore
from .destruction_adapter import StateLifecycleAdapter
from .dispatcher import ProtectedCommandDispatcher

__all__ = [
    "ProtectedStateManager",
    "CAPABILITY_V1",
    "ActionBroker",
    "PolicyRecordStore",
    "StateLifecycleAdapter",
    "ProtectedCommandDispatcher",
]
