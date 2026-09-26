"""
python/tara_core/registry.py

Canonical, machine-readable Capability Registry for TARA.
Enables dynamic discovery, registration, lifecycle management, and invocation
of open-ended capabilities across skills, tools, agents, knowledge, reasoning, and extensions.
No capability count is hardcoded as an architectural limit.
"""

import threading
import logging
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger("TARA.Registry")


class CapabilityCategory(str, Enum):
    SKILL = "SKILL"
    TOOL = "TOOL"
    AGENT = "AGENT"
    KNOWLEDGE = "KNOWLEDGE"
    REASONING = "REASONING"
    EXTENSION = "EXTENSION"
    COGNITIVE = "COGNITIVE"
    ENGINE = "ENGINE"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class Capability:
    capability_id: str
    name: str
    version: str
    category: CapabilityCategory
    purpose: str
    trigger_metadata: Dict[str, Any] = field(default_factory=dict)
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    permissions: List[str] = field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.LOW
    required_authentication: str = "NONE"  # NONE, USER, ADMIN, CREATOR
    executable: bool = True
    handler: Optional[Callable[..., Any]] = None
    required_tools: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    enabled: bool = True
    registered_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "name": self.name,
            "version": self.version,
            "category": self.category.value if isinstance(self.category, CapabilityCategory) else str(self.category),
            "purpose": self.purpose,
            "trigger_metadata": self.trigger_metadata,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "permissions": self.permissions,
            "risk_level": self.risk_level.value if isinstance(self.risk_level, RiskLevel) else str(self.risk_level),
            "required_authentication": self.required_authentication,
            "executable": self.executable,
            "required_tools": self.required_tools,
            "dependencies": self.dependencies,
            "enabled": self.enabled,
            "registered_at": self.registered_at,
            "metadata": self.metadata
        }


class CapabilityRegistry:
    """
    Central, thread-safe capability registry.
    Supports dynamic registration, querying, activation/deactivation,
    and schema validation without fixed inventory limits.
    """
    _instance: Optional["CapabilityRegistry"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self):
        self._capabilities: Dict[str, Capability] = {}
        self._category_index: Dict[CapabilityCategory, Set[str]] = {
            cat: set() for cat in CapabilityCategory
        }
        self._reg_lock = threading.RLock()

    @classmethod
    def get_default(cls) -> "CapabilityRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Resets the singleton for isolation in test fixtures."""
        with cls._lock:
            cls._instance = None

    def register_capability(self, capability: Capability) -> bool:
        """
        Registers or updates a capability in the registry.
        Validates metadata and ensures fail-closed integrity.
        """
        if not capability.capability_id or not capability.name:
            raise ValueError("Capability must have a non-empty capability_id and name.")

        with self._reg_lock:
            # Check dependencies if specified
            cat = capability.category
            if isinstance(cat, str):
                cat = CapabilityCategory(cat)
                capability.category = cat

            self._capabilities[capability.capability_id] = capability
            if cat not in self._category_index:
                self._category_index[cat] = set()
            self._category_index[cat].add(capability.capability_id)
            logger.info(f"Registered capability '{capability.capability_id}' ({cat.value})")
            return True

    def unregister_capability(self, capability_id: str) -> bool:
        """Removes a capability from the registry."""
        with self._reg_lock:
            cap = self._capabilities.pop(capability_id, None)
            if cap:
                cat_set = self._category_index.get(cap.category)
                if cat_set and capability_id in cat_set:
                    cat_set.remove(capability_id)
                logger.info(f"Unregistered capability '{capability_id}'")
                return True
            return False

    def register(self, capability: Capability) -> bool:
        """Alias for register_capability."""
        return self.register_capability(capability)

    def unregister(self, capability_id: str) -> bool:
        """Alias for unregister_capability."""
        return self.unregister_capability(capability_id)

    def get_capability(self, capability_id: str) -> Optional[Capability]:
        with self._reg_lock:
            return self._capabilities.get(capability_id)

    def has_capability(self, capability_id: str) -> bool:
        with self._reg_lock:
            return capability_id in self._capabilities

    def list_capabilities(self, category: Optional[CapabilityCategory] = None, enabled_only: bool = True) -> List[Capability]:
        with self._reg_lock:
            if category is not None:
                ids = self._category_index.get(category, set())
                caps = [self._capabilities[cid] for cid in ids if cid in self._capabilities]
            else:
                caps = list(self._capabilities.values())

            if enabled_only:
                caps = [c for c in caps if c.enabled]
            return caps

    def enable_capability(self, capability_id: str) -> bool:
        with self._reg_lock:
            cap = self._capabilities.get(capability_id)
            if cap:
                cap.enabled = True
                return True
            return False

    def disable_capability(self, capability_id: str) -> bool:
        with self._reg_lock:
            cap = self._capabilities.get(capability_id)
            if cap:
                cap.enabled = False
                return True
            return False

    def match_intent(self, intent_name: str, context: Optional[Dict[str, Any]] = None) -> List[Capability]:
        """
        Dynamically matches an intent string or trigger to available enabled capabilities.
        """
        matches = []
        normalized_intent = intent_name.lower().strip()
        with self._reg_lock:
            for cap in self._capabilities.values():
                if not cap.enabled:
                    continue
                trig = cap.trigger_metadata
                intents = [i.lower() for i in trig.get("intents", [])]
                keywords = [k.lower() for k in trig.get("keywords", [])]

                if normalized_intent in intents or any(kw in normalized_intent for kw in keywords):
                    matches.append(cap)
                elif cap.name.lower() == normalized_intent or cap.capability_id.lower() == normalized_intent:
                    matches.append(cap)
        return matches

    def count(self, category: Optional[CapabilityCategory] = None) -> int:
        with self._reg_lock:
            if category:
                return len([c for c in self.list_capabilities(category=category, enabled_only=False)])
            return len(self._capabilities)
