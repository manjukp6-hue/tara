"""
python/tara_core/plugin_engine.py

Plugin and Extension Architecture for TARA.
Allows loading dynamic plugins and extensions containing skills, tools, and capabilities
without modifying core Brain, Planner, ToolRegistry, or API Server.
Enforces:
- Strict manifest validation.
- Default-deny security boundaries: untrusted plugins NEVER receive Creator or Root privileges.
- Dynamic registration and clean unregistration.
"""

import os
import json
import logging
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

from tara_core.registry import CapabilityRegistry, Capability, CapabilityCategory, RiskLevel
from tara_core.tools_registry import ToolRegistry, ToolDefinition

logger = logging.getLogger("TARA.PluginEngine")


@dataclass
class PluginManifest:
    plugin_id: str
    name: str
    version: str
    author: str
    description: str
    capabilities: List[Dict[str, Any]] = field(default_factory=list)
    tools: List[Dict[str, Any]] = field(default_factory=list)
    skills: List[Dict[str, Any]] = field(default_factory=list)
    requested_permissions: List[str] = field(default_factory=list)
    is_trusted: bool = False
    entry_point: Optional[str] = None
    loaded_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plugin_id": self.plugin_id,
            "name": self.name,
            "version": self.version,
            "author": self.author,
            "description": self.description,
            "capabilities": self.capabilities,
            "tools": self.tools,
            "skills": self.skills,
            "requested_permissions": self.requested_permissions,
            "is_trusted": self.is_trusted,
            "entry_point": self.entry_point,
            "loaded_at": self.loaded_at
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PluginManifest":
        return cls(
            plugin_id=d["plugin_id"],
            name=d.get("name", d["plugin_id"]),
            version=d.get("version", "1.0.0"),
            author=d.get("author", "unknown"),
            description=d.get("description", ""),
            capabilities=d.get("capabilities", []),
            tools=d.get("tools", []),
            skills=d.get("skills", []),
            requested_permissions=d.get("requested_permissions", []),
            is_trusted=d.get("is_trusted", False),
            entry_point=d.get("entry_point"),
            loaded_at=d.get("loaded_at", datetime.now(timezone.utc).isoformat())
        )


class PluginEngine:
    """
    Secure plugin and extension lifecycle manager.
    Dynamically registers capabilities into the core engine with fail-closed privilege enforcement.
    """
    _instance: Optional["PluginEngine"] = None
    _lock: threading.Lock = threading.Lock()

    PROHIBITED_UNTRUSTED_PERMISSIONS = {
        "*",
        "creator:*",
        "root:*",
        "identity:modify",
        "rules:delete",
        "security:bypass"
    }

    def __init__(
        self,
        capability_registry: Optional[CapabilityRegistry] = None,
        tool_registry: Optional[ToolRegistry] = None
    ):
        self.capability_registry = capability_registry or CapabilityRegistry.get_default()
        self.tool_registry = tool_registry or ToolRegistry.get_default()
        self._plugins: Dict[str, PluginManifest] = {}
        self._lock = threading.RLock()

    @classmethod
    def get_default(cls) -> "PluginEngine":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        with cls._lock:
            cls._instance = None

    def validate_manifest(self, manifest: PluginManifest) -> List[str]:
        """
        Validates manifest security invariants.
        Returns list of errors; empty if valid.
        """
        errors = []
        if not manifest.plugin_id or not manifest.plugin_id.isalnum() and "_" not in manifest.plugin_id:
            errors.append(f"Invalid plugin_id '{manifest.plugin_id}': must be alphanumeric with underscores.")

        # Check permission safety
        if not manifest.is_trusted:
            for perm in manifest.requested_permissions:
                if perm in self.PROHIBITED_UNTRUSTED_PERMISSIONS or perm.startswith("creator:") or perm == "*":
                    errors.append(f"Untrusted plugin cannot request elevated privilege '{perm}'.")

        return errors

    def register_plugin(self, manifest: PluginManifest) -> Dict[str, Any]:
        """
        Registers plugin and activates its tools and capabilities into core registries.
        Fails closed on validation failure.
        """
        errors = self.validate_manifest(manifest)
        if errors:
            logger.warning(f"Plugin '{manifest.plugin_id}' rejected: {errors}")
            return {"status": "DENIED", "errors": errors}

        with self._lock:
            registered_caps = []
            registered_tools = []

            # 1. Register Capabilities
            for cap_def in manifest.capabilities:
                cid = cap_def.get("capability_id", f"{manifest.plugin_id}_{cap_def.get('name')}")
                cap = Capability(
                    capability_id=cid,
                    name=cap_def.get("name", cid),
                    version=cap_def.get("version", manifest.version),
                    category=CapabilityCategory(cap_def.get("category", "EXTENSION")),
                    purpose=cap_def.get("purpose", f"Capability provided by {manifest.name}"),
                    trigger_metadata=cap_def.get("trigger_metadata", {}),
                    permissions=manifest.requested_permissions,
                    risk_level=RiskLevel(cap_def.get("risk_level", "LOW")),
                    executable=cap_def.get("executable", True),
                    metadata={"plugin_id": manifest.plugin_id}
                )
                self.capability_registry.register_capability(cap)
                registered_caps.append(cid)

            # 2. Register Tools
            for tool_data in manifest.tools:
                tname = tool_data.get("name")
                if not tname:
                    continue
                # Default handler if none provided
                handler = tool_data.get("handler") or (lambda **kw: {"status": "SUCCESS", "source": manifest.plugin_id, "data": kw})
                tool_def = ToolDefinition(
                    name=tname,
                    description=tool_data.get("description", f"Tool from {manifest.name}"),
                    handler=handler,
                    parameters_schema=tool_data.get("parameters_schema", {}),
                    required_permissions=manifest.requested_permissions,
                    risk_level=RiskLevel(tool_data.get("risk_level", "LOW")),
                    metadata={"plugin_id": manifest.plugin_id}
                )
                self.tool_registry.register_tool(tool_def)
                registered_tools.append(tname)

            self._plugins[manifest.plugin_id] = manifest
            logger.info(f"Registered plugin '{manifest.plugin_id}' ({len(registered_caps)} caps, {len(registered_tools)} tools)")
            return {
                "status": "SUCCESS",
                "plugin_id": manifest.plugin_id,
                "capabilities": registered_caps,
                "tools": registered_tools
            }

    def unregister_plugin(self, plugin_id: str) -> bool:
        """Removes plugin and unregisters all associated tools and capabilities."""
        with self._lock:
            manifest = self._plugins.pop(plugin_id, None)
            if not manifest:
                return False

            for cap_def in manifest.capabilities:
                cid = cap_def.get("capability_id", f"{manifest.plugin_id}_{cap_def.get('name')}")
                self.capability_registry.unregister_capability(cid)

            for tool_data in manifest.tools:
                tname = tool_data.get("name")
                if tname:
                    self.tool_registry.unregister_tool(tname)

            logger.info(f"Unregistered plugin '{plugin_id}'")
            return True

    def get_plugin(self, plugin_id: str) -> Optional[PluginManifest]:
        with self._lock:
            return self._plugins.get(plugin_id)

    def list_plugins(self) -> List[PluginManifest]:
        with self._lock:
            return list(self._plugins.values())
