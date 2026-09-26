"""
python/tara_core/tools_registry.py

Dynamic, schema-validated Tool Registry for TARA.
Decouples tool definitions and execution from the core brain.
New tools register dynamically with input schemas, permission checks,
and execution handlers without modifying brain.py or requiring fixed counts.
"""

import os
import sys
import inspect
import logging
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

_pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

try:
    from tara_core.registry import CapabilityRegistry, Capability, CapabilityCategory, RiskLevel
except ImportError:
    from python.tara_core.registry import CapabilityRegistry, Capability, CapabilityCategory, RiskLevel

# Baseline tools import
from TARA.TOOLS.file_inspector import inspect_file
from TARA.TOOLS.hash_verifier import compute_hash, verify_file_hash
from TARA.TOOLS.knowledge_retriever import search_knowledge
from TARA.TOOLS.provenance_tracker import verify_provenance, create_provenance_record

logger = logging.getLogger("TARA.ToolsRegistry")


@dataclass
class ToolDefinition:
    name: str
    description: str
    handler: Callable[..., Any]
    parameters_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    required_permissions: List[str] = field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.LOW
    required_authentication: str = "NONE"
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters_schema": self.parameters_schema,
            "output_schema": self.output_schema,
            "required_permissions": self.required_permissions,
            "risk_level": self.risk_level.value if isinstance(self.risk_level, RiskLevel) else str(self.risk_level),
            "required_authentication": self.required_authentication,
            "enabled": self.enabled,
            "metadata": self.metadata
        }


class ToolRegistry:
    """
    Registry for dynamic tools in TARA.
    Allows runtime registration, listing, schema verification, and dispatch.
    """
    _instance: Optional["ToolRegistry"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self, repo_root: Optional[str] = None):
        self.repo_root = repo_root or os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        self._tools: Dict[str, ToolDefinition] = {}
        self._tool_lock = threading.RLock()
        self._capability_registry = CapabilityRegistry.get_default()
        self._register_baseline_tools()

    @classmethod
    def get_default(cls, repo_root: Optional[str] = None) -> "ToolRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(repo_root=repo_root)
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        with cls._lock:
            cls._instance = None

    def register_tool(self, tool_def: ToolDefinition, sync_capability: bool = True) -> bool:
        if not tool_def.name:
            raise ValueError("Tool name must not be empty.")
        with self._tool_lock:
            self._tools[tool_def.name] = tool_def
            if sync_capability:
                cap = Capability(
                    capability_id=f"tool_{tool_def.name}",
                    name=tool_def.name,
                    version=tool_def.metadata.get("version", "1.0.0"),
                    category=CapabilityCategory.TOOL,
                    purpose=tool_def.description,
                    trigger_metadata={
                        "intents": [tool_def.name, f"execute_{tool_def.name}", f"run_{tool_def.name}"],
                        "keywords": [tool_def.name] + tool_def.metadata.get("keywords", [])
                    },
                    input_schema=tool_def.parameters_schema,
                    output_schema=tool_def.output_schema,
                    permissions=tool_def.required_permissions,
                    risk_level=tool_def.risk_level,
                    required_authentication=tool_def.required_authentication,
                    executable=True,
                    handler=tool_def.handler,
                    enabled=tool_def.enabled,
                    metadata=tool_def.metadata
                )
                self._capability_registry.register_capability(cap)
            logger.info(f"Registered tool '{tool_def.name}'")
            return True

    def unregister_tool(self, tool_name: str) -> bool:
        with self._tool_lock:
            if tool_name in self._tools:
                del self._tools[tool_name]
                self._capability_registry.unregister_capability(f"tool_{tool_name}")
                logger.info(f"Unregistered tool '{tool_name}'")
                return True
            return False

    def enable_tool(self, tool_name: str) -> bool:
        with self._tool_lock:
            tool = self._tools.get(tool_name)
            if tool:
                tool.enabled = True
                self._capability_registry.enable_capability(f"tool_{tool_name}")
                return True
            return False

    def disable_tool(self, tool_name: str) -> bool:
        with self._tool_lock:
            tool = self._tools.get(tool_name)
            if tool:
                tool.enabled = False
                self._capability_registry.disable_capability(f"tool_{tool_name}")
                return True
            return False

    def get_tool(self, tool_name: str) -> Optional[ToolDefinition]:
        with self._tool_lock:
            return self._tools.get(tool_name)

    def list_tools(self, enabled_only: bool = True) -> List[ToolDefinition]:
        with self._tool_lock:
            if enabled_only:
                return [t for t in self._tools.values() if t.enabled]
            return list(self._tools.values())

    def execute_tool(self, tool_name: str, params: Dict[str, Any], actor_id: str = "default_user", **kwargs) -> Dict[str, Any]:
        """
        Executes a registered tool dynamically with input parameters.
        Fails closed with structured status if tool is not found, disabled, or fails.
        """
        with self._tool_lock:
            tool = self._tools.get(tool_name)
        
        if not tool:
            return {
                "status": "ERROR",
                "error": f"Tool '{tool_name}' is not registered in ToolRegistry.",
                "exists": False
            }

        if not tool.enabled:
            return {
                "status": "BLOCKED_BY_POLICY",
                "reason": f"Tool '{tool_name}' is currently disabled.",
                "decision": "DENY"
            }

        try:
            handler = tool.handler
            sig = inspect.signature(handler)
            call_args = {}
            for param_name in sig.parameters:
                if param_name in params:
                    call_args[param_name] = params[param_name]
                elif param_name == "actor_id":
                    call_args["actor_id"] = actor_id
                elif param_name == "repo_root":
                    call_args["repo_root"] = kwargs.get("repo_root", self.repo_root)
                elif param_name in kwargs:
                    call_args[param_name] = kwargs[param_name]
                elif param_name == "params":
                    call_args["params"] = params
                elif sig.parameters[param_name].default is not inspect.Parameter.empty:
                    pass  # use default
                else:
                    call_args[param_name] = params.get(param_name)

            res = handler(**call_args)
            if not isinstance(res, dict):
                return {"status": "SUCCESS", "result": res}
            if "status" not in res:
                res["status"] = "SUCCESS"
            return res
        except Exception as e:
            logger.error(f"Error executing tool '{tool_name}': {e}", exc_info=True)
            return {
                "status": "ERROR",
                "error": str(e),
                "tool_name": tool_name
            }

    # ------------------------------------------------------------------------
    # Baseline tools setup
    # ------------------------------------------------------------------------
    def _register_baseline_tools(self) -> None:
        # 1. file_inspector
        def _exec_file_inspector(file_path: str = "", repo_root: Optional[str] = None, **kw):
            return inspect_file(file_path, repo_root=repo_root or self.repo_root)

        self.register_tool(ToolDefinition(
            name="file_inspector",
            description="Safe metadata inspection with secret boundary isolation and path validation.",
            handler=_exec_file_inspector,
            parameters_schema={
                "type": "object",
                "properties": {"file_path": {"type": "string"}},
                "required": ["file_path"]
            },
            risk_level=RiskLevel.LOW
        ))

        # 2. hash_verifier
        def _exec_hash_verifier(file_path: Optional[str] = None, expected_hash: Optional[str] = None,
                                algorithm: str = "sha256", data: Optional[Any] = None,
                                repo_root: Optional[str] = None, **kw):
            target_repo = repo_root or self.repo_root
            if file_path:
                target_path = file_path
                if not os.path.isabs(target_path) and not os.path.exists(target_path):
                    target_path = os.path.join(target_repo, target_path)
                if expected_hash is not None:
                    res = verify_file_hash(
                        file_path=target_path,
                        expected_hash=expected_hash,
                        algorithm=algorithm
                    )
                    res["status"] = "SUCCESS" if res.get("verified") else "FAILURE"
                    if "hash" not in res and "computed_hash" in res:
                        res["hash"] = res["computed_hash"]
                    return res
                else:
                    if not os.path.exists(target_path):
                        return {"status": "ERROR", "error": f"File '{file_path}' does not exist"}
                    with open(target_path, "rb") as f:
                        content = f.read()
                    digest = compute_hash(content, algorithm=algorithm)
                    return {"status": "SUCCESS", "hash": digest, "file_path": file_path, "algorithm": algorithm}
            else:
                data_bytes = data or b""
                if isinstance(data_bytes, str):
                    data_bytes = data_bytes.encode("utf-8")
                digest = compute_hash(data_bytes, algorithm=algorithm)
                return {"status": "SUCCESS", "hash": digest, "algorithm": algorithm}

        self.register_tool(ToolDefinition(
            name="hash_verifier",
            description="Cryptographic checksum and integrity validation for files or memory data.",
            handler=_exec_hash_verifier,
            parameters_schema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "expected_hash": {"type": "string"},
                    "algorithm": {"type": "string", "default": "sha256"},
                    "data": {"type": "string"}
                }
            },
            risk_level=RiskLevel.LOW
        ))

        # 3. knowledge_retriever
        def _exec_knowledge_retriever(query: str = "", topic: Optional[str] = None,
                                      min_confidence: float = 0.0, verified_only: bool = False, **kw):
            return search_knowledge(
                query=query,
                topic=topic,
                min_confidence=min_confidence,
                verified_only=verified_only
            )

        self.register_tool(ToolDefinition(
            name="knowledge_retriever",
            description="Retrieves verified facts and citations from local TARA AI knowledge base.",
            handler=_exec_knowledge_retriever,
            parameters_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "topic": {"type": "string"},
                    "min_confidence": {"type": "number", "default": 0.0},
                    "verified_only": {"type": "boolean", "default": False}
                },
                "required": ["query"]
            },
            risk_level=RiskLevel.LOW
        ))

        # 4. provenance_tracker
        def _exec_provenance_tracker(actor_id: str = "default_user",
                                     actor_role: str = "USER",
                                     action: str = "tara_brain_action",
                                     target: str = "operation",
                                     session_id: Optional[str] = None,
                                     details: Optional[Dict[str, Any]] = None,
                                     model_file: Optional[str] = None,
                                     training_metadata_file: Optional[str] = None,
                                     repo_root: Optional[str] = None, **kw):
            if model_file or training_metadata_file:
                target_repo = repo_root or self.repo_root
                m_path = model_file or os.path.join(target_repo, "storage", "models", "tara", "model.safetensors")
                t_path = training_metadata_file or os.path.join(target_repo, "storage", "models", "tara", "training_metadata.json")
                return verify_provenance(model_path=m_path, metadata_path=t_path)
            return create_provenance_record(
                actor_id=actor_id,
                actor_role=actor_role,
                action=action,
                target=target,
                session_id=session_id,
                details=details
            )

        self.register_tool(ToolDefinition(
            name="provenance_tracker",
            description="Validates cryptographic provenance of model weights, training runs, and datasets.",
            handler=_exec_provenance_tracker,
            parameters_schema={
                "type": "object",
                "properties": {
                    "model_file": {"type": "string"},
                    "training_metadata_file": {"type": "string"}
                }
            },
            risk_level=RiskLevel.LOW
        ))
