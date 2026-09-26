"""
python/tara_core/control_plane/worker_registry.py

Dynamic Worker Registry & Lifecycle State Machine for TARA Control Plane.
Manages compute nodes across serverless, container, GPU, CPU, and local devices.
Implements the 7-step worker lifecycle:
DISCOVER -> AUTHENTICATE -> REGISTER -> SYNC -> VERIFY -> HEALTH CHECK -> READY
with support for QUARANTINED and REVOKED security states.
"""

import os
import sys
import time
import json
import uuid
import logging
import threading
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tara_core.contracts import (
    CANONICAL_MODEL_SHA256,
    CANONICAL_PROTOCOL_VERSION,
    CANONICAL_MODEL_IDENTITY,
    CANONICAL_PARAM_COUNT,
)

logger = logging.getLogger("tara_core.control_plane.worker_registry")


class WorkerState(str, Enum):
    DISCOVERED = "DISCOVERED"
    AUTHENTICATING = "AUTHENTICATING"
    SYNCING = "SYNCING"
    READY = "READY"
    BUSY = "BUSY"
    DEGRADED = "DEGRADED"
    UNAVAILABLE = "UNAVAILABLE"
    QUARANTINED = "QUARANTINED"
    REVOKED = "REVOKED"


@dataclass
class ClusterWorkerNode:
    node_id: str
    endpoint_url: str
    provider: str
    deployment_type: str
    public_key_hex: Optional[str] = None
    auth_token: Optional[str] = None

    # Hardware capabilities & limits
    cpu_capacity: float = 1.0
    ram_mb: int = 1024
    has_gpu: bool = False
    gpu_model: Optional[str] = None
    gpu_vram_mb: int = 0
    storage_free_mb: int = 5000

    # Real-time node load
    current_load: float = 0.0      # 0.0 = idle, 1.0 = saturated
    queue_depth: int = 0
    average_latency_ms: float = 0.0

    # Workload & version compatibility
    supported_workloads: List[str] = field(default_factory=lambda: ["NORMAL_CHAT", "INFERENCE"])
    runtime_version: str = "1.0.0"
    protocol_version: str = CANONICAL_PROTOCOL_VERSION
    model_identity: str = CANONICAL_MODEL_IDENTITY
    model_version: str = "TARA"
    model_sha256: str = CANONICAL_MODEL_SHA256
    skills_compatibility: List[str] = field(default_factory=list)
    security_policy_version: str = "1.0.0"

    # Lifecycle state
    state: WorkerState = WorkerState.DISCOVERED
    security_state: str = "ACTIVE"
    capabilities: Dict[str, Any] = field(default_factory=dict)
    registered_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    failure_count: int = 0
    quarantine_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["state"] = self.state.value if isinstance(self.state, WorkerState) else str(self.state)
        # Never leak auth token or secrets in serialization
        d.pop("auth_token", None)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClusterWorkerNode":
        state_str = data.get("state", "DISCOVERED")
        try:
            state = WorkerState(state_str)
        except ValueError:
            state = WorkerState.DISCOVERED

        return cls(
            node_id=str(data["node_id"]),
            endpoint_url=str(data.get("endpoint_url", "")),
            provider=str(data.get("provider", "generic_container")),
            deployment_type=str(data.get("deployment_type", "CONTAINER")),
            public_key_hex=data.get("public_key_hex"),
            auth_token=data.get("auth_token"),
            cpu_capacity=float(data.get("cpu_capacity", 1.0)),
            ram_mb=int(data.get("ram_mb", 1024)),
            has_gpu=bool(data.get("has_gpu", False)),
            gpu_model=data.get("gpu_model"),
            gpu_vram_mb=int(data.get("gpu_vram_mb", 0)),
            storage_free_mb=int(data.get("storage_free_mb", 5000)),
            current_load=float(data.get("current_load", 0.0)),
            queue_depth=int(data.get("queue_depth", 0)),
            average_latency_ms=float(data.get("average_latency_ms", 0.0)),
            supported_workloads=list(data.get("supported_workloads", ["NORMAL_CHAT", "INFERENCE"])),
            runtime_version=str(data.get("runtime_version", "1.0.0")),
            protocol_version=str(data.get("protocol_version", CANONICAL_PROTOCOL_VERSION)),
            model_identity=str(data.get("model_identity", CANONICAL_MODEL_IDENTITY)),
            model_version=str(data.get("model_version", "TARA")),
            model_sha256=str(data.get("model_sha256", CANONICAL_MODEL_SHA256)),
            skills_compatibility=list(data.get("skills_compatibility", [])),
            security_policy_version=str(data.get("security_policy_version", "1.0.0")),
            state=state,
            security_state=str(data.get("security_state", "ACTIVE")),
            capabilities=dict(data.get("capabilities", {})),
            registered_at=float(data.get("registered_at", time.time())),
            last_heartbeat=float(data.get("last_heartbeat", time.time())),
            failure_count=int(data.get("failure_count", 0)),
            quarantine_reason=data.get("quarantine_reason")
        )


class DynamicWorkerRegistry:
    """
    Authoritative, thread-safe cluster worker registry.
    Governs node lifecycle, validation, quarantine, revocation, and capability discovery.
    """

    def __init__(self, persistence_file: Optional[str] = None):
        self._lock = threading.RLock()
        self._nodes: Dict[str, ClusterWorkerNode] = {}
        if persistence_file is None:
            self.persistence_file = os.path.join(REPO_ROOT, "TARA", "ACCESS", "registered_endpoints.json")
        else:
            self.persistence_file = persistence_file
        self._load_from_storage()

    def _load_from_storage(self):
        with self._lock:
            if os.path.exists(self.persistence_file):
                try:
                    with open(self.persistence_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    for item in data.get("nodes", []):
                        node = ClusterWorkerNode.from_dict(item)
                        # Reset transient states on reload
                        if node.state in (WorkerState.READY, WorkerState.BUSY):
                            node.state = WorkerState.DISCOVERED
                        self._nodes[node.node_id] = node
                except Exception as e:
                    logger.warning(f"Could not load worker registry from storage: {e}")

    def _persist(self):
        try:
            os.makedirs(os.path.dirname(self.persistence_file), exist_ok=True)
            with open(self.persistence_file, "w", encoding="utf-8") as f:
                nodes_data = [node.to_dict() for node in self._nodes.values() if node.state != WorkerState.REVOKED]
                json.dump({"version": "1.0", "updated_at": time.time(), "nodes": nodes_data}, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist worker registry: {e}")

    def discover_node(self, node_data: Dict[str, Any]) -> ClusterWorkerNode:
        """Step 1: DISCOVER — Adds or updates an unverified candidate node."""
        node_id = str(node_data.get("node_id") or f"node_{uuid.uuid4().hex[:8]}")
        with self._lock:
            existing = self._nodes.get(node_id)
            if existing and existing.state == WorkerState.REVOKED:
                raise PermissionError(f"Node '{node_id}' is permanently revoked.")

            node = ClusterWorkerNode.from_dict({**node_data, "node_id": node_id, "state": WorkerState.DISCOVERED.value})
            self._nodes[node_id] = node
            return node

    def authenticate_node(self, node_id: str, auth_token: str, expected_token: str) -> bool:
        """Step 2: AUTHENTICATE — Validates internal worker credential before trust is granted."""
        with self._lock:
            node = self._nodes.get(node_id)
            if not node:
                return False
            if node.state in (WorkerState.QUARANTINED, WorkerState.REVOKED):
                return False

            if auth_token != expected_token:
                node.failure_count += 1
                if node.failure_count >= 3:
                    node.state = WorkerState.QUARANTINED
                    node.quarantine_reason = "Repeated authentication failures"
                return False

            node.state = WorkerState.SYNCING
            node.auth_token = auth_token
            return True

    def register_node(self, node_data: Dict[str, Any], auth_token: str, expected_token: str) -> ClusterWorkerNode:
        """Convenience method executing DISCOVER and AUTHENTICATE."""
        node = self.discover_node(node_data)
        if not self.authenticate_node(node.node_id, auth_token, expected_token):
            raise PermissionError(f"Worker authentication failed for node '{node.node_id}'")
        return node

    def update_sync_state(self, node_id: str, state: WorkerState) -> bool:
        """Step 4: SYNC — Updates worker syncing progress."""
        with self._lock:
            node = self._nodes.get(node_id)
            if not node or node.state in (WorkerState.QUARANTINED, WorkerState.REVOKED):
                return False
            node.state = state
            return True

    def verify_and_set_ready(
        self,
        node_id: str,
        reported_model_sha256: str,
        reported_protocol_version: str
    ) -> bool:
        """
        Step 5 & 6: VERIFY & HEALTH CHECK -> READY.
        Strictly enforces that the worker holds the verified production SafeTensors checksum
        and compatible protocol version. Rejects node if checksum drifts.
        """
        with self._lock:
            node = self._nodes.get(node_id)
            if not node:
                return False
            if node.state in (WorkerState.QUARANTINED, WorkerState.REVOKED):
                return False

            # Model SHA256 Invariant Check
            if reported_model_sha256.lower() != CANONICAL_MODEL_SHA256.lower():
                node.state = WorkerState.QUARANTINED
                node.security_state = "QUARANTINED"
                node.capabilities = {"inference": False, "network": "none", "filesystem": "none"}
                node.quarantine_reason = f"Model SHA256 mismatch: got {reported_model_sha256}, expected {CANONICAL_MODEL_SHA256}"
                logger.error(f"Node '{node_id}' rejected from READY: {node.quarantine_reason}")
                return False

            # Protocol Version Invariant Check
            if reported_protocol_version != CANONICAL_PROTOCOL_VERSION:
                node.state = WorkerState.DEGRADED
                node.security_state = "SUSPICIOUS"
                node.quarantine_reason = f"Protocol version mismatch: got {reported_protocol_version}, expected {CANONICAL_PROTOCOL_VERSION}"
                return False

            node.state = WorkerState.READY
            node.security_state = "ACTIVE"
            node.capabilities = {"inference": True, "network": "restricted", "filesystem": "temporary-job-only"}
            node.last_heartbeat = time.time()
            node.failure_count = 0
            node.quarantine_reason = None
            self._persist()
            return True

    def heartbeat(
        self,
        node_id: str,
        current_load: float = 0.0,
        queue_depth: int = 0,
        latency_ms: float = 0.0
    ) -> bool:
        """Updates real-time telemetry on a node."""
        with self._lock:
            node = self._nodes.get(node_id)
            if not node:
                return False
            if node.state in (WorkerState.QUARANTINED, WorkerState.REVOKED):
                return False

            node.last_heartbeat = time.time()
            node.current_load = max(0.0, min(1.0, current_load))
            node.queue_depth = max(0, queue_depth)
            if latency_ms > 0:
                node.average_latency_ms = (node.average_latency_ms * 0.7) + (latency_ms * 0.3)
            return True

    def quarantine_worker(self, node_id: str, reason: str = "Quarantined by policy") -> bool:
        """Isolates a worker immediately to prevent task allocation."""
        with self._lock:
            node = self._nodes.get(node_id)
            if not node:
                return False
            node.state = WorkerState.QUARANTINED
            node.security_state = "QUARANTINED"
            node.capabilities = {"inference": False, "network": "none", "filesystem": "none"}
            node.quarantine_reason = reason
            self._persist()
            logger.warning(f"Worker '{node_id}' QUARANTINED: {reason}")
            return True

    def revoke_worker(self, node_id: str, reason: str = "Permanent revocation") -> bool:
        """Permanently revokes a compromised or decommissioned worker."""
        with self._lock:
            node = self._nodes.get(node_id)
            if not node:
                return False
            node.state = WorkerState.REVOKED
            node.security_state = "QUARANTINED"
            node.capabilities = {"inference": False, "network": "none", "filesystem": "none"}
            node.quarantine_reason = reason
            self._persist()
            logger.warning(f"Worker '{node_id}' REVOKED: {reason}")
            return True

    def get_node(self, node_id: str) -> Optional[ClusterWorkerNode]:
        with self._lock:
            return self._nodes.get(node_id)

    def list_nodes(self, state_filter: Optional[WorkerState] = None) -> List[Dict[str, Any]]:
        with self._lock:
            nodes = []
            for n in self._nodes.values():
                if state_filter is None or n.state == state_filter:
                    nodes.append(n.to_dict())
            return nodes

    def get_ready_workers(self, requires_gpu: bool = False, workload_type: Optional[str] = None) -> List[ClusterWorkerNode]:
        """Returns all healthy, READY workers that satisfy capability requirements."""
        with self._lock:
            now = time.time()
            ready_workers = []
            for node in self._nodes.values():
                if node.state != WorkerState.READY:
                    continue
                if node.security_state != "ACTIVE":
                    continue
                # Mark as UNAVAILABLE if heartbeat is dead (>60s)
                if now - node.last_heartbeat > 60.0:
                    node.state = WorkerState.UNAVAILABLE
                    continue
                if requires_gpu and not node.has_gpu:
                    continue
                if workload_type and workload_type not in node.supported_workloads and "INFERENCE" not in node.supported_workloads:
                    continue
                ready_workers.append(node)
            return ready_workers

    def select_best_worker(self, requires_gpu: bool = False, workload_type: Optional[str] = None) -> Optional[ClusterWorkerNode]:
        """
        Multi-factor score-based selection:
        Minimizes load, queue depth, and latency while maximizing available capacity.
        """
        candidates = self.get_ready_workers(requires_gpu=requires_gpu, workload_type=workload_type)
        if not candidates:
            return None

        def score(n: ClusterWorkerNode) -> float:
            load_pen = n.current_load * 50.0
            queue_pen = float(n.queue_depth) * 20.0
            lat_pen = min(100.0, n.average_latency_ms * 0.2)
            cap_bonus = min(50.0, n.cpu_capacity * 5.0)
            return (load_pen + queue_pen + lat_pen) - cap_bonus

        return min(candidates, key=score)

    def create_signed_snapshot(self) -> Dict[str, Any]:
        """Creates a consistent snapshot of the cluster state for control plane failover."""
        with self._lock:
            return {
                "snapshot_time": time.time(),
                "nodes": [n.to_dict() for n in self._nodes.values()],
                "canonical_checksum": CANONICAL_MODEL_SHA256,
                "protocol_version": CANONICAL_PROTOCOL_VERSION
            }
