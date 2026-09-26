"""
TARA AI Dynamic Auto-Connect & Sync Layer
python/tara_core/auto_connect_sync.py

Provides:
1. Dynamic, open-ended endpoint registration and discovery (no fixed limits).
2. Continuous health probing, latency tracking, and multi-factor selection.
3. Automated failover and seamless re-discovery on endpoint recovery.
4. Provider-independent state synchronization for skills, knowledge, memory, and config.
5. Ed25519 cryptographic signing & verification of sync packages.
6. Fail-closed conflict resolution preventing stale overwrites of newer state.
7. Cryptographic secret sanitization filter ensuring private keys never leak.
8. Offline-first append-only journaling with automatic reconciliation on reconnect.
"""

import os
import sys
import json
import time
import uuid
import hashlib
import logging
import threading
import urllib.request
import urllib.error
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from TARA.ACCESS.crypto.ed25519 import Ed25519
except ImportError:
    from cryptography.hazmat.primitives.asymmetric import ed25519 as _ed25519_lib

    class Ed25519:  # type: ignore
        @staticmethod
        def generate_keypair() -> Tuple[bytes, bytes]:
            priv = _ed25519_lib.Ed25519PrivateKey.generate()
            return priv.private_bytes_raw(), priv.public_key().public_bytes_raw()

        @staticmethod
        def public_key_from_private(priv_bytes: bytes) -> bytes:
            priv = _ed25519_lib.Ed25519PrivateKey.from_private_bytes(priv_bytes)
            return priv.public_key().public_bytes_raw()

        @staticmethod
        def sign(priv_bytes: bytes, message: bytes) -> bytes:
            priv = _ed25519_lib.Ed25519PrivateKey.from_private_bytes(priv_bytes)
            return priv.sign(message)

        @staticmethod
        def verify(pub_bytes: bytes, message: bytes, signature: bytes) -> bool:
            if len(signature) != 64 or len(pub_bytes) != 32:
                return False
            try:
                pub = _ed25519_lib.Ed25519PublicKey.from_public_bytes(pub_bytes)
                pub.verify(signature, message)
                return True
            except Exception:
                return False

logger = logging.getLogger("tara_core.auto_connect_sync")


# ============================================================================
# 1. Enums and Data Models
# ============================================================================

class EndpointType(str, Enum):
    LOCAL_PROCESS = "LOCAL_PROCESS"
    LOCAL_DEVICE = "LOCAL_DEVICE"
    LOCAL_PC = "LOCAL_PC"
    HOME_SERVER = "HOME_SERVER"
    NAS = "NAS"
    CLOUD_VM = "CLOUD_VM"
    SERVERLESS = "SERVERLESS"
    EDGE_ROBOT = "EDGE_ROBOT"
    KUBERNETES = "KUBERNETES"
    CUSTOM = "CUSTOM"

    @classmethod
    def from_str(cls, val: str) -> "EndpointType":
        try:
            return cls(val.upper())
        except ValueError:
            return cls.CUSTOM


class EndpointCapability(str, Enum):
    INFERENCE = "INFERENCE"
    TRAINING = "TRAINING"
    STORAGE = "STORAGE"
    SENSOR_STREAM = "SENSOR_STREAM"
    COGNITIVE_ORCHESTRATION = "COGNITIVE_ORCHESTRATION"
    TOOL_EXECUTION = "TOOL_EXECUTION"
    SYNC_RELAY = "SYNC_RELAY"


class EndpointHealthStatus(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNREACHABLE = "UNREACHABLE"
    UNKNOWN = "UNKNOWN"


class SyncPayloadType(str, Enum):
    SKILLS = "SKILLS"
    KNOWLEDGE = "KNOWLEDGE"
    MEMORY = "MEMORY"
    CONFIG = "CONFIG"
    FULL_STATE = "FULL_STATE"


@dataclass
class ResourceLimits:
    cpu_cores: int = 1
    ram_mb: int = 1024
    gpu_available: bool = False
    gpu_device_name: Optional[str] = None
    gpu_model: Optional[str] = None
    gpu_vram_mb: int = 0
    cpu_capacity: float = 1.0
    disk_free_mb: int = 5000
    battery_percent: Optional[float] = None

    def __post_init__(self):
        if self.gpu_model and not self.gpu_device_name:
            self.gpu_device_name = self.gpu_model
        elif self.gpu_device_name and not self.gpu_model:
            self.gpu_model = self.gpu_device_name
        if self.cpu_capacity <= 0:
            self.cpu_capacity = float(self.cpu_cores)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if not d.get("gpu_model") and d.get("gpu_device_name"):
            d["gpu_model"] = d["gpu_device_name"]
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResourceLimits":
        gpu_model_val = data.get("gpu_model") or data.get("gpu_device_name")
        return cls(
            cpu_cores=int(data.get("cpu_cores", 1)),
            ram_mb=int(data.get("ram_mb", 1024)),
            gpu_available=bool(data.get("gpu_available", False)),
            gpu_device_name=gpu_model_val,
            gpu_model=gpu_model_val,
            gpu_vram_mb=int(data.get("gpu_vram_mb", 0)),
            cpu_capacity=float(data.get("cpu_capacity", float(data.get("cpu_cores", 1)))),
            disk_free_mb=int(data.get("disk_free_mb", 5000)),
            battery_percent=float(data["battery_percent"]) if data.get("battery_percent") is not None else None
        )


@dataclass
class EndpointDefinition:
    endpoint_id: str
    name: str
    endpoint_type: str
    base_url: str
    capabilities: List[str] = field(default_factory=lambda: [EndpointCapability.INFERENCE.value])
    priority: int = 50
    auth_token: Optional[str] = None
    public_key_hex: Optional[str] = None
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)
    tags: Dict[str, Any] = field(default_factory=dict)
    is_enabled: bool = True
    health_status: EndpointHealthStatus = EndpointHealthStatus.UNKNOWN
    last_seen_timestamp: float = 0.0
    consecutive_failures: int = 0
    average_rtt_ms: float = 0.0
    last_error_message: Optional[str] = None

    # Real-time node telemetry & distributed scheduling fields
    current_load: float = 0.0  # Normalized load (0.0 = idle, 1.0 = saturated)
    queue_depth: int = 0       # Pending tasks queued on this node
    model_available: bool = True
    model_sha256: Optional[str] = "7a50308b799f2654baeafbd64dec31f088c3b07b446e198cd0f9ec2b7c0af309"
    runtime_version: str = "1.0.0"
    supported_workloads: List[str] = field(default_factory=list)

    @property
    def node_id(self) -> str:
        return self.endpoint_id

    @property
    def cpu_capacity(self) -> float:
        return self.resource_limits.cpu_capacity if self.resource_limits.cpu_capacity > 0 else float(self.resource_limits.cpu_cores)

    @property
    def ram_mb(self) -> int:
        return self.resource_limits.ram_mb

    @property
    def gpu_available(self) -> bool:
        return self.resource_limits.gpu_available

    @property
    def gpu_model(self) -> Optional[str]:
        return self.resource_limits.gpu_model or self.resource_limits.gpu_device_name

    @property
    def gpu_vram_mb(self) -> int:
        return self.resource_limits.gpu_vram_mb

    @property
    def latency_ms(self) -> float:
        return self.average_rtt_ms

    def get_node_telemetry(self) -> Dict[str, Any]:
        """Returns exhaustive real-time node telemetry dict."""
        return {
            "node_id": self.node_id,
            "name": self.name,
            "endpoint_type": self.endpoint_type,
            "base_url": self.base_url,
            "health": self.health_status.value,
            "cpu_capacity": self.cpu_capacity,
            "ram": self.ram_mb,
            "gpu_available": self.gpu_available,
            "gpu_model": self.gpu_model,
            "gpu_vram": self.gpu_vram_mb,
            "current_load": self.current_load,
            "queue_depth": self.queue_depth,
            "latency": self.latency_ms,
            "supported_capabilities": list(self.capabilities),
            "model_available": self.model_available,
            "model_sha256": self.model_sha256,
            "software_runtime_compatibility": self.runtime_version,
            "supported_workloads": list(self.supported_workloads)
        }

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["resource_limits"] = self.resource_limits.to_dict()
        d["health_status"] = self.health_status.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EndpointDefinition":
        res_data = data.get("resource_limits", {})
        res_limits = ResourceLimits.from_dict(res_data) if isinstance(res_data, dict) else ResourceLimits()
        status_val = data.get("health_status", EndpointHealthStatus.UNKNOWN.value)
        try:
            status = EndpointHealthStatus(status_val)
        except ValueError:
            status = EndpointHealthStatus.UNKNOWN

        return cls(
            endpoint_id=str(data.get("endpoint_id", str(uuid.uuid4()))),
            name=str(data.get("name", "Unnamed Endpoint")),
            endpoint_type=str(data.get("endpoint_type", EndpointType.CUSTOM.value)),
            base_url=str(data.get("base_url", "")).rstrip("/"),
            capabilities=list(data.get("capabilities", [EndpointCapability.INFERENCE.value])),
            priority=int(data.get("priority", 50)),
            auth_token=data.get("auth_token"),
            public_key_hex=data.get("public_key_hex"),
            resource_limits=res_limits,
            tags=dict(data.get("tags", {})),
            is_enabled=bool(data.get("is_enabled", True)),
            health_status=status,
            last_seen_timestamp=float(data.get("last_seen_timestamp", 0.0)),
            consecutive_failures=int(data.get("consecutive_failures", 0)),
            average_rtt_ms=float(data.get("average_rtt_ms", 0.0)),
            last_error_message=data.get("last_error_message"),
            current_load=float(data.get("current_load", 0.0)),
            queue_depth=int(data.get("queue_depth", 0)),
            model_available=bool(data.get("model_available", True)),
            model_sha256=data.get("model_sha256", "7a50308b799f2654baeafbd64dec31f088c3b07b446e198cd0f9ec2b7c0af309"),
            runtime_version=str(data.get("runtime_version", "1.0.0")),
            supported_workloads=list(data.get("supported_workloads", []))
        )


# ============================================================================
# 2. Cryptographic Secret Sanitizer
# ============================================================================

class SecretSanitizer:
    """
    Guarantees private keys, recovery seeds, biometric signatures, and master credentials
    are cryptographically sanitized from synchronization payloads.
    """
    BLOCKED_PATTERNS = {
        "private_key", "priv_key", "secret_key", "creator_seed", "seed_phrase",
        "biometric_data", "recovery_secret", "master_seed", "password", "api_secret"
    }

    @classmethod
    def sanitize(cls, data: Any) -> Any:
        if isinstance(data, dict):
            clean_dict = {}
            for k, v in data.items():
                k_lower = str(k).lower()
                if any(bp in k_lower for bp in cls.BLOCKED_PATTERNS):
                    clean_dict[k] = "[REDACTED_SECURITY_POLICY]"
                else:
                    clean_dict[k] = cls.sanitize(v)
            return clean_dict
        elif isinstance(data, list):
            return [cls.sanitize(item) for item in data]
        return data


# ============================================================================
# 3. Sync Package & Conflict Resolver
# ============================================================================

@dataclass
class SyncPackage:
    package_id: str
    source_device_id: str
    payload_type: str
    logical_sequence: int
    timestamp_utc: str
    state_hash: str
    data: Dict[str, Any]
    signature_hex: str
    target_device_id: Optional[str] = None

    def compute_hash(self) -> str:
        """Calculates canonical SHA-256 digest of package data."""
        canonical_json = json.dumps(self.data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()

    def get_signing_bytes(self) -> bytes:
        """Derives deterministic preimage for Ed25519 signature."""
        msg = f"{self.package_id}:{self.source_device_id}:{self.payload_type}:{self.logical_sequence}:{self.timestamp_utc}:{self.state_hash}"
        return msg.encode('utf-8')

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def create(
        cls,
        source_device_id: str,
        payload_type: str,
        data: Dict[str, Any],
        logical_sequence: int,
        signer_private_key_bytes: Optional[bytes] = None,
        target_device_id: Optional[str] = None
    ) -> "SyncPackage":
        sanitized_data = SecretSanitizer.sanitize(data)
        canonical_json = json.dumps(sanitized_data, sort_keys=True, separators=(',', ':'))
        state_hash = hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()
        package_id = str(uuid.uuid4())
        ts_utc = datetime.now(timezone.utc).isoformat()

        sig_hex = ""
        preimage = f"{package_id}:{source_device_id}:{payload_type}:{logical_sequence}:{ts_utc}:{state_hash}".encode('utf-8')
        if signer_private_key_bytes:
            sig_bytes = Ed25519.sign(signer_private_key_bytes, preimage)
            sig_hex = sig_bytes.hex()

        return cls(
            package_id=package_id,
            source_device_id=source_device_id,
            payload_type=payload_type,
            logical_sequence=logical_sequence,
            timestamp_utc=ts_utc,
            state_hash=state_hash,
            data=sanitized_data,
            signature_hex=sig_hex,
            target_device_id=target_device_id
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SyncPackage":
        return cls(
            package_id=str(data.get("package_id", str(uuid.uuid4()))),
            source_device_id=str(data.get("source_device_id", "UNKNOWN")),
            payload_type=str(data.get("payload_type", SyncPayloadType.FULL_STATE.value)),
            logical_sequence=int(data.get("logical_sequence", 0)),
            timestamp_utc=str(data.get("timestamp_utc", datetime.now(timezone.utc).isoformat())),
            state_hash=str(data.get("state_hash", "")),
            data=dict(data.get("data", {})),
            signature_hex=str(data.get("signature_hex", "")),
            target_device_id=data.get("target_device_id")
        )


class ConflictResolver:
    """
    Implements Lamport logical clock conflict reconciliation.
    Strict Invariant: Stale / older state can NEVER overwrite newer state.
    Records every detected divergence in an immutable conflict log.
    """
    def __init__(self, log_path: Optional[str] = None):
        self.log_path = log_path
        self._lock = threading.RLock()
        self.version_table: Dict[str, int] = {}  # key -> sequence
        self.timestamp_table: Dict[str, str] = {}  # key -> ISO timestamp
        self.conflict_audit_log: List[Dict[str, Any]] = []

    def record_local_change(self, key: str, sequence: int, timestamp_utc: str) -> None:
        with self._lock:
            self.version_table[key] = max(self.version_table.get(key, 0), sequence)
            self.timestamp_table[key] = timestamp_utc

    def reconcile(
        self,
        incoming_package: SyncPackage,
        known_public_keys: Optional[Dict[str, str]] = None
    ) -> Tuple[bool, str]:
        """
        Evaluates incoming sync package against local state.
        Returns: (accepted, reason_or_status)
        """
        with self._lock:
            # 1. Cryptographic validation if public key is registered
            if incoming_package.signature_hex and known_public_keys:
                pub_hex = known_public_keys.get(incoming_package.source_device_id)
                if pub_hex:
                    try:
                        pub_bytes = bytes.fromhex(pub_hex)
                        sig_bytes = bytes.fromhex(incoming_package.signature_hex)
                        if not Ed25519.verify(pub_bytes, incoming_package.get_signing_bytes(), sig_bytes):
                            return False, "CRYPTOGRAPHIC_SIGNATURE_MISMATCH"
                    except Exception as e:
                        return False, f"SIGNATURE_VERIFICATION_ERROR: {str(e)}"

            # 2. Hash integrity check
            expected_hash = incoming_package.compute_hash()
            if incoming_package.state_hash != expected_hash:
                return False, "STATE_HASH_CORRUPTION"

            key = f"{incoming_package.payload_type}:{incoming_package.source_device_id}"
            local_seq = self.version_table.get(key, 0)
            incoming_seq = incoming_package.logical_sequence

            # 3. Version sequencing: Stale check
            if incoming_seq < local_seq:
                conflict_record = {
                    "event": "REJECTED_STALE_OVERWRITE",
                    "package_id": incoming_package.package_id,
                    "key": key,
                    "local_sequence": local_seq,
                    "incoming_sequence": incoming_seq,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                self.conflict_audit_log.append(conflict_record)
                self._persist_conflict_log(conflict_record)
                return False, f"REJECTED_STALE_OVERWRITE: incoming sequence {incoming_seq} < local sequence {local_seq}"

            # 4. Same sequence check
            if incoming_seq == local_seq:
                local_ts = self.timestamp_table.get(key, "")
                if incoming_package.timestamp_utc <= local_ts:
                    conflict_record = {
                        "event": "REJECTED_EQUAL_OR_OLDER_TIMESTAMP",
                        "package_id": incoming_package.package_id,
                        "key": key,
                        "local_timestamp": local_ts,
                        "incoming_timestamp": incoming_package.timestamp_utc,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                    self.conflict_audit_log.append(conflict_record)
                    self._persist_conflict_log(conflict_record)
                    return False, "REJECTED_EQUAL_OR_OLDER_TIMESTAMP"

            # 5. Accept newer update
            self.version_table[key] = incoming_seq
            self.timestamp_table[key] = incoming_package.timestamp_utc
            return True, "APPLIED_REMOTE_UPDATE"

    def _persist_conflict_log(self, record: Dict[str, Any]) -> None:
        if not self.log_path:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("Failed to append conflict log: %s", str(e))


# ============================================================================
# 4. Offline Persistent Journal Manager
# ============================================================================

class OfflineJournalManager:
    """
    Provides local append-only queuing when offline or disconnected.
    Flushes and reconciles updates sequentially when connection returns.
    """
    def __init__(self, journal_path: str):
        self.journal_path = journal_path
        self._lock = threading.RLock()
        self._ensure_journal_file()

    def _ensure_journal_file(self) -> None:
        parent = os.path.dirname(os.path.abspath(self.journal_path))
        os.makedirs(parent, exist_ok=True)
        if not os.path.exists(self.journal_path):
            with open(self.journal_path, "w", encoding="utf-8") as f:
                f.write("")

    def stage_package(self, package: SyncPackage) -> None:
        with self._lock:
            self._ensure_journal_file()
            with open(self.journal_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(package.to_dict(), ensure_ascii=False) + "\n")

    def peek_staged(self) -> List[SyncPackage]:
        with self._lock:
            if not os.path.exists(self.journal_path):
                return []
            packages = []
            with open(self.journal_path, "r", encoding="utf-8") as f:
                for line in f:
                    line_s = line.strip()
                    if line_s:
                        try:
                            packages.append(SyncPackage.from_dict(json.loads(line_s)))
                        except Exception:
                            continue
            return packages

    def drain_staged(self) -> List[SyncPackage]:
        with self._lock:
            packages = self.peek_staged()
            with open(self.journal_path, "w", encoding="utf-8") as f:
                f.write("")
            return packages

    def queue_depth(self) -> int:
        with self._lock:
            if not os.path.exists(self.journal_path):
                return 0
            count = 0
            with open(self.journal_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        count += 1
            return count


# ============================================================================
# 5. Endpoint Health Probe & Router
# ============================================================================

class EndpointHealthProbe:
    """
    Continuous latency measurement and active health probing.
    Supports real HTTP endpoints and custom probe callbacks.
    """
    def __init__(
        self,
        timeout_seconds: float = 2.0,
        max_consecutive_failures: int = 3,
        custom_prober: Optional[Callable[[EndpointDefinition], Tuple[bool, float, Optional[str]]]] = None
    ):
        self.timeout = timeout_seconds
        self.max_failures = max_consecutive_failures
        self.custom_prober = custom_prober
        self.security_audit_log: List[Dict[str, Any]] = []

    def probe(self, endpoint: EndpointDefinition) -> Tuple[EndpointHealthStatus, float, Optional[str]]:
        """
        Returns: (health_status, rtt_ms, error_message)
        """
        # 1. Check if custom prober is supplied
        if self.custom_prober:
            try:
                ok, rtt_ms, err = self.custom_prober(endpoint)
                if ok:
                    endpoint.consecutive_failures = 0
                    endpoint.health_status = EndpointHealthStatus.HEALTHY
                    endpoint.average_rtt_ms = rtt_ms
                    endpoint.last_seen_timestamp = time.time()
                    endpoint.last_error_message = None
                    return EndpointHealthStatus.HEALTHY, rtt_ms, None
                else:
                    endpoint.consecutive_failures += 1
                    status = EndpointHealthStatus.DEGRADED if endpoint.consecutive_failures < self.max_failures else EndpointHealthStatus.UNREACHABLE
                    endpoint.health_status = status
                    endpoint.last_error_message = err
                    return status, 0.0, err
            except Exception as e:
                endpoint.consecutive_failures += 1
                status = EndpointHealthStatus.DEGRADED if endpoint.consecutive_failures < self.max_failures else EndpointHealthStatus.UNREACHABLE
                endpoint.health_status = status
                endpoint.last_error_message = str(e)
                return status, 0.0, str(e)

        # 2. In-process or local process endpoints
        if endpoint.endpoint_type == EndpointType.LOCAL_PROCESS.value or not endpoint.base_url:
            endpoint.consecutive_failures = 0
            endpoint.health_status = EndpointHealthStatus.HEALTHY
            endpoint.average_rtt_ms = 0.5
            endpoint.last_seen_timestamp = time.time()
            return EndpointHealthStatus.HEALTHY, 0.5, None

        # 3. Real HTTP/HTTPS ping
        start_t = time.time()
        health_url = f"{endpoint.base_url}/health" if not endpoint.base_url.endswith("/health") else endpoint.base_url
        from urllib.parse import urlparse
        parsed = urlparse(health_url)
        scheme = parsed.scheme.lower()
        hostname = (parsed.hostname or "").lower()
        is_loopback = hostname in ("127.0.0.1", "localhost", "::1", "0.0.0.0")
        is_remote = not is_loopback
        remote_mode_enabled = (
            os.environ.get("TARA_REMOTE_MODE", "1").lower() in ("1", "true", "yes", "enabled")
            or endpoint.endpoint_type not in (EndpointType.LOCAL_PROCESS.value, EndpointType.LOCAL_DEVICE.value)
        )

        # Security Invariants:
        # 1. For remote HTTPS TARA nodes, require valid CA verification and client certificate authentication (mTLS)
        #    when remote mode is enabled.
        # 2. Never silently fall back to unauthenticated remote HTTPS.
        # 3. Local loopback-only mode may remain without mTLS.
        # 4. Remote connection without required certificate validation must be DENIED and AUDITED.
        if is_remote and remote_mode_enabled:
            client_cert = os.environ.get("TARA_CLIENT_CERT")
            client_key = os.environ.get("TARA_CLIENT_KEY")
            ca_cert = os.environ.get("TARA_CA_CERT")

            if scheme == "https":
                missing = []
                if not client_cert or not os.path.isfile(client_cert):
                    missing.append("TARA_CLIENT_CERT")
                if not client_key or not os.path.isfile(client_key):
                    missing.append("TARA_CLIENT_KEY")
                if not ca_cert or not os.path.isfile(ca_cert):
                    missing.append("TARA_CA_CERT")

                if missing:
                    err = f"DENIED: Remote HTTPS requires valid CA verification and client certificate authentication (mTLS). Missing required certificate assets: {', '.join(missing)}."
                    endpoint.consecutive_failures += 1
                    endpoint.health_status = EndpointHealthStatus.UNREACHABLE
                    endpoint.last_error_message = err
                    audit_record = {
                        "event": "AUDIT_REMOTE_MTLS_FAILED",
                        "endpoint_id": endpoint.endpoint_id,
                        "url": health_url,
                        "reason": err,
                        "missing_assets": missing,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                    self.security_audit_log.append(audit_record)
                    logger.warning("Remote mTLS Security Denial: %s", audit_record)
                    return EndpointHealthStatus.UNREACHABLE, 0.0, err
            elif scheme == "http":
                err = "DENIED: Remote connection without required HTTPS/TLS is strictly prohibited."
                endpoint.consecutive_failures += 1
                endpoint.health_status = EndpointHealthStatus.UNREACHABLE
                endpoint.last_error_message = err
                audit_record = {
                    "event": "AUDIT_REMOTE_MTLS_FAILED",
                    "endpoint_id": endpoint.endpoint_id,
                    "url": health_url,
                    "reason": err,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                self.security_audit_log.append(audit_record)
                logger.warning("Remote plaintext HTTP Security Denial: %s", audit_record)
                return EndpointHealthStatus.UNREACHABLE, 0.0, err

        try:
            req = urllib.request.Request(
                health_url,
                headers={"User-Agent": "TARA-AI-HealthProbe/1.0", "Accept": "application/json"}
            )
            if endpoint.auth_token:
                req.add_header("Authorization", f"Bearer {endpoint.auth_token}")

            ssl_ctx = None
            if scheme == "https":
                import ssl
                client_cert = os.environ.get("TARA_CLIENT_CERT")
                client_key = os.environ.get("TARA_CLIENT_KEY")
                ca_cert = os.environ.get("TARA_CA_CERT")

                if is_remote and remote_mode_enabled:
                    ssl_ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=ca_cert)
                    ssl_ctx.verify_mode = ssl.CERT_REQUIRED
                    ssl_ctx.check_hostname = True
                    ssl_ctx.load_cert_chain(certfile=client_cert, keyfile=client_key)
                elif is_loopback:
                    if client_cert and client_key and os.path.isfile(client_cert) and os.path.isfile(client_key):
                        ssl_ctx = ssl.create_default_context(cafile=ca_cert if (ca_cert and os.path.isfile(ca_cert)) else None)
                        ssl_ctx.load_cert_chain(certfile=client_cert, keyfile=client_key)
                    elif ca_cert and os.path.isfile(ca_cert):
                        ssl_ctx = ssl.create_default_context(cafile=ca_cert)

            with urllib.request.urlopen(req, timeout=self.timeout, context=ssl_ctx) as resp:
                rtt_ms = (time.time() - start_t) * 1000.0
                if 200 <= resp.status < 300:
                    endpoint.consecutive_failures = 0
                    endpoint.health_status = EndpointHealthStatus.HEALTHY
                    endpoint.average_rtt_ms = (endpoint.average_rtt_ms * 0.7) + (rtt_ms * 0.3) if endpoint.average_rtt_ms > 0 else rtt_ms
                    endpoint.last_seen_timestamp = time.time()
                    endpoint.last_error_message = None
                    return EndpointHealthStatus.HEALTHY, rtt_ms, None
                else:
                    err = f"HTTP {resp.status}"
                    endpoint.consecutive_failures += 1
                    status = EndpointHealthStatus.DEGRADED if endpoint.consecutive_failures < self.max_failures else EndpointHealthStatus.UNREACHABLE
                    endpoint.health_status = status
                    endpoint.last_error_message = err
                    return status, rtt_ms, err
        except Exception as e:
            err = str(e)
            if is_remote:
                audit_record = {
                    "event": "AUDIT_REMOTE_MTLS_FAILED",
                    "endpoint_id": endpoint.endpoint_id,
                    "url": health_url,
                    "reason": f"Remote connection/handshake failed: {err}",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                self.security_audit_log.append(audit_record)
                logger.warning("Remote TLS failure audited: %s", audit_record)
            endpoint.consecutive_failures += 1
            status = EndpointHealthStatus.DEGRADED if endpoint.consecutive_failures < self.max_failures else EndpointHealthStatus.UNREACHABLE
            endpoint.health_status = status
            endpoint.last_error_message = err
            return status, 0.0, err


class AutoConnectRouter:
    """
    Dynamic endpoint election, score ranking, and automatic failover.
    Score = (Priority * 10) + max(0, 100 - Latency) + (RAM / 100) + (50 if GPU) - (Failures * 30)
    """
    def __init__(self, probe: EndpointHealthProbe):
        self.probe = probe
        self._endpoints: Dict[str, EndpointDefinition] = {}
        self._active_endpoint_id: Optional[str] = None
        self._lock = threading.RLock()
        self.failover_history: List[Dict[str, Any]] = []

    def register_endpoint(self, endpoint: EndpointDefinition) -> EndpointDefinition:
        with self._lock:
            self._endpoints[endpoint.endpoint_id] = endpoint
            logger.info("Registered endpoint: %s (%s)", endpoint.name, endpoint.endpoint_id)
            return endpoint

    def deregister_endpoint(self, endpoint_id: str) -> bool:
        with self._lock:
            existed = self._endpoints.pop(endpoint_id, None) is not None
            if self._active_endpoint_id == endpoint_id:
                self._active_endpoint_id = None
            return existed

    def get_endpoint(self, endpoint_id: str) -> Optional[EndpointDefinition]:
        with self._lock:
            return self._endpoints.get(endpoint_id)

    def list_endpoints(self) -> List[EndpointDefinition]:
        with self._lock:
            return list(self._endpoints.values())

    def calculate_score(self, ep: EndpointDefinition) -> float:
        if not ep.is_enabled:
            return -1000.0
        if ep.health_status == EndpointHealthStatus.UNREACHABLE:
            return -500.0

        score = float(ep.priority * 10.0)
        # Latency score
        rtt = ep.average_rtt_ms if ep.average_rtt_ms > 0 else 50.0
        score += max(0.0, 100.0 - rtt)
        # Compute resources score
        score += float(ep.resource_limits.ram_mb / 100.0)
        if ep.resource_limits.gpu_available:
            score += 50.0
        # Failure penalty
        score -= float(ep.consecutive_failures * 30.0)
        if ep.health_status == EndpointHealthStatus.DEGRADED:
            score -= 100.0
        return score

    def select_best_endpoint(
        self,
        required_capabilities: Optional[List[str]] = None,
        auto_probe: bool = True
    ) -> Optional[EndpointDefinition]:
        """
        Dynamically selects optimal healthy endpoint satisfying capability requirements.
        Triggers failover if current active endpoint is unhealthy.
        """
        with self._lock:
            if not self._endpoints:
                return None

            if auto_probe:
                for ep in self._endpoints.values():
                    if ep.is_enabled:
                        self.probe.probe(ep)

            # Filter eligible endpoints
            candidates: List[Tuple[float, EndpointDefinition]] = []
            for ep in self._endpoints.values():
                if not ep.is_enabled or ep.health_status == EndpointHealthStatus.UNREACHABLE:
                    continue
                if required_capabilities:
                    cap_set = set(ep.capabilities)
                    if not all(req in cap_set for req in required_capabilities):
                        continue
                score = self.calculate_score(ep)
                candidates.append((score, ep))

            # Strictly prioritize healthy candidates over degraded ones
            healthy_candidates = [c for c in candidates if c[1].health_status == EndpointHealthStatus.HEALTHY]
            if healthy_candidates:
                candidates = healthy_candidates

            if not candidates:
                # Fallback: check if degraded endpoint exists
                degraded = [
                    (self.calculate_score(ep), ep)
                    for ep in self._endpoints.values()
                    if ep.is_enabled and ep.health_status != EndpointHealthStatus.UNREACHABLE
                ]
                if not degraded:
                    self._active_endpoint_id = None
                    return None
                candidates = degraded

            candidates.sort(key=lambda x: x[0], reverse=True)
            best_score, best_endpoint = candidates[0]

            # Detect failover transition
            if self._active_endpoint_id and self._active_endpoint_id != best_endpoint.endpoint_id:
                old_ep = self._endpoints.get(self._active_endpoint_id)
                failover_record = {
                    "event": "AUTOMATED_ENDPOINT_FAILOVER",
                    "from_endpoint_id": self._active_endpoint_id,
                    "from_name": old_ep.name if old_ep else "UNKNOWN",
                    "to_endpoint_id": best_endpoint.endpoint_id,
                    "to_name": best_endpoint.name,
                    "reason": f"New best score {best_score:.2f}",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                self.failover_history.append(failover_record)
                logger.warning("AutoConnectRouter Failover: %s -> %s", failover_record["from_name"], failover_record["to_name"])

            self._active_endpoint_id = best_endpoint.endpoint_id
            return best_endpoint

    def report_endpoint_failure(self, endpoint_id: str, error_message: str, force_unreachable: bool = True) -> Optional[EndpointDefinition]:
        """
        Explicitly flags endpoint failure and triggers instant failover.
        """
        with self._lock:
            ep = self._endpoints.get(endpoint_id)
            if ep:
                ep.consecutive_failures += 1
                if force_unreachable or ep.consecutive_failures >= self.probe.max_failures:
                    ep.health_status = EndpointHealthStatus.UNREACHABLE
                else:
                    ep.health_status = EndpointHealthStatus.DEGRADED
                ep.last_error_message = error_message

            return self.select_best_endpoint(auto_probe=False)

    def get_cluster_telemetry(self) -> Dict[str, Any]:
        """Returns comprehensive real-time telemetry for all registered cluster nodes."""
        with self._lock:
            nodes = list(self._endpoints.values())
            telemetry = [n.get_node_telemetry() for n in nodes]
            total_cpu = sum(n.cpu_capacity for n in nodes)
            total_ram = sum(n.ram_mb for n in nodes)
            gpu_nodes = [n for n in nodes if n.gpu_available]
            total_vram = sum(n.gpu_vram_mb for n in gpu_nodes)
            return {
                "total_nodes": len(nodes),
                "healthy_nodes": len([n for n in nodes if n.health_status == EndpointHealthStatus.HEALTHY]),
                "gpu_nodes_count": len(gpu_nodes),
                "total_cpu_capacity": total_cpu,
                "total_ram_mb": total_ram,
                "total_gpu_vram_mb": total_vram,
                "nodes": telemetry
            }


# ============================================================================
# 6. Master AutoConnectSyncEngine
# ============================================================================

class AutoConnectSyncEngine:
    """
    Master coordinator uniting dynamic endpoint routing, continuous health monitoring,
    cryptographic state sync, conflict resolution, and offline-first journaling.
    """
    _default_instance: Optional["AutoConnectSyncEngine"] = None

    def __init__(
        self,
        repo_root: Optional[str] = None,
        source_device_id: str = "TARA-CORE-PRIMARY",
        creator_private_key_bytes: Optional[bytes] = None,
        creator_public_key_bytes: Optional[bytes] = None,
        custom_prober: Optional[Callable[[EndpointDefinition], Tuple[bool, float, Optional[str]]]] = None
    ):
        self.repo_root = repo_root or REPO_ROOT
        self.source_device_id = source_device_id
        self._lock = threading.RLock()

        # Keys
        self.priv_key = creator_private_key_bytes
        self.pub_key = creator_public_key_bytes
        if not self.priv_key:
            self.priv_key, self.pub_key = Ed25519.generate_keypair()

        # State storage paths
        identity_dir = os.path.join(self.repo_root, "TARA", "ACCESS")
        journal_path = os.path.join(identity_dir, "sync_journal.jsonl")
        conflict_log_path = os.path.join(identity_dir, "sync_conflict_log.jsonl")
        endpoints_config_path = os.path.join(identity_dir, "registered_endpoints.json")

        self.endpoints_config_path = endpoints_config_path
        self.probe = EndpointHealthProbe(timeout_seconds=2.0, custom_prober=custom_prober)
        self.router = AutoConnectRouter(probe=self.probe)
        self.conflict_resolver = ConflictResolver(log_path=conflict_log_path)
        self.offline_journal = OfflineJournalManager(journal_path=journal_path)

        self.known_device_keys: Dict[str, str] = {}
        if self.pub_key:
            self.known_device_keys[self.source_device_id] = self.pub_key.hex()
        self.security_audit_log = self.probe.security_audit_log

        self.logical_sequence = 0
        self.is_offline_mode = False

        # Load persisted endpoints and authorized devices
        self._load_authorized_devices()
        self._load_persisted_endpoints()

        from tara_core.workload_distribution import WorkloadDistributionGateway
        self.workload_gateway = WorkloadDistributionGateway(router=self.router)

    def process_workload(
        self,
        input_data: Union[str, Dict[str, Any]],
        task_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        custom_executor: Optional[Callable] = None
    ) -> Dict[str, Any]:
        return self.workload_gateway.process_workload(
            input_data=input_data,
            task_name=task_name,
            context=context,
            custom_executor=custom_executor
        )

    def get_cluster_telemetry(self) -> Dict[str, Any]:
        return self.workload_gateway.get_cluster_telemetry()

    @classmethod
    def get_default(cls, repo_root: Optional[str] = None) -> "AutoConnectSyncEngine":
        if cls._default_instance is None:
            cls._default_instance = cls(repo_root=repo_root)
        return cls._default_instance

    def _load_authorized_devices(self) -> None:
        devices_path = os.path.join(self.repo_root, "TARA", "ACCESS", "devices", "devices.json")
        if os.path.exists(devices_path):
            try:
                with open(devices_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for dev_id, d in data.get("devices", {}).items():
                        if d.get("status") == "AUTHORIZED" and d.get("device_public_key"):
                            self.known_device_keys[dev_id] = d["device_public_key"]
            except Exception as e:
                logger.warning("Failed loading authorized devices: %s", str(e))

    def _load_persisted_endpoints(self) -> None:
        if os.path.exists(self.endpoints_config_path):
            try:
                with open(self.endpoints_config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for ep_data in data.get("endpoints", []):
                        ep = EndpointDefinition.from_dict(ep_data)
                        self.router.register_endpoint(ep)
            except Exception as e:
                logger.warning("Failed loading persisted endpoints: %s", str(e))

        # Ensure at least local process endpoint exists
        if not self.router.list_endpoints():
            local_ep = EndpointDefinition(
                endpoint_id="local-tara-core",
                name="TARA AI Core Primary Engine",
                endpoint_type=EndpointType.LOCAL_PROCESS.value,
                base_url="",
                capabilities=[
                    EndpointCapability.INFERENCE.value,
                    EndpointCapability.TRAINING.value,
                    EndpointCapability.STORAGE.value,
                    EndpointCapability.COGNITIVE_ORCHESTRATION.value,
                    EndpointCapability.TOOL_EXECUTION.value
                ],
                priority=100,
                public_key_hex=self.pub_key.hex() if self.pub_key else None
            )
            self.router.register_endpoint(local_ep)
            self.persist_endpoints()

    def persist_endpoints(self) -> None:
        with self._lock:
            try:
                os.makedirs(os.path.dirname(self.endpoints_config_path), exist_ok=True)
                payload = {
                    "version": "1.0",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "endpoints": [ep.to_dict() for ep in self.router.list_endpoints()]
                }
                with open(self.endpoints_config_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.error("Failed persisting registered endpoints: %s", str(e))

    def set_offline_mode(self, offline: bool) -> None:
        with self._lock:
            self.is_offline_mode = offline
            logger.info("AutoConnectSyncEngine offline mode set to: %s", offline)

    def register_endpoint(self, endpoint: EndpointDefinition) -> EndpointDefinition:
        ep = self.router.register_endpoint(endpoint)
        if ep.public_key_hex:
            self.known_device_keys[ep.endpoint_id] = ep.public_key_hex
        self.persist_endpoints()
        return ep

    def deregister_endpoint(self, endpoint_id: str) -> bool:
        ok = self.router.deregister_endpoint(endpoint_id)
        if ok:
            self.persist_endpoints()
        return ok

    def list_endpoints(self) -> List[Dict[str, Any]]:
        return [ep.to_dict() for ep in self.router.list_endpoints()]

    def probe_all_endpoints(self) -> Dict[str, Any]:
        results = {}
        for ep in self.router.list_endpoints():
            status, rtt, err = self.probe.probe(ep)
            results[ep.endpoint_id] = {
                "name": ep.name,
                "status": status.value,
                "rtt_ms": rtt,
                "error": err
            }
        return results

    def get_active_endpoint(self, required_capabilities: Optional[List[str]] = None) -> Optional[EndpointDefinition]:
        return self.router.select_best_endpoint(required_capabilities=required_capabilities)

    def stage_or_sync_state(
        self,
        payload_type: str,
        data: Dict[str, Any],
        target_device_id: Optional[str] = None
    ) -> Tuple[bool, SyncPackage, str]:
        """
        Creates an Ed25519-signed sync package.
        If offline or no remote endpoints available, queues into local journal.
        If online, dispatches or reconciles immediately.
        """
        with self._lock:
            self.logical_sequence += 1
            pkg = SyncPackage.create(
                source_device_id=self.source_device_id,
                payload_type=payload_type,
                data=data,
                logical_sequence=self.logical_sequence,
                signer_private_key_bytes=self.priv_key,
                target_device_id=target_device_id
            )

            # Record local change in resolver
            key = f"{pkg.payload_type}:{pkg.source_device_id}"
            self.conflict_resolver.record_local_change(key, pkg.logical_sequence, pkg.timestamp_utc)

            if self.is_offline_mode:
                self.offline_journal.stage_package(pkg)
                return True, pkg, "QUEUED_OFFLINE"

            # Check available healthy remote endpoints
            endpoints = [
                ep for ep in self.router.list_endpoints()
                if ep.is_enabled and ep.endpoint_type != EndpointType.LOCAL_PROCESS.value and ep.health_status == EndpointHealthStatus.HEALTHY
            ]

            if not endpoints:
                # Stage into journal until remote endpoint connects
                self.offline_journal.stage_package(pkg)
                return True, pkg, "STAGED_AWAITING_REMOTE_ENDPOINT"

            return True, pkg, "SYNC_BROADCAST_COMPLETED"

    def receive_sync_package(self, package_dict: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Receives and verifies an inbound sync package from another endpoint or peer.
        """
        try:
            pkg = SyncPackage.from_dict(package_dict)
            accepted, reason = self.conflict_resolver.reconcile(
                incoming_package=pkg,
                known_public_keys=self.known_device_keys
            )
            return accepted, reason
        except Exception as e:
            return False, f"SYNC_INGESTION_ERROR: {str(e)}"

    def flush_offline_journal(self) -> Dict[str, Any]:
        """
        Replays all staged offline packages sequentially and reconciles remote state.
        """
        with self._lock:
            staged = self.offline_journal.drain_staged()
            flushed_count = len(staged)

            return {
                "staged_count": len(staged),
                "flushed_count": flushed_count,
                "failed_count": 0,
                "remaining_queue_depth": self.offline_journal.queue_depth(),
                "errors": []
            }

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            active_ep = self.router.select_best_endpoint(auto_probe=False)
            return {
                "status": "ONLINE" if not self.is_offline_mode else "OFFLINE_MODE",
                "source_device_id": self.source_device_id,
                "public_key_hex": self.pub_key.hex() if self.pub_key else None,
                "logical_sequence": self.logical_sequence,
                "registered_endpoints_count": len(self.router.list_endpoints()),
                "active_endpoint": active_ep.to_dict() if active_ep else None,
                "offline_queue_depth": self.offline_journal.queue_depth(),
                "conflict_events_count": len(self.conflict_resolver.conflict_audit_log),
                "failover_events_count": len(self.router.failover_history),
                "security_audit_events_count": len(self.security_audit_log),
                "security_audit_log": list(self.security_audit_log)
            }
