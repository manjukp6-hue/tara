"""
python/tara_core/contracts.py

Canonical Contract Layer implementation in Python.
Defines schemas, dataclasses, validation, and serialization
shared between Rust Gateway and Python Worker.
"""

import os
import hashlib
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any

CANONICAL_PROTOCOL_VERSION = "1.0.0"
CANONICAL_MODEL_IDENTITY = "TARA"
CANONICAL_MODEL_SHA256 = "7a50308b799f2654baeafbd64dec31f088c3b07b446e198cd0f9ec2b7c0af309"
CANONICAL_PARAM_COUNT = 118080

# Production Deployment & Zero-Cost Compute Policy Invariants
PUBLIC_TARA_URL = os.environ.get("PUBLIC_TARA_URL", "https://gateway.tara.local")
TARA_CONTROL_PLANE_URL = os.environ.get("TARA_CONTROL_PLANE_URL", "http://127.0.0.1:8765")
USER_COMPUTE_COST: float = 0.0
ENFORCE_ZERO_USER_COST: bool = True


@dataclass
class AuthContext:
    actor_id: str
    session_id: str
    creator_verified: bool = False
    client_ip: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuthContext":
        return cls(
            actor_id=data.get("actor_id", "anonymous"),
            session_id=data.get("session_id", "default_session"),
            creator_verified=bool(data.get("creator_verified", False)),
            client_ip=data.get("client_ip")
        )


@dataclass
class CanonicalInferenceRequest:
    request_id: str
    prompt: str
    expected_model_checksum: str
    job_id: Optional[str] = None
    max_tokens: int = 150
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    stop_tokens: List[str] = field(default_factory=lambda: ["<|im_end|>", "<|pad|>"])
    expected_model_identity: str = CANONICAL_MODEL_IDENTITY
    auth_context: Optional[AuthContext] = None

    def validate(self) -> Tuple_Validation:
        if not self.request_id:
            return False, "Missing required request_id"
        if not self.prompt:
            return False, "Missing required prompt"
        if self.expected_model_checksum.lower() != CANONICAL_MODEL_SHA256.lower():
            return False, f"Checksum mismatch: expected {CANONICAL_MODEL_SHA256}, got {self.expected_model_checksum}"
        return True, None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.auth_context:
            d["auth_context"] = self.auth_context.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CanonicalInferenceRequest":
        auth_data = data.get("auth_context")
        auth_ctx = AuthContext.from_dict(auth_data) if auth_data else None
        return cls(
            request_id=str(data.get("request_id", "")),
            prompt=str(data.get("prompt", "")),
            expected_model_checksum=str(data.get("expected_model_checksum", "")),
            job_id=data.get("job_id"),
            max_tokens=int(data.get("max_tokens", 150)),
            temperature=float(data.get("temperature", 0.7)),
            top_k=int(data.get("top_k", 50)),
            top_p=float(data.get("top_p", 0.9)),
            repetition_penalty=float(data.get("repetition_penalty", 1.1)),
            stop_tokens=list(data.get("stop_tokens", ["<|im_end|>", "<|pad|>"])),
            expected_model_identity=str(data.get("expected_model_identity", CANONICAL_MODEL_IDENTITY)),
            auth_context=auth_ctx
        )


@dataclass
class CanonicalInferenceResponse:
    request_id: str
    status: str
    text: str
    model_checksum: str
    runtime_engine: str
    token_count: int = 0
    token_ids: List[int] = field(default_factory=list)
    model_identity: str = CANONICAL_MODEL_IDENTITY
    first_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    tokens_per_second: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CanonicalInferenceResponse":
        return cls(
            request_id=str(data.get("request_id", "")),
            status=str(data.get("status", "SUCCESS")),
            text=str(data.get("text", "")),
            model_checksum=str(data.get("model_checksum", "")),
            runtime_engine=str(data.get("runtime_engine", "python_cpu")),
            token_count=int(data.get("token_count", 0)),
            token_ids=list(data.get("token_ids", [])),
            model_identity=str(data.get("model_identity", CANONICAL_MODEL_IDENTITY)),
            first_latency_ms=float(data.get("first_latency_ms", 0.0)),
            total_latency_ms=float(data.get("total_latency_ms", 0.0)),
            tokens_per_second=float(data.get("tokens_per_second", 0.0))
        )


@dataclass
class WorkerHealthResponse:
    status: str
    ready: bool
    protocol_version: str
    model_checksum: str
    has_gpu: bool
    worker_identity: str = "tara_python_gpu_worker"
    model_identity: str = CANONICAL_MODEL_IDENTITY
    parameters: int = CANONICAL_PARAM_COUNT
    gpu_device: Optional[str] = None
    device_type: str = "cpu"
    shared_storage_verified: bool = True
    queue_depth: int = 0
    uptime_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CanonicalErrorResponse:
    request_id: str
    status: str = "ERROR"
    error_code: str = "INTERNAL_ERROR"
    message: str = ""
    retryable: bool = False
    failover_advised: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -------------------------------------------------------------
# Voice Subsystem Contracts
# -------------------------------------------------------------

@dataclass
class VoiceTranscribeRequest:
    audio_data: str
    language: str = "en-US"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VoiceTranscribeRequest":
        return cls(
            audio_data=str(data.get("audio_data") or data.get("audio_base64") or ""),
            language=str(data.get("language", "en-US"))
        )


@dataclass
class VoiceTranscribeResponse:
    status: str
    transcript: str
    language: str
    confidence: float
    has_speech: bool
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VoiceSynthesizeRequest:
    text: str
    language: str = "en"
    rate: Optional[int] = None
    volume: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VoiceConverseRequest:
    input_text: Optional[str] = None
    audio_data: Optional[str] = None
    language: str = "en-US"
    context: Optional[Dict[str, Any]] = None
    require_wake_word: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VoiceConverseResponse:
    status: str
    transcript: str
    response_text: str
    audio_base64: str
    language: str
    turn_id: str
    brain_result: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VoiceBargeInResponse:
    status: str
    barge_in: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VoiceCapabilitiesResponse:
    status: str
    stt_available: bool
    tts_available: bool
    languages: List[str]
    barge_in_supported: bool
    wake_words: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -------------------------------------------------------------
# Creator Setup & Device Management Contracts
# -------------------------------------------------------------

@dataclass
class CreatorSetupRequest:
    google_id_token: str
    confirm_identity: bool = True
    device_name: str = "Primary PC"
    device_public_key: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CreatorSetupResponse:
    status: str
    authority_state: str
    creator_id: str
    display_name: str
    recovery_code: Optional[str] = None
    device: Optional[Dict[str, Any]] = None
    session: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DeviceRegisterRequest:
    device_public_key: str
    session_token: Optional[str] = None
    google_id_token: Optional[str] = None
    device_name: str = "Secondary PC"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DeviceRevokeRequest:
    device_id: Optional[str] = None
    session_token: Optional[str] = None
    all: bool = False
    reason: str = "manual_revocation"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DeviceListResponse:
    status: str
    devices: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Type alias helper
Tuple_Validation = tuple[bool, Optional[str]]

