"""
TARA/ACCESS/biometrics/provider.py

Abstract Base Class & Capability Contracts for TARA Biometric Subsystem.

Security Rules:
- NEVER read, export, or store raw biometric templates on disk or in memory.
- NEVER transmit raw biometric data over network/API interfaces.
- TARA receives strictly the cryptographic verification result (True/False + Capability Status).
- Enforce strict fail-closed capability detection.
- Simulation is FORBIDDEN in production and allowed only during isolated tests with TARA_TEST_MODE=1.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional
from datetime import datetime, timezone


class BiometricCapability(Enum):
    AVAILABLE = "AVAILABLE"              # Hardware present, configured, and enrolled
    NOT_AVAILABLE = "NOT_AVAILABLE"      # Hardware sensor absent or unsupported by OS
    NOT_ENROLLED = "NOT_ENROLLED"        # Hardware present but no user biometric credentials enrolled
    DISABLED = "DISABLED"                # Biometric subsystem disabled by admin policy
    ERROR = "ERROR"                      # Subsystem error or driver malfunction


class BiometricType(Enum):
    WINDOWS_HELLO = "windows_hello"
    FINGERPRINT = "fingerprint"
    FACE = "face"
    IRIS = "iris"
    VOICE_AUXILIARY = "voice_auxiliary"
    DEVICE_CREDENTIAL = "device_credential"


@dataclass
class BiometricAuthResult:
    success: bool
    capability: BiometricCapability
    auth_type: str
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "capability": self.capability.value,
            "auth_type": self.auth_type,
            "error": self.error,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class BiometricProvider(ABC):
    """
    Abstract base provider defining standardized lifecycle, hardware interrogation,
    and authentication interface across Windows Hello, Fingerprint, Face, Iris, and Voice.
    """

    def __init__(self, provider_name: str, test_mode: bool = False):
        self.provider_name = provider_name
        self.test_mode = test_mode

    @abstractmethod
    def get_capability(self) -> BiometricCapability:
        """
        Query system/hardware for genuine biometric capability.
        Must never claim AVAILABLE if sensor is missing or unenrolled.
        """
        pass

    @abstractmethod
    def is_enrolled(self) -> bool:
        """Checks if valid user credentials/templates are currently enrolled."""
        pass

    @abstractmethod
    def authenticate(
        self,
        prompt: str = "TARA Creator Verification",
        context: Optional[Dict[str, Any]] = None
    ) -> BiometricAuthResult:
        """
        Request user biometric verification.
        Fails closed on any hardware or security anomaly.
        """
        pass
