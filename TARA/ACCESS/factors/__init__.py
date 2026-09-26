"""
TARA Multi-Factor Verification Subsystem Package.
Provides genuine capability detection and authentic verification providers for:
- Platform Authenticator (Windows Hello)
- Fingerprint
- Face
- Iris
- Voice (auxiliary secondary factor only)
"""

from .provider import (
    BiometricProvider,
    BiometricCapability,
    BiometricType,
    BiometricAuthResult
)
from .windows_hello import WindowsHelloProvider
from .fingerprint import FingerprintProvider
from .face import FaceProvider
from .iris import IrisProvider
from .voice_verifier import VoiceVerificationProvider

__all__ = [
    "BiometricProvider",
    "BiometricCapability",
    "BiometricType",
    "BiometricAuthResult",
    "WindowsHelloProvider",
    "FingerprintProvider",
    "FaceProvider",
    "IrisProvider",
    "VoiceVerificationProvider",
]
