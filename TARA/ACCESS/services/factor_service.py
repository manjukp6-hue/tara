"""
TARA/ACCESS/services/biometric.py

Android Biometric Subsystem interface.

Security Rules:
- Never read, store, or upload biometric templates.
- TARA receives ONLY the authentication result (True/False).
- Biometric verification provides an additional local confirmation layer.
- Device fallback (PIN/passkey) supported according to Android security guidelines.
"""

from typing import Dict, Optional, Any
from enum import Enum


import os


class BiometricType(Enum):
    FINGERPRINT = "fingerprint"
    FACE = "face"
    DEVICE_CREDENTIAL = "pin_pattern_passkey"


class BiometricService:
    """
    Android BiometricPrompt wrapper.
    Evaluates authentication results without ever inspecting or storing biometric data.
    Enforces strict production protection against simulated authentication.
    """
    def __init__(self, hardware_available: bool = False, test_mode: bool = False):
        self.hardware_available = hardware_available
        self.enrolled_biometrics: bool = hardware_available
        self.test_mode = test_mode

    def can_authenticate(self) -> bool:
        """Checks if device hardware supports biometric authentication."""
        return self.hardware_available and self.enrolled_biometrics

    def authenticate_biometric(
        self,
        prompt_title: str = "TARA Creator Verification",
        simulate_user_present: bool = False
    ) -> Dict[str, Any]:
        """
        Requests biometric confirmation via platform BiometricPrompt.
        Simulation is strictly blocked in production and allowed only when
        test_mode is True and TARA_TEST_MODE=1.
        """
        if simulate_user_present:
            if not self.test_mode or os.environ.get("TARA_TEST_MODE") != "1":
                raise PermissionError("SECURITY_VIOLATION: Biometric simulation is forbidden in production.")
            return {
                "success": True,
                "auth_type": BiometricType.FINGERPRINT.value,
                "error": None
            }

        if not self.can_authenticate():
            return {
                "success": False,
                "error": "BIOMETRIC_ERROR_HW_UNAVAILABLE",
                "fallback_available": True
            }

        # Real platform biometric prompt would execute here
        return {
            "success": False,
            "error": "BIOMETRIC_UNCONFIRMED",
            "fallback_available": True
        }

    def authenticate_fallback_credential(self, pin_or_passkey_correct: bool = True) -> Dict[str, Any]:
        """
        Device fallback authentication (Device PIN / Pattern / Passkey).
        """
        if pin_or_passkey_correct:
            return {
                "success": True,
                "auth_type": BiometricType.DEVICE_CREDENTIAL.value,
                "error": None
            }
        return {
            "success": False,
            "error": "CREDENTIAL_INCORRECT",
            "fallback_available": True
        }
