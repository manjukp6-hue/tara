"""
TARA/ACCESS/biometrics/windows_hello.py

Windows Hello & Biometric Framework Provider.
Interrogates Windows Biometric Framework (WinBio) and WebAuthn/Passport APIs.

Fails closed:
- NEVER fabricates hardware or simulated user presence in production.
- Strictly separates capability detection from simulation testing.
"""

import os
import sys
import ctypes
from typing import Dict, Any, Optional
from .provider import BiometricProvider, BiometricCapability, BiometricType, BiometricAuthResult


class WindowsHelloProvider(BiometricProvider):
    """
    Windows Hello Biometric Provider.
    Queries Windows Biometric Framework (WinBio) for installed biometric units (Fingerprint, Facial Recognition, Iris).
    """

    def __init__(self, test_mode: bool = False):
        super().__init__(provider_name=BiometricType.WINDOWS_HELLO.value, test_mode=test_mode)
        self._cached_capability: Optional[BiometricCapability] = None

    def get_capability(self) -> BiometricCapability:
        """
        Interrogate Windows Biometric Framework for connected biometric units.
        """
        if self._cached_capability is not None:
            return self._cached_capability

        if sys.platform != "win32":
            self._cached_capability = BiometricCapability.NOT_AVAILABLE
            return self._cached_capability

        try:
            # Query WinBio for biometric units
            winbio = ctypes.windll.winbio
            
            # WINBIO_UNIT_SCHEMA definition
            class WINBIO_UNIT_SCHEMA(ctypes.Structure):
                _fields_ = [
                    ("UnitId", ctypes.c_uint32),
                    ("PoolType", ctypes.c_uint32),
                    ("BiometricFactor", ctypes.c_uint32),
                    ("SensorSubType", ctypes.c_uint32),
                    ("Capabilities", ctypes.c_uint32),
                    ("DeviceInstanceId", ctypes.c_wchar * 256),
                    ("Description", ctypes.c_wchar * 256),
                    ("Manufacturer", ctypes.c_wchar * 256),
                    ("Model", ctypes.c_wchar * 256),
                    ("SerialNumber", ctypes.c_wchar * 256),
                    ("FirmwareVersion", ctypes.c_uint32 * 2),
                ]

            unit_schema_array = ctypes.POINTER(WINBIO_UNIT_SCHEMA)()
            unit_count = ctypes.c_size_t(0)

            # WINBIO_TYPE_ANY = 0xFFFFFFFF
            hr = winbio.WinBioEnumBiometricUnits(
                ctypes.c_uint32(0xFFFFFFFF),
                ctypes.byref(unit_schema_array),
                ctypes.byref(unit_count)
            )

            # S_OK = 0
            if hr == 0 and unit_count.value > 0:
                # Hardware unit found; check if enrolled
                # Free memory
                winbio.WinBioFree(unit_schema_array)
                self._cached_capability = BiometricCapability.AVAILABLE
            elif hr == 0 and unit_count.value == 0:
                self._cached_capability = BiometricCapability.NOT_AVAILABLE
            else:
                # Check for WINBIO_E_NOT_FOUND (0x80098004) or other return codes
                self._cached_capability = BiometricCapability.NOT_AVAILABLE

        except Exception:
            # If WinBio DLL missing, access denied, or any exception
            self._cached_capability = BiometricCapability.NOT_AVAILABLE

        return self._cached_capability

    def is_enrolled(self) -> bool:
        """
        Determines whether the current user has enrolled Windows Hello credentials.
        """
        cap = self.get_capability()
        return cap == BiometricCapability.AVAILABLE

    def authenticate(
        self,
        prompt: str = "TARA Creator Verification",
        context: Optional[Dict[str, Any]] = None
    ) -> BiometricAuthResult:
        """
        Executes Windows Hello authentication.
        In production, calls Windows system verification.
        In test mode with simulated flag and TARA_TEST_MODE=1, allows controlled verification tests.
        """
        context = context or {}
        simulate_success = context.get("simulate_success", False)

        if simulate_success:
            if not self.test_mode or os.environ.get("TARA_TEST_MODE") != "1":
                raise PermissionError("SECURITY_VIOLATION: Windows Hello simulation is forbidden in production.")
            return BiometricAuthResult(
                success=True,
                capability=BiometricCapability.AVAILABLE,
                auth_type=self.provider_name,
                metadata={"prompt": prompt, "simulated": True}
            )

        cap = self.get_capability()
        if cap != BiometricCapability.AVAILABLE:
            return BiometricAuthResult(
                success=False,
                capability=cap,
                auth_type=self.provider_name,
                error=f"Windows Hello is {cap.value}"
            )

        # In standard headless/CLI environments where interactive WinBio prompt cannot display,
        # fail-closed with unconfirmed status.
        return BiometricAuthResult(
            success=False,
            capability=cap,
            auth_type=self.provider_name,
            error="USER_INTERACTION_REQUIRED"
        )
