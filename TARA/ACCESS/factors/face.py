"""
TARA/ACCESS/biometrics/face.py

Facial Recognition Biometric Provider.
Checks specifically for IR/depth facial sensor presence and enrollment.
"""

import os
import sys
import ctypes
from typing import Dict, Any, Optional
from .provider import BiometricProvider, BiometricCapability, BiometricType, BiometricAuthResult

WINBIO_TYPE_FACIAL_FEATURES = 0x00000010


class FaceProvider(BiometricProvider):
    """
    Dedicated Facial Recognition biometric sensor provider.
    """

    def __init__(self, test_mode: bool = False):
        super().__init__(provider_name=BiometricType.FACE.value, test_mode=test_mode)
        self._cached_capability: Optional[BiometricCapability] = None

    def get_capability(self) -> BiometricCapability:
        if self._cached_capability is not None:
            return self._cached_capability

        if sys.platform != "win32":
            self._cached_capability = BiometricCapability.NOT_AVAILABLE
            return self._cached_capability

        try:
            winbio = ctypes.windll.winbio

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

            hr = winbio.WinBioEnumBiometricUnits(
                ctypes.c_uint32(WINBIO_TYPE_FACIAL_FEATURES),
                ctypes.byref(unit_schema_array),
                ctypes.byref(unit_count)
            )

            if hr == 0 and unit_count.value > 0:
                winbio.WinBioFree(unit_schema_array)
                self._cached_capability = BiometricCapability.AVAILABLE
            else:
                self._cached_capability = BiometricCapability.NOT_AVAILABLE

        except Exception:
            self._cached_capability = BiometricCapability.NOT_AVAILABLE

        return self._cached_capability

    def is_enrolled(self) -> bool:
        return self.get_capability() == BiometricCapability.AVAILABLE

    def authenticate(
        self,
        prompt: str = "Facial Recognition Required",
        context: Optional[Dict[str, Any]] = None
    ) -> BiometricAuthResult:
        context = context or {}
        if context.get("simulate_success", False):
            if not self.test_mode or os.environ.get("TARA_TEST_MODE") != "1":
                raise PermissionError("SECURITY_VIOLATION: Biometric simulation is forbidden in production.")
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
                error=f"Facial sensor is {cap.value}"
            )

        return BiometricAuthResult(
            success=False,
            capability=cap,
            auth_type=self.provider_name,
            error="USER_INTERACTION_REQUIRED"
        )
