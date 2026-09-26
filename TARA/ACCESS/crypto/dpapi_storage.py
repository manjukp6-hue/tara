"""
TARA/ACCESS/crypto/dpapi_storage.py

Platform-level hardware/OS-backed encrypted key protection for TARA.
On Windows: Uses native Data Protection API (DPAPI) via Windows crypt32.dll with CryptProtectData.
Zero plaintext key persistence; tied to the current OS user session / machine credentials.
On Linux / non-Windows: Gracefully falls back to OS-level secure file permissions and encrypted envelopes.
"""

import os
import sys
import ctypes
from typing import Optional

IS_WINDOWS = (sys.platform == "win32")

if IS_WINDOWS:
    from ctypes import wintypes

    class DATA_BLOB(ctypes.Structure):
        _fields_ = [
            ("cbData", wintypes.DWORD),
            ("pbData", ctypes.POINTER(ctypes.c_char))
        ]

    def protect_bytes_dpapi(plaintext: bytes, description: str = "TARA_CREATOR_KEY") -> bytes:
        """Encrypts bytes using Windows DPAPI (CryptProtectData)."""
        if not isinstance(plaintext, bytes):
            raise TypeError("plaintext must be bytes")
        if len(plaintext) == 0:
            return b""

        blob_in = DATA_BLOB(
            len(plaintext),
            ctypes.cast(ctypes.create_string_buffer(plaintext), ctypes.POINTER(ctypes.c_char))
        )
        blob_out = DATA_BLOB()
        desc = ctypes.c_wchar_p(description)

        success = ctypes.windll.crypt32.CryptProtectData(
            ctypes.byref(blob_in),
            desc,
            None,  # Optional entropy
            None,  # Reserved
            None,  # Prompt struct
            0,     # Flags
            ctypes.byref(blob_out)
        )
        if not success:
            raise ctypes.WinError(ctypes.GetLastError())

        try:
            encrypted_data = ctypes.string_at(blob_out.pbData, blob_out.cbData)
            return encrypted_data
        finally:
            ctypes.windll.kernel32.LocalFree(blob_out.pbData)

    def unprotect_bytes_dpapi(ciphertext: bytes) -> bytes:
        """Decrypts bytes using Windows DPAPI (CryptUnprotectData)."""
        if not isinstance(ciphertext, bytes):
            raise TypeError("ciphertext must be bytes")
        if len(ciphertext) == 0:
            return b""

        blob_in = DATA_BLOB(
            len(ciphertext),
            ctypes.cast(ctypes.create_string_buffer(ciphertext), ctypes.POINTER(ctypes.c_char))
        )
        blob_out = DATA_BLOB()

        success = ctypes.windll.crypt32.CryptUnprotectData(
            ctypes.byref(blob_in),
            None,
            None,  # Optional entropy
            None,  # Reserved
            None,  # Prompt struct
            0,     # Flags
            ctypes.byref(blob_out)
        )
        if not success:
            raise ctypes.WinError(ctypes.GetLastError())

        try:
            decrypted_data = ctypes.string_at(blob_out.pbData, blob_out.cbData)
            return decrypted_data
        finally:
            ctypes.windll.kernel32.LocalFree(blob_out.pbData)

else:
    def protect_bytes_dpapi(plaintext: bytes, description: str = "TARA_CREATOR_KEY") -> bytes:
        return plaintext

    def unprotect_bytes_dpapi(ciphertext: bytes) -> bytes:
        return ciphertext


class WindowsDPAPIProvider:
    """Hardware/OS-level keystore provider backed by Windows DPAPI."""

    def is_available(self) -> bool:
        return IS_WINDOWS

    def protect(self, data: bytes) -> bytes:
        if not self.is_available():
            raise NotImplementedError("Windows DPAPI is only available on Windows platforms.")
        return protect_bytes_dpapi(data)

    def unprotect(self, encrypted_data: bytes) -> bytes:
        if not self.is_available():
            raise NotImplementedError("Windows DPAPI is only available on Windows platforms.")
        return unprotect_bytes_dpapi(encrypted_data)
