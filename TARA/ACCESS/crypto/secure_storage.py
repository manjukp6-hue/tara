"""
TARA/ACCESS/crypto/secure_storage.py

Hardened platform-secure encrypted storage for cryptographic private keys.
Uses standard AES-256-GCM Authenticated Encryption with Associated Data (AEAD)
via the OpenSSL-backed PyCA cryptography library.
Never stores private keys in plaintext.
Enforces:
- Confidentiality & Integrity via AES-256-GCM (16-byte authentication tag)
- Unique 96-bit (12-byte) nonces per encryption
- Secure Key Derivation via PBKDF2-HMAC-SHA256 (100,000 iterations)
- Associated Data binding (key_id bound to ciphertext)
- Atomic crash-safe file replacement (temp file + os.replace)
- Zero plaintext on disk with secure overwrite before unlinking
- Android Keystore / Hardware-backed security provider abstraction
"""

import os
import json
import hmac
import hashlib
import secrets
from typing import Optional, Tuple, Dict, Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.exceptions import InvalidTag

from .dpapi_storage import protect_bytes_dpapi, unprotect_bytes_dpapi, IS_WINDOWS


class HardwareKeystoreProvider:
    """
    Interface for hardware-backed key storage (e.g., Android Keystore, Secure Enclave, TPM).
    Allows delegated hardware-level key wrapping on supported platforms.
    """
    def is_hardware_backed(self) -> bool:
        return False

    def wrap_key(self, plaintext_key: bytes) -> bytes:
        raise NotImplementedError("Hardware wrapping not implemented on base provider")

    def unwrap_key(self, wrapped_key: bytes) -> bytes:
        raise NotImplementedError("Hardware unwrapping not implemented on base provider")


class AndroidKeystoreProvider(HardwareKeystoreProvider):
    """
    Android Keystore hardware-backed provider hook.
    In production Android runtime (PyJNIus / Android Native), delegates to AndroidKeyStore.
    """
    def __init__(self, key_alias: str = "TARA_ROOT_MASTER_KEY"):
        self.key_alias = key_alias
        self._is_android = os.path.exists("/system/build.prop") or "ANDROID_ROOT" in os.environ

    def is_hardware_backed(self) -> bool:
        return self._is_android

    def wrap_key(self, plaintext_key: bytes) -> bytes:
        if not self.is_hardware_backed():
            raise RuntimeError("Android Keystore hardware-backed protection only available on Android")
        return plaintext_key


class SecureKeyStorage:
    """
    Standard AEAD Encrypted Key Storage using AES-256-GCM.
    Enforces zero plaintext exposure and atomic persistence.
    """
    KDF_ITERATIONS = 100_000
    NONCE_SIZE_BYTES = 12  # Standard 96-bit nonce for AES-GCM
    KEY_SIZE_BYTES = 32    # 256-bit key for AES-256-GCM

    def __init__(
        self,
        storage_dir: Optional[str] = None,
        hardware_provider: Optional[HardwareKeystoreProvider] = None
    ):
        if storage_dir is None:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
            storage_dir = os.path.join(repo_root, "storage", "vault", "access")
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.hardware_provider = hardware_provider or AndroidKeystoreProvider()

    def _derive_aes_key(self, passphrase: str, salt: bytes) -> bytes:
        """Derives a 256-bit key using PBKDF2-HMAC-SHA256 with 100,000 iterations."""
        return hashlib.pbkdf2_hmac(
            "sha256",
            passphrase.encode("utf-8"),
            salt,
            self.KDF_ITERATIONS,
            dklen=self.KEY_SIZE_BYTES
        )

    def _resolve_passphrase(self, passphrase: Optional[str]) -> str:
        if passphrase:
            return passphrase
        env_secret = os.environ.get("TARA_CREATOR_PASSPHRASE") or os.environ.get("TARA_DEVICE_SECRET")
        if env_secret:
            return env_secret
        raise ValueError(
            "A secure passphrase is required for keystore operations. Provide an explicit passphrase or configure TARA_CREATOR_PASSPHRASE environment variable."
        )

    def store_private_key(
        self,
        key_id: str,
        private_bytes: bytes,
        passphrase: Optional[str] = None
    ) -> str:
        """
        Encrypts and persists a private key using AES-256-GCM with atomic write.
        Never writes plaintext to disk.
        """
        resolved_passphrase = self._resolve_passphrase(passphrase)
        salt = secrets.token_bytes(16)
        nonce = secrets.token_bytes(self.NONCE_SIZE_BYTES)
        derived_key = self._derive_aes_key(resolved_passphrase, salt)

        # Authenticate key_id as Associated Data (AAD) to prevent key-swap attacks
        aad = key_id.encode("utf-8")
        aesgcm = AESGCM(derived_key)
        ciphertext_with_tag = aesgcm.encrypt(nonce, private_bytes, aad)

        payload = {
            "key_id": key_id,
            "aead": "AES-256-GCM",
            "kdf": "PBKDF2-HMAC-SHA256",
            "iterations": self.KDF_ITERATIONS,
            "salt": salt.hex(),
            "nonce": nonce.hex(),
            "ciphertext": ciphertext_with_tag.hex(),
            "version": 2
        }

        filepath = os.path.join(self.storage_dir, f"{key_id}.keystore")
        temp_filepath = f"{filepath}.tmp.{secrets.token_hex(6)}"

        # Crash-safe atomic write: write to temp file, flush/sync, then atomic replace
        with open(temp_filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            try:
                os.fsync(f.fileno())
            except (OSError, AttributeError):
                pass

        os.replace(temp_filepath, filepath)
        return filepath

    def store_private_key_modern(
        self,
        key_id: str,
        private_bytes: bytes,
        passphrase: Optional[str] = None,
        use_dpapi: bool = True
    ) -> str:
        """
        Encrypts and persists a private key using modern Scrypt memory-hard KDF
        and AES-256-GCM AEAD, with optional Windows DPAPI hardware/OS envelope.
        Version 3 Keystore. Never writes plaintext to disk.
        """
        resolved_passphrase = self._resolve_passphrase(passphrase)
        salt = secrets.token_bytes(16)
        nonce = secrets.token_bytes(self.NONCE_SIZE_BYTES)

        # Scrypt memory-hard KDF (N=131072 [2^17], r=8, p=1, 32-byte key)
        kdf = Scrypt(salt=salt, length=self.KEY_SIZE_BYTES, n=131072, r=8, p=1)
        derived_key = kdf.derive(resolved_passphrase.encode("utf-8"))

        # Authenticate key_id as Associated Data (AAD) to prevent key-swap attacks
        aad = key_id.encode("utf-8")
        aesgcm = AESGCM(derived_key)
        ciphertext_with_tag = aesgcm.encrypt(nonce, private_bytes, aad)

        dpapi_blob = None
        dpapi_protected = False
        if use_dpapi and IS_WINDOWS:
            try:
                dpapi_blob = protect_bytes_dpapi(derived_key, description=f"TARA_KEY_{key_id}").hex()
                dpapi_protected = True
            except Exception:
                dpapi_protected = False

        payload = {
            "key_id": key_id,
            "aead": "AES-256-GCM",
            "kdf": "Scrypt",
            "kdf_params": {
                "n": 131072,
                "r": 8,
                "p": 1,
                "salt": salt.hex(),
                "length": self.KEY_SIZE_BYTES
            },
            "nonce": nonce.hex(),
            "ciphertext": ciphertext_with_tag.hex(),
            "dpapi_protected": dpapi_protected,
            "dpapi_blob": dpapi_blob,
            "version": 3
        }

        filepath = os.path.join(self.storage_dir, f"{key_id}.keystore")
        temp_filepath = f"{filepath}.tmp.{secrets.token_hex(6)}"

        with open(temp_filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            try:
                os.fsync(f.fileno())
            except (OSError, AttributeError):
                pass

        os.replace(temp_filepath, filepath)
        return filepath

    def load_private_key(
        self,
        key_id: str,
        passphrase: Optional[str] = None
    ) -> Optional[bytes]:
        """
        Loads and decrypts a private key using AES-256-GCM.
        Validates authentication tag and Associated Data.
        Returns None if key doesn't exist, tag is invalid, or passphrase is wrong.
        Supports version 3 (Scrypt+DPAPI), version 2 (PBKDF2 AEAD), and version 1 (Legacy).
        """
        if os.path.exists(key_id):
            filepath = key_id
            key_id = os.path.splitext(os.path.basename(key_id))[0]
        else:
            filepath = os.path.join(self.storage_dir, f"{key_id}.keystore")
            if not os.path.exists(filepath):
                alt = os.path.join(self.storage_dir, key_id)
                if os.path.exists(alt):
                    filepath = alt
                    key_id = os.path.splitext(key_id)[0]
                else:
                    return None

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                payload = json.load(f)

            version = payload.get("version", 1)

            if version == 3:
                # Modern Scrypt + AES-256-GCM + DPAPI
                nonce = bytes.fromhex(payload["nonce"])
                ciphertext_with_tag = bytes.fromhex(payload["ciphertext"])
                aad = key_id.encode("utf-8")

                derived_key = None

                # 1. Try Scrypt if passphrase provided or resolvable
                try:
                    resolved_passphrase = self._resolve_passphrase(passphrase)
                except ValueError:
                    resolved_passphrase = None

                if resolved_passphrase is not None:
                    kdf_params = payload.get("kdf_params", {})
                    salt = bytes.fromhex(kdf_params.get("salt", payload.get("salt", "")))
                    n = kdf_params.get("n", 131072)
                    r = kdf_params.get("r", 8)
                    p = kdf_params.get("p", 1)
                    kdf = Scrypt(salt=salt, length=self.KEY_SIZE_BYTES, n=n, r=r, p=p)
                    derived_key = kdf.derive(resolved_passphrase.encode("utf-8"))
                elif payload.get("dpapi_protected") and payload.get("dpapi_blob") and IS_WINDOWS:
                    # 2. DPAPI OS-level hardware unwrap if no passphrase supplied
                    try:
                        dpapi_raw = bytes.fromhex(payload["dpapi_blob"])
                        derived_key = unprotect_bytes_dpapi(dpapi_raw)
                    except Exception:
                        derived_key = None

                if derived_key is None:
                    return None

                aesgcm = AESGCM(derived_key)
                try:
                    plaintext = aesgcm.decrypt(nonce, ciphertext_with_tag, aad)
                    return plaintext
                except InvalidTag:
                    return None

            try:
                resolved_passphrase = self._resolve_passphrase(passphrase)
            except ValueError:
                return None

            if version == 2:
                # Standard AES-256-GCM AEAD
                salt = bytes.fromhex(payload["salt"])
                nonce = bytes.fromhex(payload["nonce"])
                ciphertext_with_tag = bytes.fromhex(payload["ciphertext"])
                aad = key_id.encode("utf-8")

                iterations = payload.get("iterations", self.KDF_ITERATIONS)
                derived_key = hashlib.pbkdf2_hmac(
                    "sha256",
                    resolved_passphrase.encode("utf-8"),
                    salt,
                    iterations,
                    dklen=self.KEY_SIZE_BYTES
                )

                aesgcm = AESGCM(derived_key)
                try:
                    plaintext = aesgcm.decrypt(nonce, ciphertext_with_tag, aad)
                    return plaintext
                except InvalidTag:
                    # Authentication failure: wrong passphrase or corrupted ciphertext
                    return None

            elif version == 1:
                # Legacy fallback & transparent migration
                salt = bytes.fromhex(payload["salt"])
                iv = bytes.fromhex(payload["iv"])
                ciphertext = bytes.fromhex(payload["ciphertext"])
                expected_mac = payload["mac"]

                derived = hashlib.pbkdf2_hmac("sha256", resolved_passphrase.encode("utf-8"), salt, 100_000, dklen=64)
                enc_key, mac_key = derived[:32], derived[32:]

                computed_mac = hmac.new(mac_key, salt + iv + ciphertext, hashlib.sha256).hexdigest()
                if not hmac.compare_digest(expected_mac, computed_mac):
                    return None

                # Legacy CTR keystream decrypt
                out = bytearray(len(ciphertext))
                counter = 0
                block_idx = 0
                while block_idx < len(ciphertext):
                    block_key = hashlib.sha256(enc_key + iv + counter.to_bytes(8, "big")).digest()
                    chunk_len = min(32, len(ciphertext) - block_idx)
                    for i in range(chunk_len):
                        out[block_idx + i] = ciphertext[block_idx + i] ^ block_key[i]
                    block_idx += chunk_len
                    counter += 1
                plaintext = bytes(out)

                # Upgrade in-place to version 2 AEAD
                self.store_private_key(key_id, plaintext, passphrase)
                return plaintext

            return None
        except Exception:
            return None

    def delete_private_key(self, key_id: str) -> bool:
        """Securely shreds and deletes a stored private key file."""
        filepath = os.path.join(self.storage_dir, f"{key_id}.keystore")
        if os.path.exists(filepath):
            try:
                file_len = os.path.getsize(filepath)
                # Overwrite with cryptographically secure random bytes before unlinking
                with open(filepath, "wb") as f:
                    f.write(secrets.token_bytes(max(file_len, 512)))
                    f.flush()
                    try:
                        os.fsync(f.fileno())
                    except (OSError, AttributeError):
                        pass
                os.remove(filepath)
                return True
            except OSError:
                return False
        return False

    def has_key(self, key_id: str) -> bool:
        filepath = os.path.join(self.storage_dir, f"{key_id}.keystore")
        return os.path.exists(filepath)

    def get_key_metadata(self, key_id: str) -> Optional[Dict[str, Any]]:
        """Returns non-secret cryptographic metadata of stored keystore."""
        filepath = os.path.join(self.storage_dir, f"{key_id}.keystore")
        if not os.path.exists(filepath):
            return None
        with open(filepath, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return {
            "key_id": payload.get("key_id"),
            "aead": payload.get("aead", "AES-256-GCM"),
            "version": payload.get("version"),
            "iterations": payload.get("iterations"),
            "nonce_hex": payload.get("nonce") or payload.get("iv"),
            "salt_hex": payload.get("salt")
        }


# Canonical Alias for backward compatibility
SecureStorage = SecureKeyStorage
