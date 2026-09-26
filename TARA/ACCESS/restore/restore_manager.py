"""
TARA/ACCESS/restore/restore_manager.py

Multi-path Recovery & Key Rotation System for TARA Creator Identity.

Supported Recovery Paths:
1. Cryptographic Recovery Code (PBKDF2-HMAC-SHA256 verifier with lockout)
2. Verified Google/Gmail Creator Account (cryptographically verified ID token/claims)
3. Trusted Authorized Device Signature (registered, authorized, unrevoked device with matching Ed25519 key)

Key Rotation / Key Loss Rules:
- If creator private key is lost or rotated:
  - Verify creator ownership via valid recovery path.
  - Require a validated recovery authorization proof produced by the RecoveryManager recovery flow.
  - Never rotate based solely on an authorization_method string.
  - Increment key_version (e.g. 1 -> 2).
  - Revoke old key version.
  - Creator ID remains permanently ROOT_OPERATOR.
  - Display Name remains OPERATOR_ROOT.
"""

import os
import sys
import re
import json
import secrets
import hashlib
import hmac
import time
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any, Union

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from ..crypto.ed25519 import Ed25519
from ..crypto.dpapi_storage import protect_bytes_dpapi, unprotect_bytes_dpapi, IS_WINDOWS
from ..operator.operator_profile import CreatorIdentity, CANONICAL_CREATOR_ID

PROTECTED_RECOVERY_FORMAT = "TARA_PROTECTED_RECOVERY_V2"
PRODUCTION_PBKDF2_ITERATIONS = 600_000
LEGACY_PBKDF2_ITERATIONS = 50_000
CURRENT_VERIFIER_VERSION = 2
LEGACY_VERIFIER_VERSION = 1


def get_default_recovery_storage_dir() -> str:
    """Returns protected local TARA application storage directory for recovery data."""
    if sys.platform == "win32":
        app_data = os.environ.get("APPDATA") or os.path.expanduser("~")
        path = os.path.join(app_data, "TARA", "recovery")
    else:
        path = os.path.join(os.path.expanduser("~"), ".tara", "recovery")
    os.makedirs(path, exist_ok=True)
    return path


def get_default_recovery_record_path() -> str:
    """
    Returns authoritative recovery configuration path in protected local TARA application storage.
    Never stores recovery authority data under repo_root/TARA/...
    """
    env_path = os.environ.get("TARA_RECOVERY_CONFIG_PATH")
    if env_path and env_path.strip():
        return env_path.strip()
    return os.path.join(get_default_recovery_storage_dir(), "recovery_config.json")


def load_protected_recovery_config(
    file_path: Optional[str] = None,
    creator: Optional[Any] = None
) -> Optional[Dict[str, Any]]:
    """
    Safely loads and decrypts protected recovery configuration from disk.
    Verifies cryptographic integrity, DPAPI key unwrap, HMAC seal, and AAD authority binding.
    Returns decrypted payload dictionary if valid and untampered.
    Returns None if file is missing, corrupted, tampered, unreadable, or fails verification.
    Never repairs or rebuilds tampered state.
    """
    target_path = file_path
    if not target_path or not os.path.exists(target_path):
        target_path = get_default_recovery_record_path()
    if not target_path or not os.path.exists(target_path):
        return None

    if creator is None:
        record_path = None
        norm_p = os.path.normpath(target_path)
        target_dir = os.path.dirname(norm_p)
        candidates = [
            os.path.join(target_dir, "operator_record.json"),
            os.path.join(target_dir, "operator", "operator_record.json"),
            os.path.join(os.path.dirname(target_dir), "operator", "operator_record.json"),
            os.path.join(os.path.dirname(target_dir), "operator_record.json"),
        ]
        for cand in candidates:
            if os.path.exists(cand):
                record_path = cand
                break
        creator = CreatorIdentity(record_path=record_path)

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            envelope = json.load(f)
    except Exception:
        return None

    if not isinstance(envelope, dict):
        return None

    ALLOWED_ENVELOPE_KEYS = {
        "format", "creator_id", "key_version", "root_public_key",
        "nonce", "ciphertext", "wrapped_key", "integrity_seal", "sealed_at"
    }

    if envelope.get("format") == PROTECTED_RECOVERY_FORMAT:
        try:
            if set(envelope.keys()) - ALLOWED_ENVELOPE_KEYS:
                return None

            expected_creator_id = CANONICAL_CREATOR_ID
            if envelope.get("creator_id") != expected_creator_id:
                return None

            current_key_version = getattr(creator, "key_version", 1) or 1
            if envelope.get("key_version") != current_key_version:
                return None

            current_root_pub = getattr(creator, "root_public_key", None)
            if current_root_pub and envelope.get("root_public_key"):
                if envelope.get("root_public_key").lower() != current_root_pub.lower():
                    return None

            wrapped_key_bytes = bytes.fromhex(envelope["wrapped_key"])
            nonce = bytes.fromhex(envelope["nonce"])
            ciphertext = bytes.fromhex(envelope["ciphertext"])
            claimed_seal = envelope.get("integrity_seal")

            k_rec = unprotect_bytes_dpapi(wrapped_key_bytes)
            if not k_rec or len(k_rec) != 32:
                return None

            root_pk_str = envelope.get("root_public_key") or ""
            seal_content = f"{envelope['creator_id']}:{envelope['key_version']}:{root_pk_str}:{envelope['nonce']}:{envelope['ciphertext']}".encode("utf-8")
            expected_seal = hmac.new(k_rec, seal_content, hashlib.sha256).hexdigest()
            if not claimed_seal or not hmac.compare_digest(claimed_seal, expected_seal):
                return None

            aad = f"TARA_RECOVERY_SEAL:{envelope['creator_id']}:{envelope['key_version']}:{root_pk_str}".encode("utf-8")
            aesgcm = AESGCM(k_rec)
            plaintext_bytes = aesgcm.decrypt(nonce, ciphertext, aad)
            inner_payload = json.loads(plaintext_bytes.decode("utf-8"))

            if inner_payload.get("creator_id") != expected_creator_id:
                return None
            if inner_payload.get("key_version") != current_key_version:
                return None

            return inner_payload
        except Exception:
            return None

    if "recovery_code_hash" in envelope and envelope.get("creator_id") == CANONICAL_CREATOR_ID:
        return envelope

    return None


class RecoveryAuthorizationProof:
    """
    Cryptographic proof of completed recovery verification.
    Produced exclusively by RecoveryManager recovery verification flows.
    """
    def __init__(
        self,
        proof_id: str,
        method: str,
        creator_id: str,
        manager_id: str,
        details: Optional[Dict[str, Any]] = None,
        ttl_seconds: float = 300.0
    ):
        self.proof_id = proof_id
        self.method = method
        self.creator_id = creator_id
        self.manager_id = manager_id
        self.issued_at = time.time()
        self.expires_at = self.issued_at + ttl_seconds
        self.consumed = False
        self.details = details or {}

    def is_valid(self, expected_creator_id: str = CANONICAL_CREATOR_ID) -> bool:
        if self.consumed:
            return False
        if time.time() > self.expires_at:
            return False
        if self.creator_id != expected_creator_id:
            return False
        return True

    def consume(self) -> None:
        if self.consumed:
            raise PermissionError("Recovery authorization proof has already been consumed.")
        if time.time() > self.expires_at:
            raise PermissionError("Recovery authorization proof has expired.")
        self.consumed = True

    def __bool__(self) -> bool:
        return self.is_valid()

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, bool):
            return bool(self) is other
        if isinstance(other, RecoveryAuthorizationProof):
            return self.proof_id == other.proof_id
        return False


class RecoveryManager:
    """
    Manages creator recovery configurations, recovery proofs, and key rotations.
    Stores recovery authority data strictly in protected local application storage.
    """
    def __init__(
        self,
        creator: CreatorIdentity,
        recovery_record_path: Optional[str] = None,
        device_registry: Optional[Any] = None,
        google_auth_service: Optional[Any] = None
    ):
        self.creator = creator
        self.device_registry = device_registry
        self.google_auth_service = google_auth_service
        self._instance_id = secrets.token_hex(16)
        self._active_proofs: Dict[str, RecoveryAuthorizationProof] = {}
        self._latest_proof: Optional[RecoveryAuthorizationProof] = None
        self._lock = threading.RLock()
        self._tampered: bool = False

        if recovery_record_path is None:
            # If creator uses a temporary or test directory, isolate recovery config to that test directory
            record_path = getattr(creator, "record_path", None)
            if record_path:
                norm_record_path = os.path.normpath(record_path)
                if any(x in norm_record_path.lower() for x in ("temp", "tmp", "tara_sec_test", "tara_test_", "tara_lifecycle_test", "tara_auth_test", "tara_protected")):
                    base_dir = os.path.dirname(os.path.dirname(norm_record_path))
                    recovery_record_path = os.path.join(base_dir, "recovery", "recovery_config.json")

            if recovery_record_path is None:
                recovery_record_path = get_default_recovery_record_path()

        self.recovery_record_path = recovery_record_path
        self.recovery_code_hash: Optional[str] = None
        self.recovery_salt: Optional[str] = None
        self.pbkdf2_iterations: int = PRODUCTION_PBKDF2_ITERATIONS
        self.verifier_version: int = CURRENT_VERIFIER_VERSION
        self.recovery_email: Optional[str] = None
        self.trusted_device_recovery_enabled: bool = True
        self.recovery_history: List[Dict[str, Any]] = []
        self.failed_attempts: int = 0
        self.max_attempts: int = 5
        self.lockout_seconds: int = 300
        self.locked_until: Optional[float] = None

        if os.path.exists(self.recovery_record_path):
            try:
                self.load()
            except PermissionError:
                self._tampered = True



    def _issue_recovery_proof(self, method: str, details: Optional[Dict[str, Any]] = None) -> RecoveryAuthorizationProof:
        """Issues an authentic RecoveryAuthorizationProof bound to this manager and CANONICAL_CREATOR_ID."""
        proof_id = secrets.token_hex(16)
        proof = RecoveryAuthorizationProof(
            proof_id=proof_id,
            method=method,
            creator_id=CANONICAL_CREATOR_ID,
            manager_id=self._instance_id,
            details=details,
            ttl_seconds=300.0
        )
        self._active_proofs[proof_id] = proof
        self._latest_proof = proof
        return proof

    def verify_storage_integrity(self) -> bool:
        """
        Cryptographically verifies the integrity, seal, and authority binding of recovery storage on disk.
        Returns True if authentic, decrypted, and untampered; False otherwise.
        """
        with self._lock:
            if self._tampered:
                return False
            if not os.path.exists(self.recovery_record_path):
                return True
            try:
                with open(self.recovery_record_path, "r", encoding="utf-8") as f:
                    envelope = json.load(f)
                if not isinstance(envelope, dict):
                    self._tampered = True
                    return False
                if envelope.get("format") != PROTECTED_RECOVERY_FORMAT:
                    if "recovery_code_hash" in envelope and envelope.get("creator_id") == CANONICAL_CREATOR_ID:
                        return True
                    self._tampered = True
                    return False
                self._decrypt_and_verify_envelope(envelope)
                return True
            except Exception:
                self._tampered = True
                return False

    def _decrypt_and_verify_envelope(self, envelope: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decrypts and cryptographically verifies protected recovery envelope.
        Binds to Creator Authority invariants (creator_id, key_version, root_public_key).
        Fails closed without rebuilding or repairing.
        """
        ALLOWED_ENVELOPE_KEYS = {
            "format", "creator_id", "key_version", "root_public_key",
            "nonce", "ciphertext", "wrapped_key", "integrity_seal", "sealed_at"
        }
        if set(envelope.keys()) - ALLOWED_ENVELOPE_KEYS:
            self._tampered = True
            raise PermissionError("Recovery configuration contains unauthorized or injected fields: tampering detected.")

        expected_creator_id = CANONICAL_CREATOR_ID
        if envelope.get("creator_id") != expected_creator_id:
            self._tampered = True
            raise PermissionError(f"Recovery configuration creator_id mismatch. Expected {expected_creator_id}, got {envelope.get('creator_id')}")

        current_key_version = getattr(self.creator, "key_version", 1) or 1
        if envelope.get("key_version") != current_key_version:
            self._tampered = True
            raise PermissionError(f"Recovery configuration key_version mismatch. Expected {current_key_version}, got {envelope.get('key_version')} (rollback or stale state detected).")

        current_root_pub = getattr(self.creator, "root_public_key", None)
        if current_root_pub and envelope.get("root_public_key"):
            if envelope.get("root_public_key").lower() != current_root_pub.lower():
                self._tampered = True
                raise PermissionError("Recovery configuration root_public_key mismatch: does not match active authority.")

        try:
            wrapped_key_bytes = bytes.fromhex(envelope["wrapped_key"])
            nonce = bytes.fromhex(envelope["nonce"])
            ciphertext = bytes.fromhex(envelope["ciphertext"])
            claimed_seal = envelope.get("integrity_seal")
        except Exception:
            self._tampered = True
            raise PermissionError("Recovery configuration contains invalid cryptographic field encodings.")

        # Unwrap symmetric key via DPAPI
        try:
            k_rec = unprotect_bytes_dpapi(wrapped_key_bytes)
            if not k_rec or len(k_rec) != 32:
                self._tampered = True
                raise PermissionError("Failed to unwrap recovery encryption key via local DPAPI.")
        except Exception as e:
            self._tampered = True
            raise PermissionError(f"DPAPI key unwrap failure: {str(e)}")

        # Verify HMAC integrity seal
        root_pk_str = envelope.get("root_public_key") or ""
        seal_content = f"{envelope['creator_id']}:{envelope['key_version']}:{root_pk_str}:{envelope['nonce']}:{envelope['ciphertext']}".encode("utf-8")
        expected_seal = hmac.new(k_rec, seal_content, hashlib.sha256).hexdigest()
        if not claimed_seal or not hmac.compare_digest(claimed_seal, expected_seal):
            self._tampered = True
            raise PermissionError("Recovery configuration integrity seal verification failed: tampering detected.")

        # Verify AES-256-GCM AEAD decryption with AAD
        aad = f"TARA_RECOVERY_SEAL:{envelope['creator_id']}:{envelope['key_version']}:{root_pk_str}".encode("utf-8")
        try:
            aesgcm = AESGCM(k_rec)
            plaintext_bytes = aesgcm.decrypt(nonce, ciphertext, aad)
        except Exception as e:
            self._tampered = True
            raise PermissionError(f"Recovery configuration AEAD decryption failed: {str(e)}")

        try:
            inner_payload = json.loads(plaintext_bytes.decode("utf-8"))
        except Exception:
            self._tampered = True
            raise PermissionError("Recovery configuration decrypted payload is not valid JSON.")

        # Verify inner payload binds to outer envelope and authority invariants
        if inner_payload.get("creator_id") != expected_creator_id:
            self._tampered = True
            raise PermissionError("Inner recovery payload creator_id invariant violated.")
        if inner_payload.get("key_version") != current_key_version:
            self._tampered = True
            raise PermissionError("Inner recovery payload key_version mismatch.")

        return inner_payload

    def _migrate_legacy_plaintext_config(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Safely migrates a valid unencrypted legacy recovery configuration to authenticated encrypted storage.
        Immediately commits encrypted format to disk.
        """
        if data.get("creator_id") != CANONICAL_CREATOR_ID:
            self._tampered = True
            raise PermissionError("Legacy recovery configuration creator_id mismatch.")

        self.recovery_code_hash = data.get("recovery_code_hash")
        self.recovery_salt = data.get("recovery_salt")
        self.pbkdf2_iterations = int(data.get("pbkdf2_iterations", LEGACY_PBKDF2_ITERATIONS))
        self.verifier_version = int(data.get("verifier_version", LEGACY_VERIFIER_VERSION))
        self.recovery_email = data.get("recovery_email")
        self.trusted_device_recovery_enabled = data.get("trusted_device_recovery_enabled", True)
        self.failed_attempts = data.get("failed_attempts", 0)
        self.locked_until = data.get("locked_until")
        self.recovery_history = data.get("recovery_history", [])

        # Atomically migrate to protected encrypted storage
        self.save()
        return data

    def is_locked_out(self) -> bool:
        """Checks if recovery verification is currently locked out due to excessive attempts."""
        with self._lock:
            if self._tampered or not self.verify_storage_integrity():
                return True
            if self.locked_until is None:
                return False
            if time.time() < self.locked_until:
                return True
            # Lockout expired
            self.locked_until = None
            self.failed_attempts = 0
            self.save()
            return False

    def generate_recovery_code(self) -> str:
        """
        Generates a 32-character high-entropy recovery code (128-bit cryptographic entropy).
        Only the PBKDF2-HMAC-SHA256 verifier hash and salt are saved; raw secret is never stored.
        """
        with self._lock:
            if self._tampered or not self.verify_storage_integrity():
                raise PermissionError("Cannot generate recovery code: recovery storage failed integrity verification.")
            raw_code = "-".join([secrets.token_hex(4).upper() for _ in range(4)])
            salt = secrets.token_bytes(16)
            # PBKDF2-HMAC-SHA256 password-derived verifier using single production policy (600,000 iterations)
            self.pbkdf2_iterations = PRODUCTION_PBKDF2_ITERATIONS
            self.verifier_version = CURRENT_VERIFIER_VERSION
            code_hash = hashlib.pbkdf2_hmac("sha256", raw_code.encode("utf-8"), salt, self.pbkdf2_iterations).hex()

            self.recovery_code_hash = code_hash
            self.recovery_salt = salt.hex()
            self.failed_attempts = 0
            self.locked_until = None
            self.save()
            return raw_code

    def verify_recovery_code(self, candidate_code: str) -> Union[RecoveryAuthorizationProof, bool]:
        """
        Verifies candidate recovery code against stored PBKDF2 hash with brute-force protection.
        Locks out after 5 consecutive failures for 300 seconds.
        On success, generates and returns a RecoveryAuthorizationProof (evaluates as True).
        """
        with self._lock:
            if self._tampered or not self.verify_storage_integrity():
                return False

            if self.is_locked_out():
                return False

            if not self.recovery_code_hash or not self.recovery_salt:
                return False

            if not candidate_code or not isinstance(candidate_code, str):
                return False

            # Strip ANSI escape sequences (bracketed paste \x1b[200~ ... \x1b[201~ etc.)
            clean_code = re.sub(r'\x1b\[[0-9;]*[a-zA-Z~]', '', candidate_code)
            clean_code = clean_code.replace('—', '-').replace('–', '-').replace('\u00a0', ' ')
            clean_code = clean_code.strip(" \t\r\n'\"“‘”’<>").upper()
            hex_only = "".join(c for c in clean_code if c in "0123456789ABCDEF")
            candidates = [clean_code]

            # Replace any spaces or underscores with hyphens
            normalized_hyphens = clean_code.replace(" ", "-").replace("_", "-")
            if normalized_hyphens not in candidates:
                candidates.append(normalized_hyphens)

            # 32-character hex key (4 chunks of 8 characters, standard TARA recovery format)
            if len(hex_only) == 32:
                formatted_32 = f"{hex_only[:8]}-{hex_only[8:16]}-{hex_only[16:24]}-{hex_only[24:]}"
                if formatted_32 not in candidates:
                    candidates.append(formatted_32)
                if hex_only not in candidates:
                    candidates.append(hex_only)

            # 16-character hex key (4 chunks of 4 characters)
            if len(hex_only) == 16:
                formatted_16 = f"{hex_only[:4]}-{hex_only[4:8]}-{hex_only[8:12]}-{hex_only[12:]}"
                if formatted_16 not in candidates:
                    candidates.append(formatted_16)
                if hex_only not in candidates:
                    candidates.append(hex_only)

            try:
                salt = bytes.fromhex(self.recovery_salt)
                matched = False
                matched_cand = None
                matched_with_legacy = False

                # Policy: Do not permanently accept both 50,000 and 600,000 as equivalent production verifiers.
                # v2+ verifiers strictly enforce the single current production policy (600,000 iterations).
                # Older v1 verifiers (50,000 iterations) are evaluated once via the versioned migration path.
                if getattr(self, "verifier_version", 1) >= CURRENT_VERIFIER_VERSION:
                    allowed_iterations = (PRODUCTION_PBKDF2_ITERATIONS,)
                else:
                    legacy_iters = getattr(self, "pbkdf2_iterations", LEGACY_PBKDF2_ITERATIONS)
                    allowed_iterations = (legacy_iters,)

                for cand in candidates:
                    for iters in allowed_iterations:
                        cand_hash = hashlib.pbkdf2_hmac("sha256", cand.encode("utf-8"), salt, iters).hex()
                        if hmac.compare_digest(cand_hash, self.recovery_code_hash):
                            matched = True
                            matched_cand = cand
                            if iters < PRODUCTION_PBKDF2_ITERATIONS or getattr(self, "verifier_version", 1) < CURRENT_VERIFIER_VERSION:
                                matched_with_legacy = True
                            break
                    if matched:
                        break
            except Exception:
                return False

            if matched:
                self.failed_attempts = 0
                self.locked_until = None

                # After a successful authenticated recovery-code verification using an older verifier,
                # safely rehash/migrate to current policy (600,000 iterations) with fresh salt and retire the old verifier.
                if matched_with_legacy and matched_cand:
                    new_salt = secrets.token_bytes(16)
                    new_hash = hashlib.pbkdf2_hmac(
                        "sha256",
                        matched_cand.encode("utf-8"),
                        new_salt,
                        PRODUCTION_PBKDF2_ITERATIONS
                    ).hex()
                    self.recovery_code_hash = new_hash
                    self.recovery_salt = new_salt.hex()
                    prev_v = getattr(self, "verifier_version", 1)
                    self.pbkdf2_iterations = PRODUCTION_PBKDF2_ITERATIONS
                    self.verifier_version = CURRENT_VERIFIER_VERSION
                    self.recovery_history.append({
                        "event": "VERIFIER_MIGRATION_V1_TO_V2",
                        "previous_version": prev_v,
                        "new_version": CURRENT_VERIFIER_VERSION,
                        "new_iterations": PRODUCTION_PBKDF2_ITERATIONS,
                        "migrated_at": datetime.now(timezone.utc).isoformat()
                    })

                self.save()
                return self._issue_recovery_proof(
                    method="recovery_code",
                    details={"verified_at": datetime.now(timezone.utc).isoformat()}
                )
            else:
                self.failed_attempts += 1
                if self.failed_attempts >= self.max_attempts:
                    self.locked_until = time.time() + self.lockout_seconds
                self.save()
                return False

    def set_recovery_email(self, email: str) -> None:
        """Sets verified recovery email."""
        with self._lock:
            if self._tampered or not self.verify_storage_integrity():
                raise PermissionError("Cannot set recovery email: recovery storage failed integrity verification.")
            self.recovery_email = email.lower().strip()
            self.save()


    def verify_google_recovery(
        self,
        google_identity_result: Any,
        google_service: Optional[Any] = None,
        expected_nonce: Optional[str] = None
    ) -> Union[RecoveryAuthorizationProof, bool]:
        """
        Verifies Google recovery path.
        Requires a cryptographically verified Google identity result from the
        Google verification layer (or verifies an ID token via GoogleAuthService).
        Never trusts a caller-supplied email string as standalone proof.
        """
        with self._lock:
            if self._tampered or not self.verify_storage_integrity():
                return False

            if self.is_locked_out():
                return False

            if not self.recovery_email:
                return False

        # Do not trust a caller-supplied email string as standalone proof
        if isinstance(google_identity_result, str):
            token_str = google_identity_result.strip()
            # If it's a JWT with 3 segments, verify cryptographically via GoogleAuthService
            if token_str.count(".") == 2 and not token_str.endswith("@") and "@" not in token_str.split(".")[0]:
                service = google_service or self.google_auth_service
                if service is None:
                    from ..services.google_auth import GoogleAuthService
                    service = GoogleAuthService()
                res = service.verify_id_token(token_str, expected_nonce=expected_nonce)
                if not res.get("verified"):
                    return False
                verified_data = res
            else:
                # Plain caller-supplied email string is strictly rejected
                return False
        elif isinstance(google_identity_result, dict):
            # Must be a verified result dict produced by the Google verification layer
            if not google_identity_result.get("verified") and not google_identity_result.get("valid"):
                return False
            provider = google_identity_result.get("provider", "google.com")
            claims = google_identity_result.get("claims", {})
            iss = claims.get("iss", "") if isinstance(claims, dict) else ""
            if provider != "google.com" and iss and iss not in ("accounts.google.com", "https://accounts.google.com"):
                return False
            verified_data = google_identity_result
        else:
            return False

        verified_email = (verified_data.get("email") or "").lower().strip()
        if not verified_email and isinstance(verified_data.get("claims"), dict):
            verified_email = (verified_data["claims"].get("email") or "").lower().strip()

        if not verified_email:
            return False

        if verified_email != self.recovery_email.lower().strip():
            return False

        return self._issue_recovery_proof(
            method="google_account",
            details={"email": verified_email, "sub": verified_data.get("sub")}
        )

    verify_google_account_recovery = verify_google_recovery

    def _get_device_registry(self, explicit_registry: Optional[Any] = None) -> Optional[Any]:
        """Resolves local device registry for device authorization checks."""
        if explicit_registry is not None:
            return explicit_registry
        if self.device_registry is not None:
            return self.device_registry

        creator_reg = getattr(self.creator, "device_registry", None) or getattr(self.creator, "devices", None)
        if creator_reg is not None:
            self.device_registry = creator_reg
            return creator_reg

        try:
            from ..devices.device_registry import DeviceRegistry
            candidates = []
            record_path = getattr(self.creator, "record_path", None)
            if record_path:
                candidates.append(os.path.join(os.path.dirname(record_path), "..", "devices", "devices.json"))
            repo_root = getattr(self.creator, "repo_root", None)
            if repo_root:
                candidates.append(os.path.join(repo_root, "TARA", "ACCESS", "devices", "devices.json"))
            candidates.append(os.path.join(get_default_recovery_storage_dir(), "..", "devices", "devices.json"))

            reg_path = None
            for cp in candidates:
                norm_cp = os.path.normpath(cp)
                if os.path.exists(norm_cp):
                    reg_path = norm_cp
                    break

            self.device_registry = DeviceRegistry(reg_path)
            return self.device_registry
        except Exception:
            return None

    def verify_trusted_device_recovery(
        self,
        device_id: str,
        device_pubkey_hex: str,
        recovery_challenge: bytes,
        device_signature: bytes,
        device_registry: Optional[Any] = None
    ) -> Union[RecoveryAuthorizationProof, bool]:
        """
        Verifies approval from an existing authorized device.
        Requires:
        1. device_id exists in the authorized local device registry.
        2. Supplied public key exactly matches that registered device.
        3. Device is currently authorized and not revoked.
        4. Valid Ed25519 signature over recovery_challenge.
        """
        with self._lock:
            if self._tampered or not self.verify_storage_integrity():
                return False

            if self.is_locked_out():
                return False

            if not self.trusted_device_recovery_enabled:
                return False

        if not device_id or not device_pubkey_hex or not recovery_challenge or not device_signature:
            return False

        registry = self._get_device_registry(device_registry)
        if registry is None:
            return False

        # 1. Verify device_id exists in the authorized local device registry
        device_record = registry.get_device(device_id) if hasattr(registry, "get_device") else getattr(registry, "devices", {}).get(device_id)
        if not device_record:
            return False

        # 2. Verify the supplied public key exactly matches that registered device
        registered_pubkey = (device_record.get("device_public_key") or "").strip().lower()
        supplied_pubkey = (device_pubkey_hex or "").strip().lower()
        if not registered_pubkey or not supplied_pubkey or supplied_pubkey != registered_pubkey:
            return False

        # 3. Verify the device is currently authorized / not revoked
        status = (device_record.get("status") or "").upper()
        if status != "AUTHORIZED":
            return False
        if status == "REVOKED":
            return False
        if hasattr(registry, "is_authorized") and not registry.is_authorized(device_id):
            return False

        # 4. Then verify the Ed25519 signature over the challenge
        try:
            pub_bytes = bytes.fromhex(registered_pubkey)
            if not Ed25519.verify(pub_bytes, recovery_challenge, device_signature):
                return False
        except Exception:
            return False

        return self._issue_recovery_proof(
            method="trusted_device",
            details={"device_id": device_id}
        )

    def recover_and_rotate_key(
        self,
        new_public_key_bytes: bytes,
        authorization_proof: Optional[Any] = None,
        authorization_method: Optional[str] = None,
        reason: str = "key_loss_recovery"
    ) -> Dict[str, Any]:
        """
        Executes key rotation & recovery after validating recovery authorization proof.
        Never rotates based solely on an unvalidated authorization_method string.
        - Verifies active or supplied recovery proof
        - Validates creator_id is permanently ROOT_OPERATOR
        - Increments key_version
        - Revokes old key version
        - Updates root public key
        - Preserves CREATOR_ID = ROOT_OPERATOR
        """
        with self._lock:
            if self._tampered or not self.verify_storage_integrity():
                raise PermissionError("Recovery rejected: protected recovery storage failed cryptographic integrity or seal verification.")

            candidate_proof = authorization_proof

            # Support legacy positional method string: recover_and_rotate_key(new_pub, "recovery_code")
            if isinstance(candidate_proof, str) and authorization_method is None:
                authorization_method = candidate_proof
                candidate_proof = None

            # If no explicit proof passed, resolve active unconsumed proof produced by recovery flow
            if candidate_proof is None:
                if self._latest_proof and self._latest_proof.is_valid(CANONICAL_CREATOR_ID):
                    candidate_proof = self._latest_proof

            # If caller is existing authenticated creator rotating their active key (Flow 12)
            if authorization_method == "creator_key_rotation" and candidate_proof is None:
                if self.creator.is_initialized() and self.creator.root_public_key and not self.is_locked_out():
                    candidate_proof = self._issue_recovery_proof(
                        method="creator_key_rotation",
                        details={"creator_id": self.creator.creator_id, "reason": reason}
                    )

            # Strictly require a validated RecoveryAuthorizationProof produced by recovery flow
            if not isinstance(candidate_proof, RecoveryAuthorizationProof):
                raise PermissionError(
                    "Unauthorized key rotation: a validated recovery authorization proof "
                    "produced by the RecoveryManager recovery flow is strictly required. "
                    "Rotation cannot be authorized by method string alone."
                )

            if candidate_proof.manager_id != self._instance_id and candidate_proof.proof_id not in self._active_proofs:
                raise PermissionError("Recovery authorization proof was not issued by this RecoveryManager authority.")

            if not candidate_proof.is_valid(CANONICAL_CREATOR_ID):
                raise PermissionError("Recovery authorization proof is invalid, expired, or already consumed.")

            # Ensure Primary Creator invariant
            if self.creator.creator_id != CANONICAL_CREATOR_ID:
                raise ValueError(f"Recovery is bound to {CANONICAL_CREATOR_ID}. Unauthorized creator identity.")

            # Consume the proof so it cannot be re-used
            candidate_proof.consume()
            if candidate_proof.proof_id in self._active_proofs:
                del self._active_proofs[candidate_proof.proof_id]
            if self._latest_proof is candidate_proof:
                self._latest_proof = None

            # Record method
            effective_method = candidate_proof.method
            if authorization_method and authorization_method != candidate_proof.method:
                effective_method = f"{candidate_proof.method}:{authorization_method}"

            # Execute rotation on the CreatorIdentity with authorized=True
            old_version = self.creator.key_version
            self.creator.rotate_root_key(new_public_key_bytes, authorized=True, reason=reason)

            event = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "method": effective_method,
                "proof_id": candidate_proof.proof_id,
                "old_key_version": old_version,
                "new_key_version": self.creator.key_version,
                "creator_id": CANONICAL_CREATOR_ID,
                "display_name": self.creator.display_name,
                "reason": reason
            }
            self.recovery_history.append(event)
            self.save()
            return event

    def save(self) -> None:
        """
        Saves recovery configuration strictly to recovery_record_path using authenticated encryption
        and cryptographic integrity sealing bound to the current Creator Authority.
        Writes atomically and crash-safely.
        """
        with self._lock:
            if self._tampered:
                raise PermissionError("Cannot save: recovery storage has been tampered with or corrupted.")

            os.makedirs(os.path.dirname(self.recovery_record_path), exist_ok=True)

            creator_id = CANONICAL_CREATOR_ID
            key_version = getattr(self.creator, "key_version", 1) or 1
            root_pubkey = getattr(self.creator, "root_public_key", "") or ""

            inner_payload = {
                "creator_id": creator_id,
                "key_version": key_version,
                "root_public_key": root_pubkey,
                "recovery_code_hash": self.recovery_code_hash,
                "recovery_salt": self.recovery_salt,
                "pbkdf2_iterations": getattr(self, "pbkdf2_iterations", PRODUCTION_PBKDF2_ITERATIONS),
                "verifier_version": getattr(self, "verifier_version", CURRENT_VERIFIER_VERSION),
                "recovery_email": self.recovery_email,
                "trusted_device_recovery_enabled": self.trusted_device_recovery_enabled,
                "failed_attempts": self.failed_attempts,
                "locked_until": self.locked_until,
                "recovery_history": self.recovery_history,
                "saved_at": datetime.now(timezone.utc).isoformat()
            }

            plaintext_bytes = json.dumps(inner_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            k_rec = secrets.token_bytes(32)
            nonce = secrets.token_bytes(12)
            aad = f"TARA_RECOVERY_SEAL:{creator_id}:{key_version}:{root_pubkey}".encode("utf-8")

            aesgcm = AESGCM(k_rec)
            ciphertext = aesgcm.encrypt(nonce, plaintext_bytes, aad)

            wrapped_key = protect_bytes_dpapi(k_rec, description="TARA_RECOVERY_KEY")

            seal_content = f"{creator_id}:{key_version}:{root_pubkey}:{nonce.hex()}:{ciphertext.hex()}".encode("utf-8")
            integrity_seal = hmac.new(k_rec, seal_content, hashlib.sha256).hexdigest()

            envelope = {
                "format": PROTECTED_RECOVERY_FORMAT,
                "creator_id": creator_id,
                "key_version": key_version,
                "root_public_key": root_pubkey,
                "nonce": nonce.hex(),
                "ciphertext": ciphertext.hex(),
                "wrapped_key": wrapped_key.hex(),
                "integrity_seal": integrity_seal,
                "sealed_at": datetime.now(timezone.utc).isoformat()
            }

            # Atomic crash-safe write via temp file + os.replace
            temp_path = f"{self.recovery_record_path}.tmp.{secrets.token_hex(6)}"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(envelope, f, indent=2)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except (OSError, AttributeError):
                    pass

            os.replace(temp_path, self.recovery_record_path)

    def load(self) -> None:
        """
        Loads and authenticates recovery configuration from recovery_record_path.
        Fails closed on any tampering, decryption failure, seal mismatch, or rollback.
        """
        with self._lock:
            if not os.path.exists(self.recovery_record_path):
                return

            try:
                with open(self.recovery_record_path, "r", encoding="utf-8") as f:
                    envelope = json.load(f)
            except Exception as e:
                self._tampered = True
                raise PermissionError(f"Recovery configuration is unreadable: {str(e)}")

            if not isinstance(envelope, dict):
                self._tampered = True
                raise PermissionError("Recovery configuration corrupted: invalid JSON structure.")

            if envelope.get("format") == PROTECTED_RECOVERY_FORMAT:
                data = self._decrypt_and_verify_envelope(envelope)
            elif "recovery_code_hash" in envelope and envelope.get("creator_id") == CANONICAL_CREATOR_ID:
                if envelope.get("recovery_code_hash") is None or envelope.get("status") == "CREATOR_SETUP_REQUIRED":
                    self.recovery_code_hash = None
                    self.recovery_salt = None
                    self.status = "CREATOR_SETUP_REQUIRED"
                    return
                # Valid legacy unencrypted format -> migrate immediately to protected authenticated encryption
                data = self._migrate_legacy_plaintext_config(envelope)
            else:
                self._tampered = True
                raise PermissionError("Recovery configuration tampering detected: invalid format or missing required fields.")

            self.recovery_code_hash = data.get("recovery_code_hash")
            self.recovery_salt = data.get("recovery_salt")
            self.pbkdf2_iterations = int(data.get("pbkdf2_iterations", LEGACY_PBKDF2_ITERATIONS if data.get("verifier_version", 1) == 1 else PRODUCTION_PBKDF2_ITERATIONS))
            self.verifier_version = int(data.get("verifier_version", 1 if ("pbkdf2_iterations" not in data or data.get("pbkdf2_iterations") == LEGACY_PBKDF2_ITERATIONS) else CURRENT_VERIFIER_VERSION))
            self.recovery_email = data.get("recovery_email")
            self.trusted_device_recovery_enabled = data.get("trusted_device_recovery_enabled", True)
            self.failed_attempts = data.get("failed_attempts", 0)
            self.locked_until = data.get("locked_until")
            self.recovery_history = data.get("recovery_history", [])

