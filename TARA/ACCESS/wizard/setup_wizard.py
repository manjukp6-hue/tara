"""
TARA/ACCESS/wizard/setup_wizard.py

Interactive Creator Setup Wizard and Cryptographic Session Authenticator.
Guarantees:
- Enforces permanent Creator Identity (CREATOR_ID = ROOT_OPERATOR, DISPLAY_NAME = OPERATOR_ROOT, RECOVERY_EMAIL = operator@internal.local).
- Never stores master passphrase or unhashed emergency secret on disk.
- Generates 32-byte Ed25519 root keypair protected via Scrypt (N=131072) + AES-256-GCM + Windows DPAPI.
- Issues signed auth_manifest.json manifest with cryptographic proof-of-possession.
- Rate-limited lockout: 5 consecutive failed attempts locks session for 300 seconds.
- Session manager provides secure unlock, sign_action, lock, and emergency recovery flows.
"""

import os
import sys
import json
import time
import secrets
import getpass
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple

from ..operator.operator_profile import (
    CreatorIdentity,
    CANONICAL_CREATOR_ID,
    DEFAULT_DISPLAY_NAME
)
from ..operator.operator_lifecycle import AuthorityLifecycleManager, AuthorityState
from ..audit.security_logger import SecurityAuditLogger
from ..operator.multi_operator_registry import MultiCreatorRegistry
from ..crypto.ed25519 import Ed25519
from ..crypto.secure_storage import SecureKeyStorage
from ..crypto.dpapi_storage import IS_WINDOWS
from ..restore.restore_manager import RecoveryManager, get_default_recovery_record_path
from ..activation.activation_manager import PrivateTriggerManager

CANONICAL_RECOVERY_EMAIL = os.environ.get("TARA_CREATOR_EMAIL", "creator@internal.local")


class CreatorSetupWizard:
    """
    Sets up and configures the TARA Root Creator identity, keys, and signed manifest.
    Supports both headless programmatic API and interactive CLI.
    """

    def __init__(
        self,
        repo_root: Optional[str] = None,
        creator_record_path: Optional[str] = None,
        recovery_config_path: Optional[str] = None,
        storage_dir: Optional[str] = None,
        auth_manifest_path: Optional[str] = None,
        trigger_file_path: Optional[str] = None,
        creators_registry_path: Optional[str] = None,
        seal_path: Optional[str] = None,
        google_auth: Optional[Any] = None
    ):
        if repo_root is None:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        self.repo_root = repo_root
        self.google_auth = google_auth

        if creator_record_path is None:
            creator_record_path = os.path.join(self.repo_root, "TARA", "ACCESS", "operator", "operator_record.json")
        self.creator = CreatorIdentity(record_path=creator_record_path)

        if recovery_config_path is None:
            recovery_config_path = get_default_recovery_record_path()
        self.recovery = RecoveryManager(creator=self.creator, recovery_record_path=recovery_config_path)

        if storage_dir is None:
            storage_dir = os.path.join(self.repo_root, "storage", "vault", "access")
        self.storage = SecureKeyStorage(storage_dir=storage_dir)

        if trigger_file_path is None:
            trigger_file_path = os.path.join(self.repo_root, "TARA", "ACCESS", "activation", "private_trigger.hash")
        self.trigger_mgr = PrivateTriggerManager(trigger_file_path=trigger_file_path)

        if auth_manifest_path is None:
            auth_manifest_path = os.path.join(self.repo_root, "TARA", "ACCESS", "operator", "auth_manifest.json")
        self.auth_manifest_path = auth_manifest_path

        if creators_registry_path is None:
            creators_registry_path = os.path.join(self.repo_root, "TARA", "ACCESS", "operator", "operators_registry.json")
        self.creators_registry_path = creators_registry_path

        if seal_path is None:
            seal_path = os.path.join(self.repo_root, "storage", "vault", "access", "access_seal.json")
        self.seal_path = seal_path

        self.lifecycle = AuthorityLifecycleManager(
            repo_root=self.repo_root,
            seal_path=self.seal_path,
            creator_record_path=creator_record_path,
            creators_registry_path=self.creators_registry_path,
            recovery_config_path=recovery_config_path,
            auth_manifest_path=self.auth_manifest_path
        )
        self.audit_logger = SecurityAuditLogger()

    def is_configured(self) -> bool:
        """Returns True only if creator authority is verified ACTIVE, keys exist, and manifest is valid."""
        if not self.creator.is_initialized():
            return False
        if not self.lifecycle.is_active():
            return False
        has_keystore = self.storage.has_key("creator_root_key")
        has_manifest = os.path.exists(self.auth_manifest_path)
        return has_keystore and has_manifest

    def verify_auth_manifest(self) -> Dict[str, Any]:
        """
        Validates the integrity of auth_manifest.json.
        Verifies Ed25519 signature and matches with operator_record.json.
        """
        if not os.path.exists(self.auth_manifest_path):
            return {"valid": False, "error": "Manifest file auth_manifest.json not found"}

        try:
            with open(self.auth_manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            sig_hex = manifest.get("signature")
            if not sig_hex:
                return {"valid": False, "error": "Manifest missing signature"}

            pub_hex = manifest.get("root_public_key")
            if not pub_hex:
                return {"valid": False, "error": "Manifest missing root_public_key"}

            # Build canonical body for signature verification
            manifest_copy = dict(manifest)
            manifest_copy.pop("signature", None)
            canonical_bytes = json.dumps(manifest_copy, sort_keys=True).encode("utf-8")

            sig_bytes = bytes.fromhex(sig_hex)
            pub_bytes = bytes.fromhex(pub_hex)

            if not Ed25519.verify(pub_bytes, canonical_bytes, sig_bytes):
                return {"valid": False, "error": "Invalid cryptographic signature on auth_manifest.json"}

            if manifest.get("creator_id") != CANONICAL_CREATOR_ID:
                return {"valid": False, "error": f"Invalid creator_id {manifest.get('creator_id')}"}

            return {"valid": True, "manifest": manifest}
        except Exception as e:
            return {"valid": False, "error": str(e)}

    def run_setup(
        self,
        master_passphrase: str,
        confirm_passphrase: str,
        force: bool = False,
        private_trigger_phrase: Optional[str] = None,
        google_id_token: Optional[str] = None,
        confirm_identity: bool = True
    ) -> Dict[str, Any]:
        """
        Executes complete production setup of the Root Creator:
        1. Validates identity confirmation.
        2. Validates Google ID token if provided (must match canonical recovery email).
        3. Validates passphrase strength and match.
        4. Generates 32-byte emergency recovery code (8 segments).
        5. Configures RecoveryManager with PBKDF2 hash of recovery code.
        6. Configures PrivateTriggerManager with salted hash of private trigger phrase.
        7. Generates authentic Ed25519 root keypair.
        8. Encrypts private key with Scrypt + AES-256-GCM + Windows DPAPI.
        9. Updates operator_record.json with new public key.
        10. Signs and writes auth_manifest.json v2 manifest.
        """
        if not confirm_identity:
            raise ValueError("Creator identity confirmation is required to initialize root authority.")

        creator_email = None
        if google_id_token:
            if self.google_auth is not None:
                google_auth = self.google_auth
            else:
                from ..services.google_auth import GoogleAuthService
                google_auth = GoogleAuthService()
            verif = google_auth.verify_id_token(google_id_token)
            if not verif.get("valid") and not verif.get("verified"):
                raise PermissionError(f"Google ID token verification failed: {verif.get('error')}")
            if google_auth.authorized_email and verif.get("email") != google_auth.authorized_email:
                raise PermissionError(f"Google identity '{verif.get('email')}' does not match authorized email '{google_auth.authorized_email}'.")
            creator_email = verif.get("email") or verif.get("claims", {}).get("email")

        if not master_passphrase or len(master_passphrase) < 8:
            raise ValueError("Master passphrase must be at least 8 characters in length.")

        if master_passphrase != confirm_passphrase:
            raise ValueError("Master passphrase and confirmation do not match.")

        # Check permanent initialization lock
        state = self.lifecycle.get_state()
        if state != AuthorityState.CREATOR_SETUP_REQUIRED or self.creator.is_initialized():
            self.audit_logger.log_event(
                event_type="UNAUTHORIZED_SETUP_ATTEMPT",
                severity="CRITICAL",
                details={"reason": "Attempted to re-run setup on already initialized creator authority", "state": state}
            )
            raise PermissionError("Creator authority is already initialized. Authentication or authorized recovery is required for changes.")

        # Begin one-time initial setup transaction
        self.lifecycle.begin_initialization()

        # 1. Configure Private Trigger Phrase
        if private_trigger_phrase:
            self.trigger_mgr.set_trigger_phrase(private_trigger_phrase)
        elif not self.trigger_mgr.has_trigger():
            self.trigger_mgr.set_trigger_phrase(secrets.token_hex(16))

        # 2. Generate high-entropy 32-character recovery code (128-bit entropy)
        recovery_code = self.recovery.generate_recovery_code()
        effective_recovery_email = creator_email or os.environ.get("TARA_CREATOR_EMAIL") or CANONICAL_RECOVERY_EMAIL
        self.recovery.set_recovery_email(effective_recovery_email)

        # 3. Generate authentic Ed25519 keypair
        priv_bytes, pub_bytes = Ed25519.generate_keypair()

        # 4. Initialize root creator identity
        self.creator.initialize_root_creator(pub_bytes, display_name=DEFAULT_DISPLAY_NAME)

        # 5. Store private key using modern Scrypt + DPAPI (version 3 keystore)
        keystore_path = self.storage.store_private_key_modern(
            "creator_root_key",
            priv_bytes,
            passphrase=master_passphrase,
            use_dpapi=True
        )

        # 6. Build canonical manifest
        manifest_body = {
            "version": 2,
            "creator_id": CANONICAL_CREATOR_ID,
            "display_name": DEFAULT_DISPLAY_NAME,
            "recovery_email": effective_recovery_email,
            "root_public_key": pub_bytes.hex(),
            "key_version": self.creator.key_version,
            "status": "ACTIVE",
            "kdf_method": "Scrypt-N131072-r8-p1",
            "hardware_protection": "Windows DPAPI" if IS_WINDOWS else "OS_ENVELOPE",
            "created_at": datetime.now(timezone.utc).isoformat()
        }

        canonical_bytes = json.dumps(manifest_body, sort_keys=True).encode("utf-8")
        sig_bytes = Ed25519.sign(priv_bytes, canonical_bytes)

        manifest_data = dict(manifest_body)
        manifest_data["signature"] = sig_bytes.hex()

        os.makedirs(os.path.dirname(self.auth_manifest_path), exist_ok=True)
        with open(self.auth_manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        # Update multi-creator registry with root public key and authorized email
        try:
            reg = MultiCreatorRegistry(registry_path=self.lifecycle.creators_registry_path)
            reg.set_public_key(CANONICAL_CREATOR_ID, pub_bytes.hex())
            if effective_recovery_email:
                reg.set_authorized_google_email(CANONICAL_CREATOR_ID, effective_recovery_email)
        except Exception:
            pass

        # Cryptographically seal initial authority
        seal_data = self.lifecycle.seal_initial_authority(
            priv_bytes=priv_bytes,
            pub_bytes=pub_bytes,
            master_passphrase=master_passphrase,
            display_name=DEFAULT_DISPLAY_NAME
        )

        self.audit_logger.log_event(
            event_type="FIRST_CREATOR_INITIALIZED",
            severity="INFO",
            details={
                "creator_id": CANONICAL_CREATOR_ID,
                "display_name": DEFAULT_DISPLAY_NAME,
                "key_id": seal_data.get("key_id"),
                "recovery_configured": True
            }
        )

        return {
            "status": "SUCCESS",
            "setup_complete": True,
            "creator_id": CANONICAL_CREATOR_ID,
            "display_name": DEFAULT_DISPLAY_NAME,
            "recovery_email": effective_recovery_email,
            "root_public_key": pub_bytes.hex(),
            "key_version": self.creator.key_version,
            "emergency_recovery_secret": recovery_code,
            "manifest_path": self.auth_manifest_path,
            "keystore_path": keystore_path,
            "seal_path": self.seal_path,
            "hardware_protection": "Windows DPAPI" if IS_WINDOWS else "OS_ENVELOPE",
            "trigger_configured": self.trigger_mgr.has_trigger()
        }

    execute_first_time_setup = run_setup

    def interactive_cli(self) -> None:
        """Runs the Creator Setup Wizard in an interactive terminal."""
        print("=" * 65)
        print("         TARA ROOT CREATOR SETUP WIZARD (PRODUCTION)")
        print("=" * 65)
        print(f" Creator ID    : {CANONICAL_CREATOR_ID}")
        print(f" Display Name  : {DEFAULT_DISPLAY_NAME}")
        print(f" Recovery Email: {CANONICAL_RECOVERY_EMAIL}")
        print("=" * 65)
        print("This wizard configures your master cryptographic root credentials.")
        print("Your private key will be encrypted with Scrypt + AES-256-GCM")
        if IS_WINDOWS:
            print("and hardware-wrapped with Windows DPAPI (tied to this OS user).")
        print()

        while True:
            passphrase = getpass.getpass("Enter Master Creator Passphrase (min 8 chars): ")
            if len(passphrase) < 8:
                print("[!] Passphrase too short. Minimum length is 8 characters.")
                continue
            confirm = getpass.getpass("Confirm Master Creator Passphrase: ")
            if passphrase != confirm:
                print("[!] Passphrases do not match. Please re-enter.")
                continue
            break

        print()
        custom_trigger = input("Enter your Private Creator Trigger Phrase (optional, press Enter for default): ").strip()

        print("\n[*] Generating Ed25519 Root Keypair & Hardened Keystore...")
        result = self.run_setup(
            passphrase,
            confirm,
            private_trigger_phrase=custom_trigger if custom_trigger else None
        )

        print("\n" + "#" * 65)
        print("             CRITICAL: EMERGENCY RECOVERY CODE")
        print("#" * 65)
        print(f"\n   >>>  {result['emergency_recovery_secret']}  <<<\n")
        print(" Store this 32-character code in a secure offline vault.")
        print(" If you lose your master passphrase, this code is the ONLY way")
        print(" to regain Root Authority over TARA.")
        print("#" * 65)
        print(f"\n[+] Setup Completed Successfully!")
        print(f"    - Public Key        : {result['root_public_key']}")
        print(f"    - Key Version       : {result['key_version']}")
        print(f"    - Manifest          : {result['manifest_path']}")
        print(f"    - Protection        : {result['hardware_protection']}")
        print(f"    - Trigger Configured: {result['trigger_configured']}")
        print("=" * 65)


class CreatorSessionManager:
    """
    Manages creator authentication sessions with rate-limiting, lockout protection,
    digital signature operations, and emergency recovery.
    """

    LOCKOUT_THRESHOLD = 5
    LOCKOUT_DURATION_SECONDS = 300
    SESSION_DURATION_SECONDS = 3600

    def __init__(
        self,
        repo_root: Optional[str] = None,
        creator_record_path: Optional[str] = None,
        recovery_config_path: Optional[str] = None,
        storage_dir: Optional[str] = None,
        auth_manifest_path: Optional[str] = None,
        trigger_file_path: Optional[str] = None,
        creators_registry_path: Optional[str] = None,
        seal_path: Optional[str] = None
    ):
        if repo_root is None:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        self.repo_root = repo_root

        self.creator = CreatorIdentity(record_path=creator_record_path)
        self.recovery = RecoveryManager(creator=self.creator, recovery_record_path=recovery_config_path)
        self.storage = SecureKeyStorage(storage_dir=storage_dir)
        self.trigger_mgr = PrivateTriggerManager(trigger_file_path=trigger_file_path)

        if auth_manifest_path is None:
            auth_manifest_path = os.path.join(self.repo_root, "TARA", "ACCESS", "operator", "auth_manifest.json")
        self.auth_manifest_path = auth_manifest_path

        if creators_registry_path is None:
            creators_registry_path = os.path.join(self.repo_root, "TARA", "ACCESS", "operator", "operators_registry.json")
        self.creators_registry_path = creators_registry_path

        if seal_path is None:
            seal_path = os.path.join(self.storage.storage_dir, "access_seal.json")
        self.seal_path = seal_path

        self.lifecycle = AuthorityLifecycleManager(
            repo_root=self.repo_root,
            seal_path=self.seal_path,
            creator_record_path=creator_record_path,
            creators_registry_path=self.creators_registry_path,
            recovery_config_path=recovery_config_path,
            auth_manifest_path=self.auth_manifest_path
        )
        self.audit_logger = SecurityAuditLogger()

        self._active_key: Optional[bytes] = None
        self._session_token: Optional[str] = None
        self._session_expires_at: Optional[float] = None
        self._failed_attempts: int = 0
        self._locked_until: Optional[float] = None

    def verify_trigger(self, candidate_text: str) -> bool:
        """Verifies if candidate text matches the configured private creator trigger phrase."""
        return self.trigger_mgr.verify_trigger(candidate_text)

    def is_locked_out(self) -> Tuple[bool, int]:
        """Returns (is_locked, remaining_seconds)."""
        if self._locked_until is None:
            return False, 0
        now = time.time()
        if now < self._locked_until:
            return True, int(self._locked_until - now)
        # Lockout expired
        self._locked_until = None
        self._failed_attempts = 0
        return False, 0

    def is_unlocked(self) -> bool:
        """Checks whether a valid unlocked creator session is currently active."""
        if self._active_key is None or self._session_expires_at is None:
            return False
        if time.time() >= self._session_expires_at:
            self.lock()
            return False
        return True

    def get_status(self) -> Dict[str, Any]:
        """Returns current authentication state and creator metadata."""
        locked, rem_secs = self.is_locked_out()
        return {
            "creator_id": CANONICAL_CREATOR_ID,
            "display_name": self.creator.display_name,
            "recovery_email": CANONICAL_RECOVERY_EMAIL,
            "key_version": self.creator.key_version,
            "root_public_key": self.creator.root_public_key,
            "is_unlocked": self.is_unlocked(),
            "is_locked_out": locked,
            "lockout_remaining_seconds": rem_secs,
            "failed_attempts": self._failed_attempts,
            "hardware_backed": IS_WINDOWS
        }

    def unlock(self, passphrase: Optional[str] = None) -> Dict[str, Any]:
        """
        Unlocks Root Creator authority using master passphrase or DPAPI OS credentials.
        Enforces 5-attempt brute-force protection with 300s lockout.
        Returns short-lived cryptographically signed session proof on success.
        """
        locked, rem_secs = self.is_locked_out()
        if locked:
            return {
                "status": "LOCKED_OUT",
                "error": f"Too many failed attempts. Try again in {rem_secs} seconds.",
                "remaining_seconds": rem_secs
            }

        priv_bytes = self.storage.load_private_key("creator_root_key", passphrase=passphrase)
        if priv_bytes is None:
            self._failed_attempts += 1
            if self._failed_attempts >= self.LOCKOUT_THRESHOLD:
                self._locked_until = time.time() + self.LOCKOUT_DURATION_SECONDS
                return {
                    "status": "LOCKED_OUT",
                    "error": f"Max attempts reached. Locked for {self.LOCKOUT_DURATION_SECONDS} seconds.",
                    "remaining_seconds": self.LOCKOUT_DURATION_SECONDS
                }
            return {
                "status": "FAILED",
                "error": "Authentication failed. Invalid passphrase.",
                "failed_attempts": self._failed_attempts,
                "remaining_attempts": self.LOCKOUT_THRESHOLD - self._failed_attempts
            }

        # Validate key matches active root public key
        try:
            derived_pub = Ed25519.public_key_from_private(priv_bytes)
            if not self.creator.is_key_active(derived_pub.hex()):
                # Key does not match active record
                return {
                    "status": "FAILED",
                    "error": "Decrypted key does not match active root identity record."
                }
        except Exception as e:
            return {"status": "ERROR", "error": str(e)}

        # Reset failure state
        self._failed_attempts = 0
        self._locked_until = None

        # Generate session token and cryptographic proof-of-possession
        token = secrets.token_hex(24)
        ts = int(time.time())
        proof_payload = f"{CANONICAL_CREATOR_ID}:{token}:{ts}".encode("utf-8")
        proof_signature = Ed25519.sign(priv_bytes, proof_payload).hex()

        self._active_key = priv_bytes
        self._session_token = token
        self._session_expires_at = time.time() + self.SESSION_DURATION_SECONDS

        return {
            "status": "SUCCESS",
            "creator_id": CANONICAL_CREATOR_ID,
            "display_name": self.creator.display_name,
            "session_token": token,
            "proof_signature": proof_signature,
            "timestamp": ts,
            "expires_in": self.SESSION_DURATION_SECONDS,
            "key_version": self.creator.key_version
        }

    def lock(self) -> Dict[str, Any]:
        """Locks the active session and destroys private key from memory."""
        self._active_key = None
        self._session_token = None
        self._session_expires_at = None
        return {"status": "LOCKED"}

    def sign_action(self, action_payload: bytes) -> bytes:
        """
        Signs an administrative or critical action with the unlocked Ed25519 private key.
        Raises PermissionError if session is locked.
        """
        if not self.is_unlocked() or self._active_key is None:
            raise PermissionError("Creator session is locked. Master passphrase authentication required.")
        return Ed25519.sign(self._active_key, action_payload)

    def recover(
        self,
        emergency_secret: str,
        new_passphrase: str,
        confirm_passphrase: str
    ) -> Dict[str, Any]:
        """
        Executes emergency key recovery using the 32-character recovery code:
        - Validates recovery secret against stored PBKDF2 hash.
        - Re-keys the identity: increments key_version, generates new Ed25519 root key.
        - Encrypts new key with new master passphrase.
        - Issues new recovery secret for future protection.
        - Updates signed auth_manifest.json manifest.
        """
        locked, rem_secs = self.is_locked_out()
        if locked:
            return {
                "status": "LOCKED_OUT",
                "error": f"Too many failed attempts. Try again in {rem_secs} seconds.",
                "remaining_seconds": rem_secs
            }

        if not self.recovery.verify_recovery_code(emergency_secret):
            self._failed_attempts += 1
            if self._failed_attempts >= self.LOCKOUT_THRESHOLD:
                self._locked_until = time.time() + self.LOCKOUT_DURATION_SECONDS
                return {
                    "status": "LOCKED_OUT",
                    "error": f"Max attempts reached. Locked for {self.LOCKOUT_DURATION_SECONDS} seconds.",
                    "remaining_seconds": self.LOCKOUT_DURATION_SECONDS
                }
            return {
                "status": "FAILED",
                "error": "Invalid emergency recovery code.",
                "failed_attempts": self._failed_attempts
            }

        if not new_passphrase or len(new_passphrase) < 8:
            return {"status": "FAILED", "error": "New passphrase must be at least 8 characters."}

        if new_passphrase != confirm_passphrase:
            return {"status": "FAILED", "error": "New passphrase and confirmation do not match."}

        # 1. Generate new Ed25519 keypair
        new_priv, new_pub = Ed25519.generate_keypair()

        # 2. Execute rotation & revoke old key
        self.recovery.recover_and_rotate_key(
            new_pub,
            authorization_method="emergency_recovery_secret",
            reason="emergency_creator_recovery"
        )

        # 3. Store new private key
        keystore_path = self.storage.store_private_key_modern(
            "creator_root_key",
            new_priv,
            passphrase=new_passphrase,
            use_dpapi=True
        )

        # 4. Generate fresh emergency recovery code
        new_recovery_code = self.recovery.generate_recovery_code()

        # 5. Build and sign updated auth_manifest.json
        manifest_body = {
            "version": 2,
            "creator_id": CANONICAL_CREATOR_ID,
            "display_name": DEFAULT_DISPLAY_NAME,
            "recovery_email": CANONICAL_RECOVERY_EMAIL,
            "root_public_key": new_pub.hex(),
            "key_version": self.creator.key_version,
            "status": "ACTIVE",
            "kdf_method": "Scrypt-N131072-r8-p1",
            "hardware_protection": "Windows DPAPI" if IS_WINDOWS else "OS_ENVELOPE",
            "created_at": datetime.now(timezone.utc).isoformat()
        }

        canonical_bytes = json.dumps(manifest_body, sort_keys=True).encode("utf-8")
        sig_bytes = Ed25519.sign(new_priv, canonical_bytes)
        manifest_data = dict(manifest_body)
        manifest_data["signature"] = sig_bytes.hex()

        with open(self.auth_manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        # Update multi-creator registry with new root public key
        try:
            reg = MultiCreatorRegistry(registry_path=self.lifecycle.creators_registry_path)
            reg.set_public_key(CANONICAL_CREATOR_ID, new_pub.hex())
        except Exception:
            pass

        # Re-seal authority seal
        try:
            self.lifecycle.reseal_authority(
                new_priv_bytes=new_priv,
                new_pub_bytes=new_pub,
                key_version=self.creator.key_version,
                display_name=DEFAULT_DISPLAY_NAME
            )
        except Exception as e:
            self.audit_logger.log_event(
                event_type="RESEAL_ERROR_DURING_RECOVERY",
                severity="CRITICAL",
                details={"error": str(e)}
            )

        # 6. Reset failure counters and establish active session
        self._failed_attempts = 0
        self._locked_until = None
        token = secrets.token_hex(24)
        ts = int(time.time())
        proof_payload = f"{CANONICAL_CREATOR_ID}:{token}:{ts}".encode("utf-8")
        proof_sig = Ed25519.sign(new_priv, proof_payload).hex()

        self._active_key = new_priv
        self._session_token = token
        self._session_expires_at = time.time() + self.SESSION_DURATION_SECONDS

        return {
            "status": "SUCCESS",
            "message": "Identity successfully recovered and re-keyed.",
            "creator_id": CANONICAL_CREATOR_ID,
            "new_key_version": self.creator.key_version,
            "new_root_public_key": new_pub.hex(),
            "new_emergency_recovery_secret": new_recovery_code,
            "session_token": token,
            "proof_signature": proof_sig
        }


if __name__ == "__main__":
    wizard = CreatorSetupWizard()
    wizard.interactive_cli()
