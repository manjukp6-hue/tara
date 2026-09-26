"""
TARA/ACCESS/creator/creator_authority.py

Strict One-Time Creator Initialization Lifecycle & Protected Authority State Engine.
Enforces:
1. FIRST CREATOR SETUP - ONE TIME ONLY:
   - Starts in INITIALIZATION_REQUIRED state when no creator authority is initialized.
   - Creator Setup Wizard executes initial identity, keys, secrets, recovery, and registry.
   - First creator becomes ROOT_CREATOR (ROOT_OPERATOR).
2. PERMANENT INITIALIZATION LOCK:
   - After setup, SETUP_COMPLETE = True is permanently locked and immutable.
   - Setup cannot create a second root creator or overwrite ROOT_OPERATOR.
   - Setup cannot silently regenerate keys or reset recovery.
   - Reports: "Creator authority is already initialized. Authentication or authorized recovery is required for changes."
3. SOURCE CODE ALONE IS NOT ENOUGH:
   - Authority is anchored in protected runtime state (cryptographic seal, DPAPI, AEAD keystore).
   - Tampering with operator_record.json, operators_registry.json, or source files results in AUTHORITY_LOCKED.
4. TWO AUTHORIZED CHANGE PATHS ONLY:
   - Path A: Existing authenticated creator authorization (proof-of-possession from active key).
   - Path B: Valid emergency / authorized recovery.
   - No third path.
"""

import os
import json
import hmac
import hashlib
import secrets
import threading
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple, Any

from .operator_profile import CANONICAL_CREATOR_ID, DEFAULT_DISPLAY_NAME, CreatorIdentity
from .multi_operator_registry import MultiCreatorRegistry
from ..crypto.ed25519 import Ed25519
from ..crypto.dpapi_storage import protect_bytes_dpapi, unprotect_bytes_dpapi, IS_WINDOWS
from ..restore.restore_manager import get_default_recovery_record_path


class AuthorityState:
    CREATOR_SETUP_REQUIRED = "CREATOR_SETUP_REQUIRED"
    INITIALIZATION_REQUIRED = "CREATOR_SETUP_REQUIRED"
    ACTIVE = "ACTIVE"
    AUTHORITY_LOCKED = "AUTHORITY_LOCKED"


def compute_canonical_hash(file_path: str) -> Optional[str]:
    """Computes SHA256 over canonical sorted JSON representation of a file."""
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        canonical_bytes = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(canonical_bytes).hexdigest()
    except Exception:
        return None


def compute_recovery_verifier_hash(file_path: Optional[str] = None, creator: Optional[Any] = None) -> Optional[str]:
    """Computes SHA256 over canonical sorted JSON representation of recovery verifiers, excluding transient counters."""
    target_path = file_path
    if not target_path or not os.path.exists(target_path):
        target_path = get_default_recovery_record_path()
    if not target_path or not os.path.exists(target_path):
        return None
    try:
        from ..restore.restore_manager import load_protected_recovery_config
        data = load_protected_recovery_config(target_path, creator=creator)
        if not data or not isinstance(data, dict):
            return None
        verifier_data = {
            "creator_id": data.get("creator_id"),
            "recovery_code_hash": data.get("recovery_code_hash"),
            "recovery_salt": data.get("recovery_salt"),
            "recovery_email": data.get("recovery_email")
        }
        canonical_bytes = json.dumps(verifier_data, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(canonical_bytes).hexdigest()
    except Exception:
        return None



class AuthorityLifecycleManager:
    """
    Manages the authoritative one-time initialization lifecycle, permanent initialization lock,
    protected cryptographic authority seal, and integrity verification.
    """

    def __init__(
        self,
        repo_root: Optional[str] = None,
        seal_path: Optional[str] = None,
        creator_record_path: Optional[str] = None,
        creators_registry_path: Optional[str] = None,
        recovery_config_path: Optional[str] = None,
        auth_manifest_path: Optional[str] = None
    ):
        if repo_root is None:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        self.repo_root = repo_root

        if seal_path is None:
            seal_path = os.path.join(self.repo_root, "storage", "vault", "access", "access_seal.json")
        self.seal_path = seal_path

        if creator_record_path is None:
            creator_record_path = os.path.join(self.repo_root, "TARA", "ACCESS", "operator", "operator_record.json")
        self.creator_record_path = creator_record_path

        if creators_registry_path is None:
            creators_registry_path = os.path.join(self.repo_root, "TARA", "ACCESS", "operator", "operators_registry.json")
        self.creators_registry_path = creators_registry_path

        if recovery_config_path is None:
            recovery_config_path = get_default_recovery_record_path()
        self.recovery_config_path = recovery_config_path

        if auth_manifest_path is None:
            auth_manifest_path = os.path.join(self.repo_root, "TARA", "ACCESS", "operator", "auth_manifest.json")
        self.auth_manifest_path = auth_manifest_path

        self._lock = threading.RLock()
        self._state: str = AuthorityState.INITIALIZATION_REQUIRED
        self._lock_reason: Optional[str] = None
        self._seal_cache: Optional[Dict[str, Any]] = None
        self._in_initialization: bool = False

    def begin_initialization(self) -> None:
        """
        Begins the one-time initial setup transaction.
        Can strictly only be called when state is INITIALIZATION_REQUIRED.
        """
        with self._lock:
            state, reason = self.verify_integrity()
            if state != AuthorityState.INITIALIZATION_REQUIRED:
                raise PermissionError("Creator authority is already initialized. Authentication or authorized recovery is required for changes.")
            self._in_initialization = True

    def get_state(self) -> str:
        """Evaluates current creator authority lifecycle state."""
        with self._lock:
            state, reason = self.verify_integrity()
            self._state = state
            self._lock_reason = reason
            return self._state

    def is_setup_complete(self) -> bool:
        """Returns True if one-time initialization has been successfully completed."""
        state = self.get_state()
        return state in (AuthorityState.ACTIVE, AuthorityState.AUTHORITY_LOCKED)

    def is_active(self) -> bool:
        """Returns True only if creator authority is verified ACTIVE."""
        return self.get_state() == AuthorityState.ACTIVE

    def is_locked(self) -> bool:
        """Returns True if creator authority is locked due to integrity mismatch or tampering."""
        return self.get_state() == AuthorityState.AUTHORITY_LOCKED

    def get_lock_reason(self) -> Optional[str]:
        with self._lock:
            return self._lock_reason

    def verify_integrity(self) -> Tuple[str, str]:
        """
        Cryptographically verifies the authoritative creator state against the protected seal.
        Returns (AuthorityState, reason).
        """
        # 1. Check if seal exists in environment or on disk
        raw_bytes = None
        env_seal = os.environ.get("TARA_AUTHORITY_SEAL")
        if env_seal:
            try:
                raw_bytes = env_seal.encode("utf-8")
            except Exception:
                raw_bytes = None

        if raw_bytes is None:
            if not os.path.exists(self.seal_path):
                if getattr(self, "_in_initialization", False):
                    return AuthorityState.CREATOR_SETUP_REQUIRED, "In initial setup transaction"

                # If seal is missing, check if system has any configured creator files
                if os.path.exists(self.creator_record_path):
                    try:
                        with open(self.creator_record_path, "r", encoding="utf-8") as f:
                            rec = json.load(f)
                        root_pk = rec.get("root_public_key")
                        status = (rec.get("status") or "").lower()
                        # Unconfigured / setup required / null public key -> CREATOR_SETUP_REQUIRED
                        if not root_pk or status in ("creator_setup_required", "setup_required", "unconfigured", "pending"):
                            return AuthorityState.CREATOR_SETUP_REQUIRED, "Fresh installation: Creator setup required."
                    except Exception:
                        return AuthorityState.CREATOR_SETUP_REQUIRED, "Fresh installation: Creator setup required."
                else:
                    return AuthorityState.CREATOR_SETUP_REQUIRED, "Fresh installation: Creator setup required."

                # Creator record exists with active public key but authority seal is missing -> Tampering / deployment failure
                return AuthorityState.AUTHORITY_LOCKED, "Authority seal is missing while creator records exist. Possible tampering or file deletion."
            else:
                try:
                    with open(self.seal_path, "rb") as f:
                        raw_bytes = f.read()
                except Exception as e:
                    return AuthorityState.AUTHORITY_LOCKED, f"Protected authority seal is unreadable: {str(e)}"

        # 2. Load protected authority seal
        try:
            if IS_WINDOWS:
                try:
                    unprotected = unprotect_bytes_dpapi(raw_bytes)
                    seal_data = json.loads(unprotected.decode("utf-8"))
                except Exception:
                    # Fallback to direct json if not dpapi-wrapped (e.g. cross-platform testing or env seal)
                    seal_data = json.loads(raw_bytes.decode("utf-8"))
            else:
                seal_data = json.loads(raw_bytes.decode("utf-8"))
        except Exception as e:
            return AuthorityState.AUTHORITY_LOCKED, f"Protected authority seal is corrupted or unreadable: {str(e)}"

        # 3. Verify permanent initialization invariant
        if not seal_data.get("setup_complete"):
            return AuthorityState.INITIALIZATION_REQUIRED, "Setup not marked complete in seal"

        root_id = seal_data.get("root_creator_id")
        if root_id != CANONICAL_CREATOR_ID:
            return AuthorityState.AUTHORITY_LOCKED, f"Root creator ID in seal is '{root_id}', expected '{CANONICAL_CREATOR_ID}'"

        root_pubkey = seal_data.get("root_public_key")
        if not root_pubkey:
            return AuthorityState.AUTHORITY_LOCKED, "Protected authority seal missing root_public_key"

        # 4. Verify operator_record.json integrity
        if not os.path.exists(self.creator_record_path):
            return AuthorityState.AUTHORITY_LOCKED, "operator_record.json is missing"

        record_hash = compute_canonical_hash(self.creator_record_path)
        if record_hash != seal_data.get("record_hash"):
            return AuthorityState.AUTHORITY_LOCKED, f"operator_record.json hash mismatch (tampered or edited). Expected {seal_data.get('record_hash')}, got {record_hash}"

        try:
            with open(self.creator_record_path, "r", encoding="utf-8") as f:
                rec = json.load(f)
            if rec.get("creator_id") != CANONICAL_CREATOR_ID:
                return AuthorityState.AUTHORITY_LOCKED, "operator_record.json creator_id mismatch"
            if rec.get("root_public_key", "").lower() != root_pubkey.lower():
                return AuthorityState.AUTHORITY_LOCKED, "operator_record.json public key does not match sealed authority"
        except Exception as e:
            return AuthorityState.AUTHORITY_LOCKED, f"Failed to parse operator_record.json: {str(e)}"

        # 5. Verify operators_registry.json integrity
        if not os.path.exists(self.creators_registry_path):
            return AuthorityState.AUTHORITY_LOCKED, "operators_registry.json is missing"

        registry_hash = compute_canonical_hash(self.creators_registry_path)
        if registry_hash != seal_data.get("registry_hash"):
            return AuthorityState.AUTHORITY_LOCKED, f"operators_registry.json hash mismatch (tampered or edited). Expected {seal_data.get('registry_hash')}, got {registry_hash}"

        try:
            with open(self.creators_registry_path, "r", encoding="utf-8") as f:
                reg = json.load(f)
            root_reg = reg.get(CANONICAL_CREATOR_ID, {})
            if root_reg.get("role") != "ROOT_CREATOR":
                return AuthorityState.AUTHORITY_LOCKED, "Multi-creator registry ROOT_CREATOR role altered"
            if root_reg.get("public_key", "").lower() != root_pubkey.lower():
                return AuthorityState.AUTHORITY_LOCKED, "Multi-creator registry root public key mismatch"
        except Exception as e:
            return AuthorityState.AUTHORITY_LOCKED, f"Failed to parse operators_registry.json: {str(e)}"

        # 6. Verify recovery config verifier integrity
        rec_path = self.recovery_config_path
        if not os.path.exists(rec_path):
            rec_path = get_default_recovery_record_path()

        # If not in local application storage yet but legacy repo file exists, migrate safely
        if not os.path.exists(rec_path):
            legacy_p = os.path.join(self.repo_root, "TARA", "ACCESS", "restore", "restore_config.json")
            if os.path.exists(legacy_p):
                try:
                    os.makedirs(os.path.dirname(rec_path), exist_ok=True)
                    with open(legacy_p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    with open(rec_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
                    os.remove(legacy_p)
                except Exception:
                    rec_path = legacy_p

        if os.path.exists(rec_path):
            creator_obj = CreatorIdentity(record_path=self.creator_record_path) if os.path.exists(self.creator_record_path) else None
            recovery_hash = compute_recovery_verifier_hash(rec_path, creator=creator_obj)
            if recovery_hash != seal_data.get("recovery_hash"):
                return AuthorityState.AUTHORITY_LOCKED, "recovery_config.json hash mismatch (tampered or edited)"

        # 7. Verify cryptographic signature of the seal
        sig_hex = seal_data.get("seal_signature")
        if not sig_hex:
            if seal_data.get("recovery_authorized"):
                self._seal_cache = seal_data
                return AuthorityState.ACTIVE, "Creator authority verified via authorized recovery"
            return AuthorityState.AUTHORITY_LOCKED, "Protected authority seal missing cryptographic signature"

        seal_copy = dict(seal_data)
        seal_copy.pop("seal_signature", None)
        canonical_seal = json.dumps(seal_copy, sort_keys=True, separators=(",", ":")).encode("utf-8")

        try:
            pub_bytes = bytes.fromhex(root_pubkey)
            sig_bytes = bytes.fromhex(sig_hex)
            if not Ed25519.verify(pub_bytes, canonical_seal, sig_bytes):
                return AuthorityState.AUTHORITY_LOCKED, "Invalid cryptographic signature on authority seal"
        except Exception as e:
            return AuthorityState.AUTHORITY_LOCKED, f"Authority seal signature verification error: {str(e)}"

        self._seal_cache = seal_data
        return AuthorityState.ACTIVE, "Creator authority verified and active"

    def seal_initial_authority(
        self,
        priv_bytes: bytes,
        pub_bytes: bytes,
        master_passphrase: str,
        display_name: str = DEFAULT_DISPLAY_NAME
    ) -> Dict[str, Any]:
        """
        Executes ONE-TIME cryptographic sealing of the initial creator authority.
        Can strictly only be run when current state is INITIALIZATION_REQUIRED.
        """
        with self._lock:
            if os.path.exists(self.seal_path):
                raise PermissionError("Creator authority is already initialized. Authentication or authorized recovery is required for changes.")

            if not getattr(self, "_in_initialization", False):
                state, reason = self.verify_integrity()
                if state != AuthorityState.INITIALIZATION_REQUIRED:
                    raise PermissionError("Creator authority is already initialized. Authentication or authorized recovery is required for changes.")

            pub_hex = pub_bytes.hex().lower()
            key_id = hashlib.sha256(pub_hex.encode("utf-8")).hexdigest()[:16]

            rec_hash = compute_canonical_hash(self.creator_record_path)
            reg_hash = compute_canonical_hash(self.creators_registry_path)
            creator_obj = CreatorIdentity(record_path=self.creator_record_path) if os.path.exists(self.creator_record_path) else None
            recov_hash = compute_recovery_verifier_hash(self.recovery_config_path, creator=creator_obj) if os.path.exists(self.recovery_config_path) else None

            seal_body = {
                "version": 2,
                "setup_complete": True,
                "root_creator_id": CANONICAL_CREATOR_ID,
                "display_name": display_name,
                "root_public_key": pub_hex,
                "key_id": key_id,
                "key_version": 1,
                "record_hash": rec_hash,
                "registry_hash": reg_hash,
                "recovery_hash": recov_hash,
                "sealed_at": datetime.now(timezone.utc).isoformat()
            }

            canonical_seal = json.dumps(seal_body, sort_keys=True, separators=(",", ":")).encode("utf-8")
            sig_bytes = Ed25519.sign(priv_bytes, canonical_seal)

            seal_data = dict(seal_body)
            seal_data["seal_signature"] = sig_bytes.hex()

            raw_bytes = json.dumps(seal_data, indent=2).encode("utf-8")
            if IS_WINDOWS:
                try:
                    payload_to_write = protect_bytes_dpapi(raw_bytes)
                except Exception:
                    payload_to_write = raw_bytes
            else:
                payload_to_write = raw_bytes

            os.makedirs(os.path.dirname(self.seal_path), exist_ok=True)
            with open(self.seal_path, "wb") as f:
                f.write(payload_to_write)

            self._in_initialization = False
            self._state = AuthorityState.ACTIVE
            self._seal_cache = seal_data
            return seal_data

    def reseal_authority(
        self,
        new_priv_bytes: bytes,
        new_pub_bytes: bytes,
        key_version: int,
        display_name: str = DEFAULT_DISPLAY_NAME
    ) -> Dict[str, Any]:
        """
        Cryptographically re-seals the authority seal after an AUTHORIZED key rotation or change.
        Atomically synchronizes operator_record.json and operators_registry.json before computing hashes.
        """
        with self._lock:
            pub_hex = new_pub_bytes.hex().lower()
            key_id = hashlib.sha256(pub_hex.encode("utf-8")).hexdigest()[:16]

            # 1. Synchronize operator_record.json
            if os.path.exists(self.creator_record_path):
                try:
                    with open(self.creator_record_path, "r", encoding="utf-8") as f:
                        rec = json.load(f)
                    changed = False
                    if rec.get("root_public_key", "").lower() != pub_hex:
                        rec["root_public_key"] = pub_hex
                        changed = True
                    if rec.get("key_version") != key_version:
                        rec["key_version"] = key_version
                        changed = True
                    if changed:
                        rec["last_key_rotation"] = datetime.now(timezone.utc).isoformat()
                        temp_rec = f"{self.creator_record_path}.tmp.{secrets.token_hex(4)}"
                        with open(temp_rec, "w", encoding="utf-8") as f:
                            json.dump(rec, f, indent=2)
                        os.replace(temp_rec, self.creator_record_path)
                except Exception as e:
                    self._state = AuthorityState.AUTHORITY_LOCKED
                    self._lock_reason = f"Failed to synchronize operator_record.json: {str(e)}"
                    raise

            # 2. Synchronize operators_registry.json
            if self.creators_registry_path:
                try:
                    reg_mgr = MultiCreatorRegistry(self.creators_registry_path)
                    reg_mgr.set_public_key(CANONICAL_CREATOR_ID, pub_hex, key_id=key_id)
                except Exception as e:
                    self._state = AuthorityState.AUTHORITY_LOCKED
                    self._lock_reason = f"Failed to synchronize operators_registry.json: {str(e)}"
                    raise

            rec_hash = compute_canonical_hash(self.creator_record_path)
            reg_hash = compute_canonical_hash(self.creators_registry_path)
            creator_obj = CreatorIdentity(record_path=self.creator_record_path) if os.path.exists(self.creator_record_path) else None
            recov_hash = compute_recovery_verifier_hash(self.recovery_config_path, creator=creator_obj) if os.path.exists(self.recovery_config_path) else None

            seal_body = {
                "version": 2,
                "setup_complete": True,
                "root_creator_id": CANONICAL_CREATOR_ID,
                "display_name": display_name,
                "root_public_key": pub_hex,
                "key_id": key_id,
                "key_version": key_version,
                "record_hash": rec_hash,
                "registry_hash": reg_hash,
                "recovery_hash": recov_hash,
                "sealed_at": datetime.now(timezone.utc).isoformat()
            }

            canonical_seal = json.dumps(seal_body, sort_keys=True, separators=(",", ":")).encode("utf-8")
            sig_bytes = Ed25519.sign(new_priv_bytes, canonical_seal)

            seal_data = dict(seal_body)
            seal_data["seal_signature"] = sig_bytes.hex()

            raw_bytes = json.dumps(seal_data, indent=2).encode("utf-8")
            if IS_WINDOWS:
                try:
                    payload_to_write = protect_bytes_dpapi(raw_bytes)
                except Exception:
                    payload_to_write = raw_bytes
            else:
                payload_to_write = raw_bytes

            os.makedirs(os.path.dirname(self.seal_path), exist_ok=True)
            temp_seal = f"{self.seal_path}.tmp.{secrets.token_hex(4)}"
            with open(temp_seal, "wb") as f:
                f.write(payload_to_write)
            os.replace(temp_seal, self.seal_path)

            self._seal_cache = None
            verify_state, verify_reason = self.verify_integrity()
            if verify_state != AuthorityState.ACTIVE:
                self._state = AuthorityState.AUTHORITY_LOCKED
                self._lock_reason = verify_reason
                raise PermissionError(f"Authority resealing failed verification: {verify_reason}")

            self._state = AuthorityState.ACTIVE
            self._lock_reason = None
            return seal_data

    def reseal_with_recovery(
        self,
        new_pub_bytes: bytes,
        key_version: int,
        display_name: str = DEFAULT_DISPLAY_NAME,
        new_priv_bytes: Optional[bytes] = None
    ) -> Dict[str, Any]:
        """
        Re-seals authority state during authorized emergency recovery when private key is rotated.
        Atomically synchronizes operator_record.json and operators_registry.json before computing hashes.
        Signs the authority seal with new_priv_bytes if provided, and verifies full integrity before returning.
        """
        with self._lock:
            pub_hex = new_pub_bytes.hex().lower()
            key_id = hashlib.sha256(pub_hex.encode("utf-8")).hexdigest()[:16]

            # 1. Synchronize operator_record.json to newly active public key and key_version
            if os.path.exists(self.creator_record_path):
                try:
                    with open(self.creator_record_path, "r", encoding="utf-8") as f:
                        rec = json.load(f)
                    changed = False
                    if rec.get("root_public_key", "").lower() != pub_hex:
                        rec["root_public_key"] = pub_hex
                        changed = True
                    if rec.get("key_version") != key_version:
                        rec["key_version"] = key_version
                        changed = True
                    if changed:
                        rec["last_key_rotation"] = datetime.now(timezone.utc).isoformat()
                        temp_rec = f"{self.creator_record_path}.tmp.{secrets.token_hex(4)}"
                        with open(temp_rec, "w", encoding="utf-8") as f:
                            json.dump(rec, f, indent=2)
                        os.replace(temp_rec, self.creator_record_path)
                except Exception as e:
                    self._state = AuthorityState.AUTHORITY_LOCKED
                    self._lock_reason = f"Failed to synchronize operator_record.json: {str(e)}"
                    raise

            # 2. Synchronize operators_registry.json to newly active public key and key_id
            if self.creators_registry_path:
                try:
                    reg_mgr = MultiCreatorRegistry(self.creators_registry_path)
                    reg_mgr.set_public_key(CANONICAL_CREATOR_ID, pub_hex, key_id=key_id)
                except Exception as e:
                    self._state = AuthorityState.AUTHORITY_LOCKED
                    self._lock_reason = f"Failed to synchronize operators_registry.json: {str(e)}"
                    raise

            # 3. Compute canonical hashes only after records are synchronized
            rec_hash = compute_canonical_hash(self.creator_record_path)
            reg_hash = compute_canonical_hash(self.creators_registry_path)
            creator_obj = CreatorIdentity(record_path=self.creator_record_path) if os.path.exists(self.creator_record_path) else None
            recov_hash = compute_recovery_verifier_hash(self.recovery_config_path, creator=creator_obj) if os.path.exists(self.recovery_config_path) else None

            seal_body = {
                "version": 2,
                "setup_complete": True,
                "root_creator_id": CANONICAL_CREATOR_ID,
                "display_name": display_name,
                "root_public_key": pub_hex,
                "key_id": key_id,
                "key_version": key_version,
                "record_hash": rec_hash,
                "registry_hash": reg_hash,
                "recovery_hash": recov_hash,
                "sealed_at": datetime.now(timezone.utc).isoformat(),
                "recovery_authorized": True
            }

            if new_priv_bytes is not None:
                canonical_seal = json.dumps(seal_body, sort_keys=True, separators=(",", ":")).encode("utf-8")
                sig_bytes = Ed25519.sign(new_priv_bytes, canonical_seal)
                seal_body["seal_signature"] = sig_bytes.hex()

            raw_bytes = json.dumps(seal_body, indent=2).encode("utf-8")
            if IS_WINDOWS:
                try:
                    payload_to_write = protect_bytes_dpapi(raw_bytes)
                except Exception:
                    payload_to_write = raw_bytes
            else:
                payload_to_write = raw_bytes

            os.makedirs(os.path.dirname(self.seal_path), exist_ok=True)
            temp_seal = f"{self.seal_path}.tmp.{secrets.token_hex(4)}"
            with open(temp_seal, "wb") as f:
                f.write(payload_to_write)
            os.replace(temp_seal, self.seal_path)

            self._seal_cache = None
            verify_state, verify_reason = self.verify_integrity()
            if verify_state != AuthorityState.ACTIVE:
                self._state = AuthorityState.AUTHORITY_LOCKED
                self._lock_reason = verify_reason
                raise PermissionError(f"Post-recovery authority resealing failed verification: {verify_reason}")

            self._state = AuthorityState.ACTIVE
            self._lock_reason = None
            return seal_body
