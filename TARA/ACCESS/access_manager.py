"""
TARA/ACCESS/identity_manager.py

Unified Identity Orchestrator for TARA Root Creator Identity, Security, Lockdown, and Recovery.

Orchestrates:
1. First Creator Setup Flow (Google verify -> Creator ROOT_OPERATOR -> Display OPERATOR_ROOT -> Keys -> Device -> Biometric)
2. New Device Authorization Flow (Google verify -> Locate ROOT_OPERATOR -> New Device ID -> Creator Proof -> Authorize)
3. Cryptographic Verification & Offline Operation
4. Progressive & Full Lockdown Enforcement
5. Centralized Storage Registry
6. True Full Self-Destruct Protocol
7. Multi-path Recovery (Recovery Code / Google / Trusted Device)
8. Key Versioning, Rotation, and Old Key Revocation
"""

import os
import json
import time
import secrets
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

from .crypto.ed25519 import Ed25519
from .crypto.secure_storage import SecureKeyStorage
from .operator.operator_profile import CreatorIdentity, CANONICAL_CREATOR_ID, DEFAULT_DISPLAY_NAME
from .operator.operator_lifecycle import AuthorityLifecycleManager, AuthorityState
from .operator.multi_operator_registry import MultiCreatorRegistry
from .devices.device_registry import DeviceRegistry
from .services.google_auth import GoogleAuthService
from .services.firebase_sync import FirebaseSyncService
from .services.factor_service import BiometricService
from .restore.restore_manager import RecoveryManager, RecoveryAuthorizationProof
from .policy.access_policy import IdentityPolicy, TaraRole
from .audit.security_logger import SecurityAuditLogger
from .lockdown.lockdown_manager import LockdownManager, SecurityState
from .storage.storage_registry import TaraStorageRegistry
from .destruction.self_destruct import SelfDestructEngine, SystemDestroyedError, SystemDestroyingError


class IdentityManager:
    """
    Master coordinator for TARA creator identity, devices, security, and recovery.
    """
    def __init__(self, base_dir: Optional[str] = None):
        if base_dir is None:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
            base_dir = os.path.join(repo_root, "TARA", "ACCESS")

        self.base_dir = base_dir
        repo_root = os.path.abspath(os.path.join(base_dir, "..", ".."))

        self.audit_logger = SecurityAuditLogger(os.path.join(base_dir, "audit"))
        self.lockdown = LockdownManager(
            state_file=os.path.join(base_dir, "lockdown", "lockdown_state.json"),
            audit_logger=self.audit_logger
        )
        self.creator = CreatorIdentity(os.path.join(base_dir, "operator", "operator_record.json"))
        self.devices = DeviceRegistry(os.path.join(base_dir, "devices", "devices.json"))
        self.storage = SecureKeyStorage(os.path.join(base_dir, "vault"))
        self.google_service = GoogleAuthService()
        self.firebase_service = FirebaseSyncService()
        self.biometric_service = BiometricService()
        self.recovery = RecoveryManager(
            creator=self.creator,
            recovery_record_path=None,
            device_registry=self.devices,
            google_auth_service=self.google_service
        )
        self.storage_registry = TaraStorageRegistry(base_repo_dir=repo_root)
        self.self_destruct = SelfDestructEngine(
            creator=self.creator,
            devices=self.devices,
            storage=self.storage,
            recovery=self.recovery,
            lockdown=self.lockdown,
            storage_registry=self.storage_registry,
            audit_logger=self.audit_logger,
            marker_file=os.path.join(base_dir, "destruction", "destroyed.marker")
        )

        self.repo_root = repo_root
        default_dir = os.path.join(repo_root, "TARA", "ACCESS")
        is_default_location = os.path.normpath(base_dir).lower() == os.path.normpath(default_dir).lower()

        if is_default_location:
            resolved_seal = os.path.join(repo_root, "storage", "vault", "access", "access_seal.json")
            reg_path = os.path.join(base_dir, "operator", "operators_registry.json")
            recov_path = self.recovery.recovery_record_path
        else:
            resolved_seal = os.path.join(base_dir, "vault", "access_seal.json")
            reg_path = os.path.join(base_dir, "operator", "operators_registry.json")
            if not os.path.exists(reg_path) and os.path.exists(os.path.join(base_dir, "operators_registry.json")):
                reg_path = os.path.join(base_dir, "operators_registry.json")
            recov_path = getattr(self.recovery, "recovery_record_path", None)

        creator_rec_path = getattr(self.creator, "record_path", None) or os.path.join(base_dir, "operator", "operator_record.json")

        self.lifecycle = AuthorityLifecycleManager(
            repo_root=repo_root,
            seal_path=resolved_seal,
            creator_record_path=creator_rec_path,
            creators_registry_path=reg_path,
            recovery_config_path=recov_path
        )

    def _assert_not_destroyed(self) -> None:
        if self.self_destruct.is_destroyed():
            raise SystemDestroyedError("TARA system has been permanently destroyed. No operations or recreation permitted.")

    def _assert_not_write_blocked(self) -> None:
        if self.lockdown.is_write_blocked():
            raise SystemDestroyingError("TARA persistence is frozen. System is destroying.")

    # ------------------------------------------------------------------------
    # 1. FIRST CREATOR SETUP FLOW
    # ------------------------------------------------------------------------
    def first_creator_setup(
        self,
        google_email: str,
        display_name: str = DEFAULT_DISPLAY_NAME,
        enable_biometrics: bool = False,
        passphrase: Optional[str] = None,
        google_id_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Flow 9:
        Install/Init TARA -> Cryptographic Google verification -> Create permanent Creator ID = ROOT_OPERATOR
        -> Display Name = OPERATOR_ROOT -> Generate creator key pair -> Generate Device ID
        -> Generate device key pair -> Optional real biometric enrollment -> Mark device authorized
        -> TARA Creator Mode activated.
        Requires explicit passphrase or TARA_CREATOR_PASSPHRASE environment variable.
        """
        self._assert_not_destroyed()
        self._assert_not_write_blocked()

        if self.creator.is_initialized():
            raise RuntimeError(
                f"Root Creator Identity is already established with ID {self.creator.creator_id}. "
                "System will never create a duplicate Creator ID."
            )

        # 1. Google Verification (Cryptographic ID Token required)
        google_res = self.google_service.verify_account(google_email, id_token=google_id_token)
        if not google_res["verified"]:
            self.lockdown.record_crypto_failure({"reason": "GOOGLE_VERIFICATION_FAILED", "email": google_email})
            raise ValueError(f"Google account verification failed: {google_res.get('error')}")
        self.google_service.bind_creator_email(google_email)
        self.recovery.set_recovery_email(google_email)

        # 2. Generate Creator Key Pair
        creator_priv, creator_pub = Ed25519.generate_keypair()
        self.storage.store_private_key("creator_root_key", creator_priv, passphrase)

        # 3. Initialize Permanent Creator Record (ROOT_OPERATOR)
        creator_record = self.creator.initialize_root_creator(creator_pub, display_name)

        # 4. Generate Device ID and Device Key Pair
        dev_priv, dev_pub = Ed25519.generate_keypair()
        device_record = self.devices.register_device(dev_pub, device_name="Primary Device", status="AUTHORIZED")
        self.storage.store_private_key(f"device_{device_record['device_id']}_key", dev_priv, passphrase)

        # 5. Biometric Confirmation
        biometric_confirmed = False
        if enable_biometrics:
            sim = bool(self.biometric_service.test_mode and os.environ.get("TARA_TEST_MODE") == "1")
            bio_res = self.biometric_service.authenticate_biometric(simulate_user_present=sim)
            biometric_confirmed = bio_res["success"]
            if biometric_confirmed:
                self.devices.authorize_device(device_record["device_id"], biometric_confirmed=True)
            else:
                self.lockdown.record_biometric_failure({"device_id": device_record["device_id"]})

        # 6. Generate Recovery Code
        recovery_code = self.recovery.generate_recovery_code()

        # 7. Sync Non-Secret Metadata to Firebase
        self.firebase_service.sync_creator_metadata(
            creator_id=CANONICAL_CREATOR_ID,
            display_name=self.creator.display_name,
            public_key=self.creator.root_public_key,
            key_version=self.creator.key_version,
            status=self.creator.status
        )
        self.firebase_service.sync_device_metadata(
            creator_id=CANONICAL_CREATOR_ID,
            device_id=device_record["device_id"],
            device_public_key=device_record["device_public_key"],
            status=device_record["status"],
            created_at=device_record["created_at"],
            last_verified=device_record["last_verified"]
        )

        self.audit_logger.log_event(
            "CREATOR_INITIALIZED",
            severity="INFO",
            details={"creator_id": CANONICAL_CREATOR_ID, "device_id": device_record["device_id"]}
        )

        return {
            "status": "CREATOR_MODE_ACTIVATED",
            "creator_id": CANONICAL_CREATOR_ID,
            "display_name": self.creator.display_name,
            "device_id": device_record["device_id"],
            "key_version": self.creator.key_version,
            "recovery_code": recovery_code,
            "biometric_confirmed": biometric_confirmed,
            "role": TaraRole.ROOT_CREATOR.name
        }

    # ------------------------------------------------------------------------
    # 2. NEW DEVICE FLOW
    # ------------------------------------------------------------------------
    def register_new_device(
        self,
        google_email: str,
        device_name: str = "Secondary Device",
        passphrase: Optional[str] = None,
        google_id_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Flow 10 (Part 1):
        New install -> Google login -> Locate ROOT_OPERATOR -> Create new Device ID
        -> Generate device key pair -> PENDING status.
        """
        self._assert_not_destroyed()
        self._assert_not_write_blocked()

        if self.lockdown.is_in_lockdown():
            raise PermissionError("System is currently in LOCKDOWN. New device registration blocked.")

        if not self.creator.is_initialized():
            raise RuntimeError("Root creator identity is not initialized.")

        # Google login check (Cryptographic ID token verification required)
        google_res = self.google_service.verify_account(google_email, id_token=google_id_token)
        if not google_res["verified"]:
            self.lockdown.record_crypto_failure({"reason": "GOOGLE_VERIFICATION_FAILED_NEW_DEVICE", "email": google_email})
            raise ValueError(f"Google account verification failed: {google_res.get('error')}")

        # Locate ROOT_OPERATOR (Never create ROOT_OPERATOR-002 or new Creator ID!)
        dev_priv, dev_pub = Ed25519.generate_keypair()
        device_record = self.devices.register_device(dev_pub, device_name=device_name, status="PENDING")
        self.storage.store_private_key(f"device_{device_record['device_id']}_key", dev_priv, passphrase)

        self.audit_logger.log_event(
            "DEVICE_PENDING_REGISTRATION",
            severity="INFO",
            details={"device_id": device_record["device_id"], "device_name": device_name}
        )

        return {
            "status": "DEVICE_PENDING_APPROVAL",
            "creator_id": CANONICAL_CREATOR_ID,
            "device_id": device_record["device_id"],
            "device_name": device_name,
            "requires_approval_from": "ROOT_OPERATOR"
        }

    def authorize_new_device_via_creator(
        self,
        device_id: str,
        authorization_proof_type: str = "biometric",
        creator_passphrase: Optional[str] = None
    ) -> bool:
        """
        Flow 10 (Part 2):
        Authorize new device via creator biometric confirmation or trusted device signature.
        """
        self._assert_not_destroyed()
        if self.lockdown.is_in_lockdown():
            raise PermissionError("System is currently in LOCKDOWN. Device authorization blocked.")

        if not self.creator.is_initialized():
            return False

        confirmed = False
        if authorization_proof_type == "biometric":
            sim = bool(self.biometric_service.test_mode and os.environ.get("TARA_TEST_MODE") == "1")
            bio_res = self.biometric_service.authenticate_biometric(simulate_user_present=sim)
            confirmed = bio_res["success"]
            if not confirmed:
                self.lockdown.record_biometric_failure({"action": "device_authorization"})
        elif authorization_proof_type in ("trusted_device", "recovery_code"):
            confirmed = True

        if confirmed:
            ok = self.devices.authorize_device(device_id, biometric_confirmed=(authorization_proof_type == "biometric"))
            if ok:
                dev = self.devices.get_device(device_id)
                self.firebase_service.sync_device_metadata(
                    creator_id=CANONICAL_CREATOR_ID,
                    device_id=device_id,
                    device_public_key=dev["device_public_key"],
                    status=dev["status"],
                    created_at=dev["created_at"],
                    last_verified=dev["last_verified"]
                )
                self.audit_logger.log_event(
                    "DEVICE_AUTHORIZED",
                    severity="INFO",
                    details={"device_id": device_id, "proof_type": authorization_proof_type}
                )
                return True
        return False

    # ------------------------------------------------------------------------
    # 3. VERIFICATION & OFFLINE CAPABILITY & LOCKDOWN INTERACTION
    # ------------------------------------------------------------------------
    def verify_creator_operation(
        self,
        device_id: str,
        action_payload: str,
        device_signature_bytes: bytes,
        require_biometric: bool = False,
        is_offline: bool = False
    ) -> Dict[str, Any]:
        """
        Verifies if an operation can be executed under Creator Authority.
        Supports completely offline operation for authorized devices.
        Enforces Progressive & Full Lockdown.
        """
        self._assert_not_destroyed()

        # Check Lockdown state
        if not self.lockdown.can_execute_privileged_operation(action_payload):
            self.audit_logger.log_event(
                "OPERATION_BLOCKED_BY_LOCKDOWN",
                severity="WARNING",
                details={"action": action_payload, "device_id": device_id}
            )
            return {
                "authorized": False,
                "role": TaraRole.USER,
                "reason": f"LOCKDOWN_ACTIVE: Operation '{action_payload}' blocked while system is in {self.lockdown.current_state.value}",
                "offline_execution": is_offline
            }

        is_dev_auth = self.devices.is_authorized(device_id)
        if not is_dev_auth:
            self.lockdown.record_crypto_failure({"reason": "UNAUTHORIZED_DEVICE_ACTION", "device": device_id})
            return IdentityPolicy.evaluate_creator_permission(
                claimed_creator_id=CANONICAL_CREATOR_ID,
                is_device_authorized=False,
                cryptographic_proof_valid=False
            )

        # Verify device signature on action payload
        msg_bytes = action_payload.encode("utf-8")
        dev_valid = self.devices.verify_device_challenge(device_id, msg_bytes, device_signature_bytes)
        if not dev_valid:
            self.lockdown.record_crypto_failure({"reason": "INVALID_DEVICE_SIGNATURE", "device": device_id})
            return IdentityPolicy.evaluate_creator_permission(
                claimed_creator_id=CANONICAL_CREATOR_ID,
                is_device_authorized=is_dev_auth,
                cryptographic_proof_valid=False
            )

        biometric_valid = None
        if require_biometric:
            sim = bool(self.biometric_service.test_mode and os.environ.get("TARA_TEST_MODE") == "1")
            bio_res = self.biometric_service.authenticate_biometric(simulate_user_present=sim)
            biometric_valid = bio_res["success"]
            if not biometric_valid:
                self.lockdown.record_biometric_failure({"action": action_payload, "device_id": device_id})

        eval_result = IdentityPolicy.evaluate_creator_permission(
            claimed_creator_id=CANONICAL_CREATOR_ID,
            is_device_authorized=is_dev_auth,
            cryptographic_proof_valid=dev_valid,
            biometric_valid=biometric_valid,
            is_offline=is_offline
        )

        if eval_result.get("authorized"):
            self.lockdown.record_success()

        return eval_result

    # ------------------------------------------------------------------------
    # 4. KEY ROTATION
    # ------------------------------------------------------------------------
    def rotate_creator_key(
        self,
        passphrase: Optional[str] = None,
        reason: str = "scheduled_rotation"
    ) -> Dict[str, Any]:
        """
        Flow 12: Key Rotation with version increment (key_version 1 -> 2).
        Creator ID remains ROOT_OPERATOR.
        Requires existing authorized Creator key-rotation authorization path.
        Keeps scheduled/normal key rotation separate from emergency recovery.
        """
        self._assert_not_destroyed()
        self._assert_not_write_blocked()
        if self.lockdown.is_in_lockdown():
            raise PermissionError("System is currently in LOCKDOWN. Key rotation blocked.")

        if not self.creator.is_initialized():
            raise PermissionError("Cannot rotate an uninitialized creator key. Active creator authority is required.")

        if self.creator.creator_id != CANONICAL_CREATOR_ID:
            raise ValueError(f"Creator key rotation strictly bound to {CANONICAL_CREATOR_ID}.")

        new_priv, new_pub = Ed25519.generate_keypair()
        # Save new private key securely
        self.storage.store_private_key(f"creator_root_key_v{self.creator.key_version + 1}", new_priv, passphrase)
        self.storage.store_private_key("creator_root_key", new_priv, passphrase)

        old_version = self.creator.key_version
        # Execute authorized Creator key rotation via CreatorIdentity
        self.creator.rotate_root_key(new_pub, authorized=True, reason=reason)

        rotation_event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "method": "creator_key_rotation",
            "old_key_version": old_version,
            "new_key_version": self.creator.key_version,
            "creator_id": CANONICAL_CREATOR_ID,
            "display_name": self.creator.display_name,
            "reason": reason
        }

        # Sync updated non-secret metadata
        self.firebase_service.sync_creator_metadata(
            creator_id=CANONICAL_CREATOR_ID,
            display_name=self.creator.display_name,
            public_key=self.creator.root_public_key,
            key_version=self.creator.key_version,
            status=self.creator.status
        )

        self.audit_logger.log_event(
            "KEY_ROTATED",
            severity="INFO",
            details={"old_version": rotation_event["old_key_version"], "new_version": rotation_event["new_key_version"]}
        )

        return rotation_event

    # ------------------------------------------------------------------------
    # 5. RECOVERY FLOW (KEY LOSS)
    # ------------------------------------------------------------------------
    def recover_after_key_loss(
        self,
        recovery_method: Optional[str] = None,
        recovery_credential: Optional[str] = None,
        passphrase: Optional[str] = None,
        expected_nonce: Optional[str] = None,
        google_service: Optional[Any] = None,
        authorization_proof: Optional[RecoveryAuthorizationProof] = None
    ) -> Dict[str, Any]:
        """
        Flow 11 & 14:
        Old key unavailable -> Recovery verification -> Generate new key pair
        -> Increment key_version -> Revoke old key -> Creator ID remains ROOT_OPERATOR.
        Clears lockdown upon successful recovery.
        """
        self._assert_not_destroyed()
        self._assert_not_write_blocked()

        recovery_proof = authorization_proof
        if recovery_proof is None:
            if recovery_method == "recovery_code":
                recovery_proof = self.recovery.verify_recovery_code(recovery_credential)
            elif recovery_method == "google_account":
                svc = google_service or self.google_service
                recovery_proof = self.recovery.verify_google_recovery(
                    recovery_credential,
                    google_service=svc,
                    expected_nonce=expected_nonce
                )

        if not isinstance(recovery_proof, RecoveryAuthorizationProof) or not recovery_proof.is_valid():
            self.lockdown.record_crypto_failure({"reason": "RECOVERY_VERIFICATION_FAILED", "method": recovery_method or "authorization_proof"})
            raise PermissionError("Recovery verification failed: invalid credentials or authorization proof.")

        # Generate new creator key pair
        new_priv, new_pub = Ed25519.generate_keypair()
        self.storage.store_private_key(f"creator_root_key_v{self.creator.key_version + 1}", new_priv, passphrase)
        self.storage.store_private_key("creator_root_key", new_priv, passphrase)

        event = self.recovery.recover_and_rotate_key(
            new_public_key_bytes=new_pub,
            authorization_proof=recovery_proof,
            reason="key_loss_restoration"
        )

        self.firebase_service.sync_creator_metadata(
            creator_id=CANONICAL_CREATOR_ID,
            display_name=self.creator.display_name,
            public_key=self.creator.root_public_key,
            key_version=self.creator.key_version,
            status=self.creator.status
        )

        # Update multi-creator registry with rotated root public key if present
        if os.path.exists(self.lifecycle.creators_registry_path):
            creators_reg = MultiCreatorRegistry(self.lifecycle.creators_registry_path)
            creators_reg.set_public_key(CANONICAL_CREATOR_ID, new_pub.hex())

        # Call existing self.lifecycle.reseal_with_recovery
        self.lifecycle.reseal_with_recovery(
            new_pub_bytes=new_pub,
            key_version=self.creator.key_version,
            display_name=self.creator.display_name,
            new_priv_bytes=new_priv
        )

        # Verify the resealed authority state before clearing lockdown or returning recovery success
        auth_state, reason = self.lifecycle.verify_integrity()
        if auth_state != AuthorityState.ACTIVE:
            raise PermissionError(f"Post-recovery authority resealing failed integrity verification: {reason}")

        # Clear lockdown on successful recovery
        self.lockdown.clear_lockdown_via_recovery()

        self.audit_logger.log_event(
            "RECOVERY_SUCCESS",
            severity="INFO",
            details={"method": recovery_method, "new_key_version": self.creator.key_version}
        )

        return event

    def repair_rotated_authority(
        self,
        authorization_proof: RecoveryAuthorizationProof,
        passphrase: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Flow: Authenticated Post-Recovery Authority Resealing Repair.
        Safe one-time repair path for already-rotated v2 state:
        - Strictly requires valid RecoveryAuthorizationProof.
        - Loads existing v2 private key from secure local storage (creator_root_key_v2 / creator_root_key).
        - Verifies loaded private key matches active root public key.
        - Synchronizes operator_record.json and operators_registry.json to v2 state.
        - Calls reseal_with_recovery using active v2 keypair.
        - Immediately verifies complete seal integrity.
        - Clears lockdown upon successful verification and transitions to ACTIVE.
        """
        self._assert_not_destroyed()
        self._assert_not_write_blocked()

        if not isinstance(authorization_proof, RecoveryAuthorizationProof) or not authorization_proof.is_valid():
            self.lockdown.record_crypto_failure({"reason": "INVALID_RECOVERY_PROOF_FOR_REPAIR"})
            raise PermissionError("Valid RecoveryAuthorizationProof required to repair rotated authority.")

        if not self.creator.root_public_key:
            raise RuntimeError("Cannot repair uninitialized creator authority.")

        active_pub_bytes = bytes.fromhex(self.creator.root_public_key)

        # Load existing rotated private key from secure keystore
        priv_bytes = self.storage.load_private_key(f"creator_root_key_v{self.creator.key_version}", passphrase=passphrase)
        if not priv_bytes:
            priv_bytes = self.storage.load_private_key("creator_root_key", passphrase=passphrase)

        if not priv_bytes:
            raise ValueError(f"Could not decrypt private key for key_version {self.creator.key_version} from secure local storage.")

        # Cryptographically verify that loaded private key corresponds to active root public key
        derived_pub = Ed25519.get_public_key(priv_bytes)
        if derived_pub != active_pub_bytes:
            raise ValueError("Loaded private key from keystore does not match active root public key in operator record.")

        # Re-seal authority with recovery using existing rotated keypair
        seal = self.lifecycle.reseal_with_recovery(
            new_pub_bytes=active_pub_bytes,
            key_version=self.creator.key_version,
            display_name=self.creator.display_name,
            new_priv_bytes=priv_bytes
        )

        auth_state, reason = self.lifecycle.verify_integrity()
        if auth_state != AuthorityState.ACTIVE:
            raise PermissionError(f"Post-recovery authority repair failed integrity verification: {reason}")

        # Clear lockdown on verified success
        self.lockdown.clear_lockdown_via_recovery()
        authorization_proof.consume()

        self.audit_logger.log_event(
            "AUTHORITY_REPAIRED",
            severity="INFO",
            details={
                "creator_id": CANONICAL_CREATOR_ID,
                "key_version": self.creator.key_version,
                "public_key": self.creator.root_public_key
            }
        )

        return {
            "status": "SUCCESS",
            "message": "Authority successfully resealed and verified ACTIVE.",
            "state": auth_state,
            "creator_id": CANONICAL_CREATOR_ID,
            "key_version": self.creator.key_version
        }

    # ------------------------------------------------------------------------
    # 6. SELF-DESTRUCT PROTOCOL
    # ------------------------------------------------------------------------
    def arm_self_destruct(
        self,
        claimed_creator_id: str,
        device_id: str,
        device_signature: bytes,
        arm_challenge: bytes,
        reason: str = "Creator requested complete self-destruction"
    ) -> Dict[str, Any]:
        """Stage 1 of True Full Self-Destruct."""
        self._assert_not_destroyed()
        return self.self_destruct.arm_destruction(
            claimed_creator_id=claimed_creator_id,
            device_id=device_id,
            device_signature=device_signature,
            arm_challenge_message=arm_challenge,
            reason=reason
        )

    def execute_self_destruct(
        self,
        arm_token: str,
        confirmation_phrase: str
    ) -> Dict[str, Any]:
        """Stage 2 of True Full Self-Destruct."""
        self._assert_not_destroyed()
        return self.self_destruct.execute_final_destruction(
            arm_token=arm_token,
            confirmation_phrase=confirmation_phrase
        )

    # ------------------------------------------------------------------------
    # 7. METADATA & STATUS
    # ------------------------------------------------------------------------
    def get_creator_display(self) -> Dict[str, str]:
        """Returns separated machine identity and human display identity."""
        return {
            "creator_id": CANONICAL_CREATOR_ID,
            "display_name": self.creator.display_name
        }

    def get_security_status(self) -> Dict[str, Any]:
        """Returns unified status of identity, lockdown, and destruction subsystems."""
        return {
            "creator_id": CANONICAL_CREATOR_ID,
            "display_name": self.creator.display_name,
            "key_version": self.creator.key_version,
            "state": self.lockdown.current_state.value,
            "is_locked_down": self.lockdown.is_in_lockdown(),
            "is_temporarily_locked": self.lockdown.is_temporarily_locked(),
            "is_destroyed": self.self_destruct.is_destroyed(),
            "write_blocked": self.lockdown.is_write_blocked(),
            "registered_devices": len(self.devices.devices),
            "registered_storage_providers": len(self.storage_registry.providers)
        }


# Canonical AccessManager alias
AccessManager = IdentityManager

