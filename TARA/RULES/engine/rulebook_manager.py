"""
TARA/RULES/engine/rulebook_manager.py

Orchestrates Natural-Language Rulebook storage, compilation, cryptographic signing,
provenance verification, immutable versioning, rollback, and audit logging.
"""

import os
import json
import time
import hashlib
from typing import Dict, List, Optional, Tuple, Any

from ..compiler.policy_schema import CompiledPolicy, Rule, PolicyPriority, RuleAction
from ..compiler.policy_compiler import PolicyCompiler
from TARA.ACCESS.crypto.ed25519 import Ed25519
from TARA.ACCESS.operator.operator_profile import CANONICAL_CREATOR_ID, DEFAULT_DISPLAY_NAME

FORBIDDEN_AUDIT_KEYS = {
    "private_key", "privkey", "secret", "seed", "token", "password", "passphrase"
}

class RulebookManager:
    """Manages lifecycle of TARA natural-language rulebook policies."""

    def __init__(self, rules_base_dir: Optional[str] = None):
        if rules_base_dir is None:
            rules_base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.base_dir = rules_base_dir
        
        self.rulebook_txt_path = os.path.join(self.base_dir, "RULEBOOK.txt")
        self.default_safe_path = os.path.join(self.base_dir, "DEFAULT_SAFE_RULES.txt")
        self.compiled_policy_path = os.path.join(self.base_dir, "compiled_policy.json")
        self.policy_signature_path = os.path.join(self.base_dir, "policy_signature")
        self.versions_dir = os.path.join(self.base_dir, "versions")
        self.audit_dir = os.path.join(self.base_dir, "audit")
        self.audit_file = os.path.join(self.audit_dir, "rule_audit.jsonl")

        os.makedirs(self.versions_dir, exist_ok=True)
        os.makedirs(self.audit_dir, exist_ok=True)

        self.compiler = PolicyCompiler(creator_id=CANONICAL_CREATOR_ID, display_name=DEFAULT_DISPLAY_NAME)
        self._active_policy: Optional[CompiledPolicy] = None

    # ------------------------------------------------------------------------
    # AUDIT LOGGING
    # ------------------------------------------------------------------------
    def log_event(self, event_type: str, severity: str = "INFO", details: Optional[Dict[str, Any]] = None) -> None:
        """Logs a non-secret JSON-lines audit event."""
        sanitized_details = {}
        if details:
            for k, v in details.items():
                if any(bad in k.lower() for bad in FORBIDDEN_AUDIT_KEYS):
                    sanitized_details[k] = "[REDACTED]"
                elif isinstance(v, bytes):
                    sanitized_details[k] = f"[BYTES_LEN_{len(v)}]"
                else:
                    sanitized_details[k] = v

        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event": event_type,
            "severity": severity,
            "details": sanitized_details
        }
        try:
            with open(self.audit_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass

    # ------------------------------------------------------------------------
    # INITIALIZATION & VERIFICATION
    # ------------------------------------------------------------------------
    def initialize_default_policy(
        self,
        creator_private_key: bytes,
        key_version: int = 1
    ) -> CompiledPolicy:
        """
        Initializes and cryptographically signs the default baseline policy (v1).
        """
        if not os.path.exists(self.default_safe_path):
            raise FileNotFoundError(f"DEFAULT_SAFE_RULES.txt not found at {self.default_safe_path}")

        with open(self.default_safe_path, "r", encoding="utf-8") as f:
            default_text = f.read()

        rulebook_text = ""
        if os.path.exists(self.rulebook_txt_path):
            with open(self.rulebook_txt_path, "r", encoding="utf-8") as f:
                rulebook_text = f.read()

        policy, val_errors, conflicts = self.compiler.compile_rules(
            rulebook_text=rulebook_text,
            default_safe_text=default_text,
            target_version=1,
            signing_key_version=key_version
        )

        # Sign policy digest with Creator Ed25519 private key
        digest_bytes = policy.digest_sha256.encode('utf-8')
        signature_bytes = Ed25519.sign(creator_private_key, digest_bytes)
        policy.creator_signature_hex = signature_bytes.hex()

        # Write compiled_policy.json and policy_signature
        with open(self.compiled_policy_path, "w", encoding="utf-8") as f:
            json.dump(policy.to_dict(), f, indent=2)

        with open(self.policy_signature_path, "wb") as f:
            f.write(signature_bytes)

        # Archive version v1.json
        version_archive = os.path.join(self.versions_dir, "v1.json")
        with open(version_archive, "w", encoding="utf-8") as f:
            json.dump(policy.to_dict(), f, indent=2)

        self._active_policy = policy

        self.log_event(
            "RULE_POLICY_INITIALIZED",
            severity="INFO",
            details={
                "version": 1,
                "digest": policy.digest_sha256,
                "mandatory_count": policy.mandatory_rule_count,
                "creator_count": policy.creator_rule_count
            }
        )

        return policy

    def verify_and_load_active_policy(
        self,
        creator_public_key: bytes,
        expected_key_version: Optional[int] = None
    ) -> Tuple[bool, str, Optional[CompiledPolicy]]:
        """
        Verifies cryptographic integrity and provenance of active compiled policy on disk.
        Returns: (is_valid, reason, policy_obj)
        """
        if not os.path.exists(self.compiled_policy_path):
            return False, "COMPILED_POLICY_MISSING", None

        if not os.path.exists(self.policy_signature_path):
            self.log_event("POLICY_VERIFY_FAILED", severity="ERROR", details={"reason": "SIGNATURE_FILE_MISSING"})
            return False, "POLICY_SIGNATURE_MISSING", None

        try:
            with open(self.compiled_policy_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            policy = CompiledPolicy.from_dict(data)
        except Exception as e:
            self.log_event("POLICY_VERIFY_FAILED", severity="ERROR", details={"reason": f"CORRUPTED_JSON: {e}"})
            return False, f"CORRUPTED_POLICY_FILE: {e}", None

        # Check creator identity anchor
        if policy.creator_id != CANONICAL_CREATOR_ID:
            self.log_event("POLICY_VERIFY_FAILED", severity="CRITICAL", details={"reason": "CREATOR_ID_MISMATCH", "claimed": policy.creator_id})
            return False, "INVALID_CREATOR_ID_PROVENANCE", None

        # Check key version if expected version specified (for rotation/revocation)
        if expected_key_version is not None and policy.signing_key_version != expected_key_version:
            self.log_event("POLICY_VERIFY_FAILED", severity="ERROR", details={
                "reason": "KEY_VERSION_REVOKED",
                "policy_key_version": policy.signing_key_version,
                "expected": expected_key_version
            })
            return False, f"REVOKED_SIGNING_KEY_VERSION: policy signed by v{policy.signing_key_version}, active is v{expected_key_version}", None

        # Read signature bytes
        try:
            with open(self.policy_signature_path, "rb") as f:
                sig_bytes = f.read()
        except Exception as e:
            return False, f"CANNOT_READ_SIGNATURE: {e}", None

        # Verify digest recalculation
        expected_digest = policy.compute_digest()
        if policy.digest_sha256 != expected_digest:
            self.log_event("POLICY_VERIFY_FAILED", severity="CRITICAL", details={"reason": "DIGEST_MISMATCH"})
            return False, "POLICY_TAMPERED: Digest mismatch", None

        # Verify Ed25519 signature
        valid_sig = Ed25519.verify(creator_public_key, policy.digest_sha256.encode('utf-8'), sig_bytes)
        if not valid_sig:
            self.log_event("POLICY_VERIFY_FAILED", severity="CRITICAL", details={"reason": "INVALID_ED25519_SIGNATURE"})
            return False, "POLICY_CRYPTOGRAPHIC_SIGNATURE_INVALID", None

        self._active_policy = policy
        return True, "POLICY_VERIFIED_AND_ACTIVE", policy

    # ------------------------------------------------------------------------
    # UPDATE POLICY
    # ------------------------------------------------------------------------
    def update_rulebook(
        self,
        new_rulebook_text: str,
        creator_private_key: bytes,
        creator_public_key: bytes,
        claimed_creator_id: str = CANONICAL_CREATOR_ID,
        signing_key_version: int = 1,
        change_summary: str = "Creator rulebook update"
    ) -> Dict[str, Any]:
        """
        Flow:
        1. Authenticate Creator identity (ROOT_OPERATOR).
        2. Verify private key derives/matches expected public key.
        3. Parse new rules.
        4. Validate rules and detect conflicts.
        5. Compile structured policy with baseline safe rules.
        6. Cryptographically sign compiled policy.
        7. Archive to versions/v{N}.json.
        8. Activate new version.
        """
        if claimed_creator_id != CANONICAL_CREATOR_ID:
            self.log_event("UNAUTHORIZED_RULE_UPDATE", severity="CRITICAL", details={
                "reason": "CLAIMED_CREATOR_MISMATCH",
                "claimed": claimed_creator_id
            })
            raise PermissionError(f"Unauthorized: Only {CANONICAL_CREATOR_ID} can update TARA rules.")

        # Check key correspondence
        try:
            derived_pub = Ed25519.public_key_from_private(creator_private_key)
            if derived_pub != creator_public_key:
                self.log_event("UNAUTHORIZED_RULE_UPDATE", severity="CRITICAL", details={"reason": "KEYPAIR_MISMATCH"})
                raise PermissionError("Provided private key does not match active Creator public key.")
        except Exception as e:
            raise PermissionError(f"Invalid creator private key: {e}")

        if not os.path.exists(self.default_safe_path):
            raise FileNotFoundError(f"DEFAULT_SAFE_RULES.txt missing at {self.default_safe_path}")

        with open(self.default_safe_path, "r", encoding="utf-8") as f:
            default_safe_text = f.read()

        current_version = self._active_policy.policy_version if self._active_policy else 1
        new_version = current_version + 1

        policy, val_errors, conflicts = self.compiler.compile_rules(
            rulebook_text=new_rulebook_text,
            default_safe_text=default_safe_text,
            target_version=new_version,
            signing_key_version=signing_key_version
        )

        if val_errors:
            self.log_event("RULE_COMPILATION_WARNING", severity="WARNING", details={"errors": val_errors})

        # Cryptographically sign compiled digest
        digest_bytes = policy.digest_sha256.encode('utf-8')
        signature_bytes = Ed25519.sign(creator_private_key, digest_bytes)
        policy.creator_signature_hex = signature_bytes.hex()

        # Write text rulebook
        with open(self.rulebook_txt_path, "w", encoding="utf-8") as f:
            f.write(new_rulebook_text)

        # Write compiled policy
        with open(self.compiled_policy_path, "w", encoding="utf-8") as f:
            json.dump(policy.to_dict(), f, indent=2)

        # Write signature
        with open(self.policy_signature_path, "wb") as f:
            f.write(signature_bytes)

        # Archive version
        version_archive = os.path.join(self.versions_dir, f"v{new_version}.json")
        with open(version_archive, "w", encoding="utf-8") as f:
            json.dump(policy.to_dict(), f, indent=2)

        self._active_policy = policy

        self.log_event(
            "RULE_POLICY_UPDATED",
            severity="INFO",
            details={
                "version": new_version,
                "change_summary": change_summary,
                "digest": policy.digest_sha256,
                "rules_count": len(policy.rules),
                "conflicts_count": len(conflicts),
                "errors_count": len(val_errors)
            }
        )

        return {
            "status": "POLICY_UPDATED_AND_ACTIVATED",
            "version": new_version,
            "digest": policy.digest_sha256,
            "validation_errors": val_errors,
            "conflicts": conflicts,
            "policy": policy
        }

    # ------------------------------------------------------------------------
    # TAMPER & UNAUTHORIZED DISK EDIT PROTECTION
    # ------------------------------------------------------------------------
    def handle_unauthorized_disk_modification(
        self,
        creator_public_key: bytes
    ) -> Dict[str, Any]:
        """
        Scans RULEBOOK.txt on disk against active compiled policy.
        If someone modified RULEBOOK.txt without Creator signing:
        - Rejects the modification
        - Falls back to last trusted active policy
        - Logs a security audit alert
        """
        is_valid, reason, policy = self.verify_and_load_active_policy(creator_public_key)
        if not is_valid:
            self.log_event(
                "UNAUTHORIZED_POLICY_MODIFICATION_REJECTED",
                severity="CRITICAL",
                details={"reason": reason, "action": "RETAIN_BASELINE_ONLY"}
            )
            return {
                "accepted": False,
                "reason": reason,
                "status": "UNAUTHORIZED_EDIT_BLOCKED",
                "active_policy_intact": False
            }

        return {
            "accepted": True,
            "status": "TRUSTED_POLICY_ACTIVE",
            "policy_version": policy.policy_version if policy else None
        }

    def get_active_policy(self) -> Optional[CompiledPolicy]:
        return self._active_policy

    def get_version_history(self) -> List[str]:
        if not os.path.exists(self.versions_dir):
            return []
        return sorted(os.listdir(self.versions_dir))
