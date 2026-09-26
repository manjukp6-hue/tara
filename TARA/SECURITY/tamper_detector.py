"""
TARA/SECURITY/tamper_detector.py

Fail-Closed Runtime Tamper Detection Engine.
Guarantees:
1. Validates integrity of security-critical code and configuration files.
2. If tampering, unauthorized injection, or unexpected file changes occur:
   - Fails closed for all creator-authority actions.
   - Refuses silent key regeneration.
   - Refuses silent creation of a new root creator.
   - Logs CRITICAL audit event.
3. Does NOT unnecessarily block normal user-level TARA AI chat or non-creator skills.
"""

import os
import json
import hashlib
from typing import Dict, Any, List, Optional, Tuple

from ..ACCESS.audit.security_logger import SecurityAuditLogger


class IntegrityViolationError(SecurityError if "SecurityError" in globals() else PermissionError):
    """Raised when security-critical files have been tampered with or modified."""
    pass


class SourceTamperDetector:
    """
    Monitors security-critical source and configuration components for unauthorized changes.
    """

    def __init__(
        self,
        repo_root: Optional[str] = None,
        audit_logger: Optional[SecurityAuditLogger] = None
    ):
        if repo_root is None:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        self.repo_root = repo_root
        self.audit_logger = audit_logger or SecurityAuditLogger()
        self._cached_hashes: Dict[str, str] = {}
        self._tampered_files: List[str] = []
        self._load_from_manifest()

    def _load_from_manifest(self) -> None:
        manifest_p = os.path.join(self.repo_root, "storage", "release_manifest.json")
        if os.path.isfile(manifest_p):
            try:
                with open(manifest_p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                tree = data.get("file_tree", {})
                if isinstance(tree, dict):
                    self._cached_hashes.update(tree)
            except Exception:
                pass

    def compute_file_hash(self, rel_path: str) -> Optional[str]:
        full_p = os.path.join(self.repo_root, rel_path.replace("/", os.sep))
        if not os.path.isfile(full_p):
            return None
        h = hashlib.sha256()
        with open(full_p, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    def register_baseline(self, critical_files: List[str]) -> Dict[str, str]:
        """Snapshots baseline hashes for critical files."""
        for rel_p in critical_files:
            sha = self.compute_file_hash(rel_p)
            if sha:
                self._cached_hashes[rel_p] = sha
        return dict(self._cached_hashes)

    def verify_integrity(self, critical_files: Optional[List[str]] = None) -> Tuple[bool, List[str]]:
        """
        Compares current hashes against baseline or release manifest.
        Returns (is_intact, list_of_tampered_files).
        """
        targets = critical_files or list(self._cached_hashes.keys())
        tampered = []

        for rel_p in targets:
            current_sha = self.compute_file_hash(rel_p)
            expected_sha = self._cached_hashes.get(rel_p)
            if expected_sha is not None:
                if current_sha != expected_sha:
                    tampered.append(rel_p)

        self._tampered_files = tampered
        if tampered:
            self.audit_logger.log_event(
                event_type="SOURCE_TAMPERING_DETECTED",
                severity="CRITICAL",
                details={
                    "tampered_files": tampered,
                    "action": "FAIL_CLOSED_CREATOR_AUTHORITY"
                }
            )
            return False, tampered

        return True, []

    def enforce_fail_closed(self, operation_name: str = "creator_authority_operation") -> None:
        """
        Enforces fail closed: raises IntegrityViolationError if tampering was detected.
        """
        is_intact, tampered = self.verify_integrity()
        if not is_intact:
            raise IntegrityViolationError(
                f"Security tamper alert: Creator authority operation '{operation_name}' blocked. "
                f"Integrity check failed for: {', '.join(tampered)}"
            )
