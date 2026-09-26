"""
python/tara_core/control_plane/sync_manager.py

Auto-Synchronization Manager for TARA Control Plane.
Enforces differential synchronization against the Canonical Artifact Manifest:
1. Compares worker's reported artifact checksums against TARA/MANIFEST/canonical_manifest.json.
2. Identifies missing or diverged artifacts without blindly copying the entire repository.
3. Strictly validates SafeTensors model SHA-256 before granting READY state.
4. Fail-closed: If a worker has an unverified or corrupted model file, marks DEGRADED.
"""

import os
import sys
import json
import time
import hashlib
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tara_core.contracts import CANONICAL_MODEL_SHA256, CANONICAL_PROTOCOL_VERSION
from tara_core.control_plane.worker_registry import DynamicWorkerRegistry, WorkerState

logger = logging.getLogger("tara_core.control_plane.sync_manager")


@dataclass
class SyncEvaluationResult:
    is_fully_synchronized: bool
    is_model_verified: bool
    matching_artifacts: List[str] = field(default_factory=list)
    missing_artifacts: List[str] = field(default_factory=list)
    diverged_artifacts: List[str] = field(default_factory=list)
    model_sha256_found: Optional[str] = None
    protocol_version_found: Optional[str] = None
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_fully_synchronized": self.is_fully_synchronized,
            "is_model_verified": self.is_model_verified,
            "matching_count": len(self.matching_artifacts),
            "missing_artifacts": self.missing_artifacts,
            "diverged_artifacts": self.diverged_artifacts,
            "model_sha256_found": self.model_sha256_found,
            "protocol_version_found": self.protocol_version_found,
            "reason": self.reason
        }


class AutoSyncManager:
    """
    Coordinates artifact manifest checks and differential updates across cluster nodes.
    """

    def __init__(self, manifest_path: Optional[str] = None):
        if manifest_path is None:
            self.manifest_path = os.path.join(REPO_ROOT, "TARA", "MANIFEST", "canonical_manifest.json")
        else:
            self.manifest_path = manifest_path
        self._manifest_cache: Optional[Dict[str, Any]] = None

    def load_canonical_manifest(self) -> Dict[str, Any]:
        """Loads and returns the authoritative canonical artifact manifest."""
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(f"Canonical manifest not found at: {self.manifest_path}")

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self._manifest_cache = json.load(f)
        return self._manifest_cache

    def evaluate_worker(self, reported_artifacts: Dict[str, str], reported_protocol: str = "1.0.0") -> SyncEvaluationResult:
        """
        Evaluates a worker's reported artifact hashes against the canonical manifest.
        """
        manifest = self.load_canonical_manifest()
        artifacts_spec = manifest.get("artifacts", {})

        matching = []
        missing = []
        diverged = []

        model_hash_found = None

        for art_key, spec in artifacts_spec.items():
            expected_sha = spec.get("sha256", "").lower()
            rel_path = spec.get("path", "")
            is_model = "model.safetensors" in rel_path

            # Look up by relative path or key
            found_hash = reported_artifacts.get(rel_path) or reported_artifacts.get(art_key)

            if is_model:
                model_hash_found = found_hash

            if not found_hash:
                missing.append(art_key)
            elif found_hash.lower() == expected_sha:
                matching.append(art_key)
            else:
                diverged.append(art_key)

        model_verified = (
            model_hash_found is not None and
            model_hash_found.lower() == CANONICAL_MODEL_SHA256.lower()
        )

        protocol_ok = (reported_protocol == CANONICAL_PROTOCOL_VERSION)
        fully_sync = (len(missing) == 0 and len(diverged) == 0 and model_verified and protocol_ok)

        reason = None
        if not model_verified:
            reason = f"Model weights checksum mismatch (found: {model_hash_found}, expected: {CANONICAL_MODEL_SHA256})"
        elif not protocol_ok:
            reason = f"Protocol version mismatch (found: {reported_protocol}, expected: {CANONICAL_PROTOCOL_VERSION})"
        elif not fully_sync:
            reason = f"Artifacts out of sync (missing: {missing}, diverged: {diverged})"

        return SyncEvaluationResult(
            is_fully_synchronized=fully_sync,
            is_model_verified=model_verified,
            matching_artifacts=matching,
            missing_artifacts=missing,
            diverged_artifacts=diverged,
            model_sha256_found=model_hash_found,
            protocol_version_found=reported_protocol,
            reason=reason
        )

    def synchronize_and_promote_worker(
        self,
        node_id: str,
        reported_artifacts: Dict[str, str],
        reported_protocol: str,
        registry: DynamicWorkerRegistry
    ) -> Tuple[bool, str]:
        """
        Validates worker synchronization.
        If all required artifacts and model SHA256 match, promotes the worker to READY.
        Otherwise, refuses promotion and sets DEGRADED status.
        """
        eval_result = self.evaluate_worker(reported_artifacts, reported_protocol)

        if not eval_result.is_model_verified:
            registry.quarantine_worker(
                node_id,
                reason=f"Model SHA256 mismatch during sync: {eval_result.model_sha256_found}"
            )
            return False, f"Worker promotion rejected: {eval_result.reason}"

        if not eval_result.is_fully_synchronized:
            registry.update_sync_state(node_id, WorkerState.DEGRADED)
            return False, f"Worker partially out of sync: {eval_result.reason}"

        # Promotion to READY
        success = registry.verify_and_set_ready(
            node_id=node_id,
            reported_model_sha256=eval_result.model_sha256_found or CANONICAL_MODEL_SHA256,
            reported_protocol_version=reported_protocol
        )

        if success:
            logger.info(f"Worker '{node_id}' successfully synchronized and promoted to READY.")
            return True, "Worker verified and promoted to READY"
        return False, "Failed to promote worker in registry"
