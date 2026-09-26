"""
TARA/SECURITY/__init__.py

TARA Source-Code and Runtime Security Hardening Subsystem.
Enforces:
1. Separation of source code, creator authority, recovery secrets, and production models.
2. Cryptographic build and release integrity verification.
3. Fail-closed tamper detection for security-critical components.
4. Encrypted artifact handling and secure runtime execution envelopes.
5. Automated secret scanning for repository hygiene.
"""

from .release_manifest import ReleaseManifestManager, verify_current_release
from .tamper_detector import SourceTamperDetector, IntegrityViolationError
from .source_protector import ProtectedArtifactManager, compile_protected_bundle
from .token_scanner import RepositoryTokenScanner, RepositorySecretScanner

__all__ = [
    "ReleaseManifestManager",
    "verify_current_release",
    "SourceTamperDetector",
    "IntegrityViolationError",
    "ProtectedArtifactManager",
    "compile_protected_bundle",
    "RepositoryTokenScanner",
    "RepositorySecretScanner",
]
