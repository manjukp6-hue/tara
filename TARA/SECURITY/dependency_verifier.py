"""
TARA/SECURITY/dependency_verifier.py

Dependency Integrity Verification Engine for TARA Core.
Guarantees:
1. Pinned dependencies in requirements.txt match storage/dependency_manifest.json.
2. Cryptographic checksum verification of requirements.txt.
3. In-environment runtime verification of installed package versions.
4. Tamper detection for unauthorized dependency additions or version drift.
"""

import os
import json
import hashlib
from typing import Dict, Any, List, Tuple

try:
    import importlib.metadata as importlib_metadata
except ImportError:
    import importlib_metadata


def compute_file_sha256(path: str) -> str:
    """Computes streaming SHA256 of a file."""
    if not os.path.isfile(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class DependencyVerifier:
    def __init__(self, repo_root: str = None, manifest_path: str = None):
        if repo_root is None:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        self.repo_root = os.path.abspath(repo_root)

        if manifest_path is None:
            manifest_path = os.path.join(self.repo_root, "storage", "dependency_manifest.json")
        self.manifest_path = os.path.abspath(manifest_path)

        self.requirements_path = os.path.join(self.repo_root, "requirements.txt")

    def parse_requirements(self) -> Dict[str, str]:
        """Parses pinned requirements from requirements.txt."""
        reqs: Dict[str, str] = {}
        if not os.path.isfile(self.requirements_path):
            return reqs

        with open(self.requirements_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "==" in line:
                    pkg, ver = line.split("==", 1)
                    reqs[pkg.strip()] = f"=={ver.strip()}"
                elif ">=" in line:
                    pkg, ver = line.split(">=", 1)
                    reqs[pkg.strip()] = f">={ver.strip()}"
                elif "<=" in line:
                    pkg, ver = line.split("<=", 1)
                    reqs[pkg.strip()] = f"<={ver.strip()}"
                else:
                    reqs[line.strip()] = "*"
        return reqs

    def generate_manifest(self) -> Dict[str, Any]:
        """Generates canonical dependency manifest with checksums and runtime versions."""
        req_sha256 = compute_file_sha256(self.requirements_path)
        reqs = self.parse_requirements()

        installed: Dict[str, str] = {}
        for pkg in reqs.keys():
            try:
                ver = importlib_metadata.version(pkg)
                installed[pkg] = ver
            except Exception:
                installed[pkg] = "NOT_INSTALLED"

        manifest = {
            "version": "1.0",
            "requirements_file": "requirements.txt",
            "requirements_sha256": req_sha256,
            "declared_requirements": reqs,
            "runtime_installed": installed
        }

        os.makedirs(os.path.dirname(self.manifest_path), exist_ok=True)
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return manifest

    def verify_dependencies(self) -> Tuple[bool, List[str]]:
        """
        Verifies requirements.txt against the stored manifest and validates
        installed package availability and versions in the current runtime environment.
        """
        errors: List[str] = []

        if not os.path.isfile(self.requirements_path):
            return False, ["requirements.txt missing"]

        if not os.path.isfile(self.manifest_path):
            return False, ["Dependency manifest missing"]

        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as e:
            return False, [f"Corrupted dependency manifest: {str(e)}"]

        # 1. Verify requirements.txt hash
        current_req_sha = compute_file_sha256(self.requirements_path)
        expected_req_sha = manifest.get("requirements_sha256")
        if current_req_sha != expected_req_sha:
            errors.append(
                f"requirements.txt checksum mismatch: expected {expected_req_sha}, got {current_req_sha}"
            )

        # 2. Verify declared requirements match manifest
        current_reqs = self.parse_requirements()
        declared_reqs = manifest.get("declared_requirements", {})
        for pkg, spec in declared_reqs.items():
            if pkg not in current_reqs:
                errors.append(f"Declared package missing from requirements.txt: {pkg}")
            elif current_reqs[pkg] != spec:
                errors.append(f"Specification mismatch for {pkg}: expected {spec}, got {current_reqs[pkg]}")

        for pkg in current_reqs:
            if pkg not in declared_reqs:
                errors.append(f"Undeclared package found in requirements.txt: {pkg}")

        # 3. Verify installed runtime package versions
        for pkg in declared_reqs.keys():
            try:
                inst_ver = importlib_metadata.version(pkg)
                spec = declared_reqs[pkg]
                if spec.startswith("=="):
                    pinned = spec[2:]
                    if inst_ver != pinned:
                        errors.append(f"Version mismatch for {pkg}: installed {inst_ver}, requires =={pinned}")
            except Exception as e:
                errors.append(f"Package {pkg} is not installed in runtime environment: {str(e)}")

        return (len(errors) == 0, errors)
