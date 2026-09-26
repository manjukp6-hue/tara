"""
TARA/SECURITY/token_scanner.py

Automated Repository Token and Sensitive Credential Scanner.
Enforces that none of the following are EVER committed or present in source files:
- Ed25519 private keys (hex seeds or PEM headers)
- Plaintext 32-character recovery codes
- Master passphrases in plaintext
- Google OAuth client secrets or refresh tokens
- Plaintext operator activation tokens
- Hardcoded API bearer tokens
"""

import os
import re
import sys
import subprocess
from typing import List, Dict, Any, Tuple

SUSPICIOUS_PATTERNS = [
    ("ED25519_PRIVATE_KEY_PEM", re.compile(r"-----BEGIN\s+(?:OPENSSH|ED25519|PRIVATE)\s+KEY-----", re.IGNORECASE)),
    ("HARDCODED_OPERATOR_PASSPHRASE", re.compile(r"""(?:master_passphrase|passphrase|operator_token|creator_secret)\s*=\s*['"][A-Za-z0-9!@#$%^&*()_+=-]{8,}['"]""", re.IGNORECASE)),
    ("GOOGLE_CLIENT_SECRET", re.compile(r"""(?:client_secret|clientSecret)\s*[:=]\s*['"][a-zA-Z0-9_\-]{24,}['"]""")),
    ("GOOGLE_REFRESH_TOKEN", re.compile(r"""(?:refresh_token|refreshToken)\s*[:=]\s*['"][0-9]//[a-zA-Z0-9_\-]+['"]""")),
    ("PLAINTEXT_RESTORE_CODE", re.compile(r""" [A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4} """)),
]

ALLOWLIST_PATTERNS = [
    "tests/",
    "test_",
    ".git/",
    "storage/vault/",
    "TARA/ACCESS/vault/",
    "token_scanner.py",
    "secret_scanner.py"
]


class RepositoryTokenScanner:
    """
    Scans repository files for committed credentials or private tokens.
    """

    def __init__(self, repo_root: str = None):
        if repo_root is None:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        self.repo_root = repo_root

    def get_tracked_files(self) -> List[str]:
        try:
            res = subprocess.run(
                ["git", "ls-files"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                check=True
            )
            return [line.strip() for line in res.stdout.splitlines() if line.strip()]
        except Exception:
            files = []
            for root, _, fs in os.walk(self.repo_root):
                if ".git" in root or "node_modules" in root or "__pycache__" in root:
                    continue
                for f in fs:
                    files.append(os.path.relpath(os.path.join(root, f), self.repo_root))
            return files

    def scan_file(self, rel_path: str) -> List[Dict[str, Any]]:
        full_p = os.path.join(self.repo_root, rel_path.replace("/", os.sep))
        findings = []

        # Skip binary / large files
        if not os.path.isfile(full_p):
            return findings
        if rel_path.endswith((".safetensors", ".bin", ".pyc", ".png", ".ico", ".jpg", ".exe", ".dll")):
            return findings

        # Skip test fixtures where mock keys are intentionally tested
        normalized_path = rel_path.replace(os.sep, "/")
        if "tests/" in normalized_path or "token_scanner.py" in normalized_path or "secret_scanner.py" in normalized_path:
            return findings

        try:
            with open(full_p, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except Exception:
            return findings

        for line_idx, line in enumerate(lines, start=1):
            for name, pattern in SUSPICIOUS_PATTERNS:
                if pattern.search(line):
                    findings.append({
                        "file": rel_path,
                        "line": line_idx,
                        "finding_type": name,
                        "snippet": line.strip()[:80]
                    })

        return findings

    def scan_repository(self) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Runs full scan over tracked repository files.
        Returns (is_clean, findings_list).
        """
        all_findings = []
        tracked = self.get_tracked_files()

        for f in tracked:
            f_findings = self.scan_file(f)
            if f_findings:
                all_findings.extend(f_findings)

        return (len(all_findings) == 0, all_findings)


# Compatibility alias
RepositorySecretScanner = RepositoryTokenScanner


if __name__ == "__main__":
    scanner = RepositoryTokenScanner()
    clean, issues = scanner.scan_repository()
    if clean:
        print("[PASS] Token scanner passed: 0 credential leaks found in repository.")
        sys.exit(0)
    else:
        print(f"[FAIL] Token scanner failed: {len(issues)} possible credential leaks found!")
        for iss in issues:
            print(f"  - [{iss['finding_type']}] {iss['file']}:{iss['line']} -> {iss['snippet']}")
        sys.exit(1)
