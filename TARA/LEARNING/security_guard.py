"""
TARA/LEARNING/security_guard.py

Security guard and privilege isolation for TARA Online Learning and Research System.
Enforces:
- Creator-only access to LEARN ONLINE, duration settings, LOOP mode, and deep research
- Tool discovery filtering: hides LEARN ONLINE from normal user tool listings
- Rulebook immutability: blocks any automated learning/research action from touching TARA/RULES/
- Non-secret audit logging with sensitive key sanitization
"""

import os
import json
import time
from typing import Dict, List, Optional, Any

from TARA.ACCESS.crypto.ed25519 import Ed25519
from TARA.ACCESS.operator.operator_profile import CANONICAL_CREATOR_ID

FORBIDDEN_AUDIT_KEYS = {
    "private_key", "privkey", "secret", "seed", "token", "password", "passphrase"
}

PROTECTED_RULE_PATHS = [
    "TARA/RULES", "TARA\\RULES", "RULEBOOK.txt", "DEFAULT_SAFE_RULES.txt", "compiled_policy.json"
]

class LearningSecurityGuard:
    """Enforces authorization, tool discovery hiding, and rulebook isolation."""

    def __init__(self, audit_dir: Optional[str] = None):
        if audit_dir is None:
            audit_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "audit"))
        self.audit_dir = audit_dir
        self.audit_file = os.path.join(self.audit_dir, "learning_audit.jsonl")
        os.makedirs(self.audit_dir, exist_ok=True)

    def log_event(self, event_type: str, severity: str = "INFO", details: Optional[Dict[str, Any]] = None) -> None:
        sanitized = {}
        if details:
            for k, v in details.items():
                if any(bad in k.lower() for bad in FORBIDDEN_AUDIT_KEYS):
                    sanitized[k] = "[REDACTED]"
                elif isinstance(v, bytes):
                    sanitized[k] = f"[BYTES_LEN_{len(v)}]"
                else:
                    sanitized[k] = v

        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event": event_type,
            "severity": severity,
            "details": sanitized
        }
        try:
            with open(self.audit_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass

    def verify_creator_authorization(
        self,
        claimed_creator_id: str,
        creator_public_key: bytes,
        challenge_message: bytes,
        signature: bytes
    ) -> bool:
        """
        Verifies that an operation is authorized by ROOT_OPERATOR via Ed25519.
        """
        if claimed_creator_id != CANONICAL_CREATOR_ID:
            self.log_event(
                "LEARNING_ACCESS_DENIED",
                severity="CRITICAL",
                details={"reason": "CREATOR_ID_MISMATCH", "claimed": claimed_creator_id}
            )
            return False

        is_valid = Ed25519.verify(creator_public_key, challenge_message, signature)
        if not is_valid:
            self.log_event(
                "LEARNING_ACCESS_DENIED",
                severity="CRITICAL",
                details={"reason": "INVALID_CREATOR_SIGNATURE", "creator_id": claimed_creator_id}
            )
            return False

        self.log_event(
            "CREATOR_LEARNING_AUTHORIZED",
            severity="INFO",
            details={"creator_id": claimed_creator_id}
        )
        return True

    def filter_available_tools_for_user(self, role: str) -> List[str]:
        """
        Returns list of accessible tools based on role.
        Normal users (role='USER') CANNOT see or discover LEARN ONLINE or deep topic research.
        """
        base_tools = [
            "web_search",
            "read_url_content",
            "query_knowledge_base",
            "calculate",
            "format_document"
        ]

        if role.upper() in ["CREATOR", "ROOT_CREATOR"]:
            # Creator sees full suite including hidden learning controls
            return base_tools + [
                "learn_online_control",
                "start_autonomous_learning",
                "stop_autonomous_learning",
                "deep_topic_research",
                "synthesize_capability"
            ]

        # Normal user sees only base tools, zero learning controls
        return base_tools

    def assert_rulebook_protected(self, target_path: str) -> None:
        """
        Guarantees learning systems, web downloads, or research cannot modify TARA/RULES/.
        """
        norm_path = target_path.replace("\\", "/")
        for protected in ["TARA/RULES", "RULEBOOK.txt", "DEFAULT_SAFE_RULES.txt", "compiled_policy.json"]:
            if protected in norm_path:
                self.log_event(
                    "RULEBOOK_MODIFICATION_BLOCKED",
                    severity="CRITICAL",
                    details={"attempted_path": target_path}
                )
                raise PermissionError(
                    f"Access Denied: Path '{target_path}' is part of TARA Rulebook and is strictly protected from learning modifications."
                )
