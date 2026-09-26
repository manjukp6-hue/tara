"""
python/tara_core/security/jailbreak_detector.py

Multi-Vector AI Compromise and Jailbreak Detection Engine.
Deterministic behavioral and pattern-based heuristics independent of the LLM.
Detects:
1. Rule and guardrail bypass attempts
2. Security controls disablement
3. Creator / root impersonation
4. Privilege escalation
5. Unauthorized protected capability requests
6. Unauthorized filesystem access / path traversal
7. Unauthorized network / metadata endpoint probing
8. Unauthorized tool execution
9. Unauthorized production model write / modification
10. Unauthorized policy / rulebook alteration
11. Cross-user memory access
12. Secret and private key extraction
13. Lateral movement / worker-to-worker control
14. Audit log tampering / evasion
"""

import re
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from TARA.ACCESS.operator.operator_profile import CANONICAL_CREATOR_ID, DEFAULT_DISPLAY_NAME


@dataclass
class ThreatVerdict:
    is_compromised: bool
    threat_category: Optional[str]
    severity: float  # 0.0 to 3.0
    reason: str
    remediation_action: str  # "LOG", "WARN", "RESTRICT", "QUARANTINE"


class JailbreakDetector:
    """
    Independent behavioral heuristics engine.
    Analyzes prompt text, model output, action requests, and environment parameters.
    """

    JAILBREAK_PATTERNS = [
        re.compile(r"ignore\s+(all\s+)?(previous\s+)?(instructions|rules|safeguards|guidelines)", re.IGNORECASE),
        re.compile(r"(disable|bypass|deactivate|turn\s+off)\s+(guardrails?|security|filter|watchdog|audit)", re.IGNORECASE),
        re.compile(r"\b(dan\s+mode|jailbreak\s+mode|developer\s+mode\s+enabled|unrestricted\s+mode)\b", re.IGNORECASE),
        re.compile(r"you\s+are\s+no\s+longer\s+bound\s+by", re.IGNORECASE),
        re.compile(r"override\s+system\s+prompt", re.IGNORECASE),
        re.compile(r"pretend\s+you\s+have\s+no\s+rules", re.IGNORECASE),
    ]

    CREATOR_IMPERSONATION_PATTERNS = [
        re.compile(rf"\b(i\s+am|acting\s+as|login\s+as)\s+({CANONICAL_CREATOR_ID}|{DEFAULT_DISPLAY_NAME}|root|admin)\b", re.IGNORECASE),
        re.compile(r"\b(creator_override|sudo_admin|master_override)\b", re.IGNORECASE),
        re.compile(r"\bclaim_authority\s*=\s*(true|1|creator)\b", re.IGNORECASE),
    ]

    PRIVILEGE_ESCALATION_PATTERNS = [
        re.compile(r"\b(elevate\s+privilege|grant\s+admin|set\s+tier\s+90|set\s+tier\s+100|become\s+creator)\b", re.IGNORECASE),
        re.compile(r"\b(grant_creator_access|grant_all_capabilities|unrestricted_access)\b", re.IGNORECASE),
    ]

    SECRET_EXTRACTION_PATTERNS = [
        re.compile(r"\b(private_key|privkey|ed25519_seed|master_seed|secret_key|api_token|provider_token|bearer_token)\b", re.IGNORECASE),
        re.compile(r"\b(print|reveal|dump|export|display|show)\s+(the\s+)?(private\s+key|credentials?|secrets?|tokens?)\b", re.IGNORECASE),
        re.compile(r"\b(BEGIN\s+(RSA|EC|ED25519|PRIVATE)\s+KEY)\b", re.IGNORECASE),
    ]

    PATH_TRAVERSAL_PATTERNS = [
        re.compile(r"(\.\./|\.\.\\|\.\./\.\./|%2e%2e%2f)", re.IGNORECASE),
        re.compile(r"(TARA/RULES|TARA\\RULES|compiled_policy\.json|DEFAULT_SAFE_RULES\.txt)", re.IGNORECASE),
        re.compile(r"(storage/models/tara/model\.safetensors|model\.safetensors)", re.IGNORECASE),
        re.compile(r"(/etc/passwd|/etc/shadow|C:\\Windows\\system32)", re.IGNORECASE),
    ]

    METADATA_NETWORK_PATTERNS = [
        re.compile(r"169\.254\.169\.254", re.IGNORECASE),  # Cloud instance metadata service
        re.compile(r"metadata\.google\.internal", re.IGNORECASE),
    ]

    DESTRUCTIVE_COMMAND_PATTERNS = [
        re.compile(r"\b(rm\s+-rf|del\s+/[fF]\s+/[sS]|format\s+[a-zA-Z]:|mkfs|shred)\b", re.IGNORECASE),
        re.compile(r"(os\.system|subprocess\.(Popen|call|run))\s*\(.*(rm|del|drop|format|shred)", re.IGNORECASE),
    ]

    # Confusable character map: visually-identical non-Latin codepoints → ASCII Latin equivalents.
    # NFKC does NOT collapse these. Attackers use them to bypass regex keyword patterns.
    _CONFUSABLE_MAP = str.maketrans({
        # Cyrillic homoglyphs → Latin
        '\u0410': 'A', '\u0430': 'a',  # А а
        '\u0412': 'B', '\u0432': 'v',  # В в  (capital В looks like B)
        '\u0421': 'C', '\u0441': 'c',  # С с
        '\u0415': 'E', '\u0435': 'e',  # Е е
        '\u041d': 'H', '\u043d': 'h',  # Н н  (capital Н looks like H)
        '\u0406': 'I', '\u0456': 'i',  # І і  (Ukrainian I)
        '\u0419': 'J',                 # Й (visually close to J)
        '\u041a': 'K', '\u043a': 'k',  # К к
        '\u041c': 'M', '\u043c': 'm',  # М м  (only caps)
        '\u041e': 'O', '\u043e': 'o',  # О о
        '\u0420': 'P', '\u0440': 'p',  # Р р
        '\u0405': 'S', '\u0455': 's',  # Ѕ ѕ  (Cyrillic DZE)
        '\u0422': 'T', '\u0442': 't',  # Т т  (only caps)
        '\u0425': 'X', '\u0445': 'x',  # Х х
        '\u0423': 'Y', '\u0443': 'y',  # У у  (caps looks like Y)
        # Greek homoglyphs → Latin
        '\u0391': 'A', '\u03b1': 'a',  # Α α
        '\u0392': 'B', '\u03b2': 'b',  # Β β  (capital only)
        '\u0395': 'E', '\u03b5': 'e',  # Ε ε
        '\u0397': 'H', '\u03b7': 'h',  # Η η  (capital only)
        '\u0399': 'I', '\u03b9': 'i',  # Ι ι
        '\u039a': 'K', '\u03ba': 'k',  # Κ κ
        '\u039c': 'M',                 # Μ
        '\u039d': 'N',                 # Ν
        '\u039f': 'O', '\u03bf': 'o',  # Ο ο
        '\u03a1': 'P', '\u03c1': 'p',  # Ρ ρ  (capital only)
        '\u03a4': 'T', '\u03c4': 't',  # Τ τ  (capital only)
        '\u03a5': 'Y', '\u03c5': 'y',  # Υ υ  (capital only)
        '\u03a7': 'X', '\u03c7': 'x',  # Χ χ
        '\u0396': 'Z', '\u03b6': 'z',  # Ζ ζ
        # Fullwidth Latin → ASCII
        '\uff21': 'A', '\uff22': 'B', '\uff23': 'C', '\uff24': 'D', '\uff25': 'E',
        '\uff26': 'F', '\uff27': 'G', '\uff28': 'H', '\uff29': 'I', '\uff2a': 'J',
        '\uff2b': 'K', '\uff2c': 'L', '\uff2d': 'M', '\uff2e': 'N', '\uff2f': 'O',
        '\uff30': 'P', '\uff31': 'Q', '\uff32': 'R', '\uff33': 'S', '\uff34': 'T',
        '\uff35': 'U', '\uff36': 'V', '\uff37': 'W', '\uff38': 'X', '\uff39': 'Y',
        '\uff3a': 'Z',
        '\uff41': 'a', '\uff42': 'b', '\uff43': 'c', '\uff44': 'd', '\uff45': 'e',
        '\uff46': 'f', '\uff47': 'g', '\uff48': 'h', '\uff49': 'i', '\uff4a': 'j',
        '\uff4b': 'k', '\uff4c': 'l', '\uff4d': 'm', '\uff4e': 'n', '\uff4f': 'o',
        '\uff50': 'p', '\uff51': 'q', '\uff52': 'r', '\uff53': 's', '\uff54': 't',
        '\uff55': 'u', '\uff56': 'v', '\uff57': 'w', '\uff58': 'x', '\uff59': 'y',
        '\uff5a': 'z',
    })

    @classmethod
    def _normalize_and_decode_candidates(cls, text: str) -> List[str]:
        """
        Normalizes unicode and extracts safe decoded candidates (Base64, URL-encoded, Hex)
        for multi-vector defense-in-depth inspection.
        Includes confusable character transliteration (Cyrillic/Greek/Fullwidth → Latin).
        """
        import unicodedata
        import urllib.parse
        import base64

        candidates: List[str] = []
        if not text:
            return candidates

        # 1. Unicode NFKC normalization and zero-width character stripping
        clean_text = unicodedata.normalize("NFKC", text)
        clean_text = re.sub(r"[\u200b\u200c\u200d\ufeff\u00ad]", "", clean_text)
        candidates.append(clean_text)

        # 2. Confusable character transliteration (Cyrillic/Greek homoglyphs → Latin ASCII)
        # NFKC does NOT collapse Cyrillic а→a or Greek ο→o. Attackers exploit this.
        transliterated = clean_text.translate(cls._CONFUSABLE_MAP)
        if transliterated != clean_text:
            candidates.append(transliterated)

        # 3. URL decoding if '%' is present
        if "%" in clean_text:
            try:
                unquoted = urllib.parse.unquote_plus(clean_text)
                if unquoted != clean_text:
                    candidates.append(unquoted)
            except Exception:
                pass

        # 4. Hex escape decoding (\x41\x42 or \u0041...)
        if r"\x" in clean_text or r"\u" in clean_text:
            try:
                decoded_escapes = clean_text.encode("utf-8").decode("unicode_escape")
                if decoded_escapes != clean_text:
                    candidates.append(decoded_escapes)
            except Exception:
                pass

        # 5. Safe Base64 chunk discovery and decoding
        b64_matches = re.findall(r"[A-Za-z0-9+/]{12,}={0,2}", clean_text)
        for chunk in b64_matches:
            try:
                decoded_bytes = base64.b64decode(chunk, validate=True)
                decoded_str = decoded_bytes.decode("utf-8", errors="ignore")
                if len(decoded_str) >= 4 and any(c.isalpha() for c in decoded_str):
                    candidates.append(decoded_str)
            except Exception:
                pass

        return candidates

    def scan_content(self, text: str, context_metadata: Optional[Dict[str, Any]] = None) -> ThreatVerdict:
        """
        Inspects raw text (prompt, response, tool argument, code payload) for malicious intent.
        Performs multi-layer decoding (Base64, URL, hex, Unicode) for defense-in-depth.
        Note: The Execution Guard / CapabilityGate remains the authoritative authorization boundary.
        """
        if not text:
            return ThreatVerdict(False, None, 0.0, "Empty payload", "LOG")

        candidates = self._normalize_and_decode_candidates(text)

        for candidate_text in candidates:
            # 1. Jailbreak & bypass attempt
            for pattern in self.JAILBREAK_PATTERNS:
                if pattern.search(candidate_text):
                    return ThreatVerdict(
                        is_compromised=True,
                        threat_category="JAILBREAK_ATTEMPT",
                        severity=3.0,
                        reason=f"Detected security rule bypass or jailbreak pattern matching '{pattern.pattern}'",
                        remediation_action="QUARANTINE"
                    )

            # 2. Creator impersonation
            for pattern in self.CREATOR_IMPERSONATION_PATTERNS:
                if pattern.search(candidate_text):
                    # Verify if actual cryptographic creator token is present in context
                    has_valid_creator_token = bool(context_metadata and context_metadata.get("creator_authenticated") is True)
                    if not has_valid_creator_token:
                        return ThreatVerdict(
                            is_compromised=True,
                            threat_category="CREATOR_IMPERSONATION",
                            severity=3.0,
                            reason="Attempted to assert creator identity without cryptographic proof",
                            remediation_action="QUARANTINE"
                        )

            # 3. Privilege escalation
            for pattern in self.PRIVILEGE_ESCALATION_PATTERNS:
                if pattern.search(candidate_text):
                    return ThreatVerdict(
                        is_compromised=True,
                        threat_category="PRIVILEGE_ESCALATION",
                        severity=2.5,
                        reason=f"Detected unauthorized privilege escalation request matching '{pattern.pattern}'",
                        remediation_action="RESTRICT"
                    )

            # 4. Secret / Private key extraction
            for pattern in self.SECRET_EXTRACTION_PATTERNS:
                if pattern.search(candidate_text):
                    return ThreatVerdict(
                        is_compromised=True,
                        threat_category="SECRET_EXTRACTION",
                        severity=3.0,
                        reason="Attempted to extract cryptographic private keys or sensitive credentials",
                        remediation_action="QUARANTINE"
                    )

            # 5. Metadata / unauthorized network probing
            for pattern in self.METADATA_NETWORK_PATTERNS:
                if pattern.search(candidate_text):
                    return ThreatVerdict(
                        is_compromised=True,
                        threat_category="UNAUTHORIZED_NETWORK",
                        severity=3.0,
                        reason="Attempted to probe cloud instance metadata or restricted internal endpoint",
                        remediation_action="QUARANTINE"
                    )

            # 6. Destructive command injection
            for pattern in self.DESTRUCTIVE_COMMAND_PATTERNS:
                if pattern.search(candidate_text):
                    return ThreatVerdict(
                        is_compromised=True,
                        threat_category="DESTRUCTIVE_COMMAND_INJECTION",
                        severity=3.0,
                        reason=f"Detected destructive system command pattern matching '{pattern.pattern}'",
                        remediation_action="QUARANTINE"
                    )

        return ThreatVerdict(False, None, 0.0, "Content verified clean", "LOG")

    def inspect_action_request(
        self,
        action_name: str,
        target_path: Optional[str] = None,
        requester_user_id: Optional[str] = None,
        target_user_id: Optional[str] = None,
        target_worker_id: Optional[str] = None,
        requester_worker_id: Optional[str] = None,
        creator_authenticated: bool = False
    ) -> ThreatVerdict:
        """
        Validates structural boundary conditions on actions.
        """
        # 1. Production Model Immutability Guard
        if action_name in ("write_model", "modify_weights", "delete_model"):
            if target_path and "storage/models/tara" in target_path.replace("\\", "/"):
                return ThreatVerdict(
                    is_compromised=True,
                    threat_category="PRODUCTION_MODEL_WRITE",
                    severity=3.0,
                    reason="Direct in-place modification of production model artifact is strictly forbidden",
                    remediation_action="QUARANTINE"
                )

        # 2. Security Policy / Rulebook Guard
        if action_name in ("modify_rulebook", "write_policy", "disable_security"):
            if not creator_authenticated:
                return ThreatVerdict(
                    is_compromised=True,
                    threat_category="UNAUTHORIZED_POLICY_MODIFICATION",
                    severity=3.0,
                    reason="Modifying security policy or rulebooks requires verified creator authorization",
                    remediation_action="QUARANTINE"
                )

        # 3. Cross-User Memory Access Guard
        if requester_user_id and target_user_id:
            if requester_user_id != CANONICAL_CREATOR_ID and requester_user_id != target_user_id:
                return ThreatVerdict(
                    is_compromised=True,
                    threat_category="CROSS_USER_MEMORY_ACCESS",
                    severity=3.0,
                    reason=f"User '{requester_user_id}' attempted cross-boundary access to memory of '{target_user_id}'",
                    remediation_action="QUARANTINE"
                )

        # 4. Lateral Movement Guard (Worker commanding Worker)
        if requester_worker_id and target_worker_id:
            if requester_worker_id != "control_plane" and requester_worker_id != target_worker_id:
                return ThreatVerdict(
                    is_compromised=True,
                    threat_category="LATERAL_MOVEMENT",
                    severity=3.0,
                    reason=f"Worker '{requester_worker_id}' attempted unauthorized control over worker '{target_worker_id}'",
                    remediation_action="QUARANTINE"
                )

        # 5. Path Traversal Guard
        if target_path:
            norm_path = os.path.normpath(target_path)
            for pattern in self.PATH_TRAVERSAL_PATTERNS:
                if pattern.search(target_path) or pattern.search(norm_path):
                    # Distinguish authorized creator maintenance vs unauthorized escape
                    if not creator_authenticated:
                        return ThreatVerdict(
                            is_compromised=True,
                            threat_category="UNAUTHORIZED_FILESYSTEM_ACCESS",
                            severity=2.5,
                            reason=f"Target path '{target_path}' points to protected system boundaries",
                            remediation_action="RESTRICT"
                        )

        return ThreatVerdict(False, None, 0.0, "Action structurally valid", "LOG")
