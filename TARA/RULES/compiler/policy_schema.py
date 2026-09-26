"""
TARA/RULES/compiler/policy_schema.py

Data models and schemas for compiled TARA policies.
"""

from enum import Enum, IntEnum
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
import hashlib
import json
import time

class PolicyPriority(IntEnum):
    MANDATORY_LAW_SAFETY = 1
    TARA_SECURITY = 2
    CREATOR_RULE = 3
    USER_REQUEST = 4

class RuleAction(str, Enum):
    ALLOW = "ALLOW"
    DENY = "DENY"
    REQUIRE_CREATOR_CONFIRMATION = "REQUIRE_CREATOR_CONFIRMATION"
    ENFORCE_SETTING = "ENFORCE_SETTING"
    AMBIGUOUS = "AMBIGUOUS"
    REJECTED = "REJECTED"

class RuleCategory(str, Enum):
    MANDATORY_LAW = "MANDATORY_LAW"
    CHILD_SAFETY = "CHILD_SAFETY"
    CONTENT_SAFETY = "CONTENT_SAFETY"
    SECURITY = "SECURITY"
    CREDENTIAL_PROTECTION = "CREDENTIAL_PROTECTION"
    PRIVACY = "PRIVACY"
    DATA_MUTATION = "DATA_MUTATION"
    SYSTEM_INTEGRITY = "SYSTEM_INTEGRITY"
    LANGUAGE_PREFERENCE = "LANGUAGE_PREFERENCE"
    CREATOR_AUTHORITY = "CREATOR_AUTHORITY"
    CREATOR_CUSTOM = "CREATOR_CUSTOM"
    AMBIGUOUS = "AMBIGUOUS"

@dataclass
class Rule:
    rule_id: str
    version: int
    category: str
    meaning: str
    priority: int
    scope: str
    action: str
    conditions: Dict[str, Any]
    status: str  # ACTIVE, AMBIGUOUS, CONFLICTING, REJECTED
    original_text: str
    language: str
    is_mandatory: bool
    created_at: str
    updated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Rule":
        return cls(**data)

@dataclass
class CompiledPolicy:
    policy_version: int
    creator_id: str
    display_name: str
    rules: List[Rule] = field(default_factory=list)
    mandatory_rule_count: int = 0
    creator_rule_count: int = 0
    ambiguous_rule_count: int = 0
    compiled_at: str = ""
    digest_sha256: str = ""
    creator_signature_hex: Optional[str] = None
    signing_key_version: int = 1

    def compute_digest(self) -> str:
        """Computes deterministic SHA256 digest over normalized policy content."""
        canonical_repr = {
            "policy_version": self.policy_version,
            "creator_id": self.creator_id,
            "display_name": self.display_name,
            "rules": [r.to_dict() for r in self.rules],
            "signing_key_version": self.signing_key_version
        }
        encoded = json.dumps(canonical_repr, sort_keys=True, separators=(',', ':')).encode('utf-8')
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "creator_id": self.creator_id,
            "display_name": self.display_name,
            "rules": [r.to_dict() for r in self.rules],
            "mandatory_rule_count": self.mandatory_rule_count,
            "creator_rule_count": self.creator_rule_count,
            "ambiguous_rule_count": self.ambiguous_rule_count,
            "compiled_at": self.compiled_at,
            "digest_sha256": self.digest_sha256 or self.compute_digest(),
            "creator_signature_hex": self.creator_signature_hex,
            "signing_key_version": self.signing_key_version
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompiledPolicy":
        rules = [Rule.from_dict(r) for r in data.get("rules", [])]
        return cls(
            policy_version=data["policy_version"],
            creator_id=data["creator_id"],
            display_name=data["display_name"],
            rules=rules,
            mandatory_rule_count=data.get("mandatory_rule_count", 0),
            creator_rule_count=data.get("creator_rule_count", 0),
            ambiguous_rule_count=data.get("ambiguous_rule_count", 0),
            compiled_at=data.get("compiled_at", ""),
            digest_sha256=data.get("digest_sha256", ""),
            creator_signature_hex=data.get("creator_signature_hex"),
            signing_key_version=data.get("signing_key_version", 1)
        )
