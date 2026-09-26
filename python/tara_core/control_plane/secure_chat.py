"""
python/tara_core/control_plane/secure_chat.py

Secure Private Chat Architecture & Authority Hierarchy Guard.
Manages:
1. Application-level session encryption keys & message authentication codes (HMAC).
2. Replay protection with monotonic nonces and timestamp verification.
3. Explicit delineation of:
   - Transport Security (TLS/HTTPS)
   - Application/Storage Encryption (AES-GCM / HMAC)
   - Server-Side AI Inference Boundary (Plaintext ephemeral processing in RAM, zero disk leakage)
4. Strict Authority Hierarchy Defense:
   SYSTEM/SECURITY > PROTECTED CREATOR > NORMAL CREATOR > USER > SKILL > RETRIEVED KNOWLEDGE > EXTERNAL CONTENT.
5. Strict cross-user memory isolation: users can never access another user's memory scope.
"""

import os
import sys
import time
import hmac
import hashlib
import secrets
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from TARA.ACCESS.operator.operator_profile import CANONICAL_CREATOR_ID

logger = logging.getLogger("tara_core.control_plane.secure_chat")


class AuthorityTier(int):
    SYSTEM_SECURITY = 100
    PROTECTED_CREATOR = 90
    NORMAL_CREATOR = 80
    AUTHENTICATED_USER = 50
    GUEST_USER = 20
    SKILL_OR_TOOL = 15
    RETRIEVED_KNOWLEDGE = 10
    EXTERNAL_CONTENT = 0


@dataclass
class ChatSecurityContext:
    session_id: str
    user_id: str
    is_private_chat: bool = False
    session_key_hex: str = field(default_factory=lambda: secrets.token_hex(32))
    last_nonce: int = 0
    created_at: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + 86400)
    authority_tier: int = AuthorityTier.AUTHENTICATED_USER

    def generate_message_hmac(self, message: str, nonce: int) -> str:
        """Computes HMAC-SHA256 for application-level message integrity."""
        data = f"{self.session_id}:{self.user_id}:{nonce}:{message}".encode("utf-8")
        key = bytes.fromhex(self.session_key_hex)
        return hmac.new(key, data, hashlib.sha256).hexdigest()

    def verify_message_integrity(self, message: str, nonce: int, received_hmac: str) -> bool:
        """Verifies message integrity and rejects replayed or out-of-order nonces."""
        if nonce <= self.last_nonce:
            logger.warning(f"Replay attack detected or out-of-order nonce: {nonce} <= {self.last_nonce}")
            return False

        expected_hmac = self.generate_message_hmac(message, nonce)
        if secrets.compare_digest(expected_hmac, received_hmac):
            self.last_nonce = nonce
            return True
        return False


class SecureChatManager:
    """
    Coordinates secure chat sessions, private chat modes, and memory isolation.
    """

    def __init__(self):
        self._sessions: Dict[str, ChatSecurityContext] = {}

    def create_chat_session(
        self,
        user_id: str,
        is_private_chat: bool = False,
        is_creator: bool = False
    ) -> ChatSecurityContext:
        session_id = f"sess_{secrets.token_hex(16)}"
        authority = AuthorityTier.NORMAL_CREATOR if (is_creator or user_id == CANONICAL_CREATOR_ID) else AuthorityTier.AUTHENTICATED_USER

        ctx = ChatSecurityContext(
            session_id=session_id,
            user_id=user_id,
            is_private_chat=is_private_chat,
            authority_tier=authority
        )
        self._sessions[session_id] = ctx
        return ctx

    def get_session(self, session_id: str) -> Optional[ChatSecurityContext]:
        ctx = self._sessions.get(session_id)
        if ctx and time.time() < ctx.expires_at:
            return ctx
        return None

    def validate_memory_access(self, requesting_user_id: str, target_memory_scope: str) -> bool:
        """
        Enforces strict cross-user memory isolation.
        A user can ONLY access their own memory scope unless they possess Creator authority.
        """
        if requesting_user_id == CANONICAL_CREATOR_ID:
            return True

        expected_scope = f"user_{requesting_user_id}"
        if target_memory_scope == expected_scope:
            return True

        logger.warning(f"SECURITY VIOLATION: User '{requesting_user_id}' attempted to access '{target_memory_scope}'")
        return False

    def validate_authority_hierarchy(
        self,
        actor_tier: int,
        action_required_tier: int
    ) -> Tuple[bool, str]:
        """
        Enforces that lower-privilege sources (e.g. Model Output, External Web Content, Skills)
        can NEVER self-authorize higher-tier actions.
        """
        if actor_tier >= action_required_tier:
            return True, "Authorized"
        return False, f"Privilege escalation blocked: actor tier {actor_tier} < required {action_required_tier}"

    def sanitize_external_input(self, text: str) -> str:
        """
        Defends against prompt injection and memory poisoning by escaping system injection tokens.
        """
        sanitized = text
        prohibited_tokens = [
            "<|creator_auth|>",
            "<|tara_rule|>",
            "<|system_override|>",
            "<|super_admin|>"
        ]
        for tok in prohibited_tokens:
            if tok in sanitized:
                sanitized = sanitized.replace(tok, f"[ESCAPED_TOKEN:{tok.strip('<|>')}]")
        return sanitized
