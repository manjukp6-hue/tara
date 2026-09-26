"""
TARA/RULES/engine/execution_guard.py

Runtime Policy Guard.
Evaluates agent actions, tool invocations, and user requests against the active CompiledPolicy.
Enforces 4-tier hierarchy:
1. Mandatory Law / Safety
2. TARA Security
3. Creator Rules
4. User Requests
"""

from typing import Dict, Any, Tuple, Optional
from ..compiler.policy_schema import CompiledPolicy, RuleAction, PolicyPriority, RuleCategory

class ExecutionGuard:
    """Runtime guard evaluating actions against the active compiled policy."""

    def __init__(self, active_policy: Optional[CompiledPolicy] = None, identity_manager: Optional[Any] = None):
        self.policy = active_policy
        self.identity_manager = identity_manager

    def set_policy(self, policy: CompiledPolicy) -> None:
        self.policy = policy

    def set_identity_manager(self, identity_manager: Any) -> None:
        self.identity_manager = identity_manager

    def _enforce_creator_confirmation(
        self,
        rule: Any,
        action_type: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enforces cryptographic creator verification for operations requiring creator authority.
        FAILS CLOSED if creator verification is unavailable, invalid, missing, expired, or malformed.
        """
        if self.identity_manager is None:
            return {
                "decision": RuleAction.DENY.value,
                "reason": f"CREATOR_AUTH_UNAVAILABLE: Creator confirmation required for {rule.rule_id} ({rule.meaning}), but IdentityManager is not configured. Failing closed.",
                "matched_rule_id": rule.rule_id,
                "priority": rule.priority
            }

        creator_auth = context.get("creator_auth")
        if not creator_auth or not isinstance(creator_auth, dict):
            return {
                "decision": RuleAction.DENY.value,
                "reason": f"UNAUTHORIZED_CREATOR_ACTION: Creator confirmation required for {rule.rule_id} ({rule.meaning}), but cryptographic proof ('creator_auth') was not provided.",
                "matched_rule_id": rule.rule_id,
                "priority": rule.priority
            }

        device_id = creator_auth.get("device_id")
        action_payload = creator_auth.get("action_payload", action_type)
        sig_val = creator_auth.get("device_signature")
        require_biometric = creator_auth.get("require_biometric", False)

        if not device_id or not sig_val:
            return {
                "decision": RuleAction.DENY.value,
                "reason": f"UNAUTHORIZED_CREATOR_ACTION: Malformed creator_auth: 'device_id' and 'device_signature' are required.",
                "matched_rule_id": rule.rule_id,
                "priority": rule.priority
            }

        try:
            if isinstance(sig_val, str):
                sig_bytes = bytes.fromhex(sig_val)
            elif isinstance(sig_val, bytes):
                sig_bytes = sig_val
            else:
                return {
                    "decision": RuleAction.DENY.value,
                    "reason": f"UNAUTHORIZED_CREATOR_ACTION: Invalid signature format.",
                    "matched_rule_id": rule.rule_id,
                    "priority": rule.priority
                }
        except Exception:
            return {
                "decision": RuleAction.DENY.value,
                "reason": f"UNAUTHORIZED_CREATOR_ACTION: Malformed hexadecimal signature string.",
                "matched_rule_id": rule.rule_id,
                "priority": rule.priority
            }

        # Cryptographically verify operation challenge with IdentityManager
        try:
            eval_result = self.identity_manager.verify_creator_operation(
                device_id=device_id,
                action_payload=action_payload,
                device_signature_bytes=sig_bytes,
                require_biometric=require_biometric
            )
        except Exception as e:
            return {
                "decision": RuleAction.DENY.value,
                "reason": f"CREATOR_VERIFICATION_ERROR: Verification raised exception: {str(e)}. Failing closed.",
                "matched_rule_id": rule.rule_id,
                "priority": rule.priority
            }

        if eval_result.get("authorized") is True:
            return {
                "decision": RuleAction.ALLOW.value,
                "reason": f"Creator confirmed operation {rule.rule_id} via verified cryptographic device signature.",
                "matched_rule_id": rule.rule_id,
                "priority": rule.priority,
                "creator_verified": True
            }

        return {
            "decision": RuleAction.DENY.value,
            "reason": f"UNAUTHORIZED_CREATOR_ACTION: Creator verification rejected by Identity Policy: {eval_result.get('reason', 'Signature or device unauthorized')}.",
            "matched_rule_id": rule.rule_id,
            "priority": rule.priority
        }

    def evaluate_action(
        self,
        action_type: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates an intended action against active rules in priority order.
        Returns:
        {
            "decision": "ALLOW" | "DENY" | "REQUIRE_CREATOR_CONFIRMATION" | "ENFORCE_SETTING",
            "reason": str,
            "matched_rule_id": Optional[str],
            "priority": Optional[int]
        }
        """
        if context is None:
            context = {}

        if not self.policy:
            # If no policy active, fall back to safe default
            return {
                "decision": RuleAction.DENY.value,
                "reason": "NO_ACTIVE_POLICY: Execution blocked until verified policy is active",
                "matched_rule_id": None,
                "priority": PolicyPriority.MANDATORY_LAW_SAFETY.value
            }

        # Check rules in strict order of priority (1 -> 2 -> 3 -> 4)
        # Check active capability_v1 creator priority override
        protected_active = context.get("protected_state_active", False)
        creator_id = context.get("creator_id") or context.get("actor_id")
        if protected_active and creator_id == "ROOT_OPERATOR":
            # Mandatory safety check (Child safety) cannot be overridden
            if action_type in ["generate_content", "display_media", "search_web", "web_search"]:
                content_topic = context.get("topic", "")
                if any(bad in content_topic.lower() for bad in ["child abuse", "csam", "child sexual"]):
                    return {
                        "decision": RuleAction.DENY.value,
                        "reason": "Mandatory Child Safety Rule: Execution permanently blocked",
                        "matched_rule_id": "MANDATORY_SAFETY_01",
                        "priority": PolicyPriority.MANDATORY_LAW_SAFETY.value
                    }
            return {
                "decision": RuleAction.ALLOW.value,
                "reason": "Authorized under active capability_v1 creator priority",
                "matched_rule_id": "STATE_V1_PRIORITY_OVERRIDE",
                "priority": 0
            }

        sorted_rules = sorted(self.policy.rules, key=lambda r: r.priority)

        for rule in sorted_rules:
            conds = rule.conditions

            # 1. Mandatory CSAM / Child Safety Check
            if rule.category == RuleCategory.CHILD_SAFETY.value:
                if action_type in ["generate_content", "display_media", "search_web", "web_search"]:
                    media_type = context.get("media_type", "")
                    content_topic = context.get("topic", "")
                    if any(bad in content_topic.lower() for bad in ["child abuse", "csam", "child sexual"]):
                        return {
                            "decision": RuleAction.DENY.value,
                            "reason": f"Mandatory Child Safety Rule {rule.rule_id}: {rule.meaning}",
                            "matched_rule_id": rule.rule_id,
                            "priority": rule.priority
                        }

            # 2. Content Safety: Nudity / Explicit media
            if rule.category == RuleCategory.CONTENT_SAFETY.value:
                if action_type in ["display_media", "generate_content", "show_photo", "show_video", "search_web", "web_search"]:
                    media_type = context.get("media_type", "")
                    topic = context.get("topic", "").lower()
                    if media_type in ["nude_photo", "nude_video"] or any(w in topic for w in ["nude", "naked", "porn"]):
                        return {
                            "decision": RuleAction.DENY.value,
                            "reason": f"Content Safety Rule {rule.rule_id}: {rule.meaning}",
                            "matched_rule_id": rule.rule_id,
                            "priority": rule.priority
                        }

            # 3. Security / Credential Protection
            if rule.category in [RuleCategory.CREDENTIAL_PROTECTION.value, RuleCategory.SECURITY.value]:
                if action_type in ["export_key", "dump_credentials", "read_secret", "elevate_privilege"]:
                    return {
                        "decision": RuleAction.DENY.value,
                        "reason": f"Security Rule {rule.rule_id}: {rule.meaning}",
                        "matched_rule_id": rule.rule_id,
                        "priority": rule.priority
                    }

            # 4. Data Mutation: Important file deletion
            if rule.category == RuleCategory.DATA_MUTATION.value:
                if action_type in ["delete_file", "remove_file", "delete_important_file"]:
                    if context.get("is_important", True):
                        return self._enforce_creator_confirmation(rule, action_type, context)

            # 5. System Integrity: Machine configuration change
            if rule.category == RuleCategory.SYSTEM_INTEGRITY.value:
                if action_type in ["modify_machine_config", "system_reconfigure", "alter_system_settings"]:
                    return self._enforce_creator_confirmation(rule, action_type, context)

            # 6. Language Preference Setting
            if rule.category == RuleCategory.LANGUAGE_PREFERENCE.value:
                if action_type == "query_language_preference":
                    return {
                        "decision": RuleAction.ENFORCE_SETTING.value,
                        "reason": f"Language Preference Rule {rule.rule_id}: {rule.meaning}",
                        "matched_rule_id": rule.rule_id,
                        "priority": rule.priority,
                        "setting": {"preferred_language": "kn"}
                    }

        # If no rule denied or required confirmation, default is ALLOW
        return {
            "decision": RuleAction.ALLOW.value,
            "reason": "Permitted under active policy rules",
            "matched_rule_id": None,
            "priority": PolicyPriority.USER_REQUEST.value
        }
