"""
TARA/RULES/rule_engine.py

Unified, open-ended RuleEngine for TARA.
Wraps and exposes PolicyCompiler, RulebookManager, and ExecutionGuard
with dynamic policy modification, versioning, rollback, and runtime safety evaluation.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

from TARA.RULES.compiler.policy_schema import CompiledPolicy, Rule, PolicyPriority, RuleAction, RuleCategory
from TARA.RULES.compiler.policy_compiler import PolicyCompiler
from TARA.RULES.engine.rulebook_manager import RulebookManager
from TARA.RULES.engine.execution_guard import ExecutionGuard

logger = logging.getLogger("TARA.RuleEngine")


class RuleEngine:
    """
    Unified interface for TARA policy and rule evaluation.
    Supports dynamic adding of rules, versioning, priority ordering, and rollback.
    """

    def __init__(self, rules_base_dir: Optional[str] = None):
        if rules_base_dir is None:
            rules_base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
        self.base_dir = rules_base_dir
        self.manager = RulebookManager(rules_base_dir=self.base_dir)
        self.policy: Optional[CompiledPolicy] = self._load_or_compile_policy()
        self.guard = ExecutionGuard(active_policy=self.policy)

    def _load_or_compile_policy(self) -> CompiledPolicy:
        pol_path = os.path.join(self.base_dir, "compiled_policy.json")
        if os.path.exists(pol_path):
            try:
                with open(pol_path, "r", encoding="utf-8") as f:
                    return CompiledPolicy.from_dict(json.load(f))
            except Exception as e:
                logger.warning(f"Could not load compiled policy: {e}")

        # Fallback to compiling default rules
        compiler = PolicyCompiler()
        safe_path = os.path.join(self.base_dir, "DEFAULT_SAFE_RULES.txt")
        if os.path.exists(safe_path):
            with open(safe_path, "r", encoding="utf-8") as f:
                return compiler.compile_rulebook_text(f.read())
        return CompiledPolicy(creator_id="ROOT_OPERATOR", policy_version=1)

    @property
    def rules(self) -> List[Rule]:
        return self.policy.rules if self.policy else []

    def add_rule(self, rule_data: Dict[str, Any]) -> Rule:
        """
        Dynamically adds or updates a rule in the active policy.
        """
        if not self.policy:
            self.policy = CompiledPolicy(creator_id="ROOT_OPERATOR", policy_version=1)

        priority_val = rule_data.get("priority", PolicyPriority.USER_REQUEST.value)
        action_val = rule_data.get("decision", RuleAction.DENY.value)
        cat_val = rule_data.get("category", RuleCategory.SECURITY.value)
        now_iso = datetime.now(timezone.utc).isoformat()

        action_type = rule_data.get("action_type")
        conditions = {"action_type": action_type} if action_type else {}
        if "conditions" in rule_data:
            conditions.update(rule_data["conditions"])

        new_rule = Rule(
            rule_id=rule_data.get("rule_id", f"RULE-DYN-{len(self.policy.rules)+1:03d}"),
            version=rule_data.get("version", 1),
            category=cat_val,
            meaning=rule_data.get("reason", rule_data.get("name", "Dynamic safety constraint")),
            priority=int(priority_val),
            scope=rule_data.get("scope", "global"),
            action=action_val,
            conditions=conditions,
            status="ACTIVE",
            original_text=rule_data.get("reason", rule_data.get("name", "")),
            language="en",
            is_mandatory=rule_data.get("is_mandatory", False),
            created_at=now_iso,
            updated_at=now_iso
        )

        self.policy.rules.append(new_rule)
        self.guard.set_policy(self.policy)
        return new_rule

    def evaluate(
        self,
        action_type: str,
        parameters: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates an action against the active policy.
        Returns {'allowed': bool, 'decision': 'ALLOW'|'DENY', 'reason': str}
        """
        if not self.guard or not self.policy:
            return {"allowed": True, "decision": "ALLOW", "reason": "No guard configured"}

        combined_context = dict(context or {})
        if parameters:
            combined_context["parameters"] = parameters

        # Check directly if any rule matches action_type in conditions
        for rule in self.policy.rules:
            if rule.status == "ACTIVE" and rule.conditions.get("action_type") == action_type:
                if rule.action == RuleAction.DENY.value or rule.action == "DENY":
                    return {
                        "allowed": False,
                        "decision": "DENY",
                        "reason": rule.meaning,
                        "matched_rule_id": rule.rule_id
                    }

        eval_res = self.guard.evaluate_action(action_type, context=combined_context)
        decision = eval_res.get("decision", "ALLOW")
        allowed = (decision == "ALLOW" or decision == RuleAction.ALLOW.value)
        return {
            "allowed": allowed,
            "decision": decision,
            "reason": eval_res.get("reason", "Permitted under active policy rules"),
            "matched_rule_id": eval_res.get("matched_rule_id")
        }
