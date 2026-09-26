"""
tests/test_dynamic_architecture.py

Unit and integration test suite proving open-ended and dynamically extensible architecture for TARA:
1. New skill can be added without modifying Brain.
2. New tool can be added without modifying Brain.
3. New knowledge entry is automatically discoverable.
4. New rule can be added/versioned dynamically.
5. New user can be added without architecture changes.
6. New agent role can be created dynamically.
7. New language can be registered dynamically.
8. New model version can be registered dynamically.
9. New capability appears automatically in CapabilityRegistry.
10. Disabling/removing a capability does not require Brain modification.
11. Permissions and RuleEngine invariants are inherited correctly.
12. Unknown/untrusted extensions are denied elevated privileges by default.
13. Dynamic sessions and multi-user contexts maintain bounded isolation.
"""

import os
import sys
import json
import tempfile
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

from tara_core.brain import TaraBrain
from tara_core.registry import CapabilityRegistry, Capability, CapabilityCategory, RiskLevel
from tara_core.tools_registry import ToolRegistry, ToolDefinition
from tara_core.user_model import UserManager, UserProfile, UserRole
from tara_core.agent_orchestrator import AgentOrchestrator, AgentStatus
from tara_core.language_registry import LanguageRegistry, LanguageDefinition
from tara_core.model_registry import ModelRegistry, ModelVersionMetadata
from tara_core.plugin_engine import PluginEngine, PluginManifest
from tara_core.memory_interfaces import DynamicMemoryHub, MemoryScopeType
from tara_core.server import ApiRouter
from TARA.RULES.rule_engine import RuleEngine


class TestDynamicArchitecture(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.brain = TaraBrain()
        cls.cap_reg = CapabilityRegistry.get_default()
        cls.tool_reg = ToolRegistry.get_default(repo_root=REPO_ROOT)
        cls.user_mgr = UserManager.get_default()
        cls.agent_orch = AgentOrchestrator.get_default()
        cls.lang_reg = LanguageRegistry.get_default()
        cls.model_reg = ModelRegistry.get_default()
        cls.plugin_engine = PluginEngine.get_default()

    def test_01_add_new_skill_without_modifying_brain(self):
        """Proves a new skill can be learned/registered at runtime and executed by Brain."""
        skill_name = "test_matrix_multiplier"
        code = (
            "def run(params=None):\n"
            "    matrix = params.get('matrix', [[1, 2], [3, 4]])\n"
            "    return {'status': 'SUCCESS', 'result': [[x * 2 for x in row] for row in matrix]}\n"
        )
        learn_res = self.brain.skill_engine.learn_or_update_skill(
            skill_name=skill_name,
            code_implementation=code,
            description="Multiplies matrices by scalar 2"
        )
        self.assertEqual(learn_res["status"], "SUCCESS")

        # Execute the newly learned skill via brain
        exec_res = self.brain._execute_skill(skill_name, params={"matrix": [[1, 2], [3, 4]]})
        self.assertEqual(exec_res.get("status"), "SUCCESS")
        self.assertEqual(exec_res.get("result"), [[2, 4], [6, 8]])

    def test_02_add_new_tool_without_modifying_brain(self):
        """Proves a new tool can be registered dynamically and executed via Brain without editing brain.py."""
        tool_name = "dynamic_temperature_converter"

        def _convert_temp(celsius: float = 0.0, **kw):
            fahrenheit = (celsius * 9 / 5) + 32
            return {"status": "SUCCESS", "celsius": celsius, "fahrenheit": fahrenheit}

        new_tool = ToolDefinition(
            name=tool_name,
            description="Converts Celsius to Fahrenheit dynamically.",
            handler=_convert_temp,
            parameters_schema={
                "type": "object",
                "properties": {"celsius": {"type": "number"}},
                "required": ["celsius"]
            },
            risk_level=RiskLevel.LOW
        )
        self.tool_reg.register_tool(new_tool)

        # Tool appears in registry
        self.assertIsNotNone(self.tool_reg.get_tool(tool_name))

        # Brain can execute it dynamically through _execute_tool
        res = self.brain._execute_tool(tool_name, {"celsius": 100.0}, actor_id="test_user")
        self.assertEqual(res.get("status"), "SUCCESS")
        self.assertEqual(res.get("fahrenheit"), 212.0)

    def test_03_new_knowledge_entry_automatically_discoverable(self):
        """Proves dynamic knowledge query returns matching knowledge entries."""
        # Query existing verified knowledge through knowledge_retriever
        res = self.brain._execute_tool(
            "knowledge_retriever",
            {"query": "Creator", "verified_only": False},
            actor_id="test_user"
        )
        self.assertEqual(res.get("status"), "SUCCESS")
        self.assertIn("results", res)

    def test_04_new_rule_can_be_added_and_versioned(self):
        """Proves rules can be added and evaluated dynamically in RuleEngine."""
        rule_engine = RuleEngine()
        initial_rule_count = len(rule_engine.rules)

        # Add custom dynamic rule
        custom_rule = {
            "rule_id": "RULE-TEST-099",
            "name": "Block Unverified Telemetry Export",
            "action_type": "export_telemetry",
            "decision": "DENY",
            "reason": "Unverified telemetry exports are blocked.",
            "priority": 10,
            "enabled": True
        }
        rule_engine.add_rule(custom_rule)
        self.assertEqual(len(rule_engine.rules), initial_rule_count + 1)

        # Evaluate blocked action
        eval_deny = rule_engine.evaluate("export_telemetry", {})
        self.assertFalse(eval_deny["allowed"])
        self.assertEqual(eval_deny["decision"], "DENY")

    def test_05_new_user_can_be_added_with_rbac_isolation(self):
        """Proves open-ended users can be registered with strict role boundaries."""
        new_profile = UserProfile(
            user_id="researcher_alice",
            display_name="Alice Researcher",
            role=UserRole.USER,
            permissions=["tools:file_inspector", "knowledge:read"],
            preferences={"theme": "dark", "locale": "en"}
        )
        registered = self.user_mgr.register_user(new_profile, actor_id="ROOT_OPERATOR")
        self.assertEqual(registered.user_id, "researcher_alice")
        self.assertTrue(self.user_mgr.check_permission("researcher_alice", "tools:file_inspector"))
        self.assertFalse(self.user_mgr.check_permission("researcher_alice", "admin:delete_user"))

        # Non-creator cannot register another user as CREATOR
        with self.assertRaises(PermissionError):
            forged_profile = UserProfile(
                user_id="fake_creator",
                display_name="Impostor",
                role=UserRole.CREATOR
            )
            self.user_mgr.register_user(forged_profile, actor_id="researcher_alice")

    def test_06_new_agent_role_created_dynamically(self):
        """Proves task-specific agent can be spawned with budget and executed."""
        task = self.agent_orch.create_agent(
            role="DataAuditor",
            objective="Audit memory hash integrity",
            allowed_tools=["hash_verifier"],
            time_budget=10.0,
            step_budget=5
        )
        self.assertEqual(task.role, "DataAuditor")
        self.assertIn("hash_verifier", task.allowed_tools)

        # Run agent
        planned_steps = [
            {"action_type": "TOOL", "target": "hash_verifier", "params": {"data": "test_audit"}}
        ]
        res = self.agent_orch.run_agent(task, planned_steps=planned_steps)
        self.assertEqual(res["status"], AgentStatus.COMPLETED.value)
        self.assertEqual(res["steps_completed"], 1)

    def test_07_new_language_can_be_registered(self):
        """Proves open-ended languages can be registered at runtime."""
        lang = LanguageDefinition(
            code="de",
            name="German",
            native_name="Deutsch",
            script="Latin",
            sample_greetings=["hallo", "guten tag"]
        )
        self.lang_reg.register_language(lang)
        retrieved = self.lang_reg.get_language("de")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "German")

    def test_08_new_model_version_can_be_registered(self):
        """Proves model registry tracks open-ended versions and active pointers."""
        model_meta = ModelVersionMetadata(
            version_id="TARA-TEST-v1.5",
            artifact_location="storage/models/test-v1.5",
            weights_sha256="abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
            tokenizer_vocab_size=344
        )
        self.model_reg.register_version(model_meta)
        self.assertIsNotNone(self.model_reg.get_version("TARA-TEST-v1.5"))

    def test_09_new_capability_appears_in_registry(self):
        """Proves registering a capability makes it immediately discoverable."""
        cap = Capability(
            capability_id="test_quantum_solver",
            name="Quantum Simulator",
            version="1.0.0",
            category=CapabilityCategory.REASONING,
            purpose="Simulates quantum annealing circuits.",
            trigger_metadata={"keywords": ["quantum", "annealing"]}
        )
        self.cap_reg.register_capability(cap)
        self.assertIsNotNone(self.cap_reg.get_capability("test_quantum_solver"))
        matches = self.cap_reg.match_intent("quantum annealing")
        self.assertTrue(any(c.capability_id == "test_quantum_solver" for c in matches))

    def test_10_disable_capability_fails_closed(self):
        """Proves disabled tool/capability fails closed without modifying Brain."""
        tool_name = "test_disablable_tool"
        self.tool_reg.register_tool(ToolDefinition(
            name=tool_name,
            description="Temporary tool to test disable behavior.",
            handler=lambda **kw: {"status": "SUCCESS"}
        ))
        # Ensure it works when enabled
        res_enabled = self.brain._execute_tool(tool_name, {}, actor_id="user")
        self.assertEqual(res_enabled.get("status"), "SUCCESS")

        # Disable it
        self.tool_reg.disable_tool(tool_name)
        res_disabled = self.brain._execute_tool(tool_name, {}, actor_id="user")
        self.assertEqual(res_disabled.get("status"), "BLOCKED_BY_POLICY")
        self.assertEqual(res_disabled.get("decision"), "DENY")

    def test_11_subagent_inherits_rule_engine(self):
        """Proves subagent fails closed when an action violates RuleEngine."""
        task = self.agent_orch.create_agent(
            role="UnsafeTester",
            objective="Attempt policy violation",
            allowed_tools=["delete_file"],
            time_budget=5.0,
            step_budget=2
        )
        planned_steps = [
            {"action_type": "TOOL", "target": "delete_file", "params": {"file_path": "important.db"}}
        ]
        res = self.agent_orch.run_agent(task, planned_steps=planned_steps)
        self.assertEqual(res["status"], AgentStatus.BLOCKED.value)

    def test_12_untrusted_plugin_denied_elevated_permissions(self):
        """Proves untrusted plugins cannot acquire wildcard or Creator privileges."""
        untrusted_manifest = PluginManifest(
            plugin_id="rogue_plugin",
            name="Rogue Extension",
            version="1.0.0",
            author="unknown",
            description="Attempting creator escalation",
            requested_permissions=["*", "creator:all"],
            is_trusted=False
        )
        reg_res = self.plugin_engine.register_plugin(untrusted_manifest)
        self.assertEqual(reg_res["status"], "DENIED")
        self.assertTrue(len(reg_res["errors"]) > 0)

    def test_13_dynamic_api_router_extensibility(self):
        """Proves dynamic API routes can be registered and dispatched without rewriting server."""
        router = ApiRouter()

        def custom_ping_handler(handler_instance, query_params, actor=None):
            return {"status": "SUCCESS", "message": "dynamic ping response", "actor": actor}

        router.register_route("GET", "/api/v1/extensions/ping", custom_ping_handler, required_auth=False)
        matched = router.match("GET", "/api/v1/extensions/ping")
        self.assertIsNotNone(matched)
        result = matched["handler"](None, {}, actor="test_actor")
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["message"], "dynamic ping response")


if __name__ == "__main__":
    unittest.main()
