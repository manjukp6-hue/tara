"""
tests/test_dynamic_runtime.py

Comprehensive test suite for TARA Dynamic Runtime Engine (Python layer):
Sandboxes, Agents, Workers, Teams, Naming, Isolation, and Lifecycle.
"""

import os
import shutil
import tempfile
import unittest

from python.tara_core.runtime.dynamic_manager import (
    TaraDynamicManager,
    DynamicIsolatedSandbox,
    DynamicNamingManager,
    DynamicAgent,
    DynamicWorker,
    DynamicTeam,
    DynamicSandboxBroker,
    SandboxConfig,
    EntityType,
    EntityState,
)


class TestDynamicRuntime(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="tara_runtime_test_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_01_dynamic_naming_manager(self):
        naming = DynamicNamingManager()

        # Contextual naming based on domain
        math_name = naming.choose_name(EntityType.SANDBOX, "mathematics")
        self.assertTrue(len(math_name) > 0)

        med_name = naming.choose_name(EntityType.SANDBOX, "medicine-surgery")
        self.assertTrue(len(med_name) > 0)
        self.assertNotEqual(math_name, med_name)

        # ID + Name registration
        entity = naming.register_entity(EntityType.AGENT, "calculus-research")
        self.assertTrue(entity.internal_id.startswith("agt_"))
        self.assertTrue(len(entity.display_name) > 0)

        # Collision prevention on active names
        with self.assertRaises(ValueError):
            naming.register_entity(EntityType.AGENT, "calculus-research", preferred_name=entity.display_name)

        # Renaming changes display name while ID remains immutable
        orig_id = entity.internal_id
        renamed = naming.rename_entity(orig_id, "Promoted-Specialist")
        self.assertEqual(renamed.internal_id, orig_id)
        self.assertEqual(renamed.display_name, "Promoted-Specialist")

        # Retirement tracking
        naming.retire_entity(orig_id, "Test retirement")
        self.assertTrue(any(r["name"] == "Promoted-Specialist" for r in naming.retired_names))

    def test_02_sandbox_host_isolation_and_traversal_blocking(self):
        root = os.path.join(self.test_dir, "sbx_test_01")
        cfg = SandboxConfig(
            sandbox_id="sbx_sec_01",
            display_name="Security-Box",
            sandbox_type="Code",
            root_dir=root,
            execution_timeout_secs=5.0
        )
        sandbox = DynamicIsolatedSandbox(cfg)

        # Safe file write
        safe_path = sandbox.write_file("data/test.json", b'{"key": "value"}')
        self.assertTrue(os.path.exists(safe_path))
        self.assertEqual(sandbox.read_file("data/test.json"), b'{"key": "value"}')

        # Path traversal blocking
        with self.assertRaises(PermissionError):
            sandbox.validate_path_safety("../../sensitive_file.txt")

        # Command execution
        if os.name == "nt":
            cmd = ["cmd.exe", "/C", "echo Sandbox Live Test"]
        else:
            cmd = ["echo", "Sandbox Live Test"]

        res = sandbox.execute(cmd)
        self.assertEqual(res.exit_code, 0)
        self.assertIn("Sandbox Live Test", res.stdout)

        sandbox.cleanup()
        self.assertFalse(os.path.exists(root))

    def test_03_sandbox_broker_cross_isolation(self):
        root_a = os.path.join(self.test_dir, "sbx_a")
        root_b = os.path.join(self.test_dir, "sbx_b")

        sbx_a = DynamicIsolatedSandbox(SandboxConfig("sbx_a", "Box-A", "Code", root_a))
        sbx_b = DynamicIsolatedSandbox(SandboxConfig("sbx_b", "Box-B", "Data", root_b))

        # Transfer payload through broker
        payload = b"Verified Analysis Result"
        res = DynamicSandboxBroker.transfer(sbx_a, sbx_b, payload, "Transfer verified analysis")
        self.assertTrue(res["success"])
        self.assertEqual(res["bytes_transferred"], len(payload))
        self.assertTrue(len(res["sha256"]) == 64)

        # Verify delivered safely into target inbox
        inbox_file = os.path.join(root_b, "inbox", f"{res['transfer_id']}.bin")
        self.assertTrue(os.path.exists(inbox_file))

        # Rejection on payload exceeding limit
        with self.assertRaises(ValueError):
            DynamicSandboxBroker.transfer(sbx_a, sbx_b, b"X" * 200, "Exceed limit", max_size_bytes=100)

        sbx_a.cleanup()
        sbx_b.cleanup()

    def test_04_agent_and_worker_buddy_performance(self):
        agent = DynamicAgent("agt_01", "Active-Dynamic-Agent", "Math", {"tool:math"})
        self.assertEqual(agent.performance.rank_tier, 1)

        # Assign and complete tasks
        agent.assign_task("task_01", "sbx_01")
        self.assertEqual(agent.state_machine.current_state, EntityState.BUSY)

        agent.complete_task(success=True, quality=0.95, security_violation=False)
        self.assertEqual(agent.state_machine.current_state, EntityState.IDLE)
        self.assertEqual(agent.performance.successful_tasks, 1)

        # 10 successful tasks for promotion
        for i in range(2, 12):
            agent.assign_task(f"task_{i}", "sbx_01")
            agent.complete_task(success=True, quality=0.92, security_violation=False)

        promoted = agent.evaluate_promotion()
        self.assertTrue(promoted)
        self.assertEqual(agent.performance.rank_tier, 2)

        # Security violation drops rank immediately
        agent.assign_task("task_bad", "sbx_01")
        agent.complete_task(success=False, quality=0.0, security_violation=True)
        self.assertEqual(agent.state_machine.current_state, EntityState.QUARANTINED)
        self.assertFalse(agent.evaluate_promotion())
        self.assertEqual(agent.performance.rank_tier, 1)

    def test_05_dynamic_team_lifecycle(self):
        team = DynamicTeam("team_01", "Team-Bhishma", "Complex Math DAG", {"agt_01", "agt_02"})
        self.assertEqual(team.state_machine.current_state, EntityState.ACTIVE)
        self.assertEqual(len(team.member_ids), 2)

        freed = team.dissolve("Task completed")
        self.assertEqual(len(freed), 2)
        self.assertEqual(team.state_machine.current_state, EntityState.RETIRED)

    def test_06_tara_dynamic_manager_full_lifecycle(self):
        mgr = TaraDynamicManager(base_dir=self.test_dir)

        # TASK -> CREATE -> NAME -> ASSIGN -> WORK -> VERIFY -> RELEASE -> CLEANUP
        if os.name == "nt":
            cmd = ["cmd.exe", "/C", "echo Full Dynamic Pipeline Execution"]
        else:
            cmd = ["echo", "Full Dynamic Pipeline Execution"]

        res = mgr.execute_dynamic_task("task_99", "simulation", cmd)
        self.assertEqual(res.exit_code, 0)
        self.assertIn("Full Dynamic Pipeline Execution", res.stdout)

        # Verify all transient sandboxes cleaned up on-demand (no permanent clutter)
        self.assertEqual(len(mgr.sandboxes), 0)

    def test_07_sandbox_degraded_refusal(self):
        """Verify that when sandbox is in DEGRADED state, execution is strictly refused."""
        sbx_dir = os.path.join(self.test_dir, "sbx_degraded_test")
        cfg = SandboxConfig(
            sandbox_id="sbx_deg_01",
            display_name="Degraded-Test-Sbx",
            sandbox_type="Code",
            root_dir=sbx_dir
        )
        sbx = DynamicIsolatedSandbox(cfg)
        self.assertEqual(sbx.state_machine.current_state, EntityState.READY)

        # Transition sandbox to DEGRADED
        sbx.state_machine.transition_to(
            EntityState.DEGRADED,
            "Host isolation mechanism could not be enforced",
            "SUPERVISOR"
        )
        self.assertEqual(sbx.state_machine.current_state, EntityState.DEGRADED)

        # Attempting execution must be strictly refused
        cmd = ["cmd.exe", "/C", "echo test"] if os.name == "nt" else ["echo", "test"]
        with self.assertRaises(PermissionError) as ctx:
            sbx.execute(cmd)
        self.assertIn("DEGRADED", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
