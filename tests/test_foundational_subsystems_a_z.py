"""
tests/test_foundational_subsystems_a_z.py

Exhaustive Test Suite for TARA AI Foundational Subsystems:
1. TaraEventBus (thread-safe pub-sub, wildcard routing, subscriber error isolation, bounded history)
2. RoboticsHAL & SafetyInterlock (device registry, spatial boundaries, velocity clamping, watchdog, Creator-authorized E-Stop)
3. WorkingMemoryGovernor (token budgeting, salience preservation, semantic compression of historical context)
4. ExperientialStagingBridge (closed-loop lesson synthesis, secret scrubbing, deduplication, staging queue)
5. End-to-End Integration with TaraBrain & Cognitive Loop
"""

import os
import sys
import json
import time
import shutil
import tempfile
import unittest
from typing import Dict, List, Any

# Ensure project and python root are in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON_DIR = os.path.join(PROJECT_ROOT, "python")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)

from tara_core.event_bus import TaraEventBus
from tara_core.robotics_hal import (
    RoboticsHAL, SafetyInterlock, DeviceType, DeviceInterfaceType
)
from tara_core.working_memory_governor import WorkingMemoryGovernor, WorkingMemorySnapshot
from tara_core.experiential_bridge import ExperientialStagingBridge
from tara_core.experiential_learning import ExperienceEpisode
from tara_core.brain import TaraBrain
from tara_core.registry import CapabilityRegistry


class TestTaraEventBus(unittest.TestCase):
    """Verifies the TaraEventBus pub-sub messaging architecture."""

    def setUp(self):
        TaraEventBus.reset_instance()
        self.bus = TaraEventBus.get_default()

    def tearDown(self):
        TaraEventBus.reset_instance()

    def test_01_exact_and_wildcard_subscription(self):
        exact_events = []
        wildcard_events = []
        catchall_events = []

        self.bus.subscribe("hardware.motion_completed", lambda e: exact_events.append(e))
        self.bus.subscribe("hardware.*", lambda e: wildcard_events.append(e))
        self.bus.subscribe("*", lambda e: catchall_events.append(e))

        self.bus.publish("hardware.motion_completed", {"device_id": "arm_01", "status": "OK"}, source="HAL")
        self.bus.publish("hardware.sensor_read", {"temp": 24.5}, source="HAL")
        self.bus.publish("cognitive.turn_started", {"actor": "creator"}, source="Brain")

        # exact should only have hardware.motion_completed
        self.assertEqual(len(exact_events), 1)
        self.assertEqual(exact_events[0].topic, "hardware.motion_completed")

        # hardware.* should have both hardware events
        self.assertEqual(len(wildcard_events), 2)

        # catchall should have all 3 events
        self.assertEqual(len(catchall_events), 3)

    def test_02_subscriber_error_isolation(self):
        def faulty_sub(event):
            raise RuntimeError("Intentional crash in subscriber")

        good_events = []
        def healthy_sub(event):
            good_events.append(event)

        self.bus.subscribe("test.topic", faulty_sub)
        self.bus.subscribe("test.topic", healthy_sub)

        # Publishing should NOT crash even though one subscriber throws
        delivered = self.bus.publish("test.topic", {"msg": "resilience"})
        self.assertTrue(delivered)
        self.assertEqual(len(good_events), 1)
        self.assertEqual(good_events[0].payload["msg"], "resilience")

    def test_03_history_buffering(self):
        for i in range(15):
            self.bus.publish(f"stream.event_{i}", {"index": i})

        history = self.bus.get_history(limit=5)
        self.assertEqual(len(history), 5)
        self.assertEqual(history[0]["payload"]["index"], 14)


class TestRoboticsHALSafetyInterlock(unittest.TestCase):
    """Verifies Robotics Hardware Abstraction Layer and physical fail-closed interlock."""

    def setUp(self):
        RoboticsHAL.reset_instance()
        self.hal = RoboticsHAL.get_default()

    def tearDown(self):
        RoboticsHAL.reset_instance()

    def test_01_device_registration_and_telemetry(self):
        dev = self.hal.register_device(
            device_id="actuator_gripper_01",
            name="Robotic Gripper",
            device_type=DeviceType.ACTUATOR,
            interface=DeviceInterfaceType.CAN_BUS,
            initial_position=(0.0, 0.0, 0.0),
            metadata={"torque_limit_nm": 5.0}
        )
        self.assertIsNotNone(dev)
        self.assertEqual(dev.device_id, "actuator_gripper_01")

        devices = self.hal.list_devices()
        self.assertTrue(any(d["device_id"] == "actuator_gripper_01" for d in devices))

        telem = self.hal.read_telemetry("actuator_gripper_01")
        self.assertEqual(telem["status"], "SUCCESS")
        self.assertEqual(telem["interlock_status"], "NOMINAL")

    def test_02_spatial_envelope_and_velocity_clamping(self):
        self.hal.register_device(
            device_id="arm_axis_1",
            name="Joint Axis 1",
            device_type=DeviceType.MOTOR,
            interface=DeviceInterfaceType.SERIAL,
            initial_position=(0.0, 0.0, 0.0)
        )

        # In-bounds motion with high velocity (should clamp to max 2.0 m/s)
        res = self.hal.send_motion_command(
            device_id="arm_axis_1",
            target_coords=(0.5, 0.5, 0.5),
            requested_velocity=5.0
        )
        self.assertEqual(res["status"], "SUCCESS")
        self.assertLessEqual(res["velocity_mps"], 2.0)
        self.assertEqual(res["new_position"], [0.5, 0.5, 0.5])

        # Out-of-bounds motion (exceeds envelope limit of +-2.0m)
        res_blocked = self.hal.send_motion_command(
            device_id="arm_axis_1",
            target_coords=(10.0, 0.0, 0.0),
            requested_velocity=0.5
        )
        self.assertEqual(res_blocked["status"], "BLOCKED")
        self.assertIn("envelope", res_blocked["reason"].lower())

    def test_03_emergency_stop_creator_authority(self):
        self.hal.register_device(
            device_id="conveyor_01",
            name="Conveyor Motor",
            device_type=DeviceType.MOTOR,
            interface=DeviceInterfaceType.GPIO,
            initial_position=(0.0, 0.0, 0.0)
        )

        # Trigger Emergency Stop
        self.hal.interlock.trigger_emergency_stop("Obstacle detected in workcell")
        self.assertTrue(self.hal.interlock.e_stop_active)

        # All motion must be strictly blocked
        motion_res = self.hal.send_motion_command("conveyor_01", (0.1, 0.0, 0.0), 0.5)
        self.assertEqual(motion_res["status"], "BLOCKED")
        self.assertIn("emergency stop is active", motion_res["reason"].lower())

        # Unauthorized reset attempt must be rejected fail-closed
        unauth_reset = self.hal.interlock.reset_emergency_stop(actor_id="guest_user")
        self.assertFalse(unauth_reset["success"])
        self.assertTrue(self.hal.interlock.e_stop_active)

        # Creator authorized reset succeeds
        creator_reset = self.hal.interlock.reset_emergency_stop(actor_id="ROOT_OPERATOR")
        self.assertTrue(creator_reset["success"])
        self.assertFalse(self.hal.interlock.e_stop_active)

        # Actuation nominal once more
        nominal_res = self.hal.send_motion_command("conveyor_01", (0.1, 0.0, 0.0), 0.5)
        self.assertEqual(nominal_res["status"], "SUCCESS")


class TestWorkingMemoryGovernor(unittest.TestCase):
    """Verifies dynamic token budgeting, salience preservation, and context compaction."""

    def setUp(self):
        WorkingMemoryGovernor.reset_instance()
        self.gov = WorkingMemoryGovernor(max_uncompacted_turns=3)

    def tearDown(self):
        WorkingMemoryGovernor.reset_instance()

    def test_01_under_threshold_retains_verbatim(self):
        turns = [
            {"user_input": "Turn 1", "response": "Response 1"},
            {"user_input": "Turn 2", "response": "Response 2"}
        ]
        snapshot = self.gov.govern_session(
            session_id="session_01",
            turns=turns,
            active_goal="Test Goal",
            active_slots={"slot_a": "value_a"},
            security_verdicts=["ALLOW"]
        )
        self.assertEqual(len(snapshot.recent_turns), 2)
        self.assertEqual(snapshot.compacted_summary, "")
        self.assertEqual(snapshot.active_goal, "Test Goal")
        self.assertEqual(snapshot.active_slots["slot_a"], "value_a")

    def test_02_over_threshold_compacts_older_turns(self):
        compacted_events = []
        TaraEventBus.get_default().subscribe("memory.context_compacted", lambda e: compacted_events.append(e))

        turns = [
            {"user_input": f"User question number {i}", "response": f"Assistant response {i}"}
            for i in range(6)
        ]
        snapshot = self.gov.govern_session(
            session_id="session_02",
            turns=turns,
            active_goal="Critical Objective",
            active_slots={"target": "config.yaml", "mode": "safe"},
            security_verdicts=["ALLOW", "ALLOW"]
        )

        # With max_uncompacted_turns=3, recent_turns should have exactly 3
        self.assertEqual(len(snapshot.recent_turns), 3)
        # Compacted summary should contain condensed history
        self.assertNotEqual(snapshot.compacted_summary, "")
        self.assertIn("T1:", snapshot.compacted_summary)
        # Salience invariants strictly preserved
        self.assertEqual(snapshot.active_goal, "Critical Objective")
        self.assertEqual(snapshot.active_slots["target"], "config.yaml")
        self.assertEqual(snapshot.security_verdicts, ["ALLOW", "ALLOW"])

        # Event was published
        self.assertGreaterEqual(len(compacted_events), 1)
        self.assertEqual(compacted_events[0].payload["session_id"], "session_02")


class TestExperientialStagingBridge(unittest.TestCase):
    """Verifies automatic staging of verified experiential lessons into training dataset."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        ExperientialStagingBridge.reset_instance()
        self.bridge = ExperientialStagingBridge.get_default(repo_root=self.temp_dir)

    def tearDown(self):
        ExperientialStagingBridge.reset_instance()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_stage_and_deduplicate_episode(self):
        staged_events = []
        TaraEventBus.get_default().subscribe("learning.experience_staged", lambda e: staged_events.append(e))

        episode = ExperienceEpisode(
            episode_id="exp_test_01",
            observation="Calibrate spatial manipulator",
            understanding={"domain": "robotics"},
            plan={"steps": ["verify_envelope", "home_axes"]},
            action={"command": "home"},
            action_result={"homed": True},
            verification={"verified": True},
            explanation="Manipulator was successfully homed to baseline origin.",
            outcome="SUCCESS",
            extracted_lesson="Always verify workspace envelope before issuing high-speed homing pulses."
        )

        record = self.bridge.stage_episode(episode)
        self.assertIsNotNone(record)
        self.assertIn("prompt", record)
        self.assertIn("completion", record)
        self.assertEqual(self.bridge.count_staged(), 1)

        # Staging the exact same episode again must be deduplicated
        dup_record = self.bridge.stage_episode(episode)
        self.assertIsNone(dup_record)
        self.assertEqual(self.bridge.count_staged(), 1)

        # Event was published
        self.assertEqual(len(staged_events), 1)
        self.assertEqual(staged_events[0].payload["episode_id"], "exp_test_01")

    def test_02_secret_scrubbing_safety(self):
        episode_with_secret = ExperienceEpisode(
            episode_id="exp_secret_leak",
            observation="Exporting key private_key=SECRET_TOKEN_VALUE",
            understanding={"secret": True},
            plan={"steps": []},
            action={},
            action_result={},
            verification={"verified": True},
            explanation="Done",
            outcome="SUCCESS",
            extracted_lesson="Private key private_key=SECRET_TOKEN_VALUE should not be stored in plain text."
        )
        record = self.bridge.stage_episode(episode_with_secret)
        # Secret scrubber should skip staging entirely due to detected secret
        self.assertIsNone(record)


class TestFoundationalIntegrationWithBrain(unittest.TestCase):
    """Verifies end-to-end integration of all 4 foundational subsystems inside TaraBrain."""

    @classmethod
    def setUpClass(cls):
        cls.brain = TaraBrain()

    def test_01_subsystems_attached_to_brain(self):
        self.assertIsNotNone(self.brain.event_bus)
        self.assertIsNotNone(self.brain.robotics_hal)
        self.assertIsNotNone(self.brain.working_memory)
        self.assertIsNotNone(self.brain.experiential_bridge)

    def test_02_event_bus_telemetry_on_turn(self):
        turn_events = []
        self.brain.event_bus.subscribe("cognitive.*", lambda e: turn_events.append(e))

        res = self.brain.process(actor_id="user_test_foundation", input_text="What is your identity?")
        self.assertEqual(res["outcome"], "SUCCESS")

        topics = [e.topic for e in turn_events]
        self.assertIn("cognitive.turn_started", topics)
        self.assertIn("cognitive.turn_completed", topics)

    def test_03_robotics_hal_via_skill_dispatcher(self):
        self.brain.robotics_hal.register_device(
            device_id="sensor_lidar_01",
            name="2D Lidar Scanner",
            device_type=DeviceType.SENSOR,
            interface=DeviceInterfaceType.ETHERNET_IP,
            initial_position=(0.0, 0.0, 0.0)
        )

        res = self.brain._execute_skill("robotics_hal", {"device_id": "sensor_lidar_01", "action": "telemetry"})
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["device"]["name"], "2D Lidar Scanner")

    def test_04_agent_task_orchestration_intent(self):
        # Direct execution via brain process
        res = self.brain.process(
            actor_id="user_test_agent",
            input_text="Spawn researcher agent to analyze sensors",
            context={"action_type": "agent_orchestration"}
        )
        self.assertIn(res["decision"], ("ALLOW", "NEEDS_CLARIFICATION"))

    def test_05_experiential_cycle_bridges_to_staging(self):
        exp_res = self.brain.run_experiential_cycle("Calibrate vision sensor matrix")
        self.assertEqual(exp_res["status"], "SUCCESS")
        self.assertIn("extracted_lesson", exp_res)


if __name__ == "__main__":
    unittest.main()
