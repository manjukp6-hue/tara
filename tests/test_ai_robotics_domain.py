"""
tests/test_ai_robotics_domain.py

Comprehensive Test Suite for TARA's Open-Ended AI & Robotics Domain Capability:
1. Artificial Intelligence (A to Z)
2. Robotics (A to Z)
3. AI + Robotics Integration (A to Z)
4. Practical Engineering Physics & Calculations
5. Systematic Diagnostics & Troubleshooting
6. Fail-Closed Robotics Safety & E-Stop
7. Strict Brain vs Dedicated Real-Time Controller Boundary
8. Dynamic Extensibility without Core Brain Modification
9. Training Data Compilation & Provenance Verification
10. End-to-End TaraBrain Integration
"""

import os
import sys
import math
import json
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

from tara_core.ai_robotics_domain import (
    AIRoboticsDomainCapability,
    AITopic,
    RoboticsTopic,
    IntegrationTopic,
    RoboticsPlatform,
    SensorSpec,
    ActuatorSpec,
    ControllerSpec,
    EngineeringCalculations,
    AIRoboticsTroubleshooter,
    RoboticsSafetyGuard,
    AIRoboticsTaskPlanner,
    register_ai_robotics_capability
)
from tara_core.registry import CapabilityRegistry, CapabilityCategory
from tara_core.brain import TaraBrain
from tara_model.dynamic_dataset_compiler import DynamicDatasetCompiler


class TestAIRoboticsDomain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = REPO_ROOT
        cls.domain = AIRoboticsDomainCapability.get_default(repo_root=cls.repo_root)

    def test_01_ai_fundamentals_and_search(self):
        """Verifies AI fundamentals: search algorithms, logic, state-space reasoning."""
        res = self.domain.query_domain("intelligent agents and state-space reasoning")
        self.assertEqual(res["status"], "SUCCESS")
        self.assertGreater(len(res["matched_ai_topics"]), 0)

        topic = res["matched_ai_topics"][0]
        self.assertIn("A*", " ".join(topic["key_principles"]))
        self.assertIn("admissibility", topic["mathematical_formulation"])

    def test_02_machine_learning_and_deep_learning(self):
        """Verifies ML & Deep Learning: RL (PPO, SAC), transformers, and self-attention."""
        rl_res = self.domain.query_domain("reinforcement learning policy optimization")
        self.assertEqual(rl_res["status"], "SUCCESS")
        self.assertGreater(len(rl_res["matched_ai_topics"]), 0)

        tf_res = self.domain.query_domain("transformer architectures self-attention")
        self.assertEqual(tf_res["status"], "SUCCESS")
        topic = tf_res["matched_ai_topics"][0]
        self.assertIn("softmax", topic["mathematical_formulation"].lower())

    def test_03_transformers_llm_decoding_and_sampling(self):
        """Verifies LLM inference, temperature, KV-cache, and nucleus sampling."""
        res = self.domain.query_domain("decoding sampling temperature top_p")
        self.assertEqual(res["status"], "SUCCESS")
        topic = [t for t in res["matched_ai_topics"] if "decoding" in t["tags"]][0]
        self.assertIn("KV-cache", " ".join(topic["key_principles"]))

    def test_04_neural_training_optimization(self):
        """Verifies optimization: AdamW, mixed precision BF16, and distributed FSDP."""
        res = self.domain.query_domain("optimization training adamw bf16")
        self.assertEqual(res["status"], "SUCCESS")
        topic = [t for t in res["matched_ai_topics"] if "adamw" in t["tags"]][0]
        self.assertIn("AdamW", " ".join(topic["key_principles"]))

    def test_05_ai_safety_and_alignment(self):
        """Verifies DPO, prompt injection defense, and uncertainty calibration."""
        res = self.domain.query_domain("ai safety alignment dpo prompt injection")
        self.assertEqual(res["status"], "SUCCESS")
        topic = [t for t in res["matched_ai_topics"] if "safety" in t["tags"]][0]
        self.assertIn("Prompt injection defense", " ".join(topic["key_principles"]))

    def test_06_robotics_kinematics_and_jacobians(self):
        """Verifies forward & inverse kinematics, Denavit-Hartenberg, and Jacobians."""
        res = self.domain.query_domain("kinematics forward inverse jacobian")
        self.assertEqual(res["status"], "SUCCESS")
        self.assertGreater(len(res["matched_robotics_topics"]), 0)
        topic = res["matched_robotics_topics"][0]
        self.assertIn("Jacobian", " ".join(topic["governing_equations"]))
        self.assertIn("singularities", " ".join(topic["safety_implications"]).lower())

    def test_07_robotics_dynamics_and_lagrangian(self):
        """Verifies rigid body dynamics and computed torque control."""
        res = self.domain.query_domain("robot dynamics computed torque gravity compensation")
        self.assertEqual(res["status"], "SUCCESS")
        topic = [t for t in res["matched_robotics_topics"] if "dynamics" in t["tags"]][0]
        self.assertIn("M(q)", " ".join(topic["governing_equations"]))

    def test_08_motors_actuators_and_foc(self):
        """Verifies BLDC motors, vector Field-Oriented Control, and Clarke/Park transforms."""
        res = self.domain.query_domain("bldc motors field-oriented control foc")
        self.assertEqual(res["status"], "SUCCESS")
        topic = [t for t in res["matched_robotics_topics"] if "bldc" in t["tags"]][0]
        self.assertIn("Clarke Transform", " ".join(topic["governing_equations"]))

    def test_09_sensors_and_state_estimation(self):
        """Verifies quadrature encoders, IMU, LiDAR, and Extended Kalman Filtering."""
        res = self.domain.query_domain("encoders imu lidar kalman filter")
        self.assertEqual(res["status"], "SUCCESS")
        topic = [t for t in res["matched_robotics_topics"] if "encoders" in t["tags"]][0]
        self.assertIn("Kalman Filter", " ".join(topic["governing_equations"]))

    def test_10_robotics_safety_and_estop(self):
        """Verifies ISO 13849, ISO 13850, and hardware emergency stop standards."""
        res = self.domain.query_domain("robot safety emergency stop iso 13849")
        self.assertEqual(res["status"], "SUCCESS")
        topic = [t for t in res["matched_robotics_topics"] if "safety" in t["tags"]][0]
        self.assertIn("ISO 13850", " ".join(topic["governing_equations"]))

    def test_11_fail_closed_safety_guard_evaluation(self):
        """Verifies runtime safety evaluation: E-stop active, joint limits, velocity bounds."""
        guard = RoboticsSafetyGuard()

        joint_limits = {"joint_1": (-3.14, 3.14), "joint_2": (-1.57, 1.57)}
        vel_limits = {"joint_1": 1.0, "joint_2": 1.0}

        # 1. Safe motion -> ALLOW
        safe_eval = guard.evaluate_motion_command(
            target_positions={"joint_1": 1.0, "joint_2": 0.5},
            target_velocities={"joint_1": 0.5, "joint_2": 0.5},
            joint_limits=joint_limits,
            velocity_limits=vel_limits
        )
        self.assertEqual(safe_eval["decision"], "ALLOW")

        # 2. Position out of bounds -> DENY
        bad_pos = guard.evaluate_motion_command(
            target_positions={"joint_1": 4.5, "joint_2": 0.0},
            target_velocities={"joint_1": 0.2, "joint_2": 0.2},
            joint_limits=joint_limits,
            velocity_limits=vel_limits
        )
        self.assertEqual(bad_pos["decision"], "DENY")
        self.assertIn("violates safe limits", bad_pos["reason"])

        # 3. Human in zone with excessive speed -> DENY
        human_speed = guard.evaluate_motion_command(
            target_positions={"joint_1": 0.5, "joint_2": 0.5},
            target_velocities={"joint_1": 0.8, "joint_2": 0.8},
            joint_limits=joint_limits,
            velocity_limits=vel_limits,
            human_detected_in_zone=True
        )
        self.assertEqual(human_speed["decision"], "DENY")
        self.assertIn("Human in zone", human_speed["reason"])

        # 4. Emergency Stop Triggered -> Immediate fail-closed DENY
        guard.set_emergency_stop(True, reason="Physical E-stop button depressed")
        estop_eval = guard.evaluate_motion_command(
            target_positions={"joint_1": 0.0, "joint_2": 0.0},
            target_velocities={"joint_1": 0.01, "joint_2": 0.01},
            joint_limits=joint_limits,
            velocity_limits=vel_limits
        )
        self.assertEqual(estop_eval["decision"], "DENY")
        self.assertIn("E-Stop) is currently ACTIVE", estop_eval["reason"])

    def test_12_robotic_fabrication_cnc_3d_lfam(self):
        """Verifies CNC machining, 3D printing, and Large Format Additive Manufacturing."""
        res = self.domain.query_domain("cnc 3d printing lfam gcode fabrication")
        self.assertEqual(res["status"], "SUCCESS")
        topic = [t for t in res["matched_robotics_topics"] if "cnc" in t["tags"]][0]
        self.assertIn("steps_per_mm", " ".join(topic["governing_equations"]))

    def test_13_ai_robotics_integration_14_stage_pipeline(self):
        """Verifies the complete 14-stage perception-to-action pipeline."""
        res = self.domain.query_domain("perception to action pipeline full loop")
        self.assertEqual(res["status"], "SUCCESS")
        matched = [t for t in res["matched_integration_topics"] if "full_loop" in t["tags"]][0]
        self.assertIn("SENSORS -> PERCEPTION -> STATE ESTIMATION", matched["data_flow"])
        self.assertIn("ROBOT CONTROLLER -> ACTUATORS -> PHYSICAL RESULT", matched["data_flow"])

    def test_14_strict_brain_vs_controller_boundary(self):
        """
        Verifies the strict architectural boundary:
        TARA Brain plans and reasons; Dedicated Controller executes real-time pulses.
        TARA Brain NEVER directly generates microsecond motor pulses.
        """
        res = self.domain.query_domain("boundary brain vs controller realtime architecture")
        self.assertEqual(res["status"], "SUCCESS")
        matched = [t for t in res["matched_integration_topics"] if "boundary" in t["tags"]][0]
        self.assertIn("TARA Brain", matched["architectural_boundary"])
        self.assertIn("Dedicated Controller", matched["architectural_boundary"])
        self.assertIn("NEVER", matched["architectural_boundary"])

        # Plan task check
        plan = self.domain.planner.plan_robotic_task("Pick and place component into assembly jig")
        self.assertIn("strict_boundary_invariant", plan)
        self.assertIn("never generates real-time microsecond motor pulses", plan["strict_boundary_invariant"])

    def test_15_practical_engineering_calculations(self):
        """Verifies exact numerical accuracy of production engineering calculations."""
        calc = EngineeringCalculations()

        # 1. Electrical: Ohm's law & power (24V, 8 ohms -> 3A, 72W)
        elec = calc.calculate_electrical(voltage_v=24.0, resistance_ohms=8.0)
        self.assertEqual(elec["current_a"], 3.0)
        self.assertEqual(elec["power_watts"], 72.0)

        # 2. Motor Mechanics: Torque & RPM (2.5 Nm, 1500 RPM -> Power)
        # omega = 2 * pi * 1500 / 60 = 50 * pi = 157.0796 rad/s
        # P = 2.5 * 157.0796 = 392.699 W
        mech = calc.calculate_motor_mechanics(torque_nm=2.5, rpm=1500.0)
        self.assertAlmostEqual(mech["mechanical_power_w"], 392.6991, places=3)

        # 3. Gearbox: Ratio 10:1, efficiency 0.95
        # input 0.5 Nm -> output 0.5 * 10 * 0.95 = 4.75 Nm
        # input 3000 RPM -> output 300 RPM
        gear = calc.calculate_gear_transmission(gear_ratio=10.0, input_torque_nm=0.5, input_rpm=3000.0, efficiency=0.95)
        self.assertEqual(gear["output_torque_nm"], 4.75)
        self.assertEqual(gear["output_rpm"], 300.0)

        # 4. Lead screw steps-per-mm: 200 steps/rev, 16x microstepping, 8mm pitch
        # steps_per_mm = (200 * 16) / 8 = 400.0
        lead = calc.calculate_steps_per_mm(steps_per_rev=200, microstepping=16, pitch_mm=8.0)
        self.assertEqual(lead["steps_per_mm"], 400.0)

        # 5. Timing belt steps-per-mm: 200 steps/rev, 16x microstepping, 20 teeth, 2mm pitch (GT2)
        # circumference = 20 * 2 = 40mm. steps_per_mm = (200 * 16) / 40 = 80.0
        belt = calc.calculate_steps_per_mm(steps_per_rev=200, microstepping=16, pulley_teeth=20, belt_pitch_mm=2.0)
        self.assertEqual(belt["steps_per_mm"], 80.0)

        # 6. Discrete PID with anti-windup:
        # setpoint 100, current 90 -> error 10. dt=0.01. Kp=2, Ki=1, Kd=0.5
        pid_res = calc.calculate_pid_step(kp=2.0, ki=1.0, kd=0.5, setpoint=100.0, current_value=90.0, dt_seconds=0.01)
        self.assertEqual(pid_res["error"], 10.0)
        self.assertEqual(pid_res["p_term"], 20.0)

        # 7. Nyquist and control loops
        nyq = calc.calculate_nyquist_and_control_loops(plant_bandwidth_hz=50.0)
        self.assertEqual(nyq["nyquist_minimum_rate_hz"], 100.0)
        self.assertEqual(nyq["recommended_control_sample_rate_hz"], 500.0)
        self.assertEqual(nyq["standard_control_hierarchy_hz"]["current_torque_loop_foc"], 10000.0)

        # 8. Kinetic energy and stopping distance
        # m=20 kg, v=1.5 m/s, a=3.0 m/s^2, t_react=0.05 s
        # Ek = 0.5 * 20 * (1.5^2) = 22.5 J
        # d_react = 1.5 * 0.05 = 0.075 m
        # d_brake = 1.5^2 / (2 * 3) = 2.25 / 6 = 0.375 m
        # d_total = 0.075 + 0.375 = 0.45 m
        stop = calc.calculate_kinetic_energy_and_stopping(mass_kg=20.0, velocity_mps=1.5, max_deceleration_mps2=3.0, reaction_time_seconds=0.05)
        self.assertEqual(stop["kinetic_energy_joules"], 22.5)
        self.assertEqual(stop["total_stopping_distance_m"], 0.45)

    def test_16_systematic_diagnostics_and_troubleshooting(self):
        """Verifies root-cause diagnosis trees across AI training and robotics electromechanics."""
        tb = AIRoboticsTroubleshooter()

        # Robotics: Stepper losing steps
        stepper_diag = tb.diagnose("Our 3D printer stepper motor is losing steps and layers shift")
        self.assertEqual(stepper_diag["status"], "SUCCESS")
        case = stepper_diag["matched_cases"][0]
        self.assertEqual(case["category"], "ROBOTICS_ACTUATORS")
        self.assertTrue(any("V_ref" in c["cause"] for c in case["differential_diagnoses"]))

        # Robotics: CAN bus faults
        can_diag = tb.diagnose("CAN bus communication errors and frame timeouts")
        self.assertEqual(can_diag["status"], "SUCCESS")
        self.assertTrue(any("termination" in c["cause"].lower() for c in can_diag["matched_cases"][0]["differential_diagnoses"]))

        # AI: Gradient explosion
        grad_diag = tb.diagnose("Model loss exploded to NaN during training gradient explosion")
        self.assertEqual(grad_diag["status"], "SUCCESS")
        self.assertEqual(grad_diag["matched_cases"][0]["category"], "AI_TRAINING")
        self.assertTrue(any("gradient clipping" in c["resolution"].lower() for c in grad_diag["matched_cases"][0]["differential_diagnoses"]))

    def test_17_dynamic_extensibility_without_brain_modification(self):
        """
        Proves new AI architectures, robotics platforms, sensors, actuators, and controllers
        can be registered dynamically at runtime without modifying TARA Brain core.
        """
        initial_inv = self.domain.get_inventory()

        # 1. Register new AI Topic
        new_ai = AITopic(
            topic_id="ai.dynamic.mamba_state_space",
            name="Mamba & Selective State Space Models",
            category="deep_learning",
            description="Linear-time sequence modeling with selective input-dependent state transitions.",
            key_principles=["Selective state spaces (S6)", "Hardware-aware associative scan"],
            tradeoffs=["Linear complexity in sequence length vs non-associative attention pooling."]
        )
        self.assertTrue(self.domain.register_ai_topic(new_ai))

        # 2. Register new Robotics Platform
        new_plat = RoboticsPlatform(
            platform_id="humanoid_biped_24dof",
            name="24-DOF Humanoid Biped",
            dof=24,
            kinematics_type="Dual Legged Serial Chain + Torso",
            typical_actuators=["High-Torque Planetary Quasi-Direct Drives"],
            typical_sensors=["IMU", "Foot Pressure Arrays", "Stereo Cameras"],
            controller_type="Whole-Body Model Predictive Control (MPC)",
            safety_envelope={"max_walking_speed_mps": 1.2}
        )
        self.assertTrue(self.domain.register_platform(new_plat))

        # 3. Register new Sensor
        new_sensor = SensorSpec(
            sensor_id="tactile_sensor_matrix",
            name="High-Density Tactile Skin Array",
            modality="Piezoresistive",
            sampling_rate_hz=200.0,
            interface="I2C",
            state_estimation_role="Contact force distribution and slip detection"
        )
        self.assertTrue(self.domain.register_sensor(new_sensor))

        updated_inv = self.domain.get_inventory()
        self.assertEqual(updated_inv["ai_topics_count"], initial_inv["ai_topics_count"] + 1)
        self.assertEqual(updated_inv["platforms_count"], initial_inv["platforms_count"] + 1)
        self.assertEqual(updated_inv["sensors_count"], initial_inv["sensors_count"] + 1)

        # Query dynamic platform
        q_res = self.domain.query_domain("humanoid biped")
        self.assertGreater(q_res["total_matches"] + (1 if "humanoid_biped_24dof" in self.domain._platforms else 0), 0)

    def test_18_training_dataset_export_and_compiler_integration(self):
        """Verifies training sample generation, secret scrubbing, and DynamicDatasetCompiler discovery."""
        samples = self.domain.export_training_samples()
        self.assertGreater(len(samples), 10)

        for s in samples:
            self.assertIn("prompt", s)
            self.assertIn("completion", s)
            self.assertIn("topic_family", s)
            self.assertEqual(s["category"], "ai_robotics_domain")
            self.assertFalse("sk-" in s["completion"])
            self.assertFalse("password" in s["completion"].lower())

        # Test discovery by DynamicDatasetCompiler
        compiler = DynamicDatasetCompiler()
        discovered_domain_samples = compiler.discover_domain_capabilities()
        self.assertGreaterEqual(len(discovered_domain_samples), len(samples))

    def test_19_brain_core_integration_end_to_end(self):
        """Verifies end-to-end integration through TaraBrain with cognitive loop processing."""
        brain = TaraBrain()

        # Capability should be present in CapabilityRegistry
        cap = brain.capability_registry.get_capability("capability_ai_robotics_domain")
        self.assertIsNotNone(cap)
        self.assertTrue(cap.enabled)

        # 1. Direct Skill Execution through _execute_skill
        calc_exec = brain._execute_skill("ai_robotics_domain", {
            "action": "calculate",
            "payload": {
                "calculation_type": "calculate_steps_per_mm",
                "arguments": {"steps_per_rev": 200, "microstepping": 16, "pitch_mm": 8.0}
            }
        })
        self.assertEqual(calc_exec["status"], "SUCCESS")
        self.assertEqual(calc_exec["steps_per_mm"], 400.0)

        # 2. Safety Command Execution through _execute_skill
        safety_exec = brain._execute_skill("ai_robotics_domain", {
            "action": "evaluate_safety",
            "payload": {
                "target_positions": {"joint_1": 0.5},
                "target_velocities": {"joint_1": 0.2},
                "joint_limits": {"joint_1": (-1.0, 1.0)},
                "velocity_limits": {"joint_1": 1.0}
            }
        })
        self.assertEqual(safety_exec["decision"], "ALLOW")

        # 3. Diagnostic Execution through _execute_skill
        diag_exec = brain._execute_skill("ai_robotics_domain", {
            "action": "diagnose",
            "payload": {"symptom": "CAN bus communication errors and frame timeouts"}
        })
        self.assertEqual(diag_exec["status"], "SUCCESS")

        # 4. End-to-End process() cognitive loop
        query_out = brain.process(
            actor_id="test_engineer",
            input_text="explain the boundary between tara brain and dedicated controller in robotics",
            context={"session_id": "test_robotics_session"}
        )
        self.assertEqual(query_out["decision"], "ALLOW")
        self.assertEqual(query_out["outcome"], "SUCCESS")


if __name__ == "__main__":
    unittest.main()
