"""
tests/test_dynamic_engine_system.py

Comprehensive tests for TARA AI's Dynamic Engine System.
Verifies all 12 stages of the Dynamic Engine Lifecycle,
CapabilityRegistry integration, AST security validation,
dynamic routing, health monitoring, circuit breaker isolation,
and REST API endpoints.
"""

import os
import sys
import json
import time
import shutil
import tempfile
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
python_dir = os.path.join(REPO_ROOT, "python")
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

from tara_core.dynamic_engine_system import (
    DynamicEngineSystem,
    EngineRegistry,
    EngineRouter,
    EngineSecurityValidator,
    EngineResourceChecker,
    EngineHealthMonitor,
    EngineManifest,
    EngineResourceRequirements,
    EngineStatus,
    DynamicCallableEngine,
    RiskLevel
)
from tara_core.registry import CapabilityRegistry, CapabilityCategory
from tara_core.auto_connect_sync import AutoConnectSyncEngine
from tara_core.brain import TaraBrain
from tara_core.server import create_server, GLOBAL_API_ROUTER
import http.client
import threading


class TestDynamicEngineSystem(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="tara_engine_test_")
        self.engines_dir = os.path.join(self.temp_dir, "TARA", "ENGINES")
        os.makedirs(self.engines_dir, exist_ok=True)
        self.cap_registry = CapabilityRegistry()
        self.engine_system = DynamicEngineSystem(
            engines_dir=self.engines_dir,
            capability_registry=self.cap_registry
        )

    def tearDown(self):
        try:
            shutil.rmtree(self.temp_dir)
        except Exception:
            pass

    def test_01_dynamic_registration_no_fixed_limit(self):
        """Verify dynamic registration of multiple novel engines with open-ended categories."""
        categories = ["quantum_sim", "bio_genomics", "hyperspectral", "neuro_symbolic", "point_cloud_3d"]
        for i, cat in enumerate(categories):
            engine_id = f"test_engine_{cat}"
            manifest = EngineManifest(
                engine_id=engine_id,
                name=f"{cat.title()} Engine",
                version="1.0.0",
                category=cat,
                supported_tasks=[f"{cat}:compute", f"{cat}:analyze"],
                input_schema={"type": "object", "properties": {"data": {"type": "string"}}},
                output_schema={"type": "object", "properties": {"result": {"type": "string"}}},
                handler=lambda payload, cat=cat: {"status": "SUCCESS", "engine": cat, "processed": payload.get("data", "")}
            )
            success, err = self.engine_system.register_manifest(manifest)
            self.assertTrue(success, f"Failed to register {engine_id}: {err}")

        engines = self.engine_system.list_engines()
        self.assertEqual(len(engines), len(categories))
        for cat in categories:
            self.assertTrue(any(e["category"] == cat for e in engines))

    def test_02_engine_discovery_from_manifest_file(self):
        """Verify dynamic discovery of engine manifests from directory."""
        eng_dir = os.path.join(self.engines_dir, "vision_advanced")
        os.makedirs(eng_dir, exist_ok=True)
        manifest_data = {
            "engine_id": "vision_advanced_01",
            "name": "Advanced Vision Engine",
            "version": "2.1.0",
            "category": "computer_vision",
            "supported_tasks": ["vision:segmentation", "vision:ocr"],
            "input_schema": {"type": "object", "properties": {"image_url": {"type": "string"}}},
            "output_schema": {"type": "object", "properties": {"segments": {"type": "array"}}},
            "resources": {"min_ram_mb": 256, "min_cpu_cores": 1}
        }
        with open(os.path.join(eng_dir, "engine_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest_data, f)

        discovered = self.engine_system.discover_engines()
        self.assertIn("vision_advanced_01", discovered)
        manifest = discovered["vision_advanced_01"]
        self.assertEqual(manifest.category, "computer_vision")
        self.assertIn("vision:segmentation", manifest.supported_tasks)

    def test_03_interface_and_schema_detection(self):
        """Verify parameter schema matching and input validation."""
        def sample_worker(payload: dict) -> dict:
            return {"status": "SUCCESS", "pixels": payload["width"] * payload["height"]}

        manifest = EngineManifest(
            engine_id="geometry_calc_engine",
            name="Geometry Engine",
            version="1.0.0",
            category="geometry",
            supported_tasks=["geometry:area"],
            input_schema={
                "type": "object",
                "properties": {
                    "width": {"type": "number"},
                    "height": {"type": "number"}
                },
                "required": ["width", "height"]
            },
            handler=sample_worker
        )
        self.engine_system.register_manifest(manifest)

        # Valid input execution
        res_ok = self.engine_system.execute("geometry:area", {"width": 10, "height": 20})
        self.assertTrue(res_ok["success"])
        self.assertEqual(res_ok["output"]["pixels"], 200)

        # Invalid input execution (missing required height)
        res_bad = self.engine_system.execute("geometry:area", {"width": 10})
        self.assertFalse(res_bad["success"])
        self.assertIn("Missing required property", res_bad["error"])

    def test_04_runtime_compatibility_and_dependencies(self):
        """Verify engine dependency check fails cleanly when required module is missing."""
        manifest = EngineManifest(
            engine_id="nonexistent_dep_engine",
            name="Dep Engine",
            version="1.0.0",
            category="experimental",
            supported_tasks=["dep:test"],
            dependencies=["nonexistent_python_module_xyz_12345"]
        )
        success, err = self.engine_system.register_manifest(manifest)
        self.assertFalse(success)
        self.assertIn("Missing required dependency", err)

    def test_05_resource_requirements_checking(self):
        """Verify resource requirements are validated against host capability."""
        # Unrealistic RAM requirement (e.g. 1000 Terabytes)
        manifest = EngineManifest(
            engine_id="huge_engine",
            name="Huge Engine",
            version="1.0.0",
            category="extreme_compute",
            supported_tasks=["extreme:run"],
            resources=EngineResourceRequirements(min_ram_mb=1024 * 1024 * 1000)
        )
        success, err = self.engine_system.register_manifest(manifest)
        self.assertFalse(success)
        self.assertIn("Insufficient RAM", err)

    def test_06_security_ast_validation_rejects_malicious_code(self):
        """Verify AST security scanner rejects dangerous operations and illegal privilege escalation."""
        # 1. Malicious os.system call
        malicious_code = """
import os
def run(payload):
    os.system("rm -rf /")
    return {"status": "hacked"}
"""
        val = EngineSecurityValidator()
        passed, err = val.validate_source(malicious_code)
        self.assertFalse(passed)
        self.assertIn("Dangerous call 'system'", err)

        # 2. Forbidden privilege escalation in permissions
        manifest = EngineManifest(
            engine_id="malicious_priv_engine",
            name="Priv Escalation Engine",
            version="1.0.0",
            category="exploit",
            supported_tasks=["exploit:run"],
            required_permissions=["*"]
        )
        success, err = self.engine_system.register_manifest(manifest)
        self.assertFalse(success)
        self.assertIn("Wildcard privilege", err)

    def test_07_capability_registry_integration(self):
        """Verify registered engines are exposed as capabilities in CapabilityRegistry."""
        manifest = EngineManifest(
            engine_id="audio_spectral_engine",
            name="Audio Spectral Engine",
            version="1.0.0",
            category="audio",
            supported_tasks=["audio:fft", "audio:spectrogram"],
            handler=lambda p: {"status": "SUCCESS", "spectrum": [0.1, 0.5, 0.9]}
        )
        self.engine_system.register_manifest(manifest)

        cap = self.cap_registry.get_capability("engine_audio_spectral_engine")
        self.assertIsNotNone(cap)
        self.assertEqual(cap.category, CapabilityCategory.ENGINE)
        self.assertTrue(cap.executable)
        # Execute through capability handler
        res = cap.handler({"sample_rate": 44100})
        self.assertEqual(res["status"], "SUCCESS")

    def test_08_automatic_task_routing(self):
        """Verify routing to best engine based on supported task and lowest error rate."""
        m1 = EngineManifest(
            engine_id="image_proc_v1",
            name="Image V1",
            version="1.0.0",
            category="image",
            supported_tasks=["image:upscale"],
            handler=lambda p: {"status": "SUCCESS", "version": "v1"}
        )
        m2 = EngineManifest(
            engine_id="image_proc_v2",
            name="Image V2",
            version="2.0.0",
            category="image",
            supported_tasks=["image:upscale"],
            handler=lambda p: {"status": "SUCCESS", "version": "v2"}
        )
        self.engine_system.register_manifest(m1)
        self.engine_system.register_manifest(m2)

        # Degrade m1 with failures
        for _ in range(2):
            self.engine_system.health_monitor.record_failure("image_proc_v1", "fail")

        # Router should select m2 due to lower error count
        res = self.engine_system.execute("image:upscale", {"image": "test.png"})
        self.assertTrue(res["success"])
        self.assertEqual(res["engine_id"], "image_proc_v2")

    def test_09_health_monitoring_and_invocation_metrics(self):
        """Verify invocation latencies and success metrics are tracked."""
        manifest = EngineManifest(
            engine_id="metrics_test_engine",
            name="Metrics Engine",
            version="1.0.0",
            category="benchmark",
            supported_tasks=["benchmark:run"],
            handler=lambda p: {"status": "SUCCESS", "value": 42}
        )
        self.engine_system.register_manifest(manifest)

        for _ in range(5):
            self.engine_system.execute("benchmark:run", {})

        metrics = self.engine_system.health_monitor.get_metrics("metrics_test_engine")
        self.assertEqual(metrics.total_invocations, 5)
        self.assertEqual(metrics.successful_invocations, 5)
        self.assertEqual(metrics.failed_invocations, 0)
        self.assertGreater(metrics.avg_latency_ms, 0.0)

    def test_10_circuit_breaker_and_quarantine(self):
        """Verify circuit breaker trips and quarantines engine after consecutive failures."""
        def faulty_worker(payload):
            raise RuntimeError("Engine kernel crash")

        manifest = EngineManifest(
            engine_id="faulty_engine",
            name="Faulty Engine",
            version="1.0.0",
            category="fault_test",
            supported_tasks=["fault:trigger"],
            handler=faulty_worker
        )
        self.engine_system.register_manifest(manifest)

        # Trigger 3 consecutive failures to trip circuit breaker (threshold = 3)
        for _ in range(3):
            res = self.engine_system.execute("fault:trigger", {})
            self.assertFalse(res["success"])

        # Check status is now QUARANTINED
        entry = self.engine_system.registry.get_engine("faulty_engine")
        self.assertEqual(entry.status, EngineStatus.QUARANTINED)

        # Subsequent attempts are rejected immediately by circuit breaker
        res_blocked = self.engine_system.execute("fault:trigger", {})
        self.assertFalse(res_blocked["success"])
        self.assertIn("Circuit breaker open", res_blocked["error"])

    def test_11_engine_hot_swap_and_clean_removal(self):
        """Verify hot-swapping an engine with an updated version and clean unregistration."""
        v1 = EngineManifest(
            engine_id="transcribe_engine",
            name="Speech Engine",
            version="1.0.0",
            category="audio",
            supported_tasks=["audio:transcribe"],
            handler=lambda p: {"text": "v1 text"}
        )
        self.engine_system.register_manifest(v1)
        res1 = self.engine_system.execute("audio:transcribe", {})
        self.assertEqual(res1["output"]["text"], "v1 text")

        # Hot swap to v2
        v2 = EngineManifest(
            engine_id="transcribe_engine",
            name="Speech Engine",
            version="2.0.0",
            category="audio",
            supported_tasks=["audio:transcribe"],
            handler=lambda p: {"text": "v2 enhanced text"}
        )
        self.engine_system.register_manifest(v2)
        res2 = self.engine_system.execute("audio:transcribe", {})
        self.assertEqual(res2["output"]["text"], "v2 enhanced text")

        # Clean removal
        unreg = self.engine_system.unregister_engine("transcribe_engine")
        self.assertTrue(unreg)
        res3 = self.engine_system.execute("audio:transcribe", {})
        self.assertFalse(res3["success"])
        self.assertIn("No registered engine found", res3["error"])

    def test_12_novel_engine_category_without_core_modification(self):
        """Verify a completely novel category (e.g. quantum_teleportation_sim) runs seamlessly."""
        manifest = EngineManifest(
            engine_id="quantum_teleport_engine",
            name="Quantum Teleportation Engine",
            version="1.0.0",
            category="quantum_teleportation_sim",
            supported_tasks=["quantum:entangle", "quantum:teleport"],
            handler=lambda p: {"fidelity": 0.9998, "state": "|psi> = (|00> + |11>)/sqrt(2)"}
        )
        success, _ = self.engine_system.register_manifest(manifest)
        self.assertTrue(success)

        res = self.engine_system.execute("quantum:teleport", {"qubits": 2})
        self.assertTrue(res["success"])
        self.assertEqual(res["output"]["fidelity"], 0.9998)

    def test_13_brain_integration_and_process_dispatch(self):
        """Verify TaraBrain interacts with DynamicEngineSystem and brain.process handles EXECUTE_ENGINE."""
        brain = TaraBrain()
        # Register a dynamic 3D rendering engine directly
        m3d = EngineManifest(
            engine_id="engine_mesh_render_3d",
            name="3D Mesh Renderer",
            version="1.0.0",
            category="3d_graphics",
            supported_tasks=["3d:render_mesh"],
            handler=lambda p: {"status": "SUCCESS", "polygons": p.get("count", 1000), "format": "gltf"}
        )
        brain.engine_system.register_manifest(m3d)

        # 1. Direct brain.execute_engine invocation
        res = brain.execute_engine("3d:render_mesh", {"count": 5000})
        self.assertTrue(res["success"])
        self.assertEqual(res["output"]["polygons"], 5000)

        # 2. brain.process dispatch with EXECUTE_ENGINE intent
        context = {
            "session_id": "test_engine_session",
            "action_type": "engine_engine_mesh_render_3d"
        }
        res_proc = brain.process(
            actor_id="test_admin",
            input_text="execute engine 3d:render_mesh",
            context={
                **context,
                "intent_override": {
                    "intent": "EXECUTE_ENGINE",
                    "task_type": "3d:render_mesh",
                    "engine_id": "engine_mesh_render_3d",
                    "payload": {"count": 8000}
                }
            }
        )
        # Verify brain executed engine and outputted response
        self.assertIsNotNone(res_proc)
        self.assertIn("ENGINE-RESULT", res_proc.get("context_tags", []))

    def test_14_server_rest_api_endpoints(self):
        """Verify REST API routes for Dynamic Engine System on TaraServer."""
        brain = TaraBrain()
        # Register an engine
        manifest = EngineManifest(
            engine_id="api_test_engine",
            name="API Test Engine",
            version="1.0.0",
            category="api_test",
            supported_tasks=["api:ping"],
            handler=lambda p: {"pong": True, "received": p.get("data")}
        )
        brain.engine_system.register_manifest(manifest)

        server = create_server(host="127.0.0.1", port=0, brain=brain)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        port = server.server_address[1]

        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            headers = {
                "Authorization": "Bearer tara_default_test_token",
                "Content-Type": "application/json"
            }

            # 1. GET /api/v1/engines/list
            conn.request("GET", "/api/v1/engines/list", headers=headers)
            resp = conn.getresponse()
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode())
            self.assertEqual(data["status"], "SUCCESS")
            self.assertTrue(any(e["engine_id"] == "api_test_engine" for e in data["engines"]))

            # 2. POST /api/v1/engines/execute
            payload = json.dumps({
                "task_type": "api:ping",
                "payload": {"data": "hello_server"}
            })
            conn.request("POST", "/api/v1/engines/execute", body=payload, headers=headers)
            resp = conn.getresponse()
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode())
            self.assertEqual(data["status"], "SUCCESS")
            self.assertTrue(data["result"]["output"]["pong"])
            self.assertEqual(data["result"]["output"]["received"], "hello_server")

            # 3. GET /api/v1/engines/health
            conn.request("GET", "/api/v1/engines/health", headers=headers)
            resp = conn.getresponse()
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode())
            self.assertEqual(data["status"], "SUCCESS")
            self.assertIn("api_test_engine", data["health"])

            # 4. POST /api/v1/engines/quarantine
            q_payload = json.dumps({"engine_id": "api_test_engine", "reason": "Test Quarantine"})
            conn.request("POST", "/api/v1/engines/quarantine", body=q_payload, headers=headers)
            resp = conn.getresponse()
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode())
            self.assertEqual(data["status"], "SUCCESS")
            self.assertTrue(data["quarantined"])

            conn.close()
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
