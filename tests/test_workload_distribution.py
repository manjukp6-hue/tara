"""
tests/test_workload_distribution.py

Comprehensive Test Suite for TARA Dynamic Workload Distribution:
1. One CPU node
2. Multiple CPU nodes
3. GPU detection (GPU presence, model, VRAM)
4. GPU preference (GPU preferred workloads routed to GPU over CPU)
5. No GPU -> distributed CPU fallback (elastic sharding across N CPU nodes)
6. Single CPU fallback (when workload is unsplittable)
7. Dynamic node join (nodes join at runtime without code changes)
8. Dynamic node leave (nodes cleanly removed from active pool)
9. Node failure (mid-flight failure handling)
10. Worker replacement (reassigning work from failed node)
11. Workload redistribution (pending/failed chunks rerun on surviving healthy nodes)
12. Unsplittable workload (atomic workload stays on single node)
13. Small workload remains single-node (e.g. "Hi TARA" never distributed)
14. Large workload distributes (batch/multi-item workload divided across available workers)
15. Mixed CPU/GPU pool (proper segmentation and capability selection)
16. Model compatibility filtering (nodes with mismatched model SHA256 rejected)
17. No fixed worker count (2, 3, 5, 7 dynamic nodes without hardcoded limits)
"""

import os
import sys
import unittest
import uuid
import time
from typing import Dict, List, Any

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
python_dir = os.path.join(REPO_ROOT, "python")
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

from tara_core.auto_connect_sync import (
    EndpointDefinition,
    EndpointHealthStatus,
    EndpointCapability,
    ResourceLimits,
    AutoConnectRouter,
    EndpointHealthProbe
)
from tara_core.workload_distribution import (
    WorkloadType,
    ExecutionMode,
    WorkloadDescriptor,
    WorkloadDetector,
    CapabilityRouter,
    DynamicLoadDistributor,
    ResultAggregator,
    WorkloadDistributionGateway,
    CANONICAL_MODEL_SHA256
)


class TestWorkloadDistribution(unittest.TestCase):

    def setUp(self):
        self.probe = EndpointHealthProbe(timeout_seconds=0.5)
        self.router = AutoConnectRouter(probe=self.probe)
        self.gateway = WorkloadDistributionGateway(router=self.router)

    def _create_cpu_node(
        self,
        node_id: str,
        name: str = "CPU Worker",
        cpu_cores: int = 4,
        ram_mb: int = 4096,
        current_load: float = 0.1,
        queue_depth: int = 0,
        model_sha: str = CANONICAL_MODEL_SHA256,
        health: EndpointHealthStatus = EndpointHealthStatus.HEALTHY
    ) -> EndpointDefinition:
        ep = EndpointDefinition(
            endpoint_id=node_id,
            name=name,
            endpoint_type="LOCAL_PROCESS",
            base_url="",
            capabilities=[EndpointCapability.INFERENCE.value],
            priority=50,
            resource_limits=ResourceLimits(
                cpu_cores=cpu_cores,
                ram_mb=ram_mb,
                gpu_available=False,
                cpu_capacity=float(cpu_cores)
            ),
            current_load=current_load,
            queue_depth=queue_depth,
            model_available=True,
            model_sha256=model_sha,
            health_status=health
        )
        return ep

    def _create_gpu_node(
        self,
        node_id: str,
        name: str = "GPU Worker",
        gpu_model: str = "NVIDIA RTX 4090",
        gpu_vram_mb: int = 24576,
        current_load: float = 0.1,
        queue_depth: int = 0,
        model_sha: str = CANONICAL_MODEL_SHA256,
        health: EndpointHealthStatus = EndpointHealthStatus.HEALTHY
    ) -> EndpointDefinition:
        ep = EndpointDefinition(
            endpoint_id=node_id,
            name=name,
            endpoint_type="LOCAL_PC",
            base_url="http://127.0.0.1:9090",
            capabilities=[EndpointCapability.INFERENCE.value],
            priority=80,
            resource_limits=ResourceLimits(
                cpu_cores=16,
                ram_mb=32768,
                gpu_available=True,
                gpu_model=gpu_model,
                gpu_device_name=gpu_model,
                gpu_vram_mb=gpu_vram_mb
            ),
            current_load=current_load,
            queue_depth=queue_depth,
            model_available=True,
            model_sha256=model_sha,
            health_status=health
        )
        return ep

    def test_01_one_cpu_node(self):
        """Verify routing and execution with exactly one CPU node."""
        node = self._create_cpu_node("node-cpu-1", "Single CPU")
        self.gateway.register_node(node)

        res = self.gateway.process_workload("Hi TARA")
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["execution_mode"], ExecutionMode.SINGLE_CPU.value)
        self.assertEqual(res["nodes_used"], ["node-cpu-1"])
        self.assertEqual(res["shard_count"], 1)

    def test_02_multiple_cpu_nodes_weighted_distribution(self):
        """Verify weighted dynamic load distribution across multiple CPU nodes."""
        # Node 1 is busy (high load)
        node1 = self._create_cpu_node("node-cpu-busy", "Busy CPU", cpu_cores=4, current_load=0.9, queue_depth=5)
        # Node 2 is idle and powerful
        node2 = self._create_cpu_node("node-cpu-idle", "Idle CPU", cpu_cores=16, current_load=0.05, queue_depth=0)

        self.gateway.register_node(node1)
        self.gateway.register_node(node2)

        # Single request should prefer idle/higher-capacity node2
        res = self.gateway.process_workload("Explain quantum entanglement")
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["execution_mode"], ExecutionMode.SINGLE_CPU.value)
        self.assertEqual(res["nodes_used"], ["node-cpu-idle"])

    def test_03_gpu_detection(self):
        """Verify comprehensive GPU telemetry detection."""
        gpu_node = self._create_gpu_node(
            "node-gpu-tesla",
            "Tesla GPU",
            gpu_model="NVIDIA A100-SXM4-80GB",
            gpu_vram_mb=81920
        )
        self.gateway.register_node(gpu_node)

        telemetry = self.gateway.get_cluster_telemetry()
        self.assertEqual(telemetry["gpu_nodes_count"], 1)
        self.assertEqual(telemetry["total_gpu_vram_mb"], 81920)

        node_tel = telemetry["nodes"][0]
        self.assertTrue(node_tel["gpu_available"])
        self.assertEqual(node_tel["gpu_model"], "NVIDIA A100-SXM4-80GB")
        self.assertEqual(node_tel["gpu_vram"], 81920)

    def test_04_gpu_preference(self):
        """Verify GPU-benefiting workloads route to GPU over CPU nodes."""
        cpu_node = self._create_cpu_node("node-cpu-1", "Fast CPU", cpu_cores=32)
        gpu_node = self._create_gpu_node("node-gpu-1", "Fast GPU", gpu_vram_mb=16384)
        self.gateway.register_node(cpu_node)
        self.gateway.register_node(gpu_node)

        # Image generation request
        img_payload = {
            "task_name": "image_generation",
            "prompt": "generate image of high tech robotic laboratory",
            "num_images": 1
        }
        res = self.gateway.process_workload(img_payload)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["execution_mode"], ExecutionMode.SINGLE_GPU.value)
        self.assertEqual(res["nodes_used"], ["node-gpu-1"])
        self.assertFalse(res["fallback_occurred"])

    def test_05_no_gpu_distributed_cpu_fallback(self):
        """Verify that when no GPU exists, large parallelizable workloads fall back to distributed CPU."""
        cpu1 = self._create_cpu_node("node-cpu-1", "CPU 1", cpu_cores=8)
        cpu2 = self._create_cpu_node("node-cpu-2", "CPU 2", cpu_cores=8)
        cpu3 = self._create_cpu_node("node-cpu-3", "CPU 3", cpu_cores=8)
        self.gateway.register_node(cpu1)
        self.gateway.register_node(cpu2)
        self.gateway.register_node(cpu3)

        # Batch image generation requiring GPU, but only CPUs exist
        batch_payload = {
            "task_name": "image_generation",
            "prompt": "generate image",
            "items": ["cat", "dog", "bird", "tree", "mountain", "ocean"],
            "parallelizable": True,
            "is_large": True
        }
        res = self.gateway.process_workload(batch_payload)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["execution_mode"], ExecutionMode.DISTRIBUTED_CPU.value)
        self.assertTrue(res["fallback_occurred"])
        self.assertEqual(len(res["nodes_used"]), 3)
        self.assertEqual(res["shard_count"], 3)
        self.assertEqual(len(res["result"]), 6)

    def test_06_single_cpu_fallback(self):
        """Verify that when no GPU exists and workload is unsplittable, it falls back to best single CPU."""
        cpu1 = self._create_cpu_node("node-cpu-weak", "Weak CPU", cpu_cores=2, current_load=0.8)
        cpu2 = self._create_cpu_node("node-cpu-strong", "Strong CPU", cpu_cores=16, current_load=0.1)
        self.gateway.register_node(cpu1)
        self.gateway.register_node(cpu2)

        unsplittable_gpu_task = {
            "task_name": "image_generation",
            "prompt": "single atomic image render",
            "items": ["single_canvas"],
            "parallelizable": False,
            "is_large": False
        }
        res = self.gateway.process_workload(unsplittable_gpu_task)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["execution_mode"], ExecutionMode.SINGLE_CPU.value)
        self.assertTrue(res["fallback_occurred"])
        self.assertEqual(res["nodes_used"], ["node-cpu-strong"])
        self.assertEqual(res["shard_count"], 1)

    def test_07_dynamic_node_join(self):
        """Verify dynamic node join at runtime without code changes."""
        node1 = self._create_cpu_node("node-cpu-1", "Initial CPU", cpu_cores=4)
        self.gateway.register_node(node1)

        res1 = self.gateway.process_workload({"task_name": "batch_ai_task", "items": [1, 2, 3, 4], "is_large": True})
        self.assertEqual(res1["nodes_used"], ["node-cpu-1"])

        # Dynamically join a new more powerful node at runtime
        node2 = self._create_cpu_node("node-cpu-2", "New Joined Node", cpu_cores=32)
        self.gateway.register_node(node2)

        res2 = self.gateway.process_workload({"task_name": "batch_ai_task", "items": [1, 2, 3, 4], "is_large": True})
        # Both nodes should now participate automatically
        self.assertIn("node-cpu-2", res2["nodes_used"])
        self.assertEqual(len(res2["nodes_used"]), 2)

    def test_08_dynamic_node_leave(self):
        """Verify dynamic node removal from active pool."""
        node1 = self._create_cpu_node("node-1", "Node 1")
        node2 = self._create_cpu_node("node-2", "Node 2")
        self.gateway.register_node(node1)
        self.gateway.register_node(node2)

        self.assertEqual(len(self.gateway.get_all_nodes()), 2)
        existed = self.gateway.deregister_node("node-2")
        self.assertTrue(existed)
        self.assertEqual(len(self.gateway.get_all_nodes()), 1)

        res = self.gateway.process_workload("Test query")
        self.assertEqual(res["nodes_used"], ["node-1"])

    def test_09_node_failure_and_redistribution(self):
        """Verify worker failure mid-flight, partial result preservation, and chunk redistribution."""
        node_good = self._create_cpu_node("node-healthy", "Healthy Node", cpu_cores=8)
        node_faulty = self._create_cpu_node("node-faulty", "Faulty Node", cpu_cores=8)
        self.gateway.register_node(node_good)
        self.gateway.register_node(node_faulty)

        # Custom executor that fails specifically on node-faulty
        def crashing_executor(chunk, node):
            if node.node_id == "node-faulty":
                raise RuntimeError("Kernel panic on faulty compute node!")
            return [f"processed_{item}_by_{node.node_id}" for item in chunk.items]

        workload = {
            "task_name": "batch_ai_task",
            "items": ["doc_A", "doc_B", "doc_C", "doc_D"],
            "parallelizable": True,
            "is_large": True
        }

        res = self.gateway.process_workload(workload, custom_executor=crashing_executor)
        self.assertEqual(res["status"], "SUCCESS")
        # All 4 items should be processed because the failed chunk was redistributed!
        self.assertEqual(len(res["result"]), 4)
        # Verify failover event was recorded
        failovers = res.get("failover_events", [])
        self.assertTrue(any(f.get("event") == "WORKER_FAILED" and f.get("failed_node_id") == "node-faulty" for f in failovers))
        self.assertTrue(any(f.get("event") == "CHUNK_REDISTRIBUTED" for f in failovers))

    def test_10_unsplittable_workload(self):
        """Verify unsplittable workload remains on a single node even with multiple nodes available."""
        for i in range(4):
            self.gateway.register_node(self._create_cpu_node(f"node-{i}", f"Node {i}"))

        payload = {
            "task_name": "sequential_chain_state",
            "input": "Run non-parallelizable state machine",
            "parallelizable": False,
            "is_large": True
        }
        res = self.gateway.process_workload(payload)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["execution_mode"], ExecutionMode.SINGLE_CPU.value)
        self.assertEqual(len(res["nodes_used"]), 1)
        self.assertEqual(res["shard_count"], 1)

    def test_11_small_workload_remains_single_node(self):
        """Verify tiny requests ('Hi TARA') never trigger distributed execution."""
        for i in range(5):
            self.gateway.register_node(self._create_cpu_node(f"node-{i}", f"Node {i}"))

        res = self.gateway.process_workload("Hi TARA")
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["execution_mode"], ExecutionMode.SINGLE_CPU.value)
        self.assertEqual(len(res["nodes_used"]), 1)
        self.assertEqual(res["shard_count"], 1)

    def test_12_large_workload_distributes(self):
        """Verify large multi-item workload distributes proportionally across nodes."""
        for i in range(4):
            self.gateway.register_node(self._create_cpu_node(f"node-{i}", f"Node {i}", cpu_cores=4))

        large_payload = {
            "task_name": "embedding",
            "items": [f"doc_{idx}" for idx in range(20)],
            "parallelizable": True,
            "is_large": True
        }
        res = self.gateway.process_workload(large_payload)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["execution_mode"], ExecutionMode.DISTRIBUTED_CPU.value)
        self.assertEqual(len(res["nodes_used"]), 4)
        self.assertEqual(res["shard_count"], 4)
        self.assertEqual(len(res["result"]), 20)

    def test_13_mixed_cpu_gpu_pool(self):
        """Verify proper segmentation in a mixed CPU & GPU cluster."""
        cpu1 = self._create_cpu_node("cpu-1", "CPU Node")
        gpu1 = self._create_gpu_node("gpu-1", "GPU Node")
        self.gateway.register_node(cpu1)
        self.gateway.register_node(gpu1)

        # Normal chat routes to CPU
        chat_res = self.gateway.process_workload("Hello from user")
        self.assertEqual(chat_res["nodes_used"], ["cpu-1"])

        # Image generation routes to GPU
        img_res = self.gateway.process_workload({"task_name": "image_generation", "prompt": "scenic landscape"})
        self.assertEqual(img_res["nodes_used"], ["gpu-1"])

    def test_14_model_compatibility_filtering(self):
        """Verify node with mismatched or corrupted model SHA256 is filtered out for model inference."""
        # Node with canonical production model
        valid_node = self._create_cpu_node("valid-node", "Valid Node", model_sha=CANONICAL_MODEL_SHA256)
        # Node with invalid / unpromoted / mismatched model
        invalid_node = self._create_cpu_node("invalid-node", "Mismatched Model", model_sha="deadbeef00000000000000000000000000000000000000000000000000000000")

        self.gateway.register_node(valid_node)
        self.gateway.register_node(invalid_node)

        inference_payload = {
            "task_name": "chat",
            "input": "Generate inference output",
            "requires_model": True,
            "target_model_sha256": CANONICAL_MODEL_SHA256
        }
        res = self.gateway.process_workload(inference_payload)
        self.assertEqual(res["status"], "SUCCESS")
        # Must select valid_node and strictly reject invalid_node
        self.assertEqual(res["nodes_used"], ["valid-node"])

    def test_15_no_fixed_worker_count_elasticity(self):
        """Verify architecture dynamically expands across 2, 3, 5, 7 workers without hardcoded limits."""
        workload = {
            "task_name": "batch_ai_task",
            "items": [f"item_{i}" for i in range(14)],
            "parallelizable": True,
            "is_large": True
        }

        for worker_count in [2, 3, 5, 7]:
            # Recreate gateway with exactly worker_count nodes
            gw = WorkloadDistributionGateway()
            for w in range(worker_count):
                gw.register_node(self._create_cpu_node(f"worker-{worker_count}-{w}", f"Worker {w}"))

            res = gw.process_workload(workload)
            self.assertEqual(res["status"], "SUCCESS")
            self.assertEqual(len(res["nodes_used"]), worker_count)
            self.assertEqual(res["shard_count"], worker_count)
            self.assertEqual(len(res["result"]), 14)


    def test_16_cluster_api_integration(self):
        """Verify REST API cluster routing, telemetry, and execution endpoints."""
        from tara_core.server import create_server
        import urllib.request
        import json
        import threading

        class DummyClusterBrain:
            def __init__(self, gw):
                self.auto_sync = type("DummyAutoSync", (), {"workload_gateway": gw})()

            def get_cluster_telemetry(self):
                return self.auto_sync.workload_gateway.get_cluster_telemetry()

            def process_distributed_workload(self, input_data):
                return self.auto_sync.workload_gateway.process_workload(input_data)

        node = self._create_cpu_node("api-test-node", "API Node", cpu_cores=8)
        self.gateway.register_node(node)

        brain = DummyClusterBrain(self.gateway)
        server = create_server("127.0.0.1", 19876, brain=brain, api_keys={"test-token": "admin"})
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        time.sleep(0.1)

        try:
            # 1. GET /api/v1/cluster/telemetry
            req = urllib.request.Request(
                "http://127.0.0.1:19876/api/v1/cluster/telemetry",
                headers={"Authorization": "Bearer test-token"}
            )
            with urllib.request.urlopen(req) as resp:
                self.assertEqual(resp.status, 200)
                data = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(data["status"], "SUCCESS")
                self.assertIn("cluster_telemetry", data)
                self.assertGreaterEqual(data["cluster_telemetry"]["total_nodes"], 1)

            # 2. POST /api/v1/cluster/route
            route_payload = json.dumps({"input": "Hi TARA"}).encode("utf-8")
            req = urllib.request.Request(
                "http://127.0.0.1:19876/api/v1/cluster/route",
                data=route_payload,
                headers={"Authorization": "Bearer test-token", "Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req) as resp:
                self.assertEqual(resp.status, 200)
                data = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(data["status"], "SUCCESS")
                self.assertEqual(data["execution_mode"], ExecutionMode.SINGLE_CPU.value)

            # 3. POST /api/v1/cluster/execute
            exec_payload = json.dumps({
                "task_name": "batch_ai_task",
                "items": ["alpha", "beta"],
                "parallelizable": True
            }).encode("utf-8")
            req = urllib.request.Request(
                "http://127.0.0.1:19876/api/v1/cluster/execute",
                data=exec_payload,
                headers={"Authorization": "Bearer test-token", "Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req) as resp:
                self.assertEqual(resp.status, 200)
                data = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(data["status"], "SUCCESS")
                self.assertEqual(data["execution_result"]["status"], "SUCCESS")
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
