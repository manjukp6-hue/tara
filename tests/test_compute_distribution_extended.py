"""
tests/test_compute_distribution_extended.py

Extended Test Suite for TARA Elastic Compute Architecture (Workstream B):
1. Model-capability matching: RAM and VRAM requirements dynamically scale with model parameter count.
2. Large expanded model routing: Automatically routes larger models to nodes with adequate capacity.
3. GPU dynamic arrival: New GPU node becomes available; routing automatically switches to GPU.
4. GPU dynamic departure: Active GPU node is deregistered or fails; routing seamlessly falls back to distributed CPU.
5. CPU/GPU hybrid pipeline execution: Mixed cluster executes CPU preprocessing -> GPU inference -> CPU postprocessing.
6. Elastic multi-worker scaling: Dynamic partitioning across arbitrary worker counts (3, 5, 8, 12).
"""

import os
import sys
import unittest
import time

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
    WorkloadDistributionGateway,
    CANONICAL_MODEL_SHA256,
    CANONICAL_PARAM_COUNT
)


class TestComputeDistributionExtended(unittest.TestCase):

    def setUp(self):
        self.probe = EndpointHealthProbe(timeout_seconds=0.5)
        self.router = AutoConnectRouter(probe=self.probe)
        self.gateway = WorkloadDistributionGateway(router=self.router)

    def _create_node(
        self,
        node_id: str,
        name: str,
        ram_mb: int = 4096,
        gpu_available: bool = False,
        gpu_vram_mb: int = 0,
        gpu_model: str = "",
        model_sha: str = CANONICAL_MODEL_SHA256
    ) -> EndpointDefinition:
        return EndpointDefinition(
            endpoint_id=node_id,
            name=name,
            endpoint_type="LOCAL_PROCESS",
            base_url="",
            capabilities=[EndpointCapability.INFERENCE.value],
            priority=50,
            resource_limits=ResourceLimits(
                cpu_cores=4,
                ram_mb=ram_mb,
                gpu_available=gpu_available,
                gpu_model=gpu_model,
                gpu_vram_mb=gpu_vram_mb,
                cpu_capacity=4.0
            ),
            current_load=0.1,
            queue_depth=0,
            model_available=True,
            model_sha256=model_sha,
            health_status=EndpointHealthStatus.HEALTHY
        )

    def test_01_ram_vram_scaling_with_model_parameter_count(self):
        """Test that WorkloadDescriptor dynamically computes required RAM and VRAM based on model parameter count."""
        detector = WorkloadDetector()

        # 1. Baseline model (118,080 parameters)
        wl_base = detector.detect(
            input_data={"input": "Summarize this batch", "model_param_count": 118080},
            task_name="batch_ai_task"
        )
        self.assertEqual(wl_base.model_param_count, 118080)
        # Baseline model fits within standard 512 MB allocation
        self.assertGreaterEqual(wl_base.min_ram_mb, 512)

        # 2. Large future expanded model (e.g. 100 Million parameters)
        # 100M * 4 bytes * 1.5 overhead ~= 600 MB
        wl_expanded = detector.detect(
            input_data={"input": "Summarize this batch", "model_param_count": 100_000_000},
            task_name="batch_ai_task"
        )
        self.assertEqual(wl_expanded.model_param_count, 100_000_000)
        self.assertGreaterEqual(wl_expanded.min_ram_mb, 570)

        # 3. 1 Billion parameter model requiring ~5.7 GB RAM
        wl_1b = detector.detect(
            input_data={"input": "Analyze full corpus", "model_param_count": 1_000_000_000},
            task_name="batch_ai_task"
        )
        self.assertGreaterEqual(wl_1b.min_ram_mb, 5000)

    def test_02_model_capability_matching_filters_inadequate_nodes(self):
        """Nodes with insufficient RAM for an expanded model must be rejected."""
        small_node = self._create_node("small-node", "Small 1GB Node", ram_mb=1024)
        large_node = self._create_node("large-node", "Large 16GB Node", ram_mb=16384)

        self.gateway.register_node(small_node)
        self.gateway.register_node(large_node)

        # Request requiring 4GB RAM (expanded model with ~700M params)
        res = self.gateway.process_workload(
            input_data={"input": "Heavy inference", "model_param_count": 700_000_000},
            task_name="batch_ai_task"
        )

        self.assertEqual(res["status"], "SUCCESS")
        # Only the large node has adequate RAM; small node must be filtered out
        self.assertIn("large-node", res["nodes_used"])
        self.assertNotIn("small-node", res["nodes_used"])

    def test_03_gpu_dynamic_arrival_and_departure(self):
        """Verify dynamic GPU registration immediately routes heavy workloads to GPU, and deregistration falls back to CPU."""
        cpu1 = self._create_node("cpu-1", "CPU Node 1", ram_mb=4096)
        cpu2 = self._create_node("cpu-2", "CPU Node 2", ram_mb=4096)

        self.gateway.register_node(cpu1)
        self.gateway.register_node(cpu2)

        # Phase 1: Only CPU nodes available. Image generation should fallback to DISTRIBUTED_CPU
        res_cpu = self.gateway.process_workload(
            input_data={"input": "Render batch", "num_images": 4, "parallelizable": True},
            task_name="image_generation"
        )
        self.assertEqual(res_cpu["execution_mode"], ExecutionMode.DISTRIBUTED_CPU.value)
        self.assertTrue(res_cpu["fallback_occurred"])

        # Phase 2: GPU dynamically arrives
        gpu_node = self._create_node(
            "gpu-worker-1",
            "NVIDIA RTX 4090",
            ram_mb=32768,
            gpu_available=True,
            gpu_vram_mb=24576,
            gpu_model="RTX 4090"
        )
        self.gateway.register_node(gpu_node)

        res_gpu = self.gateway.process_workload(
            input_data={"input": "Render batch", "num_images": 4, "parallelizable": True},
            task_name="image_generation"
        )
        self.assertEqual(res_gpu["execution_mode"], ExecutionMode.SINGLE_GPU.value)
        self.assertFalse(res_gpu["fallback_occurred"])
        self.assertIn("gpu-worker-1", res_gpu["nodes_used"])

        # Phase 3: GPU departs / deregisters
        self.gateway.deregister_node("gpu-worker-1")

        res_fallback = self.gateway.process_workload(
            input_data={"input": "Render batch", "num_images": 4, "parallelizable": True},
            task_name="image_generation"
        )
        self.assertEqual(res_fallback["execution_mode"], ExecutionMode.DISTRIBUTED_CPU.value)
        self.assertTrue(res_fallback["fallback_occurred"])

    def test_04_cpu_gpu_hybrid_pipeline_execution(self):
        """Verify CPU_GPU_HYBRID execution correctly pipelines tasks across heterogeneous nodes."""
        # Setup mixed cluster: 3 CPU nodes and 1 GPU node
        for i in range(3):
            self.gateway.register_node(self._create_node(f"cpu-{i}", f"CPU Worker {i}", ram_mb=8192))
        self.gateway.register_node(
            self._create_node("gpu-primary", "Tesla T4", ram_mb=16384, gpu_available=True, gpu_vram_mb=16384, gpu_model="Tesla T4")
        )

        res = self.gateway.process_workload(
            input_data={"items": ["chunk_A", "chunk_B", "chunk_C", "chunk_D"], "parallelizable": True},
            task_name="batch_ai_task"
        )

        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["execution_mode"], ExecutionMode.CPU_GPU_HYBRID.value)
        self.assertGreaterEqual(len(res["nodes_used"]), 2)
        # Verify output contains hybrid executed fragments
        for item in res["result"]:
            self.assertIn("Hybrid_Processed", item)

    def test_05_elastic_arbitrary_worker_scaling(self):
        """Verify elastic load distribution across variable node pools (e.g. 5 nodes)."""
        for i in range(5):
            self.gateway.register_node(self._create_node(f"worker-{i}", f"Worker {i}", ram_mb=4096))

        res = self.gateway.process_workload(
            input_data={"items": [f"doc_{x}" for x in range(15)], "parallelizable": True},
            task_name="batch_ai_task"
        )

        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["execution_mode"], ExecutionMode.DISTRIBUTED_CPU.value)
        self.assertEqual(res["shard_count"], 5)
        self.assertEqual(len(res["result"]), 15)


if __name__ == "__main__":
    unittest.main()
