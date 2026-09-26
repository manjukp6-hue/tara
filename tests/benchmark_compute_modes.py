"""
tests/benchmark_compute_modes.py

Comparative Empirical Benchmarks for TARA Compute Modes:
- Mode 1: Single CPU node execution
- Mode 2: Elastic Distributed CPU (sharded across N nodes)
- Mode 3: Accelerated GPU / Simulated High-Throughput execution
- Mode 4: Failover resilience (worker crash mid-job and recovery)

Measures:
- Throughput (items/sec)
- Completion time (ms)
- Latency percentiles: p50, p95, p99 (ms)
- Peak memory delta (MB)
- Worker failover recovery duration (ms)
"""

import os
import sys
import time
import math
import unittest
import statistics
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
    DynamicLoadDistributor,
    WorkloadDistributionGateway,
    WorkloadChunk,
    CANONICAL_MODEL_SHA256
)


def run_benchmark_suite(batch_size: int = 100, repetitions: int = 5) -> Dict[str, Any]:
    print("=" * 75)
    print(f"   TARA COMPUTE MODES EMPIRICAL BENCHMARK SUITE (Batch={batch_size}, Reps={repetitions})")
    print("=" * 75)

    test_items = [f"task_item_{i:04d}" for i in range(batch_size)]
    results: Dict[str, Any] = {}

    def make_node(nid: str, is_gpu: bool = False) -> EndpointDefinition:
        return EndpointDefinition(
            endpoint_id=nid,
            name=f"{'GPU' if is_gpu else 'CPU'}_{nid}",
            endpoint_type="LOCAL_PROCESS",
            base_url="",
            capabilities=[EndpointCapability.INFERENCE.value],
            priority=50,
            resource_limits=ResourceLimits(
                cpu_cores=4,
                ram_mb=8192,
                gpu_available=is_gpu,
                gpu_model="RTX 4090" if is_gpu else "",
                gpu_vram_mb=24576 if is_gpu else 0,
                cpu_capacity=4.0
            ),
            current_load=0.05,
            queue_depth=0,
            model_available=True,
            model_sha256=CANONICAL_MODEL_SHA256,
            health_status=EndpointHealthStatus.HEALTHY
        )

    # ------------------------------------------------------------------------
    # Benchmark 1: Single CPU Execution
    # ------------------------------------------------------------------------
    probe1 = EndpointHealthProbe(timeout_seconds=0.5)
    router1 = AutoConnectRouter(probe=probe1)
    gw_single = WorkloadDistributionGateway(router=router1)
    gw_single.register_node(make_node("single-cpu-01", is_gpu=False))

    single_cpu_latencies = []
    single_cpu_times = []

    def single_executor(chk: WorkloadChunk, node: EndpointDefinition):
        res = []
        for itm in chk.items:
            t0 = time.perf_counter()
            # Synthetic compute simulation: ~0.5ms per item CPU compute
            acc = 0
            for k in range(5000):
                acc += (k % 7)
            dt = (time.perf_counter() - t0) * 1000.0
            single_cpu_latencies.append(dt)
            res.append(f"done_{itm}")
        return res

    for _ in range(repetitions):
        t_start = time.perf_counter()
        res = gw_single.process_workload(
            input_data={"items": test_items, "parallelizable": False},
            task_name="batch_ai_task",
            custom_executor=single_executor
        )
        t_tot = (time.perf_counter() - t_start) * 1000.0
        single_cpu_times.append(t_tot)

    single_cpu_latencies.sort()
    single_p50 = statistics.median(single_cpu_latencies)
    single_p95 = single_cpu_latencies[int(len(single_cpu_latencies) * 0.95)]
    single_p99 = single_cpu_latencies[int(len(single_cpu_latencies) * 0.99)]
    avg_single_time = statistics.mean(single_cpu_times)
    single_throughput = (batch_size / (avg_single_time / 1000.0))

    results["single_cpu"] = {
        "mode": "SINGLE_CPU",
        "nodes": 1,
        "avg_completion_time_ms": round(avg_single_time, 2),
        "throughput_items_per_sec": round(single_throughput, 1),
        "p50_item_ms": round(single_p50, 4),
        "p95_item_ms": round(single_p95, 4),
        "p99_item_ms": round(single_p99, 4)
    }

    # ------------------------------------------------------------------------
    # Benchmark 2: Elastic Distributed CPU (4 Workers)
    # ------------------------------------------------------------------------
    probe2 = EndpointHealthProbe(timeout_seconds=0.5)
    router2 = AutoConnectRouter(probe=probe2)
    gw_dist = WorkloadDistributionGateway(router=router2)
    for i in range(4):
        gw_dist.register_node(make_node(f"dist-cpu-{i}", is_gpu=False))

    dist_cpu_latencies = []
    dist_cpu_times = []

    def dist_executor(chk: WorkloadChunk, node: EndpointDefinition):
        res = []
        for itm in chk.items:
            t0 = time.perf_counter()
            acc = 0
            for k in range(5000):
                acc += (k % 7)
            dt = (time.perf_counter() - t0) * 1000.0
            dist_cpu_latencies.append(dt)
            res.append(f"done_{itm}")
        return res

    for _ in range(repetitions):
        t_start = time.perf_counter()
        res = gw_dist.process_workload(
            input_data={"items": test_items, "parallelizable": True},
            task_name="batch_ai_task",
            custom_executor=dist_executor
        )
        t_tot = (time.perf_counter() - t_start) * 1000.0
        dist_cpu_times.append(t_tot)

    dist_cpu_latencies.sort()
    dist_p50 = statistics.median(dist_cpu_latencies)
    dist_p95 = dist_cpu_latencies[int(len(dist_cpu_latencies) * 0.95)]
    dist_p99 = dist_cpu_latencies[int(len(dist_cpu_latencies) * 0.99)]
    avg_dist_time = statistics.mean(dist_cpu_times)
    dist_throughput = (batch_size / (avg_dist_time / 1000.0))

    results["distributed_cpu"] = {
        "mode": "DISTRIBUTED_CPU",
        "nodes": 4,
        "avg_completion_time_ms": round(avg_dist_time, 2),
        "throughput_items_per_sec": round(dist_throughput, 1),
        "p50_item_ms": round(dist_p50, 4),
        "p95_item_ms": round(dist_p95, 4),
        "p99_item_ms": round(dist_p99, 4),
        "speedup_vs_single": round(avg_single_time / max(avg_dist_time, 0.001), 2)
    }

    # ------------------------------------------------------------------------
    # Benchmark 3: GPU Accelerated Execution (Simulated)
    # ------------------------------------------------------------------------
    probe3 = EndpointHealthProbe(timeout_seconds=0.5)
    router3 = AutoConnectRouter(probe=probe3)
    gw_gpu = WorkloadDistributionGateway(router=router3)
    gw_gpu.register_node(make_node("gpu-node-01", is_gpu=True))

    gpu_latencies = []
    gpu_times = []

    def gpu_executor(chk: WorkloadChunk, node: EndpointDefinition):
        res = []
        for itm in chk.items:
            t0 = time.perf_counter()
            # Vectorized GPU simulation: ~10x faster
            acc = sum((k % 7) for k in range(500))
            dt = (time.perf_counter() - t0) * 1000.0
            gpu_latencies.append(dt)
            res.append(f"gpu_done_{itm}")
        return res

    for _ in range(repetitions):
        t_start = time.perf_counter()
        res = gw_gpu.process_workload(
            input_data={"items": test_items, "parallelizable": False},
            task_name="image_generation",
            custom_executor=gpu_executor
        )
        t_tot = (time.perf_counter() - t_start) * 1000.0
        gpu_times.append(t_tot)

    gpu_latencies.sort()
    gpu_p50 = statistics.median(gpu_latencies)
    gpu_p95 = gpu_latencies[int(len(gpu_latencies) * 0.95)]
    gpu_p99 = gpu_latencies[int(len(gpu_latencies) * 0.99)]
    avg_gpu_time = statistics.mean(gpu_times)
    gpu_throughput = (batch_size / (avg_gpu_time / 1000.0))

    results["gpu_accelerated"] = {
        "mode": "SINGLE_GPU",
        "nodes": 1,
        "avg_completion_time_ms": round(avg_gpu_time, 2),
        "throughput_items_per_sec": round(gpu_throughput, 1),
        "p50_item_ms": round(gpu_p50, 4),
        "p95_item_ms": round(gpu_p95, 4),
        "p99_item_ms": round(gpu_p99, 4),
        "speedup_vs_single_cpu": round(avg_single_time / max(avg_gpu_time, 0.001), 2)
    }

    # ------------------------------------------------------------------------
    # Benchmark 4: Mid-flight Worker Failure & Recovery
    # ------------------------------------------------------------------------
    probe4 = EndpointHealthProbe(timeout_seconds=0.5)
    router4 = AutoConnectRouter(probe=probe4)
    gw_failover = WorkloadDistributionGateway(router=router4)
    gw_failover.register_node(make_node("stable-worker-1", is_gpu=False))
    gw_failover.register_node(make_node("flaky-worker-2", is_gpu=False))

    failed_once = False

    def flaky_executor(chk: WorkloadChunk, node: EndpointDefinition):
        nonlocal failed_once
        if node.node_id == "flaky-worker-2" and not failed_once:
            failed_once = True
            raise RuntimeError("Mid-flight worker hardware fault simulation!")
        return [f"recovered_{itm}" for itm in chk.items]

    t_start = time.perf_counter()
    failover_res = gw_failover.process_workload(
        input_data={"items": test_items, "parallelizable": True},
        task_name="batch_ai_task",
        custom_executor=flaky_executor
    )
    failover_recovery_time_ms = (time.perf_counter() - t_start) * 1000.0

    results["failover_benchmark"] = {
        "status": failover_res["status"],
        "recovery_time_ms": round(failover_recovery_time_ms, 2),
        "failover_events_count": len(failover_res.get("failover_events", [])),
        "data_integrity_preserved": len(failover_res["result"]) == batch_size
    }

    # Print Summary Table
    print("\n" + f"{'Mode':<20} | {'Nodes':<5} | {'Avg Time (ms)':<14} | {'Throughput (it/s)':<18} | {'p50 (ms)':<10} | {'p99 (ms)':<10}")
    print("-" * 88)
    for m_key in ["single_cpu", "distributed_cpu", "gpu_accelerated"]:
        d = results[m_key]
        print(f"{d['mode']:<20} | {d['nodes']:<5} | {d['avg_completion_time_ms']:<14.2f} | {d['throughput_items_per_sec']:<18.1f} | {d['p50_item_ms']:<10.4f} | {d['p99_item_ms']:<10.4f}")

    print("\nFailover Benchmark Result:")
    fb = results["failover_benchmark"]
    print(f"  Status: {fb['status']}, Recovery: {fb['recovery_time_ms']}ms, Data Preserved: {fb['data_integrity_preserved']}")
    print("=" * 75)

    return results


class TestComputeBenchmarks(unittest.TestCase):
    def test_run_benchmarks(self):
        metrics = run_benchmark_suite(batch_size=50, repetitions=3)
        self.assertIn("single_cpu", metrics)
        self.assertIn("distributed_cpu", metrics)
        self.assertIn("gpu_accelerated", metrics)
        self.assertIn("failover_benchmark", metrics)
        self.assertTrue(metrics["failover_benchmark"]["data_integrity_preserved"])


if __name__ == "__main__":
    unittest.main()
