"""
python/tara_core/workload_distribution.py

TARA AI Distributed Workload Architecture:
1. Multi-Server Load Distribution (Dynamic weighted balancing, real-time node telemetry, open-ended node discovery).
2. CPU/GPU Capability Routing (Automatic classification of workloads, hardware matching, GPU preference).
3. Elastic Distributed CPU Fallback (Dynamic N-worker CPU sharding when GPU unavailable, failure recovery & redistribution).

Pipeline:
TARA Gateway -> Workload Detector -> Capability Router -> Dynamic Load Distributor -> Pools -> Result Aggregator -> TARA Response
"""

import os
import sys
import time
import uuid
import math
import logging
import threading
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Callable, Union, Set
from dataclasses import dataclass, field, asdict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tara_core.auto_connect_sync import (
    EndpointDefinition,
    EndpointHealthStatus,
    EndpointCapability,
    ResourceLimits,
    AutoConnectRouter
)

logger = logging.getLogger("tara_core.workload_distribution")

CANONICAL_MODEL_SHA256 = "7a50308b799f2654baeafbd64dec31f088c3b07b446e198cd0f9ec2b7c0af309"
CANONICAL_PARAM_COUNT = 118080


# ============================================================================
# 1. Enums & Data Structures
# ============================================================================

class WorkloadType(str, Enum):
    NORMAL_CHAT = "NORMAL_CHAT"
    HEAVY_TEXT_LLM = "HEAVY_TEXT_LLM"
    IMAGE_GENERATION = "IMAGE_GENERATION"
    VIDEO_GENERATION = "VIDEO_GENERATION"
    EMBEDDING = "EMBEDDING"
    BATCH_AI_TASK = "BATCH_AI_TASK"
    CUSTOM = "CUSTOM"


class ExecutionMode(str, Enum):
    SINGLE_CPU = "SINGLE_CPU"
    DISTRIBUTED_CPU = "DISTRIBUTED_CPU"
    SINGLE_GPU = "SINGLE_GPU"
    MULTI_GPU = "MULTI_GPU"
    CPU_GPU_HYBRID = "CPU_GPU_HYBRID"


@dataclass
class WorkloadChunk:
    chunk_id: str
    workload_id: str
    index: int
    total_chunks: int
    items: List[Any] = field(default_factory=list)
    payload: Dict[str, Any] = field(default_factory=dict)
    assigned_node_id: Optional[str] = None
    status: str = "PENDING"  # PENDING, RUNNING, COMPLETED, FAILED
    result: Optional[Any] = None
    error: Optional[str] = None
    retries: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "workload_id": self.workload_id,
            "index": self.index,
            "total_chunks": self.total_chunks,
            "items_count": len(self.items),
            "assigned_node_id": self.assigned_node_id,
            "status": self.status,
            "error": self.error,
            "retries": self.retries
        }


@dataclass
class WorkloadDescriptor:
    workload_id: str
    workload_type: WorkloadType
    task_name: str
    input_text: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    items: List[Any] = field(default_factory=list)
    is_parallelizable: bool = False
    is_large: bool = False
    estimated_item_count: int = 1
    required_capabilities: List[str] = field(default_factory=list)
    preferred_device: str = "AUTO"  # "CPU", "GPU", "AUTO"
    requires_model: bool = False
    target_model_sha256: Optional[str] = CANONICAL_MODEL_SHA256
    model_param_count: int = CANONICAL_PARAM_COUNT
    min_ram_mb: int = 256
    min_vram_mb: int = 0
    context: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Dynamically scale required RAM and VRAM according to model parameter count
        # Formula: Required RAM/VRAM = (params * 4 bytes [F32] * 1.5 overhead) / 1024^2 MB
        if self.requires_model and self.model_param_count > 0:
            model_mem_mb = math.ceil((self.model_param_count * 4 * 1.5) / (1024 * 1024))
            self.min_ram_mb = max(self.min_ram_mb, model_mem_mb)
            if self.preferred_device == "GPU":
                self.min_vram_mb = max(self.min_vram_mb, model_mem_mb)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workload_id": self.workload_id,
            "workload_type": self.workload_type.value,
            "task_name": self.task_name,
            "is_parallelizable": self.is_parallelizable,
            "is_large": self.is_large,
            "estimated_item_count": self.estimated_item_count,
            "required_capabilities": self.required_capabilities,
            "preferred_device": self.preferred_device,
            "requires_model": self.requires_model,
            "target_model_sha256": self.target_model_sha256,
            "model_param_count": self.model_param_count,
            "min_ram_mb": self.min_ram_mb,
            "min_vram_mb": self.min_vram_mb
        }


# ============================================================================
# 2. Workload Detector
# ============================================================================

class WorkloadDetector:
    """
    Analyzes incoming user requests or API payloads.
    Automatically classifies into WorkloadType, detects parallelizability,
    assesses size/complexity, and specifies required capabilities.
    """

    IMAGE_KEYWORDS = {"generate image", "draw", "diffusion", "txt2img", "create image", "render picture", "photograph", "artwork"}
    VIDEO_KEYWORDS = {"generate video", "render animation", "create video", "video clip", "txt2video", "render video"}
    EMBEDDING_KEYWORDS = {"embed", "embedding", "vectorize", "similarity search", "compute embeddings"}
    HEAVY_TEXT_KEYWORDS = {"summarize document", "code review", "synthesize", "deep analysis", "batch transcribe", "long-form"}

    def detect(
        self,
        input_data: Union[str, Dict[str, Any]],
        task_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> WorkloadDescriptor:
        workload_id = f"wl_{uuid.uuid4().hex[:12]}"
        ctx = dict(context or {})

        if isinstance(input_data, str):
            text = input_data.strip()
            text_lower = text.lower()
            payload: Dict[str, Any] = {"input": text}
            items: List[Any] = []
        else:
            payload = dict(input_data or {})
            text = str(payload.get("input", "")).strip()
            text_lower = text.lower()
            items = list(payload.get("items", payload.get("batch", payload.get("texts", []))))

        # 1. Detect explicit declared task
        explicit_task = (task_name or payload.get("task_name") or payload.get("task_type") or "").lower()

        # 2. Classification logic
        if explicit_task in ("image_generation", "txt2img", "image_render") or any(kw in text_lower for kw in self.IMAGE_KEYWORDS):
            w_type = WorkloadType.IMAGE_GENERATION
            pref_dev = "GPU"
            req_caps = [EndpointCapability.INFERENCE.value]
            min_vram = 2048
            min_ram = 1024
            # Image tasks can be parallelized if batch of images or multi-tile
            parallelizable = payload.get("parallelizable", len(items) > 1 or payload.get("num_images", 1) > 1)
            item_cnt = max(len(items), int(payload.get("num_images", 1)))
            is_large = item_cnt > 1 or payload.get("resolution", 512) >= 1024

        elif explicit_task in ("video_generation", "txt2video", "video_render") or any(kw in text_lower for kw in self.VIDEO_KEYWORDS):
            w_type = WorkloadType.VIDEO_GENERATION
            pref_dev = "GPU"
            req_caps = [EndpointCapability.INFERENCE.value]
            min_vram = 4096
            min_ram = 2048
            # Video generation is parallelizable across frames / chunks
            frames = int(payload.get("frames", payload.get("num_frames", len(items) or 1)))
            parallelizable = payload.get("parallelizable", frames > 4)
            item_cnt = frames
            is_large = frames > 8 or payload.get("duration_sec", 1) > 3

        elif explicit_task in ("embedding", "text_embeddings", "vectorize") or any(kw in text_lower for kw in self.EMBEDDING_KEYWORDS):
            w_type = WorkloadType.EMBEDDING
            pref_dev = "AUTO"
            req_caps = [EndpointCapability.INFERENCE.value]
            min_vram = 0
            min_ram = 512
            item_cnt = max(1, len(items) if items else int(payload.get("count", 1)))
            parallelizable = item_cnt > 1
            is_large = item_cnt >= 4

        elif explicit_task in ("batch_ai_task", "batch_inference") or len(items) > 1:
            w_type = WorkloadType.BATCH_AI_TASK
            pref_dev = "AUTO"
            req_caps = [EndpointCapability.INFERENCE.value]
            min_vram = 0
            min_ram = 512
            item_cnt = len(items)
            parallelizable = payload.get("parallelizable", True)
            is_large = item_cnt >= 2

        elif len(text) > 500 or any(kw in text_lower for kw in self.HEAVY_TEXT_KEYWORDS) or payload.get("max_tokens", 0) > 300:
            w_type = WorkloadType.HEAVY_TEXT_LLM
            pref_dev = "AUTO"
            req_caps = [EndpointCapability.INFERENCE.value]
            min_vram = 0
            min_ram = 512
            item_cnt = max(1, len(items))
            # Heavy text can be parallelized if multi-section or batch of documents
            parallelizable = payload.get("parallelizable", len(items) > 1 or payload.get("sections", 0) > 1)
            is_large = len(text) > 1000 or item_cnt > 1

        else:
            # Normal conversational chat (e.g. "Hi TARA", "What is your name?")
            w_type = WorkloadType.NORMAL_CHAT
            pref_dev = "CPU"
            req_caps = [EndpointCapability.INFERENCE.value]
            min_vram = 0
            min_ram = 256
            item_cnt = 1
            parallelizable = False  # Normal chat is strictly non-parallelizable atomic generation
            is_large = False

        # Allow explicit override from payload if provided
        if "parallelizable" in payload:
            parallelizable = bool(payload["parallelizable"])
        if "is_large" in payload:
            is_large = bool(payload["is_large"])

        return WorkloadDescriptor(
            workload_id=workload_id,
            workload_type=w_type,
            task_name=explicit_task or w_type.value.lower(),
            input_text=text,
            payload=payload,
            items=items,
            is_parallelizable=parallelizable,
            is_large=is_large,
            estimated_item_count=item_cnt,
            required_capabilities=req_caps,
            preferred_device=pref_dev,
            requires_model=payload.get("requires_model", True),
            target_model_sha256=payload.get("target_model_sha256", CANONICAL_MODEL_SHA256),
            model_param_count=int(payload.get("model_param_count", CANONICAL_PARAM_COUNT)),
            min_ram_mb=min_ram,
            min_vram_mb=min_vram,
            context=ctx
        )


# ============================================================================
# 3. Capability Router
# ============================================================================

class CapabilityRouter:
    """
    Evaluates hardware capabilities, active node health, model SHA256 integrity,
    and selects the optimal ExecutionMode (SINGLE_CPU, DISTRIBUTED_CPU, SINGLE_GPU,
    MULTI_GPU, CPU_GPU_HYBRID).
    """

    def __init__(self, canonical_model_sha: str = CANONICAL_MODEL_SHA256):
        self.canonical_model_sha = canonical_model_sha

    def filter_eligible_nodes(
        self,
        nodes: List[EndpointDefinition],
        workload: WorkloadDescriptor
    ) -> List[EndpointDefinition]:
        eligible = []
        for n in nodes:
            # 1. Enabled & not unreachable
            if not n.is_enabled:
                continue
            if n.health_status == EndpointHealthStatus.UNREACHABLE:
                continue

            # 2. Capabilities check
            if workload.required_capabilities:
                node_caps = set(n.capabilities)
                if not all(c in node_caps for c in workload.required_capabilities):
                    continue

            # 3. Model compatibility check (if inference requires model)
            if workload.requires_model:
                if not n.model_available:
                    continue
                # If node has a model_sha256 declared, it MUST match the target canonical model SHA256!
                if n.model_sha256 and workload.target_model_sha256:
                    if n.model_sha256.lower() != workload.target_model_sha256.lower():
                        logger.warning(
                            "Node %s rejected: model SHA256 %s does not match required canonical %s",
                            n.node_id, n.model_sha256, workload.target_model_sha256
                        )
                        continue

            # 4. RAM check
            if n.ram_mb < workload.min_ram_mb:
                continue

            eligible.append(n)

        return eligible

    def route(
        self,
        workload: WorkloadDescriptor,
        available_nodes: List[EndpointDefinition]
    ) -> Tuple[ExecutionMode, List[EndpointDefinition]]:
        """
        Determines execution mode and candidates based on hardware & workload.
        """
        eligible = self.filter_eligible_nodes(available_nodes, workload)
        if not eligible:
            logger.warning("No eligible nodes found for workload %s", workload.workload_id)
            return ExecutionMode.SINGLE_CPU, []

        gpu_nodes = [n for n in eligible if n.gpu_available and n.gpu_vram_mb >= workload.min_vram_mb]
        cpu_nodes = [n for n in eligible if not n.gpu_available]

        # Rule 1: Normal Chat -> Always single suitable CPU node (do NOT distribute simple requests)
        if workload.workload_type == WorkloadType.NORMAL_CHAT:
            candidates = cpu_nodes if cpu_nodes else eligible
            return ExecutionMode.SINGLE_CPU, candidates

        # Rule 2: GPU preferred/required workloads (Image, Video, Heavy GPU task)
        if workload.preferred_device == "GPU" or workload.workload_type in (WorkloadType.IMAGE_GENERATION, WorkloadType.VIDEO_GENERATION):
            if gpu_nodes:
                if len(gpu_nodes) > 1 and workload.is_parallelizable and workload.is_large:
                    return ExecutionMode.MULTI_GPU, gpu_nodes
                return ExecutionMode.SINGLE_GPU, gpu_nodes

            # === NO GPU AVAILABLE -> ELASTIC DISTRIBUTED CPU FALLBACK ===
            logger.info("No GPU node available for %s. Evaluating CPU fallback...", workload.workload_type.value)
            all_cpu_candidates = eligible  # all available can execute CPU tasks

            if workload.is_parallelizable and workload.is_large and len(all_cpu_candidates) > 1:
                # Dynamically divide workload across available CPU nodes
                return ExecutionMode.DISTRIBUTED_CPU, all_cpu_candidates
            else:
                # Workload not safely parallelizable or too small -> use best single CPU node
                return ExecutionMode.SINGLE_CPU, all_cpu_candidates

        # Rule 3: Heavy Text / Embedding / Batch AI tasks
        if workload.workload_type in (WorkloadType.HEAVY_TEXT_LLM, WorkloadType.EMBEDDING, WorkloadType.BATCH_AI_TASK):
            if gpu_nodes and not cpu_nodes:
                if len(gpu_nodes) > 1 and workload.is_parallelizable and workload.is_large:
                    return ExecutionMode.MULTI_GPU, gpu_nodes
                return ExecutionMode.SINGLE_GPU, gpu_nodes

            if gpu_nodes and cpu_nodes and workload.is_parallelizable and workload.is_large and len(eligible) >= 4:
                # Mixed CPU/GPU cluster hybrid execution
                return ExecutionMode.CPU_GPU_HYBRID, eligible

            if gpu_nodes:
                return ExecutionMode.SINGLE_GPU, gpu_nodes

            # CPU-only pool
            if workload.is_parallelizable and workload.is_large and len(eligible) > 1:
                return ExecutionMode.DISTRIBUTED_CPU, eligible

            return ExecutionMode.SINGLE_CPU, eligible

        # Default fallback
        if workload.is_parallelizable and workload.is_large and len(eligible) > 1:
            return ExecutionMode.DISTRIBUTED_CPU, eligible
        return ExecutionMode.SINGLE_CPU, eligible


# ============================================================================
# 4. Dynamic Weighted Load Distributor
# ============================================================================

class DynamicLoadDistributor:
    """
    Computes dynamic weights for node selection taking into account:
    - health (healthy vs degraded)
    - available capacity (CPU cores/score, RAM, GPU VRAM)
    - current load (penalizes busy nodes)
    - queue depth (penalizes pending work)
    - latency (penalizes slow RTT)
    - capability & model compatibility
    """

    def calculate_node_weight(self, node: EndpointDefinition, workload: WorkloadDescriptor) -> float:
        if not node.is_enabled or node.health_status == EndpointHealthStatus.UNREACHABLE:
            return 0.0

        # Base compute capacity
        cpu_score = max(1.0, float(node.cpu_capacity)) * 10.0
        ram_score = (node.ram_mb / 1024.0) * 2.0
        capacity = cpu_score + ram_score

        # GPU bonus if applicable (only when workload benefits from GPU)
        if node.gpu_available and workload.preferred_device != "CPU" and workload.workload_type != WorkloadType.NORMAL_CHAT:
            gpu_bonus = 50.0 + (node.gpu_vram_mb / 512.0)
            capacity += gpu_bonus
        elif node.gpu_available and (workload.preferred_device == "CPU" or workload.workload_type == WorkloadType.NORMAL_CHAT):
            # Deprioritize GPU node for pure CPU chat to preserve GPU capacity
            capacity -= 20.0

        # Current load penalty (normalized 0.0 to 1.0)
        load = max(0.0, float(node.current_load))
        load_penalty = min(load * 60.0, 90.0)

        # Queue depth penalty
        queue = max(0, int(node.queue_depth))
        queue_penalty = min(queue * 15.0, 100.0)

        # Latency penalty
        rtt = node.average_rtt_ms if node.average_rtt_ms > 0 else 5.0
        latency_penalty = min(rtt * 0.5, 40.0)

        # Health penalty
        health_mult = 1.0
        if node.health_status == EndpointHealthStatus.DEGRADED:
            health_mult = 0.35

        # Failures penalty
        fail_penalty = node.consecutive_failures * 20.0

        raw_score = capacity - load_penalty - queue_penalty - latency_penalty - fail_penalty
        final_weight = max(1.0, raw_score) * health_mult
        return final_weight

    def select_best_single_node(
        self,
        nodes: List[EndpointDefinition],
        workload: WorkloadDescriptor
    ) -> Optional[EndpointDefinition]:
        if not nodes:
            return None
        scored = [(self.calculate_node_weight(n, workload), n) for n in nodes]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    def select_nodes_for_distribution(
        self,
        nodes: List[EndpointDefinition],
        workload: WorkloadDescriptor,
        max_workers: Optional[int] = None
    ) -> List[EndpointDefinition]:
        """
        Selects all healthy nodes eligible for distributed execution.
        NO hardcoded limit: automatically uses all healthy available workers.
        """
        if not nodes:
            return []
        scored = [(self.calculate_node_weight(n, workload), n) for n in nodes if n.health_status != EndpointHealthStatus.UNREACHABLE]
        scored.sort(key=lambda x: x[0], reverse=True)

        selected = [n for s, n in scored if s > 0.0]
        if max_workers and max_workers > 0:
            selected = selected[:max_workers]
        return selected

    def partition_workload(
        self,
        workload: WorkloadDescriptor,
        workers: List[EndpointDefinition]
    ) -> List[WorkloadChunk]:
        """
        Dynamically divides workload across available workers.
        NO fixed worker/shard count: shard count equals min(item_count, len(workers)).
        """
        if not workers:
            return []

        worker_count = len(workers)
        items = list(workload.items)

        # Case A: Workload has explicit discrete items/batch
        if items:
            total_items = len(items)
            num_shards = min(total_items, worker_count)
            weights = [self.calculate_node_weight(w, workload) for w in workers[:num_shards]]
            total_weight = sum(weights) or float(num_shards)

            chunks: List[WorkloadChunk] = []
            curr_idx = 0
            for i in range(num_shards):
                prop = weights[i] / total_weight
                share = max(1, int(round(prop * total_items)))
                if i == num_shards - 1:
                    chunk_items = items[curr_idx:]
                else:
                    end_idx = min(total_items, curr_idx + share)
                    chunk_items = items[curr_idx:end_idx]
                    curr_idx = end_idx

                chunk = WorkloadChunk(
                    chunk_id=f"chk_{workload.workload_id}_{i}",
                    workload_id=workload.workload_id,
                    index=i,
                    total_chunks=num_shards,
                    items=chunk_items,
                    payload=dict(workload.payload),
                    assigned_node_id=workers[i].node_id
                )
                chunks.append(chunk)

            return chunks

        # Case B: Workload has numeric count (e.g. frames, tokens, simulations)
        total_count = workload.estimated_item_count
        if total_count > 1:
            num_shards = min(total_count, worker_count)
            chunk_size = math.ceil(total_count / num_shards)
            chunks = []
            for i in range(num_shards):
                start_i = i * chunk_size
                end_i = min(total_count, (i + 1) * chunk_size)
                if start_i >= total_count:
                    break
                chk_payload = dict(workload.payload)
                chk_payload["shard_start"] = start_i
                chk_payload["shard_end"] = end_i
                chk_payload["shard_count"] = end_i - start_i

                chunk = WorkloadChunk(
                    chunk_id=f"chk_{workload.workload_id}_{i}",
                    workload_id=workload.workload_id,
                    index=i,
                    total_chunks=num_shards,
                    items=list(range(start_i, end_i)),
                    payload=chk_payload,
                    assigned_node_id=workers[i].node_id
                )
                chunks.append(chunk)
            return chunks

        # Case C: Single unit workload
        chunk = WorkloadChunk(
            chunk_id=f"chk_{workload.workload_id}_0",
            workload_id=workload.workload_id,
            index=0,
            total_chunks=1,
            items=[workload.input_text] if workload.input_text else [],
            payload=dict(workload.payload),
            assigned_node_id=workers[0].node_id
        )
        return [chunk]


# ============================================================================
# 5. Result Aggregator & Fault-Tolerant Execution
# ============================================================================

class ResultAggregator:
    """
    Coordinates distributed chunk execution, monitors worker health,
    preserves completed work, and dynamically redistributes unfinished work
    upon worker failure.
    """

    def __init__(self, router: Optional[AutoConnectRouter] = None):
        self.router = router
        self._lock = threading.Lock()

    def aggregate_results(
        self,
        workload: WorkloadDescriptor,
        completed_chunks: List[WorkloadChunk]
    ) -> Dict[str, Any]:
        """
        Combines partial results from all completed chunks into a unified response.
        """
        completed_chunks.sort(key=lambda c: c.index)

        aggregated_items = []
        text_fragments = []
        metrics: Dict[str, Any] = {"total_chunks": len(completed_chunks), "node_contributors": []}

        for chunk in completed_chunks:
            if chunk.assigned_node_id and chunk.assigned_node_id not in metrics["node_contributors"]:
                metrics["node_contributors"].append(chunk.assigned_node_id)

            res = chunk.result
            if isinstance(res, list):
                aggregated_items.extend(res)
            elif isinstance(res, dict):
                if "items" in res and isinstance(res["items"], list):
                    aggregated_items.extend(res["items"])
                elif "text" in res:
                    text_fragments.append(str(res["text"]))
                elif "data" in res:
                    aggregated_items.append(res["data"])
                else:
                    aggregated_items.append(res)
            elif res is not None:
                if isinstance(res, str):
                    text_fragments.append(res)
                else:
                    aggregated_items.append(res)

        final_data: Any
        if aggregated_items:
            final_data = aggregated_items
        elif text_fragments:
            final_data = " ".join(text_fragments)
        else:
            final_data = {"status": "COMPLETED", "chunks_count": len(completed_chunks)}

        return {
            "status": "SUCCESS",
            "workload_id": workload.workload_id,
            "workload_type": workload.workload_type.value,
            "result": final_data,
            "metrics": metrics
        }

    def execute_with_failover(
        self,
        workload: WorkloadDescriptor,
        chunks: List[WorkloadChunk],
        worker_pool: Dict[str, EndpointDefinition],
        executor_fn: Callable[[WorkloadChunk, EndpointDefinition], Any]
    ) -> Dict[str, Any]:
        """
        Executes chunks across worker_pool with mid-flight failure detection,
        partial result preservation, and unfinished work redistribution.
        """
        pending_chunks = list(chunks)
        completed_chunks: List[WorkloadChunk] = []
        active_nodes = dict(worker_pool)
        failover_events: List[Dict[str, Any]] = []

        max_attempts_per_chunk = 3

        while pending_chunks:
            chunk = pending_chunks.pop(0)

            assigned_node = active_nodes.get(chunk.assigned_node_id or "")
            if not assigned_node or assigned_node.health_status == EndpointHealthStatus.UNREACHABLE:
                surviving = [n for n in active_nodes.values() if n.health_status != EndpointHealthStatus.UNREACHABLE]
                if not surviving:
                    return {
                        "status": "ERROR",
                        "workload_id": workload.workload_id,
                        "error": "All available workers failed or became unreachable. Impossible to complete distributed execution.",
                        "completed_chunks_count": len(completed_chunks),
                        "failover_events": failover_events
                    }
                surviving.sort(key=lambda n: (n.current_load, n.queue_depth))
                assigned_node = surviving[0]
                chunk.assigned_node_id = assigned_node.node_id
                failover_events.append({
                    "event": "CHUNK_REDISTRIBUTED",
                    "chunk_id": chunk.chunk_id,
                    "to_node_id": assigned_node.node_id,
                    "reason": "Previous worker unavailable or failed"
                })

            chunk.status = "RUNNING"
            assigned_node.queue_depth += 1

            try:
                res = executor_fn(chunk, assigned_node)
                chunk.result = res
                chunk.status = "COMPLETED"
                completed_chunks.append(chunk)
                assigned_node.queue_depth = max(0, assigned_node.queue_depth - 1)
            except Exception as e:
                err_msg = str(e)
                chunk.retries += 1
                chunk.error = err_msg
                assigned_node.queue_depth = max(0, assigned_node.queue_depth - 1)
                assigned_node.consecutive_failures += 1

                logger.warning("Worker %s failed executing chunk %s: %s", assigned_node.node_id, chunk.chunk_id, err_msg)

                if self.router:
                    self.router.report_endpoint_failure(assigned_node.node_id, err_msg, force_unreachable=True)

                assigned_node.health_status = EndpointHealthStatus.UNREACHABLE
                active_nodes.pop(assigned_node.node_id, None)

                failover_events.append({
                    "event": "WORKER_FAILED",
                    "failed_node_id": assigned_node.node_id,
                    "chunk_id": chunk.chunk_id,
                    "error": err_msg,
                    "timestamp": time.time()
                })

                if chunk.retries < max_attempts_per_chunk:
                    chunk.assigned_node_id = None
                    pending_chunks.insert(0, chunk)
                else:
                    return {
                        "status": "ERROR",
                        "workload_id": workload.workload_id,
                        "error": f"Chunk {chunk.chunk_id} failed after {chunk.retries} attempts: {err_msg}",
                        "completed_chunks_count": len(completed_chunks),
                        "failover_events": failover_events
                    }

        output = self.aggregate_results(workload, completed_chunks)
        output["failover_events"] = failover_events
        return output


# ============================================================================
# 6. Master Workload Distribution Gateway
# ============================================================================

class WorkloadDistributionGateway:
    """
    Unified entry point orchestrating:
    TARA Gateway -> Workload Detector -> Capability Router -> Dynamic Load Distributor -> Pools -> Result Aggregator -> TARA Response
    """

    def __init__(
        self,
        router: Optional[AutoConnectRouter] = None,
        canonical_model_sha: str = CANONICAL_MODEL_SHA256
    ):
        self.router = router
        self.detector = WorkloadDetector()
        self.capability_router = CapabilityRouter(canonical_model_sha=canonical_model_sha)
        self.distributor = DynamicLoadDistributor()
        self.aggregator = ResultAggregator(router=router)
        self._custom_nodes: Dict[str, EndpointDefinition] = {}
        self._lock = threading.RLock()

    def register_node(self, node: EndpointDefinition) -> None:
        with self._lock:
            self._custom_nodes[node.node_id] = node
            if self.router:
                self.router.register_endpoint(node)

    def deregister_node(self, node_id: str) -> bool:
        with self._lock:
            existed = self._custom_nodes.pop(node_id, None) is not None
            if self.router:
                self.router.deregister_endpoint(node_id)
            return existed

    def get_all_nodes(self) -> List[EndpointDefinition]:
        with self._lock:
            if self.router:
                return self.router.list_endpoints()
            return list(self._custom_nodes.values())

    def process_workload(
        self,
        input_data: Union[str, Dict[str, Any]],
        task_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        custom_executor: Optional[Callable[[WorkloadChunk, EndpointDefinition], Any]] = None
    ) -> Dict[str, Any]:
        """
        End-to-end execution of a workload according to dynamic scheduling.
        """
        # 1. Detect Workload
        workload = self.detector.detect(input_data=input_data, task_name=task_name, context=context)

        # 2. Capability Routing
        available_nodes = self.get_all_nodes()
        mode, candidate_nodes = self.capability_router.route(workload, available_nodes)

        if not candidate_nodes:
            return {
                "status": "ERROR",
                "workload_id": workload.workload_id,
                "error": "No healthy, compatible nodes available to execute workload.",
                "workload_type": workload.workload_type.value,
                "mode": mode.value
            }

        # 3. Dynamic Load Distribution & Partitioning
        fallback_occurred = (
            workload.preferred_device == "GPU" and
            mode in (ExecutionMode.DISTRIBUTED_CPU, ExecutionMode.SINGLE_CPU) and
            not any(n.gpu_available for n in candidate_nodes)
        )

        if mode == ExecutionMode.DISTRIBUTED_CPU or mode == ExecutionMode.MULTI_GPU or mode == ExecutionMode.CPU_GPU_HYBRID:
            selected_workers = self.distributor.select_nodes_for_distribution(candidate_nodes, workload)
            chunks = self.distributor.partition_workload(workload, selected_workers)
        else:
            best_node = self.distributor.select_best_single_node(candidate_nodes, workload)
            if not best_node:
                return {
                    "status": "ERROR",
                    "workload_id": workload.workload_id,
                    "error": "Failed selecting optimal node."
                }
            selected_workers = [best_node]
            chunks = self.distributor.partition_workload(workload, [best_node])

        worker_pool = {w.node_id: w for w in selected_workers}

        # 4. Define default executor if not supplied
        def default_executor(chk: WorkloadChunk, node: EndpointDefinition) -> Any:
            if mode == ExecutionMode.CPU_GPU_HYBRID:
                # Hybrid pipeline: CPU Preprocessing -> GPU Inference -> CPU Postprocessing
                dev_label = "GPU" if node.gpu_available else "CPU"
                return [f"Hybrid_Processed_{chk.index}_{item}_via_{dev_label}_on_{node.node_id}" for item in (chk.items or ["item"])]
            elif workload.workload_type == WorkloadType.NORMAL_CHAT:
                return f"Response from {node.name} ({node.node_id}): Processed chat '{workload.input_text}'"
            elif workload.workload_type == WorkloadType.IMAGE_GENERATION:
                dev = "GPU" if node.gpu_available else "CPU"
                return [f"Image_Artifact_{item}_{dev}_{node.node_id}.png" for item in (chk.items or [1])]
            elif workload.workload_type == WorkloadType.VIDEO_GENERATION:
                dev = "GPU" if node.gpu_available else "CPU"
                return [f"Frame_{f}_{dev}_{node.node_id}.mp4" for f in (chk.items or [1])]
            elif workload.workload_type == WorkloadType.EMBEDDING:
                return [[0.01 * (i + 1), 0.02 * (i + 1)] for i, _ in enumerate(chk.items or [1])]
            else:
                return [f"Result_chunk_{chk.index}_{item}_on_{node.node_id}" for item in (chk.items or ["item"])]

        executor = custom_executor or default_executor

        # 5. Result Aggregation & Fault Tolerance
        exec_start = time.time()
        res = self.aggregator.execute_with_failover(
            workload=workload,
            chunks=chunks,
            worker_pool=worker_pool,
            executor_fn=executor
        )
        duration_ms = (time.time() - exec_start) * 1000.0

        res["execution_mode"] = mode.value
        res["fallback_occurred"] = fallback_occurred
        res["nodes_used"] = list(worker_pool.keys())
        res["shard_count"] = len(chunks)
        res["duration_ms"] = duration_ms
        return res

    def get_cluster_telemetry(self) -> Dict[str, Any]:
        """Returns comprehensive telemetry for all cluster nodes."""
        nodes = self.get_all_nodes()
        telemetry = [n.get_node_telemetry() for n in nodes]
        total_cpu = sum(n.cpu_capacity for n in nodes)
        total_ram = sum(n.ram_mb for n in nodes)
        gpu_nodes = [n for n in nodes if n.gpu_available]
        total_vram = sum(n.gpu_vram_mb for n in gpu_nodes)

        return {
            "total_nodes": len(nodes),
            "healthy_nodes": len([n for n in nodes if n.health_status == EndpointHealthStatus.HEALTHY]),
            "gpu_nodes_count": len(gpu_nodes),
            "total_cpu_capacity": total_cpu,
            "total_ram_mb": total_ram,
            "total_gpu_vram_mb": total_vram,
            "nodes": telemetry
        }
