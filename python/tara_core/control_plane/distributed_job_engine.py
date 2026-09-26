"""
python/tara_core/control_plane/distributed_job_engine.py

Distributed Multi-Server Job Engine & Scheduler for TARA Control Plane.
Coordinates the execution of single-node and multi-node distributed workloads:
1. JobPlanner: Decides whether to execute on a single node or shard across multiple workers.
   Guarantees that tiny tasks are NOT split unnecessarily.
2. DistributedJob & Chunks: Idempotency keys, worker leases, checkpoints, retry tracking.
3. Fault Recovery: If a worker fails, preserves completed chunks and reassigns unfinished chunks.
4. Result Aggregator: Aggregates chunk outputs into a final coherent result in original order.
"""

import os
import sys
import time
import json
import uuid
import logging
import threading
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field, asdict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tara_core.control_plane.worker_registry import DynamicWorkerRegistry, ClusterWorkerNode, WorkerState

logger = logging.getLogger("tara_core.control_plane.distributed_job_engine")


class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    RECOVERABLE = "RECOVERABLE"


class ChunkStatus(str, Enum):
    PENDING = "PENDING"
    LEASED = "LEASED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class JobChunk:
    chunk_id: str
    job_id: str
    index: int
    total_chunks: int
    input_data: Any
    assigned_node_id: Optional[str] = None
    lease_expires_at: float = 0.0
    status: ChunkStatus = ChunkStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    retries: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "job_id": self.job_id,
            "index": self.index,
            "total_chunks": self.total_chunks,
            "input_preview": str(self.input_data)[:60] if self.input_data else "",
            "assigned_node_id": self.assigned_node_id,
            "lease_expires_at": self.lease_expires_at,
            "status": self.status.value if isinstance(self.status, ChunkStatus) else str(self.status),
            "result": self.result,
            "error": self.error,
            "retries": self.retries
        }


@dataclass
class DistributedJob:
    job_id: str
    user_id: str
    request_id: str
    task_type: str
    idempotency_key: str
    parent_job_id: Optional[str] = None
    chunks: List[JobChunk] = field(default_factory=list)
    status: JobStatus = JobStatus.PENDING
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    timeout_sec: float = 120.0
    checkpoint_data: Dict[str, Any] = field(default_factory=dict)
    final_result: Optional[Any] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "user_id": self.user_id,
            "request_id": self.request_id,
            "task_type": self.task_type,
            "idempotency_key": self.idempotency_key,
            "parent_job_id": self.parent_job_id,
            "status": self.status.value if isinstance(self.status, JobStatus) else str(self.status),
            "chunks_count": len(self.chunks),
            "completed_chunks": sum(1 for c in self.chunks if c.status == ChunkStatus.COMPLETED),
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "timeout_sec": self.timeout_sec,
            "final_result": self.final_result,
            "error": self.error,
            "chunks": [c.to_dict() for c in self.chunks]
        }


class JobPlanner:
    """
    Intelligently analyzes workloads to decide whether to run on a single worker or split across nodes.
    Prevents splitting tiny tasks unnecessarily.
    """

    @staticmethod
    def plan(
        task_type: str,
        input_payload: Dict[str, Any],
        available_nodes_count: int
    ) -> Tuple[bool, List[Any]]:
        """
        Returns (should_split, chunks_data).
        If should_split is False, chunks_data contains exactly 1 item representing the entire task.
        """
        items = input_payload.get("items") or input_payload.get("batch") or input_payload.get("documents")
        text = str(input_payload.get("input") or input_payload.get("prompt") or "")

        # Case 1: Multiple explicit items (e.g. batch inference, batch embeddings)
        if items and isinstance(items, list) and len(items) > 1:
            if available_nodes_count > 1 and len(items) >= 2:
                # Calculate optimal chunk size: distribute across available nodes
                chunk_count = min(available_nodes_count, len(items))
                chunk_size = max(1, (len(items) + chunk_count - 1) // chunk_count)
                chunks = []
                for i in range(0, len(items), chunk_size):
                    chunk_slice = items[i:i + chunk_size]
                    chunks.append({"items": chunk_slice, "base_payload": {k: v for k, v in input_payload.items() if k not in ("items", "batch", "documents")}})
                return True, chunks

        # Case 2: Very large single text document (e.g. document summarization > 2000 words)
        words = text.split()
        if len(words) > 2000 and available_nodes_count > 1 and input_payload.get("allow_sharding", True):
            chunk_size = 1000
            chunks = []
            for i in range(0, len(words), chunk_size):
                sub_text = " ".join(words[i:i + chunk_size])
                chunks.append({"text": sub_text, "chunk_index": len(chunks)})
            return True, chunks

        # Case 3: Standard single task (tiny task or normal chat) — DO NOT SPLIT
        return False, [input_payload]


class DistributedJobEngine:
    """
    Core distributed execution coordinator.
    Manages job lifecycle, leases, reassignment on node failure, and result aggregation.
    """

    def __init__(self, worker_registry: DynamicWorkerRegistry, storage_dir: Optional[str] = None):
        self.worker_registry = worker_registry
        self._lock = threading.RLock()
        self._jobs: Dict[str, DistributedJob] = {}
        self._idempotency_map: Dict[str, str] = {}  # idempotency_key -> job_id
        if storage_dir is None:
            self.storage_dir = os.path.join(REPO_ROOT, "storage", "persistence", "distributed_jobs")
        else:
            self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self._load_persisted_jobs()

    def _load_persisted_jobs(self):
        with self._lock:
            if os.path.exists(self.storage_dir):
                for fname in os.listdir(self.storage_dir):
                    if fname.endswith(".json"):
                        try:
                            fpath = os.path.join(self.storage_dir, fname)
                            with open(fpath, "r", encoding="utf-8") as f:
                                data = json.load(f)
                            job = self._job_from_dict(data)
                            self._jobs[job.job_id] = job
                            self._idempotency_map[job.idempotency_key] = job.job_id
                        except Exception as e:
                            logger.warning(f"Could not load job file '{fname}': {e}")

    def _persist_job(self, job: DistributedJob):
        try:
            fpath = os.path.join(self.storage_dir, f"{job.job_id}.json")
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(job.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist job '{job.job_id}': {e}")

    def _job_from_dict(self, data: Dict[str, Any]) -> DistributedJob:
        chunks = []
        for c in data.get("chunks", []):
            st_str = c.get("status", "PENDING")
            try:
                st = ChunkStatus(st_str)
            except ValueError:
                st = ChunkStatus.PENDING
            chunk = JobChunk(
                chunk_id=c["chunk_id"],
                job_id=c["job_id"],
                index=int(c["index"]),
                total_chunks=int(c["total_chunks"]),
                input_data=c.get("input_data", {}),
                assigned_node_id=c.get("assigned_node_id"),
                lease_expires_at=float(c.get("lease_expires_at", 0.0)),
                status=st,
                result=c.get("result"),
                error=c.get("error"),
                retries=int(c.get("retries", 0))
            )
            chunks.append(chunk)

        st_str = data.get("status", "PENDING")
        try:
            status = JobStatus(st_str)
        except ValueError:
            status = JobStatus.PENDING

        return DistributedJob(
            job_id=data["job_id"],
            user_id=data.get("user_id", "anonymous"),
            request_id=data.get("request_id", ""),
            task_type=data.get("task_type", "INFERENCE"),
            idempotency_key=data.get("idempotency_key", data["job_id"]),
            parent_job_id=data.get("parent_job_id"),
            chunks=chunks,
            status=status,
            created_at=float(data.get("created_at", time.time())),
            completed_at=data.get("completed_at"),
            timeout_sec=float(data.get("timeout_sec", 120.0)),
            checkpoint_data=data.get("checkpoint_data", {}),
            final_result=data.get("final_result"),
            error=data.get("error")
        )

    def submit_job(
        self,
        user_id: str,
        request_id: str,
        task_type: str,
        payload: Dict[str, Any],
        idempotency_key: Optional[str] = None
    ) -> DistributedJob:
        """Submits a new job or returns an existing one if idempotency key matches."""
        idem_key = idempotency_key or f"idem_{uuid.uuid4().hex}"

        with self._lock:
            # Deduplication: return existing job if idempotency key matches
            if idem_key in self._idempotency_map:
                existing_id = self._idempotency_map[idem_key]
                existing_job = self._jobs.get(existing_id)
                if existing_job:
                    return existing_job

            ready_nodes = self.worker_registry.get_ready_workers()
            available_nodes_count = len(ready_nodes)

            should_split, chunks_data = JobPlanner.plan(task_type, payload, available_nodes_count)
            job_id = f"job_{uuid.uuid4().hex[:10]}"

            chunks: List[JobChunk] = []
            for idx, cdata in enumerate(chunks_data):
                cid = f"{job_id}_chk_{idx:03d}"
                chunk = JobChunk(
                    chunk_id=cid,
                    job_id=job_id,
                    index=idx,
                    total_chunks=len(chunks_data),
                    input_data=cdata,
                    status=ChunkStatus.PENDING
                )
                chunks.append(chunk)

            job = DistributedJob(
                job_id=job_id,
                user_id=user_id,
                request_id=request_id,
                task_type=task_type,
                idempotency_key=idem_key,
                chunks=chunks,
                status=JobStatus.PENDING
            )

            self._jobs[job_id] = job
            self._idempotency_map[idem_key] = job_id
            self._persist_job(job)
            return job

    def assign_next_chunks(self) -> List[Tuple[JobChunk, ClusterWorkerNode]]:
        """
        Leases pending chunks to best available workers.
        Handles lease timeouts and reassignments.
        """
        with self._lock:
            assignments = []
            now = time.time()

            for job in self._jobs.values():
                if job.status not in (JobStatus.PENDING, JobStatus.RUNNING):
                    continue

                for chunk in job.chunks:
                    # Check for lease timeout: reassign if node expired without completing
                    if chunk.status == ChunkStatus.LEASED and now > chunk.lease_expires_at:
                        logger.warning(f"Chunk '{chunk.chunk_id}' lease expired on '{chunk.assigned_node_id}'. Reassigning.")
                        chunk.status = ChunkStatus.PENDING
                        chunk.retries += 1
                        chunk.assigned_node_id = None

                    if chunk.status == ChunkStatus.PENDING:
                        worker = self.worker_registry.select_best_worker(workload_type=job.task_type)
                        if worker:
                            chunk.status = ChunkStatus.LEASED
                            chunk.assigned_node_id = worker.node_id
                            chunk.lease_expires_at = now + 45.0  # 45 second lease
                            job.status = JobStatus.RUNNING
                            worker.queue_depth += 1
                            assignments.append((chunk, worker))

            return assignments

    def complete_chunk(self, chunk_id: str, result: Any) -> Optional[DistributedJob]:
        """
        Marks a chunk as successfully completed.
        Checks if all chunks in the job are complete, and aggregates results.
        """
        with self._lock:
            for job in self._jobs.values():
                for chunk in job.chunks:
                    if chunk.chunk_id == chunk_id:
                        chunk.status = ChunkStatus.COMPLETED
                        chunk.result = result
                        chunk.error = None

                        # Check if all chunks completed
                        if all(c.status == ChunkStatus.COMPLETED for c in job.chunks):
                            job.status = JobStatus.COMPLETED
                            job.completed_at = time.time()
                            job.final_result = self._aggregate_results(job)
                            self._persist_job(job)
                            return job
                        self._persist_job(job)
                        return None
            return None

    def fail_chunk(self, chunk_id: str, error: str, allow_retry: bool = True) -> Optional[DistributedJob]:
        """
        Marks a chunk as failed and reassigns it to another node, preserving completed chunks.
        """
        with self._lock:
            for job in self._jobs.values():
                for chunk in job.chunks:
                    if chunk.chunk_id == chunk_id:
                        chunk.retries += 1
                        chunk.error = error
                        if allow_retry and chunk.retries < 3:
                            # Put back into PENDING to be reassigned to another healthy worker
                            chunk.status = ChunkStatus.PENDING
                            chunk.assigned_node_id = None
                            job.status = JobStatus.RUNNING
                            logger.info(f"Chunk '{chunk_id}' failed with '{error}'. Re-queued for retry {chunk.retries}/3.")
                        else:
                            chunk.status = ChunkStatus.FAILED
                            job.status = JobStatus.FAILED
                            job.error = f"Chunk '{chunk_id}' failed: {error}"
                        self._persist_job(job)
                        return job
            return None

    def _aggregate_results(self, job: DistributedJob) -> Any:
        """Aggregates all chunk results into a structured result."""
        # Sort chunks by original index to ensure deterministic order
        sorted_chunks = sorted(job.chunks, key=lambda c: c.index)

        # If it was a single un-split chunk, return its result directly
        if len(sorted_chunks) == 1:
            return sorted_chunks[0].result

        # If batch items, concatenate item results
        combined_items = []
        for chk in sorted_chunks:
            res = chk.result
            if isinstance(res, list):
                combined_items.extend(res)
            elif isinstance(res, dict) and "items" in res:
                combined_items.extend(res["items"])
            else:
                combined_items.append(res)

        return {
            "job_id": job.job_id,
            "status": "COMPLETED",
            "total_chunks": len(sorted_chunks),
            "aggregated_items": combined_items
        }

    def get_job(self, job_id: str) -> Optional[DistributedJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            res = []
            for j in self._jobs.values():
                if user_id is None or j.user_id == user_id:
                    res.append(j.to_dict())
            return res
