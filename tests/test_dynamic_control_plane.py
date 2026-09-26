"""
Comprehensive Test Suite for TARA Dynamic Compute Control Plane
Verifies:
1. Provider-Neutral Adapter Layer (Cloudflare, ModelScope, HuggingFace, Render, GenericContainer, LocalDevice, FutureProvider)
2. Dynamic Worker Registry & Lifecycle (9 states, multi-factor scoring, quarantine, revoke)
3. Model SHA256 Invariant & Canonical Manifest Differential Sync
4. Distributed Job Engine (planning, chunk leasing, failure reassignment, result aggregation)
5. User Model Lifecycle & Zero-Trust Privilege Hierarchy
6. Memory Isolation & Session Integrity
"""

import os
import sys
import unittest
import json
import time
import tempfile
import shutil

# Ensure workspace root and python dirs are in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON_DIR = os.path.join(REPO_ROOT, "python")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)

from tara_core.compute.provider_adapters import (
    ProviderManager,
    CloudflareAdapter,
    ModelScopeAdapter,
    HuggingFaceAdapter,
    RenderAdapter,
    GenericContainerAdapter,
    LocalDeviceAdapter,
    FutureProviderAdapter,
    DeploymentType,
)
from tara_core.control_plane.worker_registry import (
    DynamicWorkerRegistry,
    ClusterWorkerNode,
    WorkerState,
    CANONICAL_MODEL_SHA256,
)
from tara_core.contracts import CANONICAL_PROTOCOL_VERSION
from tara_core.control_plane.distributed_job_engine import (
    DistributedJobEngine,
    JobPlanner,
    ChunkStatus,
    JobStatus,
)
from tara_core.control_plane.sync_manager import (
    AutoSyncManager,
    SyncEvaluationResult,
)
from tara_core.control_plane.secure_chat import (
    SecureChatManager,
    AuthorityTier,
    ChatSecurityContext,
)
from tara_core.user_model import UserManager, UserProfile, UserRole


class TestProviderAdapters(unittest.TestCase):

    """Tests the provider-neutral adapter layer."""

    def setUp(self):
        self.manager = ProviderManager()

    def test_all_seven_providers_registered(self):
        providers = self.manager.list_providers()
        names = {p["name"] for p in providers}
        expected = {
            "cloudflare",
            "modelscope",
            "huggingface",
            "render",
            "generic_container",
            "local_device",
            "future_provider",
        }
        self.assertEqual(names, expected)

    def test_provider_capability_detection(self):
        for p in self.manager.list_providers():
            adapter = self.manager.get_adapter(p["name"])
            self.assertIsNotNone(adapter)
            self.assertGreater(len(adapter.spec.capabilities), 0)
            status, reason = adapter.check_availability()
            self.assertIsNotNone(status)

    def test_provider_deployment_and_fallback(self):
        res = self.manager.deploy_with_fallback(
            preferred_provider_name="local_device",
            deployment_type=DeploymentType.LOCAL_DEVICE,
            worker_config={"concurrency": 2},
        )
        self.assertEqual(res["provider"], "local_device")
        self.assertIn("worker_id", res)
        self.assertEqual(res["status"], "READY")

    def test_generic_container_deployment(self):
        adapter = self.manager.get_adapter("generic_container")
        self.assertIsNotNone(adapter)
        res = adapter.deploy_worker({"image": "tara:latest"})
        self.assertEqual(res["provider"], "generic_container")
        self.assertIn("worker_id", res)


class TestWorkerRegistryLifecycle(unittest.TestCase):
    """Tests dynamic worker registry, 9-stage lifecycle, and quarantine."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.persist_file = os.path.join(self.tmp_dir, "test_workers.json")
        self.registry = DynamicWorkerRegistry(persistence_file=self.persist_file)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_worker_registration_valid_model(self):
        node = self.registry.discover_node({
            "node_id": "cf_worker_01",
            "endpoint_url": "https://cf-worker.internal",
            "provider": "cloudflare",
            "deployment_type": "SERVERLESS_EDGE",
            "cpu_capacity": 2.0,
            "has_gpu": False
        })
        self.assertEqual(node.state, WorkerState.DISCOVERED)

        # Authenticate
        auth_ok = self.registry.authenticate_node("cf_worker_01", "tok_secret", "tok_secret")
        self.assertTrue(auth_ok)
        self.assertEqual(node.state, WorkerState.SYNCING)

        # Verify valid model SHA -> READY
        ready_ok = self.registry.verify_and_set_ready(
            "cf_worker_01",
            reported_model_sha256=CANONICAL_MODEL_SHA256,
            reported_protocol_version=CANONICAL_PROTOCOL_VERSION
        )
        self.assertTrue(ready_ok)
        self.assertEqual(node.state, WorkerState.READY)
        self.assertIsNone(node.quarantine_reason)

    def test_worker_registration_invalid_model_rejected_from_ready(self):
        self.registry.discover_node({
            "node_id": "corrupt_node_01",
            "endpoint_url": "http://10.0.0.9:8080",
            "provider": "generic_container",
            "deployment_type": "CONTAINER"
        })
        self.registry.authenticate_node("corrupt_node_01", "tok_1", "tok_1")

        # Try to verify with bad model SHA
        ready_ok = self.registry.verify_and_set_ready(
            "corrupt_node_01",
            reported_model_sha256="bad_sha_hash_value_12345",
            reported_protocol_version=CANONICAL_PROTOCOL_VERSION
        )
        self.assertFalse(ready_ok)
        node = self.registry.get_node("corrupt_node_01")
        self.assertNotEqual(node.state, WorkerState.READY)
        self.assertIn("mismatch", node.quarantine_reason.lower())

    def test_worker_quarantine_and_revoke(self):
        self.registry.discover_node({
            "node_id": "test_node_render",
            "endpoint_url": "https://render.internal",
            "provider": "render",
            "deployment_type": "CONTAINER"
        })
        self.registry.authenticate_node("test_node_render", "tok_2", "tok_2")
        self.registry.verify_and_set_ready("test_node_render", CANONICAL_MODEL_SHA256, CANONICAL_PROTOCOL_VERSION)

        node = self.registry.get_node("test_node_render")
        self.assertEqual(node.state, WorkerState.READY)

        # Quarantine
        self.registry.quarantine_worker("test_node_render", "Suspicious load anomaly")
        self.assertEqual(node.state, WorkerState.QUARANTINED)
        self.assertEqual(node.quarantine_reason, "Suspicious load anomaly")

        # Revoke
        self.registry.revoke_worker("test_node_render", "Decommissioned")
        self.assertEqual(node.state, WorkerState.REVOKED)

    def test_multi_factor_scoring_selection(self):
        # Create two ready nodes: one fast, one loaded
        self.registry.discover_node({
            "node_id": "node_fast",
            "endpoint_url": "http://fast.internal",
            "provider": "local_device",
            "cpu_capacity": 4.0,
            "average_latency_ms": 10.0,
            "current_load": 0.1
        })
        self.registry.authenticate_node("node_fast", "tok", "tok")
        self.registry.verify_and_set_ready("node_fast", CANONICAL_MODEL_SHA256, CANONICAL_PROTOCOL_VERSION)

        self.registry.discover_node({
            "node_id": "node_busy",
            "endpoint_url": "http://busy.internal",
            "provider": "render",
            "cpu_capacity": 1.0,
            "average_latency_ms": 100.0,
            "current_load": 0.9,
            "queue_depth": 5
        })
        self.registry.authenticate_node("node_busy", "tok", "tok")
        self.registry.verify_and_set_ready("node_busy", CANONICAL_MODEL_SHA256, CANONICAL_PROTOCOL_VERSION)

        best = self.registry.select_best_worker()
        self.assertIsNotNone(best)
        self.assertEqual(best.node_id, "node_fast")


class TestDistributedJobEngine(unittest.TestCase):
    """Tests Distributed Job Engine: sharding, leasing, retries, aggregation."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.registry = DynamicWorkerRegistry(persistence_file=os.path.join(self.tmp_dir, "workers.json"))
        # Register a ready node
        self.registry.discover_node({
            "node_id": "worker_01",
            "endpoint_url": "http://worker01.internal",
            "provider": "local_device",
            "supported_workloads": ["INFERENCE", "NORMAL_CHAT"]
        })
        self.registry.authenticate_node("worker_01", "tok", "tok")
        self.registry.verify_and_set_ready("worker_01", CANONICAL_MODEL_SHA256, CANONICAL_PROTOCOL_VERSION)

        self.engine = DistributedJobEngine(
            worker_registry=self.registry,
            storage_dir=os.path.join(self.tmp_dir, "jobs")
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_job_planner_no_oversharding_small_workload(self):
        # 1 item -> should NOT split
        should_split, chunks = JobPlanner.plan("INFERENCE", {"items": [{"prompt": "test"}]}, available_nodes_count=4)
        self.assertFalse(should_split)
        self.assertEqual(len(chunks), 1)

        # 8 items with 4 nodes -> should split into 4 chunks
        should_split, chunks = JobPlanner.plan("INFERENCE", {"items": [{"id": i} for i in range(8)]}, available_nodes_count=4)
        self.assertTrue(should_split)
        self.assertEqual(len(chunks), 4)

    def test_idempotent_job_submission(self):
        job1 = self.engine.submit_job(
            user_id="user_test",
            request_id="req_001",
            task_type="INFERENCE",
            payload={"items": [{"prompt": "hi"}]},
            idempotency_key="key_abc_123",
        )
        job2 = self.engine.submit_job(
            user_id="user_test",
            request_id="req_002",
            task_type="INFERENCE",
            payload={"items": [{"prompt": "different"}]},
            idempotency_key="key_abc_123",
        )
        self.assertEqual(job1.job_id, job2.job_id)

    def test_chunk_leasing_and_completion(self):
        items = [{"id": i} for i in range(6)]
        job = self.engine.submit_job(
            user_id="user_test",
            request_id="req_003",
            task_type="INFERENCE",
            payload={"items": items}
        )
        self.assertEqual(job.status, JobStatus.PENDING)

        # Assign chunks
        assignments = self.engine.assign_next_chunks()
        self.assertGreater(len(assignments), 0)
        self.assertEqual(job.status, JobStatus.RUNNING)

        # Complete each chunk
        for chunk in job.chunks:
            self.engine.complete_chunk(
                chunk.chunk_id,
                result={"items": [f"result_{chunk.index}"]}
            )

        finished = self.engine.get_job(job.job_id)
        self.assertEqual(finished.status, JobStatus.COMPLETED)
        self.assertIsNotNone(finished.final_result)

    def test_chunk_failure_and_retry_reassignment(self):
        job = self.engine.submit_job(
            user_id="user_test",
            request_id="req_004",
            task_type="INFERENCE",
            payload={"prompt": "test retry"}
        )
        self.engine.assign_next_chunks()
        chunk_id = job.chunks[0].chunk_id

        # Fail chunk once: should return to PENDING with retries=1
        self.engine.fail_chunk(chunk_id, "Temporary socket timeout")
        job_state = self.engine.get_job(job.job_id)
        chunk_state = job_state.chunks[0]
        self.assertEqual(chunk_state.status, ChunkStatus.PENDING)
        self.assertEqual(chunk_state.retries, 1)

        # Fail twice more -> reaches 3 retries -> FAILED
        self.engine.fail_chunk(chunk_id, "Error 2")
        self.engine.fail_chunk(chunk_id, "Error 3")
        failed_job = self.engine.get_job(job.job_id)
        self.assertEqual(failed_job.status, JobStatus.FAILED)
        self.assertEqual(failed_job.chunks[0].status, ChunkStatus.FAILED)


class TestSyncManager(unittest.TestCase):
    """Tests canonical manifest differential sync."""

    def setUp(self):
        self.sync_mgr = AutoSyncManager()

    def test_canonical_manifest_loads_and_verifies(self):
        manifest = self.sync_mgr.load_canonical_manifest()
        self.assertIn("artifacts", manifest)
        self.assertEqual(manifest["canonical_model_sha256"], CANONICAL_MODEL_SHA256)

        artifacts = {
            "storage/models/tara/model.safetensors": CANONICAL_MODEL_SHA256,
            "storage/models/tara/config.json": manifest["artifacts"]["model_config"]["sha256"],
            "storage/models/tara/tokenizer.json": manifest["artifacts"]["tokenizer"]["sha256"],
            "TARA/CONTRACTS/v1/schemas.json": manifest["artifacts"]["contract_schemas"]["sha256"],
            "python/tara_core/contracts.py": manifest["artifacts"]["python_contracts"]["sha256"],
        }
        res = self.sync_mgr.evaluate_worker(artifacts, reported_protocol=CANONICAL_PROTOCOL_VERSION)
        self.assertTrue(res.is_fully_synchronized)
        self.assertTrue(res.is_model_verified)

    def test_tampered_model_rejected(self):
        tampered_artifacts = {
            "storage/models/tara/model.safetensors": "tampered_sha256_hash_99999",
        }
        res = self.sync_mgr.evaluate_worker(tampered_artifacts, reported_protocol=CANONICAL_PROTOCOL_VERSION)
        self.assertFalse(res.is_model_verified)
        self.assertFalse(res.is_fully_synchronized)
        self.assertIn("Model weights checksum mismatch", res.reason)


class TestUserModelAndSecurity(unittest.TestCase):
    """Tests User Model lifecycle, memory isolation, and authority hierarchy."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.user_mgr = UserManager(storage_file=os.path.join(self.tmp_dir, "users.json"))
        self.chat_mgr = SecureChatManager()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_user_lifecycle_transitions(self):
        user = self.user_mgr.register_user(UserProfile(user_id="user_101", display_name="Alice"))
        self.assertEqual(user.status, "active")

        # Suspend
        self.user_mgr.set_user_status("user_101", "suspended")
        self.assertEqual(self.user_mgr.get_user("user_101").status, "suspended")

        # Block
        self.user_mgr.set_user_status("user_101", "blocked")
        self.assertEqual(self.user_mgr.get_user("user_101").status, "blocked")

        # Delete
        self.user_mgr.delete_user("user_101")
        self.assertEqual(self.user_mgr.get_user("user_101").status, "deleted")

    def test_device_pairing_and_privacy_export(self):
        user = self.user_mgr.register_user(UserProfile(
            user_id="user_202",
            display_name="Bob",
            device_bindings=["dev_laptop"]
        ))
        self.assertIn("dev_laptop", user.device_bindings)

        export = self.user_mgr.export_user_data("user_202")
        self.assertIsNotNone(export)
        self.assertEqual(export["profile"]["user_id"], "user_202")

    def test_authority_hierarchy_enforcement(self):
        # SYSTEM_SECURITY > PROTECTED_CREATOR > NORMAL_CREATOR > AUTHENTICATED_USER > SKILL_OR_TOOL
        self.assertGreater(AuthorityTier.SYSTEM_SECURITY, AuthorityTier.PROTECTED_CREATOR)
        self.assertGreater(AuthorityTier.PROTECTED_CREATOR, AuthorityTier.NORMAL_CREATOR)
        self.assertGreater(AuthorityTier.NORMAL_CREATOR, AuthorityTier.AUTHENTICATED_USER)
        self.assertGreater(AuthorityTier.AUTHENTICATED_USER, AuthorityTier.SKILL_OR_TOOL)

        # Creator authority allowed creator action
        allowed, msg = self.chat_mgr.validate_authority_hierarchy(
            actor_tier=AuthorityTier.NORMAL_CREATOR,
            action_required_tier=AuthorityTier.NORMAL_CREATOR,
        )
        self.assertTrue(allowed)

        # Regular user denied creator-level action
        denied, msg = self.chat_mgr.validate_authority_hierarchy(
            actor_tier=AuthorityTier.AUTHENTICATED_USER,
            action_required_tier=AuthorityTier.NORMAL_CREATOR,
        )
        self.assertFalse(denied)
        self.assertIn("Privilege escalation blocked", msg)

    def test_memory_isolation_per_user(self):
        # Alice cannot access Bob's memory
        can_access = self.chat_mgr.validate_memory_access("alice", "user_bob")
        self.assertFalse(can_access)

        # Alice can access her own memory
        can_access_self = self.chat_mgr.validate_memory_access("alice", "user_alice")
        self.assertTrue(can_access_self)

    def test_message_integrity_and_replay_protection(self):
        sess = self.chat_mgr.create_chat_session("alice")
        mac = sess.generate_message_hmac("Hello TARA", nonce=1)

        # Valid message
        self.assertTrue(sess.verify_message_integrity("Hello TARA", nonce=1, received_hmac=mac))

        # Replay attack with same nonce -> rejected!
        self.assertFalse(sess.verify_message_integrity("Hello TARA", nonce=1, received_hmac=mac))

        # Out-of-order nonce -> rejected!
        mac_old = sess.generate_message_hmac("Old message", nonce=0)
        self.assertFalse(sess.verify_message_integrity("Old message", nonce=0, received_hmac=mac_old))


if __name__ == "__main__":
    unittest.main()
