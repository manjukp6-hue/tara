"""
tests/test_auto_connect_sync.py

Comprehensive test suite for TARA AI Dynamic Auto-Connect & Sync Layer.
Validates:
1. Dynamic, open-ended endpoint registration (no fixed 3-option limit).
2. Health probing, latency tracking, and multi-factor selection scoring.
3. Automatic failover and seamless re-discovery on recovery.
4. Cryptographic state sync with Ed25519 digital signatures.
5. Conflict resolution preventing stale or older data from silently overwriting newer state.
6. Cryptographic secret sanitization filter (private keys and seeds never leak).
7. Offline-first append-only journaling and reconnection flush.
8. TaraBrain integration and REST API server routes.
"""

import os
import sys
import json
import time
import shutil
import tempfile
import unittest
from datetime import datetime, timezone

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
python_dir = os.path.join(REPO_ROOT, "python")
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

from tara_core.auto_connect_sync import (
    EndpointType,
    EndpointCapability,
    EndpointHealthStatus,
    ResourceLimits,
    EndpointDefinition,
    SecretSanitizer,
    SyncPackage,
    SyncPayloadType,
    ConflictResolver,
    OfflineJournalManager,
    EndpointHealthProbe,
    AutoConnectRouter,
    AutoConnectSyncEngine
)
from TARA.ACCESS.crypto.ed25519 import Ed25519
from tara_core.brain import TaraBrain
from tara_core.server import create_server, GLOBAL_API_ROUTER, ApiRouter


class TestAutoConnectSync(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="tara_test_sync_")
        self.journal_path = os.path.join(self.temp_dir, "sync_journal.jsonl")
        self.conflict_path = os.path.join(self.temp_dir, "sync_conflict.jsonl")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_dynamic_endpoint_registration_open_ended(self):
        """Verify dynamic registration without hardcoded 3-option limits."""
        probe = EndpointHealthProbe()
        router = AutoConnectRouter(probe=probe)

        # Diverse endpoints: local PC, home NAS, cloud VM, edge robot, custom cluster
        endpoints = [
            EndpointDefinition(
                endpoint_id="ep-local-pc",
                name="Workstation PC",
                endpoint_type=EndpointType.LOCAL_PC.value,
                base_url="http://192.168.1.10:8080",
                capabilities=[EndpointCapability.INFERENCE.value, EndpointCapability.TRAINING.value],
                priority=90,
                resource_limits=ResourceLimits(cpu_cores=16, ram_mb=32768, gpu_available=True)
            ),
            EndpointDefinition(
                endpoint_id="ep-home-nas",
                name="Synology NAS",
                endpoint_type=EndpointType.NAS.value,
                base_url="http://192.168.1.50:5000",
                capabilities=[EndpointCapability.STORAGE.value, EndpointCapability.SYNC_RELAY.value],
                priority=70,
                resource_limits=ResourceLimits(cpu_cores=4, ram_mb=8192, disk_free_mb=500000)
            ),
            EndpointDefinition(
                endpoint_id="ep-cloud-render",
                name="Render Cloud Instance",
                endpoint_type=EndpointType.CLOUD_VM.value,
                base_url="https://tara-api.onrender.com",
                capabilities=[EndpointCapability.INFERENCE.value, EndpointCapability.COGNITIVE_ORCHESTRATION.value],
                priority=80,
                resource_limits=ResourceLimits(cpu_cores=2, ram_mb=2048)
            ),
            EndpointDefinition(
                endpoint_id="ep-edge-robot-jetson",
                name="Jetson Orin Edge Node",
                endpoint_type=EndpointType.EDGE_ROBOT.value,
                base_url="http://192.168.1.120:9000",
                capabilities=[EndpointCapability.SENSOR_STREAM.value, EndpointCapability.TOOL_EXECUTION.value],
                priority=60,
                resource_limits=ResourceLimits(cpu_cores=8, ram_mb=16384, gpu_available=True, battery_percent=95.0)
            ),
            EndpointDefinition(
                endpoint_id="ep-k8s-cluster",
                name="Private Kubernetes Cluster",
                endpoint_type=EndpointType.KUBERNETES.value,
                base_url="https://k8s.internal:6443",
                capabilities=[EndpointCapability.TRAINING.value, EndpointCapability.INFERENCE.value],
                priority=95,
                resource_limits=ResourceLimits(cpu_cores=64, ram_mb=131072, gpu_available=True)
            ),
            EndpointDefinition(
                endpoint_id="ep-custom-quantum",
                name="Future Compute Endpoint",
                endpoint_type="QUANTUM_NODE_CUSTOM",
                base_url="https://quantum.lab:8000",
                capabilities=["QUANTUM_ANNEALING", EndpointCapability.INFERENCE.value],
                priority=85,
                resource_limits=ResourceLimits(cpu_cores=128, ram_mb=262144)
            )
        ]

        for ep in endpoints:
            router.register_endpoint(ep)

        registered = router.list_endpoints()
        self.assertEqual(len(registered), 6)
        self.assertTrue(any(ep.endpoint_type == "QUANTUM_NODE_CUSTOM" for ep in registered))
        self.assertTrue(any("QUANTUM_ANNEALING" in ep.capabilities for ep in registered))

    def test_health_probe_and_latency_scoring(self):
        """Verify latency tracking, health status updates, and multi-factor scoring."""
        # Simulated prober
        latencies = {"ep-fast": 15.0, "ep-slow": 120.0}

        def mock_prober(ep: EndpointDefinition):
            rtt = latencies.get(ep.endpoint_id, 30.0)
            return True, rtt, None

        probe = EndpointHealthProbe(custom_prober=mock_prober)
        router = AutoConnectRouter(probe=probe)

        ep_fast = EndpointDefinition(
            endpoint_id="ep-fast",
            name="Low Latency Node",
            endpoint_type=EndpointType.LOCAL_DEVICE.value,
            base_url="http://127.0.0.1:8001",
            priority=50,
            resource_limits=ResourceLimits(ram_mb=4096)
        )
        ep_slow = EndpointDefinition(
            endpoint_id="ep-slow",
            name="High Latency Node",
            endpoint_type=EndpointType.CLOUD_VM.value,
            base_url="http://127.0.0.1:8002",
            priority=50,
            resource_limits=ResourceLimits(ram_mb=4096)
        )

        router.register_endpoint(ep_fast)
        router.register_endpoint(ep_slow)

        selected = router.select_best_endpoint(auto_probe=True)
        self.assertIsNotNone(selected)
        self.assertEqual(selected.endpoint_id, "ep-fast")
        self.assertGreater(router.calculate_score(ep_fast), router.calculate_score(ep_slow))

    def test_automatic_failover_and_recovery(self):
        """Simulate endpoint outage, verify instant failover, then recovery and restoration."""
        status_map = {"ep-primary": True, "ep-backup": True}

        def mock_prober(ep: EndpointDefinition):
            is_ok = status_map.get(ep.endpoint_id, True)
            if is_ok:
                return True, 10.0, None
            return False, 0.0, "Connection refused"

        probe = EndpointHealthProbe(max_consecutive_failures=2, custom_prober=mock_prober)
        router = AutoConnectRouter(probe=probe)

        primary = EndpointDefinition(
            endpoint_id="ep-primary",
            name="Primary Compute Node",
            endpoint_type=EndpointType.HOME_SERVER.value,
            base_url="http://192.168.1.100:8080",
            priority=100
        )
        backup = EndpointDefinition(
            endpoint_id="ep-backup",
            name="Backup Compute Node",
            endpoint_type=EndpointType.CLOUD_VM.value,
            base_url="http://192.168.1.200:8080",
            priority=70
        )

        router.register_endpoint(primary)
        router.register_endpoint(backup)

        # 1. Initial election -> Primary
        sel1 = router.select_best_endpoint(auto_probe=True)
        self.assertEqual(sel1.endpoint_id, "ep-primary")

        # 2. Primary goes offline -> Failover to Backup
        status_map["ep-primary"] = False
        sel2 = router.report_endpoint_failure("ep-primary", "Connection timed out")
        self.assertEqual(sel2.endpoint_id, "ep-backup")
        self.assertEqual(len(router.failover_history), 1)
        self.assertEqual(router.failover_history[0]["from_endpoint_id"], "ep-primary")
        self.assertEqual(router.failover_history[0]["to_endpoint_id"], "ep-backup")

        # 3. Primary recovers -> Re-discovery and restoration
        status_map["ep-primary"] = True
        sel3 = router.select_best_endpoint(auto_probe=True)
        self.assertEqual(sel3.endpoint_id, "ep-primary")
        self.assertEqual(len(router.failover_history), 2)
        self.assertEqual(router.failover_history[1]["to_endpoint_id"], "ep-primary")

    def test_state_sync_push_and_conflict_resolution(self):
        """Verify Ed25519 signed state sync and rejection of stale overwrites."""
        resolver = ConflictResolver(log_path=self.conflict_path)
        priv_a, pub_a = Ed25519.generate_keypair()
        keys_map = {"DEVICE-A": pub_a.hex()}

        # 1. Apply valid initial package (seq 10)
        pkg1 = SyncPackage.create(
            source_device_id="DEVICE-A",
            payload_type=SyncPayloadType.KNOWLEDGE.value,
            data={"facts": ["tara_online_ready", "ed25519_verified"]},
            logical_sequence=10,
            signer_private_key_bytes=priv_a
        )
        accepted, reason = resolver.reconcile(pkg1, known_public_keys=keys_map)
        self.assertTrue(accepted)
        self.assertEqual(reason, "APPLIED_REMOTE_UPDATE")

        # 2. Attempt stale update (seq 8 < 10) -> MUST BE REJECTED
        pkg_stale = SyncPackage.create(
            source_device_id="DEVICE-A",
            payload_type=SyncPayloadType.KNOWLEDGE.value,
            data={"facts": ["stale_fact_should_not_overwrite"]},
            logical_sequence=8,
            signer_private_key_bytes=priv_a
        )
        accepted, reason = resolver.reconcile(pkg_stale, known_public_keys=keys_map)
        self.assertFalse(accepted)
        self.assertIn("REJECTED_STALE_OVERWRITE", reason)
        self.assertEqual(len(resolver.conflict_audit_log), 1)

        # 3. Apply newer update (seq 12 > 10) -> MUST SUCCEED
        pkg_newer = SyncPackage.create(
            source_device_id="DEVICE-A",
            payload_type=SyncPayloadType.KNOWLEDGE.value,
            data={"facts": ["latest_knowledge_item"]},
            logical_sequence=12,
            signer_private_key_bytes=priv_a
        )
        accepted, reason = resolver.reconcile(pkg_newer, known_public_keys=keys_map)
        self.assertTrue(accepted)
        self.assertEqual(reason, "APPLIED_REMOTE_UPDATE")

    def test_ed25519_cryptographic_verification(self):
        """Verify Ed25519 signature validation and rejection of tampered packages."""
        resolver = ConflictResolver(log_path=self.conflict_path)
        priv_valid, pub_valid = Ed25519.generate_keypair()
        priv_attacker, pub_attacker = Ed25519.generate_keypair()

        keys_map = {"SECURE-NODE": pub_valid.hex()}

        # Valid signature
        pkg_valid = SyncPackage.create(
            source_device_id="SECURE-NODE",
            payload_type=SyncPayloadType.SKILLS.value,
            data={"skill": "autonomous_navigation"},
            logical_sequence=1,
            signer_private_key_bytes=priv_valid
        )
        accepted, reason = resolver.reconcile(pkg_valid, known_public_keys=keys_map)
        self.assertTrue(accepted)

        # Tampered signature (signed with attacker key)
        pkg_tampered = SyncPackage.create(
            source_device_id="SECURE-NODE",
            payload_type=SyncPayloadType.SKILLS.value,
            data={"skill": "malicious_injected_skill"},
            logical_sequence=2,
            signer_private_key_bytes=priv_attacker
        )
        accepted, reason = resolver.reconcile(pkg_tampered, known_public_keys=keys_map)
        self.assertFalse(accepted)
        self.assertEqual(reason, "CRYPTOGRAPHIC_SIGNATURE_MISMATCH")

        # Corrupted hash
        pkg_corrupt = SyncPackage.create(
            source_device_id="SECURE-NODE",
            payload_type=SyncPayloadType.SKILLS.value,
            data={"skill": "corrupted_payload"},
            logical_sequence=3,
            signer_private_key_bytes=priv_valid
        )
        pkg_corrupt.data["skill"] = "modified_after_signing"
        accepted, reason = resolver.reconcile(pkg_corrupt, known_public_keys=keys_map)
        self.assertFalse(accepted)
        self.assertEqual(reason, "STATE_HASH_CORRUPTION")

    def test_sensitive_secret_sanitization(self):
        """Verify that private keys, seeds, and biometric data are never included in sync payloads."""
        raw_payload = {
            "node_name": "TARA-PROD-1",
            "private_key": "c3e7f...super_secret_private_key",
            "creator_seed": "abandon ability able about above absent absorb abstract",
            "biometric_data": {"fingerprint_hash": "a1b2c3d4"},
            "config": {
                "api_secret": "my_hidden_jwt_secret",
                "max_concurrency": 8
            },
            "public_key": "383414ba3932c09285222288642dd6ac6e013d42ef37322eb0dc8418f3877f5a"
        }

        clean_payload = SecretSanitizer.sanitize(raw_payload)

        # Sensitive keys sanitized
        self.assertEqual(clean_payload["private_key"], "[REDACTED_SECURITY_POLICY]")
        self.assertEqual(clean_payload["creator_seed"], "[REDACTED_SECURITY_POLICY]")
        self.assertEqual(clean_payload["biometric_data"], "[REDACTED_SECURITY_POLICY]")
        self.assertEqual(clean_payload["config"]["api_secret"], "[REDACTED_SECURITY_POLICY]")

        # Safe keys preserved
        self.assertEqual(clean_payload["node_name"], "TARA-PROD-1")
        self.assertEqual(clean_payload["config"]["max_concurrency"], 8)
        self.assertEqual(clean_payload["public_key"], "383414ba3932c09285222288642dd6ac6e013d42ef37322eb0dc8418f3877f5a")

    def test_offline_first_journaling_and_reconnection_flush(self):
        """Verify offline queueing when network is disconnected and sequential flush on reconnect."""
        journal = OfflineJournalManager(journal_path=self.journal_path)
        self.assertEqual(journal.queue_depth(), 0)

        # Stage 3 packages while offline
        for i in range(1, 4):
            pkg = SyncPackage.create(
                source_device_id="OFFLINE-NODE",
                payload_type=SyncPayloadType.MEMORY.value,
                data={"memory_entry": f"trace_{i}"},
                logical_sequence=i
            )
            journal.stage_package(pkg)

        self.assertEqual(journal.queue_depth(), 3)
        staged = journal.peek_staged()
        self.assertEqual(len(staged), 3)
        self.assertEqual(staged[0].data["memory_entry"], "trace_1")
        self.assertEqual(staged[2].data["memory_entry"], "trace_3")

        # Drain and flush
        drained = journal.drain_staged()
        self.assertEqual(len(drained), 3)
        self.assertEqual(journal.queue_depth(), 0)

    def test_tarabrain_auto_sync_integration(self):
        """Verify seamless integration with TaraBrain kernel."""
        brain = TaraBrain()
        self.assertTrue(hasattr(brain, "auto_sync"))
        self.assertTrue(hasattr(brain, "auto_connect_router"))

        # Test active endpoint retrieval
        ep = brain.get_active_compute_endpoint()
        self.assertIsNotNone(ep)
        self.assertIn("endpoint_id", ep)

        # Test sync_state method
        sync_result = brain.sync_state(
            payload_type=SyncPayloadType.CONFIG.value,
            data={"theme": "dark", "autonomous_mode": True}
        )
        self.assertTrue(sync_result["success"])
        self.assertIn("package_id", sync_result)
        self.assertGreater(sync_result["sequence"], 0)

    def test_server_rest_api_endpoints(self):
        """Verify REST API endpoints on TaraRequestHandler."""
        brain = TaraBrain()
        server = create_server(host="127.0.0.1", port=8099, brain=brain)
        router = server.router

        routes = [r["path"] for r in router.list_routes()]
        self.assertIn("/api/v1/endpoints/register", routes)
        self.assertIn("/api/v1/endpoints/list", routes)
        self.assertIn("/api/v1/endpoints/select", routes)
        self.assertIn("/api/v1/endpoints/probe", routes)
        self.assertIn("/api/v1/sync/push", routes)
        self.assertIn("/api/v1/sync/status", routes)
        self.assertIn("/api/v1/sync/flush", routes)

        # Test direct route handler invocation
        class MockHandler:
            def __init__(self, srv):
                self.server = srv

        h = MockHandler(server)

        try:
            # 1. List endpoints
            list_route = router.match("GET", "/api/v1/endpoints/list")
            res_list = list_route["handler"](h, {})
            self.assertEqual(res_list["status"], "SUCCESS")
            self.assertIsInstance(res_list["endpoints"], list)

            # 2. Register endpoint
            reg_route = router.match("POST", "/api/v1/endpoints/register")
            new_ep_payload = {
                "endpoint_id": "test-remote-vm",
                "name": "Test Cloud Worker",
                "endpoint_type": EndpointType.CLOUD_VM.value,
                "base_url": "http://10.0.0.5:8080",
                "capabilities": [EndpointCapability.INFERENCE.value],
                "priority": 75
            }
            res_reg = reg_route["handler"](h, new_ep_payload)
            self.assertEqual(res_reg["status"], "SUCCESS")
            self.assertEqual(res_reg["endpoint"]["endpoint_id"], "test-remote-vm")

            # 3. Status
            status_route = router.match("GET", "/api/v1/sync/status")
            res_status = status_route["handler"](h, {})
            self.assertEqual(res_status["status"], "SUCCESS")
            self.assertIn("sync_status", res_status)
            self.assertIn("registered_endpoints_count", res_status["sync_status"])
        finally:
            server.server_close()

    def test_remote_mtls_enforcement_and_audit(self):
        """
        Verify that remote HTTPS endpoints strictly require valid CA and client certificate
        authentication, never silently fall back, and deny/audit unauthorized attempts.
        """
        probe = EndpointHealthProbe()

        # 1. Remote HTTPS endpoint without configured mTLS certificates
        remote_ep = EndpointDefinition(
            endpoint_id="ep-remote-cloud-secure",
            name="Remote Secure Cluster",
            endpoint_type=EndpointType.CLOUD_VM.value,
            base_url="https://secure-node.tara-ai.internal:8443",
            capabilities=[EndpointCapability.INFERENCE.value],
            priority=85
        )

        status, rtt, err = probe.probe(remote_ep)
        self.assertEqual(status, EndpointHealthStatus.UNREACHABLE)
        self.assertIn("DENIED", err)
        self.assertIn("mTLS", err)

        # Verify audit log recorded the failure
        audit_events = [e for e in probe.security_audit_log if e.get("event") == "AUDIT_REMOTE_MTLS_FAILED"]
        self.assertTrue(len(audit_events) >= 1)
        self.assertEqual(audit_events[-1]["endpoint_id"], "ep-remote-cloud-secure")

        # 2. Local loopback mode remains permitted without mTLS
        loopback_ep = EndpointDefinition(
            endpoint_id="ep-local-loopback",
            name="Local Loopback Process",
            endpoint_type=EndpointType.LOCAL_PROCESS.value,
            base_url="",
            capabilities=[EndpointCapability.INFERENCE.value]
        )
        status_lb, _, err_lb = probe.probe(loopback_ep)
        self.assertEqual(status_lb, EndpointHealthStatus.HEALTHY)
        self.assertIsNone(err_lb)


if __name__ == "__main__":
    unittest.main()
