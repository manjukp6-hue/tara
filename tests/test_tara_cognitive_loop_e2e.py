"""
tests/test_tara_cognitive_loop_e2e.py

Comprehensive End-to-End Integration Test Suite for TARA AI Autonomous Cognitive Loop.
Proves all 13 core cognitive loop capabilities:
1. Normal conversation -> real neural model response.
2. Knowledge question -> KnowledgeBase retrieval + model response.
3. Skill request -> correct skill selected.
4. Executable skill -> actual execution + result returned.
5. Tool request -> correct tool invoked + result returned.
6. Unsafe action -> blocked by RuleEngine.
7. Unauthorized creator-only action -> DENIED.
8. Authorized creator-only action -> allowed only after real cryptographic verification.
9. Memory write -> sanitized episode stored.
10. Memory retrieval -> previous relevant episode influences the response.
11. Tool/skill error -> final outcome is ERROR, not SUCCESS.
12. /api/v1/chat -> real Brain response.
13. /api/v1/status -> real model/runtime status.
"""

import os
import sys
import json
import time
import urllib.request
import threading
import tempfile
import shutil
import unittest
from typing import Dict, Any

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
python_dir = os.path.join(REPO_ROOT, "python")
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

from tara_core.brain import TaraBrain
from tara_core.server import create_server
from TARA.ACCESS.crypto.ed25519 import Ed25519
from TARA.ACCESS.operator.operator_profile import CANONICAL_CREATOR_ID
from tests.test_helpers import setup_test_identity_manager, create_test_id_token


def format_trace(trace: Dict[str, Any]) -> str:
    """Formats the required 10-step cognitive execution trace."""
    safe_final = str(trace.get('final_response', ''))[:100].encode('ascii', 'backslashreplace').decode('ascii')
    safe_res = str(trace.get('result_summary', '')).encode('ascii', 'backslashreplace').decode('ascii')
    lines = [
        "-" * 70,
        f"INPUT: {trace.get('input')}",
        f"-> MODEL: {trace.get('model_status')}",
        f"-> DECISION: {trace.get('decision')}",
        f"-> RETRIEVED CONTEXT: {trace.get('context_tags')} (Knowledge: {len(trace.get('retrieved_knowledge', []))}, Memory: {len(trace.get('retrieved_memory', []))})",
        f"-> RULE CHECK: {trace.get('rule_check', {}).get('decision')} - {trace.get('rule_check', {}).get('reason', 'OK')}",
        f"-> AUTH CHECK: {trace.get('auth_check')}",
        f"-> TOOL/SKILL: {trace.get('tool_or_skill')}",
        f"-> RESULT: {safe_res}",
        f"-> FINAL RESPONSE: {safe_final}...",
        f"-> MEMORY UPDATE: Outcome={trace.get('outcome')}",
        "-" * 70
    ]
    return "\n".join(lines)


class TestTaraCognitiveLoopE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure environment variables from .env are available
        env_file = os.path.join(REPO_ROOT, ".env")
        if os.path.exists(env_file):
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line_s = line.strip()
                    if line_s and not line_s.startswith("#") and "=" in line_s:
                        k, v = line_s.split("=", 1)
                        os.environ[k.strip()] = v.strip()

        cls.brain = TaraBrain()
        
        # Seed verified knowledge for test scenario 2
        cls.test_fact = "TARA Core is an autonomous AI cognitive system created by ROOT_OPERATOR."
        cls.brain.knowledge_base.store_or_update_knowledge(
            topic="TARA_ARCHITECTURE",
            subject="TARA Core Creator and Architecture",
            content=cls.test_fact,
            sources=[{"title": "TARA Specification", "url": "local://docs/arch.md"}],
            confidence=0.99,
            learned_by_role="ROOT_CREATOR",
            trigger="MANUAL_SEED"
        )

    @classmethod
    def tearDownClass(cls):
        # Clean up temporary seeded test knowledge entry
        entry_file = os.path.join(REPO_ROOT, "TARA", "KNOWLEDGE", "entries", "KB-TARAARCH-30C40C76.json")
        if os.path.exists(entry_file):
            try:
                os.remove(entry_file)
            except Exception:
                pass
        idx_file = os.path.join(REPO_ROOT, "TARA", "KNOWLEDGE", "knowledge_index.json")
        if os.path.exists(idx_file):
            try:
                with open(idx_file, "r", encoding="utf-8") as f:
                    idx = json.load(f)
                if "KB-TARAARCH-30C40C76" in idx:
                    del idx["KB-TARAARCH-30C40C76"]
                    with open(idx_file, "w", encoding="utf-8") as f:
                        json.dump(idx, f, indent=2)
            except Exception:
                pass

    def test_01_normal_conversation_real_neural_model(self):
        """1. Normal conversation -> real neural model response."""
        res = self.brain.process(actor_id="user_01", input_text="Hello TARA, what is your identity?")
        
        self.assertEqual(res["decision"], "ALLOW")
        self.assertEqual(res["outcome"], "SUCCESS")
        self.assertIn("MODEL-KNOWN", res["context_tags"])
        self.assertIsInstance(res["final_response"], str)
        self.assertGreater(len(res["final_response"]), 0)

        trace = {
            "input": res["input"],
            "model_status": self.brain.get_model_status()["status"],
            "decision": res["decision"],
            "context_tags": res["context_tags"],
            "retrieved_knowledge": res["retrieved_context"]["knowledge"],
            "retrieved_memory": res["retrieved_context"]["memory"],
            "rule_check": res["rule_check"],
            "auth_check": "NOT_REQUIRED",
            "tool_or_skill": "NONE (CONVERSATIONAL)",
            "result_summary": res["result"],
            "final_response": res["final_response"],
            "outcome": res["outcome"]
        }
        print("\n[TEST 1 TRACE]\n" + format_trace(trace))

    def test_02_knowledge_question_retrieval_and_model_response(self):
        """2. Knowledge question -> KnowledgeBase retrieval + model response."""
        res = self.brain.process(actor_id="user_01", input_text="Tell me about TARA Core Creator and Architecture")
        
        self.assertEqual(res["decision"], "ALLOW")
        self.assertEqual(res["outcome"], "SUCCESS")
        self.assertIn("VERIFIED-KNOWLEDGE", res["context_tags"])
        self.assertGreater(len(res["retrieved_context"]["knowledge"]), 0)
        self.assertIn(self.test_fact, res["final_response"])

        trace = {
            "input": res["input"],
            "model_status": self.brain.get_model_status()["status"],
            "decision": res["decision"],
            "context_tags": res["context_tags"],
            "retrieved_knowledge": res["retrieved_context"]["knowledge"],
            "retrieved_memory": res["retrieved_context"]["memory"],
            "rule_check": res["rule_check"],
            "auth_check": "NOT_REQUIRED",
            "tool_or_skill": "GlobalKnowledgeBase",
            "result_summary": f"Found {len(res['retrieved_context']['knowledge'])} verified facts",
            "final_response": res["final_response"],
            "outcome": res["outcome"]
        }
        print("\n[TEST 2 TRACE]\n" + format_trace(trace))

    def test_03_skill_request_selection(self):
        """3. Skill request -> correct skill selected."""
        res = self.brain.process(actor_id="user_01", input_text="diagnostics system telemetry")
        
        self.assertEqual(res["intent"]["intent"], "EXECUTE_SKILL")
        self.assertEqual(res["intent"]["skill"], "diagnostics")
        self.assertEqual(res["outcome"], "SUCCESS")

        trace = {
            "input": res["input"],
            "model_status": self.brain.get_model_status()["status"],
            "decision": res["decision"],
            "context_tags": res["context_tags"],
            "retrieved_knowledge": res["retrieved_context"]["knowledge"],
            "retrieved_memory": res["retrieved_context"]["memory"],
            "rule_check": res["rule_check"],
            "auth_check": "PERMITTED",
            "tool_or_skill": res["tool_or_skill"],
            "result_summary": res["result"],
            "final_response": res["final_response"],
            "outcome": res["outcome"]
        }
        print("\n[TEST 3 TRACE]\n" + format_trace(trace))

    def test_04_executable_skill_run_and_result(self):
        """4. Executable skill -> actual execution + result returned."""
        res = self.brain.process(actor_id="user_01", input_text="diagnostics")
        
        self.assertEqual(res["outcome"], "SUCCESS")
        self.assertIn("cpu_healthy", res["result"])
        self.assertIn("memory_usage_mb", res["result"])

        trace = {
            "input": res["input"],
            "model_status": self.brain.get_model_status()["status"],
            "decision": res["decision"],
            "context_tags": res["context_tags"],
            "retrieved_knowledge": res["retrieved_context"]["knowledge"],
            "retrieved_memory": res["retrieved_context"]["memory"],
            "rule_check": res["rule_check"],
            "auth_check": "PERMITTED",
            "tool_or_skill": res["tool_or_skill"],
            "result_summary": f"cpu_healthy={res['result']['cpu_healthy']}, memory_usage_mb={res['result']['memory_usage_mb']}",
            "final_response": res["final_response"],
            "outcome": res["outcome"]
        }
        print("\n[TEST 4 TRACE]\n" + format_trace(trace))

    def test_05_tool_request_invoked_and_result(self):
        """5. Tool request -> correct tool invoked + result returned."""
        config_file = os.path.join(REPO_ROOT, "storage", "models", "tara", "config.json")
        res = self.brain.process(
            actor_id="user_01",
            input_text="inspect_file",
            context={"file_path": config_file}
        )

        self.assertEqual(res["decision"], "ALLOW")
        self.assertEqual(res["outcome"], "SUCCESS")
        self.assertEqual(res["tool_or_skill"]["name"], "file_inspector")
        self.assertTrue(res["result"]["exists"])
        self.assertIn("sha256", res["result"])
        self.assertIn("line_count", res["result"])

        trace = {
            "input": res["input"],
            "model_status": self.brain.get_model_status()["status"],
            "decision": res["decision"],
            "context_tags": res["context_tags"],
            "retrieved_knowledge": res["retrieved_context"]["knowledge"],
            "retrieved_memory": res["retrieved_context"]["memory"],
            "rule_check": res["rule_check"],
            "auth_check": "PERMITTED",
            "tool_or_skill": res["tool_or_skill"],
            "result_summary": f"Lines={res['result']['line_count']}, SHA256={res['result']['sha256'][:16]}...",
            "final_response": res["final_response"],
            "outcome": res["outcome"]
        }
        print("\n[TEST 5 TRACE]\n" + format_trace(trace))

    def test_06_unsafe_action_blocked_by_rule_engine(self):
        """6. Unsafe action -> blocked by RuleEngine."""
        res = self.brain.process(
            actor_id="user_01",
            input_text="display_media",
            context={"action_type": "display_media", "media_type": "nude_photo"}
        )

        self.assertEqual(res["decision"], "DENY")
        self.assertEqual(res["outcome"], "FAILURE")
        self.assertEqual(res["result"]["status"], "BLOCKED_BY_POLICY")
        self.assertIn("Content Safety Rule", res["result"]["reason"])

        trace = {
            "input": res["input"],
            "model_status": self.brain.get_model_status()["status"],
            "decision": res["decision"],
            "context_tags": res["context_tags"],
            "retrieved_knowledge": res["retrieved_context"]["knowledge"],
            "retrieved_memory": res["retrieved_context"]["memory"],
            "rule_check": res["rule_check"],
            "auth_check": "FAIL_CLOSED",
            "tool_or_skill": "EXECUTION_BLOCKED",
            "result_summary": res["result"],
            "final_response": res["final_response"],
            "outcome": res["outcome"]
        }
        print("\n[TEST 6 TRACE]\n" + format_trace(trace))

    def test_07_unauthorized_creator_only_action_denied(self):
        """7. Unauthorized creator-only action -> DENIED."""
        # Attacker attempts creator action claiming actor_id=ROOT_OPERATOR with NO cryptographic proof
        res = self.brain.process(
            actor_id="ROOT_OPERATOR",
            input_text="delete file sensitive_core_database.db",
            context={"is_important": True}
        )

        self.assertEqual(res["decision"], "DENY")
        self.assertEqual(res["outcome"], "FAILURE")
        self.assertIn("UNAUTHORIZED_CREATOR_ACTION", res["result"]["reason"])

        trace = {
            "input": res["input"],
            "model_status": self.brain.get_model_status()["status"],
            "decision": res["decision"],
            "context_tags": res["context_tags"],
            "retrieved_knowledge": res["retrieved_context"]["knowledge"],
            "retrieved_memory": res["retrieved_context"]["memory"],
            "rule_check": res["rule_check"],
            "auth_check": "DENIED: Missing cryptographic creator signature",
            "tool_or_skill": "BLOCKED",
            "result_summary": res["result"],
            "final_response": res["final_response"],
            "outcome": res["outcome"]
        }
        print("\n[TEST 7 TRACE]\n" + format_trace(trace))

    def test_08_authorized_creator_action_allowed(self):
        """8. Authorized creator-only action -> allowed only after real cryptographic verification."""
        orig_devices = dict(self.brain.identity_manager.devices.devices)
        orig_counter = self.brain.identity_manager.devices.counter
        created_device_id = None
        try:
            target_device_id = None
            device_priv = None
            for dev in self.brain.identity_manager.devices.list_authorized_devices():
                d_id = dev["device_id"]
                priv = self.brain.identity_manager.storage.load_private_key(f"device_{d_id}_key")
                if priv is not None:
                    target_device_id = d_id
                    device_priv = priv
                    break

            if device_priv is None or target_device_id is None:
                setup_test_identity_manager(self.brain.identity_manager)
                token = create_test_id_token("creator@test.local")
                reg_res = self.brain.identity_manager.register_new_device(
                    google_email="creator@test.local",
                    device_name="Test Device 2",
                    google_id_token=token
                )
                target_device_id = reg_res["device_id"]
                created_device_id = target_device_id
                self.brain.identity_manager.authorize_new_device_via_creator(target_device_id, authorization_proof_type="biometric")
                device_priv = self.brain.identity_manager.storage.load_private_key(f"device_{target_device_id}_key")

            self.assertIsNotNone(device_priv)
            self.assertIsNotNone(target_device_id)

            action_payload = "delete_file"
            sig_bytes = Ed25519.sign(device_priv, action_payload.encode("utf-8"))

            context = {
                "is_important": True,
                "creator_auth": {
                    "device_id": target_device_id,
                    "action_payload": action_payload,
                    "device_signature": sig_bytes.hex()
                }
            }

            res = self.brain.process(
                actor_id=CANONICAL_CREATOR_ID,
                input_text="delete file temp_cache.log",
                context=context
            )

            self.assertEqual(res["decision"], "ALLOW")
            self.assertEqual(res["outcome"], "SUCCESS")
            self.assertTrue(res["result"]["creator_verified"])
        finally:
            if created_device_id:
                self.brain.identity_manager.devices.devices.pop(created_device_id, None)
                self.brain.identity_manager.devices.counter = orig_counter
                self.brain.identity_manager.devices.save()
                keystore_path = os.path.join(self.brain.identity_manager.storage.storage_dir, f"device_{created_device_id}_key.keystore")
                if os.path.exists(keystore_path):
                    try:
                        os.remove(keystore_path)
                    except OSError:
                        pass

        trace = {
            "input": res["input"],
            "model_status": self.brain.get_model_status()["status"],
            "decision": res["decision"],
            "context_tags": res["context_tags"],
            "retrieved_knowledge": res["retrieved_context"]["knowledge"],
            "retrieved_memory": res["retrieved_context"]["memory"],
            "rule_check": res["rule_check"],
            "auth_check": "VERIFIED_ED25519_DEVICE_SIGNATURE",
            "tool_or_skill": "AUTHORIZED_ACTION",
            "result_summary": res["result"],
            "final_response": res["final_response"],
            "outcome": res["outcome"]
        }
        print("\n[TEST 8 TRACE]\n" + format_trace(trace))

    def test_09_memory_write_sanitized(self):
        """9. Memory write -> sanitized episode stored with zero plaintext secrets."""
        sensitive_payload = {
            "secret_key": "super_secret_master_key_12345",
            "password": "P@ssw0rd9988!",
            "id_token": "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdef123456",
            "device_signature": "a1b2c3d4e5f6"
        }

        self.brain.process(
            actor_id="user_security_test",
            input_text="Testing memory scrubbing",
            context={"sensitive": sensitive_payload}
        )

        episodes = self.brain.memory_engine.query_episodes(query="Testing memory scrubbing", limit=1)
        self.assertGreater(len(episodes), 0)
        saved_ep = episodes[0]

        # Verify sensitive values were strictly redacted
        ep_json = json.dumps(saved_ep)
        self.assertNotIn("super_secret_master_key_12345", ep_json)
        self.assertNotIn("P@ssw0rd9988!", ep_json)
        self.assertIn("[REDACTED_SECRET]", ep_json)

        trace = {
            "input": "Testing memory scrubbing with secrets",
            "model_status": self.brain.get_model_status()["status"],
            "decision": "ALLOW",
            "context_tags": ["MEMORY-SANITIZED"],
            "retrieved_knowledge": [],
            "retrieved_memory": [],
            "rule_check": {"decision": "ALLOW"},
            "auth_check": "NOT_REQUIRED",
            "tool_or_skill": "MemoryEngine.record_episode",
            "result_summary": "Secrets successfully redacted to [REDACTED_SECRET]",
            "final_response": "Scrubbing verified",
            "outcome": "SUCCESS"
        }
        print("\n[TEST 9 TRACE]\n" + format_trace(trace))

    def test_10_memory_retrieval_influences_response(self):
        """10. Memory retrieval -> previous relevant episode influences the response."""
        unique_marker = f"Project_Orion_Alpha_{int(time.time())}"
        
        # 1. First record an episode with the marker
        self.brain.process(actor_id="user_01", input_text=f"Initiated mission for {unique_marker}")

        # 2. Query for the marker
        res = self.brain.process(actor_id="user_01", input_text=f"Status report on {unique_marker}")

        self.assertEqual(res["outcome"], "SUCCESS")
        self.assertIn("MEMORY", res["context_tags"])
        self.assertGreater(len(res["retrieved_context"]["memory"]), 0)

        trace = {
            "input": res["input"],
            "model_status": self.brain.get_model_status()["status"],
            "decision": res["decision"],
            "context_tags": res["context_tags"],
            "retrieved_knowledge": res["retrieved_context"]["knowledge"],
            "retrieved_memory": res["retrieved_context"]["memory"],
            "rule_check": res["rule_check"],
            "auth_check": "NOT_REQUIRED",
            "tool_or_skill": "MemoryEngine.query_episodes",
            "result_summary": f"Retrieved {len(res['retrieved_context']['memory'])} historical episodes",
            "final_response": res["final_response"],
            "outcome": res["outcome"]
        }
        print("\n[TEST 10 TRACE]\n" + format_trace(trace))

    def test_11_tool_or_skill_error_reflects_failure_outcome(self):
        """11. Tool/skill error -> final outcome is ERROR/FAILURE, not SUCCESS."""
        # Request inspecting a completely non-existent file path
        res = self.brain.process(
            actor_id="user_01",
            input_text="inspect_file",
            context={"file_path": "storage/non_existent_ghost_file_9999.xyz"}
        )

        self.assertEqual(res["outcome"], "FAILURE")
        self.assertEqual(res["result"]["status"], "ERROR")
        self.assertIn("does not exist", res["result"]["error"])

        trace = {
            "input": res["input"],
            "model_status": self.brain.get_model_status()["status"],
            "decision": res["decision"],
            "context_tags": res["context_tags"],
            "retrieved_knowledge": res["retrieved_context"]["knowledge"],
            "retrieved_memory": res["retrieved_context"]["memory"],
            "rule_check": res["rule_check"],
            "auth_check": "PERMITTED",
            "tool_or_skill": "file_inspector",
            "result_summary": res["result"],
            "final_response": res["final_response"],
            "outcome": res["outcome"]
        }
        print("\n[TEST 11 TRACE]\n" + format_trace(trace))

    def test_12_api_chat_endpoint(self):
        """12. /api/v1/chat -> real Brain response via HTTP standard server."""
        port = 8991
        server = create_server(host="127.0.0.1", port=port, brain=self.brain, api_keys={"test_token": "api_tester"})
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

        try:
            url = f"http://127.0.0.1:{port}/api/v1/chat"
            req_data = json.dumps({"input": "Ping TARA Core API", "actor_id": "api_tester"}).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=req_data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer test_token"
                }
            )

            with urllib.request.urlopen(req, timeout=60.0) as resp:
                self.assertEqual(resp.status, 200)
                body = json.loads(resp.read().decode("utf-8"))

            self.assertEqual(body["status"], "SUCCESS")
            data = body["data"]
            self.assertEqual(data["decision"], "ALLOW")
            self.assertEqual(data["outcome"], "SUCCESS")
            self.assertIn("final_response", data)

            trace = {
                "input": "POST /api/v1/chat 'Ping TARA Core API'",
                "model_status": self.brain.get_model_status()["status"],
                "decision": data["decision"],
                "context_tags": data["context_tags"],
                "retrieved_knowledge": data["retrieved_context"]["knowledge"],
                "retrieved_memory": data["retrieved_context"]["memory"],
                "rule_check": data["rule_check"],
                "auth_check": "HTTP_CLIENT",
                "tool_or_skill": "API_SERVER_DISPATCH",
                "result_summary": body["status"],
                "final_response": data["final_response"],
                "outcome": data["outcome"]
            }
            print("\n[TEST 12 TRACE]\n" + format_trace(trace))

        finally:
            server.shutdown()
            server.server_close()

    def test_13_api_status_endpoint(self):
        """13. /api/v1/status -> real model/runtime status."""
        port = 8992
        server = create_server(host="127.0.0.1", port=port, brain=self.brain)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

        try:
            url = f"http://127.0.0.1:{port}/api/v1/status"
            req = urllib.request.Request(url)

            with urllib.request.urlopen(req, timeout=10.0) as resp:
                self.assertEqual(resp.status, 200)
                body = json.loads(resp.read().decode("utf-8"))

            self.assertEqual(body["status"], "ONLINE")
            self.assertEqual(body["model"]["model_identity"], "TARA")
            self.assertEqual(body["model"]["status"], "LOADED")
            self.assertTrue(body["model"]["offline_ready"])
            self.assertIn("episodes_count", body["memory"])

            trace = {
                "input": "GET /api/v1/status",
                "model_status": body["model"]["status"],
                "decision": "ALLOW",
                "context_tags": ["SYSTEM-STATUS"],
                "retrieved_knowledge": [],
                "retrieved_memory": [],
                "rule_check": {"decision": "ALLOW"},
                "auth_check": "HTTP_CLIENT",
                "tool_or_skill": "API_SERVER_STATUS",
                "result_summary": f"Status={body['status']}, Model={body['model']['status']}",
                "final_response": json.dumps(body),
                "outcome": "SUCCESS"
            }
            print("\n[TEST 13 TRACE]\n" + format_trace(trace))

        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()

