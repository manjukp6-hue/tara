"""
Unit and integration tests for TARA API Authentication, Spoofing Prevention, and Rate Limiting.
"""

import unittest
import urllib.request
import urllib.error
import json
import threading
import time
import os
import sys

# Ensure repository root and python directory are in sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_DIR = os.path.join(REPO_ROOT, "python")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)

from tara_core.server import create_server, SlidingWindowRateLimiter
from tara_core.brain import TaraBrain


class TestApiAuthAndRateLimiting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.brain = TaraBrain()
        cls.port = 8993
        cls.rate_limiter = SlidingWindowRateLimiter(limit_per_minute=1000, burst_limit=100)
        cls.server = create_server(
            host="127.0.0.1",
            port=cls.port,
            brain=cls.brain,
            api_keys={
                "valid_user_token": "alice",
                "valid_admin_token": "admin"
            },
            rate_limiter=cls.rate_limiter
        )
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def _make_request(self, path, method="GET", data=None, headers=None):
        url = f"http://127.0.0.1:{self.port}{path}"
        req_headers = dict(headers or {})
        req_data = None
        if data is not None:
            req_data = json.dumps(data).encode("utf-8")
            req_headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=req_data, headers=req_headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=60.0) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return resp.status, body, dict(resp.headers)
        except urllib.error.HTTPError as e:
            try:
                body = json.loads(e.read().decode("utf-8"))
            except Exception:
                body = {}
            return e.code, body, dict(e.headers)

    def test_01_public_status_endpoint_no_auth(self):
        """Verify /api/v1/status is publicly accessible without credentials."""
        status, body, _ = self._make_request("/api/v1/status")
        self.assertEqual(status, 200)
        self.assertEqual(body.get("status"), "ONLINE")
        self.assertIn("model", body)

    def test_01b_status_endpoint_model_integrity_fields(self):
        """Verify /api/v1/status includes exact computed SHA256, model_file, and model_integrity."""
        status, body, _ = self._make_request("/api/v1/status")
        self.assertEqual(status, 200)
        model = body.get("model", {})
        
        # Verify required fields
        self.assertIn("model_file", model)
        self.assertIn("model_sha256", model)
        self.assertIn("model_checkpoint", model)
        self.assertIn("model_integrity", model)
        
        # Verify model file points to model.safetensors
        self.assertTrue(model["model_file"].endswith("storage/models/tara/model.safetensors"))
        
        # Verify computed SHA256 matches actual runtime file
        expected_sha = "7a50308b799f2654baeafbd64dec31f088c3b07b446e198cd0f9ec2b7c0af309"
        self.assertEqual(model["model_sha256"], expected_sha)
        self.assertEqual(model["model_integrity"], "PASS")
        
        # Verify secrets are not exposed
        body_str = json.dumps(body)
        self.assertNotIn("HF_TOKEN", body_str)
        self.assertNotIn("TARA_CREATOR_PASSPHRASE", body_str)

    def test_02_chat_endpoint_missing_auth_returns_401(self):
        """Verify /api/v1/chat returns 401 when no auth header is provided."""
        status, body, _ = self._make_request(
            "/api/v1/chat",
            method="POST",
            data={"input": "Hello", "actor_id": "alice"}
        )
        self.assertEqual(status, 401)
        self.assertEqual(body.get("status"), "ERROR")
        self.assertIn("Unauthorized", body.get("error", ""))

    def test_03_chat_endpoint_invalid_auth_returns_401(self):
        """Verify /api/v1/chat returns 401 when invalid Bearer token is provided."""
        status, body, _ = self._make_request(
            "/api/v1/chat",
            method="POST",
            data={"input": "Hello", "actor_id": "alice"},
            headers={"Authorization": "Bearer invalid_secret_token"}
        )
        self.assertEqual(status, 401)
        self.assertEqual(body.get("status"), "ERROR")
        self.assertIn("Invalid API key or token", body.get("error", ""))

    def test_04_chat_endpoint_valid_bearer_token(self):
        """Verify /api/v1/chat returns 200 with valid Bearer token."""
        status, body, headers = self._make_request(
            "/api/v1/chat",
            method="POST",
            data={"input": "What is 2 + 2?", "actor_id": "alice"},
            headers={"Authorization": "Bearer valid_user_token"}
        )
        self.assertEqual(status, 200)
        self.assertEqual(body.get("status"), "SUCCESS")
        self.assertIn("X-Request-ID", headers)

    def test_05_skills_endpoint_x_api_key_header(self):
        """Verify /api/v1/skills accepts X-API-Key and returns skills list."""
        status, body, _ = self._make_request(
            "/api/v1/skills",
            headers={"X-API-Key": "valid_user_token"}
        )
        self.assertEqual(status, 200)
        self.assertEqual(body.get("status"), "SUCCESS")
        self.assertIn("skills", body)

    def test_06_actor_id_spoofing_prevented(self):
        """Verify user authenticated as 'alice' cannot claim to be 'bob' (returns 403)."""
        status, body, _ = self._make_request(
            "/api/v1/chat",
            method="POST",
            data={"input": "Show my account", "actor_id": "bob"},
            headers={"Authorization": "Bearer valid_user_token"}
        )
        self.assertEqual(status, 403)
        self.assertEqual(body.get("status"), "ERROR")
        self.assertIn("Forbidden", body.get("error", ""))
        self.assertIn("cannot spoof", body.get("error", ""))

    def test_07_admin_actor_spoofing_allowed(self):
        """Verify admin actor can proxy/specify another actor_id."""
        status, body, _ = self._make_request(
            "/api/v1/chat",
            method="POST",
            data={"input": "System check", "actor_id": "bob"},
            headers={"Authorization": "Bearer valid_admin_token"}
        )
        self.assertEqual(status, 200)
        self.assertEqual(body.get("status"), "SUCCESS")

    def test_08_creator_only_action_still_requires_ed25519_signature(self):
        """Verify an API token alone cannot authorize creator-only actions without Ed25519 signature."""
        status, body, _ = self._make_request(
            "/api/v1/chat",
            method="POST",
            data={
                "input": "delete_file critical_system_config.json",
                "actor_id": "alice",
                "context": {
                    "action_type": "delete_file",
                    "critical_target": True,
                    "is_important": True
                }
            },
            headers={"Authorization": "Bearer valid_user_token"}
        )
        self.assertEqual(status, 200)
        data = body.get("data", {})
        self.assertEqual(data.get("decision"), "DENY")
        self.assertIn("creator", str(data.get("rule_check", "")).lower() + str(data.get("final_response", "")).lower())

    def test_09_rate_limiting_burst_protection(self):
        """Verify rate limiter blocks bursts exceeding limit with 429 and Retry-After."""
        limiter = SlidingWindowRateLimiter(limit_per_minute=60, burst_limit=3)
        self.assertTrue(limiter.is_allowed("1.2.3.4")[0])
        self.assertTrue(limiter.is_allowed("1.2.3.4")[0])
        self.assertTrue(limiter.is_allowed("1.2.3.4")[0])
        allowed, retry_after = limiter.is_allowed("1.2.3.4")
        self.assertFalse(allowed)
        self.assertGreaterEqual(retry_after, 1)


if __name__ == "__main__":
    unittest.main()
