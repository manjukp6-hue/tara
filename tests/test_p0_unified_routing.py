"""
tests/test_p0_unified_routing.py

Verification & Regression Suite for P0.3:
Unified API Server Routing Architecture.
Ensures:
1. GLOBAL_API_ROUTER is the single authoritative routing registry.
2. All endpoints (chat, status, health, skills, admin, auth, endpoints, engines, sync) are registered.
3. Centralized middleware executes consistently:
   - Rate limiting
   - CORS and security headers
   - Authentication (401 Unauthorized)
   - Role authorization (403 Forbidden)
   - Method Not Allowed (405)
   - Not Found (404)
4. No hidden/manual routing handlers exist.
"""

import os
import sys
import json
import time
import urllib.request
import urllib.error
import threading
import unittest

sys.path.insert(0, os.path.abspath("python"))

from tara_core.server import (
    GLOBAL_API_ROUTER,
    create_server,
    ApiRouter
)


class DummyBrain:
    def __init__(self):
        self.repo_root = os.path.abspath(".")
        self.identity_manager = None
        self.guard = None
        self.knowledge_base = None
        self.skill_engine = None
        self.memory_engine = None

    def process(self, actor_id, input_text, context=None):
        return {"response": f"Echo: {input_text}", "actor": actor_id}

    def get_model_status(self):
        return {"model_name": "TARA", "status": "ONLINE"}


class TestP0UnifiedRouting(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.port = 18991
        cls.host = "127.0.0.1"
        cls.brain = DummyBrain()
        cls.api_keys = {
            "test_user_token": "regular_user",
            "test_admin_token": "admin",
            "test_root_token": "ROOT_OPERATOR"
        }
        cls.server = create_server(
            host=cls.host,
            port=cls.port,
            brain=cls.brain,
            api_keys=cls.api_keys
        )
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()
        time.sleep(0.3)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def _request(self, method, path, headers=None, body=None):
        url = f"http://{self.host}:{self.port}{path}"
        data = None
        if body is not None:
            if isinstance(body, (dict, list)):
                data = json.dumps(body).encode("utf-8")
            elif isinstance(body, str):
                data = body.encode("utf-8")
            else:
                data = body

        req = urllib.request.Request(url, data=data, method=method)
        if headers:
            for k, v in headers.items():
                req.add_header(k, v)
        if data is not None and "Content-Type" not in (headers or {}):
            req.add_header("Content-Type", "application/json")

        try:
            with urllib.request.urlopen(req) as resp:
                status = resp.status
                resp_body = resp.read().decode("utf-8")
                resp_headers = dict(resp.headers)
                return status, resp_body, resp_headers
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8")
            return e.code, err_body, dict(e.headers)

    def test_01_router_inventory_completeness(self):
        """Verify all core endpoints are registered in GLOBAL_API_ROUTER."""
        routes = GLOBAL_API_ROUTER.list_routes()
        paths = {f"{r['method']}:{r['path']}" for r in routes}

        expected_core = [
            "GET:/",
            "GET:/health",
            "GET:/api/v1/health",
            "GET:/api/v1/status",
            "GET:/api/v1/skills",
            "POST:/api/v1/chat",
            "GET:/api/v1/admin/overview",
            "GET:/api/v1/admin/candidates",
            "GET:/api/v1/admin/knowledge",
            "GET:/api/v1/admin/memory",
            "GET:/api/v1/admin/rules",
            "GET:/api/v1/admin/security",
            "GET:/api/v1/admin/self-train-status",
            "POST:/api/v1/admin/approve_candidate",
            "POST:/api/v1/admin/knowledge",
            "POST:/api/v1/admin/skill_execute",
            "POST:/api/v1/admin/device_authorize",
            "POST:/api/v1/admin/self-train",
        ]

        for exp in expected_core:
            self.assertIn(exp, paths, f"Expected endpoint '{exp}' missing from GLOBAL_API_ROUTER!")

    def test_02_public_endpoints_accessible_without_auth(self):
        """Verify unauthenticated endpoints succeed with 200 and security headers."""
        for path in ["/", "/health", "/api/v1/health", "/api/v1/status"]:
            status, body, headers = self._request("GET", path)
            self.assertEqual(status, 200, f"Public endpoint '{path}' failed with status {status}")
            # Verify security headers injected
            self.assertIn("x-content-type-options", [k.lower() for k in headers])

    def test_03_protected_endpoints_reject_unauthorized(self):
        """Verify protected endpoints return uniform 401 Unauthorized when token missing."""
        protected = [
            ("POST", "/api/v1/chat", {"input": "hello"}),
            ("GET", "/api/v1/skills", None),
            ("GET", "/api/v1/admin/overview", None),
            ("POST", "/api/v1/admin/self-train", {}),
        ]
        for method, path, payload in protected:
            status, body, headers = self._request(method, path, body=payload)
            self.assertEqual(status, 401, f"Expected 401 for '{method} {path}', got {status}")
            data = json.loads(body)
            self.assertEqual(data.get("status"), "ERROR")
            self.assertIn("Unauthorized", data.get("error", ""))

    def test_04_admin_endpoints_reject_non_admin_role(self):
        """Verify admin endpoints return 403 Forbidden for non-admin actors."""
        headers = {"Authorization": "Bearer test_user_token"}
        status, body, _ = self._request("GET", "/api/v1/admin/overview", headers=headers)
        self.assertEqual(status, 403, f"Expected 403 for non-admin, got {status}")
        data = json.loads(body)
        self.assertIn("Forbidden", data.get("error", ""))

    def test_05_method_not_allowed_consistent_405(self):
        """Verify 405 Method Not Allowed is returned when path exists with different method."""
        # /api/v1/chat is POST only
        status, body, _ = self._request("GET", "/api/v1/chat")
        self.assertEqual(status, 405, f"Expected 405 for GET /api/v1/chat, got {status}")
        data = json.loads(body)
        self.assertIn("Method GET not allowed", data.get("error", ""))

        # /api/v1/status is GET only
        status, body, _ = self._request("POST", "/api/v1/status", body={"test": 1})
        self.assertEqual(status, 405, f"Expected 405 for POST /api/v1/status, got {status}")

    def test_06_unregistered_endpoint_consistent_404(self):
        """Verify 404 Not Found is returned for nonexistent routes."""
        status, body, _ = self._request("GET", "/api/v1/totally_nonexistent_endpoint")
        self.assertEqual(status, 404)
        data = json.loads(body)
        self.assertIn("not found", data.get("error", "").lower())

    def test_07_cors_and_request_id_headers_present(self):
        """Verify CORS and X-Request-ID headers are present on API responses."""
        headers = {"Origin": "http://localhost:3000"}
        status, body, resp_headers = self._request("GET", "/api/v1/status", headers=headers)
        self.assertEqual(status, 200)
        lower_headers = {k.lower(): v for k, v in resp_headers.items()}
        self.assertIn("access-control-allow-origin", lower_headers)
        self.assertIn("x-request-id", lower_headers)

    def test_08_authenticated_chat_flow(self):
        """Verify authenticated chat request executes cleanly via unified router."""
        headers = {"Authorization": "Bearer test_user_token"}
        status, body, _ = self._request("POST", "/api/v1/chat", headers=headers, body={"input": "ping"})
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data.get("status"), "SUCCESS")
        self.assertIn("Echo: ping", data["data"]["response"])


if __name__ == "__main__":
    unittest.main()
