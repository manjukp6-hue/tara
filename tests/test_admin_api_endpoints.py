"""
Unit and integration tests for TARA Admin API endpoints (/api/v1/admin/*).
"""

import unittest
import urllib.request
import urllib.error
import json
import threading
import time
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_DIR = os.path.join(REPO_ROOT, "python")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)

from tara_core.server import create_server, SlidingWindowRateLimiter
from tara_core.brain import TaraBrain


class TestAdminApiEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.brain = TaraBrain()
        cls.port = 8996
        cls.rate_limiter = SlidingWindowRateLimiter(limit_per_minute=1000, burst_limit=100)
        cls.server = create_server(
            host="127.0.0.1",
            port=cls.port,
            brain=cls.brain,
            api_keys={
                "admin_secret_token": "admin"
            },
            rate_limiter=cls.rate_limiter
        )
        cls.created_kids = set()
        cls.created_cids = set()
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()
        time.sleep(0.1)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        kb = cls.brain.knowledge_base
        for kid in getattr(cls, "created_kids", set()):
            kpath = os.path.join(kb.entries_dir, f"{kid}.json")
            if os.path.exists(kpath):
                try:
                    os.remove(kpath)
                except Exception:
                    pass
            kb.index.pop(kid, None)
        kb._save_index()

        for cid in getattr(cls, "created_cids", set()):
            cpath = os.path.join(kb.candidates_dir, f"{cid}.json")
            if os.path.exists(cpath):
                try:
                    os.remove(cpath)
                except Exception:
                    pass

    def _make_request(self, path, method="GET", data=None, headers=None):
        url = f"http://127.0.0.1:{self.port}{path}"
        req_headers = dict(headers or {})
        req_data = None
        if data is not None:
            req_data = json.dumps(data).encode("utf-8")
            req_headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=req_data, headers=req_headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=30.0) as resp:
                body = resp.read().decode("utf-8")
                return resp.status, json.loads(body)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8")
            try:
                data = json.loads(body)
            except Exception:
                data = {"raw": body}
            return e.code, data

    def test_01_admin_endpoints_require_authentication(self):
        status, data = self._make_request("/api/v1/admin/overview")
        self.assertEqual(status, 401)
        self.assertEqual(data.get("status"), "ERROR")

    def test_02_admin_overview_returns_system_state(self):
        headers = {"Authorization": "Bearer admin_secret_token"}
        status, data = self._make_request("/api/v1/admin/overview", headers=headers)
        self.assertEqual(status, 200)
        self.assertEqual(data.get("status"), "SUCCESS")
        self.assertIn("quarantine", data)
        self.assertIn("knowledge_entries_count", data)
        self.assertIn("skills_count", data)
        self.assertIn("memory", data)
        self.assertIn("devices", data)
        self.assertIn("model", data)

    def test_03_admin_candidates_endpoint(self):
        headers = {"Authorization": "Bearer admin_secret_token"}
        status, data = self._make_request("/api/v1/admin/candidates", headers=headers)
        self.assertEqual(status, 200)
        self.assertEqual(data.get("status"), "SUCCESS")
        self.assertIsInstance(data.get("candidates"), list)

    def test_04_admin_knowledge_query_and_store(self):
        headers = {"Authorization": "Bearer admin_secret_token"}
        # Query knowledge
        status, data = self._make_request("/api/v1/admin/knowledge", headers=headers)
        self.assertEqual(status, 200)
        self.assertEqual(data.get("status"), "SUCCESS")
        self.assertIsInstance(data.get("knowledge"), list)

        # Store knowledge
        payload = {
            "topic": "system_admin",
            "subject": "desktop_console_testing",
            "content": "Administrative console connected for TARA system monitoring.",
            "sources": [{"title": "Admin Test", "url": "local:test"}]
        }
        status_post, data_post = self._make_request("/api/v1/admin/knowledge", method="POST", data=payload, headers=headers)
        self.assertEqual(status_post, 200)
        self.assertEqual(data_post.get("status"), "SUCCESS")
        res_entry = data_post.get("result", {})
        kid = res_entry.get("knowledge_id")
        if kid:
            self.__class__.created_kids.add(kid)

    def test_05_admin_memory_endpoint(self):
        headers = {"Authorization": "Bearer admin_secret_token"}
        status, data = self._make_request("/api/v1/admin/memory", headers=headers)
        self.assertEqual(status, 200)
        self.assertEqual(data.get("status"), "SUCCESS")
        self.assertIsInstance(data.get("episodes"), list)

    def test_06_admin_rules_endpoint(self):
        headers = {"Authorization": "Bearer admin_secret_token"}
        status, data = self._make_request("/api/v1/admin/rules", headers=headers)
        self.assertEqual(status, 200)
        self.assertEqual(data.get("status"), "SUCCESS")
        self.assertIn("policy", data)

    def test_07_admin_security_endpoint(self):
        headers = {"Authorization": "Bearer admin_secret_token"}
        status, data = self._make_request("/api/v1/admin/security", headers=headers)
        self.assertEqual(status, 200)
        self.assertEqual(data.get("status"), "SUCCESS")
        self.assertIn("devices", data)
        self.assertIn("lockdown_state", data)

    def test_08_admin_skill_execution(self):
        headers = {"Authorization": "Bearer admin_secret_token"}
        payload = {
            "skill_name": "utilities",
            "parameters": {"operation": "hash", "data": "TARA"}
        }
        status, data = self._make_request("/api/v1/admin/skill_execute", method="POST", data=payload, headers=headers)
        self.assertEqual(status, 200)
        self.assertEqual(data.get("status"), "SUCCESS")
        self.assertEqual(data.get("skill_name"), "utilities")

    def test_09_quarantine_approval_flow(self):
        headers = {"Authorization": "Bearer admin_secret_token"}
        # Store a test candidate directly
        cand = self.brain.knowledge_base.store_candidate(
            topic="testing",
            subject="quarantine_flow",
            content="Quarantine approval test content.",
            source_query="test quarantine",
            sources=[{"title": "Test", "url": "https://example.com/test"}],
            learned_by_role="TESTER"
        )
        cid = cand["candidate_id"]
        self.__class__.created_cids.add(cid)

        # Attempt approval by non-creator -> 403
        bad_payload = {"candidate_id": cid, "creator_id": "attacker"}
        st_bad, d_bad = self._make_request("/api/v1/admin/approve_candidate", method="POST", data=bad_payload, headers=headers)
        self.assertEqual(st_bad, 403)

        # Valid creator approval -> 200
        good_payload = {"candidate_id": cid, "creator_id": "ROOT_OPERATOR"}
        st_good, d_good = self._make_request("/api/v1/admin/approve_candidate", method="POST", data=good_payload, headers=headers)
        self.assertEqual(st_good, 200)
        self.assertEqual(d_good.get("action"), "APPROVED")
        appr_kid = d_good.get("result", {}).get("knowledge_id")
        if appr_kid:
            self.__class__.created_kids.add(appr_kid)

    def test_10_admin_self_train_status(self):
        headers = {"Authorization": "Bearer admin_secret_token"}
        status, data = self._make_request("/api/v1/admin/self-train-status", headers=headers)
        self.assertEqual(status, 200)
        self.assertEqual(data.get("status"), "SUCCESS")
        self.assertIn("self_training_status", data)


if __name__ == "__main__":
    unittest.main()
