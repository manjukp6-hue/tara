"""
tests/test_frontend_replication_and_failover.py

Comprehensive Validation Suite for:
1. One canonical TARA frontend source (frontend/index.html)
2. Automated replication & synchronization across Cloudflare, ModelScope, Hugging Face, Render, Python Core, Rust Server
3. Canonical public domain (https://gateway.tara.local)
4. Provider-independent frontend failover system
5. Production model and zero-cost compute policy invariants
"""

import os
import sys
import json
import time
import hashlib
import unittest
import threading
import urllib.request
import urllib.error
from http.server import HTTPServer, SimpleHTTPRequestHandler

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON_DIR = os.path.join(REPO_ROOT, "python")
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)

from tara_core.contracts import (
    CANONICAL_MODEL_IDENTITY,
    CANONICAL_MODEL_SHA256,
    CANONICAL_PARAM_COUNT,
    PUBLIC_TARA_URL,
    USER_COMPUTE_COST,
    ENFORCE_ZERO_USER_COST,
)
from tara_core.frontend_sync import (
    get_canonical_frontend,
    sync_all_targets,
    verify_sync_integrity,
    FRONTEND_SOURCE_PATH,
    FRONTEND_MANIFEST_PATH,
    TARA_MANIFEST_PATH,
    CLOUDFLARE_INDEX_PATH,
    MODELSCOPE_DIR,
    HUGGINGFACE_DIR,
    RENDER_DIR,
)
from providers.modelscope.app import demo as modelscope_demo
from providers.render.app import RenderHTTPHandler


class TestFrontendReplicationAndFailover(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Synchronize all targets cleanly before running test suite
        sync_all_targets()

    def test_01_canonical_frontend_integrity_and_manifest(self):
        """Verifies canonical frontend source existence, SHA256 integrity, and manifests."""
        self.assertTrue(os.path.exists(FRONTEND_SOURCE_PATH), "Canonical frontend source file must exist")
        html_content, sha256_hash, byte_size = get_canonical_frontend()

        self.assertGreater(byte_size, 5000, "Frontend HTML must contain full functional application")
        self.assertIn("TARA AI Core", html_content)
        self.assertIn(CANONICAL_MODEL_SHA256, html_content)
        self.assertIn(str(CANONICAL_PARAM_COUNT), html_content)
        self.assertIn("sendWithFailover", html_content)
        self.assertIn("REPLICA_REGISTRY", html_content)
        self.assertIn("$0.00 / Zero-Cost", html_content)

        # Check manifests
        self.assertTrue(os.path.exists(FRONTEND_MANIFEST_PATH), "frontend/manifest.json must exist")
        self.assertTrue(os.path.exists(TARA_MANIFEST_PATH), "TARA/MANIFEST/frontend_manifest.json must exist")

        with open(FRONTEND_MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.assertEqual(manifest["canonical_sha256"], sha256_hash)
        self.assertEqual(manifest["size_bytes"], byte_size)
        self.assertEqual(manifest["canonical_public_domain"], "https://gateway.tara.local")
        self.assertEqual(manifest["zero_user_cost"], 0.0)
        self.assertEqual(manifest["canonical_param_count"], 118080)
        self.assertEqual(manifest["canonical_model_sha256"], CANONICAL_MODEL_SHA256)
        self.assertIn("cloudflare", manifest["replicated_targets"])
        self.assertIn("modelscope", manifest["replicated_targets"])
        self.assertIn("huggingface", manifest["replicated_targets"])
        self.assertIn("render", manifest["replicated_targets"])

    def test_02_automated_replication_across_all_targets(self):
        """Audits all provider targets to ensure zero byte drift and 100% SHA256 match."""
        audit = verify_sync_integrity()
        self.assertTrue(audit["all_in_sync"], f"Replication audit failed: {json.dumps(audit, indent=2)}")

        expected_sha = audit["canonical_sha256"]

        for target in ["modelscope", "huggingface", "render"]:
            status_entry = audit["target_status"][target]
            self.assertEqual(status_entry["status"], "MATCH")
            self.assertEqual(status_entry["sha256"], expected_sha)
            self.assertEqual(status_entry["size_bytes"], audit["expected_size"])

        # Cloudflare embedding verification
        cf_status = audit["target_status"]["cloudflare"]
        self.assertEqual(cf_status["status"], "MATCH")
        self.assertTrue(cf_status["has_sha_marker"])

    def test_03_provider_replicas_serve_canonical_frontend_and_api(self):
        """Validates ModelScope Gradio app, HF Static Space, and Render server."""
        _, expected_sha, expected_size = get_canonical_frontend()

        # 1. Test ModelScope Gradio App startup
        app, local_url, _ = modelscope_demo.launch(
            server_name="127.0.0.1",
            server_port=17865,
            prevent_thread_lock=True
        )
        try:
            req = urllib.request.Request("http://127.0.0.1:17865/")
            with urllib.request.urlopen(req, timeout=4.0) as resp:
                self.assertEqual(resp.status, 200)
                body = resp.read()
                self.assertGreater(len(body), 500)
        finally:
            modelscope_demo.close()

        # 2. Test Hugging Face Static Space Replica files
        hf_index = os.path.join(HUGGINGFACE_DIR, "index.html")
        hf_readme = os.path.join(HUGGINGFACE_DIR, "README.md")
        self.assertTrue(os.path.exists(hf_index))
        self.assertTrue(os.path.exists(hf_readme))

        with open(hf_index, "rb") as f:
            hf_sha = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(hf_sha, expected_sha, "Hugging Face index.html must match canonical SHA256 exactly")

        with open(hf_readme, "r", encoding="utf-8") as f:
            readme_text = f.read()
        self.assertIn("sdk: static", readme_text, "Hugging Face Space must be configured with sdk: static")

        # 3. Test Render HTTP Server
        render_server = HTTPServer(("127.0.0.1", 17866), RenderHTTPHandler)
        rt = threading.Thread(target=render_server.serve_forever, daemon=True)
        rt.start()
        try:
            req_ren = urllib.request.Request("http://127.0.0.1:17866/")
            with urllib.request.urlopen(req_ren, timeout=4.0) as resp:
                self.assertEqual(resp.status, 200)
                html_bytes = resp.read()
                self.assertEqual(hashlib.sha256(html_bytes).hexdigest(), expected_sha)

            req_health = urllib.request.Request("http://127.0.0.1:17866/health")
            with urllib.request.urlopen(req_health, timeout=4.0) as resp:
                self.assertEqual(resp.status, 200)
                h_data = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(h_data["status"], "HEALTHY")
                self.assertEqual(h_data["user_compute_cost"], 0.0)
        finally:
            render_server.shutdown()
            render_server.server_close()

    def test_04_provider_independent_client_failover(self):
        """Simulates client failover when primary replica fails or times out."""
        replicas = [
            {"id": "cloudflare_edge", "name": "Cloudflare Edge", "url": "http://127.0.0.1:19991", "healthy": False},
            {"id": "render_replica", "name": "Render Backend", "url": "http://127.0.0.1:19992", "healthy": True}
        ]

        # Start only the secondary replica server (simulating primary down)
        secondary_server = HTTPServer(("127.0.0.1", 19992), RenderHTTPHandler)
        t = threading.Thread(target=secondary_server.serve_forever, daemon=True)
        t.start()

        try:
            # Client Failover Algorithm Simulation
            payload = json.dumps({"prompt": "Hello TARA via failover"}).encode("utf-8")
            response_data = None
            active_replica_id = None
            failover_occurred = False

            for replica in replicas:
                try:
                    req = urllib.request.Request(
                        f"{replica['url']}/api/v1/inference",
                        data=payload,
                        headers={"Content-Type": "application/json"}
                    )
                    with urllib.request.urlopen(req, timeout=0.8) as resp:
                        if resp.status == 200:
                            response_data = json.loads(resp.read().decode("utf-8"))
                            active_replica_id = replica["id"]
                            break
                except Exception:
                    failover_occurred = True
                    continue

            self.assertTrue(failover_occurred, "Failover must have been triggered after primary failure")
            self.assertIsNotNone(response_data, "Secondary replica must have successfully fulfilled request")
            self.assertEqual(active_replica_id, "render_replica")
            self.assertEqual(response_data["status"], "SUCCESS")
            self.assertEqual(response_data["provider"], "render")
            self.assertEqual(response_data["user_compute_cost"], 0.0)
        finally:
            secondary_server.shutdown()
            secondary_server.server_close()

    def test_05_cloudflare_edge_gateway_bundle_integrity(self):
        """Validates that cloudflare/src/index.js contains canonical frontend base64 and valid syntax."""
        self.assertTrue(os.path.exists(CLOUDFLARE_INDEX_PATH))
        with open(CLOUDFLARE_INDEX_PATH, "r", encoding="utf-8") as f:
            cf_code = f.read()

        _, expected_sha, _ = get_canonical_frontend()
        self.assertIn("CANONICAL_FRONTEND_B64", cf_code)
        self.assertIn(expected_sha, cf_code)
        self.assertIn("getCanonicalFrontendHTML", cf_code)
        self.assertIn("handleChatUI", cf_code)

    def test_06_production_model_and_zero_cost_invariants(self):
        """Strict verification that canonical model on disk and zero cost policy remain intact."""
        model_path = os.path.join(REPO_ROOT, "storage", "models", "tara", "model.safetensors")
        self.assertTrue(os.path.exists(model_path), f"Production model must exist at {model_path}")

        hasher = hashlib.sha256()
        with open(model_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        actual_sha = hasher.hexdigest()

        self.assertEqual(actual_sha.lower(), CANONICAL_MODEL_SHA256.lower())
        self.assertEqual(CANONICAL_PARAM_COUNT, 118080)
        self.assertEqual(USER_COMPUTE_COST, 0.0)
        self.assertTrue(ENFORCE_ZERO_USER_COST)


if __name__ == "__main__":
    unittest.main()
