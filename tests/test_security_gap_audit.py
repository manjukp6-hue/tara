"""
tests/test_security_gap_audit.py

Verification suite for TARA Creator Authentication and Deployment Security Audit.
Verifies:
1. HTTP Security headers (X-Content-Type-Options, X-Frame-Options, CSP, HSTS, Referrer-Policy).
2. Reverse proxy IP extraction (X-Forwarded-For) in rate limiting.
3. Tamper-evident cryptographic hash chaining in SecurityAuditLogger.
4. Tamper detection on modified/inserted/deleted audit events.
5. Production model SHA-256 invariant preservation.
6. Dependency version pinning.
"""

import os
import sys
import json
import shutil
import tempfile
import unittest
import hashlib
from io import BytesIO
from unittest.mock import MagicMock

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from python.tara_core.server import TaraRequestHandler
from TARA.ACCESS.audit.security_logger import SecurityAuditLogger, GENESIS_HASH
from TARA.SECURITY.release_manifest import EXPECTED_PRODUCTION_MODEL_SHA256, compute_file_sha256


class MockServer:
    def __init__(self, rate_limiter=None, api_keys=None):
        self.rate_limiter = rate_limiter
        self.api_keys = api_keys or {"test_key": "admin"}
        self.router = None


class TestSecurityGapAudit(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="tara_sec_audit_test_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_01_json_response_security_headers(self):
        """Verify _send_json attaches all mandatory security headers."""
        handler = TaraRequestHandler.__new__(TaraRequestHandler)
        handler.server = MockServer()
        handler.request_id = "test-req-123"
        handler.wfile = BytesIO()

        captured_headers = {}
        def mock_send_header(name, value):
            captured_headers[name] = value

        def mock_send_response(code):
            captured_headers["Status-Code"] = code

        def mock_end_headers():
            pass

        handler.send_header = mock_send_header
        handler.send_response = mock_send_response
        handler.end_headers = mock_end_headers

        handler._send_json(200, {"status": "OK"})

        self.assertEqual(captured_headers.get("X-Content-Type-Options"), "nosniff")
        self.assertEqual(captured_headers.get("X-Frame-Options"), "DENY")
        self.assertEqual(captured_headers.get("Referrer-Policy"), "strict-origin-when-cross-origin")
        self.assertIn("max-age=31536000", captured_headers.get("Strict-Transport-Security", ""))
        self.assertEqual(captured_headers.get("X-Request-ID"), "test-req-123")
        self.assertEqual(captured_headers.get("Access-Control-Allow-Origin"), "*")

    def test_02_chat_ui_response_security_headers_and_csp(self):
        """Verify handle_get_chat_ui attaches CSP and web security headers."""
        handler = TaraRequestHandler.__new__(TaraRequestHandler)
        handler.server = MockServer()
        handler.wfile = BytesIO()

        captured_headers = {}
        def mock_send_header(name, value):
            captured_headers[name] = value

        def mock_send_response(code):
            captured_headers["Status-Code"] = code

        def mock_end_headers():
            pass

        handler.send_header = mock_send_header
        handler.send_response = mock_send_response
        handler.end_headers = mock_end_headers

        handler.handle_get_chat_ui()

        self.assertEqual(captured_headers.get("Status-Code"), 200)
        self.assertEqual(captured_headers.get("X-Content-Type-Options"), "nosniff")
        self.assertEqual(captured_headers.get("X-Frame-Options"), "DENY")
        self.assertEqual(captured_headers.get("Referrer-Policy"), "strict-origin-when-cross-origin")
        self.assertIn("max-age=31536000", captured_headers.get("Strict-Transport-Security", ""))
        csp = captured_headers.get("Content-Security-Policy", "")
        self.assertIn("default-src 'self'", csp)
        self.assertIn("accounts.google.com", csp)

    def test_03_proxy_aware_rate_limiting(self):
        """Verify _check_rate_limit respects X-Forwarded-For behind reverse proxies."""
        mock_limiter = MagicMock()
        mock_limiter.is_allowed.return_value = (True, 0)

        handler = TaraRequestHandler.__new__(TaraRequestHandler)
        handler.server = MockServer(rate_limiter=mock_limiter)
        handler.client_address = ("10.0.0.1", 12345)  # Internal proxy IP
        handler.headers = {"X-Forwarded-For": "203.0.113.42, 198.51.100.1"}

        result = handler._check_rate_limit()

        self.assertTrue(result)
        mock_limiter.is_allowed.assert_called_with("203.0.113.42")

    def test_04_rate_limiting_fallback_to_client_address(self):
        """Verify _check_rate_limit falls back to client_address if no proxy header."""
        mock_limiter = MagicMock()
        mock_limiter.is_allowed.return_value = (True, 0)

        handler = TaraRequestHandler.__new__(TaraRequestHandler)
        handler.server = MockServer(rate_limiter=mock_limiter)
        handler.client_address = ("192.168.1.50", 54321)
        handler.headers = {}

        result = handler._check_rate_limit()

        self.assertTrue(result)
        mock_limiter.is_allowed.assert_called_with("192.168.1.50")

    def test_05_audit_logger_cryptographic_hash_chain(self):
        """Verify SecurityAuditLogger creates unbroken, valid cryptographic hash chain."""
        logger = SecurityAuditLogger(log_dir=self.test_dir)

        e1 = logger.log_event("TEST_EVENT_1", severity="INFO", details={"action": "step1"})
        self.assertEqual(e1["prev_hash"], GENESIS_HASH)
        self.assertIn("entry_hash", e1)

        e2 = logger.log_event("TEST_EVENT_2", severity="WARNING", details={"action": "step2"})
        self.assertEqual(e2["prev_hash"], e1["entry_hash"])

        e3 = logger.log_event("TEST_EVENT_3", severity="CRITICAL", details={"action": "step3"})
        self.assertEqual(e3["prev_hash"], e2["entry_hash"])

        # Check integrity
        is_valid, err = logger.verify_log_integrity()
        self.assertTrue(is_valid)
        self.assertIsNone(err)

    def test_06_audit_logger_tamper_detection_modified_content(self):
        """Verify verify_log_integrity detects tampered event details."""
        logger = SecurityAuditLogger(log_dir=self.test_dir)

        logger.log_event("EVENT_A", severity="INFO", details={"count": 1})
        logger.log_event("EVENT_B", severity="INFO", details={"count": 2})
        logger.log_event("EVENT_C", severity="INFO", details={"count": 3})

        # Tamper with the log file by altering details of second event
        with open(logger.log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        tampered_entry = json.loads(lines[1])
        tampered_entry["details"]["count"] = 999999  # Unauthorized modification
        lines[1] = json.dumps(tampered_entry) + "\n"

        with open(logger.log_file, "w", encoding="utf-8") as f:
            f.writelines(lines)

        # Verification must detect the tampering
        is_valid, err = logger.verify_log_integrity()
        self.assertFalse(is_valid)
        self.assertIn("Tampered entry content", err)

    def test_07_audit_logger_tamper_detection_deleted_entry(self):
        """Verify verify_log_integrity detects a deleted/omitted entry in the chain."""
        logger = SecurityAuditLogger(log_dir=self.test_dir)

        logger.log_event("STEP_1", severity="INFO")
        logger.log_event("STEP_2", severity="INFO")
        logger.log_event("STEP_3", severity="INFO")

        # Delete step 2 from log file
        with open(logger.log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Keep only step 1 and step 3
        tampered_lines = [lines[0], lines[2]]
        with open(logger.log_file, "w", encoding="utf-8") as f:
            f.writelines(tampered_lines)

        is_valid, err = logger.verify_log_integrity()
        self.assertFalse(is_valid)
        self.assertIn("Broken chain link", err)

    def test_08_production_model_sha256_invariance(self):
        """Verify production model safetensors matches expected SHA256 exactly."""
        model_path = os.path.join(REPO_ROOT, "storage", "models", "tara", "model.safetensors")
        self.assertTrue(os.path.isfile(model_path), f"Missing model file: {model_path}")
        actual_hash = compute_file_sha256(model_path)
        self.assertEqual(
            actual_hash,
            EXPECTED_PRODUCTION_MODEL_SHA256,
            f"Production model SHA-256 invariant violation! Expected {EXPECTED_PRODUCTION_MODEL_SHA256}, got {actual_hash}"
        )

    def test_09_requirements_version_pinning(self):
        """Verify requirements.txt has pinned versions for critical packages."""
        req_path = os.path.join(REPO_ROOT, "requirements.txt")
        with open(req_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

        req_map = {}
        for line in lines:
            if "==" in line:
                pkg, ver = line.split("==", 1)
                req_map[pkg.strip()] = ver.strip()
            elif ">=" in line:
                pkg, ver = line.split(">=", 1)
                req_map[pkg.strip()] = f">={ver.strip()}"

        self.assertEqual(req_map.get("cryptography"), "50.0.1")
        self.assertEqual(req_map.get("huggingface_hub"), "1.31.0")
        self.assertEqual(req_map.get("safetensors"), "0.8.0")
        self.assertIn("torch", req_map)


if __name__ == "__main__":
    unittest.main()
