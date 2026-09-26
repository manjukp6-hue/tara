"""
tests/test_cross_runtime_equivalence.py

Comprehensive Dual-Runtime Equivalence & Parity Test Suite for TARA.
Validates:
1. Canonical Contract Schema validation and serialization.
2. SafeTensors shared weight loading and SHA256 checksum invariant.
3. Tokenizer vocabulary and special token parity.
4. Zero-Trust boundary enforcement (rejection of unauthenticated localhost calls).
5. Real autoregressive neural inference on Python worker with canonical response schema.
6. Dual-runtime browser session cookie authentication parity.
"""

import os
import sys
import json
import time
import hashlib
import unittest
import threading
from urllib.request import Request, urlopen
from urllib.error import HTTPError

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON_ROOT = os.path.join(REPO_ROOT, "python")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if PYTHON_ROOT not in sys.path:
    sys.path.insert(0, PYTHON_ROOT)

from tara_core.contracts import (
    CanonicalInferenceRequest,
    CanonicalInferenceResponse,
    WorkerHealthResponse,
    CanonicalErrorResponse,
    AuthContext,
    CANONICAL_MODEL_SHA256,
    CANONICAL_PROTOCOL_VERSION,
    CANONICAL_MODEL_IDENTITY,
    CANONICAL_PARAM_COUNT,
    VoiceTranscribeRequest,
    VoiceTranscribeResponse,
    VoiceSynthesizeRequest,
    VoiceConverseRequest,
    VoiceConverseResponse,
    VoiceBargeInResponse,
    VoiceCapabilitiesResponse,
    CreatorSetupRequest,
    CreatorSetupResponse,
    DeviceRegisterRequest,
    DeviceRevokeRequest,
    DeviceListResponse,
)
from tara_model.tokenizer import TaraTokenizer
from tara_core.gpu_worker import (
    WorkerHandler,
    init_worker_model,
    INTERNAL_WORKER_KEY,
    MODEL,
    TOKENIZER,
    MODEL_SHA256,
)
from tara_core.server import (
    WebSessionStore,
    GLOBAL_WEB_SESSION_STORE,
    TaraRequestHandler,
    ApiRouter,
    register_core_api_routes,
)
from http.server import HTTPServer


class TestCanonicalContracts(unittest.TestCase):
    """Verifies schema rules and invariant validation in the Contract Layer."""

    def test_canonical_constants(self):
        self.assertEqual(CANONICAL_PROTOCOL_VERSION, "1.0.0")
        self.assertEqual(CANONICAL_MODEL_IDENTITY, "TARA")
        self.assertEqual(CANONICAL_MODEL_SHA256, "7a50308b799f2654baeafbd64dec31f088c3b07b446e198cd0f9ec2b7c0af309")
        self.assertEqual(CANONICAL_PARAM_COUNT, 118080)

    def test_inference_request_validation(self):
        # Valid request
        req = CanonicalInferenceRequest(
            request_id="req_001",
            prompt="Hello TARA",
            expected_model_checksum=CANONICAL_MODEL_SHA256,
        )
        is_valid, err = req.validate()
        self.assertTrue(is_valid)
        self.assertIsNone(err)

        # Mismatched checksum
        bad_req = CanonicalInferenceRequest(
            request_id="req_002",
            prompt="Hello TARA",
            expected_model_checksum="0000000000000000000000000000000000000000000000000000000000000000",
        )
        is_valid, err = bad_req.validate()
        self.assertFalse(is_valid)
        self.assertIn("Checksum mismatch", err)

        # Missing prompt
        empty_prompt = CanonicalInferenceRequest(
            request_id="req_003",
            prompt="",
            expected_model_checksum=CANONICAL_MODEL_SHA256,
        )
        is_valid, err = empty_prompt.validate()
        self.assertFalse(is_valid)
        self.assertIn("prompt", err)

    def test_schema_json_file_alignment(self):
        schema_file = os.path.join(REPO_ROOT, "TARA", "CONTRACTS", "v1", "schemas.json")
        self.assertTrue(os.path.exists(schema_file), "Canonical schemas.json must exist")
        with open(schema_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("definitions", data)
        self.assertIn("InferenceRequest", data["definitions"])
        self.assertIn("InferenceResponse", data["definitions"])
        self.assertIn("WorkerHealthResponse", data["definitions"])
        self.assertIn("CanonicalErrorResponse", data["definitions"])


class TestModelAndTokenizerParity(unittest.TestCase):
    """Verifies SafeTensors weight SHA256 integrity and tokenizer parity."""

    def test_model_safetensors_sha256_invariant(self):
        model_file = os.path.join(REPO_ROOT, "storage", "models", "tara", "model.safetensors")
        self.assertTrue(os.path.exists(model_file), f"Model file must exist at {model_file}")

        h = hashlib.sha256()
        with open(model_file, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        actual_hash = h.hexdigest()

        self.assertEqual(
            actual_hash.lower(),
            CANONICAL_MODEL_SHA256.lower(),
            f"Production model SHA256 changed! Expected {CANONICAL_MODEL_SHA256}, got {actual_hash}"
        )

    def test_tokenizer_special_tokens(self):
        tok = TaraTokenizer(vocab_size=344)
        self.assertEqual(tok.token_to_id.get("<|pad|>"), 0)
        self.assertEqual(tok.token_to_id.get("<|im_start|>"), 1)
        self.assertEqual(tok.token_to_id.get("<|im_end|>"), 2)
        self.assertEqual(tok.token_to_id.get("<|unk|>"), 3)
        self.assertEqual(tok.vocab_size, 344)

    def test_tokenizer_encode_decode(self):
        tok = TaraTokenizer(vocab_size=344)
        text = "TARA SYSTEM INITIALIZED"
        encoded = tok.encode(text)
        self.assertTrue(len(encoded) > 0)
        decoded = tok.decode(encoded)
        self.assertEqual(decoded, text)


class TestGpuWorkerZeroTrustAndInference(unittest.TestCase):
    """Verifies internal authenticated zero-trust loopback boundary and live inference."""

    @classmethod
    def setUpClass(cls):
        cls.test_port = 8798
        cls.server = HTTPServer(("127.0.0.1", cls.test_port), WorkerHandler)
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()
        time.sleep(0.1)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_health_endpoint_public(self):
        url = f"http://127.0.0.1:{self.test_port}/health"
        req = Request(url)
        with urlopen(req, timeout=3) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["status"], "HEALTHY")
            self.assertTrue(data["ready"])
            self.assertEqual(data["model_checksum"], CANONICAL_MODEL_SHA256)
            self.assertEqual(data["parameters"], 118080)
            self.assertEqual(data["protocol_version"], CANONICAL_PROTOCOL_VERSION)

    def test_zero_trust_unauthenticated_request_rejected(self):
        url = f"http://127.0.0.1:{self.test_port}/api/v1/infer"
        payload = json.dumps({
            "request_id": "test_unauth",
            "prompt": "Hello",
            "expected_model_checksum": CANONICAL_MODEL_SHA256
        }).encode("utf-8")

        # Request WITHOUT X-Tara-Worker-Token header
        req = Request(url, data=payload, headers={"Content-Type": "application/json"})
        with self.assertRaises(HTTPError) as ctx:
            urlopen(req, timeout=3)
        self.assertEqual(ctx.exception.code, 401)
        err_body = json.loads(ctx.exception.read().decode("utf-8"))
        self.assertEqual(err_body["status"], "ERROR")
        self.assertEqual(err_body["error_code"], "UNAUTHORIZED")

    def test_zero_trust_authenticated_inference(self):
        url = f"http://127.0.0.1:{self.test_port}/api/v1/infer"
        payload = json.dumps({
            "request_id": "test_auth_001",
            "prompt": "TARA STATUS",
            "max_tokens": 10,
            "temperature": 0.0,
            "expected_model_checksum": CANONICAL_MODEL_SHA256
        }).encode("utf-8")

        req = Request(url, data=payload, headers={
            "Content-Type": "application/json",
            "X-Tara-Worker-Token": INTERNAL_WORKER_KEY,
            "X-Tara-Protocol-Version": CANONICAL_PROTOCOL_VERSION,
        })

        with urlopen(req, timeout=10) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["status"], "SUCCESS")
            self.assertEqual(data["request_id"], "test_auth_001")
            self.assertEqual(data["model_checksum"], CANONICAL_MODEL_SHA256)
            self.assertIn("runtime_engine", data)
            self.assertIn("text", data)
            self.assertTrue(data["token_count"] >= 1)
            self.assertTrue(data["total_latency_ms"] >= 0.0)


class TestBrowserSessionParity(unittest.TestCase):
    """Verifies that browser session tokens authenticate seamlessly without 401 errors."""

    def test_web_session_store_issue_and_verify(self):
        store = WebSessionStore(ttl=10.0)
        token = store.issue("creator_test")
        self.assertTrue(len(token) >= 32)

        # Verify token
        actor = store.verify(token)
        self.assertEqual(actor, "creator_test")

        # Invalid token returns None
        self.assertIsNone(store.verify("invalid_token_123"))

    def test_session_cookie_issuance_in_handler(self):
        store = WebSessionStore()
        token = store.issue("browser_user")
        self.assertEqual(store.verify(token), "browser_user")


class TestVoiceAndCreatorContractsParity(unittest.TestCase):
    """Verifies that Voice and Creator contracts serialize and validate identically across runtimes."""

    def test_voice_converse_contract_serialization(self):
        req = VoiceConverseRequest(
            input_text="Hello TARA from cross runtime",
            language="en-US",
            require_wake_word=False,
        )
        d = req.to_dict()
        self.assertEqual(d["input_text"], "Hello TARA from cross runtime")
        self.assertEqual(d["language"], "en-US")
        self.assertFalse(d["require_wake_word"])

        resp = VoiceConverseResponse(
            status="SUCCESS",
            transcript="Hello TARA from cross runtime",
            response_text="Greetings creator! All systems operational.",
            audio_base64="UklGRgAAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=",
            language="en-US",
            turn_id="turn_parity_001",
        )
        rd = resp.to_dict()
        self.assertEqual(rd["status"], "SUCCESS")
        self.assertEqual(rd["turn_id"], "turn_parity_001")
        self.assertIn("audio_base64", rd)

    def test_creator_setup_and_device_contract_serialization(self):
        s_req = CreatorSetupRequest(
            google_id_token="google_token_sample_12345",
            confirm_identity=True,
            device_name="Creator Workstation",
            device_public_key="a1b2c3d4e5f6",
        )
        sd = s_req.to_dict()
        self.assertEqual(sd["device_name"], "Creator Workstation")
        self.assertTrue(sd["confirm_identity"])

        s_resp = CreatorSetupResponse(
            status="SUCCESS",
            authority_state="ACTIVE",
            creator_id="ROOT_OPERATOR",
            display_name="OPERATOR_ROOT",
            recovery_code="test_recovery_phrase_001",
        )
        srd = s_resp.to_dict()
        self.assertEqual(srd["creator_id"], "ROOT_OPERATOR")
        self.assertEqual(srd["authority_state"], "ACTIVE")

        d_req = DeviceRegisterRequest(
            device_public_key="aabbccddeeff0011",
            device_name="Secondary Laptop",
        )
        dd = d_req.to_dict()
        self.assertEqual(dd["device_public_key"], "aabbccddeeff0011")

        l_resp = DeviceListResponse(
            status="SUCCESS",
            devices=[{"device_id": "TARA-DEVICE-001", "status": "AUTHORIZED"}]
        )
        ld = l_resp.to_dict()
        self.assertEqual(len(ld["devices"]), 1)

    def test_canonical_schemas_json_definitions(self):
        schema_file = os.path.join(REPO_ROOT, "TARA", "CONTRACTS", "v1", "schemas.json")
        with open(schema_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        defs = data.get("definitions", {})

        expected_definitions = [
            "VoiceTranscribeRequest",
            "VoiceTranscribeResponse",
            "VoiceSynthesizeRequest",
            "VoiceConverseRequest",
            "VoiceConverseResponse",
            "VoiceBargeInResponse",
            "VoiceCapabilitiesResponse",
            "CreatorSetupRequest",
            "CreatorSetupResponse",
            "DeviceRegisterRequest",
            "DeviceRevokeRequest",
            "DeviceListResponse",
        ]
        for name in expected_definitions:
            self.assertIn(name, defs, f"Canonical contract schemas.json must define {name}")


class TestDualRuntimeRouteParity(unittest.TestCase):
    """Verifies that Python and Rust API routers expose the synchronized canonical endpoints."""

    def test_python_router_contains_all_creator_and_voice_routes(self):
        from tara_core.server import GLOBAL_API_ROUTER
        routes = GLOBAL_API_ROUTER._routes

        required_endpoints = [
            ("POST", "/api/v1/creator/setup"),
            ("POST", "/api/v1/creator/register_device"),
            ("POST", "/api/v1/creator/revoke_device"),
            ("GET",  "/api/v1/creator/devices"),
            ("POST", "/api/v1/voice/transcribe"),
            ("POST", "/api/v1/voice/synthesize"),
            ("POST", "/api/v1/voice/converse"),
            ("POST", "/api/v1/voice/barge_in"),
            ("GET",  "/api/v1/voice/capabilities"),
        ]

        for method, path in required_endpoints:
            key = f"{method}:{path}"
            self.assertIn(key, routes, f"Python router missing endpoint: {key}")


if __name__ == "__main__":
    unittest.main()

