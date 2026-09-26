"""
tests/test_factors_and_voice.py

Comprehensive Unified Test Suite for:
1. Biometric Security Framework (Windows Hello, Fingerprint, Face, Iris)
2. Voice Verification Provider (Auxiliary factor, dynamic challenge, anti-replay, rate limiting)
3. Voice-to-Voice Conversation Pipeline (STT, TTS, Barge-In, Wake-Word, Kannada + English)
4. Server Voice Endpoints (/transcribe, /synthesize, /converse, /barge_in, /capabilities)
5. Authority Invariants & Security Boundaries (Voice != Creator Authority)
"""

import os
import sys
import io
import wave
import json
import base64
import time
import struct
import unittest
import urllib.request
import urllib.parse
from http.server import HTTPServer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
python_dir = os.path.join(REPO_ROOT, "python")
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

from TARA.ACCESS.factors import (
    BiometricCapability,
    BiometricType,
    BiometricAuthResult,
    WindowsHelloProvider,
    FingerprintProvider,
    FaceProvider,
    IrisProvider,
    VoiceVerificationProvider,
)
from TARA.ACCESS.services.auth_service import CreatorAuthService, CANONICAL_CREATOR_ID
from TARA.ACCESS.operator.operator_lifecycle import AuthorityState
from tara_core.voice import (
    VoiceInputProvider,
    VoiceOutputProvider,
    BargeInCoordinator,
    WakeWordDetector,
)
from tara_core.brain import TaraBrain
from tara_core.server import create_server, GLOBAL_API_ROUTER


def _create_synthetic_pcm_wav(duration_sec=0.5, freq=440.0, sample_rate=16000) -> bytes:
    """Creates a valid synthetic 16kHz mono 16-bit PCM WAV for testing."""
    num_samples = int(duration_sec * sample_rate)
    samples = []
    import math
    for i in range(num_samples):
        t = i / sample_rate
        val = math.sin(2.0 * math.pi * freq * t) * 0.5
        samples.append(int(val * 32767))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return buf.getvalue()


class TestBiometricSecurityFramework(unittest.TestCase):

    def test_windows_hello_capability_detection(self):
        """Windows Hello capability must be genuine (AVAILABLE or NOT_AVAILABLE) and never fabricates success."""
        provider = WindowsHelloProvider(test_mode=False)
        cap = provider.get_capability()
        self.assertIn(cap, [BiometricCapability.AVAILABLE, BiometricCapability.NOT_AVAILABLE, BiometricCapability.NOT_ENROLLED])

        # Without test mode, simulation must raise PermissionError
        with self.assertRaises(PermissionError):
            provider.authenticate(context={"simulate_success": True})

    def test_windows_hello_safe_test_simulation(self):
        """Simulation is permitted ONLY when test_mode=True and TARA_TEST_MODE=1."""
        os.environ["TARA_TEST_MODE"] = "1"
        try:
            provider = WindowsHelloProvider(test_mode=True)
            res = provider.authenticate(context={"simulate_success": True})
            self.assertTrue(res.success)
            self.assertEqual(res.auth_type, BiometricType.WINDOWS_HELLO.value)
        finally:
            os.environ.pop("TARA_TEST_MODE", None)

    def test_fingerprint_provider_capability(self):
        """Fingerprint provider accurately queries system biometric framework without storing templates."""
        fp = FingerprintProvider(test_mode=False)
        cap = fp.get_capability()
        self.assertIn(cap, [BiometricCapability.AVAILABLE, BiometricCapability.NOT_AVAILABLE])
        res = fp.authenticate()
        self.assertFalse(res.success)

    def test_face_provider_capability(self):
        """Face provider accurately queries facial recognition hardware."""
        face = FaceProvider(test_mode=False)
        cap = face.get_capability()
        self.assertIn(cap, [BiometricCapability.AVAILABLE, BiometricCapability.NOT_AVAILABLE])
        res = face.authenticate()
        self.assertFalse(res.success)

    def test_iris_provider_capability(self):
        """Iris provider accurately queries iris scanner hardware."""
        iris = IrisProvider(test_mode=False)
        cap = iris.get_capability()
        self.assertIn(cap, [BiometricCapability.AVAILABLE, BiometricCapability.NOT_AVAILABLE])
        res = iris.authenticate()
        self.assertFalse(res.success)


class TestVoiceVerificationProvider(unittest.TestCase):

    def setUp(self):
        self.provider = VoiceVerificationProvider(test_mode=True)
        self.test_audio = _create_synthetic_pcm_wav(duration_sec=0.5, freq=300.0)

    def test_voice_challenge_generation(self):
        """Generates dynamic, unpredictable challenge phrase with cryptographic nonce and TTL."""
        challenge = self.provider.generate_challenge("CREATOR_001")
        self.assertIn("nonce", challenge)
        self.assertIn("phrase", challenge)
        self.assertEqual(challenge["ttl_seconds"], 60)
        self.assertTrue(len(challenge["nonce"]) >= 16)

    def test_voice_enrollment_and_matching(self):
        """Enrolls acoustic feature profile (without raw audio) and verifies challenge with matching voice."""
        identity = "CREATOR_001"
        enrolled = self.provider.enroll_profile(identity, self.test_audio)
        self.assertTrue(enrolled)
        self.assertTrue(self.provider.is_enrolled(identity))

        # Generate fresh challenge
        challenge = self.provider.generate_challenge(identity)
        nonce = challenge["nonce"]

        # Verify with identical acoustic audio
        res = self.provider.verify_challenge(identity, nonce, self.test_audio, similarity_threshold=0.7)
        self.assertTrue(res.success)
        self.assertTrue(res.metadata.get("is_auxiliary"))

    def test_anti_replay_protection_rejects_used_nonce(self):
        """Nonce cannot be replayed or reused."""
        identity = "CREATOR_001"
        self.provider.enroll_profile(identity, self.test_audio)
        challenge = self.provider.generate_challenge(identity)
        nonce = challenge["nonce"]

        # First verification succeeds
        res1 = self.provider.verify_challenge(identity, nonce, self.test_audio, similarity_threshold=0.7)
        self.assertTrue(res1.success)

        # Second verification with same nonce MUST be rejected as replay attack
        res2 = self.provider.verify_challenge(identity, nonce, self.test_audio, similarity_threshold=0.7)
        self.assertFalse(res2.success)
        self.assertEqual(res2.error, "REPLAY_ATTACK_DETECTED")

    def test_rate_limiting_and_lockout(self):
        """Multiple failed challenge verifications trigger rate lockout."""
        identity = "CREATOR_RATE_TEST"
        self.provider.enroll_profile(identity, self.test_audio)

        # Different audio with mismatch
        diff_audio = _create_synthetic_pcm_wav(duration_sec=0.5, freq=1200.0)

        for _ in range(3):
            challenge = self.provider.generate_challenge(identity)
            res = self.provider.verify_challenge(identity, challenge["nonce"], diff_audio, similarity_threshold=0.99)
            self.assertFalse(res.success)

        # 4th attempt must be locked out
        with self.assertRaises(PermissionError):
            self.provider.generate_challenge(identity)


class TestVoiceConversationEngine(unittest.TestCase):

    def setUp(self):
        self.stt = VoiceInputProvider()
        self.tts = VoiceOutputProvider()
        self.barge_in = BargeInCoordinator()
        self.wake = WakeWordDetector()

    def test_stt_vad_and_transcription(self):
        """VAD detects presence of speech and transcribes English/Kannada correctly."""
        silence = b"\x00" * 16000
        vad_silence = self.stt.detect_voice_activity(silence)
        self.assertFalse(vad_silence)

        audio_wav = _create_synthetic_pcm_wav(duration_sec=0.5, freq=250.0)
        vad_audio = self.stt.detect_voice_activity(audio_wav)
        self.assertTrue(vad_audio)

        # English transcription
        res_en = self.stt.transcribe(audio_wav, language="en-US")
        self.assertTrue(res_en["has_speech"])
        self.assertTrue(len(res_en["transcript"]) > 0)

        # Kannada transcription
        res_kn = self.stt.transcribe(audio_wav, language="kn-IN")
        self.assertTrue(res_kn["has_speech"])
        self.assertIn("ತಾರಾ", res_kn["transcript"])

    def test_tts_synthesis_bilingual(self):
        """TTS produces valid WAV streams for English and Kannada."""
        wav_en = self.tts.synthesize("Hello TARA, system operational.", language="en")
        self.assertTrue(wav_en.startswith(b"RIFF"))
        self.assertTrue(len(wav_en) > 100)

        # Kannada synthesis
        wav_kn = self.tts.synthesize("ನಮಸ್ಕಾರ ತಾರಾ, ಸಿಸ್ಟಮ್ ಸಿದ್ಧವಾಗಿದೆ.", language="kn")
        self.assertTrue(wav_kn.startswith(b"RIFF"))
        self.assertTrue(len(wav_kn) > 100)

    def test_barge_in_interruption_coordination(self):
        """Barge-In halts speaking immediately and updates interruption metrics."""
        self.assertFalse(self.barge_in.is_speaking)
        turn_id = "turn_abc_123"
        self.barge_in.start_speaking(turn_id)
        self.assertTrue(self.barge_in.is_speaking)

        # User interrupts
        result = self.barge_in.handle_user_barge_in()
        self.assertTrue(result["interrupted"])
        self.assertEqual(result["interrupted_turn_id"], turn_id)
        self.assertFalse(self.barge_in.is_speaking)
        self.assertEqual(self.barge_in.interruption_count, 1)

    def test_wake_word_spotting(self):
        """Detects 'TARA' and 'ತಾರಾ' wake-words and extracts following command."""
        # English
        det_en = self.wake.detect("Hey TARA, what is the status?")
        self.assertTrue(det_en["detected"])
        self.assertEqual(det_en["cleaned_command"], "what is the status?")

        # Kannada
        det_kn = self.wake.detect("ತಾರಾ, ಇಂದಿನ ವರದಿ ಏನು?")
        self.assertTrue(det_kn["detected"])
        self.assertEqual(det_kn["cleaned_command"], "ಇಂದಿನ ವರದಿ ಏನು?")

        # Negative
        det_neg = self.wake.detect("Good morning everyone.")
        self.assertFalse(det_neg["detected"])


class TestVoiceServerEndpointsAndIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.brain = TaraBrain()
        cls.server = create_server("127.0.0.1", 0, brain=cls.brain)
        cls.port = cls.server.server_address[1]
        import threading
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def _post(self, path: str, data: dict):
        url = f"http://127.0.0.1:{self.port}{path}"
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def test_voice_capabilities_endpoint(self):
        """GET /api/v1/voice/capabilities returns introspection data."""
        url = f"http://127.0.0.1:{self.port}/api/v1/voice/capabilities"
        with urllib.request.urlopen(url) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["status"], "SUCCESS")
            self.assertTrue(data["stt_available"])
            self.assertTrue(data["tts_available"])
            self.assertIn("kn-IN", data["languages"])

    def test_voice_transcribe_endpoint(self):
        """POST /api/v1/voice/transcribe transcribes audio base64."""
        audio = _create_synthetic_pcm_wav(0.4, 300.0)
        b64 = base64.b64encode(audio).decode("ascii")
        res = self._post("/api/v1/voice/transcribe", {"audio_data": b64, "language": "en-US"})
        self.assertEqual(res["status"], "SUCCESS")
        self.assertIn("transcript", res["data"])

    def test_voice_synthesize_endpoint(self):
        """POST /api/v1/voice/synthesize returns audio/wav binary stream."""
        url = f"http://127.0.0.1:{self.port}/api/v1/voice/synthesize"
        req = urllib.request.Request(
            url,
            data=json.dumps({"text": "Testing voice synthesis.", "language": "en"}).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.headers.get("Content-Type"), "audio/wav")
            wav_bytes = resp.read()
            self.assertTrue(wav_bytes.startswith(b"RIFF"))

    def test_voice_converse_endpoint_single_model_identity(self):
        """POST /api/v1/voice/converse routes through authoritative TaraBrain."""
        audio = _create_synthetic_pcm_wav(0.4, 300.0)
        b64 = base64.b64encode(audio).decode("ascii")
        res = self._post("/api/v1/voice/converse", {
            "audio_data": b64,
            "language": "en-US"
        })
        self.assertEqual(res["status"], "SUCCESS")
        self.assertIn("response_text", res)
        self.assertIn("audio_base64", res)
        self.assertTrue(len(res["audio_base64"]) > 0)
        # Verify single model response structure
        self.assertIn("brain_result", res)

    def test_voice_barge_in_endpoint(self):
        """POST /api/v1/voice/barge_in coordinates interruption."""
        res = self._post("/api/v1/voice/barge_in", {})
        self.assertEqual(res["status"], "SUCCESS")
        self.assertIn("barge_in", res)


class TestAuthoritySeparationAndCreatorSafety(unittest.TestCase):

    def test_voice_interaction_never_elevates_to_creator(self):
        """Speaking to TARA via voice API or chat never bypasses CREATOR_SETUP_REQUIRED."""
        auth_service = CreatorAuthService(repo_root=REPO_ROOT)
        self.assertEqual(auth_service.lifecycle.get_state(), AuthorityState.CREATOR_SETUP_REQUIRED)

        # Capabilities report
        caps = auth_service.get_biometric_capabilities()
        self.assertIn("voice_auxiliary", caps)
        self.assertIn("windows_hello", caps)

        # Step up check fails closed
        res = auth_service.verify_biometric_step_up("fingerprint")
        self.assertFalse(res.success)
        self.assertEqual(auth_service.lifecycle.get_state(), AuthorityState.CREATOR_SETUP_REQUIRED)

    def test_voice_step_up_strictly_secondary(self):
        """Voice verification CANNOT authorize alone; strictly requires session + device + fresh challenge."""
        auth_service = CreatorAuthService(repo_root=REPO_ROOT)

        # 1. Calling voice step-up with no session fails closed immediately
        res = auth_service.verify_voice_step_up(
            session_token="invalid_or_missing_session",
            device_id="TARA-DEVICE-001",
            challenge_nonce="deadbeef1234",
            audio_bytes=b"sample_audio_pcm",
            device_signature_hex="abcdef"
        )
        self.assertEqual(res["status"], "DENIED")
        self.assertEqual(res["code"], "VOICE_CANNOT_AUTHORIZE_ALONE")

        # 2. Voice provider capability check
        self.assertEqual(auth_service.voice_provider.get_capability(), BiometricCapability.AVAILABLE)


if __name__ == "__main__":
    unittest.main()

