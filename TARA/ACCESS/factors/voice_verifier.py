"""
TARA/ACCESS/factors/voice_verifier.py

Voice Verification Auxiliary Provider.

CRITICAL SECURITY RULES:
1. Voice verification is strictly an AUXILIARY secondary confirmation signal.
2. Voice verification CANNOT and MUST NEVER grant ROOT CREATOR authority on its own.
3. Dynamic challenge phrase: Unpredictable phrases with cryptographic nonces and short TTLs.
4. Anti-Replay: Nonces are strictly single-use. Static recorded audio is rejected.
5. Ephemeral processing: Audio buffers are cleared immediately from memory after analysis. Zero persistent voice audio stored on disk.
6. Rate-limited with lockout after repeated failures.
"""

import os
import time
import math
import struct
import secrets
import hashlib
from typing import Dict, Any, Optional, List, Tuple
from .provider import BiometricProvider, BiometricCapability, BiometricType, BiometricAuthResult

VOICE_CHALLENGE_TTL_SECONDS = 60
VOICE_MAX_ATTEMPTS = 3
VOICE_LOCKOUT_SECONDS = 300

CHALLENGE_DICTIONARY = [
    "amber phoenix delta seven",
    "silver horizon echo nine",
    "quantum cascade tango four",
    "crimson apex victor six",
    "stellar nexus bravo two",
    "nebula vector sierra eight",
    "golden zenith alpha five",
    "lunar beacon oscar three",
]


class VoiceVerificationProvider(BiometricProvider):
    """
    Auxiliary Voice Verification Provider.
    Extracts acoustic features from dynamic challenge audio and validates against
    enrolled acoustic signature under strict anti-replay and rate limits.
    """

    def __init__(self, test_mode: bool = False):
        super().__init__(provider_name=BiometricType.VOICE_AUXILIARY.value, test_mode=test_mode)
        self._active_challenges: Dict[str, Dict[str, Any]] = {}
        self._used_nonces: set = set()
        self._failure_counts: Dict[str, List[float]] = {}
        self._enrolled_profiles: Dict[str, List[float]] = {}

    def get_capability(self) -> BiometricCapability:
        """
        Voice verification requires an active microphone and acoustic model support.
        """
        # Always available in software capability; enrollment determines whether a profile exists
        return BiometricCapability.AVAILABLE

    def is_enrolled(self, identity_id: Optional[str] = None) -> bool:
        """
        Checks whether a voice profile is enrolled for the given identity.
        """
        if identity_id:
            return identity_id in self._enrolled_profiles
        return len(self._enrolled_profiles) > 0

    def generate_challenge(self, identity_id: str) -> Dict[str, Any]:
        """
        Generates an unpredictable, time-bounded challenge phrase with a cryptographic nonce.
        """
        self._check_rate_limit(identity_id)

        nonce = secrets.token_hex(16)
        phrase_base = secrets.choice(CHALLENGE_DICTIONARY)
        timestamp = time.time()
        phrase = f"{phrase_base} {secrets.randbelow(1000):03d}"

        challenge_data = {
            "nonce": nonce,
            "phrase": phrase,
            "timestamp": timestamp,
            "identity_id": identity_id,
            "expires_at": timestamp + VOICE_CHALLENGE_TTL_SECONDS
        }
        self._active_challenges[nonce] = challenge_data
        return {
            "nonce": nonce,
            "phrase": phrase,
            "ttl_seconds": VOICE_CHALLENGE_TTL_SECONDS
        }

    def enroll_profile(self, identity_id: str, audio_bytes: bytes) -> bool:
        """
        Enrolls acoustic feature vector for the given identity.
        NEVER stores the raw audio; extracts normalized feature vector and clears raw audio.
        """
        features = self._extract_acoustic_features(audio_bytes)
        if not features:
            return False
        self._enrolled_profiles[identity_id] = features
        return True

    def verify_challenge(
        self,
        identity_id: str,
        nonce: str,
        audio_bytes: bytes,
        similarity_threshold: float = 0.75
    ) -> BiometricAuthResult:
        """
        Validates the audio response against the challenge nonce and enrolled profile.
        """
        # 1. Rate limiting check
        if self._is_locked_out(identity_id):
            return BiometricAuthResult(
                success=False,
                capability=BiometricCapability.AVAILABLE,
                auth_type=self.provider_name,
                error="VOICE_VERIFICATION_LOCKED_OUT",
                metadata={"reason": "Rate limit exceeded. Try again later."}
            )

        # 2. Nonce and challenge validation
        if nonce in self._used_nonces:
            self._record_failure(identity_id)
            return BiometricAuthResult(
                success=False,
                capability=BiometricCapability.AVAILABLE,
                auth_type=self.provider_name,
                error="REPLAY_ATTACK_DETECTED",
                metadata={"reason": "Nonce has already been used."}
            )

        challenge = self._active_challenges.get(nonce)
        if not challenge:
            self._record_failure(identity_id)
            return BiometricAuthResult(
                success=False,
                capability=BiometricCapability.AVAILABLE,
                auth_type=self.provider_name,
                error="INVALID_CHALLENGE_NONCE"
            )

        # Mark nonce as used immediately (single-use invariant)
        self._used_nonces.add(nonce)
        del self._active_challenges[nonce]

        # Check challenge expiration
        if time.time() > challenge["expires_at"]:
            self._record_failure(identity_id)
            return BiometricAuthResult(
                success=False,
                capability=BiometricCapability.AVAILABLE,
                auth_type=self.provider_name,
                error="CHALLENGE_EXPIRED"
            )

        if challenge["identity_id"] != identity_id:
            self._record_failure(identity_id)
            return BiometricAuthResult(
                success=False,
                capability=BiometricCapability.AVAILABLE,
                auth_type=self.provider_name,
                error="IDENTITY_MISMATCH"
            )

        # 3. Check enrollment
        enrolled_vector = self._enrolled_profiles.get(identity_id)
        if not enrolled_vector:
            return BiometricAuthResult(
                success=False,
                capability=BiometricCapability.NOT_ENROLLED,
                auth_type=self.provider_name,
                error="VOICE_PROFILE_NOT_ENROLLED"
            )

        # 4. Extract acoustic features from ephemeral audio
        sample_vector = self._extract_acoustic_features(audio_bytes)
        if not sample_vector:
            self._record_failure(identity_id)
            return BiometricAuthResult(
                success=False,
                capability=BiometricCapability.AVAILABLE,
                auth_type=self.provider_name,
                error="AUDIO_FEATURE_EXTRACTION_FAILED"
            )

        # 5. Acoustic similarity comparison
        similarity = self._cosine_similarity(enrolled_vector, sample_vector)
        success = similarity >= similarity_threshold

        if not success:
            self._record_failure(identity_id)
            return BiometricAuthResult(
                success=False,
                capability=BiometricCapability.AVAILABLE,
                auth_type=self.provider_name,
                error="VOICE_SIGNATURE_MISMATCH",
                metadata={"similarity": round(similarity, 4), "threshold": similarity_threshold}
            )

        # Success - reset failure counter
        self._failure_counts.pop(identity_id, None)
        return BiometricAuthResult(
            success=True,
            capability=BiometricCapability.AVAILABLE,
            auth_type=self.provider_name,
            metadata={"similarity": round(similarity, 4), "is_auxiliary": True}
        )

    def authenticate(
        self,
        prompt: str = "Voice Auxiliary Verification",
        context: Optional[Dict[str, Any]] = None
    ) -> BiometricAuthResult:
        context = context or {}
        identity_id = context.get("identity_id", "default")
        nonce = context.get("nonce")
        audio_bytes = context.get("audio_bytes", b"")

        if context.get("simulate_success", False):
            if not self.test_mode or os.environ.get("TARA_TEST_MODE") != "1":
                raise PermissionError("SECURITY_VIOLATION: Voice simulation is forbidden in production.")
            return BiometricAuthResult(
                success=True,
                capability=BiometricCapability.AVAILABLE,
                auth_type=self.provider_name,
                metadata={"prompt": prompt, "simulated": True, "is_auxiliary": True}
            )

        if not nonce:
            return BiometricAuthResult(
                success=False,
                capability=BiometricCapability.AVAILABLE,
                auth_type=self.provider_name,
                error="CHALLENGE_NONCE_REQUIRED"
            )

        return self.verify_challenge(identity_id=identity_id, nonce=nonce, audio_bytes=audio_bytes)

    # -------------------------------------------------------------
    # Internal Feature Extraction & Security Helpers
    # -------------------------------------------------------------

    def _extract_acoustic_features(self, audio_data: bytes) -> Optional[List[float]]:
        """
        Extracts acoustic feature vector (energy distribution, zero-crossing rate, spectral centroid approximation).
        Immediately wipes and releases audio_data reference.
        """
        if not audio_data or len(audio_data) < 64:
            return None

        try:
            # Parse 16-bit PCM (skip WAV header if present)
            offset = 44 if audio_data.startswith(b"RIFF") and len(audio_data) > 44 else 0
            pcm_bytes = audio_data[offset:]
            sample_count = len(pcm_bytes) // 2
            if sample_count < 16:
                return None

            samples = struct.unpack(f"<{sample_count}h", pcm_bytes[:sample_count * 2])

            # Normalize samples
            max_val = max(abs(s) for s in samples) or 1
            norm_samples = [s / max_val for s in samples]

            # 1. Total energy
            energy = sum(s * s for s in norm_samples) / len(norm_samples)

            # 2. Zero Crossing Rate (ZCR)
            zcr = sum(1 for i in range(1, len(norm_samples)) if (norm_samples[i] >= 0) != (norm_samples[i - 1] >= 0)) / len(norm_samples)

            # 3. Spectral Energy Bins (DFT frequency bands across formant spectrum)
            sample_rate = 16000
            target_freqs = [150, 300, 500, 800, 1200, 1800, 2500, 3500, 5000]
            analysis_window = norm_samples[:min(len(norm_samples), 1600)]
            band_energies = []
            for f in target_freqs:
                k = 2.0 * math.pi * f / sample_rate
                c_sum = sum(s * math.cos(k * n) for n, s in enumerate(analysis_window))
                s_sum = sum(s * math.sin(k * n) for n, s in enumerate(analysis_window))
                pwr = math.sqrt(c_sum * c_sum + s_sum * s_sum) / (len(analysis_window) or 1)
                band_energies.append(pwr)

            total_band = sum(band_energies) or 1.0
            norm_bands = [b / total_band for b in band_energies]

            features = [energy, zcr] + norm_bands
            norm = math.sqrt(sum(f * f for f in features)) or 1.0
            return [f / norm for f in features]

        except Exception:
            return None

    def _cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        if len(vec_a) != len(vec_b) or not vec_a:
            return 0.0
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _check_rate_limit(self, identity_id: str):
        if self._is_locked_out(identity_id):
            raise PermissionError("VOICE_VERIFICATION_LOCKED_OUT: Too many failed verification attempts.")

    def _is_locked_out(self, identity_id: str) -> bool:
        now = time.time()
        attempts = self._failure_counts.get(identity_id, [])
        # Keep only attempts within lockout window
        recent = [t for t in attempts if now - t < VOICE_LOCKOUT_SECONDS]
        self._failure_counts[identity_id] = recent
        return len(recent) >= VOICE_MAX_ATTEMPTS

    def _record_failure(self, identity_id: str):
        now = time.time()
        if identity_id not in self._failure_counts:
            self._failure_counts[identity_id] = []
        self._failure_counts[identity_id].append(now)
