"""
tests/test_multimodal_sensory_engine.py

Verification suite for Native Multi-Modal Sensory Engine (Vision, Acoustic, Fusion).
"""

import unittest
import math
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

from tara_core.multimodal_sensory_engine import (
    VisionPerceptionEngine,
    AcousticPerceptionEngine,
    CrossModalSensorFusion
)


class TestMultimodalSensoryEngine(unittest.TestCase):

    def setUp(self):
        self.vision = VisionPerceptionEngine()
        self.acoustic = AcousticPerceptionEngine()
        self.fusion = CrossModalSensorFusion()

    def test_01_vision_feature_extraction(self):
        # Create an 8x8 synthetic image with high contrast checkerboard pattern
        pixels = [
            [255 if (r + c) % 2 == 0 else 0 for c in range(8)]
            for r in range(8)
        ]
        feat = self.vision.process_raw_pixels(pixels, width=8, height=8)

        self.assertEqual(feat.width, 8)
        self.assertEqual(feat.height, 8)
        self.assertAlmostEqual(feat.mean_luminance, 127.5, delta=1.0)
        self.assertGreater(feat.contrast_std, 100.0)
        self.assertGreater(feat.edge_density, 0.5)
        self.assertEqual(len(feat.spatial_quadrants), 4)
        self.assertGreater(len(feat.visual_hash), 0)

    def test_02_acoustic_spectral_decomposition(self):
        # Generate a pure 440 Hz sinusoidal waveform (A4 concert pitch) at 16kHz
        sr = 16000
        duration = 0.05  # 50 ms
        num_samples = int(sr * duration)
        sine_wave = [0.8 * math.sin(2 * math.pi * 440 * (i / sr)) for i in range(num_samples)]

        feat = self.acoustic.process_waveform(sine_wave, sample_rate=sr)

        self.assertEqual(feat.sample_rate, sr)
        self.assertGreater(feat.rms_energy, 0.4)
        self.assertGreater(feat.zero_crossing_rate, 0.04)
        self.assertGreater(feat.spectral_centroid, 100.0)
        self.assertIn("mid", feat.frequency_bands)
        self.assertGreater(len(feat.acoustic_hash), 0)

    def test_03_cross_modal_sensor_fusion(self):
        pixels = [[100] * 4 for _ in range(4)]
        sine_wave = [0.1 * i for i in range(100)]

        fused = self.fusion.fuse_multimodal_inputs(
            text_context="Observe robotic gripper posture and motor hum",
            raw_image=pixels,
            image_dims=(4, 4),
            raw_audio=sine_wave,
            audio_sr=16000
        )

        self.assertEqual(set(fused["active_modalities"]), {"TEXT", "VISION", "AUDIO"})
        self.assertEqual(fused["fusion_coherence_score"], 1.0)
        self.assertIsNotNone(fused["vision"]["hash"])
        self.assertIsNotNone(fused["acoustic"]["hash"])
        self.assertIn("Observe", fused["text"])


if __name__ == "__main__":
    unittest.main()
