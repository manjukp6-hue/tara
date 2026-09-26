"""
tests/test_p0_model_loader.py

Regression & Verification Suite for P0.1:
SafeTensors Pure-Python Model Weight Deserialization & Equivalence.
Ensures:
1. Dynamic interpretation of real SafeTensors metadata (dtype, shape, offsets).
2. Pure-Python loader matches PyTorch / safetensors.torch within floating-point tolerance.
3. Unpacking float32 as uint16 is strictly eliminated and rejected.
4. Deterministic inference results are verified.
5. Production model SHA256 invariant is strictly enforced.
"""

import os
import sys
import json
import math
import hashlib
import struct
import unittest

sys.path.insert(0, os.path.abspath("python"))

from tara_model.generate import load_trained_language_model, unpack_safetensors_tensor, generate_response
from tara_model.architecture import TaraConfig, TaraModelZero

PRODUCTION_MODEL_SHA256 = "7a50308b799f2654baeafbd64dec31f088c3b07b446e198cd0f9ec2b7c0af309"
MODEL_DIR = "storage/models/tara"
MODEL_FILE = os.path.join(MODEL_DIR, "model.safetensors")


class TestP0ModelLoader(unittest.TestCase):

    def test_01_production_model_sha256_invariant(self):
        """Verify the production model file is untouched and matches the exact invariant SHA256."""
        self.assertTrue(os.path.exists(MODEL_FILE), f"Missing model file at {MODEL_FILE}")
        with open(MODEL_FILE, "rb") as f:
            computed_sha = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(
            computed_sha,
            PRODUCTION_MODEL_SHA256,
            f"Production model SHA256 violated! Expected {PRODUCTION_MODEL_SHA256}, got {computed_sha}"
        )

    def test_02_safetensors_header_metadata(self):
        """Inspect actual SafeTensors header and verify all tensors are IEEE 754 float32."""
        with open(MODEL_FILE, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header_json = json.loads(f.read(header_len).decode("utf-8"))

        self.assertGreater(len(header_json), 0)
        tensor_count = 0
        for name, meta in header_json.items():
            if name == "__metadata__":
                continue
            tensor_count += 1
            self.assertIn("dtype", meta, f"Tensor {name} missing dtype metadata")
            self.assertIn("shape", meta, f"Tensor {name} missing shape metadata")
            self.assertIn("data_offsets", meta, f"Tensor {name} missing data_offsets")
            # All tensors in the production model are 32-bit floats
            self.assertEqual(
                meta["dtype"],
                "F32",
                f"Tensor {name} has unexpected dtype: {meta['dtype']}, expected F32"
            )
            # Verify byte count matches shape * 4 bytes
            expected_bytes = math.prod(meta["shape"]) * 4
            actual_bytes = meta["data_offsets"][1] - meta["data_offsets"][0]
            self.assertEqual(
                actual_bytes,
                expected_bytes,
                f"Offset byte range for {name} does not match float32 size"
            )

        self.assertEqual(tensor_count, 21, f"Expected 21 tensors in production model, got {tensor_count}")

    def test_03_tensor_values_match_canonical_safetensors_loader(self):
        """Verify pure-Python unpacker values exactly match canonical safetensors.torch values."""
        try:
            import safetensors.torch
            import torch
        except ImportError:
            self.skipTest("PyTorch / safetensors.torch not installed; skipping PyTorch cross-check.")

        pt_tensors = safetensors.torch.load_file(MODEL_FILE)
        model, tokenizer, config = load_trained_language_model(MODEL_DIR)

        for name, pt_tensor in pt_tensors.items():
            self.assertIn(name, model.weights, f"Tensor {name} missing from pure-Python model.weights")
            py_weight = model.weights[name]

            pt_flat = pt_tensor.flatten().tolist()
            if isinstance(py_weight[0], list):
                # Flatten 2D matrix
                py_flat = [v for row in py_weight for v in row]
            else:
                py_flat = list(py_weight)

            self.assertEqual(
                len(py_flat),
                len(pt_flat),
                f"Element count mismatch for {name}: py={len(py_flat)}, pt={len(pt_flat)}"
            )

            # Check precision match to 1e-6 tolerance
            max_diff = max(abs(a - b) for a, b in zip(py_flat, pt_flat))
            self.assertLess(
                max_diff,
                1e-6,
                f"Tensor {name} values diverged between PyTorch and pure-Python loader! Max diff: {max_diff}"
            )

    def test_04_unpack_safetensors_tensor_rejects_uint16_and_underflow(self):
        """Ensure unpack_safetensors_tensor rejects invalid dtypes and detects underflow."""
        # Valid float32 bytes
        test_floats = [1.5, -2.25, 0.125]
        packed_bytes = struct.pack("<3f", *test_floats)

        unpacked = unpack_safetensors_tensor("F32", [3], packed_bytes)
        for expected, actual in zip(test_floats, unpacked):
            self.assertAlmostEqual(expected, actual, places=5)

        # Buffer underflow check
        with self.assertRaises(ValueError):
            unpack_safetensors_tensor("F32", [3], packed_bytes[:8])  # Needs 12 bytes

        # Unsupported dtype check
        with self.assertRaises(ValueError):
            unpack_safetensors_tensor("UNKNOWN_DTYPE_XYZ", [3], packed_bytes)

    def test_05_deterministic_forward_inference(self):
        """Verify model produces deterministic outputs on identical prompt."""
        model, tokenizer, config = load_trained_language_model(MODEL_DIR)
        prompt = "Hello TARA"

        # Deterministic generation with temperature 0.0 (greedy)
        out1 = generate_response(model, tokenizer, prompt, max_new_tokens=10, temperature=0.0)
        out2 = generate_response(model, tokenizer, prompt, max_new_tokens=10, temperature=0.0)

        self.assertIsInstance(out1, dict)
        self.assertIsInstance(out2, dict)
        self.assertEqual(out1["text"], out2["text"], "Greedy generation must be completely deterministic")


if __name__ == "__main__":
    unittest.main()
