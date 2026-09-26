"""
tests/test_device_capability_system.py
================================================================================
Comprehensive Verification Suite for TARA Automatic Capability Detection &
Model Loading Architecture
================================================================================

Tests:
1. High-RAM PC
2. Low-RAM PC
3. GPU PC
4. CPU-Only PC
5. Mobile-like Resource Profile
6. Single-File Model
7. 100 MB Sharded Model
8. Quantized Model (INT8 & INT4)
9. Lazy Loading & LRU Cache Eviction
10. Fallback Behavior
11. Insufficient-Memory Protection
"""

import os
import sys
import json
import struct
import tempfile
import unittest

# Ensure workspace is in sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from TARA.MODEL.device_detector import DeviceCapabilityDetector, DeviceProfile, GpuInfo, GB, MB
from TARA.MODEL.loading_policy import (
    LoadingPolicyEngine,
    LoadingPlan,
    ModelLoadingStrategy,
    DeviceTarget,
    QuantizationPrecision,
    InsufficientMemoryError
)
from TARA.MODEL.shard_manager import ShardedSafeTensorsManager
from TARA.MODEL.quantization import QuantizationEngine, QuantizedTensor
from TARA.MODEL.auto_loader import AutoTaraModelLoader, LoadedTaraModel


def create_mock_safetensors_file(file_path: str, tensors: dict):
    """Utility to generate a genuine SafeTensors binary file without external libraries."""
    header = {}
    data_bytes = bytearray()
    curr_offset = 0

    for name, float_vals in tensors.items():
        raw_t_bytes = struct.pack(f"<{len(float_vals)}f", *float_vals)
        t_len = len(raw_t_bytes)
        header[name] = {
            "dtype": "F32",
            "shape": [len(float_vals)],
            "data_offsets": [curr_offset, curr_offset + t_len]
        }
        data_bytes.extend(raw_t_bytes)
        curr_offset += t_len

    header_json = json.dumps(header).encode("utf-8")
    header_len = len(header_json)

    with open(file_path, "wb") as f:
        f.write(header_len.to_bytes(8, "little"))
        f.write(header_json)
        f.write(data_bytes)


class TestDeviceCapabilityAndLoader(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_live_hardware_detection(self):
        """Verifies live hardware detection executes without errors on the host."""
        profile = DeviceCapabilityDetector.detect(self.temp_dir)
        self.assertIsNotNone(profile.os_name)
        self.assertGreater(profile.cpu_cores_logical, 0)
        self.assertGreater(profile.total_ram_bytes, 0)
        self.assertGreater(profile.available_ram_bytes, 0)
        self.assertIn(profile.environment_type, ["MOBILE", "DESKTOP", "SERVER"])
        summary = profile.summary()
        self.assertIn("DeviceProfile", summary)

    def test_high_ram_pc(self):
        """Tests High-RAM PC profile selecting FULL_LOAD on host memory."""
        mock_prof = DeviceProfile.mock(
            total_ram_gb=32.0,
            available_ram_gb=24.0,
            gpu_vram_gb=0.0,
            gpu_vendor="NONE",
            is_mobile=False
        )
        plan = LoadingPolicyEngine.create_loading_plan(
            profile=mock_prof,
            model_size_bytes=int(100 * MB),
            param_count=118080
        )
        self.assertEqual(plan.strategy, ModelLoadingStrategy.FULL_LOAD)
        self.assertEqual(plan.device_target, DeviceTarget.CPU)
        self.assertEqual(plan.precision, QuantizationPrecision.NONE_FP32)
        self.assertTrue(len(plan.fallback_chain) > 0)

    def test_low_ram_pc(self):
        """Tests Low-RAM PC profile selecting QUANTIZED_LOAD with lazy loading."""
        mock_prof = DeviceProfile.mock(
            total_ram_gb=4.0,
            available_ram_gb=1.8,
            gpu_vram_gb=0.0,
            gpu_vendor="NONE",
            is_mobile=False
        )
        plan = LoadingPolicyEngine.create_loading_plan(
            profile=mock_prof,
            model_size_bytes=int(100 * MB),
            param_count=118080
        )
        self.assertEqual(plan.strategy, ModelLoadingStrategy.QUANTIZED_LOAD)
        self.assertEqual(plan.device_target, DeviceTarget.CPU)
        self.assertIn(plan.precision, [QuantizationPrecision.INT8, QuantizationPrecision.INT4])
        self.assertLessEqual(plan.max_shard_cache_shards, 2)

    def test_gpu_pc(self):
        """Tests High-End PC with dedicated GPU selecting FULL_LOAD on GPU."""
        mock_prof = DeviceProfile.mock(
            total_ram_gb=32.0,
            available_ram_gb=20.0,
            gpu_vram_gb=8.0,
            gpu_vendor="NVIDIA",
            gpu_name="GeForce RTX 3080",
            is_mobile=False
        )
        plan = LoadingPolicyEngine.create_loading_plan(
            profile=mock_prof,
            model_size_bytes=int(100 * MB),
            param_count=118080
        )
        self.assertEqual(plan.strategy, ModelLoadingStrategy.FULL_LOAD)
        self.assertEqual(plan.device_target, DeviceTarget.GPU)
        self.assertIn(plan.precision, [QuantizationPrecision.NONE_FP32, QuantizationPrecision.FP16])

    def test_cpu_only_pc(self):
        """Tests system with no GPU selecting CPU device target."""
        mock_prof = DeviceProfile.mock(
            total_ram_gb=16.0,
            available_ram_gb=8.0,
            gpu_vram_gb=0.0,
            gpu_vendor="NONE",
            is_mobile=False
        )
        plan = LoadingPolicyEngine.create_loading_plan(
            profile=mock_prof,
            model_size_bytes=int(100 * MB),
            param_count=118080
        )
        self.assertEqual(plan.device_target, DeviceTarget.CPU)

    def test_mobile_resource_profile(self):
        """Tests Mobile environment selecting mobile-compatible quantized representation."""
        mock_prof = DeviceProfile.mock(
            total_ram_gb=4.0,
            available_ram_gb=1.5,
            gpu_vram_gb=0.0,
            gpu_vendor="NONE",
            is_mobile=True,
            cpu_arch="aarch64",
            os_name="Linux"
        )
        plan = LoadingPolicyEngine.create_loading_plan(
            profile=mock_prof,
            model_size_bytes=int(100 * MB),
            param_count=118080
        )
        self.assertEqual(plan.strategy, ModelLoadingStrategy.QUANTIZED_LOAD)
        self.assertEqual(plan.device_target, DeviceTarget.CPU)
        self.assertEqual(plan.precision, QuantizationPrecision.INT4)
        self.assertEqual(plan.max_shard_cache_shards, 1)

    def test_single_file_model(self):
        """Tests discovering and reading a single model.safetensors file."""
        model_dir = os.path.join(self.temp_dir, "single_model")
        os.makedirs(model_dir)

        tensors = {
            "model.embed_tokens.weight": [0.1, 0.2, 0.3, 0.4],
            "lm_head.weight": [0.5, 0.6, 0.7, 0.8]
        }
        create_mock_safetensors_file(os.path.join(model_dir, "model.safetensors"), tensors)

        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump({"hidden_size": 4, "vocab_size": 10, "num_hidden_layers": 1}, f)

        mgr = ShardedSafeTensorsManager(model_dir)
        self.assertEqual(mgr.total_shards, 1)
        self.assertIn("model.embed_tokens.weight", mgr.tensor_names)
        self.assertIn("lm_head.weight", mgr.tensor_names)

        t1 = mgr.load_tensor("model.embed_tokens.weight")
        floats = struct.unpack("<4f", t1["bytes"])
        self.assertAlmostEqual(floats[0], 0.1, places=5)
        self.assertAlmostEqual(floats[3], 0.4, places=5)
        mgr.close()

    def test_100mb_sharded_model(self):
        """Tests discovering and reading multi-shard SafeTensors model with index."""
        model_dir = os.path.join(self.temp_dir, "sharded_model")
        os.makedirs(model_dir)

        shard1_tensors = {"layer.0.weight": [1.0, 2.0, 3.0]}
        shard2_tensors = {"layer.1.weight": [4.0, 5.0, 6.0]}

        s1_name = "model-00001-of-00002.safetensors"
        s2_name = "model-00002-of-00002.safetensors"

        create_mock_safetensors_file(os.path.join(model_dir, s1_name), shard1_tensors)
        create_mock_safetensors_file(os.path.join(model_dir, s2_name), shard2_tensors)

        index_data = {
            "metadata": {"total_size": 1000},
            "weight_map": {
                "layer.0.weight": s1_name,
                "layer.1.weight": s2_name
            }
        }
        with open(os.path.join(model_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(index_data, f)

        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump({"hidden_size": 3, "vocab_size": 10, "num_hidden_layers": 2}, f)

        mgr = ShardedSafeTensorsManager(model_dir, max_cached_shards=1)
        self.assertEqual(mgr.total_shards, 2)
        self.assertEqual(sorted(mgr.tensor_names), ["layer.0.weight", "layer.1.weight"])

        t0 = mgr.load_tensor("layer.0.weight")
        t1 = mgr.load_tensor("layer.1.weight")
        self.assertEqual(struct.unpack("<3f", t0["bytes"]), (1.0, 2.0, 3.0))
        self.assertEqual(struct.unpack("<3f", t1["bytes"]), (4.0, 5.0, 6.0))
        mgr.close()

    def test_quantized_model(self):
        """Tests dynamic INT8 and INT4 quantization, scales, and dequantization reconstruction."""
        original_floats = [float(i) * 0.1 - 5.0 for i in range(128)]
        raw_bytes = struct.pack(f"<{len(original_floats)}f", *original_floats)

        # 1. INT8 Quantization
        q8 = QuantizationEngine.quantize_int8(raw_bytes, shape=[128], name="test_int8")
        self.assertEqual(q8.precision, QuantizationPrecision.INT8)
        self.assertEqual(len(q8.quantized_bytes), 128)
        rec8 = QuantizationEngine.dequantize_int8(q8)
        self.assertEqual(len(rec8), 128)
        # Check max error is within 1% of dynamic range
        max_err8 = max(abs(a - b) for a, b in zip(original_floats, rec8))
        self.assertLess(max_err8, 0.15)

        # 2. INT4 Quantization (packed 4-bit)
        q4 = QuantizationEngine.quantize_int4(raw_bytes, shape=[128], name="test_int4", group_size=64)
        self.assertEqual(q4.precision, QuantizationPrecision.INT4)
        # 128 elements packed at 2 elements per byte = 64 bytes
        self.assertEqual(len(q4.quantized_bytes), 64)
        rec4 = QuantizationEngine.dequantize_int4(q4)
        self.assertEqual(len(rec4), 128)
        max_err4 = max(abs(a - b) for a, b in zip(original_floats, rec4))
        self.assertLess(max_err4, 0.8)

    def test_lazy_loading(self):
        """Tests on-demand shard paging and LRU cache capacity limits."""
        model_dir = os.path.join(self.temp_dir, "lazy_model")
        os.makedirs(model_dir)

        for i in range(3):
            s_name = f"model-0000{i+1}-of-00003.safetensors"
            create_mock_safetensors_file(os.path.join(model_dir, s_name), {f"w_{i}": [float(i)] * 10})

        index_data = {
            "weight_map": {f"w_{i}": f"model-0000{i+1}-of-00003.safetensors" for i in range(3)}
        }
        with open(os.path.join(model_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(index_data, f)

        # Cache limit: 1 shard
        mgr = ShardedSafeTensorsManager(model_dir, max_cached_shards=1)
        self.assertEqual(len(mgr._shard_cache), 0)

        # Load w_0 -> shard 1 opened
        mgr.load_tensor("w_0")
        self.assertEqual(len(mgr._shard_cache), 1)
        self.assertIn("model-00001-of-00003.safetensors", mgr._shard_cache)

        # Load w_1 -> shard 1 evicted, shard 2 opened
        mgr.load_tensor("w_1")
        self.assertEqual(len(mgr._shard_cache), 1)
        self.assertIn("model-00002-of-00003.safetensors", mgr._shard_cache)
        self.assertNotIn("model-00001-of-00003.safetensors", mgr._shard_cache)

        mgr.close()

    def test_fallback_behavior(self):
        """Tests fallback chain when higher-tier loading fails."""
        mock_prof = DeviceProfile.mock(
            total_ram_gb=16.0,
            available_ram_gb=10.0,
            gpu_vram_gb=6.0,
            gpu_vendor="NVIDIA",
            is_mobile=False
        )
        plan = LoadingPolicyEngine.create_loading_plan(
            profile=mock_prof,
            model_size_bytes=int(100 * MB),
            param_count=118080
        )
        self.assertEqual(plan.strategy, ModelLoadingStrategy.FULL_LOAD)
        self.assertTrue(len(plan.fallback_chain) >= 2)
        # Fallback 1: FULL_LOAD on CPU
        self.assertEqual(plan.fallback_chain[0].device_target, DeviceTarget.CPU)
        # Fallback 2: LAZY_LOAD on CPU
        self.assertEqual(plan.fallback_chain[1].strategy, ModelLoadingStrategy.LAZY_LOAD)

    def test_insufficient_memory_protection(self):
        """Tests that critically low RAM raises InsufficientMemoryError and prevents OOM crash."""
        critical_prof = DeviceProfile.mock(
            total_ram_gb=1.0,
            available_ram_gb=0.005,  # 5 MB free
            gpu_vram_gb=0.0,
            gpu_vendor="NONE",
            is_mobile=False
        )
        with self.assertRaises(InsufficientMemoryError):
            LoadingPolicyEngine.create_loading_plan(
                profile=critical_prof,
                model_size_bytes=int(100 * MB),
                param_count=118080
            )

    def test_auto_tara_model_loader_integration(self):
        """Tests end-to-end AutoTaraModelLoader with live repository model."""
        # Test loading canonical model using auto-loader
        loaded = AutoTaraModelLoader.auto_load(verbose=False)
        self.assertEqual(loaded.model_identity, "TARA")
        self.assertIn(loaded.strategy, ["FULL_LOAD", "LAZY_LOAD", "QUANTIZED_LOAD"])
        # Check accessing tensor seamlessly
        names = loaded.shard_manager.tensor_names
        self.assertTrue(len(names) > 0)
        t0 = loaded.get_tensor(names[0])
        self.assertIsNotNone(t0)
        loaded.close()


if __name__ == "__main__":
    unittest.main()
