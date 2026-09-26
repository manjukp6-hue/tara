"""
TARA/MODEL/quantization.py
================================================================================
TARA Dynamic Quantization Infrastructure: INT8, INT4, FP16, BF16
================================================================================

Quantized representations are capacity and runtime variants of the permanent
TARA model, NEVER separate model identities.

Features:
- Block-wise and tensor-wise INT8 and INT4 quantization.
- 4-bit packed representation (2 values per byte).
- Automatic on-demand dequantization and runtime execution.
- FP16 and BF16 precision conversions.
"""

import struct
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Union

from .loading_policy import QuantizationPrecision


@dataclass
class QuantizedTensor:
    name: str
    precision: QuantizationPrecision
    shape: List[int]
    quantized_bytes: bytes
    scales: List[float]
    zero_points: Optional[List[int]] = None
    group_size: int = 128
    original_dtype: str = "F32"

    @property
    def memory_bytes(self) -> int:
        scale_bytes = len(self.scales) * 4
        zp_bytes = len(self.zero_points) if self.zero_points else 0
        return len(self.quantized_bytes) + scale_bytes + zp_bytes


class QuantizationEngine:
    """
    Quantizes and dequantizes weights dynamically for memory-constrained devices.
    Supports INT8, INT4 (packed), FP16, and BF16 representations.
    """

    @staticmethod
    def unpack_floats(data_bytes: bytes, dtype: str = "F32") -> List[float]:
        """Unpacks raw bytes to Python float list."""
        if dtype in ["F32", "FLOAT32"]:
            count = len(data_bytes) // 4
            return list(struct.unpack(f"<{count}f", data_bytes[:count * 4]))
        elif dtype in ["F16", "FLOAT16"]:
            count = len(data_bytes) // 2
            # Half-precision unpack
            return list(struct.unpack(f"<{count}e", data_bytes[:count * 2]))
        else:
            count = len(data_bytes) // 4
            return list(struct.unpack(f"<{count}f", data_bytes[:count * 4]))

    @classmethod
    def quantize_int8(cls, raw_bytes: bytes, shape: List[int], name: str = "", dtype: str = "F32") -> QuantizedTensor:
        """
        Symmetric INT8 Quantization.
        Maps [-max_abs, max_abs] to [-127, 127].
        """
        floats = cls.unpack_floats(raw_bytes, dtype=dtype)
        if not floats:
            return QuantizedTensor(name=name, precision=QuantizationPrecision.INT8, shape=shape, quantized_bytes=b"", scales=[1.0])

        max_abs = max(abs(x) for x in floats) or 1e-6
        scale = max_abs / 127.0

        int8_vals = []
        for val in floats:
            q = int(round(val / scale))
            q = max(-127, min(127, q))
            int8_vals.append(q)

        q_bytes = struct.pack(f"<{len(int8_vals)}b", *int8_vals)
        return QuantizedTensor(
            name=name,
            precision=QuantizationPrecision.INT8,
            shape=shape,
            quantized_bytes=q_bytes,
            scales=[scale],
            zero_points=None,
            original_dtype=dtype
        )

    @classmethod
    def dequantize_int8(cls, q_tensor: QuantizedTensor) -> List[float]:
        """Reconstructs float values from INT8 representation."""
        count = len(q_tensor.quantized_bytes)
        int8_vals = struct.unpack(f"<{count}b", q_tensor.quantized_bytes)
        scale = q_tensor.scales[0]
        return [float(q * scale) for q in int8_vals]

    @classmethod
    def quantize_int4(cls, raw_bytes: bytes, shape: List[int], name: str = "", dtype: str = "F32", group_size: int = 128) -> QuantizedTensor:
        """
        Group-wise INT4 Quantization with 2 values packed per byte.
        Maps [-max_group, max_group] to [-7, 7].
        """
        floats = cls.unpack_floats(raw_bytes, dtype=dtype)
        if not floats:
            return QuantizedTensor(name=name, precision=QuantizationPrecision.INT4, shape=shape, quantized_bytes=b"", scales=[1.0])

        scales: List[float] = []
        packed_bytes = bytearray()
        q_nibbles: List[int] = []

        # Process in groups of group_size
        num_groups = math.ceil(len(floats) / group_size)
        for g_idx in range(num_groups):
            start = g_idx * group_size
            end = min(start + group_size, len(floats))
            group = floats[start:end]

            max_abs = max(abs(x) for x in group) or 1e-6
            scale = max_abs / 7.0
            scales.append(scale)

            for val in group:
                q = int(round(val / scale))
                q = max(-7, min(7, q))
                # Store as 4-bit unsigned nibble [0, 15] with offset +7
                nibble = (q + 7) & 0x0F
                q_nibbles.append(nibble)

        # Pack 2 nibbles per byte: (high_nibble << 4) | low_nibble
        for i in range(0, len(q_nibbles), 2):
            low = q_nibbles[i]
            high = q_nibbles[i + 1] if (i + 1) < len(q_nibbles) else 0
            byte_val = ((high & 0x0F) << 4) | (low & 0x0F)
            packed_bytes.append(byte_val)

        return QuantizedTensor(
            name=name,
            precision=QuantizationPrecision.INT4,
            shape=shape,
            quantized_bytes=bytes(packed_bytes),
            scales=scales,
            zero_points=[7],  # offset of 7
            group_size=group_size,
            original_dtype=dtype
        )

    @classmethod
    def dequantize_int4(cls, q_tensor: QuantizedTensor) -> List[float]:
        """Unpacks 4-bit packed nibbles and applies group-wise scales."""
        reconstructed: List[float] = []
        total_elements = math.prod(q_tensor.shape) if q_tensor.shape else len(q_tensor.quantized_bytes) * 2

        q_nibbles: List[int] = []
        for byte_val in q_tensor.quantized_bytes:
            low = byte_val & 0x0F
            high = (byte_val >> 4) & 0x0F
            q_nibbles.append(low)
            q_nibbles.append(high)

        q_nibbles = q_nibbles[:total_elements]
        group_size = q_tensor.group_size

        for idx, nibble in enumerate(q_nibbles):
            g_idx = idx // group_size
            scale = q_tensor.scales[min(g_idx, len(q_tensor.scales) - 1)]
            q_val = nibble - 7  # undo +7 offset
            reconstructed.append(float(q_val * scale))

        return reconstructed

    @classmethod
    def quantize_to_precision(
        cls,
        raw_bytes: bytes,
        shape: List[int],
        name: str,
        target_precision: QuantizationPrecision,
        source_dtype: str = "F32"
    ) -> Union[QuantizedTensor, bytes]:
        """Dispatches quantization based on target precision."""
        if target_precision == QuantizationPrecision.INT4:
            return cls.quantize_int4(raw_bytes, shape, name=name, dtype=source_dtype)
        elif target_precision == QuantizationPrecision.INT8:
            return cls.quantize_int8(raw_bytes, shape, name=name, dtype=source_dtype)
        elif target_precision in [QuantizationPrecision.FP16, QuantizationPrecision.BF16]:
            # Convert float32 to float16 bytes
            floats = cls.unpack_floats(raw_bytes, dtype=source_dtype)
            return struct.pack(f"<{len(floats)}e", *floats)
        else:
            return raw_bytes
