"""
python/tara_core/memory.py

Episodic & Procedural Memory Engine for TARA Core (NVIDIA Voyager Architecture).
Maintains structured trajectory logs outside model weights.
Compatibility layer delegating directly to canonical TARA.MEMORY.MemoryEngine.
"""

import os
import sys

try:
    from TARA.MEMORY.memory_engine import MemoryEngine
except Exception:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
    from TARA.MEMORY.memory_engine import MemoryEngine

__all__ = ["MemoryEngine"]

