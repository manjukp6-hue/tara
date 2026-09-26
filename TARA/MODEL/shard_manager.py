"""
TARA/MODEL/shard_manager.py
================================================================================
Unified 100 MB SafeTensors Shard Manager & Lazy Paging Runtime
================================================================================

Features:
- Seamless single-model abstraction over 100 MB SafeTensors shards.
- Automatic discovery and comprehensive integrity validation.
- On-demand lazy shard paging with LRU cache eviction.
- Full model loading when resources are sufficient.
- Never treats shards as separate models.
"""

import os
import glob
import json
import mmap
import collections
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Set

from .device_detector import MB


@dataclass
class TensorMetadata:
    name: str
    dtype: str
    shape: List[int]
    data_offsets: Tuple[int, int]
    shard_filename: str
    data_bytes: int


class ShardedSafeTensorsManager:
    """
    Unified manager for single-file and ~100MB sharded SafeTensors models.
    Presents a single logical model interface to the rest of TARA.
    """
    def __init__(self, model_dir: str, max_cached_shards: int = 4, enable_mmap: bool = True):
        self.model_dir = os.path.abspath(model_dir)
        self.max_cached_shards = max(1, max_cached_shards)
        self.enable_mmap = enable_mmap

        self.index_path: Optional[str] = None
        self.shard_paths: Dict[str, str] = {}  # shard_filename -> full_path
        self.tensor_index: Dict[str, TensorMetadata] = {}  # tensor_name -> TensorMetadata
        self.weight_map: Dict[str, str] = {}  # tensor_name -> shard_filename
        self.total_weights_bytes: int = 0

        # LRU open shard cache: shard_filename -> (file_obj, mmap_obj or bytes, data_start)
        self._shard_cache: collections.OrderedDict = collections.OrderedDict()

        self._discover_and_validate()

    def _discover_and_validate(self):
        """Discovers single-file or multi-shard artifacts and validates integrity."""
        index_file = os.path.join(self.model_dir, "model.safetensors.index.json")
        single_file = os.path.join(self.model_dir, "model.safetensors")

        if os.path.exists(index_file):
            self.index_path = index_file
            try:
                with open(index_file, "r", encoding="utf-8") as f:
                    idx_data = json.load(f)
                self.weight_map = idx_data.get("weight_map", {})
            except Exception as e:
                raise ValueError(f"Corrupted model.safetensors.index.json in {self.model_dir}: {e}")

            # Verify all shard files exist
            unique_shards = sorted(list(set(self.weight_map.values())))
            for s_name in unique_shards:
                s_full = os.path.join(self.model_dir, s_name)
                if not os.path.exists(s_full):
                    raise FileNotFoundError(f"Missing required SafeTensors shard: {s_full}")
                self.shard_paths[s_name] = s_full

        elif os.path.exists(single_file):
            s_name = "model.safetensors"
            self.shard_paths[s_name] = single_file
        else:
            # Fallback scan for any *.safetensors
            found = sorted(glob.glob(os.path.join(self.model_dir, "*.safetensors")))
            if not found:
                raise FileNotFoundError(f"No SafeTensors weights found in {self.model_dir}")
            for p in found:
                s_name = os.path.basename(p)
                self.shard_paths[s_name] = p

        # Parse SafeTensors headers across all discovered shards
        for s_name, s_full in self.shard_paths.items():
            self._index_shard_header(s_name, s_full)

    def _index_shard_header(self, shard_name: str, shard_path: str):
        """Reads SafeTensors header and records tensor offsets."""
        file_size = os.path.getsize(shard_path)
        with open(shard_path, "rb") as f:
            header_len_bytes = f.read(8)
            if len(header_len_bytes) < 8:
                raise ValueError(f"File too small to be a SafeTensors file: {shard_path}")
            header_len = int.from_bytes(header_len_bytes, "little")
            if header_len < 2 or header_len > file_size - 8:
                raise ValueError(f"Corrupted SafeTensors header length ({header_len}) in {shard_path}")

            header_json_bytes = f.read(header_len)
            header_data = json.loads(header_json_bytes.decode("utf-8"))

        data_start = 8 + header_len
        for t_name, info in header_data.items():
            if t_name == "__metadata__":
                continue
            offsets = info["data_offsets"]
            t_bytes = offsets[1] - offsets[0]
            self.total_weights_bytes += t_bytes

            meta = TensorMetadata(
                name=t_name,
                dtype=info.get("dtype", "F32"),
                shape=info.get("shape", []),
                data_offsets=(offsets[0], offsets[1]),
                shard_filename=shard_name,
                data_bytes=t_bytes
            )
            self.tensor_index[t_name] = meta
            self.weight_map[t_name] = shard_name

    def _get_open_shard(self, shard_name: str) -> Tuple[Any, Any, int]:
        """Returns cached (file_obj, mmap_or_bytes, data_start) with LRU management."""
        if shard_name in self._shard_cache:
            # Move to end for LRU
            self._shard_cache.move_to_end(shard_name)
            return self._shard_cache[shard_name]

        # Evict oldest if exceeding capacity
        while len(self._shard_cache) >= self.max_cached_shards:
            evicted_name, (f_obj, mm_obj, _) = self._shard_cache.popitem(last=False)
            try:
                if hasattr(mm_obj, "close"):
                    mm_obj.close()
                if hasattr(f_obj, "close"):
                    f_obj.close()
            except Exception:
                pass

        # Open new shard
        shard_path = self.shard_paths[shard_name]
        f_obj = open(shard_path, "rb")
        file_size = os.path.getsize(shard_path)

        if self.enable_mmap:
            mm_obj = mmap.mmap(f_obj.fileno(), 0, access=mmap.ACCESS_READ)
            header_len = int.from_bytes(mm_obj[:8], "little")
            data_start = 8 + header_len
        else:
            # Read header to find data_start
            header_len = int.from_bytes(f_obj.read(8), "little")
            data_start = 8 + header_len
            mm_obj = None

        cached_entry = (f_obj, mm_obj, data_start)
        self._shard_cache[shard_name] = cached_entry
        return cached_entry

    def get_tensor_bytes(self, tensor_name: str) -> bytes:
        """Retrieves raw bytes of a tensor on demand from its host shard."""
        if tensor_name not in self.tensor_index:
            raise KeyError(f"Tensor '{tensor_name}' not found in canonical TARA model")

        meta = self.tensor_index[tensor_name]
        f_obj, mm_obj, data_start = self._get_open_shard(meta.shard_filename)
        start_off = data_start + meta.data_offsets[0]
        end_off = data_start + meta.data_offsets[1]

        if mm_obj is not None:
            return bytes(mm_obj[start_off:end_off])
        else:
            f_obj.seek(start_off)
            return f_obj.read(end_off - start_off)

    def load_tensor(self, tensor_name: str) -> Dict[str, Any]:
        """Loads a tensor with metadata."""
        meta = self.tensor_index[tensor_name]
        raw_bytes = self.get_tensor_bytes(tensor_name)
        return {
            "name": meta.name,
            "dtype": meta.dtype,
            "shape": meta.shape,
            "bytes": raw_bytes,
            "data_bytes": meta.data_bytes,
            "shard": meta.shard_filename
        }

    def load_all_tensors(self) -> Dict[str, Dict[str, Any]]:
        """Loads all tensors across all shards into memory (for FULL_LOAD strategy)."""
        loaded = {}
        for t_name in self.tensor_index.keys():
            loaded[t_name] = self.load_tensor(t_name)
        return loaded

    @property
    def total_shards(self) -> int:
        return len(self.shard_paths)

    @property
    def tensor_names(self) -> List[str]:
        return sorted(list(self.tensor_index.keys()))

    def close(self):
        """Closes all cached file handles and mmaps."""
        for f_obj, mm_obj, _ in self._shard_cache.values():
            try:
                if hasattr(mm_obj, "close"):
                    mm_obj.close()
                if hasattr(f_obj, "close"):
                    f_obj.close()
            except Exception:
                pass
        self._shard_cache.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
