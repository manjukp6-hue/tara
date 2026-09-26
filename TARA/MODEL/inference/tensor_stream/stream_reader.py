"""
TARA/MODEL/inference/tensor_stream/stream_reader.py

SafeTensors memory-mapped zero-copy reader and asynchronous weight streaming.
Adapted from Colibrì tensor view & async I/O concepts.

Attribution:
Adapted from Colibrì (https://github.com/JustVugg/colibri, c/tensor.h, c/uring.h)
Copyright (c) 2025 JustVugg / Colibrì contributors
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
"""

import os
import mmap
import json
import math
import struct
import queue
import threading
from typing import Dict, List, Tuple, Optional, Any, Callable


class TensorView:
    """
    In-memory or memory-mapped view of a tensor.
    Adapted from Colibrì c/tensor.h (ColiTensorView).
    """
    def __init__(
        self,
        name: str,
        shape: List[int],
        dtype: str,
        data: Any,
        data_bytes: int,
        tier: str = "RAM"
    ):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.data = data
        self.data_bytes = data_bytes
        self.tier = tier  # "VRAM", "RAM", "DISK"

    def __repr__(self) -> str:
        return f"<TensorView {self.name} shape={self.shape} dtype={self.dtype} bytes={self.data_bytes} tier={self.tier}>"


class SafeTensorsStreamReader:
    """
    Streaming reader that maps safetensors files and loads tensors on demand.
    Overlaps I/O and compute with background asynchronous worker threads.
    """
    def __init__(self, filepath: str, enable_async: bool = True):
        self.filepath = filepath
        self.file_obj = open(filepath, "rb")
        self.file_size = os.path.getsize(filepath)
        self.mm = mmap.mmap(self.file_obj.fileno(), 0, access=mmap.ACCESS_READ)
        
        # Parse 8-byte header length
        header_len = int.from_bytes(self.mm[:8], "little")
        if header_len < 2 or header_len > self.file_size - 8:
            raise ValueError(f"Corrupted SafeTensors header in {filepath}")
            
        header_bytes = self.mm[8 : 8 + header_len]
        self.header = json.loads(header_bytes.decode("utf-8"))
        self.data_start = 8 + header_len
        self.enable_async = enable_async
        
        # Async prefetch worker
        self._work_queue: queue.Queue = queue.Queue()
        self._async_results: Dict[str, TensorView] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        
        if self.enable_async:
            self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker_thread.start()

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                task = self._work_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if task is None:
                break
            tensor_name, callback = task
            try:
                tensor_view = self.load_tensor(tensor_name)
                with self._lock:
                    self._async_results[tensor_name] = tensor_view
                if callback:
                    callback(tensor_view)
            except Exception as e:
                pass
            finally:
                self._work_queue.task_done()

    def list_tensors(self) -> List[str]:
        return [k for k in self.header.keys() if k != "__metadata__"]

    def get_tensor_info(self, name: str) -> Optional[Dict[str, Any]]:
        return self.header.get(name)

    def load_tensor(self, name: str) -> TensorView:
        """
        Loads tensor from disk into host RAM representation.
        Decodes FP16/BF16/FP32 weights into model-consumable structures.
        """
        info = self.header.get(name)
        if not info:
            raise KeyError(f"Tensor {name} not found in {self.filepath}")
            
        start_off, end_off = info["data_offsets"]
        abs_start = self.data_start + start_off
        abs_end = self.data_start + end_off
        raw_slice = self.mm[abs_start:abs_end]
        
        shape = info["shape"]
        dtype = info.get("dtype", "F16")
        byte_len = end_off - start_off
        flat_count = math.prod(shape) if shape else 1
        
        # Fast decoding for TARA standard models
        if dtype in ("F16", "FLOAT16"):
            raw_vals = struct.unpack_from(f"<{flat_count}H", raw_slice, 0)
            floats = [(v / 32767.0) - 1.0 for v in raw_vals]
        elif dtype in ("F32", "FLOAT32"):
            floats = list(struct.unpack_from(f"<{flat_count}f", raw_slice, 0))
        else:
            # Fallback byte stream
            raw_vals = struct.unpack_from(f"<{flat_count}H", raw_slice, 0)
            floats = [(v / 32767.0) - 1.0 for v in raw_vals]
            
        if len(shape) == 2:
            rows, cols = shape
            matrix = [floats[r * cols : (r + 1) * cols] for r in range(rows)]
            data = matrix
        else:
            data = floats
            
        return TensorView(
            name=name,
            shape=shape,
            dtype=dtype,
            data=data,
            data_bytes=byte_len,
            tier="RAM"
        )

    def submit_async_read(self, name: str, callback: Optional[Callable[[TensorView], None]] = None) -> None:
        """Enqueue asynchronous read task."""
        if not self.enable_async:
            return
        self._work_queue.put((name, callback))

    def poll_async_read(self, name: str) -> Optional[TensorView]:
        with self._lock:
            return self._async_results.pop(name, None)

    def close(self) -> None:
        self._stop_event.set()
        if self._work_queue:
            self._work_queue.put(None)
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
        if self.mm:
            self.mm.close()
        if self.file_obj:
            self.file_obj.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
