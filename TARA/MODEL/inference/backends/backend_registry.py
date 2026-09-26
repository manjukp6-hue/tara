"""
TARA/MODEL/inference/backends/backend_registry.py

Pluggable backend dispatch for CPU and GPU execution.
Adapted from Colibrì backend abstraction.

Attribution:
Adapted from Colibrì (https://github.com/JustVugg/colibri, c/backend_loader.c, c/backend_cuda.h)
Copyright (c) 2025 JustVugg / Colibrì contributors
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
import sys


class DeviceBackend(ABC):
    @abstractmethod
    def name(self) -> str:
        """Name of backend."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend hardware/runtime is functional."""
        pass

    @abstractmethod
    def allocate_tensor(self, shape: List[int], dtype: str) -> Any:
        """Allocate device memory buffer."""
        pass

    @abstractmethod
    def transfer_to_device(self, host_tensor: Any) -> Any:
        """Transfer tensor from host to this device."""
        pass

    @abstractmethod
    def transfer_to_host(self, device_tensor: Any) -> Any:
        """Transfer tensor from device back to host."""
        pass


class CPUBackend(DeviceBackend):
    """Native CPU backend execution."""
    def name(self) -> str:
        return "CPU"

    def is_available(self) -> bool:
        return True

    def allocate_tensor(self, shape: List[int], dtype: str) -> Any:
        # Standard nested list or flat array
        if len(shape) == 2:
            return [[0.0] * shape[1] for _ in range(shape[0])]
        return [0.0] * shape[0]

    def transfer_to_device(self, host_tensor: Any) -> Any:
        return host_tensor

    def transfer_to_host(self, device_tensor: Any) -> Any:
        return device_tensor


class CUDABackend(DeviceBackend):
    """CUDA GPU backend execution."""
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self._available = False
        try:
            import torch
            self._available = torch.cuda.is_available() and torch.cuda.device_count() > device_id
        except Exception:
            self._available = False

    def name(self) -> str:
        return f"CUDA:{self.device_id}"

    def is_available(self) -> bool:
        return self._available

    def allocate_tensor(self, shape: List[int], dtype: str) -> Any:
        if not self._available:
            raise RuntimeError("CUDA backend is not available")
        import torch
        dt = torch.float32 if dtype in ("F32", "FLOAT32") else torch.float16
        return torch.zeros(shape, dtype=dt, device=f"cuda:{self.device_id}")

    def transfer_to_device(self, host_tensor: Any) -> Any:
        if not self._available:
            return host_tensor
        import torch
        if isinstance(host_tensor, torch.Tensor):
            return host_tensor.to(f"cuda:{self.device_id}")
        t = torch.tensor(host_tensor, dtype=torch.float32, device=f"cuda:{self.device_id}")
        return t

    def transfer_to_host(self, device_tensor: Any) -> Any:
        if not self._available:
            return device_tensor
        import torch
        if isinstance(device_tensor, torch.Tensor):
            return device_tensor.cpu().tolist()
        return device_tensor


class BackendRegistry:
    """Registry and selector for runtime backends."""
    _backends: Dict[str, DeviceBackend] = {}

    @classmethod
    def register(cls, name: str, backend: DeviceBackend) -> None:
        cls._backends[name.upper()] = backend

    @classmethod
    def get(cls, name: str) -> DeviceBackend:
        b = cls._backends.get(name.upper())
        if b and b.is_available():
            return b
        # Fallback to CPU
        return cls._backends.get("CPU", CPUBackend())

    @classmethod
    def select_best_backend(cls) -> DeviceBackend:
        cuda = cls._backends.get("CUDA:0")
        if cuda and cuda.is_available():
            return cuda
        return cls.get("CPU")


# Register built-in default backends
BackendRegistry.register("CPU", CPUBackend())
BackendRegistry.register("CUDA:0", CUDABackend(0))
