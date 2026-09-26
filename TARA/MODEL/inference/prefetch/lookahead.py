"""
TARA/MODEL/inference/prefetch/lookahead.py

Lookahead and speculative tensor prefetching.
Adapted from Colibrì lookahead prefetch pipeline.

Attribution:
Adapted from Colibrì (https://github.com/JustVugg/colibri, c/expert_store.h, c/qwen36_tier.c)
Copyright (c) 2025 JustVugg / Colibrì contributors
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
"""

from typing import List, Dict, Optional, Any, Callable
import threading


class LookaheadPrefetcher:
    """
    Prefetches predictable future layers or routing-directed experts ahead of compute.
    """
    def __init__(
        self,
        prefetch_fn: Callable[[List[str]], int],
        depth: int = 1,
        enabled: bool = True
    ):
        self.prefetch_fn = prefetch_fn
        self.depth = depth
        self.enabled = enabled
        self._lock = threading.Lock()
        self.inflight_prefetches: set = set()

    def on_layer_begin(self, current_layer: int, total_layers: int, tensor_names_by_layer: Dict[int, List[str]]) -> None:
        """
        Triggered when a layer starts computation: issues background prefetch
        for layers (current_layer + 1 .. current_layer + depth).
        """
        if not self.enabled:
            return

        keys_to_fetch = []
        for step in range(1, self.depth + 1):
            target_layer = current_layer + step
            if target_layer < total_layers:
                layer_keys = tensor_names_by_layer.get(target_layer, [])
                for k in layer_keys:
                    if k not in self.inflight_prefetches:
                        keys_to_fetch.append(k)

        if keys_to_fetch:
            with self._lock:
                self.inflight_prefetches.update(keys_to_fetch)
            # Invoke store prefetch advisory
            self.prefetch_fn(keys_to_fetch)

    def on_layer_complete(self, completed_layer: int, tensor_names_by_layer: Dict[int, List[str]]) -> None:
        """Release inflight markers once computation completes."""
        layer_keys = tensor_names_by_layer.get(completed_layer, [])
        with self._lock:
            for k in layer_keys:
                self.inflight_prefetches.discard(k)

    def prefetch_candidates(self, candidate_keys: List[str]) -> int:
        if not self.enabled or not candidate_keys:
            return 0
        return self.prefetch_fn(candidate_keys)
