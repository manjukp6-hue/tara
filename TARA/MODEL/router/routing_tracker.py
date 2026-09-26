"""
TARA/MODEL/router/routing_tracker.py

Routing trace tracker and access heat management.
Adapted from Colibrì route tracing and tier decay.

Attribution:
Adapted from Colibrì (https://github.com/JustVugg/colibri, c/route_trace.h, c/tier.h)
Copyright (c) 2025 JustVugg / Colibrì contributors
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
"""

from typing import Dict, List, Tuple, Optional
import time


class RoutingTracker:
    """
    Maintains activation heat maps and transition histories for layers and experts.
    Used by the prefetcher to anticipate next tensor needs.
    """
    def __init__(self, decay_interval_steps: int = 100):
        self.heat_map: Dict[str, int] = {}
        self.access_history: List[str] = []
        self.transitions: Dict[str, Dict[str, int]] = {}
        self.step_count: int = 0
        self.decay_interval_steps = decay_interval_steps

    def record_access(self, tensor_key: str) -> None:
        self.step_count += 1
        self.heat_map[tensor_key] = self.heat_map.get(tensor_key, 0) + 1
        
        # Track transitions
        if self.access_history:
            prev_key = self.access_history[-1]
            if prev_key not in self.transitions:
                self.transitions[prev_key] = {}
            self.transitions[prev_key][tensor_key] = self.transitions[prev_key].get(tensor_key, 0) + 1
            
        self.access_history.append(tensor_key)
        if len(self.access_history) > 1000:
            self.access_history = self.access_history[-500:]

        # Periodic decay
        if self.step_count % self.decay_interval_steps == 0:
            self.decay_all()

    def decay_all(self) -> None:
        """Decay heat counters using Colibrì half-life (heat >> 1)."""
        for k in list(self.heat_map.keys()):
            new_val = self.heat_map[k] >> 1
            if new_val == 0:
                del self.heat_map[k]
            else:
                self.heat_map[k] = new_val

    def predict_next(self, current_key: str, top_k: int = 2) -> List[str]:
        """Predict likely next tensor accesses based on transition frequencies."""
        next_candidates = self.transitions.get(current_key, {})
        if not next_candidates:
            return []
        sorted_candidates = sorted(next_candidates.items(), key=lambda x: x[1], reverse=True)
        return [k for k, _ in sorted_candidates[:top_k]]

    def get_heat(self, tensor_key: str) -> int:
        return self.heat_map.get(tensor_key, 0)
