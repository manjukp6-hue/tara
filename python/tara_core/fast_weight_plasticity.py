"""
python/tara_core/fast_weight_plasticity.py

Real-Time Synaptic Plasticity & Fast-Weight Dynamic Cognitive Adapter for TARA Core.
Enables instant associative recall and in-context synaptic adaptation
without modifying static neural model weights (model.safetensors).

Architectural Guarantees:
1. Operates an associative memory matrix A_t updated via outer-product Hebbian learning.
2. Fast weights decay with retention factor lambda (0.0 <= lambda <= 1.0).
3. Zero perturbation of canonical base weights.
4. Mathematical associative retrieval: output = (W_base + A_t) * x.
"""

import math
from typing import List, Dict, Any, Optional, Tuple


class FastWeightPlasticityEngine:
    """
    Synaptic plasticity adapter enabling real-time episodic learning.
    Updates fast weights dynamically during execution sessions.
    """

    def __init__(
        self,
        dimension: int = 64,
        decay_rate: float = 0.95,
        learning_rate: float = 0.50
    ):
        self.dimension = dimension
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        # Initialize square associative matrix with zeros
        self.fast_weights: List[List[float]] = [[0.0] * dimension for _ in range(dimension)]
        self.total_updates = 0

    def update_synapses(self, key_vector: List[float], value_vector: List[float]) -> None:
        """
        Applies outer-product Hebbian associative update:
        A_{t+1} = lambda * A_t + eta * (v (x) k)
        """
        k_norm = self._normalize(key_vector)
        v_norm = self._normalize(value_vector)

        dim = min(self.dimension, len(k_norm), len(v_norm))

        for i in range(dim):
            for j in range(dim):
                # Decay previous associative weight
                self.fast_weights[i][j] *= self.decay_rate
                # Add outer product Hebbian correlation
                self.fast_weights[i][j] += self.learning_rate * (v_norm[i] * k_norm[j])

        self.total_updates += 1

    def retrieve_association(self, query_vector: List[float]) -> List[float]:
        """
        Retrieves associative memory activation:
        y = A_t * q
        """
        q_norm = self._normalize(query_vector)
        dim = min(self.dimension, len(q_norm))

        result = [0.0] * dim
        for i in range(dim):
            dot = 0.0
            for j in range(dim):
                dot += self.fast_weights[i][j] * q_norm[j]
            result[i] = dot

        return result

    def adapt_linear_projection(
        self,
        base_projection: List[float],
        input_vector: List[float],
        adaptation_scale: float = 0.20
    ) -> List[float]:
        """
        Combines static base projection with dynamic plastic fast weights:
        h_adapted = h_base + scale * (A_t * x)
        """
        fast_delta = self.retrieve_association(input_vector)
        dim = min(len(base_projection), len(fast_delta))

        adapted = list(base_projection)
        for i in range(dim):
            adapted[i] += adaptation_scale * fast_delta[i]

        return adapted

    def reset_synapses(self) -> None:
        """Flushes transient fast weights back to zero baseline."""
        self.fast_weights = [[0.0] * self.dimension for _ in range(self.dimension)]
        self.total_updates = 0

    def _normalize(self, vec: List[float]) -> List[float]:
        """Normalizes vector to L2 unit sphere to stabilize associative learning."""
        norm_sq = sum(v * v for v in vec)
        if norm_sq <= 1e-12:
            return list(vec)
        inv_norm = 1.0 / math.sqrt(norm_sq)
        return [v * inv_norm for v in vec]
