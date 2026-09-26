"""
TARA/MODEL/inference/cache/cache_policy.py

Cache eviction and admission policies adapted from Colibrì.
Implements LRU, LFU, and Colibrì's LFRU (Least Frequently and Recently Used)
with 25% + 4-unit hysteresis margin to prevent ping-pong thrashing.

Attribution:
Adapted from Colibrì (https://github.com/JustVugg/colibri, c/tier.h)
Copyright (c) 2025 JustVugg / Colibrì contributors
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
"""

from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
import time


def tier_should_promote(hot: int, cold: int, margin_pct: int = 25, fixed_margin: int = 4) -> bool:
    """
    Shared admission contract for adaptive resident tiers.
    Adapted from Colibrì c/tier.h (tier_should_promote):
    threshold = cold + ((cold * margin_pct) // 100) + fixed_margin
    Widen before adding: prevents cold flash displacing a genuinely hot tensor.
    """
    threshold = cold + (cold >> 2) + fixed_margin  # 25% + 4
    return hot > threshold


def tier_decay_value(heat: int) -> int:
    """
    Periodic half-life decay of heat counter.
    Adapted from Colibrì c/tier.h (tier_decay_value).
    """
    return heat >> 1


def tier_lfru_score(heat: int, last: int, clock: int) -> int:
    """
    LFRU scoring adapted from Colibrì c/tier.h (tier_lfru_score):
    Frequency is the primary signal (shifted 8 bits);
    recency (0-255) breaks close calls.
    A recent access contributes at most 255 points while one frequency count
    is worth 256, so a merely recent item cannot displace a genuinely hotter one.
    """
    age = clock - last if clock >= last else 0
    recent = 255 - age if age < 255 else 0
    return (heat << 8) | recent


class CachePolicy(ABC):
    @abstractmethod
    def record_access(self, key: str) -> None:
        """Record an access to the given key."""
        pass

    @abstractmethod
    def pick_eviction(self, resident_keys: List[str]) -> Optional[str]:
        """Pick a resident key to evict."""
        pass

    @abstractmethod
    def should_admit(self, candidate_key: str, evict_key: str) -> bool:
        """Decide whether candidate should displace evict_key."""
        pass

    @abstractmethod
    def decay(self) -> None:
        """Decay historical usage counters."""
        pass


class LFRUCachePolicy(CachePolicy):
    """
    Hybrid Least-Frequently & Recently Used policy from Colibrì.
    Combines access frequency (heat) with recency clock and 25% hysteresis.
    """
    def __init__(self, margin_pct: int = 25, fixed_margin: int = 4):
        self.heat: Dict[str, int] = {}
        self.last_access: Dict[str, int] = {}
        self.clock: int = 0
        self.margin_pct = margin_pct
        self.fixed_margin = fixed_margin

    def record_access(self, key: str) -> None:
        self.clock += 1
        self.heat[key] = self.heat.get(key, 0) + 1
        self.last_access[key] = self.clock

    def score(self, key: str) -> int:
        h = self.heat.get(key, 0)
        last = self.last_access.get(key, self.clock)
        return tier_lfru_score(h, last, self.clock)

    def pick_eviction(self, resident_keys: List[str]) -> Optional[str]:
        if not resident_keys:
            return None
        coldest_key = None
        min_score = float('inf')
        for key in resident_keys:
            sc = self.score(key)
            if sc < min_score:
                min_score = sc
                coldest_key = key
        return coldest_key

    def should_admit(self, candidate_key: str, evict_key: str) -> bool:
        c_score = self.score(candidate_key)
        e_score = self.score(evict_key)
        # Hysteresis in score units: hs <= cs + (cs >> 2) + (fixed << 8)
        threshold = e_score + (e_score >> 2) + (self.fixed_margin << 8)
        return c_score > threshold

    def decay(self) -> None:
        for k in self.heat:
            self.heat[k] = tier_decay_value(self.heat[k])


class LRUCachePolicy(CachePolicy):
    """Standard Least-Recently Used cache policy."""
    def __init__(self):
        self.last_access: Dict[str, float] = {}

    def record_access(self, key: str) -> None:
        self.last_access[key] = time.time()

    def pick_eviction(self, resident_keys: List[str]) -> Optional[str]:
        if not resident_keys:
            return None
        return min(resident_keys, key=lambda k: self.last_access.get(k, 0.0))

    def should_admit(self, candidate_key: str, evict_key: str) -> bool:
        return True

    def decay(self) -> None:
        pass


class LFUCachePolicy(CachePolicy):
    """Standard Least-Frequently Used cache policy with frequency counts."""
    def __init__(self):
        self.frequency: Dict[str, int] = {}

    def record_access(self, key: str) -> None:
        self.frequency[key] = self.frequency.get(key, 0) + 1

    def pick_eviction(self, resident_keys: List[str]) -> Optional[str]:
        if not resident_keys:
            return None
        return min(resident_keys, key=lambda k: self.frequency.get(k, 0))

    def should_admit(self, candidate_key: str, evict_key: str) -> bool:
        return self.frequency.get(candidate_key, 0) >= self.frequency.get(evict_key, 0)

    def decay(self) -> None:
        for k in self.frequency:
            self.frequency[k] = self.frequency[k] >> 1


def create_cache_policy(name: str = "lfru") -> CachePolicy:
    name_lower = name.lower().strip()
    if name_lower == "lfru":
        return LFRUCachePolicy()
    elif name_lower == "lru":
        return LRUCachePolicy()
    elif name_lower == "lfu":
        return LFUCachePolicy()
    else:
        return LFRUCachePolicy()
