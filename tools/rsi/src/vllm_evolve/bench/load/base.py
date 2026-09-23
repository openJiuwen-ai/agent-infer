"""Shared sampling helpers for synthetic load generation."""
from __future__ import annotations

import math
import random


def sample_lognormal(
    rng: random.Random, mean: float, std: float, lo: int = 1, hi: int = 32768
) -> int:
    """Sample an integer length from a lognormal distribution, clamped to [lo, hi].

    Parameterized by the desired arithmetic ``mean`` and ``std`` of the
    distribution (not the underlying normal's mu/sigma).
    """
    if mean <= 0:
        return lo
    if std <= 0:
        return max(lo, min(int(mean), hi))
    sigma2 = math.log(1.0 + (std * std) / (mean * mean))
    mu = math.log(mean) - 0.5 * sigma2
    val = int(rng.lognormvariate(mu, math.sqrt(sigma2)))
    return max(lo, min(val, hi))
