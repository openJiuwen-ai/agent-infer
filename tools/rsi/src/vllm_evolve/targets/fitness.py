"""
WeightedRatioFitness — YAML-driven fitness computation.

Replaces the hardcoded fitness.py formulas. All metric weights, directions,
and normalization come from the target's YAML config.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm_evolve.config import MetricConfig

_EPS = 1e-12


class WeightedRatioFitness:
    """Compute scalar fitness from metrics using YAML-defined weights.

    For each metric:
    - ratio_to_seed normalization: candidate / (ceiling_factor * seed)
    - direction-aware: maximize → higher ratio = better; minimize → inverted
    - weighted sum across all metrics

    This replaces ALL the hardcoded formulas in the old fitness.py.
    """

    def __init__(self, metrics_config: list[MetricConfig]):
        self._metrics = metrics_config
        total_weight = sum(m.weight for m in self._metrics)
        self._normalized_weights = {
            m.name: m.weight / total_weight if total_weight > 0 else 0.0
            for m in self._metrics
        }

    def compute(
        self,
        candidate_metrics: dict[str, float],
        seed_metrics: dict[str, float],
    ) -> float:
        """Compute weighted fitness score.

        Returns a float where higher = better. 1.0 ≈ seed-equivalent.
        """
        score = 0.0
        for mc in self._metrics:
            cand_val = candidate_metrics.get(mc.name, 0.0)
            seed_val = seed_metrics.get(mc.name, 0.0)
            weight = self._normalized_weights[mc.name]

            if mc.direction == "maximize":
                # Higher is better: ratio = candidate / (ceiling * seed)
                denom = mc.ceiling_factor * max(seed_val, _EPS)
                ratio = min(1.0, cand_val / denom)
            else:
                # Lower is better (latency): ratio = 1 - candidate / (ceiling * seed)
                denom = mc.ceiling_factor * max(seed_val, _EPS)
                ratio = max(0.0, 1.0 - cand_val / denom)

            score += weight * ratio

        return score

    def compute_per_metric(
        self,
        candidate_metrics: dict[str, float],
        seed_metrics: dict[str, float],
    ) -> dict[str, float]:
        """Return per-metric normalized scores (for debugging/reporting)."""
        result: dict[str, float] = {}
        for mc in self._metrics:
            cand_val = candidate_metrics.get(mc.name, 0.0)
            seed_val = seed_metrics.get(mc.name, 0.0)

            if mc.direction == "maximize":
                denom = mc.ceiling_factor * max(seed_val, _EPS)
                result[mc.name] = min(1.0, cand_val / denom)
            else:
                denom = mc.ceiling_factor * max(seed_val, _EPS)
                result[mc.name] = max(0.0, 1.0 - cand_val / denom)

        return result
