"""vllm-evolve benchmark package.

Real-vLLM serving-policy evaluation. Replaces the deleted ``sim/`` package.

Layering (backend-free core first):
    metrics.py   -- per-request records -> aggregate serving metrics (pure)
    slo.py       -- SLO definition + per-request attainment + goodput (pure)
    datasets/    -- real trace adapters (BurstGPT/Azure/ShareGPT/Mooncake) -> JSONL
    load/        -- request-generation strategies (throughput/latency/replay)
    backend.py   -- real vLLM in Docker (GPU-gated)
    runner.py    -- multi-seed aggregation (median + CV escalation)
    compare.py   -- candidate vs baseline -> better/worse/inconclusive
    decision.py  -- decision-policy roll-up

Only ``metrics`` and ``slo`` are imported eagerly here; the rest are imported
lazily by callers so importing the package never pulls a GPU/Docker dependency.
"""
from __future__ import annotations

from vllm_evolve.bench.metrics import BenchMetrics, RequestRecord
from vllm_evolve.bench.slo import SLO, SLOResult, attains, evaluate_slo

__all__ = [
    "BenchMetrics",
    "RequestRecord",
    "SLO",
    "SLOResult",
    "attains",
    "evaluate_slo",
]
