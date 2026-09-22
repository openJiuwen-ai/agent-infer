"""Throughput regime: saturating load.

All requests are marked ready at t=0; the backend dispatches them at maximum
concurrency to drive the server to saturation. The metric of interest is how
much SLO-meeting goodput the policy sustains at peak.
"""
from __future__ import annotations

import random

from vllm_evolve.bench.datasets.base import TraceRequest
from vllm_evolve.bench.load.base import sample_lognormal


def generate(workload: dict, seed: int = 0) -> list[TraceRequest]:
    rng = random.Random(seed)
    n = int(workload.get("num_requests", 1000))
    pm = float(workload.get("prompt_len_mean", 512))
    ps = float(workload.get("prompt_len_std", 128))
    om = float(workload.get("output_len_mean", 256))
    os_ = float(workload.get("output_len_std", 64))
    out: list[TraceRequest] = []
    for i in range(n):
        out.append(
            TraceRequest(
                arrival_s=0.0,
                prompt_tokens=sample_lognormal(rng, pm, ps),
                output_tokens=sample_lognormal(rng, om, os_),
                request_id=f"thr_{i:06d}",
            )
        )
    return out
