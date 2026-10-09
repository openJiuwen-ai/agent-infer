"""Latency regime: low-rate Poisson arrivals with no queueing.

A low ``request_rate_qps`` keeps the server unsaturated so each request's TTFT /
TPOT reflect single-request responsiveness rather than queue wait.
"""
from __future__ import annotations

import random

from vllm_evolve.bench.datasets.base import TraceRequest
from vllm_evolve.bench.load.base import sample_lognormal


def generate(workload: dict, seed: int = 0) -> list[TraceRequest]:
    rng = random.Random(seed)
    n = int(workload.get("num_requests", 200))
    rate = float(workload.get("request_rate_qps", 1.0))
    pm = float(workload.get("prompt_len_mean", 512))
    ps = float(workload.get("prompt_len_std", 128))
    om = float(workload.get("output_len_mean", 256))
    os_ = float(workload.get("output_len_std", 64))
    out: list[TraceRequest] = []
    t = 0.0
    for i in range(n):
        if rate > 0:
            t += rng.expovariate(rate)
        out.append(
            TraceRequest(
                arrival_s=t,
                prompt_tokens=sample_lognormal(rng, pm, ps),
                output_tokens=sample_lognormal(rng, om, os_),
                request_id=f"lat_{i:06d}",
            )
        )
    return out
