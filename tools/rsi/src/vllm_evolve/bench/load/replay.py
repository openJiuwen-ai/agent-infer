"""Replay regime: drive a real production trace at its own arrival pattern.

Loads the trace via the dataset registry. Most traces carry their own arrival
timestamps; ShareGPT does not, so arrivals are synthesized as a Poisson process
at ``request_rate_qps``.
"""
from __future__ import annotations

import random

from vllm_evolve.bench.datasets import load_dataset
from vllm_evolve.bench.datasets.base import TraceRequest


def generate(workload: dict, seed: int = 0, trace_path: str | None = None) -> list[TraceRequest]:
    dataset = workload.get("dataset")
    if not dataset:
        raise ValueError("replay workload requires a 'dataset' name")
    path = trace_path or workload.get("trace_path")
    if not path:
        raise ValueError(
            "replay workload requires a 'trace_path' (set it in the profile or pass trace_path)"
        )

    ds = load_dataset(dataset, path, max_requests=workload.get("max_requests"))
    requests = list(ds.requests)

    # ShareGPT has no real arrival timestamps -> synthesize Poisson arrivals.
    if dataset == "sharegpt":
        rate = float(workload.get("request_rate_qps", 10.0))
        rng = random.Random(seed)
        t = 0.0
        for r in requests:
            if rate > 0:
                t += rng.expovariate(rate)
            r.arrival_s = t
    return requests
