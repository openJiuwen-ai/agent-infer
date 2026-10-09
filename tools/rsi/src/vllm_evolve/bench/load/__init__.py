"""Load generators: turn a profile's workload spec into a list of requests.

All three regimes emit ``list[TraceRequest]`` (the same canonical type the
dataset adapters produce), so the backend dispatches one uniform shape:

* throughput -> saturating (all requests ready at t=0; backend sends at max concurrency)
* latency    -> low-rate Poisson arrivals (no queueing)
* replay     -> a real trace at its own arrival pattern

Generators are pure and deterministic given a seed.
"""
from __future__ import annotations

from vllm_evolve.bench.datasets.base import TraceRequest
from vllm_evolve.bench.load import latency, replay, throughput


def build_load(profile, seed: int = 0, trace_path: str | None = None) -> list[TraceRequest]:
    wl = profile.workload
    if profile.regime == "throughput":
        return throughput.generate(wl, seed)
    if profile.regime == "latency":
        return latency.generate(wl, seed)
    if profile.regime == "replay":
        return replay.generate(wl, seed, trace_path=trace_path)
    raise ValueError(f"unknown regime {profile.regime!r}")


__all__ = ["build_load", "throughput", "latency", "replay", "TraceRequest"]
