"""Unit tests for the load generators (throughput / latency / replay)."""
from __future__ import annotations

from pathlib import Path

from vllm_evolve.bench.load import build_load, latency, replay, throughput
from vllm_evolve.bench.profiles import BenchProfile

FIX = Path(__file__).parent / "fixtures"


def test_throughput_saturating_and_deterministic():
    wl = {"num_requests": 50, "prompt_len_mean": 512, "output_len_mean": 256}
    a = throughput.generate(wl, seed=1)
    b = throughput.generate(wl, seed=1)
    assert len(a) == 50
    assert all(r.arrival_s == 0.0 for r in a)          # all ready at t=0
    assert all(r.prompt_tokens >= 1 and r.output_tokens >= 1 for r in a)
    assert [r.prompt_tokens for r in a] == [r.prompt_tokens for r in b]  # deterministic


def test_latency_poisson_monotonic_arrivals():
    wl = {"num_requests": 100, "request_rate_qps": 2.0}
    reqs = latency.generate(wl, seed=7)
    arrivals = [r.arrival_s for r in reqs]
    assert len(reqs) == 100
    assert arrivals == sorted(arrivals)                # non-decreasing
    assert arrivals[-1] > 0.0


def test_replay_mooncake_uses_trace_arrivals():
    wl = {"dataset": "mooncake"}
    reqs = replay.generate(wl, seed=0, trace_path=str(FIX / "mooncake.jsonl"))
    assert [r.arrival_s for r in reqs] == [0.0, 27.0]   # ms -> s, sorted


def test_replay_sharegpt_synthesizes_arrivals():
    wl = {"dataset": "sharegpt", "request_rate_qps": 5.0}
    reqs = replay.generate(wl, seed=3, trace_path=str(FIX / "sharegpt.json"))
    arrivals = [r.arrival_s for r in reqs]
    assert len(reqs) == 2
    assert arrivals == sorted(arrivals)
    assert arrivals[0] > 0.0                            # synthesized, not the index placeholder


def test_replay_requires_trace_path():
    import pytest

    with pytest.raises(ValueError):
        replay.generate({"dataset": "mooncake"}, seed=0, trace_path=None)


def test_build_load_dispatches_by_regime():
    p = BenchProfile(name="t", regime="throughput", workload={"num_requests": 10})
    reqs = build_load(p, seed=0)
    assert len(reqs) == 10
    assert all(r.arrival_s == 0.0 for r in reqs)
