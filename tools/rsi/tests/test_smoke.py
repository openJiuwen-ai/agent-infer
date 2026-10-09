"""Smoke tests for vllm-evolve (the supported `scheduling` target).

These verify the scheduling path is importable and minimally functional. They do NOT test live
vLLM integration or statistical correctness. (Per-target smoke tests for the experimental targets
were removed when those targets were deleted — only `scheduling` is supported.)

Run with: pytest tests/ -v
"""
from __future__ import annotations

import os
import sys

import pytest

# project root on the path so we can import targets/ and src/
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ── import tests ────────────────────────────────────────────────────────────

def test_import_scheduling_skeleton():
    from targets.scheduling.skeleton import RequestInfo, ScheduleDecision
    assert RequestInfo is not None and ScheduleDecision is not None


def test_import_scheduling_seed():
    from targets.scheduling.seed import schedule_batch
    assert callable(schedule_batch)


def test_import_gen4_score():
    from targets.scheduling.seeds.gen4_score_based import schedule_batch
    assert callable(schedule_batch)


def test_import_gen5_bimodal():
    from targets.scheduling.seeds.gen5_bimodal_kv_guard import schedule_batch
    assert callable(schedule_batch)


def test_import_config():
    from pathlib import Path

    from src.vllm_evolve.config import load_config
    cfg = load_config(Path(os.path.join(_ROOT, "config", "default.yaml")))
    assert cfg is not None


# ── scheduling skeleton smoke tests ─────────────────────────────────────────

def _make_request(rid: str, prompt_tokens: int, arrival: float, is_prefill: bool = True):
    from targets.scheduling.skeleton import RequestInfo
    return RequestInfo(
        request_id=rid, num_prompt_tokens=prompt_tokens, num_computed_tokens=0,
        num_output_tokens=0, arrival_time_s=arrival, is_prefill=is_prefill,
        kv_blocks_used=0, prefix_cached_tokens=0,
    )


def test_seed_schedule_basic():
    """FCFS seed should return a valid ScheduleDecision."""
    from targets.scheduling.seed import schedule_batch
    waiting = [_make_request("r1", 128, 0.0), _make_request("r2", 256, 0.1),
               _make_request("r3", 512, 0.2)]
    decision = schedule_batch(
        waiting_requests=waiting, running_requests=[], max_num_batched_tokens=2048,
        max_num_seqs=8, available_kv_blocks=512, prefix_cache_hit_rate=0.0)
    assert isinstance(decision.prefill_batch, list) and isinstance(decision.decode_batch, list)
    waiting_ids = {r.request_id for r in waiting}
    for rid in decision.prefill_batch:
        assert rid in waiting_ids


def test_gen4_score_schedule_basic():
    from targets.scheduling.seeds.gen4_score_based import schedule_batch
    waiting = [_make_request("short1", 96, 0.0), _make_request("long1", 640, 0.0),
               _make_request("short2", 96, 0.1)]
    decision = schedule_batch(
        waiting_requests=waiting, running_requests=[], max_num_batched_tokens=2048,
        max_num_seqs=8, available_kv_blocks=96, prefix_cache_hit_rate=0.0)
    assert isinstance(decision.prefill_batch, list)
    assert len(decision.prefill_batch) <= len(waiting)


def test_schedule_empty_waiting_queue():
    from targets.scheduling.seed import schedule_batch
    running = [_make_request("r1", 128, 0.0, is_prefill=False)]
    decision = schedule_batch(
        waiting_requests=[], running_requests=running, max_num_batched_tokens=2048,
        max_num_seqs=8, available_kv_blocks=512, prefix_cache_hit_rate=0.0)
    assert decision.prefill_batch == []


def test_schedule_token_budget_not_exceeded():
    from targets.scheduling.seeds.gen4_score_based import schedule_batch
    max_tokens = 300
    waiting = [_make_request(f"r{i}", 200, float(i)) for i in range(5)]
    decision = schedule_batch(
        waiting_requests=waiting, running_requests=[], max_num_batched_tokens=max_tokens,
        max_num_seqs=8, available_kv_blocks=512, prefix_cache_hit_rate=0.0)
    token_map = {r.request_id: r.remaining_prompt_tokens for r in waiting}
    total = sum(token_map[rid] for rid in decision.prefill_batch)
    assert total <= max_tokens, f"Exceeded token budget: {total} > {max_tokens}"


def test_scheduling_fn_rejects_wrong_signature():
    """schedule_batch should fail when called with wrong args, not silently succeed."""
    from targets.scheduling.seed import schedule_batch
    with pytest.raises(TypeError):
        schedule_batch()


# ── fitness formula (inline; no deleted-target dependency) ───────────────────

def _inline_fitness(r, base_r):
    _EPS = 1e-12

    def _sd(a, b):
        return a / b if b > _EPS else 0.0
    tput = min(1.0, _sd(r["tput"], 2.0 * max(base_r["tput"], 1.0)))
    p50 = max(0.0, 1.0 - _sd(r["p50"], 2.0 * max(base_r["p50"], 1.0)))
    p99 = max(0.0, 1.0 - _sd(r["p99"], 3.0 * max(base_r["p99"], 1.0)))
    hit = min(1.0, _sd(r["hit"], max(2.0 * base_r["hit"], 0.01)))
    tpot = max(0.0, 1.0 - _sd(r["tpot"], max(2.0 * base_r["tpot"], 1.0)))
    return 0.30 * tput + 0.25 * p50 + 0.25 * p99 + 0.10 * hit + 0.10 * tpot


def test_fitness_formula_seed_scores_half():
    """Seed policy compared to itself should score approximately 0.5."""
    r = {"tput": 100.0, "p50": 200.0, "p99": 500.0, "hit": 0.1, "tpot": 50.0}
    fitness = _inline_fitness(r, r)
    assert 0.4 < fitness < 0.65, f"Expected seed fitness ~0.5, got {fitness}"
