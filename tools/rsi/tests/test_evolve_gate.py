"""M-D3-full: evolve gate — only evolve code when scheduling is the bottleneck."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(SRC))

from vllm_evolve.engine.evolve_gate import should_evolve_code  # noqa: E402


def test_scheduling_queue_warrants_evolution():
    ok, why = should_evolve_code(diagnosis_bottleneck="scheduling_queue")
    assert ok is True and "schedule_batch" in why


def test_conflicting_evidence_does_not_evolve():
    # diagnosis says scheduling_queue but profiling shows compute-bound -> resolve, don't evolve
    levers = [{"from_rule": "gemm_compute_bound"}]
    ok, why = should_evolve_code(levers, diagnosis_bottleneck="scheduling_queue")
    assert ok is False and "conflict" in why


def test_compute_bound_does_not_evolve_code():
    levers = [{"from_rule": "gemm_compute_bound", "lever": "quantization"}]
    ok, why = should_evolve_code(levers)
    assert ok is False and "quantization" in why


def test_memory_bound_does_not_evolve_code():
    ok, why = should_evolve_code([{"from_rule": "kv_bandwidth_bound"}])
    assert ok is False and "config knob" in why


def test_no_bottleneck_asks_to_profile():
    ok, why = should_evolve_code(None)
    assert ok is False and "profile more" in why
