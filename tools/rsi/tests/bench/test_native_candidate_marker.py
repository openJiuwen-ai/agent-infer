"""Codex review P1 (off-GPU): native_bench must prove the candidate scheduler plugin actually ran
before emitting a promotable real eval. If the nonce marker is absent (the default scheduler ran),
the result must be demoted to a non-promotable outcome — never a clean ``real_vllm`` candidate win.

GPU is mocked away: serve_vllm yields a fake handle whose log we control, and run_profile returns a
pre-built EVAL_RESULT BenchResult, so only the marker-gating logic under test executes.
"""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - targets/ namespace bootstrap
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(SRC))

import vllm_evolve.bench.native as nat  # noqa: E402
from vllm_evolve.bench.metrics import RequestRecord  # noqa: E402
from vllm_evolve.bench.outcome import OutcomeClass  # noqa: E402
from vllm_evolve.bench.runner import run_profile  # noqa: E402
from vllm_evolve.bench.runtime import (  # noqa: E402
    PLUGIN_INACTIVE_MARKER,
    PLUGIN_INVOKED_MARKER,
    PLUGIN_REORDERED_MARKER,
)
from vllm_evolve.bench.slo import SLO  # noqa: E402

_NONCE = "n0nce123"
_SEED_SRC = "def schedule_batch(running, state):\n    return None\n"


def test_open_loop_client_pool_matches_requested_concurrency():
    assert nat._client_connection_limits(8192) == (8192, 512)
    assert nat._client_connection_limits(0) == (1, 1)


def test_open_loop_shards_high_concurrency_across_bounded_event_loops():
    assert nat._client_shard_count(1024) == 1
    assert nat._client_shard_count(1025) == 2
    assert nat._client_shard_count(7680) == 8
    assert nat._client_shard_count(15360) == 15
    assert nat._client_shard_count(65536) == 16


def test_open_loop_uses_uvloop_when_available(monkeypatch):
    awaitable = object()
    calls = []
    fake_uvloop = SimpleNamespace(
        run=lambda value: calls.append(value) or "completed"
    )
    monkeypatch.setitem(sys.modules, "uvloop", fake_uvloop)

    assert nat._run_load_loop(awaitable) == "completed"
    assert calls == [awaitable]


def test_open_loop_raises_process_nofile_limit_before_server(monkeypatch):
    limits = [1024, 1048576]
    calls = []

    monkeypatch.setattr(nat.resource, "getrlimit", lambda _kind: tuple(limits))

    def setrlimit(_kind, value):
        calls.append(value)
        limits[:] = value

    monkeypatch.setattr(nat.resource, "setrlimit", setrlimit)
    assert nat._ensure_nofile_capacity(8192) == 9216
    assert calls == [(9216, 1048576)]


def test_open_loop_fails_closed_when_hard_nofile_limit_is_too_low(monkeypatch):
    monkeypatch.setattr(nat.resource, "getrlimit", lambda _kind: (1024, 4096))
    monkeypatch.setattr(nat.resource, "setrlimit", lambda _kind, _value: None)
    with pytest.raises(RuntimeError, match="capacity is insufficient"):
        nat._ensure_nofile_capacity(8192)


def _good_result():
    # a real EVAL_RESULT BenchResult, built off-GPU via the actual runner + synthetic records
    def run_one(seed):
        recs = [RequestRecord(request_id=f"{seed}-{i}", ttft_ms=100.0, tpot_ms=10.0, e2e_ms=600.0,
                              num_prompt_tokens=100, num_output_tokens=50, success=True)
                for i in range(10)]
        return recs, 1.0
    res = run_profile("throughput", "output_throughput_tok_s", run_one,
                      lambda m, s: 7.0, SLO(), [1, 2, 3], seed_tiers=(3,), cv_threshold=1.0)
    assert res.outcome_class is OutcomeClass.EVAL_RESULT
    return res


def _run(monkeypatch, tmp_path, log_text):
    import vllm_evolve.bench.runner as runner_mod
    good = _good_result()                       # build BEFORE patching run_profile
    log_file = tmp_path / "serve.log"
    log_file.write_text(log_text, encoding="utf-8")

    class _Handle:
        def __init__(self):
            self.log_path = str(log_file)
            self.nonce = _NONCE

    @contextlib.contextmanager
    def fake_serve(*a, **k):
        yield _Handle()

    monkeypatch.setattr(nat, "serve_vllm", fake_serve)
    # native_bench does `from ...runner import run_profile` at call time -> patch the source module
    monkeypatch.setattr(runner_mod, "run_profile", lambda *a, **k: good)
    from vllm_evolve.bench.profiles import load_profile
    profile = load_profile(str(REPO_ROOT / "config" / "bench" / "profiles" / "throughput.yaml"))
    return nat.native_bench(_SEED_SRC, profile, runner_kind="candidate",
                            model="facebook/opt-125m", n_requests=8, max_seeds=1)


def test_candidate_without_plugin_marker_is_demoted(monkeypatch, tmp_path):
    # default scheduler ran (no nonce marker) -> NOT a promotable candidate result
    er, _ = _run(monkeypatch, tmp_path, "Running: 2 reqs, Waiting: 0 reqs\n")
    assert er["outcome_class"] == "plugin_load_failure"
    assert er["marker_verified"] is False
    assert er["effective"] is False


def test_candidate_invoked_but_ineffective_is_demoted(monkeypatch, tmp_path):
    # the plugin LOADED (invoked) but did not reorder -> NOT effective -> demoted to
    # plugin_ineffective so compare/keep (LOCK D, source+outcome_class) reject it too, not just the
    # accept gate's effective check (Codex review P1). marker_verified still records the load.
    er, _ = _run(monkeypatch, tmp_path, f"{PLUGIN_INVOKED_MARKER} {_NONCE}\nRunning: 2 reqs\n")
    assert er["outcome_class"] == "plugin_ineffective"
    assert er["marker_verified"] is True
    assert er["effective"] is False


def test_candidate_invoked_and_reordered_is_effective(monkeypatch, tmp_path):
    # invoked + reordered + no fallback -> effective (the only state the accept gate may adopt)
    log = f"{PLUGIN_INVOKED_MARKER} {_NONCE}\n{PLUGIN_REORDERED_MARKER} {_NONCE}\n"
    er, _ = _run(monkeypatch, tmp_path, log)
    assert er["outcome_class"] == "eval_result"
    assert er["marker_verified"] is True
    assert er["effective"] is True


def test_candidate_can_declare_mechanism_inapplicable_without_fake_reorder(
    monkeypatch, tmp_path,
):
    log = (
        f"{PLUGIN_INVOKED_MARKER} {_NONCE}\n"
        f"{PLUGIN_INACTIVE_MARKER} {_NONCE}\n"
    )
    er, _ = _run(monkeypatch, tmp_path, log)
    assert er["outcome_class"] == "eval_result"
    assert er["marker_verified"] is True
    assert er["effective"] is False
    assert er["mechanism_applicable"] is False


def test_exact_trace_is_fully_warmed_and_cache_reset_between_seeds(
    monkeypatch, tmp_path,
):
    import vllm_evolve.bench.runner as runner_mod

    good = _good_result()
    log_file = tmp_path / "serve.log"
    log_file.write_text("", encoding="utf-8")

    class _Handle:
        log_path = str(log_file)
        nonce = ""

    @contextlib.contextmanager
    def fake_serve(*args, **kwargs):
        yield _Handle()

    drive_calls = []
    resets = []

    def fake_drive(requests, **kwargs):
        drive_calls.append([request["request_id"] for request in requests])
        return []

    def fake_run_profile(
        profile, primary_metric, run_one, primary_metric_fn, slo, seeds, **kwargs,
    ):
        for seed in seeds:
            run_one(seed)
        return good

    monkeypatch.setattr(nat, "serve_vllm", fake_serve)
    monkeypatch.setattr(nat, "drive_load", fake_drive)
    monkeypatch.setattr(nat, "reset_prefix_cache", resets.append)
    monkeypatch.setattr(runner_mod, "run_profile", fake_run_profile)
    from vllm_evolve.bench.profiles import load_profile

    profile = load_profile(
        str(REPO_ROOT / "config" / "bench" / "profiles" / "throughput.yaml")
    )
    workload = [
        {"request_id": "a", "num_prompt_tokens": 8, "num_output_tokens": 2},
        {"request_id": "b", "num_prompt_tokens": 16, "num_output_tokens": 2},
    ]
    er, _ = nat.native_bench(
        "",
        profile,
        runner_kind="strong_baseline",
        workload_requests=workload,
        max_seeds=3,
        port=8260,
    )

    assert drive_calls[0] == ["a-warmup", "b-warmup"]
    assert drive_calls[1:] == [
        ["a-seed1", "b-seed1"],
        ["a-seed2", "b-seed2"],
        ["a-seed3", "b-seed3"],
    ]
    assert resets == [8260, 8260, 8260, 8260]
    assert er["measurement_protocol"] == {
        "full_trace_warmup_replays": 1,
        "warmup_request_count": 2,
        "warmup_is_outside_measured_window": True,
        "prefix_cache_reset_before_each_seed": True,
    }
