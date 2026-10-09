"""M-B3: saturation calibration judgment (zero GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(SRC))

from vllm_evolve.engine.calibrate import (  # noqa: E402
    SweepPoint,
    is_saturated,
    select_saturation_point,
    throughput_plateaued,
)


def _p(cfg, tok_s, util, memf, **kw):
    kw.setdefault("client_saturated", False)   # default: measured NOT client-bound
    return SweepPoint(config=cfg, tok_s=tok_s, sm_util_max=util,
                      mem_used_mb=memf, mem_total_mb=1.0, **kw)


def test_is_saturated_needs_util_and_mem_and_not_client_bound():
    assert is_saturated(_p({"x": 1}, 1800, 95, 0.95)) is True
    assert is_saturated(_p({"x": 1}, 1800, 60, 0.95)) is False   # mem full but SM idle
    assert is_saturated(_p({"x": 1}, 1800, 95, 0.50)) is False   # SM busy but mem not full
    assert is_saturated(_p({"x": 1}, 1800, 95, 0.95, client_saturated=True)) is False
    # unknown client bottleneck (None) is NOT saturation — absence of evidence
    assert is_saturated(_p({"x": 1}, 1800, 95, 0.95, client_saturated=None)) is False


def test_plateau_detection():
    assert throughput_plateaued([_p({}, 1800, 95, 0.95), _p({}, 1850, 95, 0.95)]) is True
    assert throughput_plateaued([_p({}, 1800, 95, 0.95), _p({}, 2300, 95, 0.95)]) is False


def test_select_locks_saturated_plateau_point():
    sweep = [_p({"c": 64}, 1000, 60, 0.50),
             _p({"c": 128}, 1800, 92, 0.93),
             _p({"c": 256}, 1850, 95, 0.95)]   # +2.8% -> plateau
    r = select_saturation_point(sweep)
    assert r.saturated is True and r.locked_config == {"c": 256} and r.locked_tok_s == 1850


def test_select_honest_when_no_saturation():
    sweep = [_p({"c": 64}, 1000, 55, 0.40), _p({"c": 128}, 1100, 60, 0.50)]
    r = select_saturation_point(sweep)
    assert r.saturated is False and r.locked_config is None and r.reasons


def test_select_rejects_when_still_rising():
    sweep = [_p({"c": 128}, 1800, 92, 0.93), _p({"c": 256}, 2300, 95, 0.95)]  # +27%
    r = select_saturation_point(sweep)
    assert r.saturated is False and any("plateau" in x for x in r.reasons)


def test_run_calibration_sweep_locks_saturated_plateau():
    from vllm_evolve.engine.calibrate import run_calibration_sweep

    def evf(cfg):
        tok = {32: 1000, 64: 1700, 128: 1850, 256: 1880, 512: 1890}[cfg["concurrency"]]
        sat = cfg["concurrency"] >= 128
        return {"tok_s": tok, "sm_util_max": 95 if sat else 70,
                "mem_used_mb": 0.95 if sat else 0.6, "mem_total_mb": 1.0,
                "client_saturated": False}

    r = run_calibration_sweep({"model": "m"}, evf)
    assert r.saturated is True and r.locked_config["concurrency"] == 512
    assert r.evidence["sweep_failures"] == 0 and r.evidence["n_evaluated"] == 5


def test_run_calibration_sweep_records_failures_honestly():
    from vllm_evolve.engine.calibrate import run_calibration_sweep

    def boom(cfg):
        raise RuntimeError("box unreachable")

    r = run_calibration_sweep({"model": "m"}, boom, grid=[{"concurrency": 64}])
    assert r.saturated is False and r.evidence["sweep_failures"] == 1


def test_calibration_records_per_point_evidence_including_failures():
    # Codex R3: audit-grade — every point's attempted config recorded; failures keep their text
    from vllm_evolve.engine.calibrate import run_calibration_sweep

    def evf(cfg):
        if cfg["concurrency"] == 64:
            raise RuntimeError("point 64 box error")
        return {"tok_s": 1800, "sm_util_max": 95, "mem_used_mb": 0.95, "mem_total_mb": 1.0,
                "client_saturated": False, "_eval_result": {"bench_config": {"x": 1}},
                "_rendered_serve_args": ["--gpu-memory-utilization", "0.9"]}

    r = run_calibration_sweep({"model": "m"}, evf)
    pts = r.evidence["points"]
    assert len(pts) == 5                                  # EVERY grid point recorded
    failed = [p for p in pts if p["status"] == "failed"]
    assert len(failed) == 1 and failed[0]["config"]["concurrency"] == 64
    assert "box error" in failed[0]["error"]
    ok = [p for p in pts if p["status"] == "ok"]
    assert ok and all("signals" in p and p["rendered_serve_args"] for p in ok)


def test_locked_eval_result_carried_for_freezing():
    sweep = [_p({"c": 128}, 1800, 92, 0.93, eval_result={"bench_config": {"a": 1}}),
             _p({"c": 256}, 1850, 95, 0.95, eval_result={"bench_config": {"b": 2}})]
    r = select_saturation_point(sweep)
    assert r.saturated is True and r.evidence["locked_eval_result"] == {"bench_config": {"b": 2}}
