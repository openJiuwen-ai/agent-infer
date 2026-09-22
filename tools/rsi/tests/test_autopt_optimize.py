"""M3: L3 config search — grid/coordinate/budget/cache/anti-fabrication (no GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(SRC))

from vllm_evolve.core.schemas import Profile, Spec  # noqa: E402
from vllm_evolve.engine.optimize import (  # noqa: E402
    EvalBudget,
    coordinate_descent,
    grid_search,
    optimize,
)


def _evalfn(metric_by_val, knob, metric="goodput_req_s", verified=True):
    def fn(config):
        return Profile(metrics={metric: metric_by_val[config[knob]]}, marker_verified=verified)
    return fn


def _spec(metric="goodput_req_s", direction="max"):
    return Spec(metric=metric, direction=direction)


def test_grid_picks_best_verified():
    knobs = {"gpu_memory_utilization": [0.85, 0.90, 0.95]}
    fn = _evalfn({0.85: 5.0, 0.90: 7.0, 0.95: 6.0}, "gpu_memory_utilization")
    cand = grid_search(knobs, {}, _spec(), fn, EvalBudget(10))
    assert cand.value == {"gpu_memory_utilization": 0.90}
    assert cand.score == 7.0 and cand.marker_verified is True and cand.evals_used == 3


def test_unverified_never_wins():
    # anti-fabrication: a config that wasn't marker-verified cannot be selected
    knobs = {"gpu_memory_utilization": [0.85, 0.90]}
    fn = _evalfn({0.85: 99.0, 0.90: 1.0}, "gpu_memory_utilization", verified=False)
    cand = grid_search(knobs, {}, _spec(), fn, EvalBudget(10))
    assert cand.value == {} and cand.score is None and cand.marker_verified is False
    assert "unverified" in cand.note or "no verified" in cand.note


def test_budget_caps_evals():
    knobs = {"max_num_seqs": [64, 128, 256, 512]}
    fn = _evalfn({64: 1.0, 128: 2.0, 256: 9.0, 512: 3.0}, "max_num_seqs")
    cand = grid_search(knobs, {}, _spec(), fn, EvalBudget(2))  # stops before 256
    assert cand.evals_used == 2 and cand.value == {"max_num_seqs": 128}


def test_cache_skips_repeat_evals():
    knobs = {"gpu_memory_utilization": [0.85, 0.90, 0.95]}
    fn = _evalfn({0.85: 5.0, 0.90: 7.0, 0.95: 6.0}, "gpu_memory_utilization")
    cache: dict = {}
    b1 = EvalBudget(10)
    grid_search(knobs, {}, _spec(), fn, b1, cache)
    assert b1.used == 3
    b2 = EvalBudget(10)
    grid_search(knobs, {}, _spec(), fn, b2, cache)  # all cached now
    assert b2.used == 0


def test_min_direction():
    knobs = {"max_model_len": [2048, 4096, 8192]}
    fn = _evalfn({2048: 100.0, 4096: 50.0, 8192: 80.0}, "max_model_len", metric="ttft_p99_ms")
    cand = grid_search(knobs, {}, _spec("ttft_p99_ms", "min"), fn, EvalBudget(10))
    assert cand.value == {"max_model_len": 4096} and cand.score == 50.0  # lowest TTFT


def test_eval_failure_is_non_fatal():
    knobs = {"gpu_memory_utilization": [0.85, 0.90]}

    def fn(config):
        if config["gpu_memory_utilization"] == 0.85:
            raise RuntimeError("bench failed")
        return Profile(metrics={"goodput_req_s": 4.0}, marker_verified=True)

    cand = grid_search(knobs, {}, _spec(), fn, EvalBudget(10))
    assert cand.value == {"gpu_memory_utilization": 0.90} and cand.score == 4.0
    assert cand.trials[0]["score"] is None  # the failed trial recorded honestly


def test_coordinate_descent_fewer_evals_than_grid():
    knobs = {"x": [1, 2, 3], "y": [10, 20, 30]}

    def fn(config):
        return Profile(metrics={"goodput_req_s": (config.get("x") or 0) + (config.get("y") or 0)},
                       marker_verified=True)

    cand = coordinate_descent(knobs, {}, _spec(), fn, EvalBudget(20))
    assert cand.evals_used == 6  # 3 + 3, vs 9 for full grid
    assert cand.value == {"x": 3, "y": 30} and cand.score == 33.0


def test_optimize_dispatch():
    knobs = {"gpu_memory_utilization": [0.85, 0.90]}
    fn = _evalfn({0.85: 5.0, 0.90: 7.0}, "gpu_memory_utilization")
    assert optimize(knobs, {}, _spec(), fn, strategy="grid", max_evals=5).score == 7.0


def test_cmd_optimize_cli(monkeypatch, capsys):
    import json

    from vllm_evolve.cli import main as cli_main
    from vllm_evolve.engine import profile as prof_mod

    def fake_collect(config):
        val = config["max_num_seqs"]
        return Profile(metrics={"goodput_req_s": {64: 3.0, 128: 9.0, 256: 5.0}[val]},
                       marker_verified=True)

    monkeypatch.setattr(prof_mod, "collect_profile", fake_collect)
    rc = cli_main.main(["optimize", "--target", "config:max_num_seqs",
                      "--space", "64,128,256", "--metric", "goodput_req_s"])
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rc == 0 and out["ok"] is True
    assert out["value"] == {"max_num_seqs": 128} and out["score"] == 9.0


def test_cmd_optimize_empty_space(capsys):
    import json

    from vllm_evolve.cli import main as cli_main
    rc = cli_main.main(["optimize", "--target", "config:max_num_seqs", "--space", " "])
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rc == 2 and out["outcome_class"] == "empty_search_space"


def test_cmd_optimize_rejects_unwired_knob(capsys):
    # honesty: a knob the bench cannot render must be refused, not "searched".
    # quantization/kv_cache_dtype ARE wired now (the bench renders them), so use a knob
    # genuinely absent from the unified BenchConfig lever registry.
    import json

    from vllm_evolve.cli import main as cli_main
    rc = cli_main.main(["optimize", "--target", "config:made_up_knob", "--space", "1,2"])
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rc == 2 and out["outcome_class"] == "knob_not_wired"
    assert "max_num_seqs" in out["supported"]
