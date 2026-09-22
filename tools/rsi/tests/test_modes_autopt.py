"""P4-3: autopt mode is a thin, gate-preserving wrapper over engine.orchestrate. No GPU."""
from __future__ import annotations

from vllm_evolve.core.schemas import Profile, Spec
from vllm_evolve.modes import autopt as autopt_mode


def _fake_eval(config: dict) -> Profile:
    # a profile with no verified candidate -> orchestrator concludes DoD-B (no gain)
    return Profile(config=dict(config), metrics={"tok_s": 100.0},
                   gpu={"sm_util_max": 50.0}, vllm={}, saturated=False)


def test_autopt_mode_runs_and_tags_backend():
    spec = Spec(metric="goodput_req_s", direction="max")
    res = autopt_mode.run(spec, eval_fn=_fake_eval, base_config={"concurrency": 4},
                          max_rounds=1, max_evals=1, run_holdout=False, backend="local_smoke")
    assert isinstance(res, dict)
    assert res["backend"] == "local_smoke"
    assert "conclusion" in res and "rounds" in res         # real orchestrator verdict shape
    # a no-verified-candidate run must NOT claim a gain / adoption
    assert not res["adopted"]


def test_autopt_mode_does_not_expose_keep_or_adopt_shortcut():
    # the mode returns the orchestrator verdict only; adoption stays in the frozen gate
    spec = Spec(metric="goodput_req_s", direction="max")
    res = autopt_mode.run(spec, eval_fn=_fake_eval, max_rounds=1, max_evals=1, run_holdout=False)
    assert "kept" not in res and "committed" not in res
