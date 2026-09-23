"""P4-4: `ve tune` mode — plumbing + adoption-gate guards (local_smoke). No GPU."""
from __future__ import annotations

import json

from vllm_evolve.cli import main as ve_cli
from vllm_evolve.tools import phase_guard


def _advance_to_commit():
    phase_guard.reset_state()


def test_tune_local_smoke_plumbing_runs_and_cannot_gain(tmp_path, capsys):
    phase_guard.reset_state()
    pol = tmp_path / "work.py"
    pol.write_text("def schedule_batch(running, state):\n    return None\n", encoding="utf-8")
    rc = ve_cli.main(["tune", str(pol), "scheduling", "--backend", "local_smoke",
                      "--max-rounds", "1", "--max-evals", "1", "--no-holdout"])
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rc == 0 and out["ok"] is True
    assert out["mode"] == "tune" and out["backend"] == "local_smoke"
    assert not out.get("adopted")               # synthetic plumbing never adopts -> no gain
    assert "kept" not in out and "committed" not in out
    phase_guard.reset_state()


def test_tune_rejects_non_scheduling_target(tmp_path, capsys):
    phase_guard.reset_state()
    pol = tmp_path / "work.py"
    pol.write_text("def schedule_batch(running, state):\n    return None\n", encoding="utf-8")
    rc = ve_cli.main(["tune", str(pol), "kv_eviction", "--backend", "local_smoke"])
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rc == 2 and out["outcome_class"] == "unsupported_target"
    phase_guard.reset_state()


def test_tune_missing_policy_rejected(tmp_path, capsys):
    phase_guard.reset_state()
    rc = ve_cli.main(["tune", str(tmp_path / "nope.py"), "--backend", "local_smoke"])
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rc == 2 and out["outcome_class"] == "policy_not_found"
    phase_guard.reset_state()
