"""No-mock local-evolution probe against the real Frontier subprocess.

This test skips unless both Frontier and the official BurstGPT CSV are explicitly configured. It
never substitutes a fixture subprocess or parser fixture for the simulator/data prerequisites.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from vllm_evolve.bench.frontier_sim import resolve_frontier_env
from vllm_evolve.engine.local_frontier_evolve import build_scenarios, evaluate_policy

REPO_ROOT = Path(__file__).resolve().parent.parent
WINNER = (
    REPO_ROOT
    / "reports"
    / "frontier_local_evolution"
    / "sim_winner"
    / "work_variant.py"
)


def _real_inputs() -> tuple[Path, Path, Path] | None:
    csv_value = os.environ.get("VE_BURSTGPT_CSV", "")
    if not csv_value:
        return None
    csv_path = Path(csv_value).expanduser().resolve()
    try:
        frontier_repo, frontier_python = resolve_frontier_env()
    except RuntimeError:
        return None
    repo = Path(frontier_repo)
    python = Path(frontier_python)
    bridge = (
        repo
        / "frontier"
        / "scheduler"
        / "replica_scheduler"
        / "ve_policy_replica_scheduler.py"
    )
    if not (csv_path.is_file() and python.is_file() and bridge.is_file() and WINNER.is_file()):
        return None
    return csv_path, repo, python


@pytest.mark.skipif(
    _real_inputs() is None,
    reason=(
        "requires VE_BURSTGPT_CSV plus VE_FRONTIER_REPO/VE_FRONTIER_PYTHON and the "
        "applied ve_policy bridge; no fake replacement is allowed"
    ),
)
def test_frozen_winner_executes_in_real_frontier_on_official_burstgpt(tmp_path):
    inputs = _real_inputs()
    assert inputs is not None
    csv_path, frontier_repo, frontier_python = inputs
    scenarios, manifest = build_scenarios(
        csv_path,
        fragment_size=8,
        splits=("test",),
    )
    official = next(
        scenario
        for scenario in scenarios
        if scenario.source_kind == "official_burstgpt_fragment"
    )

    result = evaluate_policy(
        policy_name="d3q_real_e2e",
        policy_path=str(WINNER),
        scenario=official,
        seeds=[0],
        out_dir=tmp_path,
    )

    assert manifest["source_path"] == str(csv_path)
    assert result.source_kind == "official_burstgpt_fragment"
    assert result.completed_min == 8
    assert result.marker_ok is True
    assert result.invocations > 0
    assert result.fallbacks < result.invocations
    assert result.per_seed[0].marker["policy_sha256"] == result.policy_sha256

    commands = list(tmp_path.rglob("frontier_command.json"))
    assert len(commands) == 1
    command = json.loads(commands[0].read_text(encoding="utf-8"))
    assert command["cwd"] == str(frontier_repo)
    assert command["argv"][:3] == [str(frontier_python), "-m", "frontier.main"]
    assert command["seed"] == 0
    assert command["policy_path"] == str(WINNER)
