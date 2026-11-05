# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Malformed dependency graphs must be rejected before execution is possible."""

from dataclasses import replace

import pytest

from agentinfer.agentbench.replay.analyzer import analyze_replay_trace
from agentinfer.agentbench.replay.config import ReplayBenchConfig
from agentinfer.agentbench.replay.planner import _validate_plan, build_replay_plan
from tests.agentbench.replay.helpers.fixtures import build_fixture, config

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("mutation,message", [("missing", "missing"), ("self", "cycle"), ("cycle", "cycle")])
def test_invalid_send_dependencies_fail_validation(tmp_path, mutation, message):
    path, expected = build_fixture(tmp_path / "fixture", "agentinfer")
    cfg = ReplayBenchConfig.model_validate(config(path, expected, 1, "http://127.0.0.1:1", tmp_path / "result"))
    plan = build_replay_plan(cfg, analyze_replay_trace(path))
    task = plan.tasks[0]
    a, b, c = task.requests
    reference = {"missing": "absent", "self": a.source_key, "cycle": b.source_key}[mutation]
    corrupted = replace(task, requests=(replace(a, send_after=reference), b, c))
    with pytest.raises(ValueError, match=message):
        _validate_plan((corrupted,))
