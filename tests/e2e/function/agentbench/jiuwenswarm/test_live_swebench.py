# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Run both Code profiles on the same first configured SWE-bench task."""

import json
import os
from pathlib import Path

import pytest

from agentinfer.agentbench.benchkit.cli import main
from tests.helpers.marks import test_marks

_REPO = Path(__file__).resolve().parents[5]
_BASE_URL = os.environ.get("AGENTBENCH_VLLM_URL", "")
_MODEL = os.environ.get("AGENTBENCH_VLLM_MODEL", "")

pytestmark = [
    *test_marks(hardware="cpu"),
    pytest.mark.skipif(not (_BASE_URL and _MODEL), reason="requires AGENTBENCH_VLLM_URL and AGENTBENCH_VLLM_MODEL"),
]


@pytest.mark.parametrize("profile", ["agent.code.normal", "team.code.normal"])
def test_live_swebench(profile, tmp_path):
    config = os.environ.get(
        "AGENTBENCH_JIUWEN_TEST_CONFIG", str(_REPO / "agentinfer/agentbench/configs/swebench_vllm.yaml")
    )
    executable = os.environ.get("AGENTBENCH_JIUWENSWARM_EXECUTABLE", "jiuwenswarm")
    output = tmp_path / profile
    code = main(
        [
            "run",
            "--config",
            config,
            "--agent-type",
            "jiuwenswarm",
            "--agent-profile",
            profile,
            "--agent-executable",
            executable,
            "--endpoint",
            "/v1/chat/completions",
            "--base-url",
            _BASE_URL,
            "--model",
            _MODEL,
            "--task-num",
            "1",
            "--max-subagents",
            "8",
            "--max-concurrent-subagents",
            "4",
            "--result-dir",
            str(output),
            "--with-correctness",
        ]
    )
    assert code in (0, 1)  # An unresolved patch is independent of runtime completion.
    summary = json.loads((output / "summary.json").read_text())
    assert summary["lifecycle"]["status"] == "completed"
    index = json.loads((output / "task_index.json").read_text())
    assert len(index["tasks"]) == 1
    task = index["tasks"][0]
    assert task["outcome"] == "completed", task
    task_dir = output / "tasks" / task["instance_id"]
    runtime = json.loads((task_dir / "jiuwenswarm-runtime.json").read_text())
    assert runtime["profile"] == profile
    events = [json.loads(line) for line in (task_dir / "jiuwenswarm-agent-limits.jsonl").read_text().splitlines()]
    assert events
    assert max(event["launched"] for event in events) <= 8
    assert max(event["peak"] for event in events) <= 4
    if profile == "team.code.normal":
        assert any(event["event"] == "launch" for event in events)
    evaluations = list((output / "evaluations").glob("*/manifest.json"))
    assert len(evaluations) == 1
    assert json.loads(evaluations[0].read_text())["status"] == "completed"
