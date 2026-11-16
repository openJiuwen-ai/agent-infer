# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Options for real SWE-bench correctness evaluation."""

import json
from pathlib import Path

import pytest

from tests.e2e.helpers.hardware_marks import resolve_pytest_marks


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("correctness")
    group.addoption("--correctness-run-dir", help="Existing finalized SWE-bench run to evaluate")
    group.addoption("--correctness-evaluation-config", help="Evaluation YAML (defaults to current Python)")
    group.addoption("--correctness-baseline", help="Baseline correctness.json for per-task comparison")
    group.addoption("--correctness-candidate", help="Candidate correctness.json for per-task comparison")


def pytest_itemcollected(item: pytest.Item) -> None:
    if item.name != "test_run_with_correctness":
        return
    raw = item.config.getoption("--test-config-file")
    if not raw or not Path(raw).is_file():
        return
    payload = json.loads(Path(raw).read_text(encoding="utf-8"))
    for mark in resolve_pytest_marks(payload.get("mark")):
        item.add_marker(mark)
