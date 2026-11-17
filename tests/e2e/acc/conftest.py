# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Options for real SWE-bench correctness evaluation."""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("correctness")
    group.addoption("--correctness-run-dir", help="Existing finalized SWE-bench run to evaluate")
    group.addoption("--correctness-evaluation-config", help="Evaluation YAML (defaults to current Python)")
    group.addoption("--correctness-baseline", help="Baseline correctness.json for per-task comparison")
    group.addoption("--correctness-candidate", help="Candidate correctness.json for per-task comparison")
