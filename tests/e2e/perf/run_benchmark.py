# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Explicit pytest entry point for JSON-driven E2E performance benchmarks."""

from __future__ import annotations

from pathlib import Path

import pytest

from ..helpers.benchmark import run_benchmark, validate_completed_run
from ..helpers.case_loader import load_pytest_case


def test_benchmark_completes(pytestconfig: pytest.Config, case_file: Path) -> None:
    """Run one independently collected JSON benchmark case."""

    config = load_pytest_case(pytestconfig, case_file)
    run_dir = run_benchmark(config)
    summary = validate_completed_run(run_dir)
    print(
        f"[benchmark:{config.scenario.value}] completed: "
        f"tasks={summary['tasks']['completed']} requests={summary['requests']['requests']} "
        f"dir={run_dir}",
        flush=True,
    )
