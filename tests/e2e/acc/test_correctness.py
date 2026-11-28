# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Real CLI E2E for PR #64; requires Docker and swebench==4.1.0."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from agentinfer.agentbench.benchkit.evaluation.compare import load_evaluation
from tests.e2e.helpers.benchmark import run_benchmark
from tests.e2e.helpers.case_loader import BenchmarkCase, E2EConfig

from .output import export_correctness, output_dir_name


def _cli(*args: str) -> str:
    completed = subprocess.run(
        ["vllm", "bench", "serve", "--agentinfer", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed.stdout


def _validate(evaluation: Path) -> None:
    summary, _, _ = load_evaluation(evaluation)
    print(f"[correctness] counts={summary.counts.model_dump()} resolved_rate={summary.resolved_rate} {evaluation}")
    assert summary.counts.error == 0, f"scoring errors: {summary.counts.error}; see {evaluation}"


@pytest.mark.usefixtures("correctness_environment")
def test_run_with_correctness(
    pytestconfig: pytest.Config,
    benchmark_case: BenchmarkCase,
    correctness_server,
    correctness_artifacts: Path,
    correctness_results: Path,
) -> None:
    case = benchmark_case
    assert case.benchmark_params.get("with-correctness") is True
    evaluation_config = pytestconfig.getoption("--correctness-evaluation-config")
    if evaluation_config:
        case = replace(case, benchmark_params={**case.benchmark_params, "evaluation-config": evaluation_config})
    config = E2EConfig.from_case(case).with_benchmark_load(
        task_num=pytestconfig.getoption("--task-num"),
        max_concurrency=pytestconfig.getoption("--max-concurrency"),
    )
    user = pytestconfig.getoption("--benchmark-run-as-user")
    if user is not None:
        config = config.with_runtime_overrides(
            benchmark_run_as_user=str(user).strip() or None,
            benchmark_run_as_user_set=True,
        )
    agent = case.benchmark_params.get("agent-type", "claude")
    config = replace(
        config,
        result_root=correctness_artifacts.parent,
        vllm_log_dir=correctness_artifacts / config.scenario.value / "vllm-logs",
    )
    config = correctness_server(config)
    print(f"[correctness] agent={agent} scenario={config.scenario.value}", flush=True)
    run_dir = config.result_root / output_dir_name(f"{agent}-{config.scenario.value}")
    try:
        run_benchmark(config, result_dir=run_dir, manage_server=False)
    finally:
        evaluations = list((run_dir / "evaluations").glob("*/correctness.json"))
        if len(evaluations) == 1:
            export_correctness(evaluations[0], correctness_results / run_dir.name)
    assert len(evaluations) == 1, f"expected one scoring result, found {evaluations}"
    _validate(evaluations[0])


@pytest.mark.cpu
def test_evaluate_existing_run(pytestconfig: pytest.Config) -> None:
    raw = pytestconfig.getoption("--correctness-run-dir")
    if not raw:
        pytest.skip("provide --correctness-run-dir to score existing artifacts")
    run_dir = Path(raw).resolve()
    before = set((run_dir / "evaluations").glob("*/correctness.json"))
    args = ["evaluate", "--result-path", str(run_dir)]
    config = pytestconfig.getoption("--correctness-evaluation-config")
    if config:
        args.extend(["--config", config])
    _cli(*args)
    created = set((run_dir / "evaluations").glob("*/correctness.json")) - before
    assert len(created) == 1, created
    _validate(created.pop())


@pytest.mark.cpu
def test_compare_correctness(pytestconfig: pytest.Config) -> None:
    baseline = pytestconfig.getoption("--correctness-baseline")
    candidate = pytestconfig.getoption("--correctness-candidate")
    if not baseline and not candidate:
        pytest.skip("provide baseline and candidate correctness.json paths")
    assert baseline and candidate, "both --correctness-baseline and --correctness-candidate are required"
    report = json.loads(_cli("compare-correctness", "--baseline", baseline, "--candidate", candidate))
    assert report["status"] == "passed", report
    assert report["regressions"] == [], report
