# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Accuracy CI configuration without Docker, users, model downloads or GPUs."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.agentbench.test_compare import write_run
from tests.e2e.acc import conftest as accuracy_fixtures, test_correctness as accuracy
from tests.e2e.helpers.case_loader import load_case_file
from tests.helpers.marks import test_marks

pytestmark = test_marks(hardware="cpu", ci="ready")
CASES = sorted((Path(__file__).resolve().parents[1] / "e2e/acc/cases").rglob("*.json"))


def test_manual_runs_do_not_prepare_ci(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("regular accuracy runs must not prepare the CI host")

    monkeypatch.setattr(accuracy_fixtures.subprocess, "run", unexpected)
    config = SimpleNamespace(getoption=lambda name: False)
    assert accuracy_fixtures.correctness_ci_environment.__wrapped__(config) is None


def test_jiuwen_probe_uses_selected_cli_environment(tmp_path, monkeypatch):
    executable = tmp_path / "selected-venv" / "bin" / "jiuwenswarm"
    executable.parent.mkdir(parents=True)
    executable.touch()
    monkeypatch.setattr(accuracy_fixtures.shutil, "which", lambda name: str(executable))
    assert accuracy_fixtures._jiuwen_python("jiuwenswarm") == executable.with_name("python")


@pytest.mark.parametrize("agent", ["claude", "dsh", "zcode", "jiuwenswarm"])
def test_ci_reports_use_selected_agent(agent):
    options = SimpleNamespace(xmlpath=None, log_file=None)
    values = {"--correctness-ci": True, "keyword": agent}
    accuracy_fixtures.pytest_configure(SimpleNamespace(getoption=values.get, option=options))
    assert Path(options.xmlpath) == Path("nightly-logs") / agent / "results.xml"
    assert Path(options.log_file) == Path("nightly-logs") / agent / "pytest.log"


def test_ci_reports_preserve_explicit_paths():
    options = SimpleNamespace(xmlpath="custom/results.xml", log_file="custom/pytest.log")
    values = {"--correctness-ci": True, "keyword": "dsh"}
    accuracy_fixtures.pytest_configure(SimpleNamespace(getoption=values.get, option=options))
    assert options.xmlpath == "custom/results.xml"
    assert options.log_file == "custom/pytest.log"


def test_manual_runs_keep_default_report_options():
    options = SimpleNamespace(xmlpath=None, log_file=None)
    accuracy_fixtures.pytest_configure(SimpleNamespace(getoption=lambda name: False, option=options))
    assert options.xmlpath is None
    assert options.log_file is None


@pytest.mark.parametrize("case_path", CASES, ids=lambda path: path.stem)
@pytest.mark.parametrize("evaluation", [None, "/opt/agentbench/evaluation.yaml"])
def test_evaluation_override_preserves_cases(monkeypatch, case_path, evaluation):
    original_bytes = case_path.read_bytes()
    original_case = load_case_file(case_path)
    original_params = dict(original_case.benchmark_params)
    monkeypatch.setattr(accuracy, "load_case_file", lambda path: original_case)
    captured = []

    class BenchmarkIntercepted(Exception):
        pass

    def capture(config):
        captured.append(config)
        raise BenchmarkIntercepted

    monkeypatch.setattr(accuracy, "run_benchmark", capture)
    options = {
        "--correctness-evaluation-config": evaluation,
        "--task-num": None,
        "--max-concurrency": None,
        "--benchmark-run-as-user": "test-user",
    }
    with pytest.raises(BenchmarkIntercepted):
        accuracy.test_run_with_correctness(SimpleNamespace(getoption=options.get), case_path)
    config = captured[0]
    expected = evaluation or original_params.get("evaluation-config")
    assert config.case.benchmark_params.get("evaluation-config") == expected
    assert config.benchmark_run_as_user == "test-user"
    assert original_case.benchmark_params == original_params
    assert case_path.read_bytes() == original_bytes


@pytest.mark.parametrize(
    ("completed", "failed", "outcomes", "message"),
    [
        (1, 1, ["completed", "failed"], "confirmation_hang"),
        (1, 0, ["completed"], "configured task count"),
        (1, 0, ["completed", "completed"], "not all configured tasks completed"),
        (2, 1, ["completed", "completed"], "summary contains failed tasks"),
    ],
)
def test_automatic_scoring_rejects_partial_runs(tmp_path, completed, failed, outcomes, message):
    run = tmp_path / "run"
    write_run(run, completed=completed, failed=failed)
    rows = [
        {
            "instance_id": f"task-{i}",
            "outcome": outcome,
            "termination_reason": "confirmation_hang" if outcome == "failed" else None,
        }
        for i, outcome in enumerate(outcomes)
    ]
    (run / "task_index.json").write_text(json.dumps({"tasks": rows}), encoding="utf-8")
    with pytest.raises(AssertionError, match=message):
        accuracy._validate(run, run / "evaluations/correctness.json", expected_task_num=2)


@pytest.mark.parametrize("expected", [None, 2])
def test_existing_run_evaluation_retains_completed_task_scope(monkeypatch, tmp_path, expected):
    run = tmp_path / "run"
    completed, failed = (1, 1) if expected is None else (2, 0)
    write_run(run, completed=completed, failed=failed)
    outcomes = ["completed", "failed"] if expected is None else ["completed", "completed"]
    (run / "task_index.json").write_text(
        json.dumps({"tasks": [{"instance_id": f"task-{i}", "outcome": item} for i, item in enumerate(outcomes)]}),
        encoding="utf-8",
    )

    class EvaluationReached(Exception):
        pass

    def load(path):
        raise EvaluationReached

    monkeypatch.setattr(accuracy, "load_evaluation", load)
    with pytest.raises(EvaluationReached):
        accuracy._validate(run, run / "evaluations/correctness.json", expected_task_num=expected)


def test_automatic_scoring_checks_overridden_task_count(monkeypatch, tmp_path):
    evaluation = tmp_path / "evaluations/eval/correctness.json"
    evaluation.parent.mkdir(parents=True)
    evaluation.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(accuracy, "run_benchmark", lambda config: tmp_path)
    captured = []
    monkeypatch.setattr(accuracy, "_validate", lambda *args, **kwargs: captured.append(kwargs))
    options = {"--task-num": 3, "--max-concurrency": None, "--benchmark-run-as-user": None}
    accuracy.test_run_with_correctness(SimpleNamespace(getoption=options.get), CASES[0])
    assert captured == [{"expected_task_num": 3}]
