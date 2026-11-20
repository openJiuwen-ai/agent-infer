# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Accuracy CI configuration without Docker, users, model downloads or GPUs."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.agentbench.test_compare import write_run
from tests.e2e.acc import conftest as accuracy_fixtures, test_correctness as accuracy
from tests.e2e.helpers.case_loader import E2EConfig, load_case_file, load_case_group
from tests.helpers.marks import test_marks

pytestmark = test_marks(hardware="cpu", ci="ready")
pytest_plugins = ["pytester"]
CASES = sorted((Path(__file__).resolve().parents[1] / "e2e/acc/cases").rglob("*.json"))
BENCHMARKS = [(path, case) for path in CASES for case in load_case_group(path)]


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


@pytest.mark.parametrize("agent", ["claude", "dsh", "zcode", "jiuwenswarm", "baseline", "agentinfer"])
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


@pytest.mark.parametrize("case_path,original_case", BENCHMARKS, ids=[case.test_name for _, case in BENCHMARKS])
@pytest.mark.parametrize("evaluation", [None, "/opt/agentbench/evaluation.yaml"])
def test_evaluation_override_preserves_cases(monkeypatch, case_path, original_case, evaluation):
    original_bytes = case_path.read_bytes()
    original_params = dict(original_case.benchmark_params)
    captured = []

    class BenchmarkIntercepted(Exception):
        pass

    def capture(config, *, start_server):
        assert start_server is False
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
        accuracy.test_run_with_correctness(SimpleNamespace(getoption=options.get), original_case, lambda config: config)
    config = captured[0]
    expected = evaluation or original_params.get("evaluation-config")
    assert config.case.benchmark_params.get("evaluation-config") == expected
    assert config.benchmark_run_as_user == "test-user"
    assert config.result_root.parts[-2:] == (original_params["agent-type"], original_case.scenario.value)
    assert config.vllm_log_dir.parts[-2:] == (original_case.scenario.value, "vllm-logs")
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
    monkeypatch.setattr(accuracy, "run_benchmark", lambda config, **kwargs: tmp_path)
    captured = []
    monkeypatch.setattr(accuracy, "_validate", lambda *args, **kwargs: captured.append(kwargs))
    options = {"--task-num": 3, "--max-concurrency": None, "--benchmark-run-as-user": None}
    accuracy.test_run_with_correctness(SimpleNamespace(getoption=options.get), BENCHMARKS[0][1], lambda config: config)
    assert captured == [{"expected_task_num": 3}]


def test_group_loader_preserves_single_benchmark_format(tmp_path):
    payload = json.loads(CASES[0].read_text(encoding="utf-8"))
    payload["benchmark_params"] = payload["benchmark_params"][0]
    path = tmp_path / "single.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_case_group(path) == (load_case_file(path),)


@pytest.mark.parametrize("invalid", [[], [None], [{"agent-type": "../escape"}]])
def test_group_loader_rejects_invalid_benchmarks(tmp_path, invalid):
    payload = json.loads(CASES[0].read_text(encoding="utf-8"))
    payload["benchmark_params"] = invalid
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_case_group(path)


@pytest.mark.parametrize("field,value", [("agent-type", "claude"), ("port", 9000), ("model", "other-model")])
def test_group_loader_rejects_duplicate_agents_or_different_servers(tmp_path, field, value):
    payload = json.loads(CASES[0].read_text(encoding="utf-8"))
    payload["benchmark_params"][1][field] = value
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_case_group(path)


@pytest.mark.parametrize("keyword", [None, "dsh", "baseline"])
def test_grouped_pytest_reuses_server_and_continues_after_agent_failure(pytester, monkeypatch, keyword):
    from contextlib import contextmanager

    cases_dir = pytester.path / "cases"
    cases_dir.mkdir()
    for path in CASES:
        (cases_dir / path.name).write_bytes(path.read_bytes())
    events = []

    @contextmanager
    def server(config):
        events.append(("start", config.scenario.value, config.lifecycle_socket, config.port))
        try:
            yield
        finally:
            events.append(("stop", config.scenario.value))

    def request(config):
        events.append(("request", config.scenario.value, config.lifecycle_socket, config.port))

    monkeypatch.setattr(accuracy_fixtures, "managed_vllm", server)
    monkeypatch.setattr(E2EConfig, "validate_prerequisites", lambda self: [])
    monkeypatch.setattr(accuracy_fixtures, "_record_test_request", request, raising=False)
    pytester.makeini("[pytest]\nmarkers =\n    L20: [hardware] GPU\n    cards_2: two GPUs")
    pytester.makepyfile("""
        from tests.e2e.conftest import pytest_generate_tests
        from tests.e2e.acc.conftest import correctness_server, _record_test_request
        from tests.e2e.helpers.case_loader import E2EConfig

        def test_agents(case_file, benchmark_case, correctness_server):
            config = correctness_server(E2EConfig.from_case(benchmark_case))
            _record_test_request(config)
            agent = benchmark_case.benchmark_params['agent-type']
            assert not (agent == 'dsh' and config.scenario.value == 'baseline'), 'simulated agent failure'
    """)
    args = ["-q"] + (["-k", keyword] if keyword else [])
    result = pytester.runpytest_inprocess(*args)
    total = {None: 8, "dsh": 2, "baseline": 4}[keyword]
    result.assert_outcomes(passed=total - 1, failed=1, deselected=8 - total)
    starts = [event for event in events if event[0] == "start"]
    stops = [event for event in events if event[0] == "stop"]
    assert len(starts) == len(stops) == (1 if keyword == "baseline" else 2)
    assert len([event for event in events if event[0] == "request"]) == total
    for _, scenario, lifecycle_socket, port in starts:
        requests = [event for event in events if event[:2] == ("request", scenario)]
        assert all(event[2:] == (lifecycle_socket, port) for event in requests)
