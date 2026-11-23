# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Accuracy CI configuration without Docker, users, model downloads or GPUs."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

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
    assert accuracy_fixtures._prepare_correctness_ci_environment(config) is None


@pytest.fixture
def ci_environment(tmp_path, monkeypatch):
    import huggingface_hub

    from agentinfer.agentbench.agents import registry
    from agentinfer.agentbench.benchkit.evaluation import config as evaluation_config

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.setattr(accuracy_fixtures, "__file__", str(checkout / "tests/e2e/acc/conftest.py"))
    environment = {"PATH": "/usr/local/bin"}
    monkeypatch.setattr(accuracy_fixtures, "os", SimpleNamespace(name="posix", geteuid=lambda: 0, environ=environment))
    socket = SimpleNamespace(stat=lambda: SimpleNamespace(st_gid=1000))
    monkeypatch.setattr(
        accuracy_fixtures, "Path", lambda value: socket if value == "/var/run/docker.sock" else Path(value)
    )
    monkeypatch.setitem(sys.modules, "grp", SimpleNamespace(getgrgid=lambda gid: None))
    commands = []

    def run(argv, **kwargs):
        commands.append(argv)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(accuracy_fixtures.subprocess, "run", run)
    monkeypatch.setattr(accuracy_fixtures, "_freeze", lambda *args: None)
    monkeypatch.setattr(accuracy_fixtures, "_jiuwen_python", lambda executable: Path("python"))
    monkeypatch.setattr(
        evaluation_config, "load_evaluation_config", lambda path: SimpleNamespace(harness_python="python")
    )
    runtime = SimpleNamespace(preflight=AsyncMock())
    monkeypatch.setattr(registry, "get_runtime", lambda name: runtime)
    downloads = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *args, **kwargs: downloads.append((args, kwargs)))

    def configure(*, user="test-user", cache=None):
        if cache is not None:
            environment["HF_HUB_CACHE"] = cache
        options = {
            "--correctness-ci": True,
            "--benchmark-run-as-user": user,
            "--correctness-evaluation-config": "eval.yaml",
        }
        prepare = accuracy_fixtures._prepare_correctness_ci_environment(
            SimpleNamespace(getoption=options.get, stash={})
        )
        return SimpleNamespace(prepare=prepare, commands=commands, downloads=downloads, runtime=runtime)

    return configure


@pytest.mark.parametrize("cache", [None, "/tmp/accuracy-hf-cache"])
def test_ci_model_download_supports_optional_hf_cache(ci_environment, cache):
    ci = ci_environment(cache=cache)
    case = load_case_group(CASES[0])[0]
    ci.prepare(case)
    ci.prepare(case)
    assert ci.downloads == [((case.deploy.model.as_posix(),), {"cache_dir": cache, "max_workers": 4})]
    ci.runtime.preflight.assert_awaited_once()


@pytest.mark.parametrize("user", [None, "test-user"])
def test_jiuwen_ci_probes_match_root_benchmark_identity(ci_environment, user):
    ci = ci_environment(user=user)
    case = next(case for _, case in BENCHMARKS if case.benchmark_params["agent-type"] == "jiuwenswarm")
    ci.prepare(case)
    namespace = next(command for command in ci.commands if "bwrap" in command)
    assert namespace[0] == "bwrap"
    client = next(command for command in ci.commands if "PYTHONPATH" in command)
    assert client[:3] == ["env", "-u", "PYTHONPATH"]
    ci.runtime.preflight.assert_awaited_once()
    assert len(ci.downloads) == 1
    if user is None:
        assert not any(command[0] in {"sudo", "id", "usermod"} for command in ci.commands)


def test_plan_subagent_ci_requires_benchmark_user(ci_environment):
    ci = ci_environment(user=None)
    with pytest.raises(pytest.UsageError, match="--benchmark-run-as-user for plan-subagent"):
        ci.prepare(BENCHMARKS[0][1])
    assert not ci.downloads
    ci.runtime.preflight.assert_not_awaited()


def test_jiuwen_ci_stops_before_download_if_sandbox_probe_fails(ci_environment, monkeypatch):
    ci = ci_environment(user=None)

    def fail(argv, **kwargs):
        raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(accuracy_fixtures.subprocess, "run", fail)
    case = next(case for _, case in BENCHMARKS if case.benchmark_params["agent-type"] == "jiuwenswarm")
    with pytest.raises(subprocess.CalledProcessError):
        ci.prepare(case)
    assert not ci.downloads
    ci.runtime.preflight.assert_not_awaited()


@pytest.mark.parametrize("agent,stem", [("claude", "swebench_vllm"), ("dsh", "swebench_vllm_dsh")])
def test_accuracy_reuses_shared_config_defaults(agent, stem):
    from agentinfer.agentbench.benchkit.config import load_config

    repo = Path(__file__).resolve().parents[2]
    shared = load_config(repo / "agentinfer/agentbench/configs" / f"{stem}.yaml")
    assert shared.agent.context_window == 262144
    assert shared.backend.model.endswith("-FP8")
    for _, case in BENCHMARKS:
        if case.benchmark_params["agent-type"] == agent:
            config = load_config(repo / case.benchmark_params["config"])
            assert Path(case.benchmark_params["config"]).stem == stem
            assert config.agent == shared.agent
            assert config.backend.model == case.deploy.model.as_posix()


def test_accuracy_cases_use_fp8_with_65536_server_context():
    for _, case in BENCHMARKS:
        model = "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
        assert case.deploy.model.as_posix() == case.benchmark_params["model"] == model
        assert case.deploy.serve_args["served-model-name"] == model
        assert case.deploy.serve_args["max-model-len"] == 65536


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
    assert options.xmlpath is None
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


def test_automatic_scoring_rejects_jiuwen_sandbox_failure_in_completed_run(tmp_path):
    run = tmp_path / "run"
    write_run(run, completed=2, failed=0)
    log = run / "tasks/task-0/jiuwenbox-server.stderr.log"
    log.parent.mkdir(parents=True)
    log.write_text("bwrap: Can't mount proc on /newroot/proc: Operation not permitted\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="Jiuwen sandbox failed") as error:
        accuracy._validate(run, run / "evaluations/correctness.json", expected_task_num=2)
    assert str(log) in str(error.value)


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


@pytest.mark.parametrize("keyword", [None, "dsh", "baseline", "agentinfer", "jiuwenswarm"])
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
    total = {None: 8, "dsh": 2, "baseline": 4, "agentinfer": 4, "jiuwenswarm": 2}[keyword]
    failures = 0 if keyword in {"agentinfer", "jiuwenswarm"} else 1
    result.assert_outcomes(passed=total - failures, failed=failures, deselected=8 - total)
    starts = [event for event in events if event[0] == "start"]
    stops = [event for event in events if event[0] == "stop"]
    assert len(starts) == len(stops) == (1 if keyword in {"baseline", "agentinfer"} else 2)
    assert len([event for event in events if event[0] == "request"]) == total
    for _, scenario, lifecycle_socket, port in starts:
        requests = [event for event in events if event[:2] == ("request", scenario)]
        assert all(event[2:] == (lifecycle_socket, port) for event in requests)
