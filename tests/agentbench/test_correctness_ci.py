# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Accuracy CI configuration without Docker, users, model downloads or GPUs."""

import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tests.agentbench.test_compare import write_run
from tests.e2e.acc import conftest as accuracy_fixtures, output as accuracy_output, test_correctness as accuracy
from tests.e2e.helpers.case_loader import E2EConfig, benchmark_params_to_run_args, load_case_file, load_case_group
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


def test_explicit_benchmark_directory_hands_off_parent_to_benchmark_user(monkeypatch, tmp_path):
    from contextlib import nullcontext

    from tests.e2e.helpers import benchmark

    config = E2EConfig.from_case(BENCHMARKS[0][1]).with_runtime_overrides(
        benchmark_run_as_user="benchmark-user", benchmark_run_as_user_set=True
    )
    output = tmp_path / "run/acc/example"
    owners = {}

    def prepare_parent(path, *, run_as_user):
        path.mkdir(parents=True)
        owners[path] = run_as_user

    def invoke(config, argv):
        assert owners[output.parent] == "benchmark-user"
        output.mkdir()

    monkeypatch.setattr(E2EConfig, "validate_prerequisites", lambda self: [])
    monkeypatch.setattr(E2EConfig, "resolve_run_hardware_slug", lambda self: "test-hardware")
    monkeypatch.setattr(benchmark, "_repo_cache_owner", lambda config: nullcontext())
    monkeypatch.setattr(benchmark, "ensure_benchmark_inputs_prepared", lambda config: None)
    monkeypatch.setattr(benchmark, "ensure_directory", prepare_parent)
    monkeypatch.setattr(benchmark, "build_benchmark_argv", lambda config, result_dir: [])
    monkeypatch.setattr(benchmark, "_invoke_benchmark", invoke)
    assert benchmark.run_benchmark(config, result_dir=output, manage_server=False) == output
    assert output.is_dir()


@pytest.mark.parametrize("fail", [False, True])
def test_repo_cache_owner_is_restored_between_cases(tmp_path, monkeypatch, fail):
    from tests.e2e.helpers import benchmark

    cache = tmp_path / "repo-cache"
    config = tmp_path / "benchmark.yaml"
    config.write_text(json.dumps({"dataset": {"cache_dir": cache.as_posix()}}), encoding="utf-8")
    monkeypatch.setattr(benchmark, "os", SimpleNamespace(name="posix", geteuid=lambda: 0, getegid=lambda: 0))
    owners = {cache: (2002, 2002)}
    monkeypatch.setattr(benchmark, "_chown_tree", lambda path, uid, gid: owners.update({path: (uid, gid)}))
    original_git_config = {key: value for key, value in os.environ.items() if key.startswith("GIT_CONFIG_")}

    def run_case():
        with benchmark._repo_cache_owner(SimpleNamespace(bench_config=config)):
            assert owners[cache] == (0, 0)
            owners[cache] = (2002, 2002)
            if fail:
                raise RuntimeError("simulated benchmark failure")

    if fail:
        with pytest.raises(RuntimeError, match="simulated benchmark failure"):
            run_case()
    else:
        run_case()
    assert owners[cache] == (0, 0)
    with benchmark._repo_cache_owner(SimpleNamespace(bench_config=config)):
        assert owners[cache] == (0, 0)
    assert original_git_config == {key: value for key, value in os.environ.items() if key.startswith("GIT_CONFIG_")}


@pytest.mark.parametrize("platform,uid", [("nt", 0), ("posix", 2002)])
def test_repo_cache_owner_does_not_chown_without_root(monkeypatch, platform, uid):
    from tests.e2e.helpers import benchmark

    monkeypatch.setattr(benchmark, "os", SimpleNamespace(name=platform, geteuid=lambda: uid))

    def unexpected(*args):
        pytest.fail("cache ownership must not change without root")

    monkeypatch.setattr(benchmark, "_chown_tree", unexpected)
    with benchmark._repo_cache_owner(None):
        pass


def test_claude_accuracy_budget_reaches_sudo_command_and_restores_for_other_agents(pytester, monkeypatch):
    case = next(case for _, case in BENCHMARKS if case.benchmark_params["agent-type"] == "claude")
    monkeypatch.setenv("CLAUDE_CODE_MAX_CONTEXT_TOKENS", "original")
    monkeypatch.setenv("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "original")
    pytester.makepyfile(f"""
        import os
        from dataclasses import replace
        from pathlib import Path
        import pytest
        from tests.e2e.acc.conftest import correctness_environment
        from tests.e2e.helpers.case_loader import load_case_group, wrap_argv_for_user

        @pytest.fixture
        def correctness_ci_environment():
            return None

        @pytest.fixture(params=['claude', 'dsh'])
        def benchmark_case(request):
            original = load_case_group(Path({str(CASES[0])!r}))[0]
            return replace(original, benchmark_params={{**original.benchmark_params, 'agent-type': request.param}})

        def test_budget(benchmark_case, correctness_environment):
            if benchmark_case.benchmark_params['agent-type'] == 'claude':
                command = wrap_argv_for_user(['vllm'], run_as_user='test-user', cwd=Path('.'), effective_uid=0)
                assert 'CLAUDE_CODE_MAX_CONTEXT_TOKENS={case.deploy.serve_args["max-model-len"]}' in command
                assert 'CLAUDE_CODE_MAX_OUTPUT_TOKENS=8192' in command
            else:
                assert os.environ['CLAUDE_CODE_MAX_CONTEXT_TOKENS'] == 'original'
                assert os.environ['CLAUDE_CODE_MAX_OUTPUT_TOKENS'] == 'original'
    """)
    pytester.runpytest_inprocess("-q").assert_outcomes(passed=2)


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
    requirements = []
    monkeypatch.setattr(accuracy_fixtures, "_freeze", lambda python, path: requirements.append(path))
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
        config = SimpleNamespace(getoption=options.get, stash={})
        prepare = accuracy_fixtures._prepare_correctness_ci_environment(config)
        return SimpleNamespace(
            prepare=prepare,
            commands=commands,
            downloads=downloads,
            runtime=runtime,
            requirements=requirements,
            artifacts=accuracy_fixtures._artifact_root(config),
        )

    return configure


@pytest.mark.parametrize("cache", [None, "/tmp/accuracy-hf-cache"])
def test_ci_model_download_supports_optional_hf_cache(ci_environment, cache):
    ci = ci_environment(cache=cache)
    case = load_case_group(CASES[0])[0]
    ci.prepare(case)
    ci.prepare(case)
    assert ci.downloads == [((case.deploy.model.as_posix(),), {"cache_dir": cache, "max_workers": 4})]
    ci.runtime.preflight.assert_awaited_once()
    agent = case.benchmark_params["agent-type"]
    assert ci.requirements == [
        ci.artifacts / agent / "bench-requirements.txt",
        ci.artifacts / agent / "evaluation-requirements.txt",
    ]


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
def test_accuracy_reuses_shared_agent_config(agent, stem):
    for _, case in BENCHMARKS:
        if case.benchmark_params["agent-type"] == agent:
            assert Path(case.benchmark_params["config"]) == Path("agentinfer/agentbench/configs") / f"{stem}.yaml"


@pytest.mark.parametrize("case_path,case", BENCHMARKS, ids=[case.test_name for _, case in BENCHMARKS])
def test_accuracy_effective_config_matches_served_model_and_context(tmp_path, case_path, case):
    from agentinfer.agentbench.benchkit.cli import _apply_cli_overrides, _parser
    from agentinfer.agentbench.benchkit.config import load_config

    repo = Path(__file__).resolve().parents[2]
    config_path = repo / case.benchmark_params["config"]
    original = config_path.read_bytes()
    argv = benchmark_params_to_run_args(case.benchmark_params, tmp_path / "run", repo)
    args = _parser().parse_args(argv[4:])
    effective = _apply_cli_overrides(args, load_config(config_path))
    served_model = case.deploy.serve_args.get("served-model-name", case.deploy.model.as_posix())
    assert effective.backend.model == served_model
    # Claude's effective budget is checked through its actual accuracy fixture.
    if effective.agent.type != "claude":
        assert effective.agent.context_window == case.deploy.serve_args["max-model-len"]
        assert effective.agent.max_tokens <= effective.agent.context_window
    assert config_path.read_bytes() == original


def test_jiuwen_probe_uses_selected_cli_environment(tmp_path, monkeypatch):
    executable = tmp_path / "selected-venv" / "bin" / "jiuwenswarm"
    executable.parent.mkdir(parents=True)
    executable.touch()
    monkeypatch.setattr(accuracy_fixtures.shutil, "which", lambda name: str(executable))
    assert accuracy_fixtures._jiuwen_python("jiuwenswarm") == executable.with_name("python")


@pytest.fixture
def report_config(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    suite = checkout / "tests/e2e/acc"
    (suite / "cases").mkdir(parents=True)
    for path in CASES:
        (suite / "cases" / path.name).write_bytes(path.read_bytes())
    monkeypatch.setattr(accuracy_fixtures, "__file__", str(suite / "conftest.py"))
    monkeypatch.setattr(accuracy_output, "datetime", SimpleNamespace(now=lambda: datetime(2026, 11, 24, 10, 30)))

    def configure(**values):
        options = SimpleNamespace(xmlpath=values.pop("xmlpath", None), log_file=values.pop("log_file", None))
        return SimpleNamespace(getoption=values.get, option=options, stash={})

    return configure


@pytest.mark.parametrize("feature", [None, "claude", "dsh", "zcode", "jiuwenswarm", "baseline", "agentinfer"])
@pytest.mark.parametrize("ci", [False, True])
def test_reports_share_one_timestamped_feature_directory(report_config, feature, ci):
    config = report_config(keyword=feature, **{"--correctness-ci": ci})
    accuracy_fixtures.pytest_configure(config)
    artifacts = accuracy_fixtures._artifact_root(config)
    assert config.option.xmlpath is None
    assert Path(config.option.log_file) == artifacts / "pytest.log"
    assert artifacts.parent.parts[-3:] == ("tests", "run", "acc")
    assert re.fullmatch(rf"{feature or 'all'}-\d{{8}}-\d{{6}}-[0-9a-f]{{6}}", artifacts.name)
    assert accuracy_fixtures._artifact_root(config) == artifacts
    second = report_config(keyword=feature)
    assert accuracy_fixtures._artifact_root(second) != artifacts


def test_ci_reports_preserve_explicit_paths(report_config):
    config = report_config(keyword="dsh", xmlpath="custom/results.xml", log_file="custom/pytest.log")
    accuracy_fixtures.pytest_configure(config)
    assert config.option.xmlpath == "custom/results.xml"
    assert config.option.log_file == "custom/pytest.log"


def test_collection_does_not_create_report_directories(report_config):
    config = report_config(collectonly=True)
    accuracy_fixtures.pytest_configure(config)
    assert config.option.xmlpath is None
    assert config.option.log_file is None
    assert accuracy_fixtures._ARTIFACT_ROOT not in config.stash


def test_selected_case_preserves_its_result_root(report_config, tmp_path):
    case = json.loads(CASES[0].read_text(encoding="utf-8"))
    case["result_root"] = str(tmp_path / "custom-results")
    path = tmp_path / "custom.json"
    path.write_text(json.dumps(case), encoding="utf-8")
    config = report_config(**{"--test-config-file": str(path)})
    artifacts = accuracy_fixtures._artifact_root(config)
    assert artifacts.parent == tmp_path / "custom-results"
    assert artifacts.name.startswith(case["scenario"] + "-")


def test_real_pytest_keeps_repeated_invocation_logs_separate(pytester, report_config):
    pytester.makeconftest("""
        from tests.e2e.acc.conftest import pytest_configure, correctness_artifacts

        def pytest_addoption(parser):
            parser.addoption('--test-config-file')
    """)
    pytester.makepyfile("""
        import logging

        def test_artifacts(correctness_artifacts):
            logging.warning('accuracy artifact layout probe')
            (correctness_artifacts / 'case-artifact.txt').write_text('completed')
    """)
    for _ in range(2):
        pytester.runpytest_inprocess("-q").assert_outcomes(passed=1)
    checkout = Path(accuracy_fixtures.__file__).resolve().parents[3]
    directories = sorted((checkout / "tests/run/acc").glob("all-*"))
    assert len(directories) == 2
    for directory in directories:
        assert "accuracy artifact layout probe" in (directory / "pytest.log").read_text(encoding="utf-8")
        assert (directory / "case-artifact.txt").read_text() == "completed"


def test_generated_artifacts_are_not_collected_as_tests(pytester):
    pytester.makeconftest("from tests.conftest import collect_ignore")
    for name in ("run", "result"):
        directory = pytester.path / name / "acc" / "example"
        directory.mkdir(parents=True)
        (directory / "test_generated.py").write_text("raise AssertionError('generated artifact was collected')")
    pytester.makepyfile("def test_real(): pass")
    pytester.runpytest_inprocess("-q").assert_outcomes(passed=1)


@pytest.mark.parametrize("pipeline", ["test-ready.yml", "test-nightly.yml"])
def test_ci_uploads_accuracy_results_and_diagnostics_only(report_config, pipeline):
    import shlex

    import yaml

    repo = Path(__file__).resolve().parents[2]
    document = yaml.safe_load((repo / ".buildkite" / pipeline).read_text(encoding="utf-8"))

    def jobs(steps):
        for step in steps:
            if isinstance(step, dict):
                if "steps" in step:
                    yield from jobs(step["steps"])
                else:
                    yield step

    def expand_braces(pattern):
        match = re.search(r"\{([^{}]+)\}", pattern)
        if match is None:
            yield pattern
            return
        for choice in match[1].split(","):
            yield from expand_braces(pattern[: match.start()] + choice + pattern[match.end() :])

    checked = 0
    for job in jobs(document["steps"]):
        commands = job.get("commands", [])
        if not any("tests/e2e/acc/" in command for command in commands):
            continue
        argv = shlex.split(next(command for command in commands if "tests/e2e/acc/" in command))
        config = report_config(keyword=argv[argv.index("-k") + 1])
        artifacts = accuracy_fixtures._artifact_root(config)
        scenario = "baseline" if config.getoption("keyword") == "baseline" else "agentinfer"
        agent = "jiuwenswarm" if config.getoption("keyword") == "jiuwenswarm" else "claude"
        case_name = accuracy_output.output_dir_name(f"{agent}-{scenario}")
        checkout = artifacts.parents[3]
        run_dir = artifacts.parent / case_name
        result_dir = checkout / "tests/result/acc" / case_name
        paths = [
            artifacts / "pytest.log",
            artifacts / agent / "bench-requirements.txt",
            artifacts / agent / "evaluation-requirements.txt",
            artifacts / scenario / "vllm-logs/worker.log",
            run_dir / "manifest.json",
            run_dir / "summary.json",
            run_dir / "task_index.json",
            run_dir / "proxy.stderr.log",
            run_dir / "tasks/task-example/result.json",
            run_dir / "tasks/task-example/model.patch",
            run_dir / "tasks/task-example/transcript.jsonl",
            run_dir / "tasks/task-example/jiuwenswarm.jsonl",
            run_dir / "tasks/task-example/dsh-run-status.json",
            run_dir / "tasks/task-example/terminal-final.log",
            run_dir / "tasks/task-example/jiuwenbox-server.stderr.log",
            run_dir / "tasks/task-example/dsh.stdout.log",
            run_dir / "tasks/task-example/zcode-error.log",
            run_dir / "tasks/task-example/jiuwenswarm-instance/home/.jiuwenswarm/logs/server.log",
            run_dir / "tasks/task-example/jiuwenswarm-instance/data/agent/.logs/agent_server.log",
            run_dir / "tasks/task-example/jiuwenswarm-instance/data/logs/logs/run/jiuwen.log",
            run_dir / "tasks/task-example/jiuwenbox-audit/example.audit.log",
            run_dir / "evaluations/eval-example/correctness.json",
            run_dir / "evaluations/eval-example/evaluation_manifest.json",
            run_dir / "evaluations/eval-example/task_results.jsonl",
            run_dir / "evaluations/eval-example/harness.stdout.log",
            run_dir / "evaluations/eval-example/harness.stderr.log",
            run_dir / "evaluations/eval-example/raw/model.eval-example.json",
            run_dir
            / "evaluations/eval-example/raw/logs/run_evaluation/eval-example/model/task-example/run_instance.log",
            run_dir / "evaluations/eval-example/raw/logs/run_evaluation/eval-example/model/task-example/report.json",
            run_dir
            / "evaluations/eval-example/raw/logs/run_evaluation/eval-example/model/task-example/test_output.txt",
            result_dir / "correctness.json",
        ]
        excluded = [
            run_dir / "workspaces/task-example/source.py",
            run_dir / "workspaces/task-example/report.json",
            run_dir / "workspaces/task-example/.git/config",
            run_dir / "tasks/task-example/claude_state/cache/cached.json",
            run_dir / "tasks/task-example/dsh-home/.git/config",
            run_dir / "tasks/task-example/zcode-home/node_modules/package/package.json",
            run_dir / "tasks/task-example/jiuwenswarm-instance/data/agent/workspace/skills/example.py",
            run_dir / "tasks/task-example/jiuwenswarm-instance/home/.npm/_logs/install.log",
            run_dir / "tasks/task-example/terminal-1.log",
            run_dir / "evaluations/eval-example/inputs/dataset.json",
            run_dir / "requests.jsonl",
            result_dir / "unrelated.txt",
            checkout / "tests/run/acc/unselected-example/pytest.log",
        ]
        for path in paths + excluded:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        uploaded = {
            path
            for pattern in job["artifact_paths"]
            for expanded in expand_braces(pattern)
            for path in checkout.glob(expanded)
            if path.is_file()
        }
        assert set(paths) == uploaded
        assert "--log-file" not in job["env"]["PYTEST_ADDOPTS"]
        checked += 1
    assert checked


def test_ci_gpu_image_is_built_once_and_retained_until_both_pipelines_finish():
    import runpy
    import shlex

    import yaml

    repo = Path(__file__).resolve().parents[2]
    buildkite = repo / ".buildkite"
    render = runpy.run_path(str(buildkite / "upload_pipeline.py"))["render"]
    pipelines = {
        name: yaml.safe_load(render(buildkite / name, buildkite / "hardware.yml"))
        for name in ("pipeline.yml", "test-ready.yml", "test-nightly.yml")
    }

    def jobs(steps, inherited=()):
        for step in steps:
            dependencies = step.get("depends_on", inherited)
            if isinstance(dependencies, str):
                dependencies = [dependencies]
            if "steps" in step:
                yield from jobs(step["steps"], dependencies)
            else:
                yield step, dependencies

    all_jobs = [job for pipeline in pipelines.values() for job in jobs(pipeline["steps"])]
    keys = [job["key"] for job, _ in all_jobs if "key" in job]
    assert len(keys) == len(set(keys))
    for _, dependencies in all_jobs:
        assert set(dependencies) <= set(keys), "a dynamically uploaded job has an unknown dependency"
    builds = [
        job
        for job, _ in all_jobs
        if any("build --pull agentinfer-ci" in command for command in job.get("commands", []))
    ]
    assert len(builds) == 1
    producer = builds[0]
    metadata = {}
    for command in producer["commands"]:
        argv = shlex.split(command)
        if argv[:3] == ["buildkite-agent", "meta-data", "set"]:
            metadata[argv[3]] = argv[4]
    consumers = 0
    for job, dependencies in all_jobs:
        for plugin in job.get("plugins", []):
            for config in plugin.values():
                service = config.get("run")
                if service not in {"agentinfer-ci", "agentinfer-accuracy"}:
                    continue
                key = f"docker-compose-plugin-built-image-tag-{service}-docker/docker-compose.yml"
                assert key in metadata, "a GPU consumer has no prebuilt image metadata"
                assert producer["key"] in dependencies
                consumers += 1
    assert consumers == 5  # serve smoke, functional E2E, ready accuracy, two nightly jobs
    assert len(set(metadata.values())) == 1
    cleanup = next(job for job, _ in all_jobs if job.get("key") == "cleanup-gpu-image")
    uploads = {job["key"] for job in pipelines["pipeline.yml"]["steps"] if "command" in job}
    assert set(cleanup["depends_on"]) == uploads | {producer["key"]}
    assert cleanup["allow_dependency_failure"] is True
    # A child pipeline must not remove the image while the other lane still uses it.
    for name in ("test-ready.yml", "test-nightly.yml"):
        for job, _ in jobs(pipelines[name]["steps"]):
            assert not any("docker rmi" in command and "-gpu" in command for command in job.get("commands", []))


@pytest.mark.parametrize("case_path,original_case", BENCHMARKS, ids=[case.test_name for _, case in BENCHMARKS])
@pytest.mark.parametrize("evaluation", [None, "/opt/agentbench/evaluation.yaml"])
def test_evaluation_override_preserves_cases(monkeypatch, tmp_path, case_path, original_case, evaluation):
    original_bytes = case_path.read_bytes()
    original_params = dict(original_case.benchmark_params)
    captured = []

    class BenchmarkIntercepted(Exception):
        pass

    def capture(config, *, manage_server, result_dir):
        assert manage_server is False
        assert result_dir.parent == tmp_path.parent
        assert result_dir.name.startswith(f"{original_params['agent-type']}-{original_case.scenario.value}-")
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
        accuracy.test_run_with_correctness(
            SimpleNamespace(getoption=options.get), original_case, lambda config: config, tmp_path, tmp_path / "results"
        )
    config = captured[0]
    expected = evaluation or original_params.get("evaluation-config")
    assert config.case.benchmark_params.get("evaluation-config") == expected
    assert config.benchmark_run_as_user == "test-user"
    assert config.result_root == tmp_path.parent
    assert config.vllm_log_dir == tmp_path / original_case.scenario.value / "vllm-logs"
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
    def benchmark(config, *, result_dir, **kwargs):
        evaluation = result_dir / "evaluations/eval/correctness.json"
        evaluation.parent.mkdir(parents=True)
        evaluation.write_text("{}", encoding="utf-8")
        return result_dir

    monkeypatch.setattr(accuracy, "run_benchmark", benchmark)
    captured = []
    monkeypatch.setattr(accuracy, "_validate", lambda *args, **kwargs: captured.append(kwargs))
    options = {"--task-num": 3, "--max-concurrency": None, "--benchmark-run-as-user": None}
    accuracy.test_run_with_correctness(
        SimpleNamespace(getoption=options.get), BENCHMARKS[0][1], lambda config: config, tmp_path, tmp_path / "results"
    )
    assert captured == [{"expected_task_num": 3}]


@pytest.mark.parametrize("failed", [False, True])
def test_score_export_preserves_original_evidence_even_when_benchmark_fails(monkeypatch, tmp_path, failed):
    evidence = []

    def benchmark(config, *, result_dir, manage_server):
        evaluation = result_dir / "evaluations/eval-example/correctness.json"
        evaluation.parent.mkdir(parents=True)
        evaluation.write_text(json.dumps({"counts": {"error": int(failed)}}), encoding="utf-8")
        manifest = evaluation.parent / "evaluation_manifest.json"
        manifest.write_text('{"original": true}', encoding="utf-8")
        evidence.append((result_dir, evaluation, manifest.read_bytes()))
        if failed:
            raise RuntimeError("benchmark exited with code 1")
        return result_dir

    validated = []
    monkeypatch.setattr(accuracy, "run_benchmark", benchmark)
    monkeypatch.setattr(accuracy, "_validate", lambda *args, **kwargs: validated.append(args))
    session = tmp_path / "tests/run/acc/all-session"
    results = tmp_path / "tests/result/acc"
    options = {"--task-num": None, "--max-concurrency": None, "--benchmark-run-as-user": None}

    def run():
        accuracy.test_run_with_correctness(
            SimpleNamespace(getoption=options.get), BENCHMARKS[0][1], lambda config: config, session, results
        )

    if failed:
        with pytest.raises(RuntimeError, match="benchmark exited with code 1"):
            run()
    else:
        run()
    run_dir, evaluation, original_manifest = evidence[0]
    exported = results / run_dir.name
    assert (exported / "correctness.json").read_bytes() == evaluation.read_bytes()
    assert (evaluation.parent / "evaluation_manifest.json").read_bytes() == original_manifest
    assert exported.name == run_dir.name
    assert {path.name for path in exported.iterdir()} == {"correctness.json"}
    assert validated == ([] if failed else [(run_dir, evaluation)])


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


@pytest.mark.parametrize(
    "keyword,lane,total,failures",
    [
        (None, None, 8, 1),
        ("dsh", None, 2, 1),
        ("baseline", None, 4, 1),
        ("agentinfer", None, 4, 0),
        ("jiuwenswarm", None, 2, 0),
        (None, "ready", 1, 0),
        (None, "nightly", 7, 1),
        ("baseline", "nightly", 4, 1),
        ("agentinfer", "nightly", 3, 0),
    ],
)
def test_grouped_pytest_reuses_server_and_continues_after_agent_failure(
    pytester, monkeypatch, keyword, lane, total, failures
):
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
    pytester.makeini(
        "[pytest]\nmarkers =\n    L20: [hardware] GPU\n    cards_2: two GPUs\n"
        "    ready: ready pipeline\n    nightly: nightly pipeline"
    )
    pytester.makepyfile("""
        from tests.e2e.conftest import pytest_generate_tests
        from tests.e2e.acc.conftest import correctness_server, _record_test_request
        from tests.e2e.helpers.case_loader import E2EConfig

        def test_agents(case_file, benchmark_case, correctness_server):
            config = correctness_server(E2EConfig.from_case(benchmark_case))
            _record_test_request(config)
            agent = benchmark_case.benchmark_params['agent-type']
            ready = agent == 'jiuwenswarm' and config.scenario.value == 'agentinfer'
            assert benchmark_case.mark['ci'] == ('ready' if ready else 'nightly')
            assert 'mark' not in benchmark_case.benchmark_params
            assert not (agent == 'dsh' and config.scenario.value == 'baseline'), 'simulated agent failure'
    """)
    args = ["-q"] + (["-k", keyword] if keyword else [])
    if lane:
        args.extend(["-m", lane])
    result = pytester.runpytest_inprocess(*args)
    result.assert_outcomes(passed=total - failures, failed=failures, deselected=8 - total)
    starts = [event for event in events if event[0] == "start"]
    stops = [event for event in events if event[0] == "stop"]
    expected_scenarios = {keyword} if keyword in {"baseline", "agentinfer"} else {"baseline", "agentinfer"}
    if lane == "ready":
        expected_scenarios = {"agentinfer"}
    assert {event[1] for event in starts} == expected_scenarios
    assert len(starts) == len(stops) == len(expected_scenarios)
    assert len([event for event in events if event[0] == "request"]) == total
    for _, scenario, lifecycle_socket, port in starts:
        requests = [event for event in events if event[:2] == ("request", scenario)]
        assert all(event[2:] == (lifecycle_socket, port) for event in requests)


@pytest.mark.parametrize("invalid", [None, [], {"hardware": "cpu"}, {"num_cards": 1}])
def test_benchmark_mark_override_only_changes_ci_lane(tmp_path, invalid):
    payload = json.loads(CASES[0].read_text(encoding="utf-8"))
    payload["benchmark_params"][-1]["mark"] = invalid
    path = tmp_path / "invalid-mark.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="may only override ci"):
        load_case_group(path)
