# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Verify the synthetic task traverses real dataset/cache/workspace preparation."""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from agentinfer.agentbench.benchkit.config import load_config
from agentinfer.agentbench.benchkit.dataset import load_tasks
from agentinfer.agentbench.benchkit.workspace import ensure_repo_cache, prepare_workspace
from tests.e2e.function.agentbench import test_smoke as smoke_suite
from tests.e2e.function.agentbench.smoke_inputs import prepare_smoke_case
from tests.e2e.helpers.case_loader import E2EConfig, load_case_file, load_pytest_case, validate_benchmark_user
from tests.helpers.marks import test_marks

pytestmark = test_marks(hardware="cpu", ci="ready")
SMOKE_CASE = Path(__file__).resolve().parents[2] / "tests/e2e/function/agentbench/cases/qwen3_coder_agents_1x1.json"


def test_local_task_needs_no_remote_repository_or_dependencies(tmp_path: Path) -> None:
    case = load_case_file(SMOKE_CASE, benchmark_index=0)
    original = replace(E2EConfig.from_case(case), result_root=tmp_path)
    config = prepare_smoke_case(original)
    bench = load_config(config.bench_config)
    assert bench.dataset.name == "smoke"
    tasks = load_tasks(bench.dataset.index_path, bench.dataset.selection_path, 1)
    assert len(tasks) == 1
    # An absent origin proves cache preparation cannot silently fetch GitHub.
    cache = ensure_repo_cache(tasks[0], bench.dataset.cache_dir)
    remotes = subprocess.check_output(["git", "remote"], cwd=cache, text=True)
    assert remotes.strip() == ""
    workspace = prepare_workspace(tasks[0], bench.dataset.cache_dir, tmp_path / "workspace")
    check = [sys.executable, "-B", "check_arithmetic.py"]
    broken = subprocess.run(check, cwd=workspace, capture_output=True, text=True)
    assert broken.returncode != 0
    assert "AssertionError" in broken.stderr
    arithmetic = workspace / "arithmetic.py"
    arithmetic.write_text(arithmetic.read_text().replace("return a - b", "return a + b"))
    fixed = subprocess.run(check, cwd=workspace, capture_output=True, text=True)
    assert fixed.returncode == 0, fixed.stderr
    assert "Ran 3 tests" in fixed.stderr


def test_agent_matrix_shares_deployment_and_preserves_each_benchmark() -> None:
    """Each array entry retains its runtime while sharing the declared deployment."""
    payload = json.loads(SMOKE_CASE.read_text(encoding="utf-8"))
    cases = [load_case_file(SMOKE_CASE, benchmark_index=index) for index in range(len(payload["benchmark_params"]))]
    assert [(case.benchmark_params["agent-type"], case.benchmark_params["agent-profile"]) for case in cases] == [
        ("claude", "single"),
        ("claude", "plan-subagent"),
        ("dsh", "single"),
        ("dsh", "plan-subagent"),
        ("jiuwenswarm", "code.normal"),
        ("zcode", "single"),
        ("zcode", "plan-subagent"),
    ]
    assert all(case.deploy == cases[0].deploy and case.mark == cases[0].mark for case in cases)
    assert cases[0].deploy.result_root == Path("test-results/function/agentbench")


def test_single_benchmark_object_remains_supported(tmp_path: Path) -> None:
    """Existing performance/replay-style cases do not require an array index."""
    payload = json.loads(SMOKE_CASE.read_text(encoding="utf-8"))
    payload["benchmark_params"] = payload["benchmark_params"][0]
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_case_file(path) == load_case_file(path, benchmark_index=0)
    assert load_case_file(path).benchmark_params["agent-type"] == "claude"


@pytest.mark.parametrize("params", [[], [None], [{}, "invalid"], "invalid"])
def test_invalid_benchmark_array_is_rejected(tmp_path: Path, params) -> None:
    """Reject empty arrays and malformed entries before dispatching a benchmark."""
    payload = json.loads(SMOKE_CASE.read_text(encoding="utf-8"))
    payload["benchmark_params"] = params
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="benchmark_params must be"):
        load_case_file(path, benchmark_index=0)


@pytest.mark.parametrize("index", [None, -1, 7, True])
def test_benchmark_array_requires_valid_explicit_index(index) -> None:
    """Do not silently select an array entry or accept an out-of-range index."""
    with pytest.raises(ValueError, match="benchmark_index"):
        load_case_file(SMOKE_CASE, benchmark_index=index)


def test_pytest_overrides_come_from_command_not_json(tmp_path: Path) -> None:
    """The shared loader applies load overrides and ignores a JSON OS user."""
    payload = json.loads(SMOKE_CASE.read_text(encoding="utf-8"))
    payload["benchmark_params"] = payload["benchmark_params"][1]
    payload["benchmark_params"]["benchmark-run-as-user"] = "json-user"
    path = tmp_path / "case.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    options = {"--task-num": None, "--max-concurrency": None, "--benchmark-run-as-user": None}
    config = load_pytest_case(SimpleNamespace(getoption=options.get), path)
    assert config.benchmark_run_as_user is None
    assert config.task_num == config.max_concurrency == 1
    options.update({"--task-num": 2, "--max-concurrency": 2, "--benchmark-run-as-user": " bench-user "})
    config = load_pytest_case(SimpleNamespace(getoption=options.get), path)
    assert config.task_num == config.max_concurrency == 2
    assert config.benchmark_run_as_user == "bench-user"


@pytest.mark.parametrize("index,uid", [(0, 0), (4, 0), (1, 1000)])
def test_non_planning_or_non_root_smoke_does_not_require_user(index, uid) -> None:
    """Direct profiles keep root; an ordinary caller already satisfies planning policy."""
    config = E2EConfig.from_case(load_case_file(SMOKE_CASE, benchmark_index=index))
    validate_benchmark_user(config, effective_uid=uid)


@pytest.mark.parametrize("index", [1, 3, 6])
def test_root_plan_smoke_requires_explicit_benchmark_user(index) -> None:
    """Every planning agent requires the same explicit CLI user when pytest is root."""
    config = E2EConfig.from_case(load_case_file(SMOKE_CASE, benchmark_index=index))
    assert config.benchmark_run_as_user is None
    with pytest.raises(ValueError, match="requires --benchmark-run-as-user"):
        validate_benchmark_user(config, effective_uid=0)


@pytest.mark.parametrize(
    "user,expected", [("root", "ordinary user"), ("missing-user", "does not exist"), ("bench-user", None)]
)
def test_root_plan_smoke_validates_target_account(monkeypatch, user, expected) -> None:
    """Validate account existence and UID without depending on host OS accounts."""

    def getpwnam(name):
        if name == "missing-user":
            raise KeyError(name)
        return SimpleNamespace(pw_uid=0 if name == "root" else 1000)

    monkeypatch.setitem(sys.modules, "pwd", SimpleNamespace(getpwnam=getpwnam))
    config = replace(E2EConfig.from_case(load_case_file(SMOKE_CASE, benchmark_index=1)), benchmark_run_as_user=user)
    if expected:
        with pytest.raises(ValueError, match=expected):
            validate_benchmark_user(config, effective_uid=0)
    else:
        validate_benchmark_user(config, effective_uid=0)


def test_runtime_preflight_fails_before_fixture_or_model_start(monkeypatch) -> None:
    """Missing runtime dependencies stop the smoke before expensive side effects."""
    preflight = AsyncMock(side_effect=RuntimeError("socat is unavailable"))
    monkeypatch.setattr(smoke_suite, "check_agent_preflight", preflight)
    monkeypatch.setattr(E2EConfig, "validate_prerequisites", lambda self: [])
    monkeypatch.setattr(smoke_suite.shutil, "which", lambda name: "/usr/bin/git")
    prepare = Mock()
    monkeypatch.setattr(smoke_suite, "prepare_smoke_case", prepare)
    server = Mock()
    options = SimpleNamespace(getoption=lambda name: None)
    with pytest.raises(RuntimeError, match="socat is unavailable"):
        smoke_suite.test_agent_smoke(options, SMOKE_CASE, 0, server)
    preflight.assert_awaited_once_with("claude", Path("claude"))
    prepare.assert_not_called()
    server.assert_not_called()
