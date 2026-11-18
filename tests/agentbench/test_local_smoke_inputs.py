# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Verify the synthetic task traverses real dataset/cache/workspace preparation."""

import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from agentinfer.agentbench.benchkit.config import load_config
from agentinfer.agentbench.benchkit.dataset import load_tasks
from agentinfer.agentbench.benchkit.workspace import ensure_repo_cache, prepare_workspace
from tests.e2e.function.agentbench import conftest as smoke_fixtures
from tests.e2e.function.agentbench.smoke_inputs import prepare_smoke_case
from tests.e2e.helpers.case_loader import E2EConfig, load_case_file
from tests.helpers.marks import test_marks

pytestmark = test_marks(hardware="cpu", ci="ready")


def test_local_task_needs_no_remote_repository_or_dependencies(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    case = load_case_file(repo / "tests/e2e/function/agentbench/cases/claude/qwen3_coder_claude_1x1_single.json")
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


def test_model_service_reuses_configuration_and_stops_on_session_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo = Path(__file__).resolve().parents[2]
    case = load_case_file(repo / "tests/e2e/function/agentbench/cases/claude/qwen3_coder_claude_1x1_single.json")
    config = replace(E2EConfig.from_case(case), result_root=tmp_path)
    servers = []

    def make_server(*, config):
        server = Mock()
        server.config = config
        server.proc.poll.return_value = None
        servers.append(server)
        return server

    monkeypatch.setattr(smoke_fixtures, "VllmServer", make_server)
    session = smoke_fixtures.agentbench_server.__wrapped__()
    use = next(session)
    try:
        first = use(config)
        second = use(replace(config, port=config.port + 1, run_tag="another-task"))
        assert first.port == second.port
        assert len(servers) == 1
        servers[0].start.assert_called_once()
        servers[0].stop.assert_not_called()
        changed = replace(
            config, case=replace(case, deploy=replace(case.deploy, serve_args={**case.deploy.serve_args, "seed": 7}))
        )
        use(changed)
        assert len(servers) == 2
        servers[0].stop.assert_called_once()
    finally:
        session.close()
    servers[1].stop.assert_called_once()
