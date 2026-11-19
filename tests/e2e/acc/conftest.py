# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Options for real SWE-bench correctness evaluation."""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests.e2e.helpers.case_loader import BenchmarkCase, load_case_file

_CHECKOUT_OWNER = pytest.StashKey[tuple[Path, int, int]]()


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("correctness")
    group.addoption("--correctness-ci", action="store_true", help="Prepare the container's accuracy CI environment")
    group.addoption("--correctness-run-dir", help="Existing finalized SWE-bench run to evaluate")
    group.addoption("--correctness-evaluation-config", help="Evaluation YAML for benchmark runs or existing artifacts")
    group.addoption("--correctness-baseline", help="Baseline correctness.json for per-task comparison")
    group.addoption("--correctness-candidate", help="Candidate correctness.json for per-task comparison")


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    """Choose CI report paths before pytest initializes JUnit and logging."""
    if not config.getoption("--correctness-ci"):
        return
    agents = {path.name for path in (Path(__file__).parent / "cases").iterdir() if path.is_dir()}
    selected = config.getoption("keyword")
    logs = Path("nightly-logs") / (selected if selected in agents else "all")
    if not config.option.xmlpath:
        config.option.xmlpath = str(logs / "results.xml")
    if not config.option.log_file:
        config.option.log_file = str(logs / "pytest.log")


def _freeze(python: str | Path, path: Path) -> None:
    with path.open("w", encoding="utf-8") as output:
        subprocess.run([str(python), "-m", "pip", "freeze"], stdout=output, check=True)


def _jiuwen_python(executable: str) -> Path:
    resolved = shutil.which(executable)
    if resolved is None:
        raise RuntimeError(f"Jiuwen executable is unavailable: {executable}")
    return Path(resolved).resolve().with_name("python")


@pytest.fixture(scope="session")
def correctness_ci_environment(pytestconfig: pytest.Config):
    """Prepare CI once; regular/manual pytest runs do not change the host."""
    if not pytestconfig.getoption("--correctness-ci"):
        return None
    if os.name != "posix" or os.geteuid() != 0:
        raise pytest.UsageError("--correctness-ci requires root inside the accuracy container")
    user = pytestconfig.getoption("--benchmark-run-as-user")
    evaluation = pytestconfig.getoption("--correctness-evaluation-config")
    if not user or not evaluation:
        raise pytest.UsageError("--correctness-ci requires --benchmark-run-as-user and --correctness-evaluation-config")

    import grp

    from huggingface_hub import snapshot_download

    from agentinfer.agentbench.agents.registry import get_runtime
    from agentinfer.agentbench.benchkit.evaluation.config import load_evaluation_config

    checkout = Path(__file__).resolve().parents[3]
    owner = checkout.stat()
    pytestconfig.stash[_CHECKOUT_OWNER] = (checkout, owner.st_uid, owner.st_gid)
    if subprocess.run(["id", "-u", user], capture_output=True).returncode:
        subprocess.run(["useradd", "-m", "-s", "/bin/bash", user], check=True)
    docker_gid = Path("/var/run/docker.sock").stat().st_gid
    try:
        grp.getgrgid(docker_gid)
    except KeyError:
        subprocess.run(["groupadd", "-g", str(docker_gid), "accuracy-docker"], check=True)
    subprocess.run(["usermod", "-aG", str(docker_gid), user], check=True)
    sudo = ["sudo", "-u", user, "-E", "env", f"PATH={os.environ['PATH']}"]
    harness = load_evaluation_config(Path(evaluation)).harness_python
    subprocess.run(
        sudo
        + [
            str(harness),
            "-c",
            "import docker; from importlib.metadata import version; "
            "from swebench.harness.test_spec.test_spec import make_test_spec; "
            "assert version('swebench') == '4.1.0'; assert docker.from_env().ping()",
        ],
        check=True,
    )
    models: set[str] = set()
    agents: set[str] = set()

    def prepare(case: BenchmarkCase) -> None:
        agent = case.benchmark_params.get("agent-type", "claude")
        logs = checkout / "nightly-logs" / agent
        logs.mkdir(parents=True, exist_ok=True)
        if agent not in agents:
            if agent == "jiuwenswarm":
                probe = [
                    "--unshare-user",
                    "--unshare-pid",
                    "--ro-bind",
                    "/",
                    "/",
                    "--proc",
                    "/proc",
                    "--dev",
                    "/dev",
                    "/bin/true",
                ]
                subprocess.run(sudo + ["bwrap", *probe], check=True)
            asyncio.run(get_runtime(agent).preflight(Path(case.benchmark_params["agent-executable"])))
            if agent == "jiuwenswarm":
                python = _jiuwen_python(case.benchmark_params["agent-executable"])
                probe = textwrap.dedent("""\
                    import os
                    import tempfile
                    from importlib.metadata import version
                    with tempfile.TemporaryDirectory(prefix="jw-probe-") as home:
                        os.chdir(home)
                        os.environ.update(HOME=home, JIUWENSWARM_HOME=home, JIUWENSWARM_DATA_DIR=home + "/data")
                        from agentinfer.agentbench.agents.jiuwenswarm.extension import AgentBenchAffinityModelClient
                        assert version("jiuwenswarm") == "0.2.4b3"
                        print(AgentBenchAffinityModelClient.__client_name__)
                    """)
                subprocess.run(sudo + ["env", "-u", "PYTHONPATH", str(python), "-c", probe], check=True)
                _freeze(python, logs / "agent-requirements.txt")
            _freeze(sys.executable, logs / "bench-requirements.txt")
            _freeze(harness, logs / "evaluation-requirements.txt")
            agents.add(agent)
        model = case.deploy.model.as_posix()
        if model not in models:
            if not case.deploy.model.is_dir():
                snapshot_download(model, cache_dir=os.environ["HF_HUB_CACHE"], max_workers=4)
            models.add(model)

    return prepare


@pytest.fixture
def correctness_environment(case_file: Path, correctness_ci_environment) -> None:
    if correctness_ci_environment is not None:
        correctness_ci_environment(load_case_file(case_file))


def pytest_unconfigure(config: pytest.Config) -> None:
    # JUnit and logging finish before unconfigure; restore report/artifact owners too.
    owner = config.stash.get(_CHECKOUT_OWNER, None)
    if owner is not None:
        checkout, uid, gid = owner
        subprocess.run(["chown", "-R", f"{uid}:{gid}", str(checkout)], check=True)
