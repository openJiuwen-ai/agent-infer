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
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

import pytest

from tests.e2e.helpers.case_loader import BenchmarkCase, E2EConfig, load_case_group, resolve_benchmark_run_as_user
from tests.e2e.helpers.server import managed_vllm

from .output import output_dir_name

_CHECKOUT_OWNER = pytest.StashKey[tuple[Path, int, int]]()
_ARTIFACT_ROOT = pytest.StashKey[Path]()


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("correctness")
    group.addoption("--correctness-ci", action="store_true", help="Prepare the container's accuracy CI environment")
    group.addoption("--correctness-run-dir", help="Existing finalized SWE-bench run to evaluate")
    group.addoption("--correctness-evaluation-config", help="Evaluation YAML for benchmark runs or existing artifacts")
    group.addoption("--correctness-baseline", help="Baseline correctness.json for per-task comparison")
    group.addoption("--correctness-candidate", help="Candidate correctness.json for per-task comparison")


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    """Default pytest logs to this invocation's artifacts; leave explicit paths intact."""
    if config.getoption("collectonly"):
        return
    artifacts = _artifact_root(config)
    if not config.option.log_file:
        config.option.log_file = str(artifacts / "pytest.log")


def _artifact_root(config: pytest.Config) -> Path:
    """Return one feature/timestamp directory shared by this pytest invocation."""
    existing = config.stash.get(_ARTIFACT_ROOT, None)
    if existing is not None:
        return existing
    selected_file = config.getoption("--test-config-file")
    paths = [Path(selected_file)] if selected_file else sorted((Path(__file__).parent / "cases").glob("*.json"))
    cases = [case for path in paths for case in load_case_group(path)]
    agents = {case.benchmark_params.get("agent-type", "claude") for case in cases}
    selected = config.getoption("keyword")
    feature = selected if selected in agents | {"baseline", "agentinfer"} else "all"
    if feature == "all" and selected_file:
        feature = cases[0].scenario.value
    bases = {case.deploy.result_root for case in cases} or {Path("tests/run/acc")}
    if len(bases) != 1:
        raise pytest.UsageError("accuracy cases in one invocation must share result_root")
    base = bases.pop().expanduser()
    if not base.is_absolute():
        base = Path(__file__).resolve().parents[3] / base
    artifacts = base.resolve() / output_dir_name(feature)
    config.stash[_ARTIFACT_ROOT] = artifacts
    return artifacts


@pytest.fixture(scope="session")
def correctness_artifacts(pytestconfig: pytest.Config) -> Path:
    """Share the output root across agent cases and their server/pytest logs."""
    artifacts = _artifact_root(pytestconfig)
    print(f"[correctness] artifacts={artifacts}", flush=True)
    return artifacts


@pytest.fixture(scope="session")
def correctness_results() -> Path:
    """Return the score-summary base directory paired with the accuracy run artifacts."""
    return Path(__file__).resolve().parents[3] / "tests/result/acc"


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
    return _prepare_correctness_ci_environment(pytestconfig)


def _prepare_correctness_ci_environment(pytestconfig: pytest.Config):
    """Return a per-agent preparation callback only when CI setup is requested."""
    if not pytestconfig.getoption("--correctness-ci"):
        return None
    if os.name != "posix" or os.geteuid() != 0:
        raise pytest.UsageError("--correctness-ci requires root inside the accuracy container")
    user = pytestconfig.getoption("--benchmark-run-as-user")
    evaluation = pytestconfig.getoption("--correctness-evaluation-config")
    if not evaluation:
        raise pytest.UsageError("--correctness-ci requires --correctness-evaluation-config")

    import grp

    from huggingface_hub import snapshot_download

    from agentinfer.agentbench.agents.registry import get_runtime
    from agentinfer.agentbench.benchkit.evaluation.config import load_evaluation_config

    checkout = Path(__file__).resolve().parents[3]
    owner = checkout.stat()
    pytestconfig.stash[_CHECKOUT_OWNER] = (checkout, owner.st_uid, owner.st_gid)
    sudo = []
    if user:
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
        profile = case.benchmark_params.get("agent-profile", "plan-subagent")
        if profile == "plan-subagent" and not user:
            raise pytest.UsageError("--correctness-ci requires --benchmark-run-as-user for plan-subagent")
        probe_prefix = sudo if resolve_benchmark_run_as_user(agent_profile=profile, benchmark_run_as_user=user) else []
        logs = _artifact_root(pytestconfig) / agent
        logs.mkdir(parents=True, exist_ok=True)
        if agent not in agents:
            if agent == "jiuwenswarm":
                namespace_probe = [
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
                subprocess.run(probe_prefix + ["bwrap", *namespace_probe], check=True)
            asyncio.run(get_runtime(agent).preflight(Path(case.benchmark_params["agent-executable"])))
            if agent == "jiuwenswarm":
                python = _jiuwen_python(case.benchmark_params["agent-executable"])
                client_probe_script = textwrap.dedent("""\
                    import os
                    import subprocess
                    import sys
                    import tempfile
                    from importlib.metadata import version
                    from pathlib import Path
                    assert version("workswarm") == "0.2.8"
                    assert version("openjiuwen") == "0.1.20"
                    with tempfile.TemporaryDirectory(prefix="jw-probe-") as home:
                        os.chdir(home)
                        os.environ.update(
                            HOME=home,
                            JIUWENSWARM_HOME=home,
                            JIUWENSWARM_DATA_DIR=home + "/data",
                            AGENTBENCH_ROOT_SESSION_ID="preflight",
                            AGENTBENCH_MAX_SUBAGENTS="1",
                            AGENTBENCH_MAX_CONCURRENT_SUBAGENTS="1",
                            AGENTBENCH_LIMITS_LOG=home + "/limits.jsonl",
                        )
                        os.environ.pop("AGENTBENCH_JIUWEN_RUNTIME_DIR", None)
                        subprocess.run(
                            [str(Path(sys.executable).with_name("jiuwenswarm-init"))],
                            stdin=subprocess.DEVNULL, check=True, timeout=120,
                        )
                        from agentinfer.agentbench.agents.jiuwenswarm.extension import AgentBenchAffinityModelClient
                        print(AgentBenchAffinityModelClient.__client_name__)
                    """)
                subprocess.run(
                    probe_prefix + ["env", "-u", "PYTHONPATH", str(python), "-c", client_probe_script], check=True
                )
                _freeze(python, logs / "agent-requirements.txt")
            _freeze(sys.executable, logs / "bench-requirements.txt")
            _freeze(harness, logs / "evaluation-requirements.txt")
            agents.add(agent)
        model = case.deploy.model.as_posix()
        if model not in models:
            if not case.deploy.model.is_dir():
                snapshot_download(model, cache_dir=os.environ.get("HF_HUB_CACHE"), max_workers=4)
            models.add(model)

    return prepare


@pytest.fixture
def correctness_environment(
    benchmark_case: BenchmarkCase, correctness_ci_environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    if benchmark_case.benchmark_params.get("agent-type", "claude") == "claude":
        context = benchmark_case.deploy.serve_args["max-model-len"]
        monkeypatch.setenv("CLAUDE_CODE_MAX_CONTEXT_TOKENS", str(context))
        monkeypatch.setenv("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "8192")
    if correctness_ci_environment is not None:
        correctness_ci_environment(benchmark_case)


@pytest.fixture(scope="module")
def correctness_server(case_file: Path):
    """Start lazily after per-agent preflight and reuse serve within one case file."""
    server_config: E2EConfig | None = None
    with ExitStack() as stack:

        def use(config: E2EConfig) -> E2EConfig:
            nonlocal server_config
            blockers = config.validate_prerequisites()
            if blockers:
                raise RuntimeError("; ".join(blockers))
            if server_config is None:
                stack.enter_context(managed_vllm(config))
                server_config = config
            return replace(
                config,
                host=server_config.host,
                port=server_config.port,
                lifecycle_socket=server_config.lifecycle_socket,
            )

        yield use


def pytest_unconfigure(config: pytest.Config) -> None:
    # JUnit and logging finish before unconfigure; restore report/artifact owners too.
    owner = config.stash.get(_CHECKOUT_OWNER, None)
    if owner is not None:
        checkout, uid, gid = owner
        subprocess.run(["chown", "-R", f"{uid}:{gid}", str(checkout)], check=True)
