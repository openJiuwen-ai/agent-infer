# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Prepare the pinned Terminal-Bench task package without changing SWE-bench."""

import asyncio
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import time
import uuid
from contextlib import suppress
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict
from urllib.parse import urlparse

from tqdm import tqdm

from agentinfer.scheduling.headers import CLAUDE_SESSION_HEADER, DSH_SESSION_HEADER

from ..agents.contracts import AgentPreflightReport, AgentRunOutcome, AgentRunResult, TerminationReason
from .common import atomic_write_json
from .config import AgentBenchConfig
from .dataset import PreparedDataset, Task, load_tasks

if TYPE_CHECKING:
    from .runner import RunContext

DATASET_REF = "terminal-bench/terminal-bench@4.1.0"
DSH_VERSION = "0.2.1-rc.2"
_CLI_VERSION = re.compile(r"\b\d+\.\d+\.\d+\b")
_VALID_REWARD_EXCEPTIONS = frozenset(
    {
        "AgentTimeoutError",
        "ContextWindowExceededError",
        "OutputTokenExceededError",
        "ContextLengthExceededError",
        "OutputLengthExceededError",
        "AgentSafetyRefusalError",
        "ContentPolicyViolationError",
    }
)


def prepare_terminalbench(
    output_dir: Path,
    *,
    harbor_executable: str = "harbor",
    with_images: bool = False,
    task_nums: int | None = None,
) -> PreparedDataset:
    """Export task files or extend image preparation for an existing export."""

    from .terminalbench_images import prepare_images, read_task_config

    if task_nums is not None and (task_nums < 1 or not with_images):
        raise ValueError("task_nums must be positive and requires with_images")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        if with_images and output_dir.is_dir():
            prepared = _reuse_prepared_dataset(output_dir)
            prepare_images(output_dir, task_nums, harbor_executable=harbor_executable)
            return prepared
        raise FileExistsError(f"Output path already exists and is not empty: {output_dir}. Use a new --output-dir.")
    harbor = shutil.which(harbor_executable)
    if harbor is None:
        raise FileNotFoundError(f"Harbor executable not found: {harbor_executable}")
    tasks_dir = output_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    download = [harbor, "datasets", "download", DATASET_REF, "--output-dir", str(tasks_dir), "--export"]
    # Harbor reuses completed task directories on a retry. A transient Hub
    # timeout should not force users to download every completed task again.
    for attempt in range(3):
        completed = subprocess.run(download, check=False)
        if completed.returncode == 0:
            break
        if attempt == 2:
            raise RuntimeError(
                f"Harbor export failed after three attempts (exit {completed.returncode}). "
                f"Partial files remain in {tasks_dir}. Retry with a new --output-dir."
            )
    rows: list[dict[str, object]] = []
    used_ids: set[str] = set()
    for config_path in sorted(tasks_dir.rglob("task.toml")):
        task_dir = config_path.parent
        instruction = task_dir / "instruction.md"
        if not instruction.is_file():
            continue
        name = read_task_config(task_dir)["task"]["name"]
        content_hash = hashlib.sha256()
        for path in sorted(task_dir.rglob("*")):
            if not path.is_file() or ".git" in path.relative_to(task_dir).parts:
                continue
            content_hash.update(str(path.relative_to(task_dir)).encode("utf-8"))
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    content_hash.update(chunk)
        digest = content_hash.hexdigest()
        instance_id = re.sub(r"[^A-Za-z0-9_.-]", "__", name)
        if instance_id in used_ids:
            raise ValueError(f"Duplicate task identifier: {instance_id}")
        used_ids.add(instance_id)
        rows.append(
            {
                "instance_id": instance_id,
                "source_task_name": name,
                "task_path": str(task_dir.resolve()),
                "sha256": digest,
            }
        )
    if not rows:
        raise ValueError("Harbor export contained no usable Terminal-Bench tasks")
    lists_dir = output_dir / "task-lists"
    lists_dir.mkdir(exist_ok=True)
    index = output_dir / "instances.jsonl"
    index.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    selection = lists_dir / "all.txt"
    selection.write_text("\n".join(str(row["instance_id"]) for row in rows) + "\n", encoding="utf-8")
    manifest = output_dir / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset": "terminalbench",
                "source": DATASET_REF,
                "rows": len(rows),
                "selection_status": "all",
                "validation": {
                    "oracle": "not_run",
                    "agent": "not_run",
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if with_images:
        prepare_images(output_dir, task_nums, harbor_executable=harbor_executable)
    return PreparedDataset(index, selection, manifest, len(rows))


def _reuse_prepared_dataset(output_dir: Path) -> PreparedDataset:
    """Validate an existing export before preparing its images."""

    manifest = output_dir / "manifest.json"
    index = output_dir / "instances.jsonl"
    selection = output_dir / "task-lists" / "all.txt"
    if not all(path.is_file() for path in (manifest, index, selection)):
        raise FileExistsError(
            f"Output path already exists and is not empty: {output_dir}. "
            "Appending images requires a complete Terminal-Bench export. Retry with a new --output-dir."
        )
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    if (
        not isinstance(metadata, dict)
        or metadata.get("dataset") != "terminalbench"
        or metadata.get("source") != DATASET_REF
    ):
        raise ValueError(f"Expected prepared {DATASET_REF} data at {output_dir}")
    tasks = load_tasks(index, selection)
    if (
        not tasks
        or len(tasks) != metadata.get("rows")
        or any(
            task.task_path is None
            or not (task.task_path / "task.toml").is_file()
            or not (task.task_path / "instruction.md").is_file()
            for task in tasks
        )
    ):
        raise ValueError(f"Incomplete Terminal-Bench task export at {output_dir}")
    return PreparedDataset(index, selection, manifest, len(tasks))


async def _copy_stream(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    finally:
        try:
            writer.write_eof()
        except (OSError, RuntimeError):
            writer.close()


async def _forward_model_request(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    proxy_host: str,
    proxy_port: int,
    session_id: str,
    agent_type: str = "claude",
) -> None:
    try:
        upstream_reader, upstream_writer = await asyncio.open_connection(proxy_host, proxy_port)
    except OSError:
        writer.close()
        return
    try:
        headers = await reader.readuntil(b"\r\n\r\n")
        if agent_type != "jiuwenswarm":
            header = DSH_SESSION_HEADER if agent_type == "dsh" else CLAUDE_SESSION_HEADER
            # The HTTP server closes after this response, so each request gets a task identity.
            headers = re.sub(rb"(?im)^(?:connection|" + header.encode() + rb"):[^\r\n]*\r\n", b"", headers)
            headers = headers[:-2] + f"{header}: {session_id}\r\nConnection: close\r\n\r\n".encode()
        # Jiuwen supplies canonical task/agent lineage in the request body.
        upstream_writer.write(headers)
        await upstream_writer.drain()
        await asyncio.gather(_copy_stream(reader, upstream_writer), _copy_stream(upstream_reader, writer))
    except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError):
        pass
    finally:
        upstream_writer.close()
        writer.close()


def _command_output(command: list[str], label: str) -> str:
    """Bound host probes and retain useful diagnostics when they fail."""

    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"{label} preflight failed: {type(exc).__name__}: {exc}") from exc
    if result.returncode:
        detail = getattr(result, "stderr", "") or getattr(result, "stdout", "")
        raise RuntimeError(f"{label} preflight failed (exit {result.returncode}): {detail.strip()}")
    return result.stdout.strip()


def _executable_version(executable: Path, label: str) -> tuple[Path, str]:
    resolved = shutil.which(str(executable))
    if resolved is None:
        raise FileNotFoundError(f"{label} executable not found: {executable}")
    output = _command_output([resolved, "--version"], label)
    match = _CLI_VERSION.search(output)
    if match is None:
        raise ValueError(f"Cannot determine {label} version from {resolved}: {output}")
    return Path(resolved).resolve(), match.group()


def _validate_agent_support(agent_type: str, profile: str) -> None:
    if agent_type == "zcode":
        raise ValueError("Terminal-Bench does not yet support zcode: a Harbor agent adapter is required")
    if (agent_type, profile) not in {
        ("claude", "single"),
        ("dsh", "single"),
        ("dsh", "plan-subagent"),
        ("jiuwenswarm", "agent.code.normal"),
    }:
        raise ValueError(
            "Terminal-Bench supports claude/single, dsh/single or dsh/plan-subagent, and jiuwenswarm/agent.code.normal"
        )


def _docker_gateway() -> str:
    _command_output(["docker", "compose", "version"], "Docker Compose CLI plugin")
    gateway = _command_output(
        ["docker", "network", "inspect", "bridge", "--format", "{{(index .IPAM.Config 0).Gateway}}"],
        "Docker bridge",
    )
    if not gateway:
        raise RuntimeError("Docker bridge has no IPv4 gateway for the benchmark model proxy")
    return gateway


def validate_dsh_executable(executable: Path) -> tuple[Path, Path]:
    """Resolve the pinned host DSH package and compatible Node runtime."""

    resolved = shutil.which(str(executable))
    if resolved is None:
        raise FileNotFoundError(f"Agent executable not found: {executable}")
    package_json = Path(resolved).resolve().parent.parent / "package.json"
    if not package_json.is_file() or json.loads(package_json.read_text(encoding="utf-8")).get("version") != DSH_VERSION:
        raise ValueError(f"DSH executable must point to @deepseek-ai/dsh@{DSH_VERSION}: {executable}")
    node = shutil.which(os.environ.get("DSH_NODE_EXECUTABLE", "node"))
    if node is None:
        raise FileNotFoundError("Node.js not found; DSH requires Node.js >=22.19")
    version = _command_output([node, "--version"], "DSH Node.js")
    match = re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)", version)
    if match is None or (int(match[1]), int(match[2])) < (22, 19):
        raise ValueError(f"DSH requires Node.js >=22.19; found {version} at {node}")
    return package_json.parent, Path(node).resolve()


def validate_claude_executable(executable: Path, target: str) -> None:
    """Require an already installed native Linux CLI; do not fetch a runtime during run."""

    with executable.open("rb") as stream:
        header = stream.read(20)
    if len(header) < 20 or header[:6] != b"\x7fELF\x02\x01":
        raise ValueError(
            "Terminal-Bench offline Claude setup requires a local native Linux Claude binary; prepare it before run"
        )
    arch = {62: "amd64", 183: "arm64"}.get(int.from_bytes(header[18:20], "little"))
    if f"linux/{arch}" != target:
        raise ValueError(f"Claude binary architecture linux/{arch} does not match Docker {target}")


def check_terminalbench_preflight(config: AgentBenchConfig) -> AgentPreflightReport:
    """Check host prerequisites for the selected Harbor execution path."""

    agent = config.agent
    _validate_agent_support(agent.type, agent.profile)
    harbor, harbor_version = _executable_version(Path(os.environ.get("HARBOR_EXECUTABLE", "harbor")), "Harbor")
    gateway = _docker_gateway()
    from .terminalbench_images import docker_platform, native_platform

    target = docker_platform()
    versions = {"harbor": harbor_version}
    installation: dict[str, object] = {"harbor_executable": str(harbor)}
    if agent.type == "claude":
        executable, versions["claude"] = _executable_version(agent.executable, "Claude Code")
        validate_claude_executable(executable, target)
        installation["container_installation"] = "local Claude native binary; no network installation"
    elif agent.type == "jiuwenswarm":
        from .jiuwenswarm_runtime import validate_runtime

        runtime, manifest = validate_runtime(agent.executable)
        runtime_platform = native_platform(manifest["platform"], manifest["machine"])
        if runtime_platform != target:
            raise ValueError(
                f"JiuwenSwarm runtime {runtime_platform} does not match Docker {target}; prepare a matching runtime"
            )
        resolved = shutil.which(str(agent.executable))
        assert resolved is not None  # validate_runtime checked the same executable.
        executable = Path(resolved).resolve()
        versions.update(workswarm=manifest["workswarm"], openjiuwen=manifest["openjiuwen"])
        installation.update(runtime=str(runtime), sha256=manifest["sha256"], isolation="harbor-docker")
    else:
        package, node = validate_dsh_executable(agent.executable)
        node_platform = _command_output(
            [str(node), "-p", "process.platform + '/' + process.arch"], "DSH Node.js platform"
        )
        node_os, node_arch = node_platform.split("/", 1)
        node_target = native_platform(node_os, "amd64" if node_arch == "x64" else node_arch)
        if node_target != target:
            raise ValueError(
                f"DSH Node.js runtime {node_target} does not match Docker {target}; provide a matching runtime"
            )
        executable = (package / "lib" / "bin.js").resolve()
        output = _command_output([str(node), str(executable), "--version"], "DSH")
        version = re.search(r"\b\d+\.\d+\.\d+(?:-[\w.]+)?", output)
        if version is None or version.group() != DSH_VERSION:
            raise ValueError(f"Expected DSH {DSH_VERSION}, got {output}")
        versions.update(dsh=DSH_VERSION, node=_command_output([str(node), "--version"], "DSH Node.js"))
        installation.update(dsh_package=str(package), node_executable=str(node))
    return AgentPreflightReport(
        runtime=agent.type,
        executable=str(executable),
        versions=versions,
        capabilities={
            "execution_environment": "docker",
            "docker_platform": target,
            "profile": agent.profile,
            "docker_compose": True,
            "docker_bridge_gateway": gateway,
            "container_setup_verified": False,
        },
        installation=installation,
    )


_HARBOR_AGENTS = {"claude": "LocalClaudeCode", "dsh": "DshHarborAgent", "jiuwenswarm": "JiuwenHarborAgent"}
_CLAUDE_HARBOR_HOST_ENV = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "LANG",
        "LC_ALL",
        "TMPDIR",
        "XDG_CONFIG_HOME",
        "XDG_RUNTIME_DIR",
        "PYTHONPATH",
        "DOCKER_HOST",
        "DOCKER_CONTEXT",
        "DOCKER_CONFIG",
        "DOCKER_TLS_VERIFY",
        "DOCKER_CERT_PATH",
        "DOCKER_DEFAULT_PLATFORM",
        "COMPOSE_PROFILES",
    }
)


def _harbor_env(agent_type: str) -> dict[str, str]:
    """Give Claude's Harbor process its host environment and a local credential."""

    env = dict(os.environ)
    if agent_type == "claude":
        env = {name: value for name, value in env.items() if name in _CLAUDE_HARBOR_HOST_ENV}
        env.update(ANTHROPIC_API_KEY="agentbench", PYTHON_DOTENV_DISABLED="1")
    repo_root = str(Path(__file__).resolve().parents[3])
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (repo_root, env.get("PYTHONPATH"))))
    return env


def _harbor_command(context: "RunContext", task: Task, directory: Path, model_url: str, session_id: str) -> list[str]:
    agent = context.config.agent
    report = context.preflight
    assert report is not None
    installation = report.installation
    version = report.versions["workswarm" if agent.type == "jiuwenswarm" else agent.type]
    command = [
        str(installation["harbor_executable"]),
        "run",
        "--path",
        str(task.task_path),
        "--agent",
        f"agentinfer.agentbench.benchkit.harbor_{agent.type}:{_HARBOR_AGENTS[agent.type]}",
        "--model",
        context.config.backend.model,
        "--skip-model-parsing-and-provider-mapping",
        "--jobs-dir",
        str(directory / "harbor"),
        "--job-name",
        task.instance_id,
        "--n-concurrent",
        "1",
        "--max-retries",
        "0",
        "--env",
        "agentinfer.agentbench.benchkit.harbor_environment:PreparedDockerEnvironment",
    ]
    if not context.with_correctness:
        command.append("--disable-verification")
    options: dict[str, object] = {"version": version}
    timeout = context.config.experiment.task_timeout_seconds
    if agent.type == "claude":
        command += ["--api-format", "anthropic_messages", "--base-url", model_url]
        options["host_executable"] = report.executable
    else:
        options.update(profile=agent.profile, proxy_url=model_url, task_session_id=session_id)
        if agent.type == "jiuwenswarm":
            options.update(
                host_runtime=installation["runtime"],
                timeout_seconds=timeout,
                max_subagents=agent.max_subagents,
                max_concurrent_subagents=agent.max_concurrent_subagents,
            )
        else:
            options.update(
                host_dsh_package=installation["dsh_package"],
                host_node_executable=installation["node_executable"],
                context_window=agent.context_window,
                max_tokens=agent.max_tokens,
            )
    for key, value in options.items():
        command += ["--agent-kwarg", f"{key}={value}"]
    if timeout != 28800:
        command += ["--agent-timeout-multiplier", str(timeout / 28800)]
    return command


async def _run_harbor(
    context: "RunContext", task: Task, directory: Path, proxy_url: str, session_id: str
) -> tuple[int, dict[str, Any]]:
    assert context.preflight is not None
    gateway = context.preflight.capabilities["docker_bridge_gateway"]
    assert isinstance(gateway, str)
    agent_type = context.config.agent.type
    proxy = urlparse(proxy_url)
    server = await asyncio.start_server(
        lambda reader, writer: _forward_model_request(
            reader, writer, proxy.hostname or "localhost", proxy.port or 80, session_id, agent_type
        ),
        host=gateway,
        port=0,
    )
    async with server:
        port = server.sockets[0].getsockname()[1]
        command = _harbor_command(context, task, directory, f"http://{gateway}:{port}", session_id)
        atomic_write_json(directory / "harbor-command.json", command)
        with (
            (directory / "harbor.stdout.log").open("wb") as stdout,
            (directory / "harbor.stderr.log").open("wb") as stderr,
        ):
            process = await asyncio.create_subprocess_exec(
                *command, env=_harbor_env(agent_type), stdout=stdout, stderr=stderr, start_new_session=True
            )
            try:
                await process.wait()
            except asyncio.CancelledError:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    await asyncio.wait_for(process.wait(), 20)
                except asyncio.TimeoutError:
                    with suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                    await process.wait()
                raise
    assert process.returncode is not None
    return process.returncode, _read_harbor_result(directory, task)


def _read_harbor_result(directory: Path, task: Task) -> dict[str, Any]:
    """Load the single raw Harbor result, whose remaining fields we leave to Harbor."""

    reports = list((directory / "harbor" / task.instance_id).glob("*/result.json"))
    if len(reports) != 1:
        raise RuntimeError(f"Expected one Harbor trial result for {task.instance_id}, found {len(reports)}")
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError(f"Expected a Harbor trial result object for {task.instance_id}")
    return report


class _VerifierRecord(TypedDict):
    available: bool
    rewards: dict[str, int | float] | None
    exception: dict[str, Any] | None


def _verifier_record(trial: dict[str, Any], *, cancelled: bool = False) -> _VerifierRecord:
    exception = trial.get("exception_info")
    exception_type = exception.get("exception_type") if isinstance(exception, dict) else None
    verifier = trial.get("verifier_result")
    rewards = verifier.get("rewards") if isinstance(verifier, dict) else None
    valid = (
        isinstance(rewards, dict)
        and bool(rewards)
        and all(type(value) is int or (isinstance(value, float) and isfinite(value)) for value in rewards.values())
    )
    acceptable = exception is None or (
        isinstance(exception_type, str)
        and (exception_type in _VALID_REWARD_EXCEPTIONS or (cancelled and exception_type == "CancelledError"))
    )
    return {
        "available": bool(valid and acceptable),
        "rewards": rewards if valid else None,
        "exception": exception if isinstance(exception, dict) else None,
    }


async def _run_one_trial(
    context: "RunContext", task: Task, proxy_url: str, position: int, results: list[AgentRunResult]
) -> AgentRunResult:
    from .artifacts import build_task_result
    from .metrics.request import derive_session_topology

    started = time.monotonic()
    agent = context.config.agent
    result = AgentRunResult(
        agent.type,
        agent.profile,
        instance_id=task.instance_id,
        session_id=str(uuid.uuid4()),
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    directory = context.output_dir / "tasks" / task.instance_id
    verifier: _VerifierRecord = {"available": False, "rewards": None, "exception": None}
    error = None
    try:
        code, trial = await _run_harbor(context, task, directory, proxy_url, result.session_id)
        exception = trial.get("exception_info")
        exception_type = exception.get("exception_type") if isinstance(exception, dict) else None
        if context.with_correctness:
            verifier = _verifier_record(trial)
        if code == 0 and exception is None:
            result.outcome, result.termination_reason = AgentRunOutcome.COMPLETED, None
        elif exception_type == "AgentTimeoutError":
            result.termination_reason = TerminationReason.TIMEOUT
    except (Exception, asyncio.CancelledError) as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        if isinstance(exc, asyncio.CancelledError):
            result.termination_reason = TerminationReason.CANCELLED
            if context.with_correctness:
                with suppress(OSError, ValueError, RuntimeError):
                    verifier = _verifier_record(_read_harbor_result(directory, task), cancelled=True)
        verifier["exception"] = {"exception_type": error["type"], "exception_message": error["message"]}
        raise
    finally:
        result.duration_seconds = time.monotonic() - started
        result.finished_at = datetime.now(timezone.utc).isoformat()
        results.append(result)
        if context.with_correctness:
            atomic_write_json(directory / "verifier.json", verifier)
        topology = derive_session_topology([], result.session_id)
        artifact = build_task_result(result, topology, error, task_position=position)
        atomic_write_json(directory / "result.json", artifact.to_dict())
    return result


async def run_terminalbench_tasks(context: "RunContext", proxy_url: str, results: list[AgentRunResult]) -> None:
    semaphore = asyncio.Semaphore(context.config.experiment.max_concurrency)
    running: dict[str, str] = {}
    with tqdm(total=len(context.tasks), desc="Terminal-Bench", unit="task") as progress:

        async def bounded(task: Task, position: int) -> None:
            async with semaphore:
                running[task.instance_id] = task.instance_id.removeprefix("terminal-bench__")
                progress.set_postfix(running=", ".join(running.values()))
                try:
                    await _run_one_trial(context, task, proxy_url, position, results)
                finally:
                    del running[task.instance_id]
                    progress.set_postfix(running=", ".join(running.values()) or "-", refresh=False)
                    progress.update(1)

        outcomes = await asyncio.gather(
            *(bounded(task, position) for position, task in enumerate(context.tasks)), return_exceptions=True
        )
        for error in outcomes:
            if isinstance(error, BaseException):
                raise error


def write_terminalbench_correctness(context: "RunContext") -> None:
    trials = []
    for task in context.tasks:
        verifier = context.output_dir / "tasks" / task.instance_id / "verifier.json"
        row = (
            json.loads(verifier.read_text(encoding="utf-8"))
            if verifier.is_file()
            else {
                "available": False,
                "rewards": None,
                "exception": {"exception_type": "NotRun"},
            }
        )
        trials.append({"instance_id": task.instance_id, **row})
    scored = sum(trial["available"] for trial in trials)
    timeout = context.config.experiment.task_timeout_seconds
    atomic_write_json(
        context.output_dir / "correctness.json",
        {
            "dataset": DATASET_REF,
            "scope": "selected_tasks",
            "task_ids": [task.instance_id for task in context.tasks],
            "planned_trials": len(trials),
            "scored_trials": scored,
            "evaluation_coverage": scored / len(trials) if trials else None,
            "agent_timeout_seconds": timeout,
            "custom_timeout": timeout != 28800,
            "trials": trials,
        },
    )
