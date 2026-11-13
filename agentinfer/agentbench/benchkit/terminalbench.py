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
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from ..agents.contracts import AgentRunOutcome, AgentRunResult, TerminationReason
from .common import atomic_write_json
from .dataset import PreparedDataset

DATASET_REF = "terminal-bench/terminal-bench@4.0.0"
DSH_VERSION = "0.2.1-rc.2"
_TASK_NAME = re.compile(r'^name\s*=\s*"([^"]+)"', re.MULTILINE)
_GPU_COUNT = re.compile(r"^gpus\s*=\s*(\d+)", re.MULTILINE)
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


def prepare_terminalbench(output_dir: Path, *, harbor_executable: str = "harbor") -> PreparedDataset:
    """Export the published tasks and index a deterministic CPU-only candidate set."""

    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Output path already exists and is not empty: {output_dir}")
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
            raise subprocess.CalledProcessError(completed.returncode, download)
    rows: list[dict[str, object]] = []
    selected: list[str] = []
    used_ids: set[str] = set()
    for config_path in sorted(tasks_dir.rglob("task.toml")):
        task_dir = config_path.parent
        instruction = task_dir / "instruction.md"
        if not instruction.is_file():
            continue
        match = _TASK_NAME.search(config_path.read_text(encoding="utf-8"))
        if match is None:
            raise ValueError(f"Missing task name: {config_path}")
        name = match.group(1)
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
        config_text = config_path.read_text(encoding="utf-8")
        gpu_counts = [int(value) for value in _GPU_COUNT.findall(config_text)]
        # Harbor treats an omitted GPU count as zero.
        cpu_candidate = (
            all(count == 0 for count in gpu_counts)
            and not (task_dir / "environment" / "docker-compose.yaml").exists()
            and not (task_dir / "environment" / "docker-compose.yml").exists()
            and not (task_dir / "steps").exists()
        )
        rows.append(
            {
                "instance_id": instance_id,
                "source_task_name": name,
                "task_path": str(task_dir.resolve()),
                "sha256": digest,
                "cpu_candidate": cpu_candidate,
            }
        )
        if cpu_candidate:
            selected.append(instance_id)
    if not rows or not selected:
        raise ValueError("Harbor export contained no usable Terminal-Bench CPU tasks")
    lists_dir = output_dir / "task-lists"
    lists_dir.mkdir(exist_ok=True)
    index = output_dir / "instances.jsonl"
    index.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (lists_dir / "all.txt").write_text("\n".join(str(row["instance_id"]) for row in rows) + "\n", encoding="utf-8")
    selection = lists_dir / "default.txt"
    selection.write_text("\n".join(selected) + "\n", encoding="utf-8")
    manifest = output_dir / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset": "terminalbench",
                "source": DATASET_REF,
                "rows": len(rows),
                "cpu_candidates": len(selected),
                "selection_status": "candidate; requires oracle and agent validation before benchmark claims",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return PreparedDataset(index, selection, manifest, len(rows))


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
        if len(headers) > 65536:
            raise ValueError("Model request headers exceeded 64 KiB")
        header_name = "X-DeepSeek-Harness-Session-Id" if agent_type == "dsh" else "X-Claude-Code-Session-Id"
        identity = f"{header_name}: {session_id}\r\n".encode()
        headers = re.sub(rb"(?im)^" + header_name.encode() + rb":[^\r\n]*\r\n", b"", headers)
        upstream_writer.write(headers[:-2] + identity + b"\r\n")
        await upstream_writer.drain()
        try:
            await asyncio.gather(_copy_stream(reader, upstream_writer), _copy_stream(upstream_reader, writer))
        except (ConnectionResetError, BrokenPipeError):
            # A client may close after receiving its response while the other
            # direction is still draining. The proxy has already recorded it.
            pass
    finally:
        upstream_writer.close()
        writer.close()


def _docker_gateway() -> str:
    compose = subprocess.run(["docker", "compose", "version"], capture_output=True, text=True, check=False)
    if compose.returncode:
        raise RuntimeError("Harbor Docker runs require the Docker Compose CLI plugin")
    result = subprocess.run(
        ["docker", "network", "inspect", "bridge", "--format", "{{(index .IPAM.Config 0).Gateway}}"],
        capture_output=True,
        text=True,
        check=True,
    )
    gateway = result.stdout.strip()
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
    version = subprocess.run([node, "--version"], capture_output=True, text=True, check=True).stdout.strip()
    match = re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)", version)
    if match is None or (int(match[1]), int(match[2])) < (22, 19):
        raise ValueError(f"DSH requires Node.js >=22.19; found {version} at {node}")
    return package_json.parent, Path(node).resolve()


async def _run_one_trial(context, task, proxy_url: str, task_position: int) -> AgentRunResult:
    from ..request_proxy import load_request_facts
    from .artifacts import build_task_result
    from .metrics.request import derive_session_topology

    if task.task_path is None or not task.task_path.is_dir():
        raise FileNotFoundError(f"Terminal-Bench task directory unavailable: {task.instance_id}")
    harbor_command = os.environ.get("HARBOR_EXECUTABLE", "harbor")
    harbor = shutil.which(harbor_command)
    if harbor is None:
        raise FileNotFoundError(f"Harbor executable not found: {harbor_command}")
    agent_executable = shutil.which(str(context.config.agent.executable))
    if agent_executable is None:
        raise FileNotFoundError(f"Agent executable not found: {context.config.agent.executable}")
    agent_type = context.config.agent.type
    if agent_type == "claude":
        version_output = subprocess.run(
            [agent_executable, "--version"], capture_output=True, text=True, check=True
        ).stdout
        version_match = _CLI_VERSION.search(version_output)
        if version_match is None:
            raise ValueError(f"Cannot determine Claude Code version from {agent_executable}")
        agent_version = version_match.group()
    else:
        host_dsh_package, host_node_executable = validate_dsh_executable(Path(agent_executable))
        agent_version = DSH_VERSION
    artifact_dir = context.output_dir / "tasks" / task.instance_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    session_id = str(uuid.uuid4())
    start = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    parsed = urlparse(proxy_url)
    gateway = _docker_gateway()
    server = await asyncio.start_server(
        lambda reader, writer: _forward_model_request(
            reader, writer, parsed.hostname or "127.0.0.1", parsed.port or 80, session_id, agent_type
        ),
        host=gateway,
        port=0,
    )
    port = server.sockets[0].getsockname()[1]
    command = [
        harbor,
        "run",
        "--path",
        str(task.task_path),
        "--agent",
        "claude-code" if agent_type == "claude" else "agentinfer.agentbench.benchkit.harbor_dsh:DshHarborAgent",
        "--agent-kwarg",
        f"version={agent_version}",
        "--model",
        context.config.backend.model,
        "--skip-model-parsing-and-provider-mapping",
        "--jobs-dir",
        str(artifact_dir / "harbor"),
        "--job-name",
        task.instance_id,
        "--n-concurrent",
        "1",
        "--max-retries",
        "0",
    ]
    if agent_type == "claude":
        command.extend(["--api-format", "anthropic_messages", "--base-url", f"http://{gateway}:{port}"])
    else:
        command.extend(
            [
                "--agent-kwarg",
                f"profile={context.config.agent.profile}",
                "--agent-kwarg",
                f"proxy_url=http://{gateway}:{port}",
                "--agent-kwarg",
                f"task_session_id={session_id}",
                "--agent-kwarg",
                f"host_dsh_package={host_dsh_package}",
                "--agent-kwarg",
                f"host_node_executable={host_node_executable}",
                "--agent-kwarg",
                f"context_window={context.config.agent.context_window}",
                "--agent-kwarg",
                f"max_tokens={context.config.agent.max_tokens}",
            ]
        )
    if context.config.experiment.task_timeout_seconds != 28800:
        command.extend(["--agent-timeout-multiplier", str(context.config.experiment.task_timeout_seconds / 28800)])
    atomic_write_json(artifact_dir / "harbor-command.json", command)
    env = dict(os.environ)
    if agent_type == "claude":
        env.setdefault("ANTHROPIC_API_KEY", "agentbench")
    else:
        repo_root = str(Path(__file__).resolve().parents[3])
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (repo_root, env.get("PYTHONPATH"))))
    try:
        with (
            (artifact_dir / "harbor.stdout.log").open("wb") as stdout,
            (artifact_dir / "harbor.stderr.log").open("wb") as stderr,
        ):
            process = await asyncio.create_subprocess_exec(
                *command,
                env=env,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            try:
                await process.wait()
            except asyncio.CancelledError:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    await asyncio.wait_for(process.wait(), 20)
                except asyncio.TimeoutError:
                    os.killpg(process.pid, signal.SIGKILL)
                    await process.wait()
                raise
    finally:
        server.close()
        await server.wait_closed()
    trials = []
    for path in (artifact_dir / "harbor").rglob("result.json"):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(raw, dict) and "verifier_result" in raw and "task_name" in raw:
            trials.append(raw)
    if len(trials) != 1:
        raise RuntimeError(f"Expected one Harbor trial result for {task.instance_id}, found {len(trials)}")
    trial = trials[0]
    verifier = trial.get("verifier_result") or {}
    rewards = verifier.get("rewards") if isinstance(verifier, dict) else None
    exception = trial.get("exception_info")
    exception_type = exception.get("exception_type") if isinstance(exception, dict) else None
    valid_reward = (
        isinstance(rewards, dict)
        and bool(rewards)
        and (exception is None or exception_type in _VALID_REWARD_EXCEPTIONS)
    )
    atomic_write_json(
        artifact_dir / "verifier.json",
        {"available": valid_reward, "rewards": rewards, "exception": exception},
    )
    success = process.returncode == 0 and exception is None
    result = AgentRunResult(
        agent_type=context.config.agent.type,
        profile_name=context.config.agent.profile,
        outcome=AgentRunOutcome.COMPLETED if success else AgentRunOutcome.FAILED,
        termination_reason=(
            None
            if success
            else TerminationReason.TIMEOUT
            if exception_type == "AgentTimeoutError"
            else TerminationReason.HARNESS_ERROR
        ),
        started_at=started_at,
        finished_at=datetime.now(timezone.utc).isoformat(),
        duration_seconds=time.monotonic() - start,
        session_id=session_id,
        instance_id=task.instance_id,
    )
    topology = derive_session_topology(load_request_facts(context.trace_path, session_id=session_id), session_id)
    atomic_write_json(
        artifact_dir / "result.json", build_task_result(result, topology, task_position=task_position).to_dict()
    )
    return result


async def run_terminalbench_tasks(context, proxy_url: str, results: list[AgentRunResult]) -> None:
    """Run the supported Harbor agent path and import its raw reward."""

    agent = context.config.agent
    if (agent.type, agent.profile) not in {
        ("claude", "single"),
        ("dsh", "single"),
        ("dsh", "plan-subagent"),
    }:
        raise ValueError("Terminal-Bench supports claude/single and dsh/single or dsh/plan-subagent")
    unsupported = [task.instance_id for task in context.tasks if not task.cpu_candidate]
    if unsupported:
        raise ValueError("Selected tasks are outside the supported CPU candidate subset: " + ", ".join(unsupported))
    semaphore = asyncio.Semaphore(context.config.experiment.max_concurrency)

    async def bounded(task, position):
        async with semaphore:
            result = await _run_one_trial(context, task, proxy_url, position)
            results.append(result)

    await asyncio.gather(*(bounded(task, position) for position, task in enumerate(context.tasks)))
    trials = []
    for task in context.tasks:
        path = context.output_dir / "tasks" / task.instance_id / "verifier.json"
        trials.append({"instance_id": task.instance_id, **json.loads(path.read_text(encoding="utf-8"))})
    scored = sum(trial["available"] for trial in trials)
    atomic_write_json(
        context.output_dir / "correctness.json",
        {
            "dataset": DATASET_REF,
            "scope": "cpu_candidate_subset",
            "task_ids": [task.instance_id for task in context.tasks],
            "planned_trials": len(trials),
            "scored_trials": scored,
            "evaluation_coverage": scored / len(trials) if trials else None,
            "agent_timeout_seconds": context.config.experiment.task_timeout_seconds,
            "custom_timeout": context.config.experiment.task_timeout_seconds != 28800,
            "trials": trials,
        },
    )
