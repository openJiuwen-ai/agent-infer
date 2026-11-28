# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Run Workspace-Bench cases in task-local directories."""

import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from ...agents.contracts import AgentPreflightReport, AgentRunOutcome, AgentRunResult, TerminationReason
from ..common import atomic_write_json, utc_now
from ..config import AgentBenchConfig
from ..dataset import Task, load_tasks
from . import FILES_REVISION, IMAGE, REVISION, TASK_REVISION

WORKER = "agentinfer.agentbench.benchkit.workspacebench.worker"


def image_id() -> str:
    name = os.environ.get("AGENTBENCH_WORKSPACEBENCH_IMAGE", IMAGE)
    return subprocess.check_output(["docker", "image", "inspect", "-f", "{{.Id}}", name], text=True, timeout=15).strip()


def writable_directory(path: Path) -> None:
    path.mkdir(parents=True)
    if os.getuid() == 0:
        os.chown(path, 65534, 65534)


def container_args(name: str, *, jiuwen: bool = False) -> list[str]:
    uid, gid = (os.getuid(), os.getgid()) if os.getuid() else (65534, 65534)
    args = [
        "docker",
        "run",
        "--rm",
        "--init",
        "-i",
        "--name",
        name,
        "--network",
        "host",
        "--user",
        f"{uid}:{gid}",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--read-only",
        "--cpus",
        "2",
        "--memory",
        "8g",
        "--pids-limit",
        "512",
        "--tmpfs",
        "/tmp:rw,exec,mode=1777",
        "-e",
        "HOME=/tmp",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
    ]
    if jiuwen:
        # WorkSwarm's unprivileged bubblewrap needs nested user/mount namespaces.
        args.extend(
            [
                "--security-opt",
                "seccomp=unconfined",
                "--security-opt",
                "apparmor=unconfined",
                "--security-opt",
                "systempaths=unconfined",
            ]
        )
    return args


def claude_model_environment(model: str) -> list[str]:
    # The SDK makes auxiliary requests (e.g. Haiku), even in single-agent mode.
    flags = ["-e", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1"]
    for key in (
        "CLAUDE_CODE_SUBAGENT_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
    ):
        flags.extend(["-e", f"{key}={model}"])
    return flags


def preflight(config: AgentBenchConfig) -> AgentPreflightReport:
    import uuid

    agent = config.agent
    expected = "agent.code.normal" if agent.type == "jiuwenswarm" else "single"
    if agent.profile != expected:
        raise ValueError(f"Workspace-Bench supports {agent.type}/{expected}")
    if config.dataset.workspacebench_execution == "host":
        from .prepare import evaluation_path
        from .worker import probe

        if sys.platform != "linux" or agent.type != "jiuwenswarm":
            raise ValueError("Workspace-Bench host execution supports JiuwenSwarm on Linux")
        executable = shutil.which(agent.executable)
        if executable is None:
            raise FileNotFoundError(f"JiuwenSwarm executable not found: {agent.executable}")
        checkout = config.dataset.workspacebench_checkout or config.dataset.index_path.parent / "upstream"
        evaluation = evaluation_path(checkout)
        executable = str(Path(executable).absolute())
        if not (evaluation / "src" / "agent_runner.py").is_file():
            raise FileNotFoundError(f"Workspace-Bench task runner missing from {evaluation}")
        versions = probe({**agent.model_dump(mode="json"), "executable": executable})
        return AgentPreflightReport(
            agent.type,
            executable,
            versions,
            {"profile": expected, "execution_environment": "host"},
            {"evaluation_checkout": str(evaluation), "evaluation_revision": REVISION},
        )
    image = image_id()
    command = container_args(f"wsb-probe-{uuid.uuid4().hex}", jiuwen=agent.type == "jiuwenswarm")
    raw = subprocess.check_output(
        [*command, image, "python", "-m", WORKER, "probe"], input=agent.model_dump_json(), text=True, timeout=120
    )
    return AgentPreflightReport(
        agent.type,
        str(agent.executable),
        json.loads(raw.splitlines()[-1]),
        {"profile": expected, "execution_environment": "docker"},
        {"image_id": image, "evaluation_revision": REVISION},
    )


def select_tasks(config: AgentBenchConfig, result_dir: Path) -> tuple[Task, ...]:
    index = config.dataset.index_path
    provenance = json.loads((index.parent / "manifest.json").read_text())
    for key, value in {
        "dataset": "workspacebench",
        "evaluation_revision": REVISION,
        "task_revision": TASK_REVISION,
        "workspace_revision": FILES_REVISION,
    }.items():
        if provenance.get(key) != value:
            raise ValueError(f"Incompatible Workspace-Bench dataset manifest: {key}")
    tasks = load_tasks(index, config.dataset.selection_path, config.experiment.task_num, workspacebench=True)
    evidence = json.loads((result_dir / "evidence" / f"{config.agent.type}-runtime.json").read_text())
    planned = []
    for task in tasks:
        assert task.metadata_path and task.standard_path
        meta = json.loads(task.metadata_path.read_text())
        lang, number = meta["language"], int(meta["absolute_id"])
        if task.instance_id != f"wsb-{lang}-{number:04d}" or not task.standard_path.is_dir():
            raise ValueError(f"Inconsistent prepared Workspace-Bench case {task.instance_id}")
        planned.append(
            {
                "instance_id": task.instance_id,
                "language": lang,
                "metadata_path": str(task.metadata_path.resolve()),
                "absolute_id": number,
            }
        )
    installation = evidence["installation"]
    execution_environment = config.dataset.workspacebench_execution
    execution_key = "evaluation_checkout" if execution_environment == "host" else "image_id"
    atomic_write_json(
        result_dir / "workspacebench.json",
        {
            "source": provenance,
            "execution_environment": execution_environment,
            execution_key: installation[execution_key],
            "tasks": planned,
        },
    )
    return tuple(tasks)


async def run_case(config: AgentBenchConfig, task: Task, proxy: str, output: Path, session: str) -> AgentRunResult:
    assert task.metadata_path and task.standard_path
    meta = json.loads(task.metadata_path.read_text())
    atomic_write_json(output / "metadata.json", {**meta, "__metadata_path": "/inputs/metadata.json"})
    case = output / "official"
    host = config.dataset.workspacebench_execution == "host"
    name = f"agentbench-wsb-{session}"
    agent = config.agent
    # Send the instruction and output contract only. Full metadata and reference inputs stay on the host.
    public = {key: meta[key] for key in ("task", "language", "file_system", "output_files")}
    public["id"] = str(meta["absolute_id"])
    payload = {
        "agent": agent.model_dump(mode="json"),
        "task": public,
        "model": config.backend.model,
        "proxy": proxy,
        "session_id": session,
        "timeout": config.experiment.task_timeout_seconds,
        "prompt_timeout": config.experiment.prompt_delivery_timeout_seconds,
    }
    manifest = json.loads((output.parents[1] / "workspacebench.json").read_text())
    if host:
        case.mkdir()
        evidence = json.loads((output.parents[1] / "evidence/jiuwenswarm-runtime.json").read_text())
        payload["agent"]["executable"] = evidence["executable"]
        command = [
            sys.executable,
            "-m",
            WORKER,
            "run-host",
            manifest["evaluation_checkout"],
            str(case.resolve()),
            str(task.standard_path.resolve()),
        ]
    else:
        writable_directory(case)
        command = [
            *container_args(name, jiuwen=agent.type == "jiuwenswarm"),
            *(claude_model_environment(config.backend.model) if agent.type == "claude" else []),
            "-v",
            f"{task.standard_path.resolve()}:/standard:ro",
            "-v",
            f"{case.resolve()}:/cases:rw",
            manifest["image_id"],
            "python",
            "-m",
            WORKER,
            "run",
        ]
    artifact = case / public["id"]
    started_at, started = utc_now().isoformat(), time.monotonic()
    timed_out = False
    with (output / "worker.log").open("wb") as log:
        environment = None
        if host:
            environment = {
                key: value
                for key, value in os.environ.items()
                if key in {"PATH", "LANG", "LANGUAGE", "LC_ALL", "LC_CTYPE", "PYTHONPATH"}
            }
        proc = await asyncio.create_subprocess_exec(
            *command, stdin=asyncio.subprocess.PIPE, stdout=log, stderr=log, env=environment, start_new_session=host
        )
        try:
            await asyncio.wait_for(
                proc.communicate(json.dumps(payload).encode()),
                config.experiment.task_timeout_seconds + config.experiment.cleanup_timeout_seconds,
            )
        except asyncio.TimeoutError:
            timed_out = True
        finally:
            if proc.returncode is None and host:
                await _stop_host_worker(proc, config.experiment.cleanup_timeout_seconds)
            elif proc.returncode is None:
                # Stop the client before removing the named container. The CLI
                # can forward SIGTERM to its child and keep waiting indefinitely.
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                await proc.wait()
                await asyncio.to_thread(
                    subprocess.run, ["docker", "rm", "-f", name], stdout=log, stderr=log, timeout=30, check=False
                )
            if host:
                workdir = artifact / "workdir"
                if workdir.is_dir() and not workdir.is_symlink():
                    await asyncio.to_thread(shutil.rmtree, workdir)
    if timed_out:
        return AgentRunResult(
            agent.type,
            agent.profile,
            termination_reason=TerminationReason.TIMEOUT,
            instance_id=task.instance_id,
            session_id=session,
            started_at=started_at,
            finished_at=utc_now().isoformat(),
            duration_seconds=time.monotonic() - started,
        )
    if proc.returncode:
        raise RuntimeError(f"Workspace-Bench worker exited {proc.returncode}. See {output / 'worker.log'}")
    execution = json.loads((artifact / "agent.json").read_text())
    status = execution["runnerStatus"]
    reason = None
    if status != "ok":
        try:
            reason = TerminationReason(execution.get("errorMessage"))
        except ValueError:
            reason = TerminationReason.TIMEOUT if status == "timeout" else TerminationReason.HARNESS_ERROR
    return AgentRunResult(
        agent.type,
        agent.profile,
        outcome=AgentRunOutcome.FAILED if reason else AgentRunOutcome.COMPLETED,
        termination_reason=reason,
        instance_id=task.instance_id,
        session_id=session,
        started_at=started_at,
        finished_at=utc_now().isoformat(),
        duration_seconds=time.monotonic() - started,
    )


async def _stop_host_worker(process: asyncio.subprocess.Process, grace_seconds: float) -> None:
    """Interrupt the official runner so the Jiuwen runtime can stop its services."""
    import psutil

    try:
        children = psutil.Process(process.pid).children(recursive=True)
        os.killpg(process.pid, signal.SIGINT)
    except (ProcessLookupError, psutil.NoSuchProcess):
        await process.wait()
        return
    try:
        await asyncio.wait_for(process.wait(), grace_seconds)
    except asyncio.TimeoutError:
        for child in reversed(children):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()
