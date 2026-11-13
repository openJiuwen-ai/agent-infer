# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Run ZCode through its CLI, preserving task evidence on cancellation."""

import asyncio
import json
import os
import signal
import time
import traceback
from datetime import datetime, timezone

from ...benchkit.workspace import WorkspaceProcessOwner, export_patch
from ..contracts import (
    AgentRunOutcome,
    AgentRunRequest,
    AgentRunResult,
    TerminationReason,
)
from .bridge import ZCodeBridge
from .instance import bootstrap
from .profiles import explore_name, get_profile


async def run_zcode(request: AgentRunRequest) -> AgentRunResult:
    started = time.monotonic()
    result = AgentRunResult(
        agent_type="zcode",
        profile_name=request.profile_name,
        instance_id=request.task.instance_id,
        session_id=request.session_id,
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    request.artifact_dir.mkdir(parents=True, exist_ok=True)
    profile = get_profile(request.profile_name)
    prompt = profile.build_prompt(
        request.task,
        explore=explore_name(request.session_id),
        max_subagents=request.max_subagents,
        max_concurrent_subagents=request.max_concurrent_subagents,
    )
    prompt_path = (request.artifact_dir / "prompt.txt").resolve()
    prompt_path.write_text(prompt, encoding="utf-8")
    transcript = request.artifact_dir / "transcript.jsonl"
    result.transcript = str(transcript)
    bridge = ZCodeBridge(request)
    cancellation = None
    return_code = None
    native_error = None
    process = None
    try:
        async with bridge.serve() as url:
            environment = bootstrap(request, url)
            command = [
                str(request.executable),
                "--prompt",
                str(prompt_path),
                "--cwd",
                str(request.workspace),
                "--mode",
                "plan" if profile.planning else "yolo",
                "--workflow-mode",
                "disabled",
                "--disallowed-tools",
                profile.disallowed_tools,
                "--output-format",
                "stream-json",
                "--no-color",
            ]
            (request.artifact_dir / "zcode-command.json").write_text(json.dumps(command, indent=2) + "\n")
            with transcript.open("wb") as stdout, (request.artifact_dir / "zcode.stderr.log").open("wb") as stderr:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    cwd=request.workspace,
                    env=environment,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                )
                try:
                    return_code = await asyncio.wait_for(
                        process.wait(), request.timeout_seconds - (time.monotonic() - started)
                    )
                finally:
                    await _stop_process_group(process, request.cleanup_timeout_seconds)
        events = [json.loads(line) for line in transcript.read_text().splitlines() if line.strip()]
        completed = next((row for row in reversed(events) if row["type"] == "result"), None)
        native_error = next(
            (
                row["payload"]["error"]
                for row in reversed(events)
                if row["type"] == "turn.failed" and not row["sessionId"].startswith("sess_subagent_")
            ),
            None,
        )
        if native_error:
            result.termination_reason = (
                TerminationReason.MODEL_CONTEXT_EXCEEDED
                if native_error.get("code") == "MODEL_CONTEXT_EXCEEDED"
                else TerminationReason.HARNESS_ERROR
            )
        elif bridge.failure:
            result.termination_reason = bridge.failure
        elif return_code != 0:
            result.termination_reason = TerminationReason.HARNESS_ERROR
        elif profile.planning and (not bridge.approved or not bridge.reports or bridge.pending):
            result.termination_reason = TerminationReason.PLAN_NOT_APPROVED
        elif return_code == 0 and completed and completed["projection"]["status"] == "idle":
            result.outcome = AgentRunOutcome.COMPLETED
            result.termination_reason = None
    except asyncio.CancelledError as exc:
        cancellation = exc
        result.termination_reason = TerminationReason.CANCELLED
    except asyncio.TimeoutError:
        result.termination_reason = TerminationReason.TIMEOUT
    except (FileNotFoundError, PermissionError):
        result.termination_reason = TerminationReason.AGENT_STARTUP_FAILED
    except Exception:
        (request.artifact_dir / "zcode-error.log").write_text(traceback.format_exc())
        result.termination_reason = TerminationReason.HARNESS_ERROR

    result.auto_plan_approvals = int(bridge.approved)
    owner = WorkspaceProcessOwner(time.monotonic() + request.cleanup_timeout_seconds)
    try:
        patch = await asyncio.to_thread(export_patch, request.workspace, owner)
        if patch:
            (request.artifact_dir / "model.patch").write_text(patch)
            result.has_patch = True
            result.patch_bytes = len(patch.encode())
    except Exception:
        (request.artifact_dir / "zcode-patch-error.log").write_text(traceback.format_exc())
        if result.outcome is AgentRunOutcome.COMPLETED:
            result.outcome = AgentRunOutcome.FAILED
            result.termination_reason = TerminationReason.HARNESS_ERROR
    finally:
        owner.terminate_all()
    try:
        model_errors = await asyncio.to_thread(_model_errors, request.artifact_dir / "zcode-home/logs")
    except Exception as exc:
        # Optional diagnostics must not replace the task result or its cancellation.
        model_errors = {
            "model_stream_error_count": 0,
            "last_model_error": None,
            "model_log_warnings": [{"error": f"{type(exc).__name__}: {exc}"}],
        }
    (request.artifact_dir / "zcode-run-status.json").write_text(
        json.dumps(
            {
                "pid": process.pid if process else None,
                "exit_code": process.returncode if process else None,
                "termination_reason": result.termination_reason,
                "outcome": result.outcome,
                "native_error": native_error,
                "prompt_loaded": (request.artifact_dir / "zcode-home/prompt.loaded").exists(),
                "execution_phase": "agent" if bridge.sessions else "startup",
                **model_errors,
            },
            indent=2,
        )
        + "\n"
    )
    result.finished_at = datetime.now(timezone.utc).isoformat()
    result.duration_seconds = time.monotonic() - started
    if cancellation:
        raise cancellation
    return result


def _model_errors(log_dir):
    count = 0
    last = None
    warnings = []
    for path in sorted(log_dir.glob("*.jsonl")):
        try:
            with path.open("rb") as handle:
                for number, line in enumerate(handle, 1):
                    try:
                        if not line.endswith(b"\n"):
                            raise ValueError("incomplete final log record")
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            raise ValueError("expected a JSON object")
                        if row.get("event") == "model.sdk.stream.failed":
                            last = {"timestamp": row["timestamp"], "error": row["error"], "log_path": str(path)}
                            count += 1
                    except (ValueError, KeyError) as exc:
                        warnings.append(
                            {"log_path": str(path), "line": number, "error": f"{type(exc).__name__}: {exc}"}
                        )
        except OSError as exc:
            warnings.append({"log_path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    return {"model_stream_error_count": count, "last_model_error": last, "model_log_warnings": warnings}


async def _stop_process_group(process: asyncio.subprocess.Process, cleanup_seconds: float) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        # The pinned CLI allows six seconds to reap its detached Bash processes.
        await asyncio.wait_for(process.wait(), min(7, cleanup_seconds))
    except asyncio.TimeoutError:
        pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()
