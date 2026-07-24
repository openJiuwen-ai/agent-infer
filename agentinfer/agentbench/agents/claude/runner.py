"""Run one benchmark task through Claude Code."""

import asyncio
import json
import os
import shlex
import time
from datetime import datetime, timezone
from pathlib import Path

from ...benchkit.workspace import export_patch, workspace_has_changes
from ..contracts import AgentRunRequest, AgentRunResult
from ..outcomes import AgentRunOutcome, TerminationReason
from .interaction import InteractionContext, InteractionController, InteractionState, handler_types_for_profile
from .profiles import get_profile
from .settings import bootstrap_claude_state, build_claude_env, build_claude_settings
from .tmux import (
    TmuxTarget,
    capture_pane,
    kill_server,
    load_and_paste_file,
    pane_has_exited,
    remove_global_environment,
    send_keys,
    session_exists,
    set_history_limit,
    start_session,
)
from .transcript import RawEvent, StuckDetector, TranscriptLoader, TranscriptNormalizer, is_transcript_complete

_FINALIZATION_TIMEOUT_SECONDS = 30.0
_STARTUP_TIMEOUT_SECONDS = 30.0
_TMUX_OPERATION_TIMEOUT_SECONDS = 30.0


def _prompt_paste_delay(prompt_bytes: int) -> float:
    return min(8.0, max(0.5, prompt_bytes / 32768))


def _find_transcript_path(config_dir: Path, session_id: str) -> Path | None:
    for path in config_dir.rglob("**/projects/**/*.jsonl"):
        if session_id in path.name:
            return path
    return None


def _write_transcript_jsonl(path: Path, rows: list[RawEvent]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


# Blocking filesystem and process operations are offloaded from the event loop.


async def _mkdir_async(path: Path) -> None:
    await asyncio.to_thread(path.mkdir, parents=True, exist_ok=True)


async def _write_text_async(path: Path, text: str) -> None:
    await asyncio.to_thread(path.write_text, text, encoding="utf-8")


async def _write_transcript_jsonl_async(path: Path, rows: list[RawEvent]) -> None:
    await asyncio.to_thread(_write_transcript_jsonl, path, rows)


def _remaining_seconds(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Claude task deadline expired")
    return remaining


async def _capture_pane_async(target: TmuxTarget, deadline: float) -> str:
    return await asyncio.to_thread(
        capture_pane,
        target,
        timeout_seconds=_remaining_seconds(deadline),
    )


async def _session_exists_async(target: TmuxTarget, deadline: float) -> bool:
    return await asyncio.to_thread(
        session_exists,
        target,
        timeout_seconds=_remaining_seconds(deadline),
    )


async def _pane_has_exited_async(target: TmuxTarget, deadline: float) -> bool:
    return await asyncio.to_thread(
        pane_has_exited,
        target,
        timeout_seconds=_remaining_seconds(deadline),
    )


async def _send_keys_async(target: TmuxTarget, deadline: float, *keys: str) -> None:
    await asyncio.to_thread(
        send_keys,
        target,
        *keys,
        timeout_seconds=_remaining_seconds(deadline),
    )


async def _paste_prompt_async(
    target: TmuxTarget,
    buffer_name: str,
    path: Path,
    deadline: float,
) -> None:
    await asyncio.to_thread(
        load_and_paste_file,
        target,
        buffer_name,
        path,
        timeout_seconds=_remaining_seconds(deadline),
        deadline=deadline,
    )


async def _find_transcript_path_async(config_dir: Path, session_id: str) -> Path | None:
    return await asyncio.to_thread(_find_transcript_path, config_dir, session_id)


async def _load_transcript_async(loader: TranscriptLoader, path: Path) -> list[RawEvent]:
    return await asyncio.to_thread(loader.load, path)


async def _file_signature_async(path: Path) -> tuple[int, int]:
    stat = await asyncio.to_thread(path.stat)
    return stat.st_mtime_ns, stat.st_size


async def _export_patch_async(workspace: Path) -> str:
    return await asyncio.to_thread(export_patch, workspace)


async def _workspace_has_changes_async(workspace: Path) -> bool:
    return await asyncio.to_thread(workspace_has_changes, workspace)


async def _sleep_with_deadline(delay: float, deadline: float) -> bool:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return False
    await asyncio.sleep(min(delay, remaining))
    return time.monotonic() < deadline


# Prompt delivery is bounded by the same task deadline as runtime polling.


async def _confirm_prompt_submission(
    config_dir: Path,
    session_id: str,
    target: TmuxTarget,
    deadline: float,
) -> tuple[Path | None, int]:
    next_retry = time.monotonic() + 2
    retries = 0
    while time.monotonic() < deadline:
        transcript_path = await _find_transcript_path_async(config_dir, session_id)
        if transcript_path:
            return transcript_path, retries
        if not await _session_exists_async(target, deadline) or await _pane_has_exited_async(target, deadline):
            return None, retries
        terminal = await _capture_pane_async(target, deadline)
        now = time.monotonic()
        if now >= next_retry and "[Pasted text #" in terminal:
            await _send_keys_async(target, deadline, "Enter")
            retries += 1
            next_retry = now + 2
        await _sleep_with_deadline(0.25, deadline)
    return await _find_transcript_path_async(config_dir, session_id), retries


# Final evidence is collected independently of the execution outcome.


async def _collect_final_evidence(
    request: AgentRunRequest,
    result: AgentRunResult,
    target: TmuxTarget,
    loader: TranscriptLoader,
    claude_state_dir: Path,
    transcript_path: Path,
) -> None:
    finalization_deadline = time.monotonic() + _FINALIZATION_TIMEOUT_SECONDS

    if await _session_exists_async(target, finalization_deadline):
        terminal = await _capture_pane_async(target, finalization_deadline)
        await _write_text_async(request.artifact_dir / "terminal-final.log", terminal)
        if not await _pane_has_exited_async(target, finalization_deadline):
            await _send_keys_async(target, finalization_deadline, "C-d")
            await _sleep_with_deadline(0.5, finalization_deadline)
            if not await _pane_has_exited_async(target, finalization_deadline):
                await _send_keys_async(target, finalization_deadline, "C-d")

    if request.patch_flush_seconds:
        await asyncio.sleep(request.patch_flush_seconds)

    transcript_file = await _find_transcript_path_async(claude_state_dir, request.session_id)
    if transcript_file:
        transcript = await _load_transcript_async(loader, transcript_file)
        await _write_transcript_jsonl_async(transcript_path, transcript)
        result.transcript = str(transcript_path)

    patch = await _export_patch_async(request.workspace)
    if patch:
        await _write_text_async(request.artifact_dir / "model.patch", patch)
        result.has_patch = True
        result.patch_bytes = len(patch.encode("utf-8"))


async def run_claude(request: AgentRunRequest) -> AgentRunResult:
    """Run one task through Claude Code and return its benchmark result."""

    profile = get_profile(request.profile_name)
    artifact_dir = request.artifact_dir
    session_id = request.session_id
    target = TmuxTarget(socket_name=f"agentinfer-{session_id.replace('-', '')[:16]}")
    transcript_path = artifact_dir / "transcript.jsonl"
    start_mono = time.monotonic()
    task_deadline = start_mono + request.timeout_seconds
    result = AgentRunResult(
        instance_id=request.task.instance_id,
        session_id=session_id,
        tmux_session=f"{target.socket_name}:claude",
        agent_type=request.agent_type,
        profile_name=profile.name,
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    tmux_owned = False
    launch_task: asyncio.Task[None] | None = None
    cancellation: asyncio.CancelledError | None = None

    await _mkdir_async(artifact_dir)
    prompt = profile.build_prompt(request.task)
    prompt_path = artifact_dir / "prompt.txt"
    await _write_text_async(prompt_path, prompt)

    claude_state_dir = artifact_dir / "claude_state"
    await _mkdir_async(claude_state_dir)
    await asyncio.to_thread(bootstrap_claude_state, claude_state_dir)

    settings_path = artifact_dir / "claude-settings.json"
    settings = build_claude_settings(request.api_base_url)
    await _write_text_async(settings_path, json.dumps(settings, indent=2))

    assignments = build_claude_env(claude_state_dir, request.api_base_url)
    command = [
        str(request.executable),
        "--session-id",
        session_id,
        "--settings",
        str(settings_path),
        "--permission-mode",
        profile.permission_mode,
        "--model",
        request.model,
        "--disable-slash-commands",
        "--disallowedTools",
        ",".join(profile.tool_policy.disallowed_tools),
    ]
    await _write_text_async(artifact_dir / "claude-launch-command.txt", shlex.join(command) + "\n")

    state = InteractionState()
    controller = InteractionController(
        handler_types_for_profile(profile.interaction_profile),
        enable_idle_detection=profile.name == "single",
    )
    loader = TranscriptLoader()
    normalizer = TranscriptNormalizer()
    stuck_detector = StuckDetector()

    try:
        _remaining_seconds(task_deadline)
        tmux_owned = True
        launch_task = asyncio.create_task(
            asyncio.to_thread(
                start_session,
                target,
                command=command,
                cwd=request.workspace,
                launch_env={**os.environ, **assignments},
                timeout_seconds=min(_TMUX_OPERATION_TIMEOUT_SECONDS, _remaining_seconds(task_deadline)),
            )
        )
        await asyncio.shield(launch_task)
        await asyncio.to_thread(
            remove_global_environment,
            target,
            "ANTHROPIC_AUTH_TOKEN",
            timeout_seconds=_remaining_seconds(task_deadline),
        )
        await asyncio.to_thread(
            set_history_limit,
            target,
            200000,
            timeout_seconds=_remaining_seconds(task_deadline),
        )

        startup_deadline = min(task_deadline, time.monotonic() + _STARTUP_TIMEOUT_SECONDS)
        if not await _sleep_with_deadline(request.tmux_startup_seconds, startup_deadline):
            result.termination_reason = TerminationReason.AGENT_STARTUP_TIMEOUT
        else:
            transcript_file: Path | None = None
            transcript_signature: tuple[int, int] | None = None
            transcript_raw: list[RawEvent] = []
            capture_count = 0
            snapshot_counter = 0

            while time.monotonic() < task_deadline:
                if not await _sleep_with_deadline(0.5, task_deadline):
                    result.termination_reason = TerminationReason.TIMEOUT
                    break
                if not await _session_exists_async(target, task_deadline):
                    result.termination_reason = (
                        TerminationReason.AGENT_STARTUP_FAILED
                        if transcript_file is None
                        else TerminationReason.INTERRUPTED
                    )
                    break

                terminal = await _capture_pane_async(target, task_deadline)
                capture_count += 1
                if capture_count <= 10 or capture_count % (request.terminal_capture_interval_seconds * 2) == 0:
                    await _write_text_async(artifact_dir / f"terminal-{snapshot_counter}.log", terminal)
                    snapshot_counter += 1

                if await _pane_has_exited_async(target, task_deadline):
                    result.termination_reason = (
                        TerminationReason.AGENT_STARTUP_FAILED
                        if transcript_file is None
                        else TerminationReason.INTERRUPTED
                    )
                    break

                if not state.startup_dismissed:
                    action = controller.process(
                        InteractionContext(terminal, None, time.monotonic() - start_mono, state)
                    )
                    if action.kind == "send_keys":
                        await _send_keys_async(target, task_deadline, action.keys)
                        if not await _sleep_with_deadline(1.0, task_deadline):
                            result.termination_reason = TerminationReason.TIMEOUT
                            break
                        continue
                    if not state.startup_dismissed:
                        if time.monotonic() >= startup_deadline:
                            result.termination_reason = TerminationReason.AGENT_STARTUP_TIMEOUT
                            break
                        continue

                if transcript_file is None and _ready_for_user_prompt(terminal):
                    await _paste_prompt_async(
                        target,
                        f"agentinfer-prompt-{session_id[:8]}",
                        prompt_path,
                        task_deadline,
                    )
                    if not await _sleep_with_deadline(
                        _prompt_paste_delay(len(prompt.encode("utf-8"))),
                        task_deadline,
                    ):
                        result.termination_reason = TerminationReason.TIMEOUT
                        break
                    await _send_keys_async(target, task_deadline, "Enter")
                    prompt_deadline = min(
                        task_deadline,
                        time.monotonic() + request.prompt_delivery_timeout_seconds,
                    )
                    transcript_file, retries = await _confirm_prompt_submission(
                        claude_state_dir,
                        session_id,
                        target,
                        prompt_deadline,
                    )
                    result.prompt_submission_retries = retries
                    if transcript_file is None:
                        if not await _session_exists_async(target, task_deadline):
                            result.termination_reason = TerminationReason.AGENT_STARTUP_FAILED
                        elif await _pane_has_exited_async(target, task_deadline):
                            result.termination_reason = TerminationReason.AGENT_STARTUP_FAILED
                        else:
                            result.termination_reason = (
                                TerminationReason.TIMEOUT
                                if time.monotonic() >= task_deadline
                                else TerminationReason.PROMPT_SUBMISSION_FAILED
                            )
                        break

                if transcript_file is None:
                    continue

                signature = await _file_signature_async(transcript_file)
                if signature != transcript_signature:
                    transcript_signature = signature
                    transcript_raw = await _load_transcript_async(loader, transcript_file)

                if is_transcript_complete(transcript_raw):
                    result.outcome = AgentRunOutcome.COMPLETED
                    result.termination_reason = None
                    break

                if transcript_raw:
                    stuck_reason = stuck_detector.detect(normalizer.normalize(transcript_raw))
                    if stuck_reason is not None:
                        result.termination_reason = stuck_reason
                        break

                now = time.monotonic()
                action = controller.process(InteractionContext(terminal, transcript_raw, now - start_mono, state))
                if action.kind == "send_keys":
                    await _send_keys_async(target, task_deadline, action.keys)
                elif action.kind == "check_workspace":
                    if await _workspace_has_changes_async(request.workspace):
                        result.termination_reason = TerminationReason.IDLE_AFTER_PATCH
                        break
            else:
                result.termination_reason = (
                    TerminationReason.AGENT_STARTUP_TIMEOUT
                    if not state.startup_dismissed
                    else TerminationReason.TIMEOUT
                )

        result.auto_yes_confirmations = state.auto_yes_confirmations
        result.auto_plan_approvals = state.auto_plan_approvals
        try:
            await asyncio.wait_for(
                _collect_final_evidence(
                    request,
                    result,
                    target,
                    loader,
                    claude_state_dir,
                    transcript_path,
                ),
                timeout=request.patch_flush_seconds + _FINALIZATION_TIMEOUT_SECONDS,
            )
        except Exception:
            if result.outcome is AgentRunOutcome.COMPLETED:
                result.outcome = AgentRunOutcome.FAILED
                result.termination_reason = TerminationReason.HARNESS_ERROR
    except asyncio.CancelledError as exc:
        cancellation = exc
        if launch_task is not None:
            try:
                await asyncio.shield(launch_task)
            except Exception:
                pass
    except Exception:
        result.outcome = AgentRunOutcome.FAILED
        result.termination_reason = TerminationReason.HARNESS_ERROR
    finally:
        if tmux_owned:
            try:
                await asyncio.shield(asyncio.to_thread(kill_server, target))
            except Exception:
                if cancellation is None and result.outcome is AgentRunOutcome.COMPLETED:
                    result.outcome = AgentRunOutcome.FAILED
                    result.termination_reason = TerminationReason.HARNESS_ERROR

    if cancellation is not None:
        raise cancellation

    result.finished_at = datetime.now(timezone.utc).isoformat()
    result.duration_seconds = time.monotonic() - start_mono
    return result


def _ready_for_user_prompt(terminal: str) -> bool:
    """Return whether Claude is ready for the benchmark prompt."""

    low = terminal.lower()
    if "[pasted text #" in low:
        return False
    return "❯" in terminal or "welcome back" in low
