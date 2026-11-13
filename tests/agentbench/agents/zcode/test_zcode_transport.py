# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

import asyncio
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from agentinfer.agentbench.agents.zcode import stream
from agentinfer.agentbench.agents.zcode.runner import run_zcode
from tests.agentbench.agents.zcode.test_zcode_integration import ZCodeMockServer
from tests.agentbench.helpers.dsh_mock_server import seed_tasks


def test_fragmented_sse_heartbeats_preserve_events(monkeypatch):
    monkeypatch.setattr(stream, "SSE_HEARTBEAT_SECONDS", 0.01)
    wire = 'data: {"arguments":"你好"}\r\n\r\ndata: [DONE]\n\n'.encode()

    async def scenario():
        closed = asyncio.Event()

        async def chunks():
            try:
                for byte in wire:
                    await asyncio.sleep(0.015)
                    yield bytes([byte])
            finally:
                closed.set()

        output = [chunk async for chunk in stream.with_heartbeats(chunks())]
        assert b": agentbench keep-alive\n\n" in output
        assert b"".join(chunk for chunk in output if not chunk.startswith(b":")) == wire
        assert closed.is_set()
        # Simulate a disconnected CLI while the producer is waiting for its next event.
        closed.clear()
        iterator = stream.with_heartbeats(chunks())
        assert (await anext(iterator)).startswith(b":")
        await iterator.aclose()
        assert closed.is_set()

    asyncio.run(scenario())


def test_prompt_file_and_heartbeat_with_real_cli(tmp_path, monkeypatch):
    import time

    from agentinfer.agentbench.agents.contracts import AgentRunRequest
    from agentinfer.agentbench.benchkit.dataset import Task
    from tests.agentbench.helpers.dsh_mock_server import _Handler

    executable = os.environ.get("ZCODE_TEST_EXECUTABLE")
    if not executable:
        pytest.skip("Set ZCODE_TEST_EXECUTABLE to the pinned launcher")
    monkeypatch.setattr(stream, "SSE_HEARTBEAT_SECONDS", 0.02)

    class SlowHandler(_Handler):
        def end_headers(self):
            super().end_headers()
            self.wfile.flush()
            time.sleep(0.15)

    seed_tasks(tmp_path, 1)
    # Above Linux's single-argument limit, with non-ASCII and a tail sentinel.
    description = "Fix add. " + "输入内容\n" * 15000 + "PROMPT_END_完整"
    request = AgentRunRequest(
        agent_type="zcode",
        task=Task("calc-0", "agentbench-fixture/tiny-calc", "HEAD", description),
        profile_name="single",
        executable=Path(executable),
        model="dsh-test",
        api_base_url="",
        workspace=tmp_path / "cache/agentbench-fixture__tiny-calc",
        artifact_dir=tmp_path / "artifacts",
        session_id="long-prompt",
        timeout_seconds=60,
        patch_flush_seconds=0,
        prompt_delivery_timeout_seconds=10,
        tmux_startup_seconds=0,
        terminal_capture_interval_seconds=1,
    )
    with ZCodeMockServer("single") as server:
        server.RequestHandlerClass = SlowHandler
        result = asyncio.run(run_zcode(replace(request, api_base_url=server.base_url)))
    assert result.outcome.value == "completed" and result.has_patch
    saved = (request.artifact_dir / "prompt.txt").read_text()
    assert len(saved.encode()) > 131072
    assert any(description in message.get("content", "") for message in server.requests[0]["body"]["messages"])
    assert not any("agentbench keep-alive" in json.dumps(row["body"]) for row in server.requests)
    status = json.loads((request.artifact_dir / "zcode-run-status.json").read_text())
    assert status["prompt_loaded"] and status["execution_phase"] == "agent"
    command = json.loads((request.artifact_dir / "zcode-command.json").read_text())
    assert saved not in command


@pytest.mark.parametrize("mode", ["completed", "timeout", "cancelled", "collector_error"])
def test_auxiliary_logs_preserve_task_outcome(tmp_path, monkeypatch, mode):
    from agentinfer.agentbench.agents.contracts import AgentRunRequest
    from agentinfer.agentbench.benchkit.dataset import Task

    seed_tasks(tmp_path, 1)
    error = {"name": "TypeError", "cause": {"code": "UND_ERR_BODY_TIMEOUT"}}
    event = {"event": "model.sdk.stream.failed", "timestamp": "2026-11-10T00:00:00Z", "error": error}
    log_bytes = (
        json.dumps(event).encode()
        + b"\nnot-json\n[]\n\xff\n"
        + json.dumps({"event": "model.sdk.stream.failed"}).encode()
        + b"\n"
        + json.dumps(event).encode()
        + b'\n{"partial":'
    )
    executable = tmp_path / "timeout-cli"
    executable.write_text(
        f"#!{sys.executable}\nimport os,time\nfrom pathlib import Path\n"
        "p=Path(os.environ['ZCODE_LOG_DIR']);p.mkdir(parents=True)\n"
        "(p/'unreadable.jsonl').mkdir()\n"
        "Path('calc.py').write_text('def add(a, b): return a + b\\n')\n"
        f"(p/'runtime.jsonl').write_bytes({log_bytes!r})\n"
        f"time.sleep({10 if mode in ('timeout', 'cancelled') else 0})\n"
        f"print({json.dumps({'type': 'result', 'projection': {'status': 'idle'}})!r})\n"
    )
    executable.chmod(0o755)
    request = AgentRunRequest(
        agent_type="zcode",
        task=Task("calc-0", "agentbench-fixture/tiny-calc", "HEAD", "fix"),
        profile_name="single",
        executable=executable,
        model="dsh-test",
        api_base_url="http://unused",
        workspace=tmp_path / "cache/agentbench-fixture__tiny-calc",
        artifact_dir=tmp_path / "artifacts",
        session_id="timeout",
        timeout_seconds=1 if mode == "timeout" else 20,
        patch_flush_seconds=0,
        prompt_delivery_timeout_seconds=1,
        tmux_startup_seconds=0,
        terminal_capture_interval_seconds=1,
    )
    if mode == "collector_error":

        def broken_collector(_):
            raise OSError("diagnostic collector unavailable")

        monkeypatch.setattr("agentinfer.agentbench.agents.zcode.runner._model_errors", broken_collector)

    async def scenario():
        job = asyncio.create_task(run_zcode(request))
        if mode == "cancelled":
            for _ in range(100):
                if (request.artifact_dir / "zcode-home/logs/runtime.jsonl").exists():
                    break
                await asyncio.sleep(0.02)
            job.cancel()
            with pytest.raises(asyncio.CancelledError):
                await job
        else:
            result = await job
            assert result.outcome.value == ("failed" if mode == "timeout" else "completed")
            assert result.has_patch

    asyncio.run(scenario())
    status = json.loads((request.artifact_dir / "zcode-run-status.json").read_text())
    assert status["termination_reason"] == (mode if mode in ("timeout", "cancelled") else None)
    assert (request.artifact_dir / "model.patch").exists()
    if mode == "collector_error":
        assert "collector unavailable" in status["model_log_warnings"][0]["error"]
    else:
        assert status["model_stream_error_count"] == 2
        assert status["last_model_error"]["error"] == error
        assert len(status["model_log_warnings"]) == 6
