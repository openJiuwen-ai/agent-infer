# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentCache project
"""
E2E test for AgentAwareScheduler via vllm-acache serve subprocess.

Mirrors vllm's RemoteOpenAIServer pattern:
- Launches vllm-acache serve in a subprocess
- Waits for /health readiness
- Sends completions via openai.OpenAI client
- Verifies AgentAwareScheduler is loaded in server logs
"""

import os
import signal
import subprocess
import time

import openai
import pytest

MODEL = "Qwen/Qwen3-0.6B"
LOG_FILE = os.path.join(os.path.dirname(__file__), "_e2e_server.log")


def _get_open_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_health(port: int, timeout: int) -> None:
    import urllib.request

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = urllib.request.urlopen(f"http://localhost:{port}/health", timeout=5)
            if resp.status == 200:
                return
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"Server on port {port} did not become healthy within {timeout}s")


def _shutdown(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()


@pytest.fixture(scope="module")
def server():
    """Launch vllm serve, yield port, then shutdown."""
    port = _get_open_port()

    env = os.environ.copy()
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    cmd = [
        "vllm-acache",
        "serve",
        MODEL,
        "--port",
        str(port),
        "--enforce-eager",
        "--max-model-len",
        "512",
        "--gpu-memory-utilization",
        "0.5",
        "--no-enable-prefix-caching",
    ]

    with open(LOG_FILE, "w") as log_fp:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    try:
        _wait_for_health(port, timeout=300)
    except Exception:
        _shutdown(proc)
        raise

    yield port
    _shutdown(proc)
    try:
        os.remove(LOG_FILE)
    except OSError:
        pass


@pytest.fixture
def client(server):
    return openai.OpenAI(
        base_url=f"http://localhost:{server}/v1",
        api_key="token-abc123",
        max_retries=0,
    )


def test_agent_scheduler_configured(server):
    """Verify AgentAwareScheduler appears in the server logs."""
    with open(LOG_FILE) as f:
        content = f.read()
    assert "Using custom scheduler class" in content, (
        "AgentAwareScheduler warning not found in server logs"
    )


def test_completion(client):
    """Send a completion request and verify the response."""
    completion = client.completions.create(
        model=MODEL,
        prompt="Hello, my name is",
        max_tokens=8,
        temperature=0.0,
    )
    assert len(completion.choices) == 1
    assert completion.choices[0].text


def test_chat_completion(client):
    """Send a chat completion request and verify the response."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Say hello in one word."}],
        max_tokens=8,
        temperature=0.0,
    )
    assert len(response.choices) == 1
    assert response.choices[0].message.content


def test_agent_aware_queue_logged(server):
    """Verify AgentAwareQueue is referenced in server logs."""
    with open(LOG_FILE) as f:
        content = f.read()
    assert "AgentAwareQueue" in content or "AgentAwareScheduler" in content, (
        "AgentAwareQueue/AgentAwareScheduler not found in server logs"
    )


def test_multi_request_throughput(client):
    """Send multiple requests to exercise the AgentAwareQueue FCFS path."""
    prompts = [f"Hello my name is {name}" for name in ("Alice", "Bob", "Charlie", "Diana", "Eve")]
    for prompt in prompts:
        completion = client.completions.create(
            model=MODEL,
            prompt=prompt,
            max_tokens=4,
            temperature=0.0,
        )
        assert len(completion.choices) == 1
        assert completion.choices[0].text
