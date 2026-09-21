# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""
E2E test for `vllm serve MODEL --agentinfer` through the installed vllm entrypoint.

- Launches the single-flag AgentInfer serving path in a subprocess
- Waits for /health readiness
- Sends completions through the OpenAI-compatible API
- Verifies the injection banner and the lifecycle socket
- Verifies conflicting flags exit with code 2 before engine startup
"""

import importlib.util
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

if importlib.util.find_spec("vllm") is None:
    pytest.skip("vllm is not installed; serve e2e requires a vLLM environment", allow_module_level=True)

pytestmark = pytest.mark.gpu_test

MODEL = os.environ.get("AGENTINFER_E2E_MODEL", "Qwen/Qwen3-0.6B")
STARTUP_TIMEOUT = int(os.environ.get("AGENTINFER_E2E_STARTUP_TIMEOUT", "300"))


def _vllm_bin() -> str:
    """Resolve the installed agentinfer vllm console script."""

    candidates = [
        os.path.join(os.path.dirname(sys.executable), "vllm"),
        os.path.join(sys.prefix, "bin", "vllm"),
        shutil.which("vllm"),
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    pytest.skip("vllm console script not found; run inside the agentinfer environment")


def _get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_health(port: int, timeout: int) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = urllib.request.urlopen(f"http://localhost:{port}/health", timeout=5)
            if resp.status == 200:
                return
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"Server on port {port} did not become healthy within {timeout}s")


def _post_json(port: int, path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        f"http://localhost:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as resp:
        return json.loads(resp.read())


def _shutdown(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()


@pytest.fixture(scope="module")
def server(tmp_path_factory: pytest.TempPathFactory):
    """Launch `vllm serve MODEL --agentinfer`, yield (port, log_file, socket), then shutdown."""
    workdir = tmp_path_factory.mktemp("agentinfer-serve-e2e")
    log_file = workdir / "server.log"
    lifecycle_socket = workdir / "lifecycle.sock"
    port = _get_open_port()

    vllm_bin = _vllm_bin()
    env = os.environ.copy()
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    env["AGENTCACHE_VLLM_LIFECYCLE_SOCKET"] = str(lifecycle_socket)

    cmd = [
        vllm_bin,
        "serve",
        MODEL,
        "--agentinfer",
        "--port",
        str(port),
        "--enforce-eager",
        "--max-model-len",
        "512",
        "--gpu-memory-utilization",
        "0.5",
        "--no-enable-prefix-caching",
    ]

    with open(log_file, "w") as log_fp:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    try:
        _wait_for_health(port, STARTUP_TIMEOUT)
    except Exception:
        _shutdown(proc)
        with open(log_file) as f:
            pytest.fail(f"server failed to start; log tail:\n{f.read()[-4000:]}")

    yield port, str(log_file), str(lifecycle_socket)
    _shutdown(proc)


def test_injection_banner_logged(server):
    """The takeover prints the injected scheduler bridge and middleware to stderr."""
    _, log_file, _ = server
    with open(log_file) as f:
        content = f.read()
    assert "[agentinfer] --agentinfer injected:" in content
    assert "AgentCacheAsyncSchedulerBridge" in content
    assert "AgentCacheLifecycleMiddleware" in content


def test_lifecycle_socket_bound(server):
    """The injected lifecycle middleware binds the configured unix socket."""
    _, _, lifecycle_socket = server
    rank_socket = f"{lifecycle_socket}.dp0"
    assert os.path.exists(rank_socket), f"lifecycle socket {rank_socket} was not created"


def test_completion(server):
    """A completion request round-trips through the AgentInfer serving path."""
    port, _, _ = server
    body = _post_json(
        port,
        "/v1/completions",
        {"model": MODEL, "prompt": "Hello, my name is", "max_tokens": 8, "temperature": 0.0},
    )
    assert body["choices"] and body["choices"][0]["text"]


def test_chat_completion(server):
    """A chat completion request round-trips through the AgentInfer serving path."""
    port, _, _ = server
    body = _post_json(
        port,
        "/v1/chat/completions",
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Say hello in one word."}],
            "max_tokens": 8,
            "temperature": 0.0,
        },
    )
    assert body["choices"] and body["choices"][0]["message"]["content"]


def test_conflicting_scheduler_cls_exits_before_engine_startup():
    """--agentinfer plus --scheduler-cls exits with code 2 without booting the engine."""
    cmd = [
        _vllm_bin(),
        "serve",
        MODEL,
        "--agentinfer",
        "--scheduler-cls",
        "other.OtherScheduler",
        "--port",
        str(_get_open_port()),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 2
    assert "[agentinfer] error:" in proc.stderr
    assert "--scheduler-cls" in proc.stderr
