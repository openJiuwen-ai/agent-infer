# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Verify raw benchmark evidence collectors."""

import asyncio
import json
import platform
import subprocess
import sys

import httpx
import pytest

from agentinfer.agentbench.benchkit.collectors.correctness import load_correctness_artifact
from agentinfer.agentbench.benchkit.collectors.environment import collect_environment
from agentinfer.agentbench.benchkit.collectors.source_control import collect_source_control
from agentinfer.agentbench.benchkit.collectors.vllm import capture_vllm_metrics

from .http_fakes import FakeClient, FakeResponse


def test_load_correctness_artifact_returns_raw_object(tmp_path) -> None:
    path = tmp_path / "correctness.json"
    path.write_text(json.dumps({"resolved": True, "score": 1}), encoding="utf-8")

    capture = load_correctness_artifact(path)

    assert capture.available is True
    assert capture.path == path
    assert capture.metadata == {"resolved": True, "score": 1}


def test_load_correctness_artifact_reports_missing_directory_and_invalid_input(tmp_path) -> None:
    missing = load_correctness_artifact(tmp_path / "missing.json")
    assert missing.available is False
    assert missing.reason == "artifact missing"

    directory = load_correctness_artifact(tmp_path)
    assert directory.available is False
    assert directory.reason == "artifact missing"

    invalid = tmp_path / "invalid.json"
    invalid.write_text("[]", encoding="utf-8")
    capture = load_correctness_artifact(invalid)
    assert capture.available is False
    assert capture.reason == "artifact must contain a JSON object"

    invalid_encoding = tmp_path / "invalid-encoding.json"
    invalid_encoding.write_bytes(b"\xff")
    capture = load_correctness_artifact(invalid_encoding)
    assert capture.available is False
    assert capture.path == invalid_encoding
    assert capture.reason is not None


def test_collect_environment_returns_reproducibility_metadata(monkeypatch) -> None:
    monkeypatch.setattr(platform, "platform", lambda: "test-platform")
    monkeypatch.setattr(platform, "node", lambda: "test-host")

    capture = collect_environment()

    assert capture.available is True
    assert capture.path is None
    assert capture.metadata == {
        "platform": "test-platform",
        "hostname": "test-host",
        "python_version": sys.version,
    }


def test_collect_source_control_resolves_root_from_imported_path(tmp_path, monkeypatch) -> None:
    source = tmp_path / "checkout" / "agentinfer" / "module.py"
    source.parent.mkdir(parents=True)
    source.touch()
    root = tmp_path / "checkout"
    calls = []
    responses = [
        subprocess.CompletedProcess([], 0, stdout=f"{root}\n", stderr=""),
        subprocess.CompletedProcess([], 0, stdout="abc123\n", stderr=""),
        subprocess.CompletedProcess([], 0, stdout=" M changed.py\n", stderr=""),
    ]

    def run(command, **kwargs):
        calls.append(command)
        return responses.pop(0)

    monkeypatch.setattr(subprocess, "run", run)

    capture = collect_source_control(source)

    assert calls[0] == ["git", "-C", str(source.parent), "rev-parse", "--show-toplevel"]
    assert calls[1][:3] == ["git", "-C", str(root)]
    assert capture.available is True
    assert capture.path is None
    assert capture.metadata == {"root": str(root), "commit": "abc123", "dirty": True}


@pytest.mark.parametrize(
    "error",
    [
        subprocess.TimeoutExpired(["git"], 10),
        FileNotFoundError("git executable missing"),
        subprocess.CalledProcessError(1, ["git"]),
    ],
)
def test_collect_source_control_converts_process_failures_to_evidence(tmp_path, monkeypatch, error) -> None:
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(subprocess, "run", fail)

    capture = collect_source_control(tmp_path / "imported" / "agentinfer")

    assert capture.available is False
    assert capture.path is None
    assert capture.reason


def test_capture_vllm_metrics_returns_raw_text(monkeypatch) -> None:
    response = FakeResponse(text="vllm:prefix_cache_hits_total 4\n")
    urls = []

    class Client(FakeClient):
        async def get(self, url: str):
            urls.append(url)
            return await super().get(url)

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: Client(response, **kwargs))

    capture = asyncio.run(capture_vllm_metrics("http://vllm:8000/metrics"))

    assert capture.available is True
    assert capture.path is None
    assert capture.metadata == {"text": "vllm:prefix_cache_hits_total 4\n"}
    assert urls == ["http://vllm:8000/metrics"]


def test_capture_vllm_metrics_reports_http_failure(monkeypatch) -> None:
    error = httpx.ConnectError("connection failed", request=httpx.Request("GET", "http://vllm:8000/metrics"))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: FakeClient(error=error, **kwargs))

    capture = asyncio.run(capture_vllm_metrics("http://vllm:8000/metrics"))

    assert capture.available is False
    assert capture.path is None
    assert "connection failed" in capture.reason
