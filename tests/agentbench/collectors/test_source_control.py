# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

import subprocess

import pytest

from agentinfer.agentbench.benchkit.collectors.source_control import collect_source_control


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
