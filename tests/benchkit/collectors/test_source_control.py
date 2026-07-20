import subprocess

import pytest

from agentcache.benchmarks.benchkit.collectors.source_control import collect_source_control


def test_collect_source_control_reports_commit_and_dirty_state(tmp_path, monkeypatch) -> None:
    responses = [
        subprocess.CompletedProcess([], 0, stdout="abc123\n", stderr=""),
        subprocess.CompletedProcess([], 0, stdout=" M changed.py\n", stderr=""),
    ]
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: responses.pop(0))

    capture = collect_source_control(tmp_path)

    assert capture.available is True
    assert capture.path is None
    assert capture.metadata == {"commit": "abc123", "dirty": True}


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

    capture = collect_source_control(tmp_path)

    assert capture.available is False
    assert capture.path is None
    assert capture.reason
