"""AC6 readiness: box preflight is honest (never fabricates reachability) (zero GPU)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(SRC))

from vllm_evolve.tools import box_preflight as bp  # noqa: E402


def _fake(returncode, stdout="", stderr=""):
    def run(*a, **k):
        r = type("R", (), {})()
        r.returncode, r.stdout, r.stderr = returncode, stdout, stderr
        return r
    return run


def test_closed_connection_is_unreachable(monkeypatch):
    # the box's actual failure mode this whole session: ssh closes pre-auth (exit 255)
    monkeypatch.setattr(subprocess, "run", _fake(255, stderr="Connection closed by remote host"))
    ok, detail = bp.box_reachable("box")
    assert ok is False and "255" in detail


def test_ssh_exception_is_unreachable(monkeypatch):
    def boom(*a, **k):
        raise OSError("network down")
    monkeypatch.setattr(subprocess, "run", boom)
    ok, detail = bp.box_reachable("box")
    assert ok is False and "ssh failed" in detail


def test_reachable_with_gpus(monkeypatch):
    monkeypatch.setattr(subprocess, "run",
                        _fake(0, stdout="GPU 0: A6000\nGPU 1: A6000\n"))
    ok, detail = bp.box_reachable("box")
    assert ok is True and "2 GPU" in detail


def test_ssh_ok_but_no_gpu_is_not_reachable(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake(0, stdout="\n"))
    ok, _ = bp.box_reachable("box")
    assert ok is False   # ssh worked but no GPU -> not usable for the real run


def test_cli_degrades_honestly_when_unreachable(monkeypatch, capsys):
    monkeypatch.setattr(subprocess, "run", _fake(255, stderr="Connection closed"))
    rc = bp.main(["--remote", "box"])
    out = json.loads(capsys.readouterr().out.strip())
    assert rc == 1 and out["ok"] is False and out["outcome_class"] == "box_unreachable"
    assert "fabricate" in out["hint"].lower()


def test_cli_ok_when_reachable(monkeypatch, capsys):
    monkeypatch.setattr(subprocess, "run", _fake(0, stdout="GPU 0: A6000\n"))
    rc = bp.main(["--remote", "box"])
    out = json.loads(capsys.readouterr().out.strip())
    assert rc == 0 and out["ok"] is True and "AC6_RUNBOOK" in out["next"]
