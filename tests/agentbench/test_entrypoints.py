import subprocess
import sys
import types
from pathlib import Path

import pytest

from agentinfer.agentcache.entrypoints.bench import _normalize_delegated_argv
from agentinfer.agentcache.entrypoints.cli import main as cli_main


def test_only_explicit_agentinfer_benchmark_is_intercepted() -> None:
    assert cli_main._is_bench_delegation(["vllm", "bench", "serve", "--model", "m"]) is False
    assert cli_main._is_bench_delegation(["vllm", "bench", "serve", "--agentinfer", "--config", "c"]) is True
    assert cli_main._is_bench_delegation(["/usr/bin/vllm", "bench", "serve", "--agentinfer"]) is True
    assert cli_main._is_bench_delegation(["vllm", "bench", "serve", "--agentcache", "--config", "c"]) is False


def test_delegated_argv_normalizes_implicit_and_explicit_commands() -> None:
    implicit, metadata = _normalize_delegated_argv(["vllm", "bench", "serve", "--agentinfer", "--config", "c"])
    explicit, _ = _normalize_delegated_argv(["vllm", "bench", "serve", "--agentinfer", "compare"])
    assert implicit == ["run", "--config", "c"]
    assert explicit == ["compare"]
    assert metadata["entrypoint"] == "vllm bench serve --agentinfer"


def test_vllm_delegation_restores_sys_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    original = ["pytest"]
    seen = []
    monkeypatch.setattr(sys, "argv", original)
    fake = types.ModuleType("vllm.entrypoints.cli.main")
    fake.main = lambda: seen.append(list(sys.argv))
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.entrypoints", types.ModuleType("vllm.entrypoints"))
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.cli", types.ModuleType("vllm.entrypoints.cli"))
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.cli.main", fake)

    assert cli_main.main(["vllm", "serve", "model"]) == 0
    assert seen == [["vllm", "serve", "model"]]
    assert sys.argv is original


def test_vllm_delegation_restores_sys_argv_after_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    original = ["pytest"]
    monkeypatch.setattr(sys, "argv", original)
    fake = types.ModuleType("vllm.entrypoints.cli.main")

    def fail():
        raise RuntimeError("failed")

    fake.main = fail
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.cli.main", fake)
    with pytest.raises(RuntimeError, match="failed"):
        cli_main.main(["vllm", "serve", "model"])
    assert sys.argv is original


def test_absent_vllm_import_and_benchmark_help_in_isolated_interpreter() -> None:
    repo = str(Path.cwd())
    script = f"""
import builtins
import sys
sys.path.insert(0, {repo!r})
real_import = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == 'vllm' or name.startswith('vllm.'):
        raise ModuleNotFoundError("No module named 'vllm'", name='vllm')
    return real_import(name, *args, **kwargs)
builtins.__import__ = blocked
import agentinfer.agentbench
from agentinfer.agentbench.benchkit.cli import main
try:
    main(['--help'])
except SystemExit as exc:
    assert exc.code == 0
"""
    result = subprocess.run([sys.executable, "-I", "-c", script], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
