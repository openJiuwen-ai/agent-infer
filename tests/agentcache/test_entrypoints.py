import sys
import types

import pytest

from agentinfer.agentcache.entrypoints.cli import main as cli_main


def _install_vllm(monkeypatch: pytest.MonkeyPatch, entrypoint: object) -> None:
    module = types.ModuleType("vllm.entrypoints.cli.main")
    module.main = entrypoint
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.entrypoints", types.ModuleType("vllm.entrypoints"))
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.cli", types.ModuleType("vllm.entrypoints.cli"))
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.cli.main", module)


def test_vllm_delegation_restores_sys_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    original = ["pytest"]
    seen: list[list[str]] = []
    monkeypatch.setattr(sys, "argv", original)
    _install_vllm(monkeypatch, lambda: seen.append(list(sys.argv)))

    assert cli_main.main(["vllm", "serve", "model"]) == 0
    assert seen == [["vllm", "serve", "model"]]
    assert sys.argv is original


def test_vllm_delegation_propagates_return_code(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_vllm(monkeypatch, lambda: 2)

    assert cli_main.main(["vllm", "serve", "model"]) == 2


def test_vllm_delegation_restores_sys_argv_after_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    original = ["pytest"]
    monkeypatch.setattr(sys, "argv", original)

    def fail() -> None:
        raise RuntimeError("failed")

    _install_vllm(monkeypatch, fail)

    with pytest.raises(RuntimeError, match="failed"):
        cli_main.main(["vllm", "serve", "model"])
    assert sys.argv is original
