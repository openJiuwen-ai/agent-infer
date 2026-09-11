# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

import asyncio
import importlib.util
import json
import sys
from importlib import metadata
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _extension_modules() -> dict[str, ModuleType]:
    class AscendAffinityModelClient:
        def _build_ascend_affinity_request_params(self, **kwargs):
            return {
                "agent_hint": {
                    "session_id": kwargs["session_id"],
                    "parent_session_id": kwargs["parent_session_id"],
                },
                "stream": True,
            }

        @staticmethod
        def _raise_config_error(message):
            raise ValueError(message)

    modules = {
        "openjiuwen": ModuleType("openjiuwen"),
        "openjiuwen.core": ModuleType("openjiuwen.core"),
        "openjiuwen.core.foundation": ModuleType("openjiuwen.core.foundation"),
        "openjiuwen.core.foundation.kv_cache": ModuleType("openjiuwen.core.foundation.kv_cache"),
        "openjiuwen.core.foundation.llm": ModuleType("openjiuwen.core.foundation.llm"),
        "openjiuwen.core.foundation.llm.model_clients": ModuleType("openjiuwen.core.foundation.llm.model_clients"),
        "openjiuwen.core.foundation.llm.model_clients.ascend_affinity_model_client": ModuleType(
            "openjiuwen.core.foundation.llm.model_clients.ascend_affinity_model_client"
        ),
        "openjiuwen.core.session": ModuleType("openjiuwen.core.session"),
    }
    modules["openjiuwen.core.foundation.kv_cache"].resolve_session_lineage = lambda _session: (None, None)
    modules[
        "openjiuwen.core.foundation.llm.model_clients.ascend_affinity_model_client"
    ].AscendAffinityModelClient = AscendAffinityModelClient
    modules["openjiuwen.core.session"].get_current_session = lambda: None
    return modules


_DEFAULT_COMMAND_TOOLS = object()


def _load_extension(monkeypatch, *, command_tools=_DEFAULT_COMMAND_TOOLS):
    if command_tools is _DEFAULT_COMMAND_TOOLS:
        command_tools = _command_tools(_FakeLocalFunction())
    modules = _extension_modules()
    if command_tools is not None:
        modules.update(
            {
                "jiuwenswarm": ModuleType("jiuwenswarm"),
                "jiuwenswarm.agents": ModuleType("jiuwenswarm.agents"),
                "jiuwenswarm.agents.harness": ModuleType("jiuwenswarm.agents.harness"),
                "jiuwenswarm.agents.harness.common": ModuleType("jiuwenswarm.agents.harness.common"),
                "jiuwenswarm.agents.harness.common.tools": ModuleType("jiuwenswarm.agents.harness.common.tools"),
                "jiuwenswarm.agents.harness.common.tools.command_tools": command_tools,
            }
        )
        modules["jiuwenswarm.agents.harness.common.tools"].command_tools = command_tools
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    path = Path(__file__).resolve().parents[4] / "agentinfer" / "agentbench" / "agents" / "jiuwenswarm" / "extension.py"
    spec = importlib.util.spec_from_file_location("test_agentbench_jiuwenswarm_extension", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeLocalFunction:
    def __init__(self) -> None:
        async def native(**_kwargs):
            return "native"

        self._func = native

    async def invoke(self, inputs):
        return await self._func(**inputs)


def _command_tools(fake: object) -> ModuleType:
    module = ModuleType("jiuwenswarm.agents.harness.common.tools.command_tools")
    module.mcp_exec_command = fake
    return module


def test_root_identity_uses_canonical_program_without_self_parent(monkeypatch) -> None:
    module = _load_extension(monkeypatch)
    monkeypatch.setenv(module.ROOT_SESSION_ENV, "root-session")

    params = module.AgentBenchAffinityModelClient()._build_ascend_affinity_request_params(
        session_id="root-session",
        parent_session_id="root-session",
    )

    context = json.loads(params["vllm_xargs"]["agentic_context"])
    assert context == {
        "program_id": "root-session",
        "task_id": "root-session",
        "session_id": "root-session",
        "agent_id": "lead",
        "blocks_parent": False,
        "expected_resume": True,
        "agent_role": "lead",
    }
    assert params["agent_hint"]["parent_session_id"] == "root-session"
    assert params["stream_options"] == {"include_usage": True}


def test_child_identity_keeps_direct_root_parent(monkeypatch) -> None:
    module = _load_extension(monkeypatch)
    monkeypatch.setenv(module.ROOT_SESSION_ENV, "root-session")

    params = module.AgentBenchAffinityModelClient()._build_ascend_affinity_request_params(
        session_id="child-session",
        parent_session_id="root-session",
    )

    context = json.loads(params["vllm_xargs"]["agentic_context"])
    assert context == {
        "program_id": "child-session",
        "task_id": "root-session",
        "session_id": "child-session",
        "agent_id": "child-session",
        "parent_program_id": "root-session",
        "blocks_parent": True,
        "expected_resume": False,
        "agent_role": "subagent",
    }


def test_compound_shell_override_marker_requires_every_subcommand(monkeypatch) -> None:
    module = _load_extension(monkeypatch)
    approved = (
        "tiered_policy:shell_subcommands:cd /workspace=>"
        "tiered_policy:approval_overrides:approval_overrides[agentbench]+"
        "pytest -q=>tiered_policy:approval_overrides:approval_overrides[agentbench]"
    )
    mixed = (
        "tiered_policy:shell_subcommands:cd /workspace=>tiered_policy:tools.bash+"
        "pytest -q=>tiered_policy:approval_overrides:approval_overrides[agentbench]"
    )

    assert module._all_shell_subcommands_use_approval_overrides(approved) is True
    assert module._all_shell_subcommands_use_approval_overrides(mixed) is False


def test_complex_shell_override_marker_requires_explicit_override(monkeypatch) -> None:
    module = _load_extension(monkeypatch)
    approved = (
        "tiered_policy:shell_ast:too_complex:tree-sitter detected unsupported complex shell structure|"
        "tiered_policy:approval_overrides:approval_overrides[agentbench_noninteractive_shell]"
    )
    unapproved = "tiered_policy:shell_ast:too_complex:tree-sitter detected unsupported complex shell structure"
    mixed = approved + "|tiered_policy:tools.bash"

    assert module._complex_shell_uses_approval_override(approved) is True
    assert module._complex_shell_uses_approval_override(unapproved) is False
    assert module._complex_shell_uses_approval_override(mixed) is False


def test_mcp_exec_command_is_always_disabled(monkeypatch) -> None:
    obj = _FakeLocalFunction()
    original = obj._func
    monkeypatch.setattr(metadata, "version", lambda _name: "test-ver")

    module = _load_extension(
        monkeypatch,
        command_tools=_command_tools(obj),
    )

    result = asyncio.run(obj.invoke({"command": "printf ESCAPED"}))
    assert "disabled" in result
    assert "test-ver" in result
    assert obj._func is not original
    assert obj._func._agentbench_disabled is True
    assert obj._func._agentbench_pinned_version == "test-ver"
    module._patch_disable_mcp_exec_command()
    assert asyncio.run(obj.invoke({"command": "printf ESCAPED"})) == result


def test_mcp_exec_command_fail_closed_on_missing_func(monkeypatch) -> None:
    with pytest.raises(RuntimeError, match="mcp_exec_command structure changed"):
        _load_extension(
            monkeypatch,
            command_tools=_command_tools(SimpleNamespace()),
        )


def test_mcp_exec_command_fail_closed_when_command_tools_import_fails(monkeypatch) -> None:
    with pytest.raises(RuntimeError, match="cannot import mcp_exec_command"):
        _load_extension(monkeypatch, command_tools=None)
