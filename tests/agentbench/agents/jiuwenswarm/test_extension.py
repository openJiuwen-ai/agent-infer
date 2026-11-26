# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from agentinfer.agentbench.agents.jiuwenswarm.limits import TaskBudget
from tests.helpers.marks import test_marks

pytestmark = test_marks(hardware="cpu", ci="ready")


def _load_extension(monkeypatch, actor, *, compatible_tool=True):
    class OpenAIModelClient:
        def _build_request_params(self, **kwargs):
            return kwargs

    client_module = ModuleType("openjiuwen.core.foundation.llm.model_clients.openai_model_client")
    client_module.OpenAIModelClient = OpenAIModelClient
    hooks = ModuleType("agentinfer.agentbench.agents.jiuwenswarm.hooks")
    hooks.current_actor = lambda: actor
    hooks.install = lambda: None
    hooks.record_extension_ready = lambda: None
    tools = ModuleType("jiuwenswarm.agents.harness.common.tools")
    tool = SimpleNamespace(invoke=lambda _: None)
    if compatible_tool:
        tool._func = lambda **_: None
    tools.command_tools = SimpleNamespace(mcp_exec_command=tool)
    for name, module in {
        client_module.__name__: client_module,
        hooks.__name__: hooks,
        tools.__name__: tools,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).resolve().parents[4] / "agentinfer/agentbench/agents/jiuwenswarm/extension.py"
    spec = importlib.util.spec_from_file_location("agentbench_test_extension", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, tools.command_tools.mcp_exec_command


def test_root_identity_and_extra_body_preserved(monkeypatch):
    budget = TaskBudget("root", 3, 2)
    module, _ = _load_extension(monkeypatch, budget.root)
    params = module.AgentBenchAffinityModelClient()._build_request_params(
        stream=True, extra_body={"other": 1, "vllm_xargs": {"existing": 2}}
    )
    extra = params["extra_body"]
    assert extra["other"] == 1
    assert extra["vllm_xargs"]["existing"] == 2
    identity = json.loads(extra["vllm_xargs"]["agentic_context"])
    assert identity["program_id"] == identity["task_id"] == "root"
    assert identity["agent_role"] == "lead"
    assert "parent_program_id" not in identity
    assert extra["agent_hint"] == {"session_id": "root", "parent_session_id": "root"}
    assert "vllm_xargs" not in params


def test_nested_identity_only_blocks_a_waiting_parent(monkeypatch):
    budget = TaskBudget("root", 3, 2)
    parent = budget.admit("member", budget.root)
    child = budget.admit("child", parent)
    module, _ = _load_extension(monkeypatch, child)
    client = module.AgentBenchAffinityModelClient()

    def identity():
        return json.loads(client._build_request_params()["extra_body"]["vllm_xargs"]["agentic_context"])

    result = identity()
    assert result["parent_program_id"] == "member"
    assert result["task_id"] == "root"
    assert result["expected_resume"] is True
    assert result["blocks_parent"] is False
    parent.waiting = 1
    assert identity()["blocks_parent"] is True
    parent.work = 1
    assert identity()["blocks_parent"] is False


def test_mcp_exec_command_remains_disabled(monkeypatch):
    _, obj = _load_extension(monkeypatch, TaskBudget("root", 1, 1).root)
    result = asyncio.run(obj._func(command="ignored"))
    assert "mcp_exec_command is disabled" in result


def test_extension_rejects_unexpected_command_tool_interface(monkeypatch):
    with pytest.raises(RuntimeError, match="unexpected tool interface"):
        _load_extension(monkeypatch, TaskBudget("root", 1, 1).root, compatible_tool=False)
