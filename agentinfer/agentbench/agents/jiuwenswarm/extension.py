# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""WorkSwarm 0.2.8 model identity and task execution integration."""

from typing import Any

from openjiuwen.core.foundation.llm.model_clients.openai_model_client import OpenAIModelClient

from agentinfer.agentbench.agents.jiuwenswarm.hooks import current_actor, install, record_extension_ready
from agentinfer.scheduling.identity import AgentIdentity, encode_agent_identity


async def _disabled_mcp_exec_command(**_kwargs: object) -> str:
    return "[ERROR]: mcp_exec_command is disabled under AgentBench isolation."


def _seal_command_tool() -> None:
    try:
        from jiuwenswarm.agents.harness.common.tools import command_tools
    except ImportError as exc:
        raise RuntimeError("Cannot load mcp_exec_command under AgentBench isolation") from exc

    tool = getattr(command_tools, "mcp_exec_command", None)
    if tool is None or not callable(getattr(tool, "_func", None)) or not callable(getattr(tool, "invoke", None)):
        raise RuntimeError("Cannot disable mcp_exec_command under AgentBench isolation: unexpected tool interface")
    tool._func = _disabled_mcp_exec_command


class AgentBenchAffinityModelClient(OpenAIModelClient):
    """Use the native OpenAI transport with canonical task/agent identity."""

    __client_name__ = "AgentBenchAffinity"

    def _build_request_params(self, **kwargs: Any) -> dict[str, Any]:
        params = super()._build_request_params(**kwargs)
        actor = current_actor()
        root = actor.budget.root
        parent = actor.parent
        identity = AgentIdentity(
            program_id=actor.program_id,
            task_id=root.program_id,
            session_id=actor.program_id,
            agent_id="lead" if actor is root else actor.program_id,
            parent_program_id=parent.program_id if parent else None,
            blocks_parent=bool(parent and parent.waiting and not parent.work),
            expected_resume=actor.persistent,
            agent_role="lead" if actor is root else "subagent",
        )
        extra_body = dict(params.get("extra_body") or {})
        xargs = dict(extra_body.get("vllm_xargs") or {})
        xargs["agentic_context"] = encode_agent_identity(identity)
        extra_body["vllm_xargs"] = xargs
        extra_body["agent_hint"] = {
            "session_id": actor.program_id,
            "parent_session_id": parent.program_id if parent else actor.program_id,
        }
        params["extra_body"] = extra_body
        return params


_seal_command_tool()
install()
record_extension_ready()
