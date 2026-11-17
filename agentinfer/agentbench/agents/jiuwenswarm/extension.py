# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""WorkSwarm 0.2.8 model identity and task execution integration."""

from importlib.metadata import version

from jiuwenswarm.agents.harness.common.tools import command_tools
from openjiuwen.core.foundation.llm.model_clients.openai_model_client import OpenAIModelClient

from agentinfer.agentbench.agents.jiuwenswarm.hooks import current_actor, install
from agentinfer.scheduling.identity import AgentIdentity, encode_agent_identity

install()


async def _disabled_mcp_exec_command(**_kwargs: object) -> str:
    return f"[ERROR]: mcp_exec_command is disabled under AgentBench isolation (workswarm {version('workswarm')})."


# Without a sandbox execution binding this tool can run on the host.
command_tools.mcp_exec_command._func = _disabled_mcp_exec_command


class AgentBenchAffinityModelClient(OpenAIModelClient):
    """Use the native OpenAI transport with canonical task/agent identity."""

    __client_name__ = "AgentBenchAffinity"

    def _build_request_params(self, **kwargs):
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
