# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""JiuwenSwarm runtime integration for AgentBench.

The JiuwenSwarm AgentServer imports this module as an extension. Importing it
registers an OpenJiuwen model client that attaches canonical AgentInfer
identity and streaming usage metadata to LLM requests. It also installs a
narrow permission compatibility patch for noninteractive compound shells.
"""

import os

from openjiuwen.core.foundation.kv_cache import resolve_session_lineage
from openjiuwen.core.foundation.llm.model_clients.ascend_affinity_model_client import (
    AscendAffinityModelClient,
)
from openjiuwen.core.session import get_current_session

from agentinfer.scheduling.identity import AgentIdentity, encode_agent_identity

# Injected by JiuwenInstance to identify the benchmark task's root/lead session.
ROOT_SESSION_ENV = "AGENTBENCH_ROOT_SESSION_ID"

# OpenJiuwen 0.1.16.post1 matched-rule encodings recognized by the
# compound-shell approval compatibility patch below.
_APPROVAL_PREFIX = "tiered_policy:approval_overrides:"
_SHELL_AST_TOO_COMPLEX_PREFIX = "tiered_policy:shell_ast:too_complex:"
_SHELL_SUBCOMMANDS_PREFIX = "tiered_policy:shell_subcommands:"


def _all_shell_subcommands_use_approval_overrides(matched_rule: str | None) -> bool:
    """Recognize OpenJiuwen 0.1.16's aggregated approval-override marker."""

    if not isinstance(matched_rule, str) or not matched_rule.startswith(_SHELL_SUBCOMMANDS_PREFIX):
        return False
    decisions = matched_rule.split("=>")[1:]
    return bool(decisions) and all(decision.startswith(_APPROVAL_PREFIX) for decision in decisions)


def _complex_shell_uses_approval_override(matched_rule: str | None) -> bool:
    """Recognize a complex-shell marker followed by an explicit override."""

    if not isinstance(matched_rule, str) or not matched_rule.startswith(_SHELL_AST_TOO_COMPLEX_PREFIX):
        return False
    parts = matched_rule.split("|")
    return len(parts) == 2 and parts[1].startswith(_APPROVAL_PREFIX)


def _patch_compound_shell_approval_detection() -> None:
    """Work around OpenJiuwen 0.1.16.post1 compound-shell approval handling.

    That version can re-escalate a compound command to ASK after AgentBench's
    explicit approval override has matched. Noninteractive benchmark runs
    cannot answer ASK, so extend the original check only for the two known
    aggregate marker formats above. Remove this patch after upgrading to an
    OpenJiuwen release containing the upstream fix.
    """

    try:
        from openjiuwen.harness.security import core as security_core
    except ImportError:
        return
    original = security_core.matched_rule_uses_approval_override
    if getattr(original, "_agentbench_compound_shell", False):
        return

    def matched_rule_uses_approval_override(matched_rule: str | None) -> bool:
        return (
            original(matched_rule)
            or _all_shell_subcommands_use_approval_overrides(matched_rule)
            or _complex_shell_uses_approval_override(matched_rule)
        )

    matched_rule_uses_approval_override._agentbench_compound_shell = True  # type: ignore[attr-defined]
    security_core.matched_rule_uses_approval_override = matched_rule_uses_approval_override


_patch_compound_shell_approval_detection()


# mcp_exec_command bypasses SysOperation and shells out directly on the host.
# This extension is loaded only by AgentBench's Jiuwen runtime, where isolation
# is mandatory, so the command must always be sealed.
def _patch_disable_mcp_exec_command() -> None:
    """Neutralize mcp_exec_command under AgentBench isolation.

    mcp_exec_command bypasses SysOperation and shells out via raw
    subprocess.Popen, so it is the one Jiuwen tool the jiuwenbox sandbox policy
    cannot confine. Under isolation, replace its LocalFunction._func so every
    caller (main agent and sub-agent, which share the same object reference
    returned by get_mcp_tools) hits the disable stub instead. Fail closed if the
    upstream structure no longer exposes _func, so a silent Jiuwen upgrade that
    removes the attribute surfaces loudly at Gateway startup rather than
    silently re-enabling an unconfined shell. The pinned version is embedded in
    the disable message for triage.
    """

    try:
        from jiuwenswarm.agents.harness.common.tools import command_tools
    except ImportError as exc:
        raise RuntimeError("cannot import mcp_exec_command for sealing under AgentBench isolation") from exc
    obj = getattr(command_tools, "mcp_exec_command", None)
    if obj is None or not hasattr(obj, "_func") or not callable(getattr(obj, "invoke", None)):
        raise RuntimeError(
            "mcp_exec_command structure changed (_func/invoke unavailable); refusing to "
            "start AgentBench without sealing the mcp_exec_command escape path"
        )
    if getattr(obj._func, "_agentbench_disabled", False):
        return
    try:
        from importlib.metadata import version

        pinned_version = version("jiuwenswarm")
    except Exception:
        pinned_version = "unknown"

    async def disabled(**_kwargs: object) -> str:
        return (
            "[ERROR]: mcp_exec_command is disabled under AgentBench isolation "
            f"(patched against jiuwenswarm {pinned_version})."
        )

    disabled._agentbench_disabled = True  # type: ignore[attr-defined]
    disabled._agentbench_pinned_version = pinned_version  # type: ignore[attr-defined]
    obj._func = disabled


_patch_disable_mcp_exec_command()


class AgentBenchAffinityModelClient(AscendAffinityModelClient):
    """Attach task-root and current-agent session identity to each request."""

    __client_name__ = "AgentBenchAffinity"

    def _build_ascend_affinity_request_params(self, **kwargs: object) -> dict[str, object]:
        """Build OpenJiuwen request parameters with AgentBench identity and usage metadata."""

        if not kwargs.get("session_id"):
            root_session_id = os.environ.get(ROOT_SESSION_ENV, "").strip()
            session_id, parent_session_id = resolve_session_lineage(get_current_session())
            if not root_session_id:
                self._raise_config_error(f"{ROOT_SESSION_ENV} is required")
            if not session_id:
                self._raise_config_error("current OpenJiuwen session is required for AgentBench identity")
            if session_id == root_session_id:
                parent_session_id = root_session_id
            elif not parent_session_id or parent_session_id == session_id:
                parent_session_id = root_session_id
            kwargs["session_id"] = session_id
            kwargs["parent_session_id"] = parent_session_id
        raw_params = super()._build_ascend_affinity_request_params(**kwargs)
        if not isinstance(raw_params, dict):
            self._raise_config_error("OpenJiuwen request params must be an object")
        params: dict[str, object] = raw_params
        self._inject_agentinfer_identity(params)
        if params.get("stream") is True:
            stream_options = params.get("stream_options")
            if isinstance(stream_options, dict):
                stream_options.setdefault("include_usage", True)
            elif stream_options is None:
                params["stream_options"] = {"include_usage": True}
        return params

    def _inject_agentinfer_identity(self, params: dict[str, object]) -> None:
        """Attach canonical AgentInfer identity derived from Jiuwen session lineage.

        OpenJiuwen's agent_hint contains only the current and parent session IDs.
        Combine it with the AgentBench root session to preserve task identity and
        root/subagent relationships for scheduling.
        """

        root_session_id = os.environ.get(ROOT_SESSION_ENV, "").strip()
        hint = params.get("agent_hint")
        if not root_session_id:
            self._raise_config_error(f"{ROOT_SESSION_ENV} is required")
        if not isinstance(hint, dict):
            self._raise_config_error("OpenJiuwen agent_hint is required for AgentBench identity")
        session_id = hint.get("session_id")
        parent_session_id = hint.get("parent_session_id")
        if not isinstance(session_id, str) or not session_id.strip():
            self._raise_config_error("OpenJiuwen agent_hint.session_id is required")
        session_id = session_id.strip()
        is_root = session_id == root_session_id
        if not is_root:
            if not isinstance(parent_session_id, str) or not parent_session_id.strip():
                parent_session_id = root_session_id
            else:
                parent_session_id = parent_session_id.strip()
            if parent_session_id == session_id:
                parent_session_id = root_session_id

        identity = AgentIdentity(
            program_id=session_id,
            task_id=root_session_id,
            session_id=session_id,
            agent_id="lead" if is_root else session_id,
            parent_program_id=None if is_root else parent_session_id,
            blocks_parent=not is_root,
            expected_resume=is_root,
            agent_role="lead" if is_root else "subagent",
        )
        raw_xargs = params.get("vllm_xargs")
        if raw_xargs is None:
            xargs: dict[str, object] = {}
        elif isinstance(raw_xargs, dict):
            xargs = dict(raw_xargs)
        else:
            self._raise_config_error("vllm_xargs must be an object")
        xargs["agentic_context"] = encode_agent_identity(identity)
        params["vllm_xargs"] = xargs
