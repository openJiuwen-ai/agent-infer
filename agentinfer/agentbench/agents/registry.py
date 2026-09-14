# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Explicit agent-runtime registry.

One module owns the mapping from ``agent_type`` to a concrete runtime. An
explicit dict (not ``register()``-at-import) keeps registration free of
import-time side effects; adding a runtime means one entry here plus one new
runtime module.
"""

from .claude.runtime import ClaudeRuntime
from .dsh.runtime import DshRuntime
from .jiuwenswarm.runtime import JiuwenSwarmRuntime
from .runtime import AgentRuntime

RUNTIMES: dict[str, AgentRuntime] = {
    "claude": ClaudeRuntime(),
    "jiuwenswarm": JiuwenSwarmRuntime(),
    "dsh": DshRuntime(),
}


def get_runtime(agent_type: str) -> AgentRuntime:
    """Return the runtime registered under ``agent_type``.

    Raises ``ValueError`` for unknown types so every dispatch site fails
    uniformly instead of silently skipping validation.
    """

    try:
        return RUNTIMES[agent_type]
    except KeyError:
        raise ValueError(f"Unsupported agent.type: {agent_type!r}") from None
