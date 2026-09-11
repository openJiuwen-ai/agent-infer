# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Define supported JiuwenSwarm SWE-bench profiles and user prompts."""

from collections.abc import Callable
from dataclasses import dataclass

from ...benchkit.dataset import IMPLEMENTATION_CONSTRAINTS, Task, task_context

REQUIRED_ENDPOINT = "/v1/chat/completions"


@dataclass(frozen=True)
class JiuwenSwarmProfile:
    """Bundle one runnable JiuwenSwarm mode with its benchmark prompt."""

    name: str
    build_prompt: Callable[[Task], str]


def _code_normal_prompt(task: Task) -> str:
    """Build the noninteractive prompt for JiuwenSwarm code.normal mode."""

    return f"""You are an AI coding agent. Solve this SWE-bench task.

{task_context(task)}
Work in the provided repository and implement the fix.

{IMPLEMENTATION_CONSTRAINTS}- Do not ask the user questions, request approval, or switch execution mode.
- When you are done, provide the final result and stop.
"""


CODE_NORMAL_PROFILE = JiuwenSwarmProfile("code.normal", _code_normal_prompt)

_RUNNABLE = {CODE_NORMAL_PROFILE.name: CODE_NORMAL_PROFILE}
_PLANNED = frozenset({"code.team"})


def get_profile(profile_name: str) -> JiuwenSwarmProfile:
    """Return a runnable JiuwenSwarm profile.

    Raises:
        ValueError: If the profile is unknown or planned but not yet implemented.
    """

    if profile_name in _PLANNED:
        raise ValueError(f"JiuwenSwarm profile is designed but not implemented: {profile_name!r}")
    try:
        return _RUNNABLE[profile_name]
    except KeyError:
        raise ValueError(f"Unknown JiuwenSwarm profile: {profile_name!r}") from None
