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
    supports_subagent_limits: bool = True
    is_team: bool = False


def _code_normal_prompt(task: Task) -> str:
    """Build the noninteractive prompt for JiuwenSwarm code.normal mode."""

    return f"""You are an AI coding agent. Solve this SWE-bench task.

{task_context(task)}
Work in the provided repository and implement the fix.

{IMPLEMENTATION_CONSTRAINTS}- Do not ask the user questions, request approval, or switch execution mode.
- When you are done, provide the final result and stop.
"""


def _code_team_prompt(task: Task) -> str:
    return _code_normal_prompt(task) + (
        "\nBuild a team and delegate at least one focused task to a teammate. "
        "Teammates may delegate further within the shared task limits. "
        "Coordinate edits in the provided repository, wait for all delegated work, "
        "verify the combined changes, then finish the team round.\n"
    )


CODE_NORMAL_PROFILE = JiuwenSwarmProfile("agent.code.normal", _code_normal_prompt)
CODE_TEAM_PROFILE = JiuwenSwarmProfile("team.code.normal", _code_team_prompt, is_team=True)

_RUNNABLE = {profile.name: profile for profile in (CODE_NORMAL_PROFILE, CODE_TEAM_PROFILE)}
_ALIASES = {"code.normal": CODE_NORMAL_PROFILE.name, "code.team": CODE_TEAM_PROFILE.name}


def get_profile(profile_name: str) -> JiuwenSwarmProfile:
    """Return a runnable JiuwenSwarm profile.

    Raises:
        ValueError: If the profile is unsupported.
    """

    try:
        return _RUNNABLE[_ALIASES.get(profile_name, profile_name)]
    except KeyError:
        raise ValueError(f"Unknown JiuwenSwarm profile: {profile_name!r}") from None
